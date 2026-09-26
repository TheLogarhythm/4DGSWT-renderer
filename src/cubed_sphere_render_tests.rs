//! Offscreen acceptance of the real GS shader, including streamed instance IDs.
use crate::{
    camera::Camera,
    renderer::GSWTRenderer,
    scene::Scene,
    structure::*,
    test_support::gpu::{GpuTestContext, MissingGpu, read_texture_bytes},
    texture::Texture,
    utils::*,
};

#[test]
fn cubed_sphere_renders_all_faces_in_both_draw_paths_and_reuses_uploads() {
    let GpuTestContext { device, queue, .. } = GpuTestContext::new(
        "sphere render acceptance",
        MissingGpu::Fail,
        wgpu::PowerPreference::None,
        crate::profiler::renderer_required_features,
    )
    .unwrap();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: 64,
        height: 64,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Sphere acceptance"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut scene = Scene::new();
    scene.splat_count = 1;
    scene.tex_width = 2048;
    scene.tex_height = 1;
    scene.tex_data = vec![0; 2048 * 4];
    scene.tex_data[..3].copy_from_slice(&[2_f32.to_bits(), 2_f32.to_bits(), 0.2_f32.to_bits()]);
    scene.tex_data[4] = half::f16::from_f32(0.08).to_bits() as u32;
    scene.tex_data[5] = (half::f16::from_f32(0.12).to_bits() as u32) << 16;
    scene.tex_data[6] = (half::f16::from_f32(0.06).to_bits() as u32) << 16;
    scene.tex_data[7] = 0xff44bbee;
    let base = TileBaseData {
        splat_count: 1,
        tile_center: vec3(2., 2., 0.2),
        aabb: (vec3(1., 1., 0.), vec3(3., 3., 1.)),
        raw_depth: vec![0],
        gs_index: vec![0],
        gs_lod_id: vec![0],
    };
    let mut bases = vec![vec![vec![base]]];
    let mut gs = GSWTRenderer::new(
        &device,
        &queue,
        &config,
        PreloadData {
            tile_splats_merged: scene,
            tile_base_data: &mut bases,
            merged_motion: None,
        },
    )
    .unwrap();
    let mut user = UserData::new();
    user.surface_type = SurfaceType::Sphere;
    user.tile_map_half_wh = vec2(7, 5);
    user.n_tiles = (3, 1, 1);
    user.lod_transition_dist = vec![4.0, 6.0, 100.0];
    user.height_map_wh = vec2(1, 1);
    user.height_map = vec![0.];
    let mut data = RenderData::new(3);
    data.profiler_enabled = true;
    data.render_config.scene_scale = vec3(2., 0.5, 1.3); // old planar values cannot distort the sphere
    data.render_config.lod_enable[1] = false;
    data.depth_texture = Some(Texture::create_depth_texture(
        &device,
        &config,
        "Sphere depth",
    ));
    data.cur_scene_data = Some(SceneData::new());
    for n in [1, 3] {
        user.sphere_tiles_per_face = n;
        user.tile_map_wh = vec2(6 * n, n);
        gs.configure(&device, &queue, &user, &data);
        for face in 0..6 {
            let coord = vec2(face * n + n / 2, n / 2);
            let mut tile = TileInstance::new();
            tile.map_coord = coord;
            tile.map_index = coord.x * n + coord.y;
            tile.tile_offset = vec3(
                (coord.x as i32 - 7) as f32 * 4.,
                (coord.y as i32 - 5) as f32 * 4.,
                0.,
            );
            let (normal, _, up) = crate::cubed_sphere::basis(face);
            let camera = Camera::new_perspective(
                winit::dpi::PhysicalSize::new(64, 64),
                normal * 30.,
                normal * 20.,
                up,
                degrees(60.),
                0.1,
                100.,
            );
            let mut first = None;
            for draw_mode in 0..4 {
                let streamed = draw_mode != 0;
                let transitioning = draw_mode >= 2;
                let mut drawn_tile = tile.clone();
                if transitioning {
                    drawn_tile.tid.0 = usize::from(draw_mode == 3);
                    drawn_tile.transition_status = TileTransitionStatus::Changing(draw_mode == 2);
                    data.render_config.lod_enable[1] = true;
                } else {
                    data.render_config.lod_enable[1] = false;
                }
                let mut key = RenderDataKey::new();
                key.tid = vec![(0, 0)];
                let value = streamed.then(|| RenderDataValue {
                    splat_count: 1,
                    gs_index: vec![0],
                    gs_map_id: vec![tile.map_index as u32],
                    merge_from_vec: vec![tile.map_index],
                    single_lod_id: if transitioning { -1 } else { 0 },
                    gs_lod_id: transitioning.then(|| vec![1]),
                });
                let mut instances = vec![];
                let mut draws = vec![];
                // A high sort index needs only one compact visible uniform slot.
                // These dummy rows are LoD-disabled.
                if n == 1 && face == 0 && !streamed {
                    let mut disabled = tile.clone();
                    disabled.tid.0 = 1;
                    instances = vec![disabled; 6 * 128 * 128 - 1];
                    draws = vec![(RenderDataKey::new(), None); instances.len()];
                }
                instances.push(drawn_tile);
                draws.push((key, value));
                data.cur_sort_data = Some(SortData {
                    scene_id: 0,
                    authored: crate::motion_tagging::AuthoredSortMetadata::untagged(
                        instances.len(),
                    ),
                    tile_instance_vec: instances,
                    render_data_vec: draws,
                });
                data.sort_revision += 1;
                let view = target.create_view(&Default::default());
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }
                let first_work = gs.render(
                    &device,
                    &queue,
                    &mut encoder,
                    &view,
                    &camera,
                    &data,
                    false,
                    None,
                    None,
                );
                queue.submit(Some(encoder.finish()));
                let frame = read_texture_bytes(&device, &queue, &target, [64, 64]);
                if streamed {
                    let mut repeat = device.create_command_encoder(&Default::default());
                    {
                        let _clear = repeat.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    }
                    let work = gs.render(
                        &device,
                        &queue,
                        &mut repeat,
                        &view,
                        &camera,
                        &data,
                        false,
                        None,
                        None,
                    );
                    queue.submit(Some(repeat.finish()));
                    assert!(
                        work.uploaded_bytes < first_work.uploaded_bytes,
                        "unchanged streamed draws must reuse GPU uploads"
                    );
                    let repeated_frame = read_texture_bytes(&device, &queue, &target, [64, 64]);
                    assert!(
                        frame == repeated_frame,
                        "resident draws must preserve pixels"
                    );
                }
                let center = &frame[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 3];
                assert!(
                    center.iter().map(|&v| v as u32).sum::<u32>() > 150,
                    "N={n}, face={face}, draw_mode={draw_mode}: {center:?}"
                );
                if let Some(reference) = first.as_ref() {
                    assert_eq!(
                        &frame, reference,
                        "draw paths disagree for N={n}, face={face}"
                    );
                } else {
                    first = Some(frame);
                }
            }
        }
    }
}
