//! Offscreen water + GS rendering; fixtures and assertions live elsewhere.
use super::fixtures::{SceneFixture, two_splats};
use crate::{
    camera::Camera,
    renderer::GSWTRenderer,
    structure::*,
    test_support::gpu::{GpuTestContext, MissingGpu, RgbaFrame, read_texture_bytes},
    texture::Texture,
    utils::*,
    water::WaterRenderer,
};
const SIZE: u32 = 128;

pub(super) struct Harness {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    target: wgpu::Texture,
    pub gs: GSWTRenderer,
    pub water: WaterRenderer,
    pub user: UserData,
    pub data: RenderData,
    pub camera: Camera,
    pub config: wgpu::SurfaceConfiguration,
    pub proxy: Option<crate::proxy::Proxy>,
    pub motion_time: Option<f32>,
    pub skybox: crate::skybox::Skybox,
    // Reference path for verifying that removing the overwritten background preserves pixels.
    pub legacy_background: bool,
}
impl Harness {
    pub fn new() -> Self {
        Self::with_motion(false)
    }
    pub fn with_motion(dynamic: bool) -> Self {
        Self::from_scene(two_splats(dynamic), [SIZE, SIZE])
    }
    pub fn from_scene(fixture: SceneFixture, dimensions: [u32; 2]) -> Self {
        let GpuTestContext {
            device,
            queue,
            adapter,
        } = GpuTestContext::new(
            "water acceptance",
            MissingGpu::Fail,
            wgpu::PowerPreference::None,
            crate::profiler::renderer_required_features,
        )
        .unwrap();
        if std::env::var_os("WATER_PERF_DIR").is_some() {
            eprintln!("GPU adapter: {:?}", adapter.get_info());
        }
        let dynamic = fixture.motion.is_some();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: dimensions[0],
            height: dimensions[1],
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water acceptance target"),
            size: wgpu::Extent3d {
                width: dimensions[0],
                height: dimensions[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut base = vec![vec![vec![fixture.base]]];
        let mut gs = GSWTRenderer::new(
            &device,
            &queue,
            &config,
            PreloadData {
                tile_splats_merged: fixture.scene,
                tile_base_data: &mut base,
                merged_motion: fixture.motion,
            },
        )
        .unwrap();
        let water = WaterRenderer::new(&device, config.format);
        let skybox = crate::skybox::Skybox::new(&device, &config);
        let mut user = UserData::new();
        user.surface_type = SurfaceType::None;
        user.tile_map_half_wh = vec2(1, 1);
        user.tile_map_wh = vec2(3, 3);
        user.n_tiles = (1, 1, 1);
        user.height_map_wh = vec2(1, 1);
        user.height_map = vec![0.0];
        let mut data = RenderData::new(1);
        data.profiler_enabled = false;
        // Existing stage-1 cases explicitly exercise the zero-amplitude path.
        data.render_config.water.amplitude = 0.0;
        data.render_config.water.ripple_strength = 0.0;
        data.render_config.water.varied_waves = false;
        data.depth_texture = Some(Texture::create_depth_texture(
            &device,
            &config,
            "water acceptance depth",
        ));
        data.cur_scene_data = Some(SceneData::new());
        data.cur_sort_data = Some(SortData {
            scene_id: 0,
            tile_instance_vec: vec![TileInstance::new()],
            render_data_vec: vec![(RenderDataKey::new(), None)],
            authored: crate::motion_tagging::AuthoredSortMetadata::untagged(1),
        });
        gs.configure(&device, &queue, &user, &data);
        let camera = Camera::new_perspective(
            winit::dpi::PhysicalSize::new(dimensions[0], dimensions[1]),
            vec3(0.0, 0.0, 5.0),
            Vec3::zero(),
            vec3(0.0, 1.0, 0.0),
            degrees(60.0),
            0.1,
            100.0,
        );
        Self {
            device,
            queue,
            skybox,
            legacy_background: false,
            target,
            gs,
            water,
            user,
            data,
            camera,
            config,
            proxy: None,
            motion_time: dynamic.then_some(0.0),
        }
    }
    fn encode_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queries: Option<&wgpu::QuerySet>,
    ) -> bool {
        let view = self.target.create_view(&Default::default());
        let underwater = if self
            .data
            .render_config
            .water
            .is_active(self.user.surface_type)
        {
            let position = self.camera.position();
            self.data.render_config.water.underwater_frame(
                [position.x, position.y, position.z],
                crate::water::bounds(&self.user, &self.data),
            )
        } else {
            None
        };
        if let Some(query_set) = queries {
            let _start = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water test frame start"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                }),
            });
        }
        if let Some(time) = self.motion_time {
            self.gs
                .update_motion(
                    &self.queue,
                    encoder,
                    crate::motion::TimelineSample::Source { u: time },
                    true,
                    7,
                    Default::default(),
                    None,
                )
                .unwrap();
        }
        let water_drawn = self.water.prepare(
            &self.device,
            &self.queue,
            encoder,
            &self.camera,
            &self.user,
            &self.data,
            self.skybox
                .water_environment
                .as_ref()
                .filter(|_| self.data.use_skybox),
            crate::profiler::WaterPassTimestamps {
                intersection: queries.map(|query_set| wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(1),
                    end_of_pass_write_index: Some(2),
                }),
                shading: queries.map(|query_set| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(3),
                    end_of_pass_write_index: Some(4),
                }),
            },
        );
        if self.legacy_background || !(water_drawn && underwater.is_some()) {
            if self.data.use_skybox {
                self.skybox
                    .render(&self.queue, encoder, &view, &self.camera);
            } else {
                let background =
                    underwater.map_or([0.0; 3], |frame| frame.color.map(|c| c * frame.strength));
                {
                    let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("water test background"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: background[0] as f64,
                                    g: background[1] as f64,
                                    b: background[2] as f64,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                }
            }
        }
        if water_drawn && underwater.is_some() {
            self.water.render_background(
                encoder,
                &view,
                self.skybox
                    .water_environment
                    .as_ref()
                    .filter(|_| self.data.use_skybox),
            );
        }
        if let Some(proxy) = self.proxy.as_mut() {
            proxy.render_with_water(
                &self.queue,
                encoder,
                &view,
                &self.camera,
                &self.data,
                self.water.hits().filter(|_| water_drawn).map(|h| &h.read),
            );
        }
        if water_drawn {
            self.water.render_surface(
                encoder,
                &view,
                &self.data,
                self.skybox
                    .water_environment
                    .as_ref()
                    .filter(|_| self.data.use_skybox),
                crate::profiler::WaterPassTimestamps {
                    intersection: None,
                    shading: queries.map(|query_set| wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(3),
                        end_of_pass_write_index: Some(4),
                    }),
                },
            );
        }
        self.gs.render(
            &self.queue,
            encoder,
            &view,
            &self.camera,
            &self.data,
            self.data.use_proxy || water_drawn,
            self.water.hits().filter(|_| water_drawn).map(|h| &h.read),
            queries.map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(5),
                end_of_pass_write_index: Some(6),
            }),
        );
        water_drawn
    }
    pub fn resize(&mut self, size: [u32; 2]) {
        self.config.width = size[0];
        self.config.height = size[1];
        self.camera.set_viewport(size[0], size[1]);
        self.target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Resized water test target"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.data.depth_texture = Some(Texture::create_depth_texture(
            &self.device,
            &self.config,
            "Resized water test depth",
        ));
    }
    pub fn frame(&mut self) -> RgbaFrame {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_frame(&mut encoder, None);
        self.queue.submit(Some(encoder.finish()));
        let dimensions = [self.config.width, self.config.height];
        RgbaFrame {
            width: dimensions[0],
            height: dimensions[1],
            pixels: read_texture_bytes(&self.device, &self.queue, &self.target, dimensions),
        }
    }
    /// GPU timestamps exclude CPU encoding, submission, readback and file IO.
    pub fn time_frame_ms(&mut self) -> Option<f64> {
        self.time_passes_ms().map(|v| v[0])
    }
    pub fn time_passes_ms(&mut self) -> Option<[f64; 4]> {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("water frame timing"),
            ty: wgpu::QueryType::Timestamp,
            count: 7,
        });
        let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 3 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT + 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let drawn = self.encode_frame(&mut encoder, Some(&queries));
        // Never resolve unwritten queries: some native backends wait for them.
        encoder.resolve_query_set(&queries, 0..1, &resolved, 0);
        if drawn {
            encoder.resolve_query_set(
                &queries,
                1..3,
                &resolved,
                wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
            );
            encoder.resolve_query_set(
                &queries,
                3..5,
                &resolved,
                2 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
            );
        }
        encoder.resolve_query_set(
            &queries,
            5..7,
            &resolved,
            3 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
        );
        self.queue.submit(Some(encoder.finish()));
        let bytes = crate::test_support::gpu::read_buffer(
            &self.device,
            &self.queue,
            &resolved,
            0,
            3 * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT + 16,
        );
        let values: &[u64] = bytemuck::cast_slice(&bytes);
        Some([(0, 97), (32, 33), (64, 65), (96, 97)].map(|(a, b)| {
            values[b].saturating_sub(values[a]) as f64 * self.queue.get_timestamp_period() as f64
                / 1e6
        }))
    }
    pub fn at_world(&self, frame: &RgbaFrame, position: Vec3) -> [u8; 3] {
        let clip = self.camera.view_proj() * position.extend(1.0);
        let x = ((clip.x / clip.w * 0.5 + 0.5) * self.config.width as f32) as usize;
        let y = ((0.5 - clip.y / clip.w * 0.5) * self.config.height as f32) as usize;
        frame.pixel(x, y)
    }
}
