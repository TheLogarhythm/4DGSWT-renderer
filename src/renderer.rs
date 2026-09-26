use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniforms};
use crate::log;
#[cfg(test)]
use crate::motion::MergedMotion;
#[cfg(test)]
use crate::motion_brush::MotionBrushPreview;
use crate::motion_brush::MotionOverlayChannel;
use crate::motion_brush_gpu::GpuMotionField;
use crate::motion_controller_gpu::GpuMotionControllerRuntime;
use crate::motion_gpu::GpuMotionRuntime;
use crate::motion_graph::MotionGraph;
use crate::motion_tagging::AuthoredRegistryUpdate;
use crate::profiler::FrameCounters;
#[cfg(test)]
use crate::scene_archive::DynamicArchiveSummary;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;

mod draw;
mod motion;
mod pipelines;
use draw::{DrawComposition, DrawResources, DrawSource, PreparedDraw, TILE_STRIDE};
use motion::{
    AuthoredDispatchKey, AuthoredRegistryKey, PreparedMotion, active_members,
    draw_requires_motion_pipeline, prepare_motion, preview_intersects_draw,
};
#[cfg(test)]
use motion::{build_motion_graph_outcome, map_coord_and_offset};
use pipelines::PipelineSet;
#[cfg(test)]
use pipelines::{gs_shader_source, strip_motion_field_shader_blocks};
const INSTANCE_BUFFER_GROWTH_BYTES: u64 = 4 * 1024 * 1024;

#[cfg(test)]
fn required_instance_bytes(splat_counts: impl IntoIterator<Item = usize>) -> Option<u64> {
    splat_counts.into_iter().try_fold(0_u64, |total, count| {
        total.checked_add(u64::try_from(count).ok()?.checked_mul(4)?)
    })
}

fn instance_buffer_capacity(current: u64, required: u64, limit: u64) -> Option<u64> {
    if required > limit {
        return None;
    }
    if required <= current {
        return Some(current);
    }
    let rounded = required
        .div_ceil(INSTANCE_BUFFER_GROWTH_BYTES)
        .checked_mul(INSTANCE_BUFFER_GROWTH_BYTES)
        .unwrap_or(limit);
    Some(rounded.min(limit))
}

pub struct GSWTRenderer {
    render_pipeline: wgpu::RenderPipeline,
    motion_field_render_pipeline: wgpu::RenderPipeline,
    water_render_pipeline: wgpu::RenderPipeline,
    water_motion_field_render_pipeline: wgpu::RenderPipeline,
    underwater_render_pipeline: wgpu::RenderPipeline,
    underwater_motion_field_render_pipeline: wgpu::RenderPipeline,
    empty_water_base_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,

    camera_uniforms_buffer: wgpu::Buffer,
    scene_uniforms_buffer: wgpu::Buffer,
    gaussian_texture: Texture,
    motion_runtime: Option<GpuMotionRuntime>,
    motion_controller_runtime: Option<GpuMotionControllerRuntime>,
    motion_field: Option<GpuMotionField>,
    motion_graph: Option<Arc<MotionGraph>>,
    motion_graph_error: Option<String>,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group: Option<wgpu::BindGroup>,
    motion_field_bind_group_layout: wgpu::BindGroupLayout,
    motion_field_bind_group: Option<wgpu::BindGroup>,
    empty_authored_base_rows_buffer: wgpu::Buffer,
    motion_field_render_active: bool,
    authored_registry_key: Option<AuthoredRegistryKey>,
    authored_registry_update: Option<Arc<AuthoredRegistryUpdate>>,
    authored_output_count: usize,
    authored_dispatch_key: Option<AuthoredDispatchKey>,
    authored_runtime_error: Option<String>,

    draw_resources: DrawResources,
    prepared_draws: Vec<PreparedDraw>,
    buffer_base_data: Vec<Vec<Vec<BufferDataValue>>>,

    user_data: UserData,
}
impl GSWTRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        preload_data: PreloadData,
    ) -> Result<Self, String> {
        let PipelineSet {
            render_pipeline,
            motion_field_render_pipeline,
            water_render_pipeline,
            water_motion_field_render_pipeline,
            underwater_render_pipeline,
            underwater_motion_field_render_pipeline,
            empty_water_base_group,
            scene_bind_group_layout,
            tile_bind_group_layout,
            motion_field_bind_group_layout,
        } = PipelineSet::new(device, config.format);

        // Vertex buffer
        let vertices = &mut [
            // quad
            Vertex2D {
                position: [-2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, -2.0],
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Scene global data
        let camera_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<CameraUniforms>() as u64,
            mapped_at_creation: false,
        });
        let scene_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<SceneUniforms>() as u64,
            mapped_at_creation: false,
        });
        let texture_width = preload_data.tile_splats_merged.tex_width as u32;
        let texture_height = preload_data.tile_splats_merged.tex_height as u32;
        let base_texels = preload_data.tile_splats_merged.tex_data.as_slice();
        let PreparedMotion {
            gaussian_texture,
            motion_runtime,
            motion_graph,
            motion_graph_error,
        } = prepare_motion(
            device,
            queue,
            preload_data.merged_motion,
            base_texels,
            texture_width,
            texture_height,
        )?;
        let draw_resources = DrawResources::new(device, &tile_bind_group_layout);

        // Preloaded instance buffers
        let mut buffer_base_data = Vec::with_capacity(preload_data.tile_base_data.len());
        for i in 0..preload_data.tile_base_data.len() {
            let tile_data_vec = &preload_data.tile_base_data[i];
            let mut tile_buf_vec: Vec<Vec<BufferDataValue>> =
                Vec::with_capacity(tile_data_vec.len());
            for j in 0..tile_data_vec.len() {
                let view_data_vec = &tile_data_vec[j];
                let mut view_buf_vec: Vec<BufferDataValue> =
                    Vec::with_capacity(view_data_vec.len());
                for k in 0..view_data_vec.len() {
                    let base_data = &view_data_vec[k];

                    let gs_index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("gs_index_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_index.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let lod_id_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("lod_id_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_lod_id.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let buffer_data = BufferDataValue {
                        splat_count: base_data.splat_count as u32,
                        gs_index_buffer,
                        lod_id_buffer: Some(lod_id_buffer),
                    };
                    view_buf_vec.push(buffer_data);
                }
                tile_buf_vec.push(view_buf_vec);
            }
            buffer_base_data.push(tile_buf_vec);
        }

        let empty_authored_base_rows_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Empty Authored Motion Base Rows"),
                contents: bytemuck::bytes_of(&0_u32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(Self {
            render_pipeline,
            motion_field_render_pipeline,
            empty_water_base_group,
            water_render_pipeline,
            water_motion_field_render_pipeline,
            underwater_render_pipeline,
            underwater_motion_field_render_pipeline,
            vertex_buffer,

            camera_uniforms_buffer,
            scene_uniforms_buffer,
            gaussian_texture,
            motion_runtime,
            motion_controller_runtime: None,
            motion_field: None,
            motion_graph,
            motion_graph_error,
            scene_bind_group_layout,
            scene_bind_group: None,
            motion_field_bind_group_layout,
            motion_field_bind_group: None,
            empty_authored_base_rows_buffer,
            motion_field_render_active: false,
            authored_registry_key: None,
            authored_registry_update: None,
            authored_output_count: 0,
            authored_dispatch_key: None,
            authored_runtime_error: None,

            draw_resources,
            prepared_draws: Vec::new(),
            buffer_base_data,

            user_data: UserData::new(),
        })
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        user_data: &UserData,
        render_data: &RenderData,
    ) {
        self.user_data = user_data.clone();
        self.draw_resources.invalidate();

        let mut group_entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.scene_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&self.gaussian_texture.view),
            },
        ];

        let height_map: Texture;
        height_map = Texture::from_bytes(
            device,
            queue,
            bytemuck::cast_slice(self.user_data.height_map.as_slice()),
            self.user_data.height_map_wh.x as u32,
            self.user_data.height_map_wh.y as u32,
            4,
            wgpu::TextureFormat::R32Float,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::Repeat,
            Some("Dummy Height Map Texture"),
        )
        .unwrap();
        group_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&height_map.view),
        });
        group_entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Sampler(height_map.sampler.as_ref().unwrap()),
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.scene_bind_group_layout,
            entries: &group_entries,
            label: Some("scene_bind_group"),
        });

        self.scene_bind_group = Some(scene_bind_group);
    }

    /// Standalone entry point for tests and offscreen callers.
    /// Render a standalone frame. A renderer owns one stream of sort revisions:
    /// replacing or mutating sort rows must advance `RenderData::sort_revision`,
    /// including when switching between offscreen inputs. `configure` resets
    /// residency when starting a new revision sequence.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
        depth_prepared: bool,
        water_hits: Option<&wgpu::BindGroup>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> FrameCounters {
        let frame = crate::water::WaterFrame::new(camera, &self.user_data, render_data);
        self.render_prepared(
            device,
            queue,
            encoder,
            view,
            camera,
            render_data,
            depth_prepared,
            water_hits,
            timestamp_writes,
            &frame,
        )
    }

    /// Shares frame inputs with preceding passes; uses the same sort-revision
    /// ownership contract as `render`.
    pub(crate) fn render_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
        depth_prepared: bool,
        water_hits: Option<&wgpu::BindGroup>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
        frame: &crate::water::WaterFrame,
    ) -> FrameCounters {
        let scene_data = render_data.cur_scene_data.as_ref().unwrap();
        let sort_data = render_data.cur_sort_data.as_ref().unwrap();
        let render_config = &render_data.render_config;
        let profiling = render_data.profiler_enabled;
        let water_active = water_hits.is_some() && frame.active;
        let underwater_active = water_active && frame.underwater.is_some();
        let view_proj = camera.view_proj();
        let visible_draws: Vec<bool> = sort_data
            .render_data_vec
            .iter()
            .enumerate()
            .map(|(i, (render_data_key, _))| {
                let tile_instance = &sort_data.tile_instance_vec[i];
                let mut culled = false;
                // A curved patch can enter the frustum while all four corners
                // remain outside it. The planar corner test is not conservative.
                if self.user_data.surface_type != SurfaceType::Sphere
                    && render_data_key.tid.len() == 1
                {
                    let mut pos2d = vec3(f32::MAX, f32::MAX, -f32::MAX);
                    for ci in 0..4 {
                        let corner = view_proj
                            * tile_instance.corner_data.as_ref().unwrap()[ci]
                                .0
                                .extend(1.0);
                        let corner = corner.truncate() / corner.w;
                        pos2d.x = pos2d.x.min(corner.x.abs());
                        pos2d.y = pos2d.y.min(corner.y.abs());
                        pos2d.z = pos2d.z.max(corner.z);
                    }
                    let clip = render_config.culling_dist;
                    culled = pos2d.z < -clip || pos2d.x > clip || pos2d.y > clip;
                }
                !culled && render_config.lod_enable[tile_instance.tid.0]
            })
            .collect();
        let (motion_overlay, motion_preview) = render_data
            .motion
            .as_ref()
            .and_then(|motion| motion.spatial.as_ref())
            .map(|spatial| (spatial.overlay, spatial.preview()))
            .unwrap_or((MotionOverlayChannel::Off, None));
        // Tagged rows must keep using the authored pipeline while an asynchronous
        // membership update is in flight, even if the field was just erased.
        // The shader falls back to the matching global base row when the field
        // uniform is disabled.
        let motion_bind_available = self.motion_field_bind_group.is_some();
        let motion_diagnostics_active = self.motion_field_render_active;
        let authored_runtime_ready = self.authored_runtime_error.is_none()
            && self.authored_output_count > 0
            && self.authored_registry_key.is_some_and(|key| {
                key.request_revision == sort_data.authored.request_revision
                    && key.registry_revision == sort_data.authored.registry_revision
                    && key.scene_id == scene_data.scene_id
            });

        self.prepared_draws.clear();
        let mut stream_offset = 0u64;
        for (i, visible) in visible_draws.iter().copied().enumerate() {
            if !visible {
                continue;
            }
            let tile = &sort_data.tile_instance_vec[i];
            let (key, value) = &sort_data.render_data_vec[i];
            let authored = sort_data
                .authored
                .authored_draws
                .get(i)
                .copied()
                .unwrap_or(false);
            let mut uniform = TileUniforms::from_tile(tile, value);
            uniform.authored_tag_mode = u32::from(authored && authored_runtime_ready);
            let motion = motion_bind_available
                && draw_requires_motion_pipeline(
                    authored,
                    if motion_diagnostics_active {
                        motion_overlay
                    } else {
                        MotionOverlayChannel::Off
                    },
                    motion_diagnostics_active
                        && preview_intersects_draw(
                            motion_preview,
                            tile,
                            value,
                            &self.user_data,
                            scene_data,
                        ),
                );
            let source = if let Some(value) = value {
                let Some(count) = u32::try_from(value.splat_count).ok() else {
                    log!("Draw instance count exceeds u32; skipping frame");
                    return FrameCounters::default();
                };
                if count == 0 {
                    continue;
                }
                let bytes = u64::from(count) * 4;
                let source = DrawSource::Streamed {
                    offset: stream_offset,
                    bytes,
                    per_splat_lod: value.single_lod_id == -1,
                };
                let Some(next) = stream_offset.checked_add(bytes) else {
                    return FrameCounters::default();
                };
                stream_offset = next;
                source
            } else {
                let lod = match tile.transition_status {
                    TileTransitionStatus::Changing(false) => tile.tid.0 - 1,
                    _ => tile.tid.0,
                };
                let splats = self.buffer_base_data[lod][tile.tid.1][tile.view_id].splat_count;
                if splats == 0 {
                    continue;
                }
                DrawSource::Presorted { splats }
            };
            if render_config.debug_log && key.tid.len() >= 9 {
                log!("{tile:?}\n{key:?}\n{value:?}");
            }
            self.prepared_draws.push(PreparedDraw {
                sort_index: i,
                source,
                uniform,
                motion,
            });
        }
        let Some(uploads) = self.draw_resources.prepare(
            device,
            queue,
            sort_data,
            render_data.sort_revision,
            &self.prepared_draws,
        ) else {
            log!("Draw uploads exceed GPU limits or contain invalid data; skipping frame");
            return FrameCounters::default();
        };
        queue.write_buffer(
            &self.camera_uniforms_buffer,
            0,
            bytemuck::bytes_of(&CameraUniforms::from_camera(camera)),
        );
        queue.write_buffer(
            &self.scene_uniforms_buffer,
            0,
            bytemuck::bytes_of(&SceneUniforms::from_data(
                &self.user_data,
                scene_data,
                render_data,
                frame,
            )),
        );
        let mut counters = if profiling {
            FrameCounters::for_render(
                scene_data.splat_count,
                scene_data.blending_splat_count,
                (std::mem::size_of::<CameraUniforms>() + std::mem::size_of::<SceneUniforms>())
                    as u64
                    + uploads.bytes,
            )
        } else {
            FrameCounters::default()
        };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_data.depth_texture.as_ref().unwrap().view,
                depth_ops: Some(wgpu::Operations {
                    load: if depth_prepared {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes,
        });

        let mut using_motion_pipeline = false;
        render_pass.set_pipeline(if underwater_active {
            &self.underwater_render_pipeline
        } else if water_active {
            &self.water_render_pipeline
        } else {
            &self.render_pipeline
        });
        render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
        if water_active {
            render_pass.set_bind_group(2, &self.empty_water_base_group, &[]);
            render_pass.set_bind_group(3, water_hits.unwrap(), &[]);
        }
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        for (draw_index, draw) in self.prepared_draws.iter().enumerate() {
            let tile_instance = &sort_data.tile_instance_vec[draw.sort_index];
            let tid = tile_instance.tid;
            if draw.motion != using_motion_pipeline {
                using_motion_pipeline = draw.motion;
                render_pass.set_pipeline(
                    match (using_motion_pipeline, water_active, underwater_active) {
                        (true, true, true) => &self.underwater_motion_field_render_pipeline,
                        (false, true, true) => &self.underwater_render_pipeline,
                        (true, true, false) => &self.water_motion_field_render_pipeline,
                        (true, false, false) => &self.motion_field_render_pipeline,
                        (false, true, false) => &self.water_render_pipeline,
                        (false, false, false) => &self.render_pipeline,
                        (_, false, true) => unreachable!("underwater requires water"),
                    },
                );
                render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
                if using_motion_pipeline {
                    render_pass.set_bind_group(
                        2,
                        self.motion_field_bind_group
                            .as_ref()
                            .expect("authored render path requires a motion field bind group"),
                        &[],
                    );
                } else if water_active {
                    render_pass.set_bind_group(2, &self.empty_water_base_group, &[]);
                }
                if water_active {
                    render_pass.set_bind_group(3, water_hits.unwrap(), &[]);
                }
            }

            let splat_count;
            match draw.source {
                DrawSource::Streamed {
                    offset,
                    bytes,
                    per_splat_lod,
                } => {
                    splat_count = (bytes / 4) as u32;
                    render_pass.set_vertex_buffer(
                        1,
                        self.draw_resources.indices.slice(offset..offset + bytes),
                    );
                    render_pass.set_vertex_buffer(
                        2,
                        self.draw_resources.map_ids.slice(offset..offset + bytes),
                    );
                    render_pass.set_vertex_buffer(
                        3,
                        if per_splat_lod {
                            self.draw_resources.lod_ids.slice(offset..offset + bytes)
                        } else {
                            self.draw_resources.lod_ids.slice(..)
                        },
                    );
                }
                DrawSource::Presorted { .. } => {
                    let base_data: &BufferDataValue;
                    if let TileTransitionStatus::Changing(to_lower) =
                        tile_instance.transition_status
                    {
                        if to_lower {
                            base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                        } else {
                            base_data =
                                &self.buffer_base_data[tid.0 - 1][tid.1][tile_instance.view_id];
                        }
                    } else {
                        base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                    }
                    splat_count = base_data.splat_count;

                    render_pass.set_vertex_buffer(1, base_data.gs_index_buffer.slice(..));
                    render_pass.set_vertex_buffer(2, self.draw_resources.map_ids.slice(..));
                    render_pass
                        .set_vertex_buffer(3, base_data.lod_id_buffer.as_ref().unwrap().slice(..));
                }
            }
            render_pass.set_bind_group(
                1,
                &self.draw_resources.tile_bind_group,
                &[(draw_index as u64 * TILE_STRIDE) as u32],
            );
            render_pass.draw(0..6, 0..splat_count);
            if profiling {
                counters.record_draw(
                    splat_count as usize,
                    0,
                    active_members(&sort_data.render_data_vec[draw.sort_index].0),
                );
            }
        }
        counters
    }
}

struct BufferDataValue {
    splat_count: u32,
    gs_index_buffer: wgpu::Buffer,
    lod_id_buffer: Option<wgpu::Buffer>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    splat_scale: f32,
    tile_width: f32,
    use_clip: u32,
    clip_height: f32,
    surface_type: u32,
    sphere_radius: f32,
    point_cloud_radius: f32,
    transition_width_ratio: f32,
    num_lod: u32,
    draw_mode: u32,

    map_half_wh: [u32; 2],
    center_coord: [i32; 2],
    sphere_tiles_per_face: u32,
    _pad0: u32,
    transition_dist_vec: [f32; 16],
    height_map_scale: [f32; 4],
    scene_scale: [f32; 4],
    water_level: [f32; 4],
    water_bounds: [f32; 4],
    water_waves: [f32; 4],
    water_phases: [f32; 4],
}
impl SceneUniforms {
    fn expand_to_array<const N: usize, T: Copy>(slice: &[T], pad_value: T) -> [T; N] {
        let mut arr = [pad_value; N];
        let len = slice.len().min(N); // avoid overflow
        arr[..len].copy_from_slice(&slice[..len]);
        arr
    }

    fn lod_count(user_data: &UserData) -> u32 {
        user_data.n_tiles.0 as u32
    }

    fn from_data(
        user_data: &UserData,
        scene_data: &SceneData,
        render_data: &RenderData,
        frame: &crate::water::WaterFrame,
    ) -> Self {
        let render_config = &render_data.render_config;
        Self {
            water_level: [
                render_config.water.height,
                u32::from(frame.active) as f32,
                0.0,
                0.0,
            ],
            water_bounds: frame.bounds.unwrap_or([0.0; 4]),
            water_waves: render_config.water.waves(),
            water_phases: render_config.water.phases(),
            splat_scale: render_config.splat_scale,
            tile_width: user_data.tile_width,
            use_clip: render_config.use_clip as u32,
            clip_height: render_config.clip_height,
            surface_type: user_data.surface_type as u32,
            sphere_radius: user_data.sphere_radius,
            point_cloud_radius: if render_config.draw_point_cloud {
                render_config.point_cloud_radius
            } else {
                0.0
            },
            transition_width_ratio: user_data.lod_transition_width_ratio,
            num_lod: Self::lod_count(user_data),
            draw_mode: render_config.draw_mode as u32,

            map_half_wh: [
                user_data.tile_map_half_wh.x as u32,
                user_data.tile_map_half_wh.y as u32,
            ],
            center_coord: [scene_data.center_coord.x, scene_data.center_coord.y],
            sphere_tiles_per_face: user_data.sphere_tiles_per_face as u32,
            _pad0: 0,
            transition_dist_vec: Self::expand_to_array::<16, f32>(
                &user_data.lod_transition_dist,
                0.0,
            ),
            scene_scale: [
                if user_data.surface_type == SurfaceType::Sphere {
                    1.0
                } else {
                    render_config.scene_scale.x
                },
                if user_data.surface_type == SurfaceType::Sphere {
                    1.0
                } else {
                    render_config.scene_scale.y
                },
                if user_data.surface_type == SurfaceType::Sphere {
                    1.0
                } else {
                    render_config.scene_scale.z
                },
                0.0,
            ],
            height_map_scale: [
                user_data.height_map_scale.x,
                user_data.height_map_scale.y,
                user_data.height_map_scale.z * render_config.height_map_scale_v,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniforms {
    single_draw: u32,
    map_index: u32,
    single_lod_id: i32,
    valid_lod_id: i32,
    changing: u32,
    changing_to_lower: i32,
    authored_tag_mode: u32,
    _pad0: u32,

    tile_id: [u32; 4],
    offset: [f32; 4],
    map_coord: [u32; 4],
}
impl TileUniforms {
    fn from_tile(tile: &TileInstance, render_data_value: &Option<RenderDataValue>) -> Self {
        let mut uniforms = Self {
            single_draw: 0,
            map_index: tile.map_index as u32,
            single_lod_id: -1,
            valid_lod_id: -1,
            changing: 0,
            changing_to_lower: -1,
            authored_tag_mode: 0,
            _pad0: 0,

            tile_id: [tile.tid.0 as u32, tile.tid.1 as u32, tile.view_id as u32, 0],
            offset: [
                tile.tile_offset.x,
                tile.tile_offset.y,
                tile.tile_offset.z,
                0.0,
            ],
            map_coord: [tile.map_coord.x as u32, tile.map_coord.y as u32, 0, 0],
        };

        if let Some(data_value) = render_data_value {
            uniforms.single_draw = 1;
            uniforms.single_lod_id = data_value.single_lod_id;
            uniforms.changing = (uniforms.single_lod_id == -1) as u32;
            // Streaming indices does not imply merging multiple tiles. Preserve
            // a single tile's known LoD pair (rear sphere and authored streams).
            if DrawComposition::of(tile) == DrawComposition::Single {
                if let TileTransitionStatus::Changing(to_lower) = tile.transition_status {
                    uniforms.changing_to_lower = to_lower as i32;
                }
            }
        } else {
            if let TileTransitionStatus::Changing(to_lower) = tile.transition_status {
                uniforms.changing = 1;
                uniforms.changing_to_lower = to_lower as i32;
            } else {
                uniforms.valid_lod_id = tile.tid.0 as i32;
            }
        }

        uniforms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubed_sphere_max_n_fits_grown_uniform_allocation() {
        let n = crate::cubed_sphere::MAX_FACE_TILES as u64;
        let required = 6 * n * n * 256;
        let capacity = instance_buffer_capacity(20_000 * 256, required, 256 * 1024 * 1024).unwrap();
        assert!(capacity >= required);
        assert!(capacity > 20_000 * 256);
        assert_eq!(
            instance_buffer_capacity(capacity, required, 256 * 1024 * 1024),
            Some(capacity)
        );
    }

    #[test]
    fn instance_buffer_plan_grows_for_the_iceberg_camera_frame() {
        let current = 40_000_000;
        let visible_splats = [9_972_941, 352_020];
        let required = required_instance_bytes(visible_splats).unwrap();
        assert_eq!(required, 41_299_844);
        let capacity = instance_buffer_capacity(current, required, 256 * 1024 * 1024).unwrap();
        assert_eq!(capacity, 41_943_040);
        assert_eq!(
            instance_buffer_capacity(capacity, required, 256 * 1024 * 1024),
            Some(capacity)
        );
    }

    #[test]
    fn instance_buffer_plan_respects_device_limit() {
        assert_eq!(
            instance_buffer_capacity(40_000_000, 41_299_844, 41_000_000),
            None
        );
        assert_eq!(
            instance_buffer_capacity(40_000_000, 41_299_844, 41_299_844),
            Some(41_299_844)
        );
        if usize::BITS == 64 {
            assert_eq!(required_instance_bytes([usize::MAX, usize::MAX]), None);
        }
    }

    #[test]
    fn brush_preview_uses_current_compact_membership_after_cached_merge_remap() {
        let mut user = UserData::new();
        user.tile_map_wh = vec2(4, 1);
        user.tile_map_half_wh = vec2(0, 0);
        user.tile_width = 4.0;
        let scene = SceneData::new();
        let tile = TileInstance::new();
        let preview = MotionBrushPreview {
            center: [8.5, 0.5],
            radius: 0.2,
            opacity: 1.0,
            falloff: crate::motion_brush::MotionBrushFalloff::Constant,
            tool: crate::motion_brush::MotionBrushTool::Erase,
            color: [1.0; 4],
        };
        let original = RenderDataValue {
            splat_count: 3,
            gs_index: vec![10, 11, 12],
            gs_map_id: vec![0, 1, 0],
            merge_from_vec: vec![0, 1],
            single_lod_id: 0,
            gs_lod_id: None,
        };
        assert!(!preview_intersects_draw(
            Some(preview),
            &tile,
            &Some(original.clone()),
            &user,
            &scene
        ));
        let mut moved = original.clone();
        moved.remap_instances(&[1, 2]).unwrap();
        assert_eq!(moved.gs_map_id, vec![1, 2, 1]);
        assert_eq!(moved.merge_from_vec, vec![1, 2]);
        assert_eq!(original.gs_map_id, vec![0, 1, 0]);
        assert!(preview_intersects_draw(
            Some(preview),
            &tile,
            &Some(moved),
            &user,
            &scene
        ));
    }
    use crate::dynamic_archive::BasisBank;
    use crate::motion::{
        ALL_MOTION_MASK, CanonicalGaussian, MOTION_SAMPLE_COUNT, MergedMotionData, TimelineSample,
    };
    use crate::motion_behavior::MotionChannelGains;
    use crate::motion_controller_palette::MotionControllerFrame;

    #[test]
    fn scene_uniform_uses_lod_count_instead_of_tile_count() {
        let mut user_data = UserData::new();
        user_data.n_tiles = (6, 16, 9);

        assert_eq!(SceneUniforms::lod_count(&user_data), 6);
    }

    #[test]
    fn active_members_include_both_sides_of_lod_transitions() {
        let key = RenderDataKey {
            view_id: 0,
            tid: vec![(2, 7), (4, 8), (1, 9)],
            transition_status: vec![
                TileTransitionStatusHash::Changing(true),
                TileTransitionStatusHash::Changing(false),
                TileTransitionStatusHash::None,
            ],
        };

        assert_eq!(
            active_members(&key),
            vec![(7, 2), (7, 3), (8, 4), (8, 3), (9, 1)]
        );
    }

    #[test]
    fn invalid_graph_analysis_is_non_fatal() {
        let motion = MergedMotion {
            data: MergedMotionData::Legacy {
                basis: BasisBank {
                    basis_count: 1,
                    values: vec![0.0; MOTION_SAMPLE_COUNT * 9],
                },
                top_k: 1,
                basis_ids: vec![0],
                weights: vec![1.0],
            },
            canonical: vec![CanonicalGaussian {
                position: [0.0; 3],
                log_scale: [0.0; 3],
                rotation: [1.0, 0.0, 0.0, 0.0],
            }],
            transform_ids: vec![0],
            motion_channel_masks: vec![ALL_MOTION_MASK],
            source_duration_seconds: 0.0,
            summary: DynamicArchiveSummary {
                schema_version: 2,
                tile_count: 1,
                lod_count: 1,
                basis_count: 1,
                top_k: 1,
                total_rows: 1,
                backend: "invalid-graph-fixture".into(),
            },
            member_offsets: vec![vec![0]],
            member_counts: vec![vec![1]],
        };

        let outcome = build_motion_graph_outcome(&motion);

        assert!(outcome.graph.is_none());
        assert!(outcome.error.is_some_and(|error| !error.is_empty()));
    }

    #[test]
    fn authored_dispatch_key_is_sort_independent_but_tracks_compute_inputs() {
        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Source { u: 0.25 },
            gains: MotionChannelGains::default(),
            valid: true,
        };
        let local = MotionControllerFrame { slot: 1, ..global };
        let key = AuthoredDispatchKey::new(global, &[local], true, 7, 13, 17);

        assert_eq!(
            key,
            AuthoredDispatchKey::new(global, &[local], true, 7, 13, 17)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(
                MotionControllerFrame {
                    sample: TimelineSample::Source { u: 0.5 },
                    ..global
                },
                &[local],
                true,
                7,
                13,
                17,
            )
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(global, &[local], false, 7, 13, 17)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(global, &[local], true, 3, 13, 17)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(global, &[local], true, 7, 19, 17)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(global, &[local], true, 7, 13, 23)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(global, &[local], true, 7, 29, 17)
        );
        assert_ne!(
            key,
            AuthoredDispatchKey::new(
                global,
                &[MotionControllerFrame {
                    gains: MotionChannelGains {
                        master: 0.5,
                        ..MotionChannelGains::default()
                    },
                    ..local
                }],
                true,
                7,
                13,
                17,
            )
        );
    }

    #[test]
    fn authored_render_fast_path_selects_only_tags_or_visible_diagnostics() {
        assert!(!draw_requires_motion_pipeline(
            false,
            MotionOverlayChannel::Off,
            false,
        ));
        assert!(draw_requires_motion_pipeline(
            true,
            MotionOverlayChannel::Off,
            false,
        ));
        assert!(draw_requires_motion_pipeline(
            false,
            MotionOverlayChannel::Strength,
            false,
        ));
        assert!(draw_requires_motion_pipeline(
            false,
            MotionOverlayChannel::Off,
            true,
        ));
    }

    #[test]
    fn authored_tag_shader_and_tile_uniform_match_compact_registry_contract() {
        assert_eq!(std::mem::size_of::<TileUniforms>(), 80);
        let uniforms = TileUniforms::from_tile(&TileInstance::new(), &None);
        let words = bytemuck::cast_slice::<TileUniforms, u32>(std::slice::from_ref(&uniforms));
        assert_eq!(words[6], 0);
        let shader = include_str!("gswt.wgsl");
        assert!(shader.contains("@group(2) @binding(5)"));
        assert!(shader.contains("u_authored_base_rows"));
        assert!(shader.contains("gs_index & 0x80000000u"));
        assert!(shader.contains("gs_index & 0x7fffffffu"));
        assert!(!shader.contains("u_authored_override_indices"));
        assert!(!shader.contains("authored_override_base"));

        let mut user_data = UserData::new();
        user_data.tile_map_half_wh = vec2(48, 48);
        user_data.tile_map_wh = vec2(97, 97);
        user_data.tile_width = 4.0;
        let mut scene = SceneData::new();
        scene.center_coord = vec2(10, -5);
        let center_map_id = 48_u32 * 97 + 48;

        let (coord, offset) = map_coord_and_offset(center_map_id, &user_data, &scene).unwrap();

        assert_eq!(coord, [48, 48]);
        assert_eq!(offset, [40.0, -20.0, 0.0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn base_and_motion_field_shaders_pass_native_wgsl_validation() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("motion field shader test skipped: no native adapter ({error})");
                    return;
                }
            };
        let (device, _) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("motion field shader validation device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();
        let authored = gs_shader_source(true, false);
        let base = strip_motion_field_shader_blocks(&authored);
        assert!(!base.contains("@group(2)"));
        assert!(!base.contains("u_motion_field"));
        for (label, source) in [
            ("base shader", base.as_str()),
            ("motion field shader", authored.as_str()),
        ] {
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let error = pollster::block_on(device.pop_error_scope());
            assert!(error.is_none(), "{label} WGSL validation failed: {error:?}");
        }
    }
    #[test]
    fn dry_gs_shader_has_no_water_depth_or_intersection_work() {
        let source = strip_motion_field_shader_blocks(&gs_shader_source(false, false));
        assert!(
            !source.contains("@builtin(frag_depth)"),
            "disabled water must use a color-only fragment shader"
        );
        assert!(
            !source.contains("water_surface_hit("),
            "disabled water cannot carry the iterative solver"
        );
        assert!(
            !source.contains("water_moments"),
            "disabled water cannot evaluate water cut moments"
        );
    }
    #[test]
    fn water_gs_shader_reads_shared_hits_without_solver() {
        let source = gs_shader_source(true, false);
        assert!(
            !source.contains("fn water_surface_hit("),
            "GS must not compile the iterative water solver"
        );
        assert!(
            source.contains("load_water_hit("),
            "GS must consume the current frame's shared hit texture"
        );
    }

    #[test]
    fn underwater_shader_is_a_separate_water_only_variant() {
        let dry = gs_shader_source(false, false);
        let wet = gs_shader_source(true, false);
        let underwater = gs_shader_source(true, true);
        assert!(!dry.contains("underwater_transmission"));
        assert!(!wet.contains("underwater_transmission"));
        assert!(underwater.contains("underwater_transmission"));
        assert!(underwater.contains("load_water_hit("));
    }
}
