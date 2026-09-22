use std::sync::Arc;

use wgpu::BufferAddress;
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniforms};
use crate::log;
use crate::motion::{MergedMotion, TimelineSample};
use crate::motion_brush::{
    DirtyRect, MotionBrushPreview, MotionBrushRuntime, MotionFieldCache, MotionOverlayChannel,
};
use crate::motion_brush_gpu::{GpuMotionField, MotionFieldUniform};
use crate::motion_controller_gpu::{AuthoredMotionDispatch, GpuMotionControllerRuntime};
use crate::motion_controller_palette::{CompiledMotionField, MotionControllerFrame};
use crate::motion_gpu::GpuMotionRuntime;
use crate::motion_graph::{MotionGraph, MotionGraphSettings};
use crate::motion_graph_analysis::build_motion_graph;
use crate::motion_tagging::AuthoredRegistryUpdate;
use crate::profiler::{FrameCounters, MotionDispatchWork};
use crate::scene_archive::DynamicArchiveSummary;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;

pub struct GSWTRenderer {
    render_pipeline: wgpu::RenderPipeline,
    motion_field_render_pipeline: wgpu::RenderPipeline,
    water_render_pipeline: wgpu::RenderPipeline,
    water_motion_field_render_pipeline: wgpu::RenderPipeline,
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

    tile_uniforms_buffer: wgpu::Buffer,
    tile_bind_group: wgpu::BindGroup,

    gs_index_buffer: wgpu::Buffer,
    map_id_buffer: wgpu::Buffer,
    lod_id_buffer: wgpu::Buffer,
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
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("scene_bind_group_layout"),
            });

        // TODO: change back to uniform
        let tile_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                }],
                label: Some("tile_bind_group_layout"),
            });

        let motion_field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion authoring field bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let motion_field_shader_source = gs_shader_source(false);
        let base_shader_source = strip_motion_field_shader_blocks(&motion_field_shader_source);
        let base_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Base GSWT shader"),
            source: wgpu::ShaderSource::Wgsl(base_shader_source.into()),
        });
        let motion_field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial motion authoring GSWT shader"),
            source: wgpu::ShaderSource::Wgsl(motion_field_shader_source.into()),
        });

        let wet_source = gs_shader_source(true);
        let wet_base = strip_motion_field_shader_blocks(&wet_source);
        let wet_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water-clipped GS shader"),
            source: wgpu::ShaderSource::Wgsl(wet_base.into()),
        });
        let wet_authored_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water-clipped authored shader"),
            source: wgpu::ShaderSource::Wgsl(wet_source.into()),
        });
        let base_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Base render pipeline layout"),
                bind_group_layouts: &[&scene_bind_group_layout, &tile_bind_group_layout],
                push_constant_ranges: &[],
            });
        let motion_field_render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Spatial motion authoring render pipeline layout"),
                bind_group_layouts: &[
                    &scene_bind_group_layout,
                    &tile_bind_group_layout,
                    &motion_field_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Unused water base group 2"),
            entries: &[],
        });
        let empty_water_base_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &empty_layout,
            entries: &[],
        });
        let hit_layout = crate::water_hits::WaterHits::read_layout(device);
        let water_base_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water base layout"),
            bind_group_layouts: &[
                &scene_bind_group_layout,
                &tile_bind_group_layout,
                &empty_layout,
                &hit_layout,
            ],
            push_constant_ranges: &[],
        });
        let water_authored_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Water authored layout"),
                bind_group_layouts: &[
                    &scene_bind_group_layout,
                    &tile_bind_group_layout,
                    &motion_field_bind_group_layout,
                    &hit_layout,
                ],
                push_constant_ranges: &[],
            });
        let render_pipeline = create_render_pipeline(
            device,
            config.format,
            &base_render_pipeline_layout,
            &base_shader,
            "Base render pipeline",
            wgpu::CompareFunction::Less,
        );
        let motion_field_render_pipeline = create_render_pipeline(
            device,
            config.format,
            &motion_field_render_pipeline_layout,
            &motion_field_shader,
            "Spatial motion authoring render pipeline",
            wgpu::CompareFunction::Less,
        );
        // Water-clipped Gaussian mass may lie exactly on the water depth. Keep
        // ordinary rendering's strict test; only water uses equality, without
        // a perspective-depth bias that could pull GS through nearer terrain.
        let water_render_pipeline = create_render_pipeline(
            device,
            config.format,
            &water_base_layout,
            &wet_shader,
            "Water-clipped GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );
        let water_motion_field_render_pipeline = create_render_pipeline(
            device,
            config.format,
            &water_authored_layout,
            &wet_authored_shader,
            "Water-clipped authored GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );

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
        let graph_outcome = preload_data
            .merged_motion
            .as_ref()
            .map(build_motion_graph_outcome)
            .unwrap_or_default();
        if let Some(graph) = graph_outcome.graph.as_ref() {
            let summary = graph.summary();
            log!(
                "Motion graph: {} jumps across {} segments ({} without jumps)",
                summary.jump_count,
                summary.segment_count,
                summary.segment_count - summary.segments_with_jumps
            );
        } else if let Some(error) = graph_outcome.error.as_deref() {
            log!("Motion graph unavailable; ordinary playback remains active: {error}");
        }
        let motion_runtime = preload_data
            .merged_motion
            .map(|motion| {
                GpuMotionRuntime::new(
                    device,
                    queue,
                    motion,
                    base_texels,
                    texture_width,
                    texture_height,
                )
            })
            .transpose()?;
        let gaussian_texture = if let Some(runtime) = motion_runtime.as_ref() {
            runtime.output_texture().clone()
        } else {
            Texture::from_bytes(
                device,
                queue,
                transmute_slice::<_, u8>(base_texels),
                texture_width,
                texture_height,
                16,
                wgpu::TextureFormat::Rgba32Uint,
                wgpu::FilterMode::Nearest,
                wgpu::AddressMode::ClampToEdge,
                Some("Gaussian Texture"),
            )
            .map_err(|error| error.to_string())?
        };
        let tile_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Per-instance Tile Uniforms Storage Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: 20_000 * u64::max(256, std::mem::size_of::<TileUniforms>() as u64),
            mapped_at_creation: false,
        });
        let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &tile_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &tile_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(256),
                }),
            }],
            label: Some("tile_bind_group"),
        });

        // Instance buffers
        let gs_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("gs_index_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let map_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("map_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let lod_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("lod_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });

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
            vertex_buffer,

            camera_uniforms_buffer,
            scene_uniforms_buffer,
            gaussian_texture,
            motion_runtime,
            motion_controller_runtime: None,
            motion_field: None,
            motion_graph: graph_outcome.graph,
            motion_graph_error: graph_outcome.error,
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

            tile_uniforms_buffer,
            tile_bind_group,

            gs_index_buffer,
            map_id_buffer,
            lod_id_buffer,
            buffer_base_data,

            user_data: UserData::new(),
        })
    }

    pub fn update_motion(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        channel_gains: crate::motion_behavior::MotionChannelGains,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<Option<MotionDispatchWork>, String> {
        if let Some(runtime) = self.motion_runtime.as_mut() {
            return runtime
                .dispatch_masked_profiled(
                    queue,
                    encoder,
                    sample,
                    motion_enabled,
                    preview_channel_mask,
                    channel_gains,
                    timestamp_writes,
                )
                .map(Some);
        }
        Ok(None)
    }

    pub fn motion_summary(&self) -> Option<&DynamicArchiveSummary> {
        self.motion_runtime.as_ref().map(GpuMotionRuntime::summary)
    }

    pub fn motion_source_duration_seconds(&self) -> Option<f32> {
        self.motion_runtime
            .as_ref()
            .map(GpuMotionRuntime::source_duration_seconds)
    }

    pub fn motion_graph(&self) -> Option<Arc<MotionGraph>> {
        self.motion_graph.clone()
    }

    pub fn motion_graph_error(&self) -> Option<&str> {
        self.motion_graph_error.as_deref()
    }

    /// Publish a prepared field (or lightweight preview). Both initial config
    /// and ordinary frames use this transaction; failed uploads remain dirty.
    pub fn sync_authoring_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spatial: Option<&mut MotionBrushRuntime>,
        compiled: Option<&CompiledMotionField>,
    ) -> Result<u64, String> {
        let Some(spatial) = spatial.filter(|spatial| spatial.render_path_active()) else {
            self.disable_motion_field_rendering();
            return Ok(0);
        };
        let Some(compiled) = compiled else {
            self.disable_motion_field_rendering();
            return Err("motion field must be compiled before upload".into());
        };
        let active = spatial.field_data_active();
        let mut dirty = if active {
            spatial.take_dirty_regions()
        } else {
            Vec::new()
        };
        dirty.extend_from_slice(compiled.dirty_regions());
        dirty.sort_unstable_by_key(|rect| (rect.min(), rect.max_exclusive()));
        dirty.dedup();
        let result = if active {
            self.sync_motion_field(device, queue, spatial.cache(), compiled, &dirty)
        } else {
            self.sync_motion_preview(device, spatial.cache());
            Ok(0)
        };
        match result {
            Ok(bytes) => {
                self.update_motion_field_uniform(
                    queue,
                    spatial.cache(),
                    active,
                    spatial.overlay,
                    compiled.palette_count(),
                    spatial.preview(),
                );
                Ok(bytes)
            }
            Err(error) => {
                spatial.restore_dirty_regions(dirty);
                self.disable_motion_field_rendering();
                Err(error)
            }
        }
    }

    fn sync_motion_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        compiled: &CompiledMotionField,
        dirty_regions: &[DirtyRect],
    ) -> Result<u64, String> {
        let dimensions = cache.layout().texture_size();
        if self
            .motion_field
            .as_ref()
            .is_none_or(|field| field.dimensions() != dimensions)
        {
            let field =
                GpuMotionField::new(device, queue, cache).map_err(|error| error.to_string())?;
            field
                .upload_full_compiled(queue, cache, compiled)
                .map_err(|error| error.to_string())?;
            let stable_gaussian_view = self
                .motion_runtime
                .as_ref()
                .map(|runtime| &runtime.base_texture().view)
                .unwrap_or(&self.gaussian_texture.view);
            let mut motion_controller_runtime = self
                .motion_runtime
                .as_ref()
                .map(|motion| GpuMotionControllerRuntime::new(device, motion, &field))
                .transpose()?;
            if let (Some(controller), Some(motion), Some(update)) = (
                motion_controller_runtime.as_mut(),
                self.motion_runtime.as_ref(),
                self.authored_registry_update.as_ref(),
            ) {
                controller.update_registry(device, queue, motion, &field, update)?;
            }
            let authored_gaussian_view = motion_controller_runtime
                .as_ref()
                .map(GpuMotionControllerRuntime::output_view)
                .unwrap_or(stable_gaussian_view);
            let authored_base_rows_buffer = motion_controller_runtime
                .as_ref()
                .map(GpuMotionControllerRuntime::output_base_rows_buffer)
                .unwrap_or(&self.empty_authored_base_rows_buffer);
            let bind_group = create_motion_field_bind_group(
                device,
                &self.motion_field_bind_group_layout,
                &field,
                stable_gaussian_view,
                authored_gaussian_view,
                authored_base_rows_buffer,
            );
            self.motion_field = Some(field);
            self.motion_controller_runtime = motion_controller_runtime;
            self.motion_field_bind_group = Some(bind_group);
            if self.authored_registry_update.is_none() {
                self.authored_registry_key = None;
                self.authored_output_count = 0;
            }
            self.authored_dispatch_key = None;
            self.authored_runtime_error = None;
            return Ok(cache.layout().estimated_gpu_bytes());
        }
        let field = self
            .motion_field
            .as_ref()
            .expect("dimension check establishes a motion field");
        let mut uploaded_bytes = 0_u64;
        for &dirty in dirty_regions {
            uploaded_bytes = uploaded_bytes.saturating_add(
                field
                    .upload_dirty_compiled(queue, cache, compiled, dirty)
                    .map_err(|error| error.to_string())?
                    .uploaded_bytes,
            );
        }
        Ok(uploaded_bytes)
    }

    pub fn sync_motion_preview(&mut self, device: &wgpu::Device, cache: &MotionFieldCache) {
        if self.motion_field.is_some() {
            return;
        }
        let field = GpuMotionField::new_preview(device, cache.layout());
        let stable_gaussian_view = self
            .motion_runtime
            .as_ref()
            .map(|runtime| &runtime.base_texture().view)
            .unwrap_or(&self.gaussian_texture.view);
        let bind_group = create_motion_field_bind_group(
            device,
            &self.motion_field_bind_group_layout,
            &field,
            stable_gaussian_view,
            stable_gaussian_view,
            &self.empty_authored_base_rows_buffer,
        );
        self.motion_field = Some(field);
        self.motion_field_bind_group = Some(bind_group);
    }

    pub fn update_motion_field_uniform(
        &mut self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        field_data_active: bool,
        overlay: MotionOverlayChannel,
        palette_count: u32,
        preview: Option<MotionBrushPreview>,
    ) {
        let Some(field) = self.motion_field.as_mut() else {
            self.motion_field_render_active = false;
            return;
        };
        field.update_uniform(
            queue,
            MotionFieldUniform::new(
                cache.layout(),
                cache.storage_offset(),
                field_data_active,
                overlay,
                palette_count,
            )
            .with_preview(preview),
        );
        self.motion_field_render_active =
            (field_data_active || preview.is_some()) && self.motion_field_bind_group.is_some();
    }

    pub fn disable_motion_field_rendering(&mut self) {
        self.motion_field_render_active = false;
    }

    pub fn update_authored_motion(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sort_data: Option<&SortData>,
        scene_data: Option<&SceneData>,
        compiled: &CompiledMotionField,
        local_frames: &[MotionControllerFrame],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<Option<AuthoredMotionDispatch>, String> {
        let (Some(motion), Some(field), Some(controller), Some(sort_data), Some(scene_data)) = (
            self.motion_runtime.as_ref(),
            self.motion_field.as_ref(),
            self.motion_controller_runtime.as_mut(),
            sort_data,
            scene_data,
        ) else {
            self.authored_dispatch_key = None;
            return Ok(None);
        };
        let registry_key = AuthoredRegistryKey {
            request_revision: sort_data.authored.request_revision,
            registry_revision: sort_data.authored.registry_revision,
            scene_id: scene_data.scene_id,
        };
        let mut bindings_changed = false;
        let mut registry_installed = false;
        if self.authored_registry_key != Some(registry_key) {
            let update = sort_data.authored.update.as_ref().ok_or_else(|| {
                "authored sort references a registry revision that is not installed".to_string()
            })?;
            if update.request_revision != registry_key.request_revision
                || update.registry_revision != registry_key.registry_revision
                || update.scene_id != registry_key.scene_id
            {
                return Err("authored registry update does not match the accepted sort".to_string());
            }
            bindings_changed = controller.update_registry(device, queue, motion, field, update)?;
            self.authored_output_count = update.inputs.len();
            self.authored_registry_update = Some(update.clone());
            self.authored_registry_key = Some(registry_key);
            self.authored_dispatch_key = None;
            registry_installed = true;
        }
        if bindings_changed {
            self.motion_field_bind_group = Some(create_motion_field_bind_group(
                device,
                &self.motion_field_bind_group_layout,
                field,
                &motion.base_texture().view,
                controller.output_view(),
                controller.output_base_rows_buffer(),
            ));
        }
        let tagged_occurrences = sort_data.authored.tagged_occurrences;
        let registry_occurrences = self.authored_output_count;
        let affected_draws = sort_data
            .authored
            .authored_draws
            .iter()
            .filter(|&&authored| authored)
            .count();
        if registry_occurrences == 0 {
            self.authored_dispatch_key = None;
            return Ok(Some(AuthoredMotionDispatch {
                registry_occurrences,
                tagged_occurrences,
                affected_draws,
                membership_request_revision: registry_key.request_revision,
                registry_revision: registry_key.registry_revision,
                worker_tag_ms: sort_data.authored.tag_time_ms,
                registry_installed,
                ..AuthoredMotionDispatch::default()
            }));
        }
        let used_frames = local_frames
            .iter()
            .copied()
            .filter(|frame| compiled.slot_is_used(frame.slot))
            .collect::<Vec<_>>();
        let local_frames = used_frames.as_slice();
        let dispatch_key = AuthoredDispatchKey::new(
            motion.global_controller_frame(),
            local_frames,
            motion.motion_enabled(),
            motion.preview_channel_mask(),
            registry_key.registry_revision,
            compiled.revision(),
        );
        if self.authored_dispatch_key.as_ref() == Some(&dispatch_key) && !bindings_changed {
            return Ok(Some(AuthoredMotionDispatch {
                registry_occurrences,
                tagged_occurrences,
                affected_draws,
                membership_request_revision: registry_key.request_revision,
                registry_revision: registry_key.registry_revision,
                worker_tag_ms: sort_data.authored.tag_time_ms,
                registry_installed,
                ..AuthoredMotionDispatch::default()
            }));
        }
        let mut work = controller.dispatch(
            queue,
            encoder,
            motion,
            motion.global_controller_frame(),
            local_frames,
            motion.motion_enabled(),
            motion.preview_channel_mask(),
            timestamp_writes,
        )?;
        work.registry_occurrences = registry_occurrences;
        work.tagged_occurrences = tagged_occurrences;
        work.affected_draws = affected_draws;
        work.membership_request_revision = registry_key.request_revision;
        work.registry_revision = registry_key.registry_revision;
        work.worker_tag_ms = sort_data.authored.tag_time_ms;
        work.registry_installed = registry_installed || work.registry_upload_bytes > 0;
        self.authored_dispatch_key = Some(dispatch_key);
        self.authored_runtime_error = None;
        Ok(Some(work))
    }

    pub fn disable_authored_motion(&mut self, error: String) {
        self.authored_dispatch_key = None;
        self.authored_runtime_error = Some(error);
    }

    pub fn clear_authored_motion(&mut self) {
        self.authored_dispatch_key = None;
        self.authored_runtime_error = None;
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        user_data: &UserData,
        render_data: &RenderData,
    ) {
        self.user_data = user_data.clone();

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

    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
        depth_prepared: bool,
        water_hits: Option<&wgpu::BindGroup>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> FrameCounters {
        let scene_data = render_data.cur_scene_data.as_ref().unwrap();
        let sort_data = render_data.cur_sort_data.as_ref().unwrap();
        let render_config = &render_data.render_config;
        let profiling = render_data.profiler_enabled;

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

        let mut counters = if profiling {
            FrameCounters::for_render(
                scene_data.splat_count,
                scene_data.blending_splat_count,
                (std::mem::size_of::<CameraUniforms>() + std::mem::size_of::<SceneUniforms>())
                    as u64,
            )
        } else {
            FrameCounters::default()
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
            )),
        );

        let tile_uniforms_block_size =
            usize::max(256, std::mem::size_of::<TileUniforms>() as usize);
        let view_proj = camera.view_proj();
        let mut buffer_offset: BufferAddress = 0;
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
        let mut using_motion_pipeline = false;
        let water_active = water_hits.is_some()
            && render_config.water.is_active(self.user_data.surface_type)
            && crate::water::bounds(&self.user_data, render_data).is_some();
        render_pass.set_pipeline(if water_active {
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
        for i in 0..sort_data.render_data_vec.len() {
            let tile_instance = &sort_data.tile_instance_vec[i];
            let (render_data_key, option_render_data_value) = &sort_data.render_data_vec[i];
            let tid = tile_instance.tid;

            // viewport culling (only for non-merged tiles)
            if render_data_key.tid.len() == 1 {
                let mut pos2d = vec3(f32::MAX, f32::MAX, -f32::MAX);
                for ci in 0..4 {
                    let corner = view_proj
                        * tile_instance.corner_data.as_ref().unwrap()[ci]
                            .0
                            .extend(1.0);
                    let corner = corner.truncate() / corner.w;
                    if corner.x.abs() < pos2d.x {
                        pos2d.x = corner.x.abs();
                    }
                    if corner.y.abs() < pos2d.y {
                        pos2d.y = corner.y.abs();
                    }
                    if corner.z > pos2d.z {
                        pos2d.z = corner.z;
                    }
                }
                let clip = render_config.culling_dist;
                if pos2d.z < -clip || pos2d.x > clip || pos2d.y > clip {
                    continue;
                }
            }
            if !render_config.lod_enable[tid.0] {
                continue;
            }

            let mut tile_uniforms =
                TileUniforms::from_tile(tile_instance, option_render_data_value);
            let authored_draw = sort_data
                .authored
                .authored_draws
                .get(i)
                .copied()
                .unwrap_or(false);
            tile_uniforms.authored_tag_mode = u32::from(authored_draw && authored_runtime_ready);
            let requires_motion_pipeline = motion_bind_available
                && draw_requires_motion_pipeline(
                    authored_draw,
                    if motion_diagnostics_active {
                        motion_overlay
                    } else {
                        MotionOverlayChannel::Off
                    },
                    motion_diagnostics_active
                        && preview_intersects_draw(
                            motion_preview,
                            tile_instance,
                            option_render_data_value,
                            &self.user_data,
                            scene_data,
                        ),
                );
            if requires_motion_pipeline != using_motion_pipeline {
                using_motion_pipeline = requires_motion_pipeline;
                render_pass.set_pipeline(match (using_motion_pipeline, water_active) {
                    (true, true) => &self.water_motion_field_render_pipeline,
                    (true, false) => &self.motion_field_render_pipeline,
                    (false, true) => &self.water_render_pipeline,
                    (false, false) => &self.render_pipeline,
                });
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
            if let Some(render_data_value) = option_render_data_value {
                tile_uniforms.single_draw = 1;
                tile_uniforms.single_lod_id = render_data_value.single_lod_id;
            }
            if render_config.debug_log && render_data_key.tid.len() >= 9 {
                log! {"{:?}", tile_instance};
                log! {"{:?}", render_data_key};
                log! {"{:?}", option_render_data_value};
            }

            queue.write_buffer(
                &self.tile_uniforms_buffer,
                (i * tile_uniforms_block_size) as BufferAddress,
                bytemuck::bytes_of(&tile_uniforms),
            );
            let mut draw_upload_bytes = if profiling {
                std::mem::size_of::<TileUniforms>() as u64
            } else {
                0
            };

            let splat_count: u32;
            if let Some(render_data_value) = option_render_data_value {
                splat_count = render_data_value.splat_count as u32;
                let size_byte = splat_count as u64 * 4;

                queue.write_buffer(
                    &self.gs_index_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_index.as_slice()),
                );
                draw_upload_bytes = draw_upload_bytes.saturating_add(size_byte);
                render_pass.set_vertex_buffer(
                    1,
                    self.gs_index_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                queue.write_buffer(
                    &self.map_id_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_map_id.as_slice()),
                );
                draw_upload_bytes = draw_upload_bytes.saturating_add(size_byte);
                render_pass.set_vertex_buffer(
                    2,
                    self.map_id_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                if render_data_value.single_lod_id == -1 {
                    queue.write_buffer(
                        &self.lod_id_buffer,
                        buffer_offset,
                        bytemuck::cast_slice(
                            render_data_value.gs_lod_id.as_ref().unwrap().as_slice(),
                        ),
                    );
                    draw_upload_bytes = draw_upload_bytes.saturating_add(size_byte);
                    render_pass.set_vertex_buffer(
                        3,
                        self.lod_id_buffer
                            .slice(buffer_offset..buffer_offset + size_byte),
                    );
                } else {
                    render_pass.set_vertex_buffer(3, self.lod_id_buffer.slice(..));
                }

                buffer_offset += size_byte;
            } else {
                let base_data: &BufferDataValue;
                if let TileTransitionStatus::Changing(to_lower) = tile_instance.transition_status {
                    if to_lower {
                        base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                    } else {
                        base_data = &self.buffer_base_data[tid.0 - 1][tid.1][tile_instance.view_id];
                    }
                } else {
                    base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                }
                splat_count = base_data.splat_count;

                render_pass.set_vertex_buffer(1, base_data.gs_index_buffer.slice(..));
                render_pass.set_vertex_buffer(2, self.map_id_buffer.slice(..));
                render_pass
                    .set_vertex_buffer(3, base_data.lod_id_buffer.as_ref().unwrap().slice(..));
            }

            render_pass.set_bind_group(
                1,
                &self.tile_bind_group,
                &[(i * tile_uniforms_block_size) as u32],
            );

            render_pass.draw(0..6, 0..splat_count);
            if profiling {
                counters.record_draw(
                    splat_count as usize,
                    draw_upload_bytes,
                    active_members(render_data_key),
                );
            }
        }
        drop(render_pass);
        counters
    }
}

fn strip_motion_field_shader_blocks(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut skipping = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("// MOTION_FIELD_BEGIN:") {
            assert!(!skipping, "motion-field shader blocks must not be nested");
            skipping = true;
            continue;
        }
        if trimmed.starts_with("// MOTION_FIELD_END:") {
            assert!(skipping, "motion-field shader block end must have a begin");
            skipping = false;
            continue;
        }
        if !skipping {
            output.push_str(line);
            output.push('\n');
        }
    }
    assert!(!skipping, "motion-field shader block must be closed");
    output
}

/// Dry pipelines are compiled without water varyings, cut moments, solver or depth output.
fn gs_shader_source(water: bool) -> String {
    if water {
        return [
            include_str!("camera.wgsl"),
            &crate::water_hits::shader_source(3),
            include_str!("gswt.wgsl"),
        ]
        .concat();
    }
    let mut body = String::new();
    let mut skipping = false;
    for line in include_str!("gswt.wgsl").lines() {
        if line.trim_start().starts_with("// WATER_BEGIN:") {
            assert!(!skipping);
            skipping = true;
            continue;
        }
        if line.trim_start().starts_with("// WATER_END:") {
            assert!(skipping);
            skipping = false;
            continue;
        }
        if !skipping {
            body.push_str(line);
            body.push('\n');
        }
    }
    assert!(!skipping, "water shader block must be closed");
    [
        include_str!("camera.wgsl"),
        &body,
        include_str!("gswt_dry.wgsl"),
    ]
    .concat()
}

fn create_render_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    let alpha_blend = Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[
                Vertex2D::desc(),
                wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![1 => Uint32],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![2 => Uint32],
                },
                wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![3 => Uint32],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: alpha_blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}

fn create_motion_field_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    field: &GpuMotionField,
    stable_gaussian_view: &wgpu::TextureView,
    authored_gaussian_view: &wgpu::TextureView,
    authored_base_rows_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("motion authoring field bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(field.continuous_view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(field.assignment_view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: field.uniform_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(stable_gaussian_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(authored_gaussian_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: authored_base_rows_buffer.as_entire_binding(),
            },
        ],
    })
}

#[derive(Default)]
struct MotionGraphBuildOutcome {
    graph: Option<Arc<MotionGraph>>,
    error: Option<String>,
}

fn build_motion_graph_outcome(motion: &MergedMotion) -> MotionGraphBuildOutcome {
    match build_motion_graph(motion, MotionGraphSettings::default()) {
        Ok(graph) => MotionGraphBuildOutcome {
            graph: Some(Arc::new(graph)),
            error: None,
        },
        Err(error) => MotionGraphBuildOutcome {
            graph: None,
            error: Some(error.to_string()),
        },
    }
}

fn active_members(render_data_key: &RenderDataKey) -> Vec<(usize, usize)> {
    let mut members = Vec::with_capacity(render_data_key.tid.len() * 2);
    for (index, &(lod, tile)) in render_data_key.tid.iter().enumerate() {
        members.push((tile, lod));
        match render_data_key.transition_status.get(index) {
            Some(TileTransitionStatusHash::Changing(true)) => {
                if let Some(adjacent_lod) = lod.checked_add(1) {
                    members.push((tile, adjacent_lod));
                }
            }
            Some(TileTransitionStatusHash::Changing(false)) => {
                if let Some(adjacent_lod) = lod.checked_sub(1) {
                    members.push((tile, adjacent_lod));
                }
            }
            _ => {}
        }
    }
    members
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AuthoredRegistryKey {
    request_revision: u64,
    registry_revision: u64,
    scene_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AuthoredDispatchKey {
    global: MotionControllerFrameKey,
    locals: Vec<MotionControllerFrameKey>,
    motion_enabled: bool,
    preview_channel_mask: u32,
    registry_revision: u64,
    field_revision: u64,
}

impl AuthoredDispatchKey {
    fn new(
        global: MotionControllerFrame,
        locals: &[MotionControllerFrame],
        motion_enabled: bool,
        preview_channel_mask: u32,
        registry_revision: u64,
        field_revision: u64,
    ) -> Self {
        Self {
            global: MotionControllerFrameKey::from(global),
            locals: locals
                .iter()
                .copied()
                .map(MotionControllerFrameKey::from)
                .collect(),
            motion_enabled,
            preview_channel_mask,
            registry_revision,
            field_revision,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MotionControllerFrameKey {
    slot: u8,
    valid: bool,
    sample: [u32; 4],
    gains: [u32; 6],
}

impl From<MotionControllerFrame> for MotionControllerFrameKey {
    fn from(frame: MotionControllerFrame) -> Self {
        let sample = match frame.sample {
            TimelineSample::Source { u } => [0, u.to_bits(), 0, 0],
            TimelineSample::Blend {
                from_u,
                to_u,
                alpha,
            } => [1, from_u.to_bits(), to_u.to_bits(), alpha.to_bits()],
        };
        Self {
            slot: frame.slot,
            valid: frame.valid,
            sample,
            gains: [
                frame.gains.translation[0].to_bits(),
                frame.gains.translation[1].to_bits(),
                frame.gains.translation[2].to_bits(),
                frame.gains.rotation.to_bits(),
                frame.gains.scale.to_bits(),
                frame.gains.master.to_bits(),
            ],
        }
    }
}

fn map_coord_and_offset(
    map_id: u32,
    user_data: &UserData,
    scene_data: &SceneData,
) -> Result<([u32; 2], [f32; 3]), String> {
    let map_height = u32::try_from(user_data.tile_map_wh.y)
        .map_err(|_| "tile-map height exceeds u32".to_string())?;
    let map_width = u32::try_from(user_data.tile_map_wh.x)
        .map_err(|_| "tile-map width exceeds u32".to_string())?;
    if map_height == 0 || map_width == 0 {
        return Err("tile-map dimensions must be positive".to_string());
    }
    let coord = [map_id / map_height, map_id % map_height];
    if coord[0] >= map_width {
        return Err("merged draw map ID exceeds the active tile map".to_string());
    }
    let world_tile = [
        i64::from(coord[0]) - user_data.tile_map_half_wh.x as i64
            + i64::from(scene_data.center_coord.x),
        i64::from(coord[1]) - user_data.tile_map_half_wh.y as i64
            + i64::from(scene_data.center_coord.y),
    ];
    Ok((
        coord,
        [
            world_tile[0] as f32 * user_data.tile_width,
            world_tile[1] as f32 * user_data.tile_width,
            0.0,
        ],
    ))
}

fn draw_requires_motion_pipeline(
    authored_draw: bool,
    overlay: MotionOverlayChannel,
    preview_intersects: bool,
) -> bool {
    authored_draw || overlay != MotionOverlayChannel::Off || preview_intersects
}

fn circle_intersects_tile(preview: MotionBrushPreview, tile_origin: [f32; 2], width: f32) -> bool {
    let closest = [
        preview.center[0].clamp(tile_origin[0], tile_origin[0] + width),
        preview.center[1].clamp(tile_origin[1], tile_origin[1] + width),
    ];
    let delta = [
        preview.center[0] - closest[0],
        preview.center[1] - closest[1],
    ];
    delta[0] * delta[0] + delta[1] * delta[1] <= preview.radius * preview.radius
}

fn preview_intersects_draw(
    preview: Option<MotionBrushPreview>,
    tile: &TileInstance,
    render_value: &Option<RenderDataValue>,
    user_data: &UserData,
    scene_data: &SceneData,
) -> bool {
    let Some(preview) = preview else {
        return false;
    };
    if let Some(value) = render_value {
        value.merge_from_vec.iter().any(|&map_id| {
            u32::try_from(map_id)
                .ok()
                .and_then(|id| map_coord_and_offset(id, user_data, scene_data).ok())
                .map(|(_, offset)| {
                    circle_intersects_tile(preview, [offset[0], offset[1]], user_data.tile_width)
                })
                .unwrap_or(true)
        })
    } else {
        circle_intersects_tile(
            preview,
            [tile.tile_offset.x, tile.tile_offset.y],
            user_data.tile_width,
        )
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
    _pad0: [u32; 2],
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

    fn from_data(user_data: &UserData, scene_data: &SceneData, render_data: &RenderData) -> Self {
        let render_config = &render_data.render_config;
        let water_bounds = crate::water::bounds(user_data, render_data);
        let water_active =
            render_config.water.is_active(user_data.surface_type) && water_bounds.is_some();
        Self {
            water_level: [
                render_config.water.height,
                u32::from(water_active) as f32,
                0.0,
                0.0,
            ],
            water_bounds: water_bounds.unwrap_or([0.0; 4]),
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
            _pad0: [0; 2],
            transition_dist_vec: Self::expand_to_array::<16, f32>(
                &user_data.lod_transition_dist,
                0.0,
            ),
            scene_scale: [
                render_config.scene_scale.x,
                render_config.scene_scale.y,
                render_config.scene_scale.z,
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
        let authored = gs_shader_source(true);
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
        let source = strip_motion_field_shader_blocks(&gs_shader_source(false));
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
        let source = gs_shader_source(true);
        assert!(
            !source.contains("fn water_surface_hit("),
            "GS must not compile the iterative water solver"
        );
        assert!(
            source.contains("load_water_hit("),
            "GS must consume the current frame's shared hit texture"
        );
    }
}
