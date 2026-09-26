//! Shader variants, bind-group layouts, and Gaussian render pipeline construction.
use crate::{structure::Vertex2D, texture::Texture};

pub(super) struct PipelineSet {
    pub(super) render_pipeline: wgpu::RenderPipeline,
    pub(super) motion_field_render_pipeline: wgpu::RenderPipeline,
    pub(super) water_render_pipeline: wgpu::RenderPipeline,
    pub(super) water_motion_field_render_pipeline: wgpu::RenderPipeline,
    pub(super) underwater_render_pipeline: wgpu::RenderPipeline,
    pub(super) underwater_motion_field_render_pipeline: wgpu::RenderPipeline,
    pub(super) empty_water_base_group: wgpu::BindGroup,
    pub(super) scene_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) tile_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) motion_field_bind_group_layout: wgpu::BindGroupLayout,
}

impl PipelineSet {
    pub(super) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
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

        let motion_field_shader_source = gs_shader_source(false, false);
        let base_shader_source = strip_motion_field_shader_blocks(&motion_field_shader_source);
        let base_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Base GSWT shader"),
            source: wgpu::ShaderSource::Wgsl(base_shader_source.into()),
        });
        let motion_field_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spatial motion authoring GSWT shader"),
            source: wgpu::ShaderSource::Wgsl(motion_field_shader_source.into()),
        });

        let wet_source = gs_shader_source(true, false);
        let wet_base = strip_motion_field_shader_blocks(&wet_source);
        let wet_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water-clipped GS shader"),
            source: wgpu::ShaderSource::Wgsl(wet_base.into()),
        });
        let wet_authored_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water-clipped authored shader"),
            source: wgpu::ShaderSource::Wgsl(wet_source.into()),
        });
        let underwater_source = gs_shader_source(true, true);
        let underwater_base = strip_motion_field_shader_blocks(&underwater_source);
        let underwater_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater GS shader"),
            source: wgpu::ShaderSource::Wgsl(underwater_base.into()),
        });
        let underwater_authored_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Underwater authored GS shader"),
                source: wgpu::ShaderSource::Wgsl(underwater_source.into()),
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
            format,
            &base_render_pipeline_layout,
            &base_shader,
            "Base render pipeline",
            wgpu::CompareFunction::Less,
        );
        let motion_field_render_pipeline = create_render_pipeline(
            device,
            format,
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
            format,
            &water_base_layout,
            &wet_shader,
            "Water-clipped GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );
        let water_motion_field_render_pipeline = create_render_pipeline(
            device,
            format,
            &water_authored_layout,
            &wet_authored_shader,
            "Water-clipped authored GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );
        let underwater_render_pipeline = create_render_pipeline(
            device,
            format,
            &water_base_layout,
            &underwater_shader,
            "Underwater GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );
        let underwater_motion_field_render_pipeline = create_render_pipeline(
            device,
            format,
            &water_authored_layout,
            &underwater_authored_shader,
            "Underwater authored GS pipeline",
            wgpu::CompareFunction::LessEqual,
        );

        Self {
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
        }
    }
}

pub(super) fn strip_motion_field_shader_blocks(source: &str) -> String {
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
pub(super) fn gs_shader_source(water: bool, underwater: bool) -> String {
    assert!(water || !underwater, "underwater rendering requires water");
    let mut body = if water {
        include_str!("../gswt.wgsl").to_owned()
    } else {
        strip_tagged_shader_blocks(include_str!("../gswt.wgsl"), "WATER")
    };
    if !underwater {
        body = strip_tagged_shader_blocks(&body, "UNDERWATER");
    }
    if water {
        [
            include_str!("../camera.wgsl"),
            include_str!("../cubed_sphere.wgsl"),
            &crate::water_hits::shader_source(3),
            &if underwater {
                crate::underwater::sampling_shader(3)
            } else {
                String::new()
            },
            &body,
        ]
        .concat()
    } else {
        [
            include_str!("../camera.wgsl"),
            include_str!("../cubed_sphere.wgsl"),
            &body,
            include_str!("../gswt_dry.wgsl"),
        ]
        .concat()
    }
}

fn strip_tagged_shader_blocks(source: &str, tag: &str) -> String {
    let mut body = String::new();
    let mut skipping = false;
    let begin = format!("// {tag}_BEGIN:");
    let end = format!("// {tag}_END:");
    for line in source.lines() {
        if line.trim_start().starts_with(&begin) {
            assert!(!skipping);
            skipping = true;
            continue;
        }
        if line.trim_start().starts_with(&end) {
            assert!(skipping);
            skipping = false;
            continue;
        }
        if !skipping {
            body.push_str(line);
            body.push('\n');
        }
    }
    assert!(!skipping, "{tag} shader block must be closed");
    body
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
