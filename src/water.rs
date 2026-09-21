//! Opaque world-space water with shared geometric waves and sky reflection.
use crate::{
    camera::{Camera, CameraUniforms},
    structure::{RenderData, SurfaceType, UserData},
    texture::Texture,
    water_environment::WaterEnvironment,
    water_hits::WaterHits,
};

#[derive(Clone, Debug)]
pub struct WaterSettings {
    pub enabled: bool,
    /// Final renderer world Z, independent of camera height and scene Z scale.
    pub height: f32,
    pub color: [f32; 3],
    /// Maximum displacement from the mean level; zero preserves the flat path.
    pub amplitude: f32,
    pub wavelength: f32,
    pub speed: f32,
    pub playing: bool,
    pub varied_waves: bool,
    pub reflection_strength: f32,
    pub roughness: f32,
    pub ripple_strength: f32,
    /// World-space size, independent of the geometric wavelength and tile window.
    pub ripple_scale: f32,
    // Integrate speed in f64, then wrap each phase before sending it to the GPU.
    phase_time: f64,
}
impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            height: 0.0,
            color: [0.04, 0.24, 0.38],
            amplitude: 0.08,
            wavelength: 4.0,
            speed: 1.0,
            playing: true,
            varied_waves: true,
            reflection_strength: 1.0,
            roughness: 0.18,
            ripple_strength: 0.22,
            ripple_scale: 0.35,
            phase_time: 0.0,
        }
    }
}
impl WaterSettings {
    pub fn advance(&mut self, seconds: f64) {
        if self.enabled
            && self.playing
            && seconds.is_finite()
            && seconds > 0.0
            && self.speed.is_finite()
        {
            let next = self.phase_time + seconds * self.speed.clamp(0.0, 5.0) as f64;
            if next.is_finite() {
                self.phase_time = next;
            }
        }
    }
    pub fn phases(&self) -> [f32; 4] {
        [1.0, 1.31, 1.73, 2.117]
            .map(|frequency| (self.phase_time * frequency).rem_euclid(std::f64::consts::TAU) as f32)
    }
    pub fn waves(&self) -> [f32; 4] {
        if !self.amplitude.is_finite() || !self.wavelength.is_finite() {
            return [0.0; 4];
        }
        let wavelength = self.wavelength.clamp(0.1, 1000.0);
        // Bound the summed slope for the calm-water model and its intersection solver.
        [
            self.amplitude.clamp(0.0, wavelength * 0.035),
            std::f32::consts::TAU / wavelength,
            if self.varied_waves { 1.0 } else { 0.0 },
            0.0,
        ]
    }

    pub fn detail_offsets(&self) -> [f32; 4] {
        [0.07, -0.093, 0.053, 0.041].map(|speed| (self.phase_time * speed).rem_euclid(256.0) as f32)
    }

    pub fn is_active(&self, surface: SurfaceType) -> bool {
        self.enabled
            && surface != SurfaceType::Sphere
            && self.height.is_finite()
            && self.color.iter().all(|channel| channel.is_finite())
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    camera: CameraUniforms,
    bounds: [f32; 4],
    color: [f32; 4],
    level: [f32; 4],
    waves: [f32; 4],
    phases: [f32; 4],
    material: [f32; 4],
    detail_offsets: [f32; 4],
}

/// Uses the committed tile window, including a one-tile border around its footprint.
/// Moving the cache changes only coverage, never the world-space water level.
pub(crate) fn bounds(user: &UserData, data: &RenderData) -> Option<[f32; 4]> {
    let center = data.cur_scene_data.as_ref()?.center_coord;
    let scale = data.render_config.scene_scale;
    let cx = center.x as f32 * user.tile_width * scale.x;
    let cy = center.y as f32 * user.tile_width * scale.y;
    let hx = (user.tile_map_half_wh.x as f32 + 1.5) * user.tile_width * scale.x.abs();
    let hy = (user.tile_map_half_wh.y as f32 + 1.5) * user.tile_width * scale.y.abs();
    let result = [cx - hx, cy - hy, cx + hx, cy + hy];
    (result.iter().all(|value| value.is_finite()) && hx > 0.0 && hy > 0.0).then_some(result)
}

pub struct WaterRenderer {
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    empty_environment: WaterEnvironment,
    intersect_pipeline: wgpu::ComputePipeline,
    hits: Option<WaterHits>,
}
impl WaterRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water pipeline layout"),
            bind_group_layouts: &[
                &layout,
                &WaterEnvironment::layout(device),
                &WaterHits::read_layout(device),
            ],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water shader"),
            source: wgpu::ShaderSource::Wgsl(
                shader_source(
                    &[
                        include_str!("water_uniforms.wgsl"),
                        &crate::water_hits::shader_source(2),
                        include_str!("water.wgsl"),
                    ]
                    .concat(),
                )
                .into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let intersect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water once-per-pixel intersection"),
            source: wgpu::ShaderSource::Wgsl(
                shader_source(
                    &[
                        include_str!("water_uniforms.wgsl"),
                        include_str!("water_intersect.wgsl"),
                    ]
                    .concat(),
                )
                .into(),
            ),
        });
        let intersect_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Water intersection layout"),
            bind_group_layouts: &[&layout, &WaterHits::write_layout(device)],
            push_constant_ranges: &[],
        });
        let intersect_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Water intersection"),
            layout: Some(&intersect_layout),
            module: &intersect_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            intersect_pipeline,
            hits: None,
            pipeline,
            uniforms,
            bind_group,
            empty_environment: WaterEnvironment::empty(device),
        }
    }

    pub(crate) fn hits(&self) -> Option<&WaterHits> {
        self.hits.as_ref()
    }

    /// Returns whether depth was prepared; the following GS pass must load it.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        user: &UserData,
        data: &RenderData,
        environment: Option<&WaterEnvironment>,
        timestamps: crate::profiler::WaterPassTimestamps<'_>,
    ) -> bool {
        let settings = &data.render_config.water;
        if !settings.is_active(user.surface_type) {
            return false;
        }
        let Some(bounds) = bounds(user, data) else {
            return false;
        };
        let Some(depth) = data.depth_texture.as_ref() else {
            return false;
        };
        let uniforms = Uniforms {
            camera: CameraUniforms::from_camera(camera),
            bounds,
            color: [
                settings.color[0].clamp(0.0, 1.0),
                settings.color[1].clamp(0.0, 1.0),
                settings.color[2].clamp(0.0, 1.0),
                1.0,
            ],
            level: [settings.height, 0.0, 0.0, 0.0],
            waves: settings.waves(),
            phases: settings.phases(),
            material: [
                if environment.is_some() {
                    finite_clamp(settings.reflection_strength, 0.0, 1.0, 0.0)
                } else {
                    0.0
                },
                finite_clamp(settings.roughness, 0.0, 1.0, 0.18),
                finite_clamp(settings.ripple_strength, 0.0, 1.0, 0.0),
                finite_clamp(settings.ripple_scale, 0.02, 100.0, 0.5),
            ],
            detail_offsets: settings.detail_offsets(),
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let size = [camera.viewport().width, camera.viewport().height];
        if size.contains(&0) {
            return false;
        }
        if self.hits.as_ref().is_none_or(|h| h.size != size) {
            self.hits = Some(WaterHits::new(device, size));
        }
        let hits = self.hits.as_ref().unwrap();
        // Independent of proxy depth: GS needs the same geometric intersection
        // even where the water material is hidden by opaque terrain.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water intersection pass"),
                timestamp_writes: timestamps.intersection,
            });
            pass.set_pipeline(&self.intersect_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &hits.write, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Water pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(wgpu::Operations {
                    load: if data.use_proxy {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timestamps.shading,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(
            1,
            &environment.unwrap_or(&self.empty_environment).bind_group,
            &[],
        );
        pass.set_bind_group(2, &hits.read, &[]);
        pass.draw(0..6, 0..1);
        true
    }
}

/// Compose the same surface math into both pipelines, outside the frame loop.
pub(crate) fn shader_source(body: &str) -> String {
    [
        include_str!("camera.wgsl"),
        include_str!("water_common.wgsl"),
        "\n",
        body,
    ]
    .concat()
}

fn finite_clamp(value: f32, low: f32, high: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(low, high)
    } else {
        fallback
    }
}
