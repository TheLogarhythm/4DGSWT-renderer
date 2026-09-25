//! Opaque world-space water with shared geometric waves and sky reflection.
use crate::{
    camera::{Camera, CameraUniforms},
    structure::{RenderData, SurfaceType, UserData},
    texture::Texture,
    water_environment::WaterEnvironment,
    water_hits::{WaterHits, WaterResources},
};

#[derive(Clone, Debug)]
pub struct UnderwaterSettings {
    pub enabled: bool,
    pub color: [f32; 3],
    /// Distance beyond the clear zone at which green scene contrast falls to 50%.
    pub visibility: f32,
    /// Preserve nearby scene color before distance fog begins (world units).
    pub clear_distance: f32,
    pub sunlight: f32,
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    pub light_shafts: bool,
    pub shaft_strength: f32,
    pub shaft_scale: f32,
    pub caustics: bool,
    pub caustic_strength: f32,
    /// Approximate spacing of the artificial bright cells in world units.
    pub caustic_scale: f32,
    /// Motion multiplier, independent of wave speed; zero freezes the pattern.
    pub caustic_speed: f32,
}

impl Default for UnderwaterSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            color: [0.16, 0.30, 0.34],
            visibility: 15.0,
            clear_distance: 3.0,
            sunlight: 0.6,
            sun_azimuth: 120.0,
            sun_elevation: 55.0,
            light_shafts: true,
            shaft_strength: 0.7,
            shaft_scale: 3.0,
            caustics: false,
            caustic_strength: 0.65,
            caustic_scale: 1.5,
            caustic_speed: 0.25,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnderwaterFrame {
    pub color: [f32; 3],
    pub extinction: [f32; 3],
    pub strength: f32,
    pub clear_distance: f32,
}

#[cfg(test)]
impl UnderwaterFrame {
    fn transmission(self, distance: f32) -> [f32; 3] {
        self.extinction
            .map(|coefficient| (-coefficient * (distance - self.clear_distance).max(0.0)).exp())
    }
}

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
    pub underwater: UnderwaterSettings,
    // Integrate speed in f64, then wrap each phase before sending it to the GPU.
    phase_time: f64,
    // Separate accumulated time prevents speed edits from moving the pattern.
    caustic_time: f64,
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
            underwater: UnderwaterSettings::default(),
            phase_time: 0.0,
            caustic_time: 0.0,
        }
    }
}
impl WaterSettings {
    pub(crate) fn underwater_frame(
        &self,
        camera: [f32; 3],
        bounds: Option<[f32; 4]>,
    ) -> Option<UnderwaterFrame> {
        if !self.enabled || !self.underwater.enabled || camera.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let [min_x, min_y, max_x, max_y] = bounds?;
        if camera[0] < min_x || camera[0] > max_x || camera[1] < min_y || camera[1] > max_y {
            return None;
        }
        let level = self.height + self.wave_offset([camera[0], camera[1]]);
        if !level.is_finite() {
            return None;
        }
        let transition = self.waves()[0].max(0.05);
        let t = ((level - camera[2] + transition) / (2.0 * transition)).clamp(0.0, 1.0);
        let strength = t * t * (3.0 - 2.0 * t);
        if strength <= 0.0 {
            return None;
        }
        let visibility = finite_clamp(self.underwater.visibility, 0.5, 1000.0, 15.0);
        // A half-contrast distance is much gentler than losing 95% by this distance.
        // Keep channel differences small: the splats already contain baked lighting/color.
        let base = std::f32::consts::LN_2 / visibility * strength;
        Some(UnderwaterFrame {
            color: self
                .underwater
                .color
                .map(|c| finite_clamp(c, 0.0, 1.0, 0.0)),
            extinction: [1.1 * base, base, 0.95 * base],
            strength,
            clear_distance: finite_clamp(self.underwater.clear_distance, 0.0, 1000.0, 3.0),
        })
    }

    fn wave_offset(&self, xy: [f32; 2]) -> f32 {
        let waves = self.waves();
        let phases = self.phases();
        let varied = self.varied_waves;
        let natural = [
            [0.946, 0.324, 0.830, 0.32],
            [0.544, -0.839, 1.271, 0.28],
            [-0.771, 0.637, 1.913, 0.23],
            [-0.217, -0.976, 2.731, 0.17],
        ];
        let regular = [
            [1.0, 0.0, 1.0, 0.55],
            [0.6, 0.8, 1.6, 0.30],
            [-0.8, 0.6, 2.7, 0.15],
        ];
        let offsets = [0.43, 2.17, 4.61, 1.29];
        let mut height = 0.0;
        for i in 0..if varied { 4 } else { 3 } {
            let component = if varied { natural[i] } else { regular[i] };
            let phase = (component[0] * xy[0] + component[1] * xy[1]) * component[2] * waves[1]
                - phases[i]
                + if varied { offsets[i] } else { 0.0 };
            height += waves[0] * component[3] * phase.sin();
        }
        height
    }

    pub fn advance(&mut self, seconds: f64) {
        if self.enabled && self.playing && seconds.is_finite() && seconds > 0.0 {
            if self.speed.is_finite() {
                let next = self.phase_time + seconds * self.speed.clamp(0.0, 5.0) as f64;
                if next.is_finite() {
                    self.phase_time = next;
                }
            }
            if self.underwater.caustic_speed.is_finite() {
                let next = self.caustic_time
                    + seconds * self.underwater.caustic_speed.clamp(0.0, 5.0) as f64;
                if next.is_finite() {
                    self.caustic_time = next;
                }
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

    pub(crate) fn caustic_offsets(&self) -> [f32; 2] {
        [0.07, -0.093].map(|speed| (self.caustic_time * speed).rem_euclid(256.0) as f32)
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
    underwater_color: [f32; 4],
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
    scattering_pipeline: wgpu::ComputePipeline,
    background_pipeline: wgpu::RenderPipeline,
    hits: Option<WaterHits>,
    resources: Option<WaterResources>,
    caustics_enabled: bool,
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
                        &crate::underwater::sampling_shader(2),
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
        let scattering_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Underwater volume integration"),
            source: wgpu::ShaderSource::Wgsl(
                shader_source(
                    &[
                        include_str!("underwater_uniforms.wgsl"),
                        include_str!("underwater_volume.wgsl"),
                    ]
                    .concat(),
                )
                .into(),
            ),
        });
        let scattering_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Underwater volume integration"),
                layout: Some(&intersect_layout),
                module: &scattering_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Underwater background"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_underwater_background"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_underwater_background"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            intersect_pipeline,
            scattering_pipeline,
            background_pipeline,
            hits: None,
            resources: None,
            caustics_enabled: false,
            pipeline,
            uniforms,
            bind_group,
            empty_environment: WaterEnvironment::empty(device),
        }
    }

    pub(crate) fn hits(&self) -> Option<&WaterHits> {
        self.hits.as_ref()
    }

    pub(crate) fn release_underwater(&mut self) {
        if self.hits.as_ref().is_some_and(|h| h.volume_size[2] > 1) {
            self.hits = None;
        }
    }

    /// Prepare hits and optional volume before drawing background, proxy, water and GS.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        user: &UserData,
        data: &RenderData,
        environment: Option<&WaterEnvironment>,
        timestamps: crate::profiler::WaterPassTimestamps<'_>,
    ) -> bool {
        let settings = &data.render_config.water;
        if !settings.is_active(user.surface_type) {
            self.release_underwater();
            return false;
        }
        let Some(bounds) = bounds(user, data) else {
            self.release_underwater();
            return false;
        };
        let Some(_) = data.depth_texture.as_ref() else {
            return false;
        };
        let position = camera.position();
        let underwater =
            settings.underwater_frame([position.x, position.y, position.z], Some(bounds));
        let uniforms = Uniforms {
            camera: CameraUniforms::from_camera(camera),
            bounds,
            color: [
                settings.color[0].clamp(0.0, 1.0),
                settings.color[1].clamp(0.0, 1.0),
                settings.color[2].clamp(0.0, 1.0),
                1.0,
            ],
            level: [
                settings.height,
                u32::from(environment.is_some()) as f32,
                // The camera's medium determines the interface, never a ripple lighting normal.
                u32::from(
                    underwater.is_some()
                        && position.z
                            < settings.height + settings.wave_offset([position.x, position.y]),
                ) as f32,
                0.0,
            ],
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
            underwater_color: underwater.map_or([0.0; 4], |frame| {
                [
                    frame.color[0],
                    frame.color[1],
                    frame.color[2],
                    frame.strength,
                ]
            }),
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&uniforms));
        let size = [camera.viewport().width, camera.viewport().height];
        if size.contains(&0) {
            return false;
        }
        let caustics_enabled = underwater.is_some() && settings.underwater.caustics;
        let resources = self
            .resources
            .get_or_insert_with(|| WaterResources::new(device, queue));
        resources.ensure_caustics(device, queue, caustics_enabled);
        if self.hits.as_ref().is_none_or(|h| {
            h.size != size
                || (h.volume_size[2] > 1) != underwater.is_some()
                || self.caustics_enabled != caustics_enabled
        }) {
            self.hits = Some(WaterHits::new(
                device,
                size,
                underwater.is_some(),
                resources,
            ));
        }
        self.caustics_enabled = caustics_enabled;
        let hits = self.hits.as_ref().unwrap();
        if let Some(frame) = underwater {
            let medium = crate::underwater::Uniforms::new(camera, settings, frame, bounds);
            queue.write_buffer(&resources.medium_uniforms, 0, bytemuck::bytes_of(&medium));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Water preparation"),
                timestamp_writes: timestamps.intersection,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, &hits.write, &[]);
            pass.set_pipeline(&self.intersect_pipeline);
            pass.dispatch_workgroups(size[0].div_ceil(8), size[1].div_ceil(8), 1);
            if underwater.is_some() {
                pass.set_pipeline(&self.scattering_pipeline);
                pass.dispatch_workgroups(
                    hits.volume_size[0].div_ceil(8),
                    hits.volume_size[1].div_ceil(8),
                    1,
                );
            }
        }
        true
    }

    pub fn render_background(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        environment: Option<&WaterEnvironment>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Underwater background"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.background_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(
            1,
            &environment.unwrap_or(&self.empty_environment).bind_group,
            &[],
        );
        pass.set_bind_group(2, &self.hits.as_ref().unwrap().read, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Draw after the proxy, loading its depth when present. Requires successful prepare.
    pub fn render_surface(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        data: &RenderData,
        environment: Option<&WaterEnvironment>,
        timestamps: crate::profiler::WaterPassTimestamps<'_>,
    ) {
        let depth = data.depth_texture.as_ref().unwrap();
        let hits = self.hits.as_ref().unwrap();
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

#[cfg(test)]
mod underwater_tests {
    use super::*;

    #[test]
    fn caustic_speed_integrates_without_jumps_or_wave_speed_coupling() {
        let mut water = WaterSettings::default();
        water.enabled = true;
        let mut fast_waves = water.clone();
        fast_waves.speed = 4.0;
        water.advance(1.0);
        fast_waves.advance(1.0);
        assert_eq!(water.caustic_offsets(), fast_waves.caustic_offsets());
        assert_ne!(water.phases(), fast_waves.phases());
        let before = water.caustic_offsets();
        water.underwater.caustic_speed = 2.0;
        assert_eq!(
            water.caustic_offsets(),
            before,
            "speed edit retimed past motion"
        );
        water.advance(0.5);
        let mut reference = WaterSettings::default();
        reference.enabled = true;
        reference.underwater.caustic_speed = 1.0;
        reference.advance(1.25); // 1 second at 0.25x, then 0.5 seconds at 2x.
        assert_eq!(water.caustic_offsets(), reference.caustic_offsets());
        let before = water.caustic_offsets();
        let wave_before = water.phases();
        water.underwater.caustic_speed = 0.0;
        water.advance(0.5);
        assert_eq!(water.caustic_offsets(), before);
        assert_ne!(
            water.phases(),
            wave_before,
            "freezing caustics stopped waves/shafts"
        );
    }

    #[test]
    fn caustic_clock_is_frame_rate_independent_and_respects_pause() {
        let mut one = WaterSettings::default();
        one.enabled = true;
        let mut many = one.clone();
        one.advance(1.0);
        for _ in 0..120 {
            many.advance(1.0 / 120.0);
        }
        assert_eq!(one.caustic_offsets(), many.caustic_offsets());
        let before = one.caustic_offsets();
        for delta in [f64::NAN, f64::INFINITY, -1.0] {
            one.advance(delta);
        }
        assert_eq!(one.caustic_offsets(), before);
        one.playing = false;
        one.advance(10.0);
        assert_eq!(one.caustic_offsets(), before);
        one.playing = true;
        one.enabled = false;
        one.advance(10.0);
        assert_eq!(one.caustic_offsets(), before);
    }

    #[test]
    fn underwater_is_opt_in_and_only_applies_below_the_covered_surface() {
        let mut water = WaterSettings::default();
        water.enabled = true;
        water.amplitude = 0.0;
        let coverage = Some([-5.0, -5.0, 5.0, 5.0]);
        assert!(water.underwater_frame([0.0, 0.0, -1.0], coverage).is_none());
        water.underwater.enabled = true;
        assert!(water.underwater_frame([0.0, 0.0, 1.0], coverage).is_none());
        assert!(water.underwater_frame([6.0, 0.0, -1.0], coverage).is_none());
        assert!(water.underwater_frame([0.0, 0.0, -1.0], None).is_none());
        assert!(water.underwater_frame([0.0, 0.0, -1.0], coverage).is_some());
        let at_surface = water.underwater_frame([0.0, 0.0, 0.0], coverage).unwrap();
        assert!((at_surface.strength - 0.5).abs() < 0.001);
        water.enabled = false;
        assert!(water.underwater_frame([0.0, 0.0, -1.0], coverage).is_none());
    }

    #[test]
    fn underwater_extinction_prefers_nearby_warm_coral() {
        let mut water = WaterSettings::default();
        water.enabled = true;
        water.amplitude = 0.0;
        water.underwater.enabled = true;
        water.underwater.visibility = 12.0;
        let frame = water
            .underwater_frame([0.0, 0.0, -1.0], Some([-5.0, -5.0, 5.0, 5.0]))
            .unwrap();
        assert_eq!(frame.strength, 1.0);
        assert!(frame.extinction[0] > frame.extinction[1]);
        assert!(frame.extinction[1] > frame.extinction[2]);
        let near = frame.transmission(1.0);
        let halfway = frame.transmission(frame.clear_distance + 12.0);
        let far = frame.transmission(frame.clear_distance + 48.0);
        assert_eq!(near, [1.0; 3]);
        assert!((halfway[1] - 0.5).abs() < 0.001);
        assert!(far[0] < far[1] && far[1] < far[2] && far[2] < 0.1);
    }
}
