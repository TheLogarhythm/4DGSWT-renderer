//! View-aligned, cumulative single scattering, sampled at each transparent layer.
//! No opaque GS depth approximation or temporal history is needed for compositing.
use crate::{
    camera::{Camera, CameraUniforms},
    water::{UnderwaterFrame, WaterSettings},
};

pub(crate) const SLICES: u32 = 48;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    camera: CameraUniforms,
    bounds: [f32; 4],
    color: [f32; 4],
    extinction: [f32; 4],
    sun: [f32; 4],
    // surface height, depth falloff, shaft strength, shaft width
    lighting: [f32; 4],
    // max positive view depth, reserved
    range: [f32; 4],
    waves: [f32; 4],
    phases: [f32; 4],
    // caustic strength, cell spacing, two periodic animation offsets
    caustics: [f32; 4],
}

impl Uniforms {
    pub fn new(
        camera: &Camera,
        water: &WaterSettings,
        frame: UnderwaterFrame,
        bounds: [f32; 4],
    ) -> Self {
        let finite = |x: f32, fallback: f32, lo: f32, hi: f32| {
            if x.is_finite() {
                x.clamp(lo, hi)
            } else {
                fallback
            }
        };
        let settings = &water.underwater;
        let azimuth = finite(settings.sun_azimuth, 120.0, -180.0, 180.0).to_radians();
        let elevation = finite(settings.sun_elevation, 55.0, 10.0, 90.0).to_radians();
        // Direction towards the sun after refraction into water (Snell's law).
        let horizontal = elevation.cos() / 1.333;
        let visibility = finite(settings.visibility, 15.0, 0.5, 1000.0);
        Self {
            camera: CameraUniforms::from_camera(camera),
            bounds,
            color: [
                frame.color[0],
                frame.color[1],
                frame.color[2],
                frame.strength,
            ],
            extinction: [
                frame.extinction[0],
                frame.extinction[1],
                frame.extinction[2],
                frame.clear_distance,
            ],
            sun: [
                horizontal * azimuth.cos(),
                horizontal * azimuth.sin(),
                (1.0 - horizontal * horizontal).sqrt(),
                finite(settings.sunlight, 0.6, 0.0, 2.0),
            ],
            lighting: [
                water.height,
                visibility * 1.25,
                if settings.light_shafts {
                    finite(settings.shaft_strength, 0.7, 0.0, 2.0)
                } else {
                    0.0
                },
                finite(settings.shaft_scale, 3.0, 0.5, 20.0),
            ],
            range: [
                (visibility * 8.0 + frame.clear_distance).clamp(8.0, 8000.0),
                0.0,
                0.0,
                0.0,
            ],
            waves: water.waves(),
            phases: water.phases(),
            caustics: [
                if settings.caustics {
                    finite(settings.caustic_strength, 0.65, 0.0, 2.0) * frame.strength
                } else {
                    0.0
                },
                finite(settings.caustic_scale, 1.5, 0.25, 20.0),
                water.caustic_offsets()[0],
                water.caustic_offsets()[1],
            ],
        }
    }
}

pub(crate) fn sampling_shader(group: u32) -> String {
    [
        include_str!("underwater_uniforms.wgsl"),
        include_str!("underwater_sample.wgsl"),
    ]
    .concat()
    .replace("UNDERWATER_GROUP", &group.to_string())
}
