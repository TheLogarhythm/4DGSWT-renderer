struct Uniforms {
    camera: CameraUniforms,
    bounds: vec4<f32>,
    color: vec4<f32>,
    level: vec4<f32>,
    waves: vec4<f32>,
    phases: vec4<f32>,
    material: vec4<f32>, // reflection strength, roughness, ripple strength, world scale
    detail_offsets: vec4<f32>,
}
@group(0) @binding(0) var<uniform> water: Uniforms;
