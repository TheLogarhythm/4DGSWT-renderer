struct CameraUniforms {
    projection: mat4x4<f32>,
    view: mat4x4<f32>,
    focal: vec2<f32>,
    viewport: vec2<f32>,
    htan_fov: vec2<f32>,
    cam_pos: vec3<f32>,
}
