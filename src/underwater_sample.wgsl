@group(UNDERWATER_GROUP) @binding(1) var underwater_volume: texture_3d<f32>;
@group(UNDERWATER_GROUP) @binding(2) var underwater_sampler: sampler;
@group(UNDERWATER_GROUP) @binding(3) var<uniform> underwater_params: UnderwaterUniforms;
@group(UNDERWATER_GROUP) @binding(4) var caustic_texture: texture_2d<f32>;
@group(UNDERWATER_GROUP) @binding(5) var caustic_sampler: sampler;

// Surface-only illumination: callers apply it before fog, never on sky/water.
// No normals/materials are assumed for reconstructed GS. This is a light cookie,
// not a wave solver or a coral shadow map.
fn underwater_caustics(color: vec3<f32>, world: vec3<f32>, footprint: f32) -> vec3<f32> {
    if underwater_params.caustics.x <= 0.0 || underwater_params.sun.w <= 0.0 { return color; }
    let depth = underwater_params.lighting.x - world.z;
    if depth <= 0.0 || any(world.xy < underwater_params.bounds.xy) || any(world.xy > underwater_params.bounds.zw) { return color; }
    let period = 8.0 * underwater_params.caustics.y;
    let uv = (world.xy - underwater_params.sun.xy * world.z / underwater_params.sun.z) / period;
    let lod = max(log2(max(footprint, 0.0001) * f32(textureDimensions(caustic_texture).x) / period), 0.0);
    let detail = 1.0 - smoothstep(3.0, 5.0, lod);
    if detail <= 0.0 { return color; }
    let a = textureSampleLevel(caustic_texture, caustic_sampler, uv + underwater_params.caustics.zw, lod).r;
    let b = textureSampleLevel(caustic_texture, caustic_sampler, vec2(uv.y, -uv.x) * 1.13 - underwater_params.caustics.wz, lod).r;
    let illumination = 3.0 * underwater_params.caustics.x * underwater_params.sun.w
        * exp(-depth / underwater_params.lighting.y) * smoothstep(0.0, 0.2, depth) * detail * (0.5 * (a + b));
    // Multiplication preserves the baked scene's hue and black level.
    return color * pow(1.0 + illumination, 1.0 / 2.2);
}

// RGB is integrated in-scattering in linear light; alpha is the water path length
// beyond the clear zone. Sampling at a layer's distance stops light behind it.
fn underwater_transport(pixel: vec2<f32>, view_depth: f32) -> vec4<f32> {
    let dims = vec3<f32>(textureDimensions(underwater_volume));
    let z = sqrt(clamp(view_depth / underwater_params.range.x, 0.0, 1.0));
    let uvw = vec3(pixel / underwater_params.camera.viewport, (z * (dims.z - 1.0) + 0.5) / dims.z);
    return textureSampleLevel(underwater_volume, underwater_sampler, uvw, 0.0);
}
fn underwater_compose(color: vec3<f32>, pixel: vec2<f32>, view_depth: f32) -> vec3<f32> {
    // This also guarantees exact preservation inside the clear zone despite
    // trilinear interpolation between the first nonzero depth slices.
    let ndc = vec2(2.0 * pixel.x / underwater_params.camera.viewport.x - 1.0, 1.0 - 2.0 * pixel.y / underwater_params.camera.viewport.y);
    let ray_length = length(vec3(ndc / vec2(underwater_params.camera.projection[0][0], underwater_params.camera.projection[1][1]), -1.0));
    if view_depth * ray_length <= underwater_params.extinction.w { return color; }
    let volume = underwater_transport(pixel, view_depth);
    let underwater_transmission = exp(-underwater_params.extinction.xyz * volume.a);
    let linear = pow(max(color, vec3(0.0)), vec3(2.2));
    return pow(max(linear * underwater_transmission + volume.rgb, vec3(0.0)), vec3(1.0 / 2.2));
}
