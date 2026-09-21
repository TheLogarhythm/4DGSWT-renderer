@group(0) @binding(0) var sky: texture_cube<f32>;
@group(0) @binding(1) var sky_sampler: sampler;
// face, dimension, perceptual roughness, source-is-equirectangular
@group(0) @binding(2) var<uniform> params: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4(p[i], 0.0, 1.0);
}
fn direction(uv: vec2<f32>, face: u32) -> vec3<f32> {
    switch face {
        case 0u: { return normalize(vec3(1.0,-uv.y,-uv.x)); }
        case 1u: { return normalize(vec3(-1.0,-uv.y,uv.x)); }
        case 2u: { return normalize(vec3(uv.x,1.0,uv.y)); }
        case 3u: { return normalize(vec3(uv.x,-1.0,-uv.y)); }
        case 4u: { return normalize(vec3(uv.x,-uv.y,1.0)); }
        default: { return normalize(vec3(-uv.x,-uv.y,-1.0)); }
    }
}
fn sample_sky(world: vec3<f32>) -> vec3<f32> {
    // Exactly the world-to-cube convention of skybox.wgsl, including HDRI's Y flip.
    var lookup = vec3(world.x, world.z, world.y);
    if params.w > 0.5 { lookup.y = -lookup.y; }
    // Existing sky cubes contain display-encoded color (HDR is already tone mapped).
    return pow(max(textureSampleLevel(sky, sky_sampler, lookup, 0.0).rgb, vec3(0.0)), vec3(2.2));
}
@fragment fn fs_main(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let n = direction(pixel.xy / params.y * 2.0 - 1.0, u32(params.x));
    if params.z == 0.0 { return vec4(sample_sky(n), 1.0); }
    let up = select(vec3(0.0,0.0,1.0), vec3(1.0,0.0,0.0), abs(n.z) > 0.99);
    let t = normalize(cross(up,n));
    let b = cross(n,t);
    let a = params.z * params.z;
    var sum = vec3(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 64u; i++) {
        let u = f32(i) / 64.0;
        let v = f32(reverseBits(i)) * 2.3283064365386963e-10;
        let phi = 6.28318530718 * u;
        let c = sqrt((1.0-v) / (1.0+(a*a-1.0)*v));
        let s = sqrt(max(0.0, 1.0-c*c));
        let h = t*(cos(phi)*s) + b*(sin(phi)*s) + n*c;
        let l = reflect(-n,h);
        let w = max(dot(n,l),0.0);
        sum += sample_sky(l) * w;
        weight += w;
    }
    return vec4(sum / max(weight,0.0001), 1.0);
}
