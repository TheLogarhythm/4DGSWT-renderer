@group(1) @binding(0) var environment: texture_cube<f32>;
@group(1) @binding(1) var environment_sampler: sampler;

fn ripple_hash(cell: vec2<f32>) -> f32 {
    // Periodic only over 256 cells; wrapping animation offsets remains continuous.
    let wrapped = cell - floor(cell / 256.0) * 256.0;
    var p = fract(vec3(wrapped.xyx) * vec3(0.1031,0.1030,0.0973));
    p += dot(p,p.yzx+33.33);
    return fract((p.x+p.y)*p.z) * 2.0 - 1.0;
}
fn noise_gradient(p: vec2<f32>) -> vec2<f32> {
    let cell = floor(p);
    let f = fract(p);
    let u = f*f*f*(f*(f*6.0-15.0)+10.0);
    let du = 30.0*f*f*(f*(f-2.0)+1.0);
    let a = ripple_hash(cell);
    let b = ripple_hash(cell+vec2(1.0,0.0));
    let c = ripple_hash(cell+vec2(0.0,1.0));
    let d = ripple_hash(cell+vec2(1.0,1.0));
    return vec2(mix(b-a,d-c,u.y), mix(c-a,d-b,u.x)) * du;
}
fn ripple_slope(xy: vec2<f32>, footprint: f32) -> vec2<f32> {
    let axes = array<vec2<f32>,4>(vec2(0.96,0.28),vec2(0.73,-0.68),vec2(0.89,0.46),vec2(0.59,-0.81));
    let frequencies = array<f32,4>(1.0,2.03,4.11,8.37);
    let weights = array<f32,4>(0.38,0.30,0.20,0.12);
    var slope = vec2(0.0);
    for (var i = 0u; i < 4u; i++) {
        let a = axes[i];
        let b = vec2(-a.y,a.x);
        let f = frequencies[i] / water.material.w;
        // Elongated ripples, independently oriented/advected in world coordinates.
        let offset = select(water.detail_offsets.xy,water.detail_offsets.zw,(i & 1u) != 0u);
        let p = vec2(dot(xy,a)*1.7,dot(xy,b)*0.7)*f + offset;
        let gradient = noise_gradient(p);
        // Fade bands smaller than the pixel footprint before they can shimmer.
        let band = 1.0-smoothstep(0.2,0.65,footprint*f*1.7);
        slope += (a*gradient.x*1.7+b*gradient.y*0.7)*weights[i]*band;
    }
    return slope * water.material.z;
}


// Filter only the lighting normal at distance; geometric height/contact stays exact.
fn shading_wave_slope(xy: vec2<f32>, footprint: f32) -> vec2<f32> {
    let varied = water.waves.z > 0.5;
    var result = vec2(0.0);
    for (var i = 0u; i < select(3u,4u,varied); i++) {
        let c = water_component(i,varied);
        let k = c.z * water.waves.y;
        let phase = dot(c.xy,xy)*k-water.phases[i]+water_phase_offset(i,varied);
        let band = 1.0-smoothstep(0.2,0.65,footprint*k/6.28318530718);
        result += water.waves.x*c.w*k*cos(phase)*c.xy*band;
    }
    return result;
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
    if water.waves.x > 0.0 {
        // The exact curved surface is intersected in the compute prepass. Full
        // viewport coverage includes crests above the old flat silhouette.
        return vec4(corners[index] * 2.0 - 1.0, 0.0, 1.0);
    }
    let xy = mix(water.bounds.xy, water.bounds.zw, corners[index]);
    var clip = water.camera.projection * water.camera.view * vec4(xy, water.level.x, 1.0);
    // The camera uses OpenGL NDC; the shared depth attachment uses WebGPU NDC.
    clip.z = 0.5 * (clip.z + clip.w);
    return clip;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fs_main(@builtin(position) pixel: vec4<f32>) -> FragmentOutput {
    let hit = load_water_hit(pixel.xy);
    let ray = water_ray(pixel.xy, water.camera);
    var distance = hit.w;
    if water.waves.x <= 0.0 {
        // Flat-plane helper lanes outside finite coverage still need the plane
        // position for derivatives; cached validity continues to control drawing.
        distance = select(0.0,(water.level.x-water.camera.cam_pos.z)/ray.z,abs(ray.z)>1e-7);
    }
    let point = water.camera.cam_pos + ray * distance;
    // Derivatives execute in uniform control flow, before any discard.
    let footprint = max(length(dpdx(point.xy)),length(dpdy(point.xy)));
    var slope = water_wave_sample(point.xy, water.waves, water.phases).yz;
    if water.material.x > 0.0 || water.material.z > 0.0 { slope = shading_wave_slope(point.xy,footprint); }
    if water.material.z > 0.0 { slope += ripple_slope(point.xy, footprint); }
    var normal = normalize(vec3(-slope,1.0));
    let view = -normalize(ray);
    if water.material.x > 0.0 || water.material.z > 0.0 { normal = faceForward(normal,-view,normal); }
    let reflected = reflect(-view,normal);
    let angular_footprint = max(length(dpdx(reflected)),length(dpdy(reflected)));
    if hit.y < 0.5 { discard; }
    let light = normalize(vec3(-0.6,-0.4,1.0));
    // Preserve the previous diagnostic appearance for a useful all-effects-off A/B.
    let diffuse_strength = mix(2.0,0.35,water.material.x);
    let brightness = clamp(1.0+diffuse_strength*(dot(normal,light)-light.z),0.35,1.6);
    var color = water.color.rgb * brightness;
    if water.material.x > 0.0 {
        // Pixel footprint also selects coarser reflection levels at distance.
        // These mips encode roughness, not an ordinary spatial mip pyramid.
        // GGX angular width is roughly roughness squared; combine pixel variance
        // in that domain instead of incorrectly using log2(texture footprint).
        let filtered_roughness = sqrt(max(water.material.y*water.material.y,angular_footprint*0.5));
        let lod = clamp(filtered_roughness*8.0,0.0,8.0);
        let sky = textureSampleLevel(environment,environment_sampler,reflected,lod).rgb;
        let cos_theta = clamp(dot(normal,view),0.0,1.0);
        let fresnel = (0.02037 + 0.97963*pow(1.0-cos_theta,5.0)) * water.material.x;
        color = pow(mix(pow(max(color,vec3(0.0)),vec3(2.2)),sky,fresnel),vec3(1.0/2.2));
    }
    var out: FragmentOutput;
    out.color = vec4(color,1.0);
    out.depth = hit.x;
    return out;
}
