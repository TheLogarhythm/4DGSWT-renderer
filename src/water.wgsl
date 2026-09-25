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

fn water_to_air_transmission(cosine: f32) -> f32 {
    let eta = 1.333;
    let cos_i = clamp(cosine, 0.0, 1.0);
    let sin_t2 = eta * eta * (1.0 - cos_i * cos_i);
    if sin_t2 >= 1.0 { return 0.0; }
    let cos_t = sqrt(max(1.0 - sin_t2, 0.0));
    let rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 1.0 - 0.5 * (rs * rs + rp * rp);
}

// View from water into air: refracted sky is visible inside Snell's window.
// Outside it, total internal reflection sees the underwater medium, not the sky.
// Without a scene-reflection pass, use the far-water radiance for that reflection.
fn underwater_surface_color(incident: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>, view_depth: f32, lod: f32) -> vec3<f32> {
    let eta = 1.333;
    let cos_i = clamp(dot(-incident, normal), 0.0, 1.0);
    // Continuous angular filtering approximates rough-facet transmission near
    // the critical angle without the visible bands of a few discrete normal taps.
    let critical_cos = sqrt(1.0 - 1.0 / (eta * eta));
    let spread = max(water.material.y * 0.25, 0.0001);
    let window = smoothstep(critical_cos - spread, critical_cos + spread, cos_i);
    let sky_weight = window * water_to_air_transmission(max(cos_i, critical_cos + spread * 0.6));
    // Approximate the reflected underwater environment separately from the top
    // material. Upward-facing reflected directions receive more ambient light.
    let reflected_direction = reflect(incident, normal);
    var interface_color = water.underwater_color.rgb * (0.85 + 0.15 * max(reflected_direction.z, 0.0));
    if sky_weight > 0.0 {
        // For rough facets just beyond the mean critical angle, use the tangent
        // direction of the transmitted lobe rather than a zero refraction vector.
        let sin_t2 = min(eta * eta * (1.0 - cos_i * cos_i), 0.9999);
        let tangent = incident + normal * cos_i;
        let transmitted = tangent / max(length(tangent), 1e-5) * sqrt(sin_t2) - normal * sqrt(1.0 - sin_t2);
        // A neutral daylight fallback also works when no environment is loaded.
        var sky = vec3(0.70, 0.82, 0.86);
        if water.level.y > 0.5 {
            let sky_linear = textureSampleLevel(environment, environment_sampler, transmitted, lod).rgb;
            sky = pow(max(sky_linear, vec3(0.0)), vec3(1.0 / 2.2));
        }
        interface_color = mix(interface_color, sky, sky_weight);
    }
    return underwater_compose(interface_color, pixel, view_depth);
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
    if water.level.z > 0.5 && water.underwater_color.a > 0.0 {
        let filtered_roughness = sqrt(max(water.material.y*water.material.y,angular_footprint*0.5));
        let lod = clamp(filtered_roughness*8.0,0.0,8.0);
        let inward = -normalize(vec3(-slope, 1.0));
        let below_normal = faceForward(inward, normalize(ray), inward);
        let color = underwater_surface_color(normalize(ray), below_normal, pixel.xy, distance, lod);
        return FragmentOutput(vec4(color, 1.0), hit.x);
    }
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

@vertex
fn vs_underwater_background(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let xy = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4(xy[index], 0.0, 1.0);
}
@fragment
fn fs_underwater_background(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let ray = normalize(water_ray(pixel.xy, water.camera));
    var sky = vec3(0.0);
    if water.level.y > 0.5 {
        sky = pow(max(textureSampleLevel(environment, environment_sampler, ray, 0.0).rgb, vec3(0.0)), vec3(1.0 / 2.2));
    }
    // Missing tiles are distant water, not a view of air through the edge of the
    // loaded window. Real air transmission is drawn by the interface pass.
    let depth = max(water.level.x - water.camera.cam_pos.z, 0.0);
    let brightness = 0.28 + 0.72 * exp(-depth / underwater_params.lighting.y);
    let distant_water = water.underwater_color.rgb * pow(brightness, 1.0 / 2.2);
    let color = underwater_compose(distant_water, pixel.xy, underwater_params.range.x);
    return vec4(mix(sky, color, water.underwater_color.a), 1.0);
}
