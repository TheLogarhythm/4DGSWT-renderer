@group(1) @binding(1) var scattering_output: texture_storage_3d<rgba16float, write>;
@group(1) @binding(2) var<uniform> medium: UnderwaterUniforms;

fn incident_water_light(point: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    let depth = max(medium.lighting.x - point.z, 0.0);
    let daylight = exp(-depth / medium.lighting.y);
    let facing = pow(max(dot(direction, medium.sun.xyz), 0.0), 4.0);
    let ambient = 0.28 + 0.72 * daylight;
    let direct = medium.sun.w * daylight * (0.15 + 0.85 * facing);
    var shaft_light = vec3(0.0);
    if medium.lighting.z > 0.0 && medium.sun.w > 0.0 {
        // Authored light columns, independent of wave shape and surface height.
        // Sparse 2D spots avoid the nearly uniform integral of repeating sheets.
        // Project onto a fixed world plane so camera motion cannot drag the light.
        let p = (point.xy - medium.sun.xy * point.z / medium.sun.z) / (4.0 * medium.lighting.w);
        let cell = floor(p);
        let random = fract(sin(vec3(dot(cell, vec2(127.1, 311.7)), dot(cell, vec2(269.5, 183.3)), dot(cell, vec2(419.2, 371.9)))) * 43758.5453);
        // Bounded drift stays inside the cell, including when phases wrap.
        let drift = 0.025 * vec2(sin(medium.phases.x + random.z * 6.283185), sin(medium.phases.y + random.x * 6.283185));
        let center = vec2(0.5) + (random.xy - 0.5) * 0.35 + drift;
        let radius = length((fract(p) - center) * vec2(1.0, 1.25));
        let beam = 1.0 - smoothstep(0.035, 0.16, radius);
        // Brighter cores, not more fog. Side views retain readable beams;
        // their darkness/occlusion still comes from layered GS compositing.
        shaft_light = vec3(0.82, 0.94, 1.0) * 3.2 * medium.lighting.z * medium.sun.w * daylight * beam * (0.75 + 0.25 * facing);
    }
    return pow(medium.color.rgb, vec3(2.2)) * (ambient + direct) + shaft_light;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(scattering_output);
    if any(id.xy >= dims.xy) { return; }
    let pixel = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(dims.xy) * medium.camera.viewport;
    let ray = water_ray(pixel, medium.camera);
    let ray_length = length(ray);
    let direction = ray / ray_length;
    // The exact waved surface is evaluated once per column, not once per GS.
    var surface = water_surface_hit(pixel, medium.camera, medium.lighting.x, medium.bounds, medium.waves, medium.phases);
    // The visible surface solver clips to the camera near plane. Transport must
    // also stop/enter at interfaces between the eye and that plane.
    let camera_wave = water_wave_sample(medium.camera.cam_pos.xy, medium.waves, medium.phases);
    let camera_level = medium.lighting.x + camera_wave.x;
    let origin_delta = medium.camera.cam_pos.z - camera_level;
    let near = max(medium.camera.projection[3][2] / (medium.camera.projection[2][2] - 1.0), 0.00001);
    let near_point = medium.camera.cam_pos + ray * near;
    let near_delta = near_point.z - medium.lighting.x - water_wave_sample(near_point.xy, medium.waves, medium.phases).x;
    if origin_delta * near_delta < 0.0 {
        var lo = 0.0;
        var hi = near;
        for (var i = 0u; i < 10u; i++) {
            let mid = (lo + hi) * 0.5;
            let p = medium.camera.cam_pos + ray * mid;
            let delta = p.z - medium.lighting.x - water_wave_sample(p.xy, medium.waves, medium.phases).x;
            if delta * origin_delta > 0.0 { lo = mid; } else { hi = mid; }
        }
        surface = vec4(0.0, 1.0, 0.0, (lo + hi) * 0.5);
    }
    var end = medium.range.x;
    var enter = 0.0;
    let starts_inside = origin_delta < 0.0 || (abs(origin_delta) < 1e-7 && dot(ray, vec3(-camera_wave.yz, 1.0)) < 0.0);
    if starts_inside {
        if surface.y > 0.5 { end = min(end, surface.w); }
    } else {
        enter = select(end, surface.w, surface.y > 0.5);
    }
    for (var axis = 0u; axis < 2u; axis++) {
        if abs(ray[axis]) > 1e-7 {
            let edge = select(medium.bounds[axis], medium.bounds[axis + 2u], ray[axis] > 0.0);
            end = min(end, max((edge - medium.camera.cam_pos[axis]) / ray[axis], 0.0));
        }
    }
    let clear = enter + medium.extinction.w / ray_length;
    var scattering = vec3(0.0);
    var optical_length = 0.0;
    var previous = 0.0;
    textureStore(scattering_output, vec3<i32>(vec3<u32>(id.xy, 0u)), vec4(0.0));
    for (var z = 1u; z < dims.z; z++) {
        let fraction = f32(z) / f32(dims.z - 1u);
        let current = min(fraction * fraction * medium.range.x, end);
        let start = max(previous, clear);
        let step = max(current - start, 0.0) * 0.5;
        for (var j = 0u; j < 2u; j++) {
            if step > 0.0 {
                let point = medium.camera.cam_pos + ray * (start + (f32(j) + 0.5) * step);
                let ds = step * ray_length;
                let transmission = exp(-medium.extinction.xyz * optical_length);
                scattering += transmission * (vec3(1.0) - exp(-medium.extinction.xyz * ds)) * incident_water_light(point, direction);
                optical_length += ds;
            }
        }
        textureStore(scattering_output, vec3<i32>(vec3<u32>(id.xy, z)), vec4(scattering, optical_length));
        previous = current;
    }
}
