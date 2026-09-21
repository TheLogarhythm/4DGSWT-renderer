// Intersection math for the per-pixel cache shared by water and GS.
// Result: (WebGPU depth, valid hit). Bounds and height are final world units.
fn water_plane_hit(pixel: vec2<f32>, camera: CameraUniforms, height: f32, bounds: vec4<f32>) -> vec2<f32> {
    let ndc = vec2(2.0 * pixel.x / camera.viewport.x - 1.0, 1.0 - 2.0 * pixel.y / camera.viewport.y);
    let view3 = mat3x3(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let ray = transpose(view3) * vec3(ndc.x / camera.projection[0][0], ndc.y / camera.projection[1][1], -1.0);
    if abs(ray.z) <= 1e-7 { return vec2(1.0, 0.0); }
    let distance = (height - camera.cam_pos.z) / ray.z;
    let hit = camera.cam_pos + distance * ray;
    if distance <= 0.0 || any(hit.xy < bounds.xy) || any(hit.xy > bounds.zw) { return vec2(1.0, 0.0); }
    let clip = camera.projection * camera.view * vec4(hit, 1.0);
    let depth = 0.5 * (clip.z / clip.w + 1.0);
    return vec2(depth, select(0.0, 1.0, depth >= 0.0 && depth <= 1.0));
}

// Each descriptor is (direction XY, relative frequency, amplitude weight).
// Fixed irregular ratios avoid a short common repeat distance. Both GS clipping
// and the water ray solver use these exact descriptors, including derivative bounds.
fn water_component(i: u32, varied: bool) -> vec4<f32> {
    let old = array<vec4<f32>, 4>(vec4(1.0,0.0,1.0,0.55), vec4(0.6,0.8,1.6,0.30), vec4(-0.8,0.6,2.7,0.15), vec4(0.0));
    let natural = array<vec4<f32>, 4>(
        vec4(0.946,0.324,0.830,0.32), vec4(0.544,-0.839,1.271,0.28),
        vec4(-0.771,0.637,1.913,0.23), vec4(-0.217,-0.976,2.731,0.17));
    if varied { return natural[i]; }
    return old[i];
}
fn water_phase_offset(i: u32, varied: bool) -> f32 {
    let offsets = array<f32,4>(0.43,2.17,4.61,1.29);
    return select(0.0,offsets[i],varied);
}
// (height offset, dHeight/dX, dHeight/dY). Weights sum to one.
fn water_wave_sample(xy: vec2<f32>, waves: vec4<f32>, phases: vec4<f32>) -> vec3<f32> {
    let varied = waves.z > 0.5;
    var result = vec3(0.0);
    for (var i = 0u; i < select(3u,4u,varied); i++) {
        let c = water_component(i,varied);
        let k = c.z * waves.y;
        let amplitude = c.w * waves.x;
        let phase = dot(c.xy,xy) * k - phases[i] + water_phase_offset(i,varied);
        result += vec3(amplitude * sin(phase), amplitude * k * cos(phase) * c.xy);
    }
    return result;
}
fn water_ray(pixel: vec2<f32>, camera: CameraUniforms) -> vec3<f32> {
    let ndc = vec2(2.0 * pixel.x / camera.viewport.x - 1.0, 1.0 - 2.0 * pixel.y / camera.viewport.y);
    let view3 = mat3x3(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    // Deliberately unnormalized: distance along this ray is positive view depth.
    return transpose(view3) * vec3(ndc.x / camera.projection[0][0], ndc.y / camera.projection[1][1], -1.0);
}
// (WebGPU depth, valid, world Z, positive view depth). Both render passes read
// the cached depth and converged distance: no depth bias is needed.
fn water_surface_hit(pixel: vec2<f32>, camera: CameraUniforms, height: f32, bounds: vec4<f32>, waves: vec4<f32>, phases: vec4<f32>) -> vec4<f32> {
    let ray = water_ray(pixel, camera);
    if waves.x <= 0.0 {
        let flat = water_plane_hit(pixel, camera, height, bounds);
        return vec4(flat, height, select(0.0, (height - camera.cam_pos.z) / ray.z, abs(ray.z) > 1e-7));
    }
    let origin = camera.cam_pos;
    let box_min = vec3(bounds.xy, height - waves.x);
    let box_max = vec3(bounds.zw, height + waves.x);
    let near = camera.projection[3][2] / (camera.projection[2][2] - 1.0);
    let far = camera.projection[3][2] / (camera.projection[2][2] + 1.0);
    var enter = max(near, 0.00001);
    var leave = far;
    for (var axis = 0u; axis < 3u; axis++) {
        if abs(ray[axis]) < 1e-7 {
            if origin[axis] < box_min[axis] || origin[axis] > box_max[axis] { return vec4(0.0); }
        } else {
            let a = (box_min[axis] - origin[axis]) / ray[axis];
            let b = (box_max[axis] - origin[axis]) / ray[axis];
            enter = max(enter, min(a, b));
            leave = min(leave, max(a, b));
        }
    }
    if enter > leave { return vec4(0.0); }
    // Bound |d(rayZ-height)/dt|. Conservative steps cannot jump across a crest,
    // unlike an unconstrained Newton solve at grazing angles.
    var rate = abs(ray.z);
    var curvature = 0.0;
    var max_frequency = 0.0;
    let varied = waves.z > 0.5;
    for (var i = 0u; i < select(3u,4u,varied); i++) {
        let c = water_component(i,varied);
        let projected_k = waves.y * c.z * dot(ray.xy,c.xy);
        rate += waves.x * c.w * abs(projected_k);
        curvature += waves.x * c.w * projected_k * projected_k;
        max_frequency = max(max_frequency,abs(projected_k));
    }
    let tolerance = max(0.00002, 0.00004 / waves.y);
    // Work scales with the traversed phase range for grazing rays.
    let cycles = max_frequency * (leave - enter) / 6.28318530718;
    let budget = u32(clamp(64.0 + ceil(32.0 * cycles), 64.0, 16384.0));
    var t = enter;
    for (var iteration = 0u; iteration < budget; iteration++) {
        let point = origin + ray * t;
        let sample = water_wave_sample(point.xy, waves, phases);
        let delta = point.z - height - sample.x;
        if abs(delta) <= tolerance {
            let clip = camera.projection * camera.view * vec4(point, 1.0);
            let depth = 0.5 * (clip.z / clip.w + 1.0);
            return vec4(depth, select(0.0, 1.0, depth >= 0.0 && depth <= 1.0), point.z, t);
        }
        var step = abs(delta) / max(rate, 1e-7);
        if curvature > 1e-10 {
            // A second-order bound takes larger safe steps through flat/grazing
            // regions. |F| + signed(F')*s - curvature*s*s/2 cannot reach zero
            // before this step, so the nearest crossing is still preserved.
            let derivative = sign(delta) * (ray.z - dot(sample.yz, ray.xy));
            let root = sqrt(derivative * derivative + 2.0 * curvature * abs(delta));
            var curved_step = (root + derivative) / curvature;
            if derivative < 0.0 { curved_step = 2.0 * abs(delta) / (root - derivative); }
            step = max(step, curved_step);
        }
        t += step;
        if t > leave { return vec4(0.0); }
    }
    return vec4(0.0);
}
