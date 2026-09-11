struct CanonicalGpu {
    position: vec4<f32>,
    log_scale: vec4<f32>,
    rotation: vec4<f32>,
}

struct BasisGpu {
    translation: vec4<f32>,
    rotation: vec4<f32>,
    log_scale: vec4<f32>,
}

struct PlacementGpu {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
}

struct MotionParamsGpu {
    row_start: u32,
    row_count: u32,
    top_k: u32,
    basis_count: u32,
    texture_width: u32,
    motion_enabled: u32,
    blend_alpha: f32,
    diagnostic_row: u32,
    preview_channel_mask: u32,
    separate_banks: u32,
    padding0: u32,
    padding1: u32,
    translation_master_gain: vec4<f32>,
    rotation_scale_gain: vec4<f32>,
}

struct MotionVector {
    translation: vec3<f32>,
    rotation: vec3<f32>,
    log_scale: vec3<f32>,
}

struct GaussianState {
    position: vec3<f32>,
    log_scale: vec3<f32>,
    rotation: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> canonical_rows: array<CanonicalGpu>;
@group(0) @binding(1) var<storage, read> basis_ids: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> transform_ids: array<u32>;
@group(0) @binding(4) var<storage, read> basis_frames: array<BasisGpu>;
@group(0) @binding(5) var<storage, read> placements: array<PlacementGpu>;
@group(0) @binding(6) var base_texture: texture_2d<u32>;
@group(0) @binding(7) var output_texture: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(8) var<uniform> params: MotionParamsGpu;
@group(0) @binding(9) var<storage, read_write> diagnostic_state: GaussianState;
@group(0) @binding(10) var<storage, read> motion_channel_masks: array<u32>;

fn texture_coord(pixel_index: u32) -> vec2<i32> {
    return vec2(
        i32(pixel_index % params.texture_width),
        i32(pixel_index / params.texture_width),
    );
}

fn reconstruct_motion(local_row: u32, frame: u32) -> MotionVector {
    var result: MotionVector;
    result.translation = vec3(0.0);
    result.rotation = vec3(0.0);
    result.log_scale = vec3(0.0);
    if (params.separate_banks != 0u) {
        let frame_offset = frame * 3u * params.basis_count;
        for (var slot = 0u; slot < params.top_k; slot += 1u) {
            let coefficient_index = (local_row * 3u) * params.top_k + slot;
            let basis_id = basis_ids[coefficient_index];
            let weight = weights[coefficient_index];
            let basis = basis_frames[frame_offset + basis_id];
            result.translation += weight * basis.translation.xyz;
        }
        for (var slot = 0u; slot < params.top_k; slot += 1u) {
            let coefficient_index = (local_row * 3u + 1u) * params.top_k + slot;
            let basis_id = basis_ids[coefficient_index];
            let weight = weights[coefficient_index];
            let basis = basis_frames[frame_offset + params.basis_count + basis_id];
            result.rotation += weight * basis.rotation.xyz;
        }
        for (var slot = 0u; slot < params.top_k; slot += 1u) {
            let coefficient_index = (local_row * 3u + 2u) * params.top_k + slot;
            let basis_id = basis_ids[coefficient_index];
            let weight = weights[coefficient_index];
            let basis = basis_frames[frame_offset + 2u * params.basis_count + basis_id];
            result.log_scale += weight * basis.log_scale.xyz;
        }
        return result;
    }
    for (var slot = 0u; slot < params.top_k; slot += 1u) {
        let coefficient_index = local_row * params.top_k + slot;
        let basis_id = basis_ids[coefficient_index];
        let weight = weights[coefficient_index];
        let basis = basis_frames[frame * params.basis_count + basis_id];
        result.translation += weight * basis.translation.xyz;
        result.rotation += weight * basis.rotation.xyz;
        result.log_scale += weight * basis.log_scale.xyz;
    }
    return result;
}

fn transform_translation(value: vec3<f32>, transform_id: u32) -> vec3<f32> {
    let placement = placements[transform_id];
    return vec3(
        dot(placement.row0.xyz, value),
        dot(placement.row1.xyz, value),
        dot(placement.row2.xyz, value),
    );
}

fn quaternion_multiply(left: vec4<f32>, right: vec4<f32>) -> vec4<f32> {
    let lw = left.x;
    let lx = left.y;
    let ly = left.z;
    let lz = left.w;
    let rw = right.x;
    let rx = right.y;
    let ry = right.z;
    let rz = right.w;
    return vec4(
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    );
}

fn quaternion_exp(rotation_vector: vec3<f32>) -> vec4<f32> {
    let theta_squared = dot(rotation_vector, rotation_vector);
    if (theta_squared <= 1.0e-12) {
        let theta_fourth = theta_squared * theta_squared;
        let scalar = 1.0 - theta_squared / 8.0 + theta_fourth / 384.0;
        let vector_scale = 0.5 - theta_squared / 48.0 + theta_fourth / 3840.0;
        return vec4(scalar, rotation_vector * vector_scale);
    }
    let theta = sqrt(theta_squared);
    let half_theta = 0.5 * theta;
    return vec4(cos(half_theta), rotation_vector * (sin(half_theta) / theta));
}

fn standardized_quaternion(value: vec4<f32>) -> vec4<f32> {
    let length_squared = dot(value, value);
    if (!finite_f32(length_squared) || length_squared <= 0.0) {
        return vec4(0.0);
    }
    var normalized = value * inverseSqrt(length_squared);
    if (normalized.x < 0.0) {
        normalized = -normalized;
    }
    return normalized;
}

fn evaluate_state(local_row: u32, frame: u32) -> GaussianState {
    let canonical = canonical_rows[local_row];
    var motion = reconstruct_motion(local_row, frame);
    let effective_mask = motion_channel_masks[local_row] & params.preview_channel_mask;
    if ((effective_mask & 1u) == 0u) {
        motion.translation = vec3(0.0);
    }
    if ((effective_mask & 2u) == 0u) {
        motion.rotation = vec3(0.0);
    }
    if ((effective_mask & 4u) == 0u) {
        motion.log_scale = vec3(0.0);
    }
    let transformed_translation = transform_translation(
        motion.translation,
        transform_ids[local_row],
    ) * params.translation_master_gain.xyz * params.translation_master_gain.w;
    let gained_rotation = motion.rotation
        * params.rotation_scale_gain.x
        * params.translation_master_gain.w;
    let gained_log_scale = motion.log_scale
        * params.rotation_scale_gain.y
        * params.translation_master_gain.w;
    var state: GaussianState;
    state.position = canonical.position.xyz + transformed_translation;
    state.log_scale = canonical.log_scale.xyz + gained_log_scale;
    state.rotation = standardized_quaternion(quaternion_multiply(
        standardized_quaternion(canonical.rotation),
        quaternion_exp(gained_rotation),
    ));
    return state;
}

fn slerp(from_value: vec4<f32>, to_value: vec4<f32>, alpha: f32) -> vec4<f32> {
    var to = to_value;
    var cosine = dot(from_value, to);
    if (cosine < 0.0) {
        to = -to;
        cosine = -cosine;
    }
    if (cosine > 0.9995) {
        return standardized_quaternion(mix(from_value, to, alpha));
    }
    let angle = acos(clamp(cosine, -1.0, 1.0));
    let sine = sin(angle);
    return standardized_quaternion(
        (sin((1.0 - alpha) * angle) / sine) * from_value
        + (sin(alpha * angle) / sine) * to,
    );
}

fn blend_states(from_state: GaussianState, to_state: GaussianState, alpha: f32) -> GaussianState {
    var state: GaussianState;
    state.position = mix(from_state.position, to_state.position, alpha);
    state.log_scale = mix(from_state.log_scale, to_state.log_scale, alpha);
    state.rotation = slerp(from_state.rotation, to_state.rotation, alpha);
    return state;
}

fn covariance(rotation: vec4<f32>, log_scale: vec3<f32>) -> array<f32, 6> {
    let w = rotation.x;
    let x = rotation.y;
    let y = rotation.z;
    let z = rotation.w;
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r01 = 2.0 * (x * y - w * z);
    let r02 = 2.0 * (x * z + w * y);
    let r10 = 2.0 * (x * y + w * z);
    let r11 = 1.0 - 2.0 * (x * x + z * z);
    let r12 = 2.0 * (y * z - w * x);
    let r20 = 2.0 * (x * z - w * y);
    let r21 = 2.0 * (y * z + w * x);
    let r22 = 1.0 - 2.0 * (x * x + y * y);
    let variance = exp(2.0 * log_scale);
    return array(
        r00 * r00 * variance.x + r01 * r01 * variance.y + r02 * r02 * variance.z,
        r00 * r10 * variance.x + r01 * r11 * variance.y + r02 * r12 * variance.z,
        r00 * r20 * variance.x + r01 * r21 * variance.y + r02 * r22 * variance.z,
        r10 * r10 * variance.x + r11 * r11 * variance.y + r12 * r12 * variance.z,
        r10 * r20 * variance.x + r11 * r21 * variance.y + r12 * r22 * variance.z,
        r20 * r20 * variance.x + r21 * r21 * variance.y + r22 * r22 * variance.z,
    );
}

fn finite_f32(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return finite_f32(value.x) && finite_f32(value.y) && finite_f32(value.z);
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return finite_f32(value.x)
        && finite_f32(value.y)
        && finite_f32(value.z)
        && finite_f32(value.w);
}

fn valid_quaternion(value: vec4<f32>) -> bool {
    if (!finite_vec4(value)) {
        return false;
    }
    let length_squared = dot(value, value);
    return finite_f32(length_squared) && length_squared > 0.0;
}

fn covariance_packable(sigma: array<f32, 6>) -> bool {
    for (var index = 0u; index < 6u; index += 1u) {
        let packed = 4.0 * sigma[index];
        if (!finite_f32(packed) || abs(packed) > 65504.0) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let local_row = invocation.x;
    if (local_row >= params.row_count) {
        return;
    }
    let global_row = params.row_start + local_row;
    let first_coord = texture_coord(global_row * 2u);
    let second_coord = texture_coord(global_row * 2u + 1u);
    let base_first = textureLoad(base_texture, first_coord, 0);
    let base_second = textureLoad(base_texture, second_coord, 0);
    if (params.motion_enabled == 0u) {
        textureStore(output_texture, first_coord, base_first);
        textureStore(output_texture, second_coord, base_second);
        return;
    }

    let from_state = evaluate_state(local_row, 0u);
    var state = from_state;
    if (params.blend_alpha > 0.0) {
        let to_state = evaluate_state(local_row, 1u);
        state = blend_states(from_state, to_state, params.blend_alpha);
    }
    let canonical = canonical_rows[local_row];
    if (!finite_f32(state.position.x)
        || !finite_f32(state.position.y)
        || !finite_f32(state.position.z)) {
        state.position = canonical.position.xyz;
    }
    var sigma = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var covariance_is_dynamic = false;
    if (finite_vec3(state.log_scale) && valid_quaternion(state.rotation)) {
        sigma = covariance(state.rotation, state.log_scale);
        covariance_is_dynamic = covariance_packable(sigma);
    }
    if (!covariance_is_dynamic) {
        state.log_scale = canonical.log_scale.xyz;
        if (finite_vec3(state.log_scale) && valid_quaternion(state.rotation)) {
            sigma = covariance(state.rotation, state.log_scale);
            covariance_is_dynamic = covariance_packable(sigma);
        }
    }
    if (global_row == params.diagnostic_row) {
        diagnostic_state = state;
    }
    textureStore(
        output_texture,
        first_coord,
        vec4(
            bitcast<u32>(state.position.x),
            bitcast<u32>(state.position.y),
            bitcast<u32>(state.position.z),
            base_first.w,
        ),
    );
    if (covariance_is_dynamic) {
        textureStore(
            output_texture,
            second_coord,
            vec4(
                pack2x16float(vec2(4.0 * sigma[0], 4.0 * sigma[1])),
                pack2x16float(vec2(4.0 * sigma[2], 4.0 * sigma[3])),
                pack2x16float(vec2(4.0 * sigma[4], 4.0 * sigma[5])),
                base_second.w,
            ),
        );
    } else {
        textureStore(output_texture, second_coord, base_second);
    }
}
