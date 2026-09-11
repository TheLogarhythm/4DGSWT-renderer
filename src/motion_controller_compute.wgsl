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

struct AuthoredOccurrenceInputGpu {
    local_row_output: vec4<u32>,
    occurrence_offset: vec4<f32>,
}

struct ControllerMetadataGpu {
    translation_master_gain: vec4<f32>,
    rotation_scale_blend_valid: vec4<f32>,
    padding: vec4<f32>,
}

struct ControllerMetadataTable {
    entries: array<ControllerMetadataGpu, 65>,
}

struct AuthoredParamsGpu {
    input_count: u32,
    row_start: u32,
    top_k: u32,
    basis_count: u32,
    texture_width: u32,
    output_width: u32,
    motion_enabled: u32,
    preview_channel_mask: u32,
    coefficient_bank_count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

struct MotionFieldUniform {
    world_origin: vec2<f32>,
    reciprocal_world_size: vec2<f32>,
    texture_size: vec2<u32>,
    enabled: u32,
    overlay_channel: u32,
    palette_count: u32,
    texels_per_tile: u32,
    storage_offset: vec2<u32>,
    preview_center: vec2<f32>,
    preview_radius: f32,
    preview_opacity: f32,
    preview_color: vec4<f32>,
    preview_active: u32,
    preview_falloff: u32,
    preview_tool: u32,
    preview_padding: u32,
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

struct FieldSample {
    continuous: vec4<f32>,
    slot: u32,
    hit: u32,
}

@group(0) @binding(0) var<storage, read> canonical_rows: array<CanonicalGpu>;
@group(0) @binding(1) var<storage, read> basis_ids: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> transform_ids: array<u32>;
@group(0) @binding(4) var<storage, read> controller_basis_frames: array<BasisGpu>;
@group(0) @binding(5) var<storage, read> placements: array<PlacementGpu>;
@group(0) @binding(6) var<storage, read> motion_channel_masks: array<u32>;
@group(0) @binding(7) var<storage, read> occurrence_inputs: array<AuthoredOccurrenceInputGpu>;
@group(0) @binding(8) var inherited_texture: texture_2d<u32>;
@group(0) @binding(9) var continuous_field: texture_2d<f32>;
@group(0) @binding(10) var assignment_field: texture_2d<u32>;
@group(0) @binding(11) var<uniform> field: MotionFieldUniform;
@group(0) @binding(12) var<uniform> controllers: ControllerMetadataTable;
@group(0) @binding(13) var output_texture: texture_storage_2d<rgba32uint, write>;
@group(0) @binding(14) var<uniform> params: AuthoredParamsGpu;

fn inherited_coord(pixel_index: u32) -> vec2<i32> {
    return vec2(i32(pixel_index % params.texture_width), i32(pixel_index / params.texture_width));
}

fn output_coord(pixel_index: u32) -> vec2<i32> {
    return vec2(i32(pixel_index % params.output_width), i32(pixel_index / params.output_width));
}

fn copy_inherited(global_row: u32, output_index: u32) {
    let inherited_first = textureLoad(inherited_texture, inherited_coord(global_row * 2u), 0);
    let inherited_second = textureLoad(inherited_texture, inherited_coord(global_row * 2u + 1u), 0);
    textureStore(output_texture, output_coord(output_index * 2u), inherited_first);
    textureStore(output_texture, output_coord(output_index * 2u + 1u), inherited_second);
}

fn sample_field(world_xy: vec2<f32>) -> FieldSample {
    var result: FieldSample;
    result.continuous = vec4(0.0, 0.5, 0.0, 1.0);
    result.slot = 0u;
    result.hit = 0u;
    if (field.enabled == 0u) {
        return result;
    }
    let uv = (world_xy - field.world_origin) * field.reciprocal_world_size;
    if (!all(uv >= vec2(0.0)) || !all(uv < vec2(1.0))) {
        return result;
    }
    let logical = vec2<u32>(uv * vec2<f32>(field.texture_size));
    let physical = vec2<i32>((logical + field.storage_offset) % field.texture_size);
    result.continuous = textureLoad(continuous_field, physical, 0);
    result.slot = textureLoad(assignment_field, physical, 0).r;
    result.hit = 1u;
    return result;
}

fn reconstruct_motion(local_row: u32, controller_slot: u32, frame_index: u32) -> MotionVector {
    var result: MotionVector;
    result.translation = vec3(0.0);
    result.rotation = vec3(0.0);
    result.log_scale = vec3(0.0);
    let frame_stride = params.coefficient_bank_count * params.basis_count;
    let controller_stride = 2u * frame_stride;
    let frame_offset = controller_slot * controller_stride + frame_index * frame_stride;
    if (params.coefficient_bank_count == 3u) {
        for (var coefficient_slot = 0u; coefficient_slot < params.top_k; coefficient_slot += 1u) {
            let coefficient_index = (local_row * 3u) * params.top_k + coefficient_slot;
            let basis_id = basis_ids[coefficient_index];
            result.translation += weights[coefficient_index]
                * controller_basis_frames[frame_offset + basis_id].translation.xyz;
        }
        for (var coefficient_slot = 0u; coefficient_slot < params.top_k; coefficient_slot += 1u) {
            let coefficient_index = (local_row * 3u + 1u) * params.top_k + coefficient_slot;
            let basis_id = basis_ids[coefficient_index];
            result.rotation += weights[coefficient_index]
                * controller_basis_frames[frame_offset + params.basis_count + basis_id].rotation.xyz;
        }
        for (var coefficient_slot = 0u; coefficient_slot < params.top_k; coefficient_slot += 1u) {
            let coefficient_index = (local_row * 3u + 2u) * params.top_k + coefficient_slot;
            let basis_id = basis_ids[coefficient_index];
            result.log_scale += weights[coefficient_index]
                * controller_basis_frames[frame_offset + 2u * params.basis_count + basis_id].log_scale.xyz;
        }
        return result;
    }
    for (var coefficient_slot = 0u; coefficient_slot < params.top_k; coefficient_slot += 1u) {
        let coefficient_index = local_row * params.top_k + coefficient_slot;
        let basis_id = basis_ids[coefficient_index];
        let basis = controller_basis_frames[frame_offset + basis_id];
        let weight = weights[coefficient_index];
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
        return vec4(
            1.0 - theta_squared / 8.0 + theta_fourth / 384.0,
            rotation_vector * (0.5 - theta_squared / 48.0 + theta_fourth / 3840.0),
        );
    }
    let theta = sqrt(theta_squared);
    let half_theta = 0.5 * theta;
    return vec4(cos(half_theta), rotation_vector * (sin(half_theta) / theta));
}

fn finite_f32(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return finite_f32(value.x) && finite_f32(value.y) && finite_f32(value.z);
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return finite_f32(value.x) && finite_f32(value.y)
        && finite_f32(value.z) && finite_f32(value.w);
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

fn evaluate_frame(local_row: u32, controller_slot: u32, frame_index: u32, strength: f32) -> GaussianState {
    let canonical = canonical_rows[local_row];
    let metadata = controllers.entries[controller_slot];
    var motion = reconstruct_motion(local_row, controller_slot, frame_index);
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
    let master = metadata.translation_master_gain.w * strength;
    let translation = transform_translation(motion.translation, transform_ids[local_row])
        * metadata.translation_master_gain.xyz * master;
    let rotation = motion.rotation * metadata.rotation_scale_blend_valid.x * master;
    let log_scale = motion.log_scale * metadata.rotation_scale_blend_valid.y * master;
    var state: GaussianState;
    state.position = canonical.position.xyz + translation;
    state.log_scale = canonical.log_scale.xyz + log_scale;
    state.rotation = standardized_quaternion(quaternion_multiply(
        standardized_quaternion(canonical.rotation),
        quaternion_exp(rotation),
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

fn evaluate_controller(local_row: u32, controller_slot: u32, strength: f32) -> GaussianState {
    let metadata = controllers.entries[controller_slot];
    let from_state = evaluate_frame(local_row, controller_slot, 0u, strength);
    if (metadata.rotation_scale_blend_valid.z <= 0.0) {
        return from_state;
    }
    let to_state = evaluate_frame(local_row, controller_slot, 1u, strength);
    return blend_states(from_state, to_state, metadata.rotation_scale_blend_valid.z);
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

fn valid_state(state: GaussianState, sigma: array<f32, 6>) -> bool {
    if (!finite_vec3(state.position) || !finite_vec3(state.log_scale) || !finite_vec4(state.rotation)) {
        return false;
    }
    let rotation_length = dot(state.rotation, state.rotation);
    if (!finite_f32(rotation_length) || rotation_length <= 0.0) {
        return false;
    }
    for (var index = 0u; index < 6u; index += 1u) {
        let packed = 4.0 * sigma[index];
        if (!finite_f32(packed) || abs(packed) > 65504.0) {
            return false;
        }
    }
    return true;
}

fn decoded_strength(encoded: f32) -> f32 {
    let byte = u32(round(encoded * 255.0));
    if (byte == 255u) {
        return 2.0;
    }
    return f32(byte) / 128.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let input_index = invocation.x;
    if (input_index >= params.input_count) {
        return;
    }
    let input = occurrence_inputs[input_index];
    let local_row = input.local_row_output.x;
    let output_index = input.local_row_output.y;
    let global_row = params.row_start + local_row;
    let authored = sample_field(canonical_rows[local_row].position.xy + input.occurrence_offset.xy);
    if (params.motion_enabled == 0u || authored.hit == 0u || authored.slot == 0u
        || authored.slot > 64u || authored.continuous.r <= 0.0
        || controllers.entries[authored.slot].rotation_scale_blend_valid.w < 0.5) {
        copy_inherited(global_row, output_index);
        return;
    }

    let local_state = evaluate_controller(local_row, authored.slot, decoded_strength(authored.continuous.g));
    var state = local_state;
    if (authored.continuous.r < 1.0) {
        let inherited_state = evaluate_controller(local_row, 0u, 1.0);
        state = blend_states(inherited_state, local_state, clamp(authored.continuous.r, 0.0, 1.0));
    }
    let sigma = covariance(state.rotation, state.log_scale);
    if (!valid_state(state, sigma)) {
        copy_inherited(global_row, output_index);
        return;
    }

    let inherited_first = textureLoad(inherited_texture, inherited_coord(global_row * 2u), 0);
    let inherited_second = textureLoad(inherited_texture, inherited_coord(global_row * 2u + 1u), 0);
    textureStore(
        output_texture,
        output_coord(output_index * 2u),
        vec4(
            bitcast<u32>(state.position.x),
            bitcast<u32>(state.position.y),
            bitcast<u32>(state.position.z),
            inherited_first.w,
        ),
    );
    textureStore(
        output_texture,
        output_coord(output_index * 2u + 1u),
        vec4(
            pack2x16float(vec2(4.0 * sigma[0], 4.0 * sigma[1])),
            pack2x16float(vec2(4.0 * sigma[2], 4.0 * sigma[3])),
            pack2x16float(vec2(4.0 * sigma[4], 4.0 * sigma[5])),
            inherited_second.w,
        ),
    );
}
