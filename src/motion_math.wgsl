// Shared global/local motion math. Reconstruction and containment policies remain separate.

struct GaussianState {
    position: vec3<f32>,
    log_scale: vec3<f32>,
    rotation: vec4<f32>,
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
