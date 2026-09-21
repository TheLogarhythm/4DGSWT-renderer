// The dry entry point cannot modify depth or reach any water code.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let A = -dot(in.v_position, in.v_position);
    if A < -4.0 { discard; }
    let alpha = exp(A) * in.v_color.a;
    return vec4(alpha * in.v_color.rgb, alpha);
}
