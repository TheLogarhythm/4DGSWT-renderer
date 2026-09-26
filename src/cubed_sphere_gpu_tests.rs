//! Compare the exact shader helper compiled into the renderer with CPU mapping.
use crate::{
    cubed_sphere,
    test_support::gpu::{GpuTestContext, MissingGpu, read_buffer},
    utils::*,
};
use wgpu::util::DeviceExt;

#[test]
fn cubed_sphere_gpu_matches_cpu_at_all_faces_seams_and_heights() {
    let GpuTestContext { device, queue, .. } = GpuTestContext::new(
        "cubed sphere",
        MissingGpu::Fail,
        wgpu::PowerPreference::None,
        |_| wgpu::Features::empty(),
    )
    .unwrap();
    let mut probes = Vec::<[u32; 8]>::new();
    for face in 0..6 {
        for n in [1, 3, 8] {
            for uv in [
                vec2(0., 0.),
                vec2(1., 1.),
                vec2(0., 0.73),
                vec2(0.32, 0.46),
                vec2(-0.02, 1.03),
            ] {
                for h in [0., 1.7] {
                    probes.push([
                        face,
                        n,
                        0,
                        0,
                        (uv.x * n as f32 * 4.).to_bits(),
                        (uv.y * n as f32 * 4.).to_bits(),
                        f32::to_bits(h),
                        0,
                    ]);
                }
            }
        }
    }
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&probes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: probes.len() as u64 * 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let source = format!(
        "{}\n{}",
        include_str!("cubed_sphere.wgsl"),
        r#"
struct Probe { face_info: vec4<u32>, position: vec4<f32> }
@group(0) @binding(0) var<storage, read> inputs: array<Probe>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p=inputs[id.x]; let result=cube_map(p.face_info.x,p.position.xyz,p.face_info.y,4.0,20.0);
    outputs[id.x*4u]=vec4(result.position,1.0);
    outputs[id.x*4u+1u]=vec4(result.jacobian[0],0.0);
    outputs[id.x*4u+2u]=vec4(result.jacobian[1],0.0);
    outputs[id.x*4u+3u]=vec4(result.jacobian[2],0.0);
}"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(probes.len() as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let bytes = read_buffer(&device, &queue, &output, 0, output.size());
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (probe, actual) in probes.iter().zip(floats.chunks_exact(16)) {
        let (p, j) = cubed_sphere::map(
            probe[0] as usize,
            vec3(
                f32::from_bits(probe[4]),
                f32::from_bits(probe[5]),
                f32::from_bits(probe[6]),
            ),
            probe[1] as usize,
            4.,
            20.,
        );
        for (i, expected) in [p, j.x, j.y, j.z].iter().enumerate() {
            let got = vec3(actual[i * 4], actual[i * 4 + 1], actual[i * 4 + 2]);
            assert!(
                (got - *expected).magnitude() < 2e-5,
                "probe {probe:?} vector {i}: {got:?} != {expected:?}"
            );
        }
    }
}
