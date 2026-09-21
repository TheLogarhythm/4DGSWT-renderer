//! Compare the production GPU intersection to a dense, independent CPU scan.
use super::harness::Harness;
use crate::{camera::CameraUniforms, test_support::gpu::read_buffer, utils::*};
use wgpu::util::DeviceExt;

#[test]
fn water_gpu_grazing_intersections_find_the_nearest_wave() {
    check_grazing(false);
    check_grazing(true);
}
fn check_grazing(varied: bool) {
    let mut h = Harness::new();
    let shader = h.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wave intersection acceptance"),
        source: wgpu::ShaderSource::Wgsl(crate::water::shader_source(r#"
            @group(0) @binding(0) var<uniform> camera: CameraUniforms;
            @group(0) @binding(1) var<storage, read_write> hits: array<vec4<f32>>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                if id.x >= 256u { return; }
                let pixel = vec2(f32(id.x % 16u) * 8.0 + 4.5, f32(id.x / 16u) * 4.0 + 40.5);
                hits[id.x] = water_surface_hit(pixel, camera, 0.0, vec4(-10.0,-10.0,10.0,10.0), vec4(0.14,1.57079632679,VARIATION,0.0), vec4(0.7,0.917,1.211,1.4819));
            }
        "#).replace("VARIATION",if varied {"1.0"} else {"0.0"}).into()),
    });
    let pipeline = h
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    for z in [2.0, 0.15, 0.05, -0.05] {
        h.camera
            .set_view(vec3(0.0, -4.0, z), vec3(0.0, 1.0, z), vec3(0.0, 0.0, 1.0));
        let camera = h
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&CameraUniforms::from_camera(&h.camera)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output = h.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = h.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(4, 1, 1);
        }
        h.queue.submit(Some(encoder.finish()));
        let bytes = read_buffer(&h.device, &h.queue, &output, 0, 4096);
        let hits: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
        for (i, hit) in hits.iter().enumerate() {
            let pixel = [(i % 16) as f32 * 8.0 + 4.5, (i / 16) as f32 * 4.0 + 40.5];
            let ray = h.camera.world_ray(pixel).unwrap();
            let direction = ray.direction / ray.direction.dot(h.camera.view_direction());
            // Uniform 0.001 world-view-depth scan + bisection is independent of
            // the shader's conservative stepping and convergence criterion.
            let value = |t: f64| {
                let x = ray.origin.x as f64 + direction.x as f64 * t;
                let y = ray.origin.y as f64 + direction.y as f64 * t;
                let k = std::f64::consts::TAU / 4.0;
                let height = if varied {
                    [
                        (0.946, 0.324, 0.830, 0.32, 0.43, 0.7),
                        (0.544, -0.839, 1.271, 0.28, 2.17, 0.917),
                        (-0.771, 0.637, 1.913, 0.23, 4.61, 1.211),
                        (-0.217, -0.976, 2.731, 0.17, 1.29, 1.4819),
                    ]
                    .iter()
                    .map(|&(dx, dy, f, a, offset, phase)| {
                        0.14 * a * (k * f * (dx * x + dy * y) - phase + offset).sin()
                    })
                    .sum()
                } else {
                    0.14 * (0.55 * (k * x - 0.7).sin()
                        + 0.30 * (k * 1.6 * (0.6 * x + 0.8 * y) - 0.917).sin()
                        + 0.15 * (k * 2.7 * (-0.8 * x + 0.6 * y) - 1.211).sin())
                };
                (
                    ray.origin.z as f64 + direction.z as f64 * t - height,
                    x.abs() <= 10.0 && y.abs() <= 10.0,
                )
            };
            let mut reference = None;
            let mut prev = value(0.1);
            for step in 101..25000 {
                let t = step as f64 * 0.001;
                let next = value(t);
                if next.1 && prev.1 && next.0 * prev.0 <= 0.0 {
                    let mut a = t - 0.001;
                    let mut b = t;
                    for _ in 0..20 {
                        let m = (a + b) * 0.5;
                        if value(a).0 * value(m).0 <= 0.0 {
                            b = m;
                        } else {
                            a = m;
                        }
                    }
                    reference = Some((a + b) * 0.5);
                    break;
                }
                prev = next;
            }
            if let Some(expected) = reference {
                assert!(
                    hit[1] > 0.5,
                    "missed wave at camera z={z}, pixel {pixel:?}, t={expected}"
                );
                assert!(
                    (hit[3] as f64 - expected).abs() < 0.025,
                    "wrong first hit at z={z}, pixel {pixel:?}: GPU {}, reference {expected}",
                    hit[3]
                );
            } else if hit[1] > 0.5 {
                assert!(
                    value(hit[3] as f64).0.abs() < 0.0001,
                    "false wave at {pixel:?}"
                );
            }
        }
    }
}

#[test]
fn water_gpu_short_wavelength_grazing_regression() {
    let mut h = Harness::new();
    let shader=h.device.create_shader_module(wgpu::ShaderModuleDescriptor {label:None,source:wgpu::ShaderSource::Wgsl(crate::water::shader_source(r#"
        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<storage, read_write> hit: vec4<f32>;
        @compute @workgroup_size(1) fn main() {
            hit=water_surface_hit(vec2(64.0,64.0),camera,0.0,vec4(-10.0,-10.0,10.0,10.0),vec4(0.0035,62.8318530718,0.0,0.0),vec4(0.0));
        }
    "#).into())});
    let pipeline = h
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    // Independently scanned/bracketed first crossings from the review repro.
    for (slope, expected) in [(-0.001, 1.6704785), (-0.0001, 16.049476)] {
        let origin = vec3(0.0, -9.0, 0.00315);
        h.camera
            .set_view(origin, origin + vec3(0.0, 1.0, slope), vec3(0.0, 0.0, 1.0));
        let camera = h
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&CameraUniforms::from_camera(&h.camera)),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output = h.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = h.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = h.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        h.queue.submit(Some(encoder.finish()));
        let bytes = read_buffer(&h.device, &h.queue, &output, 0, 16);
        let hit: &[f32] = bytemuck::cast_slice(&bytes);
        assert!(
            hit[1] > 0.5,
            "short wavelength missed first crossing at t={expected}"
        );
        assert!(
            (hit[3] - expected).abs() < 0.025,
            "wrong short-wave root {}, expected {expected}",
            hit[3]
        );
    }
}
