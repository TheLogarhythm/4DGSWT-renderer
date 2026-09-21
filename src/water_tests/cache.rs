//! Cache validity, pixel centers, full precision, resize and miss overwrite.
use super::harness::Harness;
use crate::{test_support::gpu::read_texture_bytes, utils::*};
#[test]
fn water_cache_refreshes_pixels_after_camera_phase_and_viewport_changes() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.varied_waves = true;
    for (size, eye, target, amplitude, all_miss) in [
        ([128, 128], vec3(0.0, -4.0, 3.0), Vec3::zero(), 0.0, false),
        (
            [65, 33],
            vec3(0.0, -4.0, 0.15),
            vec3(0.0, 4.0, 0.0),
            0.14,
            false,
        ),
        (
            [65, 33],
            vec3(30.0, 0.0, 4.0),
            vec3(30.0, 0.0, 0.0),
            0.14,
            true,
        ),
        ([33, 65], vec3(0.0, -4.0, 2.0), Vec3::zero(), 0.08, false),
    ] {
        h.resize(size);
        h.camera.set_view(eye, target, vec3(0.0, 1.0, 0.0));
        h.data.render_config.water.amplitude = amplitude;
        h.data.render_config.water.advance(0.37);
        h.frame();
        let hits = h.water.hits().unwrap();
        assert_eq!(hits.size, size);
        let data = read_texture_bytes(&h.device, &h.queue, &hits.texture, size);
        let values: &[[f32; 2]] = bytemuck::cast_slice(&data);
        assert_eq!(values.len(), (size[0] * size[1]) as usize);
        let mut valid = 0;
        let phases = h.data.render_config.water.phases();
        for (i, hit) in values.iter().enumerate() {
            assert!(hit[0].is_finite() && hit[1].is_finite());
            if hit[1] == 0.0 {
                continue;
            }
            valid += 1;
            let pixel = [
                (i % size[0] as usize) as f32 + 0.5,
                (i / size[0] as usize) as f32 + 0.5,
            ];
            let ray = h.camera.world_ray(pixel).unwrap();
            let direction = ray.direction / ray.direction.dot(h.camera.view_direction());
            let p = ray.origin + direction * hit[1];
            let components = [
                (0.946, 0.324, 0.830, 0.32, 0.43),
                (0.544, -0.839, 1.271, 0.28, 2.17),
                (-0.771, 0.637, 1.913, 0.23, 4.61),
                (-0.217, -0.976, 2.731, 0.17, 1.29),
            ];
            let height = components
                .iter()
                .enumerate()
                .map(|(j, &(dx, dy, k, a, offset))| {
                    amplitude as f64
                        * a
                        * (std::f64::consts::TAU / 4.0 * k * (dx * p.x as f64 + dy * p.y as f64)
                            - phases[j] as f64
                            + offset)
                            .sin()
                })
                .sum::<f64>();
            assert!(
                (p.z as f64 - height).abs() < 0.0002,
                "wrong cached surface {p:?}, expected z {height}"
            );
            let clip = h.camera.view_proj() * p.extend(1.0);
            assert!(
                (hit[0] - (clip.z / clip.w * 0.5 + 0.5)).abs() < 0.000005,
                "cached view distance and depth disagree"
            );
        }
        if all_miss {
            assert_eq!(
                valid, 0,
                "old hits survived a camera move outside water coverage"
            );
        } else {
            assert!(valid > 0);
        }
    }
    h.data.render_config.water.enabled = false;
    let dry = h.frame();
    h.data.render_config.water.advance(1.0);
    assert_eq!(
        h.frame(),
        dry,
        "disabling water cannot consume its last cached frame"
    );
}

#[test]
fn water_flat_edge_lighting_matches_the_same_surface_with_larger_coverage() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.ripple_strength = 0.6;
    h.data.render_config.water.ripple_scale = 2.0;
    h.camera
        .set_view(vec3(0.0, 0.0, 20.0), Vec3::zero(), vec3(0.0, 1.0, 0.0));
    let finite = h.frame();
    h.user.tile_map_half_wh = vec2(2, 2);
    let larger = h.frame();
    for y in 9..119 {
        for x in 9..119 {
            assert_eq!(
                finite.pixel(x, y),
                larger.pixel(x, y),
                "valid flat-water lighting changed at pixel {x},{y} when only coverage changed"
            );
        }
    }
}
