//! Observable environment/material behavior through the real GPU water pass.
use super::harness::Harness;
use crate::{skybox::SkyboxTexture, utils::*};

fn solid_sky(h: &mut Harness, color: [f32; 4]) {
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(SkyboxTexture::Cubemap(vec![color.to_vec(); 6]), vec2(1, 1)),
    );
    h.data.use_skybox = true;
}
#[test]
fn water_gpu_reflection_tracks_replaced_environment() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.2;
    solid_sky(&mut h, [1.0, 0.0, 0.0, 1.0]);
    let red = h.frame().pixel(8, 8);
    solid_sky(&mut h, [0.0, 0.0, 1.0, 1.0]);
    let blue = h.frame().pixel(8, 8);
    assert!(
        red[0] > blue[0] + 10 && blue[2] > red[2],
        "water must reflect a replaced environment: {red:?} -> {blue:?}"
    );
}

#[test]
fn water_gpu_fresnel_and_reflection_off_are_observable() {
    let mut h = Harness::new();
    solid_sky(&mut h, [1.0; 4]);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.2;
    h.data.render_config.water.color = [0.0; 3];
    let overhead = h.frame().pixel(64, 64)[0];
    h.camera.set_view(
        vec3(0.0, -4.0, 1.6),
        vec3(0.0, 0.0, 1.2),
        vec3(0.0, 0.0, 1.0),
    );
    let grazing = h.frame().pixel(64, 64)[0];
    assert!(
        overhead < 60 && grazing > 160,
        "Fresnel: overhead {overhead}, grazing {grazing}"
    );
    h.data.render_config.water.reflection_strength = 0.0;
    assert_eq!(h.frame().pixel(64, 64), [0, 0, 0]);
    h.data.render_config.water.reflection_strength = 1.0;
    h.data.use_skybox = false;
    assert_eq!(
        h.frame().pixel(64, 64),
        [0, 0, 0],
        "disabled sky cannot leave stale reflections"
    );
}
#[test]
fn water_gpu_cubemap_reflection_matches_world_orientation() {
    let mut h = Harness::new();
    let faces = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ];
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(
            SkyboxTexture::Cubemap(faces.map(|x| x.repeat(16 * 16)).to_vec()),
            vec2(16, 16),
        ),
    );
    h.data.use_skybox = true;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.2;
    h.data.render_config.water.color = [0.0; 3];
    h.data.render_config.water.roughness = 0.0;
    for (eye, expected) in [
        (vec3(-4.0, 0.0, 3.2), [true, false, false]),
        (vec3(4.0, 0.0, 3.2), [false, true, false]),
        (vec3(0.0, -4.0, 3.2), [true, true, false]),
        (vec3(0.0, 4.0, 3.2), [false, true, true]),
        (vec3(0.01, 0.0, 5.2), [false, false, true]),
    ] {
        h.camera
            .set_view(eye, vec3(0.0, 0.0, 1.2), vec3(0.0, 0.0, 1.0));
        let color = h.frame().pixel(64, 64);
        for c in 0..3 {
            assert!(
                if expected[c] {
                    color[c] > 30
                } else {
                    color[c] < 12
                },
                "wrong reflected direction {eye:?}: {color:?}"
            );
        }
    }
}
#[test]
fn water_gpu_roughness_blurs_environment_detail() {
    let mut h = Harness::new();
    let mut face = Vec::new();
    for y in 0..64 {
        for x in 0..64 {
            let c = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.0 };
            face.extend_from_slice(&[c, c, c, 1.0]);
        }
    }
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(SkyboxTexture::Cubemap(vec![face; 6]), vec2(64, 64)),
    );
    h.data.use_skybox = true;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.2;
    h.data.render_config.water.color = [0.0; 3];
    let mut variances = Vec::new();
    for roughness in [0.0, 0.9] {
        h.data.render_config.water.roughness = roughness;
        let frame = h.frame();
        let values: Vec<f64> = (24..104)
            .flat_map(|y| (24..104).map(move |x| (x, y)))
            .map(|(x, y)| frame.pixel(x, y)[0] as f64)
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        variances
            .push(values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64);
    }
    assert!(
        variances[0] > 20.0 && variances[1] < variances[0] * 0.3,
        "roughness variance {variances:?}"
    );
}
#[test]
fn water_gpu_ripples_animate_without_changing_gs_contact_and_survive_streaming() {
    let mut h = Harness::new();
    solid_sky(&mut h, [0.0, 1.0, 0.0, 1.0]);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 0.6;
    h.data.render_config.water.color = [0.0, 0.24, 0.38];
    let smooth = h.frame();
    h.data.render_config.water.ripple_strength = 0.4;
    let detailed = h.frame();
    assert_ne!(smooth, detailed);
    // Red belongs only to the crossing GS. Lighting may change G/B, never its coverage.
    for (a, b) in smooth
        .pixels
        .chunks_exact(4)
        .zip(detailed.pixels.chunks_exact(4))
    {
        assert_eq!(a[0], b[0]);
    }
    h.data.render_config.water.advance(0.75);
    let moved = h.frame();
    assert_ne!(
        moved, detailed,
        "fine normals must animate with flat geometry"
    );
    for (a, b) in moved
        .pixels
        .chunks_exact(4)
        .zip(detailed.pixels.chunks_exact(4))
    {
        assert_eq!(a[0], b[0]);
    }
    h.data.cur_scene_data.as_mut().unwrap().center_coord.x = 1;
    assert_eq!(
        h.frame(),
        moved,
        "streaming cannot restart or re-anchor detail"
    );
    h.data.render_config.water.playing = false;
    h.data.render_config.water.advance(10.0);
    assert_eq!(h.frame(), moved);
    h.data.render_config.water.ripple_strength = 0.0;
    assert_eq!(h.frame(), smooth);
}
#[test]
fn water_gpu_subpixel_ripples_are_filtered_out() {
    let mut h = Harness::new();
    solid_sky(&mut h, [1.0; 4]);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.2;
    h.user.tile_map_half_wh = vec2(64, 64);
    h.camera = crate::camera::Camera::new_perspective(
        winit::dpi::PhysicalSize::new(128, 128),
        vec3(0.0, 0.0, 100.0),
        Vec3::zero(),
        vec3(0.0, 1.0, 0.0),
        degrees(60.0),
        0.1,
        1000.0,
    );
    let smooth = h.frame();
    h.data.render_config.water.ripple_strength = 1.0;
    assert_eq!(
        h.frame(),
        smooth,
        "subpixel detail must not alias at distance"
    );
}

#[test]
fn water_gpu_hdri_reflection_matches_the_displayed_sky() {
    let mut h = Harness::new();
    let mut tex = Vec::new();
    for y in 0..32 {
        for x in 0..64 {
            tex.extend_from_slice(&[
                0.05 + 2.0 * x as f32 / 63.0,
                0.1 + 2.0 * y as f32 / 31.0,
                0.15 + 1.5 * (31 - y) as f32 / 31.0,
                1.0,
            ]);
        }
    }
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(SkyboxTexture::HDRI(tex), vec2(64, 32)),
    );
    h.data.use_skybox = true;
    h.data.render_config.water.height = 1.2;
    h.data.render_config.water.color = [0.0; 3];
    h.data.render_config.water.roughness = 0.0;
    for direction in [
        vec3(1.0, 0.0, 1.0),
        vec3(0.0, 1.0, 1.0),
        vec3(-1.0, 0.0, 0.5),
        vec3(0.0, -1.0, 1.0),
    ] {
        h.data.render_config.water.enabled = false;
        let eye = vec3(0.0, 0.0, 10.0);
        h.camera.set_view(eye, eye + direction, vec3(0.0, 0.0, 1.0));
        let sky = h.frame().pixel(64, 64);
        h.data.render_config.water.enabled = true;
        let eye = vec3(
            -direction.x * 4.0,
            -direction.y * 4.0,
            1.2 + direction.z * 4.0,
        );
        h.camera
            .set_view(eye, vec3(0.0, 0.0, 1.2), vec3(0.0, 0.0, 1.0));
        let reflected = h.frame().pixel(64, 64);
        let total_sky: f32 = sky.iter().map(|&x| x as f32).sum();
        let total_reflected: f32 = reflected.iter().map(|&x| x as f32).sum();
        assert!(total_reflected > 20.0);
        for c in 0..3 {
            assert!(
                (sky[c] as f32 / total_sky - reflected[c] as f32 / total_reflected).abs() < 0.035,
                "HDR orientation {direction:?}: sky {sky:?}, water {reflected:?}"
            );
        }
    }
}
