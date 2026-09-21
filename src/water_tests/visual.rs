//! Opt-in local archive acceptance; separate from synthetic regression cases.
use super::{fixtures::SceneFixture, harness::Harness};
use crate::{camera::Camera, scene::Scene, utils::*};

/// Opt-in visual acceptance with a real local GS archive, never a network asset.
#[test]
#[ignore = "set WATER_REVIEW_ARCHIVE and WATER_REVIEW_DIR to export local review PNGs"]
fn water_export_review_frames() {
    use std::io::Read;
    let archive = std::env::var("WATER_REVIEW_ARCHIVE").expect("set WATER_REVIEW_ARCHIVE");
    let output =
        std::path::PathBuf::from(std::env::var("WATER_REVIEW_DIR").expect("set WATER_REVIEW_DIR"));
    std::fs::create_dir_all(&output).unwrap();
    let mut zip = zip::ZipArchive::new(std::fs::File::open(archive).unwrap()).unwrap();
    let mut bytes = Vec::new();
    zip.by_name("tile0_lod1.ply")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let scene = Scene::from_ply_bytes(bytes).unwrap();
    let (aabb, mean) = scene.compute_aabb_and_center();
    let mut heights: Vec<f32> = scene
        .buffer
        .chunks_exact(32)
        .map(|row| f32::from_ne_bytes(row[8..12].try_into().unwrap()))
        .collect();
    heights.sort_by(f32::total_cmp);
    let low = heights[heights.len() / 20];
    let high = heights[heights.len() * 95 / 100];
    let span = (high - low).max(0.1);
    let center = vec3(mean.x, mean.y, low + span * 0.5);
    let width = (aabb.1.x - aabb.0.x).max(aabb.1.y - aabb.0.y).max(1.0);
    let camera = Camera::new_perspective(
        winit::dpi::PhysicalSize::new(768, 768),
        center + vec3(0.1 * width, -1.35 * width, 0.95 * width),
        center,
        vec3(0.0, 0.0, 1.0),
        degrees(50.0),
        0.1,
        2400.0,
    );
    let fixture = SceneFixture::from_scene(scene, &camera);
    let count = fixture.scene.splat_count;
    let mut h = Harness::from_scene(fixture, [768, 768]);
    h.camera = camera;
    if let Ok(path) = std::env::var("WATER_REVIEW_SKY") {
        let sky = image::open(path).unwrap().to_rgba32f();
        let wh = vec2(sky.width() as usize, sky.height() as usize);
        h.skybox.configure(
            &h.device,
            &h.queue,
            &(crate::skybox::SkyboxTexture::HDRI(sky.into_raw()), wh),
        );
        h.data.use_skybox = true;
    }
    let dry = h.frame();
    image::save_buffer(
        output.join("water-off.png"),
        &dry.pixels,
        768,
        768,
        image::ColorType::Rgba8,
    )
    .unwrap();
    for (name, height) in [
        ("water-low", low + span * 0.15),
        ("water-mid", low + span * 0.45),
        ("water-high", low + span * 0.75),
    ] {
        h.data.render_config.water.enabled = true;
        h.data.render_config.water.height = height;
        let frame = h.frame();
        image::save_buffer(
            output.join(format!("{name}.png")),
            &frame.pixels,
            768,
            768,
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
    h.data.render_config.water.varied_waves = true;
    h.data.render_config.water.ripple_strength = 0.22;
    // Animated review uses the same real scene and camera as the static images.
    h.data.render_config.water.height = low + span * 0.45;
    h.data.render_config.water.amplitude = 0.08;
    for i in 0..48 {
        let frame = h.frame();
        image::save_buffer(
            output.join(format!("wave-{i:03}.png")),
            &frame.pixels,
            768,
            768,
            image::ColorType::Rgba8,
        )
        .unwrap();
        h.data.render_config.water.advance(0.125);
    }
    let comparison_water = h.data.render_config.water.clone();
    for (name, varied, reflection, ripples) in [
        ("previous", false, 0.0, 0.0),
        ("reflection", false, 1.0, 0.0),
        ("full", true, 1.0, 0.22),
    ] {
        h.data.render_config.water = comparison_water.clone();
        h.data.render_config.water.amplitude = 0.08;
        h.data.render_config.water.varied_waves = varied;
        h.data.render_config.water.reflection_strength = reflection;
        h.data.render_config.water.ripple_strength = ripples;
        let frame = h.frame();
        image::save_buffer(
            output.join(format!("flowers-{name}.png")),
            &frame.pixels,
            768,
            768,
            image::ColorType::Rgba8,
        )
        .unwrap();
        let mut times = Vec::new();
        for i in 0..35 {
            h.data.render_config.water.advance(1.0 / 60.0);
            if let Some(ms) = h.time_frame_ms() {
                if i >= 5 {
                    times.push(ms);
                }
            }
        }
        if !times.is_empty() {
            times.sort_by(f64::total_cmp);
            eprintln!(
                "water GPU 768x768 real Flowers {name}: median {:.3} ms, p95 {:.3} ms",
                times[times.len() / 2],
                times[times.len() * 95 / 100]
            );
        }
    }
    h.data.render_config.water.enabled = false;
    assert_eq!(h.frame(), dry);
    // A low camera over continuous water makes spacing and reflection easy to inspect.
    h.user.tile_map_half_wh = vec2(64, 64);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 4.0;
    h.camera.set_view(
        vec3(0.0, -7.0, 5.0),
        vec3(0.0, 5.0, 4.7),
        vec3(0.0, 0.0, 1.0),
    );
    for (name, varied, reflection, ripples) in [
        ("previous", false, 0.0, 0.0),
        ("reflection", false, 1.0, 0.0),
        ("full", true, 1.0, 0.22),
    ] {
        h.data.render_config.water.varied_waves = varied;
        h.data.render_config.water.reflection_strength = reflection;
        h.data.render_config.water.ripple_strength = ripples;
        let frame = h.frame();
        image::save_buffer(
            output.join(format!("ocean-{name}.png")),
            &frame.pixels,
            768,
            768,
            image::ColorType::Rgba8,
        )
        .unwrap();
    }
    for i in 0..32 {
        let frame = h.frame();
        image::save_buffer(
            output.join(format!("ocean-{i:03}.png")),
            &frame.pixels,
            768,
            768,
            image::ColorType::Rgba8,
        )
        .unwrap();
        h.data.render_config.water.advance(0.125);
    }
    eprintln!(
        "water review: {count} real splats; bounds {aabb:?}; water range {low}..{high}; outputs {}",
        output.display()
    );
}
