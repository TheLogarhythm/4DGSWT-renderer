//! Opt-in visual review using local coral geometry and the real renderer.
use super::{fixtures::SceneFixture, harness::Harness};
use crate::{camera::Camera, scene::Scene, skybox::SkyboxTexture, utils::*};

#[test]
#[ignore = "set UNDERWATER_REVIEW_ARCHIVE and UNDERWATER_REVIEW_DIR to export coral frames"]
fn underwater_export_coral_review() {
    use std::io::Read;
    let archive = std::env::var("UNDERWATER_REVIEW_ARCHIVE").unwrap();
    let output = std::path::PathBuf::from(std::env::var("UNDERWATER_REVIEW_DIR").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let mut zip = zip::ZipArchive::new(std::fs::File::open(archive).unwrap()).unwrap();
    let mut bytes = Vec::new();
    zip.by_name("tile0_lod2.ply")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let tile = Scene::from_ply_bytes(bytes).unwrap();
    let mut heights: Vec<f32> = tile
        .buffer
        .chunks_exact(32)
        .map(|row| f32::from_le_bytes(row[8..12].try_into().unwrap()))
        .collect();
    heights.sort_by(f32::total_cmp);
    let ground = heights[heights.len() / 20];
    let mut scene = Scene::new();
    for x in -3..=3 {
        for y in 0..=8 {
            for row in tile.buffer.chunks_exact(32) {
                let mut copy = row.to_vec();
                for (axis, offset) in [x as f32 * 4.0 - 2.0, y as f32 * 4.0 - 2.0, -ground]
                    .into_iter()
                    .enumerate()
                {
                    let start = axis * 4;
                    let value =
                        f32::from_le_bytes(row[start..start + 4].try_into().unwrap()) + offset;
                    copy[start..start + 4].copy_from_slice(&value.to_le_bytes());
                }
                scene.buffer.extend(copy);
            }
        }
    }
    scene.splat_count = scene.buffer.len() / 32;
    for (prefix, eye, target, water_height) in [
        ("", vec3(0.3, -4.0, 1.8), vec3(0.3, 12.0, 1.0), 2.8),
        ("overview-", vec3(0.3, -4.0, 3.4), vec3(0.3, 12.0, 1.2), 4.5),
        ("upward-", vec3(0.3, -4.0, 1.8), vec3(0.3, 1.0, 6.8), 2.8),
        ("deep-", vec3(0.3, -4.0, 2.0), vec3(0.3, 12.0, 4.0), 8.0),
        ("ground-", vec3(0.3, -4.0, 4.0), vec3(0.3, 2.0, 0.0), 8.0),
    ] {
        if let Ok(view) = std::env::var("UNDERWATER_REVIEW_VIEW") {
            if prefix.trim_end_matches('-') != view {
                continue;
            }
        }
        let camera = Camera::new_perspective(
            winit::dpi::PhysicalSize::new(1280, 720),
            eye,
            target,
            vec3(0.0, 0.0, 1.0),
            degrees(65.0),
            0.1,
            200.0,
        );
        let fixture = SceneFixture::from_scene(scene.clone(), &camera);
        let mut h = Harness::from_scene(fixture, [1280, 720]);
        h.camera = camera;
        h.user.tile_map_half_wh = vec2(12, 12);
        h.user.tile_map_wh = vec2(25, 25);
        h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
        if let Ok(path) = std::env::var("UNDERWATER_REVIEW_SKY") {
            let sky = image::open(path).unwrap().to_rgba32f();
            let wh = vec2(sky.width() as usize, sky.height() as usize);
            h.skybox.configure(
                &h.device,
                &h.queue,
                &(SkyboxTexture::HDRI(sky.into_raw()), wh),
            );
        } else {
            h.skybox.configure(
                &h.device,
                &h.queue,
                &(
                    SkyboxTexture::Cubemap(vec![vec![0.65, 0.78, 0.88, 1.0]; 6]),
                    vec2(1, 1),
                ),
            );
        }
        h.data.use_skybox = true;
        h.data.render_config.water.enabled = true;
        h.data.render_config.water.height = water_height;
        h.data.render_config.water.amplitude = 0.08;
        h.data.render_config.water.ripple_strength = 0.22;
        h.data.render_config.water.varied_waves = true;
        if std::env::var_os("UNDERWATER_REVIEW_TIMING").is_some() && prefix == "overview-" {
            // Alternate block order to reduce GPU warm-up/clock bias. Only time
            // steady frames after the optional resource has been recreated.
            let mut samples: [Vec<f64>; 2] = Default::default();
            h.data.render_config.water.underwater.enabled = true;
            h.data.render_config.water.underwater.light_shafts = true;
            for round in 0..4 {
                for stage in 0..2 {
                    let mode = (round + stage) % 2;
                    h.data.render_config.water.underwater.caustics = mode == 1;
                    for i in 0..16 {
                        if let Some(ms) = h.time_passes_ms() {
                            if i >= 4 {
                                samples[mode].push(ms[0]);
                            }
                        }
                    }
                }
            }
            for (mode, times) in samples.iter_mut().enumerate() {
                times.sort_by(f64::total_cmp);
                if !times.is_empty() {
                    eprintln!(
                        "Paired shafts-on, caustics={}, 1280x720: median {:.3} ms, p95 {:.3} ms ({} samples)",
                        mode == 1,
                        times[times.len() / 2],
                        times[(times.len() * 95 / 100).min(times.len() - 1)],
                        times.len()
                    );
                }
            }
        }
        for (name, enabled, shafts, caustics) in [
            ("disabled", false, false, false),
            ("no-shafts", true, false, false),
            ("enabled", true, true, false),
            ("caustics", true, false, true),
            ("combined", true, true, true),
        ] {
            h.data.render_config.water.underwater.enabled = enabled;
            h.data.render_config.water.underwater.light_shafts = shafts;
            h.data.render_config.water.underwater.caustics = caustics;
            let frame = h.frame();
            image::save_buffer(
                output.join(format!("{prefix}{name}.png")),
                &frame.pixels,
                1280,
                720,
                image::ColorType::Rgba8,
            )
            .unwrap();
            if std::env::var_os("UNDERWATER_REVIEW_TIMING").is_some() && prefix == "overview-" {
                let mut times = Vec::new();
                for i in 0..25 {
                    if let Some(ms) = h.time_passes_ms() {
                        if i >= 5 {
                            times.push(ms[0]);
                        }
                    }
                }
                times.sort_by(f64::total_cmp);
                if !times.is_empty() {
                    eprintln!(
                        "Underwater {name}, 1280x720, {} splats: median {:.3} ms, p95 {:.3} ms",
                        scene.splat_count,
                        times[times.len() / 2],
                        times[times.len() - 2]
                    );
                }
            }
        }
    }
    eprintln!("Exported local soft coral review to {}", output.display());
}
