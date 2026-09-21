//! Opt-in fixed-camera, dense real-asset benchmark; no browser FPS claims.
use super::{fixtures::SceneFixture, harness::Harness};
use crate::{camera::Camera, scene::Scene, utils::*};
#[test]
#[ignore = "set WATER_PERF_ARCHIVE and WATER_PERF_DIR for local 1080p comparison"]
fn water_dense_shrub_benchmark() {
    use std::io::Read;
    let path = std::env::var("WATER_PERF_ARCHIVE").unwrap();
    let dir = std::path::PathBuf::from(std::env::var("WATER_PERF_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let mut zip = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut bytes = Vec::new();
    zip.by_name("tile0_lod0.ply")
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let mut scene = Scene::from_ply_bytes(bytes).unwrap();
    let (aabb, _) = scene.compute_aabb_and_center();
    let tile_width = (aabb.1.x - aabb.0.x).max(aabb.1.y - aabb.0.y);
    let low = aabb.0.z;
    let span = (aabb.1.z - low).max(0.1);
    let source = scene.buffer.clone();
    scene.buffer.clear();
    // 25 actual tile copies, fixed canonical geometry, isolate water/GS shading cost.
    for y in -2..=2 {
        for x in -2..=2 {
            for row in source.chunks_exact(32) {
                let mut out = row.to_vec();
                let px = f32::from_ne_bytes(row[0..4].try_into().unwrap()) + x as f32 * tile_width;
                let py = f32::from_ne_bytes(row[4..8].try_into().unwrap()) + y as f32 * tile_width;
                out[0..4].copy_from_slice(&px.to_ne_bytes());
                out[4..8].copy_from_slice(&py.to_ne_bytes());
                scene.buffer.extend_from_slice(&out);
            }
        }
    }
    scene.splat_count = scene.buffer.len() / 32;
    let camera = Camera::new_perspective(
        winit::dpi::PhysicalSize::new(1920, 1080),
        vec3(tile_width * 0.5, -tile_width * 2.6, low + span * 1.2),
        vec3(tile_width * 0.5, tile_width * 0.7, low + span * 0.4),
        vec3(0.0, 0.0, 1.0),
        degrees(55.0),
        0.1,
        2000.0,
    );
    let count = scene.splat_count;
    let mut h = Harness::from_scene(SceneFixture::from_scene(scene, &camera), [1920, 1080]);
    h.camera = camera;
    h.user.tile_map_half_wh = vec2(48, 48);
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(
            crate::skybox::SkyboxTexture::Cubemap(vec![vec![0.5, 0.65, 0.8, 1.0]; 6]),
            vec2(1, 1),
        ),
    );
    h.data.use_skybox = true;
    h.data.render_config.water = crate::water::WaterSettings::default();
    h.data.render_config.water.height = low + span * 0.45;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.advance(0.73);
    // Keep GS clipping bounds in sync with the water pass after changing the map.
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    let settings = h.data.render_config.water.clone();
    let mut results = Vec::new();
    for (name, enabled, amplitude, reflection, ripples) in [
        ("disabled", false, 0.08, 1.0, 0.22),
        ("flat", true, 0.0, 1.0, 0.22),
        ("waves", true, 0.08, 0.0, 0.0),
        ("full", true, 0.08, 1.0, 0.22),
    ] {
        h.data.render_config.water = settings.clone();
        let w = &mut h.data.render_config.water;
        w.enabled = enabled;
        w.amplitude = amplitude;
        w.reflection_strength = reflection;
        w.ripple_strength = ripples;
        let image = h.frame();
        image::save_buffer(
            dir.join(format!("{name}.png")),
            &image.pixels,
            1920,
            1080,
            image::ColorType::Rgba8,
        )
        .unwrap();
        let mut times = Vec::new();
        let mut passes = Vec::new();
        for i in 0..35 {
            if let Some(ms) = h.time_passes_ms() {
                if i >= 5 {
                    times.push(ms[0]);
                    passes.push(ms);
                }
            }
        }
        times.sort_by(f64::total_cmp);
        assert_eq!(times.len(), 30, "benchmark requires GPU timestamp support");
        let entry = serde_json::json!({"mode":name,"median_ms":times[15],"p95_ms":times[28],"samples":times,"pass_samples":passes});
        eprintln!(
            "{name}: median {:.3} ms, p95 {:.3} ms, {count} splats at 1920x1080",
            times[15], times[28]
        );
        results.push(entry);
    }
    std::fs::write(dir.join("timings.json"),serde_json::to_vec_pretty(&serde_json::json!({"splat_count":count,"tile_copies":25,"width":1920,"height":1080,"results":results})).unwrap()).unwrap();
}
