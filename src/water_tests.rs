//! Pixel regression tests using the real GS pipeline and depth attachment.
mod fixtures;
mod harness;
mod intersections;
mod material;
mod visual;
mod performance;
mod cache;
use crate::{structure::SurfaceType, test_support::gpu::RgbaFrame, utils::*};
use harness::Harness;
const WATER: [u8; 3] = [10, 61, 97];

fn pixel(frame: &RgbaFrame, x: usize, y: usize) -> [u8; 3] {
    frame.pixel(x, y)
}
fn assert_water(actual: [u8; 3]) {
    for i in 0..3 {
        assert!(
            (actual[i] as i32 - WATER[i] as i32).abs() <= 1,
            "expected water, got {actual:?}"
        );
    }
}

#[test]
fn water_gpu_fills_gaps_occludes_submerged_gs_and_restores_disabled_frame() {
    let mut h = Harness::new();
    let disabled = h.frame();
    assert_eq!(pixel(&disabled, 8, 8), [0, 0, 0]);
    assert!(
        h.at_world(&disabled, vec3(-0.7, 0.0, 0.6))[0] > 220,
        "test fixture: front GS absent even without water: {:?}",
        h.at_world(&disabled, vec3(-0.7, 0.0, 0.6))
    );
    assert!(h.at_world(&disabled, vec3(0.7, 0.0, -0.6))[1] > 220);
    h.data.render_config.water.enabled = true;
    let wet = h.frame();
    assert_water(pixel(&wet, 8, 8));
    assert_water(h.at_world(&wet, vec3(0.7, 0.0, -0.6)));
    assert!(
        h.at_world(&wet, vec3(-0.7, 0.0, 0.6))[0] > 220,
        "above-water GS must remain visible"
    );
    h.data.render_config.water.height = 1.2;
    let raised = h.frame();
    assert_water(h.at_world(&raised, vec3(-0.7, 0.0, 0.6)));
    h.data.render_config.water.height = -1.2;
    let lowered = h.frame();
    assert!(h.at_world(&lowered, vec3(0.7, 0.0, -0.6))[1] > 220);
    h.data.render_config.water.enabled = false;
    assert_eq!(
        h.frame(),
        disabled,
        "disabling water must clear its previous depth"
    );
}

#[test]
fn water_gpu_crossing_a_splat_does_not_pop_an_entire_footprint() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    let mut previous: Option<i32> = None;
    let mut worst = 0;
    for step in 0..61 {
        h.data.render_config.water.height = 0.3 + step as f32 * 0.01;
        let frame = h.frame();
        let red = h.at_world(&frame, vec3(-0.7, 0.0, 0.6))[0] as i32;
        if let Some(last) = previous {
            worst = worst.max((red - last).abs());
        }
        previous = Some(red);
    }
    assert!(
        worst <= 25,
        "a 0.01-world-unit water-level step changed the splat by {worst}/255; waterline must be continuous"
    );
}

#[test]
fn water_gpu_keeps_world_level_and_coverage_through_camera_and_tile_window_changes() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 0.9;
    h.data.render_config.scene_scale.z = 2.0;
    let scaled = h.frame();
    assert!(
        h.at_world(&scaled, vec3(-0.7, 0.0, 1.2))[0] > 220,
        "water height must stay in final world units"
    );
    for center in [0, 100, -100] {
        let x = center as f32 * h.user.tile_width;
        h.data.cur_scene_data.as_mut().unwrap().center_coord.x = center;
        h.data.cur_sort_data.as_mut().unwrap().tile_instance_vec[0]
            .tile_offset
            .x = x;
        for z in [5.0, 12.0] {
            h.camera
                .set_view(vec3(x, 0.0, z), vec3(x, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
            let frame = h.frame();
            for (px, py) in [(8, 8), (120, 8), (8, 120), (120, 120)] {
                assert_water(pixel(&frame, px, py));
            }
        }
    }
    // Outside the finite water footprint, submerged Gaussians must not be cut.
    h.data.cur_scene_data.as_mut().unwrap().center_coord.x = 0;
    h.data.cur_sort_data.as_mut().unwrap().tile_instance_vec[0]
        .tile_offset
        .x = 14.0;
    h.camera.set_view(
        vec3(14.0, 0.0, 5.0),
        vec3(14.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    h.data.render_config.water.height = 2.0;
    let outside = h.frame();
    assert!(h.at_world(&outside, vec3(13.3, 0.0, 1.2))[0] > 220);
    h.user.surface_type = SurfaceType::Sphere;
    assert!(!h.data.render_config.water.is_active(h.user.surface_type));
}

#[test]
fn water_gpu_respects_proxy_depth_and_heightmap_terrain() {
    let mut h = Harness::new();
    h.user.surface_type = SurfaceType::HeightMap;
    h.user.height_map_scale.z = 0.5;
    h.user.height_map = vec![1.0];
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.data.use_proxy = true;
    h.data.render_config.proxy_height = 0.6;
    h.data.render_config.water.enabled = true;
    let mut proxy = crate::proxy::Proxy::new(&h.device, &h.config);
    proxy.configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.7, 0.5, 0.1, 1.0]], vec2(1, 1)),
    );
    h.proxy = Some(proxy);
    let terrain = h.frame();
    let ground = pixel(&terrain, 8, 8);
    assert!(
        ground[0] > 150 && ground[2] < 50,
        "terrain above water must remain visible: {ground:?}"
    );
    h.data.render_config.water.amplitude = 0.08;
    h.data.render_config.water.advance(1.0);
    assert_eq!(
        pixel(&h.frame(), 8, 8),
        ground,
        "animated water must stay behind higher terrain"
    );
    h.data.render_config.water.amplitude = 0.0;
    h.data.render_config.proxy_height = -1.0;
    let flooded = h.frame();
    assert_water(pixel(&flooded, 8, 8));
    // A small fixed NDC bias becomes large in world space at distance. A splat
    // between water and higher ground must remain behind the ground.
    h.user.surface_type = SurfaceType::None;
    h.user.tile_width = 500.0;
    h.data.render_config.scene_scale = vec3(125.0, 125.0, 0.1);
    h.data.render_config.proxy_height = 0.1;
    h.camera
        .set_view(vec3(0.0, 0.0, 200.0), Vec3::zero(), vec3(0.0, 1.0, 0.0));
    h.camera
        .set_perspective_projection(degrees(60.0), 0.1, 2400.0);
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.proxy.as_mut().unwrap().configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.7, 0.5, 0.1, 1.0]], vec2(1, 1)),
    );
    let distant = h.frame();
    let covered = h.at_world(&distant, vec3(-87.5, 0.0, 0.06));
    assert_eq!(
        covered,
        pixel(&distant, 8, 8),
        "water clipping must not pull GS through nearer terrain"
    );
}

#[test]
fn water_gpu_dynamic_gs_moves_through_a_stationary_waterline() {
    let mut h = Harness::with_motion(true);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 0.6;
    let at_level = h.frame();
    let red_at = h.at_world(&at_level, vec3(-0.7, 0.0, 0.6))[0];
    h.motion_time = Some(0.25);
    let above = h.frame();
    let red_above = h.at_world(&above, vec3(-0.7, 0.0, 0.9))[0];
    assert!(
        red_above as i32 - red_at as i32 > 70,
        "GPU motion should emerge through water: {red_at} -> {red_above}"
    );
    assert_eq!(
        h.frame(),
        above,
        "paused time must produce identical pixels"
    );
    h.motion_time = Some(0.75);
    let below = h.frame();
    assert_water(h.at_world(&below, vec3(-0.7, 0.0, 0.3)));
    assert_water(pixel(&below, 8, 8));
    h.motion_time = Some(0.0);
    assert_eq!(
        h.frame(),
        at_level,
        "scrubbing back must restore the original contact"
    );
}

#[test]
fn water_gpu_oblique_contact_and_camera_height_are_stable() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    for z in [0.7, 2.0, 5.0] {
        h.camera
            .set_view(vec3(0.0, -4.0, z), Vec3::zero(), vec3(0.0, 0.0, 1.0));
        let frame = h.frame();
        assert!(
            h.at_world(&frame, vec3(-0.7, 0.0, 0.6))[0] > 200,
            "above-water foreground lost at camera z={z}"
        );
        assert_water(h.at_world(&frame, vec3(0.7, 0.0, -0.6)));
        assert_eq!(h.frame(), frame, "stationary view should not flicker");
    }
}

#[test]
fn water_gpu_rectangular_target_reads_unaligned_rows() {
    let mut h = Harness::from_scene(fixtures::two_splats(false), [65, 3]);
    h.camera
        .set_perspective_projection(degrees(3.0), 0.1, 100.0);
    h.data.render_config.water.enabled = true;
    let frame = h.frame();
    assert_eq!(frame.pixels.len(), 65 * 3 * 4);
    assert_eq!([frame.width, frame.height], [65, 3]);
    assert_water(frame.pixel(0, 0));
    assert_water(frame.pixel(64, 2));
}

#[test]
fn water_gpu_default_waves_have_spatial_shape() {
    let mut h = Harness::new();
    h.data.render_config.water = crate::water::WaterSettings::default();
    h.data.render_config.water.enabled = true;
    let frame = h.frame();
    let values: Vec<u8> = (8..120).map(|x| frame.pixel(x, 8)[2]).collect();
    assert!(
        values.iter().max().unwrap() - values.iter().min().unwrap() > 4,
        "enabled default waves must show spatially varying normal shading in empty water"
    );
}

#[test]
fn water_gpu_waves_move_the_contact_pause_and_recover_flat_water() {
    for varied in [false, true] {
        let mut h = Harness::new();
        h.data.render_config.water.enabled = true;
        h.data.render_config.water.height = 0.6;
        h.data.render_config.water.varied_waves = varied;
        let flat = h.frame();
        h.data.render_config.water.amplitude = 0.12;
        h.data.render_config.water.wavelength = 5.0;
        let mut reds = Vec::new();
        let mut previous: Option<i32> = None;
        for _ in 0..90 {
            h.data.render_config.water.advance(0.05);
            let frame = h.frame();
            let red = h.at_world(&frame, vec3(-0.7, 0.0, 0.6))[0] as i32;
            if let Some(last) = previous {
                assert!(
                    (red - last).abs() < 25,
                    "wave contact popped: {last} -> {red}"
                );
            }
            previous = Some(red);
            reds.push(red);
        }
        assert!(
            reds.iter().max().unwrap() - reds.iter().min().unwrap() > 40,
            "wave height must move GS contact, not just its shading"
        );
        h.data.render_config.water.playing = false;
        let paused = h.frame();
        h.data.render_config.water.advance(100.0);
        assert_eq!(h.frame(), paused, "pause must freeze geometry and shading");
        h.data.render_config.water.playing = true;
        h.data.render_config.water.speed = 0.0;
        h.data.render_config.water.advance(100.0);
        assert_eq!(h.frame(), paused, "zero speed must hold the current phase");
        h.data.render_config.water.amplitude = 0.0;
        assert_eq!(
            h.frame(),
            flat,
            "zero amplitude must restore stage-1 pixels exactly"
        );
    }
}

#[test]
fn water_gpu_wave_phase_survives_tile_window_updates() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.amplitude = 0.1;
    h.data.render_config.water.advance(1.2);
    let first = h.frame();
    h.data.cur_scene_data.as_mut().unwrap().center_coord.x = 1;
    assert_eq!(
        h.frame(),
        first,
        "tile streaming must not reset the world-space wave phase"
    );
}

#[test]
fn water_clock_is_frame_rate_independent_and_speed_changes_do_not_jump() {
    let mut one = crate::water::WaterSettings::default();
    one.enabled = true;
    let mut many = one.clone();
    one.advance(1.0);
    for _ in 0..120 {
        many.advance(1.0 / 120.0);
    }
    assert_eq!(one.phases(), many.phases());
    assert_eq!(one.detail_offsets(), many.detail_offsets());
    let before = one.phases();
    one.speed = 2.0;
    assert_eq!(
        one.phases(),
        before,
        "speed changes must not retime past motion"
    );
    one.advance(0.5);
    many.advance(1.0);
    assert_eq!(one.phases(), many.phases());
    assert_eq!(one.detail_offsets(), many.detail_offsets());
    let before = one.phases();
    for delta in [f64::NAN, f64::INFINITY, -1.0] {
        one.advance(delta);
    }
    assert_eq!(one.phases(), before);
    one.enabled = false;
    one.advance(10.0);
    assert_eq!(one.phases(), before);
}

#[test]
fn water_gpu_oblique_waves_and_dynamic_gs_preserve_depth_order() {
    let mut h = Harness::with_motion(true);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.amplitude = 0.08;
    for z in [0.25, 0.7, 2.0] {
        h.camera
            .set_view(vec3(0.0, -4.0, z), Vec3::zero(), vec3(0.0, 0.0, 1.0));
        for (u, offset) in [(0.0, 0.0), (0.25, 0.3), (0.75, -0.3)] {
            h.motion_time = Some(u);
            h.data.render_config.water.advance(0.7);
            let frame = h.frame();
            assert!(
                h.at_world(&frame, vec3(-0.7, 0.0, 0.6 + offset))[0] > 170,
                "above-water moving GS lost at camera z={z}, motion={u}"
            );
            let submerged = h.at_world(&frame, vec3(0.7, 0.0, -0.6 + offset));
            assert!(
                submerged[0] < 25 && submerged[1] < submerged[2],
                "submerged GS leaked through waves: {submerged:?}"
            );
            assert_eq!(
                h.frame(),
                frame,
                "stationary wave phase and GS time must be deterministic"
            );
        }
    }
}
