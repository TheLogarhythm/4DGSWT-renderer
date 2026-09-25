use super::harness::Harness;
use crate::skybox::SkyboxTexture;
use crate::utils::*;

#[test]
fn underwater_caustics_light_receivers_not_background_and_obey_controls() {
    let mut h = Harness::new();
    h.camera.set_view(
        vec3(0.0, -4.0, 2.0),
        vec3(0.0, 3.0, -1.0),
        vec3(0.0, 0.0, 1.0),
    );
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 4.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.light_shafts = false;
    h.data.render_config.water.underwater.clear_distance = 100.0;
    assert!(!h.data.render_config.water.underwater.caustics);
    let background = h.frame();
    h.data.render_config.water.underwater.caustics = true;
    let no_receiver = h.frame();
    assert_eq!(
        background.pixel(4, 4),
        no_receiver.pixel(4, 4),
        "caustics painted the water/background"
    );
    h.data.use_proxy = true;
    h.data.render_config.proxy_full = true;
    h.data.render_config.proxy_map = false;
    h.data.render_config.proxy_height = -1.0;
    let mut proxy = crate::proxy::Proxy::new(&h.device, &h.config);
    proxy.configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.35, 0.4, 0.45, 1.0]], vec2(1, 1)),
    );
    h.proxy = Some(proxy);
    h.data.render_config.water.underwater.caustics = false;
    let plain = h.frame();
    h.data.render_config.water.underwater.caustics = true;
    let lit = h.frame();
    let mut bright = 0;
    let mut gaps = 0;
    for y in 80..120 {
        for x in 8..120 {
            let delta = lit.pixel(x, y)[1] as i32 - plain.pixel(x, y)[1] as i32;
            bright += usize::from(delta >= 8);
            gaps += usize::from(delta.abs() <= 2);
        }
    }
    assert!(
        bright > 40 && gaps > 40,
        "receiver needs bright lines and unlit gaps: bright={bright}, gaps={gaps}"
    );
    h.data.render_config.water.underwater.caustic_strength = 0.0;
    assert!(
        h.frame() == plain,
        "zero strength must restore the exact receiver"
    );
    h.data.render_config.water.underwater.caustic_strength = 0.65;
    h.data.render_config.water.playing = false;
    h.data.render_config.water.advance(1.0);
    assert!(h.frame() == lit, "paused caustics moved");
    h.data.render_config.water.playing = true;
    h.data.render_config.water.underwater.caustic_speed = 0.0;
    h.data.render_config.water.advance(1.0);
    assert!(
        h.frame() == lit,
        "zero caustic speed must freeze the pattern while water runs"
    );
    h.data.render_config.water.underwater.caustic_speed = 0.25;
    assert!(
        h.frame() == lit,
        "changing caustic speed must not jump the pattern"
    );
    h.data.render_config.water.advance(1.0);
    assert!(h.frame() != lit, "caustics do not animate on flat water");
    h.data.render_config.water.underwater.sunlight = 0.0;
    let dark = h.frame();
    h.data.render_config.water.underwater.caustics = false;
    assert!(h.frame() == dark, "caustics emit light with sunlight off");
    h.camera.set_view(
        vec3(0.0, -4.0, 8.0),
        vec3(0.0, 3.0, -1.0),
        vec3(0.0, 0.0, 1.0),
    );
    let above = h.frame();
    h.data.render_config.water.underwater.caustics = true;
    assert!(
        h.frame() == above,
        "underwater caustics changed the above-water path"
    );
}

#[test]
fn underwater_caustics_reach_gaussians_without_proxy_ground() {
    let mut fixture = super::fixtures::two_splats(false);
    fixture.scene.tex_data[7] = 0xff706050;
    fixture.scene.tex_data[15] = 0xff706050;
    let mut h = Harness::from_scene(fixture, [128, 128]);
    h.camera
        .set_view(vec3(0.0, -2.0, 0.0), Vec3::zero(), vec3(0.0, 0.0, 1.0));
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 3.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.light_shafts = false;
    let plain = h.frame();
    h.data.render_config.water.underwater.caustics = true;
    let mut largest_lift = 0;
    for scale in [0.6, 0.9, 1.5, 2.4] {
        h.data.render_config.water.underwater.caustic_scale = scale;
        let lit = h.frame();
        for point in [vec3(0.7, 0.0, -0.6), vec3(-0.7, 0.0, 0.6)] {
            largest_lift = largest_lift
                .max(h.at_world(&lit, point)[1] as i32 - h.at_world(&plain, point)[1] as i32);
        }
    }
    assert!(
        largest_lift >= 5,
        "GS receivers never received caustics: {largest_lift}"
    );
}

#[test]
fn underwater_shafts_have_distinct_bright_beams_and_dark_gaps() {
    let mut h = Harness::new();
    h.resize([256, 144]);
    h.user.tile_map_half_wh = vec2(40, 40);
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.camera.set_view(
        vec3(0.0, -2.0, 0.0),
        vec3(0.0, 3.0, 0.8),
        vec3(0.0, 0.0, 1.0),
    );
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 8.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.light_shafts = false;
    let without = h.frame();
    h.data.render_config.water.underwater.light_shafts = true;
    let with = h.frame();
    // Empty water above the two fixture splats. A uniform lift must not pass
    // merely because toggling shafts changes many pixels.
    let mut lift: Vec<i32> = (8..248)
        .map(|x| with.pixel(x, 40)[1] as i32 - without.pixel(x, 40)[1] as i32)
        .collect();
    lift.sort_unstable();
    let dark = lift[lift.len() / 10];
    let bright = lift[lift.len() * 9 / 10];
    assert!(
        bright - dark >= 18 && bright >= 24 && dark <= 6,
        "shafts became a uniform wash instead of separated beams: p10={dark}, p90={bright}"
    );
}

#[test]
fn underwater_shafts_change_open_water_without_painting_near_coral() {
    let mut h = Harness::new();
    h.user.tile_map_half_wh = vec2(40, 40);
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.camera
        .set_view(vec3(0.0, -2.0, 0.0), Vec3::zero(), vec3(0.0, 0.0, 1.0));
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 3.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.shaft_strength = 2.0;
    h.data.render_config.water.underwater.light_shafts = false;
    let without = h.frame();
    h.data.render_config.water.underwater.light_shafts = true;
    let with = h.frame();
    for point in [vec3(0.7, 0.0, -0.6), vec3(-0.7, 0.0, 0.6)] {
        let a = h.at_world(&without, point);
        let b = h.at_world(&with, point);
        assert!(
            a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2),
            "light behind foreground coral leaked through: {a:?} -> {b:?}"
        );
    }
    let changed = (12..116)
        .filter(|&x| with.pixel(x, 65)[1] > without.pixel(x, 65)[1] + 3)
        .count();
    assert!(
        changed > 20,
        "shaft toggle produced no visible open-water light: {changed}"
    );
    assert_eq!(
        with,
        h.frame(),
        "a still camera must not accumulate noise/history"
    );
    h.data.render_config.water.playing = false;
    h.data.render_config.water.advance(0.8);
    assert_eq!(with, h.frame(), "water pause must also pause shaft motion");
    h.data.render_config.water.playing = true;
    h.data.render_config.water.advance(1.7);
    assert_ne!(
        with,
        h.frame(),
        "shaft illumination must follow the water clock"
    );
    h.data.render_config.water.underwater.sunlight = 0.0;
    let no_sun = h.frame();
    h.data.render_config.water.underwater.light_shafts = false;
    assert_eq!(
        no_sun,
        h.frame(),
        "shafts cannot emit light with sunlight off"
    );
}

#[test]
fn underwater_volume_resizes_and_releases_when_disabled_or_above_water() {
    let mut h = Harness::new();
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    h.camera
        .set_view(vec3(0.0, -2.0, 0.0), Vec3::zero(), vec3(0.0, 0.0, 1.0));
    h.frame();
    assert_eq!(h.water.hits().unwrap().volume_size, [1; 3]);
    h.data.render_config.water.underwater.enabled = true;
    h.frame();
    assert!(h.water.hits().unwrap().volume_size[2] > 1);
    h.resize([177, 99]);
    h.frame(); // GPU validation also checks the rebuilt storage/sampling bindings.
    assert_eq!(h.water.hits().unwrap().size, [177, 99]);
    h.camera.set_view(
        vec3(0.0, -2.0, 3.0),
        vec3(0.0, 0.0, 3.0),
        vec3(0.0, 0.0, 1.0),
    );
    let above = h.frame();
    assert_eq!(h.water.hits().unwrap().volume_size, [1; 3]);
    h.data.render_config.water.underwater.enabled = false;
    assert_eq!(above, h.frame());
    h.camera
        .set_view(vec3(0.0, -2.0, 0.0), Vec3::zero(), vec3(0.0, 0.0, 1.0));
    h.data.render_config.water.underwater.enabled = true;
    h.frame();
    h.data.render_config.water.enabled = false;
    h.frame();
    assert!(
        h.water.hits().is_none(),
        "turning all water off must release the scattering volume"
    );
}

#[test]
fn underwater_absorption_stops_at_the_water_coverage_boundary() {
    let mut h = Harness::new();
    h.user.tile_width = 1.0;
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 2.0, -2.0), vec3(0.0, 0.0, 1.0));
    h.data.use_proxy = true;
    h.data.render_config.proxy_height = -20.0;
    h.data.render_config.proxy_full = true;
    h.data.render_config.proxy_map = false;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.visibility = 2.0;
    h.data.render_config.water.underwater.clear_distance = 0.0;
    h.data.render_config.water.underwater.color = [0.0; 3];
    h.data.render_config.water.underwater.sunlight = 0.0;
    let mut proxy = crate::proxy::Proxy::new(&h.device, &h.config);
    proxy.configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.7, 0.7, 0.7, 1.0]], vec2(1, 1)),
    );
    h.proxy = Some(proxy);
    h.data.render_config.water.underwater.enabled = false;
    assert!(
        h.frame().pixel(64, 64)[1] > 150,
        "the distant receiver must be visible before testing attenuation"
    );
    h.data.render_config.water.underwater.enabled = true;
    let a = h.frame().pixel(64, 64);
    h.data.render_config.proxy_height = -40.0;
    let b = h.frame().pixel(64, 64);
    assert!(
        a[1] > 60,
        "air beyond the finite water volume was fogged: {a:?}"
    );
    assert!(
        a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 2),
        "fog continued after the ray left water: {a:?} -> {b:?}"
    );
}

#[test]
fn underwater_transport_respects_a_surface_in_front_of_the_near_plane() {
    let mut h = Harness::new();
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 2.0, 2.0), vec3(0.0, 0.0, 1.0));
    h.data.use_proxy = true;
    h.data.render_config.proxy_full = true;
    h.data.render_config.proxy_map = false;
    h.data.render_config.proxy_height = 20.0;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 0.01; // Interface is closer than camera near=0.1.
    h.data.render_config.water.underwater.visibility = 0.5;
    h.data.render_config.water.underwater.clear_distance = 0.0;
    h.data.render_config.water.underwater.color = [0.0; 3];
    h.data.render_config.water.underwater.sunlight = 0.0;
    let mut proxy = crate::proxy::Proxy::new(&h.device, &h.config);
    proxy.configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.7, 0.7, 0.7, 1.0]], vec2(1, 1)),
    );
    h.proxy = Some(proxy);
    let dry = h.frame().pixel(64, 64);
    assert!(dry[1] > 150, "air-side receiver missing: {dry:?}");
    h.data.render_config.water.underwater.enabled = true;
    let wet = h.frame().pixel(64, 64);
    assert!(
        dry.iter().zip(wet).all(|(a, b)| a.abs_diff(b) <= 5),
        "near-clipped interface caused air to be treated as water: {dry:?} -> {wet:?}"
    );
}

#[test]
fn underwater_scattering_is_brighter_near_the_surface() {
    let mut h = Harness::new();
    h.user.tile_map_half_wh = vec2(40, 40);
    h.gs.configure(&h.device, &h.queue, &h.user, &h.data);
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    h.data.render_config.water.underwater.enabled = true;
    h.camera.set_view(
        vec3(0.0, -2.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    );
    let shallow = h.frame().pixel(4, 64);
    h.camera.set_view(
        vec3(0.0, -2.0, -12.0),
        vec3(0.0, 0.0, -12.0),
        vec3(0.0, 0.0, 1.0),
    );
    let deep = h.frame().pixel(4, 64);
    assert!(
        shallow[1] > deep[1] + 12,
        "water lighting has no depth: {shallow:?} versus {deep:?}"
    );
}

#[test]
fn underwater_default_preserves_near_coral_color() {
    let mut h = Harness::new();
    h.camera
        .set_view(vec3(0.0, -2.0, 0.0), Vec3::zero(), vec3(0.0, 0.0, 1.0));
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    let ordinary = h.frame();
    h.data.render_config.water.underwater.enabled = true;
    let submerged = h.frame();
    for point in [vec3(0.7, 0.0, -0.6), vec3(-0.7, 0.0, 0.6)] {
        let before = h.at_world(&ordinary, point);
        let after = h.at_world(&submerged, point);
        for channel in 0..3 {
            assert!(
                (before[channel] as i32 - after[channel] as i32).abs() <= 12,
                "near coral should retain its original color: {before:?} -> {after:?}"
            );
        }
    }
}

#[test]
fn underwater_underside_is_independent_of_top_surface_color() {
    let mut h = Harness::new();
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.color = [1.0, 0.0, 0.0];
    let red_top = h.frame().pixel(64, 64);
    h.data.render_config.water.color = [0.0, 1.0, 0.0];
    let green_top = h.frame().pixel(64, 64);
    assert_eq!(
        red_top, green_top,
        "viewing the underside must not use the opaque top-surface pigment"
    );
    assert!(
        red_top[0] > 100 && red_top[1] > 100 && red_top[2] > 100,
        "upward transmission should remain visible without a loaded sky: {red_top:?}"
    );
}

#[test]
fn underwater_underside_transmits_sky_overhead_but_not_at_grazing_angles() {
    let mut h = Harness::new();
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(
            SkyboxTexture::Cubemap(vec![vec![1.0, 0.0, 0.0, 1.0]; 6]),
            vec2(1, 1),
        ),
    );
    h.data.use_skybox = true;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.0;
    h.data.render_config.water.reflection_strength = 0.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.color = [0.0, 0.15, 0.2];
    h.data.render_config.water.underwater.sunlight = 0.0; // Isolate sky transmission from white shaft light.
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));
    let overhead = h.frame().pixel(64, 64);
    assert!(
        overhead[0] > 220 && overhead[2] < 20,
        "overhead transmission must use the sky: {overhead:?}"
    );
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 4.0, 1.0), vec3(0.0, 0.0, 1.0));
    let grazing = h.frame().pixel(64, 64);
    assert!(
        grazing[0] < 5 && grazing[1] > 25 && grazing[2] > 40,
        "total internal reflection must not reflect the red sky: {grazing:?}"
    );
}

#[test]
fn underwater_ripple_normals_cannot_switch_to_the_top_material() {
    let mut h = Harness::new();
    h.camera
        .set_view(Vec3::zero(), vec3(0.0, 9.0, 1.0), vec3(0.0, 0.0, 1.0));
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 1.0;
    h.data.render_config.water.amplitude = 0.08;
    h.data.render_config.water.ripple_strength = 1.0;
    h.data.render_config.water.varied_waves = true;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.color = [1.0, 0.0, 0.0];
    let red_top = h.frame();
    h.data.render_config.water.color = [0.0, 1.0, 0.0];
    let green_top = h.frame();
    assert!(
        red_top == green_top,
        "ripple shading normals leaked the top material into the underside"
    );
}

#[test]
fn underwater_gpu_fades_submerged_gaussians_and_restores_the_normal_water_view() {
    let mut h = Harness::new();
    h.camera.set_view(
        vec3(0.0, -2.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    );
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    h.data.render_config.water.reflection_strength = 0.0;
    let ordinary = h.frame();
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.visibility = 2.0;
    h.data.render_config.water.underwater.clear_distance = 0.0;
    h.data.render_config.water.underwater.color = [0.0; 3];
    h.data.render_config.water.underwater.sunlight = 0.0; // Isolate absorption from added sunlight.
    let submerged = h.frame();
    let position = vec3(0.7, 0.0, -0.6);
    let normal_green = h.at_world(&ordinary, position)[1];
    let submerged_green = h.at_world(&submerged, position)[1];
    assert!(
        normal_green > 80,
        "fixture Gaussian is not visible: {normal_green}"
    );
    assert!(
        submerged_green + 25 < normal_green,
        "underwater attenuation did not fade the Gaussian: {normal_green} -> {submerged_green}"
    );
    h.data.render_config.water.underwater.enabled = false;
    assert_eq!(
        h.frame(),
        ordinary,
        "turning underwater off must restore the old path"
    );
}

#[test]
fn underwater_gpu_tints_uncovered_background() {
    let mut h = Harness::new();
    h.camera.set_view(
        vec3(0.0, -2.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    );
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.color = [0.05, 0.3, 0.4];
    let submerged = h.frame();
    let background = submerged.pixel(4, 120);
    assert!(
        background[1] > 60 && background[2] > 90,
        "background was not tinted: {background:?}"
    );
    h.data.render_config.water.underwater.enabled = false;
    assert_eq!(h.frame().pixel(4, 120), [0, 0, 0]);
}

#[test]
fn underwater_gpu_fades_proxy_ground() {
    let mut h = Harness::new();
    h.camera.set_view(
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    h.data.use_proxy = true;
    h.data.render_config.proxy_height = 0.0;
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    let mut proxy = crate::proxy::Proxy::new(&h.device, &h.config);
    proxy.configure(
        &h.device,
        &h.queue,
        &h.user,
        &h.data,
        &(vec![vec![0.7, 0.5, 0.1, 1.0]], vec2(1, 1)),
    );
    h.proxy = Some(proxy);
    let ordinary = h.frame().pixel(8, 8);
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.color = [0.0; 3];
    h.data.render_config.water.underwater.sunlight = 0.0;
    h.data.render_config.water.underwater.visibility = 1.0;
    h.data.render_config.water.underwater.clear_distance = 0.0;
    let submerged = h.frame().pixel(8, 8);
    assert!(ordinary[0] > 140, "proxy fixture not visible: {ordinary:?}");
    assert!(
        submerged[0] + 50 < ordinary[0],
        "proxy did not fade: {ordinary:?} -> {submerged:?}"
    );
}

#[test]
fn underwater_gpu_tints_loaded_sky_and_restores_it_above_water() {
    let mut h = Harness::new();
    h.skybox.configure(
        &h.device,
        &h.queue,
        &(
            SkyboxTexture::Cubemap(vec![vec![1.0, 0.0, 0.0, 1.0]; 6]),
            vec2(1, 1),
        ),
    );
    h.data.use_skybox = true;
    h.camera.set_view(
        vec3(0.0, -2.0, 0.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
    );
    h.data.render_config.water.enabled = true;
    h.data.render_config.water.height = 2.0;
    let ordinary = h.frame().pixel(4, 120);
    h.data.render_config.water.underwater.enabled = true;
    h.data.render_config.water.underwater.color = [0.0, 0.3, 0.4];
    h.data.render_config.water.underwater.sunlight = 0.0; // Detect sky leakage independently of shafts.
    let submerged = h.frame().pixel(4, 120);
    assert!(
        ordinary[0] > 200,
        "red sky fixture not visible: {ordinary:?}"
    );
    assert!(
        submerged[0] < 20 && submerged[1] > 60 && submerged[2] > 90,
        "underwater sky tint missing: {submerged:?}"
    );
    h.camera.set_view(
        vec3(0.0, -2.0, 3.0),
        vec3(0.0, 0.0, 3.0),
        vec3(0.0, 0.0, 1.0),
    );
    let above_with_feature_enabled = h.frame().pixel(4, 120);
    h.data.render_config.water.underwater.enabled = false;
    assert_eq!(h.frame().pixel(4, 120), above_with_feature_enabled);
}
