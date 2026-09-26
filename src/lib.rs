#![cfg_attr(target_arch = "wasm32", feature(stdarch_wasm_atomic_wait))]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::EventLoop;

mod app;
mod camera;
mod caustics;
mod control;
mod cubed_sphere;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod cubed_sphere_gpu_tests;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod cubed_sphere_render_tests;
mod dynamic_archive;
mod gui;
mod motion;
mod motion_authoring_ui;
mod motion_behavior;
mod motion_brush;
mod motion_brush_gpu;
mod motion_brush_ui;
mod motion_controller_gpu;
mod motion_controller_palette;
mod motion_gpu;
mod motion_graph;
mod motion_graph_analysis;
mod motion_graph_playback;
mod motion_session;
mod motion_shader;
mod motion_spatial_variation;
mod motion_tagging;
mod profiler;
mod proxy;
mod renderer;
mod scene;
mod scene_archive;
mod skybox;
mod state;
mod structure;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod test_support;
mod texture;
mod underwater;
mod utils;
mod wangtile;
mod water;
mod water_environment;
mod water_hits;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod water_tests;
mod worker;

use app::App;

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
