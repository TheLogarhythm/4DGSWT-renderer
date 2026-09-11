#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::EventLoop;

mod app;
mod camera;
mod control;
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
mod texture;
mod utils;
mod wangtile;

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
