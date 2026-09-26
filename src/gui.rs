// https://github.com/kaphula/winit-egui-wgpu-template/blob/master/src/egui_tools.rs

use std::sync::Arc;

use egui::Context;
use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, StoreOp, TextureFormat, TextureView};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
use egui_winit::State;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::camera::Camera;
use crate::control::FlyPathControl;
use crate::structure::{GUIStatus, MainChannels, RenderData, UserData, UserDataString};

mod configuration;
mod performance;
mod scene;
mod water;

pub struct GUI {
    state: State,
    renderer: Renderer,
    frame_started: bool,

    pub gui_status: GUIStatus,
    pub config_user_data: UserData,
    config_user_data_string: UserDataString,
    config_confirmed: bool,
    config_lod_count_error_msg: Option<String>,
    config_error_msg: Option<String>,
    config_next_id: u32,
    archive_error_msg: Option<String>,
    archive_load_requested: bool,
    archive_load_in_progress: bool,
}

impl GUI {
    pub fn context(&self) -> &Context {
        self.state.egui_ctx()
    }

    pub fn new(device: &Device, output_color_format: TextureFormat, window: Arc<Window>) -> Self {
        let egui_context = Context::default();

        let egui_state = egui_winit::State::new(
            egui_context,
            egui::viewport::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024), // default dimension is 2048
        );

        let egui_renderer_options = RendererOptions::default();

        let egui_renderer = Renderer::new(device, output_color_format, egui_renderer_options);

        Self {
            state: egui_state,
            renderer: egui_renderer,
            frame_started: false,

            gui_status: GUIStatus::Unloaded,
            config_user_data: UserData::new(),
            config_user_data_string: UserDataString::new(),
            config_confirmed: false,
            config_lod_count_error_msg: None,
            config_error_msg: None,
            config_next_id: 0,
            archive_error_msg: None,
            archive_load_requested: false,
            archive_load_in_progress: false,
        }
    }

    pub fn show_archive_unloaded(&mut self, error: Option<String>) {
        self.gui_status = GUIStatus::Unloaded;
        self.archive_error_msg = error;
        self.archive_load_in_progress = false;
    }

    pub fn show_archive_loading(&mut self) {
        self.archive_error_msg = None;
        self.archive_load_in_progress = true;
    }

    pub fn show_archive_loaded(&mut self) {
        self.archive_error_msg = None;
        self.archive_load_in_progress = false;
        self.gui_status = GUIStatus::Config;
    }

    pub fn take_archive_load_request(&mut self) -> bool {
        std::mem::take(&mut self.archive_load_requested)
    }

    pub fn handle_input(&mut self, window: Arc<Window>, event: &WindowEvent) {
        let _ = self.state.on_window_event(window.as_ref(), event);
    }

    pub fn render(
        &mut self,
        channels: Option<&mut MainChannels>,
        camera: &Camera,
        rd: &mut RenderData,
        fly_path_control: &mut FlyPathControl,
    ) {
        if let Some(motion) = rd.motion.as_mut() {
            crate::motion_session::poll(motion);
        }
        if self.gui_status == GUIStatus::Unloaded {
            self.render_archive_picker();
            return;
        }

        let Some(channels) = channels else {
            self.show_archive_unloaded(Some(
                "Archive runtime is unavailable; choose the archive again.".to_string(),
            ));
            return;
        };

        if self.gui_status == GUIStatus::Render
            && let Some(spatial) = rd
                .motion
                .as_ref()
                .and_then(|motion| motion.spatial.as_ref())
        {
            let scale = rd.render_config.scene_scale;
            crate::motion_brush_ui::paint_viewport_preview(
                &self.context().clone(),
                camera,
                [scale.x, scale.y],
                spatial,
            );
        }

        match self.gui_status {
            GUIStatus::Unloaded => unreachable!("unloaded GUI returned before channel access"),
            GUIStatus::Config => {
                self.render_configuration(channels, rd);
            }
            GUIStatus::PostConfig => {
                // todo!()
            }
            GUIStatus::Render => {
                self.render_scene_controls(camera, rd);

                if rd.show_motion_authoring_menu {
                    let open = &mut rd.show_motion_authoring_menu;
                    if let Some(motion) = rd.motion.as_mut() {
                        crate::motion_authoring_ui::show(&self.context().clone(), open, motion);
                    } else {
                        *open = false;
                    }
                }

                self.render_performance(rd);

                self.render_fly_path_controls(channels, camera, rd, fly_path_control);
            }
        }
    }

    pub fn ppp(&mut self, v: f32) {
        self.context().set_pixels_per_point(v);
    }

    pub fn begin_frame(&mut self, window: Arc<Window>) {
        let raw_input = self.state.take_egui_input(window.as_ref());
        self.state.egui_ctx().begin_pass(raw_input);
        self.frame_started = true;
    }

    pub fn end_frame_and_draw(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        window: &Window,
        window_surface_view: &TextureView,
        screen_descriptor: ScreenDescriptor,
    ) {
        if !self.frame_started {
            panic!("begin_frame must be called before end_frame_and_draw can be called!");
        }

        self.ppp(screen_descriptor.pixels_per_point);

        let full_output = self.state.egui_ctx().end_pass();

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .state
            .egui_ctx()
            .tessellate(full_output.shapes, self.state.egui_ctx().pixels_per_point());
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);
        let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: window_surface_view,
                resolve_target: None,
                ops: egui_wgpu::wgpu::Operations {
                    load: egui_wgpu::wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            label: Some("egui main render pass"),
            occlusion_query_set: None,
        });

        self.renderer
            .render(&mut rpass.forget_lifetime(), &tris, &screen_descriptor);
        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }

        self.frame_started = false;
    }
}
