use std::{f32, sync::Arc, sync::mpsc};

use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::ActiveEventLoop,
    keyboard::KeyCode,
    window::Window,
};

use crate::camera::Camera;
use crate::control::FlyPathControl;
use crate::control::{CameraControl, KeyboardFlyControl};
use crate::gui::GUI;
use crate::log;
use crate::motion::{MotionError, MotionPlayback, TimelineSample};
use crate::motion_brush::{
    canonical_brush_xy, motion_brush_accepts_pointer, motion_brush_drag_should_end,
    motion_brush_preview_hover,
};
use crate::motion_tagging::{AuthoredOccurrenceRegistry, AuthoredTagRequest};
use crate::profiler::{FrameProfiler, renderer_required_features};
use crate::proxy::Proxy;
use crate::renderer::GSWTRenderer;
use crate::scene_archive::{ArchiveLoadError, LoadedArchive, pick_archive};
use crate::skybox::Skybox;
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;
use crate::wangtile::WangTile;

pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    pub window: Arc<Window>,

    gui: GUI,
    gswt_renderer: Option<GSWTRenderer>,
    skybox: Skybox,
    proxy: Proxy,

    camera: Camera,
    keyboard_fly_control: KeyboardFlyControl,
    fly_path_control: FlyPathControl,
    channels: Option<MainChannels>,
    worker_thread_handle: Option<wasm_thread::JoinHandle<()>>,
    #[cfg(target_arch = "wasm32")]
    archive_selection_tx: mpsc::Sender<Result<Option<LoadedArchive>, ArchiveLoadError>>,
    #[cfg(target_arch = "wasm32")]
    archive_selection_rx: mpsc::Receiver<Result<Option<LoadedArchive>, ArchiveLoadError>>,
    archive_picker_active: bool,
    render_data: RenderData,
    input_status: InputStatus,
    profiler: FrameProfiler,
    cursor_position: Option<[f32; 2]>,
    motion_brush_dragging: bool,
}

enum ArchiveSelectionOutcome {
    Cancelled,
    Failed(String),
    Loaded(LoadedArchive),
}

fn classify_archive_selection(
    selection: Result<Option<LoadedArchive>, ArchiveLoadError>,
) -> ArchiveSelectionOutcome {
    match selection {
        Ok(Some(archive)) => ArchiveSelectionOutcome::Loaded(archive),
        Ok(None) => ArchiveSelectionOutcome::Cancelled,
        Err(error) => ArchiveSelectionOutcome::Failed(error.to_string()),
    }
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let now = get_time_milliseconds();

        // let image_width = 1920;
        // let image_height = 1080;
        // let viewport_size = PhysicalSize::<u32>::new(image_width, image_height);
        // let _ = window.request_inner_size(viewport_size);
        let viewport_size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let required_features = renderer_required_features(adapter.features());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::default()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        log!("{:?}", surface_caps);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        log!("{:?}", surface_format);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: viewport_size.width,
            height: viewport_size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let camera = Camera::new_perspective(
            viewport_size,
            vec3(0.0, 0.0, 5.0),
            vec3(0.0, 1.0, 5.0),
            vec3(0.0, 0.0, 1.0),
            degrees(45.0),
            0.1,    //0.2,
            2400.0, //200.0,
        );

        let keyboard_fly_control = KeyboardFlyControl::new();
        let fly_path_control = FlyPathControl::new();

        let gui = GUI::new(&device, config.format, window.clone());
        let skybox = Skybox::new(&device, &config);
        let proxy = Proxy::new(&device, &config);
        let profiler = FrameProfiler::new(&device, &queue, 200);
        #[cfg(target_arch = "wasm32")]
        let (archive_selection_tx, archive_selection_rx) = mpsc::channel();

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,

            gui,
            gswt_renderer: None,
            skybox,
            proxy,

            camera,
            keyboard_fly_control,
            fly_path_control,
            channels: None,
            worker_thread_handle: None,
            #[cfg(target_arch = "wasm32")]
            archive_selection_tx,
            #[cfg(target_arch = "wasm32")]
            archive_selection_rx,
            archive_picker_active: false,
            render_data: RenderData::new(1),
            input_status: InputStatus::new(),
            profiler,
            cursor_position: None,
            motion_brush_dragging: false,
        };
        state.apply_archive_selection(pick_archive().await);
        log!("Init completed in {}ms", get_time_milliseconds() - now);
        Ok(state)
    }

    fn apply_archive_selection(
        &mut self,
        selection: Result<Option<LoadedArchive>, ArchiveLoadError>,
    ) {
        self.archive_picker_active = false;
        match classify_archive_selection(selection) {
            ArchiveSelectionOutcome::Cancelled => self.gui.show_archive_unloaded(None),
            ArchiveSelectionOutcome::Failed(message) => {
                self.gui.show_archive_unloaded(Some(message));
            }
            ArchiveSelectionOutcome::Loaded(archive) => {
                if let Err(error) = self.install_archive(archive) {
                    self.gui.show_archive_unloaded(Some(error));
                }
            }
        }
    }

    fn install_archive(&mut self, archive: LoadedArchive) -> Result<(), String> {
        let max_lod_count = archive.scenes.len();
        log!("max lod count: {}", max_lod_count);
        let mut wang = WangTile::new(archive)?;
        let renderer = GSWTRenderer::new(&self.device, &self.queue, &self.config, wang.preload())?;
        let mut render_data = RenderData::new(max_lod_count);
        if let (Some(summary), Some(source_duration_seconds)) = (
            renderer.motion_summary().cloned(),
            renderer.motion_source_duration_seconds(),
        ) {
            render_data.motion = Some(
                MotionRenderData::new(
                    summary,
                    source_duration_seconds,
                    renderer.motion_graph(),
                    renderer.motion_graph_error().map(str::to_string),
                )
                .map_err(|error| error.to_string())?,
            );
        }
        let (channels, worker_thread_handle) = launch_worker_thread(wang);
        self.gswt_renderer = Some(renderer);
        self.channels = Some(channels);
        self.worker_thread_handle = Some(worker_thread_handle);
        self.render_data = render_data;
        self.gui.show_archive_loaded();
        Ok(())
    }

    fn service_archive_picker(&mut self) {
        #[cfg(target_arch = "wasm32")]
        if self.archive_picker_active {
            if let Ok(selection) = self.archive_selection_rx.try_recv() {
                self.apply_archive_selection(selection);
            }
        }

        if !self.gui.take_archive_load_request() || self.archive_picker_active {
            return;
        }
        self.archive_picker_active = true;
        self.gui.show_archive_loading();

        #[cfg(not(target_arch = "wasm32"))]
        self.apply_archive_selection(pollster::block_on(pick_archive()));

        #[cfg(target_arch = "wasm32")]
        {
            let sender = self.archive_selection_tx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let _ = sender.send(pick_archive().await);
            });
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
            self.camera.set_viewport(width, height);
            self.render_data.depth_texture = Some(Texture::create_depth_texture(
                &self.device,
                &self.config,
                "depth_texture",
            ));
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, key: KeyCode, pressed: bool) {
        if key == KeyCode::Escape && pressed {
            event_loop.exit();
            if let Some(handle) = self.worker_thread_handle.take() {
                let _ = handle.join();
            }
        }
        self.input_status.update(key, pressed);

        let rd = &mut self.render_data;
        match &mut rd.camera_control_type {
            CameraControl::KeyboardFly => {
                self.keyboard_fly_control.handle_key(key, pressed);
            }
            CameraControl::FlyPath => {}
        }

        if pressed && !self.gui.context().wants_keyboard_input() {
            match key {
                KeyCode::KeyM => {
                    rd.show_main_menu = !rd.show_main_menu;
                }
                KeyCode::KeyP => {
                    rd.show_perf_menu = !rd.show_perf_menu;
                }
                _ => {}
            }
        }
    }

    pub fn handle_mouse_input(&mut self, mouse_state: ElementState, button: MouseButton) {
        if button != MouseButton::Left {
            return;
        }
        if mouse_state == ElementState::Released && self.motion_brush_dragging {
            self.finish_motion_brush_stroke();
            return;
        }
        if mouse_state != ElementState::Pressed {
            return;
        }
        let brush_enabled = self
            .render_data
            .motion
            .as_ref()
            .and_then(|motion| motion.spatial.as_ref())
            .is_some_and(|spatial| spatial.enabled);
        if !motion_brush_accepts_pointer(
            brush_enabled,
            self.gui.context().wants_pointer_input(),
            true,
        ) {
            return;
        }
        let Some(world_xy) = self.motion_brush_world_xy() else {
            return;
        };
        if let Some(spatial) = self
            .render_data
            .motion
            .as_mut()
            .and_then(|motion| motion.spatial.as_mut())
        {
            match spatial.begin_stroke(world_xy) {
                Ok(()) => self.motion_brush_dragging = true,
                Err(error) => {
                    if let Some(motion) = self.render_data.motion.as_mut() {
                        motion.spatial_error = Some(error.to_string());
                    }
                }
            }
        }
    }

    pub fn handle_mouse_moved(&mut self, position: PhysicalPosition<f64>) {
        self.cursor_position = Some([position.x as f32, position.y as f32]);
        let world_xy = self.motion_brush_world_xy();
        let gui_owns_pointer =
            self.gui.context().wants_pointer_input() || self.gui.context().is_pointer_over_area();
        let brush_enabled = self
            .render_data
            .motion
            .as_ref()
            .and_then(|motion| motion.spatial.as_ref())
            .is_some_and(|spatial| spatial.enabled);
        if self.motion_brush_dragging
            && motion_brush_drag_should_end(brush_enabled, gui_owns_pointer, world_xy.is_some())
        {
            self.finish_motion_brush_stroke();
        }
        let Some(motion) = self.render_data.motion.as_mut() else {
            return;
        };
        let Some(spatial) = motion.spatial.as_mut() else {
            return;
        };
        spatial.set_hover(motion_brush_preview_hover(
            brush_enabled,
            gui_owns_pointer,
            world_xy,
        ));
        if self.motion_brush_dragging
            && let Some(world_xy) = world_xy
            && let Err(error) = spatial.extend_stroke(world_xy)
        {
            motion.spatial_error = Some(error.to_string());
        }
    }

    fn motion_brush_world_xy(&self) -> Option<[f32; 2]> {
        let pixel = self.cursor_position?;
        let hit = self.camera.world_ray(pixel).ok()?.intersect_z_plane(0.0)?;
        let scale = self.render_data.render_config.scene_scale;
        let canonical = canonical_brush_xy([hit.x, hit.y], [scale.x, scale.y]).ok()?;
        self.render_data
            .motion
            .as_ref()
            .and_then(|motion| motion.spatial.as_ref())
            .filter(|spatial| spatial.contains_world_xy(canonical, spatial.radius))
            .map(|_| canonical)
    }

    fn finish_motion_brush_stroke(&mut self) {
        self.motion_brush_dragging = false;
        if let Some(spatial) = self
            .render_data
            .motion
            .as_mut()
            .and_then(|motion| motion.spatial.as_mut())
            && let Err(error) = spatial.end_stroke()
            && let Some(motion) = self.render_data.motion.as_mut()
        {
            motion.spatial_error = Some(error.to_string());
        }
    }

    fn refresh_motion_brush_hover(&mut self) {
        let brush_enabled = self
            .render_data
            .motion
            .as_ref()
            .and_then(|motion| motion.spatial.as_ref())
            .is_some_and(|spatial| spatial.enabled);
        let gui_owns_pointer =
            self.gui.context().wants_pointer_input() || self.gui.context().is_pointer_over_area();
        let world_xy = if brush_enabled && !gui_owns_pointer {
            self.motion_brush_world_xy()
        } else {
            None
        };
        if let Some(spatial) = self
            .render_data
            .motion
            .as_mut()
            .and_then(|motion| motion.spatial.as_mut())
        {
            spatial.set_hover(motion_brush_preview_hover(
                brush_enabled,
                gui_owns_pointer,
                world_xy,
            ));
        }
    }

    pub fn handle_gui(&mut self, event: &WindowEvent) {
        self.gui.handle_input(self.window.clone(), event);
    }

    pub fn update(&mut self) {
        let rd = &mut self.render_data;
        match &mut rd.camera_control_type {
            CameraControl::KeyboardFly => {
                rd.update_worker = self.keyboard_fly_control.update(
                    &mut self.camera,
                    rd.frame_time_ma.calc().0 as f32,
                    rd.lockon_center,
                );
            }
            CameraControl::FlyPath => {
                rd.update_worker = self.fly_path_control.handle_events(&mut self.camera);
            }
        }
        self.refresh_motion_brush_hover();
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();
        self.service_archive_picker();

        if !self.is_surface_configured {
            return Ok(());
        }

        let output = self.surface.get_current_texture()?;
        let profiler_frame_start = get_time_milliseconds();
        if self.render_data.profiler_reset_requested {
            self.profiler.clear();
            self.render_data.profiler_reset_requested = false;
        }
        self.profiler
            .begin_frame(&self.device, self.render_data.profiler_enabled);
        if let Some(summary) = self
            .gswt_renderer
            .as_ref()
            .and_then(GSWTRenderer::motion_summary)
        {
            let counters = self.profiler.counters_mut();
            counters.archive_rows = summary.total_rows;
            counters.total_members = summary.tile_count.saturating_mul(summary.lod_count);
        }
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if self.gui.gui_status != GUIStatus::Render {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Archive UI background clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        match self.gui.gui_status {
            GUIStatus::Unloaded => {}
            GUIStatus::Config => {}
            GUIStatus::PostConfig => {
                let (Some(channels), Some(renderer)) =
                    (self.channels.as_mut(), self.gswt_renderer.as_mut())
                else {
                    self.gui.show_archive_unloaded(Some(
                        "Archive runtime is incomplete; choose the archive again.".to_string(),
                    ));
                    return Ok(());
                };
                // Clear previous build / sort data
                while channels.rx_scene_data.try_recv().is_ok() {}
                while channels.rx_sort_data.try_recv().is_ok() {}

                if let Ok(wang_user_data) = channels.rx_user_data.try_recv() {
                    if wang_user_data.config_id == self.gui.config_user_data.config_id {
                        renderer.configure(
                            &self.device,
                            &self.queue,
                            &wang_user_data,
                            &self.render_data,
                        );
                        if let Some(motion) = self.render_data.motion.as_mut() {
                            if let Err(error) =
                                motion.configure_spatial_authoring(&wang_user_data, [0, 0])
                            {
                                motion.spatial_error = Some(error);
                            } else {
                                let field_data_active = motion
                                    .spatial
                                    .as_ref()
                                    .is_some_and(|spatial| spatial.field_data_active());
                                let render_path_active = motion
                                    .spatial
                                    .as_ref()
                                    .is_some_and(|spatial| spatial.render_path_active());
                                let compilation_ok = !render_path_active
                                    || match motion.compile_spatial_field() {
                                        Ok(()) => true,
                                        Err(error) => {
                                            motion.spatial_error = Some(error.to_string());
                                            false
                                        }
                                    };
                                if compilation_ok
                                    && let (Some(spatial), Some(compiled)) =
                                        (&mut motion.spatial, &motion.compiled_spatial)
                                {
                                    if spatial.render_path_active() {
                                        let mut dirty = field_data_active
                                            .then(|| spatial.take_dirty_regions())
                                            .unwrap_or_default();
                                        dirty.extend_from_slice(compiled.dirty_regions());
                                        dirty.dedup();
                                        let sync_result = if field_data_active {
                                            renderer.sync_motion_field(
                                                &self.device,
                                                &self.queue,
                                                spatial.cache(),
                                                compiled,
                                                &dirty,
                                            )
                                        } else {
                                            renderer
                                                .sync_motion_preview(&self.device, spatial.cache());
                                            Ok(0)
                                        };
                                        if let Err(error) = sync_result {
                                            spatial.restore_dirty_regions(dirty);
                                            renderer.disable_motion_field_rendering();
                                            motion.spatial_error = Some(error);
                                        } else {
                                            renderer.update_motion_field_uniform(
                                                &self.queue,
                                                spatial.cache(),
                                                field_data_active,
                                                spatial.overlay,
                                                compiled.palette_count(),
                                                spatial.preview(),
                                            );
                                        }
                                    } else {
                                        renderer.disable_motion_field_rendering();
                                    }
                                } else if !compilation_ok {
                                    renderer.disable_motion_field_rendering();
                                }
                            }
                        }
                        if let Some(skybox_rawtex) = &self.render_data.skybox_rawtex {
                            self.render_data.use_skybox = true;
                            self.skybox
                                .configure(&self.device, &self.queue, skybox_rawtex);
                        }
                        if let Some(proxy_rawtex) = &self.render_data.proxy_rawtex {
                            self.render_data.use_proxy = true;
                            self.proxy.configure(
                                &self.device,
                                &self.queue,
                                &wang_user_data,
                                &self.render_data,
                                proxy_rawtex,
                            );
                        }

                        self.gui.gui_status = GUIStatus::Render;
                        self.gui.config_user_data = wang_user_data;
                        self.render_data.frame_prev = get_time_milliseconds();

                        log!("Config {} ready.", self.gui.config_user_data.config_id);
                    }
                }
            }
            GUIStatus::Render => {
                let (Some(channels), Some(renderer)) =
                    (self.channels.as_mut(), self.gswt_renderer.as_mut())
                else {
                    self.gui.show_archive_unloaded(Some(
                        "Archive runtime is incomplete; choose the archive again.".to_string(),
                    ));
                    return Ok(());
                };
                let now = get_time_milliseconds();
                let rd = &mut self.render_data;
                let frame_delta_milliseconds = (now - rd.frame_prev).max(0.0);
                rd.frame_time_ma.add(frame_delta_milliseconds);
                rd.frame_prev = now;

                if let Some(motion) = rd.motion.as_mut() {
                    let motion_cpu_start = get_time_milliseconds();
                    let frame_delta_seconds = (frame_delta_milliseconds / 1000.0) as f32;
                    match motion.sample_for_frame(frame_delta_seconds) {
                        Ok(Some(sample)) => {
                            let timestamp_writes = self.profiler.motion_timestamp_writes();
                            match renderer.update_motion(
                                &self.queue,
                                &mut encoder,
                                sample,
                                motion.playback.motion_enabled,
                                motion.playback.channel_selection.mask(),
                                motion.channel_gains(),
                                timestamp_writes,
                            ) {
                                Err(error) => {
                                    self.profiler.cancel_motion_timestamp();
                                    motion.playback.pause();
                                    motion.error = Some(error);
                                }
                                Ok(work) => {
                                    motion.error = None;
                                    if let Some(work) = work {
                                        if rd.profiler_enabled {
                                            let counters = self.profiler.counters_mut();
                                            counters.motion_rows = work.rows;
                                            counters.uploaded_bytes = counters
                                                .uploaded_bytes
                                                .saturating_add(work.uploaded_bytes);
                                            counters.motion_dispatched = true;
                                            counters.motion_blend = work.blend;
                                        }
                                    } else {
                                        self.profiler.cancel_motion_timestamp();
                                    }
                                }
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            motion.playback.pause();
                            motion.error = Some(error.to_string());
                        }
                    }
                    if let Err(error) = motion.advance_local_controllers(frame_delta_seconds) {
                        motion.spatial_error = Some(error.to_string());
                    }
                    self.profiler
                        .record_motion_cpu((get_time_milliseconds() - motion_cpu_start).max(0.0));
                }

                if rd.cur_scene_data_id.is_some() && rd.cur_sort_data_id.is_some() {
                    if let Ok(f) = channels.rx_sort_time.try_recv() {
                        rd.sort_time_ma.add(f);
                        rd.sort_trigger_ma.add(1.0);
                        self.profiler.record_worker_sort_cpu(f);
                    } else {
                        rd.sort_trigger_ma.add(0.0);
                    }

                    if let Ok(f) = channels.rx_build_time.try_recv() {
                        rd.build_time_ma.add(f);
                        rd.build_trigger_ma.add(1.0);
                        self.profiler.record_worker_build_cpu(f);
                    } else {
                        rd.build_trigger_ma.add(0.0);
                    }

                    if rd.set_cam_clicked {
                        self.camera.set_view(
                            rd.set_cam_pos,
                            rd.set_cam_dir + rd.set_cam_pos,
                            rd.set_cam_up,
                        );
                        rd.set_cam_clicked = false;
                    }
                }

                if rd.update_worker {
                    // Send cam pos to worker thread
                    let _ = channels
                        .tx_build_info
                        .send((!rd.lock_tile, *self.camera.position()));

                    // Send view_proj to worker thread
                    if !rd.lock_sort {
                        let _ = channels.tx_vp.send(self.camera.view_proj());
                    }
                }

                // Recv scene data from worker thread
                if let Ok(scene) = channels.rx_scene_data.try_recv() {
                    if rd.cur_scene_data_id.is_some()
                        && scene.scene_id == rd.cur_scene_data_id.unwrap()
                    {
                        // Second condition impossible for now
                        rd.cur_scene_data = Some(scene);
                    } else {
                        rd.next_scene_data_id = Some(scene.scene_id);
                        rd.next_scene_data = Some(scene);
                    }
                }

                // Recv sort data from worker thread
                if let Ok(sort_data) = channels.rx_sort_data.try_recv() {
                    let request_revision = rd
                        .motion
                        .as_ref()
                        .map(MotionRenderData::current_membership_request_revision)
                        .unwrap_or(0);
                    if accept_authored_sort(request_revision, &sort_data) {
                        if rd.cur_sort_data_id.is_some()
                            && sort_data.scene_id == rd.cur_sort_data_id.unwrap()
                        {
                            rd.cur_sort_data = Some(sort_data);
                            rd.sort_revision = rd.sort_revision.wrapping_add(1);
                        } else {
                            rd.next_sort_data_id = Some(sort_data.scene_id);
                            rd.next_sort_data = Some(sort_data);
                        }
                    }
                }

                if let Some(center_tile) = rd
                    .next_scene_data
                    .as_ref()
                    .or(rd.cur_scene_data.as_ref())
                    .map(|scene| [scene.center_coord.x, scene.center_coord.y])
                    && let Some(motion) = rd.motion.as_mut()
                    && let Some(spatial) = motion.spatial.as_mut()
                    && spatial.layout().center_tile() != center_tile
                    && let Err(error) = spatial.recenter(center_tile)
                {
                    motion.spatial_error = Some(error.to_string());
                }

                if let Some(motion) = rd.motion.as_mut() {
                    let render_path_active = motion
                        .spatial
                        .as_ref()
                        .is_some_and(|spatial| spatial.render_path_active());
                    let compilation_ok = !render_path_active
                        || match motion.compile_spatial_field() {
                            Ok(()) => true,
                            Err(error) => {
                                motion.spatial_error = Some(error.to_string());
                                false
                            }
                        };
                    if compilation_ok {
                        match motion.take_membership_request() {
                            Ok(Some(request)) => {
                                let request_revision = request.request_revision;
                                match channels.tx_authored_tag_request.send(request) {
                                    Ok(()) => {
                                        motion.acknowledge_membership_request(request_revision)
                                    }
                                    Err(error) => {
                                        motion.spatial_error = Some(format!(
                                            "Motion membership update will be retried: {error}"
                                        ));
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(error) => motion.spatial_error = Some(error.to_string()),
                        }
                    }
                }

                // Install a new scene only with sort and authored-registry data
                // produced for the membership request that is current now.
                let request_revision = rd
                    .motion
                    .as_ref()
                    .map(MotionRenderData::current_membership_request_revision)
                    .unwrap_or(0);
                let next_pair_matches = rd.next_scene_data_id.is_some()
                    && rd.next_sort_data_id == rd.next_scene_data_id
                    && rd
                        .next_sort_data
                        .as_ref()
                        .is_some_and(|sort| accept_authored_sort(request_revision, sort));
                if next_pair_matches {
                    rd.cur_scene_data = rd.next_scene_data.take();
                    rd.cur_sort_data = rd.next_sort_data.take();
                    rd.sort_revision = rd.sort_revision.wrapping_add(1);
                    rd.cur_scene_data_id = rd.next_scene_data_id.take();
                    rd.cur_sort_data_id = rd.next_sort_data_id.take();
                }

                let awaiting_scene_commit = rd.next_scene_data_id.is_some();
                if !awaiting_scene_commit && let Some(motion) = rd.motion.as_mut() {
                    let field_data_active = motion
                        .spatial
                        .as_ref()
                        .is_some_and(|spatial| spatial.field_data_active());
                    let compilation_ok = motion.compiled_spatial.is_some()
                        || !motion
                            .spatial
                            .as_ref()
                            .is_some_and(|spatial| spatial.render_path_active());
                    if compilation_ok
                        && let (Some(spatial), Some(compiled)) =
                            (&mut motion.spatial, &motion.compiled_spatial)
                    {
                        if spatial.render_path_active() {
                            let mut dirty = field_data_active
                                .then(|| spatial.take_dirty_regions())
                                .unwrap_or_default();
                            dirty.extend_from_slice(compiled.dirty_regions());
                            dirty.dedup();
                            let sync_result = if field_data_active {
                                renderer.sync_motion_field(
                                    &self.device,
                                    &self.queue,
                                    spatial.cache(),
                                    compiled,
                                    &dirty,
                                )
                            } else {
                                renderer.sync_motion_preview(&self.device, spatial.cache());
                                Ok(0)
                            };
                            match sync_result {
                                Ok(uploaded_bytes) => {
                                    if rd.profiler_enabled {
                                        let counters = self.profiler.counters_mut();
                                        counters.uploaded_bytes =
                                            counters.uploaded_bytes.saturating_add(uploaded_bytes);
                                    }
                                    motion.spatial_error = None;
                                    renderer.update_motion_field_uniform(
                                        &self.queue,
                                        spatial.cache(),
                                        field_data_active,
                                        spatial.overlay,
                                        compiled.palette_count(),
                                        spatial.preview(),
                                    );
                                }
                                Err(error) => {
                                    spatial.restore_dirty_regions(dirty);
                                    renderer.disable_motion_field_rendering();
                                    motion.spatial_error = Some(error);
                                }
                            }
                        } else {
                            renderer.disable_motion_field_rendering();
                        }
                    } else if !compilation_ok {
                        renderer.disable_motion_field_rendering();
                    }
                }

                let authored_request = (!awaiting_scene_commit)
                    .then(|| rd.motion.as_ref())
                    .flatten()
                    .and_then(|motion| {
                        let spatial = motion.spatial.as_ref()?;
                        if !spatial.field_data_active() {
                            return None;
                        }
                        Some((
                            motion.compiled_spatial.as_ref()?,
                            motion.local_controllers.frames(),
                        ))
                    });
                if let Some((compiled, local_frames)) = authored_request {
                    let timestamp_writes = self.profiler.authored_timestamp_writes();
                    match renderer.update_authored_motion(
                        &self.device,
                        &self.queue,
                        &mut encoder,
                        rd.cur_sort_data.as_ref(),
                        rd.cur_scene_data.as_ref(),
                        compiled,
                        local_frames,
                        timestamp_writes,
                    ) {
                        Ok(Some(work)) => {
                            if rd.profiler_enabled {
                                let counters = self.profiler.counters_mut();
                                counters.authored_registry_occurrences = work.registry_occurrences;
                                counters.authored_tagged_occurrences = work.tagged_occurrences;
                                counters.authored_affected_draws = work.affected_draws;
                                counters.authored_membership_request_revision =
                                    work.membership_request_revision;
                                counters.authored_registry_revision = work.registry_revision;
                                counters.authored_worker_tag_ms = work.worker_tag_ms;
                                counters.authored_registry_installed = work.registry_installed;
                                counters.authored_registry_upload_bytes =
                                    work.registry_upload_bytes;
                                counters.authored_dispatched = work.dispatched;
                                counters.uploaded_bytes =
                                    counters.uploaded_bytes.saturating_add(work.uploaded_bytes);
                            }
                            if work.output_reallocated {
                                log!(
                                    "Authored motion output resized for {} registry occurrences",
                                    work.registry_occurrences
                                );
                            }
                            if !work.dispatched {
                                self.profiler.cancel_authored_timestamp();
                            }
                        }
                        Ok(None) => self.profiler.cancel_authored_timestamp(),
                        Err(error) => {
                            self.profiler.cancel_authored_timestamp();
                            renderer.disable_authored_motion(error.clone());
                            if let Some(motion) = rd.motion.as_mut() {
                                motion.spatial_error = Some(format!(
                                    "Local motion evaluation disabled; global motion remains active: {error}"
                                ));
                            }
                        }
                    }
                } else if !awaiting_scene_commit {
                    renderer.clear_authored_motion();
                }

                if rd.cur_scene_data_id.is_some()
                    && rd.cur_sort_data_id.is_some()
                    && (!rd.freeze_frame || rd.step_frame)
                {
                    rd.step_frame = false;

                    if rd.use_skybox {
                        self.skybox
                            .render(&self.queue, &mut encoder, &view, &self.camera);
                    }

                    if rd.use_proxy {
                        self.proxy
                            .render(&self.queue, &mut encoder, &view, &self.camera, rd);
                    }

                    if rd.render_gs {
                        let render_cpu_start = get_time_milliseconds();
                        let timestamp_writes = self.profiler.render_timestamp_writes();
                        let work = renderer.render(
                            &self.queue,
                            &mut encoder,
                            &view,
                            &self.camera,
                            rd,
                            timestamp_writes,
                        );
                        self.profiler.counters_mut().merge_render_work(work);
                        self.profiler.record_render_cpu(
                            (get_time_milliseconds() - render_cpu_start).max(0.0),
                        );
                    }
                }
            }
        }

        self.render_data.profiler_snapshot = self.profiler.snapshot();

        // GUI render
        {
            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            self.gui.begin_frame(self.window.clone());

            self.gui.render(
                self.channels.as_mut(),
                &self.camera,
                &mut self.render_data,
                &mut self.fly_path_control,
            );

            self.gui.end_frame_and_draw(
                &self.device,
                &self.queue,
                &mut encoder,
                &self.window,
                &view,
                screen_descriptor,
            );
        }

        self.profiler.finish_encoding(&mut encoder);
        let command_buffer = encoder.finish();
        self.profiler.schedule_readback(&command_buffer);
        self.queue.submit(std::iter::once(command_buffer));
        output.present();
        self.profiler
            .record_frame_cpu((get_time_milliseconds() - profiler_frame_start).max(0.0));

        Ok(())
    }
}

fn advance_motion_for_frame(
    playback: &mut MotionPlayback,
    delta_seconds: f32,
) -> Result<Option<TimelineSample>, MotionError> {
    if playback.take_dirty() {
        playback.take_position_dirty();
        return playback.sample().map(Some);
    }
    let sample = playback.advance(delta_seconds)?;
    let changed = playback.take_dirty();
    playback.take_position_dirty();
    Ok(changed.then_some(sample))
}

fn view_matrix_difference(previous: Mat4, current: Mat4) -> f32 {
    let diff = previous - current;
    diff[0][0].abs()
        + diff[0][1].abs()
        + diff[0][2].abs()
        + diff[0][3].abs()
        + diff[1][0].abs()
        + diff[1][1].abs()
        + diff[1][2].abs()
        + diff[1][3].abs()
        + diff[2][0].abs()
        + diff[2][1].abs()
        + diff[2][2].abs()
        + diff[2][3].abs()
        + diff[3][0].abs()
        + diff[3][1].abs()
        + diff[3][2].abs()
        + diff[3][3].abs()
}

fn should_worker_sort(
    force_sort: bool,
    received_view: bool,
    always_sort: bool,
    previous_view: Option<Mat4>,
    current_view: Option<Mat4>,
) -> bool {
    let Some(current_view) = current_view else {
        return false;
    };
    if force_sort {
        return true;
    }
    if !received_view {
        return false;
    }
    always_sort
        || previous_view
            .is_none_or(|previous| view_matrix_difference(previous, current_view) >= 0.01)
}

fn newest_authored_request(
    requests: impl IntoIterator<Item = AuthoredTagRequest>,
) -> Option<AuthoredTagRequest> {
    requests
        .into_iter()
        .max_by_key(|request| request.request_revision)
}

fn accept_authored_sort(current_request_revision: u64, sort_data: &SortData) -> bool {
    let authored = &sort_data.authored;
    if authored.request_revision != current_request_revision
        || authored
            .validate_draw_alignment(sort_data.render_data_vec.len())
            .is_err()
    {
        return false;
    }
    let Some(update) = &authored.update else {
        return true;
    };
    update.request_revision == authored.request_revision
        && update.registry_revision == authored.registry_revision
        && update.scene_id == sort_data.scene_id
        && update.inputs.len() == update.output_base_rows.len()
}

pub fn launch_worker_thread(mut wang: WangTile) -> (MainChannels, wasm_thread::JoinHandle<()>) {
    let (tx_vp, rx_vp) = mpsc::channel::<Mat4>();
    let (tx_build_info, rx_build_info) = mpsc::channel::<(bool, Vec3)>(); // (do_build, camera_pos)
    let (tx_main_user_data, rx_worker_user_data) = mpsc::channel::<UserData>();
    let (tx_authored_tag_request, rx_authored_tag_request) = mpsc::channel::<AuthoredTagRequest>();

    let (tx_worker_user_data, rx_main_user_data) = mpsc::channel::<UserData>(); // Post config user data
    let (tx_sort_data, rx_sort_data) = mpsc::channel::<SortData>();
    let (tx_scene_data, rx_scene_data) = mpsc::channel::<SceneData>();
    let (tx_sort_time, rx_sort_time) = mpsc::channel::<f64>();
    let (tx_build_time, rx_build_time) = mpsc::channel::<f64>();

    let main_channels = MainChannels {
        tx_vp,
        tx_build_info,
        tx_user_data: tx_main_user_data,
        tx_authored_tag_request,
        rx_user_data: rx_main_user_data,
        rx_sort_data,
        rx_scene_data,
        rx_sort_time,
        rx_build_time,
        rx_fly_path_control: None,
        rx_height_tex: None,
        rx_skybox_tex: None,
        rx_proxy_tex: None,
    };

    let worker_channels = WorkerChannels {
        rx_vp,
        rx_build_info,
        rx_user_data: rx_worker_user_data,
        rx_authored_tag_request,
        tx_user_data: tx_worker_user_data,
        tx_sort_data,
        tx_scene_data,
        tx_sort_time,
        tx_build_time,
    };

    // launch another thread for view-dependent splat sorting
    let thread_handle = wasm_thread::spawn({
        let mut cur_camera_pos: Option<Vector3<f32>> = None;
        let mut prev_vp: Option<Mat4> = None;
        let mut last_vp: Option<Mat4> = None;
        let mut next_scene_id: u32 = 0;
        let mut authored_registry = AuthoredOccurrenceRegistry::default();
        let mut force_authored_sort = false;

        move || loop {
            if let Ok(user_data) = worker_channels.rx_user_data.try_recv() {
                let wang_user_data = wang.configure(user_data);
                worker_channels
                    .tx_user_data
                    .send(wang_user_data)
                    .expect("Error sending wang user data");
                cur_camera_pos = None;
                prev_vp = None;
                last_vp = None;
            }

            let mut authored_requests = Vec::new();
            while let Ok(request) = worker_channels.rx_authored_tag_request.try_recv() {
                authored_requests.push(request);
            }
            if let Some(request) = newest_authored_request(authored_requests)
                && request.request_revision > authored_registry.request_revision()
            {
                let scene_id = next_scene_id.saturating_sub(1);
                if let Err(error) = authored_registry.reset(scene_id, request) {
                    log!("Rejected authored membership request: {error}");
                } else {
                    force_authored_sort = true;
                }
            }

            let mut recv_build = false;
            let mut do_build = false;
            let mut camera_pos = Vec3::zero();
            while let Ok((a, b)) = worker_channels.rx_build_info.try_recv() {
                recv_build = true;
                do_build = a;
                camera_pos = b;
            }
            if recv_build {
                cur_camera_pos = Some(camera_pos);

                if do_build && wang.check_update(&camera_pos) {
                    let start = get_time_milliseconds();
                    let mut scene_data = wang.build_tiles(camera_pos);
                    scene_data.scene_id = next_scene_id;
                    if let Err(error) =
                        authored_registry.reset(next_scene_id, authored_registry.current_request())
                    {
                        log!("Failed to retarget authored occurrence registry: {error}");
                    }
                    let build_time = get_time_milliseconds() - start;

                    let _ = worker_channels.tx_scene_data.send(scene_data);
                    let _ = worker_channels.tx_build_time.send(build_time);
                    next_scene_id += 1;
                }
            }

            let mut recv_vp = false;
            let mut view_proj = Mat4::identity();
            while let Ok(a) = worker_channels.rx_vp.try_recv() {
                recv_vp = true;
                view_proj = a;
            }
            if recv_vp {
                last_vp = Some(view_proj);
            }
            if cur_camera_pos.is_some()
                && should_worker_sort(
                    force_authored_sort,
                    recv_vp,
                    wang.user_data.always_sort,
                    prev_vp,
                    last_vp,
                )
            {
                let view_proj = last_vp.expect("worker sort decision requires a current view");
                let start = get_time_milliseconds();
                // TODO: fix cur_camera_pos when lock tile
                match wang.sort_tiles(
                    cur_camera_pos.expect("worker sort requires a camera position"),
                    view_proj,
                    &mut authored_registry,
                ) {
                    Ok(mut sort_data) => {
                        sort_data.scene_id = next_scene_id.saturating_sub(1);
                        prev_vp = Some(view_proj);
                        force_authored_sort = false;
                        let sort_time = get_time_milliseconds() - start;
                        let _ = worker_channels.tx_sort_data.send(sort_data);
                        let _ = worker_channels.tx_sort_time.send(sort_time);
                    }
                    Err(error) => {
                        force_authored_sort = false;
                        log!("Rejected authored worker sort: {error}");
                    }
                }
            }
        }
    });

    (main_channels, thread_handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::{LoopPolicy, MotionPlayback, TimelineSample};
    use crate::motion_behavior::MotionBehaviorPreset;
    use crate::motion_brush::{
        MotionBrushFalloff, MotionBrushRuntime, MotionBrushTool, MotionFieldLayout,
        MotionFieldQuality,
    };
    use crate::motion_controller_palette::MotionRegionId;
    use crate::motion_graph::{
        MotionGraph, MotionGraphNode, MotionGraphSettings, SOURCE_SEGMENT_COUNT,
    };
    use crate::motion_tagging::{AuthoredTagRequest, MotionMembershipSnapshot};
    use crate::scene_archive::{DynamicArchiveSummary, load_archive_bytes};

    fn synthetic_summary() -> DynamicArchiveSummary {
        DynamicArchiveSummary {
            schema_version: 2,
            tile_count: 1,
            lod_count: 1,
            basis_count: 1,
            top_k: 1,
            total_rows: 1,
            backend: "synthetic".into(),
        }
    }

    fn graph_without_jumps() -> Arc<MotionGraph> {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| MotionGraphNode {
                source_segment,
                jumps: Vec::new(),
            })
            .collect();
        Arc::new(MotionGraph::new(nodes, settings).unwrap())
    }

    #[test]
    fn authored_request_forces_sort_without_a_new_or_changed_view() {
        let view = Mat4::identity();
        assert!(should_worker_sort(
            true,
            false,
            false,
            Some(view),
            Some(view)
        ));
        assert!(should_worker_sort(true, false, false, None, Some(view)));
        assert!(!should_worker_sort(
            false,
            false,
            false,
            Some(view),
            Some(view)
        ));
        assert!(!should_worker_sort(
            false,
            true,
            false,
            Some(view),
            Some(view)
        ));
        let changed = Mat4::from_translation(vec3(1.0, 0.0, 0.0));
        assert!(should_worker_sort(
            false,
            true,
            false,
            Some(view),
            Some(changed)
        ));
    }

    #[test]
    fn authored_request_drain_keeps_only_the_greatest_revision() {
        let snapshot = Arc::new(MotionMembershipSnapshot::new(
            3,
            1,
            [1, 1],
            [1, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [0, 0],
            Arc::from([1_u8]),
            Arc::from([1_u8]),
        ));
        let requests = vec![
            AuthoredTagRequest {
                request_revision: 2,
                snapshot: None,
            },
            AuthoredTagRequest {
                request_revision: 3,
                snapshot: Some(snapshot),
            },
            AuthoredTagRequest {
                request_revision: 1,
                snapshot: None,
            },
        ];

        let newest = newest_authored_request(requests).unwrap();
        assert_eq!(newest.request_revision, 3);
        assert!(newest.snapshot.is_some());
    }

    fn graph_with_jump_from_zero() -> Arc<MotionGraph> {
        use crate::motion_graph::{MotionDiscontinuity, MotionJump, rank_eligible_jumps};

        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| MotionGraphNode {
                source_segment,
                jumps: if source_segment == 0 {
                    rank_eligible_jumps(
                        source_segment,
                        vec![MotionJump::unranked(
                            10,
                            MotionDiscontinuity::uniform(0.1, 0.2),
                        )],
                        settings,
                    )
                } else {
                    Vec::new()
                },
            })
            .collect();
        Arc::new(MotionGraph::new(nodes, settings).unwrap())
    }

    fn source_u(sample: TimelineSample) -> f32 {
        match sample {
            TimelineSample::Source { u } => u,
            TimelineSample::Blend { .. } => panic!("expected a source sample"),
        }
    }

    #[test]
    fn paused_playback_does_not_dispatch_an_unchanged_frame() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        playback.pause();
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_some()
        );
        assert!(
            advance_motion_for_frame(&mut playback, 1.0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn playing_advances_by_speed_and_hold_stops_at_the_endpoint() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        assert_eq!(
            source_u(
                advance_motion_for_frame(&mut playback, 0.0)
                    .unwrap()
                    .unwrap()
            ),
            0.0
        );
        playback.set_speed(2.0).unwrap();
        let first = advance_motion_for_frame(&mut playback, 0.25)
            .unwrap()
            .unwrap();
        assert!((source_u(first) - 0.25).abs() < 1.0e-6);
        let endpoint = advance_motion_for_frame(&mut playback, 1.0)
            .unwrap()
            .unwrap();
        assert_eq!(source_u(endpoint), 1.0);
        assert!(!playback.playing);
        assert!(
            advance_motion_for_frame(&mut playback, 1.0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn restart_renders_the_exact_start_before_playback_advances_again() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        let _ = advance_motion_for_frame(&mut playback, 0.0).unwrap();
        let _ = advance_motion_for_frame(&mut playback, 0.5).unwrap();
        playback.restart();

        let restarted = advance_motion_for_frame(&mut playback, 0.25)
            .unwrap()
            .unwrap();
        assert_eq!(source_u(restarted), 0.0);
        let advanced = advance_motion_for_frame(&mut playback, 0.25)
            .unwrap()
            .unwrap();
        assert!((source_u(advanced) - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn interactive_actions_dirty_exactly_one_frame() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_some()
        );

        playback.scrub_normalized(0.4).unwrap();
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_some()
        );
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_none()
        );

        playback.set_policy(LoopPolicy::DirectWrap);
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_some()
        );
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_none()
        );

        playback.set_motion_enabled(false);
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_some()
        );
        assert!(
            advance_motion_for_frame(&mut playback, 0.0)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn invalid_control_values_preserve_the_last_valid_state() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        let original_speed = playback.speed;
        let original_transition = playback.transition_seconds;
        let original_window = playback.blend_window_seconds;
        assert!(playback.set_speed(0.0).is_err());
        assert!(playback.set_transition_seconds(f32::NAN).is_err());
        assert!(playback.set_blend_window_seconds(2.0).is_err());
        assert_eq!(playback.speed, original_speed);
        assert_eq!(playback.transition_seconds, original_transition);
        assert_eq!(playback.blend_window_seconds, original_window);
    }

    #[test]
    fn static_render_data_has_no_motion_state_or_panel_data() {
        let render_data = RenderData::new(6);
        assert!(render_data.motion.is_none());
    }

    #[test]
    fn disabled_graph_preserves_existing_timeline_sampling() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        let expected = motion.playback.sample().unwrap();

        assert_eq!(motion.sample_for_frame(0.0).unwrap(), Some(expected));
    }

    #[test]
    fn enabled_graph_advances_discrete_segments() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        motion.graph_playback.set_enabled(true);
        motion.sample_for_frame(0.0).unwrap();

        motion
            .sample_for_frame(motion.playback.source_duration() / 74.0)
            .unwrap();

        assert_eq!(motion.graph_playback.current_segment(), 1);
    }

    #[test]
    fn channel_toggle_does_not_cancel_an_active_graph_transition() {
        let mut motion = MotionRenderData::new(
            synthetic_summary(),
            2.0,
            Some(graph_with_jump_from_zero()),
            None,
        )
        .unwrap();
        motion.graph_playback.set_enabled(true);
        motion.graph_playback.set_branch_probability(1.0).unwrap();
        motion.graph_playback.set_minimum_dwell_segments(0);
        motion.sample_for_frame(0.0).unwrap();
        let boundary = motion
            .sample_for_frame(motion.playback.source_duration() / 74.0)
            .unwrap()
            .unwrap();
        assert!(boundary.is_blend());

        motion.playback.set_translation_enabled(false);
        let after_toggle = motion.sample_for_frame(0.0).unwrap().unwrap();

        assert!(after_toggle.is_blend());
        assert_eq!(motion.graph_playback.last_selected_edge(), Some((0, 10)));
    }

    #[test]
    fn a_new_dynamic_archive_resets_all_preview_channels_enabled() {
        let summary = synthetic_summary();
        let mut previous = MotionRenderData::new(summary.clone(), 2.0, None, None).unwrap();
        previous.playback.set_translation_enabled(false);
        previous.playback.set_rotation_enabled(false);
        previous.playback.set_scale_enabled(false);

        let replacement = MotionRenderData::new(summary, 2.0, None, None).unwrap();
        assert!(replacement.playback.translation_enabled());
        assert!(replacement.playback.rotation_enabled());
        assert!(replacement.playback.scale_enabled());
    }

    #[test]
    fn a_new_dynamic_archive_initializes_the_artist_motion_regions() {
        let motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();

        let ids = motion
            .regions
            .regions()
            .iter()
            .map(|region| region.id().get())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        assert_eq!(motion.local_controllers.active_local_count(), 16);
        for region in motion.regions.regions() {
            assert!(
                motion
                    .local_controllers
                    .slot_for_region(region.id())
                    .is_some()
            );
        }
    }

    #[test]
    fn local_controllers_advance_even_without_visible_brush_coverage() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        let before = motion.local_controllers.frames().to_vec();

        assert!(motion.advance_local_controllers(0.25).unwrap());

        assert_ne!(motion.local_controllers.frames(), before.as_slice());
    }

    #[test]
    fn missing_graph_keeps_motion_regions_but_skips_local_advancement() {
        let mut motion = MotionRenderData::new(synthetic_summary(), 2.0, None, None).unwrap();
        let before = motion.local_controllers.frames().to_vec();

        assert!(!motion.advance_local_controllers(0.25).unwrap());

        assert_eq!(motion.regions.regions().len(), 4);
        assert_eq!(motion.local_controllers.frames(), before.as_slice());
    }

    #[test]
    fn restart_resets_global_graph_and_local_controller_playback_together() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        motion.advance_local_controllers(0.25).unwrap();
        assert!(
            motion
                .local_controllers
                .frames()
                .iter()
                .any(|frame| frame.sample != TimelineSample::Source { u: 0.0 })
        );

        motion.restart_playback();

        assert!(
            motion
                .local_controllers
                .frames()
                .iter()
                .all(|frame| { frame.sample == TimelineSample::Source { u: 0.0 } })
        );
        assert_eq!(motion.graph_playback.current_segment(), 0);
        assert_eq!(motion.playback.preview_seconds, 0.0);
    }

    #[test]
    fn spatial_compilation_maps_painted_regions_and_is_stable_when_unchanged() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        let layout = MotionFieldLayout::new([1, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut spatial = MotionBrushRuntime::new(layout).unwrap();
        spatial.enabled = true;
        spatial.tool = MotionBrushTool::ApplyBehavior {
            region_id: MotionRegionId::local(4).unwrap(),
        };
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.spatial = Some(spatial);

        motion.compile_spatial_field().unwrap();
        let first_revision = motion.compiled_spatial.as_ref().unwrap().revision();
        assert!(
            motion
                .compiled_spatial
                .as_ref()
                .unwrap()
                .assignments()
                .iter()
                .any(|&slot| slot != 0)
        );

        motion.compile_spatial_field().unwrap();

        assert_eq!(
            motion.compiled_spatial.as_ref().unwrap().revision(),
            first_revision
        );
        assert!(
            motion
                .compiled_spatial
                .as_ref()
                .unwrap()
                .dirty_regions()
                .is_empty()
        );
    }

    #[test]
    fn membership_request_lifecycle_ignores_scalar_edits_and_orders_clear() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        let layout = MotionFieldLayout::new([1, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut spatial = MotionBrushRuntime::new(layout).unwrap();
        spatial.enabled = true;
        spatial.falloff = MotionBrushFalloff::Constant;
        motion.spatial = Some(spatial);

        motion.compile_spatial_field().unwrap();
        assert!(motion.take_membership_request().unwrap().is_none());

        let spatial = motion.spatial.as_mut().unwrap();
        spatial.tool = MotionBrushTool::ApplyBehavior {
            region_id: MotionRegionId::local(4).unwrap(),
        };
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.compile_spatial_field().unwrap();
        let paint = motion.take_membership_request().unwrap().unwrap();
        assert_eq!(paint.request_revision, 1);
        assert!(paint.snapshot.is_some());
        motion.acknowledge_membership_request(1);
        assert!(motion.take_membership_request().unwrap().is_none());

        let spatial = motion.spatial.as_mut().unwrap();
        spatial.tool = MotionBrushTool::AdjustStrength { target: 0.5 };
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.compile_spatial_field().unwrap();
        assert!(motion.take_membership_request().unwrap().is_none());

        let spatial = motion.spatial.as_mut().unwrap();
        spatial.tool = MotionBrushTool::ApplyBehavior {
            region_id: MotionRegionId::local(2).unwrap(),
        };
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.compile_spatial_field().unwrap();
        assert!(motion.take_membership_request().unwrap().is_none());

        let spatial = motion.spatial.as_mut().unwrap();
        spatial.tool = MotionBrushTool::Erase;
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.compile_spatial_field().unwrap();
        let erase = motion.take_membership_request().unwrap().unwrap();
        assert_eq!(erase.request_revision, 2);
        assert!(erase.snapshot.is_none());
    }

    #[test]
    fn membership_request_is_retryable_until_channel_send_is_acknowledged() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        let layout = MotionFieldLayout::new([1, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut spatial = MotionBrushRuntime::new(layout).unwrap();
        spatial.enabled = true;
        spatial.tool = MotionBrushTool::ApplyBehavior {
            region_id: MotionRegionId::local(4).unwrap(),
        };
        spatial.begin_stroke([2.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        motion.spatial = Some(spatial);
        motion.compile_spatial_field().unwrap();

        let first = motion.take_membership_request().unwrap().unwrap();
        let retry = motion.take_membership_request().unwrap().unwrap();
        assert_eq!(retry.request_revision, first.request_revision);
        motion.acknowledge_membership_request(first.request_revision);
        assert_eq!(motion.current_membership_request_revision(), 1);
        assert!(motion.take_membership_request().unwrap().is_none());
    }

    #[test]
    fn authored_sort_revision_gate_rejects_stale_or_cross_scene_registry_data() {
        let stale = SortData {
            scene_id: 9,
            tile_instance_vec: Vec::new(),
            render_data_vec: Vec::new(),
            authored: crate::motion_tagging::AuthoredSortMetadata {
                request_revision: 1,
                registry_revision: 3,
                update: None,
                authored_draws: Vec::new(),
                tagged_occurrences: 0,
                tag_time_ms: 0.0,
            },
        };
        assert!(!accept_authored_sort(2, &stale));

        let matching = SortData {
            authored: crate::motion_tagging::AuthoredSortMetadata {
                request_revision: 2,
                ..stale.authored.clone()
            },
            ..stale.clone()
        };
        assert!(accept_authored_sort(2, &matching));

        let mut mismatched_update = matching.clone();
        mismatched_update.authored.update =
            Some(Arc::new(crate::motion_tagging::AuthoredRegistryUpdate {
                request_revision: 2,
                registry_revision: 3,
                scene_id: 10,
                inputs: Arc::from([]),
                output_base_rows: Arc::from([]),
            }));
        assert!(!accept_authored_sort(2, &mismatched_update));
    }

    #[test]
    fn applying_a_behavior_synchronizes_artist_controls_to_runtime_playback() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        motion
            .authoring
            .apply_preset(MotionBehaviorPreset::GustyWind);

        motion.apply_authoring().unwrap();

        let behavior = motion.authoring.behavior();
        let graph_config = motion.graph_playback.config();
        assert_eq!(motion.playback.speed, 1.15);
        assert_eq!(motion.playback.policy, behavior.endpoint_policy);
        assert!(graph_config.enabled);
        assert_eq!(
            graph_config.branch_probability,
            behavior.branch_probability()
        );
        assert_eq!(
            graph_config.minimum_dwell_segments,
            behavior.minimum_dwell_segments()
        );
        assert_eq!(motion.channel_gains(), behavior.gains);
    }

    #[test]
    fn gain_only_authoring_edit_schedules_exactly_one_render_update() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph_without_jumps()), None)
                .unwrap();
        motion.playback.pause();
        motion.apply_authoring().unwrap();
        let _ = motion.sample_for_frame(0.0).unwrap();
        assert!(motion.sample_for_frame(0.0).unwrap().is_none());

        motion
            .authoring
            .update_behavior(|behavior| behavior.gains.master = 1.4)
            .unwrap();
        motion.apply_authoring().unwrap();

        assert!(motion.sample_for_frame(0.0).unwrap().is_some());
        assert!(motion.sample_for_frame(0.0).unwrap().is_none());
        assert_eq!(motion.channel_gains().master, 1.4);
    }

    #[test]
    fn applying_a_behavior_never_adds_graph_candidates() {
        let graph = graph_without_jumps();
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 2.0, Some(graph.clone()), None).unwrap();
        let before = graph.summary();

        motion
            .authoring
            .apply_preset(MotionBehaviorPreset::SteadyWind);
        motion.apply_authoring().unwrap();

        assert_eq!(graph.summary(), before);
        assert_eq!(graph.summary().jump_count, 0);
    }

    #[test]
    fn hold_policy_pauses_at_the_end_of_a_nonzero_source_range() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 7.4, Some(graph_without_jumps()), None)
                .unwrap();
        let range = motion.authoring.add_range("Middle", 10, 20).unwrap();
        motion.authoring.select_range(range).unwrap();
        motion
            .authoring
            .update_behavior(|behavior| {
                behavior.endpoint_policy = LoopPolicy::Hold;
                behavior.playback_speed = 1.0;
            })
            .unwrap();
        motion.apply_authoring().unwrap();
        motion.sample_for_frame(0.0).unwrap();

        motion.sample_for_frame(1.0).unwrap();

        assert!(!motion.playback.playing);
        assert_eq!(motion.graph_playback.current_segment(), 19);

        motion.playback.play();
        motion.sample_for_frame(0.0).unwrap();

        assert!(motion.playback.playing);
        assert_eq!(motion.graph_playback.current_segment(), 10);
    }

    #[test]
    fn source_range_remains_active_when_stochastic_branching_is_disabled() {
        let mut motion =
            MotionRenderData::new(synthetic_summary(), 7.4, Some(graph_without_jumps()), None)
                .unwrap();
        let range = motion.authoring.add_range("Calm middle", 10, 20).unwrap();
        motion.authoring.select_range(range).unwrap();
        motion.authoring.apply_preset(MotionBehaviorPreset::Calm);
        motion
            .authoring
            .update_behavior(|behavior| {
                behavior.endpoint_policy = LoopPolicy::DirectWrap;
                behavior.playback_speed = 1.0;
            })
            .unwrap();
        motion.apply_authoring().unwrap();
        assert!(!motion.graph_playback.config().enabled);
        motion.sample_for_frame(0.0).unwrap();

        motion.sample_for_frame(1.1).unwrap();

        assert!((10..20).contains(&motion.graph_playback.current_segment()));
    }

    #[test]
    fn cancelled_archive_selection_stays_unloaded_without_an_error() {
        assert!(matches!(
            classify_archive_selection(Ok(None)),
            ArchiveSelectionOutcome::Cancelled
        ));
    }

    #[test]
    fn malformed_archive_selection_becomes_a_retryable_error() {
        let load_error = load_archive_bytes(b"not a zip".to_vec()).unwrap_err();
        match classify_archive_selection(Err(load_error)) {
            ArchiveSelectionOutcome::Failed(message) => {
                assert!(message.contains("invalid ZIP archive"), "{message}");
            }
            _ => panic!("malformed selection must remain in the retryable error state"),
        }
    }
}
