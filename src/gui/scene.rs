//! Scene display, camera editing, motion summary, and fly-path controls.
use super::GUI;
use crate::camera::Camera;
use crate::control::{CameraControl, FlyPathControl, FlyPathFrame};
use crate::log;
use crate::structure::{
    DrawMode, GUIStatus, MainChannels, MotionRenderData, RenderData, SurfaceType,
};

impl GUI {
    pub(super) fn render_scene_controls(&mut self, camera: &Camera, rd: &mut RenderData) {
        if rd.show_main_menu {
            egui::Window::new("GSWT")
                .vscroll(true)
                .show(&self.context().clone(), |ui| {
                    egui::Grid::new("gswt_grid")
                        .num_columns(2)
                        .spacing([40.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            if let Some(motion) = rd.motion.as_mut() {
                                if render_motion_summary_controls(ui, motion) {
                                    rd.show_motion_authoring_menu = true;
                                }
                            }

                            super::performance::summary_rows(ui, rd);

                            ui.add(egui::Label::new("Visualization"));
                            ui.horizontal(|ui| {
                                ui.selectable_value(
                                    &mut rd.render_config.draw_mode,
                                    DrawMode::Normal,
                                    "Normal",
                                );
                                ui.selectable_value(
                                    &mut rd.render_config.draw_mode,
                                    DrawMode::TileID,
                                    "TileID",
                                );
                                ui.selectable_value(
                                    &mut rd.render_config.draw_mode,
                                    DrawMode::TileLOD,
                                    "LOD",
                                );
                                // ui.selectable_value(
                                //     &mut rd.render_config.draw_mode,
                                //     DrawMode::LOD,
                                //     "LOD",
                                // );
                                ui.selectable_value(
                                    &mut rd.render_config.draw_mode,
                                    DrawMode::View,
                                    "View",
                                );
                            });
                            ui.end_row();

                            ui.add(egui::Label::new("Point Cloud"));
                            ui.checkbox(&mut rd.render_config.draw_point_cloud, "");
                            ui.end_row();

                            if rd.render_config.draw_point_cloud {
                                ui.add(egui::Slider::new(
                                    &mut rd.render_config.point_cloud_radius,
                                    0.001..=1.0,
                                ));
                                ui.end_row();
                            }

                            ui.add(egui::Label::new("Culling Threshold"));
                            ui.add(egui::Slider::new(
                                &mut rd.render_config.culling_dist,
                                0.0..=10.0,
                            ));
                            ui.end_row();

                            ui.add(egui::Label::new("Render GS"));
                            ui.checkbox(&mut rd.render_gs, "");
                            ui.end_row();

                            if rd.use_skybox {
                                ui.add(egui::Label::new("Skybox"));
                                ui.checkbox(&mut rd.use_skybox, "");
                                ui.end_row();
                            }

                            if rd.use_proxy
                                && self.config_user_data.surface_type != SurfaceType::Sphere
                            {
                                ui.collapsing("Proxy", |ui| {
                                    ui.add(egui::Label::new("Full Proxy"));
                                    ui.checkbox(&mut rd.render_config.proxy_full, "");
                                    ui.end_row();

                                    ui.add(egui::Label::new("Map Proxy"));
                                    ui.checkbox(&mut rd.render_config.proxy_map, "");
                                    ui.end_row();

                                    ui.add(egui::Label::new("Proxy Height"));
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.proxy_height,
                                        -20.0..=20.0,
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Proxy Width Scale"));
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.proxy_width_scale,
                                        0.1..=50.0,
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Proxy Brightness"));
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.proxy_brightness,
                                        0.0..=3.0,
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Proxy black background"));
                                    ui.checkbox(&mut rd.render_config.proxy_black_background, "");
                                });
                                ui.end_row();
                            }

                            super::water::controls(ui, rd, self.config_user_data.surface_type);

                            ui.label("Clip By Z");
                            ui.checkbox(&mut rd.render_config.use_clip, "");
                            ui.end_row();

                            if rd.render_config.use_clip {
                                ui.label("Clip Height");
                                ui.add(egui::Slider::new(
                                    &mut rd.render_config.clip_height,
                                    -20.0..=20.0,
                                ));
                                ui.end_row();
                            }

                            ui.add(egui::Label::new("Lock (Sort)"));
                            ui.checkbox(&mut rd.lock_sort, "");
                            ui.end_row();

                            ui.add(egui::Label::new("Lock (Tile)"));
                            ui.checkbox(&mut rd.lock_tile, "");
                            ui.end_row();

                            ui.collapsing("Scales", |ui| {
                                ui.add(egui::Label::new("Splat Scale"));
                                ui.add(egui::Slider::new(
                                    &mut rd.render_config.splat_scale,
                                    0.01..=10.0,
                                ));
                                ui.end_row();

                                if self.config_user_data.surface_type != SurfaceType::Sphere {
                                    ui.label("Height Map Scale V");
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.height_map_scale_v,
                                        0.0..=20.0,
                                    ));
                                    ui.end_row();

                                    ui.label("Scene Scale X");
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.scene_scale.x,
                                        0.0..=3.0,
                                    ));
                                    ui.end_row();

                                    ui.label("Scene Scale Y");
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.scene_scale.y,
                                        0.0..=3.0,
                                    ));
                                    ui.end_row();

                                    ui.label("Scene Scale Z");
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.scene_scale.z,
                                        0.0..=3.0,
                                    ));
                                    ui.end_row();
                                } else {
                                    ui.label("Use sphere radius and N to set surface size.");
                                }
                            });
                            ui.end_row();

                            let cam_pos = camera.position();
                            ui.add(egui::Label::new("Camera Position"));
                            ui.label(format!(
                                "({:.2}, {:.2}, {:.2})",
                                cam_pos.x, cam_pos.y, cam_pos.z
                            ));
                            ui.end_row();

                            let cam_dir = camera.view_direction();
                            ui.add(egui::Label::new("Camera Direction"));
                            ui.label(format!(
                                "({:.2}, {:.2}, {:.2})",
                                cam_dir.x, cam_dir.y, cam_dir.z
                            ));
                            ui.end_row();

                            let cam_right = camera.right_direction();
                            ui.add(egui::Label::new("Camera Right"));
                            ui.label(format!(
                                "({:.2}, {:.2}, {:.2})",
                                cam_right.x, cam_right.y, cam_right.z
                            ));
                            ui.end_row();

                            let cam_up = camera.up();
                            ui.add(egui::Label::new("Camera Up"));
                            ui.label(format!(
                                "({:.2}, {:.2}, {:.2})",
                                cam_up.x, cam_up.y, cam_up.z
                            ));
                            ui.end_row();

                            ui.add(egui::Label::new("Camera Position"));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_pos_s.x)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_pos_s.y)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_pos_s.z)
                                        .desired_width(40.0),
                                );
                            });
                            // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_pos.x, cam_pos.y, cam_pos.z));
                            ui.end_row();

                            ui.add(egui::Label::new("Camera Direction"));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_dir_s.x)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_dir_s.y)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_dir_s.z)
                                        .desired_width(40.0),
                                );
                            });
                            // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_dir.x, cam_dir.y, cam_dir.z));
                            ui.end_row();

                            ui.add(egui::Label::new("Camera Up"));
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_up_s.x)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_up_s.y)
                                        .desired_width(40.0),
                                );
                                ui.add(
                                    egui::TextEdit::singleline(&mut rd.cam_up_s.z)
                                        .desired_width(40.0),
                                );
                            });
                            // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_dir.x, cam_dir.y, cam_dir.z));
                            ui.end_row();

                            if ui.button("Set camera").clicked() {
                                rd.parse_camera_config();
                            }
                            ui.end_row();

                            ui.add(egui::Label::new("Camera Control"));
                            ui.horizontal(|ui| {
                                ui.selectable_value(
                                    &mut rd.camera_control_type,
                                    CameraControl::KeyboardFly,
                                    "Keyboard Fly",
                                );
                                ui.selectable_value(
                                    &mut rd.camera_control_type,
                                    CameraControl::FlyPath,
                                    "Fly Path",
                                );
                            });
                            ui.end_row();

                            ui.label("Fly Path Menu");
                            ui.checkbox(&mut rd.show_fly_path_menu, "");
                            ui.end_row();

                            // Debug
                            // if !rd.freeze_frame {
                            //     if ui.button("Freeze").clicked() {
                            //         rd.freeze_frame = true;
                            //     }
                            // } else {
                            //     if ui.button("Unfreeze").clicked() {
                            //         rd.freeze_frame = false;
                            //         rd.step_frame = false;
                            //         rd.render_config.debug_log = false;
                            //     }
                            //     if ui.button("Step").clicked() {
                            //         rd.step_frame = true;
                            //     }
                            //     ui.end_row();
                            //     ui.label("Debug");
                            //     ui.checkbox(&mut rd.render_config.debug_log, "");
                            // }
                            // ui.end_row();

                            if ui.button("Reconfig scene").clicked() {
                                self.gui_status = GUIStatus::Config;
                            }
                            ui.end_row();
                        });
                });
        }
    }
}

impl GUI {
    pub(super) fn render_fly_path_controls(
        &mut self,
        channels: &mut MainChannels,
        camera: &Camera,
        rd: &mut RenderData,
        fly_path_control: &mut FlyPathControl,
    ) {
        // Fly Path Control Window
        if rd.show_fly_path_menu {
            egui::Window::new("Fly Path")
                .show(self.context(), |ui| {
                egui::Grid::new("fly_path_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                    if let Some(rx) = &channels.rx_fly_path_control {
                        if let Ok(new_control) = rx.try_recv() {
                            *fly_path_control = new_control;
                            channels.rx_fly_path_control = None;
                        }
                    }
                    for i in 0..fly_path_control.keyframes.len() {

                        ui.add(egui::Label::new(format!("Keyframe")));
                        ui.add(egui::Label::new(i.to_string()));
                        ui.end_row();

                        ui.add(egui::Label::new("Timestamp (s)"));
                        ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].timestamp_s));
                        ui.end_row();

                        ui.add(egui::Label::new("Position"));
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.x).desired_width(40.0));
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.y).desired_width(40.0));
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.z).desired_width(40.0));
                        });
                        ui.end_row();

                        ui.add(egui::Label::new("Target"));
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.x).desired_width(40.0));
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.y).desired_width(40.0));
                            ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.z).desired_width(40.0));
                        });
                        ui.end_row();
                    }

                    if ui.button("Add").clicked() {
                        let keyframe = FlyPathFrame::from_camera(&camera);
                        fly_path_control.keyframes.push(keyframe);
                    }
                    if ui.button("Remove").clicked() {
                        fly_path_control.keyframes.pop();
                    }
                    ui.end_row();

                    if ui.button("Upload").clicked() {
                        channels.rx_fly_path_control = Some(FlyPathControl::upload());
                    }
                    if ui.button("Download").clicked() {
                        FlyPathControl::download(&fly_path_control.keyframes);
                    }
                    ui.end_row();

                    ui.label("Hide Menu");
                    ui.checkbox(&mut rd.hide_menu_when_start, "");
                    ui.end_row();

                    ui.add(egui::Label::new("Elapsed Time (s)"));
                    ui.add(egui::Label::new(format!("{:.2}", fly_path_control.timer.elapsed() / 1000.0)));
                    ui.end_row();

                    if ui.button("Reset").clicked() {
                        rd.fly_path_error_msg = fly_path_control.reset_path();
                    }

                    if fly_path_control.ready && !fly_path_control.finished {
                        if fly_path_control.timer.is_paused() {
                            if ui.button("Start").clicked() {
                                if rd.hide_menu_when_start {
                                    rd.show_fly_path_menu = false;
                                    rd.show_main_menu = false;
                                }
                                fly_path_control.start_path();

                                // Start benchmark
                                rd.fly_path_benchmark = true;
                                rd.frame_time_ma.clear();
                                rd.sort_time_ma.clear();
                                rd.build_time_ma.clear();
                                rd.sort_trigger_ma.clear();
                                rd.build_trigger_ma.clear();
                            }
                        } else {
                            if ui.button("Pause").clicked() {
                                fly_path_control.pause_path();
                            }
                        }
                    } else if rd.fly_path_error_msg.is_some() {
                        ui.end_row();
                        ui.label(rd.fly_path_error_msg.clone().unwrap());
                    } else if fly_path_control.finished && rd.fly_path_benchmark {
                        // End benchmark
                        rd.fly_path_benchmark = false;
                        let (f_mean, f_std) = rd.frame_time_ma.calc();
                        let (s_mean, s_std) = rd.sort_time_ma.calc();
                        let (b_mean, b_std) = rd.build_time_ma.calc();
                        let (s_t, _) = rd.sort_trigger_ma.calc();
                        let (b_t, _) = rd.build_trigger_ma.calc();
                        let s_t = s_t * 100.0;
                        let b_t = b_t * 100.0;
                        log!("Render & Sort & Update");
                        log!(r"\( {f_mean:.2} \pm {f_std:.2} \) & \( {s_mean:.2} \pm {s_std:.2} \; ({s_t:.2}\%) \) & \( {b_mean:.2} \pm {b_std:.2} \; ({b_t:.2}\%) \)");
                        rd.frame_time_ma.clear();
                        rd.sort_time_ma.clear();
                        rd.build_time_ma.clear();
                        rd.sort_trigger_ma.clear();
                        rd.build_trigger_ma.clear();
                    }
                    ui.end_row();
                });
            });
        }
    }
}

fn render_motion_summary_controls(ui: &mut egui::Ui, motion: &mut MotionRenderData) -> bool {
    let summary = &motion.summary;
    ui.label("Dynamic asset");
    ui.label(format!(
        "v{} · {} · {} tiles × {} LoDs",
        summary.schema_version, summary.backend, summary.tile_count, summary.lod_count
    ));
    ui.end_row();

    ui.label("Motion");
    ui.label(crate::motion_authoring_ui::compact_status(
        &motion.authoring.behavior().name,
        &motion_basis_summary(summary),
        motion.playback.source_duration(),
    ));
    ui.end_row();

    ui.label("Motion editing");
    let open_authoring = ui.button("Open Motion… (B)").clicked();
    ui.end_row();

    if let Some(error) = motion.error.as_deref() {
        ui.label("Motion error");
        ui.colored_label(egui::Color32::RED, error);
        ui.end_row();
    }
    ui.separator();
    ui.end_row();
    open_authoring
}

fn motion_basis_summary(summary: &crate::scene_archive::DynamicArchiveSummary) -> String {
    if summary.schema_version == 3 {
        format!("3 banks · B{} · K{}", summary.basis_count, summary.top_k)
    } else {
        format!("{} bases · top-{}", summary.basis_count, summary.top_k)
    }
}

#[cfg(test)]
mod motion_summary_tests {
    use super::motion_basis_summary;
    use crate::scene_archive::DynamicArchiveSummary;

    #[test]
    fn version_three_summary_names_three_banks_and_compact_dimensions() {
        let summary = DynamicArchiveSummary {
            schema_version: 3,
            tile_count: 4,
            lod_count: 7,
            basis_count: 64,
            top_k: 8,
            total_rows: 12_345,
            backend: "4dgaussians".to_string(),
        };

        assert_eq!(motion_basis_summary(&summary), "3 banks · B64 · K8");
    }
}
