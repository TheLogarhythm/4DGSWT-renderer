//! Archive selection and scene configuration, including the worker handoff.
use super::GUI;
use crate::log;
use crate::proxy::upload_proxy_texture;
use crate::skybox::upload_skybox;
use crate::structure::{
    GUIStatus, HeightMapType, MainChannels, RenderData, SelectiveMergeType, SurfaceType,
    TileSortType,
};
use crate::wangtile::upload_height_map;

impl GUI {
    pub(super) fn render_archive_picker(&mut self) {
        egui::Window::new("GSWT archive")
            .collapsible(false)
            .show(&self.context().clone(), |ui| {
                if self.archive_load_in_progress {
                    ui.label("Loading and validating archive…");
                } else {
                    ui.label("No GSWT archive is loaded.");
                }
                if let Some(error) = &self.archive_error_msg {
                    ui.colored_label(egui::Color32::RED, error);
                }
                if ui
                    .add_enabled(
                        !self.archive_load_in_progress,
                        egui::Button::new("Choose archive…"),
                    )
                    .clicked()
                {
                    self.archive_load_requested = true;
                }
            });
    }
}

impl GUI {
    pub(super) fn render_configuration(
        &mut self,
        channels: &mut MainChannels,
        rd: &mut RenderData,
    ) {
        self.config_lod_count_error_msg = None;
        if self.config_lod_count_error_msg.is_none() && self.config_confirmed {
            self.config_error_msg = None;
            self.config_user_data.config_id = self.config_next_id;
            self.config_user_data_string
                .to_raw(&mut self.config_user_data, &mut self.config_error_msg);

            if let Some(rx) = &channels.rx_height_tex {
                if let Ok(height_tex) = rx.try_recv() {
                    self.config_user_data.height_tex = Some(height_tex);
                    channels.rx_height_tex = None;
                }
            }

            if let Some(rx) = &channels.rx_skybox_tex {
                if let Ok(skybox_tex) = rx.try_recv() {
                    rd.skybox_rawtex = Some(skybox_tex);
                    rd.skybox_changed = true;
                    channels.rx_skybox_tex = None;
                }
            }

            if let Some(rx) = &channels.rx_proxy_tex {
                if let Ok(proxy_tex) = rx.try_recv() {
                    rd.proxy_rawtex = Some(proxy_tex);
                    channels.rx_proxy_tex = None;
                }
            }

            if self.config_error_msg.is_none() {
                log!("Config {} confirmed.", self.config_next_id);
                self.config_confirmed = false;
                channels
                    .tx_commands
                    .send(crate::worker::WorkerCommand::Configure(Box::new(
                        self.config_user_data.clone(),
                    )))
                    .expect("Error sending user data to worker thread.");
                self.gui_status = GUIStatus::PostConfig;
                self.config_next_id += 1;
                return;
            }
        }

        self.config_confirmed = false;

        egui::Window::new("GSWT")
            .vscroll(true)
            .show(&self.context().clone(), |ui| {
                egui::Grid::new("my_grid")
                    .num_columns(7)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        if self.config_user_data.surface_type != SurfaceType::Sphere {
                            ui.label("Tile map");
                            ui.label("Width (half)");
                            ui.text_edit_singleline(
                                &mut self.config_user_data_string.tile_map_half_wh_s.x,
                            );
                            ui.label("Height (half)");
                            ui.text_edit_singleline(
                                &mut self.config_user_data_string.tile_map_half_wh_s.y,
                            );
                            ui.end_row();
                        }

                        // ui.separator();
                        // ui.end_row();
                        ui.label("Center option");
                        ui.text_edit_singleline(
                            &mut self.config_user_data_string.center_option_s,
                        );
                        ui.end_row();

                        ui.label("Update distance tolerance");
                        ui.text_edit_singleline(
                            &mut self.config_user_data_string.update_dist_s,
                        );
                        ui.end_row();

                        // ui.separator();
                        // ui.end_row();
                        ui.label("Tile width");
                        ui.text_edit_singleline(
                            &mut self.config_user_data_string.tile_width_s,
                        );
                        ui.end_row();

                        ui.label("Tile sort type");
                        ui.selectable_value(
                            &mut self.config_user_data.tile_sort_type,
                            TileSortType::Distance,
                            "Distance",
                        );
                        ui.selectable_value(
                            &mut self.config_user_data.tile_sort_type,
                            TileSortType::Viewport,
                            "Viewport",
                        );
                        ui.selectable_value(
                            &mut self.config_user_data.tile_sort_type,
                            TileSortType::Object,
                            "Object",
                        );
                        ui.selectable_value(
                            &mut self.config_user_data.tile_sort_type,
                            TileSortType::Graph,
                            "Graph",
                        );
                        ui.end_row();

                        ui.separator();
                        ui.end_row();

                        if self.config_user_data.surface_type == SurfaceType::Sphere {
                            ui.label("Tile merging");
                            ui.label("Off for curved surfaces");
                            ui.end_row();
                        } else {
                            ui.label("Selective merge");
                            ui.selectable_value(
                                &mut self.config_user_data.merge_type,
                                SelectiveMergeType::None,
                                "None",
                            );
                            ui.selectable_value(
                                &mut self.config_user_data.merge_type,
                                SelectiveMergeType::Axis,
                                "Axis",
                            );
                            ui.selectable_value(
                                &mut self.config_user_data.merge_type,
                                SelectiveMergeType::Edge,
                                "Edge",
                            );
                            ui.end_row();

                            match self.config_user_data.merge_type {
                                SelectiveMergeType::Axis => {
                                    ui.label("Merge tile distance");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.merge_tile_dist_s.x,
                                    );
                                    ui.label("to");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.merge_tile_dist_s.y,
                                    );
                                    ui.end_row();
                                }
                                SelectiveMergeType::Edge => {
                                    ui.label("Merge dot threshold");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.merge_dot_threshold_s,
                                    );
                                    ui.end_row();

                                    ui.label("Merge top-k");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.merge_topk_s,
                                    );
                                    ui.end_row();
                                }
                                SelectiveMergeType::None => {}
                            }
                            if self.config_user_data.merge_type != SelectiveMergeType::None {
                                ui.label("Use merge cache");
                                ui.checkbox(&mut self.config_user_data.use_cache, "");
                                ui.end_row();

                                if self.config_user_data.use_cache {
                                    ui.label("Cache size");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.cache_size_s,
                                    );
                                    ui.end_row();
                                }
                            }
                        }
                        ui.separator();
                        ui.end_row();

                        ui.label("Dynamic LOD");
                        ui.end_row();
                        ui.label("Max Distance");
                        ui.text_edit_singleline(
                            &mut self.config_user_data_string.lod_max_dist_s,
                        );
                        ui.label("x Tile Width");
                        ui.end_row();

                        ui.add(egui::Label::new("LOD blending"));
                        ui.checkbox(&mut self.config_user_data.lod_blending, "");
                        ui.end_row();

                        if self.config_user_data.lod_blending {
                            ui.label("Blending width ratio");
                            ui.text_edit_singleline(
                                &mut self
                                    .config_user_data_string
                                    .lod_transition_width_ratio_s,
                            );
                            ui.end_row();

                            ui.label("Precise bbox check");
                            ui.checkbox(&mut self.config_user_data.lod_bbox_check, "");
                            ui.end_row();

                            ui.label("Blending distance tolerance");
                            ui.text_edit_singleline(
                                &mut self.config_user_data_string.lod_dist_tolerance_s,
                            );
                            ui.end_row();
                        }

                        ui.separator();
                        ui.end_row();

                        ui.label("Surface mapping");
                        ui.selectable_value(
                            &mut self.config_user_data.surface_type,
                            SurfaceType::None,
                            "None",
                        );
                        ui.selectable_value(
                            &mut self.config_user_data.surface_type,
                            SurfaceType::HeightMap,
                            "HeightMap",
                        );
                        ui.selectable_value(
                            &mut self.config_user_data.surface_type,
                            SurfaceType::Sphere,
                            "Cubed Sphere",
                        );
                        ui.end_row();

                        match self.config_user_data.surface_type {
                            SurfaceType::HeightMap => {
                                ui.label("Height map");
                                ui.label("Width");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.height_map_wh_s.x,
                                );
                                ui.label("Height");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.height_map_wh_s.y,
                                );
                                ui.end_row();

                                ui.label("Type");
                                ui.selectable_value(
                                    &mut self.config_user_data.height_map_type,
                                    HeightMapType::Texture,
                                    "Texture",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.height_map_type,
                                    HeightMapType::Random,
                                    "Random",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.height_map_type,
                                    HeightMapType::SlopeX,
                                    "SlopeX",
                                );
                                // ui.selectable_value(&mut height_map_type, HeightMapType::SlopeY, "SlopeY");
                                // ui.selectable_value(&mut height_map_type, HeightMapType::DualSlope, "Dual Slope");
                                ui.end_row();

                                if self.config_user_data.height_map_type
                                    == HeightMapType::Texture
                                {
                                    ui.label("Height texture");
                                    if ui.button("Upload").clicked() {
                                        channels.rx_height_tex = Some(upload_height_map());
                                    }
                                    ui.end_row();
                                }

                                ui.label("Scale (Hori)");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.height_map_scale_s.x,
                                );
                                ui.label("Scale (Vert)");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.height_map_scale_s.y,
                                );
                                ui.end_row();
                            }
                            SurfaceType::Sphere => {
                                ui.label("Tiles per face edge (N)");
                                ui.text_edit_singleline(&mut self.config_user_data_string.sphere_tiles_per_face_s)
                                    .on_hover_text("Each of the six faces contains N × N tiles. Odd N is supported; range 1–128.");
                                if let Ok(n) = self.config_user_data_string.sphere_tiles_per_face_s.parse::<usize>() {
                                    if let Some(total) = n.checked_mul(n).and_then(|n2| n2.checked_mul(6)) {
                                        ui.label(format!("{total} tiles total"));
                                    }
                                }
                                ui.end_row();
                                ui.label("Sphere radius");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.sphere_radius_s,
                                );
                                ui.end_row();
                            }
                            SurfaceType::None => {}
                        }

                        ui.separator();
                        ui.end_row();

                        ui.label("Skybox Texture");
                        if ui.button("Upload").clicked() {
                            channels.rx_skybox_tex = Some(upload_skybox());
                        }
                        ui.end_row();

                        ui.label("Proxy Texture");
                        if ui.button("Upload").clicked() {
                            channels.rx_proxy_tex = Some(upload_proxy_texture());
                        }
                        ui.end_row();

                        ui.label("Reset Rng");
                        ui.checkbox(&mut self.config_user_data.reset_rng, "");
                        ui.label("Always Sort");
                        ui.checkbox(&mut self.config_user_data.always_sort, "");

                        if ui.button("Confirm").clicked() {
                            self.config_confirmed = true;
                        }
                        ui.end_row();
                    });

                if self.config_lod_count_error_msg.is_some() {
                    ui.label(self.config_lod_count_error_msg.clone().unwrap());
                    ui.end_row();
                }

                if self.config_error_msg.is_some() {
                    ui.label(self.config_error_msg.clone().unwrap());
                    ui.end_row();
                }
            });
    }
}
