//! Frame statistics, detailed profiling, and LoD visibility/count controls.
use super::GUI;
use crate::structure::RenderData;
use crate::utils::IncrementalMA;
use num_format::{Locale, ToFormattedString};

pub(super) fn summary_rows(ui: &mut egui::Ui, rd: &mut RenderData) {
    let frame_time = rd.frame_time_ma.calc();
    ui.add(egui::Label::new("FPS"));
    ui.label(format!("{:.2}", 1000.0 / frame_time.0));
    ui.end_row();

    ui.add(egui::Label::new("Render Time (ms)"));
    ui.label(format!("{:.2}±{:.2}", frame_time.0, frame_time.1));
    ui.end_row();

    let sort_time = rd.sort_time_ma.calc();
    let (sort_trigger, _) = rd.sort_trigger_ma.calc();
    ui.add(egui::Label::new("Sort Time (ms)"));
    ui.label(format!(
        "{:.2}±{:.2} ({:.2}%)",
        sort_time.0,
        sort_time.1,
        sort_trigger * 100.0
    ));
    ui.end_row();

    let build_time = rd.build_time_ma.calc();
    let (build_trigger, _) = rd.build_trigger_ma.calc();
    ui.add(egui::Label::new("Update Time (ms)"));
    ui.label(format!(
        "{:.2}±{:.2} ({:.2}%)",
        build_time.0,
        build_time.1,
        build_trigger * 100.0
    ));
    ui.end_row();

    ui.label("Detailed profiler");
    ui.checkbox(&mut rd.profiler_enabled, "Enabled");
    ui.end_row();

    ui.label("Timer Avg Window");
    ui.add(egui::Slider::new(&mut rd.time_ma_window, 1..=5000));
    ui.end_row();

    if ui.button("Reset Timer").clicked() {
        rd.frame_time_ma = IncrementalMA::new(rd.time_ma_window);
        rd.sort_time_ma = IncrementalMA::new(rd.time_ma_window);
        rd.build_time_ma = IncrementalMA::new(rd.time_ma_window);
        rd.sort_trigger_ma = IncrementalMA::new(rd.time_ma_window);
        rd.build_trigger_ma = IncrementalMA::new(rd.time_ma_window);
        rd.profiler_reset_requested = true;
    }
    ui.end_row();

    let mut splat_count: usize = 0;
    let mut blending_splat_count: usize = 0;
    if rd.cur_scene_data.is_some() {
        splat_count = rd.cur_scene_data.as_ref().unwrap().splat_count;
        blending_splat_count = rd.cur_scene_data.as_ref().unwrap().blending_splat_count;
    }
    ui.add(egui::Label::new("Splat Count (With Blending)"));
    ui.label(format!(
        "{} ({})",
        splat_count.to_formatted_string(&Locale::en),
        blending_splat_count.to_formatted_string(&Locale::en)
    ));
    ui.end_row();
}

impl GUI {
    pub(super) fn render_performance(&mut self, rd: &mut RenderData) {
        if rd.show_perf_menu {
            egui::Window::new("Performance").show(&self.context().clone(), |ui| {
                egui::Grid::new("performance_grid")
                    .num_columns(2)
                    .spacing([40.0, 4.0])
                    .striped(true)
                    .show(ui, |ui| {
                        let frame_time = rd.frame_time_ma.calc();
                        ui.add(egui::Label::new("FPS"));
                        ui.label(format!("{:.2}", 1000.0 / frame_time.0));
                        ui.end_row();

                        ui.add(egui::Label::new("Render Time (ms)"));
                        ui.label(format!("{:.2}±{:.2}", frame_time.0, frame_time.1));
                        ui.end_row();

                        let sort_time = rd.sort_time_ma.calc();
                        let (sort_trigger, _) = rd.sort_trigger_ma.calc();
                        ui.add(egui::Label::new("Sort Time (ms)"));
                        ui.label(format!(
                            "{:.2}±{:.2} ({:.2}%)",
                            sort_time.0,
                            sort_time.1,
                            sort_trigger * 100.0
                        ));
                        ui.end_row();

                        let build_time = rd.build_time_ma.calc();
                        let (build_trigger, _) = rd.build_trigger_ma.calc();
                        ui.add(egui::Label::new("Update Time (ms)"));
                        ui.label(format!(
                            "{:.2}±{:.2} ({:.2}%)",
                            build_time.0,
                            build_time.1,
                            build_trigger * 100.0
                        ));
                        ui.end_row();

                        render_profiler_rows(ui, rd);

                        if ui.button("Reset Timer").clicked() {
                            rd.frame_time_ma = IncrementalMA::new(rd.time_ma_window);
                            rd.sort_time_ma = IncrementalMA::new(rd.time_ma_window);
                            rd.build_time_ma = IncrementalMA::new(rd.time_ma_window);
                            rd.sort_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                            rd.build_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                            rd.profiler_reset_requested = true;
                        }
                        ui.end_row();

                        let mut splat_count: usize = 0;
                        let mut blending_splat_count: usize = 0;
                        if rd.cur_scene_data.is_some() {
                            splat_count = rd.cur_scene_data.as_ref().unwrap().splat_count;
                            blending_splat_count =
                                rd.cur_scene_data.as_ref().unwrap().blending_splat_count;
                        }
                        ui.add(egui::Label::new("Splat Count (With Blending)"));
                        ui.label(format!(
                            "{} ({})",
                            splat_count.to_formatted_string(&Locale::en),
                            blending_splat_count.to_formatted_string(&Locale::en)
                        ));
                        ui.end_row();
                    });

                ui.collapsing("LOD", |ui| {
                    egui::Grid::new("lod_grid")
                        .num_columns(2)
                        .spacing([40.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            for i in 0..rd.max_lod_count {
                                ui.label(format!("LOD {i}"));
                                ui.end_row();

                                ui.label("Enable");
                                ui.checkbox(&mut rd.render_config.lod_enable[i], "");
                                ui.end_row();

                                ui.label("Splat Count");
                                let splat_count = if rd.cur_scene_data.is_some() {
                                    rd.cur_scene_data.as_ref().unwrap().lod_splat_count[i]
                                } else {
                                    0
                                };
                                ui.label(splat_count.to_formatted_string(&Locale::en));
                                ui.end_row();

                                ui.label("Tile Instance");
                                let instance_count = if rd.cur_scene_data.is_some() {
                                    rd.cur_scene_data.as_ref().unwrap().lod_instance_count[i]
                                } else {
                                    0
                                };
                                ui.label(instance_count.to_formatted_string(&Locale::en));
                                ui.end_row();
                            }
                        });
                });
            });
        }
    }
}

fn render_profiler_rows(ui: &mut egui::Ui, rd: &mut RenderData) {
    let snapshot = &rd.profiler_snapshot;
    ui.label("Detailed profiler");
    ui.checkbox(&mut rd.profiler_enabled, "Enabled");
    ui.end_row();

    ui.label("Timing columns");
    ui.label("mean / p95 (ms)");
    ui.end_row();

    for (label, metric) in [
        ("CPU frame", snapshot.frame_cpu),
        ("CPU motion prep", snapshot.motion_cpu),
        ("CPU render encode/upload", snapshot.render_cpu),
        ("Worker sort", snapshot.worker_sort_cpu),
        ("Worker build", snapshot.worker_build_cpu),
        ("GPU motion compute", snapshot.motion_gpu),
        ("GPU authored motion", snapshot.authored_gpu),
        ("GPU Gaussian render", snapshot.gaussian_gpu),
        ("GPU water preparation", snapshot.water_hit_gpu),
        ("GPU water shading", snapshot.water_shade_gpu),
    ] {
        ui.label(label);
        if metric.samples == 0 {
            ui.label("—");
        } else {
            ui.label(format!("{:.3} / {:.3}", metric.mean, metric.p95));
        }
        ui.end_row();
    }

    ui.label("GPU timestamps");
    if let Some(error) = &snapshot.gpu_error {
        ui.colored_label(egui::Color32::RED, error);
    } else if snapshot.gpu_timestamps_supported {
        ui.label("available; asynchronous readback");
    } else {
        ui.label("unsupported by this adapter");
    }
    ui.end_row();

    let counters = &snapshot.counters;
    ui.label("Archive / motion rows");
    ui.label(format!(
        "{} / {}",
        counters.archive_rows.to_formatted_string(&Locale::en),
        counters.motion_rows.to_formatted_string(&Locale::en)
    ));
    ui.end_row();

    ui.label("Selected / rendered splats");
    ui.label(format!(
        "{} / {}",
        counters.selected_splats.to_formatted_string(&Locale::en),
        counters.rendered_splats.to_formatted_string(&Locale::en),
    ));
    ui.end_row();

    ui.label("Blending splats");
    ui.label(format!(
        "{}",
        counters.blending_splats.to_formatted_string(&Locale::en)
    ));
    ui.end_row();

    ui.label("Active tile/LoD members");
    ui.label(format!(
        "{} / {}",
        counters.active_members.to_formatted_string(&Locale::en),
        counters.total_members.to_formatted_string(&Locale::en)
    ));
    ui.end_row();

    ui.label("Draw calls");
    ui.label(counters.draw_calls.to_formatted_string(&Locale::en));
    ui.end_row();

    ui.label("Uploaded per frame");
    ui.label(format!(
        "{:.3} MiB",
        counters.uploaded_bytes as f64 / (1024.0 * 1024.0)
    ));
    ui.end_row();

    ui.label("Authored registry / tagged occurrences");
    ui.label(format!(
        "{} / {}",
        counters
            .authored_registry_occurrences
            .to_formatted_string(&Locale::en),
        counters
            .authored_tagged_occurrences
            .to_formatted_string(&Locale::en)
    ));
    ui.end_row();

    ui.label("Authored draws / worker tagging");
    ui.label(format!(
        "{} / {:.3} ms",
        counters
            .authored_affected_draws
            .to_formatted_string(&Locale::en),
        counters.authored_worker_tag_ms,
    ));
    ui.end_row();

    ui.label("Membership request / registry revision");
    ui.label(format!(
        "{} / {}",
        counters.authored_membership_request_revision, counters.authored_registry_revision,
    ));
    ui.end_row();

    ui.label("Compact registry upload");
    ui.label(format!(
        "{:.3} KiB{}",
        counters.authored_registry_upload_bytes as f64 / 1024.0,
        if counters.authored_registry_installed {
            " (installed)"
        } else {
            ""
        },
    ));
    ui.end_row();

    ui.label("Authored update");
    ui.label(if counters.authored_dispatched {
        "dispatched"
    } else {
        "idle"
    });
    ui.end_row();

    ui.label("Motion update");
    ui.label(if !counters.motion_dispatched {
        "idle"
    } else if counters.motion_blend {
        "blend sample"
    } else {
        "single-source sample"
    });
    ui.end_row();
}
