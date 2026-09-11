use egui::{Color32, Context, Id, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::motion::{LoopPolicy, TimelineSample};
use crate::motion_behavior::{MotionBehavior, MotionBehaviorPreset};
use crate::motion_graph::SOURCE_SEGMENT_COUNT;
use crate::structure::MotionRenderData;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum MotionScope {
    #[default]
    Scene,
    Local,
}

pub fn show(context: &Context, open: &mut bool, motion: &mut MotionRenderData) {
    if !*open {
        crate::motion_brush_ui::stop_painting(motion);
        return;
    }
    sync_authoring(motion);
    let scope_id = Id::new("motion_authoring_scope");
    let mut scope: MotionScope =
        context.data_mut(|data| data.get_temp(scope_id).unwrap_or_default());
    let response = egui::Window::new("Motion")
        .open(open)
        .default_width(400.0)
        .min_width(360.0)
        .vscroll(true)
        .show(context, |ui| {
            render_header(ui, motion);
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut scope, MotionScope::Scene, "Scene default");
                ui.selectable_value(&mut scope, MotionScope::Local, "Local styles");
            });
            if scope == MotionScope::Scene {
                crate::motion_brush_ui::stop_painting(motion);
                ui.weak("Base motion for areas without a local style.");
            } else {
                crate::motion_brush_ui::render_style_selector(ui, motion);
            }
            // Separate widget IDs retain each scope's expanded sections and text-edit state.
            ui.push_id(
                if scope == MotionScope::Scene {
                    "scene"
                } else {
                    "local"
                },
                |ui| {
                    render_behavior(ui, motion, scope);
                    if scope == MotionScope::Local {
                        ui.separator();
                        crate::motion_brush_ui::paint_controls(ui, motion);
                    }
                    ui.separator();
                    ui.collapsing("Source & looping", |ui| {
                        render_source_controls(ui, motion, scope)
                    });
                },
            );
            ui.collapsing("Diagnostics", |ui| {
                ui.label("Scene playback / source graph");
                render_advanced(ui, motion);
                crate::motion_brush_ui::render_diagnostics(ui, motion);
            });
            if let Some(error) = motion.error.as_deref() {
                ui.colored_label(Color32::RED, error);
            }
            if let Some(error) = motion.spatial_error.as_deref() {
                ui.colored_label(Color32::RED, error);
            }
        });
    if !*open || response.is_none_or(|response| response.inner.is_none()) {
        crate::motion_brush_ui::stop_painting(motion);
    }
    context.data_mut(|data| data.insert_temp(scope_id, scope));
}

fn render_header(ui: &mut Ui, motion: &mut MotionRenderData) {
    ui.horizontal(|ui| {
        if motion.playback.playing {
            if ui.button("Pause").clicked() {
                motion.playback.pause();
            }
        } else if ui.button("Play").clicked() {
            motion.playback.play();
        }
        if ui.button("Restart").clicked() {
            motion.restart_playback();
        }
        let mut enabled = motion.playback.motion_enabled;
        if ui.checkbox(&mut enabled, "Motion enabled").changed() {
            motion.playback.set_motion_enabled(enabled);
        }
    });

    let mut source_u = current_source_u(motion);
    ui.horizontal(|ui| {
        ui.label("Scene time");
        if ui
            .add(egui::Slider::new(&mut source_u, 0.0..=1.0).show_value(false))
            .changed()
            && let Err(error) = motion.playback.scrub_normalized(source_u)
        {
            motion.error = Some(error.to_string());
        }
        ui.monospace(format!(
            "{:.3} / {:.3} s",
            source_u * motion.playback.source_duration(),
            motion.playback.source_duration()
        ));
    });

    crate::motion_session::controls(ui, motion);
}

fn scoped_behavior(motion: &MotionRenderData, scope: MotionScope) -> MotionBehavior {
    match scope {
        MotionScope::Scene => motion.authoring.behavior().clone(),
        MotionScope::Local => motion.regions.selected_region().behavior().clone(),
    }
}

fn update_scoped_behavior(
    motion: &mut MotionRenderData,
    scope: MotionScope,
    mut draft: MotionBehavior,
) {
    let result = match scope {
        MotionScope::Scene => motion
            .authoring
            .update_behavior(|behavior| *behavior = draft)
            .map_err(|error| error.to_string()),
        MotionScope::Local => {
            draft.preset = MotionBehaviorPreset::Custom;
            draft.name = motion.regions.selected_region().name().to_string();
            let range = motion.regions.selected_region().source_range().clone();
            motion
                .regions
                .update_selected(draft, range)
                .map_err(|error| error.to_string())
        }
    };
    if let Err(error) = result {
        motion.error = Some(error);
    } else if scope == MotionScope::Scene {
        sync_authoring(motion);
    }
}

fn apply_scoped_preset(
    motion: &mut MotionRenderData,
    scope: MotionScope,
    preset: MotionBehaviorPreset,
) {
    match scope {
        MotionScope::Scene => {
            motion.authoring.apply_preset(preset);
            sync_authoring(motion);
        }
        MotionScope::Local => {
            let selected = motion.regions.selected_region();
            let mut draft = MotionBehavior::from_preset(preset);
            draft.seed = selected.behavior().seed;
            draft.name = selected.name().to_string();
            let range = selected.source_range().clone();
            if let Err(error) = motion.regions.update_selected(draft, range) {
                motion.error = Some(error.to_string());
            }
        }
    }
}

fn render_behavior(ui: &mut Ui, motion: &mut MotionRenderData, scope: MotionScope) {
    ui.horizontal_wrapped(|ui| {
        for preset in MotionBehaviorPreset::ARTIST_PRESETS {
            if ui
                .selectable_label(
                    scoped_behavior(motion, scope).preset == preset,
                    preset.label(),
                )
                .clicked()
            {
                apply_scoped_preset(motion, scope, preset);
            }
        }
    });
    let mut draft = scoped_behavior(motion, scope);
    if edit_behavior(ui, &mut draft, motion.graph.is_some()) {
        update_scoped_behavior(motion, scope, draft);
    }
}

fn set_playback_variation(behavior: &mut MotionBehavior, variation: f32) {
    behavior.variation = variation;
    behavior.stochastic_enabled = variation > 0.0;
}

/// Shared by scene defaults and local styles; this editor never owns a controller.
fn edit_behavior(ui: &mut Ui, draft: &mut MotionBehavior, graph_available: bool) -> bool {
    let mut changed = ui.add(egui::Slider::new(&mut draft.gains.master, 0.0..=2.0)
        .text("Motion amount")).on_hover_text("Scales translation, rotation and scale motion together. Does not resize the painted area.").changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut draft.playback_speed, 0.05..=4.0)
                .text("Speed")
                .suffix("×"),
        )
        .changed();
    let mut variation = if draft.stochastic_enabled {
        draft.variation
    } else {
        0.0
    };
    ui.add_enabled_ui(graph_available, |ui| {
        if ui.add(egui::Slider::new(&mut variation, 0.0..=1.0).text("Playback variation"))
            .on_hover_text("How often playback takes an eligible alternate path. Spatial variety is painted separately.")
            .changed() {
            set_playback_variation(draft, variation);
            changed = true;
        }
    }).response.on_disabled_hover_text("No analyzed motion graph is available; ordinary playback remains available.");
    if !graph_available {
        ui.weak("Playback variation unavailable for this asset.");
    }
    ui.collapsing("Motion details", |ui| {
        for (axis, gain) in ["X translation", "Y translation", "Z translation"]
            .into_iter()
            .zip(draft.gains.translation.iter_mut())
        {
            changed |= ui
                .add(egui::Slider::new(gain, 0.0..=2.0).text(axis))
                .changed();
        }
        changed |= ui
            .add(egui::Slider::new(&mut draft.gains.rotation, 0.0..=2.0).text("Rotation"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut draft.gains.scale, 0.0..=2.0).text("Scale"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut draft.transition_seconds, 0.01..=1.0)
                    .text("Transition")
                    .suffix(" s"),
            )
            .changed();
        ui.add_enabled_ui(graph_available, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut draft.gust_frequency, 0.0..=1.0)
                        .text("Change frequency"),
                )
                .on_hover_text(
                    "Controls the minimum spacing and frequency of eligible stochastic jumps.",
                )
                .changed();
            changed |= ui
                .checkbox(&mut draft.stochastic_enabled, "Stochastic branches")
                .changed();
            ui.horizontal(|ui| {
                ui.label("Seed");
                changed |= ui.add(egui::DragValue::new(&mut draft.seed)).changed();
            });
        });
    });
    changed
}

fn render_source_controls(ui: &mut Ui, motion: &mut MotionRenderData, scope: MotionScope) {
    let mut draft = scoped_behavior(motion, scope);
    let previous_policy = draft.endpoint_policy;
    ui.horizontal(|ui| {
        ui.label("Loop");
        egui::ComboBox::from_id_salt("motion_loop_policy")
            .selected_text(loop_policy_label(draft.endpoint_policy))
            .show_ui(ui, |ui| {
                for policy in [
                    LoopPolicy::Hold,
                    LoopPolicy::DirectWrap,
                    LoopPolicy::AppendedTransition,
                    LoopPolicy::BlendToStart,
                ] {
                    ui.selectable_value(
                        &mut draft.endpoint_policy,
                        policy,
                        loop_policy_label(policy),
                    );
                }
            });
    });
    let mut name_changed = false;
    if scope == MotionScope::Scene {
        ui.horizontal(|ui| {
            ui.label("Scene motion name");
            name_changed = ui.text_edit_singleline(&mut draft.name).changed();
        });
    }
    if draft.endpoint_policy != previous_policy || name_changed {
        update_scoped_behavior(motion, scope, draft);
    }
    if scope == MotionScope::Scene {
        render_timeline_tab(ui, motion);
    } else {
        if ui.button("Copy scene motion and source range").on_hover_text("Copies settings only; leaves existing paint and the style's independent seed intact.").clicked() {
            let selected = motion.regions.selected_region();
            let mut behavior = motion.authoring.behavior().clone();
            behavior.name = selected.name().to_string();
            behavior.seed = selected.behavior().seed;
            if let Err(error) = motion.regions.update_selected(behavior, motion.authoring.active_range().clone()) {
                motion.error = Some(error.to_string());
            }
        }
        let mut range = motion.regions.selected_region().source_range().clone();
        let mut changed = false;
        ui.add_enabled_ui(motion.graph.is_some(), |ui| {
            ui.horizontal(|ui| {
                ui.label("Source segments");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut range.start_segment)
                            .range(0..=SOURCE_SEGMENT_COUNT - 1),
                    )
                    .changed();
                ui.label("to");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut range.end_segment)
                            .range(1..=SOURCE_SEGMENT_COUNT),
                    )
                    .changed();
            });
        });
        range.end_segment = range.end_segment.max(range.start_segment + 1);
        ui.weak(format!(
            "{:.3} s · local style source range",
            range_seconds(
                range.start_segment,
                range.end_segment,
                motion.playback.source_duration()
            )
        ));
        if changed {
            let behavior = motion.regions.selected_region().behavior().clone();
            if let Err(error) = motion.regions.update_selected(behavior, range) {
                motion.error = Some(error.to_string());
            }
        }
    }
}

fn render_timeline_tab(ui: &mut Ui, motion: &mut MotionRenderData) {
    ui.label("Source motion");
    render_source_strip(ui, motion);
    ui.small("Ranges are contiguous source segments. Stochastic jumps remain limited to close, prevalidated graph edges inside the active range.");

    if motion.graph.is_none() {
        ui.colored_label(
            Color32::YELLOW,
            motion
                .graph_error
                .as_deref()
                .unwrap_or("Source-range authoring requires a stochastic motion graph."),
        );
        ui.add_enabled(
            false,
            egui::Button::new("Source-range controls unavailable"),
        )
        .on_disabled_hover_text("Fallback playback uses the archive's complete source timeline.");
        return;
    }

    let ranges = motion.authoring.ranges().to_vec();
    let mut active = motion.authoring.active_range_index();
    egui::ComboBox::from_id_salt("motion_source_range")
        .selected_text(&ranges[active].name)
        .show_ui(ui, |ui| {
            for (index, range) in ranges.iter().enumerate() {
                let label = if range.enabled {
                    range.name.clone()
                } else {
                    format!("{} (disabled; select to enable)", range.name)
                };
                ui.selectable_value(&mut active, index, label);
            }
        });
    if active != motion.authoring.active_range_index() {
        match motion.authoring.select_range(active) {
            Ok(()) => sync_authoring(motion),
            Err(error) => motion.error = Some(error.to_string()),
        }
    }

    let active = motion.authoring.active_range_index();
    let mut edited = motion.authoring.active_range().clone();
    let mut changed = false;
    let has_enabled_fallback = motion
        .authoring
        .ranges()
        .iter()
        .enumerate()
        .any(|(index, range)| index != active && range.enabled);
    ui.horizontal(|ui| {
        ui.label("Name");
        changed |= ui.text_edit_singleline(&mut edited.name).changed();
        ui.add_enabled_ui(has_enabled_fallback, |ui| {
            changed |= ui.checkbox(&mut edited.enabled, "Enabled").changed();
        });
    });
    ui.horizontal(|ui| {
        ui.label("Segments");
        changed |= ui
            .add(
                egui::DragValue::new(&mut edited.start_segment).range(0..=SOURCE_SEGMENT_COUNT - 1),
            )
            .changed();
        ui.label("to");
        changed |= ui
            .add(egui::DragValue::new(&mut edited.end_segment).range(1..=SOURCE_SEGMENT_COUNT))
            .changed();
    });
    if edited.start_segment >= edited.end_segment {
        edited.end_segment = (edited.start_segment + 1).min(SOURCE_SEGMENT_COUNT);
    }
    ui.label(format!(
        "{:.3} s",
        range_seconds(
            edited.start_segment,
            edited.end_segment,
            motion.playback.source_duration()
        )
    ));
    if changed {
        match motion.authoring.update_range(active, edited) {
            Ok(()) => sync_authoring(motion),
            Err(error) => motion.error = Some(error.to_string()),
        }
    }

    ui.horizontal(|ui| {
        if ui.button("Add range at playhead").clicked() {
            let start = ((current_source_u(motion) * SOURCE_SEGMENT_COUNT as f32).floor() as usize)
                .min(SOURCE_SEGMENT_COUNT - 1);
            let end = (start + 12).min(SOURCE_SEGMENT_COUNT);
            let name = format!("Range {}", motion.authoring.ranges().len() + 1);
            match motion.authoring.add_range(name, start, end) {
                Ok(index) => {
                    let _ = motion.authoring.select_range(index);
                    sync_authoring(motion);
                }
                Err(error) => motion.error = Some(error.to_string()),
            }
        }
        if ui
            .add_enabled(has_enabled_fallback, egui::Button::new("Delete active"))
            .clicked()
        {
            match motion.authoring.remove_range(active) {
                Ok(()) => sync_authoring(motion),
                Err(error) => motion.error = Some(error.to_string()),
            }
        }
    });
}

fn render_advanced(ui: &mut Ui, motion: &MotionRenderData) {
    if let Some(graph) = motion.graph.as_deref() {
        let summary = graph.summary();
        ui.label(format!(
            "{} segments · {} retained jumps · up to {} per segment",
            summary.segment_count, summary.jump_count, summary.max_jumps_per_segment
        ));
        ui.label(format!(
            "{} segments continue without a stochastic candidate",
            summary.segment_count - summary.segments_with_jumps
        ));
        if let Some(analysis) = graph.analysis_summary() {
            ui.label(format!(
                "Analyzed {} LoD0 rows in {:.1} ms",
                analysis.analyzed_rows, analysis.analysis_milliseconds
            ));
            ui.label(format!(
                "Closeness limits: p95 ≤ {:.1}, ceiling ≤ {:.1}",
                analysis.p95_limit, analysis.ceiling_limit
            ));
            ui.label(format!(
                "{} close pairs before temporal filtering · minimum jump: {} source intervals",
                analysis.close_candidates, analysis.min_jump_intervals
            ));
            for (label, distribution) in [
                ("Qualified", &analysis.qualified),
                ("Retained", &analysis.retained),
            ] {
                ui.label(format!(
                    "{label}: {} forward / {} backward · gap min/mean/max: {}/{:.1}/{}",
                    distribution.forward,
                    distribution.backward,
                    distribution.min_intervals,
                    distribution.mean_intervals(),
                    distribution.max_intervals
                ));
            }
            ui.weak("Gaps are measured from segment exit, not entry. These are full-asset counts; active source ranges, dwell and endpoint policies can reduce playable jumps.");
        }
        ui.separator();
        ui.label(format!(
            "Current segment: {}",
            motion.graph_playback.current_segment()
        ));
        ui.label(match motion.graph_playback.last_selected_edge() {
            Some((source, target)) => {
                format!(
                    "Last stochastic edge: {}",
                    format_transition_edge(source, target)
                )
            }
            None => "Last stochastic edge: none".to_string(),
        });
        let trace = motion
            .graph_playback
            .trace()
            .recent_edges
            .iter()
            .map(|&(source, target)| format_transition_edge(source, target))
            .collect::<Vec<_>>()
            .join(", ");
        ui.label(if trace.is_empty() {
            "Recent branch trace: none".to_string()
        } else {
            format!("Recent branch trace: {trace}")
        });
    } else if let Some(error) = motion.graph_error.as_deref() {
        ui.colored_label(Color32::YELLOW, error);
    } else {
        ui.label("This asset has no stochastic motion graph.");
    }
    ui.separator();
    ui.weak("Session files save motion settings and paint; loading restarts playback. Local styles have independent timelines; the trace above is scene playback.");
}

fn render_source_strip(ui: &mut Ui, motion: &MotionRenderData) {
    let width = ui.available_width().max(120.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 54.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 3.0, Color32::from_gray(24));

    let active = motion.authoring.active_range();
    let segment_width = rect.width() / SOURCE_SEGMENT_COUNT as f32;
    if let Some(graph) = motion.graph.as_deref() {
        for (segment, feature) in graph.segment_features().iter().enumerate() {
            let x0 = rect.left() + segment as f32 * segment_width;
            let height = (8.0 + feature.magnitude.sqrt().min(1.0) * 30.0).min(rect.height());
            let color = Color32::from_rgb(
                ((feature.horizontal_direction[0] * 0.5 + 0.5) * 220.0) as u8,
                110,
                ((feature.horizontal_direction[1] * 0.5 + 0.5) * 220.0) as u8,
            );
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(x0, rect.bottom() - height),
                    Pos2::new(x0 + segment_width.max(1.0), rect.bottom()),
                ),
                0.0,
                color,
            );
        }
    }
    let range_rect = Rect::from_min_max(
        Pos2::new(
            rect.left() + active.start_segment as f32 * segment_width,
            rect.top(),
        ),
        Pos2::new(
            rect.left() + active.end_segment as f32 * segment_width,
            rect.bottom(),
        ),
    );
    painter.rect_stroke(
        range_rect,
        1.0,
        Stroke::new(2.0, Color32::WHITE),
        egui::StrokeKind::Inside,
    );
    let playhead_x = rect.left() + current_source_u(motion) * rect.width();
    painter.line_segment(
        [
            Pos2::new(playhead_x, rect.top()),
            Pos2::new(playhead_x, rect.bottom()),
        ],
        Stroke::new(2.0, Color32::YELLOW),
    );
}

fn sync_authoring(motion: &mut MotionRenderData) {
    let changed = motion.authoring.is_dirty();
    match motion.apply_authoring() {
        Ok(()) if changed => motion.error = None,
        Ok(()) => {}
        Err(error) => motion.error = Some(error.to_string()),
    }
}

fn current_source_u(motion: &MotionRenderData) -> f32 {
    match motion.playback.sample() {
        Ok(TimelineSample::Source { u }) => u,
        Ok(TimelineSample::Blend { to_u, .. }) => to_u,
        Err(_) => 0.0,
    }
}

fn loop_policy_label(policy: LoopPolicy) -> &'static str {
    match policy {
        LoopPolicy::Hold => "Hold",
        LoopPolicy::DirectWrap => "Direct wrap",
        LoopPolicy::AppendedTransition => "Append transition",
        LoopPolicy::BlendToStart => "Blend to start",
    }
}

fn range_seconds(start_segment: usize, end_segment: usize, source_duration: f32) -> f32 {
    source_duration * end_segment.saturating_sub(start_segment) as f32 / SOURCE_SEGMENT_COUNT as f32
}

fn format_transition_edge(source: usize, target: usize) -> String {
    format!("{source} -> {target}")
}

pub(crate) fn compact_status(
    behavior_name: &str,
    basis_summary: &str,
    source_duration: f32,
) -> String {
    format!("{behavior_name} · {basis_summary} · {source_duration:.3}s")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> MotionRenderData {
        use crate::motion_brush::{MotionBrushRuntime, MotionFieldLayout, MotionFieldQuality};
        let mut motion = MotionRenderData::new(
            crate::scene_archive::DynamicArchiveSummary {
                schema_version: 3,
                tile_count: 16,
                lod_count: 6,
                basis_count: 64,
                top_k: 8,
                total_rows: 100,
                backend: "fixture".into(),
            },
            2.0,
            None,
            None,
        )
        .unwrap();
        motion.spatial = Some(
            MotionBrushRuntime::new(
                MotionFieldLayout::new([3, 3], 4.0, [0, 0], MotionFieldQuality::Low).unwrap(),
            )
            .unwrap(),
        );
        motion
    }

    fn visible_text(
        context: &Context,
        motion: &mut MotionRenderData,
        open: &mut bool,
    ) -> Vec<String> {
        fn collect(shape: &egui::epaint::Shape, texts: &mut Vec<String>) {
            match shape {
                egui::epaint::Shape::Text(text) => texts.push(text.galley.text().to_string()),
                egui::epaint::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, texts);
                    }
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for _ in 0..2 {
            let output = context.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0))),
                    ..Default::default()
                },
                |context| show(context, open, motion),
            );
            texts.clear();
            for shape in output.shapes {
                collect(&shape.shape, &mut texts);
            }
        }
        texts
    }

    #[test]
    fn unified_panel_exposes_both_scopes_and_keeps_details_collapsed() {
        let texts = visible_text(&Context::default(), &mut fixture(), &mut true);
        for label in [
            "Scene default",
            "Local styles",
            "Motion amount",
            "Speed",
            "Playback variation",
        ] {
            assert!(
                texts.iter().any(|text| text == label),
                "missing {label}: {texts:?}"
            );
        }
        for hidden in [
            "Wind direction",
            "Master",
            "Rotation",
            "Opacity",
            "Paint this style",
        ] {
            assert!(
                !texts.iter().any(|text| text == hidden),
                "unexpected {hidden}"
            );
        }
        assert_eq!(
            texts
                .iter()
                .filter(|text| text.as_str() == "Save session…")
                .count(),
            1
        );
    }

    #[test]
    fn closing_motion_panel_exits_paint_mode_without_erasing_paint() {
        let mut motion = fixture();
        let runtime = motion.spatial.as_mut().unwrap();
        runtime.set_enabled(true);
        runtime.begin_stroke([0.0, 0.0]).unwrap();
        runtime.end_stroke().unwrap();
        visible_text(&Context::default(), &mut motion, &mut false);
        let runtime = motion.spatial.as_ref().unwrap();
        assert!(!runtime.enabled);
        assert_eq!(runtime.document().strokes().len(), 1);
    }

    #[test]
    fn legacy_direction_fields_do_not_apply_an_invisible_scene_bias() {
        let mut motion = fixture();
        motion
            .authoring
            .update_behavior(|behavior| {
                behavior.horizontal_direction = [-1.0, 0.0];
                behavior.direction_influence = 0.7;
            })
            .unwrap();
        motion.apply_authoring().unwrap();
        assert_eq!(motion.graph_playback.config().direction_influence, 0.0);
        assert_eq!(motion.authoring.behavior().direction_influence, 0.7);
    }

    #[test]
    fn shared_editor_updates_only_the_selected_scope() {
        let mut motion = fixture();
        let scene = motion.authoring.behavior().clone();
        let original_id = motion.regions.selected();
        let original = motion.regions.selected_region().clone();
        motion.regions.new_variation().unwrap();
        let range = motion.regions.selected_region().source_range().clone();
        let mut draft = scoped_behavior(&motion, MotionScope::Local);
        draft.gains.master = 1.75;
        update_scoped_behavior(&mut motion, MotionScope::Local, draft);
        assert_eq!(motion.authoring.behavior(), &scene);
        assert_eq!(
            motion.regions.selected_region().behavior().gains.master,
            1.75
        );
        assert_eq!(motion.regions.selected_region().source_range(), &range);
        apply_scoped_preset(&mut motion, MotionScope::Scene, MotionBehaviorPreset::Calm);
        assert_eq!(
            motion.regions.selected_region().behavior().gains.master,
            1.75
        );
        motion.regions.select(original_id).unwrap();
        assert_eq!(
            motion.regions.selected_region().behavior(),
            original.behavior()
        );
    }

    #[test]
    fn local_panel_shows_painting_with_one_session_toolbar_and_no_expanded_details() {
        let context = Context::default();
        context.data_mut(|data| {
            data.insert_temp(Id::new("motion_authoring_scope"), MotionScope::Local)
        });
        let texts = visible_text(&context, &mut fixture(), &mut true);
        for label in [
            "Duplicate",
            "Paint this style",
            "Navigate",
            "Paint",
            "Erase",
            "Brush size",
        ] {
            assert!(
                texts.iter().any(|text| text == label),
                "missing {label}: {texts:?}"
            );
        }
        assert!(
            !texts
                .iter()
                .any(|text| text == "Opacity" || text == "Rotation")
        );
        assert_eq!(
            texts
                .iter()
                .filter(|text| text.as_str() == "Save session…")
                .count(),
            1
        );
    }

    #[test]
    fn scene_scope_finishes_pending_paint_and_does_not_change_styles() {
        let mut motion = fixture();
        let regions = motion.regions.regions().to_vec();
        let runtime = motion.spatial.as_mut().unwrap();
        runtime.set_enabled(true);
        runtime.begin_stroke([0.0, 0.0]).unwrap();
        visible_text(&Context::default(), &mut motion, &mut true);
        assert!(!motion.spatial.as_ref().unwrap().enabled);
        assert_eq!(
            motion.spatial.as_ref().unwrap().document().strokes().len(),
            1
        );
        assert_eq!(motion.regions.regions(), regions);
    }

    #[test]
    fn raising_variation_enables_subtle_branching_and_zero_disables_it() {
        let mut behavior = MotionBehavior::from_preset(MotionBehaviorPreset::Calm);
        set_playback_variation(&mut behavior, 0.6);
        assert!(behavior.stochastic_enabled);
        assert!(behavior.branch_probability() > 0.0);
        set_playback_variation(&mut behavior, 0.0);
        assert!(!behavior.stochastic_enabled);
        assert_eq!(behavior.branch_probability(), 0.0);
    }

    #[test]
    fn preset_display_rename_preserves_legacy_session_enum_values() {
        for serialized in [
            "\"GentleSway\"",
            "\"SteadyWind\"",
            "\"GustyWind\"",
            "\"Calm\"",
        ] {
            let preset: MotionBehaviorPreset = serde_json::from_str(serialized).unwrap();
            assert_eq!(serde_json::to_string(&preset).unwrap(), serialized);
            let behavior = MotionBehavior::from_preset(preset);
            assert_eq!(behavior.direction_influence, 0.0);
            behavior.validate().unwrap();
        }
    }

    #[test]
    fn range_duration_uses_the_fixed_source_segmentation() {
        assert!((range_seconds(10, 20, 7.4) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn compact_status_keeps_behavior_basis_and_duration_visible() {
        assert_eq!(
            compact_status("Gentle Sway", "3 banks · B64 · K8", 2.5),
            "Gentle Sway · 3 banks · B64 · K8 · 2.500s"
        );
    }

    #[test]
    fn transition_edge_uses_an_ascii_arrow_supported_by_the_ui_font() {
        assert_eq!(format_transition_edge(2, 17), "2 -> 17");
    }
}
