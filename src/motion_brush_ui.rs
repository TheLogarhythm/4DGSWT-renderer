use egui::{Color32, Context, Id, LayerId, Order, Pos2, Shape, Stroke, Ui, Vec2, epaint::Mesh};

use crate::camera::Camera;
use crate::motion_brush::{
    MotionBrushFalloff, MotionBrushPreview, MotionBrushRuntime, MotionBrushTool,
    MotionFieldQuality, MotionOverlayChannel,
};
use crate::motion_controller_palette::{MAX_LOCAL_CONTROLLERS, MotionRegionId};
use crate::structure::MotionRenderData;
use crate::utils::vec4;

const PREVIEW_RING_SEGMENTS: usize = 64;
const FALLOFF_BANDS: usize = 8;

fn project_authoring_xy(
    camera: &Camera,
    canonical_xy: [f32; 2],
    scene_scale_xy: [f32; 2],
    pixels_per_point: f32,
) -> Option<Pos2> {
    if canonical_xy.into_iter().any(|value| !value.is_finite())
        || scene_scale_xy.into_iter().any(|value| !value.is_finite())
        || !pixels_per_point.is_finite()
        || pixels_per_point <= 0.0
    {
        return None;
    }
    let clip = camera.view_proj()
        * vec4(
            canonical_xy[0] * scene_scale_xy[0],
            canonical_xy[1] * scene_scale_xy[1],
            0.0,
            1.0,
        );
    if !clip.w.is_finite() || clip.w <= 1.0e-5 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if [ndc.x, ndc.y, ndc.z]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let viewport = camera.viewport();
    Some(Pos2::new(
        (ndc.x + 1.0) * 0.5 * viewport.width as f32 / pixels_per_point,
        (1.0 - ndc.y) * 0.5 * viewport.height as f32 / pixels_per_point,
    ))
}

fn project_authoring_ring(
    camera: &Camera,
    center: [f32; 2],
    radius: f32,
    scene_scale_xy: [f32; 2],
    pixels_per_point: f32,
) -> Option<Vec<Pos2>> {
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    (0..=PREVIEW_RING_SEGMENTS)
        .map(|index| {
            let angle = std::f32::consts::TAU * index as f32 / PREVIEW_RING_SEGMENTS as f32;
            project_authoring_xy(
                camera,
                [
                    center[0] + radius * angle.cos(),
                    center[1] + radius * angle.sin(),
                ],
                scene_scale_xy,
                pixels_per_point,
            )
        })
        .collect()
}

fn dashed_screen_ring(center: Pos2, radius: f32, segment_count: usize) -> Vec<[Pos2; 2]> {
    if !radius.is_finite() || radius <= 0.0 || segment_count < 2 {
        return Vec::new();
    }
    (0..segment_count)
        .step_by(2)
        .map(|index| {
            let start_angle = std::f32::consts::TAU * index as f32 / segment_count as f32;
            let end_angle = std::f32::consts::TAU * (index + 1) as f32 / segment_count as f32;
            [
                center + Vec2::angled(start_angle) * radius,
                center + Vec2::angled(end_angle) * radius,
            ]
        })
        .collect()
}

fn project_falloff_mesh(
    camera: &Camera,
    scene_scale_xy: [f32; 2],
    pixels_per_point: f32,
    preview: MotionBrushPreview,
) -> Option<Mesh> {
    let center = project_authoring_xy(camera, preview.center, scene_scale_xy, pixels_per_point)?;
    let radial_bands = if preview.falloff == MotionBrushFalloff::Constant {
        1
    } else {
        FALLOFF_BANDS
    };
    let mut mesh = Mesh::default();
    mesh.colored_vertex(
        center,
        preview_color(preview, preview_fill_alpha(preview, 0.0)),
    );
    for band in 1..=radial_bands {
        let normalized_radius = band as f32 / radial_bands as f32;
        let mut ring = project_authoring_ring(
            camera,
            preview.center,
            preview.radius * normalized_radius,
            scene_scale_xy,
            pixels_per_point,
        )?;
        ring.pop();
        let color = preview_color(preview, preview_fill_alpha(preview, normalized_radius));
        for point in ring {
            mesh.colored_vertex(point, color);
        }
    }

    for segment in 0..PREVIEW_RING_SEGMENTS {
        let current = 1 + segment as u32;
        let next = 1 + ((segment + 1) % PREVIEW_RING_SEGMENTS) as u32;
        mesh.add_triangle(0, current, next);
    }
    for band in 1..radial_bands {
        let inner_start = 1 + (band - 1) * PREVIEW_RING_SEGMENTS;
        let outer_start = 1 + band * PREVIEW_RING_SEGMENTS;
        for segment in 0..PREVIEW_RING_SEGMENTS {
            let next = (segment + 1) % PREVIEW_RING_SEGMENTS;
            let inner_current = (inner_start + segment) as u32;
            let inner_next = (inner_start + next) as u32;
            let outer_current = (outer_start + segment) as u32;
            let outer_next = (outer_start + next) as u32;
            mesh.add_triangle(inner_current, outer_current, outer_next);
            mesh.add_triangle(inner_current, outer_next, inner_next);
        }
    }
    Some(mesh)
}

fn preview_fill_alpha(preview: MotionBrushPreview, normalized_radius: f32) -> u8 {
    let influence = preview.influence_at([
        preview.center[0] + preview.radius * normalized_radius,
        preview.center[1],
    ]);
    (32.0 * influence).round().clamp(0.0, 255.0) as u8
}

pub fn paint_viewport_preview(
    context: &Context,
    camera: &Camera,
    scene_scale_xy: [f32; 2],
    runtime: &MotionBrushRuntime,
) {
    if !runtime.enabled || context.is_pointer_over_area() {
        return;
    }
    let painter = context.layer_painter(LayerId::new(
        Order::Middle,
        Id::new("motion brush viewport preview"),
    ));
    let pixels_per_point = context.pixels_per_point();
    let Some(preview) = runtime.preview() else {
        paint_invalid_cursor(context, &painter);
        return;
    };
    let Some(center) =
        project_authoring_xy(camera, preview.center, scene_scale_xy, pixels_per_point)
    else {
        paint_invalid_cursor(context, &painter);
        return;
    };
    let Some(outer_ring) = project_authoring_ring(
        camera,
        preview.center,
        preview.radius,
        scene_scale_xy,
        pixels_per_point,
    ) else {
        paint_invalid_cursor(context, &painter);
        return;
    };

    paint_active_path(
        &painter,
        camera,
        scene_scale_xy,
        pixels_per_point,
        runtime.preview_path(),
        preview,
    );

    let Some(falloff_mesh) =
        project_falloff_mesh(camera, scene_scale_xy, pixels_per_point, preview)
    else {
        paint_invalid_cursor(context, &painter);
        return;
    };
    painter.add(Shape::mesh(falloff_mesh));

    painter.add(Shape::line(
        outer_ring,
        Stroke::new(2.25, preview_color(preview, 245)),
    ));
    if preview.falloff != MotionBrushFalloff::Constant
        && let Some(half_ring) = project_authoring_ring(
            camera,
            preview.center,
            preview.radius * 0.5,
            scene_scale_xy,
            pixels_per_point,
        )
    {
        painter.add(Shape::line(
            half_ring,
            Stroke::new(1.0, preview_color(preview, 170)),
        ));
    }

    let marker_color = preview_color(preview, 255);
    painter.circle_stroke(center, 4.0, Stroke::new(1.5, marker_color));
    painter.line_segment(
        [center + Vec2::new(-7.0, 0.0), center + Vec2::new(7.0, 0.0)],
        Stroke::new(1.25, marker_color),
    );
    painter.line_segment(
        [center + Vec2::new(0.0, -7.0), center + Vec2::new(0.0, 7.0)],
        Stroke::new(1.25, marker_color),
    );
}

fn preview_color(preview: MotionBrushPreview, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (preview.color[0] * 255.0).round() as u8,
        (preview.color[1] * 255.0).round() as u8,
        (preview.color[2] * 255.0).round() as u8,
        alpha,
    )
}

fn paint_active_path(
    painter: &egui::Painter,
    camera: &Camera,
    scene_scale_xy: [f32; 2],
    pixels_per_point: f32,
    path: &[[f32; 2]],
    preview: MotionBrushPreview,
) {
    let points = path
        .iter()
        .filter_map(|&point| project_authoring_xy(camera, point, scene_scale_xy, pixels_per_point))
        .collect::<Vec<_>>();
    if points.len() >= 2 {
        painter.add(Shape::line(
            points,
            Stroke::new(2.0, preview_color(preview, 180)),
        ));
    }
}

fn paint_invalid_cursor(context: &Context, painter: &egui::Painter) {
    let Some(center) = context.input(|input| input.pointer.hover_pos()) else {
        return;
    };
    let color = Color32::from_rgb(255, 72, 64);
    for segment in dashed_screen_ring(center, 18.0, 32) {
        painter.line_segment(segment, Stroke::new(1.75, color));
    }
    painter.line_segment(
        [center + Vec2::new(-4.0, -4.0), center + Vec2::new(4.0, 4.0)],
        Stroke::new(1.5, color),
    );
    painter.line_segment(
        [center + Vec2::new(-4.0, 4.0), center + Vec2::new(4.0, -4.0)],
        Stroke::new(1.5, color),
    );
}

/// Leaving authoring completes the current stroke, but never clears authored data.
pub(crate) fn stop_painting(motion: &mut MotionRenderData) {
    if let Some(runtime) = motion.spatial.as_mut() {
        if runtime.enabled
            && let Err(error) = runtime.end_stroke()
        {
            motion.spatial_error = Some(error.to_string());
        }
        runtime.set_enabled(false);
    }
}

pub(crate) fn render_style_selector(ui: &mut Ui, motion: &mut MotionRenderData) {
    let selected_id = motion.regions.selected();
    let selected_name = motion.regions.selected_region().name().to_string();
    let choices = motion
        .regions
        .regions()
        .iter()
        .map(|region| (region.id(), region.name().to_string(), region.enabled()))
        .collect::<Vec<_>>();
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("motion_style_selection")
            .selected_text(selected_name)
            .show_ui(ui, |ui| {
                for (id, name, enabled) in choices {
                    let label = if enabled { name } else { format!("{name} · disabled") };
                    if ui.selectable_label(id == selected_id, label).clicked()
                        && let Err(error) = motion.regions.select(id) {
                        motion.spatial_error = Some(error.to_string());
                    }
                }
            });
        if ui.button("Duplicate").on_hover_text("Copy this style's motion and source range with an independent seed. Does not copy painted areas.").clicked() {
            match motion.regions.new_variation() {
                Ok(id) => if let Some(runtime) = motion.spatial.as_mut() {
                    runtime.tool = behavior_tool_for_region(id);
                },
                Err(error) => motion.spatial_error = Some(error.to_string()),
            }
        }
        ui.menu_button("Style options", |ui| {
            if let Some(error) = render_style_name(ui, &mut motion.regions) {
                motion.spatial_error = Some(error);
            }
            let mut enabled = motion.regions.selected_region().enabled();
            if ui.checkbox(&mut enabled, "Enabled").changed() {
                motion.regions.set_selected_enabled(enabled);
            }
            if ui.add_enabled(motion.regions.can_remove_selected(), egui::Button::new("Remove style"))
                .on_hover_text("Its painted areas return to scene default. The first style is protected.")
                .clicked() && let Err(error) = crate::motion_session::remove_selected_region(motion) {
                motion.spatial_error = Some(error);
            }
            if ui.add_enabled(crate::motion_session::can_undo_region_removal(motion), egui::Button::new("Undo removal"))
                .on_hover_text("Undo newer strokes first. Editing styles invalidates this one-level undo.")
                .clicked() && let Err(error) = crate::motion_session::undo_region_removal(motion) {
                motion.spatial_error = Some(error);
            }
        });
    });
    if !motion.regions.selected_region().enabled() {
        ui.colored_label(
            Color32::YELLOW,
            "Style disabled — painted areas use scene default.",
        );
    }
}

#[derive(Clone)]
struct StyleNameDraft {
    original: String,
    edited: String,
}

fn render_style_name(
    ui: &mut Ui,
    regions: &mut crate::motion_controller_palette::MotionRegionDocument,
) -> Option<String> {
    let selected = regions.selected_region();
    let id = ui.make_persistent_id(("motion_style_name_draft", selected.id().get()));
    let mut draft = ui
        .data_mut(|data| data.get_temp::<StyleNameDraft>(id))
        .filter(|draft| draft.original == selected.name())
        .unwrap_or_else(|| StyleNameDraft {
            original: selected.name().to_string(),
            edited: selected.name().to_string(),
        });
    ui.label("Style name");
    let mut commit = false;
    ui.horizontal(|ui| {
        let response = ui.add(
            egui::TextEdit::singleline(&mut draft.edited)
                .id(id.with("input"))
                .desired_width(180.0),
        );
        let rename_clicked = ui.button("Rename").clicked();
        commit = response.lost_focus()
            || (response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)))
            || rename_clicked;
    });
    let mut error = None;
    if commit && draft.edited != draft.original {
        match regions.rename_selected(draft.edited.clone()) {
            Ok(()) => {
                draft.original = regions.selected_region().name().to_string();
                draft.edited = draft.original.clone();
            }
            Err(reason) => error = Some(reason.to_string()),
        }
    }
    ui.data_mut(|data| data.insert_temp(id, draft));
    error
}

pub(crate) fn paint_controls(ui: &mut Ui, motion: &mut MotionRenderData) {
    ui.label("Paint this style");
    let selected_region = motion.regions.selected();
    let selected_name = motion.regions.selected_region().name().to_string();
    let Some(runtime) = motion.spatial.as_mut() else {
        ui.weak("Painting is available when the tile map is ready.");
        return;
    };
    runtime.selected_region = selected_region;
    runtime.tool = match runtime.tool {
        MotionBrushTool::ApplyBehavior { .. } => behavior_tool_for_region(selected_region),
        MotionBrushTool::Synchronize { .. } => MotionBrushTool::Synchronize {
            region_id: selected_region,
        },
        tool => tool,
    };
    ui.horizontal(|ui| {
        if ui.selectable_label(!runtime.enabled, "Navigate").clicked() {
            if let Err(error) = runtime.end_stroke() {
                motion.spatial_error = Some(error.to_string());
            }
            runtime.set_enabled(false);
        }
        if ui
            .selectable_label(
                runtime.enabled && matches!(runtime.tool, MotionBrushTool::ApplyBehavior { .. }),
                "Paint",
            )
            .on_hover_text("Paint the selected style at the cursor.")
            .clicked()
        {
            activate_tool(runtime, behavior_tool_for_region(selected_region));
        }
        if ui
            .selectable_label(
                runtime.enabled && matches!(runtime.tool, MotionBrushTool::Erase),
                "Erase",
            )
            .on_hover_text("Restore scene default in the brushed area.")
            .clicked()
        {
            activate_tool(runtime, MotionBrushTool::Erase);
        }
    });
    let mut diameter = runtime.radius * 2.0;
    if ui
        .add(
            egui::Slider::new(&mut diameter, 0.2..=16.0)
                .text("Brush size")
                .suffix(" units"),
        )
        .on_hover_text(
            "World-space diameter of the brush footprint. Does not change motion magnitude.",
        )
        .changed()
    {
        runtime.radius = diameter * 0.5;
    }
    if runtime.enabled {
        ui.weak("Left-drag to paint · Navigate to move the camera");
        if !matches!(
            runtime.tool,
            MotionBrushTool::ApplyBehavior { .. } | MotionBrushTool::Erase
        ) {
            ui.label(format!("Active tool: {}", tool_label(runtime.tool)));
        }
    }
    ui.collapsing("More paint tools & settings", |ui| {
        render_tool(ui, runtime, selected_region, &selected_name);
        ui.separator();
        let opacity = ui.add(egui::Slider::new(&mut runtime.opacity, 0.0..=1.0).text("Opacity"));
        if opacity.changed() || opacity.dragged() { runtime.follow_overlay(MotionOverlayChannel::Influence); }
        ui.add(egui::Slider::new(&mut runtime.spacing, 0.01..=1.0).text("Spacing"));
        egui::ComboBox::from_id_salt("motion_brush_falloff")
            .selected_text(falloff_label(runtime.falloff))
            .show_ui(ui, |ui| {
                for falloff in [MotionBrushFalloff::Smooth, MotionBrushFalloff::Linear, MotionBrushFalloff::Constant] {
                    ui.selectable_value(&mut runtime.falloff, falloff, falloff_label(falloff));
                }
            });
        ui.weak("Targets affect the next stroke. Paint is anchored in world XY and applies through the full height.");
        let mut quality = runtime.layout().quality();
        egui::ComboBox::from_id_salt("motion_brush_quality")
            .selected_text(quality_label(quality))
            .show_ui(ui, |ui| {
                for level in [MotionFieldQuality::Low, MotionFieldQuality::Default, MotionFieldQuality::High] {
                    ui.selectable_value(&mut quality, level, quality_label(level));
                }
            });
        if quality != runtime.layout().quality() && let Err(error) = runtime.set_quality(quality) {
            motion.spatial_error = Some(error.to_string());
        }
        egui::ComboBox::from_id_salt("motion_brush_overlay")
            .selected_text(overlay_label(runtime.overlay))
            .show_ui(ui, |ui| {
                for overlay in [MotionOverlayChannel::Off, MotionOverlayChannel::Influence, MotionOverlayChannel::Strength, MotionOverlayChannel::Variation, MotionOverlayChannel::Coherence, MotionOverlayChannel::Assignment] {
                    ui.selectable_value(&mut runtime.overlay, overlay, overlay_label(overlay));
                }
            });
        ui.checkbox(&mut runtime.auto_overlay, "Follow active paint tool")
            .on_hover_text("Automatically switches the overlay when a paint tool or target changes.");
    });
    let mut show_areas = runtime.overlay != MotionOverlayChannel::Off;
    if ui.checkbox(&mut show_areas, "Show painted areas").changed() {
        runtime.overlay = if show_areas {
            MotionOverlayChannel::Assignment
        } else {
            MotionOverlayChannel::Off
        };
        runtime.auto_overlay = show_areas;
    }
    ui.horizontal(|ui| {
        if ui
            .add_enabled(runtime.can_undo(), egui::Button::new("Undo stroke"))
            .clicked()
            && let Err(error) = runtime.undo()
        {
            motion.spatial_error = Some(error.to_string());
        }
        if ui
            .add_enabled(runtime.can_redo(), egui::Button::new("Redo stroke"))
            .clicked()
            && let Err(error) = runtime.redo()
        {
            motion.spatial_error = Some(error.to_string());
        }
    });
}

fn activate_tool(runtime: &mut MotionBrushRuntime, tool: MotionBrushTool) {
    let previous_overlay = runtime.overlay;
    runtime.set_enabled(true);
    runtime.tool = tool;
    if !runtime.auto_overlay {
        runtime.overlay = previous_overlay;
    } else if let Some(channel) = tool_overlay(tool) {
        runtime.follow_overlay(channel);
    }
}

fn tool_overlay(tool: MotionBrushTool) -> Option<MotionOverlayChannel> {
    match tool {
        MotionBrushTool::ApplyBehavior { .. } => Some(MotionOverlayChannel::Assignment),
        MotionBrushTool::AdjustStrength { .. } => Some(MotionOverlayChannel::Strength),
        MotionBrushTool::AdjustVariation { .. } => Some(MotionOverlayChannel::Variation),
        MotionBrushTool::AdjustCoherence { .. } | MotionBrushTool::Synchronize { .. } => {
            Some(MotionOverlayChannel::Coherence)
        }
        MotionBrushTool::Erase => Some(MotionOverlayChannel::Influence),
        MotionBrushTool::Smooth => None,
    }
}

fn tool_label(tool: MotionBrushTool) -> &'static str {
    match tool {
        MotionBrushTool::ApplyBehavior { .. } => "Motion style",
        MotionBrushTool::AdjustStrength { .. } => "Motion amount",
        MotionBrushTool::AdjustVariation { .. } => "Spatial variety",
        MotionBrushTool::AdjustCoherence { .. } => "Grouping",
        MotionBrushTool::Synchronize { .. } => "Synchronize",
        MotionBrushTool::Smooth => "Smooth",
        MotionBrushTool::Erase => "Erase",
    }
}

pub(crate) fn render_diagnostics(ui: &mut Ui, motion: &MotionRenderData) {
    let Some(runtime) = motion.spatial.as_ref() else {
        return;
    };
    let layout = runtime.layout();
    let hovered = runtime
        .hover()
        .filter(|_| runtime.enabled)
        .and_then(|world| {
            let origin = layout.world_origin();
            let spacing = layout.texel_world_size();
            let size = layout.texture_size();
            let x = (world[0] - origin[0]) / spacing[0];
            let y = (world[1] - origin[1]) / spacing[1];
            if !x.is_finite()
                || !y.is_finite()
                || x < 0.0
                || y < 0.0
                || x >= size[0] as f32
                || y >= size[1] as f32
            {
                return None;
            }
            runtime.cache().texel(x.floor() as u32, y.floor() as u32)
        });
    ui.separator();
    ui.heading("At cursor");
    let values = hovered.map(|texel| {
        let [influence, strength, variation, coherence] = texel.continuous;
        [
            f32::from(influence) / 255.0,
            if strength == 255 {
                2.0
            } else {
                f32::from(strength) / 128.0
            },
            f32::from(variation) / 255.0,
            f32::from(coherence) / 255.0,
        ]
    });
    egui::Grid::new("motion_brush_cursor_values")
        .num_columns(2)
        .show(ui, |ui| {
            for (index, label) in ["Influence", "Strength", "Variation", "Coherence"]
                .iter()
                .enumerate()
            {
                ui.label(*label);
                ui.monospace(
                    values
                        .map(|v| format!("{:.2}", v[index]))
                        .unwrap_or_else(|| "—".into()),
                );
                ui.end_row();
            }
        });
    if let Some(texel) = hovered {
        if texel.assignment == 0 || texel.continuous[0] == 0 {
            ui.weak("Scene default · local field values are inactive.");
        } else {
            ui.weak("Local field values at the cursor.");
        }
    } else {
        ui.weak("Hover over the scene with the brush enabled.");
    }
    ui.separator();
    let [width, height] = layout.texture_size();
    let [texel_x, texel_y] = layout.texel_world_size();
    ui.monospace(format!(
        "{width} × {height} texels · {texel_x:.3} × {texel_y:.3} world"
    ));
    ui.monospace(format!(
        "{:.2} MiB · {:.2}% painted · {} strokes",
        layout.estimated_gpu_bytes() as f64 / (1024.0 * 1024.0),
        runtime.painted_fraction() * 100.0,
        runtime.document().strokes().len(),
    ));
    ui.monospace(format!(
        "Controllers  {} / {}",
        motion.local_controllers.active_local_count(),
        MAX_LOCAL_CONTROLLERS,
    ));
    let enabled_regions = motion
        .regions
        .regions()
        .iter()
        .filter(|r| r.enabled())
        .count();
    if enabled_regions > 16 {
        ui.weak("More than 16 enabled styles reduces spatial variants per style; 64 styles leave one shared timeline each.");
    }
    if let Some(compiled) = motion.compiled_spatial.as_ref() {
        ui.monospace(format!(
            "Field compiled · revision {} · {} active slots",
            compiled.revision(),
            compiled.palette_count(),
        ));
    } else {
        ui.monospace("Field compilation pending first painted stroke");
    }
    if let Some(error) = motion.spatial_error.as_deref() {
        ui.colored_label(Color32::RED, error);
    }
}

fn render_tool(
    ui: &mut Ui,
    runtime: &mut crate::motion_brush::MotionBrushRuntime,
    selected_region: MotionRegionId,
    selected_name: &str,
) {
    let previous_tool = runtime.tool;
    let mut target_interacted = false;
    ui.horizontal_wrapped(|ui| {
        for candidate in [
            behavior_tool_for_region(selected_region),
            MotionBrushTool::AdjustStrength { target: 1.0 },
            MotionBrushTool::AdjustVariation { target: 0.5 },
            MotionBrushTool::AdjustCoherence { target: 0.5 },
            MotionBrushTool::Synchronize {
                region_id: selected_region,
            },
            MotionBrushTool::Smooth,
            MotionBrushTool::Erase,
        ] {
            let current =
                std::mem::discriminant(&candidate) == std::mem::discriminant(&runtime.tool);
            if ui
                .selectable_label(runtime.enabled && current, tool_label(candidate))
                .clicked()
            {
                let tool = if current { runtime.tool } else { candidate };
                activate_tool(runtime, tool);
            }
        }
    });

    match &mut runtime.tool {
        MotionBrushTool::ApplyBehavior { .. } => {
            ui.weak(format!("Paint style · {selected_name}"));
        }
        MotionBrushTool::AdjustStrength { target } => {
            let response = ui.add(egui::Slider::new(target, 0.0..=2.0).text("Target"));
            target_interacted = response.changed() || response.dragged();
        }
        MotionBrushTool::AdjustVariation { target } => {
            let response = ui.add(egui::Slider::new(target, 0.0..=1.0).text("Target"));
            target_interacted = response.changed() || response.dragged();
            ui.weak("Higher values put more patches on alternate timelines. Zero uses the style's shared timeline.");
        }
        MotionBrushTool::AdjustCoherence { target } => {
            let response = ui.add(egui::Slider::new(target, 0.0..=1.0).text("Target"));
            target_interacted = response.changed() || response.dragged();
            ui.weak("Groups neighboring patches. Requires painted Spatial variety above zero; 1 shares one patch throughout the style.");
        }
        MotionBrushTool::Synchronize { .. } => {
            ui.monospace(format!("Match timeline · {selected_name}"));
            ui.weak("Removes spatial variation and preserves strength.");
        }
        MotionBrushTool::Smooth | MotionBrushTool::Erase => {}
    }
    if runtime.tool != previous_tool || target_interacted {
        let channel = tool_overlay(runtime.tool);
        if let Some(channel) = channel {
            runtime.follow_overlay(channel);
        }
    }
}

fn behavior_tool_for_region(region_id: MotionRegionId) -> MotionBrushTool {
    MotionBrushTool::ApplyBehavior { region_id }
}

fn quality_label(quality: MotionFieldQuality) -> &'static str {
    match quality {
        MotionFieldQuality::Low => "Low · 8/tile",
        MotionFieldQuality::Default => "Default · 16/tile",
        MotionFieldQuality::High => "High · 32/tile",
    }
}

fn falloff_label(falloff: MotionBrushFalloff) -> &'static str {
    match falloff {
        MotionBrushFalloff::Constant => "Constant",
        MotionBrushFalloff::Linear => "Linear",
        MotionBrushFalloff::Smooth => "Smooth",
    }
}

fn overlay_label(overlay: MotionOverlayChannel) -> &'static str {
    match overlay {
        MotionOverlayChannel::Off => "Overlay off",
        MotionOverlayChannel::Influence => "Overlay · Influence",
        MotionOverlayChannel::Strength => "Overlay · Strength",
        MotionOverlayChannel::Variation => "Overlay · Variation",
        MotionOverlayChannel::Coherence => "Overlay · Coherence",
        MotionOverlayChannel::Assignment => "Overlay · Style",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::utils::{degrees, vec3};
    use winit::dpi::PhysicalSize;

    fn tool_frame(
        context: &Context,
        runtime: &mut MotionBrushRuntime,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        context.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(800.0, 600.0),
                )),
                events,
                ..Default::default()
            },
            |context| {
                egui::CentralPanel::default().show(context, |ui| {
                    render_tool(ui, runtime, MotionRegionId::local(1).unwrap(), "Gentle");
                });
            },
        )
    }

    #[test]
    fn choosing_advanced_tool_from_navigate_activates_paint_and_respects_hidden_overlay() {
        for label in [
            "Motion amount",
            "Spatial variety",
            "Grouping",
            "Synchronize",
            "Smooth",
        ] {
            let context = Context::default();
            let mut runtime = MotionBrushRuntime::new(
                crate::motion_brush::MotionFieldLayout::new(
                    [3, 3],
                    4.0,
                    [0, 0],
                    MotionFieldQuality::Low,
                )
                .unwrap(),
            )
            .unwrap();
            runtime.auto_overlay = false;
            runtime.overlay = MotionOverlayChannel::Off;
            let output = tool_frame(&context, &mut runtime, Vec::new());
            let position = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    Shape::Text(text) if text.galley.text() == label => {
                        Some(text.pos + text.galley.size() * 0.5)
                    }
                    _ => None,
                })
                .unwrap();
            for pressed in [true, false] {
                tool_frame(
                    &context,
                    &mut runtime,
                    vec![
                        egui::Event::PointerMoved(position),
                        egui::Event::PointerButton {
                            pos: position,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::default(),
                        },
                    ],
                );
            }
            assert!(runtime.enabled, "{label} did not activate painting");
            assert_eq!(tool_label(runtime.tool), label);
            assert_eq!(runtime.overlay, MotionOverlayChannel::Off);
            runtime.begin_stroke([0.0, 0.0]).unwrap();
            runtime.end_stroke().unwrap();
            assert_eq!(runtime.document().strokes().len(), 1);
        }
    }

    #[test]
    fn style_rename_keeps_a_space_while_typing_the_next_word() {
        let context = Context::default();
        let mut regions = crate::motion_controller_palette::MotionRegionDocument::artist_defaults();
        let mut frame = |events| {
            context.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(800.0, 600.0),
                    )),
                    events,
                    ..Default::default()
                },
                |context| {
                    egui::CentralPanel::default().show(context, |ui| {
                        assert!(render_style_name(ui, &mut regions).is_none());
                    });
                },
            )
        };
        let output = frame(Vec::new());
        let position = output
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                Shape::Text(text) if text.galley.text() == "Gentle" => Some(
                    text.pos + Vec2::new(text.galley.size().x + 4.0, text.galley.size().y * 0.5),
                ),
                _ => None,
            })
            .unwrap();
        for pressed in [true, false] {
            frame(vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                },
            ]);
        }
        frame(vec![egui::Event::Text(" ".into())]);
        frame(vec![egui::Event::Text("trees".into())]);
        frame(vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::default(),
        }]);
        assert_eq!(regions.selected_region().name(), "Gentle trees");
    }

    #[test]
    fn quality_and_overlay_labels_are_artist_facing() {
        assert_eq!(
            quality_label(MotionFieldQuality::Default),
            "Default · 16/tile"
        );
        assert_eq!(
            overlay_label(MotionOverlayChannel::Assignment),
            "Overlay · Style"
        );
    }

    #[test]
    fn selected_motion_region_is_the_behavior_brush_identity() {
        let selected = MotionRegionId::local(37).unwrap();

        assert_eq!(
            behavior_tool_for_region(selected),
            MotionBrushTool::ApplyBehavior {
                region_id: selected
            }
        );
    }

    #[test]
    fn world_space_brush_ring_projects_to_a_closed_viewport_footprint() {
        let camera = Camera::new_perspective(
            PhysicalSize::new(1000, 1000),
            vec3(0.0, -10.0, 10.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            degrees(60.0),
            0.1,
            100.0,
        );

        let center = project_authoring_xy(&camera, [0.0, 0.0], [1.0, 1.0], 1.0).unwrap();
        let ring = project_authoring_ring(&camera, [0.0, 0.0], 2.0, [1.0, 1.0], 1.0).unwrap();

        assert!((center.x - 500.0).abs() < 0.01);
        assert!((center.y - 500.0).abs() < 0.01);
        assert_eq!(ring.len(), 65);
        let first = ring.first().unwrap();
        let last = ring.last().unwrap();
        assert!((first.x - last.x).abs() < 0.01);
        assert!((first.y - last.y).abs() < 0.01);
        assert!(
            ring.iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
        );
        assert!(ring.iter().any(|point| (point.x - center.x).abs() > 50.0));
        assert!(ring.iter().any(|point| (point.y - center.y).abs() > 50.0));
    }

    #[test]
    fn invalid_brush_cursor_uses_an_even_dashed_screen_ring() {
        let center = Pos2::new(40.0, 60.0);
        let segments = dashed_screen_ring(center, 18.0, 32);

        assert_eq!(segments.len(), 16);
        for [start, end] in segments {
            for point in [start, end] {
                let distance = ((point.x - center.x).powi(2) + (point.y - center.y).powi(2)).sqrt();
                assert!((distance - 18.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn constant_fill_is_one_uniform_non_overlapping_mesh() {
        let camera = Camera::new_perspective(
            PhysicalSize::new(1000, 1000),
            vec3(0.0, -10.0, 10.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            degrees(60.0),
            0.1,
            100.0,
        );
        let preview = MotionBrushPreview {
            center: [0.0, 0.0],
            radius: 2.0,
            opacity: 0.75,
            falloff: MotionBrushFalloff::Constant,
            tool: MotionBrushTool::Smooth,
            color: MotionBrushTool::Smooth.preview_color(),
        };

        let mesh = project_falloff_mesh(&camera, [1.0, 1.0], 1.0, preview).unwrap();

        assert_eq!(mesh.indices.len(), PREVIEW_RING_SEGMENTS * 3);
        assert_eq!(mesh.vertices.len(), PREVIEW_RING_SEGMENTS + 1);
        let alpha = mesh.vertices[0].color.a();
        assert!(alpha > 0);
        assert!(mesh.vertices.iter().all(|vertex| vertex.color.a() == alpha));
    }

    #[test]
    fn smooth_fill_uses_one_mesh_with_falloff_reaching_zero_at_the_edge() {
        let camera = Camera::new_perspective(
            PhysicalSize::new(1000, 1000),
            vec3(0.0, -10.0, 10.0),
            vec3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
            degrees(60.0),
            0.1,
            100.0,
        );
        let preview = MotionBrushPreview {
            center: [0.0, 0.0],
            radius: 2.0,
            opacity: 1.0,
            falloff: MotionBrushFalloff::Smooth,
            tool: MotionBrushTool::AdjustVariation { target: 0.5 },
            color: MotionBrushTool::AdjustVariation { target: 0.5 }.preview_color(),
        };

        let mesh = project_falloff_mesh(&camera, [1.0, 1.0], 1.0, preview).unwrap();

        assert_eq!(mesh.vertices[0].color.a(), 32);
        assert_eq!(mesh.vertices.last().unwrap().color.a(), 0);
        assert_eq!(
            mesh.indices.len(),
            PREVIEW_RING_SEGMENTS * 3 + (FALLOFF_BANDS - 1) * PREVIEW_RING_SEGMENTS * 6
        );
    }
}
