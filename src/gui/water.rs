//! Water surface, reflection, wave, and underwater controls.
use crate::structure::{RenderData, SurfaceType};

pub(super) fn controls(ui: &mut egui::Ui, rd: &mut RenderData, surface_type: SurfaceType) {
    ui.collapsing("Water", |ui| {
        let supported = surface_type != SurfaceType::Sphere;
        ui.add_enabled_ui(supported, |ui| {
            let water = &mut rd.render_config.water;
            ui.checkbox(&mut water.enabled, "Enable water");
            ui.add_enabled_ui(water.enabled, |ui| {
                egui::Grid::new("water_controls").num_columns(2).show(ui, |ui| {
                    ui.label("Water level");
                    ui.add(egui::DragValue::new(&mut water.height).speed(0.02).range(-10000.0..=10000.0))
                        .on_hover_text("Height in renderer world coordinates");
                    ui.end_row();
                    ui.label("Water color");
                    ui.color_edit_button_rgb(&mut water.color);
                    ui.end_row();
                    ui.label("Wave amplitude");
                    ui.add(egui::Slider::new(&mut water.amplitude, 0.0..=water.wavelength * 0.035).max_decimals(3))
                        .on_hover_text("Maximum displacement from the mean water level. Zero gives a flat surface.");
                    ui.end_row();
                    ui.label("Wavelength");
                    if ui.add(egui::DragValue::new(&mut water.wavelength).speed(0.05).range(0.1..=1000.0)).changed() {
                        water.amplitude = water.amplitude.min(water.wavelength * 0.035);
                    }
                    ui.end_row();
                    ui.label("Wave variation");
                    ui.checkbox(&mut water.varied_waves, "Irregular directions and spacing");
                    ui.end_row();
                    ui.label("Reflection strength");
                    ui.add(egui::Slider::new(&mut water.reflection_strength, 0.0..=1.0))
                        .on_hover_text("Reflects the displayed skybox. Zero disables reflection.");
                    ui.end_row();
                    ui.label("Roughness");
                    ui.add(egui::Slider::new(&mut water.roughness, 0.0..=1.0));
                    ui.end_row();
                    ui.label("Ripple strength");
                    ui.add(egui::Slider::new(&mut water.ripple_strength, 0.0..=1.0))
                        .on_hover_text("Fine surface detail. Zero disables ripples.");
                    ui.end_row();
                    ui.label("Ripple scale");
                    ui.add(egui::DragValue::new(&mut water.ripple_scale).speed(0.01).range(0.02..=100.0));
                    ui.end_row();
                    ui.label("Wave speed");
                    ui.add(egui::Slider::new(&mut water.speed, 0.0..=5.0));
                    ui.end_row();
                    ui.label("Wave playback");
                    ui.checkbox(&mut water.playing, "Playing");
                    ui.end_row();
                    ui.label("Underwater effect");
                    ui.checkbox(&mut water.underwater.enabled, "Enable when submerged")
                        .on_hover_text("Fades distant coral and proxy terrain when the camera is below this water surface.");
                    ui.end_row();
                    if water.underwater.enabled {
                        ui.label("Underwater color");
                        ui.color_edit_button_rgb(&mut water.underwater.color);
                        ui.end_row();
                        ui.label("Clear near distance");
                        ui.add(egui::DragValue::new(&mut water.underwater.clear_distance).speed(0.1).range(0.0..=100.0))
                            .on_hover_text("Preserve nearby coral colors before distance fog begins, in world units.");
                        ui.end_row();
                        ui.label("Fog half-distance");
                        ui.add(egui::Slider::new(&mut water.underwater.visibility, 0.5..=100.0).logarithmic(true).text("world units"))
                            .on_hover_text("Distance beyond the clear zone where scene contrast falls to 50%. Larger values give clearer water.");
                        ui.end_row();
                        ui.label("Underwater sunlight");
                        ui.add(egui::Slider::new(&mut water.underwater.sunlight, 0.0..=2.0));
                        ui.end_row();
                        ui.label("Sun azimuth");
                        ui.add(egui::Slider::new(&mut water.underwater.sun_azimuth, -180.0..=180.0).suffix("°"));
                        ui.end_row();
                        ui.label("Sun elevation");
                        ui.add(egui::Slider::new(&mut water.underwater.sun_elevation, 10.0..=90.0).suffix("°"));
                        ui.end_row();
                        ui.label("Light shafts");
                        ui.checkbox(&mut water.underwater.light_shafts, "Enabled");
                        ui.end_row();
                        if water.underwater.light_shafts {
                            ui.label("Shaft strength");
                            ui.add(egui::Slider::new(&mut water.underwater.shaft_strength, 0.0..=2.0));
                            ui.end_row();
                            ui.label("Shaft width");
                            ui.add(egui::Slider::new(&mut water.underwater.shaft_scale, 0.5..=20.0).logarithmic(true).text("world units"));
                            ui.end_row();
                        }
                        ui.label("Caustics");
                        ui.checkbox(&mut water.underwater.caustics, "Enabled")
                            .on_hover_text("Animated light patterns on submerged coral and ground.");
                        ui.end_row();
                        if water.underwater.caustics {
                            ui.label("Caustic strength");
                            ui.add(egui::Slider::new(&mut water.underwater.caustic_strength, 0.0..=2.0));
                            ui.end_row();
                            ui.label("Caustic scale");
                            ui.add(egui::Slider::new(&mut water.underwater.caustic_scale, 0.25..=20.0).logarithmic(true).text("world units"));
                            ui.end_row();
                            ui.label("Caustic speed");
                            ui.add(egui::Slider::new(&mut water.underwater.caustic_speed, 0.0..=5.0).suffix("x"))
                                .on_hover_text("Independent of wave speed. 0 freezes the pattern; 1 matches the original speed. Changing speed keeps the current pattern position.");
                            ui.end_row();
                        }
                    }
                });
            });
        });
        if rd.render_config.water.enabled && !rd.use_skybox { ui.label("Load and enable a skybox for environment reflection."); }
        if !supported { ui.label("Water is available with None or HeightMap surface mapping."); }
    });
    ui.end_row();
}
