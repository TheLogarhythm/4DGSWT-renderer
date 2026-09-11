//! Authoring documents and file dialogs. GPU caches and stochastic runtime state are derived.
use crate::motion_behavior::{MotionAuthoringState, MotionBehavior, MotionSourceRange};
use crate::motion_brush::{
    MotionBrushDocument, MotionBrushFalloff, MotionBrushRuntime, MotionBrushStroke,
    MotionBrushTool, MotionFieldLayout, MotionFieldQuality,
};
use crate::motion_controller_palette::{MotionRegion, MotionRegionDocument, MotionRegionId};
use crate::scene_archive::DynamicArchiveSummary;
use crate::structure::MotionRenderData;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, TryRecvError};

const MAX_SESSION_BYTES: usize = 16 * 1024 * 1024;
const MAX_STROKES: usize = 10_000;
const MAX_PATH_POINTS: usize = 200_000;

pub fn remove_selected_region(motion: &mut MotionRenderData) -> Result<(), String> {
    let previous = motion.regions.clone();
    let id = previous.selected();
    let mut replacement = previous.clone();
    replacement.remove_selected().map_err(|e| e.to_string())?;
    let spatial = motion
        .spatial
        .as_mut()
        .ok_or("Motion brush is unavailable")?;
    let tool = spatial.tool;
    spatial.remove_region(id).map_err(|e| e.to_string())?;
    spatial.selected_region = replacement.selected();
    spatial.tool = MotionBrushTool::ApplyBehavior {
        region_id: replacement.selected(),
    };
    motion.session_io.removal_undo = Some(RegionRemovalUndo {
        previous,
        revision: replacement.revision(),
        tool,
    });
    motion.regions = replacement;
    motion.playback.mark_render_dirty();
    Ok(())
}

pub fn can_undo_region_removal(motion: &MotionRenderData) -> bool {
    motion.session_io.removal_undo.as_ref().is_some_and(|undo| {
        motion.regions.revision() == undo.revision
            && motion
                .spatial
                .as_ref()
                .and_then(|s| s.last_removed_region())
                == Some(undo.previous.selected())
    })
}

pub fn undo_region_removal(motion: &mut MotionRenderData) -> Result<(), String> {
    if !can_undo_region_removal(motion) {
        return Err("Undo newer strokes first; region edits invalidate removal undo.".into());
    }
    motion
        .spatial
        .as_mut()
        .unwrap()
        .undo_region_removal()
        .map_err(|e| e.to_string())?;
    let undo = motion.session_io.removal_undo.take().unwrap();
    motion.regions.restore_after_removal(undo.previous);
    let spatial = motion.spatial.as_mut().unwrap();
    spatial.selected_region = motion.regions.selected();
    spatial.tool = undo.tool;
    motion.playback.mark_render_dirty();
    Ok(())
}

struct RegionRemovalUndo {
    previous: MotionRegionDocument,
    revision: u64,
    tool: MotionBrushTool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotionSession {
    version: u32,
    archive: DynamicArchiveSummary,
    source_duration_seconds: f32,
    tile_width: f32,
    behavior: MotionBehavior,
    ranges: Vec<MotionSourceRange>,
    active_range: usize,
    regions: Vec<MotionRegion>,
    selected_region: MotionRegionId,
    strokes: Vec<MotionBrushStroke>,
    brush: BrushSettings,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrushSettings {
    quality: MotionFieldQuality,
    tool: MotionBrushTool,
    radius: f32,
    opacity: f32,
    spacing: f32,
    falloff: MotionBrushFalloff,
}

impl MotionSession {
    pub fn capture(motion: &MotionRenderData) -> Result<Self, String> {
        let spatial = motion
            .spatial
            .as_ref()
            .ok_or("Configure the tile map before saving a session.")?;
        Ok(Self {
            version: if spatial
                .document()
                .strokes()
                .iter()
                .any(|s| s.removed_region.is_some())
            {
                2
            } else {
                1
            },
            archive: motion.summary.clone(),
            source_duration_seconds: motion.playback.source_duration(),
            tile_width: spatial.layout().tile_width(),
            behavior: motion.authoring.behavior().clone(),
            ranges: motion.authoring.ranges().to_vec(),
            active_range: motion.authoring.active_range_index(),
            regions: motion.regions.regions().to_vec(),
            selected_region: motion.regions.selected(),
            strokes: spatial.document().strokes().to_vec(),
            brush: BrushSettings {
                quality: spatial.layout().quality(),
                tool: spatial.tool,
                radius: spatial.radius,
                opacity: spatial.opacity,
                spacing: spatial.spacing,
                falloff: spatial.falloff,
            },
        })
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_SESSION_BYTES {
            return Err("Session exceeds the 16 MiB limit.".into());
        }
        let session: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("Invalid session JSON: {e}"))?;
        session.validate()?;
        Ok(session)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_SESSION_BYTES {
            return Err("Session exceeds the 16 MiB limit.".into());
        }
        Ok(bytes)
    }

    fn validate(&self) -> Result<(), String> {
        if !matches!(self.version, 1 | 2) {
            return Err(format!(
                "Unsupported motion session version {}.",
                self.version
            ));
        }
        if self.version == 1 && self.strokes.iter().any(|s| s.removed_region.is_some()) {
            return Err("Controller removal history requires session version 2.".into());
        }
        if !self.tile_width.is_finite()
            || self.tile_width <= 0.0
            || !self.source_duration_seconds.is_finite()
            || self.source_duration_seconds <= 0.0
        {
            return Err("Invalid session dimensions/duration.".into());
        }
        MotionAuthoringState::from_parts(
            self.behavior.clone(),
            self.ranges.clone(),
            self.active_range,
        )
        .map_err(|e| e.to_string())?;
        MotionRegionDocument::from_regions(self.regions.clone(), self.selected_region)
            .map_err(|e| e.to_string())?;
        if self.strokes.len() > MAX_STROKES
            || self.strokes.iter().map(|s| s.path.len()).sum::<usize>() > MAX_PATH_POINTS
        {
            return Err("Session exceeds the stroke/path-point limit.".into());
        }
        MotionBrushDocument::from_strokes(self.strokes.clone()).map_err(|e| e.to_string())?;
        let check_region = |id: MotionRegionId| -> Result<(), String> {
            if self.regions.iter().any(|r| r.id() == id) {
                Ok(())
            } else {
                Err(format!("Stroke references missing region {}.", id.get()))
            }
        };
        let check_tool = |tool: MotionBrushTool| -> Result<(), String> {
            if let MotionBrushTool::ApplyBehavior { region_id }
            | MotionBrushTool::Synchronize { region_id } = tool
            {
                check_region(region_id)?;
            }
            Ok(())
        };
        let mut removed_later = std::collections::HashSet::new();
        for s in self.strokes.iter().rev() {
            if let Some(id) = s.removed_region {
                removed_later.insert(id);
            }
            let check_historical = |id| {
                if removed_later.contains(&id) {
                    Ok(())
                } else {
                    check_region(id)
                }
            };
            if let MotionBrushTool::ApplyBehavior { region_id }
            | MotionBrushTool::Synchronize { region_id } = s.tool
            {
                check_historical(region_id)?;
            }
            if let Some(id) = s.fallback_region {
                check_historical(id)?;
            }
            if s.radius > self.tile_width * 8.0 || s.radius < 0.1 {
                return Err("Stroke radius is outside supported brush limits.".into());
            }
        }
        check_tool(self.brush.tool)?;
        MotionBrushStroke::new(
            1,
            self.brush.tool,
            vec![[0.0, 0.0]],
            self.brush.radius,
            self.brush.opacity,
            self.brush.spacing,
            self.brush.falloff,
        )
        .map_err(|e| e.to_string())?;
        if !(0.1..=(self.tile_width * 8.0).max(1.0)).contains(&self.brush.radius) {
            return Err("Invalid brush radius.".into());
        }
        Ok(())
    }

    /// Prepare completely before replacing live state; request IDs remain monotonic.
    pub fn prepare(&self, current: &MotionRenderData) -> Result<MotionRenderData, String> {
        self.validate()?;
        let old = current
            .spatial
            .as_ref()
            .ok_or("Configure the tile map before loading a session.")?;
        if self.archive != current.summary
            || self.source_duration_seconds != current.playback.source_duration()
            || self.tile_width != old.layout().tile_width()
        {
            return Err("Session archive structure, source duration or tile width does not match. Load the original GSWT archive first.".into());
        }
        let mut replacement = MotionRenderData::new(
            current.summary.clone(),
            current.playback.source_duration(),
            current.graph.clone(),
            current.graph_error.clone(),
        )
        .map_err(|e| e.to_string())?;
        replacement.authoring = MotionAuthoringState::from_parts(
            self.behavior.clone(),
            self.ranges.clone(),
            self.active_range,
        )
        .map_err(|e| e.to_string())?;
        replacement.regions =
            MotionRegionDocument::from_regions(self.regions.clone(), self.selected_region)
                .map_err(|e| e.to_string())?;
        let layout = old.layout();
        let layout = MotionFieldLayout::new(
            layout.map_tiles(),
            layout.tile_width(),
            layout.center_tile(),
            self.brush.quality,
        )
        .map_err(|e| e.to_string())?;
        let mut spatial = MotionBrushRuntime::new(layout).map_err(|e| e.to_string())?;
        spatial
            .replace_document(
                MotionBrushDocument::from_strokes(self.strokes.clone())
                    .map_err(|e| e.to_string())?,
                self.brush.quality,
            )
            .map_err(|e| e.to_string())?;
        spatial.selected_region = self.selected_region;
        spatial.tool = self.brush.tool;
        spatial.radius = self.brush.radius.clamp(0.1, 8.0);
        spatial.opacity = self.brush.opacity;
        spatial.spacing = self.brush.spacing;
        spatial.falloff = self.brush.falloff;
        spatial.enabled = old.enabled;
        spatial.overlay = old.overlay;
        spatial.auto_overlay = old.auto_overlay;
        replacement.spatial = Some(spatial);
        replacement.apply_authoring().map_err(|e| e.to_string())?;
        replacement
            .compile_spatial_field()
            .map_err(|e| e.to_string())?;
        replacement.restart_playback();
        replacement.playback.playing = current.playback.playing;
        replacement
            .playback
            .set_motion_enabled(current.playback.motion_enabled);
        replacement.playback.channel_selection = current.playback.channel_selection;
        replacement.continue_membership_sequence(current);
        Ok(replacement)
    }
}

enum SessionResult {
    Loaded(Vec<u8>),
    Saved,
    Cancelled,
}

#[derive(Default)]
pub struct SessionIo {
    pending: Option<Receiver<Result<SessionResult, String>>>,
    message: Option<String>,
    removal_undo: Option<RegionRemovalUndo>,
}

#[cfg(not(target_arch = "wasm32"))]
fn launch_io(future: impl std::future::Future<Output = ()> + Send + 'static) {
    std::thread::spawn(move || pollster::block_on(future));
}

// WASM file-dialog futures are not Send.
#[cfg(target_arch = "wasm32")]
fn launch_web(future: impl std::future::Future<Output = ()> + 'static) {
    wasm_bindgen_futures::spawn_local(future);
}

pub fn poll(motion: &mut MotionRenderData) {
    let Some(receiver) = motion.session_io.pending.as_ref() else {
        return;
    };
    let result = match receiver.try_recv() {
        Ok(result) => result,
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => Err("Session file operation ended unexpectedly.".into()),
    };
    motion.session_io.pending = None;
    motion.session_io.message = Some(match result {
        Ok(SessionResult::Saved) => "Session saved.".into(),
        Ok(SessionResult::Cancelled) => "File operation cancelled.".into(),
        Err(error) => error,
        Ok(SessionResult::Loaded(bytes)) => {
            match MotionSession::parse(&bytes).and_then(|s| s.prepare(motion)) {
                Ok(replacement) => {
                    *motion = replacement;
                    "Session loaded; deterministic playback restarted.".into()
                }
                Err(error) => format!("Session not loaded: {error}"),
            }
        }
    });
}

pub fn controls(ui: &mut egui::Ui, motion: &mut MotionRenderData) {
    let busy = motion.session_io.pending.is_some();
    ui.horizontal(|ui| {
        let ready = motion.spatial.is_some() && !busy;
        if ui
            .add_enabled(ready, egui::Button::new("Save session…"))
            .clicked()
        {
            let data = motion
                .spatial
                .as_mut()
                .map(|s| s.end_stroke())
                .transpose()
                .map_err(|e| e.to_string())
                .and_then(|_| MotionSession::capture(motion))
                .and_then(|s| s.to_bytes());
            match data {
                Err(error) => motion.session_io.message = Some(error),
                Ok(bytes) => {
                    let (tx, rx) = mpsc::channel();
                    motion.session_io.pending = Some(rx);
                    let task = rfd::AsyncFileDialog::new()
                        .add_filter("Motion session", &["json"])
                        .set_file_name("motion-session.json")
                        .save_file();
                    let future = async move {
                        let result = match task.await {
                            Some(file) => file
                                .write(&bytes)
                                .await
                                .map(|_| SessionResult::Saved)
                                .map_err(|e| e.to_string()),
                            None => Ok(SessionResult::Cancelled),
                        };
                        let _ = tx.send(result);
                    };
                    #[cfg(target_arch = "wasm32")]
                    launch_web(future);
                    #[cfg(not(target_arch = "wasm32"))]
                    launch_io(future);
                }
            }
        }
        if ui
            .add_enabled(ready, egui::Button::new("Load session…"))
            .clicked()
        {
            let (tx, rx) = mpsc::channel();
            motion.session_io.pending = Some(rx);
            let task = rfd::AsyncFileDialog::new()
                .add_filter("Motion session", &["json"])
                .pick_file();
            let future = async move {
                let result = match task.await {
                    Some(file) => Ok(SessionResult::Loaded(file.read().await)),
                    None => Ok(SessionResult::Cancelled),
                };
                let _ = tx.send(result);
            };
            #[cfg(target_arch = "wasm32")]
            launch_web(future);
            #[cfg(not(target_arch = "wasm32"))]
            launch_io(future);
        }
    });
    if busy {
        ui.weak("Waiting for session file…");
    }
    if let Some(message) = &motion.session_io.message {
        ui.label(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> MotionRenderData {
        let summary = DynamicArchiveSummary {
            schema_version: 2,
            tile_count: 16,
            lod_count: 6,
            basis_count: 64,
            top_k: 8,
            total_rows: 100,
            backend: "fixture".into(),
        };
        let mut motion = MotionRenderData::new(summary, 2.0, None, None).unwrap();
        motion.spatial = Some(
            MotionBrushRuntime::new(
                MotionFieldLayout::new([3, 3], 4.0, [0, 0], MotionFieldQuality::Low).unwrap(),
            )
            .unwrap(),
        );
        motion
    }

    #[test]
    fn remove_controller_preserves_other_regions_and_roundtrips() {
        let mut motion = fixture();
        let id = motion.regions.new_variation().unwrap();
        paint(
            &mut motion,
            MotionBrushTool::ApplyBehavior { region_id: id },
            [0.0, 0.0],
        );
        let before = motion.regions.regions().len();
        remove_selected_region(&mut motion).unwrap();
        assert_eq!(motion.regions.regions().len(), before - 1);
        assert!(
            !motion
                .spatial
                .as_ref()
                .unwrap()
                .cache()
                .assignments()
                .contains(&id.get())
        );
        let session = MotionSession::capture(&motion).unwrap();
        let restored = MotionSession::parse(&session.to_bytes().unwrap())
            .unwrap()
            .prepare(&motion)
            .unwrap();
        assert_eq!(restored.regions.regions().len(), before - 1);
        assert!(
            !restored
                .spatial
                .as_ref()
                .unwrap()
                .cache()
                .assignments()
                .contains(&id.get())
        );
    }

    #[test]
    fn controller_removal_only_clears_current_ownership_and_undo_restores_it() {
        let mut motion = fixture();
        let id = motion.regions.new_variation().unwrap();
        paint(
            &mut motion,
            MotionBrushTool::ApplyBehavior { region_id: id },
            [0.0, 0.0],
        );
        paint(
            &mut motion,
            MotionBrushTool::ApplyBehavior {
                region_id: MotionRegionId::local(2).unwrap(),
            },
            [0.75, 0.0],
        );
        // A scalar stroke with another fallback must not claim the already-painted region.
        paint(
            &mut motion,
            MotionBrushTool::AdjustStrength { target: 0.4 },
            [0.0, 0.0],
        );
        let before = motion
            .spatial
            .as_ref()
            .unwrap()
            .cache()
            .assignments()
            .to_vec();
        let continuous = motion
            .spatial
            .as_ref()
            .unwrap()
            .cache()
            .continuous_bytes()
            .to_vec();
        remove_selected_region(&mut motion).unwrap();
        let spatial = motion.spatial.as_ref().unwrap();
        for (index, owner) in before.iter().enumerate() {
            assert_eq!(
                spatial.cache().assignments()[index],
                if *owner == id.get() { 0 } else { *owner }
            );
            if *owner != id.get() {
                assert_eq!(
                    &spatial.cache().continuous_bytes()[index * 4..index * 4 + 4],
                    &continuous[index * 4..index * 4 + 4]
                );
            }
        }
        assert!(!spatial.can_undo());
        assert!(can_undo_region_removal(&motion));
        motion.spatial.as_mut().unwrap().recenter([20, 20]).unwrap();
        motion.spatial.as_mut().unwrap().recenter([0, 0]).unwrap();
        assert!(
            !motion
                .spatial
                .as_ref()
                .unwrap()
                .cache()
                .assignments()
                .contains(&id.get())
        );
        undo_region_removal(&mut motion).unwrap();
        assert_eq!(motion.regions.selected(), id);
        assert_eq!(
            motion.spatial.as_ref().unwrap().cache().assignments(),
            before
        );
        assert_eq!(
            motion.spatial.as_ref().unwrap().cache().continuous_bytes(),
            continuous
        );
    }

    #[test]
    fn removed_id_can_be_reused_without_reviving_older_paint() {
        let mut motion = fixture();
        let id = motion.regions.new_variation().unwrap();
        paint(
            &mut motion,
            MotionBrushTool::ApplyBehavior { region_id: id },
            [-2.0, 0.0],
        );
        remove_selected_region(&mut motion).unwrap();
        assert_eq!(motion.regions.new_variation().unwrap(), id);
        assert!(!can_undo_region_removal(&motion));
        paint(
            &mut motion,
            MotionBrushTool::ApplyBehavior { region_id: id },
            [2.0, 0.0],
        );
        let original = motion
            .spatial
            .as_ref()
            .unwrap()
            .cache()
            .assignments()
            .to_vec();
        let session = MotionSession::capture(&motion).unwrap();
        assert_eq!(session.version, 2);
        let restored = MotionSession::parse(&session.to_bytes().unwrap())
            .unwrap()
            .prepare(&motion)
            .unwrap();
        assert_eq!(
            restored.spatial.as_ref().unwrap().cache().assignments(),
            original
        );
        let mut legacy = session.clone();
        legacy.version = 1;
        assert!(legacy.to_bytes().is_err());
    }

    #[test]
    fn default_controller_is_protected_and_edits_invalidate_removal_undo() {
        let mut motion = fixture();
        assert!(remove_selected_region(&mut motion).is_err());
        motion.regions.new_variation().unwrap();
        remove_selected_region(&mut motion).unwrap();
        motion
            .regions
            .rename_selected("Edited default".into())
            .unwrap();
        assert!(!can_undo_region_removal(&motion));
        assert!(undo_region_removal(&mut motion).is_err());
    }

    fn paint(motion: &mut MotionRenderData, tool: MotionBrushTool, point: [f32; 2]) {
        let s = motion.spatial.as_mut().unwrap();
        s.tool = tool;
        s.falloff = MotionBrushFalloff::Constant;
        s.radius = 1.0;
        s.begin_stroke(point).unwrap();
        s.end_stroke().unwrap();
    }

    #[test]
    fn session_roundtrip_preserves_world_paint_regions_ranges_and_request_sequence() {
        let mut motion = fixture();
        motion
            .regions
            .select(MotionRegionId::local(4).unwrap())
            .unwrap();
        motion
            .regions
            .rename_selected("Quiet corner".into())
            .unwrap();
        motion.regions.set_selected_enabled(false);
        let range = motion.authoring.add_range("Wind", 4, 50).unwrap();
        motion.authoring.select_range(range).unwrap();
        motion
            .authoring
            .update_behavior(|b| b.playback_speed = 1.5)
            .unwrap();
        paint(
            &mut motion,
            MotionBrushTool::AdjustVariation { target: 1.0 },
            [1.0, 1.0],
        );
        paint(
            &mut motion,
            MotionBrushTool::AdjustStrength { target: 0.3 },
            [1.0, 1.0],
        );
        motion.compile_spatial_field().unwrap();
        let request = motion.take_membership_request().unwrap().unwrap();
        motion.acknowledge_membership_request(request.request_revision);
        let bytes = MotionSession::capture(&motion).unwrap().to_bytes().unwrap();
        let session = MotionSession::parse(&bytes).unwrap();
        let mut restored = session.prepare(&motion).unwrap();
        assert_eq!(
            MotionSession::capture(&restored)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(
            restored
                .spatial
                .as_ref()
                .unwrap()
                .cache()
                .continuous_bytes(),
            motion.spatial.as_ref().unwrap().cache().continuous_bytes()
        );
        assert_eq!(
            restored.spatial.as_ref().unwrap().cache().assignments(),
            motion.spatial.as_ref().unwrap().cache().assignments()
        );
        assert!(
            restored
                .take_membership_request()
                .unwrap()
                .unwrap()
                .request_revision
                > request.request_revision
        );
        restored
            .spatial
            .as_mut()
            .unwrap()
            .recenter([20, 20])
            .unwrap();
        assert_eq!(restored.spatial.as_ref().unwrap().painted_texel_count(), 0);
        restored.spatial.as_mut().unwrap().recenter([0, 0]).unwrap();
        assert_eq!(
            restored
                .spatial
                .as_ref()
                .unwrap()
                .cache()
                .continuous_bytes(),
            motion.spatial.as_ref().unwrap().cache().continuous_bytes()
        );
    }

    #[test]
    fn invalid_or_incompatible_session_never_changes_the_live_document() {
        let mut motion = fixture();
        paint(
            &mut motion,
            MotionBrushTool::AdjustStrength { target: 0.0 },
            [1.0, 1.0],
        );
        let original = MotionSession::capture(&motion).unwrap();
        let bytes = original.to_bytes().unwrap();
        let mut bad = original.clone();
        bad.version = 9;
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.archive.total_rows += 1;
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.tile_width *= 2.0;
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.strokes[0].radius = -1.0;
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.strokes[0].fallback_region = Some(MotionRegionId::local(63).unwrap());
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.strokes[0].path[0][0] = f32::NAN;
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.regions.push(bad.regions[0].clone());
        assert!(bad.prepare(&motion).is_err());
        let mut bad = original.clone();
        bad.active_range = 20;
        assert!(bad.prepare(&motion).is_err());
        assert!(MotionSession::parse(b"not json").is_err());
        assert_eq!(
            MotionSession::capture(&motion).unwrap().to_bytes().unwrap(),
            bytes
        );
    }

    #[test]
    fn strength_on_blank_space_does_not_introduce_spatial_variation() {
        let mut motion = fixture();
        paint(
            &mut motion,
            MotionBrushTool::AdjustStrength { target: 0.5 },
            [1.0, 1.0],
        );
        let cache = motion.spatial.as_ref().unwrap().cache();
        assert!(cache.painted_texel_count() > 0);
        for texel in cache.continuous_bytes().chunks_exact(4) {
            assert_eq!(texel[2], 0);
        }
    }

    #[test]
    fn scalar_brush_works_on_blank_space_and_history_survives_recenter() {
        let mut motion = fixture();
        paint(
            &mut motion,
            MotionBrushTool::AdjustVariation { target: 1.0 },
            [1.0, 1.0],
        );
        let s = motion.spatial.as_mut().unwrap();
        let painted = s.cache().continuous_bytes().to_vec();
        assert!(s.cache().assignments().contains(&1));
        s.undo().unwrap();
        assert_eq!(s.painted_texel_count(), 0);
        assert!(s.can_redo());
        s.recenter([20, 20]).unwrap();
        s.redo().unwrap();
        assert_eq!(s.painted_texel_count(), 0);
        s.recenter([0, 0]).unwrap();
        assert_eq!(s.cache().continuous_bytes(), painted);
        s.undo().unwrap();
        paint(
            &mut motion,
            MotionBrushTool::AdjustCoherence { target: 0.0 },
            [1.0, 1.0],
        );
        assert!(!motion.spatial.as_ref().unwrap().can_redo());
    }
}
