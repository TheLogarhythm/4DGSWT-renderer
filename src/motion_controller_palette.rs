use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::motion::TimelineSample;
use crate::motion_behavior::{
    MotionBehavior, MotionBehaviorPreset, MotionChannelGains, MotionSourceRange,
};
use crate::motion_brush::{DirtyRect, MotionFieldCache};
use crate::motion_graph::{MotionGraph, SOURCE_SEGMENT_COUNT};
use crate::motion_graph_playback::{MotionGraphPlaybackController, MotionGraphPlaybackError};
use crate::motion_spatial_variation::{VARIANTS_PER_REGION, spatial_variant, variant_seed};
use crate::motion_tagging::{MotionMembershipSnapshot, physical_texel_index};
use serde::{Deserialize, Serialize};

pub const MAX_LOCAL_CONTROLLERS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MotionRegionId(u8);

impl MotionRegionId {
    pub const GLOBAL: Self = Self(0);

    pub fn local(value: u8) -> Result<Self, MotionControllerError> {
        (1..=MAX_LOCAL_CONTROLLERS as u8)
            .contains(&value)
            .then_some(Self(value))
            .ok_or_else(|| {
                MotionControllerError::new("local Motion Region ID must be within 1..=64")
            })
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionRegion {
    id: MotionRegionId,
    name: String,
    behavior: MotionBehavior,
    source_range: MotionSourceRange,
    enabled: bool,
}

impl MotionRegion {
    pub const fn id(&self) -> MotionRegionId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn behavior(&self) -> &MotionBehavior {
        &self.behavior
    }

    pub fn source_range(&self) -> &MotionSourceRange {
        &self.source_range
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }
}

#[derive(Clone)]
pub struct MotionRegionDocument {
    regions: Vec<MotionRegion>,
    selected: MotionRegionId,
    next_id: u8,
    revision: u64,
}

impl MotionRegionDocument {
    pub fn from_regions(
        regions: Vec<MotionRegion>,
        selected: MotionRegionId,
    ) -> Result<Self, MotionControllerError> {
        if regions.is_empty() || regions.len() > MAX_LOCAL_CONTROLLERS {
            return Err(MotionControllerError::new("session needs 1..=64 regions"));
        }
        let mut ids = [false; MAX_LOCAL_CONTROLLERS + 1];
        for region in &regions {
            MotionRegionId::local(region.id.get())?;
            if ids[region.id.get() as usize] || region.name.trim().is_empty() {
                return Err(MotionControllerError::new(
                    "duplicate region ID or empty region name",
                ));
            }
            ids[region.id.get() as usize] = true;
            region
                .behavior
                .validate()
                .map_err(|e| MotionControllerError::new(e.to_string()))?;
            region
                .source_range
                .validate()
                .map_err(|e| MotionControllerError::new(e.to_string()))?;
        }
        if !regions.iter().any(|r| r.id == selected) {
            return Err(MotionControllerError::new("selected region does not exist"));
        }
        let next_id = regions.iter().map(|r| r.id.get()).max().unwrap() + 1;
        Ok(Self {
            regions,
            selected,
            next_id,
            revision: 1,
        })
    }

    pub fn update_selected(
        &mut self,
        behavior: MotionBehavior,
        range: MotionSourceRange,
    ) -> Result<(), MotionControllerError> {
        behavior
            .validate()
            .map_err(|e| MotionControllerError::new(e.to_string()))?;
        range
            .validate()
            .map_err(|e| MotionControllerError::new(e.to_string()))?;
        let region = self
            .regions
            .iter_mut()
            .find(|r| r.id == self.selected)
            .unwrap();
        if region.behavior != behavior || region.source_range != range {
            region.behavior = behavior;
            region.source_range = range;
            self.revision = self.revision.wrapping_add(1);
        }
        Ok(())
    }
    pub fn artist_defaults() -> Self {
        let presets = [
            MotionBehaviorPreset::GentleSway,
            MotionBehaviorPreset::SteadyWind,
            MotionBehaviorPreset::GustyWind,
            MotionBehaviorPreset::Calm,
        ];
        let regions = presets
            .into_iter()
            .enumerate()
            .map(|(index, preset)| {
                let id = MotionRegionId((index + 1) as u8);
                let mut behavior = MotionBehavior::from_preset(preset);
                behavior.seed = variation_seed(behavior.seed, id);
                MotionRegion {
                    id,
                    name: preset.label().to_string(),
                    behavior,
                    source_range: MotionSourceRange::full_source(),
                    enabled: true,
                }
            })
            .collect();
        Self {
            regions,
            selected: MotionRegionId(1),
            next_id: 5,
            revision: 0,
        }
    }

    pub fn regions(&self) -> &[MotionRegion] {
        &self.regions
    }

    pub const fn selected(&self) -> MotionRegionId {
        self.selected
    }

    pub fn select(&mut self, id: MotionRegionId) -> Result<(), MotionControllerError> {
        if self.regions.iter().any(|region| region.id == id) {
            self.selected = id;
            Ok(())
        } else {
            Err(MotionControllerError::new(format!(
                "Motion Region {} does not exist",
                id.get()
            )))
        }
    }

    pub fn selected_region(&self) -> &MotionRegion {
        self.regions
            .iter()
            .find(|region| region.id == self.selected)
            .expect("selected Motion Region must exist")
    }

    pub fn rename_selected(&mut self, name: String) -> Result<(), MotionControllerError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(MotionControllerError::new(
                "Motion Region name cannot be empty",
            ));
        }
        let selected = self.selected;
        let region = self
            .regions
            .iter_mut()
            .find(|region| region.id == selected)
            .expect("selected Motion Region must exist");
        if region.name != name {
            region.name = name.to_string();
            self.revision = self.revision.wrapping_add(1);
        }
        Ok(())
    }

    pub fn set_selected_enabled(&mut self, enabled: bool) {
        let selected = self.selected;
        let region = self
            .regions
            .iter_mut()
            .find(|region| region.id == selected)
            .expect("selected Motion Region must exist");
        if region.enabled != enabled {
            region.enabled = enabled;
            self.revision = self.revision.wrapping_add(1);
        }
    }

    pub fn new_variation(&mut self) -> Result<MotionRegionId, MotionControllerError> {
        let id = MotionRegionId::local(
            (1..=MAX_LOCAL_CONTROLLERS as u8)
                .find(|id| !self.regions.iter().any(|region| region.id.get() == *id))
                .unwrap_or(self.next_id),
        )?;
        let parent = self.selected_region().clone();
        let base_name = format!("{} copy", parent.name);
        let mut name = base_name.clone();
        let mut suffix = 2;
        while self.regions.iter().any(|region| region.name == name) {
            name = format!("{base_name} {suffix}");
            suffix += 1;
        }
        let mut behavior = parent.behavior;
        behavior.seed = variation_seed(behavior.seed, id);
        behavior.name = name.clone();
        self.regions.push(MotionRegion {
            id,
            name,
            behavior,
            source_range: parent.source_range,
            enabled: true,
        });
        self.selected = id;
        self.next_id = self.next_id.saturating_add(1);
        self.revision = self.revision.wrapping_add(1);
        Ok(id)
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub fn can_remove_selected(&self) -> bool {
        self.selected != self.regions[0].id
    }

    pub fn remove_selected(&mut self) -> Result<(), MotionControllerError> {
        if !self.can_remove_selected() {
            return Err(MotionControllerError::new(
                "The default Motion Region cannot be removed",
            ));
        }
        self.regions.retain(|region| region.id != self.selected);
        self.selected = self.regions[0].id;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub fn restore_after_removal(&mut self, mut previous: Self) {
        previous.revision = self.revision.wrapping_add(1);
        *self = previous;
    }
}

fn variation_seed(parent: u32, id: MotionRegionId) -> u32 {
    let mut value = parent ^ u32::from(id.get()).wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x85eb_ca6b);
    value ^ (value >> 13)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionControllerFrame {
    pub slot: u8,
    pub sample: TimelineSample,
    pub gains: MotionChannelGains,
    pub valid: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MotionControllerTrace {
    pub last_selected_edge: Option<(usize, usize)>,
    pub recent_edges: Vec<(usize, usize)>,
}

struct MotionControllerEntry {
    region_id: MotionRegionId,
    variant: usize,
    slot: u8,
    behavior: MotionBehavior,
    source_range: MotionSourceRange,
    playback: MotionGraphPlaybackController,
    blend_window_seconds: f32,
    frame: MotionControllerFrame,
    trace: MotionControllerTrace,
}

impl MotionControllerEntry {
    fn new(
        source_duration_seconds: f32,
        slot: u8,
        region: &MotionRegion,
    ) -> Result<Self, MotionControllerError> {
        let behavior = region.behavior.clone();
        let source_range = region.source_range.clone();
        behavior
            .validate()
            .map_err(|error| MotionControllerError::new(error.to_string()))?;
        source_range
            .validate()
            .map_err(|error| MotionControllerError::new(error.to_string()))?;
        let playback = MotionGraphPlaybackController::new(source_duration_seconds, behavior.seed)?;
        let mut entry = Self {
            region_id: region.id,
            variant: 0,
            slot,
            behavior,
            source_range,
            playback,
            blend_window_seconds: 0.0,
            frame: MotionControllerFrame {
                slot,
                sample: TimelineSample::Source { u: 0.0 },
                gains: MotionChannelGains::default(),
                valid: true,
            },
            trace: MotionControllerTrace::default(),
        };
        entry.configure(source_duration_seconds)?;
        Ok(entry)
    }

    fn sync_region(
        &mut self,
        source_duration_seconds: f32,
        region: &MotionRegion,
    ) -> Result<bool, MotionControllerError> {
        if self.behavior == region.behavior && self.source_range == region.source_range {
            return Ok(false);
        }
        self.behavior = region.behavior.clone();
        self.source_range = region.source_range.clone();
        self.configure(source_duration_seconds)?;
        Ok(true)
    }

    fn configure(&mut self, source_duration_seconds: f32) -> Result<(), MotionControllerError> {
        self.behavior
            .validate()
            .map_err(|error| MotionControllerError::new(error.to_string()))?;
        self.source_range
            .validate()
            .map_err(|error| MotionControllerError::new(error.to_string()))?;
        self.playback.set_enabled(self.behavior.stochastic_enabled);
        self.playback
            .set_branch_probability(self.behavior.branch_probability())?;
        self.playback
            .set_transition_seconds(self.behavior.transition_seconds)?;
        self.playback
            .set_minimum_dwell_segments(self.behavior.minimum_dwell_segments());
        if self.playback.config().seed != self.behavior.seed {
            self.playback.set_seed(self.behavior.seed);
        }
        self.playback.set_source_range(
            self.source_range.start_segment,
            self.source_range.end_segment,
        )?;
        // Keep old session fields round-trippable without a hidden wind bias.
        self.playback.set_direction_preference([0.0; 2], 0.0)?;
        let range_duration = source_duration_seconds
            * (self.source_range.end_segment - self.source_range.start_segment) as f32
            / SOURCE_SEGMENT_COUNT as f32;
        self.blend_window_seconds = self
            .behavior
            .transition_seconds
            .min(range_duration * 0.5)
            .min(source_duration_seconds * 0.5);
        self.refresh_frame();
        Ok(())
    }

    fn advance(
        &mut self,
        delta_seconds: f32,
        playing: bool,
        graph: &MotionGraph,
    ) -> Result<(), MotionControllerError> {
        let sample = if playing && delta_seconds > 0.0 {
            self.playback.advance(
                delta_seconds,
                graph,
                self.behavior.endpoint_policy,
                self.behavior.playback_speed,
                self.behavior.transition_seconds,
                self.blend_window_seconds,
            )?
        } else {
            self.playback.sample()
        };
        self.frame.sample = sample;
        self.frame.gains = self.behavior.gains;
        self.trace.last_selected_edge = self.playback.last_selected_edge();
        self.trace
            .recent_edges
            .clone_from(&self.playback.trace().recent_edges);
        Ok(())
    }

    fn restart(&mut self) {
        self.playback.restart();
        self.trace = MotionControllerTrace::default();
        self.refresh_frame();
    }

    fn refresh_frame(&mut self) {
        self.frame = MotionControllerFrame {
            slot: self.slot,
            sample: self.playback.sample(),
            gains: self.behavior.gains,
            valid: true,
        };
    }
}

pub struct MotionControllerPalette {
    source_duration_seconds: f32,
    entries: Vec<MotionControllerEntry>,
    frames: Vec<MotionControllerFrame>,
    mapping_revision: u64,
    dirty: bool,
    synced_revision: Option<u64>,
}

impl MotionControllerPalette {
    pub fn new(source_duration_seconds: f32) -> Result<Self, MotionControllerError> {
        if !source_duration_seconds.is_finite() || source_duration_seconds <= 0.0 {
            return Err(MotionControllerError::new(
                "controller palette source duration must be positive and finite",
            ));
        }
        Ok(Self {
            source_duration_seconds,
            entries: Vec::new(),
            frames: Vec::new(),
            mapping_revision: 0,
            dirty: true,
            synced_revision: None,
        })
    }

    pub fn sync_root_regions(
        &mut self,
        regions: &MotionRegionDocument,
    ) -> Result<(), MotionControllerError> {
        if self.synced_revision == Some(regions.revision()) {
            return Ok(());
        }
        let enabled_ids = regions
            .regions()
            .iter()
            .filter(|region| region.enabled())
            .map(MotionRegion::id)
            .collect::<Vec<_>>();
        if enabled_ids.len() > MAX_LOCAL_CONTROLLERS {
            return Err(MotionControllerError::new(
                "controller palette exceeds 64 local states",
            ));
        }

        let previous_mapping = self
            .entries
            .iter()
            .map(|entry| (entry.region_id, entry.variant, entry.slot))
            .collect::<Vec<_>>();
        let variants = if enabled_ids.is_empty() {
            1
        } else {
            (MAX_LOCAL_CONTROLLERS / enabled_ids.len()).min(VARIANTS_PER_REGION)
        };
        self.entries
            .retain(|entry| enabled_ids.contains(&entry.region_id) && entry.variant < variants);
        let mut used_slots = self
            .entries
            .iter()
            .map(|entry| entry.slot)
            .collect::<Vec<_>>();

        for region in regions.regions().iter().filter(|region| region.enabled()) {
            if let Some(entry) = self
                .entries
                .iter_mut()
                .find(|entry| entry.region_id == region.id() && entry.variant == 0)
            {
                self.dirty |= entry.sync_region(self.source_duration_seconds, region)?;
                continue;
            }
            let slot = (1..=MAX_LOCAL_CONTROLLERS as u8)
                .find(|slot| !used_slots.contains(slot))
                .ok_or_else(|| MotionControllerError::new("controller palette has no free slot"))?;
            self.entries.push(MotionControllerEntry::new(
                self.source_duration_seconds,
                slot,
                region,
            )?);
            used_slots.push(slot);
            self.dirty = true;
        }
        for variant in 1..variants {
            for region in regions.regions().iter().filter(|r| r.enabled()) {
                let mut variant_region = region.clone();
                variant_region.behavior.seed = variant_seed(region.behavior.seed, variant);
                if let Some(entry) = self
                    .entries
                    .iter_mut()
                    .find(|e| e.region_id == region.id && e.variant == variant)
                {
                    self.dirty |=
                        entry.sync_region(self.source_duration_seconds, &variant_region)?;
                } else {
                    let slot = (1..=MAX_LOCAL_CONTROLLERS as u8)
                        .find(|s| !used_slots.contains(s))
                        .ok_or_else(|| MotionControllerError::new("no spatial variant slot"))?;
                    let mut entry = MotionControllerEntry::new(
                        self.source_duration_seconds,
                        slot,
                        &variant_region,
                    )?;
                    entry.variant = variant;
                    self.entries.push(entry);
                    used_slots.push(slot);
                    self.dirty = true;
                }
            }
        }
        self.entries.sort_by_key(|entry| entry.slot);
        let current_mapping = self
            .entries
            .iter()
            .map(|entry| (entry.region_id, entry.variant, entry.slot))
            .collect::<Vec<_>>();
        if current_mapping != previous_mapping {
            self.mapping_revision = self.mapping_revision.wrapping_add(1);
            self.dirty = true;
        }
        self.refresh_frames();
        // Seeds affect the world-space selector even if slot IDs did not move.
        self.mapping_revision = self.mapping_revision.wrapping_add(1);
        self.synced_revision = Some(regions.revision());
        Ok(())
    }

    pub fn restart(&mut self) {
        for entry in &mut self.entries {
            entry.restart();
        }
        self.refresh_frames();
        self.dirty = true;
    }

    pub fn advance(
        &mut self,
        delta_seconds: f32,
        playing: bool,
        graph: &MotionGraph,
    ) -> Result<&[MotionControllerFrame], MotionControllerError> {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return Err(MotionControllerError::new(
                "controller palette frame delta must be finite and nonnegative",
            ));
        }
        for entry in &mut self.entries {
            entry.advance(delta_seconds, playing, graph)?;
        }
        self.refresh_frames();
        self.dirty |= playing && delta_seconds > 0.0;
        Ok(&self.frames)
    }

    pub fn frames(&self) -> &[MotionControllerFrame] {
        &self.frames
    }

    pub fn slot_for_region(&self, id: MotionRegionId) -> Option<u8> {
        self.entries
            .iter()
            .find(|entry| entry.region_id == id && entry.variant == 0)
            .map(|entry| entry.slot)
    }

    pub fn active_local_count(&self) -> usize {
        self.entries.len()
    }

    fn spatial_slots(&self) -> [[u8; VARIANTS_PER_REGION]; MAX_LOCAL_CONTROLLERS + 1] {
        let mut slots = [[0; VARIANTS_PER_REGION]; MAX_LOCAL_CONTROLLERS + 1];
        for entry in &self.entries {
            slots[entry.region_id.get() as usize][entry.variant] = entry.slot;
        }
        slots
    }

    pub const fn mapping_revision(&self) -> u64 {
        self.mapping_revision
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn trace_for_slot(&self, slot: u8) -> Option<&MotionControllerTrace> {
        self.entries
            .iter()
            .find(|entry| entry.slot == slot)
            .map(|entry| &entry.trace)
    }

    fn refresh_frames(&mut self) {
        self.frames.clear();
        self.frames
            .extend(self.entries.iter().map(|entry| entry.frame));
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledMotionField {
    assignments: Vec<u8>,
    active_texels: Vec<bool>,
    slot_texel_counts: [u32; MAX_LOCAL_CONTROLLERS + 1],
    tile_active_counts: Vec<u32>,
    map_tiles: [u32; 2],
    dimensions: [u32; 2],
    center_tile: [i32; 2],
    storage_offset: [u32; 2],
    tile_width_bits: u32,
    palette_count: u32,
    palette_mapping_revision: u64,
    cache_content_revision: u64,
    revision: u64,
    membership_revision: u64,
    dirty_regions: Vec<DirtyRect>,
}

impl CompiledMotionField {
    pub fn compile_roots(
        cache: &MotionFieldCache,
        regions: &MotionRegionDocument,
        palette: &MotionControllerPalette,
        previous: Option<&Self>,
    ) -> Result<Self, MotionControllerError> {
        if let Some(previous) = previous {
            let mut compiled = previous.clone();
            compiled.sync_roots(cache, regions, palette, &[])?;
            return Ok(compiled);
        }
        Self::compile_full(cache, regions, palette, 1, 1)
    }

    fn compile_full(
        cache: &MotionFieldCache,
        regions: &MotionRegionDocument,
        palette: &MotionControllerPalette,
        revision: u64,
        membership_revision: u64,
    ) -> Result<Self, MotionControllerError> {
        let layout = cache.layout();
        let dimensions = layout.texture_size();
        let map_tiles = layout.map_tiles();
        let texels_per_tile = layout.quality().texels_per_tile();
        let texel_count = usize::try_from(u64::from(dimensions[0]) * u64::from(dimensions[1]))
            .map_err(|_| MotionControllerError::new("compiled motion field is too large"))?;
        if cache.assignments().len() != texel_count
            || cache.continuous_bytes().len() != texel_count.saturating_mul(4)
        {
            return Err(MotionControllerError::new(
                "motion field cache byte lengths do not match its layout",
            ));
        }

        let region_slots = palette.spatial_slots();

        let mut assignments = vec![0_u8; texel_count];
        let covered_len = usize::try_from(u64::from(map_tiles[0]) * u64::from(map_tiles[1]))
            .map_err(|_| MotionControllerError::new("motion field tile coverage is too large"))?;
        let mut active_texels = vec![false; texel_count];
        let mut slot_texel_counts = [0; MAX_LOCAL_CONTROLLERS + 1];
        let mut tile_active_counts = vec![0_u32; covered_len];
        let storage_offset = cache.storage_offset();
        for logical_y in 0..dimensions[1] {
            for logical_x in 0..dimensions[0] {
                let physical_x = (logical_x + storage_offset[0]) % dimensions[0];
                let physical_y = (logical_y + storage_offset[1]) % dimensions[1];
                let physical_index = usize::try_from(
                    u64::from(physical_y) * u64::from(dimensions[0]) + u64::from(physical_x),
                )
                .expect("validated motion field dimensions fit usize");
                let region_byte = cache.assignments()[physical_index];
                let slot = compiled_slot(
                    cache,
                    regions,
                    &region_slots,
                    region_byte,
                    physical_index,
                    [logical_x, logical_y],
                );
                assignments[physical_index] = slot;
                let active = slot != 0 && cache.continuous_bytes()[physical_index * 4] != 0;
                active_texels[physical_index] = active;
                if active {
                    slot_texel_counts[slot as usize] += 1;
                    let tile_x = logical_x / texels_per_tile;
                    let tile_y = logical_y / texels_per_tile;
                    let tile_index = usize::try_from(
                        u64::from(tile_y) * u64::from(map_tiles[0]) + u64::from(tile_x),
                    )
                    .expect("validated motion map dimensions fit usize");
                    tile_active_counts[tile_index] += 1;
                }
            }
        }

        let palette_count = u32::try_from(palette.active_local_count())
            .expect("local controller count is bounded to 64");
        Ok(Self {
            assignments,
            active_texels,
            slot_texel_counts,
            tile_active_counts,
            map_tiles,
            dimensions,
            center_tile: layout.center_tile(),
            storage_offset: cache.storage_offset(),
            tile_width_bits: layout.tile_width().to_bits(),
            palette_count,
            palette_mapping_revision: palette.mapping_revision(),
            cache_content_revision: cache.content_revision(),
            revision,
            membership_revision,
            dirty_regions: vec![
                DirtyRect::from_bounds([0, 0], dimensions)
                    .expect("validated motion field dimensions are positive"),
            ],
        })
    }

    pub fn sync_roots(
        &mut self,
        cache: &MotionFieldCache,
        regions: &MotionRegionDocument,
        palette: &MotionControllerPalette,
        dirty_regions: &[DirtyRect],
    ) -> Result<bool, MotionControllerError> {
        let layout = cache.layout();
        let dimensions = layout.texture_size();
        let map_tiles = layout.map_tiles();
        let palette_count = u32::try_from(palette.active_local_count())
            .expect("local controller count is bounded to 64");
        let cache_unchanged = self.cache_content_revision == cache.content_revision();
        let palette_unchanged = self.palette_mapping_revision == palette.mapping_revision()
            && self.palette_count == palette_count;
        if self.dimensions == dimensions
            && self.map_tiles == map_tiles
            && cache_unchanged
            && palette_unchanged
        {
            self.dirty_regions.clear();
            return Ok(false);
        }

        let layout_changed = self.dimensions != dimensions
            || self.map_tiles != map_tiles
            || self.tile_width_bits != layout.tile_width().to_bits();
        let center_changed = self.center_tile != layout.center_tile();
        let center_delta = [
            i64::from(layout.center_tile()[0]) - i64::from(self.center_tile[0]),
            i64::from(layout.center_tile()[1]) - i64::from(self.center_tile[1]),
        ];
        let texels_per_tile = layout.quality().texels_per_tile();
        let storage_rebuilt = (0..2).any(|axis| {
            center_delta[axis].unsigned_abs() >= u64::from(map_tiles[axis])
                || (i64::from(self.storage_offset[axis])
                    + center_delta[axis] * i64::from(texels_per_tile))
                .rem_euclid(i64::from(dimensions[axis])) as u32
                    != cache.storage_offset()[axis]
        });
        if layout_changed
            || storage_rebuilt
            || !palette_unchanged
            || (!cache_unchanged && dirty_regions.is_empty())
        {
            let rebuilt = Self::compile_full(
                cache,
                regions,
                palette,
                self.revision,
                self.membership_revision,
            )?;
            let contract_changed = self.assignments != rebuilt.assignments
                || self.active_texels != rebuilt.active_texels
                || self.tile_active_counts != rebuilt.tile_active_counts
                || self.palette_count != rebuilt.palette_count
                || layout_changed;
            let membership_changed = self.active_texels != rebuilt.active_texels
                || self.tile_active_counts != rebuilt.tile_active_counts
                || layout_changed
                || center_changed;
            let old_assignments = std::mem::take(&mut self.assignments);
            let mapping_changed = layout_changed || storage_rebuilt || !palette_unchanged;
            *self = rebuilt;
            if contract_changed {
                self.revision = self.revision.wrapping_add(1);
            }
            if membership_changed {
                self.membership_revision = self.membership_revision.wrapping_add(1);
            }
            self.dirty_regions = if mapping_changed {
                vec![
                    DirtyRect::from_bounds([0, 0], dimensions)
                        .expect("validated motion field dimensions are positive"),
                ]
            } else {
                assignment_diff_bounds(
                    &old_assignments,
                    &self.assignments,
                    dimensions,
                    cache.storage_offset(),
                )
                .into_iter()
                .collect()
            };
            return Ok(contract_changed);
        }

        validate_compiled_cache_lengths(cache, dimensions)?;
        let texels_per_tile = layout.quality().texels_per_tile();
        let region_slots = palette.spatial_slots();
        let storage_offset = cache.storage_offset();
        let mut contract_changed = center_changed;
        let mut membership_changed = center_changed;
        if center_changed {
            let center_delta = [
                i64::from(layout.center_tile()[0]) - i64::from(self.center_tile[0]),
                i64::from(layout.center_tile()[1]) - i64::from(self.center_tile[1]),
            ];
            self.tile_active_counts =
                shift_tile_active_counts(&self.tile_active_counts, map_tiles, center_delta);
        }
        let mut assignment_dirty = None;
        for &rect in dirty_regions {
            let clipped_max = [
                rect.max_exclusive()[0].min(dimensions[0]),
                rect.max_exclusive()[1].min(dimensions[1]),
            ];
            let Some(rect) = DirtyRect::from_bounds(rect.min(), clipped_max) else {
                continue;
            };
            let mut rect_assignment_changed = false;
            for logical_y in rect.min()[1]..rect.max_exclusive()[1] {
                for logical_x in rect.min()[0]..rect.max_exclusive()[0] {
                    let physical_x = (logical_x + storage_offset[0]) % dimensions[0];
                    let physical_y = (logical_y + storage_offset[1]) % dimensions[1];
                    let physical_index = usize::try_from(
                        u64::from(physical_y) * u64::from(dimensions[0]) + u64::from(physical_x),
                    )
                    .expect("validated motion field dimensions fit usize");
                    let region_byte = cache.assignments()[physical_index];
                    let slot = compiled_slot(
                        cache,
                        regions,
                        &region_slots,
                        region_byte,
                        physical_index,
                        [logical_x, logical_y],
                    );
                    let active = slot != 0 && cache.continuous_bytes()[physical_index * 4] != 0;
                    let old_slot = self.assignments[physical_index];
                    let old_active = self.active_texels[physical_index];
                    // Newly exposed ring texels reuse old storage, but their new tile
                    // coverage was initialized to zero when shifting tile counts.
                    let retained_world = (0..2).all(|axis| {
                        let old_coord = i64::from([logical_x, logical_y][axis])
                            + center_delta[axis] * i64::from(texels_per_tile);
                        old_coord >= 0 && old_coord < i64::from(dimensions[axis])
                    });
                    let old_coverage = old_active && retained_world;
                    if old_slot == slot && old_active == active && old_coverage == active {
                        continue;
                    }
                    contract_changed = true;
                    if old_active {
                        self.slot_texel_counts[old_slot as usize] -= 1;
                    }
                    if active {
                        self.slot_texel_counts[slot as usize] += 1;
                    }
                    membership_changed |= old_active != active;
                    rect_assignment_changed |= old_slot != slot;
                    let tile_x = logical_x / texels_per_tile;
                    let tile_y = logical_y / texels_per_tile;
                    let tile_index = usize::try_from(
                        u64::from(tile_y) * u64::from(map_tiles[0]) + u64::from(tile_x),
                    )
                    .expect("validated motion map dimensions fit usize");
                    match (old_coverage, active) {
                        (false, true) => self.tile_active_counts[tile_index] += 1,
                        (true, false) => self.tile_active_counts[tile_index] -= 1,
                        _ => {}
                    }
                    self.assignments[physical_index] = slot;
                    self.active_texels[physical_index] = active;
                }
            }
            if rect_assignment_changed {
                assignment_dirty = Some(union_dirty_rect(assignment_dirty, rect));
            }
        }
        self.cache_content_revision = cache.content_revision();
        self.palette_mapping_revision = palette.mapping_revision();
        self.palette_count = palette_count;
        self.center_tile = layout.center_tile();
        self.storage_offset = cache.storage_offset();
        self.tile_width_bits = layout.tile_width().to_bits();
        if contract_changed {
            self.revision = self.revision.wrapping_add(1);
        }
        if membership_changed {
            self.membership_revision = self.membership_revision.wrapping_add(1);
        }
        self.dirty_regions = assignment_dirty.into_iter().collect();
        Ok(contract_changed)
    }

    pub fn assignments(&self) -> &[u8] {
        &self.assignments
    }

    pub fn slot_is_used(&self, slot: u8) -> bool {
        self.slot_texel_counts
            .get(slot as usize)
            .is_some_and(|&count| count > 0)
    }

    pub fn tile_has_local_motion(&self, map_coord: [u32; 2]) -> bool {
        if map_coord[0] >= self.map_tiles[0] || map_coord[1] >= self.map_tiles[1] {
            return false;
        }
        let index = usize::try_from(
            u64::from(map_coord[1]) * u64::from(self.map_tiles[0]) + u64::from(map_coord[0]),
        )
        .expect("validated motion map dimensions fit usize");
        self.tile_active_counts[index] != 0
    }

    pub const fn palette_count(&self) -> u32 {
        self.palette_count
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub const fn membership_revision(&self) -> u64 {
        self.membership_revision
    }

    pub fn membership_snapshot(
        &self,
        cache: &MotionFieldCache,
        request_revision: u64,
    ) -> Result<MotionMembershipSnapshot, MotionControllerError> {
        if cache.layout().texture_size() != self.dimensions
            || cache.layout().map_tiles() != self.map_tiles
            || cache.storage_offset()[0] >= self.dimensions[0]
            || cache.storage_offset()[1] >= self.dimensions[1]
        {
            return Err(MotionControllerError::new(
                "motion membership snapshot does not match the compiled field layout",
            ));
        }
        let active_texels = self
            .active_texels
            .iter()
            .map(|&active| u8::from(active))
            .collect::<Vec<_>>();
        let active_tiles = self
            .tile_active_counts
            .iter()
            .map(|&count| u8::from(count != 0))
            .collect::<Vec<_>>();
        let layout = cache.layout();
        Ok(MotionMembershipSnapshot::new(
            request_revision,
            self.membership_revision,
            self.dimensions,
            self.map_tiles,
            layout.world_origin(),
            layout.texel_world_size(),
            cache.storage_offset(),
            Arc::from(active_texels),
            Arc::from(active_tiles),
        ))
    }

    pub fn dirty_regions(&self) -> &[DirtyRect] {
        &self.dirty_regions
    }

    pub fn active_slot_at_world(&self, cache: &MotionFieldCache, world_xy: [f32; 2]) -> Option<u8> {
        if cache.layout().texture_size() != self.dimensions {
            return None;
        }
        let layout = cache.layout();
        let index = physical_texel_index(
            world_xy,
            layout.world_origin(),
            layout.texel_world_size(),
            self.dimensions,
            cache.storage_offset(),
        )?;
        self.active_texels
            .get(index)
            .copied()
            .unwrap_or(false)
            .then(|| self.assignments[index])
            .filter(|&slot| slot != 0)
    }
}

fn shift_tile_active_counts(
    previous: &[u32],
    map_tiles: [u32; 2],
    center_delta: [i64; 2],
) -> Vec<u32> {
    let mut shifted = vec![0; previous.len()];
    for new_y in 0..map_tiles[1] {
        for new_x in 0..map_tiles[0] {
            let old_x = i64::from(new_x) + center_delta[0];
            let old_y = i64::from(new_y) + center_delta[1];
            if old_x < 0
                || old_y < 0
                || old_x >= i64::from(map_tiles[0])
                || old_y >= i64::from(map_tiles[1])
            {
                continue;
            }
            let new_index =
                usize::try_from(u64::from(new_y) * u64::from(map_tiles[0]) + u64::from(new_x))
                    .expect("validated motion map dimensions fit usize");
            let old_index = usize::try_from(
                u64::try_from(old_y).expect("validated nonnegative tile row")
                    * u64::from(map_tiles[0])
                    + u64::try_from(old_x).expect("validated nonnegative tile column"),
            )
            .expect("validated motion map dimensions fit usize");
            shifted[new_index] = previous[old_index];
        }
    }
    shifted
}

fn validate_compiled_cache_lengths(
    cache: &MotionFieldCache,
    dimensions: [u32; 2],
) -> Result<(), MotionControllerError> {
    let texel_count = usize::try_from(u64::from(dimensions[0]) * u64::from(dimensions[1]))
        .map_err(|_| MotionControllerError::new("compiled motion field is too large"))?;
    if cache.assignments().len() != texel_count
        || cache.continuous_bytes().len() != texel_count.saturating_mul(4)
    {
        return Err(MotionControllerError::new(
            "motion field cache byte lengths do not match its layout",
        ));
    }
    Ok(())
}

fn compiled_slot(
    cache: &MotionFieldCache,
    regions: &MotionRegionDocument,
    slots: &[[u8; VARIANTS_PER_REGION]; MAX_LOCAL_CONTROLLERS + 1],
    region_byte: u8,
    physical: usize,
    logical: [u32; 2],
) -> u8 {
    let Some(choices) = slots.get(region_byte as usize) else {
        return 0;
    };
    if choices[0] == 0 {
        return 0;
    }
    let continuous = &cache.continuous_bytes()[physical * 4..physical * 4 + 4];
    if continuous[2] == 0 {
        return choices[0];
    }
    let layout = cache.layout();
    let origin = layout.world_origin();
    let texel = layout.texel_world_size();
    let world = [
        origin[0] + (logical[0] as f32 + 0.5) * texel[0],
        origin[1] + (logical[1] as f32 + 0.5) * texel[1],
    ];
    let seed = regions
        .regions()
        .iter()
        .find(|r| r.id.get() == region_byte)
        .map(|r| r.behavior.seed)
        .unwrap_or(0);
    choices[spatial_variant(
        world,
        layout.tile_width(),
        continuous[2],
        continuous[3],
        seed,
        choices.iter().filter(|&&s| s != 0).count(),
    )]
}

fn union_dirty_rect(existing: Option<DirtyRect>, next: DirtyRect) -> DirtyRect {
    let Some(existing) = existing else {
        return next;
    };
    DirtyRect::from_bounds(
        [
            existing.min()[0].min(next.min()[0]),
            existing.min()[1].min(next.min()[1]),
        ],
        [
            existing.max_exclusive()[0].max(next.max_exclusive()[0]),
            existing.max_exclusive()[1].max(next.max_exclusive()[1]),
        ],
    )
    .expect("union of nonempty dirty rectangles is nonempty")
}

fn assignment_diff_bounds(
    previous: &[u8],
    current: &[u8],
    dimensions: [u32; 2],
    storage_offset: [u32; 2],
) -> Option<DirtyRect> {
    let mut min = dimensions;
    let mut max = [0_u32; 2];
    let mut changed = false;
    for (physical_index, (&before, &after)) in previous.iter().zip(current).enumerate() {
        if before == after {
            continue;
        }
        changed = true;
        let physical_index = u32::try_from(physical_index).ok()?;
        let physical_x = physical_index % dimensions[0];
        let physical_y = physical_index / dimensions[0];
        let logical_x = (physical_x + dimensions[0] - storage_offset[0]) % dimensions[0];
        let logical_y = (physical_y + dimensions[1] - storage_offset[1]) % dimensions[1];
        min[0] = min[0].min(logical_x);
        min[1] = min[1].min(logical_y);
        max[0] = max[0].max(logical_x + 1);
        max[1] = max[1].max(logical_y + 1);
    }
    changed.then(|| {
        DirtyRect::from_bounds(min, max).expect("changed assignment bounds must be nonempty")
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionControllerError(String);

impl MotionControllerError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MotionControllerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for MotionControllerError {}

impl From<MotionGraphPlaybackError> for MotionControllerError {
    fn from(error: MotionGraphPlaybackError) -> Self {
        Self(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_named_style_keeps_motion_range_and_isolates_amount_edits() {
        let mut regions = MotionRegionDocument::artist_defaults();
        regions.rename_selected("Foreground".into()).unwrap();
        let original = regions.selected_region().clone();
        let copy = regions.new_variation().unwrap();
        assert_ne!(copy, original.id());
        assert!(regions.selected_region().name().starts_with("Foreground"));
        assert_eq!(
            regions.selected_region().source_range(),
            original.source_range()
        );
        assert_ne!(
            regions.selected_region().behavior().seed,
            original.behavior().seed
        );
        let mut behavior = regions.selected_region().behavior().clone();
        behavior.gains.master = 1.8;
        regions
            .update_selected(behavior, original.source_range().clone())
            .unwrap();
        regions.select(original.id()).unwrap();
        assert_eq!(regions.selected_region().behavior(), original.behavior());
        regions.new_variation().unwrap();
        let names = regions
            .regions()
            .iter()
            .map(|region| region.name())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(names.len(), regions.regions().len());
    }

    #[test]
    fn legacy_direction_fields_do_not_apply_an_invisible_local_bias() {
        let mut region = MotionRegionDocument::artist_defaults()
            .selected_region()
            .clone();
        region.behavior.horizontal_direction = [-1.0, 0.0];
        region.behavior.direction_influence = 0.7;
        let entry = MotionControllerEntry::new(2.0, 1, &region).unwrap();
        assert_eq!(entry.playback.config().direction_influence, 0.0);
        assert_eq!(entry.behavior.direction_influence, 0.7);
    }

    #[test]
    fn compiled_variation_changes_slots_without_changing_membership_and_survives_recenter() {
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let layout = MotionFieldLayout::new([9, 3], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut spatial = crate::motion_brush::MotionBrushRuntime::new(layout).unwrap();
        spatial.radius = 15.0;
        spatial.falloff = MotionBrushFalloff::Constant;
        spatial.begin_stroke([0.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(spatial.cache(), &regions, &palette, None).unwrap();
        let membership = compiled.membership_revision();
        let root = palette.slot_for_region(regions.selected()).unwrap();
        spatial.take_dirty_regions();
        spatial.tool = MotionBrushTool::AdjustVariation { target: 1.0 };
        spatial.begin_stroke([0.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        spatial.tool = MotionBrushTool::AdjustCoherence { target: 0.0 };
        spatial.begin_stroke([0.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        compiled
            .sync_roots(spatial.cache(), &regions, &palette, spatial.dirty_regions())
            .unwrap();
        assert_eq!(compiled.membership_revision(), membership);
        let before = (-3..=3)
            .map(|x| {
                compiled
                    .active_slot_at_world(spatial.cache(), [x as f32 * 4.0 + 0.5, 2.0])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(before.iter().all(|&s| s != root));
        assert!(before.iter().any(|&s| s != before[0]));
        for &slot in &before {
            assert!(compiled.slot_is_used(slot));
        }
        spatial.take_dirty_regions();
        spatial.recenter([1, 0]).unwrap();
        compiled
            .sync_roots(spatial.cache(), &regions, &palette, spatial.dirty_regions())
            .unwrap();
        let after = (-3..=3)
            .map(|x| {
                compiled
                    .active_slot_at_world(spatial.cache(), [x as f32 * 4.0 + 0.5, 2.0])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(before, after);
        spatial.take_dirty_regions();
        spatial.tool = MotionBrushTool::Synchronize {
            region_id: regions.selected(),
        };
        spatial.begin_stroke([0.0, 2.0]).unwrap();
        spatial.end_stroke().unwrap();
        compiled
            .sync_roots(spatial.cache(), &regions, &palette, spatial.dirty_regions())
            .unwrap();
        assert_eq!(
            compiled.active_slot_at_world(spatial.cache(), [0.5, 2.0]),
            Some(root)
        );
    }

    #[test]
    fn unchanged_region_sync_keeps_mapping_and_seeded_variant_traces_stable() {
        let mut regions = MotionRegionDocument::artist_defaults();
        configure_gusty_root(&mut regions, MotionRegionId::local(1).unwrap());
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let revision = palette.mapping_revision();
        palette.sync_root_regions(&regions).unwrap();
        assert_eq!(palette.mapping_revision(), revision);
        let choices = palette.spatial_slots()[1];
        let graph = branching_graph();
        let mut diverged = false;
        for _ in 0..120 {
            palette.advance(0.05, true, &graph).unwrap();
            diverged |= palette.trace_for_slot(choices[0]) != palette.trace_for_slot(choices[1]);
        }
        assert!(diverged);
        let range = regions.selected_region().source_range().clone();
        let mut behavior = regions.selected_region().behavior().clone();
        behavior.seed += 1;
        regions.update_selected(behavior, range).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        assert!(palette.mapping_revision() > revision);
    }
    use crate::motion::LoopPolicy;
    use crate::motion_behavior::MotionBehaviorPreset;
    use crate::motion_brush::{
        MotionBrushDocument, MotionBrushFalloff, MotionBrushStroke, MotionBrushTool,
        MotionFieldCache, MotionFieldLayout, MotionFieldQuality,
    };
    use crate::motion_graph::{
        MotionDiscontinuity, MotionGraph, MotionGraphNode, MotionGraphSettings, MotionJump,
        SOURCE_SEGMENT_COUNT, rank_eligible_jumps,
    };

    fn branching_graph() -> MotionGraph {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| {
                let candidates = [10, 20]
                    .into_iter()
                    .map(|offset| {
                        MotionJump::unranked(
                            (source_segment + offset) % SOURCE_SEGMENT_COUNT,
                            MotionDiscontinuity::uniform(0.1, 0.2),
                        )
                    })
                    .collect();
                MotionGraphNode {
                    source_segment,
                    jumps: rank_eligible_jumps(source_segment, candidates, settings),
                }
            })
            .collect();
        MotionGraph::new(nodes, settings).unwrap()
    }

    fn configure_gusty_root(document: &mut MotionRegionDocument, id: MotionRegionId) {
        for region in &mut document.regions {
            region.enabled = region.id == id;
        }
        let region = document
            .regions
            .iter_mut()
            .find(|region| region.id == id)
            .unwrap();
        region.behavior.stochastic_enabled = true;
        region.behavior.variation = 1.0;
        region.behavior.gust_frequency = 1.0;
        region.behavior.playback_speed = 1.0;
        region.behavior.endpoint_policy = LoopPolicy::DirectWrap;
        region.behavior.transition_seconds = 0.01;
        region.source_range = MotionSourceRange::full_source();
        document.revision = document.revision.wrapping_add(1);
    }

    #[test]
    fn motion_region_artist_defaults_have_stable_local_ids_and_presets() {
        let document = MotionRegionDocument::artist_defaults();
        let actual = document
            .regions()
            .iter()
            .map(|region| {
                (
                    region.id().get(),
                    region.name(),
                    region.behavior().preset,
                    region.enabled(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (1, "Gentle", MotionBehaviorPreset::GentleSway, true),
                (2, "Steady", MotionBehaviorPreset::SteadyWind, true),
                (3, "Lively", MotionBehaviorPreset::GustyWind, true),
                (4, "Subtle", MotionBehaviorPreset::Calm, true),
            ]
        );
        assert_eq!(document.selected().get(), 1);
    }

    #[test]
    fn motion_region_local_id_rejects_global_and_reserved_values() {
        assert_eq!(MotionRegionId::local(1).unwrap().get(), 1);
        assert_eq!(MotionRegionId::local(64).unwrap().get(), 64);
        assert!(MotionRegionId::local(0).is_err());
        assert!(MotionRegionId::local(65).is_err());
    }

    #[test]
    fn motion_region_new_variation_clones_behavior_with_distinct_identity_and_seed() {
        let mut document = MotionRegionDocument::artist_defaults();
        document.select(MotionRegionId::local(3).unwrap()).unwrap();
        let original = document.selected_region().clone();
        let previous_revision = document.revision();

        let created_id = document.new_variation().unwrap();
        let created = document.selected_region();

        assert_eq!(created_id.get(), 5);
        assert_eq!(created.id(), created_id);
        assert_eq!(created.behavior().preset, original.behavior().preset);
        assert_eq!(created.source_range(), original.source_range());
        assert_ne!(created.behavior().seed, original.behavior().seed);
        assert_eq!(created.name(), "Lively copy");
        assert_eq!(document.revision(), previous_revision + 1);
    }

    #[test]
    fn motion_region_selection_is_not_an_authoring_revision_and_disabled_regions_can_reenable() {
        let mut document = MotionRegionDocument::artist_defaults();
        let revision = document.revision();
        let calm = MotionRegionId::local(4).unwrap();

        document.select(calm).unwrap();
        assert_eq!(document.revision(), revision);
        document.set_selected_enabled(false);
        assert!(!document.selected_region().enabled());
        assert_eq!(document.revision(), revision + 1);
        document.select(calm).unwrap();
        document.set_selected_enabled(true);
        assert!(document.selected_region().enabled());
        assert_eq!(document.revision(), revision + 2);
        assert!(document.select(MotionRegionId::local(64).unwrap()).is_err());
    }

    #[test]
    fn motion_region_document_rejects_empty_names_and_exhausts_at_sixty_four() {
        let mut document = MotionRegionDocument::artist_defaults();
        assert!(document.rename_selected("   ".to_string()).is_err());

        for expected_id in 5..=64 {
            assert_eq!(document.new_variation().unwrap().get(), expected_id);
        }
        assert_eq!(document.regions().len(), MAX_LOCAL_CONTROLLERS);
        assert!(document.new_variation().is_err());
    }

    #[test]
    fn controller_palette_same_document_seed_and_deltas_reproduce_the_same_trace() {
        let graph = branching_graph();
        let mut left_document = MotionRegionDocument::artist_defaults();
        let mut right_document = MotionRegionDocument::artist_defaults();
        let root = MotionRegionId::local(3).unwrap();
        configure_gusty_root(&mut left_document, root);
        configure_gusty_root(&mut right_document, root);
        let mut left = MotionControllerPalette::new(2.0).unwrap();
        let mut right = MotionControllerPalette::new(2.0).unwrap();
        left.sync_root_regions(&left_document).unwrap();
        right.sync_root_regions(&right_document).unwrap();

        let mut left_trace = Vec::new();
        let mut right_trace = Vec::new();
        for delta in [0.01, 0.03, 0.05, 0.02].into_iter().cycle().take(160) {
            left_trace.push(left.advance(delta, true, &graph).unwrap()[0].sample);
            right_trace.push(right.advance(delta, true, &graph).unwrap()[0].sample);
        }

        assert_eq!(left_trace, right_trace);
    }

    #[test]
    fn controller_palette_new_variation_has_an_independent_deterministic_trace() {
        let graph = branching_graph();
        let mut document = MotionRegionDocument::artist_defaults();
        let root = MotionRegionId::local(3).unwrap();
        configure_gusty_root(&mut document, root);
        document.select(root).unwrap();
        let variation = document.new_variation().unwrap();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&document).unwrap();
        let root_slot = palette.slot_for_region(root).unwrap();
        let variation_slot = palette.slot_for_region(variation).unwrap();

        let mut observed_difference = false;
        for _ in 0..240 {
            let frames = palette.advance(0.03, true, &graph).unwrap();
            let root_sample = frames
                .iter()
                .find(|frame| frame.slot == root_slot)
                .unwrap()
                .sample;
            let variation_sample = frames
                .iter()
                .find(|frame| frame.slot == variation_slot)
                .unwrap()
                .sample;
            observed_difference |= root_sample != variation_sample;
        }

        assert_ne!(root_slot, variation_slot);
        assert!(observed_difference);
    }

    #[test]
    fn controller_palette_excludes_disabled_regions_and_never_mutates_graph_edges() {
        let graph = branching_graph();
        let graph_before = graph.clone();
        let mut document = MotionRegionDocument::artist_defaults();
        let disabled = MotionRegionId::local(2).unwrap();
        document.select(disabled).unwrap();
        document.set_selected_enabled(false);
        let mut palette = MotionControllerPalette::new(2.0).unwrap();

        palette.sync_root_regions(&document).unwrap();
        for _ in 0..100 {
            palette.advance(0.04, true, &graph).unwrap();
        }

        assert_eq!(palette.active_local_count(), 3 * VARIANTS_PER_REGION);
        assert_eq!(palette.slot_for_region(disabled), None);
        assert_eq!(graph, graph_before);
        for frame in palette.frames() {
            let selected = palette
                .trace_for_slot(frame.slot)
                .and_then(|trace| trace.last_selected_edge);
            if let Some((source, target)) = selected {
                assert!(
                    graph.nodes()[source]
                        .jumps
                        .iter()
                        .any(|jump| jump.target_segment == target)
                );
            }
        }
    }

    #[test]
    fn controller_palette_is_bounded_to_sixty_four_local_states() {
        let mut document = MotionRegionDocument::artist_defaults();
        while document.regions().len() < MAX_LOCAL_CONTROLLERS {
            document.new_variation().unwrap();
        }
        let mut palette = MotionControllerPalette::new(2.0).unwrap();

        palette.sync_root_regions(&document).unwrap();

        assert_eq!(palette.active_local_count(), MAX_LOCAL_CONTROLLERS);
        assert_eq!(palette.frames().len(), MAX_LOCAL_CONTROLLERS);
        assert!(
            palette
                .frames()
                .iter()
                .all(|frame| (1..=64).contains(&frame.slot))
        );
    }

    #[test]
    fn controller_palette_root_advances_without_visible_field_coverage() {
        let graph = branching_graph();
        let mut document = MotionRegionDocument::artist_defaults();
        let root = MotionRegionId::local(3).unwrap();
        configure_gusty_root(&mut document, root);
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&document).unwrap();
        let before = palette.frames()[0].sample;

        for _ in 0..10 {
            palette.advance(0.05, true, &graph).unwrap();
        }

        assert_ne!(palette.frames()[0].sample, before);
    }

    fn two_tile_cache_with_region(region_id: MotionRegionId) -> MotionFieldCache {
        let layout = MotionFieldLayout::new([2, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut cache = MotionFieldCache::new(layout).unwrap();
        let stroke = MotionBrushStroke::new(
            1,
            MotionBrushTool::ApplyBehavior { region_id },
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        cache.apply_stroke(&stroke).unwrap();
        cache
    }

    #[test]
    fn compiled_field_maps_stable_region_ids_to_runtime_slots_without_mutating_source() {
        let calm = MotionRegionId::local(4).unwrap();
        let cache = two_tile_cache_with_region(calm);
        let mut regions = MotionRegionDocument::artist_defaults();
        for region in &mut regions.regions {
            region.enabled = region.id == calm;
        }
        regions.revision += 1;
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();

        let compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();

        assert_eq!(palette.slot_for_region(calm), Some(1));
        assert!(cache.assignments().contains(&4));
        assert!(compiled.assignments().contains(&1));
        assert!(!compiled.assignments().contains(&4));
        assert!(compiled.tile_has_local_motion([0, 0]));
        assert!(!compiled.tile_has_local_motion([1, 0]));
        assert_eq!(compiled.palette_count(), VARIANTS_PER_REGION as u32);
    }

    #[test]
    fn compiled_field_revision_is_stable_and_disabled_regions_fall_back_to_global() {
        let calm = MotionRegionId::local(4).unwrap();
        let cache = two_tile_cache_with_region(calm);
        let mut regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let first = CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let unchanged =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, Some(&first)).unwrap();
        assert_eq!(unchanged.revision(), first.revision());
        assert!(unchanged.dirty_regions().is_empty());

        regions.select(calm).unwrap();
        regions.set_selected_enabled(false);
        palette.sync_root_regions(&regions).unwrap();
        let disabled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, Some(&unchanged))
                .unwrap();

        assert_eq!(disabled.revision(), first.revision() + 1);
        assert!(disabled.assignments().iter().all(|&slot| slot == 0));
        assert!(!disabled.tile_has_local_motion([0, 0]));
        assert_eq!(disabled.dirty_regions().len(), 1);
    }

    #[test]
    fn compiled_field_assignment_edits_produce_a_bounded_dirty_rectangle() {
        let calm = MotionRegionId::local(4).unwrap();
        let mut cache = two_tile_cache_with_region(calm);
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let first = CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let stroke = MotionBrushStroke::new(
            2,
            MotionBrushTool::ApplyBehavior { region_id: calm },
            vec![[1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        cache.apply_stroke(&stroke).unwrap();

        let changed =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, Some(&first)).unwrap();
        let dirty = changed.dirty_regions()[0];
        let dirty_area = (dirty.max_exclusive()[0] - dirty.min()[0])
            * (dirty.max_exclusive()[1] - dirty.min()[1]);

        assert!(dirty_area < 16 * 8);
        assert_eq!(changed.revision(), first.revision() + 1);
    }

    #[test]
    fn compiled_field_incremental_sync_skips_unchanged_and_strength_only_plan_edits() {
        let calm = MotionRegionId::local(4).unwrap();
        let mut cache = two_tile_cache_with_region(calm);
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let revision = compiled.revision();

        assert!(
            !compiled
                .sync_roots(&cache, &regions, &palette, &[])
                .unwrap()
        );
        assert_eq!(compiled.revision(), revision);

        let strength = MotionBrushStroke::new(
            2,
            MotionBrushTool::AdjustStrength { target: 2.0 },
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        let dirty = cache.apply_stroke(&strength).unwrap().unwrap();
        assert!(
            !compiled
                .sync_roots(&cache, &regions, &palette, &[dirty])
                .unwrap()
        );
        assert_eq!(compiled.revision(), revision);
        assert_eq!(compiled.active_slot_at_world(&cache, [-1.0, 2.0]), Some(4));
    }

    #[test]
    fn compiled_field_incremental_sync_updates_coverage_when_erase_crosses_zero() {
        let calm = MotionRegionId::local(4).unwrap();
        let mut cache = two_tile_cache_with_region(calm);
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let revision = compiled.revision();
        let erase = MotionBrushStroke::new(
            2,
            MotionBrushTool::Erase,
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        let dirty = cache.apply_stroke(&erase).unwrap().unwrap();

        assert!(
            compiled
                .sync_roots(&cache, &regions, &palette, &[dirty])
                .unwrap()
        );
        assert_eq!(compiled.revision(), revision.wrapping_add(1));
        assert!(!compiled.tile_has_local_motion([0, 0]));
        assert_eq!(compiled.active_slot_at_world(&cache, [-1.0, 2.0]), None);
        assert!(compiled.dirty_regions()[0].max_exclusive()[0] <= 8);
    }

    #[test]
    fn membership_revision_changes_only_when_spatial_membership_changes() {
        let calm = MotionRegionId::local(4).unwrap();
        let steady = MotionRegionId::local(2).unwrap();
        let mut cache = two_tile_cache_with_region(calm);
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let membership_revision = compiled.membership_revision();

        assert!(
            !compiled
                .sync_roots(&cache, &regions, &palette, &[])
                .unwrap()
        );
        assert_eq!(compiled.membership_revision(), membership_revision);

        let strength = MotionBrushStroke::new(
            2,
            MotionBrushTool::AdjustStrength { target: 0.5 },
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        let dirty = cache.apply_stroke(&strength).unwrap().unwrap();
        compiled
            .sync_roots(&cache, &regions, &palette, &[dirty])
            .unwrap();
        assert_eq!(compiled.membership_revision(), membership_revision);

        let reassign = MotionBrushStroke::new(
            3,
            MotionBrushTool::ApplyBehavior { region_id: steady },
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        let dirty = cache.apply_stroke(&reassign).unwrap().unwrap();
        compiled
            .sync_roots(&cache, &regions, &palette, &[dirty])
            .unwrap();
        assert_ne!(compiled.revision(), membership_revision);
        assert_eq!(compiled.membership_revision(), membership_revision);

        let erase = MotionBrushStroke::new(
            4,
            MotionBrushTool::Erase,
            vec![[-1.0, 2.0]],
            0.6,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap();
        let dirty = cache.apply_stroke(&erase).unwrap().unwrap();
        compiled
            .sync_roots(&cache, &regions, &palette, &[dirty])
            .unwrap();
        assert_eq!(
            compiled.membership_revision(),
            membership_revision.wrapping_add(1)
        );
    }

    #[test]
    fn membership_snapshot_matches_compiled_lookup_and_half_open_boundaries() {
        let calm = MotionRegionId::local(4).unwrap();
        let cache = two_tile_cache_with_region(calm);
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let snapshot = compiled.membership_snapshot(&cache, 17).unwrap();

        assert_eq!(snapshot.request_revision(), 17);
        assert_eq!(
            snapshot.membership_revision(),
            compiled.membership_revision()
        );
        assert!(snapshot.tile_has_authored([0, 0]));
        assert!(!snapshot.tile_has_authored([1, 0]));
        for world_xy in [
            [-4.0, 0.0],
            [-1.0, 2.0],
            [3.999_999, 2.0],
            [4.0, 2.0],
            [-4.000_001, 2.0],
            [f32::NAN, 2.0],
        ] {
            assert_eq!(
                snapshot.active_at_world(world_xy),
                compiled.active_slot_at_world(&cache, world_xy).is_some(),
                "lookup mismatch at {world_xy:?}"
            );
        }
    }

    #[test]
    fn compiled_field_recentering_shifts_coarse_tile_coverage_with_world_paint() {
        let calm = MotionRegionId::local(4).unwrap();
        let initial_layout =
            MotionFieldLayout::new([3, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let mut document = MotionBrushDocument::default();
        document
            .push_stroke(
                MotionBrushTool::ApplyBehavior { region_id: calm },
                vec![[2.0, 2.0]],
                0.6,
                1.0,
                0.25,
                MotionBrushFalloff::Constant,
            )
            .unwrap();
        let mut cache = MotionFieldCache::rebuild(initial_layout, &document).unwrap();
        let regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        assert!(compiled.tile_has_local_motion([1, 0]));

        let moved_layout =
            MotionFieldLayout::new([3, 1], 4.0, [1, 0], MotionFieldQuality::Low).unwrap();
        let dirty = cache.shift_window(moved_layout, &document).unwrap();
        assert!(
            compiled
                .sync_roots(&cache, &regions, &palette, &dirty)
                .unwrap()
        );

        assert!(compiled.tile_has_local_motion([0, 0]));
        assert!(!compiled.tile_has_local_motion([1, 0]));
        assert_eq!(compiled.active_slot_at_world(&cache, [2.0, 2.0]), Some(4));
    }

    #[test]
    fn compiled_field_palette_remap_forces_a_complete_sync_without_cache_dirtiness() {
        let calm = MotionRegionId::local(4).unwrap();
        let cache = two_tile_cache_with_region(calm);
        let mut regions = MotionRegionDocument::artist_defaults();
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let mut compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();

        regions.select(calm).unwrap();
        regions.set_selected_enabled(false);
        palette.sync_root_regions(&regions).unwrap();

        assert!(
            compiled
                .sync_roots(&cache, &regions, &palette, &[])
                .unwrap()
        );
        assert!(compiled.assignments().iter().all(|&slot| slot == 0));
        assert_eq!(compiled.dirty_regions()[0].min(), [0, 0]);
        assert_eq!(
            compiled.dirty_regions()[0].max_exclusive(),
            cache.layout().texture_size()
        );
    }
}
