use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

use crate::motion_controller_palette::MotionRegionId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionFieldQuality {
    Low,
    #[default]
    Default,
    High,
}

impl MotionFieldQuality {
    pub const fn texels_per_tile(self) -> u32 {
        match self {
            Self::Low => 8,
            Self::Default => 16,
            Self::High => 32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionFieldLayout {
    map_tiles: [u32; 2],
    tile_width: f32,
    center_tile: [i32; 2],
    quality: MotionFieldQuality,
    texture_size: [u32; 2],
    world_origin: [f32; 2],
    world_size: [f32; 2],
    estimated_gpu_bytes: u64,
}

impl MotionFieldLayout {
    pub fn new(
        map_tiles: [u32; 2],
        tile_width: f32,
        center_tile: [i32; 2],
        quality: MotionFieldQuality,
    ) -> Result<Self, MotionBrushError> {
        if map_tiles.into_iter().any(|dimension| dimension == 0) {
            return Err(MotionBrushError::new(
                "motion field map dimensions must be positive",
            ));
        }
        if !tile_width.is_finite() || tile_width <= 0.0 {
            return Err(MotionBrushError::new(
                "motion field tile width must be finite and positive",
            ));
        }
        let texels_per_tile = quality.texels_per_tile();
        let texture_size = [
            map_tiles[0]
                .checked_mul(texels_per_tile)
                .ok_or_else(|| MotionBrushError::new("motion field width overflow"))?,
            map_tiles[1]
                .checked_mul(texels_per_tile)
                .ok_or_else(|| MotionBrushError::new("motion field height overflow"))?,
        ];
        let texel_count = u64::from(texture_size[0])
            .checked_mul(u64::from(texture_size[1]))
            .ok_or_else(|| MotionBrushError::new("motion field texel count overflow"))?;
        let estimated_gpu_bytes = texel_count
            .checked_mul(5)
            .ok_or_else(|| MotionBrushError::new("motion field byte count overflow"))?;
        let world_size = [
            map_tiles[0] as f32 * tile_width,
            map_tiles[1] as f32 * tile_width,
        ];
        if world_size.into_iter().any(|extent| !extent.is_finite()) {
            return Err(MotionBrushError::new(
                "motion field world extent is not finite",
            ));
        }
        let half_tiles = [(map_tiles[0] / 2) as i64, (map_tiles[1] / 2) as i64];
        let world_origin = [
            (i64::from(center_tile[0]) - half_tiles[0]) as f32 * tile_width,
            (i64::from(center_tile[1]) - half_tiles[1]) as f32 * tile_width,
        ];
        if world_origin.into_iter().any(|value| !value.is_finite()) {
            return Err(MotionBrushError::new(
                "motion field world origin is not finite",
            ));
        }
        Ok(Self {
            map_tiles,
            tile_width,
            center_tile,
            quality,
            texture_size,
            world_origin,
            world_size,
            estimated_gpu_bytes,
        })
    }

    pub fn map_tiles(&self) -> [u32; 2] {
        self.map_tiles
    }

    pub fn tile_width(&self) -> f32 {
        self.tile_width
    }

    pub fn center_tile(&self) -> [i32; 2] {
        self.center_tile
    }

    pub fn quality(&self) -> MotionFieldQuality {
        self.quality
    }

    pub fn texture_size(&self) -> [u32; 2] {
        self.texture_size
    }

    pub fn world_origin(&self) -> [f32; 2] {
        self.world_origin
    }

    pub fn world_size(&self) -> [f32; 2] {
        self.world_size
    }

    pub fn texel_world_size(&self) -> [f32; 2] {
        [
            self.world_size[0] / self.texture_size[0] as f32,
            self.world_size[1] / self.texture_size[1] as f32,
        ]
    }

    pub fn estimated_gpu_bytes(&self) -> u64 {
        self.estimated_gpu_bytes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionBrushFalloff {
    Constant,
    Linear,
    #[default]
    Smooth,
}

impl MotionBrushFalloff {
    pub const fn shader_code(self) -> u32 {
        match self {
            Self::Constant => 0,
            Self::Linear => 1,
            Self::Smooth => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotionBrushTool {
    ApplyBehavior { region_id: MotionRegionId },
    AdjustStrength { target: f32 },
    AdjustVariation { target: f32 },
    AdjustCoherence { target: f32 },
    Synchronize { region_id: MotionRegionId },
    Smooth,
    Erase,
}

impl MotionBrushTool {
    fn validate(self) -> Result<(), MotionBrushError> {
        match self {
            Self::AdjustStrength { target }
                if !target.is_finite() || !(0.0..=2.0).contains(&target) =>
            {
                Err(MotionBrushError::new(
                    "motion strength target must be finite and within 0..=2",
                ))
            }
            Self::AdjustVariation { target } | Self::AdjustCoherence { target }
                if !target.is_finite() || !(0.0..=1.0).contains(&target) =>
            {
                Err(MotionBrushError::new(
                    "motion scalar target must be finite and within 0..=1",
                ))
            }
            _ => Ok(()),
        }
    }

    pub const fn preview_color(self) -> [f32; 4] {
        match self {
            Self::ApplyBehavior { .. } => [0.18, 0.55, 1.0, 1.0],
            Self::AdjustStrength { .. } => [0.05, 0.88, 1.0, 1.0],
            Self::AdjustVariation { .. } => [0.12, 0.72, 1.0, 1.0],
            Self::AdjustCoherence { .. } => [0.16, 0.90, 0.46, 1.0],
            Self::Synchronize { .. } => [0.72, 0.38, 1.0, 1.0],
            Self::Smooth => [1.0, 0.78, 0.18, 1.0],
            Self::Erase => [1.0, 0.24, 0.20, 1.0],
        }
    }

    pub const fn preview_code(self) -> u32 {
        match self {
            Self::ApplyBehavior { .. } => 1,
            Self::AdjustStrength { .. } => 2,
            Self::AdjustVariation { .. } => 3,
            Self::AdjustCoherence { .. } => 4,
            Self::Synchronize { .. } => 5,
            Self::Smooth => 6,
            Self::Erase => 7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBrushPreview {
    pub center: [f32; 2],
    pub radius: f32,
    pub opacity: f32,
    pub falloff: MotionBrushFalloff,
    pub tool: MotionBrushTool,
    pub color: [f32; 4],
}

impl MotionBrushPreview {
    pub fn influence_at(self, world_xy: [f32; 2]) -> f32 {
        if !self.radius.is_finite()
            || self.radius <= 0.0
            || world_xy.into_iter().any(|value| !value.is_finite())
        {
            return 0.0;
        }
        let delta_x = world_xy[0] - self.center[0];
        let delta_y = world_xy[1] - self.center[1];
        let normalized_distance = (delta_x * delta_x + delta_y * delta_y).sqrt() / self.radius;
        if !normalized_distance.is_finite() || normalized_distance > 1.0 {
            return 0.0;
        }
        brush_weight(self.falloff, normalized_distance) * self.opacity
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionBrushStroke {
    id: u64,
    pub tool: MotionBrushTool,
    pub path: Vec<[f32; 2]>,
    pub radius: f32,
    pub opacity: f32,
    pub spacing: f32,
    pub falloff: MotionBrushFalloff,
    #[serde(default)]
    pub fallback_region: Option<MotionRegionId>,
    /// A document-order removal event; its geometry is ignored during replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub removed_region: Option<MotionRegionId>,
}

impl MotionBrushStroke {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        tool: MotionBrushTool,
        path: Vec<[f32; 2]>,
        radius: f32,
        opacity: f32,
        spacing: f32,
        falloff: MotionBrushFalloff,
    ) -> Result<Self, MotionBrushError> {
        if id == 0 {
            return Err(MotionBrushError::new("motion stroke ID must be positive"));
        }
        tool.validate()?;
        if path.is_empty()
            || path
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
        {
            return Err(MotionBrushError::new(
                "motion stroke path must contain finite positions",
            ));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(MotionBrushError::new(
                "motion stroke radius must be finite and positive",
            ));
        }
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(MotionBrushError::new(
                "motion stroke opacity must be finite and within 0..=1",
            ));
        }
        if !spacing.is_finite() || !(0.01..=1.0).contains(&spacing) {
            return Err(MotionBrushError::new(
                "motion stroke spacing must be finite and within 0.01..=1",
            ));
        }
        Ok(Self {
            id,
            tool,
            path,
            radius,
            opacity,
            spacing,
            falloff,
            fallback_region: None,
            removed_region: None,
        })
    }

    pub fn id(&self) -> u64 {
        self.id
    }
}

#[derive(Clone, Debug)]
pub struct MotionBrushDocument {
    strokes: Vec<MotionBrushStroke>,
    next_id: u64,
}

impl Default for MotionBrushDocument {
    fn default() -> Self {
        Self {
            strokes: Vec::new(),
            next_id: 1,
        }
    }
}

impl MotionBrushDocument {
    pub fn from_strokes(strokes: Vec<MotionBrushStroke>) -> Result<Self, MotionBrushError> {
        let mut last = 0;
        for s in &strokes {
            if let Some(id) = s.removed_region {
                MotionRegionId::local(id.get())
                    .map_err(|e| MotionBrushError::new(e.to_string()))?;
                if s.tool != MotionBrushTool::Erase || s.fallback_region.is_some() {
                    return Err(MotionBrushError::new("invalid controller removal event"));
                }
            }
            if s.id <= last {
                return Err(MotionBrushError::new(
                    "stroke IDs must be strictly increasing",
                ));
            }
            MotionBrushStroke::new(
                s.id,
                s.tool,
                s.path.clone(),
                s.radius,
                s.opacity,
                s.spacing,
                s.falloff,
            )?;
            if let Some(id) = s.fallback_region {
                MotionRegionId::local(id.get())
                    .map_err(|e| MotionBrushError::new(e.to_string()))?;
            }
            last = s.id;
        }
        Ok(Self {
            strokes,
            next_id: last
                .checked_add(1)
                .ok_or_else(|| MotionBrushError::new("stroke ID overflow"))?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn push_stroke(
        &mut self,
        tool: MotionBrushTool,
        path: Vec<[f32; 2]>,
        radius: f32,
        opacity: f32,
        spacing: f32,
        falloff: MotionBrushFalloff,
    ) -> Result<u64, MotionBrushError> {
        let id = self.next_id;
        let stroke = MotionBrushStroke::new(id, tool, path, radius, opacity, spacing, falloff)?;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| MotionBrushError::new("motion stroke ID overflow"))?;
        self.strokes.push(stroke);
        Ok(id)
    }

    pub fn strokes(&self) -> &[MotionBrushStroke] {
        &self.strokes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MotionOverlayChannel {
    #[default]
    Off = 0,
    Influence = 1,
    Strength = 2,
    Variation = 3,
    Coherence = 4,
    Assignment = 5,
}

pub fn motion_brush_accepts_pointer(
    brush_enabled: bool,
    gui_wants_pointer: bool,
    primary_button: bool,
) -> bool {
    brush_enabled && !gui_wants_pointer && primary_button
}

pub fn motion_brush_drag_should_end(
    brush_enabled: bool,
    gui_wants_pointer: bool,
    has_world_hit: bool,
) -> bool {
    !brush_enabled || gui_wants_pointer || !has_world_hit
}

pub fn motion_brush_preview_hover(
    brush_enabled: bool,
    gui_wants_pointer: bool,
    world_xy: Option<[f32; 2]>,
) -> Option<[f32; 2]> {
    if brush_enabled && !gui_wants_pointer {
        world_xy
    } else {
        None
    }
}

pub fn canonical_brush_xy(
    rendered_xy: [f32; 2],
    scene_scale_xy: [f32; 2],
) -> Result<[f32; 2], MotionBrushError> {
    if rendered_xy.into_iter().any(|value| !value.is_finite())
        || scene_scale_xy
            .into_iter()
            .any(|value| !value.is_finite() || value.abs() <= f32::EPSILON)
    {
        return Err(MotionBrushError::new(
            "motion brush coordinates require finite, nonzero scene XY scale",
        ));
    }
    Ok([
        rendered_xy[0] / scene_scale_xy[0],
        rendered_xy[1] / scene_scale_xy[1],
    ])
}

pub fn motion_overlay_color(channel: MotionOverlayChannel, value: u8) -> [f32; 3] {
    let unit = f32::from(value) / 255.0;
    match channel {
        MotionOverlayChannel::Off => [0.0, 0.0, 0.0],
        MotionOverlayChannel::Influence => {
            [0.02 + 0.83 * unit, 0.04 + 0.96 * unit, 0.08 + 0.92 * unit]
        }
        MotionOverlayChannel::Strength => {
            let centered = unit * 2.0 - 1.0;
            if centered < 0.0 {
                [
                    0.2 + 0.8 * (1.0 + centered),
                    0.4 + 0.6 * (1.0 + centered),
                    1.0,
                ]
            } else {
                [1.0, 1.0 - 0.7 * centered, 1.0 - 0.8 * centered]
            }
        }
        MotionOverlayChannel::Variation => [unit, 0.35 * (1.0 - unit), 0.9],
        MotionOverlayChannel::Coherence => [0.9 * (1.0 - unit), unit, 0.2],
        MotionOverlayChannel::Assignment if value == 0 => [0.2, 0.2, 0.2],
        MotionOverlayChannel::Assignment => {
            let slot = u32::from(value);
            [
                f32::from(((slot * 97 + 53) & 255) as u8) / 255.0,
                f32::from(((slot * 57 + 131) & 255) as u8) / 255.0,
                f32::from(((slot * 23 + 211) & 255) as u8) / 255.0,
            ]
        }
    }
}

const DEFAULT_CONTINUOUS_TEXEL: [u8; 4] = [0, 128, 0, 128];
const MAX_BRUSH_STAMPS_PER_STROKE: usize = 131_072;
const MAX_PREVIEW_PATH_POINTS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionFieldTexel {
    pub continuous: [u8; 4],
    pub assignment: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyRect {
    min: [u32; 2],
    max_exclusive: [u32; 2],
}

impl DirtyRect {
    fn new(min: [u32; 2], max_exclusive: [u32; 2]) -> Option<Self> {
        (min[0] < max_exclusive[0] && min[1] < max_exclusive[1])
            .then_some(Self { min, max_exclusive })
    }

    pub fn from_bounds(min: [u32; 2], max_exclusive: [u32; 2]) -> Option<Self> {
        Self::new(min, max_exclusive)
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max_exclusive: [
                self.max_exclusive[0].max(other.max_exclusive[0]),
                self.max_exclusive[1].max(other.max_exclusive[1]),
            ],
        }
    }

    fn intersection(self, other: Self) -> Option<Self> {
        Self::new(
            [self.min[0].max(other.min[0]), self.min[1].max(other.min[1])],
            [
                self.max_exclusive[0].min(other.max_exclusive[0]),
                self.max_exclusive[1].min(other.max_exclusive[1]),
            ],
        )
    }

    fn expanded(self, amount: u32, bounds: [u32; 2]) -> Self {
        Self {
            min: [
                self.min[0].saturating_sub(amount),
                self.min[1].saturating_sub(amount),
            ],
            max_exclusive: [
                self.max_exclusive[0].saturating_add(amount).min(bounds[0]),
                self.max_exclusive[1].saturating_add(amount).min(bounds[1]),
            ],
        }
    }

    pub fn min(&self) -> [u32; 2] {
        self.min
    }

    pub fn max_exclusive(&self) -> [u32; 2] {
        self.max_exclusive
    }
}

#[derive(Clone, Debug)]
pub struct MotionFieldCache {
    layout: MotionFieldLayout,
    storage_offset: [u32; 2],
    continuous: Vec<u8>,
    assignments: Vec<u8>,
    painted_texel_count: usize,
    content_revision: u64,
}

impl MotionFieldCache {
    pub fn new(layout: MotionFieldLayout) -> Result<Self, MotionBrushError> {
        let [width, height] = layout.texture_size();
        let texel_count = usize::try_from(
            u64::from(width)
                .checked_mul(u64::from(height))
                .ok_or_else(|| MotionBrushError::new("motion field cache size overflow"))?,
        )
        .map_err(|_| MotionBrushError::new("motion field cache does not fit in memory"))?;
        let continuous_len = texel_count
            .checked_mul(4)
            .ok_or_else(|| MotionBrushError::new("motion field continuous cache size overflow"))?;
        let mut continuous = vec![0; continuous_len];
        for texel in continuous.chunks_exact_mut(4) {
            texel.copy_from_slice(&DEFAULT_CONTINUOUS_TEXEL);
        }
        Ok(Self {
            layout,
            storage_offset: [0; 2],
            continuous,
            assignments: vec![0; texel_count],
            painted_texel_count: 0,
            content_revision: 0,
        })
    }

    pub fn rebuild(
        layout: MotionFieldLayout,
        document: &MotionBrushDocument,
    ) -> Result<Self, MotionBrushError> {
        let mut cache = Self::new(layout)?;
        for stroke in document.strokes() {
            cache.apply_stroke(stroke)?;
        }
        Ok(cache)
    }

    pub fn shift_window(
        &mut self,
        layout: MotionFieldLayout,
        document: &MotionBrushDocument,
    ) -> Result<Vec<DirtyRect>, MotionBrushError> {
        let previous_revision = self.content_revision;
        let full = DirtyRect::from_bounds([0, 0], layout.texture_size())
            .expect("validated motion field dimensions are positive");
        if self.layout.texture_size() != layout.texture_size()
            || self.layout.quality() != layout.quality()
            || self.layout.map_tiles() != layout.map_tiles()
            || self.layout.tile_width() != layout.tile_width()
        {
            *self = Self::rebuild(layout, document)?;
            self.content_revision = previous_revision.wrapping_add(1);
            return Ok(vec![full]);
        }

        let old_layout = self.layout;
        let texels_per_tile = i64::from(layout.quality().texels_per_tile());
        let shift = [
            (i64::from(layout.center_tile()[0]) - i64::from(old_layout.center_tile()[0]))
                * texels_per_tile,
            (i64::from(layout.center_tile()[1]) - i64::from(old_layout.center_tile()[1]))
                * texels_per_tile,
        ];
        let [width, height] = layout.texture_size();
        if shift[0].unsigned_abs() >= u64::from(width)
            || shift[1].unsigned_abs() >= u64::from(height)
        {
            *self = Self::rebuild(layout, document)?;
            self.content_revision = previous_revision.wrapping_add(1);
            return Ok(vec![full]);
        }
        if document
            .strokes()
            .iter()
            .any(|stroke| matches!(stroke.tool, MotionBrushTool::Smooth))
        {
            // Smooth reads document-order neighbors. Until the stroke document owns a
            // sparse halo cache, rebuilding is the only boundary-independent result.
            *self = Self::rebuild(layout, document)?;
            self.content_revision = previous_revision.wrapping_add(1);
            return Ok(vec![full]);
        }

        let overlap_min = [(-shift[0]).max(0) as u32, (-shift[1]).max(0) as u32];
        let overlap_max = [
            (i64::from(width) - shift[0]).min(i64::from(width)) as u32,
            (i64::from(height) - shift[1]).min(i64::from(height)) as u32,
        ];
        self.storage_offset = [
            (i64::from(self.storage_offset[0]) + shift[0]).rem_euclid(i64::from(width)) as u32,
            (i64::from(self.storage_offset[1]) + shift[1]).rem_euclid(i64::from(height)) as u32,
        ];
        self.layout = layout;

        let mut exposed = Vec::with_capacity(4);
        if let Some(rect) = DirtyRect::new([0, 0], [overlap_min[0], height]) {
            exposed.push(rect);
        }
        if let Some(rect) = DirtyRect::new([overlap_max[0], 0], [width, height]) {
            exposed.push(rect);
        }
        if let Some(rect) = DirtyRect::new([overlap_min[0], 0], [overlap_max[0], overlap_min[1]]) {
            exposed.push(rect);
        }
        if let Some(rect) =
            DirtyRect::new([overlap_min[0], overlap_max[1]], [overlap_max[0], height])
        {
            exposed.push(rect);
        }

        for &rect in &exposed {
            self.clear_rect(rect);
        }
        for rect in &exposed {
            for stroke in document.strokes() {
                self.apply_stroke_in_rect(stroke, *rect)?;
            }
        }
        self.content_revision = previous_revision.wrapping_add(1);
        Ok(exposed)
    }

    pub fn layout(&self) -> MotionFieldLayout {
        self.layout
    }

    pub fn continuous_bytes(&self) -> &[u8] {
        &self.continuous
    }

    pub fn assignments(&self) -> &[u8] {
        &self.assignments
    }

    pub fn storage_offset(&self) -> [u32; 2] {
        self.storage_offset
    }

    pub const fn content_revision(&self) -> u64 {
        self.content_revision
    }

    pub fn physical_regions(&self, logical: DirtyRect) -> Vec<DirtyRect> {
        let size = self.layout.texture_size();
        let x_ranges = shifted_ranges(
            logical.min[0],
            logical.max_exclusive[0],
            self.storage_offset[0],
            size[0],
        );
        let y_ranges = shifted_ranges(
            logical.min[1],
            logical.max_exclusive[1],
            self.storage_offset[1],
            size[1],
        );
        let mut regions = Vec::with_capacity(x_ranges.len() * y_ranges.len());
        for &(min_x, max_x) in &x_ranges {
            for &(min_y, max_y) in &y_ranges {
                regions.push(
                    DirtyRect::new([min_x, min_y], [max_x, max_y])
                        .expect("shifted nonempty ranges remain nonempty"),
                );
            }
        }
        regions
    }

    pub fn painted_texel_count(&self) -> usize {
        self.painted_texel_count
    }

    pub fn texel(&self, x: u32, y: u32) -> Option<MotionFieldTexel> {
        let index = self.texel_index(x, y)?;
        let start = index * 4;
        Some(MotionFieldTexel {
            continuous: self.continuous[start..start + 4].try_into().ok()?,
            assignment: self.assignments[index],
        })
    }

    pub fn apply_stroke(
        &mut self,
        stroke: &MotionBrushStroke,
    ) -> Result<Option<DirtyRect>, MotionBrushError> {
        let full = DirtyRect::from_bounds([0, 0], self.layout.texture_size())
            .expect("validated motion field dimensions are positive");
        let dirty = self.apply_stroke_in_rect(stroke, full)?;
        if dirty.is_some() {
            self.content_revision = self.content_revision.wrapping_add(1);
        }
        Ok(dirty)
    }

    fn apply_stroke_in_rect(
        &mut self,
        stroke: &MotionBrushStroke,
        restrict: DirtyRect,
    ) -> Result<Option<DirtyRect>, MotionBrushError> {
        if let Some(id) = stroke.removed_region {
            let mut changed = false;
            for y in restrict.min[1]..restrict.max_exclusive[1] {
                for x in restrict.min[0]..restrict.max_exclusive[0] {
                    let index = self.texel_index(x, y).expect("clipped removal rectangle");
                    if self.assignments[index] == id.get() {
                        self.painted_texel_count -= usize::from(self.continuous[index * 4] != 0);
                        self.assignments[index] = 0;
                        self.continuous[index * 4..index * 4 + 4]
                            .copy_from_slice(&DEFAULT_CONTINUOUS_TEXEL);
                        changed = true;
                    }
                }
            }
            return Ok(changed.then_some(restrict));
        }
        let affected = self
            .stroke_bounds(stroke)
            .and_then(|rect| rect.intersection(restrict));
        let Some(affected) = affected else {
            return Ok(None);
        };
        // Validate and bound the complete stamp stream before touching cache bytes so a
        // rejected stroke is atomic and cannot leave a partially painted field behind.
        for_each_stroke_stamp(self.layout, stroke, Some(restrict), |_| {})?;
        let smooth_source = if matches!(stroke.tool, MotionBrushTool::Smooth) {
            Some(ContinuousSnapshot::capture(
                self,
                affected.expanded(1, self.layout.texture_size()),
            ))
        } else {
            None
        };
        let mut dirty = None;
        let layout = self.layout;
        for_each_stroke_stamp(layout, stroke, Some(restrict), |center| {
            let Some(rect) = self
                .stamp_rect(center, stroke.radius)
                .and_then(|rect| rect.intersection(restrict))
            else {
                return;
            };
            if self.apply_stamp(stroke, center, rect, smooth_source.as_ref()) {
                dirty = Some(dirty.map_or(rect, |accumulator: DirtyRect| accumulator.union(rect)));
            }
        })?;
        Ok(dirty)
    }

    fn stroke_bounds(&self, stroke: &MotionBrushStroke) -> Option<DirtyRect> {
        let mut min_world = [f32::INFINITY; 2];
        let mut max_world = [f32::NEG_INFINITY; 2];
        for point in &stroke.path {
            for axis in 0..2 {
                min_world[axis] = min_world[axis].min(point[axis] - stroke.radius);
                max_world[axis] = max_world[axis].max(point[axis] + stroke.radius);
            }
        }
        let origin = self.layout.world_origin();
        let texel = self.layout.texel_world_size();
        let size = self.layout.texture_size();
        let mut min = [0; 2];
        let mut max = [0; 2];
        for axis in 0..2 {
            min[axis] = ((min_world[axis] - origin[axis]) / texel[axis])
                .floor()
                .clamp(0.0, size[axis] as f32) as u32;
            max[axis] = ((max_world[axis] - origin[axis]) / texel[axis])
                .ceil()
                .clamp(0.0, size[axis] as f32) as u32;
        }
        DirtyRect::new(min, max)
    }

    fn texel_index(&self, x: u32, y: u32) -> Option<usize> {
        let [width, height] = self.layout.texture_size();
        if x >= width || y >= height {
            return None;
        }
        let physical_x =
            ((u64::from(x) + u64::from(self.storage_offset[0])) % u64::from(width)) as u32;
        let physical_y =
            ((u64::from(y) + u64::from(self.storage_offset[1])) % u64::from(height)) as u32;
        usize::try_from(u64::from(physical_y) * u64::from(width) + u64::from(physical_x)).ok()
    }

    fn clear_rect(&mut self, rect: DirtyRect) -> bool {
        let mut changed = false;
        for y in rect.min[1]..rect.max_exclusive[1] {
            for x in rect.min[0]..rect.max_exclusive[0] {
                let index = self
                    .texel_index(x, y)
                    .expect("clipped clear rectangle addresses valid texels");
                let start = index * 4;
                changed |= self.continuous[start..start + 4] != DEFAULT_CONTINUOUS_TEXEL
                    || self.assignments[index] != 0;
                self.painted_texel_count -= usize::from(self.continuous[start] != 0);
                self.continuous[start..start + 4].copy_from_slice(&DEFAULT_CONTINUOUS_TEXEL);
                self.assignments[index] = 0;
            }
        }
        changed
    }

    fn stamp_rect(&self, center: [f32; 2], radius: f32) -> Option<DirtyRect> {
        let origin = self.layout.world_origin();
        let texel_size = self.layout.texel_world_size();
        let texture_size = self.layout.texture_size();
        let mut min = [0; 2];
        let mut max_exclusive = [0; 2];
        for axis in 0..2 {
            let raw_min = ((center[axis] - radius - origin[axis]) / texel_size[axis]).floor();
            let raw_max = ((center[axis] + radius - origin[axis]) / texel_size[axis]).ceil();
            min[axis] = raw_min.clamp(0.0, texture_size[axis] as f32) as u32;
            max_exclusive[axis] = raw_max.clamp(0.0, texture_size[axis] as f32) as u32;
        }
        DirtyRect::new(min, max_exclusive)
    }

    fn apply_stamp(
        &mut self,
        stroke: &MotionBrushStroke,
        center: [f32; 2],
        rect: DirtyRect,
        smooth_source: Option<&ContinuousSnapshot>,
    ) -> bool {
        let origin = self.layout.world_origin();
        let texel_size = self.layout.texel_world_size();
        let mut changed = false;
        for y in rect.min[1]..rect.max_exclusive[1] {
            for x in rect.min[0]..rect.max_exclusive[0] {
                let world = [
                    origin[0] + (x as f32 + 0.5) * texel_size[0],
                    origin[1] + (y as f32 + 0.5) * texel_size[1],
                ];
                let distance =
                    ((world[0] - center[0]).powi(2) + (world[1] - center[1]).powi(2)).sqrt();
                if distance > stroke.radius {
                    continue;
                }
                let normalized_distance = distance / stroke.radius;
                let weight = brush_weight(stroke.falloff, normalized_distance) * stroke.opacity;
                if weight <= 0.0 {
                    continue;
                }
                let index = self
                    .texel_index(x, y)
                    .expect("clipped stamp rectangle must address valid texels");
                if self.assignments[index] == 0
                    && let Some(id) = stroke.fallback_region
                    && matches!(
                        stroke.tool,
                        MotionBrushTool::AdjustStrength { .. }
                            | MotionBrushTool::AdjustVariation { .. }
                            | MotionBrushTool::AdjustCoherence { .. }
                    )
                {
                    self.assignments[index] = id.get();
                    // Coherence needs variation to group on a fresh field. Strength
                    // alone must not introduce additional spatial randomness.
                    self.continuous[index * 4 + 2] =
                        if matches!(stroke.tool, MotionBrushTool::AdjustCoherence { .. }) {
                            128
                        } else {
                            0
                        };
                    self.continuous[index * 4 + 3] = 128;
                    changed = true;
                }
                changed |= self.apply_tool(index, x, y, stroke.tool, weight, smooth_source);
            }
        }
        changed
    }

    fn apply_tool(
        &mut self,
        index: usize,
        x: u32,
        y: u32,
        tool: MotionBrushTool,
        weight: f32,
        smooth_source: Option<&ContinuousSnapshot>,
    ) -> bool {
        let channel_start = index * 4;
        let before_continuous: [u8; 4] = self.continuous[channel_start..channel_start + 4]
            .try_into()
            .expect("motion field continuous texels contain four bytes");
        let before_assignment = self.assignments[index];
        let was_painted = self.continuous[channel_start] != 0;
        match tool {
            MotionBrushTool::ApplyBehavior { region_id } => {
                self.assignments[index] = region_id.get();
                self.continuous[channel_start] =
                    lerp_byte(self.continuous[channel_start], u8::MAX, weight);
            }
            MotionBrushTool::AdjustStrength { target } => {
                let target = unit_byte(target * 0.5);
                self.continuous[channel_start + 1] =
                    lerp_byte(self.continuous[channel_start + 1], target, weight);
                self.continuous[channel_start] =
                    lerp_byte(self.continuous[channel_start], u8::MAX, weight);
            }
            MotionBrushTool::AdjustVariation { target } => {
                let target = unit_byte(target);
                self.continuous[channel_start + 2] =
                    lerp_byte(self.continuous[channel_start + 2], target, weight);
                self.continuous[channel_start] =
                    lerp_byte(self.continuous[channel_start], u8::MAX, weight);
            }
            MotionBrushTool::AdjustCoherence { target } => {
                let target = unit_byte(target);
                self.continuous[channel_start + 3] =
                    lerp_byte(self.continuous[channel_start + 3], target, weight);
                self.continuous[channel_start] =
                    lerp_byte(self.continuous[channel_start], u8::MAX, weight);
            }
            MotionBrushTool::Synchronize { region_id } => {
                self.assignments[index] = region_id.get();
                self.continuous[channel_start + 2] = 0;
                self.continuous[channel_start + 3] = 255;
                self.continuous[channel_start] =
                    lerp_byte(self.continuous[channel_start], 255, weight);
            }
            MotionBrushTool::Smooth => {
                let source = smooth_source.expect("smooth strokes require an immutable source");
                for channel in 0..4 {
                    let average = self.neighborhood_average(source, x, y, channel);
                    self.continuous[channel_start + channel] =
                        lerp_byte(source.get(x, y, channel), average, weight);
                }
            }
            MotionBrushTool::Erase => {
                let influence = lerp_byte(self.continuous[channel_start], 0, weight);
                self.continuous[channel_start] = influence;
                if influence == 0 {
                    self.assignments[index] = 0;
                }
            }
        }
        let is_painted = self.continuous[channel_start] != 0;
        match (was_painted, is_painted) {
            (false, true) => self.painted_texel_count += 1,
            (true, false) => self.painted_texel_count -= 1,
            _ => {}
        }
        before_assignment != self.assignments[index]
            || before_continuous != self.continuous[channel_start..channel_start + 4]
    }

    fn neighborhood_average(
        &self,
        source: &ContinuousSnapshot,
        x: u32,
        y: u32,
        channel: usize,
    ) -> u8 {
        let [width, height] = self.layout.texture_size();
        let min_x = x.saturating_sub(1);
        let min_y = y.saturating_sub(1);
        let max_x = x.saturating_add(1).min(width - 1);
        let max_y = y.saturating_add(1).min(height - 1);
        let mut total = 0_u32;
        let mut count = 0_u32;
        for neighbor_y in min_y..=max_y {
            for neighbor_x in min_x..=max_x {
                total += u32::from(source.get(neighbor_x, neighbor_y, channel));
                count += 1;
            }
        }
        ((total as f32 / count as f32).round()) as u8
    }
}

struct ContinuousSnapshot {
    rect: DirtyRect,
    bytes: Vec<u8>,
}

impl ContinuousSnapshot {
    fn capture(cache: &MotionFieldCache, rect: DirtyRect) -> Self {
        let width = (rect.max_exclusive[0] - rect.min[0]) as usize;
        let mut bytes =
            Vec::with_capacity(width * (rect.max_exclusive[1] - rect.min[1]) as usize * 4);
        for y in rect.min[1]..rect.max_exclusive[1] {
            for x in rect.min[0]..rect.max_exclusive[0] {
                let index = cache
                    .texel_index(x, y)
                    .expect("snapshot rectangle addresses valid logical texels");
                bytes.extend_from_slice(&cache.continuous[index * 4..index * 4 + 4]);
            }
        }
        Self { rect, bytes }
    }

    fn get(&self, x: u32, y: u32, channel: usize) -> u8 {
        debug_assert!(x >= self.rect.min[0] && x < self.rect.max_exclusive[0]);
        debug_assert!(y >= self.rect.min[1] && y < self.rect.max_exclusive[1]);
        let width = (self.rect.max_exclusive[0] - self.rect.min[0]) as usize;
        let index = ((y - self.rect.min[1]) as usize * width + (x - self.rect.min[0]) as usize) * 4
            + channel;
        self.bytes[index]
    }
}

fn shifted_ranges(min: u32, max_exclusive: u32, offset: u32, size: u32) -> Vec<(u32, u32)> {
    debug_assert!(min < max_exclusive && max_exclusive <= size && offset < size);
    let length = max_exclusive - min;
    if length == size {
        return vec![(0, size)];
    }
    let physical_min = ((u64::from(min) + u64::from(offset)) % u64::from(size)) as u32;
    let first_length = length.min(size - physical_min);
    let mut ranges = vec![(physical_min, physical_min + first_length)];
    if first_length < length {
        ranges.push((0, length - first_length));
    }
    ranges
}

#[derive(Clone, Debug)]
pub struct MotionBrushRuntime {
    pub enabled: bool,
    pub tool: MotionBrushTool,
    pub radius: f32,
    pub opacity: f32,
    pub spacing: f32,
    pub falloff: MotionBrushFalloff,
    pub overlay: MotionOverlayChannel,
    pub auto_overlay: bool,
    pub selected_region: MotionRegionId,
    hover: Option<[f32; 2]>,
    active_path: Vec<[f32; 2]>,
    document: MotionBrushDocument,
    cache: MotionFieldCache,
    dirty_regions: Vec<DirtyRect>,
    redo_strokes: Vec<MotionBrushStroke>,
}

impl MotionBrushRuntime {
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled && !self.enabled {
            self.overlay = MotionOverlayChannel::Assignment;
        }
        self.enabled = enabled;
    }

    pub fn follow_overlay(&mut self, channel: MotionOverlayChannel) {
        if self.enabled && self.auto_overlay {
            self.overlay = channel;
        }
    }

    pub fn remove_region(&mut self, id: MotionRegionId) -> Result<(), MotionBrushError> {
        self.end_stroke()?;
        self.document.push_stroke(
            MotionBrushTool::Erase,
            vec![[0.0, 0.0]],
            0.1,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )?;
        let event = self.document.strokes.last_mut().unwrap();
        event.removed_region = Some(id);
        if let Some(rect) = self.cache.apply_stroke(event)? {
            self.dirty_regions.push(rect);
        }
        self.redo_strokes.clear();
        Ok(())
    }

    pub fn last_removed_region(&self) -> Option<MotionRegionId> {
        self.document.strokes.last().and_then(|s| s.removed_region)
    }

    pub fn undo_region_removal(&mut self) -> Result<(), MotionBrushError> {
        if self.last_removed_region().is_some() {
            self.undo_last()?;
            // The registry must be restored alongside this event; ordinary Redo
            // must never replay a deletion without updating the registry.
            self.redo_strokes.clear();
        }
        Ok(())
    }
    pub fn new(layout: MotionFieldLayout) -> Result<Self, MotionBrushError> {
        Ok(Self {
            enabled: false,
            tool: MotionBrushTool::ApplyBehavior {
                region_id: MotionRegionId::local(1)
                    .expect("the default Motion Region ID must be valid"),
            },
            radius: 2.0,
            opacity: 1.0,
            spacing: 0.25,
            falloff: MotionBrushFalloff::Smooth,
            overlay: MotionOverlayChannel::Off,
            auto_overlay: true,
            selected_region: MotionRegionId::local(1).unwrap(),
            hover: None,
            active_path: Vec::new(),
            document: MotionBrushDocument::default(),
            cache: MotionFieldCache::new(layout)?,
            dirty_regions: Vec::new(),
            redo_strokes: Vec::new(),
        })
    }

    pub fn document(&self) -> &MotionBrushDocument {
        &self.document
    }

    pub fn cache(&self) -> &MotionFieldCache {
        &self.cache
    }

    pub fn layout(&self) -> MotionFieldLayout {
        self.cache.layout()
    }

    pub fn hover(&self) -> Option<[f32; 2]> {
        self.hover
    }

    pub fn set_hover(&mut self, world_xy: Option<[f32; 2]>) {
        self.hover = world_xy;
    }

    pub fn preview(&self) -> Option<MotionBrushPreview> {
        let center = self.enabled.then_some(self.hover).flatten()?;
        Some(MotionBrushPreview {
            center,
            radius: self.radius,
            opacity: self.opacity,
            falloff: self.falloff,
            tool: self.tool,
            color: self.tool.preview_color(),
        })
    }

    pub fn preview_path(&self) -> &[[f32; 2]] {
        let start = self
            .active_path
            .len()
            .saturating_sub(MAX_PREVIEW_PATH_POINTS);
        &self.active_path[start..]
    }

    pub fn begin_stroke(&mut self, world_xy: [f32; 2]) -> Result<(), MotionBrushError> {
        validate_world_position(world_xy)?;
        if !self.contains_world_xy(world_xy, self.radius) {
            return Err(MotionBrushError::new(
                "motion brush stroke is outside the active authoring field",
            ));
        }
        self.active_path.clear();
        self.active_path.push(world_xy);
        Ok(())
    }

    pub fn extend_stroke(&mut self, world_xy: [f32; 2]) -> Result<(), MotionBrushError> {
        validate_world_position(world_xy)?;
        if self.active_path.is_empty() {
            return Err(MotionBrushError::new(
                "cannot extend a motion stroke before it begins",
            ));
        }
        let last = *self
            .active_path
            .last()
            .expect("nonempty motion stroke has a final point");
        if last != world_xy && self.contains_world_xy(world_xy, self.radius) {
            self.active_path.push(world_xy);
        }
        Ok(())
    }

    pub fn end_stroke(&mut self) -> Result<Option<DirtyRect>, MotionBrushError> {
        if self.active_path.is_empty() {
            return Ok(None);
        }
        let path = std::mem::take(&mut self.active_path);
        self.document.push_stroke(
            self.tool,
            path,
            self.radius,
            self.opacity,
            self.spacing,
            self.falloff,
        )?;
        let stroke = self
            .document
            .strokes
            .last_mut()
            .expect("a successfully committed motion stroke must exist");
        stroke.fallback_region = Some(self.selected_region);
        let updated = match self.cache.apply_stroke(stroke) {
            Ok(updated) => updated,
            Err(error) => {
                self.document.strokes.pop();
                self.document.next_id -= 1;
                return Err(error);
            }
        };
        if let Some(rect) = updated {
            self.dirty_regions.push(rect);
        }
        self.redo_strokes.clear();
        Ok(updated)
    }

    pub fn replace_document(
        &mut self,
        document: MotionBrushDocument,
        quality: MotionFieldQuality,
    ) -> Result<(), MotionBrushError> {
        let old = self.layout();
        let layout = MotionFieldLayout::new(
            old.map_tiles(),
            old.tile_width(),
            old.center_tile(),
            quality,
        )?;
        let mut cache = MotionFieldCache::rebuild(layout, &document)?;
        cache.content_revision = self.cache.content_revision.wrapping_add(1);
        self.cache = cache;
        self.document = document;
        self.redo_strokes.clear();
        self.active_path.clear();
        self.dirty_regions = vec![DirtyRect::from_bounds([0, 0], layout.texture_size()).unwrap()];
        Ok(())
    }

    pub fn can_undo(&self) -> bool {
        self.document
            .strokes
            .last()
            .is_some_and(|s| s.removed_region.is_none())
    }
    pub fn can_redo(&self) -> bool {
        !self.redo_strokes.is_empty()
    }

    pub fn undo(&mut self) -> Result<(), MotionBrushError> {
        if !self.can_undo() {
            return Ok(());
        }
        self.undo_last()
    }

    fn undo_last(&mut self) -> Result<(), MotionBrushError> {
        let mut document = self.document.clone();
        let Some(stroke) = document.strokes.pop() else {
            return Ok(());
        };
        document.next_id = stroke.id;
        let mut redo = self.redo_strokes.clone();
        redo.push(stroke);
        self.replace_document(document, self.layout().quality())?;
        self.redo_strokes = redo;
        Ok(())
    }

    pub fn redo(&mut self) -> Result<(), MotionBrushError> {
        let mut redo = self.redo_strokes.clone();
        let Some(stroke) = redo.pop() else {
            return Ok(());
        };
        let mut document = self.document.clone();
        document.next_id = stroke.id + 1;
        document.strokes.push(stroke);
        self.replace_document(document, self.layout().quality())?;
        self.redo_strokes = redo;
        Ok(())
    }

    pub fn cancel_stroke(&mut self) {
        self.active_path.clear();
    }

    pub fn take_dirty_regions(&mut self) -> Vec<DirtyRect> {
        std::mem::take(&mut self.dirty_regions)
    }

    pub fn dirty_regions(&self) -> &[DirtyRect] {
        &self.dirty_regions
    }

    pub fn restore_dirty_regions(&mut self, mut regions: Vec<DirtyRect>) {
        regions.append(&mut self.dirty_regions);
        self.dirty_regions = regions;
    }

    pub fn shift_window(&mut self, layout: MotionFieldLayout) -> Result<(), MotionBrushError> {
        let dirty = self.cache.shift_window(layout, &self.document)?;
        self.dirty_regions.extend(dirty);
        Ok(())
    }

    pub fn set_quality(&mut self, quality: MotionFieldQuality) -> Result<(), MotionBrushError> {
        if self.layout().quality() == quality {
            return Ok(());
        }
        let layout = self.layout();
        self.shift_window(MotionFieldLayout::new(
            layout.map_tiles(),
            layout.tile_width(),
            layout.center_tile(),
            quality,
        )?)
    }

    pub fn recenter(&mut self, center_tile: [i32; 2]) -> Result<(), MotionBrushError> {
        if self.layout().center_tile() == center_tile {
            return Ok(());
        }
        let layout = self.layout();
        self.shift_window(MotionFieldLayout::new(
            layout.map_tiles(),
            layout.tile_width(),
            center_tile,
            layout.quality(),
        )?)
    }

    pub fn painted_texel_count(&self) -> usize {
        self.cache.painted_texel_count()
    }

    pub fn painted_fraction(&self) -> f32 {
        let total = self.cache.assignments().len();
        if total == 0 {
            0.0
        } else {
            self.painted_texel_count() as f32 / total as f32
        }
    }

    pub fn field_data_active(&self) -> bool {
        self.painted_texel_count() > 0 || self.overlay != MotionOverlayChannel::Off
    }

    pub fn render_path_active(&self) -> bool {
        self.field_data_active() || self.preview().is_some()
    }

    pub fn contains_world_xy(&self, world_xy: [f32; 2], margin: f32) -> bool {
        let origin = self.layout().world_origin();
        let size = self.layout().world_size();
        world_xy[0] >= origin[0] - margin
            && world_xy[0] <= origin[0] + size[0] + margin
            && world_xy[1] >= origin[1] - margin
            && world_xy[1] <= origin[1] + size[1] + margin
    }
}

fn validate_world_position(position: [f32; 2]) -> Result<(), MotionBrushError> {
    if position
        .into_iter()
        .any(|coordinate| !coordinate.is_finite())
    {
        Err(MotionBrushError::new(
            "motion brush world position must be finite",
        ))
    } else {
        Ok(())
    }
}

fn for_each_stroke_stamp(
    layout: MotionFieldLayout,
    stroke: &MotionBrushStroke,
    restrict: Option<DirtyRect>,
    mut visit: impl FnMut([f32; 2]),
) -> Result<(), MotionBrushError> {
    let texel = layout.texel_world_size();
    let spacing = (stroke.radius * stroke.spacing).max(0.5 * texel[0].min(texel[1]));
    let origin = layout.world_origin();
    let size = layout.world_size();
    let mut world_min = [origin[0] - stroke.radius, origin[1] - stroke.radius];
    let mut world_max = [
        origin[0] + size[0] + stroke.radius,
        origin[1] + size[1] + stroke.radius,
    ];
    if let Some(rect) = restrict {
        for axis in 0..2 {
            world_min[axis] = origin[axis] + rect.min[axis] as f32 * texel[axis] - stroke.radius;
            world_max[axis] =
                origin[axis] + rect.max_exclusive[axis] as f32 * texel[axis] + stroke.radius;
        }
    }

    let mut count = 0usize;
    let mut emit = |point: [f32; 2]| -> Result<(), MotionBrushError> {
        count = count
            .checked_add(1)
            .ok_or_else(|| MotionBrushError::new("motion brush stamp count overflow"))?;
        if count > MAX_BRUSH_STAMPS_PER_STROKE {
            return Err(MotionBrushError::new(format!(
                "motion brush stroke exceeds the {}-stamp safety limit",
                MAX_BRUSH_STAMPS_PER_STROKE
            )));
        }
        visit(point);
        Ok(())
    };

    if stroke.path.len() == 1 {
        let point = stroke.path[0];
        if point[0] >= world_min[0]
            && point[0] <= world_max[0]
            && point[1] >= world_min[1]
            && point[1] <= world_max[1]
        {
            emit(point)?;
        }
        return Ok(());
    }

    for segment in stroke.path.windows(2) {
        let Some((enter, exit)) =
            clip_segment_to_aabb(segment[0], segment[1], world_min, world_max)
        else {
            continue;
        };
        // The stamp grid belongs to the original segment, not to a cache/dirty
        // rectangle. Double precision keeps long off-screen segments stable.
        let start = segment[0].map(f64::from);
        let delta = [
            f64::from(segment[1][0]) - start[0],
            f64::from(segment[1][1]) - start[1],
        ];
        let distance = delta[0].hypot(delta[1]);
        if distance == 0.0 {
            emit(segment[0])?;
            continue;
        }
        let steps_f = (distance / f64::from(spacing)).ceil().max(1.0);
        if !steps_f.is_finite() || steps_f > (1u64 << 53) as f64 {
            return Err(MotionBrushError::new(
                "motion brush segment exceeds the rasterization safety limit",
            ));
        }
        // Include one neighboring stamp to absorb clipping roundoff. apply_stamp
        // clips its actual footprint, so this never paints outside the region.
        let first = (enter * steps_f).floor().max(0.0) as u64;
        let last = (exit * steps_f).ceil().min(steps_f) as u64;
        if last - first >= MAX_BRUSH_STAMPS_PER_STROKE as u64 {
            return Err(MotionBrushError::new(
                "motion brush segment exceeds the rasterization safety limit",
            ));
        }
        for step in first..=last {
            let fraction = step as f64 / steps_f;
            emit([
                (start[0] + delta[0] * fraction) as f32,
                (start[1] + delta[1] * fraction) as f32,
            ])?;
        }
    }
    Ok(())
}

fn clip_segment_to_aabb(
    start: [f32; 2],
    end: [f32; 2],
    min: [f32; 2],
    max: [f32; 2],
) -> Option<(f64, f64)> {
    let start = start.map(f64::from);
    let end = end.map(f64::from);
    let min = min.map(f64::from);
    let max = max.map(f64::from);
    let delta = [end[0] - start[0], end[1] - start[1]];
    let mut enter = 0.0f64;
    let mut exit = 1.0f64;
    for axis in 0..2 {
        if delta[axis] == 0.0 {
            if start[axis] < min[axis] || start[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let first = (min[axis] - start[axis]) / delta[axis];
        let second = (max[axis] - start[axis]) / delta[axis];
        enter = enter.max(first.min(second));
        exit = exit.min(first.max(second));
        if enter > exit {
            return None;
        }
    }
    Some((enter, exit))
}

fn brush_weight(falloff: MotionBrushFalloff, normalized_distance: f32) -> f32 {
    let distance = normalized_distance.clamp(0.0, 1.0);
    match falloff {
        MotionBrushFalloff::Constant => 1.0,
        MotionBrushFalloff::Linear => 1.0 - distance,
        MotionBrushFalloff::Smooth => {
            let distance_squared = distance * distance;
            1.0 - 3.0 * distance_squared + 2.0 * distance_squared * distance
        }
    }
}

fn unit_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn lerp_byte(from: u8, to: u8, weight: f32) -> u8 {
    (f32::from(from) + (f32::from(to) - f32::from(from)) * weight.clamp(0.0, 1.0))
        .round()
        .clamp(0.0, 255.0) as u8
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionBrushError {
    message: String,
}

impl MotionBrushError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for MotionBrushError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MotionBrushError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enabling_brush_defaults_to_controller_overlay_without_painting() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.set_enabled(true);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Assignment);
        assert!(runtime.document().strokes().is_empty());
    }

    #[test]
    fn automatic_overlay_respects_manual_override_and_enable_boundaries() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.set_enabled(true);
        runtime.follow_overlay(MotionOverlayChannel::Strength);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Strength);
        runtime.set_enabled(true);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Strength);
        runtime.auto_overlay = false;
        runtime.follow_overlay(MotionOverlayChannel::Variation);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Strength);
        runtime.set_enabled(false);
        runtime.auto_overlay = true;
        runtime.follow_overlay(MotionOverlayChannel::Coherence);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Strength);
        runtime.set_enabled(true);
        assert_eq!(runtime.overlay, MotionOverlayChannel::Assignment);
        assert!(runtime.document().strokes().is_empty());
    }
    use crate::motion_controller_palette::MotionRegionId;

    fn region_id(value: u8) -> MotionRegionId {
        MotionRegionId::local(value).unwrap()
    }

    #[test]
    fn quality_levels_are_exactly_tile_aligned_for_current_map() {
        let expected = [
            (MotionFieldQuality::Low, 776),
            (MotionFieldQuality::Default, 1552),
            (MotionFieldQuality::High, 3104),
        ];
        for (quality, dimension) in expected {
            let layout = MotionFieldLayout::new([97, 97], 4.0, [0, 0], quality).unwrap();
            assert_eq!(layout.texture_size(), [dimension, dimension]);
        }
    }

    #[test]
    fn default_layout_has_quarter_world_unit_texels_and_expected_memory() {
        let layout =
            MotionFieldLayout::new([97, 97], 4.0, [0, 0], MotionFieldQuality::Default).unwrap();

        assert_eq!(layout.world_origin(), [-192.0, -192.0]);
        assert_eq!(layout.world_size(), [388.0, 388.0]);
        assert_eq!(layout.texel_world_size(), [0.25, 0.25]);
        assert_eq!(layout.estimated_gpu_bytes(), 1552_u64 * 1552 * 5);
    }

    #[test]
    fn region_behavior_and_scalar_strokes_preserve_stable_region_identity() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(4),
                },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();
        let painted = cache.texel(8, 4).unwrap();
        assert_eq!(painted.assignment, 4);

        cache
            .apply_stroke(&stroke(
                MotionBrushTool::AdjustStrength { target: 1.5 },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();

        assert_eq!(cache.texel(8, 4).unwrap().assignment, 4);
    }

    #[test]
    fn moving_window_changes_absolute_origin_but_not_cache_dimensions() {
        let first =
            MotionFieldLayout::new([97, 97], 4.0, [0, 0], MotionFieldQuality::Default).unwrap();
        let moved =
            MotionFieldLayout::new([97, 97], 4.0, [3, -2], MotionFieldQuality::Default).unwrap();

        assert_eq!(first.texture_size(), moved.texture_size());
        assert_eq!(moved.world_origin(), [-180.0, -200.0]);
    }

    #[test]
    fn invalid_layouts_and_strokes_are_rejected() {
        assert!(
            MotionFieldLayout::new([0, 97], 4.0, [0, 0], MotionFieldQuality::Default,).is_err()
        );
        assert!(
            MotionFieldLayout::new([97, 97], f32::NAN, [0, 0], MotionFieldQuality::Default,)
                .is_err()
        );
        assert!(
            MotionBrushStroke::new(
                1,
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(1),
                },
                Vec::new(),
                2.0,
                1.0,
                0.25,
                MotionBrushFalloff::Smooth,
            )
            .is_err()
        );
        assert!(MotionRegionId::local(65).is_err());
    }

    #[test]
    fn document_assigns_monotonic_stroke_ids() {
        let mut document = MotionBrushDocument::default();
        let first = document
            .push_stroke(
                MotionBrushTool::Erase,
                vec![[1.0, 2.0]],
                1.0,
                0.5,
                0.25,
                MotionBrushFalloff::Linear,
            )
            .unwrap();
        let second = document
            .push_stroke(
                MotionBrushTool::AdjustVariation { target: 0.75 },
                vec![[2.0, 3.0]],
                1.5,
                0.7,
                0.25,
                MotionBrushFalloff::Smooth,
            )
            .unwrap();

        assert_eq!((first, second), (1, 2));
        assert_eq!(document.strokes()[0].id(), 1);
        assert_eq!(document.strokes()[1].id(), 2);
    }

    fn small_layout(center_tile: [i32; 2]) -> MotionFieldLayout {
        MotionFieldLayout::new([2, 1], 4.0, center_tile, MotionFieldQuality::Low).unwrap()
    }

    fn assert_logical_cache_eq(actual: &MotionFieldCache, expected: &MotionFieldCache) {
        assert_eq!(actual.layout(), expected.layout());
        let [width, height] = actual.layout().texture_size();
        for y in 0..height {
            for x in 0..width {
                assert_eq!(actual.texel(x, y), expected.texel(x, y), "texel ({x}, {y})");
            }
        }
        assert_eq!(actual.painted_texel_count(), expected.painted_texel_count());
    }

    fn stroke(tool: MotionBrushTool, path: Vec<[f32; 2]>, radius: f32) -> MotionBrushStroke {
        MotionBrushStroke::new(
            1,
            tool,
            path,
            radius,
            1.0,
            0.25,
            MotionBrushFalloff::Constant,
        )
        .unwrap()
    }

    #[test]
    fn behavior_stamp_crosses_a_tile_boundary_without_a_seam() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        let dirty = cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(3),
                },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap()
            .unwrap();

        assert_eq!(cache.texel(7, 4).unwrap().assignment, 3);
        assert_eq!(cache.texel(8, 4).unwrap().assignment, 3);
        assert!(dirty.min()[0] <= 7 && dirty.max_exclusive()[0] >= 9);
    }

    #[test]
    fn smooth_falloff_decreases_toward_the_brush_edge() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(
                &MotionBrushStroke::new(
                    1,
                    MotionBrushTool::ApplyBehavior {
                        region_id: region_id(3),
                    },
                    vec![[0.0, 2.0]],
                    1.1,
                    1.0,
                    0.25,
                    MotionBrushFalloff::Smooth,
                )
                .unwrap(),
            )
            .unwrap();

        let center = cache.texel(8, 4).unwrap().continuous[0];
        let edge = cache.texel(6, 4).unwrap().continuous[0];
        let outside = cache.texel(5, 4).unwrap().continuous[0];
        assert!(center > edge && edge > 0);
        assert_eq!(outside, 0);
    }

    #[test]
    fn stroke_spacing_fills_a_long_drag_without_holes() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(2),
                },
                vec![[-3.0, 2.0], [3.0, 2.0]],
                0.55,
            ))
            .unwrap();

        for x in 2..14 {
            assert_eq!(cache.texel(x, 4).unwrap().assignment, 2, "x={x}");
        }
    }

    #[test]
    fn ordered_replay_and_scalar_channels_are_byte_deterministic() {
        let layout = small_layout([0, 0]);
        let mut document = MotionBrushDocument::default();
        for tool in [
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(1),
            },
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(4),
            },
            MotionBrushTool::AdjustStrength { target: 2.0 },
            MotionBrushTool::AdjustVariation { target: 0.75 },
            MotionBrushTool::AdjustCoherence { target: 0.25 },
        ] {
            document
                .push_stroke(
                    tool,
                    vec![[0.0, 2.0]],
                    0.6,
                    1.0,
                    0.25,
                    MotionBrushFalloff::Constant,
                )
                .unwrap();
        }
        let cache = MotionFieldCache::rebuild(layout, &document).unwrap();
        let texel = cache.texel(8, 4).unwrap();

        assert_eq!(texel.assignment, 4);
        assert_eq!(texel.continuous, [255, 255, 191, 64]);
    }

    #[test]
    fn smooth_changes_continuous_values_but_never_assignment_identity() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(5),
                },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();
        let before = cache.texel(8, 4).unwrap();
        cache
            .apply_stroke(&stroke(MotionBrushTool::Smooth, vec![[0.0, 2.0]], 0.6))
            .unwrap();
        let after = cache.texel(8, 4).unwrap();

        assert_eq!(after.assignment, before.assignment);
        assert!(after.continuous[0] < before.continuous[0]);
    }

    #[test]
    fn synchronize_removes_variation_and_preserves_strength() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        for tool in [
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(2),
            },
            MotionBrushTool::AdjustStrength { target: 1.5 },
            MotionBrushTool::AdjustVariation { target: 0.75 },
            MotionBrushTool::AdjustCoherence { target: 0.25 },
        ] {
            cache
                .apply_stroke(&stroke(tool, vec![[0.0, 2.0]], 0.6))
                .unwrap();
        }
        let before = cache.texel(8, 4).unwrap();

        cache
            .apply_stroke(&stroke(
                MotionBrushTool::Synchronize {
                    region_id: region_id(4),
                },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();

        let after = cache.texel(8, 4).unwrap();
        assert_eq!(after.assignment, 4);
        assert_eq!(after.continuous, [255, before.continuous[1], 0, 255]);
    }

    #[test]
    fn erase_restores_inherited_global_motion() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(6),
                },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();
        cache
            .apply_stroke(&stroke(MotionBrushTool::Erase, vec![[0.0, 2.0]], 0.6))
            .unwrap();

        let texel = cache.texel(8, 4).unwrap();
        assert_eq!(texel.assignment, 0);
        assert_eq!(texel.continuous[0], 0);
    }

    #[test]
    fn dirty_rectangle_is_clipped_to_the_cache() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        let dirty = cache
            .apply_stroke(&stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(1),
                },
                vec![[-4.0, 0.0]],
                2.0,
            ))
            .unwrap()
            .unwrap();

        assert_eq!(dirty.min(), [0, 0]);
        assert!(dirty.max_exclusive()[0] <= 16);
        assert!(dirty.max_exclusive()[1] <= 8);
    }

    #[test]
    fn moving_cache_replays_absolute_world_strokes() {
        let first_layout = small_layout([0, 0]);
        let moved_layout = small_layout([3, 0]);
        let mut document = MotionBrushDocument::default();
        document
            .push_stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(7),
                },
                vec![[10.0, 2.0]],
                0.6,
                1.0,
                0.25,
                MotionBrushFalloff::Constant,
            )
            .unwrap();

        let mut cache = MotionFieldCache::rebuild(first_layout, &document).unwrap();
        assert!(cache.assignments().iter().all(|&value| value == 0));
        cache.shift_window(moved_layout, &document).unwrap();

        assert_eq!(cache.layout(), moved_layout);
        assert!(cache.assignments().contains(&7));
    }

    #[test]
    fn pointer_capture_requires_enabled_brush_primary_button_and_free_gui() {
        assert!(motion_brush_accepts_pointer(true, false, true));
        assert!(!motion_brush_accepts_pointer(false, false, true));
        assert!(!motion_brush_accepts_pointer(true, true, true));
        assert!(!motion_brush_accepts_pointer(true, false, false));
        assert!(motion_brush_drag_should_end(true, true, true));
        assert!(motion_brush_drag_should_end(false, false, true));
        assert!(motion_brush_drag_should_end(true, false, false));
        assert!(!motion_brush_drag_should_end(true, false, true));
    }

    #[test]
    fn preview_hover_hides_when_the_brush_is_disabled_or_gui_owns_the_pointer() {
        let hit = Some([1.0, 2.0]);

        assert_eq!(motion_brush_preview_hover(true, false, hit), hit);
        assert_eq!(motion_brush_preview_hover(false, false, hit), None);
        assert_eq!(motion_brush_preview_hover(true, true, hit), None);
        assert_eq!(motion_brush_preview_hover(true, false, None), None);
    }

    #[test]
    fn runtime_groups_pointer_samples_and_reports_a_dirty_region() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.enabled = true;
        runtime.tool = MotionBrushTool::ApplyBehavior {
            region_id: region_id(2),
        };
        runtime.begin_stroke([-3.0, 2.0]).unwrap();
        runtime.extend_stroke([3.0, 2.0]).unwrap();
        runtime.end_stroke().unwrap();

        assert_eq!(runtime.document().strokes().len(), 1);
        assert!(runtime.painted_texel_count() > 0);
        assert!(!runtime.take_dirty_regions().is_empty());
        for x in 2..14 {
            assert_eq!(runtime.cache().texel(x, 4).unwrap().assignment, 2);
        }
    }

    #[test]
    fn content_revision_changes_only_for_committed_field_mutations() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        let initial = runtime.cache().content_revision();

        runtime.enabled = true;
        runtime.set_hover(Some([0.0, 2.0]));
        assert_eq!(runtime.cache().content_revision(), initial);
        assert!(runtime.begin_stroke([f32::NAN, 2.0]).is_err());
        assert_eq!(runtime.cache().content_revision(), initial);

        runtime.tool = MotionBrushTool::ApplyBehavior {
            region_id: region_id(2),
        };
        runtime.begin_stroke([0.0, 2.0]).unwrap();
        runtime.end_stroke().unwrap();
        let painted = runtime.cache().content_revision();
        assert_eq!(painted, initial.wrapping_add(1));
        assert!(!runtime.dirty_regions().is_empty());

        runtime.set_hover(None);
        assert_eq!(runtime.cache().content_revision(), painted);
        runtime.recenter([1, 0]).unwrap();
        let recentered = runtime.cache().content_revision();
        assert_eq!(recentered, painted.wrapping_add(1));
        runtime.set_quality(MotionFieldQuality::Default).unwrap();
        assert_eq!(
            runtime.cache().content_revision(),
            recentered.wrapping_add(1)
        );
    }

    #[test]
    fn valid_hover_activates_preview_render_path_without_paint() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.enabled = true;
        runtime.set_hover(Some([0.0, 2.0]));

        assert!(runtime.render_path_active());
        assert!(!runtime.field_data_active());

        runtime.set_hover(None);
        assert!(!runtime.render_path_active());
        assert!(!runtime.field_data_active());
    }

    #[test]
    fn preview_describes_the_exact_active_brush_without_mutating_the_field() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.enabled = true;
        runtime.tool = MotionBrushTool::Erase;
        runtime.radius = 2.5;
        runtime.opacity = 0.75;
        runtime.falloff = MotionBrushFalloff::Linear;
        runtime.set_hover(Some([1.0, 2.0]));

        let preview = runtime.preview().unwrap();

        assert_eq!(preview.center, [1.0, 2.0]);
        assert_eq!(preview.radius, 2.5);
        assert_eq!(preview.opacity, 0.75);
        assert_eq!(preview.falloff, MotionBrushFalloff::Linear);
        assert_eq!(preview.tool, MotionBrushTool::Erase);
        assert_eq!(preview.color, [1.0, 0.24, 0.20, 1.0]);
        assert_eq!(preview.influence_at([1.0, 2.0]), 0.75);
        assert_eq!(preview.influence_at([2.25, 2.0]), 0.375);
        assert_eq!(preview.influence_at([3.5, 2.0]), 0.0);
        assert_eq!(runtime.painted_texel_count(), 0);
        assert!(runtime.document().strokes().is_empty());
    }

    #[test]
    fn preview_path_exists_only_while_a_stroke_is_active() {
        let mut runtime = MotionBrushRuntime::new(small_layout([0, 0])).unwrap();
        runtime.begin_stroke([-1.0, 2.0]).unwrap();
        runtime.extend_stroke([0.0, 2.0]).unwrap();
        runtime.extend_stroke([1.0, 2.0]).unwrap();

        assert_eq!(
            runtime.preview_path(),
            &[[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0]]
        );

        runtime.end_stroke().unwrap();
        assert!(runtime.preview_path().is_empty());
    }

    #[test]
    fn constant_preview_still_rejects_points_outside_the_radius() {
        let preview = MotionBrushPreview {
            center: [0.0, 0.0],
            radius: 2.0,
            opacity: 0.8,
            falloff: MotionBrushFalloff::Constant,
            tool: MotionBrushTool::Smooth,
            color: MotionBrushTool::Smooth.preview_color(),
        };

        assert_eq!(preview.influence_at([1.99, 0.0]), 0.8);
        assert_eq!(preview.influence_at([2.01, 0.0]), 0.0);
    }

    #[test]
    fn scalar_paint_activates_the_inherited_global_controller() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        cache
            .apply_stroke(&stroke(
                MotionBrushTool::AdjustStrength { target: 0.0 },
                vec![[0.0, 2.0]],
                0.6,
            ))
            .unwrap();

        let texel = cache.texel(8, 4).unwrap();
        assert_eq!(texel.assignment, 0);
        assert_eq!(texel.continuous[0], 255);
        assert_eq!(texel.continuous[1], 0);
        assert_eq!(cache.painted_texel_count(), 4);
    }

    #[test]
    fn overlay_colors_are_finite_and_assignment_zero_is_neutral() {
        assert_eq!(
            motion_overlay_color(MotionOverlayChannel::Assignment, 0),
            [0.2, 0.2, 0.2]
        );
        for channel in [
            MotionOverlayChannel::Influence,
            MotionOverlayChannel::Strength,
            MotionOverlayChannel::Variation,
            MotionOverlayChannel::Coherence,
            MotionOverlayChannel::Assignment,
        ] {
            assert!(
                motion_overlay_color(channel, 197)
                    .into_iter()
                    .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
            );
        }
    }

    #[test]
    fn nonuniform_scene_scale_maps_pointer_hits_back_to_canonical_field_space() {
        assert_eq!(
            canonical_brush_xy([4.0, 3.0], [2.0, 0.5]).unwrap(),
            [2.0, 6.0]
        );
        assert!(canonical_brush_xy([4.0, 3.0], [0.0, 1.0]).is_err());
    }

    #[test]
    fn million_unit_offscreen_segment_is_clipped_before_stamp_sampling() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        let stroke = MotionBrushStroke::new(
            1,
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(2),
            },
            vec![[-1.0e6, 2.0], [1.0e6, 2.0]],
            0.3,
            1.0,
            0.01,
            MotionBrushFalloff::Constant,
        )
        .unwrap();

        cache.apply_stroke(&stroke).unwrap();
        assert!(cache.assignments().contains(&2));
    }

    #[test]
    fn one_tile_cache_shift_only_dirties_the_exposed_strip() {
        let first_layout = small_layout([0, 0]);
        let moved_layout = small_layout([1, 0]);
        let document = MotionBrushDocument::default();
        let mut cache = MotionFieldCache::new(first_layout).unwrap();

        let dirty = cache.shift_window(moved_layout, &document).unwrap();
        let dirty_texels: u32 = dirty
            .iter()
            .map(|rect| {
                let min = rect.min();
                let max = rect.max_exclusive();
                (max[0] - min[0]) * (max[1] - min[1])
            })
            .sum();

        assert_eq!(dirty_texels, 8 * 8);
        assert!(dirty_texels < 16 * 8);
        assert_eq!(cache.storage_offset(), [8, 0]);
        assert_eq!(cache.physical_regions(dirty[0])[0].min(), [0, 0]);
        assert_eq!(cache.physical_regions(dirty[0])[0].max_exclusive(), [8, 8]);
    }

    #[test]
    fn ring_shift_replays_segment_stamps_in_world_space() {
        let first_layout = small_layout([0, 0]);
        let moved_layout = small_layout([1, 0]);
        let mut document = MotionBrushDocument::default();
        document
            .push_stroke(
                MotionBrushTool::ApplyBehavior {
                    region_id: region_id(2),
                },
                vec![[-3.8, 2.0], [3.8, 2.0]],
                1.0,
                0.3,
                0.25,
                MotionBrushFalloff::Smooth,
            )
            .unwrap();
        let original = MotionFieldCache::rebuild(first_layout, &document).unwrap();
        let mut shifted = MotionFieldCache::rebuild(first_layout, &document).unwrap();
        shifted.shift_window(moved_layout, &document).unwrap();
        let rebuilt = MotionFieldCache::rebuild(moved_layout, &document).unwrap();
        assert_logical_cache_eq(&shifted, &rebuilt);
        // World [2.25, 2.25] is texel [12,4] before and [4,4] after the shift.
        assert_eq!(
            original.continuous_bytes()[original.texel_index(12, 4).unwrap() * 4],
            rebuilt.continuous_bytes()[rebuilt.texel_index(4, 4).unwrap() * 4]
        );
    }

    #[test]
    fn diagonal_ring_shifts_match_full_stroke_replay() {
        let layout =
            |center| MotionFieldLayout::new([3, 3], 4.0, center, MotionFieldQuality::Low).unwrap();
        let mut document = MotionBrushDocument::default();
        for path in [
            vec![[-9.7, -7.3], [10.1, 8.9]],
            vec![[8.7, -8.3], [-7.9, 9.1]],
        ] {
            document
                .push_stroke(
                    MotionBrushTool::ApplyBehavior {
                        region_id: region_id(2),
                    },
                    path,
                    1.1,
                    0.3,
                    0.23,
                    MotionBrushFalloff::Smooth,
                )
                .unwrap();
        }
        let mut shifted = MotionFieldCache::rebuild(layout([0, 0]), &document).unwrap();
        for center in [[1, 1], [0, 2], [-1, 1], [0, 0], [-1, -1]] {
            shifted.shift_window(layout(center), &document).unwrap();
            let rebuilt = MotionFieldCache::rebuild(layout(center), &document).unwrap();
            assert_logical_cache_eq(&shifted, &rebuilt);
        }
    }

    #[test]
    fn ring_shift_matches_a_fresh_rebuild_without_smoothing() {
        let first_layout = small_layout([0, 0]);
        let moved_layout = small_layout([1, 0]);
        let mut document = MotionBrushDocument::default();
        for (point, slot) in [([1.0, 2.0], 2), ([5.0, 2.0], 3)] {
            document
                .push_stroke(
                    MotionBrushTool::ApplyBehavior {
                        region_id: region_id(slot),
                    },
                    vec![point],
                    0.6,
                    1.0,
                    0.25,
                    MotionBrushFalloff::Constant,
                )
                .unwrap();
        }
        let mut shifted = MotionFieldCache::rebuild(first_layout, &document).unwrap();
        shifted.shift_window(moved_layout, &document).unwrap();
        let rebuilt = MotionFieldCache::rebuild(moved_layout, &document).unwrap();

        assert_logical_cache_eq(&shifted, &rebuilt);
        assert_eq!(shifted.storage_offset(), [8, 0]);
    }

    #[test]
    fn smooth_document_uses_full_deterministic_rebuild_on_shift() {
        let first_layout = small_layout([0, 0]);
        let moved_layout = small_layout([1, 0]);
        let mut document = MotionBrushDocument::default();
        for tool in [
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(2),
            },
            MotionBrushTool::Smooth,
            MotionBrushTool::AdjustVariation { target: 0.8 },
        ] {
            document
                .push_stroke(
                    tool,
                    vec![[3.8, 2.0]],
                    0.8,
                    1.0,
                    0.25,
                    MotionBrushFalloff::Constant,
                )
                .unwrap();
        }
        let mut shifted = MotionFieldCache::rebuild(first_layout, &document).unwrap();
        let dirty = shifted.shift_window(moved_layout, &document).unwrap();
        let rebuilt = MotionFieldCache::rebuild(moved_layout, &document).unwrap();

        assert_logical_cache_eq(&shifted, &rebuilt);
        assert_eq!(shifted.storage_offset(), [0, 0]);
        assert_eq!(
            dirty,
            vec![DirtyRect::from_bounds([0, 0], [16, 8]).unwrap()]
        );
    }

    #[test]
    fn smoothing_snapshot_is_bounded_to_the_affected_region() {
        let cache = MotionFieldCache::new(
            MotionFieldLayout::new([97, 97], 4.0, [0, 0], MotionFieldQuality::Default).unwrap(),
        )
        .unwrap();
        let stroke = stroke(MotionBrushTool::Smooth, vec![[0.0, 0.0]], 1.0);
        let affected = cache.stroke_bounds(&stroke).unwrap();
        let snapshot = ContinuousSnapshot::capture(
            &cache,
            affected.expanded(1, cache.layout().texture_size()),
        );

        assert!(snapshot.bytes.len() < cache.continuous_bytes().len() / 1_000);
    }

    #[test]
    fn extremely_dense_repeated_path_hits_a_deterministic_safety_limit() {
        let mut cache = MotionFieldCache::new(small_layout([0, 0])).unwrap();
        let mut path = Vec::with_capacity(MAX_BRUSH_STAMPS_PER_STROKE + 2);
        for index in 0..MAX_BRUSH_STAMPS_PER_STROKE + 2 {
            path.push([if index % 2 == 0 { -3.0 } else { 3.0 }, 2.0]);
        }
        let stroke = MotionBrushStroke::new(
            1,
            MotionBrushTool::ApplyBehavior {
                region_id: region_id(2),
            },
            path,
            0.1,
            1.0,
            0.01,
            MotionBrushFalloff::Constant,
        )
        .unwrap();

        assert!(cache.apply_stroke(&stroke).is_err());
    }
}
