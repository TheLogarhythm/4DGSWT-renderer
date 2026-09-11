use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::dynamic_archive::{BasisBank, BasisBanks};
use crate::scene_archive::{BasisData, DynamicArchiveSummary, DynamicAsset, MemberCoefficientData};

pub const MOTION_SAMPLE_COUNT: usize = 75;
const MOTION_DIMENSION: usize = 9;
pub const TRANSLATION_MASK: u32 = 1;
pub const ROTATION_MASK: u32 = 2;
pub const SCALE_MASK: u32 = 4;
pub const ALL_MOTION_MASK: u32 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionChannelSelection {
    pub translation: bool,
    pub rotation: bool,
    pub scale: bool,
}

impl MotionChannelSelection {
    pub const fn all_enabled() -> Self {
        Self {
            translation: true,
            rotation: true,
            scale: true,
        }
    }

    pub const fn mask(self) -> u32 {
        (if self.translation {
            TRANSLATION_MASK
        } else {
            0
        }) | (if self.rotation { ROTATION_MASK } else { 0 })
            | (if self.scale { SCALE_MASK } else { 0 })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanonicalGaussian {
    pub position: [f32; 3],
    pub log_scale: [f32; 3],
    pub rotation: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionState {
    pub position: [f32; 3],
    pub log_scale: [f32; 3],
    pub rotation: [f32; 4],
}

impl MotionState {
    pub fn covariance(&self) -> [f32; 6] {
        covariance(self.rotation, self.log_scale)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisFrame {
    pub values: Vec<[f32; MOTION_DIMENSION]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisBanksFrame {
    pub translation: Vec<[f32; 3]>,
    pub rotation: Vec<[f32; 3]>,
    pub scale: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct MergedMotion {
    pub data: MergedMotionData,
    pub canonical: Vec<CanonicalGaussian>,
    pub transform_ids: Vec<u32>,
    pub motion_channel_masks: Vec<u32>,
    pub source_duration_seconds: f32,
    pub summary: DynamicArchiveSummary,
    pub member_offsets: Vec<Vec<u32>>,
    pub member_counts: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub enum MergedMotionData {
    Legacy {
        basis: BasisBank,
        top_k: usize,
        basis_ids: Vec<u32>,
        weights: Vec<f32>,
    },
    Separate {
        basis_banks: BasisBanks,
        top_k: usize,
        basis_ids: Vec<u8>,
        weights: Vec<f32>,
    },
}

pub fn merge_dynamic_asset(asset: DynamicAsset) -> Result<MergedMotion, MotionError> {
    if asset.top_k == 0 {
        return Err(MotionError::new("dynamic top_k must be positive"));
    }
    if asset.members.len() != asset.summary.lod_count
        || asset
            .members
            .iter()
            .any(|members| members.len() != asset.summary.tile_count)
    {
        return Err(MotionError::new(
            "dynamic member grid does not match its summary",
        ));
    }

    let mut canonical = Vec::with_capacity(asset.summary.total_rows);
    let coefficient_capacity = asset
        .summary
        .total_rows
        .checked_mul(asset.top_k)
        .ok_or_else(|| MotionError::new("merged coefficient count overflow"))?;
    let separate = matches!(asset.bases, BasisData::Separate(_));
    let bank_coefficient_capacity = coefficient_capacity
        .checked_mul(3)
        .ok_or_else(|| MotionError::new("merged bank coefficient count overflow"))?;
    let mut legacy_basis_ids = Vec::with_capacity(coefficient_capacity);
    let mut bank_basis_ids = Vec::with_capacity(bank_coefficient_capacity);
    let mut weights = Vec::with_capacity(if separate {
        bank_coefficient_capacity
    } else {
        coefficient_capacity
    });
    let mut transform_ids = Vec::with_capacity(asset.summary.total_rows);
    let mut motion_channel_masks = Vec::with_capacity(asset.summary.total_rows);
    let mut member_offsets = Vec::with_capacity(asset.members.len());
    let mut member_counts = Vec::with_capacity(asset.members.len());

    for members in asset.members {
        let mut lod_offsets = Vec::with_capacity(members.len());
        let mut lod_counts = Vec::with_capacity(members.len());
        for member in members {
            let row_count = member.canonical.len();
            let coefficient_count = row_count
                .checked_mul(asset.top_k)
                .ok_or_else(|| MotionError::new("member coefficient count overflow"))?;
            let coefficient_aligned = match &member.coefficients {
                MemberCoefficientData::Legacy { basis_ids, weights } => {
                    !separate
                        && basis_ids.len() == coefficient_count
                        && weights.len() == coefficient_count
                }
                MemberCoefficientData::Separate { basis_ids, weights } => {
                    separate
                        && basis_ids.len() == coefficient_count * 3
                        && weights.len() == coefficient_count * 3
                }
            };
            if !coefficient_aligned
                || member.transform_ids.len() != row_count
                || member.motion_channel_masks.len() != row_count
            {
                return Err(MotionError::new(
                    "member canonical and motion row counts are not aligned",
                ));
            }
            lod_offsets.push(
                u32::try_from(canonical.len())
                    .map_err(|_| MotionError::new("merged row offset exceeds u32"))?,
            );
            lod_counts.push(
                u32::try_from(row_count)
                    .map_err(|_| MotionError::new("member row count exceeds u32"))?,
            );
            canonical.extend(member.canonical);
            match member.coefficients {
                MemberCoefficientData::Legacy {
                    basis_ids,
                    weights: member_weights,
                } => {
                    legacy_basis_ids.extend(basis_ids);
                    weights.extend(member_weights);
                }
                MemberCoefficientData::Separate {
                    basis_ids,
                    weights: member_weights,
                } => {
                    bank_basis_ids.extend(basis_ids);
                    weights.extend(member_weights);
                }
            }
            transform_ids.extend(member.transform_ids);
            motion_channel_masks.extend(member.motion_channel_masks);
        }
        member_offsets.push(lod_offsets);
        member_counts.push(lod_counts);
    }
    if canonical.len() != asset.summary.total_rows {
        return Err(MotionError::new(
            "merged row count does not match the archive summary",
        ));
    }

    Ok(MergedMotion {
        data: match asset.bases {
            BasisData::Legacy(basis) => MergedMotionData::Legacy {
                basis,
                top_k: asset.top_k,
                basis_ids: legacy_basis_ids,
                weights,
            },
            BasisData::Separate(basis_banks) => MergedMotionData::Separate {
                basis_banks,
                top_k: asset.top_k,
                basis_ids: bank_basis_ids,
                weights,
            },
        },
        canonical,
        transform_ids,
        motion_channel_masks,
        source_duration_seconds: asset.source_duration_seconds,
        summary: asset.summary,
        member_offsets,
        member_counts,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionError(String);

impl MotionError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MotionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MotionError {}

pub fn sample_basis(bank: &BasisBank, query: f32) -> BasisFrame {
    let mut frame = BasisFrame { values: Vec::new() };
    sample_basis_into(bank, query, &mut frame);
    frame
}

pub fn sample_basis_into(bank: &BasisBank, query: f32, frame: &mut BasisFrame) {
    let query = if query.is_finite() {
        query.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scaled = query * (MOTION_SAMPLE_COUNT - 1) as f32;
    let segment = (scaled.floor() as usize).min(MOTION_SAMPLE_COUNT - 2);
    let local = scaled - segment as f32;
    let indices = [
        segment.saturating_sub(1),
        segment,
        segment + 1,
        (segment + 2).min(MOTION_SAMPLE_COUNT - 1),
    ];
    frame
        .values
        .resize(bank.basis_count, [0.0; MOTION_DIMENSION]);
    for basis in 0..bank.basis_count {
        let mut value = [0.0; MOTION_DIMENSION];
        for channel in 0..MOTION_DIMENSION {
            let get = |sample: usize| {
                let index = (basis * MOTION_SAMPLE_COUNT + sample) * MOTION_DIMENSION + channel;
                bank.values.get(index).copied().unwrap_or(0.0)
            };
            let p0 = get(indices[0]);
            let p1 = get(indices[1]);
            let p2 = get(indices[2]);
            let p3 = get(indices[3]);
            let t2 = local * local;
            let t3 = t2 * local;
            value[channel] = 0.5
                * (2.0 * p1
                    + (-p0 + p2) * local
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
        }
        frame.values[basis] = value;
    }
}

pub fn sample_basis_banks(banks: &BasisBanks, query: f32) -> BasisBanksFrame {
    BasisBanksFrame {
        translation: sample_three_channel_bank(&banks.translation, query),
        rotation: sample_three_channel_bank(&banks.rotation, query),
        scale: sample_three_channel_bank(&banks.scale, query),
    }
}

fn sample_three_channel_bank(bank: &BasisBank, query: f32) -> Vec<[f32; 3]> {
    let query = if query.is_finite() {
        query.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let scaled = query * (MOTION_SAMPLE_COUNT - 1) as f32;
    let segment = (scaled.floor() as usize).min(MOTION_SAMPLE_COUNT - 2);
    let local = scaled - segment as f32;
    let indices = [
        segment.saturating_sub(1),
        segment,
        segment + 1,
        (segment + 2).min(MOTION_SAMPLE_COUNT - 1),
    ];
    (0..bank.basis_count)
        .map(|basis| {
            std::array::from_fn(|channel| {
                let get = |sample: usize| {
                    bank.values
                        .get((basis * MOTION_SAMPLE_COUNT + sample) * 3 + channel)
                        .copied()
                        .unwrap_or(0.0)
                };
                let [p0, p1, p2, p3] = indices.map(get);
                let t2 = local * local;
                let t3 = t2 * local;
                0.5 * (2.0 * p1
                    + (-p0 + p2) * local
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
            })
        })
        .collect()
}

pub fn evaluate_row(
    canonical: CanonicalGaussian,
    basis: &BasisFrame,
    ids: &[u32],
    weights: &[f32],
    transform_id: u32,
) -> Result<MotionState, MotionError> {
    evaluate_row_masked(
        canonical,
        basis,
        ids,
        weights,
        transform_id,
        ALL_MOTION_MASK,
        ALL_MOTION_MASK,
    )
}

pub fn evaluate_row_masked(
    canonical: CanonicalGaussian,
    basis: &BasisFrame,
    ids: &[u32],
    weights: &[f32],
    transform_id: u32,
    row_mask: u32,
    preview_mask: u32,
) -> Result<MotionState, MotionError> {
    if row_mask & !ALL_MOTION_MASK != 0 || preview_mask & !ALL_MOTION_MASK != 0 {
        return Err(MotionError::new(
            "motion channel mask contains reserved bits",
        ));
    }
    if ids.is_empty() || ids.len() != weights.len() {
        return Err(MotionError::new(
            "basis IDs and weights must be nonempty and aligned",
        ));
    }
    let mut motion = [0.0_f32; MOTION_DIMENSION];
    for (&basis_id, &weight) in ids.iter().zip(weights) {
        if !weight.is_finite() {
            return Err(MotionError::new("motion weight must be finite"));
        }
        let basis_value = basis
            .values
            .get(basis_id as usize)
            .ok_or_else(|| MotionError::new("basis ID is out of range"))?;
        for channel in 0..MOTION_DIMENSION {
            motion[channel] += weight * basis_value[channel];
        }
    }
    let effective_mask = row_mask & preview_mask;
    if effective_mask & TRANSLATION_MASK == 0 {
        motion[0..3].fill(0.0);
    }
    if effective_mask & ROTATION_MASK == 0 {
        motion[3..6].fill(0.0);
    }
    if effective_mask & SCALE_MASK == 0 {
        motion[6..9].fill(0.0);
    }
    if motion.iter().any(|value| !value.is_finite()) {
        return Err(MotionError::new(
            "effective reconstructed motion is non-finite",
        ));
    }
    let position_delta = match transform_id {
        0 => [motion[0], motion[1], motion[2]],
        1 => [-motion[1], motion[0], motion[2]],
        _ => return Err(MotionError::new("placement transform ID is out of range")),
    };
    let canonical_rotation = normalize_quaternion(canonical.rotation)?;
    let delta_rotation = quaternion_exp([motion[3], motion[4], motion[5]]);
    let rotation = normalize_quaternion(quaternion_multiply(canonical_rotation, delta_rotation))?;
    Ok(MotionState {
        position: [
            canonical.position[0] + position_delta[0],
            canonical.position[1] + position_delta[1],
            canonical.position[2] + position_delta[2],
        ],
        log_scale: [
            canonical.log_scale[0] + motion[6],
            canonical.log_scale[1] + motion[7],
            canonical.log_scale[2] + motion[8],
        ],
        rotation,
    })
}

pub fn evaluate_row_banks_masked(
    canonical: CanonicalGaussian,
    basis: &BasisBanksFrame,
    ids: &[u8],
    weights: &[f32],
    top_k: usize,
    transform_id: u32,
    row_mask: u32,
    preview_mask: u32,
) -> Result<MotionState, MotionError> {
    if top_k == 0 || ids.len() != top_k * 3 || weights.len() != ids.len() {
        return Err(MotionError::new(
            "three-bank basis IDs and weights must be nonempty and aligned",
        ));
    }
    let banks = [&basis.translation, &basis.rotation, &basis.scale];
    let mut motion = [0.0_f32; MOTION_DIMENSION];
    for (bank_index, bank) in banks.into_iter().enumerate() {
        for slot in 0..top_k {
            let coefficient = bank_index * top_k + slot;
            let weight = weights[coefficient];
            if !weight.is_finite() {
                return Err(MotionError::new("motion weight must be finite"));
            }
            let value = bank
                .get(ids[coefficient] as usize)
                .ok_or_else(|| MotionError::new("basis ID is out of range"))?;
            for channel in 0..3 {
                motion[bank_index * 3 + channel] += weight * value[channel];
            }
        }
    }
    evaluate_row_masked(
        canonical,
        &BasisFrame {
            values: vec![motion],
        },
        &[0],
        &[1.0],
        transform_id,
        row_mask,
        preview_mask,
    )
}

pub fn normalize_quaternion(quaternion: [f32; 4]) -> Result<[f32; 4], MotionError> {
    let length_squared = quaternion.iter().map(|value| value * value).sum::<f32>();
    if !length_squared.is_finite() || length_squared <= 0.0 {
        return Err(MotionError::new("quaternion must be finite and nonzero"));
    }
    let inverse_length = length_squared.sqrt().recip();
    let mut normalized = quaternion.map(|value| value * inverse_length);
    if normalized[0] < 0.0 {
        normalized = normalized.map(|value| -value);
    }
    Ok(normalized)
}

fn quaternion_exp(rotation_vector: [f32; 3]) -> [f32; 4] {
    let theta_squared = rotation_vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    if theta_squared <= 1.0e-12 {
        let theta_fourth = theta_squared * theta_squared;
        let scalar = 1.0 - theta_squared / 8.0 + theta_fourth / 384.0;
        let vector_scale = 0.5 - theta_squared / 48.0 + theta_fourth / 3840.0;
        return [
            scalar,
            rotation_vector[0] * vector_scale,
            rotation_vector[1] * vector_scale,
            rotation_vector[2] * vector_scale,
        ];
    }
    let theta = theta_squared.sqrt();
    let half_theta = 0.5 * theta;
    let vector_scale = half_theta.sin() / theta;
    [
        half_theta.cos(),
        rotation_vector[0] * vector_scale,
        rotation_vector[1] * vector_scale,
        rotation_vector[2] * vector_scale,
    ]
}

fn quaternion_multiply(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    let [lw, lx, ly, lz] = left;
    let [rw, rx, ry, rz] = right;
    [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
}

pub fn slerp_quaternion(from: [f32; 4], to: [f32; 4], alpha: f32) -> [f32; 4] {
    let from = normalize_quaternion(from).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let mut to = normalize_quaternion(to).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let mut dot = from
        .iter()
        .zip(to)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    if dot < 0.0 {
        to = to.map(|value| -value);
        dot = -dot;
    }
    let alpha = alpha.clamp(0.0, 1.0);
    if dot > 0.9995 {
        let blended = std::array::from_fn(|index| from[index] + alpha * (to[index] - from[index]));
        return normalize_quaternion(blended).unwrap_or(from);
    }
    let angle = dot.clamp(-1.0, 1.0).acos();
    let denominator = angle.sin();
    let from_weight = ((1.0 - alpha) * angle).sin() / denominator;
    let to_weight = (alpha * angle).sin() / denominator;
    std::array::from_fn(|index| from_weight * from[index] + to_weight * to[index])
}

pub fn blend_states(from: MotionState, to: MotionState, alpha: f32) -> MotionState {
    let alpha = alpha.clamp(0.0, 1.0);
    MotionState {
        position: std::array::from_fn(|index| {
            from.position[index] + alpha * (to.position[index] - from.position[index])
        }),
        log_scale: std::array::from_fn(|index| {
            from.log_scale[index] + alpha * (to.log_scale[index] - from.log_scale[index])
        }),
        rotation: slerp_quaternion(from.rotation, to.rotation, alpha),
    }
}

fn covariance(rotation: [f32; 4], log_scale: [f32; 3]) -> [f32; 6] {
    let [w, x, y, z] = normalize_quaternion(rotation).unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let rows = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let variances = log_scale.map(|value| (2.0 * value).exp());
    let component = |row: usize, column: usize| {
        (0..3)
            .map(|axis| rows[row][axis] * variances[axis] * rows[column][axis])
            .sum::<f32>()
    };
    [
        component(0, 0),
        component(0, 1),
        component(0, 2),
        component(1, 1),
        component(1, 2),
        component(2, 2),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LoopPolicy {
    Hold,
    DirectWrap,
    AppendedTransition,
    BlendToStart,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimelineSample {
    Source { u: f32 },
    Blend { from_u: f32, to_u: f32, alpha: f32 },
}

impl TimelineSample {
    pub fn is_blend(self) -> bool {
        matches!(self, Self::Blend { .. })
    }
}

#[derive(Clone, Debug)]
pub struct MotionPlayback {
    pub playing: bool,
    pub motion_enabled: bool,
    pub channel_selection: MotionChannelSelection,
    pub preview_seconds: f32,
    pub speed: f32,
    pub policy: LoopPolicy,
    pub transition_seconds: f32,
    pub blend_window_seconds: f32,
    source_duration: f32,
    scrub_override: Option<f32>,
    dirty: bool,
    position_dirty: bool,
}

impl MotionPlayback {
    pub fn new(source_duration: f32) -> Result<Self, MotionError> {
        if !source_duration.is_finite() || source_duration <= 0.0 {
            return Err(MotionError::new(
                "source duration must be positive and finite",
            ));
        }
        Ok(Self {
            playing: true,
            motion_enabled: true,
            channel_selection: MotionChannelSelection::all_enabled(),
            preview_seconds: 0.0,
            speed: 1.0,
            policy: LoopPolicy::Hold,
            transition_seconds: 0.5_f32.min(source_duration),
            blend_window_seconds: (0.1 * source_duration).max(f32::EPSILON),
            source_duration,
            scrub_override: None,
            dirty: true,
            position_dirty: true,
        })
    }

    pub fn source_duration(&self) -> f32 {
        self.source_duration
    }

    pub fn set_policy(&mut self, policy: LoopPolicy) {
        if self.policy != policy {
            self.policy = policy;
            self.scrub_override = None;
            self.dirty = true;
        }
    }

    pub fn set_transition_seconds(&mut self, seconds: f32) -> Result<(), MotionError> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(MotionError::new(
                "transition duration must be positive and finite",
            ));
        }
        if self.transition_seconds != seconds {
            self.transition_seconds = seconds;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn set_blend_window_seconds(&mut self, seconds: f32) -> Result<(), MotionError> {
        if !seconds.is_finite() || seconds <= 0.0 || seconds >= self.source_duration {
            return Err(MotionError::new(
                "blend window must be positive, finite, and shorter than the source",
            ));
        }
        if self.blend_window_seconds != seconds {
            self.blend_window_seconds = seconds;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn set_speed(&mut self, speed: f32) -> Result<(), MotionError> {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(MotionError::new(
                "playback speed must be positive and finite",
            ));
        }
        self.speed = speed;
        Ok(())
    }

    pub fn set_motion_enabled(&mut self, enabled: bool) {
        if self.motion_enabled != enabled {
            self.motion_enabled = enabled;
            self.dirty = true;
        }
    }

    fn set_channel_enabled(&mut self, channel: u32, enabled: bool) {
        let mut updated = self.channel_selection;
        match channel {
            TRANSLATION_MASK => updated.translation = enabled,
            ROTATION_MASK => updated.rotation = enabled,
            SCALE_MASK => updated.scale = enabled,
            _ => unreachable!("only public motion-channel constants are accepted"),
        }
        if updated != self.channel_selection {
            self.channel_selection = updated;
            self.dirty = true;
        }
    }

    pub fn set_translation_enabled(&mut self, enabled: bool) {
        self.set_channel_enabled(TRANSLATION_MASK, enabled);
    }

    pub fn set_rotation_enabled(&mut self, enabled: bool) {
        self.set_channel_enabled(ROTATION_MASK, enabled);
    }

    pub fn set_scale_enabled(&mut self, enabled: bool) {
        self.set_channel_enabled(SCALE_MASK, enabled);
    }

    pub fn translation_enabled(&self) -> bool {
        self.channel_selection.translation
    }

    pub fn rotation_enabled(&self) -> bool {
        self.channel_selection.rotation
    }

    pub fn scale_enabled(&self) -> bool {
        self.channel_selection.scale
    }

    pub fn play(&mut self) {
        if !self.playing {
            if self.policy == LoopPolicy::Hold && self.preview_seconds >= self.source_duration {
                self.preview_seconds = 0.0;
                self.position_dirty = true;
            }
            self.playing = true;
            self.scrub_override = None;
            self.dirty = true;
        }
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn restart(&mut self) {
        self.preview_seconds = 0.0;
        self.scrub_override = Some(0.0);
        self.dirty = true;
        self.position_dirty = true;
    }

    pub fn scrub_normalized(&mut self, u: f32) -> Result<(), MotionError> {
        if !u.is_finite() || !(0.0..=1.0).contains(&u) {
            return Err(MotionError::new("scrub time must be within [0, 1]"));
        }
        self.playing = false;
        self.preview_seconds = u * self.source_duration;
        self.scrub_override = Some(u);
        self.dirty = true;
        self.position_dirty = true;
        Ok(())
    }

    pub fn seek_seconds(&mut self, seconds: f32) -> Result<(), MotionError> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(MotionError::new(
                "preview time must be finite and nonnegative",
            ));
        }
        self.preview_seconds = seconds;
        self.scrub_override = None;
        self.dirty = true;
        self.position_dirty = true;
        Ok(())
    }

    pub fn advance(&mut self, delta_seconds: f32) -> Result<TimelineSample, MotionError> {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return Err(MotionError::new(
                "frame delta must be finite and nonnegative",
            ));
        }
        if self.playing && delta_seconds > 0.0 {
            self.scrub_override = None;
            self.preview_seconds += delta_seconds * self.speed;
            if self.policy == LoopPolicy::Hold && self.preview_seconds >= self.source_duration {
                self.preview_seconds = self.source_duration;
                self.playing = false;
            }
            self.dirty = true;
            self.position_dirty = true;
        }
        self.sample()
    }

    pub fn sample(&self) -> Result<TimelineSample, MotionError> {
        if let Some(u) = self.scrub_override {
            return Ok(TimelineSample::Source { u });
        }
        match self.policy {
            LoopPolicy::Hold => Ok(TimelineSample::Source {
                u: (self.preview_seconds / self.source_duration).clamp(0.0, 1.0),
            }),
            LoopPolicy::DirectWrap => {
                let phase = self.preview_seconds.rem_euclid(self.source_duration);
                Ok(TimelineSample::Source {
                    u: phase / self.source_duration,
                })
            }
            LoopPolicy::AppendedTransition => {
                let period = self.source_duration + self.transition_seconds;
                let phase = self.preview_seconds.rem_euclid(period);
                if phase <= self.source_duration {
                    Ok(TimelineSample::Source {
                        u: phase / self.source_duration,
                    })
                } else {
                    let linear = (phase - self.source_duration) / self.transition_seconds;
                    Ok(TimelineSample::Blend {
                        from_u: 1.0,
                        to_u: 0.0,
                        alpha: smoothstep(linear),
                    })
                }
            }
            LoopPolicy::BlendToStart => {
                let phase = self.preview_seconds.rem_euclid(self.source_duration);
                let blend_start = self.source_duration - self.blend_window_seconds;
                let u = phase / self.source_duration;
                if phase < blend_start {
                    Ok(TimelineSample::Source { u })
                } else {
                    let linear = (phase - blend_start) / self.blend_window_seconds;
                    Ok(TimelineSample::Blend {
                        from_u: u,
                        to_u: 0.0,
                        alpha: smoothstep(linear),
                    })
                }
            }
        }
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn mark_render_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn take_position_dirty(&mut self) -> bool {
        std::mem::take(&mut self.position_dirty)
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_archive::BasisBank;
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct ParityFixture {
        fixture_version: u32,
        sample_count: usize,
        motion_dimension: usize,
        basis_generators: Vec<ParityBasisGenerator>,
        basis_ids: Vec<u32>,
        weights: Vec<f32>,
        queries: Vec<ParityQuery>,
    }

    #[derive(Deserialize)]
    struct ParityBasisGenerator {
        offset: [f32; 9],
        slope: [f32; 9],
    }

    #[derive(Deserialize)]
    struct ParityQuery {
        kind: String,
        u: f32,
        expected_motion: [f32; 9],
    }

    #[derive(Deserialize)]
    struct ParityV3Fixture {
        fixture_version: u32,
        schema_version: u32,
        sample_count: usize,
        top_k: usize,
        placement_transform_id: u32,
        canonical: ParityV3Canonical,
        basis_banks: ParityV3Banks,
        queries: Vec<ParityV3Query>,
    }

    #[derive(Deserialize)]
    struct ParityV3Canonical {
        position: [f32; 3],
        rotation_wxyz: [f32; 4],
        log_scale: [f32; 3],
    }

    #[derive(Deserialize)]
    struct ParityV3Banks {
        translation: ParityV3Bank,
        rotation: ParityV3Bank,
        scale: ParityV3Bank,
    }

    #[derive(Deserialize)]
    struct ParityV3Bank {
        basis_generators: Vec<ParityV3BasisGenerator>,
        basis_ids: Vec<u8>,
        weights: Vec<f32>,
    }

    #[derive(Deserialize)]
    struct ParityV3BasisGenerator {
        offset: [f32; 3],
        slope: [f32; 3],
        curvature: [f32; 3],
    }

    #[derive(Deserialize)]
    struct ParityV3Query {
        kind: String,
        u: f32,
        expected_trs_blocks: [[f32; 3]; 3],
        states: Vec<ParityV3State>,
    }

    #[derive(Deserialize)]
    struct ParityV3State {
        mask: u32,
        position: [f32; 3],
        rotation_wxyz: [f32; 4],
        log_scale: [f32; 3],
        covariance_upper: [f32; 6],
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{actual} != {expected}"
        );
    }

    fn assert_close3(actual: [f32; 3], expected: [f32; 3]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected);
        }
    }

    fn canonical() -> CanonicalGaussian {
        CanonicalGaussian {
            position: [0.0, 0.0, 0.0],
            log_scale: [0.0, 0.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn constant_frame(value: [f32; 9]) -> BasisFrame {
        BasisFrame {
            values: vec![value],
        }
    }

    fn linear_bank() -> BasisBank {
        let mut values = Vec::with_capacity(75 * 9);
        for sample in 0..75 {
            let u = sample as f32 / 74.0;
            values.extend_from_slice(&[u, 2.0 * u, -u, 0.0, 0.0, 0.0, u, u, u]);
        }
        BasisBank {
            basis_count: 1,
            values,
        }
    }

    fn constant_three_banks() -> BasisBanks {
        let bank = |value: [f32; 3]| BasisBank {
            basis_count: 1,
            values: (0..75).flat_map(|_| value).collect(),
        };
        BasisBanks {
            translation: bank([1.0, 2.0, 3.0]),
            rotation: bank([0.0, 0.0, std::f32::consts::FRAC_PI_2]),
            scale: bank([0.0, 0.5 * 2.0_f32.ln(), 2.0_f32.ln()]),
        }
    }

    #[test]
    fn separate_trs_banks_honor_masks_and_covariance() {
        let frame = sample_basis_banks(&constant_three_banks(), 0.5);
        let canonical = CanonicalGaussian {
            position: [10.0, 20.0, 30.0],
            log_scale: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let root_half = 0.5_f32.sqrt();
        for (mask, position, rotation, log_scale, covariance) in [
            (
                1,
                [11.0, 22.0, 33.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0; 3],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ),
            (
                3,
                [11.0, 22.0, 33.0],
                [root_half, 0.0, 0.0, root_half],
                [0.0; 3],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ),
            (
                5,
                [11.0, 22.0, 33.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.5 * 2.0_f32.ln(), 2.0_f32.ln()],
                [1.0, 0.0, 0.0, 2.0, 0.0, 4.0],
            ),
            (
                7,
                [11.0, 22.0, 33.0],
                [root_half, 0.0, 0.0, root_half],
                [0.0, 0.5 * 2.0_f32.ln(), 2.0_f32.ln()],
                [2.0, 0.0, 0.0, 1.0, 0.0, 4.0],
            ),
        ] {
            let state = evaluate_row_banks_masked(
                canonical,
                &frame,
                &[0_u8, 0, 0],
                &[1.0, 1.0, 1.0],
                1,
                0,
                mask,
                ALL_MOTION_MASK,
            )
            .unwrap();
            assert_close3(state.position, position);
            assert_close3(state.log_scale, log_scale);
            for (actual, expected) in state.rotation.into_iter().zip(rotation) {
                assert_close(actual, expected);
            }
            for (actual, expected) in state.covariance().into_iter().zip(covariance) {
                assert_close(actual, expected);
            }
        }
    }

    #[test]
    fn samples_exact_knots_and_clamps_queries() {
        let bank = linear_bank();
        for sample in [0, 1, 37, 73, 74] {
            let u = sample as f32 / 74.0;
            assert_close(sample_basis(&bank, u).values[0][0], u);
        }
        assert_close(sample_basis(&bank, -1.0).values[0][0], 0.0);
        assert_close(sample_basis(&bank, 2.0).values[0][0], 1.0);
    }

    #[test]
    fn catmull_rom_reproduces_linear_interior_motion() {
        let frame = sample_basis(&linear_bank(), 0.5);
        assert_close3(
            [frame.values[0][0], frame.values[0][1], frame.values[0][2]],
            [0.5, 1.0, -0.5],
        );
    }

    #[test]
    fn sparse_rows_sum_repeated_and_zero_weight_slots() {
        let frame = BasisFrame {
            values: vec![
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        };
        let state = evaluate_row(canonical(), &frame, &[0, 1, 0], &[0.5, 0.25, 0.0], 0).unwrap();
        assert_close3(state.position, [1.5, 2.25, 3.0]);
    }

    #[test]
    fn placement_rotates_translation_but_not_local_rotation_or_scale() {
        let frame = constant_frame([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let state = evaluate_row(canonical(), &frame, &[0], &[1.0], 1).unwrap();
        assert_close3(state.position, [-2.0, 1.0, 3.0]);
        assert_close3(state.log_scale, [0.4, 0.5, 0.6]);
        assert!(state.rotation.iter().all(|value| value.is_finite()));
        assert!(state.rotation[1].abs() > 0.0);
        assert!(state.rotation[2].abs() > 0.0);
        assert!(state.rotation[3].abs() > 0.0);
    }

    #[test]
    fn row_and_preview_masks_isolate_independent_channels() {
        let canonical = CanonicalGaussian {
            position: [10.0, 20.0, 30.0],
            log_scale: [1.0, 2.0, 3.0],
            ..canonical()
        };
        let frame = constant_frame([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0]);
        let scale_fallback = evaluate_row_masked(
            canonical,
            &frame,
            &[0],
            &[1.0],
            0,
            TRANSLATION_MASK | ROTATION_MASK,
            ALL_MOTION_MASK,
        )
        .unwrap();
        assert_close3(scale_fallback.position, [11.0, 22.0, 33.0]);
        assert_close3(scale_fallback.log_scale, canonical.log_scale);
        assert_ne!(scale_fallback.rotation, canonical.rotation);

        let rotation_scale = evaluate_row_masked(
            canonical,
            &frame,
            &[0],
            &[1.0],
            0,
            ALL_MOTION_MASK,
            ROTATION_MASK | SCALE_MASK,
        )
        .unwrap();
        assert_close3(rotation_scale.position, canonical.position);
        assert_close3(rotation_scale.log_scale, [5.0, 7.0, 9.0]);
        assert_ne!(rotation_scale.rotation, canonical.rotation);

        for preview_mask in 0..=ALL_MOTION_MASK {
            let state = evaluate_row_masked(
                canonical,
                &frame,
                &[0],
                &[1.0],
                0,
                ALL_MOTION_MASK,
                preview_mask,
            )
            .unwrap();
            if preview_mask & TRANSLATION_MASK == 0 {
                assert_close3(state.position, canonical.position);
            } else {
                assert_close3(state.position, [11.0, 22.0, 33.0]);
            }
            if preview_mask & ROTATION_MASK == 0 {
                assert_eq!(state.rotation, canonical.rotation);
            } else {
                assert_ne!(state.rotation, canonical.rotation);
            }
            if preview_mask & SCALE_MASK == 0 {
                assert_close3(state.log_scale, canonical.log_scale);
            } else {
                assert_close3(state.log_scale, [5.0, 7.0, 9.0]);
            }
        }

        let selection = MotionChannelSelection {
            translation: true,
            rotation: false,
            scale: true,
        };
        assert_eq!(selection.mask(), TRANSLATION_MASK | SCALE_MASK);
    }

    #[test]
    fn disabled_nonfinite_scale_is_overwritten_before_effective_state_validation() {
        let frame = constant_frame([
            1.0,
            2.0,
            3.0,
            0.1,
            0.2,
            0.3,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ]);

        let state = evaluate_row_masked(
            canonical(),
            &frame,
            &[0],
            &[1.0],
            0,
            TRANSLATION_MASK | ROTATION_MASK,
            ALL_MOTION_MASK,
        )
        .unwrap();

        assert_close3(state.position, [1.0, 2.0, 3.0]);
        assert_close3(state.log_scale, canonical().log_scale);
        assert_ne!(state.rotation, canonical().rotation);
        assert!(
            evaluate_row_masked(
                canonical(),
                &frame,
                &[0],
                &[1.0],
                0,
                ALL_MOTION_MASK,
                ALL_MOTION_MASK,
            )
            .is_err()
        );
    }

    #[test]
    fn local_rotation_right_multiplies_canonical_quaternion() {
        let root_half = 0.5_f32.sqrt();
        let canonical = CanonicalGaussian {
            rotation: [root_half, 0.0, 0.0, root_half],
            ..canonical()
        };
        let frame = constant_frame([
            0.0,
            0.0,
            0.0,
            std::f32::consts::FRAC_PI_2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        let state = evaluate_row(canonical, &frame, &[0], &[1.0], 0).unwrap();
        for (actual, expected) in state.rotation.into_iter().zip([0.5, 0.5, 0.5, 0.5]) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn zero_and_small_rotation_vectors_remain_finite() {
        for omega in [[0.0, 0.0, 0.0], [1.0e-9, -2.0e-9, 3.0e-9]] {
            let frame =
                constant_frame([0.0, 0.0, 0.0, omega[0], omega[1], omega[2], 0.0, 0.0, 0.0]);
            let state = evaluate_row(canonical(), &frame, &[0], &[1.0], 0).unwrap();
            assert!(state.rotation.iter().all(|value| value.is_finite()));
            assert_close(
                state
                    .rotation
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>(),
                1.0,
            );
        }
    }

    #[test]
    fn covariance_uses_anisotropic_log_scale() {
        let frame = constant_frame([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5 * 2.0_f32.ln(),
            2.0_f32.ln(),
        ]);
        let state = evaluate_row(canonical(), &frame, &[0], &[1.0], 0).unwrap();
        let covariance = state.covariance();
        assert_close(covariance[0], 1.0);
        assert_close(covariance[3], 2.0);
        assert_close(covariance[5], 4.0);
    }

    #[test]
    fn hold_stops_at_the_finite_endpoint() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        let sample = playback.advance(1.5).unwrap();
        assert_eq!(sample, TimelineSample::Source { u: 1.0 });
        assert!(!playback.playing);
    }

    #[test]
    fn direct_wrap_preserves_the_source_discontinuity() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        playback.set_policy(LoopPolicy::DirectWrap);
        assert_eq!(
            playback.advance(1.25).unwrap(),
            TimelineSample::Source { u: 0.25 }
        );
    }

    #[test]
    fn appended_transition_preserves_full_source_then_blends_end_to_start() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        playback.set_policy(LoopPolicy::AppendedTransition);
        playback.set_transition_seconds(0.5).unwrap();
        playback.seek_seconds(1.25).unwrap();
        assert_eq!(
            playback.sample().unwrap(),
            TimelineSample::Blend {
                from_u: 1.0,
                to_u: 0.0,
                alpha: 0.5,
            }
        );
    }

    #[test]
    fn blend_to_start_keeps_prefix_and_blends_current_state_to_start() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        playback.set_policy(LoopPolicy::BlendToStart);
        playback.set_blend_window_seconds(0.2).unwrap();
        playback.seek_seconds(0.5).unwrap();
        assert_eq!(
            playback.sample().unwrap(),
            TimelineSample::Source { u: 0.5 }
        );
        playback.seek_seconds(0.9).unwrap();
        let TimelineSample::Blend {
            from_u,
            to_u,
            alpha,
        } = playback.sample().unwrap()
        else {
            panic!("expected a blend sample");
        };
        assert_close(from_u, 0.9);
        assert_close(to_u, 0.0);
        assert_close(alpha, 0.5);
    }

    #[test]
    fn manual_scrub_can_select_the_exact_endpoint_under_any_policy() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        playback.set_policy(LoopPolicy::DirectWrap);
        playback.scrub_normalized(1.0).unwrap();
        assert_eq!(
            playback.sample().unwrap(),
            TimelineSample::Source { u: 1.0 }
        );
    }

    #[test]
    fn invalid_playback_parameters_do_not_change_state() {
        let mut playback = MotionPlayback::new(2.0).unwrap();
        let old_transition = playback.transition_seconds;
        let old_window = playback.blend_window_seconds;
        assert!(playback.set_transition_seconds(0.0).is_err());
        assert!(playback.set_blend_window_seconds(2.0).is_err());
        assert_eq!(playback.transition_seconds, old_transition);
        assert_eq!(playback.blend_window_seconds, old_window);
    }

    #[test]
    fn dirty_state_is_consumed_once() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        assert!(playback.take_dirty());
        assert!(!playback.take_dirty());
        playback.pause();
        assert!(!playback.advance(0.1).unwrap().is_blend());
        assert!(!playback.take_dirty());
        playback.scrub_normalized(0.25).unwrap();
        assert!(playback.take_dirty());
        assert!(!playback.take_dirty());
    }

    #[test]
    fn paused_channel_toggle_dirties_exactly_one_update() {
        let mut playback = MotionPlayback::new(1.0).unwrap();
        playback.pause();
        assert!(playback.take_dirty());
        playback.set_scale_enabled(false);
        assert!(!playback.scale_enabled());
        assert!(playback.take_dirty());
        assert!(!playback.take_dirty());
        playback.set_scale_enabled(false);
        assert!(!playback.take_dirty());
    }

    #[test]
    fn rust_cpu_evaluator_matches_the_shared_python_motion_fixture() {
        let workspace_root = std::env::var_os("FOURDGSWT_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let fixture_path = workspace_root.join("tests/fixtures/motion_parity_v1.json");
        if !fixture_path.is_file() {
            eprintln!(
                "shared Rust/Python parity fixture skipped outside the umbrella workspace: {}",
                fixture_path.display()
            );
            return;
        }
        let fixture: ParityFixture = serde_json::from_slice(
            &std::fs::read(&fixture_path)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture_path.display())),
        )
        .unwrap();
        assert_eq!(fixture.fixture_version, 1);
        assert_eq!(fixture.sample_count, 75);
        assert_eq!(fixture.motion_dimension, 9);
        assert_eq!(fixture.basis_ids.len(), fixture.weights.len());

        let mut values = Vec::with_capacity(fixture.basis_generators.len() * 75 * 9);
        for generator in &fixture.basis_generators {
            for sample in 0..75 {
                let u = sample as f32 / 74.0;
                for channel in 0..9 {
                    values.push(generator.offset[channel] + u * generator.slope[channel]);
                }
            }
        }
        let bank = BasisBank {
            basis_count: fixture.basis_generators.len(),
            values,
        };

        for query in fixture.queries {
            let frame = sample_basis(&bank, query.u);
            let mut motion = [0.0_f32; 9];
            for (&basis_id, &weight) in fixture.basis_ids.iter().zip(&fixture.weights) {
                for (channel, value) in motion.iter_mut().enumerate() {
                    *value += weight * frame.values[basis_id as usize][channel];
                }
            }
            for (channel, (actual, expected)) in
                motion.into_iter().zip(query.expected_motion).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "{} channel {channel}: {actual} != {expected}",
                    query.kind
                );
            }
        }
    }

    #[test]
    fn rust_cpu_evaluator_matches_the_shared_v2_scale_fallback_fixture() {
        let workspace_root = std::env::var_os("FOURDGSWT_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let fixture_path = workspace_root.join("tests/fixtures/motion_parity_v2.json");
        if !fixture_path.is_file() {
            eprintln!(
                "shared v2 Rust/Python parity fixture skipped outside the umbrella workspace: {}",
                fixture_path.display()
            );
            return;
        }
        let fixture: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&fixture_path)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture_path.display())),
        )
        .unwrap();
        assert_eq!(fixture["fixture_version"], 2);
        assert_eq!(fixture["schema_version"], 2);
        assert_eq!(fixture["lod_count"], 6);
        assert_eq!(
            fixture["source_identities"][0]["archive_occurrences"],
            serde_json::json!([2, 5])
        );

        let motion: [f32; 9] =
            serde_json::from_value(fixture["basis_generator"]["offset"].clone()).unwrap();
        let canonical_value = &fixture["canonical"];
        let canonical = CanonicalGaussian {
            position: serde_json::from_value(canonical_value["position"].clone()).unwrap(),
            rotation: serde_json::from_value(canonical_value["rotation_wxyz"].clone()).unwrap(),
            log_scale: serde_json::from_value(canonical_value["log_scale"].clone()).unwrap(),
        };
        let state = evaluate_row_masked(
            canonical,
            &constant_frame(motion),
            &[0],
            &[1.0],
            fixture["placement_transform_id"].as_u64().unwrap() as u32,
            fixture["archive_channel_mask"].as_u64().unwrap() as u32,
            fixture["preview_channel_mask"].as_u64().unwrap() as u32,
        )
        .unwrap();
        let expected = &fixture["expected_state"];
        assert_close3(
            state.position,
            serde_json::from_value(expected["position"].clone()).unwrap(),
        );
        assert_close3(
            state.log_scale,
            serde_json::from_value(expected["log_scale"].clone()).unwrap(),
        );
        let expected_rotation: [f32; 4] =
            serde_json::from_value(expected["rotation_wxyz"].clone()).unwrap();
        for (actual, expected) in state.rotation.into_iter().zip(expected_rotation) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn rust_cpu_evaluator_matches_the_shared_v3_three_bank_fixture() {
        let workspace_root = std::env::var_os("FOURDGSWT_WORKSPACE_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));
        let fixture_path = workspace_root.join("tests/fixtures/motion_parity_v3.json");
        let fixture: ParityV3Fixture = serde_json::from_slice(
            &std::fs::read(&fixture_path)
                .unwrap_or_else(|error| panic!("{}: {error}", fixture_path.display())),
        )
        .unwrap();
        assert_eq!(fixture.fixture_version, 3);
        assert_eq!(fixture.schema_version, 3);
        assert_eq!(fixture.sample_count, 75);
        assert_eq!(fixture.top_k, 2);

        let make_bank = |bank: &ParityV3Bank| {
            assert_eq!(bank.basis_ids.len(), fixture.top_k);
            assert_eq!(bank.weights.len(), fixture.top_k);
            let mut values = Vec::with_capacity(bank.basis_generators.len() * 75 * 3);
            for generator in &bank.basis_generators {
                for sample in 0..75 {
                    let u = sample as f32 / 74.0;
                    for channel in 0..3 {
                        values.push(
                            generator.offset[channel]
                                + u * generator.slope[channel]
                                + u * u * generator.curvature[channel],
                        );
                    }
                }
            }
            BasisBank {
                basis_count: bank.basis_generators.len(),
                values,
            }
        };
        let banks = BasisBanks {
            translation: make_bank(&fixture.basis_banks.translation),
            rotation: make_bank(&fixture.basis_banks.rotation),
            scale: make_bank(&fixture.basis_banks.scale),
        };
        let ids: Vec<u8> = [
            &fixture.basis_banks.translation.basis_ids[..],
            &fixture.basis_banks.rotation.basis_ids[..],
            &fixture.basis_banks.scale.basis_ids[..],
        ]
        .concat();
        let weights: Vec<f32> = [
            &fixture.basis_banks.translation.weights[..],
            &fixture.basis_banks.rotation.weights[..],
            &fixture.basis_banks.scale.weights[..],
        ]
        .concat();
        let canonical = CanonicalGaussian {
            position: fixture.canonical.position,
            rotation: fixture.canonical.rotation_wxyz,
            log_scale: fixture.canonical.log_scale,
        };
        let mut saw_knot = false;
        let mut saw_intermediate = false;
        let mut masks = std::collections::BTreeSet::new();
        for query in fixture.queries {
            saw_knot |= query.kind == "knot";
            saw_intermediate |= query.kind == "intermediate";
            let frame = sample_basis_banks(&banks, query.u);
            let reconstructed = [
                [
                    weights[0] * frame.translation[ids[0] as usize][0]
                        + weights[1] * frame.translation[ids[1] as usize][0],
                    weights[0] * frame.translation[ids[0] as usize][1]
                        + weights[1] * frame.translation[ids[1] as usize][1],
                    weights[0] * frame.translation[ids[0] as usize][2]
                        + weights[1] * frame.translation[ids[1] as usize][2],
                ],
                [
                    weights[2] * frame.rotation[ids[2] as usize][0]
                        + weights[3] * frame.rotation[ids[3] as usize][0],
                    weights[2] * frame.rotation[ids[2] as usize][1]
                        + weights[3] * frame.rotation[ids[3] as usize][1],
                    weights[2] * frame.rotation[ids[2] as usize][2]
                        + weights[3] * frame.rotation[ids[3] as usize][2],
                ],
                [
                    weights[4] * frame.scale[ids[4] as usize][0]
                        + weights[5] * frame.scale[ids[5] as usize][0],
                    weights[4] * frame.scale[ids[4] as usize][1]
                        + weights[5] * frame.scale[ids[5] as usize][1],
                    weights[4] * frame.scale[ids[4] as usize][2]
                        + weights[5] * frame.scale[ids[5] as usize][2],
                ],
            ];
            for (actual, expected) in reconstructed
                .into_iter()
                .flatten()
                .zip(query.expected_trs_blocks.into_iter().flatten())
            {
                assert_close(actual, expected);
            }
            for expected in query.states {
                masks.insert(expected.mask);
                let state = evaluate_row_banks_masked(
                    canonical,
                    &frame,
                    &ids,
                    &weights,
                    fixture.top_k,
                    fixture.placement_transform_id,
                    expected.mask,
                    ALL_MOTION_MASK,
                )
                .unwrap();
                assert_close3(state.position, expected.position);
                assert_close3(state.log_scale, expected.log_scale);
                for (actual, expected) in state.rotation.into_iter().zip(expected.rotation_wxyz) {
                    assert_close(actual, expected);
                }
                for (actual, expected) in state
                    .covariance()
                    .into_iter()
                    .zip(expected.covariance_upper)
                {
                    assert_close(actual, expected);
                }
            }
        }
        assert!(saw_knot && saw_intermediate);
        assert_eq!(masks, std::collections::BTreeSet::from([1, 3, 5, 7]));
    }
}
