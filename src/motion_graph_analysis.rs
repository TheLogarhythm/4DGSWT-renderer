use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::motion::{
    ALL_MOTION_MASK, BasisBanksFrame, BasisFrame, MOTION_SAMPLE_COUNT, MergedMotion,
    MergedMotionData, MotionState, evaluate_row_banks_masked, evaluate_row_masked, sample_basis,
    sample_basis_banks,
};
use crate::motion_graph::{
    MotionDiscontinuity, MotionGraph, MotionGraphAnalysisSummary, MotionGraphNode,
    MotionGraphSettings, MotionJump, MotionJumpDistribution, MotionMetricSet, MotionSegmentFeature,
    SOURCE_SEGMENT_COUNT, rank_eligible_jumps,
};
use crate::utils::get_time_milliseconds;

pub const DEFAULT_GRAPH_ANALYSIS_ROW_BUDGET: usize = 2_048;

const TRANSLATION_SCALE_FLOOR: f32 = 1.0e-6;
const ROTATION_SCALE_FLOOR: f32 = 1.0e-5;
const LOG_SCALE_FLOOR: f32 = 1.0e-6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionGraphAnalysisError(String);

impl MotionGraphAnalysisError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MotionGraphAnalysisError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MotionGraphAnalysisError {}

pub fn build_motion_graph(
    motion: &MergedMotion,
    settings: MotionGraphSettings,
) -> Result<MotionGraph, MotionGraphAnalysisError> {
    build_motion_graph_with_clock(motion, settings, get_time_milliseconds)
}

fn build_motion_graph_with_clock(
    motion: &MergedMotion,
    settings: MotionGraphSettings,
    mut now_milliseconds: impl FnMut() -> f64,
) -> Result<MotionGraph, MotionGraphAnalysisError> {
    let start = now_milliseconds();
    let measured = measure_motion_candidates(motion)?;
    let mut graph = measured.build(settings)?;
    let mut summary = graph
        .analysis_summary()
        .expect("measured graph summary")
        .clone();
    summary.analysis_milliseconds = (now_milliseconds() - start).max(0.0) as f32;
    graph = graph.with_analysis_summary(summary);
    Ok(graph)
}

/// Measure once, then re-rank the same member-aware data for diagnostic gate sweeps.
struct MeasuredMotionGraph {
    candidates: Vec<Vec<MotionJump>>,
    segment_features: Vec<MotionSegmentFeature>,
    analyzed_rows: usize,
}

impl MeasuredMotionGraph {
    fn build(
        &self,
        settings: MotionGraphSettings,
    ) -> Result<MotionGraph, MotionGraphAnalysisError> {
        let mut close_candidates = 0;
        let mut qualified = MotionJumpDistribution::default();
        let mut retained = MotionJumpDistribution::default();
        let nodes = self
            .candidates
            .iter()
            .enumerate()
            .map(|(source_segment, candidates)| {
                for jump in candidates {
                    if jump.discontinuity.is_close(settings) {
                        close_candidates += 1;
                        if settings.allows_target(source_segment, jump.target_segment) {
                            qualified.record(source_segment, jump.target_segment);
                        }
                    }
                }
                let jumps = rank_eligible_jumps(source_segment, candidates.clone(), settings);
                for jump in &jumps {
                    retained.record(source_segment, jump.target_segment);
                }
                MotionGraphNode {
                    source_segment,
                    jumps,
                }
            })
            .collect();
        let graph = MotionGraph::new(nodes, settings)
            .and_then(|graph| graph.with_segment_features(self.segment_features.clone()))
            .map_err(|error| MotionGraphAnalysisError::new(error.to_string()))?;
        let graph_summary = graph.summary();
        Ok(graph.with_analysis_summary(MotionGraphAnalysisSummary {
            analyzed_rows: self.analyzed_rows,
            source_segments: SOURCE_SEGMENT_COUNT,
            accepted_jumps: graph_summary.jump_count,
            zero_candidate_segments: SOURCE_SEGMENT_COUNT - graph_summary.segments_with_jumps,
            p95_limit: settings.p95_limit,
            ceiling_limit: settings.ceiling_limit,
            max_jump_candidates: settings.max_jump_candidates,
            min_jump_intervals: settings.min_jump_intervals,
            close_candidates,
            qualified,
            retained,
            analysis_milliseconds: 0.0,
        }))
    }
}

fn measure_motion_candidates(
    motion: &MergedMotion,
) -> Result<MeasuredMotionGraph, MotionGraphAnalysisError> {
    if MOTION_SAMPLE_COUNT != SOURCE_SEGMENT_COUNT + 1 {
        return Err(MotionGraphAnalysisError::new(
            "motion sample and graph segment counts disagree",
        ));
    }
    if !motion.source_duration_seconds.is_finite() || motion.source_duration_seconds <= 0.0 {
        return Err(MotionGraphAnalysisError::new(
            "motion graph requires a positive finite source duration",
        ));
    }

    let rows = select_lod0_rows(motion, DEFAULT_GRAPH_ANALYSIS_ROW_BUDGET)?;
    let trajectories = evaluate_trajectories(motion, &rows)?;
    let dt = motion.source_duration_seconds / SOURCE_SEGMENT_COUNT as f32;
    let kinematics = trajectories
        .iter()
        .map(|states| derive_kinematics(states, dt))
        .collect::<Result<Vec<_>, _>>()?;
    let scales = equivalent_step_scales(&kinematics, dt)?;
    let segment_features = aggregate_segment_features(&kinematics)?;

    let mut scratch: [Vec<f32>; 9] = std::array::from_fn(|_| Vec::with_capacity(rows.len()));
    let nodes = measure_candidate_pairs(|from_sample, to_sample| {
        for values in &mut scratch {
            values.clear();
        }
        for trajectory in &kinematics {
            let raw = boundary_metrics(trajectory, from_sample, to_sample);
            if raw.into_iter().any(|value| !value.is_finite()) {
                return Err(MotionGraphAnalysisError::new(
                    "motion graph candidate produced a non-finite discontinuity",
                ));
            }
            for (metric, (value, scale)) in scratch
                .iter_mut()
                .zip(raw.into_iter().zip(scales.into_iter()))
            {
                let normalized = value / scale;
                if !normalized.is_finite() {
                    return Err(MotionGraphAnalysisError::new(
                        "motion graph candidate normalization overflowed",
                    ));
                }
                metric.push(normalized);
            }
        }
        let p95 = std::array::from_fn(|index| percentile_95(&mut scratch[index]));
        let ceiling =
            std::array::from_fn(|index| scratch[index].iter().copied().fold(0.0_f32, f32::max));

        Ok(MotionDiscontinuity {
            p95: metric_set(p95),
            ceiling: metric_set(ceiling),
        })
    })?;
    Ok(MeasuredMotionGraph {
        candidates: nodes,
        segment_features,
        analyzed_rows: rows.len(),
    })
}

/// Distances are symmetric in boundary samples, but graph edges are directed.
/// Preserve candidate order/exclusions and cache only the measured distances.
fn measure_candidate_pairs(
    mut measure: impl FnMut(usize, usize) -> Result<MotionDiscontinuity, MotionGraphAnalysisError>,
) -> Result<Vec<Vec<MotionJump>>, MotionGraphAnalysisError> {
    let mut cache = vec![None; MOTION_SAMPLE_COUNT * MOTION_SAMPLE_COUNT];
    let mut nodes = Vec::with_capacity(SOURCE_SEGMENT_COUNT);
    for source in 0..SOURCE_SEGMENT_COUNT {
        let mut candidates = Vec::with_capacity(SOURCE_SEGMENT_COUNT - 2);
        for target in 0..SOURCE_SEGMENT_COUNT {
            if target == source || target == source + 1 {
                continue;
            }
            let from = source + 1;
            let key = from.min(target) * MOTION_SAMPLE_COUNT + from.max(target);
            let discontinuity = match cache[key] {
                Some(value) => value,
                None => {
                    let value = measure(from, target)?;
                    cache[key] = Some(value);
                    value
                }
            };
            candidates.push(MotionJump::unranked(target, discontinuity));
        }
        nodes.push(candidates);
    }
    Ok(nodes)
}

fn aggregate_segment_features(
    trajectories: &[Vec<KinematicSample>],
) -> Result<Vec<MotionSegmentFeature>, MotionGraphAnalysisError> {
    if trajectories.is_empty() {
        return Err(MotionGraphAnalysisError::new(
            "motion segment features require at least one trajectory",
        ));
    }
    (0..SOURCE_SEGMENT_COUNT)
        .map(|segment| {
            let mut displacement = [0.0_f32; 2];
            let mut magnitude = 0.0_f32;
            for trajectory in trajectories {
                let delta = vec3_sub(
                    trajectory[segment + 1].translation,
                    trajectory[segment].translation,
                );
                displacement[0] += delta[0];
                displacement[1] += delta[1];
                magnitude += (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
            }
            let inverse_count = 1.0 / trajectories.len() as f32;
            displacement[0] *= inverse_count;
            displacement[1] *= inverse_count;
            magnitude *= inverse_count;
            let direction_length =
                (displacement[0] * displacement[0] + displacement[1] * displacement[1]).sqrt();
            let horizontal_direction = if direction_length > f32::EPSILON {
                [
                    displacement[0] / direction_length,
                    displacement[1] / direction_length,
                ]
            } else {
                [0.0; 2]
            };
            if horizontal_direction
                .into_iter()
                .chain(std::iter::once(magnitude))
                .any(|value| !value.is_finite())
            {
                return Err(MotionGraphAnalysisError::new(
                    "motion segment feature aggregation produced a non-finite value",
                ));
            }
            Ok(MotionSegmentFeature {
                horizontal_direction,
                magnitude,
            })
        })
        .collect()
}

fn select_lod0_rows(
    motion: &MergedMotion,
    budget: usize,
) -> Result<Vec<usize>, MotionGraphAnalysisError> {
    if budget == 0 {
        return Err(MotionGraphAnalysisError::new(
            "motion graph row budget must be positive",
        ));
    }
    if motion.canonical.len() != motion.transform_ids.len()
        || motion.canonical.len() != motion.motion_channel_masks.len()
    {
        return Err(MotionGraphAnalysisError::new(
            "canonical and motion row metadata are not aligned",
        ));
    }
    let offsets = motion
        .member_offsets
        .first()
        .ok_or_else(|| MotionGraphAnalysisError::new("motion graph requires LoD0 offsets"))?;
    let counts = motion
        .member_counts
        .first()
        .ok_or_else(|| MotionGraphAnalysisError::new("motion graph requires LoD0 counts"))?;
    if offsets.len() != counts.len() || offsets.is_empty() {
        return Err(MotionGraphAnalysisError::new(
            "LoD0 member offsets and counts are missing or misaligned",
        ));
    }

    let mut members = Vec::new();
    for (&offset, &count) in offsets.iter().zip(counts) {
        if count == 0 {
            continue;
        }
        let start = offset as usize;
        let count = count as usize;
        let end = start
            .checked_add(count)
            .ok_or_else(|| MotionGraphAnalysisError::new("LoD0 row range overflow"))?;
        if end > motion.canonical.len() {
            return Err(MotionGraphAnalysisError::new(
                "LoD0 row range exceeds canonical rows",
            ));
        }
        members.push((start, count));
    }
    if members.is_empty() {
        return Err(MotionGraphAnalysisError::new(
            "LoD0 contains no motion rows",
        ));
    }
    if members.len() > budget {
        return Err(MotionGraphAnalysisError::new(
            "motion graph row budget cannot cover every nonempty LoD0 member",
        ));
    }

    let target_count = motion.canonical.len().min(budget);
    let mut allocations = vec![1_usize; members.len()];
    let mut remaining = target_count.saturating_sub(allocations.len());
    while remaining > 0 {
        let next = members
            .iter()
            .enumerate()
            .filter(|(index, (_, count))| allocations[*index] < *count)
            .max_by(
                |(left_index, (_, left_count)), (right_index, (_, right_count))| {
                    let left_remaining = *left_count - allocations[*left_index];
                    let right_remaining = *right_count - allocations[*right_index];
                    ((left_remaining as u64) * (*right_count as u64))
                        .cmp(&((right_remaining as u64) * (*left_count as u64)))
                        .then_with(|| right_index.cmp(left_index))
                },
            )
            .map(|(index, _)| index);
        let Some(index) = next else {
            break;
        };
        allocations[index] += 1;
        remaining -= 1;
    }

    let mut rows = Vec::with_capacity(target_count - remaining);
    for ((offset, count), allocation) in members.into_iter().zip(allocations) {
        if allocation == 1 {
            rows.push(offset + (count - 1) / 2);
        } else {
            for sample in 0..allocation {
                rows.push(offset + sample * (count - 1) / (allocation - 1));
            }
        }
    }
    rows.sort_unstable();
    rows.dedup();
    if rows.is_empty() {
        return Err(MotionGraphAnalysisError::new(
            "LoD0 row selection produced no rows",
        ));
    }
    Ok(rows)
}

enum SampledFrames {
    Legacy(Vec<BasisFrame>),
    Separate(Vec<BasisBanksFrame>),
}

fn evaluate_trajectories(
    motion: &MergedMotion,
    rows: &[usize],
) -> Result<Vec<Vec<MotionState>>, MotionGraphAnalysisError> {
    let frames = match &motion.data {
        MergedMotionData::Legacy { basis, .. } => SampledFrames::Legacy(
            (0..MOTION_SAMPLE_COUNT)
                .map(|sample| sample_basis(basis, sample as f32 / SOURCE_SEGMENT_COUNT as f32))
                .collect(),
        ),
        MergedMotionData::Separate { basis_banks, .. } => SampledFrames::Separate(
            (0..MOTION_SAMPLE_COUNT)
                .map(|sample| {
                    sample_basis_banks(basis_banks, sample as f32 / SOURCE_SEGMENT_COUNT as f32)
                })
                .collect(),
        ),
    };

    rows.iter()
        .map(|&row| {
            let canonical = *motion
                .canonical
                .get(row)
                .ok_or_else(|| MotionGraphAnalysisError::new("selected row is out of range"))?;
            let transform_id = *motion.transform_ids.get(row).ok_or_else(|| {
                MotionGraphAnalysisError::new("selected transform ID is out of range")
            })?;
            let row_mask = *motion.motion_channel_masks.get(row).ok_or_else(|| {
                MotionGraphAnalysisError::new("selected motion mask is out of range")
            })?;
            let states = match (&motion.data, &frames) {
                (
                    MergedMotionData::Legacy {
                        top_k,
                        basis_ids,
                        weights,
                        ..
                    },
                    SampledFrames::Legacy(frames),
                ) => {
                    let start = row.checked_mul(*top_k).ok_or_else(|| {
                        MotionGraphAnalysisError::new("legacy coefficient offset overflow")
                    })?;
                    let end = start.checked_add(*top_k).ok_or_else(|| {
                        MotionGraphAnalysisError::new("legacy coefficient range overflow")
                    })?;
                    let ids = basis_ids.get(start..end).ok_or_else(|| {
                        MotionGraphAnalysisError::new("legacy coefficients are not row-aligned")
                    })?;
                    let row_weights = weights.get(start..end).ok_or_else(|| {
                        MotionGraphAnalysisError::new("legacy weights are not row-aligned")
                    })?;
                    frames
                        .iter()
                        .map(|frame| {
                            evaluate_row_masked(
                                canonical,
                                frame,
                                ids,
                                row_weights,
                                transform_id,
                                row_mask,
                                ALL_MOTION_MASK,
                            )
                            .map_err(|error| MotionGraphAnalysisError::new(error.to_string()))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                (
                    MergedMotionData::Separate {
                        top_k,
                        basis_ids,
                        weights,
                        ..
                    },
                    SampledFrames::Separate(frames),
                ) => {
                    let coefficient_count = top_k.checked_mul(3).ok_or_else(|| {
                        MotionGraphAnalysisError::new("three-bank coefficient count overflow")
                    })?;
                    let start = row.checked_mul(coefficient_count).ok_or_else(|| {
                        MotionGraphAnalysisError::new("three-bank coefficient offset overflow")
                    })?;
                    let end = start.checked_add(coefficient_count).ok_or_else(|| {
                        MotionGraphAnalysisError::new("three-bank coefficient range overflow")
                    })?;
                    let ids = basis_ids.get(start..end).ok_or_else(|| {
                        MotionGraphAnalysisError::new("three-bank coefficients are not row-aligned")
                    })?;
                    let row_weights = weights.get(start..end).ok_or_else(|| {
                        MotionGraphAnalysisError::new("three-bank weights are not row-aligned")
                    })?;
                    frames
                        .iter()
                        .map(|frame| {
                            evaluate_row_banks_masked(
                                canonical,
                                frame,
                                ids,
                                row_weights,
                                *top_k,
                                transform_id,
                                row_mask,
                                ALL_MOTION_MASK,
                            )
                            .map_err(|error| MotionGraphAnalysisError::new(error.to_string()))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                _ => {
                    return Err(MotionGraphAnalysisError::new(
                        "sampled motion frame type does not match its archive data",
                    ));
                }
            };
            if states.iter().any(|state| !state_is_finite(state)) {
                return Err(MotionGraphAnalysisError::new(
                    "motion graph reconstruction produced a non-finite TRS state",
                ));
            }
            Ok(states)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct KinematicSample {
    translation: [f32; 3],
    translation_velocity: [f32; 3],
    translation_acceleration: [f32; 3],
    rotation: [f32; 4],
    rotation_velocity: [f32; 3],
    rotation_acceleration: [f32; 3],
    log_scale: [f32; 3],
    scale_velocity: [f32; 3],
    scale_acceleration: [f32; 3],
}

fn derive_kinematics(
    states: &[MotionState],
    dt: f32,
) -> Result<Vec<KinematicSample>, MotionGraphAnalysisError> {
    if states.len() != MOTION_SAMPLE_COUNT || !dt.is_finite() || dt <= 0.0 {
        return Err(MotionGraphAnalysisError::new(
            "motion graph trajectory has an invalid sample count or timestep",
        ));
    }
    let translations = states
        .iter()
        .map(|state| state.position)
        .collect::<Vec<_>>();
    let scales = states
        .iter()
        .map(|state| state.log_scale)
        .collect::<Vec<_>>();
    let translation_velocity = first_derivative(&translations, dt);
    let translation_acceleration = first_derivative(&translation_velocity, dt);
    let scale_velocity = first_derivative(&scales, dt);
    let scale_acceleration = first_derivative(&scale_velocity, dt);

    let segment_angular_velocity = states
        .windows(2)
        .map(|pair| {
            vec3_scale(
                rotation_vector_between(pair[0].rotation, pair[1].rotation),
                1.0 / dt,
            )
        })
        .collect::<Vec<_>>();
    let mut rotation_velocity = Vec::with_capacity(MOTION_SAMPLE_COUNT);
    rotation_velocity.push(segment_angular_velocity[0]);
    rotation_velocity.extend(
        segment_angular_velocity
            .windows(2)
            .map(|pair| vec3_scale(vec3_add(pair[0], pair[1]), 0.5)),
    );
    rotation_velocity.push(*segment_angular_velocity.last().unwrap());
    let rotation_acceleration = first_derivative(&rotation_velocity, dt);

    let kinematics = (0..MOTION_SAMPLE_COUNT)
        .map(|sample| KinematicSample {
            translation: translations[sample],
            translation_velocity: translation_velocity[sample],
            translation_acceleration: translation_acceleration[sample],
            rotation: states[sample].rotation,
            rotation_velocity: rotation_velocity[sample],
            rotation_acceleration: rotation_acceleration[sample],
            log_scale: scales[sample],
            scale_velocity: scale_velocity[sample],
            scale_acceleration: scale_acceleration[sample],
        })
        .collect::<Vec<_>>();
    if kinematics.iter().any(|sample| !kinematic_is_finite(sample)) {
        return Err(MotionGraphAnalysisError::new(
            "motion graph derivative evaluation produced a non-finite value",
        ));
    }
    Ok(kinematics)
}

fn equivalent_step_scales(
    trajectories: &[Vec<KinematicSample>],
    dt: f32,
) -> Result<[f32; 9], MotionGraphAnalysisError> {
    let mut ordinary: [Vec<f32>; 9] = std::array::from_fn(|_| {
        Vec::with_capacity(trajectories.len().saturating_mul(SOURCE_SEGMENT_COUNT))
    });
    for trajectory in trajectories {
        for sample in 0..SOURCE_SEGMENT_COUNT {
            let metrics = boundary_metrics(trajectory, sample, sample + 1);
            if metrics.into_iter().any(|value| !value.is_finite()) {
                return Err(MotionGraphAnalysisError::new(
                    "ordinary source-step metric is non-finite",
                ));
            }
            for (values, metric) in ordinary.iter_mut().zip(metrics) {
                values.push(metric);
            }
        }
    }
    let floors = [
        TRANSLATION_SCALE_FLOOR,
        TRANSLATION_SCALE_FLOOR / dt,
        TRANSLATION_SCALE_FLOOR / (dt * dt),
        ROTATION_SCALE_FLOOR,
        ROTATION_SCALE_FLOOR / dt,
        ROTATION_SCALE_FLOOR / (dt * dt),
        LOG_SCALE_FLOOR,
        LOG_SCALE_FLOOR / dt,
        LOG_SCALE_FLOOR / (dt * dt),
    ];
    let scales =
        std::array::from_fn(|index| percentile_95(&mut ordinary[index]).max(floors[index]));
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(MotionGraphAnalysisError::new(
            "motion graph normalization produced an invalid scale",
        ));
    }
    Ok(scales)
}

fn boundary_metrics(
    trajectory: &[KinematicSample],
    from_sample: usize,
    to_sample: usize,
) -> [f32; 9] {
    let from = trajectory[from_sample];
    let to = trajectory[to_sample];
    [
        vec3_norm(vec3_sub(from.translation, to.translation)),
        vec3_norm(vec3_sub(from.translation_velocity, to.translation_velocity)),
        vec3_norm(vec3_sub(
            from.translation_acceleration,
            to.translation_acceleration,
        )),
        quaternion_geodesic(from.rotation, to.rotation),
        vec3_norm(vec3_sub(from.rotation_velocity, to.rotation_velocity)),
        vec3_norm(vec3_sub(
            from.rotation_acceleration,
            to.rotation_acceleration,
        )),
        vec3_norm(vec3_sub(from.log_scale, to.log_scale)),
        vec3_norm(vec3_sub(from.scale_velocity, to.scale_velocity)),
        vec3_norm(vec3_sub(from.scale_acceleration, to.scale_acceleration)),
    ]
}

fn metric_set(values: [f32; 9]) -> MotionMetricSet {
    MotionMetricSet {
        translation_position: values[0],
        translation_velocity: values[1],
        translation_acceleration: values[2],
        rotation_position: values[3],
        rotation_velocity: values[4],
        rotation_acceleration: values[5],
        scale_position: values[6],
        scale_velocity: values[7],
        scale_acceleration: values[8],
    }
}

fn percentile_95(values: &mut [f32]) -> f32 {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return f32::NAN;
    }
    let index = ((values.len() * 95).div_ceil(100)).saturating_sub(1);
    let (_, value, _) = values.select_nth_unstable_by(index, f32::total_cmp);
    *value
}

fn first_derivative(values: &[[f32; 3]], dt: f32) -> Vec<[f32; 3]> {
    (0..values.len())
        .map(|index| {
            if index == 0 {
                vec3_scale(vec3_sub(values[1], values[0]), 1.0 / dt)
            } else if index + 1 == values.len() {
                vec3_scale(
                    vec3_sub(values[values.len() - 1], values[values.len() - 2]),
                    1.0 / dt,
                )
            } else {
                vec3_scale(vec3_sub(values[index + 1], values[index - 1]), 0.5 / dt)
            }
        })
        .collect()
}

fn rotation_vector_between(from: [f32; 4], mut to: [f32; 4]) -> [f32; 3] {
    if quaternion_dot(from, to) < 0.0 {
        to = to.map(|value| -value);
    }
    let relative = quaternion_multiply(quaternion_conjugate(from), to);
    let vector = [relative[1], relative[2], relative[3]];
    let vector_norm = vec3_norm(vector);
    if vector_norm <= f32::EPSILON {
        [0.0; 3]
    } else {
        let angle = 2.0 * vector_norm.atan2(relative[0].clamp(-1.0, 1.0));
        vec3_scale(vector, angle / vector_norm)
    }
}

fn quaternion_geodesic(left: [f32; 4], right: [f32; 4]) -> f32 {
    2.0 * quaternion_dot(left, right).abs().clamp(-1.0, 1.0).acos()
}

fn quaternion_dot(left: [f32; 4], right: [f32; 4]) -> f32 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn quaternion_conjugate(value: [f32; 4]) -> [f32; 4] {
    [value[0], -value[1], -value[2], -value[3]]
}

fn quaternion_multiply(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    [
        left[0] * right[0] - left[1] * right[1] - left[2] * right[2] - left[3] * right[3],
        left[0] * right[1] + left[1] * right[0] + left[2] * right[3] - left[3] * right[2],
        left[0] * right[2] - left[1] * right[3] + left[2] * right[0] + left[3] * right[1],
        left[0] * right[3] + left[1] * right[2] - left[2] * right[1] + left[3] * right[0],
    ]
}

fn vec3_add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn vec3_sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|index| left[index] - right[index])
}

fn vec3_scale(value: [f32; 3], scale: f32) -> [f32; 3] {
    value.map(|component| component * scale)
}

fn vec3_norm(value: [f32; 3]) -> f32 {
    value
        .into_iter()
        .map(|component| component * component)
        .sum::<f32>()
        .sqrt()
}

fn state_is_finite(state: &MotionState) -> bool {
    state.position.into_iter().all(f32::is_finite)
        && state.log_scale.into_iter().all(f32::is_finite)
        && state.rotation.into_iter().all(f32::is_finite)
}

fn kinematic_is_finite(sample: &KinematicSample) -> bool {
    sample.translation.into_iter().all(f32::is_finite)
        && sample.translation_velocity.into_iter().all(f32::is_finite)
        && sample
            .translation_acceleration
            .into_iter()
            .all(f32::is_finite)
        && sample.rotation.into_iter().all(f32::is_finite)
        && sample.rotation_velocity.into_iter().all(f32::is_finite)
        && sample.rotation_acceleration.into_iter().all(f32::is_finite)
        && sample.log_scale.into_iter().all(f32::is_finite)
        && sample.scale_velocity.into_iter().all(f32::is_finite)
        && sample.scale_acceleration.into_iter().all(f32::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_archive::{BasisBank, BasisBanks};
    use crate::motion::{
        ALL_MOTION_MASK, CanonicalGaussian, MOTION_SAMPLE_COUNT, MergedMotion, MergedMotionData,
    };
    use crate::motion_graph::{MotionGraphSettings, SOURCE_SEGMENT_COUNT};
    use crate::scene_archive::DynamicArchiveSummary;

    fn static_motion() -> MergedMotion {
        MergedMotion {
            data: MergedMotionData::Legacy {
                basis: BasisBank {
                    basis_count: 1,
                    values: vec![0.0; MOTION_SAMPLE_COUNT * 9],
                },
                top_k: 1,
                basis_ids: vec![0],
                weights: vec![1.0],
            },
            canonical: vec![CanonicalGaussian {
                position: [0.0; 3],
                log_scale: [0.0; 3],
                rotation: [1.0, 0.0, 0.0, 0.0],
            }],
            transform_ids: vec![0],
            motion_channel_masks: vec![ALL_MOTION_MASK],
            source_duration_seconds: 1.0,
            summary: DynamicArchiveSummary {
                schema_version: 2,
                tile_count: 1,
                lod_count: 1,
                basis_count: 1,
                top_k: 1,
                total_rows: 1,
                backend: "static-analysis-fixture".into(),
            },
            member_offsets: vec![vec![0]],
            member_counts: vec![vec![1]],
        }
    }

    fn static_motion_with_members(counts: &[u32]) -> MergedMotion {
        let row_count = counts.iter().map(|count| *count as usize).sum::<usize>();
        let mut offsets = Vec::with_capacity(counts.len());
        let mut offset = 0_u32;
        for &count in counts {
            offsets.push(offset);
            offset += count;
        }
        let canonical = CanonicalGaussian {
            position: [0.0; 3],
            log_scale: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        MergedMotion {
            data: MergedMotionData::Legacy {
                basis: BasisBank {
                    basis_count: 1,
                    values: vec![0.0; MOTION_SAMPLE_COUNT * 9],
                },
                top_k: 1,
                basis_ids: vec![0; row_count],
                weights: vec![1.0; row_count],
            },
            canonical: vec![canonical; row_count],
            transform_ids: vec![0; row_count],
            motion_channel_masks: vec![ALL_MOTION_MASK; row_count],
            source_duration_seconds: 1.0,
            summary: DynamicArchiveSummary {
                schema_version: 2,
                tile_count: counts.len(),
                lod_count: 1,
                basis_count: 1,
                top_k: 1,
                total_rows: row_count,
                backend: "row-selection-fixture".into(),
            },
            member_offsets: vec![offsets],
            member_counts: vec![counts.to_vec()],
        }
    }

    #[test]
    fn symmetric_pair_cache_preserves_all_directed_measurements() {
        // Changing boundary indices to segment indices, skipping candidates, or
        // caching direction-dependent decisions must break this comparison.
        let mut motion = static_motion_with_members(&[11, 12]);
        if let MergedMotionData::Legacy { basis, weights, .. } = &mut motion.data {
            basis.values = (0..MOTION_SAMPLE_COUNT)
                .flat_map(|t| {
                    let x = t as f32 * 0.07;
                    [
                        x.sin(),
                        x * x,
                        -x,
                        0.0,
                        x * 0.2,
                        0.0,
                        x * 0.1,
                        -x.cos(),
                        x * x * 0.01,
                    ]
                })
                .collect();
            for (row, weight) in weights.iter_mut().enumerate() {
                *weight = 0.1 + row as f32 * 0.09;
            }
        }
        for (row, mask) in motion.motion_channel_masks.iter_mut().enumerate() {
            *mask = [1, 3, 7][row % 3];
        }
        let rows = select_lod0_rows(&motion, DEFAULT_GRAPH_ANALYSIS_ROW_BUDGET).unwrap();
        let dt = motion.source_duration_seconds / SOURCE_SEGMENT_COUNT as f32;
        let trajectories = evaluate_trajectories(&motion, &rows)
            .unwrap()
            .iter()
            .map(|states| derive_kinematics(states, dt).unwrap())
            .collect::<Vec<_>>();
        let scales = equivalent_step_scales(&trajectories, dt).unwrap();
        let reference = |from, to| {
            let mut values: [Vec<f32>; 9] = std::array::from_fn(|_| Vec::new());
            for trajectory in &trajectories {
                for (i, raw) in boundary_metrics(trajectory, from, to)
                    .into_iter()
                    .enumerate()
                {
                    values[i].push(raw / scales[i]);
                }
            }
            for v in &mut values {
                v.sort_by(f32::total_cmp);
            }
            MotionDiscontinuity {
                p95: metric_set(std::array::from_fn(|i| values[i][21])),
                ceiling: metric_set(std::array::from_fn(|i| values[i][22])),
            }
        };
        let mut evaluations = 0;
        let candidates = measure_candidate_pairs(|from, to| {
            evaluations += 1;
            Ok(reference(from, to))
        })
        .unwrap();
        assert_eq!(evaluations, 2773);
        assert_eq!(
            measure_motion_candidates(&motion).unwrap().candidates,
            candidates
        );
        assert_eq!(candidates.iter().map(Vec::len).sum::<usize>(), 5329);
        for (source, jumps) in candidates.iter().enumerate() {
            let expected = (0..SOURCE_SEGMENT_COUNT)
                .filter(|&target| target != source && target != source + 1)
                .map(|target| MotionJump::unranked(target, reference(source + 1, target)))
                .collect::<Vec<_>>();
            assert_eq!(*jumps, expected);
            assert_eq!(
                rank_eligible_jumps(source, jumps.clone(), MotionGraphSettings::default()),
                rank_eligible_jumps(source, expected, MotionGraphSettings::default()),
            );
        }
    }

    #[test]
    fn symmetric_pair_cache_propagates_measurement_errors() {
        let result = measure_candidate_pairs(|_, _| {
            Err(MotionGraphAnalysisError::new("invalid measurement"))
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("invalid measurement")
        );
    }

    #[test]
    fn analysis_builds_a_valid_bounded_graph_from_existing_evaluators() {
        let graph = build_motion_graph(&static_motion(), MotionGraphSettings::default())
            .expect("static finite motion should produce a graph");

        assert_eq!(graph.nodes().len(), SOURCE_SEGMENT_COUNT);
        assert!(graph.nodes().iter().all(|node| node.jumps.len() <= 3));
        let summary = graph.analysis_summary().unwrap();
        assert!(summary.close_candidates > summary.qualified.forward + summary.qualified.backward);
        assert!(summary.qualified.forward + summary.qualified.backward > summary.accepted_jumps);
        assert_eq!(
            summary.retained.forward + summary.retained.backward,
            summary.accepted_jumps
        );
        assert!(summary.retained.min_intervals >= 6);
        assert!(summary.retained.backward > 0);
    }

    /// Opt-in, CPU-only diagnostic. Uses exactly the browser's loader, motion
    /// evaluator and 2,048-row member-aware sample; no original-source substitute.
    #[test]
    #[ignore = "requires GSWT_GRAPH_ARCHIVE and loads a real archive"]
    fn archive_gate_sweep() {
        let path = std::env::var("GSWT_GRAPH_ARCHIVE").expect("set GSWT_GRAPH_ARCHIVE");
        let bytes = std::fs::read(&path).expect("read archive");
        let mut loaded =
            crate::scene_archive::load_archive_bytes(bytes).expect("renderer archive loader");
        crate::wangtile::normalize_tile_heights(&mut loaded.scenes, loaded.dynamic.as_mut())
            .expect("renderer height normalization");
        let motion =
            crate::motion::merge_dynamic_asset(loaded.dynamic.expect("dynamic asset")).unwrap();
        drop(loaded.scenes);
        println!(
            "Archive: {path}\nDuration: {} s; LoD0 member counts: {:?}",
            motion.source_duration_seconds, motion.member_counts[0]
        );
        let measured = measure_motion_candidates(&motion).unwrap();
        println!(
            "Analyzed {} rows. Distance is from segment exit; minimum = 6 intervals.",
            measured.analyzed_rows
        );
        println!(
            "p95/ceiling | close (all gaps) | qualified F/B | retained F/B | covered segments | retained gap min/mean/max"
        );
        let gates = [
            (3.0, 12.0),
            (4.0, 16.0),
            (5.0, 20.0),
            (6.0, 24.0),
            (8.0, 32.0),
            (10.0, 40.0),
            (12.0, 48.0),
            (16.0, 64.0),
            (20.0, 80.0),
        ]
        .into_iter()
        .chain([3.0, 6.0, 8.0, 10.0, 12.0].into_iter().flat_map(|p95| {
            [48.0, 64.0, 96.0, 128.0]
                .into_iter()
                .map(move |ceiling| (p95, ceiling))
        }));
        for (p95_limit, ceiling_limit) in gates {
            let settings = MotionGraphSettings {
                p95_limit,
                ceiling_limit,
                ..MotionGraphSettings::default()
            };
            let graph = measured.build(settings).unwrap();
            let s = graph.analysis_summary().unwrap();
            println!(
                "{p95_limit}/{ceiling_limit} | {} | {}/{} | {}/{} | {}/74 | {}/{:.1}/{}",
                s.close_candidates,
                s.qualified.forward,
                s.qualified.backward,
                s.retained.forward,
                s.retained.backward,
                graph.summary().segments_with_jumps,
                s.retained.min_intervals,
                s.retained.mean_intervals(),
                s.retained.max_intervals
            );
            assert!(s.retained.min_intervals == 0 || s.retained.min_intervals >= 6);
            assert!(graph.nodes().iter().all(|node| node.jumps.len() <= 3));
        }
        for (p95_limit, ceiling_limit) in [(6.0, 64.0), (8.0, 64.0), (12.0, 48.0), (12.0, 64.0)] {
            let graph = measured
                .build(MotionGraphSettings {
                    p95_limit,
                    ceiling_limit,
                    ..MotionGraphSettings::default()
                })
                .unwrap();
            println!(
                "Example retained edges at {p95_limit}/{ceiling_limit} (source -> target; gap; score):"
            );
            for node in graph
                .nodes()
                .iter()
                .filter(|node| !node.jumps.is_empty())
                .take(8)
            {
                for jump in &node.jumps {
                    println!(
                        "{} -> {}; gap {}; score {:.2}; p95 {:?}; ceiling {:?}",
                        node.source_segment,
                        jump.target_segment,
                        (node.source_segment + 1).abs_diff(jump.target_segment),
                        jump.score,
                        jump.discontinuity.p95,
                        jump.discontinuity.ceiling
                    );
                }
            }
        }
        let graph = measured.build(MotionGraphSettings::default()).unwrap();
        println!("Default retained edges (source segment -> target entry sample):");
        for node in graph.nodes().iter().filter(|node| !node.jumps.is_empty()) {
            println!(
                "{} -> {:?}",
                node.source_segment,
                node.jumps
                    .iter()
                    .map(|jump| (jump.target_segment, jump.score))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn analysis_retains_horizontal_direction_for_each_source_segment() {
        let mut motion = static_motion();
        let MergedMotionData::Legacy { basis, .. } = &mut motion.data else {
            unreachable!();
        };
        for sample in 0..MOTION_SAMPLE_COUNT {
            basis.values[sample * 9] = sample as f32 / SOURCE_SEGMENT_COUNT as f32;
        }

        let graph = build_motion_graph(&motion, MotionGraphSettings::default()).unwrap();

        assert_eq!(graph.segment_features().len(), SOURCE_SEGMENT_COUNT);
        assert!(graph.segment_features().iter().all(|feature| {
            feature.horizontal_direction[0] > 0.99
                && feature.horizontal_direction[1].abs() < 1.0e-6
                && feature.magnitude > 0.0
        }));
    }

    #[test]
    fn analysis_uses_the_supplied_monotonic_clock() {
        let mut timestamps = [100.0, 107.5].into_iter();
        let graph =
            build_motion_graph_with_clock(&static_motion(), MotionGraphSettings::default(), || {
                timestamps.next().expect("analysis reads the clock twice")
            })
            .expect("static finite motion should produce a graph");

        assert_eq!(
            graph
                .analysis_summary()
                .expect("analysis summary")
                .analysis_milliseconds,
            7.5
        );
        assert!(timestamps.next().is_none());
    }

    #[test]
    fn robust_aggregation_never_hides_a_nonfinite_metric() {
        let mut values = vec![0.0; 20];
        values[0] = f32::NAN;

        assert!(percentile_95(&mut values).is_nan());
    }

    #[test]
    fn lod0_row_selection_is_deterministic_bounded_and_covers_each_member() {
        let motion = static_motion_with_members(&[5, 7, 3]);

        let first = select_lod0_rows(&motion, 7).unwrap();
        let second = select_lod0_rows(&motion, 7).unwrap();

        assert_eq!(first, second);
        assert!(first.len() <= 7);
        assert!(first.iter().any(|row| *row < 5));
        assert!(first.iter().any(|row| (5..12).contains(row)));
        assert!(first.iter().any(|row| (12..15).contains(row)));
    }

    #[test]
    fn lod0_row_selection_rejects_out_of_range_members() {
        let mut motion = static_motion_with_members(&[1]);
        motion.member_offsets[0][0] = 2;

        assert!(select_lod0_rows(&motion, 8).is_err());
    }

    #[test]
    fn analysis_supports_separate_trs_banks() {
        let zero_bank = || BasisBank {
            basis_count: 1,
            values: vec![0.0; MOTION_SAMPLE_COUNT * 3],
        };
        let mut motion = static_motion();
        motion.summary.schema_version = 3;
        motion.data = MergedMotionData::Separate {
            basis_banks: BasisBanks {
                translation: zero_bank(),
                rotation: zero_bank(),
                scale: zero_bank(),
            },
            top_k: 1,
            basis_ids: vec![0; 3],
            weights: vec![1.0; 3],
        };

        let graph = build_motion_graph(&motion, MotionGraphSettings::default()).unwrap();

        assert_eq!(graph.nodes().len(), SOURCE_SEGMENT_COUNT);
    }
}
