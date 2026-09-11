pub const SOURCE_SEGMENT_COUNT: usize = 74;
pub const DEFAULT_MAX_JUMP_CANDIDATES: usize = 3;
pub const DEFAULT_MIN_JUMP_INTERVALS: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionMetricSet {
    pub translation_position: f32,
    pub translation_velocity: f32,
    pub translation_acceleration: f32,
    pub rotation_position: f32,
    pub rotation_velocity: f32,
    pub rotation_acceleration: f32,
    pub scale_position: f32,
    pub scale_velocity: f32,
    pub scale_acceleration: f32,
}

impl MotionMetricSet {
    fn uniform(value: f32) -> Self {
        Self {
            translation_position: value,
            translation_velocity: value,
            translation_acceleration: value,
            rotation_position: value,
            rotation_velocity: value,
            rotation_acceleration: value,
            scale_position: value,
            scale_velocity: value,
            scale_acceleration: value,
        }
    }

    fn values(self) -> [f32; 9] {
        [
            self.translation_position,
            self.translation_velocity,
            self.translation_acceleration,
            self.rotation_position,
            self.rotation_velocity,
            self.rotation_acceleration,
            self.scale_position,
            self.scale_velocity,
            self.scale_acceleration,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionDiscontinuity {
    pub p95: MotionMetricSet,
    pub ceiling: MotionMetricSet,
}

impl MotionDiscontinuity {
    pub fn uniform(p95: f32, ceiling: f32) -> Self {
        Self {
            p95: MotionMetricSet::uniform(p95),
            ceiling: MotionMetricSet::uniform(ceiling),
        }
    }

    pub fn is_close(self, settings: MotionGraphSettings) -> bool {
        settings.is_valid()
            && self
                .p95
                .values()
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0 && value <= settings.p95_limit)
            && self
                .ceiling
                .values()
                .into_iter()
                .all(|value| value.is_finite() && value >= 0.0 && value <= settings.ceiling_limit)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionGraphSettings {
    pub p95_limit: f32,
    pub ceiling_limit: f32,
    pub max_jump_candidates: usize,
    /// Distance from the source exit sample, not from the segment's entry.
    pub min_jump_intervals: usize,
}

impl MotionGraphSettings {
    fn is_valid(self) -> bool {
        self.p95_limit.is_finite()
            && self.p95_limit >= 0.0
            && self.ceiling_limit.is_finite()
            && self.ceiling_limit >= 0.0
            && self.max_jump_candidates <= DEFAULT_MAX_JUMP_CANDIDATES
            && (1..=SOURCE_SEGMENT_COUNT).contains(&self.min_jump_intervals)
    }

    pub fn allows_target(self, source: usize, target: usize) -> bool {
        source < SOURCE_SEGMENT_COUNT
            && target < SOURCE_SEGMENT_COUNT
            && target != source
            && (source + 1).abs_diff(target) >= self.min_jump_intervals
    }
}

impl Default for MotionGraphSettings {
    fn default() -> Self {
        Self {
            // Trial calibration from the constructed shrub_sorrel archive;
            // numerical qualification is not a guarantee of visual smoothness.
            p95_limit: 12.0,
            ceiling_limit: 48.0,
            max_jump_candidates: DEFAULT_MAX_JUMP_CANDIDATES,
            min_jump_intervals: DEFAULT_MIN_JUMP_INTERVALS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionJump {
    pub target_segment: usize,
    pub score: f32,
    pub discontinuity: MotionDiscontinuity,
}

impl MotionJump {
    pub fn unranked(target_segment: usize, discontinuity: MotionDiscontinuity) -> Self {
        Self {
            target_segment,
            score: f32::INFINITY,
            discontinuity,
        }
    }
}

pub fn rank_eligible_jumps(
    source_segment: usize,
    candidates: Vec<MotionJump>,
    settings: MotionGraphSettings,
) -> Vec<MotionJump> {
    if !settings.is_valid() {
        return Vec::new();
    }

    let mut eligible = candidates
        .into_iter()
        .filter(|jump| {
            settings.allows_target(source_segment, jump.target_segment)
                && jump.discontinuity.is_close(settings)
        })
        .map(|mut jump| {
            jump.score = discontinuity_score(jump.discontinuity);
            jump
        })
        .collect::<Vec<_>>();

    eligible.sort_by(|left, right| {
        left.score
            .total_cmp(&right.score)
            .then_with(|| left.target_segment.cmp(&right.target_segment))
    });
    let mut selected: Vec<MotionJump> = Vec::with_capacity(settings.max_jump_candidates);
    // Reserve representation of both time directions when the budget permits.
    // Keep the best single candidate if the requested budget is only one.
    if settings.max_jump_candidates >= 2 {
        for backward in [true, false] {
            if let Some(jump) = eligible
                .iter()
                .find(|jump| (jump.target_segment < source_segment + 1) == backward)
            {
                selected.push(*jump);
            }
        }
    }
    for jump in eligible {
        if selected.len() >= settings.max_jump_candidates {
            break;
        }
        if selected.iter().all(|other| {
            other.target_segment.abs_diff(jump.target_segment) >= settings.min_jump_intervals
        }) {
            selected.push(jump);
        }
    }
    // Graph storage remains score ordered; diversity changes membership, not this contract.
    selected.sort_by(|left, right| {
        left.score
            .total_cmp(&right.score)
            .then_with(|| left.target_segment.cmp(&right.target_segment))
    });
    selected
}

fn discontinuity_score(discontinuity: MotionDiscontinuity) -> f32 {
    let p95_sum = discontinuity.p95.values().into_iter().sum::<f32>();
    let ceiling_sum = discontinuity.ceiling.values().into_iter().sum::<f32>();
    (p95_sum + ceiling_sum) / 18.0
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotionGraphNode {
    pub source_segment: usize,
    pub jumps: Vec<MotionJump>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionSegmentFeature {
    pub horizontal_direction: [f32; 2],
    pub magnitude: f32,
}

impl MotionSegmentFeature {
    fn is_valid(self) -> bool {
        let length_squared = self.horizontal_direction[0] * self.horizontal_direction[0]
            + self.horizontal_direction[1] * self.horizontal_direction[1];
        self.horizontal_direction.into_iter().all(f32::is_finite)
            && length_squared <= 1.0 + f32::EPSILON
            && self.magnitude.is_finite()
            && self.magnitude >= 0.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotionGraphSummary {
    pub segment_count: usize,
    pub segments_with_jumps: usize,
    pub jump_count: usize,
    pub max_jumps_per_segment: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MotionGraphAnalysisSummary {
    pub analyzed_rows: usize,
    pub source_segments: usize,
    pub accepted_jumps: usize,
    pub zero_candidate_segments: usize,
    pub p95_limit: f32,
    pub ceiling_limit: f32,
    pub max_jump_candidates: usize,
    pub min_jump_intervals: usize,
    pub close_candidates: usize,
    pub qualified: MotionJumpDistribution,
    pub retained: MotionJumpDistribution,
    pub analysis_milliseconds: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MotionJumpDistribution {
    pub forward: usize,
    pub backward: usize,
    pub min_intervals: usize,
    pub max_intervals: usize,
    pub total_intervals: usize,
}

impl MotionJumpDistribution {
    pub fn record(&mut self, source: usize, target: usize) {
        let distance = (source + 1).abs_diff(target);
        if self.forward + self.backward == 0 {
            self.min_intervals = distance;
        } else {
            self.min_intervals = self.min_intervals.min(distance);
        }
        self.max_intervals = self.max_intervals.max(distance);
        self.total_intervals += distance;
        if target < source + 1 {
            self.backward += 1;
        } else {
            self.forward += 1;
        }
    }

    pub fn mean_intervals(&self) -> f32 {
        self.total_intervals as f32 / (self.forward + self.backward).max(1) as f32
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotionGraph {
    nodes: Vec<MotionGraphNode>,
    segment_features: Vec<MotionSegmentFeature>,
    settings: MotionGraphSettings,
    summary: MotionGraphSummary,
    analysis_summary: Option<MotionGraphAnalysisSummary>,
}

impl MotionGraph {
    pub fn new(
        nodes: Vec<MotionGraphNode>,
        settings: MotionGraphSettings,
    ) -> Result<Self, MotionGraphError> {
        validate_graph(&nodes, settings)?;
        let summary = MotionGraphSummary {
            segment_count: nodes.len(),
            segments_with_jumps: nodes.iter().filter(|node| !node.jumps.is_empty()).count(),
            jump_count: nodes.iter().map(|node| node.jumps.len()).sum(),
            max_jumps_per_segment: nodes.iter().map(|node| node.jumps.len()).max().unwrap_or(0),
        };
        Ok(Self {
            segment_features: vec![MotionSegmentFeature::default(); nodes.len()],
            nodes,
            settings,
            summary,
            analysis_summary: None,
        })
    }

    pub fn nodes(&self) -> &[MotionGraphNode] {
        &self.nodes
    }

    pub fn with_segment_features(
        mut self,
        features: Vec<MotionSegmentFeature>,
    ) -> Result<Self, MotionGraphError> {
        if features.len() != SOURCE_SEGMENT_COUNT
            || features.iter().copied().any(|feature| !feature.is_valid())
        {
            return Err(MotionGraphError::new(
                "motion graph segment features must be finite and aligned with all source segments",
            ));
        }
        self.segment_features = features;
        Ok(self)
    }

    pub fn segment_features(&self) -> &[MotionSegmentFeature] {
        &self.segment_features
    }

    pub fn direction_coverage(
        &self,
        direction: [f32; 2],
        start_segment: usize,
        end_segment: usize,
    ) -> Result<f32, MotionGraphError> {
        let direction_length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if direction.into_iter().any(|value| !value.is_finite())
            || direction_length > 1.0 + f32::EPSILON
            || start_segment >= end_segment
            || end_segment > SOURCE_SEGMENT_COUNT
        {
            return Err(MotionGraphError::new(
                "direction coverage query is outside the source graph",
            ));
        }
        if direction_length <= f32::EPSILON {
            return Ok(1.0);
        }
        let normalized = [
            direction[0] / direction_length,
            direction[1] / direction_length,
        ];
        let reference_magnitude = self.maximum_segment_magnitude();
        if reference_magnitude <= f32::EPSILON {
            return Ok(0.0);
        }
        let covered = self.segment_features[start_segment..end_segment]
            .iter()
            .map(|feature| {
                let alignment = (feature.horizontal_direction[0] * normalized[0]
                    + feature.horizontal_direction[1] * normalized[1])
                    .max(0.0);
                alignment * (feature.magnitude / reference_magnitude).clamp(0.0, 1.0)
            })
            .sum::<f32>();
        Ok((covered / (end_segment - start_segment) as f32).clamp(0.0, 1.0))
    }

    pub fn maximum_segment_magnitude(&self) -> f32 {
        self.segment_features
            .iter()
            .map(|feature| feature.magnitude)
            .fold(0.0_f32, f32::max)
    }

    pub fn settings(&self) -> MotionGraphSettings {
        self.settings
    }

    pub fn validate(&self) -> Result<(), MotionGraphError> {
        validate_graph(&self.nodes, self.settings)
    }

    pub fn summary(&self) -> MotionGraphSummary {
        self.summary
    }

    pub fn with_analysis_summary(mut self, summary: MotionGraphAnalysisSummary) -> Self {
        self.analysis_summary = Some(summary);
        self
    }

    pub fn analysis_summary(&self) -> Option<&MotionGraphAnalysisSummary> {
        self.analysis_summary.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionGraphError {
    message: String,
}

impl MotionGraphError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for MotionGraphError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MotionGraphError {}

fn validate_graph(
    nodes: &[MotionGraphNode],
    settings: MotionGraphSettings,
) -> Result<(), MotionGraphError> {
    if !settings.is_valid() {
        return Err(MotionGraphError::new("motion graph settings are invalid"));
    }
    if nodes.len() != SOURCE_SEGMENT_COUNT {
        return Err(MotionGraphError::new(format!(
            "motion graph must contain {SOURCE_SEGMENT_COUNT} source segments, found {}",
            nodes.len()
        )));
    }

    for (expected_source, node) in nodes.iter().enumerate() {
        if node.source_segment != expected_source {
            return Err(MotionGraphError::new(format!(
                "motion graph node {expected_source} declares source segment {}",
                node.source_segment
            )));
        }
        if node.jumps.len() > settings.max_jump_candidates
            || node.jumps.len() > DEFAULT_MAX_JUMP_CANDIDATES
        {
            return Err(MotionGraphError::new(format!(
                "source segment {expected_source} has too many stochastic jumps"
            )));
        }

        let mut targets = std::collections::BTreeSet::new();
        let mut previous_order: Option<(f32, usize)> = None;
        for jump in &node.jumps {
            if !settings.allows_target(expected_source, jump.target_segment) {
                return Err(MotionGraphError::new(format!(
                    "source segment {expected_source} has invalid jump target {}",
                    jump.target_segment
                )));
            }
            if !targets.insert(jump.target_segment) {
                return Err(MotionGraphError::new(format!(
                    "source segment {expected_source} repeats jump target {}",
                    jump.target_segment
                )));
            }
            if !jump.score.is_finite() || jump.score < 0.0 || !jump.discontinuity.is_close(settings)
            {
                return Err(MotionGraphError::new(format!(
                    "source segment {expected_source} has an ineligible jump"
                )));
            }
            if let Some((previous_score, previous_target)) = previous_order
                && (previous_score, previous_target) > (jump.score, jump.target_segment)
            {
                return Err(MotionGraphError::new(format!(
                    "source segment {expected_source} jump candidates are not ranked"
                )));
            }
            previous_order = Some((jump.score, jump.target_segment));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_candidates_do_not_displace_distant_forward_and_backward_jumps() {
        let candidates = [
            (32, 0.01),
            (33, 0.02),
            (34, 0.03),
            (40, 0.4),
            (41, 0.5),
            (10, 0.8),
            (55, 0.9),
        ]
        .into_iter()
        .map(|(target, score)| {
            MotionJump::unranked(target, MotionDiscontinuity::uniform(score, score))
        })
        .collect();
        let selected = rank_eligible_jumps(30, candidates, MotionGraphSettings::default());
        assert_eq!(
            selected
                .iter()
                .map(|jump| jump.target_segment)
                .collect::<Vec<_>>(),
            vec![40, 10, 55]
        );
    }

    #[test]
    fn separation_is_measured_from_the_departure_boundary_in_both_directions() {
        let candidates = [24, 25, 26, 30, 31, 32, 36, 37]
            .into_iter()
            .map(|target| MotionJump::unranked(target, MotionDiscontinuity::uniform(0.1, 0.1)))
            .collect();
        let selected = rank_eligible_jumps(30, candidates, MotionGraphSettings::default());
        // Departure is sample 31; target 25 is six intervals behind, 37 six ahead.
        assert!(
            selected
                .iter()
                .all(|jump| [24, 25, 37].contains(&jump.target_segment))
        );
        assert!(selected.iter().any(|jump| jump.target_segment == 37));
        for (target, accepted) in [(25, true), (26, false), (36, false), (37, true)] {
            let ranked = rank_eligible_jumps(
                30,
                vec![MotionJump::unranked(
                    target,
                    MotionDiscontinuity::uniform(0.1, 0.1),
                )],
                MotionGraphSettings::default(),
            );
            assert_eq!(!ranked.is_empty(), accepted, "target {target}");
        }
    }

    #[test]
    fn candidate_budget_is_respected_before_reserving_time_directions() {
        for (budget, expected) in [(0, vec![]), (1, vec![40]), (2, vec![40, 10])] {
            let settings = MotionGraphSettings {
                max_jump_candidates: budget,
                ..MotionGraphSettings::default()
            };
            let candidates = [(40, 0.1), (50, 0.2), (10, 0.3)]
                .into_iter()
                .map(|(target, score)| {
                    MotionJump::unranked(target, MotionDiscontinuity::uniform(score, score))
                })
                .collect();
            assert_eq!(
                rank_eligible_jumps(30, candidates, settings)
                    .iter()
                    .map(|jump| jump.target_segment)
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn one_direction_pool_deduplicates_and_separates_targets_without_forcing_slots() {
        let candidates = [10, 10, 11, 12, 20]
            .into_iter()
            .map(|target| MotionJump::unranked(target, MotionDiscontinuity::uniform(0.1, 0.1)))
            .collect();
        let selected = rank_eligible_jumps(50, candidates, MotionGraphSettings::default());
        assert_eq!(
            selected
                .iter()
                .map(|jump| jump.target_segment)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn invalid_temporal_settings_and_unsafe_distant_targets_never_create_jumps() {
        for minimum in [0, SOURCE_SEGMENT_COUNT + 1] {
            let settings = MotionGraphSettings {
                min_jump_intervals: minimum,
                ..MotionGraphSettings::default()
            };
            assert!(
                rank_eligible_jumps(
                    0,
                    vec![MotionJump::unranked(
                        20,
                        MotionDiscontinuity::uniform(0.1, 0.1)
                    )],
                    settings
                )
                .is_empty()
            );
        }
        let settings = MotionGraphSettings::default();
        assert!(
            rank_eligible_jumps(
                0,
                vec![MotionJump::unranked(
                    30,
                    MotionDiscontinuity::uniform(settings.p95_limit + 0.1, 1.0)
                )],
                settings
            )
            .is_empty()
        );
    }

    #[test]
    fn default_calibration_accepts_observed_motion_tail() {
        // Conservative envelope of the measured shrub 3 -> 59 boundary metrics.
        let discontinuity = MotionDiscontinuity::uniform(10.6, 47.6);

        assert!(discontinuity.is_close(MotionGraphSettings::default()));
    }

    #[test]
    fn hard_gate_rejects_any_failed_trs_metric() {
        let settings = MotionGraphSettings::default();
        let mut discontinuity = MotionDiscontinuity::uniform(0.25, 0.5);

        assert!(discontinuity.is_close(settings));

        discontinuity.p95.translation_position = settings.p95_limit + 0.01;

        assert!(!discontinuity.is_close(settings));
    }

    #[test]
    fn hard_gate_rejects_negative_discontinuity_metrics() {
        let settings = MotionGraphSettings::default();
        let mut discontinuity = MotionDiscontinuity::uniform(0.25, 0.5);

        discontinuity.ceiling.rotation_velocity = -0.01;

        assert!(!discontinuity.is_close(settings));
    }

    #[test]
    fn ranking_keeps_best_separated_candidates_and_rejects_failed_gate() {
        let settings = MotionGraphSettings::default();
        let candidates = vec![
            MotionJump::unranked(3, MotionDiscontinuity::uniform(0.4, 0.8)),
            MotionJump::unranked(10, MotionDiscontinuity::uniform(0.3, 0.6)),
            MotionJump::unranked(20, MotionDiscontinuity::uniform(0.1, 0.2)),
            MotionJump::unranked(30, MotionDiscontinuity::uniform(0.2, 0.4)),
            MotionJump::unranked(
                40,
                MotionDiscontinuity::uniform(settings.p95_limit + 0.1, 0.4),
            ),
        ];

        let ranked = rank_eligible_jumps(2, candidates, settings);

        assert_eq!(
            ranked
                .iter()
                .map(|jump| jump.target_segment)
                .collect::<Vec<_>>(),
            vec![20, 30, 10]
        );
        assert!(ranked.windows(2).all(|pair| pair[0].score <= pair[1].score));
    }

    #[test]
    fn graph_accepts_nodes_without_jumps_and_reports_sparse_summary() {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|segment| MotionGraphNode {
                source_segment: segment,
                jumps: Vec::new(),
            })
            .collect();

        let graph = MotionGraph::new(nodes, settings).expect("zero-jump graph should be valid");

        assert_eq!(graph.summary().segment_count, SOURCE_SEGMENT_COUNT);
        assert_eq!(graph.summary().segments_with_jumps, 0);
        assert_eq!(graph.summary().jump_count, 0);
        assert_eq!(graph.summary().max_jumps_per_segment, 0);
    }

    #[test]
    fn ranking_score_averages_equivalent_step_metrics_without_gate_rescaling() {
        let settings = MotionGraphSettings::default();
        let ranked = rank_eligible_jumps(
            0,
            vec![MotionJump::unranked(
                10,
                MotionDiscontinuity::uniform(0.5, 1.5),
            )],
            settings,
        );

        assert_eq!(ranked[0].score, 1.0);
    }

    #[test]
    fn direction_coverage_respects_the_selected_source_range() {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| MotionGraphNode {
                source_segment,
                jumps: Vec::new(),
            })
            .collect();
        let mut features = vec![MotionSegmentFeature::default(); SOURCE_SEGMENT_COUNT];
        features[0..20].fill(MotionSegmentFeature {
            horizontal_direction: [1.0, 0.0],
            magnitude: 3.0,
        });
        features[20..60].fill(MotionSegmentFeature {
            horizontal_direction: [-1.0, 0.0],
            magnitude: 3.0,
        });
        features[64] = MotionSegmentFeature {
            horizontal_direction: [1.0, 0.0],
            magnitude: 0.03,
        };
        let graph = MotionGraph::new(nodes, settings)
            .unwrap()
            .with_segment_features(features)
            .unwrap();

        assert_eq!(graph.direction_coverage([1.0, 0.0], 0, 20).unwrap(), 1.0);
        assert_eq!(graph.direction_coverage([-1.0, 0.0], 0, 20).unwrap(), 0.0);
        assert_eq!(graph.direction_coverage([-1.0, 0.0], 20, 60).unwrap(), 1.0);
        assert!(graph.direction_coverage([1.0, 0.0], 60, 70).unwrap() < 0.01);
    }
}
