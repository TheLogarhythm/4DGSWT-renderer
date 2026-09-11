use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::motion::{LoopPolicy, TimelineSample};
use crate::motion_graph::{MotionGraph, MotionJump, SOURCE_SEGMENT_COUNT};

// Hard-qualified candidates retain a bounded (at most 4:1) smoothness preference.
// Authored directional intent is still applied independently.
const SMOOTHNESS_LOGIT_SPAN: f32 = 1.3862944; // ln(4)
const DIRECTION_LOGIT_SCALE: f32 = 8.0;
const MAX_TRACE_LENGTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionGraphPlaybackConfig {
    pub enabled: bool,
    pub branch_probability: f32,
    pub transition_seconds: f32,
    pub minimum_dwell_segments: u32,
    pub seed: u32,
    pub source_start_segment: usize,
    pub source_end_segment: usize,
    pub horizontal_direction: [f32; 2],
    pub direction_influence: f32,
}

impl Default for MotionGraphPlaybackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            branch_probability: 0.2,
            transition_seconds: 0.15,
            minimum_dwell_segments: 4,
            seed: 1,
            source_start_segment: 0,
            source_end_segment: SOURCE_SEGMENT_COUNT,
            horizontal_direction: [0.0; 2],
            direction_influence: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MotionGraphPlaybackTrace {
    pub recent_edges: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ActiveTransition {
    from_u_start: f32,
    from_u_end: f32,
    to_u: f32,
    target_segment: usize,
    elapsed_seconds: f32,
    duration_seconds: f32,
}

#[derive(Clone, Debug)]
pub struct MotionGraphPlaybackController {
    source_duration: f32,
    config: MotionGraphPlaybackConfig,
    current_segment: usize,
    phase_seconds: f32,
    dwell_segments: u32,
    transition: Option<ActiveTransition>,
    rng: FixedRng,
    last_selected_edge: Option<(usize, usize)>,
    trace: MotionGraphPlaybackTrace,
    dirty: bool,
}

impl MotionGraphPlaybackController {
    pub fn new(source_duration: f32, seed: u32) -> Result<Self, MotionGraphPlaybackError> {
        if !source_duration.is_finite() || source_duration <= 0.0 {
            return Err(MotionGraphPlaybackError::new(
                "graph playback source duration must be positive and finite",
            ));
        }
        let config = MotionGraphPlaybackConfig {
            seed,
            ..MotionGraphPlaybackConfig::default()
        };
        Ok(Self {
            source_duration,
            config,
            current_segment: 0,
            phase_seconds: 0.0,
            dwell_segments: 0,
            transition: None,
            rng: FixedRng::new(seed),
            last_selected_edge: None,
            trace: MotionGraphPlaybackTrace::default(),
            dirty: true,
        })
    }

    pub fn config(&self) -> MotionGraphPlaybackConfig {
        self.config
    }

    pub fn current_segment(&self) -> usize {
        self.current_segment
    }

    pub fn at_active_range_end(&self) -> bool {
        self.transition.is_none()
            && self.current_segment + 1 == self.config.source_end_segment
            && self.phase_seconds + f32::EPSILON >= self.segment_seconds()
    }

    pub fn last_selected_edge(&self) -> Option<(usize, usize)> {
        self.last_selected_edge
    }

    pub fn trace(&self) -> &MotionGraphPlaybackTrace {
        &self.trace
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        if self.config.enabled != enabled {
            self.config.enabled = enabled;
            self.dirty = true;
        }
    }

    pub fn set_branch_probability(
        &mut self,
        probability: f32,
    ) -> Result<(), MotionGraphPlaybackError> {
        if !probability.is_finite() {
            return Err(MotionGraphPlaybackError::new(
                "branch probability must be finite",
            ));
        }
        let probability = probability.clamp(0.0, 1.0);
        if self.config.branch_probability != probability {
            self.config.branch_probability = probability;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn set_transition_seconds(&mut self, seconds: f32) -> Result<(), MotionGraphPlaybackError> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(MotionGraphPlaybackError::new(
                "graph transition duration must be positive and finite",
            ));
        }
        if self.config.transition_seconds != seconds {
            self.config.transition_seconds = seconds;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn set_minimum_dwell_segments(&mut self, segments: u32) {
        let segments = segments.min(SOURCE_SEGMENT_COUNT as u32);
        if self.config.minimum_dwell_segments != segments {
            self.config.minimum_dwell_segments = segments;
            self.dirty = true;
        }
    }

    pub fn set_seed(&mut self, seed: u32) {
        self.config.seed = seed;
        self.restart();
    }

    pub fn set_source_range(
        &mut self,
        start_segment: usize,
        end_segment: usize,
    ) -> Result<(), MotionGraphPlaybackError> {
        if start_segment >= end_segment || end_segment > SOURCE_SEGMENT_COUNT {
            return Err(MotionGraphPlaybackError::new(
                "graph source range must be a nonempty subset of the source segments",
            ));
        }
        if self.config.source_start_segment != start_segment
            || self.config.source_end_segment != end_segment
        {
            self.config.source_start_segment = start_segment;
            self.config.source_end_segment = end_segment;
            self.restart();
        }
        Ok(())
    }

    pub fn set_direction_preference(
        &mut self,
        direction: [f32; 2],
        influence: f32,
    ) -> Result<(), MotionGraphPlaybackError> {
        let direction_length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
        if direction.into_iter().any(|value| !value.is_finite())
            || direction_length > 1.0 + f32::EPSILON
            || !influence.is_finite()
            || !(0.0..=1.0).contains(&influence)
        {
            return Err(MotionGraphPlaybackError::new(
                "graph direction preference must be finite and bounded",
            ));
        }
        if self.config.horizontal_direction != direction
            || self.config.direction_influence != influence
        {
            self.config.horizontal_direction = direction;
            self.config.direction_influence = influence;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn restart(&mut self) {
        self.current_segment = self.config.source_start_segment;
        self.phase_seconds = 0.0;
        self.dwell_segments = 0;
        self.transition = None;
        self.rng = FixedRng::new(self.config.seed);
        self.last_selected_edge = None;
        self.trace.recent_edges.clear();
        self.dirty = true;
    }

    pub fn seek_normalized(&mut self, u: f32) -> Result<(), MotionGraphPlaybackError> {
        if !u.is_finite() || !(0.0..=1.0).contains(&u) {
            return Err(MotionGraphPlaybackError::new(
                "graph source time must be within [0, 1]",
            ));
        }
        let segment_seconds = self.segment_seconds();
        let range_start_u = self.config.source_start_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        let range_end_u = self.config.source_end_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        let u = u.clamp(range_start_u, range_end_u);
        if u >= range_end_u {
            self.current_segment = self.config.source_end_segment - 1;
            self.phase_seconds = segment_seconds;
        } else {
            let segment_position = u * SOURCE_SEGMENT_COUNT as f32;
            self.current_segment =
                (segment_position.floor() as usize).min(self.config.source_end_segment - 1);
            self.phase_seconds = (segment_position - self.current_segment as f32) * segment_seconds;
        }
        self.dwell_segments = 0;
        self.transition = None;
        self.last_selected_edge = None;
        self.dirty = true;
        Ok(())
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn sample(&self) -> TimelineSample {
        if let Some(transition) = self.transition {
            let linear = (transition.elapsed_seconds / transition.duration_seconds).clamp(0.0, 1.0);
            return TimelineSample::Blend {
                from_u: transition.from_u_start
                    + (transition.from_u_end - transition.from_u_start) * linear,
                to_u: transition.to_u,
                alpha: smoothstep(linear),
            };
        }
        let segment_seconds = self.segment_seconds();
        TimelineSample::Source {
            u: (self.current_segment as f32 + self.phase_seconds / segment_seconds)
                / SOURCE_SEGMENT_COUNT as f32,
        }
    }

    pub fn advance(
        &mut self,
        delta_seconds: f32,
        graph: &MotionGraph,
        endpoint_policy: LoopPolicy,
        speed: f32,
        endpoint_transition_seconds: f32,
        endpoint_blend_window_seconds: f32,
    ) -> Result<TimelineSample, MotionGraphPlaybackError> {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return Err(MotionGraphPlaybackError::new(
                "graph frame delta must be finite and nonnegative",
            ));
        }
        if !speed.is_finite() || speed <= 0.0 {
            return Err(MotionGraphPlaybackError::new(
                "graph playback speed must be positive and finite",
            ));
        }
        if graph.nodes().len() != SOURCE_SEGMENT_COUNT {
            return Err(MotionGraphPlaybackError::new(
                "graph playback requires the complete source graph",
            ));
        }
        if endpoint_policy == LoopPolicy::AppendedTransition
            && (!endpoint_transition_seconds.is_finite() || endpoint_transition_seconds <= 0.0)
        {
            return Err(MotionGraphPlaybackError::new(
                "appended endpoint transition must be positive and finite",
            ));
        }
        if endpoint_policy == LoopPolicy::BlendToStart
            && (!endpoint_blend_window_seconds.is_finite()
                || endpoint_blend_window_seconds <= 0.0
                || endpoint_blend_window_seconds >= self.range_duration_seconds())
        {
            return Err(MotionGraphPlaybackError::new(
                "endpoint blend window must be positive, finite, and shorter than the active source range",
            ));
        }

        let mut remaining = delta_seconds * speed;
        if !remaining.is_finite() {
            return Err(MotionGraphPlaybackError::new(
                "scaled graph frame delta is non-finite",
            ));
        }
        if remaining > self.range_duration_seconds() * 2.0 {
            self.recover_large_delta(
                remaining,
                endpoint_policy,
                endpoint_transition_seconds,
                endpoint_blend_window_seconds,
            );
            return Ok(self.sample());
        }

        let segment_seconds = self.segment_seconds();
        let mut iterations = 0;
        while remaining > 0.0 {
            iterations += 1;
            if iterations > SOURCE_SEGMENT_COUNT * 4 {
                return Err(MotionGraphPlaybackError::new(
                    "graph frame advancement exceeded its bounded iteration count",
                ));
            }

            if let Some(mut transition) = self.transition {
                let until_complete =
                    (transition.duration_seconds - transition.elapsed_seconds).max(0.0);
                if until_complete <= f32::EPSILON {
                    self.current_segment = transition.target_segment;
                    self.phase_seconds = 0.0;
                    self.dwell_segments = 0;
                    self.transition = None;
                    continue;
                }
                let consumed = remaining.min(until_complete);
                transition.elapsed_seconds += consumed;
                remaining -= consumed;
                if transition.elapsed_seconds + f32::EPSILON >= transition.duration_seconds {
                    self.current_segment = transition.target_segment;
                    self.phase_seconds = 0.0;
                    self.dwell_segments = 0;
                    self.transition = None;
                } else {
                    self.transition = Some(transition);
                }
                continue;
            }

            let source_seconds = self.current_source_seconds(segment_seconds);
            if endpoint_policy == LoopPolicy::BlendToStart {
                let blend_start = self.range_end_seconds() - endpoint_blend_window_seconds;
                if source_seconds + f32::EPSILON >= blend_start {
                    self.begin_endpoint_blend(
                        endpoint_blend_window_seconds,
                        (source_seconds - blend_start).max(0.0),
                    );
                    continue;
                }
            }

            if endpoint_policy == LoopPolicy::Hold
                && self.current_segment + 1 == self.config.source_end_segment
                && self.phase_seconds + f32::EPSILON >= segment_seconds
            {
                remaining = 0.0;
                continue;
            }

            let mut until_event = (segment_seconds - self.phase_seconds).max(0.0);
            if endpoint_policy == LoopPolicy::BlendToStart {
                let until_blend =
                    self.range_end_seconds() - endpoint_blend_window_seconds - source_seconds;
                if until_blend >= 0.0 {
                    until_event = until_event.min(until_blend);
                }
            }
            let consumed = remaining.min(until_event);
            self.phase_seconds += consumed;
            remaining -= consumed;
            if endpoint_policy == LoopPolicy::BlendToStart
                && self.current_source_seconds(segment_seconds) + f32::EPSILON
                    >= self.range_end_seconds() - endpoint_blend_window_seconds
            {
                self.begin_endpoint_blend(endpoint_blend_window_seconds, 0.0);
                continue;
            }
            if self.phase_seconds + f32::EPSILON < segment_seconds {
                continue;
            }
            self.phase_seconds = segment_seconds;

            if self.current_segment + 1 == self.config.source_end_segment {
                match endpoint_policy {
                    LoopPolicy::Hold => {
                        remaining = 0.0;
                    }
                    LoopPolicy::DirectWrap => {
                        self.current_segment = self.config.source_start_segment;
                        self.phase_seconds = 0.0;
                        self.dwell_segments = self.dwell_segments.saturating_add(1);
                    }
                    LoopPolicy::AppendedTransition => {
                        self.begin_endpoint_transition(endpoint_transition_seconds);
                    }
                    LoopPolicy::BlendToStart => {
                        self.begin_endpoint_blend(endpoint_blend_window_seconds, 0.0);
                    }
                }
                continue;
            }

            if self.config.enabled
                && self.dwell_segments >= self.config.minimum_dwell_segments
                && let Some(jump) =
                    self.choose_jump(&graph.nodes()[self.current_segment].jumps, graph)
            {
                self.begin_stochastic_transition(jump.target_segment);
                continue;
            }

            self.current_segment += 1;
            self.phase_seconds = 0.0;
            self.dwell_segments = self.dwell_segments.saturating_add(1);
        }
        Ok(self.sample())
    }

    fn choose_jump(
        &mut self,
        candidates: &[MotionJump],
        graph: &MotionGraph,
    ) -> Option<MotionJump> {
        let candidates = candidates
            .iter()
            .copied()
            .filter(|jump| {
                (self.config.source_start_segment..self.config.source_end_segment)
                    .contains(&jump.target_segment)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() || self.rng.next_f32() >= self.config.branch_probability {
            return None;
        }
        let maximum_logit = candidates
            .iter()
            .map(|jump| self.jump_logit(*jump, graph))
            .fold(f32::NEG_INFINITY, f32::max);
        let weights = candidates
            .iter()
            .map(|jump| (self.jump_logit(*jump, graph) - maximum_logit).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        let mut draw = self.rng.next_f32() * total;
        for (&jump, weight) in candidates.iter().zip(weights) {
            if draw <= weight {
                return Some(jump);
            }
            draw -= weight;
        }
        candidates.last().copied()
    }

    fn jump_logit(&self, jump: MotionJump, graph: &MotionGraph) -> f32 {
        let feature = graph.segment_features()[jump.target_segment];
        let direction = self.config.horizontal_direction;
        let alignment = feature.horizontal_direction[0] * direction[0]
            + feature.horizontal_direction[1] * direction[1];
        let magnitude_support = (feature.magnitude
            / graph.maximum_segment_magnitude().max(f32::EPSILON))
        .clamp(0.0, 1.0);
        let settings = graph.settings();
        let gate_score =
            (settings.p95_limit * 0.5 + settings.ceiling_limit * 0.5).max(f32::EPSILON);
        -(jump.score / gate_score).clamp(0.0, 1.0) * SMOOTHNESS_LOGIT_SPAN
            + alignment
                * magnitude_support
                * self.config.direction_influence
                * DIRECTION_LOGIT_SCALE
    }

    fn begin_stochastic_transition(&mut self, target_segment: usize) {
        let edge = (self.current_segment, target_segment);
        self.last_selected_edge = Some(edge);
        self.trace.recent_edges.push(edge);
        if self.trace.recent_edges.len() > MAX_TRACE_LENGTH {
            self.trace.recent_edges.remove(0);
        }
        let source_exit_u = (self.current_segment + 1) as f32 / SOURCE_SEGMENT_COUNT as f32;
        self.transition = Some(ActiveTransition {
            from_u_start: source_exit_u,
            from_u_end: source_exit_u,
            to_u: target_segment as f32 / SOURCE_SEGMENT_COUNT as f32,
            target_segment,
            elapsed_seconds: 0.0,
            duration_seconds: self.config.transition_seconds,
        });
    }

    fn begin_endpoint_transition(&mut self, duration_seconds: f32) {
        let range_end_u = self.config.source_end_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        let range_start_u = self.config.source_start_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        self.transition = Some(ActiveTransition {
            from_u_start: range_end_u,
            from_u_end: range_end_u,
            to_u: range_start_u,
            target_segment: self.config.source_start_segment,
            elapsed_seconds: 0.0,
            duration_seconds,
        });
    }

    fn begin_endpoint_blend(&mut self, duration_seconds: f32, elapsed_seconds: f32) {
        let range_end_u = self.config.source_end_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        let range_start_u = self.config.source_start_segment as f32 / SOURCE_SEGMENT_COUNT as f32;
        self.transition = Some(ActiveTransition {
            from_u_start: range_end_u - duration_seconds / self.source_duration,
            from_u_end: range_end_u,
            to_u: range_start_u,
            target_segment: self.config.source_start_segment,
            elapsed_seconds: elapsed_seconds.clamp(0.0, duration_seconds),
            duration_seconds,
        });
    }

    fn current_source_seconds(&self, segment_seconds: f32) -> f32 {
        self.current_segment as f32 * segment_seconds + self.phase_seconds
    }

    fn segment_seconds(&self) -> f32 {
        self.source_duration / SOURCE_SEGMENT_COUNT as f32
    }

    fn range_start_seconds(&self) -> f32 {
        self.config.source_start_segment as f32 * self.segment_seconds()
    }

    fn range_end_seconds(&self) -> f32 {
        self.config.source_end_segment as f32 * self.segment_seconds()
    }

    fn range_duration_seconds(&self) -> f32 {
        self.range_end_seconds() - self.range_start_seconds()
    }

    fn set_source_seconds(&mut self, source_seconds: f32) {
        let segment_seconds = self.segment_seconds();
        let source_seconds =
            source_seconds.clamp(self.range_start_seconds(), self.range_end_seconds());
        if source_seconds >= self.range_end_seconds() {
            self.current_segment = self.config.source_end_segment - 1;
            self.phase_seconds = segment_seconds;
        } else {
            let segment_position = source_seconds / segment_seconds;
            self.current_segment =
                (segment_position.floor() as usize).min(self.config.source_end_segment - 1);
            self.phase_seconds = (segment_position - self.current_segment as f32) * segment_seconds;
        }
    }

    fn recover_large_delta(
        &mut self,
        mut elapsed_seconds: f32,
        policy: LoopPolicy,
        endpoint_transition_seconds: f32,
        endpoint_blend_window_seconds: f32,
    ) {
        if let Some(transition) = self.transition.take() {
            let until_complete =
                (transition.duration_seconds - transition.elapsed_seconds).max(0.0);
            if elapsed_seconds < until_complete {
                self.transition = Some(ActiveTransition {
                    elapsed_seconds: transition.elapsed_seconds + elapsed_seconds,
                    ..transition
                });
                return;
            }
            elapsed_seconds -= until_complete;
            self.current_segment = transition.target_segment;
            self.phase_seconds = 0.0;
            self.dwell_segments = 0;
        }

        self.last_selected_edge = None;
        self.dwell_segments = 0;
        let segment_seconds = self.segment_seconds();
        let current_seconds = self.current_source_seconds(segment_seconds);
        let range_start = self.range_start_seconds();
        let range_duration = self.range_duration_seconds();
        let current_local = current_seconds - range_start;
        match policy {
            LoopPolicy::Hold => {
                self.set_source_seconds(
                    range_start + (current_local + elapsed_seconds).min(range_duration),
                );
            }
            LoopPolicy::DirectWrap => {
                self.set_source_seconds(
                    range_start + (current_local + elapsed_seconds).rem_euclid(range_duration),
                );
            }
            LoopPolicy::AppendedTransition => {
                let period = range_duration + endpoint_transition_seconds;
                let phase = (current_local + elapsed_seconds).rem_euclid(period);
                if phase <= range_duration {
                    self.set_source_seconds(range_start + phase);
                } else {
                    self.set_source_seconds(self.range_end_seconds());
                    self.begin_endpoint_transition(endpoint_transition_seconds);
                    if let Some(transition) = self.transition.as_mut() {
                        transition.elapsed_seconds = phase - range_duration;
                    }
                }
            }
            LoopPolicy::BlendToStart => {
                let phase = (current_local + elapsed_seconds).rem_euclid(range_duration);
                let blend_start = range_duration - endpoint_blend_window_seconds;
                if phase < blend_start {
                    self.set_source_seconds(range_start + phase);
                } else {
                    self.set_source_seconds(range_start + phase);
                    self.begin_endpoint_blend(endpoint_blend_window_seconds, phase - blend_start);
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionGraphPlaybackError(String);

impl MotionGraphPlaybackError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MotionGraphPlaybackError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MotionGraphPlaybackError {}

#[derive(Clone, Copy, Debug)]
struct FixedRng {
    state: u32,
}

impl FixedRng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x6d2b_79f5 } else { seed },
        }
    }

    fn next_f32(&mut self) -> f32 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        self.state = value;
        ((value >> 8) as f32) / 16_777_216.0
    }
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motion::LoopPolicy;
    use crate::motion_graph::{
        MotionDiscontinuity, MotionGraph, MotionGraphNode, MotionGraphSettings, MotionJump,
        MotionSegmentFeature, SOURCE_SEGMENT_COUNT, rank_eligible_jumps,
    };

    fn graph_with_branch(source_with_branch: Option<(usize, usize)>) -> MotionGraph {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| {
                let candidates = if let Some((source, target)) = source_with_branch
                    && source_segment == source
                {
                    vec![MotionJump::unranked(
                        target,
                        MotionDiscontinuity::uniform(0.1, 0.2),
                    )]
                } else {
                    Vec::new()
                };
                MotionGraphNode {
                    source_segment,
                    jumps: rank_eligible_jumps(source_segment, candidates, settings),
                }
            })
            .collect();
        MotionGraph::new(nodes, settings).unwrap()
    }

    fn branching_graph(two_candidates: bool) -> MotionGraph {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| {
                let mut candidates = vec![MotionJump::unranked(
                    (source_segment + 10) % SOURCE_SEGMENT_COUNT,
                    MotionDiscontinuity::uniform(0.1, 0.2),
                )];
                if two_candidates {
                    candidates.push(MotionJump::unranked(
                        (source_segment + 20) % SOURCE_SEGMENT_COUNT,
                        MotionDiscontinuity::uniform(0.1, 0.2),
                    ));
                }
                MotionGraphNode {
                    source_segment,
                    jumps: rank_eligible_jumps(source_segment, candidates, settings),
                }
            })
            .collect();
        MotionGraph::new(nodes, settings).unwrap()
    }

    fn directional_graph() -> MotionGraph {
        let settings = MotionGraphSettings::default();
        let nodes = (0..SOURCE_SEGMENT_COUNT)
            .map(|source_segment| {
                let candidates = if source_segment == 0 {
                    vec![
                        MotionJump::unranked(10, MotionDiscontinuity::uniform(0.1, 0.2)),
                        MotionJump::unranked(20, MotionDiscontinuity::uniform(0.1, 0.2)),
                    ]
                } else {
                    Vec::new()
                };
                MotionGraphNode {
                    source_segment,
                    jumps: rank_eligible_jumps(source_segment, candidates, settings),
                }
            })
            .collect();
        let mut features = vec![MotionSegmentFeature::default(); SOURCE_SEGMENT_COUNT];
        features[10] = MotionSegmentFeature {
            horizontal_direction: [1.0, 0.0],
            magnitude: 1.0,
        };
        features[20] = MotionSegmentFeature {
            horizontal_direction: [-1.0, 0.0],
            magnitude: 1.0,
        };
        MotionGraph::new(nodes, settings)
            .unwrap()
            .with_segment_features(features)
            .unwrap()
    }

    fn run_branch_trace(graph: &MotionGraph, seed: u32) -> Vec<(usize, usize)> {
        let mut playback = MotionGraphPlaybackController::new(2.0, seed).unwrap();
        playback.set_enabled(true);
        playback.set_branch_probability(1.0).unwrap();
        playback.set_minimum_dwell_segments(0);
        playback.set_transition_seconds(0.001).unwrap();
        for _ in 0..20 {
            playback
                .advance(2.0 / 74.0, graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
                .unwrap();
            playback
                .advance(0.001, graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
                .unwrap();
        }
        playback.trace().recent_edges.clone()
    }

    #[test]
    fn safe_higher_score_candidates_are_not_starved_by_smoothness_weighting() {
        let graph = graph_with_branch(None);
        let candidates = [
            MotionJump {
                target_segment: 10,
                score: 0.1,
                discontinuity: MotionDiscontinuity::uniform(0.1, 0.1),
            },
            MotionJump {
                target_segment: 50,
                score: 7.4,
                discontinuity: MotionDiscontinuity::uniform(2.9, 11.9),
            },
        ];
        let mut playback = MotionGraphPlaybackController::new(2.0, 17).unwrap();
        playback.set_branch_probability(1.0).unwrap();
        let farther = (0..2000)
            .filter(|_| {
                playback
                    .choose_jump(&candidates, &graph)
                    .map(|jump| jump.target_segment)
                    == Some(50)
            })
            .count();
        assert!(
            farther > 200,
            "qualified alternative was starved: {farther}/2000"
        );
        assert!(
            farther < 1000,
            "smoothness preference was lost: {farther}/2000"
        );
    }

    #[test]
    fn zero_candidate_node_uses_default_continuation() {
        let graph = graph_with_branch(None);
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.set_enabled(true);

        let sample = playback
            .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
            .unwrap();

        assert_eq!(playback.current_segment(), 1);
        assert_eq!(sample, TimelineSample::Source { u: 1.0 / 74.0 });
    }

    #[test]
    fn branch_transition_lands_exactly_on_the_target_entry() {
        let graph = graph_with_branch(Some((0, 10)));
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.set_enabled(true);
        playback.set_branch_probability(1.0).unwrap();
        playback.set_minimum_dwell_segments(0);

        let boundary = playback
            .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
            .unwrap();
        assert!(matches!(boundary, TimelineSample::Blend { alpha: 0.0, .. }));

        let landed = playback
            .advance(
                playback.config().transition_seconds,
                &graph,
                LoopPolicy::DirectWrap,
                1.0,
                0.5,
                0.2,
            )
            .unwrap();
        assert_eq!(playback.current_segment(), 10);
        assert_eq!(landed, TimelineSample::Source { u: 10.0 / 74.0 });
    }

    #[test]
    fn final_segment_obeys_hold_instead_of_taking_a_stochastic_edge() {
        let graph = graph_with_branch(Some((SOURCE_SEGMENT_COUNT - 1, 10)));
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.set_enabled(true);
        playback.set_branch_probability(1.0).unwrap();
        playback.set_minimum_dwell_segments(0);
        playback
            .seek_normalized((SOURCE_SEGMENT_COUNT - 1) as f32 / SOURCE_SEGMENT_COUNT as f32)
            .unwrap();

        let sample = playback
            .advance(2.0 / 74.0, &graph, LoopPolicy::Hold, 1.0, 0.5, 0.2)
            .unwrap();

        assert_eq!(sample, TimelineSample::Source { u: 1.0 });
        assert!(playback.trace().recent_edges.is_empty());
        assert_eq!(playback.last_selected_edge(), None);
    }

    #[test]
    fn blend_to_start_begins_inside_the_source_window() {
        let graph = graph_with_branch(None);
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.seek_normalized(0.9).unwrap();

        let sample = playback
            .advance(0.01, &graph, LoopPolicy::BlendToStart, 1.0, 0.5, 0.2)
            .unwrap();

        assert!(matches!(
            sample,
            TimelineSample::Blend {
                from_u,
                to_u: 0.0,
                alpha
            } if from_u > 0.9 && alpha > 0.0
        ));
        assert!(playback.trace().recent_edges.is_empty());
    }

    #[test]
    fn large_delta_recovery_includes_the_current_source_position() {
        let graph = graph_with_branch(None);
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.seek_normalized(0.25).unwrap();

        let sample = playback
            .advance(5.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
            .unwrap();

        assert_eq!(sample, TimelineSample::Source { u: 0.75 });
    }

    #[test]
    fn same_seed_replays_the_same_branch_trace() {
        let graph = branching_graph(true);

        assert_eq!(run_branch_trace(&graph, 41), run_branch_trace(&graph, 41));
        assert_ne!(run_branch_trace(&graph, 41), run_branch_trace(&graph, 42));
    }

    #[test]
    fn minimum_dwell_blocks_candidates_until_enough_segments_continue() {
        let graph = branching_graph(false);
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.set_enabled(true);
        playback.set_branch_probability(1.0).unwrap();
        playback.set_minimum_dwell_segments(4);

        for _ in 0..4 {
            playback
                .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
                .unwrap();
        }
        assert!(playback.trace().recent_edges.is_empty());

        playback
            .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
            .unwrap();

        assert_eq!(playback.trace().recent_edges[0].0, 4);
    }

    #[test]
    fn source_range_filters_stochastic_targets_and_wraps_to_its_start() {
        let graph = directional_graph();
        let mut playback = MotionGraphPlaybackController::new(2.0, 7).unwrap();
        playback.set_enabled(true);
        playback.set_branch_probability(1.0).unwrap();
        playback.set_minimum_dwell_segments(0);
        playback.set_source_range(0, 15).unwrap();

        playback
            .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
            .unwrap();

        assert_eq!(playback.last_selected_edge(), Some((0, 10)));

        playback.set_source_range(10, 20).unwrap();
        playback.restart();
        let sample = playback
            .advance(
                10.0 * 2.0 / 74.0,
                &graph,
                LoopPolicy::DirectWrap,
                1.0,
                0.5,
                0.2,
            )
            .unwrap();
        assert_eq!(sample, TimelineSample::Source { u: 10.0 / 74.0 });
    }

    #[test]
    fn direction_preference_softly_favors_aligned_eligible_targets() {
        let graph = directional_graph();
        let mut aligned = 0;
        let mut opposed = 0;
        for seed in 1..=32 {
            let mut playback = MotionGraphPlaybackController::new(2.0, seed).unwrap();
            playback.set_enabled(true);
            playback.set_branch_probability(1.0).unwrap();
            playback.set_minimum_dwell_segments(0);
            playback.set_direction_preference([-1.0, 0.0], 1.0).unwrap();
            playback
                .advance(2.0 / 74.0, &graph, LoopPolicy::DirectWrap, 1.0, 0.5, 0.2)
                .unwrap();
            match playback.last_selected_edge() {
                Some((0, 20)) => aligned += 1,
                Some((0, 10)) => opposed += 1,
                edge => panic!("unexpected branch {edge:?}"),
            }
        }

        assert!(aligned >= 30, "aligned={aligned}, opposed={opposed}");
    }
}
