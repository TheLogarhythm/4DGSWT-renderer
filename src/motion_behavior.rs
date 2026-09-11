use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::motion::LoopPolicy;
use crate::motion_graph::SOURCE_SEGMENT_COUNT;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionBehaviorPreset {
    GentleSway,
    SteadyWind,
    GustyWind,
    Calm,
    Custom,
}

impl MotionBehaviorPreset {
    pub const ARTIST_PRESETS: [Self; 4] = [
        Self::GentleSway,
        Self::SteadyWind,
        Self::GustyWind,
        Self::Calm,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::GentleSway => "Gentle",
            Self::SteadyWind => "Steady",
            Self::GustyWind => "Lively",
            Self::Calm => "Subtle",
            Self::Custom => "Custom",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionChannelGains {
    pub translation: [f32; 3],
    pub rotation: f32,
    pub scale: f32,
    pub master: f32,
}

impl Default for MotionChannelGains {
    fn default() -> Self {
        Self {
            translation: [1.0; 3],
            rotation: 1.0,
            scale: 1.0,
            master: 1.0,
        }
    }
}

impl MotionChannelGains {
    pub fn validate(self) -> Result<(), MotionBehaviorError> {
        let values = [
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.rotation,
            self.scale,
            self.master,
        ];
        if values
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=2.0).contains(&value))
        {
            return Err(MotionBehaviorError::new(
                "motion channel gains must be finite and within [0, 2]",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotionSourceRange {
    pub name: String,
    pub start_segment: usize,
    pub end_segment: usize,
    pub enabled: bool,
}

impl MotionSourceRange {
    pub fn new(
        name: impl Into<String>,
        start_segment: usize,
        end_segment: usize,
    ) -> Result<Self, MotionBehaviorError> {
        let range = Self {
            name: name.into(),
            start_segment,
            end_segment,
            enabled: true,
        };
        range.validate()?;
        Ok(range)
    }

    pub fn full_source() -> Self {
        Self {
            name: "Full source".to_string(),
            start_segment: 0,
            end_segment: SOURCE_SEGMENT_COUNT,
            enabled: true,
        }
    }

    pub fn validate(&self) -> Result<(), MotionBehaviorError> {
        if self.name.trim().is_empty() {
            return Err(MotionBehaviorError::new(
                "motion source range name cannot be empty",
            ));
        }
        if self.start_segment >= self.end_segment || self.end_segment > SOURCE_SEGMENT_COUNT {
            return Err(MotionBehaviorError::new(format!(
                "motion source range must be a nonempty subset of segments 0..{SOURCE_SEGMENT_COUNT}",
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MotionBehavior {
    pub preset: MotionBehaviorPreset,
    pub name: String,
    pub stochastic_enabled: bool,
    pub playback_speed: f32,
    pub endpoint_policy: LoopPolicy,
    pub gains: MotionChannelGains,
    pub horizontal_direction: [f32; 2],
    pub direction_influence: f32,
    pub variation: f32,
    pub gust_frequency: f32,
    pub spatial_coherence: f32,
    pub transition_seconds: f32,
    pub seed: u32,
}

impl MotionBehavior {
    pub fn from_preset(preset: MotionBehaviorPreset) -> Self {
        let mut behavior = match preset {
            MotionBehaviorPreset::GentleSway | MotionBehaviorPreset::Custom => Self {
                preset,
                name: preset.label().to_string(),
                stochastic_enabled: true,
                playback_speed: 0.75,
                endpoint_policy: LoopPolicy::BlendToStart,
                gains: MotionChannelGains {
                    translation: [0.8, 0.8, 0.5],
                    rotation: 0.75,
                    scale: 0.2,
                    master: 0.65,
                },
                horizontal_direction: [0.0, 0.0],
                direction_influence: 0.0,
                variation: 0.25,
                gust_frequency: 0.15,
                spatial_coherence: 0.85,
                transition_seconds: 0.25,
                seed: 1,
            },
            MotionBehaviorPreset::SteadyWind => Self {
                preset,
                name: preset.label().to_string(),
                stochastic_enabled: true,
                playback_speed: 1.0,
                endpoint_policy: LoopPolicy::BlendToStart,
                gains: MotionChannelGains {
                    translation: [1.0, 1.0, 0.65],
                    rotation: 0.9,
                    scale: 0.15,
                    master: 0.9,
                },
                horizontal_direction: [0.0, 0.0],
                direction_influence: 0.0,
                variation: 0.15,
                gust_frequency: 0.1,
                spatial_coherence: 0.95,
                transition_seconds: 0.3,
                seed: 1,
            },
            MotionBehaviorPreset::GustyWind => Self {
                preset,
                name: preset.label().to_string(),
                stochastic_enabled: true,
                playback_speed: 1.15,
                endpoint_policy: LoopPolicy::AppendedTransition,
                gains: MotionChannelGains {
                    translation: [1.15, 1.15, 0.8],
                    rotation: 1.1,
                    scale: 0.2,
                    master: 1.2,
                },
                horizontal_direction: [0.0, 0.0],
                direction_influence: 0.0,
                variation: 0.8,
                gust_frequency: 0.8,
                spatial_coherence: 0.6,
                transition_seconds: 0.3,
                seed: 7,
            },
            MotionBehaviorPreset::Calm => Self {
                preset,
                name: preset.label().to_string(),
                stochastic_enabled: false,
                playback_speed: 0.5,
                endpoint_policy: LoopPolicy::BlendToStart,
                gains: MotionChannelGains {
                    translation: [0.5, 0.5, 0.25],
                    rotation: 0.25,
                    scale: 0.0,
                    master: 0.15,
                },
                horizontal_direction: [0.0, 0.0],
                direction_influence: 0.0,
                variation: 0.0,
                gust_frequency: 0.0,
                spatial_coherence: 1.0,
                transition_seconds: 0.3,
                seed: 1,
            },
        };
        if preset == MotionBehaviorPreset::Custom {
            behavior.name = "Custom Behavior".to_string();
        }
        behavior
    }

    pub fn validate(&self) -> Result<(), MotionBehaviorError> {
        if self.name.trim().is_empty() {
            return Err(MotionBehaviorError::new(
                "motion behavior name cannot be empty",
            ));
        }
        if !self.playback_speed.is_finite() || !(0.05..=4.0).contains(&self.playback_speed) {
            return Err(MotionBehaviorError::new(
                "motion behavior speed must be within [0.05, 4]",
            ));
        }
        self.gains.validate()?;
        let direction_length_squared = self.horizontal_direction[0] * self.horizontal_direction[0]
            + self.horizontal_direction[1] * self.horizontal_direction[1];
        if self
            .horizontal_direction
            .into_iter()
            .any(|value| !value.is_finite())
            || direction_length_squared > 1.0 + f32::EPSILON
        {
            return Err(MotionBehaviorError::new(
                "motion direction must be finite and inside the unit pad",
            ));
        }
        for (label, value) in [
            ("direction influence", self.direction_influence),
            ("variation", self.variation),
            ("gust frequency", self.gust_frequency),
            ("spatial coherence", self.spatial_coherence),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(MotionBehaviorError::new(format!(
                    "motion behavior {label} must be within [0, 1]",
                )));
            }
        }
        if !self.transition_seconds.is_finite() || !(0.01..=1.0).contains(&self.transition_seconds)
        {
            return Err(MotionBehaviorError::new(
                "motion transition duration must be within [0.01, 1] seconds",
            ));
        }
        Ok(())
    }

    pub fn branch_probability(&self) -> f32 {
        (self.variation * (0.1 + 0.5 * self.gust_frequency)).clamp(0.0, 0.6)
    }

    pub fn minimum_dwell_segments(&self) -> u32 {
        (12.0 - 11.0 * self.gust_frequency).round() as u32
    }
}

#[derive(Clone, Debug)]
pub struct MotionAuthoringState {
    behavior: MotionBehavior,
    ranges: Vec<MotionSourceRange>,
    active_range_index: usize,
    dirty: bool,
}

impl Default for MotionAuthoringState {
    fn default() -> Self {
        Self {
            behavior: MotionBehavior::from_preset(MotionBehaviorPreset::GentleSway),
            ranges: vec![MotionSourceRange::full_source()],
            active_range_index: 0,
            dirty: true,
        }
    }
}

impl MotionAuthoringState {
    pub fn from_parts(
        behavior: MotionBehavior,
        ranges: Vec<MotionSourceRange>,
        active_range_index: usize,
    ) -> Result<Self, MotionBehaviorError> {
        behavior.validate()?;
        if ranges.is_empty() || ranges.len() > 256 {
            return Err(MotionBehaviorError::new(
                "session needs 1..=256 source ranges",
            ));
        }
        for range in &ranges {
            range.validate()?;
        }
        if !ranges.get(active_range_index).is_some_and(|r| r.enabled) {
            return Err(MotionBehaviorError::new(
                "active source range must exist and be enabled",
            ));
        }
        Ok(Self {
            behavior,
            ranges,
            active_range_index,
            dirty: true,
        })
    }
    pub fn behavior(&self) -> &MotionBehavior {
        &self.behavior
    }

    pub fn ranges(&self) -> &[MotionSourceRange] {
        &self.ranges
    }

    pub fn active_range_index(&self) -> usize {
        self.active_range_index
    }

    pub fn active_range(&self) -> &MotionSourceRange {
        &self.ranges[self.active_range_index]
    }

    pub fn apply_preset(&mut self, preset: MotionBehaviorPreset) {
        self.behavior = MotionBehavior::from_preset(preset);
        self.dirty = true;
    }

    pub fn update_behavior(
        &mut self,
        edit: impl FnOnce(&mut MotionBehavior),
    ) -> Result<(), MotionBehaviorError> {
        let mut candidate = self.behavior.clone();
        edit(&mut candidate);
        candidate.preset = MotionBehaviorPreset::Custom;
        candidate.validate()?;
        if candidate != self.behavior {
            self.behavior = candidate;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn add_range(
        &mut self,
        name: impl Into<String>,
        start_segment: usize,
        end_segment: usize,
    ) -> Result<usize, MotionBehaviorError> {
        let range = MotionSourceRange::new(name, start_segment, end_segment)?;
        self.ranges.push(range);
        self.dirty = true;
        Ok(self.ranges.len() - 1)
    }

    pub fn update_range(
        &mut self,
        index: usize,
        range: MotionSourceRange,
    ) -> Result<(), MotionBehaviorError> {
        range.validate()?;
        let fallback = if index == self.active_range_index && !range.enabled {
            Some(
                self.ranges
                    .iter()
                    .enumerate()
                    .find(|(candidate, range)| *candidate != index && range.enabled)
                    .map(|(candidate, _)| candidate)
                    .ok_or_else(|| {
                        MotionBehaviorError::new(
                            "at least one motion source range must remain enabled",
                        )
                    })?,
            )
        } else {
            None
        };
        let target = self
            .ranges
            .get_mut(index)
            .ok_or_else(|| MotionBehaviorError::new("motion source range index is invalid"))?;
        if *target != range {
            *target = range;
            if let Some(fallback) = fallback {
                self.active_range_index = fallback;
            }
            self.dirty = true;
        }
        Ok(())
    }

    pub fn select_range(&mut self, index: usize) -> Result<(), MotionBehaviorError> {
        if index >= self.ranges.len() {
            return Err(MotionBehaviorError::new(
                "motion source range index is invalid",
            ));
        }
        if !self.ranges[index].enabled {
            self.ranges[index].enabled = true;
            self.dirty = true;
        }
        if self.active_range_index != index {
            self.active_range_index = index;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn remove_range(&mut self, index: usize) -> Result<(), MotionBehaviorError> {
        if self.ranges.len() == 1 {
            return Err(MotionBehaviorError::new(
                "at least one motion source range must remain",
            ));
        }
        if index >= self.ranges.len() {
            return Err(MotionBehaviorError::new(
                "motion source range index is invalid",
            ));
        }
        if index == self.active_range_index
            && !self
                .ranges
                .iter()
                .enumerate()
                .any(|(candidate, range)| candidate != index && range.enabled)
        {
            return Err(MotionBehaviorError::new(
                "at least one motion source range must remain enabled",
            ));
        }
        self.ranges.remove(index);
        if index < self.active_range_index {
            self.active_range_index -= 1;
        } else if index == self.active_range_index {
            self.active_range_index = self
                .ranges
                .iter()
                .position(|range| range.enabled)
                .expect("range editing always preserves an enabled fallback");
        }
        self.dirty = true;
        Ok(())
    }

    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionBehaviorError(String);

impl MotionBehaviorError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MotionBehaviorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MotionBehaviorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_produces_a_finite_bounded_behavior() {
        for preset in [
            MotionBehaviorPreset::GentleSway,
            MotionBehaviorPreset::SteadyWind,
            MotionBehaviorPreset::GustyWind,
            MotionBehaviorPreset::Calm,
        ] {
            let behavior = MotionBehavior::from_preset(preset);

            assert_eq!(behavior.preset, preset);
            assert!(behavior.validate().is_ok());
            assert!((0.05..=4.0).contains(&behavior.playback_speed));
            assert!((0.0..=2.0).contains(&behavior.gains.master));
            assert!(
                behavior
                    .gains
                    .translation
                    .into_iter()
                    .all(|value| (0.0..=2.0).contains(&value))
            );
        }
    }

    #[test]
    fn editing_a_preset_creates_a_custom_behavior() {
        let mut authoring = MotionAuthoringState::default();

        authoring
            .update_behavior(|behavior| behavior.gains.master = 1.25)
            .unwrap();

        assert_eq!(authoring.behavior().preset, MotionBehaviorPreset::Custom);
        assert_eq!(authoring.behavior().gains.master, 1.25);
        assert!(authoring.take_dirty());
        assert!(!authoring.take_dirty());
    }

    #[test]
    fn invalid_or_empty_source_ranges_are_rejected() {
        assert!(MotionSourceRange::new("Reverse", 20, 10).is_err());
        assert!(MotionSourceRange::new("Empty", 4, 4).is_err());
        assert!(MotionSourceRange::new("Past end", 70, 75).is_err());
        assert!(MotionSourceRange::new("Useful", 10, 24).is_ok());
    }

    #[test]
    fn applying_a_preset_preserves_the_selected_asset_range() {
        let mut authoring = MotionAuthoringState::default();
        let index = authoring.add_range("Leftward", 12, 30).unwrap();
        authoring.select_range(index).unwrap();

        authoring.apply_preset(MotionBehaviorPreset::GustyWind);

        assert_eq!(authoring.active_range().name, "Leftward");
        assert_eq!(authoring.active_range().start_segment, 12);
        assert_eq!(authoring.active_range().end_segment, 30);
        assert_eq!(authoring.behavior().preset, MotionBehaviorPreset::GustyWind);
    }

    #[test]
    fn disabling_the_active_range_selects_an_enabled_fallback() {
        let mut authoring = MotionAuthoringState::default();
        let second = authoring.add_range("Wind section", 10, 30).unwrap();
        authoring.select_range(second).unwrap();
        let mut disabled = authoring.active_range().clone();
        disabled.enabled = false;

        authoring.update_range(second, disabled).unwrap();

        assert_eq!(authoring.active_range_index(), 0);
        assert!(authoring.active_range().enabled);
        let mut only_enabled = authoring.active_range().clone();
        only_enabled.enabled = false;
        assert!(authoring.update_range(0, only_enabled).is_err());

        authoring.select_range(second).unwrap();
        assert_eq!(authoring.active_range_index(), second);
        assert!(authoring.active_range().enabled);
    }
}
