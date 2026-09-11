use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};

const BASIS_MAGIC: &[u8; 8] = b"4DGSWTB\0";
const COEFFICIENT_MAGIC: &[u8; 8] = b"4DGSWTC\0";
const SAMPLE_COUNT: usize = 75;
const MOTION_DIMENSION: usize = 9;
const LOD_COUNT: usize = 6;

const CHANNELS: [&str; MOTION_DIMENSION] = [
    "delta_x",
    "delta_y",
    "delta_z",
    "rotation_vector_x",
    "rotation_vector_y",
    "rotation_vector_z",
    "delta_log_scale_x",
    "delta_log_scale_y",
    "delta_log_scale_z",
];

const PLACEMENT_TRANSFORMS: [[[f64; 3]; 3]; 2] = [
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicArchiveError(String);

impl DynamicArchiveError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for DynamicArchiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for DynamicArchiveError {}

#[derive(Clone, Debug)]
pub struct DynamicMember {
    pub tile: usize,
    pub lod: usize,
    pub gaussians: String,
    pub coefficients: String,
    pub gaussian_count: usize,
}

#[derive(Clone, Debug)]
pub struct MotionTime {
    pub sample_times: Vec<f64>,
    pub source_frame_times_normalized: Vec<f64>,
    pub source_frame_times_seconds: Vec<f64>,
    pub nominal_fps: f64,
}

impl MotionTime {
    pub fn source_duration_seconds(&self) -> f64 {
        self.source_frame_times_seconds[149] - self.source_frame_times_seconds[0]
    }
}

#[derive(Clone, Debug)]
pub struct MotionManifest {
    pub basis_count: usize,
    pub top_k: usize,
    pub bases: BasisManifest,
    pub source_to_target_scale: f64,
    pub time: MotionTime,
    pub placement_transforms: [[[f32; 3]; 3]; 2],
    pub coefficient_binary_version: u32,
}

#[derive(Clone, Debug)]
pub struct SourceSummary {
    pub backend: String,
    pub adapter_version: u32,
    pub run_id: Option<String>,
    pub encoder: Option<(String, u32)>,
}

#[derive(Clone, Debug)]
pub struct DynamicManifest {
    pub schema_version: u32,
    pub tile_count: usize,
    pub members: Vec<DynamicMember>,
    pub motion: MotionManifest,
    pub source: SourceSummary,
    pub extensions: Map<String, Value>,
}

#[derive(Clone, Debug)]
pub struct BasisBank {
    pub basis_count: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct BasisBanks {
    pub translation: BasisBank,
    pub rotation: BasisBank,
    pub scale: BasisBank,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisDescriptor {
    pub basis_count: usize,
    pub top_k: usize,
    pub path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BasisBankDescriptors {
    pub translation: BasisDescriptor,
    pub rotation: BasisDescriptor,
    pub scale: BasisDescriptor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasisManifest {
    Legacy,
    Separate(BasisBankDescriptors),
}

#[derive(Clone, Debug)]
pub enum CoefficientData {
    Legacy {
        basis_ids: Vec<u32>,
        weights: Vec<f32>,
    },
    Separate {
        basis_ids: Vec<u8>,
        weights: Vec<f32>,
    },
}

#[derive(Clone, Debug)]
pub struct CoefficientPayload {
    pub tile: usize,
    pub lod: usize,
    pub gaussian_count: usize,
    pub top_k: usize,
    pub data: CoefficientData,
    pub transform_ids: Vec<u32>,
    pub motion_channel_masks: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    asset_type: String,
    schema_version: u32,
    tile_count: usize,
    lod_count: usize,
    members: Vec<MemberWire>,
    motion: MotionWire,
    source: SourceWire,
    extensions: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemberWire {
    tile: usize,
    lod: usize,
    gaussians: String,
    coefficients: String,
    gaussian_count: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MotionWire {
    representation: String,
    basis_scope: String,
    basis_source_lod: u32,
    basis_count: Option<usize>,
    top_k: Option<usize>,
    basis: Option<String>,
    basis_banks: Option<BasisBanksWire>,
    basis_binary_version: u32,
    coefficient_binary_version: u32,
    motion_channel_mask: Option<MotionChannelMaskWire>,
    channels: Vec<String>,
    position_delta_space: String,
    rotation_delta: String,
    scale_delta: String,
    quaternion_order: String,
    source_to_target_scale: f64,
    time: TimeWire,
    interpolation: InterpolationWire,
    placement_transforms: Vec<PlacementTransformWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BasisBanksWire {
    translation: BasisDescriptorWire,
    rotation: BasisDescriptorWire,
    scale: BasisDescriptorWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BasisDescriptorWire {
    basis_count: usize,
    top_k: usize,
    path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MotionChannelMaskWire {
    encoding: String,
    translation_bit: u32,
    rotation_bit: u32,
    scale_bit: u32,
    reserved_mask: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimeWire {
    sample_times: Vec<f64>,
    source_frame_times_normalized: Vec<f64>,
    source_frame_times_seconds: Vec<f64>,
    nominal_fps: f64,
    interval: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InterpolationWire {
    kind: String,
    query_boundary: String,
    control_point_boundary: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlacementTransformWire {
    id: u32,
    name: String,
    matrix: [[f64; 3]; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceWire {
    asset_type: String,
    schema_version: u32,
    backend: String,
    adapter_version: u32,
    run_id: Value,
    basis_count: Option<usize>,
    top_k: Option<usize>,
    basis_banks: Option<SourceBasisBanksWire>,
    encoder: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBasisBanksWire {
    translation: SourceBasisDescriptorWire,
    rotation: SourceBasisDescriptorWire,
    scale: SourceBasisDescriptorWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBasisDescriptorWire {
    basis_count: usize,
    top_k: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EncoderWire {
    algorithm: String,
    algorithm_version: u32,
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueValueVisitor)
    }
}

struct UniqueValueVisitor;

impl<'de> Visitor<'de> for UniqueValueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("JSON number must be finite"))?;
        Ok(UniqueValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = HashSet::new();
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key '{key}'")));
            }
            let UniqueValue(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn expect_literal<T>(actual: &T, expected: T, location: &str) -> Result<(), DynamicArchiveError>
where
    T: PartialEq + Display,
{
    if *actual != expected {
        return Err(DynamicArchiveError::new(format!(
            "{location} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_raw_placement_types(value: &Value) -> Result<(), DynamicArchiveError> {
    let transforms = value
        .get("motion")
        .and_then(|motion| motion.get("placement_transforms"))
        .and_then(Value::as_array)
        .ok_or_else(|| DynamicArchiveError::new("motion.placement_transforms is invalid"))?;
    for (transform_index, transform) in transforms.iter().enumerate() {
        let rows = transform
            .get("matrix")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                DynamicArchiveError::new(format!(
                    "motion.placement_transforms[{transform_index}].matrix is invalid"
                ))
            })?;
        for (row_index, row) in rows.iter().enumerate() {
            let values = row.as_array().ok_or_else(|| {
                DynamicArchiveError::new(format!(
                    "motion.placement_transforms[{transform_index}].matrix[{row_index}] is invalid"
                ))
            })?;
            for value in values {
                if !value.is_number() {
                    return Err(DynamicArchiveError::new(format!(
                        "motion.placement_transforms[{transform_index}].matrix contains a non-number"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_uniform(
    values: &[f64],
    count: usize,
    location: &str,
) -> Result<(), DynamicArchiveError> {
    if values.len() != count || values.iter().any(|value| !value.is_finite()) {
        return Err(DynamicArchiveError::new(format!(
            "{location} must contain exactly {count} finite values"
        )));
    }
    for (index, value) in values.iter().enumerate() {
        let expected = exact_uniform_value(index, count);
        if *value != expected {
            return Err(DynamicArchiveError::new(format!(
                "{location}[{index}]={value:?} (bits {:016x}) is not the exact uniform inclusive value {expected:?} (bits {:016x})",
                value.to_bits(),
                expected.to_bits(),
            )));
        }
    }
    Ok(())
}

#[inline(never)]
fn exact_uniform_value(index: usize, count: usize) -> f64 {
    if index + 1 == count {
        1.0
    } else {
        let step = 1.0 / (count - 1) as f64;
        index as f64 * step
    }
}

#[inline(never)]
fn exact_source_second(index: usize, nominal_fps: f64) -> f64 {
    index as f64 / nominal_fps
}

pub fn parse_manifest(bytes: &[u8]) -> Result<DynamicManifest, DynamicArchiveError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let UniqueValue(value) = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| DynamicArchiveError::new(format!("manifest JSON is invalid: {error}")))?;
    deserializer
        .end()
        .map_err(|error| DynamicArchiveError::new(format!("manifest JSON is invalid: {error}")))?;
    validate_raw_placement_types(&value)?;
    let wire: ManifestWire = serde_json::from_value(value).map_err(|error| {
        DynamicArchiveError::new(format!("manifest validation failed: {error}"))
    })?;

    expect_literal(
        &wire.asset_type,
        "dynamic_gswt_archive".to_string(),
        "asset_type",
    )?;
    if !matches!(wire.schema_version, 1 | 2 | 3) {
        return Err(DynamicArchiveError::new(
            "schema_version must be 1, 2, or 3",
        ));
    }
    let schema_version = wire.schema_version;
    if wire.tile_count == 0 {
        return Err(DynamicArchiveError::new("tile_count must be positive"));
    }
    expect_literal(&wire.lod_count, LOD_COUNT, "lod_count")?;
    let expected_members = wire
        .tile_count
        .checked_mul(LOD_COUNT)
        .ok_or_else(|| DynamicArchiveError::new("member count overflow"))?;
    if wire.members.len() != expected_members {
        return Err(DynamicArchiveError::new(
            "members must contain tile_count * 6 entries",
        ));
    }

    let mut members = Vec::with_capacity(expected_members);
    for (index, member) in wire.members.into_iter().enumerate() {
        let expected_tile = index / LOD_COUNT;
        let expected_lod = index % LOD_COUNT;
        if member.tile != expected_tile || member.lod != expected_lod {
            return Err(DynamicArchiveError::new(
                "members must use exact tile-major LoD-major order",
            ));
        }
        let expected_gaussians = format!("tile{expected_tile}_lod{expected_lod}.ply");
        let expected_coefficients = format!("motion/tile{expected_tile}_lod{expected_lod}.bin");
        if member.gaussians != expected_gaussians || member.coefficients != expected_coefficients {
            return Err(DynamicArchiveError::new(format!(
                "member {index} does not use canonical paths"
            )));
        }
        if member.gaussian_count == 0 {
            return Err(DynamicArchiveError::new(format!(
                "member {index} gaussian_count must be positive"
            )));
        }
        members.push(DynamicMember {
            tile: member.tile,
            lod: member.lod,
            gaussians: member.gaussians,
            coefficients: member.coefficients,
            gaussian_count: member.gaussian_count,
        });
    }

    let motion = wire.motion;
    let expected_representation = if schema_version == 3 {
        "shared_sparse_separate_trs_bases"
    } else {
        "shared_sparse_joint_9d_basis"
    };
    expect_literal(
        &motion.representation,
        expected_representation.to_string(),
        "motion.representation",
    )?;
    expect_literal(
        &motion.basis_scope,
        "shared_lod0".to_string(),
        "motion.basis_scope",
    )?;
    expect_literal(&motion.basis_source_lod, 0, "motion.basis_source_lod")?;
    let (basis_count, top_k, bases) = if schema_version == 3 {
        if motion.basis_count.is_some() || motion.top_k.is_some() || motion.basis.is_some() {
            return Err(DynamicArchiveError::new(
                "version 3 motion must not contain legacy basis fields",
            ));
        }
        let banks = motion
            .basis_banks
            .as_ref()
            .ok_or_else(|| DynamicArchiveError::new("version 3 motion.basis_banks is required"))?;
        let wires = [&banks.translation, &banks.rotation, &banks.scale];
        let expected_paths = [
            "motion/basis_translation.bin",
            "motion/basis_rotation.bin",
            "motion/basis_scale.bin",
        ];
        for (wire, path) in wires.into_iter().zip(expected_paths) {
            if wire.basis_count == 0
                || wire.basis_count > 256
                || wire.top_k == 0
                || wire.top_k > wire.basis_count
                || wire.path != path
            {
                return Err(DynamicArchiveError::new(
                    "motion.basis_banks descriptor is invalid",
                ));
            }
        }
        let first = &banks.translation;
        if [&banks.rotation, &banks.scale]
            .into_iter()
            .any(|bank| bank.basis_count != first.basis_count || bank.top_k != first.top_k)
        {
            return Err(DynamicArchiveError::new(
                "motion basis banks must share basis_count and top_k",
            ));
        }
        let descriptor = |wire: &BasisDescriptorWire| BasisDescriptor {
            basis_count: wire.basis_count,
            top_k: wire.top_k,
            path: wire.path.clone(),
        };
        (
            first.basis_count,
            first.top_k,
            BasisManifest::Separate(BasisBankDescriptors {
                translation: descriptor(&banks.translation),
                rotation: descriptor(&banks.rotation),
                scale: descriptor(&banks.scale),
            }),
        )
    } else {
        if motion.basis_banks.is_some() {
            return Err(DynamicArchiveError::new(
                "legacy motion must not contain basis_banks",
            ));
        }
        let basis_count = motion
            .basis_count
            .ok_or_else(|| DynamicArchiveError::new("motion.basis_count is required"))?;
        let top_k = motion
            .top_k
            .ok_or_else(|| DynamicArchiveError::new("motion.top_k is required"))?;
        if basis_count == 0 || top_k == 0 || top_k > basis_count {
            return Err(DynamicArchiveError::new(
                "motion basis_count/top_k are invalid",
            ));
        }
        expect_literal(
            motion
                .basis
                .as_ref()
                .ok_or_else(|| DynamicArchiveError::new("motion.basis is required"))?,
            "motion/basis.bin".to_string(),
            "motion.basis",
        )?;
        (basis_count, top_k, BasisManifest::Legacy)
    };
    expect_literal(
        &motion.basis_binary_version,
        if schema_version == 3 { 2 } else { 1 },
        "motion.basis_binary_version",
    )?;
    expect_literal(
        &motion.coefficient_binary_version,
        schema_version,
        "motion.coefficient_binary_version",
    )?;
    match (schema_version, motion.motion_channel_mask.as_ref()) {
        (1, None) => {}
        (2 | 3, Some(mask))
            if mask.encoding == "uint8_bitfield"
                && mask.translation_bit == 0
                && mask.rotation_bit == 1
                && mask.scale_bit == 2
                && mask.reserved_mask == 248 => {}
        (1, Some(_)) => {
            return Err(DynamicArchiveError::new(
                "motion.motion_channel_mask is invalid for version 1",
            ));
        }
        _ => {
            return Err(DynamicArchiveError::new(
                "motion.motion_channel_mask is invalid for this version",
            ));
        }
    }
    if motion.channels != CHANNELS.map(str::to_string) {
        return Err(DynamicArchiveError::new("motion.channels are invalid"));
    }
    expect_literal(
        &motion.position_delta_space,
        "constructor_scaled_source_axes_then_row_transform".to_string(),
        "motion.position_delta_space",
    )?;
    expect_literal(
        &motion.rotation_delta,
        "local_rotation_vector".to_string(),
        "motion.rotation_delta",
    )?;
    expect_literal(
        &motion.scale_delta,
        "log_scale".to_string(),
        "motion.scale_delta",
    )?;
    expect_literal(
        &motion.quaternion_order,
        "wxyz".to_string(),
        "motion.quaternion_order",
    )?;
    if !motion.source_to_target_scale.is_finite() || motion.source_to_target_scale <= 0.0 {
        return Err(DynamicArchiveError::new(
            "motion.source_to_target_scale must be positive and finite",
        ));
    }

    let time = motion.time;
    validate_uniform(&time.sample_times, SAMPLE_COUNT, "motion.time.sample_times")?;
    validate_uniform(
        &time.source_frame_times_normalized,
        150,
        "motion.time.source_frame_times_normalized",
    )?;
    if !time.nominal_fps.is_finite() || time.nominal_fps <= 0.0 {
        return Err(DynamicArchiveError::new(
            "motion.time.nominal_fps must be positive and finite",
        ));
    }
    if time.source_frame_times_seconds.len() != 150
        || time
            .source_frame_times_seconds
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err(DynamicArchiveError::new(
            "motion.time.source_frame_times_seconds must contain 150 finite values",
        ));
    }
    for (index, value) in time.source_frame_times_seconds.iter().enumerate() {
        let expected = exact_source_second(index, time.nominal_fps);
        if *value != expected {
            return Err(DynamicArchiveError::new(format!(
                "motion.time.source_frame_times_seconds[{index}]={value:?} is inconsistent with nominal_fps; expected {expected:?}",
            )));
        }
    }
    expect_literal(
        &time.interval,
        "finite_nonperiodic".to_string(),
        "motion.time.interval",
    )?;
    expect_literal(
        &motion.interpolation.kind,
        "clamped_nonperiodic_catmull_rom".to_string(),
        "motion.interpolation.kind",
    )?;
    expect_literal(
        &motion.interpolation.query_boundary,
        "clamp".to_string(),
        "motion.interpolation.query_boundary",
    )?;
    expect_literal(
        &motion.interpolation.control_point_boundary,
        "clamp_indices".to_string(),
        "motion.interpolation.control_point_boundary",
    )?;

    if motion.placement_transforms.len() != PLACEMENT_TRANSFORMS.len() {
        return Err(DynamicArchiveError::new(
            "motion.placement_transforms must contain exactly two entries",
        ));
    }
    let names = ["identity", "quarter_turn_z_positive"];
    for (index, transform) in motion.placement_transforms.iter().enumerate() {
        if transform.id as usize != index
            || transform.name != names[index]
            || transform.matrix != PLACEMENT_TRANSFORMS[index]
        {
            return Err(DynamicArchiveError::new(format!(
                "motion.placement_transforms[{index}] is not the approved transform"
            )));
        }
    }

    let source = wire.source;
    expect_literal(
        &source.asset_type,
        "normalized_dynamic_source".to_string(),
        "source.asset_type",
    )?;
    expect_literal(
        &source.schema_version,
        schema_version,
        "source.schema_version",
    )?;
    if source.backend.is_empty() || source.adapter_version == 0 {
        return Err(DynamicArchiveError::new(
            "source backend and adapter_version are invalid",
        ));
    }
    if schema_version == 3 {
        if source.basis_count.is_some() || source.top_k.is_some() {
            return Err(DynamicArchiveError::new(
                "version 3 source must not contain legacy basis fields",
            ));
        }
        let source_banks = source
            .basis_banks
            .as_ref()
            .ok_or_else(|| DynamicArchiveError::new("version 3 source.basis_banks is required"))?;
        let BasisManifest::Separate(motion_banks) = &bases else {
            unreachable!()
        };
        for (source_bank, motion_bank) in [
            (&source_banks.translation, &motion_banks.translation),
            (&source_banks.rotation, &motion_banks.rotation),
            (&source_banks.scale, &motion_banks.scale),
        ] {
            if source_bank.basis_count != motion_bank.basis_count
                || source_bank.top_k != motion_bank.top_k
            {
                return Err(DynamicArchiveError::new(
                    "source basis banks must match motion",
                ));
            }
        }
    } else if source.basis_banks.is_some()
        || source.basis_count != Some(basis_count)
        || source.top_k != Some(top_k)
    {
        return Err(DynamicArchiveError::new(
            "source basis_count/top_k must match motion",
        ));
    }
    let run_id = match source.run_id {
        Value::Null => None,
        Value::String(value) if !value.is_empty() => Some(value),
        _ => {
            return Err(DynamicArchiveError::new(
                "source.run_id must be nonempty or null",
            ));
        }
    };
    let encoder = match source.encoder {
        Value::Null => None,
        value => {
            let encoder: EncoderWire = serde_json::from_value(value).map_err(|error| {
                DynamicArchiveError::new(format!("source.encoder is invalid: {error}"))
            })?;
            if encoder.algorithm.is_empty() || encoder.algorithm_version == 0 {
                return Err(DynamicArchiveError::new("source.encoder is invalid"));
            }
            Some((encoder.algorithm, encoder.algorithm_version))
        }
    };

    Ok(DynamicManifest {
        schema_version,
        tile_count: wire.tile_count,
        members,
        motion: MotionManifest {
            basis_count,
            top_k,
            bases,
            source_to_target_scale: motion.source_to_target_scale,
            time: MotionTime {
                sample_times: time.sample_times,
                source_frame_times_normalized: time.source_frame_times_normalized,
                source_frame_times_seconds: time.source_frame_times_seconds,
                nominal_fps: time.nominal_fps,
            },
            placement_transforms: PLACEMENT_TRANSFORMS
                .map(|matrix| matrix.map(|row| row.map(|value| value as f32))),
            coefficient_binary_version: motion.coefficient_binary_version,
        },
        source: SourceSummary {
            backend: source.backend,
            adapter_version: source.adapter_version,
            run_id,
            encoder,
        },
        extensions: wire.extensions,
    })
}

fn read_u32(bytes: &[u8], offset: usize, location: &str) -> Result<u32, DynamicArchiveError> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| DynamicArchiveError::new(format!("{location} header is truncated")))?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

pub fn parse_basis(
    bytes: &[u8],
    manifest: &DynamicManifest,
) -> Result<BasisBank, DynamicArchiveError> {
    if !matches!(manifest.motion.bases, BasisManifest::Legacy) {
        return Err(DynamicArchiveError::new(
            "legacy basis parser cannot parse separate basis banks",
        ));
    }
    parse_basis_payload(bytes, manifest.motion.basis_count, 1, MOTION_DIMENSION)
}

pub fn parse_basis_banks(
    payloads: [&[u8]; 3],
    manifest: &DynamicManifest,
) -> Result<BasisBanks, DynamicArchiveError> {
    let BasisManifest::Separate(descriptors) = &manifest.motion.bases else {
        return Err(DynamicArchiveError::new(
            "separate basis parser requires a version 3 manifest",
        ));
    };
    Ok(BasisBanks {
        translation: parse_basis_payload(payloads[0], descriptors.translation.basis_count, 2, 3)?,
        rotation: parse_basis_payload(payloads[1], descriptors.rotation.basis_count, 2, 3)?,
        scale: parse_basis_payload(payloads[2], descriptors.scale.basis_count, 2, 3)?,
    })
}

fn parse_basis_payload(
    bytes: &[u8],
    expected_basis_count: usize,
    expected_version: u32,
    expected_dimension: usize,
) -> Result<BasisBank, DynamicArchiveError> {
    const HEADER_SIZE: usize = 24;
    if bytes.len() < HEADER_SIZE {
        return Err(DynamicArchiveError::new("basis header is truncated"));
    }
    if &bytes[..8] != BASIS_MAGIC {
        return Err(DynamicArchiveError::new("basis magic is invalid"));
    }
    let version = read_u32(bytes, 8, "basis")?;
    let basis_count = read_u32(bytes, 12, "basis")? as usize;
    let sample_count = read_u32(bytes, 16, "basis")? as usize;
    let dimension = read_u32(bytes, 20, "basis")? as usize;
    if version != expected_version {
        return Err(DynamicArchiveError::new(
            "basis version does not match the manifest",
        ));
    }
    if basis_count != expected_basis_count
        || sample_count != SAMPLE_COUNT
        || dimension != expected_dimension
    {
        return Err(DynamicArchiveError::new(
            "basis dimensions do not match the manifest",
        ));
    }
    let value_count = basis_count
        .checked_mul(sample_count)
        .and_then(|value| value.checked_mul(dimension))
        .ok_or_else(|| DynamicArchiveError::new("basis value count overflow"))?;
    let payload_size = value_count
        .checked_mul(4)
        .ok_or_else(|| DynamicArchiveError::new("basis byte count overflow"))?;
    let expected_size = HEADER_SIZE
        .checked_add(payload_size)
        .ok_or_else(|| DynamicArchiveError::new("basis byte count overflow"))?;
    if bytes.len() != expected_size {
        return Err(DynamicArchiveError::new(
            "basis payload has truncation or trailing bytes",
        ));
    }
    let mut values = Vec::with_capacity(value_count);
    for chunk in bytes[HEADER_SIZE..].chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(DynamicArchiveError::new(
                "basis payload contains a non-finite value",
            ));
        }
        values.push(value);
    }
    Ok(BasisBank {
        basis_count,
        values,
    })
}

pub fn parse_coefficients(
    bytes: &[u8],
    manifest: &DynamicManifest,
    member: &DynamicMember,
) -> Result<CoefficientPayload, DynamicArchiveError> {
    const LEGACY_HEADER_SIZE: usize = 28;
    const V3_HEADER_SIZE: usize = 32;
    if bytes.len() < LEGACY_HEADER_SIZE {
        return Err(DynamicArchiveError::new("coefficient header is truncated"));
    }
    if &bytes[..8] != COEFFICIENT_MAGIC {
        return Err(DynamicArchiveError::new("coefficient magic is invalid"));
    }
    let version = read_u32(bytes, 8, "coefficient")?;
    let tile = read_u32(bytes, 12, "coefficient")? as usize;
    let lod = read_u32(bytes, 16, "coefficient")? as usize;
    let gaussian_count = read_u32(bytes, 20, "coefficient")? as usize;
    let top_k = read_u32(bytes, 24, "coefficient")? as usize;
    if version != manifest.motion.coefficient_binary_version {
        return Err(DynamicArchiveError::new(
            "coefficient version does not match the manifest",
        ));
    }
    if tile != member.tile || lod != member.lod {
        return Err(DynamicArchiveError::new(
            "coefficient tile/LoD does not match its manifest member",
        ));
    }
    if gaussian_count != member.gaussian_count || top_k != manifest.motion.top_k {
        return Err(DynamicArchiveError::new(
            "coefficient row count/top_k does not match the manifest",
        ));
    }
    let coefficient_count = gaussian_count
        .checked_mul(top_k)
        .ok_or_else(|| DynamicArchiveError::new("coefficient value count overflow"))?;
    let (header_size, ids_size, weights_size, bank_count) = if version == 3 {
        if bytes.len() < V3_HEADER_SIZE {
            return Err(DynamicArchiveError::new(
                "version 3 coefficient header is truncated",
            ));
        }
        let basis_count = read_u32(bytes, 28, "coefficient")? as usize;
        if basis_count != manifest.motion.basis_count || basis_count > 256 {
            return Err(DynamicArchiveError::new(
                "coefficient basis_count does not match the manifest",
            ));
        }
        (
            V3_HEADER_SIZE,
            coefficient_count,
            coefficient_count
                .checked_mul(4)
                .ok_or_else(|| DynamicArchiveError::new("coefficient byte count overflow"))?,
            3,
        )
    } else {
        let size = coefficient_count
            .checked_mul(4)
            .ok_or_else(|| DynamicArchiveError::new("coefficient byte count overflow"))?;
        (LEGACY_HEADER_SIZE, size, size, 1)
    };
    let all_ids_size = ids_size
        .checked_mul(bank_count)
        .ok_or_else(|| DynamicArchiveError::new("coefficient byte count overflow"))?;
    let all_weights_size = weights_size
        .checked_mul(bank_count)
        .ok_or_else(|| DynamicArchiveError::new("coefficient byte count overflow"))?;
    let expected_size = header_size
        .checked_add(all_ids_size)
        .and_then(|value| value.checked_add(all_weights_size))
        .and_then(|value| value.checked_add(gaussian_count))
        .and_then(|value| value.checked_add(if version >= 2 { gaussian_count } else { 0 }))
        .ok_or_else(|| DynamicArchiveError::new("coefficient byte count overflow"))?;
    if bytes.len() != expected_size {
        return Err(DynamicArchiveError::new(
            "coefficient payload has truncation or trailing bytes",
        ));
    }

    let ids_start = header_size;
    let weights_start = ids_start + ids_size * bank_count;
    let data = if version == 3 {
        let basis_ids = bytes[ids_start..weights_start].to_vec();
        if basis_ids
            .iter()
            .any(|&basis_id| basis_id as usize >= manifest.motion.basis_count)
        {
            return Err(DynamicArchiveError::new(
                "coefficient basis ID is out of range",
            ));
        }
        let mut weights = Vec::with_capacity(coefficient_count * 3);
        for chunk in bytes[weights_start..weights_start + weights_size * 3].chunks_exact(4) {
            let weight = f32::from_le_bytes(chunk.try_into().unwrap());
            if !weight.is_finite() {
                return Err(DynamicArchiveError::new(
                    "coefficient weights contain a non-finite value",
                ));
            }
            weights.push(weight);
        }
        CoefficientData::Separate { basis_ids, weights }
    } else {
        let mut basis_ids = Vec::with_capacity(coefficient_count);
        for chunk in bytes[ids_start..ids_start + ids_size].chunks_exact(4) {
            let basis_id = u32::from_le_bytes(chunk.try_into().unwrap());
            if basis_id as usize >= manifest.motion.basis_count {
                return Err(DynamicArchiveError::new(format!(
                    "coefficient basis ID {basis_id} is out of range"
                )));
            }
            basis_ids.push(basis_id);
        }
        let mut weights = Vec::with_capacity(coefficient_count);
        for chunk in bytes[weights_start..weights_start + weights_size].chunks_exact(4) {
            let weight = f32::from_le_bytes(chunk.try_into().unwrap());
            if !weight.is_finite() {
                return Err(DynamicArchiveError::new(
                    "coefficient weights contain a non-finite value",
                ));
            }
            weights.push(weight);
        }
        CoefficientData::Legacy { basis_ids, weights }
    };

    let transforms_start = weights_start + weights_size * bank_count;
    let mut transform_ids = Vec::with_capacity(gaussian_count);
    let transforms_end = transforms_start + gaussian_count;
    for &transform_id in &bytes[transforms_start..transforms_end] {
        if transform_id as usize >= manifest.motion.placement_transforms.len() {
            return Err(DynamicArchiveError::new(format!(
                "coefficient transform ID {transform_id} is out of range"
            )));
        }
        transform_ids.push(transform_id as u32);
    }
    let motion_channel_masks = if version >= 2 {
        let mut masks = Vec::with_capacity(gaussian_count);
        for &mask in &bytes[transforms_end..] {
            if mask & 0xf8 != 0 || (version == 3 && !matches!(mask, 0 | 1 | 3 | 5 | 7)) {
                return Err(DynamicArchiveError::new(
                    "coefficient motion channel mask contains reserved bits",
                ));
            }
            masks.push(mask as u32);
        }
        masks
    } else {
        vec![7; gaussian_count]
    };

    Ok(CoefficientPayload {
        tile,
        lod,
        gaussian_count,
        top_k,
        data,
        transform_ids,
        motion_channel_masks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const BASIS_MAGIC: &[u8; 8] = b"4DGSWTB\0";
    const COEFFICIENT_MAGIC: &[u8; 8] = b"4DGSWTC\0";

    fn valid_manifest_value() -> Value {
        let members = (0..6)
            .map(|lod| {
                json!({
                    "tile": 0,
                    "lod": lod,
                    "gaussians": format!("tile0_lod{lod}.ply"),
                    "coefficients": format!("motion/tile0_lod{lod}.bin"),
                    "gaussian_count": 2
                })
            })
            .collect::<Vec<_>>();
        let sample_times = numpy_linspace(75);
        let frame_times = numpy_linspace(150);
        let seconds = (0..150)
            .map(|i| exact_source_second(i, 30.0))
            .collect::<Vec<_>>();

        json!({
            "asset_type": "dynamic_gswt_archive",
            "schema_version": 1,
            "tile_count": 1,
            "lod_count": 6,
            "members": members,
            "motion": {
                "representation": "shared_sparse_joint_9d_basis",
                "basis_scope": "shared_lod0",
                "basis_source_lod": 0,
                "basis_count": 2,
                "top_k": 2,
                "basis": "motion/basis.bin",
                "basis_binary_version": 1,
                "coefficient_binary_version": 1,
                "channels": [
                    "delta_x", "delta_y", "delta_z",
                    "rotation_vector_x", "rotation_vector_y", "rotation_vector_z",
                    "delta_log_scale_x", "delta_log_scale_y", "delta_log_scale_z"
                ],
                "position_delta_space": "constructor_scaled_source_axes_then_row_transform",
                "rotation_delta": "local_rotation_vector",
                "scale_delta": "log_scale",
                "quaternion_order": "wxyz",
                "source_to_target_scale": 1.25,
                "time": {
                    "sample_times": sample_times,
                    "source_frame_times_normalized": frame_times,
                    "source_frame_times_seconds": seconds,
                    "nominal_fps": 30.0,
                    "interval": "finite_nonperiodic"
                },
                "interpolation": {
                    "kind": "clamped_nonperiodic_catmull_rom",
                    "query_boundary": "clamp",
                    "control_point_boundary": "clamp_indices"
                },
                "placement_transforms": [
                    {"id": 0, "name": "identity", "matrix": [[1,0,0],[0,1,0],[0,0,1]]},
                    {"id": 1, "name": "quarter_turn_z_positive", "matrix": [[0,-1,0],[1,0,0],[0,0,1]]}
                ]
            },
            "source": {
                "asset_type": "normalized_dynamic_source",
                "schema_version": 1,
                "backend": "4dgaussians",
                "adapter_version": 1,
                "run_id": "fixture",
                "basis_count": 2,
                "top_k": 2,
                "encoder": {"algorithm": "joint_svd_omp", "algorithm_version": 1}
            },
            "extensions": {}
        })
    }

    fn numpy_linspace(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| exact_uniform_value(index, count))
            .collect()
    }

    fn valid_manifest_json() -> Vec<u8> {
        serde_json::to_vec(&valid_manifest_value()).unwrap()
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn basis_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = BASIS_MAGIC.to_vec();
        for value in [1_u32, 2, 75, 9] {
            push_u32(&mut bytes, value);
        }
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn coefficient_bytes() -> Vec<u8> {
        let mut bytes = COEFFICIENT_MAGIC.to_vec();
        for value in [1_u32, 0, 0, 2, 2] {
            push_u32(&mut bytes, value);
        }
        for value in [0_u32, 1, 1, 0] {
            push_u32(&mut bytes, value);
        }
        for value in [0.75_f32, 0.25, -0.5, 1.5] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0_u8, 1]);
        bytes
    }

    fn valid_v3_manifest_value() -> Value {
        let mut value = valid_manifest_value();
        value["schema_version"] = json!(3);
        value["motion"]["representation"] = json!("shared_sparse_separate_trs_bases");
        value["motion"]
            .as_object_mut()
            .unwrap()
            .remove("basis_count");
        value["motion"].as_object_mut().unwrap().remove("top_k");
        value["motion"].as_object_mut().unwrap().remove("basis");
        value["motion"]["basis_banks"] = json!({
            "translation": {"basis_count": 2, "top_k": 2, "path": "motion/basis_translation.bin"},
            "rotation": {"basis_count": 2, "top_k": 2, "path": "motion/basis_rotation.bin"},
            "scale": {"basis_count": 2, "top_k": 2, "path": "motion/basis_scale.bin"}
        });
        value["motion"]["basis_binary_version"] = json!(2);
        value["motion"]["coefficient_binary_version"] = json!(3);
        value["motion"]["motion_channel_mask"] = json!({
            "encoding": "uint8_bitfield",
            "translation_bit": 0,
            "rotation_bit": 1,
            "scale_bit": 2,
            "reserved_mask": 248
        });
        value["source"]["schema_version"] = json!(3);
        value["source"]
            .as_object_mut()
            .unwrap()
            .remove("basis_count");
        value["source"].as_object_mut().unwrap().remove("top_k");
        value["source"]["basis_banks"] = json!({
            "translation": {"basis_count": 2, "top_k": 2},
            "rotation": {"basis_count": 2, "top_k": 2},
            "scale": {"basis_count": 2, "top_k": 2}
        });
        value
    }

    fn v3_coefficient_bytes() -> Vec<u8> {
        let mut bytes = COEFFICIENT_MAGIC.to_vec();
        for value in [3_u32, 0, 0, 2, 2, 2] {
            push_u32(&mut bytes, value);
        }
        bytes.extend_from_slice(&[0, 1, 1, 0]);
        bytes.extend_from_slice(&[1, 0, 0, 1]);
        bytes.extend_from_slice(&[0, 0, 1, 1]);
        for bank in [
            [0.75_f32, 0.25, -0.5, 1.5],
            [1.0_f32, 0.0, 0.25, 0.75],
            [0.5_f32, 0.5, 1.25, -0.25],
        ] {
            for value in bank {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&[0, 1]);
        bytes.extend_from_slice(&[1, 7]);
        bytes
    }

    #[test]
    fn parses_valid_version_one_manifest() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        assert_eq!(manifest.tile_count, 1);
        assert_eq!(manifest.members.len(), 6);
        assert_eq!(manifest.motion.basis_count, 2);
        assert_eq!(manifest.motion.top_k, 2);
        assert_eq!(manifest.motion.time.source_duration_seconds(), 149.0 / 30.0);
    }

    #[test]
    fn parses_v3_manifest_and_uint8_bank_coefficients() {
        let manifest =
            parse_manifest(&serde_json::to_vec(&valid_v3_manifest_value()).unwrap()).unwrap();
        assert!(matches!(manifest.motion.bases, BasisManifest::Separate(_)));

        let coefficients =
            parse_coefficients(&v3_coefficient_bytes(), &manifest, &manifest.members[0]).unwrap();
        let CoefficientData::Separate { basis_ids, weights } = &coefficients.data else {
            panic!("version 3 must not be represented as legacy joint coefficients");
        };
        assert_eq!(basis_ids, &vec![0_u8, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1]);
        assert_eq!(weights.len(), 12);
        assert_eq!(coefficients.motion_channel_masks, vec![1, 7]);
    }

    #[test]
    fn rejects_v3_uint8_basis_id_before_merge() {
        let manifest =
            parse_manifest(&serde_json::to_vec(&valid_v3_manifest_value()).unwrap()).unwrap();
        let mut bytes = v3_coefficient_bytes();
        bytes[32] = 2;
        let error = parse_coefficients(&bytes, &manifest, &manifest.members[0]).unwrap_err();
        assert!(error.to_string().contains("basis ID"), "{error}");
    }

    #[test]
    fn rejects_unknown_manifest_fields() {
        let mut value = valid_manifest_value();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), json!(1));
        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("unknown"));
    }

    #[test]
    fn rejects_duplicate_json_keys() {
        let json = String::from_utf8(valid_manifest_json()).unwrap();
        let duplicate = json.replacen("\"tile_count\":1", "\"tile_count\":1,\"tile_count\":1", 1);
        let error = parse_manifest(duplicate.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn rejects_boolean_integer_fields() {
        let mut value = valid_manifest_value();
        value["schema_version"] = json!(true);
        assert!(parse_manifest(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn rejects_boolean_placement_matrix_values() {
        let mut value = valid_manifest_value();
        value["motion"]["placement_transforms"][0]["matrix"][0][0] = json!(true);
        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("placement_transforms"));
    }

    #[test]
    fn rejects_members_outside_exact_tile_major_lod_major_order() {
        let mut value = valid_manifest_value();
        value["members"].as_array_mut().unwrap().swap(0, 1);
        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("order"));
    }

    #[test]
    fn rejects_nonuniform_sample_times() {
        let mut value = valid_manifest_value();
        value["motion"]["time"]["sample_times"][1] = json!(0.25);
        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("sample_times"));
    }

    #[test]
    fn rejects_one_ulp_sample_time_perturbation() {
        let mut value = valid_manifest_value();
        let original = value["motion"]["time"]["sample_times"][19]
            .as_f64()
            .unwrap();
        value["motion"]["time"]["sample_times"][19] = json!(f64::from_bits(original.to_bits() + 1));

        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(error.to_string().contains("sample_times"), "{error}");
    }

    #[test]
    fn rejects_one_ulp_source_second_perturbation() {
        let mut value = valid_manifest_value();
        let original = value["motion"]["time"]["source_frame_times_seconds"][37]
            .as_f64()
            .unwrap();
        value["motion"]["time"]["source_frame_times_seconds"][37] =
            json!(f64::from_bits(original.to_bits() + 1));

        let error = parse_manifest(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(
            error.to_string().contains("source_frame_times_seconds"),
            "{error}"
        );
    }

    #[test]
    fn parses_exact_basis_binary() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let values = vec![0.25_f32; 2 * 75 * 9];
        let basis = parse_basis(&basis_bytes(&values), &manifest).unwrap();
        assert_eq!(basis.basis_count, 2);
        assert_eq!(basis.values, values);
    }

    #[test]
    fn rejects_nonfinite_basis_values() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut values = vec![0.0_f32; 2 * 75 * 9];
        values[17] = f32::NAN;
        let error = parse_basis(&basis_bytes(&values), &manifest).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }

    #[test]
    fn rejects_basis_trailing_bytes() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut bytes = basis_bytes(&vec![0.0_f32; 2 * 75 * 9]);
        bytes.push(0);
        assert!(parse_basis(&bytes, &manifest).is_err());
    }

    #[test]
    fn parses_exact_coefficient_binary() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let coefficients =
            parse_coefficients(&coefficient_bytes(), &manifest, &manifest.members[0]).unwrap();
        let CoefficientData::Legacy { basis_ids, weights } = &coefficients.data else {
            unreachable!()
        };
        assert_eq!(basis_ids, &vec![0, 1, 1, 0]);
        assert_eq!(weights, &vec![0.75, 0.25, -0.5, 1.5]);
        assert_eq!(coefficients.transform_ids, vec![0, 1]);
    }

    #[test]
    fn rejects_out_of_range_basis_ids() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut bytes = coefficient_bytes();
        bytes[28..32].copy_from_slice(&2_u32.to_le_bytes());
        let error = parse_coefficients(&bytes, &manifest, &manifest.members[0]).unwrap_err();
        assert!(error.to_string().contains("basis ID"));
    }

    #[test]
    fn rejects_out_of_range_transform_ids() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut bytes = coefficient_bytes();
        *bytes.last_mut().unwrap() = 2;
        let error = parse_coefficients(&bytes, &manifest, &manifest.members[0]).unwrap_err();
        assert!(error.to_string().contains("transform ID"));
    }

    #[test]
    fn rejects_coefficient_tile_lod_and_row_mismatch() {
        let manifest = parse_manifest(&valid_manifest_json()).unwrap();
        let mut bytes = coefficient_bytes();
        bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());
        let error = parse_coefficients(&bytes, &manifest, &manifest.members[0]).unwrap_err();
        assert!(error.to_string().contains("tile/LoD"));
    }
}
