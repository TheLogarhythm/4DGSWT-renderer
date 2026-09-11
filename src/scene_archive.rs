use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::io::{Cursor, Read};

use crate::dynamic_archive::{
    BasisBank, BasisBanks, BasisManifest, CoefficientData, parse_basis, parse_basis_banks,
    parse_coefficients, parse_manifest,
};
use crate::motion::CanonicalGaussian;
use crate::scene::{Scene, parse_tile_filename};

const MIB: u64 = 1024 * 1024;
const MAX_COMPRESSED_ARCHIVE_BYTES: u64 = 512 * MIB;
const MAX_DECOMPRESSED_ENTRY_BYTES: u64 = 256 * MIB;
const MAX_DECOMPRESSED_ARCHIVE_BYTES: u64 = 1024 * MIB;
const MAX_ZIP_ENTRY_COUNT: usize = 512;
const LOD_COUNT: usize = 6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchiveLoadError(String);

impl ArchiveLoadError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ArchiveLoadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ArchiveLoadError {}

fn validate_archive_input_size(size: usize) -> Result<(), ArchiveLoadError> {
    if size as u128 > MAX_COMPRESSED_ARCHIVE_BYTES as u128 {
        return Err(ArchiveLoadError::new(format!(
            "compressed archive is {size} bytes; the renderer limit is {MAX_COMPRESSED_ARCHIVE_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_archive_entry_count(count: usize) -> Result<(), ArchiveLoadError> {
    if count > MAX_ZIP_ENTRY_COUNT {
        return Err(ArchiveLoadError::new(format!(
            "ZIP contains {count} entries; the renderer limit is {MAX_ZIP_ENTRY_COUNT} entries"
        )));
    }
    Ok(())
}

fn add_declared_entry_size(
    index: usize,
    size: u64,
    aggregate_size: &mut u64,
) -> Result<(), ArchiveLoadError> {
    if size > MAX_DECOMPRESSED_ENTRY_BYTES {
        return Err(ArchiveLoadError::new(format!(
            "ZIP entry {index} declares {size} decompressed bytes; the per-entry limit is {MAX_DECOMPRESSED_ENTRY_BYTES} bytes"
        )));
    }
    *aggregate_size = aggregate_size
        .checked_add(size)
        .ok_or_else(|| ArchiveLoadError::new("ZIP declared size overflow"))?;
    if *aggregate_size > MAX_DECOMPRESSED_ARCHIVE_BYTES {
        return Err(ArchiveLoadError::new(format!(
            "ZIP declares more than {MAX_DECOMPRESSED_ARCHIVE_BYTES} decompressed bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_declared_entry_sizes(
    sizes: impl IntoIterator<Item = u64>,
) -> Result<(), ArchiveLoadError> {
    let mut aggregate_size = 0_u64;
    for (index, size) in sizes.into_iter().enumerate() {
        add_declared_entry_size(index, size, &mut aggregate_size)?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct MemberMotion {
    pub canonical: Vec<CanonicalGaussian>,
    pub coefficients: MemberCoefficientData,
    pub transform_ids: Vec<u32>,
    pub motion_channel_masks: Vec<u32>,
}

#[derive(Clone, Debug)]
pub enum MemberCoefficientData {
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
pub enum BasisData {
    Legacy(BasisBank),
    Separate(BasisBanks),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DynamicArchiveSummary {
    pub schema_version: u32,
    pub tile_count: usize,
    pub lod_count: usize,
    pub basis_count: usize,
    pub top_k: usize,
    pub total_rows: usize,
    pub backend: String,
}

#[derive(Clone, Debug)]
pub struct DynamicAsset {
    pub bases: BasisData,
    pub top_k: usize,
    pub source_duration_seconds: f32,
    pub members: Vec<Vec<MemberMotion>>,
    pub summary: DynamicArchiveSummary,
}

pub struct LoadedArchive {
    pub scenes: Vec<Vec<Scene>>,
    pub dynamic: Option<DynamicAsset>,
}

impl Debug for LoadedArchive {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadedArchive")
            .field("lod_count", &self.scenes.len())
            .field(
                "tile_counts",
                &self.scenes.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .field("dynamic", &self.dynamic)
            .finish()
    }
}

fn inspect_names(
    archive: &mut zip::ZipArchive<Cursor<Vec<u8>>>,
) -> Result<Vec<String>, ArchiveLoadError> {
    validate_archive_entry_count(archive.len())?;
    let mut names = Vec::with_capacity(archive.len());
    let mut aggregate_size = 0_u64;
    for index in 0..archive.len() {
        let file = archive
            .by_index(index)
            .map_err(|error| ArchiveLoadError::new(format!("ZIP entry {index}: {error}")))?;
        add_declared_entry_size(index, file.size(), &mut aggregate_size)?;
        names.push(file.name().to_string());
    }
    Ok(names)
}

fn read_entry(
    archive: &mut zip::ZipArchive<Cursor<Vec<u8>>>,
    index: usize,
) -> Result<Vec<u8>, ArchiveLoadError> {
    let mut file = archive
        .by_index(index)
        .map_err(|error| ArchiveLoadError::new(format!("ZIP entry {index}: {error}")))?;
    let size = usize::try_from(file.size())
        .map_err(|_| ArchiveLoadError::new(format!("ZIP entry {index} is too large")))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|error| {
        ArchiveLoadError::new(format!(
            "ZIP entry {index} needs {size} bytes but memory reservation failed: {error}"
        ))
    })?;
    bytes.resize(size, 0);
    file.read_exact(&mut bytes)
        .map_err(|error| ArchiveLoadError::new(format!("ZIP entry {index}: {error}")))?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|error| ArchiveLoadError::new(format!("ZIP entry {index}: {error}")))?
        != 0
    {
        return Err(ArchiveLoadError::new(format!(
            "ZIP entry {index} produced more than its declared {size} bytes"
        )));
    }
    Ok(bytes)
}

fn expected_dynamic_names(manifest: &crate::dynamic_archive::DynamicManifest) -> Vec<String> {
    let mut names = Vec::with_capacity(4 + 2 * manifest.members.len());
    names.push("manifest.json".to_string());
    match &manifest.motion.bases {
        BasisManifest::Legacy => names.push("motion/basis.bin".to_string()),
        BasisManifest::Separate(banks) => names.extend([
            banks.translation.path.clone(),
            banks.rotation.path.clone(),
            banks.scale.path.clone(),
        ]),
    }
    for member in &manifest.members {
        names.push(member.gaussians.clone());
        names.push(member.coefficients.clone());
    }
    names
}

fn validate_exact_dynamic_names(
    actual: &[String],
    expected: &[String],
) -> Result<(), ArchiveLoadError> {
    if actual.len() != expected.len() {
        return Err(ArchiveLoadError::new(format!(
            "dynamic ZIP entry count is {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }
    let mut unique = HashSet::with_capacity(actual.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if !unique.insert(actual.as_str()) {
            return Err(ArchiveLoadError::new(format!(
                "dynamic ZIP entry '{actual}' is duplicated"
            )));
        }
        if actual != expected {
            return Err(ArchiveLoadError::new(format!(
                "dynamic ZIP entry {index} is '{actual}', expected '{expected}'"
            )));
        }
    }
    Ok(())
}

fn load_dynamic(
    mut archive: zip::ZipArchive<Cursor<Vec<u8>>>,
    names: Vec<String>,
) -> Result<LoadedArchive, ArchiveLoadError> {
    let manifest_index = names
        .iter()
        .position(|name| name == "manifest.json")
        .ok_or_else(|| ArchiveLoadError::new("dynamic archive is missing manifest.json"))?;
    let manifest_bytes = read_entry(&mut archive, manifest_index)?;
    let manifest = parse_manifest(&manifest_bytes)
        .map_err(|error| ArchiveLoadError::new(format!("manifest.json: {error}")))?;
    let expected_names = expected_dynamic_names(&manifest);
    validate_exact_dynamic_names(&names, &expected_names)?;

    let (bases, member_start) = match &manifest.motion.bases {
        BasisManifest::Legacy => {
            let bytes = read_entry(&mut archive, 1)?;
            let basis = parse_basis(&bytes, &manifest)
                .map_err(|error| ArchiveLoadError::new(format!("motion/basis.bin: {error}")))?;
            (BasisData::Legacy(basis), 2)
        }
        BasisManifest::Separate(_) => {
            let translation = read_entry(&mut archive, 1)?;
            let rotation = read_entry(&mut archive, 2)?;
            let scale = read_entry(&mut archive, 3)?;
            let banks = parse_basis_banks([&translation, &rotation, &scale], &manifest)
                .map_err(|error| ArchiveLoadError::new(format!("motion basis bank: {error}")))?;
            (BasisData::Separate(banks), 4)
        }
    };

    let mut scenes = (0..LOD_COUNT)
        .map(|_| Vec::with_capacity(manifest.tile_count))
        .collect::<Vec<_>>();
    let mut motion_members = (0..LOD_COUNT)
        .map(|_| Vec::with_capacity(manifest.tile_count))
        .collect::<Vec<_>>();
    let mut total_rows = 0_usize;

    for (member_index, member) in manifest.members.iter().enumerate() {
        let ply_bytes = read_entry(&mut archive, member_start + member_index * 2)?;
        let parsed = Scene::from_ply_bytes_with_identity(ply_bytes).map_err(|error| {
            ArchiveLoadError::new(format!(
                "{} (tile {}, LoD {}): {error}",
                member.gaussians, member.tile, member.lod
            ))
        })?;
        if parsed.scene.splat_count != member.gaussian_count {
            return Err(ArchiveLoadError::new(format!(
                "{} row count is {}, manifest declares {}",
                member.gaussians, parsed.scene.splat_count, member.gaussian_count
            )));
        }

        let coefficient_bytes = read_entry(&mut archive, member_start + 1 + member_index * 2)?;
        let coefficients = parse_coefficients(&coefficient_bytes, &manifest, member)
            .map_err(|error| ArchiveLoadError::new(format!("{}: {error}", member.coefficients)))?;
        if coefficients.gaussian_count != parsed.source_rows.len() {
            return Err(ArchiveLoadError::new(format!(
                "{} row count does not match {}",
                member.coefficients, member.gaussians
            )));
        }

        let coefficient_count = member
            .gaussian_count
            .checked_mul(coefficients.top_k)
            .ok_or_else(|| ArchiveLoadError::new("coefficient gather size overflow"))?;
        let mut legacy_basis_ids = Vec::new();
        let mut bank_basis_ids = Vec::new();
        let mut weights = Vec::new();
        match &coefficients.data {
            CoefficientData::Legacy { .. } => {
                legacy_basis_ids.reserve(coefficient_count);
                weights.reserve(coefficient_count);
            }
            CoefficientData::Separate { .. } => {
                bank_basis_ids.reserve(coefficient_count * 3);
                weights.reserve(coefficient_count * 3);
            }
        }
        let mut transform_ids = Vec::with_capacity(member.gaussian_count);
        let mut motion_channel_masks = Vec::with_capacity(member.gaussian_count);
        for &source_row in &parsed.source_rows {
            let start = source_row
                .checked_mul(coefficients.top_k)
                .ok_or_else(|| ArchiveLoadError::new("coefficient row offset overflow"))?;
            let end = start + coefficients.top_k;
            match &coefficients.data {
                CoefficientData::Legacy {
                    basis_ids,
                    weights: source_weights,
                } => {
                    legacy_basis_ids.extend_from_slice(&basis_ids[start..end]);
                    weights.extend_from_slice(&source_weights[start..end]);
                }
                CoefficientData::Separate {
                    basis_ids,
                    weights: source_weights,
                } => {
                    for bank in 0..3 {
                        let bank_start = bank * coefficient_count + start;
                        let bank_end = bank_start + coefficients.top_k;
                        bank_basis_ids.extend_from_slice(&basis_ids[bank_start..bank_end]);
                        weights.extend_from_slice(&source_weights[bank_start..bank_end]);
                    }
                }
            }
            transform_ids.push(coefficients.transform_ids[source_row]);
            motion_channel_masks.push(coefficients.motion_channel_masks[source_row]);
        }

        total_rows = total_rows
            .checked_add(member.gaussian_count)
            .ok_or_else(|| ArchiveLoadError::new("total Gaussian row count overflow"))?;
        scenes[member.lod].push(parsed.scene);
        motion_members[member.lod].push(MemberMotion {
            canonical: parsed.canonical,
            coefficients: match coefficients.data {
                CoefficientData::Legacy { .. } => MemberCoefficientData::Legacy {
                    basis_ids: legacy_basis_ids,
                    weights,
                },
                CoefficientData::Separate { .. } => MemberCoefficientData::Separate {
                    basis_ids: bank_basis_ids,
                    weights,
                },
            },
            transform_ids,
            motion_channel_masks,
        });
    }

    let source_duration_seconds = manifest.motion.time.source_duration_seconds() as f32;
    let summary = DynamicArchiveSummary {
        schema_version: manifest.schema_version,
        tile_count: manifest.tile_count,
        lod_count: LOD_COUNT,
        basis_count: manifest.motion.basis_count,
        top_k: manifest.motion.top_k,
        total_rows,
        backend: manifest.source.backend.clone(),
    };
    Ok(LoadedArchive {
        scenes,
        dynamic: Some(DynamicAsset {
            bases,
            top_k: manifest.motion.top_k,
            source_duration_seconds,
            members: motion_members,
            summary,
        }),
    })
}

fn load_static(
    mut archive: zip::ZipArchive<Cursor<Vec<u8>>>,
    names: &[String],
) -> Result<LoadedArchive, ArchiveLoadError> {
    let mut entries = BTreeMap::<(usize, usize), usize>::new();
    for (index, name) in names.iter().enumerate() {
        let file = archive
            .by_index(index)
            .map_err(|error| ArchiveLoadError::new(format!("ZIP entry {index}: {error}")))?;
        let enclosed = file.enclosed_name().ok_or_else(|| {
            ArchiveLoadError::new(format!("ZIP entry '{name}' is not a safe relative path"))
        })?;
        let Some(filename) = enclosed.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some((lod, tile)) = parse_tile_filename(filename) else {
            continue;
        };
        if entries.insert((lod, tile), index).is_some() {
            return Err(ArchiveLoadError::new(format!(
                "static archive duplicates tile {tile}, LoD {lod}"
            )));
        }
    }
    if entries.is_empty() {
        return Err(ArchiveLoadError::new(
            "static archive contains no recognized tile PLY or splat members",
        ));
    }
    let max_lod = entries
        .keys()
        .map(|(lod, _)| *lod)
        .max()
        .ok_or_else(|| ArchiveLoadError::new("static archive has no LoD index"))?;
    let max_tile = entries
        .keys()
        .map(|(_, tile)| *tile)
        .max()
        .ok_or_else(|| ArchiveLoadError::new("static archive has no tile index"))?;
    let mut scenes = Vec::with_capacity(max_lod + 1);
    for lod in 0..=max_lod {
        let mut lod_scenes = Vec::with_capacity(max_tile + 1);
        for tile in 0..=max_tile {
            let index = entries.get(&(lod, tile)).copied().ok_or_else(|| {
                ArchiveLoadError::new(format!("static archive is missing tile {tile}, LoD {lod}"))
            })?;
            let filename = names[index].clone();
            let bytes = read_entry(&mut archive, index)?;
            let scene = if filename.ends_with(".ply") {
                Scene::from_ply_bytes(bytes)
                    .map_err(|error| ArchiveLoadError::new(format!("{filename}: {error}")))?
            } else if filename.ends_with(".splat") {
                if bytes.len() % 32 != 0 {
                    return Err(ArchiveLoadError::new(format!(
                        "{filename}: splat payload length is not divisible by 32"
                    )));
                }
                let mut scene = Scene::new();
                scene.splat_count = bytes.len() / 32;
                scene.buffer = bytes;
                scene
            } else {
                return Err(ArchiveLoadError::new(format!(
                    "{filename}: unsupported static tile extension"
                )));
            };
            lod_scenes.push(scene);
        }
        scenes.push(lod_scenes);
    }
    Ok(LoadedArchive {
        scenes,
        dynamic: None,
    })
}

pub fn load_archive_bytes(bytes: Vec<u8>) -> Result<LoadedArchive, ArchiveLoadError> {
    validate_archive_input_size(bytes.len())?;
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|error| ArchiveLoadError::new(format!("invalid ZIP archive: {error}")))?;
    let names = inspect_names(&mut archive)?;
    if names.iter().any(|name| name == "manifest.json") {
        return load_dynamic(archive, names);
    }
    if names
        .iter()
        .any(|name| name.starts_with("motion/") || name.ends_with(".bin"))
    {
        return Err(ArchiveLoadError::new(
            "archive contains motion payloads but no manifest.json",
        ));
    }
    load_static(archive, &names)
}

pub async fn pick_archive() -> Result<Option<LoadedArchive>, ArchiveLoadError> {
    let Some(file) = rfd::AsyncFileDialog::new()
        .set_title("Upload GSWT archive (.zip)")
        .add_filter("GSWT archive", &["zip"])
        .pick_file()
        .await
    else {
        return Ok(None);
    };
    load_archive_bytes(file.read().await).map(Some)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use serde_json::json;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    use super::{
        MAX_COMPRESSED_ARCHIVE_BYTES, MAX_DECOMPRESSED_ARCHIVE_BYTES, MAX_DECOMPRESSED_ENTRY_BYTES,
        MAX_ZIP_ENTRY_COUNT, MemberCoefficientData, load_archive_bytes,
        validate_archive_entry_count, validate_archive_input_size, validate_declared_entry_sizes,
    };

    fn ply_rows(rows: &[[f32; 14]]) -> Vec<u8> {
        let properties = [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
            "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let mut bytes = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{}end_header\n",
            rows.len(),
            properties
                .iter()
                .map(|name| format!("property float {name}\n"))
                .collect::<String>()
        )
        .into_bytes();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn gaussian_rows() -> Vec<[f32; 14]> {
        vec![
            [
                10.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
            [
                20.0,
                0.0,
                0.0,
                0.4,
                0.5,
                0.6,
                0.0,
                std::f32::consts::LN_2,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                30.0,
                0.0,
                0.0,
                0.7,
                0.8,
                0.9,
                4.0_f32.ln(),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        ]
    }

    fn manifest_bytes() -> Vec<u8> {
        let members = (0..6)
            .map(|lod| {
                json!({
                    "tile": 0,
                    "lod": lod,
                    "gaussians": format!("tile0_lod{lod}.ply"),
                    "coefficients": format!("motion/tile0_lod{lod}.bin"),
                    "gaussian_count": 3
                })
            })
            .collect::<Vec<_>>();
        let sample_times = numpy_linspace(75);
        let frame_times = numpy_linspace(150);
        let fps = std::hint::black_box(30.0);
        let seconds = (0..150).map(|i| i as f64 / fps).collect::<Vec<_>>();
        serde_json::to_vec(&json!({
            "asset_type": "dynamic_gswt_archive",
            "schema_version": 2,
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
                "coefficient_binary_version": 2,
                "motion_channel_mask": {
                    "encoding": "uint8_bitfield",
                    "translation_bit": 0,
                    "rotation_bit": 1,
                    "scale_bit": 2,
                    "reserved_mask": 248
                },
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
                "schema_version": 2,
                "backend": "fixture",
                "adapter_version": 1,
                "run_id": "archive-loader-test",
                "basis_count": 2,
                "top_k": 2,
                "encoder": {"algorithm": "joint_svd_omp", "algorithm_version": 1}
            },
            "extensions": {}
        }))
        .unwrap()
    }

    fn numpy_linspace(count: usize) -> Vec<f64> {
        let step = std::hint::black_box(1.0 / (count - 1) as f64);
        let mut values = (0..count)
            .map(|index| index as f64 * step)
            .collect::<Vec<_>>();
        values[count - 1] = 1.0;
        values
    }

    fn basis_bytes() -> Vec<u8> {
        let mut bytes = b"4DGSWTB\0".to_vec();
        for value in [1_u32, 2, 75, 9] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..2 * 75 * 9 {
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        bytes
    }

    fn coefficient_bytes(lod: u32) -> Vec<u8> {
        let mut bytes = b"4DGSWTC\0".to_vec();
        for value in [2_u32, 0, lod, 3, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0_u32, 1, 1, 0, 0, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0_u8, 1, 0]);
        bytes.extend_from_slice(&[3_u8, 7, 1]);
        bytes
    }

    fn v3_manifest_bytes() -> Vec<u8> {
        let mut value: serde_json::Value = serde_json::from_slice(&manifest_bytes()).unwrap();
        value["schema_version"] = json!(3);
        value["motion"]["representation"] = json!("shared_sparse_separate_trs_bases");
        value["motion"]
            .as_object_mut()
            .unwrap()
            .remove("basis_count");
        value["motion"].as_object_mut().unwrap().remove("top_k");
        value["motion"].as_object_mut().unwrap().remove("basis");
        value["motion"]["basis_banks"] = json!({
            "translation": {"basis_count": 8, "top_k": 1, "path": "motion/basis_translation.bin"},
            "rotation": {"basis_count": 8, "top_k": 1, "path": "motion/basis_rotation.bin"},
            "scale": {"basis_count": 8, "top_k": 1, "path": "motion/basis_scale.bin"}
        });
        value["motion"]["basis_binary_version"] = json!(2);
        value["motion"]["coefficient_binary_version"] = json!(3);
        value["source"]["schema_version"] = json!(3);
        value["source"]
            .as_object_mut()
            .unwrap()
            .remove("basis_count");
        value["source"].as_object_mut().unwrap().remove("top_k");
        value["source"]["basis_banks"] = json!({
            "translation": {"basis_count": 8, "top_k": 1},
            "rotation": {"basis_count": 8, "top_k": 1},
            "scale": {"basis_count": 8, "top_k": 1}
        });
        serde_json::to_vec(&value).unwrap()
    }

    fn v3_basis_bytes() -> Vec<u8> {
        let mut bytes = b"4DGSWTB\0".to_vec();
        for value in [2_u32, 8, 75, 3] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for _ in 0..8 * 75 * 3 {
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        bytes
    }

    fn v3_coefficient_bytes(lod: u32) -> Vec<u8> {
        let mut bytes = b"4DGSWTC\0".to_vec();
        for value in [3_u32, 0, lod, 3, 1, 8] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0, 1, 2]);
        bytes.extend_from_slice(&[3, 4, 5]);
        bytes.extend_from_slice(&[6, 7, 0]);
        for bank in 0..3 {
            for row in 0..3 {
                bytes.extend_from_slice(&(10.0 * bank as f32 + row as f32).to_le_bytes());
            }
        }
        bytes.extend_from_slice(&[0, 1, 0]);
        bytes.extend_from_slice(&[1, 3, 7]);
        bytes
    }

    fn valid_v3_dynamic_entries() -> Vec<(String, Vec<u8>)> {
        let mut entries = vec![("manifest.json".to_string(), v3_manifest_bytes())];
        for bank in ["translation", "rotation", "scale"] {
            entries.push((format!("motion/basis_{bank}.bin"), v3_basis_bytes()));
        }
        let ply = ply_rows(&gaussian_rows());
        for lod in 0..6 {
            entries.push((format!("tile0_lod{lod}.ply"), ply.clone()));
            entries.push((
                format!("motion/tile0_lod{lod}.bin"),
                v3_coefficient_bytes(lod),
            ));
        }
        entries
    }

    fn zip_bytes(entries: &[(String, Vec<u8>)]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut writer = ZipWriter::new(cursor);
        for (name, bytes) in entries {
            writer
                .start_file(name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn valid_dynamic_entries() -> Vec<(String, Vec<u8>)> {
        let mut entries = vec![
            ("manifest.json".to_string(), manifest_bytes()),
            ("motion/basis.bin".to_string(), basis_bytes()),
        ];
        let ply = ply_rows(&gaussian_rows());
        for lod in 0..6 {
            entries.push((format!("tile0_lod{lod}.ply"), ply.clone()));
            entries.push((format!("motion/tile0_lod{lod}.bin"), coefficient_bytes(lod)));
        }
        entries
    }

    #[test]
    fn loads_a_rectangular_static_archive() {
        let ply = ply_rows(&gaussian_rows()[..1]);
        let loaded = load_archive_bytes(zip_bytes(&[
            ("tile0_lod0.ply".into(), ply.clone()),
            ("tile0_lod1.ply".into(), ply),
        ]))
        .unwrap();

        assert!(loaded.dynamic.is_none());
        assert_eq!(loaded.scenes.len(), 2);
        assert_eq!(loaded.scenes[0].len(), 1);
        assert_eq!(loaded.scenes[1][0].splat_count, 1);
    }

    #[test]
    fn loads_dynamic_members_and_gathers_coefficients_by_ply_permutation() {
        let loaded = load_archive_bytes(zip_bytes(&valid_dynamic_entries())).unwrap();
        let dynamic = loaded.dynamic.unwrap();

        assert_eq!(loaded.scenes.len(), 6);
        assert_eq!(dynamic.summary.total_rows, 18);
        assert_eq!(dynamic.members[0][0].canonical[0].position[0], 20.0);
        let MemberCoefficientData::Legacy { basis_ids, weights } =
            &dynamic.members[0][0].coefficients
        else {
            unreachable!()
        };
        assert_eq!(basis_ids, &vec![1, 0, 0, 0, 0, 1]);
        assert_eq!(weights, &vec![20.0, 21.0, 30.0, 31.0, 10.0, 11.0]);
        assert_eq!(dynamic.members[0][0].transform_ids, vec![1, 0, 0]);
        assert_eq!(dynamic.members[0][0].motion_channel_masks, vec![7, 1, 3]);
        assert_eq!(dynamic.summary.schema_version, 2);
    }

    #[test]
    fn v3_loader_gathers_every_bank_through_the_ply_permutation() {
        let loaded = load_archive_bytes(zip_bytes(&valid_v3_dynamic_entries())).unwrap();
        let dynamic = loaded.dynamic.unwrap();
        let MemberCoefficientData::Separate { basis_ids, weights } =
            &dynamic.members[0][0].coefficients
        else {
            panic!("v3 member must retain separate-bank coefficients");
        };
        assert_eq!(basis_ids, &[1_u8, 4, 7, 2, 5, 0, 0, 3, 6]);
        assert_eq!(
            weights,
            &[1.0, 11.0, 21.0, 2.0, 12.0, 22.0, 0.0, 10.0, 20.0]
        );
        assert_eq!(dynamic.members[0][0].transform_ids, vec![1, 0, 0]);
        assert_eq!(dynamic.members[0][0].motion_channel_masks, vec![3, 7, 1]);
    }

    #[test]
    fn manifest_parse_failure_never_falls_back_to_static() {
        let mut entries = valid_dynamic_entries();
        entries[0].1 = b"{}".to_vec();
        let error = load_archive_bytes(zip_bytes(&entries)).unwrap_err();
        assert!(error.to_string().contains("manifest"), "{error}");
    }

    #[test]
    fn rejects_motion_payloads_without_a_manifest() {
        let entries = vec![
            (
                "tile0_lod0.ply".to_string(),
                ply_rows(&gaussian_rows()[..1]),
            ),
            ("motion/basis.bin".to_string(), basis_bytes()),
        ];
        let error = load_archive_bytes(zip_bytes(&entries)).unwrap_err();
        assert!(error.to_string().contains("manifest"), "{error}");
    }

    #[test]
    fn rejects_duplicate_extra_and_reordered_dynamic_entries() {
        let original = valid_dynamic_entries();
        let mut duplicate_zip = zip_bytes(&original);
        let old_name = b"tile0_lod1.ply";
        let new_name = b"tile0_lod0.ply";
        for offset in 0..=duplicate_zip.len() - old_name.len() {
            if &duplicate_zip[offset..offset + old_name.len()] == old_name {
                duplicate_zip[offset..offset + old_name.len()].copy_from_slice(new_name);
            }
        }
        assert!(load_archive_bytes(duplicate_zip).is_err());

        let mut variants = Vec::new();
        let mut extra = original.clone();
        extra.push(("notes.txt".into(), b"extra".to_vec()));
        variants.push(extra);
        let mut reordered = original;
        reordered.swap(2, 3);
        variants.push(reordered);

        for entries in variants {
            assert!(load_archive_bytes(zip_bytes(&entries)).is_err());
        }
    }

    #[test]
    fn rejects_ply_and_manifest_row_count_disagreement() {
        let mut entries = valid_dynamic_entries();
        entries[2].1 = ply_rows(&gaussian_rows()[..1]);
        let error = load_archive_bytes(zip_bytes(&entries)).unwrap_err();
        assert!(error.to_string().contains("row count"), "{error}");
    }

    #[test]
    fn rejects_compressed_archives_over_the_wasm_input_budget() {
        assert!(validate_archive_input_size(MAX_COMPRESSED_ARCHIVE_BYTES as usize).is_ok());
        let error =
            validate_archive_input_size(MAX_COMPRESSED_ARCHIVE_BYTES as usize + 1).unwrap_err();
        assert!(error.to_string().contains("compressed"), "{error}");
    }

    #[test]
    fn rejects_archives_with_excessive_central_directory_entries() {
        assert!(validate_archive_entry_count(MAX_ZIP_ENTRY_COUNT).is_ok());
        let error = validate_archive_entry_count(MAX_ZIP_ENTRY_COUNT + 1).unwrap_err();
        assert!(error.to_string().contains("entries"), "{error}");
    }

    #[test]
    fn rejects_oversized_members_and_decompressed_aggregates() {
        let member_error =
            validate_declared_entry_sizes([MAX_DECOMPRESSED_ENTRY_BYTES + 1]).unwrap_err();
        assert!(
            member_error.to_string().contains("entry 0"),
            "{member_error}"
        );

        let aggregate_error = validate_declared_entry_sizes([
            MAX_DECOMPRESSED_ENTRY_BYTES,
            MAX_DECOMPRESSED_ENTRY_BYTES,
            MAX_DECOMPRESSED_ENTRY_BYTES,
            MAX_DECOMPRESSED_ENTRY_BYTES,
            1,
        ])
        .unwrap_err();
        assert!(
            aggregate_error
                .to_string()
                .contains(&MAX_DECOMPRESSED_ARCHIVE_BYTES.to_string()),
            "{aggregate_error}"
        );
    }
}
