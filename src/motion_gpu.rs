use std::borrow::Cow;

use wgpu::util::DeviceExt;

use crate::dynamic_archive::{BasisBank, BasisBanks};
use crate::log;
use crate::motion::{
    ALL_MOTION_MASK, BasisBanksFrame, BasisFrame, CanonicalGaussian, MergedMotion,
    MergedMotionData, MotionState, TimelineSample, sample_basis_banks, sample_basis_into,
};
use crate::motion_behavior::MotionChannelGains;
use crate::motion_controller_palette::{MAX_LOCAL_CONTROLLERS, MotionControllerFrame};
use crate::profiler::MotionDispatchWork;
use crate::scene_archive::DynamicArchiveSummary;
use crate::texture::Texture;

const WORKGROUP_SIZE: u32 = 64;
const HALF_PACKING_LIMIT: f32 = 65_504.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SafeDynamicOutput {
    pub state: MotionState,
    pub covariance_is_dynamic: bool,
}

fn covariance_packable(covariance: [f32; 6]) -> bool {
    covariance.into_iter().all(|value| {
        value.is_finite() && (4.0 * value).is_finite() && (4.0 * value).abs() <= HALF_PACKING_LIMIT
    })
}

fn state_covariance_packable(state: MotionState) -> bool {
    let rotation_norm_squared = state
        .rotation
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    state.rotation.iter().all(|value| value.is_finite())
        && rotation_norm_squared.is_finite()
        && rotation_norm_squared > 0.0
        && state.log_scale.iter().all(|value| value.is_finite())
        && covariance_packable(state.covariance())
}

pub(crate) fn safe_dynamic_output(
    mut state: MotionState,
    canonical: CanonicalGaussian,
) -> SafeDynamicOutput {
    if state.position.iter().any(|value| !value.is_finite()) {
        state.position = canonical.position;
    }
    if state_covariance_packable(state) {
        return SafeDynamicOutput {
            state,
            covariance_is_dynamic: true,
        };
    }
    state.log_scale = canonical.log_scale;
    SafeDynamicOutput {
        covariance_is_dynamic: state_covariance_packable(state),
        state,
    }
}

fn checked_bytes(item_count: usize, item_size: usize, label: &str) -> Result<usize, String> {
    item_count
        .checked_mul(item_size)
        .ok_or_else(|| format!("{label} byte count overflow"))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct CanonicalGpu {
    position: [f32; 4],
    log_scale: [f32; 4],
    rotation: [f32; 4],
}

impl From<CanonicalGaussian> for CanonicalGpu {
    fn from(canonical: CanonicalGaussian) -> Self {
        Self {
            position: [
                canonical.position[0],
                canonical.position[1],
                canonical.position[2],
                0.0,
            ],
            log_scale: [
                canonical.log_scale[0],
                canonical.log_scale[1],
                canonical.log_scale[2],
                0.0,
            ],
            rotation: canonical.rotation,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BasisGpu {
    translation: [f32; 4],
    rotation: [f32; 4],
    log_scale: [f32; 4],
}

impl BasisGpu {
    const fn zero() -> Self {
        Self {
            translation: [0.0; 4],
            rotation: [0.0; 4],
            log_scale: [0.0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct PlacementGpu {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionParamsGpu {
    row_start: u32,
    row_count: u32,
    top_k: u32,
    basis_count: u32,
    texture_width: u32,
    motion_enabled: u32,
    blend_alpha: f32,
    diagnostic_row: u32,
    preview_channel_mask: u32,
    separate_banks: u32,
    _padding: [u32; 2],
    translation_master_gain: [f32; 4],
    rotation_scale_gain: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct MotionStateGpu {
    position: [f32; 4],
    log_scale: [f32; 4],
    rotation: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RowChunk {
    start: usize,
    count: usize,
}

fn pack_basis_frame(frame: &BasisFrame) -> Result<Vec<BasisGpu>, String> {
    let mut packed = Vec::with_capacity(frame.values.len());
    pack_basis_frame_into(frame, &mut packed)?;
    Ok(packed)
}

fn pack_basis_frame_into(frame: &BasisFrame, packed: &mut Vec<BasisGpu>) -> Result<(), String> {
    for value in &frame.values {
        if value.iter().any(|channel| !channel.is_finite()) {
            return Err("basis frame contains a non-finite value".to_string());
        }
        packed.push(BasisGpu {
            translation: [value[0], value[1], value[2], 0.0],
            rotation: [value[3], value[4], value[5], 0.0],
            log_scale: [value[6], value[7], value[8], 0.0],
        });
    }
    Ok(())
}

fn pack_basis_banks_frame(frame: &BasisBanksFrame) -> Result<Vec<BasisGpu>, String> {
    let mut packed = Vec::with_capacity(frame.translation.len() * 3);
    pack_basis_banks_frame_into(frame, &mut packed)?;
    Ok(packed)
}

fn pack_basis_banks_frame_into(
    frame: &BasisBanksFrame,
    packed: &mut Vec<BasisGpu>,
) -> Result<(), String> {
    let basis_count = frame.translation.len();
    if basis_count == 0 || frame.rotation.len() != basis_count || frame.scale.len() != basis_count {
        return Err("sampled three-bank basis frames are not aligned".to_string());
    }
    for (bank_index, bank) in [&frame.translation, &frame.rotation, &frame.scale]
        .into_iter()
        .enumerate()
    {
        for value in bank {
            if value.iter().any(|channel| !channel.is_finite()) {
                return Err("basis frame contains a non-finite value".to_string());
            }
            let mut basis = BasisGpu {
                translation: [0.0; 4],
                rotation: [0.0; 4],
                log_scale: [0.0; 4],
            };
            let packed_value = [value[0], value[1], value[2], 0.0];
            match bank_index {
                0 => basis.translation = packed_value,
                1 => basis.rotation = packed_value,
                2 => basis.log_scale = packed_value,
                _ => unreachable!(),
            }
            packed.push(basis);
        }
    }
    Ok(())
}

fn separate_coefficient_index(
    row: usize,
    bank: usize,
    slot: usize,
    top_k: usize,
) -> Result<usize, String> {
    if bank >= 3 || slot >= top_k || top_k == 0 {
        return Err("three-bank coefficient index is out of range".to_string());
    }
    row.checked_mul(3)
        .and_then(|value| value.checked_add(bank))
        .and_then(|value| value.checked_mul(top_k))
        .and_then(|value| value.checked_add(slot))
        .ok_or_else(|| "three-bank coefficient index overflow".to_string())
}

fn pack_separate_coefficients(
    basis_ids: &[u8],
    weights: &[f32],
    row_count: usize,
    top_k: usize,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    if row_count == 0 || top_k == 0 {
        return Err("row count and top_k must be positive".to_string());
    }
    let expected_count = row_count
        .checked_mul(3)
        .and_then(|value| value.checked_mul(top_k))
        .ok_or_else(|| "three-bank coefficient count overflow".to_string())?;
    if basis_ids.len() != expected_count || weights.len() != expected_count {
        return Err("three-bank coefficients are not row-aligned".to_string());
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err("basis weights contain a non-finite value".to_string());
    }
    Ok((
        basis_ids
            .iter()
            .map(|&basis_id| u32::from(basis_id))
            .collect(),
        weights.to_vec(),
    ))
}

enum GpuBasisSource {
    Legacy(BasisBank),
    Separate(BasisBanks),
}

impl GpuBasisSource {
    fn basis_count(&self) -> Result<usize, String> {
        match self {
            Self::Legacy(basis) => Ok(basis.basis_count),
            Self::Separate(banks) => {
                let basis_count = banks.translation.basis_count;
                if basis_count == 0
                    || banks.rotation.basis_count != basis_count
                    || banks.scale.basis_count != basis_count
                {
                    return Err("three basis banks must have the same positive basis count".into());
                }
                Ok(basis_count)
            }
        }
    }

    fn coefficient_bank_count(&self) -> usize {
        match self {
            Self::Legacy(_) => 1,
            Self::Separate(_) => 3,
        }
    }

    fn source_value_count(&self) -> Result<usize, String> {
        match self {
            Self::Legacy(basis) => Ok(basis.values.len()),
            Self::Separate(banks) => banks
                .translation
                .values
                .len()
                .checked_add(banks.rotation.values.len())
                .and_then(|value| value.checked_add(banks.scale.values.len()))
                .ok_or_else(|| "source basis bank value count overflow".to_string()),
        }
    }
}

fn timeline_sample_endpoints(sample: TimelineSample) -> Result<(f32, f32), String> {
    let (from_u, to_u) = match sample {
        TimelineSample::Source { u } => (u, u),
        TimelineSample::Blend { from_u, to_u, .. } => (from_u, to_u),
    };
    if !from_u.is_finite() || !to_u.is_finite() {
        return Err("controller timeline sample must be finite".to_string());
    }
    Ok((from_u, to_u))
}

fn pack_controller_frame_into(
    basis: &GpuBasisSource,
    frame: MotionControllerFrame,
    destination: &mut [BasisGpu],
) -> Result<(), String> {
    if !frame.valid {
        destination.fill(BasisGpu::zero());
        return Ok(());
    }
    frame.gains.validate().map_err(|error| error.to_string())?;
    let (from_u, to_u) = timeline_sample_endpoints(frame.sample)?;
    let mut packed = Vec::with_capacity(destination.len());
    match basis {
        GpuBasisSource::Legacy(bank) => {
            let mut sampled = BasisFrame {
                values: vec![[0.0; 9]; bank.basis_count],
            };
            sample_basis_into(bank, from_u, &mut sampled);
            pack_basis_frame_into(&sampled, &mut packed)?;
            sample_basis_into(bank, to_u, &mut sampled);
            pack_basis_frame_into(&sampled, &mut packed)?;
        }
        GpuBasisSource::Separate(banks) => {
            let from = sample_basis_banks(banks, from_u);
            let to = sample_basis_banks(banks, to_u);
            pack_basis_banks_frame_into(&from, &mut packed)?;
            pack_basis_banks_frame_into(&to, &mut packed)?;
        }
    }
    if packed.len() != destination.len() {
        return Err("controller basis frame size changed unexpectedly".to_string());
    }
    destination.copy_from_slice(&packed);
    Ok(())
}

fn pack_controller_basis_frames(
    basis: &GpuBasisSource,
    global: MotionControllerFrame,
    locals: &[MotionControllerFrame],
) -> Result<Vec<BasisGpu>, String> {
    if global.slot != 0 {
        return Err("global controller must use slot zero".to_string());
    }
    let slot_stride = basis
        .basis_count()?
        .checked_mul(basis.coefficient_bank_count())
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "controller basis slot size overflow".to_string())?;
    let slot_count = MAX_LOCAL_CONTROLLERS + 1;
    let mut packed = vec![BasisGpu::zero(); slot_count * slot_stride];
    pack_controller_frame_into(basis, global, &mut packed[..slot_stride])?;
    let mut occupied = [false; MAX_LOCAL_CONTROLLERS + 1];
    occupied[0] = true;
    for &frame in locals {
        let slot = usize::from(frame.slot);
        if slot == 0 || slot > MAX_LOCAL_CONTROLLERS {
            return Err(format!(
                "local controller slot {} must be within 1..={MAX_LOCAL_CONTROLLERS}",
                frame.slot
            ));
        }
        if std::mem::replace(&mut occupied[slot], true) {
            return Err(format!("duplicate local controller slot {}", frame.slot));
        }
        let start = slot * slot_stride;
        pack_controller_frame_into(basis, frame, &mut packed[start..start + slot_stride])?;
    }
    Ok(packed)
}

fn pack_active_controller_slots(
    basis: &GpuBasisSource,
    global: MotionControllerFrame,
    locals: &[MotionControllerFrame],
) -> Result<Vec<(u8, Vec<BasisGpu>)>, String> {
    if global.slot != 0 || !global.valid {
        return Err("global controller slot zero must be valid".to_string());
    }
    let slot_stride = basis
        .basis_count()?
        .checked_mul(basis.coefficient_bank_count())
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "controller basis slot size overflow".to_string())?;
    let mut occupied = [false; MAX_LOCAL_CONTROLLERS + 1];
    occupied[0] = true;
    for frame in locals {
        let slot = usize::from(frame.slot);
        if slot == 0 || slot > MAX_LOCAL_CONTROLLERS {
            return Err(format!(
                "local controller slot {} must be within 1..={MAX_LOCAL_CONTROLLERS}",
                frame.slot
            ));
        }
        if std::mem::replace(&mut occupied[slot], true) {
            return Err(format!("duplicate local controller slot {}", frame.slot));
        }
    }

    let mut packed = Vec::with_capacity(1 + locals.iter().filter(|frame| frame.valid).count());
    for frame in std::iter::once(global).chain(locals.iter().copied().filter(|frame| frame.valid)) {
        let mut slot_values = vec![BasisGpu::zero(); slot_stride];
        pack_controller_frame_into(basis, frame, &mut slot_values)?;
        packed.push((frame.slot, slot_values));
    }
    Ok(packed)
}

fn plan_row_chunks(
    row_count: usize,
    coefficients_per_row: usize,
    max_storage_buffer_binding_size: u64,
) -> Result<Vec<RowChunk>, String> {
    if row_count == 0 || coefficients_per_row == 0 {
        return Err("row count and coefficient count must be positive".to_string());
    }
    let coefficient_bytes_per_row = coefficients_per_row
        .checked_mul(4)
        .ok_or_else(|| "coefficient row byte count overflow".to_string())?
        as u64;
    let max_rows = [
        max_storage_buffer_binding_size / std::mem::size_of::<CanonicalGpu>() as u64,
        max_storage_buffer_binding_size / coefficient_bytes_per_row,
        max_storage_buffer_binding_size / coefficient_bytes_per_row,
        max_storage_buffer_binding_size / 4,
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    if max_rows == 0 {
        return Err(format!(
            "max storage binding size {max_storage_buffer_binding_size} cannot hold one dynamic row"
        ));
    }
    let max_rows = usize::try_from(max_rows)
        .map_err(|_| "maximum rows per chunk exceeds usize".to_string())?;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < row_count {
        let count = (row_count - start).min(max_rows);
        chunks.push(RowChunk { start, count });
        start = start
            .checked_add(count)
            .ok_or_else(|| "chunk row offset overflow".to_string())?;
    }
    Ok(chunks)
}

pub(crate) struct GpuMotionChunk {
    layout: RowChunk,
    canonical_buffer: wgpu::Buffer,
    basis_ids_buffer: wgpu::Buffer,
    weights_buffer: wgpu::Buffer,
    transform_ids_buffer: wgpu::Buffer,
    motion_channel_masks_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuMotionChunk {
    pub(crate) fn row_start(&self) -> usize {
        self.layout.start
    }

    pub(crate) fn row_count(&self) -> usize {
        self.layout.count
    }

    pub(crate) fn canonical_buffer(&self) -> &wgpu::Buffer {
        &self.canonical_buffer
    }

    pub(crate) fn basis_ids_buffer(&self) -> &wgpu::Buffer {
        &self.basis_ids_buffer
    }

    pub(crate) fn weights_buffer(&self) -> &wgpu::Buffer {
        &self.weights_buffer
    }

    pub(crate) fn transform_ids_buffer(&self) -> &wgpu::Buffer {
        &self.transform_ids_buffer
    }

    pub(crate) fn motion_channel_masks_buffer(&self) -> &wgpu::Buffer {
        &self.motion_channel_masks_buffer
    }
}

pub struct GpuMotionRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    basis_frames_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    placement_transforms_buffer: wgpu::Buffer,
    base_texture: Texture,
    output_texture: Texture,
    #[allow(dead_code)]
    diagnostic_buffer: wgpu::Buffer,
    chunks: Vec<GpuMotionChunk>,
    basis: GpuBasisSource,
    basis_count: usize,
    coefficient_bank_count: usize,
    top_k: usize,
    texture_width: u32,
    source_duration_seconds: f32,
    summary: DynamicArchiveSummary,
    from_frame: BasisFrame,
    to_frame: BasisFrame,
    from_banks_frame: Option<BasisBanksFrame>,
    to_banks_frame: Option<BasisBanksFrame>,
    packed_basis_frames: Vec<BasisGpu>,
    global_controller_frame: MotionControllerFrame,
    motion_enabled: bool,
    preview_channel_mask: u32,
}

impl GpuMotionRuntime {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        motion: MergedMotion,
        base_texels: &[u32],
        texture_width: u32,
        texture_height: u32,
    ) -> Result<Self, String> {
        let (basis, top_k, basis_ids, weights): (
            GpuBasisSource,
            usize,
            Cow<'_, [u32]>,
            Cow<'_, [f32]>,
        ) = match &motion.data {
            MergedMotionData::Legacy {
                basis,
                top_k,
                basis_ids,
                weights,
            } => (
                GpuBasisSource::Legacy(basis.clone()),
                *top_k,
                Cow::Borrowed(basis_ids),
                Cow::Borrowed(weights),
            ),
            MergedMotionData::Separate {
                basis_banks,
                top_k,
                basis_ids,
                weights,
            } => {
                let (basis_ids, weights) = pack_separate_coefficients(
                    basis_ids,
                    weights,
                    motion.summary.total_rows,
                    *top_k,
                )?;
                (
                    GpuBasisSource::Separate(basis_banks.clone()),
                    *top_k,
                    Cow::Owned(basis_ids),
                    Cow::Owned(weights),
                )
            }
        };
        let basis_count = basis.basis_count()?;
        let coefficient_bank_count = basis.coefficient_bank_count();
        let coefficients_per_row = top_k
            .checked_mul(coefficient_bank_count)
            .ok_or_else(|| "coefficient row count overflow".to_string())?;
        if texture_width == 0 || texture_height == 0 {
            return Err("dynamic Gaussian texture dimensions must be positive".to_string());
        }
        let expected_texels = (texture_width as usize)
            .checked_mul(texture_height as usize)
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| "dynamic Gaussian texture size overflow".to_string())?;
        if base_texels.len() != expected_texels {
            return Err(format!(
                "base Gaussian texture has {} u32 values, expected {expected_texels}",
                base_texels.len()
            ));
        }
        let limits = device.limits();
        if texture_width > limits.max_texture_dimension_2d
            || texture_height > limits.max_texture_dimension_2d
        {
            return Err(format!(
                "dynamic Gaussian texture {texture_width}x{texture_height} exceeds the adapter 2D limit {}",
                limits.max_texture_dimension_2d
            ));
        }
        let expected_coefficient_count = motion
            .summary
            .total_rows
            .checked_mul(coefficients_per_row)
            .ok_or_else(|| "merged coefficient count overflow".to_string())?;
        if motion.canonical.len() != motion.summary.total_rows
            || basis_ids.len() != expected_coefficient_count
            || weights.len() != basis_ids.len()
            || motion.transform_ids.len() != motion.summary.total_rows
            || motion.motion_channel_masks.len() != motion.summary.total_rows
        {
            return Err("merged dynamic arrays are not row-aligned".to_string());
        }
        let required_pixels = motion
            .summary
            .total_rows
            .checked_mul(2)
            .ok_or_else(|| "dynamic output pixel count overflow".to_string())?;
        if required_pixels > texture_width as usize * texture_height as usize {
            return Err("dynamic output texture is too small for all Gaussian rows".to_string());
        }

        let texture_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let base_texture = Texture::from_bytes_with_usage(
            device,
            queue,
            bytemuck::cast_slice(base_texels),
            texture_width,
            texture_height,
            16,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            texture_usage,
            Some("Dynamic Gaussian Base Texture"),
        )
        .map_err(|error| error.to_string())?;
        let output_texture = Texture::from_bytes_with_usage(
            device,
            queue,
            bytemuck::cast_slice(base_texels),
            texture_width,
            texture_height,
            16,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            texture_usage | wgpu::TextureUsages::STORAGE_BINDING,
            Some("Dynamic Gaussian Output Texture"),
        )
        .map_err(|error| error.to_string())?;

        let basis_bytes = basis_count
            .checked_mul(coefficient_bank_count)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(std::mem::size_of::<BasisGpu>()))
            .ok_or_else(|| "basis frame GPU size overflow".to_string())?;
        if basis_bytes as u64 > limits.max_storage_buffer_binding_size as u64 {
            return Err(format!(
                "two sampled basis frames require {basis_bytes} bytes, exceeding the adapter storage-binding limit {}",
                limits.max_storage_buffer_binding_size
            ));
        }
        let basis_frames_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic Basis Frames"),
            size: basis_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let placements = [
            PlacementGpu {
                row0: [1.0, 0.0, 0.0, 0.0],
                row1: [0.0, 1.0, 0.0, 0.0],
                row2: [0.0, 0.0, 1.0, 0.0],
            },
            PlacementGpu {
                row0: [0.0, -1.0, 0.0, 0.0],
                row1: [1.0, 0.0, 0.0, 0.0],
                row2: [0.0, 0.0, 1.0, 0.0],
            },
        ];
        let placement_transforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Placement Transforms"),
                contents: bytemuck::cast_slice(&placements),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let diagnostic_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dynamic Motion Diagnostic State"),
            contents: bytemuck::bytes_of(&MotionStateGpu {
                position: [0.0; 4],
                log_scale: [0.0; 4],
                rotation: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dynamic Motion Bind Group Layout"),
            entries: &[
                storage_buffer_layout(0, true),
                storage_buffer_layout(1, true),
                storage_buffer_layout(2, true),
                storage_buffer_layout(3, true),
                storage_buffer_layout(4, true),
                storage_buffer_layout(5, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_buffer_layout(9, false),
                storage_buffer_layout(10, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic Motion Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Motion Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion_compute.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Motion Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let row_chunks = plan_row_chunks(
            motion.summary.total_rows,
            coefficients_per_row,
            limits.max_storage_buffer_binding_size as u64,
        )?;
        let coefficient_id_bytes = checked_bytes(
            basis_ids.len(),
            std::mem::size_of::<u32>(),
            "basis ID array",
        )?;
        let cpu_coefficient_id_bytes = checked_bytes(
            basis_ids.len(),
            if coefficient_bank_count == 3 {
                std::mem::size_of::<u8>()
            } else {
                std::mem::size_of::<u32>()
            },
            "CPU basis ID array",
        )?;
        let coefficient_weight_bytes = checked_bytes(
            weights.len(),
            std::mem::size_of::<f32>(),
            "basis weight array",
        )?;
        let transform_id_bytes = checked_bytes(
            motion.transform_ids.len(),
            std::mem::size_of::<u32>(),
            "transform ID array",
        )?;
        let source_basis_bytes = checked_bytes(
            basis.source_value_count()?,
            std::mem::size_of::<f32>(),
            "source basis bank",
        )?;
        let cpu_motion_bytes = checked_bytes(
            motion.canonical.len(),
            std::mem::size_of::<CanonicalGaussian>(),
            "canonical CPU array",
        )?
        .checked_add(cpu_coefficient_id_bytes)
        .and_then(|bytes| bytes.checked_add(coefficient_weight_bytes))
        .and_then(|bytes| bytes.checked_add(transform_id_bytes))
        .and_then(|bytes| bytes.checked_add(source_basis_bytes))
        .ok_or_else(|| "CPU motion byte estimate overflow".to_string())?;
        let texture_bytes = checked_bytes(
            base_texels.len(),
            std::mem::size_of::<u32>(),
            "Gaussian texture",
        )?;
        let dynamic_texture_bytes = checked_bytes(2, texture_bytes, "dynamic textures")?;
        let gpu_motion_bytes = checked_bytes(
            motion.summary.total_rows,
            std::mem::size_of::<CanonicalGpu>(),
            "canonical GPU array",
        )?
        .checked_add(coefficient_id_bytes)
        .and_then(|bytes| bytes.checked_add(coefficient_weight_bytes))
        .and_then(|bytes| bytes.checked_add(transform_id_bytes))
        .and_then(|bytes| bytes.checked_add(basis_bytes))
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<PlacementGpu>()))
        .and_then(|bytes| bytes.checked_add(dynamic_texture_bytes))
        .ok_or_else(|| "GPU motion byte estimate overflow".to_string())?;
        log!(
            "Dynamic archive resources: {} rows, {} chunks, estimated CPU motion {:.1} MiB, GPU motion/textures {:.1} MiB",
            motion.summary.total_rows,
            row_chunks.len(),
            cpu_motion_bytes as f64 / (1024.0 * 1024.0),
            gpu_motion_bytes as f64 / (1024.0 * 1024.0),
        );
        let canonical_gpu = motion
            .canonical
            .iter()
            .copied()
            .map(CanonicalGpu::from)
            .collect::<Vec<_>>();
        let transform_ids = motion.transform_ids.iter().copied().collect::<Vec<_>>();
        let motion_channel_masks = motion.motion_channel_masks.clone();
        let mut chunks = Vec::with_capacity(row_chunks.len());
        for layout in row_chunks {
            let row_end = layout.start + layout.count;
            let (coefficient_start, coefficient_end) = if coefficient_bank_count == 3 {
                (
                    separate_coefficient_index(layout.start, 0, 0, top_k)?,
                    separate_coefficient_index(row_end, 0, 0, top_k)?,
                )
            } else {
                (
                    layout
                        .start
                        .checked_mul(top_k)
                        .ok_or_else(|| "coefficient chunk start overflow".to_string())?,
                    row_end
                        .checked_mul(top_k)
                        .ok_or_else(|| "coefficient chunk end overflow".to_string())?,
                )
            };
            let canonical_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Canonical Chunk"),
                contents: bytemuck::cast_slice(&canonical_gpu[layout.start..row_end]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let basis_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Basis IDs Chunk"),
                contents: bytemuck::cast_slice(&basis_ids[coefficient_start..coefficient_end]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Basis Weights Chunk"),
                contents: bytemuck::cast_slice(&weights[coefficient_start..coefficient_end]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let transform_ids_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dynamic Transform IDs Chunk"),
                    contents: bytemuck::cast_slice(&transform_ids[layout.start..row_end]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let motion_channel_masks_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dynamic Motion Channel Masks Chunk"),
                    contents: bytemuck::cast_slice(&motion_channel_masks[layout.start..row_end]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let params = MotionParamsGpu {
                row_start: layout.start as u32,
                row_count: layout.count as u32,
                top_k: top_k as u32,
                basis_count: basis_count as u32,
                texture_width,
                motion_enabled: 0,
                blend_alpha: 0.0,
                diagnostic_row: u32::MAX,
                preview_channel_mask: ALL_MOTION_MASK,
                separate_banks: u32::from(coefficient_bank_count == 3),
                _padding: [0; 2],
                translation_master_gain: [1.0; 4],
                rotation_scale_gain: [1.0, 1.0, 0.0, 0.0],
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dynamic Motion Parameters"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Dynamic Motion Chunk Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    buffer_entry(0, &canonical_buffer),
                    buffer_entry(1, &basis_ids_buffer),
                    buffer_entry(2, &weights_buffer),
                    buffer_entry(3, &transform_ids_buffer),
                    buffer_entry(4, &basis_frames_buffer),
                    buffer_entry(5, &placement_transforms_buffer),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&base_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&output_texture.view),
                    },
                    buffer_entry(8, &params_buffer),
                    buffer_entry(9, &diagnostic_buffer),
                    buffer_entry(10, &motion_channel_masks_buffer),
                ],
            });
            chunks.push(GpuMotionChunk {
                layout,
                canonical_buffer,
                basis_ids_buffer,
                weights_buffer,
                transform_ids_buffer,
                motion_channel_masks_buffer,
                params_buffer,
                bind_group,
            });
        }

        Ok(Self {
            compute_pipeline,
            basis_frames_buffer,
            placement_transforms_buffer,
            base_texture,
            output_texture,
            diagnostic_buffer,
            chunks,
            basis,
            basis_count,
            coefficient_bank_count,
            top_k,
            texture_width,
            source_duration_seconds: motion.source_duration_seconds,
            summary: motion.summary,
            from_frame: BasisFrame {
                values: vec![[0.0; 9]; basis_count],
            },
            to_frame: BasisFrame {
                values: vec![[0.0; 9]; basis_count],
            },
            from_banks_frame: None,
            to_banks_frame: None,
            packed_basis_frames: Vec::with_capacity(basis_count * coefficient_bank_count * 2),
            global_controller_frame: MotionControllerFrame {
                slot: 0,
                sample: TimelineSample::Source { u: 0.0 },
                gains: MotionChannelGains::default(),
                valid: true,
            },
            motion_enabled: false,
            preview_channel_mask: ALL_MOTION_MASK,
        })
    }

    pub fn dispatch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
    ) -> Result<(), String> {
        self.dispatch_masked(queue, encoder, sample, motion_enabled, ALL_MOTION_MASK)
    }

    pub fn dispatch_masked(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
    ) -> Result<(), String> {
        self.dispatch_with_diagnostic(
            queue,
            encoder,
            sample,
            motion_enabled,
            preview_channel_mask,
            None,
        )
    }

    pub fn dispatch_masked_profiled(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        gains: MotionChannelGains,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<MotionDispatchWork, String> {
        self.dispatch_with_options(
            queue,
            encoder,
            sample,
            motion_enabled,
            preview_channel_mask,
            gains,
            None,
            timestamp_writes,
        )
    }

    fn dispatch_with_diagnostic(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        diagnostic_row: Option<u32>,
    ) -> Result<(), String> {
        self.dispatch_with_options(
            queue,
            encoder,
            sample,
            motion_enabled,
            preview_channel_mask,
            MotionChannelGains::default(),
            diagnostic_row,
            None,
        )
        .map(|_| ())
    }

    #[cfg(test)]
    fn dispatch_with_diagnostic_and_gains(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        gains: MotionChannelGains,
        diagnostic_row: Option<u32>,
    ) -> Result<(), String> {
        self.dispatch_with_options(
            queue,
            encoder,
            sample,
            motion_enabled,
            preview_channel_mask,
            gains,
            diagnostic_row,
            None,
        )
        .map(|_| ())
    }

    fn dispatch_with_options(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        gains: MotionChannelGains,
        diagnostic_row: Option<u32>,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<MotionDispatchWork, String> {
        let blend = sample.is_blend();
        let (from_u, to_u, blend_alpha) = match sample {
            TimelineSample::Source { u } => (u, u, 0.0),
            TimelineSample::Blend {
                from_u,
                to_u,
                alpha,
            } => (from_u, to_u, alpha),
        };
        if !from_u.is_finite() || !to_u.is_finite() || !blend_alpha.is_finite() {
            return Err("timeline sample must be finite".to_string());
        }
        if preview_channel_mask & !ALL_MOTION_MASK != 0 {
            return Err("preview channel mask contains reserved bits".to_string());
        }
        gains.validate().map_err(|error| error.to_string())?;
        self.packed_basis_frames.clear();
        match &self.basis {
            GpuBasisSource::Legacy(basis) => {
                sample_basis_into(basis, from_u, &mut self.from_frame);
                sample_basis_into(basis, to_u, &mut self.to_frame);
                pack_basis_frame_into(&self.from_frame, &mut self.packed_basis_frames)?;
                pack_basis_frame_into(&self.to_frame, &mut self.packed_basis_frames)?;
            }
            GpuBasisSource::Separate(banks) => {
                self.from_banks_frame = Some(sample_basis_banks(banks, from_u));
                self.to_banks_frame = Some(sample_basis_banks(banks, to_u));
                pack_basis_banks_frame_into(
                    self.from_banks_frame.as_ref().unwrap(),
                    &mut self.packed_basis_frames,
                )?;
                pack_basis_banks_frame_into(
                    self.to_banks_frame.as_ref().unwrap(),
                    &mut self.packed_basis_frames,
                )?;
            }
        }
        if self.packed_basis_frames.len() != self.basis_count * self.coefficient_bank_count * 2 {
            return Err("sampled basis frame count changed unexpectedly".to_string());
        }
        queue.write_buffer(
            &self.basis_frames_buffer,
            0,
            bytemuck::cast_slice(&self.packed_basis_frames),
        );
        for chunk in &self.chunks {
            let params = MotionParamsGpu {
                row_start: chunk.layout.start as u32,
                row_count: chunk.layout.count as u32,
                top_k: self.top_k as u32,
                basis_count: self.basis_count as u32,
                texture_width: self.texture_width,
                motion_enabled: u32::from(motion_enabled),
                blend_alpha: blend_alpha.clamp(0.0, 1.0),
                diagnostic_row: diagnostic_row.unwrap_or(u32::MAX),
                preview_channel_mask,
                separate_banks: u32::from(self.coefficient_bank_count == 3),
                _padding: [0; 2],
                translation_master_gain: [
                    gains.translation[0],
                    gains.translation[1],
                    gains.translation[2],
                    gains.master,
                ],
                rotation_scale_gain: [gains.rotation, gains.scale, 0.0, 0.0],
            };
            queue.write_buffer(&chunk.params_buffer, 0, bytemuck::bytes_of(&params));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Dynamic Motion Update"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.compute_pipeline);
        for chunk in &self.chunks {
            pass.set_bind_group(0, &chunk.bind_group, &[]);
            pass.dispatch_workgroups((chunk.layout.count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        drop(pass);
        self.global_controller_frame = MotionControllerFrame {
            slot: 0,
            sample,
            gains,
            valid: true,
        };
        self.motion_enabled = motion_enabled;
        self.preview_channel_mask = preview_channel_mask;
        let uploaded_bytes = std::mem::size_of_val(self.packed_basis_frames.as_slice())
            .saturating_add(
                self.chunks
                    .len()
                    .saturating_mul(std::mem::size_of::<MotionParamsGpu>()),
            );
        Ok(MotionDispatchWork {
            rows: self.summary.total_rows,
            uploaded_bytes: uploaded_bytes as u64,
            blend,
        })
    }

    pub fn output_texture(&self) -> &Texture {
        &self.output_texture
    }

    pub fn base_texture(&self) -> &Texture {
        &self.base_texture
    }

    #[cfg(test)]
    fn diagnostic_buffer(&self) -> &wgpu::Buffer {
        &self.diagnostic_buffer
    }

    pub fn source_duration_seconds(&self) -> f32 {
        self.source_duration_seconds
    }

    pub fn summary(&self) -> &DynamicArchiveSummary {
        &self.summary
    }

    pub(crate) fn chunks(&self) -> &[GpuMotionChunk] {
        &self.chunks
    }

    pub(crate) fn basis_count(&self) -> usize {
        self.basis_count
    }

    pub(crate) fn coefficient_bank_count(&self) -> usize {
        self.coefficient_bank_count
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    pub(crate) fn texture_width(&self) -> u32 {
        self.texture_width
    }

    pub(crate) fn placement_transforms_buffer(&self) -> &wgpu::Buffer {
        &self.placement_transforms_buffer
    }

    pub(crate) fn global_controller_frame(&self) -> MotionControllerFrame {
        self.global_controller_frame
    }

    pub(crate) fn motion_enabled(&self) -> bool {
        self.motion_enabled
    }

    pub(crate) fn preview_channel_mask(&self) -> u32 {
        self.preview_channel_mask
    }

    pub(crate) fn pack_controller_basis_frames(
        &self,
        global: MotionControllerFrame,
        locals: &[MotionControllerFrame],
        output: &mut Vec<BasisGpu>,
    ) -> Result<(), String> {
        *output = pack_controller_basis_frames(&self.basis, global, locals)?;
        Ok(())
    }

    pub(crate) fn pack_active_controller_slots(
        &self,
        global: MotionControllerFrame,
        locals: &[MotionControllerFrame],
    ) -> Result<Vec<(u8, Vec<BasisGpu>)>, String> {
        pack_active_controller_slots(&self.basis, global, locals)
    }

    pub(crate) fn controller_slot_stride_bytes(&self) -> Result<u64, String> {
        self.basis_count
            .checked_mul(self.coefficient_bank_count)
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(std::mem::size_of::<BasisGpu>()))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| "controller basis slot byte size overflow".to_string())
    }

    pub(crate) fn locate_row(&self, global_row: usize) -> Option<(usize, u32)> {
        self.chunks.iter().enumerate().find_map(|(index, chunk)| {
            let end = chunk.layout.start.checked_add(chunk.layout.count)?;
            (global_row >= chunk.layout.start && global_row < end)
                .then_some((index, (global_row - chunk.layout.start) as u32))
        })
    }
}

fn storage_buffer_layout(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_archive::{BasisBank, BasisBanks};
    use crate::motion::{
        ALL_MOTION_MASK, BasisBanksFrame, BasisFrame, CanonicalGaussian, MotionState,
        ROTATION_MASK, TRANSLATION_MASK, blend_states, evaluate_row_banks_masked,
        evaluate_row_masked, sample_basis, sample_basis_banks,
    };
    use crate::motion_behavior::MotionChannelGains;
    use crate::motion_controller_palette::{MAX_LOCAL_CONTROLLERS, MotionControllerFrame};
    use crate::scene_archive::DynamicArchiveSummary;

    #[test]
    fn controller_basis_frames_pack_global_then_fixed_local_slots() {
        let mut values = Vec::with_capacity(75 * 9);
        for sample in 0..75 {
            values.extend_from_slice(&[sample as f32; 9]);
        }
        let basis = GpuBasisSource::Legacy(BasisBank {
            basis_count: 1,
            values,
        });
        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Source { u: 0.0 },
            gains: MotionChannelGains::default(),
            valid: true,
        };
        let locals = [
            MotionControllerFrame {
                slot: 2,
                sample: TimelineSample::Source { u: 1.0 },
                gains: MotionChannelGains::default(),
                valid: true,
            },
            MotionControllerFrame {
                slot: 1,
                sample: TimelineSample::Blend {
                    from_u: 0.25,
                    to_u: 0.75,
                    alpha: 0.5,
                },
                gains: MotionChannelGains::default(),
                valid: true,
            },
        ];

        let packed = pack_controller_basis_frames(&basis, global, &locals).unwrap();

        assert_eq!(packed.len(), (MAX_LOCAL_CONTROLLERS + 1) * 2);
        assert_eq!(packed[0].translation[0], 0.0);
        assert_eq!(packed[1].translation[0], 0.0);
        assert!((packed[2].translation[0] - 18.5).abs() < 1.0e-5);
        assert!((packed[3].translation[0] - 55.5).abs() < 1.0e-5);
        assert_eq!(packed[4].translation[0], 74.0);
        assert_eq!(packed[5].translation[0], 74.0);
        assert!(packed[6..].iter().all(|basis| *basis == BasisGpu::zero()));
    }

    #[test]
    fn controller_basis_frames_reject_duplicate_or_reserved_local_slots() {
        let basis = GpuBasisSource::Legacy(BasisBank {
            basis_count: 1,
            values: vec![0.0; 75 * 9],
        });
        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Source { u: 0.0 },
            gains: MotionChannelGains::default(),
            valid: true,
        };
        let duplicate = MotionControllerFrame { slot: 1, ..global };
        let error =
            pack_controller_basis_frames(&basis, global, &[duplicate, duplicate]).unwrap_err();
        assert!(error.contains("duplicate"), "{error}");

        let error = pack_controller_basis_frames(&basis, global, &[global]).unwrap_err();
        assert!(error.contains("local controller slot"), "{error}");
    }

    #[test]
    fn controller_slot_packing_is_bounded_to_each_active_fixed_address_range() {
        let make_bank = || BasisBank {
            basis_count: 64,
            values: vec![0.0; 64 * 75 * 3],
        };
        let basis = GpuBasisSource::Separate(BasisBanks {
            translation: make_bank(),
            rotation: make_bank(),
            scale: make_bank(),
        });
        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Source { u: 0.0 },
            gains: MotionChannelGains::default(),
            valid: true,
        };
        let locals = [
            MotionControllerFrame { slot: 1, ..global },
            MotionControllerFrame { slot: 4, ..global },
        ];

        let packed = pack_active_controller_slots(&basis, global, &locals).unwrap();
        let slot_stride = 2 * 3 * 64;

        assert_eq!(packed.len(), 3);
        assert_eq!(
            packed
                .iter()
                .map(|(_, values)| values.len())
                .collect::<Vec<_>>(),
            vec![slot_stride; 3]
        );
        assert_eq!(
            packed
                .iter()
                .map(|(_, values)| std::mem::size_of_val(values.as_slice()))
                .sum::<usize>(),
            3 * slot_stride * std::mem::size_of::<BasisGpu>()
        );
        assert_eq!(
            packed.iter().map(|(slot, _)| *slot).collect::<Vec<_>>(),
            vec![0, 1, 4]
        );
    }

    #[test]
    fn gpu_structs_have_literal_storage_alignment() {
        assert_eq!(std::mem::size_of::<CanonicalGpu>(), 48);
        assert_eq!(std::mem::size_of::<BasisGpu>(), 48);
        assert_eq!(std::mem::size_of::<PlacementGpu>(), 48);
        assert_eq!(std::mem::size_of::<MotionParamsGpu>(), 80);
        assert_eq!(std::mem::size_of::<MotionStateGpu>(), 48);
    }

    #[test]
    fn numerical_safety_preserves_rotation_while_falling_back_scale_then_covariance() {
        assert!(covariance_packable([
            HALF_PACKING_LIMIT / 4.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]));
        assert!(!covariance_packable([
            HALF_PACKING_LIMIT / 4.0 + 1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]));

        let canonical = CanonicalGaussian {
            position: [1.0, 2.0, 3.0],
            log_scale: [-0.5, -0.25, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        let dynamic_rotation = [0.9238795, 0.3826834, 0.0, 0.0];
        let scale_fallback = safe_dynamic_output(
            MotionState {
                position: [f32::INFINITY, 4.0, 5.0],
                log_scale: [100.0, 0.0, 0.0],
                rotation: dynamic_rotation,
            },
            canonical,
        );
        assert_eq!(scale_fallback.state.position, canonical.position);
        assert_eq!(scale_fallback.state.log_scale, canonical.log_scale);
        assert_eq!(scale_fallback.state.rotation, dynamic_rotation);
        assert!(scale_fallback.covariance_is_dynamic);

        let canonical_covariance = safe_dynamic_output(
            MotionState {
                position: canonical.position,
                log_scale: [100.0, 0.0, 0.0],
                rotation: [f32::NAN, 0.0, 0.0, 0.0],
            },
            canonical,
        );
        assert!(!canonical_covariance.covariance_is_dynamic);
    }

    #[test]
    fn basis_frames_pack_as_translation_rotation_and_scale_vec4s() {
        let frame = BasisFrame {
            values: vec![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]],
        };

        let packed = pack_basis_frame(&frame).unwrap();

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].translation, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(packed[0].rotation, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(packed[0].log_scale, [7.0, 8.0, 9.0, 0.0]);
    }

    #[test]
    fn separate_staging_preserves_row_bank_slot_layout_and_widens_ids() {
        let ids = [10_u8, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61];
        let weights = [
            0.10_f32, 0.11, 0.20, 0.21, 0.30, 0.31, 0.40, 0.41, 0.50, 0.51, 0.60, 0.61,
        ];

        let (packed_ids, packed_weights) =
            pack_separate_coefficients(&ids, &weights, 2, 2).unwrap();

        assert_eq!(
            packed_ids,
            vec![10_u32, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61]
        );
        assert_eq!(packed_weights, weights);
        assert_eq!(separate_coefficient_index(0, 0, 0, 2).unwrap(), 0);
        assert_eq!(separate_coefficient_index(0, 2, 1, 2).unwrap(), 5);
        assert_eq!(separate_coefficient_index(1, 0, 0, 2).unwrap(), 6);
        assert_eq!(separate_coefficient_index(1, 2, 1, 2).unwrap(), 11);
    }

    #[test]
    fn separate_basis_frames_pack_bank_major_in_one_buffer() {
        let frame = BasisBanksFrame {
            translation: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            rotation: vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            scale: vec![[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        };

        let packed = pack_basis_banks_frame(&frame).unwrap();

        assert_eq!(packed.len(), 6);
        assert_eq!(packed[0].translation, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(packed[1].translation, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(packed[2].rotation, [7.0, 8.0, 9.0, 0.0]);
        assert_eq!(packed[3].rotation, [10.0, 11.0, 12.0, 0.0]);
        assert_eq!(packed[4].log_scale, [13.0, 14.0, 15.0, 0.0]);
        assert_eq!(packed[5].log_scale, [16.0, 17.0, 18.0, 0.0]);
    }

    #[test]
    fn storage_limit_chunks_rows_without_gaps_or_overlap() {
        let chunks = plan_row_chunks(5, 4, 100).unwrap();
        assert_eq!(
            chunks,
            vec![
                RowChunk { start: 0, count: 2 },
                RowChunk { start: 2, count: 2 },
                RowChunk { start: 4, count: 1 }
            ]
        );
        assert!(plan_row_chunks(1, 1, 47).is_err());
        assert!(plan_row_chunks(1, 0, 1024).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_three_bank_reconstruction_matches_cpu_for_source_blend_and_masks() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("three-bank GPU parity test skipped: no native adapter ({error})");
                    return;
                }
            };
        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Uint);
        if !format_features
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            eprintln!(
                "three-bank GPU parity test skipped: adapter lacks rgba32uint storage textures"
            );
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("three-bank motion parity device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        let make_bank = |bank_offset: f32| {
            let mut values = Vec::with_capacity(2 * 75 * 3);
            for basis in 0..2 {
                for sample in 0..75 {
                    let u = sample as f32 / 74.0;
                    let basis_scale = basis as f32 + 1.0;
                    values.extend_from_slice(&[
                        bank_offset + basis_scale * u * u,
                        -0.5 * bank_offset + basis_scale * u,
                        0.25 * bank_offset - basis_scale * u * u * u,
                    ]);
                }
            }
            BasisBank {
                basis_count: 2,
                values,
            }
        };
        let banks = BasisBanks {
            translation: make_bank(0.10),
            rotation: make_bank(0.03),
            scale: make_bank(-0.04),
        };
        let canonical = vec![
            CanonicalGaussian {
                position: [1.0, 2.0, 3.0],
                log_scale: [-0.4, 0.1, 0.5],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [-1.0, 0.5, 2.0],
                log_scale: [0.2, -0.3, 0.4],
                rotation: [0.9238795, 0.0, 0.3826834, 0.0],
            },
            CanonicalGaussian {
                position: [0.25, -1.0, 0.75],
                log_scale: [-0.2, 0.35, 0.05],
                rotation: [0.9659258, 0.258819, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [0.0, 1.0, -2.0],
                log_scale: [-0.1, -0.2, -0.3],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let row_coefficients = [
            0_u8, 1, 1, 0, 0, 1, // translation, rotation, scale for K=2
        ];
        let row_weights = [0.75_f32, -0.25, 0.5, 0.125, -0.2, 0.4];
        let basis_ids = row_coefficients.repeat(4);
        let weights = row_weights.repeat(4);
        let masks = vec![1_u32, 3, 5, 7];
        let motion = MergedMotion {
            data: MergedMotionData::Separate {
                basis_banks: banks.clone(),
                top_k: 2,
                basis_ids: basis_ids.clone(),
                weights: weights.clone(),
            },
            canonical: canonical.clone(),
            transform_ids: vec![0, 1, 0, 1],
            motion_channel_masks: masks.clone(),
            source_duration_seconds: 1.0,
            summary: DynamicArchiveSummary {
                schema_version: 3,
                tile_count: 1,
                lod_count: 1,
                basis_count: 2,
                top_k: 2,
                total_rows: 4,
                backend: "three-bank-gpu-parity".into(),
            },
            member_offsets: vec![vec![0]],
            member_counts: vec![vec![4]],
        };
        let mut runtime =
            GpuMotionRuntime::new(&device, &queue, motion, &[0_u32; 32], 8, 1).unwrap();

        let source_sample = TimelineSample::Source { u: 0.37 };
        let source_frame = sample_basis_banks(&banks, 0.37);
        let blend_sample = TimelineSample::Blend {
            from_u: 0.83,
            to_u: 0.07,
            alpha: 0.35,
        };
        let from_frame = sample_basis_banks(&banks, 0.83);
        let to_frame = sample_basis_banks(&banks, 0.07);
        for row in 0..4 {
            let coefficient_start = row * 6;
            let coefficient_end = coefficient_start + 6;
            let ids = &basis_ids[coefficient_start..coefficient_end];
            let row_weights = &weights[coefficient_start..coefficient_end];
            let expected_source = evaluate_row_banks_masked(
                canonical[row],
                &source_frame,
                ids,
                row_weights,
                2,
                (row % 2) as u32,
                masks[row],
                ALL_MOTION_MASK,
            )
            .unwrap();
            let actual_source =
                read_diagnostic_state(&mut runtime, &device, &queue, source_sample, row as u32);
            assert_state_close(
                &format!("three-bank source mask {}", masks[row]),
                actual_source,
                expected_source,
            );

            let from = evaluate_row_banks_masked(
                canonical[row],
                &from_frame,
                ids,
                row_weights,
                2,
                (row % 2) as u32,
                masks[row],
                ALL_MOTION_MASK,
            )
            .unwrap();
            let to = evaluate_row_banks_masked(
                canonical[row],
                &to_frame,
                ids,
                row_weights,
                2,
                (row % 2) as u32,
                masks[row],
                ALL_MOTION_MASK,
            )
            .unwrap();
            let expected_blend = blend_states(from, to, 0.35);
            let actual_blend =
                read_diagnostic_state(&mut runtime, &device, &queue, blend_sample, row as u32);
            assert_state_close(
                &format!("three-bank blend mask {}", masks[row]),
                actual_blend,
                expected_blend,
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_reconstruction_matches_cpu_for_zero_rigid_mixed_and_blended_motion() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("GPU parity test skipped: no native adapter ({error})");
                    return;
                }
            };
        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Uint);
        if !format_features
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            eprintln!("GPU parity test skipped: adapter lacks rgba32uint storage textures");
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("motion parity device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        let mut basis_values = Vec::with_capacity(7 * 75 * 9);
        for basis in 0..7 {
            for sample in 0..75 {
                let u = sample as f32 / 74.0;
                let value = match basis {
                    0 => [0.0; 9],
                    1 => [0.5, -0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    2 => [
                        0.5 * u,
                        -0.25 * u,
                        u,
                        0.2 * u,
                        -0.1 * u,
                        0.3 * u,
                        0.1 * u,
                        -0.2 * u,
                        0.05 * u,
                    ],
                    3 => [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 100.0, 0.0, 0.0],
                    4 => [0.0, 0.0, 0.0, 1.0e20, 0.0, 0.0, 0.0, 0.0, 0.0],
                    5 => [1.0e37, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    6 => [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, -1.0e37, 0.0, 0.0],
                    _ => unreachable!(),
                };
                basis_values.extend_from_slice(&value);
            }
        }
        let canonical = vec![
            CanonicalGaussian {
                position: [1.0, 2.0, 3.0],
                log_scale: [-0.4, 0.1, 0.5],
                rotation: [0.9238795, 0.0, 0.3826834, 0.0],
            },
            CanonicalGaussian {
                position: [-1.0, 0.5, 2.0],
                log_scale: [0.2, -0.3, 0.4],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [0.25, -1.0, 0.75],
                log_scale: [-0.2, 0.35, 0.05],
                rotation: [0.9659258, 0.258819, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [0.25, -1.0, 0.75],
                log_scale: [-0.2, 0.35, 0.05],
                rotation: [0.9659258, 0.258819, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [0.0, 0.0, 0.0],
                log_scale: [-0.2, -0.1, 0.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [0.0, 0.0, 0.0],
                log_scale: [-0.2, -0.1, 0.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [3.35e38, 2.0, 3.0],
                log_scale: [-0.2, -0.1, 0.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
            CanonicalGaussian {
                position: [1.0, 2.0, 3.0],
                log_scale: [-0.2, -0.1, 0.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let bank = BasisBank {
            basis_count: 7,
            values: basis_values,
        };
        let motion = MergedMotion {
            data: MergedMotionData::Legacy {
                basis: bank.clone(),
                top_k: 1,
                basis_ids: vec![0, 1, 2, 2, 3, 4, 5, 6],
                weights: vec![1.0, 1.0, 0.75, 0.75, 1.0, 1.0, 1.0, 40.0],
            },
            canonical: canonical.clone(),
            transform_ids: vec![0, 0, 1, 1, 0, 0, 0, 0],
            motion_channel_masks: vec![7, 7, 7, 3, 7, 7, 7, 7],
            source_duration_seconds: 1.0,
            summary: DynamicArchiveSummary {
                schema_version: 1,
                tile_count: 1,
                lod_count: 7,
                basis_count: 7,
                top_k: 1,
                total_rows: 8,
                backend: "gpu-parity".into(),
            },
            member_offsets: vec![
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![6],
                vec![7],
            ],
            member_counts: vec![vec![1]; 8],
        };
        let mut base_texels = vec![0_u32; 16 * 1 * 4];
        base_texels[7] = 0x4433_2211;
        base_texels[15] = 0x8877_6655;
        base_texels[23] = 0xccbb_aa99;
        base_texels[31] = 0x1020_3040;
        base_texels[39] = 0x5060_7080;
        base_texels[44..47].copy_from_slice(&[0x1111_2222, 0x3333_4444, 0x5555_6666]);
        base_texels[47] = 0x90a0_b0c0;
        base_texels[55] = 0xd0e0_f000;
        base_texels[60..63].copy_from_slice(&[0x1234_5678, 0x9abc_def0, 0x0bad_cafe]);
        let mut runtime =
            GpuMotionRuntime::new(&device, &queue, motion, &base_texels, 16, 1).unwrap();

        let source_sample = TimelineSample::Source { u: 0.37 };
        let source_frame = sample_basis(&bank, 0.37);
        for (row, (basis_id, weight, transform_id, label)) in [
            (0_u32, 1.0_f32, 0_u32, "zero"),
            (1, 1.0, 0, "rigid"),
            (2, 0.75, 1, "mixed"),
            (2, 0.75, 1, "source scale fallback"),
        ]
        .into_iter()
        .enumerate()
        {
            let row_mask = if row == 3 { 3 } else { 7 };
            let expected = evaluate_row_masked(
                canonical[row],
                &source_frame,
                &[basis_id],
                &[weight],
                transform_id,
                row_mask,
                ALL_MOTION_MASK,
            )
            .unwrap();
            let actual =
                read_diagnostic_state(&mut runtime, &device, &queue, source_sample, row as u32);
            assert_state_close(label, actual, expected);
        }

        let blend_sample = TimelineSample::Blend {
            from_u: 0.83,
            to_u: 0.07,
            alpha: 0.35,
        };
        let from = evaluate_row_masked(
            canonical[3],
            &sample_basis(&bank, 0.83),
            &[2],
            &[0.75],
            1,
            3,
            ALL_MOTION_MASK,
        )
        .unwrap();
        let to = evaluate_row_masked(
            canonical[2],
            &sample_basis(&bank, 0.07),
            &[2],
            &[0.75],
            1,
            3,
            ALL_MOTION_MASK,
        )
        .unwrap();
        let expected_blend = blend_states(from, to, 0.35);
        let actual_blend = read_diagnostic_state(&mut runtime, &device, &queue, blend_sample, 3);
        assert_state_close("blended loop policy", actual_blend, expected_blend);

        let from_scale_disabled = evaluate_row_masked(
            canonical[2],
            &sample_basis(&bank, 0.83),
            &[2],
            &[0.75],
            1,
            ALL_MOTION_MASK,
            TRANSLATION_MASK | ROTATION_MASK,
        )
        .unwrap();
        let to_scale_disabled = evaluate_row_masked(
            canonical[2],
            &sample_basis(&bank, 0.07),
            &[2],
            &[0.75],
            1,
            ALL_MOTION_MASK,
            TRANSLATION_MASK | ROTATION_MASK,
        )
        .unwrap();
        let expected_scale_disabled_blend =
            blend_states(from_scale_disabled, to_scale_disabled, 0.35);
        let actual_scale_disabled_blend = read_diagnostic_state_masked(
            &mut runtime,
            &device,
            &queue,
            blend_sample,
            2,
            TRANSLATION_MASK | ROTATION_MASK,
        );
        assert_state_close(
            "blended loop policy with scale disabled",
            actual_scale_disabled_blend,
            expected_scale_disabled_blend,
        );

        for preview_mask in 0..=ALL_MOTION_MASK {
            let expected = evaluate_row_masked(
                canonical[2],
                &source_frame,
                &[2],
                &[0.75],
                1,
                ALL_MOTION_MASK,
                preview_mask,
            )
            .unwrap();
            let actual = read_diagnostic_state_masked(
                &mut runtime,
                &device,
                &queue,
                source_sample,
                2,
                preview_mask,
            );
            assert_state_close(&format!("preview mask {preview_mask}"), actual, expected);
        }

        let gains = MotionChannelGains {
            translation: [2.0, 0.5, 0.0],
            rotation: 0.0,
            scale: 0.0,
            master: 0.5,
        };
        let actual_gained = read_diagnostic_state_masked_with_gains(
            &mut runtime,
            &device,
            &queue,
            source_sample,
            1,
            ALL_MOTION_MASK,
            gains,
        );
        assert_state_close(
            "artist channel gains",
            actual_gained,
            MotionState {
                position: [
                    canonical[1].position[0] + 0.5,
                    canonical[1].position[1] - 0.0625,
                    canonical[1].position[2],
                ],
                log_scale: canonical[1].log_scale,
                rotation: canonical[1].rotation,
            },
        );

        let actual_zero_strength = read_diagnostic_state_masked_with_gains(
            &mut runtime,
            &device,
            &queue,
            source_sample,
            2,
            ALL_MOTION_MASK,
            MotionChannelGains {
                master: 0.0,
                ..MotionChannelGains::default()
            },
        );
        assert_state_close(
            "zero master strength",
            actual_zero_strength,
            MotionState {
                position: canonical[2].position,
                log_scale: canonical[2].log_scale,
                rotation: canonical[2].rotation,
            },
        );

        let mut rotation_only_frame = source_frame.clone();
        for value in &mut rotation_only_frame.values {
            value[0..3].fill(0.0);
            value[3..6].iter_mut().for_each(|channel| *channel *= 1.5);
            value[6..9].fill(0.0);
        }
        let expected_rotation_only = evaluate_row_masked(
            canonical[2],
            &rotation_only_frame,
            &[2],
            &[0.75],
            1,
            ALL_MOTION_MASK,
            ALL_MOTION_MASK,
        )
        .unwrap();
        let actual_rotation_only = read_diagnostic_state_masked_with_gains(
            &mut runtime,
            &device,
            &queue,
            source_sample,
            2,
            ALL_MOTION_MASK,
            MotionChannelGains {
                translation: [0.0; 3],
                rotation: 1.5,
                scale: 0.0,
                master: 1.0,
            },
        );
        assert_state_close(
            "rotation-only gain",
            actual_rotation_only,
            expected_rotation_only,
        );

        let mut scale_only_frame = source_frame.clone();
        for value in &mut scale_only_frame.values {
            value[0..6].fill(0.0);
            value[6..9].iter_mut().for_each(|channel| *channel *= 1.75);
        }
        let expected_scale_only = evaluate_row_masked(
            canonical[2],
            &scale_only_frame,
            &[2],
            &[0.75],
            1,
            ALL_MOTION_MASK,
            ALL_MOTION_MASK,
        )
        .unwrap();
        let actual_scale_only = read_diagnostic_state_masked_with_gains(
            &mut runtime,
            &device,
            &queue,
            source_sample,
            2,
            ALL_MOTION_MASK,
            MotionChannelGains {
                translation: [0.0; 3],
                rotation: 0.0,
                scale: 1.75,
                master: 1.0,
            },
        );
        assert_state_close("scale-only gain", actual_scale_only, expected_scale_only);

        let output = read_output_texture(&mut runtime, &device, &queue, source_sample, 16);
        for row in 0..4 {
            let basis_id = [0, 1, 2, 2][row];
            let weight = if row >= 2 { 0.75 } else { 1.0 };
            let transform_id = if row >= 2 { 1 } else { 0 };
            let expected = evaluate_row_masked(
                canonical[row],
                &source_frame,
                &[basis_id],
                &[weight],
                transform_id,
                if row == 3 { 3 } else { 7 },
                ALL_MOTION_MASK,
            )
            .unwrap();
            let pixel = row * 8;
            let actual_covariance = unpack_covariance(&output[pixel + 4..pixel + 7]);
            let expected_covariance = expected.covariance();
            for component in 0..6 {
                assert!(
                    (actual_covariance[component] - expected_covariance[component]).abs() < 0.02,
                    "packed row {row} covariance {component}: {} != {}",
                    actual_covariance[component],
                    expected_covariance[component]
                );
            }
        }
        assert_eq!(output[7], 0x4433_2211);
        assert_eq!(output[15], 0x8877_6655);
        assert_eq!(output[23], 0xccbb_aa99);
        assert_eq!(output[31], 0x1020_3040);

        let pathological_scale = safe_dynamic_output(
            evaluate_row_masked(
                canonical[4],
                &source_frame,
                &[3],
                &[1.0],
                0,
                ALL_MOTION_MASK,
                ALL_MOTION_MASK,
            )
            .unwrap(),
            canonical[4],
        );
        assert!(pathological_scale.covariance_is_dynamic);
        let actual_pathological = unpack_covariance(&output[36..39]);
        for (actual, expected) in actual_pathological
            .into_iter()
            .zip(pathological_scale.state.covariance())
        {
            assert!((actual - expected).abs() < 0.02);
        }
        assert_eq!(&output[44..47], &[0x1111_2222, 0x3333_4444, 0x5555_6666]);
        assert_eq!(f32::from_bits(output[48]), canonical[6].position[0]);
        assert_eq!(f32::from_bits(output[49]), canonical[6].position[1]);
        assert_eq!(f32::from_bits(output[50]), canonical[6].position[2]);
        let actual_nonfinite_scale = unpack_covariance(&output[60..63]);
        let expected_nonfinite_scale = MotionState {
            position: canonical[7].position,
            log_scale: canonical[7].log_scale,
            rotation: [4.0_f32.cos(), 4.0_f32.sin(), 0.0, 0.0],
        }
        .covariance();
        for (actual, expected) in actual_nonfinite_scale
            .into_iter()
            .zip(expected_nonfinite_scale)
        {
            assert!(
                (actual - expected).abs() < 0.02,
                "nonfinite effective log scale must fall back to canonical scale"
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_diagnostic_state(
        runtime: &mut GpuMotionRuntime,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sample: TimelineSample,
        row: u32,
    ) -> MotionStateGpu {
        read_diagnostic_state_masked(runtime, device, queue, sample, row, ALL_MOTION_MASK)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_diagnostic_state_masked(
        runtime: &mut GpuMotionRuntime,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sample: TimelineSample,
        row: u32,
        preview_channel_mask: u32,
    ) -> MotionStateGpu {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion state parity encoder"),
        });
        runtime
            .dispatch_with_diagnostic(
                queue,
                &mut encoder,
                sample,
                true,
                preview_channel_mask,
                Some(row),
            )
            .unwrap();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion state parity readback"),
            size: std::mem::size_of::<MotionStateGpu>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            runtime.diagnostic_buffer(),
            0,
            &readback,
            0,
            std::mem::size_of::<MotionStateGpu>() as u64,
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let state = *bytemuck::from_bytes::<MotionStateGpu>(&mapped);
        drop(mapped);
        readback.unmap();
        state
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_diagnostic_state_masked_with_gains(
        runtime: &mut GpuMotionRuntime,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sample: TimelineSample,
        row: u32,
        preview_channel_mask: u32,
        gains: MotionChannelGains,
    ) -> MotionStateGpu {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion gains parity encoder"),
        });
        runtime
            .dispatch_with_diagnostic_and_gains(
                queue,
                &mut encoder,
                sample,
                true,
                preview_channel_mask,
                gains,
                Some(row),
            )
            .unwrap();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion gains parity readback"),
            size: std::mem::size_of::<MotionStateGpu>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            runtime.diagnostic_buffer(),
            0,
            &readback,
            0,
            std::mem::size_of::<MotionStateGpu>() as u64,
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let state = *bytemuck::from_bytes::<MotionStateGpu>(&mapped);
        drop(mapped);
        readback.unmap();
        state
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn read_output_texture(
        runtime: &mut GpuMotionRuntime,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sample: TimelineSample,
        width: u32,
    ) -> Vec<u32> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("packed motion parity encoder"),
        });
        runtime.dispatch(queue, &mut encoder, sample, true).unwrap();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("packed motion parity readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &runtime.output_texture().texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output = bytemuck::cast_slice::<u8, u32>(&mapped[..width as usize * 16]).to_vec();
        drop(mapped);
        readback.unmap();
        output
    }

    fn assert_state_close(label: &str, actual: MotionStateGpu, expected: MotionState) {
        for component in 0..3 {
            assert!(
                (actual.position[component] - expected.position[component]).abs() <= 2.0e-5,
                "{label} position {component}: {} != {}",
                actual.position[component],
                expected.position[component]
            );
            assert!(
                (actual.log_scale[component] - expected.log_scale[component]).abs() <= 2.0e-5,
                "{label} log scale {component}: {} != {}",
                actual.log_scale[component],
                expected.log_scale[component]
            );
        }
        let actual_norm = actual
            .rotation
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let expected_norm = expected
            .rotation
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let dot = (actual
            .rotation
            .iter()
            .zip(expected.rotation)
            .map(|(left, right)| left * right)
            .sum::<f32>()
            / (actual_norm * expected_norm))
            .abs()
            .clamp(-1.0, 1.0);
        let angular_error = 2.0 * dot.acos();
        assert!(
            angular_error <= 1.0e-4,
            "{label} quaternion angular error is {angular_error}"
        );
        let actual_covariance = MotionState {
            position: actual.position[..3].try_into().unwrap(),
            log_scale: actual.log_scale[..3].try_into().unwrap(),
            rotation: actual.rotation,
        }
        .covariance();
        let expected_covariance = expected.covariance();
        for component in 0..6 {
            let difference = (actual_covariance[component] - expected_covariance[component]).abs();
            let tolerance = 2.0e-4 * expected_covariance[component].abs().max(1.0e-6);
            assert!(
                difference <= tolerance,
                "{label} covariance {component}: {} != {} (difference {difference}, tolerance {tolerance})",
                actual_covariance[component],
                expected_covariance[component]
            );
        }
    }

    fn unpack_covariance(packed: &[u32]) -> [f32; 6] {
        let mut result = [0.0; 6];
        for (pair, value) in packed.iter().copied().enumerate() {
            result[pair * 2] = half::f16::from_bits(value as u16).to_f32() / 4.0;
            result[pair * 2 + 1] = half::f16::from_bits((value >> 16) as u16).to_f32() / 4.0;
        }
        result
    }
}
