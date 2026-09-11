use wgpu::util::DeviceExt;

use crate::motion::TimelineSample;
use crate::motion_brush_gpu::GpuMotionField;
use crate::motion_controller_palette::{MAX_LOCAL_CONTROLLERS, MotionControllerFrame};
use crate::motion_gpu::{BasisGpu, GpuMotionRuntime};
use crate::motion_tagging::AuthoredRegistryUpdate;
use crate::texture::Texture;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AuthoredOccurrenceInputGpu {
    pub local_row_output: [u32; 4],
    pub occurrence_offset: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AuthoredChunkInput {
    pub chunk_index: usize,
    pub local_row: u32,
    pub output_index: u32,
    pub occurrence_offset: [f32; 3],
}

fn pack_registry_inputs<F>(
    update: &AuthoredRegistryUpdate,
    mut locate_row: F,
) -> Result<Vec<AuthoredChunkInput>, String>
where
    F: FnMut(usize) -> Option<(usize, u32)>,
{
    if update.inputs.len() != update.output_base_rows.len() {
        return Err("authored registry inputs and base rows are not aligned".to_string());
    }
    let mut packed = Vec::with_capacity(update.inputs.len());
    for (output_index, input) in update.inputs.iter().enumerate() {
        if usize::try_from(input.output_index).ok() != Some(output_index) {
            return Err("authored registry outputs must be compact and ordered".to_string());
        }
        if update.output_base_rows[output_index] != input.global_row {
            return Err("authored registry base row does not match its input".to_string());
        }
        if input
            .occurrence_offset
            .iter()
            .any(|component| !component.is_finite())
        {
            return Err("authored occurrence offset must be finite".to_string());
        }
        let global_row = usize::try_from(input.global_row)
            .map_err(|_| "authored registry row cannot be represented".to_string())?;
        let (chunk_index, local_row) = locate_row(global_row)
            .ok_or_else(|| "authored registry row is absent from dynamic motion".to_string())?;
        packed.push(AuthoredChunkInput {
            chunk_index,
            local_row,
            output_index: input.output_index,
            occurrence_offset: input.occurrence_offset,
        });
    }
    Ok(packed)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct ControllerMetadataGpu {
    translation_master_gain: [f32; 4],
    rotation_scale_blend_valid: [f32; 4],
    _padding: [f32; 4],
}

impl ControllerMetadataGpu {
    const fn zero() -> Self {
        Self {
            translation_master_gain: [0.0; 4],
            rotation_scale_blend_valid: [0.0; 4],
            _padding: [0.0; 4],
        }
    }

    fn from_frame(frame: MotionControllerFrame) -> Result<Self, String> {
        frame.gains.validate().map_err(|error| error.to_string())?;
        let blend_alpha = match frame.sample {
            TimelineSample::Source { u } => {
                if !u.is_finite() {
                    return Err("controller source sample must be finite".to_string());
                }
                0.0
            }
            TimelineSample::Blend {
                from_u,
                to_u,
                alpha,
            } => {
                if !from_u.is_finite() || !to_u.is_finite() || !alpha.is_finite() {
                    return Err("controller blend sample must be finite".to_string());
                }
                alpha.clamp(0.0, 1.0)
            }
        };
        Ok(Self {
            translation_master_gain: [
                frame.gains.translation[0],
                frame.gains.translation[1],
                frame.gains.translation[2],
                frame.gains.master,
            ],
            rotation_scale_blend_valid: [
                frame.gains.rotation,
                frame.gains.scale,
                blend_alpha,
                f32::from(frame.valid),
            ],
            _padding: [0.0; 4],
        })
    }
}

fn pack_controller_metadata(
    global: MotionControllerFrame,
    locals: &[MotionControllerFrame],
) -> Result<[ControllerMetadataGpu; MAX_LOCAL_CONTROLLERS + 1], String> {
    if global.slot != 0 || !global.valid {
        return Err("global controller slot zero must be valid".to_string());
    }
    let mut metadata = [ControllerMetadataGpu::zero(); MAX_LOCAL_CONTROLLERS + 1];
    metadata[0] = ControllerMetadataGpu::from_frame(global)?;
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
        if frame.valid {
            metadata[slot] = ControllerMetadataGpu::from_frame(frame)?;
        }
    }
    Ok(metadata)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct AuthoredParamsGpu {
    input_count: u32,
    row_start: u32,
    top_k: u32,
    basis_count: u32,
    texture_width: u32,
    output_width: u32,
    motion_enabled: u32,
    preview_channel_mask: u32,
    coefficient_bank_count: u32,
    _padding: [u32; 3],
}

struct GpuAuthoredMotionChunk {
    input_buffer: wgpu::Buffer,
    input_capacity: usize,
    input_count: usize,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct AuthoredMotionDispatch {
    pub dispatched: bool,
    pub registry_occurrences: usize,
    pub tagged_occurrences: usize,
    pub affected_draws: usize,
    pub uploaded_bytes: u64,
    pub registry_upload_bytes: u64,
    pub membership_request_revision: u64,
    pub registry_revision: u64,
    pub worker_tag_ms: f64,
    pub registry_installed: bool,
    pub output_reallocated: bool,
}

pub(crate) struct GpuMotionControllerRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    controller_frames_buffer: wgpu::Buffer,
    controller_metadata_buffer: wgpu::Buffer,
    output_base_rows_buffer: wgpu::Buffer,
    output_base_rows_capacity: usize,
    chunks: Vec<GpuAuthoredMotionChunk>,
    output_texture: Texture,
    output_width: u32,
    output_capacity: usize,
    pending_registry_upload_bytes: u64,
    output_reallocated_since_dispatch: bool,
}

impl GpuMotionControllerRuntime {
    pub(crate) fn new(
        device: &wgpu::Device,
        motion: &GpuMotionRuntime,
        field: &GpuMotionField,
    ) -> Result<Self, String> {
        if field
            .dimensions()
            .into_iter()
            .any(|dimension| dimension == 0)
        {
            return Err("motion authoring field dimensions must be positive".to_string());
        }
        let limits = device.limits();
        let controller_frame_count = (MAX_LOCAL_CONTROLLERS + 1)
            .checked_mul(2)
            .and_then(|value| value.checked_mul(motion.coefficient_bank_count()))
            .and_then(|value| value.checked_mul(motion.basis_count()))
            .ok_or_else(|| "controller basis frame count overflow".to_string())?;
        let controller_frame_bytes = controller_frame_count
            .checked_mul(std::mem::size_of::<BasisGpu>())
            .ok_or_else(|| "controller basis frame size overflow".to_string())?;
        if controller_frame_bytes as u64 > limits.max_storage_buffer_binding_size as u64 {
            return Err(format!(
                "controller basis frames require {controller_frame_bytes} bytes, exceeding the adapter storage-binding limit {}",
                limits.max_storage_buffer_binding_size
            ));
        }
        let metadata_bytes = std::mem::size_of::<ControllerMetadataGpu>()
            .checked_mul(MAX_LOCAL_CONTROLLERS + 1)
            .ok_or_else(|| "controller metadata size overflow".to_string())?;
        if metadata_bytes as u64 > limits.max_uniform_buffer_binding_size as u64 {
            return Err(format!(
                "controller metadata requires {metadata_bytes} bytes, exceeding the adapter uniform-binding limit {}",
                limits.max_uniform_buffer_binding_size
            ));
        }
        let controller_frames_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Authored Motion Controller Basis Frames"),
                contents: &vec![0_u8; controller_frame_bytes],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let controller_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Authored Motion Controller Metadata"),
            size: metadata_bytes as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_base_rows_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Authored Motion Output Base Rows"),
                contents: bytemuck::bytes_of(&0_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group_layout = create_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Authored Motion Controller Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Authored Motion Controller Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("motion_controller_compute.wgsl").into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Authored Motion Controller Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let output_width = motion
            .texture_width()
            .min(limits.max_texture_dimension_2d)
            .max(1);
        let output_texture = create_output_texture(device, output_width, 1)?;
        let mut chunks = Vec::with_capacity(motion.chunks().len());
        for motion_chunk in motion.chunks() {
            let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Authored Motion Occurrence Inputs"),
                size: std::mem::size_of::<AuthoredOccurrenceInputGpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let params = AuthoredParamsGpu {
                input_count: 0,
                row_start: motion_chunk.row_start() as u32,
                top_k: motion.top_k() as u32,
                basis_count: motion.basis_count() as u32,
                texture_width: motion.texture_width(),
                output_width,
                motion_enabled: 0,
                preview_channel_mask: motion.preview_channel_mask(),
                coefficient_bank_count: motion.coefficient_bank_count() as u32,
                _padding: [0; 3],
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Authored Motion Parameters"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = create_chunk_bind_group(
                device,
                &bind_group_layout,
                motion_chunk,
                motion,
                field,
                &controller_frames_buffer,
                &controller_metadata_buffer,
                &input_buffer,
                &output_texture,
                &params_buffer,
            );
            chunks.push(GpuAuthoredMotionChunk {
                input_buffer,
                input_capacity: 1,
                input_count: 0,
                params_buffer,
                bind_group,
            });
        }
        Ok(Self {
            compute_pipeline,
            bind_group_layout,
            controller_frames_buffer,
            controller_metadata_buffer,
            output_base_rows_buffer,
            output_base_rows_capacity: 1,
            chunks,
            output_texture,
            output_width,
            output_capacity: 0,
            pending_registry_upload_bytes: 0,
            output_reallocated_since_dispatch: false,
        })
    }

    pub(crate) fn update_registry(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        motion: &GpuMotionRuntime,
        field: &GpuMotionField,
        update: &AuthoredRegistryUpdate,
    ) -> Result<bool, String> {
        let inputs = pack_registry_inputs(update, |row| motion.locate_row(row))?;
        self.update_inputs(
            device,
            queue,
            motion,
            field,
            &inputs,
            &update.output_base_rows,
        )
    }

    fn update_inputs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        motion: &GpuMotionRuntime,
        field: &GpuMotionField,
        inputs: &[AuthoredChunkInput],
        output_base_rows: &[u32],
    ) -> Result<bool, String> {
        if self.chunks.len() != motion.chunks().len() {
            return Err("authored and global motion chunk counts differ".to_string());
        }
        let mut grouped = vec![Vec::<AuthoredOccurrenceInputGpu>::new(); self.chunks.len()];
        let output_count = output_base_rows.len();
        let mut seen_outputs = vec![false; output_count];
        for input in inputs {
            let motion_chunk = motion.chunks().get(input.chunk_index).ok_or_else(|| {
                "authored occurrence references an invalid motion chunk".to_string()
            })?;
            if input.local_row as usize >= motion_chunk.row_count() {
                return Err("authored occurrence local row exceeds its motion chunk".to_string());
            }
            let output_index = input.output_index as usize;
            if output_index >= output_count {
                return Err("authored occurrence output index is out of range".to_string());
            }
            if std::mem::replace(&mut seen_outputs[output_index], true) {
                return Err("authored occurrence output index is duplicated".to_string());
            }
            if input
                .occurrence_offset
                .iter()
                .any(|component| !component.is_finite())
            {
                return Err("authored occurrence offset must be finite".to_string());
            }
            grouped[input.chunk_index].push(AuthoredOccurrenceInputGpu {
                local_row_output: [input.local_row, input.output_index, 0, 0],
                occurrence_offset: [
                    input.occurrence_offset[0],
                    input.occurrence_offset[1],
                    input.occurrence_offset[2],
                    0.0,
                ],
            });
        }
        if inputs.len() != output_count || seen_outputs.iter().any(|seen| !seen) {
            return Err("authored occurrence outputs must be contiguous and complete".to_string());
        }
        let mut bindings_changed = false;
        if output_base_rows.len() > self.output_base_rows_capacity {
            self.output_base_rows_capacity = output_base_rows
                .len()
                .checked_next_power_of_two()
                .ok_or_else(|| "authored base-row capacity overflow".to_string())?;
            let size = self
                .output_base_rows_capacity
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| "authored base-row storage size overflow".to_string())?;
            if size as u64 > device.limits().max_storage_buffer_binding_size as u64 {
                return Err(
                    "authored base-row storage exceeds the adapter storage-binding limit"
                        .to_string(),
                );
            }
            self.output_base_rows_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Authored Motion Output Base Rows"),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            bindings_changed = true;
        }
        let mut base_row_upload_bytes = 0_u64;
        if !output_base_rows.is_empty() {
            queue.write_buffer(
                &self.output_base_rows_buffer,
                0,
                bytemuck::cast_slice(output_base_rows),
            );
            base_row_upload_bytes = std::mem::size_of_val(output_base_rows) as u64;
        }
        if output_count > self.output_capacity {
            let grown_capacity = output_count
                .checked_next_power_of_two()
                .ok_or_else(|| "authored occurrence output capacity overflow".to_string())?;
            let required_pixels = grown_capacity
                .checked_mul(2)
                .ok_or_else(|| "authored occurrence output pixel count overflow".to_string())?;
            let output_height = u32::try_from(required_pixels.div_ceil(self.output_width as usize))
                .map_err(|_| "authored occurrence output height exceeds u32".to_string())?;
            if output_height > device.limits().max_texture_dimension_2d {
                return Err("authored occurrence texture exceeds the adapter 2D limit".to_string());
            }
            self.output_texture =
                create_output_texture(device, self.output_width, output_height.max(1))?;
            self.output_capacity = grown_capacity;
            self.output_reallocated_since_dispatch = true;
            bindings_changed = true;
        }
        let mut uploaded_bytes = 0_u64;
        for (chunk, packed) in self.chunks.iter_mut().zip(&grouped) {
            if packed.len() > chunk.input_capacity {
                let grown_capacity = packed
                    .len()
                    .checked_next_power_of_two()
                    .ok_or_else(|| "authored occurrence input capacity overflow".to_string())?;
                let size = grown_capacity
                    .checked_mul(std::mem::size_of::<AuthoredOccurrenceInputGpu>())
                    .ok_or_else(|| "authored occurrence input size overflow".to_string())?;
                if size as u64 > device.limits().max_storage_buffer_binding_size as u64 {
                    return Err("authored occurrence input buffer exceeds the adapter storage-binding limit".to_string());
                }
                chunk.input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Authored Motion Occurrence Inputs"),
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                chunk.input_capacity = grown_capacity;
                bindings_changed = true;
            }
            chunk.input_count = packed.len();
            if !packed.is_empty() {
                queue.write_buffer(&chunk.input_buffer, 0, bytemuck::cast_slice(packed));
                uploaded_bytes =
                    uploaded_bytes.saturating_add(std::mem::size_of_val(packed.as_slice()) as u64);
            }
        }
        if bindings_changed {
            for (index, chunk) in self.chunks.iter_mut().enumerate() {
                chunk.bind_group = create_chunk_bind_group(
                    device,
                    &self.bind_group_layout,
                    &motion.chunks()[index],
                    motion,
                    field,
                    &self.controller_frames_buffer,
                    &self.controller_metadata_buffer,
                    &chunk.input_buffer,
                    &self.output_texture,
                    &chunk.params_buffer,
                );
            }
        }
        self.pending_registry_upload_bytes = uploaded_bytes.saturating_add(base_row_upload_bytes);
        Ok(bindings_changed)
    }

    pub(crate) fn dispatch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        motion: &GpuMotionRuntime,
        global: MotionControllerFrame,
        locals: &[MotionControllerFrame],
        motion_enabled: bool,
        preview_channel_mask: u32,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<AuthoredMotionDispatch, String> {
        if preview_channel_mask & !crate::motion::ALL_MOTION_MASK != 0 {
            return Err("preview channel mask contains reserved bits".to_string());
        }
        let metadata = pack_controller_metadata(global, locals)?;
        let packed_slots = motion.pack_active_controller_slots(global, locals)?;
        let slot_stride_bytes = motion.controller_slot_stride_bytes()?;
        let mut controller_upload_bytes = 0_u64;
        for (slot, packed) in &packed_slots {
            queue.write_buffer(
                &self.controller_frames_buffer,
                u64::from(*slot) * slot_stride_bytes,
                bytemuck::cast_slice(packed),
            );
            controller_upload_bytes = controller_upload_bytes
                .saturating_add(std::mem::size_of_val(packed.as_slice()) as u64);
        }
        queue.write_buffer(
            &self.controller_metadata_buffer,
            0,
            bytemuck::cast_slice(&metadata),
        );
        let mut affected_rows = 0_usize;
        for (chunk, motion_chunk) in self.chunks.iter().zip(motion.chunks()) {
            affected_rows = affected_rows.saturating_add(chunk.input_count);
            let params = AuthoredParamsGpu {
                input_count: chunk.input_count as u32,
                row_start: motion_chunk.row_start() as u32,
                top_k: motion.top_k() as u32,
                basis_count: motion.basis_count() as u32,
                texture_width: motion.texture_width(),
                output_width: self.output_width,
                motion_enabled: u32::from(motion_enabled),
                preview_channel_mask,
                coefficient_bank_count: motion.coefficient_bank_count() as u32,
                _padding: [0; 3],
            };
            queue.write_buffer(&chunk.params_buffer, 0, bytemuck::bytes_of(&params));
        }
        if affected_rows > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Authored Occurrence Motion Update"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.compute_pipeline);
            for chunk in &self.chunks {
                if chunk.input_count == 0 {
                    continue;
                }
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                pass.dispatch_workgroups((chunk.input_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        let registry_upload_bytes = std::mem::take(&mut self.pending_registry_upload_bytes);
        let uploaded_bytes = usize::try_from(controller_upload_bytes)
            .unwrap_or(usize::MAX)
            .saturating_add(std::mem::size_of_val(&metadata))
            .saturating_add(usize::try_from(registry_upload_bytes).unwrap_or(usize::MAX))
            .saturating_add(
                self.chunks
                    .len()
                    .saturating_mul(std::mem::size_of::<AuthoredParamsGpu>()),
            );
        Ok(AuthoredMotionDispatch {
            dispatched: affected_rows > 0,
            registry_occurrences: affected_rows,
            tagged_occurrences: affected_rows,
            affected_draws: 0,
            uploaded_bytes: uploaded_bytes as u64,
            registry_upload_bytes,
            output_reallocated: std::mem::take(&mut self.output_reallocated_since_dispatch),
            ..AuthoredMotionDispatch::default()
        })
    }

    pub(crate) fn output_view(&self) -> &wgpu::TextureView {
        &self.output_texture.view
    }

    pub(crate) fn output_base_rows_buffer(&self) -> &wgpu::Buffer {
        &self.output_base_rows_buffer
    }

    #[cfg(test)]
    fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture.texture
    }
}

fn create_output_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<Texture, String> {
    if width == 0 || height == 0 {
        return Err("authored occurrence texture dimensions must be positive".to_string());
    }
    Texture::empty_with_usage(
        device,
        cgmath::Vector2::new(width, height),
        wgpu::TextureFormat::Rgba32Uint,
        wgpu::FilterMode::Nearest,
        wgpu::AddressMode::ClampToEdge,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        Some("Authored Occurrence Motion Output"),
    )
    .map_err(|error| error.to_string())
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Authored Motion Controller Bind Group Layout"),
        entries: &[
            storage_layout(0),
            storage_layout(1),
            storage_layout(2),
            storage_layout(3),
            storage_layout(4),
            storage_layout(5),
            storage_layout(6),
            storage_layout(7),
            sampled_texture_layout(8, wgpu::TextureSampleType::Uint),
            sampled_texture_layout(9, wgpu::TextureSampleType::Float { filterable: false }),
            sampled_texture_layout(10, wgpu::TextureSampleType::Uint),
            uniform_layout(11),
            uniform_layout(12),
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            uniform_layout(14),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_chunk_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    chunk: &crate::motion_gpu::GpuMotionChunk,
    motion: &GpuMotionRuntime,
    field: &GpuMotionField,
    controller_frames: &wgpu::Buffer,
    controller_metadata: &wgpu::Buffer,
    inputs: &wgpu::Buffer,
    output: &Texture,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Authored Motion Controller Chunk Bind Group"),
        layout,
        entries: &[
            buffer_entry(0, chunk.canonical_buffer()),
            buffer_entry(1, chunk.basis_ids_buffer()),
            buffer_entry(2, chunk.weights_buffer()),
            buffer_entry(3, chunk.transform_ids_buffer()),
            buffer_entry(4, controller_frames),
            buffer_entry(5, motion.placement_transforms_buffer()),
            buffer_entry(6, chunk.motion_channel_masks_buffer()),
            buffer_entry(7, inputs),
            texture_entry(8, &motion.output_texture().view),
            texture_entry(9, field.continuous_view()),
            texture_entry(10, field.assignment_view()),
            buffer_entry(11, field.uniform_buffer()),
            buffer_entry(12, controller_metadata),
            texture_entry(13, &output.view),
            buffer_entry(14, params),
        ],
    })
}

fn storage_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn sampled_texture_layout(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn texture_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::dynamic_archive::BasisBank;
    use crate::motion::{
        ALL_MOTION_MASK, CanonicalGaussian, MergedMotion, MergedMotionData, TimelineSample,
    };
    use crate::motion_behavior::MotionChannelGains;
    use crate::motion_brush::{
        MotionBrushDocument, MotionBrushFalloff, MotionBrushTool, MotionFieldCache,
        MotionFieldLayout, MotionFieldQuality,
    };
    use crate::motion_controller_palette::MotionRegionId;
    use crate::motion_tagging::{AuthoredOccurrenceInput, AuthoredRegistryUpdate};
    use crate::scene_archive::DynamicArchiveSummary;

    #[test]
    fn gpu_layout_is_literal_and_controller_metadata_is_slot_stable() {
        assert_eq!(std::mem::size_of::<AuthoredOccurrenceInputGpu>(), 32);
        assert_eq!(std::mem::align_of::<AuthoredOccurrenceInputGpu>(), 4);
        assert_eq!(std::mem::size_of::<ControllerMetadataGpu>(), 48);

        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Blend {
                from_u: 0.1,
                to_u: 0.8,
                alpha: 0.25,
            },
            gains: MotionChannelGains {
                translation: [0.5, 0.75, 1.25],
                rotation: 0.8,
                scale: 0.6,
                master: 1.5,
            },
            valid: true,
        };
        let local = MotionControllerFrame {
            slot: 3,
            sample: TimelineSample::Source { u: 0.5 },
            gains: MotionChannelGains::default(),
            valid: true,
        };

        let metadata = pack_controller_metadata(global, &[local]).unwrap();

        assert_eq!(metadata[0].translation_master_gain, [0.5, 0.75, 1.25, 1.5]);
        assert_eq!(
            metadata[0].rotation_scale_blend_valid,
            [0.8, 0.6, 0.25, 1.0]
        );
        assert_eq!(metadata[1], ControllerMetadataGpu::zero());
        assert_eq!(metadata[3].rotation_scale_blend_valid[3], 1.0);
        assert_eq!(metadata[4], ControllerMetadataGpu::zero());
    }

    #[test]
    fn controller_metadata_rejects_duplicate_local_slots() {
        let global = MotionControllerFrame {
            slot: 0,
            sample: TimelineSample::Source { u: 0.0 },
            gains: MotionChannelGains::default(),
            valid: true,
        };
        let local = MotionControllerFrame { slot: 2, ..global };

        let error = pack_controller_metadata(global, &[local, local]).unwrap_err();

        assert!(error.contains("duplicate"), "{error}");
    }

    #[test]
    fn registry_upload_packs_compact_outputs_in_registry_order() {
        let update = AuthoredRegistryUpdate {
            request_revision: 3,
            registry_revision: 7,
            scene_id: 11,
            inputs: Arc::from([
                AuthoredOccurrenceInput {
                    global_row: 9,
                    output_index: 0,
                    occurrence_offset: [4.0, 8.0, 0.0],
                },
                AuthoredOccurrenceInput {
                    global_row: 2,
                    output_index: 1,
                    occurrence_offset: [-4.0, 0.0, 1.0],
                },
            ]),
            output_base_rows: Arc::from([9_u32, 2]),
        };

        let packed = pack_registry_inputs(&update, |row| match row {
            9 => Some((1, 4)),
            2 => Some((0, 2)),
            _ => None,
        })
        .unwrap();

        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].chunk_index, 1);
        assert_eq!(packed[0].local_row, 4);
        assert_eq!(packed[0].output_index, 0);
        assert_eq!(packed[1].chunk_index, 0);
        assert_eq!(packed[1].local_row, 2);
        assert_eq!(packed[1].output_index, 1);
        assert_eq!(&*update.output_base_rows, &[9, 2]);
    }

    #[test]
    fn registry_upload_rejects_noncompact_or_misaligned_outputs() {
        let make = |inputs: Vec<AuthoredOccurrenceInput>, rows: Vec<u32>| AuthoredRegistryUpdate {
            request_revision: 3,
            registry_revision: 7,
            scene_id: 11,
            inputs: Arc::from(inputs),
            output_base_rows: Arc::from(rows),
        };
        let valid = AuthoredOccurrenceInput {
            global_row: 9,
            output_index: 0,
            occurrence_offset: [0.0; 3],
        };

        assert!(pack_registry_inputs(&make(vec![valid], vec![]), |_| Some((0, 0))).is_err());
        assert!(
            pack_registry_inputs(
                &make(
                    vec![AuthoredOccurrenceInput {
                        output_index: 1,
                        ..valid
                    }],
                    vec![9],
                ),
                |_| Some((0, 0)),
            )
            .is_err()
        );
        assert!(pack_registry_inputs(&make(vec![valid], vec![8]), |_| Some((0, 0))).is_err());
        assert!(pack_registry_inputs(&make(vec![valid], vec![9]), |_| None).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn occurrence_compute_applies_local_controller_and_copies_outside_field_exactly() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(error) => {
                    eprintln!("authored occurrence GPU test skipped: no adapter ({error})");
                    return;
                }
            };
        if !adapter
            .get_texture_format_features(wgpu::TextureFormat::Rgba32Uint)
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            eprintln!("authored occurrence GPU test skipped: rgba32uint storage unsupported");
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("authored occurrence parity device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();
        let mut basis_values = Vec::with_capacity(75 * 9);
        for sample in 0..75 {
            let u = sample as f32 / 74.0;
            basis_values.extend_from_slice(&[u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let motion = MergedMotion {
            data: MergedMotionData::Legacy {
                basis: BasisBank {
                    basis_count: 1,
                    values: basis_values,
                },
                top_k: 1,
                basis_ids: vec![0],
                weights: vec![1.0],
            },
            canonical: vec![CanonicalGaussian {
                position: [1.0, 1.0, 0.0],
                log_scale: [0.0; 3],
                rotation: [1.0, 0.0, 0.0, 0.0],
            }],
            transform_ids: vec![0],
            motion_channel_masks: vec![ALL_MOTION_MASK],
            source_duration_seconds: 1.0,
            summary: DynamicArchiveSummary {
                schema_version: 1,
                tile_count: 1,
                lod_count: 1,
                basis_count: 1,
                top_k: 1,
                total_rows: 1,
                backend: "authored-occurrence-test".into(),
            },
            member_offsets: vec![vec![0]],
            member_counts: vec![vec![1]],
        };
        let mut global =
            GpuMotionRuntime::new(&device, &queue, motion, &[0_u32; 32], 8, 1).unwrap();
        let mut global_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        global
            .dispatch(
                &queue,
                &mut global_encoder,
                TimelineSample::Source { u: 0.0 },
                true,
            )
            .unwrap();
        queue.submit(Some(global_encoder.finish()));

        for (opacity, local_sample, expected_x) in [
            (1.0, TimelineSample::Source { u: 1.0 }, 2.0),
            (
                1.0,
                TimelineSample::Blend {
                    from_u: 0.2,
                    to_u: 1.0,
                    alpha: 0.5,
                },
                1.6,
            ),
            (
                0.5,
                TimelineSample::Blend {
                    from_u: 0.2,
                    to_u: 1.0,
                    alpha: 0.5,
                },
                1.0 + 0.6 * 128.0 / 255.0,
            ),
            (
                1.0,
                TimelineSample::Blend {
                    from_u: 0.2,
                    to_u: 1.0,
                    alpha: 0.0,
                },
                1.2,
            ),
        ] {
            let layout =
                MotionFieldLayout::new([1, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
            let mut document = MotionBrushDocument::default();
            document
                .push_stroke(
                    MotionBrushTool::ApplyBehavior {
                        region_id: MotionRegionId::local(1).unwrap(),
                    },
                    vec![[1.0, 1.0]],
                    1.0,
                    opacity,
                    0.25,
                    MotionBrushFalloff::Constant,
                )
                .unwrap();
            let cache = MotionFieldCache::rebuild(layout, &document).unwrap();
            let field = GpuMotionField::new(&device, &queue, &cache).unwrap();
            let mut authored = GpuMotionControllerRuntime::new(&device, &global, &field).unwrap();
            authored
                .update_inputs(
                    &device,
                    &queue,
                    &global,
                    &field,
                    &[
                        AuthoredChunkInput {
                            chunk_index: 0,
                            local_row: 0,
                            output_index: 0,
                            occurrence_offset: [0.0; 3],
                        },
                        AuthoredChunkInput {
                            chunk_index: 0,
                            local_row: 0,
                            output_index: 1,
                            occurrence_offset: [100.0, 0.0, 0.0],
                        },
                    ],
                    &[0, 0],
                )
                .unwrap();
            let local = MotionControllerFrame {
                slot: 1,
                sample: local_sample,
                gains: MotionChannelGains::default(),
                valid: true,
            };
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            authored
                .dispatch(
                    &queue,
                    &mut encoder,
                    &global,
                    global.global_controller_frame(),
                    &[local],
                    true,
                    ALL_MOTION_MASK,
                    None,
                )
                .unwrap();
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("authored occurrence parity readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let inherited_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("authored occurrence inherited readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: authored.output_texture(),
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
                    width: 4,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &global.output_texture().texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &inherited_readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 2,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            let inherited_slice = inherited_readback.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            let inherited_sender = sender.clone();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap()
            });
            inherited_slice.map_async(wgpu::MapMode::Read, move |result| {
                inherited_sender.send(result).unwrap()
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            receiver.recv().unwrap().unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let inherited_mapped = inherited_slice.get_mapped_range();
            let words = bytemuck::cast_slice::<u8, u32>(&mapped[..64]);
            let inherited_words = bytemuck::cast_slice::<u8, u32>(&inherited_mapped[..32]);

            assert!(
                (f32::from_bits(words[0]) - expected_x).abs() < 1.0e-5,
                "opacity={opacity}, sample={local_sample:?}"
            );
            assert_eq!(f32::from_bits(words[8]), 1.0);
            assert_eq!(&words[8..16], inherited_words);
            drop(inherited_mapped);
            drop(mapped);
            inherited_readback.unmap();
            readback.unmap();
        }
    }
}
