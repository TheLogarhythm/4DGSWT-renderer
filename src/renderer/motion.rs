//! Motion GPU preparation, field synchronization, authored dispatch, and preview support.
use std::sync::Arc;

use super::GSWTRenderer;
use crate::log;
use crate::motion::{MergedMotion, TimelineSample};
use crate::motion_brush::{
    DirtyRect, MotionBrushPreview, MotionBrushRuntime, MotionFieldCache, MotionOverlayChannel,
};
use crate::motion_brush_gpu::{GpuMotionField, MotionFieldUniform};
use crate::motion_controller_gpu::{AuthoredMotionDispatch, GpuMotionControllerRuntime};
use crate::motion_controller_palette::{CompiledMotionField, MotionControllerFrame};
use crate::motion_gpu::GpuMotionRuntime;
use crate::motion_graph::{MotionGraph, MotionGraphSettings};
use crate::motion_graph_analysis::build_motion_graph;
use crate::profiler::MotionDispatchWork;
use crate::scene_archive::DynamicArchiveSummary;
use crate::structure::{
    RenderDataKey, RenderDataValue, SceneData, SortData, TileInstance, TileTransitionStatusHash,
    UserData,
};
use crate::texture::Texture;
use crate::utils::{get_time_milliseconds, transmute_slice};

pub(super) struct PreparedMotion {
    pub(super) gaussian_texture: Texture,
    pub(super) motion_runtime: Option<GpuMotionRuntime>,
    pub(super) motion_graph: Option<Arc<MotionGraph>>,
    pub(super) motion_graph_error: Option<String>,
}

pub(super) fn prepare_motion(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    merged_motion: Option<MergedMotion>,
    base_texels: &[u32],
    texture_width: u32,
    texture_height: u32,
) -> Result<PreparedMotion, String> {
    let graph_start = get_time_milliseconds();
    let graph_outcome = merged_motion
        .as_ref()
        .map(build_motion_graph_outcome)
        .unwrap_or_default();
    if merged_motion.is_some() {
        log!(
            "Startup motion graph: {:.1} ms",
            get_time_milliseconds() - graph_start
        );
    }
    if let Some(graph) = graph_outcome.graph.as_ref() {
        let summary = graph.summary();
        log!(
            "Motion graph: {} jumps across {} segments ({} without jumps)",
            summary.jump_count,
            summary.segment_count,
            summary.segment_count - summary.segments_with_jumps
        );
    } else if let Some(error) = graph_outcome.error.as_deref() {
        log!("Motion graph unavailable; ordinary playback remains active: {error}");
    }
    let motion_runtime = merged_motion
        .map(|motion| {
            let preparation_start = get_time_milliseconds();
            let result = GpuMotionRuntime::new(
                device,
                queue,
                motion,
                base_texels,
                texture_width,
                texture_height,
            );
            log!(
                "Startup motion GPU preparation/submission: {:.1} ms",
                get_time_milliseconds() - preparation_start
            );
            result
        })
        .transpose()?;
    let gaussian_texture = if let Some(runtime) = motion_runtime.as_ref() {
        runtime.output_texture().clone()
    } else {
        Texture::from_bytes(
            device,
            queue,
            transmute_slice::<_, u8>(base_texels),
            texture_width,
            texture_height,
            16,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            Some("Gaussian Texture"),
        )
        .map_err(|error| error.to_string())?
    };
    Ok(PreparedMotion {
        gaussian_texture,
        motion_runtime,
        motion_graph: graph_outcome.graph,
        motion_graph_error: graph_outcome.error,
    })
}

impl GSWTRenderer {
    pub fn update_motion(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sample: TimelineSample,
        motion_enabled: bool,
        preview_channel_mask: u32,
        channel_gains: crate::motion_behavior::MotionChannelGains,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<Option<MotionDispatchWork>, String> {
        if let Some(runtime) = self.motion_runtime.as_mut() {
            return runtime
                .dispatch_masked_profiled(
                    queue,
                    encoder,
                    sample,
                    motion_enabled,
                    preview_channel_mask,
                    channel_gains,
                    timestamp_writes,
                )
                .map(Some);
        }
        Ok(None)
    }

    pub fn motion_summary(&self) -> Option<&DynamicArchiveSummary> {
        self.motion_runtime.as_ref().map(GpuMotionRuntime::summary)
    }

    pub fn motion_source_duration_seconds(&self) -> Option<f32> {
        self.motion_runtime
            .as_ref()
            .map(GpuMotionRuntime::source_duration_seconds)
    }

    pub fn motion_graph(&self) -> Option<Arc<MotionGraph>> {
        self.motion_graph.clone()
    }

    pub fn motion_graph_error(&self) -> Option<&str> {
        self.motion_graph_error.as_deref()
    }

    /// Publish a prepared field (or lightweight preview). Both initial config
    /// and ordinary frames use this transaction; failed uploads remain dirty.
    pub fn sync_authoring_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spatial: Option<&mut MotionBrushRuntime>,
        compiled: Option<&CompiledMotionField>,
    ) -> Result<u64, String> {
        let Some(spatial) = spatial.filter(|spatial| spatial.render_path_active()) else {
            self.disable_motion_field_rendering();
            return Ok(0);
        };
        let Some(compiled) = compiled else {
            self.disable_motion_field_rendering();
            return Err("motion field must be compiled before upload".into());
        };
        let active = spatial.field_data_active();
        let mut dirty = if active {
            spatial.take_dirty_regions()
        } else {
            Vec::new()
        };
        dirty.extend_from_slice(compiled.dirty_regions());
        dirty.sort_unstable_by_key(|rect| (rect.min(), rect.max_exclusive()));
        dirty.dedup();
        let result = if active {
            self.sync_motion_field(device, queue, spatial.cache(), compiled, &dirty)
        } else {
            self.sync_motion_preview(device, spatial.cache());
            Ok(0)
        };
        match result {
            Ok(bytes) => {
                self.update_motion_field_uniform(
                    queue,
                    spatial.cache(),
                    active,
                    spatial.overlay,
                    compiled.palette_count(),
                    spatial.preview(),
                );
                Ok(bytes)
            }
            Err(error) => {
                spatial.restore_dirty_regions(dirty);
                self.disable_motion_field_rendering();
                Err(error)
            }
        }
    }

    fn sync_motion_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        compiled: &CompiledMotionField,
        dirty_regions: &[DirtyRect],
    ) -> Result<u64, String> {
        let dimensions = cache.layout().texture_size();
        if self
            .motion_field
            .as_ref()
            .is_none_or(|field| field.dimensions() != dimensions)
        {
            let field =
                GpuMotionField::new(device, queue, cache).map_err(|error| error.to_string())?;
            field
                .upload_full_compiled(queue, cache, compiled)
                .map_err(|error| error.to_string())?;
            let stable_gaussian_view = self
                .motion_runtime
                .as_ref()
                .map(|runtime| &runtime.base_texture().view)
                .unwrap_or(&self.gaussian_texture.view);
            let mut motion_controller_runtime = self
                .motion_runtime
                .as_ref()
                .map(|motion| GpuMotionControllerRuntime::new(device, motion, &field))
                .transpose()?;
            if let (Some(controller), Some(motion), Some(update)) = (
                motion_controller_runtime.as_mut(),
                self.motion_runtime.as_ref(),
                self.authored_registry_update.as_ref(),
            ) {
                controller.update_registry(device, queue, motion, &field, update)?;
            }
            let authored_gaussian_view = motion_controller_runtime
                .as_ref()
                .map(GpuMotionControllerRuntime::output_view)
                .unwrap_or(stable_gaussian_view);
            let authored_base_rows_buffer = motion_controller_runtime
                .as_ref()
                .map(GpuMotionControllerRuntime::output_base_rows_buffer)
                .unwrap_or(&self.empty_authored_base_rows_buffer);
            let bind_group = create_motion_field_bind_group(
                device,
                &self.motion_field_bind_group_layout,
                &field,
                stable_gaussian_view,
                authored_gaussian_view,
                authored_base_rows_buffer,
            );
            self.motion_field = Some(field);
            self.motion_controller_runtime = motion_controller_runtime;
            self.motion_field_bind_group = Some(bind_group);
            if self.authored_registry_update.is_none() {
                self.authored_registry_key = None;
                self.authored_output_count = 0;
            }
            self.authored_dispatch_key = None;
            self.authored_runtime_error = None;
            return Ok(cache.layout().estimated_gpu_bytes());
        }
        let field = self
            .motion_field
            .as_ref()
            .expect("dimension check establishes a motion field");
        let mut uploaded_bytes = 0_u64;
        for &dirty in dirty_regions {
            uploaded_bytes = uploaded_bytes.saturating_add(
                field
                    .upload_dirty_compiled(queue, cache, compiled, dirty)
                    .map_err(|error| error.to_string())?
                    .uploaded_bytes,
            );
        }
        Ok(uploaded_bytes)
    }

    pub fn sync_motion_preview(&mut self, device: &wgpu::Device, cache: &MotionFieldCache) {
        if self.motion_field.is_some() {
            return;
        }
        let field = GpuMotionField::new_preview(device, cache.layout());
        let stable_gaussian_view = self
            .motion_runtime
            .as_ref()
            .map(|runtime| &runtime.base_texture().view)
            .unwrap_or(&self.gaussian_texture.view);
        let bind_group = create_motion_field_bind_group(
            device,
            &self.motion_field_bind_group_layout,
            &field,
            stable_gaussian_view,
            stable_gaussian_view,
            &self.empty_authored_base_rows_buffer,
        );
        self.motion_field = Some(field);
        self.motion_field_bind_group = Some(bind_group);
    }

    pub fn update_motion_field_uniform(
        &mut self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        field_data_active: bool,
        overlay: MotionOverlayChannel,
        palette_count: u32,
        preview: Option<MotionBrushPreview>,
    ) {
        let Some(field) = self.motion_field.as_mut() else {
            self.motion_field_render_active = false;
            return;
        };
        field.update_uniform(
            queue,
            MotionFieldUniform::new(
                cache.layout(),
                cache.storage_offset(),
                field_data_active,
                overlay,
                palette_count,
            )
            .with_preview(preview),
        );
        self.motion_field_render_active =
            (field_data_active || preview.is_some()) && self.motion_field_bind_group.is_some();
    }

    pub fn disable_motion_field_rendering(&mut self) {
        self.motion_field_render_active = false;
    }

    pub fn update_authored_motion(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sort_data: Option<&SortData>,
        scene_data: Option<&SceneData>,
        compiled: &CompiledMotionField,
        local_frames: &[MotionControllerFrame],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<Option<AuthoredMotionDispatch>, String> {
        let (Some(motion), Some(field), Some(controller), Some(sort_data), Some(scene_data)) = (
            self.motion_runtime.as_ref(),
            self.motion_field.as_ref(),
            self.motion_controller_runtime.as_mut(),
            sort_data,
            scene_data,
        ) else {
            self.authored_dispatch_key = None;
            return Ok(None);
        };
        let registry_key = AuthoredRegistryKey {
            request_revision: sort_data.authored.request_revision,
            registry_revision: sort_data.authored.registry_revision,
            scene_id: scene_data.scene_id,
        };
        let mut bindings_changed = false;
        let mut registry_installed = false;
        if self.authored_registry_key != Some(registry_key) {
            let update = sort_data.authored.update.as_ref().ok_or_else(|| {
                "authored sort references a registry revision that is not installed".to_string()
            })?;
            if update.request_revision != registry_key.request_revision
                || update.registry_revision != registry_key.registry_revision
                || update.scene_id != registry_key.scene_id
            {
                return Err("authored registry update does not match the accepted sort".to_string());
            }
            bindings_changed = controller.update_registry(device, queue, motion, field, update)?;
            self.authored_output_count = update.inputs.len();
            self.authored_registry_update = Some(update.clone());
            self.authored_registry_key = Some(registry_key);
            self.authored_dispatch_key = None;
            registry_installed = true;
        }
        if bindings_changed {
            self.motion_field_bind_group = Some(create_motion_field_bind_group(
                device,
                &self.motion_field_bind_group_layout,
                field,
                &motion.base_texture().view,
                controller.output_view(),
                controller.output_base_rows_buffer(),
            ));
        }
        let tagged_occurrences = sort_data.authored.tagged_occurrences;
        let registry_occurrences = self.authored_output_count;
        let affected_draws = sort_data
            .authored
            .authored_draws
            .iter()
            .filter(|&&authored| authored)
            .count();
        if registry_occurrences == 0 {
            self.authored_dispatch_key = None;
            return Ok(Some(AuthoredMotionDispatch {
                registry_occurrences,
                tagged_occurrences,
                affected_draws,
                membership_request_revision: registry_key.request_revision,
                registry_revision: registry_key.registry_revision,
                worker_tag_ms: sort_data.authored.tag_time_ms,
                registry_installed,
                ..AuthoredMotionDispatch::default()
            }));
        }
        let used_frames = local_frames
            .iter()
            .copied()
            .filter(|frame| compiled.slot_is_used(frame.slot))
            .collect::<Vec<_>>();
        let local_frames = used_frames.as_slice();
        let dispatch_key = AuthoredDispatchKey::new(
            motion.global_controller_frame(),
            local_frames,
            motion.motion_enabled(),
            motion.preview_channel_mask(),
            registry_key.registry_revision,
            compiled.revision(),
        );
        if self.authored_dispatch_key.as_ref() == Some(&dispatch_key) && !bindings_changed {
            return Ok(Some(AuthoredMotionDispatch {
                registry_occurrences,
                tagged_occurrences,
                affected_draws,
                membership_request_revision: registry_key.request_revision,
                registry_revision: registry_key.registry_revision,
                worker_tag_ms: sort_data.authored.tag_time_ms,
                registry_installed,
                ..AuthoredMotionDispatch::default()
            }));
        }
        let mut work = controller.dispatch(
            queue,
            encoder,
            motion,
            motion.global_controller_frame(),
            local_frames,
            motion.motion_enabled(),
            motion.preview_channel_mask(),
            timestamp_writes,
        )?;
        work.registry_occurrences = registry_occurrences;
        work.tagged_occurrences = tagged_occurrences;
        work.affected_draws = affected_draws;
        work.membership_request_revision = registry_key.request_revision;
        work.registry_revision = registry_key.registry_revision;
        work.worker_tag_ms = sort_data.authored.tag_time_ms;
        work.registry_installed = registry_installed || work.registry_upload_bytes > 0;
        self.authored_dispatch_key = Some(dispatch_key);
        self.authored_runtime_error = None;
        Ok(Some(work))
    }

    pub fn disable_authored_motion(&mut self, error: String) {
        self.authored_dispatch_key = None;
        self.authored_runtime_error = Some(error);
    }

    pub fn clear_authored_motion(&mut self) {
        self.authored_dispatch_key = None;
        self.authored_runtime_error = None;
    }
}

fn create_motion_field_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    field: &GpuMotionField,
    stable_gaussian_view: &wgpu::TextureView,
    authored_gaussian_view: &wgpu::TextureView,
    authored_base_rows_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("motion authoring field bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(field.continuous_view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(field.assignment_view()),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: field.uniform_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(stable_gaussian_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(authored_gaussian_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: authored_base_rows_buffer.as_entire_binding(),
            },
        ],
    })
}

#[derive(Default)]
pub(super) struct MotionGraphBuildOutcome {
    pub(super) graph: Option<Arc<MotionGraph>>,
    pub(super) error: Option<String>,
}

pub(super) fn build_motion_graph_outcome(motion: &MergedMotion) -> MotionGraphBuildOutcome {
    match build_motion_graph(motion, MotionGraphSettings::default()) {
        Ok(graph) => MotionGraphBuildOutcome {
            graph: Some(Arc::new(graph)),
            error: None,
        },
        Err(error) => MotionGraphBuildOutcome {
            graph: None,
            error: Some(error.to_string()),
        },
    }
}

pub(super) fn active_members(render_data_key: &RenderDataKey) -> Vec<(usize, usize)> {
    let mut members = Vec::with_capacity(render_data_key.tid.len() * 2);
    for (index, &(lod, tile)) in render_data_key.tid.iter().enumerate() {
        members.push((tile, lod));
        match render_data_key.transition_status.get(index) {
            Some(TileTransitionStatusHash::Changing(true)) => {
                if let Some(adjacent_lod) = lod.checked_add(1) {
                    members.push((tile, adjacent_lod));
                }
            }
            Some(TileTransitionStatusHash::Changing(false)) => {
                if let Some(adjacent_lod) = lod.checked_sub(1) {
                    members.push((tile, adjacent_lod));
                }
            }
            _ => {}
        }
    }
    members
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AuthoredRegistryKey {
    pub(super) request_revision: u64,
    pub(super) registry_revision: u64,
    pub(super) scene_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AuthoredDispatchKey {
    global: MotionControllerFrameKey,
    locals: Vec<MotionControllerFrameKey>,
    motion_enabled: bool,
    preview_channel_mask: u32,
    registry_revision: u64,
    field_revision: u64,
}

impl AuthoredDispatchKey {
    pub(super) fn new(
        global: MotionControllerFrame,
        locals: &[MotionControllerFrame],
        motion_enabled: bool,
        preview_channel_mask: u32,
        registry_revision: u64,
        field_revision: u64,
    ) -> Self {
        Self {
            global: MotionControllerFrameKey::from(global),
            locals: locals
                .iter()
                .copied()
                .map(MotionControllerFrameKey::from)
                .collect(),
            motion_enabled,
            preview_channel_mask,
            registry_revision,
            field_revision,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MotionControllerFrameKey {
    slot: u8,
    valid: bool,
    sample: [u32; 4],
    gains: [u32; 6],
}

impl From<MotionControllerFrame> for MotionControllerFrameKey {
    fn from(frame: MotionControllerFrame) -> Self {
        let sample = match frame.sample {
            TimelineSample::Source { u } => [0, u.to_bits(), 0, 0],
            TimelineSample::Blend {
                from_u,
                to_u,
                alpha,
            } => [1, from_u.to_bits(), to_u.to_bits(), alpha.to_bits()],
        };
        Self {
            slot: frame.slot,
            valid: frame.valid,
            sample,
            gains: [
                frame.gains.translation[0].to_bits(),
                frame.gains.translation[1].to_bits(),
                frame.gains.translation[2].to_bits(),
                frame.gains.rotation.to_bits(),
                frame.gains.scale.to_bits(),
                frame.gains.master.to_bits(),
            ],
        }
    }
}

pub(super) fn map_coord_and_offset(
    map_id: u32,
    user_data: &UserData,
    scene_data: &SceneData,
) -> Result<([u32; 2], [f32; 3]), String> {
    let map_height = u32::try_from(user_data.tile_map_wh.y)
        .map_err(|_| "tile-map height exceeds u32".to_string())?;
    let map_width = u32::try_from(user_data.tile_map_wh.x)
        .map_err(|_| "tile-map width exceeds u32".to_string())?;
    if map_height == 0 || map_width == 0 {
        return Err("tile-map dimensions must be positive".to_string());
    }
    let coord = [map_id / map_height, map_id % map_height];
    if coord[0] >= map_width {
        return Err("merged draw map ID exceeds the active tile map".to_string());
    }
    let world_tile = [
        i64::from(coord[0]) - user_data.tile_map_half_wh.x as i64
            + i64::from(scene_data.center_coord.x),
        i64::from(coord[1]) - user_data.tile_map_half_wh.y as i64
            + i64::from(scene_data.center_coord.y),
    ];
    Ok((
        coord,
        [
            world_tile[0] as f32 * user_data.tile_width,
            world_tile[1] as f32 * user_data.tile_width,
            0.0,
        ],
    ))
}

pub(super) fn draw_requires_motion_pipeline(
    authored_draw: bool,
    overlay: MotionOverlayChannel,
    preview_intersects: bool,
) -> bool {
    authored_draw || overlay != MotionOverlayChannel::Off || preview_intersects
}

fn circle_intersects_tile(preview: MotionBrushPreview, tile_origin: [f32; 2], width: f32) -> bool {
    let closest = [
        preview.center[0].clamp(tile_origin[0], tile_origin[0] + width),
        preview.center[1].clamp(tile_origin[1], tile_origin[1] + width),
    ];
    let delta = [
        preview.center[0] - closest[0],
        preview.center[1] - closest[1],
    ];
    delta[0] * delta[0] + delta[1] * delta[1] <= preview.radius * preview.radius
}

pub(super) fn preview_intersects_draw(
    preview: Option<MotionBrushPreview>,
    tile: &TileInstance,
    render_value: &Option<RenderDataValue>,
    user_data: &UserData,
    scene_data: &SceneData,
) -> bool {
    let Some(preview) = preview else {
        return false;
    };
    if let Some(value) = render_value {
        value.merge_from_vec.iter().any(|&map_id| {
            u32::try_from(map_id)
                .ok()
                .and_then(|id| map_coord_and_offset(id, user_data, scene_data).ok())
                .map(|(_, offset)| {
                    circle_intersects_tile(preview, [offset[0], offset[1]], user_data.tile_width)
                })
                .unwrap_or(true)
        })
    } else {
        circle_intersects_tile(
            preview,
            [tile.tile_offset.x, tile.tile_offset.y],
            user_data.tile_width,
        )
    }
}
