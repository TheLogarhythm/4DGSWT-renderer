use std::fmt::{Display, Formatter};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::motion_brush::{
    DirtyRect, MotionBrushPreview, MotionFieldCache, MotionFieldLayout, MotionOverlayChannel,
};
use crate::motion_controller_palette::CompiledMotionField;

pub type MotionFieldOverlay = MotionOverlayChannel;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MotionFieldUniform {
    pub world_origin: [f32; 2],
    pub reciprocal_world_size: [f32; 2],
    pub texture_size: [u32; 2],
    pub active: u32,
    pub overlay_channel: u32,
    pub palette_count: u32,
    pub texels_per_tile: u32,
    pub storage_offset: [u32; 2],
    pub preview_center: [f32; 2],
    pub preview_radius: f32,
    pub preview_opacity: f32,
    pub preview_color: [f32; 4],
    pub preview_active: u32,
    pub preview_falloff: u32,
    pub preview_tool: u32,
    pub preview_padding: u32,
}

impl MotionFieldUniform {
    pub fn new(
        layout: MotionFieldLayout,
        storage_offset: [u32; 2],
        active: bool,
        overlay: MotionFieldOverlay,
        palette_count: u32,
    ) -> Self {
        let world_size = layout.world_size();
        Self {
            world_origin: layout.world_origin(),
            reciprocal_world_size: [1.0 / world_size[0], 1.0 / world_size[1]],
            texture_size: layout.texture_size(),
            active: u32::from(active),
            overlay_channel: overlay as u32,
            palette_count,
            texels_per_tile: layout.quality().texels_per_tile(),
            storage_offset,
            preview_center: [0.0; 2],
            preview_radius: 0.0,
            preview_opacity: 0.0,
            preview_color: [0.0; 4],
            preview_active: 0,
            preview_falloff: 0,
            preview_tool: 0,
            preview_padding: 0,
        }
    }

    pub fn with_preview(mut self, preview: Option<MotionBrushPreview>) -> Self {
        if let Some(preview) = preview {
            self.preview_center = preview.center;
            self.preview_radius = preview.radius;
            self.preview_opacity = preview.opacity;
            self.preview_color = preview.color;
            self.preview_active = 1;
            self.preview_falloff = preview.falloff.shader_code();
            self.preview_tool = preview.tool.preview_code();
        } else {
            self.preview_center = [0.0; 2];
            self.preview_radius = 0.0;
            self.preview_opacity = 0.0;
            self.preview_color = [0.0; 4];
            self.preview_active = 0;
            self.preview_falloff = 0;
            self.preview_tool = 0;
        }
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MotionFieldUpload {
    pub texel_count: u64,
    pub uploaded_bytes: u64,
}

pub struct GpuMotionField {
    continuous_texture: wgpu::Texture,
    continuous_view: wgpu::TextureView,
    assignment_texture: wgpu::Texture,
    assignment_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    uniform: MotionFieldUniform,
    dimensions: [u32; 2],
    allocated_texture_bytes: u64,
}

impl GpuMotionField {
    pub fn new_preview(device: &wgpu::Device, layout: MotionFieldLayout) -> Self {
        let dimensions = [1, 1];
        let extent = wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        };
        let continuous_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion preview placeholder continuous field"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let assignment_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion preview placeholder assignment field"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let continuous_view =
            continuous_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let assignment_view =
            assignment_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = MotionFieldUniform::new(layout, [0, 0], false, MotionFieldOverlay::Off, 0);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion preview uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            continuous_texture,
            continuous_view,
            assignment_texture,
            assignment_view,
            uniform_buffer,
            uniform,
            dimensions,
            allocated_texture_bytes: 5,
        }
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
    ) -> Result<Self, MotionFieldGpuError> {
        let dimensions = cache.layout().texture_size();
        let max_dimension = device.limits().max_texture_dimension_2d;
        if dimensions
            .into_iter()
            .any(|dimension| dimension > max_dimension)
        {
            return Err(MotionFieldGpuError::new(format!(
                "motion field {:?} exceeds max_texture_dimension_2d {}",
                dimensions, max_dimension
            )));
        }
        let extent = wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let continuous_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion authoring continuous field"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        });
        let assignment_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion authoring assignment field"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage,
            view_formats: &[],
        });
        let continuous_view =
            continuous_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let assignment_view =
            assignment_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = MotionFieldUniform::new(
            cache.layout(),
            cache.storage_offset(),
            true,
            MotionFieldOverlay::Off,
            0,
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion authoring field uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let field = Self {
            continuous_texture,
            continuous_view,
            assignment_texture,
            assignment_view,
            uniform_buffer,
            uniform,
            dimensions,
            allocated_texture_bytes: cache.layout().estimated_gpu_bytes(),
        };
        field.upload_full(queue, cache)?;
        Ok(field)
    }

    pub fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    pub fn continuous_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba8Unorm
    }

    pub fn assignment_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::R8Uint
    }

    pub fn allocated_texture_bytes(&self) -> u64 {
        self.allocated_texture_bytes
    }

    pub fn continuous_view(&self) -> &wgpu::TextureView {
        &self.continuous_view
    }

    pub fn assignment_view(&self) -> &wgpu::TextureView {
        &self.assignment_view
    }

    pub fn uniform_buffer(&self) -> &wgpu::Buffer {
        &self.uniform_buffer
    }

    pub fn update_uniform(&mut self, queue: &wgpu::Queue, uniform: MotionFieldUniform) -> bool {
        if self.uniform == uniform {
            return false;
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        self.uniform = uniform;
        true
    }

    pub fn upload_full(
        &self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
    ) -> Result<MotionFieldUpload, MotionFieldGpuError> {
        self.upload_dirty(
            queue,
            cache,
            DirtyRect::from_bounds([0, 0], self.dimensions)
                .expect("motion field dimensions are validated as positive"),
        )
    }

    pub fn upload_dirty(
        &self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        rect: DirtyRect,
    ) -> Result<MotionFieldUpload, MotionFieldGpuError> {
        self.upload_dirty_sources(queue, cache, cache.assignments(), rect)
    }

    pub fn upload_full_compiled(
        &self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        compiled: &CompiledMotionField,
    ) -> Result<MotionFieldUpload, MotionFieldGpuError> {
        self.upload_dirty_compiled(
            queue,
            cache,
            compiled,
            DirtyRect::from_bounds([0, 0], self.dimensions)
                .expect("motion field dimensions are validated as positive"),
        )
    }

    pub fn upload_dirty_compiled(
        &self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        compiled: &CompiledMotionField,
        rect: DirtyRect,
    ) -> Result<MotionFieldUpload, MotionFieldGpuError> {
        self.upload_dirty_sources(queue, cache, compiled.assignments(), rect)
    }

    fn upload_dirty_sources(
        &self,
        queue: &wgpu::Queue,
        cache: &MotionFieldCache,
        assignment_bytes: &[u8],
        rect: DirtyRect,
    ) -> Result<MotionFieldUpload, MotionFieldGpuError> {
        if cache.layout().texture_size() != self.dimensions {
            return Err(MotionFieldGpuError::new(
                "motion field CPU and GPU dimensions do not match",
            ));
        }
        let expected_assignments =
            usize::try_from(u64::from(self.dimensions[0]) * u64::from(self.dimensions[1]))
                .map_err(|_| MotionFieldGpuError::new("motion field assignment size overflow"))?;
        if assignment_bytes.len() != expected_assignments {
            return Err(MotionFieldGpuError::new(
                "compiled motion field dimensions do not match the GPU cache",
            ));
        }
        let logical_min = rect.min();
        let logical_max = rect.max_exclusive();
        if logical_max[0] > self.dimensions[0] || logical_max[1] > self.dimensions[1] {
            return Err(MotionFieldGpuError::new(
                "motion field dirty rectangle exceeds the GPU cache",
            ));
        }
        let mut texel_count = 0_u64;
        for physical in cache.physical_regions(rect) {
            let min = physical.min();
            let max = physical.max_exclusive();
            let width = max[0] - min[0];
            let height = max[1] - min[1];
            let continuous =
                pack_region(cache.continuous_bytes(), self.dimensions[0], physical, 4)?;
            let assignments = pack_region(assignment_bytes, self.dimensions[0], physical, 1)?;
            write_region(
                queue,
                &self.continuous_texture,
                min,
                [width, height],
                &continuous,
                4,
            );
            write_region(
                queue,
                &self.assignment_texture,
                min,
                [width, height],
                &assignments,
                1,
            );
            texel_count = texel_count.saturating_add(u64::from(width) * u64::from(height));
        }
        debug_assert_eq!(
            texel_count,
            u64::from(logical_max[0] - logical_min[0]) * u64::from(logical_max[1] - logical_min[1])
        );
        Ok(MotionFieldUpload {
            texel_count,
            uploaded_bytes: texel_count * 5,
        })
    }
}

fn pack_region(
    source: &[u8],
    source_width: u32,
    rect: DirtyRect,
    bytes_per_texel: usize,
) -> Result<Vec<u8>, MotionFieldGpuError> {
    let min = rect.min();
    let max = rect.max_exclusive();
    let copy_width = usize::try_from(max[0] - min[0])
        .map_err(|_| MotionFieldGpuError::new("motion field copy width overflow"))?;
    let copy_height = usize::try_from(max[1] - min[1])
        .map_err(|_| MotionFieldGpuError::new("motion field copy height overflow"))?;
    let row_bytes = copy_width
        .checked_mul(bytes_per_texel)
        .ok_or_else(|| MotionFieldGpuError::new("motion field copy row overflow"))?;
    let mut packed = Vec::with_capacity(
        row_bytes
            .checked_mul(copy_height)
            .ok_or_else(|| MotionFieldGpuError::new("motion field copy size overflow"))?,
    );
    let source_width = usize::try_from(source_width)
        .map_err(|_| MotionFieldGpuError::new("motion field source width overflow"))?;
    for y in min[1]..max[1] {
        let texel_start = usize::try_from(y)
            .ok()
            .and_then(|row| row.checked_mul(source_width))
            .and_then(|row| row.checked_add(min[0] as usize))
            .ok_or_else(|| MotionFieldGpuError::new("motion field row offset overflow"))?;
        let byte_start = texel_start
            .checked_mul(bytes_per_texel)
            .ok_or_else(|| MotionFieldGpuError::new("motion field byte offset overflow"))?;
        let byte_end = byte_start
            .checked_add(row_bytes)
            .ok_or_else(|| MotionFieldGpuError::new("motion field row end overflow"))?;
        let row = source
            .get(byte_start..byte_end)
            .ok_or_else(|| MotionFieldGpuError::new("motion field source is truncated"))?;
        packed.extend_from_slice(row);
    }
    Ok(packed)
}

fn write_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: [u32; 2],
    size: [u32; 2],
    bytes: &[u8],
    bytes_per_texel: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: origin[0],
                y: origin[1],
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size[0] * bytes_per_texel),
            rows_per_image: Some(size[1]),
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MotionFieldGpuError {
    message: String,
}

impl MotionFieldGpuError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for MotionFieldGpuError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MotionFieldGpuError {}

#[cfg(test)]
mod tests {
    #[cfg(not(target_arch = "wasm32"))]
    use crate::test_support::gpu::{GpuTestContext, MissingGpu, read_texture_bytes};

    use super::*;
    use crate::motion_brush::{
        DirtyRect, MotionBrushDocument, MotionBrushFalloff, MotionBrushPreview, MotionBrushTool,
        MotionFieldCache, MotionFieldLayout, MotionFieldQuality,
    };
    use crate::motion_controller_palette::{
        CompiledMotionField, MotionControllerPalette, MotionRegionDocument, MotionRegionId,
    };

    fn default_layout() -> MotionFieldLayout {
        MotionFieldLayout::new([97, 97], 4.0, [0, 0], MotionFieldQuality::Default).unwrap()
    }

    #[test]
    fn field_uniform_is_stable_and_uses_absolute_world_bounds() {
        let uniform = MotionFieldUniform::new(
            default_layout(),
            [0, 0],
            true,
            MotionFieldOverlay::Variation,
            7,
        );

        assert_eq!(std::mem::size_of::<MotionFieldUniform>(), 96);
        assert_eq!(uniform.world_origin, [-192.0, -192.0]);
        assert_eq!(uniform.reciprocal_world_size, [1.0 / 388.0, 1.0 / 388.0]);
        assert_eq!(uniform.texture_size, [1552, 1552]);
        assert_eq!(uniform.active, 1);
        assert_eq!(
            uniform.overlay_channel,
            MotionFieldOverlay::Variation as u32
        );
        assert_eq!(uniform.palette_count, 7);
        assert_eq!(uniform.storage_offset, [0, 0]);
    }

    #[test]
    fn field_uniform_carries_the_shared_brush_preview_contract() {
        let preview = MotionBrushPreview {
            center: [3.5, -2.0],
            radius: 4.25,
            opacity: 0.6,
            falloff: MotionBrushFalloff::Smooth,
            tool: MotionBrushTool::AdjustCoherence { target: 0.8 },
            color: [0.16, 0.90, 0.46, 1.0],
        };
        let uniform =
            MotionFieldUniform::new(default_layout(), [0, 0], true, MotionFieldOverlay::Off, 0)
                .with_preview(Some(preview));

        assert_eq!(uniform.preview_active, 1);
        assert_eq!(uniform.preview_center, [3.5, -2.0]);
        assert_eq!(uniform.preview_radius, 4.25);
        assert_eq!(uniform.preview_opacity, 0.6);
        assert_eq!(uniform.preview_color, [0.16, 0.90, 0.46, 1.0]);
        assert_eq!(uniform.preview_falloff, 2);
        assert_eq!(uniform.preview_tool, 4);

        let inactive = uniform.with_preview(None);
        assert_eq!(inactive.preview_active, 0);
        assert_eq!(inactive.preview_radius, 0.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn preview_placeholder_uses_one_texel_without_field_data() {
        let Some(GpuTestContext {
            device,
            queue: _queue,
            ..
        }) = GpuTestContext::new(
            "motion preview test device",
            MissingGpu::Skip,
            wgpu::PowerPreference::LowPower,
            |_| wgpu::Features::empty(),
        )
        else {
            return;
        };

        let field = GpuMotionField::new_preview(&device, default_layout());

        assert_eq!(field.dimensions(), [1, 1]);
        assert_eq!(field.allocated_texture_bytes(), 5);
        assert_eq!(field.uniform.active, 0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_field_uses_exact_dimensions_formats_and_dirty_upload_size() {
        let Some(GpuTestContext { device, queue, .. }) = GpuTestContext::new(
            "motion field test device",
            MissingGpu::Skip,
            wgpu::PowerPreference::LowPower,
            |_| wgpu::Features::empty(),
        ) else {
            return;
        };

        let cache = MotionFieldCache::new(default_layout()).unwrap();
        let field = GpuMotionField::new(&device, &queue, &cache).unwrap();

        assert_eq!(field.dimensions(), [1552, 1552]);
        assert_eq!(field.continuous_format(), wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(field.assignment_format(), wgpu::TextureFormat::R8Uint);
        assert_eq!(field.allocated_texture_bytes(), 1552_u64 * 1552 * 5);

        let rect = DirtyRect::from_bounds([10, 20], [14, 23]).unwrap();
        let upload = field.upload_dirty(&queue, &cache, rect).unwrap();
        assert_eq!(upload.texel_count, 12);
        assert_eq!(upload.uploaded_bytes, 60);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_ring_cache_matches_cpu_physical_storage_after_recenter() {
        let Some(GpuTestContext { device, queue, .. }) = GpuTestContext::new(
            "motion field ring test device",
            MissingGpu::Skip,
            wgpu::PowerPreference::LowPower,
            |_| wgpu::Features::empty(),
        ) else {
            return;
        };

        let first = MotionFieldLayout::new([2, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let moved = MotionFieldLayout::new([2, 1], 4.0, [1, 0], MotionFieldQuality::Low).unwrap();
        let mut document = MotionBrushDocument::default();
        for (point, slot) in [([1.0, 2.0], 2), ([5.0, 2.0], 3)] {
            document
                .push_stroke(
                    MotionBrushTool::ApplyBehavior {
                        region_id: MotionRegionId::local(slot).unwrap(),
                    },
                    vec![point],
                    0.6,
                    1.0,
                    0.25,
                    MotionBrushFalloff::Constant,
                )
                .unwrap();
        }
        let mut cache = MotionFieldCache::rebuild(first, &document).unwrap();
        let mut field = GpuMotionField::new(&device, &queue, &cache).unwrap();
        let dirty = cache.shift_window(moved, &document).unwrap();
        for rect in dirty {
            field.upload_dirty(&queue, &cache, rect).unwrap();
        }
        field.update_uniform(
            &queue,
            MotionFieldUniform::new(
                cache.layout(),
                cache.storage_offset(),
                true,
                MotionFieldOverlay::Off,
                0,
            ),
        );

        assert_eq!(field.uniform.storage_offset, [8, 0]);
        assert_eq!(
            read_texture_bytes(&device, &queue, &field.continuous_texture, [16, 8]),
            cache.continuous_bytes()
        );
        assert_eq!(
            read_texture_bytes(&device, &queue, &field.assignment_texture, [16, 8]),
            cache.assignments()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_field_uploads_compiled_slots_without_mutating_stable_region_ids() {
        let Some(GpuTestContext { device, queue, .. }) = GpuTestContext::new(
            "compiled motion field test device",
            MissingGpu::Skip,
            wgpu::PowerPreference::LowPower,
            |_| wgpu::Features::empty(),
        ) else {
            return;
        };

        let layout = MotionFieldLayout::new([1, 1], 4.0, [0, 0], MotionFieldQuality::Low).unwrap();
        let calm = MotionRegionId::local(4).unwrap();
        let mut document = MotionBrushDocument::default();
        document
            .push_stroke(
                MotionBrushTool::ApplyBehavior { region_id: calm },
                vec![[2.0, 2.0]],
                4.0,
                1.0,
                0.25,
                MotionBrushFalloff::Constant,
            )
            .unwrap();
        let cache = MotionFieldCache::rebuild(layout, &document).unwrap();
        let mut regions = MotionRegionDocument::artist_defaults();
        for disabled in 1..=3 {
            regions
                .select(MotionRegionId::local(disabled).unwrap())
                .unwrap();
            regions.set_selected_enabled(false);
        }
        let mut palette = MotionControllerPalette::new(2.0).unwrap();
        palette.sync_root_regions(&regions).unwrap();
        let compiled =
            CompiledMotionField::compile_roots(&cache, &regions, &palette, None).unwrap();
        let field = GpuMotionField::new(&device, &queue, &cache).unwrap();

        field
            .upload_full_compiled(&queue, &cache, &compiled)
            .unwrap();

        assert!(cache.assignments().contains(&4));
        assert!(compiled.assignments().contains(&1));
        let uploaded = read_texture_bytes(
            &device,
            &queue,
            &field.assignment_texture,
            layout.texture_size(),
        );
        assert!(uploaded.contains(&1));
        assert!(!uploaded.contains(&4));
    }
}
