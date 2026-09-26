//! GPU residency and batched uploads, independent of render-pass encoding.
use super::{TileUniforms, instance_buffer_capacity};
use crate::structure::{SortData, TileInstance, TileMergeStatus};

pub(super) const TILE_STRIDE: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrawSource {
    Presorted {
        splats: u32,
    },
    Streamed {
        offset: u64,
        bytes: u64,
        per_splat_lod: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrawComposition {
    Single,
    Merged,
}

impl DrawComposition {
    pub fn of(tile: &TileInstance) -> Self {
        if matches!(tile.merge_status, TileMergeStatus::MergedFrom(_)) {
            Self::Merged
        } else {
            Self::Single
        }
    }
}

pub(super) struct PreparedDraw {
    pub sort_index: usize,
    pub source: DrawSource,
    pub uniform: TileUniforms,
    pub motion: bool,
}

#[derive(Default, Debug)]
pub(super) struct UploadWork {
    pub bytes: u64,
    pub writes: usize,
}

#[derive(Default)]
struct ResidentDraws {
    revision: Option<(u32, u64, u64, u64)>,
    layout: Vec<(usize, DrawSource)>,
}

impl ResidentDraws {
    fn matches(&self, sort: &SortData, revision: u64, draws: &[PreparedDraw]) -> bool {
        self.revision
            == Some((
                sort.scene_id,
                revision,
                sort.authored.request_revision,
                sort.authored.registry_revision,
            ))
            && self
                .layout
                .iter()
                .copied()
                .eq(draws.iter().map(|d| (d.sort_index, d.source)))
    }

    fn record(&mut self, sort: &SortData, revision: u64, draws: &[PreparedDraw]) {
        self.revision = Some((
            sort.scene_id,
            revision,
            sort.authored.request_revision,
            sort.authored.registry_revision,
        ));
        self.layout.clear();
        self.layout
            .extend(draws.iter().map(|d| (d.sort_index, d.source)));
    }
}

pub(super) struct DrawResources {
    pub tile_buffer: wgpu::Buffer,
    pub tile_bind_group: wgpu::BindGroup,
    pub indices: wgpu::Buffer,
    pub map_ids: wgpu::Buffer,
    pub lod_ids: wgpu::Buffer,
    tile_layout: wgpu::BindGroupLayout,
    resident: ResidentDraws,
    uniforms: Vec<[u8; TILE_STRIDE as usize]>,
    staging: Vec<[u8; TILE_STRIDE as usize]>,
}

impl DrawResources {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let tile_buffer = Self::tile_buffer(device, TILE_STRIDE);
        let tile_bind_group = Self::tile_binding(device, layout, &tile_buffer);
        Self {
            tile_buffer,
            tile_bind_group,
            indices: Self::instance_buffer(device, "Streamed Gaussian indices", 4),
            map_ids: Self::instance_buffer(device, "Streamed map IDs", 4),
            lod_ids: Self::instance_buffer(device, "Streamed LoD IDs", 4),
            tile_layout: layout.clone(),
            resident: ResidentDraws::default(),
            uniforms: Vec::new(),
            staging: Vec::new(),
        }
    }

    fn instance_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | if cfg!(test) {
                    wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        })
    }

    fn tile_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched tile uniforms"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn tile_binding(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batched tile binding"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(TILE_STRIDE),
                }),
            }],
        })
    }

    pub fn invalidate(&mut self) {
        self.resident.revision = None;
        self.uniforms.clear();
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sort: &SortData,
        revision: u64,
        draws: &[PreparedDraw],
    ) -> Option<UploadWork> {
        let limit = device.limits().max_buffer_size;
        let uniform_bytes = (draws.len() as u64).checked_mul(TILE_STRIDE)?;
        let tile_capacity = instance_buffer_capacity(
            self.tile_buffer.size(),
            uniform_bytes,
            limit.min(u32::MAX as u64),
        )?;
        let required = draws.iter().try_fold(0u64, |required, draw| {
            let end = match draw.source {
                DrawSource::Streamed { offset, bytes, .. } => offset.checked_add(bytes)?,
                DrawSource::Presorted { splats } => u64::from(splats) * 4,
            };
            Some(required.max(end))
        })?;
        let capacity = instance_buffer_capacity(self.indices.size(), required, limit)?;
        let grew = capacity > self.indices.size();
        if grew {
            self.indices = Self::instance_buffer(device, "Streamed Gaussian indices", capacity);
            self.map_ids = Self::instance_buffer(device, "Streamed map IDs", capacity);
            self.lod_ids = Self::instance_buffer(device, "Streamed LoD IDs", capacity);
        }
        let tiles_grew = tile_capacity > self.tile_buffer.size();
        if tiles_grew {
            self.tile_buffer = Self::tile_buffer(device, tile_capacity);
            self.tile_bind_group = Self::tile_binding(device, &self.tile_layout, &self.tile_buffer);
        }
        let mut work = UploadWork::default();
        if grew || !self.resident.matches(sort, revision, draws) {
            for draw in draws {
                if let DrawSource::Streamed {
                    offset,
                    bytes,
                    per_splat_lod,
                } = draw.source
                {
                    let value = sort.render_data_vec[draw.sort_index].1.as_ref()?;
                    if bytes == 0 {
                        continue;
                    }
                    queue.write_buffer(
                        &self.indices,
                        offset,
                        bytemuck::cast_slice(&value.gs_index),
                    );
                    queue.write_buffer(
                        &self.map_ids,
                        offset,
                        bytemuck::cast_slice(&value.gs_map_id),
                    );
                    work.bytes += bytes * 2;
                    work.writes += 2;
                    if per_splat_lod {
                        queue.write_buffer(
                            &self.lod_ids,
                            offset,
                            bytemuck::cast_slice(value.gs_lod_id.as_ref()?),
                        );
                        work.bytes += bytes;
                        work.writes += 1;
                    }
                }
            }
            self.resident.record(sort, revision, draws);
        }
        self.staging.resize(draws.len(), [0; TILE_STRIDE as usize]);
        for (slot, draw) in self.staging.iter_mut().zip(draws) {
            let bytes = bytemuck::bytes_of(&draw.uniform);
            slot[..bytes.len()].copy_from_slice(bytes);
        }
        if !draws.is_empty() && (tiles_grew || self.staging != self.uniforms) {
            queue.write_buffer(&self.tile_buffer, 0, bytemuck::cast_slice(&self.staging));
            work.bytes += uniform_bytes;
            work.writes += 1;
        }
        std::mem::swap(&mut self.staging, &mut self.uniforms);
        Some(work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        motion_tagging::AuthoredSortMetadata,
        structure::{RenderDataKey, RenderDataValue},
        test_support::gpu::{GpuTestContext, MissingGpu, read_buffer},
    };

    #[test]
    fn draw_uploads_reuse_batches_and_invalidate_on_changes() {
        let GpuTestContext { device, queue, .. } = GpuTestContext::new(
            "draw residency",
            MissingGpu::Fail,
            wgpu::PowerPreference::None,
            |_| wgpu::Features::empty(),
        )
        .unwrap();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(TILE_STRIDE),
                },
                count: None,
            }],
        });
        let mut resources = DrawResources::new(&device, &layout);
        let mut sort = SortData {
            scene_id: 4,
            tile_instance_vec: vec![TileInstance::new(); 2],
            render_data_vec: (0..2)
                .map(|index| {
                    (
                        RenderDataKey::new(),
                        Some(RenderDataValue {
                            splat_count: 2,
                            gs_index: vec![index * 2, index * 2 + 1],
                            gs_map_id: vec![index; 2],
                            merge_from_vec: vec![index as usize],
                            single_lod_id: -1,
                            gs_lod_id: Some(vec![0, 1]),
                        }),
                    )
                })
                .collect(),
            authored: AuthoredSortMetadata::untagged(2),
        };
        let mut draws: Vec<_> = (0..2)
            .map(|i| PreparedDraw {
                sort_index: i,
                source: DrawSource::Streamed {
                    offset: i as u64 * 8,
                    bytes: 8,
                    per_splat_lod: true,
                },
                uniform: TileUniforms::from_tile(
                    &sort.tile_instance_vec[i],
                    &sort.render_data_vec[i].1,
                ),
                motion: false,
            })
            .collect();
        let initial = resources
            .prepare(&device, &queue, &sort, 1, &draws)
            .unwrap();
        assert_eq!(initial.writes, 7, "six index writes and one uniform batch");
        assert_eq!(initial.bytes, 48 + 2 * TILE_STRIDE);
        let repeat = resources
            .prepare(&device, &queue, &sort, 1, &draws)
            .unwrap();
        assert_eq!((repeat.writes, repeat.bytes), (0, 0));

        // Changing only the uniform does not resend splat streams.
        draws[0].uniform.authored_tag_mode = 1;
        let uniforms = resources
            .prepare(&device, &queue, &sort, 1, &draws)
            .unwrap();
        assert_eq!((uniforms.writes, uniforms.bytes), (1, 2 * TILE_STRIDE));
        // Replace equal-length streams and advance the renderer's revision.
        sort.render_data_vec[0].1.as_mut().unwrap().gs_index = vec![19, 23];
        let sorted = resources
            .prepare(&device, &queue, &sort, 2, &draws)
            .unwrap();
        assert_eq!((sorted.writes, sorted.bytes), (6, 48));
        let resident = read_buffer(&device, &queue, &resources.indices, 0, 16);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&resident), &[19, 23, 2, 3]);
        sort.authored.registry_revision += 1;
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &draws)
                .unwrap()
                .writes,
            6
        );
        sort.authored.request_revision += 1;
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &draws)
                .unwrap()
                .writes,
            6
        );

        // The second row becomes the first visible row, requiring compact offsets.
        draws.remove(0);
        draws[0].source = DrawSource::Streamed {
            offset: 0,
            bytes: 8,
            per_splat_lod: true,
        };
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &draws)
                .unwrap()
                .writes,
            4
        );
        let compact = read_buffer(&device, &queue, &resources.indices, 0, 8);
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&compact), &[2, 3]);
        let empty = resources.prepare(&device, &queue, &sort, 2, &[]).unwrap();
        assert_eq!((empty.writes, empty.bytes), (0, 0));
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &draws)
                .unwrap()
                .writes,
            4
        );
        resources.invalidate();
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &draws)
                .unwrap()
                .writes,
            4
        );

        // Growth must restore resident index contents, and >20k visible uniforms
        // must still use one write, with slots aligned for dynamic offsets.
        let large_count = 20_001;
        let mut large: Vec<_> = (0..large_count)
            .map(|_| PreparedDraw {
                sort_index: 0,
                source: DrawSource::Presorted { splats: 1_100_000 },
                uniform: draws[0].uniform,
                motion: false,
            })
            .collect();
        large.push(draws.pop().unwrap());
        let grown = resources
            .prepare(&device, &queue, &sort, 2, &large)
            .unwrap();
        assert_eq!(grown.writes, 4);
        assert_eq!(grown.bytes, 24 + large.len() as u64 * TILE_STRIDE);
        assert!(resources.map_ids.size() >= 1_100_000 * 4);
        assert!(resources.tile_buffer.size() >= large.len() as u64 * TILE_STRIDE);
        assert_eq!(
            resources
                .prepare(&device, &queue, &sort, 2, &large)
                .unwrap()
                .writes,
            0
        );
        queue.submit([]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
}
