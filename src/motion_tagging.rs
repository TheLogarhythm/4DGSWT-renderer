use std::{collections::HashMap, sync::Arc};

pub const AUTHORED_TAG_BIT: u32 = 1 << 31;
pub const AUTHORED_INDEX_MASK: u32 = AUTHORED_TAG_BIT - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderIndex {
    Base(u32),
    Authored(u32),
}

pub fn encode_authored_index(output_index: u32) -> Result<u32, String> {
    if output_index & AUTHORED_TAG_BIT != 0 {
        return Err("authored output index uses the reserved tag bit".to_string());
    }
    Ok(AUTHORED_TAG_BIT | output_index)
}

pub const fn decode_render_index(raw: u32) -> RenderIndex {
    if raw & AUTHORED_TAG_BIT == 0 {
        RenderIndex::Base(raw)
    } else {
        RenderIndex::Authored(raw & AUTHORED_INDEX_MASK)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotionMembershipSnapshot {
    request_revision: u64,
    membership_revision: u64,
    dimensions: [u32; 2],
    map_tiles: [u32; 2],
    world_origin: [f32; 2],
    texel_world_size: [f32; 2],
    storage_offset: [u32; 2],
    active_texels: Arc<[u8]>,
    active_tiles: Arc<[u8]>,
}

impl MotionMembershipSnapshot {
    pub(crate) fn new(
        request_revision: u64,
        membership_revision: u64,
        dimensions: [u32; 2],
        map_tiles: [u32; 2],
        world_origin: [f32; 2],
        texel_world_size: [f32; 2],
        storage_offset: [u32; 2],
        active_texels: Arc<[u8]>,
        active_tiles: Arc<[u8]>,
    ) -> Self {
        Self {
            request_revision,
            membership_revision,
            dimensions,
            map_tiles,
            world_origin,
            texel_world_size,
            storage_offset,
            active_texels,
            active_tiles,
        }
    }

    pub const fn request_revision(&self) -> u64 {
        self.request_revision
    }

    pub const fn membership_revision(&self) -> u64 {
        self.membership_revision
    }

    pub fn active_at_world(&self, world_xy: [f32; 2]) -> bool {
        physical_texel_index(
            world_xy,
            self.world_origin,
            self.texel_world_size,
            self.dimensions,
            self.storage_offset,
        )
        .and_then(|index| self.active_texels.get(index))
        .copied()
        .unwrap_or(0)
            != 0
    }

    pub fn tile_has_authored(&self, map_coord: [u32; 2]) -> bool {
        if map_coord[0] >= self.map_tiles[0] || map_coord[1] >= self.map_tiles[1] {
            return false;
        }
        let index = usize::try_from(
            u64::from(map_coord[1]) * u64::from(self.map_tiles[0]) + u64::from(map_coord[0]),
        )
        .ok();
        index
            .and_then(|index| self.active_tiles.get(index))
            .copied()
            .unwrap_or(0)
            != 0
    }

    pub fn has_authored(&self) -> bool {
        self.active_tiles.iter().any(|&active| active != 0)
    }
}

#[derive(Clone, Debug)]
pub struct AuthoredTagRequest {
    pub request_revision: u64,
    pub snapshot: Option<Arc<MotionMembershipSnapshot>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AuthoredOccurrenceInput {
    pub global_row: u32,
    pub output_index: u32,
    pub occurrence_offset: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthoredRegistryUpdate {
    pub request_revision: u64,
    pub registry_revision: u64,
    pub scene_id: u32,
    pub inputs: Arc<[AuthoredOccurrenceInput]>,
    pub output_base_rows: Arc<[u32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuthoredSortMetadata {
    pub request_revision: u64,
    pub registry_revision: u64,
    pub update: Option<Arc<AuthoredRegistryUpdate>>,
    pub authored_draws: Vec<bool>,
    pub tagged_occurrences: usize,
    pub tag_time_ms: f64,
}

impl AuthoredSortMetadata {
    pub fn untagged(draw_count: usize) -> Self {
        Self {
            request_revision: 0,
            registry_revision: 0,
            update: None,
            authored_draws: vec![false; draw_count],
            tagged_occurrences: 0,
            tag_time_ms: 0.0,
        }
    }

    pub fn validate_draw_alignment(&self, draw_count: usize) -> Result<(), String> {
        if self.authored_draws.len() != draw_count {
            return Err(format!(
                "authored draw metadata has {} entries for {draw_count} draws",
                self.authored_draws.len()
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct AuthoredOccurrenceRegistry {
    scene_id: u32,
    request_revision: u64,
    snapshot: Option<Arc<MotionMembershipSnapshot>>,
    indices: HashMap<u64, u32>,
    inputs: Vec<AuthoredOccurrenceInput>,
    output_base_rows: Vec<u32>,
    registry_revision: u64,
    changed: bool,
}

impl AuthoredOccurrenceRegistry {
    pub const fn request_revision(&self) -> u64 {
        self.request_revision
    }

    pub const fn scene_id(&self) -> u32 {
        self.scene_id
    }

    pub fn current_request(&self) -> AuthoredTagRequest {
        AuthoredTagRequest {
            request_revision: self.request_revision,
            snapshot: self.snapshot.clone(),
        }
    }

    pub fn tile_has_authored(&self, map_coord: [u32; 2]) -> bool {
        self.snapshot
            .as_ref()
            .is_some_and(|snapshot| snapshot.tile_has_authored(map_coord))
    }

    pub fn reset(&mut self, scene_id: u32, request: AuthoredTagRequest) -> Result<(), String> {
        if let Some(snapshot) = &request.snapshot
            && snapshot.request_revision() != request.request_revision
        {
            return Err("authored request and membership snapshot revisions differ".to_string());
        }
        self.scene_id = scene_id;
        self.request_revision = request.request_revision;
        self.snapshot = request.snapshot;
        self.indices.clear();
        self.inputs.clear();
        self.output_base_rows.clear();
        self.changed = true;
        Ok(())
    }

    pub fn tag_occurrence(
        &mut self,
        map_id: u32,
        map_coord: [u32; 2],
        global_row: u32,
        occurrence_offset: [f32; 3],
        canonical_xy: &[[f32; 2]],
    ) -> Result<Option<u32>, String> {
        let Some(snapshot) = &self.snapshot else {
            return Ok(None);
        };
        if !snapshot.tile_has_authored(map_coord) {
            return Ok(None);
        }
        if global_row & AUTHORED_TAG_BIT != 0 {
            return Err("base Gaussian row uses the reserved authored tag bit".to_string());
        }
        if occurrence_offset.iter().any(|value| !value.is_finite()) {
            return Err("authored occurrence offset must be finite".to_string());
        }
        let canonical = canonical_xy
            .get(usize::try_from(global_row).map_err(|_| {
                "base Gaussian row cannot be represented on this platform".to_string()
            })?)
            .ok_or_else(|| "authored occurrence base row is out of range".to_string())?;
        if canonical.iter().any(|value| !value.is_finite()) {
            return Err("authored occurrence canonical position must be finite".to_string());
        }
        let world_xy = [
            canonical[0] + occurrence_offset[0],
            canonical[1] + occurrence_offset[1],
        ];
        if !snapshot.active_at_world(world_xy) {
            return Ok(None);
        }

        let key = (u64::from(map_id) << 32) | u64::from(global_row);
        if let Some(&output_index) = self.indices.get(&key) {
            return encode_authored_index(output_index).map(Some);
        }
        let output_index = u32::try_from(self.inputs.len())
            .map_err(|_| "authored occurrence count exceeds u32".to_string())?;
        let encoded = encode_authored_index(output_index)?;
        self.indices.insert(key, output_index);
        self.inputs.push(AuthoredOccurrenceInput {
            global_row,
            output_index,
            occurrence_offset,
        });
        self.output_base_rows.push(global_row);
        self.changed = true;
        Ok(Some(encoded))
    }

    pub fn finish_sort(
        &mut self,
        authored_draws: Vec<bool>,
        tagged_occurrences: usize,
        tag_time_ms: f64,
    ) -> Result<AuthoredSortMetadata, String> {
        if !tag_time_ms.is_finite() || tag_time_ms < 0.0 {
            return Err("authored tag duration must be finite and nonnegative".to_string());
        }
        let update = if self.changed {
            self.registry_revision = self.registry_revision.wrapping_add(1);
            self.changed = false;
            Some(Arc::new(AuthoredRegistryUpdate {
                request_revision: self.request_revision,
                registry_revision: self.registry_revision,
                scene_id: self.scene_id,
                inputs: Arc::from(self.inputs.clone()),
                output_base_rows: Arc::from(self.output_base_rows.clone()),
            }))
        } else {
            None
        };
        Ok(AuthoredSortMetadata {
            request_revision: self.request_revision,
            registry_revision: self.registry_revision,
            update,
            authored_draws,
            tagged_occurrences,
            tag_time_ms,
        })
    }
}

pub(crate) fn physical_texel_index(
    world_xy: [f32; 2],
    world_origin: [f32; 2],
    texel_world_size: [f32; 2],
    dimensions: [u32; 2],
    storage_offset: [u32; 2],
) -> Option<usize> {
    if world_xy.iter().any(|value| !value.is_finite())
        || texel_world_size
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || dimensions.contains(&0)
    {
        return None;
    }
    let world_size = [
        texel_world_size[0] * dimensions[0] as f32,
        texel_world_size[1] * dimensions[1] as f32,
    ];
    if world_xy[0] < world_origin[0]
        || world_xy[1] < world_origin[1]
        || world_xy[0] >= world_origin[0] + world_size[0]
        || world_xy[1] >= world_origin[1] + world_size[1]
    {
        return None;
    }
    let logical_x = ((world_xy[0] - world_origin[0]) / texel_world_size[0]).floor() as u32;
    let logical_y = ((world_xy[1] - world_origin[1]) / texel_world_size[1]).floor() as u32;
    let physical_x = (logical_x + storage_offset[0]) % dimensions[0];
    let physical_y = (logical_y + storage_offset[1]) % dimensions[1];
    usize::try_from(u64::from(physical_y) * u64::from(dimensions[0]) + u64::from(physical_x)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_index_tag_round_trips_without_changing_base_indices() {
        assert_eq!(encode_authored_index(7).unwrap(), 0x8000_0007);
        assert_eq!(decode_render_index(19), RenderIndex::Base(19));
        assert_eq!(decode_render_index(0x8000_0007), RenderIndex::Authored(7));
    }

    #[test]
    fn authored_index_tag_rejects_values_that_use_the_reserved_bit() {
        assert!(encode_authored_index(AUTHORED_TAG_BIT).is_err());
        assert!(encode_authored_index(u32::MAX).is_err());
    }

    fn two_tile_membership() -> Arc<MotionMembershipSnapshot> {
        Arc::new(MotionMembershipSnapshot::new(
            4,
            9,
            [2, 1],
            [2, 1],
            [0.0, 0.0],
            [1.0, 1.0],
            [0, 0],
            Arc::from([1_u8, 0]),
            Arc::from([1_u8, 0]),
        ))
    }

    #[test]
    fn authored_occurrence_registry_keeps_indices_stable_across_sort_order() {
        let request = AuthoredTagRequest {
            request_revision: 4,
            snapshot: Some(two_tile_membership()),
        };
        let canonical = [[0.25, 0.25], [0.75, 0.25]];
        let mut registry = AuthoredOccurrenceRegistry::default();
        registry.reset(7, request).unwrap();

        let first = registry
            .tag_occurrence(12, [0, 0], 0, [0.0, 0.0, 0.0], &canonical)
            .unwrap();
        let second = registry
            .tag_occurrence(12, [0, 0], 1, [0.0, 0.0, 0.0], &canonical)
            .unwrap();
        assert_eq!(
            decode_render_index(first.unwrap()),
            RenderIndex::Authored(0)
        );
        assert_eq!(
            decode_render_index(second.unwrap()),
            RenderIndex::Authored(1)
        );

        let second_again = registry
            .tag_occurrence(12, [0, 0], 1, [0.0, 0.0, 0.0], &canonical)
            .unwrap();
        let first_again = registry
            .tag_occurrence(12, [0, 0], 0, [0.0, 0.0, 0.0], &canonical)
            .unwrap();
        assert_eq!(second_again, second);
        assert_eq!(first_again, first);

        let repeated = registry
            .tag_occurrence(13, [0, 0], 0, [0.0, 0.0, 0.0], &canonical)
            .unwrap();
        assert_eq!(
            decode_render_index(repeated.unwrap()),
            RenderIndex::Authored(2)
        );
    }

    #[test]
    fn authored_occurrence_registry_rejects_uncovered_and_invalid_rows_without_mutation() {
        let request = AuthoredTagRequest {
            request_revision: 4,
            snapshot: Some(two_tile_membership()),
        };
        let mut registry = AuthoredOccurrenceRegistry::default();
        registry.reset(7, request).unwrap();

        assert_eq!(
            registry
                .tag_occurrence(50, [1, 0], u32::MAX, [0.0; 3], &[])
                .unwrap(),
            None
        );
        assert!(
            registry
                .tag_occurrence(50, [0, 0], 3, [0.0; 3], &[[0.25, 0.25]])
                .is_err()
        );
        let metadata = registry.finish_sort(vec![false], 0, 0.25).unwrap();
        let update = metadata.update.unwrap();
        assert!(update.inputs.is_empty());
        assert!(update.output_base_rows.is_empty());
    }

    #[test]
    fn authored_occurrence_registry_finishes_one_atomic_update_per_change() {
        let request = AuthoredTagRequest {
            request_revision: 4,
            snapshot: Some(two_tile_membership()),
        };
        let canonical = [[0.25, 0.25]];
        let mut registry = AuthoredOccurrenceRegistry::default();
        registry.reset(7, request).unwrap();
        registry
            .tag_occurrence(12, [0, 0], 0, [0.0, 0.0, 4.0], &canonical)
            .unwrap();

        let first = registry.finish_sort(vec![true, false], 1, 0.5).unwrap();
        assert_eq!(first.request_revision, 4);
        assert_eq!(first.registry_revision, 1);
        assert_eq!(first.authored_draws, vec![true, false]);
        assert_eq!(first.tagged_occurrences, 1);
        let update = first.update.unwrap();
        assert_eq!(update.scene_id, 7);
        assert_eq!(update.inputs[0].global_row, 0);
        assert_eq!(update.inputs[0].output_index, 0);
        assert_eq!(update.inputs[0].occurrence_offset, [0.0, 0.0, 4.0]);
        assert_eq!(&*update.output_base_rows, &[0]);

        let unchanged = registry.finish_sort(vec![true], 1, 0.1).unwrap();
        assert_eq!(unchanged.registry_revision, 1);
        assert!(unchanged.update.is_none());

        registry
            .reset(
                8,
                AuthoredTagRequest {
                    request_revision: 5,
                    snapshot: None,
                },
            )
            .unwrap();
        let cleared = registry.finish_sort(vec![false], 0, 0.0).unwrap();
        assert_eq!(cleared.registry_revision, 2);
        assert!(cleared.update.unwrap().inputs.is_empty());
    }

    #[test]
    fn authored_sort_metadata_rejects_draw_misalignment() {
        let metadata = AuthoredSortMetadata::untagged(2);
        assert!(metadata.validate_draw_alignment(2).is_ok());
        assert!(metadata.validate_draw_alignment(1).is_err());
    }
}
