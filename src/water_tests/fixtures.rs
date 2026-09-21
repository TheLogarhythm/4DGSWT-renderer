//! Scene data independent of GPU allocation.
use crate::{
    camera::Camera, motion::MergedMotion, scene::Scene, structure::TileBaseData, utils::*,
};

pub(super) struct SceneFixture {
    pub scene: Scene,
    pub base: TileBaseData,
    pub motion: Option<MergedMotion>,
}
impl SceneFixture {
    pub fn from_scene(mut scene: Scene, camera: &Camera) -> Self {
        let (aabb, mean) = scene.compute_aabb_and_center();
        let matrix = camera.view_proj();
        let view: &[f32; 16] = matrix.as_ref();
        let (gs_index, raw_depth) = scene.sort_self(view);
        let count = scene.splat_count;
        scene.generate_texture();
        Self {
            scene,
            motion: None,
            base: TileBaseData {
                splat_count: count,
                tile_center: mean,
                aabb,
                raw_depth,
                gs_index,
                gs_lod_id: vec![0; count],
            },
        }
    }
}

pub(super) fn two_splats(dynamic: bool) -> SceneFixture {
    let mut scene = Scene::new();
    scene.splat_count = 2;
    scene.tex_width = 2048;
    scene.tex_height = 1;
    scene.tex_data = vec![0; 2048 * 4];
    for (row, (position, color)) in [
        ([0.7f32, 0.0, -0.6], 0xff00ff00u32),
        ([-0.7, 0.0, 0.6], 0xff0000ff),
    ]
    .into_iter()
    .enumerate()
    {
        let out = &mut scene.tex_data[row * 8..row * 8 + 8];
        for i in 0..3 {
            out[i] = position[i].to_bits();
        }
        let cov = half::f16::from_f32(0.02).to_bits() as u32;
        // Unequal XY variances avoid the legacy eigensolver's zero-length axis for perfectly symmetric splats.
        out[4] = cov;
        out[5] = (half::f16::from_f32(0.03).to_bits() as u32) << 16;
        out[6] = cov << 16;
        out[7] = color;
    }
    let base = TileBaseData {
        splat_count: 2,
        tile_center: Vec3::zero(),
        aabb: (vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0)),
        raw_depth: vec![],
        gs_index: vec![0, 1],
        gs_lod_id: vec![0, 0],
    };
    let motion = dynamic.then(|| {
        use crate::{
            dynamic_archive::BasisBank,
            motion::{CanonicalGaussian, MergedMotion, MergedMotionData},
            scene_archive::DynamicArchiveSummary,
        };
        let mut values = vec![0.0; 75 * 9];
        for i in 0..75 {
            values[i * 9 + 2] = 0.3 * (std::f32::consts::TAU * i as f32 / 74.0).sin();
        }
        MergedMotion {
            data: MergedMotionData::Legacy {
                basis: BasisBank {
                    basis_count: 1,
                    values,
                },
                top_k: 1,
                basis_ids: vec![0, 0],
                weights: vec![1.0, 1.0],
            },
            canonical: [[0.7, 0.0, -0.6], [-0.7, 0.0, 0.6]]
                .into_iter()
                .map(|position| CanonicalGaussian {
                    position,
                    log_scale: [
                        0.005f32.sqrt().ln(),
                        0.0075f32.sqrt().ln(),
                        0.005f32.sqrt().ln(),
                    ],
                    rotation: [1.0, 0.0, 0.0, 0.0],
                })
                .collect(),
            transform_ids: vec![0, 0],
            motion_channel_masks: vec![7, 7],
            source_duration_seconds: 5.0,
            summary: DynamicArchiveSummary {
                schema_version: 1,
                tile_count: 1,
                lod_count: 1,
                basis_count: 1,
                top_k: 1,
                total_rows: 2,
                backend: "water-test".into(),
            },
            member_offsets: vec![vec![0]],
            member_counts: vec![vec![2]],
        }
    });
    SceneFixture {
        scene,
        base,
        motion,
    }
}
