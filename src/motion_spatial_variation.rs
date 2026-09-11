//! Stable world-space selection; evaluated when the field changes, never per Gaussian/frame.
pub const VARIANTS_PER_REGION: usize = 4;

pub fn spatial_variant(
    world: [f32; 2],
    tile_width: f32,
    variation: u8,
    coherence: u8,
    seed: u32,
    count: usize,
) -> usize {
    if variation == 0 || count <= 1 {
        return 0;
    }
    let patch = if coherence == 255 {
        [0_i32; 2]
    } else {
        let width = tile_width * 2.0_f32.powi((u32::from(coherence) * 5 / 255) as i32);
        [
            (world[0] / width).floor() as i32,
            (world[1] / width).floor() as i32,
        ]
    };
    let hash = mix(seed
        ^ (patch[0] as u32).wrapping_mul(0x9e3779b9)
        ^ (patch[1] as u32).wrapping_mul(0x85ebca6b));
    if variation != 255 && hash % 255 >= u32::from(variation) {
        return 0;
    }
    1 + (mix(hash ^ 0xc2b2ae35) as usize % (count - 1))
}

pub fn variant_seed(seed: u32, variant: usize) -> u32 {
    if variant == 0 {
        seed
    } else {
        mix(seed ^ (variant as u32).wrapping_mul(0x9e3779b9))
    }
}

fn mix(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x85ebca6b);
    value ^= value >> 13;
    value = value.wrapping_mul(0xc2b2ae35);
    value ^ (value >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variation_diverges_and_full_coherence_synchronizes_world_patches() {
        let low = (0..64)
            .map(|x| spatial_variant([x as f32 * 4.0, -3.0], 4.0, 255, 0, 17, 4))
            .collect::<Vec<_>>();
        assert!(low.iter().any(|&x| x != low[0]));
        let shared = spatial_variant([0.0, 0.0], 4.0, 255, 255, 17, 4);
        for x in -64..64 {
            assert_eq!(
                spatial_variant([x as f32 * 4.0, -30.0], 4.0, 255, 255, 17, 4),
                shared
            );
            assert_eq!(spatial_variant([x as f32 * 4.0, 0.0], 4.0, 0, 0, 17, 4), 0);
        }
        assert_eq!(
            spatial_variant([-0.1, -0.1], 4.0, 255, 0, 17, 4),
            spatial_variant([-3.9, -3.9], 4.0, 255, 0, 17, 4)
        );
    }
}
