//! A small, periodic artistic light cookie. Generated once, never ray traced.
//! Two drifting samples animate it; mip levels suppress shimmer on large splats.
use std::sync::OnceLock;

const SIZE: u32 = 256;
const CELLS: i32 = 8;

fn mip_data() -> &'static Vec<Vec<u8>> {
    static DATA: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    DATA.get_or_init(|| {
        let hash = |mut v: u32| {
            v ^= v >> 16;
            v = v.wrapping_mul(0x7feb352d);
            v ^= v >> 15;
            v = v.wrapping_mul(0x846ca68b);
            v ^= v >> 16;
            (v & 0xffff) as f32 / 65535.0
        };
        let mut texels = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let p = [
                    (x as f32 + 0.5) * CELLS as f32 / SIZE as f32,
                    (y as f32 + 0.5) * CELLS as f32 / SIZE as f32,
                ];
                let cell = [p[0].floor() as i32, p[1].floor() as i32];
                let mut nearest = [f32::MAX; 2];
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = [cell[0] + dx, cell[1] + dy];
                        let id = (c[0].rem_euclid(CELLS) + CELLS * c[1].rem_euclid(CELLS)) as u32;
                        let q = [
                            c[0] as f32 + 0.2 + 0.6 * hash(id + 19),
                            c[1] as f32 + 0.2 + 0.6 * hash(id + 173),
                        ];
                        let d = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                        if d < nearest[0] {
                            nearest[1] = nearest[0];
                            nearest[0] = d;
                        } else if d < nearest[1] {
                            nearest[1] = d;
                        }
                    }
                }
                let edge = (nearest[1].sqrt() - nearest[0].sqrt()) / 0.085;
                texels.push((255.0 * (-edge * edge).exp()).round() as u8);
            }
        }
        let mut levels = vec![texels];
        let mut width = SIZE as usize;
        while width > 1 {
            let source = levels.last().unwrap();
            let mut next = Vec::with_capacity(width * width / 4);
            for y in 0..width / 2 {
                for x in 0..width / 2 {
                    let i = 2 * y * width + 2 * x;
                    let sum = source[i] as u32
                        + source[i + 1] as u32
                        + source[i + width] as u32
                        + source[i + width + 1] as u32;
                    next.push(((sum + 2) / 4) as u8);
                }
            }
            levels.push(next);
            width /= 2;
        }
        levels
    })
}

pub(crate) fn texture(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) -> wgpu::Texture {
    let size = if enabled { SIZE } else { 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Artificial caustic cookie"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: size.ilog2() + 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if enabled {
        for (level, data) in mip_data().iter().enumerate() {
            let width = SIZE >> level;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: Some(width),
                },
                wgpu::Extent3d {
                    width,
                    height: width,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    texture
}
