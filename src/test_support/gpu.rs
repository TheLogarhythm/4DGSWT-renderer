//! GPU allocation and synchronous readback for tests, never the production frame loop.

pub(crate) enum MissingGpu {
    Fail,
    Skip,
}

pub(crate) struct GpuTestContext {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}
impl GpuTestContext {
    /// Each caller owns a device; no shared mutable GPU state across parallel tests.
    /// Skip applies only to unavailable adapters, not device creation/validation errors.
    pub fn new(
        label: &str,
        missing: MissingGpu,
        power_preference: wgpu::PowerPreference,
        features: impl FnOnce(wgpu::Features) -> wgpu::Features,
    ) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                ..Default::default()
            })) {
                Ok(adapter) => adapter,
                Err(error) => match missing {
                    MissingGpu::Fail => panic!("{label}: GPU required ({error})"),
                    MissingGpu::Skip => {
                        eprintln!("{label}: skipped, no GPU adapter ({error})");
                        return None;
                    }
                },
            };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: features(adapter.features()),
            ..Default::default()
        }))
        .unwrap_or_else(|error| panic!("{label}: device creation failed ({error})"));
        Some(Self {
            adapter,
            device,
            queue,
        })
    }
}

fn map_readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap()
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let bytes = slice.get_mapped_range().to_vec();
    buffer.unmap();
    bytes
}

pub(crate) fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    offset: u64,
    size: u64,
) -> Vec<u8> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test buffer readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, offset, &readback, 0, size);
    queue.submit(Some(encoder.finish()));
    map_readback(device, &readback)
}

/// Reads an uncompressed 2D texture region, removing GPU row padding.
pub(crate) fn read_texture_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    dimensions: [u32; 2],
) -> Vec<u8> {
    assert_eq!(texture.format().block_dimensions(), (1, 1));
    assert!(dimensions[0] > 0 && dimensions[1] > 0);
    let bytes_per_texel = texture
        .format()
        .block_copy_size(None)
        .expect("copyable color texture");
    let row_bytes = dimensions[0] * bytes_per_texel;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row_bytes = row_bytes.div_ceil(alignment) * alignment;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test texture readback"),
        size: u64::from(padded_row_bytes) * u64::from(dimensions[1]),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(dimensions[1]),
            },
        },
        wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    let padded = map_readback(device, &readback);
    if row_bytes == padded_row_bytes {
        return padded;
    }
    let mut bytes = Vec::with_capacity(row_bytes as usize * dimensions[1] as usize);
    for row in padded.chunks_exact(padded_row_bytes as usize) {
        bytes.extend_from_slice(&row[..row_bytes as usize]);
    }
    bytes
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
impl RgbaFrame {
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        assert!(
            x < self.width as usize && y < self.height as usize,
            "pixel outside frame"
        );
        let offset = (y * self.width as usize + x) * 4;
        self.pixels[offset..offset + 3].try_into().unwrap()
    }
}

// Distinct rows/columns expose lost padding removal and incorrect row strides.
#[test]
fn texture_readback_preserves_rectangular_rows_and_channels() {
    let GpuTestContext { device, queue, .. } = GpuTestContext::new(
        "rectangular readback",
        MissingGpu::Fail,
        wgpu::PowerPreference::None,
        |_| wgpu::Features::empty(),
    )
    .unwrap();
    for (format, channels) in [
        (wgpu::TextureFormat::Rgba8Unorm, 4),
        (wgpu::TextureFormat::R8Uint, 1),
    ] {
        let size = wgpu::Extent3d {
            width: 65,
            height: 3,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("distinct rectangular rows"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let expected: Vec<u8> = (0..3)
            .flat_map(|y| {
                (0..65).flat_map(move |x| {
                    (0..channels).map(move |channel| (x + 65 * y + 17 * channel) as u8)
                })
            })
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &expected,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(65 * channels),
                rows_per_image: Some(3),
            },
            size,
        );
        assert_eq!(
            read_texture_bytes(&device, &queue, &texture, [65, 3]),
            expected,
            "{format:?} readback must preserve every row and channel without padding"
        );
    }
}
