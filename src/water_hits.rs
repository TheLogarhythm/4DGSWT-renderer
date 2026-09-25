//! Viewport-sized hits and optional scattering volume, sharing stable sampling resources.
//! Resizing, entering water and enabling caustics have independent allocation lifetimes.
pub(crate) struct WaterResources {
    read_layout: wgpu::BindGroupLayout,
    write_layout: wgpu::BindGroupLayout,
    volume_sampler: wgpu::Sampler,
    caustic_sampler: wgpu::Sampler,
    caustic_texture: wgpu::Texture,
    caustics_ready: bool,
    pub medium_uniforms: wgpu::Buffer,
}

impl WaterResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            read_layout: WaterHits::read_layout(device),
            write_layout: WaterHits::write_layout(device),
            volume_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Underwater trilinear sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            caustic_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Repeating caustic mip sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            caustic_texture: crate::caustics::texture(device, queue, false),
            caustics_ready: false,
            medium_uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Underwater medium uniforms"),
                size: std::mem::size_of::<crate::underwater::Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Upload the immutable cookie once, on first use. Keep it through toggles and resize.
    pub fn ensure_caustics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enabled: bool,
    ) -> bool {
        if enabled && !self.caustics_ready {
            self.caustic_texture = crate::caustics::texture(device, queue, true);
            self.caustics_ready = true;
            true
        } else {
            false
        }
    }

    fn bind(
        &self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        volume: &wgpu::Texture,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let hits = texture.create_view(&Default::default());
        let volume = volume.create_view(&Default::default());
        let caustics = self.caustic_texture.create_view(&Default::default());
        let read = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water sampling resources"),
            layout: &self.read_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hits),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.volume_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.medium_uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&caustics),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.caustic_sampler),
                },
            ],
        });
        let write = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water preparation resources"),
            layout: &self.write_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hits),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&volume),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.medium_uniforms.as_entire_binding(),
                },
            ],
        });
        (read, write)
    }
}

pub(crate) struct WaterHits {
    pub texture: wgpu::Texture,
    pub volume_texture: wgpu::Texture,
    pub read: wgpu::BindGroup,
    pub write: wgpu::BindGroup,
    pub size: [u32; 2],
    pub volume_size: [u32; 3],
}
impl WaterHits {
    pub fn read_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water hits read layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            crate::underwater::Uniforms,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }
    pub fn write_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water hits write layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            crate::underwater::Uniforms,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        })
    }
    pub fn new(
        device: &wgpu::Device,
        size: [u32; 2],
        underwater: bool,
        resources: &WaterResources,
    ) -> Self {
        let texture = Self::hit_texture(device, size);
        let volume_size = Self::volume_size(size, underwater);
        let volume_texture = Self::volume_texture(device, volume_size);
        let (read, write) = resources.bind(device, &texture, &volume_texture);
        Self {
            texture,
            volume_texture,
            read,
            write,
            size,
            volume_size,
        }
    }

    /// Returns which computed textures were replaced and therefore need fresh contents.
    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
        underwater: bool,
        resources: &WaterResources,
        bindings_changed: bool,
    ) -> (bool, bool) {
        let hits_changed = self.size != size;
        let volume_size = Self::volume_size(size, underwater);
        let volume_changed = self.volume_size != volume_size;
        if hits_changed {
            self.texture = Self::hit_texture(device, size);
            self.size = size;
        }
        if volume_changed {
            self.volume_texture = Self::volume_texture(device, volume_size);
            self.volume_size = volume_size;
        }
        if hits_changed || volume_changed || bindings_changed {
            (self.read, self.write) = resources.bind(device, &self.texture, &self.volume_texture);
        }
        (hits_changed, volume_changed)
    }

    fn volume_size(size: [u32; 2], underwater: bool) -> [u32; 3] {
        if underwater {
            [
                size[0].div_ceil(8).min(320),
                size[1].div_ceil(8).min(180),
                crate::underwater::SLICES,
            ]
        } else {
            [1; 3]
        }
    }

    fn volume_texture(device: &wgpu::Device, size: [u32; 3]) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Underwater cumulative scattering"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: size[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn hit_texture(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water hit depth and view distance"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }
}
pub(crate) fn shader_source(group: u32) -> String {
    include_str!("water_hits.wgsl").replace("WATER_HIT_GROUP", &group.to_string())
}
