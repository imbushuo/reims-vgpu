//! Native CPU-write oracle. Callers complete sampling before the next update;
//! pixels change under IOSurfaceLock/Unlock, not by mutating a sampled snapshot.

use super::*;
use metal::{MTLCommandBufferStatus, MTLPixelFormat, MTLTextureType, MTLTextureUsage};

unsafe extern "C" {
    fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
    fn IOSurfaceGetSeed(surface: *mut c_void) -> u32;
}

pub(crate) struct CpuSurface {
    surface: Surface,
    texture: Texture,
    layout: Option<Layout>,
}

impl CpuSurface {
    pub(crate) fn packed() -> Self {
        objc::rc::autoreleasepool(|| {
            let properties = dictionary();
            put(properties, c"IOSurfaceWidth", 8);
            put(properties, c"IOSurfaceHeight", 4);
            put(
                properties,
                c"IOSurfacePixelFormat",
                u32::from_be_bytes(*b"BGRA").into(),
            );
            put(properties, c"IOSurfaceBytesPerElement", 4);
            put(properties, c"IOSurfaceAllocSize", 1 << 14);
            let surface = Surface(unsafe { IOSurfaceCreate(properties) });
            assert!(!surface.0.is_null());
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_width(8);
            descriptor.set_height(4);
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
            let device: &metal::DeviceRef = super::super::super::runtime::system_device().unwrap();
            let raw: *mut Object = unsafe {
                msg_send![device, newTextureWithDescriptor: descriptor.as_ptr()
                    iosurface: surface.0 plane: 0u64]
            };
            assert!(!raw.is_null());
            let texture = unsafe { Texture::from_ptr(raw.cast()) };
            Self {
                surface,
                texture,
                layout: None,
            }
        })
    }

    pub(crate) fn planar() -> Self {
        objc::rc::autoreleasepool(|| {
            let layout =
                Layout::decode(&device_descriptor(BackingFormat::VideoRange), 8, 4).unwrap();
            let description = TextureDescription::decode(
                &texture_descriptor(SampleFormat::Rgb10_420TwoPlane),
                11,
            )
            .unwrap();
            let surface = create_surface(&layout).unwrap();
            let device = super::super::super::runtime::system_device().unwrap();
            let texture = create_texture(device, &surface, description, &layout).unwrap();
            Self {
                surface,
                texture,
                layout: Some(layout),
            }
        })
    }

    pub(crate) fn texture(&self) -> &metal::TextureRef {
        &self.texture
    }

    pub(crate) fn pitch(&self) -> usize {
        unsafe { IOSurfaceGetBytesPerRow(self.surface.0) }
    }

    pub(crate) fn seed(&self) -> u32 {
        unsafe { IOSurfaceGetSeed(self.surface.0) }
    }

    fn update(&mut self, write: impl FnOnce(&mut [u8], Option<&Layout>, usize)) -> Vec<u8> {
        assert_eq!(
            unsafe { IOSurfaceLock(self.surface.0, 0, std::ptr::null_mut()) },
            0
        );
        let lock = LockedSurface {
            surface: &self.surface,
            active: true,
        };
        // SAFETY: the lock owns CPU access to this live IOSurface allocation.
        let base = unsafe { IOSurfaceGetBaseAddress(self.surface.0).cast::<u8>() };
        let len = unsafe { IOSurfaceGetAllocSize(self.surface.0) };
        assert!(!base.is_null());
        let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
        bytes.fill(0);
        write(bytes, self.layout.as_ref(), self.pitch());
        let snapshot = bytes.to_vec();
        lock.unlock().unwrap();
        snapshot
    }

    pub(crate) fn packed_pixels(&mut self, bgra: [u8; 4]) -> Vec<u8> {
        self.update(|bytes, layout, pitch| {
            assert!(layout.is_none());
            for row in 0..4 {
                bytes[row * pitch..row * pitch + 32].copy_from_slice(&bgra.repeat(8));
            }
        })
    }

    pub(crate) fn planar_pixels(&mut self, y: u16, cb: u16, cr: u16) -> Vec<u8> {
        self.update(|bytes, layout, _| {
            let layout = layout.unwrap();
            for (index, plane) in layout.planes.iter().enumerate() {
                let bytes = &mut bytes[plane.base as usize..plane.end() as usize];
                if index == 0 {
                    for word in bytes.as_chunks_mut::<2>().0 {
                        *word = (y << 6).to_le_bytes();
                    }
                } else {
                    for pair in bytes.as_chunks_mut::<4>().0 {
                        pair[..2].copy_from_slice(&(cb << 6).to_le_bytes());
                        pair[2..].copy_from_slice(&(cr << 6).to_le_bytes());
                    }
                }
            }
        })
    }
}

pub(crate) struct Sampler {
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
    output: metal::Buffer,
}

impl Sampler {
    pub(crate) fn new() -> Self {
        let device = super::super::super::runtime::system_device().unwrap();
        let library = device
            .new_library_with_source(
                "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void sample_cpu_surface(texture2d<float, access::sample> image [[texture(0)]], \
             device float4 *out [[buffer(0)]], uint2 p [[thread_position_in_grid]]) { \
             out[p.y*image.get_width()+p.x]=image.read(p); }",
                &CompileOptions::new(),
            )
            .unwrap();
        let function = library.get_function("sample_cpu_surface", None).unwrap();
        Self {
            pipeline: device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap(),
            queue: device.new_command_queue(),
            output: device.new_buffer(8 * 4 * 16, MTLResourceOptions::StorageModeShared),
        }
    }

    pub(crate) fn sample(&self, texture: &metal::TextureRef) -> Vec<f32> {
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_texture(0, Some(texture));
        encoder.set_buffer(0, Some(&self.output), 0);
        encoder.dispatch_threads(MTLSize::new(8, 4, 1), MTLSize::new(8, 4, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: the completed command initialized the entire shared output.
        unsafe { std::slice::from_raw_parts(self.output.contents().cast::<f32>(), 128).to_vec() }
    }

    pub(crate) fn clear(&self, texture: &metal::TextureRef) {
        let pass = metal::RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(texture));
        color.set_load_action(metal::MTLLoadAction::Clear);
        color.set_store_action(metal::MTLStoreAction::Store);
        color.set_clear_color(metal::MTLClearColor::new(0.25, 0.5, 0.75, 1.0));
        let command = self.queue.new_command_buffer();
        command.new_render_command_encoder(pass).end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    }
}
