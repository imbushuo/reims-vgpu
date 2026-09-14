use super::*;
use crate::runtime::host::{FakeHost, HostMemory, HostOps};

struct Memory {
    host: FakeHost,
    guest: Arc<GuestRamImport>,
    gpa: u64,
}

impl Memory {
    fn new(needed: u64) -> Self {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let len = needed.max(page).div_ceil(page) * page;
        let gpa = 64 * page;
        let mut host = FakeHost::new();
        host.owned_map_pages = true;
        host.map_range(gpa, len as usize, 0xcc);
        let pages: Vec<_> = (0..len / page).map(|i| gpa + i * page).collect();
        let ptr = host.map_pages(&pages, page as usize).unwrap();
        let guest = Arc::new(GuestRamImport::new_host_allocation(ptr, len, page).unwrap());
        Self { host, guest, gpa }
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) {
        self.host.write_gpa(self.gpa + offset, bytes).unwrap();
    }

    fn retire(&mut self) {
        self.guest.retire();
        guest_writeback::retire(self.guest.id());
        for (ptr, len) in guest_writeback::released() {
            self.host.unmap_pages(ptr, len);
        }
    }
}

struct Reader {
    pipeline: ComputePipelineState,
    queue: CommandQueue,
}

impl Reader {
    fn new(device: &DeviceRef) -> Self {
        let library = device
            .new_library_with_source(
                "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void read_image(texture2d<float, access::sample> image [[texture(0)]], \
             device float4 *out [[buffer(0)]], uint2 p [[thread_position_in_grid]]) { \
             constexpr sampler s(coord::normalized,address::clamp_to_edge,filter::linear); \
             out[p.y*image.get_width()+p.x]=image.sample(s,(float2(p)+float2(0.75,0.25))/\
             float2(image.get_width(),image.get_height())); }",
                &CompileOptions::new(),
            )
            .unwrap();
        let function = library.get_function("read_image", None).unwrap();
        Self {
            pipeline: device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap(),
            queue: device.new_command_queue(),
        }
    }

    fn record(&self, device: &DeviceRef, texture: &TextureRef) -> (CommandBuffer, Buffer) {
        let len = texture.width() * texture.height() * 16;
        let output = device.new_buffer(len, MTLResourceOptions::StorageModeShared);
        let command = self.queue.new_command_buffer().to_owned();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_texture(0, Some(texture));
        encoder.set_buffer(0, Some(&output), 0);
        encoder.dispatch_threads(
            MTLSize::new(texture.width(), texture.height(), 1),
            MTLSize::new(8, 1, 1),
        );
        encoder.end_encoding();
        (command, output)
    }

    fn sample(&self, device: &DeviceRef, texture: &TextureRef) -> Vec<u8> {
        let (command, output) = self.record(device, texture);
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // SAFETY: this completed command initialized every float4 in output.
        unsafe {
            std::slice::from_raw_parts(output.contents().cast::<u8>(), output.length() as usize)
                .to_vec()
        }
    }
}

#[test]
fn mapped_sample_matches_full_rgba_loader_for_every_channel_and_srgb_bytes() {
    let device = super::super::runtime::system_device().unwrap();
    let reader = Reader::new(device);
    for format in [
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
        pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB,
    ] {
        let mut layout = Layout {
            width: 257,
            height: 3,
            pitch: 0,
            format,
        };
        let alignment = device
            .minimum_linear_texture_alignment_for_pixel_format(layout.pixel_format().unwrap());
        layout.pitch =
            ((u64::from(layout.width) * 4).div_ceil(alignment) * alignment + alignment) as u32;
        let offset = alignment;
        let span = layout.span().unwrap();
        let mut memory = Memory::new(offset + span);
        let mut raw = vec![0xce; span as usize];
        let mut expected = vec![0; layout.width as usize * layout.height as usize * 4];
        let converter = pixel_format::RowToRgba8::for_format(format).unwrap();
        for y in 0..layout.height as usize {
            let row = &mut raw
                [y * layout.pitch as usize..y * layout.pitch as usize + layout.width as usize * 4];
            for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let value = (x + y * 37) as u8;
                *pixel = [
                    value,
                    255 - value,
                    value.wrapping_mul(13),
                    value.wrapping_mul(17),
                ];
            }
            let start = y * layout.width as usize * 4;
            assert!(converter.convert(row, layout.width, &mut expected[start..start + row.len()]));
        }
        memory.write(offset, &raw);
        objc::rc::autoreleasepool(|| {
            let imported = Image::new(
                device,
                memory.guest.clone(),
                memory.guest.slice(offset, span).unwrap(),
                layout,
            )
            .unwrap();
            let staged = super::super::packed::SampledImage::new(
                super::super::packed::Layout {
                    width: layout.width,
                    height: layout.height,
                    pixel_format: 0,
                    bytes_per_row: layout.width * 4,
                },
                expected,
            )
            .unwrap();
            assert_eq!(
                reader.sample(device, imported.texture(device).unwrap()),
                reader.sample(device, staged.texture()),
                "format={format:#x} padded pitch={} offset={offset}",
                layout.pitch,
            );
        });
        memory.retire();
    }
}

#[test]
fn mapped_sample_cpu_changes_after_completion_are_visible_without_any_dirty_generation() {
    let device = super::super::runtime::system_device().unwrap();
    let reader = Reader::new(device);
    let pitch =
        device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    let layout = Layout {
        width: 1,
        height: 1,
        pitch: pitch as u32,
        format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
    };
    let mut memory = Memory::new(pitch);
    objc::rc::autoreleasepool(|| {
        let imported = Image::new(
            device,
            memory.guest.clone(),
            memory.guest.slice(0, pitch).unwrap(),
            layout,
        )
        .unwrap();
        for round in 0..256u16 {
            let pixel = [round as u8, (round * 13) as u8, (255 - round) as u8, 255];
            memory.write(0, &pixel);
            let actual = reader.sample(device, imported.texture(device).unwrap());
            for (bytes, value) in actual.as_chunks::<4>().0.iter().zip(pixel) {
                let actual = f32::from_ne_bytes(*bytes);
                assert!(
                    (actual - f32::from(value) / 255.0).abs() < 0.000001,
                    "completed round={round} actual={actual} byte={value}"
                );
            }
        }
    });
    memory.retire();
}

#[test]
fn mapped_half_sample_gpu_conversion_matches_every_half_encoding_and_refreshes_each_command() {
    let device = super::super::runtime::system_device().unwrap();
    let reader = Reader::new(device);
    let layout = Layout {
        width: 256,
        height: 64,
        pitch: 256 * 8 + 16,
        format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    };
    let span = layout.span().unwrap();
    let mut memory = Memory::new(span);
    objc::rc::autoreleasepool(|| {
        let image = Image::new(
            device, memory.guest.clone(), memory.guest.slice(0, span).unwrap(), layout,
        ).unwrap();
        assert!(image.needs_conversion());
        for round in 0..2u16 {
            let mut raw = vec![0xa5; span as usize];
            let mut expected = vec![0; layout.width as usize * layout.height as usize * 4];
            for row in 0..layout.height as usize {
                let source = &mut raw[row * layout.pitch as usize..][..layout.width as usize * 8];
                for (channel, bytes) in source.chunks_exact_mut(2).enumerate() {
                    let bits = ((row * 1024 + channel) as u16).wrapping_add(round * 7919);
                    bytes.copy_from_slice(&bits.to_le_bytes());
                }
                assert!(pixel_format::RowToRgba8::Rgba16Float.convert(
                    source, layout.width, &mut expected[row * 1024..][..1024],
                ));
            }
            memory.write(0, &raw);
            let command = reader.queue.new_command_buffer().to_owned();
            image.encode_conversion(device, &command).unwrap();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let staged = super::super::packed::SampledImage::new(
                super::super::packed::Layout {
                    width: layout.width, height: layout.height,
                    pixel_format: 0, bytes_per_row: layout.width * 4,
                },
                expected,
            ).unwrap();
            assert_eq!(
                reader.sample(device, image.texture(device).unwrap()),
                reader.sample(device, staged.texture()),
            );
        }
    });
    memory.retire();
}

#[test]
fn mapped_sample_retires_before_encode_and_keeps_submitted_native_references_owned() {
    let device = super::super::runtime::system_device().unwrap();
    let reader = Reader::new(device);
    let pitch =
        device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    let layout = Layout {
        width: 1,
        height: 1,
        pitch: pitch as u32,
        format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
    };
    let mut memory = Memory::new(pitch);
    memory.write(0, &[255, 0, 0, 255]);
    objc::rc::autoreleasepool(|| {
        let image = Image::new(
            device,
            memory.guest.clone(),
            memory.guest.slice(0, pitch).unwrap(),
            layout,
        )
        .unwrap();
        let (command, output) = reader.record(device, image.texture(device).unwrap());
        command.commit();
        memory.guest.retire();
        assert!(guest_writeback::retire(memory.guest.id()).is_some());
        assert!(
            image.texture(device).is_err(),
            "a retired capture cannot be newly encoded"
        );
        drop(image);
        assert!(
            guest_writeback::released().is_empty(),
            "Metal still owns the encoded read"
        );
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let red: f32 = unsafe { *output.contents().cast::<f32>() };
        assert_eq!(red, 1.0);
    });
    let released = guest_writeback::released();
    assert_eq!(
        released,
        vec![(memory.guest.host_base(), memory.guest.len() as usize)]
    );
    for (ptr, len) in released {
        memory.host.unmap_pages(ptr, len);
    }
}

#[test]
fn mapped_sample_unsupported_formats_alignment_and_foreign_slices_are_refused() {
    let device = super::super::runtime::system_device().unwrap();
    let pitch =
        device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    let layout = Layout {
        width: 2,
        height: 1,
        pitch: pitch as u32,
        format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
    };
    let mut memory = Memory::new(pitch * 2);
    let foreign = Arc::new(
        GuestRamImport::new_host_allocation(
            memory.guest.host_base(),
            memory.guest.len(),
            memory.guest.align(),
        )
        .unwrap(),
    );
    for (shape, slice) in [
        (
            Layout {
                format: pixel_format::MTL_FORMAT_RG16_FLOAT,
                ..layout
            },
            memory.guest.slice(0, pitch).unwrap(),
        ),
        (
            Layout { pitch: 9, ..layout },
            memory.guest.slice(0, pitch).unwrap(),
        ),
        (layout, memory.guest.slice(1, pitch).unwrap()),
        (layout, foreign.slice(0, pitch).unwrap()),
        (layout, memory.guest.slice(0, 1).unwrap()),
    ] {
        assert!(Image::new(device, memory.guest.clone(), slice, shape).is_err());
    }
    memory.retire();
}
