use super::*;
use crate::observe::Refusal;
use foreign_types::{ForeignType, ForeignTypeRef};
use std::alloc::{alloc, dealloc, Layout as AllocationLayout};
use std::ptr::NonNull;

struct Memory {
    ptr: NonNull<u8>,
    layout: AllocationLayout,
}

impl Memory {
    fn new() -> Self {
        Self::with_len(0)
    }

    fn with_len(len: usize) -> Self {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let layout =
            AllocationLayout::from_size_align(len.max(page * 2).div_ceil(page) * page, page)
                .unwrap();
        let ptr = NonNull::new(unsafe { alloc(layout) }).expect("test backing");
        unsafe {
            ptr.as_ptr().write_bytes(0xca, layout.size());
        }
        Self { ptr, layout }
    }

    fn guest(&self) -> Arc<GuestRamImport> {
        Arc::new(
            GuestRamImport::new_host_allocation(
                self.ptr.as_ptr() as usize,
                self.layout.size() as u64,
                self.layout.align() as u64,
            )
            .unwrap(),
        )
    }

    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.layout.size()) }
    }
}

#[test]
#[ignore = "isolated native GPU throughput measurement"]
fn benchmark_full_frame_store() {
    use objc::{msg_send, sel, sel_impl};

    let device = runtime::system_device().expect("Metal device");
    assert!(device.has_unified_memory());
    for output in [Output::Bgra8, Output::Rgba16Float] {
        let layout = Layout {
            width: 1920,
            height: 1080,
            pitch: 1920 * output.bytes() as u32,
            output,
        };
        let memory = Memory::with_len(layout.span().unwrap() as usize);
        let guest = memory.guest();
        let (texture, _) = source(device, layout.width, layout.height);
        let mut prepare_ns = 0u128;
        let mut execute_ns = 0u128;
        let mut gpu_seconds = 0.0;
        for iteration in 0..110 {
            objc::rc::autoreleasepool(|| {
                let started = std::time::Instant::now();
                let prepared = Prepared::new(
                    device,
                    &texture,
                    guest.clone(),
                    guest.slice(0, layout.span().unwrap()).unwrap(),
                    layout,
                )
                .unwrap();
                let command = prepared.command.clone();
                let prepared_at = std::time::Instant::now();
                prepared.execute().unwrap();
                if iteration >= 10 {
                    prepare_ns += prepared_at.duration_since(started).as_nanos();
                    execute_ns += prepared_at.elapsed().as_nanos();
                    let start: f64 = unsafe { msg_send![command.as_ptr(), GPUStartTime] };
                    let end: f64 = unsafe { msg_send![command.as_ptr(), GPUEndTime] };
                    assert!(start > 0.0 && end >= start);
                    gpu_seconds += end - start;
                }
            });
        }
        println!(
            "GPU_STORE_BENCH bpp={} prepare_us={:.2} execute_us={:.2} gpu_us={:.2}",
            output.bytes(),
            prepare_ns as f64 / 100_000.0,
            execute_ns as f64 / 100_000.0,
            gpu_seconds * 10_000.0,
        );
        guest.retire();
        retire(guest.id()).unwrap();
        assert_eq!(released(), [(guest.host_base(), guest.len() as usize)]);
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

fn source(device: &Device, width: u32, height: u32) -> (Texture, Vec<u8>) {
    let align =
        device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    let pitch = (u64::from(width) * 4).div_ceil(align) * align + align;
    let offset = align;
    let buffer = raw_metal::new_buffer(
        device,
        offset + pitch * u64::from(height),
        MTLResourceOptions::StorageModeShared,
    )
    .unwrap();
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_width(u64::from(width));
    descriptor.set_height(u64::from(height));
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = raw_metal::new_linear_texture(&buffer, &descriptor, offset, pitch)
        .expect("test host supports linear RGBA8 textures");
    let rgba: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let v = i as u8;
            [v, 255 - v, v ^ 0x5a, v.wrapping_mul(17)]
        })
        .collect();
    texture.replace_region(
        MTLRegion::new_2d(0, 0, u64::from(width), u64::from(height)),
        0,
        rgba.as_ptr().cast(),
        u64::from(width) * 4,
    );
    (texture, rgba)
}

#[test]
fn gpu_store_matches_cpu_all_levels_offsets_and_padded_rows_without_srgb_encoding() {
    let device = runtime::system_device().expect("Metal device");
    if !device.has_unified_memory() {
        return;
    }
    for format in [
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    ] {
        let output = Output::for_format(format).unwrap();
        let layout = Layout {
            width: 257,
            height: 37,
            pitch: 257 * output.bytes() as u32 + 7,
            output,
        };
        let offset = 17;
        let memory = Memory::with_len((layout.span().unwrap() + offset) as usize);
        let guest = memory.guest();
        let slice = guest.slice(offset, layout.span().unwrap()).unwrap();
        let mut expected = vec![0xca; memory.layout.size()];
        objc::rc::autoreleasepool(|| {
            let (source, rgba) = source(device, layout.width, layout.height);
            for y in 0..layout.height as usize {
                let dst = offset as usize + y * layout.pitch as usize;
                let row = layout.width as usize * 4;
                assert!(pixel_format::Rgba8ToRow::for_format(format)
                    .unwrap()
                    .convert(
                        &rgba[y * row..(y + 1) * row],
                        layout.width,
                        &mut expected[dst..dst + layout.width as usize * output.bytes() as usize],
                    ));
            }
            Prepared::new(device, &source, guest.clone(), slice, layout)
                .unwrap()
                .execute()
                .unwrap();
        });
        assert_eq!(memory.bytes(), expected, "format={format}");
        guest.retire();
        assert_eq!(
            retire(guest.id()),
            Some((guest.host_base(), guest.len() as usize))
        );
        assert_eq!(released(), [(guest.host_base(), guest.len() as usize)]);
    }
}

#[test]
fn checked_import_is_reused_and_retirement_waits_for_every_metal_reference() {
    let device = runtime::system_device().expect("Metal device");
    if !device.has_unified_memory() {
        return;
    }
    let memory = Memory::new();
    let guest = memory.guest();
    let first = import(device, guest.clone()).unwrap();
    let second = import(device, guest.clone()).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    let native = first.buffer.clone();
    let pointer = native.as_ptr();
    drop((first, second));
    guest.retire();
    assert_eq!(
        retire(guest.id()),
        Some((guest.host_base(), guest.len() as usize))
    );
    assert!(
        released().is_empty(),
        "the Metal deallocator, not Rust cache removal, owns release"
    );
    assert_eq!(native.as_ptr(), pointer);
    assert!(matches!(import(device, guest.clone()), Err(status)
        if status.refusal() == Some("metal_guest_import_retired")));
    drop(native);
    assert_eq!(released(), [(guest.host_base(), guest.len() as usize)]);
    assert!(retire(guest.id()).is_none());
}

#[test]
fn readonly_source_window_exposes_checked_coordinates_without_submitting_work() {
    let device = runtime::system_device().unwrap();
    let memory = Memory::new();
    let guest = memory.guest();
    let layout = ReadLayout {
        width: 4, height: 3, pitch: 48,
        format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    };
    let source = ReadSource::new(
        device, guest.clone(), guest.slice(17, 256).unwrap(), layout,
    ).unwrap();
    assert_eq!(source.offset(), 17);
    assert_eq!(source.span(), 128);
    assert_eq!(source.available(), 256);
    assert_eq!(source.layout().format, pixel_format::MTL_FORMAT_RGBA16_FLOAT);
    assert_eq!(source.import_id(), guest.id());
    let native = source.buffer().unwrap().to_owned();
    let again = ReadSource::new(
        device, guest.clone(), guest.slice(17, 256).unwrap(), layout,
    ).unwrap();
    assert_eq!(native.as_ptr(), again.buffer().unwrap().as_ptr());
    assert!(!outstanding(), "a read source creates neither a command nor a write");
    guest.retire();
    retire(guest.id()).unwrap();
    assert!(source.check_live().is_err());
    assert!(source.buffer().is_err());
    drop((source, again));
    assert!(released().is_empty(), "the command owner's buffer reference still pins the alias");
    drop(native);
    assert_eq!(released(), [(guest.host_base(), guest.len() as usize)]);
}
#[test]
fn reset_retires_ready_and_failed_imports_without_unmapping_live_metal_references() {
    let device = runtime::system_device().expect("Metal device");
    if !device.has_unified_memory() {
        return;
    }
    let memory = Memory::new();
    let guest = memory.guest();
    let ready = import(device, guest.clone()).unwrap();
    let native = ready.buffer.clone();
    drop(ready);
    let invalid =
        Arc::new(GuestRamImport::new_host_allocation(memory.ptr.as_ptr() as usize, 1, 1).unwrap());
    assert!(import(device, invalid.clone()).is_err());
    reset();
    assert!(guest.is_retired() && invalid.is_retired());
    assert!(import(device, guest.clone()).is_err());
    assert!(import(device, invalid).is_err());
    assert!(released().is_empty());
    assert_eq!(
        retire(guest.id()),
        Some((guest.host_base(), guest.len() as usize))
    );
    drop(native);
    assert_eq!(released(), [(guest.host_base(), guest.len() as usize)]);
}

#[test]
fn guest_store_refuses_foreign_short_and_retired_slices_before_submission() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        if !device.has_unified_memory() {
            return;
        }
        let memory = Memory::new();
        let guest = memory.guest();
        let foreign = memory.guest();
        let (source, _) = source(device, 4, 2);
        let layout = Layout {
            width: 4,
            height: 2,
            pitch: 32,
            output: Output::Bgra8,
        };
        for slice in [foreign.slice(0, 64).unwrap(), guest.slice(0, 1).unwrap()] {
            assert!(Prepared::new(device, &source, guest.clone(), slice, layout).is_err());
        }
        let slice = guest.slice(0, 64).unwrap();
        guest.retire();
        assert!(Prepared::new(device, &source, guest, slice, layout).is_err());
        assert!(memory.bytes().iter().all(|&byte| byte == 0xca));
        assert!(!outstanding());
    });
}

#[test]
fn pending_command_retains_import_until_completion_and_autorelease_drain() {
    let device = runtime::system_device().expect("Metal device");
    if !device.has_unified_memory() {
        return;
    }
    let memory = Memory::new();
    let guest = memory.guest();
    let mut ready = objc::rc::autoreleasepool(|| {
        let (source, _) = source(device, 4, 2);
        let mut write = Prepared::new(
            device,
            &source,
            guest.clone(),
            guest.slice(0, 64).unwrap(),
            Layout {
                width: 4,
                height: 2,
                pitch: 32,
                output: Output::Bgra8,
            },
        )
        .unwrap();
        write.submitted = true;
        write.command.commit();
        guest.retire();
        retire(guest.id()).unwrap();
        assert!(released().is_empty());
        drop(write); // Drop waits for the committed command.
        released()
    });
    ready.extend(released());
    assert_eq!(ready, [(guest.host_base(), guest.len() as usize)]);
}
