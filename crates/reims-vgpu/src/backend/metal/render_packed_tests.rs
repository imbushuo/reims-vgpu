use super::super::{packed, resident::PublishedSample};
use super::*;
use objc::rc::{autoreleasepool, WeakPtr};

const VERTEX: &[u8] = b"immutable-packed-pixel-parity-vertex";
const FRAGMENT: &[u8] = b"immutable-packed-pixel-parity-fragment";
const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 packed_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    fragment float4 packed_fragment(float4 p [[position]], texture2d<float> image [[texture(3)]]) {
        constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
        return image.sample(s, (p.xy + float2(0.25, -0.25)) / 4.0);
    }
"#;

fn prepare() {
    let device = system_device().expect("Metal device");
    let library = super::super::raw_metal::new_library_with_source(device, SOURCE).unwrap();
    for (bytes, name) in [(VERTEX, "packed_vertex"), (FRAGMENT, "packed_fragment")] {
        super::super::cache::fn_cache_insert(
            &BlobKey::new(bytes),
            library.get_function(name, None).unwrap(),
        );
    }
}

fn encode(image: &ReimsVgpuSampledImage, capture: bool) -> (Vec<u8>, RenderBatch) {
    autoreleasepool(|| {
        let mut output = vec![0u8; 128];
        let mut batch = RenderBatch::default();
        let mut color = ColorRt {
            slot: 0,
            pixel_format: MTLPixelFormat::RGBA16Float as u32,
            seed_rgba8: None,
            out_rgba8: capture.then_some(output.as_mut_slice()),
            clear_r: 0.0,
            clear_g: 0.0,
            clear_b: 0.0,
            clear_a: 0.0,
            load_action: REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
            blend: None,
            write_mask: 0xf,
            target: ColorTarget::Transient,
        };
        let status = render_core_mrt(
            VERTEX,
            FRAGMENT,
            4,
            4,
            crate::protocol::draw::DrawArgs {
                vertex_count: 3,
                instance_count: 1,
                primitive_type: 3,
                first_vertex: 0,
                base_instance: 0,
            },
            None,
            None,
            &[],
            &[],
            &[],
            &[],
            &[],
            std::slice::from_ref(image),
            &[],
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            std::slice::from_mut(&mut color),
            None,
            (ptr::null_mut(), 0),
            &mut batch,
            true,
        );
        assert!(status.is_ok(), "{status:?}");
        let completes = capture || image.needs_completion();
        assert_eq!(batch.pending(), !completes);
        assert_eq!(batch.submissions, usize::from(completes));
        (output, batch)
    })
}

#[test]
fn imported_sample_is_read_only_retained_until_batch_completion_and_rejects_retirement() {
    use crate::runtime::host::{FakeHost, HostMemory, HostOps};
    use crate::runtime::guest_ram::GuestRamImport;
    let device = system_device().unwrap();
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let mut host = FakeHost::new();
    host.owned_map_pages = true;
    host.map_range(64 * page as u64, page, 0);
    let pointer = host.map_pages(&[64 * page as u64], page).unwrap();
    let guest = Arc::new(GuestRamImport::new_host_allocation(
        pointer, page as u64, page as u64,
    ).unwrap());
    let pitch = device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    for row in 0..4 {
        host.write_gpa(64 * page as u64 + row * pitch, &[17, 43, 91, 255].repeat(4)).unwrap();
    }
    autoreleasepool(|| {
        prepare();
        let image = Arc::new(super::super::mapped_sample::Image::new(
            device, guest.clone(), guest.slice(0, pitch * 4).unwrap(),
            super::super::mapped_sample::Layout {
                width: 4, height: 4, pitch: pitch as u32,
                format: crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM,
            },
        ).unwrap());
        let binding = ReimsVgpuSampledImage::ImportedRead {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3, image,
        };
        assert!(!binding.needs_completion());
        for vertex in [false, true] {
            for access in [RenderTextureAccess::Write, RenderTextureAccess::ReadWrite] {
                assert!(!validate_render_texture_bindings(
                    std::slice::from_ref(&binding),
                    &[RenderTextureUsage { binding: binding.binding(), access }],
                    vertex,
                ).is_ok());
            }
        }
        let (_, mut batch) = encode(&binding, false);
        assert!(batch.pending());
        assert!(batch.pending_guest_reads());
        assert_eq!(batch.submissions, 0);
        batch.finish((ptr::null_mut(), 0)).unwrap();
        assert!(!batch.pending());
        assert!(batch.imported_samples.is_empty());
        assert_eq!(batch.submissions, 1);
        let (_, mut revoked) = encode(&binding, false);
        guest.retire();
        assert!(revoked.finish((ptr::null_mut(), 0)).is_err());
        assert!(!revoked.pending());
        assert!(revoked.imported_samples.is_empty());
        assert_eq!(revoked.submissions, 0);
        super::super::guest_writeback::retire(guest.id());
        let output = new_color_target(device, MTLPixelFormat::RGBA16Float, 4, 4, MTLStorageMode::Shared).unwrap();
        let pass = RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(&output));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_store_action(MTLStoreAction::Store);
        let queue = thread_queue(device);
        let command = queue.new_command_buffer();
        let encoder = command.new_render_command_encoder(pass);
        assert!(!bind_sampled_images(
            device, encoder, &mut Vec::new(), &[binding], true, (ptr::null_mut(), 0),
        ).is_ok());
        encoder.end_encoding();
    });
    let released = super::super::guest_writeback::released();
    assert_eq!(released, vec![(pointer, page)]);
    for (pointer, len) in released { host.unmap_pages(pointer, len); }
}

fn packed_binding(image: Arc<packed::SampledImage>) -> ReimsVgpuSampledImage {
    ReimsVgpuSampledImage::Resident {
        binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
        image: PublishedSample::packed(image),
    }
}

#[test]
fn packed_snapshot_native_rendering_matches_uncached_upload_for_formats_and_pitches() {
    autoreleasepool(|| {
        prepare();
        let half = [0x3400u16, 0x3800, 0x4000, 0x3a00]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let float = [0.25f32, 0.5, 2.0, 0.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        for (format, pixel, pitch) in [
            (0, vec![19, 87, 201, 255], 128), // format zero ignores native pitch
            (MTLPixelFormat::RGBA8Unorm as u32, vec![19, 87, 201, 255], 0),
            (
                MTLPixelFormat::BGRA8Unorm as u32,
                vec![19, 87, 201, 255],
                32,
            ),
            (
                MTLPixelFormat::BGRA8Unorm_sRGB as u32,
                vec![19, 87, 201, 255],
                32,
            ),
            (MTLPixelFormat::RG8Unorm as u32, vec![19, 201], 16),
            (
                MTLPixelFormat::R16Float as u32,
                0x3800u16.to_le_bytes().to_vec(),
                16,
            ),
            (MTLPixelFormat::RGBA16Float as u32, half, 48),
            (MTLPixelFormat::RGBA32Float as u32, float, 80),
        ] {
            let layout = packed::Layout {
                width: 4,
                height: 4,
                pixel_format: format,
                bytes_per_row: pitch,
            };
            let row = if format == 0 || pitch == 0 {
                4 * pixel.len()
            } else {
                pitch as usize
            };
            let mut bytes = vec![0xee; row * 4];
            for y in 0..4 {
                for x in 0..4 {
                    let start = y * row + x * pixel.len();
                    bytes[start..start + pixel.len()].copy_from_slice(&pixel);
                }
            }
            let immutable = Arc::new(packed::SampledImage::new(layout, bytes.clone()).unwrap());
            assert_eq!(immutable.texture().mipmap_level_count(), 1);
            assert_eq!(immutable.texture().texture_type(), MTLTextureType::D2);
            assert_eq!(immutable.texture().usage(), MTLTextureUsage::ShaderRead);
            let cached = packed_binding(immutable);
            let uncached = ReimsVgpuSampledImage::Packed(
                layout.image(&bytes, REIMS_VGPU_BINDING_TEXTURE_BASE + 3),
            );
            let baseline = encode(&uncached, true).0;
            assert!(
                baseline.iter().any(|&byte| byte != 0),
                "the oracle draw must render"
            );
            assert_eq!(
                encode(&cached, true).0,
                baseline,
                "format={format:#x} pitch={pitch}"
            );
        }
    });
}

#[test]
fn packed_snapshot_is_read_only_and_batch_owns_it_through_completion() {
    let (weak_native, weak_image, mut pending) = autoreleasepool(|| {
        prepare();
        let layout = packed::Layout {
            width: 4,
            height: 4,
            pixel_format: 0,
            bytes_per_row: 16,
        };
        let image = Arc::new(packed::SampledImage::new(layout, vec![0x5a; 64]).unwrap());
        let weak_native = unsafe { WeakPtr::new(image.texture().as_ptr().cast()) };
        let weak_image = Arc::downgrade(&image);
        let binding = packed_binding(image);
        let before = encode(&binding, true).0;
        let replacement = packed_binding(Arc::new(
            packed::SampledImage::new(layout, vec![0xa5; 64]).unwrap(),
        ));
        assert_ne!(encode(&replacement, true).0, before);
        assert_eq!(
            encode(&binding, true).0,
            before,
            "a replacement upload cannot mutate pixels in an earlier publication",
        );
        assert!(
            !binding.needs_completion(),
            "immutable uploads must not force batch flushes"
        );
        for vertex in [false, true] {
            for access in [RenderTextureAccess::Write, RenderTextureAccess::ReadWrite] {
                assert!(!validate_render_texture_bindings(
                    std::slice::from_ref(&binding),
                    &[RenderTextureUsage {
                        binding: binding.binding(),
                        access
                    }],
                    vertex,
                )
                .is_ok());
            }
        }
        let (_, pending) = encode(&binding, false);
        assert_eq!(pending.published_samples.len(), 1);
        (weak_native, weak_image, pending)
    });
    assert!(weak_image.upgrade().is_some());
    assert!(
        !weak_native.load().is_null(),
        "autorelease pools cannot free an in-flight image"
    );
    autoreleasepool(|| pending.finish((ptr::null_mut(), 0)).unwrap());
    assert!(weak_image.upgrade().is_none());
    assert!(
        weak_native.load().is_null(),
        "completed batches retain no retired image"
    );
}

#[test]
fn packed_snapshot_and_uncached_upload_share_validation() {
    let device = system_device().expect("Metal device");
    let base = packed::Layout {
        width: 4,
        height: 4,
        pixel_format: MTLPixelFormat::RGBA8Unorm as u32,
        bytes_per_row: 16,
    };
    for (layout, len) in [
        (packed::Layout { width: 0, ..base }, 64),
        (packed::Layout { height: 0, ..base }, 64),
        (
            packed::Layout {
                pixel_format: u32::MAX,
                ..base
            },
            64,
        ),
        (base, 63),
        (
            packed::Layout {
                pixel_format: 0,
                ..base
            },
            63,
        ),
    ] {
        let bytes = vec![0; len];
        let uncached = upload_packed_sampled_image(
            device,
            &layout.image(&bytes, REIMS_VGPU_BINDING_TEXTURE_BASE),
            false,
            (ptr::null_mut(), 0),
        )
        .unwrap_err();
        let cached = packed::SampledImage::new(layout, bytes).unwrap_err();
        assert_eq!(format!("{cached:?}"), format!("{uncached:?}"));
    }
}
