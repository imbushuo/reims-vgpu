use super::*;
use crate::backend::metal::runtime;
use crate::observe::Refusal;
use crate::protocol::pixel_format;

fn buffer(device: &DeviceRef, bytes: &[u8]) -> Buffer {
    unsafe {
        raw_metal::new_buffer_with_data(
            device,
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .unwrap()
}

fn target(device: &DeviceRef, width: u32, height: u32) -> (Texture, Buffer) {
    let align =
        device.minimum_linear_texture_alignment_for_pixel_format(MTLPixelFormat::RGBA8Unorm);
    let pitch = (u64::from(width) * 4).div_ceil(align) * align + align;
    let storage = buffer(
        device,
        &vec![0xa7; (align + pitch * u64::from(height) + align) as usize],
    );
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_width(u64::from(width));
    descriptor.set_height(u64::from(height));
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = raw_metal::new_linear_texture(&storage, &descriptor, align, pitch).unwrap();
    (texture, storage)
}

fn bytes(buffer: &BufferRef) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast(), buffer.length() as usize) }
        .to_vec()
}

fn command(device: &Device) -> CommandBuffer {
    raw_metal::new_command_buffer(&runtime::thread_queue(device))
        .unwrap()
        .to_owned()
}

fn finish(command: &CommandBufferRef) {
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
}

fn parity(format: u16, width: u32, height: u32, pixel: impl Fn(u32) -> Vec<u8>) {
    let device = runtime::system_device().unwrap();
    let conversion = RowToRgba8::for_format(format).unwrap();
    let bpp = conversion.source_bytes_per_pixel() as usize;
    let pitch = width as usize * bpp + 7;
    let offset = 17;
    let mut source = vec![0x6d; offset + pitch * height as usize + 19];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let at = offset + y * pitch + x * bpp;
            source[at..at + bpp].copy_from_slice(&pixel((y * width as usize + x) as u32));
        }
    }
    let layout = Layout {
        source_offset: offset as u64,
        source_length: span(width, height, pitch as u64, bpp as u64).unwrap(),
        source_pitch: pitch as u64,
        width,
        height,
        source_format: format,
    };
    let source_buffer = buffer(device, &source);
    let (target, storage) = target(device, width, height);
    let mut expected = bytes(&storage);
    for y in 0..height as usize {
        let src = &source[offset + y * pitch..][..width as usize * bpp];
        let start = target.buffer_offset() as usize + y * target.buffer_stride() as usize;
        assert!(conversion.convert(src, width, &mut expected[start..start + width as usize * 4]));
    }
    let cmd = command(device);
    let _job = Prepared::encode(device, &cmd, &source_buffer, &target, layout).unwrap();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::NotEnqueued);
    assert!(
        bytes(&storage).iter().all(|&byte| byte == 0xa7),
        "encoding must not execute or CPU-fill pixels"
    );
    finish(&cmd);
    assert_eq!(
        bytes(&storage),
        expected,
        "format={format}; includes prefix, padded rows and suffix"
    );
    assert_eq!(
        bytes(&source_buffer),
        source,
        "source bytes and guards are read-only"
    );
}

#[test]
fn gpu_seed_matches_all_8bit_levels_raw_bgra_rgba_and_srgb() {
    objc::rc::autoreleasepool(|| {
        for format in [
            pixel_format::MTL_FORMAT_RGBA8_UNORM,
            pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
            pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB,
        ] {
            parity(format, 257, 3, |i| {
                let value = i as u8;
                vec![value, 255 - value, value ^ 0xa5, value.wrapping_mul(17)]
            });
        }
    });
}

#[test]
fn gpu_seed_matches_all_65536_half_patterns_in_every_channel() {
    objc::rc::autoreleasepool(|| {
        parity(pixel_format::MTL_FORMAT_RGBA16_FLOAT, 257, 256, |i| {
            (0..4)
                .flat_map(|channel| (i.wrapping_add(channel * 16381) as u16).to_le_bytes())
                .collect()
        });
    });
}

#[test]
fn gpu_seed_validates_authorized_source_bounds_formats_and_aliases_before_encoding() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let (texture, destination) = target(device, 3, 2);
        let source = buffer(device, &[0x31; 128]);
        let valid = Layout {
            source_offset: 5,
            source_length: 28,
            source_pitch: 16,
            width: 3,
            height: 2,
            source_format: pixel_format::MTL_FORMAT_BGRA8_UNORM,
        };
        for (layout, reason) in [
            (
                Layout {
                    source_length: 27,
                    ..valid
                },
                "metal_guest_seed_source_bounds",
            ),
            (
                Layout {
                    source_offset: u64::MAX,
                    ..valid
                },
                "metal_guest_seed_source_bounds",
            ),
            (
                Layout {
                    source_length: 128,
                    ..valid
                },
                "metal_guest_seed_source_bounds",
            ),
            (
                Layout {
                    source_pitch: 11,
                    ..valid
                },
                "metal_guest_seed_source_layout",
            ),
            (
                Layout {
                    source_pitch: u64::MAX,
                    ..valid
                },
                "metal_guest_seed_source_layout",
            ),
            (
                Layout { width: 2, ..valid },
                "metal_guest_seed_target_format",
            ),
            (
                Layout {
                    source_format: pixel_format::MTL_FORMAT_R8_UNORM,
                    ..valid
                },
                "metal_guest_seed_source_format",
            ),
        ] {
            let cmd = command(device);
            let Err(status) = Prepared::encode(device, &cmd, &source, &texture, layout) else {
                panic!("must refuse")
            };
            assert_eq!(status.refusal(), Some(reason));
            assert_eq!(cmd.status(), MTLCommandBufferStatus::NotEnqueued);
            assert!(bytes(&destination).iter().all(|&byte| byte == 0xa7));
        }
        let cmd = command(device);
        let overlap = Layout {
            source_offset: texture.buffer_offset(),
            source_length: destination.length() - texture.buffer_offset(),
            source_pitch: texture.buffer_stride(),
            ..valid
        };
        let Err(status) = Prepared::encode(device, &cmd, &destination, &texture, overlap) else {
            panic!("alias must refuse")
        };
        assert_eq!(
            status.refusal(),
            Some("metal_guest_seed_overlapping_storage")
        );
        finish(&cmd);
        let Err(status) = Prepared::encode(device, &cmd, &source, &texture, valid) else {
            panic!("completed command must refuse")
        };
        assert_eq!(status.refusal(), Some("metal_guest_seed_command_state"));
    });
}

#[test]
fn gpu_seed_destination_span_arithmetic_refuses_overflow_and_short_storage() {
    assert_eq!(span(3, 2, 16, 4), Some(28));
    assert_eq!(span(3, 2, 11, 4), None);
    assert_eq!(span(3, 2, u64::MAX, 4), None);
    assert_eq!(span(0, 2, 16, 4), None);
    assert_eq!(span(3, 0, 16, 4), None);
    assert_eq!(checked_end(64, 28, 92), Some(92));
    assert_eq!(checked_end(64, 28, 91), None);
    assert_eq!(checked_end(u64::MAX, 1, u64::MAX), None);
}

#[test]
fn gpu_seed_refuses_tiled_targets_and_untracked_native_resources() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let layout = Layout {
            source_offset: 0,
            source_length: 4,
            source_pitch: 4,
            width: 1,
            height: 1,
            source_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
        };
        let source = buffer(device, &[1, 2, 3, 4]);
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_width(1);
        descriptor.set_height(1);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let tiled = raw_metal::new_texture(device, &descriptor).unwrap();
        let cmd = command(device);
        let Err(status) = Prepared::encode(device, &cmd, &source, &tiled, layout) else {
            panic!("tiled target must refuse")
        };
        assert_eq!(status.refusal(), Some("metal_guest_seed_target_not_linear"));
        let (linear, _) = target(device, 1, 1);
        let untracked = raw_metal::new_buffer(
            device,
            4,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .unwrap();
        let Err(status) = Prepared::encode(device, &cmd, &untracked, &linear, layout) else {
            panic!("untracked source must refuse")
        };
        assert_eq!(
            status.refusal(),
            Some("metal_guest_seed_untracked_resource")
        );
        assert_eq!(cmd.status(), MTLCommandBufferStatus::NotEnqueued);
    });
}

#[test]
fn gpu_seed_abandoned_unsubmitted_command_releases_its_native_lease() {
    let held = objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let (texture, _) = target(device, 1, 1);
        let source = buffer(device, &[1, 2, 3, 4]);
        let cmd = command(device);
        let job = Prepared::encode(
            device,
            &cmd,
            &source,
            &texture,
            Layout {
                source_offset: 0,
                source_length: 4,
                source_pitch: 4,
                width: 1,
                height: 1,
                source_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
            },
        )
        .unwrap();
        Arc::downgrade(&job._resources)
    });
    assert!(held.upgrade().is_none());
}

#[test]
fn gpu_seed_holds_native_resources_through_completion_even_if_job_is_dropped() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let (texture, _) = target(device, 1, 1);
        let source = buffer(device, &[1, 2, 3, 4]);
        let cmd = command(device);
        let job = Prepared::encode(
            device,
            &cmd,
            &source,
            &texture,
            Layout {
                source_offset: 0,
                source_length: 4,
                source_pitch: 4,
                width: 1,
                height: 1,
                source_format: pixel_format::MTL_FORMAT_RGBA8_UNORM,
            },
        )
        .unwrap();
        let held = Arc::downgrade(&job._resources);
        drop(job);
        drop(source);
        drop(texture);
        assert!(
            held.upgrade().is_some(),
            "command completion block owns the resources"
        );
        finish(&cmd);
        assert!(
            held.upgrade().is_none(),
            "completion releases the extra native lease"
        );
    });
}

#[test]
fn gpu_seed_private_source_then_render_load_is_ordered_in_the_same_command() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let source_bytes = [30u8, 20, 10, 255].repeat(6);
        let upload = buffer(device, &source_bytes);
        let source =
            raw_metal::new_buffer(device, 24, MTLResourceOptions::StorageModePrivate).unwrap();
        let (texture, destination) = target(device, 3, 2);
        let cmd = command(device);
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&upload, 0, &source, 0, 24);
        blit.end_encoding();
        let _seed = Prepared::encode(
            device,
            &cmd,
            &source,
            &texture,
            Layout {
                source_offset: 0,
                source_length: 24,
                source_pitch: 12,
                width: 3,
                height: 2,
                source_format: pixel_format::MTL_FORMAT_BGRA8_UNORM,
            },
        )
        .unwrap();
        assert!(bytes(&destination).iter().all(|&byte| byte == 0xa7));
        let library = raw_metal::new_library_with_source(
            device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 v(uint i [[vertex_id]]) {
                const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
                return float4(p[i], 0, 1);
            }
            fragment float4 f() { return float4(1.0/255.0, 0, 0, 0); }
        "#,
        )
        .unwrap();
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("v", None).unwrap()));
        descriptor.set_fragment_function(Some(&library.get_function("f", None).unwrap()));
        let attachment = descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        attachment.set_blending_enabled(true);
        attachment.set_source_rgb_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_rgb_blend_factor(MTLBlendFactor::One);
        attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
        attachment.set_destination_alpha_blend_factor(MTLBlendFactor::One);
        let pso = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        let pass = RenderPassDescriptor::new();
        let attachment = pass.color_attachments().object_at(0).unwrap();
        attachment.set_texture(Some(&texture));
        attachment.set_load_action(MTLLoadAction::Load);
        attachment.set_store_action(MTLStoreAction::Store);
        let encoder = raw_metal::new_render_command_encoder(&cmd, pass).unwrap();
        encoder.set_render_pipeline_state(&pso);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        finish(&cmd);
        let actual = bytes(&destination);
        for y in 0..2 {
            let start = texture.buffer_offset() as usize + y * texture.buffer_stride() as usize;
            assert_eq!(&actual[start..start + 12], &[11, 20, 30, 255].repeat(3));
        }
    });
}
