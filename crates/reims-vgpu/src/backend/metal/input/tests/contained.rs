use super::*;
use crate::backend::metal::buffer_extent::{self, Stage};

#[test]
fn contained_native_views_preserve_size_offset_and_gpu_output_for_both_placements() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        if !device.supports_family(metal::MTLGPUFamily::Apple2) {
            return;
        }
        let pool = runtime::thread_input_pool(device);
        let proof = buffer_extent::tests::unbounded_readonly()
            .capture_for(Class::Fragment, 1)
            .unwrap()
            .1
            .unwrap();
        let narrow = buffer_extent::tests::object(Stage::Fragment, 1, 4)
            .capture_for(Class::Fragment, 1)
            .unwrap()
            .1
            .unwrap();
        let library = raw_metal::new_library_with_source(
            device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 contained_v(uint i [[vertex_id]]) {
                const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
                return float4(p[i], 0, 1);
            }
            fragment float4 contained_f(constant uint &size [[buffer(0)]],
                                         device const uchar *data [[buffer(1)]]) {
                return float4(float(data[0]), float(data[size-1]), float(size), 255.0) / 255.0;
            }
            fragment float4 contained_short(constant uint &size [[buffer(0)]],
                                             constant uchar4 &data [[buffer(1)]]) {
                return float4(float(data.x), float(data.w), float(size), 255.0) / 255.0;
            }
        "#,
        )
        .unwrap();
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("contained_v", None).unwrap()));
        descriptor.set_fragment_function(Some(&library.get_function("contained_f", None).unwrap()));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        let pipeline = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        descriptor.set_fragment_function(Some(
            &library.get_function("contained_short", None).unwrap(),
        ));
        let short_pipeline = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        for compact in [false, true] {
            pool.borrow_mut().clear_available();
            pool.borrow_mut().shape_headroom = compact.then_some(0);
            let before = snapshot(Class::Fragment);
            let image = fill_read_only_resource_prefix::<()>(
                device,
                128,
                4,
                proof,
                Class::Fragment,
                "test_contained_allocation",
                |bytes| {
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        *byte = (i + 4) as u8;
                    }
                    Ok(bytes.len())
                },
            )
            .unwrap()
            .freeze()
            .unwrap();
            assert_eq!(
                image.allocation.buffer.length(),
                if compact { 124 } else { 128 }
            );
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_width(4);
            descriptor.set_height(1);
            descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::RenderTarget);
            let texture = raw_metal::new_texture(device, &descriptor).unwrap();
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            let pass = RenderPassDescriptor::new();
            let attachment = pass.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(&texture));
            attachment.set_load_action(MTLLoadAction::Clear);
            attachment.set_store_action(MTLStoreAction::Store);
            let encoder = raw_metal::new_render_command_encoder(&cmd, pass).unwrap();
            encoder.set_viewport(MTLViewport {
                originX: 0.,
                originY: 0.,
                width: 4.,
                height: 1.,
                znear: 0.,
                zfar: 1.,
            });
            let mut sizes = Vec::new();
            let mut expected = Vec::new();
            for (x, delta) in [0, 4, 8, 12].into_iter().enumerate() {
                let short = x % 2 == 1;
                encoder.set_render_pipeline_state(if short { &short_pipeline } else { &pipeline });
                let input = image
                    .bind_window(device, if short { narrow } else { proof }, delta)
                    .unwrap();
                assert_eq!(input.len(), 124 - delta);
                assert_eq!(input.captured_len(), if short { 4 } else { 124 - delta });
                let offset = input.binding_offset();
                let native = submission.seal(input).unwrap();
                // Forward the actual native length-minus-offset, not an
                // expected test length. GPU output and last-byte access test it.
                let size = (native.length() - offset) as u32;
                let size_buffer = device.new_buffer_with_data(
                    (&size as *const u32).cast(),
                    4,
                    MTLResourceOptions::StorageModeShared,
                );
                encoder.set_fragment_buffer(0, Some(&size_buffer), 0);
                sizes.push(size_buffer);
                encoder.set_fragment_buffer(1, Some(&native), offset);
                encoder.set_scissor_rect(MTLScissorRect {
                    x: x as u64,
                    y: 0,
                    width: 1,
                    height: 1,
                });
                encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
                expected.extend([
                    4 + delta as u8,
                    if short { 7 + delta as u8 } else { 127 },
                    (124 - delta) as u8,
                    255,
                ]);
            }
            assert_eq!(submission.snapshots.len(), 1);
            encoder.end_encoding();
            finish(&mut submission, &cmd);
            let mut pixels = [0; 16];
            texture.get_bytes(
                pixels.as_mut_ptr().cast(),
                16,
                MTLRegion::new_2d(0, 0, 4, 1),
                0,
            );
            assert_eq!(pixels.as_slice(), expected);
            assert_eq!(
                snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
                124
            );
            assert_eq!(
                snapshot(Class::Fragment).allocations - before.allocations,
                1
            );
            drop(sizes);
            image.retire(device);
        }
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn contained_native_views_preserve_dirty_coverage_and_wait_for_last_completion() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        if !device.supports_family(metal::MTLGPUFamily::Apple2) {
            return;
        }
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let wide = buffer_extent::tests::object(Stage::Fragment, 0, 32)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1
            .unwrap();
        let narrow = buffer_extent::tests::object(Stage::Fragment, 0, 4)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1
            .unwrap();
        let image = fill_read_only_resource_prefix::<()>(
            device,
            64,
            0,
            wide,
            Class::Fragment,
            "test_contained_allocation",
            |bytes| {
                bytes.fill(0x72);
                Ok(bytes.len())
            },
        )
        .unwrap()
        .freeze()
        .unwrap();
        let view = image.bind_window(device, narrow, 16).unwrap();
        assert_eq!(view.len(), 48);
        assert_eq!(view.captured_len(), 4);
        assert_eq!(view.allocation.dirty, 0..32);
        assert!(image
            .bind_window(device, narrow, 16)
            .unwrap()
            .freeze()
            .is_err());
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        submission.seal(view).unwrap();
        let weak = Arc::downgrade(&image);
        image.retire(device);
        assert!(weak.upgrade().is_some());
        assert!(pool.borrow().inventory().is_empty());
        assert!(submission.completed(&cmd).is_err());
        finish(&mut submission, &cmd);
        assert!(weak.upgrade().is_none());
        let before = snapshot(Class::Fragment);
        let fresh = fill_resource_prefix::<()>(
            device,
            64,
            0,
            4,
            Class::Fragment,
            "test_contained_allocation",
            |bytes| {
                bytes.fill(0x53);
                Ok(bytes.len())
            },
        )
        .unwrap();
        let bytes = fresh.test_contents();
        assert_eq!(&bytes[..4], &[0x53; 4]);
        assert!(bytes[4..].iter().all(|&b| b == 0));
        assert_eq!(
            snapshot(Class::Fragment).direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            28
        );
        drop(fresh);
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn contained_native_views_reject_unknown_coverage_overflow_and_unsupported_offsets() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let bounded = buffer_extent::tests::object(Stage::Fragment, 0, 16)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1
            .unwrap();
        let narrow = buffer_extent::tests::object(Stage::Fragment, 0, 4)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1
            .unwrap();
        let full = buffer_extent::tests::unbounded_readonly()
            .capture_for(Class::Fragment, 1)
            .unwrap()
            .1
            .unwrap();
        let image = fill_read_only_resource_prefix::<()>(
            device,
            64,
            0,
            bounded,
            Class::Fragment,
            "test_contained_allocation",
            |bytes| {
                bytes.fill(0x72);
                Ok(bytes.len())
            },
        )
        .unwrap()
        .freeze()
        .unwrap();
        assert!(image.bind_window(device, full, 0).is_none());
        assert!(image.bind_window(device, narrow, 16).is_none());
        assert!(image.bind_window(device, narrow, usize::MAX).is_none());
        assert!(image.bind_window(device, narrow, 1).is_none());
        assert!(image.bind_window(device, narrow, 0).is_some());
    });
}
