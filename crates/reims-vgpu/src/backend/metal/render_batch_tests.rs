use super::*;

#[test]
fn submitted_render_retains_completion_and_drop_waits_without_resubmitting() {
    objc::rc::autoreleasepool(|| {
        let device = system_device().unwrap();
        for explicit_finish in [false, true] {
            let target = new_color_target(
                device, MTLPixelFormat::RGBA8Unorm, 4, 4, MTLStorageMode::Shared,
            ).unwrap();
            let pass = RenderPassDescriptor::new();
            let color = pass.color_attachments().object_at(0).unwrap();
            color.set_texture(Some(&target));
            color.set_load_action(MTLLoadAction::Clear);
            color.set_clear_color(MTLClearColor::new(1.0, 0.0, 0.0, 1.0));
            color.set_store_action(MTLStoreAction::Store);
            let mut batch = RenderBatch::default();
            batch.encoder(device, pass).unwrap();
            let command = batch.command.as_ref().unwrap().clone();
            let mut submitted = batch.submit().unwrap();
            assert!(!batch.pending());
            assert_eq!(batch.submissions, 1);
            assert!(submitted.0.begin_commands(device).is_err());
            if explicit_finish {
                submitted.finish().unwrap();
            } else {
                drop(submitted);
            }
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let mut output = [0u8; 64];
            target.get_bytes(output.as_mut_ptr().cast(), 16, MTLRegion::new_2d(0, 0, 4, 4), 0);
            assert_eq!(output, [255, 0, 0, 255].repeat(16).as_slice());
        }
    });
}

#[test]
fn setup_and_render_share_one_completed_submission() {
    objc::rc::autoreleasepool(|| {
        let device = system_device().expect("Metal device");
        let buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        unsafe { buffer.contents().cast::<u8>().write_bytes(0, 16) };
        let mut batch = RenderBatch::default();
        let command = batch.begin_commands(device).unwrap();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(&buffer, NSRange::new(0, 16), 0xa7);
        blit.end_encoding();
        assert_eq!(batch.submissions, 0);

        let target = new_color_target(
            device,
            MTLPixelFormat::RGBA8Unorm,
            4,
            4,
            MTLStorageMode::Shared,
        )
        .unwrap();
        let pass = RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(&target));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(1.0, 0.0, 0.0, 1.0));
        color.set_store_action(MTLStoreAction::Store);
        batch.encoder(device, pass).unwrap();
        assert_eq!(batch.command.as_ref().unwrap().as_ptr(), command.as_ptr());
        assert!(batch.begin_commands(device).is_err());
        batch.finish((ptr::null_mut(), 0)).unwrap();
        assert_eq!(batch.submissions, 1);
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), 16) };
        assert_eq!(bytes, [0xa7; 16]);
        let mut rgba = [0u8; 64];
        target.get_bytes(
            rgba.as_mut_ptr().cast(),
            16,
            MTLRegion::new_2d(0, 0, 4, 4),
            0,
        );
        assert!(rgba
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [255, 0, 0, 255]));
    });
}

#[test]
fn abandoned_setup_is_not_submitted_by_finish_or_drop() {
    objc::rc::autoreleasepool(|| {
        let device = system_device().expect("Metal device");
        for explicit_finish in [false, true] {
            let buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
            unsafe { buffer.contents().cast::<u8>().write_bytes(0, 16) };
            let mut batch = RenderBatch::default();
            let command = batch.begin_commands(device).unwrap();
            let blit = command.new_blit_command_encoder();
            blit.fill_buffer(&buffer, NSRange::new(0, 16), 0xa7);
            blit.end_encoding();
            if explicit_finish {
                batch.finish((ptr::null_mut(), 0)).unwrap();
                assert_eq!(batch.submissions, 0);
            }
            drop(batch);
            assert_eq!(command.status(), MTLCommandBufferStatus::NotEnqueued);
            let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), 16) };
            assert_eq!(bytes, [0; 16]);
        }
    });
}
