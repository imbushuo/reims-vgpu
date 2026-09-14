use super::*;
use crate::backend::metal::buffer_extent::{self, Stage};
use crate::runtime::draw::metal::inputs::{prepare, Capture, PreparedInput};
use crate::runtime::host::HostMemory;

#[test]
fn readonly_snapshot_completed_commands_refresh_inline_colors_before_dirty_harvest() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let library = raw_metal::new_library_with_source(
            device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 v(uint i [[vertex_id]]) {
                const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
                return float4(p[i], 0, 1);
            }
            fragment float4 f(constant float4 &color [[buffer(0)]]) { return color; }
        "#,
        )
        .unwrap();
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("v", None).unwrap()));
        descriptor.set_fragment_function(Some(&library.get_function("f", None).unwrap()));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let pipeline = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_width(1);
        descriptor.set_height(1);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = raw_metal::new_texture(device, &descriptor).unwrap();
        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 96, 32);
        let access = buffer_extent::tests::object(Stage::Fragment, 0, 16);
        let before = snapshot(Class::Fragment);
        for value in 0..256u32 {
            let scope = crate::runtime::draw::BufferSnapshotScope::new();
            // The preceding command has actually completed. CPU writes now are
            // legal; the host's harvest-based generation can still be unchanged.
            let red = value as u8;
            let green = (255 - value) as u8;
            let blue = (value * 13) as u8;
            let color = [red, green, blue, 255].map(|c| f32::from(c) / 255.);
            let bytes: Vec<_> = color.into_iter().flat_map(f32::to_ne_bytes).collect();
            fixture
                .host
                .write_gpa(8 * fixture.page + 32, &bytes)
                .unwrap();
            let PreparedInput::Native(input) = prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Class::Fragment,
                Capture::Native(Some(access)).in_scope(Some(&scope.reference())),
                "test_inline_color",
            )
            .unwrap() else {
                panic!("native input")
            };
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            let offset = input.bytes.binding_offset();
            let buffer = submission.seal(input.bytes).unwrap();
            let pass = RenderPassDescriptor::new();
            let attachment = pass.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(&texture));
            attachment.set_load_action(MTLLoadAction::Clear);
            attachment.set_store_action(MTLStoreAction::Store);
            let encoder = raw_metal::new_render_command_encoder(&cmd, pass).unwrap();
            encoder.set_render_pipeline_state(&pipeline);
            encoder.set_fragment_buffer(0, Some(&buffer), offset);
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
            finish(&mut submission, &cmd);
            let mut actual = [0; 4];
            texture.get_bytes(
                actual.as_mut_ptr().cast(),
                4,
                MTLRegion::new_2d(0, 0, 1, 1),
                0,
            );
            assert_eq!(actual, [blue, green, red, 255], "completed round={value}");
        }
        assert_eq!(
            snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
            256 * 16
        );
    });
}
