use super::*;
use crate::backend::blob::BlobKey;
use crate::backend::metal::abi::ReimsVgpuBlendState;
use crate::backend::metal::render::{fill_render_pso_key, get_render_pipeline_state};
use crate::backend::render_pso_key::RenderPsoLookup;

const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 r32_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    struct Colors { float4 output [[color(0)]]; float4 tile [[color(1)]]; };
    fragment Colors r32_fragment(float4 prior [[color(1)]],
                                  constant float &increment [[buffer(0)]]) {
        float next = prior.r + increment;
        return {float4(next * 0.25, prior.g + 0.125, prior.b + 0.0625, prior.a * 0.25),
                float4(next, 9.0, -3.0, 0.25)};
    }
"#;

fn source_over() -> ReimsVgpuBlendState {
    ReimsVgpuBlendState {
        enable: 1,
        src_rgb: MTLBlendFactor::One as u32,
        dst_rgb: MTLBlendFactor::OneMinusSourceAlpha as u32,
        op_rgb: MTLBlendOperation::Add as u32,
        src_alpha: MTLBlendFactor::One as u32,
        dst_alpha: MTLBlendFactor::OneMinusSourceAlpha as u32,
        op_alpha: MTLBlendOperation::Add as u32,
        has_blend_color: 0,
        blend_color: [0.0; 4],
    }
}

fn color(slot: u32, format: MTLPixelFormat, target: ColorTarget<'_>) -> ColorRt<'_> {
    ColorRt {
        slot,
        pixel_format: format as u32,
        seed_rgba8: None,
        out_rgba8: None,
        clear_r: 2.0000009536743164,
        clear_g: 8.0,
        clear_b: -7.0,
        clear_a: 0.125,
        load_action: u32::from(MTL_LOAD_ACTION_CLEAR),
        blend: None,
        write_mask: 0xf,
        target,
    }
}

#[test]
fn memoryless_secondary_blending_is_independent_of_color_zero() {
    let blend = source_over();
    let secondary = color(1, MTLPixelFormat::R32Float, ColorTarget::Guest(None));
    let key = fill_render_pso_key(&[], Some(&blend),
        &[secondary.pipeline_key(secondary.pixel_format)], 0, 0);
    assert_eq!(key.color_blend_enable[0], 0,
        "a secondary coverage attachment must not inherit colour0's blend state");
    let mut primary = color(0, MTLPixelFormat::RGBA32Float, ColorTarget::Guest(None));
    let key = fill_render_pso_key(&[], Some(&blend),
        &[primary.pipeline_key(primary.pixel_format)], 0, 0);
    assert_eq!(key.color_blend_enable[0], 1, "legacy colour0 fallback is preserved");
    primary.blend = Some(ReimsVgpuBlendState { enable: 0, ..blend });
    let key = fill_render_pso_key(&[], Some(&blend),
        &[primary.pipeline_key(primary.pixel_format)], 0, 0);
    assert_eq!(key.color_blend_enable[0], 0, "an explicit disabled state overrides the fallback");
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn memoryless_r32float_still_refuses_the_vulkan_rail() {
    let mut req = request();
    req.colors[0].format = MTLPixelFormat::R32Float as u16;
    let mut state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    let mut host = crate::runtime::host::FakeHost::new();
    let result = crate::runtime::draw::vulkan::encode_draw_chain(
        &mut state, &mut host, &mut req, true, false,
    );
    assert!(matches!(result.0, EncodeStatus::BadArgs("draw_vk_memoryless_unsupported")));
    assert!(result.1.is_none());
}

#[test]
fn memoryless_r32float_native_clear_fetch_and_blend_match_one_encoder() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else { return; };
        if !device.supports_family(MTLGPUFamily::Apple2) { return; }
        let library = crate::backend::metal::raw_metal::new_library_with_source(device, SOURCE).unwrap();
        let vertex = library.get_function("r32_vertex", None).unwrap();
        let fragment = library.get_function("r32_fragment", None).unwrap();
        let queue = device.new_command_queue();
        let increments = [0.25f32, -3.0, 0.5];
        let scissors = [
            MTLScissorRect { x: 0, y: 0, width: 8, height: 4 },
            MTLScissorRect { x: 2, y: 0, width: 6, height: 4 },
            MTLScissorRect { x: 0, y: 1, width: 5, height: 2 },
        ];
        let draw = |encoder: &RenderCommandEncoderRef, pipeline: &RenderPipelineStateRef, index: usize| {
            encoder.set_render_pipeline_state(pipeline);
            encoder.set_fragment_bytes(0, 4, (&increments[index] as *const f32).cast());
            encoder.set_scissor_rect(scissors[index]);
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        };
        let output_attachment = |pass: &RenderPassDescriptorRef, output: &TextureRef, initial: bool| {
            let attachment = pass.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(output));
            attachment.set_load_action(if initial { MTLLoadAction::Clear } else { MTLLoadAction::Load });
            attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
            attachment.set_store_action(MTLStoreAction::Store);
        };
        let read = |texture: &TextureRef| {
            let mut bits = vec![0u32; 8 * 4 * 4];
            texture.get_bytes(bits.as_mut_ptr().cast(), 8 * 16, MTLRegion::new_2d(0, 0, 8, 4), 0);
            bits
        };

        for blend_tile in [false, true] {
            let native_output = new_color_target(device, MTLPixelFormat::RGBA32Float, 8, 4,
                MTLStorageMode::Shared).unwrap();
            let native_tile = new_color_target(device, MTLPixelFormat::R32Float, 8, 4,
                MTLStorageMode::Memoryless).unwrap();
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(&vertex));
            descriptor.set_fragment_function(Some(&fragment));
            for (slot, format, enabled) in [
                (0, MTLPixelFormat::RGBA32Float, true),
                (1, MTLPixelFormat::R32Float, blend_tile),
            ] {
                let attachment = descriptor.color_attachments().object_at(slot).unwrap();
                attachment.set_pixel_format(format);
                attachment.set_blending_enabled(enabled);
                attachment.set_source_rgb_blend_factor(MTLBlendFactor::One);
                attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
                attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
                attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
            }
            let native_pipeline = device.new_render_pipeline_state(&descriptor).unwrap();
            let pass = RenderPassDescriptor::new();
            output_attachment(pass, &native_output, true);
            let tile = pass.color_attachments().object_at(1).unwrap();
            tile.set_texture(Some(&native_tile));
            tile.set_load_action(MTLLoadAction::Clear);
            tile.set_clear_color(MTLClearColor::new(2.0000009536743164, 8.0, -7.0, 0.125));
            tile.set_store_action(MTLStoreAction::DontCare);
            let command = queue.new_command_buffer();
            let encoder = command.new_render_command_encoder(pass);
            for index in 0..3 { draw(encoder, &native_pipeline, index); }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let expected = read(&native_output);
            assert_eq!(f32::from_bits(expected[0]), (2.0000009536743164f32 + 0.25) * 0.25,
                "R32Float clear/fetch must retain precision beyond half-float or UNORM");
            assert_eq!(&expected[1..4], &[0.125f32.to_bits(), 0.0625f32.to_bits(), 0.25f32.to_bits()],
                "a single-channel framebuffer fetch supplies G=0, B=0 and A=1");

            let output = new_color_target(device, MTLPixelFormat::RGBA32Float, 8, 4,
                MTLStorageMode::Shared).unwrap();
            let mut guest_pass = MetalRenderPass::default();
            let mut req = request();
            req.colors[0].format = MTLPixelFormat::R32Float as u16;
            let blend = source_over();
            let mut lifetime = None;
            for index in 0..3 {
                req.continues_render_pass = index != 0;
                req.render_pass_continues = index != 2;
                req.colors[0].load_action = if index == 0 { MTL_LOAD_ACTION_CLEAR } else { MTL_LOAD_ACTION_LOAD };
                let result = guest_pass.with_draw(&mut req, |owner, req| {
                    let target = owner.target(&req.colors[0]).unwrap();
                    if index == 0 { lifetime = Some(weak_texture(target.texture())); }
                    assert_eq!(target.texture().pixel_format(), MTLPixelFormat::R32Float);
                    let mut tile = color(1, MTLPixelFormat::R32Float, ColorTarget::Memoryless(target));
                    tile.load_action = u32::from(req.colors[0].load_action);
                    tile.blend = blend_tile.then_some(blend);
                    let primary = color(0, MTLPixelFormat::RGBA32Float, ColorTarget::Guest(None));
                    let key = fill_render_pso_key(&[], Some(&blend), &[
                        primary.pipeline_key(primary.pixel_format), tile.pipeline_key(tile.pixel_format),
                    ], 0, 0);
                    let lookup = RenderPsoLookup {
                        desc: &key,
                        vert: BlobKey::new(SOURCE.as_bytes()),
                        frag: BlobKey::new(SOURCE.as_bytes()),
                    };
                    let (pipeline, _, _, _) = get_render_pipeline_state(
                        device, &vertex, &fragment, None, &lookup, (std::ptr::null_mut(), 0),
                    ).unwrap();
                    let pass = RenderPassDescriptor::new();
                    output_attachment(pass, &output, index == 0);
                    tile.attach(pass, target.texture());
                    let command = queue.new_command_buffer();
                    let encoder = command.new_render_command_encoder(pass);
                    draw(encoder, &pipeline, index);
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    (EncodeStatus::Ok, None)
                });
                assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
                assert_eq!(lifetime.as_ref().unwrap().load().is_null(), index == 2);
            }
            assert_eq!(read(&output), expected,
                "split native R32Float pass differs, secondary blending={blend_tile}");
            assert_eq!(guest_pass.phase, Phase::Finished);
            assert!(guest_pass.targets.is_empty());
        }
    });
}
