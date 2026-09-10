use super::*;
use crate::backend::metal::render::{ColorRt, ColorTarget};
use metal::*;

mod r32float;

fn weak_texture(texture: &Texture) -> objc::rc::WeakPtr {
    use foreign_types::ForeignType;
    // Balance the temporary retain; the returned weak reference owns no texture.
    unsafe { objc::rc::StrongPtr::retain(texture.as_ptr().cast()).weak() }
}

fn request() -> DrawEncodeRequest {
    DrawEncodeRequest {
        task_id: 7,
        vertex_count: 3,
        instance_count: 1,
        primitive_type: 3,
        render_pass_continues: true,
        colors: vec![ColorRtRequest {
            storage: ColorStorage::Memoryless,
            slot: 1,
            texture_ref: 37,
            width: 8,
            height: 4,
            format: MTLPixelFormat::RGBA16Float as u16,
            sample_count: 1,
            load_action: MTL_LOAD_ACTION_CLEAR,
            clear_color: [2.0, -0.5, 0.0009765625, 0.5],
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn memoryless_pass_requires_its_owner_and_no_external_content() {
    let mut req = request();
    req.continues_render_pass = true;
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_render_pass_sequence"));
    req.continues_render_pass = false;
    req.colors[0].sample_count = 4;
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_memoryless_multisample"));
    req.colors[0].sample_count = 1;
    req.colors[0].target_gva = 0x4000;
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_memoryless_backing"));
    req.colors[0].target_gva = 0;
    req.colors[0].target_seed_rgba = Some(vec![0; 4]);
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_memoryless_backing"));
    req.colors[0].target_seed_rgba = None;
    req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_memoryless_pass_load"));
    req.colors[0].load_action = MTL_LOAD_ACTION_CLEAR;
    let mut alias = req.colors[0].clone();
    alias.slot = 2;
    req.colors.push(alias);
    assert_eq!(MetalRenderPass::default().prepare(&req), Err("draw_mtl_memoryless_attachment_alias"));
}

#[test]
fn memoryless_pass_identity_is_fixed_and_end_retires_its_contents() {
    if super::super::runtime::system_device().is_none() { return; }
    let mut pass = MetalRenderPass::default();
    let mut req = request();
    pass.prepare(&req).unwrap();
    assert!(!pass.target(&req.colors[0]).unwrap().initialized());
    assert_eq!(pass.target(&req.colors[0]).unwrap().texture().storage_mode(), MTLStorageMode::Private);
    pass.completed(&mut req, &mut None);
    req.continues_render_pass = true;
    req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
    pass.prepare(&req).unwrap();
    assert!(pass.target(&req.colors[0]).unwrap().initialized());
    req.colors[0].texture_ref += 1;
    assert_eq!(pass.prepare(&req), Err("draw_mtl_memoryless_pass_attachment_changed"));
    req.colors[0].texture_ref -= 1;
    req.colors[0].format = MTLPixelFormat::RGBA8Unorm as u16;
    assert_eq!(pass.prepare(&req), Err("draw_mtl_memoryless_pass_attachment_changed"));
    req.render_pass_continues = false;
    pass.completed(&mut req, &mut None);
    assert!(pass.targets.is_empty());
    assert_eq!(pass.prepare(&request()), Err("draw_mtl_render_pass_sequence"));
    let mut fresh = MetalRenderPass::default();
    fresh.prepare(&request()).unwrap();
    assert!(!fresh.targets[0].initialized());
}

#[test]
fn memoryless_failed_draw_poisoning_retires_pass_storage() {
    if super::super::runtime::system_device().is_none() { return; }
    let mut pass = MetalRenderPass::default();
    let mut state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    let mut host = crate::runtime::host::FakeHost::new();
    let mut req = request();
    let result = pass.encode_draw(&mut state, &mut host, &mut req, false, false);
    assert!(!matches!(result.0, EncodeStatus::Ok), "the fixture has no pipeline");
    assert_eq!(pass.phase, Phase::Failed);
    assert!(pass.targets.is_empty());
    assert_eq!(pass.prepare(&req), Err("draw_mtl_render_pass_sequence"));
}

#[test]
fn memoryless_refusal_releases_autoreleased_attachment_references() {
    if super::super::runtime::system_device().is_none() { return; }
    let mut pass = MetalRenderPass::default();
    let mut req = request();
    let mut lifetime = None;
    let result = pass.with_draw(&mut req, |pass, req| {
        let target = pass.target(&req.colors[0]).unwrap();
        lifetime = Some(weak_texture(target.texture()));
        let descriptor = RenderPassDescriptor::new();
        descriptor.color_attachments().object_at(1).unwrap()
            .set_texture(Some(target.texture()));
        (EncodeStatus::BadArgs("test_encode_refusal"), None)
    });
    assert!(matches!(result.0, EncodeStatus::BadArgs("test_encode_refusal")));
    assert_eq!(pass.phase, Phase::Failed);
    assert!(pass.targets.is_empty());
    assert!(lifetime.unwrap().load().is_null(),
        "clearing the Rust owner is insufficient if an autoreleased descriptor still retains it");
}

#[test]
fn memoryless_color_zero_chains_only_inside_its_pass() {
    if super::super::runtime::system_device().is_none() { return; }
    let mut pass = MetalRenderPass::default();
    let mut req = request();
    req.colors[0].slot = 0;
    pass.prepare(&req).unwrap();
    let mut output = Some(Vec::new());
    pass.completed(&mut req, &mut output);
    assert!(output.is_none(), "memoryless colour0 is not an empty CPU seed");
    assert!(req.chain_resident_established);
    req.continues_render_pass = true;
    req.render_pass_continues = false;
    req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
    pass.prepare(&req).unwrap();
    pass.completed(&mut req, &mut output);
    assert!(!req.chain_resident_established);
    assert!(output.is_none());
    assert!(pass.targets.is_empty());
}

#[test]
fn memoryless_multi_draw_framebuffer_fetch_matches_one_native_encoder() {
    let Some(device) = super::super::runtime::system_device() else { return; };
    if !device.supports_family(MTLGPUFamily::Apple1) {
        eprintln!("native memoryless oracle requires Apple GPU support");
        return;
    }
    let library = super::super::raw_metal::new_library_with_source(device, r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 pass_vertex(uint i [[vertex_id]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        struct Colors { half4 output [[color(0)]]; half4 tile [[color(1)]]; };
        fragment Colors pass_fragment(half4 prior [[color(1)]],
                                       constant float4 &increment [[buffer(0)]]) {
            half4 next = prior + half4(increment);
            return {next, next};
        }
    "#).unwrap();
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&library.get_function("pass_vertex", None).unwrap()));
    descriptor.set_fragment_function(Some(&library.get_function("pass_fragment", None).unwrap()));
    for slot in 0..2 {
        descriptor.color_attachments().object_at(slot).unwrap()
            .set_pixel_format(MTLPixelFormat::RGBA16Float);
    }
    let pipeline = device.new_render_pipeline_state(&descriptor).unwrap();
    let queue = device.new_command_queue();
    let increments = [
        [0.125f32, -0.25, 0.001953125, 0.125],
        [0.0625, 0.125, 0.0009765625, 0.125],
        [-0.03125, 0.25, 0.00390625, -0.0625],
    ];
    let scissors = [
        MTLScissorRect { x: 0, y: 0, width: 8, height: 4 },
        MTLScissorRect { x: 2, y: 0, width: 6, height: 4 },
        MTLScissorRect { x: 0, y: 1, width: 5, height: 2 },
    ];
    let draw = |encoder: &RenderCommandEncoderRef, index: usize| {
        encoder.set_render_pipeline_state(&pipeline);
        encoder.set_fragment_bytes(0, 16, increments[index].as_ptr().cast());
        encoder.set_scissor_rect(scissors[index]);
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
    };
    let descriptor_for = |output: &TextureRef, tile: &TextureRef, initial: bool| {
        let descriptor = RenderPassDescriptor::new();
        let color = descriptor.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(output));
        color.set_load_action(if initial { MTLLoadAction::Clear } else { MTLLoadAction::Load });
        color.set_store_action(MTLStoreAction::Store);
        let attachment = descriptor.color_attachments().object_at(1).unwrap();
        attachment.set_texture(Some(tile));
        attachment.set_load_action(if initial { MTLLoadAction::Clear } else { MTLLoadAction::Load });
        attachment.set_clear_color(MTLClearColor::new(2.0, -0.5, 0.0009765625, 0.5));
        descriptor
    };
    let read_output = |output: &TextureRef| {
        let mut pixels = vec![0u16; 8 * 4 * 4];
        output.get_bytes(pixels.as_mut_ptr().cast(), 8 * 8, MTLRegion::new_2d(0, 0, 8, 4), 0);
        pixels
    };

    let native_output = new_color_target(device, MTLPixelFormat::RGBA16Float, 8, 4,
        MTLStorageMode::Shared).unwrap();
    let native_tile = new_color_target(device, MTLPixelFormat::RGBA16Float, 8, 4,
        MTLStorageMode::Memoryless).unwrap();
    let native_pass = descriptor_for(&native_output, &native_tile, true);
    native_pass.color_attachments().object_at(1).unwrap().set_store_action(MTLStoreAction::DontCare);
    let command = queue.new_command_buffer();
    let encoder = command.new_render_command_encoder(&native_pass);
    for index in 0..3 { draw(encoder, index); }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let expected = read_output(&native_output);
    assert_ne!(&expected[..4], &expected[3 * 4..4 * 4], "partial draws must differ");
    assert!(expected.iter().any(|&bits| bits == 0xba00), "retain negative half-float values");

    let output = new_color_target(device, MTLPixelFormat::RGBA16Float, 8, 4,
        MTLStorageMode::Shared).unwrap();
    let mut guest_pass = MetalRenderPass::default();
    let mut req = request();
    let mut tile_lifetime = None;
    for index in 0..3 {
        req.continues_render_pass = index != 0;
        req.render_pass_continues = index != 2;
        req.colors[0].load_action = if index == 0 { MTL_LOAD_ACTION_CLEAR } else { MTL_LOAD_ACTION_LOAD };
        let result = guest_pass.with_draw(&mut req, |pass, req| {
            let target = pass.target(&req.colors[0]).unwrap();
            if index == 0 {
                tile_lifetime = Some(weak_texture(target.texture()));
            }
            let initial = !target.initialized();
            let descriptor = descriptor_for(&output, target.texture(), initial);
            let color = &req.colors[0];
            ColorRt {
                slot: color.slot,
                pixel_format: u32::from(color.format),
                seed_rgba8: None,
                out_rgba8: None,
                clear_r: color.clear_color[0],
                clear_g: color.clear_color[1],
                clear_b: color.clear_color[2],
                clear_a: color.clear_color[3],
                load_action: u32::from(color.load_action),
                blend: None,
                write_mask: 0xf,
                target: ColorTarget::Memoryless(target),
            }.attach(&descriptor, target.texture());
            assert_eq!(descriptor.color_attachments().object_at(1).unwrap().load_action() as u64,
                if initial { MTLLoadAction::Clear as u64 } else { MTLLoadAction::Load as u64 });
            let command = queue.new_command_buffer();
            let encoder = command.new_render_command_encoder(&descriptor);
            draw(encoder, index);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            (EncodeStatus::Ok, None)
        });
        assert!(matches!(result.0, EncodeStatus::Ok));
        assert_eq!(tile_lifetime.as_ref().unwrap().load().is_null(), index == 2,
            "native target must survive between draws, but not its completed guest pass");
    }
    assert_eq!(read_output(&output), expected, "split passes must preserve every half-float texel");
    assert!(guest_pass.targets.is_empty(), "no texture survives the guest pass");
    assert_eq!(guest_pass.phase, Phase::Finished);
}
