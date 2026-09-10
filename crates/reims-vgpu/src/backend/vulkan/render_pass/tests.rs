use super::*;
use crate::protocol::pixel_format::{MTL_FORMAT_R32_FLOAT, MTL_FORMAT_RGBA16_FLOAT};

mod gpu;

#[test]
fn memoryless_secondary_lowering_keeps_native_clear_load_and_independent_blend() {
    use crate::runtime::decode::resource::{PipelineColorAttachment, RenderPipelineDescriptor};
    let mut req = request();
    let mut secondary = req.colors[0].clone();
    secondary.slot = 1;
    secondary.texture_ref = 38;
    req.colors.push(secondary);
    let mut owner = VulkanRenderPass::default();
    owner.prepare(&req).unwrap();
    let mut state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    let mut host = crate::runtime::host::FakeHost::new();
    for blend in [false, true] {
        let pipeline = RenderPipelineDescriptor {
            color_attachments: vec![
                PipelineColorAttachment { slot: 0, blending_enabled: true, src_rgb: 4, dst_rgb: 5,
                    ..Default::default() },
                PipelineColorAttachment { slot: 1, blending_enabled: blend, src_rgb: 1, dst_rgb: 1,
                    ..Default::default() },
            ], ..Default::default()
        };
        for load in [MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_LOAD] {
            req.colors[1].load_action = load;
            let secondary = crate::runtime::draw::vulkan::build_secondary_targets_in_pass(
                &mut state, &mut host, req.task_id, &req.colors, &pipeline,
                owner.identity(&req.colors[0]).unwrap(), 8, 4, &owner,
            ).unwrap().remove(0);
            assert_eq!(secondary.identity, *owner.identity(&req.colors[1]).unwrap());
            assert_eq!(secondary.attachment.format(), ash::vk::Format::R32_SFLOAT);
            assert_eq!(secondary.attachment.clear(),
                crate::backend::vulkan::engine::ColorClearValue::Float(
                    [2.0000009536743164f32, 8.0, -7.0, 0.125]));
            assert_eq!(secondary.load, load == MTL_LOAD_ACTION_LOAD);
            assert_eq!(secondary.blend.is_some(), blend);
            if let Some(blend) = secondary.blend {
                assert_eq!((blend.src_rgb, blend.dst_rgb), (1, 1));
            }
        }
    }
}

fn request() -> DrawEncodeRequest {
    DrawEncodeRequest {
        task_id: 7,
        render_pass_continues: true,
        colors: vec![ColorRtRequest {
            storage: ColorStorage::Memoryless, slot: 0, texture_ref: 37,
            width: 8, height: 4, format: MTL_FORMAT_R32_FLOAT, sample_count: 1,
            load_action: MTL_LOAD_ACTION_CLEAR,
            clear_color: [2.0000009536743164, 8.0, -7.0, 0.125],
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn memoryless_native_format_survives_split_draws_without_guest_backing() {
    let mut pass = VulkanRenderPass::default();
    let mut req = request();
    let mut identity = None;
    for index in 0..3 {
        req.continues_render_pass = index != 0;
        req.render_pass_continues = index != 2;
        req.colors[0].load_action = if index == 0 { MTL_LOAD_ACTION_CLEAR } else { MTL_LOAD_ACTION_LOAD };
        let result = pass.with_draw(&mut req, |owner, req| {
            let current = owner.identity(&req.colors[0]).unwrap();
            assert_eq!(current.resident_format(), ash::vk::Format::R32_SFLOAT);
            assert_eq!((current.width(), current.height()), (8, 4));
            if let Some(prior) = &identity { assert_eq!(current, prior); }
            else { identity = Some(current.clone()); }
            assert_eq!(req.colors[0].target_gva, 0);
            assert_eq!(req.colors[0].mapping_id, 0);
            assert!(req.colors[0].target_seed_rgba.is_none());
            (EncodeStatus::Ok, None)
        });
        assert!(matches!(result.0, EncodeStatus::Ok));
        assert!(result.1.is_none());
        assert_eq!(req.chain_resident_established, index != 2);
    }
    assert_eq!(pass.phase, Phase::Finished);
    assert!(pass.targets.is_empty());
    assert_eq!(pass.prepare(&request()), Err("draw_vk_render_pass_sequence"));
    let mut fresh = VulkanRenderPass::default();
    fresh.prepare(&request()).unwrap();
    assert_ne!(fresh.targets[0].native.identity(), identity.as_ref().unwrap());
}

#[test]
fn memoryless_slots_and_encoders_have_independent_native_allocations() {
    let mut req = request();
    let mut second = req.colors[0].clone();
    second.slot = 1;
    second.texture_ref = 38;
    second.format = MTL_FORMAT_RGBA16_FLOAT;
    req.colors.push(second);
    let mut first = VulkanRenderPass::default();
    let mut concurrent = VulkanRenderPass::default();
    first.prepare(&req).unwrap();
    concurrent.prepare(&req).unwrap();
    let a = first.identity(&req.colors[0]).unwrap();
    let b = first.identity(&req.colors[1]).unwrap();
    assert!(!a.aliases(b));
    assert_eq!(b.resident_format(), ash::vk::Format::R16G16B16A16_SFLOAT);
    assert!(!a.aliases(concurrent.identity(&req.colors[0]).unwrap()));
    assert!(crate::backend::vulkan::translate::pixel::color_attachment(MTL_FORMAT_R32_FLOAT).is_err(),
        "pass-local R32Float does not license guest-backed R32Float Store");
}

#[test]
fn memoryless_refusal_ends_all_attachment_lifetimes() {
    let mut req = request();
    let mut pass = VulkanRenderPass::default();
    let result = pass.with_draw(&mut req, |_, _| (EncodeStatus::BadArgs("test_refusal"), None));
    assert!(matches!(result.0, EncodeStatus::BadArgs("test_refusal")));
    assert!(pass.targets.is_empty());
    assert_eq!(pass.phase, Phase::Failed);
    assert_eq!(pass.prepare(&req), Err("draw_vk_render_pass_sequence"));
}

#[test]
fn memoryless_invalid_continuation_cannot_replace_native_contents() {
    for changed in 0..5 {
        let mut req = request();
        let mut pass = VulkanRenderPass::default();
        pass.with_draw(&mut req, |_, _| (EncodeStatus::Ok, None));
        req.continues_render_pass = true;
        req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
        match changed {
            0 => req.colors[0].texture_ref += 1,
            1 => req.colors[0].format = MTL_FORMAT_RGBA16_FLOAT,
            2 => req.colors[0].width += 1,
            3 => req.colors.clear(),
            _ => req.task_id += 1,
        }
        let result = pass.with_draw(&mut req, |_, _| panic!("invalid continuation encoded"));
        assert!(matches!(result.0, EncodeStatus::BadArgs(_)));
        assert!(pass.targets.is_empty());
        assert_eq!(pass.phase, Phase::Failed);
    }
}

#[test]
fn memoryless_admission_requires_no_guest_content_and_single_sample() {
    for invalid in 0..10 {
        let mut req = request();
        match invalid {
            0 => req.continues_render_pass = true,
            1 => req.colors[0].sample_count = 4,
            2 => req.colors[0].target_gva = 0x4000,
            3 => req.colors[0].mapping_id = 1,
            4 => req.colors[0].target_seed_rgba = Some(vec![0; 4]),
            5 => req.colors[0].store_action = 1,
            6 => req.colors[0].load_action = MTL_LOAD_ACTION_LOAD,
            7 => req.colors[0].width = 0,
            8 => req.colors[0].slot = 1,
            _ => req.colors[0].texture_ref = 0,
        }
        assert!(VulkanRenderPass::default().prepare(&req).is_err());
    }
    let mut req = request();
    let mut second = req.colors[0].clone();
    second.slot = 1;
    req.colors.push(second);
    assert_eq!(VulkanRenderPass::default().prepare(&req), Err("draw_vk_memoryless_attachment_alias"));
}
