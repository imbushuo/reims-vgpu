use super::super::graphics_storage::{
    GraphicsStorageTexture, GraphicsTextureAccess, GraphicsTextureBinding,
};
use super::super::types::{
    ColorAttachmentState, ColorClearValue, SampledImageResource, SecondaryColorTarget,
    StorageImageFormat,
};
use super::*;

fn identity(id: u32) -> TargetIdentity {
    TargetIdentity::Surface {
        id,
        width: 16,
        height: 8,
        generation: 1,
        format: vk::Format::B8G8R8A8_UNORM,
    }
}

pub(super) fn request() -> DrawRequest {
    DrawRequest {
        width: 16,
        height: 8,
        vertex_count: 3,
        target_identity: Some(identity(1)),
        load_from_target: true,
        skip_readback: true,
        continues_render_pass: true,
        render_pass_continues: true,
        secondary_targets: vec![SecondaryColorTarget {
            identity: identity(2),
            width: 16,
            height: 8,
            attachment: ColorAttachmentState::new(
                vk::Format::B8G8R8A8_UNORM,
                ColorClearValue::Float([0.0, 1.0, 0.0, 1.0]),
            ),
            load: true,
            blend: None,
            color_write_mask: Default::default(),
        }],
        ..Default::default()
    }
}

fn terms(req: &DrawRequest) -> JoinTerms {
    JoinTerms::for_draw(
        req,
        BatchFit::Open(vk::CommandBuffer::null(), vk::Fence::null()),
        false,
        false,
        false,
        JoinTerms::draw_batch_target(req, None).is_some(),
    )
}

#[test]
fn mrt_batch_eligible_load_chain_can_open_and_join() {
    let req = request();
    assert!(terms(&req).batch_eligible());
    assert_eq!(terms(&req).refusal(), None);
    let first = JoinTerms::for_draw(&req, BatchFit::None, false, false, false, true);
    assert!(first.batch_eligible());
    assert_eq!(first.refusal(), Some("nojoin_no_open_batch"));
}

#[test]
fn mrt_batch_clear_and_encoder_boundaries_open_a_new_submission() {
    for change in 0..3 {
        let mut req = request();
        match change {
            0 => req.secondary_targets[0].load = false,
            1 => req.load_from_target = false,
            _ => req.continues_render_pass = false,
        }
        let decision = terms(&req);
        assert!(
            decision.batch_eligible(),
            "a boundary may open a fresh batch"
        );
        assert_eq!(decision.refusal(), Some("nojoin_mrt_boundary"));
    }
}

#[test]
fn mrt_batch_preserves_query_cpu_result_and_writable_texture_flushes() {
    let mut queried = request();
    queried.occlusion_query = Some(VisibilityResultMode::Counting);
    assert!(!terms(&queried).batch_eligible());
    assert_eq!(terms(&queried).refusal(), Some("nojoin_query"));
    let mut cpu = request();
    cpu.skip_readback = false;
    assert!(!terms(&cpu).batch_eligible());
    assert_eq!(terms(&cpu).refusal(), Some("nojoin_reads_back"));
    let mut writable = request();
    writable.storage_textures.push(GraphicsStorageTexture {
        format: StorageImageFormat::Rgba8Unorm,
        width: 1,
        height: 1,
        bytes: vec![0; 4],
        bindings: vec![GraphicsTextureBinding {
            binding: 32,
            access: GraphicsTextureAccess::Storage,
            stage: vk::ShaderStageFlags::FRAGMENT,
        }],
    });
    assert!(!terms(&writable).batch_eligible());
    assert_eq!(terms(&writable).refusal(), Some("nojoin_reads_back"));
}

#[test]
fn mrt_batch_cpu_and_gpu_seeds_cannot_join_the_previous_submission() {
    let mut cpu = request();
    cpu.target_rgba8 = Some(std::sync::Arc::new(vec![0; 16 * 8 * 4]));
    assert_eq!(terms(&cpu).refusal(), Some("nojoin_cpu_seed"));
    cpu.target_rgba8 = None;
    cpu.target_native_seed = Some(crate::runtime::draw::NativeColorSeed {
        layout: crate::protocol::pixel_format::TexelLayout::Bgra8,
        bytes: std::sync::Arc::new(vec![0; 16 * 8 * 4]),
    });
    assert_eq!(terms(&cpu).refusal(), Some("nojoin_cpu_seed"));
    let mut gpu = request();
    gpu.seed_from_target = Some(identity(3));
    assert_eq!(terms(&gpu).refusal(), Some("nojoin_gpu_seed"));
}

#[test]
fn mrt_batch_sampling_any_written_attachment_breaks_the_join() {
    for sampled in [identity(1), identity(2), identity(3)] {
        let mut req = request();
        req.sampled_images.push(SampledImageResource {
            binding: 32,
            array_element: 0,
            descriptor_count: 1,
            width: 16,
            height: 8,
            layers: 1,
            kind: reims_vgpu_core::texture_shape::TextureKind::D2,
            multisampled: false,
            source: SampledSource::Target(sampled.clone()),
            byte_origin: Default::default(),
            format: vk::Format::B8G8R8A8_UNORM,
            identity: None,
            swizzle: Default::default(),
        });
        assert_eq!(
            terms(&req).refusal(),
            (sampled != identity(3)).then_some("nojoin_mrt_sampled_attachment"),
        );
    }
}

#[test]
fn mrt_batch_identity_contains_every_color_extent_view_and_render_area() {
    let original = JoinTerms::draw_batch_target(&request(), None).unwrap();
    for change in 0..5 {
        let mut req = request();
        match change {
            0 => req.secondary_targets[0].identity = identity(3),
            1 => req.secondary_targets[0].width = 8,
            2 => {
                req.secondary_targets[0].attachment = ColorAttachmentState::new(
                    vk::Format::R16G16B16A16_SFLOAT,
                    ColorClearValue::default(),
                )
            }

            3 => req.width = 8,
            _ => req.color_sample_count = 4,
        }
        assert_ne!(JoinTerms::draw_batch_target(&req, None).unwrap(), original);
    }
    let mut clears = request();
    clears.secondary_targets[0].load = false;
    assert_eq!(
        JoinTerms::draw_batch_target(&clears, None).unwrap(),
        original,
        "load/clear are ordered operations, not allocation identity"
    );
    assert!(JoinTerms::mrt_batch_boundary(&clears));
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_mrt_batch_preserves_uniforms_scissors_and_secondary_sampling() {
    use crate::backend::vulkan::engine::{
        counter_snapshot, execute_draw_request, read_target, SamplerResource, StorageBufferResource,
    };
    use crate::backend::vulkan::sampled_shader::graphics_tests::{assemble, shader};
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    use std::sync::Arc;

    let state = DeviceState::new(DeviceId(0xabc1), PAGE_SHIFT_ARM64E);
    let fragment = Arc::new(assemble(
        r#"OpCapability Shader
            OpMemoryModel Logical GLSL450
            OpEntryPoint Fragment %main "main" %out0 %out1
            OpExecutionMode %main OriginUpperLeft
            OpDecorate %out0 Location 0
            OpDecorate %out1 Location 1
            OpDecorate %params DescriptorSet 0
            OpDecorate %params Binding 0
            OpDecorate %Params BufferBlock
            OpMemberDecorate %Params 0 Offset 0
            OpMemberDecorate %Params 1 Offset 16
            %void = OpTypeVoid
            %fn = OpTypeFunction %void
            %float = OpTypeFloat 32
            %vec4 = OpTypeVector %float 4
            %uint = OpTypeInt 32 0
            %zero = OpConstant %uint 0
            %one = OpConstant %uint 1
            %Params = OpTypeStruct %vec4 %vec4
            %params_ptr = OpTypePointer Uniform %Params
            %value_ptr = OpTypePointer Uniform %vec4
            %out_ptr = OpTypePointer Output %vec4
            %params = OpVariable %params_ptr Uniform
            %out0 = OpVariable %out_ptr Output
            %out1 = OpVariable %out_ptr Output
            %main = OpFunction %void None %fn
            %entry = OpLabel
            %p0 = OpAccessChain %value_ptr %params %zero
            %p1 = OpAccessChain %value_ptr %params %one
            %c0 = OpLoad %vec4 %p0
            %c1 = OpLoad %vec4 %p1
            OpStore %out0 %c0
            OpStore %out1 %c1
            OpReturn
            OpFunctionEnd
            "#,
    ));
    let uniforms = |colors: [[f32; 4]; 2]| {
        BufferContent::Bytes(Arc::new(
            colors
                .into_iter()
                .flatten()
                .flat_map(f32::to_le_bytes)
                .collect(),
        ))
    };
    let mut req = request();
    req.vert_spirv = Arc::new(shader(true, false, 0, 0, 1.0));
    req.frag_spirv = fragment;
    req.continues_render_pass = false;
    req.load_from_target = false;
    req.secondary_targets[0].load = false;
    req.scissors = vec![ScissorResource {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    }];
    req.storage_buffers = vec![StorageBufferResource {
        binding: 0,
        content: uniforms([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]),
    }];
    let before = counter_snapshot();
    assert!(execute_draw_request(&state, &req)
        .unwrap()
        .pixels
        .is_empty());
    req.continues_render_pass = true;
    req.render_pass_continues = false;
    req.load_from_target = true;
    req.secondary_targets[0].load = true;
    req.scissors[0].x = 8;
    req.storage_buffers[0].content = uniforms([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]]);
    assert!(execute_draw_request(&state, &req)
        .unwrap()
        .pixels
        .is_empty());
    assert_eq!(
        counter_snapshot().batch_joins - before.batch_joins,
        1,
        "the pixel result must exercise an actual MRT submission join"
    );

    let mut sample = request();
    sample.target_identity = Some(identity(3));
    sample.secondary_targets.clear();
    sample.continues_render_pass = false;
    sample.render_pass_continues = false;
    sample.load_from_target = false;
    sample.vert_spirv = Arc::clone(&req.vert_spirv);
    sample.frag_spirv = Arc::new(shader(false, true, 32, 160, 1.0));
    sample.sampled_images = vec![SampledImageResource {
        binding: 32,
        array_element: 0,
        descriptor_count: 1,
        width: 16,
        height: 8,
        layers: 1,
        kind: reims_vgpu_core::texture_shape::TextureKind::D2,
        multisampled: false,
        source: SampledSource::Target(identity(2)),
        byte_origin: Default::default(),
        format: vk::Format::B8G8R8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    }];
    let mut sampler = SamplerResource::normalized_default(160);
    sampler.min_filter = 0;
    sampler.mag_filter = 0;
    sample.samplers = vec![sampler];
    execute_draw_request(&state, &sample).unwrap();

    for (target, left, right) in [
        (identity(1), [255, 0, 0, 255], [0, 0, 255, 255]),
        (identity(2), [0, 255, 0, 255], [255, 255, 0, 255]),
        (identity(3), [255, 255, 0, 255], [255, 255, 0, 255]),
    ] {
        let pixels = read_target(&target).unwrap().into_rgba8().unwrap();
        assert_eq!(pixels.len(), 16 * 8 * 4);
        for y in 0..8 {
            for x in 0..16 {
                assert_eq!(
                    &pixels[(y * 16 + x) * 4..(y * 16 + x + 1) * 4],
                    if x < 8 { &left } else { &right },
                    "target={target:?} ({x},{y})"
                );
            }
        }
    }
}
