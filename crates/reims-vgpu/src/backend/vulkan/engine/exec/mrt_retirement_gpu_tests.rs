use super::*;

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_mrt_last_pin_retires_pass_local_images_without_idle_maintenance() {
    use crate::backend::vulkan::engine::{self, pass_local::PassLocalTarget};
    use crate::backend::vulkan::sampled_shader::graphics_tests::{assemble, shader};
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    use std::sync::Arc;

    let state = DeviceState::new(DeviceId(0xabc2), PAGE_SHIFT_ARM64E);
    let first = PassLocalTarget::new(16, 8, vk::Format::B8G8R8A8_UNORM).unwrap();
    let second = PassLocalTarget::new(16, 8, vk::Format::B8G8R8A8_UNORM).unwrap();
    let mut req = super::mrt_batch_tests::request();
    req.target_identity = Some(first.identity().clone());
    req.secondary_targets[0].identity = second.identity().clone();
    req.continues_render_pass = false;
    req.render_pass_continues = false;
    req.load_from_target = false;
    req.secondary_targets[0].load = false;
    req.vert_spirv = Arc::new(shader(true, false, 0, 0, 1.0));
    req.frag_spirv = Arc::new(assemble(
        r#"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out0 %out1
OpExecutionMode %main OriginUpperLeft
OpDecorate %out0 Location 0
OpDecorate %out1 Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%vec4 = OpTypeVector %float 4
%ptr = OpTypePointer Output %vec4
%zero = OpConstant %float 0
%one = OpConstant %float 1
%red = OpConstantComposite %vec4 %one %zero %zero %one
%green = OpConstantComposite %vec4 %zero %one %zero %one
%out0 = OpVariable %ptr Output
%out1 = OpVariable %ptr Output
%main = OpFunction %void None %fn
%entry = OpLabel
OpStore %out0 %red
OpStore %out1 %green
OpReturn
OpFunctionEnd
"#,
    ));
    let before = engine::counter_snapshot().registry_pass_local_count;
    engine::execute_draw_request(&state, &req).unwrap();
    drop(first);
    drop(second);
    assert_eq!(
        engine::counter_snapshot().registry_pass_local_count,
        before + 2,
        "ending the encoder cannot free the unsubmitted batch's attachments"
    );
    {
        let mut guard = engine::lock_engine();
        let engine::EngineState {
            ref owner,
            ref mut pools,
            ref counters,
            ..
        } = *guard;
        let ctx = owner.ctx.as_ref().unwrap();
        unsafe { pools.retire_all(ctx, counters).unwrap() };
    }
    let after = engine::counter_snapshot();
    assert_eq!(
        after.registry_pass_local_count, before,
        "fence retirement must complete exact released-resource cleanup without an idle tick"
    );
    assert_eq!(after.pass_local_retiring_count, 0);
}
