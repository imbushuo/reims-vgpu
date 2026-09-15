use super::*;

#[test]
#[ignore = "requires an exclusive Vulkan GPU and REIMS_VGPU_PIPELINE_DIAGNOSTICS=on"]
fn vulkan_gpu_pso_diagnostics_capture_variants_feedback_and_skip_cached_draws() {
    use crate::backend::vulkan::engine::context::native_cache::Origin;
    use crate::backend::vulkan::engine::{
        self, pass_local::PassLocalTarget, BlendStateResource, DrawRequest,
    };
    use crate::backend::vulkan::sampled_shader::graphics_tests::shader;
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};

    assert!(
        requested(),
        "set REIMS_VGPU_PIPELINE_DIAGNOSTICS=on for this oracle"
    );
    let state = DeviceState::new(DeviceId(0xabd0), PAGE_SHIFT_ARM64E);
    let target = PassLocalTarget::new(16, 8, vk::Format::B8G8R8A8_UNORM).unwrap();
    let mut request = DrawRequest {
        width: 16,
        height: 8,
        vertex_count: 3,
        skip_readback: true,
        target_identity: Some(target.identity().clone()),
        vert_spirv: Arc::new(shader(true, false, 0, 0, 1.0)),
        frag_spirv: Arc::new(shader(false, false, 0, 0, 1.0)),
        ..Default::default()
    };
    engine::execute_draw_request(&state, &request).unwrap();
    let first = engine::read_target(target.identity())
        .unwrap()
        .into_rgba8()
        .unwrap();
    request.blend = Some(BlendStateResource {
        src_rgb: 1,
        dst_rgb: 0,
        op_rgb: 0,
        src_alpha: 1,
        dst_alpha: 0,
        op_alpha: 0,
    });
    engine::execute_draw_request(&state, &request).unwrap();
    let second = engine::read_target(target.identity())
        .unwrap()
        .into_rgba8()
        .unwrap();
    request.blend = None;
    engine::execute_draw_request(&state, &request).unwrap();
    let third = engine::read_target(target.identity())
        .unwrap()
        .into_rgba8()
        .unwrap();
    assert_eq!(first.len(), 16 * 8 * 4);
    assert_eq!(first, second);
    assert_eq!(first, third);

    engine::device_caches(&state).unwrap().with(|caches| {
        let diagnostics = caches.diagnostics.as_ref().unwrap();
        assert_eq!(
            diagnostics.observations.len(),
            2,
            "a PSO hit must not emit another creation trace"
        );
        assert_eq!(
            diagnostics.source_digests.len(),
            2,
            "source words are memoized per immutable allocation"
        );
        let a = &diagnostics.observations[0];
        let b = &diagnostics.observations[1];
        assert_ne!(a.fingerprint, b.fingerprint);
        assert!(a.declaration.is_some() && b.declaration.is_some());
        assert_eq!(a.source, b.source);
        assert_eq!(a.driver, b.driver);
        assert_eq!(a.source, a.driver, "this oracle has no capability patch");
        assert_eq!(a.cache.program, b.cache.program);
        assert_eq!(a.cache.handle, b.cache.handle);
        assert_eq!(a.cache.initial_payload, b.cache.initial_payload);
        assert_eq!(b.cache.origin, Origin::Resident);
        for observation in &diagnostics.observations {
            assert_eq!(observation.result, vk::Result::SUCCESS);
            assert!(observation.elapsed_ns > 0);
            if let Some(feedback) = observation.feedback {
                assert!(feedback.valid);
                assert!(feedback.duration_ns.is_some());
                assert!(observation.stages.unwrap().iter().all(|stage| stage.valid));
            } else {
                assert!(observation.stages.is_none());
            }
        }
        eprintln!(
            "pso_diagnostics_gpu creations=2 cached_draws=1 source_memos=2 feedback={:?} stages={:?} origin={:?}->{:?}",
            a.feedback, a.stages, a.cache.origin, b.cache.origin,
        );
    });
}
