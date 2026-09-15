use super::*;
use crate::backend::vulkan::engine::{self, ColorWriteMask};
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
use std::{path::Path, sync::atomic::Ordering, sync::Arc};

fn module(path: &Path) -> Arc<Vec<u32>> {
    let bytes =
        std::fs::read(path).expect("stage private vertex.spv and fragment.spv replay inputs");
    assert!(bytes.len().is_multiple_of(4));
    Arc::new(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect(),
    )
}

#[test]
#[ignore = "requires exclusive Vulkan GPU and private target/vulkan-pipeline-replay inputs"]
fn vulkan_gpu_sampled_descriptor_captured_pipeline_creation_only_reuses_layout() {
    let inputs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/vulkan-pipeline-replay");
    let vertex = module(&inputs.join("vertex.spv"));
    let fragment = module(&inputs.join("fragment.spv"));
    let state = DeviceState::new(DeviceId(0xabd2), PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let submits = engine::counter_snapshot().queue_async_submits;
    {
        let mut guard = engine::lock_engine();
        let engine::EngineState {
            owner,
            pools,
            counters,
            ..
        } = &mut *guard;
        let ctx = owner.ensure(counters).unwrap();
        unsafe {
            pools.ensure_init(ctx, counters).unwrap();
        }
        engine::device_caches(&state).unwrap().with(|caches| unsafe {
            let (vert, vertex_module) =
                caches.get_or_create_shader_memoized(ctx, &vertex, counters, pools).unwrap();
            let (frag, fragment_module) =
                caches.get_or_create_shader_memoized(ctx, &fragment, counters, pools).unwrap();
            let admission = StorageDescriptorAdmission::new(
                &vertex_module.declarations, &fragment_module.declarations,
            );
            assert_eq!(vertex_module.declarations.complete_bindings(), Some([0, 1, 2, 3, 4].as_slice()));
            assert_eq!(fragment_module.declarations.complete_bindings(), Some([
                192, 193, 672, 673, 674, 675, 676, 677, 679, 680,
                707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718,
                832, 833, 834, 835,
            ].as_slice()));
            let mut pass = PassKey::single(Color0Load::Preserve, vk::Format::R16G16B16A16_SFLOAT);
            pass.secondary_count = 1;
            pass.secondary[0] = SecondaryAttachKey {
                format: vk::Format::R16G16B16A16_SFLOAT, load: true,
            };
            pass.color_input = 3;
            let render_pass = caches.get_or_create_pass(ctx, pass, counters, pools).unwrap();
            let attrs = caches.intern_attrs(&[]);
            let topology = reims_vgpu_vulkan::topology::key(
                reims_vgpu_core::topology::PrimitiveType::Triangle,
                reims_vgpu_vulkan::topology::TopologyCell {
                    dynamic: ctx.features.extended_dynamic_state,
                    unrestricted: ctx.features.dynamic_primitive_topology_unrestricted,
                },
            );
            let raster = reims_vgpu_vulkan::raster::plan(
                reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT,
                reims_vgpu_vulkan::raster::RasterCell {
                    depth_clamp: ctx.features.depth_clamp,
                    fill_mode_non_solid: ctx.features.fill_mode_non_solid,
                    dynamic_cull_and_winding: ctx.features.extended_dynamic_state,
                    dynamic_polygon_mode: ctx.features.dynamic_polygon_mode,
                    dynamic_depth_clamp: ctx.features.dynamic_depth_clamp,
                },
            ).unwrap().state;
            let depth_stencil = reims_vgpu_vulkan::depth_stencil::plan(
                &depth_stencil_state(None),
                reims_vgpu_vulkan::depth_stencil::DepthStencilCell {
                    extended_dynamic_state: ctx.features.extended_dynamic_state,
                },
                false,
            ).state;
            let before = counters.pipeline_misses.load(Ordering::Relaxed);
            let mut held = None;
            for extra in [false, true, false] {
                let mut bindings: Vec<_> = (0..5).chain([672, 673, 674, 675, 676, 677, 679, 680])
                    .map(|binding| BindingSig {
                        binding, ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                        stages: 17, count: 1,
                    }).collect();
                bindings.extend((707..=718).map(|binding| BindingSig {
                    binding, ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
                    stages: 17, count: 1,
                }));
                bindings.extend((832..=835).map(|binding| BindingSig {
                    binding, ty: vk::DescriptorType::SAMPLER.as_raw() as u32,
                    stages: 17, count: 1,
                }));
                bindings.extend((192..=193).map(|binding| BindingSig {
                    binding, ty: vk::DescriptorType::INPUT_ATTACHMENT.as_raw() as u32,
                    stages: 16, count: 1,
                }));
                if extra {
                    bindings.push(BindingSig {
                        binding: 704, ty: vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
                        stages: 17, count: 1,
                    });
                }
                assert_eq!(bindings.len(), if extra { 32 } else { 31 });
                canonicalize_layout_bindings(&mut bindings).unwrap();
                admission.filter_layout(&mut bindings);
                assert_eq!(bindings.len(), 31);
                let layout = caches.get_or_create_layout(ctx, &bindings, None, counters).unwrap();
                let key = PipelineKey {
                    vert, frag, attrs, topology,
                    blend: None, secondary_blend: [None; MAX_SECONDARY_ATTACH],
                    color_write_mask: [ColorWriteMask::ALL; 1 + MAX_SECONDARY_ATTACH],
                    pass: pass.compatibility(), feedback_colors: 0,
                    raster, depth_stencil, viewport_slots: 1, layout: layout.id,
                };
                let pipeline = caches.get_or_create_pipeline(
                    ctx, &key, None, vertex_module.module, &vertex, fragment_module.module,
                    &fragment, layout.pipeline_layout, render_pass, counters, pools,
                ).unwrap();
                assert_eq!((layout.id, pipeline), *held.get_or_insert((layout.id, pipeline)));
                assert_eq!(counters.pipeline_misses.load(Ordering::Relaxed) - before, 1);
            }
            assert_eq!(caches.levels()[1], 1);
            assert_eq!(caches.levels()[4], 1);
            eprintln!("captured_pipeline_replay vertex={vert:?} fragment={frag:?} provided_layouts=31,32,31 canonical_layouts=1 native_graphics_creates=1 draws=0 submissions=0");
        });
    }
    assert_eq!(engine::counter_snapshot().queue_async_submits, submits);
    engine::test_reset_engine(&state);
}
