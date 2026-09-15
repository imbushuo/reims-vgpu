use super::super::caches::storage_descriptors::DeclarationProof;
use super::*;
use ash::vk::Handle;

mod captured_pipeline;

#[test]
fn storage_descriptor_layout_and_push_or_allocated_writes_share_admission() {
    let vertex = DeclarationProof::for_test(&[0]);
    let fragment = DeclarationProof::for_test(&[672]);
    let unknown = DeclarationProof::unproven_for_test();
    let slots: Vec<_> = [0, 5, 672, 678, 681]
        .into_iter()
        .map(|binding| {
            (
                binding,
                BoundBuffer {
                    buffer: vk::Buffer::from_raw(9),
                    offset: u64::from(binding) * 16,
                },
                16,
            )
        })
        .collect();
    for (fragment, expected) in [
        (&fragment, vec![0, 672]),
        (&unknown, vec![0, 5, 672, 678, 681]),
    ] {
        let plan = StorageDescriptorAdmission::new(&vertex, fragment);
        let mut layout: Vec<_> = slots
            .iter()
            .map(|(binding, _, _)| BindingSig {
                binding: *binding,
                ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                stages: 17,
                count: 1,
            })
            .collect();
        plan.filter_layout(&mut layout);
        let mut descriptors = Vec::new();
        fill_descriptor_bindings(&mut descriptors, &slots, &[], &[], None);
        plan.filter_writes(&mut descriptors);
        assert_eq!(
            layout
                .iter()
                .map(|binding| binding.binding)
                .collect::<Vec<_>>(),
            expected
        );
        for set in [vk::DescriptorSet::null(), vk::DescriptorSet::from_raw(1)] {
            with_descriptor_writes(&descriptors, set, |writes| {
                assert_eq!(
                    writes
                        .iter()
                        .map(|write| write.dst_binding)
                        .collect::<Vec<_>>(),
                    expected
                );
                assert!(writes.iter().all(|write| write.dst_set == set));
            });
        }
        assert_eq!(
            slots.len(),
            5,
            "descriptor admission does not remove any staged buffer"
        );
        assert_eq!(slots[1].0, 5, "the same buffer can still be a vertex input");
    }
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_storage_descriptor_admission_reuses_pso_and_preserves_stage_in_mutation() {
    descriptor_reuse_oracle(false, None);
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_sampled_descriptor_admission_reuses_fp16_pso_and_preserves_staging() {
    descriptor_reuse_oracle(true, None);
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_sampled_descriptor_declared_in_either_stage_retains_layout_variants() {
    descriptor_reuse_oracle(true, Some(true));
    descriptor_reuse_oracle(true, Some(false));
}

fn descriptor_reuse_oracle(sampled_extra: bool, declared_vertex: Option<bool>) {
    use crate::backend::vulkan::engine::{
        self, pass_local::PassLocalTarget, ColorAttachmentState, ColorClearValue,
        StorageBufferResource, VertexAttributeFormat, VertexAttributeResource,
    };
    use crate::backend::vulkan::sampled_shader::graphics_tests::assemble;
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    use std::sync::Arc;

    let mut vertex = Arc::new(assemble(
        r#"OpCapability Shader
        OpMemoryModel Logical GLSL450
        OpEntryPoint Vertex %main "main" %input %position
        OpDecorate %input Location 0
        OpDecorate %position BuiltIn Position
        %void = OpTypeVoid
        %fn = OpTypeFunction %void
        %float = OpTypeFloat 32
        %vec4 = OpTypeVector %float 4
        %input_ptr = OpTypePointer Input %vec4
        %output_ptr = OpTypePointer Output %vec4
        %input = OpVariable %input_ptr Input
        %position = OpVariable %output_ptr Output
        %main = OpFunction %void None %fn
        %entry = OpLabel
        %value = OpLoad %vec4 %input
        OpStore %position %value
        OpReturn
        OpFunctionEnd
    "#,
    ));
    let mut fragment = Arc::new(assemble(
        r#"OpCapability Shader
        OpMemoryModel Logical GLSL450
        OpEntryPoint Fragment %main "main" %out
        OpExecutionMode %main OriginUpperLeft
        OpDecorate %out Location 0
        OpDecorate %params DescriptorSet 0
        OpDecorate %params Binding 0
        OpDecorate %Params BufferBlock
        OpMemberDecorate %Params 0 Offset 0
        %void = OpTypeVoid
        %fn = OpTypeFunction %void
        %float = OpTypeFloat 32
        %vec4 = OpTypeVector %float 4
        %uint = OpTypeInt 32 0
        %zero = OpConstant %uint 0
        %Params = OpTypeStruct %vec4
        %params_ptr = OpTypePointer Uniform %Params
        %value_ptr = OpTypePointer Uniform %vec4
        %out_ptr = OpTypePointer Output %vec4
        %params = OpVariable %params_ptr Uniform
        %out = OpVariable %out_ptr Output
        %main = OpFunction %void None %fn
        %entry = OpLabel
        %p = OpAccessChain %value_ptr %params %zero
        %value = OpLoad %vec4 %p
        OpStore %out %value
        OpReturn
        OpFunctionEnd
    "#,
    ));
    if let Some(vertex_stage) = declared_vertex {
        let words = if vertex_stage {
            &mut vertex
        } else {
            &mut fragment
        };
        *words = super::super::serial_interlock::tests::with_resource(words, 704, true);
    }
    let bytes = |values: &[f32]| {
        BufferContent::Bytes(Arc::new(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        ))
    };
    let vertices = bytes(&[
        -1.0, -1.0, 0.0, 1.0, 3.0, -1.0, 0.0, 1.0, -1.0, 3.0, 0.0, 1.0,
    ]);
    let state = DeviceState::new(DeviceId(0xabd1), PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let format = if sampled_extra {
        vk::Format::R16G16B16A16_SFLOAT
    } else {
        vk::Format::B8G8R8A8_UNORM
    };
    let target = PassLocalTarget::new(16, 8, format).unwrap();
    let mut request = DrawRequest {
        width: 16,
        height: 8,
        vertex_count: 3,
        skip_readback: true,
        target_identity: Some(target.identity().clone()),
        color_attachment: Some(ColorAttachmentState::new(
            format,
            ColorClearValue::Float([0.0; 4]),
        )),
        vert_spirv: vertex,
        frag_spirv: fragment,
        vertex_attributes: vec![VertexAttributeResource {
            location: 0,
            binding: 5,
            format: VertexAttributeFormat::Float4,
            offset: 0,
            stride: 16,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
            content: vertices.clone(),
        }],
        ..Default::default()
    };
    let before = engine::counter_snapshot();
    let mut first_hash_words = None;
    for (iteration, (extra, mut color, pixel)) in [
        (vec![5, 681], [1.0, 0.0, 0.0, 1.0], [255, 0, 0, 255]),
        (vec![5, 678, 681], [0.0, 1.0, 0.0, 1.0], [0, 255, 0, 255]),
        (vec![], [0.0, 0.0, 1.0, 1.0], [0, 0, 255, 255]),
        (vec![5, 678, 681], [1.0, 1.0, 0.0, 1.0], [255, 255, 0, 255]),
    ]
    .into_iter()
    .enumerate()
    {
        if sampled_extra {
            request.sampled_images.clear();
            if iteration % 2 == 1 {
                request
                    .sampled_images
                    .push(sampled_fixture(704, iteration as u8));
            }
            if iteration == 3 {
                color = [0x1001u16, 0x4000, 0xb800, 0x3555]
                    .map(crate::protocol::pixel_format::f16_to_f32);
            }
        }
        request.storage_buffers = vec![StorageBufferResource {
            binding: 0,
            content: bytes(&color),
        }];
        request
            .storage_buffers
            .extend(extra.into_iter().map(|binding| StorageBufferResource {
                binding,
                content: vertices.clone(),
            }));
        engine::execute_draw_request(&state, &request).unwrap();
        if sampled_extra {
            let pixels = engine::read_target_native(target.identity()).unwrap();
            let expected = [
                [0x3c00u16, 0, 0, 0x3c00],
                [0, 0x3c00, 0, 0x3c00],
                [0, 0, 0x3c00, 0x3c00],
                [0x1001, 0x4000, 0xb800, 0x3555],
            ][iteration]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            assert_eq!(pixels.pixels, expected.repeat(16 * 8), "{color:?}");
        } else {
            let pixels = engine::read_target(target.identity())
                .unwrap()
                .into_rgba8()
                .unwrap();
            assert_eq!(pixels.len(), 16 * 8 * 4);
            assert!(
                pixels.chunks_exact(4).all(|actual| actual == pixel),
                "{color:?}"
            );
        }
        let now = engine::counter_snapshot();
        assert_eq!(
            now.shader_hash_words,
            *first_hash_words.get_or_insert(now.shader_hash_words)
        );
        assert_eq!(
            now.shader_descriptor_proofs - before.shader_descriptor_proofs,
            2
        );
        assert_eq!(
            now.pipeline_misses - before.pipeline_misses,
            if declared_vertex.is_some() && iteration > 0 {
                2
            } else {
                1
            }
        );
    }
    let expected_variants = if declared_vertex.is_some() { 2 } else { 1 };
    engine::device_caches(&state).unwrap().with(|caches| {
        let levels = caches.levels();
        assert_eq!(levels[0], 2);
        assert_eq!(levels[1], expected_variants);
        assert_eq!(levels[4], expected_variants);
    });
    let after = engine::counter_snapshot();
    if sampled_extra {
        assert_eq!(after.sampled_reuploads - before.sampled_reuploads, 2);
        assert_eq!(
            after.sampled_reupload_bytes - before.sampled_reupload_bytes,
            8,
            "unused descriptors do not skip the provided image uploads"
        );
    }
    let pushes = after.descriptor_pushes - before.descriptor_pushes;
    let allocated = after.descriptor_set_updates - before.descriptor_set_updates;
    assert!(pushes > 0 || allocated > 0);
    if crate::config::read(crate::config::PUSH_DESCRIPTORS).0 == crate::config::Switch::Off {
        assert_eq!(pushes, 0);
        assert_eq!(allocated, 4);
    }
    eprintln!(
        "storage_descriptor_gpu sampled_extra={sampled_extra} declared_vertex={declared_vertex:?} draws=4 shader_proofs=2 layouts={expected_variants} native_graphics_creates={expected_variants} \
         stage_in=PASS declared_mutation=PASS hash_rewalks=0 pushes={pushes} allocated={allocated}"
    );
    drop(target);
    engine::test_quiesce_ring();
    engine::test_reset_engine(&state);
}

fn sampled_fixture(binding: u32, byte: u8) -> super::super::types::SampledImageResource {
    use super::super::types::*;
    SampledImageResource {
        binding,
        array_element: 0,
        descriptor_count: 1,
        width: 1,
        height: 1,
        layers: 1,
        kind: reims_vgpu_core::texture_shape::TextureKind::D2,
        multisampled: false,
        source: SampledSource::Bytes(std::sync::Arc::new(vec![byte, 33, 65, 255])),
        byte_origin: SampledByteOrigin::Synthetic,
        format: vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    }
}

#[test]
fn sampled_descriptor_layout_and_all_final_writes_share_complete_two_stage_admission() {
    use super::super::pools::PushDescriptorBinding;
    let empty = DeclarationProof::for_test(&[]);
    let base = DeclarationProof::for_test(&[0, 707]);
    let declared = DeclarationProof::for_test(&[0, 704, 707]);
    let unknown = DeclarationProof::unproven_for_test();
    for (vertex, fragment, keep704, keep681) in [
        (&empty, &base, false, false),
        (&declared, &base, true, false),
        (&empty, &declared, true, false),
        (&unknown, &base, true, true),
        (&empty, &unknown, true, true),
    ] {
        let admission = StorageDescriptorAdmission::new(vertex, fragment);
        let storage = [0, 681].map(|binding| {
            (
                binding,
                BoundBuffer {
                    buffer: vk::Buffer::from_raw(5),
                    offset: 0,
                },
                16,
            )
        });
        let sampled = [(704, 0), (704, 1), (707, 0)].map(|(binding, array_element)| {
            PreparedSampled::Feedback {
                binding,
                array_element,
                view: vk::ImageView::from_raw(7),
            }
        });
        let mut writes = Vec::new();
        fill_descriptor_bindings(
            &mut writes,
            &storage,
            &sampled,
            &[(832, vk::Sampler::from_raw(8))],
            Some((vk::ImageView::from_raw(9), vk::ImageLayout::GENERAL)),
        );
        writes.push(PushDescriptorBinding::Image {
            binding: 1155,
            array_element: 0,
            ty: vk::DescriptorType::STORAGE_IMAGE,
            sampler: vk::Sampler::null(),
            view: vk::ImageView::from_raw(10),
            layout: vk::ImageLayout::GENERAL,
        });
        writes.push(PushDescriptorBinding::Image {
            binding: 709,
            array_element: 0,
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            sampler: vk::Sampler::null(),
            view: vk::ImageView::from_raw(11),
            layout: vk::ImageLayout::GENERAL,
        });
        let mut layout: Vec<_> = [
            (0, vk::DescriptorType::STORAGE_BUFFER, 1),
            (681, vk::DescriptorType::STORAGE_BUFFER, 1),
            (704, vk::DescriptorType::SAMPLED_IMAGE, 2),
            (707, vk::DescriptorType::SAMPLED_IMAGE, 1),
            (709, vk::DescriptorType::SAMPLED_IMAGE, 1),
            (832, vk::DescriptorType::SAMPLER, 1),
            (192, vk::DescriptorType::INPUT_ATTACHMENT, 1),
            (1155, vk::DescriptorType::STORAGE_IMAGE, 1),
        ]
        .into_iter()
        .map(|(binding, ty, count)| BindingSig {
            binding,
            ty: ty.as_raw() as u32,
            stages: 17,
            count,
        })
        .collect();
        admission.filter_layout(&mut layout);
        admission.filter_writes(&mut writes);
        assert_eq!(layout.iter().any(|b| b.binding == 704), keep704);
        assert_eq!(layout.iter().any(|b| b.binding == 681), keep681);
        for set in [vk::DescriptorSet::null(), vk::DescriptorSet::from_raw(1)] {
            with_descriptor_writes(&writes, set, |native| {
                for binding in &layout {
                    let emitted: Vec<_> = native
                        .iter()
                        .filter(|w| w.dst_binding == binding.binding)
                        .collect();
                    assert_eq!(emitted.len(), binding.count as usize);
                    assert!(emitted
                        .iter()
                        .all(|w| w.descriptor_type.as_raw() as u32 == binding.ty));
                }
                assert!(native
                    .iter()
                    .all(|w| layout.iter().any(|b| b.binding == w.dst_binding)));
                for binding in [192, 832, 1155] {
                    assert!(native.iter().any(|w| w.dst_binding == binding));
                }
            });
        }
        assert_eq!(sampled.len(), 3, "all prepared sampled images remain owned");
        assert_eq!(storage.len(), 2, "all staged storage buffers remain owned");
    }
}

#[test]
fn sampled_descriptor_admission_preserves_format_and_mrt_compatibility_variants() {
    let mut variants = std::collections::HashSet::new();
    for format in [
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R16G16B16A16_SFLOAT,
    ] {
        for count in 0..3 {
            let mut pass = PassKey::single(Color0Load::Preserve, format);
            pass.secondary_count = count;
            for secondary in &mut pass.secondary[..count as usize] {
                secondary.format = vk::Format::R16G16B16A16_SFLOAT;
                secondary.load = true;
            }
            assert!(variants.insert(pass.compatibility()));
        }
    }
    assert_eq!(variants.len(), 9);
}
