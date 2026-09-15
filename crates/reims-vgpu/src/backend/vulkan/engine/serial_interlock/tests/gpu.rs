use super::*;
use crate::backend::vulkan::engine::{self, *};
use crate::backend::vulkan::sampled_shader::graphics_tests::assemble;
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};

const W: usize = 16;
const H: usize = 12;
const DW: usize = 24;
const DH: usize = 20;
const HALF: [[u16; 4]; 3] = [
    [0x1001, 0x3400, 0x3800, 0x3c00],
    [0x4000, 0xb800, 0x3555, 0x3800],
    [0x3800, 0x3a00, 0x3000, 0x3c00],
];
const BGRA: [[u8; 4]; 3] = [[128, 64, 0, 255], [85, 0, 255, 128], [32, 191, 128, 255]];

fn vertex() -> Arc<Vec<u32>> {
    Arc::new(assemble(
        r#"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %index %position %uv
OpDecorate %index BuiltIn VertexIndex
OpDecorate %position BuiltIn Position
OpDecorate %uv Location 1
OpDecorate %params DescriptorSet 0
OpDecorate %params Binding 0
OpDecorate %Params BufferBlock
OpMemberDecorate %Params 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%in_uint = OpTypePointer Input %uint
%out_v2 = OpTypePointer Output %v2
%out_v4 = OpTypePointer Output %v4
%Params = OpTypeStruct %v4
%params_ptr = OpTypePointer Uniform %Params
%value_ptr = OpTypePointer Uniform %v4
%params = OpVariable %params_ptr Uniform
%index = OpVariable %in_uint Input
%position = OpVariable %out_v4 Output
%uv = OpVariable %out_v2 Output
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%zero = OpConstant %float 0
%one = OpConstant %float 1
%two = OpConstant %float 2
%half = OpConstant %float 0.5
%main = OpFunction %void None %fn
%entry = OpLabel
%i = OpLoad %uint %index
%xb = OpBitwiseAnd %uint %i %u1
%shift = OpShiftRightLogical %uint %i %u1
%yb = OpBitwiseAnd %uint %shift %u1
%x = OpConvertUToF %float %xb
%y = OpConvertUToF %float %yb
%x2 = OpFMul %float %x %two
%y2 = OpFMul %float %y %two
%px = OpFSub %float %x2 %one
%py = OpFSub %float %y2 %one
%p = OpCompositeConstruct %v4 %px %py %zero %one
OpStore %position %p
%ptr = OpAccessChain %value_ptr %params %u0
%values = OpLoad %v4 %ptr
%width = OpCompositeExtract %float %values 0
%height = OpCompositeExtract %float %values 1
%dx = OpCompositeExtract %float %values 2
%dy = OpCompositeExtract %float %values 3
%sx = OpFMul %float %x %width
%yf = OpFSub %float %one %y
%sy = OpFMul %float %yf %height
%tx = OpFAdd %float %sx %dx
%ty = OpFAdd %float %sy %dy
%coord = OpCompositeConstruct %v2 %tx %ty
OpStore %uv %coord
OpReturn
OpFunctionEnd
"#,
    ))
}

fn fragment(mode: &str, count: usize, bgra: bool) -> Arc<Vec<u32>> {
    let copy = mode == "copy";
    let mut declarations = String::new();
    let mut variables = String::new();
    let mut interface = "%uv_in".to_string();
    if copy {
        declarations.push_str("OpDecorate %source DescriptorSet 0\nOpDecorate %source Binding 192\nOpDecorate %source InputAttachmentIndex 0\nOpDecorate %destination DescriptorSet 0\nOpDecorate %destination Binding 1155\nOpDecorate %destination Coherent\n");
        variables.push_str(&format!(
            "%source_type = OpTypeImage %float SubpassData 0 0 0 2 Unknown\n\
             %source_ptr = OpTypePointer UniformConstant %source_type\n\
             %source = OpVariable %source_ptr UniformConstant\n\
             %destination_type = OpTypeImage %float 2D 0 0 0 2 {}\n\
             %destination_ptr = OpTypePointer UniformConstant %destination_type\n\
             %destination = OpVariable %destination_ptr UniformConstant\n",
            if bgra { "Unknown" } else { "Rgba16f" },
        ));
    } else {
        for slot in 0..if mode == "seed" { count } else { 1 } {
            interface.push_str(&format!(" %out{slot}"));
            declarations.push_str(&format!("OpDecorate %out{slot} Location {slot}\n"));
            variables.push_str(&format!("%out{slot} = OpVariable %out_ptr Output\n"));
        }
    }
    let body = if copy {
        r#"OpBeginInvocationInterlockEXT
%uv_value = OpLoad %v2 %uv_in
%bounded = OpExtInst %v2 %glsl FClamp %uv_value %zero2 %max2
%short_coord = OpConvertFToU %v2short %bounded
%coord = OpUConvert %v2uint %short_coord
%src = OpLoad %source_type %source
%prior = OpImageRead %v4 %src %zeroi2
%half_value = OpFConvert %v4half %prior
%value = OpFConvert %v4 %half_value
%dst = OpLoad %destination_type %destination
OpImageWrite %dst %coord %value
OpEndInvocationInterlockEXT
"#
        .to_string()
    } else if mode == "paint" {
        "%uv_value = OpLoad %v2 %uv_in\nOpStore %out0 %green\n".into()
    } else {
        let mut body = r#"%uv_value = OpLoad %v2 %uv_in
%xf = OpCompositeExtract %float %uv_value 0
%yf = OpCompositeExtract %float %uv_value 1
%x = OpConvertFToU %uint %xf
%y = OpConvertFToU %uint %yf
%x3 = OpIEqual %bool %x %u3
%yge2 = OpUGreaterThanEqual %bool %y %u2
%yle9 = OpULessThanEqual %bool %y %u9
%vertical_y = OpLogicalAnd %bool %yge2 %yle9
%vertical = OpLogicalAnd %bool %x3 %vertical_y
%xge3 = OpUGreaterThanEqual %bool %x %u3
%xle10 = OpULessThanEqual %bool %x %u10
%xle8 = OpULessThanEqual %bool %x %u8
%y2 = OpIEqual %bool %y %u2
%y5 = OpIEqual %bool %y %u5
%topx = OpLogicalAnd %bool %xge3 %xle10
%midx = OpLogicalAnd %bool %xge3 %xle8
%top = OpLogicalAnd %bool %y2 %topx
%mid = OpLogicalAnd %bool %y5 %midx
%bars = OpLogicalOr %bool %top %mid
%glyph = OpLogicalOr %bool %vertical %bars
%glyph4 = OpCompositeConstruct %b4 %glyph %glyph %glyph %glyph
"#
        .to_string();
        for slot in 0..count {
            body.push_str(&format!(
                "%row{slot} = OpIMul %uint %y %u{slot}\n\
                 %sum{slot} = OpIAdd %uint %x %row{slot}\n\
                 %kind{slot} = OpUMod %uint %sum{slot} %u3\n\
                 %is0_{slot} = OpIEqual %bool %kind{slot} %u0\n\
                 %is1_{slot} = OpIEqual %bool %kind{slot} %u1\n\
                 %b0_{slot} = OpCompositeConstruct %b4 %is0_{slot} %is0_{slot} %is0_{slot} %is0_{slot}\n\
                 %b1_{slot} = OpCompositeConstruct %b4 %is1_{slot} %is1_{slot} %is1_{slot} %is1_{slot}\n\
                 %pick{slot} = OpSelect %v4 %b1_{slot} %palette1 %palette2\n\
                 %plain{slot} = OpSelect %v4 %b0_{slot} %palette0 %pick{slot}\n"));
            if slot == 0 {
                body.push_str(
                    "%color0 = OpSelect %v4 %glyph4 %green %plain0\nOpStore %out0 %color0\n",
                );
            } else {
                body.push_str(&format!("OpStore %out{slot} %plain{slot}\n"));
            }
        }
        body
    };
    let interlock = if copy {
        "OpCapability FragmentShaderPixelInterlockEXT\nOpCapability InputAttachment\nOpExtension \"SPV_EXT_fragment_shader_interlock\"\n"
    } else {
        ""
    };
    let unformatted = if copy && bgra {
        "OpCapability StorageImageWriteWithoutFormat\n"
    } else {
        ""
    };
    Arc::new(assemble(&format!(
        r#"OpCapability Shader
OpCapability Float16
OpCapability Int16
{unformatted}{interlock}%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" {interface}
OpExecutionMode %main OriginUpperLeft
{execution}OpDecorate %uv_in Location 1
{declarations}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%b4 = OpTypeVector %bool 4
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%short = OpTypeInt 16 0
%float = OpTypeFloat 32
%half = OpTypeFloat 16
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%v4half = OpTypeVector %half 4
%v2short = OpTypeVector %short 2
%v2uint = OpTypeVector %uint 2
%v2int = OpTypeVector %int 2
%input_ptr = OpTypePointer Input %v2
%out_ptr = OpTypePointer Output %v4
%uv_in = OpVariable %input_ptr Input
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u2 = OpConstant %uint 2
%u3 = OpConstant %uint 3
%u5 = OpConstant %uint 5
%u8 = OpConstant %uint 8
%u9 = OpConstant %uint 9
%u10 = OpConstant %uint 10
%i0 = OpConstant %int 0
%zeroi2 = OpConstantComposite %v2int %i0 %i0
%zero = OpConstant %float 0
%one = OpConstant %float 1
%two = OpConstant %float 2
%negative = OpConstant %float -0.5
%quarter = OpConstant %float 0.25
%half_f = OpConstant %float 0.5
%three_quarters = OpConstant %float 0.75
%eighth = OpConstant %float 0.125
%tiny = OpConstant %float 0.000488758087158203125
%third = OpConstant %float 0.333251953125
%max = OpConstant %float 65535
%zero2 = OpConstantComposite %v2 %zero %zero
%max2 = OpConstantComposite %v2 %max %max
%palette0 = OpConstantComposite %v4 %tiny %quarter %half_f %one
%palette1 = OpConstantComposite %v4 %two %negative %third %half_f
%palette2 = OpConstantComposite %v4 %half_f %three_quarters %eighth %one
%green = OpConstantComposite %v4 %zero %one %zero %one
{variables}
%main = OpFunction %void None %fn
%entry = OpLabel
{body}OpReturn
OpFunctionEnd
"#,
        execution = if copy {
            "OpExecutionMode %main PixelInterlockOrderedEXT\n"
        } else {
            ""
        }
    )))
}

fn glyph(x: usize, y: usize) -> bool {
    (x == 3 && (2..=9).contains(&y))
        || (y == 2 && (3..=10).contains(&x))
        || (y == 5 && (3..=8).contains(&x))
}

fn pixels(width: usize, height: usize, bgra: bool, slot: usize, text: bool) -> Vec<u8> {
    let mut result = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = (x + slot * y) % 3;
            let text = text && glyph(x, y);
            if bgra {
                result.extend(if text { [0, 255, 0, 255] } else { BGRA[index] });
            } else {
                result.extend(
                    (if text {
                        [0, 0x3c00, 0, 0x3c00]
                    } else {
                        HALF[index]
                    })
                    .into_iter()
                    .flat_map(u16::to_le_bytes),
                );
            }
        }
    }
    result
}

fn request(
    targets: &[PassLocalTarget],
    vertex: &Arc<Vec<u32>>,
    fragment: &Arc<Vec<u32>>,
    clear: bool,
    offset: (f32, f32),
    rect: (u32, u32, u32, u32),
    writable: bool,
) -> DrawRequest {
    DrawRequest {
        width: W as u32,
        height: H as u32,
        vertex_count: 4,
        instance_count: Some(1),
        raster_sample_count: 1,
        color_sample_count: 1,
        target_identity: Some(targets[0].identity().clone()),
        color_attachment: Some(ColorAttachmentState::new(
            targets[0].identity().resident_format(),
            ColorClearValue::Float([0.0; 4]),
        )),
        load_from_target: !clear,
        skip_readback: true,
        color_write_mask: if writable {
            ColorWriteMask::ALL
        } else {
            ColorWriteMask::NONE
        },
        vert_spirv: vertex.clone(),
        frag_spirv: fragment.clone(),
        secondary_targets: targets
            .iter()
            .skip(1)
            .map(|target| SecondaryColorTarget {
                identity: target.identity().clone(),
                width: W as u32,
                height: H as u32,
                attachment: ColorAttachmentState::new(
                    target.identity().resident_format(),
                    ColorClearValue::Float([0.0; 4]),
                ),
                load: !clear,
                blend: None,
                color_write_mask: if clear {
                    ColorWriteMask::ALL
                } else {
                    ColorWriteMask::NONE
                },
            })
            .collect(),
        storage_buffers: vec![StorageBufferResource {
            binding: 0,
            content: BufferContent::Bytes(Arc::new(
                [W as f32, H as f32, offset.0, offset.1]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )),
        }],
        indexed: Some(IndexedDrawResource {
            index_type: IndexType::U16,
            index_count: 6,
            vertex_offset: 0,
            content: BufferContent::from(
                [0u16, 1, 2, 2, 1, 3]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        }),
        scissors: vec![ScissorResource {
            x: rect.0,
            y: rect.1,
            width: rect.2,
            height: rect.3,
        }],
        ..Default::default()
    }
}

#[test]
#[ignore = "requires exclusive Vulkan GPU, core and synchronization validation"]
fn serial_interlock_gpu_texture3_copy_preserves_offsets_overlap_half_and_cache_guard() {
    texture3_copy_oracle(false);
}

#[test]
#[ignore = "requires exclusive Vulkan GPU, core and synchronization validation"]
fn serial_interlock_gpu_sticky_sampled_and_fragment_buffers_preserve_texture3_pixels() {
    texture3_copy_oracle(true);
}

fn texture3_copy_oracle(sticky: bool) {
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(DeviceId(0xac01), PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let vertex = vertex();
    for (bgra, count) in [(false, 2), (true, 3)] {
        let source_format = if bgra {
            vk::Format::B8G8R8A8_UNORM
        } else {
            vk::Format::R16G16B16A16_SFLOAT
        };
        let targets: Vec<_> = (0..count)
            .map(|slot| {
                PassLocalTarget::new(
                    W as u32,
                    H as u32,
                    if slot == 0 {
                        source_format
                    } else {
                        vk::Format::R16G16B16A16_SFLOAT
                    },
                )
                .unwrap()
            })
            .collect();
        let seed = fragment("seed", count, bgra);
        let paint = fragment("paint", count, bgra);
        let copier = fragment("copy", count, bgra);
        let mut source = pixels(W, H, bgra, 0, true);
        let mut initial = request(
            &targets,
            &vertex,
            &seed,
            true,
            (0.0, 0.0),
            (0, 0, W as u32, H as u32),
            true,
        );
        initial.color_write_mask = ColorWriteMask::NONE;
        initial.target_native_seed = Some(crate::runtime::draw::NativeColorSeed {
            layout: if bgra {
                crate::protocol::pixel_format::TexelLayout::Bgra8
            } else {
                crate::protocol::pixel_format::TexelLayout::Rgba16Float
            },
            bytes: Arc::new(source.clone()),
        });
        engine::execute_draw_request(&state, &initial).unwrap();
        let mut expected = pixels(DW, DH, bgra, 2, false);
        let mut destination = expected.clone();
        let bpp = if bgra { 4 } else { 8 };
        let before = engine::counter_snapshot();
        for (iteration, (rect, offset)) in [
            ((1, 1, 11, 8), (3, 2)),
            ((3, 0, 11, 10), (3, 2)),
            ((0, 0, 3, 3), (0, 0)),
        ]
        .into_iter()
        .enumerate()
        {
            if iteration == 1 {
                engine::execute_draw_request(
                    &state,
                    &request(
                        &targets,
                        &vertex,
                        &paint,
                        false,
                        (0.0, 0.0),
                        (4, 2, 7, 7),
                        true,
                    ),
                )
                .unwrap();
                let green = if bgra {
                    vec![0, 255, 0, 255]
                } else {
                    [0u16, 0x3c00, 0, 0x3c00]
                        .into_iter()
                        .flat_map(u16::to_le_bytes)
                        .collect()
                };
                for y in 2..9 {
                    for x in 4..11 {
                        source[(y * W + x) * bpp..(y * W + x + 1) * bpp].copy_from_slice(&green);
                    }
                }
            }
            let mut req = request(
                &targets,
                &vertex,
                &copier,
                false,
                (offset.0 as f32, offset.1 as f32),
                rect,
                false,
            );
            req.color_input = 1;
            req.storage_textures = vec![GraphicsStorageTexture {
                format: if bgra {
                    StorageImageFormat::Bgra8Unorm
                } else {
                    StorageImageFormat::Rgba16Float
                },
                width: DW as u32,
                height: DH as u32,
                bytes: destination,
                bindings: vec![GraphicsTextureBinding {
                    binding: 1155,
                    access: GraphicsTextureAccess::Storage,
                    stage: vk::ShaderStageFlags::FRAGMENT,
                }],
            }];
            if sticky {
                use crate::runtime::spirv_bind::{
                    FRAG_BUFFER_BINDING_OFFSET, FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
                    TEXTURE_BINDING_BASE,
                };
                for index in [5, 7, 10, 12, 4, 6, 8] {
                    req.sampled_images.push(SampledImageResource {
                        binding: FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                            + TEXTURE_BINDING_BASE
                            + index,
                        array_element: 0,
                        descriptor_count: 1,
                        width: 1,
                        height: 1,
                        layers: 1,
                        kind: reims_vgpu_core::texture_shape::TextureKind::D2,
                        multisampled: false,
                        format: vk::Format::R8G8B8A8_UNORM,
                        source: SampledSource::Bytes(Arc::new(vec![17, 33, 65, 255])),
                        byte_origin: SampledByteOrigin::Synthetic,
                        identity: None,
                        swizzle: Default::default(),
                    });
                }
                for index in [1, 2, 0, 3, 4, 5, 7, 8] {
                    req.storage_buffers.push(StorageBufferResource {
                        binding: FRAG_BUFFER_BINDING_OFFSET + index,
                        content: BufferContent::Bytes(Arc::new(vec![0xa5; 16])),
                    });
                }
            }
            req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
            let output = engine::execute_draw_request(&state, &req);
            if output.is_err() {
                engine::test_quiesce_ring();
            }
            let output = output.unwrap();
            destination = output.storage_textures.into_iter().next().unwrap();
            for y in rect.1 as usize..(rect.1 + rect.3) as usize {
                for x in rect.0 as usize..(rect.0 + rect.2) as usize {
                    let to = ((y + offset.1) * DW + x + offset.0) * bpp;
                    expected[to..to + bpp]
                        .copy_from_slice(&source[(y * W + x) * bpp..(y * W + x + 1) * bpp]);
                }
            }
            assert_eq!(
                destination, expected,
                "texture3 bgra={bgra} draw={iteration}"
            );
            if iteration == 2 {
                let native_interlock = engine::lock_engine()
                    .owner
                    .ctx
                    .as_ref()
                    .unwrap()
                    .features
                    .fragment_shader_pixel_interlock;
                if !native_interlock {
                    let after = engine::counter_snapshot();
                    assert_eq!(
                        after.serial_interlock_draws - before.serial_interlock_draws,
                        3
                    );
                    assert_eq!(
                        after.serial_interlock_primitives - before.serial_interlock_primitives,
                        6
                    );
                    assert_eq!(
                        after.serial_interlock_unused_sampled_draws
                            - before.serial_interlock_unused_sampled_draws,
                        if sticky { 3 } else { 0 },
                    );
                    req.interlock_isolation = None;
                    assert!(
                        matches!(
                            engine::execute_draw_request(&state, &req),
                            Err(DrawError::GraphicsStorage(
                                GraphicsStorageDecline::PixelInterlockUnsupported
                            ))
                        ),
                        "a warm lowered module must never answer the ordinary route"
                    );
                    req.interlock_isolation = Some(Err(InterlockIsolationRefusal::GuestAlias {
                        source: 1,
                        destination: 2,
                    }));
                    assert!(matches!(
                        engine::execute_draw_request(&state, &req),
                        Err(DrawError::GraphicsStorage(
                            GraphicsStorageDecline::SerialInterlock(Refusal::Isolation(_))
                        ))
                    ));
                    if sticky {
                        assert_eq!(req.sampled_images.len(), 7);
                        assert_eq!(req.storage_buffers.len(), 9);
                        for case in 0..4 {
                            req.vert_spirv = vertex.clone();
                            req.frag_spirv = copier.clone();
                            match case {
                                0 => req.vert_spirv = with_resource(&vertex, 709, true),
                                1 => req.frag_spirv = with_resource(&copier, 709, true),
                                2 => req.frag_spirv = with_resource(&copier, 672, false),
                                _ => Arc::make_mut(&mut req.vert_spirv)[1] = 0x0001_0700,
                            }
                            req.interlock_isolation =
                                Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
                            let failure = engine::execute_draw_request(&state, &req);
                            engine::test_quiesce_ring();
                            assert!(
                                matches!(
                                    failure,
                                    Err(DrawError::GraphicsStorage(
                                        GraphicsStorageDecline::SerialInterlock(
                                            Refusal::State("isolated_fragment_storage_required")
                                                | Refusal::Shader("only_storage_and_input_images")
                                                | Refusal::Shader("fragment_buffer_resource")
                                                | Refusal::Shader("vertex_declarations_unproven")
                                        )
                                    )),
                                ),
                                "changed warm shader must refuse case={case}: {failure:?}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            engine::pass_local::read_native_for_test(targets[0].identity()).unwrap(),
            source,
            "source framebuffer contents are preserved except the declared paint"
        );
        for (slot, target) in targets.iter().enumerate().skip(1) {
            assert_eq!(
                engine::pass_local::read_native_for_test(target.identity()).unwrap(),
                pixels(W, H, false, slot, false),
                "untouched secondary{slot}"
            );
        }
        eprintln!("serial_interlock_texture3 bgra={bgra} attachments={count} sticky={sticky} offsets_overlap_half_untouched=PASS cache_guard=PASS");
        drop(targets);
        engine::test_quiesce_ring();
    }
    engine::test_reset_engine(&state);
}
