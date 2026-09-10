use super::*;
use crate::backend::vulkan::engine::{self, BlendStateResource, DrawRequest, SecondaryColorTarget};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

mod primitives;

// Authored shaders; assembler input and binary output stay in pipes.
fn assemble(source: &str) -> Arc<Vec<u32>> {
    let mut child = Command::new("spirv-as")
        .args(["--target-env", "vulkan1.2", "-o", "-", "-"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().expect("spirv-as is required by this explicit GPU regression");
    child.stdin.take().unwrap().write_all(source.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    Arc::new(output.stdout.chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect())
}

const VERTEX: &str = r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Vertex %main "main" %index %position
OpDecorate %index BuiltIn VertexIndex
OpDecorate %position BuiltIn Position
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%vec4 = OpTypeVector %float 4
%input = OpTypePointer Input %uint
%output = OpTypePointer Output %vec4
%index = OpVariable %input Input
%position = OpVariable %output Output
%one = OpConstant %uint 1
%two = OpConstant %uint 2
%zero = OpConstant %float 0
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%main = OpFunction %void None %fn
%entry = OpLabel
%i = OpLoad %uint %index
%shift = OpShiftLeftLogical %uint %i %one
%xbits = OpBitwiseAnd %uint %shift %two
%ybits = OpBitwiseAnd %uint %i %two
%x = OpConvertUToF %float %xbits
%y = OpConvertUToF %float %ybits
%xx = OpFMul %float %x %f2
%yy = OpFMul %float %y %f2
%px = OpFSub %float %xx %f1
%py = OpFSub %float %yy %f1
%p = OpCompositeConstruct %vec4 %px %py %zero %f1
OpStore %position %p
OpReturn
OpFunctionEnd
"#;

const FRAGMENT: &str = r#"
OpCapability Shader
OpCapability InputAttachment
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %tile %output %tile_output
OpExecutionMode %main OriginUpperLeft
OpDecorate %tile DescriptorSet 0
OpDecorate %tile Binding 193
OpDecorate %tile InputAttachmentIndex 1
OpDecorate %output Location 0
OpDecorate %tile_output Location 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%bool = OpTypeBool
%int = OpTypeInt 32 1
%ivec2 = OpTypeVector %int 2
%vec4 = OpTypeVector %float 4
%image = OpTypeImage %float SubpassData 0 0 0 2 Unknown
%uniform = OpTypePointer UniformConstant %image
%out = OpTypePointer Output %vec4
%tile = OpVariable %uniform UniformConstant
%output = OpVariable %out Output
%tile_output = OpVariable %out Output
%zero = OpConstant %int 0
%coord = OpConstantComposite %ivec2 %zero %zero
%quarter = OpConstant %float 0.25
%eighth = OpConstant %float 0.125
%sixteenth = OpConstant %float 0.0625
%nine = OpConstant %float 9
%negative = OpConstant %float -3
%zero_float = OpConstant %float 0
%one_float = OpConstant %float 1
%main = OpFunction %void None %fn
%entry = OpLabel
%img = OpLoad %image %tile
%prior = OpImageRead %vec4 %img %coord
%r = OpCompositeExtract %float %prior 0
%g = OpCompositeExtract %float %prior 1
%b = OpCompositeExtract %float %prior 2
%a = OpCompositeExtract %float %prior 3
%next = OpFAdd %float %r %quarter
%scaled_r = OpFMul %float %next %quarter
%g_ok = OpFOrdEqual %bool %g %zero_float
%b_ok = OpFOrdEqual %bool %b %zero_float
%a_ok = OpFOrdEqual %bool %a %one_float
%gb_ok = OpLogicalAnd %bool %g_ok %b_ok
%defaults_ok = OpLogicalAnd %bool %gb_ok %a_ok
%out_r = OpSelect %float %defaults_ok %scaled_r %negative
%out_g = OpFAdd %float %g %eighth
%out_b = OpFAdd %float %b %sixteenth
%out_a = OpFMul %float %a %quarter
%value = OpCompositeConstruct %vec4 %out_r %out_g %out_b %out_a
%tile_value = OpCompositeConstruct %vec4 %next %nine %negative %quarter
OpStore %output %value
OpStore %tile_output %tile_value
OpReturn
OpFunctionEnd
"#;

fn source_over() -> BlendStateResource {
    use reims_vgpu_core::blend::*;
    BlendStateResource {
        src_rgb: MTL_BLEND_FACTOR_ONE,
        dst_rgb: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_rgb: MTL_BLEND_OPERATION_ADD,
        src_alpha: MTL_BLEND_FACTOR_ONE,
        dst_alpha: MTL_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA,
        op_alpha: MTL_BLEND_OPERATION_ADD,
    }
}

fn native_pair_request() -> DrawEncodeRequest {
    let mut req = request();
    let tile = req.colors[0].clone();
    req.colors[0].format = crate::protocol::pixel_format::MTL_FORMAT_R32_FLOAT;
    req.colors[0].texture_ref = 36;
    req.colors[0].clear_color = [0.0; 4];
    req.colors.push(ColorRtRequest { slot: 1, ..tile });
    req
}

fn native_draw(
    owner: &VulkanRenderPass,
    req: &DrawEncodeRequest,
    vertex: Arc<Vec<u32>>,
    fragment: Arc<Vec<u32>>,
    blend_tile: bool,
) -> DrawRequest {
    let attachment = |color: &ColorRtRequest| crate::backend::vulkan::translate::pixel::
        memoryless_color_attachment(color.format).unwrap().with_clear(color.clear_color);
    DrawRequest {
        vert_spirv: vertex, frag_spirv: fragment,
        width: 8, height: 4, vertex_count: 3, instance_count: Some(1),
        raster_sample_count: 1, color_sample_count: 1,
        primitive_topology: engine::PrimitiveTopology(
            reims_vgpu_core::topology::PrimitiveType::Triangle),
        target_identity: Some(owner.identity(&req.colors[0]).unwrap().clone()),
        color_attachment: Some(attachment(&req.colors[0])),
        blend: Some(source_over()), load_from_target: req.continues_render_pass,
        color0_declared: Some(if req.continues_render_pass {
            reims_vgpu_protocol::pass_action::LoadAction::Load
        } else { reims_vgpu_protocol::pass_action::LoadAction::Clear }),
        secondary_targets: vec![SecondaryColorTarget {
            identity: owner.identity(&req.colors[1]).unwrap().clone(),
            width: 8, height: 4,
            attachment: attachment(&req.colors[1]), load: req.continues_render_pass,
            blend: blend_tile.then(source_over), color_write_mask: Default::default(),
        }],
        color_input: 2, skip_readback: true,
        continues_render_pass: req.continues_render_pass,
        render_pass_continues: req.render_pass_continues,
        ..Default::default()
    }
}

#[test]
#[ignore = "requires the parent's exclusive GPU slot"]
fn memoryless_r32float_gpu_split_fetch_blend_and_retirement() {
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let vertex = assemble(VERTEX);
    let fragment = assemble(FRAGMENT);
    for blend_tile in [false, true] {
        let mut req = native_pair_request();
        let mut pass = VulkanRenderPass::default();
        let mut expected = [[0.0f32; 4]; 32];
        let mut tiles = [2.0000009536743164f32; 32];
        let mut identities = Vec::new();
        for (index, (x, y, width, height)) in [(0, 0, 8, 4), (2, 0, 6, 4), (0, 1, 5, 2)]
            .into_iter().enumerate()
        {
            req.continues_render_pass = index != 0;
            req.render_pass_continues = index != 2;
            for color in &mut req.colors {
                color.load_action = if index == 0 { MTL_LOAD_ACTION_CLEAR } else { MTL_LOAD_ACTION_LOAD };
            }
            for py in y..y + height {
                for px in x..x + width {
                    let p = (py * 8 + px) as usize;
                    let next = tiles[p] + 0.25;
                    let source = [next * 0.25, 0.125, 0.0625, 0.25];
                    for (dst, src) in expected[p].iter_mut().zip(source) { *dst = src + *dst * 0.75; }
                    tiles[p] = if blend_tile { next + tiles[p] * 0.75 } else { next };
                }
            }
            let result = pass.with_draw(&mut req, |owner, req| {
                if index == 0 {
                    identities.extend(req.colors.iter().map(|color| owner.identity(color).unwrap().clone()));
                }
                let mut draw = native_draw(owner, req, vertex.clone(), fragment.clone(), blend_tile);
                draw.scissors = vec![engine::ScissorResource { x, y, width, height }];
                engine::execute_draw_request(&state, &draw).expect("native memoryless draw");
                if index == 2 {
                    let pixels = engine::pass_local::read_native_for_test(
                        draw.target_identity.as_ref().unwrap(),
                    ).expect("test-only native readback, not a guest Store");
                    assert_eq!(pixels.len(), 32 * 4);
                    for (p, texel) in pixels.chunks_exact(4).enumerate() {
                        for (component, bytes) in texel.chunks_exact(4).enumerate() {
                            let actual = f32::from_le_bytes(bytes.try_into().unwrap());
                            if p == 0 {
                                assert_eq!(actual, expected[p][component],
                                    "the untouched first pixel retains R32 precision and default G/B/A");
                            } else {
                                assert!((actual - expected[p][component]).abs() <= 0.000001,
                                    "pixel={p} component={component} secondary_blend={blend_tile} \
                                     actual={actual} expected={}", expected[p][component]);
                            }
                        }
                    }
                }
                (EncodeStatus::Ok, None)
            });
            assert!(matches!(result.0, EncodeStatus::Ok), "pass admission: {:?}", result.0);
        }
        assert!(identities.iter().all(|identity| !engine::resident_content_ready(identity)),
            "end-of-pass retires both native images, not just their Rust names");
    }
}
