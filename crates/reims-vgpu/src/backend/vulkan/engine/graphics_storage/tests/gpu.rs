use super::*;
use crate::backend::vulkan::engine;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

fn shader(vertex: bool, interlock: bool, after_completion: bool) -> String {
    let (entry, interface, execution, decorations, variables, body) = if vertex {
        ("Vertex", "%index %position %storage", "",
         "OpDecorate %index BuiltIn VertexIndex\nOpDecorate %position BuiltIn Position\nOpDecorate %storage Binding 480",
         "%index = OpVariable %input_uint Input\n%position = OpVariable %output_vec Output",
         r#"
%i = OpLoad %uint %index
%first = OpIEqual %bool %i %u0
OpSelectionMerge %merge None
OpBranchConditional %first %write %merge
%write = OpLabel
%img = OpLoad %storage_image %storage
OpImageWrite %img %vertex_coord %vertex_value
OpBranch %merge
%merge = OpLabel
%is_x = OpIEqual %bool %i %u1
%is_y = OpIEqual %bool %i %u2
%x = OpSelect %float %is_x %three %negative
%y = OpSelect %float %is_y %three %negative
%p = OpCompositeConstruct %vec4 %x %y %zero %one
OpStore %position %p
"#)
    } else {
        ("Fragment", "%position %color %storage %read_texture",
         "OpExecutionMode %main OriginUpperLeft",
         "OpDecorate %position BuiltIn FragCoord\nOpDecorate %color Location 0\nOpDecorate %storage Binding 1152\nOpDecorate %read_texture DescriptorSet 0\nOpDecorate %read_texture Binding 704",
         "%position = OpVariable %input_vec Input\n%color = OpVariable %output_vec Output\n%read_texture = OpVariable %sampled_ptr UniformConstant",
         r#"
%p = OpLoad %vec4 %position
%xy = OpVectorShuffle %vec2 %p %p 0 1
%coord = OpConvertFToU %uvec2 %xy
%img = OpLoad %storage_image %storage
OpImageWrite %img %coord %fragment_value
OpMemoryBarrier %u1 %image_acquire_release
%written = OpImageRead %vec4 %img %coord
%sampled = OpLoad %sampled_image %read_texture
%seed = OpImageFetch %vec4 %sampled %untouched_coord Lod %u0
%seed_matches = OpFOrdEqual %bvec4 %seed %seed_value
%seed_ok = OpAll %bool %seed_matches
%seed_ok4 = OpCompositeConstruct %bvec4 %seed_ok %seed_ok %seed_ok %seed_ok
%value = OpSelect %vec4 %seed_ok4 %written %black
"#)
    };
    let body = if !vertex && after_completion { r#"
%p = OpLoad %vec4 %position
%xy = OpVectorShuffle %vec2 %p %p 0 1
%coord = OpConvertFToU %uvec2 %xy
%sampled = OpLoad %sampled_image %read_texture
%value = OpImageFetch %vec4 %sampled %coord Lod %u0
"# } else { body };
    let capabilities = if interlock && !vertex {
        "OpCapability FragmentShaderPixelInterlockEXT\nOpExtension \"SPV_EXT_fragment_shader_interlock\""
    } else { "" };
    let ordering = if interlock && !vertex {
        "OpExecutionMode %main PixelInterlockOrderedEXT"
    } else { "" };
    let begin = if interlock && !vertex { "OpBeginInvocationInterlockEXT" } else { "" };
    let end = if interlock && !vertex { "OpEndInvocationInterlockEXT" } else { "" };
    let output = if vertex { "" } else { "OpStore %color %value" };
    format!(r#"
OpCapability Shader
{capabilities}
OpMemoryModel Logical GLSL450
OpEntryPoint {entry} %main "main" {interface}
{execution}
{ordering}
OpDecorate %storage DescriptorSet 0
OpDecorate %storage Coherent
{decorations}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%bvec4 = OpTypeVector %bool 4
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec4 = OpTypeVector %float 4
%uvec2 = OpTypeVector %uint 2
%input_uint = OpTypePointer Input %uint
%input_vec = OpTypePointer Input %vec4
%output_vec = OpTypePointer Output %vec4
%storage_image = OpTypeImage %float 2D 0 0 0 2 Rgba16f
%sampled_image = OpTypeImage %float 2D 0 0 0 1 Unknown
%storage_ptr = OpTypePointer UniformConstant %storage_image
%sampled_ptr = OpTypePointer UniformConstant %sampled_image
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u2 = OpConstant %uint 2
%u3 = OpConstant %uint 3
%image_acquire_release = OpConstant %uint 2056
%zero = OpConstant %float 0
%one = OpConstant %float 1
%negative = OpConstant %float -1
%three = OpConstant %float 3
%eighth = OpConstant %float 0.125
%quarter = OpConstant %float 0.25
%half = OpConstant %float 0.5
%three_quarters = OpConstant %float 0.75
%tiny = OpConstant %float 0.000488758087158203125
%third = OpConstant %float 0.333251953125
%vertex_coord = OpConstantComposite %uvec2 %u3 %u0
%untouched_coord = OpConstantComposite %uvec2 %u3 %u1
%vertex_value = OpConstantComposite %vec4 %three_quarters %half %quarter %one
%fragment_value = OpConstantComposite %vec4 %half %quarter %eighth %one
%seed_value = OpConstantComposite %vec4 %tiny %third %three_quarters %one
%black = OpConstantComposite %vec4 %zero %zero %zero %zero
%storage = OpVariable %storage_ptr UniformConstant
{variables}
%main = OpFunction %void None %fn
%entry = OpLabel
{begin}
{body}
{end}
{output}
OpReturn
OpFunctionEnd
"#)
}

fn assemble(source: &str) -> (Arc<Vec<u32>>, Vec<u8>) {
    let mut child = Command::new("spirv-as")
        .args(["--target-env", "vulkan1.2", "-o", "-", "-"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().expect("explicit graphics storage fixture requires spirv-as");
    child.stdin.take().unwrap().write_all(source.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let words = output.stdout.chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap())).collect();
    (Arc::new(words), output.stdout)
}

#[test]
#[ignore = "CPU-only optional spirv-as/spirv-val fixture check"]
fn graphics_storage_shaders_cpu_validate() {
    for (vertex, interlock, after_completion) in [
        (true, false, false), (false, false, false), (false, true, false),
        (false, false, true), (false, true, true),
    ] {
        let (_, bytes) = assemble(&shader(vertex, interlock, after_completion));
        let mut child = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2", "-"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().unwrap();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
#[ignore = "requires the parent's exclusive Vulkan GPU slot"]
fn graphics_storage_gpu_native_writes_alias_reads_and_completed_output() {
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let vertex = assemble(&shader(true, false, false)).0;
    let seed_texel = [0x1001u16, 0x3555, 0x3a00, 0x3c00]
        .into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let fragment_texel = [0x3800u16, 0x3400, 0x3000, 0x3c00]
        .into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    let vertex_texel = [0x3a00u16, 0x3800, 0x3400, 0x3c00]
        .into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
    for interlock in [false, true] {
        let fragment = assemble(&shader(false, interlock, false)).0;
        let completed_fragment = assemble(&shader(false, interlock, true)).0;
        for bgra in [false, true] {
            for defer_attachment in [false, true] {
                let pixel_format = if bgra { crate::protocol::pixel_format::MTL_FORMAT_BGRA8_UNORM }
                    else { crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM };
                let attachment = crate::backend::vulkan::translate::pixel::color_attachment(
                    pixel_format,
                ).unwrap().0.with_clear([0.0; 4]);
                let mut req = request();
                req.vert_spirv = vertex.clone();
                req.frag_spirv = fragment.clone();
                req.width = 4; req.height = 2;
                req.vertex_count = 3; req.instance_count = Some(1);
                req.primitive_topology = engine::PrimitiveTopology(
                    reims_vgpu_core::topology::PrimitiveType::Triangle);
                req.raster_sample_count = 1; req.color_sample_count = 1;
                req.scissors = vec![engine::ScissorResource {
                    x: 0, y: 0, width: 2, height: 2,
                }];
                req.color_attachment = Some(attachment);
                req.skip_readback = defer_attachment;
                req.render_pass_continues = defer_attachment;
                req.target_identity = defer_attachment.then_some(engine::TargetIdentity::Surface {
                    id: 77, width: 4, height: 2, generation: 1, format: attachment.format(),
                });
                let texture = &mut req.storage_textures[0];
                texture.bytes = seed_texel.repeat(8);
                texture.bindings[0].binding = 480;
                texture.bindings[1].binding = 1152;
                texture.bindings[2].binding = 704;
                let out = engine::execute_draw_request(&state, &req).unwrap();
                assert_eq!(out.storage_textures.len(), 1);
                assert_eq!(out.storage_textures[0].len(), 64);
                for (index, texel) in out.storage_textures[0].chunks_exact(8).enumerate() {
                    let expected = if index == 3 { &vertex_texel }
                        else if index % 4 < 2 { &fragment_texel } else { &seed_texel };
                    assert_eq!(texel, expected, "texel={index} interlock={interlock}");
                }
                assert_eq!(out.pixels_bgra, bgra);
                if defer_attachment {
                    assert!(out.pixels.is_empty());
                } else {
                    assert_eq!(out.pixels.len(), 32);
                    for (index, pixel) in out.pixels.chunks_exact(4).enumerate() {
                        let rgba = if bgra { [pixel[2], pixel[1], pixel[0], pixel[3]] }
                            else { pixel.try_into().unwrap() };
                        let expected = if index % 4 < 2 { [128, 64, 32, 255] } else { [0; 4] };
                        for (actual, expected) in rgba.into_iter().zip(expected) {
                            assert!((i32::from(actual) - expected).abs() <= 1,
                                "same-reference storage read and untouched sampled alias: {rgba:?}");
                        }
                    }
                    // Keep completed-draw visibility independent of intra-invocation
                    // sampling. The fenced-sampled-alias regression below separately
                    // checks the native texture.fence() contract.
                    req.storage_textures[0].bytes = out.storage_textures[0].clone();
                    req.frag_spirv = completed_fragment.clone();
                    req.skip_readback = false;
                    req.render_pass_continues = false;
                    let sampled = engine::execute_draw_request(&state, &req).unwrap();
                    assert_eq!(sampled.storage_textures, out.storage_textures);
                    assert_eq!(sampled.pixels.len(), 32);
                    assert_eq!(sampled.pixels_bgra, bgra);
                    for (index, pixel) in sampled.pixels.chunks_exact(4).enumerate() {
                        let rgba = if bgra { [pixel[2], pixel[1], pixel[0], pixel[3]] }
                            else { pixel.try_into().unwrap() };
                        let expected = if index % 4 < 2 { [128, 64, 32, 255] } else { [0; 4] };
                        for (actual, expected) in rgba.into_iter().zip(expected) {
                            assert!((i32::from(actual) - expected).abs() <= 1,
                                "next draw's sampled alias must observe completed writes: {rgba:?}");
                        }
                    }
                }
            }
        }
    }
    engine::test_reset_engine(&state);
}

#[test]
#[ignore = "requires the parent's exclusive Vulkan GPU slot"]
fn graphics_storage_gpu_fenced_sampled_alias_visibility() {
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let current_read = r#"%written = OpImageRead %vec4 %img %coord
%sampled = OpLoad %sampled_image %read_texture
%seed = OpImageFetch %vec4 %sampled %untouched_coord Lod %u0
%seed_matches = OpFOrdEqual %bvec4 %seed %seed_value
%seed_ok = OpAll %bool %seed_matches
%seed_ok4 = OpCompositeConstruct %bvec4 %seed_ok %seed_ok %seed_ok %seed_ok
%value = OpSelect %vec4 %seed_ok4 %written %black"#;
    for interlock in [false, true] {
        let source = shader(false, interlock, false);
        assert!(source.contains(current_read));
        let source = source.replace(current_read,
            "%sampled = OpLoad %sampled_image %read_texture\n%value = OpImageFetch %vec4 %sampled %coord Lod %u0");
        let (fragment, bytes) = assemble(&source);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/graphics-test-artifacts/fenced-sampled-alias-interlock-{interlock}.spv"));
        std::fs::write(path, bytes).unwrap();
        let mut req = request();
        req.vert_spirv = assemble(&shader(true, false, false)).0;
        req.frag_spirv = fragment;
        req.width = 4; req.height = 2;
        req.vertex_count = 3; req.instance_count = Some(1);
        req.raster_sample_count = 1; req.color_sample_count = 1;
        req.primitive_topology = engine::PrimitiveTopology(
            reims_vgpu_core::topology::PrimitiveType::Triangle);
        req.scissors = vec![engine::ScissorResource { x: 0, y: 0, width: 2, height: 2 }];
        req.skip_readback = false;
        let texture = &mut req.storage_textures[0];
        texture.bytes = [0x1001u16, 0x3555, 0x3a00, 0x3c00]
            .into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>().repeat(8);
        texture.bindings[0].binding = 480;
        texture.bindings[1].binding = 1152;
        texture.bindings[2].binding = 704;
        let out = engine::execute_draw_request(&state, &req).unwrap();
        assert_eq!(out.pixels.len(), 32);
        for index in [0usize, 1, 4, 5] {
            let pixel = &out.pixels[index * 4..index * 4 + 4];
            assert_eq!(pixel, &[128, 64, 32, 255],
                "explicit texture-fence alias visibility: interlock={interlock} pixel={index}");
        }
    }
    engine::test_reset_engine(&state);
}
