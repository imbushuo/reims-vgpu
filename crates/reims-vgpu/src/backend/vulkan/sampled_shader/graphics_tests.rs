use super::specialize;
use crate::backend::vulkan::{engine::*, planar, translate};
use crate::protocol::planar::{BackingFormat, SampleFormat};
use std::sync::Arc;

fn shader(vertex: bool, sample: bool, binding: u32, sampler: u32, scale: f32) -> Vec<u32> {
    let (entry, execution, decorations, variables, position) = if vertex {
        ("Vertex", "", "OpDecorate %index BuiltIn VertexIndex\nOpDecorate %position BuiltIn Position",
         "%index = OpVariable %input_uint Input\n%position = OpVariable %output_vec Output",
         "%i = OpLoad %uint %index\n%x_test = OpIEqual %bool %i %one_uint\n%y_test = OpIEqual %bool %i %two_uint\n%x = OpSelect %float %x_test %three %negative\n%y = OpSelect %float %y_test %three %negative\n%p = OpCompositeConstruct %vec4 %x %y %zero %one\nOpStore %position %p")
    } else {
        ("Fragment", "OpExecutionMode %main OriginUpperLeft",
         "OpDecorate %input_color Location 0", "%input_color = OpVariable %input_vec Input", "")
    };
    let interface = if vertex { "%index %position %color" } else { "%input_color %color" };
    let resource_decorations = if sample {
        format!("OpDecorate %texture DescriptorSet 0\nOpDecorate %texture Binding {binding}\nOpDecorate %sampler DescriptorSet 0\nOpDecorate %sampler Binding {sampler}")
    } else { String::new() };
    let resource_variables = if sample {
        "%texture = OpVariable %image_ptr UniformConstant\n%sampler = OpVariable %sampler_ptr UniformConstant"
    } else { "" };
    let color = if sample {
        "%image = OpLoad %image_type %texture\n%smp = OpLoad %sampler_type %sampler\n%combined = OpSampledImage %sampled_type %image %smp\n%value = OpImageSampleExplicitLod %vec4 %combined %uv Lod %zero\n%scaled = OpVectorTimesScalar %vec4 %value %scale\nOpStore %color %scaled"
    } else if vertex {
        "OpStore %color %white"
    } else {
        "%value = OpLoad %vec4 %input_color\nOpStore %color %value"
    };
    let source = format!(r#"OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint {entry} %main "main" {interface}
{execution}
OpDecorate %color Location 0
{decorations}
{resource_decorations}
%void = OpTypeVoid
%function = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec4 = OpTypeVector %float 4
%input_uint = OpTypePointer Input %uint
%input_vec = OpTypePointer Input %vec4
%output_vec = OpTypePointer Output %vec4
%image_type = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampler_type = OpTypeSampler
%sampled_type = OpTypeSampledImage %image_type
%image_ptr = OpTypePointer UniformConstant %image_type
%sampler_ptr = OpTypePointer UniformConstant %sampler_type
%zero = OpConstant %float 0
%one = OpConstant %float 1
%half = OpConstant %float 0.5
%negative = OpConstant %float -1
%three = OpConstant %float 3
%scale = OpConstant %float {scale:.1}
%one_uint = OpConstant %uint 1
%two_uint = OpConstant %uint 2
%uv = OpConstantComposite %vec2 %half %half
%white = OpConstantComposite %vec4 %one %one %one %one
%color = OpVariable %output_vec Output
{variables}
{resource_variables}
%main = OpFunction %void None %function
%label = OpLabel
{position}
{color}
OpReturn
OpFunctionEnd
"#);
    let scratch = std::path::PathBuf::from(format!("target/planar-graphics-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let input = scratch.join("shader.spvasm");
    let output = scratch.join("shader.spv");
    std::fs::write(&input, source).unwrap();
    let assembled = std::process::Command::new("spirv-as")
        .args(["--target-env", "vulkan1.0"]).arg(&input).arg("-o").arg(&output)
        .output().unwrap();
    assert!(assembled.status.success(), "{}", String::from_utf8_lossy(&assembled.stderr));
    let bytes = std::fs::read(&output).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();
    bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_planar_vertex_fragment_oracles_and_q11_ties() {
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    for vertex in [true, false] {
        let offset = if vertex { 0 } else { crate::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET };
        let binding = 32 + offset;
        let sampler_binding = 160 + offset;
        for case in 0..5 {
            let format = if case % 2 == 0 { SampleFormat::Ycbcr10_420TwoPlane } else {
                SampleFormat::Rgb10_420TwoPlane
            };
            let backing = if case < 2 { BackingFormat::VideoRange } else { BackingFormat::FullRange };
            let mut image = planar::tests::image(format, backing);
            let expected = if case == 4 {
                image.width = 2; image.height = 2;
                image.bytes = [0u16, 1, 2, 3].into_iter().flat_map(|q|
                    [q, q, q, 2048].into_iter().flat_map(|q|
                        crate::protocol::planar::sampling::q11_half(q).to_le_bytes())
                ).collect();
                [0.5f32, 0.5, 0.5, 1.0]
            } else if format == SampleFormat::Rgb10_420TwoPlane {
                [768.0 / 1023.0, 512.0 / 1023.0, 640.0 / 1023.0, 1.0]
            } else if backing == BackingFormat::VideoRange {
                [1867.0 / 2048.0, 529.0 / 2048.0, 1566.0 / 2048.0, 1.0]
            } else { [1742.0 / 2048.0, 571.0 / 2048.0, 1479.0 / 2048.0, 1.0] };
            for filter in if case == 4 { 1..2 } else { 0..2 } {
                let mut sampler = SamplerResource::normalized_default(sampler_binding);
                sampler.min_filter = filter; sampler.mag_filter = filter;
                let q11 = if format == SampleFormat::Ycbcr10_420TwoPlane { vec![binding] } else { vec![] };
                let scale = if case == 4 { 512.0 } else { 1.0 };
                let vertex_words = shader(true, vertex, binding, sampler_binding, scale);
                let fragment_words = shader(false, !vertex, binding, sampler_binding, scale);
                let req = DrawRequest {
                    vert_spirv: Arc::new(specialize(&vertex_words, &q11, &[sampler.clone()]).unwrap()),
                    frag_spirv: Arc::new(specialize(&fragment_words, &q11, &[sampler.clone()]).unwrap()),
                    width: 4, height: 4, vertex_count: 3, instance_count: Some(1),
                    primitive_topology: PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Triangle),
                    sampled_images: vec![SampledImageResource {
                        binding, array_element: 0, descriptor_count: 1,
                        width: image.width, height: image.height, layers: 1,
                        kind: reims_vgpu_core::texture_shape::TextureKind::D2, multisampled: false,
                        source: SampledSource::Bytes(Arc::new(image.bytes.clone())),
                        format: translate::pixel::vk_sampled_bytes(image.byte_format()),
                        byte_origin: Default::default(), identity: None, swizzle: Default::default(),
                    }],
                    samplers: vec![sampler], ..Default::default()
                };
                let output = execute_draw_request(&state, &req).unwrap();
                assert_eq!(output.pixels.len(), 64);
                for pixel in output.pixels.chunks_exact(4) {
                    let actual = if output.pixels_bgra { [pixel[2], pixel[1], pixel[0], pixel[3]] } else {
                        pixel.try_into().unwrap()
                    };
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        assert!((i32::from(actual) - (expected * 255.0).round() as i32).abs() <= 1,
                            "vertex={vertex} case={case} filter={filter}: {pixel:?}");
                    }
                }

                eprintln!("planar graphics PASS vertex={vertex} case={case} filter={filter}");
            }
        }
    }
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_scalar_half_native_upload_retains_small_values() {
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    assert!(supports_sampled_layout_linear_filter(crate::protocol::pixel_format::TexelLayout::R16Float));
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    for vertex in [true, false] {
        let offset = if vertex { 0 } else { crate::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET };
        let binding = 32 + offset;
        let sampler_binding = 160 + offset;
        // This scalar half rounds to zero through RGBA8. Amplifying it after
        // sampling distinguishes a native upload from an eight-bit round trip.
        let req = DrawRequest {
            vert_spirv: Arc::new(shader(true, vertex, binding, sampler_binding, 512.0)),
            frag_spirv: Arc::new(shader(false, !vertex, binding, sampler_binding, 512.0)),
            width: 4, height: 2, vertex_count: 3, instance_count: Some(1),
            primitive_topology: PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Triangle),
            sampled_images: vec![SampledImageResource {
                binding, array_element: 0, descriptor_count: 1, width: 4, height: 2, layers: 1,
                kind: reims_vgpu_core::texture_shape::TextureKind::D2, multisampled: false,
                source: SampledSource::Bytes(Arc::new(0x1001u16.to_le_bytes().repeat(8))),
                format: ash::vk::Format::R16_SFLOAT, byte_origin: Default::default(),
                identity: None, swizzle: Default::default(),
            }],
            samplers: vec![SamplerResource::normalized_default(sampler_binding)],
            ..Default::default()
        };
        let output = execute_draw_request(&state, &req).unwrap();
        assert_eq!(output.pixels.len(), 32);
        for pixel in output.pixels.chunks_exact(4) {
            let red = if output.pixels_bgra { 2 } else { 0 };
            let blue = 2 - red;
            assert!((i32::from(pixel[red]) - 64).abs() <= 1, "{pixel:?}");
            assert_eq!(pixel[1], 0); assert_eq!(pixel[blue], 0); assert_eq!(pixel[3], 255);
        }
        eprintln!("scalar-half graphics PASS vertex={vertex}");
    }
}
