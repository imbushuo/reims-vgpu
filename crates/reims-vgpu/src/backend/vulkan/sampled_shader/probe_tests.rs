//! Authored sampler regressions and an optional CPU-only retained-AIR probe.

fn synthesized_sampler_shader() -> crate::runtime::m2v_cache::CachedShader {
    let source = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @read_pair(ptr addrspace(1) %texture, ptr addrspace(2) %sampler, ptr addrspace(1) %out, ptr addrspace(1) %destination) {
entry:
  %read_sampler = call ptr addrspace(2) @air.get_read_sampler()
  %a = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %texture, ptr addrspace(2) %sampler, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> <i32 1, i32 0>, i1 false, float 0.0, float 0.0, i32 0)
  %b = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %texture, ptr addrspace(2) %read_sampler, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %av = extractvalue { <4 x float>, i8 } %a, 0
  %bv = extractvalue { <4 x float>, i8 } %b, 0
  %sum = fadd <4 x float> %av, %bv
  store <4 x float> %sum, ptr addrspace(1) %out
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %destination, <2 x i32> zeroinitializer, <4 x float> %sum, i32 0, i32 2)
  ret void
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{!0}
!0 = !{ptr @read_pair, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"texture"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"destination"}
"#;
    let scratch = std::path::PathBuf::from(format!("target/synthesized-sampler-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let translated = metal2vulkan::translate_sanitized_native_reflected(
        source, metal2vulkan::passes::Stage::Kernel, &scratch,
        metal2vulkan::passes::TransformOptions { kernel_local_size: [1, 1, 1], ..Default::default() },
    );
    std::fs::remove_dir_all(&scratch).unwrap();
    let (bytes, reflection) = translated.unwrap();
    crate::runtime::m2v_cache::CachedShader::new(bytes, std::sync::Arc::new(reflection))
}

#[test]
fn synthesized_read_sampler_survives_reflection_and_final_compute_variants() {
    use crate::runtime::{m2v_cache::*, spirv_bind::*};
    let shader = synthesized_sampler_shader();
    let destination = reflected_texture_descriptor(&shader.reflection, 1).unwrap().binding;
    let variant = shader.variant(false, false);
    assert_eq!(variant.samplers.iter().map(|s| s.binding).collect::<Vec<_>>(), [160, 161]);
    assert!(variant.samplers[0].guest_supplied());
    assert!(!variant.samplers[1].guest_supplied());
    assert_eq!(variant.samplers[1].source, ReflectedSamplerSource::SynthesizedRead);
    for mode in [
        crate::protocol::compute::TextureWriteRoundingMode::Default,
        crate::protocol::compute::TextureWriteRoundingMode::TowardZero,
    ] {
        let words = shader.compute_texture_variant(ComputeTextureOptions {
            rounding_mode: mode,
            native_rounding_mode: NATIVE_TEXTURE_WRITE_ROUNDING,
            image_formats: vec![(destination, ImageFormat::Rgba16Float)], rounding_targets: vec![],
        }).unwrap();
        assert_ne!(words.as_ref(), shader.words.as_ref(),
            "half-write format and rounding constants actually specialize this module");
        assert!(descriptor_static_use(&words, 161).is_violation());
        let mut samplers: Vec<_> = variant.samplers.iter().map(|s|
            crate::backend::vulkan::engine::SamplerResource::normalized_default(s.binding)).collect();
        samplers[0].unnormalized_coordinates = true;
        let final_words = super::specialize(&words, &[], &samplers).unwrap();
        assert!(final_words.len() > words.len(), "pixel offset lowering composes after rounding");
        let layout = [0, 32, samplers[0].binding, samplers[1].binding, destination];
        assert!(declared_binding_numbers(&final_words).into_iter().filter(|binding|
            descriptor_static_use(&final_words, *binding).is_violation()
        ).all(|binding| layout.contains(&binding)));
        assert_eq!(validate(&final_words), SpirvValidation::Accepted);
    }
    let relocated = reflected_sampler_descriptors(&shader.reflection, true);
    assert_eq!(relocated[1].binding, 161 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET);
    assert_eq!(relocated[1].source, ReflectedSamplerSource::SynthesizedRead);
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_synthesized_read_sampler_and_half_write_variant() {
    use crate::backend::vulkan::engine::*;
    use crate::runtime::{m2v_cache::*, spirv_bind::*};
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
    let shader = synthesized_sampler_shader();
    let variant = shader.variant(false, false);
    let destination = reflected_texture_descriptor(&shader.reflection, 1).unwrap().binding;
    let words = shader.compute_texture_variant(ComputeTextureOptions {
        rounding_mode: crate::protocol::compute::TextureWriteRoundingMode::Default,
        native_rounding_mode: NATIVE_TEXTURE_WRITE_ROUNDING,
        image_formats: vec![(destination, ImageFormat::Rgba16Float)], rounding_targets: vec![],
    }).unwrap();
    let mut samplers: Vec<_> = variant.samplers.iter()
        .map(|s| SamplerResource::normalized_default(s.binding)).collect();
    assert_eq!(samplers.iter().map(|s| s.binding).collect::<Vec<_>>(), [160, 161]);
    samplers[0].unnormalized_coordinates = true;
    let texel: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0].into_iter().flat_map(f32::to_le_bytes).collect();
    let req = ComputeRequest {
        spirv: super::specialize(&words, &[], &samplers).unwrap(),
        entry: "main".into(), dispatch: ComputeDispatch::Workgroups([1, 1, 1]),
        storage_buffers: vec![ComputeBufferResource { binding: 0, bytes: vec![0xff; 16], writable: true }],
        sampled_images: vec![ComputeSampledImageResource {
            binding: 32, array_element: 0, descriptor_count: 1, format: StorageImageFormat::Rgba32Float,
            width: 2, height: 2, mip_levels: 1, source: ComputeSampledSource::Bytes(texel.repeat(4)),
        }],
        samplers,
        storage_images: vec![ComputeStorageImageResource {
            binding: destination, array_element: 0, descriptor_count: 1,
            format: StorageImageFormat::Rgba16Float, width: 1, height: 1, bytes: vec![0; 8],
            destination: Default::default(), residency: None, seed_skipped: false,
        }],
    };
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let output = execute_compute_request(&state, &req).unwrap();
    let expected: Vec<u8> = [0.5f32, 1.0, 1.5, 2.0].into_iter().flat_map(f32::to_le_bytes).collect();
    assert_eq!(output.buffers[0].bytes, expected);
    let expected_half: Vec<u8> = [0x3800u16, 0x3c00, 0x3e00, 0x4000]
        .into_iter().flat_map(u16::to_le_bytes).collect();
    assert_eq!(output.images[0].bytes().unwrap(), expected_half);
    eprintln!("synthesized sampler161 PASS with pixel offset and native half-write variant");
}

#[test]
fn captured_planar_fragment_specialization_probe() {
    let Some(air) = std::env::var_os("SAMPLED_SHADER_PROBE_AIR") else { return; };
    let texture_index: u32 = std::env::var("SAMPLED_SHADER_PROBE_TEXTURE")
        .unwrap_or_else(|_| "3".into()).parse().unwrap();
    let scratch = std::path::PathBuf::from(format!("target/planar-probe-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let translated = metal2vulkan::translate_reflected(
        std::path::Path::new(&air).to_str().unwrap(),
        metal2vulkan::passes::Stage::Fragment, &scratch,
    );
    std::fs::remove_dir_all(&scratch).unwrap();
    let (bytes, reflection) = translated.expect("translate retained AIR");
    use crate::runtime::spirv_bind;
    let mut words: Vec<u32> = bytes.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    spirv_bind::widen_sampled_bands(&mut words);
    spirv_bind::offset_fragment_sampled_resource_bindings(&mut words);
    let texture = spirv_bind::reflected_texture_descriptor(&reflection, texture_index).unwrap();
    let binding = texture.binding + spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET;
    let samplers: Vec<_> = spirv_bind::reflected_sampler_descriptors(&reflection, true).iter()
        .map(|sampler| match sampler.static_state() {
            Some(state) => crate::runtime::draw::vulkan::reflected_static_sampler_resource(
                "fragment", sampler.binding, state,
            ).unwrap(),
            None => crate::backend::vulkan::engine::SamplerResource::normalized_default(sampler.binding),
        }).collect();
    let output = super::specialize(&words, &[binding], &samplers).unwrap();
    assert_eq!(spirv_bind::validate(&output), spirv_bind::SpirvValidation::Accepted);
    eprintln!("captured planar PASS binding={binding} input_words={} output_words={} samplers={:?}",
        words.len(), output.len(), samplers.iter().map(|s| s.binding).collect::<Vec<_>>());
}
