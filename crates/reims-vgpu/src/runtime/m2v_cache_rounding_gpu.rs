use super::{CachedShader, ComputeTextureOptions};
use crate::backend::vulkan::engine::*;
use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
use crate::protocol::compute::TextureWriteRoundingMode as DescriptorMode;
use crate::runtime::spirv_bind::ImageFormat;
use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::texture_write_rounding::TextureWriteRoundingMode;
use std::sync::Arc;

// A binary search of representable values is independent of the shader's bit-shift converter.
fn expected_half(value: f32, nearest: bool) -> u16 {
    let sign = ((value.to_bits() >> 16) & 0x8000) as u16;
    if value.is_nan() { return sign | 0x7e00; }
    if value.is_infinite() { return sign | 0x7c00; }
    let value = value.abs();
    if value >= 65504.0 {
        return sign | if nearest && value >= 65520.0 { 0x7c00 } else { 0x7bff };
    }
    let widen = crate::protocol::pixel_format::f16_to_f32;
    let (mut low, mut high) = (0u16, 0x7bffu16);
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if widen(mid) <= value { low = mid; } else { high = mid; }
    }
    if !nearest { return sign | low; }
    let middle = (f64::from(widen(low)) + f64::from(widen(high))) * 0.5;
    sign | if f64::from(value) > middle || (f64::from(value) == middle && low & 1 != 0) {
        high
    } else { low }
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU slot"]
fn vulkan_gpu_texture_rounding_air_descriptor_matrix() {
    let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let values = [
        f32::from_bits(0x3f80_3000), f32::from_bits(0xbf80_3000),
        f32::from_bits(0x3300_0000), f32::from_bits(0x33c0_0000),
        f32::from_bits(0x387f_f000), 65520.0, 70000.0, -70000.0, -0.0,
        f32::INFINITY, f32::NAN,
    ];
    let input: Vec<u8> = values.iter().flat_map(|value| [*value; 4])
        .flat_map(f32::to_le_bytes).collect();
    let scratch = std::path::PathBuf::from(format!("target/rounding-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for suffix in ["", "rte.", "rtz."] {
        let source = format!(r#"
target triple = "spirv-unknown-vulkan1.2"
define void @probe(ptr addrspace(1) %image, ptr addrspace(1) %values, i32 %index) {{
entry:
  %pointer = getelementptr <4 x float>, ptr addrspace(1) %values, i32 %index
  %value = load <4 x float>, ptr addrspace(1) %pointer, align 16
  %coordinate = insertelement <2 x i32> zeroinitializer, i32 %index, i32 0
  call void @air.write_texture_2d.{suffix}v4f32(ptr addrspace(1) %image, <2 x i32> %coordinate, <4 x float> %value, i32 0, i32 2)
  ret void
}}
declare void @air.write_texture_2d.{suffix}v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.kernel = !{{!0}}
!0 = !{{ptr @probe, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"image"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"values"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}}
"#);
        let (bytes, reflection) = metal2vulkan::translate_sanitized_native_reflected(
            &source, Stage::Kernel, &scratch,
            TransformOptions {
                kernel_local_size: [1, 1, 1],
                kernel_dispatch: Some(metal2vulkan::reflect::KernelDispatch::Workgroups),
                ..Default::default()
            },
        ).unwrap();
        let shader = CachedShader::new(bytes, Arc::new(reflection));
        let binding = metal2vulkan::reflect::DEFAULT_DESCRIPTOR_LAYOUT.storage_textures.start;
        for descriptor in [DescriptorMode::Default, DescriptorMode::TowardZero, DescriptorMode::ToNearestEven] {
            for (format, spirv_format) in [
                (StorageImageFormat::Rgba16Float, ImageFormat::Rgba16Float),
                (StorageImageFormat::Rgba32Float, ImageFormat::Rgba32Float),
            ] {
                let words = shader.compute_texture_variant(ComputeTextureOptions {
                    rounding_mode: descriptor,
                    native_rounding_mode: TextureWriteRoundingMode::TowardZero,
                    image_formats: vec![(binding, spirv_format)],
                    rounding_targets: vec![],
                }).unwrap();
                let request = ComputeRequest {
                    spirv: (*words).clone(),
                    entry: shader.reflection.entry_point.clone().unwrap(),
                    dispatch: ComputeDispatch::Workgroups([values.len() as u32, 1, 1]),
                    storage_buffers: vec![ComputeBufferResource { binding: 0, bytes: input.clone(), writable: false }],
                    sampled_images: vec![],
                    samplers: vec![],
                    storage_images: vec![ComputeStorageImageResource {
                        binding, array_element: 0, descriptor_count: 1, format,
                        width: values.len() as u32, height: 1,
                        bytes: vec![0; values.len() * format.bytes_per_texel()],
                        destination: ComputeImageDestination::Host,
                        residency: None, seed_skipped: false,
                    }],
                };
                let output = execute_compute_request(&state, &request).unwrap();
                let ComputeImageResult::Bytes(bytes) = &output.images[0] else { panic!("host readback") };
                if format == StorageImageFormat::Rgba16Float {
                    for (i, bytes) in bytes.chunks_exact(2).enumerate() {
                        let actual = u16::from_le_bytes(bytes.try_into().unwrap());
                        let value = values[i / 4];
                        if value.is_nan() {
                            assert!(actual & 0x7c00 == 0x7c00 && actual & 0x3ff != 0);
                        } else {
                            assert_eq!(actual, expected_half(value, suffix == "rte."),
                                "AIR={suffix:?} descriptor={descriptor:?} pixel={} value={value}", i / 4);
                        }
                    }
                } else {
                    for (i, bytes) in bytes.chunks_exact(4).enumerate() {
                        let actual = f32::from_le_bytes(bytes.try_into().unwrap());
                        let expected = values[i / 4];
                        if expected.is_nan() { assert!(actual.is_nan()); }
                        else { assert_eq!(actual.to_bits(), expected.to_bits()); }
                    }
                }
                eprintln!("rounding GPU PASS AIR={suffix:?} descriptor={descriptor:?} format={format:?}");
            }
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}
