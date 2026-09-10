use super::*;

const PRIVATE_KERNEL: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
%Block = type { ptr addrspace(4) }
define void @k(%Block %block, ptr addrspace(1) %dst) {
entry:
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %cell, align 8
  call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %cell, i1 false, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i1 false, i32 2)
  ret void
}
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<Cell, layout_explicit>", !"air.arg_name", !"block"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"value"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
"#;

const IMPLICIT_FRAGMENT: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @f() {
entry:
  call void @air.store.implicit_imageblock.v4f16(<4 x half> zeroinitializer, i32 0, <2 x i16> zeroinitializer, i32 0, i16 0)
  ret void
}
declare void @air.store.implicit_imageblock.v4f16(<4 x half>, i32, <2 x i16>, i32, i16)
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !1}
!1 = !{}
"#;

const PRIVATE_CELL_WITHOUT_TILE_ABI: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %dst) {
entry:
  %cell = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %cell, align 8
  call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %cell, i1 true, <2 x i16> zeroinitializer, <2 x i16> <i16 1, i16 1>, <2 x i32> zeroinitializer, i32 0, i1 false, i32 2)
  ret void
}
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
"#;

const CUSTOM_FRAGMENT: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
%Cell = type { half }
define { <4 x half>, %Cell } @f(%Cell %input) {
entry:
  %a = insertvalue { <4 x half>, %Cell } poison, <4 x half> zeroinitializer, 0
  %b = insertvalue { <4 x half>, %Cell } %a, %Cell %input, 1
  ret { <4 x half>, %Cell } %b
}
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !2}
!1 = !{!3, !4}
!2 = !{!5}
!3 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{!"air.imageblock_data", !"air.imageblock_data_size", i32 2, !"air.struct_type_info", !6, !"air.imageblock_master", !7}
!5 = !{i32 0, !"air.imageblock_data", !"air.imageblock_data_size", i32 2, !"air.struct_type_info", !6, !"air.imageblock_master", !7}
!6 = !{i32 0, i32 2, i32 0, !"half", !"user(value)"}
!7 = !{i32 0, i32 2, i32 0, !"half", !"user(value)", !"air.raster_order_group", i32 0}
"#;

#[test]
fn graphics_storage_other_image_write_producers_keep_existing_interface_refusals() {
    use metal2vulkan::passes::{Stage, TransformOptions};
    let scratch = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/graphics-test-artifacts/imageblock-admission-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for (source, stage, expected_stage, feature) in [
        (PRIVATE_KERNEL, Stage::Kernel, ShaderStage::Kernel, "kernel_imageblock"),
        (IMPLICIT_FRAGMENT, Stage::Fragment, ShaderStage::Fragment, "implicit_imageblock_attachments"),
        (CUSTOM_FRAGMENT, Stage::Fragment, ShaderStage::Fragment, "fragment_imageblock"),
    ] {
        let reflected = metal2vulkan::reflect_sanitized(source, stage, TransformOptions::default()).unwrap();
        // Both Vulkan entry points apply this existing gate before storage
        // staging/rounding. Private imageblock scratch still carries its tile ABI.
        let refused = spirv_bind::first_unsupported_vulkan_interface(&reflected, expected_stage).unwrap();
        assert_eq!(refused.feature, feature);
        let bytes = metal2vulkan::translate_sanitized_native(source, stage, &scratch).unwrap();
        let words: Vec<u32> = bytes.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut offset = 5;
        let mut writes = 0;
        while offset < words.len() {
            let count = (words[offset] >> 16) as usize;
            assert!(count > 0 && offset + count <= words.len());
            writes += usize::from(words[offset] & 0xffff == 99);
            offset += count;
        }
        assert!(writes > 0, "fixture must exercise an actual imageblock image write");
        assert_eq!(metal2vulkan::texture_write_rounding::specialize_texture_write_rounding(
            &words, metal2vulkan::texture_write_rounding::TextureWriteRoundingMode::Default, &[],
        ).unwrap(), words, "an unrelated producer keeps its existing default conversion");
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn graphics_storage_metadata_free_private_slice_keeps_its_admitted_conversion() {
    use metal2vulkan::passes::{Stage, TransformOptions};
    let reflected = metal2vulkan::reflect_sanitized(
        PRIVATE_CELL_WITHOUT_TILE_ABI, Stage::Kernel, TransformOptions::default(),
    ).unwrap();
    assert!(spirv_bind::first_unsupported_vulkan_interface(&reflected, ShaderStage::Kernel).is_none());
    assert!(spirv_bind::first_unsupported_vulkan_resource(&reflected).is_none());
    assert!(!crate::runtime::compute_exec::vulkan::linux_stage_input_or_imageblock_unsupported(
        false, &crate::runtime::compute_exec::ComputeAccum::default(),
    ));
    let scratch = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/graphics-test-artifacts/private-cell-admission-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let bytes = metal2vulkan::translate_sanitized_native(
        PRIVATE_CELL_WITHOUT_TILE_ABI, Stage::Kernel, &scratch,
    ).unwrap();
    std::fs::remove_dir_all(scratch).unwrap();
    let words: Vec<u32> = bytes.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(metal2vulkan::texture_write_rounding::specialize_texture_write_rounding(
        &words, crate::runtime::m2v_cache::NATIVE_TEXTURE_WRITE_ROUNDING, &[],
    ).is_ok(), "metadata-free private slices need their own preserved-conversion provenance");
}
