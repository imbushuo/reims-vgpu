use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::decode::resource::*;
use crate::runtime::gva_mem::{define_task_pages_arm64e, read_task_gva, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;
use metal2vulkan::reflect::*;

mod imageblock_admission;

fn fixture(levels: u16) -> (DeviceState, FakeHost, DrawEncodeRequest, Vec<u8>) {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let mut backing = vec![0xEE; 80];
    for y in 0..2 {
        for x in 0..32 { backing[y * 48 + x] = (y * 32 + x) as u8; }
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
    let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
    st64(&mut descriptor[LINEAR_DESC_SIZE..], backing.len() as u64);
    st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
    st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels);
    st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], backing.len() as u32);
    st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], 48);
    st32(&mut descriptor[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], 2);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
    st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], pixel_format::MTL_FORMAT_RGBA16_FLOAT);
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], 4);
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], 2);
    st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
    write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
    let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8));
    st64(&mut entry[4..], 0x200);
    write_task_gva_arm64e(
        &mut host, &state.tasks[1], list_object_entry_offset(7, 32).unwrap(), &entry,
    );
    (state, host, DrawEncodeRequest { task_id: 1, ..Default::default() }, backing)
}

fn reflection(stage: ShaderStage, declarations: &[(u32, bool)]) -> ShaderReflection {
    ShaderReflection {
        reflection_version: REFLECTION_VERSION, stage, entry_point: None,
        bindings: declarations.iter().map(|&(index, writable)| ResourceBinding {
            kind: if writable { ResourceKind::StorageImage } else { ResourceKind::Texture },
            metal_index: index,
            descriptor: Some(DescriptorLocation {
                set: 0, binding: if writable {
                    DEFAULT_DESCRIPTOR_LAYOUT.storage_textures.start + index
                } else { spirv_bind::TEXTURE_BINDING_BASE + index }, count: 1,
            }),
            texture_shape: Some(metal2vulkan::meta::TextureShape {
                dimension: metal2vulkan::meta::TextureDimension::D2,
                arrayed: false, multisampled: false,
                component: metal2vulkan::meta::TextureComponent::Float,
                writable, array_ref: false, array_length: None,
                storage_format: writable.then_some(metal2vulkan::meta::TextureFormat::Rgba32f),
            }),
            access: Some(if writable { ResourceAccess::Storage } else { ResourceAccess::Sampled }),
            param_index: None, stage_input_location: None, address_space: None,
            declared_size: None, extent: None, footprint: None, type_layout: None,
            type_name: None, embedded_source: None, static_sampler: None,
        }).collect(),
        argument_buffer_fields: vec![], vertex_attributes: vec![], varyings: vec![],
        render_targets: vec![], depth_members: vec![], depth_qualifier: None,
        stencil_members: vec![], local_size: None, max_work_group_size: None,
        vertex_builtins: None, tessellation: None, imageblock_layouts: vec![],
        implicit_imageblock_attachments: vec![], fragment_imageblock: None,
        datalayout: None, descriptor_layout: DescriptorLayout::default(),
        kernel_dispatch: None, runtime_sampler_specializations: vec![],
        runtime_storage_image_specializations: vec![], function_constants: vec![],
    }
}

#[test]
fn graphics_storage_capture_reflection_stages_before_binding_both_graphics_stages() {
    for kind in [ResourceKind::StorageImage, ResourceKind::Texture] {
        let (mut state, mut host, mut req, expected) = fixture(1);
        req.vertex_textures = vec![
            TextureBind { index: 0, texture_ref: 7, ..Default::default() },
            TextureBind { index: 1, texture_ref: 7, ..Default::default() },
        ].into();
        req.fragment_textures = vec![
            TextureBind { index: 0, texture_ref: 7, ..Default::default() },
        ].into();
        let mut vertex = reflection(ShaderStage::Vertex, &[(0, true), (1, false)]);
        let mut fragment = reflection(ShaderStage::Fragment, &[(0, true)]);
        vertex.bindings[0].kind = kind;
        fragment.bindings[0].kind = kind;
        let mut storage = StorageTextures::stage(
            &mut state, &mut host, &req, &vertex, &fragment,
        ).unwrap();
        assert_eq!(storage.textures.len(), 1, "{kind:?} must reach native staging");
        let vertex_binding = DEFAULT_DESCRIPTOR_LAYOUT.storage_textures.start;
        let fragment_binding = vertex_binding + spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET;
        assert!(storage.bind(7, 0, &vertex, vertex_binding, false).unwrap());
        assert!(storage.bind(7, 0, &fragment, fragment_binding, true).unwrap());
        assert!(storage.bind(7, 1, &vertex, spirv_bind::TEXTURE_BINDING_BASE + 1, false).unwrap());
        let texture = &storage.textures[&7];
        assert_eq!(texture.bindings.len(), 3);
        assert_eq!(texture.bindings[1].binding, fragment_binding);
        assert_eq!(texture.bindings[2].access, GraphicsTextureAccess::Sampled);
        assert_eq!(&texture.staged.bytes[..32], &expected[..32]);
        let completed = texture.staged.bytes.clone();
        storage.publish(&mut state, &mut host, req.task_id, vec![completed]).unwrap();
        let mut actual = vec![0; expected.len()];
        read_task_gva(&host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E,
            &mut actual, PAGE_SHIFT_ARM64E).unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn graphics_storage_native_seed_and_publication_preserve_untouched_texels_and_padding() {
    let (mut state, mut host, req, mut expected) = fixture(1);
    let mut storage = StorageTextures::default();
    storage.add(&mut state, &mut host, &req, 7, 300).unwrap();
    storage.add(&mut state, &mut host, &req, 7, 301).unwrap();
    assert_eq!(storage.textures.len(), 1);
    let mut completed = storage.textures[&7].staged.bytes.clone();
    assert_eq!(&completed[..32], &expected[..32]);
    assert_eq!(&completed[32..], &expected[48..]);
    let replacement = [0x00, 0x40, 0x00, 0xBC, 0x01, 0x38, 0x00, 0x3C];
    completed[40..48].copy_from_slice(&replacement);
    storage.publish(&mut state, &mut host, req.task_id, vec![completed]).unwrap();
    expected[56..64].copy_from_slice(&replacement);
    let mut actual = vec![0; expected.len()];
    read_task_gva(
        &host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &mut actual, PAGE_SHIFT_ARM64E,
    ).unwrap();
    assert_eq!(actual, expected);
    let next = stage_texture_raw::<VulkanStage, _>(&mut state, &mut host, 1, 7, 32, false).unwrap();
    assert_eq!(&next.bytes[40..48], &replacement, "next draw observes native values");
}

#[test]
fn graphics_storage_vertex_fragment_and_sampled_aliases_name_one_physical_texture() {
    let (mut state, mut host, req, _) = fixture(1);
    let mut storage = StorageTextures::default();
    storage.add(&mut state, &mut host, &req, 7, 300).unwrap();
    let vertex = reflection(ShaderStage::Vertex, &[(0, true)]);
    let fragment = reflection(ShaderStage::Fragment, &[(0, true), (1, false)]);
    assert!(storage.bind(7, 0, &vertex, 300, false).unwrap());
    assert!(storage.bind(7, 0, &fragment, 972, true).unwrap());
    assert!(storage.bind(7, 1, &fragment, 705, true).unwrap());
    assert_eq!(storage.textures.len(), 1);
    let bindings = &storage.textures[&7].bindings;
    assert_eq!(bindings.len(), 3);
    assert_eq!(bindings[2].access, GraphicsTextureAccess::Sampled);
    assert_eq!(bindings[0].stage, ash::vk::ShaderStageFlags::VERTEX);
    assert_eq!(bindings[1].stage, ash::vk::ShaderStageFlags::FRAGMENT);
}

#[test]
fn graphics_storage_writable_shapes_and_attachment_aliases_refuse_before_staging() {
    let (mut state, mut host, mut req, _) = fixture(2);
    let mut storage = StorageTextures::default();
    assert_eq!(storage.add(&mut state, &mut host, &req, 7, 300),
        Err(Refused::Shape { texture_ref: 7 }.into()));
    req.colors.push(ColorRtRequest { texture_ref: 7, ..Default::default() });
    assert_eq!(storage.add(&mut state, &mut host, &req, 7, 300),
        Err(Refused::AttachmentAlias { texture_ref: 7 }.into()));
    assert!(storage.is_empty());
}

#[test]
fn graphics_storage_reflected_writes_are_not_dropped_when_unbound_or_unknown() {
    let (mut state, mut host, req, _) = fixture(1);
    let vertex = reflection(ShaderStage::Vertex, &[]);
    let mut fragment = reflection(ShaderStage::Fragment, &[(0, true)]);
    assert!(matches!(StorageTextures::stage(&mut state, &mut host, &req, &vertex, &fragment),
        Err(DrawError::GraphicsStorage(Refused::MissingTexture { .. }))));
    fragment.bindings[0].access = None;
    assert!(matches!(StorageTextures::stage(&mut state, &mut host, &req, &vertex, &fragment),
        Err(DrawError::GraphicsStorage(Refused::Reflection { index: 0 }))));
}

#[test]
fn graphics_storage_array_images_and_descriptor_arrays_refuse_before_staging() {
    for descriptor_array in [false, true] {
        let (mut state, mut host, mut req, _) = fixture(1);
        req.fragment_textures = vec![
            TextureBind { index: 0, texture_ref: 7, ..Default::default() },
        ].into();
        let vertex = reflection(ShaderStage::Vertex, &[]);
        let mut fragment = reflection(ShaderStage::Fragment, &[(0, true)]);
        if descriptor_array {
            fragment.bindings[0].descriptor.as_mut().unwrap().count = 2;
        } else {
            fragment.bindings[0].texture_shape.as_mut().unwrap().arrayed = true;
        }
        assert!(matches!(StorageTextures::stage(
            &mut state, &mut host, &req, &vertex, &fragment,
        ), Err(DrawError::GraphicsStorage(Refused::Shape { texture_ref: 7 }))));
    }
}

#[test]
fn graphics_storage_incomplete_completion_cannot_publish_seed_as_output() {
    let (mut state, mut host, req, _) = fixture(1);
    let mut storage = StorageTextures::default();
    storage.add(&mut state, &mut host, &req, 7, 300).unwrap();
    assert_eq!(storage.publish(&mut state, &mut host, req.task_id, vec![]),
        Err(Refused::Output.into()));
    assert_eq!(storage.publish(&mut state, &mut host, req.task_id, vec![vec![0; 4]]),
        Err(Refused::Output.into()));
}

fn rounding_spec_values(words: &[u32]) -> BTreeMap<u32, u32> {
    let mut decorations = BTreeMap::new();
    let mut values = BTreeMap::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count > 0 && offset + count <= words.len());
        let inst = &words[offset..offset + count];
        match (inst[0] & 0xffff, inst.len()) {
            (71, 4) if inst[2] == 1 => { decorations.insert(inst[1], inst[3]); }
            (50, 4) => { values.insert(inst[2], inst[3]); }
            _ => {}
        }
        offset += count;
    }
    decorations.into_iter().filter_map(|(id, spec)| values.get(&id).map(|v| (spec, *v))).collect()
}

#[test]
fn graphics_storage_rounding_follows_final_native_formats_and_relocated_stage_bindings() {
    use crate::backend::vulkan::engine::StorageImageFormat as F;
    use metal2vulkan::texture_write_rounding::{
        NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID, TEXTURE_WRITE_FORMAT_SPEC_ID_BASE,
    };
    let scratch = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/graphics-test-artifacts/rounding-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for vertex in [true, false] {
        let (air_stage, return_type, return_value, outputs, stage) = if vertex {
            ("vertex", "<4 x float>", "ret <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>",
             "!1 = !{!4}\n!4 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}",
             ash::vk::ShaderStageFlags::VERTEX)
        } else {
            ("fragment", "void", "ret void", "!1 = !{}", ash::vk::ShaderStageFlags::FRAGMENT)
        };
        for suffix in ["", ".rte", ".rtz"] {
            let source = format!(r#"
target triple = "spirv-unknown-vulkan1.2"
define {return_type} @write_image(ptr addrspace(1) %image) {{
entry:
  call void @air.write_texture_2d{suffix}.v4f32(ptr addrspace(1) %image, <2 x i32> zeroinitializer, <4 x float> <float f0x3f803000, float f0x3f803000, float f0x3f803000, float f0x3f803000>, i32 0, i32 2)
  {return_value}
}}
declare void @air.write_texture_2d{suffix}.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
!air.{air_stage} = !{{!0}}
!0 = !{{ptr @write_image, !1, !2}}
{outputs}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"image"}}
"#);
            let bytes = metal2vulkan::translate_sanitized_native(&source,
                if vertex { metal2vulkan::passes::Stage::Vertex } else { metal2vulkan::passes::Stage::Fragment },
                &scratch,
            ).unwrap();
            let mut original: Vec<u32> = bytes.chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            let mut binding = DEFAULT_DESCRIPTOR_LAYOUT.storage_textures.start;
            if !vertex {
                assert_eq!(spirv_bind::offset_fragment_storage_bindings(&mut original), 1);
                binding += spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET;
            }
            for (format, image_format, precision) in [
                (F::Rgba16Float, spirv_bind::ImageFormat::Rgba16Float, 16),
                (F::Rgba32Float, spirv_bind::ImageFormat::Rgba32Float, 0),
                (F::Bgra8Unorm, spirv_bind::ImageFormat::Unknown, 0),
            ] {
                let resources = [GraphicsStorageTexture {
                    format, width: 1, height: 1, bytes: vec![0; format.bytes_per_texel()],
                    bindings: vec![GraphicsTextureBinding {
                        binding, access: GraphicsTextureAccess::Storage, stage,
                    }],
                }];
                let mut words = std::sync::Arc::new(original.clone());
                spirv_bind::specialize_image_formats(
                    std::sync::Arc::make_mut(&mut words), &[(binding, image_format)],
                ).unwrap();
                specialize_write_rounding(&resources, &mut words, stage).unwrap();
                let values = rounding_spec_values(&words);
                assert_eq!(values[&NATIVE_TEXTURE_WRITE_ROUNDING_SPEC_ID], 1);
                assert_eq!(values[&TEXTURE_WRITE_FORMAT_SPEC_ID_BASE], precision,
                    "stage={stage:?} source={suffix} format={format:?}");
            }
        }
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn graphics_storage_rounding_specialization_failure_is_typed() {
    let stage = ash::vk::ShaderStageFlags::FRAGMENT;
    let resources = [GraphicsStorageTexture {
        format: crate::backend::vulkan::engine::StorageImageFormat::Rgba16Float,
        width: 1, height: 1, bytes: vec![0; 8],
        bindings: vec![GraphicsTextureBinding {
            binding: 1152, access: GraphicsTextureAccess::Storage, stage,
        }],
    }];
    let mut words = std::sync::Arc::new(Vec::new());
    assert!(matches!(specialize_write_rounding(&resources, &mut words, stage),
        Err(Refused::WriteRounding { stage: failed, .. }) if failed == stage));
}
