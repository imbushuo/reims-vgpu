use super::*;

fn request() -> DrawRequest {
    DrawRequest {
        skip_readback: true,
        storage_textures: vec![GraphicsStorageTexture {
            format: StorageImageFormat::Rgba16Float,
            width: 4, height: 2, bytes: vec![0; 64],
            bindings: vec![
                GraphicsTextureBinding { binding: 300, access: GraphicsTextureAccess::Storage,
                    stage: vk::ShaderStageFlags::VERTEX },
                GraphicsTextureBinding { binding: 972, access: GraphicsTextureAccess::Storage,
                    stage: vk::ShaderStageFlags::FRAGMENT },
                GraphicsTextureBinding { binding: 33, access: GraphicsTextureAccess::Sampled,
                    stage: vk::ShaderStageFlags::FRAGMENT },
            ],
        }],
        ..Default::default()
    }
}

#[test]
fn graphics_storage_preserves_descriptor_class_and_stage_for_aliases() {
    let request = request();
    validate(&request).unwrap();
    let mut layout = Vec::new();
    layout_bindings(&request, &mut layout);
    assert_eq!(layout.len(), 3);
    assert_eq!(layout[0].ty, vk::DescriptorType::STORAGE_IMAGE.as_raw() as u32);
    assert_eq!(layout[0].stages, vk::ShaderStageFlags::VERTEX.as_raw());
    assert_eq!(layout[1].stages, vk::ShaderStageFlags::FRAGMENT.as_raw());
    assert_eq!(layout[2].ty, vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32);
    assert!(requires_completion(&request), "deferred color Store cannot defer texture publication");
    assert!(!requires_completion(&DrawRequest::default()));
}

#[test]
fn graphics_storage_rejects_truncated_native_seed_and_duplicate_descriptors() {
    let mut request = request();
    request.storage_textures[0].bytes.pop();
    assert_eq!(validate(&request), Err(GraphicsStorageDecline::Geometry.into()));
    request.storage_textures[0].bytes.push(0);
    request.storage_textures[0].bindings[2].binding = 300;
    assert_eq!(validate(&request),
        Err(GraphicsStorageDecline::DuplicateBinding { binding: 300 }.into()));
}
