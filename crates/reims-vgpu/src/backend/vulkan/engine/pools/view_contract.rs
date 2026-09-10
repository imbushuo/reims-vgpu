use ash::vk;
use super::super::{reason::DrawReason, registry_subresource_range};
use reims_vgpu_vulkan::view::{SubresourceRange, ViewPlan};

pub(super) fn sampled_components(
    format: vk::Format,
    swizzle: &crate::protocol::pixel_format::SwizzlePlan,
    enabled: bool,
) -> Result<vk::ComponentMapping, super::super::DrawError> {
    if !enabled && !swizzle.is_identity() {
        return Err(super::super::DrawError::Unsupported(
            DrawReason::ImageViewSwizzleUnsupported { format },
        ));
    }
    Ok(reims_vgpu_vulkan::pixel::vk_component_mapping(swizzle))
}

/// Registry residents are single-level, single-layer images. Mutable color
/// views may change interpretation within an uncompressed Vulkan size class;
/// depth/stencil residents require their original format.
pub(super) fn registry_view_plan(
    allocation: vk::Format,
    requested: vk::Format,
    reinterpretation: bool,
) -> Result<ViewPlan, DrawReason> {
    use reims_vgpu_vulkan::view::format::{validate, Refusal};
    validate(allocation, requested, reinterpretation).map_err(|refusal| match refusal {
        Refusal::IncompatibleClass =>
            DrawReason::ResidentViewFormatIncompatible { allocation, requested },
        Refusal::ReinterpretationUnsupported =>
            DrawReason::ResidentViewFormatReinterpretationUnsupported { allocation, requested },
        Refusal::UnknownColorFormat =>
            DrawReason::ResidentViewFormatUnknown { allocation, requested },
    })?;
    let range = registry_subresource_range(requested);
    Ok(ViewPlan {
        view_type: vk::ImageViewType::TYPE_2D,
        format: requested,
        range: SubresourceRange {
            aspect_mask: range.aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_view_rejects_incompatible_texel_widths_before_native_creation() {
        for (allocation, requested) in [
            (vk::Format::R8_UNORM, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R16_SFLOAT, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8G8B8A8_UNORM),
        ] {
            let refusal = registry_view_plan(allocation, requested, false).unwrap_err();
            assert_eq!(refusal, DrawReason::ResidentViewFormatIncompatible { allocation, requested });
            let message = refusal.to_string();
            assert!(message.contains(&format!("allocation={allocation:?}")));
            assert!(message.contains(&format!("requested={requested:?}")));
        }
    }

    #[test]
    fn resident_view_preserves_native_formats_and_compatible_interpretations() {
        for format in [
            vk::Format::R8_UNORM, vk::Format::R16_SFLOAT, vk::Format::R16G16_SFLOAT,
            vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM,
            vk::Format::R16G16B16A16_SFLOAT, vk::Format::R32G32B32A32_SFLOAT,
        ] {
            let plan = registry_view_plan(format, format, false).unwrap();
            assert_eq!(plan.format, format);
            assert_eq!(plan.view_type, vk::ImageViewType::TYPE_2D);
            assert_eq!(plan.range, SubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1,
            });
        }
        for (allocation, requested) in [
            (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
            (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
            (vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R16_SFLOAT, vk::Format::R16_UNORM),
        ] {
            assert_eq!(registry_view_plan(allocation, requested, false).unwrap().format, requested);
        }
    }

    #[test]
    fn resident_view_does_not_confuse_depth_or_unknown_formats_with_color_classes() {
        assert!(registry_view_plan(vk::Format::D32_SFLOAT, vk::Format::R32_SFLOAT, false).is_err());
        assert!(registry_view_plan(vk::Format::S8_UINT, vk::Format::R8_UNORM, false).is_err());
        assert!(registry_view_plan(vk::Format::R8_UNORM, vk::Format::UNDEFINED, false).is_err());
        assert_eq!(
            registry_view_plan(vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT, false)
                .unwrap().range.aspect_mask,
            vk::ImageAspectFlags::DEPTH,
        );
    }

    #[test]
    fn resident_view_portability_refusal_is_typed_and_uses_the_enabled_capability() {
        let allocation = vk::Format::R32_SFLOAT;
        let requested = vk::Format::R8G8B8A8_UNORM;
        assert_eq!(registry_view_plan(allocation, requested, false),
            Err(DrawReason::ResidentViewFormatReinterpretationUnsupported { allocation, requested }));
        assert!(registry_view_plan(allocation, requested, true).is_ok());
    }

    #[test]
    fn image_view_alpha_swizzle_requires_its_enabled_portability_capability() {
        use crate::protocol::pixel_format as p;
        assert!(sampled_components(vk::Format::R8_UNORM, &p::swizzle_identity(), false).is_ok());
        let alpha = reims_vgpu_vulkan::pixel::translate(p::MTL_FORMAT_A8_UNORM).unwrap();
        assert!(matches!(sampled_components(alpha.vk, &alpha.components, false),
            Err(super::super::super::DrawError::Unsupported(
                DrawReason::ImageViewSwizzleUnsupported { format: vk::Format::R8_UNORM },
            ))));
        let native = sampled_components(alpha.vk, &alpha.components, true).unwrap();
        assert_eq!(native.r, vk::ComponentSwizzle::ZERO);
        assert_eq!(native.g, vk::ComponentSwizzle::ZERO);
        assert_eq!(native.b, vk::ComponentSwizzle::ZERO);
        assert_eq!(native.a, vk::ComponentSwizzle::R);
    }
}
