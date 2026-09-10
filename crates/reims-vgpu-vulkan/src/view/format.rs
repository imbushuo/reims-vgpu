//! Format compatibility for mutable, uncompressed registry images.
//! Component shape is a separate portability requirement from texel size.

use ash::vk;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    IncompatibleClass,
    ReinterpretationUnsupported,
    UnknownColorFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColorLayout {
    bytes: u32,
    bits: [u8; 4],
}

fn color_layout(format: vk::Format) -> Option<ColorLayout> {
    let bits = match format {
        vk::Format::R8_UNORM | vk::Format::R8_UINT => [8, 0, 0, 0],
        vk::Format::R8G8_UNORM | vk::Format::R8G8_UINT => [8, 8, 0, 0],
        vk::Format::R16_UNORM | vk::Format::R16_SFLOAT => [16, 0, 0, 0],
        vk::Format::R16G16_UNORM | vk::Format::R16G16_UINT | vk::Format::R16G16_SFLOAT =>
            [16, 16, 0, 0],
        vk::Format::R32_UINT | vk::Format::R32_SINT | vk::Format::R32_SFLOAT => [32, 0, 0, 0],
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB |
        vk::Format::R8G8B8A8_UINT | vk::Format::R8G8B8A8_SINT |
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => [8, 8, 8, 8],
        vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32 =>
            [10, 10, 10, 2],
        vk::Format::B10G11R11_UFLOAT_PACK32 => [11, 11, 10, 0],
        vk::Format::E5B9G9R9_UFLOAT_PACK32 => [9, 9, 9, 0],
        vk::Format::R16G16B16A16_UNORM | vk::Format::R16G16B16A16_UINT |
        vk::Format::R16G16B16A16_SFLOAT => [16, 16, 16, 16],
        vk::Format::R32G32B32A32_UINT | vk::Format::R32G32B32A32_SFLOAT => [32, 32, 32, 32],
        _ => return None,
    };
    // The shared exponent of RGB9E5 contributes storage bits, not a component.
    let bytes = crate::pixel::bytes_per_texel(format)?;
    Some(ColorLayout { bytes, bits })
}

fn depth_stencil(format: vk::Format) -> bool {
    matches!(format, vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 |
        vk::Format::D32_SFLOAT | vk::Format::S8_UINT | vk::Format::D16_UNORM_S8_UINT |
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT)
}

/// The registry creates mutable images without block-texel reinterpretation.
/// Every translated linear color format is covered; unknown classes refuse
/// rather than being inferred from byte size alone.
pub fn validate(
    allocation: vk::Format,
    requested: vk::Format,
    reinterpretation: bool,
) -> Result<(), Refusal> {
    if allocation == requested {
        return Ok(());
    }
    if depth_stencil(allocation) || depth_stencil(requested) {
        return Err(Refusal::IncompatibleClass);
    }
    let source = color_layout(allocation).ok_or(Refusal::UnknownColorFormat)?;
    let view = color_layout(requested).ok_or(Refusal::UnknownColorFormat)?;
    if source.bytes != view.bytes {
        return Err(Refusal::IncompatibleClass);
    }
    if !reinterpretation && source.bits != view.bits {
        return Err(Refusal::ReinterpretationUnsupported);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_view_classes_and_portability_are_independent_requirements() {
        for reinterpretation in [false, true] {
            assert_eq!(validate(vk::Format::R8_UNORM, vk::Format::B8G8R8A8_UNORM, reinterpretation),
                Err(Refusal::IncompatibleClass));
            assert_eq!(validate(vk::Format::R16_SFLOAT, vk::Format::R8G8B8A8_UNORM, reinterpretation),
                Err(Refusal::IncompatibleClass));
        }
        for (a, b) in [
            (vk::Format::R32_SFLOAT, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::R16_SFLOAT, vk::Format::R8G8_UNORM),
            (vk::Format::R16G16_SFLOAT, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::A2R10G10B10_UNORM_PACK32, vk::Format::R8G8B8A8_UNORM),
            (vk::Format::B10G11R11_UFLOAT_PACK32, vk::Format::E5B9G9R9_UFLOAT_PACK32),
        ] {
            assert_eq!(validate(a, b, false), Err(Refusal::ReinterpretationUnsupported));
            assert_eq!(validate(a, b, true), Ok(()));
        }
        for (a, b) in [
            (vk::Format::R8G8B8A8_UNORM, vk::Format::B8G8R8A8_UNORM),
            (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB),
            (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB),
            (vk::Format::R16_SFLOAT, vk::Format::R16_UNORM),
            (vk::Format::A2R10G10B10_UNORM_PACK32, vk::Format::A2B10G10R10_UNORM_PACK32),
        ] {
            assert_eq!(validate(a, b, false), Ok(()));
        }
    }

    #[test]
    fn image_view_component_table_covers_every_translated_linear_color_format() {
        use reims_vgpu_core::pixel_format as p;
        for guest in 0..=u16::MAX {
            let Ok(format) = crate::pixel::translate(guest) else { continue };
            if p::format_has_depth_aspect(guest) || p::format_has_stencil_aspect(guest) {
                continue;
            }
            let Some(block) = crate::pixel::vk_block_geometry(format.vk) else { continue };
            if block.width != 1 || block.height != 1 { continue; }
            let layout = color_layout(format.vk).expect("translated color format needs a view class");
            assert_eq!(layout.bytes, block.bytes, "{guest:#x}");
            assert_eq!(validate(format.linear_vk, format.vk, false), Ok(()), "{guest:#x}");
        }
    }
}
