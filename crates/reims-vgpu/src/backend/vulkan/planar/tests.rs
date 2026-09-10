use super::*;
use crate::protocol::planar::{BackingFormat, ExtendedPixels, Plane};

pub(in crate::backend::vulkan) fn image(format: SampleFormat, backing: BackingFormat) -> Image {
    let plane = |base, width, height, bytes_per_element| Plane {
        base, offset: base + 128, size: 1024, width, height,
        bytes_per_row: 64, bytes_per_element,
        extended: ExtendedPixels { left: 1, top: 1, right: 1, bottom: 1 },
    };
    let layout = Layout {
        backing_format: backing, width: 8, height: 4, allocation_size: 2048,
        bytes_per_row: 64, planes: [plane(0, 8, 4, 2), plane(1024, 4, 2, 4)],
    };
    let mut planes = [vec![0xee; 1024], vec![0xee; 1024]];
    for (index, p) in layout.planes.iter().enumerate() {
        for y in 0..p.height as usize {
            for x in 0..p.width as usize {
                for component in 0..p.bytes_per_element as usize / 2 {
                    let code = if index == 0 { 512u16 } else if component == 0 { 640 } else { 768 };
                    let offset = 128 + y * 64 + x * p.bytes_per_element as usize + component * 2;
                    planes[index][offset..offset + 2].copy_from_slice(&((code << 6) | 63).to_le_bytes());
                }
            }
        }
    }
    Image::expand(TextureDescription {
        mapping_id: 5, format, width: 8, height: 4, allow_gpu_optimized_contents: true,
    }, layout, planes).unwrap()
}

#[test]
fn four_native_oracles_keep_precision_range_channel_order_and_plane_origins() {
    for backing in [BackingFormat::VideoRange, BackingFormat::FullRange] {
        let raw = image(SampleFormat::Rgb10_420TwoPlane, backing);
        assert_eq!(raw.engine_format(), StorageImageFormat::Rgb10a2Unorm);
        for texel in raw.bytes.chunks_exact(4) {
            assert_eq!(u32::from_le_bytes(texel.try_into().unwrap()),
                768 | (512 << 10) | (640 << 20) | (3 << 30));
        }
        let rgb = image(SampleFormat::Ycbcr10_420TwoPlane, backing);
        assert_eq!(rgb.engine_format(), StorageImageFormat::Rgba16Float);
        let expected = if backing == BackingFormat::VideoRange {
            [1867u16, 529, 1566, 2048]
        } else { [1742, 571, 1479, 2048] };
        for texel in rgb.bytes.chunks_exact(8) {
            for (word, q) in texel.chunks_exact(2).zip(expected) {
                assert_eq!(crate::protocol::pixel_format::f16_to_f32(
                    u16::from_le_bytes(word.try_into().unwrap()),
                ), f32::from(q) / 2048.0);
            }
        }
    }
}
