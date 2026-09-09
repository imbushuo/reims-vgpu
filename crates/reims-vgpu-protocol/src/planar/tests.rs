use super::*;
use crate::endian::{st16, st32, st64};

fn descriptor() -> [u8; pages::DEVICE_DESC_LEN] {
    let mut b = [0; pages::DEVICE_DESC_LEN];
    st32(&mut b[4..], BackingFormat::VideoRange.word());
    st32(&mut b[16..], 4096);
    st64(&mut b[20..], 0x80 | (8 << 8) | (0x80 << 32) | (4 << 40));
    st32(&mut b[28..], 64);
    st16(&mut b[32..], 1);
    b[36] = 2;
    for (i, (w, h, bpe, base, names)) in [
        (8, 4, 2, 0, &[5][..]), (4, 2, 4, 2048, &[7, 6][..]),
    ].into_iter().enumerate() {
        let p = &mut b[64 + i * 64..128 + i * 64];
        st32(&mut p[8..], base + 64 + 16);
        st32(&mut p[12..], base);
        st32(&mut p[16..], 1024);
        st64(&mut p[20..], 0x80 | (w << 8) | (0x80 << 32) | (h << 40));
        st32(&mut p[28..], 64);
        st16(&mut p[32..], bpe);
        p[PLANE_COMPONENT_COUNT] = names.len() as u8;
        p[PLANE_EXTENDED_PIXELS..PLANE_EXTENDED_PIXELS + 4].fill(1);
        for (c, &name) in names.iter().enumerate() {
            p[PLANE_COMPONENT_DEPTHS + c] = 10;
            p[PLANE_COMPONENT_NAMES + c] = name;
            p[PLANE_COMPONENT_RANGES + c] = 2;
        }
    }
    b
}

#[test]
fn planar_layout_preserves_padding_offsets_and_extended_pixels() {
    let b = descriptor();
    let d = Layout::decode(&b, 8, 4).unwrap();
    assert_eq!(d.backing_format, BackingFormat::VideoRange);
    assert_eq!(d.planes[0].base, 0);
    assert_eq!(d.planes[0].offset, 80);
    assert_eq!(d.planes[1].offset, 2128);
    assert_eq!(d.planes[1].bytes_per_row, 64);
    assert_eq!(d.planes[1].extended, ExtendedPixels { left: 1, top: 1, right: 1, bottom: 1 });
}

#[test]
fn planar_control_refusals_do_not_confuse_extensions_with_compression() {
    for (offset, reason) in [
        (PLANE_COMPRESSION_FOOTPRINT, Refusal::CompressionFootprint),
        (PLANE_ADDRESS_FORMAT, Refusal::AddressFormat),
        (PLANE_COMPRESSION_TYPE, Refusal::Compression),
        (PLANE_COMPONENT_DEPTHS, Refusal::Components),
    ] {
        let mut b = descriptor();
        b[64 + offset] = 1;
        assert_eq!(Layout::decode(&b, 8, 4), Err(reason));
    }
}

#[test]
fn planar_layout_refuses_truncated_overlapping_and_out_of_span_planes() {
    let b = descriptor();
    assert_eq!(Layout::decode(&b[..511], 8, 4), Err(Refusal::DescriptorShort));
    let mut b = descriptor();
    st32(&mut b[128 + 12..], 0);
    st32(&mut b[128 + 8..], 80);
    assert_eq!(Layout::decode(&b, 8, 4), Err(Refusal::PlaneOverlap));
    let mut b = descriptor();
    st32(&mut b[64 + 16..], 100);
    assert_eq!(Layout::decode(&b, 8, 4), Err(Refusal::PlaneSpan));
    assert_eq!(Layout::decode(&descriptor(), 7, 4), Err(Refusal::Extent));
}

fn texture_descriptor(format: SampleFormat) -> [u8; 56] {
    let mut b = [0; 56];
    st32(&mut b, 5);
    st32(&mut b[8..], 0x2f);
    st32(&mut b[12..], backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN);
    st32(&mut b[16..], 11);
    st32(&mut b[20..], (u32::from(format.word()) << 16) | 0x0142);
    st32(&mut b[24..], 8);
    st32(&mut b[28..], 4);
    st32(&mut b[32..], 1);
    st16(&mut b[36..], 1);
    st16(&mut b[38..], 1);
    st16(&mut b[40..], 1);
    st16(&mut b[42..], 0x10);
    b
}

#[test]
fn planar_type11_checks_the_entire_narrow_texture_contract() {
    for format in [SampleFormat::Ycbcr10_420TwoPlane, SampleFormat::Rgb10_420TwoPlane] {
        let b = texture_descriptor(format);
        assert_eq!(type11_sample_format(&b), Some(format));
        let t = TextureDescription::decode(&b, 11).unwrap();
        assert_eq!(t.format, format);
        assert_eq!((t.mapping_id, t.width, t.height), (5, 8, 4));
        for (offset, reason) in [
            (36, Refusal::TextureShape), (38, Refusal::TextureShape),
            (40, Refusal::TextureShape), (44, Refusal::TextureProtection),
            (52, Refusal::TexturePlane),
        ] {
            let mut changed = b;
            changed[offset] = 2;
            assert_eq!(TextureDescription::decode(&changed, 11), Err(reason));
        }
        let mut padding = b;
        padding[54..].fill(0xaa);
        assert_eq!(TextureDescription::decode(&padding, 11), Ok(t));
        let mut command_stream_opcode = b;
        st32(&mut command_stream_opcode[8..], backed_texture::OPCODE_IOSURFACE_TEXTURE);
        assert_eq!(type11_sample_format(&command_stream_opcode), Some(format));
        assert_eq!(
            TextureDescription::decode(&command_stream_opcode, 11),
            Err(Refusal::TextureLayout),
        );
    }
}
