//! Composite P010 sampling arithmetic. Chroma is centered, reconstructed at
//! luma centers and rounded to ten bits before the format's color transform.
//! The RGB transform uses quantized BT.601 coefficients, not the ideal matrix.
//! Its output is clamped Q11; full-range luma first normalizes to Q15.

use super::BackingFormat;

/// Clamped chroma neighbors and their weights in quarters.
pub fn chroma_axis(pixel: u32, chroma_extent: u32) -> [(u32, u32); 2] {
    let lower = (i64::from(pixel) * 2 - 1).div_euclid(4);
    let upper_weight = if pixel & 1 == 0 { 3 } else { 1 };
    let clamp = |v: i64| v.clamp(0, i64::from(chroma_extent) - 1) as u32;
    [(clamp(lower), 4 - upper_weight), (clamp(lower + 1), upper_weight)]
}

pub fn chroma_code(codes: [u16; 4], x_weights: [u32; 2], y_weights: [u32; 2]) -> u16 {
    let mut sum = 0;
    for y in 0..2 {
        for x in 0..2 {
            sum += u32::from(codes[y * 2 + x]) * x_weights[x] * y_weights[y];
        }
    }
    ((sum + 8) / 16) as u16
}

pub fn rgb_q11(backing: BackingFormat, y: u16, cb: u16, cr: u16) -> [u16; 4] {
    let y = i32::from(y);
    let cb = i32::from(cb) - 512;
    let cr = i32::from(cr) - 512;
    let (luma, coefficients) = match backing {
        BackingFormat::VideoRange => (
            9576 * (y - 64),
            [[-9, 13123], [-3218, -6686], [16591, 9]],
        ),
        BackingFormat::FullRange => (
            ((y * 1025 + 16) / 32) * 256,
            [[-8, 11483], [-2816, -5850], [14518, 8]],
        ),
    };
    let mut out = [0, 0, 0, 2048];
    for (channel, [u, v]) in coefficients.into_iter().enumerate() {
        out[channel] = ((luma + u * cb + v * cr + 2048).div_euclid(4096))
            .clamp(0, 2048) as u16;
    }
    out
}

/// Q11 values in [0,1] are exactly representable as binary16.
pub fn q11_half(value: u16) -> u16 {
    if value == 0 { return 0; }
    if value == 2048 { return 0x3c00; }
    let exponent = 15 - value.leading_zeros();
    ((exponent + 4) << 10) as u16 | ((value - (1 << exponent)) << (10 - exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_chroma_clamps_edges_and_rounds_half_up() {
        assert_eq!(chroma_axis(0, 4), [(0, 1), (0, 3)]);
        assert_eq!(chroma_axis(1, 4), [(0, 3), (1, 1)]);
        assert_eq!(chroma_axis(2, 4), [(0, 1), (1, 3)]);
        assert_eq!(chroma_axis(7, 4), [(3, 3), (3, 1)]);
        assert_eq!(chroma_code([0, 2, 0, 2], [3, 1], [3, 1]), 1);
        assert_eq!(chroma_code([512, 512, 512, 0], [1, 3], [1, 3]), 224);
    }

    #[test]
    fn quantized_matrix_matches_native_format_range_oracles() {
        assert_eq!(rgb_q11(BackingFormat::VideoRange, 512, 640, 768), [1867, 529, 1566, 2048]);
        assert_eq!(rgb_q11(BackingFormat::FullRange, 512, 640, 768), [1742, 571, 1479, 2048]);
        assert_eq!(rgb_q11(BackingFormat::VideoRange, 512, 0, 512), [1049, 1450, 0, 2048]);
        assert_eq!(rgb_q11(BackingFormat::VideoRange, 512, 512, 768), [1868, 630, 1048, 2048]);
    }

    #[test]
    fn full_range_luma_normalizes_before_conversion_and_video_range_clamps() {
        assert_eq!(rgb_q11(BackingFormat::FullRange, 240, 512, 512), [481, 481, 481, 2048]);
        assert_eq!(rgb_q11(BackingFormat::FullRange, 752, 512, 512), [1506, 1506, 1506, 2048]);
        assert_eq!(rgb_q11(BackingFormat::VideoRange, 0, 512, 512), [0, 0, 0, 2048]);
        assert_eq!(rgb_q11(BackingFormat::VideoRange, 1023, 512, 512), [2048; 4]);
    }

    #[test]
    fn binary16_upload_represents_every_q11_code_exactly() {
        for code in 0..=2048 {
            assert_eq!(crate::pixel_format::f16_to_f32(q11_half(code)), f32::from(code) / 2048.0);
        }
    }
}
