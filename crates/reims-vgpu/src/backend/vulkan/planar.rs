//! Vulkan composite samples: native half/packed images, never Metal textures.
//! Plane reads are staged by the shared guest-memory owner before expansion.

use crate::protocol::planar::{sampling, Layout, SampleFormat, TextureDescription};
use crate::protocol::pixel_format::{SampledByteFormat, TexelLayout};
use super::engine::StorageImageFormat;

#[cfg(test)]
pub(crate) mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Extent,
    PlaneBytes,
    PlaneOffsetAlignment,
    HostLength,
    Shader(&'static str),
}

impl Refusal {
    pub(crate) fn slug(self) -> &'static str {
        match self {
            Self::Extent => "planar_extent",
            Self::PlaneBytes => "planar_image_bytes",
            Self::PlaneOffsetAlignment => "vulkan_planar_plane_offset_alignment",
            Self::HostLength => "planar_host_length",
            Self::Shader(reason) => reason,
        }
    }
}

impl crate::observe::Refusal for Refusal {
    fn refusal(&self) -> Option<&'static str> { Some(self.slug()) }
}

#[derive(Debug)]
pub(crate) struct Image {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: SampleFormat,
    pub(crate) bytes: Vec<u8>,
}

impl Image {
    pub(crate) fn engine_format(&self) -> StorageImageFormat {
        match self.format {
            SampleFormat::Ycbcr10_420TwoPlane => StorageImageFormat::Rgba16Float,
            SampleFormat::Rgb10_420TwoPlane => StorageImageFormat::Rgb10a2Unorm,
        }

    }

    pub(crate) fn byte_format(&self) -> SampledByteFormat {
        SampledByteFormat::from_source(match self.format {
            SampleFormat::Ycbcr10_420TwoPlane => TexelLayout::Rgba16Float,
            SampleFormat::Rgb10_420TwoPlane => TexelLayout::Rgb10a2Unorm,
        }, self.format.word())
    }

    pub(crate) fn expand(
        description: TextureDescription, layout: Layout, planes: [Vec<u8>; 2],
    ) -> Result<Self, Refusal> {
        if description.width != layout.width || description.height != layout.height {
            return Err(Refusal::Extent);
        }
        for (plane, bytes) in layout.planes.iter().zip(&planes) {
            if bytes.len() as u64 != plane.size { return Err(Refusal::PlaneBytes); }
            // Native IOSurface texture bases are cache-line aligned. A probe
            // with an unaligned plane offset did not address the logical origin;
            // its low-offset contract is not inferred from that one observation.
            if plane.offset % 64 != 0 { return Err(Refusal::PlaneOffsetAlignment); }
        }
        let bpp = match description.format {
            SampleFormat::Ycbcr10_420TwoPlane => 8u64,
            SampleFormat::Rgb10_420TwoPlane => 4,
        };
        let len = u64::from(layout.width).checked_mul(u64::from(layout.height))
            .and_then(|pixels| pixels.checked_mul(bpp))
            .and_then(crate::runtime::draw::host_alloc_len).ok_or(Refusal::HostLength)?;
        let mut bytes = Vec::with_capacity(len);
        let code = |plane: usize, x: u32, y: u32, component: usize| -> u16 {
            let p = &layout.planes[plane];
            let offset = (p.offset - p.base + u64::from(y) * u64::from(p.bytes_per_row)
                + u64::from(x) * u64::from(p.bytes_per_element)) as usize + component * 2;
            u16::from_le_bytes([planes[plane][offset], planes[plane][offset + 1]]) >> 6
        };
        for y in 0..layout.height {
            let ys = sampling::chroma_axis(y, layout.planes[1].height);
            for x in 0..layout.width {
                let xs = sampling::chroma_axis(x, layout.planes[1].width);
                let chroma = |component| sampling::chroma_code(
                    [
                        code(1, xs[0].0, ys[0].0, component),
                        code(1, xs[1].0, ys[0].0, component),
                        code(1, xs[0].0, ys[1].0, component),
                        code(1, xs[1].0, ys[1].0, component),
                    ], [xs[0].1, xs[1].1], [ys[0].1, ys[1].1],
                );
                let (luma, cb, cr) = (code(0, x, y, 0), chroma(0), chroma(1));
                match description.format {
                    SampleFormat::Ycbcr10_420TwoPlane => {
                        for q in sampling::rgb_q11(layout.backing_format, luma, cb, cr) {
                            bytes.extend_from_slice(&sampling::q11_half(q).to_le_bytes());
                        }
                    }
                    SampleFormat::Rgb10_420TwoPlane => {
                        let packed = u32::from(cr) | (u32::from(luma) << 10)
                            | (u32::from(cb) << 20) | (3 << 30);
                        bytes.extend_from_slice(&packed.to_le_bytes());
                    }
                }
            }
        }
        Ok(Self { width: layout.width, height: layout.height, format: description.format, bytes })
    }
}
