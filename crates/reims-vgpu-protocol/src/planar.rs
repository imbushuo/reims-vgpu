//! Whole-surface, two-plane 10-bit samples. These are not packed pixel formats.
//!
//! Native IOSurface/Metal oracles distinguish `YCBCR10_420_2P` (YCbCr conversion)
//! from `RGB10_420_2P` (normalized Cr/Y/Cb channels). Both consume MSB-aligned
//! ten-bit components in sixteen-bit containers. The backing FourCC selects
//! video or full range; it must survive independently of the Metal ordinal.
//!
//! Device-plane controls were derived from Apple's host-side
//! `upgradeDescriptor` and `PGIOSurfaceHostDevice::createPlaneDictionary:`.
//! The former widens legacy +0x24..+0x27 to +0x3c..+0x3f and the four
//! +0x28 bytes to four u32s at +0x40. One-field dictionary perturbations identify
//! footprint, address format, compression type, component count, then extended
//! left/top/right/bottom pixels. The component arrays start at legacy +0x2c.

use crate::endian::{ld16, ld32};
use crate::iosurface_pages::{self as pages, DevicePlaneRecord};
use reims_vgpu_wire::ops::backed_texture;

pub mod sampling;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SampleFormat {
    Ycbcr10_420TwoPlane = 0x1f9,
    Rgb10_420TwoPlane = 0x21f,
}

impl SampleFormat {
    pub const fn parse(word: u16) -> Option<Self> {
        match word {
            0x1f9 => Some(Self::Ycbcr10_420TwoPlane),
            0x21f => Some(Self::Rgb10_420TwoPlane),
            _ => None,
        }
    }

    pub const fn word(self) -> u16 { self as u16 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BackingFormat {
    VideoRange = u32::from_be_bytes(*b"x420"),
    FullRange = u32::from_be_bytes(*b"xf20"),
}

impl BackingFormat {
    pub const fn parse(word: u32) -> Option<Self> {
        match word {
            0x7834_3230 => Some(Self::VideoRange),
            0x7866_3230 => Some(Self::FullRange),
            _ => None,
        }
    }

    pub const fn word(self) -> u32 { self as u32 }
    pub const fn component_range(self) -> u8 {
        match self { Self::VideoRange => 2, Self::FullRange => 1 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    DescriptorShort,
    TextureLayout,
    TextureReference,
    TextureFormat,
    TextureFlags,
    TextureUsage,
    TextureShape,
    TextureOptions,
    TextureProtection,
    TexturePlane,
    BackingFormat,
    SurfaceControls,
    PlaneCount,
    Extent,
    PlaneControls,
    CompressionFootprint,
    AddressFormat,
    Compression,
    ElementGeometry,
    Components,
    PlaneGeometry,
    RowPitch,
    PlaneSpan,
    PlaneOverlap,
    ImageBytes,
}

impl Refusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::DescriptorShort => "planar_descriptor_short",
            Self::TextureLayout => "planar_texture_descriptor_layout",
            Self::TextureReference => "planar_texture_reference",
            Self::TextureFormat => "planar_texture_format",
            Self::TextureFlags => "planar_texture_flags",
            Self::TextureUsage => "planar_texture_usage",
            Self::TextureShape => "planar_texture_shape",
            Self::TextureOptions => "planar_texture_resource_options",
            Self::TextureProtection => "planar_texture_protection_options",
            Self::TexturePlane => "planar_texture_plane",
            Self::BackingFormat => "planar_backing_format",
            Self::SurfaceControls => "planar_surface_controls",
            Self::PlaneCount => "planar_plane_count",
            Self::Extent => "planar_extent",
            Self::PlaneControls => "planar_plane_controls",
            Self::CompressionFootprint => "planar_compression_footprint",
            Self::AddressFormat => "planar_address_format",
            Self::Compression => "planar_compression",
            Self::ElementGeometry => "planar_element_geometry",
            Self::Components => "planar_components",
            Self::PlaneGeometry => "planar_plane_geometry",
            Self::RowPitch => "planar_row_pitch",
            Self::PlaneSpan => "planar_plane_span",
            Self::PlaneOverlap => "planar_plane_overlap",
            Self::ImageBytes => "planar_image_bytes",
        }
    }
}

impl reims_vgpu_observe::Refusal for Refusal {
    fn refusal(&self) -> Option<&'static str> { Some(self.slug()) }
}

/// Embedded argument kind in the guest's narrow type11 object descriptor.
/// This is not the host serializer's IOSurface creation opcode (`0x0c`).
pub const TYPE11_TEXTURE_ARGS_KIND: u32 = 0x2f;

/// Only the completely decoded narrow IOSurface texture record is admitted.
///
/// Apple's `newTextureDescriptor(PGSerializedTextureDescriptor const*, ...)`
/// reconstructs the tail as mip/sample/array counts, resource options and
/// protection options. It carries no color-space matrix override: the native
/// descriptor's default applies. Newer/extended records are not guessed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureDescription {
    pub mapping_id: u32,
    pub format: SampleFormat,
    pub width: u32,
    pub height: u32,
    pub allow_gpu_optimized_contents: bool,
}

impl TextureDescription {
    pub fn decode(bytes: &[u8], object_ref: u32) -> Result<Self, Refusal> {
        if bytes.len() < 0x18 { return Err(Refusal::DescriptorShort); }
        if ld32(&bytes[8..]) != TYPE11_TEXTURE_ARGS_KIND
            || bytes.len() != 8 + backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN as usize
            || ld32(&bytes[12..]) != backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN
        {
            return Err(Refusal::TextureLayout);
        }
        let op = reims_vgpu_wire::op::op(&bytes[8..], 0).map_err(|_| Refusal::TextureLayout)?;
        let body = reims_vgpu_wire::view::view::<backed_texture::IOSurfaceTextureBody>(op.payload)
            .map_err(|_| Refusal::TextureLayout)?;
        if body.object_ref.get() != object_ref { return Err(Refusal::TextureReference); }
        let d = &body.desc;
        let format = SampleFormat::parse(d.pixel_format()).ok_or(Refusal::TextureFormat)?;
        if d.unidentified_flags() != 0 { return Err(Refusal::TextureFlags); }
        if d.usage() != 1 { return Err(Refusal::TextureUsage); }
        if d.texture_type() != 2 || d.depth.get() != 1 || d.mipmap_level_count.get() != 1
            || d.sample_count.get() != 1 || d.array_length.get() != 1
        {
            return Err(Refusal::TextureShape);
        }
        if d.resource_options.get() & !0x0331 != 0 || d.cpu_cache_mode() > 1
            || d.storage_mode() > 2 || d.hazard_tracking_mode() > 2
        {
            return Err(Refusal::TextureOptions);
        }
        if d.unidentified_u64.get() != 0 { return Err(Refusal::TextureProtection); }
        if body.plane.get() != 0 { return Err(Refusal::TexturePlane); }
        Ok(Self {
            mapping_id: ld32(bytes),
            format,
            width: d.width.get(),
            height: d.height.get(),
            allow_gpu_optimized_contents: d.allow_gpu_optimized_contents(),
        })
    }
}

/// Recognize the two ordinals without accepting an undecoded descriptor version.
pub fn type11_sample_format(bytes: &[u8]) -> Option<SampleFormat> {
    if bytes.len() < 0x18 { return None; }
    let offset = match ld32(&bytes[8..]) {
        TYPE11_TEXTURE_ARGS_KIND | backed_texture::OPCODE_IOSURFACE_TEXTURE => 20 + 2,
        backed_texture::OPCODE_IOSURFACE_TEXTURE_WIDE => 20 + 1,
        _ => return None,
    };
    SampleFormat::parse(ld16(&bytes[offset..]))
}

pub const PLANE_COMPRESSION_FOOTPRINT: usize = 0x24;
pub const PLANE_ADDRESS_FORMAT: usize = 0x25;
pub const PLANE_COMPRESSION_TYPE: usize = 0x26;
pub const PLANE_COMPONENT_COUNT: usize = 0x27;
pub const PLANE_EXTENDED_PIXELS: usize = 0x28;
pub const PLANE_COMPONENT_DEPTHS: usize = 0x2c;
pub const PLANE_COMPONENT_OFFSETS: usize = 0x30;
pub const PLANE_COMPONENT_NAMES: usize = 0x34;
pub const PLANE_COMPONENT_TYPES: usize = 0x38;
pub const PLANE_COMPONENT_RANGES: usize = 0x3c;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtendedPixels {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub base: u64,
    pub offset: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u16,
    pub extended: ExtendedPixels,
}

impl Plane {
    pub fn end(self) -> u64 { self.base + self.size }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub backing_format: BackingFormat,
    pub width: u32,
    pub height: u32,
    pub allocation_size: u64,
    pub bytes_per_row: u32,
    pub planes: [Plane; 2],
}

impl Layout {
    pub fn decode(bytes: &[u8], width: u32, height: u32) -> Result<Self, Refusal> {
        let surface = pages::decode_device_surface(bytes).ok_or(Refusal::DescriptorShort)?;
        let format = BackingFormat::parse(surface.pixel_format).ok_or(Refusal::BackingFormat)?;
        if surface.plane_count != 2 { return Err(Refusal::PlaneCount); }
        if width == 0 || height == 0 || width & 1 != 0 || height & 1 != 0
            || surface.width != width || surface.height != height
        {
            return Err(Refusal::Extent);
        }
        if surface.base_offset != 0 || ld32(&bytes[12..]) != 0
            || bytes[0x25..0x30].iter().any(|&b| b != 0)
            || surface.bytes_per_element != 1
        {
            return Err(Refusal::SurfaceControls);
        }
        if bytes[pages::DEVICE_DESC_DIMS] != 0x80
            || bytes[pages::DEVICE_DESC_DIMS + 4] != 0x80
        {
            return Err(Refusal::ElementGeometry);
        }
        let plane = |index: usize| -> Result<Plane, Refusal> {
            let start = pages::DEVICE_DESC_PLANES + index * pages::DEVICE_PLANE_DESC_LEN;
            let raw = &bytes[start..start + pages::DEVICE_PLANE_DESC_LEN];
            let p = pages::decode_device_plane(raw).ok_or(Refusal::DescriptorShort)?;
            decode_plane(raw, p, index, width, height, format, surface.alloc_size as u64)
        };
        let planes = [plane(0)?, plane(1)?];
        if planes[0].base < planes[1].end() && planes[1].base < planes[0].end() {
            return Err(Refusal::PlaneOverlap);
        }
        Ok(Self {
            backing_format: format,
            width,
            height,
            allocation_size: surface.alloc_size as u64,
            bytes_per_row: surface.bytes_per_row,
            planes,
        })
    }
}

fn decode_plane(
    raw: &[u8], p: DevicePlaneRecord, index: usize, width: u32, height: u32,
    format: BackingFormat, allocation: u64,
) -> Result<Plane, Refusal> {
    if raw[..8].iter().any(|&b| b != 0) { return Err(Refusal::PlaneControls); }
    if raw[PLANE_COMPRESSION_FOOTPRINT] != 0 { return Err(Refusal::CompressionFootprint); }
    if raw[PLANE_ADDRESS_FORMAT] != 0 { return Err(Refusal::AddressFormat); }
    if raw[PLANE_COMPRESSION_TYPE] != 0 { return Err(Refusal::Compression); }
    if raw[pages::DEVICE_PLANE_DIMS] != 0x80 || raw[pages::DEVICE_PLANE_DIMS + 4] != 0x80 {
        return Err(Refusal::ElementGeometry);
    }
    let (w, h, bpe, names): (_, _, _, &[u8]) =
        if index == 0 { (width, height, 2, &[5]) } else { (width / 2, height / 2, 4, &[7, 6]) };
    if p.width != w || p.height != h || p.bytes_per_element != bpe {
        return Err(Refusal::PlaneGeometry);
    }
    if usize::from(raw[PLANE_COMPONENT_COUNT]) != names.len() { return Err(Refusal::Components); }
    for (c, &name) in names.iter().enumerate() {
        if raw[PLANE_COMPONENT_DEPTHS + c] != 10 || raw[PLANE_COMPONENT_OFFSETS + c] != 0
            || raw[PLANE_COMPONENT_NAMES + c] != name || raw[PLANE_COMPONENT_TYPES + c] != 0
            || raw[PLANE_COMPONENT_RANGES + c] != format.component_range()
        {
            return Err(Refusal::Components);
        }
    }
    let extended = ExtendedPixels {
        left: raw[PLANE_EXTENDED_PIXELS] as u32,
        top: raw[PLANE_EXTENDED_PIXELS + 1] as u32,
        right: raw[PLANE_EXTENDED_PIXELS + 2] as u32,
        bottom: raw[PLANE_EXTENDED_PIXELS + 3] as u32,
    };
    let row = (u64::from(w) + u64::from(extended.left) + u64::from(extended.right))
        * u64::from(bpe);
    if row > u64::from(p.bytes_per_row) { return Err(Refusal::RowPitch); }
    let base = u64::from(p.plane_base);
    let offset = u64::from(p.plane_offset);
    let size = u64::from(p.plane_size);
    let before = u64::from(extended.top) * u64::from(p.bytes_per_row)
        + u64::from(extended.left) * u64::from(bpe);
    let after = (u64::from(h) - 1 + u64::from(extended.bottom)) * u64::from(p.bytes_per_row)
        + (u64::from(w) + u64::from(extended.right)) * u64::from(bpe);
    if offset < base || offset - base < before || base + size > allocation
        || offset + after > base + size
    {
        return Err(Refusal::PlaneSpan);
    }
    Ok(Plane {
        base, offset, size, width: w, height: h,
        bytes_per_row: p.bytes_per_row, bytes_per_element: bpe, extended,
    })
}

#[cfg(test)]
mod tests;
