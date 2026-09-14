//! Immutable packed sampled snapshots, not guest-memory freshness witnesses.
//!
//! The caller stages guest bytes on every load, then compares the complete
//! staged vector and layout. The native uploader is also the uncached render
//! uploader: this owner neither converts formats nor invents a mip/view shape.
//! Its textures are always single-level D2, Shared, ShaderRead-only. Views and
//! swizzles have already been realized by the existing staging/conversion path.

use super::abi::{ReimsVgpuPackedSampledImage, REIMS_VGPU_BINDING_TEXTURE_BASE};
use super::error::Status;
use metal::{Texture, TextureRef};

/// Every varying texture-layout field consumed by the packed uploader.
/// Binding index is not texture identity; native type, levels and access above
/// are fixed by that uploader. Keep format zero distinct from explicit RGBA8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub width: u32,
    pub height: u32,
    pub pixel_format: u32,
    pub bytes_per_row: u32,
}

impl Layout {
    pub(crate) fn image(self, bytes: &[u8], binding: u32) -> ReimsVgpuPackedSampledImage {
        ReimsVgpuPackedSampledImage {
            binding,
            width: self.width,
            height: self.height,
            rgba8: bytes.as_ptr(),
            len: bytes.len(),
            pixel_format: self.pixel_format,
            bytes_per_row: self.bytes_per_row,
            data: bytes.as_ptr(),
            data_len: bytes.len(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SampledImage {
    layout: Layout,
    bytes: Vec<u8>,
    texture: Texture,
}

impl SampledImage {
    pub(crate) fn new(layout: Layout, bytes: Vec<u8>) -> Result<Self, Status> {
        let device = super::runtime::system_device()
            .ok_or_else(|| Status::execute("metal_packed_sampled_device_unavailable"))?;
        let texture = objc::rc::autoreleasepool(|| {
            super::render::upload_packed_sampled_image(
                device,
                &layout.image(&bytes, REIMS_VGPU_BINDING_TEXTURE_BASE),
                false,
                (std::ptr::null_mut(), 0),
            )
        })?;
        Ok(Self {
            layout,
            bytes,
            texture,
        })
    }

    pub(crate) fn matches(&self, layout: Layout, bytes: &[u8]) -> bool {
        self.layout == layout && self.bytes == bytes
    }

    pub(crate) fn byte_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    pub(super) fn texture(&self) -> &TextureRef {
        &self.texture
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}
