//! Readonly samples over owned guest imports.
//!
//! A lease pins the existing import owner; retirement is checked before encode,
//! and the render batch keeps the lease until actual Metal completion. The
//! caller must redeem the mapping/page/layout proof before entering that batch.
//! Half-float sources are freshly converted to the CPU loader's RGBA8 contract
//! on that same command buffer, never retained as cross-command pixel values.

use super::{guest_writeback, raw_metal, util::Status};
use crate::protocol::pixel_format;
use crate::runtime::guest_ram::{GuestRamImport, GuestSlice};
use foreign_types::ForeignTypeRef;
use metal::*;
use std::sync::{Arc, OnceLock};

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    guest_writeback::imports_enabled()
        && *ENABLED.get_or_init(|| {
            let (setting, value) = crate::config::read(crate::config::METAL_MAPPED_SAMPLING);
            match setting {
                crate::config::Switch::Off => false,
                crate::config::Switch::On | crate::config::Switch::Unset => true,
                crate::config::Switch::Unrecognized => {
                    crate::observe::Emit::refusal(
                        "metal_mapped_sample",
                        &Status::args("metal_mapped_sample_switch_unrecognized"),
                    )
                    .unwrap()
                    .field("value", value.unwrap_or_default())
                    .fail();
                    false
                }
            }
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub format: u16,
}

impl Layout {
    pub(crate) fn pixel_format(self) -> Option<MTLPixelFormat> {
        // The existing mapped RGBA8 reader preserves encoded sRGB byte values;
        // it uploads linear RGBA8. An sRGB Metal view would change those values.
        match self.format {
            pixel_format::MTL_FORMAT_BGRA8_UNORM | pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB => {
                Some(MTLPixelFormat::BGRA8Unorm)
            }
            pixel_format::MTL_FORMAT_RGBA8_UNORM | pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB => {
                Some(MTLPixelFormat::RGBA8Unorm)
            }
            pixel_format::MTL_FORMAT_RGBA16_FLOAT => Some(MTLPixelFormat::RGBA8Unorm),
            _ => None,
        }
    }

    pub(crate) fn span(self) -> Option<u64> {
        self.pixel_format()?;
        let row = u64::from(self.width)
            .checked_mul(u64::from(pixel_format::bytes_per_pixel(self.format)?))?;
        if self.width == 0 || self.height == 0 || u64::from(self.pitch) < row {
            return None;
        }
        // Metal's linear view also requires the final row's pitch to be backed.
        u64::from(self.pitch).checked_mul(u64::from(self.height))
    }
}

#[derive(Debug)]
pub(crate) struct Image {
    source: guest_writeback::ReadSource,
    texture: Texture,
    conversion: bool,
}

impl Image {
    pub(crate) fn new(
        device: &DeviceRef,
        guest: Arc<GuestRamImport>,
        slice: GuestSlice,
        layout: Layout,
    ) -> Result<Self, Status> {
        let format = layout
            .pixel_format()
            .ok_or_else(|| Status::args("metal_mapped_sample_format"))?;
        let span = layout
            .span()
            .ok_or_else(|| Status::args("metal_mapped_sample_span"))?;
        let source = guest_writeback::ReadSource::new(
            device, guest, slice,
            guest_writeback::ReadLayout {
                width: layout.width, height: layout.height, pitch: layout.pitch, format: layout.format,
            },
        )?;
        let conversion = layout.format == pixel_format::MTL_FORMAT_RGBA16_FLOAT;
        if conversion {
            let texture = objc::rc::autoreleasepool(|| {
                let descriptor = TextureDescriptor::new();
                descriptor.set_texture_type(MTLTextureType::D2);
                descriptor.set_width(u64::from(layout.width));
                descriptor.set_height(u64::from(layout.height));
                descriptor.set_pixel_format(format);
                descriptor.set_storage_mode(MTLStorageMode::Shared);
                descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
                raw_metal::new_linear_texture_with_storage(device, &descriptor, 4)
            })
            .ok_or_else(|| Status::args("metal_mapped_sample_conversion_target"))?;
            return Ok(Self { source, texture, conversion });
        }
        let offset = source.offset();
        let alignment = device.minimum_linear_texture_alignment_for_pixel_format(format);
        if alignment == 0
            || !offset.is_multiple_of(alignment)
            || !u64::from(layout.pitch).is_multiple_of(alignment)
            || span > source.available()
        {
            return Err(Status::args("metal_mapped_sample_alignment"));
        }

        let buffer = source.buffer()?;
        let texture = objc::rc::autoreleasepool(|| {
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_width(u64::from(layout.width));
            descriptor.set_height(u64::from(layout.height));
            descriptor.set_pixel_format(format);
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead);
            raw_metal::new_linear_texture(
                buffer,
                &descriptor,
                offset,
                u64::from(layout.pitch),
            )
        })
        .ok_or_else(|| Status::args("metal_mapped_sample_linear_unavailable"))?;
        let image = Self { source, texture, conversion };
        image.texture(device)?;
        Ok(image)
    }

    pub(crate) fn live(&self) -> bool {
        self.source.check_live().is_ok()
    }

    pub(crate) fn needs_conversion(&self) -> bool {
        self.conversion
    }

    pub(crate) fn encode_conversion(
        &self,
        device: &DeviceRef,
        command: &CommandBufferRef,
    ) -> Result<(), Status> {
        if self.conversion {
            let layout = self.source.layout();
            super::guest_seed::Prepared::encode(
                device,
                command,
                self.source.buffer()?,
                self.texture(device)?,
                super::guest_seed::Layout {
                    source_offset: self.source.offset(),
                    source_length: self.source.available(),
                    source_pitch: u64::from(layout.pitch),
                    width: layout.width,
                    height: layout.height,
                    source_format: layout.format,
                },
            )?;
            crate::runtime::drain::note_store_route("metal_mapped_sample_gpu_conversions");
        }
        Ok(())
    }

    pub(crate) fn texture(&self, device: &DeviceRef) -> Result<&TextureRef, Status> {
        if !self.live() {
            return Err(Status::args("metal_mapped_sample_retired"));
        }
        if self.texture.device().as_ptr() != device.as_ptr() {
            return Err(Status::args("metal_mapped_sample_device"));
        }
        Ok(&self.texture)
    }
}

#[cfg(test)]
mod tests;
