//! Type 9 is a memoryless texture, not an `APVObjectTexture` allocation.
//!
//! The guest's `AppleParavirtDevice::createMemorylessTexture` validates a
//! serializer command and passes tag 9, its length and its bytes directly to
//! `createObjectInternal`. Unlike `allocateTextureHandle`, it adds neither an
//! allocation size nor a page handle. The descriptor is a `newTexture` command.
//! Perturbing `storageMode` alone through `PGSerializer` changes its options
//! nibble from 0/1/2 to 3; the live type-9 records carry that same value.
//!
//! Memoryless contents belong to one render pass. They must not acquire a guest
//! address, be read back, or survive a pass through an ordinary surface cache.

use reims_vgpu_wire::ops::texture;

pub const OBJECT_TYPE_MEMORYLESS_TEXTURE: u8 = 9;

/// Whether a colour target has contents outside its render pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorStorage {
    #[default]
    GuestBacked,
    Memoryless,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorylessTexture {
    pub object_ref: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_format: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorylessRefusal {
    Framing,
    Opcode,
    Storage,
    Shape,
    Flags,
    Usage,
    Unidentified,
}

impl MemorylessRefusal {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Framing => "memoryless_descriptor_framing",
            Self::Opcode => "memoryless_descriptor_opcode",
            Self::Storage => "memoryless_descriptor_storage",
            Self::Shape => "memoryless_descriptor_shape",
            Self::Flags => "memoryless_descriptor_flags",
            Self::Usage => "memoryless_descriptor_usage",
            Self::Unidentified => "memoryless_descriptor_unidentified",
        }
    }
}

/// Decode the single-sample 2D memoryless attachment shape this executor can
/// represent. Other shapes are refusals, not ordinary linear textures.
pub fn decode(bytes: &[u8]) -> Result<MemorylessTexture, MemorylessRefusal> {
    use MemorylessRefusal as R;
    let op = reims_vgpu_wire::op(bytes, 0).map_err(|_| R::Framing)?;
    if op.opcode() != texture::OPCODE_NEW_TEXTURE {
        return Err(R::Opcode);
    }
    if bytes.len() != texture::NEW_TEXTURE_TOTAL_LEN as usize
        || op.length() != texture::NEW_TEXTURE_TOTAL_LEN
    {
        return Err(R::Framing);
    }
    let body = texture::new_texture(&op).map_err(|_| R::Framing)?;
    let d = &body.desc;
    if d.storage_mode() != 3 {
        return Err(R::Storage);
    }
    if d.texture_type() != 2 || d.width.get() == 0 || d.height.get() == 0
        || d.depth.get() != 1 || d.mipmap_level_count.get() != 1
        || d.sample_count.get() != 1 || d.array_length.get() != 1
    {
        return Err(R::Shape);
    }
    if d.unidentified_flags() != 0 || d.resource_options_raw() & !0x0330 != 0
        || d.hazard_tracking_mode() == 3
    {
        return Err(R::Flags);
    }
    if d.usage() & 4 == 0 || d.usage() & !5 != 0 {
        return Err(R::Usage);
    }
    if d.unidentified_u64.get() != 0 {
        return Err(R::Unidentified);
    }
    Ok(MemorylessTexture {
        object_ref: body.object_ref.get(),
        width: d.width.get(),
        height: d.height.get(),
        pixel_format: d.pixel_format(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Vec<u8> {
        [1u32, 44, 37, (115 << 16) | (5 << 8) | 0x42, 23, 17, 1,
            0x0001_0001, 0x0030_0001, 0, 0]
            .into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn memoryless_creation_has_geometry_but_no_guest_backing() {
        let mut bytes = descriptor();
        assert_eq!(decode(&bytes), Ok(MemorylessTexture {
            object_ref: 37, width: 23, height: 17, pixel_format: 115,
        }));
        bytes[12] |= 0x80;
        assert!(decode(&bytes).is_ok(), "unwritten ring bit is not a flag");
    }

    #[test]
    fn memoryless_decode_refuses_other_backing_and_unimplemented_shapes() {
        for (offset, value, reason) in [
            (34, 0x20, MemorylessRefusal::Storage),
            (30, 4, MemorylessRefusal::Shape),
            (12, 4, MemorylessRefusal::Shape),
            (28, 2, MemorylessRefusal::Shape),
            (13, 7, MemorylessRefusal::Usage),
            (36, 1, MemorylessRefusal::Unidentified),
            (0, 0x34, MemorylessRefusal::Opcode),
        ] {
            let mut bytes = descriptor();
            bytes[offset] = value;
            assert_eq!(decode(&bytes), Err(reason));
        }
        let bytes = descriptor();
        for n in 0..bytes.len() {
            assert!(decode(&bytes[..n]).is_err());
        }
        let mut extra = bytes;
        extra.push(0);
        assert_eq!(decode(&extra), Err(MemorylessRefusal::Framing));
    }
}
