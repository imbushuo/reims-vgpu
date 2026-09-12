//! Native packed or whole-surface planar uploads for Metal sampled bindings.

use super::*;
use crate::backend::metal::abi::{
    ReimsVgpuPackedSampledImage, ReimsVgpuSampledImage, REIMS_VGPU_BINDING_TEXTURE_BASE,
};
use crate::backend::metal::planar::SampledImage;
use crate::runtime::compute_exec::{
    metal::{try_stage_planar_sampled, MetalStage}, stage_texture_raw,
};
use std::sync::Arc;

pub(super) enum SampledUpload {
    Packed {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        pixel_format: u32,
        bytes_per_row: u32,
    },
    Planar(Arc<SampledImage>),
}

impl SampledUpload {
    pub fn byte_len(&self) -> u64 {
        match self {
            Self::Packed { bytes, .. } => bytes.len() as u64,
            Self::Planar(image) => image.layout().planes.iter().map(|plane| plane.size).sum(),
        }
    }

    pub fn image(&self, index: u32) -> ReimsVgpuSampledImage {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + index;
        match self {
            Self::Planar(image) => ReimsVgpuSampledImage::Planar {
                binding,
                image: image.clone(),
            },
            Self::Packed { bytes, width, height, pixel_format, bytes_per_row } => {
                ReimsVgpuSampledImage::Packed(ReimsVgpuPackedSampledImage {
                    binding,
                    width: *width,
                    height: *height,
                    rgba8: bytes.as_ptr(),
                    len: bytes.len(),
                    pixel_format: *pixel_format,
                    bytes_per_row: *bytes_per_row,
                    data: bytes.as_ptr(),
                    data_len: bytes.len(),
                })
            }
        }
    }
}

pub(super) fn load<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<SampledUpload> {
    match try_stage_planar_sampled(state, host, task_id, texture_ref) {
        Ok(Some(image)) => return Some(SampledUpload::Planar(image)),
        Ok(None) => {}
        Err(reason) => {
            crate::observe::Emit::refusal("draw_mtl_planar_texture", &reason)
                .expect("a staging error is a refusal")
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail();
            return None;
        }
    }
    let native_packed = objects::lookup_list_entry(state, host, task_id, texture_ref)
        .is_some_and(|entry| match entry.object_type {
            OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS => {
                objects::read_descriptor(state, host, task_id, &entry)
                    .and_then(|bytes| decode_texture_descriptor(&bytes).ok())
                    .is_some_and(|texture| {
                        texture.mipmap_level_count == 1
                            && texture.depth == 1
                            && texture.sample_count == Some(1)
                    })
            }
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW => {
                buffer_texture_descriptor(state, host, task_id, texture_ref, None).is_some()
            }
            _ => false,
        });
    if native_packed {
        // Reuse the raw staging owner: it pays both texture/buffer writeback
        // debts and removes row padding without quantizing the guest format.
        // Multi-level and view textures retain their existing loading path.
        let staged = match stage_texture_raw::<MetalStage, _>(
            state, host, task_id, texture_ref, 0, false,
        ) {
            Ok(staged) => staged,
            Err(reason) => {
                crate::observe::Emit::refusal("draw_mtl_packed_texture", &reason)
                    .expect("a staging error is a refusal")
                    .field("task", task_id)
                    .field("ref", texture_ref)
                    .fail();
                return None;
            }
        };
        let Some(bytes_per_row) =
            pixel_format::tight_row_bytes(staged.width, staged.pixel_format)
        else {
            crate::observe::Emit::refusal(
                "draw_mtl_packed_texture",
                &EncodeStatus::BadArgs("draw_mtl_packed_texture_pitch"),
            )
            .expect("BadArgs is a refusal")
            .field("ref", texture_ref)
            .fail();
            return None;
        };
        return Some(SampledUpload::Packed {
            bytes: staged.bytes,
            width: staged.width,
            height: staged.height,
            pixel_format: u32::from(staged.pixel_format),
            bytes_per_row,
        });
    }

    let (width, height, bytes) = load_sampled_rgba(state, host, task_id, texture_ref)?;
    let Some(bytes_per_row) = width.checked_mul(RGBA8_BPP) else {
        crate::observe::Emit::refusal(
            "draw_mtl_sampled_texture",
            &EncodeStatus::BadArgs("draw_mtl_sampled_texture_pitch"),
        )
        .expect("BadArgs is a refusal")
        .field("ref", texture_ref)
        .fail();
        return None;
    };
    Some(SampledUpload::Packed {
        bytes,
        width,
        height,
        pixel_format: 0,
        bytes_per_row,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::protocol::endian::{st16, st32, st64};
    use crate::runtime::decode::resource::*;
    use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
    use crate::runtime::host::FakeHost;
    use reims_vgpu_wire::ops::texture::WideTextureDescriptorBody as W;
    use std::mem::offset_of;

    #[test]
    fn planar_sampled_binding_keeps_whole_surface_owned() {
        use crate::backend::metal::planar::tests::image;
        use crate::protocol::planar::{BackingFormat, SampleFormat};

        for format in [SampleFormat::Ycbcr10_420TwoPlane, SampleFormat::Rgb10_420TwoPlane] {
            let image = Arc::new(image(format, BackingFormat::VideoRange));
            let lifetime = Arc::downgrade(&image);
            let upload = SampledUpload::Planar(image);
            assert_eq!(upload.byte_len(), 2048);
            let binding = upload.image(5);
            drop(upload);
            let ReimsVgpuSampledImage::Planar { binding: slot, image } = &binding else {
                panic!("whole-surface samples must not become packed RGBA8");
            };
            assert_eq!(*slot, REIMS_VGPU_BINDING_TEXTURE_BASE + 5);
            assert_eq!(image.description().format, format);
            assert_eq!(image.layout().planes[0].offset, 128);
            assert_eq!(image.layout().planes[1].offset, 2176);
            assert!(lifetime.upgrade().is_some());
            drop(binding);
            assert!(lifetime.upgrade().is_none());
        }
    }

    #[test]
    fn linear_sampled_upload_preserves_float_precision_channels_and_padded_rows() {
        for (format, bpp) in [
            (pixel_format::MTL_FORMAT_R16_FLOAT, 2usize),
            (pixel_format::MTL_FORMAT_RG8_UNORM, 2),
            (pixel_format::MTL_FORMAT_RGBA16_FLOAT, 8),
            (pixel_format::MTL_FORMAT_RGBA32_FLOAT, 16),
        ] {
            let mut host = FakeHost::new();
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));
            let width = 4;
            let height = 2;
            let tight = width * bpp;
            let pitch = tight + 16;
            let mut backing = vec![0xEE; pitch + tight];
            let expected: Vec<u8> = (0..tight * height).map(|v| v as u8).collect();
            for y in 0..height {
                backing[y * pitch..y * pitch + tight]
                    .copy_from_slice(&expected[y * tight..(y + 1) * tight]);
            }
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
            let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
            st64(&mut descriptor[LINEAR_DESC_SIZE..], backing.len() as u64);
            st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
            st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
            st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], backing.len() as u32);
            st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], pitch as u32);
            st32(&mut descriptor[TEXTURE_DESC_WIDTH..], width as u32);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], height as u32);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
            st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], format);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], width as u32);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], height as u32);
            st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
            let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
            st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8));
            st64(&mut entry[4..], 0x200);
            write_task_gva_arm64e(
                &mut host, &state.tasks[1], list_object_entry_offset(7, 32).unwrap(), &entry,
            );
            let upload = load(&mut state, &mut host, 1, 7).expect("native linear texture");
            let SampledUpload::Packed { bytes, .. } = &upload else {
                panic!("linear textures are packed");
            };
            assert_eq!(*bytes, expected, "no channel expansion or float-to-UNORM conversion");
            let ReimsVgpuSampledImage::Packed(image) = upload.image(0) else {
                panic!("linear texture must keep its native format");
            };
            assert_eq!(image.pixel_format, u32::from(format));
            assert_eq!(image.bytes_per_row, tight as u32);
            assert_eq!(image.data_len, tight * height);
        }
    }

    #[test]
    fn metal_buffer_texture_upload_preserves_native_channels_offset_and_pitch() {
        for format in [pixel_format::MTL_FORMAT_RGBA16_UNORM, pixel_format::MTL_FORMAT_RGBA16_FLOAT] {
            let mut host = FakeHost::new();
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));

            const OFFSET: usize = 8;
            const TIGHT: usize = 16;
            const PITCH: usize = 32;
            let mut backing = vec![0xEE; OFFSET + PITCH + TIGHT];
            let expected: Vec<u8> = (0..2 * TIGHT).map(|v| v as u8).collect();
            for row in 0..2 {
                backing[OFFSET + row * PITCH..OFFSET + row * PITCH + TIGHT]
                    .copy_from_slice(&expected[row * TIGHT..(row + 1) * TIGHT]);
            }
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
            let mut buffer = [0u8; 16];
            st64(&mut buffer, backing.len() as u64);
            st32(&mut buffer[8..], 5);

            let mut body = vec![0u8; crate::runtime::heap_query::WIDE_TEXTURE_BODY_LEN];
            body[offset_of!(W, type_and_flags)] = 2;
            st16(&mut body[offset_of!(W, pixel_format)..], format);
            st32(&mut body[offset_of!(W, width)..], 2);
            st32(&mut body[offset_of!(W, height)..], 2);
            st32(&mut body[offset_of!(W, depth)..], 1);
            st16(&mut body[offset_of!(W, mipmap_level_count)..], 1);
            st16(&mut body[offset_of!(W, sample_count)..], 1);
            st16(&mut body[offset_of!(W, array_length)..], 1);
            let mut texture = vec![0u8; BUF_TEX_WIDE_LEN];
            st32(&mut texture[TEXTURE_VIEW_DESC_OPCODE..], TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE);
            st32(&mut texture[TEXTURE_VIEW_DESC_LEN..], BUF_TEX_WIDE_LEN as u32);
            st32(&mut texture[TEXTURE_VIEW_DESC_TEXTURE_REF..], 21);
            st32(&mut texture[BUF_TEX_DESC_BUFFER_REF..], 7);
            st64(&mut texture[BUF_TEX_DESC_OFFSET..], OFFSET as u64);
            st64(&mut texture[BUF_TEX_DESC_BYTES_PER_ROW..], PITCH as u64);
            texture[BUF_TEX_WIDE_DESC_BODY..].copy_from_slice(&body);

            for (reference, kind, gva, descriptor) in [
                (7, OBJECT_TYPE_BUFFER, 0x200, buffer.as_slice()),
                (21, OBJECT_TYPE_TEXTURE_VIEW, 0x300, texture.as_slice()),
            ] {
                write_task_gva_arm64e(&mut host, &state.tasks[1], gva, descriptor);
                let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
                st32(&mut entry, u32::from(kind) | ((descriptor.len() as u32) << 8));
                st64(&mut entry[4..], gva);
                write_task_gva_arm64e(
                    &mut host, &state.tasks[1],
                    list_object_entry_offset(reference, 32).unwrap(), &entry,
                );
            }
            let upload = load(&mut state, &mut host, 1, 21).expect("native buffer texture");
            let SampledUpload::Packed { bytes, .. } = &upload else {
                panic!("buffer textures must use packed uploads");
            };
            assert_eq!(*bytes, expected, "no UNORM8 conversion or row padding");
            assert_eq!(upload.byte_len(), expected.len() as u64);
            let ReimsVgpuSampledImage::Packed(image) = upload.image(3) else {
                panic!("buffer textures must retain their native packed format");
            };
            assert_eq!(image.binding, REIMS_VGPU_BINDING_TEXTURE_BASE + 3);
            assert_eq!((image.width, image.height), (2, 2));
            assert_eq!(image.pixel_format, u32::from(format));
            assert_eq!(image.bytes_per_row, TIGHT as u32);
            assert_eq!(image.data_len, expected.len());
            assert_eq!(image.data, bytes.as_ptr());
        }
    }
}
