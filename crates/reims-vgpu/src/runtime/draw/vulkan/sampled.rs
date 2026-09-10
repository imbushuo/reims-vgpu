//! Native sampled byte uploads. Admission comes from the Vulkan pixel table,
//! not from the lossy RGBA8 loader's list of conversion arms.

use super::*;
use crate::runtime::compute_exec::{stage_buffer_texture, vulkan::VulkanStage, ComputeStatus};

/// `None` retains a real channel conversion (notably A8); unsupported host
/// layouts are refusals, never permission to quantize native float channels.
pub(super) fn native_byte_format(
    format: u16,
    supported: impl FnOnce(TexelLayout) -> bool,
) -> Result<Option<SampledByteFormat>, ComputeStatus> {
    let Ok((layout, _, components)) = translate::pixel::sampled_pixels(format) else {
        return Ok(None);
    };
    if components != pixel_format::swizzle_identity() {
        return Ok(None);
    }
    if !supported(layout) {
        return Err(ComputeStatus::Unsupported("sampled_native_format_host_unsupported"));
    }
    Ok(Some(SampledByteFormat::from_source(layout, format)))
}

pub(super) fn buffer_source<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    descriptor: &BufferTextureDescriptor,
    supported: impl FnOnce(TexelLayout) -> bool,
) -> Result<(u32, u32, SampledSourceRequest), ComputeStatus> {
    let d = &descriptor.desc;
    if d.texture_type != 2 || d.depth != 1 || d.mipmap_level_count != 1
        || d.sample_count != 1 || d.array_length != 1
    {
        return Err(ComputeStatus::Unsupported("sampled_buffer_texture_shape"));
    }
    let format = if d.pixel_format == 0 { MTL_FORMAT_BGRA8_UNORM } else { d.pixel_format };
    let Some(byte_format) = native_byte_format(format, supported)? else {
        let (w, h, rgba) = load_buffer_texture_rgba(state, host, task_id, texture_ref, descriptor)
            .ok_or(ComputeStatus::Unsupported("sampled_buffer_texture_conversion"))?;
        return Ok((w, h, SampledSourceRequest::Bytes(
            std::sync::Arc::new(rgba), None,
            SampledByteFormat::from_source(TexelLayout::Rgba8, format),
            crate::backend::vulkan::engine::SampledByteOrigin::BufferBackedTexture,
        )));
    };
    // The shared staging owner pays both aliases' writeback debt, validates the
    // exact final-row extent, and gathers native rows without RGBA8 narrowing.
    let staged = stage_buffer_texture::<VulkanStage, _>(
        state, host, task_id, texture_ref, 0, false, descriptor,
    )?;
    Ok((staged.width, staged.height, SampledSourceRequest::Bytes(
        std::sync::Arc::new(staged.bytes), None, byte_format,
        crate::backend::vulkan::engine::SampledByteOrigin::BufferBackedTexture,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::protocol::endian::{st16, st32, st64};
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER,
    };
    use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
    use crate::runtime::host::FakeHost;

    fn descriptor(format: u16, tight: usize) -> BufferTextureDescriptor {
        BufferTextureDescriptor {
            new_texture_ref: 21, buffer_ref: 7, offset: 8, bytes_per_row: (tight + 16) as u64,
            desc: crate::runtime::heap_query::TextureDescriptor {
                texture_type: 2, framebuffer_only: false, is_drawable: false,
                allow_gpu_optimized_contents: false, usage: 1, pixel_format: format,
                width: 2, height: 2, depth: 1, mipmap_level_count: 1, sample_count: 1,
                array_length: 1, resource_options: 0, protection_options: 0, swizzle: None,
            },
        }
    }

    #[test]
    fn buffer_upload_preserves_native_bytes_offset_pitch_and_last_row_extent() {
        for format in [
            pixel_format::MTL_FORMAT_R16_FLOAT,
            pixel_format::MTL_FORMAT_RG16_FLOAT,
            pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            pixel_format::MTL_FORMAT_RGBA32_FLOAT,
            pixel_format::MTL_FORMAT_RGBA16_UNORM,
            pixel_format::MTL_FORMAT_RGB10A2_UNORM,
            pixel_format::MTL_FORMAT_BGR10A2_UNORM,
            pixel_format::MTL_FORMAT_RG11B10_FLOAT,
        ] {
            let tight = pixel_format::tight_row_bytes(2, format).unwrap() as usize;
            let descriptor = descriptor(format, tight);
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));
            let expected: Vec<u8> = (0..tight * 2).map(|b| b as u8).collect();
            let mut bytes = vec![0xee; 8 + tight + 16 + tight];
            for y in 0..2 {
                let start = 8 + y * (tight + 16);
                bytes[start..start + tight].copy_from_slice(&expected[y*tight..(y+1)*tight]);
            }
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &bytes);
            let mut buffer = [0; 16];
            st64(&mut buffer, bytes.len() as u64);
            st32(&mut buffer[8..], 5);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &buffer);
            let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
            st32(&mut entry, u32::from(OBJECT_TYPE_BUFFER) | (16 << 8));
            st64(&mut entry[4..], 0x200);
            write_task_gva_arm64e(&mut host, &state.tasks[1],
                list_object_entry_offset(7, 32).unwrap(), &entry);
            let (w, h, source) = buffer_source(
                &mut state, &mut host, 1, 21, &descriptor, |_| true,
            ).unwrap();
            let SampledSourceRequest::Bytes(bytes, _, actual_format, _) = source else {
                panic!("buffer-backed image must own copied native rows");
            };
            assert_eq!((w, h), (2, 2));
            assert_eq!(*bytes, expected);
            assert_eq!(actual_format.layout(),
                translate::pixel::sampled_pixels(format).unwrap().0);
        }
    }

    #[test]
    fn buffer_upload_refuses_unadmitted_shapes_before_guest_io() {
        for shape in 0..5 {
            let mut descriptor = descriptor(pixel_format::MTL_FORMAT_RGBA16_FLOAT, 16);
            match shape {
                0 => descriptor.desc.texture_type = 7,
                1 => descriptor.desc.depth = 2,
                2 => descriptor.desc.mipmap_level_count = 2,
                3 => descriptor.desc.sample_count = 4,
                _ => descriptor.desc.array_length = 2,
            }
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            assert!(matches!(buffer_source(
                &mut state, &mut host, 1, 21, &descriptor, |_| panic!("shape precedes capability"),
            ), Err(ComputeStatus::Unsupported("sampled_buffer_texture_shape"))));
        }
    }

    #[test]
    fn compute_and_draw_sampled_r16float_and_rgba32float_stage_native_padded_rows() {
        use crate::runtime::compute_exec::stage_texture_raw;
        use crate::runtime::decode::resource::*;
        use crate::backend::vulkan::engine::StorageImageFormat;

        for (format, expected_format, width, height) in [
            (pixel_format::MTL_FORMAT_R16_FLOAT, StorageImageFormat::R16Float, 4u32, 2u32),
            (pixel_format::MTL_FORMAT_RGBA32_FLOAT, StorageImageFormat::Rgba32Float, 1, 1),
            (pixel_format::MTL_FORMAT_RGBA32_FLOAT, StorageImageFormat::Rgba32Float, 4, 1),
        ] {
            let tight = pixel_format::tight_row_bytes(width, format).unwrap() as usize;
            let pitch = tight + 16;
            let values: Vec<u8> = if format == pixel_format::MTL_FORMAT_R16_FLOAT {
                [0x0001u16, 0x3555, 0x3c01, 0xbc00, 0x4000, 0x7c00, 0x7e55, 0x8000]
                    .into_iter().flat_map(u16::to_le_bytes).collect()
            } else {
                (0..width).flat_map(|x|
                    [-0.75f32 - x as f32, 0.003, 3.5, -2.25]
                        .into_iter().flat_map(f32::to_le_bytes)
                ).collect()
            };
            let mut backing = vec![0xee; pitch * height as usize];
            for y in 0..height as usize {
                backing[y*pitch..y*pitch+tight].copy_from_slice(&values[y*tight..(y+1)*tight]);
            }
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            let mut host = FakeHost::new();
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
            let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
            st64(&mut descriptor[LINEAR_DESC_SIZE..], backing.len() as u64);
            st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
            st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
            st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], backing.len() as u32);
            st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], pitch as u32);
            st32(&mut descriptor[TEXTURE_DESC_WIDTH..], width);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], height);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
            st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], format);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], width);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], height);
            st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
            let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
            st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8));
            st64(&mut entry[4..], 0x200);
            write_task_gva_arm64e(&mut host, &state.tasks[1],
                list_object_entry_offset(7, 32).unwrap(), &entry);
            let staged = stage_texture_raw::<VulkanStage, _>(
                &mut state, &mut host, 1, 7, 32, false,
            ).unwrap_or_else(|error| panic!("format={format:#x} {width}x{height}: {error:?}"));
            assert_eq!(staged.bytes, values);
            assert_eq!(staged.pixel_format, format);
            assert_eq!(staged.mip_levels, 1);
            assert!(!staged.is_storage);
            assert!(staged.rail.serve.is_none());
            assert_eq!(translate::pixel::sampled_image(format).unwrap(), expected_format);
            let (draw_bytes, draw_format) = crate::runtime::draw::texture_view::load_linear_texture_host(
                &mut state, &mut host, 1, 7, 0, None,
                crate::runtime::draw::texture_view::NativeUploads::ALL,
                crate::runtime::render_writeback::SettleSite::LinearTextureSampled,
            ).expect("draw's CPU-origin native upload must admit the same bytes");
            assert_eq!(draw_bytes, values);
            assert_eq!(draw_format.layout(),
                translate::pixel::sampled_pixels(format).unwrap().0);
        }
    }

    #[test]
    fn native_upload_admission_preserves_float_packed_and_plane_layouts() {
        for format in [
            pixel_format::MTL_FORMAT_R16_FLOAT,
            pixel_format::MTL_FORMAT_RG16_FLOAT,
            pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            pixel_format::MTL_FORMAT_R32_FLOAT,
            pixel_format::MTL_FORMAT_RGBA32_FLOAT,
            pixel_format::MTL_FORMAT_RGBA16_UNORM,
            pixel_format::MTL_FORMAT_RGB10A2_UNORM,
            pixel_format::MTL_FORMAT_BGR10A2_UNORM,
            pixel_format::MTL_FORMAT_RG11B10_FLOAT,
            pixel_format::MTL_FORMAT_R16_UNORM,
            pixel_format::MTL_FORMAT_RG16_UNORM,
        ] {
            let expected = translate::pixel::sampled_pixels(format).unwrap().0;
            assert_eq!(native_byte_format(format, |_| true).unwrap().unwrap().layout(), expected);
            assert!(matches!(native_byte_format(format, |_| false),
                Err(ComputeStatus::Unsupported("sampled_native_format_host_unsupported"))));
        }
    }

    #[test]
    fn native_upload_does_not_lose_channel_mapping_or_transfer_function() {
        assert!(native_byte_format(pixel_format::MTL_FORMAT_A8_UNORM, |_| true)
            .unwrap().is_none());
        let srgb = native_byte_format(pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB, |_| true)
            .unwrap().unwrap();
        assert_eq!(srgb.layout(), TexelLayout::Bgra8);
        assert!(srgb.srgb_source().is_some());
    }
}
