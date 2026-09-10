//! The CPU GVA Store consumes native draw bytes and publishes canonical caches.

use super::*;

pub(super) fn publish_gva_draw_pixels<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    color: &ColorRtRequest,
    pixels: &mut [u8],
    pixels_bgra: bool,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> bool {
    // Both the licensed writer and host-cache publisher take semantic RGBA.
    // Anonymous BGRA attachments are not an exception to that input contract.
    reorder_rb_in_place(pixels, pixels_bgra, false);
    if write_gva_rgba8_within(
        state, host, task_id, color.target_gva, color.width, color.height,
        color.row_stride, color.format, pixels, allowed,
    ).is_err() {
        return false;
    }
    let object_type = objects::lookup_list_entry(state, host, task_id, color.texture_ref)
        .map(|entry| entry.object_type).unwrap_or(0);
    // Only successful guest publication licenses guest_holds_bytes.
    host_cache_store_gva_layer(
        state, host, task_id, color.texture_ref, object_type, color.target_gva,
        color.width, color.height, pixels, true,
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::gva_mem::{
        define_task_pages_arm64e, read_task_gva, write_task_gva_arm64e,
    };
    use crate::runtime::host::FakeHost;
    use crate::runtime::surface_cache;
    use std::collections::HashSet;

    fn fixture(format: u16) -> (DeviceState, FakeHost, ColorRtRequest) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        let color = ColorRtRequest {
            texture_ref: 7, target_gva: 1 << PAGE_SHIFT_ARM64E,
            width: 2, height: 2, row_stride: 16, format, ..Default::default()
        };
        write_task_gva_arm64e(&mut host, &state.tasks[1], color.target_gva, &[0xee; 32]);
        (state, host, color)
    }

    #[test]
    fn graphics_output_native_orders_publish_correct_guest_bytes_and_sampling_caches() {
        let rgba = [
            0xab, 0xff, 0xff, 0xff, 0x55, 0x40, 0x00, 0xff,
            0x10, 0x20, 0x30, 0x7f, 0x43, 0x65, 0x87, 0x9a,
        ];
        let bgra = swap_rb_channels(&rgba);
        let allowed = HashSet::from([5u64 << PAGE_SHIFT_ARM64E]);
        for (format, destination_bgra) in [
            (pixel_format::MTL_FORMAT_RGBA8_UNORM, false),
            (pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB, false),
            (pixel_format::MTL_FORMAT_BGRA8_UNORM, true),
            (pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB, true),
        ] {
            for source_bgra in [false, true] {
                let (mut state, mut host, color) = fixture(format);
                let mut pixels = if source_bgra { bgra.clone() } else { rgba.to_vec() };
                assert!(publish_gva_draw_pixels(
                    &mut state, &mut host, 1, &color, &mut pixels, source_bgra, Some(&allowed),
                ));
                assert_eq!(pixels, rgba, "the publisher owns canonicalization");
                let mut guest = [0; 32];
                read_task_gva(
                    &host, &state.tasks[1], color.target_gva, &mut guest, PAGE_SHIFT_ARM64E,
                ).unwrap();
                let expected = if destination_bgra { bgra.as_slice() } else { &rgba };
                assert_eq!(&guest[..8], &expected[..8]);
                assert_eq!(&guest[16..24], &expected[8..]);
                assert_eq!(&guest[8..16], &[0xee; 8]);
                assert_eq!(&guest[24..32], &[0xee; 8]);
                assert_eq!(surface_cache::get_gva(&state, color.target_gva, 2, 2), Some(bgra.as_slice()));
                assert_eq!(surface_cache::get_texture(&state, 1, 7, 2, 2), Some(bgra.as_slice()));
                assert!(state.host_gva_surfaces[&color.target_gva].guest_holds_bytes);
            }
        }
    }

    #[test]
    fn graphics_output_failed_licensed_store_cannot_publish_sampling_cache() {
        let (mut state, mut host, color) = fixture(pixel_format::MTL_FORMAT_BGRA8_UNORM);
        let mut pixels = vec![0, 0x40, 0x55, 0xff].repeat(4);
        assert!(!publish_gva_draw_pixels(
            &mut state, &mut host, 1, &color, &mut pixels, true, Some(&HashSet::new()),
        ));
        let mut guest = [0; 32];
        read_task_gva(
            &host, &state.tasks[1], color.target_gva, &mut guest, PAGE_SHIFT_ARM64E,
        ).unwrap();
        assert_eq!(guest, [0xee; 32]);
        assert!(surface_cache::get_gva(&state, color.target_gva, 2, 2).is_none());
        assert!(surface_cache::get_texture(&state, 1, 7, 2, 2).is_none());
    }
}
