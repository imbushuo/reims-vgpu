use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::runtime::gva_mem::define_task_pages_arm64e;
use crate::runtime::host::FakeHost;
use crate::runtime::writeback_debt::{gva_resource_generation, GvaResourceKey};
use reims_vgpu_protocol::pass_action::{MTL_STORE_ACTION_DONT_CARE, MTL_STORE_ACTION_STORE};

fn request(format: u16) -> DrawEncodeRequest {
    DrawEncodeRequest {
        task_id: 1,
        gva_alloc_gen: 1,
        colors: vec![ColorRtRequest {
            texture_ref: 7,
            target_gva: 1 << PAGE_SHIFT_ARM64E,
            width: 4,
            height: 2,
            row_stride: 80,
            format,
            store_action: MTL_STORE_ACTION_STORE,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn primary_gva_store_selection_preserves_deferred_and_native_only_sync_routes() {
    let state = DeviceState::new(DeviceId(0xad77), PAGE_SHIFT_ARM64E);
    for format in [
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        pixel_format::MTL_FORMAT_RG16_FLOAT,
        pixel_format::MTL_FORMAT_RGBA32_FLOAT,
    ] {
        for allow_deferred in [false, true] {
            let req = request(format);
            let mut resources = DrawRequest::default();
            let store = GvaStore::prepare(&state, &req, &mut resources, allow_deferred, true);
            let native = requires_native_store(format);
            assert_eq!(store.is_some(), allow_deferred || native);
            if let Some(store) = store {
                assert_eq!(resources.target_identity.as_ref(), Some(&store.identity));
                assert!(resources.skip_readback);
                assert_eq!(matches!(store.timing, Timing::MayDefer), allow_deferred);
            } else {
                assert!(!resources.skip_readback);
                assert!(resources.target_identity.is_none());
            }
        }
    }
}

#[test]
fn primary_gva_store_never_claims_intermediate_discard_or_mapped_records() {
    let state = DeviceState::new(DeviceId(0xad77), PAGE_SHIFT_ARM64E);
    for case in 0..5 {
        let mut req = request(pixel_format::MTL_FORMAT_RGBA16_FLOAT);
        let writeback_guest = case != 0;
        match case {
            1 => req.colors[0].store_action = MTL_STORE_ACTION_DONT_CARE,
            2 => req.colors[0].mapping_id = 1,
            3 => req.colors[0].target_gva = 0,
            4 => req.colors.clear(),
            _ => {}
        }
        let mut resources = DrawRequest::default();
        assert!(GvaStore::prepare(&state, &req, &mut resources, true, writeback_guest).is_none());
        assert!(resources.target_identity.is_none());
        assert!(!resources.skip_readback);
    }
}

#[test]
fn primary_gva_store_successful_deferred_arm_never_reads_or_copies_pixels() {
    for format in [
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    ] {
        let mut state = DeviceState::new(DeviceId(0xad76), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));
        let mut descriptor = [0; 16];
        descriptor[..8].copy_from_slice(&160u64.to_le_bytes());
        descriptor[8..].copy_from_slice(&1u64.to_le_bytes());
        crate::runtime::gva_mem::write_task_gva_arm64e(
            &mut host,
            &state.tasks[1],
            0x200,
            &descriptor,
        );
        let mut entry = [0; 12];
        entry[..4].copy_from_slice(
            &(u32::from(crate::runtime::decode::resource::OBJECT_TYPE_BUFFER) | (16 << 8))
                .to_le_bytes(),
        );
        entry[4..].copy_from_slice(&0x200u64.to_le_bytes());
        crate::runtime::gva_mem::write_task_gva_arm64e(&mut host, &state.tasks[1], 7 * 12, &entry);
        crate::runtime::objects::resolve_resource(&state, &host, 1, 7).unwrap();
        let mut req = request(format);
        let color = &req.colors[0];
        req.gva_alloc_gen = gva_resource_generation(
            &mut state,
            &host,
            GvaResourceKey {
                task_id: 1,
                texture_ref: 7,
            },
            color.target_gva,
            u64::from(color.row_stride * color.height),
        );
        let mut resources = DrawRequest::default();
        let store = GvaStore::prepare(&state, &req, &mut resources, true, true).unwrap();
        let window = crate::backend::vulkan::gva_window(&store.identity).unwrap();
        let sync = crate::runtime::drain::store_route_count("gva_store_sync");
        // There is deliberately no Vulkan device or rendered image in this fixture.
        assert!(matches!(
            store.finish(&mut state, &mut host, &req, None),
            Ok(StoreOutput::Complete)
        ));
        assert!(crate::runtime::writeback_debt::gva_resident_authoritative(
            &state, window
        ));
        assert_eq!(
            crate::runtime::drain::store_route_count("gva_store_sync"),
            sync
        );
    }
}
