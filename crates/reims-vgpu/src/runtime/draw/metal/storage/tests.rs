use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::decode::resource::*;
use crate::runtime::gva_mem::{define_task_pages_arm64e, read_task_gva, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;

fn fixture(levels: u16) -> (DeviceState, FakeHost, DrawEncodeRequest, Vec<u8>) {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let mut backing = vec![0xEE; 80];
    for y in 0..2 {
        for x in 0..32 {
            backing[y * 48 + x] = (y * 32 + x) as u8;
        }
    }
    write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
    let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
    st64(&mut descriptor[LINEAR_DESC_SIZE..], backing.len() as u64);
    st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
    st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels);
    st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], backing.len() as u32);
    st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], 48);
    st32(&mut descriptor[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], 2);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
    st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], pixel_format::MTL_FORMAT_RGBA16_FLOAT);
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], 4);
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], 2);
    st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
    write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
    let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
    st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8));
    st64(&mut entry[4..], 0x200);
    write_task_gva_arm64e(
        &mut host, &state.tasks[1], list_object_entry_offset(7, 32).unwrap(), &entry,
    );
    (state, host, DrawEncodeRequest { task_id: 1, ..Default::default() }, backing)
}

#[test]
fn writable_aliases_share_native_identity_and_publish_partial_native_texels() {
    objc::rc::autoreleasepool(|| {
        let (mut state, mut host, req, mut expected) = fixture(1);
        let mut storage = StorageTextures::default();
        storage.add(&mut state, &mut host, &req, 7, 3).unwrap();
        storage.add(&mut state, &mut host, &req, 7, 4).unwrap();
        assert_eq!(storage.textures.len(), 1);
        let Some(ReimsVgpuSampledImage::Native { texture: first, .. }) = storage.image(7, 3)
            else { panic!("writable native binding") };
        let Some(ReimsVgpuSampledImage::Native { texture: second, .. }) = storage.image(7, 4)
            else { panic!("sampled alias must use the writable texture") };
        assert!(std::ptr::eq(first.as_ref(), second.as_ref()));
        let replacement = [0x00u8, 0x40, 0x00, 0xBC, 0x01, 0x38, 0x00, 0x3C];
        first.replace_region(
            ::metal::MTLRegion::new_2d(1, 1, 1, 1), 0,
            replacement.as_ptr().cast(), replacement.len() as u64,
        );
        storage.publish(&mut state, &mut host, req.task_id).unwrap();
        expected[56..64].copy_from_slice(&replacement);
        let mut actual = vec![0; expected.len()];
        read_task_gva(
            &host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &mut actual, PAGE_SHIFT_ARM64E,
        ).unwrap();
        assert_eq!(actual, expected, "untouched texels and row padding survive");
        let staged = stage_texture_raw::<MetalStage, _>(
            &mut state, &mut host, 1, 7, REIMS_VGPU_BINDING_TEXTURE_BASE, false,
        ).unwrap();
        assert_eq!(&staged.bytes[40..48], &replacement, "next draw sees native float bytes");
    });
}

#[test]
fn writable_pyramids_and_attachment_aliases_are_not_flattened() {
    let (mut state, mut host, mut req, _) = fixture(2);
    let mut storage = StorageTextures::default();
    assert_eq!(
        storage.add(&mut state, &mut host, &req, 7, 0),
        Err(EncodeStatus::Unsupported("draw_mtl_storage_texture_shape")),
    );
    req.colors.push(ColorRtRequest { texture_ref: 7, ..Default::default() });
    assert_eq!(
        storage.add(&mut state, &mut host, &req, 7, 0),
        Err(EncodeStatus::Unsupported("draw_mtl_storage_attachment_alias")),
    );
    assert!(storage.textures.is_empty());
}
