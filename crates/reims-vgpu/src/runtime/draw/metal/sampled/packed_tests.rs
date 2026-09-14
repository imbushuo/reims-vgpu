use super::*;
use crate::backend::metal::error::Status;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::decode::resource::*;
use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;

fn layout() -> packed::Layout {
    packed::Layout {
        width: 2,
        height: 2,
        pixel_format: u32::from(pixel_format::MTL_FORMAT_RGBA8_UNORM),
        bytes_per_row: 16,
    }
}

fn native(upload: SampledUpload) -> Arc<packed::SampledImage> {
    match upload {
        SampledUpload::ImmutablePacked(image) => image,
        _ => panic!("a lawful packed resource must own its snapshot"),
    }
}

#[test]
fn packed_cache_unchanged_content_allocates_and_uploads_once() {
    let mut held = RetainedPacked::default();
    let bytes = vec![0x5a; 64];
    let first = held
        .stage(layout(), bytes.clone(), packed::SampledImage::new)
        .unwrap();
    for _ in 0..128 {
        let next = held
            .stage(layout(), bytes.clone(), |_, _| {
                panic!("an exact hit cannot allocate or upload")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&first, &next));
    }
}

#[test]
fn packed_cache_compares_every_staged_byte_length_and_layout_field() {
    let bytes = vec![0x5a; 64];
    let base = layout();
    for changed in [
        packed::Layout { width: 4, ..base },
        packed::Layout { height: 4, ..base },
        packed::Layout {
            pixel_format: u32::from(pixel_format::MTL_FORMAT_BGRA8_UNORM),
            ..base
        },
        packed::Layout {
            bytes_per_row: 32,
            ..base
        },
        packed::Layout {
            pixel_format: 0,
            ..base
        },
    ] {
        let mut held = RetainedPacked::default();
        let first = held
            .stage(base, bytes.clone(), packed::SampledImage::new)
            .unwrap();
        let next = held
            .stage(changed, bytes.clone(), packed::SampledImage::new)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &next), "{changed:?}");
        assert_eq!(next.layout(), changed);
    }
    for offset in 0..bytes.len() {
        let mut held = RetainedPacked::default();
        let first = held
            .stage(base, bytes.clone(), packed::SampledImage::new)
            .unwrap();
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        let next = held
            .stage(base, changed.clone(), packed::SampledImage::new)
            .unwrap();
        assert!(
            !Arc::ptr_eq(&first, &next),
            "includes row padding and unused tail"
        );
        assert_eq!(next.bytes(), changed);
        assert_eq!(first.bytes(), bytes);
    }
    let mut held = RetainedPacked::default();
    let first = held
        .stage(base, bytes.clone(), packed::SampledImage::new)
        .unwrap();
    let longer = held
        .stage(base, vec![0x5a; 65], packed::SampledImage::new)
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &longer));
}

#[test]
fn packed_cache_replacement_failure_and_retirement_do_not_mutate_prior_images() {
    let resource = crate::model::TaskResource::new(
        ListObjectEntry {
            object_type: OBJECT_TYPE_TEXTURE,
            descriptor_length: 0,
            descriptor_gva: 0,
        },
        Arc::from([]),
    );
    let first = resource
        .with_rail_state(|held: &mut RetainedPacked| {
            held.stage(layout(), vec![0x41; 64], packed::SampledImage::new)
                .unwrap()
        })
        .unwrap();
    let binding = SampledUpload::ImmutablePacked(Arc::clone(&first)).image(3);
    let weak = Arc::downgrade(&first);
    let next = resource
        .with_rail_state(|held: &mut RetainedPacked| {
            held.stage(layout(), vec![0x42; 64], packed::SampledImage::new)
                .unwrap()
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &next));
    assert_eq!(first.bytes(), &[0x41; 64]);
    assert_eq!(next.bytes(), &[0x42; 64]);
    resource
        .with_rail_state(|held: &mut RetainedPacked| {
            let error = held.stage(layout(), vec![0x43; 64], |_, _| {
                Err(Status::execute("metal_render_sampled_texture_alloc_failed"))
            });
            assert!(error.is_err());
            assert!(
                held.latest.is_none(),
                "a failed replacement cannot leave an old candidate"
            );
        })
        .unwrap();
    drop(resource);
    assert_eq!(Arc::strong_count(&next), 1);
    drop(first);
    assert!(
        weak.upgrade().is_some(),
        "the pending binding owns its old immutable image"
    );
    drop(binding);
    assert!(weak.upgrade().is_none());
}

fn fixture() -> (DeviceState, FakeHost) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        5 << PAGE_SHIFT_ARM64E,
        &[0x51; 16],
    );
    let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
    st64(&mut descriptor[LINEAR_DESC_SIZE..], 16);
    st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
    st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], 16);
    st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], 8);
    st32(&mut descriptor[TEXTURE_DESC_WIDTH..], 2);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], 2);
    st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
    st16(
        &mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..],
        pixel_format::MTL_FORMAT_RGBA8_UNORM,
    );
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], 2);
    st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], 2);
    st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
    write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
    let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8),
    );
    st64(&mut entry[4..], 0x200);
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(7, 32).unwrap(),
        &entry,
    );
    (state, host)
}

#[test]
fn packed_load_reads_guest_without_dirty_tracking_and_honors_resource_retirement() {
    for reset in [false, true] {
        let (mut state, mut host) = fixture();
        host.guest_writes_unobservable = true;
        let first = native(load(&mut state, &mut host, 1, 7).unwrap());
        let again = native(load(&mut state, &mut host, 1, 7).unwrap());
        assert!(Arc::ptr_eq(&first, &again));
        // Deliberately no guest-generation or host-write notification: this
        // cache must compare the actual bytes its staging owner just read.
        write_task_gva_arm64e(
            &mut host,
            &state.tasks[1],
            5 << PAGE_SHIFT_ARM64E,
            &[0x52; 16],
        );
        let changed = native(load(&mut state, &mut host, 1, 7).unwrap());
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(changed.bytes(), &[0x52; 16]);
        assert_eq!(first.bytes(), &[0x51; 16]);
        assert_eq!(Arc::strong_count(&changed), 2);
        if reset {
            state.reset();
        } else {
            assert!(state.delete_object(1, 7));
            let recreated = native(load(&mut state, &mut host, 1, 7).unwrap());
            assert!(!Arc::ptr_eq(&changed, &recreated));
        }
        assert_eq!(Arc::strong_count(&changed), 1);
    }
}

#[test]
fn packed_cache_cannot_displace_a_different_resource_rail_owner() {
    #[derive(Default)]
    struct Other;
    impl crate::model::RailResourceState for Other {}
    let (mut state, mut host) = fixture();
    let resource = objects::resolve_resource(&state, &host, 1, 7).unwrap();
    resource.with_rail_state(|_: &mut Other| ()).unwrap();
    assert!(matches!(
        load(&mut state, &mut host, 1, 7),
        Some(SampledUpload::Packed { .. })
    ));
    assert!(resource.with_rail_state(|_: &mut Other| ()).is_some());
}
