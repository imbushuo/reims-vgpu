use super::*;
use crate::model::{DeviceId, TaskResource, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::resource::{
    ListObjectEntry, LINEAR_DESC_HANDLE, LINEAR_DESC_MIN_LEN, LINEAR_DESC_SIZE, OBJECT_TYPE_BUFFER,
};
use crate::runtime::host::{FakeHost, MemError};
use std::sync::Arc;

pub(crate) struct Fixture {
    pub state: DeviceState,
    pub host: FakeHost,
    pub page: u64,
}

impl Fixture {
    pub fn new(shift: u32) -> Self {
        use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        let page = 1u64 << shift;
        let mut state = DeviceState::new(DeviceId(1), shift);
        let mut host = FakeHost::new();
        host.map_range(2 * page, 8, 0);
        host.map_range(3 * page, page as usize, 0);
        host.write_gpa(
            2 * page + u64::from(DIRECTORY_ROOT_PFN),
            &3u32.to_le_bytes(),
        )
        .unwrap();
        host.write_gpa(2 * page + u64::from(DIRECTORY_DEPTH), &1u32.to_le_bytes())
            .unwrap();
        for (virtual_page, pfn, byte) in [(1u64, 8u32, 0x11), (2, 13, 0x22), (3, 21, 0x33)] {
            host.map_range(u64::from(pfn) * page, page as usize, byte);
            host.write_gpa(3 * page + virtual_page * 4, &pfn.to_le_bytes())
                .unwrap();
        }
        state.define_task(1, 0x1000, 2);
        Self { state, host, page }
    }

    pub fn resource(&self, handle: u32, size: u64) -> Arc<TaskResource> {
        let mut descriptor = vec![0; LINEAR_DESC_MIN_LEN];
        descriptor[LINEAR_DESC_SIZE..LINEAR_DESC_SIZE + 8].copy_from_slice(&size.to_le_bytes());
        descriptor[LINEAR_DESC_HANDLE..LINEAR_DESC_HANDLE + 8]
            .copy_from_slice(&u64::from(handle).to_le_bytes());
        Arc::new(TaskResource::new(
            ListObjectEntry {
                object_type: OBJECT_TYPE_BUFFER,
                descriptor_length: LINEAR_DESC_MIN_LEN as u32,
                descriptor_gva: 0,
            },
            descriptor.into(),
        ))
    }

    pub fn bind(&self, reference: u32, handle: u32, size: u64, offset: u64) -> BufferBind {
        let resource = self.resource(handle, size);
        let declared = self
            .state
            .declare_object(
                1,
                reference,
                reims_vgpu_core::lifecycle::Storage::Dedicated {
                    backing: reims_vgpu_core::access::BackingId(u64::from(handle) * self.page),
                    extent: reims_vgpu_core::access::ByteRange {
                        offset: 0,
                        length: size,
                    },
                },
            )
            .unwrap();
        self.state
            .task_resources
            .register(1, declared.id, Arc::clone(&resource));
        BufferBind {
            buffer_ref: reference,
            resource: Some(resource),
            offset,
            ..Default::default()
        }
    }
}

#[test]
fn checked_buffer_destination_preserves_full_suffix_and_scattered_page_geometry() {
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        let mut fixture = Fixture::new(shift);
        let bind = fixture.bind(7, 1, fixture.page + 5, fixture.page - 3);
        let window =
            prepare_bound_buffer_read(&mut fixture.state, &mut fixture.host, 1, &bind).unwrap();
        assert_eq!(
            window.len, 8,
            "allocation size minus offset, not a reflected extent"
        );
        let mut bytes = [0xa5; 8];
        window.read_into(&mut bytes).unwrap();
        assert_eq!(bytes, [0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22, 0x22]);
        assert_eq!(window.read_vec().unwrap(), bytes);
    }
}

#[test]
fn checked_buffer_destination_refuses_wrong_length_without_touching_it() {
    let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
    let bind = fixture.bind(7, 1, 8, 0);
    let window =
        prepare_bound_buffer_read(&mut fixture.state, &mut fixture.host, 1, &bind).unwrap();
    for len in [0, 7, 9] {
        let mut bytes = vec![0xa5; len];
        assert_eq!(window.read_into(&mut bytes), Err(MemError::BadArgs));
        assert_eq!(bytes, vec![0xa5; len]);
    }
}

#[test]
fn checked_buffer_destination_propagates_partial_page_failure() {
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        let mut fixture = Fixture::new(shift);
        let bind = fixture.bind(7, 1, fixture.page + 5, fixture.page - 3);
        fixture
            .host
            .write_gpa(3 * fixture.page + 2 * 4, &0u32.to_le_bytes())
            .unwrap();
        let window =
            prepare_bound_buffer_read(&mut fixture.state, &mut fixture.host, 1, &bind).unwrap();
        let mut bytes = [0xa5; 8];
        assert_eq!(
            window.read_into(&mut bytes),
            Err(MemError::Unresolved(
                reims_vgpu_paging::resolve::ResolveStatus::ErrZeroPfn
            )),
        );
        assert_eq!(
            &bytes[..3],
            &[0x11; 3],
            "the reader may have written a prefix before refusal"
        );
        assert_eq!(
            &bytes[3..],
            &[0xa5; 5],
            "that prefix is not a complete, sealable input"
        );
        assert!(window.read_vec().is_none());
    }
}

#[test]
fn checked_buffer_window_rejects_empty_and_overflowing_addresses() {
    let mut fixture = Fixture::new(PAGE_SHIFT_X86);
    for (gva, size, offset) in [
        (fixture.page, 8, 8),
        (fixture.page, 8, 9),
        (u64::MAX - 4, 8, 0),
    ] {
        assert!(prepare_buffer_read(
            &mut fixture.state,
            &mut fixture.host,
            1,
            7,
            &BufferBacking { gva, size },
            offset,
            None,
        )
        .is_none());
    }
}

#[test]
fn checked_bound_buffer_keeps_resource_identity_across_numeric_reference_reuse() {
    let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
    let old = fixture.bind(7, 1, 8, 2);
    assert!(fixture.state.delete_object(1, 7));
    let new = fixture.bind(7, 2, 16, 4);
    let synthetic = BufferBind {
        resource: None,
        ..new.clone()
    };
    for (bind, byte, len) in [(&old, 0x11, 6), (&new, 0x22, 12), (&synthetic, 0x22, 12)] {
        let window =
            prepare_bound_buffer_read(&mut fixture.state, &mut fixture.host, 1, bind).unwrap();
        assert_eq!(window.read_vec().unwrap(), vec![byte; len]);
    }
    fixture
        .host
        .write_gpa(8 * fixture.page + 2, &[0x77; 6])
        .unwrap();
    let window = prepare_bound_buffer_read(&mut fixture.state, &mut fixture.host, 1, &old).unwrap();
    assert_eq!(
        window.read_vec().unwrap(),
        [0x77; 6],
        "resource identity is retained, not a cached byte snapshot"
    );
}

#[test]
#[cfg(feature = "backend-vulkan")]
fn checked_buffer_window_settles_writeback_debt_before_destination_fill() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    state.define_task(1, 0x1000, 9);
    assert!(
        state
            .pending_writebacks
            .arm(
                7,
                crate::runtime::writeback_debt::test_resident_identity(7, 64, 64, 1),
                64,
                64,
                1,
            )
            .is_none(),
        "a first debt must not evict another writeback obligation"
    );
    let window = prepare_buffer_read(
        &mut state,
        &mut host,
        1,
        7,
        &BufferBacking {
            gva: 0x8000,
            size: 8,
        },
        0,
        None,
    )
    .unwrap();
    assert!(
        window.state.pending_writebacks.get(7).is_none(),
        "preparation, not a later read, pays the debt"
    );
    assert!(window.read_into(&mut [0; 8]).is_err());
}
