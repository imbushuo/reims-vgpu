use super::*;
use crate::backend::metal::{raw_metal, runtime};
use crate::observe::Refusal;
use metal::{
    BufferRef, CommandBuffer, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRegion, MTLScissorRect, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLViewport, RenderPassDescriptor, RenderPipelineDescriptor, TextureDescriptor,
};

mod command_freshness;
mod contained;
mod inventory;

fn command(device: &Device) -> CommandBuffer {
    raw_metal::new_command_buffer(&runtime::thread_queue(device))
        .unwrap()
        .to_owned()
}

fn filled(device: &Device, bytes: &[u8]) -> Filled {
    unsafe {
        copy(
            device,
            bytes.as_ptr(),
            bytes.len(),
            Class::Vertex,
            "test_input_allocation",
        )
    }
    .unwrap()
}

fn contents(buffer: &BufferRef) -> Vec<u8> {
    // Only used before submission or after completion, never during GPU work.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast(), buffer.length() as usize) }
        .to_vec()
}

fn finish(submission: &mut Submission, command: &CommandBufferRef) {
    command.commit();
    command.wait_until_completed();
    submission.completed(command).unwrap();
}

fn submit_lengths(device: &Device, lengths: &[usize]) -> Vec<Buffer> {
    let pool = runtime::thread_input_pool(device);
    let cmd = command(device);
    let mut submission = Submission::default();
    submission.begin(&cmd, pool).unwrap();
    let buffers = lengths
        .iter()
        .enumerate()
        .map(|(index, &len)| {
            submission
                .seal(filled(device, &vec![index as u8 + 1; len]))
                .unwrap()
        })
        .collect();
    finish(&mut submission, &cmd);
    buffers
}

fn assert_index(pool: &Pool) {
    use std::collections::BTreeSet;
    let mut global = BTreeSet::new();
    let mut previous = None;
    let mut current = pool.oldest;
    let mut bytes = 0;
    while let Some(index) = current {
        assert!(global.insert(index), "global list cycle");
        let node = pool.node(index);
        assert_eq!(node.older, previous);
        assert!(node.input.0.account.retained);
        bytes += node.input.0.account.len;
        previous = Some(index);
        current = node.newer;
    }
    assert_eq!(previous, pool.newest);
    assert_eq!(global.len(), pool.available_count);
    assert_eq!(bytes, pool.available_bytes);
    let mut keyed = BTreeSet::new();
    let mut empty = BTreeSet::new();
    for (&len, bucket) in &pool.by_len {
        let mut previous = None;
        let mut current = bucket.newest;
        let mut count = 0;
        while let Some(index) = current {
            assert!(keyed.insert(index), "length list cycle or duplicate slot");
            let node = pool.node(index);
            assert_eq!(node.same_newer, previous);
            assert_eq!(node.input.0.account.len as usize, len);
            previous = Some(index);
            current = node.same_older;
            count += 1;
        }
        assert_eq!(count, bucket.count);
        if count == 0 {
            empty.insert(len);
        }
    }
    assert_eq!(global, keyed);
    assert_eq!(
        empty,
        pool.exhausted_lengths
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(empty.len(), pool.exhausted_lengths.len());
    let mut free = BTreeSet::new();
    let mut current = pool.free;
    while let Some(index) = current {
        assert!(free.insert(index), "free list cycle");
        assert!(!global.contains(&index));
        let Slot::Free(next) = pool.slots[index] else {
            panic!("occupied free-list slot");
        };
        current = next;
    }
    assert_eq!(free.len() + global.len(), pool.slots.len());
    assert!(pool.slots.len() <= pool.peak.count);
    assert!(pool.by_len.len() <= pool.peak.count);
    assert!(pool.available_count <= pool.peak.count);
    assert!(pool.available_bytes <= pool.peak.bytes);
}

#[test]
fn exact_lengths_full_refill_and_completion_owned_inventory() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Vertex);
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let mut source = vec![0xa5; 17];
        let first = submission.seal(filled(device, &source)).unwrap();
        source.fill(0x5a);
        let second = submission.seal(filled(device, &source)).unwrap();
        source.fill(0);
        drop(source);
        assert_eq!(first.length(), 17);
        assert_eq!(second.length(), 17);
        assert_ne!(
            first.as_ptr(),
            second.as_ptr(),
            "queued inputs cannot share live storage"
        );
        assert_eq!(contents(&first), [0xa5; 17]);
        assert_eq!(contents(&second), [0x5a; 17]);
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(
            submission.completed(&cmd).unwrap_err().refusal(),
            Some("metal_render_input_not_completed")
        );
        assert_eq!(submission.inputs.len(), 2);
        assert_eq!(snapshot(Class::Vertex).allocations - before.allocations, 2);
        finish(&mut submission, &cmd);
        assert_eq!(pool.borrow().inventory(), [(17, 2)]);

        let next = command(device);
        submission.begin(&next, pool.clone()).unwrap();
        let reused = submission.seal(filled(device, &[0xcc; 17])).unwrap();
        assert!([first.as_ptr(), second.as_ptr()].contains(&reused.as_ptr()));
        assert_eq!(reused.length(), 17);
        assert_eq!(
            contents(&reused),
            [0xcc; 17],
            "reuse refills the entire native extent"
        );
        finish(&mut submission, &next);
        assert_eq!(
            pool.borrow().inventory(),
            [(17, 2)],
            "small completions preserve unused completed inputs"
        );

        let next = command(device);
        submission.begin(&next, pool.clone()).unwrap();
        let changed = submission.seal(filled(device, &[0xdd; 19])).unwrap();
        assert_eq!(changed.length(), 19);
        finish(&mut submission, &next);
        assert_eq!(
            pool.borrow().inventory(),
            [(19, 1)],
            "historical sizes do not accumulate"
        );
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations - before.allocations, 3);
        assert_eq!(after.allocated_bytes - before.allocated_bytes, 53);
        assert_eq!(after.copies - before.copies, 4);
        assert_eq!(after.copied_bytes - before.copied_bytes, 70);
        assert_eq!(after.reuses - before.reuses, 1);
        assert_eq!(after.requests - before.requests, 4);
        assert_eq!(after.miss_absent_length - before.miss_absent_length, 3);
        assert_eq!(after.budget_discards - before.budget_discards, 2);
        assert_eq!(after.budget_discard_bytes - before.budget_discard_bytes, 34);
        assert_eq!(after.completed_inputs - before.completed_inputs, 4);
        assert_eq!(
            after.completed_input_bytes - before.completed_input_bytes,
            70
        );
        assert_eq!(after.live_bytes, before.live_bytes);
        assert_eq!(after.retained_bytes, before.retained_bytes + 19);
        assert!(after.high_water_bytes >= before.owned_bytes + 34);
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (2, 34)
        );

        let empty = command(device);
        submission.begin(&empty, pool.clone()).unwrap();
        finish(&mut submission, &empty);
        assert_eq!(
            pool.borrow().inventory(),
            [(19, 1)],
            "empty completion preserves inventory"
        );
    });
}

#[test]
fn wrong_queue_completion_and_abandoned_fill_cannot_recycle() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        drop(filled(device, &[1; 13]));
        assert!(
            pool.borrow().inventory().is_empty(),
            "unsealed fills are discarded"
        );
        let foreign_queue = device.new_command_queue();
        let foreign_command = raw_metal::new_command_buffer(&foreign_queue)
            .unwrap()
            .to_owned();
        let mut submission = Submission::default();
        assert_eq!(
            submission
                .begin(&foreign_command, pool.clone())
                .unwrap_err()
                .refusal(),
            Some("metal_render_input_queue_mismatch")
        );
        let foreign_owner = Pool::new(foreign_queue);
        submission.begin(&foreign_command, foreign_owner).unwrap();
        let Err(status) = submission.seal(filled(device, &[2; 13])) else {
            panic!("input from a different queue must refuse");
        };
        assert_eq!(status.refusal(), Some("metal_render_input_owner_mismatch"));
        finish(&mut submission, &foreign_command);

        let cmd = command(device);
        submission.begin(&cmd, pool.clone()).unwrap();
        submission.seal(filled(device, &[3; 13])).unwrap();
        assert_eq!(
            submission
                .completed(&foreign_command)
                .unwrap_err()
                .refusal(),
            Some("metal_render_input_completion_mismatch")
        );
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(submission.inputs.len(), 1);
        finish(&mut submission, &cmd);
        assert_eq!(pool.borrow().inventory(), [(13, 1)]);
        pool.borrow_mut().clear_available();
        assert!(pool.borrow().inventory().is_empty());
    });
}

#[test]
fn completed_source_is_not_recycled_into_its_own_snapshot() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let first = submission.seal(filled(device, &[0xa5; 17])).unwrap();
        finish(&mut submission, &cmd);
        let before = super::snapshot(Class::Vertex);

        let next = command(device);
        submission.begin(&next, pool.clone()).unwrap();
        let input = unsafe {
            copy(
                device,
                first.contents().cast(),
                first.length() as usize,
                Class::Vertex,
                "test_input_allocation",
            )
        }
        .unwrap();
        let snapshot = submission.seal(input).unwrap();
        let after = super::snapshot(Class::Vertex);
        assert_eq!(after.requests - before.requests, 1);
        assert_eq!(after.miss_source_overlap - before.miss_source_overlap, 1);
        assert_eq!(after.miss_absent_length - before.miss_absent_length, 0);
        assert_eq!(
            after.miss_exhausted_length - before.miss_exhausted_length,
            0
        );
        assert_ne!(first.as_ptr(), snapshot.as_ptr());
        unsafe {
            std::ptr::write_bytes(first.contents().cast::<u8>(), 0x5a, 17);
        }
        assert_eq!(contents(&snapshot), [0xa5; 17]);
        finish(&mut submission, &next);
    });
}

#[test]
fn exhausted_native_allocation_is_typed_and_not_counted_as_reuse() {
    let before = snapshot(Class::Index);
    let Err(status) = Filling::allocate(
        64,
        Class::Index,
        "metal_render_index_buffer_alloc_failed",
        || None,
    ) else {
        panic!("nil native allocation must refuse");
    };
    assert_eq!(
        status.refusal(),
        Some("metal_render_index_buffer_alloc_failed")
    );
    let after = snapshot(Class::Index);
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.copied_bytes, before.copied_bytes);
    assert_eq!(after.reuses, before.reuses);
    assert_eq!(after.live_bytes, before.live_bytes);
}

#[test]
fn native_extent_mismatch_is_typed_before_filling() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let Err(status) = Filling::allocate(17, Class::Index, "test_input_allocation", || {
            raw_metal::new_buffer(device, 19, MTLResourceOptions::StorageModeShared)
        }) else {
            panic!("different native extent must refuse");
        };
        assert_eq!(status.refusal(), Some("metal_render_input_length_mismatch"));
    });
}

#[test]
fn isolated_queue_inventory_drops_without_host_or_thread_retention() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let before = snapshot(Class::Attribute);
        let owner = Pool::new(device.new_command_queue());
        let weak = Rc::downgrade(&owner);
        let allocation = Filling::allocate(31, Class::Attribute, "test_input_allocation", || {
            raw_metal::new_buffer(device, 31, MTLResourceOptions::StorageModeShared)
        })
        .unwrap_or_else(|status| panic!("{status:?}"))
        .0;
        unsafe {
            std::ptr::write_bytes(allocation.buffer.contents().cast::<u8>(), 0, 31);
        }
        let cmd = raw_metal::new_command_buffer(&owner.borrow().queue)
            .unwrap()
            .to_owned();
        let mut submission = Submission::default();
        submission.begin(&cmd, owner.clone()).unwrap();
        submission
            .seal(Filled {
                allocation: Storage::Exclusive(allocation),
                owner: owner.clone(),
                offset: 0,
                captured_len: 31,
                gpu_read_only: None,
            })
            .unwrap();
        finish(&mut submission, &cmd);
        assert_eq!(
            snapshot(Class::Attribute).retained_bytes,
            before.retained_bytes + 31
        );
        drop(owner);
        assert!(weak.upgrade().is_none());
        assert_eq!(
            snapshot(Class::Attribute).retained_bytes,
            before.retained_bytes
        );
        assert_eq!(snapshot(Class::Attribute).owned_bytes, before.owned_bytes);
    });
}

#[test]
fn resource_suffixes_reuse_original_lengths_and_zero_uncaptured_prefixes() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        if !device.supports_family(metal::MTLGPUFamily::Apple2) {
            return;
        }
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Vertex);
        let mut submission = Submission::default();
        let mut first = Vec::new();
        for (round, offsets) in [[0u64, 4, 12], [8, 16, 24]].into_iter().enumerate() {
            let cmd = command(device);
            submission.begin(&cmd, pool.clone()).unwrap();
            let mut buffers = Vec::new();
            for (index, offset) in offsets.into_iter().enumerate() {
                if round == 0 {
                    pool.borrow_mut().poison_fresh = true;
                }
                let byte = (round * 8 + index + 1) as u8;
                let input = fill_resource_suffix::<()>(
                    device,
                    32,
                    offset,
                    Class::Vertex,
                    "test_input_allocation",
                    |suffix| {
                        assert_eq!(suffix.len() as u64, 32 - offset);
                        if round == 0 {
                            assert!(suffix.iter().all(|byte| *byte == 0));
                        }
                        suffix.fill(byte);
                        Ok(suffix.len())
                    },
                )
                .unwrap();
                assert_eq!(input.len() as u64, 32 - offset);
                assert_eq!(input.binding_offset(), offset);
                let buffer = submission.seal(input).unwrap();
                assert_eq!(buffer.length() - offset, 32 - offset);
                let bytes = contents(&buffer);
                assert_eq!(&bytes[..offset as usize], vec![0; offset as usize]);
                assert_eq!(
                    &bytes[offset as usize..],
                    vec![byte; (32 - offset) as usize]
                );
                buffers.push(buffer);
            }
            assert_eq!(
                buffers
                    .iter()
                    .map(|buffer| buffer.as_ptr() as usize)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                3,
                "uncompleted offsets need independent snapshots",
            );
            assert!(pool.borrow().inventory().is_empty());
            assert!(submission.completed(&cmd).is_err());
            if round == 0 {
                first = buffers.iter().map(|buffer| buffer.as_ptr()).collect();
            } else {
                assert!(buffers
                    .iter()
                    .all(|buffer| first.contains(&buffer.as_ptr())));
            }
            finish(&mut submission, &cmd);
            assert_eq!(pool.borrow().inventory(), [(32, 3)]);
            assert_index(&pool.borrow());
        }
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations - before.allocations, 3);
        assert_eq!(after.allocated_bytes - before.allocated_bytes, 96);
        assert_eq!(after.reuses - before.reuses, 3);
        assert_eq!(
            after.resource_shaped_fills - before.resource_shaped_fills,
            6
        );
        assert_eq!(after.direct_fill_bytes - before.direct_fill_bytes, 128);
        assert_eq!(
            after.direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            144
        );
        assert_eq!(after.live_bytes, before.live_bytes);
    });
}

#[test]
fn resource_suffix_fallbacks_preserve_bounds_without_expanded_allocations() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        for (size, offset) in [
            (32, 1),
            (device.max_buffer_length() + 16, device.max_buffer_length()),
        ] {
            let before = snapshot(Class::Fragment);
            let input = fill_resource_suffix::<()>(
                device,
                size,
                offset,
                Class::Fragment,
                "test_input_allocation",
                |suffix| {
                    assert_eq!(suffix.len() as u64, size - offset);
                    suffix.fill(0x5b);
                    Ok(suffix.len())
                },
            )
            .unwrap();
            assert_eq!(input.binding_offset(), 0);
            assert_eq!(input.allocation.buffer.length(), size - offset);
            assert_eq!(
                contents(&input.allocation.buffer),
                vec![0x5b; (size - offset) as usize]
            );
            let after = snapshot(Class::Fragment);
            assert_eq!(
                after.allocated_bytes - before.allocated_bytes,
                size - offset
            );
        }
        if device.supports_family(metal::MTLGPUFamily::Apple2) {
            for deny_headroom in [true, false] {
                if deny_headroom {
                    pool.borrow_mut().shape_headroom = Some(0);
                } else {
                    pool.borrow_mut().shape_headroom = None;
                    pool.borrow_mut().fail_next_allocation();
                }
                let before = snapshot(Class::Fragment);
                let input = fill_resource_suffix::<()>(
                    device,
                    64,
                    48,
                    Class::Fragment,
                    "test_input_allocation",
                    |suffix| {
                        assert_eq!(suffix.len(), 16);
                        suffix.fill(0x6c);
                        Ok(suffix.len())
                    },
                )
                .unwrap();
                assert_eq!(input.binding_offset(), 0);
                assert_eq!(input.allocation.buffer.length(), 16);
                assert_eq!(
                    snapshot(Class::Fragment).allocated_bytes - before.allocated_bytes,
                    16
                );
            }
            pool.borrow_mut().shape_headroom = None;
            submit_lengths(device, &[64]);
            pool.borrow_mut().shape_headroom = Some(0);
            let before = snapshot(Class::Fragment);
            let input = fill_resource_suffix::<()>(
                device,
                64,
                48,
                Class::Fragment,
                "test_input_allocation",
                |suffix| {
                    suffix.fill(0x7d);
                    Ok(suffix.len())
                },
            )
            .unwrap();
            assert_eq!(
                input.binding_offset(),
                48,
                "reuse requires no new working-set headroom"
            );
            assert_eq!(input.allocation.buffer.length(), 64);
            assert_eq!(snapshot(Class::Fragment).allocations, before.allocations);
            assert_eq!(snapshot(Class::Fragment).reuses - before.reuses, 1);
        }
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn plain_draw_offsets_reuse_one_declared_allocation_after_completion() {
    use crate::runtime::draw::metal::inputs::{prepare, Capture, PreparedInput};
    use crate::runtime::host::HostMemory;
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        if !device.supports_family(metal::MTLGPUFamily::Apple2) {
            return;
        }
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let size = fixture.page;
        let mut bind = fixture.bind(7, 1, size, 0);
        let before = snapshot(Class::Vertex);
        let mut first = None;
        let offsets = [0, 4, 20, 256, 512, size / 2, size - 16];
        for (index, offset) in offsets.into_iter().enumerate() {
            bind.offset = offset;
            let byte = index as u8 + 1;
            fixture
                .host
                .write_gpa(8 * size, &vec![byte; size as usize])
                .unwrap();
            let PreparedInput::Native(input) = prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Class::Vertex,
                Capture::Native(None),
                "draw_mtl_vertex_buffer_miss",
            )
            .unwrap() else {
                panic!("plain input must keep its native snapshot");
            };
            assert_eq!(input.bytes.binding_offset(), offset);
            assert_eq!(input.bytes.len() as u64, size - offset);
            fixture
                .host
                .write_gpa(8 * size, &vec![0xee; size as usize])
                .unwrap();
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            let buffer = submission.seal(input.bytes).unwrap();
            assert_eq!(buffer.length(), size);
            assert_eq!(buffer.length() - offset, size - bind.offset);
            let bytes = contents(&buffer);
            assert!(bytes[..offset as usize].iter().all(|byte| *byte == 0));
            assert!(bytes[offset as usize..]
                .iter()
                .all(|actual| *actual == byte));
            if let Some(first) = first {
                assert_eq!(
                    buffer.as_ptr(),
                    first,
                    "offset changes must not change the pool key"
                );
            } else {
                first = Some(buffer.as_ptr());
            }
            finish(&mut submission, &cmd);
            assert_eq!(pool.borrow().inventory(), [(size as usize, 1)]);
        }
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.reuses - before.reuses, offsets.len() as u64 - 1);
        assert_eq!(after.allocated_bytes - before.allocated_bytes, size);
    });
}

#[test]
fn resource_suffix_partial_capture_is_never_sealed_or_retried() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let mut calls = 0;
        let before = snapshot(Class::Vertex);
        let result = fill_resource_suffix::<()>(
            device,
            64,
            48,
            Class::Vertex,
            "test_input_allocation",
            |suffix| {
                calls += 1;
                suffix[..3].fill(0x31);
                Ok(3)
            },
        );
        assert!(matches!(result, Err(FillError::Backend(_))));
        assert_eq!(calls, 1);
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(snapshot(Class::Vertex).live_bytes, before.live_bytes);
        for (size, offset) in [(0, 0), (8, 8), (8, 9)] {
            assert!(fill_resource_suffix::<()>(
                device,
                size,
                offset,
                Class::Vertex,
                "test_input_allocation",
                |_| panic!("invalid suffix must not expose a slice"),
            )
            .is_err());
        }
    });
}

#[test]
fn bounded_prefix_keeps_full_native_length_and_zeroes_recycled_tail() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        submit_lengths(device, &[64]);
        for (offset, captured) in [(0, 4), (16, 8), (12, 1), (3, 5), (4, 0)] {
            let input = fill_resource_prefix::<()>(
                device,
                64,
                offset,
                captured,
                Class::Fragment,
                "test_input_allocation",
                |view| {
                    assert_eq!(view.len(), captured);
                    view.fill(0xab);
                    Ok(view.len())
                },
            )
            .unwrap();
            let native_offset = input.binding_offset();
            assert_eq!(input.len() as u64, 64 - offset);
            assert_eq!(input.captured_len(), captured);
            let bytes = contents(&input.allocation.buffer);
            assert_eq!(
                input.allocation.buffer.length() - native_offset,
                64 - offset
            );
            assert!(bytes[..native_offset as usize]
                .iter()
                .all(|&byte| byte == 0));
            assert!(
                bytes[native_offset as usize..native_offset as usize + captured]
                    .iter()
                    .all(|&byte| byte == 0xab)
            );
            assert!(bytes[native_offset as usize + captured..]
                .iter()
                .all(|&byte| byte == 0));
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            submission.seal(input).unwrap();
            finish(&mut submission, &cmd);
        }
        for reported in [3, 5] {
            let result = fill_resource_prefix::<()>(
                device,
                64,
                16,
                4,
                Class::Fragment,
                "test_input_allocation",
                |view| {
                    assert_eq!(view.len(), 4);
                    view.fill(1);
                    Ok(reported)
                },
            );
            assert!(matches!(result, Err(FillError::Backend(_))));
        }
        assert!(fill_resource_prefix::<()>(
            device,
            64,
            16,
            49,
            Class::Fragment,
            "test_input_allocation",
            |_| panic!("an out-of-bounds capture cannot expose a slice"),
        )
        .is_err());
    });
}

#[test]
fn readonly_completed_inputs_reuse_known_zero_ranges_with_measured_savings() {
    use crate::backend::metal::buffer_extent::{self, Stage};
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        if !device.supports_family(metal::MTLGPUFamily::Apple2) {
            return;
        }
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        pool.borrow_mut().poison_fresh = true;
        let (_, proof) = buffer_extent::tests::object(Stage::Fragment, 0, 16)
            .capture_for(Class::Fragment, 0)
            .unwrap();
        let proof = proof.unwrap();
        const ALLOCATION: usize = 256 * 1024;
        const FILLS: usize = 64;
        let before = snapshot(Class::Fragment);
        let mut pointer = None;
        for index in 0..FILLS {
            let offset = if index < FILLS / 2 { 128 } else { 256 };
            let input = fill_read_only_resource_prefix::<()>(
                device,
                ALLOCATION as u64,
                offset,
                proof,
                Class::Fragment,
                "test_input_allocation",
                |view| {
                    assert_eq!(view.len(), 16);
                    view.fill(index as u8 + 1);
                    Ok(view.len())
                },
            )
            .unwrap();
            assert_eq!(input.binding_offset(), offset);
            assert_eq!(input.len(), ALLOCATION - offset as usize);
            assert_eq!(input.captured_len(), 16);
            let bytes = contents(&input.allocation.buffer);
            assert!(bytes[..offset as usize].iter().all(|&byte| byte == 0));
            assert_eq!(
                &bytes[offset as usize..offset as usize + 16],
                &[index as u8 + 1; 16]
            );
            assert!(bytes[offset as usize + 16..].iter().all(|&byte| byte == 0));
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            let buffer = submission.seal(input).unwrap();
            if let Some(pointer) = pointer {
                assert_eq!(buffer.as_ptr(), pointer);
            } else {
                pointer = Some(buffer.as_ptr());
            }
            assert!(pool.borrow().inventory().is_empty());
            assert!(submission.completed(&cmd).is_err());
            finish(&mut submission, &cmd);
            assert_eq!(pool.borrow().inventory(), [(ALLOCATION, 1)]);
        }
        let after = snapshot(Class::Fragment);
        let baseline_zeroed = ALLOCATION + (FILLS - 1) * (ALLOCATION - 16);
        assert_eq!(
            after.direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            (ALLOCATION + 16) as u64
        );
        assert_eq!(
            after.direct_fill_zero_reused_bytes - before.direct_fill_zero_reused_bytes,
            (baseline_zeroed - ALLOCATION - 16) as u64
        );
        assert_eq!(
            after.direct_fill_bytes - before.direct_fill_bytes,
            (FILLS * 16) as u64
        );
        assert_eq!(
            after.direct_fill_readonly - before.direct_fill_readonly,
            FILLS as u64
        );
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.live_bytes, before.live_bytes);
        let failed = fill_read_only_resource_prefix::<()>(
            device,
            ALLOCATION as u64,
            128,
            proof,
            Class::Fragment,
            "test_input_allocation",
            |view| {
                view[..3].fill(0x7f);
                Ok(3)
            },
        );
        assert!(matches!(failed, Err(FillError::Backend(_))));
        assert_eq!(snapshot(Class::Fragment).live_bytes, before.live_bytes);
        assert!(
            pool.borrow().inventory().is_empty(),
            "a failed readonly fill cannot recycle"
        );
    });
}

#[test]
fn writable_gpu_use_discards_zero_knowledge_before_the_next_capture() {
    use crate::backend::metal::buffer_extent::{self, Stage};
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let (_, proof) = buffer_extent::tests::object(Stage::Fragment, 0, 16)
            .capture_for(Class::Fragment, 0)
            .unwrap();
        let proof = proof.unwrap();
        let input = fill_read_only_resource_prefix::<()>(
            device,
            64,
            0,
            proof,
            Class::Fragment,
            "test_input_allocation",
            |view| {
                view.fill(0x31);
                Ok(view.len())
            },
        )
        .unwrap();
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        submission.seal(input).unwrap();
        finish(&mut submission, &cmd);

        // This input has no no-write proof. Even with a small captured prefix,
        // a later GPU use can modify any byte of the full native allocation.
        let input = fill_resource_prefix::<()>(
            device,
            64,
            0,
            4,
            Class::Fragment,
            "test_input_allocation",
            |view| {
                view.fill(0x42);
                Ok(view.len())
            },
        )
        .unwrap();
        let library = raw_metal::new_library_with_source(
            device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void dirty_tail(device uchar *data [[buffer(0)]]) {
                data[63] = 0x7f;
            }
        "#,
        )
        .unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(
                &library.get_function("dirty_tail", None).unwrap(),
            )
            .unwrap();
        let cmd = command(device);
        submission.begin(&cmd, pool.clone()).unwrap();
        let buffer = submission.seal(input).unwrap();
        let encoder = raw_metal::new_compute_command_encoder_with_dispatch_type(
            &cmd,
            metal::MTLDispatchType::Serial,
        )
        .unwrap();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&buffer), 0);
        encoder.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        finish(&mut submission, &cmd);
        assert_eq!(contents(&buffer)[63], 0x7f);

        let before = snapshot(Class::Fragment);
        let input = fill_read_only_resource_prefix::<()>(
            device,
            64,
            0,
            proof,
            Class::Fragment,
            "test_input_allocation",
            |view| {
                view.fill(0x53);
                Ok(view.len())
            },
        )
        .unwrap();
        assert!(contents(&input.allocation.buffer)[16..]
            .iter()
            .all(|&byte| byte == 0));
        assert_eq!(
            snapshot(Class::Fragment).direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            48
        );
        assert_eq!(
            snapshot(Class::Fragment).direct_fill_zero_reused_bytes,
            before.direct_fill_zero_reused_bytes
        );
        drop(input);
        assert!(
            pool.borrow().inventory().is_empty(),
            "unsealed reuse still cannot recycle"
        );
    });
}

#[test]
fn bounded_plain_input_skips_unreachable_pages_but_preserves_full_buffer_size() {
    use crate::backend::blob::BlobKey;
    use crate::backend::metal::buffer_extent::{self, Stage};
    use crate::runtime::draw::metal::inputs::{prepare, Capture, PreparedInput};
    use crate::runtime::host::HostMemory;
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let size = fixture.page + 32;
        let offset = fixture.page - 16;
        let bind = fixture.bind(7, 1, size, offset);
        let proof = buffer_extent::tests::object(Stage::Vertex, bind.index, 8);
        fixture
            .host
            .write_gpa(3 * fixture.page + 2 * 4, &0u32.to_le_bytes())
            .unwrap();
        let before = snapshot(Class::Vertex);
        let PreparedInput::Native(input) = prepare(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Class::Vertex,
            Capture::Native(Some(proof)),
            "draw_mtl_vertex_buffer_miss",
        )
        .unwrap() else {
            panic!("plain input must retain the native allocation");
        };
        let native_offset = input.bytes.binding_offset();
        assert_eq!(input.bytes.len(), 48);
        assert_eq!(input.bytes.captured_len(), 8);
        assert_eq!(input.bytes.allocation.buffer.length() - native_offset, 48);
        let bytes = contents(&input.bytes.allocation.buffer);
        assert!(bytes[..native_offset as usize]
            .iter()
            .all(|&byte| byte == 0));
        assert_eq!(
            &bytes[native_offset as usize..native_offset as usize + 8],
            &[0x11; 8]
        );
        assert!(bytes[native_offset as usize + 8..]
            .iter()
            .all(|&byte| byte == 0));
        assert_eq!(
            snapshot(Class::Vertex).direct_fill_bytes - before.direct_fill_bytes,
            8
        );
        drop(input);
        // Unknown or failed metadata must still attempt the full suffix, not
        // turn inaccessible trailing bytes into a successful short snapshot.
        let failed =
            buffer_extent::cached(BlobKey::new(b"invalid metadata fixture"), Stage::Vertex);
        assert!(prepare(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Class::Vertex,
            Capture::Native(failed.bound(bind.index)),
            "draw_mtl_vertex_buffer_miss",
        )
        .is_err());
        assert!(prepare(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Class::Vertex,
            Capture::Cpu,
            "draw_mtl_vertex_buffer_miss",
        )
        .is_err());
        assert!(
            pool.borrow().inventory().is_empty(),
            "unsealed inputs never recycle"
        );
    });
}

#[test]
fn readonly_snapshot_hits_keep_native_pixels_and_pending_input_lifetimes() {
    use crate::backend::metal::buffer_extent::{self, Stage};
    use crate::runtime::draw::metal::inputs::{prepare, Capture, PreparedInput};
    use crate::runtime::host::HostMemory;
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 bounded_vertex(uint i [[vertex_id]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        fragment float4 bounded_fragment(constant float4 &color [[buffer(0)]]) {
            return color;
        }
    "#;
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let library = raw_metal::new_library_with_source(device, SOURCE).unwrap();
        let descriptor = RenderPipelineDescriptor::new();
        descriptor
            .set_vertex_function(Some(&library.get_function("bounded_vertex", None).unwrap()));
        descriptor.set_fragment_function(Some(
            &library.get_function("bounded_fragment", None).unwrap(),
        ));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        let pipeline = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_width(3);
        descriptor.set_height(1);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = raw_metal::new_texture(device, &descriptor).unwrap();
        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 96, 32);
        let proof = buffer_extent::tests::object(Stage::Fragment, bind.index, 16);
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let pass = RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(&texture));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_store_action(MTLStoreAction::Store);
        let encoder = raw_metal::new_render_command_encoder(&cmd, pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline);
        encoder.set_viewport(MTLViewport {
            originX: 0.,
            originY: 0.,
            width: 3.,
            height: 1.,
            znear: 0.,
            zfar: 1.,
        });
        let before = snapshot(Class::Fragment);
        let mut buffers = Vec::new();
        for (x, color) in [[1f32, 0., 0., 1.], [1., 0., 0., 1.], [0., 1., 0., 1.]]
            .into_iter()
            .enumerate()
        {
            let bytes: Vec<_> = color.into_iter().flat_map(f32::to_ne_bytes).collect();
            if x != 1 {
                fixture
                    .host
                    .write_gpa(8 * fixture.page + 32, &bytes)
                    .unwrap();
                fixture.host.guest_wrote_page(8 * fixture.page);
            }
            let PreparedInput::Native(input) = prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Class::Fragment,
                Capture::Native(Some(proof)).in_scope(Some(&fixture.input_scope.reference())),
                "draw_mtl_fragment_buffer_miss",
            )
            .unwrap() else {
                panic!("plain fragment input must remain native");
            };
            let offset = input.bytes.binding_offset();
            assert_eq!(input.bytes.len(), 64);
            assert_eq!(input.bytes.captured_len(), 16);
            let buffer = submission.seal(input.bytes).unwrap();
            assert_eq!(buffer.length() - offset, 64);
            buffers.push(buffer.clone());
            encoder.set_fragment_buffer(0, Some(&buffer), offset);
            encoder.set_scissor_rect(MTLScissorRect {
                x: x as u64,
                y: 0,
                width: 1,
                height: 1,
            });
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        assert!(pool.borrow().inventory().is_empty());
        assert!(submission.inputs.is_empty());
        assert_eq!(
            submission.snapshots.len(),
            2,
            "one lease per physical snapshot"
        );
        assert_eq!(
            buffers[0].as_ptr(),
            buffers[1].as_ptr(),
            "clean binds share"
        );
        assert_ne!(
            buffers[0].as_ptr(),
            buffers[2].as_ptr(),
            "guest writes recapture"
        );
        fixture
            .host
            .write_gpa(8 * fixture.page + 32, &[0; 16])
            .unwrap();
        fixture.host.guest_wrote_page(8 * fixture.page);
        encoder.end_encoding();
        finish(&mut submission, &cmd);
        let mut pixels = [0u8; 12];
        texture.get_bytes(
            pixels.as_mut_ptr().cast(),
            12,
            MTLRegion::new_2d(0, 0, 3, 1),
            0,
        );
        assert_eq!(pixels, [255, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255]);
        assert_eq!(
            snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
            32
        );
        drop(bind);
        drop(fixture);
        assert_eq!(snapshot(Class::Fragment).live_bytes, before.live_bytes);
    });
}

#[test]
fn texture_participants_use_one_private_buffer_capture_through_encode_and_refusal() {
    use crate::backend::blob::BlobKey;
    use crate::backend::metal::abi::{
        ReimsVgpuPackedSampledImage, ReimsVgpuSampledImage, REIMS_VGPU_BINDING_TEXTURE_BASE,
    };
    use crate::backend::metal::buffer_extent::{self, Stage};
    use crate::backend::metal::render::{
        prepare_render_pipeline, render_core_mrt_inputs, ColorRt, ColorRtKey, ColorTarget,
        RenderBatch, RenderBuffers, RenderPipelineLayout,
    };
    use crate::runtime::draw::metal::inputs::{prepare, Capture, PreparedInput};
    use crate::runtime::host::HostMemory;
    const VERTEX: &[u8] = b"owned-textured-input-vertex";
    const FRAGMENT: &[u8] = b"owned-textured-input-fragment";
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 textured_input_vertex(uint i [[vertex_id]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        fragment float4 textured_input_fragment(float4 p [[position]],
            constant float4 &color [[buffer(0)]],
            texture2d<float, access::read> source [[texture(1)]],
            texture2d<float, access::write> destination [[texture(2)]]) {
            float4 value = color * source.read(uint2(p.xy));
            destination.write(value, uint2(p.xy));
            return value;
        }
    "#;
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().expect("Metal device");
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Fragment);
        let library = raw_metal::new_library_with_source(device, SOURCE).unwrap();
        for (key, name) in [
            (VERTEX, "textured_input_vertex"),
            (FRAGMENT, "textured_input_fragment"),
        ] {
            crate::backend::metal::cache::fn_cache_insert(
                &BlobKey::new(key),
                library.get_function(name, None).unwrap(),
            );
        }
        let keys = [ColorRtKey {
            slot: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm as u32,
            blend: None,
            write_mask: 0xf,
        }];
        let pipeline = prepare_render_pipeline(
            BlobKey::new(VERTEX),
            BlobKey::new(FRAGMENT),
            RenderPipelineLayout {
                attrs: &[],
                blend: None,
                colors: &keys,
                depth_format: 0,
                stencil_format: 0,
            },
        )
        .unwrap();
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_width(2);
        descriptor.set_height(1);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA32Float);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        let destination = raw_metal::new_texture(device, &descriptor).unwrap();
        let texels = [255u8, 255, 255, 255, 0, 255, 255, 255];
        let packed = ReimsVgpuPackedSampledImage {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 1,
            width: 2,
            height: 1,
            rgba8: texels.as_ptr(),
            len: texels.len(),
            pixel_format: 0,
            bytes_per_row: 8,
            data: texels.as_ptr(),
            data_len: texels.len(),
        };
        let mut fixture =
            crate::runtime::draw::buffer_read_tests::Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 96, 32);
        let proof = buffer_extent::tests::object(Stage::Fragment, bind.index, 16);
        let mut batch = RenderBatch::default();
        for invalid_texture in [false, true] {
            let values = if invalid_texture {
                [0f32, 1., 0., 0.5]
            } else {
                [1f32, 0.5, 0.25, 1.]
            };
            let color: Vec<_> = values.into_iter().flat_map(f32::to_ne_bytes).collect();
            fixture
                .host
                .write_gpa(8 * fixture.page + 32, &color)
                .unwrap();
            fixture.host.guest_wrote_page(8 * fixture.page);
            let PreparedInput::Native(input) = prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Class::Fragment,
                Capture::Native(Some(proof)).in_scope(Some(&fixture.input_scope.reference())),
                "draw_mtl_fragment_buffer_miss",
            )
            .unwrap() else {
                panic!("textured non-attribute input escaped its native owner");
            };
            assert_eq!(input.bytes.len(), 64);
            assert_eq!(input.bytes.captured_len(), 16);
            let captured = contents(&input.bytes.allocation.buffer);
            let offset = input.bytes.binding_offset() as usize;
            assert_eq!(
                &captured[offset..offset + 16],
                color.as_slice(),
                "each use captures fresh guest bytes"
            );
            fixture
                .host
                .write_gpa(8 * fixture.page + 32, &[0; 16])
                .unwrap();
            let images = [
                ReimsVgpuSampledImage::Packed(ReimsVgpuPackedSampledImage {
                    width: if invalid_texture { 0 } else { 2 },
                    ..packed
                }),
                ReimsVgpuSampledImage::Native {
                    binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 2,
                    texture: destination.clone(),
                },
            ];
            let mut output = [0xa5u8; 8];
            let mut target = ColorRt {
                slot: 0,
                pixel_format: MTLPixelFormat::RGBA8Unorm as u32,
                seed_rgba8: None,
                out_rgba8: Some(&mut output),
                clear_r: 0.,
                clear_g: 0.,
                clear_b: 0.,
                clear_a: 0.,
                load_action: MTLLoadAction::Clear as u32,
                blend: None,
                write_mask: 0xf,
                target: ColorTarget::Transient,
            };
            let status = render_core_mrt_inputs(
                &pipeline,
                2,
                1,
                crate::protocol::draw::DrawArgs {
                    vertex_count: 3,
                    instance_count: 1,
                    primitive_type: 3,
                    first_vertex: 0,
                    base_instance: 0,
                },
                None,
                None,
                RenderBuffers::Host(&[]),
                RenderBuffers::Native(vec![input]),
                &[],
                &[],
                &images,
                &[],
                &[],
                &[],
                None,
                None,
                None,
                None,
                None,
                None,
                std::slice::from_mut(&mut target),
                None,
                (std::ptr::null_mut(), 0),
                &mut batch,
                true,
            );
            if invalid_texture {
                assert!(!status.is_ok());
                assert!(
                    batch.pending(),
                    "late texture refusal still owns the sealed input"
                );
                assert!(
                    pool.borrow().inventory().is_empty(),
                    "refusal does not recycle pending storage"
                );
                batch.finish((std::ptr::null_mut(), 0)).unwrap();
                assert_eq!(output, [0xa5; 8]);
            } else {
                assert!(status.is_ok(), "{status:?}");
                assert!(
                    !batch.pending(),
                    "writable texture keeps its synchronous boundary"
                );
                assert_eq!(output, [255, 128, 64, 255, 0, 128, 64, 255]);
                let mut stored = [0f32; 8];
                destination.get_bytes(
                    stored.as_mut_ptr().cast(),
                    32,
                    MTLRegion::new_2d(0, 0, 2, 1),
                    0,
                );
                assert_eq!(stored, [1., 0.5, 0.25, 1., 0., 0.5, 0.25, 1.]);
            }
        }
        drop(bind);
        drop(fixture);
        let after = snapshot(Class::Fragment);
        assert_eq!(
            after.copies, before.copies,
            "no intermediate CPU-to-Metal copy"
        );
        assert_eq!(after.direct_fills - before.direct_fills, 2);
        assert_eq!(after.direct_fill_bytes - before.direct_fill_bytes, 32);
        assert_eq!(after.live_bytes, before.live_bytes);
    });
}

#[test]
fn readonly_snapshot_recycles_only_after_last_lease_and_verified_completion() {
    use crate::backend::metal::buffer_extent::{self, Stage};
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let proof = buffer_extent::tests::object(Stage::Fragment, 0, 16)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1
            .unwrap();
        let make = || {
            fill_read_only_resource_prefix::<()>(
                device,
                64,
                4,
                proof,
                Class::Fragment,
                "test_input_allocation",
                |bytes| {
                    bytes.fill(0x72);
                    Ok(bytes.len())
                },
            )
            .unwrap()
            .freeze()
            .unwrap()
        };
        let before = snapshot(Class::Fragment);
        let image = make();
        let weak = Arc::downgrade(&image);
        let first = command(device);
        let second = command(device);
        let mut a = Submission::default();
        let mut b = Submission::default();
        a.begin(&first, pool.clone()).unwrap();
        b.begin(&second, pool.clone()).unwrap();
        for _ in 0..8 {
            a.seal(image.bind(device, proof).unwrap()).unwrap();
        }
        b.seal(image.bind(device, proof).unwrap()).unwrap();
        assert_eq!(a.snapshots.len(), 1);
        assert!(a.completed(&second).is_err());
        assert!(a.completed(&first).is_err());
        finish(&mut a, &first);
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(
            snapshot(Class::Fragment).owned_bytes - before.owned_bytes,
            64
        );
        image.retire(device);
        assert!(
            weak.upgrade().is_some(),
            "second command still holds storage"
        );
        assert!(pool.borrow().inventory().is_empty());
        finish(&mut b, &second);
        assert!(weak.upgrade().is_none());
        assert_eq!(pool.borrow().inventory(), [(64, 1)]);
        assert_eq!(
            pool.borrow().peak.count,
            1,
            "duplicate bindings do not inflate budget"
        );
        assert_eq!(pool.borrow().peak.bytes, 64);

        let image = make();
        let cmd = command(device);
        let mut abandoned = Submission::default();
        abandoned.begin(&cmd, pool.clone()).unwrap();
        abandoned.seal(image.bind(device, proof).unwrap()).unwrap();
        abandoned.discard();
        cmd.commit();
        cmd.wait_until_completed();
        image.retire(device);
        assert!(
            pool.borrow().inventory().is_empty(),
            "lost completion cannot recycle"
        );
        assert_eq!(snapshot(Class::Fragment).live_bytes, before.live_bytes);
    });
}

#[test]
fn completed_inventory_is_not_transferred_between_threads() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let parent = submission.seal(filled(device, &[0x12; 23])).unwrap();
        finish(&mut submission, &cmd);
        let retained = snapshot(Class::Vertex).retained_bytes;
        let child_id = std::thread::spawn(|| {
            objc::rc::autoreleasepool(|| {
                let device = runtime::system_device().unwrap();
                let pool = runtime::thread_input_pool(device);
                assert!(pool.borrow().inventory().is_empty());
                let cmd = command(device);
                let mut submission = Submission::default();
                submission.begin(&cmd, pool.clone()).unwrap();
                let child = submission.seal(filled(device, &[0x34; 23])).unwrap();
                finish(&mut submission, &cmd);
                assert_eq!(pool.borrow().inventory(), [(23, 1)]);
                child.as_ptr() as usize
            })
        })
        .join()
        .unwrap();
        assert_ne!(parent.as_ptr() as usize, child_id);
        assert_eq!(contents(&parent), [0x12; 23]);
        assert_eq!(pool.borrow().inventory(), [(23, 1)]);
        assert_eq!(
            snapshot(Class::Vertex).retained_bytes,
            retained,
            "child TLS releases idle inputs"
        );
    });
}

#[test]
fn byte_and_count_bounds_evict_oldest_completed_inputs_independently() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let old = submit_lengths(device, &[17, 17]);
        let before = snapshot(Class::Vertex);
        submit_lengths(device, &[19]);
        assert_eq!(pool.borrow().peak.count, 2);
        assert_eq!(pool.borrow().peak.bytes, 34);
        assert_eq!(pool.borrow().inventory(), [(19, 1)]);
        let after = snapshot(Class::Vertex);
        assert_eq!(after.budget_discards - before.budget_discards, 2);
        assert_eq!(after.budget_discard_bytes - before.budget_discard_bytes, 34);
        assert_eq!(
            contents(&old[0]),
            [1; 17],
            "external native references remain valid"
        );
        assert_index(&pool.borrow());

        pool.borrow_mut().clear_available();
        submit_lengths(device, &[32]);
        submit_lengths(device, &[8]);
        let before = snapshot(Class::Vertex);
        assert!(pool.borrow().available_bytes + 9 < pool.borrow().peak.bytes);
        submit_lengths(device, &[9]);
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (1, 32)
        );
        assert_eq!(pool.borrow().inventory(), [(9, 1)]);
        let after = snapshot(Class::Vertex);
        assert_eq!(after.budget_discards - before.budget_discards, 1);
        assert_eq!(after.budget_discard_bytes - before.budget_discard_bytes, 8);
        assert_index(&pool.borrow());
    });
}

#[test]
fn eviction_order_follows_completion_returns_not_allocation_birth() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let initial = submit_lengths(device, &[8, 12]);
        let returned = submit_lengths(device, &[8]);
        assert_eq!(returned[0].as_ptr(), initial[0].as_ptr());
        assert_index(&pool.borrow());
        submit_lengths(device, &[4]);
        assert_eq!(pool.borrow().inventory(), [(4, 1), (8, 1)]);
        let pool = pool.borrow();
        assert_eq!(pool.node(pool.oldest.unwrap()).input.0.account.len, 8);
        assert_index(&pool);
    });
}

#[test]
fn request_categories_and_bounds_do_not_confuse_pending_with_completed_demand() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Vertex);
        submit_lengths(device, &[17, 17]);
        let warm = snapshot(Class::Vertex);
        assert_eq!(warm.requests - before.requests, 2);
        assert_eq!(warm.miss_absent_length - before.miss_absent_length, 2);
        let cmd = command(device);
        let mut pending = Submission::default();
        pending.begin(&cmd, pool.clone()).unwrap();
        let held: Vec<_> = (0..3)
            .map(|i| pending.seal(filled(device, &vec![i; 17])).unwrap())
            .collect();
        let queued = snapshot(Class::Vertex);
        assert_eq!(queued.requests - warm.requests, 3);
        assert_eq!(queued.reuses - warm.reuses, 2);
        assert_eq!(queued.miss_exhausted_length - warm.miss_exhausted_length, 1);
        assert_eq!(queued.miss_absent_length - warm.miss_absent_length, 0);
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (2, 34)
        );
        assert_eq!(
            pending.completed(&cmd).unwrap_err().refusal(),
            Some("metal_render_input_not_completed")
        );
        assert!(pool.borrow().inventory().is_empty());
        assert_index(&pool.borrow());
        // Another actual completion may populate/trim AVAILABLE resources but
        // must not reclaim the three inputs of the still-unsubmitted command.
        submit_lengths(device, &[19]);
        assert_eq!(pool.borrow().inventory(), [(19, 1)]);
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (2, 34)
        );
        for (i, buffer) in held.iter().enumerate() {
            assert_eq!(contents(buffer), vec![i as u8; 17]);
        }
        finish(&mut pending, &cmd);
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (3, 51)
        );
        assert_eq!(pool.borrow().inventory(), [(17, 3)]);
        assert_index(&pool.borrow());
    });
}

#[test]
fn overlapping_source_is_skipped_without_losing_other_exact_length_matches() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let originals = submit_lengths(device, &[17, 17]);
        let before = snapshot(Class::Vertex);
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let source = &originals[1];
        let input = unsafe {
            copy(
                device,
                source.contents().cast(),
                17,
                Class::Vertex,
                "test_input_allocation",
            )
        }
        .unwrap();
        let snapshot_buffer = submission.seal(input).unwrap();
        assert_eq!(snapshot_buffer.as_ptr(), originals[0].as_ptr());
        unsafe {
            std::ptr::write_bytes(source.contents().cast::<u8>(), 0xee, 17);
        }
        assert_eq!(contents(&snapshot_buffer), [2; 17]);
        assert_index(&pool.borrow());
        finish(&mut submission, &cmd);
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations - before.allocations, 0);
        assert_eq!(after.reuses - before.reuses, 1);
        assert_eq!(after.miss_source_overlap - before.miss_source_overlap, 0);
        assert_index(&pool.borrow());
    });
}

#[test]
fn changing_lengths_do_not_accumulate_index_history_or_vacant_slots() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        for iteration in 0..80 {
            let len = 8 + iteration % 13;
            submit_lengths(device, &[len, len + 1, len + 2]);
            let pool = pool.borrow();
            assert_index(&pool);
            assert_eq!(pool.peak.count, 3);
            assert_eq!(pool.slots.len(), 3);
            assert!(pool.exhausted_lengths.is_empty());
        }
    });
}

#[test]
fn large_small_empty_large_reuses_completed_inputs_and_preserves_native_pixels() {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 pool_vertex(uint i [[vertex_id]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        fragment float4 pool_fragment(constant float4 &color [[buffer(0)]]) {
            return color;
        }
    "#;
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let library = raw_metal::new_library_with_source(device, SOURCE).unwrap();
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("pool_vertex", None).unwrap()));
        descriptor
            .set_fragment_function(Some(&library.get_function("pool_fragment", None).unwrap()));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        let pipeline = raw_metal::new_render_pipeline_state(device, &descriptor).unwrap();
        let texture_descriptor = TextureDescriptor::new();
        texture_descriptor.set_texture_type(MTLTextureType::D2);
        texture_descriptor.set_width(3);
        texture_descriptor.set_height(1);
        texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        texture_descriptor.set_storage_mode(MTLStorageMode::Shared);
        texture_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = raw_metal::new_texture(device, &texture_descriptor).unwrap();
        let frames: [&[[f32; 4]]; 4] = [
            &[[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]],
            &[[0.25, 0.5, 0.75, 1.]],
            &[],
            &[[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 0., 1.]],
        ];
        let before = snapshot(Class::Fragment);
        for (frame, colors) in frames.into_iter().enumerate() {
            let cmd = command(device);
            let mut submission = Submission::default();
            submission.begin(&cmd, pool.clone()).unwrap();
            let pass = RenderPassDescriptor::new();
            let attachment = pass.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(&texture));
            attachment.set_load_action(MTLLoadAction::Clear);
            attachment.set_store_action(MTLStoreAction::Store);
            attachment.set_clear_color(MTLClearColor::new(0., 0., 0., 1.));
            let encoder = (!colors.is_empty()).then(|| {
                let encoder = raw_metal::new_render_command_encoder(&cmd, &pass).unwrap();
                encoder.set_render_pipeline_state(&pipeline);
                encoder.set_viewport(MTLViewport {
                    originX: 0.,
                    originY: 0.,
                    width: 3.,
                    height: 1.,
                    znear: 0.,
                    zfar: 1.,
                });
                encoder
            });
            let mut held = Vec::new();
            for (x, color) in colors.iter().enumerate() {
                let mut bytes = [0u8; 16];
                for (chunk, value) in bytes.chunks_exact_mut(4).zip(color) {
                    chunk.copy_from_slice(&value.to_ne_bytes());
                }
                let expected = bytes;
                let input = if frame == 3 {
                    fill::<()>(
                        device,
                        bytes.len(),
                        Class::Fragment,
                        "test_input_allocation",
                        |view| {
                            view.copy_from_slice(&bytes);
                            Ok(view.len())
                        },
                    )
                    .unwrap()
                } else {
                    unsafe {
                        copy(
                            device,
                            bytes.as_ptr(),
                            bytes.len(),
                            Class::Fragment,
                            "test_input_allocation",
                        )
                    }
                    .unwrap()
                };
                let buffer = submission.seal(input).unwrap();
                bytes.fill(0);
                assert_eq!(buffer.length(), 16);
                assert_eq!(contents(&buffer), expected);
                let encoder = encoder.unwrap();
                encoder.set_fragment_buffer(0, Some(&buffer), 0);
                encoder.set_scissor_rect(MTLScissorRect {
                    x: x as u64,
                    y: 0,
                    width: 1,
                    height: 1,
                });
                encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
                held.push(buffer);
            }
            assert_eq!(
                held.iter()
                    .map(|buffer| buffer.as_ptr() as usize)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                colors.len(),
                "unsubmitted inputs must have independent storage",
            );
            assert_eq!(pool.borrow().peak.count, if frame == 0 { 0 } else { 3 });
            if let Some(encoder) = encoder {
                encoder.end_encoding();
            }
            finish(&mut submission, &cmd);
            assert_eq!(
                (pool.borrow().peak.count, pool.borrow().peak.bytes),
                (3, 48)
            );
            assert_eq!(pool.borrow().inventory(), [(16, 3)]);
            assert_index(&pool.borrow());
            assert_eq!(
                snapshot(Class::Fragment).allocations - before.allocations,
                3
            );
            if frame == 0 || frame == 3 {
                let mut pixels = [0u8; 12];
                texture.get_bytes(
                    pixels.as_mut_ptr().cast(),
                    12,
                    MTLRegion::new_2d(0, 0, 3, 1),
                    0,
                );
                assert_eq!(
                    pixels,
                    if frame == 0 {
                        [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]
                    } else {
                        [0, 255, 255, 255, 255, 0, 255, 255, 255, 255, 0, 255]
                    }
                );
            }
        }
        let after = snapshot(Class::Fragment);
        assert_eq!(after.allocations - before.allocations, 3);
        assert_eq!(after.allocated_bytes - before.allocated_bytes, 48);
        assert_eq!(after.requests - before.requests, 7);
        assert_eq!(after.copies - before.copies, 4);
        assert_eq!(after.copied_bytes - before.copied_bytes, 64);
        assert_eq!(after.direct_fills - before.direct_fills, 3);
        assert_eq!(after.direct_fill_bytes - before.direct_fill_bytes, 48);
        assert_eq!(
            after.direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            0
        );
        assert_eq!(after.reuses - before.reuses, 4);
        assert_eq!(after.budget_discards - before.budget_discards, 0);
        assert_eq!(after.live_bytes, before.live_bytes);
    });
}

#[test]
fn retirement_allocation_failure_is_terminal_after_actual_gpu_completion() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        submit_lengths(device, &[17]);
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let buffers = [
            submission.seal(filled(device, &[3; 28])).unwrap(),
            submission.seal(filled(device, &[4; 36])).unwrap(),
        ];
        let completed_before = POOLS.completed_submissions.load(Relaxed);
        pool.borrow_mut().fail_inventory = true;
        cmd.commit();
        cmd.wait_until_completed();
        assert_eq!(
            submission.completed(&cmd).unwrap_err().refusal(),
            Some("metal_render_input_inventory_alloc_failed")
        );
        assert_eq!(
            POOLS.completed_submissions.load(Relaxed),
            completed_before + 1
        );
        assert!(submission.inputs.is_empty());
        assert!(submission.owner.is_none());
        assert!(submission.command.is_none());
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(
            (pool.borrow().peak.count, pool.borrow().peak.bytes),
            (2, 64)
        );
        assert_index(&pool.borrow());
        assert_eq!(contents(&buffers[0]), [3; 28]);
        assert_eq!(contents(&buffers[1]), [4; 36]);
        assert_eq!(
            submission.completed(&cmd).unwrap_err().refusal(),
            Some("metal_render_input_completion_mismatch")
        );
        assert_eq!(
            POOLS.completed_submissions.load(Relaxed),
            completed_before + 1
        );
    });
}

#[test]
fn direct_fill_initializes_fresh_views_and_reuses_only_initialized_completed_storage() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        pool.borrow_mut().poison_fresh = true;
        let before = snapshot(Class::Vertex);
        let input = fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            assert_eq!(
                view, &[0u8; 17],
                "fresh native poison must be initialized before exposure"
            );
            view.fill(0x31);
            Ok(view.len())
        })
        .unwrap();
        assert_eq!(input.len(), 17);
        assert!(pool.borrow().inventory().is_empty());
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        let first = submission.seal(input).unwrap();
        finish(&mut submission, &cmd);

        let input = fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            assert_eq!(
                view, &[0x31u8; 17],
                "recycled initialized storage is not zeroed again"
            );
            view.fill(0x52);
            Ok(view.len())
        })
        .unwrap();
        assert_eq!(contents(&input.allocation.buffer), [0x52; 17]);
        let next = command(device);
        submission.begin(&next, pool.clone()).unwrap();
        let second = submission.seal(input).unwrap();
        assert_eq!(first.as_ptr(), second.as_ptr());
        finish(&mut submission, &next);
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.reuses - before.reuses, 1);
        assert_eq!(after.direct_fill_requests - before.direct_fill_requests, 2);
        assert_eq!(after.direct_fills - before.direct_fills, 2);
        assert_eq!(after.direct_fill_bytes - before.direct_fill_bytes, 34);
        assert_eq!(
            after.direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            17
        );
        assert_eq!(after.direct_fill_failures - before.direct_fill_failures, 0);
        assert_eq!(after.copies, before.copies);
        assert_eq!(after.copied_bytes, before.copied_bytes);
        assert_index(&pool.borrow());
    });
}

#[test]
fn direct_fill_failures_never_produce_or_recycle_partially_filled_inputs() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Vertex);
        match fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            view[..3].fill(0x41);
            Ok(3)
        }) {
            Err(FillError::Backend(status)) => {
                assert_eq!(status.refusal(), Some("metal_render_input_fill_incomplete"));
            }
            _ => panic!("short read must not produce Filled"),
        }
        match fill(device, 17, Class::Vertex, "test_input_allocation", |view| {
            view[..4].fill(0x42);
            Err::<usize, _>("guest_read_failed")
        }) {
            Err(FillError::Callback("guest_read_failed")) => {}
            _ => panic!("callback error must be preserved without producing Filled"),
        }
        match fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            view.fill(0x43);
            Ok(view.len() + 1)
        }) {
            Err(FillError::Backend(status)) => {
                assert_eq!(status.refusal(), Some("metal_render_input_fill_incomplete"));
            }
            _ => panic!("overreported read must not produce Filled"),
        }
        pool.borrow_mut().fail_next_allocation();
        match fill::<()>(
            device,
            17,
            Class::Vertex,
            "test_direct_alloc_failed",
            |_| {
                panic!("allocation failure must not expose a CPU view");
            },
        ) {
            Err(FillError::Backend(status)) => {
                assert_eq!(status.refusal(), Some("test_direct_alloc_failed"));
            }
            _ => panic!("native allocation failure must remain typed"),
        }
        let after = snapshot(Class::Vertex);
        assert_eq!(after.direct_fill_requests - before.direct_fill_requests, 4);
        assert_eq!(after.direct_fill_failures - before.direct_fill_failures, 4);
        assert_eq!(
            after.direct_fill_partial_bytes - before.direct_fill_partial_bytes,
            3
        );
        assert_eq!(
            after.direct_fill_zeroed_bytes - before.direct_fill_zeroed_bytes,
            51
        );
        assert_eq!(after.direct_fills, before.direct_fills);
        assert_eq!(after.direct_fill_bytes, before.direct_fill_bytes);
        assert_eq!(after.copies, before.copies);
        assert_eq!(after.copied_bytes, before.copied_bytes);
        assert_eq!(after.live_bytes, before.live_bytes);
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(pool.borrow().peak.count, 0);
        assert_index(&pool.borrow());
    });
}

#[test]
fn direct_fill_callback_can_reenter_tls_pool_and_dependency_completion() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        submit_lengths(device, &[17, 17]);
        let outer = fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            let same_pool = runtime::thread_input_pool(device);
            assert!(Rc::ptr_eq(&pool, &same_pool), "TLS access is reentrant");
            drop(
                pool.try_borrow_mut()
                    .expect("no pool borrow may cover the callback"),
            );
            view.fill(0x61);
            let inner = fill::<()>(
                device,
                17,
                Class::Vertex,
                "test_input_allocation",
                |nested| {
                    assert_ne!(
                        nested.as_ptr(),
                        view.as_ptr(),
                        "the outer filling lease is unavailable"
                    );
                    nested.fill(0x72);
                    Ok(nested.len())
                },
            )
            .unwrap();
            let cmd = command(device);
            let mut dependency = Submission::default();
            dependency.begin(&cmd, pool.clone()).unwrap();
            dependency.seal(inner).unwrap();
            finish(&mut dependency, &cmd);
            assert_eq!(view, &[0x61u8; 17]);
            assert_eq!(pool.borrow().inventory(), [(17, 1)]);
            Ok(view.len())
        })
        .unwrap();
        assert_eq!(contents(&outer.allocation.buffer), [0x61; 17]);
        assert_eq!(pool.borrow().inventory(), [(17, 1)]);
        let cmd = command(device);
        let mut submission = Submission::default();
        submission.begin(&cmd, pool.clone()).unwrap();
        submission.seal(outer).unwrap();
        finish(&mut submission, &cmd);
        assert_eq!(pool.borrow().inventory(), [(17, 2)]);
        assert_index(&pool.borrow());
    });
}

#[test]
fn unsealed_direct_fill_is_not_a_completed_inventory_entry() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = snapshot(Class::Vertex);
        let input = fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            view.fill(0x81);
            Ok(view.len())
        })
        .unwrap();
        assert_eq!(input.len(), 17);
        assert_eq!(snapshot(Class::Vertex).live_bytes - before.live_bytes, 17);
        drop(input);
        assert_eq!(snapshot(Class::Vertex).live_bytes, before.live_bytes);
        assert_eq!(
            snapshot(Class::Vertex).completed_inputs,
            before.completed_inputs
        );
        assert!(pool.borrow().inventory().is_empty());
        assert_eq!(pool.borrow().peak.count, 0);
    });
}

#[test]
fn failed_recycled_direct_fill_is_discarded_without_sealing_or_reinitializing() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = runtime::system_device() else {
            return;
        };
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        submit_lengths(device, &[17]);
        let before = snapshot(Class::Vertex);
        match fill::<()>(device, 17, Class::Vertex, "test_input_allocation", |view| {
            assert_eq!(view, &[1u8; 17]);
            view[..2].fill(0xcc);
            Ok(2)
        }) {
            Err(FillError::Backend(status)) => {
                assert_eq!(status.refusal(), Some("metal_render_input_fill_incomplete"));
            }
            _ => panic!("partial overwrite must not produce Filled"),
        }
        let after = snapshot(Class::Vertex);
        assert_eq!(after.allocations, before.allocations);
        assert_eq!(after.reuses - before.reuses, 1);
        assert_eq!(after.direct_fill_failures - before.direct_fill_failures, 1);
        assert_eq!(
            after.direct_fill_partial_bytes - before.direct_fill_partial_bytes,
            2
        );
        assert_eq!(
            after.direct_fill_zeroed_bytes,
            before.direct_fill_zeroed_bytes
        );
        assert_eq!(after.direct_fills, before.direct_fills);
        assert_eq!(after.completed_inputs, before.completed_inputs);
        assert_eq!(after.live_bytes, before.live_bytes);
        assert!(pool.borrow().inventory().is_empty());
        assert_index(&pool.borrow());
    });
}
