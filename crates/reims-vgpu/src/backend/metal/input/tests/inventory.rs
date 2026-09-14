use super::*;
use crate::backend::metal::buffer_extent::{self, ReadOnlyCapture, Stage};

fn proof() -> ReadOnlyCapture {
    buffer_extent::tests::object(Stage::Fragment, 0, 16)
        .capture_for(Class::Fragment, 0)
        .unwrap()
        .1
        .unwrap()
}

fn image(device: &Device, len: usize, proof: ReadOnlyCapture, value: u8) -> Arc<ReadOnlySnapshot> {
    fill_read_only_resource_prefix::<()>(
        device,
        len as u64,
        0,
        proof,
        Class::Fragment,
        "test_input_allocation",
        |bytes| {
            bytes.fill(value);
            Ok(bytes.len())
        },
    )
    .unwrap()
    .freeze()
    .unwrap()
}

fn complete_image(device: &Device, image: &Arc<ReadOnlySnapshot>, proof: ReadOnlyCapture) {
    let cmd = command(device);
    let mut submission = Submission::default();
    submission
        .begin(&cmd, runtime::thread_input_pool(device))
        .unwrap();
    submission.seal(image.bind(device, proof).unwrap()).unwrap();
    finish(&mut submission, &cmd);
}

#[test]
fn readonly_snapshot_return_then_reexhaust_has_no_duplicate_tombstone() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let proof = proof();
        let cached = image(device, 32, proof, 0x31);
        complete_image(device, &cached, proof);
        submit_lengths(device, &[32]);
        let first = filled(device, &[0x42; 32]);
        assert_eq!(pool.borrow().exhausted_lengths, [32]);
        cached.retire(device);
        assert_index(&pool.borrow());
        let second = filled(device, &[0x53; 32]);
        // Catch so the reproducer can drain the old malformed tombstone list
        // before asserting, rather than panicking again in the TLS destructor.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.borrow_mut().forget_exhausted();
        }));
        assert!(
            result.is_ok(),
            "an exhausted bucket must be named exactly once"
        );
        assert_index(&pool.borrow());
        assert_eq!(first.test_contents(), [0x42; 32]);
        assert_eq!(second.test_contents(), [0x53; 32]);
        drop((first, second));
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn readonly_snapshot_return_eviction_and_clear_preserve_live_buckets_and_budgets() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        let proof = proof();
        for byte_budget in [false, true] {
            pool.borrow_mut().clear_available();
            let first = image(device, 32, proof, 0x31);
            let second = image(device, 40, proof, 0x52);
            complete_image(device, &first, proof);
            complete_image(device, &second, proof);
            if byte_budget {
                submit_lengths(device, &[32, 16]);
                assert_eq!(
                    (pool.borrow().peak.count, pool.borrow().peak.bytes),
                    (2, 48)
                );
            } else {
                submit_lengths(device, &[32]);
                assert_eq!(
                    (pool.borrow().peak.count, pool.borrow().peak.bytes),
                    (1, 40)
                );
            }
            let held = filled(device, &[0x63; 32]);
            assert_eq!(pool.borrow().exhausted_lengths, [32]);
            first.retire(device);
            assert_index(&pool.borrow());
            second.retire(device);
            assert_index(&pool.borrow());
            assert_eq!(pool.borrow().inventory(), [(40, 1)]);
            pool.borrow_mut().forget_exhausted();
            assert_eq!(pool.borrow().inventory(), [(40, 1)]);
            assert_eq!(held.test_contents(), [0x63; 32]);
            pool.borrow_mut().clear();
            assert_index(&pool.borrow());
            assert!(pool.borrow().by_len.is_empty());
            drop(held);
        }
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn readonly_snapshot_return_new_length_closes_depletion_without_growing_metadata() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let proof = proof();
        let late = image(device, 24, proof, 0x31);
        complete_image(device, &late, proof);
        submit_lengths(device, &[32]);
        let held = filled(device, &[0x42; 32]);
        assert_eq!(pool.borrow().exhausted_lengths, [32]);
        late.retire(device);
        assert_index(&pool.borrow());
        assert_eq!(pool.borrow().by_len.len(), 1);
        assert_eq!(pool.borrow().inventory(), [(24, 1)]);
        assert!(pool.borrow().exhausted_lengths.is_empty());
        drop(held);
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn readonly_snapshot_return_late_lease_and_repeated_revival_keep_one_tombstone() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let proof = proof();
        let images: Vec<_> = (0..16)
            .map(|value| image(device, 32, proof, value))
            .collect();
        for image in &images {
            complete_image(device, image, proof);
        }
        submit_lengths(device, &[32]);
        let mut held = Vec::new();
        for image in images {
            held.push(filled(device, &[0x79; 32]));
            assert_eq!(pool.borrow().exhausted_lengths, [32]);
            let lease = Arc::clone(&image);
            image.retire(device);
            assert_eq!(pool.borrow().exhausted_lengths, [32]);
            assert!(
                pool.borrow().inventory().is_empty(),
                "a late lease still owns storage"
            );
            lease.retire(device);
            assert_index(&pool.borrow());
            assert!(pool.borrow().exhausted_lengths.is_empty());
            assert_eq!(pool.borrow().inventory(), [(32, 1)]);
            assert_eq!(pool.borrow().peak.count, 1);
        }
        for input in &held {
            assert_eq!(input.test_contents(), [0x79; 32]);
        }
        drop(held);
        pool.borrow_mut().forget_exhausted();
        assert_index(&pool.borrow());
        pool.borrow_mut().clear_available();
    });
}

#[test]
fn readonly_snapshot_return_inventory_stress_preserves_indices_bytes_and_lifetimes() {
    objc::rc::autoreleasepool(|| {
        let device = runtime::system_device().unwrap();
        let pool = runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before_vertex = snapshot(Class::Vertex);
        let before_fragment = snapshot(Class::Fragment);
        let proof = proof();
        let lengths = [16, 24, 32, 40, 48, 64];
        for seed in 1u64..=8 {
            let mut random = seed;
            let mut captured = Vec::new();
            let mut cached: Vec<(Arc<ReadOnlySnapshot>, u8)> = Vec::new();
            let mut late: Vec<(Arc<ReadOnlySnapshot>, u8)> = Vec::new();
            for step in 0..512 {
                random = random.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                let index = (random >> 32) as usize;
                let len = lengths[index % lengths.len()];
                match index % 8 {
                    0 if captured.len() < 12 => {
                        captured.push(filled(device, &vec![step as u8; len]));
                    }
                    1 if cached.len() < 12 => {
                        cached.push((image(device, len, proof, step as u8), step as u8));
                    }
                    2 => {
                        let cmd = command(device);
                        let mut submission = Submission::default();
                        submission.begin(&cmd, pool.clone()).unwrap();
                        for _ in 0..index % 3 + 1 {
                            if let Some(input) = captured.pop() {
                                submission.seal(input).unwrap();
                            }
                        }
                        for (image, _) in cached.iter().chain(&late).take(index % 5) {
                            submission.seal(image.bind(device, proof).unwrap()).unwrap();
                        }
                        finish(&mut submission, &cmd);
                    }
                    3 if !cached.is_empty() => {
                        let (image, _) = cached.swap_remove(index % cached.len());
                        image.retire(device);
                    }
                    4 if !cached.is_empty() && late.len() < 12 => {
                        let (image, value) = &cached[index % cached.len()];
                        late.push((Arc::clone(image), *value));
                    }
                    5 if !late.is_empty() => {
                        let (image, _) = late.swap_remove(index % late.len());
                        image.retire(device);
                    }
                    6 => pool.borrow_mut().clear(),
                    7 => pool.borrow_mut().forget_exhausted(),
                    _ => {}
                }
                assert_index(&pool.borrow());
                for (image, value) in cached.iter().chain(&late) {
                    assert_eq!(&contents(&image.allocation.buffer)[..16], &[*value; 16]);
                }
            }
            drop(captured);
            for (image, _) in cached.into_iter().chain(late) {
                image.retire(device);
            }
            pool.borrow_mut().clear_available();
            assert_eq!(
                snapshot(Class::Vertex).owned_bytes,
                before_vertex.owned_bytes
            );
            assert_eq!(
                snapshot(Class::Fragment).owned_bytes,
                before_fragment.owned_bytes
            );
        }
    });
}
