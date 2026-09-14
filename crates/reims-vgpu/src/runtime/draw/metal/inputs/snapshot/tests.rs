use super::*;
use crate::backend::metal::buffer_extent::{self, BufferRead, Stage};
use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::draw::buffer_read_tests::Fixture;

mod contained;
mod diagnostics;

fn prepare(fixture: &mut Fixture, bind: &BufferBind, proof: Option<BufferRead>) -> Filled {
    let PreparedInput::Native(input) = super::super::prepare(
        &mut fixture.state,
        &mut fixture.host,
        1,
        bind,
        Class::Fragment,
        Capture::ScopedNative(proof, fixture.input_scope.reference()),
        "test_input_read",
    )
    .unwrap() else {
        panic!("native capture")
    };
    input.bytes
}

fn retained(bind: &BufferBind) -> bool {
    bind.resource
        .as_ref()
        .unwrap()
        .with_rail_state(|held: &mut RetainedBuffer| held.latest.is_some())
        .unwrap()
}

#[test]
fn readonly_snapshot_scope_change_or_expiry_requires_fresh_bytes_without_a_harvest() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let old_scope = fixture.input_scope.reference();
        let original = prepare(&mut fixture, &bind, proof);
        let before = input::snapshot(Class::Fragment);
        fixture.input_scope = crate::runtime::draw::BufferSnapshotScope::new();
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x42; 16])
            .unwrap();
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x42; 16]);
        assert_eq!(old_scope.current(), None);
        // An expired request cannot resurrect its previous scope or use the
        // resource's newer candidate, even when every write witness is quiet.
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x53; 16])
            .unwrap();
        let PreparedInput::Native(expired) = super::super::prepare(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Class::Fragment,
            Capture::ScopedNative(proof, old_scope),
            "test_expired_scope",
        )
        .unwrap() else {
            panic!("native capture")
        };
        assert_eq!(captured(&expired.bytes), [0x53; 16]);
        assert_eq!(captured(&original), [0x11; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
    });
}

#[test]
fn readonly_snapshot_unscoped_native_calls_always_read_guest_bytes() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let before = input::snapshot(Class::Fragment);
        for value in [0x42, 0x53] {
            fixture
                .host
                .write_gpa(8 * fixture.page + 4, &[value; 16])
                .unwrap();
            let PreparedInput::Native(input) = super::super::prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Class::Fragment,
                Capture::Native(proof),
                "test_unscoped_capture",
            )
            .unwrap() else {
                panic!("native capture")
            };
            assert_eq!(captured(&input.bytes), [value; 16]);
        }
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
        assert!(!retained(&bind));
    });
}

#[test]
fn readonly_snapshot_consumes_explicit_buffer_invalidations_even_before_harvest() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let original = prepare(&mut fixture, &bind, proof);
        let before = input::snapshot(Class::Fragment);
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x64; 16])
            .unwrap();
        let outcome = crate::runtime::resource_validity::apply(
            &mut fixture.state,
            1,
            7,
            crate::protocol::fifo::InvalidateValidityOps {
                clear_host_valid: 1,
                set_guest_valid: 1,
                ..Default::default()
            },
            crate::runtime::resource_validity::ValiditySite::ExecTable,
        );
        assert!(
            outcome.missed,
            "the buffer has no IOSurface mapping generation"
        );
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x64; 16]);
        assert_eq!(captured(&original), [0x11; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            1
        );
    });
}

#[test]
fn readonly_snapshot_scope_ending_during_capture_cannot_retain_a_candidate() {
    objc::rc::autoreleasepool(|| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = buffer_extent::tests::object(Stage::Fragment, 0, 16)
            .capture_for(Class::Fragment, 0)
            .unwrap()
            .1;
        let span = prepare_bound_buffer_read_with_extent(
            &mut fixture.state,
            &mut fixture.host,
            1,
            &bind,
            Some(16),
        )
        .unwrap()
        .span;
        let mut owner = Some(crate::runtime::draw::BufferSnapshotScope::new());
        let scope = owner.as_ref().unwrap().reference();
        let result = Request {
            device,
            bind: &bind,
            span,
            class: Class::Fragment,
            proof,
            scope: Some(scope.clone()),
        }
        .fill_with(
            &mut fixture.state,
            &mut fixture.host,
            |state, host, bytes| {
                span.read_into(state, host, bytes)?;
                owner.take();
                Ok(bytes.len())
            },
        )
        .unwrap();
        assert_eq!(scope.current(), None);
        assert!(!retained(&bind));
        assert_eq!(captured(&result), [0x11; 16]);
    });
}

#[test]
fn input_snapshot_operator_policy_only_narrows_and_reports_unknown_values() {
    use crate::config::Switch;
    let capture = crate::observe::FailCapture::start();
    assert!(permitted(Switch::Unset, None));
    assert!(permitted(Switch::On, Some("on")));
    assert!(!permitted(Switch::Off, Some("off")));
    assert!(!permitted(Switch::Unrecognized, Some("invalid-fixture")));
    assert!(capture.lines().iter().any(|line| {
        line.contains("metal_input_snapshot_switch_unrecognized")
            && line.contains("invalid-fixture")
            && line.contains(crate::config::METAL_INPUT_SNAPSHOT_REUSE)
    }));
}

#[test]
fn input_snapshot_operator_disabled_retains_fresh_private_native_captures() {
    if enabled() {
        return;
    }
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let before = input::snapshot(Class::Fragment);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x11; 16]);
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x76; 16])
            .unwrap();
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x76; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
        assert!(!retained(&bind));
    });
}

#[test]
fn deferred_guest_observations_capture_fresh_inputs_without_cache_validation() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        fixture.host.guest_write_deferred = true;
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let audits = crate::runtime::drain::store_route_count("gw_rail_buffer");
        let before = input::snapshot(Class::Fragment);
        for value in [0x35, 0x76, 0x91] {
            fixture.host.write_gpa(8 * fixture.page + 4, &[value; 16]).unwrap();
            assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [value; 16]);
        }
        assert_eq!(input::snapshot(Class::Fragment).direct_fills - before.direct_fills, 3);
        assert_eq!(crate::runtime::drain::store_route_count("gw_rail_buffer"), audits);
        assert!(!retained(&bind));
    });
}

fn captured(input: &Filled) -> Vec<u8> {
    let bytes = input.test_contents();
    let offset = input.binding_offset() as usize;
    bytes[offset..offset + input.captured_len()].to_vec()
}

#[test]
fn readonly_snapshot_hits_avoid_full_captures_without_imports() {
    objc::rc::autoreleasepool(|| {
        for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
            let mut fixture = Fixture::new(shift);
            let mut bind = fixture.bind(7, 1, fixture.page * 2 + 20, 4);
            bind.index = 1;
            let proof = Some(buffer_extent::tests::unbounded_readonly());
            let before = input::snapshot(Class::Fragment);
            let first = prepare(&mut fixture, &bind, proof);
            let expected = captured(&first);
            let source_len = fixture.page * 2 + 16;
            assert_eq!(first.len() as u64, source_len);
            assert_eq!(first.captured_len() as u64, source_len);
            assert!(retained(&bind));
            for _ in 0..63 {
                let input = prepare(&mut fixture, &bind, proof);
                assert_eq!(input.len(), first.len());
                assert_eq!(input.binding_offset(), first.binding_offset());
                assert_eq!(captured(&input), expected);
            }
            let after = input::snapshot(Class::Fragment);
            assert_eq!(after.direct_fills - before.direct_fills, 1);
            assert_eq!(
                after.direct_fill_bytes - before.direct_fill_bytes,
                source_len
            );
            assert_eq!(after.allocations - before.allocations, 1);
            assert_eq!(
                fixture.host.map_pages_calls, 0,
                "private copies need no alias"
            );
            eprintln!(
                "snapshot captured={} avoided={} over 64 binds",
                source_len,
                63 * source_len
            );
        }
    });
}

#[test]
fn readonly_snapshot_refreshes_guest_and_host_writes_but_not_disjoint_writes() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let original = prepare(&mut fixture, &bind, proof);
        let gpa = 8 * fixture.page + 4;
        let before = input::snapshot(Class::Fragment);
        fixture.host.guest_wrote_page(13 * fixture.page);
        fixture
            .state
            .host_writes
            .note_pages(vec![21 * fixture.page]);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x11; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills,
            before.direct_fills
        );
        fixture.host.write_gpa(gpa, &[0x42; 16]).unwrap();
        fixture.host.guest_wrote_page(gpa);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x42; 16]);
        fixture.host.write_gpa(gpa, &[0x53; 16]).unwrap();
        fixture.state.host_writes.note_pages(vec![8 * fixture.page]);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x53; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
        assert_eq!(
            captured(&original),
            [0x11; 16],
            "earlier private leases are immutable"
        );
    });
}

#[test]
fn readonly_snapshot_rewalks_interior_pte_without_mapping_generation() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let mut bind = fixture.bind(7, 1, fixture.page * 3, 0);
        bind.index = 1;
        let proof = Some(buffer_extent::tests::unbounded_readonly());
        let original = prepare(&mut fixture, &bind, proof);
        let before = input::snapshot(Class::Fragment);
        fixture
            .host
            .write_gpa(3 * fixture.page + 2 * 4, &21u32.to_le_bytes())
            .unwrap();
        let changed = prepare(&mut fixture, &bind, proof);
        let bytes = captured(&changed);
        assert!(bytes[..fixture.page as usize].iter().all(|&b| b == 0x11));
        assert!(bytes[fixture.page as usize..].iter().all(|&b| b == 0x33));
        assert_eq!(captured(&original)[fixture.page as usize], 0x22);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            1
        );
    });
}

#[test]
fn readonly_snapshot_keys_source_size_offset_and_captured_coverage_exactly() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let mut bind = fixture.bind(7, 1, 64, 4);
        let wide = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let narrow = Some(buffer_extent::tests::object(Stage::Fragment, 0, 4));
        let first = prepare(&mut fixture, &bind, wide);
        let before = input::snapshot(Class::Fragment);
        let short = prepare(&mut fixture, &bind, narrow);
        assert_eq!(short.captured_len(), 4);
        assert_eq!(short.test_contents(), first.test_contents());
        bind.offset = 8;
        let moved = prepare(&mut fixture, &bind, wide);
        assert_eq!(moved.len(), 56);
        assert_eq!(moved.captured_len(), 16);
        assert_eq!(captured(&moved), [0x11; 16]);
        assert!(fixture.state.delete_object(1, 7));
        let replacement = fixture.bind(7, 2, 80, 8);
        let new = prepare(&mut fixture, &replacement, wide);
        assert_eq!(new.len(), 72);
        assert_eq!(captured(&new), [0x22; 16]);
        assert_eq!(captured(&first), [0x11; 16]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
    });
}

#[test]
fn readonly_snapshot_unknown_writable_and_unobservable_inputs_always_capture() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let readonly = Some(buffer_extent::tests::object(Stage::Fragment, 0, 4));
        let first = prepare(&mut fixture, &bind, readonly);
        let before = input::snapshot(Class::Fragment);
        for proof in [None, Some(buffer_extent::tests::writable_object())] {
            let input = prepare(&mut fixture, &bind, proof);
            assert_eq!(captured(&input)[0], 0x11);
            assert!(
                input.freeze().is_err(),
                "mutable/unknown native use cannot share"
            );
        }
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
        drop(first);
        // A fresh host with no guest-write observation must never retain.
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        fixture.host.guest_writes_unobservable = true;
        let bind = fixture.bind(7, 1, 64, 4);
        let before = input::snapshot(Class::Fragment);
        for _ in 0..3 {
            drop(prepare(&mut fixture, &bind, readonly));
        }
        assert!(!retained(&bind));
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            3
        );
    });
}

#[test]
fn readonly_snapshot_partial_failures_and_mid_capture_changes_do_not_retain() {
    objc::rc::autoreleasepool(|| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
        for kind in 0..4 {
            let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
            let bind = fixture.bind(7, 1, 64, 4);
            let access = buffer_extent::tests::object(Stage::Fragment, 0, 16);
            drop(prepare(&mut fixture, &bind, Some(access)));
            fixture.host.guest_wrote_page(8 * fixture.page);
            let span = prepare_bound_buffer_read_with_extent(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                Some(16),
            )
            .unwrap()
            .span;
            let (_, proof) = access.capture_for(Class::Fragment, 0).unwrap();
            let page = fixture.page;
            let result = Request {
                device,
                bind: &bind,
                span,
                class: Class::Fragment,
                proof,
                scope: Some(fixture.input_scope.reference()),
            }
            .fill_with(
                &mut fixture.state,
                &mut fixture.host,
                |state, host, bytes| {
                    span.read_into(state, host, bytes)?;
                    match kind {
                        0 => return Ok(3),
                        1 => {
                            host.write_gpa(8 * page + 4, &[0x77; 16])?;
                            host.guest_wrote_page(8 * page);
                        }
                        2 => {
                            host.write_gpa(8 * page + 4, &[0x77; 16])?;
                            state.host_writes.note_pages(vec![8 * page]);
                        }
                        _ => host.write_gpa(3 * page + 4, &13u32.to_le_bytes())?,
                    }
                    Ok(bytes.len())
                },
            );
            assert!(
                !retained(&bind),
                "partial fill or changed proof cannot be cached"
            );
            if kind == 0 {
                assert!(result.is_err());
            } else {
                assert!(result.is_ok());
            }
            let before = input::snapshot(Class::Fragment);
            let next = prepare(&mut fixture, &bind, Some(access));
            assert_eq!(
                input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
                1
            );
            assert_eq!(
                captured(&next),
                [match kind {
                    0 => 0x11,
                    3 => 0x22,
                    _ => 0x77,
                }; 16]
            );
        }
    });
}

#[test]
fn readonly_snapshot_waits_for_guest_tracking_startup_and_preserves_other_rail_state() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        fixture.host.guest_write_startup_window = true;
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        for _ in 0..3 {
            drop(prepare(&mut fixture, &bind, proof));
            assert!(!retained(&bind));
        }
        fixture.host.guest_wrote_page(8 * fixture.page);
        drop(prepare(&mut fixture, &bind, proof));
        assert!(
            retained(&bind),
            "a readable generation must permit recovery"
        );
        let before = input::snapshot(Class::Fragment);
        drop(prepare(&mut fixture, &bind, proof));
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills,
            before.direct_fills
        );

        #[derive(Default)]
        struct OtherRail;
        impl crate::model::RailResourceState for OtherRail {}
        let bind = fixture.bind(9, 2, 64, 4);
        bind.resource
            .as_ref()
            .unwrap()
            .with_rail_state(|_: &mut OtherRail| ())
            .unwrap();
        for _ in 0..2 {
            drop(prepare(&mut fixture, &bind, proof));
        }
        assert!(bind
            .resource
            .as_ref()
            .unwrap()
            .with_rail_state(|_: &mut OtherRail| ())
            .is_some());
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fills - before.direct_fills,
            2
        );
    });
}

#[test]
fn readonly_snapshot_resource_retirement_drops_candidate_not_existing_lease() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let input = prepare(&mut fixture, &bind, proof);
        let weak = bind
            .resource
            .as_ref()
            .unwrap()
            .with_rail_state(|held: &mut RetainedBuffer| {
                Arc::downgrade(&held.latest.as_ref().unwrap().image)
            })
            .unwrap();
        assert!(fixture.state.delete_object(1, 7));
        drop(bind);
        assert!(
            weak.upgrade().is_some(),
            "the native lease outlives the resource"
        );
        assert_eq!(captured(&input), [0x11; 16]);
        drop(input);
        assert!(
            weak.upgrade().is_none(),
            "no process-wide native snapshot cache"
        );
    });
}
