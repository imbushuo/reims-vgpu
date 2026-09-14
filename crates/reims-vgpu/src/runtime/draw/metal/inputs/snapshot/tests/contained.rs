use super::*;
use crate::runtime::drain::store_route_count;

#[test]
fn contained_snapshot_alternating_vertex_fragment_extents_avoids_actual_captures() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let vertex = buffer_extent::tests::object(Stage::Vertex, 0, 16);
        let fragment = buffer_extent::tests::object(Stage::Fragment, 0, 4);
        let v = input::snapshot(Class::Vertex);
        let f = input::snapshot(Class::Fragment);
        let rearmed = store_route_count("gw_rearm");
        let avoided = store_route_count("metal_input_snapshot_capture_bytes_avoided");
        for index in 0..64 {
            let (class, access, len) = if index % 2 == 0 {
                (Class::Vertex, vertex, 16)
            } else {
                (Class::Fragment, fragment, 4)
            };
            let PreparedInput::Native(input) = crate::runtime::draw::metal::inputs::prepare(
                &mut fixture.state,
                &mut fixture.host,
                1,
                &bind,
                class,
                Capture::ScopedNative(Some(access), fixture.input_scope.reference()),
                "test_contained_input",
            )
            .unwrap() else {
                panic!("native capture")
            };
            assert_eq!(input.bytes.len(), 60);
            assert_eq!(input.bytes.captured_len(), len);
            assert_eq!(captured(&input.bytes), vec![0x11; len]);
        }
        assert_eq!(
            input::snapshot(Class::Vertex).direct_fill_bytes - v.direct_fill_bytes,
            16
        );
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fill_bytes - f.direct_fill_bytes,
            0
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_capture_bytes_avoided") - avoided,
            624
        );
        assert_eq!(store_route_count("gw_rearm") - rearmed, 1);
    });
}

#[test]
fn contained_snapshot_increasing_offsets_avoid_full_suffix_copies() {
    objc::rc::autoreleasepool(|| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
        if !device.supports_family(::metal::MTLGPUFamily::Apple2) {
            return;
        }
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let size = fixture.page * 2 + 128;
        let mut bind = fixture.bind(7, 1, size, 0);
        bind.index = 1;
        let proof = Some(buffer_extent::tests::unbounded_readonly());
        let before = input::snapshot(Class::Fragment);
        let avoided = store_route_count("metal_input_snapshot_capture_bytes_avoided");
        let rearmed = store_route_count("gw_rearm");
        let guest_avoided = store_route_count("metal_input_snapshot_guest_bytes_avoided");
        let audit = store_route_count("metal_input_snapshot_hit_audit_bytes");
        let extra = store_route_count("metal_input_snapshot_extra_audit_bytes");
        let expected: u64 = (1..70).map(|i| size - i * 4).sum();
        for index in 0..70 {
            bind.offset = index * 4;
            let input = prepare(&mut fixture, &bind, proof);
            assert_eq!(input.binding_offset(), bind.offset);
            assert_eq!(input.len() as u64, size - bind.offset);
            assert_eq!(input.captured_len(), input.len());
            let bytes = captured(&input);
            assert_eq!(bytes[0], 0x11);
            assert_eq!(bytes.last(), Some(&0x33));
        }
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
            size
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_capture_bytes_avoided") - avoided,
            expected
        );
        assert_eq!(store_route_count("gw_rearm") - rearmed, 1);
        let guest_avoided =
            store_route_count("metal_input_snapshot_guest_bytes_avoided") - guest_avoided;
        let audit = store_route_count("metal_input_snapshot_hit_audit_bytes") - audit;
        let extra = store_route_count("metal_input_snapshot_extra_audit_bytes") - extra;
        assert_eq!(
            guest_avoided as i128 - extra as i128,
            expected as i128 - audit as i128
        );
        eprintln!(
            "contained offsets: captured={size} avoided={expected} audit={audit} extra_audit={extra} \
             net_guest_avoided={} over 70 bindings",
            guest_avoided as i128 - extra as i128,
        );
    });
}

#[test]
fn contained_snapshot_metadata_plan_preserves_tracker_startup_without_inventing_bytes() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        fixture.host.guest_write_startup_window = true;
        let bind = fixture.bind(7, 1, 64, 4);
        let wide = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let narrow = Some(buffer_extent::tests::object(Stage::Fragment, 0, 4));
        let rearmed = store_route_count("gw_rearm");
        let partial = store_route_count("metal_input_snapshot_metadata_plan_unvouched");
        for proof in [wide, narrow, wide, narrow] {
            drop(prepare(&mut fixture, &bind, proof));
            assert!(!retained(&bind));
        }
        assert_eq!(store_route_count("gw_rearm") - rearmed, 1);
        assert_eq!(
            store_route_count("metal_input_snapshot_metadata_plan_unvouched") - partial,
            2
        );
        fixture.host.guest_wrote_page(8 * fixture.page);
        drop(prepare(&mut fixture, &bind, narrow));
        assert!(
            !retained(&bind),
            "a narrow fresh capture never claims the wider plan's bytes"
        );
        drop(prepare(&mut fixture, &bind, wide));
        assert!(retained(&bind));
        let before = input::snapshot(Class::Fragment);
        drop(prepare(&mut fixture, &bind, narrow));
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fill_bytes,
            before.direct_fill_bytes
        );
        assert_eq!(store_route_count("gw_rearm") - rearmed, 1);
    });
}

#[test]
fn contained_snapshot_revalidates_writes_and_ptes_outside_the_requested_subrange() {
    objc::rc::autoreleasepool(|| {
        for rewire in [false, true] {
            let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
            let mut bind = fixture.bind(7, 1, fixture.page * 2 + 128, 0);
            bind.index = 1;
            let proof = Some(buffer_extent::tests::unbounded_readonly());
            let original = prepare(&mut fixture, &bind, proof);
            bind.offset = fixture.page * 2;
            if rewire {
                fixture
                    .host
                    .write_gpa(3 * fixture.page + 2 * 4, &21u32.to_le_bytes())
                    .unwrap();
            } else {
                fixture
                    .host
                    .write_gpa(8 * fixture.page, &[0x72; 4])
                    .unwrap();
                fixture.state.host_writes.note_pages(vec![8 * fixture.page]);
            }
            let before = input::snapshot(Class::Fragment);
            let next = prepare(&mut fixture, &bind, proof);
            assert_eq!(captured(&next), [0x33; 128]);
            assert_eq!(
                input::snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
                128
            );
            assert!(
                !retained(&bind),
                "no full-range stable bytes were captured after revocation"
            );
            assert_eq!(captured(&original)[0], 0x11);
        }
    });
}

#[test]
fn contained_snapshot_invalid_outer_pages_do_not_block_a_valid_narrow_capture() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let mut bind = fixture.bind(7, 1, fixture.page * 2 + 128, 0);
        bind.index = 1;
        let proof = Some(buffer_extent::tests::unbounded_readonly());
        drop(prepare(&mut fixture, &bind, proof));
        fixture
            .host
            .write_gpa(3 * fixture.page + 2 * 4, &0u32.to_le_bytes())
            .unwrap();
        bind.offset = fixture.page * 2;
        let before = input::snapshot(Class::Fragment);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x33; 128]);
        assert!(retained(&bind));
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x33; 128]);
        assert_eq!(
            input::snapshot(Class::Fragment).direct_fill_bytes - before.direct_fill_bytes,
            128
        );
    });
}
