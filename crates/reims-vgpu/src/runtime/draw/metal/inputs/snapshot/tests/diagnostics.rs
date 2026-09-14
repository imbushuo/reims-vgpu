use super::*;
use crate::runtime::drain::store_route_count;

fn staged(fixture: &mut Fixture, bind: &BufferBind, class: Class, proof: BufferRead) -> Filled {
    let PreparedInput::Native(input) = crate::runtime::draw::metal::inputs::prepare(
        &mut fixture.state,
        &mut fixture.host,
        1,
        bind,
        class,
        Capture::ScopedNative(Some(proof), fixture.input_scope.reference()),
        "test_snapshot_diagnostics",
    )
    .unwrap() else {
        panic!("native capture")
    };
    input.bytes
}

#[test]
fn snapshot_diagnostics_identify_alternating_stage_offsets_without_reusing_displaced_storage() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let vertex = fixture.bind(7, 1, 64, 4);
        let fragment = BufferBind {
            offset: 24,
            ..vertex.clone()
        };
        fixture
            .host
            .write_gpa(8 * fixture.page + 24, &[0x22; 16])
            .unwrap();
        let vertex_proof = buffer_extent::tests::object(Stage::Vertex, 0, 16);
        let fragment_proof = buffer_extent::tests::object(Stage::Fragment, 0, 16);
        let before_stage = store_route_count("metal_input_snapshot_candidate_stage_changed");
        let before_offset = store_route_count("metal_input_snapshot_miss_offset");
        let before_displaced = store_route_count("metal_input_snapshot_displaced_vouched");
        let before_bytes = store_route_count("metal_input_snapshot_displaced_vouched_bytes");
        let before_hits = store_route_count("metal_input_snapshot_hits");
        let first = staged(&mut fixture, &vertex, Class::Vertex, vertex_proof);
        assert_eq!(captured(&first), [0x11; 16]);
        let old_native = vertex
            .resource
            .as_ref()
            .unwrap()
            .with_rail_state(|held: &mut RetainedBuffer| {
                Arc::downgrade(&held.latest.as_ref().unwrap().image)
            })
            .unwrap();
        drop(first);
        assert_eq!(
            captured(&staged(
                &mut fixture,
                &fragment,
                Class::Fragment,
                fragment_proof
            )),
            [0x22; 16]
        );
        assert!(
            old_native.upgrade().is_none(),
            "diagnostic metadata must not hold native storage"
        );
        let before = input::snapshot(Class::Vertex);
        assert_eq!(
            captured(&staged(&mut fixture, &vertex, Class::Vertex, vertex_proof)),
            [0x11; 16]
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_candidate_stage_changed") - before_stage,
            2
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_miss_offset") - before_offset,
            2
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_vouched") - before_displaced,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_vouched_bytes") - before_bytes,
            16
        );
        assert_eq!(store_route_count("metal_input_snapshot_hits"), before_hits);
        assert_eq!(
            input::snapshot(Class::Vertex).direct_fills - before.direct_fills,
            1
        );
    });
}

#[test]
fn snapshot_diagnostics_distinguish_unvouched_capture_from_page_or_layout_mutation() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        fixture.host.guest_writes_unobservable = true;
        let bind = fixture.bind(7, 1, 64, 4);
        let unvouched = store_route_count("metal_input_snapshot_post_unvouched");
        let identity = store_route_count("metal_input_snapshot_post_identity_changed");
        let key = store_route_count("metal_input_snapshot_post_key_changed");
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x11; 16]);
        assert!(!retained(&bind));
        assert_eq!(
            store_route_count("metal_input_snapshot_post_unvouched") - unvouched,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_post_identity_changed") - identity,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_post_key_changed"),
            key
        );
    });
}

#[test]
fn snapshot_diagnostics_repeated_displaced_key_still_requires_current_write_proof() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let first = fixture.bind(7, 1, 64, 4);
        let second = BufferBind {
            offset: 24,
            ..first.clone()
        };
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        drop(prepare(&mut fixture, &first, proof));
        drop(prepare(&mut fixture, &second, proof));
        let exact = store_route_count("metal_input_snapshot_displaced_exact");
        let vouched = store_route_count("metal_input_snapshot_displaced_vouched");
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x43; 16])
            .unwrap();
        fixture.host.guest_wrote_page(8 * fixture.page);
        assert_eq!(captured(&prepare(&mut fixture, &first, proof)), [0x43; 16]);
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_exact") - exact,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_vouched"),
            vouched
        );
    });
}

#[test]
fn snapshot_contained_same_gva_extents_preserve_the_original_witness() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let wide = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        let narrow = Some(buffer_extent::tests::object(Stage::Fragment, 0, 4));
        let retained_before = store_route_count("metal_input_snapshot_retained");
        let lengths = store_route_count("metal_input_snapshot_miss_capture_length");
        let displaced = store_route_count("metal_input_snapshot_displaced_exact");
        let vouched = store_route_count("metal_input_snapshot_displaced_vouched");
        let rearmed = store_route_count("gw_rearm");
        let hits = store_route_count("metal_input_snapshot_hits");
        let host_epoch = fixture.state.host_writes.epoch();
        for proof in [wide, narrow, wide] {
            drop(prepare(&mut fixture, &bind, proof));
            assert!(retained(&bind));
        }
        assert_eq!(
            store_route_count("metal_input_snapshot_retained") - retained_before,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_miss_capture_length") - lengths,
            0
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_exact") - displaced,
            0
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_displaced_vouched"),
            vouched
        );
        assert_eq!(store_route_count("gw_rearm") - rearmed, 1);
        assert_eq!(store_route_count("metal_input_snapshot_hits") - hits, 2);
        assert_eq!(fixture.state.host_writes.epoch(), host_epoch);
    });
}

#[test]
fn snapshot_diagnostics_separate_exact_metadata_from_changed_write_proof() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let bind = fixture.bind(7, 1, 64, 4);
        let proof = Some(buffer_extent::tests::object(Stage::Fragment, 0, 16));
        drop(prepare(&mut fixture, &bind, proof));
        let metadata = store_route_count("metal_input_snapshot_metadata_exact");
        let identity = store_route_count("metal_input_snapshot_miss_identity");
        let unvouched = store_route_count("metal_input_snapshot_lookup_unvouched");
        let hits = store_route_count("metal_input_snapshot_hits");
        fixture
            .host
            .write_gpa(8 * fixture.page + 4, &[0x42; 16])
            .unwrap();
        fixture.state.host_writes.note_pages(vec![8 * fixture.page]);
        assert_eq!(captured(&prepare(&mut fixture, &bind, proof)), [0x42; 16]);
        assert_eq!(
            store_route_count("metal_input_snapshot_metadata_exact") - metadata,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_miss_identity") - identity,
            1
        );
        assert_eq!(
            store_route_count("metal_input_snapshot_lookup_unvouched") - unvouched,
            1
        );
        assert_eq!(store_route_count("metal_input_snapshot_hits"), hits);
    });
}

#[test]
fn snapshot_contained_binding_keeps_original_physical_initialization() {
    objc::rc::autoreleasepool(|| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
        if !device.supports_family(::metal::MTLGPUFamily::Apple2) {
            return;
        }
        let mut fixture = Fixture::new(PAGE_SHIFT_ARM64E);
        let mut wide = fixture.bind(7, 1, 64, 0);
        wide.index = 1;
        drop(prepare(
            &mut fixture,
            &wide,
            Some(buffer_extent::tests::object(Stage::Fragment, 1, 32)),
        ));
        let narrow = BufferBind {
            offset: 16,
            ..wide.clone()
        };
        let contained = store_route_count("metal_input_snapshot_contained_hits");
        let hits = store_route_count("metal_input_snapshot_hits");
        let input = prepare(
            &mut fixture,
            &narrow,
            Some(buffer_extent::tests::object(Stage::Fragment, 1, 4)),
        );
        assert_eq!(input.len(), 48);
        assert_eq!(input.captured_len(), 4);
        let offset = input.binding_offset() as usize;
        let bytes = input.test_contents();
        assert_eq!(&bytes[..32], &[0x11; 32]);
        assert_eq!(&bytes[offset..offset + 4], &[0x11; 4]);
        assert!(bytes[32..].iter().all(|&b| b == 0));
        assert_eq!(
            store_route_count("metal_input_snapshot_contained_hits") - contained,
            1
        );
        assert_eq!(store_route_count("metal_input_snapshot_hits") - hits, 1);
    });
}
