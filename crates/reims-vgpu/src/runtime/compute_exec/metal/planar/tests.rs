use super::*;
use crate::backend::metal::planar::tests::{device_descriptor, texture_descriptor};
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st32, st64};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::protocol::planar::{BackingFormat, SampleFormat};
use crate::runtime::compute_exec::{metal::try_stage_planar_sampled_in_scope, stage_texture_raw};
use crate::runtime::decode::resource::{
    list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_MAPPER_REF_TEXTURE,
};
use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;
use crate::runtime::{mapper, objects};

const PAGE: usize = 1 << PAGE_SHIFT_ARM64E;
const GPA: u64 = 0x20 << PAGE_SHIFT_ARM64E;

struct Fixture {
    state: DeviceState,
    host: FakeHost,
    scope: crate::runtime::draw::BufferSnapshotScope,
}

impl Fixture {
    fn new() -> Self {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));
        host.map_range(GPA, PAGE, 0x5a);
        assert!(state.map_surface(5));
        state.mappings.get_mut(&5).unwrap().page_entries =
            vec![(0x20 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.set_mapping_device_desc(5, &device_descriptor(BackingFormat::VideoRange)));
        assert!(state.set_mapping_geom(5, 8, 4, SampleFormat::Rgb10_420TwoPlane.word()));
        let mut fixture = Self {
            state,
            host,
            scope: crate::runtime::draw::BufferSnapshotScope::new(),
        };
        fixture.descriptor(SampleFormat::Rgb10_420TwoPlane);
        fixture
    }

    fn descriptor(&mut self, format: SampleFormat) {
        let descriptor = texture_descriptor(format);
        write_task_gva_arm64e(&mut self.host, &self.state.tasks[1], 0x300, &descriptor);
        let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | ((descriptor.len() as u32) << 8),
        );
        st64(&mut entry[4..], 0x300);
        write_task_gva_arm64e(
            &mut self.host,
            &self.state.tasks[1],
            list_object_entry_offset(11, 32).unwrap(),
            &entry,
        );
    }

    fn source(&mut self) -> PlanarSource {
        let resource = objects::resolve_resource(&self.state, &self.host, 1, 11).unwrap();
        assert_eq!(
            objects::resolve_mapper_ref_texture_resource(&mut self.state, 1, 11, &resource),
            Some(5),
        );
        PlanarSource {
            description: TextureDescription::decode(&resource.descriptor, 11).unwrap(),
            resource,
            texture_ref: 11,
            layout: Layout::decode(&self.state.mappings[&5].device_desc, 8, 4).unwrap(),
            map_generation: self.state.mappings[&5].map_generation,
            scope: Some(self.scope.reference()),
        }
    }

    fn fragment(&mut self) -> Arc<SampledImage> {
        try_stage_planar_sampled_in_scope(
            &mut self.state,
            &mut self.host,
            1,
            11,
            Some(&self.scope.reference()),
        )
        .unwrap()
        .unwrap()
    }

    fn write(&mut self, byte: u8) {
        self.host.write_gpa(GPA, &[byte; PAGE]).unwrap();
    }
}

#[test]
fn both_quiet_reuses_within_render_scope_but_not_for_unscoped_compute() {
    let mut f = Fixture::new();
    let first = f.fragment();
    let source = f.source();
    for _ in 0..128 {
        let second = stage_with(&mut f.state, &mut f.host, &source, |_, _, _| {
            panic!("a vouched hit must not allocate or fill an IOSurface")
        })
        .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }
    let compute = stage_texture_raw::<MetalStage, _>(&mut f.state, &mut f.host, 1, 11, 33, false)
        .ok()
        .unwrap();
    assert!(!Arc::ptr_eq(&first, compute.rail.planar.as_ref().unwrap()));
    assert!(compute.bytes.is_empty());
    assert_eq!(first.plane_bytes(0), &[0x5a; 1024]);
    assert_eq!(first.plane_bytes(1), &[0x5a; 1024]);
}

#[test]
fn guest_cpu_write_restages_even_when_host_epochs_are_quiet() {
    let mut f = Fixture::new();
    let first = f.fragment();
    let epoch = f.state.host_writes.epoch();
    f.write(0x61);
    f.host.guest_wrote_page(GPA);
    assert_eq!(f.state.host_writes.epoch(), epoch);
    let second = f.fragment();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.plane_bytes(0), &[0x5a; 1024]);
    assert_eq!(second.plane_bytes(1), &[0x61; 1024]);
    assert!(Arc::ptr_eq(&second, &f.fragment()));
}

#[test]
fn current_write_observations_reuse_across_passes_but_never_trust_unharvested_writes() {
    let mut f = Fixture::new();
    f.host.guest_write_current_supported = true;
    let first = f.fragment();
    f.scope = crate::runtime::draw::BufferSnapshotScope::new();
    assert!(Arc::ptr_eq(&first, &f.fragment()));
    f.write(0x73);
    f.host.guest_write_current_unavailable = true;
    let changed = f.fragment();
    assert!(!Arc::ptr_eq(&first, &changed));
    assert_eq!(changed.plane_bytes(0), &[0x73; 1024]);
    assert_eq!(first.plane_bytes(0), &[0x5a; 1024]);
    f.host.guest_wrote_page(GPA);
    f.host.guest_write_current_unavailable = false;
    let current = f.fragment();
    f.scope = crate::runtime::draw::BufferSnapshotScope::new();
    assert!(Arc::ptr_eq(&current, &f.fragment()));
    f.host.guest_write_current_supported = false;
    f.scope = crate::runtime::draw::BufferSnapshotScope::new();
    f.write(0x91);
    let deferred = f.fragment();
    assert!(!Arc::ptr_eq(&current, &deferred));
    assert_eq!(deferred.plane_bytes(1), &[0x91; 1024]);
}

#[test]
fn planar_completed_commands_refresh_cpu_iosurface_updates_without_a_harvest() {
    use crate::backend::metal::planar::tests::cpu_surface::{CpuSurface, Sampler};
    objc::rc::autoreleasepool(|| {
        let mut source = CpuSurface::planar();
        let sampler = Sampler::new();
        let mut f = Fixture::new();
        let device = crate::backend::metal::runtime::system_device().unwrap();
        for round in 0..256u16 {
            f.scope = crate::runtime::draw::BufferSnapshotScope::new();
            let seed = source.seed();
            let bytes = source.planar_pixels(
                64 + (round * 29) % 876,
                64 + (round * 47) % 896,
                64 + (round * 71) % 896,
            );
            assert_ne!(source.seed(), seed);
            let oracle = sampler.sample(source.texture());
            f.host.write_gpa(GPA, &bytes).unwrap();
            let snapshot = f.fragment();
            let actual = sampler.sample(&snapshot.texture(device).unwrap());
            assert_eq!(actual, oracle, "completed planar command round={round}");
        }
    });
}
#[test]
fn exact_host_page_write_restages_but_disjoint_writes_do_not() {
    let mut f = Fixture::new();
    let first = f.fragment();
    f.state.note_host_wrote_pages(vec![GPA + PAGE as u64]);
    assert!(Arc::ptr_eq(&first, &f.fragment()));
    f.state.note_host_wrote_pages(vec![GPA]);
    f.write(0x62);
    let second = f.fragment();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.plane_bytes(0), &[0x62; 1024]);
    assert!(Arc::ptr_eq(&second, &f.fragment()));
    f.state.note_host_wrote_guest_ram();
    assert!(
        !Arc::ptr_eq(&second, &f.fragment()),
        "an unnamed write fails closed"
    );
}

#[test]
fn planar_expired_or_absent_scope_always_reads_fresh_planes() {
    let mut f = Fixture::new();
    f.fragment();
    let expired = f.scope.reference();
    f.scope = crate::runtime::draw::BufferSnapshotScope::new();
    assert!(expired.current().is_none());
    for scope in [Some(&expired), None] {
        for byte in [0x61, 0x72] {
            f.write(byte);
            let image = try_stage_planar_sampled_in_scope(&mut f.state, &mut f.host, 1, 11, scope)
                .unwrap()
                .unwrap();
            for plane in 0..2 {
                assert_eq!(image.plane_bytes(plane), &[byte; 1024]);
            }
            let source = f.source();
            assert!(source
                .resource
                .with_rail_state(|held: &mut RetainedPlanar| held.latest.is_none())
                .unwrap());
        }
    }
}

#[test]
fn planar_scope_ending_during_fill_cannot_retain_an_image() {
    let mut f = Fixture::new();
    let mut owner = Some(crate::runtime::draw::BufferSnapshotScope::new());
    let mut source = f.source();
    source.scope = Some(owner.as_ref().unwrap().reference());
    let image = stage_with(&mut f.state, &mut f.host, &source, |state, host, source| {
        let image = fill(state, host, source)?;
        drop(owner.take());
        Ok(image)
    })
    .unwrap();
    assert_eq!(image.plane_bytes(0), &[0x5a; 1024]);
    assert!(source
        .resource
        .with_rail_state(|held: &mut RetainedPlanar| held.latest.is_none())
        .unwrap());
}

#[test]
fn unobservable_or_unreadable_guest_tracking_never_retains() {
    for unobservable in [true, false] {
        let mut f = Fixture::new();
        f.host.guest_writes_unobservable = unobservable;
        f.host.guest_write_startup_window = !unobservable;
        let first = f.fragment();
        f.write(0x63);
        let second = f.fragment();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.plane_bytes(1), &[0x63; 1024]);
        let source = f.source();
        assert!(source
            .resource
            .with_rail_state(|held: &mut RetainedPlanar| { held.latest.is_none() })
            .unwrap());
        if !unobservable {
            // Close the fixture's arming window by publishing its first
            // observable write; never reinterpret generation zero as quiet.
            f.host.guest_wrote_page(GPA);
            let armed = f.fragment();
            assert!(!Arc::ptr_eq(&second, &armed));
            assert!(Arc::ptr_eq(&armed, &f.fragment()));
        }
    }
}

#[test]
fn remapped_physical_pages_and_stated_content_generation_cannot_hit() {
    let mut f = Fixture::new();
    let first = f.fragment();
    assert!(f.state.invalidate_mapping_pages(5));
    f.host.map_range(GPA + PAGE as u64, PAGE, 0x64);
    f.state.mappings.get_mut(&5).unwrap().page_entries =
        vec![(0x21 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    let second = f.fragment();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.plane_bytes(1), &[0x64; 1024]);
    assert!(Arc::ptr_eq(&second, &f.fragment()));
    f.state.mappings.get_mut(&5).unwrap().content_generation += 1;
    assert!(!Arc::ptr_eq(&second, &f.fragment()));
}

#[test]
fn complete_layout_and_texture_description_are_cache_identity() {
    let mut f = Fixture::new();
    let first = f.fragment();
    // Same bytes, dimensions and allocation size, different native conversion.
    assert!(f
        .state
        .set_mapping_device_desc(5, &device_descriptor(BackingFormat::FullRange)));
    let second = f.fragment();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second.layout().backing_format, BackingFormat::FullRange);
    // Offset is independent of base/size: a compact key must not lose it.
    let mut backing = device_descriptor(BackingFormat::FullRange);
    st32(&mut backing[64 + 8..], 192);
    assert!(f.state.set_mapping_device_desc(5, &backing));
    let third = f.fragment();
    assert!(!Arc::ptr_eq(&second, &third));
    assert_eq!(third.layout().planes[0].offset, 192);
    assert!(f.state.delete_object(1, 11));
    f.descriptor(SampleFormat::Ycbcr10_420TwoPlane);
    let fourth = f.fragment();
    assert!(!Arc::ptr_eq(&third, &fourth));
    assert_eq!(
        fourth.description().format,
        SampleFormat::Ycbcr10_420TwoPlane
    );
}

#[test]
fn writes_during_fill_cannot_publish_under_the_post_write_generation() {
    for guest in [false, true] {
        let mut f = Fixture::new();
        let source = f.source();
        let image = stage_with(&mut f.state, &mut f.host, &source, |state, host, source| {
            MetalStage::stage_planar(
                source.texture_ref,
                source.description,
                source.layout.clone(),
                |[first, second]| {
                    assert!(mapper::read_mapping_into(state, host, 5, 0, first));
                    if guest {
                        host.guest_wrote_page(GPA);
                    } else {
                        state.note_host_wrote_pages(vec![GPA]);
                    }
                    host.write_gpa(GPA, &[0x65; PAGE]).unwrap();
                    assert!(mapper::read_mapping_into(state, host, 5, 2048, second));
                    Ok(())
                },
            )
            .map(|stage| stage.planar.unwrap())
        })
        .unwrap();
        assert!(source
            .resource
            .with_rail_state(|held: &mut RetainedPlanar| { held.latest.is_none() })
            .unwrap());
        assert_eq!(image.plane_bytes(0), &[0x5a; 1024]);
        assert_eq!(image.plane_bytes(1), &[0x65; 1024]);
        let next = f.fragment();
        assert!(!Arc::ptr_eq(&image, &next));
        assert_eq!(next.plane_bytes(1), &[0x65; 1024]);
        assert!(Arc::ptr_eq(&next, &f.fragment()));
    }
}

#[test]
fn failed_fill_drops_the_previous_entry_without_touching_inflight_images() {
    let mut f = Fixture::new();
    let first = f.fragment();
    f.host.guest_wrote_page(GPA);
    let source = f.source();
    assert!(matches!(
        stage_with(&mut f.state, &mut f.host, &source, |_, _, _| {
            Err(ComputeStatus::GuestIo("planar_mapping_read"))
        }),
        Err(ComputeStatus::GuestIo("planar_mapping_read")),
    ));
    assert_eq!(Arc::strong_count(&first), 1);
    assert_eq!(first.plane_bytes(1), &[0x5a; 1024]);
    assert!(!Arc::ptr_eq(&first, &f.fragment()));
}

#[test]
fn resource_deletion_and_reset_release_the_entry_but_not_inflight_arc() {
    for reset in [false, true] {
        let mut f = Fixture::new();
        let inflight = f.fragment();
        let weak = Arc::downgrade(&inflight);
        assert_eq!(Arc::strong_count(&inflight), 2);
        if reset {
            f.state.reset();
        } else {
            assert!(f.state.delete_object(1, 11));
        }
        assert_eq!(Arc::strong_count(&inflight), 1);
        assert_eq!(inflight.plane_bytes(0), &[0x5a; 1024]);
        drop(inflight);
        assert!(weak.upgrade().is_none());
        mapper::flush_retired_views(&mut f.state, &mut f.host);
    }
}

#[test]
fn separate_devices_never_share_images_for_equal_numeric_keys() {
    let mut a = Fixture::new();
    let mut b = Fixture::new();
    b.write(0x66);
    let first = a.fragment();
    let second = b.fragment();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first.plane_bytes(0), &[0x5a; 1024]);
    assert_eq!(second.plane_bytes(0), &[0x66; 1024]);
}

#[test]
fn audit_folds_actual_guest_bytes_not_the_cached_image() {
    let mut f = Fixture::new();
    let first = f.fragment();
    // Deliberately model a continuing escaping writer, so every audit pair
    // sees different guest bytes but the immutable cached image never moves.
    for bind in 0..=2 * crate::runtime::gather_witness::AUDIT_STRIDE {
        let byte = 0x67 + (bind % 2) as u8;
        f.write(byte);
        let next = f.fragment();
        if !Arc::ptr_eq(&first, &next) {
            assert_eq!(next.plane_bytes(1), &[byte; 1024]);
            return;
        }
    }
    panic!("the guest-content audit never invalidated the stale image");
}
