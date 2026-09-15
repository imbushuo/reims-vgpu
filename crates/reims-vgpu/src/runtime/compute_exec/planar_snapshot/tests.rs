use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::protocol::endian::{st16, st32, st64};
use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::protocol::planar::{self, BackingFormat, SampleFormat};
use crate::runtime::decode::resource::{
    list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_MAPPER_REF_TEXTURE,
};
use crate::runtime::draw::BufferSnapshotScope;
use crate::runtime::gva_mem::write_task_gva;
use crate::runtime::host::FakeHost;
use crate::runtime::objects;
use reims_vgpu_memory::{ReadBuffer, ReadDestination};

pub(crate) fn device_descriptor(backing: BackingFormat) -> [u8; 512] {
    let mut b = [0; 512];
    st32(&mut b[4..], backing.word());
    st32(&mut b[16..], 4096);
    st64(&mut b[20..], 0x80 | (8 << 8) | (0x80 << 32) | (4 << 40));
    st32(&mut b[28..], 64);
    st16(&mut b[32..], 1);
    b[36] = 2;
    for (i, (w, h, bpe, base, names)) in [(8, 4, 2, 0, &[5][..]), (4, 2, 4, 2048, &[7, 6][..])]
        .into_iter()
        .enumerate()
    {
        let p = &mut b[64 + i * 64..128 + i * 64];
        st32(&mut p[8..], base + 128);
        st32(&mut p[12..], base);
        st32(&mut p[16..], 1024);
        st64(&mut p[20..], 0x80 | (w << 8) | (0x80 << 32) | (h << 40));
        st32(&mut p[28..], 64);
        st16(&mut p[32..], bpe);
        p[planar::PLANE_COMPONENT_COUNT] = names.len() as u8;
        p[planar::PLANE_EXTENDED_PIXELS..planar::PLANE_EXTENDED_PIXELS + 4].fill(1);
        for (c, &name) in names.iter().enumerate() {
            p[planar::PLANE_COMPONENT_DEPTHS + c] = 10;
            p[planar::PLANE_COMPONENT_NAMES + c] = name;
            p[planar::PLANE_COMPONENT_RANGES + c] = backing.component_range();
        }
    }
    b
}

pub(crate) fn texture_descriptor(format: SampleFormat) -> [u8; 56] {
    let mut b = [0; 56];
    st32(&mut b, 5);
    st32(&mut b[8..], 0x2f);
    st32(
        &mut b[12..],
        reims_vgpu_wire::ops::backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN,
    );
    st32(&mut b[16..], 11);
    st32(&mut b[20..], (u32::from(format.word()) << 16) | 0x0142);
    st32(&mut b[24..], 8);
    st32(&mut b[28..], 4);
    st32(&mut b[32..], 1);
    st16(&mut b[36..], 1);
    st16(&mut b[38..], 1);
    st16(&mut b[40..], 1);
    st16(&mut b[42..], 0x10);
    b
}

pub(crate) struct Fixture {
    pub state: DeviceState,
    pub host: FakeHost,
    pub scope: BufferSnapshotScope,
}

impl Fixture {
    pub(crate) fn new(shift: u32) -> Self {
        let mut state = DeviceState::new(DeviceId(1), shift);
        let mut host = FakeHost::new();
        for pfn in [2, 3, 4, 0x20] {
            host.map_range(state.pfn_gpa(pfn), state.page_size() as usize, 0x5a);
        }
        let mut directory = [0; 8];
        st32(&mut directory[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut directory[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(state.pfn_gpa(2), &directory).unwrap();
        host.write_gpa(state.pfn_gpa(3), &4u32.to_le_bytes())
            .unwrap();
        state.define_task(1, 0x1000, 2);
        assert!(state.set_object_list(1, 0, 32));
        let descriptor = texture_descriptor(SampleFormat::Rgb10_420TwoPlane);
        write_task_gva(&mut host, &state.tasks[1], 0x300, &descriptor, shift).unwrap();
        let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | ((descriptor.len() as u32) << 8),
        );
        st64(&mut entry[4..], 0x300);
        write_task_gva(
            &mut host,
            &state.tasks[1],
            list_object_entry_offset(11, 32).unwrap(),
            &entry,
            shift,
        )
        .unwrap();
        assert!(state.map_surface(5));
        state.mappings.get_mut(&5).unwrap().page_entries =
            vec![(0x20 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.set_mapping_device_desc(5, &device_descriptor(BackingFormat::VideoRange)));
        assert!(state.set_mapping_geom(5, 8, 4, SampleFormat::Rgb10_420TwoPlane.word()));
        Self {
            state,
            host,
            scope: BufferSnapshotScope::new(),
        }
    }

    fn source(&mut self) -> PlanarSource {
        let resource = objects::resolve_resource(&self.state, &self.host, 1, 11).unwrap();
        assert_eq!(
            objects::resolve_mapper_ref_texture_resource(&mut self.state, 1, 11, &resource,),
            Some(5)
        );
        PlanarSource {
            description: TextureDescription::decode(&resource.descriptor, 11).unwrap(),
            layout: Layout::decode(&self.state.mappings[&5].device_desc, 8, 4).unwrap(),
            map_generation: self.state.mappings[&5].map_generation,
            resource,
            texture_ref: 11,
            scope: Some(self.scope.reference()),
        }
    }

    fn capture(&mut self) -> Arc<Vec<Vec<u8>>> {
        let source = self.source();
        stage_with(&mut self.state, &mut self.host, &source, &COUNTERS, fill).unwrap()
    }

    fn write(&mut self, value: u8) {
        self.host
            .write_gpa(
                self.state.pfn_gpa(0x20),
                &vec![value; self.state.page_size() as usize],
            )
            .unwrap();
    }
}

const COUNTERS: Counters = Counters {
    unscoped: "test_planar_unscoped",
    reuses: "test_planar_reuses",
    reuse_bytes: "test_planar_reuse_bytes",
    misses: "test_planar_misses",
    unretained: "test_planar_unretained",
};

fn fill(
    state: &mut DeviceState,
    host: &mut FakeHost,
    source: &PlanarSource,
) -> Result<Arc<Vec<Vec<u8>>>, ComputeStatus> {
    let mut first = vec![std::mem::MaybeUninit::uninit(); source.layout.planes[0].size as usize];
    let mut second = vec![std::mem::MaybeUninit::uninit(); source.layout.planes[1].size as usize];
    let mut first = ReadBuffer::new(&mut first);
    let mut second = ReadBuffer::new(&mut second);
    source.fill(state, host, [&mut first, &mut second])?;
    Ok(Arc::new(vec![
        first.initialized().unwrap().to_vec(),
        second.initialized().unwrap().to_vec(),
    ]))
}

#[test]
fn planar_snapshot_quiet_scope_reuses_without_filling_on_both_page_geometries() {
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        let mut f = Fixture::new(shift);
        let first = f.capture();
        let source = f.source();
        for _ in 0..128 {
            let next = stage_with(&mut f.state, &mut f.host, &source, &COUNTERS, |_, _, _| {
                panic!("a vouched hit must not read or convert planes")
            })
            .unwrap();
            assert!(Arc::ptr_eq(&first, &next));
        }
        assert_eq!(first[0], vec![0x5a; 1024]);
        assert_eq!(first[1], vec![0x5a; 1024]);
    }
}

#[test]
fn planar_snapshot_current_proof_crosses_passes_but_delayed_proof_does_not() {
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        for current in [false, true] {
            let mut f = Fixture::new(shift);
            f.host.guest_write_current_supported = current;
            let first = f.capture();
            f.scope = BufferSnapshotScope::new();
            assert_eq!(Arc::ptr_eq(&first, &f.capture()), current);
        }
    }
}

#[test]
fn planar_snapshot_observation_quality_changes_force_fresh_capture() {
    let mut f = Fixture::new(PAGE_SHIFT_ARM64E);
    f.host.guest_write_current_supported = true;
    let first = f.capture();
    f.host.guest_write_current_unavailable = true;
    f.write(0x61);
    let delayed = f.capture();
    assert!(!Arc::ptr_eq(&first, &delayed));
    assert_eq!(delayed[0], vec![0x61; 1024]);
    f.host.guest_write_current_unavailable = false;
    f.write(0x62);
    let current = f.capture();
    assert!(!Arc::ptr_eq(&delayed, &current));
    assert_eq!(current[0], vec![0x62; 1024]);
}

#[test]
fn planar_snapshot_guest_and_overlapping_host_writes_invalidate_but_disjoint_writes_do_not() {
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        let mut f = Fixture::new(shift);
        let first = f.capture();
        f.state.note_host_wrote_pages(vec![f.state.pfn_gpa(0x21)]);
        assert!(Arc::ptr_eq(&first, &f.capture()));
        f.write(0x63);
        f.state.note_host_wrote_pages(vec![f.state.pfn_gpa(0x20)]);
        let host_written = f.capture();
        assert!(!Arc::ptr_eq(&first, &host_written));
        assert_eq!(host_written[0], vec![0x63; 1024]);
        f.write(0x64);
        f.host.guest_wrote_page(f.state.pfn_gpa(0x20));
        assert_eq!(f.capture()[1], vec![0x64; 1024]);
    }
}

#[test]
fn planar_snapshot_unscoped_or_expired_capture_never_tracks_or_retains() {
    let mut f = Fixture::new(PAGE_SHIFT_X86);
    let first = f.capture();
    let mut source = f.source();
    f.scope = BufferSnapshotScope::new();
    for scope in [source.scope.clone(), None] {
        source.scope = scope;
        f.write(0x65);
        let image = stage_with(&mut f.state, &mut f.host, &source, &COUNTERS, fill).unwrap();
        assert!(!Arc::ptr_eq(&first, &image));
        assert_eq!(image[1], vec![0x65; 1024]);
        assert!(source
            .resource
            .with_rail_state(|held: &mut RetainedPlanar<Vec<Vec<u8>>>| held.is_empty(),)
            .unwrap());
    }
    let mut fresh = Fixture::new(PAGE_SHIFT_X86);
    let mut source = fresh.source();
    source.scope = None;
    stage_with(&mut fresh.state, &mut fresh.host, &source, &COUNTERS, fill).unwrap();
    assert_eq!(fresh.host.tracked_guest_write_sets(), 0);
}

#[test]
fn planar_snapshot_backing_and_layout_changes_cannot_hit() {
    let mut f = Fixture::new(PAGE_SHIFT_X86);
    let first = f.capture();
    f.host
        .map_range(f.state.pfn_gpa(0x21), f.state.page_size() as usize, 0x68);
    assert!(f.state.invalidate_mapping_pages(5));
    f.state.mappings.get_mut(&5).unwrap().page_entries =
        vec![(0x21 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    let remapped = f.capture();
    assert!(!Arc::ptr_eq(&first, &remapped));
    assert_eq!(remapped[0], vec![0x68; 1024]);
    assert!(f
        .state
        .set_mapping_device_desc(5, &device_descriptor(BackingFormat::FullRange)));
    let changed_layout = f.capture();
    assert!(!Arc::ptr_eq(&remapped, &changed_layout));
    f.state.mappings.get_mut(&5).unwrap().content_generation += 1;
    assert!(!Arc::ptr_eq(&changed_layout, &f.capture()));
}

#[test]
fn planar_snapshot_unobservable_guest_writes_never_reuse() {
    let mut f = Fixture::new(PAGE_SHIFT_X86);
    f.host.guest_writes_unobservable = true;
    let first = f.capture();
    f.write(0x66);
    let second = f.capture();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(second[0], vec![0x66; 1024]);
}

#[test]
fn planar_snapshot_failed_fill_discards_candidate_but_preserves_submitted_copy() {
    let mut f = Fixture::new(PAGE_SHIFT_X86);
    let first = f.capture();
    f.host.guest_wrote_page(f.state.pfn_gpa(0x20));
    let source = f.source();
    let result =
        stage_with::<Vec<Vec<u8>>, _>(&mut f.state, &mut f.host, &source, &COUNTERS, |_, _, _| {
            Err(ComputeStatus::GuestIo("planar_mapping_read"))
        });
    assert!(matches!(
        result,
        Err(ComputeStatus::GuestIo("planar_mapping_read"))
    ));
    assert_eq!(Arc::strong_count(&first), 1);
    assert_eq!(first[1], vec![0x5a; 1024]);
}

#[test]
fn planar_snapshot_mutation_or_scope_end_during_fill_cannot_publish_candidate() {
    for mutate in [false, true] {
        let mut f = Fixture::new(PAGE_SHIFT_ARM64E);
        let source = f.source();
        let mut scope = Some(f.scope);
        stage_with(
            &mut f.state,
            &mut f.host,
            &source,
            &COUNTERS,
            |state, host, source| {
                let result = fill(state, host, source)?;
                if mutate {
                    host.guest_wrote_page(state.pfn_gpa(0x20));
                } else {
                    drop(scope.take());
                }
                Ok(result)
            },
        )
        .unwrap();
        assert!(source
            .resource
            .with_rail_state(|held: &mut RetainedPlanar<Vec<Vec<u8>>>| held.is_empty(),)
            .unwrap());
    }
}

#[test]
fn planar_snapshot_resource_retirement_drops_cache_not_inflight_bytes() {
    for reset in [false, true] {
        let mut f = Fixture::new(PAGE_SHIFT_X86);
        let image = f.capture();
        assert_eq!(Arc::strong_count(&image), 2);
        if reset {
            f.state.reset();
        } else {
            assert!(f.state.delete_object(1, 11));
        }
        assert_eq!(Arc::strong_count(&image), 1);
        assert_eq!(image[0], vec![0x5a; 1024]);
    }
}

#[cfg(feature = "backend-vulkan")]
#[test]
fn vulkan_planar_snapshot_keeps_exact_expansion_and_fresh_compute_staging() {
    use crate::runtime::compute_exec::{stage_texture_raw_in_scope, vulkan::VulkanStage};
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        let mut f = Fixture::new(shift);
        let stage = |f: &mut Fixture, scoped: bool| {
            let scope = scoped.then(|| f.scope.reference());
            stage_texture_raw_in_scope::<VulkanStage, _>(
                &mut f.state,
                &mut f.host,
                1,
                11,
                0,
                false,
                scope.as_ref(),
            )
            .unwrap()
            .rail
            .planar
            .unwrap()
        };
        let first = stage(&mut f, true);
        assert!(Arc::ptr_eq(&first, &stage(&mut f, true)));
        let compute = stage(&mut f, false);
        assert!(!Arc::ptr_eq(&first, &compute));
        assert_eq!(first.bytes, compute.bytes);
        f.write(0x67);
        f.host.guest_wrote_page(f.state.pfn_gpa(0x20));
        let fresh = stage(&mut f, true);
        assert_ne!(first.bytes, fresh.bytes);
        let checked = stage(&mut f, false);
        assert_eq!(fresh.bytes, checked.bytes);
    }
}
