use super::*;
use crate::backend::metal::planar::tests::texture_descriptor;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::protocol::iosurface_pages::*;
use crate::protocol::pixel_format::MTL_FORMAT_RGBA8_UNORM;
use crate::protocol::planar::SampleFormat;
use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;

const MID: u32 = 5;
const REF: u32 = 11;
const PAGE: usize = 1 << PAGE_SHIFT_ARM64E;
const GPA: u64 = 9 << PAGE_SHIFT_ARM64E;
const BASE: usize = 64;
const PITCH: usize = 48;
const PIXEL: [u8; 4] = [11, 37, 89, 255];

struct Fixture {
    state: DeviceState,
    host: FakeHost,
    base: usize,
    descriptor: Vec<u8>,
    scope: crate::runtime::draw::BufferSnapshotScope,
}

impl Fixture {
    fn new() -> Self {
        crate::runtime::guest_ram_map::reset();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 16);
        assert!(state.set_object_list(1, 0, 32));
        assert!(state.map_surface(MID));
        let mapping = state.mappings.get_mut(&MID).unwrap();
        mapping.mapping_internal = 1;
        mapping.page_entries = vec![(9 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        let mut device = vec![0; DEVICE_DESC_LEN];
        st32(
            &mut device[DEVICE_DESC_PIXEL_FORMAT..],
            u32::from(MTL_FORMAT_BGRA8_UNORM),
        );
        st32(&mut device[DEVICE_DESC_BASE_OFFSET..], BASE as u32);
        st32(&mut device[DEVICE_DESC_ALLOC_SIZE..], PAGE as u32);
        st64(
            &mut device[DEVICE_DESC_DIMS..],
            0x80 | (8 << 8) | (0x80 << 32) | (4 << 40),
        );
        st32(&mut device[DEVICE_DESC_BPR..], PITCH as u32);
        st16(&mut device[DEVICE_DESC_BPE..], 4);
        assert!(state.set_mapping_device_desc(MID, &device));
        assert!(state.set_mapping_geom(MID, 8, 4, MTL_FORMAT_BGRA8_UNORM));
        let mut descriptor = texture_descriptor(SampleFormat::Rgb10_420TwoPlane).to_vec();
        st16(&mut descriptor[22..], MTL_FORMAT_BGRA8_UNORM);
        let mut f = Self {
            state,
            host,
            base: BASE,
            descriptor,
            scope: crate::runtime::draw::BufferSnapshotScope::new(),
        };
        f.write_descriptor();
        f.write(PIXEL);
        f
    }

    fn write_descriptor(&mut self) {
        write_task_gva_arm64e(
            &mut self.host,
            &self.state.tasks[1],
            0x400,
            &self.descriptor,
        );
        let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | ((self.descriptor.len() as u32) << 8),
        );
        st64(&mut entry[4..], 0x400);
        write_task_gva_arm64e(
            &mut self.host,
            &self.state.tasks[1],
            list_object_entry_offset(REF, 32).unwrap(),
            &entry,
        );
    }

    fn write_at(&mut self, gpa: u64, pixel: [u8; 4]) {
        for y in 0..4 {
            self.host
                .write_gpa(gpa + (self.base + y * PITCH) as u64, &pixel.repeat(8))
                .unwrap();
        }
    }

    fn write(&mut self, pixel: [u8; 4]) {
        self.write_at(GPA, pixel);
    }

    fn source(&mut self) -> (Arc<TaskResource>, Description) {
        let resource = objects::resolve_resource(&self.state, &self.host, 1, REF).unwrap();
        assert_eq!(
            objects::resolve_mapper_ref_texture_resource(&mut self.state, 1, REF, &resource),
            Some(MID),
        );
        let description = Description::decode(&resource, REF).unwrap();
        (resource, description)
    }

    fn load(&mut self) -> Arc<packed::SampledImage> {
        native(
            super::super::load_in_scope(
                &mut self.state,
                &mut self.host,
                1,
                REF,
                Some(&self.scope.reference()),
            )
            .unwrap(),
        )
    }

    fn checked(&mut self, should_stage: bool) -> Arc<packed::SampledImage> {
        let (resource, description) = self.source();
        let mut staged = false;
        let image = load_with(
            &mut self.state,
            &mut self.host,
            &resource,
            description,
            Some(&self.scope.reference()),
            |state, host| {
                assert!(
                    should_stage,
                    "a hit must not enter ordinary staging, conversion, comparison or upload"
                );
                staged = true;
                super::super::load_rgba_packed(state, host, 1, REF)
            },
        )
        .unwrap();
        assert_eq!(staged, should_stage);
        native(image)
    }

    fn has_proof(&mut self) -> bool {
        let (resource, _) = self.source();
        resource
            .with_rail_state(|held: &mut RetainedPacked| held.mapping.is_some())
            .unwrap()
    }
}

fn native(upload: SampledUpload) -> Arc<packed::SampledImage> {
    match upload {
        SampledUpload::ImmutablePacked(image) => image,
        _ => panic!("a packed texture retains an immutable native image"),
    }
}

fn imported_fixture() -> (Fixture, u32) {
    let mut f = Fixture::new();
    f.host.owned_map_pages = true;
    crate::backend::metal::guest_writeback::publish_import_limits();
    let device = crate::backend::metal::runtime::system_device().unwrap();
    let alignment = device
        .minimum_linear_texture_alignment_for_pixel_format(::metal::MTLPixelFormat::BGRA8Unorm);
    let pitch = (32u64.div_ceil(alignment) * alignment) as u32;
    let m = f.state.mappings.get_mut(&MID).unwrap();
    st32(
        &mut m.device_desc[DEVICE_DESC_BASE_OFFSET..],
        alignment as u32,
    );
    st32(&mut m.device_desc[DEVICE_DESC_BPR..], pitch);
    for row in 0..4u64 {
        f.host
            .write_gpa(GPA + alignment + row * u64::from(pitch), &PIXEL.repeat(8))
            .unwrap();
    }
    (f, pitch)
}

#[test]
fn direct_mapped_sample_redeems_layout_and_retires_before_alias_unmap() {
    let (mut f, _) = imported_fixture();
    let upload = super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, None).unwrap();
    let SampledUpload::Imported(read) = &upload else {
        panic!("owned import must be admitted");
    };
    read.revalidate(&mut f.state, &mut f.host).unwrap();
    let binding = upload.image(0);
    let before = f.host.unmap_pages_calls;
    assert!(f.state.invalidate_mapping_pages(MID));
    assert!(read.check(&f.state).is_err());
    mapper::flush_retired_views(&mut f.state, &mut f.host);
    assert_eq!(f.host.unmap_pages_calls, before);
    drop(upload);
    assert_eq!(
        f.host.unmap_pages_calls, before,
        "native binding still holds the import"
    );
    drop(binding);
    mapper::flush_retired_views(&mut f.state, &mut f.host);
    assert_eq!(f.host.unmap_pages_calls, before + 1);
}

#[test]
fn direct_mapped_sample_refuses_source_layout_change_and_unsupported_alignment() {
    let (mut f, pitch) = imported_fixture();
    let upload = super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, None).unwrap();
    let SampledUpload::Imported(read) = &upload else {
        panic!("owned import must be admitted");
    };
    st32(
        &mut f.state.mappings.get_mut(&MID).unwrap().device_desc[DEVICE_DESC_BPR..],
        pitch + 1,
    );
    assert!(read.revalidate(&mut f.state, &mut f.host).is_err());
    drop(upload);
    let fallback = super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, None).unwrap();
    assert!(matches!(fallback, SampledUpload::ImmutablePacked(_)));
    f.state.invalidate_mapping_pages(MID);
    mapper::flush_retired_views(&mut f.state, &mut f.host);
}

#[test]
fn direct_mapped_sample_revalidates_backing_ptes_before_encoding() {
    let mut f = backing_fixture();
    f.host.owned_map_pages = true;
    crate::backend::metal::guest_writeback::publish_import_limits();
    let device = crate::backend::metal::runtime::system_device().unwrap();
    let alignment = device
        .minimum_linear_texture_alignment_for_pixel_format(::metal::MTLPixelFormat::BGRA8Unorm);
    let pitch = 32u64.div_ceil(alignment) * alignment;
    st32(
        &mut f.state.mappings.get_mut(&MID).unwrap().device_desc[DEVICE_DESC_BPR..],
        pitch as u32,
    );
    let upload = super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, None).unwrap();
    let SampledUpload::Imported(read) = &upload else {
        panic!("owned backing import");
    };
    let generation = f.state.mappings[&MID].map_generation;
    f.host.map_range(22 << PAGE_SHIFT_ARM64E, PAGE, 0);
    f.host
        .write_gpa((3 << PAGE_SHIFT_ARM64E) + 5 * 4, &22u32.to_le_bytes())
        .unwrap();
    assert_eq!(f.state.mappings[&MID].map_generation, generation);
    assert!(read.revalidate(&mut f.state, &mut f.host).is_err());
    assert!(
        !read.image.live(),
        "the old imported identity is retired, not re-pointed"
    );
    drop(upload);
    mapper::flush_retired_views(&mut f.state, &mut f.host);
}

#[test]
fn direct_mapped_sample_observes_locked_cpu_iosurface_updates_after_completed_commands() {
    use crate::backend::metal::planar::tests::cpu_surface::{CpuSurface, Sampler};
    objc::rc::autoreleasepool(|| {
        let mut source = CpuSurface::packed();
        let sampler = Sampler::new();
        let mut f = Fixture::new();
        f.host.owned_map_pages = true;
        crate::backend::metal::guest_writeback::publish_import_limits();
        let m = f.state.mappings.get_mut(&MID).unwrap();
        st32(&mut m.device_desc[DEVICE_DESC_BASE_OFFSET..], 0);
        st32(&mut m.device_desc[DEVICE_DESC_BPR..], source.pitch() as u32);
        let device = crate::backend::metal::runtime::system_device().unwrap();
        for round in 0..256u16 {
            let pixel = [round as u8, (round * 13) as u8, (255 - round) as u8, 255];
            let bytes = source.packed_pixels(pixel);
            let expected = sampler.sample(source.texture());
            f.host.write_gpa(GPA, &bytes).unwrap();
            let upload =
                super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, None).unwrap();
            assert!(matches!(&upload, SampledUpload::Imported(_)));
            super::super::validate_imported(
                &mut f.state,
                &mut f.host,
                [std::slice::from_ref(&upload), &[]],
            )
            .unwrap();
            let ReimsVgpuSampledImage::ImportedRead { image, .. } = upload.image(0) else {
                panic!();
            };
            assert_eq!(
                sampler.sample(image.texture(device).unwrap()),
                expected,
                "round={round}"
            );
        }
        f.state.invalidate_mapping_pages(MID);
        mapper::flush_retired_views(&mut f.state, &mut f.host);
    });
}

fn rgba([b, g, r, a]: [u8; 4]) -> Vec<u8> {
    [r, g, b, a].repeat(32)
}

#[test]
fn packed_mapping_clean_hit_skips_staging_without_publishing_a_cpu_frame() {
    let mut f = Fixture::new();
    let epochs = (
        f.state.mappings[&MID].content_generation,
        f.state.mappings[&MID].surface_content_epoch,
    );
    let first = f.checked(true);
    for _ in 0..128 {
        assert!(Arc::ptr_eq(&first, &f.checked(false)));
    }
    assert_eq!(first.bytes(), rgba(PIXEL));
    assert_eq!(
        (
            f.state.mappings[&MID].content_generation,
            f.state.mappings[&MID].surface_content_epoch
        ),
        epochs,
        "sampling CPU bytes is not a frame publication",
    );
    assert!(sampled_surface_frame(&f.state, &f.host, MID, None).is_none());
}

#[test]
fn packed_mapping_completed_commands_refresh_cpu_iosurface_updates_without_a_harvest() {
    use crate::backend::metal::planar::tests::cpu_surface::{CpuSurface, Sampler};
    objc::rc::autoreleasepool(|| {
        let mut source = CpuSurface::packed();
        let sampler = Sampler::new();
        let mut f = Fixture::new();
        let m = f.state.mappings.get_mut(&MID).unwrap();
        st32(&mut m.device_desc[DEVICE_DESC_BASE_OFFSET..], 0);
        st32(&mut m.device_desc[DEVICE_DESC_BPR..], source.pitch() as u32);
        let epochs = (
            m.content_generation,
            m.surface_content_epoch,
            m.map_generation,
        );
        for round in 0..256u16 {
            f.scope = crate::runtime::draw::BufferSnapshotScope::new();
            let seed = source.seed();
            let pixel = [round as u8, (round * 13) as u8, (255 - round) as u8, 255];
            let bytes = source.packed_pixels(pixel);
            assert_ne!(source.seed(), seed);
            let oracle = sampler.sample(source.texture());
            // Model the guest CPU's locked/unlocked bytes, with no fabricated
            // device write or dirty-harvest notification.
            f.host.write_gpa(GPA, &bytes).unwrap();
            let snapshot = f.load();
            let binding = SampledUpload::ImmutablePacked(snapshot).image(0);
            let ReimsVgpuSampledImage::Resident { image, .. } = binding else {
                panic!();
            };
            let actual = sampler.sample(image.texture());
            assert_eq!(actual, oracle, "completed packed command round={round}");
            let m = &f.state.mappings[&MID];
            assert_eq!(
                (
                    m.content_generation,
                    m.surface_content_epoch,
                    m.map_generation
                ),
                epochs
            );
        }
    });
}
#[test]
fn packed_mapping_guest_and_host_writes_independently_require_full_staging() {
    for guest in [true, false] {
        let mut f = Fixture::new();
        let first = f.load();
        let pixel = [17, 43, 91, 255];
        f.write(pixel);
        if guest {
            f.host.guest_wrote_page(GPA);
        } else {
            f.state.note_host_wrote_pages(vec![GPA]);
        }
        let changed = f.checked(true);
        assert!(!Arc::ptr_eq(&first, &changed));
        assert_eq!(first.bytes(), rgba(PIXEL));
        assert_eq!(changed.bytes(), rgba(pixel));
        assert!(Arc::ptr_eq(&changed, &f.checked(false)));
        f.state.note_host_wrote_pages(vec![GPA + PAGE as u64]);
        assert!(
            Arc::ptr_eq(&changed, &f.checked(false)),
            "disjoint writes are not overlap"
        );
        f.state.note_host_wrote_guest_ram();
        f.checked(true);
    }
}

#[test]
fn completed_gpu_publications_cannot_override_later_locked_cpu_pixels() {
    use crate::backend::metal::planar::tests::cpu_surface::{CpuSurface, Sampler};
    use crate::backend::metal::{resident, runtime::system_device};
    use crate::runtime::mapping_write::{write_rgba8_image_changed, FramePublication};
    objc::rc::autoreleasepool(|| {
        let device = system_device().unwrap();
        let sampler = Sampler::new();
        for publication in [FramePublication::HostCache, FramePublication::RailResident] {
            let mut source = CpuSurface::packed();
            let mut f = Fixture::new();
            let m = f.state.mappings.get_mut(&MID).unwrap();
            st32(&mut m.device_desc[DEVICE_DESC_BASE_OFFSET..], 0);
            st32(&mut m.device_desc[DEVICE_DESC_BPR..], source.pitch() as u32);
            resident::forget(MID);
            let key = resident::ResidentColorKey::for_surface(MID, 8, 4);
            let gpu =
                resident::create(device, &key, ::metal::MTLPixelFormat::RGBA8Unorm, 4).unwrap();
            for round in 0..256u16 {
                // Real completed GPU work, followed by the product's ordinary
                // completed-write publication owner. No frame stamp is forged.
                sampler.clear(&gpu);
                sampler.clear(source.texture());
                let seed = sampler.sample(&gpu);
                assert_eq!(seed, sampler.sample(source.texture()));
                let gpu_rgba: Vec<u8> = seed.iter().map(|v| (v * 255.0).round() as u8).collect();
                assert!(write_rgba8_image_changed(
                    &mut f.state,
                    &mut f.host,
                    MID,
                    &gpu_rgba,
                    None,
                    8,
                    4,
                    publication,
                ));
                let generation =
                    crate::runtime::surface_cache::frame_generation(&f.state, MID, 8, 4).unwrap();
                if publication == FramePublication::RailResident {
                    resident::retain_completed(key, &gpu);
                    resident::published(&key, generation);
                }
                let token = f.state.mappings[&MID].guest_write_token;
                let observed = f.host.guest_write_gen(token);
                let epochs = (
                    f.state.mappings[&MID].content_generation,
                    f.state.mappings[&MID].surface_content_epoch,
                );
                // Legal CPU update after completion. It is deliberately before
                // the next harvest and does not notify a device-side writer.
                let pixel = [round as u8, (round * 13) as u8, (255 - round) as u8, 255];
                let bytes = source.packed_pixels(pixel);
                let wanted = sampler.sample(source.texture());
                f.host.write_gpa(GPA, &bytes).unwrap();
                f.scope = crate::runtime::draw::BufferSnapshotScope::new();
                let upload = super::super::load_in_scope(
                    &mut f.state,
                    &mut f.host,
                    1,
                    REF,
                    Some(&f.scope.reference()),
                )
                .unwrap();
                let ReimsVgpuSampledImage::Resident { image, .. } = upload.image(0) else {
                    panic!("packed upload owns a read-only native snapshot");
                };
                assert_eq!(
                    sampler.sample(image.texture()),
                    wanted,
                    "{publication:?} round={round}"
                );
                assert_eq!(
                    super::super::load_sampled_rgba(&mut f.state, &mut f.host, 1, REF)
                        .unwrap()
                        .2,
                    rgba(pixel),
                    "the ordinary fallback must not select the historical host-frame cache",
                );
                assert_eq!(
                    crate::runtime::draw::seed_color_load(
                        &mut f.state,
                        &mut f.host,
                        1,
                        REF,
                        0,
                        8,
                        4,
                    )
                    .unwrap(),
                    rgba(pixel),
                    "a LOAD must preserve the CPU-updated guest pixels too",
                );
                assert_eq!(f.host.guest_write_gen(token), observed);
                assert_eq!(
                    (
                        f.state.mappings[&MID].content_generation,
                        f.state.mappings[&MID].surface_content_epoch
                    ),
                    epochs,
                );
                assert_eq!(
                    crate::runtime::surface_cache::frame_generation(&f.state, MID, 8, 4),
                    Some(generation),
                    "historical publication identity remains available for capture",
                );
            }
            resident::forget(MID);
        }
    });
}
#[test]
fn packed_mapping_expired_or_absent_scope_keeps_exact_byte_staging_fallback() {
    let mut f = Fixture::new();
    f.load();
    let expired = f.scope.reference();
    f.scope = crate::runtime::draw::BufferSnapshotScope::new();
    assert!(expired.current().is_none());
    for scope in [Some(&expired), None] {
        let pixel = if scope.is_some() {
            [19, 41, 73, 255]
        } else {
            [29, 51, 83, 255]
        };
        f.write(pixel);
        let image =
            native(super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, scope).unwrap());
        assert_eq!(image.bytes(), rgba(pixel));
        assert!(!f.has_proof());
        let same =
            native(super::super::load_in_scope(&mut f.state, &mut f.host, 1, REF, scope).unwrap());
        assert!(
            Arc::ptr_eq(&image, &same),
            "full-byte upload reuse remains available"
        );
    }
}

#[test]
fn packed_mapping_scope_ending_during_fill_cannot_retain_read_elision() {
    let mut f = Fixture::new();
    let (resource, description) = f.source();
    let mut owner = Some(crate::runtime::draw::BufferSnapshotScope::new());
    let scope = owner.as_ref().unwrap().reference();
    let image = load_with(
        &mut f.state,
        &mut f.host,
        &resource,
        description,
        Some(&scope),
        |state, host| {
            let image = super::super::load_rgba_packed(state, host, 1, REF);
            drop(owner.take());
            image
        },
    )
    .unwrap();
    assert_eq!(native(image).bytes(), rgba(PIXEL));
    assert!(scope.current().is_none());
    assert!(!f.has_proof());
}

#[test]
fn packed_mapping_usage_capabilities_do_not_replace_actual_writer_witnesses() {
    use crate::protocol::texture_shape::TextureUsage;

    for usage in [
        TextureUsage::UNKNOWN,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureUsage::DECLARED,
    ] {
        for guest in [false, true] {
            let mut f = Fixture::new();
            f.descriptor[21] = u8::try_from(usage.0).unwrap();
            f.write_descriptor();
            let first = f.checked(true);
            assert!(Arc::ptr_eq(&first, &f.checked(false)));
            let pixel = [29, 53, 97, 255];
            f.write(pixel);
            if guest {
                f.host.guest_wrote_page(GPA);
            } else {
                f.state.note_host_wrote_pages(vec![GPA]);
            }
            let changed = f.checked(true);
            assert_eq!(first.bytes(), rgba(PIXEL));
            assert_eq!(changed.bytes(), rgba(pixel));
            assert!(!Arc::ptr_eq(&first, &changed));
            assert!(Arc::ptr_eq(&changed, &f.checked(false)));
        }
    }
}

#[test]
fn packed_mapping_unknown_tracking_keeps_full_staging_and_exact_byte_upload_reuse() {
    for startup in [false, true] {
        let mut f = Fixture::new();
        f.host.guest_writes_unobservable = !startup;
        f.host.guest_write_startup_window = startup;
        let first = f.checked(true);
        assert!(
            Arc::ptr_eq(&first, &f.checked(true)),
            "upload-only reuse survives fallback"
        );
        assert!(!f.has_proof());
        let pixel = [21, 47, 101, 255];
        f.write(pixel);
        let changed = f.checked(true);
        assert_eq!(changed.bytes(), rgba(pixel));
        assert!(!Arc::ptr_eq(&first, &changed));
        assert!(!f.has_proof());
        if startup {
            f.host.guest_wrote_page(GPA);
            f.checked(true);
            f.checked(false);
        }
    }
}

#[test]
fn packed_mapping_map_generation_layout_and_interpretation_cannot_inherit_a_proof() {
    let mut f = Fixture::new();
    let first = f.load();
    assert!(f.state.invalidate_mapping_pages(MID));
    f.state.mappings.get_mut(&MID).unwrap().page_entries =
        vec![(9 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert!(
        Arc::ptr_eq(&first, &f.checked(true)),
        "only full-byte comparison may reuse here"
    );
    f.checked(false);
    st32(
        &mut f.state.mappings.get_mut(&MID).unwrap().device_desc[DEVICE_DESC_ALLOC_SIZE..],
        (PAGE / 2) as u32,
    );
    f.checked(true);
    f.checked(false);
    f.state.mappings.get_mut(&MID).unwrap().format = MTL_FORMAT_RGBA8_UNORM;
    st32(
        &mut f.state.mappings.get_mut(&MID).unwrap().device_desc[DEVICE_DESC_PIXEL_FORMAT..],
        u32::from(MTL_FORMAT_RGBA8_UNORM),
    );
    let reinterpreted = f.checked(true);
    assert_eq!(reinterpreted.bytes(), PIXEL.repeat(32));
    assert!(
        !f.has_proof(),
        "a format override is not a direct unreinterpreted sample"
    );
    f.checked(true);
    f.state.mappings.get_mut(&MID).unwrap().width = 4;
    st64(
        &mut f.state.mappings.get_mut(&MID).unwrap().device_desc[DEVICE_DESC_DIMS..],
        0x80 | (4 << 8) | (0x80 << 32) | (4 << 40),
    );
    let resized = f.checked(true);
    assert_eq!(resized.layout().width, 4);
    assert!(!f.has_proof());
}

#[test]
fn packed_mapping_observed_mutation_during_fill_never_retains_read_elision() {
    for guest in [false, true] {
        let mut f = Fixture::new();
        let (resource, description) = f.source();
        let image = native(
            load_with(
                &mut f.state,
                &mut f.host,
                &resource,
                description,
                Some(&f.scope.reference()),
                |state, host| {
                    let image = super::super::load_rgba_packed(state, host, 1, REF);
                    for y in 0..4 {
                        host.write_gpa(
                            GPA + (BASE + y * PITCH) as u64,
                            &[7, 23, 71, 255].repeat(8),
                        )
                        .unwrap();
                    }
                    if guest {
                        host.guest_wrote_page(GPA);
                    } else {
                        state.note_host_wrote_pages(vec![GPA]);
                    }
                    image
                },
            )
            .unwrap(),
        );
        assert_eq!(image.bytes(), rgba(PIXEL));
        assert!(!f.has_proof());
        let fresh = f.checked(true);
        assert_eq!(fresh.bytes(), rgba([7, 23, 71, 255]));
        assert!(Arc::ptr_eq(&fresh, &f.checked(false)));
    }
}

#[test]
fn packed_mapping_audit_reads_guest_bytes_and_revokes_an_escaping_writer() {
    let mut f = Fixture::new();
    let first = f.load();
    for bind in 0..=2 * crate::runtime::gather_witness::AUDIT_STRIDE {
        let pixel = [7 + (bind % 2) as u8, 23, 71, 255];
        f.write(pixel); // deliberately escape both writer records
        let image = f.load();
        if !Arc::ptr_eq(&first, &image) {
            assert_eq!(image.bytes(), rgba(pixel));
            return;
        }
    }
    panic!("the actual guest-byte audit never revoked a stale read-elision proof");
}

#[test]
fn packed_mapping_audit_byte_accounting_includes_padding_only_when_folded() {
    let mut f = Fixture::new();
    let (_, description) = f.source();
    let window = Window::resolve(&mut f.state, &mut f.host, description)
        .unwrap()
        .unwrap();
    assert_eq!(window.key.guest_read_bytes, 128);
    assert_eq!(window.key.end - window.key.base, 176);
    let mut folded = 0;
    for _ in 0..=2 * crate::runtime::gather_witness::AUDIT_STRIDE + 4 {
        let seen = window.observe(&mut f.state, &mut f.host, Some(&f.scope.reference()));
        if seen.audit_bytes != 0 {
            assert_eq!(seen.audit_bytes, 176);
            folded += 1;
        }
    }
    assert!(
        folded >= 2,
        "both baseline and comparison report their real byte reads"
    );
}

#[test]
fn packed_mapping_miss_counts_both_pre_and_post_audit_reads_as_extra() {
    use crate::runtime::drain::store_route_count;

    for prime in [0, 1] {
        let mut f = Fixture::new();
        let (resource, description) = f.source();
        let window = Window::resolve(&mut f.state, &mut f.host, description)
            .unwrap()
            .unwrap();
        let mut control = Fixture::new();
        let (_, control_description) = control.source();
        let control_window =
            Window::resolve(&mut control.state, &mut control.host, control_description)
                .unwrap()
                .unwrap();
        for _ in 0..prime {
            window.observe(&mut f.state, &mut f.host, Some(&f.scope.reference()));
            control_window.observe(
                &mut control.state,
                &mut control.host,
                Some(&control.scope.reference()),
            );
        }
        let initial = store_route_count("metal_packed_mapping_extra_audit_bytes");
        let mut expected = 0;
        for _ in 0..2 * crate::runtime::gather_witness::AUDIT_STRIDE + 2 {
            resource.with_rail_state(|held: &mut RetainedPacked| held.mapping = None);
            f.checked(true);
            for _ in 0..2 {
                expected += control_window
                    .observe(
                        &mut control.state,
                        &mut control.host,
                        Some(&control.scope.reference()),
                    )
                    .audit_bytes;
            }
        }
        assert!(expected > 0);
        assert_eq!(
            store_route_count("metal_packed_mapping_extra_audit_bytes") - initial,
            expected,
        );
    }
}

#[test]
fn packed_mapping_half_float_reuse_counts_guest_bytes_not_converted_rgba_bytes() {
    let mut f = Fixture::new();
    let format = pixel_format::MTL_FORMAT_RGBA16_FLOAT;
    st16(&mut f.descriptor[22..], format);
    f.write_descriptor();
    let mapping = f.state.mappings.get_mut(&MID).unwrap();
    mapping.format = format;
    st32(
        &mut mapping.device_desc[DEVICE_DESC_PIXEL_FORMAT..],
        u32::from(format),
    );
    st32(&mut mapping.device_desc[DEVICE_DESC_BPR..], 80);
    st16(&mut mapping.device_desc[DEVICE_DESC_BPE..], 8);
    let pixel: Vec<_> = [0x3400u16, 0x3800, 0x3a00, 0x3c00]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    for row in 0..4 {
        f.host
            .write_gpa(GPA + (BASE + row * 80) as u64, &pixel.repeat(8))
            .unwrap();
    }
    let first = f.checked(true);
    assert_eq!(first.bytes(), [64, 128, 191, 255].repeat(32));
    assert!(Arc::ptr_eq(&first, &f.checked(false)));
    let (_, description) = f.source();
    let window = Window::resolve(&mut f.state, &mut f.host, description)
        .unwrap()
        .unwrap();
    assert_eq!(window.key.guest_read_bytes, 256);
    assert_eq!(window.key.end - window.key.base, 304);
    assert_eq!(
        first.byte_len(),
        128,
        "converted bytes are not the guest-read footprint"
    );
}

#[test]
fn packed_mapping_unknown_shapes_and_unverified_backing_provenance_fall_back() {
    for change in [0, 1, 2, 3] {
        let mut f = Fixture::new();
        match change {
            0 => st16(&mut f.descriptor[36..], 2), // serialized mip count
            1 => f.descriptor.resize(0x58, 0),     // undecoded extended record
            2 => st32(&mut f.descriptor[52..], 1), // an explicit plane view
            _ => f.descriptor[21] = 0x80,          // an undecoded usage bit
        }
        f.write_descriptor();
        assert!(super::load(
            &mut f.state,
            &mut f.host,
            1,
            REF,
            Some(&f.scope.reference())
        )
        .unwrap()
        .is_none());
        let image = f.load();
        assert_eq!(
            image.bytes(),
            rgba(PIXEL),
            "ordinary staging still owns these shapes"
        );
    }
    let mut f = Fixture::new();
    f.state.mappings.get_mut(&MID).unwrap().mapping_internal = 0;
    f.checked(true);
    f.checked(true);
    assert!(
        !f.has_proof(),
        "cached pages without a provenance check cannot vouch"
    );
}

#[test]
fn packed_mapping_deletion_drops_proof_and_cache_but_not_an_inflight_binding() {
    for reset in [false, true] {
        let mut f = Fixture::new();
        let image = f.load();
        assert!(f.has_proof());
        let weak = Arc::downgrade(&image);
        let binding = SampledUpload::ImmutablePacked(Arc::clone(&image)).image(3);
        if reset {
            f.state.reset();
        } else {
            assert!(f.state.delete_object(1, REF));
        }
        assert_eq!(Arc::strong_count(&image), 2);
        drop(image);
        assert!(weak.upgrade().is_some());
        drop(binding);
        assert!(weak.upgrade().is_none());
        mapper::flush_retired_views(&mut f.state, &mut f.host);
    }
}

fn backing_fixture() -> Fixture {
    let mut f = Fixture::new();
    // This fixture's stable-import arm only aliases a single contiguous RAM
    // range. Enable its owned remap aliases for the deliberately scattered PTEs.
    f.host.stable_map_pages = false;
    f.state.mappings.get_mut(&MID).unwrap().mapping_internal = 0;
    let mut backing = vec![0u8; 0x30];
    st64(&mut backing, (3 * PAGE) as u64);
    st32(&mut backing[8..], 5); // task GVA pages 5..8, not host pointers
    st32(&mut backing[12..], u32::from_be_bytes(*b"BGRA"));
    backing[16] = 1;
    st32(&mut backing[24..], 8);
    st32(&mut backing[28..], 4);
    st32(&mut backing[32..], PITCH as u32);
    write_task_gva_arm64e(&mut f.host, &f.state.tasks[1], 0x200, &backing);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        u32::from(objects::OBJECT_TYPE_BACKING) | ((backing.len() as u32) << 8),
    );
    st64(&mut entry[4..], 0x200);
    write_task_gva_arm64e(
        &mut f.host,
        &f.state.tasks[1],
        list_object_entry_offset(MID, 32).unwrap(),
        &entry,
    );
    assert!(objects::resolve_backing_force(&mut f.state, &f.host, MID));
    assert!(f.state.mappings[&MID].backing_walk.is_some());
    f.base = 0;
    f.write(PIXEL);
    f
}

#[test]
fn packed_mapping_backing_pte_rewires_revalidate_every_page_and_rebuild_before_sampling() {
    let mut f = backing_fixture();
    let first = f.checked(true);
    f.checked(false);
    let before = f.state.mappings[&MID].map_generation;
    let root = 3u64 << PAGE_SHIFT_ARM64E;
    f.host.map_range(20 << PAGE_SHIFT_ARM64E, PAGE, 0);
    f.host
        .write_gpa(root + 6 * 4, &20u32.to_le_bytes())
        .unwrap();
    assert_eq!(
        f.state.mappings[&MID].map_generation, before,
        "no mapping packet arrived"
    );
    f.checked(true); // even an unsampled middle allocation page changed
    assert_ne!(f.state.mappings[&MID].map_generation, before);
    f.checked(false);
    f.host.map_range(21 << PAGE_SHIFT_ARM64E, PAGE, 0);
    f.write_at(21 << PAGE_SHIFT_ARM64E, [19, 29, 139, 255]);
    f.host
        .write_gpa(root + 5 * 4, &21u32.to_le_bytes())
        .unwrap();
    let changed = f.checked(true);
    assert!(!Arc::ptr_eq(&first, &changed));
    assert_eq!(changed.bytes(), rgba([19, 29, 139, 255]));
    f.checked(false);
    f.host.write_gpa(root + 6 * 4, &0u32.to_le_bytes()).unwrap();
    assert!(
        super::load(
            &mut f.state,
            &mut f.host,
            1,
            REF,
            Some(&f.scope.reference())
        )
        .is_err(),
        "a failed backing revalidation cannot silently sample a retired alias"
    );
    f.host
        .write_gpa(root + 6 * 4, &20u32.to_le_bytes())
        .unwrap();
    let recovered = f.checked(true);
    assert_eq!(recovered.bytes(), rgba([19, 29, 139, 255]));
    f.checked(false);
    f.state.mappings.get_mut(&MID).unwrap().mapped = false;
    assert!(super::load(
        &mut f.state,
        &mut f.host,
        1,
        REF,
        Some(&f.scope.reference())
    )
    .is_err());
    assert!(
        !f.state.mappings[&MID].mapped,
        "read elision must not resurrect an unmapped surface"
    );
}
