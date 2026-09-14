use super::*;
use crate::backend::blob::BlobKey;
use crate::backend::metal::{cache, guest_writeback, raw_metal, resident, runtime};
use crate::protocol::endian::{st16, st32, st64};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::protocol::pass_action::MTL_STORE_ACTION_STORE;
use crate::protocol::pixel_format;
use crate::runtime::decode::resource::*;
use crate::runtime::draw::Color0Readback;
use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
use crate::runtime::host::{FakeHost, HostMemory, HostOps};

const VERTEX: &[u8] = b"owned-gpu-store-vertex";
const FRAGMENT: &[u8] = b"owned-gpu-store-fragment";

struct Fixture {
    state: DeviceState,
    host: FakeHost,
    mid: u32,
    page: u64,
}

impl Fixture {
    fn new(format: u16) -> Self {
        Self::with_dimensions(format, 4, 3)
    }

    fn with_dimensions(format: u16, width: u32, height: u32) -> Self {
        guest_writeback::publish_import_limits();
        crate::runtime::guest_ram_map::reset();
        let mut host = FakeHost::new();
        // Darwin's owned remaps are not VM-stable borrowed-run aliases.
        host.stable_map_pages = false;
        host.owned_map_pages = true;
        let mut state =
            DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
        define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));
        let mid = 501;
        let page = 0x100000;
        host.map_range(page, state.page_size() as usize, 0xca);
        state.map_surface(mid);
        state.set_mapping_geom(mid, width, height, format);
        let mapping = state.mappings.get_mut(&mid).unwrap();
        mapping.mapped = true;
        mapping.mapping_internal = 1;
        mapping.page_entries =
            vec![(((page >> state.page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];

        let device = runtime::system_device().unwrap();
        let library = raw_metal::new_library_with_source(
            device,
            r#"
            #include <metal_stdlib>
            using namespace metal;
            vertex float4 gpu_store_vertex(uint i [[vertex_id]]) {
                const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
                return float4(p[i], 0, 1);
            }
            fragment half4 gpu_store_fragment() { return half4(1, 0, 0, 1); }
        "#,
        )
        .unwrap();
        cache::fn_cache_insert(
            &BlobKey::new(VERTEX),
            library.get_function("gpu_store_vertex", None).unwrap(),
        );
        cache::fn_cache_insert(
            &BlobKey::new(FRAGMENT),
            library.get_function("gpu_store_fragment", None).unwrap(),
        );
        let mut pipeline = vec![0u8; 32];
        st32(&mut pipeline, SERIALIZER_OBJECT_RENDER_PIPELINE);
        st32(&mut pipeline[4..], 32);
        st32(&mut pipeline[8..], 1);
        st32(&mut pipeline[12..], 13);
        pipeline[SERIALIZER_OBJECT_FIRST_TLVS] = 2;
        pipeline[17] = PIPELINE_TAG_VERTEX_FUNC;
        pipeline[18] = 4;
        st32(&mut pipeline[19..], 2);
        pipeline[23] = PIPELINE_TAG_FRAGMENT_FUNC;
        pipeline[24] = 4;
        st32(&mut pipeline[25..], 3);
        let mut vertex = vec![0u8; FUNCTION_DESC_MIN_LEN];
        st64(&mut vertex[FUNCTION_DESC_BLOB_GVA..], 0x1000);
        st32(&mut vertex[FUNCTION_DESC_BLOB_SIZE..], VERTEX.len() as u32);
        let mut fragment = vec![0u8; FUNCTION_DESC_MIN_LEN];
        st64(&mut fragment[FUNCTION_DESC_BLOB_GVA..], 0x1100);
        st32(
            &mut fragment[FUNCTION_DESC_BLOB_SIZE..],
            FRAGMENT.len() as u32,
        );
        let mut texture = vec![0u8; 32];
        st32(&mut texture, mid);
        st16(&mut texture[0x16..], format);
        st32(&mut texture[0x18..], width);
        st32(&mut texture[0x1c..], height);
        for (reference, kind, gva, descriptor) in [
            (1, OBJECT_TYPE_SERIALIZER_OBJECT, 0x200, pipeline),
            (2, OBJECT_TYPE_FUNCTION, 0x300, vertex),
            (3, OBJECT_TYPE_FUNCTION, 0x340, fragment),
            (4, OBJECT_TYPE_MAPPER_REF_TEXTURE, 0x400, texture),
        ] {
            write_task_gva_arm64e(&mut host, &state.tasks[1], gva, &descriptor);
            let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            st32(
                &mut entry,
                u32::from(kind) | ((descriptor.len() as u32) << 8),
            );
            st64(&mut entry[4..], gva);
            write_task_gva_arm64e(
                &mut host,
                &state.tasks[1],
                list_object_entry_offset(reference, 32).unwrap(),
                &entry,
            );
        }
        write_task_gva_arm64e(&mut host, &state.tasks[1], 0x1000, VERTEX);
        write_task_gva_arm64e(&mut host, &state.tasks[1], 0x1100, FRAGMENT);
        Self {
            state,
            host,
            mid,
            page,
        }
    }

    fn request(&self, readback: Color0Readback) -> DrawEncodeRequest {
        DrawEncodeRequest {
            task_id: 1,
            pipeline_ref: 1,
            vertex_count: 3,
            instance_count: 1,
            primitive_type: 3,
            color0_readback: readback,
            colors: vec![ColorRtRequest {
                slot: 0,
                storage: ColorStorage::GuestBacked,
                texture_ref: 4,
                mapping_id: self.mid,
                width: self.state.mappings[&self.mid].width,
                height: self.state.mappings[&self.mid].height,
                sample_count: 1,
                format: self.state.mappings[&self.mid].format,
                load_action: MTL_LOAD_ACTION_CLEAR,
                store_action: MTL_STORE_ACTION_STORE,
                ..Default::default()
            }],
            ..Default::default()
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let views = self.state.take_all_host_views();
        self.state.retired_views.extend(views);
        crate::runtime::mapper::flush_retired_views(&mut self.state, &mut self.host);
        resident::forget(self.mid);
        crate::runtime::guest_ram_map::reset();
    }
}

#[test]
fn owned_darwin_remap_is_importable_without_legacy_stability_and_unmaps_after_metal_release() {
    let device = runtime::system_device().unwrap();
    if !device.has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
    let second = 0x300000;
    let page_size = fixture.state.page_size();
    fixture.host.map_range(second, page_size as usize, 0xdb);
    fixture
        .state
        .mappings
        .get_mut(&fixture.mid)
        .unwrap()
        .page_entries
        .push(
            (((second >> fixture.state.page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT)
                | PAGE_ENTRY_VALID,
        );
    assert!(!fixture.host.map_pages_stable());
    assert!(fixture.host.map_pages_owned());
    let maps = fixture.host.map_pages_calls;
    assert!(
        crate::runtime::mapper::ensure_contig_import_with_footprint(
            &mut fixture.state,
            &mut fixture.host,
            fixture.mid,
        )
        .is_none(),
        "legacy imports must not acquire the owned-view capability"
    );
    assert_eq!(fixture.host.map_pages_calls, maps);
    let mut pass = MetalRenderPass::default();
    let mut req = fixture.request(Color0Readback::Optional);
    req.render_pass_continues = true;
    assert!(matches!(
        pass.encode_draw(
            &mut fixture.state,
            &mut fixture.host,
            &mut req,
            false,
            false,
        )
        .0,
        EncodeStatus::Ok
    ));
    assert!(pass.target(&req.colors[0]).unwrap().gpu_store.is_some());
    let import = fixture.state.mappings[&fixture.mid]
        .contig_import
        .clone()
        .unwrap();
    let view = (import.host_base(), import.len() as usize);
    assert_eq!(view.1, page_size as usize * 2);
    assert!(
        fixture.host.owned_page_views().contains(&view),
        "a real mach_vm_remap, not a RAMBlock borrow"
    );
    let unmaps = fixture.host.unmap_pages_calls;
    fixture.state.invalidate_mapping_pages(fixture.mid);
    crate::runtime::mapper::flush_retired_views(&mut fixture.state, &mut fixture.host);
    assert!(import.is_retired());
    assert_eq!(fixture.host.unmap_pages_calls, unmaps);
    assert!(fixture.host.owned_page_views().contains(&view));
    objc::rc::autoreleasepool(|| drop(pass));
    assert_eq!(
        crate::runtime::mapper::drain_deferred_unmaps(&mut fixture.host),
        1
    );
    assert_eq!(fixture.host.unmap_pages_calls, unmaps + 1);
    assert!(
        !fixture.host.owned_page_views().contains(&view),
        "matching view is actually unmapped"
    );
}

#[test]
fn unknown_owned_view_contract_keeps_the_mapping_copy_backed() {
    let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
    fixture.host.owned_map_pages = false;
    let maps = fixture.host.map_pages_calls;
    assert!(
        crate::runtime::mapper::ensure_owned_contig_import_with_footprint(
            &mut fixture.state,
            &mut fixture.host,
            fixture.mid,
        )
        .is_none()
    );
    assert_eq!(fixture.host.map_pages_calls, maps);
}

#[test]
fn mapped_store_does_not_publish_when_its_render_producer_failed() {
    let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
    let mut request = fixture.request(Color0Readback::Optional);
    request.render_pass_continues = true;
    let mut pass = MetalRenderPass::default();
    assert!(matches!(
        pass.encode_draw(
            &mut fixture.state, &mut fixture.host, &mut request, false, false,
        ).0,
        EncodeStatus::Ok
    ));
    pass.flush_checked(&mut fixture.state, &mut fixture.host, "test_render_complete").unwrap();
    let epoch = fixture.state.mappings[&fixture.mid].surface_content_epoch;
    let mut store = pass.target_mut(&request.colors[0]).unwrap().gpu_store.take().unwrap();
    let result = store.finish_after(&mut fixture.state, &mut fixture.host, || {
        Err(crate::backend::metal::util::Status::execute("test_render_producer_failed"))
    });
    assert!(result.is_err());
    assert_eq!(fixture.state.mappings[&fixture.mid].surface_content_epoch, epoch);
}

#[test]
fn owned_darwin_remap_gpu_store_crosses_scattered_pages_before_publication() {
    let device = runtime::system_device().unwrap();
    if !device.has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::with_dimensions(pixel_format::MTL_FORMAT_BGRA8_UNORM, 64, 65);
        let second = 0x300000;
        let page_size = fixture.state.page_size();
        fixture.host.map_range(second, page_size as usize, 0xdb);
        fixture
            .state
            .mappings
            .get_mut(&fixture.mid)
            .unwrap()
            .page_entries
            .push(
                (((second >> fixture.state.page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT)
                    | PAGE_ENTRY_VALID,
            );
        let mut pass = MetalRenderPass::default();
        let mut req = fixture.request(Color0Readback::Optional);
        let result = pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, false);
        assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
        assert!(result.1.is_none());
        assert_eq!(pass.batch.borrow().readbacks, 0);
        assert!(!fixture.host.map_pages_stable());
        assert_eq!(
            fixture.host.owned_page_views().len(),
            1,
            "packed remap remains owned by the mapping"
        );
        let mapping = &fixture.state.mappings[&fixture.mid];
        let (base, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
            mapping,
            64,
            65,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .unwrap();
        assert_eq!(base, 0);
        assert_eq!(pitch, 256);
        let mut first = vec![0u8; page_size as usize];
        fixture.host.read_gpa(fixture.page, &mut first).unwrap();
        assert_eq!(first, [0, 0, 255, 255].repeat(page_size as usize / 4));
        let mut last = vec![0u8; page_size as usize];
        fixture.host.read_gpa(second, &mut last).unwrap();
        assert_eq!(&last[..256], [0, 0, 255, 255].repeat(64));
        assert!(last[256..].iter().all(|&byte| byte == 0xdb));
        assert!(matches!(
            crate::runtime::draw::published_mapping_frame(
                &fixture.state,
                &fixture.host,
                fixture.mid,
                64,
                65,
            ),
            Err(crate::runtime::draw::NoPublishedFrame::Unscoped(_))
        ));
        assert!(crate::runtime::surface_cache::frame_generation(
            &fixture.state,
            fixture.mid,
            64,
            65,
        )
        .is_some());
    });
}

#[test]
fn mapped_gpu_load_seeds_current_guest_pixels_in_the_render_submission() {
    use crate::runtime::drain::store_route_count;
    use crate::runtime::render_pass::ScissorRect;

    for format in [
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    ] {
        let mut fixture = Fixture::new(format);
        for background in [[11, 37, 89, 255], [73, 19, 131, 255]] {
            let (base, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
                &fixture.state.mappings[&fixture.mid],
                4,
                3,
                format,
            )
            .unwrap();
            let mut row = vec![0xca; pitch as usize];
            assert!(pixel_format::Rgba8ToRow::for_format(format)
                .unwrap()
                .convert(&background.repeat(4), 4, &mut row));
            for y in 0..3 {
                fixture
                    .host
                    .write_gpa(fixture.page + base + u64::from(pitch) * y, &row)
                    .unwrap();
            }
            let gpu_before = store_route_count("metal_seed_from_guest_gpu");
            let cpu_before = store_route_count("metal_seed_load_asked");
            let mut request = fixture.request(Color0Readback::Required);
            request.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
            request.scissors = vec![ScissorRect {
                x: 0,
                y: 0,
                width: 1,
                height: 3,
            }];
            let mut pass = MetalRenderPass::default();
            let (status, output) = pass.encode_draw(
                &mut fixture.state,
                &mut fixture.host,
                &mut request,
                true,
                false,
            );
            assert!(matches!(status, EncodeStatus::Ok), "{status:?}");
            let mut expected = background.repeat(12);
            for y in 0..3 {
                expected[y * 16..y * 16 + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
            assert_eq!(output.unwrap(), expected);
            assert_eq!(
                store_route_count("metal_seed_from_guest_gpu") - gpu_before,
                1
            );
            assert_eq!(store_route_count("metal_seed_load_asked") - cpu_before, 0);
            assert_eq!(pass.batch.borrow().submissions, 1);
        }
    }
}

#[test]
fn mapped_gpu_store_skips_readback_only_when_optional_and_publishes_current_resident() {
    let device = runtime::system_device().unwrap();
    if !device.has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    for format in [
        pixel_format::MTL_FORMAT_BGRA8_UNORM,
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    ] {
        for readback in [Color0Readback::Optional, Color0Readback::Required] {
            objc::rc::autoreleasepool(|| {
                let mut fixture = Fixture::new(format);
                let mut pass = MetalRenderPass::default();
                let mut req = fixture.request(readback);
                req.render_pass_continues = true;
                let first = pass.encode_draw(
                    &mut fixture.state,
                    &mut fixture.host,
                    &mut req,
                    false,
                    false,
                );
                assert!(matches!(first.0, EncodeStatus::Ok), "{:?}", first.0);
                assert!(first.1.is_none());
                assert!(pass.target(&req.colors[0]).unwrap().gpu_store.is_some());
                assert_eq!(pass.batch.borrow().submissions, 0);
                let mut before = [0u8; 16];
                fixture.host.read_gpa(fixture.page, &mut before).unwrap();
                assert_eq!(before, [0xca; 16], "recording a GPU Store writes nothing");
                req.render_pass_continues = false;
                req.continues_render_pass = true;
                req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
                let result =
                    pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, true);
                assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
                let rgba = [255u8, 0, 0, 255].repeat(12);
                assert_eq!(
                    result.1,
                    (readback == Color0Readback::Required).then(|| rgba.clone())
                );
                assert_eq!(
                    pass.batch.borrow().readbacks,
                    usize::from(readback == Color0Readback::Required)
                );
                let mapping = &fixture.state.mappings[&fixture.mid];
                let (base, pitch, _) =
                    crate::runtime::mapping_write::mapper_ref_texture_sample_window(
                        mapping, 4, 3, format,
                    )
                    .unwrap();
                let row = pixel_format::tight_row_bytes(4, format).unwrap() as usize;
                let mut expected = vec![0u8; row];
                assert!(pixel_format::Rgba8ToRow::for_format(format)
                    .unwrap()
                    .convert(&rgba[..16], 4, &mut expected));
                for y in 0..3 {
                    let mut actual = vec![0; row];
                    fixture
                        .host
                        .read_gpa(fixture.page + base + u64::from(pitch) * y, &mut actual)
                        .unwrap();
                    assert_eq!(actual, expected);
                }
                let generation = crate::runtime::surface_cache::frame_generation(
                    &fixture.state,
                    fixture.mid,
                    4,
                    3,
                )
                .unwrap();
                assert!(
                    crate::runtime::draw::published_mapping_frame(
                        &fixture.state,
                        &fixture.host,
                        fixture.mid,
                        4,
                        3,
                    )
                    .is_err(),
                    "a historical GPU Store does not imply current guest pixels"
                );
                assert!(
                    crate::runtime::surface_cache::get(&fixture.state, fixture.mid, 4, 3).is_none()
                );
                assert_eq!(
                    resident::read_published_rgba8(
                        &resident::ResidentColorKey::for_surface(fixture.mid, 4, 3),
                        generation,
                    ),
                    Some(rgba)
                );
            });
        }
    }
}

#[test]
fn mapped_gpu_store_cpu_fallback_keeps_pixels_and_requested_readback() {
    let formats = if guest_writeback::enabled() {
        vec![pixel_format::MTL_FORMAT_RGBA8_UNORM]
    } else {
        vec![
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
            pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        ]
    };
    for format in formats {
        objc::rc::autoreleasepool(|| {
            let mut fixture = Fixture::new(format);
            let mut pass = MetalRenderPass::default();
            let mut req = fixture.request(Color0Readback::Optional);
            let result =
                pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, false);
            assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
            assert_eq!(result.1, Some([255, 0, 0, 255].repeat(12)));
            assert_eq!(pass.batch.borrow().readbacks, 1);
        });
    }
}

#[test]
fn partial_scissor_stays_on_the_cpu_store_path() {
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
        let mut pass = MetalRenderPass::default();
        let mut req = fixture.request(Color0Readback::Optional);
        req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
        req.colors[0].target_seed_rgba = Some(vec![0x11; 4 * 3 * 4]);
        req.scissors = vec![crate::runtime::render_pass::ScissorRect {
            x: 0,
            y: 0,
            width: 2,
            height: 3,
        }];
        let result = pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, false);
        assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
        assert_eq!(pass.batch.borrow().readbacks, 1);
        let row = [
            255, 0, 0, 255, 255, 0, 0, 255, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        ];
        assert_eq!(result.1, Some(row.repeat(3)));
        let (_, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
            &fixture.state.mappings[&fixture.mid],
            4,
            3,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .unwrap();
        for y in 0..3 {
            let mut untouched = [0; 8];
            fixture
                .host
                .read_gpa(fixture.page + u64::from(pitch) * y + 8, &mut untouched)
                .unwrap();
            assert_eq!(
                untouched, [0xca; 8],
                "partial Store must leave other guest pixels alone"
            );
        }
    });
}

#[test]
fn mapped_gpu_store_with_a_memoryless_peer_preserves_both_ownership_domains() {
    if !runtime::system_device().unwrap().has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
        let mut pass = MetalRenderPass::default();
        let mut req = fixture.request(Color0Readback::Optional);
        req.colors.push(ColorRtRequest {
            slot: 1,
            texture_ref: 37,
            storage: ColorStorage::Memoryless,
            width: 4,
            height: 3,
            format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            sample_count: 1,
            load_action: MTL_LOAD_ACTION_CLEAR,
            store_action: MTL_STORE_ACTION_DONT_CARE,
            ..Default::default()
        });
        req.render_pass_continues = true;
        let first = pass.encode_draw(
            &mut fixture.state,
            &mut fixture.host,
            &mut req,
            false,
            false,
        );
        assert!(matches!(first.0, EncodeStatus::Ok), "{:?}", first.0);
        assert!(pass.target(&req.colors[0]).unwrap().gpu_store.is_some());
        assert!(pass.target(&req.colors[1]).unwrap().gpu_store.is_none());
        assert_eq!(
            pass.target(&req.colors[1])
                .unwrap()
                .texture()
                .storage_mode(),
            MTLStorageMode::Private
        );
        req.continues_render_pass = true;
        req.render_pass_continues = false;
        for color in &mut req.colors {
            color.load_action = MTL_LOAD_ACTION_LOAD;
        }
        let result = pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, true);
        assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
        assert!(result.1.is_none());
        assert_eq!(pass.batch.borrow().readbacks, 0);
        let (_, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
            &fixture.state.mappings[&fixture.mid],
            4,
            3,
            pixel_format::MTL_FORMAT_BGRA8_UNORM,
        )
        .unwrap();
        for y in 0..3 {
            let mut row = [0; 16];
            fixture
                .host
                .read_gpa(fixture.page + u64::from(pitch) * y, &mut row)
                .unwrap();
            assert_eq!(row.as_slice(), [0, 0, 255, 255].repeat(4));
        }
    });
}

#[test]
fn clipped_draws_with_whole_pass_stores_use_gpu_writeback() {
    if !runtime::system_device().unwrap().has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    for multi_draw in [false, true] {
        objc::rc::autoreleasepool(|| {
            let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
            let mut pass = MetalRenderPass::default();
            let mut req = fixture.request(Color0Readback::Optional);
            req.scissors = vec![crate::runtime::render_pass::ScissorRect {
                x: 0,
                y: 0,
                width: 2,
                height: 3,
            }];
            req.render_pass_continues = multi_draw;
            let first = pass.encode_draw(
                &mut fixture.state,
                &mut fixture.host,
                &mut req,
                !multi_draw,
                false,
            );
            assert!(matches!(first.0, EncodeStatus::Ok), "{:?}", first.0);
            assert!(first.1.is_none());
            if multi_draw {
                req.continues_render_pass = true;
                req.render_pass_continues = false;
                req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
                req.scissors[0].x = 2;
                let last =
                    pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, true);
                assert!(matches!(last.0, EncodeStatus::Ok), "{:?}", last.0);
                assert!(last.1.is_none());
            }
            assert_eq!(pass.batch.borrow().readbacks, 0);
            let (_, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
                &fixture.state.mappings[&fixture.mid],
                4,
                3,
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .unwrap();
            let mut expected = [0, 0, 255, 255].repeat(2);
            expected.extend(if multi_draw {
                [0, 0, 255, 255].repeat(2)
            } else {
                vec![0; 8]
            });
            for y in 0..3 {
                let mut actual = [0; 16];
                fixture
                    .host
                    .read_gpa(fixture.page + u64::from(pitch) * y, &mut actual)
                    .unwrap();
                assert_eq!(actual.as_slice(), expected);
            }
        });
    }
}

#[test]
fn mapped_gpu_store_rechecks_backing_pt_without_a_mapping_generation_change() {
    let device = runtime::system_device().unwrap();
    if !device.has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
        let page_size = fixture.state.page_size();
        let pte = 3 * page_size + 6 * 4;
        fixture
            .host
            .write_gpa(
                pte,
                &((fixture.page >> fixture.state.page_shift) as u32).to_le_bytes(),
            )
            .unwrap();
        let mapping = fixture.state.mappings.get_mut(&fixture.mid).unwrap();
        mapping.mapping_internal = 0;
        mapping.backing_walk = Some(crate::model::BackingWalk {
            task_id: 1,
            backing_pfn: 6,
            map_generation: mapping.map_generation,
        });
        let generation = mapping.map_generation;
        let mut pass = MetalRenderPass::default();
        let mut req = fixture.request(Color0Readback::Optional);
        req.render_pass_continues = true;
        assert!(matches!(
            pass.encode_draw(
                &mut fixture.state,
                &mut fixture.host,
                &mut req,
                false,
                false
            )
            .0,
            EncodeStatus::Ok
        ));
        assert!(pass.target(&req.colors[0]).unwrap().gpu_store.is_some());
        let replacement = 0x200000;
        fixture
            .host
            .map_range(replacement, page_size as usize, 0xdd);
        fixture
            .host
            .write_gpa(
                pte,
                &((replacement >> fixture.state.page_shift) as u32).to_le_bytes(),
            )
            .unwrap();
        assert_eq!(
            fixture.state.mappings[&fixture.mid].map_generation,
            generation
        );
        req.render_pass_continues = false;
        req.continues_render_pass = true;
        req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
        let result = pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, true);
        assert!(!matches!(result.0, EncodeStatus::Ok));
        for (page, expected) in [(fixture.page, 0xca), (replacement, 0xdd)] {
            let mut bytes = [0u8; 16];
            fixture.host.read_gpa(page, &mut bytes).unwrap();
            assert_eq!(
                bytes, [expected; 16],
                "a stale Store must write neither allocation"
            );
        }
    });
}

#[test]
fn pending_gpu_load_is_cancelled_when_its_mapping_is_revoked() {
    let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
    let mut pass = MetalRenderPass::default();
    let mut request = fixture.request(Color0Readback::Optional);
    request.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
    request.render_pass_continues = true;
    assert!(matches!(
        pass.encode_draw(
            &mut fixture.state,
            &mut fixture.host,
            &mut request,
            false,
            false
        )
        .0,
        EncodeStatus::Ok,
    ));
    assert!(pass.target(&request.colors[0]).unwrap().gpu_load.is_some());
    assert_eq!(pass.batch.borrow().submissions, 0);
    fixture.state.invalidate_mapping_pages(fixture.mid);
    let replacement = 0x200000;
    fixture
        .host
        .map_range(replacement, fixture.state.page_size() as usize, 0xdd);
    fixture
        .state
        .mappings
        .get_mut(&fixture.mid)
        .unwrap()
        .page_entries = vec![
        (((replacement >> fixture.state.page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT)
            | PAGE_ENTRY_VALID,
    ];
    request.render_pass_continues = false;
    request.continues_render_pass = true;
    assert!(!matches!(
        pass.encode_draw(
            &mut fixture.state,
            &mut fixture.host,
            &mut request,
            true,
            true
        )
        .0,
        EncodeStatus::Ok,
    ));
    assert_eq!(pass.batch.borrow().submissions, 0);
    for (page, expected) in [(fixture.page, 0xca), (replacement, 0xdd)] {
        let mut bytes = [0u8; 16];
        fixture.host.read_gpa(page, &mut bytes).unwrap();
        assert_eq!(bytes, [expected; 16]);
    }
}

#[test]
fn mapped_gpu_store_revocation_cannot_fall_back_or_materialize_into_replacement_pages() {
    let device = runtime::system_device().unwrap();
    if !device.has_unified_memory() || !guest_writeback::enabled() {
        return;
    }
    objc::rc::autoreleasepool(|| {
        let mut fixture = Fixture::new(pixel_format::MTL_FORMAT_BGRA8_UNORM);
        let mut pass = MetalRenderPass::default();
        let mut req = fixture.request(Color0Readback::Optional);
        req.render_pass_continues = true;
        assert!(matches!(
            pass.encode_draw(
                &mut fixture.state,
                &mut fixture.host,
                &mut req,
                false,
                false
            )
            .0,
            EncodeStatus::Ok
        ));
        assert!(pass.target(&req.colors[0]).unwrap().gpu_store.is_some());
        fixture.state.invalidate_mapping_pages(fixture.mid);
        let replacement = 0x200000;
        fixture
            .host
            .map_range(replacement, fixture.state.page_size() as usize, 0xdd);
        fixture
            .state
            .mappings
            .get_mut(&fixture.mid)
            .unwrap()
            .page_entries = vec![
            (((replacement >> fixture.state.page_shift) as u32) << PAGE_ENTRY_PFN_SHIFT)
                | PAGE_ENTRY_VALID,
        ];
        req.render_pass_continues = false;
        req.continues_render_pass = true;
        req.colors[0].load_action = MTL_LOAD_ACTION_LOAD;
        let result = pass.encode_draw(&mut fixture.state, &mut fixture.host, &mut req, true, true);
        assert!(!matches!(result.0, EncodeStatus::Ok));
        for (page, expected) in [(fixture.page, 0xca), (replacement, 0xdd)] {
            let mut bytes = [0u8; 16];
            fixture.host.read_gpa(page, &mut bytes).unwrap();
            assert_eq!(bytes, [expected; 16]);
        }
    });
}
