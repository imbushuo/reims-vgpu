use super::*;
use crate::backend::{vulkan::VulkanBackend, Backend};
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::endian::{st16, st32, st64};
use crate::runtime::decode::resource::*;
use crate::runtime::gva_mem::{define_task_pages_arm64e, read_task_gva, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;

fn fixture() -> (DeviceState, FakeHost, GvaSpan, Vec<u8>, Vec<u8>) {
    let mut host = FakeHost::new();
    let mut state = DeviceState::new(DeviceId(0xad73), PAGE_SHIFT_ARM64E);
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    assert!(state.set_object_list(1, 0, 32));
    let pixels: Vec<u8> = (0..8)
        .flat_map(|i| {
            [0x1001u16 + i, 0x4000, 0xb800, 0x3555]
                .into_iter()
                .flat_map(u16::to_le_bytes)
        })
        .collect();
    let mut padded = vec![0xee; 80];
    padded[..32].copy_from_slice(&pixels[..32]);
    padded[48..80].copy_from_slice(&pixels[32..]);
    let gva = 5 << PAGE_SHIFT_ARM64E;
    write_task_gva_arm64e(&mut host, &state.tasks[1], gva, &padded);
    let mut desc = vec![0; TEXTURE_DESC_BASE_LEN];
    st64(&mut desc[LINEAR_DESC_SIZE..], padded.len() as u64);
    st32(&mut desc[LINEAR_DESC_HANDLE..], 5);
    st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
    st32(&mut desc[TEXTURE_DESC_USED_SIZE..], padded.len() as u32);
    st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 48);
    st32(&mut desc[TEXTURE_DESC_WIDTH..], 4);
    st32(&mut desc[TEXTURE_DESC_HEIGHT..], 2);
    st32(&mut desc[TEXTURE_DESC_HEIGHT + 4..], 1);
    st16(
        &mut desc[TEXTURE_DESC_PIXEL_FORMAT..],
        pixel_format::MTL_FORMAT_RGBA16_FLOAT,
    );
    st32(&mut desc[TEXTURE_DESC_TRAILER_WIDTH..], 4);
    st32(&mut desc[TEXTURE_DESC_TRAILER_HEIGHT..], 2);
    st16(&mut desc[TEXTURE_DESC_SAMPLE_COUNT..], 1);
    write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &desc);
    let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_TEXTURE) | ((desc.len() as u32) << 8),
    );
    st64(&mut entry[4..], 0x200);
    write_task_gva_arm64e(
        &mut host,
        &state.tasks[1],
        list_object_entry_offset(7, 32).unwrap(),
        &entry,
    );
    (
        state,
        host,
        GvaSpan {
            texture_ref: 7,
            gva,
            row_stride: 48,
            width: 4,
            height: 2,
            format: pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        },
        pixels,
        padded,
    )
}

fn capture(state: &mut DeviceState, host: &mut FakeHost, span: GvaSpan) -> NativeColorSeed {
    match VulkanBackend
        .gva_color_load_seed(state, host, 1, span, 0)
        .unwrap()
    {
        ColorLoadSeed::Native(seed) => seed,
        ColorLoadSeed::Rgba8(_) => panic!("native LOAD may not pass through RGBA8"),
    }
}

#[test]
fn native_color_load_seed_preserves_tiny_hdr_negative_padding_and_fresh_mutation() {
    let (mut state, mut host, span, mut expected, mut padded) = fixture();
    let legacy = crate::runtime::draw::seed_color_load(
        &mut state,
        &mut host,
        1,
        span.texture_ref,
        span.gva,
        span.width,
        span.height,
    )
    .unwrap();
    assert_eq!(legacy[0], 0, "this is the old lossy LOAD boundary");
    let first = capture(&mut state, &mut host, span);
    assert_eq!(first.layout, TexelLayout::Rgba16Float);
    assert_eq!(*first.bytes, expected);
    expected[..2].copy_from_slice(&0x1002u16.to_le_bytes());
    padded[..2].copy_from_slice(&0x1002u16.to_le_bytes());
    write_task_gva_arm64e(&mut host, &state.tasks[1], span.gva, &padded);
    let second = capture(&mut state, &mut host, span);
    assert_eq!(*second.bytes, expected);
    assert_eq!(&first.bytes[..2], &0x1001u16.to_le_bytes());
    let mut actual = vec![0; padded.len()];
    read_task_gva(
        &host,
        &state.tasks[1],
        span.gva,
        &mut actual,
        PAGE_SHIFT_ARM64E,
    )
    .unwrap();
    assert_eq!(
        actual, padded,
        "native capture never changes guest bytes or padding"
    );
}

#[test]
fn native_color_load_seed_refuses_wrong_extent_layout_address_or_pitch_without_narrowing() {
    for changed in 0..4 {
        let (mut state, mut host, mut span, _, _) = fixture();
        match changed {
            0 => span.width = 2,
            1 => span.format = pixel_format::MTL_FORMAT_RGBA32_FLOAT,
            2 => span.gva += 8,
            _ => span.row_stride = 40,
        }
        assert!(
            VulkanBackend
                .gva_color_load_seed(&mut state, &mut host, 1, span, 0)
                .is_none(),
            "invalid native capture case {changed}"
        );
    }
}

#[test]
fn native_color_load_seed_carries_resolved_attachment_and_view_relative_mips() {
    for view in [false, true] {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(0xad74), PAGE_SHIFT_ARM64E);
        define_task_pages_arm64e(&mut host, &mut state, 4, 16);
        assert!(state.set_object_list(1, 0, 32));
        let extra = 2 * TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
        let mut desc =
            vec![0; (TEXTURE_DESC_BASE_LEN + extra).max(TEXTURE_DESC_PIXEL_FORMAT + extra + 2)];
        st64(&mut desc[LINEAR_DESC_SIZE..], 8192);
        st32(&mut desc[LINEAR_DESC_HANDLE..], 5);
        st32(&mut desc[TEXTURE_DESC_ROW_STRIDE..], 144);
        st32(&mut desc[TEXTURE_DESC_WIDTH..], 16);
        st32(&mut desc[TEXTURE_DESC_HEIGHT..], 8);
        st32(&mut desc[TEXTURE_DESC_HEIGHT + 4..], 1);
        st16(&mut desc[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 3);
        for (index, (offset, width, height, pitch)) in
            [(2048u64, 8u32, 4u32, 80u64), (4096, 4, 2, 48)]
                .into_iter()
                .enumerate()
        {
            let at = TEXTURE_DESC_LEVEL_RECORDS + index * TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
            st64(&mut desc[at + TEXTURE_LEVEL_OFFSET..], offset);
            st64(
                &mut desc[at + TEXTURE_LEVEL_SIZE..],
                pitch * u64::from(height),
            );
            st64(&mut desc[at + TEXTURE_LEVEL_ROW_STRIDE..], pitch);
            st32(&mut desc[at + TEXTURE_LEVEL_WIDTH..], width);
            st32(&mut desc[at + TEXTURE_LEVEL_HEIGHT..], height);
        }
        st16(
            &mut desc[TEXTURE_DESC_PIXEL_FORMAT + extra..],
            pixel_format::MTL_FORMAT_RGBA16_FLOAT,
        );
        let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            u32::from(OBJECT_TYPE_TEXTURE) | ((desc.len() as u32) << 8),
        );
        st64(&mut entry[4..], 0x1000);
        write_task_gva_arm64e(&mut host, &state.tasks[1], 0x1000, &desc);
        write_task_gva_arm64e(
            &mut host,
            &state.tasks[1],
            list_object_entry_offset(7, 32).unwrap(),
            &entry,
        );
        let reference = if view {
            let mut bytes = vec![0; TEXTURE_VIEW_MIN_RANGED];
            st32(
                &mut bytes[TEXTURE_VIEW_DESC_OPCODE..],
                TEXTURE_VIEW_OPCODE_RANGED,
            );
            st32(
                &mut bytes[TEXTURE_VIEW_DESC_LEN..],
                TEXTURE_VIEW_MIN_RANGED as u32,
            );
            st32(&mut bytes[TEXTURE_VIEW_DESC_TEXTURE_REF..], 8);
            st32(&mut bytes[TEXTURE_VIEW_DESC_BASE_REF..], 7);
            st16(
                &mut bytes[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
                pixel_format::MTL_FORMAT_RGBA16_FLOAT,
            );
            st16(
                &mut bytes[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
                TEXTURE_VIEW_MTL_TYPE_2D,
            );
            st64(&mut bytes[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1);
            st64(&mut bytes[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 2);
            st64(&mut bytes[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
            st32(
                &mut entry,
                u32::from(OBJECT_TYPE_TEXTURE_VIEW) | ((bytes.len() as u32) << 8),
            );
            st64(&mut entry[4..], 0x2000);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x2000, &bytes);
            write_task_gva_arm64e(
                &mut host,
                &state.tasks[1],
                list_object_entry_offset(8, 32).unwrap(),
                &entry,
            );
            8
        } else {
            7
        };
        let (level, offset, width, height, pitch) = if view {
            (2, 4096, 4, 2, 48)
        } else {
            (1, 2048, 8, 4, 80)
        };
        let base = 5 << PAGE_SHIFT_ARM64E;
        let mut allocation = vec![0xcc; 8192];
        let texel: Vec<u8> = [0x1001u16, 0x4000, 0xb800, 0x3555]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        let expected = texel.repeat(width * height);
        for row in 0..height {
            allocation[offset + row * pitch..offset + row * pitch + width * 8]
                .copy_from_slice(&expected[row * width * 8..(row + 1) * width * 8]);
        }
        write_task_gva_arm64e(&mut host, &state.tasks[1], base, &allocation);
        let attachment = crate::runtime::render_pass::ColorAttachment {
            texture_ref: reference,
            level: 1,
            load_action: MTL_LOAD_ACTION_LOAD,
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            ..Default::default()
        };
        let mut req = crate::runtime::draw::mrt_draw_request(
            &mut state,
            &mut host,
            1,
            0,
            &[(0, attachment)],
            &[],
            Default::default(),
        )
        .unwrap();
        let color = &req.colors[0];
        assert_eq!(color.guest_mip_level, level);
        assert_eq!(
            (
                color.target_gva,
                color.width,
                color.height,
                color.row_stride
            ),
            (
                base + offset as u64,
                width as u32,
                height as u32,
                pitch as u32
            )
        );
        let span = GvaSpan {
            texture_ref: reference,
            gva: color.target_gva,
            row_stride: color.row_stride,
            width: color.width,
            height: color.height,
            format: color.format,
        };
        let seed = VulkanBackend.gva_color_load_seed(
            &mut state,
            &mut host,
            1,
            span,
            color.guest_mip_level,
        );
        let Some(ColorLoadSeed::Native(seed)) = seed else {
            panic!("native mip capture failed");
        };
        assert_eq!(*seed.bytes, expected);
        req.colors[0].set_load_seed(None);
        req.gva_load_from_resident = true;
        let mut chained = false;
        super::super::honour_gva_load_elision(&mut state, &mut host, &mut req, &mut chained);
        assert!(!chained);
        assert_eq!(
            *req.colors[0].target_seed_native.as_ref().unwrap().bytes,
            expected
        );
        let mut unchanged = vec![0; allocation.len()];
        read_task_gva(
            &host,
            &state.tasks[1],
            base,
            &mut unchanged,
            PAGE_SHIFT_ARM64E,
        )
        .unwrap();
        assert_eq!(
            unchanged, allocation,
            "other mips and all padding remain untouched"
        );
        let color = &req.colors[0];
        let generation = crate::runtime::writeback_debt::gva_resource_generation(
            &mut state,
            &host,
            crate::runtime::writeback_debt::GvaResourceKey {
                task_id: 1,
                texture_ref: color.texture_ref,
            },
            color.target_gva,
            u64::from(color.row_stride) * u64::from(color.height),
        );
        let authority = crate::runtime::render_writeback::vulkan::NativeEagerStore::capture(
            &state, 1, color, generation,
        )
        .expect("nonzero plain/view mip must retain its real allocation owner");
        authority.validate_live(&state).unwrap();
        crate::runtime::objects::replace_physical(&mut state, &mut host, 1, 7);
        assert!(
            authority.validate_live(&state).is_err(),
            "a view's resolved allocation incarnation must be checked, not its mip GVA"
        );
    }
}

#[test]
fn native_color_load_seed_residency_refusal_recaptures_native_not_rgba8() {
    let (mut state, mut host, span, expected, _) = fixture();
    let mut req = DrawEncodeRequest {
        task_id: 1,
        gva_load_from_resident: true,
        colors: vec![ColorRtRequest {
            texture_ref: span.texture_ref,
            target_gva: span.gva,
            row_stride: span.row_stride,
            width: span.width,
            height: span.height,
            format: span.format,
            load_action: MTL_LOAD_ACTION_LOAD,
            ..Default::default()
        }],
        ..Default::default()
    };
    let mut chained = false;
    assert!(
        super::super::honour_gva_load_elision(&mut state, &mut host, &mut req, &mut chained)
            .is_none()
    );
    assert!(!chained);
    assert!(req.colors[0].target_seed_rgba.is_none());
    assert_eq!(
        *req.colors[0].target_seed_native.as_ref().unwrap().bytes,
        expected
    );
}

#[test]
#[ignore = "requires exclusive Vulkan GPU and synchronization validation"]
fn native_color_load_gpu_roundtrip_preserves_untouched_half_and_guest_store_bytes() {
    use crate::backend::vulkan::engine::{self, *};
    use crate::backend::vulkan::sampled_shader::graphics_tests::shader;
    use std::sync::Arc;
    let (mut state, mut host, span, mut expected, mut padded) = fixture();
    let seed = capture(&mut state, &mut host, span);
    let target =
        engine::pass_local::PassLocalTarget::new(4, 2, ash::vk::Format::R16G16B16A16_SFLOAT)
            .unwrap();
    let mut req = DrawRequest {
        width: 4,
        height: 2,
        vertex_count: 3,
        target_identity: Some(target.identity().clone()),
        color_attachment: Some(ColorAttachmentState::new(
            ash::vk::Format::R16G16B16A16_SFLOAT,
            ColorClearValue::Float([0.0; 4]),
        )),
        color_write_mask: ColorWriteMask::NONE,
        target_native_seed: Some(seed),
        vert_spirv: Arc::new(shader(true, false, 0, 0, 1.0)),
        frag_spirv: Arc::new(shader(false, false, 0, 0, 1.0)),
        skip_readback: true,
        ..Default::default()
    };
    engine::execute_draw_request(&state, &req).unwrap();
    assert_eq!(
        engine::read_target_native(target.identity())
            .unwrap()
            .pixels,
        expected
    );
    req.target_native_seed = None;
    req.load_from_target = true;
    req.color_write_mask = ColorWriteMask::ALL;
    req.scissors = vec![ScissorResource {
        x: 1,
        y: 0,
        width: 2,
        height: 2,
    }];
    engine::execute_draw_request(&state, &req).unwrap();
    for y in 0..2 {
        for x in 1..3 {
            for component in 0..4 {
                let offset = (y * 4 + x) * 8 + component * 2;
                expected[offset..offset + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
    }
    assert_eq!(
        engine::read_target_native(target.identity())
            .unwrap()
            .pixels,
        expected
    );
    padded[..32].fill(0);
    padded[48..].fill(0);
    write_task_gva_arm64e(&mut host, &state.tasks[1], span.gva, &padded);
    let color = ColorRtRequest {
        texture_ref: span.texture_ref,
        target_gva: span.gva,
        row_stride: span.row_stride,
        width: span.width,
        height: span.height,
        format: span.format,
        store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
        ..Default::default()
    };
    let pages = crate::runtime::draw::StoreTargetPages::capture(
        &state,
        &host,
        1,
        span.gva,
        u64::from(span.row_stride) * u64::from(span.height),
    );
    crate::runtime::render_writeback::vulkan::store_gva_frame(
        &mut state,
        &mut host,
        1,
        target.identity(),
        &color,
        span.texture_ref,
        Some(&pages),
        &[],
    )
    .unwrap();
    engine::test_quiesce_ring();
    let mut actual = vec![0; padded.len()];
    read_task_gva(
        &host,
        &state.tasks[1],
        span.gva,
        &mut actual,
        PAGE_SHIFT_ARM64E,
    )
    .unwrap();
    padded[..32].copy_from_slice(&expected[..32]);
    padded[48..].copy_from_slice(&expected[32..]);
    assert_eq!(
        actual, padded,
        "Store/getBytes preserves texels and row padding"
    );
    assert_eq!(&actual[..2], &0x1001u16.to_le_bytes());
    eprintln!("native_color_load_gpu capture=PASS native_LOAD=PASS partial_untouched=PASS Store_getBytes=PASS half0=1001");
}

#[test]
#[ignore = "requires exclusive Vulkan GPU and synchronization validation"]
fn native_primary_store_gpu_unwitnessed_fallback_preserves_half_and_guest_bytes() {
    native_primary_store_roundtrip(true, None);
}

#[test]
#[ignore = "requires exclusive Vulkan GPU and synchronization validation"]
fn native_primary_store_gpu_nondeferred_preserves_half_and_guest_bytes() {
    native_primary_store_roundtrip(false, None);
}

#[derive(Clone, Copy, Debug)]
enum DestinationChange {
    Pages,
    OwnerRetired,
    Retired,
    Recreated,
    SamePagesRepoint,
}

#[test]
#[ignore = "requires exclusive Vulkan GPU and synchronization validation"]
fn native_primary_store_gpu_rejects_changed_destination_authority_with_imports() {
    for allow_deferred in [false, true] {
        for changed in [
            DestinationChange::Pages,
            DestinationChange::OwnerRetired,
            DestinationChange::Retired,
            DestinationChange::Recreated,
            DestinationChange::SamePagesRepoint,
        ] {
            native_primary_store_roundtrip(allow_deferred, Some(changed));
        }
    }
}

fn native_primary_store_roundtrip(allow_deferred: bool, changed: Option<DestinationChange>) {
    use super::super::{output, primary_store};
    use crate::backend::vulkan::engine::{self, *};
    use crate::backend::vulkan::sampled_shader::graphics_tests::shader;
    use crate::runtime::writeback_debt::{gva_resource_generation, GvaResourceKey};
    use std::sync::Arc;

    let (mut state, mut host, span, mut expected, mut padded) = fixture();
    crate::runtime::guest_ram_map::reset();
    host.stable_map_pages = true;
    host.guest_write_startup_window = true;
    let seed = capture(&mut state, &mut host, span);
    let key = GvaResourceKey {
        task_id: 1,
        texture_ref: span.texture_ref,
    };
    let declared_span = u64::from(span.row_stride) * u64::from(span.height);
    let generation = gva_resource_generation(&mut state, &host, key, span.gva, declared_span);
    let color = ColorRtRequest {
        texture_ref: span.texture_ref,
        target_gva: span.gva,
        row_stride: span.row_stride,
        width: span.width,
        height: span.height,
        format: span.format,
        store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
        ..Default::default()
    };
    let mut runtime = crate::runtime::draw::DrawEncodeRequest {
        task_id: 1,
        colors: vec![color.clone()],
        gva_alloc_gen: generation,
        chain_from_resident: true,
        ..Default::default()
    };
    let pages =
        super::super::sync_store_allowed_pages(&state, &host, 1, Some(&color), true).unwrap();
    let identity = TargetIdentity::Gva {
        gva: span.gva,
        width: span.width,
        height: span.height,
        generation,
        format: ash::vk::Format::R16G16B16A16_SFLOAT,
    };
    let mut draw = DrawRequest {
        width: span.width,
        height: span.height,
        vertex_count: 3,
        target_identity: Some(identity.clone()),
        color_attachment: Some(ColorAttachmentState::new(
            identity.resident_format(),
            ColorClearValue::Float([0.0; 4]),
        )),
        color_write_mask: ColorWriteMask::NONE,
        target_native_seed: Some(seed),
        vert_spirv: Arc::new(shader(true, false, 0, 0, 1.0)),
        frag_spirv: Arc::new(shader(false, false, 0, 0, 1.0)),
        skip_readback: true,
        ..Default::default()
    };
    engine::execute_draw_request(&state, &draw).unwrap();
    let store = primary_store::GvaStore::prepare(&state, &runtime, &mut draw, allow_deferred, true)
        .expect("a final native primary Store must not choose the RGBA8 readback");
    let unlicensed =
        primary_store::GvaStore::prepare(&state, &runtime, &mut draw, allow_deferred, true)
            .unwrap();
    assert_eq!(draw.target_identity.as_ref(), Some(&identity));
    assert!(draw.skip_readback);
    draw.target_native_seed = None;
    draw.load_from_target = true;
    draw.color_write_mask = ColorWriteMask::ALL;
    draw.scissors = vec![ScissorResource {
        x: 1,
        y: 0,
        width: 2,
        height: 2,
    }];
    engine::execute_draw_request(&state, &draw).unwrap();
    for y in 0..2 {
        for x in 1..3 {
            for component in 0..4 {
                let at = (y * 4 + x) * 8 + component * 2;
                expected[at..at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
        }
    }
    assert_eq!(
        engine::read_target_native(&identity).unwrap().pixels,
        expected
    );
    if let Some(changed) = changed {
        let direct = crate::runtime::drain::store_route_count("gva_flush_gpu_direct");
        crate::runtime::render_writeback::vulkan::store_gva_frame(
            &mut state,
            &mut host,
            1,
            &identity,
            &color,
            span.texture_ref,
            Some(&pages),
            &[],
        )
        .unwrap();
        engine::test_quiesce_ring();
        assert_eq!(
            crate::runtime::drain::store_route_count("gva_flush_gpu_direct"),
            direct + 1,
            "the fixture must demonstrate actual GPU-direct imports before revocation",
        );
        let page = 1u64 << PAGE_SHIFT_ARM64E;
        let old_gpa = 9 * page;
        let new_gpa = 10 * page;
        host.write_gpa(old_gpa, &vec![0xa6; padded.len()]).unwrap();
        host.write_gpa(new_gpa, &vec![0x7d; padded.len()]).unwrap();
        match changed {
            DestinationChange::Pages => {
                host.write_gpa(3 * page + 5 * 4, &10u32.to_le_bytes())
                    .unwrap();
            }
            DestinationChange::OwnerRetired => {
                let name = state.object_name(1, span.texture_ref).unwrap();
                assert!(state.retire_object_name(1, name).is_some());
            }
            DestinationChange::Retired => {
                assert!(crate::runtime::writeback_debt::retire_gva_resource(
                    &mut state,
                    1,
                    span.texture_ref,
                ));
            }
            DestinationChange::Recreated => {
                assert!(crate::runtime::writeback_debt::retire_gva_resource(
                    &mut state,
                    1,
                    span.texture_ref,
                ));
                assert_ne!(
                    gva_resource_generation(&mut state, &host, key, span.gva, declared_span),
                    generation,
                );
            }
            DestinationChange::SamePagesRepoint => {
                let before =
                    crate::runtime::objects::backing_id(&state, &host, 1, span.texture_ref)
                        .unwrap();
                crate::runtime::objects::replace_physical(
                    &mut state,
                    &mut host,
                    1,
                    span.texture_ref,
                );
                assert_ne!(
                    crate::runtime::objects::backing_id(&state, &host, 1, span.texture_ref)
                        .unwrap(),
                    before,
                    "same physical bytes do not preserve the allocation's incarnation",
                );
            }
        }
        let result = store.finish(&mut state, &mut host, &runtime, Some(&pages));
        engine::test_quiesce_ring();
        let mut old = vec![0; padded.len()];
        let mut new = vec![0; padded.len()];
        host.read_gpa(old_gpa, &mut old).unwrap();
        host.read_gpa(new_gpa, &mut new).unwrap();
        assert_eq!(
            old,
            vec![0xa6; padded.len()],
            "{changed:?}: retired destination pages were written"
        );
        assert_eq!(
            new,
            vec![0x7d; padded.len()],
            "{changed:?}: replacement pages were written"
        );
        assert!(
            matches!(
                result,
                Err(crate::runtime::draw::EncodeStatus::WritebackFailed(_))
            ),
            "{changed:?}: changed destination must refuse"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("gva_flush_gpu_direct"),
            direct + 1,
            "revoked eager Store must not queue a guest-memory GPU copy"
        );
        engine::reset_guest_state();
        crate::runtime::guest_ram_map::reset();
        eprintln!("native_primary_store_authority deferred={allow_deferred} change={changed:?} imports=PROVEN refused=PASS old_pages=UNCHANGED new_pages=UNCHANGED");
        return;
    }
    // A later generation cannot redirect the Store away from the image rendered above.
    runtime.gva_alloc_gen += 1;
    assert!(matches!(
        unlicensed.finish(&mut state, &mut host, &runtime, None),
        Err(crate::runtime::draw::EncodeStatus::WritebackFailed(_)),
    ));
    let mut unchanged = vec![0; padded.len()];
    read_task_gva(
        &host,
        &state.tasks[1],
        span.gva,
        &mut unchanged,
        PAGE_SHIFT_ARM64E,
    )
    .unwrap();
    assert_eq!(
        unchanged, padded,
        "unlicensed primary Store must not touch guest bytes"
    );
    let sync_before = crate::runtime::drain::store_route_count("gva_store_sync");
    let native_before = crate::runtime::drain::store_route_count("gva_store_native");
    let copied_before = crate::runtime::drain::store_route_count("gva_eager_copied_native");
    let direct_before = crate::runtime::drain::store_route_count("gva_flush_gpu_direct");
    let unwitnessed_before = crate::runtime::drain::store_route_count("gvadebt_arm_unwitnessed");
    match store
        .finish(&mut state, &mut host, &runtime, Some(&pages))
        .unwrap()
    {
        primary_store::StoreOutput::Complete => {}
        primary_store::StoreOutput::Rgba8(mut bytes) => {
            assert!(output::publish_gva_draw_pixels(
                &mut state,
                &mut host,
                1,
                &color,
                &mut bytes,
                false,
                Some(pages.membership()),
            ));
        }
    }
    engine::test_quiesce_ring();
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_store_sync"),
        sync_before + 1
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_store_native"),
        native_before + 1
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_eager_copied_native"),
        copied_before + 1
    );
    assert_eq!(
        crate::runtime::drain::store_route_count("gva_flush_gpu_direct"),
        direct_before,
        "eager native publication must not submit a GPU write even with imports available",
    );
    assert!(!engine::guest_writes_outstanding());
    assert_eq!(
        crate::runtime::drain::store_route_count("gvadebt_arm_unwitnessed"),
        unwitnessed_before + u64::from(allow_deferred),
        "the deferred candidate must take the actual witness-startup refusal",
    );
    assert!(!crate::runtime::writeback_debt::gva_resident_authoritative(
        &state,
        crate::backend::vulkan::gva_window(&identity).unwrap(),
    ));
    let mut actual = vec![0; padded.len()];
    read_task_gva(
        &host,
        &state.tasks[1],
        span.gva,
        &mut actual,
        PAGE_SHIFT_ARM64E,
    )
    .unwrap();
    padded[..32].copy_from_slice(&expected[..32]);
    padded[48..].copy_from_slice(&expected[32..]);
    assert_eq!(
        actual, padded,
        "primary Store/getBytes must preserve native half bits and padding"
    );
    assert_eq!(&actual[..2], &0x1001u16.to_le_bytes());
    assert!(
        crate::runtime::surface_cache::get_gva(&state, span.gva, span.width, span.height,)
            .is_none(),
        "native publication must not leave a narrowed RGBA8 cache"
    );
    assert!(crate::runtime::surface_cache::get_texture(
        &state,
        1,
        span.texture_ref,
        span.width,
        span.height,
    )
    .is_none());
    engine::reset_guest_state();
    crate::runtime::guest_ram_map::reset();
    eprintln!("native_primary_store_gpu deferred_capability={allow_deferred} native_resident=PASS primary_Store_getBytes=PASS half0=1001");
}
