use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
use crate::runtime::gva_mem::{define_task_pages_arm64e, read_task_gva, write_task_gva_arm64e};
use crate::runtime::host::FakeHost;
use reims_vgpu_protocol::pass_action::{MTL_STORE_ACTION_DONT_CARE, MTL_STORE_ACTION_STORE};

const PAGE: u64 = 1 << PAGE_SHIFT_ARM64E;
const MAPPING: u32 = 42;
const MAPPING_GPA: u64 = 0x30 * PAGE;

fn color(slot: u32, gva: u64, mapping_id: u32) -> ColorRtRequest {
    ColorRtRequest {
        slot, texture_ref: slot + 10, target_gva: gva, mapping_id,
        width: 4, height: 2, row_stride: 24, sample_count: 1,
        format: pixel_format::MTL_FORMAT_BGRA8_UNORM,
        store_action: MTL_STORE_ACTION_STORE, ..Default::default()
    }
}

fn target(color: &ColorRtRequest) -> SecondaryColorTarget {
    let format = translate::pixel::color_attachment(color.format).unwrap().0;
    let identity = if color.mapping_id != 0 {
        TargetIdentity::Surface {
            id: color.mapping_id, width: color.width, height: color.height,
            generation: 37, format: format.vk,
        }
    } else {
        TargetIdentity::Gva {
            gva: color.target_gva, width: color.width, height: color.height,
            generation: 51, format: format.vk,
        }
    };
    SecondaryColorTarget {
        identity, width: color.width, height: color.height, attachment: format.with_clear(color.clear_color),
        load: false, blend: None, color_write_mask: Default::default(),
    }
}

fn fixture() -> (DeviceState, FakeHost, Vec<ColorRtRequest>, Vec<SecondaryColorTarget>) {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    define_task_pages_arm64e(&mut host, &mut state, 4, 8);
    host.map_range(MAPPING_GPA, PAGE as usize, 0xee);
    state.map_surface(MAPPING);
    state.attach_mapping_internal(MAPPING, 0);
    let mapping = state.mappings.get_mut(&MAPPING).unwrap();
    mapping.mapping_internal = 1;
    mapping.page_entries = vec![(0x30 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
    assert!(state.set_mapping_geom(MAPPING, 4, 2, pixel_format::MTL_FORMAT_BGRA8_UNORM));
    let colors = vec![color(0, PAGE * 2, 0), color(1, PAGE, 0), color(2, 0, MAPPING)];
    for color in &colors[..2] {
        write_task_gva_arm64e(&mut host, &state.tasks[1], color.target_gva, &[0xee; 48]);
    }
    let targets = colors.iter().skip(1).map(target).collect();
    (state, host, colors, targets)
}

#[test]
fn secondary_mapping_and_gva_stores_publish_their_own_pixels_not_the_primary() {
    let (mut state, mut host, colors, targets) = fixture();
    let stores = Stores::capture(&state, &host, 1, &colors, &targets, true).unwrap();
    let expected_ids: Vec<_> = targets.iter().map(|target| target.identity.clone()).collect();
    let mut seen = Vec::new();
    stores.publish_with(&mut state, &mut host, |state, host, task, store| {
        seen.push(store.identity.clone());
        let c = &store.color;
        let pixel = if c.slot == 1 { [255, 0, 0, 255] } else { [0, 255, 0, 255] };
        let bytes = pixel.repeat((c.width * c.height) as usize);
        match &store.destination {
            Destination::Mapping(_) => crate::runtime::mapping_write::write_native_image(
                state, host, c.mapping_id, &bytes, c.width * 4, c.width, c.height, c.format,
            ),
            Destination::Gva(pages) => crate::runtime::draw::write_gva_frame_within_skipping(
                state, host, task, c.target_gva, c.width, c.height, c.row_stride, c.format,
                crate::runtime::draw::FrameRows::Native(&bytes), Some(pages.membership()), &[],
            ).is_ok(),
        }
    }).unwrap();
    assert_eq!(seen, expected_ids);
    let mut primary = [0; 48];
    read_task_gva(&host, &state.tasks[1], PAGE * 2, &mut primary, PAGE_SHIFT_ARM64E).unwrap();
    assert_eq!(primary, [0xee; 48]);
    let mut secondary = [0; 48];
    read_task_gva(&host, &state.tasks[1], PAGE, &mut secondary, PAGE_SHIFT_ARM64E).unwrap();
    for y in 0..2 {
        assert_eq!(&secondary[y * 24..y * 24 + 16], &[255, 0, 0, 255].repeat(4));
        assert_eq!(&secondary[y * 24 + 16..y * 24 + 24], &[0xee; 8]);
    }
    let (base, pitch, _) = crate::runtime::mapping_write::mapper_ref_texture_sample_window(
        &state.mappings[&MAPPING], 4, 2, pixel_format::MTL_FORMAT_BGRA8_UNORM,
    ).unwrap();
    for y in 0..2u64 {
        let mut row = [0; 16];
        host.read_gpa(MAPPING_GPA + base + y * u64::from(pitch), &mut row).unwrap();
        assert_eq!(row.as_slice(), [0, 255, 0, 255].repeat(4));
    }
    assert!(state.pending_writebacks.is_empty(), "no lazy secondary Store");
}

#[test]
fn secondary_dontcare_memoryless_and_intermediate_draws_publish_nothing() {
    let (mut state, mut host, mut colors, targets) = fixture();
    let intermediate = Stores::capture(&state, &host, 1, &colors, &targets, false).unwrap();
    intermediate.publish_with(&mut state, &mut host, |_, _, _, _| panic!("intermediate Store")).unwrap();
    colors[1].store_action = MTL_STORE_ACTION_DONT_CARE;
    colors[2].store_action = MTL_STORE_ACTION_DONT_CARE;
    colors[2].storage = ColorStorage::Memoryless;
    let discarded = Stores::capture(&state, &host, 1, &colors, &targets, true).unwrap();
    discarded.publish_with(&mut state, &mut host, |_, _, _, _| panic!("DontCare Store")).unwrap();
}

#[test]
fn secondary_store_refuses_missing_pages_unknown_actions_and_memoryless_backing() {
    let (state, host, colors, _) = fixture();
    for (mutated, want) in [
        (ColorRtRequest { target_gva: PAGE * 100, ..colors[1].clone() }, "draw_vk_secondary_store_pages"),
        (ColorRtRequest { store_action: u16::MAX, ..colors[1].clone() }, "draw_vk_secondary_store_action"),
        (ColorRtRequest { storage: ColorStorage::Memoryless, ..colors[1].clone() }, "draw_vk_secondary_store_memoryless"),
    ] {
        let mut input = colors.clone();
        input[1] = mutated;
        let inputs_targets: Vec<_> = input.iter().skip(1).map(target).collect();
        let Err(refusal) = Stores::capture(&state, &host, 1, &input, &inputs_targets, true) else {
            panic!("unlicensed Store was accepted")
        };
        assert_eq!(refusal.slug(), want);
    }
}

#[test]
fn secondary_mapping_replacement_refuses_before_any_attachment_is_published() {
    let (mut state, mut host, colors, targets) = fixture();
    let stores = Stores::capture(&state, &host, 1, &colors, &targets, true).unwrap();
    DeviceState::bump_map_generation(state.mappings.get_mut(&MAPPING).unwrap());
    let refusal = stores.publish_with(
        &mut state, &mut host, |_, _, _, _| panic!("one destination moved"),
    ).unwrap_err();
    assert_eq!(refusal.slug(), "draw_vk_secondary_store_mapping_moved");
}

#[test]
fn failed_secondary_publication_is_not_success() {
    let (mut state, mut host, colors, targets) = fixture();
    let stores = Stores::capture(&state, &host, 1, &colors, &targets, true).unwrap();
    let refusal = stores.publish_with(&mut state, &mut host, |_, _, _, _| false).unwrap_err();
    assert_eq!(refusal.slot, 1);
    assert_eq!(refusal.slug(), "draw_vk_secondary_store_transfer");
}
