//! Exact native surface publication, independent of sampled channel mappings.

use crate::backend::vulkan::{engine::NativeTargetReadback, translate::pixel};
use crate::model::DeviceState;
use crate::runtime::{host::{HostMemory, HostOps}, mapping_write};

pub(super) fn required(format: u16) -> bool {
    pixel::verbatim_texel(format)
        .and_then(|(format, _)| pixel::texel_layout_of(format))
        .is_some_and(|layout| !layout.is_four_byte_color())
}

#[derive(Debug, PartialEq, Eq)]
enum NativeSurfaceDecline {
    Format { declared: u16, held: ash::vk::Format, wanted: Option<ash::vk::Format> },
    Extent { wanted: (u32, u32), held: (u32, u32) },
    WriteRefused,
}

impl crate::observe::Decline for NativeSurfaceDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Format { .. } => "surface_native_format_mismatch",
            Self::Extent { .. } => "surface_native_extent_mismatch",
            Self::WriteRefused => "surface_native_write_refused",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Format { declared, held, wanted } => vec![
                ("fmt", format!("{declared:#x}")), ("held", format!("{held:?}")),
                ("wanted", format!("{wanted:?}")),
            ],
            Self::Extent { wanted, held } => vec![
                ("wanted", format!("{}x{}", wanted.0, wanted.1)),
                ("held", format!("{}x{}", held.0, held.1)),
            ],
            Self::WriteRefused => Vec::new(),
        }
    }
}

fn plan(
    declared: u16,
    width: u32,
    height: u32,
    source: &NativeTargetReadback,
) -> Result<u32, NativeSurfaceDecline> {
    let wanted = pixel::verbatim_texel(declared);
    if wanted != Some((source.format, source.layout.bytes_per_texel())) {
        return Err(NativeSurfaceDecline::Format {
            declared, held: source.format, wanted: wanted.map(|(format, _)| format),
        });
    }
    if (width, height) != (source.width, source.height) {
        return Err(NativeSurfaceDecline::Extent {
            wanted: (width, height), held: (source.width, source.height),
        });
    }
    width.checked_mul(source.layout.bytes_per_texel()).ok_or(
        NativeSurfaceDecline::Extent {
            wanted: (width, height), held: (source.width, source.height),
        },
    )
}

pub(super) fn store<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    source: &NativeTargetReadback,
) -> bool {
    store_skipping(state, host, mapping_id, width, height, source, &[])
}

pub(super) fn store_skipping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    width: u32,
    height: u32,
    source: &NativeTargetReadback,
    skip: &[(u64, u64)],
) -> bool {
    let declared = state.mappings.get(&mapping_id)
        .map(mapping_write::mapping_store_format).unwrap_or(0);
    let result = plan(declared, width, height, source).and_then(|stride| {
        if mapping_write::write_native_image_skipping(
            state, host, mapping_id, &source.pixels, stride, width, height, declared, skip,
        ) {
            Ok(())
        } else {
            Err(NativeSurfaceDecline::WriteRefused)
        }
    });
    if let Err(refusal) = result {
        crate::observe::Emit::decline("render_store_lost", &refusal)
            .field("mapping", mapping_id).fail();
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::protocol::{iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID}, pixel_format as p};
    use crate::runtime::{host::FakeHost, surface_cache};

    const MID: u32 = 5;
    const GPA: u64 = 0x12 << PAGE_SHIFT_ARM64E;
    const WIDTH: u32 = 4;
    const HEIGHT: u32 = 3;

    fn fixture(format: u16) -> (DeviceState, FakeHost, NativeTargetReadback) {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        host.map_range(GPA, 1 << PAGE_SHIFT_ARM64E, 0xee);
        state.map_surface(MID);
        let mapping = state.mappings.get_mut(&MID).unwrap();
        mapping.mapped = true;
        mapping.mapping_internal = 1;
        mapping.page_entries = vec![(0x12 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        assert!(state.set_mapping_geom(MID, WIDTH, HEIGHT, format));
        surface_cache::store(&mut state, MID, WIDTH, HEIGHT, vec![0xcc; (WIDTH * HEIGHT * 4) as usize]);
        let source = NativeTargetReadback {
            pixels: vec![0, 17, 37, 255, 64, 99, 128, 191, 3, 78, 213, 231],
            layout: p::TexelLayout::R8,
            format: ash::vk::Format::R8_UNORM,
            width: WIDTH,
            height: HEIGHT,
        };
        (state, host, source)
    }

    #[test]
    fn surface_native_publication_preserves_a8_r8_bytes_padding_and_provenance() {
        for format in [p::MTL_FORMAT_A8_UNORM, p::MTL_FORMAT_R8_UNORM] {
            let (mut state, mut host, source) = fixture(format);
            let (offset, pitch, span) = mapping_write::mapper_ref_texture_sample_window(
                &state.mappings[&MID], WIDTH, HEIGHT, format,
            ).unwrap();
            let before = state.host_writes.epoch();
            assert!(required(format));
            assert!(store(&mut state, &mut host, MID, WIDTH, HEIGHT, &source));
            let mut landed = vec![0; span as usize];
            host.read_gpa(GPA, &mut landed).unwrap();
            let mut expected = vec![0xee; span as usize];
            for y in 0..HEIGHT as usize {
                let row = offset as usize + y * pitch as usize;
                expected[row..row + WIDTH as usize].copy_from_slice(
                    &source.pixels[y * WIDTH as usize..(y + 1) * WIDTH as usize],
                );
            }
            assert_eq!(landed, expected, "native bytes, including zeros and untouched row padding");
            assert!(state.host_writes.epoch() > before, "publication records the guest-page write");
            assert!(surface_cache::get(&state, MID, WIDTH, HEIGHT).is_none(),
                "a mapping-global converted cache cannot impersonate both R8 and A8");
        }
    }

    #[test]
    fn surface_native_publication_refuses_incompatible_format_or_extent_before_writing() {
        for wrong_extent in [false, true] {
            let (mut state, mut host, mut source) = fixture(p::MTL_FORMAT_A8_UNORM);
            if wrong_extent {
                source.width += 1;
            } else {
                source.format = ash::vk::Format::B8G8R8A8_UNORM;
                source.layout = p::TexelLayout::Bgra8;
            }
            let before = state.host_writes.epoch();
            assert!(!store(&mut state, &mut host, MID, WIDTH, HEIGHT, &source));
            let mut unchanged = [0; 1024];
            host.read_gpa(GPA, &mut unchanged).unwrap();
            assert_eq!(unchanged, [0xee; 1024]);
            assert_eq!(state.host_writes.epoch(), before);
            assert!(surface_cache::get(&state, MID, WIDTH, HEIGHT).is_some());
        }
    }

    #[test]
    fn surface_native_publication_merge_preserves_guest_owned_bytes_and_padding() {
        for format in [p::MTL_FORMAT_A8_UNORM, p::MTL_FORMAT_R8_UNORM] {
            let (mut state, mut host, source) = fixture(format);
            let (offset, pitch, span) = mapping_write::mapper_ref_texture_sample_window(
                &state.mappings[&MID], WIDTH, HEIGHT, format,
            ).unwrap();
            let skip = [
                (offset + 1, offset + 3),
                (offset + u64::from(pitch), offset + u64::from(pitch) + 3),
            ];
            assert!(store_skipping(&mut state, &mut host, MID, WIDTH, HEIGHT, &source, &skip));
            let mut landed = vec![0; span as usize];
            host.read_gpa(GPA, &mut landed).unwrap();
            let mut expected = vec![0xee; span as usize];
            for y in 0..HEIGHT as usize {
                for x in 0..WIDTH as usize {
                    let at = offset + (y as u64) * u64::from(pitch) + x as u64;
                    if !skip.iter().any(|&(lo, hi)| at >= lo && at < hi) {
                        expected[at as usize] = source.pixels[y * WIDTH as usize + x];
                    }
                }
            }
            assert_eq!(landed, expected);
            assert!(surface_cache::get(&state, MID, WIDTH, HEIGHT).is_none());
        }
    }

    #[test]
    fn surface_native_publication_unlicensed_pages_do_not_publish() {
        let (mut state, mut host, source) = fixture(p::MTL_FORMAT_A8_UNORM);
        state.mappings.get_mut(&MID).unwrap().page_entries.clear();
        let before = state.host_writes.epoch();
        assert!(!store(&mut state, &mut host, MID, WIDTH, HEIGHT, &source));
        let mut unchanged = [0; 1024];
        host.read_gpa(GPA, &mut unchanged).unwrap();
        assert_eq!(unchanged, [0xee; 1024]);
        assert_eq!(state.host_writes.epoch(), before);
        assert!(surface_cache::get(&state, MID, WIDTH, HEIGHT).is_some());
    }

    #[test]
    fn surface_native_publication_keeps_native_precision_without_changing_scanout_copy() {
        for format in [p::MTL_FORMAT_R16_FLOAT, p::MTL_FORMAT_RG16_FLOAT,
                       p::MTL_FORMAT_RGBA16_FLOAT, p::MTL_FORMAT_RGBA32_FLOAT] {
            assert!(required(format), "{format:#x} must not narrow through RGBA8");
        }
        for format in [p::MTL_FORMAT_RGBA8_UNORM, p::MTL_FORMAT_BGRA8_UNORM,
                       p::MTL_FORMAT_BGRA8_UNORM_SRGB] {
            assert!(!required(format), "the existing scanout byte-copy contract stays intact");
        }
        assert_eq!(p::render_target_bpp(p::MTL_FORMAT_A8_UNORM), None);
    }
}
