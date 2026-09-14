//! Packed, planar, or leased published-resident Metal sampled bindings.
//!
//! Packed reuse normally removes native allocation/upload only. Direct,
//! unreinterpreted mapper-ref textures can also avoid staging under the shared
//! gather witness and its guest-byte audit, within a live decoded render pass.
//! No resource lifetime or IOSurface construction descriptor extends that scope.
//! All other routes still stage before
//! exact byte/layout comparison. Mapping read-elision counters are separate
//! from the upload-only reuse-byte counters.

use super::*;
use crate::backend::metal::abi::{
    ReimsVgpuPackedSampledImage, ReimsVgpuSampledImage, REIMS_VGPU_BINDING_TEXTURE_BASE,
};
use crate::backend::metal::planar::SampledImage;
use crate::backend::metal::{packed, resident::PublishedSample};
use crate::runtime::compute_exec::{
    metal::{try_stage_planar_sampled_in_scope, MetalStage}, stage_texture_raw,
};
use std::sync::Arc;

mod mapping;

pub(super) enum SampledUpload {
    Packed {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        pixel_format: u32,
        bytes_per_row: u32,
    },
    ImmutablePacked(Arc<packed::SampledImage>),
    Imported(Arc<mapping::ImportedSample>),
    Planar(Arc<SampledImage>),
    Resident(crate::backend::metal::resident::PublishedSample),
}

#[derive(Default)]
pub(crate) struct ImportedReads {
    sources: Vec<Arc<mapping::ImportedSample>>,
}

impl ImportedReads {
    pub(super) fn retain(&mut self, groups: [&[SampledUpload]; 2]) {
        for upload in groups.into_iter().flatten() {
            if let SampledUpload::Imported(source) = upload {
                if !self.sources.iter().any(|held| held.same_view(source)) {
                    self.sources.push(Arc::clone(source));
                }
            }
        }
    }

    pub(crate) fn validate<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<(), crate::backend::metal::error::Status> {
        for source in &self.sources {
            source.revalidate(state, host)?;
        }
        for source in &self.sources {
            source.check(state)?;
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.sources.clear();
    }
}

impl SampledUpload {
    pub fn byte_len(&self) -> u64 {
        match self {
            Self::Packed { bytes, .. } => bytes.len() as u64,
            Self::ImmutablePacked(image) => image.byte_len(),
            Self::Imported(image) => image.byte_len(),
            Self::Planar(image) => image.layout().planes.iter().map(|plane| plane.size).sum(),
            Self::Resident(image) => image.byte_len(),
        }
    }

    pub fn image(&self, index: u32) -> ReimsVgpuSampledImage {
        let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + index;
        match self {
            Self::Imported(image) => ReimsVgpuSampledImage::ImportedRead {
                binding,
                image: image.image(),
            },
            Self::ImmutablePacked(image) => ReimsVgpuSampledImage::Resident {
                binding,
                image: PublishedSample::packed(Arc::clone(image)),
            },
            Self::Resident(image) => ReimsVgpuSampledImage::Resident {
                binding,
                image: image.clone(),
            },
            Self::Planar(image) => ReimsVgpuSampledImage::Planar {
                binding,
                image: image.clone(),
            },
            Self::Packed { bytes, width, height, pixel_format, bytes_per_row } => {
                ReimsVgpuSampledImage::Packed(ReimsVgpuPackedSampledImage {
                    binding,
                    width: *width,
                    height: *height,
                    rgba8: bytes.as_ptr(),
                    len: bytes.len(),
                    pixel_format: *pixel_format,
                    bytes_per_row: *bytes_per_row,
                    data: bytes.as_ptr(),
                    data_len: bytes.len(),
                })
            }
        }
    }
}

/// Seal the complete sampled-input set after all staging/debt resolution and
/// before encoding. Revalidating one source may retire another alias, so the
/// final current-identity pass is read-only and covers the whole set.
#[cfg(test)]
pub(super) fn validate_imported<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    groups: [&[SampledUpload]; 2],
) -> Result<(), crate::backend::metal::error::Status> {
    for image in groups.into_iter().flatten() {
        if let SampledUpload::Imported(image) = image {
            image.revalidate(state, host)?;
        }
    }
    for image in groups.into_iter().flatten() {
        if let SampledUpload::Imported(image) = image {
            image.check(state)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn load<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<SampledUpload> {
    load_in_scope(state, host, task_id, texture_ref, None)
}

pub(super) fn load_in_scope<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    scope: Option<&crate::runtime::draw::SnapshotScopeRef>,
) -> Option<SampledUpload> {
    match try_stage_planar_sampled_in_scope(state, host, task_id, texture_ref, scope) {
        Ok(Some(image)) => return Some(SampledUpload::Planar(image)),
        Ok(None) => {}
        Err(reason) => {
            crate::observe::Emit::refusal("draw_mtl_planar_texture", &reason)
                .expect("a staging error is a refusal")
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail();
            return None;
        }
    }
    // Only a direct, unreinterpreted mapper-ref surface can use this route.
    // Views (including swizzles/format overrides), planar and native packed
    // images keep their existing conversion/staging owners.
    if let Some(image) = load_resident(state, host, task_id, texture_ref) {
        return Some(SampledUpload::Resident(image));
    }
    match mapping::load(state, host, task_id, texture_ref, scope) {
        Ok(Some(image)) => return Some(image),
        Ok(None) => {}
        Err(reason) => {
            crate::observe::Emit::refusal("draw_mtl_packed_mapping", &reason)
                .expect("mapping read refusal")
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail();
            return None;
        }
    }
    let native_packed = objects::lookup_list_entry(state, host, task_id, texture_ref)
        .is_some_and(|entry| match entry.object_type {
            OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS => {
                objects::read_descriptor(state, host, task_id, &entry)
                    .and_then(|bytes| decode_texture_descriptor(&bytes).ok())
                    .is_some_and(|texture| {
                        texture.mipmap_level_count == 1
                            && texture.depth == 1
                            && texture.sample_count == Some(1)
                    })
            }
            crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW => {
                buffer_texture_descriptor(state, host, task_id, texture_ref, None).is_some()
            }
            _ => false,
        });
    if native_packed {
        // Reuse the raw staging owner: it pays both texture/buffer writeback
        // debts and removes row padding without quantizing the guest format.
        // Multi-level and view textures retain their existing loading path.
        let staged = match stage_texture_raw::<MetalStage, _>(
            state, host, task_id, texture_ref, 0, false,
        ) {
            Ok(staged) => staged,
            Err(reason) => {
                crate::observe::Emit::refusal("draw_mtl_packed_texture", &reason)
                    .expect("a staging error is a refusal")
                    .field("task", task_id)
                    .field("ref", texture_ref)
                    .fail();
                return None;
            }
        };
        let Some(bytes_per_row) =
            pixel_format::tight_row_bytes(staged.width, staged.pixel_format)
        else {
            crate::observe::Emit::refusal(
                "draw_mtl_packed_texture",
                &EncodeStatus::BadArgs("draw_mtl_packed_texture_pitch"),
            )
            .expect("BadArgs is a refusal")
            .field("ref", texture_ref)
            .fail();
            return None;
        };
        return retain_packed(state, host, task_id, texture_ref, staged.bytes, packed::Layout {
            width: staged.width,
            height: staged.height,
            pixel_format: u32::from(staged.pixel_format),
            bytes_per_row,
        });
    }

    load_rgba_packed(state, host, task_id, texture_ref)
}

fn load_rgba_packed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) -> Option<SampledUpload> {
    let (width, height, bytes) = load_sampled_rgba(state, host, task_id, texture_ref)?;
    let Some(bytes_per_row) = width.checked_mul(RGBA8_BPP) else {
        crate::observe::Emit::refusal(
            "draw_mtl_sampled_texture",
            &EncodeStatus::BadArgs("draw_mtl_sampled_texture_pitch"),
        )
        .expect("BadArgs is a refusal")
        .field("ref", texture_ref)
        .fail();
        return None;
    };
    retain_packed(state, host, task_id, texture_ref, bytes, packed::Layout {
        width,
        height,
        pixel_format: 0,
        bytes_per_row,
    })
}

/// Packed and planar routes are disjoint for an immutable construction
/// descriptor. No other Metal owner currently uses this resource's rail slot.
/// One latest candidate follows deletion/reset, not a process cache or an LRU.
#[derive(Default)]
struct RetainedPacked {
    latest: Option<Arc<packed::SampledImage>>,
    mapping: Option<mapping::Proof>,
}

impl crate::model::RailResourceState for RetainedPacked {}

impl RetainedPacked {
    fn stage(
        &mut self,
        layout: packed::Layout,
        bytes: Vec<u8>,
        create: impl FnOnce(packed::Layout, Vec<u8>)
            -> Result<packed::SampledImage, crate::backend::metal::error::Status>,
    ) -> Result<Arc<packed::SampledImage>, crate::backend::metal::error::Status> {
        use crate::runtime::drain::{note_store_route, note_store_route_n};
        // An ordinary staging call may have used a different source route.
        // Only a completed pre/post mapping witness can restore this capability.
        self.mapping = None;
        let same = {
            let _compare = crate::runtime::chain_phase::CostSpan::new(
                "metal_packed_sampled_compare_us",
            );
            self.latest.as_ref().filter(|image| image.matches(layout, &bytes))
        };
        if let Some(image) = same {
            note_store_route("metal_packed_sampled_reuses");
            note_store_route_n("metal_packed_sampled_reuse_bytes", bytes.len() as u64);
            return Ok(Arc::clone(image));
        }
        note_store_route("metal_packed_sampled_misses");
        self.latest = None;
        let image = Arc::new(create(layout, bytes)?);
        self.latest = Some(Arc::clone(&image));
        Ok(image)
    }
}

/// This runs *after* the existing reader and paired debt settlement on every
/// load. Exact staged-byte equality is the only freshness test; neither a hash
/// nor the guest/host write witnesses license skipping a read here.
fn retain_packed<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
    bytes: Vec<u8>,
    layout: packed::Layout,
) -> Option<SampledUpload> {
    let fallback = |bytes| SampledUpload::Packed {
        bytes,
        width: layout.width,
        height: layout.height,
        pixel_format: layout.pixel_format,
        bytes_per_row: layout.bytes_per_row,
    };
    let Ok(resource) = objects::resolve_resource(state, host, task_id, texture_ref) else {
        // Legacy direct mapping refs can stage without naming a TaskResource.
        // They retain their ordinary upload path, not an invented lifetime.
        crate::runtime::drain::note_store_route("metal_packed_sampled_unowned");
        return Some(fallback(bytes));
    };
    let mut bytes = Some(bytes);
    let staged = resource.with_rail_state(|held: &mut RetainedPacked| {
        held.stage(layout, bytes.take().expect("one staging call"), packed::SampledImage::new)
    });
    match staged {
        Some(Ok(image)) => Some(SampledUpload::ImmutablePacked(image)),
        Some(Err(reason)) => {
            crate::observe::Emit::refusal("draw_mtl_packed_texture", &reason)
                .expect("native upload error is a refusal")
                .field("task", task_id)
                .field("ref", texture_ref)
                .fail();
            None
        }
        None => {
            crate::observe::Emit::refusal(
                "draw_mtl_packed_texture",
                &crate::backend::metal::error::Status::args(
                    "metal_packed_sampled_resource_state_conflict",
                ),
            )
            .expect("a conflicting owner is a refusal")
            .field("task", task_id)
            .field("ref", texture_ref)
            .fail();
            Some(fallback(bytes.expect("a conflicting slot did not stage")))
        }
    }
}

fn load_resident<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<crate::backend::metal::resident::PublishedSample> {
    let mapping = objects::resolve_mapper_ref_texture(state, host, task_id, texture_ref)?;
    let (key, generation) = sampled_surface_frame(state, host, mapping, None)?;
    if crate::runtime::surface_cache::get_shared(state, mapping, key.width, key.height).is_some() {
        return None;
    }
    let image = crate::backend::metal::resident::sample_published_rgba8(&key, generation)?;
    crate::runtime::drain::note_store_route("metal_sampled_native_resident");
    crate::runtime::drain::note_store_route_n("metal_sampled_surface_bytes", image.byte_len());
    Some(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::protocol::endian::{st16, st32, st64};
    use crate::runtime::decode::resource::*;
    use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
    use crate::runtime::host::FakeHost;
    use reims_vgpu_wire::ops::texture::WideTextureDescriptorBody as W;
    use std::mem::offset_of;

    #[test]
    fn historical_resident_lease_survives_but_current_sampling_reads_guest_pixels() {
        use crate::backend::metal::{resident, runtime::system_device};
        use crate::protocol::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use foreign_types::ForeignType;
        use objc::rc::{autoreleasepool, WeakPtr};

        autoreleasepool(|| {
            let device = system_device().expect("Metal device");
            let mut host = FakeHost::new();
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));
            let mapping = 0xffff_a112;
            let reference = 7;
            let key = resident::ResidentColorKey::for_surface(mapping, 4, 4);
            let mut desc = [0u8; 0x20];
            st32(&mut desc, mapping);
            st16(&mut desc[0x16..], pixel_format::MTL_FORMAT_BGRA8_UNORM);
            st32(&mut desc[0x18..], 4);
            st32(&mut desc[0x1c..], 4);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &desc);
            let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
            st32(
                &mut entry,
                u32::from(OBJECT_TYPE_MAPPER_REF_TEXTURE) | ((desc.len() as u32) << 8),
            );
            st64(&mut entry[4..], 0x200);
            write_task_gva_arm64e(
                &mut host,
                &state.tasks[1],
                list_object_entry_offset(reference, 32).unwrap(),
                &entry,
            );
            assert!(state.map_surface(mapping));
            let mapped = state.mappings.get_mut(&mapping).unwrap();
            mapped.mapped = true;
            mapped.mapping_internal = 1;
            mapped.page_entries = vec![(9 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            assert!(state.set_mapping_geom(mapping, 4, 4, pixel_format::MTL_FORMAT_BGRA8_UNORM));
            let guest_bgra = [3, 17, 91, 255].repeat(16);
            let guest_rgba = swap_rb_channels(&guest_bgra);
            let guest_page = 9 << PAGE_SHIFT_ARM64E;
            host.write_gpa(
                guest_page,
                &[3, 17, 91, 255].repeat(state.page_size() as usize / 4),
            )
            .unwrap();
            assert!(crate::runtime::surface_cache::cede_surface_to_resident(
                &mut state, mapping, 4, 4,
            ));
            let generation =
                crate::runtime::surface_cache::frame_generation(&state, mapping, 4, 4).unwrap();
            let pixels: Vec<_> = (0..64).map(|i| (i * 3) as u8).collect();
            let weak = autoreleasepool(|| {
                let texture =
                    resident::create(device, &key, ::metal::MTLPixelFormat::RGBA8Unorm, 4).unwrap();
                texture.replace_region(
                    ::metal::MTLRegion::new_2d(0, 0, 4, 4),
                    0,
                    pixels.as_ptr().cast(),
                    16,
                );
                resident::published(&key, generation);
                unsafe { WeakPtr::new(texture.as_ptr().cast()) }
            });
            let packed_bytes = |upload| match upload {
                SampledUpload::Packed { bytes, .. } => bytes,
                SampledUpload::ImmutablePacked(image) => image.bytes().to_vec(),
                _ => panic!("unproven resident must use the existing packed fallback"),
            };
            assert_eq!(
                packed_bytes(load(&mut state, &mut host, 1, reference).unwrap()),
                guest_rgba
            );
            let token =
                crate::runtime::mapper::ensure_guest_write_token(&mut state, &mut host, mapping)
                    .unwrap();
            state
                .mappings
                .get_mut(&mapping)
                .unwrap()
                .guest_write_gen_at_store = host.guest_write_gen(token).unwrap();
            let upload = load(&mut state, &mut host, 1, reference).unwrap();
            assert!(!matches!(&upload, SampledUpload::Resident(_)));
            assert_eq!(upload.byte_len(), 64);
            assert_eq!(
                load_sampled_rgba(&mut state, &mut host, 1, reference)
                    .unwrap()
                    .2,
                guest_rgba,
                "ordinary fallback cannot use a historical frame as current guest pixels",
            );
            let image = ReimsVgpuSampledImage::Resident {
                binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
                image: resident::sample_published_rgba8(&key, generation).unwrap(),
            };
            assert_eq!(image.binding(), REIMS_VGPU_BINDING_TEXTURE_BASE + 3);
            assert!(
                !image.needs_completion(),
                "a read lease preserves existing batching"
            );
            drop(upload);
            assert!(
                resident::take(&key, generation).is_none(),
                "binding retains the read lease"
            );

            // A format/swizzle view must still pass through the original view
            // interpreter, even though its base has an eligible resident.
            let view_ref = 8;
            let mut view = vec![0u8; TEXTURE_VIEW_MIN_SWIZZLE];
            st32(
                &mut view[TEXTURE_VIEW_DESC_OPCODE..],
                TEXTURE_VIEW_OPCODE_SWIZZLE,
            );
            let view_len = view.len() as u32;
            st32(&mut view[TEXTURE_VIEW_DESC_LEN..], view_len);
            st32(&mut view[TEXTURE_VIEW_DESC_TEXTURE_REF..], view_ref);
            st32(&mut view[TEXTURE_VIEW_DESC_BASE_REF..], reference);
            st16(
                &mut view[TEXTURE_VIEW_DESC_PIXEL_FORMAT..],
                pixel_format::MTL_FORMAT_BGRA8_UNORM,
            );
            st16(
                &mut view[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
                TEXTURE_VIEW_MTL_TYPE_2D,
            );
            st64(&mut view[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
            st64(&mut view[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
            view[TEXTURE_VIEW_DESC_SWIZZLE..TEXTURE_VIEW_DESC_SWIZZLE + 4]
                .copy_from_slice(&[4, 3, 2, 5]);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x300, &view);
            st32(
                &mut entry,
                u32::from(OBJECT_TYPE_TEXTURE_VIEW) | (view_len << 8),
            );
            st64(&mut entry[4..], 0x300);
            write_task_gva_arm64e(
                &mut host,
                &state.tasks[1],
                list_object_entry_offset(view_ref, 32).unwrap(),
                &entry,
            );
            assert!(load_resident(&mut state, &host, 1, view_ref).is_none());
            assert_eq!(
                packed_bytes(load(&mut state, &mut host, 1, view_ref).unwrap()),
                load_sampled_rgba(&mut state, &mut host, 1, view_ref)
                    .unwrap()
                    .2,
            );

            // Neither a CPU repaint nor a new publication can serve this old
            // native allocation as the new frame.
            host.guest_wrote_page(guest_page);
            assert_eq!(
                packed_bytes(load(&mut state, &mut host, 1, reference).unwrap()),
                guest_rgba
            );
            state
                .mappings
                .get_mut(&mapping)
                .unwrap()
                .guest_write_gen_at_store = host.guest_write_gen(token).unwrap();
            assert!(crate::runtime::surface_cache::cede_surface_to_resident(
                &mut state, mapping, 4, 4,
            ));
            assert_eq!(
                packed_bytes(load(&mut state, &mut host, 1, reference).unwrap()),
                guest_rgba
            );
            crate::runtime::surface_cache::store(
                &mut state,
                mapping,
                4,
                4,
                [33, 44, 55, 255].repeat(16),
            );
            assert_eq!(
                packed_bytes(load(&mut state, &mut host, 1, reference).unwrap()),
                guest_rgba,
                "a historical host-frame cache cannot override guest bytes either",
            );
            resident::forget(mapping);
            assert!(
                !weak.load().is_null(),
                "dropping the source registry cannot drop a binding"
            );
            drop(image);
            assert!(
                weak.load().is_null(),
                "no native sample escapes its last owner"
            );
        });
    }

    #[test]
    fn planar_sampled_binding_keeps_whole_surface_owned() {
        use crate::backend::metal::planar::tests::image;
        use crate::protocol::planar::{BackingFormat, SampleFormat};

        for format in [SampleFormat::Ycbcr10_420TwoPlane, SampleFormat::Rgb10_420TwoPlane] {
            let image = Arc::new(image(format, BackingFormat::VideoRange));
            let lifetime = Arc::downgrade(&image);
            let upload = SampledUpload::Planar(image);
            assert_eq!(upload.byte_len(), 2048);
            let binding = upload.image(5);
            drop(upload);
            let ReimsVgpuSampledImage::Planar { binding: slot, image } = &binding else {
                panic!("whole-surface samples must not become packed RGBA8");
            };
            assert_eq!(*slot, REIMS_VGPU_BINDING_TEXTURE_BASE + 5);
            assert_eq!(image.description().format, format);
            assert_eq!(image.layout().planes[0].offset, 128);
            assert_eq!(image.layout().planes[1].offset, 2176);
            assert!(lifetime.upgrade().is_some());
            drop(binding);
            assert!(lifetime.upgrade().is_none());
        }
    }

    #[test]
    fn linear_sampled_upload_preserves_float_precision_channels_and_padded_rows() {
        for (format, bpp) in [
            (pixel_format::MTL_FORMAT_R16_FLOAT, 2usize),
            (pixel_format::MTL_FORMAT_RG8_UNORM, 2),
            (pixel_format::MTL_FORMAT_RGBA16_FLOAT, 8),
            (pixel_format::MTL_FORMAT_RGBA32_FLOAT, 16),
        ] {
            let mut host = FakeHost::new();
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));
            let width = 4;
            let height = 2;
            let tight = width * bpp;
            let pitch = tight + 16;
            let mut backing = vec![0xEE; pitch + tight];
            let expected: Vec<u8> = (0..tight * height).map(|v| v as u8).collect();
            for y in 0..height {
                backing[y * pitch..y * pitch + tight]
                    .copy_from_slice(&expected[y * tight..(y + 1) * tight]);
            }
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
            let mut descriptor = vec![0; TEXTURE_DESC_BASE_LEN];
            st64(&mut descriptor[LINEAR_DESC_SIZE..], backing.len() as u64);
            st32(&mut descriptor[LINEAR_DESC_HANDLE..], 5);
            st16(&mut descriptor[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 1);
            st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], backing.len() as u32);
            st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], pitch as u32);
            st32(&mut descriptor[TEXTURE_DESC_WIDTH..], width as u32);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], height as u32);
            st32(&mut descriptor[TEXTURE_DESC_HEIGHT + 4..], 1);
            st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], format);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_WIDTH..], width as u32);
            st32(&mut descriptor[TEXTURE_DESC_TRAILER_HEIGHT..], height as u32);
            st16(&mut descriptor[TEXTURE_DESC_SAMPLE_COUNT..], 1);
            write_task_gva_arm64e(&mut host, &state.tasks[1], 0x200, &descriptor);
            let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
            st32(&mut entry, u32::from(OBJECT_TYPE_TEXTURE) | ((descriptor.len() as u32) << 8));
            st64(&mut entry[4..], 0x200);
            write_task_gva_arm64e(
                &mut host, &state.tasks[1], list_object_entry_offset(7, 32).unwrap(), &entry,
            );
            let upload = load(&mut state, &mut host, 1, 7).expect("native linear texture");
            let SampledUpload::ImmutablePacked(image) = &upload else {
                panic!("linear textures are packed");
            };
            assert_eq!(image.bytes(), expected, "no channel expansion or float-to-UNORM conversion");
            assert_eq!(image.layout().pixel_format, u32::from(format));
            assert_eq!(image.layout().bytes_per_row, tight as u32);
            assert_eq!(image.byte_len(), (tight * height) as u64);
            let ReimsVgpuSampledImage::Resident { image, .. } = upload.image(0) else {
                panic!("an immutable packed upload must bind read-only");
            };
            assert_eq!(image.texture().pixel_format() as u32, u32::from(format));
        }
    }

    #[test]
    fn metal_buffer_texture_upload_preserves_native_channels_offset_and_pitch() {
        for format in [pixel_format::MTL_FORMAT_RGBA16_UNORM, pixel_format::MTL_FORMAT_RGBA16_FLOAT] {
            let mut host = FakeHost::new();
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
            define_task_pages_arm64e(&mut host, &mut state, 4, 8);
            assert!(state.set_object_list(1, 0, 32));

            const OFFSET: usize = 8;
            const TIGHT: usize = 16;
            const PITCH: usize = 32;
            let mut backing = vec![0xEE; OFFSET + PITCH + TIGHT];
            let expected: Vec<u8> = (0..2 * TIGHT).map(|v| v as u8).collect();
            for row in 0..2 {
                backing[OFFSET + row * PITCH..OFFSET + row * PITCH + TIGHT]
                    .copy_from_slice(&expected[row * TIGHT..(row + 1) * TIGHT]);
            }
            write_task_gva_arm64e(&mut host, &state.tasks[1], 5 << PAGE_SHIFT_ARM64E, &backing);
            let mut buffer = [0u8; 16];
            st64(&mut buffer, backing.len() as u64);
            st32(&mut buffer[8..], 5);

            let mut body = vec![0u8; crate::runtime::heap_query::WIDE_TEXTURE_BODY_LEN];
            body[offset_of!(W, type_and_flags)] = 2;
            st16(&mut body[offset_of!(W, pixel_format)..], format);
            st32(&mut body[offset_of!(W, width)..], 2);
            st32(&mut body[offset_of!(W, height)..], 2);
            st32(&mut body[offset_of!(W, depth)..], 1);
            st16(&mut body[offset_of!(W, mipmap_level_count)..], 1);
            st16(&mut body[offset_of!(W, sample_count)..], 1);
            st16(&mut body[offset_of!(W, array_length)..], 1);
            let mut texture = vec![0u8; BUF_TEX_WIDE_LEN];
            st32(&mut texture[TEXTURE_VIEW_DESC_OPCODE..], TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE);
            st32(&mut texture[TEXTURE_VIEW_DESC_LEN..], BUF_TEX_WIDE_LEN as u32);
            st32(&mut texture[TEXTURE_VIEW_DESC_TEXTURE_REF..], 21);
            st32(&mut texture[BUF_TEX_DESC_BUFFER_REF..], 7);
            st64(&mut texture[BUF_TEX_DESC_OFFSET..], OFFSET as u64);
            st64(&mut texture[BUF_TEX_DESC_BYTES_PER_ROW..], PITCH as u64);
            texture[BUF_TEX_WIDE_DESC_BODY..].copy_from_slice(&body);

            for (reference, kind, gva, descriptor) in [
                (7, OBJECT_TYPE_BUFFER, 0x200, buffer.as_slice()),
                (21, OBJECT_TYPE_TEXTURE_VIEW, 0x300, texture.as_slice()),
            ] {
                write_task_gva_arm64e(&mut host, &state.tasks[1], gva, descriptor);
                let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
                st32(&mut entry, u32::from(kind) | ((descriptor.len() as u32) << 8));
                st64(&mut entry[4..], gva);
                write_task_gva_arm64e(
                    &mut host, &state.tasks[1],
                    list_object_entry_offset(reference, 32).unwrap(), &entry,
                );
            }
            let upload = load(&mut state, &mut host, 1, 21).expect("native buffer texture");
            let SampledUpload::ImmutablePacked(image) = &upload else {
                panic!("buffer textures must use packed uploads");
            };
            assert_eq!(image.bytes(), expected, "no UNORM8 conversion or row padding");
            assert_eq!(upload.byte_len(), expected.len() as u64);
            assert_eq!((image.layout().width, image.layout().height), (2, 2));
            assert_eq!(image.layout().pixel_format, u32::from(format));
            assert_eq!(image.layout().bytes_per_row, TIGHT as u32);
            let binding = upload.image(3);
            assert_eq!(binding.binding(), REIMS_VGPU_BINDING_TEXTURE_BASE + 3);
            assert!(!binding.needs_completion());
            let ReimsVgpuSampledImage::Resident { image, .. } = binding else {
                panic!("buffer textures must retain their native packed format");
            };
            assert_eq!(image.texture().pixel_format() as u32, u32::from(format));
        }
    }
}

#[cfg(test)]
mod packed_tests;
