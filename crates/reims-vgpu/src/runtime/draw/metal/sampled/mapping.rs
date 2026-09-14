//! Read elision for a direct, decoded, single-plane mapper-ref sample.
//!
//! This does not publish a CPU texture as a rendered frame. It witnesses the
//! exact guest window the ordinary RGBA loader reads, within one live decoded
//! render pass only. Host/guest write proofs,
//! their harvest semantics, and the audit remain gather_witness's contract.
//! Unknown layouts, imports or tracking retain full staging and byte comparison.

use super::*;
use crate::backend::metal::error::Status;
use crate::model::{BackingWalk, TaskResource};
use crate::runtime::draw::SnapshotScopeRef;
use crate::runtime::gather_witness::{
    note_scoped_gather, GatherKey, GatherOutcome, GatherRail, GatherWindow, GatheredIdentity,
};
use crate::runtime::guest_ram::GuestRun;
use crate::runtime::mapper;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Description {
    mapping_id: u32,
    width: u32,
    height: u32,
    format: u16,
}

impl Description {
    fn decode(resource: &TaskResource, reference: u32) -> Option<Self> {
        use crate::runtime::decode::resource::{Descriptor, IOSURFACE_TEX_MIN_LEN};
        use reims_vgpu_wire::ops::backed_texture;
        if resource.entry.object_type != OBJECT_TYPE_MAPPER_REF_TEXTURE {
            return None;
        }
        let Descriptor::IOSurfaceTexture {
            mapping_id,
            object_ref,
            pixel_format: format,
            width,
            height,
        } = resource.decoded().as_ref().ok()?
        else {
            return None;
        };
        if *object_ref != reference
            || crate::protocol::planar::SampleFormat::parse(*format).is_some()
            || !crate::model::scanout_extent_ok(*width, *height)
            || pixel_format::RowToRgba8::for_format(*format).is_none()
        {
            return None;
        }
        let bytes = &resource.descriptor;
        // The legacy record names only an IOSurface view (single-level by that
        // contract). For a serialized narrow record, admit the fully known
        // D2 shape. Extended/unknown tails remain on ordinary staging.
        if bytes.len() != IOSURFACE_TEX_MIN_LEN {
            if bytes.len() != 8 + backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN as usize
                || crate::protocol::endian::ld32(&bytes[8..])
                    != crate::protocol::planar::TYPE11_TEXTURE_ARGS_KIND
            {
                return None;
            }
            let op = reims_vgpu_wire::op::op(&bytes[8..], 0).ok()?;
            let body =
                reims_vgpu_wire::view::view::<backed_texture::IOSurfaceTextureBody>(op.payload)
                    .ok()?;
            let d = &body.desc;
            if body.object_ref.get() != reference
                || body.plane.get() != 0
                || d.texture_type() != 2
                || d.depth.get() != 1
                || d.mipmap_level_count.get() != 1
                || d.sample_count.get() != 1
                || d.array_length.get() != 1
            {
                crate::runtime::drain::note_store_route("metal_packed_mapping_shape_fallback");
                return None;
            }
            // Usage names capabilities, not current writers. The read witness
            // below, not a ShaderRead-only declaration, establishes currency.
            let usage = crate::protocol::texture_shape::TextureUsage(u32::from(d.usage()));
            if usage.undeclared() != 0 {
                crate::runtime::drain::note_store_route("metal_packed_mapping_usage_fallback");
                return None;
            }
            if d.unidentified_flags() != 0
                || d.unidentified_u64.get() != 0
                || d.resource_options.get() & !0x0331 != 0
                || d.cpu_cache_mode() > 1
                || d.storage_mode() > 2
                || d.hazard_tracking_mode() > 2
            {
                crate::runtime::drain::note_store_route("metal_packed_mapping_flags_fallback");
                return None;
            }
        }
        Some(Self {
            mapping_id: *mapping_id,
            width: *width,
            height: *height,
            format: *format,
        })
    }

    fn output(self) -> packed::Layout {
        packed::Layout {
            width: self.width,
            height: self.height,
            pixel_format: 0,
            bytes_per_row: self.width * RGBA8_BPP,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Key {
    description: Description,
    device_descriptor: Vec<u8>,
    map_generation: u32,
    content_generation: u32,
    surface_content_epoch: u32,
    mapping_internal: u64,
    page_table_gpa: Option<u64>,
    backing: Option<BackingWalk>,
    page_size: u64,
    pages: Arc<[u64]>,
    mapped_len: usize,
    base: u64,
    pitch: u32,
    end: u64,
    guest_read_bytes: u64,
}

pub(super) struct Proof {
    key: Key,
    identity: GatheredIdentity,
}

struct Window {
    key: Key,
    run: GuestRun,
    first_page: usize,
    end_page: usize,
}

/// A mutable guest read, not a retained pixel value. The resource and owned
/// import stay held while the synchronous encode runs under DeviceInner's
/// state lock; the native batch additionally retains the import through Metal
/// completion and its no-copy deallocator controls eventual unmapping.
pub(in crate::runtime::draw::metal) struct ImportedSample {
    key: Key,
    import: crate::runtime::guest_ram::ImportId,
    footprint: crate::runtime::guest_ram::GuestPageFootprint,
    _resource: Arc<TaskResource>,
    image: Arc<crate::backend::metal::mapped_sample::Image>,
}

impl ImportedSample {
    pub(super) fn same_view(&self, other: &Self) -> bool {
        self.import == other.import && self.key == other.key
    }

    fn prepare<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        resource: Arc<TaskResource>,
        description: Description,
    ) -> Result<Option<Self>, Status> {
        use crate::backend::metal::mapped_sample;
        if !mapped_sample::enabled() || !host.map_pages_owned() {
            return Ok(None);
        }
        let shape = mapped_sample::Layout {
            width: description.width,
            height: description.height,
            pitch: 0,
            format: description.format,
        };
        if shape.pixel_format().is_none() {
            return Ok(None);
        }
        let Some(window) = Window::resolve(state, host, description)? else {
            return Ok(None);
        };
        let layout = mapped_sample::Layout {
            pitch: window.key.pitch,
            ..shape
        };
        let Some(span) = layout.span() else {
            return Ok(None);
        };
        let surface =
            crate::protocol::iosurface_pages::decode_device_surface(&window.key.device_descriptor)
                .expect("resolved device descriptor");
        if window
            .key
            .base
            .checked_add(span)
            .is_none_or(|end| end > u64::from(surface.alloc_size))
        {
            return Ok(None);
        }
        let Some((import, footprint)) =
            mapper::ensure_owned_contig_import_with_footprint(state, host, description.mapping_id)
        else {
            return Ok(None);
        };
        let Ok(slice) = import.slice(window.key.base, span) else {
            return Ok(None);
        };
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return Ok(None);
        };
        let id = import.id();
        let image = match mapped_sample::Image::new(device, import, slice, layout) {
            Ok(image) => Arc::new(image),
            Err(status) => {
                crate::runtime::drain::note_store_route("metal_mapped_sample_fallback");
                crate::observe::Emit::refusal("metal_mapped_sample_fallback", &status)
                    .unwrap()
                    .field("mapping", description.mapping_id)
                    .fail_once(u64::from(description.mapping_id));
                return Ok(None);
            }
        };
        let captured = Self {
            key: window.key,
            import: id,
            footprint,
            _resource: resource,
            image,
        };
        captured.check(state)?;
        crate::runtime::drain::note_store_route("metal_mapped_sample_reads");
        crate::runtime::drain::note_store_route_n(
            "metal_mapped_sample_guest_bytes",
            captured.byte_len(),
        );
        Ok(Some(captured))
    }

    pub(super) fn image(&self) -> Arc<crate::backend::metal::mapped_sample::Image> {
        Arc::clone(&self.image)
    }

    pub(super) fn byte_len(&self) -> u64 {
        self.key.guest_read_bytes
    }

    pub(super) fn check(&self, state: &DeviceState) -> Result<(), Status> {
        let key = &self.key;
        let valid = self.image.live()
            && state
                .mappings
                .get(&key.description.mapping_id)
                .is_some_and(|m| {
                    m.mapped
                        && m.has_geom
                        && m.map_generation == key.map_generation
                        && m.mapping_internal == key.mapping_internal
                        && m.page_table_gpa == key.page_table_gpa
                        && (m.width, m.height, m.format)
                            == (
                                key.description.width,
                                key.description.height,
                                key.description.format,
                            )
                        && m.device_desc_complete() == Some(key.device_descriptor.as_slice())
                        && m.contig_import.as_ref().is_some_and(|import| {
                            import.id() == self.import && !import.is_retired()
                        })
                        && m.contig_footprint
                            .as_ref()
                            .is_some_and(|pages| pages.same_allocation(&self.footprint))
                        && crate::runtime::mapping_write::mapper_ref_texture_sample_window(
                            m, m.width, m.height, m.format,
                        ) == Some((key.base, key.pitch, key.end))
                });
        if valid {
            Ok(())
        } else {
            Err(Status::args("metal_mapped_sample_capture_stale")
                .field("mapping", key.description.mapping_id))
        }
    }

    pub(super) fn revalidate<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<(), Status> {
        let mapping = self.key.description.mapping_id;
        crate::runtime::writeback_debt::settle_for_mapping(
            state,
            host,
            mapping,
            crate::runtime::render_writeback::SettleSite::SampledMappingRead,
        );
        if !refresh_backing(state, host, mapping)?
            || !mapper::revalidate_mapping_pages(state, host, mapping)
        {
            return Err(
                Status::args("metal_mapped_sample_mapping_changed").field("mapping", mapping)
            );
        }
        self.check(state)
    }
}

/// MappingInternal mappings are revalidated by the contig-view owner.
/// Backing-derived mappings additionally require their *entire* task PTE walk.
fn refresh_backing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &M,
    mapping_id: u32,
) -> Result<bool, Status> {
    let Some(mapping) = state.mappings.get(&mapping_id) else {
        return Ok(false);
    };
    if !mapping.mapped {
        return Ok(false);
    }
    if mapping.backing_walk.is_none() {
        return Ok(mapping.mapping_internal != 0);
    }
    let needs_backing = mapping.mapping_internal == 0 && mapping.page_entries.is_empty();
    match mapper::mapping_pages_verdict(state, host, mapping_id) {
        mapper::PagesVerdict::Ours => Ok(true),
        mapper::PagesVerdict::Unwitnessed(_) if !needs_backing => Ok(false),
        mapper::PagesVerdict::Unwitnessed(_) | mapper::PagesVerdict::Drifted => {
            // The witness already invalidated the old list and alias. Rebuild
            // through the backing owner, never synthesize entries from a pointer.
            // An empty backing also retries after a temporarily missing PTE;
            // refusing once must not strand a subsequently repaired mapping.
            if objects::resolve_backing_force(state, host, mapping_id)
                && matches!(
                    mapper::mapping_pages_verdict(state, host, mapping_id),
                    mapper::PagesVerdict::Ours,
                )
            {
                Ok(true)
            } else {
                Err(Status::args("metal_packed_mapping_backing_changed")
                    .field("mapping", mapping_id))
            }
        }
    }
}

impl Window {
    fn resolve<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        description: Description,
    ) -> Result<Option<Self>, Status> {
        use crate::backend::{Backend as _, GuestWriteReach};
        let mid = description.mapping_id;
        crate::runtime::writeback_debt::settle_for_mapping(
            state,
            host,
            mid,
            crate::runtime::render_writeback::SettleSite::SampledMappingRead,
        );
        if !refresh_backing(state, host, mid)? {
            return Ok(None);
        }
        let Some((ptr, len, pages)) = mapper::ensure_contig_view_with_pages(state, host, mid)
        else {
            return Ok(None);
        };
        // A publication is a different source, even when its bytes happen to
        // match RAM. Preserve the existing CPU-publication/resident priority.
        if sampled_surface_frame(state, host, mid, None).is_some() {
            return Ok(None);
        }
        let Some(mapping) = state.mappings.get(&mid) else {
            return Ok(None);
        };
        if !mapping.mapped
            || !mapping.has_geom
            || (mapping.width, mapping.height, mapping.format)
                != (description.width, description.height, description.format)
            || objects::mapping_is_multiplanar(mapping)
            || mapping
                .backing_walk
                .is_some_and(|walk| walk.map_generation != mapping.map_generation)
        {
            return Ok(None);
        }
        let Some(device_descriptor) = mapping.device_desc_complete() else {
            return Ok(None);
        };
        let Some(surface) =
            crate::protocol::iosurface_pages::decode_device_surface(device_descriptor)
        else {
            return Ok(None);
        };
        if surface.plane_count > 1
            || surface.alloc_size == 0
            || (surface.width, surface.height) != (description.width, description.height)
            || objects::device_desc_format_to_mtl(surface.pixel_format) != description.format
            || u64::from(surface.alloc_size) > len as u64
        {
            return Ok(None);
        }
        let Some((base, pitch, end)) =
            crate::runtime::mapping_write::mapper_ref_texture_sample_window(
                mapping,
                description.width,
                description.height,
                description.format,
            )
        else {
            return Ok(None);
        };
        let Some(span) = end.checked_sub(base).filter(|&span| span != 0) else {
            return Ok(None);
        };
        if end > u64::from(surface.alloc_size) {
            return Ok(None);
        }
        let Some(tight) = pixel_format::tight_row_bytes(description.width, description.format)
        else {
            return Ok(None);
        };
        if pitch < tight {
            return Ok(None);
        }
        let Some(run) = GuestRun::in_mapping(ptr, len as u64, base, span) else {
            return Ok(None);
        };
        let page_size = state.page_size();
        let first_page = (base / page_size) as usize;
        let end_page = end.div_ceil(page_size) as usize;
        let Some(footprint) = pages.get(first_page..end_page) else {
            return Ok(None);
        };
        let backend = crate::backend::selected();
        if backend.guest_writes_outstanding()
            && !matches!(
                backend.guest_writes_reaching(footprint),
                GuestWriteReach::Disjoint
            )
        {
            // A revalidation can reveal different pages from the settle's
            // reach. Do not vouch for them while a submitted write can reach.
            return Ok(None);
        }
        Ok(Some(Self {
            key: Key {
                description,
                device_descriptor: device_descriptor.to_vec(),
                map_generation: mapping.map_generation,
                content_generation: mapping.content_generation,
                surface_content_epoch: mapping.surface_content_epoch,
                mapping_internal: mapping.mapping_internal,
                page_table_gpa: mapping.page_table_gpa,
                backing: mapping.backing_walk,
                page_size,
                pages,
                mapped_len: len,
                base,
                pitch,
                end,
                guest_read_bytes: u64::from(tight) * u64::from(description.height),
            },
            run,
            first_page,
            end_page,
        }))
    }

    fn observe<M: HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        scope: Option<&SnapshotScopeRef>,
    ) -> GatherOutcome {
        note_scoped_gather(
            state,
            host,
            GatherRail::MapperRefTexture,
            GatherKey::Mapping {
                mid: self.key.description.mapping_id,
                base_off: self.key.base,
            },
            GatherWindow {
                gpas: &self.key.pages[self.first_page..self.end_page],
                runs: std::slice::from_ref(&self.run),
                span: self.key.end - self.key.base,
                page_size: self.key.page_size as usize,
            },
            scope,
        )
    }
}

pub(super) fn load<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    scope: Option<&SnapshotScopeRef>,
) -> Result<Option<SampledUpload>, Status> {
    let Ok(resource) = objects::resolve_resource(state, host, task_id, texture_ref) else {
        return Ok(None);
    };
    let Some(description) = Description::decode(&resource, texture_ref) else {
        if resource.entry.object_type == OBJECT_TYPE_MAPPER_REF_TEXTURE {
            crate::runtime::drain::note_store_route("metal_packed_mapping_descriptor_fallback");
        }
        return Ok(None);
    };
    if objects::resolve_mapper_ref_texture_resource(state, task_id, texture_ref, &resource)
        != Some(description.mapping_id)
    {
        return Ok(None);
    }
    if let Some(image) = ImportedSample::prepare(state, host, Arc::clone(&resource), description)? {
        return Ok(Some(SampledUpload::Imported(Arc::new(image))));
    }
    load_with(state, host, &resource, description, scope, |state, host| {
        super::load_rgba_packed(state, host, task_id, texture_ref)
    })
    .map(Some)
}

fn load_with<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    resource: &TaskResource,
    description: Description,
    scope: Option<&SnapshotScopeRef>,
    stage: impl FnOnce(&mut DeviceState, &mut M) -> Option<SampledUpload>,
) -> Result<SampledUpload, Status> {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    let scope_id = scope.and_then(SnapshotScopeRef::current);
    if scope_id.is_none() {
        note_store_route("metal_packed_mapping_unscoped");
    }
    let before = Window::resolve(state, host, description)?.map(|window| {
        let seen = window.observe(state, host, scope);
        (window.key, seen)
    });
    if before.is_none() {
        note_store_route("metal_packed_mapping_unwitnessed");
    }
    let hit = resource
        .with_rail_state(|held: &mut RetainedPacked| {
            if let (Some((key, seen)), Some(proof), Some(image)) =
                (&before, &held.mapping, &held.latest)
            {
                if seen.cpu_read_vouched()
                    && scope_id.is_some()
                    && scope.and_then(SnapshotScopeRef::current) == scope_id
                    && seen.identity == proof.identity
                    && *key == proof.key
                    && image.layout() == description.output()
                {
                    return Some(Arc::clone(image));
                }
            }
            held.mapping = None;
            None
        })
        .flatten();
    if let Some(image) = hit {
        let (key, seen) = before.expect("a hit has a window");
        note_store_route("metal_packed_mapping_reuses");
        note_store_route_n(
            "metal_packed_mapping_staging_bytes_avoided",
            key.guest_read_bytes,
        );
        note_store_route_n(
            "metal_packed_mapping_guest_bytes_avoided",
            key.guest_read_bytes.saturating_sub(seen.audit_bytes),
        );
        note_store_route_n("metal_packed_mapping_hit_audit_bytes", seen.audit_bytes);
        // A strided audit also reads inter-row padding the normal contig
        // reader skips. Report that extra work separately: net guest traffic
        // saved is guest_bytes_avoided minus extra_audit_bytes.
        note_store_route_n(
            "metal_packed_mapping_extra_audit_bytes",
            seen.audit_bytes.saturating_sub(key.guest_read_bytes),
        );
        return Ok(SampledUpload::ImmutablePacked(image));
    }
    note_store_route("metal_packed_mapping_stages");
    if let Some((_, seen)) = &before {
        note_store_route_n("metal_packed_mapping_extra_audit_bytes", seen.audit_bytes);
    }
    let image = stage(state, host).ok_or_else(|| {
        Status::args("metal_packed_mapping_staging_failed").field("mapping", description.mapping_id)
    })?;
    if let (Some((key, seen)), SampledUpload::ImmutablePacked(snapshot)) = (before, &image) {
        if let Some(after) = Window::resolve(state, host, description)? {
            let final_seen = after.observe(state, host, scope);
            note_store_route_n(
                "metal_packed_mapping_extra_audit_bytes",
                final_seen.audit_bytes,
            );
            if key == after.key
                && scope_id.is_some()
                && scope.and_then(SnapshotScopeRef::current) == scope_id
                && seen.cpu_read_settled
                && final_seen.cpu_read_vouched()
                && seen.identity == final_seen.identity
                && snapshot.layout() == description.output()
            {
                resource.with_rail_state(|held: &mut RetainedPacked| {
                    if held
                        .latest
                        .as_ref()
                        .is_some_and(|latest| Arc::ptr_eq(latest, snapshot))
                    {
                        held.mapping = Some(Proof {
                            key,
                            identity: final_seen.identity,
                        });
                    }
                });
            }
        }
    }
    // An unvouched snapshot remains usable only by the original full-byte
    // comparison cache; no read-elision capability was retained for it.
    Ok(image)
}

#[cfg(test)]
mod tests;
