//! A pass-owned checked mapped destination, recorded before the first draw and
//! redeemed only while the same allocation and layout remain authoritative.

use super::*;
use crate::backend::metal::util::Status;
use crate::backend::metal::{guest_writeback, resident};
use crate::runtime::guest_ram::{GuestPageFootprint, ImportId};
use ::metal::{Texture, TextureRef};

pub(crate) struct Store {
    mapping: u32,
    width: u32,
    height: u32,
    format: u16,
    base: u64,
    pitch: u32,
    span_end: u64,
    footprint: GuestPageFootprint,
    import: ImportId,
    vouched: mapper::PagesVouched,
    source: Texture,
    prepared: Option<guest_writeback::Prepared>,
}

impl Store {
    pub(crate) fn prepare<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        mapping: u32,
        width: u32,
        height: u32,
        source: &TextureRef,
    ) -> Option<Self> {
        if !guest_writeback::enabled() {
            return None;
        }
        match Self::prepare_checked(state, host, mapping, width, height, source) {
            Ok(store) => Some(store),
            Err(status) => {
                crate::observe::Emit::refusal("metal_gpu_store_fallback", status.as_ref())
                    .unwrap()
                    .field("mapping", mapping)
                    .fail_once(u64::from(mapping));
                None
            }
        }
    }

    fn prepare_checked<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        mapping: u32,
        width: u32,
        height: u32,
        source: &TextureRef,
    ) -> Result<Self, Box<Status>> {
        if !scanout_extent_ok(width, height) {
            return Err(Status::args("metal_gpu_store_geometry").into());
        }
        crate::runtime::writeback_debt::settle_for_mapping(
            state,
            host,
            mapping,
            crate::runtime::render_writeback::SettleSite::MappingRgba8Write,
        );
        let m = state
            .mappings
            .get(&mapping)
            .filter(|m| m.mapped && !m.page_entries.is_empty())
            .ok_or_else(|| Status::args("metal_gpu_store_mapping"))?;
        let (mw, mh, format) = mapping_write_geometry(m, width, height);
        if (mw, mh) != (width, height) {
            return Err(Status::args("metal_gpu_store_geometry_moved").into());
        }
        let output = guest_writeback::Output::for_format(format)
            .ok_or_else(|| Status::args("metal_gpu_store_format").field("format", format))?;
        let (base, pitch, span_end) = mapper_ref_texture_sample_window(m, width, height, format)
            .ok_or_else(|| Status::args("metal_gpu_store_window"))?;
        let layout = guest_writeback::Layout {
            width,
            height,
            pitch,
            output,
        };
        let span = layout
            .span()
            .and_then(|span| {
                base.checked_add(span)
                    .filter(|end| *end <= span_end)
                    .map(|_| span)
            })
            .ok_or_else(|| Status::args("metal_gpu_store_span"))?;
        let (import, footprint) =
            mapper::ensure_owned_contig_import_with_footprint(state, host, mapping)
                .ok_or_else(|| Status::args("metal_gpu_store_import_unavailable"))?;
        let vouched = vouch_for_write(state, host, mapping, "metal_gpu_store")
            .ok_or_else(|| Status::args("metal_gpu_store_pages_not_ours"))?;
        let slice = import
            .slice(base, span)
            .map_err(|_| Status::args("metal_gpu_store_slice"))?;
        let device = crate::backend::metal::runtime::system_device()
            .ok_or_else(|| Status::execute("metal_gpu_store_device"))?;
        let import_id = import.id();
        let prepared = guest_writeback::Prepared::new(device, source, import, slice, layout)?;
        let store = Self {
            mapping,
            width,
            height,
            format,
            base,
            pitch,
            span_end,
            footprint,
            import: import_id,
            vouched,
            source: source.to_owned(),
            prepared: Some(prepared),
        };
        if !store.current(state) {
            return Err(Status::args("metal_gpu_store_capture_stale").into());
        }
        Ok(store)
    }

    fn current(&self, state: &DeviceState) -> bool {
        if !self.vouched.covers(state, self.mapping) {
            return false;
        }
        state.mappings.get(&self.mapping).is_some_and(|m| {
            m.mapped
                && m.contig_import
                    .as_ref()
                    .is_some_and(|import| import.id() == self.import && !import.is_retired())
                && m.contig_footprint
                    .as_ref()
                    .is_some_and(|p| p.same_allocation(&self.footprint))
                && mapping_write_geometry(m, self.width, self.height)
                    == (self.width, self.height, self.format)
                && mapper_ref_texture_sample_window(m, self.width, self.height, self.format)
                    == Some((self.base, self.pitch, self.span_end))
        })
    }

    pub(crate) fn usable<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> bool {
        mapper::revalidate_mapping_pages(state, host, self.mapping)
            && vouch_for_write(state, host, self.mapping, "metal_gpu_store_redeem").is_some()
            && self.current(state)
    }

    pub(crate) fn check<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<(), Box<Status>> {
        if self.usable(state, host) {
            Ok(())
        } else {
            Err(Status::args("metal_gpu_store_capture_stale").into())
        }
    }

    pub(crate) fn load_source<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<guest_writeback::ReadSource, Box<Status>> {
        self.check(state, host)?;
        let import = state
            .mappings
            .get(&self.mapping)
            .and_then(|mapping| mapping.contig_import.as_ref())
            .filter(|import| import.id() == self.import)
            .cloned()
            .ok_or_else(|| Status::args("metal_gpu_load_import_changed"))?;
        let slice = import
            .slice(self.base, self.span_end - self.base)
            .map_err(|_| Status::args("metal_gpu_load_slice"))?;
        let device = crate::backend::metal::runtime::system_device()
            .ok_or_else(|| Status::execute("metal_gpu_load_device"))?;
        Ok(guest_writeback::ReadSource::new(
            device,
            import,
            slice,
            guest_writeback::ReadLayout {
                width: self.width,
                height: self.height,
                pitch: self.pitch,
                format: self.format,
            },
        )?)
    }

    /// A revoked allocation is not permission to fall back onto replacement
    /// pages. Neither that failure nor a failed command publishes a frame.
    /// The same-queue writeback completion also orders the render producer.
    /// Check that producer's status before publishing any successful frame.
    pub(crate) fn finish_after<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        complete_render: impl FnOnce() -> Result<(), Status>,
    ) -> Result<(), Box<Status>> {
        self.check(state, host)?;
        let prepared = self
            .prepared
            .take()
            .ok_or_else(|| Status::args("metal_gpu_store_command_consumed"))?;
        crate::runtime::surface_cache::forget(state, self.mapping);
        let _span = crate::runtime::chain_phase::CostSpan::new("metal_gpu_store_us");
        let result = prepared.execute();
        // Failed GPU work can have written a prefix. Record its admitted pages,
        // never a page list reconstructed after completion.
        mapper::note_physical_page_write_footprint(
            &self.footprint,
            self.base,
            self.span_end - self.base,
        );
        state.host_writes.note_footprint(&self.footprint);
        result?;
        complete_render()?;
        if !self.usable(state, host) {
            return Err(Status::args("metal_gpu_store_completion_stale").into());
        }
        publish_rgba8_written(
            state,
            host,
            self.mapping,
            self.width,
            self.height,
            self.base..self.span_end,
            CompletedRgba8::Resident,
        );
        let key = resident::ResidentColorKey::for_surface(self.mapping, self.width, self.height);
        resident::retain_completed(key, &self.source);
        if let Some(generation) = crate::runtime::surface_cache::frame_generation(
            state,
            self.mapping,
            self.width,
            self.height,
        ) {
            resident::published(&key, generation);
        }
        Ok(())
    }
}
