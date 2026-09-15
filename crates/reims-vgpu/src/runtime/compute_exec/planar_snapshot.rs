//! One immutable planar snapshot per serialized texture resource.
//!
//! Both fragment and compute staging enter here after the common layout decoder
//! and writeback settlement. A hit requires exact source identity and the shared
//! guest/host gather witness, including its audit of the *guest* allocation.
//! Delayed observations confine read elision to one live decoded render pass.
//! Only the optional current-write observer can establish freshness between
//! passes; a construction descriptor or harvested generation cannot. A miss
//! brackets the ordinary plane fill: a changed or
//! unreadable generation may serve this copy, but can never retain it for reuse.
//! Replacement/deletion drops only our Arc; encoded commands retain their own.

use super::{ComputeStatus, DeviceState, HostMemory, HostOps, PlanarSource};
use crate::model::RailResourceState;
use crate::protocol::planar::{Layout, TextureDescription};
use crate::runtime::draw::SnapshotScopeRef;
use crate::runtime::gather_witness::{
    note_scoped_gather, GatherKey, GatherOutcome, GatherRail, GatherWindow, GatheredIdentity,
};
use crate::runtime::guest_ram::GuestRun;
use std::sync::Arc;

#[cfg(test)]
pub(crate) mod tests;

#[derive(Debug)]
pub(super) struct RetainedPlanar<I> {
    latest: Option<Entry<I>>,
}

impl<I> Default for RetainedPlanar<I> {
    fn default() -> Self {
        Self { latest: None }
    }
}

impl<I: Send + Sync + 'static> RailResourceState for RetainedPlanar<I> {}

#[cfg(test)]
impl<I> RetainedPlanar<I> {
    pub(super) fn is_empty(&self) -> bool {
        self.latest.is_none()
    }
}

pub(super) struct Counters {
    pub unscoped: &'static str,
    pub reuses: &'static str,
    pub reuse_bytes: &'static str,
    pub misses: &'static str,
    pub unretained: &'static str,
}

#[derive(Debug)]
struct Entry<I> {
    source: SourceKey,
    identity: GatheredIdentity,
    image: Arc<I>,
}

/// No hash-only equality: every byte-layout and physical-identity term survives.
#[derive(Debug, PartialEq, Eq)]
struct SourceKey {
    window: GatherKey,
    description: TextureDescription,
    layout: Layout,
    map_generation: u32,
    content_generation: u32,
    page_size: usize,
    pages: Arc<[u64]>,
}

struct Window {
    key: SourceKey,
    run: GuestRun,
}

impl Window {
    fn resolve<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        source: &PlanarSource,
    ) -> Result<Option<Self>, ComputeStatus> {
        use crate::backend::{Backend as _, GuestWriteReach};
        let mid = source.description.mapping_id;
        let Some((ptr, len, pages)) =
            crate::runtime::mapper::ensure_contig_view_with_pages(state, host, mid)
        else {
            // A packed alias is optional; the ordinary mapping reader also has
            // a scatter-copy path. No alias means no audit and thus no reuse.
            return Ok(None);
        };
        let mapping = state
            .mappings
            .get(&mid)
            .ok_or(ComputeStatus::MissingTexture("planar_mapping_missing"))?;
        if !mapping.mapped
            || mapping.map_generation != source.map_generation
            || Layout::decode(
                &mapping.device_desc,
                source.description.width,
                source.description.height,
            )
            .as_ref()
                != Ok(&source.layout)
        {
            return Err(ComputeStatus::GuestIo("planar_mapping_changed"));
        }
        let backend = crate::backend::selected();
        if backend.guest_writes_outstanding()
            && !matches!(
                backend.guest_writes_reaching(&pages),
                GuestWriteReach::Disjoint
            )
        {
            // note_gather's pending-write arm suppresses its CPU audit, not
            // every generation vouch. This CPU snapshot requires settlement.
            return Ok(None);
        }
        let Some(run) = GuestRun::in_mapping(ptr, len as u64, 0, source.layout.allocation_size)
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            key: SourceKey {
                window: GatherKey::Mapping { mid, base_off: 0 },
                description: source.description,
                layout: source.layout.clone(),
                map_generation: mapping.map_generation,
                content_generation: mapping.content_generation,
                page_size: state.page_size() as usize,
                pages,
            },
            run,
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
            self.key.window,
            GatherWindow {
                gpas: &self.key.pages,
                runs: std::slice::from_ref(&self.run),
                span: self.key.layout.allocation_size,
                page_size: self.key.page_size,
            },
            scope,
        )
    }
}

pub(super) fn stage_with<I: Send + Sync + 'static, M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    source: &PlanarSource,
    counters: &Counters,
    fill: impl FnOnce(&mut DeviceState, &mut M, &PlanarSource) -> Result<Arc<I>, ComputeStatus>,
) -> Result<Arc<I>, ComputeStatus> {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    let scope = source.scope.as_ref();
    let scope_id = scope.and_then(SnapshotScopeRef::current);
    if scope_id.is_none() {
        note_store_route(counters.unscoped);
        source
            .resource
            .with_rail_state(|held: &mut RetainedPlanar<I>| held.latest = None);
        note_store_route(counters.misses);
        let image = fill(state, host, source)?;
        note_store_route(counters.unretained);
        return Ok(image);
    }
    let before = Window::resolve(state, host, source)?;
    // Do not keep a pointer-bearing window across a fill, which can revalidate
    // and retire an alias. Only the exact key and witness identity survive it.
    let before = before.map(|window| {
        let outcome = window.observe(state, host, scope);
        (window.key, outcome)
    });
    let hit = source
        .resource
        .with_rail_state(|held: &mut RetainedPlanar<I>| {
            if let (Some((key, outcome)), Some(entry)) = (&before, &held.latest) {
                if outcome.cpu_read_vouched()
                    && scope_id.is_some()
                    && scope.and_then(SnapshotScopeRef::current) == scope_id
                    && entry.source == *key
                    && entry.identity == outcome.identity
                {
                    return Some(Arc::clone(&entry.image));
                }
            }
            // An error during the replacement fill must not leave an old candidate.
            held.latest = None;
            None
        })
        .flatten();
    if let Some(image) = hit {
        note_store_route(counters.reuses);
        note_store_route_n(
            counters.reuse_bytes,
            source.layout.planes.iter().map(|plane| plane.size).sum(),
        );
        return Ok(image);
    }
    note_store_route(counters.misses);
    let image = fill(state, host, source)?;
    if let Some((key, before)) = before {
        if let Some(after) = Window::resolve(state, host, source)? {
            let outcome = after.observe(state, host, scope);
            if after.key == key
                && outcome.cpu_read_vouched()
                && outcome.identity == before.identity
                && before.cpu_read_settled
                && scope_id.is_some()
                && scope.and_then(SnapshotScopeRef::current) == scope_id
            {
                let retained = source
                    .resource
                    .with_rail_state(|held: &mut RetainedPlanar<I>| {
                        held.latest = Some(Entry {
                            source: key,
                            identity: outcome.identity,
                            image: Arc::clone(&image),
                        });
                    });
                if retained.is_some() {
                    return Ok(image);
                }
            }
        }
    }
    note_store_route(counters.unretained);
    Ok(image)
}
