//! One private immutable native snapshot per buffer TaskResource. The source
//! is freshly walked on every lookup and on both sides of a miss's capture.
//! GatherWitness owns guest/host write tracking and the existing content audit;
//! no guest alias, copied descriptor, or native pointer decides freshness.
//! Harvest observations are not a cross-command CPU mutation fence. Reuse is
//! additionally confined to one live decoded render-pass scope, and the guest's
//! explicit per-buffer write declaration also invalidates a retained capture.
//! Contained loans keep the original full capture witness. A wider metadata-only
//! plan may survive tracking startup, but only an actual full planned capture
//! can create the native candidate; narrower fresh fills never stand for it.

use super::*;
use crate::backend::metal::buffer_extent::ReadOnlyCapture;
use crate::backend::metal::input::{Filled, ReadOnlySnapshot};
use crate::runtime::decode::resource::OBJECT_TYPE_BUFFER;
use crate::runtime::draw::BufferReadSpan;
use crate::runtime::gather_witness::{self, GatherKey, GatherOutcome, GatheredIdentity};
use crate::runtime::host::MemError;
use std::sync::Arc;

mod diagnostics;

#[derive(Debug, PartialEq, Eq)]
struct Key {
    scope: u64,
    declared_write: crate::runtime::buffer_write_gen::BufferWriteStamp,
    span: BufferReadSpan,
    pages: Vec<u64>,
}

struct Candidate {
    key: Key,
    identity: GatheredIdentity,
    // Original capture and physical zero/dirty coverage remain immutable.
    // A contained binding may advance its offset under a new readonly proof.
    image: Arc<ReadOnlySnapshot>,
    origin: diagnostics::Origin,
}

/// Buffer construction descriptors are disjoint from packed/planar textures.
/// A conflicting rail slot falls back to uncached capture; it is never replaced.
#[derive(Default)]
struct RetainedBuffer {
    latest: Option<Candidate>,
    /// Metadata only: the most recently displaced candidate, never reusable
    /// storage. This measures two-key thrashing without enlarging the cache.
    displaced: Option<diagnostics::Displaced>,
    /// Geometry only, from a checked observation in this scope. Keeping a
    /// covering witness stable lets its token mature without retaining bytes.
    plan: Option<WitnessPlan>,
}

#[derive(Clone, Copy)]
struct WitnessPlan {
    scope: u64,
    span: BufferReadSpan,
}

fn contained_offset(outer: BufferReadSpan, inner: BufferReadSpan) -> Option<usize> {
    if outer.task != inner.task
        || outer.backing != inner.backing
        || outer.page_shift != inner.page_shift
    {
        return None;
    }
    let delta = inner.offset.checked_sub(outer.offset)?;
    if outer.gva.checked_add(delta)? != inner.gva
        || delta.checked_add(inner.len as u64)? > outer.len as u64
    {
        return None;
    }
    usize::try_from(delta).ok()
}

impl RetainedBuffer {
    fn witness_span(&self, scope: u64, requested: BufferReadSpan) -> BufferReadSpan {
        self.latest
            .as_ref()
            .filter(|candidate| {
                candidate.key.scope == scope
                    && contained_offset(candidate.key.span, requested).is_some()
            })
            .map(|candidate| candidate.key.span)
            .or_else(|| {
                self.plan
                    .filter(|plan| {
                        plan.scope == scope && contained_offset(plan.span, requested).is_some()
                    })
                    .map(|plan| plan.span)
            })
            .unwrap_or(requested)
    }
}

impl crate::model::RailResourceState for RetainedBuffer {}

fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (setting, value) = crate::config::read(crate::config::METAL_INPUT_SNAPSHOT_REUSE);
        permitted(setting, value.as_deref())
    })
}

fn permitted(setting: crate::config::Switch, value: Option<&str>) -> bool {
    match setting {
        crate::config::Switch::Off => false,
        crate::config::Switch::Unset | crate::config::Switch::On => true,
        crate::config::Switch::Unrecognized => {
            crate::observe::Emit::refusal(
                "metal_input_snapshot",
                &crate::backend::metal::util::Status::args(
                    "metal_input_snapshot_switch_unrecognized",
                ),
            )
            .unwrap()
            .field("name", crate::config::METAL_INPUT_SNAPSHOT_REUSE)
            .field("value", value.unwrap_or_default())
            .fail();
            false
        }
    }
}

fn observe<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    span: BufferReadSpan,
    scope: u64,
    buffer_ref: u32,
) -> Result<(Key, GatherOutcome), MemError> {
    let page_size = 1u64
        .checked_shl(span.page_shift)
        .ok_or(MemError::UnsupportedPageShift)?;
    let count = (span.gva % page_size + span.len as u64).div_ceil(page_size);
    let mut pages = Vec::new();
    pages
        .try_reserve_exact(count as usize)
        .map_err(|_| MemError::Overflow)?;
    let mut complete = true;
    gva_mem::visit_task_gva_pages_in_order(
        host,
        &state.tasks,
        span.task,
        span.gva,
        span.len as u64,
        span.page_shift,
        &mut |page| {
            match page {
                Some(page) => pages.push(page),
                None => complete = false,
            }
            complete
        },
    );
    if !complete || pages.len() as u64 != count {
        return Err(MemError::Unmapped);
    }
    let task = state
        .tasks
        .get(span.task)
        .ok_or(MemError::NoSuchTask)?
        .clone();
    let seen = gather_witness::note_cpu_read(
        state,
        host,
        GatherKey::TaskBuffer {
            task_id: span.task,
            gva: span.gva,
        },
        &pages,
        span.len as u64,
        page_size as usize,
        |host| {
            let mut bytes = vec![0; span.len];
            gva_mem::read_task_gva(host, &task, span.gva, &mut bytes, span.page_shift)?;
            Ok(bytes)
        },
    )?;
    Ok((
        Key {
            scope,
            declared_write: state.buffer_write_gen.stamp(span.task, buffer_ref),
            span,
            pages,
        },
        seen,
    ))
}

pub(super) struct Request<'a> {
    pub device: &'a ::metal::Device,
    pub bind: &'a BufferBind,
    pub span: BufferReadSpan,
    pub class: Class,
    pub proof: Option<ReadOnlyCapture>,
    pub scope: Option<SnapshotScopeRef>,
}

impl Request<'_> {
    pub(super) fn capture<M: HostMemory + HostOps>(
        self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<Filled, FillError<MemError>> {
        let span = self.span;
        self.fill_with(state, host, |state, host, bytes| {
            span.read_into(state, host, bytes).map(|()| bytes.len())
        })
    }

    fn fill_with<M: HostMemory + HostOps>(
        self,
        state: &mut DeviceState,
        host: &mut M,
        fill: impl FnOnce(&mut DeviceState, &mut M, &mut [u8]) -> Result<usize, MemError>,
    ) -> Result<Filled, FillError<MemError>> {
        let Self {
            device,
            bind,
            span,
            class,
            proof,
            scope,
        } = self;
        use crate::runtime::drain::{note_store_route, note_store_route_n};
        let scope_id = scope.as_ref().and_then(SnapshotScopeRef::current);
        if scope_id.is_none() {
            note_store_route("metal_input_snapshot_unscoped");
        }
        let resource = proof
            .filter(|_| enabled() && scope_id.is_some() && host.guest_write_gen_is_current())
            .and_then(|_| {
                bind.resource.clone().or_else(|| {
                    objects::resolve_resource(state, host, span.task, bind.buffer_ref).ok()
                })
            })
            .filter(|resource| resource.entry.object_type == OBJECT_TYPE_BUFFER);
        let before = if let Some(resource) = &resource {
            // A miss removes the old candidate before filling, including failures.
            // No guest read or settlement runs while this resource lock is held.
            let planned = resource
                .with_rail_state(|held: &mut RetainedBuffer| {
                    held.witness_span(scope_id.unwrap(), span)
                })
                .unwrap_or(span);
            // The ordinary caller settled only the requested range. Auditing
            // or vouching for a wider original capture owes the same settlement
            // over that whole range, including named debt and pending writes.
            if planned != span {
                planned.settle(state, host, bind.buffer_ref);
            }
            let mut seen = observe(state, host, planned, scope_id.unwrap(), bind.buffer_ref);
            if planned != span {
                note_store_route("metal_input_snapshot_witness_plan_reused");
                if seen.is_err() {
                    // An unprovable covering range cannot block a valid narrow
                    // read, nor remain the next request's witness plan.
                    note_store_route("metal_input_snapshot_witness_plan_refused");
                    resource.with_rail_state(|held: &mut RetainedBuffer| held.plan = None);
                    seen = observe(state, host, span, scope_id.unwrap(), bind.buffer_ref);
                }
            }
            let held = resource.with_rail_state(|held: &mut RetainedBuffer| {
                if let Ok((key, _)) = &seen {
                    if scope.as_ref().and_then(SnapshotScopeRef::current) == Some(key.scope) {
                        held.plan = Some(WitnessPlan {
                            scope: key.scope,
                            span: key.span,
                        });
                    }
                }
                if let (Ok((key, seen)), Some(candidate), Some(proof)) =
                    (&seen, &held.latest, proof)
                {
                    if key.scope != candidate.key.scope {
                        note_store_route("metal_input_snapshot_scope_changed");
                    }
                    if key.declared_write != candidate.key.declared_write {
                        note_store_route("metal_input_snapshot_declared_write_changed");
                    }
                    if seen.cpu_read_vouched()
                        && seen.identity == candidate.identity
                        && *key == candidate.key
                        && scope.as_ref().and_then(SnapshotScopeRef::current) == Some(key.scope)
                    {
                        if let Some(delta) = contained_offset(candidate.key.span, span) {
                            if let Some(input) = candidate.image.bind_window(device, proof, delta) {
                                if candidate.key.span != span {
                                    note_store_route("metal_input_snapshot_contained_hits");
                                }
                                return (Some(input), None);
                            }
                            note_store_route("metal_input_snapshot_miss_native_binding");
                        } else {
                            note_store_route("metal_input_snapshot_miss_capture_coverage");
                        }
                    }
                }
                diagnostics::lookup(
                    held,
                    &seen,
                    span,
                    scope.as_ref().and_then(SnapshotScopeRef::current),
                    diagnostics::Origin {
                        class,
                        slot: bind.index,
                    },
                );
                let retired = held.latest.take().map(|candidate| {
                    held.displaced = Some(diagnostics::Displaced {
                        key: candidate.key,
                        identity: candidate.identity,
                    });
                    candidate.image
                });
                (None, retired)
            });
            let slot_owned = held.is_some();
            let (hit, retired) = held.unwrap_or_default();
            if !slot_owned {
                crate::observe::Emit::refusal(
                    "metal_input_snapshot",
                    &crate::backend::metal::util::Status::args(
                        "metal_input_snapshot_resource_state_conflict",
                    ),
                )
                .unwrap()
                .fail();
            }
            if let Some(retired) = retired {
                retired.retire(device);
            }
            if let Some(hit) = hit {
                let (_, seen) = seen.expect("a hit has a witnessed window");
                note_store_route("metal_input_snapshot_hits");
                note_store_route_n(
                    "metal_input_snapshot_capture_bytes_avoided",
                    span.len as u64,
                );
                note_store_route_n(
                    "metal_input_snapshot_guest_bytes_avoided",
                    (span.len as u64).saturating_sub(seen.audit_bytes),
                );
                note_store_route_n("metal_input_snapshot_hit_audit_bytes", seen.audit_bytes);
                note_store_route_n(
                    "metal_input_snapshot_extra_audit_bytes",
                    seen.audit_bytes.saturating_sub(span.len as u64),
                );
                return Ok(hit);
            }
            match seen {
                Ok(before) => {
                    note_store_route_n(
                        "metal_input_snapshot_extra_audit_bytes",
                        before.1.audit_bytes,
                    );
                    slot_owned.then_some(before)
                }
                Err(reason) => {
                    // Ordinary capture remains the authoritative checked reader.
                    // Optional audit/page failure cannot preserve a cache candidate.
                    crate::observe::Emit::decline("metal_input_snapshot", &reason).fail();
                    None
                }
            }
        } else {
            note_store_route("metal_input_snapshot_ineligible");
            None
        };
        note_store_route("metal_input_snapshot_captures");
        let callback = |bytes: &mut [u8]| fill(state, host, bytes);
        let input = match proof {
            Some(proof) => input::fill_read_only_resource_prefix(
                device,
                span.backing.size,
                span.offset,
                proof,
                class,
                "metal_render_buffer_create_failed",
                callback,
            ),
            None => input::fill_resource_prefix(
                device,
                span.backing.size,
                span.offset,
                span.len,
                class,
                "metal_render_buffer_create_failed",
                callback,
            ),
        }?;
        if let (Some(resource), Some((key, seen)), Some(proof)) = (resource, before, proof) {
            match observe(state, host, key.span, key.scope, bind.buffer_ref) {
                Ok((after, final_seen)) => {
                    diagnostics::retention(
                        &key,
                        seen,
                        &after,
                        final_seen,
                        scope.as_ref().and_then(SnapshotScopeRef::current),
                    );
                    note_store_route_n(
                        "metal_input_snapshot_extra_audit_bytes",
                        final_seen.audit_bytes,
                    );
                    if key == after
                        && scope.as_ref().and_then(SnapshotScopeRef::current) == Some(key.scope)
                        && seen.cpu_read_settled
                        && final_seen.cpu_read_vouched()
                        && seen.identity == final_seen.identity
                        && key.span == span
                    {
                        let image = input
                            .freeze()
                            .expect("only a certified capture is retained");
                        let input = image
                            .bind(device, proof)
                            .expect("capture queue and shape unchanged");
                        resource.with_rail_state(|held: &mut RetainedBuffer| {
                            held.latest = Some(Candidate {
                                key,
                                identity: final_seen.identity,
                                image,
                                origin: diagnostics::Origin {
                                    class,
                                    slot: bind.index,
                                },
                            });
                        });
                        note_store_route("metal_input_snapshot_retained");
                        return Ok(input);
                    }
                    if key.span != span {
                        note_store_route("metal_input_snapshot_metadata_plan_only");
                        if !final_seen.cpu_read_vouched() {
                            note_store_route("metal_input_snapshot_metadata_plan_unvouched");
                        }
                    } else {
                        note_store_route("metal_input_snapshot_changed_during_capture");
                    }
                }
                Err(reason) => {
                    note_store_route("metal_input_snapshot_post_observe_failed");
                    crate::observe::Emit::decline("metal_input_snapshot_post", &reason).fail();
                }
            }
        }
        Ok(input)
    }
}

#[cfg(test)]
mod tests;
