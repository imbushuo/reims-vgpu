//! Orthogonal rejection counters, not an alternative eligibility predicate.
//! A displaced entry holds only already-produced metadata: observing it again
//! cannot lend its old native allocation, which may already have been recycled.
//! Its one-entry bound measures A/B thrashing, not every possible working set.

use super::*;
use crate::runtime::drain::{note_store_route, note_store_route_n};

#[derive(Clone, Copy)]
pub(super) struct Origin {
    pub class: Class,
    pub slot: u32,
}

pub(super) struct Displaced {
    pub key: Key,
    pub identity: GatheredIdentity,
}

fn count_if(condition: bool, name: &'static str) {
    if condition {
        note_store_route(name);
    }
}

pub(super) fn lookup(
    held: &mut RetainedBuffer,
    seen: &Result<(Key, GatherOutcome), MemError>,
    requested: BufferReadSpan,
    current_scope: Option<u64>,
    origin: Origin,
) {
    if held
        .displaced
        .as_ref()
        .is_some_and(|old| Some(old.key.scope) != current_scope)
    {
        held.displaced = None;
    }
    let Ok((key, seen)) = seen else {
        note_store_route("metal_input_snapshot_lookup_observe_failed");
        return;
    };
    count_if(
        !seen.vouch.is_vouched(),
        "metal_input_snapshot_lookup_unvouched",
    );
    count_if(
        !seen.cpu_read_settled,
        "metal_input_snapshot_lookup_unsettled",
    );
    if let Some(old) = &held.displaced {
        if old.key == *key && old.key.span == requested {
            note_store_route("metal_input_snapshot_displaced_exact");
            note_store_route_n(
                "metal_input_snapshot_displaced_exact_bytes",
                key.span.len as u64,
            );
            if seen.cpu_read_vouched()
                && seen.identity == old.identity
                && current_scope == Some(key.scope)
            {
                note_store_route("metal_input_snapshot_displaced_vouched");
                note_store_route_n(
                    "metal_input_snapshot_displaced_vouched_bytes",
                    key.span.len as u64,
                );
            }
        }
    }
    let Some(old) = &held.latest else {
        note_store_route("metal_input_snapshot_miss_no_candidate");
        return;
    };
    count_if(
        origin.class as usize != old.origin.class as usize,
        "metal_input_snapshot_candidate_stage_changed",
    );
    count_if(
        origin.slot != old.origin.slot,
        "metal_input_snapshot_candidate_slot_changed",
    );
    count_if(
        key.scope != old.key.scope,
        "metal_input_snapshot_miss_scope",
    );
    count_if(
        current_scope != Some(key.scope),
        "metal_input_snapshot_miss_scope_expired",
    );
    count_if(
        key.declared_write != old.key.declared_write,
        "metal_input_snapshot_miss_declared_write",
    );
    let same_source = requested.task == old.key.span.task
        && requested.backing == old.key.span.backing
        && requested.page_shift == old.key.span.page_shift;
    count_if(!same_source, "metal_input_snapshot_miss_source");
    if requested.offset != old.key.span.offset {
        note_store_route("metal_input_snapshot_miss_offset");
        note_store_route_n(
            "metal_input_snapshot_miss_offset_bytes",
            requested.len as u64,
        );
    }
    if requested.len != old.key.span.len {
        note_store_route("metal_input_snapshot_miss_capture_length");
        note_store_route_n(
            "metal_input_snapshot_miss_capture_length_bytes",
            requested.len as u64,
        );
    }
    count_if(
        key.pages != old.key.pages,
        "metal_input_snapshot_miss_pages",
    );
    count_if(
        seen.identity != old.identity,
        "metal_input_snapshot_miss_identity",
    );
    if *key == old.key && requested == old.key.span {
        note_store_route("metal_input_snapshot_metadata_exact");
        note_store_route_n(
            "metal_input_snapshot_metadata_exact_bytes",
            requested.len as u64,
        );
        count_if(
            !seen.vouch.is_vouched(),
            "metal_input_snapshot_metadata_exact_unvouched",
        );
        count_if(
            seen.identity != old.identity,
            "metal_input_snapshot_metadata_exact_identity_changed",
        );
        count_if(
            !seen.cpu_read_settled,
            "metal_input_snapshot_metadata_exact_unsettled",
        );
    }
    // Geometry only, NOT permission to rebind: different native offsets,
    // captured coverage and initialized-zero semantics still need a proof.
    if same_source
        && key.scope == old.key.scope
        && requested.offset >= old.key.span.offset
        && requested
            .offset
            .checked_add(requested.len as u64)
            .zip(old.key.span.offset.checked_add(old.key.span.len as u64))
            .is_some_and(|(end, old_end)| end <= old_end)
    {
        note_store_route("metal_input_snapshot_geometry_contained");
        note_store_route_n(
            "metal_input_snapshot_geometry_contained_bytes",
            requested.len as u64,
        );
    }
}

pub(super) fn retention(
    before: &Key,
    first: GatherOutcome,
    after: &Key,
    last: GatherOutcome,
    current_scope: Option<u64>,
) {
    count_if(before != after, "metal_input_snapshot_post_key_changed");
    count_if(
        before.pages != after.pages,
        "metal_input_snapshot_post_pages_changed",
    );
    count_if(
        before.declared_write != after.declared_write,
        "metal_input_snapshot_post_declared_write_changed",
    );
    count_if(
        current_scope != Some(before.scope),
        "metal_input_snapshot_post_scope_expired",
    );
    count_if(
        !first.cpu_read_settled || !last.cpu_read_settled,
        "metal_input_snapshot_post_unsettled",
    );
    count_if(
        !last.vouch.is_vouched(),
        "metal_input_snapshot_post_unvouched",
    );
    count_if(
        first.identity != last.identity,
        "metal_input_snapshot_post_identity_changed",
    );
}
