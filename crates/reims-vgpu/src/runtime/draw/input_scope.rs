//! Read elision belongs to an executing command, not a resource's lifetime.
//! The decoded render-pass owner holds the strong scope until its work returns;
//! requests and resource caches hold only revocable references. This is not a
//! dirty-harvest epoch or a claim that a resource stays immutable after completion.

use std::sync::{
    atomic::{AtomicU64, Ordering::Relaxed},
    Arc, Weak,
};

#[derive(Debug)]
struct Identity(u64);

pub(crate) struct BufferSnapshotScope(Arc<Identity>);

/// Capability carried by a request within one live decoded render pass.
#[derive(Clone, Debug)]
pub struct SnapshotScopeRef(Weak<Identity>);

impl BufferSnapshotScope {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT
            .fetch_update(Relaxed, Relaxed, |id| id.checked_add(1))
            .expect("buffer snapshot scope identity exhausted");
        Self(Arc::new(Identity(id)))
    }

    pub(crate) fn reference(&self) -> SnapshotScopeRef {
        SnapshotScopeRef(Arc::downgrade(&self.0))
    }
}

impl SnapshotScopeRef {
    pub(crate) fn current(&self) -> Option<u64> {
        self.0.upgrade().map(|identity| identity.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_scope_references_expire_and_never_name_a_later_command() {
        let scope = BufferSnapshotScope::new();
        let first = scope.reference();
        let id = first.current().unwrap();
        assert_eq!(first.clone().current(), Some(id));
        drop(scope);
        assert_eq!(first.current(), None);
        let next = BufferSnapshotScope::new();
        assert_ne!(next.reference().current(), Some(id));
        assert_eq!(first.current(), None);
    }
}
