//! Completion snapshots for the presents that existed when an object retired.
//! Newer presents cannot extend that object's lifetime. Each captured claim
//! completes only when its own frame fence retires (or the device quiesces).

use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

#[derive(Debug)]
pub(super) struct WindowWork(Arc<AtomicBool>);

impl WindowWork {
    pub(super) fn complete(&self) {
        self.0.store(true, Ordering::Release);
    }
}

#[derive(Default)]
pub(crate) struct WindowRetirement(Vec<Arc<AtomicBool>>);

impl WindowRetirement {
    pub(crate) fn is_complete(&self) -> bool {
        self.0.iter().all(|work| work.load(Ordering::Acquire))
    }
}

struct Registry {
    pending: Vec<Arc<AtomicBool>>,
}

impl Registry {
    fn prune(&mut self) {
        self.pending.retain(|work| !work.load(Ordering::Acquire));
    }

    fn begin(&mut self) -> WindowWork {
        self.prune();
        let work = Arc::new(AtomicBool::new(false));
        self.pending.push(Arc::clone(&work));
        WindowWork(work)
    }

    fn snapshot(&mut self) -> WindowRetirement {
        self.prune();
        WindowRetirement(self.pending.clone())
    }
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { pending: Vec::new() });

pub(super) fn begin() -> WindowWork {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner()).begin()
}

pub(super) fn snapshot() -> WindowRetirement {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner()).snapshot()
}

pub(super) fn in_flight() -> bool {
    let mut registry = REGISTRY.lock().unwrap_or_else(|error| error.into_inner());
    registry.prune();
    !registry.pending.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_presents_cannot_extend_a_retirement_snapshot() {
        let mut registry = Registry { pending: Vec::new() };
        let old = registry.begin();
        let retired = registry.snapshot();
        let newer = registry.begin();
        assert!(!retired.is_complete());
        old.complete();
        assert!(retired.is_complete());
        assert!(!registry.snapshot().is_complete(), "the newer present is still running");
        newer.complete();
        assert!(registry.snapshot().is_complete());
    }

    #[test]
    fn every_captured_present_must_complete_even_out_of_order() {
        let mut registry = Registry { pending: Vec::new() };
        let first = registry.begin();
        let second = registry.begin();
        let retired = registry.snapshot();
        second.complete();
        assert!(!retired.is_complete());
        first.complete();
        assert!(retired.is_complete());
        assert!(registry.snapshot().0.is_empty());
    }

    #[test]
    fn an_empty_snapshot_never_acquires_future_dependencies() {
        let mut registry = Registry { pending: Vec::new() };
        let empty = registry.snapshot();
        let later = registry.begin();
        assert!(empty.is_complete());
        assert!(!registry.snapshot().is_complete());
        later.complete();
    }
}
