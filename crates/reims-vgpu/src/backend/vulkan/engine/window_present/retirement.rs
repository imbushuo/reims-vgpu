//! Completion snapshots for the presents that existed when an object retired.
//! Newer presents cannot extend that object's lifetime. Each captured claim
//! completes only when its own frame fence retires (or the device quiesces).

use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use ash::vk;

#[derive(Debug)]
struct Completion {
    fence: vk::Fence,
    done: AtomicBool,
}

#[derive(Debug)]
pub(super) struct WindowWork(Arc<Completion>);

impl WindowWork {
    pub(super) fn complete(&self) {
        self.0.done.store(true, Ordering::Release);
    }
}

#[derive(Default)]
pub(crate) struct WindowRetirement(Vec<Arc<Completion>>);

impl WindowRetirement {
    pub(crate) fn is_complete(&self) -> bool {
        self.0.iter().all(|work| work.done.load(Ordering::Acquire))
    }
}

struct Registry {
    pending: Vec<Arc<Completion>>,
}

impl Registry {
    fn prune(&mut self) {
        self.pending.retain(|work| !work.done.load(Ordering::Acquire));
    }

    fn begin(&mut self, fence: vk::Fence) -> WindowWork {
        self.prune();
        let work = Arc::new(Completion { fence, done: AtomicBool::new(false) });
        self.pending.push(Arc::clone(&work));
        WindowWork(work)
    }

    fn snapshot(&mut self) -> WindowRetirement {
        self.prune();
        WindowRetirement(self.pending.clone())
    }

    fn poll<E>(&mut self, mut ready: impl FnMut(vk::Fence) -> Result<bool, E>) -> Result<usize, E> {
        self.prune();
        let mut retired = 0;
        for work in &self.pending {
            if ready(work.fence)? {
                work.done.store(true, Ordering::Release);
                retired += 1;
            }
        }
        self.prune();
        Ok(retired)
    }
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { pending: Vec::new() });

pub(super) fn begin(fence: vk::Fence) -> WindowWork {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner()).begin(fence)
}

pub(super) fn snapshot() -> WindowRetirement {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner()).snapshot()
}

pub(super) fn in_flight() -> bool {
    let mut registry = REGISTRY.lock().unwrap_or_else(|error| error.into_inner());
    registry.prune();
    !registry.pending.is_empty()
}

/// Caller holds ENGINE, which also serializes window fence reset/destruction.
/// Polling here avoids depending on another redraw to discover GPU completion.
pub(super) unsafe fn poll(device: &ash::Device) -> Result<usize, vk::Result> {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner())
        .poll(|fence| unsafe { device.get_fence_status(fence) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_presents_cannot_extend_a_retirement_snapshot() {
        let mut registry = Registry { pending: Vec::new() };
        let old = registry.begin(vk::Fence::null());
        let retired = registry.snapshot();
        let newer = registry.begin(vk::Fence::null());
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
        let first = registry.begin(vk::Fence::null());
        let second = registry.begin(vk::Fence::null());
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
        let later = registry.begin(vk::Fence::null());
        assert!(empty.is_complete());
        assert!(!registry.snapshot().is_complete());
        later.complete();
    }

    #[test]
    fn completed_fences_retire_without_another_window_redraw() {
        let mut registry = Registry { pending: Vec::new() };
        let _frame_still_holds_its_claim = registry.begin(vk::Fence::null());
        let retired = registry.snapshot();
        assert_eq!(registry.poll::<()>(|_| Ok(false)), Ok(0));
        assert!(!retired.is_complete());
        assert_eq!(registry.poll::<()>(|_| Ok(true)), Ok(1));
        assert!(retired.is_complete());
        // Reusing the frame's fence creates a new completion, not an ABA alias.
        let later = registry.begin(vk::Fence::null());
        assert!(retired.is_complete());
        assert!(!registry.snapshot().is_complete());
        later.complete();
    }

    #[test]
    fn a_failed_fence_query_does_not_release_unfinished_work() {
        let mut registry = Registry { pending: Vec::new() };
        let work = registry.begin(vk::Fence::null());
        let retired = registry.snapshot();
        assert_eq!(registry.poll(|_| Err("device lost")), Err("device lost"));
        assert!(!retired.is_complete());
        work.complete();
    }
}
