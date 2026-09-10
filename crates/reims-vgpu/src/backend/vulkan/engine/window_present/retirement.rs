//! Completion snapshots for the presents that existed when an object retired.
//! Newer presents cannot extend that object's lifetime. Each captured claim
//! completes only when its own frame fence retires (or the device quiesces).

use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use ash::vk;
use super::super::queue_owner::SubmissionReceipt;

#[derive(Debug)]
struct Completion {
    fence: vk::Fence,
    done: AtomicBool,
    host_submission: SubmissionReceipt,
}

#[derive(Debug)]
pub(super) struct WindowWork(Arc<Completion>);

impl WindowWork {
    pub(super) fn complete(&self) {
        self.0.done.store(true, Ordering::Release);
    }

    pub(super) fn poll(
        &self,
        query: impl FnOnce() -> Result<bool, vk::Result>,
    ) -> Result<bool, vk::Result> {
        self.0.host_submission.poll(query)
    }

    pub(super) fn wait_submission(&self) -> Result<(), vk::Result> {
        self.0.host_submission.wait(u64::MAX).map(|_| ())
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

    fn begin(&mut self, fence: vk::Fence, host_submission: SubmissionReceipt) -> WindowWork {
        self.prune();
        let work = Arc::new(Completion { fence, done: AtomicBool::new(false), host_submission });
        self.pending.push(Arc::clone(&work));
        WindowWork(work)
    }

    fn snapshot(&mut self) -> WindowRetirement {
        self.prune();
        WindowRetirement(self.pending.clone())
    }

    fn poll(
        &mut self,
        mut ready: impl FnMut(vk::Fence) -> Result<bool, vk::Result>,
    ) -> Result<usize, vk::Result> {
        self.prune();
        let mut retired = 0;
        for work in &self.pending {
            if work.host_submission.poll(|| ready(work.fence))? {
                work.done.store(true, Ordering::Release);
                retired += 1;
            }
        }
        self.prune();
        Ok(retired)
    }
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry { pending: Vec::new() });

pub(super) fn begin(fence: vk::Fence, host_submission: SubmissionReceipt) -> WindowWork {
    REGISTRY.lock().unwrap_or_else(|error| error.into_inner()).begin(fence, host_submission)
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
/// Each receipt additionally excludes the queue worker's unfinished host call.
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
        let old = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        let retired = registry.snapshot();
        let newer = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
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
        let first = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        let second = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
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
        let later = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        assert!(empty.is_complete());
        assert!(!registry.snapshot().is_complete());
        later.complete();
    }

    #[test]
    fn completed_fences_retire_without_another_window_redraw() {
        let mut registry = Registry { pending: Vec::new() };
        let _frame_still_holds_its_claim =
            registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        let retired = registry.snapshot();
        assert_eq!(registry.poll(|_| Ok(false)), Ok(0));
        assert!(!retired.is_complete());
        assert_eq!(registry.poll(|_| Ok(true)), Ok(1));
        assert!(retired.is_complete());
        // Reusing the frame's fence creates a new completion, not an ABA alias.
        let later = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        assert!(retired.is_complete());
        assert!(!registry.snapshot().is_complete());
        later.complete();
    }

    #[test]
    fn a_failed_fence_query_does_not_release_unfinished_work() {
        let mut registry = Registry { pending: Vec::new() };
        let work = registry.begin(vk::Fence::null(), SubmissionReceipt::completed());
        let retired = registry.snapshot();
        assert_eq!(
            registry.poll(|_| Err(vk::Result::ERROR_DEVICE_LOST)),
            Err(vk::Result::ERROR_DEVICE_LOST),
        );
        assert!(!retired.is_complete());
        work.complete();
    }

    #[test]
    fn window_retirement_cannot_poll_a_queued_or_still_submitting_fence() {
        let mut registry = Registry { pending: Vec::new() };
        let (receipt, returned) = SubmissionReceipt::pending_for_test();
        let work = registry.begin(vk::Fence::null(), receipt);
        let retired = registry.snapshot();
        assert_eq!(registry.poll(|_| panic!("worker still owns the host fence")), Ok(0));
        assert_eq!(work.poll(|| panic!("present must use the same host gate")), Ok(false));
        assert!(!retired.is_complete());
        returned(Ok(()));
        assert_eq!(registry.poll(|_| Ok(false)), Ok(0));
        assert!(!retired.is_complete(), "host return is not GPU completion");
        assert_eq!(registry.poll(|_| Ok(true)), Ok(1));
        assert!(retired.is_complete());
        let (later_receipt, later_returned) = SubmissionReceipt::pending_for_test();
        let _later = registry.begin(vk::Fence::null(), later_receipt);
        assert!(retired.is_complete(), "fence reuse cannot reopen the old snapshot");
        assert_eq!(registry.poll(|_| panic!("reused fence is submitting again")), Ok(0));
        later_returned(Ok(()));
        assert_eq!(registry.poll(|_| Ok(true)), Ok(1));
    }

    #[test]
    fn a_failed_present_submit_cannot_release_its_retirement_snapshot() {
        let mut registry = Registry { pending: Vec::new() };
        let (receipt, returned) = SubmissionReceipt::pending_for_test();
        let work = registry.begin(vk::Fence::null(), receipt);
        let retired = registry.snapshot();
        returned(Err(vk::Result::ERROR_DEVICE_LOST));
        assert_eq!(
            registry.poll(|_| panic!("failed submission must not query its fence")),
            Err(vk::Result::ERROR_DEVICE_LOST),
        );
        assert!(!retired.is_complete());
        work.complete();
    }
}
