//! A fence cannot be queried, waited or reset while vkQueueSubmit still owns
//! its host access. This receipt ends that host call, not the GPU submission.

use ash::vk;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Completion {
    result: Mutex<Option<Result<(), vk::Result>>>,
    changed: Condvar,
}

impl Completion {
    fn finish(&self, result: Result<(), vk::Result>) {
        *self.result.lock().unwrap_or_else(|error| error.into_inner()) = Some(result);
        self.changed.notify_all();
    }
}

#[derive(Clone, Debug, Default)]
#[must_use = "retain the host-submission receipt until accessing the submitted fence"]
pub(crate) struct SubmissionReceipt(Option<Arc<Completion>>);

pub(super) struct SubmissionCompletion {
    completion: Arc<Completion>,
    finished: bool,
}

impl SubmissionCompletion {
    pub(super) fn finish(mut self, result: Result<(), vk::Result>) {
        self.completion.finish(result);
        self.finished = true;
    }
}

impl Drop for SubmissionCompletion {
    fn drop(&mut self) {
        if !self.finished {
            // A rejected handoff or stopped worker never submitted this fence.
            self.completion.finish(Err(vk::Result::ERROR_DEVICE_LOST));
        }
    }
}

impl SubmissionReceipt {
    /// A synchronous submit has already returned.
    pub(crate) fn completed() -> Self {
        Self::default()
    }

    pub(super) fn pending() -> (Self, SubmissionCompletion) {
        let completion = Arc::new(Completion {
            result: Mutex::new(None),
            changed: Condvar::new(),
        });
        (
            Self(Some(Arc::clone(&completion))),
            SubmissionCompletion {
                completion,
                finished: false,
            },
        )
    }

    #[cfg(test)]
    pub(crate) fn pending_for_test() -> (Self, impl FnOnce(Result<(), vk::Result>)) {
        let (receipt, completion) = Self::pending();
        (receipt, move |result| completion.finish(result))
    }

    /// Never enter the driver while the submitting host call is unfinished.
    pub(crate) fn poll(
        &self,
        query: impl FnOnce() -> Result<bool, vk::Result>,
    ) -> Result<bool, vk::Result> {
        if let Some(completion) = &self.0 {
            let result = *completion.result.lock().unwrap_or_else(|error| error.into_inner());
            match result {
                None => return Ok(false),
                Some(Err(error)) => return Err(error),
                Some(Ok(())) => {}
            }
        }
        query()
    }

    /// Wait only for this host call, returning the budget left for a GPU wait.
    /// A later queue request cannot extend this receipt's lifetime.
    pub(crate) fn wait(&self, timeout_ns: u64) -> Result<u64, vk::Result> {
        let Some(completion) = &self.0 else {
            return Ok(timeout_ns);
        };
        let result = completion.result.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(result) = *result {
            return result.map(|()| timeout_ns);
        }
        if timeout_ns == u64::MAX {
            let result = completion
                .changed
                .wait_while(result, |result| result.is_none())
                .unwrap_or_else(|error| error.into_inner());
            return result.expect("the host call completed").map(|()| timeout_ns);
        }
        let started = Instant::now();
        let (result, _) = completion
            .changed
            .wait_timeout_while(result, Duration::from_nanos(timeout_ns), |result| result.is_none())
            .unwrap_or_else(|error| error.into_inner());
        result.unwrap_or(Err(vk::Result::TIMEOUT)).map(|()| {
            timeout_ns.saturating_sub(
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::mpsc;

    #[test]
    fn an_enqueued_submit_cannot_poll_even_an_already_signaled_fence() {
        let (receipt, completion) = SubmissionReceipt::pending();
        let queried = Cell::new(false);
        assert_eq!(receipt.poll(|| {
            queried.set(true);
            Ok(true)
        }), Ok(false));
        assert!(!queried.get());
        completion.finish(Ok(()));
        assert_eq!(receipt.poll(|| Ok(true)), Ok(true));
    }

    #[test]
    fn another_thread_cannot_poll_until_the_submitting_host_call_returns() {
        let (receipt, completion) = SubmissionReceipt::pending();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (return_tx, return_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            return_rx.recv().unwrap();
            completion.finish(Ok(()));
        });
        entered_rx.recv().unwrap();
        assert_eq!(
            receipt.poll(|| panic!("the driver still owns host access to this fence")),
            Ok(false),
        );
        return_tx.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(receipt.poll(|| Ok(false)), Ok(false), "host return is not GPU completion");
        assert_eq!(receipt.poll(|| Ok(true)), Ok(true));
    }

    #[test]
    fn a_lost_worker_or_failed_submit_never_queries_an_unsubmitted_fence() {
        let (lost, completion) = SubmissionReceipt::pending();
        drop(completion);
        assert_eq!(lost.poll(|| panic!("no submit reached the driver")),
            Err(vk::Result::ERROR_DEVICE_LOST));
        let (failed, completion) = SubmissionReceipt::pending();
        completion.finish(Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert_eq!(failed.wait(0), Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
        assert_eq!(failed.poll(|| panic!("a failed submit has no completion fence")),
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY));
    }

    #[test]
    fn a_host_wait_timeout_does_not_complete_or_cancel_the_submission() {
        let (receipt, completion) = SubmissionReceipt::pending();
        assert_eq!(receipt.wait(0), Err(vk::Result::TIMEOUT));
        assert_eq!(receipt.poll(|| panic!("host call is still pending")), Ok(false));
        completion.finish(Ok(()));
        assert_eq!(receipt.wait(123), Ok(123));
    }

    #[test]
    fn a_later_submission_cannot_extend_an_older_receipt() {
        let (old, complete_old) = SubmissionReceipt::pending();
        complete_old.finish(Ok(()));
        let (later, _complete_later) = SubmissionReceipt::pending();
        assert_eq!(old.wait(0), Ok(0));
        assert_eq!(old.poll(|| Ok(true)), Ok(true));
        assert_eq!(later.poll(|| panic!("later host call has not returned")), Ok(false));
    }
}
