//! A bounded compile mailbox. Work and result owners contain no guest memory.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, Weak};

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Admission {
    Full,
    Stopped,
}

pub(super) enum Status<T> {
    Pending,
    Ready(Arc<T>),
    Cancelled,
    Failed,
}

struct ResultCell<T> {
    value: Mutex<Status<T>>,
}

pub(super) struct Ticket<T>(Arc<ResultCell<T>>);

impl<T> Clone for Ticket<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Ticket<T> {
    pub(super) fn status(&self) -> Status<T> {
        match &*self
            .0
            .value
            .lock()
            .unwrap_or_else(|error| error.into_inner())
        {
            Status::Pending => Status::Pending,
            Status::Ready(value) => Status::Ready(value.clone()),
            Status::Cancelled => Status::Cancelled,
            Status::Failed => Status::Failed,
        }
    }

    pub(super) fn cancel(&self) {
        let old = std::mem::replace(
            &mut *self
                .0
                .value
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            Status::Cancelled,
        );
        drop(old);
    }
}

struct Job<T> {
    result: Weak<ResultCell<T>>,
    build: Box<dyn FnOnce() -> T + Send>,
}

struct Mailbox<T> {
    queued: VecDeque<Job<T>>,
    active: usize,
    stopped: bool,
}

struct Shared<T> {
    mailbox: Mutex<Mailbox<T>>,
    changed: Condvar,
}

pub(super) struct Worker<T> {
    shared: Arc<Shared<T>>,
    capacity: usize,
}

impl<T: Send + Sync + 'static> Worker<T> {
    pub(super) fn new(capacity: usize) -> std::io::Result<Self> {
        assert!(capacity != 0);
        let shared = Arc::new(Shared {
            mailbox: Mutex::new(Mailbox {
                queued: VecDeque::new(),
                active: 0,
                stopped: false,
            }),
            changed: Condvar::new(),
        });
        let owner = shared.clone();
        std::thread::Builder::new()
            .name("reims-pso-compile".into())
            .spawn(move || run(owner))?;
        Ok(Self { shared, capacity })
    }

    pub(super) fn submit(
        &self,
        build: impl FnOnce() -> T + Send + 'static,
    ) -> Result<Ticket<T>, Admission> {
        let mut mailbox = self
            .shared
            .mailbox
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if mailbox.stopped {
            return Err(Admission::Stopped);
        }
        if mailbox.active + mailbox.queued.len() >= self.capacity {
            return Err(Admission::Full);
        }
        let ticket = Ticket(Arc::new(ResultCell {
            value: Mutex::new(Status::Pending),
        }));
        mailbox.queued.push_back(Job {
            result: Arc::downgrade(&ticket.0),
            build: Box::new(build),
        });
        self.shared.changed.notify_one();
        Ok(ticket)
    }
}

impl<T> Drop for Worker<T> {
    fn drop(&mut self) {
        let mut mailbox = self
            .shared
            .mailbox
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mailbox.stopped = true;
        for job in mailbox.queued.drain(..) {
            if let Some(result) = job.result.upgrade() {
                *result
                    .value
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = Status::Cancelled;
            }
        }
        self.shared.changed.notify_one();
    }
}

fn run<T: Send + Sync + 'static>(shared: Arc<Shared<T>>) {
    loop {
        let job = {
            let mut mailbox = shared
                .mailbox
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            while mailbox.queued.is_empty() && !mailbox.stopped {
                mailbox = shared
                    .changed
                    .wait(mailbox)
                    .unwrap_or_else(|error| error.into_inner());
            }
            if mailbox.stopped {
                return;
            }
            mailbox.active += 1;
            mailbox
                .queued
                .pop_front()
                .expect("nonempty compile mailbox")
        };
        if let Some(result) = job.result.upgrade() {
            if matches!(
                *result
                    .value
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()),
                Status::Pending
            ) {
                let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job.build));
                let stopped = shared
                    .mailbox
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .stopped;
                let mut cell = result
                    .value
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if matches!(*cell, Status::Pending) {
                    *cell = match value {
                        Ok(value) if !stopped => Status::Ready(Arc::new(value)),
                        Ok(_) => Status::Cancelled,
                        Err(_) => {
                            crate::observe::fail("native_pipeline_compile reason=worker_panicked");
                            Status::Failed
                        }
                    };
                }
            }
        }
        let mut mailbox = shared
            .mailbox
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        mailbox.active -= 1;
        shared.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn held_compile_keeps_service_free_and_capacity_bounded() {
        let worker = Worker::new(2).unwrap();
        let (entered, started) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let first = worker
            .submit(move || {
                entered.send(()).unwrap();
                wait.recv().unwrap();
                41
            })
            .unwrap();
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        let second = worker.submit(|| 42).unwrap();
        assert!(matches!(worker.submit(|| 43), Err(Admission::Full)));
        assert!(matches!(first.status(), Status::Pending));
        assert!(matches!(second.status(), Status::Pending));
        // Polling/admission takes only the mailbox/result locks; neither waits
        // for the active native call, and dependent work still has no result.
        second.cancel();
        assert!(matches!(second.status(), Status::Cancelled));
        release.send(()).unwrap();
        let mut mailbox = worker.shared.mailbox.lock().unwrap();
        while mailbox.active != 0 || !mailbox.queued.is_empty() {
            let (next, timeout) = worker
                .shared
                .changed
                .wait_timeout(mailbox, Duration::from_secs(2))
                .unwrap();
            assert!(
                !timeout.timed_out(),
                "compile worker did not retire its jobs"
            );
            mailbox = next;
        }
        assert!(matches!(first.status(), Status::Ready(value) if *value == 41));
        assert!(matches!(second.status(), Status::Cancelled));
    }

    #[test]
    fn cancellation_discards_late_result_and_releases_owners_without_joining() {
        struct Owner(mpsc::Sender<()>);
        impl Drop for Owner {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let worker = Worker::new(1).unwrap();
        let (entered, started) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let (dropped, observe) = mpsc::channel();
        let ticket = worker
            .submit(move || {
                let owner = Owner(dropped);
                entered.send(()).unwrap();
                wait.recv().unwrap();
                owner
            })
            .unwrap();
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        ticket.cancel();
        drop(worker);
        assert!(observe.try_recv().is_err());
        release.send(()).unwrap();
        observe.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(matches!(ticket.status(), Status::Cancelled));
    }

    #[test]
    fn failure_is_a_reusable_terminal_result_and_worker_survives_panics() {
        let worker = Worker::new(3).unwrap();
        let error = worker
            .submit(|| Err::<u32, _>("native_create_failed"))
            .unwrap();
        let panicked = worker
            .submit(|| -> Result<u32, &str> { panic!("injected compiler panic") })
            .unwrap();
        let success = worker.submit(|| Ok(42)).unwrap();
        let mut mailbox = worker.shared.mailbox.lock().unwrap();
        while mailbox.active != 0 || !mailbox.queued.is_empty() {
            let (next, timeout) = worker
                .shared
                .changed
                .wait_timeout(mailbox, Duration::from_secs(2))
                .unwrap();
            assert!(!timeout.timed_out());
            mailbox = next;
        }
        let Status::Ready(first) = error.status() else {
            panic!("missing failure result");
        };
        let Status::Ready(again) = error.status() else {
            panic!("missing cached result");
        };
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(*first, Err("native_create_failed"));
        assert!(matches!(panicked.status(), Status::Failed));
        assert!(matches!(success.status(), Status::Ready(value) if *value == Ok(42)));
    }
}
