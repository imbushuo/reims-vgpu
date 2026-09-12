//! Synchronous IOSFC admission, without keeping device state alive while QEMU
//! releases BQL to wait. Tickets retain only this owner and their write.

use parking_lot::{Condvar, Mutex, MutexGuard};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Ready,
    Busy,
    Cancelled,
    Reentrant,
    WrongThread,
}

#[derive(Debug)]
enum Refusal {
    Reentrant,
    WrongThread,
}

pub(crate) enum End {
    Reset,
    Destroy,
}

impl crate::observe::Decline for End {
    fn slug(&self) -> &'static str {
        match self {
            Self::Reset => "iosfc_admission_cancelled_by_reset",
            Self::Destroy => "iosfc_admission_cancelled_by_destroy",
        }
    }
}

impl crate::observe::Decline for Refusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Reentrant => "iosfc_reentrant_device_io",
            Self::WrongThread => "iosfc_admission_wrong_thread",
        }
    }
}

pub(crate) fn report_refusal(status: Status, device: u64) {
    let refusal = match status {
        Status::Reentrant => Refusal::Reentrant,
        Status::WrongThread => Refusal::WrongThread,
        _ => return,
    };
    crate::observe::Emit::decline("iosfc_admission", &refusal)
        .field("device", device)
        .fail();
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Write {
    pub offset: u64,
    pub data: u64,
    pub size: u32,
}

struct State {
    epoch: u64,
    live: bool,
    revision: u64,
    queue: VecDeque<u64>,
    worker: bool,
}

pub(crate) struct Admission {
    state: Mutex<State>,
    changed: Condvar,
}

impl Admission {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                epoch: 1,
                live: true,
                revision: 0,
                queue: VecDeque::new(),
                worker: false,
            }),
            changed: Condvar::new(),
        })
    }

    fn notify(&self, state: &mut State) {
        state.revision = state.revision.wrapping_add(1);
        self.changed.notify_all();
    }

    pub fn issue(self: &Arc<Self>, id: u64, device: u64, write: Write) -> Result<Ticket, Status> {
        if execution_active() {
            return Err(Status::Reentrant);
        }
        let mut state = self.state.lock();
        if !state.live {
            return Err(Status::Cancelled);
        }
        state.queue.push_back(id);
        self.notify(&mut state);
        Ok(Ticket {
            owner: Arc::clone(self),
            id,
            device,
            write,
            epoch: state.epoch,
            thread: std::thread::current().id(),
            observed: AtomicU64::new(state.revision),
            completed: AtomicBool::new(false),
        })
    }

    /// Cancel waiters without waiting for them to return through BQL. The
    /// lifecycle caller holds BQL and separately quiesces the GPU worker.
    pub fn close(&self, device: u64, end: End) {
        let mut state = self.state.lock();
        let cancelled = state.queue.len();
        state.live = false;
        state.epoch = state.epoch.checked_add(1).expect("IOSFC epoch exhausted");
        state.queue.clear();
        self.notify(&mut state);
        drop(state);
        if cancelled != 0 {
            crate::observe::Emit::decline("iosfc_admission", &end)
                .field("device", device)
                .field("cancelled", cancelled)
                .fail();
        }
    }

    pub fn reopen(&self) {
        let mut state = self.state.lock();
        state.live = true;
        self.notify(&mut state);
    }

    /// A queued producer has priority over every subsequent GPU tranche.
    pub fn worker(self: &Arc<Self>) -> Option<Worker> {
        let mut state = self.state.lock();
        if !state.live || state.worker || !state.queue.is_empty() {
            return None;
        }
        state.worker = true;
        Some(Worker {
            owner: Arc::clone(self),
            epoch: state.epoch,
        })
    }

    fn state_released(&self) {
        let mut state = self.state.lock();
        self.notify(&mut state);
    }
}

pub(crate) struct Ticket {
    pub owner: Arc<Admission>,
    pub device: u64,
    pub write: Write,
    id: u64,
    epoch: u64,
    thread: std::thread::ThreadId,
    observed: AtomicU64,
    completed: AtomicBool,
}

impl Ticket {
    fn valid(&self, state: &State) -> bool {
        state.live && state.epoch == self.epoch && state.queue.contains(&self.id)
    }

    pub fn poll(&self) -> Status {
        if execution_active() {
            return Status::Reentrant;
        }
        if self.thread != std::thread::current().id() {
            return Status::WrongThread;
        }
        let state = self.owner.state.lock();
        self.observed.store(state.revision, Ordering::Release);
        if !self.valid(&state) {
            Status::Cancelled
        } else if state.worker || state.queue.front() != Some(&self.id) {
            Status::Busy
        } else {
            Status::Ready
        }
    }

    /// No device lookup, state mutex, guest memory, or HostOps on this path.
    /// The observed revision closes the try/unlock/wait lost-wakeup window.
    pub fn wait(&self) -> Status {
        if execution_active() {
            return Status::Reentrant;
        }
        if self.thread != std::thread::current().id() {
            return Status::WrongThread;
        }
        let observed = self.observed.load(Ordering::Acquire);
        let mut state = self.owner.state.lock();
        while self.valid(&state) && state.revision == observed {
            self.owner.changed.wait(&mut state);
        }
        if self.valid(&state) {
            Status::Ready
        } else {
            Status::Cancelled
        }
    }

    pub fn completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    pub fn complete(&self) {
        self.completed.store(true, Ordering::Release);
    }

    /// The final live admission rearms the worker even if its earlier wakeup
    /// was consumed while producers had priority. Cancelled lifetimes never
    /// call back into their former HostOps context.
    pub fn finish(&self) -> bool {
        let mut state = self.owner.state.lock();
        if !self.valid(&state) {
            return false;
        }
        state.queue.retain(|id| *id != self.id);
        self.owner.notify(&mut state);
        state.queue.is_empty()
    }
}

impl crate::runtime::drain::census::checkpoints::DemandSource for Admission {
    fn checkpoint_demand(&self, epoch: u64) -> crate::runtime::drain::census::checkpoints::Demand {
        use crate::runtime::drain::census::checkpoints::Demand;
        let Some(state) = self.state.try_lock() else {
            return Demand::Contended;
        };
        if !state.live || state.epoch != epoch {
            Demand::Closed
        } else if state.queue.is_empty() {
            Demand::Absent
        } else {
            Demand::Waiting
        }
    }
}

pub(crate) struct Worker {
    owner: Arc<Admission>,
    epoch: u64,
}

impl Worker {
    pub fn observe_tranche(
        &self,
        device: u64,
        started_us: u64,
    ) -> crate::runtime::drain::census::checkpoints::Scope {
        crate::runtime::drain::census::checkpoints::Scope::start(
            device,
            self.epoch,
            self.owner.clone(),
            started_us,
        )
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let mut state = self.owner.state.lock();
        state.worker = false;
        self.owner.notify(&mut state);
    }
}

/// Every state-lock release wakes pure admission waiters, including releases
/// by poll/action readers rather than the GPU worker. Unlock precedes notify:
/// nobody may acquire BQL while a waiter still owns this mutex.
pub(crate) struct StateMutex<T> {
    value: Mutex<T>,
    admission: Arc<Admission>,
}

impl<T> StateMutex<T> {
    pub fn new(value: T, admission: Arc<Admission>) -> Self {
        Self {
            value: Mutex::new(value),
            admission,
        }
    }

    pub fn lock(&self) -> StateGuard<'_, T> {
        StateGuard {
            value: Some(self.value.lock()),
            admission: &self.admission,
        }
    }

    pub fn try_lock(&self) -> Option<StateGuard<'_, T>> {
        Some(StateGuard {
            value: Some(self.value.try_lock()?),
            admission: &self.admission,
        })
    }
}

pub(crate) struct StateGuard<'a, T> {
    value: Option<MutexGuard<'a, T>>,
    admission: &'a Admission,
}

impl<T> Deref for StateGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.as_deref().expect("held state")
    }
}

impl<T> DerefMut for StateGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_deref_mut().expect("held state")
    }
}

impl<T> Drop for StateGuard<'_, T> {
    fn drop(&mut self) {
        drop(self.value.take());
        self.admission.state_released();
    }
}

thread_local! {
    static EXECUTING: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn reentrant(device: u64) -> bool {
    EXECUTING.with(|active| active.borrow().contains(&device))
}

fn execution_active() -> bool {
    // Even a different device's nested admission must not release BQL while
    // the caller still owns the outer device's state mutex.
    EXECUTING.with(|active| !active.borrow().is_empty())
}

/// QEMU's lockless region no longer supplies its device reentrancy guard.
/// KVA debug translation can reach either register window of this device.
pub(crate) struct Execution(u64);

impl Execution {
    pub fn enter(device: u64) -> Result<Self, Status> {
        if reentrant(device) {
            return Err(Status::Reentrant);
        }
        EXECUTING.with(|active| active.borrow_mut().push(device));
        Ok(Self(device))
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        EXECUTING.with(|active| {
            let popped = active.borrow_mut().pop();
            debug_assert_eq!(popped, Some(self.0));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write() -> Write {
        Write {
            offset: 0x1018,
            data: 1,
            size: 4,
        }
    }

    #[test]
    fn checkpoint_snapshot_is_nonblocking_and_does_not_mutate_admission() {
        use crate::runtime::drain::census::checkpoints::{Demand, DemandSource};
        let owner = Admission::new();
        let worker = owner.worker().unwrap();
        assert_eq!(owner.checkpoint_demand(worker.epoch), Demand::Absent);
        let ticket = owner.issue(1, 7, write()).unwrap();
        let held = owner.state.lock();
        let revision = held.revision;
        assert_eq!(owner.checkpoint_demand(worker.epoch), Demand::Contended);
        drop(held);
        assert_eq!(owner.checkpoint_demand(worker.epoch), Demand::Waiting);
        assert_eq!(owner.state.lock().revision, revision);
        assert_eq!(ticket.poll(), Status::Busy);
        drop(worker);
        assert_eq!(ticket.poll(), Status::Ready);
        assert!(ticket.finish());
    }

    #[test]
    fn checkpoint_snapshot_rejects_a_reopened_admission_epoch() {
        use crate::runtime::drain::census::checkpoints::{Demand, DemandSource};
        let owner = Admission::new();
        let worker = owner.worker().unwrap();
        let epoch = worker.epoch;
        owner.close(7, End::Reset);
        assert_eq!(owner.checkpoint_demand(epoch), Demand::Closed);
        drop(worker);
        owner.reopen();
        let ticket = owner.issue(1, 7, write()).unwrap();
        assert_eq!(owner.checkpoint_demand(epoch), Demand::Closed);
        assert_eq!(owner.checkpoint_demand(ticket.epoch), Demand::Waiting);
        assert!(ticket.finish());
    }

    #[test]
    fn queued_producers_have_fifo_priority_over_later_gpu_tranches() {
        let owner = Admission::new();
        let worker = owner.worker().unwrap();
        let first = owner.issue(1, 7, write()).unwrap();
        let second = owner.issue(2, 7, write()).unwrap();
        assert_eq!(first.poll(), Status::Busy);
        drop(worker);
        assert_eq!(first.poll(), Status::Ready);
        assert_eq!(second.poll(), Status::Busy);
        assert!(owner.worker().is_none());
        assert!(!first.finish());
        assert_eq!(second.poll(), Status::Ready);
        assert!(owner.worker().is_none());
        assert!(
            second.finish(),
            "the final admission owes the worker a wake"
        );
        assert!(owner.worker().is_some());
    }

    #[test]
    fn release_between_try_and_wait_cannot_lose_the_wakeup() {
        let owner = Admission::new();
        let worker = owner.worker().unwrap();
        let ticket = owner.issue(1, 7, write()).unwrap();
        assert_eq!(ticket.poll(), Status::Busy);
        drop(worker);
        assert_eq!(ticket.wait(), Status::Ready);
        assert!(ticket.finish());
    }

    #[test]
    fn state_release_wakes_waiter_without_retaining_the_state_lock() {
        let owner = Admission::new();
        let state = Arc::new(StateMutex::new((), Arc::clone(&owner)));
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let other_state = Arc::clone(&state);
        let holder = std::thread::spawn(move || {
            let held = other_state.lock();
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(held);
        });
        held_rx.recv().unwrap();
        let ticket = owner.issue(1, 7, write()).unwrap();
        assert_eq!(ticket.poll(), Status::Ready);
        assert!(state.try_lock().is_none());
        release_tx.send(()).unwrap();
        assert_eq!(ticket.wait(), Status::Ready);
        assert!(state.try_lock().is_some(), "wait returns holding no state");
        assert!(ticket.finish());
        holder.join().unwrap();
    }

    #[test]
    fn reset_cancels_all_old_tickets_without_waiting_for_their_return() {
        let owner = Admission::new();
        let first = owner.issue(1, 7, write()).unwrap();
        let second = owner.issue(2, 7, write()).unwrap();
        assert_eq!(second.poll(), Status::Busy);
        owner.close(7, End::Reset);
        assert_eq!(first.wait(), Status::Cancelled);
        assert_eq!(second.wait(), Status::Cancelled);
        owner.reopen();
        let fresh = owner.issue(3, 7, write()).unwrap();
        assert_eq!(first.poll(), Status::Cancelled);
        assert!(!first.finish(), "old tickets cannot rearm a new lifetime");
        assert!(!second.finish());
        assert_eq!(fresh.poll(), Status::Ready);
        assert!(fresh.finish());
    }

    #[test]
    fn another_thread_cannot_execute_or_wait_for_a_vcpu_handoff() {
        let owner = Admission::new();
        let ticket = Arc::new(owner.issue(1, 7, write()).unwrap());
        let other = Arc::clone(&ticket);
        std::thread::spawn(move || {
            assert_eq!(other.poll(), Status::WrongThread);
            assert_eq!(other.wait(), Status::WrongThread);
        })
        .join()
        .unwrap();
        assert!(ticket.finish());
    }

    #[test]
    fn same_transaction_reentry_is_refused_and_unwind_clears_the_guard() {
        let owner = Admission::new();
        let ticket = owner.issue(1, 8, write()).unwrap();
        let result = std::panic::catch_unwind(|| {
            let _outer = Execution::enter(7).unwrap();
            assert!(matches!(Execution::enter(7), Err(Status::Reentrant)));
            {
                let _different_device = Execution::enter(8).unwrap();
                assert!(reentrant(7));
                assert!(reentrant(8));
            }
            panic!("exercise guard unwind");
        });
        assert!(result.is_err());
        assert!(!reentrant(7));
        assert!(!reentrant(8));
        {
            let _outer = Execution::enter(7).unwrap();
            assert!(matches!(owner.issue(2, 8, write()), Err(Status::Reentrant)));
            assert_eq!(ticket.poll(), Status::Reentrant);
            assert_eq!(ticket.wait(), Status::Reentrant);
        }
        assert!(ticket.finish());
    }
}
