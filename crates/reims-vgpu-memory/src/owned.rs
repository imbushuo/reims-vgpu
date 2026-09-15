//! One HostOps alias lease, shared by CPU references and native allocations.
//!
//! Dropping metadata requests native retirement; dropping the last physical
//! lease requests host unmap. Neither drop calls a backend or HostOps.

use super::{GuestPageFootprint, ImportId};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Default)]
pub(crate) struct Returns {
    retired: Mutex<Vec<ImportId>>,
    released: Mutex<Vec<(usize, usize)>>,
}

fn returns() -> &'static Arc<Returns> {
    static RETURNS: OnceLock<Arc<Returns>> = OnceLock::new();
    RETURNS.get_or_init(|| Arc::new(Returns::default()))
}

/// Ownership of one successful `map_pages` call, not a stable-pointer promise.
#[derive(Debug)]
pub struct OwnedHostAllocation {
    base: usize,
    len: usize,
    footprint: GuestPageFootprint,
    returns: Arc<Returns>,
}

impl OwnedHostAllocation {
    pub(crate) fn new(base: usize, len: usize, footprint: GuestPageFootprint) -> Self {
        Self { base, len, footprint, returns: Arc::clone(returns()) }
    }

    pub fn host_base(&self) -> usize { self.base }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn footprint(&self) -> &GuestPageFootprint { &self.footprint }

    pub(crate) fn retire(&self, id: ImportId) {
        self.returns.retired.lock().unwrap_or_else(|e| e.into_inner()).push(id);
    }
}

impl Drop for OwnedHostAllocation {
    fn drop(&mut self) {
        self.returns.released.lock().unwrap_or_else(|e| e.into_inner())
            .push((self.base, self.len));
    }
}

pub fn take_owned_import_retirements() -> Vec<ImportId> {
    std::mem::take(&mut *returns().retired.lock().unwrap_or_else(|e| e.into_inner()))
}

/// One item per acquired alias lease. Do not deduplicate equal addresses:
/// a host may return the same address for two separately counted acquisitions.
pub fn take_released_owned_host_allocations() -> Vec<(usize, usize)> {
    std::mem::take(&mut *returns().released.lock().unwrap_or_else(|e| e.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GuestRamImport, GuestRef, GuestRun};

    fn import(queue: &Arc<Returns>) -> Arc<GuestRamImport> {
        let owner = Arc::new(OwnedHostAllocation {
            base: 0x10000,
            len: 0x4000,
            footprint: GuestPageFootprint::new(Arc::from([0x4000, 0x8000, 0xc000, 0x10000]), 0x1000)
                .unwrap(),
            returns: Arc::clone(queue),
        });
        Arc::new(GuestRamImport::from_owned_allocation(owner, 0x4000).unwrap())
    }

    #[test]
    fn cpu_runs_and_native_retirement_both_precede_the_only_host_unmap() {
        let queue = Arc::new(Returns::default());
        let import = import(&queue);
        let id = import.id();
        let native = import.owned_host_allocation().unwrap();
        let reference = GuestRef::new(Arc::clone(&import), import.slice(128, 17).unwrap()).unwrap();
        let run = GuestRun::from_reference(&reference).unwrap();
        assert_eq!((run.host_ptr(), run.len()), (0x10080, 17));
        import.retire();
        import.retire();
        assert_eq!(*queue.retired.lock().unwrap(), vec![id]);
        drop(reference);
        drop(import);
        assert!(queue.released.lock().unwrap().is_empty());
        drop(native);
        assert!(queue.released.lock().unwrap().is_empty(), "CPU fallback still owns the alias");
        drop(run);
        assert_eq!(*queue.released.lock().unwrap(), vec![(0x10000, 0x4000)]);
        assert_eq!(*queue.retired.lock().unwrap(), vec![id], "drop cannot retire twice");
    }

    #[test]
    fn abandoned_input_requests_retirement_but_native_owner_holds_memory() {
        let queue = Arc::new(Returns::default());
        let import = import(&queue);
        let id = import.id();
        let native = import.owned_host_allocation().unwrap();
        drop(import);
        assert_eq!(*queue.retired.lock().unwrap(), vec![id]);
        assert!(queue.released.lock().unwrap().is_empty());
        drop(native);
        assert_eq!(*queue.released.lock().unwrap(), vec![(0x10000, 0x4000)]);
    }

    #[test]
    fn failed_native_admission_and_views_release_one_owned_allocation() {
        let queue = Arc::new(Returns::default());
        let import = import(&queue);
        let views: Vec<_> = [0, 128, 256].into_iter().map(|off| {
            GuestRef::new(Arc::clone(&import), import.slice(off, 32).unwrap()).unwrap()
        }).collect();
        drop(import);
        assert!(queue.released.lock().unwrap().is_empty());
        assert_eq!(views[0].import().id(), views[2].import().id());
        drop(views);
        assert_eq!(queue.retired.lock().unwrap().len(), 1);
        assert_eq!(*queue.released.lock().unwrap(), vec![(0x10000, 0x4000)]);
    }

    #[test]
    fn equal_addresses_from_distinct_owned_acquisitions_release_twice() {
        let queue = Arc::new(Returns::default());
        let first = import(&queue);
        let second = import(&queue);
        assert_ne!(first.id(), second.id());
        drop(first);
        drop(second);
        assert_eq!(*queue.released.lock().unwrap(), vec![(0x10000, 0x4000); 2]);
    }

    #[test]
    fn native_alignment_cannot_widen_an_owned_allocation() {
        let queue = Arc::new(Returns::default());
        let import = import(&queue);
        let owner = import.owned_host_allocation().unwrap();
        assert!(GuestRamImport::from_owned_allocation(owner.clone(), 0x8000).is_err());
        assert!(queue.released.lock().unwrap().is_empty());
        assert_eq!(owner.footprint().pages().len(), 4);
        drop(import);
        drop(owner);
        assert_eq!(*queue.released.lock().unwrap(), vec![(0x10000, 0x4000)]);
    }
}
