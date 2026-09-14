//! Exact-bound, copied draw inputs. Plain buffer snapshots may retain the
//! original allocation length and bind offset; no padded shader-visible tail.
//! Only completed allocations enter the
//! thread's queue-owned inventory; its contents are never a guest-data cache.
//! Available storage is bounded by the queue's maximum completed-submission
//! input count AND bytes. Tiny/empty submissions do not reset those bounds.
//! Allocation counters count native constructors; byte levels count owned
//! logical extents, not the driver's physical residency or eliminated copies.
//! Known-zero ranges survive completion only for a reflected no-write input.
//! A resource owner may freeze a certified no-write capture. These shared
//! allocations reenter mutable inventory only after the final immutable lease
//! and every command using them have completed.
//! Shared bytes are attributed once to their capture class, not to every loan.

use super::util::Status;
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, Device, MTLCommandBufferStatus,
    MTLResourceOptions,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{
    AtomicBool, AtomicU64,
    Ordering::{AcqRel, Acquire, Relaxed, Release},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(crate) enum Class {
    Attribute,
    Vertex,
    Fragment,
    Index,
    Indirect,
}

const CLASSES: [Class; 5] = [
    Class::Attribute,
    Class::Vertex,
    Class::Fragment,
    Class::Index,
    Class::Indirect,
];

// Metal Feature Set Tables, Apple2–Apple10: minimum constant-buffer offset.
const APPLE_CONSTANT_BUFFER_OFFSET_ALIGNMENT: u64 = 4;

fn binding_offset_supported(device: &Device, offset: u64) -> bool {
    offset == 0
        || (offset.is_multiple_of(APPLE_CONSTANT_BUFFER_OFFSET_ALIGNMENT)
            && device.supports_family(metal::MTLGPUFamily::Apple2))
}

macro_rules! counters {
    ($($field:ident),+ $(,)?) => {
        struct Counters { $($field: AtomicU64),+ }
        impl Counters {
            const fn new() -> Self { Self { $($field: AtomicU64::new(0)),+ } }
            fn snapshot(&self) -> Snapshot {
                Snapshot { $($field: self.$field.load(Relaxed)),+ }
            }
        }
        #[derive(Clone, Copy, Debug, Default)]
        pub(crate) struct Snapshot { $(pub $field: u64),+ }
    };
}

counters!(
    allocations,
    allocated_bytes,
    copies,
    copied_bytes,
    direct_fill_requests,
    direct_fills,
    direct_fill_bytes,
    direct_fill_failures,
    direct_fill_partial_bytes,
    direct_fill_zeroed_bytes,
    direct_fill_zero_reused_bytes,
    direct_fill_readonly,
    resource_shaped_fills,
    resource_compact_fills,
    reuses,
    requests,
    miss_absent_length,
    miss_exhausted_length,
    miss_source_overlap,
    available_discards,
    available_discard_bytes,
    budget_discards,
    budget_discard_bytes,
    completed_inputs,
    completed_input_bytes,
    completed_peak_buffers,
    completed_peak_bytes,
    live_buffers,
    live_bytes,
    retained_buffers,
    retained_bytes,
    owned_bytes,
    high_water_bytes,
);
static COUNTERS: [Counters; 5] = [const { Counters::new() }; 5];

#[derive(Default)]
struct PoolCounters {
    queues: AtomicU64,
    capacity_buffers: AtomicU64,
    capacity_bytes: AtomicU64,
    metadata_slots: AtomicU64,
    metadata_keys: AtomicU64,
    completed_submissions: AtomicU64,
    completed_inputs: AtomicU64,
    completed_input_bytes: AtomicU64,
    completed_peak_buffers: AtomicU64,
    completed_peak_bytes: AtomicU64,
}

static POOLS: PoolCounters = PoolCounters {
    queues: AtomicU64::new(0),
    capacity_buffers: AtomicU64::new(0),
    capacity_bytes: AtomicU64::new(0),
    metadata_slots: AtomicU64::new(0),
    metadata_keys: AtomicU64::new(0),
    completed_submissions: AtomicU64::new(0),
    completed_inputs: AtomicU64::new(0),
    completed_input_bytes: AtomicU64::new(0),
    completed_peak_buffers: AtomicU64::new(0),
    completed_peak_bytes: AtomicU64::new(0),
};

impl Class {
    fn counters(self) -> &'static Counters {
        &COUNTERS[self as usize]
    }
    fn name(self) -> &'static str {
        match self {
            Self::Attribute => "attribute",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Index => "index",
            Self::Indirect => "indirect",
        }
    }
}

pub(super) fn emit_census() {
    // Requests partition into reuse or one miss. Absent means no inventory key
    // (possibly evicted), exhausted means its entries were leased in the current
    // depletion interval, and overlap means all remaining candidates overlap
    // the source. Completion observations and completed-storage returns end
    // that interval, including snapshots returning after their command completed.
    // None of these is a claim about historical length churn.
    // A resource-shaped fill can request two lengths before capture if its
    // original-size allocation falls back to the compact suffix.
    for class in CLASSES {
        let s = class.counters().snapshot();
        crate::observe::off(format!(
            "metal_input_buffers (cumulative counts; current levels; lifetime high-water) \
             class={} allocations={} allocated_bytes={} copies={} copied_bytes={} reuses={} \
             direct_fill_requests={} direct_fills={} direct_fill_bytes={} direct_fill_failures={} \
             direct_fill_partial_bytes={} direct_fill_zeroed_bytes={} \
             direct_fill_zero_reused_bytes={} direct_fill_readonly={} \
             resource_shaped_fills={} resource_compact_fills={} \
             requests={} miss_absent_length={} miss_exhausted_length={} miss_source_overlap={} \
             available_discards={} available_discard_bytes={} budget_discards={} budget_discard_bytes={} \
             completed_inputs={} completed_input_bytes={} completed_peak_buffers={} completed_peak_bytes={} \
             live_buffers={} live_bytes={} retained_buffers={} retained_bytes={} \
             owned_bytes={} high_water_bytes={}",
            class.name(),
            s.allocations,
            s.allocated_bytes,
            s.copies,
            s.copied_bytes,
            s.reuses,
            s.direct_fill_requests,
            s.direct_fills,
            s.direct_fill_bytes,
            s.direct_fill_failures,
            s.direct_fill_partial_bytes,
            s.direct_fill_zeroed_bytes,
            s.direct_fill_zero_reused_bytes,
            s.direct_fill_readonly,
            s.resource_shaped_fills,
            s.resource_compact_fills,
            s.requests,
            s.miss_absent_length,
            s.miss_exhausted_length,
            s.miss_source_overlap,
            s.available_discards,
            s.available_discard_bytes,
            s.budget_discards,
            s.budget_discard_bytes,
            s.completed_inputs,
            s.completed_input_bytes,
            s.completed_peak_buffers,
            s.completed_peak_bytes,
            s.live_buffers,
            s.live_bytes,
            s.retained_buffers,
            s.retained_bytes,
            s.owned_bytes,
            s.high_water_bytes,
        ));
    }
    // Capacities/metadata are sums over current queue owners. Peaks are maxima
    // of ONE actually completed submission, not sums of class/queue peaks.
    crate::observe::off(format!(
        "metal_input_inventory (current queue levels; cumulative completions; lifetime demand maxima) \
         queues={} capacity_buffers={} capacity_bytes={} metadata_slots={} metadata_keys={} \
         completed_submissions={} completed_inputs={} completed_input_bytes={} \
         completed_peak_buffers={} completed_peak_bytes={}",
        POOLS.queues.load(Relaxed),
        POOLS.capacity_buffers.load(Relaxed),
        POOLS.capacity_bytes.load(Relaxed),
        POOLS.metadata_slots.load(Relaxed),
        POOLS.metadata_keys.load(Relaxed),
        POOLS.completed_submissions.load(Relaxed),
        POOLS.completed_inputs.load(Relaxed),
        POOLS.completed_input_bytes.load(Relaxed),
        POOLS.completed_peak_buffers.load(Relaxed),
        POOLS.completed_peak_bytes.load(Relaxed),
    ));
}

// Accounting follows ownership, including rejected fills and thread teardown.
// It does not decide whether an allocation may be reused.
struct Account {
    class: Class,
    len: u64,
    retained: bool,
}

impl Account {
    fn new(class: Class, len: u64) -> Self {
        let c = class.counters();
        c.live_buffers.fetch_add(1, Relaxed);
        c.live_bytes.fetch_add(len, Relaxed);
        let owned = c.owned_bytes.fetch_add(len, Relaxed) + len;
        c.high_water_bytes.fetch_max(owned, Relaxed);
        Self {
            class,
            len,
            retained: false,
        }
    }

    fn completed(&mut self) {
        let c = self.class.counters();
        c.live_buffers.fetch_sub(1, Relaxed);
        c.live_bytes.fetch_sub(self.len, Relaxed);
        c.retained_buffers.fetch_add(1, Relaxed);
        c.retained_bytes.fetch_add(self.len, Relaxed);
        self.retained = true;
    }
}

impl Drop for Account {
    fn drop(&mut self) {
        let c = self.class.counters();
        if self.retained {
            c.retained_buffers.fetch_sub(1, Relaxed);
            c.retained_bytes.fetch_sub(self.len, Relaxed);
        } else {
            c.live_buffers.fetch_sub(1, Relaxed);
            c.live_bytes.fetch_sub(self.len, Relaxed);
        }
        c.owned_bytes.fetch_sub(self.len, Relaxed);
    }
}

struct Allocation {
    buffer: Buffer,
    account: Account,
    /// Initialized bytes outside this interval are known zero. Unqualified GPU
    /// use widens it to the complete allocation before the handle is exposed.
    dirty: std::ops::Range<usize>,
}

struct Available(Allocation);
struct Filling(Allocation);
struct Sealed(Allocation);

pub(super) type Owner = Rc<RefCell<Pool>>;

#[derive(Default)]
struct Bucket {
    newest: Option<usize>,
    count: usize,
}

struct Node {
    input: Available,
    older: Option<usize>,
    newer: Option<usize>,
    same_older: Option<usize>,
    same_newer: Option<usize>,
}

enum Slot {
    Available(Node),
    Free(Option<usize>),
}

#[derive(Clone, Copy, Default)]
struct Demand {
    count: usize,
    bytes: u64,
}

pub(super) struct Pool {
    queue: CommandQueue,
    by_len: HashMap<usize, Bucket>,
    slots: Vec<Slot>,
    free: Option<usize>,
    oldest: Option<usize>,
    newest: Option<usize>,
    available_count: usize,
    available_bytes: u64,
    peak: Demand,
    // Empty buckets describe depletion since the last completion observation
    // or completed-storage return, not historical length churn. Every return
    // ends this interval before adding storage, even when a snapshot's final
    // cache lease retires long after the completion that authorized its return.
    exhausted_lengths: Vec<usize>,
    #[cfg(test)]
    fail_allocation: bool,
    #[cfg(test)]
    fail_inventory: bool,
    #[cfg(test)]
    poison_fresh: bool,
    #[cfg(test)]
    shape_headroom: Option<u64>,
}

impl Pool {
    pub(super) fn new(queue: CommandQueue) -> Owner {
        POOLS.queues.fetch_add(1, Relaxed);
        Rc::new(RefCell::new(Self {
            queue,
            by_len: HashMap::new(),
            slots: Vec::new(),
            free: None,
            oldest: None,
            newest: None,
            available_count: 0,
            available_bytes: 0,
            peak: Demand::default(),
            exhausted_lengths: Vec::new(),
            #[cfg(test)]
            fail_allocation: false,
            #[cfg(test)]
            fail_inventory: false,
            #[cfg(test)]
            poison_fresh: false,
            #[cfg(test)]
            shape_headroom: None,
        }))
    }

    fn allocate(&mut self, device: &Device, len: usize) -> Option<Buffer> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_allocation) {
            return None;
        }
        if len as u64 > device.max_buffer_length() {
            return None;
        }
        let buffer =
            super::raw_metal::new_buffer(device, len as u64, MTLResourceOptions::StorageModeShared);
        #[cfg(test)]
        if std::mem::take(&mut self.poison_fresh) {
            if let Some(buffer) = buffer.as_ref() {
                assert_eq!(buffer.length(), len as u64);
                assert!(!buffer.contents().is_null());
                unsafe {
                    std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0xa7, len);
                }
            }
        }
        buffer
    }

    fn take(&mut self, len: usize, class: Class, source: Option<*const u8>) -> Option<Filling> {
        let counters = class.counters();
        counters.requests.fetch_add(1, Relaxed);
        let Some(bucket) = self.by_len.get(&len) else {
            counters.miss_absent_length.fetch_add(1, Relaxed);
            return None;
        };
        let Some(mut index) = bucket.newest else {
            counters.miss_exhausted_length.fetch_add(1, Relaxed);
            return None;
        };
        // A caller may read a retained completed buffer. Keep its source alive
        // and separate from the new immutable snapshot, even on an exact match.
        loop {
            let node = self.node(index);
            if source.is_none_or(|source| {
                (node.input.0.buffer.contents() as usize).abs_diff(source as usize) >= len
            }) {
                break;
            }
            let Some(older) = node.same_older else {
                counters.miss_source_overlap.fetch_add(1, Relaxed);
                return None;
            };
            index = older;
        }
        let Available(mut allocation) = self.remove(index, true);
        // The new use, not the historical use, owns the live-byte accounting.
        drop(allocation.account);
        allocation.account = Account::new(class, len as u64);
        counters.reuses.fetch_add(1, Relaxed);
        Some(Filling(allocation))
    }

    fn node(&self, index: usize) -> &Node {
        match &self.slots[index] {
            Slot::Available(node) => node,
            Slot::Free(_) => unreachable!("available index points to a free slot"),
        }
    }

    fn node_mut(&mut self, index: usize) -> &mut Node {
        match &mut self.slots[index] {
            Slot::Available(node) => node,
            Slot::Free(_) => unreachable!("available index points to a free slot"),
        }
    }

    // Both intrusive lists are unlinked before a slot is reused. No slot ID
    // leaves Pool, and neither index can retain stale IDs or ordering tombstones.
    fn remove(&mut self, index: usize, exhausted: bool) -> Available {
        let Slot::Available(node) =
            std::mem::replace(&mut self.slots[index], Slot::Free(self.free))
        else {
            unreachable!("removing a free input slot");
        };
        self.free = Some(index);
        let len = node.input.0.account.len as usize;
        if let Some(older) = node.older {
            self.node_mut(older).newer = node.newer;
        } else {
            self.oldest = node.newer;
        }
        if let Some(newer) = node.newer {
            self.node_mut(newer).older = node.older;
        } else {
            self.newest = node.older;
        }
        if let Some(older) = node.same_older {
            self.node_mut(older).same_newer = node.same_newer;
        }
        if let Some(newer) = node.same_newer {
            self.node_mut(newer).same_older = node.same_older;
        } else {
            self.by_len.get_mut(&len).unwrap().newest = node.same_older;
        }
        let bucket = self.by_len.get_mut(&len).unwrap();
        bucket.count -= 1;
        if bucket.count == 0 {
            if exhausted {
                // Reserved at completion for every potentially available key.
                debug_assert!(self.exhausted_lengths.len() < self.exhausted_lengths.capacity());
                self.exhausted_lengths.push(len);
            } else {
                self.by_len.remove(&len);
                POOLS.metadata_keys.fetch_sub(1, Relaxed);
            }
        }
        self.available_count -= 1;
        self.available_bytes -= node.input.0.account.len;
        node.input
    }

    fn evict_oldest(&mut self, budget: bool) {
        let Available(allocation) = self.remove(self.oldest.unwrap(), false);
        let c = allocation.account.class.counters();
        c.available_discards.fetch_add(1, Relaxed);
        c.available_discard_bytes
            .fetch_add(allocation.account.len, Relaxed);
        if budget {
            c.budget_discards.fetch_add(1, Relaxed);
            c.budget_discard_bytes
                .fetch_add(allocation.account.len, Relaxed);
        }
        drop(allocation);
    }

    fn forget_exhausted(&mut self) {
        for len in self.exhausted_lengths.drain(..) {
            let bucket = self.by_len.remove(&len).unwrap();
            debug_assert_eq!(bucket.count, 0);
            POOLS.metadata_keys.fetch_sub(1, Relaxed);
        }
    }

    fn clear(&mut self) {
        while self.oldest.is_some() {
            self.evict_oldest(false);
        }
        self.forget_exhausted();
    }

    fn observe_completion(
        &mut self,
        inputs: &[Sealed],
        snapshots: &[Arc<ReadOnlySnapshot>],
    ) -> Result<(), Status> {
        let mut demand = Demand {
            count: inputs.len() + snapshots.len(),
            bytes: 0,
        };
        let mut classes = [Demand::default(); CLASSES.len()];
        for account in inputs.iter().map(|input| &input.0.account).chain(
            snapshots
                .iter()
                .map(|snapshot| &snapshot.allocation.account),
        ) {
            demand.bytes = demand
                .bytes
                .checked_add(account.len)
                .ok_or_else(|| Status::execute("metal_render_input_demand_overflow"))?;
            let class = &mut classes[account.class as usize];
            class.count += 1;
            class.bytes += account.len;
        }
        let peak = Demand {
            count: self.peak.count.max(demand.count),
            bytes: self.peak.bytes.max(demand.bytes),
        };
        POOLS
            .capacity_buffers
            .fetch_add((peak.count - self.peak.count) as u64, Relaxed);
        POOLS
            .capacity_bytes
            .fetch_add(peak.bytes - self.peak.bytes, Relaxed);
        self.peak = peak;
        POOLS.completed_submissions.fetch_add(1, Relaxed);
        POOLS
            .completed_inputs
            .fetch_add(demand.count as u64, Relaxed);
        POOLS.completed_input_bytes.fetch_add(demand.bytes, Relaxed);
        POOLS
            .completed_peak_buffers
            .fetch_max(demand.count as u64, Relaxed);
        POOLS.completed_peak_bytes.fetch_max(demand.bytes, Relaxed);
        for (class, demand) in CLASSES.into_iter().zip(classes) {
            let c = class.counters();
            c.completed_inputs.fetch_add(demand.count as u64, Relaxed);
            c.completed_input_bytes.fetch_add(demand.bytes, Relaxed);
            c.completed_peak_buffers
                .fetch_max(demand.count as u64, Relaxed);
            c.completed_peak_bytes.fetch_max(demand.bytes, Relaxed);
        }
        self.forget_exhausted();
        #[cfg(test)]
        if std::mem::take(&mut self.fail_inventory) {
            return Err(Status::execute("metal_render_input_inventory_alloc_failed"));
        }
        let possible_keys = self
            .available_count
            .saturating_add(demand.count)
            .min(peak.count);
        self.exhausted_lengths
            .try_reserve(possible_keys)
            .map_err(|_| Status::execute("metal_render_input_inventory_alloc_failed"))?;
        Ok(())
    }

    fn insert_completed(&mut self, mut allocation: Allocation) -> Result<(), Status> {
        // This is the inventory's return boundary, not observe_completion:
        // snapshot retirement can reach it between acquisitions. Clear the
        // depletion-only keys before any bucket can be revived, evicted, or
        // replaced by a different length. Thus each tombstone names exactly one
        // empty bucket and metadata stays bounded by completed demand.
        self.forget_exhausted();
        let len = allocation.account.len;
        debug_assert!(self.peak.count > 0 && len <= self.peak.bytes);
        let needs_eviction =
            self.available_count >= self.peak.count || self.available_bytes > self.peak.bytes - len;
        if !self.by_len.contains_key(&(len as usize)) {
            self.by_len
                .try_reserve(1)
                .map_err(|_| Status::execute("metal_render_input_inventory_alloc_failed"))?;
        }
        if self.free.is_none() && !needs_eviction {
            self.slots
                .try_reserve(1)
                .map_err(|_| Status::execute("metal_render_input_inventory_alloc_failed"))?;
        }
        // Evict before inserting so even temporary slot/index storage never
        // exceeds the demand-derived count. Newly returned inputs are youngest.
        while self.available_count >= self.peak.count
            || self.available_bytes > self.peak.bytes - len
        {
            self.evict_oldest(true);
        }
        let index = if let Some(free) = self.free {
            let Slot::Free(next) = self.slots[free] else {
                unreachable!("free list points to an available slot");
            };
            self.free = next;
            free
        } else {
            self.slots.push(Slot::Free(None));
            POOLS.metadata_slots.fetch_add(1, Relaxed);
            self.slots.len() - 1
        };
        let same_older = self
            .by_len
            .get(&(len as usize))
            .and_then(|bucket| bucket.newest);
        if let Some(older) = self.newest {
            self.node_mut(older).newer = Some(index);
        } else {
            self.oldest = Some(index);
        }
        if let Some(older) = same_older {
            self.node_mut(older).same_newer = Some(index);
        }
        allocation.account.completed();
        self.slots[index] = Slot::Available(Node {
            input: Available(allocation),
            older: self.newest,
            newer: None,
            same_older,
            same_newer: None,
        });
        self.newest = Some(index);
        let bucket = self.by_len.entry(len as usize).or_insert_with(|| {
            POOLS.metadata_keys.fetch_add(1, Relaxed);
            Bucket::default()
        });
        bucket.newest = Some(index);
        bucket.count += 1;
        self.available_count += 1;
        self.available_bytes += len;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn clear_available(&mut self) {
        // Test isolation starts a new demand window. Production discard does
        // not reset the queue's completed-demand maxima.
        self.clear();
        POOLS
            .capacity_buffers
            .fetch_sub(self.peak.count as u64, Relaxed);
        POOLS.capacity_bytes.fetch_sub(self.peak.bytes, Relaxed);
        POOLS
            .metadata_slots
            .fetch_sub(self.slots.len() as u64, Relaxed);
        self.peak = Demand::default();
        self.slots = Vec::new();
        self.free = None;
        self.by_len = HashMap::new();
        self.exhausted_lengths = Vec::new();
        self.fail_allocation = false;
        self.fail_inventory = false;
        self.poison_fresh = false;
        self.shape_headroom = None;
    }

    #[cfg(test)]
    pub(super) fn fail_next_allocation(&mut self) {
        self.fail_allocation = true;
    }

    #[cfg(test)]
    pub(super) fn inventory(&self) -> Vec<(usize, usize)> {
        let mut sizes: Vec<_> = self
            .by_len
            .iter()
            .filter(|(_, bucket)| bucket.count != 0)
            .map(|(&size, bucket)| (size, bucket.count))
            .collect();
        sizes.sort_unstable();
        sizes
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.clear();
        POOLS
            .capacity_buffers
            .fetch_sub(self.peak.count as u64, Relaxed);
        POOLS.capacity_bytes.fetch_sub(self.peak.bytes, Relaxed);
        POOLS
            .metadata_slots
            .fetch_sub(self.slots.len() as u64, Relaxed);
        POOLS.queues.fetch_sub(1, Relaxed);
    }
}

impl Filling {
    fn allocate(
        len: usize,
        class: Class,
        failure: &'static str,
        allocate: impl FnOnce() -> Option<Buffer>,
    ) -> Result<Self, Status> {
        let buffer = allocate().ok_or_else(|| Status::execute(failure).field("len", len))?;
        let c = class.counters();
        c.allocations.fetch_add(1, Relaxed);
        c.allocated_bytes.fetch_add(buffer.length(), Relaxed);
        if buffer.length() != len as u64 {
            return Err(Status::execute("metal_render_input_length_mismatch")
                .field("requested", len)
                .field("actual", buffer.length()));
        }
        Ok(Self(Allocation {
            buffer,
            account: Account::new(class, len as u64),
            dirty: 0..len,
        }))
    }
}

/// Immutable input binding, freshly filled or a contained readonly loan. The Rc
/// keeps it on its originating thread and queue; dropping it never recycles it.
pub(crate) struct Filled {
    allocation: Storage,
    owner: Owner,
    offset: usize,
    captured_len: usize,
    gpu_read_only: Option<super::buffer_extent::ReadOnlyCapture>,
}

enum Storage {
    Exclusive(Allocation),
    ReadOnly(Arc<ReadOnlySnapshot>),
}

impl std::ops::Deref for Storage {
    type Target = Allocation;

    fn deref(&self) -> &Allocation {
        match self {
            Self::Exclusive(allocation) => allocation,
            Self::ReadOnly(snapshot) => &snapshot.allocation,
        }
    }
}

/// Private copied storage, not an import or a guest-memory lease. One physical
/// allocation owns one Account; cloning a read lease never double-counts it.
pub(crate) struct ReadOnlySnapshot {
    allocation: Allocation,
    queue: CommandQueue,
    offset: usize,
    captured_len: usize,
    pending: AtomicU64,
    completed: AtomicBool,
}

impl ReadOnlySnapshot {
    /// A discarded/unverified submission leaves its pending count charged.
    /// Even if a cache remains, that allocation can never be recycled early.
    fn into_completed(self: Arc<Self>) -> Option<Allocation> {
        let snapshot = Arc::try_unwrap(self).ok()?;
        (snapshot.pending.load(Acquire) == 0 && snapshot.completed.load(Acquire))
            .then_some(snapshot.allocation)
    }

    pub(crate) fn retire(self: Arc<Self>, device: &Device) {
        let owner = super::runtime::thread_input_pool(device);
        if owner.borrow().queue.as_ptr() != self.queue.as_ptr() {
            return;
        }
        if let Some(allocation) = self.into_completed() {
            if let Err(status) = owner.borrow_mut().insert_completed(allocation) {
                if let Some(emit) =
                    crate::observe::Emit::refusal("metal_input_snapshot_retire", &status)
                {
                    emit.fail();
                }
            }
        }
    }

    pub(crate) fn bind(
        self: &Arc<Self>,
        device: &Device,
        proof: super::buffer_extent::ReadOnlyCapture,
    ) -> Option<Filled> {
        self.bind_window(device, proof, 0)
    }

    /// Lend a contained readonly view, keeping the allocation's original
    /// initialized/dirty intervals. Bytes outside this binding's reflected
    /// reach need not become zero, but remain initialized and cannot be read
    /// by the certified shader. Never relabel them as known zero for recycling.
    pub(crate) fn bind_window(
        self: &Arc<Self>,
        device: &Device,
        proof: super::buffer_extent::ReadOnlyCapture,
        relative_offset: usize,
    ) -> Option<Filled> {
        let remaining = self
            .len()
            .checked_sub(relative_offset)
            .filter(|&len| len != 0)?;
        let captured_len = usize::try_from(
            proof
                .bytes()
                .unwrap_or(remaining as u64)
                .min(remaining as u64),
        )
        .ok()?;
        if relative_offset.checked_add(captured_len)? > self.captured_len {
            return None;
        }
        let offset = self.offset.checked_add(relative_offset)?;
        if !binding_offset_supported(device, offset as u64) {
            return None;
        }
        let owner = super::runtime::thread_input_pool(device);
        if owner.borrow().queue.as_ptr() != self.queue.as_ptr() {
            return None;
        }
        Some(Filled {
            allocation: Storage::ReadOnly(Arc::clone(self)),
            owner,
            offset,
            captured_len,
            gpu_read_only: Some(proof),
        })
    }

    fn len(&self) -> usize {
        self.allocation.account.len as usize - self.offset
    }
}

impl Filled {
    /// Freeze only an already complete, certified capture. Neither the cache
    /// nor its submission leases expose a refill route for this allocation.
    pub(crate) fn freeze(self) -> Result<Arc<ReadOnlySnapshot>, Self> {
        if self.gpu_read_only.is_none() {
            return Err(self);
        }
        if let Storage::ReadOnly(snapshot) = &self.allocation {
            // A view is not a new capture: returning the original snapshot
            // under the view's source key would mislabel its offset/coverage.
            if self.offset != snapshot.offset || self.captured_len != snapshot.captured_len {
                return Err(self);
            }
        }
        match self.allocation {
            Storage::ReadOnly(snapshot) => Ok(snapshot),
            Storage::Exclusive(allocation) => Ok(Arc::new(ReadOnlySnapshot {
                allocation,
                queue: self.owner.borrow().queue.clone(),
                offset: self.offset,
                captured_len: self.captured_len,
                pending: AtomicU64::new(0),
                completed: AtomicBool::new(false),
            })),
        }
    }

    /// Shader-visible length, never the allocation's uncaptured prefix.
    pub(crate) fn len(&self) -> usize {
        self.allocation.account.len as usize - self.offset
    }

    pub(crate) fn binding_offset(&self) -> u64 {
        self.offset as u64
    }

    /// Required captured coverage for this binding. A readonly loan may be
    /// backed by a wider immutable capture; copy counters name actual fills.
    pub(crate) fn captured_len(&self) -> usize {
        self.captured_len
    }

    #[cfg(test)]
    pub(crate) fn test_contents(&self) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                self.allocation.buffer.contents().cast(),
                self.allocation.account.len as usize,
            )
        }
        .to_vec()
    }
}

impl std::fmt::Debug for Filled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filled")
            .field("len", &self.len())
            .field("offset", &self.offset)
            .field("captured_len", &self.captured_len)
            .field("gpu_read_only", &self.gpu_read_only.is_some())
            .finish_non_exhaustive()
    }
}

struct Acquired {
    allocation: Allocation,
    owner: Owner,
    destination: *mut u8,
    fresh: bool,
}

fn acquire(
    device: &Device,
    len: usize,
    class: Class,
    allocation_failure: &'static str,
    source: Option<*const u8>,
    resource_shaped: bool,
) -> Result<Acquired, Status> {
    if len == 0 {
        return Err(Status::args("metal_render_input_span_invalid").field("len", len));
    }
    let owner = super::runtime::thread_input_pool(device);
    let recycled = owner.borrow_mut().take(len, class, source);
    let fresh = recycled.is_none();
    let Filling(allocation) = match recycled {
        Some(input) => input,
        None => Filling::allocate(len, class, allocation_failure, || {
            // Reuse spends no new residency. A cold expanded snapshot must fit
            // the device's own working-set headroom, not an arbitrary ratio or
            // a guest-data cache limit. This is placement policy, not permission
            // to truncate: failure returns to the exact suffix representation.
            if resource_shaped {
                let headroom = device
                    .recommended_max_working_set_size()
                    .saturating_sub(device.current_allocated_size());
                #[cfg(test)]
                let headroom = owner.borrow().shape_headroom.unwrap_or(headroom);
                if len as u64 > headroom {
                    return None;
                }
            }
            owner.borrow_mut().allocate(device, len)
        })?,
    };
    if allocation.buffer.length() != len as u64 {
        return Err(Status::execute("metal_render_input_length_mismatch")
            .field("requested", len)
            .field("actual", allocation.buffer.length()));
    }
    let destination = allocation.buffer.contents().cast::<u8>();
    if destination.is_null() {
        return Err(Status::execute("metal_render_input_contents_missing").field("len", len));
    }
    // Both the TLS access and all pool borrows end before this value returns.
    Ok(Acquired {
        allocation,
        owner,
        destination,
        fresh,
    })
}

/// Copy all bytes before returning. Never aliases the caller, even for aligned
/// allocations, and never exposes mutable contents after filling.
///
/// # Safety
/// `data` must point to `len` readable bytes for this call.
pub(super) unsafe fn copy(
    device: &Device,
    data: *const u8,
    len: usize,
    class: Class,
    allocation_failure: &'static str,
) -> Result<Filled, Status> {
    if data.is_null() || len == 0 {
        return Err(Status::args("metal_render_input_span_invalid").field("len", len));
    }
    let mut acquired = acquire(device, len, class, allocation_failure, Some(data), false)?;
    // SAFETY: the source is caller-bounded and the destination is an exclusively
    // filling, exact-length Shared allocation, either fresh or disjoint from
    // the source. No submitted draw can read it while filling.
    unsafe {
        std::ptr::copy_nonoverlapping(data, acquired.destination, len);
    }
    let c = class.counters();
    c.copies.fetch_add(1, Relaxed);
    c.copied_bytes.fetch_add(len as u64, Relaxed);
    acquired.allocation.dirty = 0..len;
    Ok(Filled {
        allocation: Storage::Exclusive(acquired.allocation),
        owner: acquired.owner,
        offset: 0,
        captured_len: len,
        gpu_read_only: None,
    })
}

#[derive(Debug)]
pub(crate) enum FillError<E> {
    Backend(Status),
    Callback(E),
}

struct DirectAttempt {
    class: Class,
    success: bool,
}

impl DirectAttempt {
    fn new(class: Class) -> Self {
        class.counters().direct_fill_requests.fetch_add(1, Relaxed);
        Self {
            class,
            success: false,
        }
    }
}

impl Drop for DirectAttempt {
    fn drop(&mut self) {
        if !self.success {
            self.class
                .counters()
                .direct_fill_failures
                .fetch_add(1, Relaxed);
        }
    }
}

/// Synchronously fill an exact-length input without an intermediate host copy.
///
/// The callback must report the number of bytes successfully read into the
/// destination. Only a full-length success produces an immutable `Filled`.
/// Its higher-ranked view cannot escape in either the result or callback error.
/// No pool/TLS borrow is held during the callback, so dependency settlement may
/// reenter this queue. Fresh storage is zeroed before any Rust slice is formed;
/// recycled storage is initialized but its previous contents are not a seed.
///
/// Failure counters include allocation/callback failures. Partial bytes count
/// only explicit short-success reports; an opaque callback error has no known
/// byte count. Initialization writes are separate from successful fill bytes.
pub(crate) fn fill<E>(
    device: &Device,
    len: usize,
    class: Class,
    allocation_failure: &'static str,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    let mut attempt = DirectAttempt::new(class);
    if len > isize::MAX as usize {
        return Err(FillError::Backend(
            Status::args("metal_render_input_fill_span_too_large").field("len", len),
        ));
    }
    let acquired =
        acquire(device, len, class, allocation_failure, None, false).map_err(FillError::Backend)?;
    let filled = complete_fill(acquired, 0, None, class, callback)?;
    attempt.success = true;
    Ok(filled)
}

/// Snapshot the declared buffer's full logical suffix at its actual bind offset.
///
/// `MTLBuffer.length - offset` is exactly the captured byte count on both paths.
/// Apple GPU families 2+ specify four-byte constant-buffer offset alignment
/// (Metal Feature Set Tables, Resources); other families and unaligned offsets
/// retain the existing compact representation. No size rounding is allowed.
/// Cold original-size allocations are bounded by maxBufferLength and the
/// device's recommended working-set headroom. A failed expansion retries the
/// compact suffix before any guest bytes have been read.
pub(crate) fn fill_resource_suffix<E>(
    device: &Device,
    allocation_len: u64,
    offset: u64,
    class: Class,
    allocation_failure: &'static str,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    fill_resource_region(
        device,
        allocation_len,
        offset,
        None,
        class,
        allocation_failure,
        callback,
    )
}

/// Preserve the complete native suffix while capturing only a proved prefix.
/// Every byte outside capture is initialized to zero, including recycled tails.
pub(crate) fn fill_resource_prefix<E>(
    device: &Device,
    allocation_len: u64,
    offset: u64,
    read_len: usize,
    class: Class,
    allocation_failure: &'static str,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    if allocation_len.checked_sub(offset) == Some(read_len as u64) {
        return fill_resource_suffix(
            device,
            allocation_len,
            offset,
            class,
            allocation_failure,
            callback,
        );
    }
    fill_resource_region(
        device,
        allocation_len,
        offset,
        Some(read_len),
        class,
        allocation_failure,
        callback,
    )
}

/// A no-write shader input. The proof is tied to the same stage/slot
/// as capture, and callers must seal it only for that input's draw.
pub(crate) fn fill_read_only_resource_prefix<E>(
    device: &Device,
    allocation_len: u64,
    offset: u64,
    proof: super::buffer_extent::ReadOnlyCapture,
    class: Class,
    allocation_failure: &'static str,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    let read_len = allocation_len
        .checked_sub(offset)
        .and_then(|len| usize::try_from(len.min(proof.bytes().unwrap_or(len))).ok())
        .ok_or_else(|| {
            FillError::Backend(
                Status::args("metal_render_input_suffix_invalid")
                    .field("allocation_len", allocation_len)
                    .field("offset", offset),
            )
        })?;
    let mut filled = fill_resource_prefix(
        device,
        allocation_len,
        offset,
        read_len,
        class,
        allocation_failure,
        callback,
    )?;
    filled.gpu_read_only = Some(proof);
    class.counters().direct_fill_readonly.fetch_add(1, Relaxed);
    Ok(filled)
}

fn fill_resource_region<E>(
    device: &Device,
    allocation_len: u64,
    offset: u64,
    read_len: Option<usize>,
    class: Class,
    allocation_failure: &'static str,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    if offset == 0 && read_len.is_none() {
        let len = usize::try_from(allocation_len).map_err(|_| {
            FillError::Backend(
                Status::args("metal_render_input_fill_span_too_large").field("len", allocation_len),
            )
        })?;
        let filled = fill(device, len, class, allocation_failure, callback)?;
        class.counters().resource_shaped_fills.fetch_add(1, Relaxed);
        return Ok(filled);
    }
    let mut attempt = DirectAttempt::new(class);
    let len = allocation_len
        .checked_sub(offset)
        .and_then(|len| usize::try_from(len).ok())
        .filter(|&len| len > 0 && len <= isize::MAX as usize)
        .ok_or_else(|| {
            FillError::Backend(
                Status::args("metal_render_input_suffix_invalid")
                    .field("allocation_len", allocation_len)
                    .field("offset", offset),
            )
        })?;
    let original = usize::try_from(allocation_len)
        .ok()
        .filter(|&size| size <= isize::MAX as usize);
    let shaped = original.filter(|_| offset != 0 && binding_offset_supported(device, offset));
    let acquired = shaped.and_then(|size| {
        acquire(device, size, class, allocation_failure, None, offset != 0)
            .inspect_err(|status| {
                if let Some(emit) =
                    crate::observe::Emit::refusal("metal_input_resource_shape", status)
                {
                    emit.field("allocation_len", allocation_len)
                        .field("fallback_len", len)
                        .fail_once(allocation_len);
                }
            })
            .ok()
            .map(|acquired| (acquired, offset as usize))
    });
    let (acquired, native_offset) = match acquired {
        Some(acquired) => acquired,
        None => (
            acquire(device, len, class, allocation_failure, None, false)
                .map_err(FillError::Backend)?,
            0,
        ),
    };
    let filled = complete_fill(acquired, native_offset, read_len, class, callback)?;
    let c = class.counters();
    if native_offset as u64 == offset {
        c.resource_shaped_fills.fetch_add(1, Relaxed);
    } else {
        c.resource_compact_fills.fetch_add(1, Relaxed);
    }
    attempt.success = true;
    Ok(filled)
}

fn complete_fill<E>(
    mut acquired: Acquired,
    offset: usize,
    read_len: Option<usize>,
    class: Class,
    callback: impl for<'bytes> FnOnce(&'bytes mut [u8]) -> Result<usize, E>,
) -> Result<Filled, FillError<E>> {
    let c = class.counters();
    let allocation_len = acquired.allocation.account.len as usize;
    let available = allocation_len - offset;
    let len = read_len.unwrap_or(available);
    if len > available {
        return Err(FillError::Backend(
            Status::args("metal_render_input_capture_past_suffix")
                .field("captured", len)
                .field("available", available),
        ));
    }
    let tail = offset + len;
    if acquired.fresh {
        // Native allocation does not promise initialized Rust u8 values.
        // SAFETY: the exclusively filling allocation owns every byte.
        unsafe {
            std::ptr::write_bytes(acquired.destination, 0, allocation_len);
        }
        c.direct_fill_zeroed_bytes
            .fetch_add(allocation_len as u64, Relaxed);
    } else {
        let dirty = &acquired.allocation.dirty;
        let mut zeroed = 0;
        for range in [
            dirty.start..dirty.end.min(offset),
            dirty.start.max(tail)..dirty.end,
        ] {
            if range.start < range.end {
                unsafe {
                    std::ptr::write_bytes(
                        acquired.destination.add(range.start),
                        0,
                        range.end - range.start,
                    );
                }
                zeroed += range.end - range.start;
            }
        }
        if zeroed != 0 {
            c.direct_fill_zeroed_bytes.fetch_add(zeroed as u64, Relaxed);
        }
        let reused = allocation_len - len - zeroed;
        if reused != 0 {
            c.direct_fill_zero_reused_bytes
                .fetch_add(reused as u64, Relaxed);
        }
    }
    // SAFETY: the full exact-length storage is initialized, exclusively filling,
    // and remains owned through this synchronous, non-escaping callback.
    let view = unsafe { std::slice::from_raw_parts_mut(acquired.destination.add(offset), len) };
    let reported = callback(view).map_err(FillError::Callback)?;
    if reported != len {
        if reported < len {
            c.direct_fill_partial_bytes
                .fetch_add(reported as u64, Relaxed);
        }
        return Err(FillError::Backend(
            Status::execute("metal_render_input_fill_incomplete")
                .field("requested", len)
                .field("reported", reported),
        ));
    }
    c.direct_fills.fetch_add(1, Relaxed);
    c.direct_fill_bytes.fetch_add(len as u64, Relaxed);
    acquired.allocation.dirty = offset..tail;
    Ok(Filled {
        allocation: Storage::Exclusive(acquired.allocation),
        owner: acquired.owner,
        offset,
        captured_len: len,
        gpu_read_only: None,
    })
}

/// The render batch supplies its actual command identity. There is no API to
/// return individual sealed inputs early, or to recycle merely submitted work.
#[derive(Default)]
pub(super) struct Submission {
    owner: Option<Owner>,
    command: Option<CommandBuffer>,
    inputs: Vec<Sealed>,
    snapshots: Vec<Arc<ReadOnlySnapshot>>,
}

impl Submission {
    pub(super) fn begin(&mut self, command: &CommandBufferRef, owner: Owner) -> Result<(), Status> {
        if self.command.is_some() || !self.inputs.is_empty() || !self.snapshots.is_empty() {
            return Err(Status::execute("metal_render_input_submission_pending"));
        }
        if !super::raw_metal::command_buffer_uses_queue(command, &owner.borrow().queue) {
            return Err(Status::execute("metal_render_input_queue_mismatch"));
        }
        self.owner = Some(owner);
        self.command = Some(command.to_owned());
        Ok(())
    }

    /// Record only the input binding for which `Filled` was captured. A no-write
    /// certificate belongs to that immutable shader/stage/slot, not to arbitrary
    /// uses of a cloned Metal handle. Uncertified uses conservatively dirty all.
    pub(super) fn seal(&mut self, input: Filled) -> Result<Buffer, Status> {
        if !self
            .owner
            .as_ref()
            .is_some_and(|owner| Rc::ptr_eq(owner, &input.owner))
            || self.command.is_none()
        {
            return Err(Status::execute("metal_render_input_owner_mismatch"));
        }
        let buffer = input.allocation.buffer.clone();
        match input.allocation {
            Storage::Exclusive(mut allocation) => {
                self.inputs
                    .try_reserve(1)
                    .map_err(|_| Status::execute("metal_render_input_tracking_alloc_failed"))?;
                if input.gpu_read_only.is_none() {
                    allocation.dirty = 0..allocation.account.len as usize;
                }
                self.inputs.push(Sealed(allocation));
            }
            Storage::ReadOnly(snapshot) => {
                if !self
                    .snapshots
                    .iter()
                    .any(|held| Arc::ptr_eq(held, &snapshot))
                {
                    self.snapshots
                        .try_reserve(1)
                        .map_err(|_| Status::execute("metal_render_input_tracking_alloc_failed"))?;
                    snapshot.pending.fetch_add(1, AcqRel);
                    self.snapshots.push(snapshot);
                }
            }
        }
        Ok(buffer)
    }

    pub(super) fn completed(&mut self, command: &CommandBufferRef) -> Result<(), Status> {
        if self.command.as_ref().map(|pending| pending.as_ptr()) != Some(command.as_ptr()) {
            return Err(Status::execute("metal_render_input_completion_mismatch"));
        }
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Status::execute("metal_render_input_not_completed"));
        }
        let Some(owner) = self.owner.as_ref() else {
            return Err(Status::execute("metal_render_input_owner_missing"));
        };
        let result = (|| {
            let mut pool = owner.borrow_mut();
            pool.observe_completion(&self.inputs, &self.snapshots)?;
            for Sealed(allocation) in self.inputs.drain(..) {
                pool.insert_completed(allocation)?;
            }
            for snapshot in self.snapshots.drain(..) {
                snapshot.pending.fetch_sub(1, AcqRel);
                snapshot.completed.store(true, Release);
                if let Some(allocation) = snapshot.into_completed() {
                    pool.insert_completed(allocation)?;
                }
            }
            Ok(())
        })();
        if let Err(status) = result {
            // GPU completion was already verified. Terminal retirement errors
            // must not leave inputs live or let a retry count this completion twice.
            self.discard();
            return Err(status);
        }
        self.owner = None;
        self.command = None;
        self.snapshots.clear();
        Ok(())
    }

    pub(super) fn discard(&mut self) {
        self.inputs.clear();
        self.snapshots.clear();
        if let Some(owner) = self.owner.take() {
            owner.borrow_mut().clear();
        }
        self.command = None;
    }
}

#[cfg(test)]
pub(crate) fn snapshot(class: Class) -> Snapshot {
    class.counters().snapshot()
}

#[cfg(test)]
mod tests;
