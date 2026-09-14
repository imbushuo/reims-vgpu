use super::*;

fn snapshot(pending: usize, parked: usize) -> Snapshot {
    Snapshot {
        pending: Some(pending),
        parked,
        ready: 0,
    }
}

#[test]
fn outer_and_parked_transactions_block_quiescence() {
    let mut tranche = Tranche::new(100);
    for (pending, parked) in [(1, 0), (1, 1), (3, 2)] {
        tranche.packet_started();
        tranche.completed(120, snapshot(pending, parked), Demand::Waiting);
    }
    let report = tranche.finish(200);
    assert_eq!(report.counts.completed, 3);
    assert_eq!(report.counts.outer_blocked, 2);
    assert_eq!(report.counts.parked_blocked, 2);
    assert_eq!(report.counts.post_pending_max, 3);
    assert_eq!(report.counts.post_parked_max, 2);
    assert_eq!(report.counts.post_outer_max, 1);
    assert_eq!(report.counts.quiescent, 0);
    assert_eq!(report.counts.eligible_further, 0);
}

#[test]
fn only_observed_live_demand_qualifies_further_work() {
    let mut tranche = Tranche::new(100);
    tranche.packet_started();
    for (now, demand) in [
        (110, Demand::Absent),
        (120, Demand::Contended),
        (130, Demand::Closed),
        (140, Demand::Waiting),
    ] {
        tranche.completed(now, snapshot(0, 0), demand);
        tranche.packet_started();
    }
    let report = tranche.finish(200);
    assert_eq!(report.counts.quiescent_further, 4);
    assert_eq!(report.counts.eligible_further, 1);
    assert_eq!(report.counts.demand_unknown, 1);
    assert_eq!(report.counts.demand_closed, 1);
    assert_eq!(report.first_eligible_age_us, Some(40));
    assert_eq!(report.remaining_wall_us, Some(60));
}

#[test]
fn terminal_checkpoint_is_not_a_resumable_opportunity() {
    let mut tranche = Tranche::new(100);
    tranche.packet_started();
    tranche.completed(110, snapshot(0, 0), Demand::Waiting);
    let report = tranche.finish(10_000);
    assert_eq!(report.counts.quiescent_demand, 1);
    assert_eq!(report.counts.quiescent_terminal, 1);
    assert_eq!(report.counts.demand_terminal, 1);
    assert_eq!(report.counts.eligible_further, 0);
    assert_eq!(report.first_eligible_age_us, None);
    assert_eq!(report.remaining_wall_us, None);
}

#[test]
fn first_eligible_age_and_remaining_wall_are_counted_once_per_tranche() {
    let mut first = Tranche::new(100);
    first.packet_started();
    first.completed(130, snapshot(0, 0), Demand::Waiting);
    first.packet_started();
    first.completed(150, snapshot(0, 0), Demand::Waiting);
    first.packet_started();
    first.completed(200, snapshot(0, 0), Demand::Waiting);
    let first = first.finish(220);
    assert_eq!(first.counts.eligible_further, 2);
    assert_eq!(first.first_eligible_age_us, Some(30));
    assert_eq!(first.remaining_wall_us, Some(90));
    assert_eq!(first.counts.demand_terminal, 1);

    let mut second = Tranche::new(300);
    second.packet_started();
    second.completed(305, snapshot(0, 0), Demand::Waiting);
    second.packet_started();
    let mut window = Window::new(
        Key {
            device: 1,
            epoch: 1,
        },
        100,
    );
    window.include(first);
    window.include(second.finish(340));
    assert_eq!(window.counts.eligible_further, 3);
    assert_eq!(window.eligible_tranches, 2);
    assert_eq!(window.first_eligible_age_sum_us, 35);
    assert_eq!(window.first_eligible_age_max_us, 30);
    assert_eq!(window.remaining_wall_sum_us, 125);
    assert_eq!(window.remaining_wall_max_us, 90);
    assert_eq!(window.longest.wall_us, 120);
    assert_eq!(window.longest.first_eligible_age_us, Some(30));
    assert_eq!(window.longest.remaining_wall_us, Some(90));
}

#[test]
fn zero_age_is_a_real_eligible_checkpoint() {
    let mut tranche = Tranche::new(100);
    tranche.packet_started();
    tranche.completed(100, snapshot(0, 0), Demand::Waiting);
    tranche.packet_started();
    let report = tranche.finish(120);
    assert_eq!(report.first_eligible_age_us, Some(0));
    assert_eq!(report.remaining_wall_us, Some(20));
}

#[test]
fn unknown_or_inconsistent_snapshots_never_claim_quiescence() {
    let mut tranche = Tranche::new(0);
    for state in [
        Snapshot {
            pending: None,
            parked: 0,
            ready: 0,
        },
        snapshot(0, 1),
        Snapshot {
            pending: Some(0),
            parked: 0,
            ready: 1,
        },
    ] {
        tranche.packet_started();
        tranche.completed(10, state, Demand::Waiting);
    }
    let report = tranche.finish(20);
    assert_eq!(report.counts.pending_unknown, 1);
    assert_eq!(report.counts.inconsistent, 2);
    assert_eq!(report.counts.quiescent, 0);
    assert_eq!(report.counts.eligible_further, 0);
}

struct FixedDemand;

impl DemandSource for FixedDemand {
    fn checkpoint_demand(&self, _: u64) -> Demand {
        Demand::Waiting
    }
}

fn record_pair(now: u64, demand: Demand, further: bool) {
    CONTEXT.with(|context| {
        let mut context = context.borrow_mut();
        let tranche = &mut context.active.as_mut().unwrap().tranche;
        tranche.packet_started();
        tranche.completed(now, snapshot(0, 0), demand);
        if further {
            tranche.packet_started();
        }
    });
}

#[test]
fn reset_between_tranches_discards_old_epoch_aggregates() {
    let scope = Scope::start(71, 1, Arc::new(FixedDemand), 0);
    record_pair(10, Demand::Waiting, true);
    assert!(scope.finish(100).is_none());

    let scope = Scope::start(71, 2, Arc::new(FixedDemand), 200);
    record_pair(250, Demand::Waiting, false);
    let report = scope.finish(1_000_200).unwrap();
    assert_eq!(report.window.key.epoch, 2);
    assert_eq!(report.window.tranches, 1);
    assert_eq!(report.window.counts.completed, 1);
    assert_eq!(report.window.counts.demand_terminal, 1);
    assert_eq!(report.window.eligible_tranches, 0);
    assert_eq!(report.window.remaining_wall_sum_us, 0);
    assert!(report.line().contains("admission_epoch=2"));
}

#[test]
fn nested_scope_cannot_replace_an_outer_observation() {
    let outer = Scope::start(81, 1, Arc::new(FixedDemand), 0);
    let inner = Scope::start(82, 1, Arc::new(FixedDemand), 5);
    assert!(!inner.active);
    assert!(inner.finish(20).is_none());
    record_pair(30, Demand::Waiting, true);
    let report = outer.finish(1_000_000).unwrap();
    assert_eq!(report.window.key.device, 81);
    assert_eq!(report.window.tranches, 1);
    assert_eq!(report.window.counts.completed, 1);
}

#[test]
fn reentrant_observation_does_not_borrow_or_replace_the_active_scope() {
    struct ReentrantDemand {
        state: DeviceState,
    }
    impl DemandSource for ReentrantDemand {
        fn checkpoint_demand(&self, _: u64) -> Demand {
            let nested = Scope::start(112, 1, Arc::new(FixedDemand), 0);
            assert!(!nested.active);
            packet_started(&self.state);
            completed(&self.state);
            Demand::Waiting
        }
    }
    let state = DeviceState::new(crate::model::DeviceId(111), crate::model::PAGE_SHIFT_X86);
    let source = Arc::new(ReentrantDemand {
        state: DeviceState::new(crate::model::DeviceId(112), crate::model::PAGE_SHIFT_X86),
    });
    let start = crate::observe::elapsed_us();
    let scope = Scope::start(111, 1, source, start);
    packet_started(&state);
    completed(&state);
    packet_started(&state);
    let report = scope
        .finish(crate::observe::elapsed_us().max(start + 1_000_000))
        .unwrap();
    assert_eq!(report.window.counts.started, 2);
    assert_eq!(report.window.counts.completed, 1);
    assert_eq!(report.window.counts.eligible_further, 1);
}

#[test]
fn unwind_discards_incomplete_tranche_candidates() {
    let result = std::panic::catch_unwind(|| {
        let _scope = Scope::start(91, 1, Arc::new(FixedDemand), 0);
        record_pair(10, Demand::Waiting, true);
        panic!("unfinished observation");
    });
    assert!(result.is_err());
    let scope = Scope::start(91, 1, Arc::new(FixedDemand), 100);
    record_pair(120, Demand::Absent, false);
    let report = scope.finish(1_000_000).unwrap();
    assert_eq!(report.window.tranches, 1);
    assert_eq!(report.window.counts.completed, 1);
    assert_eq!(report.window.eligible_tranches, 0);
}

#[test]
fn post_completion_maxima_are_not_summed_across_tranches() {
    let mut window = Window::new(
        Key {
            device: 1,
            epoch: 1,
        },
        0,
    );
    for count in [3, 2] {
        let mut tranche = Tranche::new(0);
        tranche.packet_started();
        tranche.completed(10, snapshot(count, count), Demand::Absent);
        window.include(tranche.finish(20));
    }
    assert_eq!(window.counts.post_pending_max, 3);
    assert_eq!(window.counts.post_parked_max, 3);
    assert_eq!(window.counts.parked_blocked, 2);
}

#[test]
fn drain_observation_follows_publication_without_replay_or_scheduling() {
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::drain;
    use crate::runtime::host::{
        FakeHost, HostAction, HostActionKind, HostMemory, HostOps, MemError,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    struct PublishingHost {
        inner: FakeHost,
        stamp_gpa: u64,
        publications: Arc<AtomicU64>,
    }
    impl HostMemory for PublishingHost {
        fn read_gpa(&self, gpa: u64, bytes: &mut [u8]) -> Result<(), MemError> {
            self.inner.read_gpa(gpa, bytes)
        }
        fn write_gpa(&mut self, gpa: u64, bytes: &[u8]) -> Result<(), MemError> {
            self.inner.write_gpa(gpa, bytes)
        }
    }
    impl HostOps for PublishingHost {
        fn mono_ns(&self) -> u64 {
            self.inner.mono_ns()
        }
        fn enqueue(&mut self, action: HostAction) {
            if action.kind == HostActionKind::IrqGfxPulse {
                let next = self.publications.load(Ordering::Relaxed) + 1;
                assert_eq!(u64::from(self.inner.get_u32(self.stamp_gpa)), 100 + next);
                self.publications.fetch_add(1, Ordering::Relaxed);
            }
            self.inner.enqueue(action);
        }
        fn schedule_bh(&mut self) {
            self.inner.schedule_bh();
        }
        fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
            self.inner.map_pages(gpas, page_size)
        }
        fn unmap_pages(&mut self, ptr: usize, len: usize) {
            self.inner.unmap_pages(ptr, len);
        }
    }
    struct PublishedDemand {
        publications: Arc<AtomicU64>,
        observations: AtomicU64,
    }
    impl DemandSource for PublishedDemand {
        fn checkpoint_demand(&self, _: u64) -> Demand {
            let observation = self.observations.fetch_add(1, Ordering::Relaxed) + 1;
            assert_eq!(
                self.publications.load(Ordering::Relaxed),
                observation,
                "the complete publication must precede its checkpoint"
            );
            Demand::Waiting
        }
    }

    let mut state = DeviceState::new(DeviceId(101), PAGE_SHIFT_X86);
    let page_size = state.page_size();
    state.gfx.fifo_base_page = 0x40;
    state.gfx.fifo_start = page_size as u32;
    state.gfx.fifo_length = (page_size * 3) as u32;
    let base = state.pfn_gpa(state.gfx.fifo_base_page);
    let publications = Arc::new(AtomicU64::new(0));
    let mut host = PublishingHost {
        inner: FakeHost::new(),
        stamp_gpa: base + drain::stamp_slot_offset(0, page_size).unwrap(),
        publications: Arc::clone(&publications),
    };
    host.inner.map_range(base, (page_size * 3) as usize, 0);
    let mut packets = Vec::new();
    for (channel, stamp) in [(1u32, 101u32), (2, 102)] {
        packets.extend_from_slice(&drain::ROOT_OP_DEFINE_FIFO.to_le_bytes());
        packets.extend_from_slice(&0u16.to_le_bytes());
        packets.extend_from_slice(&(drain::PACKET_HEADER_LEN + 4).to_le_bytes());
        packets.extend_from_slice(&stamp.to_le_bytes());
        packets.extend_from_slice(&channel.to_le_bytes());
    }
    host.write_gpa(base + page_size, &packets).unwrap();
    state.gfx.fifo_written = packets.len() as u32;
    let source = Arc::new(PublishedDemand {
        publications: Arc::clone(&publications),
        observations: AtomicU64::new(0),
    });
    let start = crate::observe::elapsed_us();
    let scope = Scope::start(101, 1, source.clone(), start);
    drain::drain_main_fifo(&mut state, &mut host);
    drain::drain_main_fifo(&mut state, &mut host);
    let report = scope
        .finish(crate::observe::elapsed_us().max(start + 1_000_000))
        .unwrap();

    assert_eq!(
        state.gfx.fifo_read.load(Ordering::Acquire),
        packets.len() as u32
    );
    assert_eq!(state.pending_transactions_for_observation(), Some(0));
    assert!(state.parked.is_empty());
    assert_eq!(publications.load(Ordering::Relaxed), 2);
    assert_eq!(source.observations.load(Ordering::Relaxed), 2);
    assert_eq!(
        host.inner
            .actions
            .iter()
            .filter(|a| a.kind == HostActionKind::IrqGfxPulse)
            .count(),
        2
    );
    assert!(
        !host.inner.bh_scheduled,
        "observation does not rearm the worker"
    );
    assert_eq!(report.window.counts.started, 2);
    assert_eq!(report.window.counts.completed, 2);
    assert_eq!(report.window.counts.eligible_further, 1);
    assert_eq!(report.window.counts.quiescent_terminal, 1);
}
