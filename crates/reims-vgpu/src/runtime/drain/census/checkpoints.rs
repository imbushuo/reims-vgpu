//! Observed model-quiescent prefixes, never a scheduling decision.
//!
//! A checkpoint has further work only once another packet actually starts in
//! the same tranche. This is retrospective wall-time attribution: it neither
//! proves that the later packet was published at the checkpoint nor promises
//! that yielding there would save the remaining wall time.
//! Eligibility describes model/admission metadata, not GPU-idle certification.
//! All per-packet storage is inline; only the once-per-window log is formatted.

use crate::model::DeviceState;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Demand {
    Absent,
    Waiting,
    Contended,
    Closed,
}

pub(crate) trait DemandSource {
    /// A failed nonblocking snapshot is unknown, not an empty admission queue.
    fn checkpoint_demand(&self, epoch: u64) -> Demand;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    device: u64,
    epoch: u64,
}

#[derive(Clone, Copy)]
struct Snapshot {
    pending: Option<usize>,
    parked: usize,
    ready: usize,
}

/// Blocking causes may overlap. Maxima sample post-completion state only.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    started: u64,
    completed: u64,
    outer_blocked: u64,
    parked_blocked: u64,
    pending_unknown: u64,
    inconsistent: u64,
    demand_waiting: u64,
    demand_absent: u64,
    demand_unknown: u64,
    demand_closed: u64,
    quiescent: u64,
    quiescent_demand: u64,
    quiescent_further: u64,
    eligible_further: u64,
    quiescent_terminal: u64,
    demand_terminal: u64,
    post_pending_max: usize,
    post_parked_max: usize,
    post_ready_max: usize,
    post_outer_max: usize,
}

impl Counts {
    fn include(&mut self, other: Self) {
        macro_rules! sum {
            ($($field:ident),+ $(,)?) => {
                $(self.$field += other.$field;)+
            };
        }
        sum!(
            started,
            completed,
            outer_blocked,
            parked_blocked,
            pending_unknown,
            inconsistent,
            demand_waiting,
            demand_absent,
            demand_unknown,
            demand_closed,
            quiescent,
            quiescent_demand,
            quiescent_further,
            eligible_further,
            quiescent_terminal,
            demand_terminal,
        );
        self.post_pending_max = self.post_pending_max.max(other.post_pending_max);
        self.post_parked_max = self.post_parked_max.max(other.post_parked_max);
        self.post_ready_max = self.post_ready_max.max(other.post_ready_max);
        self.post_outer_max = self.post_outer_max.max(other.post_outer_max);
    }
}

struct Candidate {
    age_us: u64,
    demand: bool,
}

struct Tranche {
    started_us: u64,
    counts: Counts,
    candidate: Option<Candidate>,
    first_eligible_age_us: Option<u64>,
}

impl Tranche {
    fn new(started_us: u64) -> Self {
        Self {
            started_us,
            counts: Counts::default(),
            candidate: None,
            first_eligible_age_us: None,
        }
    }

    fn packet_started(&mut self) {
        self.counts.started += 1;
        if let Some(candidate) = self.candidate.take() {
            self.counts.quiescent_further += 1;
            if candidate.demand {
                self.counts.eligible_further += 1;
                self.first_eligible_age_us.get_or_insert(candidate.age_us);
            }
        }
    }

    fn completed(&mut self, now_us: u64, snapshot: Snapshot, demand: Demand) {
        let c = &mut self.counts;
        c.completed += 1;
        match demand {
            Demand::Absent => c.demand_absent += 1,
            Demand::Waiting => c.demand_waiting += 1,
            Demand::Contended => c.demand_unknown += 1,
            Demand::Closed => c.demand_closed += 1,
        }
        c.post_parked_max = c.post_parked_max.max(snapshot.parked);
        c.post_ready_max = c.post_ready_max.max(snapshot.ready);
        c.parked_blocked += u64::from(snapshot.parked != 0);
        let Some(pending) = snapshot.pending else {
            c.pending_unknown += 1;
            return;
        };
        c.post_pending_max = c.post_pending_max.max(pending);
        // Released work leaves ParkedStore before it runs, but remains pending
        // in the scheduler until completion. After this packet's publication,
        // their difference is the unfinished outer execution, not another queue.
        let Some(outer) = pending.checked_sub(snapshot.parked) else {
            c.inconsistent += 1;
            return;
        };
        if snapshot.ready > snapshot.parked {
            c.inconsistent += 1;
            return;
        }
        c.post_outer_max = c.post_outer_max.max(outer);
        c.outer_blocked += u64::from(outer != 0);
        if pending == 0 {
            c.quiescent += 1;
            c.quiescent_demand += u64::from(demand == Demand::Waiting);
            self.candidate = Some(Candidate {
                age_us: now_us.saturating_sub(self.started_us),
                demand: demand == Demand::Waiting,
            });
        }
    }

    fn finish(mut self, ended_us: u64) -> CompletedTranche {
        if let Some(candidate) = self.candidate.take() {
            self.counts.quiescent_terminal += 1;
            self.counts.demand_terminal += u64::from(candidate.demand);
        }
        let wall_us = ended_us.saturating_sub(self.started_us);
        CompletedTranche {
            started_us: self.started_us,
            ended_us,
            wall_us,
            counts: self.counts,
            first_eligible_age_us: self.first_eligible_age_us,
            remaining_wall_us: self
                .first_eligible_age_us
                .map(|age| wall_us.saturating_sub(age)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompletedTranche {
    started_us: u64,
    ended_us: u64,
    wall_us: u64,
    counts: Counts,
    first_eligible_age_us: Option<u64>,
    remaining_wall_us: Option<u64>,
}

struct Window {
    key: Key,
    started_us: u64,
    tranches: u64,
    counts: Counts,
    wall_sum_us: u64,
    eligible_tranches: u64,
    first_eligible_age_sum_us: u64,
    first_eligible_age_max_us: u64,
    remaining_wall_sum_us: u64,
    remaining_wall_max_us: u64,
    longest: CompletedTranche,
}

impl Window {
    fn new(key: Key, started_us: u64) -> Self {
        Self {
            key,
            started_us,
            tranches: 0,
            counts: Counts::default(),
            wall_sum_us: 0,
            eligible_tranches: 0,
            first_eligible_age_sum_us: 0,
            first_eligible_age_max_us: 0,
            remaining_wall_sum_us: 0,
            remaining_wall_max_us: 0,
            longest: CompletedTranche::default(),
        }
    }

    fn include(&mut self, tranche: CompletedTranche) {
        self.tranches += 1;
        self.counts.include(tranche.counts);
        self.wall_sum_us += tranche.wall_us;
        if let Some(age) = tranche.first_eligible_age_us {
            self.eligible_tranches += 1;
            self.first_eligible_age_sum_us += age;
            self.first_eligible_age_max_us = self.first_eligible_age_max_us.max(age);
        }
        if let Some(remaining) = tranche.remaining_wall_us {
            self.remaining_wall_sum_us += remaining;
            self.remaining_wall_max_us = self.remaining_wall_max_us.max(remaining);
        }
        if self.tranches == 1 || tranche.wall_us > self.longest.wall_us {
            self.longest = tranche;
        }
    }
}

pub(crate) struct Report {
    window: Window,
    ended_us: u64,
}

impl Report {
    pub fn line(&self) -> String {
        let w = &self.window;
        let c = &w.counts;
        let longest = &w.longest;
        format!(
            "drain_checkpoints device={} admission_epoch={} win_us={} tranches={} \
             wall_sum_us={} packets_started={} completed={} outer_blocked={} parked_blocked={} \
             pending_unknown={} inconsistent={} iosfc_waiting={} iosfc_absent={} \
             iosfc_unknown={} iosfc_closed={} post_pending_max={} post_parked_max={} \
             post_ready_max={} post_outer_max={} quiescent={} quiescent_demand={} \
             quiescent_further={} eligible_with_demand={} quiescent_terminal={} demand_terminal={} \
             eligible_tranches={} first_eligible_age_sum_us={} first_eligible_age_max_us={} \
             remaining_wall_sum_us={} remaining_wall_max_us={} max_tranche_wall_us={} \
             max_tranche_start_us={} max_tranche_end_us={} max_tranche_eligible={} \
             max_tranche_first_eligible_age_us={} max_tranche_remaining_wall_us={} \
             max_tranche_outer_blocked={} max_tranche_parked_blocked={}",
            w.key.device,
            w.key.epoch,
            self.ended_us.saturating_sub(w.started_us),
            w.tranches,
            w.wall_sum_us,
            c.started,
            c.completed,
            c.outer_blocked,
            c.parked_blocked,
            c.pending_unknown,
            c.inconsistent,
            c.demand_waiting,
            c.demand_absent,
            c.demand_unknown,
            c.demand_closed,
            c.post_pending_max,
            c.post_parked_max,
            c.post_ready_max,
            c.post_outer_max,
            c.quiescent,
            c.quiescent_demand,
            c.quiescent_further,
            c.eligible_further,
            c.quiescent_terminal,
            c.demand_terminal,
            w.eligible_tranches,
            w.first_eligible_age_sum_us,
            w.first_eligible_age_max_us,
            w.remaining_wall_sum_us,
            w.remaining_wall_max_us,
            longest.wall_us,
            longest.started_us,
            longest.ended_us,
            longest.counts.eligible_further,
            longest.first_eligible_age_us.unwrap_or(0),
            longest.remaining_wall_us.unwrap_or(0),
            longest.counts.outer_blocked,
            longest.counts.parked_blocked,
        )
    }
}

struct Active {
    key: Key,
    source: Arc<dyn DemandSource>,
    tranche: Tranche,
}

#[derive(Default)]
struct Context {
    active: Option<Active>,
    window: Option<Window>,
}

thread_local! {
    static CONTEXT: RefCell<Context> = const {
        RefCell::new(Context { active: None, window: None })
    };
}

/// Owns observation of one worker invocation, not device/backend lifetime.
/// Nested scopes cannot replace it; unwinding discards incomplete observations.
pub(crate) struct Scope {
    active: bool,
    _thread: PhantomData<*mut ()>,
}

impl Scope {
    pub fn start(device: u64, epoch: u64, source: Arc<dyn DemandSource>, started_us: u64) -> Self {
        let active = CONTEXT.with(|context| {
            let Ok(mut context) = context.try_borrow_mut() else {
                return false;
            };
            if context.active.is_some() {
                return false;
            }
            let key = Key { device, epoch };
            if context
                .window
                .as_ref()
                .is_none_or(|window| window.key != key)
            {
                context.window = Some(Window::new(key, started_us));
            }
            context.active = Some(Active {
                key,
                source,
                tranche: Tranche::new(started_us),
            });
            true
        });
        Self {
            active,
            _thread: PhantomData,
        }
    }

    pub fn finish(mut self, ended_us: u64) -> Option<Report> {
        if !self.active {
            return None;
        }
        self.active = false;
        CONTEXT.with(|context| {
            let mut context = context.borrow_mut();
            let active = context.active.take()?;
            let window = context.window.as_mut()?;
            window.include(active.tranche.finish(ended_us));
            if ended_us.saturating_sub(window.started_us) < super::DRAIN_DUTY_REPORT_MS * 1000 {
                return None;
            }
            let key = window.key;
            Some(Report {
                window: std::mem::replace(window, Window::new(key, ended_us)),
                ended_us,
            })
        })
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.active {
            let _ = CONTEXT.try_with(|context| {
                if let Ok(mut context) = context.try_borrow_mut() {
                    context.active = None;
                }
            });
        }
    }
}

pub(crate) fn packet_started(state: &DeviceState) {
    CONTEXT.with(|context| {
        let Ok(mut context) = context.try_borrow_mut() else {
            return;
        };
        if let Some(active) = context
            .active
            .as_mut()
            .filter(|a| a.key.device == state.id.0)
        {
            active.tranche.packet_started();
        }
    });
}

pub(crate) fn completed(state: &DeviceState) {
    CONTEXT.with(|context| {
        let Ok(mut context) = context.try_borrow_mut() else {
            return;
        };
        let Some(active) = context
            .active
            .as_mut()
            .filter(|a| a.key.device == state.id.0)
        else {
            return;
        };
        let snapshot = Snapshot {
            pending: state.pending_transactions_for_observation(),
            parked: state.parked.len(),
            ready: state.parked.ready_len(),
        };
        let demand = active.source.checkpoint_demand(active.key.epoch);
        active
            .tranche
            .completed(crate::observe::elapsed_us(), snapshot, demand);
    });
}

#[cfg(test)]
mod tests;
