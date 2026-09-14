//! Cumulative observations at normal FIFO boundaries, never additional guest reads.
//! A stale ring snapshot is not an empty ring; its timestamp travels with it.
//! Model retirement is not proof of successful execution or guest publication.
//! Inline stamps record successful writes; queued stamps are requests, not landings.

use crate::model::{DeviceState, MAX_CHANNELS};
use crate::protocol::packets::Channel;
use crate::runtime::drain::{Arrival, Packet};
use reims_vgpu_core::transaction::{classify, PayloadClass};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Arrived,
    Admitted,
    Started,
    Deferred,
    ModelRetired,
    Refused,
}

impl Stage {
    const ALL: [Self; 6] = [
        Self::Arrived,
        Self::Admitted,
        Self::Started,
        Self::Deferred,
        Self::ModelRetired,
        Self::Refused,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Arrived => "arrived",
            Self::Admitted => "admitted",
            Self::Started => "started",
            Self::Deferred => "deferred",
            Self::ModelRetired => "model_retired",
            Self::Refused => "refused",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Count {
    packets: u64,
    presents: u64,
    last_present: Option<(u32, u64)>,
}

#[derive(Clone, Copy, Debug)]
struct RingSnapshot {
    head: u32,
    tail: u32,
    slot: u32,
    at_us: u64,
}

#[derive(Debug, Default)]
struct Fifo {
    ring: Option<RingSnapshot>,
    samples: u64,
    empty: u64,
    incomplete: u64,
    faults: u64,
    stages: [Count; 6],
}

#[derive(Debug, Default)]
struct Stamp {
    queued: u64,
    inline: u64,
    failed: u64,
    last_queued: Option<(u32, u64)>,
    last_inline: Option<(u32, u64)>,
}

pub(crate) enum StampWrite {
    Queued,
    Inline,
    Failed,
}

#[derive(Debug)]
pub(crate) struct FifoProgress {
    fifos: [Fifo; MAX_CHANNELS],
    stamps: BTreeMap<u32, Stamp>,
}

impl Default for FifoProgress {
    fn default() -> Self {
        Self {
            fifos: std::array::from_fn(|_| Fifo::default()),
            stamps: BTreeMap::new(),
        }
    }
}

impl FifoProgress {
    pub(crate) fn ring(
        &mut self,
        domain: u32,
        head: u32,
        tail: u32,
        slot: u32,
        arrival: &Arrival,
        at_us: u64,
    ) {
        let Some(fifo) = self.fifos.get_mut(domain as usize) else {
            return;
        };
        fifo.ring = Some(RingSnapshot {
            head,
            tail,
            slot,
            at_us,
        });
        fifo.samples += 1;
        match arrival {
            Arrival::Nothing if head == tail => fifo.empty += 1,
            Arrival::Nothing => fifo.incomplete += 1,
            Arrival::Fault(_) => fifo.faults += 1,
            Arrival::Packet(_) => {}
        }
    }

    pub(crate) fn packet(&mut self, domain: u32, packet: &Packet, stage: Stage, at_us: u64) {
        let Some(fifo) = self.fifos.get_mut(domain as usize) else {
            return;
        };
        let count = &mut fifo.stages[stage as usize];
        count.packets += 1;
        let channel = if domain == 0 {
            Channel::Root
        } else {
            Channel::Child
        };
        if classify(channel, packet.opcode) == Some(PayloadClass::Present) {
            count.presents += 1;
            count.last_present = Some((packet.completion_stamp, at_us));
        }
    }

    pub(crate) fn stamp(&mut self, slot: u32, value: u32, write: StampWrite, at_us: u64) {
        let stamp = self.stamps.entry(slot).or_default();
        match write {
            StampWrite::Queued => {
                stamp.queued += 1;
                stamp.last_queued = Some((value, at_us));
            }
            StampWrite::Inline => {
                stamp.inline += 1;
                stamp.last_inline = Some((value, at_us));
            }
            StampWrite::Failed => stamp.failed += 1,
        }
    }
}

fn word(fields: &mut String, label: &str, observation: Option<(u32, u64)>) {
    if let Some((value, at_us)) = observation {
        fields.push_str(&format!(" {label}={value} {label}_us={at_us}"));
    }
}

pub(crate) fn emit(state: &DeviceState) {
    let mono_before_us = crate::observe::elapsed_us();
    let utc = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
    let mono_after_us = crate::observe::elapsed_us();
    match utc {
        Ok(utc) => crate::observe::off(format!(
            "fifo_clock device={} mono_before_us={mono_before_us} utc_us={} \
             mono_after_us={mono_after_us}",
            state.id.0,
            utc.as_micros(),
        )),
        Err(error) => crate::observe::fail(format!("fifo_clock_unavailable error={error}")),
    }
    let mut waiting = [0usize; MAX_CHANNELS];
    let mut ready = [0usize; MAX_CHANNELS];
    for (positions, counts) in [
        (state.parked.waiting_in_order(), &mut waiting),
        (state.parked.ready_in_order(), &mut ready),
    ] {
        for position in positions {
            if let Some(domain) = state.parked.domain_of(position) {
                if let Some(count) = counts.get_mut(domain as usize) {
                    *count += 1;
                }
            }
        }
    }
    let generation = state.session_generation().get();
    for (domain, fifo) in state.fifo_progress.fifos.iter().enumerate() {
        if fifo.ring.is_none() && fifo.stages[Stage::Arrived as usize].packets == 0 {
            continue;
        }
        let mut fields = format!(
            "fifo_progress device={} generation={generation} ch={domain} cumulative=1 \
             samples={} empty={} incomplete={} faults={} waiting={} ready={}",
            state.id.0,
            fifo.samples,
            fifo.empty,
            fifo.incomplete,
            fifo.faults,
            waiting[domain],
            ready[domain],
        );
        if let Some(ring) = fifo.ring {
            fields.push_str(&format!(
                " head={} tail={} slot={} sample_us={}",
                ring.head, ring.tail, ring.slot, ring.at_us,
            ));
        }
        for stage in Stage::ALL {
            let count = fifo.stages[stage as usize];
            let label = stage.label();
            fields.push_str(&format!(
                " {label}={} present_{label}={}",
                count.packets, count.presents,
            ));
            word(
                &mut fields,
                &format!("present_{label}_stamp"),
                count.last_present,
            );
        }
        crate::observe::off(fields);
    }
    for (slot, stamp) in &state.fifo_progress.stamps {
        let mut fields = format!(
            "stamp_progress device={} generation={generation} slot={slot} cumulative=1 \
             queued={} inline={} failed={}",
            state.id.0, stamp.queued, stamp.inline, stamp.failed,
        );
        word(&mut fields, "last_queued", stamp.last_queued);
        word(&mut fields, "last_inline", stamp.last_inline);
        crate::observe::off(fields);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CHILD_OP_DISPLAY_SWAP;

    #[test]
    fn an_incomplete_ring_is_not_counted_as_empty() {
        let mut progress = FifoProgress::default();
        progress.ring(2, 8, 8, 4, &Arrival::Nothing, 10);
        progress.ring(2, 8, 12, 4, &Arrival::Nothing, 20);
        let fifo = &progress.fifos[2];
        assert_eq!((fifo.samples, fifo.empty, fifo.incomplete), (2, 1, 1));
        assert_eq!(fifo.ring.unwrap().at_us, 20);
    }

    #[test]
    fn admission_dispatch_and_completion_are_distinct_observations() {
        let mut progress = FifoProgress::default();
        let packet = Packet {
            opcode: CHILD_OP_DISPLAY_SWAP,
            stamp_waits: Vec::new(),
            total_size: 16,
            completion_stamp: 7,
            payload: Vec::new(),
            next_head: 16,
        };
        progress.packet(2, &packet, Stage::Arrived, 10);
        progress.packet(2, &packet, Stage::Admitted, 11);
        let fifo = &progress.fifos[2];
        assert_eq!(fifo.stages[Stage::Admitted as usize].presents, 1);
        assert_eq!(fifo.stages[Stage::Started as usize].presents, 0);
        assert_eq!(fifo.stages[Stage::ModelRetired as usize].presents, 0);
        progress.packet(2, &packet, Stage::Started, 20);
        progress.packet(2, &packet, Stage::Deferred, 21);
        assert_eq!(
            progress.fifos[2].stages[Stage::Deferred as usize].last_present,
            Some((7, 21))
        );
    }

    #[test]
    fn queued_or_failed_writes_do_not_claim_guest_visible_completion() {
        let mut progress = FifoProgress::default();
        progress.stamp(4, 6, StampWrite::Inline, 10);
        progress.stamp(4, 7, StampWrite::Queued, 20);
        progress.stamp(4, 8, StampWrite::Failed, 30);
        let stamp = &progress.stamps[&4];
        assert_eq!((stamp.inline, stamp.queued, stamp.failed), (1, 1, 1));
        assert_eq!(stamp.last_inline, Some((6, 10)));
        assert_eq!(stamp.last_queued, Some((7, 20)));
    }
}
