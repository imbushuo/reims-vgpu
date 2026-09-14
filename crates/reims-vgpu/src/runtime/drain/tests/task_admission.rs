use super::*;
use crate::protocol::endian::st64;
use crate::protocol::fifo::*;
use crate::protocol::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use crate::protocol::segment::{SegmentKind, SEGMENT_HEADER_LEN};
use crate::runtime::host::MemError;
use std::cell::Cell;

struct CaptureBeforeHead {
    inner: FakeHost,
    head: u64,
    stream: u64,
    captured: Cell<bool>,
}

impl HostMemory for CaptureBeforeHead {
    fn read_gpa(&self, gpa: u64, bytes: &mut [u8]) -> Result<(), MemError> {
        if gpa == self.stream {
            assert_eq!(
                self.inner.get_u32(self.head),
                0,
                "stream captured after ring release"
            );
            self.captured.set(true);
        }
        self.inner.read_gpa(gpa, bytes)
    }

    fn write_gpa(&mut self, gpa: u64, bytes: &[u8]) -> Result<(), MemError> {
        self.inner.write_gpa(gpa, bytes)
    }
}

impl HostOps for CaptureBeforeHead {
    fn mono_ns(&self) -> u64 {
        self.inner.mono_ns()
    }

    fn schedule_bh(&mut self) {
        self.inner.schedule_bh();
    }

    fn enqueue(&mut self, action: HostAction) {
        self.inner.enqueue(action);
    }

    fn map_pages(&mut self, pages: &[u64], page_size: usize) -> Option<usize> {
        self.inner.map_pages(pages, page_size)
    }

    fn unmap_pages(&mut self, pointer: usize, len: usize) {
        self.inner.unmap_pages(pointer, len);
    }
}

fn child(
    state: &DeviceState,
    host: &mut FakeHost,
    channel: u32,
    ring: u32,
    list: u32,
    packet: &[u8],
) -> u64 {
    let registers = state.pfn_gpa(state.gfx.root_page) + child_reg_block_offset(channel).unwrap();
    host.write_gpa(state.pfn_gpa(list), &ring.to_le_bytes())
        .unwrap();
    host.write_gpa(registers + CHILD_REG_BASE_PFN, &list.to_le_bytes())
        .unwrap();
    host.write_gpa(registers + CHILD_REG_STAMP_INDEX, &channel.to_le_bytes())
        .unwrap();
    host.write_gpa(
        registers + CHILD_REG_TAIL,
        &(packet.len() as u32).to_le_bytes(),
    )
    .unwrap();
    host.write_gpa(state.pfn_gpa(ring), packet).unwrap();
    registers
}

#[test]
fn first_exec_captures_its_task_after_the_sibling_definition() {
    const TASK: u32 = 17;
    for shift in [PAGE_SHIFT_X86, PAGE_SHIFT_ARM64E] {
        for control in [2, 3, 4] {
            let capture = crate::observe::FailCapture::start();
            let mut state = DeviceState::new(DeviceId(1), shift);
            let mut host = FakeHost::new();
            state.gfx.fifo_base_page = 0x40;
            state.gfx.root_page = 0x50;
            state.open_child_domains_for_test((1 << 1) | (1 << control));
            for pfn in [0x40, 0x50, 0x60, 0x61, 0x70, 0x71, 0x80, 0x81, 0x82] {
                host.map_range(state.pfn_gpa(pfn), state.page_size() as usize, 0);
            }
            let mut directory = [0; 8];
            st32(&mut directory[DIRECTORY_ROOT_PFN as usize..], 0x81);
            st32(&mut directory[DIRECTORY_DEPTH as usize..], 1);
            host.write_gpa(state.pfn_gpa(0x80), &directory).unwrap();
            host.write_gpa(state.pfn_gpa(0x81), &0x82u32.to_le_bytes())
                .unwrap();

            let mut stream = vec![0; SEGMENT_HEADER_LEN];
            st32(&mut stream, SEGMENT_HEADER_LEN as u32);
            stream[4] = SegmentKind::Render.wire_type();
            host.write_gpa(state.pfn_gpa(0x82) + 0x100, &stream)
                .unwrap();
            let mut exec = vec![
                0;
                (CHILD_EXEC_INDIRECT_HEADER_LEN + CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN)
                    as usize
            ];
            st32(&mut exec[CHILD_EXEC_INDIRECT_TASK_ID as usize..], TASK);
            st32(&mut exec[CHILD_EXEC_INDIRECT_CMDBUF_COUNT as usize..], 1);
            let descriptor = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
            st64(
                &mut exec[descriptor + CHILD_EXEC_INDIRECT_CMDBUF_GVA as usize..],
                0x100,
            );
            st64(
                &mut exec[descriptor + CHILD_EXEC_INDIRECT_CMDBUF_LENGTH as usize..],
                stream.len() as u64,
            );
            let exec_packet = packet_bytes(CHILD_OP_EXEC_INDIRECT2, 101, &exec);
            let exec_registers = child(&state, &mut host, 1, 0x60, 0x70, &exec_packet);

            let mut definition = vec![0; DEFINE_TASK_LEN];
            st32(&mut definition, TASK << DEFINE_TASK_ID_SHIFT);
            st64(&mut definition[DEFINE_TASK_LENGTH..], state.page_size());
            st32(&mut definition[DEFINE_TASK_DIRECTORY_PFN..], 0x80);
            child(
                &state,
                &mut host,
                control,
                0x61,
                0x71,
                &packet_bytes(CHILD_OP_DEFINE_TASK2, 303, &definition),
            );
            assert!(!state.tasks.is_active(TASK));
            let mut host = CaptureBeforeHead {
                inner: host,
                head: exec_registers + CHILD_REG_HEAD,
                stream: state.pfn_gpa(0x82) + 0x100,
                captured: Cell::new(false),
            };
            let streams = store_route_count("exec_streams_loaded");
            drain_child_fifo(&mut state, &mut host, 1);
            assert!(
                state.tasks.is_active(TASK),
                "definition must precede GVA capture"
            );
            assert!(host.captured.get());
            assert_eq!(store_route_count("exec_streams_loaded") - streams, 1);
            assert_eq!(
                host.inner.get_u32(exec_registers + CHILD_REG_HEAD),
                exec_packet.len() as u32
            );
            let stamp = state.pfn_gpa(state.gfx.fifo_base_page)
                + stamp_slot_offset(1, state.page_size()).unwrap();
            assert_eq!(host.inner.get_u32(stamp), 101);
            assert!(
                !capture.lines().iter().any(|line| {
                    line.contains("cmd_task_dead") || line.contains("packet_unadmitted")
                }),
                "{:?}",
                capture.lines()
            );
            assert!(state.parked.is_empty());
        }
    }
}

#[test]
fn unresolved_task_does_not_invent_a_context_and_a_yield_keeps_the_packet_owned() {
    let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    let fifo = crate::runtime::ingress::Fifo::child(1).unwrap();
    let mut payload = vec![0; CHILD_EXEC_INDIRECT_HEADER_LEN as usize];
    st32(&mut payload[CHILD_EXEC_INDIRECT_TASK_ID as usize..], 17);
    let packet = Packet {
        opcode: CHILD_OP_EXEC_INDIRECT2,
        total_size: PACKET_HEADER_LEN + payload.len() as u32,
        payload,
        stamp_waits: Vec::new(),
        completion_stamp: 1,
        next_head: 32,
    };
    let absent = child_arrival_work(&mut state, &mut host, fifo, 1, &packet).unwrap();
    assert!(absent.submission.is_none());
    assert!(!state.tasks.is_active(17));
    state.pending.host_action_yield = true;
    assert!(child_arrival_work(&mut state, &mut host, fifo, 1, &packet).is_none());
    assert_ne!(state.pending.child_mask & (1 << 1), 0);
    assert!(!state.tasks.is_active(17));
}
