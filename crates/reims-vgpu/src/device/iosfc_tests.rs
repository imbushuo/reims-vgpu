use super::*;
use crate::model::*;
use crate::protocol::iosurface_pages as wire;
use crate::runtime::host::{FakeHost, HostMemory};
use std::ffi::c_void;

#[test]
fn authoritative_reads_do_not_take_render_state_and_preserve_widths() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).unwrap();
    let slot = device_slot(id).unwrap();
    let held = slot.inner.lock();
    slot.iosfc_regs.set_ring_base(0x1234_5678_9abc_def0);
    slot.iosfc_regs.set_capacity(0x1234_5678);
    slot.iosfc_regs.set_desc_table(0x2345_6789_abcd_ef01);
    slot.iosfc_regs.set_producer(0x3456_789a);
    slot.iosfc_regs.set_consumer(0x4567_89ab);
    for (offset, value) in [
        (IOSFC_REG_RING_BASE, 0x1234_5678_9abc_def0),
        (IOSFC_REG_CAPACITY, 0x1234_5678),
        (IOSFC_REG_DESC_TABLE, 0x2345_6789_abcd_ef01),
        (IOSFC_REG_PRODUCER, 0x3456_789a),
        (IOSFC_REG_CONSUMER, 0x4567_89ab),
        (0x1040, 0),
    ] {
        for size in [0, 1, 2, 3, 4, 8, 9] {
            let expected = if size > 0 && size < 8 {
                value & ((1u64 << (size * 8)) - 1)
            } else {
                value
            };
            assert_eq!(device_iosfc_read(id, offset, size), Some(expected));
        }
    }
    held.device.state.iosfc.set_consumer(19);
    assert_eq!(device_iosfc_read(id, IOSFC_REG_CONSUMER, 4), Some(19));
    drop(held);
    assert!(device_reset(id));
    assert!(Arc::ptr_eq(
        &slot.iosfc_regs,
        &slot.inner.lock().device.state.iosfc
    ));
    assert_eq!(device_iosfc_read(id, IOSFC_REG_CONSUMER, 8), Some(0));
    assert!(device_destroy(id));
}

struct CaptureHost {
    memory: Mutex<FakeHost>,
    calls: AtomicU64,
    wakes: AtomicU64,
    device: AtomicU64,
    reentry_checked: AtomicBool,
    alive: AtomicBool,
    after_teardown: AtomicBool,
}

impl CaptureHost {
    fn note_access(&self) {
        if !self.alive.load(Ordering::Acquire) {
            self.after_teardown.store(true, Ordering::Release);
        }
    }
}

unsafe extern "C" fn read_memory(ctx: *mut c_void, address: u64, out: *mut u8, len: usize) -> i32 {
    let host = unsafe { &*(ctx.cast::<CaptureHost>()) };
    host.note_access();
    host.calls.fetch_add(1, Ordering::Relaxed);
    let bytes = unsafe { std::slice::from_raw_parts_mut(out, len) };
    if host.memory.lock().read_gpa(address, bytes).is_ok() {
        0
    } else {
        -1
    }
}

unsafe extern "C" fn read_kva(ctx: *mut c_void, address: u64, out: *mut u8, len: usize) -> i32 {
    let host = unsafe { &*(ctx.cast::<CaptureHost>()) };
    host.note_access();
    let id = host.device.load(Ordering::Acquire);
    let rejected = device_iosfc_begin(id, IOSFC_REG_CAPACITY, 99, 4)
        == Err(IosfcAdmissionStatus::Reentrant)
        && device_iosfc_read(id, IOSFC_REG_CONSUMER, 4).is_none()
        && device_gfx_read(id, GFX_REG_VERSION, 4).is_none()
        && !device_gfx_write(id, GFX_REG_ROOT_PAGE, 99, 4);
    host.reentry_checked.store(rejected, Ordering::Release);
    unsafe { read_memory(ctx, address, out, len) }
}

unsafe extern "C" fn read_xreg(ctx: *mut c_void, index: u32, out: *mut u64) -> i32 {
    let host = unsafe { &*(ctx.cast::<CaptureHost>()) };
    host.note_access();
    host.calls.fetch_add(1, Ordering::Relaxed);
    match host.memory.lock().read_xreg(index) {
        Ok(value) => {
            unsafe { *out = value };
            0
        }
        Err(_) => -1,
    }
}

unsafe extern "C" fn ram_gpa(ctx: *mut c_void, address: u64) -> i32 {
    let host = unsafe { &*(ctx.cast::<CaptureHost>()) };
    host.note_access();
    host.calls.fetch_add(1, Ordering::Relaxed);
    i32::from(host.memory.lock().is_ram_gpa(address))
}

unsafe extern "C" fn wake(ctx: *mut c_void) {
    let host = unsafe { &*(ctx.cast::<CaptureHost>()) };
    host.note_access();
    host.wakes.fetch_add(1, Ordering::Relaxed);
}

fn capture_device() -> (u64, Box<CaptureHost>, u64) {
    let internal = 0xffff_fe00_1000_0000;
    let mapper = internal + 0x1000;
    let descriptor = internal + 0x4000;
    let ring = 0x7000_0000;
    let table = 0x40000;
    let page = 0x80000;
    let mut memory = FakeHost::new();
    let mut put = |address, bytes: &[u8]| {
        memory.map_range(address, bytes.len(), 0);
        memory.write_gpa(address, bytes).unwrap();
    };
    let mut request = [0u8; wire::MAPPER_REQUEST_ENTRY_LEN];
    request[..4].copy_from_slice(&wire::MAPPER_REQUEST_MAP.to_le_bytes());
    request[4..8].copy_from_slice(&7u32.to_le_bytes());
    put(ring, &request);
    put(
        internal + wire::MAPPING_INTERNAL_BACKPTR,
        &mapper.to_le_bytes(),
    );
    put(internal + wire::MAPPING_INTERNAL_ID, &7u32.to_le_bytes());
    put(
        internal + wire::MAPPING_INTERNAL_SIZE,
        &wire::MAPPING_INTERNAL_EXPECTED_SIZE.to_le_bytes(),
    );
    put(
        internal + wire::MAPPING_INTERNAL_DESC_PTR,
        &descriptor.to_le_bytes(),
    );
    let entry = |gpa: u64| {
        ((gpa >> PAGE_SHIFT_ARM64E) as u32) << wire::PAGE_ENTRY_PFN_SHIFT | wire::PAGE_ENTRY_VALID
    };
    let mut desc = [0u8; wire::DEVICE_DESC_LEN];
    desc[wire::DEVICE_DESC_PAGE_TABLE..wire::DEVICE_DESC_PAGE_TABLE + 4]
        .copy_from_slice(&entry(table).to_le_bytes());
    let alloc_size =
        u32::try_from(PAGE_SIZE_ARM64E).expect("fixture allocation fits the wire field");
    desc[wire::DEVICE_DESC_ALLOC_SIZE..wire::DEVICE_DESC_ALLOC_SIZE + 4]
        .copy_from_slice(&alloc_size.to_le_bytes());
    put(descriptor, &desc);
    put(table, &entry(page).to_le_bytes());
    memory.map_range(page, PAGE_SIZE_ARM64E as usize, 0);
    memory.set_xreg(wire::MAPPER_CAPTURE_REG_MAPPER_DEVICE, mapper);
    memory.set_xreg(
        wire::MAPPER_CAPTURE_REG_REQUEST_TYPE,
        wire::MAPPER_REQUEST_MAP as u64,
    );
    memory.set_xreg(wire::MAPPER_CAPTURE_REG_MAPPING_INTERNAL, internal);
    let mut host = Box::new(CaptureHost {
        memory: Mutex::new(memory),
        calls: AtomicU64::new(0),
        wakes: AtomicU64::new(0),
        device: AtomicU64::new(0),
        reentry_checked: AtomicBool::new(false),
        alive: AtomicBool::new(true),
        after_teardown: AtomicBool::new(false),
    });
    let mut ops = ReimsVgpuHostOps::null();
    ops.ctx = (&mut *host as *mut CaptureHost).cast();
    ops.read_gpa = Some(read_memory);
    ops.read_kva = Some(read_kva);
    ops.read_xreg = Some(read_xreg);
    ops.is_ram_gpa = Some(ram_gpa);
    ops.schedule_bh = Some(wake);
    ops.notify_actions = Some(wake);
    let id = device_create(Some(ops), PAGE_SHIFT_ARM64E).unwrap();
    host.device.store(id, Ordering::Release);
    device_slot(id).unwrap().iosfc_regs.set_ring_base(ring);
    (id, host, internal)
}

#[test]
fn busy_and_wait_do_no_host_work_then_capture_and_ack_stay_synchronous() {
    let (id, host, internal) = capture_device();
    let slot = device_slot(id).unwrap();
    let held = slot.inner.lock();
    let ticket = device_iosfc_begin(id, IOSFC_REG_PRODUCER, 1, 4).unwrap();
    assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Busy);
    assert_eq!(host.calls.load(Ordering::Acquire), 0);
    assert_eq!(host.wakes.load(Ordering::Acquire), 0);
    assert_eq!(slot.iosfc_regs.producer(), 0);
    assert_eq!(slot.iosfc_regs.consumer(), 0);
    drop(held);
    assert_eq!(device_iosfc_wait(ticket), IosfcAdmissionStatus::Ready);
    assert_eq!(host.calls.load(Ordering::Acquire), 0);
    assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Ready);
    let calls = host.calls.load(Ordering::Acquire);
    assert!(calls > 0);
    assert!(host.reentry_checked.load(Ordering::Acquire));
    assert_eq!(slot.iosfc_regs.consumer(), 1);
    {
        let state = &slot.inner.lock().device.state;
        assert_eq!(state.mappings[&7].mapping_internal, internal);
        assert_eq!(state.mappings[&7].page_entries.len(), 1);
        assert!(state.mapper_capture.is_none());
        assert!(!state.pending.iosfc);
    }
    assert_eq!(
        device_pop_action(id).unwrap().kind,
        crate::runtime::host::HostActionKind::IrqIosfcPulse
    );
    assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Ready);
    assert_eq!(host.calls.load(Ordering::Acquire), calls, "no replay");
    let wakes = host.wakes.load(Ordering::Acquire);
    device_iosfc_finish(ticket);
    assert_eq!(host.wakes.load(Ordering::Acquire), wakes + 1);
    device_iosfc_finish(ticket);
    assert_eq!(host.wakes.load(Ordering::Acquire), wakes + 1);
    assert!(device_destroy(id));
}

#[test]
fn capture_failure_preserves_the_existing_consumer_and_irq_behavior() {
    let (id, host, _) = capture_device();
    host.memory.lock().set_xreg(
        wire::MAPPER_CAPTURE_REG_REQUEST_TYPE,
        wire::MAPPER_REQUEST_UNMAP as u64,
    );
    let ticket = device_iosfc_begin(id, IOSFC_REG_PRODUCER, 1, 4).unwrap();
    assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Ready);
    let slot = device_slot(id).unwrap();
    assert_eq!(slot.iosfc_regs.consumer(), 1);
    {
        let held = slot.inner.lock();
        let mapping = &held.device.state.mappings[&7];
        assert!(mapping.mapped);
        assert_eq!(mapping.mapping_internal, 0);
        assert!(mapping.page_entries.is_empty());
        assert!(held.device.state.mapper_capture.is_none());
    }
    assert_eq!(
        device_pop_action(id).unwrap().kind,
        crate::runtime::host::HostActionKind::IrqIosfcPulse
    );
    device_iosfc_finish(ticket);
    assert!(device_destroy(id));
}

#[test]
fn queued_admission_prevents_worker_stealing_and_rearms_without_a_doorbell() {
    let (id, host, _) = capture_device();
    let slot = device_slot(id).unwrap();
    let first = device_iosfc_begin(id, IOSFC_REG_CAPACITY, 1, 4).unwrap();
    let second = device_iosfc_begin(id, IOSFC_REG_CAPACITY, 2, 4).unwrap();
    for _ in 0..3 {
        assert!(device_drain(id));
    }
    assert_eq!(host.calls.load(Ordering::Acquire), 0);
    assert_eq!(device_iosfc_write(second), IosfcAdmissionStatus::Busy);
    assert_eq!(device_iosfc_write(first), IosfcAdmissionStatus::Ready);
    device_iosfc_finish(first);
    assert_eq!(host.wakes.load(Ordering::Acquire), 0);
    assert_eq!(device_iosfc_write(second), IosfcAdmissionStatus::Ready);
    assert_eq!(slot.iosfc_regs.capacity(), 2);
    device_iosfc_finish(second);
    assert_eq!(host.wakes.load(Ordering::Acquire), 1);
    assert!(slot.iosfc_admission.worker().is_some());
    assert!(device_destroy(id));
}

#[test]
fn reset_and_destroy_cancel_waiters_without_retaining_backend_or_callbacks() {
    let (id, host, _) = capture_device();
    let ticket = device_iosfc_begin(id, IOSFC_REG_CAPACITY, 1, 4).unwrap();
    assert!(device_reset(id));
    let wakes = host.wakes.load(Ordering::Acquire);
    assert_eq!(device_iosfc_wait(ticket), IosfcAdmissionStatus::Cancelled);
    assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Cancelled);
    device_iosfc_finish(ticket);
    assert_eq!(host.wakes.load(Ordering::Acquire), wakes);
    let fresh = device_iosfc_begin(id, IOSFC_REG_CAPACITY, 2, 4).unwrap();
    let weak = Arc::downgrade(&device_slot(id).unwrap());
    assert!(device_destroy(id));
    assert!(
        weak.upgrade().is_none(),
        "tickets must not retain BoundDevice"
    );
    assert_eq!(device_iosfc_wait(fresh), IosfcAdmissionStatus::Cancelled);
    assert_eq!(device_iosfc_write(fresh), IosfcAdmissionStatus::Cancelled);
    device_iosfc_finish(fresh);
    assert_eq!(host.wakes.load(Ordering::Acquire), wakes);
}

#[test]
fn multiple_vcpus_apply_producer_publications_in_admission_order() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).unwrap();
    let slot = device_slot(id).unwrap();
    let worker = slot.iosfc_admission.worker().unwrap();
    let bql = Arc::new(Mutex::new(()));
    let applied = Arc::new(Mutex::new(Vec::new()));
    let mut vcpus = Vec::new();
    for producer in [1, 2] {
        let bql = Arc::clone(&bql);
        let applied = Arc::clone(&applied);
        let (issued_tx, issued_rx) = std::sync::mpsc::channel();
        vcpus.push(std::thread::spawn(move || {
            let held = bql.lock();
            let ticket = device_iosfc_begin(id, IOSFC_REG_PRODUCER, producer, 4).unwrap();
            assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Busy);
            issued_tx.send(()).unwrap();
            drop(held);
            loop {
                assert_eq!(device_iosfc_wait(ticket), IosfcAdmissionStatus::Ready);
                let held = bql.lock();
                match device_iosfc_write(ticket) {
                    IosfcAdmissionStatus::Busy => drop(held),
                    IosfcAdmissionStatus::Ready => {
                        applied.lock().push(producer);
                        device_iosfc_finish(ticket);
                        break;
                    }
                    status => panic!("unexpected admission: {status:?}"),
                }
            }
        }));
        issued_rx.recv().unwrap();
    }
    drop(worker);
    for vcpu in vcpus {
        vcpu.join().unwrap();
    }
    assert_eq!(*applied.lock(), [1, 2]);
    assert_eq!(slot.iosfc_regs.producer(), 2);
    assert_eq!(slot.iosfc_regs.consumer(), 2);
    assert!(device_destroy(id));
}

#[test]
fn cancelled_vcpu_returns_after_teardown_without_callbacks_or_lock_inversion() {
    let (id, host, _) = capture_device();
    let slot = device_slot(id).unwrap();
    let weak = Arc::downgrade(&slot);
    let worker = slot.iosfc_admission.worker().unwrap();
    drop(slot);
    let bql = Arc::new(Mutex::new(()));
    let vcpu_bql = Arc::clone(&bql);
    let (issued_tx, issued_rx) = std::sync::mpsc::channel();
    let vcpu = std::thread::spawn(move || {
        let held = vcpu_bql.lock();
        let ticket = device_iosfc_begin(id, IOSFC_REG_PRODUCER, 1, 4).unwrap();
        assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Busy);
        issued_tx.send(()).unwrap();
        drop(held);
        let waited = device_iosfc_wait(ticket);
        let _held = vcpu_bql.lock();
        assert!(matches!(
            waited,
            IosfcAdmissionStatus::Ready | IosfcAdmissionStatus::Cancelled
        ));
        assert_eq!(device_iosfc_write(ticket), IosfcAdmissionStatus::Cancelled);
        device_iosfc_finish(ticket);
    });
    issued_rx.recv().unwrap();
    let held = bql.lock();
    drop(worker);
    assert!(device_destroy(id));
    assert!(weak.upgrade().is_none());
    host.alive.store(false, Ordering::Release);
    // The real C callback retains only its QOM allocation at this point.
    // Its HostOps/backend fields have ended while it is returning through BQL.
    drop(held);
    vcpu.join().unwrap();
    assert_eq!(host.calls.load(Ordering::Acquire), 0);
    assert_eq!(host.wakes.load(Ordering::Acquire), 0);
    assert!(!host.after_teardown.load(Ordering::Acquire));
}
