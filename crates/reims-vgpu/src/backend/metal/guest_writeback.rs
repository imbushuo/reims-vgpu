//! Synchronous, byte-exact RGBA8 Stores through checked guest-memory imports.
//!
//! Imports are keyed by allocation identity and retained until Metal's no-copy
//! deallocator runs, not merely until a Rust handle or command wrapper drops.
//! The same import owner also lends readonly buffer leases to mapped sampling.
//! No guest completion is published here; the runtime lands the physical write
//! witness and shared publication metadata only after `execute` completes.

use super::{raw_metal, runtime, util::Status};
use crate::protocol::pixel_format;
use crate::runtime::guest_ram::{GuestRamImport, GuestSlice, ImportId};
use foreign_types::ForeignTypeRef;
use metal::*;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

type View = (usize, usize);
type ImportedResult = Result<Arc<Imported>, (Arc<GuestRamImport>, Status)>;

#[derive(Default)]
struct Imports {
    active: HashMap<ImportId, ImportedResult>,
    retired: HashMap<ImportId, View>,
}

static IMPORTS: Mutex<Option<Imports>> = Mutex::new(None);
static RELEASED: Mutex<Vec<(ImportId, View)>> = Mutex::new(Vec::new());
static WRITER: Mutex<()> = Mutex::new(());
static WRITING: AtomicBool = AtomicBool::new(false);

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        for name in [
            crate::config::GUEST_IMPORT,
            crate::config::METAL_GPU_WRITEBACK,
        ] {
            let (setting, value) = crate::config::read(name);
            match setting {
                crate::config::Switch::Off => return false,
                crate::config::Switch::Unrecognized => {
                    crate::observe::Emit::refusal(
                        "metal_gpu_writeback",
                        &Status::args("metal_gpu_writeback_switch_unrecognized"),
                    )
                    .unwrap()
                    .field("name", name)
                    .field("value", value.unwrap_or_default())
                    .fail();
                    return false;
                }
                crate::config::Switch::On | crate::config::Switch::Unset => {}
            }
        }
        true
    })
}

pub(crate) fn publish_import_limits() {
    // Import capability is shared by readers and writers. Disabling GPU Stores
    // must not invent a restriction on an independently enabled readonly import.
    let limits = imports_enabled()
        .then(runtime::system_device)
        .flatten()
        .and_then(|device| {
            if !device.has_unified_memory() {
                return None;
            }
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let page = u64::try_from(page)
                .ok()
                .filter(|page| page.is_power_of_two())?;
            Some((
                page,
                device.recommended_max_working_set_size(),
                device.max_buffer_length(),
            ))
        });
    match limits {
        Some((page, budget, span)) => {
            crate::runtime::guest_ram::latch_import_limits(page, budget, span)
        }
        None => crate::runtime::guest_ram::forget_import_limits(),
    }
}

#[derive(Debug)]
struct Imported {
    import: Arc<GuestRamImport>,
    buffer: Buffer,
}

pub(super) fn imports_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let (setting, value) = crate::config::read(crate::config::GUEST_IMPORT);
        match setting {
            crate::config::Switch::Off => false,
            crate::config::Switch::On | crate::config::Switch::Unset => true,
            crate::config::Switch::Unrecognized => {
                crate::observe::Emit::refusal(
                    "metal_guest_import",
                    &Status::args("metal_guest_import_switch_unrecognized"),
                ).unwrap().field("value", value.unwrap_or_default()).fail();
                false
            }
        }
    })
}

/// Retains both the native no-copy buffer and its guest-allocation identity.
/// Retirement uses the exact same native deallocator handshake as a GPU Store.
#[derive(Clone, Debug)]
pub(super) struct ReadLease(Arc<Imported>);

impl ReadLease {
    pub(super) fn new(device: &DeviceRef, guest: Arc<GuestRamImport>) -> Result<Self, Status> {
        import(device, guest).map(Self)
    }

    pub(super) fn buffer(&self) -> &BufferRef { &self.0.buffer }

    pub(super) fn live(&self) -> bool { !self.0.import.is_retired() }
}

/// Checked readonly source coordinates for a caller-provided GPU command.
/// This owns the existing imported buffer; it creates no command or GPU wait.
/// The runtime still redeems the mapping/page/layout proof before submission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadLayout {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub format: u16,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadSource {
    lease: ReadLease,
    layout: ReadLayout,
    offset: u64,
    span: u64,
    available: u64,
}

impl ReadSource {
    pub(crate) fn new(
        device: &DeviceRef,
        guest: Arc<GuestRamImport>,
        slice: GuestSlice,
        layout: ReadLayout,
    ) -> Result<Self, Status> {
        pixel_format::bytes_per_pixel(layout.format)
            .ok_or_else(|| Status::args("metal_guest_read_format"))?;
        let row = pixel_format::tight_row_bytes(layout.width, layout.format)
            .ok_or_else(|| Status::args("metal_guest_read_format"))?;
        if layout.width == 0 || layout.height == 0 || layout.pitch < row {
            return Err(Status::args("metal_guest_read_layout"));
        }
        let span = u64::from(layout.pitch)
            .checked_mul(u64::from(layout.height - 1))
            .and_then(|span| span.checked_add(u64::from(row)))
            .ok_or_else(|| Status::args("metal_guest_read_span"))?;
        let resolved = guest.resolve(&slice)
            .map_err(|_| Status::args("metal_guest_read_foreign_slice"))?;
        let offset = resolved.offset.checked_add(slice.head())
            .ok_or_else(|| Status::args("metal_guest_read_span"))?;
        let available = slice.requested();
        if span > available || offset.checked_add(available).is_none_or(|end| end > guest.len()) {
            return Err(Status::args("metal_guest_read_span"));
        }
        let source = Self { lease: ReadLease::new(device, guest)?, layout, offset, span, available };
        source.check_live()?;
        if source.lease.buffer().device().as_ptr() != device.as_ptr() {
            return Err(Status::args("metal_guest_read_device_mismatch"));
        }
        Ok(source)
    }

    pub(crate) fn buffer(&self) -> Result<&BufferRef, Status> {
        self.check_live()?;
        Ok(self.lease.buffer())
    }

    pub(crate) fn check_live(&self) -> Result<(), Status> {
        if self.lease.live() { Ok(()) }
        else { Err(Status::args("metal_guest_read_retired")) }
    }

    pub(crate) fn import_id(&self) -> ImportId { self.lease.0.import.id() }
    pub(crate) fn layout(&self) -> ReadLayout { self.layout }
    pub(crate) fn offset(&self) -> u64 { self.offset }
    pub(crate) fn span(&self) -> u64 { self.span }
    pub(crate) fn available(&self) -> u64 { self.available }
}

fn import(device: &DeviceRef, guest: Arc<GuestRamImport>) -> Result<Arc<Imported>, Status> {
    if guest.is_retired() {
        return Err(Status::args("metal_guest_import_retired"));
    }
    let mut registry = IMPORTS.lock();
    let registry = registry.get_or_insert_with(Imports::default);
    if let Some(entry) = registry.active.get(&guest.id()) {
        return entry.as_ref().cloned().map_err(|(_, status)| *status);
    }
    registry
        .active
        .try_reserve(1)
        .map_err(|_| Status::execute("metal_guest_import_tracking_failed"))?;
    let created = (|| {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = u64::try_from(page)
            .ok()
            .filter(|page| page.is_power_of_two())
            .ok_or_else(|| Status::execute("metal_guest_import_page_size"))?;
        if !device.has_unified_memory()
            || !guest.host_base().is_multiple_of(page as usize)
            || !guest.len().is_multiple_of(page)
            || guest.len() > device.max_buffer_length()
        {
            return Err(Status::args("metal_guest_import_layout"));
        }
        let lifetime = guest.clone();
        let admitted = Arc::new(AtomicBool::new(false));
        let release_admitted = admitted.clone();
        let release = block::ConcreteBlock::new(move |_: *mut std::ffi::c_void, _: u64| {
            // Only retirement transfers unmapping to this callback. A rejected
            // constructor can release a temporary object while its map is live.
            if !release_admitted.load(Ordering::Acquire) || !lifetime.is_retired() {
                return;
            }
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                RELEASED.lock().push((
                    lifetime.id(),
                    (lifetime.host_base(), lifetime.len() as usize),
                ));
            }))
            .is_err()
            {
                let _ = std::panic::catch_unwind(|| {
                    crate::observe::Emit::refusal(
                        "metal_guest_import",
                        &Status::execute("metal_guest_import_release_tracking_failed"),
                    )
                    .unwrap()
                    .fail();
                });
            }
        })
        .copy();
        let buffer = unsafe {
            raw_metal::new_buffer_no_copy_with_release(
                device,
                guest.host_base() as *mut _,
                guest.len(),
                &release,
            )
        }
        .ok_or_else(|| Status::execute("metal_guest_import_create_failed"))?;
        if buffer.length() != guest.len() {
            return Err(Status::execute("metal_guest_import_length_mismatch"));
        }
        if guest.is_retired() {
            return Err(Status::args("metal_guest_import_retired"));
        }
        admitted.store(true, Ordering::Release);
        crate::runtime::drain::note_store_route("metal_guest_import_created");
        Ok(Arc::new(Imported {
            import: guest.clone(),
            buffer,
        }))
    })();
    registry.active.insert(
        guest.id(),
        created.clone().map_err(|status| (guest, status)),
    );
    created
}

pub(crate) fn retire(id: ImportId) -> Option<View> {
    let removed = {
        let mut registry = IMPORTS.lock();
        let registry = registry.as_mut()?;
        let Some(entry) = registry.active.remove(&id) else {
            return registry.retired.get(&id).copied();
        };
        match entry {
            Ok(held) => {
                held.import.retire();
                let view = (held.import.host_base(), held.import.len() as usize);
                registry.retired.insert(id, view);
                (held, view)
            }
            Err((import, _)) => {
                import.retire();
                return None;
            }
        }
    };
    let view = removed.1;
    drop(removed);
    Some(view)
}

pub(crate) fn released() -> Vec<View> {
    let ready: Vec<_> = RELEASED.lock().drain(..).collect();
    let mut registry = IMPORTS.lock();
    if let Some(registry) = registry.as_mut() {
        for (id, _) in &ready {
            registry.retired.remove(id);
        }
    }
    ready.into_iter().map(|(_, view)| view).collect()
}

pub(crate) fn reset() {
    let _writer = WRITER.lock();
    let ids: Vec<_> = IMPORTS
        .lock()
        .as_ref()
        .map(|registry| registry.active.keys().copied().collect())
        .unwrap_or_default();
    for id in ids {
        retire(id);
    }
}

pub(crate) fn outstanding() -> bool {
    WRITING.load(Ordering::Acquire)
}
pub(crate) fn quiesce() {
    drop(WRITER.lock());
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Output {
    Bgra8,
    Rgba16Float,
}

impl Output {
    pub(crate) fn for_format(format: u16) -> Option<Self> {
        match format {
            pixel_format::MTL_FORMAT_BGRA8_UNORM | pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB => {
                Some(Self::Bgra8)
            }
            pixel_format::MTL_FORMAT_RGBA16_FLOAT => Some(Self::Rgba16Float),
            _ => None,
        }
    }

    fn bytes(self) -> u64 {
        match self {
            Self::Bgra8 => 4,
            Self::Rgba16Float => 8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub output: Output,
}

impl Layout {
    pub(crate) fn span(self) -> Option<u64> {
        let row = u64::from(self.width).checked_mul(self.output.bytes())?;
        if self.width == 0 || self.height == 0 || u64::from(self.pitch) < row {
            return None;
        }
        u64::from(self.pitch)
            .checked_mul(u64::from(self.height - 1))?
            .checked_add(row)
    }
}

#[repr(C)]
struct Params {
    src_offset: u64,
    src_pitch: u64,
    dst_offset: u64,
    dst_pitch: u64,
    width: u32,
    height: u32,
    half_output: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Params>() == 48);
const _: () = assert!(std::mem::offset_of!(Params, width) == 32);

struct Kernel {
    pso: ComputePipelineState,
    half_lut: Buffer,
}

fn kernel(device: &DeviceRef) -> Result<&'static Kernel, Status> {
    static KERNEL: OnceLock<Result<Kernel, Status>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            let library =
                raw_metal::new_library_with_source(device, include_str!("guest_writeback.metal"))
                    .map_err(|_| Status::execute("metal_gpu_writeback_shader"))?;
            let function = library
                .get_function("store_rgba8", None)
                .map_err(|_| Status::execute("metal_gpu_writeback_function"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Status::execute("metal_gpu_writeback_pipeline"))?;
            // Derive every half bit pattern from the existing CPU conversion.
            // The GPU performs byte stores and no floating-point narrowing.
            let mut lut = [0u16; 256];
            for (value, bits) in lut.iter_mut().enumerate() {
                let mut converted = [0u8; 8];
                if !pixel_format::Rgba8ToRow::Rgba16Float.convert(
                    &[value as u8; 4],
                    1,
                    &mut converted,
                ) {
                    return Err(Status::execute("metal_gpu_writeback_half_table"));
                }
                *bits = u16::from_le_bytes([converted[0], converted[1]]);
            }
            let half_lut = unsafe {
                raw_metal::new_buffer_with_data(
                    device,
                    lut.as_ptr().cast(),
                    std::mem::size_of_val(&lut) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| Status::execute("metal_gpu_writeback_lut_buffer"))?;
            Ok(Kernel { pso, half_lut })
        })
        .as_ref()
        .map_err(|status| *status)
}

/// Fully recorded but unsubmitted Store. All imports and source bytes stay
/// owned until completion or abandonment of this command.
pub(crate) struct Prepared {
    command: CommandBuffer,
    destination: Arc<Imported>,
    _source: Buffer,
    submitted: bool,
    writer: Option<parking_lot::MutexGuard<'static, ()>>,
}

impl Prepared {
    pub(crate) fn new(
        device: &Device,
        source: &TextureRef,
        guest: Arc<GuestRamImport>,
        slice: GuestSlice,
        layout: Layout,
    ) -> Result<Self, Status> {
        if source.pixel_format() != MTLPixelFormat::RGBA8Unorm
            || source.texture_type() != MTLTextureType::D2
            || source.width() != u64::from(layout.width)
            || source.height() != u64::from(layout.height)
            || source.sample_count() != 1
        {
            return Err(Status::args("metal_gpu_writeback_source_format"));
        }
        let source_buffer = source
            .buffer()
            .ok_or_else(|| Status::args("metal_gpu_writeback_source_not_linear"))?
            .to_owned();
        let src_offset = source.buffer_offset();
        let src_pitch = source.buffer_stride();
        let src_row = u64::from(layout.width) * 4;
        let src_end = src_pitch
            .checked_mul(u64::from(layout.height.saturating_sub(1)))
            .and_then(|end| end.checked_add(src_row))
            .and_then(|end| src_offset.checked_add(end));
        let span = layout
            .span()
            .ok_or_else(|| Status::args("metal_gpu_writeback_layout"))?;
        if src_pitch < src_row
            || src_end.is_none_or(|end| end > source_buffer.length())
            || span > slice.requested()
        {
            return Err(Status::args("metal_gpu_writeback_span"));
        }
        let range = guest
            .resolve(&slice)
            .map_err(|_| Status::args("metal_gpu_writeback_foreign_slice"))?;
        let dst_offset = range
            .offset
            .checked_add(slice.head())
            .ok_or_else(|| Status::args("metal_gpu_writeback_span"))?;
        if dst_offset
            .checked_add(span)
            .is_none_or(|end| end > guest.len())
        {
            return Err(Status::args("metal_gpu_writeback_span"));
        }
        let kernel = kernel(device)?;
        let destination = import(device, guest)?;
        let command = raw_metal::new_command_buffer(&runtime::thread_queue(device))
            .ok_or_else(|| Status::execute("metal_gpu_writeback_command"))?
            .to_owned();
        let encoder = raw_metal::new_compute_command_encoder_with_dispatch_type(
            &command,
            MTLDispatchType::Serial,
        )
        .ok_or_else(|| Status::execute("metal_gpu_writeback_encoder"))?;
        let params = Params {
            src_offset,
            src_pitch,
            dst_offset,
            dst_pitch: u64::from(layout.pitch),
            width: layout.width,
            height: layout.height,
            half_output: u32::from(matches!(layout.output, Output::Rgba16Float)),
            reserved: 0,
        };
        encoder.set_compute_pipeline_state(&kernel.pso);
        encoder.set_buffer(0, Some(&source_buffer), 0);
        encoder.set_buffer(1, Some(&destination.buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.set_buffer(3, Some(&kernel.half_lut), 0);
        let threads = kernel
            .pso
            .thread_execution_width()
            .min(kernel.pso.max_total_threads_per_threadgroup());
        if threads == 0 {
            encoder.end_encoding();
            return Err(Status::execute("metal_gpu_writeback_threadgroup"));
        }
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(layout.width).div_ceil(threads),
                u64::from(layout.height),
                1,
            ),
            MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        Ok(Self {
            command,
            destination,
            _source: source_buffer,
            submitted: false,
            writer: None,
        })
    }

    pub(crate) fn live(&self) -> bool {
        !self.destination.import.is_retired()
    }

    pub(crate) fn execute(mut self) -> Result<(), Status> {
        if !self.live() {
            return Err(Status::args("metal_guest_import_retired"));
        }
        self.writer = Some(WRITER.lock());
        if !self.live() {
            return Err(Status::args("metal_guest_import_retired"));
        }
        WRITING.store(true, Ordering::Release);
        self.submitted = true;
        self.command.commit();
        self.command.wait_until_completed();
        if self.command.status() != MTLCommandBufferStatus::Completed {
            return Err(Status::execute("metal_gpu_writeback_command_failed"));
        }
        crate::runtime::drain::note_store_route("metal_gpu_writebacks");
        Ok(())
    }
}

impl Drop for Prepared {
    fn drop(&mut self) {
        if self.submitted {
            self.command.wait_until_completed();
            WRITING.store(false, Ordering::Release);
        }
        self.writer.take();
    }
}

#[cfg(test)]
mod tests;
