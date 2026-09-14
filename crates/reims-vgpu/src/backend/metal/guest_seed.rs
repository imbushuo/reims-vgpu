//! Byte-exact LOAD seeding into an existing linear RGBA8 render target.
//! Guest import ownership, currency and precommit validation stay with the
//! caller. This component only validates native windows and records a compute
//! encoder on the supplied command; it never commits, waits or reads pixels.

use super::{raw_metal, util::Status};
use crate::protocol::pixel_format::RowToRgba8;
use foreign_types::ForeignTypeRef;
use metal::*;
use objc::{msg_send, sel, sel_impl};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub source_offset: u64,
    /// Authorized bytes after source_offset, not the enclosing import's size.
    pub source_length: u64,
    pub source_pitch: u64,
    pub width: u32,
    pub height: u32,
    pub source_format: u16,
}

#[derive(Clone, Copy)]
enum Input {
    Rgba8 = 0,
    Bgra8 = 1,
    Rgba16Float = 2,
}

impl Input {
    fn for_format(format: u16) -> Result<Self, Status> {
        match RowToRgba8::for_format(format) {
            Some(RowToRgba8::Rgba8) => Ok(Self::Rgba8),
            Some(RowToRgba8::Bgra8) => Ok(Self::Bgra8),
            Some(RowToRgba8::Rgba16Float) => Ok(Self::Rgba16Float),
            _ => Err(Status::args("metal_guest_seed_source_format")),
        }
    }

    fn bytes(self) -> u64 {
        if matches!(self, Self::Rgba16Float) {
            8
        } else {
            4
        }
    }
}

fn span(width: u32, height: u32, pitch: u64, bytes: u64) -> Option<u64> {
    let row = u64::from(width).checked_mul(bytes)?;
    if width == 0 || height == 0 || pitch < row {
        return None;
    }
    pitch.checked_mul(u64::from(height - 1))?.checked_add(row)
}

fn checked_end(offset: u64, bytes: u64, capacity: u64) -> Option<u64> {
    offset.checked_add(bytes).filter(|&end| end <= capacity)
}

#[repr(C)]
struct Params {
    src_offset: u64,
    src_pitch: u64,
    dst_offset: u64,
    dst_pitch: u64,
    width: u32,
    height: u32,
    input: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Params>() == 48);
const _: () = assert!(std::mem::offset_of!(Params, width) == 32);

struct Kernel {
    device: Device,
    pso: ComputePipelineState,
    half_lut: Buffer,
    threads: u64,
}

fn kernel(device: &DeviceRef) -> Result<&'static Kernel, Status> {
    static KERNEL: OnceLock<Result<Kernel, Status>> = OnceLock::new();
    let kernel = KERNEL
        .get_or_init(|| {
            let library =
                raw_metal::new_library_with_source(device, include_str!("guest_seed.metal"))
                    .map_err(|_| Status::execute("metal_guest_seed_shader"))?;
            let function = library
                .get_function("seed_rgba8", None)
                .map_err(|_| Status::execute("metal_guest_seed_function"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Status::execute("metal_guest_seed_pipeline"))?;
            let threads = pso
                .thread_execution_width()
                .min(pso.max_total_threads_per_threadgroup());
            if threads == 0 {
                return Err(Status::execute("metal_guest_seed_threadgroup"));
            }
            // Use the CPU row contract itself, including every NaN and signed zero.
            // No half/f32 conversion or rounding is performed by the GPU.
            let mut lut = vec![0u8; 65536];
            for (bits, value) in lut.iter_mut().enumerate() {
                let pair = (bits as u16).to_le_bytes();
                let source = [pair[0], pair[1], 0, 0, 0, 0, 0, 0];
                let mut rgba = [0; 4];
                if !RowToRgba8::Rgba16Float.convert(&source, 1, &mut rgba) {
                    return Err(Status::execute("metal_guest_seed_half_table"));
                }
                *value = rgba[0];
            }
            let half_lut = unsafe {
                raw_metal::new_buffer_with_data(
                    device,
                    lut.as_ptr().cast(),
                    lut.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| Status::execute("metal_guest_seed_lut_buffer"))?;
            Ok(Kernel {
                device: device.to_owned(),
                pso,
                half_lut,
                threads,
            })
        })
        .as_ref()
        .map_err(|status| *status)?;
    if kernel.device.as_ptr() != device.as_ptr() {
        return Err(Status::args("metal_guest_seed_device"));
    }
    Ok(kernel)
}

struct Resources {
    _source: Buffer,
    _destination: Buffer,
    _target: Texture,
}

/// A recorded seed job. Native references are also held by a completion block,
/// so dropping this handle early cannot release resources during GPU use.
pub(crate) struct Prepared {
    _resources: Arc<Resources>,
}

impl Prepared {
    /// Record before opening the render encoder. The caller must retain its
    /// checked guest read/import lease and validate its currency before commit.
    /// All refusals occur before a compute encoder is opened.
    pub(crate) fn encode(
        device: &DeviceRef,
        command: &CommandBufferRef,
        source: &BufferRef,
        target: &TextureRef,
        layout: Layout,
    ) -> Result<Self, Status> {
        let input = Input::for_format(layout.source_format)?;
        if !matches!(
            command.status(),
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return Err(Status::args("metal_guest_seed_command_state"));
        }
        // SAFETY: MTLCommandBuffer's device is a borrowed native object kept
        // alive by the provided command; no pixel or guest memory is accessed.
        let command_device: *mut objc::runtime::Object = unsafe { msg_send![command, device] };
        if command_device != device.as_ptr().cast()
            || source.device().as_ptr() != device.as_ptr()
            || target.device().as_ptr() != device.as_ptr()
        {
            return Err(Status::args("metal_guest_seed_device"));
        }
        if target.pixel_format() != MTLPixelFormat::RGBA8Unorm
            || target.texture_type() != MTLTextureType::D2
            || target.sample_count() != 1
            || target.mipmap_level_count() != 1
            || target.depth() != 1
            || target.array_length() != 1
            || !target.usage().contains(MTLTextureUsage::RenderTarget)
            || target.width() != u64::from(layout.width)
            || target.height() != u64::from(layout.height)
        {
            return Err(Status::args("metal_guest_seed_target_format"));
        }
        let destination = target
            .buffer()
            .ok_or_else(|| Status::args("metal_guest_seed_target_not_linear"))?;
        if source.hazard_tracking_mode() == MTLHazardTrackingMode::Untracked
            || destination.hazard_tracking_mode() == MTLHazardTrackingMode::Untracked
            || target.hazard_tracking_mode() == MTLHazardTrackingMode::Untracked
        {
            return Err(Status::args("metal_guest_seed_untracked_resource"));
        }
        let src_span = span(
            layout.width,
            layout.height,
            layout.source_pitch,
            input.bytes(),
        )
        .ok_or_else(|| Status::args("metal_guest_seed_source_layout"))?;
        let dst_span = span(layout.width, layout.height, target.buffer_stride(), 4)
            .ok_or_else(|| Status::args("metal_guest_seed_target_layout"))?;
        let src_end = layout
            .source_offset
            .checked_add(src_span)
            .ok_or_else(|| Status::args("metal_guest_seed_source_bounds"))?;
        if src_span > layout.source_length
            || checked_end(layout.source_offset, layout.source_length, source.length()).is_none()
        {
            return Err(Status::args("metal_guest_seed_source_bounds"));
        }
        let dst_offset = target.buffer_offset();
        let dst_end = checked_end(dst_offset, dst_span, destination.length())
            .ok_or_else(|| Status::args("metal_guest_seed_target_bounds"))?;
        if !dst_offset.is_multiple_of(4) || !target.buffer_stride().is_multiple_of(4) {
            return Err(Status::args("metal_guest_seed_target_alignment"));
        }
        if source.as_ptr() == destination.as_ptr()
            && layout.source_offset < dst_end
            && dst_offset < src_end
        {
            return Err(Status::args("metal_guest_seed_overlapping_storage"));
        }
        let kernel = kernel(device)?;
        let resources = Arc::new(Resources {
            _source: source.to_owned(),
            _destination: destination.to_owned(),
            _target: target.to_owned(),
        });
        let encoder = raw_metal::new_compute_command_encoder_with_dispatch_type(
            command,
            MTLDispatchType::Serial,
        )
        .ok_or_else(|| Status::execute("metal_guest_seed_encoder"))?;
        let params = Params {
            src_offset: layout.source_offset,
            src_pitch: layout.source_pitch,
            dst_offset,
            dst_pitch: target.buffer_stride(),
            width: layout.width,
            height: layout.height,
            input: input as u32,
            reserved: 0,
        };
        encoder.set_compute_pipeline_state(&kernel.pso);
        encoder.set_buffer(0, Some(source), 0);
        encoder.set_buffer(1, Some(destination), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.set_buffer(3, Some(&kernel.half_lut), 0);
        // Declare the texture alias as written so the following render LOAD
        // participates in tracked resource ordering, not just buffer ordering.
        encoder.use_resource(target, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(layout.width).div_ceil(kernel.threads),
                u64::from(layout.height),
                1,
            ),
            MTLSize::new(kernel.threads, 1, 1),
        );
        encoder.end_encoding();
        let until_completed = Arc::new(Mutex::new(Some(Arc::clone(&resources))));
        let completed = block::ConcreteBlock::new(move |_: &CommandBufferRef| {
            until_completed.lock().take();
        })
        .copy();
        command.add_completed_handler(&completed);
        crate::runtime::drain::note_store_route("metal_guest_seed_encoded");
        crate::runtime::drain::note_store_route_n(
            "metal_guest_seed_pixels",
            u64::from(layout.width) * u64::from(layout.height),
        );
        Ok(Self {
            _resources: resources,
        })
    }
}

#[cfg(test)]
mod tests;
