//! Pass-scoped storage for memoryless colour attachments.
//!
//! This rail submits and waits once per draw. A native memoryless allocation
//! cannot cross those encoder boundaries. For the supported single-sample 2D
//! colour contract, private storage in the *declared attachment format* is
//! equivalent: the first draw applies the guest load action, each completed
//! draw stores the attachment's format-converted texels, and the next loads
//! those same texels for blending/framebuffer fetch. No CPU conversion,
//! readback, sampling outside the pass, or resource cache participates.
//!
//! The private allocation is owned by one guest encoder, not by the texture's
//! object number. End-of-pass and every refusal retire it. The native regression
//! compares this split encoding with one real memoryless encoder, including
//! half-float values outside UNORM range and overlapping partial draws.

use super::render::new_color_target;
use crate::model::DeviceState;
use crate::runtime::draw::{ColorRtRequest, ColorStorage, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};
use metal::{MTLStorageMode, Texture};
use reims_vgpu_protocol::pass_action::{
    MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_DONT_CARE, MTL_LOAD_ACTION_LOAD,
    MTL_STORE_ACTION_DONT_CARE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Attachment {
    slot: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
    format: u16,
}

impl From<&ColorRtRequest> for Attachment {
    fn from(color: &ColorRtRequest) -> Self {
        Self {
            slot: color.slot,
            texture_ref: color.texture_ref,
            width: color.width,
            height: color.height,
            format: color.format,
        }
    }
}

pub(crate) struct PassLocalColorTarget {
    attachment: Attachment,
    texture: Texture,
    initialized: bool,
}

impl PassLocalColorTarget {
    pub(super) fn texture(&self) -> &Texture {
        &self.texture
    }

    pub(super) fn initialized(&self) -> bool {
        self.initialized
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    New,
    Open { task_id: u32 },
    Finished,
    Failed,
}

#[derive(Default)]
pub struct MetalRenderPass {
    phase: Phase,
    targets: Vec<PassLocalColorTarget>,
}

impl MetalRenderPass {
    fn prepare(&mut self, request: &DrawEncodeRequest) -> Result<(), &'static str> {
        match self.phase {
            Phase::New if !request.continues_render_pass => {}
            Phase::Open { task_id } if request.continues_render_pass
                && task_id == request.task_id => {}
            _ => return Err("draw_mtl_render_pass_sequence"),
        }
        let initial = self.phase == Phase::New;
        let colors: Vec<_> = request.colors.iter()
            .filter(|color| color.storage == ColorStorage::Memoryless).collect();
        for (index, color) in colors.iter().enumerate() {
            if colors[..index].iter().any(|prior| prior.slot == color.slot) {
                return Err("draw_mtl_memoryless_duplicate_slot");
            }
            if colors[..index].iter().any(|prior| prior.texture_ref == color.texture_ref) {
                return Err("draw_mtl_memoryless_attachment_alias");
            }
            if color.sample_count != 1 || color.multisample_source_ref != 0 {
                return Err("draw_mtl_memoryless_multisample");
            }
            if color.width == 0 || color.height == 0 {
                return Err("draw_mtl_memoryless_pass_geometry");
            }
            if reims_vgpu_protocol::memoryless::color_target_bpp(color.format).is_none() {
                return Err("draw_mtl_memoryless_pass_format");
            }
            if color.mapping_id != 0 || color.target_gva != 0
                || color.target_seed_rgba.is_some()
                || color.store_action != MTL_STORE_ACTION_DONT_CARE
            {
                return Err("draw_mtl_memoryless_backing");
            }
            let valid_load = if initial {
                matches!(color.load_action, MTL_LOAD_ACTION_DONT_CARE | MTL_LOAD_ACTION_CLEAR)
            } else {
                color.load_action == MTL_LOAD_ACTION_LOAD
            };
            if !valid_load {
                return Err("draw_mtl_memoryless_pass_load");
            }
        }
        if initial {
            for color in colors {
                let format = super::mtl_enum::pixel_format(u32::from(color.format))
                    .ok_or("draw_mtl_memoryless_pass_format")?;
                let device = super::runtime::system_device()
                    .ok_or("draw_mtl_memoryless_pass_device")?;
                let texture = new_color_target(
                    device, format, color.width, color.height, MTLStorageMode::Private,
                ).ok_or("draw_mtl_memoryless_pass_allocation")?;
                self.targets.push(PassLocalColorTarget {
                    attachment: Attachment::from(color),
                    texture,
                    initialized: false,
                });
            }
        } else if colors.len() != self.targets.len()
            || colors.iter().any(|color| {
                !self.targets.iter().any(|target| target.attachment == Attachment::from(*color))
            })
        {
            return Err("draw_mtl_memoryless_pass_attachment_changed");
        }
        self.phase = Phase::Open { task_id: request.task_id };
        Ok(())
    }

    pub(crate) fn target(&self, color: &ColorRtRequest) -> Result<&PassLocalColorTarget, &'static str> {
        self.targets.iter().find(|target| target.attachment == Attachment::from(color))
            .ok_or("draw_mtl_memoryless_pass_target_missing")
    }

    fn completed(&mut self, request: &mut DrawEncodeRequest, output: &mut Option<Vec<u8>>) {
        // Colour0 can itself be memoryless. Its chain is this pass's native
        // storage, never an empty RGBA8 buffer handed to the next draw.
        if request.colors.first().is_some_and(|c| c.storage == ColorStorage::Memoryless) {
            request.chain_resident_established = request.render_pass_continues;
            *output = None;
        }
        if request.render_pass_continues {
            for target in &mut self.targets {
                target.initialized = true;
            }
        } else {
            self.targets.clear();
            self.phase = Phase::Finished;
        }
    }

    fn refused(&mut self) {
        self.targets.clear();
        self.phase = Phase::Failed;
    }

    pub fn encode_draw<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        request: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        self.with_draw(request, |pass, request| {
            crate::runtime::draw::metal::encode_draw_in_pass(
                state, host, request, writeback_guest, force_full_store, pass,
            )
        })
    }

    fn with_draw(
        &mut self,
        request: &mut DrawEncodeRequest,
        encode: impl FnOnce(&Self, &mut DrawEncodeRequest) -> (EncodeStatus, Option<Vec<u8>>),
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        // QEMU's worker has no Cocoa event-loop pool. Completed command buffers,
        // encoders and pass descriptors otherwise retain every draw's resources.
        // Only owned pass targets and resident cache entries survive this scope.
        objc::rc::autoreleasepool(|| {
            if let Err(reason) = self.prepare(request) {
                self.refused();
                return (EncodeStatus::BadArgs(reason), None);
            }
            let mut result = encode(self, request);
            if matches!(result.0, EncodeStatus::Ok) {
                self.completed(request, &mut result.1);
            } else {
                self.refused();
            }
            result
        })
    }
}

#[cfg(test)]
mod tests;
