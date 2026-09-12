//! Native attachments and submissions owned by one guest render encoder.
//!
//! Ordinary attachments keep the rail's RGBA8 conversion contract; memoryless
//! attachments keep their declared format. Neither uses cross-pass publication
//! as evidence of unfinished content. Compatible draws share an encoder until
//! a CPU-visible dependency or the guest pass end requires completion.

use super::render::new_color_target;
use crate::model::DeviceState;
use crate::runtime::draw::{ColorRtRequest, ColorStorage, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};
use metal::{MTLStorageMode, Texture};
use reims_vgpu_protocol::pass_action::{
    MTL_LOAD_ACTION_CLEAR, MTL_LOAD_ACTION_DONT_CARE, MTL_LOAD_ACTION_LOAD,
    MTL_STORE_ACTION_DONT_CARE,
};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Attachment {
    slot: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
    format: u16,
    storage: ColorStorage,
    mapping_id: u32,
    target_gva: u64,
    row_stride: u32,
    store_action: u16,
    sample_count: u32,
    multisample_source_ref: u32,
}

impl From<&ColorRtRequest> for Attachment {
    fn from(color: &ColorRtRequest) -> Self {
        Self {
            slot: color.slot,
            texture_ref: color.texture_ref,
            width: color.width,
            height: color.height,
            format: color.format,
            storage: color.storage,
            mapping_id: color.mapping_id,
            target_gva: color.target_gva,
            row_stride: color.row_stride,
            store_action: color.store_action,
            sample_count: color.sample_count,
            multisample_source_ref: color.multisample_source_ref,
        }
    }
}

pub(crate) struct PassLocalColorTarget {
    attachment: Attachment,
    pub(crate) texture: Option<Texture>,
    // An earlier record in this owner established contents in GPU order. This
    // is not CPU readiness: every CPU consumer must complete the batch first.
    initialized: bool,
    pub(crate) resident: Option<crate::runtime::draw::metal::ResidentPlan>,
}

impl PassLocalColorTarget {
    pub(crate) fn texture(&self) -> &Texture {
        self.texture
            .as_ref()
            .expect("pass target installed before encoding")
    }

    pub(crate) fn initialized(&self) -> bool {
        self.initialized
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    New,
    Open {
        task_id: u32,
    },
    Finished,
    Failed,
}

#[derive(Default)]
pub struct MetalRenderPass {
    phase: Phase,
    targets: Vec<PassLocalColorTarget>,
    pub(crate) batch: RefCell<super::render::RenderBatch>,
    pub(crate) depth: Option<crate::runtime::draw::metal::HostDepthStencil>,
    pub(crate) stencil: Option<crate::runtime::draw::metal::HostDepthStencil>,
    pub(crate) dependencies: RefCell<crate::runtime::draw::metal::PassDependencies>,
    depth_identity: Option<(u32, crate::runtime::render_pass::AttachSubresource, u16)>,
    stencil_identity: Option<(u32, crate::runtime::render_pass::AttachSubresource, u16)>,
}

impl MetalRenderPass {
    fn prepare(&mut self, request: &DrawEncodeRequest) -> Result<(), &'static str> {
        match self.phase {
            Phase::New if !request.continues_render_pass => {}
            Phase::Open { task_id }
                if request.continues_render_pass && task_id == request.task_id => {}
            _ => return Err("draw_mtl_render_pass_sequence"),
        }
        let initial = self.phase == Phase::New;
        let depth = request
            .depth_attach
            .map(|a| (a.texture_ref, a.into(), a.store_action));
        let stencil = request
            .stencil_attach
            .map(|a| (a.texture_ref, a.into(), a.store_action));
        if !initial && (self.depth_identity != depth || self.stencil_identity != stencil) {
            return Err("draw_mtl_depth_stencil_pass_attachment_changed");
        }
        self.depth_identity = depth;
        self.stencil_identity = stencil;
        for (index, color) in request.colors.iter().enumerate() {
            if color.storage == ColorStorage::Memoryless {
                continue;
            }
            if request.colors[..index]
                .iter()
                .any(|prior| prior.slot == color.slot)
            {
                return Err("draw_mtl_render_pass_duplicate_slot");
            }
            if request.colors[..index].iter().any(|prior| {
                prior.texture_ref == color.texture_ref
                    || (color.mapping_id != 0 && prior.mapping_id == color.mapping_id)
                    || (color.target_gva != 0 && prior.target_gva == color.target_gva)
            }) {
                return Err("draw_mtl_render_pass_attachment_alias");
            }
            if !initial && color.load_action != MTL_LOAD_ACTION_LOAD {
                return Err("draw_mtl_render_pass_load");
            }
        }
        let colors: Vec<_> = request
            .colors
            .iter()
            .filter(|color| color.storage == ColorStorage::Memoryless)
            .collect();
        for (index, color) in colors.iter().enumerate() {
            if colors[..index].iter().any(|prior| prior.slot == color.slot) {
                return Err("draw_mtl_memoryless_duplicate_slot");
            }
            if colors[..index]
                .iter()
                .any(|prior| prior.texture_ref == color.texture_ref)
            {
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
            if color.mapping_id != 0
                || color.target_gva != 0
                || color.target_seed_rgba.is_some()
                || color.store_action != MTL_STORE_ACTION_DONT_CARE
            {
                return Err("draw_mtl_memoryless_backing");
            }
            let valid_load = if initial {
                matches!(
                    color.load_action,
                    MTL_LOAD_ACTION_DONT_CARE | MTL_LOAD_ACTION_CLEAR
                )
            } else {
                color.load_action == MTL_LOAD_ACTION_LOAD
            };
            if !valid_load {
                return Err("draw_mtl_memoryless_pass_load");
            }
        }
        if initial {
            for color in &request.colors {
                let texture = if color.storage == ColorStorage::Memoryless {
                    let format = super::mtl_enum::pixel_format(u32::from(color.format))
                        .ok_or("draw_mtl_memoryless_pass_format")?;
                    let device =
                        super::runtime::system_device().ok_or("draw_mtl_memoryless_pass_device")?;
                    Some(
                        new_color_target(
                            device,
                            format,
                            color.width,
                            color.height,
                            MTLStorageMode::Private,
                        )
                        .ok_or("draw_mtl_memoryless_pass_allocation")?,
                    )
                } else {
                    None
                };
                self.targets.push(PassLocalColorTarget {
                    attachment: Attachment::from(color),
                    texture,
                    initialized: false,
                    resident: None,
                });
            }
        } else if request.colors.len() != self.targets.len()
            || request.colors.iter().any(|color| {
                !self
                    .targets
                    .iter()
                    .any(|target| target.attachment == Attachment::from(color))
            })
        {
            return Err(
                if request
                    .colors
                    .iter()
                    .any(|color| color.storage == ColorStorage::Memoryless)
                {
                    "draw_mtl_memoryless_pass_attachment_changed"
                } else {
                    "draw_mtl_render_pass_attachment_changed"
                },
            );
        }
        self.phase = Phase::Open {
            task_id: request.task_id,
        };
        Ok(())
    }

    pub(crate) fn target(
        &self,
        color: &ColorRtRequest,
    ) -> Result<&PassLocalColorTarget, &'static str> {
        self.targets
            .iter()
            .find(|target| target.attachment == Attachment::from(color))
            .ok_or(match color.storage {
                ColorStorage::Memoryless => "draw_mtl_memoryless_pass_target_missing",
                ColorStorage::GuestBacked => "draw_mtl_render_pass_target_missing",
            })
    }

    pub(crate) fn target_mut(
        &mut self,
        color: &ColorRtRequest,
    ) -> Result<&mut PassLocalColorTarget, &'static str> {
        self.targets
            .iter_mut()
            .find(|target| target.attachment == Attachment::from(color))
            .ok_or("draw_mtl_render_pass_target_missing")
    }

    pub(crate) fn flush(&self, reason: &'static str) -> Result<(), super::util::Status> {
        let mut batch = self.batch.borrow_mut();
        if batch.pending() {
            crate::runtime::drain::note_store_route(reason);
        }
        batch.finish((std::ptr::null_mut(), 0))
    }

    fn completed(&mut self, request: &mut DrawEncodeRequest, output: &mut Option<Vec<u8>>) {
        // Colour0 can itself be memoryless. Its chain is this pass's native
        // storage, never an empty RGBA8 buffer handed to the next draw.
        if request.render_pass_continues
            || request
                .colors
                .first()
                .is_some_and(|c| c.storage == ColorStorage::Memoryless)
        {
            request.chain_resident_established = request.render_pass_continues;
            *output = None;
        }
        if request.render_pass_continues {
            for target in &mut self.targets {
                target.initialized = true;
            }
        } else {
            self.targets.clear();
            self.depth = None;
            self.stencil = None;
            self.phase = Phase::Finished;
        }
    }

    fn refused(&mut self) {
        // Submitted work must complete before caller-owned guest state can be
        // released, even when a later record refuses before encoding.
        if let Err(status) = self.flush("metal_batch_refusal") {
            crate::observe::Emit::refusal("metal_render_pass", &status)
                .unwrap()
                .fail();
        }
        self.targets.clear();
        self.depth = None;
        self.stencil = None;
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
            let result = crate::runtime::draw::metal::encode_draw_in_pass(
                state,
                host,
                request,
                writeback_guest,
                force_full_store,
                pass,
            );
            if !matches!(result.0, EncodeStatus::Ok) {
                if let Err(status) =
                    crate::runtime::draw::metal::land_before_refusal(state, host, request, pass)
                {
                    crate::observe::Emit::refusal("metal_pass_abandon", &status)
                        .unwrap()
                        .fail();
                }
            }
            result
        })
    }

    fn with_draw(
        &mut self,
        request: &mut DrawEncodeRequest,
        encode: impl FnOnce(&mut Self, &mut DrawEncodeRequest) -> (EncodeStatus, Option<Vec<u8>>),
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
            if matches!(result.0, EncodeStatus::Ok) && !request.render_pass_continues {
                if let Err(status) = self.flush("metal_batch_pass_end") {
                    result = (EncodeStatus::RailRefused(status), None);
                }
            }
            if matches!(result.0, EncodeStatus::Ok) {
                self.completed(request, &mut result.1);
            } else {
                self.refused();
            }
            result
        })
    }
}

impl Drop for MetalRenderPass {
    fn drop(&mut self) {
        if let Err(status) = self.flush("metal_batch_drop") {
            crate::observe::Emit::refusal("metal_render_pass_drop", &status)
                .unwrap()
                .fail();
        }
    }
}

#[cfg(test)]
mod tests;
