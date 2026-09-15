//! Primary GVA Store routing retains the identity selected before the draw.

use super::*;
use crate::backend::vulkan::engine::{DrawRequest, TargetIdentity};
use crate::observe::Decline;
use crate::runtime::render_writeback::vulkan::{
    requires_native_store, GvaWritebackDecline, NativeEagerStore,
};

enum Timing {
    MayDefer,
    Synchronous,
}

pub(super) struct GvaStore {
    identity: TargetIdentity,
    timing: Timing,
    native: Option<Result<NativeEagerStore, GvaWritebackDecline>>,
}

pub(super) enum StoreOutput {
    Complete,
    Rgba8(Vec<u8>),
}

impl GvaStore {
    pub(super) fn prepare(
        state: &DeviceState,
        req: &DrawEncodeRequest,
        resources: &mut DrawRequest,
        gpu_only_content_allowed: bool,
        writeback_guest: bool,
    ) -> Option<Self> {
        let color = req.colors.first()?;
        if !writeback_guest
            || !reims_vgpu_protocol::pass_action::store_action_publishes_single_sample(
                color.store_action,
            )
        {
            return None;
        }
        let identity = gva_chain_identity(req)?;
        let timing = if gpu_only_content_allowed && gva_store_defer_eligible(req) {
            Timing::MayDefer
        } else if requires_native_store(color.format) {
            // The ordinary draw readback converts float attachments to RGBA8.
            // Keep this exact identity for the already-synchronous native Store.
            Timing::Synchronous
        } else {
            return None;
        };
        resources.target_identity = Some(identity.clone());
        resources.skip_readback = true;
        let native = requires_native_store(color.format)
            .then(|| NativeEagerStore::capture(state, req.task_id, color, req.gva_alloc_gen));
        Some(Self {
            identity,
            timing,
            native,
        })
    }

    pub(super) fn finish<M: HostMemory + HostOps>(
        self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        pages: Option<&StoreTargetPages>,
    ) -> Result<StoreOutput, EncodeStatus> {
        let _store_span = crate::runtime::chain_phase::CostSpan::new("gva_store_us");
        note_mapper_ref_texture_store_route("gva_flush");
        let Some(color) = req.colors.first() else {
            crate::observe::fail("draw_vk_gva_store reason=primary_target_missing");
            return Err(EncodeStatus::WritebackFailed(
                "draw_vk_gva_store_target_missing",
            ));
        };
        let native = match self.native.as_ref().map(Result::as_ref).transpose() {
            Ok(native) => native,
            Err(refusal) => return Err(report_refusal(req, refusal)),
        };
        if let Some(native) = native {
            if let Err(refusal) = native.validate_live(state) {
                return Err(report_refusal(req, &refusal));
            }
        }
        let landed = matches!(self.timing, Timing::MayDefer)
            && crate::backend::vulkan::gva_window(&self.identity).is_some_and(|window| {
                crate::runtime::writeback_debt::arm_gva(
                    crate::backend::vulkan::VulkanBackend,
                    state,
                    host,
                    req.task_id,
                    color,
                    window,
                )
            });
        if landed {
            note_mapper_ref_texture_store_route("gva_resident_authoritative");
            return Ok(StoreOutput::Complete);
        }
        note_mapper_ref_texture_store_route("gva_store_sync");
        if let Some(native) = native {
            if let Err(refusal) = native.publish(state, host, &self.identity, pages) {
                return Err(report_refusal(req, &refusal));
            }

            note_mapper_ref_texture_store_route("gva_store_native");
            crate::observe::off(format!(
                "m2v_store_gva_native task={} pipe={} tex_ref={} gva={:#x} {}x{} fmt={:#x} bpr={} writer=cpu_native route={}",
                req.task_id, req.pipeline_ref, color.texture_ref, color.target_gva,
                color.width, color.height, color.format, color.row_stride,
                match self.timing {
                    Timing::MayDefer => "deferred_refused",
                    Timing::Synchronous => "synchronous",
                },
            ));
            return Ok(StoreOutput::Complete);
        }
        read_resident_chain(req, &self.identity)
            .map(StoreOutput::Rgba8)
            .ok_or(EncodeStatus::WritebackFailed(
                "draw_vk_gva_resident_readback",
            ))
    }
}

fn report_refusal(req: &DrawEncodeRequest, refusal: &GvaWritebackDecline) -> EncodeStatus {
    crate::observe::Emit::decline("draw_vk_gva_native_store", refusal)
        .field("task", req.task_id)
        .field("pipeline", req.pipeline_ref)
        .field(
            "texture",
            req.colors
                .first()
                .map(|color| color.texture_ref)
                .unwrap_or(0),
        )
        .fail();
    EncodeStatus::WritebackFailed(refusal.slug())
}

#[cfg(test)]
mod tests;
