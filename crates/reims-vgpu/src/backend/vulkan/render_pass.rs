//! One guest encoder's native-format memoryless color allocations.
//!
//! Vulkan submissions may split that encoder. Device-local attachments with
//! native format conversion and Store/Load at each split preserve precisely
//! the pass-local texels without ever fabricating guest backing or RGBA seeds.

use super::engine::{pass_local::PassLocalTarget, TargetIdentity};
use crate::model::DeviceState;
use crate::runtime::draw::{ColorRtRequest, ColorStorage, DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};
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
        Self { slot: color.slot, texture_ref: color.texture_ref,
            width: color.width, height: color.height, format: color.format }
    }
}

struct Target {
    attachment: Attachment,
    native: PassLocalTarget,
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
pub struct VulkanRenderPass {
    phase: Phase,
    targets: Vec<Target>,
}

impl VulkanRenderPass {
    fn prepare(&mut self, request: &DrawEncodeRequest) -> Result<(), &'static str> {
        match self.phase {
            Phase::New if !request.continues_render_pass => {}
            Phase::Open { task_id } if request.continues_render_pass && task_id == request.task_id => {}
            _ => return Err("draw_vk_render_pass_sequence"),
        }
        let initial = self.phase == Phase::New;
        let colors: Vec<_> = request.colors.iter()
            .filter(|color| color.storage == ColorStorage::Memoryless).collect();
        if !colors.is_empty() && request.colors.first().is_some_and(|color| color.slot != 0) {
            return Err("draw_vk_memoryless_primary_slot");
        }
        for color in &colors {
            if color.texture_ref == 0 { return Err("draw_vk_memoryless_texture_ref"); }
            if request.colors.iter().filter(|prior| prior.slot == color.slot).count() != 1 {
                return Err("draw_vk_memoryless_duplicate_slot");
            }
            if request.colors.iter().filter(|prior| prior.texture_ref == color.texture_ref).count() != 1 {
                return Err("draw_vk_memoryless_attachment_alias");
            }
            if color.sample_count != 1 || color.multisample_source_ref != 0 {
                return Err("draw_vk_memoryless_multisample");
            }
            if color.width == 0 || color.height == 0 {
                return Err("draw_vk_memoryless_pass_geometry");
            }
            super::translate::pixel::memoryless_color_attachment(color.format)
                .map_err(|_| "draw_vk_memoryless_pass_format")?;
            if color.mapping_id != 0 || color.target_gva != 0 || color.target_seed_rgba.is_some()
                || color.store_action != MTL_STORE_ACTION_DONT_CARE
            {
                return Err("draw_vk_memoryless_backing");
            }
            let valid_load = if initial {
                matches!(color.load_action, MTL_LOAD_ACTION_CLEAR | MTL_LOAD_ACTION_DONT_CARE)
            } else {
                color.load_action == MTL_LOAD_ACTION_LOAD
            };
            if !valid_load { return Err("draw_vk_memoryless_pass_load"); }
        }
        if initial {
            for color in colors {
                let format = super::translate::pixel::memoryless_color_attachment(color.format)
                    .map_err(|_| "draw_vk_memoryless_pass_format")?;
                self.targets.push(Target {
                    attachment: Attachment::from(color),
                    native: PassLocalTarget::new(color.width, color.height, format.vk)?,
                });
            }
        } else if colors.len() != self.targets.len() || colors.iter().any(|color| {
            !self.targets.iter().any(|target| target.attachment == Attachment::from(*color))
        }) {
            return Err("draw_vk_memoryless_pass_attachment_changed");
        }
        self.phase = Phase::Open { task_id: request.task_id };
        Ok(())
    }

    pub(crate) fn identity(&self, color: &ColorRtRequest) -> Option<&TargetIdentity> {
        self.targets.iter().find(|target| target.attachment == Attachment::from(color))
            .map(|target| target.native.identity())
    }

    pub fn encode_draw<M: HostMemory + HostOps>(
        &mut self, state: &mut DeviceState, host: &mut M, request: &mut DrawEncodeRequest,
        writeback_guest: bool, force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        self.with_draw(request, |pass, request| {
            crate::runtime::draw::vulkan::encode_draw_in_pass(
                state, host, request, writeback_guest, force_full_store, pass,
            )
        })
    }

    fn with_draw(
        &mut self, request: &mut DrawEncodeRequest,
        encode: impl FnOnce(&Self, &mut DrawEncodeRequest) -> (EncodeStatus, Option<Vec<u8>>),
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        if let Err(reason) = self.prepare(request) {
            self.targets.clear();
            self.phase = Phase::Failed;
            return (EncodeStatus::BadArgs(reason), None);
        }
        let mut result = encode(self, request);
        if matches!(result.0, EncodeStatus::Ok) {
            if request.colors.first().is_some_and(|color| color.storage == ColorStorage::Memoryless) {
                request.chain_resident_established = request.render_pass_continues;
                result.1 = None;
            }
            if !request.render_pass_continues {
                self.targets.clear();
                self.phase = Phase::Finished;
            }
        } else {
            self.targets.clear();
            self.phase = Phase::Failed;
        }
        result
    }
}

#[cfg(test)]
mod tests;
