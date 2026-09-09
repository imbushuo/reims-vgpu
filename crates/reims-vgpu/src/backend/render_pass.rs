//! One guest render encoder's backend-owned lifetime.
//!
//! The stream executor owns this value around its whole draw list, never in a
//! per-draw request or a device cache. Dropping it ends the lifetime even when
//! a draw refuses and the executor abandons the remaining records.

use crate::model::DeviceState;
use crate::runtime::draw::{DrawEncodeRequest, EncodeStatus};
use crate::runtime::host::{HostMemory, HostOps};

pub enum RenderPass {
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    Metal(super::metal::render_pass::MetalRenderPass),
    #[cfg(feature = "backend-vulkan")]
    Vulkan,
}

impl RenderPass {
    pub fn encode_draw<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        request: &mut DrawEncodeRequest,
        writeback_guest: bool,
        force_full_store: bool,
    ) -> (EncodeStatus, Option<Vec<u8>>) {
        match self {
            #[cfg(all(feature = "backend-metal", target_os = "macos"))]
            Self::Metal(pass) => pass.encode_draw(
                state, host, request, writeback_guest, force_full_store,
            ),
            #[cfg(feature = "backend-vulkan")]
            Self::Vulkan => crate::runtime::draw::vulkan::encode_draw_chain(
                state, host, request, writeback_guest, force_full_store,
            ),
        }
    }
}
