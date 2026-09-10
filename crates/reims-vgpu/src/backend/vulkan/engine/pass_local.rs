//! Native render-encoder allocations, never guest resources or CPU seeds.
//!
//! The registry is the engine's image/barrier owner; this lease, not registry
//! policy, owns lifetime. Registry insertion pins a pass allocation immediately.
//! End/refusal/drop releases it through the normal fence-safe retirement path.
//! Unlike guest residents, ended pass images never enter the target recycler:
//! their allocations are released as soon as the commands using them retire.

use std::sync::atomic::{AtomicU64, Ordering};

use super::TargetIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PassLocalId(u64);

impl PassLocalId {
    pub(crate) fn get(self) -> u64 { self.0 }
}

#[derive(Debug)]
pub(crate) struct PassLocalTarget {
    identity: TargetIdentity,
}

impl PassLocalTarget {
    pub(crate) fn new(width: u32, height: u32, format: ash::vk::Format) -> Result<Self, &'static str> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| "draw_vk_memoryless_identity_exhausted")?;
        Ok(Self { identity: TargetIdentity::PassLocal {
            id: PassLocalId(id), width, height, format,
        } })
    }

    pub(crate) fn identity(&self) -> &TargetIdentity { &self.identity }
}

impl Drop for PassLocalTarget {
    fn drop(&mut self) {
        let mut guard = super::lock_engine();
        let super::EngineState { ref owner, ref mut pools, ref counters, .. } = *guard;
        // Do not create a device merely to retire an unallocated attachment.
        if let Some(ctx) = owner.ctx.as_ref() {
            unsafe { pools.release_resident_resource(ctx, &self.identity, counters); }
        }
    }
}

pub(super) unsafe fn validate_formats(
    ctx: &super::context::DeviceContext, req: &super::DrawRequest,
) -> Result<(), super::DrawError> {
    let primary = req.target_identity.as_ref().map(|identity| (identity, req.blend.is_some()));
    for (identity, blend) in primary.into_iter().chain(req.secondary_targets.iter()
        .map(|target| (&target.identity, target.blend.is_some())))
    {
        if !matches!(identity, TargetIdentity::PassLocal { .. }) { continue; }
        let format = identity.resident_format();
        let available = unsafe { ctx.instance.get_physical_device_format_properties(ctx.pd, format) }
            .optimal_tiling_features;
        if !supports_native_attachment(available, blend) {
            return Err(super::DrawError::Unsupported(
                super::reason::DrawReason::PassLocalFormatUnsupported { format, blend },
            ));
        }
    }
    Ok(())
}

fn supports_native_attachment(available: ash::vk::FormatFeatureFlags, blend: bool) -> bool {
    use ash::vk::FormatFeatureFlags as F;
    let required = F::COLOR_ATTACHMENT | F::SAMPLED_IMAGE | F::TRANSFER_SRC | F::TRANSFER_DST
        | if blend { F::COLOR_ATTACHMENT_BLEND } else { F::empty() };
    available.contains(required)
}

#[cfg(test)]
pub(crate) fn read_native_for_test(identity: &TargetIdentity) -> Result<Vec<u8>, super::DrawError> {
    assert!(matches!(identity, TargetIdentity::PassLocal { .. }));
    let mut guard = super::lock_engine();
    let super::EngineState { ref mut owner, ref mut pools, ref counters, .. } = *guard;
    let ctx = owner.ensure(counters)?;
    unsafe { pools.ensure_init(ctx, counters)? };
    let snap = super::resident_read_snapshot(pools, identity)?;
    let bytes = crate::backend::vulkan::translate::pixel::bytes_per_texel(snap.format).ok_or(
        super::DrawError::TargetRead(super::reason::TargetReadDecline::TexelNotFourBytes {
            format: snap.format,
        }),
    )?;
    let access = super::pools::ResidentAccess::transfer_read(false);
    let pixels = unsafe {
        super::copy_image_level0_to_host(
            ctx, pools, counters, snap.image, snap.layout, access.layout(),
            super::RESIDENT_READ_SRC_ACCESS, snap.width, snap.height,
            u64::from(snap.width) * u64::from(snap.height) * u64::from(bytes),
            super::target_readback_ops(),
        )?
    };
    pools.registry_note_access(identity, access);
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memoryless_native_format_capability_never_guesses_blend_support() {
        use ash::vk::FormatFeatureFlags as F;
        let render = F::COLOR_ATTACHMENT | F::SAMPLED_IMAGE | F::TRANSFER_SRC | F::TRANSFER_DST;
        assert!(supports_native_attachment(render, false));
        assert!(!supports_native_attachment(render, true));
        assert!(supports_native_attachment(render | F::COLOR_ATTACHMENT_BLEND, true));
        assert!(!supports_native_attachment(render & !F::COLOR_ATTACHMENT, false));
    }
}
