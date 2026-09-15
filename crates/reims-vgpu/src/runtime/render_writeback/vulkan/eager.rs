//! An eager Store owns native CPU pixels before acquiring its live GVA writer.
//! A pre-render page snapshot is a bound, not an asynchronous physical-write lease.

use super::*;
use crate::backend::vulkan::engine::{self, TargetIdentity};
use crate::model::TaskResource;
use crate::runtime::draw::{ColorRtRequest, StoreTargetPages};
use crate::runtime::objects;
use crate::runtime::writeback_debt::GvaResourceKey;
use std::sync::Arc;

pub(crate) struct NativeEagerStore {
    key: GvaResourceKey,
    color: ColorRtRequest,
    generation: u64,
    span: u64,
    owners: Vec<(u32, Arc<TaskResource>)>,
    backing: reims_vgpu_core::access::BackingId,
}

impl NativeEagerStore {
    pub(crate) fn capture(
        state: &DeviceState,
        task: u32,
        color: &ColorRtRequest,
        generation: u64,
    ) -> Result<Self, GvaWritebackDecline> {
        let (owners, backing) = allocation_owners(state, task, color.texture_ref)?;
        let result = Self {
            key: GvaResourceKey {
                task_id: task,
                texture_ref: color.texture_ref,
            },
            color: ColorRtRequest {
                texture_ref: color.texture_ref,
                target_gva: color.target_gva,
                width: color.width,
                height: color.height,
                row_stride: color.row_stride,
                format: color.format,
                ..Default::default()
            },
            generation,
            span: u64::from(color.row_stride) * u64::from(color.height),
            owners,
            backing,
        };
        result.validate_live(state)?;
        Ok(result)
    }

    pub(crate) fn validate_live(&self, state: &DeviceState) -> Result<(), GvaWritebackDecline> {
        if self.owners.iter().any(|(reference, captured)| {
            !state
                .constructed_object(self.key.task_id, *reference)
                .is_some_and(|owner| Arc::ptr_eq(&owner, captured))
        }) || !state.pending_writebacks.gva_store_generation_is_live(
            self.key,
            self.color.target_gva,
            self.generation,
            self.span,
        ) {
            return Err(GvaWritebackDecline::DestinationRetired);
        }
        let (reference, owner) = self
            .owners
            .last()
            .ok_or(GvaWritebackDecline::DestinationBackingUnknown)?;
        if objects::backing_id_of(
            state,
            self.key.task_id,
            *reference,
            &owner.entry,
            &owner.descriptor,
        )
        .ok()
            != Some(self.backing)
        {
            return Err(GvaWritebackDecline::DestinationBackingChanged);
        }

        Ok(())
    }

    fn validate_pages<M: HostMemory>(
        &self,
        state: &DeviceState,
        host: &M,
        pages: &StoreTargetPages,
    ) -> Result<(), GvaWritebackDecline> {
        let wanted = pages
            .ordered_complete(self.color.target_gva, state.page_size())
            .ok_or(GvaWritebackDecline::SpanIncomplete)?;
        let current = StoreTargetPages::capture(
            state,
            host,
            self.key.task_id,
            self.color.target_gva,
            self.span,
        );
        if current.ordered_complete(self.color.target_gva, state.page_size()) != Some(wanted) {
            return Err(GvaWritebackDecline::DestinationPagesChanged);
        }
        Ok(())
    }

    pub(crate) fn publish<M: HostMemory + HostOps>(
        &self,
        state: &mut DeviceState,
        host: &mut M,
        identity: &TargetIdentity,
        pages: Option<&StoreTargetPages>,
    ) -> Result<u64, GvaWritebackDecline> {
        let pages = pages.ok_or(GvaWritebackDecline::Unlicensed)?;
        self.validate_live(state)?;
        self.validate_pages(state, host, pages)?;
        let native = engine::read_target_native(identity)
            .map_err(|inner| GvaWritebackDecline::CopiedReadRefused { inner })?;
        self.validate_live(state)?;
        self.validate_pages(state, host, pages)?;
        // This writer re-walks and applies the captured page bound to the
        // actual CPU mapping. It releases that mapping after memcpy, and queues
        // no asynchronous guest-memory write that could outlive this authority.
        let extent = land_native_gva_frame(
            state,
            host,
            self.key.task_id,
            &self.color,
            self.key.texture_ref,
            &native,
            pages,
            &[],
        )?;
        engine::note_resident_content_copied_out(identity);
        crate::runtime::drain::note_store_route("gva_eager_copied_native");
        Ok(extent)
    }
}

type AllocationOwners = (
    Vec<(u32, Arc<TaskResource>)>,
    reims_vgpu_core::access::BackingId,
);

fn allocation_owners(
    state: &DeviceState,
    task: u32,
    mut reference: u32,
) -> Result<AllocationOwners, GvaWritebackDecline> {
    use crate::runtime::decode::resource::{
        decode_buffer_texture_descriptor, Descriptor, OBJECT_TYPE_TEXTURE_VIEW,
    };
    let mut owners = Vec::new();
    // Geometry/mip resolution already happened. This walk retains the existing
    // view/buffer owners only, without reinterpreting their texel offsets.
    for _ in 0..=crate::runtime::draw::MAX_TEXTURE_VIEW_CHAIN {
        if reference == 0 || owners.iter().any(|(prior, _)| *prior == reference) {
            break;
        }
        let owner = state
            .constructed_object(task, reference)
            .ok_or(GvaWritebackDecline::DestinationRetired)?;
        if let Ok(backing) =
            objects::backing_id_of(state, task, reference, &owner.entry, &owner.descriptor)
        {
            owners.push((reference, owner));
            return Ok((owners, backing));
        }
        if owner.entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
            break;
        }
        let next = match owner.decoded() {
            Ok(Descriptor::TextureView(view)) => Some(view.base_texture_ref),
            _ => decode_buffer_texture_descriptor(&owner.descriptor)
                .ok()
                .map(|buffer| buffer.buffer_ref),
        };
        owners.push((reference, owner));
        let Some(next) = next else {
            break;
        };
        reference = next;
    }
    Err(GvaWritebackDecline::DestinationBackingUnknown)
}
