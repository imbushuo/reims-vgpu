//! Each secondary attachment owns its Store source and pre-submit destination.

use super::*;
use crate::backend::vulkan::engine::{SecondaryColorTarget, TargetIdentity};
use crate::observe::Decline;
use reims_vgpu_protocol::pass_action::{
    is_declared_store_action, store_action_publishes_single_sample, MTL_STORE_ACTION_STORE,
};

enum Destination {
    Mapping(crate::runtime::render_writeback::vulkan::SecondaryMappingStore),
    Gva(StoreTargetPages),
}

struct Store {
    color: ColorRtRequest,
    identity: TargetIdentity,
    destination: Destination,
}

#[derive(Default)]
pub(super) struct Stores {
    task_id: u32,
    stores: Vec<Store>,
}

#[derive(Debug)]
pub(super) struct Refusal {
    slot: u32,
    texture_ref: u32,
    reason: &'static str,
}

impl Refusal {
    fn at(color: &ColorRtRequest, reason: &'static str) -> Self {
        Self { slot: color.slot, texture_ref: color.texture_ref, reason }
    }
}

impl Decline for Refusal {
    fn slug(&self) -> &'static str { self.reason }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("slot", self.slot.to_string()), ("texture_ref", self.texture_ref.to_string())]
    }
}

impl Stores {
    pub(super) fn capture<M: HostMemory>(
        state: &DeviceState,
        host: &M,
        task_id: u32,
        colors: &[ColorRtRequest],
        targets: &[SecondaryColorTarget],
        writeback_guest: bool,
    ) -> Result<Self, Refusal> {
        let mut result = Self { task_id, stores: Vec::new() };
        if !writeback_guest {
            return Ok(result);
        }
        for (index, color) in colors.iter().enumerate().skip(1) {
            if !is_declared_store_action(color.store_action) {
                return Err(Refusal::at(color, "draw_vk_secondary_store_action"));
            }
            if !store_action_publishes_single_sample(color.store_action) {
                continue;
            }
            if color.storage == ColorStorage::Memoryless {
                return Err(Refusal::at(color, "draw_vk_secondary_store_memoryless"));
            }
            if color.sample_count != 1 || color.multisample_source_ref != 0
                || color.store_action != MTL_STORE_ACTION_STORE
            {
                return Err(Refusal::at(color, "draw_vk_secondary_store_multisample"));
            }
            let target = targets.get(index - 1)
                .filter(|target| target.width == color.width && target.height == color.height)
                .ok_or_else(|| Refusal::at(color, "draw_vk_secondary_store_identity"))?;
            let names_destination = match &target.identity {
                TargetIdentity::Gva { gva, .. } => *gva == color.target_gva && color.mapping_id == 0,
                TargetIdentity::Surface { id, .. } => *id == color.mapping_id && color.target_gva == 0,
                _ => false,
            };
            if !names_destination {
                return Err(Refusal::at(color, "draw_vk_secondary_store_identity"));
            }
            let destination = if color.target_gva != 0 && color.mapping_id == 0 {
                let pages = sync_store_target_pages(state, host, task_id, color)
                    .filter(|pages| pages.ordered_complete(color.target_gva, state.page_size()).is_some())
                    .ok_or_else(|| Refusal::at(color, "draw_vk_secondary_store_pages"))?;
                Destination::Gva(pages)
            } else if color.mapping_id != 0 && color.target_gva == 0 {
                let mapping = crate::runtime::render_writeback::vulkan::SecondaryMappingStore::capture(
                    state, color,
                )
                    .ok_or_else(|| Refusal::at(color, "draw_vk_secondary_store_mapping"))?;
                Destination::Mapping(mapping)
            } else {
                return Err(Refusal::at(color, "draw_vk_secondary_store_destination"));
            };
            result.stores.push(Store {
                color: color.clone(),
                identity: target.identity.clone(),
                destination,
            });
        }
        Ok(result)
    }

    pub(super) fn publish<M: HostMemory + HostOps>(
        self,
        state: &mut DeviceState,
        host: &mut M,
    ) -> Result<(), Refusal> {
        self.publish_with(state, host, |state, host, task_id, store| {
            let color = &store.color;
            match &store.destination {
                Destination::Mapping(mapping) => mapping.publish(state, host, &store.identity),
                Destination::Gva(pages) => {
                    let result = crate::runtime::render_writeback::vulkan::store_gva_frame(
                        state, host, task_id, &store.identity, color, color.texture_ref,
                        Some(pages), &[],
                    );
                    if let Err(refusal) = &result {
                        crate::observe::Emit::decline("draw_vk_secondary_gva_store", refusal)
                            .field("slot", color.slot).field("texture_ref", color.texture_ref).fail();
                    }
                    result.is_ok()
                }
            }
        })
    }

    fn publish_with<M: HostMemory + HostOps>(
        self,
        state: &mut DeviceState,
        host: &mut M,
        mut transfer: impl FnMut(&mut DeviceState, &mut M, u32, &Store) -> bool,
    ) -> Result<(), Refusal> {
        // Refuse a moved destination before publishing any of this draw's Stores.
        for store in &self.stores {
            if let Destination::Mapping(mapping) = &store.destination {
                if !mapping.current(state) {
                    return Err(Refusal::at(&store.color, "draw_vk_secondary_store_mapping_moved"));
                }
            }
        }
        for store in &self.stores {
            if !transfer(state, host, self.task_id, store) {
                return Err(Refusal::at(&store.color, "draw_vk_secondary_store_transfer"));
            }
            if store.color.mapping_id != 0 {
                publish_surface_store(
                    state, host, store.color.mapping_id,
                    store.color.width, store.color.height, store.color.format,
                );
            }
            crate::runtime::drain::note_store_route("mrt_secondary_store_published");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
