//! Draw-owned writable textures, published after native completion, not at pass Store.

use super::*;
use crate::backend::metal::abi::{ReimsVgpuSampledImage, REIMS_VGPU_BINDING_TEXTURE_BASE};
use crate::backend::metal::{compute::upload_storage_texture, runtime::system_device};
use crate::runtime::compute_exec::{
    metal::MetalStage, stage_texture_raw, writeback_texture, ComputeStatus, StagedTexture,
};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

struct StorageTexture {
    staged: StagedTexture<MetalStage>,
    texture: ::metal::Texture,
    bytes_per_row: u32,
}

#[derive(Default)]
pub(super) struct StorageTextures {
    textures: BTreeMap<u32, StorageTexture>,
}

fn draw_status(status: ComputeStatus) -> EncodeStatus {
    match status {
        ComputeStatus::RailRefused(reason) => EncodeStatus::RailRefused(reason),
        ComputeStatus::MissingPipeline(reason) => EncodeStatus::MissingPipeline(reason),
        ComputeStatus::MissingMtlb(reason) => EncodeStatus::MissingMtlb(reason),
        ComputeStatus::MissingBuffer(reason)
        | ComputeStatus::MissingTexture(reason)
        | ComputeStatus::MissingSampler(reason)
        | ComputeStatus::BadGrid(reason) => EncodeStatus::BadArgs(reason),
        ComputeStatus::GuestIo(reason) => EncodeStatus::WritebackFailed(reason),
        ComputeStatus::MetalFailed(reason) => EncodeStatus::MetalFailed(reason),
        ComputeStatus::NoMetal(reason) => EncodeStatus::NoMetal(reason),
        ComputeStatus::Unsupported(reason) => EncodeStatus::Unsupported(reason),
        ComputeStatus::Ok => EncodeStatus::BadArgs("draw_mtl_storage_invalid_status"),
    }
}

impl StorageTextures {
    pub(super) fn add<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        texture_ref: u32,
        index: u32,
    ) -> Result<(), EncodeStatus> {
        if self.textures.contains_key(&texture_ref) {
            return Ok(());
        }

        if req.colors.iter().any(|color| color.texture_ref == texture_ref)
            || req.depth_attach.as_ref().is_some_and(|a| {
                a.texture_ref == texture_ref || a.resolve_texture_ref == texture_ref
            })
            || req.stencil_attach.as_ref().is_some_and(|a| {
                a.texture_ref == texture_ref || a.resolve_texture_ref == texture_ref
            })
        {
            return Err(EncodeStatus::Unsupported("draw_mtl_storage_attachment_alias"));
        }
        let entry = objects::lookup_list_entry(state, host, req.task_id, texture_ref)
            .ok_or(EncodeStatus::BadArgs("draw_mtl_storage_missing_texture"))?;
        // Raw storage staging currently exposes one level. Do not flatten a
        // writable pyramid, array, or unresolved view into an independent D2.
        let single_level = matches!(
            entry.object_type, OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS
        ) && objects::read_descriptor(state, host, req.task_id, &entry)
            .and_then(|bytes| decode_texture_descriptor(&bytes).ok())
            .is_some_and(|texture| {
                texture.mipmap_level_count == 1
                    && texture.depth == 1
                    && texture.sample_count == Some(1)
            });
        if !single_level {
            return Err(EncodeStatus::Unsupported("draw_mtl_storage_texture_shape"));
        }
        let staged = stage_texture_raw::<MetalStage, _>(
            state, host, req.task_id, texture_ref, REIMS_VGPU_BINDING_TEXTURE_BASE + index, true,
        ).map_err(draw_status)?;
        let selector = staged.storage_selector_or_refuse(req.task_id, req.pipeline_ref)
            .map_err(draw_status)?;
        let bytes_per_row = pixel_format::tight_row_bytes(staged.width, staged.pixel_format)
            .ok_or(EncodeStatus::BadArgs("draw_mtl_storage_texture_pitch"))?;
        let device = system_device().ok_or(EncodeStatus::NoMetal("draw_mtl_storage_device"))?;
        let mut err = [0i8; 256];
        let texture = upload_storage_texture(
            device, selector, staged.width, staged.height, &staged.bytes,
            (err.as_mut_ptr(), err.len()),
        ).map_err(EncodeStatus::RailRefused)?;
        crate::runtime::drain::note_store_route("metal_storage_binds");
        crate::runtime::drain::note_store_route_n("metal_storage_bytes", staged.bytes.len() as u64);
        self.textures.insert(texture_ref, StorageTexture { staged, texture, bytes_per_row });
        Ok(())
    }

    pub(super) fn image(&self, texture_ref: u32, index: u32) -> Option<ReimsVgpuSampledImage> {
        self.textures.get(&texture_ref).map(|storage| ReimsVgpuSampledImage::Native {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + index,
            texture: storage.texture.clone(),
        })
    }

    /// Shader side effects are visible to the next draw even when color Store
    /// is deferred. Seeding every texel also preserves regions not written.
    pub(super) fn publish<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
    ) -> Result<(), EncodeStatus> {
        for storage in self.textures.values_mut() {
            storage.texture.get_bytes(
                storage.staged.bytes.as_mut_ptr().cast(),
                u64::from(storage.bytes_per_row),
                ::metal::MTLRegion::new_2d(
                    0, 0, u64::from(storage.staged.width), u64::from(storage.staged.height),
                ),
                0,
            );
            writeback_texture(state, host, task_id, &storage.staged).map_err(draw_status)?;
        }
        Ok(())
    }
}
