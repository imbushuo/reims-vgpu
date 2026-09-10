//! Draw-local native storage images. A physical image owns all of its aliases
//! and its seeded bytes; every draw completes and returns these bytes to its
//! licensed publisher, independently of attachment Store.

use ash::vk;
use super::caches::BindingSig;
use super::context::DeviceContext;
use super::counters::EngineCounters;
use super::pools::{BufferSlot, PushDescriptorBinding, ResourcePools, StorageImageKey, StorageImageSlot};
use super::types::{DrawError, DrawRequest, StorageImageFormat};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphicsTextureAccess {
    Sampled,
    Storage,
}

impl GraphicsTextureAccess {
    pub(crate) fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::Sampled => vk::DescriptorType::SAMPLED_IMAGE,
            Self::Storage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GraphicsTextureBinding {
    pub binding: u32,
    pub access: GraphicsTextureAccess,
    pub stage: vk::ShaderStageFlags,
}

/// One texture, not one descriptor: same-ref aliases across slots and stages
/// must observe one image rather than independent seeded snapshots.
#[derive(Debug)]
pub struct GraphicsStorageTexture {
    pub format: StorageImageFormat,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub bindings: Vec<GraphicsTextureBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsStorageDecline {
    MissingTexture { texture_ref: u32 },
    AttachmentAlias { texture_ref: u32 },
    Shape { texture_ref: u32 },
    Reflection { index: u32 },
    Format { texture_ref: u32 },
    Specialization,
    Staging(Box<crate::runtime::compute_exec::ComputeStatus>),
    Publication(Box<crate::runtime::compute_exec::ComputeStatus>),
    Output,
    Geometry,
    EmptyBindings,
    DuplicateBinding { binding: u32 },
    StageUnsupported,
    PixelInterlockUnsupported,
    FormatUnsupported { format: vk::Format },
}

impl std::fmt::Display for GraphicsStorageDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl crate::observe::Decline for GraphicsStorageDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::MissingTexture { .. } => "draw_vk_storage_missing_texture",
            Self::AttachmentAlias { .. } => "draw_vk_storage_attachment_alias",
            Self::Shape { .. } => "draw_vk_storage_texture_shape",
            Self::Reflection { .. } => "draw_vk_storage_reflection",
            Self::Format { .. } => "draw_vk_storage_texture_format",
            Self::Specialization => "draw_vk_storage_specialization",
            Self::Staging(_) => "draw_vk_storage_staging",
            Self::Publication(_) => "draw_vk_storage_publication",
            Self::Output => "draw_vk_storage_output",
            Self::Geometry => "draw_vk_storage_geometry",
            Self::EmptyBindings => "draw_vk_storage_empty_bindings",
            Self::DuplicateBinding { .. } => "draw_vk_storage_duplicate_binding",
            Self::StageUnsupported => "draw_vk_storage_stage_unsupported",
            Self::PixelInterlockUnsupported => "draw_vk_storage_pixel_interlock_unsupported",
            Self::FormatUnsupported { .. } => "draw_vk_storage_format_unsupported",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("detail", format!("{self:?}"))]
    }
}

impl From<GraphicsStorageDecline> for DrawError {
    fn from(reason: GraphicsStorageDecline) -> Self {
        Self::GraphicsStorage(reason)
    }
}

pub(super) fn requires_completion(req: &DrawRequest) -> bool {
    !req.storage_textures.is_empty()
}

pub(super) fn validate(req: &DrawRequest) -> Result<(), DrawError> {
    let mut bindings = std::collections::BTreeSet::new();
    bindings.extend(req.storage_buffers.iter().map(|b| b.binding));
    bindings.extend(req.sampled_images.iter().map(|b| b.binding));
    bindings.extend(req.samplers.iter().map(|b| b.binding));
    for slot in 0..8u32 {
        if req.color_input & (1 << slot) != 0 {
            bindings.insert(super::types::COLOR_INPUT_BINDING + slot);
        }
    }
    for texture in &req.storage_textures {
        let length = (texture.width as usize)
            .checked_mul(texture.height as usize)
            .and_then(|n| n.checked_mul(texture.format.bytes_per_texel()));
        if texture.width == 0 || texture.height == 0 || length != Some(texture.bytes.len()) {
            return Err(GraphicsStorageDecline::Geometry.into());
        }
        if !texture.bindings.iter().any(|b| b.access == GraphicsTextureAccess::Storage) {
            return Err(GraphicsStorageDecline::EmptyBindings.into());
        }
        for binding in &texture.bindings {
            if binding.stage != vk::ShaderStageFlags::VERTEX
                && binding.stage != vk::ShaderStageFlags::FRAGMENT
            {
                return Err(GraphicsStorageDecline::StageUnsupported.into());
            }
            if !bindings.insert(binding.binding) {
                return Err(GraphicsStorageDecline::DuplicateBinding {
                    binding: binding.binding,
                }.into());
            }
        }
    }
    Ok(())
}

pub(super) fn layout_bindings(req: &DrawRequest, out: &mut Vec<BindingSig>) {
    for texture in &req.storage_textures {
        for binding in &texture.bindings {
            out.push(BindingSig {
                binding: binding.binding,
                ty: binding.access.descriptor_type().as_raw() as u32,
                stages: binding.stage.as_raw(),
                count: 1,
            });
        }
    }
}

pub(super) struct PreparedTexture {
    image: StorageImageSlot,
    seed: BufferSlot,
    readback: BufferSlot,
    len: u64,
}

pub(super) unsafe fn prepare(
    ctx: &DeviceContext,
    pools: &mut ResourcePools,
    counters: &EngineCounters,
    req: &DrawRequest,
) -> Result<Vec<PreparedTexture>, DrawError> {
    let mut prepared = Vec::with_capacity(req.storage_textures.len());
    for texture in &req.storage_textures {
        for binding in &texture.bindings {
            if binding.access == GraphicsTextureAccess::Storage
                && ((binding.stage == vk::ShaderStageFlags::VERTEX
                    && !ctx.features.vertex_pipeline_stores_and_atomics)
                    || (binding.stage == vk::ShaderStageFlags::FRAGMENT
                        && !ctx.features.fragment_stores_and_atomics))
            {
                return Err(GraphicsStorageDecline::StageUnsupported.into());
            }
        }
        let format = texture.format.vk_format();
        let properties = unsafe {
            ctx.instance.get_physical_device_format_properties(ctx.pd, format)
        };
        let required = vk::FormatFeatureFlags::STORAGE_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        if !properties.optimal_tiling_features.contains(required) {
            return Err(GraphicsStorageDecline::FormatUnsupported { format }.into());
        }
        let key = StorageImageKey {
            sampled_alias: true,
            width: texture.width, height: texture.height, format: texture.format,
            sampled_only: false, mip_levels: 1,
        };
        let len = texture.bytes.len() as u64;
        let image = unsafe { pools.acquire_storage_image(ctx, key, counters)? };
        let seed = unsafe { pools.acquire_staging(ctx, len, counters)? };
        unsafe { pools.write_staging(ctx, &seed, &texture.bytes)? };
        let readback = unsafe { pools.acquire_readback_extra(ctx, len, counters)? };
        prepared.push(PreparedTexture { image, seed, readback, len });
    }
    Ok(prepared)
}

pub(super) fn descriptors(
    req: &DrawRequest,
    prepared: &[PreparedTexture],
    out: &mut Vec<PushDescriptorBinding>,
) {
    for (texture, prepared) in req.storage_textures.iter().zip(prepared) {
        for binding in &texture.bindings {
            out.push(PushDescriptorBinding::Image {
                binding: binding.binding, array_element: 0,
                ty: binding.access.descriptor_type(), sampler: vk::Sampler::null(),
                view: prepared.image.view, layout: vk::ImageLayout::GENERAL,
            });
        }
    }
}

fn graphics_stages() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
}

impl PreparedTexture {
    fn copy_region(&self) -> [vk::BufferImageCopy; 1] {
        [vk::BufferImageCopy::default()
            .image_subresource(super::color_subresource_layers())
            .image_extent(vk::Extent3D {
                width: self.image.key.width, height: self.image.key.height, depth: 1,
            })]
    }

}

impl PreparedTexture {
    pub(super) unsafe fn seed(&self, ctx: &DeviceContext, cb: vk::CommandBuffer) {
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(self.image.image).subresource_range(super::color_subresource_range())],
            );
            ctx.device.cmd_copy_buffer_to_image(
                cb, self.seed.buffer, self.image.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &self.copy_region(),
            );
            ctx.device.cmd_pipeline_barrier(
                cb, vk::PipelineStageFlags::TRANSFER, graphics_stages(),
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.image.image).subresource_range(super::color_subresource_range())],
            );
        }
    }

    pub(super) unsafe fn copy_output(&self, ctx: &DeviceContext, cb: vk::CommandBuffer) {
        unsafe {
            ctx.device.cmd_pipeline_barrier(
                cb, graphics_stages(), vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.image.image).subresource_range(super::color_subresource_range())],
            );
            ctx.device.cmd_copy_image_to_buffer(
                cb, self.image.image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.buffer, &self.copy_region(),
            );
            ctx.device.cmd_pipeline_barrier(
                cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)], &[], &[],
            );
        }
    }

    /// The owning draw has waited its fence; the pool retains all slots until
    /// that submission retires, including on a timeout or device-loss return.
    pub(super) unsafe fn read(&self, ctx: &DeviceContext) -> Result<Vec<u8>, DrawError> {
        unsafe {
            super::pools::read_back_slot(
                ctx, &self.readback, self.len,
                super::vk_call::VkOp::ExecMapReadback,
                super::vk_call::VkOp::ExecInvalidateReadback,
            )
        }
    }
}
