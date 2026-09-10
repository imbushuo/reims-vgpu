//! Graphics texture access, native staging and licensed publication.

use super::*;
use crate::backend::vulkan::engine::graphics_storage::{
    GraphicsStorageDecline as Refused, GraphicsStorageTexture, GraphicsTextureAccess,
    GraphicsTextureBinding,
};
use crate::runtime::compute_exec::{
    stage_texture_raw, writeback_texture, StagedTexture, vulkan::VulkanStage,
};
use crate::runtime::spirv_bind::{
    self, ReflectedTextureAccess, ReflectedTextureDescriptor, ReflectedSampledKind, SampledImageKind,
};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

pub(super) fn used_bindings(words: &[u32]) -> std::sync::Arc<[u32]> {
    spirv_bind::declared_binding_numbers(words).into_iter()
        .filter(|binding| spirv_bind::descriptor_static_use(words, *binding).is_violation())
        .collect::<Vec<_>>().into()
}

struct StorageTexture {
    staged: StagedTexture<VulkanStage>,
    bindings: Vec<GraphicsTextureBinding>,
}

#[derive(Default)]
pub(super) struct StorageTextures {
    textures: BTreeMap<u32, StorageTexture>,
}

fn scalar_2d(
    reflection: &metal2vulkan::reflect::ShaderReflection,
    descriptor: ReflectedTextureDescriptor,
) -> bool {
    descriptor.descriptor_count == 1 && descriptor.array_element == 0
        && spirv_bind::reflected_sampled_kind(reflection, descriptor.binding)
            == ReflectedSampledKind::Kind(SampledImageKind::D2)
}

fn ordinary_single_level(
    entry: &crate::runtime::decode::resource::ListObjectEntry,
    descriptor: &[u8],
) -> bool {
    matches!(entry.object_type, OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS)
        && decode_texture_descriptor(descriptor).ok().is_some_and(|texture| {
            texture.mipmap_level_count == 1 && texture.depth == 1 && texture.sample_count == Some(1)
        })
}

// Reflected imageblock ABIs are refused before staging. Metadata-free private
// slices can still be admitted: those producers must carry their own preserved
// conversion contract, not inherit ordinary AIR texture-write policy.
fn specialize_write_rounding(
    resources: &[GraphicsStorageTexture],
    words: &mut std::sync::Arc<Vec<u32>>,
    stage: ash::vk::ShaderStageFlags,
) -> Result<(), Refused> {
    use metal2vulkan::texture_write_rounding::{
        specialize_texture_write_rounding, TextureWriteFormat, TextureWriteTarget,
    };
    if !resources.iter().any(|texture| texture.bindings.iter().any(|binding| {
        binding.stage == stage && binding.access == GraphicsTextureAccess::Storage
    })) {
        return Ok(());
    }
    // BGRA has no SPIR-V image-format token. Its runtime descriptor supplies the
    // normalized target fact; every other admitted format is now in TypeImage.
    let targets: Vec<_> = resources.iter()
        .filter(|texture| texture.format == crate::backend::vulkan::engine::StorageImageFormat::Bgra8Unorm)
        .flat_map(|texture| &texture.bindings)
        .filter(|binding| binding.stage == stage && binding.access == GraphicsTextureAccess::Storage)
        .map(|binding| TextureWriteTarget {
            descriptor_set: 0, binding: binding.binding, format: TextureWriteFormat::Normalized,
        }).collect();
    *words = std::sync::Arc::new(specialize_texture_write_rounding(
        words, crate::runtime::m2v_cache::NATIVE_TEXTURE_WRITE_ROUNDING, &targets,
    ).map_err(|detail| Refused::WriteRounding { stage, detail })?);
    Ok(())
}

impl StorageTextures {
    pub(super) fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }
    pub(super) fn stage<M: HostMemory + HostOps>(
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        vertex: &metal2vulkan::reflect::ShaderReflection,
        fragment: &metal2vulkan::reflect::ShaderReflection,
    ) -> Result<Self, DrawError> {
        let mut result = Self::default();
        for (reflection, textures) in [
            (vertex, &req.vertex_textures), (fragment, &req.fragment_textures),
        ] {
            for resource in &reflection.bindings {
                if !spirv_bind::is_texture_kind(resource.kind) {
                    continue;
                }

                let index = resource.metal_index;
                let descriptor = spirv_bind::reflected_texture_descriptor(reflection, index)
                    .ok_or(Refused::Reflection { index })?;
                match descriptor.access {
                    ReflectedTextureAccess::Sampled => continue,
                    ReflectedTextureAccess::Unknown => return Err(Refused::Reflection { index }.into()),
                    ReflectedTextureAccess::Storage => {}
                }
                let texture_ref = textures.iter().find(|t| t.index == index)
                    .map(|t| t.texture_ref).filter(|r| *r != 0)
                    .ok_or(Refused::MissingTexture { texture_ref: 0 })?;
                if !scalar_2d(reflection, descriptor) {
                    return Err(Refused::Shape { texture_ref }.into());
                }
                result.add(state, host, req, texture_ref, descriptor.binding)?;
            }
        }
        Ok(result)
    }

    fn add<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        texture_ref: u32,
        binding: u32,
    ) -> Result<(), DrawError> {
        if self.textures.contains_key(&texture_ref) {
            return Ok(());
        }
        if req.colors.iter().any(|c| {
            c.texture_ref == texture_ref || c.multisample_source_ref == texture_ref
        }) || req.depth_attach.as_ref().is_some_and(|a| {
            a.texture_ref == texture_ref || a.resolve_texture_ref == texture_ref
        }) || req.stencil_attach.as_ref().is_some_and(|a| {
            a.texture_ref == texture_ref || a.resolve_texture_ref == texture_ref
        }) {
            return Err(Refused::AttachmentAlias { texture_ref }.into());
        }
        let entry = objects::lookup_list_entry(state, host, req.task_id, texture_ref)
            .ok_or(Refused::MissingTexture { texture_ref })?;
        let descriptor = objects::read_descriptor(state, host, req.task_id, &entry)
            .ok_or(Refused::MissingTexture { texture_ref })?;
        if !ordinary_single_level(&entry, &descriptor) {
            return Err(Refused::Shape { texture_ref }.into());
        }
        let staged = stage_texture_raw::<VulkanStage, _>(
            state, host, req.task_id, texture_ref, binding, true,
        ).map_err(|status| Refused::Staging(Box::new(status)))?;
        if staged.storage_selector.is_none() {
            return Err(Refused::Format { texture_ref }.into());
        }
        self.textures.insert(texture_ref, StorageTexture { staged, bindings: Vec::new() });
        Ok(())
    }

    /// Called before ordinary sampled resolution so a read alias reuses the
    /// writable image rather than taking an independent pre-draw snapshot.
    pub(super) fn bind(
        &mut self,
        texture_ref: u32,
        index: u32,
        reflection: &metal2vulkan::reflect::ShaderReflection,
        binding: u32,
        fragment: bool,
    ) -> Result<bool, DrawError> {
        let Some(texture) = self.textures.get_mut(&texture_ref) else { return Ok(false) };
        let Some(descriptor) = spirv_bind::reflected_texture_descriptor(reflection, index) else {
            // Sticky unused argument-table slots have no shader descriptor.
            return Ok(true);
        };
        if !scalar_2d(reflection, descriptor) {
            return Err(Refused::Shape { texture_ref }.into());
        }
        let access = match descriptor.access {
            ReflectedTextureAccess::Sampled => GraphicsTextureAccess::Sampled,
            ReflectedTextureAccess::Storage => GraphicsTextureAccess::Storage,
            ReflectedTextureAccess::Unknown => return Err(Refused::Reflection { index }.into()),
        };
        texture.bindings.push(GraphicsTextureBinding {
            binding, access,
            stage: if fragment { ash::vk::ShaderStageFlags::FRAGMENT } else { ash::vk::ShaderStageFlags::VERTEX },
        });
        Ok(true)
    }

    pub(super) fn resources(
        &mut self,
        vertex: &metal2vulkan::reflect::ShaderReflection,
        fragment: &metal2vulkan::reflect::ShaderReflection,
        vertex_words: &mut std::sync::Arc<Vec<u32>>,
        fragment_words: &mut std::sync::Arc<Vec<u32>>,
    ) -> Result<Vec<GraphicsStorageTexture>, DrawError> {
        let mut resources = Vec::with_capacity(self.textures.len());
        for (&texture_ref, texture) in &mut self.textures {
            let selector = texture.staged.storage_selector.ok_or(Refused::Format { texture_ref })?;
            let format = translate::pixel::storage_image_from_selector(selector);
            for binding in &texture.bindings {
                if binding.access != GraphicsTextureAccess::Storage { continue }
                let is_fragment = binding.stage == ash::vk::ShaderStageFlags::FRAGMENT;
                let raw_binding = binding.binding - if is_fragment {
                    spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
                } else { 0 };
                let reflection = if is_fragment { fragment } else { vertex };
                let declared = spirv_bind::reflected_storage_image_format(reflection, raw_binding)
                    .ok_or(Refused::Format { texture_ref })?;
                let specialized = crate::runtime::compute_exec::vulkan::specialized_storage_image_format(
                    format, declared,
                    crate::backend::vulkan::engine::supports_storage_image_write_without_format(),
                ).map_err(|_| Refused::Format { texture_ref })?;
                // Never take compute's legacy degraded BGRA/RGBA reinterpretation.
                if specialized != spirv_bind::ImageFormat::Unknown
                    && crate::runtime::compute_exec::vulkan::spirv_image_format_to_engine_storage(specialized)
                        != Some(format)
                {
                    return Err(Refused::Format { texture_ref }.into());
                }
                let words = std::sync::Arc::make_mut(if is_fragment {
                    &mut *fragment_words
                } else { &mut *vertex_words });
                spirv_bind::specialize_image_formats(words, &[(binding.binding, specialized)])
                    .map_err(|_| Refused::Specialization)?;
            }
            resources.push(GraphicsStorageTexture {
                format, width: texture.staged.width, height: texture.staged.height,
                bytes: std::mem::take(&mut texture.staged.bytes),
                bindings: texture.bindings.clone(),
            });
        }
        specialize_write_rounding(&resources, vertex_words, ash::vk::ShaderStageFlags::VERTEX)?;
        specialize_write_rounding(&resources, fragment_words, ash::vk::ShaderStageFlags::FRAGMENT)?;
        Ok(resources)
    }

    pub(super) fn publish<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        outputs: Vec<Vec<u8>>,
    ) -> Result<(), DrawError> {
        if outputs.len() != self.textures.len() {
            return Err(Refused::Output.into());
        }
        for (texture, bytes) in self.textures.values_mut().zip(outputs) {
            let expected = pixel_format::tight_row_bytes(texture.staged.width, texture.staged.pixel_format)
                .and_then(|row| (row as usize).checked_mul(texture.staged.height as usize));
            if expected != Some(bytes.len()) {
                return Err(Refused::Output.into());
            }
            texture.staged.bytes = bytes;
            writeback_texture(state, host, task_id, &texture.staged)
                .map_err(|status| Refused::Publication(Box::new(status)))?;
        }
        Ok(())
    }
}
