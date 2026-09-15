//! Metadata-only native preparation over the model's retained record stream.
//! Unsupported metadata stays on the synchronous rail with an explicit reason.

use super::*;
use crate::backend::vulkan::engine::precreated::{
    BindingSig, Color0Load, PassKey, SecondaryAttachKey,
};
use crate::backend::vulkan::engine::precreated::{Metadata, Progress, SourceInfo};
use crate::backend::vulkan::{engine, pipeline_resolve, translate};
use crate::runtime::decode::resource::{Descriptor, PipelineColorAttachment};
use crate::runtime::spirv_bind;
use reims_vgpu_core::exec::{ExecWork, ResolvedOperation};
use reims_vgpu_core::render::{DrawOp, RenderOp};

#[derive(Debug)]
pub(super) struct Prepared(pub Result<Vec<Metadata>, ScanFailure>);

#[derive(Debug)]
pub(super) struct ScanFailure {
    reason: &'static str,
    pipeline: Option<reims_vgpu_core::identity::ResourceId>,
}

pub(super) fn pending<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    submission: &ExecSubmission,
    resolved: &ExecWork,
) -> bool {
    let prepared = submission
        .native_pipelines
        .get_or_init(|| Prepared(collect(state, host, submission.task_id, resolved)));
    match &prepared.0 {
        Ok(plans) => plans.iter().fold(false, |pending, plan| {
            matches!(
                engine::precreated::preflight(state, plan),
                Progress::Pending
            ) | pending
        }),
        Err(failure) => {
            report(submission.task_id, failure.pipeline, None, failure.reason);
            false
        }
    }
}

fn collect<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    work: &ExecWork,
) -> Result<Vec<Metadata>, ScanFailure> {
    scan(work, |bindings, pass, raster, topology, viewports| {
        draw_metadata(
            state, host, task, bindings, pass, raster, topology, viewports,
        )
    })
}

fn scan<T>(
    work: &ExecWork,
    mut prepare: impl FnMut(
        &reims_vgpu_core::encoder::RenderEncoderState,
        &reims_vgpu_core::pass::PassDescriptor,
        reims_vgpu_vulkan::raster::GuestRasterState,
        reims_vgpu_core::topology::PrimitiveType,
        u32,
    ) -> Result<T, &'static str>,
) -> Result<Vec<T>, ScanFailure> {
    let mut plans = Vec::new();
    for stream in &work.streams {
        let mut bindings = reims_vgpu_core::encoder::RenderEncoderState::default();
        let mut pass = None;
        let mut raster = reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT;
        let (mut viewports, mut scissors) = (1, 1);
        let mut prepared_topology = None;
        for record in &stream.records {
            let ResolvedOperation::Render(op) = &record.op else {
                prepared_topology = None;
                continue;
            };
            if !matches!(op, RenderOp::Draw(_)) {
                prepared_topology = None;
            }
            bindings.apply(
                op,
                &work.arenas.buffer_bindings,
                &work.arenas.object_bindings,
            );
            match op {
                RenderOp::WriteDescriptor { descriptor } => {
                    pass = work.arenas.pass_descriptors.get(descriptor.0 as usize)
                }
                RenderOp::SetViewports(span) => viewports = span.len.max(1),
                RenderOp::SetScissorRects(span) => scissors = span.len.max(1),
                RenderOp::SetCullMode(value) => raster.cull_mode = *value,
                RenderOp::SetFrontFacingWinding(value) => raster.winding = *value,
                RenderOp::SetDepthClipMode(value) => raster.depth_clip_mode = *value,
                RenderOp::SetTriangleFillMode(value) => raster.fill_mode = *value,
                RenderOp::Draw(draw) => {
                    let failure = |reason| ScanFailure {
                        reason,
                        pipeline: bindings.pipeline,
                    };
                    let primitive = match draw {
                        DrawOp::Primitives { primitive, .. }
                        | DrawOp::Indexed { primitive, .. } => primitive.0,
                        _ => return Err(failure("native_preflight_indirect_draw")),
                    };
                    let topology = reims_vgpu_core::topology::PrimitiveType::parse(primitive)
                        .ok_or_else(|| failure("native_preflight_topology"))?;
                    if prepared_topology == Some(topology) {
                        continue;
                    }
                    prepared_topology = Some(topology);
                    if let Ok(plan) = prepare(
                        &bindings,
                        pass.ok_or_else(|| failure("native_preflight_no_pass"))?,
                        raster,
                        topology,
                        viewports.max(scissors),
                    ) {
                        plans.push(plan);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(plans)
}

fn color_blend(color: PipelineColorAttachment) -> Option<engine::types::BlendKey> {
    color.blending_enabled.then(|| {
        engine::BlendStateResource {
            src_rgb: color.src_rgb,
            dst_rgb: color.dst_rgb,
            op_rgb: color.op_rgb,
            src_alpha: color.src_alpha,
            dst_alpha: color.dst_alpha,
            op_alpha: color.op_alpha,
        }
        .key()
    })
}

#[allow(clippy::too_many_arguments)]
fn draw_metadata<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    bindings: &reims_vgpu_core::encoder::RenderEncoderState,
    pass: &reims_vgpu_core::pass::PassDescriptor,
    raster: reims_vgpu_vulkan::raster::GuestRasterState,
    topology: reims_vgpu_core::topology::PrimitiveType,
    viewport_count: u32,
) -> Result<Metadata, &'static str> {
    let mut sources = None;
    let result = (|| {
        let pipeline = bindings.pipeline.ok_or("native_preflight_no_pipeline")?;
        let resolved = pipeline_resolve::resolve(state, host, task, pipeline.slot.0)
            .map_err(|_| "native_preflight_pipeline_unavailable")?;
        let inputs = PipelineInputs::new(state, &resolved, bindings)?;
        sources = Some(inputs.sources.clone());
        draw_metadata_resolved(
            state,
            host,
            task,
            bindings,
            pass,
            raster,
            topology,
            viewport_count,
            &resolved,
            &inputs,
        )
    })();
    report(
        task,
        bindings.pipeline,
        sources.as_ref(),
        result
            .as_ref()
            .err()
            .copied()
            .unwrap_or("native_preflight_eligible"),
    );
    result
}

struct PipelineInputs {
    vertex: Arc<crate::runtime::m2v_cache::ShaderVariant>,
    fragment: Arc<crate::runtime::m2v_cache::ShaderVariant>,
    sources: [Arc<SourceInfo>; 2],
    buf_collide: bool,
    separate: bool,
}

impl PipelineInputs {
    fn new(
        state: &DeviceState,
        resolved: &pipeline_resolve::ResolvedRenderPipeline,
        bindings: &reims_vgpu_core::encoder::RenderEncoderState,
    ) -> Result<Self, &'static str> {
        let buf_collide = bindings
            .vertex
            .buffers
            .bound()
            .filter(|(_, b)| b.buffer.is_some())
            .any(|(i, _)| {
                bindings
                    .fragment
                    .buffers
                    .bound()
                    .any(|(j, b)| i == j && b.buffer.is_some())
            });
        let has_sampled = |stage: &reims_vgpu_core::encoder::StageTables| {
            stage
                .textures
                .bound()
                .chain(stage.samplers.bound())
                .any(|(_, b)| b.object.is_some())
        };
        let separate = (has_sampled(&bindings.vertex) && has_sampled(&bindings.fragment))
            || buf_collide
            || crate::runtime::draw::vulkan::reflected_sampled_binding_collision(
                &resolved.vertex.reflection,
                &resolved.fragment.reflection,
            );
        let vertex = resolved.vertex.variant(false, false);
        let fragment = resolved.fragment.variant(separate, buf_collide);
        let sources = engine::precreated::metadata_sources(state, &vertex.words, &fragment.words)
            .ok_or("native_preflight_caches_unavailable")?;
        Ok(Self {
            vertex,
            fragment,
            sources,
            buf_collide,
            separate,
        })
    }
}

#[derive(Eq, PartialEq, Hash)]
struct DiagnosticKey {
    task: u32,
    pipeline: Option<reims_vgpu_core::identity::ResourceId>,
    sources: Option<[(u64, u64, u64); 2]>,
    reason: &'static str,
}

#[derive(Default)]
struct Diagnostics(std::collections::HashSet<DiagnosticKey>);

impl Diagnostics {
    const LIMIT: usize = 256;

    fn admit(
        &mut self,
        task: u32,
        pipeline: Option<reims_vgpu_core::identity::ResourceId>,
        sources: Option<[(u64, u64, u64); 2]>,
        reason: &'static str,
    ) -> bool {
        self.0.len() < Self::LIMIT
            && self.0.insert(DiagnosticKey {
                task,
                pipeline,
                sources,
                reason,
            })
    }
}

fn report(
    task: u32,
    pipeline: Option<reims_vgpu_core::identity::ResourceId>,
    sources: Option<&[Arc<SourceInfo>; 2]>,
    reason: &'static str,
) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<Diagnostics>> = std::sync::OnceLock::new();
    crate::runtime::drain::note_store_route(reason);
    let admitted = SEEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .admit(
            task,
            pipeline,
            sources.map(|s| {
                s.each_ref().map(|s| {
                    let d = s.digest;
                    (d.a, d.b, d.len)
                })
            }),
            reason,
        );
    if !admitted {
        return;
    }
    let identity = |stage: usize| {
        sources.map_or_else(
            || "unknown".to_owned(),
            |s| {
                let d = s[stage].digest;
                format!("{:x}{:x}:{}", d.a, d.b, d.len)
            },
        )
    };
    crate::observe::fail(format!(
        "native_pipeline_preflight task={task} pipeline={} generation={} sourceVS={} runtimeFS={} reason={reason} route={}",
        pipeline.map_or(0, |p| p.slot.0),
        pipeline.map_or(0, |p| p.generation.0),
        identity(0),
        identity(1),
        if reason == "native_preflight_eligible" {
            "exact_metadata"
        } else {
            "synchronous"
        },
    ));
}

#[allow(clippy::too_many_arguments)]
fn draw_metadata_resolved<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    bindings: &reims_vgpu_core::encoder::RenderEncoderState,
    pass: &reims_vgpu_core::pass::PassDescriptor,
    raster: reims_vgpu_vulkan::raster::GuestRasterState,
    topology: reims_vgpu_core::topology::PrimitiveType,
    viewport_count: u32,
    resolved: &pipeline_resolve::ResolvedRenderPipeline,
    inputs: &PipelineInputs,
) -> Result<Metadata, &'static str> {
    if host.map_pages_stable() {
        return Err("native_preflight_shared_target_placement");
    }
    if viewport_count != 1 {
        return Err("native_preflight_multiview");
    }
    if pass.depth.texture.is_some()
        || pass.stencil.texture.is_some()
        || bindings.depth_stencil.is_some()
    {
        return Err("native_preflight_depth");
    }
    let pd = &resolved.desc;
    let vertex_buffers: Vec<_> = bindings
        .vertex
        .buffers
        .bound()
        .filter_map(|(index, bind)| {
            Some(crate::runtime::draw::BufferBind {
                index,
                buffer_ref: bind.buffer?.slot.0,
                offset: bind.offset,
                attribute_stride: bind.stride,
                resource: None,
            })
        })
        .collect();
    if pd.vertex_attributes.iter().any(|a| {
        crate::runtime::draw::active_vertex_attribute_stride(
            a.format,
            crate::runtime::draw::bind_attribute_stride(&vertex_buffers, a.buffer_index, a.stride),
        )
        .is_some()
    }) {
        return Err("native_preflight_vertex_attributes");
    }
    if pd.raster_sample_count > 1 {
        return Err("native_preflight_multisample");
    }
    let v = &resolved.vertex;
    let f = &resolved.fragment;
    if [&v.reflection, &f.reflection]
        .iter()
        .any(|reflection| spirv_bind::first_unsupported_vulkan_resource(reflection).is_some())
    {
        return Err("native_preflight_resource_interface");
    }
    if v.reflection
        .bindings
        .iter()
        .chain(f.reflection.bindings.iter())
        .any(|binding| {
            matches!(
                binding.kind,
                metal2vulkan::reflect::ResourceKind::StorageImage
            )
        })
        || inputs.fragment.pixel_interlock
    {
        return Err("native_preflight_current_storage_proof");
    }
    let v_buffers: Vec<_> = bindings
        .vertex
        .buffers
        .bound()
        .filter(|(_, b)| b.buffer.is_some())
        .collect();
    let f_buffers: Vec<_> = bindings
        .fragment
        .buffers
        .bound()
        .filter(|(_, b)| b.buffer.is_some())
        .collect();
    for (_, bind) in v_buffers.iter().chain(&f_buffers) {
        let reference = bind.buffer.ok_or("native_preflight_buffer")?.slot.0;
        let (_, size) = objects::resolve_buffer_span(state, host, task, reference)
            .map_err(|_| "native_preflight_buffer")?;
        if bind.offset >= size {
            return Err("native_preflight_buffer_extent");
        }
    }
    let PipelineInputs {
        separate,
        buf_collide,
        ..
    } = inputs;
    let (separate, buf_collide) = (*separate, *buf_collide);
    let mut layout = Vec::new();
    let stages = (ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT).as_raw();
    for (stage, reflection, fragment_stage) in [
        (&bindings.vertex, &v.reflection, false),
        (&bindings.fragment, &f.reflection, true),
    ] {
        let offset = if fragment_stage && separate {
            spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        } else {
            0
        };
        for (index, bind) in stage.buffers.bound().filter(|(_, b)| b.buffer.is_some()) {
            let _ = bind;
            layout.push(BindingSig {
                binding: index
                    .checked_add(if fragment_stage && buf_collide {
                        spirv_bind::FRAG_BUFFER_BINDING_OFFSET
                    } else {
                        0
                    })
                    .ok_or("native_preflight_binding_overflow")?,
                ty: ash::vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                stages,
                count: 1,
            });
        }
        for (index, bind) in stage.textures.bound() {
            let Some(texture) = bind.object else { continue };
            let descriptor = spirv_bind::reflected_texture_descriptor(reflection, index);
            let binding = descriptor
                .map(|d| d.binding)
                .or_else(|| spirv_bind::TEXTURE_BINDING_BASE.checked_add(index))
                .and_then(|binding| binding.checked_add(offset))
                .ok_or("native_preflight_binding_overflow")?;
            if descriptor.is_none_or(|d| {
                d.descriptor_count == 1 && d.access == spirv_bind::ReflectedTextureAccess::Sampled
            }) && SourceInfo::absent_from_both(&inputs.sources, binding)
            {
                continue;
            }
            if pass
                .color
                .iter()
                .any(|color| color.texture == Some(texture))
            {
                return Err("native_preflight_attachment_feedback");
            }
            let (format, memoryless) = resource_format(state, host, task, texture.slot.0, 0)?;
            if memoryless {
                return Err("native_preflight_sampled_memoryless");
            }
            let source = resource_backing(state, host, task, texture.slot.0, 0)?
                .ok_or("native_preflight_sampled_backing")?;
            for target in pass.color.iter().filter_map(|color| color.texture) {
                if resource_backing(state, host, task, target.slot.0, 0)? == Some(source) {
                    return Err("native_preflight_attachment_feedback");
                }
            }
            if crate::runtime::draw::native_color_layout(format).is_none() {
                return Err("native_preflight_sampled_specialization");
            }
            if descriptor.is_some_and(|d| d.descriptor_count != 1) {
                return Err("native_preflight_descriptor_array");
            }
            if descriptor.is_some_and(|d| d.access != spirv_bind::ReflectedTextureAccess::Sampled) {
                return Err("native_preflight_texture_access");
            }
            layout.push(BindingSig {
                binding,
                ty: ash::vk::DescriptorType::SAMPLED_IMAGE.as_raw() as u32,
                stages,
                count: descriptor.map_or(1, |d| d.descriptor_count),
            });
        }
    }
    for sampler in metadata_samplers(state, host, task, bindings, inputs)? {
        layout.push(BindingSig {
            binding: sampler.binding,
            ty: ash::vk::DescriptorType::SAMPLER.as_raw() as u32,
            stages,
            count: 1,
        });
    }
    let mut targets = Vec::new();
    for (slot, color) in pass.color.iter().enumerate() {
        let Some(reference) = color.texture else {
            continue;
        };
        if color.resolve_texture.is_some() || color.slice != 0 || color.depth_plane != 0 {
            return Err("native_preflight_attachment_subresource");
        }
        let (format, memoryless) = resource_format(state, host, task, reference.slot.0, 0)?;
        let format = if memoryless {
            translate::pixel::memoryless_color_attachment(format).map(|f| f.vk)
        } else {
            translate::pixel::color_attachment(format).map(|(f, _)| f.vk)
        }
        .map_err(|_| "native_preflight_attachment_format")?;
        targets.push((slot as u32, format));
    }
    let Some((_, first)) = targets.first() else {
        return Err("native_preflight_no_color");
    };
    if targets
        .iter()
        .enumerate()
        .any(|(index, (slot, _))| index as u32 != *slot)
    {
        return Err("native_preflight_sparse_attachments");
    }
    let mut native_pass = PassKey::single(Color0Load::Preserve, *first);
    let mut masks = [engine::ColorWriteMask::ALL; 8];
    masks[0] = pd.color0.write_mask;
    let mut secondary_blend = [None; 7];
    for (index, (slot, format)) in targets.iter().enumerate().skip(1) {
        if index > 7 {
            return Err("native_preflight_attachment_count");
        }
        native_pass.secondary[index - 1] = SecondaryAttachKey {
            format: *format,
            load: true,
        };
        if let Some(color) = pd
            .color_attachments
            .iter()
            .find(|color| color.slot == *slot)
        {
            masks[index] = color.write_mask;
            secondary_blend[index - 1] = color_blend(*color);
        }
    }
    native_pass.secondary_count = (targets.len() - 1) as u8;
    for binding in &f.reflection.bindings {
        if binding.kind == metal2vulkan::reflect::ResourceKind::ColorInput {
            let slot = binding.metal_index;
            if slot >= 8 || !targets.iter().any(|(target, _)| *target == slot) {
                return Err("native_preflight_input_attachment");
            }
            native_pass.color_input |= 1 << slot;
            layout.push(BindingSig {
                binding: engine::types::COLOR_INPUT_BINDING + slot,
                ty: ash::vk::DescriptorType::INPUT_ATTACHMENT.as_raw() as u32,
                stages: 16,
                count: 1,
            });
        }
    }
    Ok(Metadata {
        vertex: inputs.sources[0].words.clone(),
        fragment: inputs.sources[1].words.clone(),
        bindings: layout,
        pass: native_pass,
        blend: color_blend(pd.color0),
        secondary_blend,
        color_write_mask: masks,
        raster,
        topology,
        viewport_count,
    })
}

fn metadata_samplers<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    bindings: &reims_vgpu_core::encoder::RenderEncoderState,
    inputs: &PipelineInputs,
) -> Result<Vec<engine::SamplerResource>, &'static str> {
    let mut seen = std::collections::BTreeSet::new();
    let mut samplers = Vec::new();
    for (stage, variant, offset) in [
        (&bindings.vertex, &inputs.vertex, 0),
        (
            &bindings.fragment,
            &inputs.fragment,
            if inputs.separate {
                spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
            } else {
                0
            },
        ),
    ] {
        for (index, bind) in stage.samplers.bound() {
            let Some(reference) = bind.object.filter(|r| r.slot.0 != 0) else {
                continue;
            };
            let binding = spirv_bind::SAMPLER_BINDING_BASE
                .checked_add(index)
                .and_then(|binding| binding.checked_add(offset))
                .ok_or("native_preflight_binding_overflow")?;
            if !crate::runtime::draw::vulkan::guest_sampler_binding(&variant.samplers, binding)
                || !seen.insert(binding)
            {
                continue;
            }
            let mut sampler = crate::runtime::draw::vulkan::load_vulkan_sampler(
                state,
                host,
                task,
                reference.slot.0,
                binding,
            )
            .map_err(|_| "native_preflight_sampler")?;
            if let Some((min, max)) = bind.lod_clamps {
                sampler.lod_min = min.0;
                sampler.lod_max = max.0;
            }
            samplers.push(sampler);
        }
    }
    for (variant, stage) in [(&inputs.vertex, "vertex"), (&inputs.fragment, "fragment")] {
        for reflected in variant.samplers.iter() {
            if !seen.insert(reflected.binding) {
                continue;
            }
            samplers.push(if let Some(value) = reflected.static_state() {
                crate::runtime::draw::vulkan::reflected_static_sampler_resource(
                    stage,
                    reflected.binding,
                    value,
                )
                .map_err(|_| "native_preflight_static_sampler")?
            } else {
                engine::SamplerResource::normalized_default(reflected.binding)
            });
        }
    }
    if samplers
        .iter()
        .any(|sampler| sampler.unnormalized_coordinates)
    {
        return Err("native_preflight_unnormalized_sampler");
    }
    Ok(samplers)
}

fn resource_format<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    reference: u32,
    depth: usize,
) -> Result<(u16, bool), &'static str> {
    if depth >= crate::runtime::draw::MAX_TEXTURE_VIEW_CHAIN {
        return Err("native_preflight_view_chain");
    }

    let owner = objects::resolve_resource(state, host, task, reference)
        .map_err(|_| "native_preflight_texture_owner")?;
    if owner.entry.object_type == crate::runtime::decode::resource::OBJECT_TYPE_MEMORYLESS_TEXTURE {
        let desc = reims_vgpu_protocol::memoryless::decode(&owner.descriptor)
            .map_err(|_| "native_preflight_memoryless_descriptor")?;
        return Ok((desc.pixel_format, true));
    }
    if let Ok(buffer) =
        crate::runtime::decode::resource::decode_buffer_texture_descriptor(&owner.descriptor)
    {
        return Ok((buffer.desc.pixel_format, false));
    }
    match owner.decoded() {
        Ok(Descriptor::Texture(texture)) => Ok((texture.pixel_format, false)),
        Ok(Descriptor::TextureView(view)) => {
            let (format, memoryless) =
                resource_format(state, host, task, view.base_texture_ref, depth + 1)?;
            Ok((
                if view.pixel_format == 0 {
                    format
                } else {
                    view.pixel_format
                },
                memoryless,
            ))
        }
        Ok(Descriptor::IOSurfaceTexture {
            pixel_format,
            mapping_id,
            ..
        }) => {
            let mapping = state
                .mappings
                .get(mapping_id)
                .filter(|m| m.mapped)
                .ok_or("native_preflight_mapping_unavailable")?;
            Ok((
                if *pixel_format != 0 {
                    *pixel_format
                } else {
                    mapping.format
                },
                false,
            ))
        }
        _ => Err("native_preflight_texture_descriptor"),
    }
}

fn resource_backing<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task: u32,
    reference: u32,
    depth: usize,
) -> Result<Option<reims_vgpu_core::access::BackingId>, &'static str> {
    if depth >= crate::runtime::draw::MAX_TEXTURE_VIEW_CHAIN {
        return Err("native_preflight_backing_chain");
    }
    let owner = objects::resolve_resource(state, host, task, reference)
        .map_err(|_| "native_preflight_backing_owner")?;
    if owner.entry.object_type == crate::runtime::decode::resource::OBJECT_TYPE_MEMORYLESS_TEXTURE {
        return Ok(None);
    }
    if let Ok(backing) =
        objects::backing_id_of(state, task, reference, &owner.entry, &owner.descriptor)
    {
        return Ok(Some(backing));
    }
    if let Ok(buffer) =
        crate::runtime::decode::resource::decode_buffer_texture_descriptor(&owner.descriptor)
    {
        return resource_backing(state, host, task, buffer.buffer_ref, depth + 1);
    }
    if let Ok(Descriptor::TextureView(view)) = owner.decoded() {
        return resource_backing(state, host, task, view.base_texture_ref, depth + 1);
    }
    Err("native_preflight_backing_unavailable")
}

#[cfg(test)]
mod activation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::exec::ExecBuilder;
    use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
    use reims_vgpu_core::render::{Instancing, PassDescriptorSlot, PrimitiveType};
    use reims_vgpu_protocol::segment::{SegmentKind, SegmentLifetime};

    #[test]
    fn native_metadata_walk_retains_exact_draw_state_without_running_submission() {
        let mut builder = ExecBuilder::new();
        builder
            .arenas_mut()
            .pass_descriptors
            .push(reims_vgpu_core::pass::PassDescriptor::empty());
        builder
            .begin_encoder(SegmentKind::Render, SegmentLifetime::SELF_CONTAINED)
            .unwrap();
        let pipeline = ResourceId {
            slot: ObjectListRef(44),
            generation: SlotGeneration(3),
        };
        let draw = RenderOp::Draw(DrawOp::Primitives {
            primitive: PrimitiveType(3),
            vertex_start: 0,
            vertex_count: 3,
            instances: Instancing::default(),
        });
        for op in [
            RenderOp::WriteDescriptor {
                descriptor: PassDescriptorSlot(0),
            },
            RenderOp::SetPipeline { pipeline },
            draw,
            RenderOp::SetCullMode(2),
            RenderOp::SetTriangleFillMode(1),
            draw,
        ] {
            builder
                .record(
                    ResolvedOperation::Render(op),
                    &mut |_: &reims_vgpu_core::access::Participation| {
                        unreachable!("this metadata fixture names no guest allocation")
                    },
                )
                .unwrap();
        }
        builder.end_segment().unwrap();
        let work = builder.finish().unwrap();
        let records = work.streams.clone();
        let collect = || {
            scan(&work, |bindings, _, raster, topology, slots| {
                assert_eq!(bindings.pipeline, Some(pipeline));
                assert_eq!(topology, reims_vgpu_core::topology::PrimitiveType::Triangle);
                assert_eq!(slots, 1);
                Ok((raster.cull_mode, raster.fill_mode))
            })
            .unwrap()
        };
        assert_eq!(collect(), vec![(0, 0), (2, 1)]);
        assert_eq!(collect(), vec![(0, 0), (2, 1)]);
        assert_eq!(work.streams, records);
    }

    #[test]
    fn unavailable_native_metadata_is_memoized_without_resource_table_consumption() {
        let submission = ExecSubmission::stated(1, vec![vec![1, 2, 3]]);
        let state = DeviceState::new(crate::model::DeviceId(42), crate::model::PAGE_SHIFT_ARM64E);
        let host = crate::runtime::host::FakeHost::new();
        let mut work = ExecWork::default();
        work.streams.push(reims_vgpu_core::exec::ResolvedStream {
            begin: reims_vgpu_core::stream::SegmentBegin {
                kind: SegmentKind::Render,
                protection: None,
            },
            records: vec![reims_vgpu_core::exec::StreamRecord {
                at: reims_vgpu_core::stream::StreamPosition {
                    segment: 0,
                    record: 0,
                },
                op: ResolvedOperation::Render(RenderOp::Draw(DrawOp::Primitives {
                    primitive: PrimitiveType(3),
                    vertex_start: 0,
                    vertex_count: 3,
                    instances: Instancing::default(),
                })),
            }],
        });
        assert!(!pending(&state, &host, &submission, &work));
        assert!(matches!(
            submission.native_pipelines.get(),
            Some(Prepared(Err(ScanFailure {
                reason: "native_preflight_no_pass",
                ..
            })))
        ));
        assert!(!pending(&state, &host, &submission, &work));
        assert_eq!(submission.streams(), &[vec![1, 2, 3]]);
        assert!(submission.resource_descs.is_empty());
    }
}
