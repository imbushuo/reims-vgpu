//! Physical-page dependencies between pass-owned attachments and CPU staging.
//!
//! Object numbers and mapping IDs alone cannot establish disjointness: views,
//! buffers and separately mapped surfaces can name the same guest pages.
//! Unknown footprints take the materializing fallback, never a guessed cache
//! currency claim. These footprints live only for the guest encoder lifetime.

use super::*;
use crate::runtime::decode::resource::OBJECT_TYPE_BUFFER;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(crate) struct PassDependencies {
    colors: Vec<Option<BTreeSet<u64>>>,
    inputs: BTreeMap<u32, Option<BTreeSet<u64>>>,
    roots: BTreeMap<u32, u32>,
    color_roots: Vec<u32>,
}

fn gva_pages<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task: u32,
    gva: u64,
    span: u64,
) -> Option<BTreeSet<u64>> {
    if span == 0 {
        return Some(BTreeSet::new());
    }
    gva.checked_add(span)?;
    let pages = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task,
        gva,
        span,
        state.page_shift,
    );
    let expected = reims_vgpu_paging::span::pages_spanned(gva, span, state.page_size());
    (pages.len() as u64 == expected).then(|| pages.into_iter().collect())
}

fn texture_pages<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task: u32,
    reference: u32,
) -> Option<BTreeSet<u64>> {
    let resource = objects::resolve_resource(state, host, task, reference).ok()?;
    match resource.entry.object_type {
        OBJECT_TYPE_BUFFER => {
            let (gva, span) = objects::resolve_buffer_span_from_resource(state, &resource).ok()?;
            gva_pages(state, host, task, gva, span)
        }
        OBJECT_TYPE_MAPPER_REF_TEXTURE => {
            let mapping =
                objects::resolve_mapper_ref_texture_resource(state, task, reference, &resource)?;
            mapper::mapping_page_gpas(state, host, mapping).map(|pages| pages.into_iter().collect())
        }
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_GENERATE_MIPMAPS => {
            let texture = decode_texture_descriptor(&resource.descriptor).ok()?;
            let (gva, span) = texture.backing_gva_size(state.page_shift)?;
            gva_pages(state, host, task, gva, span)
        }
        OBJECT_TYPE_TEXTURE_VIEW => {
            if let Some(view) =
                buffer_texture_descriptor(state, host, task, reference, Some(&resource))
            {
                let (gva, span) =
                    objects::resolve_buffer_span(state, host, task, view.buffer_ref).ok()?;
                gva_pages(
                    state,
                    host,
                    task,
                    gva.checked_add(view.offset)?,
                    span.checked_sub(view.offset)?,
                )
            } else {
                let view = resolve_texture_view(state, host, task, reference)?;
                // The resolver collapses the complete view chain to a non-view
                // base, so this recursion cannot follow guest cycles.
                texture_pages(state, host, task, view.base_texture_ref)
            }
        }
        _ => None,
    }
}

fn may_overlap(left: Option<&BTreeSet<u64>>, right: Option<&BTreeSet<u64>>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => !left.is_disjoint(right),
        _ => true,
    }
}

impl PassDependencies {
    fn prepare<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
    ) {
        if !self.colors.is_empty() {
            return;
        }
        self.color_roots = req
            .colors
            .iter()
            .map(|color| {
                if color.storage == ColorStorage::Memoryless {
                    resolve_texture_view(state, host, req.task_id, color.texture_ref)
                        .map_or(color.texture_ref, |view| view.base_texture_ref)
                } else {
                    color.texture_ref
                }
            })
            .collect();
        self.colors = req
            .colors
            .iter()
            .map(|color| {
                if color.storage == ColorStorage::Memoryless {
                    Some(BTreeSet::new())
                } else if color.mapping_id != 0 {
                    mapper::mapping_page_gpas(state, host, color.mapping_id)
                        .map(|pages| pages.into_iter().collect())
                } else {
                    gva_pages(
                        state,
                        host,
                        req.task_id,
                        color.target_gva,
                        u64::from(color.row_stride) * u64::from(color.height),
                    )
                }
            })
            .collect();
    }

    fn input<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        task: u32,
        reference: u32,
    ) -> &Option<BTreeSet<u64>> {
        if !self.inputs.contains_key(&reference) {
            let root = resolve_texture_view(state, host, task, reference)
                .map_or(reference, |view| view.base_texture_ref);
            self.roots.insert(reference, root);
            self.inputs
                .insert(reference, texture_pages(state, host, task, reference));
        }
        &self.inputs[&reference]
    }

    pub(super) fn aliases<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        req: &DrawEncodeRequest,
        reference: u32,
    ) -> Vec<usize> {
        self.prepare(state, host, req);
        self.input(state, host, req.task_id, reference);
        if self.inputs[&reference].is_none() {
            crate::runtime::drain::note_store_route("metal_batch_footprint_unknown");
        }
        req.colors
            .iter()
            .enumerate()
            .filter_map(|(index, color)| {
                (self.color_roots[index] == self.roots[&reference]
                    || (color.storage == ColorStorage::GuestBacked
                        && may_overlap(
                            self.colors[index].as_ref(),
                            self.inputs[&reference].as_ref(),
                        )))
                .then_some(index)
            })
            .collect()
    }
}

pub(super) fn materialize_inputs<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    pass: &mut crate::backend::metal::render_pass::MetalRenderPass,
) -> Result<(), EncodeStatus> {
    let references: BTreeSet<_> = req
        .vertex_buffers
        .iter()
        .chain(req.fragment_buffers.iter())
        .map(|bind| bind.buffer_ref)
        .chain(
            req.vertex_textures
                .iter()
                .chain(req.fragment_textures.iter())
                .map(|bind| bind.texture_ref),
        )
        .chain(req.indexed.as_ref().map(|draw| draw.index_buffer_ref))
        .chain(req.depth_attach.as_ref().map(|a| a.texture_ref))
        .chain(req.stencil_attach.as_ref().map(|a| a.texture_ref))
        .filter(|reference| *reference != 0)
        .collect();
    let mut affected = BTreeSet::new();
    for &reference in &references {
        for index in pass
            .dependencies
            .borrow_mut()
            .aliases(state, host, req, reference)
        {
            let color = &req.colors[index];
            if color.storage == ColorStorage::Memoryless {
                return Err(EncodeStatus::Unsupported(
                    "draw_mtl_memoryless_sample_alias",
                ));
            }
            if pass.target(color).is_ok_and(|target| target.initialized()) {
                affected.insert(index);
            }
        }
    }
    if affected.is_empty() {
        return Ok(());
    }
    materialize_targets(
        state,
        host,
        req,
        pass,
        affected,
        "metal_batch_input_materialization",
    )?;
    for reference in references {
        state.invalidate_object_host_copies(req.task_id, reference);
    }
    Ok(())
}

fn materialize_targets<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    pass: &crate::backend::metal::render_pass::MetalRenderPass,
    affected: BTreeSet<usize>,
    boundary: &'static str,
) -> Result<(), EncodeStatus> {
    let stores: Vec<_> = affected
        .into_iter()
        .map(|index| {
            (
                index,
                sync_store_target_pages(state, host, req.task_id, &req.colors[index]),
            )
        })
        .collect();
    pass.flush(boundary).map_err(EncodeStatus::RailRefused)?;
    for (index, pages) in stores {
        let color = &req.colors[index];
        let target = pass.target(color).map_err(EncodeStatus::BadArgs)?;
        let len = reims_vgpu_protocol::extent::tight_image_bytes(color.width, color.height, 4)
            .ok_or(EncodeStatus::BadArgs("draw_mtl_dependency_geometry"))?;
        let mut bytes = vec![0; len];
        {
            let _readback = chain_phase::CostSpan::new("metal_readback_us");
            target.texture().get_bytes(
                bytes.as_mut_ptr().cast(),
                u64::from(color.width) * 4,
                ::metal::MTLRegion::new_2d(0, 0, u64::from(color.width), u64::from(color.height)),
                0,
            );
        }
        crate::runtime::drain::note_store_route("metal_readback_tiled");
        crate::runtime::drain::note_store_route_n("metal_readback_tiled_bytes", bytes.len() as u64);
        crate::runtime::drain::note_store_route("metal_batch_materialized_targets");
        let wrote = if color.mapping_id != 0 {
            mapping_write::write_rgba8_image_changed(
                state,
                host,
                color.mapping_id,
                &bytes,
                None,
                color.width,
                color.height,
                mapping_write::FramePublication::HostCache,
            )
        } else {
            write_gva_rgba8_within(
                state,
                host,
                req.task_id,
                color.target_gva,
                color.width,
                color.height,
                color.row_stride,
                color.format,
                &bytes,
                pages.as_ref().map(StoreTargetPages::membership),
            )
            .is_ok()
        };
        if !wrote {
            return Err(EncodeStatus::WritebackFailed(
                "draw_mtl_dependency_writeback",
            ));
        }
        state.invalidate_object_host_copies(req.task_id, color.texture_ref);
        if color.target_gva != 0 {
            crate::runtime::surface_cache::evict_gva(state, color.target_gva);
        }
    }
    Ok(())
}

pub(crate) fn land_before_refusal<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    pass: &crate::backend::metal::render_pass::MetalRenderPass,
) -> Result<(), EncodeStatus> {
    let affected = req
        .colors
        .iter()
        .enumerate()
        .filter_map(|(index, color)| {
            (color.storage == ColorStorage::GuestBacked
                && color.store_action != MTL_STORE_ACTION_DONT_CARE
                && pass.target(color).is_ok_and(|target| target.initialized()))
            .then_some(index)
        })
        .collect();
    materialize_targets(state, host, req, pass, affected, "metal_batch_refusal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_aliases_and_unknown_footprints_require_materialization() {
        let a = BTreeSet::from([0x4000, 0x8000]);
        let alias = BTreeSet::from([0x8000]);
        let disjoint = BTreeSet::from([0xc000]);
        assert!(may_overlap(Some(&a), Some(&alias)));
        assert!(!may_overlap(Some(&a), Some(&disjoint)));
        assert!(may_overlap(Some(&a), None));
        assert!(may_overlap(None, Some(&disjoint)));
    }

    #[test]
    fn distinct_buffer_objects_and_virtual_addresses_can_alias_an_attachment() {
        use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
        use crate::protocol::endian::{st32, st64};
        use crate::runtime::decode::resource::{list_object_entry_offset, OBJECT_LIST_ENTRY_LEN};
        use crate::runtime::gva_mem::{define_task_pages_arm64e, write_task_gva_arm64e};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut host = FakeHost::new();
        define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));
        // GVA page 2 and attachment page 1 deliberately name the same GPA.
        host.write_gpa((3 << PAGE_SHIFT_ARM64E) + 2 * 4, &5u32.to_le_bytes())
            .unwrap();
        for (reference, page) in [(8, 2), (9, 3)] {
            let descriptor_gva = 0x200 + u64::from(reference) * 16;
            let mut descriptor = [0; 16];
            st64(&mut descriptor, 64);
            st32(&mut descriptor[8..], page);
            write_task_gva_arm64e(&mut host, &state.tasks[1], descriptor_gva, &descriptor);
            let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
            st32(&mut entry, u32::from(OBJECT_TYPE_BUFFER) | (16 << 8));
            st64(&mut entry[4..], descriptor_gva);
            write_task_gva_arm64e(
                &mut host,
                &state.tasks[1],
                list_object_entry_offset(reference, 32).unwrap(),
                &entry,
            );
        }
        let req = DrawEncodeRequest {
            task_id: 1,
            colors: vec![ColorRtRequest {
                texture_ref: 37,
                target_gva: 1 << PAGE_SHIFT_ARM64E,
                row_stride: 16,
                width: 4,
                height: 4,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut dependencies = PassDependencies::default();
        assert_eq!(
            dependencies.aliases(&mut state, &mut host, &req, 8),
            vec![0]
        );
        assert!(dependencies
            .aliases(&mut state, &mut host, &req, 9)
            .is_empty());
        assert_eq!(
            dependencies.aliases(&mut state, &mut host, &req, 10),
            vec![0],
            "an unresolved object must not become evidence of disjointness"
        );
    }
}
