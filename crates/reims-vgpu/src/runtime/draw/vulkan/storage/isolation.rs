use super::*;
use crate::backend::vulkan::engine::{DrawRequest, TargetIdentity};
use crate::runtime::guest_ram::{GuestPageFootprint, GuestRamImport};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterlockIsolationRefusal {
    ScopeExpired,
    OwnerUnavailable { reference: u32 },
    BackingUnavailable { reference: u32 },
    ResidentSeedUnavailable { reference: u32 },
    GuestAlias { source: u32, destination: u32 },
    BindingChanged,
}

#[derive(Debug, PartialEq, Eq)]
struct StorageSignature {
    width: u32,
    height: u32,
    format: crate::backend::vulkan::engine::StorageImageFormat,
    bindings: Vec<(u32, u32, u32)>,
}

fn storage_signature(req: &DrawRequest) -> Vec<StorageSignature> {
    req.storage_textures
        .iter()
        .map(|texture| StorageSignature {
            width: texture.width,
            height: texture.height,
            format: texture.format,
            bindings: texture
                .bindings
                .iter()
                .map(|binding| {
                    (
                        binding.binding,
                        binding.access.descriptor_type().as_raw() as u32,
                        binding.stage.as_raw(),
                    )
                })
                .collect(),
        })
        .collect()
}

fn targets(req: &DrawRequest) -> Vec<Option<TargetIdentity>> {
    std::iter::once(req.target_identity.clone())
        .chain(
            req.secondary_targets
                .iter()
                .map(|target| Some(target.identity.clone())),
        )
        .collect()
}

fn covered_spans(footprint: &GuestPageFootprint, minimum: u64) -> Option<Vec<(u64, u64)>> {
    let spans = super::backing_coverage::byte_spans(footprint.pages(), footprint.page_size())?;
    let bytes = (footprint.pages().len() as u64).checked_mul(footprint.page_size())?;
    let distinct: u64 = spans.iter().map(|(start, end)| end - start).sum();
    (minimum != 0 && bytes >= minimum && distinct == bytes).then_some(spans)
}

/// A single draw's backing isolation, retaining its serialized resource owners.
/// Only live preparation or the explicitly host-owned test fixture can construct it.
#[derive(Debug)]
pub struct InterlockIsolation {
    scope: Option<crate::runtime::draw::SnapshotScopeRef>,
    fragment: Arc<Vec<u32>>,
    targets: Vec<Option<TargetIdentity>>,
    dimensions: (u32, u32),
    storage: Vec<StorageSignature>,
    _owners: Vec<Arc<crate::model::TaskResource>>,
    _imports: Vec<Arc<GuestRamImport>>,
    _footprints: Vec<GuestPageFootprint>,
}

impl InterlockIsolation {
    pub(crate) fn matches(&self, req: &DrawRequest) -> Result<(), InterlockIsolationRefusal> {
        if self
            .scope
            .as_ref()
            .is_some_and(|scope| scope.current().is_none())
        {
            return Err(InterlockIsolationRefusal::ScopeExpired);
        }
        if !Arc::ptr_eq(&self.fragment, &req.frag_spirv)
            || self.targets != targets(req)
            || self.dimensions != (req.width, req.height)
            || self.storage != storage_signature(req)
        {
            return Err(InterlockIsolationRefusal::BindingChanged);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn host_owned_fixture(req: &DrawRequest) -> Self {
        assert!(targets(req)
            .iter()
            .all(|target| matches!(target, Some(TargetIdentity::PassLocal { .. }))));
        Self {
            scope: None,
            fragment: req.frag_spirv.clone(),
            targets: targets(req),
            dimensions: (req.width, req.height),
            storage: storage_signature(req),
            _owners: Vec::new(),
            _imports: Vec::new(),
            _footprints: Vec::new(),
        }
    }
}

impl StorageTextures {
    pub(in crate::runtime::draw::vulkan) fn isolate_interlock<M: HostMemory>(
        &self,
        state: &DeviceState,
        host: &M,
        request: &DrawEncodeRequest,
        native: &DrawRequest,
    ) -> Result<InterlockIsolation, InterlockIsolationRefusal> {
        use InterlockIsolationRefusal as E;
        let scope = request
            .input_snapshot_scope
            .as_ref()
            .filter(|scope| scope.current().is_some())
            .ok_or(E::ScopeExpired)?
            .clone();
        let mut owners = Vec::new();
        let mut imports = Vec::new();
        let mut footprints = Vec::new();
        let mut sources = Vec::new();
        let native_targets = targets(native);
        if native_targets.len() != request.colors.len()
            || native.storage_textures.len() != self.textures.len()
        {
            return Err(E::BindingChanged);
        }
        for (color, target) in request.colors.iter().zip(&native_targets) {
            if color.storage == ColorStorage::Memoryless {
                if !matches!(target, Some(TargetIdentity::PassLocal { .. })) {
                    return Err(E::BackingUnavailable {
                        reference: color.texture_ref,
                    });
                }
                continue;
            }
            let owner = state
                .object_name(request.task_id, color.texture_ref)
                .and_then(|name| state.task_resources.get(request.task_id, name))
                .ok_or(E::OwnerUnavailable {
                    reference: color.texture_ref,
                })?;
            let tight_row = pixel_format::tight_row_bytes(color.width, color.format)
                .map(u64::from)
                .ok_or(E::BackingUnavailable { reference: color.texture_ref })?;
            let minimum = tight_row.checked_mul(u64::from(color.height))
                .ok_or(E::BackingUnavailable { reference: color.texture_ref })?;
            let footprint = if color.mapping_id != 0 {
                let mapping = state
                    .mappings
                    .get(&color.mapping_id)
                    .filter(|mapping| mapping.mapped)
                    .ok_or(E::BackingUnavailable {
                        reference: color.texture_ref,
                    })?;
                if let Some(import) = &mapping.contig_import {
                    imports.push(import.clone());
                }
                state
                    .mapping_reach_pages(color.mapping_id)
                    .and_then(|pages| GuestPageFootprint::new(pages.into(), state.page_size()))
            } else {
                // A read footprint remains required even with final StoreDontCare.
                u64::from(color.row_stride).checked_mul(u64::from(color.height))
                    .filter(|span| *span != 0 && color.target_gva != 0
                        && u64::from(color.row_stride) >= tight_row)
                    .and_then(|span| {
                        let pages = crate::runtime::draw::StoreTargetPages::capture(
                            state, host, request.task_id, color.target_gva, span,
                        );
                        pages
                            .ordered_complete(color.target_gva, state.page_size())
                            .and_then(|pages| {
                                GuestPageFootprint::new(Arc::from(pages), state.page_size())
                            })
                    })
            }
            .ok_or(E::BackingUnavailable {
                reference: color.texture_ref,
            })?;
            let spans =
                covered_spans(&footprint, minimum)
                    .ok_or(E::BackingUnavailable {
                        reference: color.texture_ref,
                    })?;
            owners.push(owner);
            footprints.push(footprint);
            sources.push((color.texture_ref, spans));
        }
        let mut destinations: Vec<(u32, Vec<(u64, u64)>)> = Vec::new();
        for (&reference, texture) in &self.textures {
            if texture.staged.rail.serve.is_some() {
                return Err(E::ResidentSeedUnavailable { reference });
            }
            owners.push(
                texture
                    .owner
                    .clone()
                    .ok_or(E::OwnerUnavailable { reference })?,
            );
            let footprint = texture
                .staged
                .writeback_footprint(state)
                .ok_or(E::BackingUnavailable { reference })?;
            let minimum = pixel_format::tight_row_bytes(texture.staged.width, texture.staged.pixel_format)
                .and_then(|row| u64::from(row).checked_mul(u64::from(texture.staged.height)))
                .ok_or(E::BackingUnavailable { reference })?;
            let spans =
                covered_spans(&footprint, minimum)
                    .ok_or(E::BackingUnavailable { reference })?;
            for (source, prior) in sources.iter().chain(&destinations) {
                if super::backing_coverage::overlap_bytes(prior, &spans) != Some(0) {
                    return Err(E::GuestAlias {
                        source: *source,
                        destination: reference,
                    });
                }
            }
            footprints.push(footprint);
            destinations.push((reference, spans));
        }
        if destinations.is_empty() {
            return Err(E::BackingUnavailable { reference: 0 });
        }
        Ok(InterlockIsolation {
            scope: Some(scope),
            fragment: native.frag_spirv.clone(),
            targets: targets(native),
            dimensions: (native.width, native.height),
            storage: storage_signature(native),
            _owners: owners,
            _imports: imports,
            _footprints: footprints,
        })
    }
}
