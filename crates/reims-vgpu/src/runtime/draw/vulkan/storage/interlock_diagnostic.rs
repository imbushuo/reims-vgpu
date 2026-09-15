use super::*;
use super::backing_coverage::{byte_spans, overlap_bytes};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

fn describe(spans: &Option<Vec<(u64, u64)>>) -> String {
    match spans {
        Some(spans) => {
            let bytes: u64 = spans.iter().map(|(start, end)| end - start).sum();
            format!(
                "complete_recorded_page_union:ranges{}:bytes{}:first{:#x}:last_end{:#x}",
                spans.len(),
                bytes,
                spans[0].0,
                spans[spans.len() - 1].1
            )
        }
        None => "unknown:incomplete_or_invalid_page_coverage".into(),
    }
}

struct PipelineCount {
    task: u32,
    pipeline: u32,
    count: usize,
}

fn admit_observation(task: u32, pipeline: u32, key: u64) -> Option<usize> {
    static COUNTS: Mutex<Vec<PipelineCount>> = Mutex::new(Vec::new());
    let mut counts = COUNTS
        .lock()
        .expect("interlock diagnostic counters poisoned");
    let index = match counts
        .iter()
        .position(|entry| entry.task == task && entry.pipeline == pipeline)
    {
        Some(index) => index,
        None if counts.len() < 16 => {
            counts.push(PipelineCount {
                task,
                pipeline,
                count: 0,
            });
            counts.len() - 1
        }
        None => return None,
    };
    if counts[index].count >= 16
        || !crate::observe::first_sight("interlock_backing_observation", key)
    {
        return None;
    }
    counts[index].count += 1;
    Some(counts.iter().map(|entry| entry.count).sum())
}

pub(super) fn report<M: HostMemory>(
    textures: &StorageTextures,
    state: &DeviceState,
    host: &M,
    req: &DrawEncodeRequest,
    raster_samples: u32,
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(
        || match crate::config::switch(crate::config::PIPELINE_DIAGNOSTICS) {
            crate::config::Switch::On => true,
            crate::config::Switch::Off | crate::config::Switch::Unset => false,
            crate::config::Switch::Unrecognized => {
                crate::observe::fail("interlock_backing_diagnostic reason=unrecognized_override");
                false
            }
        },
    ) {
        return;
    }
    let scope = req
        .input_snapshot_scope
        .as_ref()
        .and_then(|scope| scope.current());
    for (&texture_ref, texture) in &textures.textures {
        let destination_residency = texture
            .staged
            .rail
            .residency
            .map(|r| (r.key, r.seed_generation));
        let destination_owner_generation = texture.staged.diagnostic_writeback_generation(state);
        for color in &req.colors {
            let source_map_generation = state
                .mappings
                .get(&color.mapping_id)
                .map(|mapping| mapping.map_generation);
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            (
                req.task_id,
                req.pipeline_ref,
                texture_ref,
                color.slot,
                color.texture_ref,
                color.mapping_id,
                color.target_gva,
                source_map_generation,
                req.gva_alloc_gen,
                (destination_residency, destination_owner_generation, scope),
            )
                .hash(&mut hash);
            let Some(observation) = admit_observation(req.task_id, req.pipeline_ref, hash.finish())
            else {
                continue;
            };
            let page_bytes = state.page_size();
            let (destination, destination_pages) = texture.staged.diagnostic_writeback_pages(state);
            let source_pages = if color.mapping_id != 0 {
                state
                    .mappings
                    .get(&color.mapping_id)
                    .filter(|mapping| mapping.mapped)
                    .and_then(|_| state.mapping_reach_pages(color.mapping_id))
            } else if color.target_gva != 0 {
                crate::runtime::draw::sync_store_target_pages(state, host, req.task_id, color)
                    .and_then(|pages| {
                        pages
                            .ordered_complete(color.target_gva, page_bytes)
                            .filter(|pages| !pages.is_empty())
                            .map(<[u64]>::to_vec)
                    })
            } else {
                None
            };
            let source = source_pages
                .as_deref()
                .and_then(|pages| byte_spans(pages, page_bytes));
            let dest = destination_pages
                .as_deref()
                .and_then(|pages| byte_spans(pages, page_bytes));
            let overlap = source
                .as_ref()
                .zip(dest.as_ref())
                .and_then(|(source, dest)| overlap_bytes(source, dest));
            let relation = if color.storage == ColorStorage::Memoryless {
                "source_pass_local_no_guest_backing"
            } else {
                match overlap {
                    Some(0) => "recorded_guest_byte_spans_disjoint",
                    Some(_) => "recorded_guest_byte_spans_overlap",
                    None => "unknown_coverage",
                }
            };
            crate::observe::off(format!(
                "interlock_backing_observation observation={observation} scope={scope:?} \
                 task={} pipe={} destination_ref={} \
                 destination={} destination_geometry={}x{} destination_format={:#x} \
                 destination_mips={} destination_subresource=resolved_writeback_window \
                 destination_owner_generation={destination_owner_generation:?} \
                 destination_residency={destination_residency:?} destination_coverage={} \
                 destination_page_entries={:?} destination_page_bytes={page_bytes} \
                 source_slot={} source_ref={} source_mid={} source_gva={:#x} \
                 source_map_generation={source_map_generation:?} source_gva_generation={:?} \
                 source_geometry={}x{} source_format={:#x} source_pitch={} source_coverage={} \
                 source_page_entries={:?} source_page_bytes={page_bytes} \
                 source_subresource=whole_mapping_or_linear_span source_plane=unrecorded \
                 source_slice=unrecorded source_mip={} source_native_mip=0 source_native_layer=0 \
                 relation={relation} overlap_bytes={overlap:?} raster_samples={raster_samples} \
                 samples={} first_vertex={} vertices={} instances={} first_instance={} \
                 indexed={:?} continues={} more={} viewports={:?} scissors={:?} \
                 (observation only; no ownership lease or renderer admission is granted)",
                req.task_id,
                req.pipeline_ref,
                texture_ref,
                destination,
                texture.staged.width,
                texture.staged.height,
                texture.staged.pixel_format,
                texture.staged.mip_levels,
                describe(&dest),
                destination_pages.as_ref().map(Vec::len),
                color.slot,
                color.texture_ref,
                color.mapping_id,
                color.target_gva,
                (color.slot == 0 && color.target_gva != 0).then_some(req.gva_alloc_gen),
                color.width,
                color.height,
                color.format,
                color.row_stride,
                describe(&source),
                source_pages.as_ref().map(Vec::len),
                color.guest_mip_level,
                color.sample_count,
                req.first_vertex,
                req.vertex_count,
                req.instance_count,
                req.base_instance,
                req.indexed,
                req.continues_render_pass,
                req.render_pass_continues,
                req.viewports,
                req.scissors,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interlock_diagnostic_compares_physical_bytes_across_page_geometries() {
        let a = byte_spans(&[0x4000], 16384).unwrap();
        let b = byte_spans(&[0x5000], 4096).unwrap();
        assert_eq!(overlap_bytes(&a, &b), Some(4096));
        assert_eq!(overlap_bytes(&b, &a), Some(4096));
        let adjacent = byte_spans(&[0x8000], 4096).unwrap();
        assert_eq!(overlap_bytes(&a, &adjacent), Some(0));
    }

    #[test]
    fn interlock_diagnostic_merges_repeated_and_scattered_pages_without_losing_overlap() {
        let a = byte_spans(&[0xc000, 0x4000, 0x4000], 16384).unwrap();
        let b = byte_spans(&[0xd000, 0x5000, 0xc000], 4096).unwrap();
        assert_eq!(overlap_bytes(&a, &b), Some(12288));
        assert_eq!(a, vec![(0x4000, 0x8000), (0xc000, 0x10000)]);
    }

    #[test]
    fn interlock_diagnostic_incomplete_or_overflowing_coverage_never_proves_disjoint() {
        for (pages, size) in [
            (vec![], 4096),
            (vec![0x5000], 16384),
            (vec![0x4000], 0),
            (vec![0x4000], 6000),
            (vec![u64::MAX - 4095], 4096),
        ] {
            assert!(byte_spans(&pages, size).is_none());
        }
    }
}
