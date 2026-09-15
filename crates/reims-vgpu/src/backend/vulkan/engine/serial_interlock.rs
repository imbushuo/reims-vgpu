use ash::vk;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::caches::storage_descriptors::DeclarationProof;
use super::caches::PassKey;
use super::types::DrawRequest;
use crate::runtime::draw::vulkan::InterlockIsolationRefusal;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    Isolation(InterlockIsolationRefusal),
    State(&'static str),
    Shader(&'static str),
    Split(reims_vgpu_vulkan::framebuffer_fetch::Refusal),
}

impl crate::observe::Decline for Refusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Isolation(_) => "draw_vk_serial_interlock_isolation",
            Self::State(_) => "draw_vk_serial_interlock_state",
            Self::Shader(_) => "draw_vk_serial_interlock_shader",
            Self::Split(_) => "draw_vk_serial_interlock_split",
        }
    }
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("detail", format!("{self:?}"))]
    }
}

/// A projection, not an executable permission. Only an admitted Plan exposes it.
pub(super) struct Program {
    original: Arc<Vec<u32>>,
    lowered: Arc<Vec<u32>>,
    declarations: DeclarationProof,
}

impl Program {
    pub(super) fn original(&self) -> &Arc<Vec<u32>> {
        &self.original
    }

    pub(super) fn analyze(original: &Arc<Vec<u32>>) -> Result<Arc<Self>, Refusal> {
        let lowered = lower(original)?;
        Ok(Arc::new(Self {
            original: original.clone(),
            declarations: DeclarationProof::of_final_module(&lowered),
            lowered: Arc::new(lowered),
        }))
    }
}

pub(super) struct VertexDeclarations {
    original: Arc<Vec<u32>>,
    declarations: DeclarationProof,
}

impl VertexDeclarations {
    pub(super) fn analyze(original: &Arc<Vec<u32>>) -> Arc<Self> {
        Arc::new(Self {
            original: original.clone(),
            declarations: DeclarationProof::of_final_module(original),
        })
    }
}

pub(super) struct Plan {
    program: Arc<Program>,
    _vertex: Arc<VertexDeclarations>,
    primitives: reims_vgpu_vulkan::framebuffer_fetch::Primitives,
}

impl Plan {
    pub(super) fn admit(
        req: &DrawRequest,
        dynamic_front_face: bool,
        polygon_mode: vk::PolygonMode,
        program: impl FnOnce() -> Result<(Arc<Program>, Arc<VertexDeclarations>), Refusal>,
    ) -> Result<Self, Refusal> {
        let isolation = req
            .interlock_isolation
            .as_ref()
            .ok_or(Refusal::State("isolation_not_supplied"))?
            .as_ref()
            .map_err(|reason| Refusal::Isolation(reason.clone()))?;
        isolation.matches(req).map_err(Refusal::Isolation)?;
        if req.depth.is_some()
            || req.occlusion_query.is_some()
            || req.multisample_resolve
            || super::viewport_slot_count(req) != 1
        {
            return Err(Refusal::State("depth_query_resolve_or_multiview"));
        }
        if req.storage_textures.is_empty()
            || req.storage_textures.iter().any(|texture| {
                texture
                    .bindings
                    .iter()
                    .any(|binding| binding.stage != vk::ShaderStageFlags::FRAGMENT)
            })
        {
            return Err(Refusal::State("isolated_fragment_storage_required"));
        }
        let samples = req.raster_sample_count.max(1);
        if req.color_sample_count.max(1) != samples
            || samples != 1
            || polygon_mode != vk::PolygonMode::FILL
        {
            return Err(Refusal::State("single_sample_filled_required"));
        }
        let draw = reims_vgpu_vulkan::framebuffer_fetch::Draw {
            reads_attachment: true,
            topology: req.primitive_topology.0,
            count: req
                .indexed
                .as_ref()
                .map_or(req.vertex_count, |indices| indices.index_count),
            first: if req.indexed.is_some() {
                0
            } else {
                req.first_vertex
            },
            instances: req.instance_count.unwrap_or(1),
            first_instance: req.base_instance,
            indexed: req.indexed.is_some(),
            samples,
            polygon_mode,
        };
        let primitives = reims_vgpu_vulkan::framebuffer_fetch::plan(
            reims_vgpu_vulkan::framebuffer_fetch::Cell {
                ordered_color_access: false,
                dynamic_front_face,
            },
            draw,
            &req.vert_spirv,
            &req.frag_spirv,
        )
        .map_err(Refusal::Split)?
        .ok_or(Refusal::State("no_complete_primitive"))?;
        let (program, vertex) = program()?;
        if !Arc::ptr_eq(program.original(), &req.frag_spirv)
            || !Arc::ptr_eq(&vertex.original, &req.vert_spirv)
        {
            return Err(Refusal::State("shader_identity_changed"));
        }
        let vertex_bindings = vertex
            .declarations
            .complete_bindings()
            .ok_or(Refusal::Shader("vertex_declarations_unproven"))?;
        let fragment_bindings = program
            .declarations
            .complete_bindings()
            .ok_or(Refusal::Shader("fragment_declarations_unproven"))?;
        // Provided images still stage and retain their ordinary hazards/owners.
        // Only this strategy's admission ignores bindings proven absent in both modules.
        if req.sampled_images.iter().any(|image| {
            vertex_bindings.binary_search(&image.binding).is_ok()
                || fragment_bindings.binary_search(&image.binding).is_ok()
        }) {
            return Err(Refusal::State("isolated_fragment_storage_required"));
        }
        Ok(Self {
            program,
            _vertex: vertex,
            primitives,
        })
    }

    pub(super) fn words(&self) -> &Arc<Vec<u32>> {
        &self.program.lowered
    }
    pub(super) fn primitives(&self) -> reims_vgpu_vulkan::framebuffer_fetch::Primitives {
        self.primitives
    }
}

pub(super) fn preserving_pass(mut pass: PassKey) -> PassKey {
    pass.color0_load = super::caches::Color0Load::Preserve;
    for secondary in &mut pass.secondary[..pass.secondary_count as usize] {
        secondary.load = true;
    }
    if let Some(depth) = &mut pass.depth {
        depth.load = true;
    }
    pass
}

pub(super) fn dependency() -> (vk::PipelineStageFlags, vk::MemoryBarrier<'static>) {
    (
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE),
    )
}

fn lower(words: &[u32]) -> Result<Vec<u32>, Refusal> {
    use Refusal::Shader as E;
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(E("malformed_header"));
    }
    let mut entry = None;
    let mut function = None;
    let mut modes = 0;
    let mut begins = 0;
    let mut ends = 0;
    let mut inside = false;
    let mut pixel_capability = false;
    let mut extension = false;
    let mut pointer_classes = BTreeMap::new();
    let mut value_types = BTreeMap::new();
    let mut image_types = BTreeMap::new();
    let mut uniform_pointers = Vec::new();
    let mut stores = Vec::new();
    let mut result = words[..5].to_vec();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        if count == 0 || count > words.len() - at {
            return Err(E("malformed_instruction"));
        }
        let args = &words[at + 1..at + count];
        let mut omit = false;
        match (op, args) {
            (14, [0, 1]) => {}
            (14, _) => return Err(E("memory_model")),
            (15, [4, id, ..]) if entry.is_none() => entry = Some(*id),
            (15, _) => return Err(E("entry_point")),
            (16, [id, 5366 | 5367]) if Some(*id) == entry => {
                modes += 1;
                omit = true;
            }
            (16, [_, 5366..=5371, ..]) => return Err(E("interlock_mode")),
            (16, [_, 7]) => {}
            (16, _) => return Err(E("execution_mode")),
            (17, [5378]) => {
                pixel_capability = true;
                omit = true;
            }
            (10, string) => {
                let bytes: Vec<_> = string.iter().flat_map(|word| word.to_le_bytes()).collect();
                let name = bytes.split(|byte| *byte == 0).next().unwrap_or_default();
                if name != b"SPV_EXT_fragment_shader_interlock" {
                    return Err(E("extension"));
                }
                extension = true;
                omit = true;
            }
            (11, [_, string @ ..]) => {
                let bytes: Vec<_> = string.iter().flat_map(|word| word.to_le_bytes()).collect();
                if bytes.split(|byte| *byte == 0).next() != Some(b"GLSL.std.450".as_slice()) {
                    return Err(E("extended_instruction_set"));
                }
            }
            (54, [_, id, ..]) => function = Some(*id),
            (56, _) => {
                if inside {
                    return Err(E("unterminated_interlock"));
                }
                function = None;
            }
            (5364, []) if function == entry && !inside => {
                begins += 1;
                inside = true;
                omit = true;
            }
            (5365, []) if function == entry && inside => {
                ends += 1;
                inside = false;
                omit = true;
            }
            (5364 | 5365, _) => return Err(E("interlock_region")),
            (252 | 4416 | 5380, _) => return Err(E("discard_or_helper")),
            (253..=255, _) if inside => return Err(E("exit_inside_interlock")),
            (227..=242 | 318 | 319 | 333..=366, _) => return Err(E("atomic_or_subgroup")),
            (71, [_, 11, 36..=41 | 4416..=4420, ..])
            | (72, [_, _, 11, 36..=41 | 4416..=4420, ..]) => return Err(E("subgroup_builtin")),
            (25, [id, _, dimension, _, arrayed, multisampled, sampled, ..]) => {
                image_types.insert(*id, (*dimension, *arrayed, *multisampled, *sampled));
            }
            (32, [id, class, pointee]) => {
                pointer_classes.insert(*id, *class);
                if *class == 0 {
                    uniform_pointers.push((*id, *pointee));
                }
            }
            (59, [_, _, 2 | 12, ..]) => return Err(E("fragment_buffer_resource")),
            (55 | 57 | 59 | 61 | 65 | 66 | 67 | 70 | 83 | 169 | 245, [ty, id, ..]) => {
                value_types.insert(*id, *ty);
            }
            (62..=64, [pointer, ..]) => stores.push(*pointer),
            (99, _) if function != entry || !inside => {
                return Err(E("storage_write_outside_region"))
            }
            _ if op > 366 => return Err(E("unhandled_instruction")),
            _ => {}
        }
        if !omit {
            result.extend_from_slice(&words[at..at + count]);
        }
        at += count;
    }
    if entry.is_none() || modes != 1 || begins != 1 || ends != 1 || !pixel_capability || !extension
    {
        return Err(E("single_pixel_interlock_required"));
    }
    for pointer in stores {
        if !matches!(
            value_types
                .get(&pointer)
                .and_then(|ty| pointer_classes.get(ty)),
            Some(3 | 6 | 7)
        ) {
            return Err(E("buffer_or_unknown_write"));
        }
    }
    for (_, pointee) in uniform_pointers {
        if !matches!(image_types.get(&pointee), Some((1 | 6, 0, 0, 2))) {
            return Err(E("only_storage_and_input_images"));
        }
    }
    Ok(result)
}

#[cfg(test)]
pub(in crate::backend::vulkan::engine) mod tests;
