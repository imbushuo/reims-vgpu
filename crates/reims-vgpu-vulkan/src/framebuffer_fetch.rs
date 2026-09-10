//! Ordered input-attachment feedback, including the Vulkan 1.2 fallback.
//!
//! A subpass self-dependency only permits barriers; it does not execute one.
//! Without rasterization-order attachment access, issue one primitive at a
//! time, with a framebuffer-local color-write → input-read barrier before each.
//! A filled, single-sample primitive has at most one covered fragment per
//! pixel, so it cannot race its own attachment output. Keeping the fixed
//! function pipeline preserves blending, depth/stencil tests and write masks.
//!
//! The strip remains a strip: rebasing its three-vertex window preserves
//! provoking vertices, while reversing front-face state for odd triangles
//! preserves culling, front-facing inputs and two-sided stencil. No index
//! bytes are read or rewritten. Builtins whose meaning changes when draws
//! split, and vertex side effects whose invocation count could change, refuse.
//!
//! Vulkan's "Render Pass Memory Dependencies" explicitly permits a pipeline
//! barrier inside a subpass to synchronize prior attachment writes with
//! subsequent input attachment reads.

use ash::vk;
use reims_vgpu_core::topology::PrimitiveType;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    pub ordered_color_access: bool,
    pub dynamic_front_face: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub reads_attachment: bool,
    pub topology: PrimitiveType,
    /// Index count for indexed draws, vertex count otherwise.
    pub count: u32,
    /// First index (normally zero), or first vertex for non-indexed draws.
    pub first: u32,
    pub instances: u32,
    pub first_instance: u32,
    pub indexed: bool,
    pub samples: u32,
    pub polygon_mode: vk::PolygonMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Multisample,
    TriangleWireframe,
    StripWinding,
    ArgumentOverflow,
    MalformedShader,
    SplitBuiltin(u32),
    VertexSideEffect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitive {
    pub first: u32,
    pub count: u32,
    pub first_instance: u32,
    pub reverse_winding: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Primitives {
    draw: Draw,
    primitive_count: u32,
    vertices_per_primitive: u32,
    stride: u32,
}

impl Primitives {
    pub fn iter(self) -> impl Iterator<Item = Primitive> {
        (0..self.draw.instances).flat_map(move |instance| {
            (0..self.primitive_count).map(move |primitive| Primitive {
                first: self.draw.first + primitive * self.stride,
                count: self.vertices_per_primitive,
                first_instance: self.draw.first_instance + instance,
                reverse_winding: self.draw.topology == PrimitiveType::TriangleStrip
                    && primitive & 1 != 0,
            })
        })
    }
}

/// `None` is the original unsplit draw, either without feedback or with the
/// hardware ordering feature. The fallback never assumes a driver identity.
pub fn plan(
    cell: Cell,
    draw: Draw,
    vertex: &[u32],
    fragment: &[u32],
) -> Result<Option<Primitives>, Refusal> {
    if !draw.reads_attachment || cell.ordered_color_access {
        return Ok(None);
    }
    let (vertices_per_primitive, stride) = match draw.topology {
        PrimitiveType::Point => (1, 1),
        PrimitiveType::Line => (2, 2),
        PrimitiveType::LineStrip => (2, 1),
        PrimitiveType::Triangle => (3, 3),
        PrimitiveType::TriangleStrip => (3, 1),
    };
    let primitive_count = if draw.count < vertices_per_primitive {
        0
    } else {
        1 + (draw.count - vertices_per_primitive) / stride
    };
    if primitive_count == 0 || draw.instances == 0 {
        // No fragment can fetch. Preserve the original draw, including any
        // vertex invocations associated with an incomplete primitive.
        return Ok(None);
    }
    if draw.samples != 1 {
        return Err(Refusal::Multisample);
    }
    if matches!(draw.topology, PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
        && draw.polygon_mode != vk::PolygonMode::FILL
    {
        return Err(Refusal::TriangleWireframe);
    }
    if draw.topology == PrimitiveType::TriangleStrip && primitive_count > 1
        && !cell.dynamic_front_face
    {
        return Err(Refusal::StripWinding);
    }
    draw.first.checked_add(draw.count.saturating_sub(1))
        .ok_or(Refusal::ArgumentOverflow)?;
    draw.first_instance.checked_add(draw.instances.saturating_sub(1))
        .ok_or(Refusal::ArgumentOverflow)?;
    if primitive_count > 1 || draw.instances > 1 || draw.count != vertices_per_primitive {
        check_shader(vertex, true, draw, primitive_count)?;
        check_shader(fragment, false, draw, primitive_count)?;
    }
    Ok(Some(Primitives { draw, primitive_count, vertices_per_primitive, stride }))
}

/// Shared by render-pass declaration and command recording, so the recorded
/// barrier can never exceed its subpass self-dependency's permitted scopes.
pub fn dependency() -> vk::SubpassDependency {
    vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        .dependency_flags(vk::DependencyFlags::BY_REGION)
}

fn check_shader(
    words: &[u32],
    vertex: bool,
    draw: Draw,
    primitive_count: u32,
) -> Result<(), Refusal> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Refusal::MalformedShader);
    }
    let mut pointer_classes = BTreeMap::new();
    let mut value_types = BTreeMap::new();
    let mut stores = Vec::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let len = (words[cursor] >> 16) as usize;
        let op = words[cursor] & 0xffff;
        if len == 0 || cursor + len > words.len() {
            return Err(Refusal::MalformedShader);
        }
        let args = &words[cursor + 1..cursor + len];
        // OpDecorate / OpMemberDecorate: BuiltIn is decoration 11.
        let builtin = match (op, args) {
            (71, [_, 11, builtin, ..]) | (72, [_, _, 11, builtin, ..]) => Some(*builtin),
            _ => None,
        };
        if let Some(builtin) = builtin {
            let changed = match builtin {
                7 => primitive_count > 1, // PrimitiveId
                4424 => primitive_count > 1 && !draw.indexed, // BaseVertex
                4425 => draw.instances > 1, // BaseInstance
                36..=41 | 4416..=4420 if vertex => true, // Vertex subgroup identity
                4992..=4998 | 5286 | 5287 => true, // AMD/NV/KHR barycentrics
                _ => false,
            };
            if changed {
                return Err(Refusal::SplitBuiltin(builtin));
            }
        }
        if vertex {
            match (op, args) {
                (32, [id, class, ..]) => { pointer_classes.insert(*id, *class); }
                // Every ordinary pointer-producing instruction. An unknown
                // producer feeding a store refuses rather than hiding a write.
                (55 | 57 | 59 | 61 | 65 | 66 | 67 | 70 | 83 | 169 | 245,
                    [ty, id, ..]) => { value_types.insert(*id, *ty); }
                (62..=64, [pointer, ..]) => stores.push(*pointer),
                // ImageWrite, atomics, and vertex subgroup communication.
                (99 | 227..=242 | 318 | 319 | 333..=366 | 5614 | 5615 | 6035, _) => {
                    return Err(Refusal::VertexSideEffect);
                }
                _ => {}
            }
        }
        cursor += len;
    }
    for pointer in stores {
        let class = value_types.get(&pointer).and_then(|ty| pointer_classes.get(ty));
        if !matches!(class, Some(3 | 6 | 7)) { // Output, Private, Function
            return Err(Refusal::VertexSideEffect);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader(instructions: &[u32]) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
        words.extend_from_slice(instructions);
        words
    }

    fn draw(topology: PrimitiveType, count: u32) -> Draw {
        Draw {
            reads_attachment: true, topology, count, first: 9,
            instances: 1, first_instance: 4, indexed: false, samples: 1,
            polygon_mode: vk::PolygonMode::FILL,
        }
    }

    fn primitives(draw: Draw) -> Vec<Primitive> {
        plan(Cell { dynamic_front_face: true, ..Cell::default() }, draw,
            &shader(&[]), &shader(&[])).unwrap().map(|p| p.iter().collect()).unwrap_or_default()
    }

    #[test]
    fn attachment_fetch_list_preserves_vertices_instances_and_tail() {
        let mut draw = draw(PrimitiveType::Triangle, 8);
        draw.instances = 2;
        assert_eq!(primitives(draw), vec![
            Primitive { first: 9, count: 3, first_instance: 4, reverse_winding: false },
            Primitive { first: 12, count: 3, first_instance: 4, reverse_winding: false },
            Primitive { first: 9, count: 3, first_instance: 5, reverse_winding: false },
            Primitive { first: 12, count: 3, first_instance: 5, reverse_winding: false },
        ]);
    }

    #[test]
    fn attachment_fetch_strip_preserves_index_windows_and_winding() {
        let mut draw = draw(PrimitiveType::TriangleStrip, 5);
        draw.indexed = true;
        draw.first = 0;
        assert_eq!(primitives(draw), vec![
            Primitive { first: 0, count: 3, first_instance: 4, reverse_winding: false },
            Primitive { first: 1, count: 3, first_instance: 4, reverse_winding: true },
            Primitive { first: 2, count: 3, first_instance: 4, reverse_winding: false },
        ]);
    }

    #[test]
    fn attachment_fetch_points_lines_and_empty_draws() {
        for (topology, count, firsts) in [
            (PrimitiveType::Point, 3, vec![9, 10, 11]),
            (PrimitiveType::Line, 5, vec![9, 11]),
            (PrimitiveType::LineStrip, 4, vec![9, 10, 11]),
            (PrimitiveType::Triangle, 2, vec![]),
        ] {
            assert_eq!(primitives(draw(topology, count)).iter().map(|p| p.first)
                .collect::<Vec<_>>(), firsts);
        }
        let mut empty = draw(PrimitiveType::Triangle, 6);
        empty.instances = 0;
        assert!(primitives(empty).is_empty());
        assert!(plan(Cell::default(), draw(PrimitiveType::Triangle, 2), &[], &[])
            .unwrap().is_none(), "incomplete primitive vertex work stays in the original draw");
    }

    #[test]
    fn attachment_fetch_native_ordering_does_not_split_or_inspect() {
        let d = draw(PrimitiveType::TriangleStrip, 100);
        assert!(plan(Cell { ordered_color_access: true, ..Cell::default() }, d, &[], &[])
            .unwrap().is_none());
        assert!(plan(Cell::default(), Draw { reads_attachment: false, ..d }, &[], &[])
            .unwrap().is_none());
    }

    #[test]
    fn attachment_fetch_missing_winding_multisample_wireframe_and_overflow_refuse() {
        for (d, expected) in [
            (draw(PrimitiveType::TriangleStrip, 4), Refusal::StripWinding),
            (Draw { samples: 4, ..draw(PrimitiveType::Triangle, 3) }, Refusal::Multisample),
            (Draw { polygon_mode: vk::PolygonMode::LINE, ..draw(PrimitiveType::Triangle, 3) },
                Refusal::TriangleWireframe),
            (Draw { first: u32::MAX, ..draw(PrimitiveType::Triangle, 3) },
                Refusal::ArgumentOverflow),
            (Draw { first_instance: u32::MAX, instances: 2,
                ..draw(PrimitiveType::Triangle, 3) }, Refusal::ArgumentOverflow),
        ] {
            assert_eq!(plan(Cell::default(), d, &shader(&[]), &shader(&[])).unwrap_err(),
                expected);
        }
    }

    #[test]
    fn attachment_fetch_observable_split_builtins_refuse() {
        let mut d = draw(PrimitiveType::Triangle, 6);
        d.instances = 2;
        for builtin in [7, 41, 4416, 4424, 4425, 4994, 4998, 5286, 5287] {
            let decorated = shader(&[(4 << 16) | 71, 1, 11, builtin]);
            assert_eq!(plan(Cell::default(), d, &decorated, &shader(&[])).unwrap_err(),
                Refusal::SplitBuiltin(builtin));
        }
    }

    #[test]
    fn attachment_fetch_vertex_memory_writes_cannot_be_replayed() {
        let d = draw(PrimitiveType::Triangle, 6);
        for class in [2, 12, 5349] {
            let vertex = shader(&[
                (4 << 16) | 32, 1, class, 9,
                (4 << 16) | 59, 1, 2, class,
                (5 << 16) | 65, 1, 3, 2, 4,
                (3 << 16) | 62, 3, 4,
            ]);
            assert_eq!(plan(Cell::default(), d, &vertex, &shader(&[])).unwrap_err(),
                Refusal::VertexSideEffect);
        }
        for op in [99, 234, 333] {
            assert_eq!(plan(Cell::default(), d, &shader(&[(1 << 16) | op]), &shader(&[]))
                .unwrap_err(), Refusal::VertexSideEffect);
        }
    }

    #[test]
    fn attachment_fetch_local_vertex_stores_and_fragment_side_effects_remain_legal() {
        let d = draw(PrimitiveType::Triangle, 6);
        for class in [3, 6, 7] {
            let vertex = shader(&[
                (4 << 16) | 32, 1, class, 9,
                (4 << 16) | 59, 1, 2, class,
                (3 << 16) | 62, 2, 4,
            ]);
            assert!(plan(Cell::default(), d, &vertex, &shader(&[(1 << 16) | 99])).is_ok());
        }
    }

    #[test]
    fn attachment_fetch_indexed_base_vertex_and_single_instance_base_are_unchanged() {
        let d = Draw { indexed: true, ..draw(PrimitiveType::Triangle, 6) };
        let vertex = shader(&[
            (4 << 16) | 71, 1, 11, 4424,
            (4 << 16) | 71, 2, 11, 4425,
        ]);
        assert!(plan(Cell::default(), d, &vertex, &shader(&[])).is_ok());
    }

    #[test]
    fn attachment_fetch_malformed_shader_and_unknown_store_pointer_refuse() {
        let d = draw(PrimitiveType::Triangle, 6);
        for vertex in [vec![], shader(&[0]), shader(&[(9 << 16) | 59])] {
            assert_eq!(plan(Cell::default(), d, &vertex, &shader(&[])).unwrap_err(),
                Refusal::MalformedShader);
        }
        assert_eq!(plan(Cell::default(), d, &shader(&[(3 << 16) | 62, 2, 3]),
            &shader(&[])).unwrap_err(), Refusal::VertexSideEffect);
    }

    #[test]
    fn attachment_fetch_barrier_matches_self_dependency() {
        let d = dependency();
        assert_eq!(d.src_subpass, d.dst_subpass);
        assert_eq!(d.src_stage_mask, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(d.dst_stage_mask, vk::PipelineStageFlags::FRAGMENT_SHADER);
        assert_eq!(d.src_access_mask, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        assert_eq!(d.dst_access_mask, vk::AccessFlags::INPUT_ATTACHMENT_READ);
        assert_eq!(d.dependency_flags, vk::DependencyFlags::BY_REGION);
    }
}
