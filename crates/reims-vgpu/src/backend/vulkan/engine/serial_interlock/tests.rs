use super::*;
use crate::backend::vulkan::engine::{
    graphics_storage::*, pass_local::PassLocalTarget, StorageImageFormat,
};
use crate::runtime::draw::vulkan::InterlockIsolation;

fn instruction(words: &mut Vec<u32>, op: u32, args: &[u32]) {
    words.push(((args.len() as u32 + 1) << 16) | op);
    words.extend_from_slice(args);
}

pub(super) fn fragment() -> Arc<Vec<u32>> {
    let mut w = vec![0x0723_0203, 0x0001_0400, 0, 64, 0];
    instruction(&mut w, 17, &[1]);
    instruction(&mut w, 17, &[5378]);
    let mut text = b"SPV_EXT_fragment_shader_interlock\0".to_vec();
    while text.len() % 4 != 0 {
        text.push(0);
    }
    let text: Vec<_> = text
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    instruction(&mut w, 10, &text);
    instruction(&mut w, 14, &[0, 1]);
    instruction(&mut w, 15, &[4, 1, 0x6e69_616d, 0, 8]);
    instruction(&mut w, 16, &[1, 7]);
    instruction(&mut w, 16, &[1, 5366]);
    instruction(&mut w, 71, &[8, 34, 0]);
    instruction(&mut w, 71, &[8, 33, 1155]);
    instruction(&mut w, 19, &[2]);
    instruction(&mut w, 33, &[3, 2]);
    instruction(&mut w, 22, &[4, 32]);
    instruction(&mut w, 23, &[5, 4, 4]);
    instruction(&mut w, 25, &[6, 4, 1, 0, 0, 0, 2, 2]);
    instruction(&mut w, 32, &[7, 0, 6]);
    instruction(&mut w, 59, &[7, 8, 0]);
    instruction(&mut w, 21, &[9, 32, 0]);
    instruction(&mut w, 23, &[10, 9, 2]);
    instruction(&mut w, 43, &[9, 11, 0]);
    instruction(&mut w, 44, &[10, 12, 11, 11]);
    instruction(&mut w, 43, &[4, 13, 0]);
    instruction(&mut w, 44, &[5, 14, 13, 13, 13, 13]);
    instruction(&mut w, 54, &[2, 1, 0, 3]);
    instruction(&mut w, 248, &[15]);
    instruction(&mut w, 5364, &[]);
    instruction(&mut w, 61, &[6, 16, 8]);
    instruction(&mut w, 99, &[16, 12, 14]);
    instruction(&mut w, 5365, &[]);
    instruction(&mut w, 253, &[]);
    instruction(&mut w, 56, &[]);
    Arc::new(w)
}

fn request() -> DrawRequest {
    let target = PassLocalTarget::new(4, 2, vk::Format::R16G16B16A16_SFLOAT).unwrap();
    let mut req = DrawRequest {
        width: 4,
        height: 2,
        vertex_count: 6,
        instance_count: Some(2),
        primitive_topology: super::super::types::PrimitiveTopology(
            reims_vgpu_core::topology::PrimitiveType::Triangle,
        ),
        target_identity: Some(target.identity().clone()),
        vert_spirv: vertex(),
        frag_spirv: fragment(),
        storage_textures: vec![GraphicsStorageTexture {
            format: StorageImageFormat::Rgba16Float,
            width: 4,
            height: 2,
            bytes: vec![0; 64],
            bindings: vec![GraphicsTextureBinding {
                binding: 1155,
                access: GraphicsTextureAccess::Storage,
                stage: vk::ShaderStageFlags::FRAGMENT,
            }],
        }],
        ..Default::default()
    };
    req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
    req
}

fn vertex() -> Arc<Vec<u32>> {
    let mut words = vec![0x0723_0203, 0x0001_0400, 0, 16, 0];
    instruction(&mut words, 17, &[1]);
    instruction(&mut words, 14, &[0, 1]);
    instruction(&mut words, 15, &[0, 1, 0x6e69_616d, 0]);
    instruction(&mut words, 19, &[2]);
    instruction(&mut words, 33, &[3, 2]);
    instruction(&mut words, 54, &[2, 1, 0, 3]);
    instruction(&mut words, 248, &[4]);
    instruction(&mut words, 253, &[]);
    instruction(&mut words, 56, &[]);
    Arc::new(words)
}

fn programs(req: &DrawRequest) -> Result<(Arc<Program>, Arc<VertexDeclarations>), Refusal> {
    Ok((
        Program::analyze(&req.frag_spirv)?,
        VertexDeclarations::analyze(&req.vert_spirv),
    ))
}

fn sampled(binding: u32) -> super::super::types::SampledImageResource {
    use super::super::types::*;
    SampledImageResource {
        binding,
        array_element: 0,
        descriptor_count: 1,
        width: 1,
        height: 1,
        layers: 1,
        kind: reims_vgpu_core::texture_shape::TextureKind::D2,
        multisampled: false,
        source: SampledSource::Bytes(Arc::new(vec![17, 33, 65, 255])),
        byte_origin: SampledByteOrigin::Synthetic,
        format: vk::Format::R8G8B8A8_UNORM,
        identity: None,
        swizzle: Default::default(),
    }
}

fn insert_before(words: &mut Vec<u32>, opcode: u32, extra: Vec<u32>) {
    let mut at = 5;
    while at < words.len() {
        if words[at] & 0xffff == opcode {
            words.splice(at..at, extra);
            return;
        }
        at += (words[at] >> 16) as usize;
    }
    panic!("missing insertion opcode {opcode}");
}

pub(in crate::backend::vulkan::engine) fn with_resource(
    source: &Arc<Vec<u32>>, binding: u32, sampled: bool,
) -> Arc<Vec<u32>> {
    let mut words = (**source).clone();
    let mut at = 5;
    let mut float = None;
    while at < words.len() {
        if words[at] & 0xffff == 22 && words[at + 2] == 32 {
            float = Some(words[at + 1]);
        }
        at += (words[at] >> 16) as usize;
    }
    let first = words[3];
    words[3] += 4;
    let (ty, pointer, variable) = (first + 1, first + 2, first + 3);
    let mut globals = Vec::new();
    let float = float.unwrap_or_else(|| {
        instruction(&mut globals, 22, &[first, 32]);
        first
    });
    let class = if sampled { 0 } else { 12 };
    if sampled {
        instruction(&mut globals, 25, &[ty, float, 1, 0, 0, 0, 1, 0]);
    } else {
        instruction(&mut globals, 30, &[ty, float]);
    }
    instruction(&mut globals, 32, &[pointer, class, ty]);
    instruction(&mut globals, 59, &[pointer, variable, class]);
    let mut decorations = Vec::new();
    instruction(&mut decorations, 71, &[variable, 33, binding]);
    instruction(&mut decorations, 71, &[variable, 34, 0]);
    if !sampled {
        instruction(&mut decorations, 71, &[ty, 2]);
        instruction(&mut decorations, 72, &[ty, 0, 35, 0]);
    }
    insert_before(&mut words, 19, decorations);
    insert_before(&mut words, 54, globals);
    Arc::new(words)
}

#[test]
fn serial_interlock_admission_ignores_only_complete_two_stage_absence() {
    use super::super::types::{BufferContent, StorageBufferResource};
    let mut req = request();
    req.sampled_images = [5, 7, 10, 12, 4, 6, 8]
        .into_iter()
        .map(|index| sampled(704 + index))
        .collect();
    req.storage_buffers = [1, 2, 0, 3, 4, 5, 7, 8]
        .into_iter()
        .map(|index| StorageBufferResource {
            binding: 672 + index,
            content: BufferContent::Bytes(Arc::new(vec![0xa5; 16])),
        })
        .collect();
    let (fragment, vertex) = programs(&req).unwrap();
    assert!(
        DeclarationProof::of_final_module(&req.frag_spirv)
            .complete_bindings()
            .is_none(),
        "the unsupported original interlock module cannot supply the native declaration proof"
    );
    assert_eq!(
        fragment.declarations.complete_bindings(),
        Some([1155].as_slice())
    );
    assert_eq!(vertex.declarations.complete_bindings(), Some([].as_slice()));
    Plan::admit(&req, true, vk::PolygonMode::FILL, || Ok((fragment, vertex))).unwrap();
    assert_eq!(req.sampled_images.len(), 7);
    assert_eq!(req.storage_buffers.len(), 8);
    assert_eq!(req.storage_textures.len(), 1);
}

#[test]
fn serial_interlock_rejects_actual_sampled_and_fragment_buffer_declarations() {
    for fragment_stage in [false, true] {
        let mut req = request();
        req.sampled_images.push(sampled(709));
        if fragment_stage {
            req.frag_spirv = with_resource(&req.frag_spirv, 709, true);
        } else {
            req.vert_spirv = with_resource(&req.vert_spirv, 709, true);
        }
        req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
        let refusal = Plan::admit(&req, true, vk::PolygonMode::FILL, || programs(&req))
            .err()
            .unwrap();
        assert_eq!(
            refusal,
            if fragment_stage {
                Refusal::Shader("only_storage_and_input_images")
            } else {
                Refusal::State("isolated_fragment_storage_required")
            }
        );
    }
    let mut req = request();
    req.frag_spirv = with_resource(&req.frag_spirv, 672, false);
    req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
    assert!(matches!(
        Plan::admit(&req, true, vk::PolygonMode::FILL, || programs(&req)),
        Err(Refusal::Shader("fragment_buffer_resource")),
    ));
}

#[test]
fn serial_interlock_unknown_grouped_malformed_and_unsupported_proofs_refuse() {
    for fragment_stage in [false, true] {
        for case in 0..4 {
            let mut req = request();
            req.sampled_images.push(sampled(709));
            let words = if fragment_stage {
                &mut req.frag_spirv
            } else {
                &mut req.vert_spirv
            };
            let words = Arc::make_mut(words);
            match case {
                0 => words.push(0),
                1 => instruction(words, 73, &[12]),
                2 => words[1] = 0x0001_0700,
                _ => instruction(words, 6000, &[]),
            }
            req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
            assert!(
                matches!(
                    Plan::admit(&req, true, vk::PolygonMode::FILL, || programs(&req)),
                    Err(Refusal::Shader(_)) | Err(Refusal::Split(_)),
                ),
                "unknown proof stage={fragment_stage} case={case}"
            );
        }
    }
}

#[test]
fn serial_interlock_cached_declaration_proof_is_bound_to_both_final_shader_arcs() {
    for fragment_stage in [false, true] {
        let mut req = request();
        let (fragment, vertex) = programs(&req).unwrap();
        Plan::admit(&req, true, vk::PolygonMode::FILL, || {
            Ok((fragment.clone(), vertex.clone()))
        })
        .unwrap();
        let words = if fragment_stage {
            &mut req.frag_spirv
        } else {
            &mut req.vert_spirv
        };
        *words = Arc::new((**words).clone());
        req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
        assert!(matches!(
            Plan::admit(&req, true, vk::PolygonMode::FILL, || Ok((fragment, vertex))),
            Err(Refusal::State("shader_identity_changed")),
        ));
        Plan::admit(&req, true, vk::PolygonMode::FILL, || programs(&req)).unwrap();
    }
}

#[test]
fn serial_interlock_projection_preserves_every_non_ordering_instruction() {
    let original = fragment();
    let plan = Plan::admit(&request(), true, vk::PolygonMode::FILL, || {
        let req = request();
        programs(&req)
    });
    assert!(matches!(
        plan,
        Err(Refusal::State("shader_identity_changed"))
    ));
    let lowered = lower(&original).unwrap();
    assert!(!crate::runtime::spirv_bind::requires_pixel_interlock(
        &lowered
    ));
    let keep = |words: &[u32]| {
        let mut result = words[..5].to_vec();
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let op = words[at] & 65535;
            let removed = matches!(op, 10 | 5364 | 5365)
                || (op == 17 && words[at + 1] == 5378)
                || (op == 16 && matches!(words[at + 2], 5366 | 5367));
            if !removed {
                result.extend_from_slice(&words[at..at + count]);
            }
            at += count;
        }
        result
    };
    assert_eq!(lowered, keep(&original));
}

#[test]
fn serial_interlock_requires_live_matching_isolation_before_projecting() {
    let mut req = request();
    req.interlock_isolation = None;
    assert!(matches!(
        Plan::admit(&req, true, vk::PolygonMode::FILL, || panic!(
            "projection without proof"
        )),
        Err(Refusal::State("isolation_not_supplied"))
    ));
    req.interlock_isolation = Some(Ok(InterlockIsolation::host_owned_fixture(&req)));
    req.storage_textures[0].bindings[0].binding += 1;
    assert!(matches!(
        Plan::admit(&req, true, vk::PolygonMode::FILL, || panic!(
            "changed binding"
        )),
        Err(Refusal::Isolation(
            InterlockIsolationRefusal::BindingChanged
        ))
    ));
}

#[test]
fn serial_interlock_retains_guarded_primitive_and_instance_order() {
    let req = request();
    let plan = Plan::admit(&req, true, vk::PolygonMode::FILL, || programs(&req)).unwrap();
    let indices: Vec<_> = plan
        .primitives()
        .iter()
        .map(|p| (p.first, p.first_instance))
        .collect();
    assert_eq!(indices, vec![(0, 0), (3, 0), (0, 1), (3, 1)]);
}

#[test]
fn serial_interlock_refuses_queries_msaa_wireframe_and_split_sensitive_builtins() {
    for kind in 0..5 {
        let mut req = request();
        match kind {
            0 => req.occlusion_query = Some(super::super::types::VisibilityResultMode::Boolean),
            1 => {
                req.raster_sample_count = 4;
                req.color_sample_count = 4;
            }
            2 => {}
            3 => {
                let vertex = Arc::make_mut(&mut req.vert_spirv);
                instruction(vertex, 71, &[1, 11, 7]);
            }
            _ => {
                let vertex = Arc::make_mut(&mut req.vert_spirv);
                instruction(vertex, 99, &[1, 1, 1]);
            }
        }
        let polygon = if kind == 2 {
            vk::PolygonMode::LINE
        } else {
            vk::PolygonMode::FILL
        };
        assert!(Plan::admit(&req, true, polygon, || panic!("unsafe projection")).is_err());
    }
}

#[test]
fn serial_interlock_rejects_unhandled_effects_and_unbalanced_regions() {
    for extra in [
        vec![(1 << 16) | 227],
        vec![(1 << 16) | 252],
        vec![(1 << 16) | 5364],
        vec![(1 << 16) | 6000],
        vec![(4 << 16) | 62, 50, 14, 0],
    ] {
        let mut words = (*fragment()).clone();
        words.extend(extra);
        assert!(lower(&words).is_err());
    }
}

#[test]
fn serial_interlock_global_dependency_and_continuation_preserve_all_colors() {
    let (stages, barrier) = dependency();
    assert_eq!(stages, vk::PipelineStageFlags::ALL_COMMANDS);
    assert!(barrier
        .src_access_mask
        .contains(vk::AccessFlags::MEMORY_WRITE));
    assert!(barrier
        .dst_access_mask
        .contains(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE));
    let mut pass = PassKey::single(
        super::super::caches::Color0Load::Clear,
        vk::Format::B8G8R8A8_UNORM,
    );
    pass.secondary_count = 2;
    pass.secondary[0].format = vk::Format::R16G16B16A16_SFLOAT;
    pass.secondary[1].format = vk::Format::R32_SFLOAT;
    let preserved = preserving_pass(pass);
    assert_eq!(preserved.compatibility(), pass.compatibility());
    assert_eq!(
        preserved.color0_load,
        super::super::caches::Color0Load::Preserve
    );
    assert!(preserved.secondary[..2]
        .iter()
        .all(|attachment| attachment.load));
}

mod gpu;
