use super::*;

fn instanced_vertex_source() -> String {
    VERTEX
        .replace("%index %position", "%index %instance %position %step_output")
        .replace("OpDecorate %position BuiltIn Position", r#"OpDecorate %position BuiltIn Position
OpDecorate %instance BuiltIn InstanceIndex
OpDecorate %step_output Location 0
OpDecorate %step_output Flat"#)
        .replace("%index = OpVariable %input Input", r#"%index = OpVariable %input Input
%instance = OpVariable %input Input
%step_pointer = OpTypePointer Output %float
%step_output = OpVariable %step_pointer Output
%three = OpConstant %uint 3
%quarter = OpConstant %float 0.25"#)
        .replace("%i = OpLoad %uint %index", r#"%vertex_id = OpLoad %uint %index
%i = OpUMod %uint %vertex_id %three
%group = OpUDiv %uint %vertex_id %three
%instance_id = OpLoad %uint %instance
%instance_step = OpIMul %uint %instance_id %three
%group_step = OpIAdd %uint %instance_step %group
%step_index = OpIAdd %uint %group_step %one
%step_float = OpConvertUToF %float %step_index
%step = OpFMul %float %step_float %quarter
OpStore %step_output %step"#)
}

fn instanced_fragment_source() -> String {
    FRAGMENT
        .replace("%tile %output %tile_output", "%tile %output %tile_output %step_input")
        .replace("OpDecorate %output Location 0", r#"OpDecorate %output Location 0
OpDecorate %step_input Location 0
OpDecorate %step_input Flat"#)
        .replace("%tile = OpVariable %uniform UniformConstant", r#"%tile = OpVariable %uniform UniformConstant
%step_pointer = OpTypePointer Input %float
%step_input = OpVariable %step_pointer Input"#)
        .replace("%next = OpFAdd %float %r %quarter", r#"%step = OpLoad %float %step_input
%next = OpFAdd %float %r %step"#)
}

fn strip_vertex_source() -> String {
    VERTEX
        .replace("%xx = OpFMul %float %x %f2", "%xx = OpCopyObject %float %x")
        .replace("%yy = OpFMul %float %y %f2", "%yy = OpCopyObject %float %y")
}

fn facing_fragment_source() -> String {
    FRAGMENT
        .replace("%tile %output %tile_output", "%tile %output %tile_output %front")
        .replace("OpDecorate %output Location 0", r#"OpDecorate %output Location 0
OpDecorate %front BuiltIn FrontFacing"#)
        .replace("%tile = OpVariable %uniform UniformConstant", r#"%tile = OpVariable %uniform UniformConstant
%front_pointer = OpTypePointer Input %bool
%front = OpVariable %front_pointer Input"#)
        .replace("%next = OpFAdd %float %r %quarter", r#"%facing = OpLoad %bool %front
%step = OpSelect %float %facing %quarter %nine
%next = OpFAdd %float %r %step"#)
}

fn indices(values: &[u16], vertex_offset: i32) -> engine::IndexedDrawResource {
    engine::IndexedDrawResource {
        index_type: engine::IndexType::U16,
        index_count: values.len() as u32,
        vertex_offset,
        content: engine::BufferContent::from(
            values.iter().flat_map(|index| index.to_le_bytes()).collect::<Vec<_>>()),
    }
}

fn draw_once(
    state: &DeviceState,
    vertex: Arc<Vec<u32>>,
    fragment: Arc<Vec<u32>>,
    blend_tile: bool,
    configure: impl FnOnce(&mut DrawRequest),
) -> Vec<u8> {
    let mut req = native_pair_request();
    req.render_pass_continues = false;
    let mut pass = VulkanRenderPass::default();
    let mut identities = Vec::new();
    let mut pixels = Vec::new();
    let result = pass.with_draw(&mut req, |owner, req| {
        let mut draw = native_draw(owner, req, vertex, fragment, blend_tile);
        configure(&mut draw);
        identities.push(draw.target_identity.clone().unwrap());
        identities.push(draw.secondary_targets[0].identity.clone());
        engine::execute_draw_request(state, &draw).expect("ordered native MRT draw");
        pixels = engine::pass_local::read_native_for_test(
            draw.target_identity.as_ref().unwrap(),
        ).expect("test-only native readback, not a guest Store");
        (EncodeStatus::Ok, None)
    });
    assert!(matches!(result.0, EncodeStatus::Ok), "pass admission: {:?}", result.0);
    assert_eq!(pixels.len(), 32 * 4);
    assert!(identities.iter().all(|identity| !engine::resident_content_ready(identity)),
        "every completed single-request pass retires both native attachments");
    pixels
}

#[test]
#[ignore = "requires the parent's exclusive GPU slot"]
fn memoryless_r32float_gpu_intradraw_primitives_and_instances() {
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let vertex = assemble(&instanced_vertex_source());
    let fragment = assemble(&instanced_fragment_source());
    for indexed in [false, true] {
        for blend_tile in [false, true] {
            let mut expected = [0.0f32; 4];
            let mut tile = 2.0000009536743164f32;
            let groups = if indexed { [1, 0] } else { [1, 2] };
            for instance in 5..8 {
                for group in groups {
                    let next = tile + (1 + group + instance * 3) as f32 * 0.25;
                    let source = [next * 0.25, 0.125, 0.0625, 0.25];
                    for (dst, src) in expected.iter_mut().zip(source) {
                        *dst = src + *dst * 0.75;
                    }
                    tile = if blend_tile { next + tile * 0.75 } else { next };
                }
            }
            let pixels = draw_once(&state, vertex.clone(), fragment.clone(), blend_tile, |draw| {
                draw.vertex_count = 6;
                draw.instance_count = Some(3);
                draw.base_instance = 5;
                if !indexed && !blend_tile {
                    // DrawRequest's unspecified zero counts mean one sample;
                    // fallback admission must use the engine-normalized count.
                    draw.raster_sample_count = 0;
                    draw.color_sample_count = 0;
                }
                if indexed {
                    // Effective VertexIndex groups 1 then 0, with original
                    // index ordering and signed baseVertex both observable.
                    draw.indexed = Some(indices(&[6, 7, 8, 3, 4, 5], -3));
                } else {
                    draw.first_vertex = 3;
                }
            });
            for (pixel, texel) in pixels.chunks_exact(4).enumerate() {
                for (component, bytes) in texel.chunks_exact(4).enumerate() {
                    let actual = f32::from_le_bytes(bytes.try_into().unwrap());
                    let wanted = expected[component];
                    assert!((actual - wanted).abs() <= wanted.abs().max(1.0) * 0.000001,
                        "indexed={indexed} blend_tile={blend_tile} pixel={pixel} \
                         component={component} actual={actual} expected={wanted}");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires the parent's exclusive GPU slot; strip fallback needs dynamic front-face"]
fn memoryless_r32float_gpu_strip_winding_cull_and_frontfacing() {
    use reims_vgpu_core::topology::PrimitiveType;
    use reims_vgpu_vulkan::raster::*;
    crate::observe::redirect_logs_for_tests();
    let state = DeviceState::new(crate::model::DeviceId(1), crate::model::PAGE_SHIFT_ARM64E);
    engine::test_reset_engine(&state);
    let vertex = assemble(&strip_vertex_source());
    let fragment = assemble(&facing_fragment_source());
    for winding in [MTL_WINDING_CLOCKWISE, MTL_WINDING_COUNTER_CLOCKWISE] {
        for cull_mode in [MTL_CULL_MODE_NONE, MTL_CULL_MODE_FRONT, MTL_CULL_MODE_BACK] {
            let raster = GuestRasterState { winding, cull_mode, ..GuestRasterState::DEFAULT };
            let baseline = draw_once(&state, vertex.clone(), fragment.clone(), true, |draw| {
                draw.vertex_count = 6;
                draw.instance_count = Some(2);
                draw.indexed = Some(indices(&[0, 1, 2, 2, 1, 3], 0));
                draw.raster = raster;
            });
            if cull_mode == MTL_CULL_MODE_NONE {
                assert!(baseline.chunks_exact(4).all(|texel|
                    f32::from_le_bytes(texel[..4].try_into().unwrap()) > 0.0),
                    "the explicit two-triangle list must cover both halves of the target");
            }
            for indexed in [false, true] {
                let strip = draw_once(&state, vertex.clone(), fragment.clone(), true, |draw| {
                    draw.vertex_count = 4;
                    draw.instance_count = Some(2);
                    draw.primitive_topology = engine::PrimitiveTopology(PrimitiveType::TriangleStrip);
                    draw.raster = raster;
                    if indexed { draw.indexed = Some(indices(&[0, 1, 2, 3], 0)); }
                });
                assert_eq!(strip, baseline,
                    "strip odd parity must preserve culling and FrontFacing: \
                     indexed={indexed} winding={winding} cull={cull_mode}");
            }
        }
    }
}

#[test]
#[ignore = "requires SPIRV-Tools; CPU-only, does not initialize Vulkan"]
fn memoryless_gpu_fixtures_validate_vulkan12() {
    VulkanRenderPass::default().prepare(&native_pair_request())
        .expect("GPU fixtures must use the same admitted memoryless contract as guest draws");
    for source in [
        VERTEX.to_owned(), FRAGMENT.to_owned(),
        instanced_vertex_source(), instanced_fragment_source(),
        strip_vertex_source(), facing_fragment_source(),
    ] {
        let words = assemble(&source);
        let mut child = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2", "-"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
            .spawn().expect("spirv-val is required by this CPU-only fixture validation");
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
    use reims_vgpu_core::topology::PrimitiveType;
    use reims_vgpu_vulkan::framebuffer_fetch::{self, Cell, Draw};
    for (topology, count, vertex, fragment) in [
        (PrimitiveType::Triangle, 6, instanced_vertex_source(), instanced_fragment_source()),
        (PrimitiveType::TriangleStrip, 4, strip_vertex_source(), facing_fragment_source()),
    ] {
        for indexed in [false, true] {
            let plan = framebuffer_fetch::plan(
                Cell { ordered_color_access: false, dynamic_front_face: true },
                Draw {
                    reads_attachment: true, topology, count,
                    first: if indexed { 0 } else { 3 },
                    instances: 3, first_instance: 5, indexed,
                    samples: 1, polygon_mode: ash::vk::PolygonMode::FILL,
                },
                &assemble(&vertex), &assemble(&fragment),
            ).expect("authored shader observations must permit the ordered fallback").unwrap();
            assert_eq!(plan.iter().count(), 6);
        }
    }
}
