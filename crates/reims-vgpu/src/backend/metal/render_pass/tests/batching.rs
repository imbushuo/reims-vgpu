use super::*;
use crate::backend::blob::BlobKey;
use crate::backend::metal::abi::*;
use crate::backend::metal::input::{self, Class as InputClass};
use crate::backend::metal::render::{
    render_core_mrt, render_core_mrt_inputs, RenderBuffers, VisibilityQuery,
};

const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 batch_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    struct VertexInput { float2 p [[attribute(0)]]; };
    vertex float4 batch_input_vertex(VertexInput v [[stage_in]],
                                    constant float2 &shift [[buffer(1)]]) {
        return float4(v.p + shift, 0, 1);
    }
    struct Colors { half4 a [[color(0)]]; half4 b [[color(1)]]; };
    fragment Colors batch_fragment(constant float4 &color [[buffer(0)]]) {
        return {half4(color), half4(color.zyxw)};
    }
"#;
const VERT: &[u8] = b"reims-owned-test-batch-vertex";
const FRAG: &[u8] = b"reims-owned-test-batch-fragment";
const INPUT_VERT: &[u8] = b"reims-owned-test-batch-input-vertex";
const NATIVE_VERT: &[u8] = b"reims-owned-test-batch-filled-vertex";
const NATIVE_FRAG: &[u8] = b"reims-owned-test-batch-filled-fragment";

fn native_shaders(device: &Device) {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 filled_vertex(uint i [[vertex_id]], uint instance [[instance_id]],
            constant uint *words [[buffer(1)]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            bool valid = words[0] == 0x12345678 && words[4] == 0xabcdef01
                && instance == 9 && i >= 3 && i < 6;
            return valid ? float4(p[i - 3], 0, 1) : float4(4, 4, 0, 1);
        }
        struct Colors { half4 a [[color(0)]]; half4 b [[color(1)]]; };
        fragment Colors filled_fragment(constant float4 *colors [[buffer(0)]]) {
            float4 color = colors[2];
            return {half4(color), half4(color.zyxw)};
        }
    "#;
    let library =
        crate::backend::metal::raw_metal::new_library_with_source(device, SOURCE).unwrap();
    for (key, function) in [
        (NATIVE_VERT, "filled_vertex"),
        (NATIVE_FRAG, "filled_fragment"),
    ] {
        crate::backend::metal::cache::fn_cache_insert(
            &BlobKey::new(key),
            library.get_function(function, None).unwrap(),
        );
    }
}

fn shaders(device: &Device) -> Library {
    let library =
        crate::backend::metal::raw_metal::new_library_with_source(device, SOURCE).unwrap();
    crate::backend::metal::cache::fn_cache_insert(
        &BlobKey::new(VERT),
        library.get_function("batch_vertex", None).unwrap(),
    );
    crate::backend::metal::cache::fn_cache_insert(
        &BlobKey::new(FRAG),
        library.get_function("batch_fragment", None).unwrap(),
    );
    crate::backend::metal::cache::fn_cache_insert(
        &BlobKey::new(INPUT_VERT),
        library.get_function("batch_input_vertex", None).unwrap(),
    );
    library
}

fn ordinary_request() -> DrawEncodeRequest {
    let colors = (0..2)
        .map(|slot| ColorRtRequest {
            storage: ColorStorage::GuestBacked,
            slot,
            texture_ref: 100 + slot,
            target_gva: 0x4000 * u64::from(slot + 1),
            row_stride: 32,
            width: 8,
            height: 4,
            format: MTLPixelFormat::RGBA8Unorm as u16,
            sample_count: 1,
            load_action: if slot == 0 {
                MTL_LOAD_ACTION_LOAD
            } else {
                MTL_LOAD_ACTION_CLEAR
            },
            store_action: reims_vgpu_protocol::pass_action::MTL_STORE_ACTION_STORE,
            clear_color: [0.125, 0.25, 0.5, 1.0],
            ..Default::default()
        })
        .collect();
    DrawEncodeRequest {
        colors,
        ..request()
    }
}

#[repr(align(16384))]
struct AlignedInput([u8; 16384]);

#[derive(Clone, Copy)]
enum InputMode {
    FragmentOnly,
    Direct,
    NativeFilled { fail_fragment: bool },
    PrimitiveIndirect,
    Indexed { wide: bool, indirect: bool },
}

fn record(
    pass: &mut MetalRenderPass,
    req: &mut DrawEncodeRequest,
    rgba: [f32; 4],
    scissor: ReimsVgpuScissor,
    outputs: &mut [Vec<u8>],
    query: Option<&mut VisibilityQuery>,
) -> EncodeStatus {
    record_inputs(
        pass,
        req,
        rgba,
        scissor,
        outputs,
        query,
        InputMode::FragmentOnly,
    )
}

fn record_inputs(
    pass: &mut MetalRenderPass,
    req: &mut DrawEncodeRequest,
    rgba: [f32; 4],
    scissor: ReimsVgpuScissor,
    outputs: &mut [Vec<u8>],
    query: Option<&mut VisibilityQuery>,
    mode: InputMode,
) -> EncodeStatus {
    use crate::runtime::draw::metal::inputs::{prepare, PreparedInput};
    use crate::runtime::host::HostMemory;
    let native = matches!(mode, InputMode::NativeFilled { .. });
    let mut guest_inputs = if let InputMode::NativeFilled { fail_fragment } = mode {
        use crate::runtime::draw::buffer_read_tests::Fixture;
        let mut fixture = Fixture::new(crate::model::PAGE_SHIFT_ARM64E);
        let mut vertex = fixture.bind(7, 1, 52, 32);
        vertex.index = 1;
        let fragment = fixture.bind(
            8,
            2,
            if fail_fragment { fixture.page + 16 } else { 80 },
            if fail_fragment { fixture.page - 16 } else { 32 },
        );
        let words: Vec<_> = [0x12345678u32, 0, 0, 0, 0xabcdef01]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        fixture
            .host
            .write_gpa(8 * fixture.page + 32, &words)
            .unwrap();
        let color: Vec<_> = rgba.into_iter().flat_map(f32::to_le_bytes).collect();
        fixture
            .host
            .write_gpa(13 * fixture.page + 64, &color)
            .unwrap();
        if fail_fragment {
            fixture
                .host
                .write_gpa(3 * fixture.page + 3 * 4, &0u32.to_le_bytes())
                .unwrap();
        }
        Some((fixture, vertex, fragment))
    } else {
        None
    };
    let mut input = Box::new(AlignedInput([0; 16384]));
    for (chunk, value) in input.0[..16].chunks_exact_mut(4).zip(rgba) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    let buffer = ReimsVgpuBuffer {
        binding: 0,
        data: input.0.as_mut_ptr(),
        len: input.0.len(),
        attribute_stride: 0,
        has_attribute_stride: 0,
        reserved0: 0,
        backing_data: std::ptr::null_mut(),
        backing_len: 0,
        backing_offset: 0,
    };
    let mut positions = [[-1f32, -1.], [3., -1.], [-1., 3.]];
    let mut shift = [0f32; 2];
    let attrs = [ReimsVgpuVertexAttr {
        location: 0,
        format: MTLVertexFormat::Float2 as u32,
        offset: 0,
        buffer_index: 2,
        stride: 8,
        data: positions.as_ptr().cast(),
        len: size_of_val(&positions),
        step_function: MTLVertexStepFunction::PerVertex as u32,
        step_rate: 1,
    }];
    let vertex_buffer = ReimsVgpuBuffer {
        binding: 1,
        data: shift.as_mut_ptr().cast(),
        len: size_of_val(&shift),
        ..buffer
    };
    let mut primitive_args = [3u32, 1, 0, 0];
    let primitive = ReimsVgpuPrimitiveIndirectDraw {
        arguments: primitive_args.as_ptr().cast(),
        arguments_len: size_of_val(&primitive_args),
    };
    let mut indices16 = [0u16, 1, 2];
    let mut indices32 = [0u32, 1, 2];
    let mut indexed_args = [3u32, 1, 0, 0, 0];
    let indirect = ReimsVgpuIndexedIndirectDraw {
        arguments: indexed_args.as_ptr().cast(),
        arguments_len: size_of_val(&indexed_args),
    };
    let indexed = match mode {
        InputMode::Indexed {
            wide,
            indirect: uses_indirect,
        } => Some(ReimsVgpuIndexedDraw {
            index_type: if wide {
                MTLIndexType::UInt32
            } else {
                MTLIndexType::UInt16
            } as u32,
            index_count: 3,
            base_vertex: 0,
            indices: if wide {
                indices32.as_ptr().cast()
            } else {
                indices16.as_ptr().cast()
            },
            indices_len: if wide {
                size_of_val(&indices32)
            } else {
                size_of_val(&indices16)
            },
            indirect: if uses_indirect {
                &indirect
            } else {
                std::ptr::null()
            },
        }),
        _ => None,
    };
    let stage_in = !matches!(
        mode,
        InputMode::FragmentOnly | InputMode::NativeFilled { .. }
    );
    let result = pass.with_draw(req, |owner, req| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
        let mut native_vertex = Vec::new();
        let mut native_fragment = Vec::new();
        if let Some((fixture, vertex, fragment)) = guest_inputs.as_mut() {
            for (bind, class, destination, miss) in [
                (
                    vertex,
                    InputClass::Vertex,
                    &mut native_vertex,
                    "draw_mtl_vertex_buffer_miss",
                ),
                (
                    fragment,
                    InputClass::Fragment,
                    &mut native_fragment,
                    "draw_mtl_fragment_buffer_miss",
                ),
            ] {
                match prepare(
                    &mut fixture.state,
                    &mut fixture.host,
                    1,
                    bind,
                    class,
                    true,
                    miss,
                ) {
                    Ok(PreparedInput::Native(buffer)) => destination.push(buffer),
                    Ok(PreparedInput::Cpu(_)) => {
                        panic!("plain input must use an owned native fill")
                    }
                    Err(status) => return (status, None),
                }
            }
        }
        for color in &req.colors {
            let target = owner.target_mut(color).unwrap();
            if target.texture.is_none() {
                target.texture = Some(
                    new_color_target(
                        device,
                        MTLPixelFormat::RGBA8Unorm,
                        8,
                        4,
                        MTLStorageMode::Shared,
                    )
                    .unwrap(),
                );
            }
        }
        let seed = [17u8, 31, 63, 255].repeat(32);
        let blend = r32float::source_over();
        let raster = req.cull_mode.map(|cull_mode| ReimsVgpuRasterState {
            has_cull_mode: 1,
            cull_mode,
            has_front_facing_winding: 0,
            front_facing_winding: 0,
            has_fill_mode: 0,
            fill_mode: 0,
            has_depth_clip_mode: 0,
            depth_clip_mode: 0,
        });
        let mut colors: Vec<_> = req
            .colors
            .iter()
            .zip(outputs.iter_mut())
            .map(|(color, output)| ColorRt {
                slot: color.slot,
                pixel_format: 0,
                seed_rgba8: (!req.continues_render_pass && color.slot == 0)
                    .then_some(seed.as_slice()),
                out_rgba8: (!req.render_pass_continues).then_some(output.as_mut_slice()),
                clear_r: color.clear_color[0],
                clear_g: color.clear_color[1],
                clear_b: color.clear_color[2],
                clear_a: color.clear_color[3],
                load_action: u32::from(color.load_action),
                blend: Some(blend),
                write_mask: 0xf,
                target: ColorTarget::PassLocal(owner.target(color).unwrap()),
            })
            .collect();
        let status = render_core_mrt_inputs(
            if native {
                NATIVE_VERT
            } else if stage_in {
                INPUT_VERT
            } else {
                VERT
            },
            if native { NATIVE_FRAG } else { FRAG },
            8,
            4,
            crate::protocol::draw::DrawArgs {
                vertex_count: 3,
                instance_count: 1,
                primitive_type: 3,
                first_vertex: if native { 3 } else { 0 },
                base_instance: if native { 9 } else { 0 },
            },
            matches!(mode, InputMode::PrimitiveIndirect).then_some(&primitive),
            indexed.as_ref(),
            if stage_in { &attrs } else { &[] },
            if native {
                RenderBuffers::Native(native_vertex)
            } else if stage_in {
                RenderBuffers::Host(std::slice::from_ref(&vertex_buffer))
            } else {
                RenderBuffers::Host(&[])
            },
            if native {
                RenderBuffers::Native(native_fragment)
            } else {
                RenderBuffers::Host(std::slice::from_ref(&buffer))
            },
            &[],
            &[],
            &[],
            &[],
            &[],
            &[scissor],
            raster.as_ref(),
            None,
            None,
            None,
            None,
            None,
            None,
            &mut colors,
            query,
            (std::ptr::null_mut(), 0),
            &mut owner.batch.borrow_mut(),
            req.render_pass_continues,
        );
        (
            if status.is_ok() {
                EncodeStatus::Ok
            } else {
                EncodeStatus::RailRefused(status)
            },
            None,
        )
    });
    // The page-aligned input is deliberately overwritten and freed before the
    // last record submits. A no-copy MTLBuffer would now read the wrong colour.
    input.0.fill(0);
    positions.fill([0.; 2]);
    shift.fill(123.);
    primitive_args.fill(0);
    indices16.fill(0);
    indices32.fill(0);
    indexed_args.fill(0);
    if let Some((fixture, _, _)) = guest_inputs.as_mut() {
        for pfn in [8, 13] {
            fixture
                .host
                .write_gpa(pfn * fixture.page, &vec![0; fixture.page as usize])
                .unwrap();
        }
    }
    result.0
}

fn rect(index: usize) -> ReimsVgpuScissor {
    let (x, y, width, height) = [(0, 0, 4, 4), (2, 0, 6, 4), (0, 1, 5, 2)][index];
    ReimsVgpuScissor {
        x,
        y,
        width,
        height,
    }
}

#[test]
fn ordinary_direct_filled_suffixes_keep_native_sizes_full_tails_and_deferred_snapshots() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        native_shaders(device);
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before_vertex = input::snapshot(InputClass::Vertex);
        let before_fragment = input::snapshot(InputClass::Fragment);
        let colors = [
            [0.125, 0.0, 0.0, 0.25],
            [0.0, 0.25, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.75],
        ];
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut actual = vec![vec![0xcc; 128]; 2];
        for index in 0..3 {
            req.continues_render_pass = index != 0;
            req.render_pass_continues = index != 2;
            if index != 0 {
                for color in &mut req.colors {
                    color.load_action = MTL_LOAD_ACTION_LOAD;
                }
            }
            let status = record_inputs(
                &mut pass,
                &mut req,
                colors[index],
                rect(index),
                &mut actual,
                None,
                InputMode::NativeFilled {
                    fail_fragment: false,
                },
            );
            assert!(matches!(status, EncodeStatus::Ok), "{status:?}");
            if index != 2 {
                assert_eq!(pass.batch.borrow().submissions, 0);
                assert!(
                    pool.borrow().inventory().is_empty(),
                    "unfinished input leases cannot recycle"
                );
                assert!(actual.iter().flatten().all(|byte| *byte == 0xcc));
            }
        }
        assert_eq!(pass.batch.borrow().submissions, 1);
        assert_eq!(pool.borrow().inventory(), [(20, 3), (48, 3)]);
        for (class, before, bytes) in [
            (InputClass::Vertex, before_vertex, 20),
            (InputClass::Fragment, before_fragment, 48),
        ] {
            let after = input::snapshot(class);
            assert_eq!(after.direct_fills - before.direct_fills, 3);
            assert_eq!(
                after.direct_fill_bytes - before.direct_fill_bytes,
                3 * bytes
            );
            assert_eq!(
                after.copies, before.copies,
                "no intermediate host-to-Metal input copy"
            );
            assert_eq!(after.copied_bytes, before.copied_bytes);
            assert_eq!(after.live_buffers, before.live_buffers);
        }
        let mut reference = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut expected = vec![vec![0xcc; 128]; 2];
        for index in 0..3 {
            req.continues_render_pass = index != 0;
            req.render_pass_continues = index != 2;
            if index != 0 {
                for color in &mut req.colors {
                    color.load_action = MTL_LOAD_ACTION_LOAD;
                }
            }
            assert!(matches!(
                record(
                    &mut reference,
                    &mut req,
                    colors[index],
                    rect(index),
                    &mut expected,
                    None,
                ),
                EncodeStatus::Ok
            ));
        }
        assert_eq!(actual, expected,
                "exact suffix lengths, zero native offsets, full array tails, first vertex/base instance \
                 and guest bytes overwritten before completion must preserve the image");
    });
}

#[test]
fn ordinary_direct_fill_partial_guest_read_refuses_draw_and_completes_prior_work() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        native_shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before_vertex = input::snapshot(InputClass::Vertex);
        let before_fragment = input::snapshot(InputClass::Fragment);
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0xcc; 128]; 2];
        assert!(matches!(
            record_inputs(
                &mut pass,
                &mut req,
                [1., 0., 0., 1.],
                rect(0),
                &mut outputs,
                None,
                InputMode::NativeFilled {
                    fail_fragment: false
                },
            ),
            EncodeStatus::Ok
        ));
        let target = pass.target(&req.colors[0]).unwrap().texture().to_owned();
        assert_eq!(pass.batch.borrow().submissions, 0);
        req.continues_render_pass = true;
        for color in &mut req.colors {
            color.load_action = MTL_LOAD_ACTION_LOAD;
        }
        let failure = record_inputs(
            &mut pass,
            &mut req,
            [0., 1., 0., 1.],
            rect(1),
            &mut outputs,
            None,
            InputMode::NativeFilled {
                fail_fragment: true,
            },
        );
        assert!(
            matches!(
                failure,
                EncodeStatus::MetalFailed("draw_mtl_fragment_buffer_miss")
            ),
            "{failure:?}"
        );
        assert_eq!(pass.phase, Phase::Failed);
        assert_eq!(pass.batch.borrow().submissions, 1);
        assert!(!pass.batch.borrow().pending());
        assert_eq!(
            pool.borrow().inventory(),
            [(20, 1), (48, 1)],
            "no partial or unsubmitted input recycled"
        );
        let vertex = input::snapshot(InputClass::Vertex);
        let fragment = input::snapshot(InputClass::Fragment);
        assert_eq!(vertex.live_buffers, before_vertex.live_buffers);
        assert_eq!(fragment.live_buffers, before_fragment.live_buffers);
        assert_eq!(
            fragment.direct_fill_failures - before_fragment.direct_fill_failures,
            1
        );
        let mut completed = [0u8; 128];
        target.get_bytes(
            completed.as_mut_ptr().cast(),
            32,
            MTLRegion::new_2d(0, 0, 8, 4),
            0,
        );
        assert_eq!(
            &completed[..4],
            &[255, 0, 0, 255],
            "the first draw actually completed"
        );
        assert_eq!(
            &completed[7 * 4..8 * 4],
            &[17, 31, 63, 255],
            "the refused draw encoded no geometry"
        );
    });
}

#[test]
fn ordinary_mrt_load_clear_store_batches_without_intermediate_readbacks() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        let library = shaders(device);
        let colors = [
            [0.125, 0.0, 0.0, 0.25],
            [0.0, 0.25, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.75],
        ];
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0xcc; 128]; 2];
        for index in 0..3 {
            req.continues_render_pass = index != 0;
            req.render_pass_continues = index != 2;
            if index != 0 {
                for color in &mut req.colors {
                    color.load_action = MTL_LOAD_ACTION_LOAD;
                }
            }
            assert!(matches!(
                record(
                    &mut pass,
                    &mut req,
                    colors[index],
                    rect(index),
                    &mut outputs,
                    None
                ),
                EncodeStatus::Ok
            ));
            if index != 2 {
                assert!(req.chain_resident_established);
                assert_eq!(pass.batch.borrow().submissions, 0);
                assert_eq!(pass.batch.borrow().readbacks, 0);
                assert!(outputs.iter().flatten().all(|byte| *byte == 0xcc));
            }
        }
        assert_eq!(pass.batch.borrow().submissions, 1);
        assert_eq!(pass.batch.borrow().readbacks, 2);
        assert_eq!(pass.phase, Phase::Finished);

        let targets: Vec<_> = (0..2)
            .map(|_| {
                new_color_target(
                    device,
                    MTLPixelFormat::RGBA8Unorm,
                    8,
                    4,
                    MTLStorageMode::Shared,
                )
                .unwrap()
            })
            .collect();
        let seed = [17u8, 31, 63, 255].repeat(32);
        targets[0].replace_region(MTLRegion::new_2d(0, 0, 8, 4), 0, seed.as_ptr().cast(), 32);
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&library.get_function("batch_vertex", None).unwrap()));
        descriptor
            .set_fragment_function(Some(&library.get_function("batch_fragment", None).unwrap()));
        let native = RenderPassDescriptor::new();
        for (index, target) in targets.iter().enumerate() {
            let pipeline_color = descriptor
                .color_attachments()
                .object_at(index as u64)
                .unwrap();
            pipeline_color.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
            pipeline_color.set_blending_enabled(true);
            pipeline_color.set_source_rgb_blend_factor(MTLBlendFactor::One);
            pipeline_color.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
            pipeline_color.set_source_alpha_blend_factor(MTLBlendFactor::One);
            pipeline_color.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
            let color = native.color_attachments().object_at(index as u64).unwrap();
            color.set_texture(Some(target));
            color.set_load_action(if index == 0 {
                MTLLoadAction::Load
            } else {
                MTLLoadAction::Clear
            });
            color.set_clear_color(MTLClearColor::new(0.125, 0.25, 0.5, 1.0));
            color.set_store_action(MTLStoreAction::Store);
        }
        let pipeline = device.new_render_pipeline_state(&descriptor).unwrap();
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_render_command_encoder(native);
        encoder.set_render_pipeline_state(&pipeline);
        for (index, color) in colors.iter().enumerate() {
            encoder.set_fragment_bytes(0, 16, color.as_ptr().cast());
            let r = rect(index);
            encoder.set_scissor_rect(MTLScissorRect {
                x: r.x as u64,
                y: r.y as u64,
                width: r.width as u64,
                height: r.height as u64,
            });
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        for (index, target) in targets.iter().enumerate() {
            let mut expected = vec![0; 128];
            target.get_bytes(
                expected.as_mut_ptr().cast(),
                32,
                MTLRegion::new_2d(0, 0, 8, 4),
                0,
            );
            assert_eq!(outputs[index], expected, "MRT/scissor/blend slot {index}");
        }
    });
}

#[test]
fn ordinary_query_boundary_completes_before_answering_and_resumes_batching() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = input::snapshot(InputClass::Fragment);
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0; 128]; 2];
        let mut query = VisibilityQuery {
            mode: MTLVisibilityResultMode::Counting as u32,
            samples: None,
        };
        for index in 0..3 {
            req.continues_render_pass = index != 0;
            req.render_pass_continues = index != 2;
            if index != 0 {
                for color in &mut req.colors {
                    color.load_action = MTL_LOAD_ACTION_LOAD;
                }
            }
            assert!(matches!(
                record(
                    &mut pass,
                    &mut req,
                    [0.125; 4],
                    rect(index),
                    &mut outputs,
                    (index == 1).then_some(&mut query)
                ),
                EncodeStatus::Ok
            ));
            if index == 1 {
                assert_eq!(query.samples, Some(24));
                assert!(!pass.batch.borrow().pending());
                assert_eq!(pass.batch.borrow().readbacks, 0);
            }
        }
        assert_eq!(pass.batch.borrow().submissions, 3);
        assert_eq!(pass.batch.borrow().readbacks, 2);
        let after = input::snapshot(InputClass::Fragment);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.reuses - before.reuses, 2);
        assert_eq!(after.copied_bytes - before.copied_bytes, 3 * 16384);
        assert_eq!(pool.borrow().inventory(), [(16384, 1)]);
    });
}

#[test]
fn ordinary_refusal_and_drop_complete_queued_resources() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        for refuse in [false, true] {
            let mut pass = MetalRenderPass::default();
            let mut req = ordinary_request();
            let mut outputs = vec![vec![0; 128]; 2];
            assert!(matches!(
                record(&mut pass, &mut req, [0.125; 4], rect(0), &mut outputs, None),
                EncodeStatus::Ok
            ));
            let texture = pass.target(&req.colors[0]).unwrap().texture().clone();
            if refuse {
                req.continues_render_pass = true;
                for color in &mut req.colors {
                    color.load_action = MTL_LOAD_ACTION_LOAD;
                }
                let result = pass.with_draw(&mut req, |_, _| {
                    (EncodeStatus::BadArgs("test_refusal"), None)
                });
                assert!(matches!(result.0, EncodeStatus::BadArgs("test_refusal")));
                assert_eq!(pass.phase, Phase::Failed);
                assert_eq!(pass.batch.borrow().submissions, 1);
            }
            drop(pass);
            assert_eq!(pool.borrow().inventory(), [(16384, 1)]);
            let mut pixels = [0u8; 128];
            texture.get_bytes(
                pixels.as_mut_ptr().cast(),
                32,
                MTLRegion::new_2d(0, 0, 8, 4),
                0,
            );
            assert_ne!(
                &pixels[..4],
                &[17, 31, 63, 255],
                "drop must finish the queued draw"
            );
        }
    });
}

#[test]
fn ordinary_deferred_raster_refusal_is_not_success_and_completes_prior_draw() {
    use crate::observe::Refusal;
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = input::snapshot(InputClass::Fragment);
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0xcc; 128]; 2];
        assert!(matches!(
            record(&mut pass, &mut req, [0.125; 4], rect(0), &mut outputs, None),
            EncodeStatus::Ok
        ));
        assert_eq!(pass.batch.borrow().submissions, 0);
        let texture = pass.target(&req.colors[0]).unwrap().texture().clone();
        req.continues_render_pass = true;
        req.cull_mode = Some(u32::MAX);
        for color in &mut req.colors {
            color.load_action = MTL_LOAD_ACTION_LOAD;
        }
        match record(&mut pass, &mut req, [0.75; 4], rect(1), &mut outputs, None) {
            EncodeStatus::RailRefused(status) => {
                assert_eq!(status.refusal(), Some("metal_render_cull_mode_unsupported"));
            }
            status => panic!("invalid raster state must refuse, got {status:?}"),
        }
        assert_eq!(pass.phase, Phase::Failed);
        assert_eq!(pass.batch.borrow().submissions, 1);
        assert_eq!(pass.batch.borrow().readbacks, 0);
        assert!(!pass.batch.borrow().pending());
        assert_eq!(pool.borrow().inventory(), [(16384, 1)]);
        let after = input::snapshot(InputClass::Fragment);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(
            after.copies - before.copies,
            1,
            "refused raster state encodes no new input"
        );
        assert!(outputs.iter().flatten().all(|byte| *byte == 0xcc));
        let mut pixels = [0u8; 128];
        texture.get_bytes(
            pixels.as_mut_ptr().cast(),
            32,
            MTLRegion::new_2d(0, 0, 8, 4),
            0,
        );
        assert_ne!(
            &pixels[..4],
            &[17, 31, 63, 255],
            "the queued first draw completed"
        );
        assert_eq!(
            &pixels[7 * 4..8 * 4],
            &[17, 31, 63, 255],
            "the refused second draw never ran"
        );
    });
}

#[test]
fn ordinary_owner_rejects_changed_backing_aliases_and_nonload_continuations() {
    let mut req = ordinary_request();
    let mut owner = MetalRenderPass::default();
    owner.prepare(&req).unwrap();
    owner.completed(&mut req, &mut None);
    req.continues_render_pass = true;
    for color in &mut req.colors {
        color.load_action = MTL_LOAD_ACTION_LOAD;
    }
    req.colors[1].target_gva += 0x4000;
    assert!(owner.prepare(&req).is_err());
    req.colors[1].target_gva -= 0x4000;
    req.colors[1].load_action = MTL_LOAD_ACTION_CLEAR;
    assert_eq!(owner.prepare(&req), Err("draw_mtl_render_pass_load"));
    req.colors[1].load_action = MTL_LOAD_ACTION_LOAD;
    req.colors[1].target_gva = req.colors[0].target_gva;
    assert_eq!(
        owner.prepare(&req),
        Err("draw_mtl_render_pass_attachment_alias")
    );
    req = ordinary_request();
    req.colors[1].slot = req.colors[0].slot;
    assert_eq!(
        MetalRenderPass::default().prepare(&req),
        Err("draw_mtl_render_pass_duplicate_slot")
    );
}

#[test]
fn ordinary_writable_texture_boundary_observes_prior_draw_before_publishing() {
    const WRITE_SOURCE: &str = r#"
                    #include <metal_stdlib>
                    using namespace metal;
                    fragment void batch_snapshot(float4 p [[position]], half4 prior [[color(0)]],
                        texture2d<half, access::write> output [[texture(3), raster_order_group(0)]]) {
                        output.write(prior, uint2(p.xy));
                    }
                "#;
    const WRITE_FRAG: &[u8] = b"reims-owned-test-batch-storage-fragment";
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        if !device.supports_family(MTLGPUFamily::Apple2) {
            return;
        }
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = input::snapshot(InputClass::Fragment);
        let library =
            crate::backend::metal::raw_metal::new_library_with_source(device, WRITE_SOURCE)
                .unwrap();
        crate::backend::metal::cache::fn_cache_insert(
            &BlobKey::new(WRITE_FRAG),
            library.get_function("batch_snapshot", None).unwrap(),
        );
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0; 128]; 2];
        assert!(matches!(
            record(&mut pass, &mut req, [0.125; 4], rect(0), &mut outputs, None),
            EncodeStatus::Ok
        ));
        assert_eq!(pass.batch.borrow().submissions, 0);
        req.continues_render_pass = true;
        for color in &mut req.colors {
            color.load_action = MTL_LOAD_ACTION_LOAD;
        }
        let destination = crate::backend::metal::compute::upload_storage_texture(
            device,
            crate::protocol::pixel_format::StorageImageSelector::Rgba8Unorm,
            8,
            4,
            &[0; 128],
            (std::ptr::null_mut(), 0),
        )
        .unwrap();
        let result = pass.with_draw(&mut req, |owner, req| {
            let mut colors: Vec<_> = req
                .colors
                .iter()
                .map(|color| ColorRt {
                    slot: color.slot,
                    pixel_format: 0,
                    seed_rgba8: None,
                    out_rgba8: None,
                    clear_r: 0.0,
                    clear_g: 0.0,
                    clear_b: 0.0,
                    clear_a: 0.0,
                    load_action: u32::from(MTL_LOAD_ACTION_LOAD),
                    blend: None,
                    write_mask: 0,
                    target: ColorTarget::PassLocal(owner.target(color).unwrap()),
                })
                .collect();
            let status = render_core_mrt(
                VERT,
                WRITE_FRAG,
                8,
                4,
                crate::protocol::draw::DrawArgs {
                    vertex_count: 3,
                    instance_count: 1,
                    primitive_type: 3,
                    first_vertex: 0,
                    base_instance: 0,
                },
                None,
                None,
                &[],
                &[],
                &[],
                &[],
                &[],
                &[ReimsVgpuSampledImage::Native {
                    binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
                    texture: destination.clone(),
                }],
                &[],
                &[],
                &[],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                &mut colors,
                None,
                (std::ptr::null_mut(), 0),
                &mut owner.batch.borrow_mut(),
                true,
            );
            (
                if status.is_ok() {
                    EncodeStatus::Ok
                } else {
                    EncodeStatus::RailRefused(status)
                },
                None,
            )
        });
        assert!(matches!(result.0, EncodeStatus::Ok), "{:?}", result.0);
        assert_eq!(pass.batch.borrow().submissions, 2);
        assert!(!pass.batch.borrow().pending());
        assert_eq!(
            pool.borrow().inventory(),
            [(16384, 1)],
            "an input-free writable submission does not erase completed input demand"
        );
        let after = input::snapshot(InputClass::Fragment);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.copied_bytes - before.copied_bytes, 16384);
        assert_eq!(after.live_bytes, before.live_bytes);
        let mut snapshot = [0u8; 128];
        let mut color = [0u8; 128];
        destination.get_bytes(
            snapshot.as_mut_ptr().cast(),
            32,
            MTLRegion::new_2d(0, 0, 8, 4),
            0,
        );
        pass.target(&req.colors[0]).unwrap().texture().get_bytes(
            color.as_mut_ptr().cast(),
            32,
            MTLRegion::new_2d(0, 0, 8, 4),
            0,
        );
        assert_eq!(
            snapshot, color,
            "writable result must include the preceding deferred draw"
        );
        assert_ne!(&snapshot[..4], &snapshot[7 * 4..8 * 4]);
    });
}

#[test]
fn ordinary_completed_inputs_recycle_across_compatible_passes() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        let mut reference = None;
        for mode in [
            InputMode::FragmentOnly,
            InputMode::Direct,
            InputMode::PrimitiveIndirect,
            InputMode::Indexed {
                wide: false,
                indirect: false,
            },
            InputMode::Indexed {
                wide: true,
                indirect: false,
            },
            InputMode::Indexed {
                wide: false,
                indirect: true,
            },
            InputMode::Indexed {
                wide: true,
                indirect: true,
            },
        ] {
            pool.borrow_mut().clear_available();
            let mut classes = vec![(InputClass::Fragment, 16384u64)];
            if !matches!(mode, InputMode::FragmentOnly) {
                classes.extend([(InputClass::Attribute, 24), (InputClass::Vertex, 8)]);
            }

            if let InputMode::Indexed { wide, .. } = mode {
                classes.push((InputClass::Index, if wide { 12 } else { 6 }));
            }
            match mode {
                InputMode::PrimitiveIndirect => classes.push((InputClass::Indirect, 16)),
                InputMode::Indexed { indirect: true, .. } => {
                    classes.push((InputClass::Indirect, 20))
                }
                _ => {}
            }
            let before: Vec<_> = classes
                .iter()
                .map(|&(class, _)| input::snapshot(class))
                .collect();
            for repetition in 0..2 {
                let mut pass = MetalRenderPass::default();
                let mut req = ordinary_request();
                let mut outputs = vec![vec![0xcc; 128]; 2];
                for index in 0..3 {
                    req.continues_render_pass = index != 0;
                    req.render_pass_continues = index != 2;
                    if index != 0 {
                        for color in &mut req.colors {
                            color.load_action = MTL_LOAD_ACTION_LOAD;
                        }
                    }
                    let rgba = [
                        [0.125, 0.0, 0.0, 0.25],
                        [0.0, 0.25, 0.0, 0.5],
                        [0.0, 0.0, 0.5, 0.75],
                    ][index];
                    let result = record_inputs(
                        &mut pass,
                        &mut req,
                        rgba,
                        rect(index),
                        &mut outputs,
                        None,
                        mode,
                    );
                    assert!(matches!(result, EncodeStatus::Ok), "{result:?}");
                    for (&(class, len), baseline) in classes.iter().zip(&before) {
                        let now = input::snapshot(class);
                        assert_eq!(
                            now.allocations - baseline.allocations,
                            if repetition == 0 { index as u64 + 1 } else { 3 },
                            "{class:?}"
                        );
                        assert_eq!(
                            now.copied_bytes - baseline.copied_bytes,
                            (repetition * 3 + index as u64 + 1) * len
                        );
                        if index != 2 {
                            assert_eq!(
                                now.live_bytes - baseline.live_bytes,
                                (index as u64 + 1) * len
                            );
                        }
                    }
                    if index != 2 {
                        assert_eq!(pass.batch.borrow().submissions, 0);
                        assert_eq!(pass.batch.borrow().readbacks, 0);
                        assert!(outputs.iter().flatten().all(|byte| *byte == 0xcc));
                    }
                }
                assert_eq!(pass.batch.borrow().submissions, 1);
                assert_eq!(pass.batch.borrow().readbacks, 2);
                if let Some(expected) = &reference {
                    assert_eq!(
                        &outputs, expected,
                        "every input route must preserve the native pixels"
                    );
                } else {
                    reference = Some(outputs);
                }
                let mut expected: Vec<_> =
                    classes.iter().map(|&(_, len)| (len as usize, 3)).collect();
                expected.sort_unstable();
                assert_eq!(pool.borrow().inventory(), expected);
            }
            for (&(class, len), baseline) in classes.iter().zip(&before) {
                let after = input::snapshot(class);
                assert_eq!(after.allocations - baseline.allocations, 3);
                assert_eq!(after.allocated_bytes - baseline.allocated_bytes, 3 * len);
                assert_eq!(after.reuses - baseline.reuses, 3);
                assert_eq!(after.copies - baseline.copies, 6);
                assert_eq!(after.live_bytes, baseline.live_bytes);
                assert_eq!(after.retained_bytes - baseline.retained_bytes, 3 * len);
            }
        }
    });
}

#[test]
fn ordinary_input_allocation_refusal_completes_prior_draw_without_recycling_early() {
    use crate::observe::Refusal;
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
        let pool = crate::backend::metal::runtime::thread_input_pool(device);
        pool.borrow_mut().clear_available();
        let before = input::snapshot(InputClass::Fragment);
        let mut pass = MetalRenderPass::default();
        let mut req = ordinary_request();
        let mut outputs = vec![vec![0xcc; 128]; 2];
        assert!(matches!(
            record(&mut pass, &mut req, [0.125; 4], rect(0), &mut outputs, None),
            EncodeStatus::Ok
        ));
        assert_eq!(pass.batch.borrow().submissions, 0);
        assert!(pool.borrow().inventory().is_empty());
        let texture = pass.target(&req.colors[0]).unwrap().texture().clone();
        pool.borrow_mut().fail_next_allocation();
        req.continues_render_pass = true;
        for color in &mut req.colors {
            color.load_action = MTL_LOAD_ACTION_LOAD;
        }
        match record(&mut pass, &mut req, [0.75; 4], rect(1), &mut outputs, None) {
            EncodeStatus::RailRefused(status) => {
                assert_eq!(status.refusal(), Some("metal_render_buffer_create_failed"));
            }
            status => panic!("exhausted allocation must refuse, got {status:?}"),
        }
        assert_eq!(pass.phase, Phase::Failed);
        assert_eq!(pass.batch.borrow().submissions, 1);
        assert_eq!(pass.batch.borrow().readbacks, 0);
        assert!(!pass.batch.borrow().pending());
        assert_eq!(pool.borrow().inventory(), [(16384, 1)]);
        let after = input::snapshot(InputClass::Fragment);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.copies - before.copies, 1);
        assert_eq!(after.reuses - before.reuses, 0);
        assert_eq!(after.live_bytes, before.live_bytes);
        let mut pixels = [0u8; 128];
        texture.get_bytes(
            pixels.as_mut_ptr().cast(),
            32,
            MTLRegion::new_2d(0, 0, 8, 4),
            0,
        );
        assert_ne!(
            &pixels[..4],
            &[17, 31, 63, 255],
            "the earlier draw completed"
        );
        assert_eq!(
            &pixels[7 * 4..8 * 4],
            &[17, 31, 63, 255],
            "the refused draw never ran"
        );
        assert!(outputs.iter().flatten().all(|byte| *byte == 0xcc));
    });
}
