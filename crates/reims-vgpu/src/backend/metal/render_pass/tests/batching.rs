use super::*;
use crate::backend::blob::BlobKey;
use crate::backend::metal::abi::*;
use crate::backend::metal::render::{render_core_mrt, VisibilityQuery};

const SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 batch_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    struct Colors { half4 a [[color(0)]]; half4 b [[color(1)]]; };
    fragment Colors batch_fragment(constant float4 &color [[buffer(0)]]) {
        return {half4(color), half4(color.zyxw)};
    }
"#;
const VERT: &[u8] = b"reims-owned-test-batch-vertex";
const FRAG: &[u8] = b"reims-owned-test-batch-fragment";

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

fn record(
    pass: &mut MetalRenderPass,
    req: &mut DrawEncodeRequest,
    rgba: [f32; 4],
    scissor: ReimsVgpuScissor,
    outputs: &mut [Vec<u8>],
    query: Option<&mut VisibilityQuery>,
) -> EncodeStatus {
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
    let result = pass.with_draw(req, |owner, req| {
        let device = crate::backend::metal::runtime::system_device().unwrap();
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
        let status = render_core_mrt(
            VERT,
            FRAG,
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
            &[buffer],
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
    });
}

#[test]
fn ordinary_refusal_and_drop_complete_queued_resources() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = crate::backend::metal::runtime::system_device() else {
            return;
        };
        shaders(device);
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
        let mut snapshot = [0; 128];
        let mut color = [0; 128];
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
