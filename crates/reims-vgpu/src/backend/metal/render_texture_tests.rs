use super::*;
use crate::backend::metal::raw_metal::{BindingInfo, BINDING_TYPE_TEXTURE};

fn raw(index: u64, access: u64, count: u64) -> BindingInfo {
    BindingInfo { used: true, type_: BINDING_TYPE_TEXTURE, access, index, array_length: count }
}

#[test]
fn render_texture_contract_expands_arrays_and_refuses_unknown_or_overflowing_reflection() {
    let usages = texture_usages(&[raw(3, 2, 2), raw(8, 0, 1), raw(9, 1, 1)], false).unwrap();
    assert_eq!(usages, vec![
        RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3, access: RenderTextureAccess::Write },
        RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 4, access: RenderTextureAccess::Write },
        RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 8, access: RenderTextureAccess::Read },
        RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 9, access: RenderTextureAccess::ReadWrite },
    ]);
    assert!(texture_usages(&[raw(3, 7, 1)], false).is_err());
    assert!(texture_usages(&[raw(REIMS_VGPU_METAL_MAX_TEXTURES as u64 - 1, 0, 2)], true).is_err());
    assert!(texture_usages(&[raw(u64::MAX, 0, 2)], true).is_err());
    assert!(texture_usages(&[raw(3, 0, 2), raw(4, 2, 1)], false).is_err());
}

#[test]
fn render_texture_contract_writes_never_accept_upload_only_or_unbound_images() {
    let usage = RenderTextureUsage {
        binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
        access: RenderTextureAccess::Write,
    };
    let packed = ReimsVgpuSampledImage::Packed(ReimsVgpuPackedSampledImage {
        binding: usage.binding, width: 4, height: 4, rgba8: ptr::null(), len: 0,
        pixel_format: 0, bytes_per_row: 0, data: ptr::null(), data_len: 0,
    });
    assert!(!validate_render_texture_bindings(&[], &[usage], false).is_ok());
    assert!(!validate_render_texture_bindings(&[packed.clone()], &[usage], false).is_ok());
    assert!(!validate_render_texture_bindings(&[packed.clone(), packed.clone()], &[usage], false).is_ok());
    assert!(validate_render_texture_bindings(
        &[packed], &[RenderTextureUsage { access: RenderTextureAccess::Read, ..usage }], false,
    ).is_ok());
}

const SNAPSHOT_SHADER: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    vertex float4 snapshot_vertex(uint i [[vertex_id]]) {
        const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
        return float4(p[i], 0, 1);
    }
    fragment void snapshot_fragment(float4 p [[position]], half4 prior [[color(0)]],
        texture2d<half, access::write> destination [[texture(3), raster_order_group(0)]]) {
        destination.write(prior, uint2(p.xy));
    }
    fragment half4 consume_fragment(float4 p [[position]],
        texture2d<half, access::read> snapshot [[texture(3)]]) {
        return snapshot.read(uint2(p.xy));
    }
"#;

fn pipeline_from_source(
    device: &Device,
    library: &LibraryRef,
    source: &'static str,
    vertex: &str,
    fragment: &str,
    write_mask: u32,
) -> (RenderPipelineState, std::sync::Arc<RenderTextureUsages>) {
    let key = fill_render_pso_key(&[], None, &[ColorRtKey {
        slot: 0, pixel_format: MTLPixelFormat::RGBA16Float as u32, blend: None, write_mask,
    }], 0, 0);
    // Native test sources contain multiple functions. Include entry names in
    // the synthetic blob identity, as product single-function MTLBs do.
    let vertex_id = format!("{source}\n{vertex}");
    let fragment_id = format!("{source}\n{fragment}");
    let lookup = RenderPsoLookup {
        desc: &key,
        vert: BlobKey::new(vertex_id.as_bytes()),
        frag: BlobKey::new(fragment_id.as_bytes()),
    };
    let v = library.get_function(vertex, None).unwrap();
    let f = library.get_function(fragment, None).unwrap();
    let pipeline = get_render_pipeline_state(
        device, &lookup, (ptr::null_mut(), 0), || Ok((v, f, None)),
    ).unwrap();
    let cached = get_render_pipeline_state(
        device, &lookup, (ptr::null_mut(), 0),
        || panic!("a PSO hit must not prepare functions or a vertex descriptor"),
    ).unwrap();
    assert!(std::sync::Arc::ptr_eq(&pipeline, &cached), "reflection belongs to the cached PSO");
    (pipeline.pso.clone(), pipeline.textures.clone())
}

#[test]
fn prepared_render_pipeline_retains_content_and_specialization_after_staging() {
    const VERTEX: &[u8] = b"owned-prepared-pipeline-vertex";
    const FRAGMENT: &[u8] = b"owned-prepared-pipeline-fragment";
    let Some(device) = system_device() else {
        return;
    };
    let colors = [ColorRtKey {
        slot: 0,
        pixel_format: MTLPixelFormat::RGBA16Float as u32,
        blend: None,
        write_mask: 0xf,
    }];
    let layout = |colors| RenderPipelineLayout {
        attrs: &[],
        blend: None,
        colors,
        depth_format: 0,
        stencil_format: 0,
    };
    let prepared = objc::rc::autoreleasepool(|| {
        let library =
            super::super::raw_metal::new_library_with_source(device, SNAPSHOT_SHADER).unwrap();
        for (bytes, name) in [(VERTEX, "snapshot_vertex"), (FRAGMENT, "consume_fragment")] {
            super::super::cache::fn_cache_insert(
                &BlobKey::new(bytes),
                library.get_function(name, None).unwrap(),
            );
        }
        // The prepared PSO must outlive these owned snapshots and this pool.
        let vertex = VERTEX.to_vec();
        let fragment = FRAGMENT.to_vec();
        prepare_render_pipeline(
            BlobKey::new(&vertex),
            BlobKey::new(&fragment),
            layout(&colors),
        )
        .unwrap()
    });
    let again = prepare_render_pipeline(
        BlobKey::new(VERTEX),
        BlobKey::new(FRAGMENT),
        layout(&colors),
    )
    .unwrap();
    assert!(Arc::ptr_eq(&prepared.entry, &again.entry));
    assert!(std::ptr::eq(
        prepared.texture_usages(),
        again.texture_usages()
    ));
    assert_eq!(
        prepared.texture_usages().fragment,
        vec![RenderTextureUsage {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
            access: RenderTextureAccess::Read,
        }]
    );

    let masked_colors = [ColorRtKey {
        write_mask: 0,
        ..colors[0]
    }];
    let masked = prepare_render_pipeline(
        BlobKey::new(VERTEX),
        BlobKey::new(FRAGMENT),
        layout(&masked_colors),
    )
    .unwrap();
    assert!(
        !Arc::ptr_eq(&prepared.entry, &masked.entry),
        "a prepared pipeline still specializes an unblended attachment's write mask"
    );

    for vertex in [true, false] {
        let mut collided = prepared.entry.id.as_lookup();
        if vertex {
            collided.vert.bytes = b"other-prepared-pipeline-vertex";
        } else {
            collided.frag.bytes = b"other-prepared-pipeline-fragment";
        }
        let mut rebuilt = false;
        let result = get_render_pipeline_state(device, &collided, (ptr::null_mut(), 0), || {
            rebuilt = true;
            Err(Status::args("metal_function_mtlb_empty"))
        });
        assert!(
            rebuilt && result.is_err(),
            "a collided shader digest must miss and attempt preparation"
        );
    }

    let mut target = ColorRt {
        slot: 0,
        pixel_format: colors[0].pixel_format,
        seed_rgba8: None,
        out_rgba8: None,
        clear_r: 0.0,
        clear_g: 0.0,
        clear_b: 0.0,
        clear_a: 1.0,
        load_action: REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
        blend: None,
        write_mask: 0xf,
        target: ColorTarget::Transient,
    };
    assert!(prepared
        .validate_attachments(std::slice::from_ref(&target), 0, 0)
        .is_ok());
    assert!(prepared.validate_attachments(&[], 0, 0).is_err());
    assert!(prepared
        .validate_attachments(
            std::slice::from_ref(&target),
            MTLPixelFormat::Depth32Float as u32,
            0,
        )
        .is_err());
    assert!(prepared
        .validate_attachments(
            std::slice::from_ref(&target),
            0,
            MTLPixelFormat::Stencil8 as u32,
        )
        .is_err());
    target.slot = 1;
    assert!(prepared
        .validate_attachments(std::slice::from_ref(&target), 0, 0)
        .is_err());
    target.slot = 0;
    target.pixel_format = MTLPixelFormat::RGBA8Unorm as u32;
    assert!(prepared
        .validate_attachments(std::slice::from_ref(&target), 0, 0)
        .is_err());
}

fn color_pass<'a>(target: &TextureRef, clear: MTLClearColor) -> &'a RenderPassDescriptorRef {
    let pass = RenderPassDescriptor::new();
    let color = pass.color_attachments().object_at(0).unwrap();
    color.set_texture(Some(target));
    color.set_load_action(MTLLoadAction::Clear);
    color.set_store_action(MTLStoreAction::Store);
    color.set_clear_color(clear);
    pass
}

#[test]
fn resident_sample_matches_readback_upload_and_owns_deferred_snapshot() {
    use super::super::resident;
    use objc::rc::{autoreleasepool, WeakPtr};
    const VERTEX: &[u8] = b"resident-sample-owned-vertex";
    const FRAGMENT: &[u8] = b"resident-sample-owned-fragment";
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 resident_vertex(uint i [[vertex_id]]) {
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        fragment half4 resident_fragment(float4 p [[position]],
            texture2d<half> image [[texture(3)]], sampler s [[sampler(2)]]) {
            return image.sample(s, (p.xy + float2(0.25, -0.25)) / 4.0);
        }
    "#;
    let device = system_device().expect("Metal device");
    let key = resident::ResidentColorKey::for_surface(0xffff_a113, 4, 4);
    let pixels: Vec<_> = (0..64).map(|i| (i * 3) as u8).collect();
    let (weak, image, readback) = autoreleasepool(|| {
        let library = super::super::raw_metal::new_library_with_source(device, SOURCE).unwrap();
        for (bytes, name) in [(VERTEX, "resident_vertex"), (FRAGMENT, "resident_fragment")] {
            super::super::cache::fn_cache_insert(
                &BlobKey::new(bytes),
                library.get_function(name, None).unwrap(),
            );
        }
        let texture = resident::create(device, &key, MTLPixelFormat::RGBA8Unorm, 4).unwrap();
        texture.replace_region(MTLRegion::new_2d(0, 0, 4, 4), 0, pixels.as_ptr().cast(), 16);
        let weak = unsafe { WeakPtr::new(texture.as_ptr().cast()) };
        resident::published(&key, 7);
        let readback = resident::read_published_rgba8(&key, 7).unwrap();
        let image = ReimsVgpuSampledImage::Resident {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3,
            image: resident::sample_published_rgba8(&key, 7).unwrap(),
        };
        (weak, image, readback)
    });
    assert_eq!(readback, pixels);
    assert!(
        !weak.load().is_null(),
        "the published sample retains its source"
    );
    for access in [RenderTextureAccess::Write, RenderTextureAccess::ReadWrite] {
        assert!(
            !validate_render_texture_bindings(
                std::slice::from_ref(&image),
                &[RenderTextureUsage {
                    binding: image.binding(),
                    access
                }],
                false,
            )
            .is_ok(),
            "a published frame cannot be rebound as writable storage"
        );
    }
    let packed = ReimsVgpuSampledImage::Packed(ReimsVgpuPackedSampledImage {
        binding: image.binding(),
        width: 4,
        height: 4,
        rgba8: readback.as_ptr(),
        len: readback.len(),
        pixel_format: 0,
        bytes_per_row: 16,
        data: readback.as_ptr(),
        data_len: readback.len(),
    });
    let encode = |image: &ReimsVgpuSampledImage, capture: bool| {
        autoreleasepool(|| {
            let mut output = vec![0u8; 128];
            let mut batch = RenderBatch::default();
            let mut color = ColorRt {
                slot: 0,
                pixel_format: MTLPixelFormat::RGBA16Float as u32,
                seed_rgba8: None,
                out_rgba8: capture.then_some(output.as_mut_slice()),
                clear_r: 0.0,
                clear_g: 0.0,
                clear_b: 0.0,
                clear_a: 1.0,
                load_action: REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
                blend: None,
                write_mask: 0xf,
                target: ColorTarget::Transient,
            };
            let status = render_core_mrt(
                VERTEX,
                FRAGMENT,
                4,
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
                std::slice::from_ref(image),
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
                std::slice::from_mut(&mut color),
                None,
                (ptr::null_mut(), 0),
                &mut batch,
                true,
            );
            assert!(status.is_ok(), "{status:?}");
            if capture {
                assert!(!batch.pending());
                assert_eq!(batch.submissions, 1);
                assert!(
                    batch.published_samples.is_empty(),
                    "completion releases native read leases"
                );
            } else {
                assert!(
                    batch.pending(),
                    "leased snapshots preserve existing batching"
                );
                assert_eq!(batch.submissions, 0);
                assert_eq!(batch.published_samples.len(), 1);
            }
            (output, batch)
        })
    };
    assert_eq!(
        encode(&image, true).0,
        encode(&packed, true).0,
        "retained RGBA8 sampling must equal readback followed by packed upload"
    );
    let (_, mut pending) = encode(&image, false);
    drop(image);
    assert!(
        resident::take(&key, 7).is_none(),
        "the batch's lease alone excludes writable reuse"
    );
    resident::forget(key.mapping_id);
    assert!(
        !weak.load().is_null(),
        "the batch owns the source after the caller drops it"
    );
    autoreleasepool(|| pending.finish((ptr::null_mut(), 0)).unwrap());
    assert!(
        weak.load().is_null(),
        "completed sampling retains no evicted source"
    );
}

fn texture_half_bits(texture: &TextureRef) -> Vec<u16> {
    let mut output = vec![0u16; 4 * 4 * 4];
    texture.get_bytes(output.as_mut_ptr().cast(), 4 * 8, MTLRegion::new_2d(0, 0, 4, 4), 0);
    output
}

#[test]
fn render_texture_fragment_framebuffer_snapshot_survives_completion_and_next_draw() {
    objc::rc::autoreleasepool(|| {
        let Some(device) = system_device() else { return; };
        if !device.supports_family(MTLGPUFamily::Apple2) { return; }
        let library = super::super::raw_metal::new_library_with_source(device, SNAPSHOT_SHADER).unwrap();
        let (copy_pipeline, usage) = pipeline_from_source(
            device, &library, SNAPSHOT_SHADER, "snapshot_vertex", "snapshot_fragment", 0,
        );
        assert!(usage.vertex.is_empty());
        assert_eq!(usage.fragment, vec![RenderTextureUsage {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 3, access: RenderTextureAccess::Write,
        }]);
        let seed_bits = [0x3c00u16, 0, 0x3800, 0x3c00].repeat(16);
        let seed: Vec<u8> = seed_bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let destination = super::super::compute::upload_storage_texture(
            device, crate::protocol::pixel_format::StorageImageSelector::Rgba16Float,
            4, 4, &seed, (ptr::null_mut(), 0),
        ).unwrap();
        let binding = ReimsVgpuSampledImage::Native {
            binding: usage.fragment[0].binding, texture: destination.clone(),
        };
        assert!(validate_render_texture_bindings(&[binding.clone()], &usage.fragment, false).is_ok());
        assert!(!validate_render_texture_bindings(
            &[binding.clone(), binding.clone()], &usage.fragment, false,
        ).is_ok());
        let queue = thread_queue(device);
        objc::rc::autoreleasepool(|| {
            let source = new_color_target(device, MTLPixelFormat::RGBA16Float, 4, 4,
                MTLStorageMode::Shared).unwrap();
            let pass = color_pass(&source, MTLClearColor::new(0.25, 0.5, 2.0, 0.75));
            let command = queue.new_command_buffer();
            let encoder = command.new_render_command_encoder(pass);
            encoder.set_render_pipeline_state(&copy_pipeline);
            encoder.set_scissor_rect(MTLScissorRect { x: 1, y: 1, width: 2, height: 2 });
            let mut retained = Vec::new();
            assert!(bind_sampled_images(device, encoder, &mut retained, &[binding.clone()],
                true, (ptr::null_mut(), 0)).is_ok());
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(retained.len(), 1, "supplied native texture stays retained through completion");
        });
        let mut expected = seed_bits;
        for y in 1..3 {
            for x in 1..3 {
                expected[(y * 4 + x) * 4..(y * 4 + x + 1) * 4]
                    .copy_from_slice(&[0x3400, 0x3800, 0x4000, 0x3a00]);
            }
        }
        assert_eq!(texture_half_bits(&destination), expected,
            "fragment texture writes must survive outside the draw's autorelease pool");

        let (consume_pipeline, read_usage) = pipeline_from_source(
            device, &library, SNAPSHOT_SHADER, "snapshot_vertex", "consume_fragment", 0xf,
        );
        assert_eq!(read_usage.fragment[0].access, RenderTextureAccess::Read);
        let output = new_color_target(device, MTLPixelFormat::RGBA16Float, 4, 4,
            MTLStorageMode::Shared).unwrap();
        let pass = color_pass(&output, MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
        let command = queue.new_command_buffer();
        let encoder = command.new_render_command_encoder(pass);
        encoder.set_render_pipeline_state(&consume_pipeline);
        let mut retained = Vec::new();
        assert!(bind_sampled_images(device, encoder, &mut retained, &[binding],
            true, (ptr::null_mut(), 0)).is_ok());
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(texture_half_bits(&output), expected, "the next draw must see the saved framebuffer");
    });
}

#[test]
fn render_texture_vertex_write_and_fragment_array_reflect_independently() {
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        vertex float4 write_vertex(uint i [[vertex_id]],
            texture2d<float, access::write> destination [[texture(2)]]) {
            destination.write(float4(0.25, 0.5, 2.0, 0.75), uint2(i, 0));
            const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
            return float4(p[i], 0, 1);
        }
        fragment half4 read_fragment(float4 p [[position]],
            array<texture2d<half, access::read>, 2> sources [[texture(5)]]) {
            return sources[uint(p.x) % 2].read(uint2(0, 0));
        }
    "#;
    objc::rc::autoreleasepool(|| {
        let Some(device) = system_device() else { return; };
        let library = super::super::raw_metal::new_library_with_source(device, SOURCE).unwrap();
        let (pipeline, usage) = pipeline_from_source(
            device, &library, SOURCE, "write_vertex", "read_fragment", 0xf,
        );
        assert_eq!(usage.vertex, vec![RenderTextureUsage {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 2, access: RenderTextureAccess::Write,
        }]);
        assert_eq!(usage.fragment, vec![
            RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 5, access: RenderTextureAccess::Read },
            RenderTextureUsage { binding: REIMS_VGPU_BINDING_TEXTURE_BASE + 6, access: RenderTextureAccess::Read },
        ]);
        let texture = super::super::compute::upload_storage_texture(
            device, crate::protocol::pixel_format::StorageImageSelector::Rgba16Float,
            4, 4, &[0; 128], (ptr::null_mut(), 0),
        ).unwrap();
        let image = |index| ReimsVgpuSampledImage::Native {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + index, texture: texture.clone(),
        };
        let output = new_color_target(device, MTLPixelFormat::RGBA16Float, 4, 4,
            MTLStorageMode::Shared).unwrap();
        let queue = thread_queue(device);
        let command = queue.new_command_buffer();
        let pass = color_pass(&output, MTLClearColor::new(0.0, 0.0, 0.0, 0.0));
        let encoder = command.new_render_command_encoder(pass);
        encoder.set_render_pipeline_state(&pipeline);
        let mut retained = Vec::new();
        assert!(bind_sampled_images(device, encoder, &mut retained, &[image(2)],
            false, (ptr::null_mut(), 0)).is_ok());
        // Use a separate read-only source; simultaneous cross-stage aliasing
        // is not the ordering contract this regression establishes.
        let read_source = super::super::compute::upload_storage_texture(
            device, crate::protocol::pixel_format::StorageImageSelector::Rgba16Float,
            4, 4, &[0; 128], (ptr::null_mut(), 0),
        ).unwrap();
        let reads: Vec<_> = [5, 6].into_iter().map(|index| ReimsVgpuSampledImage::Native {
            binding: REIMS_VGPU_BINDING_TEXTURE_BASE + index, texture: read_source.clone(),
        }).collect();
        assert!(bind_sampled_images(device, encoder, &mut retained, &reads,
            true, (ptr::null_mut(), 0)).is_ok());
        encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let bits = texture_half_bits(&texture);
        for pixel in bits[..12].chunks_exact(4) {
            assert_eq!(pixel, &[0x3400, 0x3800, 0x4000, 0x3a00]);
        }
        assert!(bits[12..].iter().all(|&b| b == 0));
    });
}
