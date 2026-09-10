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
    let (pipeline, _, _, usage) = get_render_pipeline_state(
        device, &v, &f, None, &lookup, (ptr::null_mut(), 0),
    ).unwrap();
    let (_, _, _, cached) = get_render_pipeline_state(
        device, &v, &f, None, &lookup, (ptr::null_mut(), 0),
    ).unwrap();
    assert!(std::sync::Arc::ptr_eq(&usage, &cached), "reflection belongs to the cached PSO");
    (pipeline, usage)
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
