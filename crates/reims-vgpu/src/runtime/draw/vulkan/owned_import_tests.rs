use super::*;
use crate::backend::vulkan::VulkanBackend;
use crate::runtime::guest_ram::{
    GuestPageFootprint, GuestRamImport, GuestRef, GuestRun,
    take_released_owned_host_allocations,
};
use crate::runtime::guest_ram_map::with_test_import_limits;
use crate::runtime::host::FakeHost;
use std::sync::Arc;

fn host_and_pages(page: u64) -> (FakeHost, Vec<u64>) {
    let mut host = FakeHost::new();
    host.owned_map_pages = true;
    let pages = vec![0x40000, 0x80000];
    host.map_range(pages[0], page as usize, 0x31);
    host.map_range(pages[1], page as usize, 0x72);
    (host, pages)
}

fn drain_owned(host: &mut FakeHost) -> usize {
    // This is a Vulkan ownership test even in a binary whose default is Metal.
    for (ptr, len) in VulkanBackend::new().take_released_host_aliases() {
        host.unmap_pages(ptr, len);
    }
    let releases = take_released_owned_host_allocations();
    let count = releases.len();
    for (ptr, len) in releases {
        host.unmap_pages(ptr, len);
    }
    count
}

#[test]
fn owned_nonstable_packed_inputs_retain_exact_views_through_native_retirement() {
    with_test_import_limits(Some(4096), || {
        for page in [4096, 16384] {
            crate::runtime::guest_ram_map::reset();
            let (mut host, pages) = if cfg!(target_os = "macos") && page == 16384 {
                host_and_pages(page)
            } else {
                // A 16 KiB host cannot alias separate 4 KiB guest pages;
                // owned fixtures must use a real alias, never a bounce copy.
                let mut host = FakeHost::new();
                host.owned_map_pages = true;
                let pages = vec![0x40000, 0x40000 + page];
                host.map_range(pages[0], (page * 2) as usize, 0x31);
                host.write_gpa(pages[1], &vec![0x72; page as usize]).unwrap();
                (host, pages)
            };
            let backing = BufferBacking { gva: 0x100000 + 64, size: page * 2 - 64 };
            let packed = owned_packed_buffer(&mut host, &backing, pages.clone(), page).unwrap();
            let native = packed.import.owned_host_allocation().unwrap();
            assert_eq!(native.footprint().pages(), pages);
            assert_eq!(native.footprint().page_size(), page);
            let first = slice_packed_buffer(&packed, 0, 32).unwrap();
            let second = slice_packed_buffer(&packed, page - 64, 32).unwrap();
            assert_eq!(bound_buffer_content(&first).cpu_bytes().as_ref(), &[0x31; 32]);
            assert_eq!(bound_buffer_content(&second).cpu_bytes().as_ref(), &[0x72; 32]);
            let observed = Arc::clone(&packed.import);
            let mut held = crate::runtime::bound_buffers::BoundBuffers::default();
            held.insert_packed(
                1, 7, crate::runtime::bound_buffers::PackedBufferResolution::Available(packed),
            );
            held.retire_ref(1, 7);
            assert!(observed.is_retired(), "logical removal revokes new native admission");
            drop(observed);
            assert_eq!(drain_owned(&mut host), 0);
            drop(first);
            drop(second);
            assert_eq!(drain_owned(&mut host), 0, "native work still retains the allocation");
            drop(native);
            assert_eq!(drain_owned(&mut host), 1);
            assert_eq!(host.unmap_pages_calls, 1);
            assert_eq!(drain_owned(&mut host), 0);
        }
    });
}

#[test]
fn owned_nonstable_generic_runs_use_retained_ram_references_not_borrowed_aliases() {
    with_test_import_limits(Some(4096), || {
        for page in [4096, 16384] {
            crate::runtime::guest_ram_map::reset();
            let (mut host, pages) = host_and_pages(page);
            let runs = coalesce_pages_to_runs(&mut host, &pages, page, 17, page * 2 - 31).unwrap();
            assert_eq!(host.map_pages_calls, 0, "generic runs use stable RAMBlock imports");
            assert_eq!(runs.iter().map(GuestRun::len).sum::<u64>(), page * 2 - 31);
            let source = crate::backend::vulkan::engine::BufferContent::GuestRuns(
                crate::backend::vulkan::engine::GuestRunSource {
                    runs: Arc::new(runs), source_offset: 0, total_len: page * 2 - 31,
                    row_length_texels: 0, pages: None, direct_image: None,
                },
            );
            let bytes = source.cpu_bytes();
            assert!(bytes[..page as usize - 17].iter().all(|byte| *byte == 0x31));
            assert!(bytes[page as usize - 17..].iter().all(|byte| *byte == 0x72));
        }
    });
}

#[test]
fn owned_alias_admission_without_import_capability_keeps_cpu_fallback() {
    with_test_import_limits(None, || {
        let (mut host, pages) = host_and_pages(4096);
        let backing = BufferBacking { gva: 0x100000, size: 8192 };
        assert!(owned_packed_buffer(&mut host, &backing, pages, 4096).is_none());
        assert!(guest_run_admission(&mut host).is_none());
        assert_eq!(host.map_pages_calls, 0);
        assert_eq!(host.unmap_pages_calls, 0);
    });
}

#[test]
fn failed_owned_alias_alignment_releases_the_unadmitted_view_once() {
    with_test_import_limits(Some(16384), || {
        let mut host = FakeHost::new();
        host.owned_map_pages = true;
        host.map_range(0x40000, 0x10000, 0x42);
        let backing = BufferBacking { gva: 0x100000, size: 4096 };
        assert!(owned_packed_buffer(&mut host, &backing, vec![0x40000], 4096).is_none());
        assert_eq!(host.map_pages_calls, 1);
        assert_eq!(host.unmap_pages_calls, 1);
        assert_eq!(drain_owned(&mut host), 0);
    });
}

#[test]
fn mapping_views_share_owned_import_and_reset_waits_for_cpu_and_native_holders() {
    with_test_import_limits(Some(4096), || {
        let mut host = FakeHost::new();
        host.owned_map_pages = true;
        host.map_range(0x40000, 0x4000, 0x42);
        let mut state = DeviceState::new(crate::model::DeviceId(1), 12);
        state.map_surface(7);
        let pages: Arc<[u64]> = Arc::from([0x40000, 0x41000, 0x42000, 0x43000]);
        let base = host.map_pages(&pages, 4096).unwrap();
        let footprint = GuestPageFootprint::new(pages.clone(), 4096).unwrap();
        let import = Arc::new(unsafe {
            GuestRamImport::new_owned_host_allocation(base, 0x4000, 4096, footprint.clone())
        }.unwrap());
        let mapping = state.mappings.get_mut(&7).unwrap();
        mapping.contig_ptr = base;
        mapping.contig_len = 0x4000;
        mapping.contig_footprint = Some(footprint.clone());
        mapping.contig_import = Some(Arc::clone(&import));
        mapping.page_entries = pages.iter().map(|page| {
            (((page >> 12) as u32) << crate::protocol::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
                | crate::protocol::iosurface_pages::PAGE_ENTRY_VALID
        }).collect();
        let (same, same_pages) = mapper::ensure_owned_contig_import_with_footprint(
            &mut state, &mut host, 7,
        ).unwrap();
        assert!(Arc::ptr_eq(&same, &import));
        assert!(same_pages.same_allocation(&footprint));
        assert_eq!(host.map_pages_calls, 1);
        let reference = GuestRef::new(Arc::clone(&import), import.slice(13, 27).unwrap()).unwrap();
        let run = GuestRun::from_reference(&reference).unwrap();
        let native = import.owned_host_allocation().unwrap();
        drop(reference);
        drop(same);
        assert!(state.take_all_host_views().is_empty(), "the typed lease owns unmap");
        assert!(import.is_retired());
        drop(import);
        assert_eq!(drain_owned(&mut host), 0);
        drop(native);
        assert_eq!(drain_owned(&mut host), 0, "CPU source remains valid after native retirement");
        drop(run);
        assert_eq!(drain_owned(&mut host), 1);
        assert_eq!(host.unmap_pages_calls, 1);
    });
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU with host-pointer buffer imports"]
fn vulkan_gpu_owned_guest_runs_observe_mutation_and_outlive_metadata_retirement() {
    use crate::backend::vulkan::engine::{self, pass_local::PassLocalTarget, StorageBufferResource};
    use crate::backend::vulkan::sampled_shader::graphics_tests::{assemble, shader};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};

    crate::runtime::guest_ram_map::reset();
    let mut state = DeviceState::new(DeviceId(0xabc3), PAGE_SHIFT_ARM64E);
    let target = PassLocalTarget::new(16, 8, ash::vk::Format::B8G8R8A8_UNORM).unwrap();
    let fragment = Arc::new(assemble(r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %output Location 0
OpDecorate %params DescriptorSet 0
OpDecorate %params Binding 0
OpDecorate %Params BufferBlock
OpMemberDecorate %Params 0 Offset 0
%void = OpTypeVoid
%function = OpTypeFunction %void
%float = OpTypeFloat 32
%color = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%zero = OpConstant %uint 0
%Params = OpTypeStruct %color
%params_ptr = OpTypePointer Uniform %Params
%color_ptr = OpTypePointer Uniform %color
%output_ptr = OpTypePointer Output %color
%params = OpVariable %params_ptr Uniform
%output = OpVariable %output_ptr Output
%main = OpFunction %void None %function
%entry = OpLabel
%address = OpAccessChain %color_ptr %params %zero
%value = OpLoad %color %address
OpStore %output %value
OpReturn
OpFunctionEnd
"#));
    let bytes = |color: [f32; 4]| -> Vec<u8> {
        color.into_iter().flat_map(f32::to_le_bytes).collect()
    };
    let mut request = engine::DrawRequest {
        width: 16,
        height: 8,
        vertex_count: 3,
        target_identity: Some(target.identity().clone()),
        skip_readback: true,
        vert_spirv: Arc::new(shader(true, false, 0, 0, 1.0)),
        frag_spirv: fragment,
        storage_buffers: vec![StorageBufferResource {
            binding: 0,
            content: engine::BufferContent::Bytes(Arc::new(bytes([1.0; 4]))),
        }],
        ..Default::default()
    };
    engine::execute_draw_request(&state, &request).unwrap();
    engine::read_target(target.identity()).unwrap();

    let (mut host, pages) = host_and_pages(16384);
    assert!(!host.map_pages_stable() && host.map_pages_owned());
    host.write_gpa(pages[0], &bytes([1.0, 0.0, 0.0, 1.0])).unwrap();
    let backing = BufferBacking { gva: 0x100000, size: 32768 };
    let packed = owned_packed_buffer(&mut host, &backing, pages.clone(), 16384).unwrap();
    request.storage_buffers[0].content =
        bound_buffer_content(&slice_packed_buffer(&packed, 0, backing.size).unwrap());
    state.bound_buffers.insert_packed(
        77, 17, crate::runtime::bound_buffers::PackedBufferResolution::Available(packed),
    );

    let before = engine::counter_snapshot();
    engine::execute_draw_request(&state, &request).unwrap();
    let red = engine::read_target(target.identity()).unwrap().into_rgba8().unwrap();
    assert_eq!(red.len(), 16 * 8 * 4);
    assert!(red.chunks_exact(4).all(|pixel| pixel == [255, 0, 0, 255]));
    assert!(engine::counter_snapshot().buffer_guest_imports > before.buffer_guest_imports,
        "the test must bind an actual native guest buffer, not a CPU snapshot");

    host.write_gpa(pages[0], &bytes([0.0, 1.0, 0.0, 1.0])).unwrap();
    let warm_before = engine::counter_snapshot().buffer_guest_imports;
    assert!(engine::execute_draw_request(&state, &request).unwrap().pixels.is_empty());
    assert!(engine::counter_snapshot().buffer_guest_imports > warm_before);
    state.bound_buffers.retire_ref(77, 17);
    request.storage_buffers.clear();
    assert_eq!(mapper::drain_deferred_unmaps(&mut host), 0,
        "only the native recorded/in-flight owners remain; they still need the host view");
    assert_eq!(host.unmap_pages_calls, 0);

    let green = engine::read_target(target.identity()).unwrap().into_rgba8().unwrap();
    assert_eq!(green.len(), 16 * 8 * 4);
    assert!(green.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    engine::quiesce_guest_reads();
    assert_eq!(mapper::drain_deferred_unmaps(&mut host), 1);
    assert_eq!(host.map_pages_calls, 1);
    assert_eq!(host.unmap_pages_calls, 1);
    assert_eq!(mapper::drain_deferred_unmaps(&mut host), 0);
    eprintln!("owned_guest_gpu imports={} mutation=PASS retired_unmaps=1",
        engine::counter_snapshot().buffer_guest_imports - before.buffer_guest_imports);
}

#[test]
#[ignore = "requires an exclusive Vulkan GPU with host-pointer buffer imports"]
fn vulkan_gpu_owned_image_admission_preserves_native_pixels_mutation_and_lifetime() {
    use crate::backend::vulkan::engine::{
        self, pass_local::PassLocalTarget, GuestRunSource, SampledImageResource,
        SampledSource, SamplerResource,
    };
    use crate::backend::vulkan::sampled_shader::graphics_tests::{assemble, shader};
    use crate::model::{DeviceId, TaskResource, PAGE_SHIFT_ARM64E};
    use crate::runtime::gather_witness::GatherVouch;

    for native_half in [false, true] {
        crate::runtime::guest_ram_map::reset();
        let mut state = DeviceState::new(DeviceId(0xabc4 + u64::from(native_half)), PAGE_SHIFT_ARM64E);
        let target = PassLocalTarget::new(16, 8, ash::vk::Format::B8G8R8A8_UNORM).unwrap();
        let mut request = engine::DrawRequest {
            width: 16, height: 8, vertex_count: 3, skip_readback: true,
            target_identity: Some(target.identity().clone()),
            vert_spirv: Arc::new(shader(true, false, 0, 0, 1.0)),
            frag_spirv: Arc::new(shader(false, false, 0, 0, 1.0)),
            ..Default::default()
        };
        engine::execute_draw_request(&state, &request).unwrap();
        engine::read_target(target.identity()).unwrap();

        let (mut host, pages) = host_and_pages(16384);
        let backing = BufferBacking { gva: 0x100000, size: 32768 };
        let packed = owned_packed_buffer(&mut host, &backing, pages.clone(), 16384).unwrap();
        assert!(packed.import.owned_host_allocation().is_some());
        let owner = TaskResource::new(Default::default(), Arc::from([]));
        let texel_bytes = if native_half { 8 } else { 4 };
        let row_pitch = 512u64;
        let plane_offset = 64u64;
        let span = 63 * row_pitch + 16 * texel_bytes;
        let direct = sampled_backing_from_packed(
            &packed, plane_offset, row_pitch, span, owner.lifetime_ref(),
        ).unwrap();
        let source = GuestRunSource {
            runs: Arc::clone(&packed.runs), source_offset: plane_offset, total_len: span,
            row_length_texels: (row_pitch / texel_bytes) as u32,
            pages: Some(Arc::clone(&packed.pages)), direct_image: Some(direct),
        };
        state.bound_buffers.insert_packed(
            88, 18, crate::runtime::bound_buffers::PackedBufferResolution::Available(packed),
        );
        let scaling = if native_half {
            "%scale = OpConstantComposite %vec4 %half %eighth %one %one\n\
             %offset = OpConstantComposite %vec4 %half %zero %zero %zero"
        } else {
            "%scale = OpConstantComposite %vec4 %one %one %one %one\n\
             %offset = OpConstantComposite %vec4 %zero %zero %zero %zero"
        };
        request.frag_spirv = Arc::new(assemble(&format!(r#"
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %output
OpExecutionMode %main OriginUpperLeft
OpDecorate %output Location 0
OpDecorate %image DescriptorSet 0
OpDecorate %image Binding 32
OpDecorate %sampler DescriptorSet 0
OpDecorate %sampler Binding 160
%void = OpTypeVoid
%function = OpTypeFunction %void
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec4 = OpTypeVector %float 4
%zero = OpConstant %float 0
%one = OpConstant %float 1
%half = OpConstant %float 0.5
%eighth = OpConstant %float 0.125
%coord = OpConstantComposite %vec2 %half %half
{scaling}
%image_type = OpTypeImage %float 2D 0 0 0 1 Unknown
%sampler_type = OpTypeSampler
%combined_type = OpTypeSampledImage %image_type
%image_ptr = OpTypePointer UniformConstant %image_type
%sampler_ptr = OpTypePointer UniformConstant %sampler_type
%output_ptr = OpTypePointer Output %vec4
%image = OpVariable %image_ptr UniformConstant
%sampler = OpVariable %sampler_ptr UniformConstant
%output = OpVariable %output_ptr Output
%main = OpFunction %void None %function
%entry = OpLabel
%loaded_image = OpLoad %image_type %image
%loaded_sampler = OpLoad %sampler_type %sampler
%combined = OpSampledImage %combined_type %loaded_image %loaded_sampler
%sample = OpImageSampleExplicitLod %vec4 %combined %coord Lod %zero
%scaled = OpFMul %vec4 %sample %scale
%value = OpFAdd %vec4 %scaled %offset
OpStore %output %value
OpReturn
OpFunctionEnd
"#)));
        request.sampled_images = vec![SampledImageResource {
            binding: 32, array_element: 0, descriptor_count: 1,
            width: 16, height: 64, layers: 1,
            kind: reims_vgpu_core::texture_shape::TextureKind::D2,
            multisampled: false, source: SampledSource::GuestRuns(source, GatherVouch::Fresh),
            byte_origin: Default::default(),
            format: if native_half {
                ash::vk::Format::R16G16B16A16_SFLOAT
            } else {
                ash::vk::Format::B8G8R8A8_UNORM
            },
            identity: None, swizzle: Default::default(),
        }];
        let mut sampler = SamplerResource::normalized_default(160);
        sampler.min_filter = 0;
        sampler.mag_filter = 0;
        request.samplers = vec![sampler];

        let before = engine::counter_snapshot();
        for changed in [false, true] {
            let texel = if native_half {
                let words: [u16; 4] = if changed {
                    [0xbc00, 0x4800, 0x3800, 0x3400] // -1, 8, 0.5, 0.25
                } else {
                    [0xb800, 0x4400, 0x3600, 0x3c00] // -0.5, 4, 0.375, 1
                };
                words.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>()
            } else if changed {
                vec![0, 255, 0, 255]
            } else {
                vec![0, 0, 255, 255]
            };
            let mut bytes = vec![0xa5; 32768];
            for y in 0..64 {
                for x in 0..16 {
                    let at = (plane_offset + y * row_pitch + x * texel_bytes) as usize;
                    bytes[at..at + texel.len()].copy_from_slice(&texel);
                }
            }
            for (page, data) in pages.iter().zip(bytes.chunks_exact(16384)) {
                host.write_gpa(*page, data).unwrap();
            }
            assert!(engine::execute_draw_request(&state, &request).unwrap().pixels.is_empty());
            let now = engine::counter_snapshot();
            assert!(now.guest_image_initial_contents_refused > before.guest_image_initial_contents_refused,
                "the owned IMAGE admission must reach its preservation refusal");
            assert!(now.sampled_guest_imports > before.sampled_guest_imports,
                "the preserving fallback must use the native guest-buffer transfer");
            if changed {
                state.bound_buffers.retire_ref(88, 18);
                request.sampled_images.clear();
                assert_eq!(mapper::drain_deferred_unmaps(&mut host), 0);
                assert_eq!(host.unmap_pages_calls, 0);
            }
            let pixels = engine::read_target(target.identity()).unwrap().into_rgba8().unwrap();
            let expected = match (native_half, changed) {
                (false, false) => [255, 0, 0, 255],
                (false, true) => [0, 255, 0, 255],
                (true, false) => [64, 128, 96, 255],
                (true, true) => [0, 255, 128, 64],
            };
            assert_eq!(pixels.len(), 16 * 8 * 4);
            assert!(pixels.chunks_exact(4).all(|pixel| pixel == expected),
                "native_half={native_half} changed={changed} first={:?}", &pixels[..4]);
            for (page, expected) in pages.iter().zip(bytes.chunks_exact(16384)) {
                let mut observed = vec![0; 16384];
                host.read_gpa(*page, &mut observed).unwrap();
                assert_eq!(observed, expected, "sampling must not alter guest texels or padding");
            }
        }
        engine::quiesce_guest_reads();
        assert_eq!(mapper::drain_deferred_unmaps(&mut host), 1);
        assert_eq!(host.unmap_pages_calls, 1);
        assert_eq!(mapper::drain_deferred_unmaps(&mut host), 0);
        eprintln!("owned_image_gpu native_half={native_half} image_refused={} gpu_transfers={} pixels_mutation_padding=PASS unmaps=1",
            engine::counter_snapshot().guest_image_initial_contents_refused - before.guest_image_initial_contents_refused,
            engine::counter_snapshot().sampled_guest_imports - before.sampled_guest_imports);
    }
}
