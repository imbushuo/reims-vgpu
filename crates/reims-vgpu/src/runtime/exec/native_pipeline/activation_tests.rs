use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, TaskResource};
use crate::runtime::decode::resource::*;
use crate::runtime::host::FakeHost;
use crate::runtime::m2v_cache::CachedShader;
use metal2vulkan::reflect::{DescriptorLocation, ResourceBinding, ResourceKind, ShaderReflection};
use reims_vgpu_core::bind::{BufferBinding, ObjectBinding};
use reims_vgpu_core::encoder::RenderEncoderState;
use reims_vgpu_core::identity::{ObjectListRef, ResourceId, SlotGeneration};
use reims_vgpu_core::render::{Instancing, PassDescriptorSlot, PrimitiveType};
use reims_vgpu_protocol::endian::{st16, st32, st64};

const TASK: u32 = 17;
const PIPELINE: ResourceId = ResourceId {
    slot: ObjectListRef(44),
    generation: SlotGeneration(3),
};

fn declared_module(bindings: &[u32]) -> Vec<u8> {
    let mut words = vec![
        0x0723_0203,
        0x0001_0300,
        0,
        1024,
        0,
        (2 << 16) | 17,
        1,
        (3 << 16) | 14,
        0,
        1,
        (5 << 16) | 15,
        4,
        1,
        0x6e69_616d,
        0,
        (3 << 16) | 22,
        2,
        32,
        (9 << 16) | 25,
        3,
        2,
        1,
        0,
        0,
        0,
        1,
        0,
        (2 << 16) | 26,
        4,
        (9 << 16) | 25,
        5,
        2,
        6,
        0,
        0,
        0,
        2,
        0,
        (4 << 16) | 32,
        20,
        12,
        2,
        (4 << 16) | 32,
        21,
        0,
        3,
        (4 << 16) | 32,
        22,
        0,
        4,
        (4 << 16) | 32,
        23,
        0,
        5,
    ];
    for (i, binding) in bindings.iter().enumerate() {
        let id = 100 + i as u32;
        let (pointer, storage) = match *binding {
            32..=159 | 704..=831 => (21, 0),
            160..=191 => (22, 0),
            192..=199 => (23, 0),
            _ => (20, 12),
        };
        words.extend([
            (4 << 16) | 71,
            id,
            33,
            *binding,
            (4 << 16) | 71,
            id,
            34,
            0,
            (4 << 16) | 59,
            pointer,
            id,
            storage,
        ]);
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn reflection_binding(kind: ResourceKind, index: u32, binding: u32) -> ResourceBinding {
    ResourceBinding {
        kind,
        metal_index: index,
        descriptor: Some(DescriptorLocation {
            set: 0,
            binding,
            count: 1,
        }),
        param_index: None,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        footprint: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

struct Fixture {
    state: DeviceState,
    host: FakeHost,
    pipeline: pipeline_resolve::ResolvedRenderPipeline,
    bindings: RenderEncoderState,
    pass: reims_vgpu_core::pass::PassDescriptor,
}

impl Fixture {
    fn new() -> Self {
        let mut state = DeviceState::new(DeviceId(0xac71), PAGE_SHIFT_ARM64E);
        state.define_task(TASK, 1 << 24, 2);
        state.set_object_list(TASK, 0, 256);
        let mut host = FakeHost::new();
        for page in 2..=4 {
            host.map_range(page << PAGE_SHIFT_ARM64E, 1 << PAGE_SHIFT_ARM64E, 0);
        }
        let mut directory = [0u8; 8];
        st32(&mut directory, 3);
        st32(&mut directory[4..], 1);
        host.write_gpa(2 << PAGE_SHIFT_ARM64E, &directory).unwrap();
        host.write_gpa(3 << PAGE_SHIFT_ARM64E, &4u32.to_le_bytes())
            .unwrap();
        let desc = RenderPipelineDescriptor {
            vertex_attributes: vec![VertexAttribute::default(); 32],
            ..Default::default()
        };
        let mut pipeline = (*pipeline_resolve::retained_pipeline_with_desc_for_test(desc)).clone();
        let mut reflection = (*pipeline.fragment.reflection).clone();
        reflection
            .bindings
            .extend((0..2).map(|i| reflection_binding(ResourceKind::ColorInput, i, 192 + i)));
        reflection
            .bindings
            .extend((0..4).map(|i| reflection_binding(ResourceKind::Sampler, i, 160 + i)));
        pipeline.vertex = Arc::new(CachedShader::new(
            declared_module(&[0, 1, 2, 3, 4]),
            pipeline.vertex.reflection.clone(),
        ));
        let declared: Vec<_> = [0, 1, 2, 3, 4, 5, 7, 8, 192, 193]
            .into_iter()
            .chain(35..=46)
            .chain(160..=163)
            .collect();
        pipeline.fragment = Arc::new(CachedShader::new(
            declared_module(&declared),
            Arc::new(reflection),
        ));
        let mut result = Self {
            state,
            host,
            pipeline,
            bindings: RenderEncoderState::default(),
            pass: reims_vgpu_core::pass::PassDescriptor::empty(),
        };
        result.bindings.pipeline = Some(PIPELINE);
        for i in 0..5 {
            let buffer = result.buffer(1 + i);
            result.bindings.vertex.buffers.set(
                i,
                Some(BufferBinding {
                    buffer: Some(buffer),
                    offset: 0,
                    stride: None,
                }),
            );
        }
        for i in [0, 1, 2, 3, 4, 5, 7, 8] {
            let buffer = result.buffer(10 + i);
            result.bindings.fragment.buffers.set(
                i,
                Some(BufferBinding {
                    buffer: Some(buffer),
                    offset: 0,
                    stride: None,
                }),
            );
        }
        for i in 3..15 {
            let object = result.texture(50 + i, 0x73);
            result.bindings.fragment.textures.set(
                i,
                Some(ObjectBinding {
                    object: Some(object),
                    lod_clamps: None,
                }),
            );
        }
        for i in 0..2 {
            let object = result.texture(100 + i, 0x73);
            result.pass.color[i as usize].texture = Some(object);
        }
        for i in 0..4 {
            let object = result.sampler(20 + i);
            result.bindings.fragment.samplers.set(
                i,
                Some(ObjectBinding {
                    object: Some(object),
                    lod_clamps: None,
                }),
            );
        }
        // No uniform or texel bytes exist: only immutable object metadata and serializer pages.
        result.install_pipeline();
        result
    }

    fn install_pipeline(&self) {
        let retained = pipeline_resolve::retained(&self.state).unwrap();
        retained.delete(TASK, PIPELINE.slot.0);
        retained.register(TASK, PIPELINE.slot.0, Arc::new(self.pipeline.clone()));
    }

    fn resource(&self, reference: u32, object_type: u8, descriptor: Vec<u8>) -> ResourceId {
        let entry = ListObjectEntry {
            object_type,
            descriptor_length: descriptor.len() as u32,
            descriptor_gva: 0,
        };
        let storage =
            objects::declared_storage(&self.state, TASK, reference, &entry, &descriptor).unwrap();
        let declaration = self.state.declare_object(TASK, reference, storage).unwrap();
        self.state.task_resources.register(
            TASK,
            declaration.id,
            Arc::new(TaskResource::new(entry, descriptor.into())),
        );
        declaration.id
    }

    fn buffer(&self, reference: u32) -> ResourceId {
        let mut descriptor = vec![0; LINEAR_DESC_MIN_LEN];
        st64(&mut descriptor[LINEAR_DESC_SIZE..], 4096);
        st64(
            &mut descriptor[LINEAR_DESC_HANDLE..],
            u64::from(0x100 + reference),
        );
        self.resource(reference, OBJECT_TYPE_BUFFER, descriptor)
    }

    fn texture(&self, reference: u32, format: u16) -> ResourceId {
        let mut descriptor = vec![0; TEXTURE_DESC_PIXEL_FORMAT + 2];
        st64(&mut descriptor[LINEAR_DESC_SIZE..], 4096);
        st32(&mut descriptor[LINEAR_DESC_HANDLE..], 0x100 + reference);
        st32(&mut descriptor[TEXTURE_DESC_WIDTH..], 4);
        st32(&mut descriptor[TEXTURE_DESC_HEIGHT..], 4);
        st32(&mut descriptor[TEXTURE_DESC_ROW_STRIDE..], 32);
        st32(&mut descriptor[TEXTURE_DESC_USED_SIZE..], 128);
        descriptor[TEXTURE_DESC_BYTES_PER_ELEMENT] = 8;
        st16(&mut descriptor[TEXTURE_DESC_PIXEL_FORMAT..], format);
        self.resource(reference, OBJECT_TYPE_TEXTURE, descriptor)
    }

    fn sampler(&mut self, reference: u32) -> ResourceId {
        use reims_vgpu_wire::ops::sampler::NEW_SAMPLER_TOTAL_LEN;
        let gva = 0x2000 + u64::from(reference) * 64;
        let data = 4 << PAGE_SHIFT_ARM64E;
        let mut entry = [0; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry,
            u32::from(OBJECT_TYPE_SERIALIZER_OBJECT) | (NEW_SAMPLER_TOTAL_LEN << 8),
        );
        st64(&mut entry[4..], gva);
        self.host
            .write_gpa(
                data + u64::from(reference) * OBJECT_LIST_ENTRY_LEN as u64,
                &entry,
            )
            .unwrap();
        let mut descriptor = vec![0; NEW_SAMPLER_TOTAL_LEN as usize];
        st32(&mut descriptor, SERIALIZER_OBJECT_SAMPLER);
        st32(&mut descriptor[4..], NEW_SAMPLER_TOTAL_LEN);
        st32(&mut descriptor[8..], reference);
        st32(&mut descriptor[12..], 0x8400_0000);
        self.host.write_gpa(data + gva, &descriptor).unwrap();
        ResourceId {
            slot: ObjectListRef(reference),
            generation: SlotGeneration(1),
        }
    }

    fn metadata(&self) -> Result<Metadata, &'static str> {
        draw_metadata(
            &self.state,
            &self.host,
            TASK,
            &self.bindings,
            &self.pass,
            Default::default(),
            reims_vgpu_core::topology::PrimitiveType::Triangle,
            1,
        )
    }

    fn work(&self, repeats: usize) -> ExecWork {
        use reims_vgpu_protocol::render::ShaderStage;
        let mut work = ExecWork::default();
        work.arenas.pass_descriptors.push(self.pass);
        let mut ops = vec![
            RenderOp::WriteDescriptor {
                descriptor: PassDescriptorSlot(0),
            },
            RenderOp::SetPipeline { pipeline: PIPELINE },
        ];
        for (stage, bindings) in [
            (ShaderStage::Vertex, &self.bindings.vertex),
            (ShaderStage::Fragment, &self.bindings.fragment),
        ] {
            for (index, bind) in bindings.buffers.bound() {
                let start = work.arenas.buffer_bindings.len() as u32;
                work.arenas.buffer_bindings.push(bind);
                ops.push(RenderOp::BindBuffers {
                    stage,
                    first: index,
                    entries: reims_vgpu_core::sync::ResourceSpan { start, len: 1 },
                });
            }
            for (sampler, table) in [(false, &bindings.textures), (true, &bindings.samplers)] {
                for (index, bind) in table.bound() {
                    let start = work.arenas.object_bindings.len() as u32;
                    work.arenas.object_bindings.push(bind);
                    let entries = reims_vgpu_core::sync::ResourceSpan { start, len: 1 };
                    ops.push(if sampler {
                        RenderOp::BindSamplers {
                            stage,
                            first: index,
                            entries,
                        }
                    } else {
                        RenderOp::BindTextures {
                            stage,
                            first: index,
                            entries,
                        }
                    });
                }
            }
        }
        ops.extend(std::iter::repeat_n(
            RenderOp::Draw(DrawOp::Primitives {
                primitive: PrimitiveType(3),
                vertex_start: 0,
                vertex_count: 3,
                instances: Instancing::default(),
            }),
            repeats,
        ));
        work.streams.push(reims_vgpu_core::exec::ResolvedStream {
            begin: reims_vgpu_core::stream::SegmentBegin {
                kind: reims_vgpu_protocol::segment::SegmentKind::Render,
                protection: None,
            },
            records: ops
                .into_iter()
                .enumerate()
                .map(|(i, op)| reims_vgpu_core::exec::StreamRecord {
                    at: reims_vgpu_core::stream::StreamPosition {
                        segment: 0,
                        record: i as u32,
                    },
                    op: ResolvedOperation::Render(op),
                })
                .collect(),
        });
        work
    }
}

fn assert_b20_shape(metadata: &Metadata) {
    let mut got = metadata.bindings.clone();
    got.sort_by_key(|b| b.binding);
    let mut expected: Vec<_> = (0..5)
        .chain([672, 673, 674, 675, 676, 677, 679, 680])
        .map(|binding| BindingSig {
            binding,
            ty: 7,
            stages: 17,
            count: 1,
        })
        .collect();
    expected.extend((707..=718).map(|binding| BindingSig {
        binding,
        ty: 2,
        stages: 17,
        count: 1,
    }));
    expected.extend((832..=835).map(|binding| BindingSig {
        binding,
        ty: 0,
        stages: 17,
        count: 1,
    }));
    expected.extend((192..=193).map(|binding| BindingSig {
        binding,
        ty: 10,
        stages: 16,
        count: 1,
    }));
    expected.sort_by_key(|b| b.binding);
    assert_eq!(got, expected);
    assert_eq!(
        metadata.pass.color0_format,
        ash::vk::Format::R16G16B16A16_SFLOAT
    );
    assert_eq!(
        metadata.pass.secondary[0].format,
        metadata.pass.color0_format
    );
    assert_eq!(metadata.pass.secondary_count, 1);
    assert_eq!(metadata.pass.color_input, 3);
    assert_eq!(metadata.color_write_mask, [engine::ColorWriteMask::ALL; 8]);
}

#[test]
fn native_collector_admits_inactive_attributes_unused_sampled_and_serializer_samplers() {
    let mut fixture = Fixture::new();
    let absent = ResourceId {
        slot: ObjectListRef(250),
        generation: SlotGeneration(1),
    };
    fixture.bindings.fragment.textures.set(
        0,
        Some(ObjectBinding {
            object: Some(absent),
            lod_clamps: None,
        }),
    );
    fixture.bindings.fragment.samplers.set(
        9,
        Some(ObjectBinding {
            object: Some(absent),
            lod_clamps: None,
        }),
    );
    let work = fixture.work(200);
    let records = work.streams.clone();
    let mut calls = 0;
    let plans = scan(&work, |bindings, pass, raster, topology, viewports| {
        calls += 1;
        draw_metadata(
            &fixture.state,
            &fixture.host,
            TASK,
            bindings,
            pass,
            raster,
            topology,
            viewports,
        )
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(plans.len(), 1);
    assert_b20_shape(&plans[0]);
    assert_eq!(records, work.streams);
    assert!(fixture.host.actions.is_empty());
    assert_eq!(fixture.host.map_pages_calls, 0);
    assert!(fixture.state.task_sampler_states.get(TASK, 20).is_some());
}

#[test]
fn native_collector_preserves_active_attributes_stride_and_format_variants() {
    let mut fixture = Fixture::new();
    Arc::make_mut(&mut fixture.pipeline.desc).vertex_attributes[0] = VertexAttribute {
        format: 30,
        stride: 16,
        ..Default::default()
    };
    fixture.install_pipeline();
    assert_eq!(
        fixture.metadata().unwrap_err(),
        "native_preflight_vertex_attributes"
    );
    let mut bind = fixture.bindings.vertex.buffers.get(0).unwrap();
    bind.stride = Some(0);
    fixture.bindings.vertex.buffers.set(0, Some(bind));
    assert!(fixture.metadata().is_ok());
    bind.stride = Some(16);
    fixture.bindings.vertex.buffers.set(0, Some(bind));
    assert_eq!(
        fixture.metadata().unwrap_err(),
        "native_preflight_vertex_attributes"
    );
    bind.stride = Some(0);
    fixture.bindings.vertex.buffers.set(0, Some(bind));
    let fp16 = fixture.metadata().unwrap().pass;
    fixture.pass.color[0].texture = Some(fixture.texture(1010, 0x46));
    let rgba = fixture.metadata().unwrap().pass;
    fixture.pass.color[0].texture = Some(fixture.texture(1011, 0x50));
    let bgra = fixture.metadata().unwrap().pass;
    assert_ne!(fp16, rgba);
    assert_ne!(rgba, bgra);
    assert_ne!(fp16, bgra);
}

#[test]
fn native_collector_complete_both_stage_proof_required_before_skipping_unknown_owner() {
    for vertex_declares in [false, true] {
        let mut fixture = Fixture::new();
        let absent = ResourceId {
            slot: ObjectListRef(250),
            generation: SlotGeneration(1),
        };
        fixture.bindings.fragment.textures.set(
            0,
            Some(ObjectBinding {
                object: Some(absent),
                lod_clamps: None,
            }),
        );
        let old = if vertex_declares {
            &mut fixture.pipeline.vertex
        } else {
            &mut fixture.pipeline.fragment
        };
        *old = Arc::new(CachedShader::new(
            declared_module(&[if vertex_declares { 704 } else { 32 }]),
            old.reflection.clone(),
        ));
        fixture.install_pipeline();
        assert_eq!(
            fixture.metadata().unwrap_err(),
            "native_preflight_texture_owner"
        );
    }
    for extra in [
        vec![0u32],
        vec![(2 << 16) | 73, 10],
        vec![(1 << 16) | 65000],
    ] {
        let mut fixture = Fixture::new();
        let mut bytes = fixture.pipeline.vertex.spirv.clone();
        bytes.extend(extra.into_iter().flat_map(u32::to_le_bytes));
        fixture.pipeline.vertex = Arc::new(CachedShader::new(
            bytes,
            fixture.pipeline.vertex.reflection.clone(),
        ));
        fixture.install_pipeline();
        fixture.bindings.fragment.textures.set(
            0,
            Some(ObjectBinding {
                object: Some(ResourceId {
                    slot: ObjectListRef(250),
                    generation: SlotGeneration(1),
                }),
                lod_clamps: None,
            }),
        );
        assert_eq!(
            fixture.metadata().unwrap_err(),
            "native_preflight_texture_owner"
        );
    }
}

#[test]
fn native_collector_source_proofs_are_retained() {
    let fixture = Fixture::new();
    let first = PipelineInputs::new(&fixture.state, &fixture.pipeline, &fixture.bindings).unwrap();
    let again = PipelineInputs::new(&fixture.state, &fixture.pipeline, &fixture.bindings).unwrap();
    for i in 0..2 {
        assert!(Arc::ptr_eq(&first.sources[i], &again.sources[i]));
    }
}

#[test]
fn native_collector_sampler_clamps_defaults_and_unnormalized_fallback_match_renderer() {
    let mut fixture = Fixture::new();
    let mut binding = fixture.bindings.fragment.samplers.get(0).unwrap();
    binding.lod_clamps = Some((
        reims_vgpu_core::bind::LodClamp(1.25f32.to_bits()),
        reims_vgpu_core::bind::LodClamp(2.5f32.to_bits()),
    ));
    fixture.bindings.fragment.samplers.set(0, Some(binding));
    fixture.bindings.fragment.samplers.set(1, None);
    let inputs = PipelineInputs::new(&fixture.state, &fixture.pipeline, &fixture.bindings).unwrap();
    let samplers = metadata_samplers(
        &fixture.state,
        &fixture.host,
        TASK,
        &fixture.bindings,
        &inputs,
    )
    .unwrap();
    let overridden = samplers.iter().find(|s| s.binding == 832).unwrap();
    assert_eq!(
        (overridden.lod_min, overridden.lod_max),
        (1.25f32.to_bits(), 2.5f32.to_bits())
    );
    assert!(
        samplers
            .iter()
            .any(|s| s.binding == 833 && !s.unnormalized_coordinates)
    );
    let mut descriptor = fixture
        .state
        .task_sampler_states
        .get(TASK, 20)
        .unwrap()
        .descriptor
        .clone();
    descriptor.normalized_coordinates = false;
    fixture.state.task_sampler_states.delete(TASK, 20);
    fixture.state.task_sampler_states.register(
        TASK,
        20,
        Arc::new(crate::model::TaskSamplerState { descriptor }),
    );
    assert_eq!(
        fixture.metadata().unwrap_err(),
        "native_preflight_unnormalized_sampler"
    );
}

#[test]
fn native_collector_dirty_records_topology_and_fallback_invalidate_consecutive_reuse() {
    let fixture = Fixture::new();
    let mut work = fixture.work(100);
    let draw = work.streams[0].records.last().unwrap().op;
    let mut calls = 0;
    let changes = [
        RenderOp::RebindBufferOffset {
            stage: reims_vgpu_protocol::render::ShaderStage::Vertex,
            index: 0,
            offset: 16,
            stride: Some(0),
        },
        RenderOp::SetViewports(reims_vgpu_core::sync::ResourceSpan { start: 0, len: 2 }),
        RenderOp::SetScissorRects(reims_vgpu_core::sync::ResourceSpan { start: 0, len: 2 }),
        RenderOp::SetCullMode(2),
        RenderOp::WriteDescriptor {
            descriptor: PassDescriptorSlot(0),
        },
        RenderOp::SetPipeline { pipeline: PIPELINE },
    ];
    for change in changes {
        for op in [ResolvedOperation::Render(change), draw, draw] {
            let mut record = *work.streams[0].records.last().unwrap();
            record.op = op;
            work.streams[0].records.push(record);
        }
    }
    let mut record = *work.streams[0].records.last().unwrap();
    record.op = ResolvedOperation::Render(RenderOp::Draw(DrawOp::Primitives {
        primitive: PrimitiveType(1),
        vertex_start: 0,
        vertex_count: 2,
        instances: Instancing::default(),
    }));
    work.streams[0].records.push(record);
    let plans = scan::<()>(&work, |_, _, _, _, _| {
        calls += 1;
        Err("fixture_fallback")
    })
    .unwrap();
    assert!(plans.is_empty());
    assert_eq!(calls, 1 + changes.len() + 1);
}
#[test]
fn native_collector_diagnostics_are_identity_bounded() {
    let mut diagnostics = Diagnostics::default();
    let source = Some([(1, 2, 3), (4, 5, 6)]);
    assert!(diagnostics.admit(TASK, Some(PIPELINE), source, "refusal"));
    assert!(!diagnostics.admit(TASK, Some(PIPELINE), source, "refusal"));
    assert!(diagnostics.admit(TASK + 1, Some(PIPELINE), source, "refusal"));
    assert!(diagnostics.admit(
        TASK,
        Some(ResourceId {
            generation: SlotGeneration(4),
            ..PIPELINE
        }),
        source,
        "refusal"
    ));
    assert!(diagnostics.admit(
        TASK,
        Some(PIPELINE),
        Some([(1, 2, 3), (4, 5, 7)]),
        "refusal"
    ));
    for task in 0..400 {
        diagnostics.admit(task, Some(PIPELINE), source, "refusal");
    }
    assert_eq!(diagnostics.0.len(), Diagnostics::LIMIT);
}

#[test]
#[ignore = "CPU only; requires private target/vulkan-native-collector-replay captured modules/reflection"]
fn native_collector_captured_b20_full_record_metadata_without_uniform_contents() {
    struct MetadataHost<'a>(&'a FakeHost);
    impl HostMemory for MetadataHost<'_> {
        fn read_gpa(
            &self,
            gpa: u64,
            bytes: &mut [u8],
        ) -> Result<(), crate::runtime::host::MemError> {
            assert!(
                gpa >= 2 << PAGE_SHIFT_ARM64E
                    && gpa.checked_add(bytes.len() as u64).unwrap() <= 5 << PAGE_SHIFT_ARM64E,
                "collector attempted a read outside page tables and serializer metadata"
            );
            self.0.read_gpa(gpa, bytes)
        }
        fn write_gpa(&mut self, _: u64, _: &[u8]) -> Result<(), crate::runtime::host::MemError> {
            panic!("metadata preparation must not write guest memory")
        }
    }
    impl HostOps for MetadataHost<'_> {
        fn mono_ns(&self) -> u64 {
            0
        }
        fn enqueue(&mut self, _: crate::runtime::host::HostAction) {
            panic!("metadata must not publish");
        }
        fn schedule_bh(&mut self) {
            panic!("metadata must not execute");
        }
    }
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/vulkan-native-collector-replay");
    let mut fixture = Fixture::new();
    let reflection: ShaderReflection =
        serde_json::from_slice(&std::fs::read(directory.join("fragment.json")).unwrap()).unwrap();
    fixture.pipeline.vertex = Arc::new(CachedShader::new(
        std::fs::read(directory.join("vertex.spv")).unwrap(),
        fixture.pipeline.vertex.reflection.clone(),
    ));
    let captured = std::fs::read(directory.join("fragment.spv")).unwrap();
    let mut base: Vec<_> = captured
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    // This capture is already relocated. Undo that exact recorded layout for
    // the CachedShader input, then require the production variant to round-trip it.
    let mut at = 5;
    while at < base.len() {
        let count = (base[at] >> 16) as usize;
        assert!(count > 0 && at + count <= base.len());
        if base[at] & 0xffff == 71
            && count == 4
            && base[at + 2] == 33
            && base[at + 3] >= spirv_bind::FRAG_BUFFER_BINDING_OFFSET
        {
            base[at + 3] -= spirv_bind::FRAG_BUFFER_BINDING_OFFSET;
        }
        at += count;
    }
    fixture.pipeline.fragment = Arc::new(CachedShader::new(
        base.into_iter().flat_map(u32::to_le_bytes).collect(),
        Arc::new(reflection),
    ));
    let round_trip: Vec<_> = fixture
        .pipeline
        .fragment
        .variant(true, true)
        .words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    assert_eq!(round_trip, captured);
    fixture.install_pipeline();
    // Sticky absent texture0 and unrelated provided sampler9 have no metadata owner.
    let absent = ResourceId {
        slot: ObjectListRef(250),
        generation: SlotGeneration(1),
    };
    fixture.bindings.fragment.textures.set(
        0,
        Some(ObjectBinding {
            object: Some(absent),
            lod_clamps: None,
        }),
    );
    fixture.bindings.fragment.samplers.set(
        9,
        Some(ObjectBinding {
            object: Some(absent),
            lod_clamps: None,
        }),
    );
    for index in [2, 3] {
        fixture.bindings.fragment.samplers.set(
            index,
            Some(ObjectBinding {
                object: Some(absent),
                lod_clamps: None,
            }),
        );
    }
    let plans = collect(
        &fixture.state,
        &MetadataHost(&fixture.host),
        TASK,
        &fixture.work(200),
    )
    .unwrap();
    assert_eq!(plans.len(), 1);
    assert_b20_shape(&plans[0]);
    let sources =
        engine::precreated::metadata_sources(&fixture.state, &plans[0].vertex, &plans[0].fragment)
            .unwrap();
    let ids = sources
        .each_ref()
        .map(|s| format!("{:x}{:x}:{}", s.digest.a, s.digest.b, s.digest.len));
    assert_eq!(
        ids,
        [
            "4e390ef564752fca6314f31ef0cf9fda:8480",
            "fca3c0a9bb3a8fe12e5457e29e8c903e:729252"
        ]
    );
    assert!(fixture.host.actions.is_empty());
    assert_eq!(fixture.host.map_pages_calls, 0);
    eprintln!(
        "native_collector_b20 records=200 metadata_plans=1 layout_bindings=31 exact_sources=PASS texels=0 uniform_contents=0 native_calls=0"
    );
}
