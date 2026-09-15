use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use super::caches::{AttrKey, ObjectCaches, PipelineKey};
pub(crate) use super::caches::{BindingSig, Color0Load, PassKey, SecondaryAttachKey};
use super::compile_work::{Status, Ticket, Worker};
use super::context::DeviceContext;
use super::pools::ResourcePools;
use super::{DrawError, EngineCounters};

#[derive(Clone, Debug)]
pub(super) struct Key {
    pipeline: PipelineKey,
    attrs: Vec<AttrKey>,
    bindings: Vec<BindingSig>,
    push_constant: Option<(u32, u32)>,
    push_descriptors: bool,
    source: [Arc<Vec<u32>>; 2],
}

fn same_sources(left: &[Arc<Vec<u32>>; 2], right: &[&Arc<Vec<u32>>; 2]) -> bool {
    left.iter()
        .zip(right)
        .all(|(held, current)| Arc::ptr_eq(held, current) || held.as_slice() == current.as_slice())
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline
            && self.attrs == other.attrs
            && self.bindings == other.bindings
            && self.push_constant == other.push_constant
            && self.push_descriptors == other.push_descriptors
            && same_sources(&self.source, &[&other.source[0], &other.source[1]])
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pipeline.hash(state);
        self.attrs.hash(state);
        self.bindings.hash(state);
        self.push_constant.hash(state);
        self.push_descriptors.hash(state);
    }
}

impl Key {
    pub(super) fn new(
        key: &PipelineKey,
        attrs: &[AttrKey],
        bindings: &[BindingSig],
        push_constant: Option<(u32, u32)>,
        push_descriptors: bool,
        vertex: &Arc<Vec<u32>>,
        fragment: &Arc<Vec<u32>>,
    ) -> Self {
        let mut pipeline = key.clone();
        pipeline.normalize_interns();
        Self {
            pipeline,
            attrs: attrs.to_vec(),
            bindings: bindings.to_vec(),
            push_constant,
            push_descriptors,
            source: [vertex.clone(), fragment.clone()],
        }
    }
}

pub(super) struct Plan {
    pub key: Key,
    pub vertex: Arc<Vec<u32>>,
    pub fragment: Arc<Vec<u32>>,
    pub pass: PassKey,
}

struct NativeOwners {
    caches: ObjectCaches,
    pools: ResourcePools,
    context: DeviceContext,
}

impl Drop for NativeOwners {
    fn drop(&mut self) {
        unsafe {
            self.caches.destroy_all(&self.context.device);
            self.pools.destroy_all(&self.context.device);
            self.context.destroy();
        }
    }
}

struct Compiled {
    pipeline: ash::vk::Pipeline,
    _owners: Mutex<NativeOwners>,
}

type Outcome = Result<Compiled, DrawError>;

static WORKER: OnceLock<Result<Worker<Outcome>, std::io::Error>> = OnceLock::new();
const MAX_COMPILES: usize = 32;

pub(crate) struct SourceInfo {
    pub words: Arc<Vec<u32>>,
    pub digest: super::digest::Digest128,
    declarations: super::caches::storage_descriptors::DeclarationProof,
}

impl SourceInfo {
    pub(super) fn new(words: &Arc<Vec<u32>>) -> Arc<Self> {
        Arc::new(Self {
            words: words.clone(),
            digest: super::digest::Digest128::of_u32_words(words),
            declarations: super::caches::storage_descriptors::DeclarationProof::of_final_module(
                words,
            ),
        })
    }

    pub(crate) fn absent_from_both(sources: &[Arc<Self>; 2], binding: u32) -> bool {
        sources.iter().all(|source| {
            source
                .declarations
                .complete_bindings()
                .is_some_and(|bindings| bindings.binary_search(&binding).is_err())
        })
    }
}

pub(crate) fn metadata_sources(
    state: &crate::model::DeviceState,
    vertex: &Arc<Vec<u32>>,
    fragment: &Arc<Vec<u32>>,
) -> Option<[Arc<SourceInfo>; 2]> {
    super::device_caches(state)
        .map(|caches| caches.with(|caches| caches.metadata_sources(vertex, fragment)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Progress {
    Pending,
    Ready,
    Unavailable,
}

#[derive(Debug)]
pub(crate) struct Metadata {
    pub vertex: Arc<Vec<u32>>,
    pub fragment: Arc<Vec<u32>>,
    pub bindings: Vec<BindingSig>,
    pub pass: PassKey,
    pub blend: Option<super::types::BlendKey>,
    pub secondary_blend: [Option<super::types::BlendKey>; super::caches::MAX_SECONDARY_ATTACH],
    pub color_write_mask: [super::ColorWriteMask; 1 + super::caches::MAX_SECONDARY_ATTACH],
    pub raster: reims_vgpu_vulkan::raster::GuestRasterState,
    pub topology: reims_vgpu_core::topology::PrimitiveType,
    pub viewport_count: u32,
}

pub(crate) fn preflight(state: &crate::model::DeviceState, input: &Metadata) -> Progress {
    let Some(caches) = super::device_caches(state) else {
        crate::observe::fail(
            "native_pipeline_preflight reason=device_caches_unreachable route=synchronous",
        );
        return Progress::Unavailable;
    };
    let result = caches.with(|caches| unsafe {
        let mut guard = super::lock_engine();
        let super::EngineState {
            owner,
            pools,
            counters,
            ..
        } = &mut *guard;
        let ctx = owner.ensure(counters)?;
        let count = u32::from(input.pass.secondary_count) + 1;
        if count > ctx.features.max_color_attachments {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::ColorAttachmentLimit {
                    requested: count as usize,
                    limit: ctx.features.max_color_attachments,
                },
            ));
        }
        let inputs = u8::BITS - input.pass.color_input.leading_zeros();
        if inputs > ctx.features.max_input_attachments {
            return Err(DrawError::Unsupported(
                super::reason::DrawReason::ColorInputAttachmentLimit {
                    requested: inputs,
                    limit: ctx.features.max_input_attachments,
                },
            ));
        }
        let (vert, vertex) =
            caches.get_or_create_shader_memoized(ctx, &input.vertex, counters, pools)?;
        let (frag, fragment) =
            caches.get_or_create_shader_memoized(ctx, &input.fragment, counters, pools)?;
        let admission = super::caches::storage_descriptors::StorageDescriptorAdmission::new(
            &vertex.declarations,
            &fragment.declarations,
        );
        let mut bindings = input.bindings.clone();
        super::caches::canonicalize_layout_bindings(&mut bindings)?;
        admission.filter_layout(&mut bindings);
        for (fragment, proof) in [
            (false, &vertex.declarations),
            (true, &fragment.declarations),
        ] {
            let Some(declared) = proof.complete_bindings() else {
                crate::observe::fail(
                    "native_pipeline_preflight reason=declarations_unproven route=synchronous",
                );
                return Ok(Progress::Unavailable);
            };
            if let Some(binding) = declared.iter().find(|binding| {
                !bindings
                    .iter()
                    .any(|candidate| candidate.binding == **binding)
            }) {
                return Err(DrawError::Unsupported(
                    super::reason::DrawReason::UsedBindingAbsentFromLayout {
                        binding: *binding,
                        fragment,
                    },
                ));
            }
        }
        let layout = caches.get_or_create_layout(ctx, &bindings, None, counters)?;
        let raster = reims_vgpu_vulkan::raster::plan(
            input.raster,
            reims_vgpu_vulkan::raster::RasterCell {
                depth_clamp: ctx.features.depth_clamp,
                fill_mode_non_solid: ctx.features.fill_mode_non_solid,
                dynamic_cull_and_winding: ctx.features.extended_dynamic_state,
                dynamic_polygon_mode: ctx.features.dynamic_polygon_mode,
                dynamic_depth_clamp: ctx.features.dynamic_depth_clamp,
            },
        )
        .map_err(|reason| DrawError::Unsupported(super::reason::DrawReason::Raster(reason)))?;
        let key = super::exec::pipeline_key(
            super::exec::PipelineColorState {
                blend: input.blend,
                secondary_blend: input.secondary_blend,
                color_write_mask: input.color_write_mask,
            },
            vert,
            frag,
            caches.intern_attrs(&[]),
            input.pass.compatibility(),
            reims_vgpu_vulkan::topology::key(
                input.topology,
                reims_vgpu_vulkan::topology::TopologyCell {
                    dynamic: ctx.features.extended_dynamic_state,
                    unrestricted: ctx.features.dynamic_primitive_topology_unrestricted,
                },
            ),
            raster.state,
            reims_vgpu_vulkan::depth_stencil::plan(
                &super::exec::depth_stencil_state(None),
                reims_vgpu_vulkan::depth_stencil::DepthStencilCell {
                    extended_dynamic_state: ctx.features.extended_dynamic_state,
                },
                false,
            )
            .state,
            input.viewport_count,
            layout.id,
            0,
        );
        Ok::<_, DrawError>(caches.precreate_pipeline(
            ctx,
            &key,
            &input.vertex,
            &input.fragment,
            input.pass,
        ))
    });
    match result {
        Ok(progress) => progress,
        Err(error) => {
            crate::observe::Emit::decline("native_pipeline_preflight", &error).fail();
            Progress::Unavailable
        }
    }
}

#[derive(Default)]
pub(super) struct Cache {
    device: usize,
    entries: HashMap<Key, Ticket<Outcome>>,
    adopted: HashMap<PipelineKey, Vec<Adopted<Outcome>>>,
    negatives: VecDeque<Key>,
}

struct Adopted<T> {
    source: [Arc<Vec<u32>>; 2],
    value: Arc<T>,
}

fn find_adopted<'a, T>(
    bucket: &'a [Adopted<T>],
    vertex: &Arc<Vec<u32>>,
    fragment: &Arc<Vec<u32>>,
) -> Option<&'a Arc<T>> {
    bucket
        .iter()
        .find(|value| same_sources(&value.source, &[vertex, fragment]))
        .map(|value| &value.value)
}

impl Cache {
    #[cfg(test)]
    pub(super) fn results(&self) -> Vec<Result<(), DrawError>> {
        self.entries
            .values()
            .filter_map(|ticket| match ticket.status() {
                Status::Ready(result) => {
                    Some(result.as_ref().as_ref().map(|_| ()).map_err(Clone::clone))
                }
                _ => None,
            })
            .collect()
    }

    fn current_device(&mut self, identity: usize) {
        if self.device != identity {
            self.clear();
            self.device = identity;
        }
    }

    pub(super) fn clear(&mut self) {
        for ticket in self.entries.values() {
            ticket.cancel();
        }
        self.entries.clear();
        self.adopted.clear();
        self.negatives.clear();
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "borrow the exact current device, shader sources and native layout identity"
    )]
    pub(super) fn ready(
        &mut self,
        device: usize,
        pipeline: &PipelineKey,
        attrs: &[AttrKey],
        bindings: &[BindingSig],
        push_constant: Option<(u32, u32)>,
        push_descriptors: bool,
        vertex: &Arc<Vec<u32>>,
        fragment: &Arc<Vec<u32>>,
    ) -> Option<Result<ash::vk::Pipeline, DrawError>> {
        self.current_device(device);
        if let Some(result) = self
            .adopted
            .get(pipeline)
            .and_then(|bucket| find_adopted(bucket, vertex, fragment))
        {
            return Some(
                result
                    .as_ref()
                    .as_ref()
                    .map(|value| value.pipeline)
                    .map_err(Clone::clone),
            );
        }
        if self.entries.is_empty() {
            return None;
        }
        let key = Key::new(
            pipeline,
            attrs,
            bindings,
            push_constant,
            push_descriptors,
            vertex,
            fragment,
        );
        if let Status::Ready(result) = self.entries.get(&key)?.status() {
            let output = result
                .as_ref()
                .as_ref()
                .map(|value| value.pipeline)
                .map_err(Clone::clone);
            if output.is_ok() {
                self.adopted
                    .entry(pipeline.clone())
                    .or_default()
                    .push(Adopted {
                        source: [vertex.clone(), fragment.clone()],
                        value: result,
                    });
            } else if output.as_ref().err().is_some_and(DrawError::out_of_memory) {
                self.entries.remove(&key);
            }
            return Some(output);
        }
        None
    }

    pub(super) fn prepare(&mut self, context: &DeviceContext, plan: Plan) -> Progress {
        self.current_device(context.native_identity());
        if let Some(ticket) = self.entries.get(&plan.key) {
            return match ticket.status() {
                Status::Pending => Progress::Pending,
                Status::Ready(result) => {
                    if result.is_err() && !self.negatives.contains(&plan.key) {
                        if self.negatives.len() == super::caches::NEGATIVE_CAP {
                            if let Some(old) = self.negatives.pop_front() {
                                self.entries.remove(&old);
                            }
                        }
                        self.negatives.push_back(plan.key);
                    }
                    Progress::Ready
                }
                Status::Cancelled | Status::Failed => Progress::Unavailable,
            };
        }
        let worker = match WORKER.get_or_init(|| Worker::new(MAX_COMPILES)) {
            Ok(worker) => worker,
            Err(error) => {
                crate::observe::fail(format!(
                    "native_pipeline_preflight reason=worker_unavailable error={error}"
                ));
                return Progress::Unavailable;
            }
        };
        let key = plan.key.clone();
        let context = context.compile_view();
        match worker.submit(move || compile(context, plan)) {
            Ok(ticket) => {
                self.entries.insert(key, ticket);
                crate::runtime::drain::note_store_route("native_pipeline_compile_queued");
                Progress::Pending
            }
            Err(super::compile_work::Admission::Full) => {
                crate::runtime::drain::note_store_route("native_pipeline_compile_backpressure");
                Progress::Pending
            }
            Err(super::compile_work::Admission::Stopped) => {
                crate::observe::fail("native_pipeline_preflight reason=worker_stopped");
                Progress::Unavailable
            }
        }
    }
}

impl Drop for Cache {
    fn drop(&mut self) {
        self.clear();
    }
}

fn compile(context: DeviceContext, plan: Plan) -> Outcome {
    let start = std::time::Instant::now();
    let counters = EngineCounters::default();
    let mut owners = NativeOwners {
        caches: ObjectCaches::new(),
        pools: ResourcePools::new(),
        context,
    };
    let NativeOwners {
        caches,
        pools,
        context,
    } = &mut owners;
    let pipeline = unsafe {
        let (vert, vertex) =
            caches.get_or_create_shader_memoized(context, &plan.vertex, &counters, pools)?;
        let (frag, fragment) =
            caches.get_or_create_shader_memoized(context, &plan.fragment, &counters, pools)?;
        let layout = caches.get_or_create_layout(
            context,
            &plan.key.bindings,
            plan.key.push_constant,
            &counters,
        )?;
        let pass = caches.get_or_create_pass(context, plan.pass, &counters, pools)?;
        let mut key = plan.key.pipeline;
        key.vert = vert;
        key.frag = frag;
        key.attrs = caches.intern_attrs(&plan.key.attrs);
        key.layout = layout.id;
        caches.get_or_create_pipeline(
            context,
            &key,
            None,
            vertex.module,
            &plan.vertex,
            fragment.module,
            &plan.fragment,
            layout.pipeline_layout,
            pass,
            &counters,
            pools,
        )?
    };
    crate::observe::off(format!(
        "native_pipeline_compile completed=true vert_words={} frag_words={} elapsed_us={} service_lock_held=false",
        plan.vertex.len(),
        plan.fragment.len(),
        start.elapsed().as_micros(),
    ));
    Ok(Compiled {
        pipeline,
        _owners: Mutex::new(owners),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint_levels() -> (usize, usize) {
        super::super::lock_engine()
            .owner
            .ctx
            .as_ref()
            .map_or((0, 0), DeviceContext::native_hint_levels)
    }

    #[test]
    fn forced_digest_collision_never_adopts_another_shader_source() {
        let pipeline = PipelineKey::collision_fixture();
        let vertex = Arc::new(vec![1, 2, 3]);
        let fragment = Arc::new(vec![4, 5, 6]);
        let changed = Arc::new(vec![4, 5, 7]);
        let first = Key::new(&pipeline, &[], &[], None, false, &vertex, &fragment);
        let collision = Key::new(&pipeline, &[], &[], None, false, &vertex, &changed);
        let hash = |key: &Key| {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hash);
            hash.finish()
        };
        assert_eq!(
            hash(&first),
            hash(&collision),
            "only cached digests/metadata feed hashing"
        );
        assert_ne!(
            first, collision,
            "exact source bytes must decide bucket membership"
        );
        let mut entries = HashMap::new();
        entries.insert(first, 11);
        entries.insert(collision, 22);
        assert_eq!(entries.len(), 2);
        let bucket = vec![Adopted {
            source: [vertex.clone(), fragment.clone()],
            value: Arc::new(11),
        }];
        assert!(find_adopted(&bucket, &vertex, &changed).is_none());
        assert!(find_adopted(&bucket, &changed, &fragment).is_none());
        assert_eq!(
            **find_adopted(&bucket, &vertex, &Arc::new((*fragment).clone())).unwrap(),
            11
        );
    }

    #[test]
    #[ignore = "requires the parent-owned exclusive native Vulkan lease"]
    fn native_precreated_pipeline_is_adopted_by_exact_draw_without_recompilation() {
        use super::super::*;
        use crate::backend::vulkan::sampled_shader::graphics_tests::shader;
        use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
        let state = DeviceState::new(DeviceId(0xac41), PAGE_SHIFT_ARM64E);
        test_reset_engine(&state);
        let vertex = Arc::new(shader(true, false, 0, 0, 1.0));
        let fragment = Arc::new(shader(false, false, 0, 0, 1.0));
        let input = Metadata {
            vertex: vertex.clone(),
            fragment: fragment.clone(),
            bindings: Vec::new(),
            pass: PassKey::single(Color0Load::Preserve, ash::vk::Format::R16G16B16A16_SFLOAT),
            blend: None,
            secondary_blend: [None; 7],
            color_write_mask: [ColorWriteMask::ALL; 8],
            raster: Default::default(),
            topology: reims_vgpu_core::topology::PrimitiveType::Triangle,
            viewport_count: 1,
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            match preflight(&state, &input) {
                Progress::Pending => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "native compiler did not finish"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Progress::Ready => break,
                Progress::Unavailable => panic!("authored exact metadata must be available"),
            }
        }
        let hints = hint_levels();
        assert_eq!(
            hints.0, 1,
            "the worker's program hint must be visible in the service's shared manager"
        );
        let target =
            pass_local::PassLocalTarget::new(4, 2, ash::vk::Format::R16G16B16A16_SFLOAT).unwrap();
        let request = DrawRequest {
            width: 4,
            height: 2,
            vertex_count: 3,
            skip_readback: true,
            vert_spirv: vertex,
            frag_spirv: fragment,
            target_identity: Some(target.identity().clone()),
            color_attachment: Some(ColorAttachmentState::new(
                ash::vk::Format::R16G16B16A16_SFLOAT,
                ColorClearValue::Float([0.0; 4]),
            )),
            ..Default::default()
        };
        let before = counter_snapshot();
        execute_draw_request(&state, &request).unwrap();
        assert_eq!(
            read_target_native(target.identity()).unwrap().pixels,
            0x3c00u16.to_le_bytes().repeat(32)
        );
        let after = counter_snapshot();
        assert_eq!(
            after.pipeline_precreated_hits - before.pipeline_precreated_hits,
            1
        );
        assert_eq!(after.pipeline_misses, before.pipeline_misses);
        drop(target);
        test_quiesce_ring();
        test_reset_engine(&state);
        assert_eq!(
            hint_levels(),
            hints,
            "retiring private PSO owners must not clear shared program hints"
        );
        assert!(
            device_caches(&state)
                .unwrap()
                .with(|caches| caches.precreated_results())
                .is_empty()
        );
        eprintln!(
            "native_async_adoption pixels=PASS native_pipeline_hits=1 additional_creates=0 shared_hint_programs={} shared_hint_bytes={} private_owners_retired=PASS",
            hints.0, hints.1
        );
    }

    #[test]
    #[ignore = "requires exclusive native Vulkan and private target/vulkan-native-precreation-replay inputs"]
    fn native_precreated_captured_creation_only_reuses_live_pipeline_and_shared_hint() {
        use super::super::*;
        use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E};
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/vulkan-native-precreation-replay");
        let read = |name| {
            let bytes = std::fs::read(directory.join(name)).expect("stage private replay module");
            assert!(bytes.len().is_multiple_of(4));
            Arc::new(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| u32::from_le_bytes(*bytes))
                    .collect(),
            )
        };
        let mut bindings: Vec<_> = (0..5)
            .chain([672, 673, 674, 675, 676, 677, 679, 680])
            .map(|binding| BindingSig {
                binding,
                ty: 7,
                stages: 17,
                count: 1,
            })
            .collect();
        bindings.extend((707..=718).map(|binding| BindingSig {
            binding,
            ty: 2,
            stages: 17,
            count: 1,
        }));
        bindings.extend((832..=835).map(|binding| BindingSig {
            binding,
            ty: 0,
            stages: 17,
            count: 1,
        }));
        bindings.extend((192..=193).map(|binding| BindingSig {
            binding,
            ty: 10,
            stages: 16,
            count: 1,
        }));
        let mut pass = PassKey::single(Color0Load::Preserve, ash::vk::Format::R16G16B16A16_SFLOAT);
        pass.secondary_count = 1;
        pass.secondary[0] = SecondaryAttachKey {
            format: ash::vk::Format::R16G16B16A16_SFLOAT,
            load: true,
        };
        pass.color_input = 3;
        let input = Metadata {
            vertex: read("vertex.spv"),
            fragment: read("fragment.spv"),
            bindings,
            pass,
            blend: None,
            secondary_blend: [None; 7],
            color_write_mask: [ColorWriteMask::ALL; 8],
            raster: Default::default(),
            topology: reims_vgpu_core::topology::PrimitiveType::Triangle,
            viewport_count: 1,
        };
        let state = DeviceState::new(DeviceId(0xac42), PAGE_SHIFT_ARM64E);
        test_reset_engine(&state);
        let before = counter_snapshot().queue_async_submits;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            match preflight(&state, &input) {
                Progress::Pending => {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Progress::Ready => break,
                Progress::Unavailable => panic!("captured exact metadata was not admitted"),
            }
        }
        assert_eq!(
            device_caches(&state)
                .unwrap()
                .with(|caches| caches.precreated_results()),
            vec![Ok(())]
        );
        let hints = hint_levels();
        assert_eq!(hints.0, 1);
        for _ in 0..2 {
            assert_eq!(preflight(&state, &input), Progress::Ready);
        }
        assert_eq!(
            device_caches(&state)
                .unwrap()
                .with(|caches| caches.precreated_results())
                .len(),
            1
        );
        assert_eq!(counter_snapshot().queue_async_submits, before);
        test_reset_engine(&state);
        assert_eq!(hint_levels(), hints);
        eprintln!(
            "native_async_captured creation=PASS cached_ready=PASS draws=0 submissions=0 shared_hint_programs={} shared_hint_bytes={} private_owners_retired=PASS",
            hints.0, hints.1
        );
    }
}
