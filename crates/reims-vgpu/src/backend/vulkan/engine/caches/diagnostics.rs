//! Opt-in, creation-miss-only PSO observations. None of these identities select
//! a pipeline or authorize a cache hit.

use super::*;
use crate::backend::vulkan::engine::context::native_cache::Diagnostic as CacheDiagnostic;
use std::sync::{atomic::AtomicU64, Arc};
use std::time::Instant;

pub(crate) fn requested() -> bool {
    let (state, value) = crate::config::read(crate::config::PIPELINE_DIAGNOSTICS);
    if state == crate::config::Switch::Unrecognized {
        crate::observe::fail(format!(
            "vk_pipeline_diagnostics reason=unrecognized_override value={value:?}"
        ));
    }
    enabled_from(state)
}

fn enabled_from(state: crate::config::Switch) -> bool {
    state == crate::config::Switch::On
}

pub(crate) const fn feedback_enabled(requested: bool, advertised: bool) -> bool {
    requested && advertised
}

pub(super) struct State {
    table: u64,
    source_digests: HashMap<usize, (Arc<Vec<u32>>, Digest128)>,
    #[cfg(test)]
    pub(super) observations: Vec<Observation>,
}

impl Default for State {
    fn default() -> Self {
        static TABLE: AtomicU64 = AtomicU64::new(1);
        Self {
            table: TABLE.fetch_add(1, Ordering::Relaxed),
            source_digests: HashMap::new(),
            #[cfg(test)]
            observations: Vec::new(),
        }
    }
}

impl State {
    fn source_digest_with(
        &mut self,
        words: &Arc<Vec<u32>>,
        hash: impl FnOnce(&[u32]) -> Digest128,
    ) -> Digest128 {
        let address = Arc::as_ptr(words) as usize;
        if let Some((_, digest)) = self.source_digests.get(&address) {
            return *digest;
        }
        let digest = hash(words);
        if self.source_digests.len() >= SHADER_DIGEST_ENTRIES {
            self.source_digests.clear();
        }
        self.source_digests
            .insert(address, (Arc::clone(words), digest));
        digest
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Feedback {
    flags: u32,
    valid: bool,
    application_hit: bool,
    duration_ns: Option<u64>,
}

impl Feedback {
    fn from_native(value: vk::PipelineCreationFeedback) -> Self {
        let valid = value
            .flags
            .contains(vk::PipelineCreationFeedbackFlags::VALID);
        Self {
            flags: value.flags.as_raw(),
            valid,
            application_hit: valid
                && value
                    .flags
                    .contains(vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT),
            duration_ns: valid.then_some(value.duration),
        }
    }
}

fn feedback_fields(prefix: &str, value: Option<Feedback>) -> String {
    match value {
        Some(value) => format!(
            "{prefix}_available=true {prefix}_flags={} {prefix}_valid={} {prefix}_app_hit={} {prefix}_ns={}",
            value.flags, value.valid, value.application_hit,
            value.duration_ns.map_or_else(|| "invalid".into(), |duration| duration.to_string()),
        ),
        None => format!("{prefix}_available=false {prefix}_flags=0 {prefix}_valid=false {prefix}_app_hit=false {prefix}_ns=unavailable"),
    }
}

#[derive(Clone, Debug)]
pub(super) struct Observation {
    pub sequence: u64,
    pub table: u64,
    pub fingerprint: Digest128,
    pub declaration: Option<Digest128>,
    pub source: [Digest128; 2],
    pub driver: [Digest128; 2],
    pub cache: CacheDiagnostic,
    pub elapsed_ns: u128,
    pub result: vk::Result,
    pub feedback: Option<Feedback>,
    pub stages: Option<[Feedback; 2]>,
}

fn digest_text(value: Digest128) -> String {
    format!("{:016x}{:016x}:{}", value.a, value.b, value.len)
}

fn fingerprint_text(value: Digest128) -> String {
    format!("{:016x}{:016x}", value.a, value.b)
}

#[allow(clippy::too_many_arguments)]
fn canonical_fingerprint(
    key: &PipelineKey,
    attrs: &[AttrKey],
    formats: &[vk::Format],
    layout: &LayoutEntry,
    create_flags: u32,
    blend_flags: u32,
    dynamic: &[vk::DynamicState],
) -> Digest128 {
    let mut normalized = key.clone();
    normalized.attrs = AttrsId(0);
    normalized.layout = LayoutId(0);
    Digest128::of_items(
        &(
            normalized,
            formats,
            &layout.bindings,
            layout.push_constant,
            layout.push_descriptors,
            create_flags,
            blend_flags,
            dynamic,
        ),
        attrs,
    )
}

pub(super) struct Trace {
    sequence: u64,
    table: u64,
    fingerprint: Digest128,
    declaration: Option<Digest128>,
    source: [Digest128; 2],
    driver: [Digest128; 2],
    cache: CacheDiagnostic,
    key: String,
    vertex: String,
    layout: String,
    pass: String,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    caches: &mut ObjectCaches,
    key: &PipelineKey,
    vertex_source: &Arc<Vec<u32>>,
    fragment_source: &Arc<Vec<u32>>,
    attrs: &[AttrKey],
    formats: &[vk::Format],
    render_pass: vk::RenderPass,
    create_flags: u32,
    blend_flags: u32,
    dynamic: &[vk::DynamicState],
    cache: CacheDiagnostic,
) -> Trace {
    use ash::vk::Handle as _;
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    let state = caches.diagnostics.get_or_insert_with(State::default);
    let table = state.table;
    let source = [
        state.source_digest_with(vertex_source, Digest128::of_u32_words),
        state.source_digest_with(fragment_source, Digest128::of_u32_words),
    ];
    let layout = &caches.layouts.entries[key.layout.0 as usize];
    let fingerprint = canonical_fingerprint(
        key,
        attrs,
        formats,
        layout,
        create_flags,
        blend_flags,
        dynamic,
    );
    let actual_pass = caches
        .passes
        .map
        .iter()
        .find_map(|(key, handle)| (*handle == render_pass).then_some(key));
    let layouts = actual_pass.map(|pass| {
        (0..=pass.secondary_count as usize)
            .map(|slot| (pass.color_layout(slot), pass.color_final_layout(slot)))
            .collect::<Vec<_>>()
    });
    let declaration = actual_pass
        .zip(layouts.as_ref())
        .map(|(pass, layouts)| Digest128::of_items(&(fingerprint, pass), layouts));
    let mut normalized = key.clone();
    normalized.attrs = AttrsId(0);
    normalized.layout = LayoutId(0);
    Trace {
        sequence: SEQUENCE.fetch_add(1, Ordering::Relaxed),
        table,
        fingerprint,
        declaration,
        source,
        driver: [key.vert, key.frag],
        cache,
        key: format!("state={normalized:?} intern_attrs={} intern_layout={} create_flags={create_flags:#x} blend_flags={blend_flags:#x} dynamic={dynamic:?}", key.attrs.0, key.layout.0),
        vertex: format!("attrs={attrs:?} driver_formats={formats:?}"),
        layout: format!(
            "bindings={:?} push_constant={:?} push_descriptors={} immutable_samplers=none native_dsl={:#x} native_layout={:#x}",
            layout.bindings, layout.push_constant, layout.push_descriptors,
            layout.dsl.as_raw(), layout.pipeline_layout.as_raw(),
        ),
        pass: match actual_pass {
            Some(pass) => format!(
                "requested_compat={:?} actual_declaration={pass:?} color_layouts={layouts:?} native_render_pass={:#x}",
                key.pass, render_pass.as_raw(),
            ),
            None => format!("requested_compat={:?} actual_declaration=unresolved", key.pass),
        },
    }
}

impl Trace {
    pub(super) fn actual_cache(&mut self, cache: CacheDiagnostic) {
        self.cache = cache;
    }

    pub(super) fn begin(&self) -> Instant {
        let payload = self.cache.initial_payload.map_or_else(
            || "initial_bytes=0 initial_xxh3=none".into(),
            |payload| {
                format!(
                    "initial_bytes={} initial_xxh3={:032x}",
                    payload.bytes, payload.checksum
                )
            },
        );
        crate::observe::off(format!(
            "vk_graphics_pso_begin seq={} table={} pso={} declaration={} source_vert={} source_frag={} driver_vert={} driver_frag={} program={:032x} cache={:#x} origin={:?} {payload}",
            self.sequence, self.table, fingerprint_text(self.fingerprint),
            self.declaration.map_or_else(|| "unresolved".into(), fingerprint_text),
            digest_text(self.source[0]), digest_text(self.source[1]),
            digest_text(self.driver[0]), digest_text(self.driver[1]),
            self.cache.program, self.cache.handle, self.cache.origin,
        ));
        for (part, fields) in [
            ("key", &self.key),
            ("vertex", &self.vertex),
            ("layout", &self.layout),
            ("pass", &self.pass),
        ] {
            crate::observe::off(format!(
                "vk_graphics_pso_component seq={} part={part} {fields}",
                self.sequence,
            ));
        }
        Instant::now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    fn key() -> PipelineKey {
        PipelineKey {
            vert: Digest128::of_u32_words(&[1, 2]),
            frag: Digest128::of_u32_words(&[3, 4]),
            attrs: AttrsId(7),
            layout: LayoutId(9),
            topology: reims_vgpu_vulkan::topology::TopologyKey::Exact(
                reims_vgpu_core::topology::PrimitiveType::Triangle,
            ),
            blend: None,
            secondary_blend: [None; MAX_SECONDARY_ATTACH],
            color_write_mask: [Default::default(); 1 + MAX_SECONDARY_ATTACH],
            pass: PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM).compatibility(),
            feedback_colors: 0,
            raster: reims_vgpu_vulkan::raster::RasterizationState {
                depth_clamp_enable: false,
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                depth_bias_enable: true,
                dynamic: reims_vgpu_vulkan::raster::RasterDynamic::NONE,
            },
            depth_stencil: reims_vgpu_vulkan::depth_stencil::DepthStencilPlan {
                depth_test_enable: false,
                depth_write_enable: false,
                depth_compare_op: vk::CompareOp::ALWAYS,
                depth_bounds_test_enable: false,
                stencil_test_enable: false,
                front: reims_vgpu_vulkan::depth_stencil::FacePlan::PASS_THROUGH,
                back: reims_vgpu_vulkan::depth_stencil::FacePlan::PASS_THROUGH,
                dynamic: false,
            },
            viewport_slots: 1,
        }
    }

    fn layout() -> LayoutEntry {
        LayoutEntry {
            bindings: vec![BindingSig {
                binding: 3,
                ty: vk::DescriptorType::STORAGE_BUFFER.as_raw() as u32,
                stages: vk::ShaderStageFlags::VERTEX.as_raw(),
                count: 1,
            }],
            push_constant: None,
            push_descriptors: false,
            dsl: vk::DescriptorSetLayout::from_raw(11),
            pipeline_layout: vk::PipelineLayout::from_raw(12),
        }
    }

    fn attrs() -> Vec<AttrKey> {
        vec![AttrKey {
            location: 0,
            binding: 1,
            format: VertexAttributeFormat::Float4,
            offset: 0,
            stride: 16,
            step_function: VertexStepFunction::PerVertex,
            step_rate: 1,
        }]
    }

    fn fingerprint(key: &PipelineKey, layout: &LayoutEntry, attrs: &[AttrKey]) -> Digest128 {
        canonical_fingerprint(
            key,
            attrs,
            &[vk::Format::R32G32B32A32_SFLOAT],
            layout,
            0,
            0,
            &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
        )
    }

    #[test]
    fn pipeline_diagnostics_are_opt_in_and_feedback_requires_the_extension() {
        use crate::config::Switch;
        for state in [Switch::Unset, Switch::Off, Switch::Unrecognized] {
            assert!(!enabled_from(state));
        }
        assert!(enabled_from(Switch::On));
        assert!(!feedback_enabled(false, true));
        assert!(!feedback_enabled(true, false));
        assert!(feedback_enabled(true, true));
    }

    #[test]
    fn diagnostic_identity_ignores_interner_and_native_handle_values_only() {
        let original = key();
        let original_layout = layout();
        let attributes = attrs();
        let expected = fingerprint(&original, &original_layout, &attributes);
        let mut moved = original.clone();
        moved.attrs = AttrsId(500);
        moved.layout = LayoutId(900);
        let mut moved_layout = layout();
        moved_layout.dsl = vk::DescriptorSetLayout::from_raw(9001);
        moved_layout.pipeline_layout = vk::PipelineLayout::from_raw(9002);
        assert_eq!(expected, fingerprint(&moved, &moved_layout, &attributes));
        assert_ne!(original, moved, "runtime lookup identity is not rewritten");
    }

    #[test]
    fn diagnostic_identity_retains_resolved_layout_and_vertex_contents() {
        let key = key();
        let attributes = attrs();
        let expected = fingerprint(&key, &layout(), &attributes);
        for change in 0..5 {
            let mut changed = layout();
            match change {
                0 => changed.bindings[0].binding += 1,
                1 => changed.bindings[0].count += 1,
                2 => changed.bindings[0].stages = vk::ShaderStageFlags::FRAGMENT.as_raw(),
                3 => changed.push_constant = Some((0, 16)),
                _ => changed.push_descriptors = true,
            }
            assert_ne!(expected, fingerprint(&key, &changed, &attributes));
        }
        let mut changed = attributes.clone();
        changed[0].stride = 32;
        assert_ne!(expected, fingerprint(&key, &layout(), &changed));
        assert_ne!(
            expected,
            canonical_fingerprint(
                &key,
                &attributes,
                &[vk::Format::R16G16B16A16_SFLOAT],
                &layout(),
                0,
                0,
                &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
            )
        );
    }

    #[test]
    fn diagnostic_identity_preserves_shader_pass_raster_and_feedback_distinctions() {
        let original = key();
        let expected = fingerprint(&original, &layout(), &attrs());
        for change in 0..8 {
            let mut changed = original.clone();
            match change {
                0 => changed.vert.a ^= 1,
                1 => changed.frag.b ^= 1,
                2 => changed.pass.0.color0_format = vk::Format::R16G16B16A16_SFLOAT,
                3 => changed.pass.0.sample_count = 4,
                4 => changed.raster.cull_mode = vk::CullModeFlags::BACK,
                5 => changed.depth_stencil.depth_write_enable = true,
                6 => changed.viewport_slots = 2,
                _ => changed.feedback_colors = 1,
            }
            assert_ne!(expected, fingerprint(&changed, &layout(), &attrs()));
        }
        let mut one = original.clone();
        one.feedback_colors = 1;
        let mut two = original;
        two.feedback_colors = 2;
        assert_ne!(
            fingerprint(&one, &layout(), &attrs()),
            fingerprint(&two, &layout(), &attrs()),
            "over-specific runtime terms remain observable, not normalized away"
        );
    }

    #[test]
    fn immutable_source_digest_is_memoized_without_rehashing_or_pointer_reuse() {
        let mut state = State::default();
        let mut source = Arc::new(vec![1, 2, 3]);
        let mut calls = 0;
        let first = state.source_digest_with(&source, |words| {
            calls += 1;
            Digest128::of_u32_words(words)
        });
        assert_eq!(
            first,
            state.source_digest_with(&source, |_| panic!("rehash"))
        );
        Arc::make_mut(&mut source)[0] = 4;
        let changed = state.source_digest_with(&source, |words| {
            calls += 1;
            Digest128::of_u32_words(words)
        });
        assert_ne!(first, changed);
        assert_eq!(calls, 2);
    }

    #[test]
    fn feedback_without_valid_bit_does_not_claim_a_hit_or_duration() {
        let invalid = Feedback::from_native(vk::PipelineCreationFeedback {
            flags: vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT,
            duration: 999,
        });
        assert!(!invalid.valid && !invalid.application_hit);
        assert_eq!(invalid.duration_ns, None);
        let valid = Feedback::from_native(vk::PipelineCreationFeedback {
            flags: vk::PipelineCreationFeedbackFlags::VALID
                | vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT,
            duration: 123,
        });
        assert!(valid.valid && valid.application_hit);
        assert_eq!(valid.duration_ns, Some(123));
    }

    #[test]
    fn diagnostics_reuse_actual_driver_digests_and_resolve_native_pass_handles() {
        use crate::backend::vulkan::engine::context::native_cache::Origin;
        let mut caches = ObjectCaches::new();
        let attributes = attrs();
        let entry = layout();
        let resolved = caches.layouts.insert(
            &entry.bindings,
            entry.push_constant,
            entry.dsl,
            entry.pipeline_layout,
            false,
        );
        let mut key = key();
        key.attrs = AttrsId(caches.attr_sets.intern(&attributes));
        key.layout = resolved.id;
        let pass = PassKey::single(Color0Load::Clear, vk::Format::B8G8R8A8_UNORM);
        key.pass = pass.compatibility();
        let handle = vk::RenderPass::from_raw(100);
        caches.passes.insert(pass, handle);
        let vertex = Arc::new(vec![11, 22]);
        let fragment = Arc::new(vec![33, 44]);
        let cache = CacheDiagnostic {
            program: 1,
            handle: 2,
            origin: Origin::Empty,
            initial_payload: None,
        };
        let first = prepare(
            &mut caches,
            &key,
            &vertex,
            &fragment,
            &attributes,
            &[vk::Format::R32G32B32A32_SFLOAT],
            handle,
            0,
            0,
            &[],
            cache,
        );
        assert_eq!(first.driver, [key.vert, key.frag]);
        assert_eq!(
            first.source,
            [
                Digest128::of_u32_words(&vertex),
                Digest128::of_u32_words(&fragment),
            ]
        );
        assert_ne!(first.source, first.driver);
        assert!(first.declaration.is_some());
        let moved = vk::RenderPass::from_raw(9000);
        caches.passes.insert(pass, moved);
        let second = prepare(
            &mut caches,
            &key,
            &vertex,
            &fragment,
            &attributes,
            &[vk::Format::R32G32B32A32_SFLOAT],
            moved,
            0,
            0,
            &[],
            CacheDiagnostic {
                handle: 99,
                ..cache
            },
        );
        assert_eq!(first.fingerprint, second.fingerprint);
        assert_eq!(first.declaration, second.declaration);
        let missing = prepare(
            &mut caches,
            &key,
            &vertex,
            &fragment,
            &attributes,
            &[vk::Format::R32G32B32A32_SFLOAT],
            handle,
            0,
            0,
            &[],
            cache,
        );
        assert!(missing.declaration.is_none());
        assert!(missing.pass.contains("actual_declaration=unresolved"));
    }

    #[test]
    fn diagnostic_identity_includes_native_create_blend_and_dynamic_flags() {
        let key = key();
        let attributes = attrs();
        let layout = layout();
        let base = canonical_fingerprint(&key, &attributes, &[], &layout, 0, 0, &[]);
        assert_ne!(
            base,
            canonical_fingerprint(
                &key,
                &attributes,
                &[],
                &layout,
                vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT.as_raw(),
                0,
                &[],
            )
        );
        assert_ne!(
            base,
            canonical_fingerprint(&key, &attributes, &[], &layout, 0, 1, &[])
        );
        assert_ne!(
            base,
            canonical_fingerprint(
                &key,
                &attributes,
                &[],
                &layout,
                0,
                0,
                &[vk::DynamicState::VIEWPORT],
            )
        );
    }
}

impl Trace {
    pub(super) fn finish(
        self,
        started: Instant,
        result: vk::Result,
        supported: bool,
        pipeline: vk::PipelineCreationFeedback,
        stages: [vk::PipelineCreationFeedback; 2],
    ) -> Observation {
        let observation = Observation {
            sequence: self.sequence,
            table: self.table,
            fingerprint: self.fingerprint,
            declaration: self.declaration,
            source: self.source,
            driver: self.driver,
            cache: self.cache,
            elapsed_ns: started.elapsed().as_nanos(),
            result,
            feedback: supported.then(|| Feedback::from_native(pipeline)),
            stages: supported.then(|| stages.map(Feedback::from_native)),
        };
        let pipeline_feedback = feedback_fields("pipeline", observation.feedback);
        let vertex_feedback = feedback_fields("vs", observation.stages.map(|stages| stages[0]));
        let fragment_feedback = feedback_fields("fs", observation.stages.map(|stages| stages[1]));
        crate::observe::off(format!(
            "vk_graphics_pso_end seq={} table={} pso={} declaration={} elapsed_ns={} result={:?} program={:032x} cache={:#x} origin={:?} source_vert={} source_frag={} driver_vert={} driver_frag={} {pipeline_feedback} {vertex_feedback} {fragment_feedback}",
            observation.sequence, observation.table, fingerprint_text(observation.fingerprint),
            observation.declaration.map_or_else(|| "unresolved".into(), fingerprint_text),
            observation.elapsed_ns, observation.result,
            observation.cache.program, observation.cache.handle, observation.cache.origin,
            digest_text(observation.source[0]), digest_text(observation.source[1]),
            digest_text(observation.driver[0]), digest_text(observation.driver[1]),
        ));
        observation
    }
}

#[cfg(test)]
mod gpu;
