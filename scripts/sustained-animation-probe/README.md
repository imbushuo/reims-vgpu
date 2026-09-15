# sustained-animation-probe

Drives the guest with a **sustained, full-rate** animation and captures the
device census for exactly that window.

```sh
scripts/sustained-animation-probe/sustained-animation-probe.sh /tmp/out 40
```

Takes `(outdir, seconds)`, writes `<outdir>/window.log`, and ends by running the
analysis over it — the same interface the other driven-boot probes take, so it
is interchangeable with them in a multi-boot harness.

## Why it exists

An undriven boot measures an idle device; `AGENTS.md` already says so. This
probe exists because a **bursty** driven boot measures the bursts' *gaps*, which
is a different error and reads as a device result rather than as an idle one.

A window-server probe that opens Mission Control and Launchpad spends ~2 s of
wall clock per round waiting for their animations, so whole seconds of it have
literally zero draws. Its `present_hz` median came out at **2.8 Hz** on a device
observed sustaining **78.8 Hz** (peak 92.2) under a frame-rate test page in the
same VM, minutes apart. Nothing in the bursty capture said it was idle: the
counters were self-consistent and the log well-formed.

The consequence is not a scale factor, it is a different ranking. Same guest
rail (macos-13), same build, same quiesced host:

| `chain_phase` share | bursty window-server probe | sustained animation |
|---|---|---|
| `store` | 10.3 % | **34.9 %** |
| `engine` | 49.0 % | 28.2 % |
| `sampled` | 18.5 % | 20.9 % |
| `pipeline` | 6.2 % | 8.5 % |
| per-chain total | 129 µs | 87 µs |
| drain worker duty | 0.00 median, 0.39 peak | **0.22 median, 0.88 peak** |

The last row is the one that decides what is worth fixing. Only the sustained
arm ever makes the drain worker the bottleneck, so it is the only arm on which
a per-draw CPU saving can become frames — which is why several CPU-side wins on
the bursty probe (a bounded pipeline cache, a 39 % cut in submissions, a
twentyfold cut in `stage_us`) each bought real microseconds and **zero** frames.

Run both before any "faster" claim. A change can help one and hurt the other,
and neither is the whole workload.

## The page is served by the host, on purpose

`anim.html` is served over QEMU's user-net gateway (`10.0.2.2:8123`), not fetched
from the internet. A probe whose workload can change under it cannot be A/B'd,
and the guest rails have no reason to have working DNS. Override the port with
`ANIM_PORT`.

Everything the page draws steps per *frame*, never per wall-clock millisecond,
so a slow boot and a fast boot draw identical content per frame number and
differ only in how many frames they complete. It loads both rails that matter:
eight layers the window server composites separately, and a canvas repainted
every frame so texture content is uploaded rather than only re-composited.

## Layer promotion is forced, because a hint made the probe bimodal

`will-change: transform` is advisory. Ten boots of one pinned binary — same
snapshot, same probe, same quiesced host — split into two tight clusters with
nothing in between:

| | promoted | collapsed |
|---|---|---|
| draws per presented frame | 417.9 – 429.0 | 267.7 – 268.5 |
| `present_hz` median | 39.1 – 41.7 | 49.0 – 50.6 |

Eight of ten landed in the first. Which cluster a boot drew was uncorrelated
with anything under test. The **24 %** `present_hz` gap is larger than any
device effect yet measured against this probe, so a sweep that mixes clusters
cannot see a real 17 % change and will credit the cluster to whichever arm drew
it. Within a cluster the counters reproduce to a fraction of a percent, which is
what makes three boots per arm enough.

So the page no longer hints. `.band` carries `backface-visibility: hidden` and
`tick()` writes a `translate3d`, which together take the decision out of the
compositor's hands. **Both halves are load-bearing**: the CSS rule cannot carry
the 3D transform, because `tick()` overwrites the inline `transform` every
frame, and a CSS-only fix would be silently discarded.

**It did not work, and the first three boots said it had.** The first three
boots after the change read 424.8, 427.7 and 427.6 draws per presented frame,
all promoted, `present_hz` spanning 1.5 %. The seventh collapsed: **265.8**.
Running tally since the change is 6 promoted, 1 collapsed — about 14 %, against
2 in 10 before. There is no evidence the change reduced anything.

The promotion edit stays. It is correct, costs nothing, and rules out one real
possibility. But the split is still here, so:

**Classify every boot before comparing two.** `drain_duty draws` over
`window_publish fresh`; the clusters are ~420 and ~268 with nothing in between,
so no threshold tuning is needed. Expect to discard roughly one boot in seven,
and plan a sweep with that headroom rather than assuming three boots per arm
will all land together.

The methodological lesson is worth more than the fix: a three-boot green run
against a one-in-five failure is not evidence — it comes up about half the time.
Write the probability down beside the reading, not the verdict.

Boots taken before this change are not comparable to boots taken after it.

## What it does not do

No host input lands inside the measured window — the page animates itself — so
nothing in the capture is the probe's own cost. It also cannot report a verdict
the way a drag probe can: there is no "the window never moved" check, because
there is no host-driven motion to check. Confirm the page is live from the
screenshot the surrounding harness takes, and from `present_hz` being nowhere
near zero.

## Window state and idle controls

Close Safari's address-field/Favorites popover before scoring the page. Its
translucent backdrop adds a different compositing workload. Verify the requested
windowed or full-screen state in a host-owned capture: sending a shortcut is not
confirmation, and pressing Escape after entering full screen can leave it again.

For long performance runs, record the **guest's** display-idle setting. Host
`caffeinate` does not prevent guest display sleep. A disposable guest overlay can
use `sudo pmset -a displaysleep 0` without changing the backing image.

The page's on-screen FPS counter measures `requestAnimationFrame`, not delivery
to the host display. On the arm64 Cocoa path, use display-FIFO `present_arrived`
progress and actual host captures; `window_publish` measures the separate
host-window path. Record any recovery input as a stall, not as a successful
uninterrupted interval.

Device present arrival is not proof of host delivery. On Cocoa,
`reims_vgpu_mmio_scanout` records a completed copy into the console surface;
`cocoa_frame_update` records a notification reaching the main thread, and
`cocoa_frame_draw` records the corresponding bitmap drawing work. Count distinct
nonzero draw sequences, not every draw callback: cursor or exposure redraws can
repeat the same sequence. `console_refresh` reports the listener interval.
These distinguish a renderer bottleneck from host coalescing; Cocoa drawing
still is not a measurement of physical panel scanout.

## Comparing Metal data-movement changes

Keep cold-start and warmed measurements separate, and record the VM's age and
actual window bounds for each interval. A later, warmed interval is not a
matched control for a freshly opened page. Keep compilation and other host
loads outside scored windows.

`REIMS_VGPU_METAL_GPU_WRITEBACK=off` disables mapped GPU Stores for a same-binary
comparison. Check `metal_gpu_writebacks` on the animation itself: a passing
standalone Store case does not prove the page uses that route.

`metal_packed_sampled_reuses` and `metal_packed_sampled_reuse_bytes` describe
native texture-upload reuse, not necessarily avoided guest reads. The
exact-byte comparison path still reads and converts the guest image. Likewise,
native input direct fills remove an intermediate copy, not every source read
or initialization write. Compare their byte counters and total draw cost.
Upload preparation can move between `engine_us` and `sampled_us`; a smaller
single phase is not by itself a reduction in total work.

The direct mapped-texture read-elision route has separate
`metal_packed_mapping_guest_bytes_avoided` and
`metal_packed_mapping_extra_audit_bytes` counters. Subtract the latter to obtain
net guest traffic saved: audits on cache misses, including post-fill audits,
are additional reads rather than savings. The ordinary full-staging fallback
still applies when the mapping or freshness proof is unavailable.

`REIMS_VGPU_METAL_INPUT_SNAPSHOT_REUSE=off` disables private native input
snapshot reuse for a same-binary comparison. Its reuse capability is confined
to a live decoded render pass; an unchanged dirty-harvest observation alone
does not establish freshness across completed commands. Unscoped requests
retain ordinary capture.

Hosts that cannot report guest writes immediately now use fresh native input
captures by default, without the snapshot cache's repeated page walks and
content audits. Allocation and known-zero reuse still apply. Setting the
snapshot switch to `on` cannot widen that host capability.

Mapped half-float samples use the existing byte-exact RGBA8 conversion kernel
on the render batch's command buffer. This preserves the CPU loader's clamping,
rounding, NaN, and signed-zero behavior without a CPU image round trip. Read
leases survive until actual GPU completion, and mapping retirement is checked
before submission. `metal_mapped_sample_gpu_conversions` records activation.
Non-importable samples still use the ordinary checked reader; that reader can
produce RGBA directly instead of converting through a full BGRA image.

The arm64 console remembers a refresh that found no completed frame. Delivery
of the next completed action batch satisfies that request without waiting for
another timer tick. It still coalesces completed frames, never re-reads live
guest pixels on the refresh clock, and does not make idle screenshots wait for
future guest work.

For passes with exclusively mapped GPU Stores and no CPU results, rendering
is submitted before writeback without an intervening CPU wait. Both commands
use the same Metal queue; writeback completion orders the producer, whose
status is checked before publication. The submitted render owner retains its
inputs and waits on error/drop paths. Queries, CPU readbacks, partial Stores,
and writable texture fallbacks retain synchronous completion.
`metal_render_store_pipelined` counts the eligible passes.

The HVF host offers a separate current-write observation over its existing
dirty bitmap. It reads bitmap words, not image bytes, and never clears bits or
reprotects memory from the rendering thread. A dirty set remains unavailable
until its generation has advanced and reprotection has completed. All sets
overlapping the shared dirty-page union are advanced before any bits are
cleared; unrelated clean sets remain readable during that work. A retained
address-space view prevents a topology change from passing as unchanged RAM.

This stronger observation permits planar snapshot reuse across render passes.
When it is unavailable, reuse retains the original command-local bounds;
changes of observation quality require a fresh capture. Unscoped compute reads
remain fresh. Other accelerators keep the original delayed-observation path.

Vulkan composite-planar sampling now uses that same resource-owned snapshot
contract. The decoded render pass supplies its revocable scope; immutable
expanded bytes are shared with the sampled-upload cache instead of reading and
converting both planes again for every draw. `vulkan_planar_sampled_reuses`,
`vulkan_planar_sampled_reuse_bytes`, `vulkan_planar_sampled_misses`, and
`vulkan_planar_sampled_unretained` report this route. Reuse bytes count source
plane bytes avoided; the existing gather-witness audit traffic is separate.
No new per-draw content validation is introduced. An absent or expired scope
uses fresh checked plane reads without starting snapshot observations.
The snapshot implementation is Vulkan-only; the Metal planar implementation
and its native fixtures remain unchanged.

The Vulkan representation remains native RGBA16Float or packed RGB10A2,
including the existing Q11 conversion and sampling rules. Do not apply Metal's
RGBA16Float-to-RGBA8 conversion to Vulkan's native-float path: its precision
contract is different. Mapping replacement, overlapping guest/host writes,
observation-quality changes, and resource retirement invalidate reuse; already
submitted uploads retain their own immutable bytes. The optional ABI23 current
observer is an HVF capability, not a Metal capability. Linux/KVM remains
command-local until its host can supply an equivalent current proof.

Persistent Vulkan mapped-surface sampled/LOAD copies also require current
currency, rather than treating an unchanged delayed generation as proof.
Unscoped copied GPU gathers require the same current observation; when it is
unavailable they gather fresh bytes and skip content audits that would validate
no reuse claim. Direct guest-allocation bindings and explicit in-pass attachment
chaining keep their ordered GPU paths. These rules depend on required mapped
Stores publishing authoritative guest backing before guest completion; they
must not substitute stale guest pages for an outstanding GPU-only result.

Vulkan guest-run input admission distinguishes stable borrowed runs from
retained imports. Nonstable generic GVA windows use checked RAMBlock references,
not transient raw aliases. Owned packed and mapping views carry an exact-page
host-allocation lease; recorded/native consumers retain that lease independently
of resource metadata. Metadata retirement revokes further native admission, and
the final CPU/native lease release requests one host unmap per acquisition.
Warm native-buffer binds renew read debt even when their layout is unchanged.

`buffer_guest_imports`, `buffer_guest_gathers` and `sampled_guest_imports` identify
different paths; buffer-import support does not grant linear image or attachment
support. The dedicated
`vulkan_gpu_owned_guest_runs_observe_mutation_and_outlive_metadata_retirement`
oracle requires actual native buffer binds, observes an unharvested CPU color
change, drops CPU metadata before recorded work completes, and checks exact
pixels plus one post-completion host unmap. Unrepresentable aliases retain the
checked CPU fallback rather than widening `map_pages_stable`.

Host-initialized external images are a separate, refused case:
`VK_EXT_external_memory` requires `UNDEFINED` at image creation, while a host
allocation lease provides no preserving image-layout ownership transfer.
`host_image_initial_contents_need_copy` and
`guest_image_initial_contents_refused` report that image-only decision before
creation. Native guest-buffer transfers, exact native-format sampled images,
attachment LOAD seeds and checked Store publication remain available. The
implementation neither discards nor rewrites the incoming guest allocation to
manufacture an image layout.

The owned-image GPU oracle covers BGRA8 and negative/HDR RGBA16Float input,
cross-page padded rows, warm CPU mutation, unchanged source/padding, the
image-specific refusal and native transfer counters, and final fence-safe
host unmap. A buffer-only import oracle does not cover this image contract.

Vulkan's sampled-upload cache preserves immutable `Arc<Vec<u8>>` identity.
A matching allocation and full image/view key skips hashing and byte comparison;
`sampled_bytes_arc_reuses` and `sampled_bytes_arc_reuse_bytes` count that path.
Different allocations use XXH3-128 to select candidates and retain exact byte
equality as the authority. A successful exact comparison adopts the incoming
immutable allocation for subsequent no-hash hits; `sampled_bytes_arc_adoptions`
counts that transition. Upload staging remains owned by its submitted slot
until completion, independently of this cache witness.
Copy-on-write, format/geometry/view changes and actual
content changes cannot reuse the wrong image. This is a process-local sampled
cache optimization, not a change to persisted shader or pipeline digests.

Native Vulkan compilation hints use a separate
`reims-vgpu-vk-pipeline-v2-<vendor>-<device>-<driver UUID>` directory.
An exact fragment program shares one native cache across graphics PSO variants;
compute programs share by module and entry point. The full runtime PSO keys
remain unchanged: these files are driver compilation hints, not permission to
reuse a pipeline with different attachments, bindings or state.
File buckets are confirmed against the complete program identity, a payload
checksum and the Vulkan device/cache header before loading.

Vulkan GVA colour0 LOAD seeds retain their native texel layout. Initial LOAD
capture and a failed resident-elision reseed use the checked descriptor/mip/
extent/GVA/pitch reader and carry immutable native bytes into a validated
buffer-to-image copy. The render-target resolver's base-texture mip (view base
plus attachment level, checked once) travels through both capture paths; the
native Vulkan allocation still uses mip 0. They do not pass RGBA16Float or packed native data through
an RGBA8 host cache or clamp it to unit-range bytes. Unsupported or mismatched
native capture is a named refusal, never an RGBA8 fallback. Native and legacy
seed sources are mutually exclusive and both are cleared from subsequent
encoder-record templates. `load_seed_color_native` and `load_seed_ok_native`
identify the native path. The Metal backend's existing seed default is unchanged.

Primary GVA Stores also retain the draw's exact resident identity. Successful
resource-debt deferral is unchanged. If deferral refuses (including the dirty
witness arming window), or the host requires synchronous publication, native
formats complete an exact native CPU readback before the existing fresh-GVA
bounded row writer, never the RGBA8 draw-readback fallback. The pre-render
page snapshot bounds that writer; it is not a GPU-direct destination licence.
Retained resource/view owners, allocation incarnation, live resource generation
and current ordered backing must still agree. Revocation, missing pages and
mismatched native layouts refuse visibly; successful synchronous publication
invalidates stale host pixel caches. No eager guest-memory GPU write is queued.
`gva_store_native` and `m2v_store_gva_native` identify this path, with
`writer=cpu_native` and `route=deferred_refused` or `route=synchronous` in the
per-Store record; `gva_eager_copied_native` counts completed CPU publications.

Program payloads are limited to 64 MiB, identity envelopes to 4 MiB, and settled
program files to 504 MiB / 2048 entries, reserving another 8 MiB for utility
pipelines. Atomic writes may temporarily add one entry. Eviction removes only
the oldest necessary entries; crossing a single-entry limit preserves existing
valid files instead of deleting the entire cache. The persistence queue is
bounded independently. At most 32 program cache owners / 128 MiB of accounted
serialized data are retained, plus active compilation leases; this is not a
measurement of the driver's internal allocation size. Evicting a hint cannot
destroy a cache still leased by a host compilation or retire an executable PSO.

`vk_pipeline_cache_entry_load`, `_save`, `_evict` and `_release` describe these
hints. Compare cold and warm launches in the same new namespace, keeping the
preserved control's old namespace untouched. This targets repeated cold
compilation; it does not establish bounded first-ever driver compilation or
end-to-end frame-rate parity.
Accepted nonzero cache data proves a valid load, not a native graphics cache hit:
the large compositor program still took over 13 seconds after a warm load in
run54. Compare the same complete PSO in fresh processes, and use valid pipeline
creation feedback when supported. A same-device tiny-kernel result or a shared
fragment program does not establish that cross-process graphics benefit.

Graphics storage-buffer and ordinary sampled-image descriptor admission follows
the finalized native modules, not sticky extra guest bindings or AIR access classifications. Each
native shader record retains an immutable declaration proof; the same
two-stage admission plan selects both descriptor-layout bindings and
the final shared allocated/push-descriptor write plan. A buffer or sampled-image slot absent from both complete
declaration sets is omitted. Every declared binding remains, even when
reflection calls it unused. Input attachments, storage-image descriptors,
samplers, vertex inputs, all resource staging/uploads, pins, read debt and
writeback lifetimes are unchanged.
Unsupported addressing/extensions, grouped decorations, unknown instructions
or malformed declaration coverage retain the legacy descriptors and emit a
typed `storage_descriptor_proof_*` diagnostic once per native module creation.
`shader_descriptor_proofs` counts those cold derivations; cache hits perform
no declaration scan or new content hash. This avoids PSO variants caused only
by proved-absent descriptors; `LayoutId` remains in the pipeline key. It does not bound first-use compilation
of a genuinely different shader.

The ignored creation-only replay oracle accepts private `vertex.spv` and
`fragment.spv` fixtures staged under `target/vulkan-pipeline-replay/`. Its
two-FP16-attachment, 31-binding contract checks an extra provided sampled
binding704 without inventing uniform contents or submitting any draw. The
fixtures stay outside version control; native diagnostics identify the actual
modules and PSO. Functional pixel, staging and declared-binding regressions
run separately on authored shaders in both descriptor transports.

Ordinary decoded render submissions can precreate exact graphics variants on
a single bounded application compiler (32 active/queued jobs). A native-pending
submission remains in the existing ready-position store before resource-table
consumption or any clear/draw/Store; unrelated ready positions can still run.
Queue saturation is backpressure, not dropped work. The native worker owns
immutable shader/state inputs, private pipeline/module/layout/pass objects and a device-lifetime
lease, never guest-RAM aliases or the shared device/engine service locks.
Results are keyed by complete pipeline state, descriptor-layout contents,
attributes and layout mode, with exact shader-Arc/byte equality after cached
digest hashing; draw-time lookup still requires that exact identity.
Compiler views share the existing per-native-device 32-program/128MiB hint
cache, rather than retaining a large hint blob per pipeline. Cache creation
and serialization have per-allocation synchronization. A synchronous fallback
never waits behind the compiler's hint-cache lock: it uses a null optional cache
with `native_cache_busy_uncached` accounting; busy optional saves are deferred.
Native diagnostics record the actual cache choice (`BusyUncached` when busy).
Device retirement cancels publication while the old native device survives
until its compiler-owned objects are released, without joining the compiler
while holding service locks.

The initial metadata-only profile excludes fixed-function vertex attributes,
depth/stencil, multisampling, current storage-interlock proofs, unnormalized
sampling, descriptor arrays, sparse attachment slots and shared-target placement
on stable-map hosts. Unavailable or incomplete metadata emits
`native_pipeline_preflight ... route=synchronous` and retains normal execution.
These boundaries do not downgrade formats or grant shader admission.
The attribute boundary uses the execution path's effective stride/format rule:
inactive declarations do not become fixed vertex input. Complete two-stage
descriptor proofs, cached with immutable module owners, exclude proven-absent
ordinary sampled bindings from compile-only metadata resolution; draw-time
staging, ownership and hazards are unchanged. Sampler metadata follows the
renderer's guest-supplied/static/default selection and serializer-object resolver.
Unchanged consecutive direct/indexed draw state is prepared once per retained
stream, including fallback outcomes; state records and topology changes invalidate
that reuse. Bounded preflight records carry task, pipeline generation and cached
VS/runtime-FS source digests; `native_preflight_eligible` means metadata collection
succeeded, not that native creation or draw-time adoption succeeded.
`pipeline_precreated_hits` identifies exact draw-time reuse;
`native_pipeline_compile_queued` and `native_pipeline_compile_backpressure`
describe the compiler mailbox. Compiler breadcrumbs and watchdog reporting are
separate from synchronous device-service calls.

For creation-miss diagnostics only, set `REIMS_VGPU_PIPELINE_DIAGNOSTICS=on`.
The default is off. An advertised `VK_EXT_pipeline_creation_feedback` is
enabled only for this diagnostic; no unsupported feature is assumed from the
physical device's API version. Pipeline lookup keys, feedback masks, waits and
renderer behavior are unchanged.

Join `vk_graphics_pso_begin`, `_component` and `_end` by `seq`; `table` identifies
the device-owned PSO cache instance. `pso` normalizes only diagnostic interner
identities and includes resolved layout/attribute contents. `declaration` also
includes the actual render-pass declaration and color layouts, or is
`unresolved` when that declaration cannot be recovered. Raw handles and interner
IDs are side-band fields, never inputs to the canonical fingerprints.
Component records expose key, vertex, layout and pass state; immutable sampler
state is explicitly `none` because these layouts do not declare immutable
samplers.

`source_vert`/`source_frag` and `driver_vert`/`driver_frag` use the engine's
existing `Digest128` spelling (`a`+`b` hex, followed by byte length). Actual
driver digests come from the shader-module keys after capability preparation;
source digests are memoized per immutable allocation and computed only at
native pipeline creation misses. `program`, `cache`, `origin`, `initial_bytes`
and `initial_xxh3` identify the native hint owner and its originally accepted
payload, not a newly serialized snapshot after every variant.

End records report host `elapsed_ns` around `vkCreateGraphicsPipelines`, result,
and flat `pipeline_*`, `vs_*`, `fs_*` feedback fields. `*_available=false` means
feedback was not enabled; `*_valid=false` never licenses a hit or duration
claim. Pipeline and stage durations overlap and must not be summed. A PSO cache
hit emits no creation record and performs no diagnostic source scan.

With the same opt-in diagnostic, an interlock-bearing fragment also emits
`interlock_backing_observation` before its unsupported-interlock refusal.
At most 16 observations per task/pipeline, across 16 pipelines, compare complete
recorded page unions as checked physical byte spans. Mixed page sizes must
overlap by byte range, not by equality of page bases. Records include mapping
generations, staged writeback identity, encoder scope, geometry and raster
arguments; unavailable ownership/subresource fields remain explicitly unknown.
These are observations, not ownership leases or renderer admission. No shader
interlock is removed and no capability is enabled by this diagnostic.

For eligible ordered fragment storage writes on a device without native pixel
interlock, a separate per-draw `SerialPrimitiveInterlock` strategy retains the
serialized resource owners and proves source/destination guest byte footprints
disjoint. Each filled, single-sample primitive runs in its own native pass;
subsequent passes load every attachment, and a global memory dependency orders
storage writes and subsequent reads/writes between passes. Coordinates, native
format/rounding, indices, instances, blending and write masks remain unchanged.
Only this admitted strategy can obtain the distinct lowered shader variant;
the ordinary shader-cache route still refuses the original unsupported module.
`serial_interlock_draws` and `serial_interlock_primitives` report activation.

This strategy's sampled-resource admission uses complete declaration proofs
bound to both finalized shader allocations. The fragment proof belongs to
the validated lowered `Program`, never to the unsupported original native
module. A provided ordinary sampled binding is ignored by this admission check
only when absent from both complete declaration sets. Its staging, owners and
read hazards remain unchanged; the separate finalized-module descriptor
admission above governs the eventual layout and writes. The retained shader index caches
proofs, including unproven results, under its existing bound; there is no
per-draw module walk. `serial_interlock_unused_sampled_draws` counts serial
draws admitted with such provided-but-undeclared images.

Unknown/overlapping guest backing, expired or mismatched ownership, mapper
destinations without stage-time generation, unmaterialized resident-only seeds,
queries, depth/stencil, multiview,
MSAA, non-filled rasterization, split-sensitive builtins, vertex side effects,
fragment buffer resources, declared sampled resources, incomplete/unknown
declaration proofs and unhandled shader effects
remain named refusals. This is not general raster-order-group emulation or an
attachment-only barrier substituted for storage ordering. It adds no CPU wait
or production validation-layer setting. Shader specialization and projection
remain memoized with immutable shader owners rather than rehashing guest pixels.

Vulkan descriptor-pool blocks budget input attachments as well as buffer,
sampled/storage-image and sampler descriptors. Framebuffer-fetch layouts need
that type even on drivers that happen to allocate from an untyped pool; growing
another block cannot repair a descriptor type omitted from every block.

Vulkan MRT draws can batch command-buffer submissions when every color/depth
attachment has a retained identity and the full attachment set, native formats,
extents, render area and sample count match. Vulkan render passes still end
after each MRT draw; this is submission batching, not render-pass fusion.
Encoder/clear/seed boundaries, attachment sampling, changed or retired
attachments, queries and CPU/writable-texture results retain their required
flushes. `mrt_batch_opens` and `mrt_batch_joins` report actual activation.

Batch attachment pins transfer to the submitting slot's fence cleanup. Failed
submission restores a first-write journal of all changed residents, including
sampled and seed sources, and a failed opener returns its descriptor set to the
allocating pool. The standalone `vulkan_gpu_mrt_batch_preserves_*` oracle checks
an actual submission join, changed uniforms/scissors, both render targets and
subsequent secondary sampling against exact pixels.

When the last pin makes a released resident collectable, its exact native
allocation enters lifetime retirement immediately. Fence cleanup and abort
drain these retirements through the existing graveyard, independently of the
bounded idle-cache trim. Live/pinned residents, backed authoritative content
and replaced native allocations remain protected. This prevents completed
memoryless MRT attachments from accumulating behind an eight-item maintenance
budget; it does not recycle images still referenced by GPU work.

For Vulkan host-window runs, use `host_window_cadence` (`window_ms`, `presents`)
and `window_publish` alongside guest present arrivals, not Cocoa draw traces.
Queue presentation is not a measurement of physical-panel scanout. Keep native
driver cold compilation and startup failures separate from warmed animation
measurements; a warm cache does not establish bounded cold-start latency.

Dismiss transient notifications before scoring and retain setup captures.
In particular, the guest's first-run Tips notification adds its own translucent
backdrop and a large set of non-mapped GPU readbacks. Its presence is a different
workload, not a slow interval of the unobstructed animation. Do not click an
empty menu bar and leave it in menu tracking: it blocks AppleScript replies.

## Apple M2 verification, 2026-09-14

Metal rail, macOS 15 guest, 4 vCPUs, 6 GiB guest RAM, 1920x1080 at 60 Hz.
The unmodified animation was measured for 90 seconds per interval after
60-second warm-ups, with QEMU visible and transient notifications dismissed
before scoring:

| Guest window | Device presents/s | Distinct Cocoa draws/s |
|---|---:|---:|
| Desktop, 1280x860 | 53.32 | 52.67 |
| Full-screen, 1920x1080 | 59.73 | 59.42 |
| Return to desktop, 1280x860 | 55.55 | 54.17 |

These are interval averages, not physical-panel scanout measurements. There
were no zero-progress intervals or recovery inputs in these scored windows.
An earlier independent clean run measured 50.58 host draws/s on the warmed
desktop and 57.51 in full-screen mode. Cold-start shader work is not included
in the warmed-rate claim.

Cold submission checks also cover task-definition ordering: a first EXEC
whose task is not yet active services pending sibling FIFO work before
capturing its command stream. Capture finishes before releasing the EXEC ring
head. This cold-path ordering step neither guesses task IDs nor adds per-draw
memory validation.
