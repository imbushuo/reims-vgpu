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
