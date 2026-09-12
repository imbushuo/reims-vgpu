# qemu-build.sh

Builds the vendored `vendor/qemu` submodule into `vendor/qemu/build/qemu-system-<arch>`, linking the
thin device shim(s) to **`crates/reims-vgpu`**.

| Target | Typical host | Output | Pathway |
|--------|--------------|--------|---------|
| `aarch64` | Darwin / Apple Silicon | `qemu-system-aarch64` | arm64 macOS guest on macOS host (`vm/boot-arm64.sh`) |
| `x86_64` | Linux | `qemu-system-x86_64` | x86 macOS guest on Linux host (`vm/boot-x86.sh`) |

| Backend | Where it runs | Role |
|---------|---------------|------|
| `metal` | Apple Silicon | Metal-direct encode for the arm64 macOS guest |
| `vulkan` | Linux native ICD or Apple Silicon MoltenVK | Vulkan encode + `metal2vulkan` for x86/Linux and arm64/MoltenVK pathways |

Defaults: target by host OS (`aarch64` on Darwin, `x86_64` on Linux); backend defaults by host OS
(`metal` on Darwin, `vulkan` elsewhere) — **override for the pathway you are on**.

## What it does

`vendor/qemu` already carries the project patches — this script does **not** clone or patch, it
builds. Steps:

1. Populates the submodule if needed (`git submodule update --init vendor/qemu`).
2. Resolves `--target` (or `QEMU_TARGET`) and `--backend metal|vulkan` (or `REIMS_VGPU_BACKEND`).
3. Builds `crates/reims-vgpu` as a staticlib and links it into the device shim:
   - **Apple + metal:** real MTL frameworks + encode path.
   - **Non-Apple + metal:** rejected; Metal is Apple-only.
   - **vulkan:** ash-based in-crate engine (native Vulkan on Linux, MoltenVK on macOS).
4. **aarch64:** expects `CONFIG_VMAPPLE`, HVF/Cocoa configure, verifies `-M vmapple`.
5. **x86_64:** `x86_64-softmmu`, HVF/Cocoa off; lists PCI/sysbus device help as applicable.

Re-runs are idempotent (skips configure when the target/backend stamp matches). Switching target
or backend forces reconfigure. Patch record: `vendor/qemu-patches/`.

## GPU worker ownership

Both `reims-vgpu-mmio` and `reims-vgpu-pci` execute the Rust GPU drain on a dedicated,
RCU-registered worker without QEMU's Big QEMU Lock (BQL). `schedule_bh` wakes that
worker; `notify_actions` schedules a separate main-loop bottom half for completed
IRQ, display, and input actions. GPU completion waits must not move back into the
action bottom half.

Cursor-position doorbells use a shared Rust display-control owner, independent of
the render-state lock. Page changes, show/hide state, and cursor-action publication
are serialized there; glyph decoding remains on the GPU worker.
Popping a glyph action retains its immutable pixel payload and metadata separately
from render state, so the main-loop info/copy handoff cannot lose it to contention.

Guest-register and kernel-VA capture stays on the originating vCPU, and dirty-log
harvesting stays on the BQL-protected MMIO ingress. Reset quiesces the worker before
resetting Rust state or releasing packed guest-memory aliases. A shutdown that pauses
QEMU keeps the GPU alive for a subsequent reset. Terminal shutdown joins the worker
and stops the host window before destroying the backend and its notification target.
The shared worker's scheduling, reset, and teardown tests are in QEMU's
`tests/unit/test-reims-vgpu-worker.c`.

IOSurface-mapper register reads use independent authoritative atomics rather than
the render-state mutex. MMIO writes enter a FIFO admission queue; only the
metadata-only availability wait releases BQL. Capture, KVA access, mapping
application, consumer advancement, and IRQ publication still execute synchronously
on the originating vCPU with BQL held. Queued writes have priority over later GPU
tranches, and the final admission rearms a worker whose earlier wake was consumed.
The vCPU wait counters now include these BQL-free admission waits; they are not
measurements of how long BQL was blocked.

The IOSFC region uses lockless-I/O enrollment with explicit BQL scoping and
Rust-owned reentry refusal. Waiting tickets retain neither the backend nor its
HostOps context. Reset and destruction cancel tickets without waiting for vCPUs
to reacquire BQL; the C callback retains the QOM allocation through that interval.
This admission interface is ABI **v21**: rebuild the Rust static library and QEMU
shim together.

## Metal render passes and redraws

Compatible color draws retain ordinary and MRT attachments in the render-pass
owner and share a Metal command buffer and render encoder. Intermediate draws do
not read color pixels back to the CPU or upload them again for the next draw.
Staged inputs retain their own contents through GPU completion; keeping a Metal
object alive alone does not protect its contents from later CPU writes.

Queries, writable/native resource bindings, attachment changes, and dependencies
that need CPU-visible results complete outstanding work synchronously. Unknown or
overlapping input footprints conservatively materialize prior output.
Depth/stencil draws retain the synchronous fallback and carry intermediate
contents forward. These are semantic boundaries, not frame-time or draw-count
caps. Retained input memory therefore scales with the contents of a batch.

The `metal_submissions`, `metal_batch_encoder_reuse`,
`metal_batch_deferred_draws`, and `metal_seed_from_pass` census counters distinguish
actual batching from attachment reuse alone. Readback and dependency counters
identify remaining materialization costs.

Scanout keeps its initialized scratch buffer between captures, avoiding redundant
zero-filling at unchanged dimensions. Existing frame-push coalescing and display
refresh scheduling are unchanged.

## Run

```sh
# Explicit pathway builds
scripts/qemu-build/qemu-build.sh --target aarch64 --backend metal
scripts/qemu-build/qemu-build.sh --target aarch64 --backend vulkan
REIMS_VGPU_BACKEND=vulkan scripts/qemu-build/qemu-build.sh --target x86_64

# Point the matching boot script at the binary
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-aarch64 vm/boot-arm64.sh --device reims-vgpu-mmio --testing
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-x86_64 vm/boot-x86.sh --device reims-vgpu-pci --testing
```

### Requirements

- **Both:** cargo (`crates/reims-vgpu`), ninja, meson, pkg-config, glib, pixman.
- **aarch64 + metal:** macOS, Xcode CLT, HVF/Cocoa.
- **aarch64 + vulkan:** macOS, Xcode CLT, HVF/Cocoa, Vulkan loader, and MoltenVK ICD.
- **x86_64:** Linux QEMU build deps; KVM for boots.
