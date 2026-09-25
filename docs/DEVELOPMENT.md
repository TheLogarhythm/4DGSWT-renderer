# Developer reference

[Back to README](../README.md) · [Usage](USAGE.md)

## Build

Use the toolchain pinned in `rust-toolchain.toml` (`nightly-2025-10-13`, with
`rust-src` and the WASM target) and wasm-pack 0.13.1. Cargo defaults to
`wasm32-unknown-unknown`; native commands need an explicit target.

Put **Binaryen 132**'s `bin` first on PATH and verify `wasm-opt --version` before
`wasm-pack build --target web`. Confirm the build log uses that executable.
Bundled Binaryen 117 aborts in Precompute on this project.

Serve the generated `pkg/` with `index.html` using `sfz -r --coi`.
WebGPU requires HTTPS or loopback, and shared WASM memory requires COOP/COEP
cross-origin isolation. Rebuild and reload the browser after changing Rust/WGSL.

## Asset contract

The normal loader accepts static six-LoD ZIPs and dynamic versions 1, qualified 2
and qualified 3. A manifest always selects strict dynamic validation; malformed
dynamic data cannot fall back to static loading. Motion members without a manifest
are rejected. Limits: 1 GiB compressed, 2 GiB declared total decompressed,
256 MiB per member and 512 entries. Acceptance does not guarantee device memory.

Dynamic archives start with `manifest.json`, followed by `motion/basis.bin`
(v1/v2), or `motion/basis_translation.bin`, `motion/basis_rotation.bin`,
`motion/basis_scale.bin` (v3). Then come paired
`tile{tile}_lod{lod}.ply` and `motion/tile{tile}_lod{lod}.bin` members,
ordered LoD-major then tile-major across all six LoDs.
See the umbrella [archive schemas](../../../schemas/gswt_archive).

Geometry, canonical rows, coefficients, placement transforms and channel masks
must retain correspondence through PLY importance permutation and merging.
Motion uses 75 samples and clamped, non-periodic Catmull–Rom interpolation.
Version 1 implies mask 7; v2/v3 carry row masks (mask 3 disables dynamic scale).
Version 3 reconstructs independent translation, local rotation-vector and log-scale
banks. With top-k 8 it uses 120 sparse coefficient bytes plus one transform-ID and
one mask byte per Gaussian.

GPU reconstruction produces position and covariance. Nonfinite positions fall
back to canonical positions. Covariance exceeding factor-four binary16 packing
limits first retries with canonical scale, then canonical covariance. This
containment does not replace source qualification or constructor validation.

## Runtime invariants

- `state.rs` coordinates loading/configuration and frame submission.
  `wangtile.rs` handles placement, LoD transitions, selective merging and sorting;
  `renderer.rs` owns Gaussian GPU resources and draw pipelines.
- `worker.rs` coalesces camera requests and treats configuration as an ordering
  barrier. Only the worker waits; UI sends notify it and shutdown never joins on
  the browser main thread.
- Reconfiguration issues a monotonically numbered membership-clear request.
  Stale sorts must not reactivate old paint. Brush stamps stay anchored to the
  original world-space segment, independent of cache clipping.
- `sync_authoring_field` is shared by configuration and normal frames; failed
  uploads restore dirty regions. `motion_shader.rs` composes shared
  `motion_math.wgsl` once at pipeline creation while keeping global/local
  evaluation policies separate.

Authoring stores an ordered stroke document with a moving XY cache:
`RGBA8Unorm` continuous fields plus `R8Uint` controller assignment. A 97×97-tile
Default cache is 1552×1552 (about 11.5 MiB). Recenter reuses overlap and replays
new strips; quality changes, non-overlapping moves and Smooth history rebuild.
Field lookup uses canonical local positions plus occurrence offsets, not deformed
positions. Legacy session fields remain readable; UI Motion style/amount,
Spatial variety and Grouping map to MotionRegion/strength/variation/coherence.

Membership snapshots go to the worker; matching sorts tag compact authored
occurrences in index bit 31, preserving draw order, LoD/map metadata and unpainted
indices. Stable occurrences retain compact indices across camera sorts. Registry
uploads occur only when membership/encountered occurrences change; compute caching
tracks registry/controller/field state rather than camera sort revision.
Updates may appear one or two frames after a stroke. Pending or failed authored
results fall back through compact base-row mappings to global motion.

Static/unpainted draws use the base shader; tagged draws, diagnostics and intersecting
brush previews select the authored path. Hover uses placeholder field textures;
full field allocation is deferred until paint/overlay needs it. Only active local
controller slots are uploaded. Unchanged paused frames skip authored compute;
ordinary playback skips unused blend endpoints, retaining required transition blends.

### Motion graph and startup

Load-time graph analysis samples at most 2,048 deterministic, member-aware LoD0
rows across 74 source segments. Nine position/velocity/acceleration discontinuity
metrics must satisfy both p95 12-step and maximum 48-step gates. This shrub_sorrel
trial calibration is not a universal smoothness guarantee.

Targets are at least six intervals from the source exit sample. Retain up to three
qualified jumps, preferring backward/forward coverage and separated targets;
no eligible jump means ordinary playback. Smoothness preference is bounded at 4:1,
without directional bias. `MotionGraphSettings` owns qualification settings.
Analysis failure leaves ordinary motion available; runtime ranges/dwell may further
restrict played jumps. Graph analysis does not run each frame.

Startup computes raw depths before the required LoD-transition sort, avoiding a
discarded sort for each of 25 directions. Symmetric boundary metrics are measured
2,773 times for 5,329 directed candidates; qualification/ranking remain directional.
Upload borrows packed weights/IDs/masks, reuses chunk-sized canonical staging, and
allocates only the active coefficient format. Whole-load speedups require measurement.

### Water and profiling

Frame order is background/sky, proxy, water intersection/shading, Gaussian splats,
then UI (motion compute precedes drawing). Water computes one full-resolution
`RG32Float` hit per pixel, shared via exact texel loads by water and GS clipping.
Hits are independent of proxy depth, misses are overwritten, and resize recreates
the texture. Dry shaders exclude water clipping and fragment-depth output.

GS contact uses conditional Gaussian depth; fine ripples affect only shading.
Wave/ripple normals filter subpixel detail. Reflection uses a 256-pixel cubemap
with nine GGX roughness levels (about 4 MiB), rebuilt only when the sky source
changes or resources are missing. Scene reconfiguration reuses it. Set
`RenderData::skybox_changed` when replacing `skybox_rawtex`; the upload path clears
the flag after configuration. The prefilter pipeline survives sky replacements. It uses
linearized displayed sky color; HDR has already been tone mapped.
The geometric solver has a finite 16,384-step cap; grazing views, large coverage
and short waves can increase cost or reach it.

Profiler readback uses a four-buffer asynchronous ring; a busy ring drops samples
instead of blocking rendering. CPU preparation/submission and GPU timings have
different scopes; see [water benchmark evidence](water-performance.md).

Water preparation now reuses unchanged intersection/scattering textures. Fixed-camera,
fixed-phase benchmarks measure this cached case; advance the water clock or move the
camera when measuring animated preparation cost. Caustic offsets alone do not
invalidate the scattering volume.

## Validation

Run from the renderer directory; GPU tests should run serially:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc -- --test-threads=1
cargo fmt -- --check
cargo clippy --offline --all-targets --target x86_64-pc-windows-msvc
wasm-pack build --target web
```

Rust/Python parity tests locate umbrella `tests/fixtures/motion_parity_v{1,2,3}.json`
automatically in the normal workspace. For relocated checkouts, set
`FOURDGSWT_WORKSPACE_ROOT` to the umbrella root.
Native `src/test_support` handles GPU setup/readback; water tests require a GPU,
while optional motion tests report explicit skips. Finish tests before rebuilding
on Windows to avoid executable linker locks. Passing tests do not imply clean
Clippy output or successful browser interaction.

Targeted checks:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc --lib water_ -- --test-threads=1
```

For a CPU-only graph gate sweep, set `GSWT_GRAPH_ARCHIVE` to an absolute ZIP path:

```powershell
cargo test --offline --lib --target x86_64-pc-windows-msvc archive_gate_sweep -- --ignored --nocapture
```

This uses the renderer's loader/normalization/sample and writes no assets. Large
archives can take minutes; visual validation is still needed.

### Water review and benchmark

Set `WATER_REVIEW_ARCHIVE` to a ZIP containing `tile0_lod1.ply`,
`WATER_REVIEW_SKY` to an EXR and `WATER_REVIEW_DIR` to an output directory:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc --lib water_export_review_frames -- --ignored --nocapture --test-threads=1
```

The Flowers review fixture uses 29,279 splats at 768×768 and exports contact,
level and low-angle comparisons. Timings include clear/sky/water/GS, excluding
CPU work, readback and UI.

For the dense benchmark, set `WATER_PERF_ARCHIVE` to a shrub_sorrel ZIP and
`WATER_PERF_DIR` to an output directory:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc water_dense_shrub_benchmark -- --ignored --nocapture --test-threads=1
```

Run benchmarks separately from other GPU tests. Outputs include images and
`timings.json`; [measurement conditions and results](water-performance.md) are
maintained separately.

### Open checks and known issues

- Maintenance browser acceptance still needs a recorded paint/pan/reconfigure,
  motion-session save/load and reload/exit smoke check.
- Iceberg water needs scene-specific acceptance. Dense shrub performance does not
  establish every asset's visual correctness.
- Authored-motion performance gate: flesh_eyeballs, shrub_sorrel and anthurium at
  1920×1080, current default tile/LoD/merge settings, Default brush quality, overlay
  Off and one style-assignment stroke. Record FPS, CPU/GPU authored/GS p95,
  candidate/painted rows and affected draws. The 30 FPS desktop target remains
  unverified until browser measurements are recorded.
