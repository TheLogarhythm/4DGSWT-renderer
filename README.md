# GSWT Renderer

This is the official codebase for SIGGRAPH Asia 2025 paper: _GSWT: Gaussian Splatting Wang Tiles_ ([Project page](https://yunfan.zone/gswt_webpage/)). It only contains the GSWT renderer. For the GSWT constructor part, please visit the other repository: TBD.

## Introduction

This renderer is a _Gaussian Splatting Wang Tiles_ renderer that runs on the web, using [wgpu](https://github.com/gfx-rs/wgpu) and WebGPU as the rendering backend. It contains specific functionalities and optimizations tailored to GSWT, including _procedural tiling_, _selective merging_ and _LOD blending_. It takes a set of tiles produced by the GSWT constructor as input, and generates an infinitely expanding 3DGS terrian on-the-fly. The user can interact with the renderer and navigate the terrian in real time.

The original renderer discussed in the paper is developed with WebGL. This renderer is an improved version of the original one and might perform differently in certain cases (usually faster).

## Getting Started

The renderer is available online: [Demo](https://yunfan.zone/gswt_webpage/demo/). It requires a web browser that supports WebGPU (e.g., latest Chrome).

A set of official datasets can be found here: [Onedrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yzengbm_connect_ust_hk/IgB6p7U3s0FARoryHllz2jVuATc39DOnvkGZ9ieHkX_-hNw?e=Whq3DT)

To start the renderer:
1. Find a dataset and upload the zip file containing the set of tiles. After a moment of preprocessing (usually a few seconds), the config menu should show up.
2. Play with the config and click "Confirm". The renderer will switch to rendering stage and show the rendering menu.
3. Navigate the scene using **WASD** and hold **Space** to sprint. Use **IJKL** to look around.
4. There are some hotkeys to hide/unhide menus: **M** for the main rendering menu and **P** for the performance menu.
5. Click "Reconfig" to go back to config menu.
6. Reload the webpage to switch to another scene.

There are quite a few config options in this renderer. The default config is used in most experiments in the paper (usually with a skybox texture and a proxy texture). Below are some commonly used options: 

TBD

## Building locally

The renderer is written in Rust and targets WebAssembly (WASM). It is currently built with `rustc 1.92.0-nightly` and `wasm-pack 0.13.1`. They are required for building this project.

To build the project, run the following command:

```
wasm-pack build --target web
```

Then the package for release should be available in `./pkg`. It can be deployed on a web server together with `./index.html`. 

To test it locally, [sfz](https://github.com/weihanglo/sfz) is recommended. Run the following command to start a local server:

```
sfz -r --coi
```

## Dynamic 4DGSWT archives

The file picker accepts both the existing static six-LoD tile ZIPs and the
version-1, qualified version-2, or qualified version-3 dynamic archive produced by
`4DGSWT-constructor`. An archive with a
`manifest.json` is always validated as a manifest-based archive; malformed
dynamic data is reported as an error and is never silently loaded as static.
Motion-like binary members without a manifest are rejected.

A version-1 or version-2 dynamic ZIP has this exact member sequence:

```text
manifest.json
motion/basis.bin
tile0_lod0.ply
motion/tile0_lod0.bin
...
tile{tile_count-1}_lod5.ply
motion/tile{tile_count-1}_lod5.bin
```

Version 3 uses the same paired tile/member order but begins with three basis
members in translation, rotation, scale order:

```text
manifest.json
motion/basis_translation.bin
motion/basis_rotation.bin
motion/basis_scale.bin
tile0_lod0.ply
motion/tile0_lod0.bin
...
tile{tile_count-1}_lod5.ply
motion/tile{tile_count-1}_lod5.bin
```

The renderer preserves the coefficient row identity while it importance-sorts
each PLY and while it merges the unique tile/LoD rows. It samples the shared 75
frame, nine-dimensional motion basis with clamped non-periodic Catmull–Rom.
WebGPU then reconstructs position, local rotation, log scale, and covariance
for every unique Gaussian row. Static ZIP naming and rendering remain
supported without dynamic reconstruction resources, authoring controls, or a
spatial cache. Static scenes, and dynamic scenes without a painted field,
diagnostic overlay, or valid live brush preview, use the original two-bind-group
shader path. Field textures, sampling, and branches are enabled only while one
of those authored views is active. Hover-only preview binds two one-texel
placeholders and skips field sampling; the full spatial cache is allocated and
uploaded only after paint exists or a diagnostic overlay is selected.

Version 1 has implicit mask `7` (all motion channels enabled). Versions 2 and 3
store one authoritative mask per coefficient row; for example, mask `3`
preserves dynamic translation and rotation while using canonical scale. Version
3 independently reconstructs its translation, local rotation-vector, and
log-scale banks before applying that mask. At top-k 8, it stores 120 sparse
coefficient bytes per Gaussian plus the existing one-byte placement transform
ID and one-byte mask. The mask and all three coefficient blocks follow the same
PLY importance permutation and merge order.

### Motion runtime

The dynamic runtime separates archive contracts, CPU reference evaluation,
GPU reconstruction, stochastic playback, and spatial authoring state.
`motion_behavior` defines bounded channel gains, source ranges, loop policies
and deterministic playback parameters. Presets are direction-neutral; legacy
session direction fields remain readable but do not bias scene or local playback.

`motion_brush` stores ordered strokes in absolute Wang-world XY coordinates.
Its moving GPU cache uses 8, 16 or 32 texels per tile, with RGBA8 continuous
parameters and R8 controller assignments. `motion_controller_palette` and
`motion_spatial_variation` map authored regions to bounded independent timelines.
`motion_session` provides validated sidecar serialization, loading, stroke
history and controller-removal history without changing the source archive.
These modules are the runtime foundation for the separate authoring interface.

Authored evaluation is exact but sparse. The CPU recompiles controller
assignments only after the field content, cache layout, or controller palette
changes. After a membership-changing stroke, the Wang worker receives an
immutable occupancy snapshot and performs exact canonical-plus-instance lookup
during its normal sorting work. The committed effect can therefore appear one
or two frames after the stroke while the previous coherent result continues to
render.

Bit 31 of a sorted Gaussian index marks a compact authored occurrence. The
worker preserves draw order, selective merging, LoD transitions, map IDs, and
all unpainted indices; it changes only matching index words and supplies an
aligned per-draw tag flag. Stable world-space occurrences keep their compact
indices across camera reorderings. A full compact registry update is uploaded
only when membership or the set of encountered painted occurrences changes.
Camera-only sorts reuse it.

Only active controller slots are uploaded, and an unchanged paused frame skips
the authored compute pass entirely. Its dispatch cache is keyed by the compact
registry and controller/field state, not by camera sort revision. During
rendering, unaffected draws use the base shader. The authored shader is selected
only for a draw containing tagged indices, an active diagnostic overlay, or a
valid brush preview intersecting that draw. It decodes the tag directly and
loads either the global or compact authored texture; there is no dense
sort-aligned override buffer. If an erase, field rebuild, or authored-runtime
error is awaiting a matching worker result, tagged rows safely fall back through
their compact base-row mapping to the global motion texture.

Overlay Off is the representative setting for measuring ordinary authored
playback. Overlays and live preview deliberately enable extra field and
base-row diagnostic sampling.

The global and local compute shaders skip the second endpoint during ordinary
source playback. Fully painted local samples also skip inherited-state
reconstruction. The common authored case evaluates one state instead of four;
partially painted and transitioning samples retain the required blends. These
are reductions in reconstruction work, not a measured whole-renderer FPS
multiplier. Unchanged region documents skip controller remapping allocations.
Brush hover checks the compact contributing-tile list for each draw instead
of scanning per-Gaussian tile IDs each frame. Cached merges remap that list
together with their Gaussian IDs, preserving correct preview coverage.

### Stochastic motion preview

For a valid dynamic archive, the renderer derives a stochastic motion graph at
load time from a bounded, deterministic sample of LoD0 Gaussian rows. Its 74
nodes represent the complete translation, rotation, and log-scale source
segments defined by the archive's 75 samples. A jump is eligible only when all
nine position, velocity, and acceleration discontinuity metrics satisfy both a
robust twelve-source-step p95 limit and a forty-eight-source-step maximum-over-rows
ceiling. These are normalized equivalent-step metrics, not world-space units.
The `12/48` trial calibration was selected using the constructed shrub_sorrel
temporal-cut archive; it is not a guarantee of visual smoothness on every asset.

Targets must be at least six sample intervals from the source **exit** sample
(`source_segment + 1`), in either time direction. Each segment retains at most
three jumps: the best backward and forward candidates where available, followed
by the best remaining target at least six intervals from already selected targets.
No slot is forced when candidates fail qualification or separation. A segment
with no valid jump continues along the ordinary timeline and is not an error.
Gate limits, minimum interval separation and candidate budget are configured in
`MotionGraphSettings`; this pass does not add live gate-editing controls.

Qualified candidates receive a bounded smoothness-only probability preference
(at most 4:1). Scene and local controllers apply no directional bias. This
prevents large raw metric scores from making a valid alternative practically
unreachable. Analysis summaries distinguish all close pairs, temporally
qualified forward/backward candidates, and retained jumps with minimum/mean/maximum
departure-to-target gaps. Counts describe the full asset; active ranges, dwell and
endpoint policies can reduce the jumps actually played.

To reproduce a CPU-only gate sweep with the same archive loader, tile-height
normalization, motion evaluation and 2,048-row member-aware sample as the renderer,
run from this repository (the diagnostic is ignored by the normal test suite):

```powershell
$env:GSWT_GRAPH_ARCHIVE = 'C:\absolute\path\to\gswt.zip'
cargo test --offline --lib --target x86_64-pc-windows-msvc archive_gate_sweep -- --ignored --nocapture
```

The sweep measures the archive once and compares gates against the same candidate
metrics. It writes no assets and performs no training. Large archives may take
several minutes in a native debug build. Visual checking is still required,
especially for scale changes and painted motion-strength gains. No archive
regeneration is needed; reload the asset after rebuilding the renderer.

Behavior settings map variation and change frequency to deterministic
branch probability and minimum dwell. The selected graph sample reuses the
existing WebGPU deformation path, so graph playback does not add per-frame
graph analysis or a second motion reconstruction pass. If load-time analysis
fails, the warning is shown and ordinary dynamic playback remains available.
Scrubbing seeks to the requested authoritative source frame and resets the
current stochastic branch transition before playback resumes. Spatial strokes
remain renderer-local authoring state and the loaded GSWT archive remains
immutable.

As final containment, the compute shader rejects nonfinite dynamic positions
and covariance outside the factor-four binary16 packing range. It first retries
covariance using dynamic rotation with canonical scale, then copies fully
canonical covariance if needed. This defense does not replace normalized-source
qualification or constructor validation.

Run the native contract, playback, and GPU tests with:

```powershell
cargo test --target x86_64-pc-windows-msvc
```

The adapter-dependent parity test prints one explicit skip reason only when a
native WebGPU adapter is unavailable. The pinned browser build remains:

```powershell
wasm-pack build --target web
```

## Performance profiling

Press **P** during rendering to open the performance panel. The detailed
profiler reports rolling mean and p95 timings for the CPU frame, motion
preparation, Gaussian render encoding/uploads, worker sort/build work, GPU
global motion compute, GPU authored motion compute, and GPU Gaussian rendering.
It also reports the current archive/motion row counts, selected and rendered
splats, blending splats, active tile/LoD members, draw calls, bytes uploaded per
frame, compact registry and tagged-rendered occurrence counts, affected draws,
membership-request and registry revisions, worker tag time, compact registry
upload bytes, and whether the authored pass dispatched or remained idle.

GPU pass timings are enabled only when the selected WebGPU adapter exposes
timestamp queries. Results use a four-buffer asynchronous readback ring; a
busy ring drops that profiling sample instead of stalling rendering. The panel
shows an explicit unsupported or readback-error state when GPU timings are not
available. Profiling can be disabled in the GSWT or performance panel, and
**Reset Timer** clears both the legacy moving averages and detailed history.

## Credits

This renderer is derived from and heavily inspired by [Gauzilla](https://github.com/BladeTransformerLLC/gauzilla), under its [MIT License](https://github.com/BladeTransformerLLC/gauzilla?tab=MIT-1-ov-file). The egui rendering logic is derived from [this repository](https://github.com/kaphula/winit-egui-wgpu-template), under its [MIT License](https://github.com/kaphula/winit-egui-wgpu-template?tab=MIT-1-ov-file).

## BibTex

```
@inproceedings{Zeng:2025:gswt,
  author = {Zeng, Yunfan and Ma, Li and Sander, Pedro V.},
  title = {GSWT: Gaussian Splatting Wang Tiles},
  year = {2025},
  publisher = {Association for Computing Machinery},
  booktitle = {SIGGRAPH Asia 2025 Conference Papers},
  location = {Hong Kong, China},
  series = {SA '25}
}
```
