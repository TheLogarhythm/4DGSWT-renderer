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

After a dynamic asset is configured, **Open Motion…** in the GSWT panel opens
one **Motion** window. It opens automatically for a newly loaded dynamic
archive. Play/pause, restart, motion enable, scene-time scrubbing, and session
save/load appear once in its shared header. Local styles retain independent
timelines; the scene-time scrubber is not a local-controller scrubber.

Choose **Scene default** to edit base motion for unpainted areas, or **Local
styles** to edit a named local controller. Both use the same editor: **Motion
amount**, **Speed**, and **Playback variation**. Motion amount scales all motion
channels; it is not a brush radius or a whole-scene multiplier over local
styles. Raising Playback variation enables stochastic branching, including
from Subtle; zero disables it. Variation still requires eligible graph edges.

The four starting points are **Gentle**, **Steady**, **Lively**, and **Subtle**.
Subtle is low-amplitude motion, not a freeze preset. **Motion details** holds
XYZ translation, rotation, scale, transition duration, change frequency, seed,
and the explicit stochastic toggle. **Source & looping** holds loop policy and
source-range editing; the scene scope also retains its named range list.
**Diagnostics** holds graph qualification, the scene branch trace, and field
statistics. These sections start collapsed.

In Local styles, choose **Duplicate**, adjust Motion amount, then paint the
copy elsewhere. Duplication copies settings and source range with a new
identity and seed; it does not copy or transform painted footprints. Changing
a style updates all locations assigned to that style, not other styles.
**Style options** contains rename, enable, remove, and one-level removal undo.
Stroke undo/redo remains separate from style editing.

The wind-direction pad, influence, and directional-coverage controls are
removed. New presets are direction-neutral, and both scene and local runtime
controllers ignore legacy directional preference fields. Old enum names and
session fields remain readable and round-trippable; saved/custom style names
are not forcibly renamed. This is intentionally a UI/policy change, not a
shader, archive, or controller-capacity change.

### Spatial motion brush

The implementation/session names `MotionRegion`, strength, variation and
coherence remain unchanged. Their UI names are Motion style, Motion amount,
Spatial variety and Grouping, respectively. The technical descriptions below
use the existing field names.

Painting is embedded under **Local styles → Paint this style** in the Motion
window. Its field is fixed in absolute Wang-world XY coordinates, while its GPU texture is
a moving cache for the active tile window. Low, Default, and High use exactly
8, 16, and 32 texels per Wang tile. The normal 97 by 97 Default map therefore
uses a 1552 by 1552 cache with 0.25-world-unit texels for four-unit tiles.
Continuous influence, strength, variation, and coherence occupy one
`RGBA8Unorm` texture; behavior/controller identity occupies one `R8Uint`
texture. Together they use approximately 11.5 MiB at Default quality.

Choose **Paint**, aim at the `z=0` authoring plane, and paint with LMB. **Erase**
restores scene default; **Navigate** returns camera control. Closing/collapsing
the Motion window or switching to Scene default finishes any pending stroke
and exits paint mode without deleting paint. **Brush size** is world-space
diameter (twice the stored radius), not motion magnitude. The brush is a vertical
column, so a footprint applies through the height of a plant or other exemplar.
**More paint tools & settings** retains Motion amount, Spatial variety, Grouping,
Synchronize, Smooth, opacity, spacing, falloff, quality and overlay controls.
Selecting an advanced tool activates it directly; it does not require clicking
Paint afterward. Strokes remain anchored when the active Wang
window moves; the CPU preserves overlapping cache texels, replays the ordered
stroke document only into newly exposed strips, and uploads those strips.
Changing quality, moving to a non-overlapping window, or replaying a document
that contains order-dependent Smooth strokes performs a full rebuild.

The viewport shows a Blender/Houdini-style live brush preview before paint is
applied. Its bright outer ring is the exact world-space radius, the translucent
fill follows the selected falloff, and the inner contour marks half influence.
A center crosshair and bounded live stroke trail improve placement. Visible
Gaussians inside the same canonical, instance-aware XY column receive a
tool-colored preview tint, using the same radius and falloff as rasterization.
When the pointer cannot reach a valid part of the authoring plane, a red dashed
cursor replaces the footprint. The preview is hidden while the pointer is over
the UI and never writes to the authoring field by itself.

The panel can overlay influence, strength, variation, coherence, or controller
assignment. **Show painted areas** toggles the overlay; turning it off also
disables automatic overlay following. With **Follow active paint tool**
(default on), selecting a brush tool or dragging its Strength, Variation or
Coherence target switches to that channel; Opacity follows Influence. The overlay
stays selected after release. Manual overlay selection is preserved until the next
tool/target interaction; disable Follow active paint tool to keep it fixed. These are brush
target controls, not edits to previously painted texels or per-region motion gains.
Lookup uses each Gaussian's canonical local position plus its
particular occurrence offset, so deformation, camera movement, LoD changes,
and draw sorting do not move the paint. Apply Behavior assigns the selected
Motion Region's deterministic stochastic timeline to each affected visible
tile occurrence. Strength blends that controller's complete translation,
rotation, and scale result with inherited global motion. Synchronize assigns
the selected region, clears spatial variation, and sets full coherence while
preserving strength. Unpainted and invalid samples copy the inherited global
result exactly. Scalar tools initialize unassigned texels with the selected
region, so Strength, Variation, or Coherence painted on blank space affects motion.

Variation chooses alternate deterministic stochastic timelines. Zero uses the
region's root timeline; higher values select alternates in more world-space
patches. Coherence increases patch width from one to sixteen Wang tiles in
discrete levels; full coherence uses one shared patch throughout the region.
It affects motion only where Variation is nonzero. Default coherence is about
0.5; different patches may reuse a timeline. This bounded representation does
not create a controller per texel or Gaussian. Painting can change the selected
timeline immediately. It cannot create graph jumps, so sources with few
eligible jumps can still produce subtle variation.

Each region has one root and up to three alternate seeds, sharing a maximum of
64 local controllers. Up to sixteen enabled regions retain all four timelines;
additional regions reduce the variants evenly, and 64 regions leave only roots.
The panel reports this capacity limit. Only slots used by the active painted
field have basis data uploaded to the GPU. The same world-space position
selects the same variant after camera movement, recentering, or quality changes.
Region seeds, speed, TRS gains, stochastic parameters, and source range are
editable in the shared Local styles editor and its collapsed sections;
scene settings can be copied into a style under **Source & looping**.

**Remove style** deletes the selected Motion Region/Controller and returns only its
currently owned texels to inherited global motion. The first/default region is
protected. Selection returns to it after removal. Freed IDs can be reused without
reviving old paint: a document-order removal event clears the old ownership before
later strokes replay. Other region definitions and global playback are preserved.
**Undo removal** restores the most recent removal and its paint. Undo newer strokes
first; subsequent region edits invalidate removal undo to avoid overwriting them.
This is separate from the bounded internal stochastic variants of each region.

### Authoring sessions and history

**Save session…** downloads a `motion-session.json` sidecar containing
global behavior/ranges, local regions and seeds, ordered absolute-world strokes,
and brush settings/quality. **Load session…** validates the document and prepares
replacement state before applying it. Loading restarts deterministic playback;
it does not restore an in-progress jump or the old camera view. Play/pause,
channel preview, and motion-enabled state remain current viewer choices.
Overlay and Follow active paint tool also remain current viewer choices. Sessions without
controller-removal history retain version 1; those with removal history use
version 2, so older renderers reject them rather than replaying removals incorrectly.
Both versions load in this renderer. Historical references to deleted regions are
validated against later removal events; new references must identify live regions.

Use the original archive and tile width. Compatibility checks cover archive
structure, backend, source duration, and tile width, but are not a content
fingerprint. Unsupported versions, missing region references, invalid values,
and incompatible structures are rejected. Session input is limited to 16 MiB,
10,000 strokes, and 200,000 path points. File cancellation and errors appear in
the panel. File operations finish even if the panel is closed.

**Undo stroke / Redo stroke** replay the ordered world-space document, including
paint outside the GPU window. New paint clears redo. Stroke undo stops at a
controller-removal boundary; use Undo remove there. The latest removal undo is
viewer-local and is not restored on session load. Global/region parameter history
and before/after comparison remain future
work. Save before replacing a session if it needs to be kept. Long histories
can cause a pause during replay, load, or undo. The GSWT archive stays unchanged.

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

A paused control change schedules one motion update. Authoring affects both
ordinary and blended loop evaluation without altering the archive. Loop policy
is a preview-only choice:

- **Hold** preserves the finite source interval and stops at its final state.
- **Direct wrap** jumps from the source endpoint back to the beginning.
- **Appended transition** preserves the source interval and appends an editable
  smooth transition from the final state to the first.
- **Blend ending into beginning** uses an editable final source window to
  blend toward the first state.

None of these policies modifies the archive or its authoritative non-looping
motion. The renderer continues to sort procedural instances from canonical
tile geometry; it does not read dynamic positions back from the GPU every
frame. When diagnosing a possible motion-packaging error, compare the normal
view with point-cloud visualization so canonical alpha-sort artifacts can be
distinguished from row-identity or deformation errors.

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
unreachable. Advanced diagnostics distinguish all close pairs, temporally
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

Named behavior controls map variation and gust frequency to deterministic
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

The Rust/Python parity tests use the umbrella workspace's
`tests/fixtures/motion_parity_v1.json`, `motion_parity_v2.json`, and
`motion_parity_v3.json`. In the normal `components/renderer` layout they are
found automatically. For a standalone checkout or a relocated verification
snapshot, set `FOURDGSWT_WORKSPACE_ROOT` to the umbrella directory containing
those fixtures before running the suite. The ignored archive gate sweep is a
separate diagnostic and is not required for ordinary test runs.

The adapter-dependent parity test prints one explicit skip reason only when a
native WebGPU adapter is unavailable. The pinned browser build remains:

```powershell
wasm-pack build --target web
```

## Water

In the rendering menu, expand **Water** and enable **Enable water**. Controls update live:

- **Water level / color**: mean height in final renderer world Z, and opaque base color.
- **Wave amplitude / wavelength**: geometric displacement and world-space scale (defaults 0.08 and 4.0). Amplitude zero makes the geometry flat.
- **Wave variation**: uses four fixed, irregular directions and wavelength ratios. Turn off to compare the previous three-wave pattern.
- **Reflection strength**: Fresnel-weighted reflection of the currently displayed skybox (default 1, zero disables it). Load and enable a skybox to see reflection.
- **Roughness**: blurs the environment reflection (default 0.18).
- **Ripple strength / scale**: small animated normal detail, independent of geometric displacement (defaults 0.22 and 0.35 world units). Strength zero disables it.
- **Wave speed / Playing**: control the shared wave and ripple clock. Zero speed holds the current phase; pause is independent of GS motion.

Water starts disabled. Amplitude is limited to 3.5% of the primary wavelength.
Its mean level stays fixed when the camera moves, tiles stream, or scene Z scale
changes. Archive preprocessing can shift source coordinates, so Blender's
waterline Z is not automatically the renderer's water level.

One continuous surface covers the committed tile window plus a one-tile border,
including gaps within and between tiles. Coverage is finite. **None** and
**HeightMap** mapping are supported; Sphere disables water. Streaming does not
reset world-space wave or ripple phase. Speed changes affect future time only.
Global Freeze holds the clock; Step advances by 1/60 second when playback is on.

Water intersections are computed once per screen pixel each frame into an RG32Float
texture (depth and positive view distance, approximately 15.8 MiB at 1920x1080).
Water shading and all GS layers use exact texel loads from the same result. The
intersection pass is independent of proxy depth; misses are overwritten and the
texture is recreated on viewport resize. Dry GS pipelines compile without water
cut moments, the intersection solver or fragment-depth output.

Dynamic GS is
clipped against the current wave surface using a conditional view-depth Gaussian
distribution. This is a Gaussian approximation, not a reconstructed mesh shoreline.
Fine ripples affect only lighting, so they cannot move the waterline. Both ripple
and wave lighting normals are filtered when smaller than a pixel.

Reflection uses a 256-pixel cubemap with nine GGX-prefiltered roughness levels
(about 4 MiB), generated when a skybox is configured. Filtering/blending uses
linear color decoded from the existing displayed skybox; HDR sky color has already
been tone mapped by the skybox renderer. The pass is an economical environment
reflection approximation. It does not render GS object reflections, transmission,
refraction or foam. Disabling the displayed sky also disables its reflection.

For a stage-2 comparison, set reflection and ripple strengths to zero and turn off
wave variation. Set amplitude to zero as well to restore stage-1 pixels.
The geometric solver uses conservative first- and second-derivative bounds and
an adaptive iteration budget. Its 16,384-step cap is finite; extreme grazing views,
large coverage and short waves can cost more or reach that cap.

Native GPU acceptance includes coverage, GS contact and proxy depth, motion and
clock controls, environment replacement, cubemap/HDRI orientation, Fresnel angle
response, roughness, detail-only contact preservation and subpixel filtering.
Intersection tests compare 2,048 GPU rays (both spectra) with an independent dense
CPU reference, plus two short-wave grazing regressions. Cache tests cover camera/phase
changes, resizing, miss overwrite, and flat-water edge derivative preservation.

```powershell
cargo test --offline --target x86_64-pc-windows-msvc --lib water_ -- --test-threads=1
```

For real-scene review, set `WATER_REVIEW_ARCHIVE` to a local tile ZIP containing
`tile0_lod1.ply`, `WATER_REVIEW_SKY` to a local EXR, and `WATER_REVIEW_DIR` to an
output directory, then run:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc --lib water_export_review_frames -- --ignored --nocapture
```

This exports static levels, animated Flowers contact, and low-angle open-water
comparisons. It reports GPU timestamp median/p95 for the previous material,
reflection alone, and the full material. Timings include clear, sky, water and GS;
they exclude CPU work, readback and UI. The fixture uses one 29,279-splat Flowers
tile at 768×768, with 30 samples after five warm-up frames. It is a native offscreen
measurement, not full-world browser FPS. The iceberg GS archive still needs
scene-specific acceptance when available.

For the dense 1080p performance comparison, set `WATER_PERF_ARCHIVE` to a local
shrub_sorrel tile ZIP and `WATER_PERF_DIR` to the output directory, then run:

```powershell
cargo test --offline --target x86_64-pc-windows-msvc water_dense_shrub_benchmark -- --ignored --nocapture --test-threads=1
```

This benchmark renders 25 copies of the canonical LoD0 tile from a fixed camera.
It measures dry, flat, wave-only and full-material configurations with GPU timestamps,
and exports images plus `timings.json` (including intersection, material and GS timings).
It does not measure the browser, sorting, dynamic-motion compute or full-world FPS.
Run GPU benchmarks separately from other GPU tests and close previous native test
executables before relinking on Windows.
See [water performance validation](docs/water-performance.md) for the measured
comparison and its scope.

Build the browser package with **Binaryen 132**: put its `bin` directory first
on the current shell's `PATH`, verify `wasm-opt --version`, then run
`wasm-pack build --target web`. Confirm that the build log reports that executable.
The older bundled Binaryen 117 aborts in its Precompute pass on this project.

## Performance profiling

Press **P** during rendering to open the performance panel. The detailed
profiler reports rolling mean and p95 timings for the CPU frame, motion
preparation, Gaussian render encoding/uploads, worker sort/build work, GPU
global motion compute, GPU authored motion compute, GPU Gaussian rendering,
GPU water intersection and GPU water shading.
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

For the manual authored-motion performance gate, test `flesh_eyeballs`,
`shrub_sorrel`, and `anthurium` at 1920 by 1080 with their current default
tile, LoD, and selective-merge settings. Use Default brush quality, set the
diagnostic overlay to Off, and apply one Behavior stroke. Record FPS, CPU frame
p95, GPU authored-motion p95, GPU Gaussian-render p95, candidate/painted rows,
and affected draws. The target is at least 30 FPS on the project desktop GPU;
this README does not treat that target as passed until the browser measurements
have been recorded.

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
