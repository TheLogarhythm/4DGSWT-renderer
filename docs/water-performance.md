# Water rendering performance fix

Validated 2026-09-21 on an NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 560.94).

## Changes

- Disabled water uses separate GS shaders without water cut calculations or fragment-depth output.
- Active water computes its ray intersection once per pixel into a full-resolution RG32Float texture. Water shading and every overlapping GS fragment read the same result.
- Contact clipping, wave geometry, fine ripples and environment reflection remain enabled. The hit texture uses about 15.8 MiB at 1920x1080 and is recreated on resize.
- The P profiler now separates GPU water intersection, water shading and GS rendering.

## Reproducible comparison

The opt-in `water_dense_shrub_benchmark` renders 25 copies of the actual shrub_sorrel canonical LoD0 tile: 3,275,875 splats at 1920x1080, fixed camera, fixed wave phase and identical settings. Each mode uses five warm-up frames followed by 30 GPU timestamp samples. Before measurements use the saved pre-fix source, not a simulated slow mode.

| Mode | Before median / p95 (ms) | After median / p95 (ms) | Changed image pixels |
|---|---:|---:|---:|
| disabled | 17.797 / 20.021 | 8.225 / 9.271 | 0 / 2,073,600 |
| flat | 15.245 / 15.587 | 10.771 / 11.185 | 0 / 2,073,600 |
| waves | 108.382 / 110.038 | 10.887 / 11.251 | 0 / 2,073,600 |
| full | 108.564 / 110.009 | 11.399 / 11.942 | 0 / 2,073,600 |

Full-material pass medians: intersection 0.333 ms, shading 0.383 ms, GS 10.547 ms. These are independently calculated pass medians; their sum need not equal the sampled whole-frame median.

These measurements exclude browser presentation, CPU work, worker sorting and dynamic-motion compute. They demonstrate the rendering improvement and do not establish browser FPS or guarantee a return to the previously reported 30 FPS.

The first benchmark fixture changed the water map bounds without updating the GS renderer's stored user settings. That mismatch explained the initial pixel differences. The final fixture reconfigures GS after changing bounds; all four final images are byte-identical. Use the `water-perf-synced-before` and `water-perf-synced-after` data, not the earlier diagnostic captures.

## Validation

- Full development workspace native suite: 325 passed, 0 failed, 5 ignored.
- Isolated water-only commit native suite: 288 passed, 0 failed, 3 ignored; unrelated comparison and maintenance changes are excluded.
- GPU regression coverage includes GS contact, terrain occlusion, moving GS, near-horizon intersections, cache refresh after camera/phase/viewport changes, and flat-water boundary lighting.
- Structural checks verify that dry GS shaders contain no water solver or fragment-depth output and wet GS shaders consume the shared hit texture without compiling the solver.
- Browser package built with `wasm-pack build --target web` and Binaryen 132 optimization.

Raw comparison logs, timings and images are under the workspace `tmp/water-perf-synced-before*` and `tmp/water-perf-synced-after*`. Final validation logs: `tmp/water-perf-final-tests.log` and `tmp/water-perf-final-wasm-build.log`.

Browser verification reported approximately 45 FPS with water disabled and 35 FPS enabled, compared with 10 FPS disabled and 4 FPS enabled before the fix. These are user-reported measurements on shrub_sorrel at 1920x1080. The P profiler separates intersection, shading and GS costs.

Water-only integration logs: `tmp/water-commit-tests.log` and `tmp/water-commit-wasm.log`.
