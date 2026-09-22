# Water performance

Validated 2026-09-21 on an NVIDIA GeForce RTX 4060 Laptop GPU (Vulkan, driver 560.94).

## Changes

- Disabled water uses separate GS shaders without water cut calculations or fragment-depth output.
- Active water computes its ray intersection once per pixel into a full-resolution RG32Float texture. Water shading and every overlapping GS fragment read the same result.
- Contact clipping, waves, ripples and environment reflection are preserved. The hit texture uses about 15.8 MiB at 1920x1080. Press **P** for separate intersection, shading and GS timings.

## Benchmark

`water_dense_shrub_benchmark`: 25 copies of the shrub_sorrel canonical LoD0 tile, 3,275,875 splats at 1920x1080. Camera, wave phase and settings are fixed; each mode uses five warm-up frames and 30 GPU timestamp samples. The baseline uses actual pre-fix source. See the [developer reference](DEVELOPMENT.md#water-review-and-benchmark) for benchmark commands.

| Mode | Before median / p95 (ms) | After median / p95 (ms) | Changed image pixels |
|---|---:|---:|---:|
| disabled | 17.797 / 20.021 | 8.225 / 9.271 | 0 / 2,073,600 |
| flat | 15.245 / 15.587 | 10.771 / 11.185 | 0 / 2,073,600 |
| waves | 108.382 / 110.038 | 10.887 / 11.251 | 0 / 2,073,600 |
| full | 108.564 / 110.009 | 11.399 / 11.942 | 0 / 2,073,600 |

All four benchmark images are byte-identical before and after the fix. Timings exclude browser presentation, CPU work, worker sorting and dynamic-motion compute; they do not measure end-to-end FPS.

Separately, the user reported browser performance on shrub_sorrel at 1920x1080 improving from approximately 10 to 45 FPS with water disabled, and 4 to 35 FPS with water enabled. These are observational results, not controlled GPU benchmark measurements.
