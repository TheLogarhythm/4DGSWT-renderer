# GSWT Renderer

Interactive Gaussian Splatting Wang Tiles renderer, extended in the 4DGSWT workspace
with dynamic playback, spatial motion authoring and animated water. Rust, wgpu/WebGPU
and egui support procedural tiling, selective merging and LoD transitions.

Based on the SIGGRAPH Asia 2025 paper
[GSWT: Gaussian Splatting Wang Tiles](https://yunfan.zone/gswt_webpage/).
The offline constructor is in the sibling [constructor repository](../constructor).
The published [demo](https://yunfan.zone/gswt_webpage/demo/) represents the upstream project;
local extensions are described below.

## Quick start

Use a WebGPU-capable browser and the pinned Rust toolchain. Install wasm-pack 0.13.1
and put Binaryen 132 on PATH; see [build requirements](docs/DEVELOPMENT.md#build).

From this directory:

```powershell
wasm-opt --version
wasm-pack build --target web
sfz -r --coi
```

Open the server's `index.html`. The server must supply cross-origin isolation
headers. The build produces `pkg/`.

1. Select a static or dynamic six-LoD GSWT ZIP.
2. Adjust scene configuration and click **Confirm**.
3. Use **WASD** to move, **IJKL** to look and **Space** to sprint.
4. **M**, **P** and **B** toggle the rendering, performance and Motion panels.
5. Use **Reconfig** to change configuration; reload to select another scene.

Upstream example assets:
[official datasets](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yzengbm_connect_ust_hk/IgB6p7U3s0FARoryHllz2jVuATc39DOnvkGZ9ieHkX_-hNw?e=Whq3DT).

## Documentation

- [Usage](docs/USAGE.md): motion, painting, sessions, water and diagnostics.
- [Development](docs/DEVELOPMENT.md): builds, asset contracts, runtime invariants,
  tests and known issues.
- [Water performance](docs/water-performance.md): benchmark conditions and results.

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
