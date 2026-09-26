# Renderer usage

[Back to README](../README.md) · [Developer reference](DEVELOPMENT.md)

## Load and navigate

Load a static six-LoD tile ZIP or a supported dynamic 4DGSWT archive, adjust the
configuration and select **Confirm**. Use **Reconfig** to change the configuration;
reload the page to select another scene. Dynamic archives expose the Motion editor.
See [archive requirements](DEVELOPMENT.md#asset-contract) for supported versions and limits.

| Control | Action |
|---|---|
| WASD / IJKL | Move / look |
| Space | Sprint |
| M / P | Toggle rendering menu / performance panel |
| B | Toggle Motion when no text field has focus |

## Cubed Sphere

In **Config / Reconfig → Surface mapping**, select **Cubed Sphere** (replaces
the old Sphere mapping). Set **Tiles per face edge (N)** and **Sphere radius**,
then select **Confirm**. Start with N=8 and radius=20; the panel shows the total
`6 × N × N` tile instances. Odd N is supported. This renderer currently accepts
N=1–128; the practical limit depends on asset density, LoD and GPU memory.
The sphere is centered at the world origin. Each **Confirm** frames the complete
sphere from outside, looking at its center. Its diameter fills 95% of the limiting
screen dimension, accounting for the current viewport and field of view. Normal
camera movement remains available afterward; confirming again restores the fit.

N controls the density of the grid and radius controls its world size. Tile width
still describes the source asset. The plane's half-width/half-height controls are
hidden in this mode and their values are retained for switching back. Planar
scene-scale controls are also hidden; use N/radius for the spherical surface.

The six face orientations preserve binary Wang labels and source-pattern
direction for **rotated-edge assets** (`rotate_tile=true`), including the current
sun-surface assets. No new tile variants or runtime orientation search is needed.
Assets constructed with separate horizontal/vertical edge families are not
guaranteed to match. Projection preserves shared boundary positions, but finite
Gaussian support, motion and three-tile corner content can still reveal seams;
test the intended asset at close range.

Sphere uses per-tile sorting; planar selective merging, the flat proxy ground and
water are disabled. Rear-facing tiles reuse reversed presort index streams,
including LoD labels, instead of allocating a second full presort bank. Those
draws require streamed index uploads. FPS depends on visible splat count and N;
no fixed performance target is implied.

## Motion

The Motion window opens for a newly configured dynamic archive. Its shared header
controls play/pause, restart, motion enable, scene time and session save/load.
Scrubbing selects authoritative source time and resets the current stochastic
transition; local styles have independent timelines.

Choose **Scene default** for unpainted areas or **Local styles** for a named local
controller. **Motion amount**, **Speed** and **Playback variation** are shared
controls. Motion amount scales translation, rotation and scale for the selected
scope; it does not multiply every local style. Variation enables stochastic
branching but still requires eligible graph jumps.

**Gentle**, **Steady**, **Lively** and **Subtle** are starting points; Subtle is
low-amplitude motion, not freeze. **Motion details** exposes per-channel gains,
transition duration, change frequency, seed and the stochastic toggle.
**Source & looping** contains source ranges and these preview-only loop policies:

- **Hold** stops at the final source state.
- **Direct wrap** jumps to the beginning.
- **Appended transition** adds a smooth endpoint-to-start transition.
- **Blend ending into beginning** blends within a final source window.

**Duplicate** creates a style with copied settings/range and a new identity and seed,
without copying painted footprints. Editing a style updates all areas assigned to
it. **Style options** provides rename, enable, remove and removal undo. The default
style is protected. Playback is direction-neutral; legacy directional preferences
do not affect playback.

### Paint local styles

Under **Local styles → Paint this style**, choose **Paint** and use LMB on the
world `z=0` authoring plane. **Erase** restores scene default; **Navigate** returns
camera control. Brush size is world-space diameter, and its vertical column covers
the full height of an exemplar. The live ring/tint previews the footprint without
changing paint; a red dashed cursor indicates no valid plane intersection.

Paint stays at its absolute Wang-world XY location when the tile window moves.
Closing/collapsing the Motion window, toggling B or switching to Scene default
finishes the stroke and exits painting without deleting it. Reconfiguration resets
the painted field; save a session before replacing it.

**More paint tools & settings** contains opacity, spacing, falloff, quality,
Motion amount, Spatial variety, Grouping, Synchronize and Smooth:

- Motion amount blends the assigned local result with scene-default motion.
- Spatial variety selects deterministic alternate timelines; Grouping increases
  their spatial patch size and has an effect only when variety is nonzero.
- Synchronize assigns the style, removes spatial variety and uses full grouping,
  preserving motion amount.
- Scalar tools initialize blank areas with the selected style. Painting may switch
  timelines immediately; it cannot create new graph jumps.
- Low/Default/High quality uses 8/16/32 texels per tile. The shared limit is 64 local
  controllers; more styles leave fewer alternate timelines per style.

**Show painted areas** toggles diagnostics. **Follow active paint tool** switches
the overlay when tools or targets change; disable it to keep a manual overlay.
Targets affect new strokes, not previously painted values. Overlay Off is the
representative setting for normal playback performance.

### Sessions and undo

**Save session…** downloads a JSON sidecar with scene/local settings, seeds, source
ranges, ordered world-space strokes and brush settings. The archive is unchanged.
**Load session…** validates before replacing state and restarts deterministic
playback; it does not restore the camera or an in-progress transition.
Play/pause, motion enable, channel preview and overlay choices remain current.

Use the original archive and tile width: compatibility checks are structural,
not an archive-content fingerprint. Versions 1 and 2 load; removal history requires
version 2. Limits are 16 MiB, 10,000 strokes and 200,000 path points.

**Undo stroke / Redo stroke** includes paint outside the visible cache. New paint
clears redo. **Remove style** clears only that style's ownership; **Undo removal**
restores the latest removal and paint after newer strokes are undone. Later style
edits invalidate removal undo, which is not restored from a saved session.
Parameter-edit history is not supported. Long histories may pause during replay.

## Water

Enable **Water → Enable water** in the rendering menu. Water starts disabled and
supports None/HeightMap surface mapping; Sphere disables it.

| Control | Meaning |
|---|---|
| Water level / color | Mean final-world Z and opaque base color |
| Amplitude / wavelength | Geometric displacement; zero amplitude is flat |
| Wave variation | Irregular wave directions and spacing |
| Reflection strength / roughness | Skybox reflection amount and blur |
| Ripple strength / scale | Animated lighting detail, independent of geometry |
| Wave speed / Playing | Water clock, independent of GS motion |
| Underwater effect | Opt-in absorption, depth-dependent scattering, and underwater interface shading |
| Underwater color | Far-water tint, separate from the upper surface color |
| Clear near distance | Preserve nearby scene colors before fog begins (default 3 world units) |
| Fog half-distance | Distance beyond the clear zone where contrast falls to 50%; larger means clearer water (default 15) |
| Underwater sunlight | Strength of directional light scattered through the water |
| Sun azimuth / elevation | Direction of incoming sunlight; refraction determines the underwater shaft direction |
| Light shafts / strength / width | Optional animated beams; width is in world units |
| Caustics / strength / scale | Optional animated surface highlights; scale is cell spacing in world units (off by default) |
| Caustic speed | Independent motion multiplier, default 0.25x; 0 freezes caustics, 1x restores their original speed at wave speed 1 |

Amplitude is capped at 3.5% of wavelength. Water covers the committed tile window
plus a one-tile border, including gaps. Its mean height and wave phase stay fixed
as tiles stream; source preprocessing means Blender Z need not equal renderer Z.
Speed changes affect future time; global Freeze holds the clock and Step advances
1/60 second when water playback is on.

Enable a skybox for environment reflections. The upper water surface is opaque:
GS object reflections, scene transmission/refraction and foam are not implemented. Ripples change lighting only,
not the waterline. GS contact uses a Gaussian approximation, not a mesh shoreline.
For a simple baseline, disable reflection, ripples and wave variation; set amplitude
to zero for flat water.

For the soft coral scene, place the water level above the terrain and enable
**Underwater effect**. Nearby coral remains detailed; distant splats and proxy
ground gradually approach the water lighting. Scattering is brighter near the
surface and towards the sunlight, and darker at depth. Uncovered background
represents distant water beyond the loaded tile window, with a short smooth
transition as the camera crosses the surface.
The first 3 world units are clear by default. The fog then reduces contrast gradually,
with only a small extra loss in the red channel. The old visibility setting removed
95% of scene contrast by the selected distance; **Fog half-distance** now removes 50%
over that distance beyond the clear zone in linear light, preserving near/mid-range detail.

When submerged, the underside has its own water-to-air interface shading: refracted
sky inside Snell's window and total internal reflection outside it. Its color does
not use the opaque upper-surface pigment. Internal reflection is approximated by
far-water radiance, so coral/ground mirror reflections are not present. No-skybox
views use neutral daylight for the transmitted sky. The underside is also fogged
over its distance from the camera; ripple lighting normals cannot change which
side of the interface is rendered. The Fresnel/refraction model follows
[PBRT's dielectric interface](https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission).

The effect starts disabled. A view-aligned scattering volume is allocated only
while active underwater (one-eighth viewport dimensions, capped at 320 x 180,
48 depth slices). It integrates light in linear space. Each Gaussian samples
the accumulated scattering at its own view depth before normal alpha blending;
proxy ground and the underside use the same volume. This preserves transparent
edges without an opaque coral-depth prepass. Large splats still approximate
depth by their center, and interpolate lighting across their projected quad.
Intersection and scattering textures are reused while their inputs are unchanged.
Camera, wave, coverage or medium changes refresh the affected cache; caustic
animation updates its sampling offsets without invalidating either texture.
The underwater background replaces the ordinary sky pass, including at the waterline.

Light shafts use sparse artificial light columns with bright cores and dark gaps,
anchored in world space along the sunlight direction. Their placement does not
depend on geometric waves. They follow the water clock, including pause/freeze,
and work with the sun offscreen. This is an artistic single-scattering model:
coral does not cast volumetric shadows towards the light, and multiple scattering
and scene reflections are not implemented. Turning shafts off retains
the depth/directional water lighting; turning the underwater effect off skips
its compute/shading work and releases the large volume. No temporal history is used.

**Caustics** adds independent artistic light patterns to submerged GS and proxy
ground before fog, never to the background or water interface. A periodic 256 x 256
single-channel texture with mip levels (about 85 KiB) is generated once on first
use and sampled twice with drifting coordinates; there is no per-frame focusing
simulation or extra rendering pass. **Caustic speed** controls motion independently
of wave/shaft speed and defaults to a slower 0.25x. Speed edits preserve the current
pattern position; Water pause/global Freeze still pauses the animation.
It works with flat water and with light shafts disabled. The option defaults off;
disabled/above-water paths skip caustic sampling. A one-texel placeholder is used
until first enablement; the small cookie then stays resident through toggles,
viewport resizing and water reentry. Toggling caustics does not reallocate the
intersection or scattering textures.
GS samples at its world center with footprint filtering, so large splats soften
fine lines. This is illumination of baked GS colors, not physical relighting:
normal-dependent focusing and coral shadows are not modeled, and highlights can
reach surfaces that would be shadowed. Increase **Caustic scale** for coarse LoDs.

The profiler's **GPU water preparation** includes both intersection and optional
scattering integration. **GPU water shading** measures the surface pass.
On unchanged frames, preparation records only a timestamp boundary when profiling
is enabled; neither compute shader is dispatched.

## Performance and diagnostics

Press **P** for rolling mean/p95 CPU, worker and GPU pass timings, row/draw counts,
upload bytes and authored-motion activity. GPU timings require adapter timestamp
support; the panel reports unsupported/readback-error states. **Reset Timer** clears
history, and profiling can be disabled.

Unchanged sorted draws reuse GPU index data. Visible tile parameters are uploaded
as one batch only when needed, so upload bytes may drop while animation continues.
Flat water also reuses its intersection texture while ripples, shafts and caustics
keep animating. Compare frame timings on the same scene and camera to assess FPS.

Startup console timings measure loading and preparation separately; GPU preparation
means CPU work/API submission, not completed GPU execution. Compare the same asset
after a full reload. See [water measurements](water-performance.md) and
[validation commands and open checks](DEVELOPMENT.md#validation).

The procedural renderer sorts canonical tile geometry rather than reading back
dynamic centers each frame. Use point-cloud visualization when distinguishing
alpha-sort artifacts from motion/row-alignment errors.
