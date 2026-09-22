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

Amplitude is capped at 3.5% of wavelength. Water covers the committed tile window
plus a one-tile border, including gaps. Its mean height and wave phase stay fixed
as tiles stream; source preprocessing means Blender Z need not equal renderer Z.
Speed changes affect future time; global Freeze holds the clock and Step advances
1/60 second when water playback is on.

Enable a skybox for environment reflections. Water is opaque: GS object reflections,
transmission, refraction and foam are not implemented. Ripples change lighting only,
not the waterline. GS contact uses a Gaussian approximation, not a mesh shoreline.
For a simple baseline, disable reflection, ripples and wave variation; set amplitude
to zero for flat water.

## Performance and diagnostics

Press **P** for rolling mean/p95 CPU, worker and GPU pass timings, row/draw counts,
upload bytes and authored-motion activity. GPU timings require adapter timestamp
support; the panel reports unsupported/readback-error states. **Reset Timer** clears
history, and profiling can be disabled.

Startup console timings measure loading and preparation separately; GPU preparation
means CPU work/API submission, not completed GPU execution. Compare the same asset
after a full reload. See [water measurements](water-performance.md) and
[validation commands and open checks](DEVELOPMENT.md#validation).

The procedural renderer sorts canonical tile geometry rather than reading back
dynamic centers each frame. Use point-cloud visualization when distinguishing
alpha-sort artifacts from motion/row-alignment errors.
