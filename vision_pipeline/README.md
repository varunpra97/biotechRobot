# Vision Pipeline — Real Scene → Measured 3D CAD

Takes calibrated multi-view photos of a real lab room (a platform/bench with
an unknown number of unknown objects on it) and produces, for each object:
a 3D mesh with **measured** (not assumed) real-world dimensions, an
open-vocabulary label, and a 6-DoF placement in the platform frame.

This is the real-world counterpart to `biotech-lab-robotics/`'s simulator:
that side builds synthetic scenes from hand-authored CAD
(`cad_source/make_*.py`) with assumed dimensions. This pipeline goes the
other way — photograph a real bench, recover what's actually on it and how
big it actually is, so a real bench can be turned into a sim scene (or
compared against one) without hand-measuring every object with calipers.

**No fixed object catalog.** The platform can hold any number of objects of
types not known in advance. Nothing in this pipeline hard-codes "there is a
test tube holder" — each object's shape family is inferred per-instance from
its silhouette and an open-vocabulary label, not looked up.

## Input assumptions

- **Multi-view**: ~10 images per capture (see "Camera rig" below), all
  static (no video/SLAM) — a room with a platform, shot from a ring of
  angles plus top-down and one close detail shot.
- **Per-camera intrinsics are known ahead of time** (focal length, distortion),
  from a one-time OpenCV chessboard calibration. Loaded from a file, not
  estimated from the scene photos. See `calibration.py`.
- **Reference object = fiducial markers**, not a lab item. A small number of
  printed ArUco markers at fixed, hand-measured 3D positions in the platform
  frame (e.g. taped at the corners of the bench) anchor absolute scale and
  give every camera's pose directly in that frame via `solvePnP` — no
  separate SfM scale ambiguity to resolve. See `markers.py`.

## Coordinate convention

Matches `biotech-lab-robotics/sim/lab_scene.py`'s platform convention so
reconstructed objects drop in with zero re-expressing:

- Origin: center of the physical bench/platform top surface.
- Z: up, away from the bench surface (bench top is Z=0 here).
- X/Y: aligned with the bench's long/short edges.
- Units: meters.

A downstream consumer places an object at
`world_pos = (PLATFORM_CX + x, PLATFORM_CY + y, BENCH_Z + z)` — the same
`cx, cy` parameterization `lab_scene.py` already uses for its own fixtures
(`holder_xy(cx, cy)`, `centrifuge_pos(cx, cy)`).

Marker 3D positions in `config/marker_layout.json` must be measured in this
same frame.

## Pipeline stages

1. **Camera pose recovery** (`markers.py`, `pose_solver.py`) — detect ArUco
   markers in each image, pool all visible markers' known 3D corners (from
   `marker_layout.json`) with their detected 2D corners, solve one PnP per
   image → camera extrinsics directly in the platform frame. Build first,
   since every later stage needs calibrated views.
2. **Open-vocabulary instance segmentation** (per view) — find and mask every
   object on the platform without assuming what or how many there are, and
   attach a free-text label to each instance (e.g. "clear cylindrical tube",
   "flat perforated tray"). The label is a *hint* for stage 3's shape-prior
   choice and for naming the exported asset, not a lookup key.
3. **Cross-view instance association** — match each view's masks into 3D
   instances using the now-known camera geometry (consistent masks across
   views back-project to the same region in platform space).
4. **Shape reconstruction, hybrid per instance**:
   - **Visual hull / space carving** from calibrated silhouettes — the
     general-purpose fallback, works for any shape, and tolerates the
     texture-free / reflective / transparent surfaces common in lab-ware
     where photometric multi-view stereo fails outright.
   - **Primitive fitting** (cylinder, rectangular prism, prism-with-bore-grid,
     etc.) layered on top when an instance's silhouette across views is
     consistent with a simple parametric family — this is what makes the
     output *measured* rather than noisy: fitting one radius and one height
     to 8-10 clean silhouettes is far more accurate than trusting a carved
     voxel surface. The open-vocab label narrows which primitive families to
     try first.
5. **Mesh export + placement** — primitives become clean analytic meshes
   (same spirit as `cad_source/make_*.py`, but fitted parameters instead of
   guessed ones); non-primitive instances fall back to Poisson surface
   reconstruction off the carved hull. Placement is not a separate step:
   reconstruction already happens in the calibrated platform frame.

## Depth (RGB-D) path

When the camera also measures depth (a time-of-flight / LiDAR module, a
structured-light or laser scanner, active stereo), depth replaces the
visual hull as the base geometry. A silhouette only says "the object is
somewhere along these rays", so a hull fills every concavity no silhouette
sees into (the centrifuge bowl came out as a dome, +12.5mm). A depth pixel
says where the surface is, and that everything in front of it is empty.

```bash
# everything: poses -> (segmentation) -> depth fusion -> silhouette refinement -> (LLM) -> USD assets
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/run_pipeline.py --capture-dir CAPTURE [--scene-context ctx.txt]
```

CAPTURE holds `images/`, `depth/` (one map per photo, same stem; uint16 PNG
in 0.1mm or float `.npy` meters, Z-depth registered to the photo),
`camera_intrinsics.json`, `marker_layout.json`, `platform_dims.json`.

How (`depth_fusion.py`, `scripts/reconstruct_depth.py`):

- TSDF fusion with free-space carving on a dense grid in the platform
  frame, poses from the markers. Unobserved space counts as occupied, so
  every mesh is watertight; surfaces are exact wherever any view saw them.
- Coarse (4mm) over the whole platform finds the objects (3D components,
  calibration fixtures removed, pieces nested in another's footprint
  merged); each object is then fused finely (1mm) on its own.
- A short band behind each surface, and a second pass that takes "behind
  the surface" evidence only from views the surface faces: with a single
  symmetric band, thin walls (2mm bin walls) were averaged away (perfect
  depth) or doubled in thickness (noisy depth: 0.82 -> 0.27mm median error).
- Sensor noise is simulated, not assumed away (`depth_sensor.py`): presets
  `perfect`, `structured_light`, `tof`, `active_stereo`, `phone_lidar`
  (resolution, distance-dependent noise, ToF mixed/flying pixels, stereo
  edge fattening, grazing and random dropouts, per-frame scale error), or
  a JSON file with the real sensor's numbers.
- Silhouette refinement (`refine_freeform.py --recon-subdir reconstruction_depth`)
  pulls outlines onto the colour images' edges: each view's move is one
  constraint along its edge normal, solved jointly per vertex; vertices
  on no outline stay on the depth surface; no vertex may leave the fused
  surface by more than 3 sigma of its noise; image edges are gated on the
  depth map's own discontinuities when there's no segmentation.
  `run_pipeline.py` refines only when the fused depth is noisier than
  0.15mm: refining a perfect-depth start on 2016px images made it worse
  (0.15 -> 0.4mm), since good depth is already more precise than edges.

- Camera poses are refined from depth (`depth_registration.py`): each
  view's depth is aligned (point-to-plane ICP) first to geometry known
  exactly -- the bench plane z = 0 and the marker quads -- then to all
  other views. Marker-only poses were 1.0mm median / 5.3mm max off, and
  every camera sat ~0.6mm too far (a common scale bias no view-to-view
  alignment can see; it dilated every object by ~0.5mm). After: 0.26 /
  0.51mm, radial bias 0.09mm.
- Depth must be registered to K's pixel convention. Isaac's depth
  annotator samples each pixel at its corner (+0.5px); the manifest says
  so (`"pixel_offset_px": 0.5`) and fusion corrects for it. Uncorrected,
  every surface was 0.2mm off. A real sensor's registration needs the same
  check.

**End-to-end, Isaac Sim** (`depth_tests/random_s1`, one command: `run_pipeline.py`; 8 random objects
rendered path-traced with RGB + depth, 12MP, 25 views; poses from the
markers in the photos, refined by depth; scored against the generator's
meshes):

| sensor | size error, worst axis | surface (observed), median |
|---|---|---|
| perfect depth | <=0.38mm on 7/8 (mushroom 0.93: under its cap) | 0.03-0.055mm |
| structured light | <=0.44mm on 7/8 (mushroom 0.94) | 0.04-0.08mm |
| ToF | 0.6-5.5mm (thin rims and plates worst) | 0.19-0.51mm |
| ToF + 12MP silhouette refinement | x/y 0.07-1.16mm; heights +1.2-4.3mm | |

With ToF, a 640px pixel covers ~2.7mm at 1.5m: the top of a 2-4mm wall or
plate is all mixed pixels, so heights of thin parts come out 1-4mm high.
That is the sensor's limit, not the fusion's; outer outlines are fixed
by the photos. For 0.5mm on everything, use a structured-light / laser
scanner, or capture depth closer.

Also on the CPU (exact poses, ray-cast depth of the exact meshes; size
error = worst axis, surface = median distance over what any camera sees):

| scene / sensor | size error | surface (observed) |
|---|---|---|
| lab scene, perfect depth | <=0.5mm on 5/6 objects (pipette nozzle 1.1) | 0.05-0.12mm |
| lab scene, structured light | <=0.52mm on 5/6 (pipette 1.2) | 0.05-0.10mm |
| lab scene, ToF | 0.8-4.5mm (pipette nozzle worst) | 0.16-0.35mm |
| lab scene, active stereo | 5-13mm (edge fattening) | 0.4-0.95mm |
| 8 random objects, perfect | <=0.46mm on 6/8; mushroom 1.55, container 0.65 | 0.03-0.06mm |
| 8 random objects, structured light | <=0.38mm on 7/8; mushroom 1.55 | 0.03-0.06mm |
| 8 random objects, ToF | 0.7-3.2mm (rims) | 0.20-0.30mm |
| ... ToF + silhouette refinement (2016px images) | x/y mostly 0.2-1.0mm, heights +1-2mm | 0.21-0.25mm |

What the numbers mean:
- The centrifuge, the visual hull's worst case, goes from +2.6mm (rim cut
  heuristic) to 0.22mm: its bowl is measured, not guessed. (Silhouette
  refinement had an even-odd fill bug -- front and back faces cancelled,
  leaving holes in the projected mask; fixed by filling per triangle. On
  the hull path the centrifuge's x/y improved 1.15/0.90 -> 0.50/0.04mm but
  the rim heuristic, tuned on the buggy outlines, now reads +4.0mm; other
  refinement results moved by <0.1mm.)
- 0.5mm needs a metrology-class depth sensor (structured light / laser
  line). With ToF noise (2.25mm per pixel at 1.5m) the fused surface is
  ~0.25mm median, but extents carry the noise's extremes: thin rims come
  out 1-2mm high. Silhouettes fix outer outlines (x/y) but not the near side
  of a rim, which is never on an outline. Shooting depth closer helps
  quadratically for stereo/structured light, linearly for ToF.
- What no camera sees (under the mushroom cap's edge, tip shafts inside a
  rack, the inside of a closed housing) stays filled; scores report
  "observable" surface separately and say how much of the truth was seen.

## Composite objects: parts, materials, masses

Depth fusion knows nothing about parts: a rack and the tubes standing in it
come out as one solid. Splitting needs two things neither has alone -- a
description of how the parts relate, and geometry that finds where:

1. **LLM scene reasoning** (`scene_reasoning.py`, `scripts/reason_about_objects.py`)
   looks at photo crops plus the measured dimensions and returns, per
   object, its parts: one `base`, and per kind of removable part its
   relation (`inserted` in bores, `resting_inside` an open parent,
   `resting_on_top`), parent, count, dominant colour, material (with the
   visual evidence), density, friction, and solid fraction.
2. **Geometric split** (`decompose.py`, `scripts/decompose_objects.py`) on
   the object's fine TSDF:
   - inserted: pieces protruding above the parent's top are the parts;
     each is extended down its bore to the bore floor *as seen in the
     parent's empty bores* (only floors a camera saw count), and the parent
     gets its bores back;
   - resting inside: per height slice, blobs enclosed by the parent's walls;
     extended down to the parent's floor where a camera saw it;
   - resting on top: cut where the footprint widens going up (overhang);
   - several kinds in one role (a brass and a steel weight): pieces are
     assigned to kinds by their colour, sampled from the photos with a
     depth visibility check.
3. **Mass** per part = density x filled-envelope volume x solid fraction.
   The filled envelope closes every cavity slice by slice, seen or not, so
   the LLM's solid fraction always applies to the same computable volume.
4. **Assets**: `export_assets.py` writes one USD per part (rigid body,
   mass, convex colliders, physics material, relation/parent/removable,
   and where each value came from).

Test scenes: `scripts/make_composite_scene.py` (`composite_objects.py`) --
tube racks, a mini centrifuge with a loaded rotor, a tray with metal
weights, a capped bottle, a lidded box, a pegboard with pins; each part a
separate mesh with its own material (plastics, aluminium, steel, brass,
wood, rubber, ceramic), so counts, materials and masses are exact truth.
`scripts/score_composites.py` scores them; for the lab scene,
`isaac_validation/lab_ground_truth_composites.py` builds the same truth
(tubes in the holder, tips in the rack, the centrifuge's housing vs rotor).

LLM backends (`reason_about_objects.py --backend`):
- `claude-cli`: headless Claude Code (`claude -p`) on a Claude subscription
  login (Pro/Max) -- no API credits. Locked down: Read (the saved crops)
  and WebSearch only, `--restricted`, no permission prompts, no session
  saved, clean environment. One-time setup: `claude auth login --claudeai`.
- `api`: the Claude API (credits): `~/.anthropic/api_key` you create,
  `ANTHROPIC_API_KEY`, `ant auth login`, or workload identity federation.
- `auto` (default): `api` when API credentials exist, else `claude-cli`.

**Results with the real LLM** (Claude Opus 5.5 via headless Claude Code,
Isaac RGB-D captures, marker poses refined by depth ICP):

| assembly | parts found | materials right | mass |
|---|---|---|---|
| lab: test-tube holder + 20 glass tubes | 21/21 | 21/21 | tubes 1.6 g vs 1.3 g (+21%), holder -17% |
| lab: tip rack + 100 tips | 101/101 | 101/101 (both polypropylene) | rack +406%, tips +44% (tips called "tubes") |
| lab: centrifuge housing + rotor (interior) | 2/2, rotor within 0.9 mm | sim says "metal"; LLM: ABS / PP | rotor -52% |
| composite: tube rack 3x5 + 11 tubes | 12/12 | 12/12 | total -7% |
| composite: pegboard + 2 brass + 2 steel pins | 5/5 | 5/5 | total -6%, pins within 1% |
| composite: tray + Al/steel weights | 4/4 | 1/4 (all called steel) | total +88% |
| composite: lidded box | 2/2 | 1/2 (wood called cardboard) | total -41% |
| composite: capped bottle | 2/2 | 1/2 (ceramic called HDPE) | total -80% |
| composite: mini centrifuge + rotor + 3 tubes | 5/5 | 1/5 | total -55% |

Counts, relations and geometry come out right throughout; mass error
follows material identification. Plastics, metals and finishes that look
alike in photos (PP vs PS, aluminium vs steel, glazed ceramic vs HDPE,
pale wood vs cardboard) are where it goes wrong -- the same ambiguity a
person would face from photos alone. Unseen interiors (a capped bottle, a
lidded box) rest on the estimated solid fraction.

## Status

- [x] Stage 1: camera pose recovery from fiducial markers — validated on a
  synthetic multi-view capture against known ground truth (see
  `CAPTURE_GUIDE.md`'s "Validating the pipeline" section); not yet run on
  a real photo
- [x] Stage 2: open-set segmentation (SAM) + open-vocabulary labeling
  (CLIP) + platform-region filtering using Stage 1's camera pose —
  working end-to-end on synthetic scenes (markers + proxy objects
  together); not yet run on real lab-ware, so label quality against
  real objects is unverified
- [x] Stages 3-5: vote-based visual hull over the platform volume ->
  3D connected components (= objects; this also does cross-view
  association) -> label, measure, mesh -> `scene.json` + one OBJ per
  object. Scored against the real Isaac Sim scene meshes in
  `isaac_validation/` (see below).
- [x] Template fitting to image edges, sub-pixel, on the 12MP Isaac capture
  (scored against the scene's own meshes; every listed dimension within 0.5mm):

  | object (true) | family | size error (mm) | surface med/p90 |
  |---|---|---|---|
  | waste bin 90x90x70, 2mm wall | open_box | -0.19, -0.01, +0.02; wall +0.03 | 0.84 / 1.53 |
  | sample tray 52x24x14.4, 2x5 wells | plate_with_well_grid | -0.46, -0.15, +0.03; pitch +-0.02 | 0.36 / 0.61 |
  | tip rack 100x100x55 | box | -0.17, +0.09, +0.14 | 0.95 / 1.73 |
  | tube holder 316x254.5x40 | rack_with_tube_grid | -0.34, +0.24, +0.06 | 1.56 / 2.81 |
  | 20 tubes 7x58 (each a separate asset) | (grid) | dia -0.23, length +0.15; centers 0.2-0.3 | 0.25 / 0.5 |

  How (each step fixed a measured failure): silhouette distance map for a
  coarse start, replaced for boxes by an edge-evidence coordinate scan
  (height first; each side independently); then ICP-style refinement on
  ALL visible template edges (silhouette + creases) with a sub-pixel,
  orientation-checked search along each edge normal. Interior features
  are modelled, not ignored (well openings, tubes standing in a rack),
  otherwise their strong edges capture the fit. Tube radius comes from the
  side silhouette lines only; the top rim has its own nuisance radius.
- [~] Shape hints (grid rows/cols, rough pitch) currently come from
  labelled MANUAL STAND-INS (`isaac_validation/*_standin.json`) for what
  LLM scene reasoning will provide; every value they give is refitted
- [~] LLM scene reasoning: implemented, dry-run tested; live runs need
  ANTHROPIC_API_KEY (or an `ant auth login` profile)
- [~] Calibration: with a precisely known marker layout, camera centers
  are within 1.3mm (median). A tape-measured layout costs ~3mm in object
  size; bundle adjustment halves that but doesn't close it -> for real
  captures, use a reference whose geometry is exact by construction
  (printed calibration board), not hand-measured positions
- [x] Template assets: clean meshes from fits (wells, bores, open
  interiors cut by booleans; tubes as separate assets), USD with exact
  colliders for fixtures and convex hulls for loose parts, provenance of
  every dimension (measured / hint / assumed)
- [ ] Rotationally symmetric template (centrifuge) and freeform refinement
  (pipette) -- these still export visual-hull geometry (~1-2mm surface)
- [ ] Tips in the tip rack as separate assets (same pattern as tubes)
- [x] Depth path: TSDF fusion, sensor models, silhouette refinement on a depth start (see "Depth (RGB-D) path")
- [~] One command (`scripts/run_pipeline.py`): poses, segmentation, depth fusion or hull, refinement, LLM, assets. Template fits are not yet chosen automatically from the LLM's shape family

## Layout

| Path | What it is |
|---|---|
| `calibration.py` | Load/validate per-camera intrinsics + distortion |
| `markers.py` | Marker layout definition, detection, per-image PnP |
| `capture.py` | Multi-view capture session manifest (images ↔ cameras) |
| `pose_solver.py` | Ties calibration + markers + capture into per-image camera poses |
| `geometry.py` | Shared rotation/quaternion helpers used across every stage |
| `qa.py` | Stage 1 reprojection diagnostics, annotated overlay images |
| `segmentation.py` | Stage 2a: SAM open-set mask proposals (no text prompt, no catalog) |
| `labeling.py` | Stage 2b: CLIP zero-shot label guesses per mask (a hint, not a lookup) |
| `platform_filter.py` | Stage 2c: keep only masks over the platform, using a Stage 1 camera pose |
| `reconstruction.py` | Stages 3-5: vote-based visual hull, 3D components, labels, meshes |
| `bundle_adjust.py` | Joint refinement of all camera poses + marker bodies, with the measured layout as a prior |
| `scene_reasoning.py` | LLM (Claude + web search) per object: identity, shape family, sourced dimensions, parts, material |
| `template_fitting.py` | Sub-pixel fit of a parametric shape template to image edges across all views |
| `depth_sensor.py` | Depth I/O and sensor noise models (presets or JSON) |
| `depth_fusion.py` | TSDF + free-space fusion, facing-aware second pass, components |
| `evaluation.py` | Scoring vs ground-truth meshes, incl. observable-only surface |
| `random_objects.py` | Procedural complex test objects (holes, cavities, overhangs, handles) |
| `synthetic_depth.py` | CPU ray-cast depth + shaded RGB of meshes (tests without the GPU) |
| `composite_objects.py` | Procedural multi-part test objects with per-part materials (exact mass truth) |
| `decompose.py` | Split a fused object into parts from a relation description (inserted / inside / on top) |
| `depth_registration.py` | Depth ICP pose refinement anchored to the bench plane and marker quads |
| `asset_export.py` | USD + OBJ assets: V-HACD colliders (<=64 verts/hull), mass, physics material |
| `isaac_validation/` | End-to-end validation against the real Isaac Sim lab scene (isolated, see below) |
| `config/` | Example marker layout + intrinsics files |
| `scripts/` | CLI entry points, incl. the synthetic capture renderer/validator |
| `tests/` | Synthetic-data tests (no real photos needed to validate the geometry) |

Stage 1 (`calibration.py`/`markers.py`/`capture.py`/`pose_solver.py`/
`geometry.py`/`qa.py`) runs in the lightweight `.venv-cad` environment.
Stage 2 (`segmentation.py`/`labeling.py`) needs GPU torch + transformers
and runs in a separate `.venv-vision` environment — deliberately not
merged into `.venv-cad`, which has no other reason to carry a multi-GB
ML stack. `platform_filter.py` has no ML dependency and works in either.

## Validating against the Isaac Sim scene

`isaac_validation/` renders the actual biotech lab scene (holder + tubes,
centrifuge, pipette, tray, tip rack, waste bin) with ArUco markers added,
from a 25-camera rig, runs the full pipeline on the images only, and
scores the reconstructed objects against the scene's own meshes.

It is isolated from the simulation code in the same repo (developed
concurrently by another agent): `isaac_validation/sim_snapshot/` is a
frozen copy of `lab_scene.py`, `liquids.py` and `assets/`, with only its
two asset paths repointed. Nothing outside `vision_pipeline/` is read or
written. The repo ignores `assets/` everywhere, so the snapshot's assets
are not committed: before an Isaac render on a fresh clone, copy the
project's `assets/` into `isaac_validation/sim_snapshot/assets/`.

```bash
V=biotech-lab-robotics/vision_pipeline/isaac_validation/out
.venv-isaac/Scripts/python.exe  biotech-lab-robotics/vision_pipeline/isaac_validation/render_isaac_validation_capture.py --out $V
.venv-cad/Scripts/python.exe    biotech-lab-robotics/vision_pipeline/scripts/solve_camera_poses.py --manifest $V/manifest.json     --intrinsics $V/camera_intrinsics.json --marker-layout $V/marker_layout.json --out-dir $V/poses
.venv-cad/Scripts/python.exe    biotech-lab-robotics/vision_pipeline/isaac_validation/check_camera_positions.py --capture-dir $V
.venv-vision/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/segment_capture.py --capture-dir $V --platform-w 1.4 --platform-d 0.7
.venv-cad/Scripts/python.exe    biotech-lab-robotics/vision_pipeline/scripts/reconstruct_scene.py --capture-dir $V --platform-w 1.4 --platform-d 0.7
.venv-cad/Scripts/python.exe    biotech-lab-robotics/vision_pipeline/isaac_validation/compare_to_ground_truth.py --capture-dir $V
```
