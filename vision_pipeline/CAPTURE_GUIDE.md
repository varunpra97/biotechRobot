# Capture Guide: Markers → Calibration → Photo Set → Stage 1

A repeatable procedure for one real-world capture session. Follow in
order — each step's output feeds the next.

## 1. Print and place ArUco markers

```bash
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/generate_markers.py \
    --ids 0 1 2 3 4 5 --side-mm 50 --out biotech-lab-robotics/vision_pipeline/out/markers_to_print
```

- Print each PNG at **100% / "actual size"** — not "fit to page" or
  "shrink to fit", which silently rescales the marker.
- **Measure the printed marker with a ruler before trusting anything
  else.** Each page prints a 100mm reference line next to the marker
  specifically so you can catch printer rescaling. If the marker's
  printed side isn't the nominal 50mm, use the *measured* value (not
  50mm) in `marker_layout.json` below — the pipeline's absolute scale
  comes entirely from this number.
- Mount each marker flat (tape/glue to a stiff card so it can't curl —
  a curled marker is no longer planar and its corners won't match the
  3D positions you measure).

**How many, where:** a synthetic end-to-end test (see "Validating the
pipeline" below) measured this directly: views where 3 markers were
visible recovered camera pose to <1mm / <0.1°; views with only 1-2
visible degraded to 20-96mm / ~1° under realistic (sub-pixel) corner-
detection noise, and a single marker viewed from far away (>2.5m) can
land on a completely wrong, ambiguous pose despite a deceptively low
reprojection error. This is inherent to sparse-point PnP, not something
tuning the solver fixes.

Practically: don't rely on just the 4 corners. Distribute 6-8 markers
around the full perimeter of the platform (e.g. corners plus edge
midpoints) so that as many ring shots as possible see **3 or more**
simultaneously, not just "at least one". It's cheap to print and place
extra markers; it's expensive to discover after a shoot that half your
views only had marginal pose accuracy.

**Raise two of them off the bench.** Tape two markers on top of blocks
of known height (~12 cm; measure it) near the back corners. If every
marker lies flat on the bench, the target is a single plane, and tilt
and translation nearly trade off against each other: on the Isaac
validation capture, all-coplanar markers left the solved camera 36-127 mm
off despite sub-pixel reprojection error; adding two raised markers
brought the median to ~4 mm. Record a raised marker's center z as the
block height.

Reconstruction (`reconstruct_scene.py`) only uses views whose pose rests
on 3+ markers (`--min-markers`): 1-2 marker poses were 10-64 mm off.

## 2. Measure marker positions → `marker_layout.json`

Pick the platform-frame origin: **center of the bench top surface**,
Z up, meters (see `README.md`'s coordinate convention — this choice is
what lets the output drop into `biotech-lab-robotics/sim` without
re-expressing). Measure each marker's center and facing direction
relative to that origin with a tape measure and level.

Copy `config/marker_layout.example.json` → `config/marker_layout.json`
and fill in one entry per marker (the `center`/`x_axis`/`y_axis` form is
usually easiest to measure by hand):

```json
{
  "markers": {
    "0": {
      "center": [-0.65, -0.30, 0.0],
      "x_axis": [1, 0, 0],
      "y_axis": [0, 1, 0],
      "side_length_m": 0.0498
    }
  }
}
```

`side_length_m` is the value you measured in step 1, in meters (50mm
measured as 49.8mm → `0.0498`), not the nominal one.

## 3. Calibrate the camera

Print a chessboard (any 9x6-inner-corner pattern works, e.g. search
"opencv chessboard pattern 9x6" or generate one — flat, rigid, no curl)
and measure its actual printed square size the same way as step 1.

**Lock the camera's autofocus and zoom before shooting anything** —
calibration photos and the real capture set must use the *exact same*
focal length and focus distance, or the calibrated intrinsics won't
match the capture. On a phone: tap-to-focus once on the bench area,
then switch to manual/AE-AF-lock if your camera app supports it, and
don't touch zoom for the rest of the session.

Shoot 15–20 photos of the chessboard filling different parts of the
frame, at different distances/angles (tilt it — a perfectly fronto-
parallel set is poorly conditioned). Save them to a folder, then:

```bash
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/calibrate_camera.py \
    --images "path/to/chessboard_photos/*.jpg" \
    --pattern-cols 9 --pattern-rows 6 --square-size-m 0.0250 \
    --camera-id phone_main \
    --out biotech-lab-robotics/vision_pipeline/config/camera_intrinsics.json
```

The script prints the RMS reprojection error — it warns above 1px.
If it's high, the usual causes are: board not flat, not enough angle
variety, or focus drifted partway through the set.

## 4. Shoot the scene (same camera settings, do not refocus/zoom)

~25 shots, three rings plus one overhead (this supersedes an earlier
"one ring at 30-45° + top + detail, ~10 shots" recommendation, which the
Isaac validation showed can't measure height):

- **low ring, ~15° elevation**, 8 shots every 45° — this is what bounds
  object HEIGHT. 3D shape comes from silhouettes, and a silhouette only
  says how tall something is when the camera sees its side against the
  background. With only a ~50° ring, a flat 8 cm marker reconstructed as
  60 mm tall and a 70 mm bin as 150 mm.
- **mid ring, ~35°**, 8 shots, offset 22.5° from the others
- **high ring, ~55°**, 8 shots
- **1 straight down**, from high enough that the whole platform fits

A close "detail" shot is not useful unless it also sees 3+ markers:
Stage 1 can't pose it otherwise, and an unposed view contributes nothing.

Keep every marker you're relying on actually visible and in focus in
enough of these — check as you go, it's cheap to retake a shot, much
cheaper than redoing the whole set when Stage 1 reports a camera pose
failed. Save all photos into one folder, e.g. `captures/session1/`.

## 5. Run Stage 1

```bash
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/solve_camera_poses.py \
    --captures captures/session1 \
    --camera-id phone_main \
    --intrinsics biotech-lab-robotics/vision_pipeline/config/camera_intrinsics.json \
    --marker-layout biotech-lab-robotics/vision_pipeline/config/marker_layout.json \
    --out-dir biotech-lab-robotics/vision_pipeline/out/session1_poses
```

Check the printed summary table and `out/session1_poses/overlays/*.jpg`:
green = detected marker corners, red crosses = where the solved pose
reprojects them — they should sit right on top of each other. A `warn`
(>1.5px mean error) or `FAIL` (>3px) flag, or a view with 0 markers
used, means that shot didn't solve reliably; retake it or add another
marker to its field of view before moving on. A *low* reprojection
error is reassuring but not sufficient on its own if a view only has 1
marker (4 points) — that configuration can hit an ambiguous pose that
still reprojects well; the `marker_ids_used`/point count in
`camera_poses.json` matters as much as the error number.

## Validating the pipeline without a real capture

No real photos yet? `scripts/render_synthetic_capture.py` builds a
synthetic multi-view set (ArUco markers actually rendered and detected
via real `cv2.aruco`, not just analytic math) with known ground truth,
deliberately varying angle/distance/noise/blur, including a few shots
designed to be hard. Then `scripts/validate_synthetic_capture.py`
diffs Stage 1's output against that ground truth automatically. Useful
both as a first smoke test and as a regression check after any change
to `markers.py`/`pose_solver.py`:

```bash
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/render_synthetic_capture.py \
    --out biotech-lab-robotics/vision_pipeline/out/synthetic_capture
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/solve_camera_poses.py \
    --manifest biotech-lab-robotics/vision_pipeline/out/synthetic_capture/manifest.json \
    --intrinsics biotech-lab-robotics/vision_pipeline/out/synthetic_capture/camera_intrinsics.json \
    --marker-layout biotech-lab-robotics/vision_pipeline/out/synthetic_capture/marker_layout.json \
    --out-dir biotech-lab-robotics/vision_pipeline/out/synthetic_capture/poses
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/validate_synthetic_capture.py \
    --capture-dir biotech-lab-robotics/vision_pipeline/out/synthetic_capture
```
