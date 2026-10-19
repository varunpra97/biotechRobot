"""Render the REAL biotech-lab-robotics Isaac Sim scene (actual CAD
labware: tube holder, test tubes, centrifuge, pipette, sample tray,
waste bin -- not simplified proxies) with ArUco markers added at known
positions, from its existing multi-view cameras, for validating the
vision pipeline's reconstructed measurements against real, documented
object dimensions.

ISOLATED from biotech-lab-robotics entirely: imports lab_scene.build_scene()
from sim_snapshot/ (a frozen copy made alongside this script, with its own
copy of assets/ -- see sim_snapshot/lab_scene.py's header), not from the
other agent's live repo. Nothing under biotech-lab-robotics/ is read or
written by this script, so it can run no matter what that repo's
concurrent work is doing. If the snapshot needs to pick up a later
upstream change, re-copy it deliberately; it does not track the original.

Must run with the Isaac venv:
  .venv-isaac/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/render_isaac_validation_capture.py \
      --out biotech-lab-robotics/vision_pipeline/isaac_validation/out

Camera poses are deliberately NOT exported as ground truth here (that
would need converting USD's camera convention -- looks down local -Z,
+Y up -- into OpenCV's -- +Z forward, +Y down -- an easy place to
silently introduce a sign bug). Stage 1 instead recovers every camera's
pose itself from the rendered markers, exactly as it would for a real
photo; only marker positions (which this script places, so they're
exact by construction) and camera intrinsics (which have no axis-
convention ambiguity) are exported as inputs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

VISION_PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_PIPELINE_DIR))  # for markers.py etc.
SIM_SNAPSHOT_DIR = Path(__file__).resolve().parent / "sim_snapshot"
sys.path.insert(0, str(SIM_SNAPSHOT_DIR))

import numpy as np

from markers import DEFAULT_DICTIONARY, corners_from_center_axes

FLAT_MARKER_SIDE_M = 0.08
# Lifted off the surface they sit on: a quad exactly coplanar with the
# bench (or a cube face) z-fights with it and renders as surface-white.
# 1mm is recorded in marker_layout.json, which is derived from the same specs.
MARKER_LIFT_M = 0.001
# Calibration cubes: a marker on the top AND all four sides. Flat markers
# alone are seen edge-on by low (~15 deg) views -- 3 of 8 low views failed
# to pose -- and low views are the ones that bound object height. A cube's
# visible faces are also non-coplanar, which conditions the pose well
# (all-coplanar markers left cameras 36-127mm off).
CUBE_SIDE_M = 0.12
CUBE_MARKER_SIDE_M = 0.10
CAPTURE_RESOLUTION = (4032, 3024)  # a phone's 12MP main camera; ~0.43mm/px at the bench from 1.5m
SCENE_CAMERA_RESOLUTION = (640, 480)  # what lab_scene.build_scene() gives its policy cameras
# Depth PNGs: uint16 in 0.1mm units (max 6.5535m, 0 = no return). Lossless
# PNG compresses a 12MP depth map ~5x better than float32 .npy, and 0.05mm
# rounding is far below any sensor's noise.
DEPTH_PNG_SCALE_M = 1e-4


def save_depth_png(path: Path, depth_m: np.ndarray) -> None:
    import cv2
    q = np.where(np.isfinite(depth_m) & (depth_m > 0), np.round(depth_m / DEPTH_PNG_SCALE_M), 0)
    q[q > 65535] = 0
    cv2.imwrite(str(path), q.astype(np.uint16))


def _cube_faces(center_bottom, side):
    """(normal, x_axis, y_axis) per face, chosen so x_axis x y_axis = outward
    normal: seen from outside, x is right and y is up, i.e. the marker is
    not mirrored (a mirrored ArUco does not decode)."""
    return {
        "top":   ((0, 0, 1),  (1, 0, 0),  (0, 1, 0)),
        "front": ((0, -1, 0), (1, 0, 0),  (0, 0, 1)),
        "back":  ((0, 1, 0),  (-1, 0, 0), (0, 0, 1)),
        "left":  ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "right": ((1, 0, 0),  (0, 1, 0),  (0, 0, 1)),
    }


def marker_specs(lab_scene) -> dict[int, dict]:
    """Every marker: world center, in-plane axes, side length. Placement was
    chosen against build_platform()'s layout (labware sits in world
    x in [-0.43, 0.54], y in [1.45, 1.8]), so markers and cubes use clear
    bench: the back half (y > 1.8) and the side edges.
    """
    cx, cy = lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY
    top = lab_scene.BENCH_Z + 0.04
    specs = {}
    for mid, (dx, dy) in enumerate([(-0.62, 0.0), (0.62, 0.0), (-0.2, 0.27), (0.2, 0.27)]):
        specs[mid] = {"center": (cx + dx, cy + dy, top + MARKER_LIFT_M), "x_axis": (1, 0, 0),
                      "y_axis": (0, 1, 0), "side": FLAT_MARKER_SIDE_M}
    next_id = 4
    for cube_center_xy in cube_positions(lab_scene):
        c = np.array([*cube_center_xy, top + CUBE_SIDE_M / 2])
        for name, (n, xa, ya) in _cube_faces(c, CUBE_SIDE_M).items():
            n = np.array(n, float)
            specs[next_id] = {"center": tuple(c + n * (CUBE_SIDE_M / 2 + MARKER_LIFT_M)),
                              "x_axis": xa, "y_axis": ya, "side": CUBE_MARKER_SIDE_M}
            next_id += 1
    return specs


def cube_positions(lab_scene) -> list[tuple[float, float]]:
    cx, cy = lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY
    return [(cx - 0.55, cy + 0.24), (cx + 0.55, cy + 0.24)]


def marker_corners_world(spec) -> np.ndarray:
    return corners_from_center_axes(center=spec["center"], x_axis=spec["x_axis"],
                                    y_axis=spec["y_axis"], side_length_m=spec["side"])


def add_markers_to_stage(lab_scene, stage, marker_png_dir: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    marker_png_dir.mkdir(parents=True, exist_ok=True)
    specs = marker_specs(lab_scene)

    top = lab_scene.BENCH_Z + 0.04
    for i, (x, y) in enumerate(cube_positions(lab_scene)):
        lab_scene._box(f"/World/CalibMarkers/Cube{i}", (CUBE_SIDE_M,) * 3,
                       (x, y, top + CUBE_SIDE_M / 2), "white", collision=False)

    for marker_id, spec in specs.items():
        corners = marker_corners_world(spec)  # TL, TR, BR, BL

        # Render the marker bitmap to a PNG this process can reference as
        # a USD texture asset (needs a real file on disk, not an
        # in-memory array).
        import cv2
        dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_DICTIONARY)
        bitmap = cv2.aruco.generateImageMarker(dictionary, marker_id, 256)
        bitmap_bgr = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2BGR)
        # Absolute path: USD resolves a relative asset path against the
        # (anonymous, in-memory) stage layer, not the process's cwd.
        png_path = (marker_png_dir / f"marker_{marker_id}.png").resolve()
        cv2.imwrite(str(png_path), bitmap_bgr)

        mesh_path = f"/World/CalibMarkers/Marker{marker_id}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        # UV v-axis: this renderer's UsdUVTexture follows the OpenGL
        # convention, v=0 is the image's BOTTOM row. Getting this backwards
        # renders a mirrored marker, which ArUco cannot decode at all
        # (verified: detection failed on every view until flipped).
        # TL/TR at v=1, BR/BL at v=0.
        texcoords = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        texcoords.Set([Gf.Vec2f(0, 1), Gf.Vec2f(1, 1), Gf.Vec2f(1, 0), Gf.Vec2f(0, 0)])

        mat_path = f"{mesh_path}/Material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        tex = UsdShade.Shader.Define(stage, f"{mat_path}/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(png_path.as_posix())
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/STReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    print(f"added {len(specs)} markers ({len(cube_positions(lab_scene))} calibration cubes), PNGs in {marker_png_dir}")



PATH_TRACING_SPP_PER_FRAME = 16
PATH_TRACING_TOTAL_SPP = 64


def configure_renderer() -> None:
    """Path-traced, native resolution, no DLSS.

    Isaac defaults to real-time rendering with DLSS (/rtx/post/aa/op=3): it
    renders at reduced resolution and upscales, softening and shifting
    edges -- corrupting the sub-pixel geometry this capture exists to
    measure -- and at 12MP fails outright (NGX can't allocate). But turning
    DLSS off in real-time mode also removes its denoising, leaving heavy
    grain. Path tracing samples each pixel over its area (like a sensor),
    accumulates PATH_TRACING_TOTAL_SPP samples while the camera is still,
    and denoises with OptiX instead.

    Applied just before capturing: SimulationApp's "anti_aliasing" key and a
    set right after startup both got overridden later.
    """
    import carb
    settings = carb.settings.get_settings()
    settings.set("/rtx-transient/post/aa/limitedOps", False)
    settings.set("/rtx/post/aa/op", 0)
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", PATH_TRACING_SPP_PER_FRAME)
    settings.set("/rtx/pathtracing/totalSpp", PATH_TRACING_TOTAL_SPP)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 1)


def camera_world_position(stage, prim_path: str) -> np.ndarray:
    """Camera center from its USD world transform. A position has no
    camera-axis convention to get wrong (unlike orientation), so it's a
    safe exact ground truth for Stage 1's recovered camera centers."""
    from pxr import Usd, UsdGeom
    M = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return M[3, :3]  # row-vector convention: translation is the last row


class SingleCaptureCamera:
    """One camera prim with one render product, re-posed for every view.

    A render product per view (even destroyed after use) leaked GPU memory:
    at 4032x3024 a 12GB card ran out after 9 of 25 views. Instead, each
    view's camera prim only defines a pose and lens; this camera copies
    that prim's world transform and focal length, renders, and moves on.
    Both are USD cameras, so copying the matrix needs no axis conversion.
    """

    PATH = "/World/CalibCapture/Camera"

    def __init__(self, simulation_app, stage, resolution: tuple[int, int]):
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom

        self.app, self.stage = simulation_app, stage
        self.usd_cam = UsdGeom.Camera.Define(stage, self.PATH)
        self.usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        self.sensor = Camera(self.PATH, resolution=resolution)
        self.sensor.initialize()
        # Depth along the optical axis (OpenCV-style Z, meters) -- what an
        # RGB-D sensor registered to the colour camera reports. Perfect
        # here; sensor noise is applied downstream (depth_sensor.py), so one
        # render serves every noise level.
        self.sensor.add_distance_to_image_plane_to_frame()

    def capture(self, view_prim_path: str,
                settle_frames: int = PATH_TRACING_TOTAL_SPP // PATH_TRACING_SPP_PER_FRAME + 6
                ) -> tuple[np.ndarray, np.ndarray, dict]:
        from pxr import Gf, Usd, UsdGeom

        view = UsdGeom.Camera(self.stage.GetPrimAtPath(view_prim_path))
        M = UsdGeom.Xformable(view).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Rebuilt every time: Camera.initialize() re-authors this prim's
        # xform ops, so a cached op handle goes stale.
        UsdGeom.Xformable(self.usd_cam).MakeMatrixXform().Set(Gf.Matrix4d(M))
        self.usd_cam.GetFocalLengthAttr().Set(view.GetFocalLengthAttr().Get())
        self.usd_cam.GetHorizontalApertureAttr().Set(view.GetHorizontalApertureAttr().Get())
        configure_renderer()
        for _ in range(settle_frames):
            self.app.update()
        frame = self.sensor.get_current_frame()
        rgb = frame["rgb"]
        if rgb is None or rgb.size == 0 or rgb.std() < 5:
            raise RuntimeError(f"{view_prim_path}: no usable frame (check the log for RTX errors)")
        depth = np.asarray(frame["distance_to_image_plane"], dtype=np.float32).squeeze()
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError(f"{view_prim_path}: depth {depth.shape} doesn't match rgb {rgb.shape[:2]}")
        # Camera() keeps pixels square by rewriting verticalAperture; K is
        # read from the capture camera, i.e. what was actually rendered.
        return rgb, depth, camera_intrinsics_from_usd(self.stage, self.PATH, (rgb.shape[1], rgb.shape[0]))


def add_validation_cameras(lab_scene, stage) -> tuple[dict, dict]:
    """Multi-view rig for reconstruction, created in THIS process's stage
    only (lab_scene's own 6 cameras are framed for the robot policy, not
    for covering the platform from all sides): three 8-camera rings at
    15/35/55 deg elevation plus one top-down. (A close "detail" camera was
    dropped: it never saw a marker, so Stage 1 could never pose it.)
    Higher resolution than the policy cameras because 7mm tubes and 8mm
    bores need pixels to be measured at all.
    """
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom

    cx, cy = lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY
    bench_top = lab_scene.BENCH_Z + 0.04
    target = np.array([cx, cy, bench_top + 0.05])

    # Three rings. The low one is what bounds object HEIGHT: a visual hull
    # only gets height from views that see an object's side against the
    # background. With a single 50-deg ring, a flat 8cm marker carved to
    # 60mm tall and a 70mm bin to 150mm (phantom cap ~ half-width x
    # tan(elevation)). The middle ring is offset 22.5 deg in azimuth for
    # silhouette diversity.
    specs = {}
    for elev_deg, az_offset in ((15, 0.0), (35, 22.5), (55, 0.0)):
        for i in range(8):
            az_deg = i * 45 + az_offset
            az, el, dist = np.radians(az_deg), np.radians(elev_deg), 1.5
            eye = target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            name = f"el{elev_deg:02d}_az{az_deg:05.1f}".replace(".", "p")  # "." is illegal in USD prim names
            specs[name] = (eye, target, 18.0)
    # Small offset in y so the view direction isn't exactly parallel to
    # world up (look-at is degenerate straight down).
    specs["top_down"] = (np.array([cx, cy + 0.01, bench_top + 1.9]), np.array([cx, cy, bench_top]), 18.0)

    # Camera prims only -- no render products yet. Each render product
    # holds a full set of RTX buffers; 25 at 1280x960 at once ran a 12GB
    # GPU out of memory (one view came back blank), so main() renders
    # them one at a time through a single SingleCaptureCamera.
    cams = {}
    for name, (eye, tgt, focal) in specs.items():
        rep.create.camera(position=tuple(eye), look_at=tuple(tgt), name=name, focal_length=focal)
        prim_path = f"/Replicator/{name}_Xform/{name}"
        UsdGeom.Camera(stage.GetPrimAtPath(prim_path)).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        cams[name] = prim_path
    # Eye positions only (no orientation): a position has no camera-axis
    # convention to get wrong, so it's a safe exact ground truth for
    # checking Stage 1's recovered camera centers.
    gt_positions = {name: (eye - np.array([cx, cy, bench_top])).tolist()
                    for name, (eye, _, _) in specs.items()}
    return cams, gt_positions



def export_ground_truth_meshes(lab_scene, stage, out_dir: Path) -> dict:
    """Write the scene's actual rendered geometry, in the platform-local
    frame, as one OBJ per top-level object under /World/Platform (e.g. all
    100 tips -> Tips.obj). This is the geometry the cameras saw, taken from
    the live stage after the capture, so it reflects final poses -- the
    reference the reconstruction is scored against. Collision meshes
    (purpose=guide) and the platform itself (Stand/Bench) are skipped.
    """
    import trimesh
    from pxr import Usd, UsdGeom

    origin = np.array([lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY, lab_scene.BENCH_Z + 0.04])
    root = "/World/Platform"
    skip = {"Stand", "Bench", "WorkLight"}
    groups: dict[str, list] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        top = str(prim.GetPath())[len(root) + 1:].split("/")[0]
        if top in skip:
            continue
        mesh = UsdGeom.Mesh(prim)
        raw_pts, raw_counts, raw_idx = (mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                                         mesh.GetFaceVertexIndicesAttr().Get())
        if not raw_pts or not raw_counts:
            continue  # unauthored (e.g. liquid placeholders with no geometry yet)
        pts = np.array(raw_pts, dtype=np.float64)
        counts, idx = np.array(raw_counts), np.array(raw_idx)
        if not np.all(counts == 3):
            continue
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world = np.c_[pts, np.ones(len(pts))] @ M  # USD matrices are row-vector convention
        groups.setdefault(top, []).append(trimesh.Trimesh(world[:, :3] - origin, idx.reshape(-1, 3), process=False))

    mesh_dir = out_dir / "ground_truth_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, parts in groups.items():
        m = trimesh.util.concatenate(parts)
        m.export(mesh_dir / f"{name}.obj")
        lo, hi = m.bounds
        summary[name] = {"mesh": f"ground_truth_meshes/{name}.obj",
                         "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                         "extents": (hi - lo).tolist()}
    print(f"exported {len(summary)} ground-truth object meshes to {mesh_dir}")
    return summary


def camera_intrinsics_from_usd(stage, prim_path: str, resolution: tuple[int, int]) -> dict:
    """K from the USD camera's own authored focal length / aperture --
    no axis-convention involved (unlike extrinsics), so this is safe to
    trust directly as ground truth for Stage 1's --intrinsics input.
    """
    from pxr import UsdGeom

    cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    focal = cam.GetFocalLengthAttr().Get()
    h_ap = cam.GetHorizontalApertureAttr().Get()
    v_ap = cam.GetVerticalApertureAttr().Get()
    res_x, res_y = resolution
    if not v_ap:
        v_ap = h_ap * res_y / res_x
    fx = focal * res_x / h_ap
    fy = focal * res_y / v_ap
    return {
        "K": [[fx, 0.0, res_x / 2.0], [0.0, fy, res_y / 2.0], [0.0, 0.0, 1.0]],
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "image_size": [res_x, res_y],
    }


# Ground truth object dimensions, pulled directly from lab_scene.py's own
# documented constants/comments (not re-measured here) -- what Stage 4's
# output gets compared against.
GROUND_TRUTH_OBJECTS = {
    "waste_bin": {
        "shape": "box", "dims_m": {"w": 0.090, "d": 0.090, "h": 0.070},
        "source": "lab_scene.WASTE_BIN_XY comment: '90 x 90 x 70 mm open box'",
    },
    "pipette_0": {
        "shape": "elongated_body", "dims_m": {"w": 0.0188, "d": 0.181, "h": 0.056},
        "source": "lab_scene.build_platform(): objects['pipette_0']['size']",
    },
    "sample_tray": {
        "shape": "box_with_wells",
        "source": "lab_scene.build_platform(): objects['tray']['size'] (read from the "
                  "capture's meta, authored bbox differs per mesh -- not hand-copied here)",
    },
    "test_tube": {
        "shape": "cylinder", "dims_m": {"od": 0.007, "length": 0.058},
        "source": "biotech-lab-robotics/README.md: '20 test tubes: 7mm OD x 58mm'",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig", choices=["scene5", "ring25"], default="ring25",
                     help="ring25 (default): 3 elevation rings (15/35/55 deg) + top. scene5: exactly the 5 original lab_scene cameras -- arm_cam0/1 "
                          "(front, behind the arm), platform_cam0/1 (back), top -- at their native "
                          "640x480 (too few views for an accurate visual hull; kept for comparison).")

    ap.add_argument("--only", nargs="+", help="render just these views (quick quality checks)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.stage as stage_utils
    import lab_scene
    from PIL import Image

    arm, cams, objects = lab_scene.build_scene(simulation_app)
    print(f"scene built: {len(cams)} cameras, {len(objects)} loggable objects")

    stage = stage_utils.get_current_stage()
    # The real capture is done with no robot in view; hide it here (this
    # process's stage only -- the snapshot's scene code is untouched).
    from pxr import UsdGeom
    for path in ("/World/Arm", "/World/Pedestal"):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    # Render the CAD geometry as authored. A UsdGeom.Mesh without an
    # explicit subdivisionScheme defaults to catmullClark, and the renderer
    # then smooths it across hard edges: the 40mm-tall box tube holder came
    # out pillow-shaded (no visible top crease), and fitted box templates
    # ran consistently ~0.1-0.6mm small. lab_scene never sets the scheme;
    # set it here, in this process's stage only.
    from pxr import Usd
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    add_markers_to_stage(lab_scene, stage, out_dir / "marker_textures")
    if args.rig == "scene5":
        capture_cams = {k: prim for k, (prim, _) in cams.items() if k != "wrist"}
        resolution = SCENE_CAMERA_RESOLUTION
    else:
        capture_cams, _ = add_validation_cameras(lab_scene, stage)
        resolution = CAPTURE_RESOLUTION
    # Let the renderer pick up the prims authored above BEFORE any render
    # product is torn down: destroying the scene's cameras first dropped
    # the pending sync, and the markers and stands silently never rendered
    # (no error, just absent from every image).
    for _ in range(4):
        simulation_app.update()
    # lab_scene's own policy cameras already hold render products in this
    # process; release them (this process only) to leave the GPU to ours.
    for _, scene_cam in cams.values():
        scene_cam.destroy()

    configure_renderer()

    cameras_json = {}
    manifest_shots = []
    if args.only:
        capture_cams = {k: v for k, v in capture_cams.items() if k in args.only}
    capture_camera = SingleCaptureCamera(simulation_app, stage, resolution)
    depth_dir = out_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    for name, prim_path in capture_cams.items():
        rgb, depth, intr = capture_camera.capture(prim_path)
        Image.fromarray(rgb).save(str(images_dir / f"{name}.png"))
        save_depth_png(depth_dir / f"{name}.png", depth)
        cameras_json[name] = intr
        manifest_shots.append({"image": f"images/{name}.png", "depth": f"depth/{name}.png", "camera_id": name})
        finite = np.isfinite(depth) & (depth > 0)
        print(f"  captured {name}: {rgb.shape[1]}x{rgb.shape[0]}, depth {depth[finite].min():.3f}-"
              f"{depth[finite].max():.3f} m")

    (out_dir / "camera_intrinsics.json").write_text(
        json.dumps({"cameras": cameras_json}, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({
        "shots": manifest_shots,
        "depth": {"format": "uint16 png", "scale_m": DEPTH_PNG_SCALE_M, "invalid": 0,
                  "kind": "z (distance to the image plane), registered to the colour image",
                  "noise": "none (rendered); see depth_sensor.py for sensor models",
                  # the depth annotator samples each pixel at its corner, half a
                  # pixel off K's pixel-center convention (measured; see reconstruct_depth.py)
                  "pixel_offset_px": 0.5}}, indent=2))

    cx, cy, bench_top_z = lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY, lab_scene.BENCH_Z + 0.04
    origin = np.array([cx, cy, bench_top_z])
    specs = marker_specs(lab_scene)
    marker_layout = {"markers": {str(mid): {"corners_3d": (marker_corners_world(spec) - origin).tolist()}
                                 for mid, spec in specs.items()},
                     # markers on one calibration cube: rigid relative to each other
                     "rigid_groups": [list(range(4 + 5 * i, 9 + 5 * i)) for i in range(len(cube_positions(lab_scene)))]}
    (out_dir / "marker_layout.json").write_text(json.dumps(marker_layout, indent=2))
    (out_dir / "ground_truth_camera_positions.json").write_text(json.dumps(
        {"frame": "platform-local (origin at bench-top center, Z up, meters)",
         "positions": {n: (camera_world_position(stage, pp) - origin).tolist() for n, pp in capture_cams.items()}},
        indent=2))
    (out_dir / "ground_truth_objects.json").write_text(
        json.dumps(GROUND_TRUTH_OBJECTS, indent=2))
    (out_dir / "ground_truth_meshes.json").write_text(
        json.dumps(export_ground_truth_meshes(lab_scene, stage, out_dir), indent=2))
    (out_dir / "platform_dims.json").write_text(json.dumps(
        {"platform_w": lab_scene.PLATFORM_W, "platform_d": lab_scene.PLATFORM_D}, indent=2))

    print(f"\nwrote images to {images_dir}")
    print(f"wrote {out_dir / 'marker_layout.json'}, {out_dir / 'camera_intrinsics.json'}, "
          f"{out_dir / 'manifest.json'}, {out_dir / 'ground_truth_objects.json'}")

    # Hard exit: simulation_app.close() hung indefinitely after a 12MP
    # path-traced capture (every output already written), leaving the
    # process holding GPU memory. Everything is on disk; process exit
    # releases the GPU.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
