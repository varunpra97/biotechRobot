"""Biotech lab scene in Isaac Sim: 4 chambers, dual-arm mobile platform, 15 cameras.

Room (10m x 6m x 3m) with 4 biosafety-style chambers (2 per side, facing a
central 2m walkway). Each chamber has a pull-up window (posed open), latch,
test tube rack (3x4), 2 plungers, and a sample tray. A mobile platform
carrying 2 Franka arms slides along the walkway.

Cameras (15 total):
  - 8 chamber cameras (2 per chamber, looking at the work surface)
  - 2 wrist cameras (static viewpoints overlooking each arm's workspace)
  - 1 platform camera (on the base, views the arms)
  - 4 room corner cameras (1280x960, track the mobile base)

Usage (Isaac venv, repo root on Windows box):
    python sim\\lab_scene.py                  # GUI, animated
    python sim\\lab_scene.py --headless       # one animated pass, no window
    python sim\\lab_scene.py --capture        # save PNGs from all 15 cameras
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Layout constants
# ----------------------------------------------------------------------------
ROOM_X, ROOM_Y, ROOM_Z = 10.0, 6.0, 3.0      # room size (m)
WALKWAY_HALF = 1.0                            # walkway |y| < 1
CHAMBER_W, CHAMBER_D, CHAMBER_H = 2.0, 1.2, 1.8
BENCH_Z = 0.9                                 # work surface height
CHAMBER_XS = (-2.5, 2.5)                      # chamber centers along x
CHAMBER_Y = 2.4                               # |y| center (front face at 1.8)

FRANKA_USD = str(
    Path(r"C:\Users\varun\workspace\BiotechRobot")
    / "biotech-lab-robotics" / "assets" / "panda" / "panda.usda"
)

# Arm keyframes for gentle animation (7 joints + 2 fingers).
Q_A = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])
Q_B = np.array([0.008, 0.047, -0.008, -2.627, 0.001, 2.857, 0.741, 0.04, 0.04])
Q_C = np.array([0.331, 0.135, 1.171, -2.587, -0.590, 2.743, 0.741, 0.02, 0.02])


def smooth(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return a + (b - a) * s


# ----------------------------------------------------------------------------
# Scene construction (USD-level helpers)
# ----------------------------------------------------------------------------
def _box(stage_utils, UsdGeom, Gf, path, size, pos, color=None):
    from isaacsim.core.experimental.objects import Cube

    c = Cube(
        paths=path,
        positions=np.array(pos, dtype=float),
        orientations=np.array([1, 0, 0, 0]),
        sizes=1.0,
        scales=np.array(size, dtype=float),
        colors=color or "white",
    )
    return c


def _cylinder(stage_utils, UsdGeom, Gf, path, radius, height, pos, color=None,
              axis="z"):
    """Cylinder via USD (Isaac Cylinders are z-aligned by default)."""
    from pxr import UsdGeom as UG

    stage = stage_utils.get_current_stage()
    cyl = UG.Cylinder.Define(stage, path)
    cyl.GetRadiusAttr().Set(float(radius))
    cyl.GetHeightAttr().Set(float(height))
    if axis == "x":
        cyl.AddRotateXOp().Set(90.0)
    elif axis == "y":
        cyl.AddRotateYOp().Set(90.0)
    xform = UG.Xformable(cyl)
    # ensure translate op exists
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if color:
        from pxr import UsdShade

        mat_path = f"/World/Materials/{color}"
        # simple display color instead of full material
        cyl.GetDisplayColorAttr().Set([(0.8, 0.8, 0.8)])
    return cyl


def build_room():
    """Floor, walls, ceiling."""
    _box(None, None, None, "/World/Room/Floor", (ROOM_X, ROOM_Y, 0.1),
         (0, 0, -0.05), "lightgray")
    # walls: north/south (along x), east/west (along y)
    _box(None, None, None, "/World/Room/WallN", (ROOM_X, 0.1, ROOM_Z),
         (0, ROOM_Y / 2, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/Room/WallS", (ROOM_X, 0.1, ROOM_Z),
         (0, -ROOM_Y / 2, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/Room/WallE", (0.1, ROOM_Y, ROOM_Z),
         (ROOM_X / 2, 0, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/Room/WallW", (0.1, ROOM_Y, ROOM_Z),
         (-ROOM_X / 2, 0, ROOM_Z / 2), "white")
    # NOTE: no ceiling panel — a closed ceiling blocks the DomeLight and the
    # RTX render products return blank. Open top; walls go to 3 m.


def build_chamber(idx, cx, cy, facing):
    """One chamber: base cabinet, enclosure, open window + latch, props.

    facing = +1 → front faces +y (south chambers); -1 → faces -y (north).
    """
    root = f"/World/Chambers/Ch{idx}"
    fx = cx  # front-center x
    fz0 = BENCH_Z
    # base cabinet (0 → bench)
    _box(None, None, None, f"{root}/Base", (CHAMBER_W, CHAMBER_D, BENCH_Z),
         (cx, cy, BENCH_Z / 2), "gray")
    # work surface slab
    _box(None, None, None, f"{root}/Bench", (CHAMBER_W, CHAMBER_D, 0.04),
         (cx, cy, BENCH_Z + 0.02), "white")
    # back / side / top panels (work zone BENCH_Z → BENCH_Z+0.9)
    wz = BENCH_Z + 0.45
    _box(None, None, None, f"{root}/Back",
         (CHAMBER_W, 0.05, 0.9), (cx, cy - facing * (CHAMBER_D / 2 - 0.025), wz),
         "lightgray")
    _box(None, None, None, f"{root}/SideL",
         (0.05, CHAMBER_D, 0.9), (cx - CHAMBER_W / 2 + 0.025, cy, wz),
         "lightgray")
    _box(None, None, None, f"{root}/SideR",
         (0.05, CHAMBER_D, 0.9), (cx + CHAMBER_W / 2 - 0.025, cy, wz),
         "lightgray")
    _box(None, None, None, f"{root}/Top",
         (CHAMBER_W, CHAMBER_D, 0.05), (cx, cy, BENCH_Z + 0.9), "lightgray")
    # front sash frame (header above opening)
    _box(None, None, None, f"{root}/Header",
         (CHAMBER_W, 0.05, 0.3), (cx, cy + facing * (CHAMBER_D / 2 - 0.025),
                                  BENCH_Z + 0.75), "gray")
    # window glass — POSED OPEN (slid up, in front of header)
    _box(None, None, None, f"{root}/Window",
         (CHAMBER_W - 0.1, 0.02, 0.55),
         (cx, cy + facing * (CHAMBER_D / 2 + 0.01), BENCH_Z + 0.62),
         "lightblue")
    # latch handle on the window's lower rail
    _box(None, None, None, f"{root}/Latch",
         (0.3, 0.04, 0.05),
         (cx, cy + facing * (CHAMBER_D / 2 + 0.03), BENCH_Z + 0.36), "orange")

    # ---- props on the work surface (z = BENCH_Z + 0.04) ----
    pz = BENCH_Z + 0.04
    # test tube rack 3x4 at local (-0.55, 0)
    rx, ry = cx - 0.55, cy
    _box(None, None, None, f"{root}/Rack",
         (0.36, 0.26, 0.05), (rx, ry, pz + 0.025), "darkgray")
    for r in range(3):
        for c_ in range(4):
            tx = rx - 0.12 + c_ * 0.08
            ty = ry - 0.08 + r * 0.08
            _box(None, None, None, f"{root}/Tube_{r}_{c_}",
                 (0.03, 0.03, 0.11), (tx, ty, pz + 0.05 + 0.055), "lightblue")
    # 2 plungers (pipettes) lying at local (0.0, ±0.25)
    for i, off in enumerate((-0.25, 0.25)):
        px, py = cx + 0.0, cy + off
        _box(None, None, None, f"{root}/Plunger{i}_body",
             (0.22, 0.025, 0.025), (px, py, pz + 0.02), "white")
        _box(None, None, None, f"{root}/Plunger{i}_tip",
             (0.06, 0.012, 0.012), (px + 0.14, py, pz + 0.015), "gray")
    # sample tray with wells at local (+0.55, 0)
    tx, ty = cx + 0.55, cy
    _box(None, None, None, f"{root}/Tray",
         (0.34, 0.24, 0.035), (tx, ty, pz + 0.017), "blue")
    for r in range(2):
        for c_ in range(4):
            wx = tx - 0.12 + c_ * 0.08
            wy = ty - 0.06 + r * 0.12
            _box(None, None, None, f"{root}/Well_{r}_{c_}",
                 (0.05, 0.05, 0.038), (wx, wy, pz + 0.019), "darkblue")

    # interior light: rect light under the top panel, illuminates work surface
    # (chambers are enclosed; DomeLight alone leaves them dark)
    from pxr import UsdLux, UsdGeom, Gf
    import isaacsim.core.experimental.utils.stage as _su
    _stage = _su.get_current_stage()
    _light = UsdLux.RectLight.Define(_stage, f"{root}/InteriorLight")
    _light.GetIntensityAttr().Set(800.0)
    _light.GetWidthAttr().Set(1.2)
    _light.GetHeightAttr().Set(0.7)
    _lx = UsdGeom.Xformable(_light)
    _lx.ClearXformOpOrder()
    # position under top panel, rotate to point down (-Z -> -Z is default;
    # RectLight shines along -Z, so translate only)
    _lx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, BENCH_Z + 0.82))

    # return work-surface focus point for cameras
    return (cx, cy, pz + 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="save PNGs from all cameras and exit")
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\lab_captures")
    args = ap.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless or args.capture})

    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import DomeLight
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.experimental.prims import Articulation
    from pxr import Gf, UsdGeom

    DomeLight("/World/DomeLight").set_intensities(1500)

    build_room()
    chamber_focus = []
    idx = 0
    for cx in CHAMBER_XS:
        chamber_focus.append(build_chamber(idx, cx, CHAMBER_Y, -1))
        idx += 1
        chamber_focus.append(build_chamber(idx, cx, -CHAMBER_Y, +1))
        idx += 1

    # ---------------- mobile platform + 2 Franka arms ----------------
    _box(None, None, None, "/World/Platform/Base", (1.4, 0.9, 0.22),
         (0, 0, 0.28), "darkgray")
    _box(None, None, None, "/World/Platform/Deck", (1.3, 0.8, 0.06),
         (0, 0, 0.42), "yellow")
    for i, wx in enumerate((-0.5, 0.5)):
        for j, wy in enumerate((-0.3, 0.3)):
            _box(None, None, None, f"/World/Platform/Wheel{i}_{j}",
                 (0.18, 0.1, 0.18), (wx, wy, 0.09), "black")

    arm_paths = ["/World/Arm0", "/World/Arm1"]
    arm_y = [-0.28, 0.28]
    for p in arm_paths:
        add_reference_to_stage(usd_path=FRANKA_USD, prim_path=p)

    def set_prim_world(path, pos):
        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

    for p in arm_paths:
        set_prim_world(p, (0, 0, 0.45))

    # ---------------- cameras (Replicator look_at for orientation + Camera for capture) ----------------
    # Replicator's look_at correctly orients the USD camera prim.
    # The isaacsim Camera class then captures from that prim (reads USD transform).
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import Camera

    # helper: create a Replicator camera with look_at, return (prim_path, Camera)
    def aim(name, eye, target, res):
        rep.create.camera(position=tuple(eye), look_at=tuple(target), name=name)
        # Rep creates /Replicator/{name}_Xform/{name} (Xform containing Camera)
        prim_path = f"/Replicator/{name}_Xform/{name}"
        cam = Camera(prim_path, resolution=res)
        cam.initialize()
        return (prim_path, cam)

    cams = {}  # name -> (prim_path, Camera)
    # 8 chamber cameras (2 per chamber): front-top-left / front-top-right
    for i, (fx, fy, fz) in enumerate(chamber_focus):
        facing = -1 if fy > 0 else 1  # north chambers face -y
        for j, sx in enumerate((-0.7, 0.7)):
            eye = (fx + sx, fy + facing * 0.55, fz + 0.75)
            name = f"chamber{i}_cam{j}"
            cams[name] = aim(name, eye, (fx, fy, fz), (640, 480))
    # 1 platform camera (front of deck, views arms) — re-aimed per frame
    cams["platform"] = aim("platform",
                           (0.75, 0, 0.75), (0, 0, 0.9), (640, 480))
    # 4 corner cameras (high-res, watch the mobile base)
    corners = [(-4.7, -2.7), (4.7, -2.7), (-4.7, 2.7), (4.7, 2.7)]
    for k, (cx_, cy_) in enumerate(corners):
        name = f"corner{k}"
        eye = (cx_, cy_, 2.8)
        cams[name] = aim(name, eye, (0, 0, 0.5), (1280, 960))
    # 2 wrist cameras — STATIC viewpoints near the arms
    # Positioned to overlook each arm's workspace from front-side.
    for w, wy in enumerate([-0.28, 0.28]):
        eye = (1.2, wy + 0.9, 1.9)
        tgt = (0.0, wy, 0.75)
        cams[f"wrist{w}"] = aim(f"wrist{w}", eye, tgt, (640, 480))

    # ---------------- start physics + articulations (after cameras) ----------------
    import omni.timeline

    omni.timeline.get_timeline_interface().play()
    for _ in range(5):
        simulation_app.update()

    arms = [Articulation(p) for p in arm_paths]
    for a in arms:
        a.set_dof_positions(Q_A)
        a.set_dof_position_targets(Q_A)

    print(f"scene ready: 4 chambers, 2 arms, {len(cams)} cameras")

    # Platform camera Xform path (rep camera name="platform")
    # The Xform holds the transform; the Camera child is at .../platform
    _plat_path = "/Replicator/platform_Xform"

    def aim_rep_camera(prim_path, eye, target):
        # Update a Replicator camera's USD transform (rep reads from USD)
        prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
        xf = UsdGeom.Xformable(prim)
        eye = np.array(eye, dtype=float)
        target = np.array(target, dtype=float)
        f = target - eye; f /= np.linalg.norm(f)
        up = np.array([0.0, 0.0, 1.0])
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        M = Gf.Matrix4d(r[0], u[0], -f[0], eye[0],
                        r[1], u[1], -f[1], eye[1],
                        r[2], u[2], -f[2], eye[2],
                        0, 0, 0, 1)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(M)

    # ---------------- animation / capture ----------------
    from PIL import Image

    out_dir = Path(args.out)
    n_steps = 0

    def step_scene(t):
        # platform slides x in [-3, 3]
        px = 3.0 * np.sin(t * 0.15)
        set_prim_world("/World/Platform/Base", (px, 0, 0.28))
        set_prim_world("/World/Platform/Deck", (px, 0, 0.42))
        wi = 0
        for j in range(2):
            for k in range(2):
                wx = (-0.5, 0.5)[j]
                set_prim_world(f"/World/Platform/Wheel{j}_{k}",
                               (px + wx, (-0.3, 0.3)[k], 0.09))
            wi += 1
        # arms follow platform; gentle keyframe loop (phase-offset)
        for ai, a in enumerate(arms):
            a.set_world_poses(
                positions=np.array([px, arm_y[ai], 0.45]),
                orientations=np.array([1.0, 0.0, 0.0, 0.0]))
            ph = (t * 0.25 + ai * 0.5) % 1.5
            if ph < 0.5:
                q = smooth(Q_A, Q_B, ph / 0.5)
            elif ph < 1.0:
                q = smooth(Q_B, Q_C, ph - 0.5)
            else:
                q = smooth(Q_C, Q_A, ph - 1.0)
            a.set_dof_position_targets(q)
        # platform camera follows platform (update Replicator camera USD transform)
        if _plat_path:
            aim_rep_camera(_plat_path, (px + 0.75, 0, 0.75), (px, 0, 0.9))
        # wrist cameras are static (see creation above); no per-frame update

    def capture(tag):
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            simulation_app.update()
        for name, (prim_path, cam) in cams.items():
            rgb = cam.get_current_frame()["rgb"]
            Image.fromarray(rgb).save(str(out_dir / f"{tag}_{name}.png"))
        print(f"captured {len(cams)} images [{tag}]")

    if args.capture:
        for k, tt in enumerate([-10.5, 0.0, 10.5]):
            step_scene(tt)
            for _ in range(20):
                simulation_app.update()
            capture(f"t{k}")
        print("CAPTURE DONE:", out_dir)
    elif args.headless:
        for s in range(300):
            step_scene(s * 0.05)
            simulation_app.update()
        print("HEADLESS RUN OK")
    else:
        t = 0.0
        while simulation_app.is_running():
            step_scene(t)
            simulation_app.update()
            t += 0.02
    simulation_app.close()


if __name__ == "__main__":
    main()
