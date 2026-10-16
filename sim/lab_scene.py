"""Simplified biotech lab: 1 chamber, 1 fixed arm, 5 cameras.

Room (10m x 6m x 3m) with a single large biosafety-style chamber
(3.0m wide x 2.0m deep, 1.5m work zone, open front — no window).
Chamber contents: 3x4 test tube rack, 2 plungers, sample tray with wells.
One fixed Franka arm on a pedestal in front of the chamber.

Cameras (5 total):
  - 2 inside the chamber (back corners, pointing out at the arm)
  - 2 in the room behind the robot (pointing at arm + chamber opening)
  - 1 wrist camera (follows the end effector)

Lighting: DomeLight + interior RectLight (same as before).

Usage (Isaac venv, repo root on Windows box):
    python sim\\lab_scene.py                  # GUI, animated
    python sim\\lab_scene.py --headless       # one animated pass, no window
    python sim\\lab_scene.py --capture        # save PNGs from all 5 cameras
    python sim\\lab_scene.py --camera wrist   # GUI, viewport starts at wrist
    python sim\\lab_scene.py --list-cameras   # print camera names and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Layout constants
# ----------------------------------------------------------------------------
ROOM_X, ROOM_Y, ROOM_Z = 10.0, 6.0, 3.0
CHAMBER_W, CHAMBER_D = 3.0, 2.0       # wide x deep
BENCH_Z = 0.9                          # work surface height
WORK_H = 1.5                           # work zone height
CHAMBER_CX, CHAMBER_CY = 0.0, 2.5      # chamber center (front faces -y)
ARM_POS = (0.0, 0.6, 0.45)             # fixed arm base (0.9m in front)

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
# Scene construction
# ----------------------------------------------------------------------------
def _box(stage_utils, UsdGeom, Gf, path, size, pos, color=None):
    from isaacsim.core.experimental.objects import Cube

    c = Cube(
        paths=path,
        positions=np.array(pos, dtype=float),
        sizes=1.0,
        scales=np.array(size, dtype=float),
        colors=color or "white",
    )
    return c


def build_chamber(cx, cy):
    """Single large chamber: base, open-front enclosure, props, interior light."""
    root = "/World/Chamber"
    # base cabinet + work surface
    _box(None, None, None, f"{root}/Base", (CHAMBER_W, CHAMBER_D, BENCH_Z),
         (cx, cy, BENCH_Z / 2), "gray")
    _box(None, None, None, f"{root}/Bench", (CHAMBER_W, CHAMBER_D, 0.04),
         (cx, cy, BENCH_Z + 0.02), "white")
    # back / side / top panels (front fully open — no window/header)
    wz = BENCH_Z + WORK_H / 2
    _box(None, None, None, f"{root}/Back",
         (CHAMBER_W, 0.05, WORK_H), (cx, cy + CHAMBER_D / 2 - 0.025, wz),
         "lightgray")
    _box(None, None, None, f"{root}/SideL",
         (0.05, CHAMBER_D, WORK_H), (cx - CHAMBER_W / 2 + 0.025, cy, wz),
         "lightgray")
    _box(None, None, None, f"{root}/SideR",
         (0.05, CHAMBER_D, WORK_H), (cx + CHAMBER_W / 2 - 0.025, cy, wz),
         "lightgray")
    _box(None, None, None, f"{root}/Top",
         (CHAMBER_W, CHAMBER_D, 0.05), (cx, cy, BENCH_Z + WORK_H), "lightgray")

    # ---- props on the work surface ----
    pz = BENCH_Z + 0.04
    # test tube rack 3x4
    rx, ry = cx - 0.75, cy
    _box(None, None, None, f"{root}/Rack",
         (0.36, 0.26, 0.05), (rx, ry, pz + 0.025), "darkgray")
    for r in range(3):
        for c_ in range(4):
            tx = rx - 0.12 + c_ * 0.08
            ty = ry - 0.08 + r * 0.08
            _box(None, None, None, f"{root}/Tube_{r}_{c_}",
                 (0.03, 0.03, 0.11), (tx, ty, pz + 0.05 + 0.055), "lightblue")
    # 2 plungers
    for i, off in enumerate((-0.25, 0.25)):
        px, py = cx + 0.0, cy + off
        _box(None, None, None, f"{root}/Plunger{i}_body",
             (0.22, 0.025, 0.025), (px, py, pz + 0.02), "white")
        _box(None, None, None, f"{root}/Plunger{i}_tip",
             (0.06, 0.012, 0.012), (px + 0.14, py, pz + 0.015), "gray")
    # sample tray with wells
    tx, ty = cx + 0.75, cy
    _box(None, None, None, f"{root}/Tray",
         (0.34, 0.24, 0.035), (tx, ty, pz + 0.017), "blue")
    for r in range(2):
        for c_ in range(4):
            wx = tx - 0.12 + c_ * 0.08
            wy = ty - 0.06 + r * 0.12
            _box(None, None, None, f"{root}/Well_{r}_{c_}",
                 (0.05, 0.05, 0.038), (wx, wy, pz + 0.019), "darkblue")

    # interior RectLight under the top panel
    from pxr import UsdLux, UsdGeom, Gf
    import isaacsim.core.experimental.utils.stage as _su
    _stage = _su.get_current_stage()
    _light = UsdLux.RectLight.Define(_stage, f"{root}/InteriorLight")
    _light.GetIntensityAttr().Set(1000.0)
    _light.GetWidthAttr().Set(2.0)
    _light.GetHeightAttr().Set(1.2)
    _lx = UsdGeom.Xformable(_light)
    _lx.ClearXformOpOrder()
    _lx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, BENCH_Z + WORK_H - 0.08))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="save PNGs from all cameras and exit")
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\lab_captures")
    ap.add_argument("--camera", default=None,
                    help="GUI mode: set viewport to this camera on startup")
    ap.add_argument("--list-cameras", action="store_true",
                    help="print camera names and exit")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless or args.capture})

    from pxr import Gf, UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.experimental.objects import DomeLight
    from isaacsim.core.prims import Articulation

    # ---------------- room ----------------
    _box(None, None, None, "/World/Floor", (ROOM_X, ROOM_Y, 0.1),
         (0, 0, -0.05), "lightgray")
    # walls (no ceiling — open top for DomeLight)
    _box(None, None, None, "/World/WallN", (ROOM_X, 0.1, ROOM_Z),
         (0, ROOM_Y / 2, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/WallS", (ROOM_X, 0.1, ROOM_Z),
         (0, -ROOM_Y / 2, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/WallE", (0.1, ROOM_Y, ROOM_Z),
         (ROOM_X / 2, 0, ROOM_Z / 2), "white")
    _box(None, None, None, "/World/WallW", (0.1, ROOM_Y, ROOM_Z),
         (-ROOM_X / 2, 0, ROOM_Z / 2), "white")
    DomeLight("/World/DomeLight").set_intensities(1500)

    # ---------------- chamber ----------------
    build_chamber(CHAMBER_CX, CHAMBER_CY)

    # ---------------- fixed arm on pedestal ----------------
    _box(None, None, None, "/World/Pedestal", (0.5, 0.5, ARM_POS[2]),
         (ARM_POS[0], ARM_POS[1], ARM_POS[2] / 2), "darkgray")
    add_reference_to_stage(usd_path=FRANKA_USD, prim_path="/World/Arm")

    def set_prim_world(path, pos):
        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

    set_prim_world("/World/Arm", ARM_POS)

    # ---------------- cameras ----------------
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import Camera

    def aim(name, eye, target, res, focal_length=24.0):
        rep.create.camera(position=tuple(eye), look_at=tuple(target), name=name,
                          focal_length=focal_length)
        prim_path = f"/Replicator/{name}_Xform/{name}"
        cam = Camera(prim_path, resolution=res)
        cam.initialize()
        return (prim_path, cam)

    cams = {}
    cx, cy = CHAMBER_CX, CHAMBER_CY
    # 2 inside the chamber (back corners, pointing out at the arm)
    # Back wall at cy + D/2 = 3.5; cameras 0.2m in front of it
    for j, sx in enumerate((-1.2, 1.2)):
        eye = (cx + sx, cy + CHAMBER_D / 2 - 0.2, BENCH_Z + WORK_H * 0.7)
        target = (ARM_POS[0], ARM_POS[1], ARM_POS[2] + 0.3)
        cams[f"chamber_cam{j}"] = aim(f"chamber_cam{j}", eye, target,
                                      (640, 480), focal_length=14.0)
    # 2 in the room behind the robot (opposite view: arm + chamber opening)
    for j, sx in enumerate((-1.5, 1.5)):
        eye = (ARM_POS[0] + sx, ARM_POS[1] - 1.6, 1.8)
        target = (cx, cy - CHAMBER_D / 2, BENCH_Z + 0.3)
        cams[f"room_cam{j}"] = aim(f"room_cam{j}", eye, target, (640, 480))
    # 1 wrist camera (follows end effector; positioned per-frame below)
    # Initial guess: above the arm
    cams["wrist"] = aim("wrist",
                        (ARM_POS[0] + 0.5, ARM_POS[1] - 0.5, 1.5),
                        (ARM_POS[0], ARM_POS[1], 0.8), (640, 480))

    print(f"scene ready: 1 chamber, 1 arm, {len(cams)} cameras")

    # ---------------- physics ----------------
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    for _ in range(5):
        simulation_app.update()

    arm = Articulation("/World/Arm")
    arm.set_joint_positions(Q_A)
    arm.set_joint_position_targets(Q_A)

    # Wrist camera follows the hand (update Replicator Xform per-frame)
    wrist_xform_path = "/Replicator/wrist_Xform"
    hand_path = "/World/Arm/Geometry/link0/link1/link2/link3/link4/link5/link6/link7/hand"

    def get_hand_world():
        prim = stage_utils.get_current_stage().GetPrimAtPath(hand_path)
        if not prim.IsValid():
            return np.array(ARM_POS)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            omni.timeline.get_timeline_interface().get_current_time())
        return np.array([m[3][0], m[3][1], m[3][2]])

    def aim_wrist():
        hw = get_hand_world()
        # Camera 15cm behind-above the hand, looking at the hand
        eye = hw + np.array([-0.25, -0.25, 0.20])
        target = hw
        prim = stage_utils.get_current_stage().GetPrimAtPath(wrist_xform_path)
        xf = UsdGeom.Xformable(prim)
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

    def step_scene(t):
        # Gentle arm keyframe loop
        ph = (t * 0.25) % 1.5
        if ph < 0.5:
            q = smooth(Q_A, Q_B, ph / 0.5)
        elif ph < 1.0:
            q = smooth(Q_B, Q_C, ph - 0.5)
        else:
            q = smooth(Q_C, Q_A, ph - 1.0)
        arm.set_joint_position_targets(q)
        # Wrist camera tracks the hand
        aim_wrist()

    def capture(tag):
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            simulation_app.update()
        for name, (prim_path, cam) in cams.items():
            rgb = cam.get_current_frame()["rgb"]
            Image.fromarray(rgb).save(str(out_dir / f"{tag}_{name}.png"))
        print(f"captured {len(cams)} images [{tag}]")

    if args.list_cameras:
        print("Available cameras:")
        for name in sorted(cams.keys()):
            print(f"  {name}")
        simulation_app.close()
        return

    if args.capture:
        for k, tt in enumerate([0.0, 10.0, 20.0]):
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
        if args.camera:
            if args.camera not in cams:
                print(f"Unknown camera '{args.camera}'. Available:")
                for n in sorted(cams.keys()):
                    print(f"  {n}")
            else:
                prim_path, _ = cams[args.camera]
                try:
                    import omni.kit.viewport.utility as vpu
                    vp = vpu.get_active_viewport()
                    vp.set_active_camera(prim_path)
                    print(f"Viewport set to camera: {args.camera}")
                except Exception as e:
                    print(f"Could not set viewport: {e}")
        t = 0.0
        while simulation_app.is_running():
            step_scene(t)
            simulation_app.update()
            t += 0.02
    simulation_app.close()


if __name__ == "__main__":
    main()
