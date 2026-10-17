"""Biotech lab setup: open test-tube platform + Franka arm on a pedestal.

Room (10m x 6m x 3m), no enclosure. One open platform (1.4m x 0.7m, bench
height) holds a 3x4 test tube rack, a sample tray with wells, and 2
pipettes. One fixed Franka arm sits on a pedestal facing the platform,
base height matched to the platform so it can reach everything on it. The
arm is static -- held in its home pose, no motion -- this script only
builds and lets you view/capture the scene.

Cameras (5 total):
  - 2 behind the robot arm (arm + tube platform both in view)
  - 2 behind the tube platform (pointing back at the arm)
  - 1 on the arm (wrist-mounted, fixed since the arm doesn't move)

Usage (Isaac venv, repo root on Windows box):
    python sim\\lab_scene.py                  # GUI, static scene
    python sim\\lab_scene.py --headless       # headless sanity check, no window
    python sim\\lab_scene.py --capture        # save PNGs from all 5 cameras, exit
    python sim\\lab_scene.py --camera wrist   # GUI, viewport starts at this camera
    python sim\\lab_scene.py --multi-view     # GUI, all 5 cameras tiled at once
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
PLATFORM_W, PLATFORM_D = 1.4, 0.7      # open platform footprint
BENCH_Z = 0.9                          # platform / pedestal top height
PLATFORM_CX, PLATFORM_CY = 0.0, 1.8    # platform center
# Fixed arm base, facing the platform. Y chosen so the rack (the farthest
# task objects, ~0.2-0.6m into the platform from its near edge) fall within
# the Franka's ~0.85m kinematic reach -- verified via sim/ik_solver.py: at
# the previous ARM_POS y=0.9, the nearest rack tube was ~1.05m from the
# hand's home position and IK could not converge on it at all.
ARM_POS = (0.0, 1.1, BENCH_Z)

FRANKA_USD = str(
    Path(r"C:\Users\varun\workspace\BiotechRobot")
    / "biotech-lab-robotics" / "assets" / "panda" / "panda.usda"
)

# Franka home pose (7 arm joints + 2 fingers) -- the arm is static, held here.
Q_HOME = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])

HAND_PATH = "/World/Arm/Geometry/link0/link1/link2/link3/link4/link5/link6/link7/hand"


def get_prim_world_pose(path):
    """World pose [x,y,z, qw,qx,qy,qz] of a prim, or None if it doesn't exist."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom
    import omni.timeline

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        omni.timeline.get_timeline_interface().get_current_time())
    tf = Gf.Transform(m)
    t = tf.GetTranslation()
    q = tf.GetRotation().GetQuat()
    return np.array([t[0], t[1], t[2], q.GetReal(), *q.GetImaginary()])


# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------
def _box(path, size, pos, color=None):
    from isaacsim.core.experimental.objects import Cube

    return Cube(
        paths=path,
        positions=np.array(pos, dtype=float),
        sizes=1.0,
        scales=np.array(size, dtype=float),
        colors=color or "white",
    )


def build_platform(cx, cy):
    """Open test-tube platform: support stand, bench top, props, work light.

    Returns {name: {"path": prim_path, "size": (w, d, h)}} for the loggable
    task objects (tubes, pipettes, tray) -- not the static fixtures (stand,
    bench, rack body, light). "size" is the object's authored full bounding
    box (the dimensions it was built with), used for state-aware metadata
    and for computing task-space targets relative to an object (e.g. its
    top opening) without re-deriving geometry by hand at call sites.
    """
    root = "/World/Platform"
    _box(f"{root}/Stand", (PLATFORM_W, PLATFORM_D, BENCH_Z),
         (cx, cy, BENCH_Z / 2), "gray")
    _box(f"{root}/Bench", (PLATFORM_W, PLATFORM_D, 0.04),
         (cx, cy, BENCH_Z + 0.02), "white")

    objects = {}
    pz = BENCH_Z + 0.04
    # test tube rack (3x4)
    rx, ry = cx - 0.4, cy
    tube_size = (0.03, 0.03, 0.11)
    _box(f"{root}/Rack", (0.36, 0.26, 0.05), (rx, ry, pz + 0.025), "darkgray")
    for r in range(3):
        for c_ in range(4):
            tx = rx - 0.12 + c_ * 0.08
            ty = ry - 0.08 + r * 0.08
            path = f"{root}/Tube_{r}_{c_}"
            _box(path, tube_size, (tx, ty, pz + 0.05 + 0.055), "lightblue")
            objects[f"tube_{r}_{c_}"] = {"path": path, "size": tube_size}
    # 2 pipettes
    pipette_size = (0.22, 0.025, 0.025)
    for i, off in enumerate((-0.15, 0.15)):
        px, py = cx, cy + off
        body_path = f"{root}/Pipette{i}_body"
        _box(body_path, pipette_size, (px, py, pz + 0.02), "white")
        _box(f"{root}/Pipette{i}_tip", (0.06, 0.012, 0.012),
             (px + 0.14, py, pz + 0.015), "gray")
        objects[f"pipette_{i}"] = {"path": body_path, "size": pipette_size}
    # sample tray with wells
    tray_size = (0.34, 0.24, 0.035)
    tx, ty = cx + 0.4, cy
    tray_path = f"{root}/Tray"
    _box(tray_path, tray_size, (tx, ty, pz + 0.017), "blue")
    objects["tray"] = {"path": tray_path, "size": tray_size}
    for r in range(2):
        for c_ in range(4):
            wx = tx - 0.12 + c_ * 0.08
            wy = ty - 0.06 + r * 0.12
            _box(f"{root}/Well_{r}_{c_}", (0.05, 0.05, 0.038),
                 (wx, wy, pz + 0.019), "darkblue")

    # overhead work light
    from pxr import UsdLux, UsdGeom, Gf
    import isaacsim.core.experimental.utils.stage as _su
    _stage = _su.get_current_stage()
    _light = UsdLux.RectLight.Define(_stage, f"{root}/WorkLight")
    _light.GetIntensityAttr().Set(1200.0)
    _light.GetWidthAttr().Set(1.6)
    _light.GetHeightAttr().Set(1.0)
    _lx = UsdGeom.Xformable(_light)
    _lx.ClearXformOpOrder()
    _lx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, BENCH_Z + 1.3))

    return objects


def build_scene(simulation_app):
    """Build room + platform + arm + cameras; settle the arm into its home
    pose. Returns (arm: Articulation, cams: {name: (prim_path, Camera)},
    objects: {name: {"path": prim_path, "size": (w, d, h)}}).

    Shared by lab_scene.py's own CLI (viewing/capture) and sim/collect_data.py
    (data generation), so both always build the exact same scene.
    """
    from pxr import Gf, UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.experimental.objects import DomeLight
    from isaacsim.core.prims import Articulation

    # ---------------- room ----------------
    _box("/World/Floor", (ROOM_X, ROOM_Y, 0.1), (0, 0, -0.05), "lightgray")
    _box("/World/WallN", (ROOM_X, 0.1, ROOM_Z), (0, ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallS", (ROOM_X, 0.1, ROOM_Z), (0, -ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallE", (0.1, ROOM_Y, ROOM_Z), (ROOM_X / 2, 0, ROOM_Z / 2), "white")
    _box("/World/WallW", (0.1, ROOM_Y, ROOM_Z), (-ROOM_X / 2, 0, ROOM_Z / 2), "white")
    DomeLight("/World/DomeLight").set_intensities(1500)

    # ---------------- platform ----------------
    objects = build_platform(PLATFORM_CX, PLATFORM_CY)

    # ---------------- fixed arm on pedestal ----------------
    # Pedestal top at BENCH_Z so the arm base is level with the platform.
    _box("/World/Pedestal", (0.5, 0.5, BENCH_Z),
         (ARM_POS[0], ARM_POS[1], BENCH_Z / 2), "darkgray")
    add_reference_to_stage(usd_path=FRANKA_USD, prim_path="/World/Arm")

    def set_prim_pose(path, pos, yaw_deg):
        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateZOp().Set(float(yaw_deg))

    # Arm base yawed 90 deg so local +x (reach direction) points in world +y,
    # towards the platform.
    set_prim_pose("/World/Arm", ARM_POS, 90.0)

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

    # ---------------- physics: settle the arm into its home pose first, so ----
    # ---------------- the wrist camera can be aimed from its real hand pose --
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    for _ in range(5):
        simulation_app.update()

    arm = Articulation("/World/Arm")
    arm.set_joint_positions(Q_HOME)
    arm.set_joint_position_targets(Q_HOME)
    for _ in range(10):
        simulation_app.update()

    cams = {}
    # 2 behind the robot arm: arm + tube platform both in view
    for j, sx in enumerate((-1.3, 1.3)):
        eye = (ARM_POS[0] + sx, ARM_POS[1] - 1.4, 1.7)
        target = (PLATFORM_CX, PLATFORM_CY, BENCH_Z + 0.2)
        cams[f"arm_cam{j}"] = aim(f"arm_cam{j}", eye, target, (640, 480))
    # 2 behind the tube platform, looking back at the arm
    for j, sx in enumerate((-1.0, 1.0)):
        eye = (PLATFORM_CX + sx, PLATFORM_CY + PLATFORM_D / 2 + 0.6, BENCH_Z + 0.9)
        target = (ARM_POS[0], ARM_POS[1], ARM_POS[2] + 0.3)
        cams[f"platform_cam{j}"] = aim(f"platform_cam{j}", eye, target,
                                       (640, 480), focal_length=18.0)
    # 1 on the arm: eye-in-hand, mounted just above the hand, looking out at
    # the platform (looking back at the hand itself would clip into the arm's
    # own body). The arm never moves, so this is computed once from its
    # settled home-pose transform.
    hw = get_prim_world_pose(HAND_PATH)
    hw = hw[:3] if hw is not None else np.array(ARM_POS)
    wrist_eye = hw + np.array([0.0, 0.0, 1.0])
    wrist_target = (PLATFORM_CX, PLATFORM_CY, BENCH_Z + 0.1)
    cams["wrist"] = aim("wrist", tuple(wrist_eye), wrist_target, (640, 480))

    return arm, cams, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="save PNGs from all cameras and exit")
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\lab_captures")
    ap.add_argument("--camera", default=None,
                    help="GUI mode: set viewport to this camera on startup")
    ap.add_argument("--multi-view", action="store_true",
                    help="GUI mode: tile all cameras in one window, viewed simultaneously")
    ap.add_argument("--list-cameras", action="store_true",
                    help="print camera names and exit")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless or args.capture})

    from pxr import Sdf

    arm, cams, objects = build_scene(simulation_app)
    print(f"scene ready: 1 platform, 1 arm (static), {len(cams)} cameras")

    out_dir = Path(args.out)

    def capture():
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            simulation_app.update()
        for name, (prim_path, cam) in cams.items():
            rgb = cam.get_current_frame()["rgb"]
            Image.fromarray(rgb).save(str(out_dir / f"{name}.png"))
        print(f"captured {len(cams)} images -> {out_dir}")

    if args.list_cameras:
        print("Available cameras:")
        for name in sorted(cams.keys()):
            print(f"  {name}")
        simulation_app.close()
        return

    if args.capture:
        capture()
        print("CAPTURE DONE:", out_dir)
    elif args.headless:
        for _ in range(60):
            simulation_app.update()
        print("HEADLESS RUN OK")
    else:
        import omni.kit.viewport.utility as vpu

        if args.multi_view:
            # Tile every camera into its own window, viewed simultaneously.
            # Grid: 3 columns, tiles sized so 5 fit on a 1440x900+ screen.
            tile_w, tile_h = 480, 360
            cols = 3
            default_win = vpu.get_active_viewport_window()
            if default_win is not None:
                default_win.visible = False
            for idx, (name, (prim_path, _)) in enumerate(sorted(cams.items())):
                col, row = idx % cols, idx // cols
                win = vpu.create_viewport_window(
                    name=name, width=tile_w, height=tile_h,
                    position_x=col * tile_w, position_y=row * tile_h,
                    camera_path=Sdf.Path(prim_path),
                )
                win.viewport_api.set_active_camera(Sdf.Path(prim_path))
            print(f"Multi-view: {len(cams)} cameras tiled in a {cols}-column grid")
        elif args.camera:
            if args.camera not in cams:
                print(f"Unknown camera '{args.camera}'. Available:")
                for n in sorted(cams.keys()):
                    print(f"  {n}")
            else:
                prim_path, _ = cams[args.camera]
                try:
                    vp = vpu.get_active_viewport()
                    vp.set_active_camera(prim_path)
                    print(f"Viewport set to camera: {args.camera}")
                except Exception as e:
                    print(f"Could not set viewport: {e}")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
