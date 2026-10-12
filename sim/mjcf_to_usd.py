"""MJCF -> USD exporter for Blender visualization.

Parses a MuJoCo scene, bakes forward kinematics for a chosen configuration,
and writes a USD stage (static or time-sampled trajectory animation).

Usage (on the Windows box, in the venv, from the repo root):
    $env:PYTHONPATH = (Get-Location).Path
    python sim/mjcf_to_usd.py --out assets/scene.usd
    python sim/mjcf_to_usd.py --demo data/demos/demo_0000.npz --out assets/demo_0000.usd
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rot_x(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# Maps MuJoCo's Z-aligned capsule/cylinder to USD's Y-aligned prims.
# extra_rot must send USD +Y to MuJoCo +Z: RotX(+90).
Z_TO_Y = _rot_x(90.0)


def _usd_mat(xpos, xmat):
    from pxr import Gf
    m = Gf.Matrix4d()
    m.SetTranslate(Gf.Vec3d(*xpos))
    r = Gf.Matrix4d()
    # Gf.Matrix3d takes row-major args; MuJoCo xmat reshaped (3,3) is row-major.
    r.SetRotateOnly(Gf.Matrix3d(*xmat.flatten()))
    return m * r


def build_stage(model, data, geoms):
    """Create a USD stage with one Xform per geom, placed from mj_forward."""
    from pxr import Usd, UsdGeom, Gf
    import mujoco

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/world")

    prims = []
    for gi in geoms:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
        name = (name or f"geom_{gi}").replace("-", "_")
        gtype = int(model.geom_type[gi])
        size = model.geom_size[gi]
        rgba = model.geom_rgba[gi]

        xform = UsdGeom.Xform.Define(stage, f"/world/{name}")
        xop = xform.AddTransformOp()
        sop = xform.AddScaleOp()
        shape = None
        scale = np.ones(3)
        if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            cap = UsdGeom.Capsule.Define(stage, f"/world/{name}/shape")
            cap.GetRadiusAttr().Set(float(size[0]))
            cap.GetHeightAttr().Set(float(2 * size[1]))
            shape = cap
            extra_rot = Z_TO_Y
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cyl = UsdGeom.Cylinder.Define(stage, f"/world/{name}/shape")
            cyl.GetRadiusAttr().Set(float(size[0]))
            cyl.GetHeightAttr().Set(float(2 * size[1]))
            shape = cyl
            extra_rot = Z_TO_Y
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            cube = UsdGeom.Cube.Define(stage, f"/world/{name}/shape")
            cube.GetSizeAttr().Set(1.0)
            shape = cube
            extra_rot = np.eye(3)
            scale = np.array([2 * size[0], 2 * size[1], 2 * size[2]])
        elif gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            cube = UsdGeom.Cube.Define(stage, f"/world/{name}/shape")
            cube.GetSizeAttr().Set(1.0)
            shape = cube
            extra_rot = np.eye(3)
            # 10x10 m slab, 0.2 m thick, top at z=0
            scale = np.array([10.0, 10.0, 0.2])
            xpos_off = np.array([0.0, 0.0, -0.1])
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            sph = UsdGeom.Sphere.Define(stage, f"/world/{name}/shape")
            sph.GetRadiusAttr().Set(float(size[0]))
            shape = sph
            extra_rot = np.eye(3)
        else:
            continue
        if rgba is not None and len(rgba) == 4:
            gprim = UsdGeom.Gprim(shape.GetPrim())
            gprim.GetDisplayColorAttr().Set([(float(rgba[0]),
                                             float(rgba[1]),
                                             float(rgba[2]))])
        sop.Set(Gf.Vec3f(*scale))  # constant scale, no time sampling needed
        prims.append((xop, gi, extra_rot,
                      xpos_off if gtype == mujoco.mjtGeom.mjGEOM_PLANE
                      else np.zeros(3)))
    return stage, prims


def write_frame(stage_prims, model, data, time=None):
    import mujoco
    from pxr import Gf
    mujoco.mj_forward(model, data)
    for xop, gi, extra_rot, xpos_off in stage_prims:
        xpos = data.geom_xpos[gi] + xpos_off
        xmat = data.geom_xmat[gi].reshape(3, 3) @ extra_rot
        m = _usd_mat(xpos, xmat)
        if time is None:
            xop.Set(m)
        else:
            xop.Set(m, time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .usd file")
    ap.add_argument("--demo", default=None,
                    help="demo .npz (from make_demos.py) for animated export")
    ap.add_argument("--stride", type=int, default=20,
                    help="frame stride for animation")
    ap.add_argument("--mjcf", default=None,
                    help="MJCF file (default: build from camera_sim)")
    args = ap.parse_args()

    import mujoco
    if args.mjcf:
        xml = Path(args.mjcf).read_text()
    else:
        from sim.camera_sim import build_scene_xml
        xml = build_scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Skip the tray's free-joint dynamics; place it explicitly per frame.
    geoms = [i for i in range(model.ngeom)
             if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)]

    stage, prims = build_stage(model, data, geoms)

    if args.demo:
        d = np.load(args.demo)
        q_traj, tray_init = d["q_traj"], d["tray_init"]
        from pxr import UsdGeom
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(len(range(0, len(q_traj), args.stride)) - 1)
        tray_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     "tray_free")
        adr = model.jnt_qposadr[tray_jnt]
        for t, s in enumerate(range(0, len(q_traj), args.stride)):
            data.qpos[:6] = q_traj[s]
            data.qvel[:6] = 0
            data.qpos[adr:adr + 7] = [*tray_init, 1, 0, 0, 0]
            write_frame(prims, model, data, time=t)
        # hold last frame a bit
        end = int(stage.GetEndTimeCode())
        for t in range(end, end + 24):
            write_frame(prims, model, data, time=t)
        stage.SetEndTimeCode(end + 24)
    else:
        data.qpos[:6] = 0
        data.qvel[:6] = 0
        write_frame(prims, model, data)

    stage.GetRootLayer().Export(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
