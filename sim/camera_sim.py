"""Vision-based arm simulation: cameras, randomized states, image capture.

Scene: frozen canonical 6-DOF arm + lab table + tray (SBS well-plate sized).
Cameras: wrist camera (on the flange) + two third-person lab cameras.

Each episode randomizes:
  - initial joint configuration (within 40% of joint ranges)
  - tray initial pose on the table
  - tray goal pose (the "end state" the policy must achieve)

Outputs per episode: wrist.png, cam_a.png, cam_b.png, meta.json
These (images + goal) are the training inputs for the vision policy:
    images -> NN -> arm control actions.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from embodiments.arm_generator import ArmParams, generate_arm_mjcf


def build_scene_xml() -> str:
    arm_xml = generate_arm_mjcf(ArmParams())  # frozen canonical 6-DOF design
    root = ET.fromstring(arm_xml.split("?>\n", 1)[1])
    worldbody = root.find("worldbody")

    # Lab table (top surface at z = 0.40)
    table = ET.SubElement(worldbody, "body", name="table", pos="0.45 0 0.38")
    ET.SubElement(table, "geom", name="table_top", type="box",
                  size="0.35 0.30 0.02", rgba="0.75 0.75 0.78 1")

    # Tray: SBS well-plate footprint, free body so it can be grasped later
    tray = ET.SubElement(worldbody, "body", name="tray", pos="0.45 0.0 0.408")
    ET.SubElement(tray, "freejoint", name="tray_free")
    ET.SubElement(tray, "geom", name="tray_geom", type="box",
                  size="0.064 0.043 0.008", rgba="0.15 0.55 0.9 1")

    # Wrist camera on the flange, looking out along the flange (+Z of link6).
    # MuJoCo cameras look along their local -Z; euler 180 about Y flips it.
    link6 = next(b for b in root.iter("body") if b.get("name") == "link6")
    ET.SubElement(link6, "camera", name="wristcam", pos="0 0 0.03",
                  euler="0 180 0")

    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode")


def render_episode(model, renderer, rng, out_dir: Path, ep: int) -> None:
    import mujoco

    data = mujoco.MjData(model)
    jr = model.jnt_range[:6]  # arm joints only (tray freejoint excluded)
    data.qpos[:6] = rng.uniform(jr[:, 0] * 0.4, jr[:, 1] * 0.4)

    tray_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tray_free")
    adr = model.jnt_qposadr[tray_jnt]
    tray_pos = [float(rng.uniform(0.25, 0.65)),
                float(rng.uniform(-0.20, 0.20)), 0.408]
    data.qpos[adr:adr + 7] = tray_pos + [1, 0, 0, 0]
    goal_pos = [float(rng.uniform(0.25, 0.65)),
                float(rng.uniform(-0.20, 0.20)), 0.408]
    mujoco.mj_forward(model, data)

    ep_dir = out_dir / f"ep_{ep:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wristcam")
    renderer.update_scene(data, camera=wrist_id)
    frames = {"wrist": renderer.render().copy()}
    for name, az in (("cam_a", 120.0), ("cam_b", 300.0)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = np.array([0.3, 0.0, 0.35])
        cam.distance = 1.6
        cam.azimuth = az
        cam.elevation = -25
        renderer.update_scene(data, camera=cam)
        frames[name] = renderer.render().copy()

    try:
        from PIL import Image

        for name, img in frames.items():
            Image.fromarray(img).save(ep_dir / f"{name}.png")
    except ImportError:
        for name, img in frames.items():
            np.save(ep_dir / f"{name}.npy", img)

    meta = {
        "qpos_init": [float(v) for v in data.qpos[:6]],
        "tray_pos_init": tray_pos,
        "tray_pos_goal": goal_pos,
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {ep_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("assets/vision_episodes"))
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_string(build_scene_xml())
    renderer = mujoco.Renderer(model, height=240, width=320)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        render_episode(model, renderer, rng, args.out, ep)
    print("vision render test OK")


if __name__ == "__main__":
    main()
