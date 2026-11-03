r"""Start-state library for the tip-positioning RL task (full milestone M6:
aspirate and dispense): the arm holding the pipette with a tip, near the
tube or tray well it is about to aspirate from or dispense into, from
recorded expert aspirate_from_tube, dispense_into_tube and dispense_into_well
episodes (any layout: the task is kinematic, see tip_position/env.py).

Target: where the expert's tip end was when the plunger started to move (the
liquid level, the volume and the container decide that depth). Frames: from
the start of the episode until then, the tip end within NEAR_M of the target.

Each entry: joint positions (7 + 2 fingers); the attached tip's pose in the
hand frame (its origin is the tip's end, local +z up the tip); the
container's origin and axis (tube: its base, its axis; well: the tray's base
under the well centre, the tray's up), kind (0 tube, 1 well), the target
point -- all shifted to the default arm spot (neighbours.arm_shift); the
primitive (0 aspirate, 1 dispense).

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_pipetting.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import neighbours
from start_states_tip import T, matrix_to_quat

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_pipetting.npz"
RUNS = ("primitives_pipette_M6", "primitives_v2", "primitives_v3", "primitives_v4")
PRIMS = {"aspirate_from_tube": 0, "dispense_into_tube": 1, "dispense_into_well": 1}
NEAR_M = 0.12
TIP_LENGTH = 0.060
TRAY_PITCH = 0.009                 # lab_scene.SAMPLE_TRAY_PITCH


def slot_tube(label):
    return f"tube_r{'ABCD'.index(label[0])}_c{int(label[1:]) - 1}"


def main():
    rows = {k: [] for k in ("q", "inhand", "origin", "axis", "kind", "target", "prim", "source")}
    for run in RUNS:
        for prim, pk in PRIMS.items():
            for ep in sorted(glob.glob(str(DATA / run / prim / "ep_*"))):
                ep = Path(ep)
                meta = json.loads((ep / "meta.json").read_text())
                if not meta.get("success"):
                    continue
                names = meta.get("object_names") or []
                d = np.load(ep / "steps.npz")
                P = d["object_poses"]
                plunger = np.asarray(d["plunger"]).reshape(len(P))
                moving = np.nonzero(np.abs(np.diff(plunger)) > 1e-6)[0]
                if not len(moving):
                    continue
                t_star = int(moving[0])
                ip = names.index("pipette_0")
                tips = [i for i, n in enumerate(names) if n.startswith("tip_")]
                # the attached tip: its collar top on the nozzle (the pipette's origin)
                ti = min(tips, key=lambda i: np.linalg.norm(T(P[0, i])[:3, 3] + T(P[0, i])[:3, 2] * TIP_LENGTH
                                                            - P[0, ip, :3]))
                shift = neighbours.arm_shift(meta)
                kind = 1 if prim == "dispense_into_well" else 0
                qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(P), -1)[:, :2]], 1)
                target = T(P[t_star, ti])[:3, 3]
                for t in range(t_star + 1):
                    if kind == 0:
                        Tc = T(P[t, names.index(slot_tube(meta["args"]["slot"]))])
                        origin, axis = Tc[:3, 3], Tc[:3, 2]
                    else:
                        well = meta["args"]["well"]
                        i, j = int(well[1:]) - 1, "AB".index(well[0])
                        Tc = T(P[t, names.index("tray")])
                        origin = Tc[:3, :3] @ np.array([(i - 2) * TRAY_PITCH, (j - 0.5) * TRAY_PITCH, 0.0]) + Tc[:3, 3]
                        axis = Tc[:3, 2]
                    Tt = T(P[t, ti])
                    if np.linalg.norm(Tt[:3, 3] - target) > NEAR_M:
                        continue
                    Th = np.linalg.inv(T(d["hand_pose"][t])) @ Tt
                    rows["q"].append(qq[t])
                    rows["inhand"].append(np.concatenate([Th[:3, 3], matrix_to_quat(Th[:3, :3])]))
                    rows["origin"].append(origin + shift)
                    rows["axis"].append(axis)
                    rows["kind"].append(kind)
                    rows["target"].append(target + shift)
                    rows["prim"].append(pk)
                    rows["source"].append(str(ep))
    out = {k: np.array(v, np.float32 if k not in ("kind", "prim", "source") else None) for k, v in rows.items()}
    np.savez(OUT, **out)
    k, pr = out["kind"], out["prim"]
    print(f"{len(k)} start states from {len(set(rows['source']))} episodes -> {OUT}: aspirate {(pr == 0).sum()}, "
          f"dispense into tubes {((pr == 1) & (k == 0)).sum()}, into wells {(k == 1).sum()}")
    if len(k):
        depth = np.einsum("ij,ij->i", out["target"] - out["origin"], out["axis"])
        print(f"target height above the container origin: tubes {np.round(np.percentile(depth[k == 0], [0, 50, 100]) * 1000, 1)} mm, "
              f"wells {np.round(np.percentile(depth[k == 1], [0, 50, 100]) * 1000, 1) if (k == 1).any() else '-'} mm")


if __name__ == "__main__":
    main()
