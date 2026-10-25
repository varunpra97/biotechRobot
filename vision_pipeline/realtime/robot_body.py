"""The robot's own body as the wrist cameras see it: the Franka hand, its two
fingers (at their measured openings) and the camera bracket -- meshes in the
hand frame. The localizer raycasts them together with the scene so that an
object hidden behind the gripper is known to be hidden (no edges expected
there) instead of being fitted to the gripper's own edges.

Meshes: the Franka hand / finger visuals from MuJoCo Menagerie (the same
Franka Hand geometry the simulator's arm uses; hand frame = panda_hand) and
the bracket from the simulation's CAD (lab_scene.WRIST_CAM_MOUNT_OBJ, hand
frame). On the real robot: the same hand, plus the mount's CAD.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from camera_model import _lab_scene

WORKSPACE = Path(__file__).resolve().parents[3]
MENAGERIE_PANDA = Path(os.environ.get("MENAGERIE_PANDA", WORKSPACE / "mujoco_menagerie-main" / "franka_emika_panda"))
FINGER_Z = 0.0584                                  # finger base along the hand's z (panda.xml)


@lru_cache(maxsize=1)
def _meshes():
    a = MENAGERIE_PANDA / "assets"
    hand = trimesh.util.concatenate([trimesh.load(a / f"hand_{i}.obj", force="mesh") for i in range(5)])
    finger = trimesh.util.concatenate([trimesh.load(a / f"finger_{i}.obj", force="mesh") for i in range(2)])
    mount = trimesh.load(_lab_scene().WRIST_CAM_MOUNT_OBJ, force="mesh")
    return hand, finger, mount


def body_parts(finger_pos=(0.04, 0.04)):
    """[(name, mesh, T_hand_part)] for the hand, both fingers, the bracket.
    finger_pos: each finger's slide (m) from closed, as recorded."""
    hand, finger, mount = _meshes()
    I = np.eye(4)
    L = np.eye(4)
    L[:3, 3] = (0, finger_pos[0], FINGER_Z)
    R = np.diag([-1.0, -1.0, 1.0, 1.0])               # quat (0 0 0 1): 180 deg about z
    R[:3, 3] = (0, -finger_pos[1], FINGER_Z)
    return [("hand", hand, I), ("finger_left", finger, L), ("finger_right", finger, R), ("camera_mount", mount, I)]


def _quat(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@lru_cache(maxsize=1)
def _arm_chain():
    """[(body name, T_parent_body at q=0, has hinge joint, visual mesh)] link0..link7, hand,
    from Menagerie's panda.xml (hinges about local z)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(MENAGERIE_PANDA / "panda.xml").getroot()
    files = {m.get("name") or Path(m.get("file")).stem: m.get("file") for m in root.iter("mesh")}
    chain = []
    body = root.find("worldbody").find("body")
    while body is not None and body.get("name") != "hand":
        T = np.eye(4)
        T[:3, 3] = [float(v) for v in body.get("pos", "0 0 0").split()]
        T[:3, :3] = _quat([float(v) for v in body.get("quat", "1 0 0 0").split()])
        parts = [trimesh.load(MENAGERIE_PANDA / "assets" / files[g.get("mesh")], force="mesh")
                 for g in body.findall("geom") if g.get("class") == "visual" and g.get("mesh")]
        chain.append((body.get("name"), T, body.find("joint") is not None,
                      trimesh.util.concatenate(parts) if parts else None))
        body = next((b for b in body.findall("body")), None)
    T = np.eye(4)
    T[:3, 3] = [float(v) for v in body.get("pos").split()]
    T[:3, :3] = _quat([float(v) for v in body.get("quat").split()])
    chain.append(("hand", T, False, None))
    return chain


def arm_parts(q, T_world_hand):
    """[(name, mesh, T_world_link)] of link0..link7 for joint angles q (7,),
    anchored so the chain ends at the measured hand pose (the base pose
    need not be known)."""
    chain = _arm_chain()
    Ts, T, j = [], np.eye(4), 0
    for name, Tp, hinge, mesh in chain:
        T = T @ Tp
        if hinge:
            c, s = np.cos(q[j]), np.sin(q[j])
            Rz = np.eye(4)
            Rz[:2, :2] = [[c, -s], [s, c]]
            T = T @ Rz
            j += 1
        Ts.append((name, mesh, T))
    T_base = T_world_hand @ np.linalg.inv(Ts[-1][2])
    return [(n, m, T_base @ T) for n, m, T in Ts if m is not None]
