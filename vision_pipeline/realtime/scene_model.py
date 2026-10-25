"""The scene map the real-time localizer tracks: every object on the bench as
a mesh in its own frame, a world pose, and named target points (the
coordinates a planner asks for: a tube's grasp point, holder slot "A5"'s
bore opening, rack tip "C7", tray well "W3", centrifuge bore "B2", ...),
also in the object's frame so they follow the object's tracked pose.

Two sources:
  scene_from_sim(...)   -- the simulator's own meshes and the episode's
                           recorded poses (ground truth; for evaluation the
                           poses are perturbed to stand in for map error);
  the offline vision pipeline's export (per-object meshes + poses) -- the
                           real deployment source; same SceneObject fields.

Classes: tube, tip, holder, tip_rack, tray, pipette, centrifuge, waste_bin,
bench. `movable` marks what the arm can pick up (tracked with a looser
motion model); fixtures barely move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

from camera_model import _lab_scene, pose_to_matrix

ROW = "ABCDEFGHIJ"


@dataclass
class SceneObject:
    name: str
    cls: str
    mesh: trimesh.Trimesh                 # object frame
    T: np.ndarray                         # 4x4 world-from-object
    movable: bool
    targets: dict = field(default_factory=dict)      # name -> (3,) point in the object frame
    parent: str = None                    # seated in / hanging from this object (moves with it, within clearance)

    def world_targets(self, T=None):
        T = self.T if T is None else T
        return {k: T[:3, :3] @ v + T[:3, 3] for k, v in self.targets.items()}

    @property
    def center_local(self):
        c = self.__dict__.get("_center")
        if c is None:
            c = self.__dict__["_center"] = self.mesh.bounds.mean(0)
        return c


def _yaw_pose(x, y, z, yaw_deg):
    a = math.radians(yaw_deg)
    T = np.eye(4)
    T[:3, :3] = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = (x, y, z)
    return T


def _common(mesh):
    lo, hi = mesh.bounds
    c = (lo + hi) / 2
    return {"center": c, "top": np.array([c[0], c[1], hi[2]])}


_MESH_CACHE = {}


def _mesh(path):
    if path not in _MESH_CACHE:
        _MESH_CACHE[path] = trimesh.load(path, force="mesh")
    return _MESH_CACHE[path]


def _centrifuge_local(ls):
    """Body mesh (lid omitted, as the simulator loads it) and its six rotor
    bore openings, in the centrifuge's frame."""
    import centrifuge_geometry as cg
    parts = ls.centrifuge_parts_local()
    body = trimesh.util.concatenate([p for i, p in enumerate(parts)
                                     if i not in ls.CENTRIFUGE_LID_PARTS and len(p.faces) > 4])
    hub_top = parts[ls.CENTRIFUGE_HUB_PART].vertices[:, 2].max()
    hub_c = parts[ls.CENTRIFUGE_HUB_PART].vertices.mean(0)
    bores = []
    for idx in ls.CENTRIFUGE_SLEEVE_PARTS:
        c, a = cg._fit_axis(parts[idx].vertices)
        s_open = (hub_top - c[2]) / a[2]
        bores.append(c + s_open * a)
    # number bores counter-clockwise from the one nearest the local -y side
    ang = [math.atan2(b[1] - hub_c[1], b[0] - hub_c[0]) for b in bores]
    order = np.argsort([(t + math.pi / 2) % (2 * math.pi) for t in ang])
    targets = {f"bore_{k + 1}": bores[i] for k, i in enumerate(order)}
    targets["hub"] = np.array([hub_c[0], hub_c[1], hub_top])
    return body, targets


def scene_from_sim(meta: dict, poses: np.ndarray) -> list[SceneObject]:
    """All objects of a recorded episode at one step: the recorded task
    objects (tubes, tips, pipette, tray, centrifuge) at their recorded
    poses, the fixtures (holder, tip rack, waste bin, bench) placed from the
    episode's layout exactly as the simulator placed them."""
    ls = _lab_scene()
    names, layout = meta["object_names"], meta["layout"]
    plat = layout["platform"]
    pz = ls.BENCH_Z + 0.04
    mm = ls.MM_TO_M
    objs = []

    # fixtures
    holder = _mesh(ls.TUBE_HOLDER_OBJ)
    t = _common(holder)
    for r, y in enumerate(ls.HOLDER_HOLE_Y_MM):
        for c, x in enumerate(ls.HOLDER_HOLE_X_MM):
            t[f"slot_{ROW[r]}{c + 1}"] = np.array([x * mm, y * mm, ls.HOLDER_THICKNESS_MM * mm])
    hx, hy, hyaw = plat["holder"]
    objs.append(SceneObject("holder", "holder", holder, _yaw_pose(hx, hy, pz, hyaw), False, t))

    rack = _mesh(ls.TIP_RACK_OBJ)
    t = _common(rack)
    for r in range(10):
        for c in range(10):
            t[f"pos_{ROW[r]}{c + 1}"] = np.array([(c - 4.5) * ls.TIP_PITCH, (r - 4.5) * ls.TIP_PITCH, ls.TIP_RACK_HEIGHT])
    rx, ry, ryaw = plat["tip_rack"]
    objs.append(SceneObject("tip_rack", "tip_rack", rack, _yaw_pose(rx, ry, pz, ryaw), False, t))

    wb = _mesh(ls.WASTE_BIN_OBJ)
    t = _common(wb)
    t["drop"] = t["top"]
    wx, wy, wyaw = plat["waste_bin"]
    objs.append(SceneObject("waste_bin", "waste_bin", wb, _yaw_pose(wx, wy, pz, wyaw), False, t))

    bench = trimesh.creation.box((ls.PLATFORM_W, ls.PLATFORM_D, 0.04))
    objs.append(SceneObject("bench", "bench", bench, _yaw_pose(ls.PLATFORM_CX, ls.PLATFORM_CY, ls.BENCH_Z + 0.02, 0),
                            False, {}))

    # recorded objects
    tube, tip, tray, pip = (_mesh(ls.TEST_TUBE_OBJ), _mesh(ls.TIP_OBJ), _mesh(ls.SAMPLE_TRAY_OBJ),
                            _mesh(ls.PIPETTE_OBJ))
    cbody, ctargets = _centrifuge_local(ls)
    for i, n in enumerate(names):
        T = pose_to_matrix(poses[i])
        if n.startswith("tube_"):
            L = 0.058
            objs.append(SceneObject(n, "tube", tube, T, True,
                                    {"tip": np.zeros(3), "rim": np.array([0, 0, L]), "grasp": np.array([0, 0, L - 0.012]),
                                     "center": np.array([0, 0, L / 2])}))
        elif n.startswith("tip_"):
            objs.append(SceneObject(n, "tip", tip, T, True,
                                    {"top": np.array([0, 0, ls.TIP_LENGTH]), "end": np.zeros(3),
                                     "center": np.array([0, 0, ls.TIP_LENGTH / 2])}))
        elif n == "tray":
            t = _common(tray)
            for j in range(2):
                for c in range(5):
                    t[f"well_{ROW[j]}{c + 1}"] = np.array([(c - 2) * 0.009, (j - 0.5) * 0.009, tray.bounds[1, 2]])
            objs.append(SceneObject(n, "tray", tray, T, True, t))
        elif n.startswith("pipette"):
            t = _common(pip)
            t["nozzle"] = np.array([0.0, 0.0, 0.0])
            objs.append(SceneObject(n, "pipette", pip, T, True, t))
        elif n.startswith("centrifuge"):
            t = _common(cbody)
            t.update(ctargets)
            objs.append(SceneObject(n, "centrifuge", cbody, T, False, t))
    _seat(objs, "tube", "holder", "slot_", depth=(ls.HOLDER_THICKNESS_MM - ls.HOLDER_BORE_FLOOR_MM) * mm)
    _seat(objs, "tip", "tip_rack", "pos_", depth=ls.TIP_RACK_HEIGHT - (ls.TIP_RACK_HEIGHT - ls.TIP_COLLAR_Z))
    return objs


def _seat(objs, child_cls, parent_name, prefix, depth, tol=0.003):
    """Mark objects of child_cls standing in one of parent's openings (their
    base `depth` below an opening, within tol) as seated in it."""
    par = next((o for o in objs if o.name == parent_name), None)
    if par is None:
        return
    ops = np.array([v for k, v in par.world_targets().items() if k.startswith(prefix)])
    for o in objs:
        if o.cls != child_cls:
            continue
        base = o.T[:3, 3]
        d = np.linalg.norm(ops[:, :2] - base[:2], axis=1)
        j = int(np.argmin(d))
        if d[j] < tol and abs((ops[j, 2] - base[2]) - depth) < tol:
            o.parent = parent_name
