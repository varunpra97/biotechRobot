"""What the planner calls: the scene map tracked live from the robot's
cameras, every object's coordinates on demand.

    loc = SceneLocalizer(scene_objects)            # from the offline map (or scene_from_sim)
    every control step (15 Hz):
        loc.step({"wrist_top": rgb, "wrist_side": rgb}, hand_pose7, finger_pos, q,
                 held=name_or_None)
    then any of:
        loc.get_prim_world_pose("/World/Platform/Tube_r0_c4")   # [x,y,z,qw,qx,qy,qz], like lab_scene's
        loc.target("tube_r0_c4", "grasp")                        # world xyz of a named target
        loc.state_message()                                      # all objects, JSON-ready

PerceivedPose(loc) is a drop-in for lab_scene.get_prim_world_pose (the RL
teacher's pose_fn hook). Poses are the localizer's latest estimate; for an
object no camera has seen yet, the map's pose.

Timing: images are posed by the hand pose interpolated to the image's own
time (`image_lag` control steps before the pose sample; 0.25 measured in
the simulator's recordings -- with the wrong time, a hand moving 10 mm per
step puts every edge 1-2 px off). On the real robot, pass timestamps' worth
of lag instead.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from camera_model import exterior_camera, interpolate_pose7, pose_to_matrix, wrist_cameras
from scene_model import SceneObject
from scene_tracker import Frame, SceneTracker

# simulator prim paths <-> scene-map names
_PRIM_NAMES = {"TubeHolder": "holder", "TipRack": "tip_rack", "WasteBin": "waste_bin", "SampleTray": "tray",
               "Pipette0": "pipette_0", "CentrifugeBody": "centrifuge_body"}


def name_from_prim(path: str) -> str:
    leaf = path.rstrip("/").split("/")[-1]
    if leaf.startswith("Tube_"):
        return "tube_" + leaf[5:]
    return _PRIM_NAMES.get(leaf, leaf)


def _matrix_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class SceneLocalizer:
    def __init__(self, objects: list[SceneObject], cameras=("wrist_top", "wrist_side"), image_lag=0.25,
                 **tracker_kw):
        cams = dict(wrist_cameras())
        cams["platform_cam1"] = exterior_camera()
        self.cams = {k: cams[k] for k in cameras}
        self.tracker = SceneTracker(objects, **tracker_kw)
        self.index = {o.name: i for i, o in enumerate(objects)}
        self.image_lag = image_lag
        self.prev = None                          # (hand_pose7, q) of the previous step
        self.states = self.tracker.state()
        self.t_step = None
        self.n = 0
        self.last_seen = {}                       # object -> step a camera last supported it

    def step(self, images: dict, hand_pose7, finger_pos, q=None, held=None):
        """One control step: images {camera: HxW gray or HxWx3 RGB}, the hand
        pose (x,y,z,qw,qx,qy,qz) and finger openings sampled this step, arm
        joints, and which object (if any) the gripper holds."""
        t0 = time.perf_counter()
        hand_pose7 = np.asarray(hand_pose7, float)
        T_hand = pose_to_matrix(hand_pose7)
        if self.prev is not None and self.image_lag:
            p_img = interpolate_pose7(self.prev[0], hand_pose7, 1.0 - self.image_lag)
            q_img = None if q is None or self.prev[1] is None else \
                self.image_lag * self.prev[1] + (1 - self.image_lag) * np.asarray(q, float)
        else:
            p_img, q_img = hand_pose7, q
        T_img = pose_to_matrix(p_img)
        self.tracker.set_held(held, T_hand)
        frames = []
        for name, img in images.items():
            if name not in self.cams or img is None:
                continue
            g = img if img.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
            cam = self.cams[name]
            frames.append(Frame(name, g, cam.K, cam.world_from_cam(T_img), T_img, q_img, not cam.fixed))
        self.states = self.tracker.update(frames, T_hand, finger_pos, q)
        for st in self.states:
            if st.updated:
                self.last_seen[st.name] = self.n
        self.prev = (hand_pose7, None if q is None else np.asarray(q, float))
        self.t_step = time.perf_counter() - t0
        self.n += 1
        return self.states

    # ---- queries ----
    def pose(self, name) -> np.ndarray:
        """4x4 world pose of an object."""
        return self.states[self.index[name]].T

    def get_prim_world_pose(self, path):
        """[x, y, z, qw, qx, qy, qz] like lab_scene.get_prim_world_pose; None if unknown."""
        name = name_from_prim(path)
        if name not in self.index:
            return None
        T = self.pose(name)
        return np.r_[T[:3, 3], _matrix_to_quat(T[:3, :3])]

    def target(self, name, target):
        return self.states[self.index[name]].targets[target]

    def source(self, i) -> str:
        """Where object i's current pose comes from: "observed" (a camera supported
        it this step), "tracked" (seen before, not now), "parent" (seated, placed on
        its holder / rack's estimate, not seen itself), "hand" (held, carried by the
        hand since last seen), "map" (no camera has seen it)."""
        s, f = self.states[i], self.tracker.filters[i]
        if s.updated:
            return "observed"
        if f.ref == "hand":
            return "hand"
        if s.name in self.last_seen:
            return "tracked"
        return "parent" if f.ref == "parent" else "map"

    def info(self, name) -> dict:
        i = self.index[name]
        return {"source": self.source(i), "sigma_mm": round(self.states[i].sigma_mm, 3),
                "steps_since_seen": None if name not in self.last_seen else self.n - self.last_seen[name]}

    def state_message(self, include_targets=True) -> dict:
        """Every object: pose, 1-sigma position uncertainty, whether a camera
        supported it this step, what it sits in / who holds it, and its named
        targets (grasp points, slot / bore / well openings, ...) in world xyz."""
        out = []
        for s, f in zip(self.states, self.tracker.filters):
            T = s.T
            d = {"name": s.name, "class": s.cls,
                 "pose": [*map(float, T[:3, 3]), *map(float, _matrix_to_quat(T[:3, :3]))],
                 "sigma_mm": round(s.sigma_mm, 3), "observed": bool(s.updated), "support": int(s.support),
                 "source": self.source(self.index[s.name]),
                 "steps_since_seen": None if s.name not in self.last_seen else self.n - self.last_seen[s.name],
                 "frame": f.ref, "parent": f.obj.parent if f.ref == "parent" else None,
                 "held": f.ref == "hand"}
            if include_targets:
                d["targets"] = {k: [round(float(v), 5) for v in p] for k, p in s.targets.items()}
            out.append(d)
        return {"step": self.n, "time": time.time(), "frame": "world", "units": "m",
                "latency_ms": None if self.t_step is None else round(self.t_step * 1e3, 1), "objects": out}


class PerceivedPose:
    """Drop-in for lab_scene.get_prim_world_pose, served by a SceneLocalizer."""

    def __init__(self, loc: SceneLocalizer, fallback=None):
        self.loc, self.fallback = loc, fallback

    def get_prim_world_pose(self, path):
        p = self.loc.get_prim_world_pose(path)
        return p if p is not None or self.fallback is None else self.fallback(path)

    __call__ = get_prim_world_pose
