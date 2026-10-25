"""Client of the localizer server (server.py), for the robot / simulator
process: numpy and the standard library only -- no open3d, no pyzmq.

    from client import LocalizerClient
    loc = LocalizerClient()                                  # server.py must be running
    loc.init({"object_names": names, "layout": layout}, poses)   # scene map (sim: the episode's poses)
    every control step:
        loc.step({"wrist_top": rgb, "wrist_side": rgb}, hand_pose7, finger_pos, q, held=None)
        p = loc.get_prim_world_pose("/World/Platform/Tube_r0_c4")   # [x,y,z,qw,qx,qy,qz]
        loc.info("tube_r0_c4")      # {"source": observed|tracked|parent|hand|map, "sigma_mm", "steps_since_seen"}

wait=False (default) pipelines the localizer with the control loop: step()
sends this step's images and returns at once; the poses you read are the
reply to the PREVIOUS step (one step stale -- within the planner's budget),
so the loop never waits on the localizer unless it falls a whole step
behind. wait=True blocks for this step's reply.

TeacherFinisher(..., pose_fn=loc.get_prim_world_pose) drops it in.
"""
from __future__ import annotations

import socket

import numpy as np

import wire

_PRIM_NAMES = {"TubeHolder": "holder", "TipRack": "tip_rack", "WasteBin": "waste_bin", "SampleTray": "tray",
               "Pipette0": "pipette_0", "CentrifugeBody": "centrifuge_body"}


def name_from_prim(path: str) -> str:
    leaf = path.rstrip("/").split("/")[-1]
    if leaf.startswith("Tube_"):
        return "tube_" + leaf[5:]
    return _PRIM_NAMES.get(leaf, leaf)


class LocalizerClient:
    def __init__(self, host="127.0.0.1", port=5577, timeout_s=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.state = None
        self.targets = {}
        self._pending = False

    def init(self, meta: dict, poses, cameras=("wrist_top", "wrist_side"), image_lag=0.25):
        meta = {"object_names": list(meta["object_names"]), "layout": meta["layout"]}
        wire.send(self.sock, {"type": "init", "meta": meta, "cameras": list(cameras), "image_lag": image_lag},
                  {"poses": np.asarray(poses, np.float64)})
        self._take(wire.recv(self.sock)[0])

    def _take(self, h):
        self.state = h
        for name, o in h["objects"].items():
            if "targets" in o:
                self.targets[name] = o["targets"]

    def step(self, images: dict, hand_pose7, finger_pos, q=None, held=None, wait=False, targets=False):
        if self._pending:                                  # the previous step's reply
            self._take(wire.recv(self.sock)[0])
            self._pending = False
        arrs = {"hand_pose": np.asarray(hand_pose7, np.float64), "finger_pos": np.asarray(finger_pos, np.float64)}
        if q is not None:
            arrs["q"] = np.asarray(q, np.float64)[:7]
        for k, v in images.items():
            arrs["img:" + k] = np.ascontiguousarray(v, np.uint8)
        wire.send(self.sock, {"type": "step", "held": held, "targets": targets}, arrs)
        if wait:
            self._take(wire.recv(self.sock)[0])
        else:
            self._pending = True
        return self.state

    def get_prim_world_pose(self, path):
        o = self.state["objects"].get(name_from_prim(path)) if self.state else None
        return None if o is None else np.array(o["pose"])

    __call__ = get_prim_world_pose

    def info(self, name):
        o = self.state["objects"][name]
        return {k: o[k] for k in ("source", "sigma_mm", "steps_since_seen")}

    def target(self, name, target):
        """World xyz of a named target, as of the last reply that carried targets (init, or a
        step(..., targets=True) -- ask for them on the steps where you need them fresh)."""
        return np.asarray(self.targets[name][target], float)

    def close(self):
        try:
            if self._pending:
                wire.recv(self.sock)
            wire.send(self.sock, {"type": "close"})
        finally:
            self.sock.close()
