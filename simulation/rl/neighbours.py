"""The objects around a target on the bench, for the full-milestone teachers
(the m2 tray teacher pushed the pipette lying 9 cm from the tray: its scene
had nothing else in it).

Data side (numpy only; the start-state scripts): each recorded frame's
neighbour poses, shifted into the RL env's frame -- the RL env has the arm at
lab_scene's default ARM_POS, while randomized layouts (layout.py) also move
the arm, so object poses are taken relative to that episode's arm.

Env side (Isaac Lab; imported after the app is up): the neighbours as rigid
objects, reset to an entry's poses, and the disturbance test (the generator
fails a trial when any other object moved > 1 mm, expert.others_unmoved):
  - pipette: the real mesh (rl/assets/pipette.usda), 150 g, dynamic;
  - tray: the real mesh (rl/assets/tray.usda), 15 g, dynamic;
  - tip_rack: the rack with its 100 standing tips as one 100 x 100 x 63 mm
    block (tip collars at z 1.003), dynamic -- knocking it means knocking tips;
  - waste_bin: a 90 x 90 x 70 mm box, kinematic (an obstacle only).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]
NAMES = ("tray", "pipette", "tip_rack", "waste_bin")
BENCH_TOP = 0.94
DEFAULT_ARM_XY = (0.0, 1.1)                 # lab_scene.ARM_POS
DEFAULT_PLATFORM = {"tip_rack": (0.32, 1.53, 0.0), "waste_bin": (0.49, 1.53, 0.0)}   # lab_scene defaults
TIP_BLOCK = (0.10, 0.10, 0.063)
BIN_BOX = (0.09, 0.09, 0.07)
DISTURBED_M = 0.0015                         # generator: 1 mm (sim noise margin)


def yaw_quat_wxyz(deg):
    a = math.radians(deg) / 2
    return np.array([math.cos(a), 0.0, 0.0, math.sin(a)])


def arm_shift(meta):
    """Translation taking this episode's world into the RL env's (arm at its default spot)."""
    lay = meta.get("layout") if isinstance(meta.get("layout"), dict) else {}
    arm = (lay.get("platform") or {}).get("arm")
    if arm is None:
        return np.zeros(3)
    return np.array([DEFAULT_ARM_XY[0] - arm[0], DEFAULT_ARM_XY[1] - arm[1], 0.0])


def frame_poses(meta, object_poses_t):
    """Neighbour poses at one frame, (len(NAMES), 7) [x, y, z, qw, qx, qy, qz], shifted into the RL env."""
    shift = arm_shift(meta)
    names = meta.get("object_names") or []
    lay = meta.get("layout") if isinstance(meta.get("layout"), dict) else {}
    plat = lay.get("platform") or {}
    out = np.zeros((len(NAMES), 7))
    for i, n in enumerate(NAMES):
        if n in ("tray", "pipette"):
            p = np.array(object_poses_t[names.index("pipette_0" if n == "pipette" else "tray")], float)
        else:
            x, y, yaw = plat.get(n, DEFAULT_PLATFORM[n])
            h = (TIP_BLOCK if n == "tip_rack" else BIN_BOX)[2]
            p = np.concatenate([[x, y, BENCH_TOP + h / 2], yaw_quat_wxyz(yaw)])
        p[:3] += shift
        out[i] = p
    return out


# ------------------------------------------------------------------ env side
def object_cfgs(exclude):
    """RigidObjectCfg per neighbour (all but `exclude`, the task's own target)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    def props(kinematic=False):
        return sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                max_depenetration_velocity=0.5, sleep_threshold=0.0,
                                                kinematic_enabled=kinematic)
    plastic = sim_utils.RigidBodyMaterialCfg(static_friction=0.3, dynamic_friction=0.25, restitution=0.1)
    cfgs = {
        "tray": RigidObjectCfg(prim_path="/World/envs/env_.*/NbTray", spawn=sim_utils.UsdFileCfg(
            usd_path=str(SIM / "rl" / "assets" / "tray.usda"), rigid_props=props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 1.73, BENCH_TOP))),
        "pipette": RigidObjectCfg(prim_path="/World/envs/env_.*/NbPipette", spawn=sim_utils.UsdFileCfg(
            usd_path=str(SIM / "rl" / "assets" / "pipette.usda"), rigid_props=props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.21, 1.58, BENCH_TOP + 0.041))),
        "tip_rack": RigidObjectCfg(prim_path="/World/envs/env_.*/NbTipRack", spawn=sim_utils.CuboidCfg(
            size=TIP_BLOCK, rigid_props=props(), mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=plastic,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.45, 0.8))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 1.53, BENCH_TOP + TIP_BLOCK[2] / 2))),
        "waste_bin": RigidObjectCfg(prim_path="/World/envs/env_.*/NbWasteBin", spawn=sim_utils.CuboidCfg(
            size=BIN_BOX, rigid_props=props(kinematic=True), mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=plastic,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.49, 1.53, BENCH_TOP + BIN_BOX[2] / 2))),
    }
    return {n: c for n, c in cfgs.items() if n != exclude}


class Neighbours:
    """The neighbour rigid objects of one env: spawn, reset to library poses, disturbance."""

    def __init__(self, exclude):
        self.names = [n for n in NAMES if n != exclude]
        self.idx = [NAMES.index(n) for n in self.names]
        self.cfgs = object_cfgs(exclude)
        self.objs = {}

    def spawn(self, scene):
        from isaaclab.assets import RigidObject
        for n in self.names:
            self.objs[n] = RigidObject(self.cfgs[n])
            scene.rigid_objects[f"nb_{n}"] = self.objs[n]

    def init_buffers(self, num_envs, device, lib_neigh):
        """lib_neigh: (M, len(NAMES), 7) xyz + wxyz from the start-state library."""
        import torch
        t = np.asarray(lib_neigh, np.float32)
        self.lib = torch.tensor(np.concatenate([t[..., :3], t[..., 4:7], t[..., 3:4]], -1), device=device)   # xyzw
        self.start = torch.zeros(num_envs, len(self.names), 3, device=device)
        self.dynamic = torch.tensor([n != "waste_bin" for n in self.names], device=device)

    def reset(self, env_ids, pick, origins, jitter_m=0.0):
        import torch
        for j, (n, i) in enumerate(zip(self.names, self.idx)):
            pose = self.lib[pick, i].clone()
            if jitter_m:
                pose[:, :2] += jitter_m * (2 * torch.rand(len(env_ids), 2, device=pose.device) - 1)
            pose[:, :3] += origins
            self.objs[n].write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            self.objs[n].write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=pose.device),
                                                          env_ids=env_ids)
            self.start[env_ids, j] = pose[:, :3]

    def moved(self):
        """(N,) how far the most-moved dynamic neighbour is from its reset position (m)."""
        import torch
        pos = torch.stack([self.objs[n].data.root_pos_w.torch for n in self.names], 1)
        d = (pos - self.start).norm(dim=-1) * self.dynamic.float()
        return d.max(-1).values
