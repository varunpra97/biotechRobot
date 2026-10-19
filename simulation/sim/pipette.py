"""Electronic pipette model: tip attach/eject and liquid handling, driven by
the pipette channels of LabEnv's action ([8] plunger, [9] eject).

Physical pieces stay physical: the pipette is a rigid body held in the
gripper by friction; a tip is a rigid body that sits in its rack until it
is attached and falls under gravity when ejected. What this module adds is
what the simulator can't do on its own:

  - Tip attach: a real tip press-fits onto the nozzle. When the nozzle end
    is pushed 4-9 mm into a tip's collar (the nozzle seals at ~5.5 mm, see
    cad_source/make_lab_consumables.py), coaxial within 3 deg and 0.6 mm,
    a fixed joint locks the tip to the pipette. Deeper than 9 mm is logged
    as crushing the tip.
  - Eject: the joint is disabled; the tip falls.
  - Plunger: moves toward its command at 400 uL/s (an electronic pipette's
    aspiration speed). Raising it aspirates liquid only if the tip's end is
    inside a container AND below that container's liquid surface (else it
    draws air); lowering it dispenses into whichever tube/well the tip end
    is in, or is logged as a spill.
  - Tip-in-container clearance: tips are collision-filtered against tube and
    well interiors (hollow insides a convex collider can't represent), so
    the tip's outer profile is checked here analytically against the
    container's inner wall and floor; any contact is logged as an event.
"""
from __future__ import annotations

import math

import numpy as np

import lab_scene
import liquids

CAPACITY_UL = 1000.0
PLUNGER_RATE = 0.4                       # fraction of capacity per second (400 uL/s)
NOZZLE_SEAL_DEPTH = (0.004, 0.009)       # m of nozzle inside the collar to attach
ATTACH_MAX_RADIAL = 0.0006
ATTACH_MAX_ANGLE = math.radians(3.0)
TIP_END_R = 0.00045                      # tip outer radius at its end
TIP_BODY_TOP_R = 0.0037                  # ... under the collar (z = 52 mm)
TUBE_RIM_Z = 0.058
WALL_CLEARANCE = 0.0                     # contact if the tip's outer surface reaches the wall


def T(pose):
    M = np.eye(4)
    M[:3, :3] = lab_scene.quat_to_matrix(pose[3:7])
    M[:3, 3] = pose[:3]
    return M


def tip_outer_r(h):
    """Tip outer radius at height h above its end."""
    return TIP_END_R + (TIP_BODY_TOP_R - TIP_END_R) * min(max(h, 0.0), lab_scene.TIP_COLLAR_Z) / lab_scene.TIP_COLLAR_Z


class PipetteModel:
    def __init__(self, env, name="pipette_0"):
        self.env = env
        self.path = env.objects[name]["path"]
        self.tips = sorted(n for n in env.objects if n.startswith("tip_"))
        self.used_tips = set()
        self.plunger_cmd = 0.0
        self.plunger = 0.0
        self.eject_prev = 0.0
        self.tip = None
        self.joint_path = None
        self.tip_liquid = None
        self.tip_volume = 0.0
        self.tip_color = (0.2, 0.3, 0.9)
        self.colors = {}
        self.events = []
        self.dt = 1.0 / 60.0

    # ------------------------------------------------------------- state
    def state(self):
        vols = {n: round(L.volume_ul, 2) for n, L in lab_scene.LIQUIDS.items() if L.volume_ul > 0}
        return {"plunger": self.plunger, "tip_attached": 1.0 if self.tip else 0.0,
                "tip_volume_ul": self.tip_volume, "volumes": vols}

    def command(self, plunger, eject):
        self.plunger_cmd = float(np.clip(plunger, 0.0, 1.0))
        if eject > 0.5 and self.eject_prev <= 0.5:
            self._eject()
        self.eject_prev = eject

    def log(self, kind, **kw):
        self.events.append({"step": len(self.env.recorder.rows) if self.env.recorder else None,
                            "event": kind, **kw})

    # ------------------------------------------------------------ geometry
    def nozzle(self):
        """Nozzle end (world) and the nozzle's pointing direction (down
        the pipette, local -y)."""
        P = T(lab_scene.get_prim_world_pose(self.path))
        return P[:3, 3].copy(), -P[:3, 1]

    def tip_end(self):
        Tt = T(lab_scene.get_prim_world_pose(self.env.objects[self.tip]["path"]))
        return Tt[:3, 3].copy(), Tt[:3, 2].copy()

    def containers(self):
        """(name, world transform of the container's local frame, profile)."""
        out = []
        for n, o in self.env.objects.items():
            if n.startswith("tube_"):
                out.append((n, T(lab_scene.get_prim_world_pose(o["path"])), liquids.TUBE))
        tray = T(lab_scene.get_prim_world_pose(self.env.objects["tray"]["path"]))
        for n, L in lab_scene.LIQUIDS.items():
            if n.startswith("tray/"):
                W = tray.copy()
                W[:3, 3] = tray[:3, :3] @ L.offset + tray[:3, 3]
                out.append((n, W, liquids.WELL))
        return out

    def container_of(self, point):
        for name, Tc, prof in self.containers():
            p = np.linalg.inv(Tc) @ np.append(point, 1.0)
            z = p[2]
            if prof.z[0] - 0.003 <= z <= prof.z[-1] and math.hypot(p[0], p[1]) < prof.radius(z):
                return name, Tc, prof, p[:3]
        return None

    # ------------------------------------------------------------- update
    def update(self):
        if self.tip is None:
            self._try_attach()
        else:
            self._check_clearance()
        target = self.plunger_cmd
        step = PLUNGER_RATE * self.dt
        new = self.plunger + float(np.clip(target - self.plunger, -step, step))
        dv = (new - self.plunger) * CAPACITY_UL
        self.plunger = new
        if dv > 1e-9:
            self._aspirate(dv)
        elif dv < -1e-9:
            self._dispense(-dv)

    def attach_geometry(self, name, nozzle=None):
        """Nozzle vs a tip's collar: (Tt, depth of the nozzle end below the
        collar top, radial offset, angle between the axes) -- also logged on
        a failed attach."""
        n_end, n_dir = nozzle or self.nozzle()
        Tt = T(lab_scene.get_prim_world_pose(self.env.objects[name]["path"]))
        up = Tt[:3, 2]
        rel = Tt[:3, 3] + up * lab_scene.TIP_LENGTH - n_end
        depth = float(rel @ up)
        radial = float(np.linalg.norm(rel - depth * up))
        ang = math.acos(np.clip(float(-up @ n_dir), -1, 1))
        return {"Tt": Tt, "depth_m": depth, "radial_m": radial, "angle_deg": math.degrees(ang)}

    def _try_attach(self):
        nozzle = self.nozzle()
        for name in self.tips:
            if name in self.used_tips:
                continue
            g = self.attach_geometry(name, nozzle)
            Tt, depth, radial, ang = g["Tt"], g["depth_m"], g["radial_m"], math.radians(g["angle_deg"])
            if radial > 0.003 or depth < -0.002:
                continue
            if depth > NOZZLE_SEAL_DEPTH[1]:
                self.log("tip_crushed", tip=name, depth_m=depth)
                continue
            if depth >= NOZZLE_SEAL_DEPTH[0] and radial <= ATTACH_MAX_RADIAL and ang <= ATTACH_MAX_ANGLE:
                self._attach(name, Tt)
                return

    def _attach(self, name, Tt):
        import isaacsim.core.experimental.utils.stage as stage_utils
        from pxr import Gf, UsdPhysics

        stage = stage_utils.get_current_stage()
        tip_path = self.env.objects[name]["path"]
        Tp = T(lab_scene.get_prim_world_pose(self.path))
        rel = np.linalg.inv(Tp) @ Tt
        lab_scene.set_kinematic(tip_path, False)
        # one joint per attach, disabled (never deleted) at eject: removing a
        # prim mid-simulation invalidates the arm's articulation view
        self.joint_path = f"/World/Joints/pipette_tip_{len(self.used_tips)}"
        j = UsdPhysics.FixedJoint.Define(stage, self.joint_path)
        j.CreateBody0Rel().SetTargets([self.path])
        j.CreateBody1Rel().SetTargets([tip_path])
        q = lab_scene.matrix_to_quat(rel[:3, :3])
        j.CreateLocalPos0Attr(Gf.Vec3f(*rel[:3, 3]))
        j.CreateLocalRot0Attr(Gf.Quatf(float(q[0]), Gf.Vec3f(*(float(v) for v in q[1:]))))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        self.tip = name
        self.used_tips.add(name)
        self.tip_volume = 0.0
        self.tip_liquid = liquids.LiquidBody(tip_path, liquids.TIP, 0.0, self.tip_color)
        self.log("tip_attached", tip=name)

    def _eject(self):
        import isaacsim.core.experimental.utils.stage as stage_utils

        if self.tip is None:
            self.log("eject_without_tip")
            return
        from pxr import UsdPhysics

        joint = UsdPhysics.Joint(stage_utils.get_current_stage().GetPrimAtPath(self.joint_path))
        joint.CreateJointEnabledAttr().Set(False)
        if self.tip_volume > 0:
            self.log("ejected_with_liquid", volume_ul=self.tip_volume)
        self.log("tip_ejected", tip=self.tip)
        self.tip = None
        self.tip_liquid = None

    def _check_clearance(self):
        end, up = self.tip_end()
        hit = self.container_of(end)
        if hit is None:
            return
        name, Tc, prof, p = hit
        if p[2] < prof.z[0]:
            self.log("tip_hit_floor", container=name, below_m=float(prof.z[0] - p[2]))
        # where the tip crosses the container's rim plane, is its outer
        # surface inside the inner wall?
        Rc = Tc[:3, :3]
        up_c = Rc.T @ up
        if up_c[2] <= 0.1:
            return
        h = (prof.z[-1] - p[2]) / up_c[2]          # along the tip from its end to the rim plane
        at_rim = p + h * up_c
        gap = prof.radius(prof.z[-1]) - (math.hypot(at_rim[0], at_rim[1]) + tip_outer_r(h))
        if gap < WALL_CLEARANCE:
            self.log("tip_touches_wall", container=name, overlap_m=float(-gap))

    def _liquid(self, name, color=None):
        L = lab_scene.LIQUIDS.get(name)
        if L is None and name.startswith("tube_"):
            L = liquids.LiquidBody(self.env.objects[name]["path"], liquids.TUBE, 0.0,
                                   color or (0.2, 0.3, 0.9))
            lab_scene.LIQUIDS[name] = L
        return L

    def _aspirate(self, dv):
        if self.tip is None:
            self.log("aspirate_without_tip", volume_ul=dv)
            return
        end, _ = self.tip_end()
        hit = self.container_of(end)
        L = self._liquid(hit[0]) if hit else None
        if L is None or L.volume_ul <= 0 or hit[3][2] >= L.profile.height(L.volume_ul):
            self.log("aspirated_air", volume_ul=dv)
            return
        take = min(dv, L.volume_ul)
        L.set_volume(L.volume_ul - take)
        self.tip_color = L.color
        self.tip_volume += take
        self.tip_liquid.set_color(L.color)
        self.tip_liquid.set_volume(self.tip_volume)
        if take < dv:
            self.log("aspirated_air", volume_ul=dv - take)

    def _dispense(self, dv):
        if self.tip is None or self.tip_volume <= 0:
            return
        out = min(dv, self.tip_volume)
        self.tip_volume -= out
        self.tip_liquid.set_volume(self.tip_volume)
        end, _ = self.tip_end()
        hit = self.container_of(end)
        if hit is None:
            self.log("spilled", volume_ul=out)
            return
        L = self._liquid(hit[0], self.tip_color)
        if L.volume_ul <= 0:
            L.set_color(self.tip_color)
        L.set_volume(L.volume_ul + out)
