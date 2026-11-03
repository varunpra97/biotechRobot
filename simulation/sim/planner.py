"""Collision-checked motion planning for the Franka in the live scene.

A primitive gives a list of TCP waypoints (position + hand orientation).
The planner interpolates densely between them (<= 5 mm / 3 deg), solves
6-DOF IK at every step, and checks every configuration with PhysX overlap
queries on the arm's real colliders (and on a held object's collider) --
the same validated method as the original pick-and-place collector, now
reusable mid-episode by any primitive.

Planning teleports the arm, so it runs "offline" on a frozen world: every
task object is made kinematic in place, a held object is carried along
rigidly with the hand for checking, and afterwards the arm, the objects and
their dynamic/kinematic state are restored exactly before the plan is
executed physically by LabEnv.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import ik_solver
import lab_scene

MAX_STEP_POS = 0.005
GRIPPER_BODIES = [lab_scene.HAND_PATH, lab_scene.LEFT_FINGER_PATH, lab_scene.RIGHT_FINGER_PATH]
MAX_STEP_ROT = math.radians(3.0)


def slerp_R(R0, R1, t):
    w = ik_solver.rotation_error(R1, R0)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return R0.copy()
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = th * t
    return (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R0


def frame_from_axis(z, y_hint):
    """Hand rotation with approach axis z and closing axis as close to
    y_hint as possible (orthogonalized)."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    y = np.asarray(y_hint, float) - (np.asarray(y_hint, float) @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def pose_matrix(p):
    T = np.eye(4)
    T[:3, :3] = lab_scene.quat_to_matrix(p[3:7])
    T[:3, 3] = p[:3]
    return T


RENDER_FLUSH_UPDATES = 3        # app updates after a restore, so camera frames show the restored world


@dataclass
class Waypoint:
    phase: str
    pos: np.ndarray
    R: np.ndarray
    fingers: float            # finger joint value (m per finger) assumed while planning/checking
    gripper: float = None     # gripper action during execution (0 open .. 1 closed); default from fingers
    speed: float = 0.12       # m/s TCP speed for the segment ending here
    ang_speed: float = 0.8    # rad/s
    allow: tuple = ()         # extra prim-path prefixes allowed to touch (e.g. a tube being grasped)

    def gripper_cmd(self):
        if self.gripper is not None:
            return self.gripper
        return float(np.clip(1.0 - self.fingers / lab_scene.GRIPPER_STROKE, 0.0, 1.0))


@dataclass
class Plan:
    segments: list = field(default_factory=list)   # [(Waypoint, [q, ...], length_m, angle_rad)]
    ok: bool = True
    error: str = ""
    held_shift_m: float = 0.0   # held object's in-hand shift across planning (should be ~0)


class Planner:
    # Rehearse held-object motions with the fingers at their measured
    # positions (see plan()); False restores the old nominal-width behaviour.
    hold_fingers_when_held = True
    # After a rehearsal, hold a held object kinematic while the pads re-grip
    # it (see _restore); False: it stays dynamic throughout.
    held_regrip_kinematic = True

    def __init__(self, simulation_app, arm, task_objects):
        """task_objects: prim paths of rigid task objects (frozen while planning)."""
        from omni.physx import get_physx_scene_query_interface

        self.app = simulation_app
        self.arm = arm
        self.task_objects = list(task_objects)
        self.hand_idx = arm.get_body_index(lab_scene.HAND_PATH.rsplit("/", 1)[-1])
        self.limits = np.asarray(arm.get_dof_limits())[0]
        self.sqi = get_physx_scene_query_interface()
        self.arm_cols = [p for p in lab_scene.collider_paths("/World/Arm") if "/link0/link1/" in p]
        import franka_kinematics as fk
        self.kin, err, rerr = fk.calibrate(arm, simulation_app,
                                           lambda: lab_scene.get_prim_world_pose(lab_scene.HAND_PATH),
                                           lab_scene.TCP_LOCAL)
        print(f"[planner] analytic Franka kinematics calibrated: worst hand error "
              f"{err*1000:.3f} mm / {math.degrees(rerr):.4f} deg", flush=True)
        if err > 5e-4 or rerr > math.radians(0.1):
            raise RuntimeError("analytic kinematics don't match the simulated arm")

    # -------------------------------------------------------------- queries
    def overlaps(self, paths):
        from pxr import PhysicsSchemaTools, Sdf

        hits = set()
        for p in paths:
            def report(hit):
                hits.add(hit.collision)
                return True
            p0, p1 = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(p))
            self.sqi.overlap_shape(p0, p1, report, False)
        return hits

    def collisions(self, held=(), allow=()):
        """Interpenetrations of the arm (and of held objects) with anything
        else, as (who, what) pairs. allow: prefixes that may be touched."""
        held = list(held)

        def ok(h):
            return (h.startswith("/World/Arm") or any(h.startswith(a) for a in allow)
                    or any(h.startswith(x) for x in held))

        bad = [("arm", h) for h in self.overlaps(self.arm_cols) if not ok(h)]
        for x in held:
            bad += [("held", h) for h in self.overlaps(lab_scene.collider_paths(x)) if not ok(h)]
        return bad

    def solve(self, pos, R, q0):
        """Pure-math IK (no physics stepping); returns the full 9-vector."""
        q7, ok, pe, re = self.kin.ik(pos, R, q0)
        q = np.array(q0, float).copy()
        q[:7] = q7
        return q, ok, 0, pe, re

    def limit_margin(self, q):
        return float(np.min(np.minimum(q[:7] - self.limits[:7, 0], self.limits[:7, 1] - q[:7])))

    # ------------------------------------------------------- freeze/restore
    def _snapshot(self, extra=()):
        from pxr import UsdPhysics
        import isaacsim.core.experimental.utils.stage as stage_utils

        stage = stage_utils.get_current_stage()
        snap = {"q": np.asarray(self.arm.get_joint_positions())[0].copy(), "objects": {}}
        for p in list(dict.fromkeys(list(self.task_objects) + list(extra))):
            rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(p))
            snap["objects"][p] = (lab_scene.get_prim_world_pose(p),
                                  bool(rb.GetKinematicEnabledAttr().Get()))
        return snap

    def _freeze(self, snap, held=()):
        """Pin every task object for planning. The held object instead stays
        dynamic with gravity off and is teleported along with the hand each
        step: moving a kinematic body makes PhysX sweep it to the new pose,
        which shoved the finger pads and the arm (4 mm), while a dynamic
        body's pose change is a true teleport."""
        for p, (_, kin) in snap["objects"].items():
            if p in held:
                lab_scene.set_gravity(p, False)
            elif not kin:
                lab_scene.set_kinematic(p, True)

    def _restore(self, snap, q_cmd, held=()):
        """Put the arm and objects back exactly. A frozen (kinematic) object
        that was carried along while planning must NOT be moved back while
        still kinematic: PhysX moves kinematic bodies to a new pose by
        sweeping them there over the next step, which dragged a held tube
        57 mm back down through the closed fingers (fingers shoved open to
        5.4 mm, tube tilted 5 deg). Objects that were dynamic are switched
        back to dynamic first and then placed -- a true teleport."""
        self.arm.set_joint_positions(snap["q"])
        self.arm.set_joint_velocities(np.zeros_like(snap["q"]))
        self.arm.set_joint_position_targets(q_cmd)
        for p, (pose, kin) in snap["objects"].items():
            if p not in held and not kin:
                lab_scene.set_kinematic(p, False)
            lab_scene.set_prim_world_pose(p, pose[:3], pose[3:7])
            lab_scene.zero_velocity(p)
        # A held object's contacts with the pads are re-created from scratch
        # after the teleports; let the pads re-grip it for a few updates with
        # its gravity still off, so it can't sag or get flicked meanwhile.
        # Those fresh pad contacts overlap it slightly, and PhysX pushes a new
        # overlap apart at full speed in the first update: for a tube held
        # off-centre (re-grasped from the tilted rotor bore) that kick spun it
        # up to 17 deg / 4 mm in the grasp (measured). Held kinematic (put
        # back first, while still dynamic: a true teleport) it can't be
        # kicked; the force-limited pads settle against it instead.
        regrip_kin = list(held) if self.held_regrip_kinematic else []
        for p in regrip_kin:
            lab_scene.set_kinematic(p, True)
        # At least RENDER_FLUSH_UPDATES updates even with nothing held: the
        # cameras render 2 updates behind physics, and after a single update
        # the next recorded frame showed the arm at its last rehearsal pose
        # (one teleported frame per planning call in recorded episodes).
        for _ in range(max(4 if held else 1, RENDER_FLUSH_UPDATES)):
            self.app.update()
        for p in regrip_kin:
            lab_scene.set_kinematic(p, False)
        for p in held:
            lab_scene.zero_velocity(p)
            lab_scene.set_gravity(p, True)

    # ---------------------------------------------------------------- plan
    def plan(self, waypoints, held=(), q_cmd=None):
        """Dense, IK-solved, collision-checked joint path through waypoints,
        starting from the arm's current state. held: prim path of an object
        carried in the gripper (moves rigidly with the hand while checking).
        q_cmd: the executor's current commanded joint vector (restored as the
        drive target afterwards)."""
        held = [held] if isinstance(held, str) else list(held or ())
        snap = self._snapshot(extra=held)
        held_fingers = snap["q"][7:].copy() if held and self.hold_fingers_when_held else None
        q = snap["q"].copy()
        q_cmd = snap["q"] if q_cmd is None else q_cmd
        T_hand_held = {x: np.linalg.inv(self.kin.hand(snap["q"]))
                          @ pose_matrix(lab_scene.get_prim_world_pose(x)) for x in held}
        self._freeze(snap, held)
        plan = Plan()
        try:
            tcp0, R0 = lab_scene.get_tcp_pose()
            prev_pos, prev_R = tcp0, R0
            for wp in waypoints:
                n = max(1, math.ceil(np.linalg.norm(wp.pos - prev_pos) / MAX_STEP_POS),
                        math.ceil(np.linalg.norm(ik_solver.rotation_error(wp.R, prev_R)) / MAX_STEP_ROT))
                path = []
                for i in range(1, n + 1):
                    t = i / n
                    # Holding something: rehearse with the fingers where they
                    # really are. Rehearsing at the waypoint's nominal width
                    # (7.4 mm for a tube the pads hold at 6.2 mm) made _restore
                    # snap the pads shut on it in one step, and a tube held
                    # off-centre (re-grasped from the tilted rotor bore) was
                    # shoved 1-11 mm in the grasp, sometimes out of it.
                    q[7:] = held_fingers if held_fingers is not None else wp.fingers
                    q, ok, _, pe, re = self.solve(prev_pos + t * (wp.pos - prev_pos),
                                                  slerp_R(prev_R, wp.R, t), q)
                    if not ok:
                        plan.ok, plan.error = False, (
                            f"IK failed toward '{wp.phase}' ({i}/{n}): {pe*1000:.2f}mm "
                            f"{math.degrees(re):.2f}deg, q={np.round(q[:7], 3)}")
                        return plan
                    self.arm.set_joint_positions(q)
                    self.arm.set_joint_position_targets(q)
                    for x in held:
                        T = self.kin.hand(q) @ T_hand_held[x]
                        lab_scene.set_prim_world_pose(x, T[:3, 3], lab_scene.matrix_to_quat(T[:3, :3]))
                        lab_scene.zero_velocity(x)
                    self.app.update()
                    sim_tcp, _ = lab_scene.get_tcp_pose()
                    drift = float(np.linalg.norm(sim_tcp - self.kin.tcp(q)[0]))
                    if drift > 1e-3:
                        plan.ok, plan.error = False, (
                            f"simulated arm off its planned pose by {drift*1000:.2f} mm toward '{wp.phase}' "
                            f"({i}/{n}) -- something is pushing it")
                        return plan
                    bad = self.collisions(held, wp.allow)
                    if bad:
                        plan.ok, plan.error = False, f"collision toward '{wp.phase}' ({i}/{n}): {bad}"
                        return plan
                    path.append(q.copy())
                plan.segments.append((wp, path, float(np.linalg.norm(wp.pos - prev_pos)),
                                      float(np.linalg.norm(ik_solver.rotation_error(wp.R, prev_R)))))
                prev_pos, prev_R = wp.pos, wp.R
            return plan
        finally:
            self._restore(snap, q_cmd, held)
            if held:
                T_after = (np.linalg.inv(pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)))
                           @ pose_matrix(lab_scene.get_prim_world_pose(held[0])))
                plan.held_shift_m = float(np.linalg.norm(T_after[:3, 3] - T_hand_held[held[0]][:3, 3]))
