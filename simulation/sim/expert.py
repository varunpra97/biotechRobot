"""Scripted expert for the lab primitives: plans collision-checked paths
(planner.py), executes them at 15 Hz through LabEnv with DROID-scaled
velocity actions, and checks each primitive's physical outcome.

Every primitive works from the live physical state: after a friction grasp
the held object's actual pose in the hand is measured, and targets are
computed in object space (where the tube must go), then mapped back to the
hand -- a tube that ended up slightly off-center or tilted between the pads
is still inserted coaxially.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

import centrifuge_geometry as cg
import ik_solver
import lab_env
import lab_scene
from planner import Waypoint, frame_from_axis, pose_matrix

GRIPPER_OPEN = 0.012          # m per finger: 24 mm opening around a 7 mm tube
OPEN_CMD = 1.0 - GRIPPER_OPEN / lab_scene.GRIPPER_STROKE
CLOSE_STEPS = 10              # 0.67 s for the pads to close and load the grasp
OPEN_STEPS = 8
SETTLE_STEPS = 15             # 1 s for a released object to come to rest

HOVER = 0.10
GRASP_DEPTH = 0.012           # TCP below a tube's rim when grasped from the holder
REMOVE_GRASP_DEPTH = 0.010    # ... from the rotor (fingers must stay >~25 mm up the bore axis)
LIFT_CLEARANCE = 0.03
RELEASE_TIP_S = -0.014        # tube tip along the bore axis at release (seat ~ -15.75 mm)
INSERT_STROKE = 0.040
TRANSIT_RAISE = 0.08
PLACE_TIP_GAP = 0.002         # tube tip above the holder bore floor at release

FAST, SLOW, INSERT = 0.15, 0.04, 0.02   # m/s TCP speeds
Z_DOWN = np.array([0.0, 0.0, -1.0])
# Moving a tube along an 8 mm bore with 0.5 mm clearance, it may touch (slide
# along) the bore wall; an overlap query reports that touch as a hit, so
# those segments allow the bore's owner. Physical execution resolves the
# contact for real, and each primitive's outcome check confirms nothing else
# was disturbed.
HOLDER_PATH = "/World/Platform/TubeHolder"


def minjerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


@dataclass
class Result:
    success: bool
    info: dict = field(default_factory=dict)


class Expert:
    def __init__(self, env, planner, seed=0):
        self.env = env
        self.pl = planner
        self.bores = cg.rotor_bores()
        self.rng = random.Random(seed)
        hx, hy = lab_scene.holder_xy()
        self.holder_xy = (hx, hy)
        self.holder_top = lab_scene.BENCH_Z + 0.04 + lab_scene.HOLDER_THICKNESS_MM * lab_scene.MM_TO_M
        self.holder_floor = lab_scene.BENCH_Z + 0.04 + lab_scene.HOLDER_BORE_FLOOR_MM * lab_scene.MM_TO_M
        # Gripper closing axis for all tube work: horizontal, tangential to
        # the rotor at bore 8 -- tilting into a bore is then a pure rotation
        # about it -- with the sign that keeps joint 7 off its stop
        # (validated in the original pick-place collector).
        y = np.cross(self.bores[8].axis, [0, 0, 1.0])
        self.closing_y = -y / np.linalg.norm(y)

    # ------------------------------------------------------------ execution
    def execute(self, plan, pipette=None):
        for wp, path, length, angle in plan.segments:
            knots = np.array([self.env.q_cmd[:7]] + [q[:7] for q in path])
            joint_travel = np.abs(np.diff(knots, axis=0)).max(axis=1).sum()
            T = max(length / wp.speed, angle / wp.ang_speed, 2.0 / lab_env.CONTROL_HZ)
            K = max(2, math.ceil(T * lab_env.CONTROL_HZ),
                    math.ceil(1.875 * joint_travel / (0.9 * lab_env.MAX_JOINT_DELTA)))
            for k in range(1, K + 1):
                s = minjerk(k / K) * (len(knots) - 1)
                i = min(int(s), len(knots) - 2)
                q_ref = knots[i] + (s - i) * (knots[i + 1] - knots[i])
                v = lab_env.delta_to_droid_velocity(q_ref - self.env.q_cmd[:7])
                self.env.step(self.env.action(v, gripper=wp.gripper_cmd(), pipette=pipette))

    def gripper(self, cmd, steps, pipette=None):
        for _ in range(steps):
            self.env.step(self.env.action(np.zeros(7), gripper=cmd, pipette=pipette))

    def plan_and_go(self, waypoints, held=None, pipette=None, max_replans=3):
        """waypoints: a list, or -- for motions that place a held object --
        a function returning the list from the object's CURRENT pose in the
        hand. Planning briefly teleports the world, and re-gripping can
        shift a held object (typically ~0.5 mm, sometimes ~2-3 mm) in the
        hand; when it moved more than 1 mm -- twice the 0.5 mm bore
        clearance -- the targets are recomputed and re-planned."""
        make = waypoints if callable(waypoints) else (lambda: waypoints)
        for attempt in range(max_replans + 1):
            wps = make()
            plan = self.pl.plan(wps, held=held, q_cmd=self.env.q_cmd)
            if held:
                print(f"    [plan {wps[-1].phase}] held object in-hand shift across planning: "
                      f"{plan.held_shift_m*1000:.3f} mm", flush=True)
            if not plan.ok:
                return Result(False, {"planning_error": plan.error})
            if not (held and callable(waypoints) and plan.held_shift_m > 1e-3):
                break
        self.execute(plan, pipette=pipette)
        return Result(True)

    # -------------------------------------------------------------- helpers
    def pose(self, name):
        return lab_scene.get_prim_world_pose(self.env.objects[name]["path"])

    def tube_len(self, name):
        return self.env.objects[name]["size"][2]

    def finger_gap(self):
        q, _ = self.env.joint_state()
        return float(q[7] + q[8])

    def hand_target_for_object(self, name, tip_des, axis_des, R_hand_nominal):
        """Hand rotation and TCP position that put the held object's tip at
        tip_des with its +z axis along axis_des, given its measured pose in
        the hand. R_hand_nominal fixes the rotation about that axis."""
        T_h = pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))
        T_ho = np.linalg.inv(T_h) @ pose_matrix(self.pose(name))
        a_h = T_ho[:3, 2]                       # object axis in hand frame
        cur = R_hand_nominal @ a_h
        w = np.cross(cur, axis_des)
        ang = math.atan2(np.linalg.norm(w), float(cur @ axis_des))
        if np.linalg.norm(w) > 1e-12:
            k = w / np.linalg.norm(w)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_corr = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
        else:
            R_corr = np.eye(3)
        R_h = R_corr @ R_hand_nominal
        hand_pos = np.asarray(tip_des) - R_h @ T_ho[:3, 3]
        return hand_pos + R_h @ lab_scene.TCP_LOCAL, R_h

    def others_unmoved(self, before, exclude, tol=0.001):
        moved = {}
        for n, p in before.items():
            if n in exclude:
                continue
            d = float(np.linalg.norm(self.pose(n)[:3] - p[:3]))
            if d > tol:
                moved[n] = d
        return moved

    def snapshot_objects(self):
        return {n: self.pose(n) for n in self.env.objects}

    # ----------------------------------------------------------- primitives
    def pick_tube_from_holder(self, tube):
        path = self.env.objects[tube]["path"]
        before = self.snapshot_objects()
        tip = self.pose(tube)[:3]
        top = tip[2] + self.tube_len(tube)
        R = frame_from_axis(Z_DOWN, self.closing_y)
        r = self.env.objects[tube]["size"][0] / 2
        res = self.plan_and_go([
            Waypoint("approach", np.array([tip[0], tip[1], top + HOVER]), R, GRIPPER_OPEN, speed=FAST),
            Waypoint("descend", np.array([tip[0], tip[1], top - GRASP_DEPTH]), R, GRIPPER_OPEN, speed=SLOW),
        ])
        if not res.success:
            return res
        self.gripper(1.0, CLOSE_STEPS)
        gap = self.finger_gap()
        if abs(gap - 2 * r) > 0.001:
            return Result(False, {"grasp_gap_m": gap, "expected_m": 2 * r})
        lift_tip_z = self.holder_top + LIFT_CLEARANCE
        tcp_lift = np.array([tip[0], tip[1], lift_tip_z + self.tube_len(tube) - GRASP_DEPTH])
        res = self.plan_and_go([Waypoint("lift", tcp_lift, R, r + 0.0002, gripper=1.0, speed=SLOW,
                                         allow=(HOLDER_PATH,))], held=path)
        if not res.success:
            return res
        self.gripper(1.0, 3)
        p = self.pose(tube)
        tilt = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(p[3:7])[2, 2], -1, 1)))
        info = {"tube_tip_z": float(p[2]), "lifted_by_m": float(p[2] - tip[2]),
                "tilt_deg": tilt, "grasp_gap_m": gap,
                "others_moved": self.others_unmoved(before, {tube})}
        ok = (p[2] > self.holder_top + LIFT_CLEARANCE - 0.004 and tilt < 5 and not info["others_moved"])
        return Result(ok, info)

    def insert_tube_into_bore(self, tube, bore_idx):
        path = self.env.objects[tube]["path"]
        bore = self.bores[bore_idx]
        r = self.env.objects[tube]["size"][0] / 2
        R_nom = frame_from_axis(-bore.axis, self.closing_y)
        tgt = {}

        def waypoints():
            tcp_rel, R_h = self.hand_target_for_object(tube, bore.at(RELEASE_TIP_S), bore.axis, R_nom)
            tcp_pre, _ = self.hand_target_for_object(tube, bore.at(RELEASE_TIP_S + INSERT_STROKE),
                                                     bore.axis, R_nom)
            tgt.update(tcp_pre=tcp_pre, R_h=R_h)
            return [
                Waypoint("transit", tcp_pre + [0, 0, TRANSIT_RAISE], R_h, r + 0.0002, gripper=1.0,
                         speed=FAST, ang_speed=0.6),
                Waypoint("pre_insert", tcp_pre, R_h, r + 0.0002, gripper=1.0, speed=SLOW),
                Waypoint("insert", tcp_rel, R_h, r + 0.0002, gripper=1.0, speed=INSERT,
                         allow=(self.env.objects["centrifuge_body"]["path"],)),
            ]

        res = self.plan_and_go(waypoints, held=path)
        if not res.success:
            return res
        tcp_pre, R_h = tgt["tcp_pre"], tgt["R_h"]
        before = self.snapshot_objects()
        self.gripper(OPEN_CMD, OPEN_STEPS)
        self.gripper(OPEN_CMD, SETTLE_STEPS)
        seat = self.seat_check(tube, bore)
        if not seat["seated"]:
            return Result(False, {"seat": seat})
        seated = self.pose(tube)
        res = self.plan_and_go([Waypoint("retract", tcp_pre, R_h, GRIPPER_OPEN, speed=SLOW)])
        if not res.success:
            return Result(False, {"seat": seat, **res.info})
        moved = float(np.linalg.norm(self.pose(tube)[:3] - seated[:3]))
        return Result(moved < 5e-5 and not self.others_unmoved(before, {tube}),
                      {"seat": seat, "tube_moved_by_retract_m": moved})

    def seat_check(self, tube, bore):
        p0 = self.pose(tube)
        self.gripper(OPEN_CMD, 2)
        p = self.pose(tube)
        ax = lab_scene.quat_to_matrix(p[3:7])[:, 2]
        contacts = lab_scene.contacts_of(self.env.objects[tube]["path"])
        pen = max([0.0] + [-c["min_separation"] for c in contacts.values()])
        info = {"tip_s_m": bore.s_of(p[:3]), "tip_radial_m": bore.radial_offset(p[:3]),
                "axis_vs_bore_deg": float(np.degrees(np.arccos(np.clip(ax @ bore.axis, -1, 1)))),
                "motion_m": float(np.linalg.norm(p[:3] - p0[:3])),
                "contacts": {k: v["n"] for k, v in contacts.items()}, "max_penetration_m": pen}
        info["seated"] = (-0.0165 <= info["tip_s_m"] <= -0.0150 and info["axis_vs_bore_deg"] <= 4.0
                          and info["motion_m"] < 5e-5 and pen <= 2e-4
                          and set(contacts) <= {self.env.objects["centrifuge_body"]["path"]}
                          and len(contacts) == 1)
        return info

    def remove_tube_from_bore(self, tube, bore_idx):
        path = self.env.objects[tube]["path"]
        bore = self.bores[bore_idx]
        r = self.env.objects[tube]["size"][0] / 2
        before = self.snapshot_objects()
        p = self.pose(tube)
        ax = lab_scene.quat_to_matrix(p[3:7])[:, 2]
        top = p[:3] + self.tube_len(tube) * ax
        R = frame_from_axis(-ax, self.closing_y)
        tcp_grasp = top - REMOVE_GRASP_DEPTH * ax
        tcp_pre = tcp_grasp + 0.04 * ax
        res = self.plan_and_go([
            Waypoint("transit", tcp_pre + [0, 0, TRANSIT_RAISE], R, GRIPPER_OPEN, speed=FAST, ang_speed=0.6),
            Waypoint("pre_grasp", tcp_pre, R, GRIPPER_OPEN, speed=SLOW),
            Waypoint("descend", tcp_grasp, R, GRIPPER_OPEN, speed=INSERT),
        ])
        if not res.success:
            return res
        self.gripper(1.0, CLOSE_STEPS)
        gap = self.finger_gap()
        if abs(gap - 2 * r) > 0.001:
            return Result(False, {"grasp_gap_m": gap, "expected_m": 2 * r})
        # Out along the BORE axis (a straight bore: every point of the tube
        # moves parallel to the walls, so a tube resting against them can't
        # dig in), until the tip is clear of the opening; then up and back
        # to vertical.
        tip_clear_s = bore.opening_s + 0.015
        lift = (tip_clear_s - bore.s_of(p[:3])) * bore.axis
        cent = self.env.objects["centrifuge_body"]["path"]
        R_up = frame_from_axis(Z_DOWN, self.closing_y)
        tcp_out = tcp_grasp + lift + np.array([0, 0, TRANSIT_RAISE])
        res = self.plan_and_go([
            Waypoint("extract", tcp_grasp + lift, R, r + 0.0002, gripper=1.0, speed=INSERT, allow=(cent,)),
            Waypoint("lift_upright", tcp_out, R_up, r + 0.0002, gripper=1.0, speed=SLOW, ang_speed=0.6),
        ], held=path)
        if not res.success:
            return res
        self.gripper(1.0, 3)
        q = self.pose(tube)
        tilt = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(q[3:7])[2, 2], -1, 1)))
        info = {"tip_z": float(q[2]), "tilt_deg": tilt, "grasp_gap_m": gap,
                "others_moved": self.others_unmoved(before, {tube})}
        rim = lab_scene.centrifuge_pos()[2] + lab_scene.CENTRIFUGE_EXTENTS[2]
        ok = q[2] > bore.at(tip_clear_s)[2] and tilt < 8 and not info["others_moved"]
        info["above_rotor_by_m"] = float(q[2] - bore.at(tip_clear_s)[2])
        return Result(bool(ok), info)

    def place_tube_into_holder(self, tube, slot_rc):
        path = self.env.objects[tube]["path"]
        r = self.env.objects[tube]["size"][0] / 2
        sx = self.holder_xy[0] + lab_scene.HOLDER_HOLE_X_MM[slot_rc[1]] * lab_scene.MM_TO_M
        sy = self.holder_xy[1] + lab_scene.HOLDER_HOLE_Y_MM[slot_rc[0]] * lab_scene.MM_TO_M
        R_nom = frame_from_axis(Z_DOWN, self.closing_y)
        up = np.array([0, 0, 1.0])
        tgt = {}

        def waypoints():
            tcp_low, R_h = self.hand_target_for_object(tube, [sx, sy, self.holder_floor + PLACE_TIP_GAP], up, R_nom)
            tcp_above, _ = self.hand_target_for_object(tube, [sx, sy, self.holder_top + LIFT_CLEARANCE], up, R_nom)
            tgt.update(tcp_low=tcp_low, R_h=R_h)
            return [
                Waypoint("above_slot", tcp_above, R_h, r + 0.0002, gripper=1.0, speed=FAST, ang_speed=0.6),
                Waypoint("lower", tcp_low, R_h, r + 0.0002, gripper=1.0, speed=SLOW, allow=(HOLDER_PATH,)),
            ]

        res = self.plan_and_go(waypoints, held=path)
        if not res.success:
            return res
        tcp_low, R_h = tgt["tcp_low"], tgt["R_h"]
        before = self.snapshot_objects()
        self.gripper(OPEN_CMD, OPEN_STEPS)
        self.gripper(OPEN_CMD, 2 * SETTLE_STEPS)
        p = self.pose(tube)
        res = self.plan_and_go([Waypoint("retract", tcp_low + [0, 0, HOVER], R_h, GRIPPER_OPEN, speed=SLOW)])
        if not res.success:
            return res
        q = self.pose(tube)
        tilt = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(q[3:7])[2, 2], -1, 1)))
        info = {"tip_above_floor_m": float(q[2] - self.holder_floor),
                "radial_offset_m": float(np.hypot(q[0] - sx, q[1] - sy)), "tilt_deg": tilt,
                "moved_by_retract_m": float(np.linalg.norm(q[:3] - p[:3])),
                "others_moved": self.others_unmoved(before, {tube})}
        # Inside the bore is guaranteed by the physics; the radial check only
        # catches a wrong slot (61.5 mm pitch). The conical tip lets a standing
        # tube rest ~0.6 mm off-center.
        ok = (abs(info["tip_above_floor_m"]) < 5e-4 and info["radial_offset_m"] < 2e-3
              and tilt < 3.0 and info["moved_by_retract_m"] < 1e-3 and not info["others_moved"])
        return Result(bool(ok), info)

    # ------------------------------------------------------ sample tray
    # The 5 x 2 tray (52 x 24 x 14.4 mm, long side along x) is gripped from
    # above across its 24 mm width: fingers open 20 mm each side (8 mm
    # clearance), pad bottoms TRAY_GRASP_Z above the bench so the pads
    # cover the tray's lower side walls without touching the bench.
    TRAY_OPEN = 0.020
    TRAY_GRASP_Z = 0.005
    TRAY_LIFT = 0.06

    def tray_hand_rotation(self, tray_xy, yaw):
        """Hand pointing down, closing across the tray's width; of the two
        closing signs the one with more joint-limit margin at the tray."""
        across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        best = (-1.0, None)
        for s in (1.0, -1.0):
            R = frame_from_axis(Z_DOWN, s * across)
            tcp = np.array([*tray_xy, lab_scene.BENCH_Z + 0.04 + self.TRAY_GRASP_Z + self.TRAY_LIFT])
            q, ok, *_ = self.pl.kin.ik(tcp, R, self.env.q_cmd)
            m = self.pl.limit_margin(q) if ok else -1.0
            if m > best[0]:
                best = (m, R)
        return best[1]

    def pick_up_tray(self):
        path = self.env.objects["tray"]["path"]
        before = self.snapshot_objects()
        Tt = pose_matrix(self.pose("tray"))
        yaw = math.atan2(Tt[1, 0], Tt[0, 0])
        R = self.tray_hand_rotation(Tt[:2, 3], yaw)
        if R is None:
            return Result(False, {"error": "tray out of reach"})
        z0 = Tt[2, 3]
        g = np.array([Tt[0, 3], Tt[1, 3], z0 + self.TRAY_GRASP_Z])
        res = self.plan_and_go([
            Waypoint("approach", g + [0, 0, HOVER], R, self.TRAY_OPEN, speed=FAST),
            Waypoint("descend", g, R, self.TRAY_OPEN, speed=SLOW),
        ])
        if not res.success:
            return res
        self.gripper(1.0, CLOSE_STEPS)
        gap = self.finger_gap()
        width = self.env.objects["tray"]["size"][1]
        if abs(gap - width) > 0.0015:
            return Result(False, {"grasp_gap_m": gap, "expected_m": width})
        res = self.plan_and_go([Waypoint("lift", g + [0, 0, self.TRAY_LIFT], R, width / 2 + 0.0002,
                                         gripper=1.0, speed=SLOW)], held=path)
        if not res.success:
            return res
        self.gripper(1.0, 3)
        T1 = pose_matrix(self.pose("tray"))
        tilt = math.degrees(math.acos(np.clip(T1[2, 2], -1, 1)))
        info = {"lifted_m": float(T1[2, 3] - z0), "tilt_deg": tilt, "grasp_gap_m": gap,
                "others_moved": self.others_unmoved(before, {"tray"})}
        return Result(info["lifted_m"] > self.TRAY_LIFT - 0.005 and tilt < 3.0 and not info["others_moved"], info)

    def place_tray(self, xy, yaw=0.0):
        """Set the held tray down level at xy (bench), long side at yaw."""
        path = self.env.objects["tray"]["path"]
        z_bench = lab_scene.BENCH_Z + 0.04
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        vols = {n: L.volume_ul for n, L in lab_scene.LIQUIDS.items() if n.startswith("tray/")}
        tgt = {}

        def waypoints():
            T_ho = np.linalg.inv(pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))) \
                @ pose_matrix(self.pose("tray"))
            wps = []
            for phase, dz, spd in (("carry", self.TRAY_LIFT, FAST), ("lower", 0.0015, SLOW)):
                T_des = np.eye(4)
                T_des[:3, :3], T_des[:3, 3] = Rz, [xy[0], xy[1], z_bench + dz]
                Th = T_des @ np.linalg.inv(T_ho)
                tcp = Th[:3, 3] + Th[:3, :3] @ lab_scene.TCP_LOCAL
                tgt[phase] = (tcp, Th[:3, :3])
                wps.append(Waypoint(phase, tcp, Th[:3, :3], self.env.objects["tray"]["size"][1] / 2 + 0.0002,
                                    gripper=1.0, speed=spd, ang_speed=0.6))
            return wps

        res = self.plan_and_go(waypoints, held=path)
        if not res.success:
            return res
        before = self.snapshot_objects()
        self.gripper(1.0 - self.TRAY_OPEN / lab_scene.GRIPPER_STROKE, OPEN_STEPS)
        self.gripper(1.0 - self.TRAY_OPEN / lab_scene.GRIPPER_STROKE, SETTLE_STEPS)
        tcp, R = tgt["lower"]
        res = self.plan_and_go([Waypoint("retract", tcp + [0, 0, HOVER], R, self.TRAY_OPEN, speed=SLOW)])
        if not res.success:
            return res
        T = pose_matrix(self.pose("tray"))
        info = {"xy_error_m": float(np.hypot(T[0, 3] - xy[0], T[1, 3] - xy[1])),
                "z_above_bench_m": float(T[2, 3] - z_bench),
                "yaw_error_deg": math.degrees(abs(math.remainder(math.atan2(T[1, 0], T[0, 0]) - yaw, 2 * math.pi))),
                "tilt_deg": math.degrees(math.acos(np.clip(T[2, 2], -1, 1))),
                "liquid_unchanged": all(abs(lab_scene.LIQUIDS[n].volume_ul - v) < 1e-6 for n, v in vols.items()),
                "others_moved": self.others_unmoved(before, {"tray"})}
        ok = (info["xy_error_m"] < 0.003 and abs(info["z_above_bench_m"]) < 5e-4 and info["yaw_error_deg"] < 3
              and info["tilt_deg"] < 1.0 and info["liquid_unchanged"] and not info["others_moved"])
        return Result(bool(ok), info)
