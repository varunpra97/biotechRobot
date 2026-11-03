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
# Held tube at pre-insert (tip 10 mm above the bore opening), checked before
# inserting: re-aimed until its tip is within the TOL values of the bore's
# axis (sideways -- how far along the axis it is doesn't matter, the insert
# moves along it) and its axis within the TOL angle; if still beyond the MAX
# values after the re-aims, not inserted. The bore leaves 0.5 mm radial
# clearance; a tube 4.4 deg off jammed. Measured once a swing in the grasp
# has died out (re-aims chasing a swinging tube made it worse: 0.6 -> 3.3 mm).
PRE_INSERT_AXIS_TOL_DEG = 1.0
PRE_INSERT_POS_TOL = 0.0006
PRE_INSERT_AXIS_MAX_DEG = 2.0
PRE_INSERT_POS_MAX = 0.001
PICK_MARGIN = 0.25               # rad from the joint limits for a holder pick's grasp axis (pick_frame)
PRE_INSERT_TRIES = 6             # measurements (so up to 5 re-aims)
PRE_INSERT_SETTLE_STEPS = 10
SEAT_REST_TRIES = 4              # seat_check: re-measure "at rest" this often, 5 steps apart
RETRACT_MOVE_MAX = 0.0005        # insert: a still-seated tube may have settled this much during the retract
# Held tube above a holder slot (tip LIFT_CLEARANCE above the rack), checked
# before lowering into the 8 mm bore (0.5 mm radial clearance; the conical tip,
# 0.5 mm radius at its point, guides an offset tube in -- placed tubes rest up
# to ~0.8 mm off centre): re-aimed until within the TOL values, not lowered if
# still beyond the MAX values. A tube re-grasped from the rotor can swing in
# the grasp after a move, so it is measured only once that has died out
# (measured mid-swing, re-aims chased it: 0.6 -> 2.7 mm).
PLACE_ALIGN_TILT_TOL_DEG = 1.0
PLACE_ALIGN_XY_TOL = 0.0003
PLACE_ALIGN_TILT_MAX_DEG = 2.0
PLACE_ALIGN_XY_MAX = 0.0008
PLACE_ALIGN_TRIES = 4
PLACE_ALIGN_SETTLE_STEPS = 10
PLACE_TIP_GAP = 0.002         # tube tip above the holder bore floor at release

FAST, SLOW, INSERT = 0.15, 0.04, 0.02   # m/s TCP speeds
# Carry to a holder slot. A tube re-grasped from the rotor (remove_tube_from_bore)
# sits tilted across the pads; at FAST it fell out on the way.
PLACE_CARRY_SPEED = SLOW
# Carry to a rotor bore: with the holder and the centrifuge anywhere on the
# bench (layout.py) carries got long, and at FAST a tube arrived 3.9 deg /
# 3.8 mm off in the grasp -- more than the pre-insert re-aims recovered
INSERT_CARRY_SPEED, INSERT_CARRY_ANG_SPEED = 0.08, 0.4
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
        self._bore_closing = {}

    def closing_for_bore(self, bore_idx):
        """Closing axis for work at one rotor bore: horizontal and tangential
        to the rotor at that bore, so tilting the tube into it is a pure
        rotation about the closing axis -- for every bore, not just the one
        a fixed axis was chosen for (with bore 8's axis the hand reached the
        other bores twisted and hit the rotor, or ran out of joint range).
        Of the two signs, the one with more joint-limit margin over the
        insert and the removal grasp (pure-math IK, nothing moves)."""
        if bore_idx not in self._bore_closing:
            b = self.bores[bore_idx]
            y = np.cross(b.axis, [0, 0, 1.0])
            y /= np.linalg.norm(y)
            L = 0.058
            pts = [b.at(RELEASE_TIP_S) + (L - GRASP_DEPTH) * b.axis,
                   b.at(RELEASE_TIP_S + INSERT_STROKE) + (L - GRASP_DEPTH) * b.axis,
                   b.at(-0.0158) + (L - REMOVE_GRASP_DEPTH) * b.axis]
            pts.append(pts[1] + [0, 0, TRANSIT_RAISE])
            best = (-np.inf, -y)
            for cand in (-y, y):
                R = frame_from_axis(-b.axis, cand)
                m = np.inf
                for p in pts:
                    q, ok, *_ = self.pl.kin.ik(p, R, self.env.q_cmd)
                    m = min(m, self.pl.limit_margin(q) if ok else -1.0)
                if m > best[0]:
                    best = (m, cand)
            self._bore_closing[bore_idx] = (best[1], float(best[0]))
        return self._bore_closing[bore_idx][0]

    def bore_margin(self, bore_idx):
        """Joint-limit margin (rad) of that bore's insert and removal poses
        with its closing axis; < 0: some pose has no IK solution."""
        self.closing_for_bore(bore_idx)
        return self._bore_closing[bore_idx][1]

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
        else:
            # Still shifting after every re-plan: the grasp is loose, and the
            # last plan was aimed from a pose the object has since left. A
            # tube that shifted 1.9, 6.3, 2.6 then 7.1 mm across four plans
            # fell out of the fingers on the first move when this ran anyway.
            return Result(False, {"planning_error": f"held object kept shifting in the hand across planning "
                                                    f"({plan.held_shift_m * 1000:.1f} mm on the last of "
                                                    f"{max_replans + 1} tries); not executed",
                                  "held_object_disturbed": True})
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

    def grasp_contacts(self, name):
        """Which colliders touch a just-grasped object, and how often --
        shortened to the last two path elements (e.g. right_finger/Box_3)."""
        c = lab_scene.contacts_of(self.env.objects[name]["path"])
        return {"/".join(k.split("/")[-2:]): v["n"] for k, v in c.items()}

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
    def pick_frame(self, tip, top, tube, closing_y):
        """Hand frame for a holder pick: closing along closing_y if the
        approach, grasp and lift keep PICK_MARGIN from the joint limits, else
        the first of closing_y turned 90, -90 or 180 deg about the vertical
        that does (or the one with most margin). The tube is round, so the
        carry to a bore can turn the hand about it; with the holder anywhere
        on the bench (layout.py) the bore's own axis can put joint 7 on its
        stop at the slot."""
        c = np.asarray(closing_y, float)
        perp = np.cross([0, 0, 1.0], c)
        pts = [np.array([tip[0], tip[1], z]) for z in
               (top + HOVER, top - GRASP_DEPTH, self.holder_top + LIFT_CLEARANCE + self.tube_len(tube) - GRASP_DEPTH)]
        best = None
        for k, cand in enumerate((c, perp, -perp, -c)):
            R = frame_from_axis(Z_DOWN, cand)
            m = np.inf
            for p in pts:
                q, ok, *_ = self.pl.kin.ik(p, R, self.env.q_cmd)
                m = min(m, self.pl.limit_margin(q) if ok else -1.0)
            if m >= PICK_MARGIN:
                if k:
                    print(f"[holder grasp] closing turned {(0, 90, -90, 180)[k]} deg from the requested axis "
                          f"(margin {m:.3f} rad)", flush=True)
                return R
            if best is None or m > best[0]:
                best = (m, R)
        return best[1]

    def pick_tube_from_holder(self, tube, closing_y=None):
        """closing_y: the closing axis to grasp with -- pass the target bore's
        (closing_for_bore) so the insert needs no turn about the tube."""
        path = self.env.objects[tube]["path"]
        before = self.snapshot_objects()
        tip = self.pose(tube)[:3]
        top = tip[2] + self.tube_len(tube)
        R = self.pick_frame(tip, top, tube, self.closing_y if closing_y is None else closing_y)
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

    def tube_vs_bore(self, tube, bore, s_des):
        """Held tube against the bore at pre-insert: axis angle (deg), tip
        offset from its aim point (m, vector) and from the bore axis (m)."""
        p = self.pose(tube)
        ax = lab_scene.quat_to_matrix(p[3:7])[:, 2]
        ang = float(np.degrees(np.arccos(np.clip(ax @ bore.axis, -1, 1))))
        err = p[:3] - bore.at(s_des)
        return ang, err, bore.radial_offset(p[:3])

    def insert_tube_into_bore(self, tube, bore_idx):
        path = self.env.objects[tube]["path"]
        bore = self.bores[bore_idx]
        r = self.env.objects[tube]["size"][0] / 2
        R_nom = frame_from_axis(-bore.axis, self.closing_for_bore(bore_idx))
        tgt = {}
        s_pre = RELEASE_TIP_S + INSERT_STROKE
        bias = np.zeros(3)      # aim offset for the tube tip, from measured pre-insert error

        def approach(transit=True):
            # targets from the tube's pose in the hand measured NOW: a tube
            # that slipped or rotated in the grasp is aimed from where it is
            tcp_pre, R_h = self.hand_target_for_object(tube, bore.at(s_pre) + bias, bore.axis, R_nom)
            tgt.update(tcp_pre=tcp_pre, R_h=R_h)
            pre = Waypoint("pre_insert" if transit else "pre_insert_correct", tcp_pre, R_h, r + 0.0002,
                           gripper=1.0, speed=SLOW)
            if not transit:
                return [pre]
            return [Waypoint("transit", tcp_pre + [0, 0, TRANSIT_RAISE], R_h, r + 0.0002, gripper=1.0,
                             speed=INSERT_CARRY_SPEED, ang_speed=INSERT_CARRY_ANG_SPEED), pre]

        def insert():
            tcp_rel, R_h = self.hand_target_for_object(tube, bore.at(RELEASE_TIP_S) + bias, bore.axis, R_nom)
            tgt.update(R_h=R_h)
            return [Waypoint("insert", tcp_rel, R_h, r + 0.0002, gripper=1.0, speed=INSERT,
                             allow=(self.env.objects["centrifuge_body"]["path"],))]

        # The whole path, insert included, must plan before anything moves: a
        # bore the hand can't enter (e.g. one where the arm hits the rotor on
        # the insert) is then rejected with the world untouched, not after
        # the carry to pre-insert has already been executed.
        full = self.pl.plan(approach() + insert(), held=[path], q_cmd=self.env.q_cmd)
        if not full.ok:
            return Result(False, {"planning_error": full.error})
        res = self.plan_and_go(approach, held=path)
        if not res.success:
            return res
        # Closed loop at pre-insert, before the tube enters the 0.5 mm-clearance
        # bore: the carry from the holder (a fast 40 deg turn) can rotate or
        # slide the tube in the grasp after its in-hand pose was measured for
        # planning. A tube 4.4 deg off the bore axis jammed on the rotor and
        # fell over when released. Measure once the hand has settled; re-aim
        # from the tube's measured pose until it is coaxial, or stop here with
        # the tube still held.
        align = []
        for attempt in range(PRE_INSERT_TRIES):
            self.gripper(1.0, PRE_INSERT_SETTLE_STEPS)
            ang, err, radial = self.tube_vs_bore(tube, bore, s_pre)
            align.append({"axis_vs_bore_deg": ang, "tip_err_m": float(np.linalg.norm(err)), "tip_radial_m": radial})
            if ang <= PRE_INSERT_AXIS_TOL_DEG and radial <= PRE_INSERT_POS_TOL:
                break
            # Within the limits after the first aim: insert. A re-aim moves the
            # held tube again and can leave it worse -- 0.60 mm became 2.6, 3.4,
            # 4.1 mm over three re-aims (v4 chain 2) as it swung in the grasp.
            if attempt == 0 and ang <= PRE_INSERT_AXIS_MAX_DEG and radial <= PRE_INSERT_POS_MAX:
                break
            if attempt == PRE_INSERT_TRIES - 1:
                if ang <= PRE_INSERT_AXIS_MAX_DEG and radial <= PRE_INSERT_POS_MAX:
                    break
                return Result(False, {"error": "tube not coaxial with the bore at pre-insert; not inserted",
                                      "pre_insert_alignment": align})
            bias = bias - err
            res = self.plan_and_go(lambda: approach(transit=False), held=path)
            if not res.success:
                return Result(False, {**res.info, "pre_insert_alignment": align})
        res = self.plan_and_go(insert, held=path)
        if not res.success:
            return Result(False, {**res.info, "pre_insert_alignment": align})
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
        # The retract must not have pulled the tube out: still seated (depth,
        # axis, touching only the rotor) and within RETRACT_MOVE_MAX of where
        # it was. A tube still settling its last fraction of a mm in a tilted
        # bore moved 0.05+ mm over the 2 s retract without the hand touching it.
        after = self._seat_measure(tube, bore)
        still = (-0.0165 <= after["tip_s_m"] <= -0.0150 and after["axis_vs_bore_deg"] <= 4.0
                 and after["max_penetration_m"] <= 2e-4
                 and set(after["contacts"]) <= {self.env.objects["centrifuge_body"]["path"]})
        ok = (moved < 5e-5 or (still and moved < RETRACT_MOVE_MAX)) and not self.others_unmoved(before, {tube})
        return Result(bool(ok), {"seat": seat, "tube_moved_by_retract_m": moved, "seated_after_retract": still,
                                 "pre_insert_alignment": align})

    def seat_check(self, tube, bore):
        # "at rest" is measured over 2 steps; a tube still settling the last
        # fraction of a mm (0.065 mm vs the 0.05 mm limit, otherwise seated)
        # gets a few more steps before being judged
        for attempt in range(SEAT_REST_TRIES):
            info = self._seat_measure(tube, bore)
            if info["motion_m"] < 5e-5:
                break
            self.gripper(OPEN_CMD, 5)
        info["rest_checks"] = attempt + 1
        return info

    def _seat_measure(self, tube, bore):
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
        R = frame_from_axis(-ax, self.closing_for_bore(bore_idx))
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
        contacts = self.grasp_contacts(tube)
        if abs(gap - 2 * r) > 0.001:
            return Result(False, {"grasp_gap_m": gap, "expected_m": 2 * r, "grasp_contacts": contacts})
        # Out along the BORE axis (a straight bore: every point of the tube
        # moves parallel to the walls, so a tube resting against them can't
        # dig in), until the tip is clear of the opening; then up and back
        # to vertical.
        tip_clear_s = bore.opening_s + 0.015
        lift = (tip_clear_s - bore.s_of(p[:3])) * bore.axis
        cent = self.env.objects["centrifuge_body"]["path"]
        R_up = frame_from_axis(Z_DOWN, self.closing_for_bore(bore_idx))
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
        info = {"tip_z": float(q[2]), "tilt_deg": tilt, "grasp_gap_m": gap, "grasp_contacts": contacts,
                "grasp_depth_m": REMOVE_GRASP_DEPTH, "others_moved": self.others_unmoved(before, {tube})}
        rim = lab_scene.centrifuge_pos()[2] + lab_scene.CENTRIFUGE_EXTENTS[2]
        ok = q[2] > bore.at(tip_clear_s)[2] and tilt < 8 and not info["others_moved"]
        info["above_rotor_by_m"] = float(q[2] - bore.at(tip_clear_s)[2])
        return Result(bool(ok), info)

    def place_tube_into_holder(self, tube, slot_rc, closing_y=None):
        """closing_y: the hand's closing axis (default self.closing_y); pass
        the axis the tube is held with, e.g. its bore's after a removal."""
        path = self.env.objects[tube]["path"]
        r = self.env.objects[tube]["size"][0] / 2
        sx, sy = lab_scene.holder_slot_xy(*slot_rc)
        # the held tube's axis, or turned about the tube if that runs out of
        # joint range over this slot (pick_frame; poses as for a pick there)
        R_nom = self.pick_frame(np.array([sx, sy, self.holder_floor]), self.holder_floor + self.tube_len(tube),
                                tube, self.closing_y if closing_y is None else closing_y)
        up = np.array([0, 0, 1.0])
        tgt = {}
        above = np.array([sx, sy, self.holder_top + LIFT_CLEARANCE])
        bias = np.zeros(3)      # aim offset for the tube tip, from the measured error above the slot

        def to_above(carry=True):
            tcp_above, R_h = self.hand_target_for_object(tube, above + bias, up, R_nom)
            return [Waypoint("above_slot" if carry else "above_slot_correct", tcp_above, R_h, r + 0.0002,
                             gripper=1.0, speed=PLACE_CARRY_SPEED if carry else SLOW, ang_speed=0.6)]

        def lower():
            low = np.array([sx, sy, self.holder_floor + PLACE_TIP_GAP])
            tcp_low, R_h = self.hand_target_for_object(tube, low + bias, up, R_nom)
            tgt.update(tcp_low=tcp_low, R_h=R_h)
            return [Waypoint("lower", tcp_low, R_h, r + 0.0002, gripper=1.0, speed=SLOW, allow=(HOLDER_PATH,))]

        # As for the rotor insert: the whole path must plan before anything
        # moves; then stop above the slot, measure the tube, and re-aim until
        # it is over the 8 mm bore (0.5 mm clearance) before lowering. A tube
        # re-grasped from the rotor slipped 0.7 mm in the grasp early in the
        # carry, after its in-hand pose had been measured for planning, and
        # jammed on the bore's edge.
        full = self.pl.plan(to_above() + lower(), held=[path], q_cmd=self.env.q_cmd)
        if not full.ok:
            return Result(False, {"planning_error": full.error})
        res = self.plan_and_go(to_above, held=path)
        if not res.success:
            return res
        align = []
        for attempt in range(PLACE_ALIGN_TRIES):
            self.gripper(1.0, PLACE_ALIGN_SETTLE_STEPS)
            p = self.pose(tube)
            ang = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(p[3:7])[2, 2], -1, 1)))
            err = p[:3] - above
            align.append({"tilt_deg": ang, "tip_err_m": float(np.linalg.norm(err)),
                          "tip_xy_err_m": float(np.linalg.norm(err[:2]))})
            if ang <= PLACE_ALIGN_TILT_TOL_DEG and np.linalg.norm(err[:2]) <= PLACE_ALIGN_XY_TOL:
                break
            # within the limits after the first aim: lower (re-aims moving the
            # held tube took 0.51 mm to 1.1 mm, as at the rotor's pre-insert)
            if attempt == 0 and ang <= PLACE_ALIGN_TILT_MAX_DEG and np.linalg.norm(err[:2]) <= PLACE_ALIGN_XY_MAX:
                break
            if attempt == PLACE_ALIGN_TRIES - 1:
                if ang <= PLACE_ALIGN_TILT_MAX_DEG and np.linalg.norm(err[:2]) <= PLACE_ALIGN_XY_MAX:
                    break
                return Result(False, {"error": "tube not over the slot's bore; not lowered",
                                      "above_slot_alignment": align})
            bias = bias - err
            res = self.plan_and_go(lambda: to_above(carry=False), held=path)
            if not res.success:
                return Result(False, {**res.info, "above_slot_alignment": align})
        res = self.plan_and_go(lower, held=path)
        if not res.success:
            return Result(False, {**res.info, "above_slot_alignment": align})
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
                "others_moved": self.others_unmoved(before, {tube}), "above_slot_alignment": align}
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
    # a carry that would pass over a taller object goes up, across and down:
    # the lifted tray hangs 6 cm up, below the tube tops, and with the objects
    # anywhere (layout.py) a destination across the tube rack swept it
    # through them (clearance_z)
    TRAY_TRANSIT_MIN_XY = 0.05
    TRAY_TRANSIT_NEAR = 0.04

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

    # the tall objects a carried object can hit (prim name prefixes under
    # /World/Platform), and how far above each one's top a crossing passes
    OBSTACLE_PRIMS = {"holder": ("TubeHolder", "Tube_"), "tip_rack": ("TipRack", "Tips"),
                      "waste_bin": ("WasteBin",), "centrifuge": ("Centrifuge",)}
    # (the tops are measured bounds, and the planner rehearses the path for
    # collisions; 5 cm over the centrifuge put a far-out tip beyond IK reach)
    OBSTACLE_CLEAR = {"holder": 0.03, "tip_rack": 0.03, "waste_bin": 0.03, "centrifuge": 0.03}

    def obstacle_tops(self):
        """{object: top z} from the stage's world bounds (once per chain: the
        holder counts its tubes, the rack its tips)."""
        if getattr(self, "_tops", None) is None:
            import omni.usd
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            tops = {}
            for prim in stage.GetPrimAtPath("/World/Platform").GetChildren():
                for k, prefixes in self.OBSTACLE_PRIMS.items():
                    if any(prim.GetName().startswith(p) for p in prefixes):
                        top = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2]
                        tops[k] = max(tops.get(k, -1e9), float(top))
            self._tops = tops
            print(f"[obstacles] tops above the platform: "
                  f"{ {k: round(v - lab_scene.BENCH_Z - 0.04, 3) for k, v in tops.items()} }", flush=True)
        return self._tops

    def crossings(self, a, b, near):
        """The tall objects the straight xy path a -> b comes within `near`
        of: [(t_in, t_out, z)], z = the height a carried object's lowest point
        must cross at (top + OBSTACLE_CLEAR), t along the path; and the
        clearance height of the object a starts in (or None). Objects whose
        footprint (grown by `near`) holds a or b are not crossings: the move
        starts out of or ends over them, and its own approach handles those."""
        import layout
        if layout._CURRENT is None:
            return [], None
        a, b = np.asarray(a[:2], float), np.asarray(b[:2], float)

        def inside(q, p):
            s = [np.cross(q[(i + 1) % 4] - q[i], p - q[i]) for i in range(4)]
            return all(v >= 0 for v in s) or all(v <= 0 for v in s)

        ts = np.linspace(0.0, 1.0, 81)
        out, z_start = [], None
        for name, top in self.obstacle_tops().items():
            q = layout._corners(name, *layout._CURRENT[name], extra=near, hand=False)
            z = top + self.OBSTACLE_CLEAR[name]
            if inside(q, a):
                z_start = max(-1e9 if z_start is None else z_start, z)
                continue
            if inside(q, b):
                continue
            hit = [t for t in ts if inside(q, a + t * (b - a))]
            if hit:
                out.append((max(0.0, hit[0] - 0.0125), min(1.0, hit[-1] + 0.0125), z))
        return sorted(out), z_start

    def clearance_z(self, a, b, near):
        """Height to cross the whole path a -> b at (crossings), or None."""
        c, _ = self.crossings(a, b, near)
        return max(z for *_, z in c) if c else None

    def place_tray(self, xy, yaw=0.0):
        """Set the held tray down level at xy (bench), long side at yaw."""
        path = self.env.objects["tray"]["path"]
        z_bench = lab_scene.BENCH_Z + 0.04
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        vols = {n: L.volume_ul for n, L in lab_scene.LIQUIDS.items() if n.startswith("tray/")}
        tgt = {}

        def waypoints():
            T_now = pose_matrix(self.pose("tray"))
            T_ho = np.linalg.inv(pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))) @ T_now
            here, there = T_now[:3, 3], np.array([xy[0], xy[1], z_bench])
            steps = [("carry", xy, self.TRAY_LIFT, FAST), ("lower", xy, 0.0015, SLOW)]
            z = self.clearance_z(here, there, self.TRAY_TRANSIT_NEAR)
            if (np.hypot(*(there[:2] - here[:2])) > self.TRAY_TRANSIT_MIN_XY and z is not None
                    and z > z_bench + self.TRAY_LIFT):
                steps = [("raise", here[:2], z - z_bench, SLOW), ("cross", xy, z - z_bench, FAST)] + steps
            wps = []
            for phase, at, dz, spd in steps:
                T_des = np.eye(4)
                T_des[:3, :3], T_des[:3, 3] = Rz, [at[0], at[1], z_bench + dz]
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
