"""Scripted expert for the pipette primitives (pick up the pipette, attach
a tip, aspirate, dispense into a well or a tube, eject the tip, put the
pipette back), on top of the tube expert in expert.py.

As with the tubes, every target is computed in object space from the live
physical state: where the pipette (or its attached tip's end) must be,
mapped back to the hand through the in-hand transform measured right now.
"""
from __future__ import annotations

import math

import numpy as np

import lab_scene
import liquids
from expert import (CLOSE_STEPS, FAST, OPEN_STEPS, SETTLE_STEPS, SLOW, Expert, Result)
from planner import Waypoint, frame_from_axis, pose_matrix, slerp_R as planner_slerp

PIPETTE_OPEN = 0.016               # m per finger around the ~19 mm pipette body
PIPETTE_OPEN_CMD = 1.0 - PIPETTE_OPEN / lab_scene.GRIPPER_STROKE
PIPETTE_HELD = 0.0094              # finger value used for planning while holding it
PIPETTE_GRASP_Y = 0.11             # grasp point along the pipette (local y from the nozzle end)
PIPETTE_GRASP_BELOW_AXIS = 0.006   # fingertips this far below the body axis
READY_NOZZLE = (0.22, 1.52)        # nozzle-down "ready" spot beside the tip rack (xy)
NOZZLE_INSERT = 0.0048             # nozzle end below a tip's collar top when pressing on
                                   # (the seal engages at 4 mm; pushing further would just
                                   # drive the attached tip's collar into the rack)
TIP_DEPTH_RIM_MAX = 0.036          # deepest a tip end may go below a 6 mm tube's rim (tip widens)
ASPIRATE_BELOW_SURFACE = 0.003
TIP_AIM_TOL = 0.0003               # tip end within this of its target before moving on
TIP_SETTLE_STEPS = 6               # held pipette's swing after a move dies out in ~0.4 s
ENTER_DEPTH = 0.004               # tip end this far inside a container before the final descent
TUBE_RIM = liquids.TUBE_PROFILE[-1][0]
WELL_TOP = liquids.WELL_TOP_Z


def rotation_between(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.cross(a, b)
    s, c = np.linalg.norm(w), float(a @ b)
    if s < 1e-12:
        return np.eye(3)
    k = w / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


class PipetteExpert(Expert):
    """Tube primitives (Expert) + pipette primitives. Needs env.pipette."""

    def __init__(self, env, planner, seed=0):
        super().__init__(env, planner, seed)
        self.pip = env.pipette
        self.pip_path = env.objects["pipette_0"]["path"]
        self.pipette_home = lab_scene.get_prim_world_pose(self.pip_path)
        self.pz = lab_scene.BENCH_Z + 0.04

    # ---------------------------------------------------------- helpers
    def T_of(self, path):
        return pose_matrix(lab_scene.get_prim_world_pose(path))

    def hand_target(self, obj_path, T_obj_des):
        """TCP position and hand rotation that put obj_path (held; its pose
        in the hand measured now) at T_obj_des."""
        T_ho = np.linalg.inv(self.T_of(lab_scene.HAND_PATH)) @ self.T_of(obj_path)
        T_h = T_obj_des @ np.linalg.inv(T_ho)
        return T_h[:3, 3] + T_h[:3, :3] @ lab_scene.TCP_LOCAL, T_h[:3, :3]

    def tip_path(self):
        return self.env.objects[self.pip.tip]["path"] if self.pip.tip else None

    def held_group(self):
        return [self.pip_path] + ([self.tip_path()] if self.pip.tip else [])

    def pip_cmd(self):
        return (self.pip.plunger_cmd, 0.0)

    def path_margin(self, T_ho, T_obj_from, T_obj_to, q0, samples=6):
        """Smallest joint-limit margin (rad) along a straight carry of a held
        object from T_obj_from to T_obj_to (pure-math IK, nothing moves), and
        the final joints; margin -1 if IK fails anywhere on the way."""
        q, margin = q0, float("inf")
        for t in np.linspace(0, 1, samples):
            Tobj = np.eye(4)
            Tobj[:3, :3] = planner_slerp(T_obj_from[:3, :3], T_obj_to[:3, :3], t)
            Tobj[:3, 3] = T_obj_from[:3, 3] + t * (T_obj_to[:3, 3] - T_obj_from[:3, 3])
            Th = Tobj @ np.linalg.inv(T_ho)
            q, ok, *_ = self.pl.kin.ik(Th[:3, 3] + Th[:3, :3] @ lab_scene.TCP_LOCAL, Th[:3, :3], q)
            if not ok:
                return -1.0, q
            margin = min(margin, self.pl.limit_margin(q))
        return margin, q

    def choose_yaw(self, R_obj_des, pos_obj_des, obj_path=None, T_ho=None, T_obj_from=None, q0=None):
        """The held pipette's rotation about the vertical doesn't matter to
        any task, so choose it: of headings every 20 deg, the one whose carry
        from the current pose keeps the arm farthest from its joint limits.
        (A fixed heading drove the wrist into a limit on a 45 cm traverse; a
        fixed 'hand points away from the arm' rule made the nozzle-down turn
        infeasible.) Returns (rotation, margin)."""
        if T_ho is None:
            T_ho = np.linalg.inv(self.T_of(lab_scene.HAND_PATH)) @ self.T_of(obj_path)
        if T_obj_from is None:
            T_obj_from = self.T_of(obj_path)
        q0 = self.env.q_cmd if q0 is None else q0
        best = (-2.0, R_obj_des)
        for deg in range(-160, 181, 20):
            a = math.radians(deg)
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            T_to = np.eye(4)
            T_to[:3, :3], T_to[:3, 3] = Rz @ R_obj_des, pos_obj_des
            m, _ = self.path_margin(T_ho, T_obj_from, T_to, q0)
            if m > best[0]:
                best = (m, Rz @ R_obj_des)
        return best[1], best[0]

    def tip_pose_at(self, end_point, R=None):
        """Desired pose of the attached tip: exactly vertical (or rotation R),
        end at end_point."""
        T_des = np.eye(4)
        if R is None:
            Tt = self.T_of(self.tip_path())
            R = rotation_between(Tt[:3, 2], [0, 0, 1.0]) @ Tt[:3, :3]
        T_des[:3, :3], T_des[:3, 3] = R, end_point
        return T_des

    def go_tip_to(self, phase_points, allow=()):
        """Move the held pipette so its tip's end visits each point, vertical,
        at the heading that keeps the arm clear of its joint limits."""
        return self.go_point_to(self.tip_path(), 2, [0, 0, 1.0], phase_points, allow)

    def go_point_to(self, obj_path, axis, axis_dir, phase_points, allow=()):
        """Move the held pipette so obj_path's origin (the tip's end, or the
        pipette's nozzle end) visits each point with its local `axis` along
        axis_dir. phase_points: (phase, point, speed[, correct=True])."""
        def aligned(R_now):
            return rotation_between(R_now[:, axis], axis_dir) @ R_now

        p0 = np.asarray(phase_points[0][1], float)
        R_yaw, _ = self.choose_yaw(aligned(self.T_of(obj_path)[:3, :3]), p0, obj_path=obj_path)
        errors, bias = {}, np.zeros(3)
        for k, (phase, p, spd, *opt) in enumerate(phase_points):
            correct = opt[0] if opt else True
            # One move per point, each aimed from where the object is NOW: the
            # pipette can pivot slightly in the grasp during a carry, and at
            # the tip's end (~17 cm below the grasp) a few degrees is several
            # mm -- more than a 6 mm tube bore allows. Settle before re-aiming.
            if k:
                self.pip_hold(3)
                R_yaw = aligned(self.T_of(obj_path)[:3, :3])

            def wp(aim, phase=phase, spd=spd, R_yaw=R_yaw):
                T_des = np.eye(4)
                T_des[:3, :3], T_des[:3, 3] = R_yaw, aim
                tcp, R = self.hand_target(obj_path, T_des)
                return lambda: [Waypoint(phase, tcp, R, PIPETTE_HELD, gripper=1.0, speed=spd, ang_speed=0.6,
                                         allow=allow)]

            # Closed loop on the point: the arm's joint drives sag and the
            # pipette tilts slightly in the grasp under load (~1-2 mm at the
            # tip end), so aim off by the measured error -- starting from the
            # offset the previous point needed -- until it is within
            # TIP_AIM_TOL. The held pipette swings for ~0.4 s after a move;
            # measure only once it has settled.
            p = np.asarray(p, float)
            aim = p + bias
            for attempt in range(3 if correct else 1):
                res = self.plan_and_go(wp(aim, phase if not attempt else f"{phase}_correct"),
                                       held=self.held_group(), pipette=self.pip_cmd())
                if not res.success:
                    return res
                self.pip_hold(TIP_SETTLE_STEPS)
                err = self.T_of(obj_path)[:3, 3] - p
                if np.linalg.norm(err) < TIP_AIM_TOL:
                    break
                aim -= err
            bias = aim - p
            errors[phase] = float(np.linalg.norm(err))
        return Result(True, {"aim_error_m": errors})

    def pip_hold(self, steps, plunger=None, eject=0.0):
        p = self.pip.plunger_cmd if plunger is None else plunger
        for _ in range(steps):
            self.env.step(self.env.action(np.zeros(7), gripper=1.0, pipette=(p, eject)))

    # ------------------------------------------------------- primitives
    def pick_up_pipette(self):
        before = self.snapshot_objects()
        Tp = self.T_of(self.pip_path)
        lx, ly, lz = Tp[:3, 0], Tp[:3, 1], Tp[:3, 2]
        G = Tp[:3, 3] + ly * PIPETTE_GRASP_Y
        tcp_grasp = G - PIPETTE_GRASP_BELOW_AXIS * lz
        # Closing axis across the body. Of its two signs, take the one whose
        # whole pick -- grasp, lift, turn nozzle-down to the ready pose, with
        # the heading rule applied -- keeps the arm farthest from its joint
        # limits, checked with pure-math IK along the motion before touching
        # anything (the pipette's pose in the hand follows from the grasp).
        best = (-2.0, None)
        R_vert = rotation_between(ly, [0, 0, 1.0]) @ Tp[:3, :3]
        ready_pos = np.array([*READY_NOZZLE, self.pz + 0.16])
        up = Tp.copy()
        up[:3, 3] += [0, 0, 0.08]
        for sign in (1.0, -1.0):
            Rg = frame_from_axis(-lz, sign * lx)
            T_hand = np.eye(4)
            T_hand[:3, :3], T_hand[:3, 3] = Rg, tcp_grasp - Rg @ lab_scene.TCP_LOCAL
            T_ho = np.linalg.inv(T_hand) @ Tp
            q, ok, *_ = self.pl.kin.ik(tcp_grasp, Rg, self.env.q_cmd)
            if not ok:
                continue
            m_up, q_up = self.path_margin(T_ho, Tp, up, q)
            _, m_turn = self.choose_yaw(R_vert, ready_pos, T_ho=T_ho, T_obj_from=up, q0=q_up)
            margin = min(self.pl.limit_margin(q), m_up, m_turn)
            print(f"[pick_up_pipette] closing sign {sign:+.0f}: joint-limit margin {margin:.3f} rad", flush=True)
            if margin > best[0]:
                best = (margin, Rg)
        if best[0] <= 0:
            return Result(False, {"error": "no grasp orientation can turn the pipette nozzle-down "
                                           "within joint limits"})
        R = best[1]
        res = self.plan_and_go([
            Waypoint("approach", tcp_grasp + 0.08 * lz, R, PIPETTE_OPEN, speed=FAST),
            Waypoint("descend", tcp_grasp, R, PIPETTE_OPEN, speed=SLOW),
        ])
        if not res.success:
            return res
        self.gripper(1.0, CLOSE_STEPS)
        gap = self.finger_gap()
        if not 0.015 < gap < 0.021:
            return Result(False, {"grasp_gap_m": gap})

        def waypoints():
            Tn = self.T_of(self.pip_path)
            up = Tn.copy()
            up[:3, 3] += [0, 0, 0.08]
            vert = np.eye(4)
            vert[:3, :3], _ = self.choose_yaw(rotation_between(Tn[:3, 1], [0, 0, 1.0]) @ Tn[:3, :3],
                                              ready_pos, obj_path=self.pip_path, T_obj_from=up,
                                              q0=self.pl.kin.ik(*self.hand_target(self.pip_path, up),
                                                                self.env.q_cmd)[0])
            vert[:3, 3] = ready_pos
            t1, R1 = self.hand_target(self.pip_path, up)
            t2, R2 = self.hand_target(self.pip_path, vert)
            return [Waypoint("lift", t1, R1, PIPETTE_HELD, gripper=1.0, speed=SLOW),
                    Waypoint("nozzle_down", t2, R2, PIPETTE_HELD, gripper=1.0, speed=SLOW, ang_speed=0.5)]

        res = self.plan_and_go(waypoints, held=self.pip_path)
        if not res.success:
            return res
        self.gripper(1.0, 5)
        n_end, n_dir = self.pip.nozzle()
        ang = math.degrees(math.acos(np.clip(float(n_dir @ [0, 0, -1.0]), -1, 1)))
        info = {"nozzle_end": n_end.tolist(), "nozzle_vs_down_deg": ang, "grasp_gap_m": gap,
                "others_moved": self.others_unmoved(before, {"pipette_0"})}
        # later primitives re-aim the nozzle from its measured in-hand pose, so
        # this only has to be "turned nozzle-down and held"
        return Result(ang < 10.0 and n_end[2] > self.pz + 0.12 and not info["others_moved"], info)

    def attach_tip(self, tip):
        ev0 = len(self.pip.events)
        Tt = self.T_of(self.env.objects[tip]["path"])
        up = Tt[:3, 2]
        top = Tt[:3, 3] + up * lab_scene.TIP_LENGTH

        # The nozzle end is the pipette's origin, its axis local +y. The press
        # needs 0.6 mm / 3 deg: servo the nozzle onto the tip's axis just
        # above the collar, then press straight down (no correcting once the
        # nozzle is in the collar).
        res = self.go_point_to(self.pip_path, 1, up, [
            ("above_tip", top + 0.03 * up, FAST),
            ("align", top + 0.004 * up, 0.01),
            ("press_on", top - NOZZLE_INSERT * up, 0.01, False),
        ], allow=(self.env.objects[tip]["path"],))
        if not res.success:
            return res
        aim_err = res.info
        self.pip_hold(3)
        if self.pip.tip != tip:
            return Result(False, {"attached": self.pip.tip, "events": self.pip.events[ev0:], **aim_err,
                                  "attach_geometry": {k: v for k, v in self.pip.attach_geometry(tip).items()
                                                      if k != "Tt"}})
        tip_end0 = self.T_of(self.tip_path())[:3, 3]
        res = self.go_tip_to([("lift_out", tip_end0 + [0, 0, 0.075], SLOW)],
                             allow=("/World/Platform/TipRack",))
        if not res.success:
            return res
        self.pip_hold(3)
        end = self.T_of(self.tip_path())[:3, 3]
        n_end, _ = self.pip.nozzle()
        info = {"tip": self.pip.tip, "lifted_m": float(end[2] - tip_end0[2]), **aim_err,
                "nozzle_to_tip_end_m": float(np.linalg.norm(n_end - end)), "events": self.pip.events[ev0:]}
        bad = [e for e in info["events"] if e["event"] != "tip_attached"]
        return Result(info["lifted_m"] > 0.07 and not bad, info)

    def aspirate_from_tube(self, tube, vol_ul):
        ev0 = len(self.pip.events)
        L = lab_scene.LIQUIDS[tube]
        v0, t0 = L.volume_ul, self.pip.tip_volume
        base = self.T_of(self.env.objects[tube]["path"])[:3, 3]
        rim_z = base[2] + TUBE_RIM
        surf_after = base[2] + L.profile.height(max(v0 - vol_ul, 0.0))
        end_z = max(surf_after - ASPIRATE_BELOW_SURFACE, rim_z - TIP_DEPTH_RIM_MAX)
        if end_z >= surf_after:
            return Result(False, {"error": "too little liquid for this volume at a safe tip depth",
                                  "surface_after_z": surf_after, "deepest_end_z": rim_z - TIP_DEPTH_RIM_MAX})
        xy = base[:2]
        tube_path = self.env.objects[tube]["path"]
        res = self.go_tip_to([("above_tube", np.array([*xy, rim_z + 0.03]), FAST),
                              ("enter_tube", np.array([*xy, rim_z - ENTER_DEPTH]), 0.02),
                              ("into_liquid", np.array([*xy, end_z]), 0.01)], allow=(tube_path,))
        if not res.success:
            return res
        reach = res.info
        target = self.pip.plunger + vol_ul / 1000.0
        self.pip_hold(math.ceil(vol_ul / 400.0 * 15) + 4, plunger=target)
        self.pip_hold(5)
        res = self.go_tip_to([("withdraw", np.array([*xy, rim_z + 0.03]), 0.03)], allow=(tube_path,))
        if not res.success:
            return res
        got = self.pip.tip_volume - t0
        info = {"aspirated_ul": got, "requested_ul": vol_ul, "tube_before_ul": v0, **reach,
                "tube_after_ul": L.volume_ul, "events": self.pip.events[ev0:]}
        return Result(abs(got - vol_ul) < 1.0 and not info["events"], info)

    def _dispense_at(self, target, end_point, entry_top_z, container_path):
        ev0 = len(self.pip.events)
        t0 = self.pip.tip_volume
        xy = end_point[:2]
        pts = [("above_target", np.array([*xy, entry_top_z + 0.03]), FAST)]
        if end_point[2] < entry_top_z - ENTER_DEPTH - 0.001:  # deep target: enter first
            pts.append(("enter_target", np.array([*xy, entry_top_z - ENTER_DEPTH]), 0.02))
        res = self.go_tip_to(pts + [("into_target", end_point, 0.01)], allow=(container_path,))
        if not res.success:
            return res
        reach = res.info
        L0 = lab_scene.LIQUIDS.get(target)
        vb = L0.volume_ul if L0 else 0.0
        self.pip_hold(math.ceil(t0 / 400.0 * 15) + 6, plunger=0.0)
        res = self.go_tip_to([("withdraw", np.array([*xy, entry_top_z + 0.03]), 0.03)],
                             allow=(container_path,))
        if not res.success:
            return res
        L = lab_scene.LIQUIDS.get(target)
        info = {"dispensed_ul": t0 - self.pip.tip_volume, "target": target, "target_before_ul": vb, **reach,
                "target_after_ul": L.volume_ul if L else 0.0, "events": self.pip.events[ev0:]}
        ok = L is not None and abs((L.volume_ul - vb) - t0) < 1.0 and self.pip.tip_volume < 0.5
        return Result(bool(ok) and not info["events"], info)

    def dispense_into_well(self, well):
        tray = self.T_of(self.env.objects["tray"]["path"])
        L = lab_scene.LIQUIDS[f"tray/{well}"]
        c = tray[:3, :3] @ L.offset + tray[:3, 3]
        top = tray[2, 3] + WELL_TOP
        return self._dispense_at(f"tray/{well}", np.array([c[0], c[1], top - 0.004]), top,
                                 self.env.objects["tray"]["path"])

    def dispense_into_tube(self, tube):
        Tt = self.T_of(self.env.objects[tube]["path"])
        rim = Tt[2, 3] + TUBE_RIM
        return self._dispense_at(tube, np.array([Tt[0, 3], Tt[1, 3], rim - 0.010]), rim,
                                 self.env.objects[tube]["path"])

    def eject_tip(self):
        ev0 = len(self.pip.events)
        tip = self.pip.tip
        bx, by = lab_scene.WASTE_BIN_XY
        bin_top = self.pz + 0.070
        res = self.go_tip_to([("over_bin", np.array([bx, by, bin_top + 0.02]), FAST)])
        if not res.success:
            return res
        self.pip_hold(2)
        self.pip_hold(1, eject=1.0)
        self.pip_hold(SETTLE_STEPS)
        p = lab_scene.get_prim_world_pose(self.env.objects[tip]["path"])
        inside = abs(p[0] - bx) < 0.043 and abs(p[1] - by) < 0.043 and p[2] < bin_top
        info = {"tip": tip, "tip_pos": p[:3].tolist(), "in_bin": bool(inside), "events": self.pip.events[ev0:]}
        bad = [e for e in info["events"] if e["event"] != "tip_ejected"]
        return Result(bool(inside) and self.pip.tip is None and not bad, info)

    def return_pipette(self):
        home = pose_matrix(self.pipette_home)

        def waypoints():
            above = home.copy()
            above[:3, 3] += [0, 0, 0.08]
            low = home.copy()
            low[:3, 3] += [0, 0, 0.006]
            t1, R1 = self.hand_target(self.pip_path, above)
            t2, R2 = self.hand_target(self.pip_path, low)
            return [Waypoint("over_spot", t1, R1, PIPETTE_HELD, gripper=1.0, speed=SLOW, ang_speed=0.5),
                    Waypoint("lower", t2, R2, PIPETTE_HELD, gripper=1.0, speed=SLOW)]

        res = self.plan_and_go(waypoints, held=self.pip_path)
        if not res.success:
            return res
        self.gripper(PIPETTE_OPEN_CMD, OPEN_STEPS)
        self.gripper(PIPETTE_OPEN_CMD, 2 * SETTLE_STEPS)
        tcp, R = lab_scene.get_tcp_pose()
        # Lift straight up: the open pads straddle the pipette's sides, so a
        # vertical lift clears it, and the hand (tilted ~33 deg by the
        # pipette's resting pose) backing off along its own axis would swing
        # the wrist_side camera into the neighbouring tip rack.
        res = self.plan_and_go([Waypoint("retract", tcp + [0, 0, 0.08], R, PIPETTE_OPEN, speed=SLOW)])
        if not res.success:
            return res
        p = self.T_of(self.pip_path)
        dpos = float(np.linalg.norm(p[:3, 3] - home[:3, 3]))
        dang = math.degrees(math.acos(np.clip(float(p[:3, 1] @ home[:3, 1]), -1, 1)))
        return Result(dpos < 0.015 and dang < 5.0, {"moved_from_home_m": dpos, "axis_vs_home_deg": dang})
