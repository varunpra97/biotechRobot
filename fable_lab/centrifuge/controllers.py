"""Hand-written closed-loop controllers for the two centrifuge skills (no PPO, no learned weights), written and
revised by the model from the envs' state and failure traces. They drive the joint-space envs
(tube_remove_joint, tube_insert_joint) through the RL policies' own action interface: 7 joint-target changes
(x cfg.joint_step_rad) + the gripper.

NOT IK-free: every step maps a small Cartesian correction (of the TCP, or of the held tube's tip) to joint-target
changes with the arm's Jacobian (damped least squares) -- differential IK. The run-time IK-free version is the
network distilled from these controllers (distill.py).
"""
from __future__ import annotations

import math

import torch

from isaaclab.utils.math import compute_pose_error, quat_apply, quat_from_matrix, skew_symmetric_matrix


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _cap(v, m):
    """Shorten vectors longer than m (direction kept)."""
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return v / n * n.clamp(max=m)


def _dot(a, b):
    return (a * b).sum(-1)


class Servo:
    """Cartesian twist of a point rigidly attached to the hand -> the env's joint action."""

    def __init__(self, env, damping=0.05):
        self.env, self.dev, self.n = env, env.device, env.num_envs
        self.lam = damping ** 2 * torch.eye(6, device=self.dev)
        self.ex = torch.tensor([1.0, 0.0, 0.0], device=self.dev).expand(self.n, 3)
        self.ey = torch.tensor([0.0, 1.0, 0.0], device=self.dev).expand(self.n, 3)
        self.ez = torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(self.n, 3)

    def hand(self):
        e = self.env
        return e._robot.data.body_pos_w.torch[:, e.hand_idx], e._robot.data.body_quat_w.torch[:, e.hand_idx]

    def jac(self, point):
        e = self.env
        hp, _ = self.hand()
        J = e._robot.data.body_link_jacobian_w.torch[:, e.jac_idx, :, e.jac_cols].clone()
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(point - hp), J[:, 3:, :])
        return J

    def dq(self, point, v, w):
        """Joint change giving the hand-fixed `point` (world) the linear step v and the hand the rotation w."""
        J = self.jac(point)
        err = torch.cat([v, w], dim=1).unsqueeze(-1)
        return (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + self.lam, err)).squeeze(-1)

    def solve(self, point, v, w):
        """The same in action units, within the joint-rate limit (the whole vector scaled, direction kept).
        Tried and dropped: running a joint that hits its limit at the limit and solving the rest of the motion again
        over the other joints. It halved the remove's turn-upright (86 -> 53 steps) but with two joints at their
        limits the 6-D motion no longer fits the remaining five: the hand sank while turning, the tube's tip hit the
        rotor and 362 of 512 removals failed."""
        return self.scaled(self.dq(point, v, w))

    def scaled(self, dq):
        """Joint change in action units; beyond the joint-rate limit the whole vector is scaled (direction kept)."""
        a = dq / self.env.cfg.joint_step_rad
        return a / a.abs().max(dim=-1, keepdim=True).values.clamp(min=1.0)

    def pack(self, a, close):
        out = torch.zeros(self.n, 8, device=self.dev)
        out[:, :7] = a
        out[:, 7] = torch.where(close, 1.0, -1.0)
        return out

    def action(self, dq, close):
        return self.pack(self.scaled(dq), close)


class RemoveController:
    """Onto the tube's axis above its rim (hand z down the axis, closing along the rotor's tangent), down the axis
    to the grasp point, close, out along the bore axis until the tip is clear, then up while turning the tube
    upright."""
    APPROACH, DESCEND, CLOSE, EXTRACT, LIFT = range(5)
    NAMES = ["approach", "descend", "close", "extract", "lift"]

    def __init__(self, env, **kw):
        self.env, self.sv = env, Servo(env)
        n, dev = env.num_envs, env.device
        self.p = dict(pre_m=0.035, v_far=0.008, v_descend=0.004, v_extract=0.005, v_lift=0.005, w_max=0.05, w_lift=0.05,
                      k=0.5, close_steps=6, clear_m=0.004, lift_m=0.035, funnel=1.5)
        self.p.update(kw)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.count = torch.zeros(n, dtype=torch.long, device=dev)
        self.tsign = torch.zeros(n, device=dev)
        self.q_ref = torch.zeros(n, 4, device=dev)
        self.new = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, ids):
        self.new[ids] = True

    def act(self):
        e, sv, p = self.env, self.sv, self.p
        tcp, hq = e._tcp()
        tp, grasp, axis, baxis, s, s_clear, radial, ang, tilt, lift = e._remove_state()
        yh = quat_apply(hq, sv.ey)
        tang = _unit(torch.cross(baxis, sv.ez, dim=-1))
        if self.new.any():                                  # first call of an episode
            m = self.new
            self.tsign[m] = torch.where(_dot(tang, yh) < 0, -1.0, 1.0)[m]
            self.phase[m] = torch.where(e.start_closed_env, self.CLOSE, self.APPROACH)[m]
            self.count[m] = 0
            self.q_ref[m] = hq[m]
            self.new[m] = False
        tang = tang * self.tsign.unsqueeze(-1)
        # approach frame: z down the tube's axis, y the rotor tangent
        z = -axis
        y = _unit(tang - _dot(tang, z).unsqueeze(-1) * z)
        q_app = quat_from_matrix(torch.stack([torch.cross(y, z, dim=-1), y, z], dim=-1))
        _, r_app = compute_pose_error(tcp, hq, tcp, q_app, rot_error_type="axis_angle")
        _, r_ref = compute_pose_error(tcp, hq, tcp, self.q_ref, rot_error_type="axis_angle")
        rel = tcp - grasp
        h = _dot(rel, axis)
        lat_v = rel - h.unsqueeze(-1) * axis
        lat = lat_v.norm(dim=-1)
        d = rel.norm(dim=-1)
        pre = grasp + p["pre_m"] * axis

        ph = self.phase
        nxt = ph.clone()
        nxt[(ph == self.APPROACH) & ((tcp - pre).norm(dim=-1) < 0.004) & (r_app.norm(dim=-1) < 0.05)] = self.DESCEND
        nxt[(ph == self.DESCEND) & ((d < 0.0015) | (h < 0.0008))] = self.CLOSE
        closed_ok = (ph == self.CLOSE) & (self.count >= p["close_steps"])
        in_bore = s < e.bore_opening[e.bore]
        nxt[closed_ok] = self.EXTRACT
        # a bad hold while the tube is still in its bore -- the pads closed off the grasp point (held starts whose
        # tube was knocked 6 mm down the hand as the pads closed), or it is slipping out of them: open, approach and
        # grasp again. (Opening once the tip is out of the bore would drop the tube.)
        nxt[closed_ok & in_bore & (d > 0.004)] = self.APPROACH
        nxt[(ph == self.EXTRACT) & in_bore & (d > 0.013)] = self.APPROACH
        nxt[(ph == self.EXTRACT) & (s_clear > p["clear_m"])] = self.LIFT
        changed = nxt != ph
        self.count = torch.where(changed, torch.zeros_like(self.count), self.count + 1)
        self.q_ref = torch.where((changed & (nxt == self.CLOSE)).unsqueeze(-1), hq, self.q_ref)
        self.phase = ph = nxt

        v = torch.zeros_like(tcp)
        w = torch.zeros_like(tcp)
        # approach: towards the pre-grasp point, higher up the axis the further off it (a funnel onto the line)
        m = ph == self.APPROACH
        h_t = p["pre_m"] + (p["funnel"] * (lat - 0.003)).clamp(0.0, 0.06)
        v[m] = _cap(p["k"] * (grasp + h_t.unsqueeze(-1) * axis - tcp), p["v_far"])[m]
        w[m] = _cap(p["k"] * r_app, p["w_max"])[m]
        m = ph == self.DESCEND
        v[m] = (-p["k"] * lat_v - axis * (p["k"] * h).clamp(0.0, p["v_descend"]).unsqueeze(-1))[m]
        w[m] = _cap(p["k"] * r_app, p["w_max"])[m]
        m = ph == self.EXTRACT                             # out along the bore, the hand's orientation held
        v[m] = (baxis * p["v_extract"])[m]
        w[m] = _cap(p["k"] * r_ref, p["w_max"])[m]
        m = ph == self.LIFT                                # up, the tube's own axis turned to vertical
        up = torch.cross(axis, sv.ez, dim=-1)
        a_up = torch.atan2(up.norm(dim=-1), axis[:, 2]).unsqueeze(-1)
        v[m] = (sv.ez * (p["k"] * (p["lift_m"] - lift)).clamp(0.0, p["v_lift"]).unsqueeze(-1))[m]
        w[m] = _cap(p["k"] * _unit(up) * a_up, p["w_lift"])[m]
        a = sv.solve(tcp, v, w)
        a[ph == self.CLOSE] = 0.0
        return sv.pack(a, ph >= self.CLOSE)


class InsertController:
    """Turn the held tube onto the bore's axis while bringing its tip above the opening (down a funnel around the
    axis), push it down the axis slowing towards the release depth, release it seated, back the hand off along the
    axis. A push that misses the bore, jams or ends cocked is pulled back above the opening and repeated.

    Far starts first move straight in joint space to a taught transit pose of the bore (one recorded joint
    configuration per bore, from the env's library of expert holds: the tube lined up, its tip 6-8 cm straight above
    the pre-insert point): steering the tube there in Cartesian space took the wrist through twice the joint travel
    (150+ of the 180 steps), and a joint-space line straight to the pre-insert pose ran the tube into the centrifuge
    (12 of 20 far starts)."""
    GROSS, CARRY, PUSH, RELEASE, RETREAT = range(5)
    NAMES = ["gross", "carry", "push", "release", "retreat"]

    def __init__(self, env, **kw):
        self.env, self.sv = env, Servo(env)
        n, dev = env.num_envs, env.device
        self.p = dict(pre_m=0.008, v_carry=0.006, v_push=0.0025, w_max=0.03, w_twist=0.02, k=0.4, funnel=2.5,
                      ang_lift=0.0015, enter_radial=0.0006, enter_deg=1.5, enter_twist_deg=8.0,
                      s_goal=-0.0138, release_s=-0.0122, release_stall_s=-0.0105, open_steps=4,
                      retreat_m=0.035, v_retreat=0.004, stall_steps=12, retry_raise=0.012, seat_radial=0.0012,
                      seat_deg=4.0, gross=1.0, gross_radial=0.03, gross_deg=10.0, gross_done_rad=0.03)
        self.p.update(kw)
        self.q_ref, self.has_ref = self._taught_poses()
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.count = torch.zeros(n, dtype=torch.long, device=dev)
        self.s_best = torch.zeros(n, device=dev)
        self.stall = torch.zeros(n, dtype=torch.long, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)
        self.lined = torch.zeros(n, dtype=torch.long, device=dev)
        self.tsign = torch.zeros(n, device=dev)
        self.new = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, ids):
        self.new[ids] = True

    def _taught_poses(self):
        """Per bore, one recorded joint configuration holding a tube lined up with the bore, its tip 6-8 cm straight
        above the pre-insert point (the expert's transit waypoint; the medoid of the library's such holds)."""
        e = self.env
        b = e.lib_bore
        d = e.lib_tube[:, :3] - e.bore_point[b]
        s = _dot(d, e.bore_axis[b])
        radial = (d - s.unsqueeze(-1) * e.bore_axis[b]).norm(dim=-1)
        tax = quat_apply(e.lib_tube[:, 3:7], torch.tensor([0.0, 0.0, 1.0], device=e.device).expand(len(b), 3))
        ang = torch.rad2deg(torch.acos(_dot(tax, e.bore_axis[b]).clamp(-1, 1)))
        pre = e.bore_point[b] + (e.bore_opening[b] + 0.010).unsqueeze(-1) * e.bore_axis[b]
        off = e.lib_tube[:, :3] - pre
        ok = (ang < 4.0) & (off[:, 2] > 0.06) & (off[:, 2] < 0.08) & (off[:, :2].norm(dim=-1) < 0.005)
        q_ref = torch.zeros(len(e.bore_point), 7, device=e.device)
        has = torch.zeros(len(e.bore_point), dtype=torch.bool, device=e.device)
        for k in range(len(e.bore_point)):
            q = e.lib_q[ok & (b == k), :7]
            if len(q):
                dist = (q.unsqueeze(1) - q.unsqueeze(0)).abs().max(-1).values.sum(1)
                q_ref[k], has[k] = q[dist.argmin()], True
        return q_ref, has

    def act(self):
        e, sv, p = self.env, self.sv, self.p
        tcp, hq = e._tcp()
        tp, grasp, axis, baxis, rel, s, radial, ang = e._tube_state()
        point = e.bore_point[e.bore] + e.scene.env_origins
        op = e.bore_opening[e.bore]
        radial_v = (tp - point) - s.unsqueeze(-1) * baxis
        top = grasp - point                                 # the grasp point's offset from the bore's axis
        top_v = top - _dot(top, baxis).unsqueeze(-1) * baxis
        yh = quat_apply(hq, sv.ey)
        tang = _unit(torch.cross(baxis, sv.ez, dim=-1))
        gap = e._robot.data.joint_pos.torch[:, e.finger_ids].sum(-1)
        if self.new.any():
            m = self.new
            self.tsign[m] = torch.where(_dot(tang, yh) < 0, -1.0, 1.0)[m]
            far = ((radial > p["gross_radial"]) | (ang > p["gross_deg"])) & self.has_ref[e.bore] & (p["gross"] > 0)
            self.phase[m] = torch.where(far, self.GROSS, self.CARRY)[m]
            self.count[m] = 0
            self.retries[m] = 0
            self.stall[m] = 0
            self.lined[m] = 0
            self.s_best[m] = s[m]
            self.new[m] = False
        tang = tang * self.tsign.unsqueeze(-1)
        twist = _dot(torch.cross(yh, tang, dim=-1), baxis)   # sin of the closing axis' angle from the rotor tangent

        ph = self.phase
        improved = s < self.s_best - 0.0003
        self.s_best = torch.where(improved, s, self.s_best)
        self.stall = torch.where(improved | (ph != self.PUSH), torch.zeros_like(self.stall), self.stall + 1)
        nxt = ph.clone()
        dq_ref = self.q_ref[e.bore] - e._robot.data.joint_pos.torch[:, e.arm_ids]
        arrived = (ph == self.GROSS) & (dq_ref.abs().max(dim=-1).values < p["gross_done_rad"])
        nxt[arrived] = self.CARRY
        # the tangent's sign again, from the taught pose's closing axis (three starts with the hand 90 deg off had
        # picked the other sign at the start; the twist then turned joint 7 into its limit, half-way round)
        self.tsign[arrived] = torch.where(_dot(tang, yh) < 0, -1.0, 1.0)[arrived] * self.tsign[arrived]
        twist = torch.where(arrived, torch.zeros_like(twist), twist)
        lined = ((radial < p["enter_radial"]) & (ang < p["enter_deg"]) & (s < op + p["pre_m"] + 0.006)
                 & (twist.abs() < math.sin(math.radians(p["enter_twist_deg"]))))
        self.lined = torch.where(lined, self.lined + 1, torch.zeros_like(self.lined))
        inside = (s < op) & (radial < 0.002)               # already in the bore (a tip below the opening's plane but
        #                                                    off the axis is beside the rotor, not in it)
        nxt[(ph == self.CARRY) & (self.count >= 3) & ((self.lined >= 2) | inside)] = self.PUSH
        seated_ok = (radial < p["seat_radial"]) & (ang < p["seat_deg"])
        deep = (s < p["release_s"]) | ((s < p["release_stall_s"]) & (self.stall >= 5))
        # a push that slid off the opening, stopped part-way, or sits cocked at depth: back above the opening, again
        missed = ((s > op - 0.003) & (radial > 0.0015)) | ((s <= op - 0.003) & (radial > 0.0025))
        jam = (ph == self.PUSH) & (missed | ((self.stall >= p["stall_steps"]) & ~(deep & seated_ok)))
        nxt[jam] = self.CARRY
        self.retries[jam] += 1
        nxt[(ph == self.PUSH) & deep & seated_ok] = self.RELEASE
        nxt[(ph == self.RELEASE) & (self.count >= p["open_steps"]) & (gap > 0.0095)] = self.RETREAT
        changed = nxt != ph
        self.count = torch.where(changed, torch.zeros_like(self.count), self.count + 1)
        self.s_best = torch.where(changed, s, self.s_best)
        self.phase = ph = nxt

        v = torch.zeros_like(tcp)
        w = torch.zeros_like(tcp)
        # the rotation taking the tube's axis onto the bore's; separately, a twist about it bringing the closing axis
        # to the rotor's tangent (the fingers then open clear of the rotor). The twist is nearly all joint 7: it gets
        # its own share of the joint-rate limit instead of slowing the carry (starts with the hand 90 deg off)
        aim = _unit(torch.cross(axis, baxis, dim=-1)) * torch.deg2rad(ang).unsqueeze(-1)
        w_align = _cap(p["k"] * aim, p["w_max"])
        w_twist = baxis * (0.5 * twist).clamp(-p["w_twist"], p["w_twist"]).unsqueeze(-1)
        m = ph == self.CARRY
        retry = (self.retries > 0) & (self.count < 8)       # after a miss or a jam: back up above the opening first
        s_t = (op + p["pre_m"] + (p["funnel"] * (radial - 0.0005)).clamp(0.0, 0.06)
               + (p["ang_lift"] * (ang - 1.0)).clamp(0.0, 0.03) + retry.float() * p["retry_raise"])
        go = _cap(p["k"] * (point + s_t.unsqueeze(-1) * baxis - tp), p["v_carry"])
        beside = (s < op + 0.005) & (radial > 0.003)        # low and off the axis: straight up the axis direction first
        go = torch.where(beside.unsqueeze(-1), baxis * 0.004, go)
        v[m] = go[m]
        w[m] = w_align[m]
        m = ph == self.PUSH
        above = (s > op + 0.002).unsqueeze(-1)
        down = (0.35 * (s - p["s_goal"])).clamp(0.0004, p["v_push"]).unsqueeze(-1)
        # above the opening: the tip held on the axis; inside: the walls guide the tip, the hand keeps the tube's top
        # on the axis (a cocked tube wedges)
        side = torch.where(above, _cap(-0.5 * radial_v, 0.001), _cap(-0.3 * top_v, 0.0006))
        v[m] = (-baxis * down + side)[m]
        w[m] = torch.where(above, w_align, torch.zeros_like(w))[m]
        a = sv.solve(tp, v, w)                              # the tube's tip is the controlled point
        m = ph == self.GROSS                                # a straight line in joint space, no Jacobian
        a[m] = sv.scaled(dq_ref)[m]
        tw = (ph == self.CARRY) | ((ph == self.PUSH) & above.squeeze(-1))
        if tw.any():
            a_t = sv.scaled(sv.dq(tp, torch.zeros_like(tp), w_twist))
            a[tw] = (a + a_t).clamp(-1.0, 1.0)[tw]
        m = ph == self.RETREAT
        if m.any():
            vt = baxis * (0.5 * (p["retreat_m"] - (tcp - grasp).norm(dim=-1))).clamp(0.0, p["v_retreat"]).unsqueeze(-1)
            a[m] = sv.solve(tcp, vt, torch.zeros_like(tcp))[m]
        a[ph == self.RELEASE] = 0.0
        return sv.pack(a, ph <= self.PUSH)


CONTROLLERS = {"remove_joint": RemoveController, "insert_joint": InsertController}
