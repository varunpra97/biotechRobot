r"""The model in the control loop (the way general-purpose models are run on RoboDojo): a simulator process that holds
one centrifuge env and waits for commands. Each command is one model call: a target pose for the hand (or a point
fixed to it) and the gripper. The server drives the arm there, then returns camera images and the robot's own state.
The simulator is paused while the model thinks.

What the model gets: two fixed cameras (wide + a crop around a point of its choice), camera calibration, the hand's
pose in the robot's base frame, the finger gap, steps left. No object poses, no depth, no bore coordinates.
(Insert only: a ring around the target bore in ONE wide image at the start of an episode, as the instruction.)

Under the model: a damped least-squares Jacobian servo (controllers.Servo) turns each pose target into the env's joint
actions -- this is IK-like, as in the published RoboDojo harnesses (pose targets -> IK -> joint path).

Env rules untouched: same resets, success / fail tests, 180-step episodes (only motion uses steps).

Commands: JSON files cmd_0000.json ... in --dir; replies out_0000.json + obs_0000.png. See loop_client.py.

Version 2 (less standing still between moves): one model call may carry a list of moves ("op": "moves"), run in
order and stopped early when a move does not arrive, a requirement on the finger gap fails or the episode ends; and
the observation that comes back can be lighter ("obs": "crop" both crops, "one" a single crop, "none" no picture).
Each finished episode is written to episodes.jsonl with its model calls (with motion / look-only), moves, env steps
and the seconds spent thinking and executing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

RL = Path(__file__).resolve().parents[2] / "simulation" / "rl"      # the skill envs live there
sys.path.insert(0, str(RL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["insert_joint", "remove_joint", "tube_joint", "tray_joint", "tray_place_joint",
                                   "pipette_joint"], required=True)
ap.add_argument("--dir", required=True)
ap.add_argument("--seed", type=int, default=21)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--idle_s", type=float, default=3000.0)
ap.add_argument("--frames", type=int, default=1, help="1: save one picture per env step (for the video)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix  # noqa: E402

from controllers import Servo  # noqa: E402
from skills import load  # noqa: E402

D = Path(args.dir)
(D / "frames").mkdir(parents=True, exist_ok=True)
Env, EnvCfg, _ = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
for kv in args.set:
    k, v = kv.split("=")
    setattr(cfg, k, type(getattr(cfg, k))(v))
W, H = 1280, 720
# where the two fixed cameras stand for each task: a point of the work area (world) and eye / look-at offsets from it
SCENES = {
    "centrifuge": (None, {"A": {"eye": [-0.30, -0.28, 0.24], "lookat": [0.0, 0.0, 0.03]},
                          "B": {"eye": [-0.10, -0.02, 0.42], "lookat": [0.0, 0.0, 0.03]}}),
    "tube_joint": ((-0.05, 1.65, 1.00), {"A": {"eye": [-0.42, -0.30, 0.36], "lookat": [0.0, 0.0, 0.0]},
                                         "B": {"eye": [-0.04, -0.14, 0.62], "lookat": [0.0, 0.0, 0.0]}}),
    "tray": ((0.30, 1.73, 0.95), {"A": {"eye": [0.34, -0.30, 0.30], "lookat": [0.0, 0.0, 0.0]},
                                  "B": {"eye": [0.04, -0.12, 0.52], "lookat": [0.0, 0.0, 0.0]}}),
    "pipette_joint": ((0.21, 1.66, 0.96), {"A": {"eye": [0.36, -0.28, 0.30], "lookat": [0.0, 0.0, 0.0]},
                                           "B": {"eye": [0.04, -0.12, 0.55], "lookat": [0.0, 0.0, 0.0]}}),
}
_scene = ("centrifuge" if args.task in ("insert_joint", "remove_joint") else
          "tray" if args.task in ("tray_joint", "tray_place_joint") else args.task)
c0 = np.array(SCENES[_scene][0]) if SCENES[_scene][0] else np.load(RL / "assets" / "centrifuge_bores.npz")["point"][0]
VIEWS = SCENES[_scene][1]
cfg.viewer.origin_type = "world"
cfg.viewer.eye = tuple(float(x) for x in c0 + np.array(VIEWS["A"]["eye"]))
cfg.viewer.lookat = tuple(float(x) for x in c0 + np.array(VIEWS["A"]["lookat"]))
cfg.viewer.resolution = (W, H)
env = Env(cfg, render_mode="rgb_array")
dev = env.device
sv = Servo(env)
insert = args.task == "insert_joint"

import omni.usd  # noqa: E402
from isaacsim.core.rendering_manager import ViewportManager  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

CAM = "/OmniverseKit_Persp"
ended = {"flag": False, "outcome": None}
_gd = env._get_dones


def gd():
    term, trunc = _gd()
    if bool(term[0] | trunc[0]):
        ended["flag"] = True
        ended["outcome"] = "SUCCESS" if bool(env.succeeded[0]) else "FAIL" if bool(env.failed[0]) else "TIMEOUT"
    return term, trunc


env._get_dones = gd


def t3(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)


def base():
    p = env._robot.data.root_pos_w.torch[0]
    R = matrix_from_quat(env._robot.data.root_quat_w.torch[0])
    return p, R


def hand():
    tcp, hq = env._tcp()
    return tcp[0], matrix_from_quat(hq[0])


def cam_calib():
    """The viewport camera's pose (world) and focal length in pixels, from the camera prim itself."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(CAM)
    cam = UsdGeom.Camera(prim)
    M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
    ax = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=1, keepdims=True)       # rows: right, up, back
    f = W * float(cam.GetFocalLengthAttr().Get()) / float(cam.GetHorizontalApertureAttr().Get())
    return {"pos": M[3, :3].copy(), "right": ax[0], "up": ax[1], "back": ax[2], "f": f}


def project(cal, P):
    d = np.asarray(P, dtype=np.float64) - cal["pos"]
    z = -d @ cal["back"]
    return np.array([W / 2 + cal["f"] * (d @ cal["right"]) / z, H / 2 - cal["f"] * (d @ cal["up"]) / z]), z


def aim(name):
    v = VIEWS[name]
    ViewportManager.set_camera_view(CAM, eye=[float(x) for x in c0 + np.array(v["eye"])],
                                    target=[float(x) for x in c0 + np.array(v["lookat"])])


def shoot(name):
    aim(name)
    img = None
    for _ in range(4):
        img = env.render()
    return cv2.cvtColor(np.asarray(img)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR), cam_calib()


def txt(img, x, y, s, col=(255, 255, 255), sc=0.45):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)


def crop_view(img, centre, cw, out_w=640, out_h=480):
    """A crop of native width cw around `centre` (native px), shown at out_w x out_h with native-pixel ticks."""
    ch = cw * out_h / out_w
    x0 = float(np.clip(centre[0] - cw / 2, 0, W - cw))
    y0 = float(np.clip(centre[1] - ch / 2, 0, H - ch))
    sc = out_w / cw
    M = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
    out = cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)
    step = 10 if cw <= 260 else 20 if cw <= 520 else 50
    lab = step * 5 if step != 50 else 100
    yel = (0, 255, 255)
    for u in range(int(x0 // step + 1) * step, int(x0 + cw) + 1, step):
        x = (u - x0) * sc
        big = u % lab == 0
        for yy in (0, out_h):
            cv2.line(out, (int(x), yy - (10 if big else 5)), (int(x), yy + (10 if big else 5)), yel, 1)
        if big:
            txt(out, x + 2, 22, str(u), yel)
            txt(out, x + 2, out_h - 12, str(u), yel)
    for v in range(int(y0 // step + 1) * step, int(y0 + ch) + 1, step):
        y = (v - y0) * sc
        big = v % lab == 0
        for xx in (0, out_w):
            cv2.line(out, (xx - (10 if big else 5), int(y)), (xx + (10 if big else 5), int(y)), yel, 1)
        if big:
            txt(out, 12, y - 3, str(v), yel)
            txt(out, out_w - 40, y - 3, str(v), yel)
    return out


state = {"episode": -1, "steps": 0, "calls": 0, "close": True, "marks": [], "first": True, "frame": 0,
         "focus": None, "crop_w": 400, "overlay": True, "finished": [], "looks": 0, "moves_done": 0,
         "think_s": 0.0, "exec_s": 0.0, "last_reply_t": None, "sticky": [], "mark_labels": True, "detail": {}}
CAL = {}


def to_base(p):
    pb, Rb = base()
    return (Rb.T @ (t3(p) - pb)).cpu().numpy()


def from_base(p):
    pb, Rb = base()
    return pb + Rb @ t3(p)


def observe(n, mode="full", view="A"):
    # Shoot the views asked for, draw what the robot itself knows (TCP, tool z, the marks the model set), write the
    # picture. mode: full = both views, wide + crop; crop = both crops; one = the crop of one view; none = no picture.
    state["detail"] = {}
    if mode == "none":
        return {}
    sticky_w = [from_base(m).cpu().numpy() for m in state["sticky"]]
    tcp, R = hand()
    tcp_w, z_w = tcp.cpu().numpy(), R[:, 2].cpu().numpy()
    marks_w = [from_base(m).cpu().numpy() for m in state["marks"]]
    focus_w = from_base(state["focus"]).cpu().numpy() if state["focus"] is not None else tcp_w
    tiles, px = {}, {}
    names = (view,) if mode == "one" else ("A", "B")
    for name in names:
        img, cal = shoot(name)
        CAL[name] = cal
        uv, _ = project(cal, tcp_w)
        px[name] = [round(float(uv[0]), 1), round(float(uv[1]), 1)]
        if state["overlay"]:
            e, _ = project(cal, tcp_w + 0.05 * z_w)
            cv2.line(img, (int(uv[0]), int(uv[1])), (int(e[0]), int(e[1])), (255, 255, 0), 1, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(img, (int(uv[0] + 5 * dx), int(uv[1] + 5 * dy)), (int(uv[0] + 14 * dx), int(uv[1] + 14 * dy)),
                         (255, 255, 0), 1, cv2.LINE_AA)
        det = {"tcp": px[name], "marks": [], "sticky": []}
        for i, m in enumerate(marks_w):
            q, _ = project(cal, m)
            det["marks"].append([round(float(q[0]), 1), round(float(q[1]), 1)])
            cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 4, (255, 0, 255), 1, cv2.LINE_AA)
            if state["mark_labels"]:
                txt(img, q[0] + 6, q[1] - 6, f"m{i}", (255, 0, 255))
        for i, m in enumerate(sticky_w):                  # points the model keeps on every picture (its own notes)
            q, _ = project(cal, m)
            det["sticky"].append([round(float(q[0]), 1), round(float(q[1]), 1)])
            cv2.circle(img, (int(round(q[0])), int(round(q[1]))), 5, (0, 165, 255), 1, cv2.LINE_AA)
            txt(img, q[0] + 7, q[1] + 4, str(i), (0, 165, 255), 0.4)
        state["detail"][name] = det
        wide = img.copy()
        if state["first"] and name == "A":                 # the instruction, drawn once in one wide view
            org = env.scene.env_origins[0].cpu().numpy()
            if insert:                                     # which bore
                b, _ = project(cal, env.bore_point[env.bore[0]].cpu().numpy() + org)
                cv2.circle(wide, (int(b[0]), int(b[1])), 26, (0, 255, 0), 2, cv2.LINE_AA)
            elif args.task == "tube_joint":                # which tube
                b, _ = project(cal, env.slot_rest[env.target[0]].cpu().numpy() + org + np.array([0.0, 0.0, 0.058]))
                cv2.circle(wide, (int(b[0]), int(b[1])), 22, (0, 255, 0), 2, cv2.LINE_AA)
        if args.task == "tray_place_joint":                # where the tray has to go: its outline, in both views
            tx, ty, tyaw = (float(v) for v in env.target[0])
            org = env.scene.env_origins[0].cpu().numpy()
            c, s = np.cos(tyaw), np.sin(tyaw)
            pts = [project(cal, org + np.array([tx + c * a - s * b, ty + s * a + c * b, 0.94]))[0]
                   for a, b in ((0.026, 0.012), (-0.026, 0.012), (-0.026, -0.012), (0.026, -0.012))]
            for img_ in (img, wide):
                cv2.polylines(img_, [np.array(pts, np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        for u in range(100, W, 100):
            cv2.line(wide, (u, 0), (u, 12), (0, 255, 255), 2)
            txt(wide, u + 3, 30, str(u), (0, 255, 255), 0.8)
        for v in range(100, H, 100):
            cv2.line(wide, (0, v), (12, v), (0, 255, 255), 2)
            txt(wide, 16, v + 8, str(v), (0, 255, 255), 0.8)
        wide = cv2.resize(wide, (640, 360), interpolation=cv2.INTER_AREA)
        txt(wide, 500, 350, f"view {name} wide", (255, 255, 255), 0.6)
        fc, _ = project(cal, focus_w)
        crop = crop_view(img, fc, float(state["crop_w"]))
        txt(crop, 470, 470, f"view {name} crop", (255, 255, 255), 0.6)
        tiles[name] = (wide, crop)
    if mode == "full":
        comp = np.vstack([np.hstack([tiles["A"][0], tiles["B"][0]]), np.hstack([tiles["A"][1], tiles["B"][1]])])
        state["first"] = False
    else:
        comp = np.hstack([tiles[k][1] for k in names])
    cv2.imwrite(str(D / f"obs_{n:04d}.png"), comp)
    pb, Rb = base()
    Rbn, pbn = Rb.cpu().numpy().astype(np.float64), pb.cpu().numpy().astype(np.float64)
    cj = {k: {"pos": (Rbn.T @ (c["pos"] - pbn)).tolist(), "right": (Rbn.T @ c["right"]).tolist(),
              "up": (Rbn.T @ c["up"]).tolist(), "back": (Rbn.T @ c["back"]).tolist(), "f": c["f"], "W": W, "H": H}
          for k, c in CAL.items()}
    (D / "calib.json").write_text(json.dumps(cj))
    if names[-1] != "A":
        aim("A")
        for _ in range(3):
            env.render()
    return px


def private():
    """Simulator truth, for analysis after the episodes only (never returned to the model)."""
    if args.task not in ("insert_joint", "remove_joint"):
        return {}
    if insert:
        tp, grasp, axis, baxis, rel, s, radial, ang = env._tube_state()
        return {"s_mm": float(s[0]) * 1e3, "radial_mm": float(radial[0]) * 1e3, "ang_deg": float(ang[0]),
                "opening_mm": float(env.bore_opening[env.bore[0]]) * 1e3, "bore": int(env.bore[0]),
                "tcp_grasp_mm": float((env._tcp()[0] - grasp).norm()) * 1e3}
    d, held, s, s_clear, radial, ang, tilt, lift, in_bore, success, fail = env._raw_status()
    return {"d_mm": float(d[0]) * 1e3, "held": bool(held[0]), "s_clear_mm": float(s_clear[0]) * 1e3,
            "tilt_deg": float(tilt[0]), "lift_mm": float(lift[0]) * 1e3, "in_bore": bool(in_bore[0]),
            "bore": int(env.bore[0])}


def frame():
    img = env.render()
    if img is None:
        return
    f = cv2.cvtColor(np.asarray(img)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(D / "frames" / f"ep{state['episode']:02d}_{state['frame']:05d}_c{state['calls']:03d}.jpg"), f,
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    state["frame"] += 1


def step(arm, close):
    with torch.inference_mode():
        env.step(sv.pack(arm, torch.tensor([close], device=dev)))
    state["steps"] += 1
    if args.frames:
        frame()


def begin_episode():
    state.update(episode=state["episode"] + 1, steps=0, calls=0, looks=0, moves_done=0, think_s=0.0, exec_s=0.0,
                 first=True, frame=0, marks=[], focus=None, mark_labels=True, crop_w=400)
    ended["flag"] = False
    if insert or args.task == "tray_place_joint":
        state["close"] = True
    elif hasattr(env, "start_closed_env"):
        state["close"] = bool(env.start_closed_env[0])
    else:                                                # tube pick: a held start has the fingers on the tube
        state["close"] = bool(env._robot.data.joint_pos.torch[0][env.finger_ids].sum() < 0.016)
    for _ in range(2):                                   # poses are stale on the first step after a reset
        step(torch.zeros(1, 7, device=dev), state["close"])


def rotvec(v):
    v = t3(v) * (np.pi / 180.0)
    a = v.norm()
    if a < 1e-9:
        return torch.eye(3, device=dev)
    k = v / a
    K = torch.zeros(3, 3, device=dev)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=dev) + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)


def move(c):
    tcp, R = hand()
    pb, Rb = base()
    off = t3(c.get("point_tool", [0.0, 0.0, 0.0]))
    pt = tcp + R @ off
    pt_t = from_base(c["pos"]) if "pos" in c else pt.clone()
    if "dpos" in c:
        pt_t = pt_t + Rb @ t3(c["dpos"])
    if "dtool" in c:
        pt_t = pt_t + R @ t3(c["dtool"])
    R_t = R.clone()
    if "z_axis" in c:
        z = Rb @ t3(c["z_axis"])
        z = z / z.norm()
        yh = Rb @ t3(c["y_axis"]) if "y_axis" in c else R[:, 1]
        y = yh - (yh @ z) * z
        y = y / y.norm()
        R_t = torch.stack([torch.linalg.cross(y, z), y, z], dim=1)
    if "rot_tool" in c:
        R_t = R_t @ rotvec(c["rot_tool"])
    if "rot_base" in c:
        R_t = Rb @ rotvec(c["rot_base"]) @ Rb.T @ R_t
    changed = False
    if "grip" in c:
        new = c["grip"] == "close"
        changed = new != state["close"]
        state["close"] = new
    wait = int(c.get("wait", 6 if changed else 0))
    speed, wmax = float(c.get("speed", 0.006)), float(c.get("wmax", 0.05))
    max_steps = int(c.get("max_steps", 40))
    arrived, perr, rerr, n = False, 0.0, 0.0, 0
    while True:
        tcp, Rn = hand()
        pn = tcp + Rn @ off
        e = pt_t - pn
        rv = axis_angle_from_quat(quat_from_matrix((R_t @ Rn.T).unsqueeze(0)))[0]
        perr, rerr = float(e.norm()) * 1e3, float(rv.norm()) * 180 / np.pi
        arrived = perr < float(c.get("tol_mm", 0.3)) and rerr < float(c.get("tol_deg", 0.3))
        if (arrived and n >= wait) or n >= max(max_steps, wait) or ended["flag"]:
            break
        v, w = 0.5 * e, 0.5 * rv
        v = v / v.norm().clamp(min=1e-9) * v.norm().clamp(max=speed)
        w = w / w.norm().clamp(min=1e-9) * w.norm().clamp(max=wmax)
        step(sv.solve(pn.unsqueeze(0), v.unsqueeze(0), w.unsqueeze(0)), state["close"])
        n += 1
    out = {"arrived": arrived, "pos_err_mm": round(perr, 2), "rot_err_deg": round(rerr, 2), "steps_this_move": n}
    if "require_gap_mm" in c:                             # e.g. [3, 8] after closing: the fingers hold something
        gap = float(env._robot.data.joint_pos.torch[0][env.finger_ids].sum()) * 1e3
        out["gap_mm"] = round(gap, 1)
        out["require_ok"] = bool(c["require_gap_mm"][0] <= gap <= c["require_gap_mm"][1])
    return out


def reply(extra):
    tcp, R = hand()
    pb, Rb = base()
    Rh = (Rb.T @ R).cpu().numpy()
    q = env._robot.data.joint_pos.torch[0]
    return {"episode": state["episode"], "call": state["calls"], "look_only_calls": state["looks"],
            "steps_used": state["steps"],
            "steps_left": int(env.max_episode_length) - 1 - int(env.episode_length_buf[0]),
            "tcp_mm": [round(float(x) * 1e3, 1) for x in to_base(tcp.cpu().numpy())],
            "tool_x": [round(float(x), 3) for x in Rh[:, 0]], "tool_y_closing": [round(float(x), 3) for x in Rh[:, 1]],
            "tool_z_approach": [round(float(x), 3) for x in Rh[:, 2]],
            "finger_gap_mm": round(float(q[env.finger_ids].sum()) * 1e3, 1),
            "gripper_commanded": "closed" if bool(env.grip_closed[0]) else "open",
            "joint_lag_deg": round(float((env.q_cmd[0] - q[env.arm_ids]).abs().max()) * 180 / np.pi, 2),
            "task": task_info(), **extra}


def task_info():
    """The instruction for this episode, as a person would give it (which tube, where to put the tray)."""
    if args.task == "tube_joint":
        r_, c_ = divmod(int(env.target[0]), 5)
        return {"pick_tube_in_slot": "ABCD"[r_] + str(c_ + 1)}
    if args.task == "tray_place_joint":
        tgt = env.target[0].cpu().numpy()
        xy = to_base(np.array([tgt[0], tgt[1], 0.94]) + env.scene.env_origins[0].cpu().numpy())
        pb, Rb = base()
        yaw_b = float(tgt[2]) - float(np.arctan2(Rb[1, 0].item(), Rb[0, 0].item()))
        return {"place_tray_centre_mm": [round(float(xy[0]) * 1e3, 1), round(float(xy[1]) * 1e3, 1)],
                "tray_long_side_yaw_deg": round(float(np.degrees(yaw_b)), 1)}
    return {}


env.reset()
begin_episode()
n, last = 0, time.time()
log = open(D / "private.jsonl", "a")
print("READY", flush=True)
while True:
    f = D / f"cmd_{n:04d}.json"
    if not f.exists():
        if time.time() - last > args.idle_s:
            print("idle, quitting", flush=True)
            break
        time.sleep(0.05)
        continue
    time.sleep(0.02)
    c = json.loads(f.read_text())
    last = t_recv = time.time()
    if state["last_reply_t"] is not None:                   # the arm stood still this long waiting for the model
        state["think_s"] += t_recv - state["last_reply_t"]
    op = c.get("op", "move")
    if op == "quit":
        (D / f"out_{n:04d}.json").write_text(json.dumps({"quit": True}))
        break
    extra, mode = {}, c.get("obs", "full")
    try:
        if op == "views":
            VIEWS.update(c["views"])
        for k in ("marks", "focus", "crop_w", "overlay", "sticky", "mark_labels"):
            if k in c:
                state[k] = c[k]
        if op in ("move", "moves"):                         # one model call: one move, or several run in order
            state["calls"] += 1
            results, why = [], None
            for m in (c["moves"] if op == "moves" else [c]):
                r = move(m)
                results.append(r)
                state["moves_done"] += 1
                if ended["flag"]:
                    why = "episode ended"
                elif not r["arrived"]:
                    why = "move did not arrive"
                elif r.get("require_ok") is False:
                    why = "requirement not met"
                if why:
                    break
            extra = results[0] if op == "move" else {"moves": results, "moves_given": len(c["moves"]), "stopped": why}
        else:
            state["looks"] += 1
        if ended["flag"]:                                   # the env has already reset itself: a new episode
            state["exec_s"] += time.time() - t_recv
            fin = {"episode": state["episode"], "outcome": ended["outcome"], "steps": state["steps"],
                   "calls": state["calls"], "looks": state["looks"], "moves_executed": state["moves_done"],
                   "think_s": round(state["think_s"], 1), "exec_s": round(state["exec_s"], 1)}
            state["finished"].append(fin)
            log.write(json.dumps({"n": n, "finished": fin}) + "\n")
            with open(D / "episodes.jsonl", "a") as fh:
                fh.write(json.dumps({"task": args.task, "seed": args.seed, **fin}) + "\n")
            begin_episode()
            t_recv = time.time()
            extra = {**extra, "episode_over": fin, "note": "the image and state below are the NEXT episode start"}
            mode = "full"                                   # a new episode always starts with the full look
        px = observe(n, mode, c.get("view", "A"))
        out = reply({**extra, "tcp_px": px, "obs": mode, "px": state["detail"], "crop_w": state["crop_w"]})
        log.write(json.dumps({"n": n, "cmd": c, "episode": state["episode"], "steps": state["steps"],
                              "truth": private()}) + "\n")
        log.flush()
    except Exception as ex:                                 # a bad command: say so, move nothing more
        import traceback
        out = {"error": repr(ex), "trace": traceback.format_exc()[-700:]}
    state["exec_s"] += time.time() - t_recv
    tmp = D / f"out_{n:04d}.tmp"
    tmp.write_text(json.dumps(out))
    tmp.replace(D / f"out_{n:04d}.json")
    state["last_reply_t"] = time.time()
    print("cmd", n, op, json.dumps(out)[:300], flush=True)
    n += 1
log.close()
env.close()
app.close()
