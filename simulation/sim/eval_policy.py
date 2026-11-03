"""Closed-loop evaluation of a fine-tuned lab policy in the Isaac sim.

The policy runs in openpi's environment (WSL) behind a websocket server
(training/serve_lab_policy.py); this script builds a seeded scene -- the
generator's layout and look randomization, with a seed the training data
never used -- and drives the arm with the policy's actions at 15 Hz:

  observation -> server -> 16-step action chunk -> execute the first
  --replan steps -> observe again ...

The policy sees what it was trained on: the three policy cameras
(exterior, wrist_top, wrist_side), each JPEG-encoded at quality 95 and
resized to 320x240 bicubic like export_lerobot.py, plus the 10-D state and
a prompt. Success is the generator's physical outcome check for the task,
holding for HOLD_STEPS consecutive steps; a trial that hasn't succeeded by
the task's step budget fails. Every trial is recorded as an episode
(lab_env.EpisodeRecorder), so sim/view_episodes.py plays them back.

Tasks (each starts from the freshly built scene, arm at home):
  pick_up_tray, pick_tube_from_holder, pick_up_pipette

Output (<out>, default data/eval/<name>):
  <task>/ep_<trial>/   recorded trial (success flag = outcome)
  results.jsonl        one line per trial; summary.json success rates
  logs/                each trial's Isaac output

Usage (Isaac venv, from the BiotechRobot folder, with the server running):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\sim\\eval_policy.py --name pilot100_2999 --trials 5
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = r"C:\Users\varun\workspace\BiotechRobot\data\eval"
EVAL_SEED = 7                 # layouts and looks the training data (seed 0) never had
IMAGE_WH = (320, 240)         # export_lerobot.IMAGE_HW, as (width, height)
JPEG_QUALITY = 95             # lab_env.EpisodeRecorder's quality
HOLD_STEPS = 15               # outcome must hold this long (1 s) to count
# --finisher (RL teacher hand-over, tube picks)
FINISHER_NEAR_M = 0.10        # hand over only this close to the grasp point (the teacher's training range)...
STALL_STEPS, STALL_M = 15, 0.01   # ... once the VLA's hand has moved < 1 cm over 1 s, or it closes the gripper
FINISHER_STEPS = 120          # the teacher's budget (its training episode length)
FINISHER_LIFT_STEPS = 4       # scripted straight-up steps (1 cm each) after the teacher's 3 cm lift
# tray picks (rl/tray_grasp): trained from up to 60 cm (pi0.5's tray attempts end 17-67 cm away),
# 150 steps, its own goal is the eval's (lifted 5.5 cm, level); one scripted 1 cm step adds margin
TRAY_FINISHER = {"near_m": 0.60, "steps": 150, "lift_steps": 1, "held_gap": (0.020, 0.0265)}
# pipette pick-ups (rl/pipette_grasp): from up to 60 cm, 150 steps; its goal is the grasp and a
# 3 cm lift, then the expert's planned turn takes it nozzle-down (pipette_expert.turn_pipette_ready)
PIPETTE_FINISHER = {"near_m": 0.60, "steps": 150, "lift_steps": 0, "held_gap": (0.015, 0.021)}
TASKS = {                     # step budget ~2x the expert's episode length
    "pick_up_tray": 300,
    "pick_tube_from_holder": 300,
    "pick_up_pipette": 450,
    "place_tube_into_holder": 300,     # M4: the expert picks the trial's tube first; pi0.5 puts it back
    "insert_tube_into_bore": 300,      # M5: the expert picks the trial's tube first; pi0.5 loads it into a bore
    "attach_tip": 300,                 # M6 (setups: see M6_TASKS)
    "aspirate_from_tube": 300,
    "dispense_into_well": 300,
}
# insert (rl/tube_insert, M5): bores reachable on the default layout (reach survey: 1-3); hand over within the
# teacher's training range; the teacher releases and backs off itself
INSERT_FINISHER = {"near_m": 0.30, "steps": 180, "bores": ("1", "2", "3")}   # M5 teacher: holds up to 30 cm out
# M6 pipette operations: setup by the expert (pick up the pipette; + attach a tip; + aspirate), pi0.5 does the task,
# the RL teacher positions (nozzle onto the tip / tip end at the liquid point), the expert's own moves finish
# (lift the tip out; plunger stroke and withdraw). Success: the expert's tests.
M6_TASKS = ("attach_tip", "aspirate_from_tube", "dispense_into_well")
M6_FINISHER = {"attach_tip": {"near_m": 0.10, "steps": 120}, "aspirate_from_tube": {"near_m": 0.12, "steps": 120},
               "dispense_into_well": {"near_m": 0.12, "steps": 120}}
M6_VOL_UL = 200
# place (rl/tube_place, M4): hand over within the teacher's training range (holds up to 15 cm from the slot);
# the teacher seats the tube while holding it, then a script opens and backs off (scripted release)
PLACE_HANDOVER_TILT_DEG = 5.0  # M4 eval: pi0.5 tilted held tubes 22-100 deg before its hand-overs; the teacher knows upright holds
PLACE_FINISHER = {"near_m": 0.15, "steps": 120, "seated_z_m": 0.001, "seated_xy_m": 0.0015, "open_steps": 4,
                  "backoff_steps": 4}


# ------------------------------------------------------------------ client
class PolicyClient:
    """openpi's websocket protocol (openpi_client.websocket_client_policy)
    without its numpy<2 pin: msgpack with numpy arrays packed as dicts."""

    def __init__(self, uri, timeout_s=600):
        import websockets.sync.client
        t0 = time.time()
        while True:
            try:
                self.ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
                break
            except (ConnectionRefusedError, OSError):
                if time.time() - t0 > timeout_s:
                    raise
                time.sleep(3)
        self.metadata = self._unpack(self.ws.recv())

    @staticmethod
    def _pack(obj):
        import msgpack
        import numpy as np

        def default(o):
            if isinstance(o, np.ndarray):
                return {b"__ndarray__": True, b"data": o.tobytes(), b"dtype": o.dtype.str, b"shape": o.shape}
            if isinstance(o, np.generic):
                return {b"__npgeneric__": True, b"data": o.item(), b"dtype": o.dtype.str}
            return o
        return msgpack.packb(obj, default=default)

    @staticmethod
    def _unpack(data):
        import msgpack
        import numpy as np

        def hook(o):
            if b"__ndarray__" in o:
                return np.ndarray(buffer=o[b"data"], dtype=np.dtype(o[b"dtype"]), shape=o[b"shape"])
            if b"__npgeneric__" in o:
                return np.dtype(o[b"dtype"]).type(o[b"data"])
            return o
        return msgpack.unpackb(data, object_hook=hook)

    def infer(self, obs):
        self.ws.send(self._pack(obs))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"policy server error:\n{resp}")
        return self._unpack(resp)


def policy_image(rgb):
    """A rendered camera frame as the policy saw frames in training: JPEG
    at the recorder's quality, decoded, resized like the exporter."""
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(img.resize(IMAGE_WH, resample=Image.BICUBIC))


# ------------------------------------------------------------------ trials
def run_trial(args):
    import collect_primitives as cp

    out = Path(args.out)
    if args.fixed_layout:                 # the pilot's arrangement instead of a sampled one
        import layout
        layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)
    rargs = argparse.Namespace(out=str(out / "scenes"), seed=EVAL_SEED, chain=args.trial, max_episodes=10 ** 9,
                               prior_valid=0, prior_chained=0, force_chained=False, pick_slot=args.pick_slot, block=None)
    (out / "scenes" / "chains").mkdir(parents=True, exist_ok=True)
    r = cp.ChainRunner(rargs)
    r.build()
    run_task(args, r, out)
    r.app.close()


def run_task(args, r, out, setup=True, tube=None, slot_rc=None, rel=None, client=None):
    """One task on a built scene (cp.ChainRunner r), recorded under out/rel; returns the result row. Scenarios
    (run_scenario.py) chain several on one scene: setup=False skips the expert's setup (the scene is already in the
    state the task starts from), tube / slot_rc pick the tube and its slot (default: the layout's pick slot)."""
    import collect_primitives as cp
    import numpy as np

    import lab_env
    import lab_scene
    import prompts
    from expert import LIFT_CLEARANCE

    env, ex, rng = r.env, r.ex, r.rng
    task, max_steps = args.task, TASKS[args.task]
    slot_rc = tuple(slot_rc or r.layout["pick_slot"])
    tube = tube or r.tube
    bore_label = INSERT_FINISHER["bores"][args.trial % len(INSERT_FINISHER["bores"])]
    sleeve = lab_scene.CENTRIFUGE_SLEEVE_PARTS[int(bore_label) - 1]
    if setup and task in ("place_tube_into_holder", "insert_tube_into_bore"):   # setup (not recorded): expert pick
        res = ex.pick_tube_from_holder(tube, ex.closing_for_bore(sleeve)) if task == "insert_tube_into_bore" \
            else ex.pick_tube_from_holder(tube)
        print(f"[setup] expert pick from {cp.slot_label(slot_rc)}: {res.success}", flush=True)
        if not res.success:
            raise SystemExit("setup pick failed")
        env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]
    m6 = {}
    if task in M6_TASKS and not setup:
        raise ValueError("M6 tasks need their expert setup")
    if task in M6_TASKS:                             # setup (not recorded)
        res = ex.pick_up_pipette()
        print(f"[setup] expert pipette pick-up: {res.success}", flush=True)
        if not res.success:
            raise SystemExit("setup pipette pick-up failed")
        tips = sorted(n for n in env.objects if n.startswith("tip_"))
        m6["tip"] = tips[(args.trial * 37) % len(tips)]                  # spread over the rack
        m6["src"] = max((n for n in lab_scene.LIQUIDS if n.startswith("tube_")),
                        key=lambda n: lab_scene.LIQUIDS[n].volume_ul)
        if task != "attach_tip":
            res = ex.attach_tip(m6["tip"])
            print(f"[setup] expert attach {m6['tip']}: {res.success}", flush=True)
            if not res.success:
                raise SystemExit("setup attach failed")
        if task == "dispense_into_well":
            res = ex.aspirate_from_tube(m6["src"], M6_VOL_UL)
            print(f"[setup] expert aspirate {M6_VOL_UL} uL from {m6['src']}: {res.success}", flush=True)
            if not res.success:
                raise SystemExit("setup aspirate failed")
            m6["well"] = "AB"[args.trial % 2] + str((args.trial // 2) % 5 + 1)
        env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]
        m6["ev0"], m6["t0"] = len(env.pipette.events), env.pipette.tip_volume
    kw = {"slot": cp.slot_label(slot_rc)} if task in ("pick_tube_from_holder", "place_tube_into_holder") else {}
    if task == "attach_tip":
        kw = {"tip": m6["tip"][4:]}
    elif task == "aspirate_from_tube":
        parts = m6["src"].split("_")
        kw = {"vol": M6_VOL_UL, "slot": cp.slot_label((int(parts[1][1:]), int(parts[2][1:])))}
    elif task == "dispense_into_well":
        kw = {"well": m6["well"]}
    if task == "insert_tube_into_bore":
        kw = {"bore": bore_label}
    prompt = prompts.make(task, rng, **kw)
    moved = {"pick_up_tray": "tray", "pick_tube_from_holder": tube, "pick_up_pipette": "pipette_0",
             "place_tube_into_holder": tube, "insert_tube_into_bore": tube, "attach_tip": "pipette_0",
             "aspirate_from_tube": "pipette_0", "dispense_into_well": "pipette_0"}[task]
    before = ex.snapshot_objects()
    start = ex.pose(moved)

    def outcome():
        """The expert's success test for the task (expert.py / pipette_expert.py)."""
        p = ex.pose(moved)
        tilt = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(p[3:7])[2, 2], -1, 1)))
        others = ex.others_unmoved(before, {moved})
        if task == "pick_up_tray":
            info = {"lifted_m": float(p[2] - start[2]), "tilt_deg": tilt}
            ok = info["lifted_m"] > ex.TRAY_LIFT - 0.005 and tilt < 3.0
        elif task == "pick_tube_from_holder":
            info = {"tube_tip_z_above_holder_m": float(p[2] - ex.holder_top), "tilt_deg": tilt}
            ok = p[2] > ex.holder_top + LIFT_CLEARANCE - 0.004 and tilt < 5.0
        elif task == "place_tube_into_holder":       # expert.place_tube_into_holder's test, + released
            sx, sy = lab_scene.holder_slot_xy(*slot_rc)
            gap = float(np.sum(np.asarray(env.arm.get_joint_positions())[0][7:9]))
            info = {"tip_above_floor_m": float(p[2] - ex.holder_floor), "radial_offset_m": float(np.hypot(p[0] - sx, p[1] - sy)),
                    "tilt_deg": tilt, "finger_gap_m": gap}
            ok = abs(info["tip_above_floor_m"]) < 5e-4 and info["radial_offset_m"] < 2e-3 and tilt < 3.0 and gap > 0.009
        elif task in M6_TASKS:                       # the expert's tests (pipette_expert), once it has finished
            # pipette events (pipette.py) other than the attach; volume events below 0.1 uL are float noise from the
            # plunger state's round trip through the policy (a 3e-6 uL "aspirated_air")
            ev = [e for e in env.pipette.events[m6["ev0"]:] if e["event"] != "tip_attached"
                  and not ("volume_ul" in e and abs(e["volume_ul"]) < 0.1)]
            info = {"events": ev[:5], "finished": m6.get("finished", False)}
            if task == "attach_tip":
                info["attached"] = env.pipette.tip
                ok = m6.get("finished", False) and env.pipette.tip == m6["tip"] and m6.get("lifted_m", 0.0) > 0.07
            elif task == "aspirate_from_tube":
                info["aspirated_ul"] = env.pipette.tip_volume - m6["t0"]
                ok = m6.get("finished", False) and abs(info["aspirated_ul"] - M6_VOL_UL) < 1.0
            else:
                L = lab_scene.LIQUIDS.get(f"tray/well_{m6['well']}")
                info["well_ul"] = L.volume_ul if L else 0.0
                ok = (m6.get("finished", False) and abs(info["well_ul"] - m6["t0"]) < 1.0
                      and env.pipette.tip_volume < 0.5)
            info["others_moved"] = {}
            return bool(ok and not ev), info
        elif task == "insert_tube_into_bore":        # expert._seat_measure's test (without stepping), + released
            bore = ex.bores[sleeve]
            ax = lab_scene.quat_to_matrix(p[3:7])[:, 2]
            contacts = lab_scene.contacts_of(env.objects[moved]["path"])
            gap = float(np.sum(np.asarray(env.arm.get_joint_positions())[0][7:9]))
            info = {"bore": bore_label, "tip_s_m": bore.s_of(p[:3]), "tip_radial_m": bore.radial_offset(p[:3]),
                    "axis_vs_bore_deg": float(np.degrees(np.arccos(np.clip(ax @ bore.axis, -1, 1)))),
                    "contacts": sorted(contacts), "finger_gap_m": gap}
            ok = (-0.0165 <= info["tip_s_m"] <= -0.0150 and info["axis_vs_bore_deg"] <= 4.0 and gap > 0.009
                  and set(contacts) <= {env.objects["centrifuge_body"]["path"]})
        else:
            n_end, n_dir = env.pipette.nozzle()
            ang = math.degrees(math.acos(np.clip(float(n_dir @ [0, 0, -1.0]), -1, 1)))
            info = {"nozzle_vs_down_deg": ang, "nozzle_end_above_bench_m": float(n_end[2] - ex.pz)}
            ok = ang < 10.0 and n_end[2] > ex.pz + 0.12
        info["others_moved"] = others
        return bool(ok and not others), info

    rel = rel or f"{task}/ep_{args.trial:03d}"
    env.recorder = lab_env.EpisodeRecorder(
        out / rel, prompt, wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA[task],
        meta={"primitive": task, "eval": True, "trial": args.trial, "layout_seed": r.seed, "layout": r.layout,
              "look": r.look, "checkpoint": args.checkpoint_label, "replan": args.replan})
    no_vla = getattr(args, "no_vla", False)          # RL only: the teacher drives from the first step, no pi0.5
    client = client or (None if no_vla else PolicyClient(args.server))
    steps, held, first_ok, infer_s, calls, info = 0, 0, None, [], 0, {}
    # --finisher: the RL teacher (rl/tube_grasp) takes the tube pick over when
    # the VLA stalls near the tube (FINISHER_NEAR_M, hand moving < STALL_M over
    # STALL_STEPS), closes the gripper near it, or runs out of its budget; it
    # gets FINISHER_STEPS more. Its own goal is a 3 cm upright lift; this
    # eval's success needs the tip LIFT_CLEARANCE above the holder (~5.4 cm),
    # so a scripted straight-up move adds the rest once the teacher is there.
    fin, loc, perr = None, None, []
    near_m, fin_steps, lift_left, held_gap = args.finisher_near, FINISHER_STEPS, FINISHER_LIFT_STEPS, (0.004, 0.0085)
    if task in ("pick_up_tray", "pick_up_pipette"):
        near_m, fin_steps, lift_left, held_gap = ((TRAY_FINISHER if task == "pick_up_tray" else PIPETTE_FINISHER)[k]
                                                  for k in ("near_m", "steps", "lift_steps", "held_gap"))
    if task == "place_tube_into_holder":
        near_m, fin_steps, lift_left = PLACE_FINISHER["near_m"], PLACE_FINISHER["steps"], 0
    if task == "insert_tube_into_bore":
        near_m, fin_steps, lift_left = INSERT_FINISHER["near_m"], INSERT_FINISHER["steps"], 0
    if task in M6_TASKS:
        near_m, fin_steps, lift_left = M6_FINISHER[task]["near_m"], M6_FINISHER[task]["steps"], 0
    released = False
    if args.finisher:
        from teacher_finisher import (OPEN_CMD, InsertFinisher, PipetteFinisher, PlaceFinisher, TeacherFinisher,
                                      TipAttachFinisher, TipPositionFinisher, TrayFinisher)
        pose_fn = None
        if args.perceived:
            # the vision pipeline's wrist-camera localizer (its own process, see
            # vision_pipeline/realtime/README.md): the teacher reads the target
            # tube's PERCEIVED pose; the success check still uses the true one
            sys.path.insert(0, str(HERE.parents[1] / "vision_pipeline" / "realtime"))
            from client import LocalizerClient
            loc = LocalizerClient(port=args.localizer_port)
            o0 = env.observe(images=False)
            names = sorted(o0["object_poses"])
            loc.init({"object_names": names, "layout": r.layout},
                     np.array([o0["object_poses"][k] for k in names], float))
            pose_fn = loc.get_prim_world_pose
        if task == "place_tube_into_holder":
            fin = PlaceFinisher(env, r.planner.kin, env.objects[moved]["path"], slot_rc, pose_fn=pose_fn)
        elif task == "insert_tube_into_bore":
            fin = InsertFinisher(env, r.planner.kin, env.objects[moved]["path"], ex.bores[sleeve], pose_fn=pose_fn)
        elif task == "attach_tip":
            fin = TipAttachFinisher(env, r.planner.kin, env.objects[m6["tip"]]["path"])
        elif task in M6_TASKS:
            import pipette_expert as pe
            tip_path = env.objects[env.pipette.tip]["path"]
            if task == "aspirate_from_tube":             # pipette_expert.aspirate_from_tube's tip-end point
                Tt = ex.T_of(env.objects[m6["src"]]["path"])
                L = lab_scene.LIQUIDS[m6["src"]]
                rim_z = Tt[2, 3] + pe.TUBE_RIM
                surf = Tt[2, 3] + L.profile.height(max(L.volume_ul - M6_VOL_UL, 0.0))
                target = np.array([Tt[0, 3], Tt[1, 3],
                                   max(surf - pe.ASPIRATE_BELOW_SURFACE, rim_z - pe.TIP_DEPTH_RIM_MAX)])
                fin = TipPositionFinisher(env, r.planner.kin, tip_path, Tt[:3, 3], Tt[:3, 2], 0, target)
                m6["withdraw"] = (np.array([Tt[0, 3], Tt[1, 3], rim_z + 0.03]), env.objects[m6["src"]]["path"])
            else:                                        # pipette_expert.dispense_into_well's
                tray = ex.T_of(env.objects["tray"]["path"])
                Lw = lab_scene.LIQUIDS[f"tray/well_{m6['well']}"]
                c = tray[:3, :3] @ Lw.offset + tray[:3, 3]
                top = tray[2, 3] + pe.WELL_TOP
                fin = TipPositionFinisher(env, r.planner.kin, tip_path, c, tray[:3, 2], 1,
                                          np.array([c[0], c[1], top - 0.004]))
                m6["withdraw"] = (np.array([c[0], c[1], top + 0.03]), env.objects["tray"]["path"])
        else:
            fin = {"pick_up_tray": TrayFinisher, "pick_up_pipette": PipetteFinisher}.get(task, TeacherFinisher)(
                env, r.planner.kin, env.objects[moved]["path"], pose_fn=pose_fn)
        max_steps += fin_steps + lift_left
    mode, handover, teacher_done, tcp_hist, planned = "vla", None, None, [], None
    if no_vla and fin is not None:
        mode, handover = "teacher", 0
    t0 = time.time()

    def localize(gripper_closed):
        """One localizer update from this step's wrist images (and the error of
        its target estimate against the true pose, for the report)."""
        if loc is None:
            return
        o = env.observe(images=True)
        gap = float(np.sum(o["finger_position_m"]))
        tube = moved if gripper_closed and held_gap[0] < gap < held_gap[1] else None
        loc.step({c: o["images"][c] for c in ("wrist_top", "wrist_side")}, o["hand_pose"], o["finger_position_m"],
                 o["joint_position"], held=tube)
        est = loc.get_prim_world_pose(env.objects[moved]["path"])
        if est is not None:
            true = lab_scene.get_prim_world_pose(env.objects[moved]["path"])
            perr.append((steps, float(np.linalg.norm(np.asarray(est[:3]) - np.asarray(true[:3]))),
                         loc.info(moved).get("source")))

    def after_step():
        nonlocal steps, held, first_ok, info
        steps += 1
        ok, info = outcome()
        held = held + 1 if ok else 0
        if held == 1:
            first_ok = steps
        return held >= HOLD_STEPS or steps >= max_steps

    def should_hand_over(a):
        if fin is None:
            return False
        d = fin.status()[0]
        tcp_hist.append(np.asarray(lab_scene.get_tcp_pose()[0]))
        stalled = len(tcp_hist) > STALL_STEPS and np.linalg.norm(tcp_hist[-1] - tcp_hist[-1 - STALL_STEPS]) < STALL_M
        closing = a[7] > 0.9 and task not in ("place_tube_into_holder", "insert_tube_into_bore") + M6_TASKS   # held closed
        if task == "place_tube_into_holder" and fin.status()[2] > PLACE_HANDOVER_TILT_DEG:
            return True                              # pi0.5 is tilting the held tube: the teacher places it now
        return d < near_m and (stalled or closing) or steps >= TASKS[task]

    while steps < max_steps and mode == "vla":
        o = env.observe(images=True)
        req = {
            "observation/exterior_image_1_left": policy_image(o["images"][lab_env.POLICY_CAMERAS[0]]),
            "observation/wrist_image_left": policy_image(o["images"][lab_env.POLICY_CAMERAS[1]]),
            "observation/wrist_image_right": policy_image(o["images"][lab_env.POLICY_CAMERAS[2]]),
            "observation/joint_position": o["joint_position"].astype(np.float32),
            "observation/gripper_position": o["gripper_position"].astype(np.float32),
            "observation/pipette_state": np.array([o["pipette"]["plunger"], o["pipette"]["tip_attached"]],
                                                  np.float32),
            "prompt": prompt,
        }
        ti = time.time()
        chunk = np.asarray(client.infer(req)["actions"], float)
        infer_s.append(time.time() - ti)
        calls += 1
        for a in chunk[:args.replan]:
            if fin is not None and task in ("place_tube_into_holder", "insert_tube_into_bore") + M6_TASKS and a[7] < 0.9:
                mode, handover = "teacher", steps       # pi0.5 is about to let go: the teacher places instead
                break
            env.step(a)
            localize(a[7] > 0.9)
            if after_step():
                break
            if should_hand_over(a):
                mode, handover = "teacher", steps
                break
        if held >= HOLD_STEPS:
            break
    if mode == "teacher" and held < HOLD_STEPS:
        print(f"[finisher] hand-over at step {steps}: |TCP-grasp| {fin.status()[0] * 100:.1f} cm", flush=True)
        fin.start()
        while steps < max_steps:
            d, lift, tilt, gap = fin.status()
            if (steps - handover) % 10 == 0:
                print(f"[finisher] step {steps}: |TCP-grasp| {d * 100:.1f} cm, lift {lift * 100:.1f} cm, tilt {tilt:.1f} deg, "
                      f"gap {gap * 1000:.1f} mm, closed {fin.closed}, last action {np.round(fin.last, 2).tolist()}", flush=True)
            if task == "pick_up_tray":
                done = lift >= 0.055 and tilt < 3.0 and fin.closed and 0.020 < gap < 0.0265
            elif task == "pick_up_pipette":             # tilt = how far it turned since the hand-over
                done = lift >= 0.03 and tilt < 10.0 and fin.closed and 0.015 < gap < 0.021
            elif task == "attach_tip":
                done = env.pipette.tip == m6["tip"]
            elif task in M6_TASKS:
                done = fin.at_target()
            elif task == "insert_tube_into_bore":        # the teacher released and backed the hand off
                q_now = np.asarray(env.arm.get_joint_positions())[0]
                tcp_now = r.planner.kin.tcp(q_now[:7])[0]
                done = not fin.closed and gap > 0.009 and np.linalg.norm(tcp_now - fin._vs_bore()[2]) > 0.02
            elif task == "place_tube_into_holder":       # seated while held (d: tip to resting point, lift: height)
                done = fin.closed and d < PLACE_FINISHER["seated_xy_m"] + 0.0003 and abs(lift) < PLACE_FINISHER["seated_z_m"]
            else:
                done = lift >= 0.03 and tilt < 5.0 and fin.closed and 0.0045 < gap < 0.0075
            if teacher_done is None and done:
                teacher_done = steps
                print(f"[finisher] teacher finished its lift ({lift * 100:.1f} cm, tilt {tilt:.1f} deg) at step {steps}",
                      flush=True)
            if teacher_done is not None and task == "pick_up_pipette" and planned is None:
                q_now = np.asarray(env.arm.get_joint_positions())[0]
                env.q_cmd[:7] = q_now[:7]                # re-sync (the teacher's command leads the arm)
                res = ex.turn_pipette_ready()
                planned = {"success": bool(res.success), "info": res.info}
                print(f"[finisher] planned turn nozzle-down: {planned}", flush=True)
                if not res.success:
                    break
                hold = env.action(np.zeros(7), gripper=1.0)   # zero joint velocity: stay where the planner ended
                continue
            if task in M6_TASKS and teacher_done is not None and not m6.get("finished"):
                import pipette_expert as pe
                env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]   # re-sync for the planner
                if task == "attach_tip":                 # pipette_expert.attach_tip's lift-out
                    tip_end0 = ex.T_of(ex.tip_path())[:3, 3]
                    ex.pip_hold(3)
                    res = ex.go_tip_to([("lift_out", tip_end0 + [0, 0, 0.075], pe.SLOW)],
                                       allow=("/World/Platform/TipRack",))
                    ex.pip_hold(3)
                    m6["lifted_m"] = float(ex.T_of(ex.tip_path())[2, 3] - tip_end0[2])
                else:                                    # the plunger stroke, then withdraw
                    if task == "aspirate_from_tube":
                        ex.pip_hold(math.ceil(M6_VOL_UL / 400.0 * 15) + 4,
                                    plunger=env.pipette.plunger + M6_VOL_UL / 1000.0)
                    else:
                        ex.pip_hold(math.ceil(m6["t0"] / 400.0 * 15) + 6, plunger=0.0)
                    ex.pip_hold(5)
                    point, allow = m6["withdraw"]
                    res = ex.go_tip_to([("withdraw", point, 0.03)], allow=(allow,))
                m6["finished"] = bool(res.success)
                max_steps = max(max_steps, steps + HOLD_STEPS + 2)     # room for the success hold after the finish
                m6["finish_info"] = {k: v for k, v in (res.info or {}).items() if k != "events"}
                print(f"[finisher] expert finish after the teacher: {res.success}", flush=True)
                continue
            if task in M6_TASKS and teacher_done is not None:
                env.step(env.action(np.zeros(7), gripper=1.0))
            elif task == "insert_tube_into_bore" and teacher_done is not None:
                env.step(env.action(np.zeros(7), gripper=OPEN_CMD))        # its job is done: hold still, open
            elif task == "place_tube_into_holder" and teacher_done is not None:
                if not released:                         # scripted release: open in place, back off straight up
                    for j in range(PLACE_FINISHER["open_steps"] + PLACE_FINISHER["backoff_steps"]):
                        env.step(fin._ik_action(np.array([0.0, 0.0, 0.0 if j < PLACE_FINISHER["open_steps"] else 1.0,
                                                          -1.0])))
                    released = True
                    print(f"[finisher] seated at step {teacher_done}: released, backed off", flush=True)
                env.step(env.action(np.zeros(7), gripper=OPEN_CMD))
            elif planned is not None:
                env.step(hold)
            elif teacher_done is not None and lift_left > 0:
                env.step(fin.lift_action())         # scripted: straight up 1 cm/step, gripper closed
                lift_left -= 1
            elif teacher_done is not None:
                env.step(fin.hold_action())
            else:
                env.step(fin.action())
            localize(fin.closed or teacher_done is not None)
            if after_step():
                break
    success = held >= HOLD_STEPS
    env.recorder.finish(success, {**info, "first_success_step": first_ok})
    env.recorder = None
    if fin is not None:
        info = {**info, "handover_step": handover, "teacher_lift_step": teacher_done, "planned_turn": planned,
                "scripted_release": released}
    if loc is not None:
        e = np.array([x[1] for x in perr if handover is not None and x[0] >= handover]) * 1000
        srcs = [x[2] for x in perr if handover is not None and x[0] >= handover]
        info = {**info, "perceived_tip_err_mm_teacher_phase": (
            {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "max": float(e.max())} if len(e) else None),
            "perceived_sources": {k: srcs.count(k) for k in set(srcs)}}
        loc.close()
    res = {"task": task, "trial": args.trial, "layout_seed": r.seed, "prompt": prompt, "success": success,
           "steps": steps, "max_steps": max_steps, "first_success_step": first_ok, "outcome": info,
           "policy_calls": calls, "mean_infer_s": round(sum(infer_s) / max(1, len(infer_s)), 3),
           "wall_s": round(time.time() - t0, 1), "dir": rel, "checkpoint": args.checkpoint_label}
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, default=float) + "\n")
    print("RESULT", json.dumps(res, default=float), flush=True)
    return res


# ------------------------------------------------------------------ driver
def summarize(out):
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines() if line.strip()] \
        if (out / "results.jsonl").exists() else []
    by = {}
    for x in rows:
        b = by.setdefault(x["task"], {"trials": 0, "success": 0, "steps_to_success": []})
        b["trials"] += 1
        b["success"] += x["success"]
        if x["success"]:
            b["steps_to_success"].append(x["first_success_step"])
    for b in by.values():
        b["success_rate"] = round(b["success"] / b["trials"], 3)
    s = {"tasks": by, "trials": len(rows), "success": sum(x["success"] for x in rows),
         "episodes": [{"dir": x["dir"], "name": x["task"], "prompt": x["prompt"], "success": True, "chained": False,
                       "steps": x["steps"]} for x in rows]}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def drive(args):
    out = Path(args.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "results.jsonl").exists():          # resume: skip finished trials
        done = {(json.loads(x)["task"], json.loads(x)["trial"])
                for x in (out / "results.jsonl").read_text().splitlines() if x.strip()}
    for task in args.tasks:
        for trial in range(args.trials):
            if (task, trial) in done:
                continue
            log = out / "logs" / f"{task}_{trial:03d}.log"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--task", task, "--trial", str(trial),
                   "--out", str(out), "--server", args.server, "--replan", str(args.replan),
                   "--checkpoint-label", args.checkpoint_label] + (["--fixed-layout"] if args.fixed_layout else [])                 + (["--pick-slot", args.pick_slot] if args.pick_slot else []) \
                + (["--finisher"] if args.finisher else []) + (["--perceived"] if args.perceived else []) \
                + ["--localizer-port", str(args.localizer_port), "--finisher-near", str(args.finisher_near)]
            t0 = time.time()
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE)).returncode
            line = next((x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                         if x.startswith("RESULT ")), None)
            res = json.loads(line[7:]) if line else None
            print(f"[{time.strftime('%H:%M')}] {task} trial {trial}: "
                  + (f"{'SUCCESS' if res['success'] else 'fail'} in {res['steps']} steps "
                     f"('{res['prompt']}', {res['mean_infer_s']} s/call)" if res else f"no result (exit {rc}), see {log}")
                  + f" [{(time.time() - t0) / 60:.1f} min]", flush=True)
    s = summarize(out)
    print("success rates:", {t: f"{b['success']}/{b['trials']}" for t, b in s["tasks"].items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="pilot100_2999", help="output folder name under data/eval")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", nargs="+", default=[t for t in TASKS if t not in ("place_tube_into_holder",
                                                                              "insert_tube_into_bore") + M6_TASKS],
                    choices=list(TASKS))
    ap.add_argument("--trials", type=int, default=5, help="per task; trial i uses layout seed (EVAL_SEED, i)")
    ap.add_argument("--server", default="ws://localhost:8000")
    ap.add_argument("--replan", type=int, default=8, help="actions executed from each 16-step chunk")
    ap.add_argument("--checkpoint-label", default="pilot100_v1/2999", help="recorded with the results only")
    ap.add_argument("--perceived", action="store_true",
                    help="with --finisher: the teacher uses the wrist-camera localizer's tube pose "
                         "(start vision_pipeline/realtime/server.py first)")
    ap.add_argument("--localizer-port", type=int, default=5577)
    ap.add_argument("--finisher", action="store_true",
                    help="tube picks: the RL teacher (rl/teacher_policy.pt) finishes when the VLA stalls near the tube")
    ap.add_argument("--finisher-near", type=float, default=FINISHER_NEAR_M,
                    help="tube picks: hand over within this of the grasp point (M1 teacher: trained from 25 cm)")
    ap.add_argument("--pick-slot", default=None, help="the tube to pick stands in this slot (e.g. B1), every trial")
    ap.add_argument("--fixed-layout", action="store_true",
                    help="every trial in the default arrangement (the pilot data's), not a sampled layout")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--task", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out = args.out or os.path.join(DEFAULT_OUT, args.name)
    if args.perceived:
        args.finisher = True
    if args.worker:
        run_trial(args)
    else:
        drive(args)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
