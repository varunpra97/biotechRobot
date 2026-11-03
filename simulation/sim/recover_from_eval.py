"""Recovery episodes from closed-loop eval trials: the policy's attempt, then
the scripted expert finishing the task from wherever the policy left the arm.

For each recorded eval trial (eval_policy.py output), in a fresh Isaac
process:
  1. rebuild the trial's scene from its recorded layout (pilot-era trials
     recorded only the pipette/tray jitter: the default arrangement plus it),
  2. replay the policy's recorded actions open loop -- physics is
     deterministic, so the arm and objects end up where the trial left them,
     then re-sync the joint command to the measured joints (a blocked arm's
     command can run far past it), so the expert continues from the actual
     state with no jump,
  3. hand over to the expert: open the gripper, back off upward, then the
     task's primitive (pick_tube_from_holder / pick_up_tray /
     pick_up_pipette) plans from the current state; if it cannot plan from
     there (the policy left the arm among the tubes, or a joint at its
     limit), the expert first takes the arm home, then plans again,
and record 1-3 as ONE episode with the trial's prompt. meta records where
the hand-over happened ("prefix_steps"), so training can use the whole
sequence or only the expert's part.

Output (<out>, default data/recovery_v1): <task>/ep_<run>_<trial>/ episodes,
_failed/ for trials the expert could not finish, summary.json (the
generator's format: export_lerobot.py and view_episodes.py read it),
logs/ per trial.

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\sim\\recover_from_eval.py \\
        --evals pilot100_2999 v2v4_3999 v2v4_3999_fixed ... [--out data\\recovery_v1]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_ROOT = Path(r"C:\Users\varun\workspace\BiotechRobot\data\eval")
DEFAULT_OUT = r"C:\Users\varun\workspace\BiotechRobot\data\recovery_v1"
BACK_OFF_M = (0.15, 0.08, 0.04, 0.0)  # tried in order before the primitive
TIMEOUT_S = 1800


def scene_layout(meta):
    """The trial's ChainRunner layout, with a "platform" entry."""
    import layout
    lay = dict(meta["layout"])
    if "platform" not in lay:                       # pilot-era trial: default arrangement + jitter
        plat = layout.default_layout()
        for k in ("pipette", "tray"):
            j = lay.get(k) or {}
            x, y, yaw = plat[k]
            plat[k] = [x + j.get("x", 0.0), y + j.get("y", 0.0), yaw + j.get("yaw", 0.0)]
        lay["platform"] = plat
    return lay


def go_home(r):
    """The arm to lab_scene.Q_HOME: a planned move to the home hand pose, then
    settling the null space onto Q_HOME; if no plan exists from here (e.g. a
    joint at its limit), a direct joint-space move after the back-off."""
    import numpy as np
    from planner import Plan, Waypoint
    q_home = np.asarray(r.ls.Q_HOME[:7], float)
    tcp_h, R_h = r.planner.kin.tcp(q_home)
    wp = Waypoint("to_home", np.asarray(tcp_h), R_h, r.ls.GRIPPER_STROKE, speed=0.15)
    if not r.ex.plan_and_go([wp]).success:
        print("[recover] no planned path home; joint-space move", flush=True)
    r.ex.execute(Plan(segments=[(wp, [q_home], 0.0, 0.0)]))
    r.ex.gripper(0.0, 5)


# ------------------------------------------------------------------ worker
def run_trial(args):
    import collect_primitives as cp

    src = Path(args.src)
    meta = json.loads((src / "meta.json").read_text())
    task, prompt = meta["primitive"], meta["prompt"]
    out = Path(args.out)
    rargs = argparse.Namespace(out=str(out / "scenes"), seed=7, chain=int(meta.get("trial", 0)), max_episodes=10 ** 9,
                               prior_valid=0, prior_chained=0, force_chained=False, pick_slot=None, block=None)
    (out / "scenes" / "chains").mkdir(parents=True, exist_ok=True)
    r = cp.ChainRunner(rargs)
    r.layout = scene_layout(meta)              # the trial's scene, not a fresh sample
    r.build()

    import numpy as np

    import lab_env
    from expert import Result
    from planner import Waypoint

    env, ex = r.env, r.ex
    actions = np.load(src / "steps.npz")["action"]
    rel = f"{task}/ep_{args.tag}"
    env.recorder = lab_env.EpisodeRecorder(
        out / rel, prompt, wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA[task],
        meta={"primitive": task, "recovery": True, "source": str(src), "prefix_steps": int(len(actions)),
              "layout": r.layout, "look": r.look, "chain": r.chain, "chain_seed": r.seed,
              "appearance": "random", "appearance_seed": r.layout["look_seed"]})
    t0 = time.time()
    # 1. the policy's attempt, replayed
    for a in actions:
        env.step(a)
    # 2. hand-over, from the arm's ACTUAL state: where the policy pressed the
    # arm into the rack its joint command ran up to 0.8 rad past the blocked
    # arm, which then lunged at the stale command once the gripper let go
    # (and the planner started from it). Re-sync the command to the measured
    # joints, then open the gripper and back off upward.
    q_meas = np.asarray(env.arm.get_joint_positions())[0]
    gap = float(np.abs(env.q_cmd[:7] - q_meas[:7]).max())
    env.q_cmd[:7] = q_meas[:7]
    env.arm.set_joint_position_targets(env.q_cmd)
    ex.gripper(0.0, 10)
    tcp, R = r.ls.get_tcp_pose()
    for dz in BACK_OFF_M:
        if dz == 0.0 or ex.plan_and_go([Waypoint("back_off", tcp + [0, 0, dz], R, r.ls.GRIPPER_STROKE,
                                                 speed=0.04)]).success:
            break
    # 3. the expert's primitive from here -- or from home if it can't plan from here
    prim = {"pick_tube_from_holder": lambda: ex.pick_tube_from_holder(r.tube),
            "pick_up_tray": ex.pick_up_tray, "pick_up_pipette": ex.pick_up_pipette}[task]
    via_home = False
    for attempt in (0, 1):
        n0 = len(env.recorder.rows)
        try:
            res = prim()
        except Exception as e:
            res = Result(False, {"exception": repr(e)})
        moved = len(env.recorder.rows) > n0
        if res.success or moved or attempt == 1:
            break
        print(f"[recover] {task} can't plan from where the policy left the arm "
              f"({json.dumps(res.info, default=float)[:200]}); going home first", flush=True)
        go_home(r)
        via_home = True
    n = len(env.recorder.rows)
    ok = bool(res.success)
    env.recorder.finish(ok, {**res.info, "prefix_steps": int(len(actions)), "expert_steps": n - len(actions),
                             "via_home": via_home, "command_gap_at_handover_rad": gap})
    env.recorder = None
    if not ok:
        dst = out / "_failed" / rel
        shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out / rel), str(dst))
        rel = f"_failed/{rel}"
    row = {"dir": rel, "name": task, "prompt": prompt, "success": ok, "chained": False, "steps": n,
           "prefix_steps": int(len(actions)), "expert_steps": n - len(actions), "via_home": via_home,
           "source": str(src),
           "seconds_wall": round(time.time() - t0, 1),
           "info_short": json.dumps(res.info, default=float)[:600]}
    print("RESULT", json.dumps(row, default=float), flush=True)
    r.app.close()


# ------------------------------------------------------------------ driver
def summarize(out):
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines() if x.strip()]
    s = {"episodes": rows, "valid_episodes": sum(x["success"] for x in rows),
         "failed_episodes": sum(not x["success"] for x in rows),
         "by_task": {t: f"{sum(x['success'] for x in rows if x['name'] == t)}/{sum(x['name'] == t for x in rows)}"
                     for t in sorted({x["name"] for x in rows})}}
    (out / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def drive(args):
    out = Path(args.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "results.jsonl").exists():          # resume
        done = {json.loads(x)["source"] for x in (out / "results.jsonl").read_text().splitlines() if x.strip()}
    srcs = [(name, ep) for name in args.evals
            for ep in sorted((EVAL_ROOT / name).glob("*/ep_*")) if (ep / "steps.npz").exists()]
    print(f"{len(srcs)} eval trials from {len(args.evals)} runs", flush=True)
    for name, ep in srcs:
        if str(ep) in done:
            continue
        tag = f"{name}_{ep.parent.name}_{ep.name[3:]}"
        log = out / "logs" / f"{tag}.log"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--src", str(ep), "--tag", tag,
               "--out", str(out)]
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE), timeout=TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        line = next((x for x in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if x.startswith("RESULT ")), None)
        res = json.loads(line[7:]) if line else {"dir": None, "name": ep.parent.name, "success": False,
                                                 "source": str(ep), "error": f"no result (exit {rc})"}
        with open(out / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[{time.strftime('%H:%M')}] {tag}: {'ok' if res['success'] else 'FAILED'} "
              f"({res.get('prefix_steps', '?')} policy + {res.get('expert_steps', '?')} expert steps) "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    s = summarize(out)
    print("expert finished:", s["by_task"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", nargs="+", help="eval run folders under data/eval")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--src", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tag", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        run_trial(args)
    else:
        drive(args)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
