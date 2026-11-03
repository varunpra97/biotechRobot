r"""Mini milestone m4 check in the generator: the expert picks the trial's
tube from its rack slot (default layout), then the RL tube-place teacher
(rl/place_policy.pt, sim/teacher_finisher.PlaceFinisher) puts it back into
the same slot. Success is the expert's own place test
(expert.place_tube_into_holder): tip on the bore floor (< 0.5 mm), within
2 mm of the slot's centre, tilt < 3 deg, unmoved by the hand backing off,
nothing else moved. Each trial is recorded like an eval episode.

Usage (Isaac venv, from the BiotechRobot folder; no pi0.5 server needed):
    .venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\eval_place_finisher.py --trials 5
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\varun\workspace\BiotechRobot\data\eval")
EVAL_SEED = 7
STEPS = 120                       # the teacher's budget (its training episode length)
# --scripted-release: once the teacher has the tube seated (tip this close to the bore floor, this close to the
# slot's axis), open in place and back the hand off straight up -- as the tube pick's finisher ends with a
# scripted lift (the m4 teacher seats tubes but never learned to let go)
SEATED_Z_M, SEATED_XY_M = 0.001, 0.0015
OPEN_STEPS, BACKOFF_STEPS = 4, 4
BACKED_OFF_M = 0.02               # done once released this far above the tube's grasp point
SETTLE_STEPS = 15


def trial(args):
    import numpy as np

    import collect_primitives as cp
    import lab_env
    import lab_scene
    import layout
    from teacher_finisher import PLACE_POLICY, PlaceFinisher

    out = Path(args.out)
    layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)
    rargs = argparse.Namespace(out=str(out / "scenes"), seed=EVAL_SEED, chain=args.trial, max_episodes=10 ** 9,
                               prior_valid=0, prior_chained=0, force_chained=False, pick_slot=None, block=None)
    (out / "scenes" / "chains").mkdir(parents=True, exist_ok=True)
    r = cp.ChainRunner(rargs)
    r.build()
    env, ex = r.env, r.ex
    slot = tuple(r.layout["pick_slot"])
    label = cp.slot_label(slot)
    rel = f"place_tube_into_holder/ep_{args.trial:03d}"
    env.recorder = lab_env.EpisodeRecorder(
        out / rel, f"put the tube back into rack slot {label}", wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA["pick_tube_from_holder"],
        meta={"primitive": "place_tube_into_holder", "eval": True, "trial": args.trial, "layout_seed": r.seed,
              "layout": r.layout, "checkpoint": f"expert pick + {args.policy or PLACE_POLICY}"})
    res = ex.pick_tube_from_holder(r.tube)
    print(f"[m4] expert pick from {label}: success {res.success}", flush=True)
    info = {"slot": label, "pick_ok": bool(res.success)}
    ok = False
    if res.success:
        before = ex.snapshot_objects()
        env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]     # re-sync to the measured arm
        fin = PlaceFinisher(env, r.planner.kin, env.objects[r.tube]["path"], slot, policy_path=args.policy or PLACE_POLICY)
        fin.start()
        for k in range(STEPS):
            d, tilt, gap, off = fin.place_status()
            if k % 10 == 0:
                print(f"[m4] step {k}: tip-rest {np.round(d * 1000, 1).tolist()} mm, tilt {tilt:.1f}, gap {gap * 1000:.1f} mm, "
                      f"closed {fin.closed}, last {np.round(fin.last, 2).tolist()}", flush=True)
            if not fin.closed and gap > 0.009 and off[2] > BACKED_OFF_M:
                break
            if args.scripted_release and fin.closed and abs(d[2]) < SEATED_Z_M and np.hypot(d[0], d[1]) < SEATED_XY_M:
                print(f"[m4] seated at step {k}: tip-rest {np.round(d * 1000, 2).tolist()} mm -> scripted release", flush=True)
                info["seated_step"] = k
                for j in range(OPEN_STEPS + BACKOFF_STEPS):
                    env.step(fin._ik_action(np.array([0.0, 0.0, 0.0 if j < OPEN_STEPS else 1.0, -1.0])))
                break
            env.step(fin.action())
        released = not fin.closed
        p0 = ex.pose(r.tube)
        for _ in range(SETTLE_STEPS):                                       # hold still; anything still moving shows
            env.step(fin.hold_action() if not released else env.action(np.zeros(7), gripper=0.0))
        q = ex.pose(r.tube)
        sx, sy = lab_scene.holder_slot_xy(*slot)
        tilt = math.degrees(math.acos(np.clip(lab_scene.quat_to_matrix(q[3:7])[2, 2], -1, 1)))
        info.update({"teacher_steps": k + 1, "released": released, "tip_above_floor_m": float(q[2] - ex.holder_floor),
                     "radial_offset_m": float(np.hypot(q[0] - sx, q[1] - sy)), "tilt_deg": tilt,
                     "moved_after_release_m": float(np.linalg.norm(q[:3] - p0[:3])),
                     "others_moved": ex.others_unmoved(before, {r.tube})})
        ok = (released and abs(info["tip_above_floor_m"]) < 5e-4 and info["radial_offset_m"] < 2e-3 and tilt < 3.0
              and info["moved_after_release_m"] < 1e-3 and not info["others_moved"])
    env.recorder.finish(ok, info)
    env.recorder = None
    row = {"trial": args.trial, "success": bool(ok), **info}
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=float) + "\n")
    print("RESULT", json.dumps(row, default=float), flush=True)
    r.app.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--name", default="m4_place_finisher")
    ap.add_argument("--policy", default=None, help="TorchScript policy (default rl/place_policy.pt)")
    ap.add_argument("--scripted-release", action="store_true", help="open and back off once the teacher has it seated")
    ap.add_argument("--trial", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.trial is not None:
        sys.path.insert(0, str(HERE))
        trial(args)
        return
    out = OUT / args.name
    (out / "logs").mkdir(parents=True, exist_ok=True)
    for t in range(args.trials):
        log = out / "logs" / f"trial_{t:03d}.log"
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trial", str(t), "--out", str(out)]
                           + (["--policy", args.policy] if args.policy else [])
                           + (["--scripted-release"] if args.scripted_release else []),
                           stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE))
        line = next((x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.startswith("RESULT ")), None)
        print(f"trial {t}: {line[7:] if line else 'no result, see ' + str(log)}", flush=True)


if __name__ == "__main__":
    main()
