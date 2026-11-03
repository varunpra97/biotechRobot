r"""Multi-step scenarios in one continuous scene: pi0.5 does each step, the
trained RL finisher takes over near the target (eval_policy.run_task), the
expert does the steps with no RL milestone (and, labelled, any step that
failed, so the scenario can go on). Every step is recorded; a labelled video
of the whole scenario is written next to them.

Scenarios (default layout):
  tubes: pick tube S1 -> put it back -> pick tube S2 -> put it back
         (M1 tube pick with the camera-estimated pose, M4 tube place)
  bench: pick up the tray -> (expert) put it back -> pick tube S1 -> put it
         back -> pick up the pipette and turn it nozzle-down
         (M2 tray, M1 tube, M4 place, M3 pipette)

Usage (Isaac venv, from the BiotechRobot folder; pi0.5 server on --server,
the localizer server on --localizer-port for camera-estimated tube poses):
    .venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\run_scenario.py --scenario tubes --chain 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(r"C:\Users\varun\workspace\BiotechRobot\data\eval\scenarios")
SCENARIOS = {
    "tubes": [("pick_tube_from_holder", "S1"), ("place_tube_into_holder", "S1"),
              ("pick_tube_from_holder", "S2"), ("place_tube_into_holder", "S2")],
    "bench": [("pick_up_tray", None), ("expert:place_tray", None), ("pick_tube_from_holder", "S1"),
              ("place_tube_into_holder", "S1"), ("pick_up_pipette", None)],
}
MILESTONE = {"pick_tube_from_holder": "M1 tube-pick teacher", "place_tube_into_holder": "M4 tube-place teacher",
             "pick_up_tray": "M2 tray teacher", "pick_up_pipette": "M3 pipette teacher"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--chain", type=int, default=0, help="layout / look seed index")
    ap.add_argument("--server", default="ws://localhost:8000")
    ap.add_argument("--checkpoint-label", default="")
    ap.add_argument("--localizer-port", type=int, default=5577)
    ap.add_argument("--no-perceived", action="store_true", help="tube steps use the exact tube pose")
    ap.add_argument("--no-vla", action="store_true", help="RL only: each step's teacher drives from its start, no pi0.5")
    args = ap.parse_args()

    import collect_primitives as cp
    import eval_policy as ev
    import layout

    name = f"{args.scenario}_{args.chain:02d}" + ("_rl_only" if args.no_vla else "")
    out = OUT / name
    (out / "scenes" / "chains").mkdir(parents=True, exist_ok=True)
    layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)
    rargs = argparse.Namespace(out=str(out / "scenes"), seed=ev.EVAL_SEED, chain=1000 + args.chain, max_episodes=10 ** 9,
                               prior_valid=0, prior_chained=0, force_chained=False, pick_slot=None, block=None,
                               extra_rigid=tuple(cp.tube_name(s) for s in cp.SLOTS))   # every tube can be moved
    r = cp.ChainRunner(rargs)
    r.build()
    s1 = tuple(r.layout["pick_slot"])
    others = sorted(s for s in r.slot_tube if s != s1 and r.survey["slots"].get(cp.slot_label(s), -1) > cp.REACH_MARGIN)
    s2 = max(others, key=lambda s: abs(s[0] - s1[0]) + abs(s[1] - s1[1]))          # a slot far from S1
    slots = {"S1": s1, "S2": s2}
    client = None if args.no_vla else ev.PolicyClient(args.server)
    steps, t0 = [], time.time()
    try:
        run_steps(args, r, out, name, slots, client, steps)
        error = None
    except Exception as e:                               # record it and exit: Isaac Sim does not exit on its own
        import traceback
        traceback.print_exc()
        error = repr(e)
    summary = {"scenario": name, "steps": steps, "rl_steps_ok": sum(1 for s in steps if s["success"]),
               "minutes": round((time.time() - t0) / 60, 1), "error": error}
    (out / "scenario.json").write_text(json.dumps(summary, indent=1, default=float))
    print("SCENARIO", json.dumps(summary, default=float), flush=True)
    if error:
        os._exit(1)
    r.app.close()


def run_steps(args, r, out, name, slots, client, steps):
    import numpy as np

    import collect_primitives as cp
    import eval_policy as ev
    import lab_env
    import lab_scene

    env, ex = r.env, r.ex
    for k, (task, slot) in enumerate(SCENARIOS[args.scenario]):
        rel = f"step_{k:02d}_{task.replace(':', '_')}"
        row = {"step": k, "task": task, "dir": rel}
        if task == "expert:place_tray":
            env.recorder = lab_env.EpisodeRecorder(out / rel, "put the tray back on its spot",
                                                   wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA["pick_up_tray"],
                                                   meta={"primitive": "place_tray", "scenario": name, "expert": True})
            res = ex.place_tray(lab_scene.SAMPLE_TRAY_XY)
            env.recorder.finish(bool(res.success), {})
            env.recorder = None
            row.update(success=bool(res.success), by="expert (no RL milestone for this step)",
                       prompt="put the tray back on its spot")
        else:
            rc = slots.get(slot) if slot else None
            tube = r.slot_tube.get(rc) if rc else None
            targs = argparse.Namespace(task=task, trial=k, finisher=True, perceived=(task == "pick_tube_from_holder"
                                                                                     and not args.no_perceived),
                                       localizer_port=args.localizer_port, finisher_near=0.25, replan=8,
                                       checkpoint_label=args.checkpoint_label, server=args.server, out=str(out),
                                       no_vla=args.no_vla)
            res = ev.run_task(targs, r, out, setup=False, tube=tube, slot_rc=rc, rel=rel, client=client)
            row.update(success=res["success"], by=(f"RL only: {MILESTONE[task]}" if args.no_vla
                                                   else f"pi0.5 + {MILESTONE[task]}"), prompt=res["prompt"],
                       handover_step=res["outcome"].get("handover_step"), teacher_step=res["outcome"].get("teacher_lift_step"),
                       slot=cp.slot_label(rc) if rc else None)
            if not res["success"]:                       # the expert does the step, so the scenario can go on
                fallback = {"pick_tube_from_holder": lambda: ex.pick_tube_from_holder(tube),
                            "place_tube_into_holder": lambda: ex.place_tube_into_holder(tube, rc),
                            "pick_up_tray": ex.pick_up_tray, "pick_up_pipette": ex.pick_up_pipette}[task]
                frel = rel + "_expert_fallback"
                env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]
                env.recorder = lab_env.EpisodeRecorder(out / frel, row["prompt"] + " (expert fallback)",
                                                       wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA[task],
                                                       meta={"primitive": task, "scenario": name, "expert": True})
                fres = fallback()
                env.recorder.finish(bool(fres.success), {})
                env.recorder = None
                row["fallback"] = {"dir": frel, "success": bool(fres.success)}
                if not fres.success:
                    steps.append(row)
                    print(f"[scenario] step {k} {task}: failed, and the expert fallback too; stopping", flush=True)
                    break
        steps.append(row)
        print(f"[scenario] step {k} {task}: {'SUCCESS' if row['success'] else 'fail'} ({row['by']})", flush=True)
        env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]


if __name__ == "__main__":
    main()
