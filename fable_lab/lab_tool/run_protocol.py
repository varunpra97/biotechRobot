r"""Execute a planned experiment (protocol.plan) in the full lab scene, in ONE continuous physics simulation:
every primitive step starts from the state the previous one left. Each step is done by the generator's scripted
expert for that primitive (sim/expert.py, sim/pipette_expert.py: motion planning + IK). Liquids are tracked volumes
(sim/liquids.py), not a fluid simulation; the centrifuge does not spin; waits are compressed.

Writes <out>/status.json after every step (the web app polls it), every step's camera frames under <out>/steps/,
and at the end <out>/protocol.mp4.

Usage (Isaac Sim venv, from the BiotechRobot folder):
    .venv-isaac\Scripts\python.exe biotech-lab-robotics\fable_lab\lab_tool\run_protocol.py --plan plan.json --out data\lab_tool\runs\demo
    ... --text "Load solvent A into the test tube, centrifuge it for 10 s, ..."     (plans it first)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parents[1] / "simulation" / "sim"
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE))
import protocol  # noqa: E402

READABLE = {
    "pick_up_pipette": "pick up the pipette", "attach_tip": "attach tip {tip}",
    "aspirate_from_tube": "aspirate {ul:.0f} uL from tube {slot}", "dispense_into_tube": "dispense into tube {slot}",
    "dispense_into_well": "dispense into well {well}", "eject_tip": "eject the tip into the bin",
    "return_pipette": "put the pipette back", "pick_tube_from_holder": "pick tube {slot} from the rack",
    "insert_tube_into_bore": "insert the tube into centrifuge bore {bore}", "wait": "wait {seconds:.0f} s ({note})",
    "remove_tube_from_bore": "remove the tube from centrifuge bore {bore}",
    "place_tube_into_holder": "put the tube back into rack slot {slot}", "pick_up_tray": "pick up the tray",
    "place_tray": "set the tray down",
}


def rc(slot):
    return ("ABCD".index(slot[0]), int(slot[1:]) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None, help="JSON written by protocol.plan()")
    ap.add_argument("--text", default=None, help="plan this description instead of --plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = protocol.plan(args.text) if args.text else json.loads(Path(args.plan).read_text())
    if plan.get("error") or not plan["steps"]:
        sys.exit("plan has an error: " + str(plan.get("error")))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    bench, steps = plan["bench"], plan["steps"]
    status = {"state": "starting the simulator", "text": plan["text"], "commands": plan["descriptions"],
              "steps": [{**s, "state": "pending", "label": None} for s in steps], "current": None,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "liquids": None, "events": [], "error": None,
              "planned_final": plan["final"], "video": None, "driver": "scripted experts (motion planning + IK)"}

    def save():
        tmp = out / "status.tmp"
        tmp.write_text(json.dumps(status, indent=1, default=float))
        tmp.replace(out / "status.json")

    save()
    import collect_primitives as cp
    import layout
    layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)       # the fixed default bench
    rargs = argparse.Namespace(out=str(out / "scenes"), seed=args.seed, chain=3000, max_episodes=10 ** 9, prior_valid=0,
                               prior_chained=0, force_chained=False, pick_slot=bench["test_tube"], block=None,
                               extra_rigid=tuple(cp.tube_name(s) for s in cp.SLOTS))      # every tube can be moved
    (out / "scenes" / "chains").mkdir(parents=True, exist_ok=True)
    r = cp.ChainRunner(rargs)
    r.layout["empty_slots"] = sorted(list(rc(s)) for s in bench["empty_slots"])
    r.layout["liquids"] = {cp.tube_name(rc(s)): float(t["ul"]) for s, t in bench["tubes"].items() if t["ul"] > 0}
    code = 1
    try:
        r.build()
        import lab_env
        import lab_scene
        import numpy as np
        env, ex = r.env, r.ex
        for s, t in bench["tubes"].items():                    # each named liquid in its own colour
            L = lab_scene.LIQUIDS.get(cp.tube_name(rc(s)))
            if L is not None:
                L.set_color(tuple(t["color"]))
        for _ in range(5):
            r.app.update()
        slot_of = {cp.tube_name(rc(s)): s for s in protocol.SLOTS}             # a tube keeps its home slot's label
        bores = sorted((i for i in lab_scene.CENTRIFUGE_SLEEVE_PARTS if r.survey["bores"].get(r.bore_label(i), -1) > cp.REACH_MARGIN),
                       key=lambda i: -r.survey["bores"][r.bore_label(i)])
        if not bores:
            raise RuntimeError("no centrifuge bore is within reach")
        B = bores[0]
        blabel = r.bore_label(B)

        def liquids():
            tubes = {slot_of[n]: round(L.volume_ul, 1) for n, L in lab_scene.LIQUIDS.items() if n in slot_of and L.volume_ul > 0.05}
            wells = {n.split("well_")[-1]: round(L.volume_ul, 1) for n, L in lab_scene.LIQUIDS.items()
                     if "well_" in n and L.volume_ul > 0.05}
            return {"tubes": tubes, "wells": wells, "in_tip": round(float(getattr(env.pipette, "tip_volume", 0.0) or 0.0), 1)}

        def call(s):
            p = s["prim"]
            tube = cp.tube_name(rc(s["slot"])) if "slot" in s else None
            if p == "pick_up_pipette":
                return ex.pick_up_pipette()
            if p == "attach_tip":
                return ex.attach_tip(f"tip_{s['tip']}")
            if p == "aspirate_from_tube":
                return ex.aspirate_from_tube(tube, float(s["ul"]))
            if p == "dispense_into_tube":
                return ex.dispense_into_tube(tube)
            if p == "dispense_into_well":
                return ex.dispense_into_well(f"well_{s['well']}")
            if p == "eject_tip":
                return ex.eject_tip()
            if p == "return_pipette":
                return ex.return_pipette()
            if p == "pick_tube_from_holder":
                return ex.pick_tube_from_holder(tube, ex.closing_for_bore(B))
            if p == "insert_tube_into_bore":
                return ex.insert_tube_into_bore(tube, B)
            if p == "remove_tube_from_bore":
                return ex.remove_tube_from_bore(tube, B)
            if p == "place_tube_into_holder":
                return ex.place_tube_into_holder(tube, rc(s["slot"]), ex.closing_for_bore(B))
            if p == "pick_up_tray":
                return ex.pick_up_tray()
            if p == "place_tray":
                return ex.place_tray(lab_scene.SAMPLE_TRAY_XY)
            if p == "wait":
                env.hold(int(round(float(s["seconds"]) * lab_env.CONTROL_HZ)))
                return None
            raise RuntimeError("unknown primitive " + p)

        import protocol_video
        status["state"] = "running"
        status["liquids"] = liquids()
        status["bore"] = blabel
        save()
        for k, s in enumerate(status["steps"]):
            s["bore"] = blabel
            s["label"] = READABLE[s["prim"]].format(**{"note": "", **s})
            s["state"], s["dir"] = "running", f"steps/step_{k:02d}_{s['prim']}"
            status["current"] = k
            save()
            t0 = time.time()
            env.recorder = lab_env.EpisodeRecorder(out / s["dir"], s["label"], wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA.get(s["prim"], "wrist_top"),
                                                   meta={"primitive": s["prim"], "protocol_step": k, "expert": True})
            n_ev = len(env.pipette.events)
            try:
                res = call(s)
                ok, info = (True, {}) if res is None else (bool(res.success), res.info)
                # A step the expert gave up on BEFORE doing anything irreversible is tried again from where the arm
                # stands: a planning rejection (the planner restores the world), or the tube still held above the
                # rack slot because its tip would not settle over the bore (first full run, step 11).
                while (not ok and s.get("tries", 1) < 3 and isinstance(info, dict)
                       and ("planning_error" in info or str(info.get("error", "")).startswith("tube not over the slot"))):
                    s["tries"] = s.get("tries", 1) + 1
                    print(f"[protocol] step {k + 1}: retry {s['tries']} after {json.dumps(info, default=float)[:160]}", flush=True)
                    env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]
                    res = call(s)
                    ok, info = bool(res.success), res.info
            except Exception as e:
                ok, info = False, {"exception": repr(e), "traceback": traceback.format_exc()[-1500:]}
            s["sim_steps"] = len(env.recorder.rows)
            env.recorder.finish(ok, {})
            env.recorder = None
            s["clip"] = protocol_video.step_clip(out / s["dir"])
            env.q_cmd[:7] = np.asarray(env.arm.get_joint_positions())[0][:7]
            new = [{"protocol_step": k, **e} for e in env.pipette.events[n_ev:]]
            status["events"] += new
            s["seconds_wall"] = round(time.time() - t0, 1)
            s["state"] = "ok" if ok else "failed"
            s["info"] = json.dumps(info, default=float)[:600]
            status["liquids"] = liquids()
            save()
            print(f"[protocol] step {k + 1}/{len(steps)} {s['label']}: {'ok' if ok else 'FAILED'} "
                  f"({s['sim_steps']} sim steps, {s['seconds_wall']} s wall) {s['info'][:200]}", flush=True)
            if not ok:
                status["error"] = f"step {k + 1} ({s['label']}) failed"
                break
        bad = [e for e in status["events"] if e["event"] in ("aspirated_air", "spilled", "aspirate_without_tip",
                                                              "ejected_with_liquid", "tip_hit_floor")]
        got, want = status["liquids"], plan["final"]
        vol_ok = all(abs(got["tubes"].get(sl, 0.0) - v) < 2.0 for sl, v in want["tubes"].items()) and \
            all(abs(got["wells"].get(w, 0.0) - v) < 2.0 for w, v in want["wells"].items())
        status["volumes_as_planned"] = bool(vol_ok)
        status["pipetting_faults"] = bad
        done = status["error"] is None
        status["result"] = ("SUCCESS" if done and vol_ok and not bad else
                            "STEPS DONE, but the liquids are not as planned" if done else "FAILED")
        status["state"] = "making the video"
        save()
        if not args.no_video:
            try:
                import protocol_video
                status["video"] = protocol_video.make(out)
            except Exception as e:
                status["video_error"] = repr(e)
        status["state"] = "done" if done else "failed"
        save()
        print("PROTOCOL", json.dumps({"result": status["result"], "steps_ok": sum(s["state"] == "ok" for s in status["steps"]),
                                      "of": len(steps), "liquids": status["liquids"]}), flush=True)
        code = 0 if done else 2
    except Exception as e:
        traceback.print_exc()
        status["state"], status["error"] = "failed", repr(e)
        save()
    sys.stdout.flush()
    os._exit(code)                                   # Isaac Sim does not always exit on its own


if __name__ == "__main__":
    main()
