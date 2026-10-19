"""Generate one recorded 15 Hz episode per lab primitive.

Runs the primitives as a chain in one physical simulation (each starts from
the state the previous one left -- the tube really is in the centrifuge
when "remove" starts) and records every primitive as its own episode with
its own prompt, as π0.5 fine-tuning wants them.

Output: <out>/<primitive>/ep_0000/ (see lab_env.EpisodeRecorder):
  images/<camera>_<step>.jpg  all cameras, 15 Hz
  steps.npz                   positions, velocities, actions, states, poses
  exterior_wrist.mp4          the two policy cameras side by side
  meta.json                   prompt, success, outcome measurements

Usage (Isaac venv, from the BiotechRobot folder):
    python biotech-lab-robotics\\simulation\\sim\\collect_primitives.py --out data\\primitives
    python biotech-lab-robotics\\simulation\\sim\\collect_primitives.py --only tubes
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\data\primitives")
    ap.add_argument("--only", default="all", choices=["all", "tubes", "pipette", "tray"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--appearance", default="baseline", choices=["baseline", "random"],
                    help="random: materials, lighting and colours randomized from --seed (appearance.py)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    import lab_env
    import lab_scene
    import prompts
    from pipette import PipetteModel
    from pipette_expert import PipetteExpert
    from planner import Planner

    TUBE = "tube_r0_c2"
    BORE = 8
    rigid = (TUBE, "pipette_0", "tray")
    rng = random.Random(args.seed)
    env = lab_env.LabEnv(simulation_app, rigid_objects=rigid)
    env.pipette = PipetteModel(env)
    planner = Planner(simulation_app, env.arm, [env.objects[n]["path"] for n in rigid])
    expert = PipetteExpert(env, planner, seed=args.seed)
    import appearance
    import isaacsim.core.experimental.utils.stage as stage_utils
    look = appearance.apply(stage_utils.get_current_stage(),
                            random.Random(args.seed + 7919) if args.appearance == "random" else None)
    for _ in range(10):        # let the renderer pick up the new materials before recording
        simulation_app.update()
    out = Path(args.out)
    slot = prompts.slot_name(0, 2)
    bore = prompts.bore_name(BORE, lab_scene.CENTRIFUGE_SLEEVE_PARTS)

    chain = []
    if args.only in ("all", "tubes"):
        chain += [
            ("pick_tube_from_holder", dict(slot=slot), lambda: expert.pick_tube_from_holder(TUBE)),
            ("insert_tube_into_bore", dict(bore=bore), lambda: expert.insert_tube_into_bore(TUBE, BORE)),
            ("remove_tube_from_bore", dict(bore=bore), lambda: expert.remove_tube_from_bore(TUBE, BORE)),
            ("place_tube_into_holder", dict(slot=slot), lambda: expert.place_tube_into_holder(TUBE, (0, 2))),
        ]

    if args.only in ("all", "pipette"):
        src, dst = "tube_r1_c0", "tube_r1_c1"
        chain += [
            ("pick_up_pipette", {}, lambda: expert.pick_up_pipette()),
            ("attach_tip", dict(tip="A1"), lambda: expert.attach_tip("tip_A1")),
            ("aspirate_from_tube", dict(vol=200, slot=prompts.slot_name(1, 0)),
             lambda: expert.aspirate_from_tube(src, 200.0)),
            ("dispense_into_well", dict(well="A1"), lambda: expert.dispense_into_well("well_A1")),
            ("eject_tip", {}, lambda: expert.eject_tip()),
            ("attach_tip", dict(tip="A2"), lambda: expert.attach_tip("tip_A2")),
            ("aspirate_from_tube", dict(vol=150, slot=prompts.slot_name(1, 0)),
             lambda: expert.aspirate_from_tube(src, 150.0)),
            ("dispense_into_tube", dict(slot=prompts.slot_name(1, 1)), lambda: expert.dispense_into_tube(dst)),
            ("eject_tip", {}, lambda: expert.eject_tip()),
            ("return_pipette", {}, lambda: expert.return_pipette()),
        ]

    if args.only in ("all", "tray"):
        spots = lab_scene.TRAY_SPOTS_XY
        chain += [
            ("pick_up_tray", {}, lambda: expert.pick_up_tray()),
            ("place_tray", dict(spot=prompts.TRAY_SPOTS["right"]), lambda: expert.place_tray(spots["right"])),
            ("pick_up_tray", {}, lambda: expert.pick_up_tray()),
            ("place_tray", dict(spot=prompts.TRAY_SPOTS["home"]), lambda: expert.place_tray(spots["home"])),
        ]

    summary = []
    counts = {}
    for name, kw, run in chain:
        prompt = prompts.make(name, rng, **kw)
        k = counts[name] = counts.get(name, -1) + 1
        env.recorder = lab_env.EpisodeRecorder(out / name / f"ep_{k:04d}", prompt,
                                               meta={"primitive": name, "args": kw,
                                                     "appearance": args.appearance, "appearance_seed": args.seed,
                                                     "look": look},
                                               wrist_camera=lab_env.PRIMITIVE_WRIST_CAMERA[name])
        t0 = time.time()
        res = run()
        ep_dir = env.recorder.finish(res.success, res.info)
        n = env.recorder.rows.__len__()
        env.recorder = None
        summary.append({"primitive": name, "prompt": prompt, "success": res.success,
                        "steps": n, "seconds_wall": round(time.time() - t0, 1), "info": res.info})
        print(f"[{name} #{k}] success={res.success} steps={n} ({n / lab_env.CONTROL_HZ:.1f}s sim, "
              f"{time.time() - t0:.0f}s wall) '{prompt}'\n    {json.dumps(res.info, default=float)[:600]}",
              flush=True)
        if not res.success:
            print(f"stopping the chain: {name} failed", flush=True)
            break
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    simulation_app.close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
