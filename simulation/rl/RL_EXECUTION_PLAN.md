# RL teachers for every lab object: execution plan (for review)

Draft 2026-10-25. Status: **awaiting your review**. Nothing below has started.

## Where we are

- **One teacher exists:** tube pick from the rack, in Isaac Lab (`rl/tube_grasp`). It's PPO on exact sim state, default layout only.
- **Alone:** 99% success from near-target states, 0% from home.
- **As a finisher for π0.5** (π0.5 drives, the teacher takes over near the tube): **17/20** on the fixed-layout tube eval, against **0/20** for π0.5 alone. The 3 failures: hand-over 15 cm and 25 cm out (beyond the teacher's range), and a tube π0.5 had already knocked flat.
- **Lessons that shape the plan:**
  - Rewards get exploited: holding just short of success, and yanking the tube out tilted.
  - Start-state coverage decides where the teacher works.
  - Physics, not GPU count, limits speed: ~900 env steps/s, about one skill per overnight run.

## Goal

A teacher per manipulation skill in the generator. Each one:

1. **Finishes reliably** (≥ 95% from realistic start states, including π0.5's actual failure states), as a finisher in the eval.
2. **Serves as a reactive expert** for π0.5 recovery data (π0.5 fails → teacher recovers → episode for fine-tuning).
3. **Runs on perceived poses** (the vision session's wrist-camera localizer) instead of exact sim state, before any real-robot use.

## Skills (from the generator's primitives)

| # | Object | Skill | Hard part | Approach | Action space |
|---|---|---|---|---|---|
| T1 | tube / holder | pick upright tube from rack slot | precision (5 mm), neighbours | **done**; extend to random layouts + home starts | xyz + grip |
| T2 | tube / holder | place tube into rack slot (8 mm bore, 0.5 mm clearance) | insertion, jamming | residual RL on the scripted place | xyz + grip (+ small tilt) |
| T3 | tube / centrifuge | insert tube into rotor bore (45° bores) | tilted insertion, contact | residual RL on the scripted insert | 6-DoF delta pose + grip |
| T4 | tube / centrifuge | remove tube from rotor bore | re-grasp of a tilted tube (known fragile) | RL from scripted pre-grasp | 6-DoF + grip |
| T5 | tube | recover a tube lying on the bench, stand it up / put it in a slot | new grasp, re-orientation | RL, curriculum from upright → tilted → lying | 6-DoF + grip |
| T6 | tray | pick up tray (grasp across, keep level) | narrow grasp, level lift | RL | xyz + yaw + grip |
| T7 | tray | place tray at a target spot | level set-down, no knocking | RL | xyz + yaw + grip |
| T8 | pipette | pick up pipette (camera-facing grasp) | grasp orientation constraint | RL | 6-DoF + grip |
| T9 | pipette | attach tip (press nozzle into tip in rack) | precise press, seal | RL near the tip, scripted press | xyz (+ yaw) |
| T10 | pipette | aspirate / dispense (tip into tube / well to depth) | depth control, plunger timing | RL for tip positioning; plunger scripted | xyz + plunger trigger |
| T11 | pipette | eject tip over bin, return pipette | placement | scripted (works) unless failures show up | — |

The holder and centrifuge are fixtures: they appear as targets (T2–T4) and as randomized scene layout, not as objects to move.

## Phase 0: shared infrastructure (3–4 days)

1. **Common env base:**
   - **Scene:** our layout sampler (`layout.py`) in Isaac Lab, so every env has a random layout (rack, centrifuge, tray, tip rack, bin, arm base), as in the v4 data. All assets exported like `holder.usda` / `tube.usda`: centrifuge body + rotor, tray, pipette, tips, tip rack, bin.
   - **Actions:** switchable xyz+grip or 6-DoF delta-pose. IK and q_cmd accumulation as in T1 (already validated).
   - **Observations:** exact sim state for teachers, plus optional noise and latency to mimic the localizer.
2. **Start-state machinery:**
   - Recorded states from expert episodes and failed evals (as T1).
   - Generated perturbations around them.
   - "Disturbed scene" states: tube tilted, lying, half out, neighbours missing.
   - Automatic curriculum: widen the start radius and hardness as success rises.
3. **Reward and success template:**
   - Success checks reuse the generator's outcome checks, so a teacher's success equals the eval's.
   - Built-in guards: remaining-episode payout on success; per-condition shaping (height, tilt, hold); disturbance penalties.
4. **Eval harness:**
   - A success matrix per skill: start type × layout × distance, on held-out seeds.
   - Video tool.
   - Teacher export to TorchScript, plus a generic finisher hook in `eval_policy.py` for any primitive (today tube-only).
5. **Physics parity checks:** for each object, a scripted oracle runs in both the Isaac Lab env and our generator scene, and their outcomes must agree before training.

## Phases 1–3: teachers, in this order

**Phase 1: tubes** (highest value; the eval's tube tasks fail today)

- **T1+:** random layouts and home starts, with a curriculum and a start radius out to the whole platform. ~1 day + 1–2 overnight runs.
- **T5:** lying-tube recovery. Covers the D5 failure (tube knocked flat). ~2 days + 2 nights.
- **T2:** slot insertion (place). ~1 day + 1 night.
- **T3 / T4:** centrifuge insert and remove, 6-DoF, residual on the planner. ~3 days + 2–3 nights (contact-heavy).

**Phase 2: tray** (T6, T7). ~2 days + 2 nights.

**Phase 3: pipette** (T8–T10). The pipette's liquid model must be ported into the Isaac Lab env first (~1 day). ~4 days + 3 nights.

**Each teacher is done when:**
- ≥ 95% success from recorded failure states and generated starts on held-out seeds;
- ≤ 1% disturbance of other objects;
- and, as a finisher in `eval_policy.py`, it beats π0.5 alone.

## Phase 4: integration (runs alongside the later phases)

1. **Finisher everywhere:** the generic hand-over in the eval for every primitive. Report π0.5 alone vs π0.5 + teacher per task.
2. **DAgger data loop:** for each π0.5 failure, the teacher recovers in the generator, and the episode goes to π0.5 fine-tuning as recovery data (the expert part only, per your earlier choice).
3. **Perception swap:** the vision session's `TargetLocalizer`. Teachers retrain with noise/latency matching its measured error; then re-evaluate with perceived poses. The coordination is already agreed with that session.
4. **(Optional) vision students:** distill teachers into image-based policies if the localizer isn't accurate enough.

## Schedule and compute

- **Calendar:** Phase 0: 3–4 days. Phase 1: ~1.5 weeks. Phases 2–3: ~2 weeks. Integration overlaps. **About 4–5 weeks** in total, sequential, on one GPU.
- **GPU:** RL training (overnight) and π0.5 fine-tunes (~3–8 h) can't share the 12 GB card. Proposed: RL at night, fine-tunes and evals in the day, or alternate by request.

## Risks

- **Sim-to-sim gap:** Isaac Lab physics vs the generator. Mitigation: parity oracles (Phase 0.5) and evaluating finishers in the generator scene, as done for T1.
- **Contact-heavy skills** (T2–T4) may learn slowly. Mitigation: residual RL on the existing planner.
- **Privileged state:** finisher results overstate real-robot ability until Phase 4.3.
- **Reward exploits:** guarded by the template, and checked by inspecting eval failure reasons per run, as with T1.
- **Coordination:** the fork session (fine-tunes) and the vision session (localizer) share the GPU and folders. Each owns its own paths; GPU via the guard.

## Decisions for you

1. **Order:** tubes → tray → pipette as above, or a different priority?
2. **Scope of T1+:** random layouts and home starts now, or keep the teacher as a near-target finisher only?
3. **Phase 4.2:** use teachers to regenerate π0.5 recovery data (needs fine-tunes), or keep them as finishers only?
4. **Wrist RGB-D camera on the real robot?** (Pending; asked for the vision session.)
5. **GPU split:** RL at night and fine-tunes in the day — OK?
6. **Success bar:** ≥ 95% per teacher, ≤ 1% disturbance — right thresholds?
