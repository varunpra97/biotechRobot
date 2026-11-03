# RL proof of concept: 3 scenarios in ~4 days (for review)

Draft 2026-10-25. Purpose: show that RL skill policies, given target poses from the camera localizer, can do lab primitives end-to-end, before committing to the full plan (`RL_EXECUTION_PLAN.md`, 4–5 weeks). Default layout only. Training overnight; development and evals in the day.

## Why these three

Each answers a different question the full plan depends on:

| # | Scenario | Question it answers | Builds on |
|---|---|---|---|
| A | **Tube pick from home, no VLA**, perceived target pose | Can an RL policy do a whole primitive alone, with real perception noise, from where a task actually starts? (Your "RL from the beginning" question.) | the existing teacher (warm start) + localizer |
| B | **Pick up a tube lying on the bench** (knocked over) | Does it extend to off-nominal states no scripted expert handles? (The D5 failure.) | same env; new start states; + wrist yaw |
| C | **Pick up the sample tray** (lift level) | Does the framework carry over to a different object with little new code? | same env pattern; tray asset |

Not in the PoC: the centrifuge (contact-heavy insertion, highest risk), the pipette (needs the liquid model ported), placing, random layouts, and camera-based students.

## Day by day

**Day 1, Scenario A**
- **Curriculum:** start-distance curriculum from near-target out to home (auto-widened as success passes 80%). Episodes extended to 12 s for home starts.
- **Robustness:** observation noise and 1-step latency matching the localizer's measured error (approach 0.3 mm median / 1.7 mm p95, in hand 2 mm / 4.6 mm, tilt ±3°).
- **Training:** warm start from the current teacher (model_2297). Overnight run, ~6–8 h.

**Day 2**
- **Morning, A eval in our generator:** no π0.5, teacher from home, target from the live localizer, 20 trials (10 random slots, 5 at B1, 5 at D5). Compare with π0.5 + finisher, 17/20.
- **Afternoon, Scenario B setup:**
  - start states with the target tube lying on the bench at random positions and angles near the rack, plus the real knocked-over state from the D5 trial;
  - action adds wrist yaw (fingers close across the tube body);
  - success: lifted 3 cm, held.
- **Night:** overnight training.

**Day 3**
- **Morning, B eval:** in Isaac Lab, plus a replay in our generator from the actual D5 failure.
- **Afternoon, Scenario C setup:**
  - tray asset (exported like the tube);
  - scene with no tubes, so faster;
  - start states from home and from recorded tray approaches;
  - action xyz + yaw + grip;
  - success: lifted 6 cm (the generator's tray lift), tilt < 3°, nothing else disturbed.
- **Night:** overnight training.

**Day 4**
- **C eval:** in Isaac Lab, plus a generator eval of `pick_up_tray` with the teacher. π0.5 alone scored 0/5 on tray picks.
- **Report:** results, video per scenario, and a go/no-go recommendation for the full plan.

## Go / no-go bar

| Scenario | Pass | Where measured |
|---|---|---|
| A | ≥ 80% of 20 trials, perceived pose, no VLA | our generator (the real eval) |
| B | ≥ 70% from lying-tube starts | Isaac Lab (200 episodes) + the D5 replay |
| C | ≥ 80% | Isaac Lab (200 episodes) + generator eval |

**Interpreting the outcome:**
- **All three pass:** the full plan is worth committing to; its estimate drops because the env pattern, curriculum and eval tooling are proven.
- **A fails:** RL-from-home with perception isn't ready, and the finisher role (17/20 today) is the fallback.
- **B or C fails:** shows which kind of skill needs more than RL (e.g. a scripted re-orientation, or residual RL).

## What it costs

- **GPU:** 3 overnight trainings plus about 2 h of evals per day. π0.5 fine-tunes would have to wait or run on the off-days.
- **Code:**
  - curriculum + noise in `tube_grasp` (small);
  - lying-tube start states + yaw action (small–medium);
  - tray variant of the env (medium);
  - teacher-only mode in `eval_policy.py` (small).
- **Coordination:** the localizer server runs during the A and C evals in the generator (vision session's code, its own process).
- **Risks:** the warm-started A may need a fresh run (+1 night). The lying-tube grasp may need pitch as well as yaw (+1 night). If either slips, the PoC is 5 days, not 4.
