# RL finishers after the VLA: mini milestones, then full milestones (for review)

Draft 2026-10-25. Replaces the "RL from the beginning" plans. **Architecture:** π0.5 drives each primitive; when it stalls near the target (or runs out of time), an RL teacher takes over and finishes. The target pose comes from the camera localizer.

**Why it's shorter:** a finisher only learns the last 10–25 cm, starting from states we can record (π0.5's actual failures + expert approaches). There's no curriculum from home, and each teacher converges in about one overnight run. The pattern is proven on tube picks: **17/20 with the finisher vs 0/20 for π0.5 alone.**

## Mini milestones first (~5 days for all six)

A small-scale version of every milestone, to see results across all skills before committing to the full runs. Each mini has:
- **Narrower scope:** one or two targets, start states within ~10 cm.
- **Short training:** 3–4 h instead of overnight.
- **Small check:** 5 generator trials (π0.5 + finisher vs π0.5 alone), or Isaac Lab only where the generator eval task doesn't exist yet.

| | Mini | Scope | Training | Check | Signal = |
|---|---|---|---|---|---|
| m1 | Tube pick, **perceived pose** | current teacher, unchanged; only the target source switches to the camera localizer | none | 10 generator trials with `--perceived` | ≥ 7/10 (vs 17/20 with exact pose) |
| m2 | **Tray pick** | default tray spot ± small offsets; starts from expert tray approaches + π0.5's tray failures | ~3 h | 5 generator trials | ≥ 3/5 (π0.5 alone 0/5) |
| m3 | **Pipette pick-up** | default pipette pose ± small offsets; xyz + yaw | ~3 h | 5 generator trials | ≥ 3/5 |
| m4 | **Tube place** (back into its own slot) | tube already held within 10 cm above the rack (from expert place episodes) | ~3 h | Isaac Lab 100 episodes + 5 generator trials from scripted holds | ≥ 60% |
| m5 | **Centrifuge insert**, one bore | starts within 5 cm of bore 1's entry, tube held; 6-DoF actions | ~4 h | Isaac Lab 100 episodes | ≥ 50% |
| m6 | **Attach tip**, one rack column | pipette held near the tip rack; success = nozzle seated in a tip (minimal port of the tip-attach rule) | ~3 h | Isaac Lab 100 episodes | ≥ 50% |

**Schedule:** set up the next mini while the current one trains.

```
Day 1   m1 eval (no training)        |  m2 setup -> train
Day 2   m2 check  |  m3 setup -> train
Day 3   m3 check  |  m4 setup -> train
Day 4   m4 check  |  m5 setup -> train
Day 5   m5 check  |  m6 setup -> train  ->  m6 check  ->  report + video per mini
```

**Then:** you pick which full milestones to run, based on the mini results. A mini that misses its signal tells us what the full milestone needs (more start states, residual RL, more actions) before we spend nights on it.

## Full milestones (after the minis)

Each milestone ends with an eval in our generator: **π0.5 alone vs π0.5 + finisher** on the same seeds, plus a video.

| | Milestone | Work | Duration | Done when |
|---|---|---|---|---|
| M0 | Tube pick finisher | done | — | 17/20 (fixed layout) |
| M1 | **Tube pick, hardened + perceived pose** | widen hand-over range to 25 cm (the 2 failures were 15 and 25 cm out); train with the localizer's measured noise/latency; rerun the eval with `--perceived` (camera-estimated pose) | 2 days (1 night training) | ≥ 18/20 with perceived pose |
| M2 | **Tray pick finisher** | tray asset + env variant (no tubes, faster); start states from π0.5's tray failures + expert approaches; finisher hook for `pick_up_tray` | 2 days (1 night) | ≥ 8/10 (π0.5 alone: 0/10) |
| M3 | **Pipette pick-up finisher** | pipette asset; action xyz + yaw (camera-facing grasp); finisher for `pick_up_pipette` | 2–3 days (1 night) | ≥ 8/10 (π0.5 alone: 0/10) |
| M4 | **Tube place into rack slot finisher** | new eval task (tube in hand above the rack → place in a named slot); precise 8 mm-bore insertion, residual RL on the scripted place if needed | 3 days (1–2 nights) | ≥ 8/10 |
| M5 | **Centrifuge insert finisher** | 6-DoF actions (45° bores); residual RL on the scripted insert; new eval task | 3–4 days (2 nights) | ≥ 7/10 |
| M6 | Pipette operations (attach tip, aspirate/dispense) | liquid model ported into Isaac Lab; tip/nozzle precision | 4–5 days (2–3 nights) | ≥ 7/10 each |
| opt | Lying-tube recovery | only if π0.5 keeps knocking tubes over (1 of 20 in M0) | 2–3 days | — |

**Checkpoint for you after M2 (~day 4):** M1 and M2 show perception-in-the-loop works and the pattern carries to a second object.

## Timeline (one GPU; training at night, evals in the day)

```
Week 1   M1 tube (perceived)  ->  M2 tray  ->  M3 pipette pick     [checkpoint after M2, ~day 4]
Week 2   M4 tube place  ->  M5 centrifuge insert
Week 3   M6 pipette operations  (optional: lying-tube recovery)
```

- **Picks across all three objects** (M1–M3): about 1 week.
- **Plus place and centrifuge** (M4–M5): about 2 weeks.
- **Everything incl. pipette operations**: about 3 weeks, vs 4–5 for RL from the beginning.

## Shared pieces (built as needed, not up front)

- **Generic finisher hook** in `eval_policy.py`: today tube-only; generalized in M2.
- **Perceived pose:** the localizer as the teacher's target source, from M1 on. The vision session runs its server; their measured in-hand tilt error (~3°) matters for tilt-based success checks.
- **Start-state library per skill:** π0.5's recorded failure states + expert near-target states, as for the tube teacher.
- **Reward template** with the two loophole guards we learned: success pays out the remaining episode, and every success condition (height, tilt, hold) is scored directly.

## Risks

- **Sim-to-sim gap**, Isaac Lab vs the generator. Mitigated by evaluating every finisher in the generator, as in M0.
- **Contact-heavy skills** (M4, M5) may need residual RL or an extra night.
- **π0.5 failures far from the target** (beyond 25 cm) stay unfinished: the finisher can't fix a wrong gross motion. If they're common, π0.5's data needs work, not the finisher.
- **The GPU is shared** with π0.5 fine-tunes and the localizer evals.

## Decisions for you

1. Order M1 → M2 → M3 → M4 → M5 → M6, or reprioritize (e.g. centrifuge before pipette)?
2. Checkpoint after M2 OK?
3. GPU: finisher training at night, fine-tunes and evals in the day?

## Mini milestone results (2026-10-25)

| | Mini | Isaac Lab (100 ep) | Generator | Signal |
|---|---|---|---|---|
| m1 | Tube pick, perceived pose | — | 10/10 | met |
| m2 | Tray pick | 100/100 | 3/5 (π0.5 alone 0/5); 2 misses pushed the pipette | met |
| m3 | Pipette pick-up | 90/100 | 5/5 (π0.5 alone 0/5) | met |
| m4 | Tube place | 0/100 (seats, never releases) | 5/5 with a scripted release | met with scripted release |
| m5 | Centrifuge insert, bore 2 | 83/100 | — | met |
| m6 | Attach tip | 66/100 | — | met |

## Full milestones: execution (started 2026-10-26, ~48 GPU h, about a week)

| | Milestone | Work | GPU h |
|---|---|---|---|
| M1 | Tube pick, perceived | hand-over from up to 25 cm; all 20 slots; train with the localizer's measured noise and 1-step latency; 20 generator trials, perceived pose | 6.7 |
| M2 | Tray pick | neighbours (pipette, tip rack, waste bin) in the env (rl/neighbours.py), disturbing one = failure; 10 trials | 7.1 |
| M3 | Pipette pick-up | same neighbours; 10 trials | 6.2 |
| M4 | Tube place | ~100 expert place episodes over all 20 slots; release learned or scripted; place hook + eval task; 10 trials | 8.5 |
| M5 | Centrifuge insert | inserts recorded at all 6 bores; one teacher; insert hook + eval task; 10 trials | 12.1 |
| M6 | Pipette operations | tip attach (all columns), aspirate, dispense teachers (rule-based, kinematic); hooks; 10 trials each | 8.0 |

Order: M2, M3, M1 (picks, ~20 GPU h, checkpoint), then M4, M5, then M6. Trainings run through a job folder
(data/rl_jobs/*.sh, one at a time); each reward change gets a 50-iteration pilot check before a full run.
