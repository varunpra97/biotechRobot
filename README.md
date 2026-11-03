# Biotech Lab Robotics

A Franka Panda working a biotech lab bench (test tubes, centrifuge,
electronic pipette, sample tray), simulated physically in Isaac Sim. The
robot runs lab primitives from plain-language instructions: Physical
Intelligence's π0.5 VLA handles the instruction and the gross motion, and
reinforcement-learning teachers trained in Isaac Lab take over near the
target for the precise part. A separate pipeline measures a real lab scene
into sim assets.

| Directory | What it is |
|---|---|
| [`simulation/`](simulation/README.md) | Isaac Sim lab scene, scripted expert and randomized episode generator, LeRobot export and π0.5 fine-tuning (`sim/`, `training/`), RL teachers in Isaac Lab (`rl/`), evaluation and multi-step scenarios |
| [`vision_pipeline/`](vision_pipeline/README.md) | Real scene → measured 3D CAD: calibrated multi-view capture to meshes, labels and placements for the sim |

The Python environments (`.venv-isaac`, `.venv-isaaclab`, `.venv-cad`) and
generated data (`data/`) live in the workspace folder above this repo; CAD
generation and measurement scripts are in `../cad_source/`.

## Status (2 Nov 2026)

All work so far is in simulation (Isaac Sim 6.0.1 / Isaac Lab 3.0) on one
12 GB GPU.

### Approach

```
instruction + 3 cameras → π0.5 → hand-over → RL teacher → scripted finish → success test
```

- **π0.5:** `pi05_droid` fine-tuned with LoRA via openpi (52M trainable
  parameters, vision encoder frozen to fit 12 GB), action normalization from
  our own data, 4 runs of up to 15k steps.
- **RL teachers:** PPO (rsl_rl) in Isaac Lab with 512–1,024 parallel envs and
  small MLPs on object pose (camera-estimated for tubes). Each action is a
  hand move of up to 1 cm plus open/close, applied through IK. Episodes start
  from recorded expert and π0.5-failure states. When π0.5 stalls near the
  target, closes the gripper, is about to release, or runs out of time, the
  teacher takes over.
- **Rewards:** a shared template:
  - reach at 10 cm / 2 cm / 5 mm scales, plus progress;
  - grasp, and lift scaled by how upright the object stays;
  - a success bonus that pays out the rest of the episode;
  - failures: knocked over, pushed, dropped, or a neighbour moved.

  Nine loopholes were found and closed: holding without finishing, tilted
  lifts, dropping, hovering, stalling 2 cm short, and jamming in the bore.

### Training data

- **766** physically validated expert episodes, **145,831** frames at 15 Hz,
  from a scripted expert with a collision-checked planner. That is about a
  fifth of the original 3,000–4,000-episode target.
- **Randomization:**
  - scene contents: appearance, empty slots, liquids;
  - placement: object placement and whole-desk layouts that also move the arm base;
  - episodes: varied start poses and chained tasks;
  - recovery episodes from π0.5's failures;
  - perception noise during RL training.

### Results on the six tasks

Success in the episode generator (closed loop, fixed layout):

| Task | π0.5 alone | π0.5 + RL teacher | Target |
|---|---|---|---|
| Pick tube from rack (camera-estimated pose) | 0 | **20/20** | met |
| Pick up sample tray | 0/5 | **10/10** | met |
| Pick up pipette | 0/5 | **10/10** | met |
| Place tube into rack slot | – | 7/10 | close (≥ 8/10) |
| Insert tube into centrifuge bore | – | 0/9 | not yet |
| Attach tip / aspirate / dispense | – | 0/10 each | not yet |

### What we learned

- π0.5 learns smooth motion but not *which* target the prompt names, nor
  millimetre precision.
- The RL teachers do the precise contact work well once the arm is near the
  target. In an RL-only run (no π0.5) they completed 7 of 9 steps, including
  every step that started near its target.
- The pipette tasks fail because π0.5 works the plunger wrongly before
  handing over. The centrifuge insert fails because the hand-over happens too
  far from the centrifuge.

### Next

- Hold the plunger during π0.5's phase, retrain tip attach, and split the
  insert into a carry teacher and an insert teacher.
- Retrain placing on tilted holds, and make every rack tube a physics body in
  the evals.
- Test a general model (for example GPT-6 Astra) as the planner over the RL
  skills, on the same evals.
