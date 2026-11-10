# fable_lab

Work from 2026-11-09 on using Claude Fable 5.2 for the lab robot simulation, without reinforcement learning:
hand-written controllers for the centrifuge skills, a model-in-the-control-loop harness, and a localhost tool that
turns an experiment description into robot steps in the full lab scene. Simulation only.

| Folder | What it is |
|---|---|
| `centrifuge/` | Tube insert / remove at the centrifuge (Isaac Lab envs `tube_insert_joint`, `tube_remove_joint`): closed-loop controllers written by the model, networks distilled from them, and the harness that puts the model itself in the control loop. Details and all numbers: `centrifuge/README.md`. |
| `lab_tool/` | Describe an experiment in words -> plan -> the simulated robot performs it in the full lab scene (`simulation/sim`), with a web page on localhost. |
| `results/` | The result files the numbers below come from (copies; the originals and the videos are under `data/eval/`, which is not in the repository). |

Both parts import the existing code in `simulation/` (the skill envs in `simulation/rl`, the full scene and scripted
experts in `simulation/sim`); nothing there was changed.

## centrifuge: what was measured

Same starts for every solver in a row (same seed, one episode per env).

| Task | Episodes | Controller (uses the Jacobian) | Distilled network (no Jacobian at run time) | PPO checkpoint |
|---|---|---|---|---|
| Insert | 100 | 99 | 99 | 46 |
| Insert | 512 | 512 | 509 | 202 |
| Remove | 100 | 100 | 94 | 97 |
| Remove | 512 | 512 | 489 | 471 |

The model in the control loop (it looks at two camera pictures and sends hand-pose targets; a Jacobian servo underneath
turns them into joint commands; the simulator is paused while the model thinks):

| Run | Result |
|---|---|
| Version 1, one move per model call | remove 6 of 6, insert 9 of 9 |
| Version 2, several moves per call, on the same seeds | remove 2 of 2; insert 4 successes, 1 failure, 1 episode not ruled (tube left cocked) |
| Remove block with recipes (`loop_recipes.py`), 25 episodes in a row | 25 of 25, one model call each |
| Check of the tube-tip alignment (`insert_fix`) | 2 episodes: brought a 4.0 mm / 5.2 degree error to 1.0 mm / 1.1 degrees; 1 failure from a push-direction bug (fixed), then 1 success |

Not yet run: a 15-20 episode insert block with the fixed procedure, and anything for the four other primitives the
loop server now accepts (`tube_joint`, `tray_joint`, `tray_place_joint`, `pipette_joint`): their cameras and recipes
have not been worked out.

Honest limits: every approach here reads exact simulator state or simulator cameras; the controllers and the
in-the-loop servo are IK-like (only the distilled networks are not); the in-the-loop notes assume the centrifuge does
not move; in real time the arm would stand still for about 10 s per model call.

## lab_tool: what it does and where it stands

    .venv-isaac\Scripts\python.exe biotech-lab-robotics\fable_lab\lab_tool\app.py      then open http://127.0.0.1:8765

- `protocol.py`: rule-based planner (keyword patterns, no language model). Text -> commands (transfer, centrifuge,
  wait, move tray) -> primitive steps, with the tracked liquid volumes checked. It says what it did not understand.
- `run_protocol.py`: executes the steps in ONE continuous simulation of the full lab scene. Each step is done by the
  existing scripted expert for that primitive (motion planning + IK). Liquids are tracked volumes, the centrifuge
  does not spin, waits are shortened, a liquid-filled tube stands in for "solvent A".
- `app.py`: the page (plan, run, stop, live picture, per-step clips, liquid table, final video). One run at a time.
- `protocol_video.py`: the labelled video of a run. `share_proxy.py`: a password gate for sharing the page through a
  tunnel (added by another session; not started by default).

Status when this was committed: the example "load solvent A into the test tube, centrifuge it, remove the solvent
with the pipette, put it in the tray" plans to 17 steps. The first full run did 10 of the 17 in one simulation
(pipette, tip, 200 uL of solvent A into the test tube, tip ejected, pipette returned, tube picked, inserted into the
centrifuge, waited, removed) and then failed putting the tube back into the rack (the tube would not settle over the
slot within the expert's tolerance). A retry for such abandoned steps was added; the run testing it had not finished.
No run has completed all 17 steps yet.

## Files worth knowing

- `centrifuge/controllers.py`, `run_eval.py`, `distill.py`, `net.py`, `record.py`: controllers, evaluation, distillation.
- `centrifuge/loop_server.py`, `loop_client.py`, `loop_tri.py`, `loop_recipes.py`: the model-in-the-loop harness.
- `centrifuge/agent_briefs/`: the written instructions given to the helper agents that drove blocks of episodes.
- `centrifuge/video_tools/`: scripts that cut the videos and tabulate calls per episode (they take paths as arguments).
