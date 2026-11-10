# Centrifuge insert and remove without PPO

An experiment (2026-11-09): instead of more PPO training, have the model (Claude Fable 5.2) solve the two centrifuge
skills directly. The model wrote a closed-loop controller for each from the environment's state, ran it, read the
failure traces and revised it over about ten rounds. Nothing here is trained with reinforcement learning.

Environments: `tube_insert_joint` and `tube_remove_joint`, unchanged (real centrifuge, same resets, same success /
fail / timeout tests, same action: 7 joint-target changes x `joint_step_rad` + the gripper).

## Files

| File | What it is |
|---|---|
| `controllers.py` | `RemoveController`, `InsertController`: state in, joint action out |
| `run_eval.py` | one episode per env, like `eval_skill.py`; runs the controller, a distilled network (`--policy`) or a PPO checkpoint (`--checkpoint`) on the same starts (`--seed`); outcome by start kind, per-step traces (`--trace`) |
| `distill.py`, `net.py` | supervised imitation (DAgger) of a controller into a network: observation in, joint action out, no Jacobian at run time |
| `record.py` | headless clips (after the scratchpad `record_skill.py`) |
| `loop_server.py`, `loop_client.py`, `loop_tri.py` | the model in the control loop: one pose target per model call from camera images |

Results: `data/eval/fable_centrifuge/results.jsonl` (one line per run).

## Is this IK-free? No, not the controllers

- **Fine motion uses the arm's Jacobian.** Each step turns a small Cartesian correction (of the TCP, or of the held
  tube's tip) into joint-target changes with a damped least-squares solve: differential IK. The commands that leave
  the controller are joint-target changes, as for the RL policies, but the mapping is IK-like.
- **The insert's gross move does not.** Far starts go in a straight line in joint space to one recorded joint
  configuration per bore (a "taught" transit pose from the env's own library of expert holds), no Jacobian.
- **The distilled network does not.** `distill.py` trains a network on the controller's actions; at run time it maps
  the env's observation to joint actions, like a PPO policy. Its teacher used the Jacobian; it does not.

## What the controllers do

Remove: onto the tube's axis 3.5 cm above the grasp point (hand z down the axis, closing axis along the rotor's
tangent), down the axis, close, out along the bore axis until the tip is 4 mm past the clear point, then up while
turning the tube's measured axis to vertical. A hold that closed off the grasp point is opened and regrasped while
the tube is still in its bore.

Insert: (far starts: joint-space move to the taught transit pose) turn the tube's measured axis onto the bore's
while bringing its tip down a funnel around the axis to 8 mm above the opening; twist the hand so the fingers will
open along the rotor's tangent; push down the axis, slowing towards the release depth; release when the tube is
seated and coaxial; back the hand off along the axis. A push that misses, jams or ends cocked is pulled back above
the opening and repeated.

## What the failure traces showed (in the order found)

1. Insert, first run 44/128: every miss was lined up above the bore and never pushed. The "tube at rest" test read
   the tube's velocity, which jitters at several cm/s inside the pads. Replaced by "lined up for two steps": 121/128.
2. Fingers that would not open after seating (15 of 512): the push overshot the seat, the arm kept pressing and
   pinned the fingers on the rotor. Push speed now falls to zero at the release depth: 0 of 512.
3. A tip below the opening's plane but beside the rotor was taken for "inside the bore" and released there.
4. Far starts timed out: turning the tube in Cartesian space drove joints 5 and 7 through about twice the joint
   travel of a straight joint-space move (150+ of 180 steps). A joint-space line straight to the pre-insert pose ran
   the tube into the centrifuge (12 of 20); via the recorded transit pose 6-8 cm above it: 20 of 20.
5. Three far starts with the hand 90 degrees off picked the wrong tangent sign and twisted joint 7 into its limit.
6. Remove: the turn-upright is slow (about 0.5 degrees per step, joint 6 at its rate limit): a median 119 of the 180
   steps. Letting the other joints take over when one saturates halved it but the hand sank and dropped tubes
   (362 of 512 failed); reverted.
7. Remove, held starts whose tube was knocked 6 mm along the hand as the pads closed: regrasp.

## Results (2026-11-09)

Same starts for every solver (same seed, one episode per env). Seeds 0 and 3 were not used while writing the
controllers (1 and 2 were); the networks were distilled on seed 11. PPO checkpoints: insert run 10 `model_7694.pt`,
remove run 4 `model_4498.pt`. Successes out of the episodes run:

| Task | Episodes | Controller (Jacobian) | Distilled network (no Jacobian) | PPO checkpoint |
|---|---|---|---|---|
| Insert | 100 (seed 0) | 99 | 99 | 46 |
| Insert | 512 (seed 3) | 512 | 509 | 202 |
| Remove | 100 (seed 0) | 100 | 94 | 97 |
| Remove | 512 (seed 3) | 512 | 489 | 471 |

Insert by start kind (successes / episodes):

| Start | Episodes run | Controller | Distilled network | PPO checkpoint |
|---|---|---|---|---|
| Tube already in the bore | 100 | 29 / 30 | 29 / 30 | 26 / 30 |
| Carry 2-6 cm | 100 | 41 / 41 | 41 / 41 | 13 / 42 |
| Carry 6-15 cm | 100 | 19 / 19 | 19 / 19 | 4 / 19 |
| Carry over 15 cm | 100 | 10 / 10 | 10 / 10 | 3 / 9 |
| Tube already in the bore | 512 | 136 / 136 | 135 / 136 | 113 / 133 |
| Carry 2-6 cm | 512 | 267 / 267 | 266 / 267 | 64 / 273 |
| Carry 6-15 cm | 512 | 75 / 75 | 75 / 75 | 21 / 74 |
| Carry over 15 cm | 512 | 34 / 34 | 33 / 34 | 4 / 32 |

(The start kind is read after the first step, so a few starts fall in neighbouring rows for different solvers. The
one insert miss in the 100 is the same start for the controller and the network: the tube springs out of the pads
in the first 5 steps, before either can act.)

Remove by start kind, 512 episodes: held 129 / 129 controller, 121 / 129 network, 121 / 129 PPO; approach under
5 cm 228 / 228, 219 / 228, 209 / 229; approach 5-15 cm 155 / 155, 149 / 155, 141 / 154.

On the tuning seed (2) the controllers scored 510 / 512 each. The misses were start states that are already
failing (the tube wedged against the rotor or springing out of the pads at reset) and one tube thrown out of its
bore as the fingers opened.

### The distilled networks

`distill.py`, 1024 envs, 10 rounds of 220 steps (2.25 million labelled steps), about 28 minutes per task on the
3080 Ti including the simulation. Networks: `data/eval/fable_centrifuge/{insert_joint,remove_joint}_distilled.pt`
(TorchScript: observation in, 8 actions out), per-round log next to them. The network's own success rate while it
drove, by round:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Insert | 23% | 19% | 71% | 93% | 98% | 98% | 98% | 99% | 99% |
| Remove | 20% | 45% | 83% | 91% | 90% | 92% | 96% | 97% | 96% |

Plain imitation of the controller's own episodes (round 1) scores about 20%: the corrections after the network
drifts off the controller's path are what make it work. The remove network is 4-6 points below its teacher and had
stopped improving by round 8; its misses are tubes dropped or knocked over between closing and the lift (not looked
into further). The insert network is within 1 point of its teacher.

## The model in the control loop (added later the same day)

How general-purpose models are run on the RoboDojo benchmark (published harnesses: one model call = look at camera
images, send one end-effector pose target, IK and a path planner underneath, about 100 calls per episode). The same
here: `loop_server.py` holds one env and waits; each call the model (Claude Fable 5.2, this session) gets two fixed
camera views (wide + a crop), camera calibration, the hand pose in the robot base frame, the finger gap and the steps
left, and sends one pose target for the hand (or for a point fixed to it) plus the gripper. `loop_client.py` sends a
command, `loop_tri.py` triangulates a point picked in both views. No object poses, no depth, no bore coordinates.

- **IK-like underneath**: the pose target is turned into joint actions by the same Jacobian servo as the controllers.
- **The simulator is paused while the model thinks.** Only motion uses the 180 steps. Measured: a median 7.5 s per
  call (mean 11 s, longest 42 s), 65 to 95 s of wall-clock per episode against the 12 s the env allows.
- Env rules unchanged (resets, success / fail tests, 180 steps).

Results, 15 counted episodes (`data/eval/fable_centrifuge/in_the_loop.jsonl`, video
`data/eval/videos/fable_centrifuge_in_the_loop.mp4`), plus 1 practice removal used to set up the cameras (also a
success):

| Task | Start | Episodes | Success | Steps used (of 180) | Move commands |
|---|---|---|---|---|---|
| Remove | tube already held | 4 | 4 | 78 - 110 | 2 |
| Remove | approach, hand 2.7 cm away | 1 | 1 | 148 | 5 |
| Remove | approach, hand 11 cm away | 1 | 1 | 158 | 5 |
| Insert | tube already in the bore | 2 | 2 | 31 - 34 | 3 |
| Insert | carry 2-6 cm | 5 | 5 | 45 - 55 | 4 |
| Insert | carry 6-15 cm | 2 | 2 | 77 - 85 | 5 |

Not a 100-episode evaluation: with 15 of 15 the true success rate could still be as low as about 80%.

What the model did. Remove: triangulate the tube rim and a second point on the tube from the two views, check them
by having them drawn back on the images, move onto that axis, down, close, out along the axis, lift while turning
upright. Insert: put the tube tip on the bore axis just above the opening with the hand turned onto the axis, check
that the hand axis drawn on both images runs into the bore, push 34 mm, open, back off.

What the simulator's own numbers say afterwards (never shown to the model during the episodes):

- Remove: the gripper closed 0.8 and 0.95 mm from the ideal grasp point.
- Insert: before the push the tip was 1.3 - 2.0 mm off the bore axis and 1.6 - 3.2 degrees off. The hand-written
  controller does not push until it is within 0.6 mm and 1.5 degrees. The tubes went in anyway (conical tip), and
  ended 0.3 - 1.0 mm off axis (limit 1.5 mm) and 1.0 - 2.4 degrees (limit 5). The margin is thin.

What made it easy, stated plainly:

- The centrifuge does not move between episodes. The bore axes were measured once from the images (first removal)
  and reused; the other two bores follow from the rotor's 60-degree symmetry. The images were then used to confirm
  the alignment, not to find it again. A moved centrifuge would need the measuring step every time (about 2 extra
  calls).
- The model had written the controllers above first, so it knew the tube length, the grasp depth and how deep the
  seat is. It also saw the three nominal bore points in world coordinates while writing the server (not used: the
  model works in the robot base frame).
- For insert, the target bore is circled in one wide image at the start of each episode (the instruction).
- Not covered: carry starts over 15 cm, starts with the hand turned 90 degrees, any failure recovery (none was
  needed).

### Version 2 of the loop: fewer and lighter model calls

Aim: cut the time the arm stands still between moves (asked for with a real robot in mind).

- **Several moves in one model call**: `{"op": "moves", "moves": [...]}` runs them in order and stops early when a
  move does not arrive, a `"require_gap_mm": [lo, hi]` on a move is not met (e.g. after closing: the fingers hold
  something), or the episode ends. One observation comes back at the end, plus a result per move.
- **Lighter observations**: `"obs": "crop"` (the two crops), `"one"` (one crop), `"none"` (no picture). A new episode
  always starts with the full two-view picture. Pictures stay 1280 x 720 per camera.
- **Counts per episode** in `episodes.jsonl` next to the run, and for all runs in
  `data/eval/fable_centrifuge/in_the_loop_calls.jsonl`: model calls with motion, look-only calls, moves, env steps,
  seconds the arm waited for the model.

All 15 counted episodes above and the practice removal ran on version 1. Version 2 was then run on the same seeds:

| Start | Version | Result | Env steps | Calls with motion | Look-only calls | Moves | Arm waiting for the model |
|---|---|---|---|---|---|---|---|
| Remove, practice start (hand 2.7 cm away) | 1 | success | 146 | 5 | 2 (+3 camera set-up) | 5 | 63 s |
| same start | 2 | success | 146 | 1 | 1 | 5 | 8 s |
| Remove, hand 7-11 cm away | 1 | success | 158 | 5 | 1 | 5 | 36 s |
| similar start | 2 | success | 157 | 2 | 0 | 5 | 17 s |
| Insert, tube already in the bore | 1 | success | 31 | 3 | 1 | 3 | 36 s |
| same start | 2 | success | 31 | 1 | 1 | 3 | 6 s |
| Insert, tube above the bore mouth | 1 | success | 55 | 4 | 0 | 4 | 44 s |
| same start | 2 | success | 55 | 2 | 0 | 4 | 14 s |
| Insert, third start of that seed | 1 | success | 49 | 4 | 0 | 4 | 42 s |
| same seed, tube sat differently in the hand | 2, all in one call | not ruled; tube left cocked, not a success | 42 | 1 | 0 | 4 | - |
| same again, with checks | 2 | FAIL | 109 | 6 | 0 | 10 | 155 s |

Env steps do not change (they are motion). Model calls drop from 3-7 to 1-2 per episode and the waiting time to
about a third or less. (Two more version-2 successes on the rerun are in the file; their waiting times are inflated
by the model working on the failure in between.)

The failure, plainly. On the third insert start the tube sat 4.2 degrees and 3.6 mm off in the fingers (2.2 degrees
and 1.5 mm in the version-1 run of the same seed). The model lines up the HAND with the bore and assumes the tube
is straight in it, so the tip was 3.6 mm off the bore axis, jammed on the bore mouth, and the hand slid 28 mm down
the tube.

- All in one call: nothing noticed it. The finger-gap requirement passed, the gripper opened and left the tube
  cocked 16 degrees in the mouth. The server was stopped before the env ruled on it (operator error), so it is
  listed as not ruled; it would not have been a success.
- With a light picture after the push: the jam was seen before the release. The recovery (pull back, find the tip by
  triangulation, set it 12 mm into the bore at 1.0 degree / 1.6 mm, let go, grip higher, push) lost the tube when it
  leaned 10 degrees while standing loose: FAIL.
- The check picture after lining up did show the tube off the hand axis; the model read it as fine.

What this says for batching: moves whose success the robot can sense itself (arrived, fingers closed on something)
can go in one call. An insert cannot be fully batched on finger gap alone: it needs one picture after the push (or
a force signal), and the alignment has to be done on the tube tip seen in the pictures, not on the hand.
With both runs together the in-the-loop insert record is 13 successes in 15 ruled episodes plus the one not ruled.

## Limits

- The controllers read exact simulator state (tube pose, bore axis) and the simulator's Jacobian. The RL policies read
  the same state through their observation.
- The taught transit poses are joint configurations for bores 1-3 at the centrifuge's default spot. A moved
  centrifuge needs them re-recorded (or the slower Cartesian carry: `--param gross=0`).
- Remove takes a median 119 steps of 180; a regrasp on top of a far approach could run out of time.
- Not tested in the data generator with pi0.5 and the sphere hand-over.
