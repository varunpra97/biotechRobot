# You are the model in a robot's control loop (simulation)

A Franka arm works at a small centrifuge in a physics simulator. You control it the way general-purpose models are
run on robot benchmarks: you look at camera pictures, choose one command, the arm executes it, you look again.
The simulator is paused while you think. Nobody else is driving: every episode's outcome depends on your choices.

## The one command you use

    bash rc.sh RUN RECIPE [args]        (rc.sh: a one-line wrapper around ../loop_recipes.py <run folder> RECIPE [args])

RUN is given in your task. Each call is one model call. It prints one JSON line (the robot's own state: hand pose,
finger gap, steps left, what each move did, and `episode_over` when an episode ended) and a line `image: <path>` or
`image: none`. Open the picture with the Read tool. A call takes 5 to 60 seconds; give Bash a 300000 ms timeout.

Recipes (each one is a short batch of hand moves; you choose which, and with which bore):

- `look` : a full picture without moving. Use it once at the very start of your block.
- `info` : no picture, no motion; prints what the hand pose says about each of the three bores.
- task recipes: listed in the task section below.
- `raw '<json>'` : any other command, for recovery. Examples:
  `raw '{"op":"moves","obs":"full","moves":[{"grip":"open","wait":6},{"dtool":[0,0,-0.03]}]}'`
  (open the gripper, back the hand off 3 cm along its own axis),
  `raw '{"op":"moves","obs":"full","moves":[{"dpos":[0,0,0.03]}]}'` (3 cm straight up).
  A move can have `"pos"` (absolute, metres, robot base frame), `"dpos"` (offset in the base frame, z is up),
  `"dtool"` (offset in the hand frame, +z is out of the fingers), `"z_axis"` (direction the fingers point),
  `"grip": "open"|"close"`, `"wait": steps`, `"speed"` (m per step, default 0.006), `"max_steps"`.
  `"obs"` is `"full"`, `"crop"`, `"one"` or `"none"`.

## The pictures

- Full picture (1280 x 840): top row = two wide views (left: side camera A, right: camera B looking down into the
  rotor). Bottom row = a zoomed crop of each view, centred on the hand.
- Cyan cross = the point between the finger pads (TCP). Cyan line = the hand's axis, 5 cm long, pointing out of the
  fingers. They are drawn from the robot's own joint readings, so they are exact.
- Orange rings numbered 0, 1, 2 = where the open top end (rim) of a tube sits when it is fully seated in bore 0, 1, 2.
  These come from notes measured in earlier episodes. The centrifuge does not move.
- The test tube is a pale, slightly blue, translucent cylinder about 6 cm long with a pointed (conical) lower end.
- Yellow numbers on the edges are pixel scales; you do not need them.

## The JSON line

- `gripper_commanded`: "closed" or "open". `finger_gap_mm`: about 24 when open, about 4 to 7 when holding a tube,
  under 2 when closed on nothing.
- `steps_left`: each episode has 180 control steps; only motion uses them.
- `nearest_bore`: the bore (0, 1, 2) whose axis is closest to the hand's axis, with the hand's height and offset
  relative to that bore. Recipes use this bore unless you give another.
- `moves`: one entry per move in the batch: "ok" or "NOT ARRIVED"; "REQUIRE FAILED" means a finger-gap check failed
  (for example the fingers closed on nothing). `stopped` says why a batch ended early.
- `episode_over`: {"outcome": "SUCCESS" | "FAIL" | "TIMEOUT", ...}. When it appears, the picture and state in that
  reply already show the START of the next episode.

## Hard rules

- Use only `rc.sh` and the Read tool on the `obs_*.png` pictures it names. You may `cat` or `wc -l` the file
  `episodes.jsonl` in the run folder (one line per finished episode) to count progress.
- Never open `private.jsonl` (it holds simulator truth that the model must not see). Never start, stop or restart
  anything, never run other programs, never edit or create files, never touch git or `data/rl_jobs`.
- Do not send `{"op":"quit"}`.
- Never report an outcome you did not see in an `episode_over`. If something goes wrong that you cannot fix (no
  reply, repeated errors), stop and say exactly what happened.
- Work through the episodes one after another without asking questions. Stop when your block is done.

## What to return

A short plain report: how many episodes you ran, how many SUCCESS / FAIL / TIMEOUT (from `episode_over`), and for
every episode that was not a plain success-by-the-recipe: the episode number, what you saw, what you did. Also say
how often you had to correct or recover, and anything about the pictures or recipes that was confusing or wrong.
