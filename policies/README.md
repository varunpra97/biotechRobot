# Policies — ChainedDiffuser-style hierarchy

Two diffusion models, trained separately, chained at inference
(following Zhou Xian's ChainedDiffuser, CoRL 2023):

```
camera images
    │  perception/train_pose_net.py  →  (tray_xyz, goal_xyz)
    ▼
perception/keyposes.py (analytic)  →  initial task-space keyposes
    │  KeyposeDiffusion  →  refined keypose sequence  (6, 7)  [pos + quat]
    ▼
TrajectoryDiffusion  →  joint-space action chunks  (T, 6)
```

## Models (`diffusion.py`, shape-tested on CPU)

- `KeyposeDiffusion`: conditional DDPM. Condition = (tray_init, tray_goal)
  12-D; output = 6 keyposes × 7-D (position + quaternion, achieved poses
  from `sim/make_demos.py`).
- `TrajectoryDiffusion`: conditional DDPM. Condition = flattened adjacent
  keypose pair (14-D: joint configs); output = joint trajectory chunk
  between them.
- Both share a 1-D FiLM-conditioned UNet (`ConditionalUNet1D`) and a
  cosine beta schedule (`diffuse()` / `denoise()` implement the DDPM
  forward/reverse equations; the `__main__` block asserts shapes and
  finite losses).

## Training data

`sim/make_demos.py` writes `.npz` episodes with:
- `q_traj` (T, 6): dense, collision-free, torque-feasible joint trajectory
- `keypose_joint` (6, 6), `keypose_task` (6, 7) achieved, `keypose_idx` (6,)
- `tray_init`, `tray_goal`, `q_init`

`policies/train_keypose.py` trains the high-level model from
`(tray_init, tray_goal) → keypose_task`.
`policies/train_traj.py` trains the low-level model from adjacent
`keypose_joint` pairs → `q_traj` chunks.

## Status

- Core DDPM math + shapes: validated.
- Training scripts: implemented and tested end-to-end on synthetic demos
  (loss decreases, checkpoints save, sample round-trip verified).
  Run on the 3080 Ti box after generating real demos:
  `PYTHONPATH=. python policies/train_keypose.py --data assets/demos`
  `PYTHONPATH=. python policies/train_traj.py --data assets/demos`
- Inference chaining + closed-loop evaluation: TODO.
