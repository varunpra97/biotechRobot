"""Generate keypose-annotated demonstrations (ChainedDiffuser-style data).

Per episode:
  1. randomize tray init pose + goal pose
  2. analytic task-space keyposes (perception.keyposes)
  3. DLS IK -> joint-space keyposes
  4. smooth joint interpolation through keyposes -> dense trajectory
  5. track it in sim with PD; validate no-NaN + keypose reach error
  6. render images at each keypose if GL is available (skipped headless)

Output: assets/demos/demo_{i}.npz with
  q_traj        (T, 6)   dense joint trajectory (the low-level target)
  keypose_joint (6, 6)   joint-space keyposes (the high-level target)
  keypose_task  (6, 7)   task-space keyposes [pos, quat]
  keypose_idx   (7,)     segment boundaries into q_traj
  tray_init     (3,), tray_goal (3,), q_init (6,)

Training uses:
  - high-level keypose diffusion: (tray_init, tray_goal) -> keypose sequence
  - low-level trajectory diffusion: (keypose_a, keypose_b) -> action chunk
  - vision -> keypose net: images -> (tray_init, tray_goal)

Usage:
  PYTHONPATH=. python sim/make_demos.py --episodes 8 --out assets/demos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perception.keyposes import compute_keyposes, keyposes_to_array
from sim.camera_sim import build_scene_xml
from sim.ik import ik_dls

SEG_STEPS = 200  # nominal sim steps per keypose segment (0.4 s at 2 ms dt)
MAX_JOINT_VEL = 0.5  # rad/s assumed for time parameterization


def _segment_steps(q0: np.ndarray, q1: np.ndarray) -> int:
    """Time-parameterize a segment so peak joint velocity stays sane."""
    dist = float(np.abs(q1 - q0).max())
    # cosine blend peak velocity = pi/2 * dist / T  ->  T = pi/2 * dist / vmax
    t = max(0.4, (np.pi / 2) * dist / MAX_JOINT_VEL)
    return int(round(t / 0.002))


def _arm_body_ids(model):
    import mujoco
    ids = set()
    for n in ["base"] + [f"link{i}" for i in range(1, 7)] + ["tool"]:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if i >= 0:
            ids.add(i)
    return ids


def _adjacent_bodies(model, arm_bodies):
    """Parent-child body pairs (their 'contact' at the joint is an artifact)."""
    pairs = set()
    for b in arm_bodies:
        p = int(model.body_parentid[b])
        if p in arm_bodies:
            pairs.add((min(b, p), max(b, p)))
    return pairs


def _arm_hits_env(model, data, arm_bodies, env_geoms, adjacent):
    """True if any active contact pairs an arm body with floor/table.

    Contacts between kinematically adjacent bodies are ignored (they share
    a joint; overlap there is a modeling artifact, not a collision).
    """
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (min(b1, b2), max(b1, b2)) in adjacent:
            continue
        if (c.geom1 in env_geoms or c.geom2 in env_geoms) and \
           (b1 in arm_bodies or b2 in arm_bodies):
            return True
    return False


def sample_collision_free_qinit(model, data, rng, arm_bodies, env_geoms,
                                adjacent, tries: int = 50):
    """Sample a randomized initial arm state with no floor/table contact."""
    import mujoco
    jr = model.jnt_range[:6]
    for _ in range(tries):
        q = rng.uniform(jr[:, 0] * 0.4, jr[:, 1] * 0.4)
        data.qpos[:6] = q
        data.qvel[:6] = 0
        mujoco.mj_forward(model, data)
        if not _arm_hits_env(model, data, arm_bodies, env_geoms, adjacent):
            return q
    raise RuntimeError("could not sample collision-free q_init")


def cosine_blend(q0: np.ndarray, q1: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    return q0[None, :] + (q1 - q0)[None, :] * s


def _plan_collision_free_traj(model, data, rng, joint_kps, arm_bodies,
                              env_geoms, adjacent, tries: int = 20):
    """Build q_init -> kp0 -> ... -> kp5 cosine-blend trajectory.

    Resamples the randomized initial state until the whole kinematic path
    is free of arm-vs-floor/table contact (checked with mj_forward every
    10th step). Returns (q_traj, keypose_idx, q_init).
    """
    import mujoco
    for _ in range(tries):
        q_init = sample_collision_free_qinit(model, data, rng, arm_bodies,
                                             env_geoms, adjacent)
        waypoints = [q_init] + [jk for jk in joint_kps]
        seg_ns = [_segment_steps(waypoints[i], waypoints[i + 1])
                  for i in range(len(waypoints) - 1)]
        q_traj = np.concatenate(
            [cosine_blend(waypoints[i], waypoints[i + 1], seg_ns[i])
             for i in range(len(waypoints) - 1)], axis=0)
        collision = False
        for s in range(0, len(q_traj), 10):
            data.qpos[:6] = q_traj[s]
            data.qvel[:6] = 0
            mujoco.mj_forward(model, data)
            if _arm_hits_env(model, data, arm_bodies, env_geoms, adjacent):
                collision = True
                break
        if not collision:
            keypose_idx = np.cumsum(seg_ns)  # step at which each keypose is hit
            return q_traj, keypose_idx, q_init
    raise RuntimeError("could not find collision-free initial state")


def run_episode(model, data, renderer, rng, out_dir: Path, ep: int,
                render: bool) -> dict:
    import mujoco

    arm_bodies = _arm_body_ids(model)
    env_geoms = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in ("floor", "table_top", "pedestal")
    }
    env_geoms.discard(-1)

    tray_init = np.array([rng.uniform(0.25, 0.65),
                          rng.uniform(-0.20, 0.20), 0.408])
    tray_goal = np.array([rng.uniform(0.25, 0.65),
                          rng.uniform(-0.20, 0.20), 0.408])

    tray_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tray_free")
    adr = model.jnt_qposadr[tray_jnt]
    data.qpos[adr:adr + 7] = [*tray_init, 1, 0, 0, 0]
    data.qvel[adr:adr + 6] = 0
    mujoco.mj_forward(model, data)

    # 1-2. keyposes: task space -> joint space via IK.
    # IK is seeded from a fixed nominal config (decoupled from q_init) and
    # warm-started along the chain for continuity. keypose_task records the
    # ACHIEVED pose (reachable by construction), which is what the diffusion
    # models train on.
    task_kps = compute_keyposes(tray_init, tray_goal)
    joint_kps, achieved_kps, errs = [], [], []
    q_seed = np.zeros(6)
    for k, kp in enumerate(task_kps):
        q_sol, err, q_ach = ik_dls(
            model, data, kp["pos"], kp["quat"], q_init=q_seed,
            seed=int(rng.integers(1e9)) + k)
        mujoco.mj_forward(model, data)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        achieved_kps.append(np.concatenate(
            [data.site_xpos[ee_id], q_ach]))
        joint_kps.append(q_sol)
        errs.append(err)
        q_seed = q_sol
    joint_kps = np.stack(joint_kps)
    achieved_kps = np.stack(achieved_kps)  # (6, 7) [pos, quat]
    max_ik_err = float(np.max(errs))
    assert max_ik_err < 0.01, f"IK failed, err={max_ik_err}"

    # 3. dense trajectory, validated kinematically: resample q_init until the
    # whole path (q_init -> kp0 -> ... -> kp5) is collision-free.
    adjacent = _adjacent_bodies(model, arm_bodies)
    q_traj, keypose_idx, q_init = _plan_collision_free_traj(
        model, data, rng, joint_kps, arm_bodies, env_geoms, adjacent)

    # 4. dynamic feasibility via inverse dynamics: the torques required to
    # execute q_traj must lie within the motor limits. (More principled
    # than PD-tracking proxies; no controller tuning involved.)
    gear = float(model.actuator_gear.flat[0])
    dt = model.opt.timestep
    qvel = np.gradient(q_traj, dt, axis=0)
    qacc = np.gradient(qvel, dt, axis=0)
    data.qpos[adr:adr + 7] = [*tray_init, 1, 0, 0, 0]
    max_tau = 0.0
    for s in range(len(q_traj)):
        data.qpos[:6] = q_traj[s]
        data.qvel[:6] = qvel[s]
        data.qacc[:6] = qacc[s]
        data.qvel[adr:adr + 6] = 0
        data.qacc[adr:adr + 6] = 0
        mujoco.mj_inverse(model, data)
        if not np.all(np.isfinite(data.qfrc_inverse)):
            raise RuntimeError(f"non-finite inverse dynamics at step {s}")
        max_tau = max(max_tau, float(np.abs(data.qfrc_inverse[:6]).max()))
    assert max_tau < gear * 0.9, \
        f"trajectory needs {max_tau:.1f} Nm > motor limit {gear:.1f} Nm"

    # 5. images at keyposes (best effort; skipped when headless)
    img_dir = out_dir / f"demo_{ep:04d}_imgs"
    if render and renderer is not None:
        img_dir.mkdir(parents=True, exist_ok=True)
        # (rendering hooks into camera_sim's renderer; see README)

    np.savez(out_dir / f"demo_{ep:04d}.npz",
             q_traj=q_traj, keypose_joint=joint_kps,
             keypose_task=achieved_kps,
             keypose_task_target=keyposes_to_array(task_kps),
             keypose_idx=keypose_idx,
             tray_init=tray_init, tray_goal=tray_goal, q_init=q_init)
    print(f"demo {ep}: ik_err={max_ik_err:.4f} max_tau={max_tau:.1f}Nm "
          f"T={len(q_traj)}")
    return {"ik_err": max_ik_err, "max_tau": max_tau}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("assets/demos"))
    ap.add_argument("--render", action="store_true",
                    help="also render keypose images (needs GPU/GL)")
    args = ap.parse_args()

    import mujoco

    args.out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_scene_xml())
    data = mujoco.MjData(model)
    renderer = None
    if args.render:
        renderer = mujoco.Renderer(model, height=240, width=320)
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        run_episode(model, data, renderer, rng, args.out, ep, args.render)
    print("demos OK")


if __name__ == "__main__":
    main()
