"""IK tracking check: from home, command a constant TCP move per step and
measure the actual TCP displacement per step and the orientation drift."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tube_grasp.env import TubeGraspEnv  # noqa: E402
from tube_grasp.env_cfg import TubeGraspEnvCfg  # noqa: E402

cfg = TubeGraspEnvCfg()
cfg.scene.num_envs = 2
cfg.library_p = 0.0                      # start at home
env = TubeGraspEnv(cfg)
env.reset()
from isaaclab.utils.math import quat_apply, skew_symmetric_matrix  # noqa: E402
env.step(torch.zeros(2, 4, device=env.device))
env.step(torch.zeros(2, 4, device=env.device))
tcp, hq = env._tcp()
J = env._robot.data.body_link_jacobian_w.torch[:, env.jac_idx, :, env.jac_cols].clone()
print("jacobian shape", tuple(env._robot.data.body_link_jacobian_w.torch.shape), "jac_idx", env.jac_idx,
      "hand_idx", env.hand_idx, "fixed", env._robot.is_fixed_base, "arm_ids", env.arm_ids, "bodies", env._robot.body_names)
# finite-difference check of the TCP jacobian: move joint i by +0.01 (teleport), compare
q0 = env._robot.data.joint_pos.torch.clone()
r = quat_apply(hq, env.tcp_local)
Jt = J.clone()
Jt[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
for i in range(7):
    q = q0.clone()
    q[:, env.arm_ids[i]] += 0.01
    env._robot.write_joint_position_to_sim_index(position=q)
    env.sim.forward() if hasattr(env.sim, "forward") else None
    env.scene.update(0.0)
    tcp2, _ = env._tcp()
    fd = (tcp2 - tcp)[0] / 0.01
    print(f"joint {i + 1}: FD dTCP/dq {fd.cpu().numpy().round(3)}  J {Jt[0, :3, i].cpu().numpy().round(3)}", flush=True)
    env._robot.write_joint_position_to_sim_index(position=q0)
env.sim.forward() if hasattr(env.sim, "forward") else None
env.scene.update(0.0)
for direction, a in (("+x", [1, 0, 0]), ("-y", [0, -1, 0]), ("-z", [0, 0, -1])):
    prev, q0 = None, None
    for t in range(12):
        act = torch.tensor([[*a, -1.0]] * 2, device=env.device, dtype=torch.float32)
        env.step(act)
        tcp, hq = env._tcp()
        if q0 is None:
            q0 = hq.clone()
        if prev is not None:
            d = (tcp - prev)[0] * 1000
            dot = (hq[0] * q0[0]).sum().abs().clamp(max=1)
            print(f"{direction} t={t:2d} TCP step (mm) {d[0]:+6.2f} {d[1]:+6.2f} {d[2]:+6.2f}  "
                  f"rot drift {torch.rad2deg(2 * torch.acos(dot)):.2f} deg  "
                  f"q_goal-q_now max {float((env.q_goal - env._robot.data.joint_pos.torch[:, env.arm_ids]).abs().max()):.4f}",
                  flush=True)
        prev = tcp.clone()
env.close()
app.close()
