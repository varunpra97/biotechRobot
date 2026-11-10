"""The distilled policy: the env's observation -> the joint action (7 joint-target changes in [-1, 1], gripper +1
closed / -1 open). No Jacobian, no IK. TorchScript-exportable (distill.py saves it scripted).

Before the MLP, fixed geometric features are computed from the observation at millimetre scale; they hold nothing
the observation does not (see distill.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 0.046 below = tube_grasp TUBE_LEN - GRASP_DEPTH: the tip to the insert env's grasp point


class PolicyNet(nn.Module):
    def __init__(self, task: str, obs_dim: int, hidden: int = 512):
        super().__init__()
        self.insert = task == "insert_joint"
        self.obs_dim = obs_dim
        feat = 19 if self.insert else 23
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))
        self.mlp = nn.Sequential(nn.Linear(obs_dim + feat, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 256), nn.ELU(), nn.Linear(256, 8))

    @torch.jit.ignore
    def set_norm(self, obs: torch.Tensor):
        self.mean.copy_(obs.mean(0))
        self.std.copy_(obs.std(0).clamp(min=1e-3))

    def features(self, o: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(o[:, 0:3])
        z[:, 2] = 1.0
        if self.insert:
            # obs: tip - release point x10 (0:3), depth x10 (3), radial x100 (4), angle / 30 (5), TCP - grasp x10 (6:9),
            # tube velocity (9:12), fingers (12:14), gripper (14), joints (15:22), joint velocities (22:29), last
            # action (29:37), tube axis (37:40), bore axis (40:43), hand z (43:46), hand y (46:49)
            rel, axis, baxis, hy = o[:, 0:3] * 0.1, o[:, 37:40], o[:, 40:43], o[:, 46:49]
            s = (rel * baxis).sum(-1, keepdim=True)
            rad = rel - s * baxis                                           # the tip's offset from the bore axis, m
            top = rad + 0.046 * (axis - (axis * baxis).sum(-1, keepdim=True) * baxis)   # ... the grasp point's
            tang = torch.cross(baxis, z, dim=-1)
            tang = tang / (tang * tang).sum(-1, keepdim=True).sqrt().clamp(min=1e-6)
            twist = (torch.cross(hy, tang, dim=-1) * baxis).sum(-1, keepdim=True)
            return torch.cat([(rad * 400.0).clamp(-3.0, 3.0), (rad * 40.0).clamp(-3.0, 3.0),
                              (top * 400.0).clamp(-3.0, 3.0), torch.cross(axis, baxis, dim=-1) * 20.0,
                              (s * 100.0).clamp(-3.0, 3.0), (s * 20.0).clamp(-3.0, 3.0),
                              twist * 3.0, (hy * tang).sum(-1, keepdim=True),
                              (o[:, 6:9] * 10.0).clamp(-3.0, 3.0)], dim=-1)
        # obs: TCP - grasp x10 (0:3), tip vs the clear point x10 (3), radial x100 (4), angle / 30 (5), tube axis (6:9),
        # bore axis (9:12), tube velocity (12:15), fingers (15:17), gripper (17), joints (18:25), joint velocities
        # (25:32), last action (32:40), hand z (40:43), hand y (43:46)
        rel, axis, baxis, hz, hy = o[:, 0:3] * 0.1, o[:, 6:9], o[:, 9:12], o[:, 40:43], o[:, 43:46]
        h = (rel * axis).sum(-1, keepdim=True)
        lat = rel - h * axis                                                # the TCP's offset from the tube's axis, m
        tang = torch.cross(baxis, z, dim=-1)
        tang = tang / (tang * tang).sum(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return torch.cat([(lat * 400.0).clamp(-3.0, 3.0), (lat * 40.0).clamp(-3.0, 3.0),
                          (h * 200.0).clamp(-3.0, 3.0), (h * 20.0).clamp(-3.0, 3.0),
                          torch.cross(hz, -axis, dim=-1) * 10.0, torch.cross(axis, z, dim=-1) * 3.0,
                          torch.cross(hy, tang, dim=-1) * 3.0, (hy * tang).sum(-1, keepdim=True),
                          (o[:, 3:4] * 5.0).clamp(-3.0, 3.0), (o[:, 3:4] * 0.5).clamp(-3.0, 3.0),
                          (rel * 100.0).clamp(-3.0, 3.0)], dim=-1)

    def raw(self, obs: torch.Tensor):
        x = torch.cat([(obs - self.mean) / self.std, self.features(obs)], dim=-1)
        y = self.mlp(x)
        return torch.tanh(y[:, :7]), y[:, 7]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        joints, grip = self.raw(obs)
        g = torch.where(grip > 0, torch.ones_like(grip), -torch.ones_like(grip))
        return torch.cat([joints, g.unsqueeze(-1)], dim=-1)
