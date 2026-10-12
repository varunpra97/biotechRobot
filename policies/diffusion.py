"""Conditional DDPMs for the chained policy (ChainedDiffuser-style).

Two levels sharing one core::

    KeyposeDiffusion:     p(keypose_seq [K,7] | tray_xyz, goal_xyz)
    TrajectoryDiffusion:  p(action_chunk [T,6] | keypose_a [7], keypose_b [7])

DDPM with a small 1D UNet + FiLM conditioning. Shape-tested on CPU here;
training runs on the Windows GPU against assets/demos/*.npz.

Training data contracts (from sim/make_demos.py):
  - KeyposeDiffusion:  x0 = keypose_task (K,7), cond = (tray_init, tray_goal)
  - TrajectoryDiffusion: x0 = action chunk between consecutive keyposes
    (slice q_traj at keypose_idx), cond = (keypose_task[i], keypose_task[i+1])
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # (B,)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.film = nn.Linear(cond_dim, 2 * ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return h * (1 + scale[..., None]) + shift[..., None]


class CondUNet1D(nn.Module):
    """Small 1D UNet with FiLM conditioning (time + task cond)."""

    def __init__(self, in_dim: int, cond_dim: int = 128, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.in_proj = nn.Conv1d(in_dim, 64, 3, padding=1)
        self.down1 = FiLMBlock(64, cond_dim)
        self.ds1 = nn.Conv1d(64, 128, 3, stride=2, padding=1)
        self.down2 = FiLMBlock(128, cond_dim)
        self.ds2 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.mid = FiLMBlock(256, cond_dim)
        self.us2 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.up2 = FiLMBlock(256, cond_dim)
        self.us1 = nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1)
        self.up1 = FiLMBlock(192, cond_dim)
        self.out = nn.Conv1d(192, in_dim, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        c = self.time_mlp(t) + cond
        h0 = self.in_proj(x)
        h1 = self.down1(h0, c)
        h2 = self.down2(self.ds1(h1), c)
        h3 = self.mid(self.ds2(h2), c)
        u2 = self.us2(h3)
        u2 = F.interpolate(u2, size=h2.shape[-1], mode="nearest")
        h4 = self.up2(torch.cat([u2, h2], dim=1), c)
        u1 = self.us1(h4)
        u1 = F.interpolate(u1, size=h1.shape[-1], mode="nearest")
        h5 = self.up1(torch.cat([u1, h1], dim=1), c)
        return self.out(h5)


class DDPM:
    def __init__(self, timesteps: int = 100):
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.register = {}
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def _ab(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(t.device)[t][:, None, None]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self._ab(t)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                 cond: torch.Tensor) -> torch.Tensor:
        ab = self._ab(t)
        ab_prev = self._ab(torch.clamp(t - 1, min=0))
        beta_t = 1 - ab / torch.clamp(ab_prev, min=1e-8)
        eps = model(x_t, t, cond)
        x0_hat = (x_t - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
        mean = (ab_prev.sqrt() * beta_t / (1 - ab).clamp(min=1e-8)) * x0_hat \
             + (ab.sqrt() * (1 - ab_prev) / (1 - ab).clamp(min=1e-8)) * x_t
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        var = beta_t * (1 - ab_prev) / (1 - ab).clamp(min=1e-8)
        return mean + var.clamp(min=0).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple,
               cond: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond)
        return x


def diffusion_loss(model: nn.Module, ddpm: DDPM,
                   x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    B = x0.shape[0]
    t = torch.randint(0, ddpm.timesteps, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = ddpm.q_sample(x0, t, noise)
    return F.mse_loss(model(x_t, t, cond), noise)


class KeyposeDiffusion(nn.Module):
    """p(keypose_seq [K,7] | tray_xyz, goal_xyz). High level (Zhou: keyposes)."""

    def __init__(self, K: int = 6, cond_dim: int = 128):
        super().__init__()
        self.K = K
        self.unet = CondUNet1D(in_dim=7, cond_dim=cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(6, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.ddpm = DDPM()

    def cond(self, tray: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.cond_mlp(torch.cat([tray, goal], dim=-1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        return self.unet(x_t, t, cond)


class TrajectoryDiffusion(nn.Module):
    """p(action_chunk [T,6] | keypose_a [7], keypose_b [7]).
    Low level (Zhou: trajectory diffuser connecting macro-actions)."""

    def __init__(self, cond_dim: int = 128):
        super().__init__()
        self.unet = CondUNet1D(in_dim=6, cond_dim=cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(14, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.ddpm = DDPM()

    def cond(self, kp_a: torch.Tensor, kp_b: torch.Tensor) -> torch.Tensor:
        return self.cond_mlp(torch.cat([kp_a, kp_b], dim=-1))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        return self.unet(x_t, t, cond)


if __name__ == "__main__":
    # CPU shape test (no training here).
    torch.manual_seed(0)
    kp = KeyposeDiffusion(K=6)
    B, K = 2, 6
    x0 = torch.randn(B, 7, K)
    tray, goal = torch.randn(B, 3), torch.randn(B, 3)
    c = kp.cond(tray, goal)
    loss = diffusion_loss(kp, kp.ddpm, x0, c)
    s = kp.ddpm.sample(kp, (B, 7, K), c)
    assert s.shape == x0.shape, s.shape
    tr = TrajectoryDiffusion()
    T = 32
    a0 = torch.randn(B, 6, T)
    c2 = tr.cond(torch.randn(B, 7), torch.randn(B, 7))
    loss2 = diffusion_loss(tr, tr.ddpm, a0, c2)
    s2 = tr.ddpm.sample(tr, (B, 6, T), c2, device="cpu")
    assert s2.shape == a0.shape, s2.shape
    print(f"shapes OK | keypose_loss={loss.item():.4f} traj_loss={loss2.item():.4f}")
