"""Train the low-level trajectory diffuser (ChainedDiffuser-style level 2).

Learns  p(action_chunk (T,6) | keypose_a (7), keypose_b (7))  from the
segments of q_traj between consecutive keyposes in sim/make_demos.py's
demos. Conditioning is on the task-space keypose pair (pos+quat), which
is what the high-level diffuser outputs; the target is the joint-space
action chunk connecting them.

The q_init -> keypose_0 segment is excluded: q_init is a randomized
start, not a keypose, and the diffuser's contract is keypose -> keypose.

Variable-length segments are resampled to a fixed T (--chunk-len) by
joint-wise linear interpolation; the demos are time-parameterized with
constant peak joint velocity, so resampling preserves dynamics.

Usage (on the Windows box, after generating demos):
  PYTHONPATH=. python policies/train_traj.py \
      --data assets/demos --epochs 200 --out assets/traj_diffusion.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from policies.diffusion import TrajectoryDiffusion, diffusion_loss


def _resample(chunk: np.ndarray, T: int) -> np.ndarray:
    """(n,6) -> (T,6) via joint-wise linear interpolation."""
    n = len(chunk)
    if n == T:
        return chunk
    src = np.linspace(0, 1, n)
    dst = np.linspace(0, 1, T)
    return np.stack(
        [np.interp(dst, src, chunk[:, j]) for j in range(chunk.shape[1])],
        axis=1,
    )


class TrajChunkDataset(Dataset):
    def __init__(self, root: Path, chunk_len: int = 64):
        files = sorted(root.glob("demo_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no demo_*.npz in {root}; run sim/make_demos.py first")
        x, c = [], []
        for f in files:
            d = np.load(f)
            q_traj = d["q_traj"].astype(np.float64)          # (Ttot,6)
            kp_task = d["keypose_task"].astype(np.float64)   # (6,7)
            idx = d["keypose_idx"].astype(int).ravel()       # (6,) cumsum
            bounds = np.concatenate([[0], idx])
            # segment i connects keypose i-1 -> i; segment 0 (q_init -> kp0)
            # is excluded (see module docstring).
            for i in range(1, len(bounds) - 1):
                seg = q_traj[bounds[i]:bounds[i + 1]]
                if len(seg) < 4:
                    continue
                x.append(_resample(seg, chunk_len))          # (T,6)
                c.append(np.concatenate(
                    [kp_task[i - 1], kp_task[i]]))           # (14,)
        if not x:
            raise RuntimeError(f"no usable chunks in {root}")
        self.x = torch.from_numpy(np.stack(x)).float()  # (N,T,6)
        self.c = torch.from_numpy(np.stack(c)).float()  # (N,14)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.c[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--chunk-len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("assets/traj_diffusion.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = TrajChunkDataset(args.data, chunk_len=args.chunk_len)
    print(f"{len(ds)} chunks from {args.data}")
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    model = TrajectoryDiffusion().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_mean, x_std = ds.x.mean(dim=(0, 1)), ds.x.std(dim=(0, 1)) + 1e-6
    c_mean, c_std = ds.c.mean(dim=0), ds.c.std(dim=0) + 1e-6

    def batch_loss(batch):
        x, c = batch
        x = ((x - x_mean) / x_std).transpose(1, 2).to(device)  # (B,6,T)
        kpa = ((c[:, :7] - c_mean[:7]) / c_std[:7]).to(device)
        kpb = ((c[:, 7:] - c_mean[7:]) / c_std[7:]).to(device)
        return diffusion_loss(model, model.ddpm, x, model.cond(kpa, kpb))

    for epoch in range(args.epochs):
        model.train()
        tr, n = 0.0, 0
        for batch in train_dl:
            opt.zero_grad()
            loss = batch_loss(batch)
            loss.backward()
            opt.step()
            tr += loss.item() * len(batch[0])
            n += len(batch[0])
        model.eval()
        va, m = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                va += batch_loss(batch).item() * len(batch[0])
                m += len(batch[0])
        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} train={tr/n:.5f} val={va/m:.5f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "chunk_len": args.chunk_len,
        "x_mean": x_mean, "x_std": x_std,
        "c_mean": c_mean, "c_std": c_std,
    }, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
