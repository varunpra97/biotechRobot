"""Train the high-level keypose diffuser (ChainedDiffuser-style level 1).

Learns  p(keypose_seq (K,7) [pos,quat] | tray_init, tray_goal)  from the
keypose-annotated demos written by sim/make_demos.py. The demos store the
ACHIEVED task-space keyposes (IK solutions), so the diffuser learns
physically reachable sequences, not idealized ones.

Usage (on the Windows box, after generating demos):
  PYTHONPATH=. python policies/train_keypose.py \
      --data assets/demos --epochs 200 --out assets/keypose_diffusion.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from policies.diffusion import KeyposeDiffusion, diffusion_loss


def _quat_continuous(kps: np.ndarray) -> np.ndarray:
    """Flip quat signs so consecutive keyposes have positive dot product."""
    kps = kps.copy()
    for i in range(1, len(kps)):
        if float(np.dot(kps[i - 1, 3:], kps[i, 3:])) < 0:
            kps[i, 3:] *= -1
    return kps


class KeyposeDataset(Dataset):
    def __init__(self, root: Path):
        files = sorted(root.glob("demo_*.npz"))
        if not files:
            raise FileNotFoundError(
                f"no demo_*.npz in {root}; run sim/make_demos.py first")
        x, c = [], []
        for f in files:
            d = np.load(f)
            kp = _quat_continuous(d["keypose_task"].astype(np.float64))  # (K,7)
            x.append(kp)
            c.append(np.concatenate(
                [d["tray_init"].ravel()[:3], d["tray_goal"].ravel()[:3]]))  # (6,)
        self.x = torch.from_numpy(np.stack(x)).float()  # (N,K,7)
        self.c = torch.from_numpy(np.stack(c)).float()  # (N,6)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.c[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("assets/keypose_diffusion.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = KeyposeDataset(args.data)
    n_val = max(1, int(len(ds) * args.val_frac))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    K = ds.x.shape[1]
    model = KeyposeDiffusion(K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Normalization from the full dataset; saved with the checkpoint so
    # inference applies identical scaling.
    x_mean, x_std = ds.x.mean(dim=(0, 1)), ds.x.std(dim=(0, 1)) + 1e-6
    c_mean, c_std = ds.c.mean(dim=0), ds.c.std(dim=0) + 1e-6

    def batch_loss(batch):
        # KeyposeDiffusion.cond takes (tray, goal) separately; c is the
        # concatenated 6-D vector, so split it back.
        x, c = batch
        x = ((x - x_mean) / x_std).transpose(1, 2).to(device)  # (B,7,K)
        tray, goal = c[:, :3].to(device), c[:, 3:].to(device)
        tray = (tray - c_mean[:3]) / c_std[:3]
        goal = (goal - c_mean[3:]) / c_std[3:]
        return diffusion_loss(model, model.ddpm, x, model.cond(tray, goal))

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
        "K": K,
        "x_mean": x_mean, "x_std": x_std,
        "c_mean": c_mean, "c_std": c_std,
    }, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
