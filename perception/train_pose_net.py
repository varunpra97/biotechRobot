"""Train the vision -> keypose input estimator.

This is the "another model" that converts camera images into the initial
keyposes: a small CNN regresses (tray xyz, goal xyz) from the three camera
views, supervised by sim ground truth (vision_episodes/*/meta.json).
Keyposes are then computed analytically from the estimated poses
(perception.keyposes), keeping them physically grounded.

Trains on the Windows box (needs rendered episodes):
  python perception/train_pose_net.py --data assets/vision_episodes --epochs 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PoseEpisodes(Dataset):
    def __init__(self, root: Path):
        self.items = []
        for ep_dir in sorted(root.glob("ep_*")):
            meta = json.loads((ep_dir / "meta.json").read_text())
            imgs = []
            for name in ("wrist", "cam_a", "cam_b"):
                p = ep_dir / f"{name}.png"
                if not p.exists():
                    raise FileNotFoundError(f"missing {p}; run camera_sim.py first")
                from PIL import Image
                img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
                imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            x = torch.stack(imgs)  # (3, 3, H, W)
            y = torch.tensor(meta["tray_pos_init"] + meta["tray_pos_goal"],
                             dtype=torch.float32)  # (6,)
            self.items.append((x, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class ViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),   # 118x158
            nn.Conv2d(32, 64, 5, stride=2), nn.ReLU(),   # 57x77
            nn.Conv2d(64, 128, 5, stride=2), nn.ReLU(),  # 27x37
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PoseNet(nn.Module):
    """3 views -> (tray xyz, goal xyz). ~1M params; trivial on a 3070."""

    def __init__(self):
        super().__init__()
        self.enc = ViewEncoder()
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.ReLU(),
            nn.Linear(256, 6),
        )

    def forward(self, x):  # x: (B, 3, 3, H, W)
        feats = torch.cat([self.enc(x[:, i]) for i in range(3)], dim=1)
        return self.head(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("assets/pose_net.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    ds = PoseEpisodes(args.data)
    n_val = max(1, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    net = PoseNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        net.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        net.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                va_loss += loss_fn(net(x), y).item() * len(x)
                n += len(x)
        print(f"epoch {epoch:3d} train_mse={tr_loss/len(train_ds):.6f} "
              f"val_mse={va_loss/max(n,1):.6f} "
              f"(~{100*np.sqrt(va_loss/max(n,1)):.1f} cm RMSE)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
