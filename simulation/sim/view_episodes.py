"""Play recorded primitive episodes from the two policy cameras (exterior
platform_cam1 | the episode's wrist camera), side by side at the recorded 15 Hz, with the
prompt, step and outcome overlaid -- what π0.5 is trained on.

Usage (any Python with opencv-python + numpy, e.g. the Isaac venv):
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives --primitive attach_tip
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives --cameras top wrist --speed 2
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives --save data\\primitives\\all_primitives.mp4
Keys: space pause, n next episode, q quit. --save writes the captioned
video instead of showing it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def frames(ep_dir: Path, cams):
    n = json.loads((ep_dir / "meta.json").read_text())["n_steps"]
    for t in range(n):
        imgs = [cv2.imread(str(ep_dir / "images" / f"{c}_{t:05d}.jpg")) for c in cams]
        if any(i is None for i in imgs):
            return
        yield t, n, np.concatenate(imgs, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder of <primitive>/ep_XXXX episodes")
    ap.add_argument("--primitive", default=None)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="default: each episode's policy cameras (exterior + its wrist camera)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="write an MP4 here instead of showing a window")
    args = ap.parse_args()

    root = Path(args.root)
    summary = root / "summary.json"
    if summary.exists():  # chain order
        order = []
        for i, s in enumerate(json.loads(summary.read_text())):
            k = sum(1 for o in order if o[0] == s["primitive"])
            order.append((s["primitive"], root / s["primitive"] / f"ep_{k:04d}"))
        eps = [p for _, p in order if p.exists()]
    else:
        eps = sorted(p.parent for p in root.glob("*/ep_*/meta.json"))
    if args.primitive:
        eps = [e for e in eps if e.parent.name == args.primitive]
    delay = max(1, int(1000 / 15 / args.speed))
    win = "episodes"
    writer = None
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        cams = args.cameras or meta.get("training_cameras", ["platform_cam1", "wrist"])
        for t, n, img in frames(ep, cams):
            ok = "success" if meta["success"] else "FAILED"
            for y, text in ((24, f"{ep.parent.name}/{ep.name}  [{ok}]  {' | '.join(cams)}"), (50, f"\"{meta['prompt']}\""),
                            (img.shape[0] - 12, f"step {t + 1}/{n}  ({(t + 1) / 15:.1f} s)")):
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15 * args.speed,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
                continue
            cv2.imshow(win, img)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord(" "):
                key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                return
            if key == ord("n"):
                break
    if writer is not None:
        writer.release()
        print(f"wrote {args.save} ({len(eps)} episodes)")
        return
    cv2.waitKey(800)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
