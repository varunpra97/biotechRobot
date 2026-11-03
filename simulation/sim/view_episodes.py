"""Play recorded primitive episodes from the three policy cameras
(lab_env.POLICY_CAMERAS: exterior platform_cam1 | wrist_top | wrist_side),
side by side at the recorded 15 Hz, with the prompt, step and outcome
overlaid -- what π0.5 is trained on. Several dataset folders play one after
another, each in its recorded order (valid episodes only).

Usage (any Python with opencv-python + numpy, e.g. the Isaac venv):
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives_v2
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives_v2 --primitive attach_tip
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\primitives_v2 --cameras wrist_top wrist_side --speed 2
    python biotech-lab-robotics\\simulation\\sim\\view_episodes.py data\\test_move_tray data\\test_3cam --save data\\new_cameras.mp4
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


def episodes_of(root: Path):
    """A dataset folder's valid episodes in recorded order."""
    summary = root / "summary.json"
    data = json.loads(summary.read_text()) if summary.exists() else None
    if isinstance(data, dict):  # collect_primitives.py generator: recorded order, valid episodes
        return [root / e["dir"] for e in data["episodes"] if e["success"] and (root / e["dir"]).exists()]
    if data is not None:  # older single-chain summary: chain order
        order = []
        for s in data:
            k = sum(1 for o in order if o[0] == s["primitive"])
            order.append((s["primitive"], root / s["primitive"] / f"ep_{k:04d}"))
        return [p for _, p in order if p.exists()]
    return sorted(p.parent for p in root.glob("*/ep_*/meta.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="folder(s) of <primitive>/ep_XXXX episodes")
    ap.add_argument("--primitive", default=None)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="default: each episode's policy cameras (exterior, wrist_top, wrist_side)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="write an MP4 here instead of showing a window")
    args = ap.parse_args()

    eps = [ep for root in args.roots for ep in episodes_of(Path(root))]
    if args.primitive:
        eps = [e for e in eps if e.parent.name == args.primitive]
    delay = max(1, int(1000 / 15 / args.speed))
    win = "episodes"
    writer = None
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        cams = args.cameras or meta.get("policy_cameras") or meta.get("training_cameras", ["platform_cam1"])
        for t, n, img in frames(ep, cams):
            ok = "success" if meta["success"] else "FAILED"
            for y, text in ((24, f"{ep.parent.parent.name}: {ep.parent.name}/{ep.name}  [{ok}]  {' | '.join(cams)}"), (50, f"\"{meta['prompt']}\""),
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
