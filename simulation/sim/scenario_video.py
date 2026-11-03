r"""One labelled video per scenario run (run_scenario.py) and an H.264 copy:
every step's recording in order, each frame tagged with the step, who is
acting (pi0.5 / the RL teacher / the expert) and the step's outcome.

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\scenario_video.py OUT.mp4 SCENARIO_DIR [...]
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from view_episodes import frames  # noqa: E402

CAMS = ["platform_cam1", "wrist_top", "wrist_side"]
BLUE, GREEN, AMBER, WHITE, RED = (255, 160, 60), (80, 220, 80), (60, 200, 230), (255, 255, 255), (90, 90, 255)


def label(img, lines):
    for y, text, sc, col in lines:
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2 if sc > 0.8 else 1, cv2.LINE_AA)


def main():
    out = sys.argv[1]
    w = None
    for d in map(Path, sys.argv[2:]):
        sc = json.loads((d / "scenario.json").read_text())
        n_steps = len(sc["steps"])
        for s in sc["steps"]:
            segs = [(d / s["dir"], s.get("by", ""), s["success"], s.get("handover_step"), s.get("teacher_step"))]
            if s.get("fallback"):
                segs.append((d / s["fallback"]["dir"], "expert fallback", s["fallback"]["success"], None, None))
            for ep, by, ok, ho, tl in segs:
                if not (ep / "meta.json").exists():
                    continue
                for t, n, img in frames(ep, CAMS):
                    if by.startswith("RL only") and (tl is None or t < tl):
                        tag, col = "RL TEACHER, no VLA (" + by.split(": ")[-1] + ")", GREEN
                    elif by.startswith("expert"):
                        tag, col = by.upper(), AMBER
                    elif ho is None or t < ho:
                        tag, col = "pi0.5 (VLA)", BLUE
                    elif tl is None or t < tl:
                        tag, col = "RL TEACHER took over (" + by.split("+ ")[-1] + ")", GREEN
                    else:
                        tag, col = "teacher done -> holding / scripted finish", GREEN
                    label(img, [(34, tag, 0.9, col),
                                (66, f"[{sc['scenario']}] step {s['step'] + 1}/{n_steps}: \"{s['prompt']}\"", 0.6, WHITE),
                                (img.shape[0] - 14, f"step outcome: {'SUCCESS' if ok else 'FAIL'}   frame {t + 1}/{n}",
                                 0.6, WHITE if ok else RED)])
                    if w is None:
                        w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
                    w.write(img)
    if w is None:
        print("no frames")
        return
    w.release()
    import os
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:                                  # not in the Isaac venv: FFMPEG=<path to ffmpeg.exe>
        ffmpeg = os.environ["FFMPEG"]
    h264 = out.replace(".mp4", "_whatsapp.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", out, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "fast", h264], check=True)
    print("->", out, h264)


if __name__ == "__main__":
    main()
