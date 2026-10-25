"""Overlay the TRUE pose of the episode's target tube on recorded wrist
frames, through camera_model: if the outline sits on the tube, intrinsics,
mount transform and conventions are right.
  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/realtime/checks/check_camera_model.py EPISODE_DIR OUT.jpg
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import trimesh

from camera_model import _lab_scene, pose_to_matrix, wrist_cameras

ep, out = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((ep / "meta.json").read_text())
names = meta["object_names"] if isinstance(meta["object_names"], list) else ast.literal_eval(meta["object_names"])
layout = meta["layout"] if isinstance(meta["layout"], dict) else ast.literal_eval(meta["layout"])
r, c = layout["pick_slot"]
target = f"tube_r{r}_c{c}"
d = np.load(ep / "steps.npz", allow_pickle=True)
ti = names.index(target)
ls = _lab_scene()
tube = trimesh.load(ls.TEST_TUBE_OBJ, force="mesh")
cams = wrist_cameras()
tiles = []
for k in np.linspace(0, len(d["hand_pose"]) - 1, 4).astype(int):
    row = []
    for cname, cam in cams.items():
        img = cv2.imread(str(ep / "images" / f"{cname}_{k:05d}.jpg"))
        Th = pose_to_matrix(d["hand_pose"][k])
        To = pose_to_matrix(d["object_poses"][k, ti])
        pts = tube.vertices @ To[:3, :3].T + To[:3, 3]
        uv, z = cam.project(Th, pts)
        if (z > 0).all():
            mask = np.zeros(img.shape[:2], np.uint8)
            for tri in np.round(uv[tube.faces] * 4).astype(np.int32):
                cv2.fillConvexPoly(mask, tri, 1, shift=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)
        cv2.putText(img, f"{cname} step {k}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        row.append(img)
    tiles.append(np.hstack(row))
cv2.imwrite(out, np.vstack(tiles))
print(f"target {target} ({meta['prompt']}); wrote {out}")
