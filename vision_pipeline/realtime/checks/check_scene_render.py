"""Raycast the scene map (true poses) plus the robot body into recorded
frames and draw the occluding contours the tracker would use: robot body
shaded, object contours in colour. If they sit on the image's edges, the
scene model, robot body model and cameras agree with the simulator.
  .venv-cad/Scripts/python.exe .../check_scene_render.py EPISODE_DIR OUT.jpg STEP[,STEP...]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from camera_model import exterior_camera, pose_to_matrix, wrist_cameras
from robot_body import arm_parts, body_parts
from scene_model import scene_from_sim
from scene_tracker import ROBOT, Renderer, occluding_contours

ep, out = Path(sys.argv[1]), sys.argv[2]
steps = [int(s) for s in sys.argv[3].split(",")]
meta = json.loads((ep / "meta.json").read_text())
d = np.load(ep / "steps.npz")
cams = dict(wrist_cameras())
cams["platform_cam1"] = exterior_camera()
objs = scene_from_sim(meta, d["object_poses"][0])
ren = Renderer(objs)
rows = []
for k in steps:
    poses = [o.T for o in scene_from_sim(meta, d["object_poses"][k])]
    Th = pose_to_matrix(d["hand_pose"][k])
    robot = [(n, m, Th @ T) for n, m, T in body_parts(d["finger_position_m"][k])] + arm_parts(d["joint_position"][k], Th)
    row = []
    for cn, cam in cams.items():
        img = cv2.imread(str(ep / "images" / f"{cn}_{k:05d}.jpg"))
        t = time.perf_counter()
        ids, depth, pts, us, vs = ren.render(cam.K, cam.world_from_cam(Th), (640, 480), poses, robot, stride=2)
        con, _ = occluding_contours(ids, depth)
        dt = (time.perf_counter() - t) * 1e3
        rob = cv2.resize((ids == ROBOT).astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST) > 0
        img[rob] = (0.5 * img[rob] + [0, 60, 120]).astype(np.uint8)
        ys, xs = np.nonzero(con)
        for y, x in zip(ys, xs):
            i = ids[y, x]
            col = (0, 0, 255) if objs[i].cls == "tube" else (255, 0, 255) if objs[i].cls == "tip" else (0, 255, 0)
            img[int(vs[y, x]), int(us[y, x])] = col
        cv2.putText(img, f"{cn} {k} {dt:.0f}ms", (8, 20), 0, 0.6, (0, 255, 255), 2)
        row.append(img)
    rows.append(np.hstack(row))
cv2.imwrite(out, np.vstack(rows))
print("objects", len(objs), "wrote", out)
