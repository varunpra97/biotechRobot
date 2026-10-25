"""The localizer as its own process (in .venv-cad, which has open3d), serving
a client in the robot / simulator process over a local socket -- the way it
would run beside a real robot, and it keeps the localizer's compute out of
the control loop.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/realtime/server.py --port 5577

Messages (wire.py), client -> server:
  {"type": "init", "meta": {"object_names": [...], "layout": {...}}, "cameras": [...], "image_lag": 0.25}
      + array "poses" (N, 7): the scene map (simulation stand-in: scene_from_sim)
  {"type": "step", "held": name | null} + arrays "hand_pose" (7), "finger_pos" (2), "q" (7),
      "img:<camera>" (H, W, 3) uint8 RGB or (H, W) gray
  {"type": "close"}
server -> client, after init and every step:
  {"type": "state", "step": n, "latency_ms": ..., "objects": {name: {"pose": [x,y,z,qw,qx,qy,qz],
   "sigma_mm", "source", "steps_since_seen"}}}  (+ "targets" per object when the step asked "targets": true)
One connection at a time; a new init rebuilds the localizer.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wire                                    # noqa: E402
from localizer import SceneLocalizer           # noqa: E402
from scene_model import scene_from_sim         # noqa: E402


def compact(loc: SceneLocalizer, targets=False):
    m = loc.state_message(include_targets=targets)
    objs = {}
    for o in m["objects"]:
        d = {k: o[k] for k in ("pose", "sigma_mm", "source", "steps_since_seen", "class")}
        if targets:
            d["targets"] = o["targets"]
        objs[o["name"]] = d
    return {"type": "state", "step": m["step"], "latency_ms": m["latency_ms"], "objects": objs}


def serve(port, host="127.0.0.1"):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"localizer listening on {host}:{port}", flush=True)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("client", addr, flush=True)
        loc = None
        try:
            while True:
                h, arrs = wire.recv(conn)
                if h["type"] == "init":
                    t = time.perf_counter()
                    objs = scene_from_sim(h["meta"], arrs["poses"])
                    loc = SceneLocalizer(objs, cameras=tuple(h.get("cameras", ("wrist_top", "wrist_side"))),
                                         image_lag=h.get("image_lag", 0.25))
                    print(f"init: {len(objs)} objects, {time.perf_counter() - t:.1f} s", flush=True)
                    wire.send(conn, compact(loc, targets=True))
                elif h["type"] == "step":
                    imgs = {k[4:]: v for k, v in arrs.items() if k.startswith("img:")}
                    loc.step(imgs, arrs["hand_pose"], arrs["finger_pos"], arrs.get("q"), h.get("held"))
                    wire.send(conn, compact(loc, targets=h.get("targets", False)))
                elif h["type"] == "close":
                    break
        except (ConnectionError, OSError) as e:
            print("client gone:", e, flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5577)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    serve(a.port, a.host)
