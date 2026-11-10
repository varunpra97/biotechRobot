"""Shorthand for the model driving loop_server.py: each recipe is ONE model call (a batch of hand-pose moves) that
the model chooses after looking at the pictures. The model decides which recipe, which bore, whether the picture
agrees, and what to do when it does not ("raw" sends any command). Nothing here reads object poses.

The numbers below are the model's own notes, measured from the camera pictures in earlier episodes (robot base
frame, metres): where the rotor axis is, how a seated tube sits in the first bore, and that the three bores used are
60 degrees apart. Everything else follows from them. They hold only while the centrifuge is not moved.

  python loop_recipes.py <dir> look                     first full look of a run (also pins the three rim points)
  python loop_recipes.py <dir> info                     what the robot's own pose says: nearest bore, height, offset
  python loop_recipes.py <dir> remove_held              tube already in the fingers: pull out, lift, turn upright
  python loop_recipes.py <dir> remove_approach [bore]   onto the tube axis, down, close, pull out, lift, turn upright
  python loop_recipes.py <dir> insert_align [bore]      hand onto the bore axis, tube tip (if straight) 4 mm above the
                                                        opening; returns two zoomed crops with rings at the place the
                                                        tip should be (m0) and at the bore opening (m1)
  python loop_recipes.py <dir> insert_fix [bore] dxA dyA dxB dyB
        the tube tip as SEEN, relative to ring m0, in picture pixels of the left crop (view A) and right crop (view B):
        x to the right, y down. Triangulates the tip, turns the hand so the TUBE (not the hand) is on the bore axis
        and puts the tip in m0. Same crops come back.
  python loop_recipes.py <dir> insert_push [bore]       push 34 mm down the bore axis; one crop (view B) comes back
  python loop_recipes.py <dir> insert_push_inbore [bore]   tube already partly in the bore: push to the seat
  python loop_recipes.py <dir> insert_release           open, back off 4 cm, wait for the verdict
  python loop_recipes.py <dir> raw '<json command>'     anything else (see loop_client.py)
bore: 0, 1 or 2 (default: the bore whose axis is nearest the hand axis; printed in every reply).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
C = np.array([0.5507, 0.4160])           # rotor axis (x, y)
RHO, Z_TIP = 0.0353, 0.1230              # a seated tube tip: this far from the rotor axis, at this height
ANG = [34.8, -25.2, 94.8]                # the three bores, degrees about the rotor axis
TUBE, INSERT_GRASP, REMOVE_GRASP, BORE_DEPTH = 0.058, 0.046, 0.048, 0.03175   # tip to rim / to the TCP / seat to opening


def bore(k):
    th = np.radians(ANG[k])
    h = np.array([np.cos(th), np.sin(th)])
    a = np.array([0.7163 * h[0], 0.7163 * h[1], -0.6978])          # down the bore
    tip = np.array([C[0] + RHO * h[0], C[1] + RHO * h[1], Z_TIP])
    return {"a": a, "tip": tip, "opening": tip - BORE_DEPTH * a, "tan": np.array([-h[1], h[0], 0.0]),
            "rim": tip - TUBE * a}


def last():
    outs = sorted(D.glob("out_*.json"))
    return json.loads(outs[-1].read_text()) if outs else None


def send(cmd):
    n = len(list(D.glob("cmd_*.json")))
    tmp = D / f"cmd_{n:04d}.tmp"
    tmp.write_text(json.dumps(cmd))
    tmp.replace(D / f"cmd_{n:04d}.json")
    out = D / f"out_{n:04d}.json"
    t0 = time.time()
    while not out.exists():
        if time.time() - t0 > 400:
            sys.exit(f"no reply to command {n} after 400 s")
        time.sleep(0.05)
    time.sleep(0.05)
    r = json.loads(out.read_text())
    img = D / f"obs_{n:04d}.png"
    return r, (str(img) if img.exists() else "none")


def pose(r):
    tcp = np.array(r["tcp_mm"]) / 1e3
    R = np.stack([np.array(r["tool_x"]), np.array(r["tool_y_closing"]), np.array(r["tool_z_approach"])], axis=1)
    return tcp, R


def nearest(r):
    _, R = pose(r)
    return int(np.argmax([R[:, 2] @ bore(k)["a"] for k in range(3)]))


def geometry(r, k):
    tcp, R = pose(r)
    b = bore(k)
    seat = b["tip"] - INSERT_GRASP * b["a"]                       # the TCP when a held tube is seated
    d = tcp - seat
    up = -float(d @ b["a"])
    return {"bore": k, "hand_axis_vs_bore_deg": round(float(np.degrees(np.arccos(np.clip(R[:, 2] @ b["a"], -1, 1)))), 1),
            "tcp_above_seated_mm": round(up * 1e3, 1), "tcp_off_axis_mm": round(float(np.linalg.norm(d + up * b["a"])) * 1e3, 1),
            "tip_if_straight_above_opening_mm": round((up - BORE_DEPTH) * 1e3, 1)}


def tangent(r, k):
    _, R = pose(r)
    t = bore(k)["tan"]
    return (t if R[:, 1] @ t >= 0 else -t).tolist()


def show(r, img):
    if "quit" in r:
        print(json.dumps(r))
        return
    keep = {k: r[k] for k in ("episode", "steps_used", "steps_left", "finger_gap_mm", "gripper_commanded", "tcp_mm",
                              "tool_z_approach", "stopped", "episode_over", "error", "obs") if k in r}
    if "moves" in r:
        keep["moves"] = [f"{'ok' if m['arrived'] else 'NOT ARRIVED ' + str(m['pos_err_mm']) + 'mm'} {m['steps_this_move']}st"
                         + (f" gap {m['gap_mm']} {'ok' if m['require_ok'] else 'REQUIRE FAILED'}" if "require_ok" in m else "")
                         for m in r["moves"]]
    if "error" not in r:
        keep["nearest_bore"] = geometry(r, nearest(r))
    print(json.dumps(keep))
    print("image:", img)


LIFT = {"dpos": [0, 0, 0.05], "z_axis": [0, 0, -1], "max_steps": 150}
rec, args = sys.argv[2], sys.argv[3:]
r0 = last()
if rec == "look":
    show(*send({"op": "obs", "sticky": [bore(k)["rim"].tolist() for k in range(3)]}))
elif rec == "info":
    print(json.dumps({"finger_gap_mm": r0["finger_gap_mm"], "gripper": r0["gripper_commanded"],
                      "bores": [geometry(r0, k) for k in range(3)], "nearest": nearest(r0)}))
elif rec == "raw":
    show(*send(json.loads(args[0])))
elif rec == "remove_held":
    show(*send({"op": "moves", "obs": "none", "moves": [{"dtool": [0, 0, -0.045], "speed": 0.005}, LIFT, {"wait": 8}]}))
elif rec == "remove_approach":
    k = int(args[0]) if args else nearest(r0)
    b = bore(k)
    g = b["tip"] - REMOVE_GRASP * b["a"]
    show(*send({"op": "moves", "obs": "none", "moves": [
        {"pos": (g - 0.025 * b["a"]).tolist(), "z_axis": b["a"].tolist(), "y_axis": tangent(r0, k), "max_steps": 60},
        {"pos": g.tolist(), "speed": 0.004},
        {"grip": "close", "wait": 8, "require_gap_mm": [3, 8]},
        {"dtool": [0, 0, -0.05], "speed": 0.006}, LIFT, {"wait": 8}]}))
elif rec in ("insert_align", "insert_fix"):
    k = int(args[0]) if len(args) in (1, 5) else nearest(r0)
    b = bore(k)
    pre = b["opening"] - 0.004 * b["a"]
    view = {"obs": "crop", "focus": b["opening"].tolist(), "crop_w": 160, "mark_labels": False,
            "marks": [pre.tolist(), b["opening"].tolist()]}
    tcp, R = pose(r0)
    if rec == "insert_align":
        moves = []
        if np.linalg.norm(tcp + INSERT_GRASP * R[:, 2] - pre) > 0.012:      # far: first onto the axis 3 cm up
            moves.append({"point_tool": [0, 0, INSERT_GRASP], "pos": (b["opening"] - 0.030 * b["a"]).tolist(),
                          "z_axis": b["a"].tolist(), "y_axis": tangent(r0, k), "speed": 0.007, "max_steps": 70})
        moves.append({"point_tool": [0, 0, INSERT_GRASP], "pos": pre.tolist(), "z_axis": b["a"].tolist(),
                      "y_axis": tangent(r0, k), "speed": 0.004, "max_steps": 40})
        show(*send({"op": "moves", "moves": moves, **view}))
    else:
        dxa, dya, dxb, dyb = (float(x) for x in args[-4:])
        sc = 640.0 / float(r0["crop_w"])
        cal = json.loads((D / "calib.json").read_text())
        rays = []
        for name, dx, dy in (("A", dxa, dya), ("B", dxb, dyb)):
            c = cal[name]
            u, v = r0["px"][name]["marks"][0][0] + dx / sc, r0["px"][name]["marks"][0][1] + dy / sc
            d = ((u - c["W"] / 2) / c["f"] * np.array(c["right"]) - (v - c["H"] / 2) / c["f"] * np.array(c["up"])
                 - np.array(c["back"]))
            rays.append((np.array(c["pos"]), d / np.linalg.norm(d)))
        (p1, d1), (p2, d2) = rays
        s = np.linalg.lstsq(np.stack([d1, -d2], axis=1), p2 - p1, rcond=None)[0]
        q1, q2 = p1 + s[0] * d1, p2 + s[1] * d2
        tip = (q1 + q2) / 2
        tube = (tip - tcp) / np.linalg.norm(tip - tcp)
        ax = np.cross(tube, b["a"])
        sn, cs = np.linalg.norm(ax), float(tube @ b["a"])
        if sn > 1e-6:                                                        # the turn that puts the tube on the bore axis
            kx = ax / sn
            K = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            Rc = np.eye(3) + sn * K + (1 - cs) * (K @ K)
        else:
            Rc = np.eye(3)
        print(json.dumps({"tip_seen_m": [round(float(x), 4) for x in tip], "rays_pass_mm": round(float(np.linalg.norm(q1 - q2)) * 1e3, 2),
                          "tip_from_tcp_mm": round(float(np.linalg.norm(tip - tcp)) * 1e3, 1),
                          "tube_vs_bore_axis_deg": round(float(np.degrees(np.arctan2(sn, cs))), 1),
                          "tip_off_place_mm": round(float(np.linalg.norm(tip - pre)) * 1e3, 1)}))
        if (np.linalg.norm(q1 - q2) > 0.003 or not 0.030 < np.linalg.norm(tip - tcp) < 0.056
                or np.degrees(np.arctan2(sn, cs)) > 12.0):
            sys.exit("REJECTED: these pixel offsets do not describe a tube tip in the hand (rays apart, tip at an "
                     "impossible distance from the fingers, or tube more than 12 degrees off). Nothing moved.")
        show(*send({"op": "moves", "moves": [{"point_tool": (R.T @ (tip - tcp)).tolist(), "pos": pre.tolist(),
                                              "z_axis": (Rc @ R[:, 2]).tolist(), "y_axis": (Rc @ R[:, 1]).tolist(),
                                              "speed": 0.003, "max_steps": 40}], **view}))
elif rec == "insert_push":                               # down the BORE axis (the hand may be turned off it)
    k = int(args[0]) if args else nearest(r0)
    show(*send({"op": "moves", "obs": "one", "view": "B", "crop_w": 300, "focus": None, "marks": [],
                "moves": [{"dpos": (0.034 * bore(k)["a"]).tolist(), "speed": 0.0025, "max_steps": 30}]}))
elif rec == "insert_push_inbore":
    k = int(args[0]) if args else nearest(r0)
    g = geometry(r0, k)
    b = bore(k)
    show(*send({"op": "moves", "obs": "one", "view": "B", "crop_w": 300,
                "moves": [{"dpos": (b["a"] * max(0.0, g["tcp_above_seated_mm"] / 1e3 - 0.0005)).tolist(), "speed": 0.002,
                           "max_steps": 30}]}))
elif rec == "insert_release":
    show(*send({"op": "moves", "obs": "none", "moves": [{"grip": "open", "wait": 8, "require_gap_mm": [15, 30]},
                                                        {"dtool": [0, 0, -0.04], "speed": 0.005, "max_steps": 30},
                                                        {"wait": 8}]}))
else:
    sys.exit("unknown recipe " + rec)
