"""Start-state library for the tube-grasp RL task: arm states near a tube
pick's grasp point, taken from recorded episodes rather than invented.

  - expert tube picks (data/primitives_tube_picks): every frame from where
    the TCP comes within NEAR_M of the grasp point until the gripper closes
    -- the approach the expert actually flies;
  - failed policy trials on the fixed layout (data/eval/*fixed*, *slot*):
    every frame within NEAR_M of the commanded tube's grasp point -- the
    states the VLA really gets stuck in.

Each entry: joint positions (7 arm + 2 fingers), the target slot (row, col),
its source. The grasp point of slot (r, c) is lab_scene.holder_slot_xy at
GRASP_Z (expert grasp TCP height, tube top - 12 mm) in the default layout --
the RL env's local frame is our sim's world frame.

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\rl\\start_states.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim"))
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states.npz"
NEAR_M = 0.10
# full milestone M1: --near 0.25 --out start_states_M1_recorded.npz (the finisher eval's misses were 15-25 cm out)
if "--near" in sys.argv:
    NEAR_M = float(sys.argv[sys.argv.index("--near") + 1])
if "--out" in sys.argv:
    OUT = HERE / sys.argv[sys.argv.index("--out") + 1]
EVAL_RUNS = ["tubepicks_1999_fixed", "tubepicks_14999_fixed", "tubepicks_14999_slotD5", "tubepicks_14999_slotB1",
             "v2v4_3999_fixed"]
if NEAR_M > 0.10:      # also pi0.5's part of the finisher evals (up to the hand-over)
    EVAL_RUNS += ["finisher_fixed", "finisher_slotB1", "finisher_slotD5", "m1_perceived"]
GRASP_Z = 0.998
ROWS = "ABCD"


def slot_rc(label):
    return ROWS.index(label[0]), int(label[1:]) - 1


def main():
    import lab_scene
    import layout
    layout.apply(layout.default_layout())
    grasp = {(r, c): np.array([*lab_scene.holder_slot_xy(r, c), GRASP_Z]) for r in range(4) for c in range(5)}

    q, slots, src, tcps = [], [], [], []

    def add(ep, label, frames):
        d = np.load(Path(ep) / "steps.npz")
        rc = slot_rc(label)
        tcp = d["tcp_pose"][:, :3]
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tcp), -1)[:, :2]], axis=1)
        for t in frames(d, tcp, grasp[rc]):
            q.append(qq[t])
            slots.append(rc)
            src.append(str(ep))
            tcps.append(tcp[t])

    # expert approaches: near the grasp point, before the gripper closes
    for f in sorted(glob.glob(str(DATA / "primitives_tube_picks" / "chains" / "*.json"))):
        for e in json.loads(Path(f).read_text())["episodes"]:
            if e["success"] and e["name"] == "pick_tube_from_holder":
                ep = DATA / "primitives_tube_picks" / e["dir"]

                def expert(d, tcp, g):
                    close = int(np.argmax(d["action"][:, 7] > 0.9))
                    return [t for t in range(close) if np.linalg.norm(tcp[t] - g) < NEAR_M]
                add(ep, e["args"]["slot"], expert)
    n_expert = len(q)

    # the VLA's failed attempts on the fixed layout
    for run in EVAL_RUNS:
        for ep in sorted(glob.glob(str(DATA / "eval" / run / "pick_tube_from_holder" / "ep_*"))):
            meta = json.loads((Path(ep) / "meta.json").read_text())
            label = meta["prompt"].split()[-1] if meta["prompt"].split()[-1][0] in ROWS else None
            label = label or next(w for w in meta["prompt"].split() if len(w) == 2 and w[0] in ROWS and w[1].isdigit())

            r = next((json.loads(x) for x in (DATA / "eval" / run / "results.jsonl").read_text().splitlines()
                      if x.strip() and json.loads(x)["dir"] == f"pick_tube_from_holder/{Path(ep).name}"), None)                 if (DATA / "eval" / run / "results.jsonl").exists() else None
            end = (r or {}).get("outcome", {}).get("handover_step") or 10 ** 9    # pi0.5's frames only

            def policy(d, tcp, g, end=end):
                return [t for t in range(min(len(tcp), end)) if np.linalg.norm(tcp[t] - g) < NEAR_M]
            add(ep, label, policy)

    q, slots = np.array(q, np.float32), np.array(slots, np.int32)
    np.savez(OUT, q=q, slot=slots, tcp=np.array(tcps, np.float32), source=np.array(src), n_expert=n_expert,
             grasp=np.array([grasp[(r, c)] for r in range(4) for c in range(5)], np.float32))
    per = {f"{ROWS[r]}{c + 1}": int(((slots[:, 0] == r) & (slots[:, 1] == c)).sum()) for r in range(4) for c in range(5)}
    print(f"{len(q)} start states ({n_expert} expert, {len(q) - n_expert} from failed policy trials) -> {OUT}")
    print("per slot:", per)


if __name__ == "__main__":
    main()
