"""Per-episode counts for every model-in-the-loop run: model calls with motion, look-only calls, moves executed,
env steps, and how long the arm stood still waiting for the model (from the command / reply file times).
  python loop_compare.py <loop dir> <out.jsonl>
"""
import json
import os
import sys
from pathlib import Path

L, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = [("setup", "remove_joint", "v1", "practice (camera set-up)"), ("rem", "remove_joint", "v1", ""),
        ("ins", "insert_joint", "v1", ""), ("insc", "insert_joint", "v1", "carry starts only"),
        ("rem2", "remove_joint", "v2", "same starts as the practice run"),
        ("ins2", "insert_joint", "v2", "same seed as the v1 insert run"),
        ("ins2b", "insert_joint", "v2", "same seed again (rerun)")]
rows = []
for name, task, ver, note in RUNS:
    D = L / name
    log = [json.loads(x) for x in (D / "private.jsonl").read_text().splitlines()]
    cmds = {r["n"]: r for r in log if "cmd" in r}
    ends = [(r["n"], r["finished"]) for r in log if "finished" in r]
    first = 0
    for end_n, fin in ends:
        ns = [n for n in sorted(cmds) if first <= n <= end_n]
        moves = [n for n in ns if cmds[n]["cmd"].get("op", "move") in ("move", "moves")]
        looks = [n for n in ns if cmds[n]["cmd"].get("op", "move") == "obs"]
        setup = [n for n in ns if cmds[n]["cmd"].get("op", "move") == "views"]
        n_moves = sum(len(cmds[n]["cmd"]["moves"]) if cmds[n]["cmd"].get("op") == "moves" else 1 for n in moves)
        think = 0.0
        for n in ns:
            if n == 0 or cmds[n]["cmd"].get("op") == "views" or (n - 1 in cmds and cmds[n - 1]["cmd"].get("op") == "views"):
                continue
            think += os.path.getmtime(D / f"cmd_{n:04d}.json") - os.path.getmtime(D / f"out_{n - 1:04d}.json")
        pics = {"full": 0, "crop": 0, "one": 0, "none": 0}
        for n in ns:
            pics[cmds[n]["cmd"].get("obs", "full") if n != end_n else "full"] += 1
        start = cmds[first - 1]["truth"] if first > 0 else cmds[ns[0]]["truth"] if looks and ns[0] == looks[0] else None
        rows.append({"run": name, "task": task, "version": ver, "note": note, "episode": fin["episode"],
                     "outcome": fin["outcome"], "env_steps": fin["steps"], "model_calls_with_motion": len(moves),
                     "look_only_calls": len(looks), "camera_setup_calls": len(setup), "moves_given": n_moves,
                     "wait_for_model_s": round(think, 1), "pictures": pics, "start_truth": start})
        first = end_n + 1
    left = [n for n in sorted(cmds) if n >= first and cmds[n]["cmd"].get("op", "move") in ("move", "moves")]
    if left:                                       # commands sent in an episode the server never saw end
        rows.append({"run": name, "task": task, "version": ver, "note": note, "episode": len(ends),
                     "outcome": "NOT RULED (server stopped before the env decided)", "env_steps": cmds[left[-1]]["steps"],
                     "model_calls_with_motion": len(left), "look_only_calls": 0, "camera_setup_calls": 0,
                     "moves_given": sum(len(cmds[n]["cmd"].get("moves", [0])) for n in left),
                     "wait_for_model_s": None, "pictures": None, "start_truth": cmds[first - 1]["truth"],
                     "end_truth": cmds[left[-1]]["truth"]})
OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
print("run    ver ep outcome   steps calls looks moves wait_s pictures")
for r in rows:
    print(f"{r['run']:6s} {r['version']} {r['episode']:2d} {r['outcome'][:9]:9s} {r['env_steps']:5d} "
          f"{r['model_calls_with_motion']:5d} {r['look_only_calls']:5d} {r['moves_given']:5d} "
          f"{str(r['wait_for_model_s']):>6s} {r['pictures']}")
