"""Generate many varied, physically validated 15 Hz episodes of the lab
primitives for π0.5 fine-tuning.

Work is done in CHAINS. A chain is one fresh Isaac process with its own
seeded layout and look, running a seeded sequence of primitives in one
physical simulation: each primitive starts from the state the previous one
left, so the tube really is in the centrifuge when "remove" starts. Each
primitive is recorded as its own episode with its own prompt, except that
~10-20% of episodes are CHAINED: one prompt over a whole sequence, e.g.
"transfer 150 uL from tube B1 to well A3" (attach tip, aspirate, dispense,
eject).

Varied per chain, all from the chain's seed:
  - where everything stands (layout.py): the tube holder, centrifuge, tip
    rack, waste bin, pipette and tray are each moved and turned on the
    bench, collision-free and within reach -- a new arrangement every chain
  - which holder slots are empty, which tubes hold liquid and how much
  - the look: appearance.apply() with a fresh seed (bench, walls, floor,
    lights, labware and liquid colours), recorded in every episode's meta
  - which task block the chain runs: one of tubes, pipette, tray (weighted
    by BLOCK_WEIGHTS), so the arrangement changes every few episodes
  - tubes: the source slot, centrifuge bore (any of the six the arm can
    reach in this layout, each approached tangentially to the rotor) and
    target slot (own slot or an empty one); only tubes without liquid are
    tilted into the rotor
  - pipette: 1-3 transfers, each with an unused tip, a volume, a source
    tube that keeps > MIN_REMAIN_UL after aspirating (so the tip reaches
    the liquid), and a target well (empty) or tube (with room)
  - tray: where it is set down, then (usually) back to its start
Targets are pre-checked with pure-math IK (no physics) and skipped if the
arm can't reach them with joint-limit margin.

Every primitive keeps its physical outcome check; only successful episodes
count. A failed episode is moved to <out>/_failed/ and ends its chain (the
world may be in a state the next primitive can't start from); the next
chain starts fresh. A primitive the planner rejects before any motion
(collision, IK, or the arm pushed by contact while checking the path) has
not touched the world: that episode is discarded and the next candidate
target is tried. Reach depends on the layout, so it is checked per chain
(pure-math IK); only tips that keep failing are excluded across chains. All cameras are recorded; the policy views are
lab_env.POLICY_CAMERAS (exterior, wrist_top, wrist_side) in every episode;
lab_env.PRIMITIVE_WRIST_CAMERA's camera for the primitive (the first one, for
a chained episode) is kept as meta "best_wrist_camera".

Output (<out> = data/primitives_v2 by default):
  <name>/ep_<chain>_<k>/      one episode (lab_env.EpisodeRecorder layout);
                              <name> is the primitive or chained sequence
  _failed/<name>/ep_.../      failed episodes, kept for debugging
  chains/chain_<chain>.json   per-chain record, updated after every episode
  logs/chain_<chain>.log      the chain's Isaac output
  summary.json                running totals, failures, episodes per GPU hour

Resume: re-run the same command. Finished chains are kept; a chain that was
interrupted keeps its completed episodes, its unfinished one is deleted,
and generation continues with the next chain index (chain seeds are
deterministic in --seed and the chain index). It stops once --episodes
valid episodes exist, or once the chains' process time reaches --hours.

Before every chain the driver checks the shared GPU (the vision pipeline's
gpu_guard.sh) and waits while another session's GPU job runs.

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\sim\\collect_primitives.py --episodes 20 --seed 0
    ...collect_primitives.py --out data\\primitives_v3 --episodes 100000 --hours 100 --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = r"C:\Users\varun\workspace\BiotechRobot\data\primitives_v2"
GPU_GUARD = HERE.parents[1] / "vision_pipeline" / "isaac_validation" / "gpu_guard.sh"
GPU_WAIT_S = 15 * 60                 # user's rule: re-check a busy GPU every ~15 min
CHAIN_TIMEOUT_S = 4 * 3600
MAX_DEAD_CHAINS = 10                 # consecutive chains with no valid episode -> stop (one block per chain: an empty chain happens)

SLOTS = [(r, c) for r in range(4) for c in range(5)]
WELLS = [f"{row}{i}" for row in "AB" for i in range(1, 6)]
TIPS = [f"{row}{c}" for row in "ABCDEFGHIJ" for c in range(1, 11)]

# Liquid (uL). A tip's end reaches at most ~36 mm below a tube's rim, so a
# tube can be aspirated from only while the liquid left after aspirating
# stands above that (~485 uL) with a few mm of immersion (~570 uL).
MIN_REMAIN_UL = 600.0
SOURCE_UL = (800, 1300)              # initial volume of a liquid tube
TUBE_MAX_UL = 1100.0                 # dispense target's surface stays below the tip's end
WELL_VOLUMES = range(20, 251, 10)    # a well holds ~400 uL; tip end sits ~6.7 mm above its floor
TUBE_VOLUMES = range(20, 301, 10)

# One task block per chain, so the arrangement of the bench changes every
# few episodes. Weights roughly balance the primitives (a pipette block
# records the most episodes, a tray block the fewest).
BLOCK_WEIGHTS = {"tubes": 0.4, "pipette": 0.3, "tray": 0.3}

REACH_MARGIN = 0.05                  # rad of joint-limit margin required by the IK pre-check
# Where a pick episode starts (move_to_start, unrecorded): the home pose --
# where an eval trial, or a new task, starts -- or a random pose near it.
# Without this only a chain's first episode started at home: a few home-start
# demos per task, and a policy that from home always went to the tube rack.
START_HOME_P = 0.4
START_OFFSET_M = dict(x=(-0.12, 0.12), y=(-0.10, 0.10), z=(-0.08, 0.06))
START_YAW_DEG = 25.0
TUBE_PICKS_PER_CHAIN = 8             # block "tube_picks": pick + place cycles per chain

# Chained sequences: name -> the primitives it covers.
CHAINED = {
    "transfer_to_well": ("attach_tip", "aspirate_from_tube", "dispense_into_well", "eject_tip"),
    "transfer_to_tube": ("attach_tip", "aspirate_from_tube", "dispense_into_tube", "eject_tip"),
    "load_tube_into_centrifuge": ("pick_tube_from_holder", "insert_tube_into_bore"),
    "unload_tube_to_holder": ("remove_tube_from_bore", "place_tube_into_holder"),
    "move_tray": ("pick_up_tray", "place_tray"),
}
CHAINED_TARGET = (0.10, 0.20)


def tube_name(slot):
    return f"tube_r{slot[0]}_c{slot[1]}"


def slot_label(slot):
    return "ABCD"[slot[0]] + str(slot[1] + 1)


def chain_seed(seed, chain):
    return seed * 1_000_003 + chain


def as_list(targets):
    """A step's target(s): None, one name, or a tuple of names."""
    return [] if not targets else list(targets) if isinstance(targets, (list, tuple)) else [targets]


def uniform(rng, lo_hi):
    return rng.uniform(*lo_hi)


# =================================================================== driver
def load_records(out):
    recs = {}
    for p in sorted((out / "chains").glob("chain_*.json")):
        try:
            r = json.loads(p.read_text())
            recs[r["chain"]] = r
        except (OSError, ValueError, KeyError):
            pass
    return recs


def save_record(out, rec):
    p = out / "chains" / f"chain_{rec['chain']:05d}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    os.replace(tmp, p)


def cleanup_chain(out, rec):
    """Delete a chain's episode folders that its record doesn't list as
    finished (an episode cut off by a crash)."""
    keep = {e["dir"] for e in rec.get("episodes", [])}
    tag = f"ep_{rec['chain']:05d}_"
    for d in list(out.glob(f"*/{tag}*")) + list(out.glob(f"_failed/*/{tag}*")):
        if d.relative_to(out).as_posix() not in keep:
            shutil.rmtree(d, ignore_errors=True)


def write_summary(out, target=None):
    recs = load_records(out)
    eps = [e for c in sorted(recs) for e in recs[c].get("episodes", [])]
    ok = [e for e in eps if e["success"]]
    by = {}
    for e in eps:
        b = by.setdefault(e["name"], {"ok": 0, "failed": 0, "rejected": 0})
        b["ok" if e["success"] else "rejected" if e.get("rejected") else "failed"] += 1
    gpu_s = sum(r.get("process_wall_s") or r.get("wall_s") or 0.0 for r in recs.values())
    startup = [r["startup_s"] for r in recs.values() if r.get("startup_s")]
    summary = {
        "target_episodes": target,
        "valid_episodes": len(ok),
        "failed_episodes": sum(1 for e in eps if not e["success"] and not e.get("rejected")),
        "planning_rejections": sum(1 for e in eps if e.get("rejected")),
        "excluded_targets": sorted(target_stats(recs)[0]),
        "chained_episodes": sum(1 for e in ok if e["chained"]),
        "chained_fraction": round(sum(1 for e in ok if e["chained"]) / max(1, len(ok)), 3),
        "chains": {str(c): {k: r.get(k) for k in ("seed", "status", "valid", "process_wall_s", "stop_reason")}
                   for c, r in sorted(recs.items())},
        "gpu_hours": round(gpu_s / 3600, 3),
        "episodes_per_gpu_hour": round(len(ok) / (gpu_s / 3600), 2) if gpu_s else None,
        "mean_startup_s": round(sum(startup) / len(startup), 1) if startup else None,
        "by_name": dict(sorted(by.items())),
        "failures": [{"dir": e["dir"], "prompt": e["prompt"], "failed_at": e.get("failed_at"),
                      "target": e.get("failed_targets") or e.get("failed_target"),
                      "rejected": e.get("rejected", False),
                      "info": e.get("info_short")} for e in eps if not e["success"]],
        "episodes": [{"dir": e["dir"], "name": e["name"], "prompt": e["prompt"], "success": e["success"],
                      "chained": e["chained"], "steps": e["steps"]} for e in eps if e["dir"]],
    }
    tmp = out / "summary.tmp"
    tmp.write_text(json.dumps(summary, indent=1, default=float))
    os.replace(tmp, out / "summary.json")
    return summary


def gpu_busy():
    """(busy, why) from the vision pipeline's guard. Fails safe."""
    bash = shutil.which("bash")    # Git bash; a bare "bash" can resolve to WSL's
    if not bash or not GPU_GUARD.exists():
        return False, "no guard available"
    env = dict(os.environ, GPU_GUARD_SELF_PID=str(os.getpid()))
    try:
        p = subprocess.run([bash, str(GPU_GUARD)], capture_output=True, text=True, env=env, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return True, f"guard failed: {e}"
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def drive(args):
    out = Path(args.out).resolve()      # the worker runs from this folder, so pass it an absolute path
    (out / "chains").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    dead = launched = 0
    while True:
        recs = load_records(out)
        for rec in recs.values():          # interrupted earlier: no worker is running now
            if rec.get("status") == "running":
                rec["status"] = "crashed"
                save_record(out, rec)
            cleanup_chain(out, rec)
        s = write_summary(out, args.episodes)
        if s["valid_episodes"] >= args.episodes:
            print(f"done: {s['valid_episodes']} valid episodes ({s['chained_episodes']} chained), "
                  f"{s['gpu_hours']} GPU h, {s['episodes_per_gpu_hour']} episodes per GPU hour -> {out}", flush=True)
            return
        if args.hours and s["gpu_hours"] >= args.hours:
            print(f"stopping: {s['gpu_hours']} GPU h reached (--hours {args.hours}): {s['valid_episodes']} valid "
                  f"episodes, {s['episodes_per_gpu_hour']} episodes per GPU hour -> {out}", flush=True)
            return
        if args.max_chains and launched >= args.max_chains:
            print(f"stopping after {launched} chain(s) (--max-chains): {s['valid_episodes']} valid episodes, "
                  f"{s['episodes_per_gpu_hour']} episodes per GPU hour -> {out}", flush=True)
            return
        if dead >= MAX_DEAD_CHAINS:
            print(f"stopping: {dead} chains in a row produced no valid episode; see {out / 'summary.json'}")
            sys.exit(1)
        chain = max(recs) + 1 if recs else 0
        if not args.no_gpu_guard:
            while True:
                busy, why = gpu_busy()
                if not busy:
                    break
                print(f"[{time.strftime('%H:%M')}] GPU busy, waiting {GPU_WAIT_S // 60} min: {why[:300]}", flush=True)
                time.sleep(GPU_WAIT_S)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(out),
               "--seed", str(args.seed), "--chain", str(chain),
               "--max-episodes", str(args.episodes - s["valid_episodes"]),
               "--prior-valid", str(s["valid_episodes"]), "--prior-chained", str(s["chained_episodes"])] + (["--force-chained"] if args.force_chained else [])             + (["--pick-slot", args.pick_slot] if args.pick_slot else [])             + (["--block", args.block] if args.block else []) + (["--fixed-layout"] if args.fixed_layout else [])
        log = out / "logs" / f"chain_{chain:05d}.log"
        print(f"[{time.strftime('%H:%M')}] chain {chain} (seed {chain_seed(args.seed, chain)}): "
              f"{s['valid_episodes']}/{args.episodes} valid so far; log {log}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=CHAIN_TIMEOUT_S,
                                    cwd=str(HERE)).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        rec = load_records(out).get(chain) or {"chain": chain, "seed": chain_seed(args.seed, chain),
                                               "episodes": [], "valid": 0}
        if rec.get("status") == "running":
            rec["status"] = "crashed"
            rec["stop_reason"] = f"worker exit {rc}"
        rec["process_wall_s"] = round(time.time() - t0, 1)
        save_record(out, rec)
        cleanup_chain(out, rec)
        dead = 0 if rec.get("valid", 0) else dead + 1
        launched += 1
        s = write_summary(out, args.episodes)
        print(f"    chain {chain}: {rec.get('status')} ({rec.get('stop_reason', '')}), {rec.get('valid', 0)} valid "
              f"in {rec['process_wall_s'] / 60:.1f} min; total {s['valid_episodes']}/{args.episodes}, "
              f"{s['episodes_per_gpu_hour']} per GPU hour", flush=True)


# =================================================================== worker
class StopChain(Exception):
    pass


def target_stats(recs):
    """Across all chains: targets that keep failing ("bore:3", "tip:J10", ...;
    at least 2 failures or planning rejections and more failures than
    successes) and targets that have succeeded at least once."""
    fail, ok = {}, {}
    for r in recs.values():
        for e in r.get("episodes", []):
            for t in e.get("failed_targets") or ([e["failed_target"]] if e.get("failed_target") else []):
                fail[t] = fail.get(t, 0) + 1
            if e["success"]:
                for t in e.get("targets", []):
                    ok[t] = ok.get(t, 0) + 1
    # only tips keep their reach from layout to layout (the rack's grid);
    # bores, slots and spots move with every arrangement
    bad = {t for t, n in fail.items() if n >= 2 and n > ok.get(t, 0) and t.startswith("tip:")}
    return bad, set(ok)


def sample_layout(rng, survey=None, bad=(), force_pick=None):
    """Everything fixed when the scene is built: where each object stands
    (layout.py), empty slots, liquids, the tube the tube block will move (it
    must be a rigid body), the block the chain runs."""
    import layout
    platform, tries = layout.sample(rng)
    pick = rng.choice(SLOTS)
    if force_pick:                       # testing: "D5" -> (3, 4)
        pick = ("ABCD".index(force_pick[0]), int(force_pick[1:]) - 1)
    others = [s for s in SLOTS if s != pick]
    empty = rng.sample(others, rng.randint(1, 4))
    filled = [s for s in others if s not in empty]
    liquid = rng.sample(filled, rng.randint(2, 4))
    return {
        "pick_slot": pick,
        "empty_slots": sorted(empty),
        "liquids": {tube_name(s): float(rng.randrange(SOURCE_UL[0], SOURCE_UL[1] + 1, 10)) for s in sorted(liquid)},
        "platform": platform, "platform_samples": tries,
        "block_order": rng.choices(list(BLOCK_WEIGHTS), weights=list(BLOCK_WEIGHTS.values()))[:1],
        "look_seed": rng.randrange(2 ** 31),
    }


class ChainRunner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.chain = args.chain
        self.seed = chain_seed(args.seed, args.chain)
        self.rng = random.Random(self.seed)
        self.t0 = time.time()
        self.survey = None
        self.bad, self.proven = target_stats(load_records(self.out))
        self.layout = sample_layout(self.rng, None, self.bad, args.pick_slot)
        if args.block:                       # testing: force the chain's block
            self.layout["block_order"] = [args.block]
        self.rec = {"chain": self.chain, "seed": self.seed, "status": "running", "layout": self.layout,
                    "excluded_targets": sorted(self.bad),
                    "episodes": [], "valid": 0, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_record(self.out, self.rec)
        self.valid = args.prior_valid
        self.chained = args.prior_chained
        self.k = 0

    # ------------------------------------------------------------ scene
    def build(self):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True})

        import lab_scene
        import layout
        lay = self.layout
        layout.apply(lay["platform"])
        lab_scene.EMPTY_HOLDER_SLOTS = {tuple(s) for s in lay["empty_slots"]}
        lab_scene.INITIAL_LIQUIDS = {n: (v, (0.2, 0.3, 0.9)) for n, v in lay["liquids"].items()}

        import appearance
        import isaacsim.core.experimental.utils.stage as stage_utils
        import lab_env
        from pipette import PipetteModel
        from pipette_expert import PipetteExpert
        from planner import Planner

        self.ls, self.le = lab_scene, lab_env
        self.tube = tube_name(lay["pick_slot"])
        # extra_rigid: more tubes as physics bodies (run_scenario.py moves several in one scene)
        present = {tube_name(s) for s in SLOTS if s not in lab_scene.EMPTY_HOLDER_SLOTS}
        extra = tuple(t for t in getattr(self.args, "extra_rigid", ()) if t != self.tube and t in present)
        rigid = (self.tube, *extra, "pipette_0", "tray")
        self.env = lab_env.LabEnv(self.app, rigid_objects=rigid)
        self.env.pipette = PipetteModel(self.env)
        self.planner = Planner(self.app, self.env.arm, [self.env.objects[n]["path"] for n in rigid])
        self.ex = PipetteExpert(self.env, self.planner, seed=self.seed)
        self.look = appearance.apply(stage_utils.get_current_stage(), random.Random(lay["look_seed"]))
        for _ in range(10):        # let the renderer pick up the new materials before recording
            self.app.update()
        # world bookkeeping: holder slot -> tube in it
        self.slot_tube = {s: tube_name(s) for s in SLOTS if tube_name(s) in self.env.objects}
        self.rec["startup_s"] = round(time.time() - self.t0, 1)
        self.rec["look"] = self.look
        self.rec["cameras"] = list(self.env.record_cameras)
        self.survey = self.reach_survey()           # reach depends on this chain's arrangement
        self.rec["reach"] = self.survey
        save_record(self.out, self.rec)

    # ---------------------------------------------------- reach pre-check
    def ik_margin(self, tcp, R):
        q, ok, *_ = self.planner.kin.ik(tcp, R, self.env.q_cmd)
        return self.planner.limit_margin(q) if ok else -1.0

    def slot_xy(self, slot):
        return self.ls.holder_slot_xy(*slot)

    def slot_margin(self, slot):
        """Top-down grasp of a standing tube in this slot, and the hover above."""
        import numpy as np
        from expert import GRASP_DEPTH, HOVER, Z_DOWN
        from planner import frame_from_axis
        R = frame_from_axis(Z_DOWN, self.ex.closing_y)
        top = self.ex.holder_floor + 0.058
        x, y = self.slot_xy(slot)
        return min(self.ik_margin(np.array([x, y, top - GRASP_DEPTH]), R),
                   self.ik_margin(np.array([x, y, top + HOVER]), R))

    def bore_margin(self, idx):
        """Held tube at release, pre-insert and transit, and the removal grasp,
        with that bore's own (tangential) closing axis."""
        return self.ex.bore_margin(idx)

    def tray_margin(self, xy, yaw):
        import numpy as np
        R = self.ex.tray_hand_rotation(xy, yaw)
        if R is None:
            return -1.0
        z = self.ls.BENCH_Z + 0.04 + self.ex.TRAY_GRASP_Z
        return min(self.ik_margin(np.array([*xy, z]), R), self.ik_margin(np.array([*xy, z + 0.1]), R))

    def reach_survey(self):
        s = {"slots": {slot_label(sl): round(self.slot_margin(sl), 3) for sl in SLOTS},
             "bores": {self.bore_label(i): round(self.bore_margin(i), 3) for i in self.ls.CENTRIFUGE_SLEEVE_PARTS},
             "margin_required_rad": REACH_MARGIN}
        print(f"[reach survey] {json.dumps(s)}", flush=True)
        return s

    def bore_label(self, idx):
        import prompts
        return prompts.bore_name(idx, self.ls.CENTRIFUGE_SLEEVE_PARTS)

    # ------------------------------------------------------------ episodes
    def want_chained(self):
        if self.args.force_chained:     # testing: every eligible sequence as one chained episode
            return True
        frac = self.chained / max(1, self.valid)
        p = 0.5 if frac < CHAINED_TARGET[0] else 0.25 if frac < 0.15 else 0.08 if frac < CHAINED_TARGET[1] else 0.0
        return self.rng.random() < p

    def episode(self, name, kw, steps):
        """Record one episode: steps = [(primitive, callable, target)], run
        in order. One step is a primitive episode; several make a chained
        one. target names what the step aims at ("bore:2", "tip:C4", ...)
        for the per-target failure stats.

        Returns True on success. Returns False -- episode discarded -- when
        the failing step recorded no motion of its own: a planning rejection
        (the planner restores the world), so the world is exactly as the
        steps before it left it. self.completed then says how many steps did
        finish (0 for a single primitive, maybe more for a chained one), and
        the caller may carry on from there with another target. A failure
        after motion raises StopChain."""
        import prompts
        if self.valid >= self.args.max_episodes + self.args.prior_valid:
            raise StopChain("episode target reached")
        chained = len(steps) > 1
        wrist = self.le.PRIMITIVE_WRIST_CAMERA[steps[0][0]]     # meta "best_wrist_camera" only
        prompt = prompts.make(name, self.rng, **kw)
        rel = f"{name}/ep_{self.chain:05d}_{self.k:02d}"
        self.k += 1
        targets = [t for _, _, ts in steps for t in as_list(ts)]
        meta = {"primitive": name, "chained": chained, "sub_primitives": [p for p, _, _ in steps], "args": kw,
                "targets": targets, "chain": self.chain, "chain_seed": self.seed, "layout": self.layout,
                "appearance": "random", "appearance_seed": self.layout["look_seed"], "look": self.look}
        self.env.recorder = self.le.EpisodeRecorder(self.out / rel, prompt, meta=meta, wrist_camera=wrist)
        t0 = time.time()
        results, ok, failed_at, failed_target = [], True, None, None
        own_rows = 0
        for prim, fn, target in steps:
            rows0 = len(self.env.recorder.rows)
            try:
                res = fn()
                success, info = bool(res.success), res.info
            except Exception as e:      # an expert/planner exception is a failed episode, not a dead chain
                success, info = False, {"exception": repr(e), "traceback": traceback.format_exc()[-2000:]}
            results.append({"primitive": prim, "success": success, "info": info})
            if not success:
                ok, failed_at, failed_target = False, prim, target
                own_rows = len(self.env.recorder.rows) - rows0
                break
        n = len(self.env.recorder.rows)
        info = results[0]["info"] if not chained else {"steps": results}
        # ...unless planning itself disturbed a held object's grasp (the
        # world is then no longer as it was: end the chain)
        rejected = not ok and own_rows == 0 and not results[-1]["info"].get("held_object_disturbed")
        # A planning rejection is about the step's own target (a bore the hand
        # can't enter); a physical failure is charged to every target the
        # step lists (an insert also to the slot the tube was picked from:
        # a poor grasp there shows up only when the tube meets the bore).
        failed_targets = [] if ok else as_list(failed_target)[:1] if rejected else as_list(failed_target)
        self.completed = len(results) - (0 if ok else 1)
        if rejected:
            shutil.rmtree(self.out / rel, ignore_errors=True)
            rel = None
        else:
            self.env.recorder.finish(ok, info)
        self.env.recorder = None
        if not ok and not rejected:
            dst = self.out / "_failed" / rel
            shutil.rmtree(dst, ignore_errors=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.out / rel), str(dst))
            rel = f"_failed/{rel}"
        elif ok:
            self.valid += 1
            self.chained += chained
            self.rec["valid"] += 1
        self.bad.update(failed_targets)       # not again in this chain
        self.rec["episodes"].append({
            "dir": rel, "name": name, "prompt": prompt, "args": kw, "success": ok, "chained": chained,
            "rejected": rejected, "completed_steps": self.completed, "targets": targets,
            "failed_at": failed_at, "failed_targets": failed_targets,
            "steps": n, "seconds_wall": round(time.time() - t0, 1),
            "info_short": json.dumps(results[-1]["info"], default=float)[:800]})
        save_record(self.out, self.rec)
        write_summary(self.out, self.args.prior_valid + self.args.max_episodes)
        status = ("success" if ok else "FAILED" if not rejected else
                  f"REJECTED by planning at {failed_at} after {self.completed} step(s) (discarded, continuing)")
        print(f"[{rel or name}] {status} steps={n} ({n / self.le.CONTROL_HZ:.1f}s sim, {time.time() - t0:.0f}s wall) "
              f"'{prompt}'\n    {json.dumps(results[-1]['info'], default=float)[:600]}", flush=True)
        if not ok and not rejected:
            raise StopChain(f"{name} failed at {failed_at}")
        return ok

    def must(self, ok, what):
        if not ok:
            raise StopChain(f"{what}: planning rejected, no alternative")

    def first_ok(self, candidates, run, what, tries=4):
        """run(candidate) for candidates in order until one succeeds (a
        planning rejection moves on to the next). Returns the candidate."""
        for c in candidates[:tries]:
            if run(c):
                return c
        raise StopChain(f"{what}: every candidate rejected by planning")

    # -------------------------------------------------------------- blocks
    def block_tubes(self):
        ex, rng = self.ex, self.rng
        S = tuple(self.layout["pick_slot"])
        bores = [i for i in self.ls.CENTRIFUGE_SLEEVE_PARTS
                 if self.survey["bores"].get(self.bore_label(i), -1) > REACH_MARGIN
                 and f"bore:{self.bore_label(i)}" not in self.bad]
        if not bores:
            print("[tubes] no usable bore; skipping the tube block", flush=True)
            return
        rng.shuffle(bores)
        tube, slot = self.tube, slot_label(S)
        # grasp with the first candidate bore's closing axis, so inserting
        # into it is a pure tilt (another bore turns the hand about the tube)
        pick = ("pick_tube_from_holder", lambda: ex.pick_tube_from_holder(tube, ex.closing_for_bore(bores[0])),
                f"pick:{slot}")

        def insert(B):
            return ("insert_tube_into_bore", lambda: ex.insert_tube_into_bore(tube, B),
                    (f"bore:{self.bore_label(B)}", f"pick:{slot}"))

        # Chained episodes use only bores and slots that have worked before; a
        # planning rejection inside one falls back to single-primitive
        # episodes from the step it reached.
        done = 0
        if self.want_chained():
            B = bores[0]
            done = 2 if self.episode("load_tube_into_centrifuge", dict(slot=slot, bore=self.bore_label(B)),
                                     [pick, insert(B)]) else self.completed
        if done == 0:
            self.must(self.episode(pick[0], dict(slot=slot), [pick]), "pick_tube_from_holder")
        if done < 2:
            B = self.first_ok([b for b in bores if f"bore:{self.bore_label(b)}" not in self.bad],
                              lambda b: self.episode("insert_tube_into_bore", dict(bore=self.bore_label(b)),
                                                     [insert(b)]), "insert_tube_into_bore")
        del self.slot_tube[S]
        bore = self.bore_label(B)
        empty = [s for s in SLOTS if s not in self.slot_tube and s != S
                 and self.survey["slots"].get(slot_label(s), -1) > REACH_MARGIN
                 and f"place:{slot_label(s)}" not in self.bad]
        rng.shuffle(empty)
        places = [S] + empty if (not empty or rng.random() < 0.5) else empty + [S]
        remove = ("remove_tube_from_bore", lambda: ex.remove_tube_from_bore(tube, B), f"bore:{bore}")

        def place(T):
            return ("place_tube_into_holder", lambda: ex.place_tube_into_holder(tube, T, ex.closing_for_bore(B)),
                    f"place:{slot_label(T)}")

        done = 0
        if self.want_chained():
            T = places[0]
            done = 2 if self.episode("unload_tube_to_holder", dict(slot=slot_label(T), bore=bore),
                                     [remove, place(T)]) else self.completed
        if done == 0:
            self.must(self.episode(remove[0], dict(bore=bore), [remove]), "remove_tube_from_bore")
        if done < 2:
            T = self.first_ok([t for t in places if f"place:{slot_label(t)}" not in self.bad],
                              lambda t: self.episode("place_tube_into_holder", dict(slot=slot_label(t)),
                                                     [place(t)]), "place_tube_into_holder")
        self.slot_tube[T] = tube

    def volume(self, container):
        L = self.ls.LIQUIDS.get(container)
        return L.volume_ul if L else 0.0

    def plan_transfer(self):
        """Volume, source slots, destination kind and destinations (each list
        in try order), or None if no transfer fits the liquid left."""
        rng = self.rng
        to_well = rng.random() < 0.6
        empty_wells = [w for w in WELLS if self.volume(f"tray/well_{w}") < 0.05 and f"well:{w}" not in self.bad]
        if to_well and not empty_wells:
            to_well = False
        vol = float(rng.choice(WELL_VOLUMES if to_well else TUBE_VOLUMES))
        tubes = {s: t for s, t in self.slot_tube.items() if t != self.tube}   # the moved tube stays dry
        sources = sorted(s for s, t in tubes.items() if self.volume(t) - vol >= MIN_REMAIN_UL
                         and f"src:{slot_label(s)}" not in self.bad)
        if to_well:
            dsts = list(empty_wells)
        else:
            dsts = sorted(s for s, t in tubes.items() if self.volume(t) + vol <= TUBE_MAX_UL
                          and f"dst:{slot_label(s)}" not in self.bad)
        rng.shuffle(sources)
        rng.shuffle(dsts)
        if not sources or not [d for d in dsts if d != sources[0]]:
            return None
        return vol, sources, ("well" if to_well else "tube"), dsts

    def block_pipette(self):
        ex, rng = self.ex, self.rng
        self.must(self.episode("pick_up_pipette", {}, [("pick_up_pipette", ex.pick_up_pipette, "pipette")]),
                  "pick_up_pipette")
        for _ in range(rng.choice([1, 1, 2, 2, 3])):
            tr = self.plan_transfer()
            if tr is None:
                break
            vol, sources, kind, dsts = tr
            v = int(vol)
            tips = [f"tip:{t}" for t in TIPS
                    if f"tip_{t}" not in self.env.pipette.used_tips and f"tip:{t}" not in self.bad]
            rng.shuffle(tips)

            def attach(t):
                return ("attach_tip", lambda: ex.attach_tip(f"tip_{t[4:]}"), t)

            def asp(s):
                return ("aspirate_from_tube", lambda: ex.aspirate_from_tube(self.slot_tube[s], vol),
                        f"src:{slot_label(s)}")

            def disp(d):
                if kind == "well":
                    return ("dispense_into_well", lambda: ex.dispense_into_well(f"well_{d}"), f"well:{d}")
                return ("dispense_into_tube", lambda: ex.dispense_into_tube(self.slot_tube[d]), f"dst:{slot_label(d)}")

            def dkw(d):
                return dict(well=d) if kind == "well" else dict(slot=slot_label(d))

            eject = ("eject_tip", ex.eject_tip, "bin")
            if self.want_chained():
                src = sources[0]
                dst = [d for d in dsts if d != src][0]
                ckw = dict(vol=v, src=slot_label(src), **({"well": dst} if kind == "well" else {"dst": slot_label(dst)}))
                name = "transfer_to_well" if kind == "well" else "transfer_to_tube"
                self.must(self.episode(name, ckw, [attach(tips[0]), asp(src), disp(dst), eject]), name)
            else:
                self.first_ok(tips, lambda t: self.episode("attach_tip", dict(tip=t[4:]), [attach(t)]), "attach_tip")
                src = self.first_ok(sources, lambda s: self.episode("aspirate_from_tube",
                                                                    dict(vol=v, slot=slot_label(s)), [asp(s)]),
                                    "aspirate_from_tube")
                self.first_ok([d for d in dsts if d != src],
                              lambda d: self.episode(disp(d)[0], dkw(d), [disp(d)]), "dispense")
                self.must(self.episode("eject_tip", {}, [eject]), "eject_tip")
        self.must(self.episode("return_pipette", {}, [("return_pipette", ex.return_pipette, "pipette_home")]),
                  "return_pipette")

    def move_to_start(self):
        """Unrecorded (call between episodes): bring the open hand to a pick
        episode's start -- the home pose exactly (lab_scene.Q_HOME, as an eval
        trial starts) or, else, a random pose near it. Returns the start used,
        or None if the arm could not get there (the caller carries on from
        where it is)."""
        import numpy as np
        from planner import Plan, Waypoint
        assert self.env.recorder is None
        ex, rng = self.ex, self.rng
        q_home = np.asarray(self.ls.Q_HOME[:7], float)
        tcp_h, R_h = self.planner.kin.tcp(q_home)
        open_m = self.ls.GRIPPER_STROKE
        wp = Waypoint("to_home", np.asarray(tcp_h), R_h, open_m, speed=0.15)
        if not ex.plan_and_go([wp]).success:
            print("[start] home pose not reachable from here; starting where the arm is", flush=True)
            return None
        # same hand pose; settle the arm's null space onto Q_HOME exactly
        ex.execute(Plan(segments=[(wp, [q_home], 0.0, 0.0)]))
        ex.gripper(0.0, 5)
        if rng.random() < START_HOME_P:
            return "home"
        a = math.radians(rng.uniform(-START_YAW_DEG, START_YAW_DEG))
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        d = np.array([rng.uniform(*START_OFFSET_M[k]) for k in "xyz"])
        if ex.plan_and_go([Waypoint("to_start", tcp_h + d, Rz @ R_h, open_m, speed=0.15)]).success:
            return "near_home"
        return "home"

    def block_tube_picks(self):
        """Focused data: up to TUBE_PICKS_PER_CHAIN cycles of start pose ->
        pick the chain's tube from its slot (recorded) -> put it into a slot
        (recorded; its own or another empty one), so one chain gives many
        tube picks that start at or near home."""
        ex, rng = self.ex, self.rng
        S = tuple(self.layout["pick_slot"])
        tube = self.tube
        for k in range(TUBE_PICKS_PER_CHAIN):
            start = self.move_to_start()
            slot = slot_label(S)
            pick = ("pick_tube_from_holder", lambda: ex.pick_tube_from_holder(tube), f"pick:{slot}")
            if not self.episode("pick_tube_from_holder", dict(slot=slot, start=start), [pick]):
                print(f"[tube_picks] pick from {slot} rejected; ending the block", flush=True)
                return
            del self.slot_tube[S]
            empty = [s for s in SLOTS if s not in self.slot_tube and s != S
                     and self.survey["slots"].get(slot_label(s), -1) > REACH_MARGIN]
            rng.shuffle(empty)
            places = ([S] + empty) if rng.random() < 0.5 else (empty + [S])
            T = self.first_ok(places, lambda T: self.episode(
                "place_tube_into_holder", dict(slot=slot_label(T)),
                [("place_tube_into_holder", lambda: ex.place_tube_into_holder(tube, T), f"place:{slot_label(T)}")]),
                "place_tube_into_holder", tries=3)
            self.slot_tube[T] = tube
            S = T

    def block_tray(self):
        import numpy as np
        import prompts
        from planner import pose_matrix
        ex, rng = self.ex, self.rng
        T0 = pose_matrix(ex.pose("tray"))
        home_xy, home_yaw = T0[:2, 3].copy(), math.atan2(T0[1, 0], T0[0, 0])
        import layout
        # destinations beside another object ("to the left of the tip box"),
        # one landmark/side per chain so a failed place can retry nearby
        spots = [(xy, yaw, key) for xy, yaw, key in layout.tray_spots(rng, n=12)
                 if self.tray_margin(xy, yaw) > REACH_MARGIN]
        if not spots:
            print("[tray] no reachable spot beside another object found; skipping the tray block", flush=True)
            return
        spot = spots[0][2]
        spots = [(xy, yaw) for xy, yaw, key in spots if key == spot][:6]

        pick = ("pick_up_tray", ex.pick_up_tray, "tray")

        def where(xy, yaw):
            return dict(xy=[round(float(v), 4) for v in xy], yaw_deg=round(math.degrees(yaw), 2))

        def place(xy, yaw, spot):
            return ("place_tray", lambda: ex.place_tray(xy, yaw), f"tray:{spot}")

        def move(spot, cands, fallback=None):
            """Tray to one of cands [(xy, yaw)]: a chained move_tray episode or
            pick_up_tray + place_tray; a rejected place tries the next spot.
            If every spot is rejected, set it down at fallback (its own start
            pose, always reachable) instead of ending the chain; returns
            whether it went to `spot`."""
            done = 0
            if self.want_chained():
                xy, yaw = cands[0]
                if self.episode("move_tray", dict(dest=prompts.tray_spot(spot, dest=True), **where(xy, yaw)),
                                [pick, place(xy, yaw, spot)]):
                    return
                done = self.completed
                if done:
                    cands = cands[1:]
            if done == 0:
                self.must(self.episode("pick_up_tray", {}, [pick]), "pick_up_tray")
            try:
                self.first_ok(cands, lambda c: self.episode("place_tray", dict(spot=prompts.tray_spot(spot),
                                                                               **where(*c)), [place(*c, spot)]),
                              f"place_tray {spot}", tries=len(cands))
                return True
            except StopChain as e:
                if fallback is None or "every candidate rejected" not in str(e):
                    raise
            # the arm can't set it down at any of these (it would
            # hit a neighbour); put it back where it was picked up
            self.first_ok([fallback], lambda c: self.episode(
                "place_tray", dict(spot=prompts.tray_spot("home"), **where(*c)), [place(*c, "home")]),
                "place_tray home (fallback)")
            return False

        if move(spot, spots, fallback=(home_xy, home_yaw)) and rng.random() < 0.7:
            move("home", [(home_xy, home_yaw)])

    def run(self):
        try:
            self.build()
            for block in self.layout["block_order"]:
                getattr(self, f"block_{block}")()
            self.rec["status"], self.rec["stop_reason"] = "done", "chain complete"
        except StopChain as e:
            self.rec["status"] = "done" if "target" in str(e) else "failed"
            self.rec["stop_reason"] = str(e)
        except Exception as e:
            self.rec["status"], self.rec["stop_reason"] = "crashed", repr(e)
            traceback.print_exc()
        self.rec["wall_s"] = round(time.time() - self.t0, 1)
        save_record(self.out, self.rec)
        write_summary(self.out, self.args.prior_valid + self.args.max_episodes)
        print(f"chain {self.chain}: {self.rec['status']} ({self.rec['stop_reason']}), {self.rec['valid']} valid "
              f"episodes in {self.rec['wall_s'] / 60:.1f} min", flush=True)
        sys.stdout.flush()
        if hasattr(self, "app"):
            self.app.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=int, default=20, help="valid episodes to have under --out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-gpu-guard", action="store_true", help="don't wait for other sessions' GPU jobs")
    ap.add_argument("--max-chains", type=int, default=0, help="stop after this many chains (0: until --episodes)")
    ap.add_argument("--hours", type=float, default=0.0, help="stop once the chains' process time reaches this (GPU h)")
    # worker (one chain, launched by the driver)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--chain", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--max-episodes", type=int, default=10 ** 9, help=argparse.SUPPRESS)
    ap.add_argument("--prior-valid", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--prior-chained", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--force-chained", action="store_true", help=argparse.SUPPRESS)   # testing only
    ap.add_argument("--pick-slot", default=None, help=argparse.SUPPRESS)              # testing only, e.g. D5
    ap.add_argument("--block", default=None, choices=["tubes", "pipette", "tray", "tube_picks"],
                    help="run only this block in every chain (tube_picks: repeated tube picks from a start pose)")
    ap.add_argument("--fixed-layout", action="store_true",
                    help="every chain in the default arrangement (the pilot's), not a sampled layout")
    args = ap.parse_args()
    if args.worker:
        if args.fixed_layout:                # every chain in the default (pilot) arrangement
            import layout
            layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)
        ChainRunner(args).run()
    else:
        drive(args)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
