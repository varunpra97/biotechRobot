"""Experiment description -> high-level commands -> primitive steps for the simulated lab bench.
Pure Python (no simulator): used by the web app (app.py) to show a plan and by run_protocol.py to execute it.

Planning here is RULE-BASED (keyword patterns), not a language model: it understands the kinds of sentences listed
in EXAMPLES and says which parts it did not understand. A model-based planner can replace `parse()`: it only has to
return the same list of command dicts.

High-level commands:
  {"cmd": "transfer", "source": "A1", "dest": {"tube": "C3"} | {"well": "A1"}, "ul": 200}
  {"cmd": "centrifuge", "tube": "C3", "seconds": 10}
  {"cmd": "wait", "seconds": 5}
  {"cmd": "move_tray"}
Primitive steps (what the robot executes, one after another, in one continuous simulation):
  pick_up_pipette, attach_tip, aspirate_from_tube, dispense_into_tube, dispense_into_well, eject_tip, return_pipette,
  pick_tube_from_holder, insert_tube_into_bore, wait, remove_tube_from_bore, place_tube_into_holder,
  pick_up_tray, place_tray
"""
from __future__ import annotations

import copy
import re

ROWS = "ABCD"
SLOTS = [f"{r}{c}" for r in ROWS for c in range(1, 6)]            # tube rack: 4 rows x 5
WELLS = [f"{r}{c}" for r in "AB" for c in range(1, 6)]            # tray: 2 rows x 5 wells
TIPS = [f"{r}{c}" for r in "ABCDEFGHIJ" for c in range(1, 11)]

# Limits of the simulated pipetting (from the generator, sim/collect_primitives.py)
MIN_REMAIN_UL = 600.0      # a tube can be aspirated from only while this much stays in it (the tip reaches no deeper)
TUBE_MAX_UL = 1100.0       # a tube is not filled above this
TIP_MAX_UL = 300.0
WELL_MAX_UL = 250.0
MAX_WAIT_S = 15.0          # waits are compressed to at most this many simulated seconds

DEFAULT_BENCH = {
    "tubes": {
        "A1": {"name": "solvent A", "ul": 1300.0, "color": [0.95, 0.55, 0.10]},
        "A2": {"name": "solvent B", "ul": 1300.0, "color": [0.15, 0.70, 0.30]},
        "A3": {"name": "buffer", "ul": 1300.0, "color": [0.60, 0.30, 0.85]},
        "C3": {"name": "test tube", "ul": 700.0, "color": [0.20, 0.30, 0.90]},
    },
    "test_tube": "C3",
    "empty_slots": ["B5", "D4"],
}
EXAMPLES = [
    "Load solvent A into the test tube, place the test tube in the centrifuge for some time, remove the solvent "
    "using the pipette and place the solvent in the tray from the pipette.",
    "Add 150 uL of solvent B to the test tube. Centrifuge the test tube for 10 seconds. "
    "Transfer 200 uL from the test tube to well B3.",
    "Transfer 100 uL of buffer to well A1 and 100 uL of solvent A to well A2.",
]


class PlanError(ValueError):
    pass


# ------------------------------------------------------------------ text -> commands
_NUM = r"(\d+(?:\.\d+)?)"


def _volume(clause, default=200.0):
    m = re.search(_NUM + r"\s*(ul|µl|microliters?|microlitres?|ml|milliliters?|millilitres?)\b", clause)
    if not m:
        return default, False
    v = float(m.group(1))
    return (v * 1000.0 if m.group(2).startswith("m") and not m.group(2).startswith("micro") else v), True


def _seconds(clause, default=10.0):
    m = re.search(_NUM + r"\s*(s|sec|secs|seconds?|min|mins|minutes?)\b", clause)
    if not m:
        return default, False
    v = float(m.group(1))
    return (v * 60.0 if m.group(2).startswith("min") else v), True


def _slot(text):
    m = re.search(r"\b(?:tube|slot)\s+([a-d][1-5])\b", text)
    return m.group(1).upper() if m else None


def _well(text):
    m = re.search(r"\bwell\s+([ab][1-5])\b", text)
    return m.group(1).upper() if m else None


def _reagent(clause, bench):
    """Slot of a stock tube named in the clause (longest name first), or None."""
    for slot, t in sorted(bench["tubes"].items(), key=lambda kv: -len(kv[1]["name"])):
        if slot != bench["test_tube"] and re.search(r"\b" + re.escape(t["name"].lower()) + r"\b", clause):
            return slot
    return None


def parse(text, bench=None):
    """Free text -> (commands, notes). notes lists what was assumed and what was not understood."""
    bench = bench or DEFAULT_BENCH
    test = bench["test_tube"]
    t = " " + re.sub(r"\s+", " ", text.lower().replace("µ", "u")) + " "
    parts = re.split(r"[.;\n]|,\s*(?:and\s+)?then\b|\band then\b|\bthen\b|,|\band\b(?=\s+(?:place|put|dispense|remove|"
                     r"take|add|load|transfer|centrifuge|spin|wait|move|aspirate|draw|collect|pipette)\b)", t)
    cmds, notes, pending = [], [], None        # pending: an aspirate waiting for its destination
    for raw in parts:
        c = raw.strip()
        if not c:
            continue
        vol, vol_given = _volume(c)
        has = lambda *w: any(re.search(r"\b" + x + r"\b", c) for x in w)       # noqa: E731
        to_tray = has("tray", "well", "wells", "plate")
        if has("centrifuge", "centrifuging", "spin", "spun"):
            secs, given = _seconds(c)
            if not given:
                notes.append(f'"{c}": no time given, using {secs:.0f} s')
            cmds.append({"cmd": "centrifuge", "tube": _slot(c) or test, "seconds": secs, "text": c})
        elif has("wait", "pause", "incubate", "rest") and not to_tray:
            secs, _ = _seconds(c, 5.0)
            cmds.append({"cmd": "wait", "seconds": secs, "text": c})
        elif has("move", "shift", "relocate") and has("tray") and not has("solvent", "liquid", "ul", "sample", "buffer"):
            cmds.append({"cmd": "move_tray", "text": c})
        elif to_tray and has("place", "put", "dispense", "transfer", "deposit", "drop", "move", "add", "pipette", "load"):
            well = _well(c)
            src = _reagent(c, bench) or _slot(c)
            if src is None and pending is not None:
                src, vol, vol_given = pending["source"], (vol if vol_given else pending["ul"]), True
            elif src is None:
                src = test
            if not vol_given:
                notes.append(f'"{c}": no volume given, using {vol:.0f} uL')
            cmds.append({"cmd": "transfer", "source": src, "dest": {"well": well}, "ul": vol, "text":
                         (pending["text"] + " + " if pending else "") + c})
            pending = None
        elif has("remove", "aspirate", "draw", "take", "collect", "withdraw", "extract", "suck"):
            if not vol_given:
                notes.append(f'"{c}": no volume given, using {vol:.0f} uL')
            pending = {"source": _slot(c) or test, "ul": vol, "text": c}
        elif has("load", "add", "put", "transfer", "pipette", "dispense", "pour", "fill"):
            src = _reagent(c, bench)
            m = re.search(r"\bfrom\s+(?:the\s+)?(?:tube|slot)\s+([a-d][1-5])\b", c)
            if m:
                src = m.group(1).upper()
            if src is None:
                notes.append(f'NOT UNDERSTOOD: "{c}" (which liquid? known: '
                             + ", ".join(v["name"] for k, v in bench["tubes"].items() if k != test) + ")")
                continue
            m = re.search(r"\b(?:into|to|in)\s+(?:the\s+)?(?:tube|slot)\s+([a-d][1-5])\b", c)
            dest = m.group(1).upper() if m else test
            if not vol_given:
                notes.append(f'"{c}": no volume given, using {vol:.0f} uL')
            cmds.append({"cmd": "transfer", "source": src, "dest": {"tube": dest}, "ul": vol, "text": c})
        else:
            notes.append(f'NOT UNDERSTOOD: "{c}"')
    if pending is not None:
        notes.append(f'"{pending["text"]}": no destination given, dispensing into the next free well')
        cmds.append({"cmd": "transfer", "source": pending["source"], "dest": {"well": None}, "ul": pending["ul"],
                     "text": pending["text"]})
    return cmds, notes


# ------------------------------------------------------------------ commands -> primitive steps
def describe(cmd, bench=None):
    bench = bench or DEFAULT_BENCH
    name = lambda s: bench["tubes"].get(s, {}).get("name", "tube") + f" ({s})"      # noqa: E731
    if cmd["cmd"] == "transfer":
        d = cmd["dest"]
        to = name(d["tube"]) if "tube" in d else f"tray well {d['well'] or '(next free)'}"
        return f"Transfer {cmd['ul']:.0f} uL from {name(cmd['source'])} to {to}"
    if cmd["cmd"] == "centrifuge":
        return f"Centrifuge {name(cmd['tube'])} for {cmd['seconds']:.0f} s"
    if cmd["cmd"] == "wait":
        return f"Wait {cmd['seconds']:.0f} s"
    return "Pick up the tray and set it down again"


def expand(cmds, bench=None):
    """Commands -> primitive steps, checking every step against the tracked state of the bench.
    Returns (steps, final volumes). Raises PlanError with a plain message when a command cannot be done."""
    bench = copy.deepcopy(bench or DEFAULT_BENCH)
    tubes = {s: float(t["ul"]) for s, t in bench["tubes"].items()}
    present = [s for s in SLOTS if s not in bench["empty_slots"]]
    wells = {w: 0.0 for w in WELLS}
    st = {"hold": None, "tips": 0}
    steps = []

    def add(k, prim, **kw):
        steps.append({"command": k, "prim": prim, **kw})

    def free_hand(k):
        if st["hold"] == "pipette":
            add(k, "return_pipette")
            st["hold"] = None

    for k, c in enumerate(cmds):
        label = describe(c, bench)
        if c["cmd"] == "transfer":
            src, ul, d = c["source"], float(c["ul"]), c["dest"]
            if src not in present:
                raise PlanError(f"{label}: there is no tube in slot {src}")
            if ul <= 0 or ul > TIP_MAX_UL:
                raise PlanError(f"{label}: one transfer moves between 1 and {TIP_MAX_UL:.0f} uL")
            if tubes.get(src, 0.0) - ul < MIN_REMAIN_UL:
                raise PlanError(f"{label}: slot {src} holds {tubes.get(src, 0.0):.0f} uL; the tip only reaches liquid "
                                f"while at least {MIN_REMAIN_UL:.0f} uL stay in the tube")
            if "tube" in d:
                if d["tube"] not in present:
                    raise PlanError(f"{label}: there is no tube in slot {d['tube']}")
                if d["tube"] == src:
                    raise PlanError(f"{label}: source and destination are the same tube")
                if tubes.get(d["tube"], 0.0) + ul > TUBE_MAX_UL:
                    raise PlanError(f"{label}: slot {d['tube']} would hold more than {TUBE_MAX_UL:.0f} uL")
            else:
                if ul > WELL_MAX_UL:
                    raise PlanError(f"{label}: a tray well takes at most {WELL_MAX_UL:.0f} uL")
                if d["well"] is None:
                    free = [w for w in WELLS if wells[w] < 0.05]
                    if not free:
                        raise PlanError(f"{label}: every tray well is in use")
                    d["well"] = free[0]
                if wells[d["well"]] + ul > WELL_MAX_UL:
                    raise PlanError(f"{label}: well {d['well']} would hold more than {WELL_MAX_UL:.0f} uL")
            if st["tips"] >= len(TIPS):
                raise PlanError("no unused pipette tips left")
            if st["hold"] != "pipette":
                add(k, "pick_up_pipette")
                st["hold"] = "pipette"
            add(k, "attach_tip", tip=TIPS[st["tips"]])
            st["tips"] += 1
            add(k, "aspirate_from_tube", slot=src, ul=ul)
            tubes[src] -= ul
            if "tube" in d:
                add(k, "dispense_into_tube", slot=d["tube"])
                tubes[d["tube"]] = tubes.get(d["tube"], 0.0) + ul
            else:
                add(k, "dispense_into_well", well=d["well"])
                wells[d["well"]] += ul
            add(k, "eject_tip")
        elif c["cmd"] == "centrifuge":
            if c["tube"] not in present:
                raise PlanError(f"{label}: there is no tube in slot {c['tube']}")
            free_hand(k)
            secs = min(float(c["seconds"]), MAX_WAIT_S)
            add(k, "pick_tube_from_holder", slot=c["tube"])
            add(k, "insert_tube_into_bore", slot=c["tube"])
            add(k, "wait", seconds=secs, asked_seconds=float(c["seconds"]), note="tube in the centrifuge (spin not simulated)")
            add(k, "remove_tube_from_bore", slot=c["tube"])
            add(k, "place_tube_into_holder", slot=c["tube"])
        elif c["cmd"] == "wait":
            add(k, "wait", seconds=min(float(c["seconds"]), MAX_WAIT_S), asked_seconds=float(c["seconds"]), note="waiting")
        elif c["cmd"] == "move_tray":
            free_hand(k)
            add(k, "pick_up_tray")
            add(k, "place_tray")
        else:
            raise PlanError(f"unknown command {c['cmd']}")
    if steps:
        free_hand(steps[-1]["command"])
    final = {"tubes": {s: round(v, 1) for s, v in tubes.items()}, "wells": {w: round(v, 1) for w, v in wells.items() if v > 0}}
    return steps, final


def plan(text, bench=None):
    """Everything the UI needs for one description."""
    bench = bench or DEFAULT_BENCH
    cmds, notes = parse(text, bench)
    out = {"text": text, "bench": bench, "commands": cmds, "notes": notes, "steps": [], "final": None, "error": None,
           "descriptions": [describe(c, bench) for c in cmds]}
    if not cmds:
        out["error"] = "Nothing in the description matched a command the bench can do."
        return out
    try:
        out["steps"], out["final"] = expand(cmds, bench)
        out["descriptions"] = [describe(c, bench) for c in cmds]
    except PlanError as e:
        out["error"] = str(e)
    return out


if __name__ == "__main__":
    import json
    import sys
    p = plan(" ".join(sys.argv[1:]) or EXAMPLES[0])
    print(json.dumps({k: p[k] for k in ("descriptions", "notes", "error", "final")}, indent=1))
    for s in p["steps"]:
        print(s)
