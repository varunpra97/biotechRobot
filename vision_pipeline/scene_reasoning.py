"""LLM scene reasoning: turn a reconstructed blob plus a few photos into an
educated guess about WHAT it is, so the geometry stages can measure it
precisely.

Per object, Claude gets: the user's scene context (what kind of scene,
which objects to expect), the coarse measured size from the visual hull,
the CLIP label hints, and full-resolution crops from the views where the
object is largest. It can search the web for standards and product specs.
It returns structured JSON (schema below):

  - identity, and the shape family to fit (template fitting needs this;
    a visual hull alone can't see bores, wells or an open bin's interior)
  - expected dimensions with their source (a standard or a manufacturer
    spec is far more precise than anything measured from photos)
  - decomposition: is this several objects in contact (tubes standing in
    a holder, tips hanging in a rack)? which parts, how many, in what
    grid, and can they be removed independently?
  - material, density and friction, for the USD physics properties
  - solid_fraction: how much of the reconstructed volume is material
    (an open bin is mostly air), for the mass estimate

Everything here is a hint for the measurement stages, never a substitute:
dimensions are still fitted to the images, and a guess the images
contradict loses.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

DEFAULT_KEY_FILE = Path.home() / ".anthropic" / "api_key"


def make_client():
    """Anthropic client. Credentials, first match wins: ANTHROPIC_API_KEY /
    ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile (the SDK resolves
    these itself), else a key file the user created outside the repo
    (ANTHROPIC_API_KEY_FILE, default ~/.anthropic/api_key). The key is
    passed to the client only -- never printed, logged or written."""
    import anthropic
    if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_FEDERATION_RULE_ID") or os.environ.get("ANTHROPIC_PROFILE")):
        return anthropic.Anthropic()      # key, token, workload identity federation or a named profile
    key_file = Path(os.environ.get("ANTHROPIC_API_KEY_FILE", DEFAULT_KEY_FILE))
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8-sig").strip()
        if key:
            return anthropic.Anthropic(api_key=key)
    return anthropic.Anthropic()          # an `ant auth login` profile, if any

SHAPE_FAMILIES = [
    "box",                    # solid rectangular block
    "open_box",               # box with an open top / hollow interior (bin, tray)
    "cylinder",               # solid or closed cylinder
    "tube_with_conical_tip",  # test tube / centrifuge tube
    "rack_with_bore_grid",    # block with a regular grid of blind bores (tube holder)
    "plate_with_well_grid",   # flat plate with a regular grid of wells (sample tray, microplate)
    "rotationally_symmetric", # lathe-like body (centrifuge housing, bottle)
    "freeform",               # anything else; measured as-is, no template
]

_SOURCES = ["standard", "manufacturer_spec", "typical_for_category", "visual_estimate"]

# How a part sits relative to its parent -- decompose.py turns each into a
# geometric split of the fused object.
RELATIONS = ["base", "inserted", "resting_inside", "resting_on_top"]

_MATERIAL = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "density_kg_m3": {"type": "number"},
        "static_friction": {"type": "number"},
        "dynamic_friction": {"type": "number"},
        "restitution": {"type": "number"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string", "description": "what in the photos indicates this material"},
    },
    "required": ["name", "density_kg_m3", "static_friction", "dynamic_friction", "restitution",
                 "confidence", "evidence"],
    "additionalProperties": False,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "identity": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "standard_or_product": {"type": "string", "description": "empty string if none identified"},
                "confidence": {"type": "number"},
            },
            "required": ["name", "category", "standard_or_product", "confidence"],
            "additionalProperties": False,
        },
        "shape_family": {"type": "string", "enum": SHAPE_FAMILIES},
        "expected_dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value_mm": {"type": "number"},
                    "tolerance_mm": {"type": "number"},
                    "source": {"type": "string", "enum": _SOURCES},
                    "source_detail": {"type": "string"},
                },
                "required": ["name", "value_mm", "tolerance_mm", "source", "source_detail"],
                "additionalProperties": False,
            },
        },
        "decomposition": {
            "type": "object",
            "properties": {
                "is_composite": {"type": "boolean"},
                "parts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "count": {"type": "integer"},
                            "shape_family": {"type": "string", "enum": SHAPE_FAMILIES},
                            "removable": {"type": "boolean"},
                            "grid_rows": {"type": "integer", "description": "0 if not in a grid"},
                            "grid_cols": {"type": "integer", "description": "0 if not in a grid"},
                            "pitch_mm": {"type": "number", "description": "0 if not in a grid"},
                            "relation": {"type": "string", "enum": RELATIONS},
                            "color_hex": {"type": "string",
                                          "description": "#rrggbb dominant colour of this part in the photos"},
                            "parent": {"type": "string", "description": "name of the part it sits in/on; empty for the base"},
                            "material": _MATERIAL,
                            "solid_fraction": {
                                "type": "number",
                                "description": "fraction of this part's FILLED outer envelope that is material (0-1)"},
                            "notes": {"type": "string"},
                        },
                        "required": ["name", "count", "shape_family", "removable", "grid_rows", "grid_cols",
                                     "pitch_mm", "relation", "color_hex", "parent", "material", "solid_fraction",
                                     "notes"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["is_composite", "parts", "reasoning"],
            "additionalProperties": False,
        },
        "material": _MATERIAL,
        "solid_fraction": {
            "type": "number",
            "description": "fraction of the object's outer envelope volume that is material (0-1)",
        },
        "notes": {"type": "string"},
    },
    "required": ["identity", "shape_family", "expected_dimensions", "decomposition",
                 "material", "solid_fraction", "notes"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You identify physical objects in a photographed scene so that a 3D reconstruction pipeline can build accurate simulation assets (geometry, collision, mass, material) for them.

For each object you receive: a description of the scene and the kinds of objects it may contain, its measured dimensions and volumes from the reconstruction (the object summary says how it was reconstructed and how far to trust it), weak label hints from an image classifier, and photo crops of the object from several angles.

What the pipeline needs from you:
- What the object is, and which shape family describes it.
- Dimensions you can source more precisely than the measurement allows. If you recognise a standard (for example an SBS/ANSI microplate footprint) or a specific product, use web search to find its published dimensions and cite the source. Otherwise give category-typical values marked as such, with honest tolerances.
- Whether the reconstructed object is really several separate objects in contact. Always list its parts: exactly one with relation "base" (the holder, container, housing or the whole object if it is a single item), and one entry per kind of removable part, with its relation to its parent:
  - "inserted": stands in a bore or hole of its parent (tubes in a rack, pins in a board, tubes in a rotor)
  - "resting_inside": stands inside an open parent (weights in a tray, a rotor in a centrifuge bowl)
  - "resting_on_top": sits on its parent (a lid, a cap)
  Parts can nest (tubes inserted in a rotor that rests inside a housing): set "parent" to the parent part's name. Count what you can see in the crops; do not assume a rack is full. When parts in the same role differ in material or colour (a brass weight and a steel weight in one tray), list them as separate entries, each with its own count and colour -- the pipeline tells them apart by colour.
- Each part's material, judged from its appearance (colour, gloss, metallic sheen, wood grain, translucency) and its role, with typical density and friction values, and what in the photos you based it on.
- Each part's solid_fraction: the fraction of the part's FILLED outer envelope that is material. The pipeline computes the filled envelope by closing every cavity, including openings and hollows (a tube's bore, a bin's interior, a bottle's inside). So a solid metal cylinder is about 1.0, an open thin-walled tray or a thin-walled tube perhaps 0.1-0.35, a closed hollow housing low, a block with a few bores slightly below 1. Estimate wall thicknesses from the crops and typical products.

If the crops contradict the scene description, trust the crops and say so in notes."""


def _image_block(jpeg_bytes: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpeg_bytes).decode("ascii")}}


def build_user_content(scene_context: str, object_summary: dict, crops: list[bytes]) -> list[dict]:
    text = (
        f"Scene context:\n{scene_context}\n\n"
        f"Object under analysis (id {object_summary['id']}):\n"
        f"{json.dumps({k: v for k, v in object_summary.items() if k != 'id'}, indent=2)}\n\n"
        f"The {len(crops)} images below are crops of this object from different camera angles."
    )
    return [{"type": "text", "text": text}] + [_image_block(c) for c in crops]


def analyze_object(client, scene_context: str, object_summary: dict, crops: list[bytes],
                   model: str = "claude-opus-5-5", effort: str = "high",
                   web_search_uses: int = 5, max_pause_resumes: int = 3) -> dict:
    """One structured-output request (with web search) per object.

    Returns {"status": "ok", "analysis": {...}} or {"status": "refused"|"incomplete", ...}.
    Refusals are retried server-side on Anthropic's recommended fallback
    model (fallbacks="default"); if the whole chain declines, the object
    falls back to plain geometric measurement downstream.
    """
    messages = [{"role": "user", "content": build_user_content(scene_context, object_summary, crops)}]
    for _ in range(max_pause_resumes + 1):
        with client.beta.messages.stream(
            model=model,
            max_tokens=32000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": web_search_uses}],
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        if response.stop_reason == "pause_turn":
            # Server-side tool loop hit its iteration limit: resend with the
            # paused turn appended and the API resumes where it stopped.
            messages.append({"role": "assistant", "content": response.content})
            continue
        if response.stop_reason == "refusal":
            details = response.stop_details
            return {"status": "refused", "category": getattr(details, "category", None),
                    "explanation": getattr(details, "explanation", None)}
        if response.stop_reason == "max_tokens":
            return {"status": "incomplete", "reason": "max_tokens"}
        text = next((b.text for b in reversed(response.content) if b.type == "text"), None)
        if text is None:
            return {"status": "incomplete", "reason": "no text block"}
        return {"status": "ok", "model": response.model, "analysis": json.loads(text),
                "usage": {"input_tokens": response.usage.input_tokens,
                          "output_tokens": response.usage.output_tokens}}
    return {"status": "incomplete", "reason": f"still paused after {max_pause_resumes} resumes"}


# ------------------------------------------------------------ Claude Code backend
# The same request, run through headless Claude Code (`claude -p`) on the
# user's Claude subscription login instead of the API (no API credits).
# Locked down: only the Read tool (to view the saved crops) and WebSearch,
# restricted mode (file tools confined to the crop folder, no code-running
# tools), no permission prompts, no session saved.

def find_claude_cli() -> str | None:
    """CLAUDE_CLI, then `claude` on PATH, then the native installer's
    location, then the newest copy bundled with the Claude desktop app."""
    import glob
    import shutil
    for c in (os.environ.get("CLAUDE_CLI"), shutil.which("claude"),
              str(Path.home() / ".local" / "bin" / "claude.exe"), str(Path.home() / ".local" / "bin" / "claude")):
        if c and Path(c).exists():
            return c
    appdata = os.environ.get("APPDATA")
    if appdata:
        found = sorted(glob.glob(os.path.join(appdata, "Claude", "claude-code", "*", "*", "claude.exe")))
        if found:
            return found[-1]
    return None


def analyze_object_cli(cli: str, scene_context: str, object_summary: dict, crop_paths: list[Path],
                       model: str = "claude-opus-5-5", effort: str = "high", timeout_s: int = 900) -> dict:
    """One headless Claude Code run per object. Returns the same shape as
    analyze_object(): {"status": "ok", "analysis": {...}, ...} or an error."""
    import subprocess
    crop_dir = Path(crop_paths[0]).parent.resolve()
    text = build_user_content(scene_context, object_summary, [b""] * len(crop_paths))[0]["text"]
    prompt = (text.replace("The {} images below are".format(len(crop_paths)),
                           "The {} images listed below are".format(len(crop_paths)))
              + "\n\nView every image with the Read tool before answering:\n"
              + "\n".join(str(Path(p).resolve()) for p in crop_paths)
              + "\n\nAnswer only through the required structured output.")
    cmd = [cli, "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
           "--model", model, "--effort", effort, "--tools", "Read,WebSearch", "--allowedTools", "Read,WebSearch",
           "--permission-mode", "dontAsk", "--permission-prompts", "none", "--restricted",
           "--no-session-persistence", "--add-dir", str(crop_dir), "--system-prompt", SYSTEM_PROMPT]
    # A clean environment: this pipeline may itself run inside a Claude Code
    # session, whose session variables must not leak into the child run.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE_CODE_", "CLAUDE_PID", "CLAUDE_EFFORT",
                                                                     "CLAUDE_PREVIEW"))}
    try:
        proc = subprocess.run(cmd, cwd=crop_dir, env=env, capture_output=True, text=True, encoding="utf-8",
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "incomplete", "reason": f"claude -p timed out after {timeout_s}s"}
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "error", "reason": "no JSON from claude -p", "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-2000:]}
    base = {"backend": "claude-code-cli", "model": model, "session_id": out.get("session_id"),
            "num_turns": out.get("num_turns"), "estimated_cost_usd": out.get("total_cost_usd"),
            "usage": out.get("usage"), "duration_ms": out.get("duration_ms")}
    if out.get("is_error") or out.get("structured_output") is None:
        msg = out.get("result", "")
        status = "not_logged_in" if "logged in" in str(msg).lower() or "/login" in str(msg) else "error"
        return {"status": status, "reason": msg, "terminal_reason": out.get("terminal_reason"), **base}
    return {"status": "ok", "analysis": out["structured_output"], **base}
