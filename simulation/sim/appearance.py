"""How the lab looks: physically based materials, lab lighting, and seeded
appearance randomization for training data.

Why: the scene was flat display colours on a white bench under a bright
uniform dome light, so glass tubes and the containers a pipette tip works
in all but vanished against the bench. Real lab benches are mostly dark
chemical-resistant epoxy or phenolic resin, tubes are clear glass, and
labs are lit by ceiling panels that cast soft shadows.

  apply(stage, rng=None)   baseline look (rng=None) or a randomized one

Randomized per call (seeded, so an episode's look is reproducible): bench
top among real lab surfaces, wall and floor colours, the two ceiling panels'
intensity / colour temperature / position, ambient fill, the colours of
labware that comes in several colours (racks, tip boxes, trays, bins,
pipette body) and the liquid colours. Geometry, physics and the robot are
never touched, and test tubes stay clear glass.

Materials are UsdPreviewSurface, bound stronger than descendants on each
object's prim (its rendered child meshes inherit it; collision meshes are
guide-purpose and never rendered). Liquids keep their per-mesh display
colour, which the pipette model changes as liquid moves.
"""
from __future__ import annotations

import colorsys

PLATFORM = "/World/Platform"

# name: (rgb, roughness, metallic, opacity, ior)
BASELINE = {
    "bench": ((0.16, 0.17, 0.18), 0.55, 0.0, 1.0, 1.5),       # dark grey epoxy resin top
    "stand": ((0.62, 0.63, 0.64), 0.45, 0.25, 1.0, 1.5),      # powder-coated steel cabinet
    "floor": ((0.42, 0.43, 0.44), 0.8, 0.0, 1.0, 1.5),        # grey vinyl
    "walls": ((0.86, 0.86, 0.83), 0.9, 0.0, 1.0, 1.5),        # off-white paint
    "pedestal": ((0.25, 0.26, 0.28), 0.5, 0.2, 1.0, 1.5),
    "holder": ((0.92, 0.92, 0.9), 0.5, 0.0, 1.0, 1.5),        # white polypropylene rack
    "tube": ((0.93, 0.97, 0.98), 0.04, 0.0, 0.18, 1.47),      # borosilicate glass
    "pipette": ((0.82, 0.84, 0.86), 0.35, 0.0, 1.0, 1.5),     # light grey ABS
    "tray": ((0.9, 0.92, 0.95), 0.15, 0.0, 0.55, 1.59),       # clear polystyrene plate
    "tip_rack": ((0.2, 0.42, 0.78), 0.4, 0.0, 1.0, 1.5),      # blue tip box
    "tip": ((0.95, 0.95, 0.92), 0.3, 0.0, 0.6, 1.49),         # natural polypropylene
    "waste_bin": ((0.45, 0.47, 0.5), 0.6, 0.0, 1.0, 1.5),
    "centrifuge": ((0.88, 0.88, 0.86), 0.4, 0.05, 1.0, 1.5),  # light grey housing
}
TARGETS = {
    "bench": [f"{PLATFORM}/Bench"], "stand": [f"{PLATFORM}/Stand"], "floor": ["/World/Floor"],
    "walls": ["/World/WallN", "/World/WallS", "/World/WallE", "/World/WallW"], "pedestal": ["/World/Pedestal"],
    "holder": [f"{PLATFORM}/TubeHolder"], "pipette": [f"{PLATFORM}/Pipette0"], "tray": [f"{PLATFORM}/SampleTray"],
    "tip_rack": [f"{PLATFORM}/TipRack"], "waste_bin": [f"{PLATFORM}/WasteBin"],
    "centrifuge": [f"{PLATFORM}/CentrifugeBody"],
}
# Randomization choices: what real lab equipment comes in
BENCH_TOPS = [((0.16, 0.17, 0.18), 0.55, 0.0), ((0.07, 0.07, 0.08), 0.5, 0.0), ((0.55, 0.56, 0.57), 0.6, 0.0),
              ((0.84, 0.84, 0.82), 0.5, 0.0), ((0.74, 0.69, 0.6), 0.6, 0.0), ((0.62, 0.63, 0.64), 0.35, 0.85)]
WALLS = [(0.86, 0.86, 0.83), (0.78, 0.8, 0.8), (0.82, 0.86, 0.9), (0.9, 0.88, 0.8), (0.7, 0.72, 0.74)]
FLOORS = [(0.42, 0.43, 0.44), (0.3, 0.32, 0.35), (0.55, 0.55, 0.52), (0.38, 0.42, 0.45)]
HOLDERS = [(0.92, 0.92, 0.9), (0.25, 0.45, 0.85), (0.3, 0.7, 0.4), (0.95, 0.8, 0.2), (0.85, 0.3, 0.3)]
TIP_RACKS = [((0.2, 0.42, 0.78), 1.0), ((0.12, 0.12, 0.13), 1.0), ((0.85, 0.88, 0.9), 0.5), ((0.3, 0.65, 0.4), 1.0)]
TRAYS = [((0.9, 0.92, 0.95), 0.55), ((0.95, 0.95, 0.95), 1.0), ((0.12, 0.12, 0.13), 1.0)]
BINS = [(0.45, 0.47, 0.5), (0.9, 0.45, 0.1), (0.85, 0.85, 0.85), (0.2, 0.35, 0.7)]
PIPETTES = [(0.82, 0.84, 0.86), (0.94, 0.94, 0.94), (0.35, 0.38, 0.42)]
LIQUID_HUES = [(0.15, 0.35, 0.9), (0.15, 0.75, 0.25), (0.85, 0.15, 0.15), (0.9, 0.75, 0.1),
               (0.6, 0.2, 0.75), (0.95, 0.5, 0.1), (0.1, 0.7, 0.75)]

PANELS = ((-0.35, 0.0), (0.35, 0.0))           # two ceiling panels over the bench (x offsets, y offset)
PANEL_HEIGHT = 1.35                            # m above the bench top
PANEL_SIZE = (1.2, 0.35)
PANEL_INTENSITY = 1800.0
PANEL_KELVIN = 5000.0
DOME_INTENSITY = 450.0


def _material(stage, name, rgb, rough, metal, opacity, ior):
    from pxr import Gf, Sdf, UsdShade

    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    for k, t, v in (("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, rgb))),
                    ("roughness", Sdf.ValueTypeNames.Float, float(rough)),
                    ("metallic", Sdf.ValueTypeNames.Float, float(metal)),
                    ("opacity", Sdf.ValueTypeNames.Float, float(opacity)),
                    ("ior", Sdf.ValueTypeNames.Float, float(ior))):
        sh.CreateInput(k, t).Set(v)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(stage, prim_path, mat):
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def _jitter(rng, rgb, amount=0.04):
    """Small brightness / saturation variation around a colour."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = min(1.0, max(0.0, l * (1 + rng.uniform(-amount, amount) * 2)))
    s = min(1.0, max(0.0, s * (1 + rng.uniform(-amount, amount) * 3)))
    return colorsys.hls_to_rgb(h, l, s)


def _lights(stage, rng):
    from pxr import Gf, UsdGeom, UsdLux

    import lab_scene

    old = stage.GetPrimAtPath(f"{PLATFORM}/WorkLight")
    if old.IsValid():
        UsdLux.RectLight(old).GetIntensityAttr().Set(0.0)      # superseded by the panels below
    out = []
    for i, (dx, dy) in enumerate(PANELS):
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/CeilingPanel{i}")
        k = PANEL_KELVIN if rng is None else rng.uniform(3500.0, 6500.0)
        inten = PANEL_INTENSITY if rng is None else PANEL_INTENSITY * rng.uniform(0.6, 1.4)
        jx, jy = (0.0, 0.0) if rng is None else (rng.uniform(-0.2, 0.2), rng.uniform(-0.25, 0.25))
        light.CreateIntensityAttr(float(inten))
        light.CreateWidthAttr(PANEL_SIZE[0])
        light.CreateHeightAttr(PANEL_SIZE[1])
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(float(k))
        xf = UsdGeom.Xformable(light)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(lab_scene.PLATFORM_CX + dx + jx, lab_scene.PLATFORM_CY + dy + jy,
                                         lab_scene.BENCH_Z + 0.04 + PANEL_HEIGHT))
        out.append(dict(kelvin=k, intensity=inten, offset=(jx, jy)))
    dome = stage.GetPrimAtPath("/World/DomeLight")
    dome_i = DOME_INTENSITY if rng is None else DOME_INTENSITY * rng.uniform(0.5, 1.5)
    if dome.IsValid():
        UsdLux.DomeLight(dome).GetIntensityAttr().Set(float(dome_i))
    return dict(panels=out, dome=dome_i)


def apply(stage, rng=None):
    """Bind materials and set the lights. rng=None: the baseline look;
    a random.Random: a randomized look. Returns what was chosen (for the
    episode's metadata)."""
    import lab_scene

    looks = {k: list(v) for k, v in BASELINE.items()}
    chosen = {}
    if rng is not None:
        rgb, rough, metal = rng.choice(BENCH_TOPS)
        looks["bench"][:3] = [_jitter(rng, rgb), rough, metal]
        looks["walls"][0] = _jitter(rng, rng.choice(WALLS))
        looks["floor"][0] = _jitter(rng, rng.choice(FLOORS))
        looks["holder"][0] = _jitter(rng, rng.choice(HOLDERS))
        rgb, op = rng.choice(TIP_RACKS)
        looks["tip_rack"][0], looks["tip_rack"][3] = _jitter(rng, rgb), op
        rgb, op = rng.choice(TRAYS)
        looks["tray"][0], looks["tray"][3] = _jitter(rng, rgb), op
        looks["waste_bin"][0] = _jitter(rng, rng.choice(BINS))
        looks["pipette"][0] = _jitter(rng, rng.choice(PIPETTES))
        looks["centrifuge"][0] = _jitter(rng, BASELINE["centrifuge"][0])
    for name, (rgb, rough, metal, opacity, ior) in looks.items():
        mat = _material(stage, name, rgb, rough, metal, opacity, ior)
        for p in TARGETS.get(name, []):
            _bind(stage, p, mat)
        chosen[name] = dict(rgb=[round(float(c), 3) for c in rgb], roughness=rough, metallic=metal, opacity=opacity)
    tube_mat = _material(stage, "tube", *looks["tube"])
    tip_mat = _material(stage, "tip", *looks["tip"])
    for prim in stage.GetPrimAtPath(PLATFORM).GetChildren():
        n = prim.GetName()
        if n.startswith("Tube_"):
            _bind(stage, str(prim.GetPath()), tube_mat)
        elif n == "Tips":
            for tip in prim.GetChildren():
                _bind(stage, str(tip.GetPath()), tip_mat)
    if rng is not None:
        hues = rng.sample(LIQUID_HUES, k=min(len(LIQUID_HUES), len(lab_scene.LIQUIDS)))
        for (name, L), rgb in zip(sorted(lab_scene.LIQUIDS.items()), hues):
            if not name.startswith("tray/"):
                L.set_color(tuple(_jitter(rng, rgb)))
        chosen["liquids"] = {n: list(L.color) for n, L in lab_scene.LIQUIDS.items() if L.volume_ul > 0}
    chosen["lights"] = _lights(stage, rng)
    return chosen
