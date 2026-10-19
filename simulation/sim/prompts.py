"""Language prompts for each primitive -- several phrasings, one picked per
episode, naming the source/target so a policy can be told which one.

Naming: holder slots are rows A-D (front to back, toward the arm first) and
columns 1-5 (left to right as seen from the arm); centrifuge bores are
numbered 1-6 in the rotor's sleeve order; tip-rack and plate wells use the
usual lettered rows and numbered columns.
"""
from __future__ import annotations

import random

ROWS = "ABCDEFGHIJ"

TEMPLATES = {
    "pick_tube_from_holder": [
        "pick up the test tube in holder slot {slot}",
        "take the tube out of slot {slot} of the rack",
        "grab the test tube from position {slot}",
        "lift the tube from rack slot {slot}",
    ],
    "insert_tube_into_bore": [
        "put the test tube into centrifuge slot {bore}",
        "place the tube in rotor position {bore} of the centrifuge",
        "load the tube into centrifuge bore {bore}",
        "insert the tube into slot {bore} of the centrifuge rotor",
    ],
    "remove_tube_from_bore": [
        "take the test tube out of centrifuge slot {bore}",
        "remove the tube from rotor position {bore}",
        "pull the tube out of centrifuge bore {bore}",
        "unload the tube from slot {bore} of the centrifuge",
    ],
    "place_tube_into_holder": [
        "put the test tube back in holder slot {slot}",
        "place the tube into rack slot {slot}",
        "return the tube to position {slot} in the holder",
        "set the tube down in slot {slot} of the rack",
    ],
    "pick_up_pipette": [
        "pick up the pipette",
        "grab the electronic pipette",
        "take the pipette off the bench",
    ],
    "attach_tip": [
        "attach a pipette tip from tip rack position {tip}",
        "put on the tip at {tip} of the tip box",
        "load a fresh tip from slot {tip}",
    ],
    "aspirate_from_tube": [
        "draw {vol} microliters from the tube in slot {slot}",
        "aspirate {vol} uL out of the test tube at {slot}",
        "pipette {vol} microliters from tube {slot}",
    ],
    "dispense_into_well": [
        "dispense the liquid into well {well} of the plate",
        "release the sample into plate well {well}",
        "empty the pipette into well {well}",
    ],
    "dispense_into_tube": [
        "dispense the liquid into the tube in slot {slot}",
        "transfer the sample into test tube {slot}",
        "empty the pipette into the tube at {slot}",
    ],
    "eject_tip": [
        "eject the pipette tip into the waste bin",
        "discard the used tip",
        "drop the tip into the waste container",
    ],
    "return_pipette": [
        "put the pipette back down",
        "return the pipette to the bench",
        "set the pipette down where it was",
    ],
    "pick_up_tray": [
        "pick up the sample tray",
        "lift the 10-well tray off the bench",
        "grab the sample plate by its sides",
    ],
    "place_tray": [
        "put the sample tray down {spot}",
        "place the tray {spot}",
        "set the sample plate down {spot}",
    ],
}

# where the tray can be put down (prompt phrase -> bench xy, see lab_scene)
TRAY_SPOTS = {
    "home": "back in its original spot",
    "right": "on the right side of the bench, next to the tip box",
}


def slot_name(r, c):
    return f"{ROWS[r]}{c + 1}"


def bore_name(sleeve_part, sleeve_parts):
    return str(list(sleeve_parts).index(sleeve_part) + 1)


def make(primitive, rng: random.Random, **kw):
    return rng.choice(TEMPLATES[primitive]).format(**kw)
