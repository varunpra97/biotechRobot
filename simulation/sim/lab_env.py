"""15 Hz physics-driven control of the lab scene -- the interface a π0.5
policy (or our scripted expert) drives, and the recorder for training data.

Action (10 values per step, DROID conventions where DROID has them):
  [0:7]  joint velocity, normalized to [-1, 1]. Each step moves joint i by
         a_i * 0.2 rad (DROID's max_joint_delta); if any |a_i| > 1 the whole
         vector is scaled down together (DROID's joint_velocity_to_delta).
  [7]    gripper position: 0 = fully open (80 mm), 1 = fully closed.
         Closing is force-limited (lab_scene.FINGER_MAX_FORCE), so a closed
         command squeezes until the pads stop on the object.
  [8]    pipette plunger: fraction of the pipette's capacity drawn in
         (0 = empty). Raising it aspirates, lowering it dispenses.
  [9]    pipette tip eject: 1 = press the eject button.

The arm is never teleported here: each step integrates the velocity command
into a commanded joint position (as DROID's controller does) and feeds
drive targets interpolated across the step's physics sub-steps, so the
recorded motion is what the motors and contacts actually produced.

Observation per step: measured joint positions/velocities, gripper
position (0 open .. 1 closed), commanded positions, TCP pose, object poses,
pipette state, and camera images.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import lab_scene

CONTROL_HZ = 15
MAX_JOINT_DELTA = 0.2          # rad per step at |action| = 1 (DROID)
APP_HZ = 60                    # one SimulationApp.update() = 1/60 s (measured)
UPDATES_PER_STEP = APP_HZ // CONTROL_HZ
ACTION_DIM = 10
STATE_DIM = 10                 # 7 joints, gripper, plunger, tip attached
EXTERIOR_CAMERA = "platform_cam1"   # fixed, whole platform: pi0.5's base image
# The wrist image pi0.5 trains on, per primitive: the one of the three wrist
# cameras (lab_scene.WRIST_CAMERAS) that shows that primitive best. All
# three are recorded in every episode; the exporter sends this one. On the
# real robot the controller must feed the same camera for the primitive it
# is running.
PRIMITIVE_WRIST_CAMERA = {
    "pick_tube_from_holder": "wrist_top", "insert_tube_into_bore": "wrist_top",
    "remove_tube_from_bore": "wrist_top", "place_tube_into_holder": "wrist_top",
    "pick_up_pipette": "wrist_top", "attach_tip": "wrist_side", "aspirate_from_tube": "wrist_side",
    "dispense_into_well": "wrist_side", "dispense_into_tube": "wrist_side", "eject_tip": "wrist_side",
    "return_pipette": "wrist_side", "pick_up_tray": "wrist_top", "place_tray": "wrist_side",
}
# (wrist_high was the best view for none of them, compared side by side on
# a full recorded chain; it is still recorded as extra context.)


def droid_velocity_to_delta(a):
    a = np.asarray(a, float)
    n = np.abs(a).max()
    if n > 1:
        a = a / n
    return a * MAX_JOINT_DELTA


def delta_to_droid_velocity(dq):
    return np.asarray(dq, float) / MAX_JOINT_DELTA


def gripper_cmd_to_finger(g):
    return float(np.clip(1.0 - g, 0.0, 1.0)) * lab_scene.GRIPPER_STROKE


def finger_to_gripper_cmd(f):
    return float(np.clip(1.0 - f / lab_scene.GRIPPER_STROKE, 0.0, 1.0))


class LabEnv:
    def __init__(self, simulation_app, rigid_objects=(), record_cameras=None, image_size=(640, 480)):
        self.app = simulation_app
        self.arm, self.cams, self.objects = lab_scene.build_scene(simulation_app, rigid_objects)
        self.record_cameras = sorted(self.cams) if record_cameras is None else list(record_cameras)
        self.q_cmd = np.asarray(self.arm.get_joint_positions())[0].copy()
        self.gripper_cmd = finger_to_gripper_cmd(self.q_cmd[7])
        self.pipette = None          # set by the pipette module when present
        self.recorder = None
        # Task objects are handled physically (friction grasps, released
        # objects settling), so they're dynamic from the start.
        for name in rigid_objects:
            lab_scene.set_kinematic(self.objects[name]["path"], False)
        for _ in range(30):
            self.app.update()

    # ------------------------------------------------------------ state
    def joint_state(self):
        q = np.asarray(self.arm.get_joint_positions())[0]
        qd = np.asarray(self.arm.get_joint_velocities())[0]
        return q, qd

    def gripper_position(self, q=None):
        q = self.joint_state()[0] if q is None else q
        return finger_to_gripper_cmd((q[7] + q[8]) / 2.0)

    def observe(self, images=True):
        q, qd = self.joint_state()
        tcp, R = lab_scene.get_tcp_pose()
        pip = self.pipette.state() if self.pipette else {"plunger": 0.0, "tip_attached": 0.0}
        obs = {
            "joint_position": q[:7].copy(),
            "joint_velocity": qd[:7].copy(),
            "gripper_position": np.array([self.gripper_position(q)]),
            "finger_position_m": q[7:9].copy(),
            "commanded_joint_position": self.q_cmd[:7].copy(),
            "tcp_pose": np.concatenate([tcp, lab_scene.matrix_to_quat(R)]),
            "hand_pose": lab_scene.get_prim_world_pose(lab_scene.HAND_PATH),
            "pipette": pip,
            "object_poses": {n: lab_scene.get_prim_world_pose(o["path"]) for n, o in self.objects.items()},
        }
        obs["state"] = np.concatenate([obs["joint_position"], obs["gripper_position"],
                                       [pip["plunger"], pip["tip_attached"]]])
        if images:
            obs["images"] = {n: self.cams[n][1].get_current_frame()["rgb"][..., :3]
                             for n in self.record_cameras}
        return obs

    # ------------------------------------------------------------- step
    def step(self, action):
        """Apply one 15 Hz action (see module docstring), advance physics by
        1/15 s, record the (observation, action) pair if recording, and
        return the next observation's lightweight state (no images)."""
        action = np.asarray(action, float)
        if self.recorder is not None:
            self.recorder.add(self.observe(images=True), action)
        q0 = self.q_cmd.copy()
        q1 = q0.copy()
        q1[:7] = q0[:7] + droid_velocity_to_delta(action[:7])
        q1[:7] = np.clip(q1[:7], self.limits[:7, 0], self.limits[:7, 1])
        self.gripper_cmd = float(np.clip(action[7], 0.0, 1.0))
        q1[7] = q1[8] = gripper_cmd_to_finger(self.gripper_cmd)
        if self.pipette is not None:
            self.pipette.command(plunger=float(action[8]), eject=float(action[9]))
        for u in range(1, UPDATES_PER_STEP + 1):
            t = u / UPDATES_PER_STEP
            tgt = q0.copy()
            tgt[:7] = q0[:7] + t * (q1[:7] - q0[:7])
            tgt[7:] = q1[7:]
            self.arm.set_joint_position_targets(tgt)
            self.app.update()
            if self.pipette is not None:
                self.pipette.update()
        self.q_cmd = q1
        return self.observe(images=False)

    @property
    def limits(self):
        return np.asarray(self.arm.get_dof_limits())[0]

    def hold(self, steps, pipette=None):
        """Zero-velocity steps (gripper/pipette keep their current command)."""
        for _ in range(steps):
            self.step(self.action(np.zeros(7), pipette=pipette))

    def action(self, velocity, gripper=None, pipette=None):
        pip = pipette if pipette is not None else (
            (self.pipette.plunger_cmd, 0.0) if self.pipette else (0.0, 0.0))
        g = self.gripper_cmd if gripper is None else gripper
        return np.concatenate([velocity, [g], pip])


class EpisodeRecorder:
    """Raw episode on disk: every recorded camera as JPEG frames at 15 Hz,
    all states/actions in steps.npz, a 2-camera (exterior | policy wrist
    camera) MP4 for viewing, and meta.json. export_lerobot.py turns these
    into the LeRobot dataset π0.5 fine-tunes on: the exterior camera plus
    `wrist_camera`, the wrist camera chosen for this primitive."""

    def __init__(self, out_dir, prompt, meta=None, jpeg_quality=95, wrist_camera="wrist_high"):
        self.dir = Path(out_dir)
        shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "images").mkdir(parents=True)
        self.prompt = prompt
        self.meta = dict(meta or {})
        self.q = jpeg_quality
        self.rows = []
        self.video_frames = []
        self.wrist_camera = wrist_camera
        self.policy_cameras = (EXTERIOR_CAMERA, wrist_camera)

    def add(self, obs, action):
        from PIL import Image

        t = len(self.rows)
        for cam, img in obs["images"].items():
            Image.fromarray(img).save(self.dir / "images" / f"{cam}_{t:05d}.jpg", quality=self.q)
        if all(c in obs["images"] for c in self.policy_cameras):
            self.video_frames.append(np.concatenate([obs["images"][c] for c in self.policy_cameras], axis=1))
        row = {k: obs[k] for k in ("joint_position", "joint_velocity", "gripper_position",
                                   "finger_position_m", "commanded_joint_position",
                                   "tcp_pose", "hand_pose", "state")}
        row["plunger"] = obs["pipette"]["plunger"]
        row["tip_attached"] = obs["pipette"]["tip_attached"]
        row["liquid_volumes_ul"] = obs["pipette"].get("volumes", {})
        row["object_poses"] = obs["object_poses"]
        row["action"] = np.asarray(action, float)
        self.rows.append(row)

    def finish(self, success, info=None):
        import cv2

        n = len(self.rows)
        names = sorted(self.rows[0]["object_poses"]) if n else []
        arr = lambda k: np.array([r[k] for r in self.rows])
        np.savez_compressed(
            self.dir / "steps.npz",
            joint_position=arr("joint_position"), joint_velocity=arr("joint_velocity"),
            gripper_position=arr("gripper_position"), finger_position_m=arr("finger_position_m"),
            commanded_joint_position=arr("commanded_joint_position"),
            tcp_pose=arr("tcp_pose"), hand_pose=arr("hand_pose"), state=arr("state"),
            plunger=arr("plunger"), tip_attached=arr("tip_attached"),
            action=arr("action"),
            object_poses=np.array([[r["object_poses"][k] for k in names] for r in self.rows]),
        )
        vols = [r["liquid_volumes_ul"] for r in self.rows]
        meta = {
            "prompt": self.prompt, "n_steps": n, "control_hz": CONTROL_HZ,
            "success": bool(success), "info": info or {},
            "object_names": names, "training_cameras": list(self.policy_cameras),
            "policy_exterior_camera": EXTERIOR_CAMERA, "policy_wrist_camera": self.wrist_camera,
            "action_spec": {
                "joint_velocity": "[0:7] DROID-normalized: joint delta per step = a * 0.2 rad",
                "gripper": "[7] 0 = open (80 mm) .. 1 = closed, force-limited close",
                "pipette_plunger": "[8] fraction of capacity drawn in (0 = empty)",
                "pipette_eject": "[9] 1 = eject tip",
            },
            "state_spec": "[0:7] joint position (rad), [7] gripper 0 open..1 closed, "
                          "[8] plunger fraction, [9] tip attached",
            "liquid_volumes_ul_first_last": [vols[0], vols[-1]] if vols else [],
            **self.meta,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
        if self.video_frames:
            h, w = self.video_frames[0].shape[:2]
            vw = cv2.VideoWriter(str(self.dir / "exterior_wrist.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_HZ, (w, h))
            for f in self.video_frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
        return self.dir
