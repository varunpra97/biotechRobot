"""Phase 1 data collector: full pick-and-place task -- the arm grasps a
test tube from the holder, transports it, and places it into a real
centrifuge rotor bucket. Builds on collect_data.py's constraint-based IK
approach (sim/ik_solver.py), extended with an actual grasp (gripper
fingers close to the tube's real radius) and a kinematic rigid-attach so
the tube rides with the gripper during transport.

Physical grounding, not guessed:
  - Grasp width is computed from the tube's real 11mm diameter, not a
    fixed/arbitrary closure value.
  - The centrifuge target is one of 6 real rotor bucket cavities --
    confirmed via volume-vs-convex-hull-volume ratio (~27%, i.e. a real
    hollow cup, not a solid decoration) and the fact all 6 share an
    identical world-Z range (so their opening genuinely faces up) --
    not assumed from the mesh's appearance. See cad_source/
    find_bucket_target.py.
  - The test tube itself was resized from an earlier arbitrary 25mm/44mm
    guess down to 11mm specifically so it fits the centrifuge's real
    cavity (~11.6mm diameter) -- a real 1.5mL microcentrifuge tube size,
    which is also a more realistic choice for this project generally.

Kinematic attach is a deliberate simplification, not full contact-friction
grasping dynamics: everything in this pipeline (the arm itself) is already
kinematically teleported via set_joint_positions rather than PD-driven
dynamics, so a tube that's friction-held via real contact physics while
the arm teleports between IK solutions isn't really meaningful here
either -- once grasped, the tube's pose is computed from the gripper's
current pose each frame, rigidly, which is the consistent choice given the
rest of the architecture.

Usage (Isaac venv, repo root on Windows box):
    python sim\\collect_data_pickplace.py --out ..\\data\\episodes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ik_solver
import lab_scene

SOURCE_TUBE = "tube_r0_c2"       # same reachable slot used by collect_data.py
TARGET_BUCKET_PART = 5           # cad_source/find_bucket_target.py part index
BUCKET_CENTER_XY = (-0.4374, 1.6491)   # measured world XY, see module docstring
BUCKET_BOTTOM_Z = 1.0198               # measured world Z, bucket cavity floor
BUCKET_TOP_Z = 1.0486                  # measured world Z, bucket cavity rim

HOVER_HEIGHT = 0.12        # m above a tube's top -- approach/retraction height
GRASP_DEPTH = 0.012        # m below the tube's top rim -- where fingers close
LIFT_HEIGHT = 0.15         # m above the bench -- transit height, clears everything
PLACE_INSERT = 0.02        # m of clearance above the bucket floor when placed --
                           # also sets how far the fingertip clears the rotor
                           # rim above (tube is taller than the cavity is deep,
                           # so it doesn't need to touch bottom); 5mm left only
                           # ~4mm of fingertip-to-rim clearance, which still hit
                           # real contact -- 20mm gives a comfortable margin
GRIPPER_OPEN = 0.04        # m per finger (matches Q_HOME / joint limit max)
GRIPPER_RADIUS_CLEARANCE = 0.0005  # m: finger closes to tube radius + this
BENCH_CLEARANCE = 0.01
GRIPPER_RADIUS = 0.045     # for the collision-check horizontal margin


def quat_to_matrix(q):
    """q = [qw, qx, qy, qz] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def check_collision(fingertip_pos, objects, exclude, radius=GRIPPER_RADIUS):
    hits = []
    for name, info in objects.items():
        if name in exclude:
            continue
        pos = lab_scene.get_prim_world_pose(info["path"])[:3]
        w, d, h = info["size"]
        lo = pos - np.array([w / 2 + radius, d / 2 + radius, h / 2])
        hi = pos + np.array([w / 2 + radius, d / 2 + radius, h / 2])
        if np.all(fingertip_pos >= lo) and np.all(fingertip_pos <= hi):
            hits.append(name)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\data\episodes")
    ap.add_argument("--ep-index", type=int, default=0)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from PIL import Image

    arm, cams, objects = lab_scene.build_scene(simulation_app)
    cam_names = sorted(cams.keys())
    obj_names = sorted(objects.keys())

    for name in cam_names:
        _, cam = cams[name]
        cam.add_distance_to_camera_to_frame()
        cam.add_instance_segmentation_to_frame()
    for _ in range(5):
        simulation_app.update()

    hand_idx = arm.get_body_index(lab_scene.HAND_PATH.rsplit("/", 1)[-1])
    joint_limits = np.asarray(arm.get_dof_limits())[0]
    bench_top_z = lab_scene.BENCH_Z + 0.04

    def get_fingertip_pos():
        return lab_scene.get_fingertip_pos()

    def step():
        simulation_app.update()

    tube_path = objects[SOURCE_TUBE]["path"]
    tube_size = objects[SOURCE_TUBE]["size"]
    tube_radius = tube_size[0] / 2.0
    grip_finger_val = max(0.0, tube_radius + GRIPPER_RADIUS_CLEARANCE)

    q = lab_scene.Q_HOME.copy()
    arm.set_joint_positions(q)
    arm.set_joint_position_targets(q)
    for _ in range(10):
        simulation_app.update()

    tube_pos0 = lab_scene.get_prim_world_pose(tube_path)[:3]
    # tube_pos0 is the tube's BASE, not its center -- load_mesh_prim places
    # the mesh's local origin (z=0 at the tube's bottom, per
    # cad_source/make_test_tube.py) directly at the given world position,
    # so the top is a full tube_size[2] above it, not half. Getting this
    # wrong drove the grasp target below the holder's top surface (into
    # solid material) once the tube was resized down to a real 40mm
    # height, causing a physics blow-up identical in kind to the
    # bucket-rim collision fixed above.
    tube_top = tube_pos0[2] + tube_size[2]
    grasp_z = tube_top - GRASP_DEPTH

    bucket_xy = BUCKET_CENTER_XY
    # The fingertip is a fixed height above the tube's BOTTOM once grasped
    # (tube_size[2] - GRASP_DEPTH, since the grasp point is GRASP_DEPTH
    # below the tube's top and rigidly attached from then on). Deriving the
    # fingertip target from that -- rather than adding GRASP_DEPTH directly
    # -- is what keeps the fingers above the rotor bucket's rim (they
    # physically can't fit through its ~11.6mm opening) while only the
    # narrow tube itself descends into the cavity. Getting this wrong
    # previously drove the fingers through solid housing geometry, which
    # blew up the physics (confirmed via a standalone IK trace showing the
    # simulated hand position diverging to NaN from real contact forces,
    # not an IK/math bug).
    fingertip_above_tube_bottom = tube_size[2] - GRASP_DEPTH
    place_z_fingertip = BUCKET_BOTTOM_Z + PLACE_INSERT + fingertip_above_tube_bottom

    # (phase, xyz_target, gripper_finger_value)
    waypoints = [
        ("approach_source", np.array([tube_pos0[0], tube_pos0[1], tube_top + HOVER_HEIGHT]), GRIPPER_OPEN),
        ("descend_to_grasp", np.array([tube_pos0[0], tube_pos0[1], grasp_z]), GRIPPER_OPEN),
        ("close_gripper", np.array([tube_pos0[0], tube_pos0[1], grasp_z]), grip_finger_val),
        ("lift", np.array([tube_pos0[0], tube_pos0[1], bench_top_z + LIFT_HEIGHT]), grip_finger_val),
        ("transit", np.array([bucket_xy[0], bucket_xy[1], bench_top_z + LIFT_HEIGHT]), grip_finger_val),
        ("approach_centrifuge", np.array([bucket_xy[0], bucket_xy[1], BUCKET_TOP_Z + HOVER_HEIGHT]), grip_finger_val),
        ("descend_into_bucket", np.array([bucket_xy[0], bucket_xy[1], place_z_fingertip]), grip_finger_val),
        ("open_gripper", np.array([bucket_xy[0], bucket_xy[1], place_z_fingertip]), GRIPPER_OPEN),
        ("retract_up", np.array([bucket_xy[0], bucket_xy[1], BUCKET_TOP_Z + HOVER_HEIGHT]), GRIPPER_OPEN),
        ("retract_home", np.array([tube_pos0[0], tube_pos0[1], tube_top + HOVER_HEIGHT]), GRIPPER_OPEN),
    ]

    # Two different windows, deliberately not the same: collision checking
    # must stop treating the tube as an obstacle as soon as the gripper
    # starts reaching for it (descend_to_grasp), but the kinematic attach
    # (tube rigidly follows the hand) should only start once the gripper
    # has actually closed (close_gripper) -- attaching one phase earlier
    # would snap the tube to the hand before the fingers ever reach it.
    APPROACH_FROM = "descend_to_grasp"
    GRASPED_FROM = "close_gripper"
    RELEASED_FROM = "open_gripper"

    def _phase_window(start, end_inclusive):
        s = set()
        active = False
        for name, *_ in waypoints:
            if name == start:
                active = True
            if active:
                s.add(name)
            if name == end_inclusive:
                active = False
        return s

    near_tube_phases = _phase_window(APPROACH_FROM, RELEASED_FROM)
    grasped_phases = _phase_window(GRASPED_FROM, RELEASED_FROM)

    frames_q, frames_grip, wp_log = [], [], []
    tube_grasp_offset_local = None  # tube pos relative to hand, in hand's local frame
    ok = True

    for phase_name, target, grip_val in waypoints:
        q, converged, iters, err = ik_solver.solve_ik(
            arm, hand_idx, target, q, joint_limits, get_fingertip_pos, step)
        q[7] = q[8] = grip_val
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)
        for _ in range(4):
            simulation_app.update()

        fingertip_pos = get_fingertip_pos()
        above_bench = bool(fingertip_pos[2] >= (bench_top_z - BENCH_CLEARANCE))
        within_limits = bool(
            np.all(q[:7] >= joint_limits[:7, 0] - 1e-6)
            and np.all(q[:7] <= joint_limits[:7, 1] + 1e-6))
        finite = bool(np.all(np.isfinite(q)))

        exclude = {SOURCE_TUBE} if phase_name in near_tube_phases else set()
        collisions = check_collision(fingertip_pos, objects, exclude)
        no_collision = len(collisions) == 0
        wp_ok = converged and above_bench and within_limits and finite and no_collision
        ok = ok and wp_ok

        # Kinematic attach: capture the tube's offset relative to the hand
        # at the moment of grasp, then carry the tube rigidly with the
        # hand for every subsequent grasped frame.
        hand_pose = lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)
        if phase_name == GRASPED_FROM:
            tube_pose_now = lab_scene.get_prim_world_pose(tube_path)
            R_hand = quat_to_matrix(hand_pose[3:7])
            tube_grasp_offset_local = R_hand.T @ (tube_pose_now[:3] - hand_pose[:3])
        if phase_name in grasped_phases and tube_grasp_offset_local is not None:
            R_hand = quat_to_matrix(hand_pose[3:7])
            new_tube_pos = hand_pose[:3] + R_hand @ tube_grasp_offset_local
            lab_scene.set_prim_world_position(tube_path, new_tube_pos)

        wp_log.append({
            "phase": phase_name, "target": target.tolist(), "gripper": float(grip_val),
            "ik_converged": converged, "ik_iters": iters, "ik_final_error_m": err,
            "above_bench": above_bench, "within_joint_limits": within_limits,
            "no_collision": no_collision, "collisions": collisions, "ok": wp_ok,
        })
        frames_q.append(q.copy())
        frames_grip.append(grip_val)
        if not wp_ok:
            print(f"  waypoint '{phase_name}' failed (converged={converged} err={err:.4f} "
                  f"above_bench={above_bench} within_limits={within_limits} collisions={collisions})")
            break

    if not ok:
        print("episode REJECTED (constraint violation) -- not saved")
        simulation_app.close()
        return

    ep = args.ep_index
    out_root = Path(args.out)
    ep_dir = out_root / f"ep_{ep:04d}"
    (ep_dir / "images").mkdir(parents=True, exist_ok=True)
    (ep_dir / "depth").mkdir(parents=True, exist_ok=True)
    (ep_dir / "segmentation").mkdir(parents=True, exist_ok=True)

    joint_pos, joint_vel, gripper = [], [], []
    ee_pose, fingertip_pose, obj_poses, action = [], [], [], []
    seg_info = None

    # The validate pass above already moved the real tube prim (kinematic
    # attach calls set_prim_world_position on it, not a dry-run copy), so
    # by now it's sitting wherever that pass left it -- in the bucket, not
    # the holder. Reset it to its original pre-grasp position before
    # replaying, otherwise the replay's early (pre-grasp) frames would
    # record/render the tube already at the bucket while the arm is still
    # over the holder.
    lab_scene.set_prim_world_position(tube_path, tube_pos0)
    for _ in range(4):
        simulation_app.update()

    # Replay the solved frames_q to actually capture images, re-applying
    # the same kinematic tube-attach so captured frames show the tube
    # correctly positioned, not snapped back to its original holder slot.
    tube_grasp_offset_local = None
    for f, (q_f, phase) in enumerate(zip(frames_q, [w[0] for w in waypoints])):
        arm.set_joint_positions(q_f)
        arm.set_joint_position_targets(q_f)
        for _ in range(8):
            simulation_app.update()

        hand_pose = lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)
        if phase == GRASPED_FROM:
            tube_pose_now = lab_scene.get_prim_world_pose(tube_path)
            R_hand = quat_to_matrix(hand_pose[3:7])
            tube_grasp_offset_local = R_hand.T @ (tube_pose_now[:3] - hand_pose[:3])
        if phase in grasped_phases and tube_grasp_offset_local is not None:
            R_hand = quat_to_matrix(hand_pose[3:7])
            new_tube_pos = hand_pose[:3] + R_hand @ tube_grasp_offset_local
            lab_scene.set_prim_world_position(tube_path, new_tube_pos)
        for _ in range(2):
            simulation_app.update()

        for name in cam_names:
            _, cam = cams[name]
            frame = cam.get_current_frame()
            Image.fromarray(frame["rgb"]).save(str(ep_dir / "images" / f"{name}_f{f:04d}.png"))
            depth = frame.get("distance_to_camera")
            if depth is not None:
                np.save(str(ep_dir / "depth" / f"{name}_f{f:04d}.npy"), np.asarray(depth))
            seg = frame.get("instance_segmentation")
            if seg is not None:
                np.save(str(ep_dir / "segmentation" / f"{name}_f{f:04d}.npy"), np.asarray(seg["data"]))
                if seg_info is None:
                    seg_info = seg["info"]

        qp = np.asarray(arm.get_joint_positions())[0]
        qv = np.asarray(arm.get_joint_velocities())[0]
        joint_pos.append(qp[:7])
        joint_vel.append(qv[:7])
        gripper.append(qp[7:9])
        ee_pose.append(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))
        fingertip_pose.append(lab_scene.get_fingertip_pos())
        obj_poses.append([lab_scene.get_prim_world_pose(objects[n]["path"]) for n in obj_names])
        action.append(q_f[:7])

    np.savez(
        str(ep_dir / "trajectory.npz"),
        joint_pos=np.array(joint_pos), joint_vel=np.array(joint_vel),
        gripper=np.array(gripper), ee_pose=np.array(ee_pose),
        fingertip_pose=np.array(fingertip_pose),
        object_poses=np.array(obj_poses), action=np.array(action),
    )

    meta = {
        "task": f"pick_place[{SOURCE_TUBE}]->centrifuge_bucket_{TARGET_BUCKET_PART}",
        "n_frames": len(frames_q),
        "cameras": cam_names,
        "object_names": obj_names,
        "source_object": SOURCE_TUBE,
        "target_bucket_part_index": TARGET_BUCKET_PART,
        "target_bucket_world_xy": list(bucket_xy),
        "grasped_phases": sorted(grasped_phases),
        "gripper_grasp_value_m": grip_finger_val,
        "waypoints": wp_log,
        "sequence_valid": ok,
        "robot": {
            "joint_names": list(arm.dof_names),
            "link_names": list(arm.body_names),
            "joint_limits": joint_limits.tolist(),
        },
        "object_bboxes": {n: objects[n]["size"] for n in obj_names},
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if seg_info is not None:
        (ep_dir / "segmentation_info.json").write_text(json.dumps(seg_info, indent=2, default=str))

    print(f"episode {ep}: ACCEPTED, {len(frames_q)} frames -> {ep_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
