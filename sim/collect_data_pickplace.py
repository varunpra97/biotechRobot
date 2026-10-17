"""Phase 1 data collector: pick a test tube out of the holder and place it
into a bore of the centrifuge's fixed-angle rotor -- one validated episode
of 10 keyframes (images/depth/segmentation from every camera at each).

What makes this physically faithful (each point was verified against the
simulator, see the cad_source/probe_*.py scripts):
  - Real geometry everywhere. The rotor's bores are 8.0mm-ID, tilted ~45 deg
    from vertical with conical floors (measured from the mesh,
    sim/centrifuge_geometry.py); the tube (7mm OD x 58mm, conical tip) and
    the holder's 8mm blind bores were sized to them. Static fixtures collide
    as their exact triangle meshes, the tube as a <=64-vertex convex hull.
  - 6-DOF IK on the real fingertip (TCP): the gripper grasps the vertical
    tube with its fingers vertical, then tilts the wrist so the held tube is
    coaxial with the bore and inserts it along the bore axis. Position-only
    IK let the hand's orientation drift, which a 0.5mm-clearance bore can't
    tolerate.
  - Full-pose rigid attach while held: the tube follows the hand's position
    AND orientation, so it stays lined up between the pads when tilted.
  - Physical release: at the release pose the tube is handed to PhysX
    (kinematic -> dynamic, 240 Hz) and slides the last ~2mm down the bore
    under gravity until it seats. The episode is rejected unless it seats
    (tip at the measured seat depth, axis within a few deg of the bore's,
    no interpenetration, at rest).
  - Collision checking on the actual colliders: every densely interpolated
    configuration between keyframes (not just the keyframes) is checked
    with PhysX overlap queries -- every arm collider and the tube against
    everything else. The old fingertip-point-vs-bounding-box check missed
    the gripper going below the rotor rim entirely.
  - The release depth comes from a clearance scan with the real finger
    geometry (cad_source/gripper_rotor_clearance.py): the pads touch the
    rotor hub once the fingertip goes deeper than ~25mm along the bore
    axis, so the tube is released with the fingertip at 32mm.

Output (data/episodes/ep_XXXX/): per-keyframe images/depth/segmentation for
all cameras, trajectory.npz (keyframe states + the full validated dense
joint path, which view_motion.py replays), meta.json.

Usage (Isaac venv, repo root on Windows box):
    python sim\\collect_data_pickplace.py --out ..\\data\\episodes
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

import centrifuge_geometry as cg
import ik_solver
import lab_scene

SOURCE_TUBE = "tube_r0_c2"
# Rotor bore on the far side of the hub from the arm: its axis leans back
# toward the arm base, so the tilted gripper's wrist points back at the arm
# (a natural forward-and-down reach) instead of across the rotor.
TARGET_SLEEVE = 8

# Rigidly held = moves as one with these; pad contact is filtered out
# while attached (otherwise the kinematic tube shoves the fingers whenever
# the hand moves within an IK solve) and restored on release.
GRIPPER_BODIES = [lab_scene.HAND_PATH, lab_scene.LEFT_FINGER_PATH, lab_scene.RIGHT_FINGER_PATH]

GRASP_DEPTH = 0.012         # TCP below the tube's rim when grasped (pads cover the top 12mm)
GRIP_CLEARANCE = 0.0001     # m between each pad and the tube when closed
GRIPPER_OPEN = 0.012        # m per finger: 24mm opening around a 7mm tube
HOVER = 0.10                # m TCP above the tube's rim, approach/retreat at the holder
LIFT_CLEARANCE = 0.03       # m tube tip above the holder's top face after lifting
RELEASE_TCP_S = 0.032       # m along the bore axis: fingertip depth at release
INSERT_STROKE = 0.040       # m straight insertion along the bore axis
TRANSIT_RAISE = 0.08        # m above the pre-insert pose, where the wrist tilts
SETTLE_UPDATES = 60         # app updates (1 s; 240 physics steps) after release

# Seat acceptance, from the bore geometry and a released-tube probe
# (cad_source/probe_release_variants.py: tip settles at s = -15.75mm, axis
# within 2.8 deg -- the tube can lean within its 0.5mm radial clearance).
SEAT_TIP_S = (-0.0165, -0.0150)
SEAT_MAX_AXIS_DEG = 4.0
# A resting body always sits a hair into its support in a contact solver;
# more than this would mean it's actually lodged in the geometry.
MAX_RESTING_PENETRATION = 0.0002

MAX_STEP_POS = 0.005        # dense-path spacing between IK solves
MAX_STEP_ROT = math.radians(3.0)


def slerp_R(R0, R1, t):
    w = ik_solver.rotation_error(R1, R0)  # R1 = exp(w) R0
    th = np.linalg.norm(w)
    if th < 1e-12:
        return R0.copy()
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = th * t
    return (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R0


def frame_from_axis(z, y_hint):
    y = y_hint - (y_hint @ z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def pose_matrix(p):
    T = np.eye(4)
    T[:3, :3] = lab_scene.quat_to_matrix(p[3:7])
    T[:3, 3] = p[:3]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\data\episodes")
    ap.add_argument("--ep-index", type=int, default=0)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from PIL import Image
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf

    arm, cams, objects = lab_scene.build_scene(simulation_app, rigid_objects=(SOURCE_TUBE,))
    cam_names = sorted(cams.keys())
    obj_names = sorted(objects.keys())
    for name in cam_names:
        _, cam = cams[name]
        cam.add_distance_to_camera_to_frame()
        cam.add_instance_segmentation_to_frame()
    update = simulation_app.update
    for _ in range(5):
        update()

    hand_idx = arm.get_body_index(lab_scene.HAND_PATH.rsplit("/", 1)[-1])
    limits = np.asarray(arm.get_dof_limits())[0]

    # ------------------------------------------------------------ geometry
    tube_path = objects[SOURCE_TUBE]["path"]
    tube_col = lab_scene.collider_paths(tube_path)
    tube_r = objects[SOURCE_TUBE]["size"][0] / 2.0
    tube_len = objects[SOURCE_TUBE]["size"][2]
    tube_pose0 = lab_scene.get_prim_world_pose(tube_path)   # tip (local origin) + orientation
    tip0 = tube_pose0[:3]
    tube_top = tip0[2] + tube_len
    holder_top = lab_scene.BENCH_Z + 0.04 + lab_scene.HOLDER_THICKNESS_MM * lab_scene.MM_TO_M
    grip = tube_r + GRIP_CLEARANCE

    bore = cg.rotor_bores()[TARGET_SLEEVE]
    tip_at_release_s = RELEASE_TCP_S - (tube_len - GRASP_DEPTH)
    print(f"bore {TARGET_SLEEVE}: tilt {bore.tilt_deg:.1f} deg, opening s={bore.opening_s*1000:.1f}mm; "
          f"tube tip at release s={tip_at_release_s*1000:.1f}mm (seat ~-15.75mm)")
    assert SEAT_TIP_S[1] < tip_at_release_s < bore.opening_s - 0.01, \
        "release pose doesn't put the tube's tip just above its seat inside the bore"

    arm_cols = [p for p in lab_scene.collider_paths("/World/Arm")
                if "/link0/link1/" in p]  # link0 sits on the pedestal by design
    sqi = get_physx_scene_query_interface()

    def overlaps(paths):
        hits = set()
        for p in paths:
            def report(hit):
                hits.add(hit.collision)
                return True
            p0, p1 = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(p))
            sqi.overlap_shape(p0, p1, report, False)
        return hits

    def collisions(tube_mode, closing_on_tube=False):
        """Real-collider interpenetrations right now, as readable pairs.
        Pad-on-tube contact is allowed while held, and while the fingers
        close on it -- that contact is the grasp. The tube itself is checked
        while carried (kinematic: nothing may touch it but the pads). Once
        released it rests on the rotor -- an overlap query always reports a
        resting contact -- so from then on it's checked by penetration
        depth (seat check) and by not being moved (tube_disturbed)."""
        bad = []
        for h in overlaps(arm_cols):
            if h.startswith("/World/Arm"):
                continue  # self / adjacent links
            if h.startswith(tube_path) and (tube_mode == "attached" or closing_on_tube):
                continue
            bad.append(("arm", h))
        if tube_mode == "attached":
            for h in overlaps(tube_col):
                if not (h.startswith(tube_path) or h.startswith("/World/Arm")):
                    bad.append(("tube", h))
        return bad

    seated_pose = None

    def tube_disturbed():
        p = lab_scene.get_prim_world_pose(tube_path)
        return float(np.linalg.norm(p[:3] - seated_pose[:3]))

    # ----------------------------------------------------- gripper frames
    # Closing axis (hand y) horizontal and tangential to the rotor at the
    # target bore, kept the same from grasp to insertion: tilting into the
    # bore is then a pure rotation about the closing axis, so the pads' long
    # (x) side and the hand body's 21cm span stay level instead of swinging
    # down into the rotor. Of the two signs of y, pick the one whose IK
    # solutions sit farther from joint limits (the other drives joint 7 to
    # its stop).
    y_tan = np.cross(bore.axis, [0, 0, 1.0])
    y_tan /= np.linalg.norm(y_tan)
    z_down = np.array([0, 0, -1.0])

    def solve(pos, R, q0):
        return ik_solver.solve_ik_pose(arm, hand_idx, pos, R, q0, limits,
                                       lab_scene.get_tcp_pose, update, lab_scene.TCP_LOCAL)

    def limit_margin(q):
        return float(np.min(np.minimum(q[:7] - limits[:7, 0], limits[:7, 1] - q[:7])))

    best = None
    for sign in (1.0, -1.0):
        Rg = frame_from_axis(z_down, sign * y_tan)
        Ri = frame_from_axis(-bore.axis, sign * y_tan)
        q0 = lab_scene.Q_HOME.copy()
        q0[7:] = GRIPPER_OPEN
        qa, oka, *_ = solve(np.array([tip0[0], tip0[1], tube_top + HOVER]), Rg, q0)
        qb, okb, *_ = solve(bore.at(RELEASE_TCP_S + INSERT_STROKE), Ri, qa)
        m = min(limit_margin(qa), limit_margin(qb)) if (oka and okb) else -1.0
        print(f"closing-axis sign {sign:+.0f}: IK ok={oka and okb}, min joint-limit margin {m:.3f} rad")
        if best is None or m > best[0]:
            best = (m, Rg, Ri, qa)
    if best[0] <= 0:
        print("episode REJECTED: no closing-axis orientation reaches both the tube and the bore")
        simulation_app.close()
        return
    _, R_grasp, R_ins, q_start = best

    # ----------------------------------------------------------- keyframes
    # (phase, TCP position, hand orientation, finger opening)
    pre_insert = bore.at(RELEASE_TCP_S + INSERT_STROKE)
    above_tube = np.array([tip0[0], tip0[1], tube_top + HOVER])
    at_grasp = np.array([tip0[0], tip0[1], tube_top - GRASP_DEPTH])
    lifted = np.array([tip0[0], tip0[1], holder_top + LIFT_CLEARANCE + tube_len - GRASP_DEPTH])
    keyframes = [
        ("approach_source", above_tube, R_grasp, GRIPPER_OPEN),
        ("descend_to_grasp", at_grasp, R_grasp, GRIPPER_OPEN),
        ("close_gripper", at_grasp, R_grasp, grip),
        ("lift", lifted, R_grasp, grip),
        ("transit", pre_insert + [0, 0, TRANSIT_RAISE], R_ins, grip),
        ("pre_insert", pre_insert, R_ins, grip),
        ("insert", bore.at(RELEASE_TCP_S), R_ins, grip),
        ("release", bore.at(RELEASE_TCP_S), R_ins, GRIPPER_OPEN),
        ("retract", pre_insert, R_ins, GRIPPER_OPEN),
        ("retract_home", above_tube, R_grasp, GRIPPER_OPEN),
    ]
    GRASP_K = 2     # tube rigidly attached from this keyframe on
    RELEASE_K = 7   # ... until here, where it's handed to the physics solver

    # -------------------------------------------- episode workspace (temp)
    out_root = Path(args.out)
    ep_dir = out_root / f"ep_{args.ep_index:04d}"
    tmp_dir = out_root / f".tmp_ep_{args.ep_index:04d}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    for sub in ("images", "depth", "segmentation"):
        (tmp_dir / sub).mkdir(parents=True, exist_ok=True)

    seg_info = None
    rec = {k: [] for k in ("joint_pos", "joint_vel", "gripper", "ee_pose", "tcp_pose",
                           "object_poses", "commanded_q")}
    dense_q, dense_key, dense_mode = [], [], []
    wp_log = []
    T_hand_tube = None
    tube_mode = "holder"
    MODE_ID = {"holder": 0, "attached": 1, "dynamic": 2}

    def set_arm(q):
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)

    def sync_tube():
        if tube_mode == "attached":
            T = pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)) @ T_hand_tube
            lab_scene.set_prim_world_pose(tube_path, T[:3, 3], lab_scene.matrix_to_quat(T[:3, :3]))

    def capture(f):
        nonlocal seg_info
        for _ in range(6):  # let the renderer catch up with the new state
            update()
        for name in cam_names:
            _, cam = cams[name]
            frame = cam.get_current_frame()
            Image.fromarray(frame["rgb"]).save(str(tmp_dir / "images" / f"{name}_f{f:04d}.png"))
            depth = frame.get("distance_to_camera")
            if depth is not None:
                np.save(str(tmp_dir / "depth" / f"{name}_f{f:04d}.npy"), np.asarray(depth))
            seg = frame.get("instance_segmentation")
            if seg is not None:
                np.save(str(tmp_dir / "segmentation" / f"{name}_f{f:04d}.npy"), np.asarray(seg["data"]))
                seg_info = seg_info or seg["info"]
        qp = np.asarray(arm.get_joint_positions())[0]
        qv = np.asarray(arm.get_joint_velocities())[0]
        tcp, R = lab_scene.get_tcp_pose()
        rec["joint_pos"].append(qp[:7])
        rec["joint_vel"].append(qv[:7])
        rec["gripper"].append(qp[7:9])
        rec["ee_pose"].append(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))
        rec["tcp_pose"].append(np.concatenate([tcp, lab_scene.matrix_to_quat(R)]))
        rec["object_poses"].append([lab_scene.get_prim_world_pose(objects[n]["path"]) for n in obj_names])

    def reject(msg):
        print(f"  {msg}\nepisode REJECTED -- not saved")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        simulation_app.close()

    # --------------------------------------------- solve + validate + record
    q = q_start.copy()
    prev_pos, prev_R = None, None
    for k, (phase, pos, R, fing) in enumerate(keyframes):
        log = {"phase": phase, "tcp_target": pos.tolist(), "gripper": float(fing)}
        if k == RELEASE_K:
            # Open the fingers and hand the tube to the physics solver; it
            # slides the last ~2mm down the bore and seats on its own.
            tube_mode = "dynamic"
            lab_scene.set_collision_filter(tube_path, [])
            lab_scene.set_kinematic(tube_path, False)
            for t in np.linspace(0, 1, 6)[1:]:
                q[7:] = grip + t * (fing - grip)
                set_arm(q)
                update()
                dense_q.append(q.copy()); dense_key.append(-1); dense_mode.append(MODE_ID[tube_mode])
            for _ in range(SETTLE_UPDATES):
                update()
            tp = lab_scene.get_prim_world_pose(tube_path)
            for _ in range(10):
                update()
            tp2 = lab_scene.get_prim_world_pose(tube_path)
            ax = lab_scene.quat_to_matrix(tp2[3:7])[:, 2]
            contacts = lab_scene.contacts_of(tube_path)
            seat = {
                "tip_s_m": bore.s_of(tp2[:3]),
                "tip_radial_offset_m": bore.radial_offset(tp2[:3]),
                "axis_vs_bore_deg": float(np.degrees(np.arccos(np.clip(ax @ bore.axis, -1, 1)))),
                "top_s_m": bore.s_of(tp2[:3] + tube_len * ax),
                "still_moving_m": float(np.linalg.norm(tp2[:3] - tp[:3])),
                "contacts": contacts,
                "max_penetration_m": max([0.0] + [-c["min_separation"] for c in contacts.values()]),
                "arm_collisions": [list(c) for c in collisions("dynamic")],
            }
            log["seat"] = seat
            print(f"  [release] tip s={seat['tip_s_m']*1000:.2f}mm radial={seat['tip_radial_offset_m']*1000:.2f}mm "
                  f"axis-vs-bore={seat['axis_vs_bore_deg']:.2f}deg motion(10 upd)={seat['still_moving_m']*1000:.3f}mm "
                  f"max penetration={seat['max_penetration_m']*1000:.3f}mm contacts="
                  + str({k_: (round(v['min_separation']*1000, 3), v['n']) for k_, v in contacts.items()})
                  + f" arm={seat['arm_collisions']}")
            seated = (SEAT_TIP_S[0] <= seat["tip_s_m"] <= SEAT_TIP_S[1]
                      and seat["axis_vs_bore_deg"] <= SEAT_MAX_AXIS_DEG
                      and seat["still_moving_m"] < 5e-5
                      and set(contacts) == {objects["centrifuge_body"]["path"]}
                      and seat["max_penetration_m"] <= MAX_RESTING_PENETRATION
                      and not seat["arm_collisions"])
            if not seated:
                return reject("tube did not seat cleanly in the rotor bore")
            seated_pose = tp2
        else:
            # dense, collision-checked path from the previous keyframe
            if prev_pos is None:
                n = 1
            else:
                n = max(1, math.ceil(np.linalg.norm(pos - prev_pos) / MAX_STEP_POS),
                        math.ceil(np.linalg.norm(ik_solver.rotation_error(R, prev_R)) / MAX_STEP_ROT))
            f_prev = q[7]
            for i in range(1, n + 1):
                t = i / n
                p_t = pos if prev_pos is None else prev_pos + t * (pos - prev_pos)
                R_t = R if prev_R is None else slerp_R(prev_R, R, t)
                q[7:] = f_prev + t * (fing - f_prev)
                q, ok, iters, pe, re = solve(p_t, R_t, q)
                if not ok:
                    margins = np.minimum(q[:7] - limits[:7, 0], limits[:7, 1] - q[:7])
                    return reject(f"IK failed on the way to '{phase}' (step {i}/{n}): "
                                  f"pos err {pe*1000:.2f}mm, rot err {math.degrees(re):.2f}deg, "
                                  f"q={np.round(q[:7], 3)}, joint-limit margins={np.round(margins, 3)}")
                set_arm(q)
                sync_tube()
                update()
                bad = collisions(tube_mode, closing_on_tube=(k == GRASP_K))
                if bad:
                    return reject(f"collision on the way to '{phase}' (step {i}/{n}): {bad}")
                if seated_pose is not None and tube_disturbed() > 5e-5:
                    return reject(f"arm disturbed the placed tube on the way to '{phase}' "
                                  f"(moved {tube_disturbed()*1000:.3f}mm)")
                dense_q.append(q.copy()); dense_key.append(-1); dense_mode.append(MODE_ID[tube_mode])
            log.update(ik_pos_err_m=pe, ik_rot_err_deg=math.degrees(re), dense_steps=n)
            if k == GRASP_K:
                tube_mode = "attached"
                lab_scene.set_collision_filter(tube_path, GRIPPER_BODIES)
                T_hand_tube = (np.linalg.inv(pose_matrix(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)))
                               @ pose_matrix(lab_scene.get_prim_world_pose(tube_path)))
                dense_mode[-1] = MODE_ID[tube_mode]
        dense_key[-1] = k
        log["within_joint_limits"] = bool(limit_margin(q) >= 0)
        log["no_collision"] = True
        wp_log.append(log)
        capture(k)
        rec["commanded_q"].append(q.copy())
        prev_pos, prev_R = pos, R
        print(f"  keyframe {k} '{phase}' ok (dense steps so far {len(dense_q)})")

    # ---------------------------------------------------------------- save
    # action[f] = commanded joint targets (7 arm + 2 finger) for the NEXT
    # keyframe -- what a policy observing frame f should command; the last
    # frame holds its own.
    cq = np.array(rec["commanded_q"])
    action = np.concatenate([cq[1:], cq[-1:]], axis=0)
    np.savez(
        str(tmp_dir / "trajectory.npz"),
        joint_pos=np.array(rec["joint_pos"]), joint_vel=np.array(rec["joint_vel"]),
        gripper=np.array(rec["gripper"]), ee_pose=np.array(rec["ee_pose"]),
        tcp_pose=np.array(rec["tcp_pose"]), object_poses=np.array(rec["object_poses"]),
        action=action,
        dense_q=np.array(dense_q), dense_keyframe=np.array(dense_key),
        dense_tube_mode=np.array(dense_mode),
    )
    meta = {
        "task": f"pick {SOURCE_TUBE} from the holder, place it in centrifuge rotor bore {TARGET_SLEEVE}",
        "n_frames": len(keyframes),
        "frame_phases": [kf[0] for kf in keyframes],
        "cameras": cam_names,
        "object_names": obj_names,
        "source_object": SOURCE_TUBE,
        "target_rotor_bore": {
            "sleeve_part": TARGET_SLEEVE, "axis_point": bore.point.tolist(),
            "axis_dir_up": bore.axis.tolist(), "tilt_deg": bore.tilt_deg,
            "opening_s_m": bore.opening_s, "bore_radius_m": cg.BORE_RADIUS,
        },
        "grasp": {"keyframe": GRASP_K, "release_keyframe": RELEASE_K,
                  "tube_in_hand_T": T_hand_tube.tolist(),
                  "grip_value_m": grip, "grasp_depth_m": GRASP_DEPTH,
                  "release_tcp_s_m": RELEASE_TCP_S},
        "dense_path": {"tube_mode_ids": MODE_ID, "max_step_pos_m": MAX_STEP_POS,
                       "max_step_rot_deg": math.degrees(MAX_STEP_ROT),
                       "settle_updates_after_release": SETTLE_UPDATES},
        "physics": {"hz": lab_scene.PHYSICS_HZ, "tube_mass_kg": lab_scene.TUBE_MASS_KG},
        "action_spec": "action[f] = commanded joint positions (7 arm + 2 finger, rad/m) "
                       "for keyframe f+1; last frame repeats its own",
        "tcp_local_in_hand": lab_scene.TCP_LOCAL.tolist(),
        "keyframes": wp_log,
        "sequence_valid": True,
        "robot": {"joint_names": list(arm.dof_names), "link_names": list(arm.body_names),
                  "joint_limits": limits.tolist()},
        "object_bboxes": {n: objects[n]["size"] for n in obj_names},
    }
    (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if seg_info is not None:
        (tmp_dir / "segmentation_info.json").write_text(json.dumps(seg_info, indent=2, default=str))
    shutil.rmtree(ep_dir, ignore_errors=True)
    tmp_dir.rename(ep_dir)
    print(f"episode {args.ep_index}: ACCEPTED, {len(keyframes)} frames, "
          f"{len(dense_q)}-step validated dense path -> {ep_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
