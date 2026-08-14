#!/usr/bin/env python3
"""Offline LAFAN1 CSV → BeyondMimic motion NPZ via MuJoCo FK (no Isaac).

Mirrors ``csv_to_npz.py`` math (30→50 fps interp + finite-diff velocities)
but runs FK in MuJoCo and writes bodies/joints in **Isaac/PhysX BFS order**
so the NPZ is drop-in compatible with ``Tracking-Flat-G1-v0``.

Why this exists: Flux Isaac ``csv_to_npz`` can hang on Omniverse extension
downloads when spawning URDF; local MuJoCo was tried as a shortcut.

**WARNING (verified 2026-08-14):** Output is NOT valid for Isaac Lab /
Flux training (``entry_train.py``). Same npz schema, wrong FK/joint semantics
vs PhysX — TASK_173 dance2: reward~3 @5000 iter vs Isaac npz TASK_072
reward~30 @~2050 iter. Use Isaac ``csv_to_npz.py`` for training motion;
this script is for local inspection / debugging only. See
``docs/G1/csv_to_npz_mujoco_dead_end.md``.

Usage::

    conda run -n nav python3 scripts/csv_to_npz_mujoco.py \\
      --input_file motions/csv/dance2_subject3.csv \\
      --input_fps 30 --output_fps 50 \\
      --output_file motions/dance2_subject3.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

# Must match play_mujoco.ISAAC_JOINT_ORDER / csv_to_npz joint_names.
ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Must match play_mujoco.NPZ_BODY_ORDER (PhysX BFS body list in dance_zui.npz).
NPZ_BODY_ORDER = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical lerp for wxyz quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def _so3_derivative(rots: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity from wxyz quaternion sequence (same idea as csv_to_npz)."""
    q_prev, q_next = rots[:-2], rots[2:]
    # q_rel = q_next * conj(q_prev)
    q_prev_conj = q_prev * np.array([1, -1, -1, -1])
    x0, y0, z0, w0 = q_prev_conj[:, 1], q_prev_conj[:, 2], q_prev_conj[:, 3], q_prev_conj[:, 0]
    x1, y1, z1, w1 = q_next[:, 1], q_next[:, 2], q_next[:, 3], q_next[:, 0]
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    axis = np.stack([x, y, z], axis=-1)
    sin_half = np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-8)
    angle = 2.0 * np.arctan2(sin_half.squeeze(-1), np.clip(w, -1.0, 1.0))
    axis_angle = axis / sin_half * angle[:, None]
    mid = axis_angle / (2.0 * dt)
    # pad ends
    out = np.vstack([mid[:1], mid, mid[-1:]])
    return out.astype(np.float32)


def interpolate_motion(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    dof: np.ndarray,
    input_fps: int,
    output_fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    n_in = base_pos.shape[0]
    duration = (n_in - 1) * input_dt
    times = np.arange(0.0, duration, output_dt, dtype=np.float64)
    phase = times / duration
    index_0 = np.floor(phase * (n_in - 1)).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, n_in - 1)
    blend = phase * (n_in - 1) - index_0

    pos = (1.0 - blend)[:, None] * base_pos[index_0] + blend[:, None] * base_pos[index_1]
    dof_out = (1.0 - blend)[:, None] * dof[index_0] + blend[:, None] * dof[index_1]
    quat = np.stack(
        [_slerp(base_quat_wxyz[i0], base_quat_wxyz[i1], float(b)) for i0, i1, b in zip(index_0, index_1, blend)],
        axis=0,
    )
    print(
        f"Motion interpolated, input frames: {n_in}, input fps: {input_fps}, "
        f"output frames: {pos.shape[0]}, output fps: {output_fps}"
    )
    return pos.astype(np.float32), quat.astype(np.float32), dof_out.astype(np.float32)


def convert(
    input_file: Path,
    mjcf: Path,
    output_file: Path,
    input_fps: int,
    output_fps: int,
    frame_range: tuple[int, int] | None,
) -> None:
    if frame_range is None:
        motion = np.loadtxt(input_file, delimiter=",", dtype=np.float64)
    else:
        start, end = frame_range
        motion = np.loadtxt(
            input_file,
            delimiter=",",
            dtype=np.float64,
            skiprows=start - 1,
            max_rows=end - start + 1,
        )
    if motion.ndim != 2 or motion.shape[1] != 36:
        raise SystemExit(f"Expected Nx36 CSV, got {motion.shape}")

    base_pos = motion[:, :3]
    # CSV quat is xyzw → wxyz
    base_quat = motion[:, [6, 3, 4, 5]]
    dof_isaac = motion[:, 7:]
    if dof_isaac.shape[1] != len(ISAAC_JOINT_ORDER):
        raise SystemExit(f"DOF cols {dof_isaac.shape[1]} != {len(ISAAC_JOINT_ORDER)}")

    duration = (motion.shape[0] - 1) / input_fps
    print(f"Motion loaded ({input_file}), duration: {duration:.2f} sec, frames: {motion.shape[0]}")

    base_pos, base_quat, dof_isaac = interpolate_motion(
        base_pos, base_quat, dof_isaac, input_fps, output_fps
    )
    output_dt = 1.0 / output_fps
    dof_vel_isaac = np.gradient(dof_isaac, output_dt, axis=0).astype(np.float32)

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    # MuJoCo actuated joint names in qpos/dof order (skip freejoint).
    mj_joint_names: list[str] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        mj_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    if len(mj_joint_names) != len(ISAAC_JOINT_ORDER):
        raise SystemExit(f"MJCF joints {len(mj_joint_names)} != Isaac {len(ISAAC_JOINT_ORDER)}")
    if set(mj_joint_names) != set(ISAAC_JOINT_ORDER):
        raise SystemExit("MJCF joint set != ISAAC_JOINT_ORDER")

    isaac2mj = np.array([mj_joint_names.index(n) for n in ISAAC_JOINT_ORDER], dtype=np.int64)
    # dof_mj[i] = dof_isaac[isaac_index] where isaac joint maps to mj slot i
    # isaac2mj[k] = mj index of ISAAC_JOINT_ORDER[k]
    # So mj_dof[isaac2mj[k]] = isaac_dof[k]
    body_ids = [model.body(name).id for name in NPZ_BODY_ORDER]

    n = base_pos.shape[0]
    joint_pos = np.zeros((n, 29), dtype=np.float32)
    joint_vel = np.zeros((n, 29), dtype=np.float32)
    body_pos_w = np.zeros((n, 30, 3), dtype=np.float32)
    body_quat_w = np.zeros((n, 30, 4), dtype=np.float32)

    for i in range(n):
        data.qpos[0:3] = base_pos[i]
        data.qpos[3:7] = base_quat[i]
        mj_dof = np.zeros(29, dtype=np.float64)
        mj_dof[isaac2mj] = dof_isaac[i]
        data.qpos[7:36] = mj_dof
        mujoco.mj_forward(model, data)

        joint_pos[i] = dof_isaac[i]
        joint_vel[i] = dof_vel_isaac[i]
        for bi, bid in enumerate(body_ids):
            body_pos_w[i, bi] = data.xpos[bid]
            # MuJoCo xquat is wxyz — same as Isaac Lab
            body_quat_w[i, bi] = data.xquat[bid]

    body_lin_vel_w = np.gradient(body_pos_w, output_dt, axis=0).astype(np.float32)
    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for bi in range(30):
        body_ang_vel_w[:, bi] = _so3_derivative(body_quat_w[:, bi], output_dt)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.array([output_fps], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"[INFO]: Motion saved locally to: {output_file}")
    print(f"  frames={n} joint_pos={joint_pos.shape} body_pos_w={body_pos_w.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="1-based inclusive frame range (same as csv_to_npz.py)",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml",
    )
    args = parser.parse_args()
    fr = tuple(args.frame_range) if args.frame_range else None
    convert(args.input_file, args.mjcf, args.output_file, args.input_fps, args.output_fps, fr)


if __name__ == "__main__":
    main()
