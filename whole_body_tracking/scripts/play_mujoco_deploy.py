#!/usr/bin/env python3
"""Standalone MuJoCo playback for the DEPLOY-parity BeyondMimic policy (154-dim + target clamp).

Derived from play_mujoco.py (full-obs 160-dim replay). This variant replays the
Tracking-Flat-G1-Deploy-v0 policy (G1FlatDeployEnvCfg, e.g. TASK_20261121_087):

Observation (154 = 5*29 + 9, deploy Wo state-estimation variant):
    1. command: ref_joint_pos (n) + ref_joint_vel (n)
    2. motion_anchor_ori_b: 6 (rotmat first 2 cols)          [anchor_pos_b REMOVED]
    3. base_ang_vel: 3 (body frame)                          [base_lin_vel REMOVED]
    4. joint_pos_rel: n
    5. joint_vel: n
    6. last_action: n (RAW action, pre-clamp — matches deploy last_action_)

Action (deploy three-segment target clamp, mirrors mdp/actions.py DeployTargetClampAction
and deploy core.h Core::targets()):
    target = default + scale * action
    1) clamp inside joint limits ± 0.01 rad margin
    2) rate-limit vs previous clamped target (8 rad/s * policy_dt)
    3) torque-radius clamp: target ∈ q + kd*dq/kp ± effort/kp
    4) joint-limit re-clamp
    The clamped target drives the PD for the whole policy step (DECIMATION sub-steps).

It also logs the same tracking metrics as training (error_anchor_pos / error_anchor_rot /
error_body_pos over the 14 tracked bodies) and the three z-only termination checks
(anchor_pos 0.25 m, anchor_ori 0.8 rad, ee_body_pos 0.25 m on ankle_roll + wrist_yaw),
so MuJoCo runs can be compared directly against the Isaac training curves.

Usage:
    python scripts/play_mujoco_deploy.py \
        --checkpoint reports/TASK_20261121_087/checkpoints/model_4999.pt \
        --motion motions/dance_zui.npz \
        --start-time 0 --duration 30 \
        --out metrics_open30.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import mujoco
import mujoco.viewer

# ─────────────────────────────────────────────────────────────
#  G1 robot parameters (mirrors g1.py / play_mujoco.py)
# ─────────────────────────────────────────────────────────────

_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0
_W2 = _NATURAL_FREQ ** 2

_ARM_5020 = 0.003609725
_ARM_7520_14 = 0.01017752004
_ARM_7520_22 = 0.025101925
_ARM_4010 = 0.00425

_KP_5020 = _ARM_5020 * _W2
_KP_7520_14 = _ARM_7520_14 * _W2
_KP_7520_22 = _ARM_7520_22 * _W2
_KP_4010 = _ARM_4010 * _W2

_KD_5020 = 2.0 * _DAMPING_RATIO * _ARM_5020 * _NATURAL_FREQ
_KD_7520_14 = 2.0 * _DAMPING_RATIO * _ARM_7520_14 * _NATURAL_FREQ
_KD_7520_22 = 2.0 * _DAMPING_RATIO * _ARM_7520_22 * _NATURAL_FREQ
_KD_4010 = 2.0 * _DAMPING_RATIO * _ARM_4010 * _NATURAL_FREQ

_JOINT_PARAMS = {
    ".*_hip_pitch_joint":  (88.0, _KP_7520_14, _KD_7520_14),
    ".*_hip_roll_joint":   (139.0, _KP_7520_22, _KD_7520_22),
    ".*_hip_yaw_joint":    (88.0, _KP_7520_14, _KD_7520_14),
    ".*_knee_joint":       (139.0, _KP_7520_22, _KD_7520_22),
    ".*_ankle_pitch_joint": (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    ".*_ankle_roll_joint":  (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_roll_joint":   (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_pitch_joint":  (50.0, 2.0 * _KP_5020, 2.0 * _KD_5020),
    "waist_yaw_joint":    (88.0, _KP_7520_14, _KD_7520_14),
    ".*_shoulder_pitch_joint": (25.0, _KP_5020, _KD_5020),
    ".*_shoulder_roll_joint":  (25.0, _KP_5020, _KD_5020),
    ".*_shoulder_yaw_joint":   (25.0, _KP_5020, _KD_5020),
    ".*_elbow_joint":          (25.0, _KP_5020, _KD_5020),
    ".*_wrist_roll_joint":     (25.0, _KP_5020, _KD_5020),
    ".*_wrist_pitch_joint":    (5.0, _KP_4010, _KD_4010),
    ".*_wrist_yaw_joint":      (5.0, _KP_4010, _KD_4010),
}

_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

INIT_HEIGHT = 0.76

# Isaac/PhysX BFS DOF order (= npz + policy native order); MuJoCo g1.xml is depth-first.
ISAAC_JOINT_ORDER = [
    "left_hip_pitch_joint",  "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint",   "right_hip_roll_joint",  "waist_roll_joint",
    "left_hip_yaw_joint",    "right_hip_yaw_joint",   "waist_pitch_joint",
    "left_knee_joint",       "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",  "right_shoulder_roll_joint",
    "left_ankle_roll_joint",     "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",   "right_shoulder_yaw_joint",
    "left_elbow_joint",          "right_elbow_joint",
    "left_wrist_roll_joint",     "right_wrist_roll_joint",
    "left_wrist_pitch_joint",    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",     "right_wrist_yaw_joint",
]

# PhysX BFS body order of the 30 npz bodies (verified in play_mujoco.py).
NPZ_BODY_ORDER = [
    "pelvis",                    # 0
    "left_hip_pitch_link",       # 1
    "right_hip_pitch_link",      # 2
    "waist_yaw_link",            # 3
    "left_hip_roll_link",        # 4
    "right_hip_roll_link",       # 5
    "waist_roll_link",           # 6
    "left_hip_yaw_link",         # 7
    "right_hip_yaw_link",        # 8
    "torso_link",                # 9  ← ANCHOR
    "left_knee_link",            # 10
    "right_knee_link",           # 11
    "left_shoulder_pitch_link",  # 12
    "right_shoulder_pitch_link", # 13
    "left_ankle_pitch_link",     # 14
    "right_ankle_pitch_link",    # 15
    "left_shoulder_roll_link",   # 16
    "right_shoulder_roll_link",  # 17
    "left_ankle_roll_link",      # 18
    "right_ankle_roll_link",     # 19
    "left_shoulder_yaw_link",    # 20
    "right_shoulder_yaw_link",   # 21
    "left_elbow_link",           # 22
    "right_elbow_link",          # 23
    "left_wrist_roll_link",      # 24
    "right_wrist_roll_link",     # 25
    "left_wrist_pitch_link",     # 26
    "right_wrist_pitch_link",    # 27
    "left_wrist_yaw_link",       # 28
    "right_wrist_yaw_link",      # 29
]
ANCHOR_BODY = "torso_link"

# 14 tracked bodies = flat_env_cfg G1FlatEnvCfg commands.motion.body_names
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
# ee termination bodies (tracking_env_cfg TerminationsCfg.ee_body_pos)
EE_BODIES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]

# Termination thresholds (tracking_env_cfg.py TerminationsCfg)
TERM_ANCHOR_POS_Z = 0.25   # |ref anchor z − robot anchor z| (m)
TERM_ANCHOR_ORI = 0.8      # |projected-gravity z difference| (rad-equiv)
TERM_EE_POS_Z = 0.25       # any ee body |relative z error| (m)

# Deploy clamp parameters (mdp/actions.py DeployTargetClampActionCfg defaults = core.h)
MAX_TARGET_SPEED = 8.0     # rad/s
LIMIT_MARGIN = 0.01        # rad

# Sim params (tracking_env_cfg.py: 200 Hz physics, 50 Hz policy)
SIM_DT = 0.005
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION  # 0.02 s

ACTOR_HIDDEN_DIMS = [512, 256, 128]
ACTIVATION = "elu"


import re as _re


def _match_joint_params(joint_name: str) -> tuple[float, float, float]:
    for pattern, params in _JOINT_PARAMS.items():
        if _re.fullmatch(pattern, joint_name):
            return params
    raise ValueError(f"No matching params for joint: {joint_name}")


def _match_default_pos(joint_name: str) -> float:
    for pattern, pos in _DEFAULT_JOINT_POS.items():
        if _re.fullmatch(pattern, joint_name):
            return pos
    return 0.0


def build_joint_arrays(joint_names: list[str]):
    n = len(joint_names)
    default_pos = np.zeros(n)
    kp = np.zeros(n)
    kd = np.zeros(n)
    action_scale = np.zeros(n)
    for i, name in enumerate(joint_names):
        effort, k_p, k_d = _match_joint_params(name)
        default_pos[i] = _match_default_pos(name)
        kp[i] = k_p
        kd[i] = k_d
        action_scale[i] = 0.25 * effort / k_p
    return default_pos, kp, kd, action_scale


# ─────────────────────────────────────────────────────────────
#  Policy loading
# ─────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, device: str = "cpu"):
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"[policy] weights_only=True failed ({type(e).__name__}); retrying weights_only=False")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_dict = ckpt.get("model_dict") or ckpt.get("model_state_dict") or ckpt

    actor_keys = sorted([k for k in model_dict if k.startswith("actor.")])
    max_idx = max(int(k.split(".")[1]) for k in actor_keys)
    layers = []
    for i in range(max_idx + 1):
        w_key, b_key = f"actor.{i}.weight", f"actor.{i}.bias"
        if w_key in model_dict:
            layers.append((model_dict[w_key], model_dict[b_key]))

    modules = []
    for idx, (w, b) in enumerate(layers):
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = w
        linear.bias.data = b
        modules.append(linear)
        if idx < len(layers) - 1:
            modules.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*modules).to(device)
    actor.eval()

    obs_dim = layers[0][0].shape[1]
    action_dim = layers[-1][0].shape[0]
    print(f"[policy] Actor MLP: {obs_dim} → {ACTOR_HIDDEN_DIMS} → {action_dim}")

    normalizer = None
    norm_state = ckpt.get("obs_normalizer") or ckpt.get("obs_norm_state_dict")
    if norm_state is not None and isinstance(norm_state, dict):
        mean_val = norm_state.get("mean") if norm_state.get("mean") is not None else norm_state.get("_mean")
        var_val = norm_state.get("var") if norm_state.get("var") is not None else norm_state.get("_var")
        mean = torch.from_numpy(np.asarray(mean_val, dtype=np.float32)).to(device)
        var = torch.from_numpy(np.asarray(var_val, dtype=np.float32)).to(device)
        count = norm_state.get("count", 0)
        normalizer = (mean, torch.sqrt(var + 1e-6))
        print(f"[policy] Empirical normalization: count={count} dim={mean.shape[-1]}")
    else:
        print("[policy] No normalizer found in checkpoint")

    return actor, normalizer, obs_dim, action_dim


def normalize_obs(obs: torch.Tensor, normalizer) -> torch.Tensor:
    if normalizer is not None:
        mean, std = normalizer
        return (obs - mean) / std
    return obs


# ─────────────────────────────────────────────────────────────
#  Quaternion helpers (w,x,y,z)
# ─────────────────────────────────────────────────────────────

def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_rotate(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_rotate_inv(q, v):
    return quat_rotate(quat_inv(q), v)


def yaw_quat(q):
    # Isaac Lab yaw_quat: keep only yaw (zero roll/pitch), (w,x,y,z)
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_error_magnitude(q1, q2):
    """Angle of q1⁻¹·q2 (rad), matching isaaclab quat_error_magnitude."""
    qd = quat_mul(quat_inv(q1), q2)
    return float(2.0 * np.arccos(np.clip(abs(qd[0]), -1.0, 1.0)))


def subtract_frame_transform(pos1, quat1, pos2, quat2):
    pos_rel = quat_rotate_inv(quat1, pos2 - pos1)
    quat_rel = quat_mul(quat_inv(quat1), quat2)
    return pos_rel, quat_rel


# ─────────────────────────────────────────────────────────────
#  Deploy target clamp (Core::targets() replication)
# ─────────────────────────────────────────────────────────────

def clamp_targets(target, q, dq, last_target, lower, upper, kp, kd, effort, dt):
    lo = lower + LIMIT_MARGIN
    hi = upper - LIMIT_MARGIN
    t = np.clip(target, lo, hi)                                     # 1) joint limits ± margin
    delta = MAX_TARGET_SPEED * dt
    t = np.clip(t, last_target - delta, last_target + delta)        # 2) rate limit vs last target
    center = q + kd * dq / kp
    radius = effort / kp
    t = np.clip(t, center - radius, center + radius)                # 3) torque radius
    t = np.clip(t, lo, hi)                                          # 4) re-clamp limits
    return t


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy-parity MuJoCo playback (154-dim obs + clamp)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", choices=["deploy", "full"], default="deploy",
                        help="deploy=154-dim Wo obs + clamp (TASK_087); "
                             "full=160-dim full obs, no clamp (control, e.g. Aug model)")
    parser.add_argument("--no-clamp", action="store_true",
                        help="diagnostic: skip deploy clamp (deploy obs kept, raw targets)")
    parser.add_argument("--motion", type=str, required=True)
    parser.add_argument("--mjcf", type=str, default=None)
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="motion time (s) to enter at (RSI frame)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run (0 = until motion end)")
    parser.add_argument("--stop-on-trip", action="store_true",
                        help="abort the run when any training termination fires")
    parser.add_argument("--no_normalization", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", type=str, default=None, help="offscreen mp4 path")
    parser.add_argument("--record_fps", type=float, default=50.0)
    parser.add_argument("--headless", action="store_true",
                        help="no viewer, no render — metrics only, run as fast as possible")
    parser.add_argument("--out", type=str, default=None, help="metrics JSON path")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.mjcf is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.mjcf = os.path.join(
            repo_root, "source", "whole_body_tracking", "whole_body_tracking",
            "assets", "unitree_description", "mjcf", "g1.xml",
        )
        print(f"[mjcf] Auto-detected: {args.mjcf}")

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)
    print(f"[mjoco] Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")

    joint_names_dof, joint_qpos_adr, joint_dof_adr = [], [], []
    for i in range(1, model.njnt):
        joint_names_dof.append(model.joint(i).name)
        joint_qpos_adr.append(model.jnt_qposadr[i])
        joint_dof_adr.append(model.jnt_dofadr[i])
    n_dof = len(joint_names_dof)

    joint_limited = np.array([bool(model.jnt_limited[i]) for i in range(1, model.njnt)])
    jnt_range_mj = np.array(model.jnt_range[1:model.njnt])  # (n_dof, 2) MuJoCo order

    default_pos_mj, kp_mj, kd_mj, action_scale_mj = build_joint_arrays(joint_names_dof)

    isaac2mj = np.array([joint_names_dof.index(n) for n in ISAAC_JOINT_ORDER])
    mj2isaac = np.argsort(isaac2mj)
    assert set(joint_names_dof) == set(ISAAC_JOINT_ORDER), "ISAAC_JOINT_ORDER must match the MJCF joint set"
    default_pos  = default_pos_mj[isaac2mj]
    kp           = kp_mj[isaac2mj]
    kd           = kd_mj[isaac2mj]
    action_scale = action_scale_mj[isaac2mj]
    lower        = np.where(joint_limited, jnt_range_mj[:, 0], -1e6)[isaac2mj]
    upper        = np.where(joint_limited, jnt_range_mj[:, 1], 1e6)[isaac2mj]
    effort       = np.array([_match_joint_params(n)[0] for n in ISAAC_JOINT_ORDER])
    reordered = int(np.count_nonzero(mj2isaac != np.arange(n_dof)))
    print(f"[mjoco] Joint remap Isaac↔MuJoCo: {reordered}/{n_dof} joints reordered")

    motion_data = np.load(args.motion)
    ref_joint_pos = motion_data["joint_pos"].astype(np.float32)
    ref_joint_vel = motion_data["joint_vel"].astype(np.float32)
    ref_body_pos_w = motion_data["body_pos_w"].astype(np.float32)
    ref_body_quat_w = motion_data["body_quat_w"].astype(np.float32)
    motion_fps = float(np.asarray(motion_data["fps"]).flat[0])
    n_motion_steps = ref_joint_pos.shape[0]
    print(f"[motion] {n_motion_steps} frames, fps={motion_fps} ({n_motion_steps/motion_fps:.1f}s)")

    anchor_npz_idx = NPZ_BODY_ORDER.index(ANCHOR_BODY)
    tracked_npz_idx = [NPZ_BODY_ORDER.index(b) for b in TRACKED_BODIES]
    ee_npz_idx = [NPZ_BODY_ORDER.index(b) for b in EE_BODIES]
    anchor_body_mj_id = model.body(ANCHOR_BODY).id
    tracked_mj_ids = [model.body(b).id for b in TRACKED_BODIES]
    print(f"[motion] anchor '{ANCHOR_BODY}': npz_idx={anchor_npz_idx}, mj_id={anchor_body_mj_id}")

    actor, normalizer, obs_dim, action_dim = load_policy(args.checkpoint, "cpu")
    if args.no_normalization:
        normalizer = None
        print("[policy] Normalization DISABLED by user flag")
    if args.variant == "deploy":
        expected_obs_dim = 5 * n_dof + 9   # command(2n) + ori(6) + ang_vel(3) + jpos + jvel + act(n)
    else:
        expected_obs_dim = 5 * n_dof + 15  # + anchor_pos_b(3) + base_lin_vel(3)
    print(f"[obs] variant={args.variant} expected dim {expected_obs_dim} (policy expects {obs_dim})")
    if expected_obs_dim != obs_dim:
        raise SystemExit(f"Obs dim mismatch: expected {expected_obs_dim}, policy {obs_dim}")
    if action_dim != n_dof:
        print(f"[WARNING] action_dim={action_dim} != n_dof={n_dof}")

    # ── RSI: seed from the motion frame at --start-time ──
    start_frame = min(int(round(args.start_time * motion_fps)), n_motion_steps - 1)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = ref_body_pos_w[start_frame, 0]
    data.qpos[3:7] = ref_body_quat_w[start_frame, 0]
    data.qvel[0:6] = 0.0
    for isaac_i, mj_i in enumerate(isaac2mj):
        data.qpos[joint_qpos_adr[mj_i]] = ref_joint_pos[start_frame, isaac_i]
        data.qvel[joint_dof_adr[mj_i]] = ref_joint_vel[start_frame, isaac_i]
    mujoco.mj_forward(model, data)
    print(f"[init] RSI at t={args.start_time:.1f}s (frame {start_frame}, "
          f"pelvis_z={data.qpos[2]:.3f})")

    # Deploy core: last_target_ = measured joints at entry
    joint_pos = np.array([data.qpos[a] for a in joint_qpos_adr])[isaac2mj]
    joint_vel = np.array([data.qvel[a] for a in joint_dof_adr])[isaac2mj]
    last_target = joint_pos.copy()
    last_action = np.zeros(action_dim, dtype=np.float32)

    n_steps = n_motion_steps - start_frame
    if args.duration > 0:
        n_steps = min(n_steps, int(round(args.duration * motion_fps)))
    print(f"[run] {n_steps} policy steps ({n_steps * POLICY_DT:.1f}s)")

    # ── Metrics state ──
    err_anchor_pos_hist, err_anchor_rot_hist, err_body_pos_hist = [], [], []
    trip_names = ["anchor_pos", "anchor_ori", "ee_body_pos"]
    first_trip = {n: None for n in trip_names}
    trip_count = {n: 0 for n in trip_names}
    bins = []  # per-10% mean error_body_pos

    record_mode = args.record is not None
    renderer = video_writer = cam = None
    if record_mode:
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = anchor_body_mj_id
            cam.distance = 3.5
            cam.azimuth = 255.0
            cam.elevation = -15.0
            import imageio.v2 as imageio
            os.makedirs(os.path.dirname(os.path.abspath(args.record)) or ".", exist_ok=True)
            video_writer = imageio.get_writer(args.record, fps=args.record_fps,
                                              codec="libx264", macro_block_size=1)
            print(f"[record] → {args.record}")
        except Exception as e:
            print(f"[record] Offscreen rendering unavailable ({type(e).__name__}: {e}); "
                  "continuing WITHOUT video")
            renderer = video_writer = cam = None
            record_mode = False

    viewer_ctx = None if (record_mode or args.headless) else mujoco.viewer.launch_passive(model, data)
    GRAV_W = np.array([0.0, 0.0, -9.81])
    stopped_by_trip = None
    t_start = time.time()

    try:
        viewer = viewer_ctx.__enter__() if viewer_ctx is not None else None
        for step_count in range(n_steps):
            motion_step = start_frame + step_count
            t = min(motion_step, n_motion_steps - 1)

            # robot state (MuJoCo → Isaac order)
            robot_anchor_pos_w = data.xpos[anchor_body_mj_id].copy()
            robot_anchor_quat_w = data.xquat[anchor_body_mj_id].copy()
            root_quat_w = data.qpos[3:7].copy()
            base_ang_vel = data.qvel[3:6].copy()  # MuJoCo angular qvel is body-frame
            robot_body_pos_w = np.array([data.xpos[i] for i in tracked_mj_ids])

            # reference at current frame
            ref_jpos = ref_joint_pos[t]
            ref_jvel = ref_joint_vel[t]
            ref_anchor_pos_w = ref_body_pos_w[t, anchor_npz_idx]
            ref_anchor_quat_w = ref_body_quat_w[t, anchor_npz_idx]

            # ── observation (154 deploy / 160 full) ──
            obs_command = np.concatenate([ref_jpos, ref_jvel])                       # 2n
            anchor_pos_b, anchor_quat_b = subtract_frame_transform(
                robot_anchor_pos_w, robot_anchor_quat_w,
                ref_anchor_pos_w, ref_anchor_quat_w)
            obs_anchor_ori_b = quat_to_rotmat(anchor_quat_b)[:, :2].flatten()        # 6
            obs_base_ang_vel = base_ang_vel                                          # 3
            obs_joint_pos_rel = joint_pos - default_pos                              # n
            obs_joint_vel = joint_vel                                                # n
            obs_last_action = last_action                                            # n
            if args.variant == "deploy":
                obs_np = np.concatenate([
                    obs_command, obs_anchor_ori_b, obs_base_ang_vel,
                    obs_joint_pos_rel, obs_joint_vel, obs_last_action,
                ]).astype(np.float32)
            else:  # full: + anchor_pos_b(3) after command, + base_lin_vel(3) after ori
                obs_base_lin_vel = quat_rotate_inv(root_quat_w, data.qvel[0:3].copy())
                obs_np = np.concatenate([
                    obs_command, anchor_pos_b, obs_anchor_ori_b, obs_base_lin_vel,
                    obs_base_ang_vel, obs_joint_pos_rel, obs_joint_vel, obs_last_action,
                ]).astype(np.float32)

            obs_normed = normalize_obs(torch.from_numpy(obs_np).unsqueeze(0), normalizer)
            with torch.no_grad():
                action = actor(obs_normed).squeeze(0).cpu().numpy().astype(np.float32)
            last_action = action.copy()

            # ── target: deploy clamp (TASK_087) or direct (full control) ──
            target_raw = default_pos + action * action_scale
            if args.variant == "deploy" and not args.no_clamp:
                target = clamp_targets(target_raw, joint_pos, joint_vel, last_target,
                                       lower, upper, kp, kd, effort, POLICY_DT)
                last_target = target.copy()
            else:
                target = target_raw

            for _ in range(DECIMATION):
                tau_isaac = kp * (target - joint_pos) - kd * joint_vel
                tau_mj = tau_isaac[mj2isaac]
                for i, adr in enumerate(joint_dof_adr):
                    limit, _, _ = _match_joint_params(joint_names_dof[i])
                    tau_mj[i] = np.clip(tau_mj[i], -limit, limit)
                data.ctrl[:] = tau_mj
                mujoco.mj_step(model, data)
                joint_pos = np.array([data.qpos[a] for a in joint_qpos_adr])[isaac2mj]
                joint_vel = np.array([data.qvel[a] for a in joint_dof_adr])[isaac2mj]

            # ── training-parity metrics + termination checks ──
            err_anchor_pos = float(np.linalg.norm(ref_anchor_pos_w - robot_anchor_pos_w))
            err_anchor_rot = quat_error_magnitude(ref_anchor_quat_w, robot_anchor_quat_w)
            # commands.py:288-294 exact: xy from robot anchor, z from REF anchor,
            # yaw-only orientation error
            delta_pos_w = robot_anchor_pos_w.copy()
            delta_pos_w[2] = ref_anchor_pos_w[2]
            delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))
            ref_body_rel_w = delta_pos_w + quat_rotate(
                delta_ori_w, ref_body_pos_w[t, tracked_npz_idx] - ref_anchor_pos_w)
            err_body = np.linalg.norm(ref_body_rel_w - robot_body_pos_w, axis=1)
            err_body_pos = float(err_body.mean())
            err_anchor_pos_hist.append(err_anchor_pos)
            err_anchor_rot_hist.append(err_anchor_rot)
            err_body_pos_hist.append(err_body_pos)

            trip = {}
            trip["anchor_pos"] = abs(ref_anchor_pos_w[2] - robot_anchor_pos_w[2]) > TERM_ANCHOR_POS_Z
            g_motion = quat_rotate_inv(ref_anchor_quat_w, GRAV_W)
            g_robot = quat_rotate_inv(robot_anchor_quat_w, GRAV_W)
            trip["anchor_ori"] = abs(g_motion[2] - g_robot[2]) > TERM_ANCHOR_ORI
            # ee termination compares the anchor-transposed reference, same frame as error_body_pos
            ee_rows = [tracked_npz_idx.index(NPZ_BODY_ORDER.index(b)) for b in EE_BODIES]
            ee_rel = ref_body_rel_w[ee_rows] - robot_body_pos_w[ee_rows]
            trip["ee_body_pos"] = bool(np.any(np.abs(ee_rel[:, 2]) > TERM_EE_POS_Z))
            for name, fired in trip.items():
                if fired:
                    trip_count[name] += 1
                    if first_trip[name] is None:
                        first_trip[name] = (step_count + 1) * POLICY_DT
            if args.stop_on_trip and any(trip.values()):
                stopped_by_trip = [n for n, f in trip.items() if f]
                print(f"[trip] t={(step_count + 1) * POLICY_DT:.2f}s fired: {stopped_by_trip}")
                break

            # 10%-progress bins
            b = min(step_count * 10 // n_steps, 9)
            while len(bins) <= b:
                bins.append([])
            bins[b].append(err_body_pos)

            if record_mode and video_writer is not None:
                if step_count == 0:
                    mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                video_writer.append_data(renderer.render())
                if (step_count + 1) % 250 == 0:
                    print(f"[record] step {step_count + 1}/{n_steps}", flush=True)
            elif viewer_ctx is not None:
                viewer.sync()
                time.sleep(POLICY_DT)
            elif (step_count + 1) % 250 == 0:
                print(f"[headless] step {step_count + 1}/{n_steps}", flush=True)
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    wall = time.time() - t_start
    n_run = len(err_body_pos_hist)
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "variant": args.variant,
        "motion": os.path.abspath(args.motion),
        "start_time_s": args.start_time,
        "requested_duration_s": args.duration if args.duration > 0 else (n_motion_steps - start_frame) / motion_fps,
        "ran_duration_s": n_run * POLICY_DT,
        "ran_steps": n_run,
        "stopped_by_trip": stopped_by_trip,
        "first_trip_s": {k: v for k, v in first_trip.items() if v is not None},
        "trip_fraction": {k: (trip_count[k] / n_run if n_run else 0.0) for k in trip_names},
        "error_anchor_pos_mean": float(np.mean(err_anchor_pos_hist)),
        "error_anchor_pos_max": float(np.max(err_anchor_pos_hist)),
        "error_anchor_rot_mean": float(np.mean(err_anchor_rot_hist)),
        "error_anchor_rot_max": float(np.max(err_anchor_rot_hist)),
        "error_body_pos_mean": float(np.mean(err_body_pos_hist)),
        "error_body_pos_max": float(np.max(err_body_pos_hist)),
        "error_body_pos_tail10pct_mean": float(np.mean(err_body_pos_hist[int(n_run * 0.9):])) if n_run else 0.0,
        "error_body_pos_bins10pct": [float(np.mean(b)) for b in bins],
        "wall_time_s": wall,
    }

    print("\n════════ sim2sim summary (deploy parity) ════════")
    print(f"  segment: t={args.start_time:.0f}s + {summary['ran_duration_s']:.1f}s "
          f"({n_run} steps, wall {wall:.1f}s)")
    print(f"  error_body_pos:  mean {summary['error_body_pos_mean']:.4f}  "
          f"max {summary['error_body_pos_max']:.4f}  "
          f"tail10% {summary['error_body_pos_tail10pct_mean']:.4f}")
    print(f"  error_anchor_pos: mean {summary['error_anchor_pos_mean']:.4f}  "
          f"max {summary['error_anchor_pos_max']:.4f}")
    print(f"  error_anchor_rot: mean {summary['error_anchor_rot_mean']:.4f}  "
          f"max {summary['error_anchor_rot_max']:.4f}")
    print(f"  first trip: {summary['first_trip_s'] or 'NONE (clean run)'}")
    print(f"  trip fraction: { {k: round(v, 3) for k, v in summary['trip_fraction'].items()} }")
    bins10 = summary["error_body_pos_bins10pct"]
    if bins10:
        print("  error_body_pos per 10%: " +
              " ".join(f"{v:.3f}" for v in bins10))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
