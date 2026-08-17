#!/usr/bin/env python3
"""Compare reference motion .npz vs MuJoCo policy playback (offline analysis).

Not a runtime/play entry. Lives in G1/analysis/, separate from
whole_body_tracking/scripts/. Imports play_mujoco only for shared PD / obs helpers.

Outputs (under --out):
  mujoco_vs_ref.npz
  ankle_vel.png
  foot_z.png
  ankle_err_spectrum.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WBT = REPO_ROOT / "whole_body_tracking"
sys.path.insert(0, str(WBT / "scripts"))

import mujoco  # noqa: E402
import torch  # noqa: E402
import play_mujoco as pm  # noqa: E402

DEFAULT_MOTION = WBT / "motions/dance_zui.npz"
DEFAULT_MJCF = (
    WBT
    / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
)

ANKLE = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
FOOT_BODIES = {
    "left_ankle_roll_link": 18,
    "right_ankle_roll_link": 19,
}


def hf_frac(x: np.ndarray, fps: float = 50.0, cutoff: float = 5.0) -> float:
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
    p = np.abs(spec) ** 2
    return float(p[freqs >= cutoff].sum() / (p.sum() + 1e-12))


def window_report(name: str, fps: int, ref_jp, ref_jv, ref_bp, mj_jp, mj_jv, mj_fz, a: int, b: int):
    print(f"\n===== {name} frames[{a}:{b}] ({(b - a) / fps:.0f}s) =====")
    print(
        f"{'joint':28s} {'ref_rms':>8} {'mj_rms':>8} {'|err|_rms':>9} "
        f"{'ref_hf':>7} {'mj_hf':>7} {'err_hf':>7} {'pos_err':>8}"
    )
    for n in ANKLE:
        i = pm.ISAAC_JOINT_ORDER.index(n)
        r = ref_jv[a:b, i]
        m = mj_jv[a:b, i]
        e = m - r
        pos = np.sqrt(np.mean((mj_jp[a:b, i] - ref_jp[a:b, i]) ** 2))
        print(
            f"{n:28s} {np.sqrt(np.mean(r**2)):8.3f} {np.sqrt(np.mean(m**2)):8.3f} "
            f"{np.sqrt(np.mean(e**2)):9.3f} {hf_frac(r, fps):7.3f} {hf_frac(m, fps):7.3f} "
            f"{hf_frac(e, fps):7.3f} {pos:8.3f}"
        )

    print(f"{'foot_z':28s} {'ref_std':>8} {'mj_std':>8} {'|err|_rms':>9} {'ref_hf':>7} {'mj_hf':>7}")
    for nm, bi in FOOT_BODIES.items():
        rz = ref_bp[a:b, bi, 2]
        mz = mj_fz[nm][a:b]
        e = mz - rz
        print(
            f"{nm:28s} {rz.std():8.4f} {mz.std():8.4f} {np.sqrt(np.mean(e**2)):9.4f} "
            f"{hf_frac(rz, fps):7.3f} {hf_frac(mz, fps):7.3f}"
        )


def run_playback(ckpt: Path, motion_path: Path, mjcf: Path, n: int, fps: int):
    motion = np.load(motion_path)
    ref_jp = motion["joint_pos"].astype(np.float32)
    ref_jv = motion["joint_vel"].astype(np.float32)
    ref_bp = motion["body_pos_w"].astype(np.float32)
    ref_bq = motion["body_quat_w"].astype(np.float32)
    if n > len(ref_jp):
        raise SystemExit(f"--seconds too long: need {n} frames, motion has {len(ref_jp)}")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    joint_names_dof, joint_qpos_adr, joint_dof_adr = [], [], []
    for i in range(1, model.njnt):
        joint_names_dof.append(model.joint(i).name)
        joint_qpos_adr.append(model.jnt_qposadr[i])
        joint_dof_adr.append(model.jnt_dofadr[i])
    n_dof = len(joint_names_dof)
    default_pos_mj, kp_mj, kd_mj, action_scale_mj = pm.build_joint_arrays(joint_names_dof)
    isaac2mj = np.array([joint_names_dof.index(name) for name in pm.ISAAC_JOINT_ORDER])
    mj2isaac = np.argsort(isaac2mj)
    default_pos = default_pos_mj[isaac2mj]
    kp = kp_mj[isaac2mj]
    kd = kd_mj[isaac2mj]
    action_scale = action_scale_mj[isaac2mj]

    anchor_npz_idx = 9
    anchor_body_mj_id = model.body("torso_link").id
    foot_mj_id = {nm: model.body(nm).id for nm in FOOT_BODIES}

    actor, normalizer, _obs_dim, action_dim = pm.load_policy(str(ckpt), "cpu")

    mujoco.mj_resetData(model, data)
    # Match play_mujoco: seed from motion frame 0 (standing default is wrong for
    # mid-dance starts such as LAFAN dance2).
    data.qpos[0:3] = ref_bp[0, 0]
    data.qpos[3:7] = ref_bq[0, 0]
    data.qvel[0:6] = 0.0
    for isaac_i, mj_i in enumerate(isaac2mj):
        data.qpos[joint_qpos_adr[mj_i]] = ref_jp[0, isaac_i]
        data.qvel[joint_dof_adr[mj_i]] = ref_jv[0, isaac_i]
    mujoco.mj_forward(model, data)

    mj_jp = np.zeros((n, n_dof), np.float32)
    mj_jv = np.zeros((n, n_dof), np.float32)
    mj_fz = {nm: np.zeros(n, np.float32) for nm in FOOT_BODIES}
    last_action = np.zeros(action_dim, np.float32)

    print(f"[run] logging {n} steps ({n / fps:.0f}s @ {fps} Hz) ...")
    for step in range(n):
        t = step
        robot_anchor_pos_w = data.xpos[anchor_body_mj_id].copy()
        robot_anchor_quat_w = data.xquat[anchor_body_mj_id].copy()
        root_quat_w = data.qpos[3:7].copy()
        base_lin_vel = pm.quat_rotate_inv(root_quat_w, data.qvel[0:3].copy())
        base_ang_vel = data.qvel[3:6].copy()
        joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adr])[isaac2mj]
        joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adr])[isaac2mj]

        mj_jp[step] = joint_pos
        mj_jv[step] = joint_vel
        for nm, mid in foot_mj_id.items():
            mj_fz[nm][step] = data.xpos[mid, 2]

        ref_jpos = ref_jp[t]
        ref_jvel = ref_jv[t]
        ref_anchor_pos_w = ref_bp[t, anchor_npz_idx]
        ref_anchor_quat_w = ref_bq[t, anchor_npz_idx]

        anchor_pos_b, anchor_quat_b = pm.subtract_frame_transform(
            robot_anchor_pos_w, robot_anchor_quat_w, ref_anchor_pos_w, ref_anchor_quat_w
        )
        rotmat_b = pm.quat_to_rotmat(anchor_quat_b)
        obs_np = np.concatenate(
            [
                np.concatenate([ref_jpos, ref_jvel]),
                anchor_pos_b,
                rotmat_b[:, :2].flatten(),
                base_lin_vel,
                base_ang_vel,
                joint_pos - default_pos,
                joint_vel,
                last_action,
            ]
        ).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
        obs_normed = pm.normalize_obs(obs_tensor, normalizer)
        with torch.no_grad():
            action = actor(obs_normed).squeeze(0).cpu().numpy().astype(np.float32)
        last_action = action.copy()
        target_pos = default_pos + action * action_scale

        for _ in range(pm.DECIMATION):
            tau_isaac = kp * (target_pos - joint_pos) - kd * joint_vel
            tau_mj = tau_isaac[mj2isaac]
            for i, adr in enumerate(joint_dof_adr):
                effort_limit, _, _ = pm._match_joint_params(joint_names_dof[i])
                tau_mj[i] = np.clip(tau_mj[i], -effort_limit, effort_limit)
            data.ctrl[:] = tau_mj
            mujoco.mj_step(model, data)
            joint_pos = np.array([data.qpos[adr] for adr in joint_qpos_adr])[isaac2mj]
            joint_vel = np.array([data.qvel[adr] for adr in joint_dof_adr])[isaac2mj]

        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{n}")

    return ref_jp, ref_jv, ref_bp, mj_jp, mj_jv, mj_fz


def save_plots(out: Path, fps: int, n: int, seconds: int, ref_jv, mj_jv, ref_bp, mj_fz):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[plot] skip:", exc)
        return

    t = np.arange(n) / fps
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, name in zip(axes, ANKLE):
        i = pm.ISAAC_JOINT_ORDER.index(name)
        ax.plot(t, ref_jv[:n, i], label="ref", lw=0.8)
        ax.plot(t, mj_jv[:n, i], label="mujoco", lw=0.8, alpha=0.85)
        ax.set_ylabel(name.replace("_joint", ""), fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Ankle joint velocity: reference vs MuJoCo (first {seconds}s)")
    fig.tight_layout()
    fig.savefig(out / "ankle_vel.png", dpi=120)
    print(f"[plot] {out / 'ankle_vel.png'}")

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes2[0].plot(t, ref_bp[:n, 18, 2], label="ref L z")
    axes2[0].plot(t, mj_fz["left_ankle_roll_link"], label="mj L z", alpha=0.85)
    axes2[1].plot(t, ref_bp[:n, 19, 2], label="ref R z")
    axes2[1].plot(t, mj_fz["right_ankle_roll_link"], label="mj R z", alpha=0.85)
    for ax in axes2:
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes2[0].set_title(f"Foot height z (first {seconds}s)")
    axes2[1].set_xlabel("time (s)")
    fig2.tight_layout()
    fig2.savefig(out / "foot_z.png", dpi=120)
    print(f"[plot] {out / 'foot_z.png'}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
    splits = [(0, min(20 * fps, n)), (max(n - 20 * fps, 0), n)]
    titles = [f"0-{min(20, seconds)}s err spectrum", f"last {min(20, seconds)}s err spectrum"]
    for ax, (a, b), title in zip(axes3, splits, titles):
        if b <= a:
            continue
        for name in ANKLE:
            i = pm.ISAAC_JOINT_ORDER.index(name)
            e = mj_jv[a:b, i] - ref_jv[a:b, i]
            e = e - e.mean()
            freqs = np.fft.rfftfreq(len(e), d=1 / fps)
            psd = np.abs(np.fft.rfft(e)) ** 2
            ax.plot(freqs, psd / (psd.max() + 1e-12), label=name.replace("_joint", ""), lw=0.9)
        ax.set_xlim(0, 15)
        ax.set_xlabel("Hz")
        ax.set_ylabel("norm power")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axvline(5, color="r", ls="--", lw=0.7, alpha=0.6)
    axes3[0].legend(fontsize=6)
    fig3.tight_layout()
    fig3.savefig(out / "ankle_err_spectrum.png", dpi=120)
    print(f"[plot] {out / 'ankle_err_spectrum.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="Offline: reference npz vs MuJoCo playback")
    p.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL .pt")
    p.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    p.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: G1/reports/<inferred>/analysis or ./out)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fps = args.fps
    n = args.seconds * fps
    out = args.out
    if out is None:
        out = Path.cwd() / "out"
    out.mkdir(parents=True, exist_ok=True)

    ref_jp, ref_jv, ref_bp, mj_jp, mj_jv, mj_fz = run_playback(
        args.checkpoint, args.motion, args.mjcf, n, fps
    )

    np.savez_compressed(
        out / "mujoco_vs_ref.npz",
        ref_jp=ref_jp[:n],
        ref_jv=ref_jv[:n],
        mj_jp=mj_jp,
        mj_jv=mj_jv,
        mj_fz_left=mj_fz["left_ankle_roll_link"],
        mj_fz_right=mj_fz["right_ankle_roll_link"],
        ref_fz_left=ref_bp[:n, 18, 2],
        ref_fz_right=ref_bp[:n, 19, 2],
    )
    print(f"[npz] {out / 'mujoco_vs_ref.npz'}")

    chunk = 20 * fps
    windows = []
    start = 0
    while start < n:
        end = min(start + chunk, n)
        windows.append((f"{start // fps}-{end // fps}s", start, end))
        start = end
    windows.append((f"0-{args.seconds}s FULL", 0, n))

    for name, a, b in windows:
        window_report(name, fps, ref_jp, ref_jv, ref_bp, mj_jp, mj_jv, mj_fz, a, b)

    print("\n===== SUMMARY: MuJoCo vs Ref ankle |vel| RMS ratio (mj/ref) =====")
    ankle_idx = [pm.ISAAC_JOINT_ORDER.index(name) for name in ANKLE]
    for name, a, b in windows:
        ratios = []
        for i in ankle_idx:
            r = np.sqrt(np.mean(ref_jv[a:b, i] ** 2)) + 1e-6
            m = np.sqrt(np.mean(mj_jv[a:b, i] ** 2))
            ratios.append(m / r)
        print(f"{name:16s} mean_ratio={np.mean(ratios):.2f}  per_joint={[round(x, 2) for x in ratios]}")

    save_plots(out, fps, n, args.seconds, ref_jv, mj_jv, ref_bp, mj_fz)


if __name__ == "__main__":
    main()
