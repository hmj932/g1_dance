#!/usr/bin/env python3
"""离屏渲染 G1 动作 → mp4（不走 mujoco live viewer，云桌面上不闪）。
读 GMR 的 pkl（gvhmr_to_robot.py 出的 /tmp/tennis.pkl）→ mujoco offscreen 逐帧渲 → imageio 编 mp4（imageio-ffmpeg 自带，不需要系统 ffmpeg）。

用法（在 g1_dance 根）：
  conda run -n gmr --live-stream python render_g1_motion_offscreen.py \
    --pkl /tmp/tennis.pkl --out /tmp/tennis_g1.mp4
"""
import sys, os, argparse
import mujoco, imageio, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", help="gvhmr_to_robot.py 出的 pkl（二选一）")
    ap.add_argument("--csv", help="gvhmr_to_csv 出的 csv（36 列；二选一，直接渲不用过 pkl）")
    ap.add_argument("--out", required=True, help="输出 mp4 路径")
    ap.add_argument("--fps", type=int, default=30, help="csv 模式的帧率（默认 30）")
    ap.add_argument("--gmr", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "GMR"))
    ap.add_argument("--xml", default="assets/unitree_g1/g1_mocap_29dof.xml")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--distance", type=float, default=3.5)
    args = ap.parse_args()
    if not args.pkl and not args.csv:
        ap.error("需要 --pkl 或 --csv 之一")

    if args.csv:
        # csv: 36 列 = root_xyz + root_quat_xyzw + 29 joints；转 mujoco wxyz
        m = np.loadtxt(args.csv, delimiter=",")
        root_pos = m[:, 0:3]
        root_rot = m[:, 3:7][:, [3, 0, 1, 2]]  # xyzw → wxyz (mujoco scalar-first)
        dof_pos = m[:, 7:36]
        fps = args.fps
    else:
        sys.path.insert(0, args.gmr)
        from general_motion_retargeting import load_robot_motion
        _motion_data, fps, root_pos, root_rot, dof_pos, *_ = load_robot_motion(args.pkl)
    print(f"loaded {len(root_pos)} frames, fps={fps}")
    xml_path = os.path.join(args.gmr, args.xml)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = args.distance
    cam.elevation = -10

    # 找 base body 名（follow 用）
    base_id = None
    for name in ("base_link", "pelvis", "torso_link", "body"):
        try:
            base_id = model.body(name).id; break
        except Exception:
            continue
    if base_id is None:
        base_id = 1  # 第一个非 world body

    frames = []
    n = len(root_pos)
    for i in range(n):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot[i]   # wxyz scalar-first (mujoco)，与 GMR step 一致
        data.qpos[7:7+dof_pos.shape[1]] = dof_pos[i]
        mujoco.mj_forward(model, data)
        cam.lookat = data.xpos[base_id]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
        if i % 50 == 0 or i == n - 1:
            print(f"rendered {i+1}/{n}")
    imageio.mimsave(args.out, frames, fps=int(fps))
    print(f"saved {n} frames -> {args.out}")

if __name__ == "__main__":
    main()
