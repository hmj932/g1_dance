# 死路：MuJoCo 离线 CSV→NPZ 用于 Isaac/Flux 训练

> 状态：**已验证失败**（2026-08-14）  
> 相关脚本：`whole_body_tracking/scripts/csv_to_npz_mujoco.py`  
> 正确路径：Isaac `csv_to_npz.py`（云桌面或 Flux，见 [[beyondmimic-motion-tracking]]）

## 结论

**不要用 `csv_to_npz_mujoco.py` 产出的 npz 做 BeyondMimic Isaac Lab 训练。**  
schema 相同（joint/body 字段、fps、帧数），但 FK/关节语义与 Isaac PhysX 不一致，策略无法收敛到可用跟踪误差。

MuJoCo 离线转换仅适合「Flux Isaac convert 挂死、急需本地看 motion 形状」的**临时预览**；训练 motion 必须用 Isaac 转换的 `*_isaac.npz`。

## 对照实验（dance2_subject3，同 CSV，4096 envs，Tracking-Flat-G1-v0）

| 指标 | MuJoCo npz `TASK_20260813_173`（5000 iter 终值） | Isaac npz `TASK_20260814_072`（5000 iter 终值） | dance_zui baseline |
|------|--------------------------------------------------|-----------------------------------------------|----------------------|
| mean_reward | **~3.0** | **33.85** | 26.04 |
| mean_episode_length | **~96** | **465.9** | 472 |
| error_body_pos (m) | **~0.148** | **0.045** | 0.062 |
| error_joint_pos (rad) | — | **0.656** | 0.999 |
| MuJoCo 60s play | 约 5s 摔倒 | 未摔，joint RMSE ~0.17 rad | — |

MuJoCo 路径训满 5000 iter 仍远低于 baseline；Isaac npz **终值全面优于 dance_zui**（reward 33.85 vs 26.04，body_err 0.045 vs 0.062）。~2000 iter 时已超过 baseline。

## 根因（简述）

- `csv_to_npz_mujoco.py` 用 MuJoCo FK + 手写 Isaac BFS 映射写 body/joint；与 Isaac `robot.data.joint_pos` / PhysX 链不一致。
- 实测同 CSV：torso 位置接近，但 `joint_pos` max diff 可达 **~3.6 rad**（Isaac vs MuJoCo npz）。
- 训练 env 在 Isaac 里用 reference body/joint 算 reward；reference 与仿真语义错位 → 早摔、低 reward、高 body_err。

## 推荐流程

1. LAFAN1 CSV 进 repo（`motions/csv/`）。
2. **Isaac 转换**：云桌面 `isaaclab.sh -p scripts/csv_to_npz.py` → `motions/<name>_isaac.npz`（或 Flux convert，注意 Omniverse ext 下载可能挂）。
3. Flux `entry_train.py` + `motions/<name>_isaac.npz`。
4. 本地验证：`play_mujoco.py` + 下载的 checkpoint（与训练 npz 同源 Isaac 文件作 motion 参考）。

## 任务 ID

| Motion | 转换 | Task | 结果 |
|--------|------|------|------|
| dance2_subject3 | MuJoCo | TASK_20260813_173 | status=5，训练完成但指标失败 |
| dance2_subject3 | Isaac | TASK_20260814_072 | status=5 完成，5000 iter 全面超 dance_zui |

## 相关

- KB：`noesis-robot/kb/motion-skills/learned-policy/beyondmimic-motion-tracking.md` §常见坑
- xp：`dead-end-csv-to-npz-mujoco-for-isaac-train`
- Flux 账号：`docs/flux/accounts.md`
