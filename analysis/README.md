# G1 数据分析（与运行脚本分离）

本目录只放 **离线分析工具**：对比参考运动、MuJoCo 回放轨迹、画图、汇总指标。

| 目录 | 放什么 |
|------|--------|
| `analysis/`（本目录） | 分析脚本 |
| `whole_body_tracking/scripts/` | 训练 / play / 转换等**运行入口**，不要往里塞分析 |
| `../../reports/<TASK_ID>/`（工作区 `G1/reports/`） | 某次任务产物（ckpt / 曲线 / 视频 / `analysis/` 输出） |

默认环境：`conda run -n nav python3 ...`（需 mujoco + torch）。

## compare_mujoco_ref.py

参考 `.npz` vs MuJoCo 策略回放（踝速、脚高、高频能量）。输出到 `--out`。

```bash
cd /home/hmj/humanoid_ws/G1/Beyondminic-Weilai-G1
conda run -n nav python3 analysis/compare_mujoco_ref.py \
  --checkpoint /home/hmj/humanoid_ws/G1/reports/TASK_20260811_112/checkpoints/model_4999.pt \
  --seconds 60 \
  --out /home/hmj/humanoid_ws/G1/reports/TASK_20260811_112/analysis
```
