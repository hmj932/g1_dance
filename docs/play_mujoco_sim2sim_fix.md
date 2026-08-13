# play_mujoco.py sim2sim 回放：问题与解决（G1 醉舞）

> BeyondMimic G1 29DOF 醉舞策略在 IsaacSim/IsaacLab（PhysX）训练，`scripts/play_mujoco.py`
> 是**独立 MuJoCo PD 回放**做 sim2sim 验证（不需 Isaac/ROS）。本文记录回放从
> 「跑不起来 → 秒摔 → 跟踪质量」三阶段的根因、验证与修复。
>
> - 仓库：`G1/Beyondminic-Weilai-G1/`（GitHub `hmj932/g1_dance` / GitLab `g1-training`）
> - 脚本：`whole_body_tracking/scripts/play_mujoco.py`
> - 模型：`trained_models/dance_zui_model_4999.pt`，参考运动：`motions/dance_zui.npz`
> - MJCF：`source/.../mjcf/g1.xml`
> - 离线分析（与 scripts 分离）：`analysis/compare_mujoco_ref.py`
> - 相关 KB 卡：`beyondmimic-motion-tracking`、`dance-zui-g1`
> - 修复提交：`6edbd23`（关节 remap + base_lin_vel）

---

## TL;DR

| # | 问题 | 严重度 | 根因 | 解决 | 谁修的 |
|---|------|--------|------|------|--------|
| 0 | 起不来（启动即崩） | **阻断** | checkpoint 键名 / normalizer 键名 / fps 数组三处不匹配 | `ckpt.get(...)` 兼容键名；`_mean/_var` 回退；`fps` 取 `.flat[0]` | Cursor（c6070c7） |
| 1 | 开局秒摔 | **主因** | 关节顺序：npz/策略=PhysX BFS，g1.xml=深度优先，脚本无 remap | obs/ctrl 两边界做 Isaac↔MuJoCo 置换 | GLM/Claude |
| 1b | 自检不叫停 | 次要 | frame0 自检只 `print` 不退出 | mismatch 时 `raise SystemExit` | Cursor |
| 2 | 跟踪变差 | 质量级 | base_lin_vel 坐标系：MuJoCo 线速=world，Isaac 要 body | `quat_rotate_inv(root_quat, qvel[0:3])` | Cursor |
| 3 | 物理 sim2sim 差异 | 遗留 | MuJoCo vs PhysX 物理不同 | 不可在此脚本「修」，以 Isaac 基线评估 | — |

---

## 0. 前置：MuJoCo 根本跑不起来（启动即崩）

在关节 remap 之前还有更早一阶段：`play_mujoco.py` **一启动就崩**，进不了回放。
三个启动期 bug，都在提交 `c6070c7`（2026-08-11，Cursor + hanmingjun，"fix local MuJoCo play"）修掉。
> 这阶段是"跑不起来"，不是"跑起来摔"。先过这关，才到 §1 的秒摔。

### 0.1 策略 checkpoint 键名不匹配（KeyError）
- 现象：`load_policy` 里 `model_dict = ckpt["model_dict"]` → `KeyError: 'model_dict'`，策略加载即崩。
- 根因：BeyondMimic 的 rsl_rl checkpoint 把模型权重存成 `model_state_dict`，不是 `model_dict`。
- 修：`ckpt.get("model_dict") or ckpt.get("model_state_dict") or ckpt`，兼容两种键名。

### 0.2 obs 归一化器键名 / 字段不匹配
- 现象：找不到归一化器（或字段对不上），策略拿到未归一化的 obs（分布偏）或被跳过。
- 根因：BeyondMimic 把经验归一化器存成 `obs_norm_state_dict`（不是 `obs_normalizer`），且 mean/var 字段名是 `_mean`/`_var`（rsl_rl `RunningObsStdOps` 内部状态），不是 `mean`/`var`。
- 修：`ckpt.get("obs_normalizer") or ckpt.get("obs_norm_state_dict")`，并 `_mean`/`_var` 回退。

### 0.3 `motion_fps = float(motion_data["fps"])` 崩
- 现象：载入 motion 时崩。
- 根因：npz 里 `fps` 是 shape `(1,)` 的 ndarray（非标量），`float(ndarray)` 在新 numpy 上 `TypeError`。
- 修：`float(np.asarray(motion_data["fps"]).flat[0])`。

---

## 1. 主因：关节顺序不一致（开局秒摔）

### 1.1 现象
`python scripts/play_mujoco.py --checkpoint ... --motion ...` 启动后机器人**几乎马上摔倒**。
不是「跟踪误差变大逐渐偏」，而是「站一下就倒」——典型的**观测/动作错位**特征，而非物理差异。

### 1.2 根因
- `dance_zui.npz` 的 `joint_pos` / `joint_vel` 是 **Isaac/PhysX BFS（广度优先）顺序**：
  `L_hip_pitch, R_hip_pitch, waist_yaw, L_hip_roll, R_hip_roll, …`
  （与 npz 里 `body_pos_w` 的 BFS 体顺序同源——脚本里硬编码的 `NPZ_BODY_ORDER` 已证。）
- MuJoCo `g1.xml` 的关节是**深度优先、左右链**：
  `L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, …, R_hip_pitch, …, waist, L_arm, R_arm`。
- 训练策略的 DOF 顺序 = npz 顺序（Isaac BFS）。**play_mujoco 原版没有 Isaac↔MuJoCo 关节置换**，直接把 MuJoCo 读出的状态喂进 obs、把策略输出直接拼回 MuJoCo ctrl → 关节错位。

两顺序差异：29 个关节里 **18 个错位**。

### 1.3 三处裂缝（原 play_mujoco）
观测向量结构 `5*n_dof + 15 = 160`：`command(2n) + anchor_pos_b(3) + anchor_ori_b(6) + base_lin_vel(3) + base_ang_vel(3) + joint_pos_rel(n) + joint_vel(n) + last_action(n)`。

1. **obs 内部自相矛盾**：`command = ref_jpos` 来自 npz（**Isaac 序**），而 `joint_pos_rel` / `joint_vel` 从 `data.qpos/qvel` 直读（**MuJoCo 序**）。策略看到的「自己的关节状态」和「要跟踪的指令」对不上号——同一向量两段顺序不一致。
2. **动作施加错位**：策略输出 `action` 是 Isaac 序，却被 `target_pos = default_pos(MuJoCo) + action*scale(MuJoCo)` 按 MuJoCo 序拼回 → 动作施加到错的关节。例：左膝的 action 被加到右踝。
3. **角速度项**同样直读 MuJoCo 序，与 Isaac 序不符。

### 1.4 为什么是「站一下就倒」而不是「一开始就倒」
- 初始化其实正确：第 0 帧 == 站立 default、高度 `INIT_HEIGHT=0.76` 对、`t=0` 时 `joint_pos_rel`/`joint_vel` 全为 0（**顺序无关**），所以 **t=0 站得住**。
- 第一个 policy action 一施加就错位 → **首步即失稳**。这正解释「几乎马上摔」。
- KB 卡 `beyondmimic-motion-tracking` 的坑「MuJoCo 回放与 Isaac 训练物理不一致 → 回放顺不代表真机顺」是**物理差异**层面的，预期是跟踪变差，**不是开局秒摔**。故物理差异不是主因。

### 1.5 证据（独立验证，不靠推断）
用 numpy 对照 npz 第 0 帧与两种顺序的站立 default：

```
CHECK A  npz[0] == Isaac-BFS 顺序 default     → True   （npz 是 BFS，frame0=站立姿）
CHECK B  npz[0] == MuJoCo 顺序 default（原喂法）→ False  （同样数值落到错关节）
```
并精确命中错位例：MuJoCo 的 `right_ankle_pitch_joint`（mj-idx 10）被读成 `0.669`——那其实是 `right_knee` 的值。

### 1.6 解决：两边界置换，其余全用 Isaac 序
策略原生顺序 = Isaac BFS，所以只在 **MuJoCo↔Isaac 边界**做转换，obs 构造与 PD 计算都在 Isaac 序（和训练 env 一致）：

- 常量 `ISAAC_JOINT_ORDER`（29 名，BFS，已用 frame0 验证）。
- 置换：`isaac2mj[k] = MuJoCo idx of Isaac joint k`，`mj2isaac = np.argsort(isaac2mj)`（逆）。
- 读状态：`joint_pos = [data.qpos[adr] …][isaac2mj]`（MuJoCo→Isaac）。
- PD 力矩：Isaac 序算 `tau_isaac = kp*(target - joint_pos) - kd*joint_vel`，再 `tau_mj = tau_isaac[mj2isaac]`（Isaac→MuJoCo）写 `data.ctrl`。
- 初始化：用 `default_pos_mj`（MuJoCo 序）直写 qpos，不经过置换。
- **frame0 自检**：`np.allclose(ref_joint_pos[0], default_pos)`，对打印 ✓，错 `raise SystemExit` 叫停（见 §1.7）。

> 注：MuJoCo `g1.xml` 的 actuator(motor) 顺序 == 关节定义顺序，故 `data.ctrl` 与 `joint_dof_adr` 同序，`tau_mj` 直写 `data.ctrl` 正确。

### 1.7 frame0 自检从「警告」升级为「硬失败」
原版 mismatch 只 `print("[WARNING]")`，与注释「do NOT proceed」不符——实际不停。
改为 mismatch 时 `raise SystemExit(...)`，避免带着错误 remap 继续跑（会秒摔）。

---

## 2. base_lin_vel 坐标系（跟踪质量，非秒摔）

### 2.1 MuJoCo free joint 的 qvel 是「混合系」（实验证明）
对 free joint 做旋转实验（关重力，转 90°，设 qvel，走一步，看世界位移/姿态差）：
- `qvel[0:3]` **线速度 = WORLD 系**（转 90° about Z、`qvel=[1,0,0]` → 世界位移沿 +x，不是 body-x 应去的 +y）。
- `qvel[3:6]` **角速度 = BODY（本体）系**（旋转矩阵有限差分反推世界 ω，与 body 轴一致）。

原脚本注释「lin vel (body frame), ang vel (body frame)」对线速度是**错的**。

### 2.2 Isaac 侧要 body 系
`tracking_env_cfg.py` 的 obs 用 `mdp.base_lin_vel` / `mdp.base_ang_vel`，二者返回
`asset.data.root_lin_vel_b` / `root_ang_vel_b`（**root/body 系**）——见
`isaaclab/envs/mdp/observations.py`。

### 2.3 解决
- 线速度：`base_lin_vel = quat_rotate_inv(root_quat_w, data.qvel[0:3])`（world→body），与 Isaac `root_lin_vel_b` 对齐。
  `root_quat_w = data.qpos[3:7]`（free-joint 四元数，body→world，与 Isaac `root_quat_w` 同约定）。
- 角速度：MuJoCo 已是 body 系 == Isaac `root_ang_vel_b`，**不动**。

> t=0 base 朝向 identity，world==body，无差别；base 转起来才分叉 → 只影响跟踪质量，**不影响开局**。
> 这是 Cursor 用 IsaacLab 源码定档后落地的（脚本里 isaaclab 不在本机，源码引用见 Cursor）。

---

## 3. 遗留：MuJoCo ↔ PhysX 物理 sim2sim 差异
- MuJoCo 与 IsaacSim（PhysX）物理参数/接触模型不同，回放顺不代表真机顺（KB 卡已写）。
- 预期：跟踪误差变大，**不是开局即倒**。以 Isaac 基线 `error_body_pos=0.062m` 为参照。
- **观感注意**：MuJoCo 前段可能出现「脚抖」——对照参考 `dance_zui.npz` 与 `analysis/compare_mujoco_ref.py`，常见是贴地段 PD/物理把小误差放大（踝速约 2× 参考），**不宜单独据此改训练**；改训以 Isaac 曲线为准，观感优先 Isaac play。
- 真机部署走 `motion_tracking_controller`（sim2real，独立仓），PD/扭矩按 G1 真机标定。

---

## 4. 修复总览（代码位置）

`scripts/play_mujoco.py`（行号随改动漂移，按符号定位）：

| 位置 | 内容 |
|------|------|
| `ISAAC_JOINT_ORDER`（~L108） | Isaac BFS 29 关节顺序常量 |
| `isaac2mj` / `mj2isaac`（~L423-424） | MuJoCo↔Isaac 置换 + 逆 + name-set assert |
| frame0 自检（~L453-465） | `allclose` → ✓ 或 `raise SystemExit` |
| init `default_pos_mj`（~L480s） | 直写 qpos，不经置换 |
| 状态读取（~L597-598） | `joint_pos/joint_vel [isaac2mj]`（→Isaac） |
| PD 力矩（~L600-610） | `tau_isaac`（Isaac）→ `tau_mj`（`[mj2isaac]`）→ `data.ctrl` |
| `base_lin_vel`（~L595） | `quat_rotate_inv(root_quat_w, qvel[0:3])`（world→body） |
| `base_ang_vel`（~L596） | `data.qvel[3:6]`（body，不动） |

修复分工：
- **Cursor**（c6070c7，2026-08-11，本对话之前）：§0 三处启动期 bug（checkpoint 键名、normalizer 键名、fps 数组）。
- **GLM/Claude**（6edbd23，本对话）：§1 关节 remap（核心）、frame0 自检（初版 WARNING）、init、MuJoCo 混合系注释改正。
- **Cursor**（本对话）：§1b frame0 自检升级硬失败、§2 base_lin_vel world→body 旋转（IsaacLab 源码定档）。

---

## 5. 验证 / 复现

```bash
cd G1/Beyondminic-Weilai-G1/whole_body_tracking
python scripts/play_mujoco.py \
  --checkpoint trained_models/dance_zui_model_4999.pt \
  --motion motions/dance_zui.npz
```
预期启动日志含：
```
[mjoco] Joint remap Isaac↔MuJoCo: 18/29 joints reordered
[motion] frame 0 == default standing pose ✓ (Isaac joint order verified)
```
然后机器人应站住并跟踪醉舞，不再秒摔。

自检项：
- 看到 ✓ = remap 顺序对；若 `SystemExit`（npz frame0 ≠ Isaac default）= `ISAAC_JOINT_ORDER` 与 npz 顺序不符，别继续。
- `obs_dim` 应 = `5*29+15 = 160`，与策略首层输入一致。
- 跑起来若仍异常，看「第几步开始偏」：t=0 即倒 → 查关节 remap；跑一段后偏 → 查速度系/物理。
- 量化 MuJoCo vs 参考：`python analysis/compare_mujoco_ref.py --checkpoint ... --seconds 60 --out ...`

---

## 6. 注意 / 遗留
- `.gitmodules` 引用 GMR/GVHMR 的上游 GitHub URL，但二者 `.gitignore` 忽略且未跟踪，**不含子仓内容**，clone 不会自动拉。
- 真机部署不走 play_mujoco（仅 sim2sim 验证），走 `motion_tracking_controller`。
