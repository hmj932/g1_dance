# Flux 账号使用记录

> 维护规则：每次换号、标记耗尽、或在新号上创建项目/任务后更新本表。  
> **勿写入完整 api_key**；只记邮箱、key 末 4 位、项目 ID、任务 ID。  
> 密钥池真源：`.cursor/skills/flux-cli/api_key.json`（勿提交 git）。

## 账号池概览

| 邮箱 | key 末 4 位 | 初始额度 | 状态 | 注册/入库 | 备注 |
|------|-------------|----------|------|-----------|------|
| yowokis920@primetor.com | `0e99` | 50 元 | **已耗尽** | 2026-08-11 | 首号；G1 项目 `PRO_20260811_003` |
| limxmsoavn9u0f6k6i@emalupe.com | `de9a` | 50 元 | **使用中** | 2026-08-11 | 2026-08-14 接替 yowokis920 |

## 按账号的任务记录

### yowokis920@primetor.com（`PRO_20260811_003` / G1）

| Task ID | 名称 | 状态 | 运行(s) | 说明 |
|---------|------|------|---------|------|
| TASK_20260811_056 | G1跳舞 | 6 失败 | 115 | 早期试跑 |
| TASK_20260811_109 | beyondmimic-g1-dance-train-play | 6 失败 | 119 | startScript 缺 `g1_dance/` 前缀 |
| TASK_20260811_112 | beyondmimic-g1-dance-train-play-v2 | 6/7 | 9659 | 5000 iter 训练完成；play 因 RTX driver 535.5 失败 |
| TASK_20260811_226 | beyondmimic-g1-play-video-v1 | 6 失败 | 173 | pip ReadTimeout |
| TASK_20260812_139 | beyondmimic-g1-play-video-v2 | 6 失败 | 790 | Isaac play driver 不足，手动停 |
| TASK_20260813_158 | dance2-subject3-ctp | 6 失败 | 1922 | convert 阶段 Omniverse ext 下载挂起 |
| TASK_20260813_170 | dance2-subject3-train | 6 失败 | — | 排队中被停 |
| TASK_20260813_173 | dance2-train-only (MuJoCo npz) | 5 成功 | 7514 | reward~4.5，效果差于 dance_zui |
| TASK_20260814_069 | dance2-isaac-train | 6 失败 | 213 | `entry_train.py` pip ReadTimeout（未带 mirror retry） |

**耗尽原因（2026-08-14）**：多轮训练 + play 试跑后余额不足；`TASK_20260814_069` 仅跑 ~3.5min 即因 pip 失败终止。

### limxmsoavn9u0f6k6i@emalupe.com

| 项目 ID | 名称 | 创建日 |
|---------|------|--------|
| PRO_20260814_006 | G1 | 2026-08-14 |

| Task ID | 名称 | 状态 | 说明 |
|---------|------|------|------|
| TASK_20260814_072 | beyondmimic-g1-dance2-isaac-train-v2 | 5 完成 | 5000 iter reward 33.85 / ep 466 / body_err 0.045；MuJoCo 60s 未摔 |

## 操作备忘

```bash
# 列出池
python3 .cursor/skills/flux-cli/scripts/use_account.py --list

# 标记旧号耗尽
python3 .cursor/skills/flux-cli/scripts/use_account.py --mark-exhausted --email 'yowokis920@primetor.com'

# 登录下一个可用号
python3 .cursor/skills/flux-cli/scripts/use_account.py

# 推送 g1_dance（Flux 用 GitHub，工作量查 GitLab）
cd G1/Beyondminic-Weilai-G1
ALL_PROXY=socks5://10.12.201.122:39000 git push origin main
git push gitlab main
```

## 相关

- Flux CLI skill：`.cursor/skills/flux-cli/SKILL.md`
- G1 任务记录：`G1/Beyondminic-Weilai-G1/.gm/task.yaml`
- 推送路由（GitHub 必须同步 GitLab）：`.cursor/skills/workspace-push-router/SKILL.md`
