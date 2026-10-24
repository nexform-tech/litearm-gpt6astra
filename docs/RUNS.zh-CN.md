# 运行记录

[English](RUNS.md) | 简体中文

## 2026-10-24 — scripted，10 个 seed

环境：Python 3.10，MuJoCo 3.8.1，EGL headless。

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m litearm_gpt6astra.cli run --mode scripted --seed "$s" \
    --no-viewer --out "runs/scripted_seed_$s"
done
```

| seed | success | score | phase | steps |
| --- | --- | --- | --- | --- |
| 0 | true | 1.000 | done | 14 |
| 1 | true | 1.000 | done | 13 |
| 2 | true | 1.000 | done | 15 |
| 3 | true | 1.000 | done | 15 |
| 4 | true | 1.000 | done | 13 |
| 5 | true | 1.000 | done | 13 |
| 6 | true | 1.000 | done | 14 |
| 7 | true | 1.000 | done | 13 |
| 8 | true | 1.000 | done | 13 |
| 9 | true | 1.000 | done | 13 |

结果：**10/10 成功**。每轮结束时方块位于目标区内、落在桌面上、夹爪已张开；
最终落点误差约 2–4 mm。14 步的轮次在 viewer 1 倍速下约 10 秒。

## 回放

scripted 运行会同时写 `runs/<...>/replay.jsonl`（每行一个实际使用的 proposal）。
回放 seed 0：

```bash
python -m litearm_gpt6astra.cli run --mode replay \
  --replay runs/scripted_seed_0/replay.jsonl --seed 0 \
  --out runs/replay_seed_0 --no-viewer
```

结果：`success=True score=1.000 phase=done steps=14`。

## Live Astra

live 验证使用的端点：

```text
api_style: chat
base_url:  https://mmu.vod-qcloud.com/v1
model:     gpt-6.1-sol
```

4 步对准 smoke 的行为符合预期：

- 第 0 步：在 XY 上对准方块并下降；
- 第 1 步：保持 XY，下降 50 mm；
- 第 2 步：切换 `near_grasp=true`，下降 5 mm；
- 第 3 步：继续最后的接近。

演示：`docs/demo_2x.mp4`（live 运行）。

完整 live 一轮没有写成固定成功率：模型仍可能选择 `hold` 或 `stop`，本地会按设计
锁存 `held`，而不是自动重试。

## 失败路径（单元测试覆盖）

- 低置信度（0.5）→ `held`，不执行运动。
- `status=stop` 或目标丢失 → `held`，记录原因。
- 粗大目标跳变、关节限位、IK 残差、关节跳变 → `held`。
- 试抬时方块不跟手 → `held`；不自动张开夹爪，也不重试。
