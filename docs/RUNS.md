# Run log

[English](RUNS.md) | [简体中文](RUNS.zh-CN.md)

## 2026-10-24 — scripted, 10 seeds

Environment: Python 3.10, MuJoCo 3.8.1, EGL headless.

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

Result: **10/10 success**. Each episode ends with the cube inside the target
zone, on the table, with the gripper open. Final placement error is about 2–4 mm.
A 14-step episode takes about 10 s at 1x with the viewer.

## Replay

Scripted runs also write `runs/<...>/replay.jsonl` (one proposal per line).
Replay seed 0:

```bash
python -m litearm_gpt6astra.cli run --mode replay \
  --replay runs/scripted_seed_0/replay.jsonl --seed 0 \
  --out runs/replay_seed_0 --no-viewer
```

Result: `success=True score=1.000 phase=done steps=14`.

## Live Astra

Endpoint used for the live checks:

```text
api_style: chat
base_url:  https://mmu.vod-qcloud.com/v1
model:     gpt-6.1-sol
```

A 4-step alignment smoke showed the intended behavior:

- step 0: align XY over the cube and descend;
- step 1: keep XY fixed, descend 50 mm;
- step 2: switch to `near_grasp=true`, descend 5 mm;
- step 3: continue the final approach.

Demo: `docs/demo_2x.mp4` (live run).

A fixed live success rate is not asserted: the model can still choose `hold` or
`stop`, and the local stack latches `held` by design instead of retrying.

## Failure paths (unit tests)

- Low confidence (0.5) → `held`, no motion.
- `status=stop` or lost target → `held`, reason recorded.
- Gross target jump, joint limit, IK residual, joint jump → `held`.
- Trial lift does not carry the cube → `held`; the gripper is not automatically
  opened and the action is not retried.
