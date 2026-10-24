# litearm-gpt6astra

[English](README.md) | [简体中文](README.zh-CN.md)

GPT-6 Astra single-step visual policy for **LiteArm SE + LiteGrip** in **MuJoCo**.





https://github.com/user-attachments/assets/b38f0e03-8a80-4ec7-a2c1-1b66e82743a1




## Quickstart

Requirements: **Linux x86_64**, **Python 3.10+**.

```bash
git clone <repo-url>
cd litearm-gpt6astra
pip install -e .
litearm-astra
```

`litearm-astra` starts the scripted pick-and-place demo and opens the MuJoCo
viewer. The repository includes the simulation assets and the prebuilt firmware
IK library; no build step is required.

If the console script is not on `PATH`:

```bash
python -m litearm_gpt6astra
```

Common options:

```bash
litearm-astra demo --no-viewer          # headless
litearm-astra demo --fast               # no real-time pacing
litearm-astra demo --viewer-speed 2     # 2x playback
litearm-astra demo --episodes 0         # run until Ctrl+C
litearm-astra doctor                    # read-only check
```

## Model-driven

```bash
export LITEARM_ASTRA_API_KEY=<key>
litearm-astra live
```

Defaults:

| Setting | Environment variable | Default |
| --- | --- | --- |
| API style | — | `chat` (also supports `responses`) |
| Model | `LITEARM_ASTRA_MODEL` | `gpt-6.1-sol` |
| Base URL | `LITEARM_ASTRA_BASE_URL` | `https://mmu.vod-qcloud.com/v1` |
| API key | `LITEARM_ASTRA_API_KEY` | required |
| Proxy | `LITEARM_ASTRA_PROXY` | empty |

The endpoint must accept images and `response_format=json_schema`. The key is read
from the environment only.

Continuous observation with one viewer:

```bash
litearm-astra live --episodes 10 --episode-pause 1.0 --out runs/live_loop
litearm-astra live --episodes 0 --out runs/live_loop    # until Ctrl+C
```

Custom endpoint or model:

```bash
litearm-astra live --base-url https://example/v1 --model <model>
litearm-astra live --api-style responses --base-url https://example/v1
```

## Commands

| Command | Purpose |
| --- | --- |
| `litearm-astra` | Scripted demo, viewer on |
| `litearm-astra demo` | Same; accepts `--no-viewer`, `--fast`, `--viewer-speed`, `--episodes` |
| `litearm-astra live` | Model-driven episode; requires `LITEARM_ASTRA_API_KEY` |
| `litearm-astra run --mode scripted\|replay\|live` | Advanced entry point |
| `litearm-astra replay <run-dir>` | Print a saved `result.json` |
| `litearm-astra doctor` | Load model, IK, and cameras; no motion |

## What you see

- The viewer opens by default and stays open for the whole run.
- Each step prints one terminal line, for example
  `[ep 00][step 09] approach -> pregrasp  grasp  exec=ok`.
- During a live request the scene pauses and the terminal prints
  `waiting for Astra ...`.
- Any invalid proposal, IK failure, or failed grasp latches `held`; the loop does
  not retry and does not return home automatically.

## Artifacts

```text
runs/<timestamp>_<mode>_<seed>/
├── result.json
├── steps.jsonl
├── config.json
├── ik.json
├── images/
└── replay.jsonl
```

`steps.jsonl` separates the raw model proposal from the executed result.
`replay.jsonl` can be replayed with `litearm-astra run --mode replay`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `missing credential` | `export LITEARM_ASTRA_API_KEY=<key>` |
| No viewer | Check `DISPLAY` / `WAYLAND_DISPLAY`, or use `--no-viewer` |
| `held` | Inspect `steps.jsonl` and `images/`; the loop is latched by design |
| `litearm-astra` not found | Use `python -m litearm_gpt6astra`, or add the script to `PATH` |

[Run log](docs/RUNS.md) · [Contributing](CONTRIBUTING.md)
