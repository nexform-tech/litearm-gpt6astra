# litearm-gpt6astra

[English](README.md) | 简体中文

GPT-6 Astra 单步视觉策略，驱动 **LiteArm SE + LiteGrip**，**MuJoCo** 仿真。



https://github.com/user-attachments/assets/3f8b19c6-2eca-44ba-b543-8aa8b27318f3



## 快速开始

要求：**Linux x86_64**、**Python 3.10+**。

```bash
git clone <repo-url>
cd litearm-gpt6astra
pip install -e .
litearm-astra
```

`litearm-astra` 启动 scripted 抓取放置演示，并打开 MuJoCo 窗口。仓库已包含仿真
资产和预编译固件 IK 库，不需要构建。

如果 `litearm-astra` 不在 `PATH` 上：

```bash
python -m litearm_gpt6astra
```

常用参数：

```bash
litearm-astra demo --no-viewer          # 不显示窗口
litearm-astra demo --fast               # 不按实时速度播放
litearm-astra demo --viewer-speed 2     # 2 倍速
litearm-astra demo --episodes 0         # 一直跑到 Ctrl+C
litearm-astra doctor                    # 只读检查
```

## 模型驱动

```bash
export LITEARM_ASTRA_API_KEY=<key>
litearm-astra live
```

默认配置：

| 配置 | 环境变量 | 默认 |
| --- | --- | --- |
| API 风格 | — | `chat`（也支持 `responses`） |
| 模型 | `LITEARM_ASTRA_MODEL` | `gpt-6.1-sol` |
| Base URL | `LITEARM_ASTRA_BASE_URL` | `https://mmu.vod-qcloud.com/v1` |
| Key | `LITEARM_ASTRA_API_KEY` | 必填 |
| 代理 | `LITEARM_ASTRA_PROXY` | 空 |

端点必须支持图片和 `response_format=json_schema`。Key 只从环境变量读取。

连续观测，viewer 一直开着：

```bash
litearm-astra live --episodes 10 --episode-pause 1.0 --out runs/live_loop
litearm-astra live --episodes 0 --out runs/live_loop    # 一直跑到 Ctrl+C
```

自定义端点或模型：

```bash
litearm-astra live --base-url https://example/v1 --model <model>
litearm-astra live --api-style responses --base-url https://example/v1
```

## 命令

| 命令 | 作用 |
| --- | --- |
| `litearm-astra` | scripted 演示，窗口打开 |
| `litearm-astra demo` | 同上；支持 `--no-viewer`、`--fast`、`--viewer-speed`、`--episodes` |
| `litearm-astra live` | 模型驱动一轮；需要 `LITEARM_ASTRA_API_KEY` |
| `litearm-astra run --mode scripted\|replay\|live` | 高级入口 |
| `litearm-astra replay <run-dir>` | 打印保存的 `result.json` |
| `litearm-astra doctor` | 加载模型、IK、相机；不运动 |

## 运行时会看到什么

- 窗口默认打开，整轮运行期间不关闭。
- 每步在终端打印一行，例如
  `[ep 00][step 09] approach -> pregrasp  grasp  exec=ok`。
- live 请求期间场景静止，终端打印 `waiting for Astra ...`。
- 校验失败、IK 失败或抓取失败会锁存 `held`；不自动重试，也不自动回 home。

## 产物

```text
runs/<timestamp>_<mode>_<seed>/
├── result.json
├── steps.jsonl
├── config.json
├── ik.json
├── images/
└── replay.jsonl
```

`steps.jsonl` 把模型原始 proposal 和实际执行结果分开记录。
`replay.jsonl` 可用 `litearm-astra run --mode replay` 回放。

## 排错

| 现象 | 处理 |
| --- | --- |
| `missing credential` | `export LITEARM_ASTRA_API_KEY=<key>` |
| 没有窗口 | 检查 `DISPLAY` / `WAYLAND_DISPLAY`，或加 `--no-viewer` |
| 锁存 `held` | 查看 `steps.jsonl` 和 `images/`；按设计不自动重试 |
| 找不到 `litearm-astra` | 用 `python -m litearm_gpt6astra`，或把脚本加入 `PATH` |

[运行记录](docs/RUNS.md) · [贡献](CONTRIBUTING.md)
