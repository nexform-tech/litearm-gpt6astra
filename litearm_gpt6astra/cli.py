"""litearm-astra 命令行。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import mujoco 之前设置；本机 headless 用 EGL。
os.environ.setdefault("MUJOCO_GL", "egl")

from .astra.schema import ACTION_SCHEMA
from .config import load_config


def _run_dir(root: str | Path, mode: str, seed: int) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(root) / f"{stamp}_{mode}_{seed}"


def _cmd_schema(args) -> int:
    print(json.dumps(ACTION_SCHEMA, indent=2, ensure_ascii=False))
    return 0


def _cmd_doctor(args) -> int:
    from .robot.ik import FirmwareIK

    cfg = load_config(args.config)
    print(f"config: {args.config}")
    print(f"model:  {cfg.scene.model_path}")
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(cfg.scene.model_path))
        data = mujoco.MjData(model)
        data.qpos[:7] = cfg.home_q
        mujoco.mj_forward(model, data)
        print(f"mujoco: {mujoco.__version__}  nq={model.nq}  nu={model.nu}")
    except Exception as exc:
        print(f"mujoco: FAILED {type(exc).__name__}: {exc}")
        return 1

    try:
        ik = FirmwareIK.from_source()
        p_fw, _ = ik.fk(cfg.home_q)
        p_mj = data.site("flange").xpos
        print(f"ik:     {ik.path}")
        print(f"fk diff at home: {float(((p_mj - p_fw) ** 2).sum() ** 0.5):.6f} m")
    except Exception as exc:
        print(f"ik: FAILED {type(exc).__name__}: {exc}")
        return 1

    try:
        from .robot.cameras import Cameras

        cams = Cameras(model, data)
        for name, cam in (("head", cfg.cameras.head), ("wrist", cfg.cameras.wrist)):
            frame = cams.render(cam.name, cam.width, cam.height)
            print(f"camera {name}: {frame.image.shape} std={float(frame.image.std()):.2f}")
        cams.close()
    except Exception as exc:
        print(f"render: FAILED {type(exc).__name__}: {exc}")
        return 1
    return 0


def _cmd_run(args) -> int:
    # 窗口模式必须用 GLFW；EGL 是无窗口渲染，不能开 passive viewer。
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_viewer = bool(args.viewer and has_display)
    if use_viewer:
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "egl"
        if args.viewer:
            print(
                "warning: no DISPLAY/WAYLAND_DISPLAY; running headless. "
                "Use --no-viewer to silence this.",
                file=sys.stderr,
            )
    from .astra.offline import LivePolicy, ReplayAstra, ScriptedPolicy
    from .robot.sim import SimBackend
    from .rollout import Rollout

    cfg = load_config(args.config)

    if args.mode == "live":
        key_env = args.api_key_env or cfg.astra.api_key_env
        if not os.environ.get(key_env):
            print(f"missing credential in {key_env}; live mode refused", file=sys.stderr)
            return 2
        live_policy = LivePolicy(
            model=args.model or cfg.astra.model,
            base_url=args.base_url or cfg.astra.base_url,
            key_env=key_env,
            timeout_s=args.timeout or cfg.astra.timeout_s,
            proxy=args.proxy if args.proxy is not None else cfg.astra.proxy,
            api_style=args.api_style or cfg.astra.api_style,
        )

        def make_policy():
            return live_policy

    elif args.mode == "replay":
        if not args.replay:
            print("--mode replay requires --replay PATH", file=sys.stderr)
            return 2

        def make_policy():
            return ReplayAstra(args.replay)

    else:

        def make_policy():
            return ScriptedPolicy()

    def callbacks(episode):
        def start(step, phase):
            label = "waiting for Astra" if args.mode == "live" else "proposing"
            print(f"[ep {episode:02d}][step {step:02d}] {phase}: {label} ...", flush=True)

        def done(step, before, after, status, ok, error):
            tail = f" ({error})" if error else ""
            print(
                f"[ep {episode:02d}][step {step:02d}] {before} -> {after}  "
                f"{status}  exec={'ok' if ok else 'FAIL'}{tail}",
                flush=True,
            )

        return start, done

    def make_rollout(policy, out_dir, seed, *, sim=None, episode=0):
        start, done = callbacks(episode)
        return Rollout(
            cfg,
            policy=policy,
            out_dir=out_dir,
            seed=seed,
            mode=args.mode,
            viewer=use_viewer,
            realtime=use_viewer and not args.fast,
            on_step_start=start,
            on_step_done=done,
            sim=sim,
            speed=args.viewer_speed,
        )

    episodes = int(args.episodes)
    if episodes == 1:
        out = Path(args.out) if args.out else _run_dir("runs", args.mode, args.seed)
        if args.out and (out / "result.json").exists():
            print(f"{out} already has result.json; refusing to overwrite", file=sys.stderr)
            return 2
        result = make_rollout(make_policy(), out, args.seed).run(max_steps=args.max_steps)
        print(f"run:    {out}")
        print(
            f"success={result.success} score={result.score:.3f} "
            f"phase={result.phase} steps={result.steps}"
        )
        if result.errors:
            for error in result.errors:
                print(f"  error: {error}")
        code = 0 if result.success else 1
        if use_viewer:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        return code

    parent = Path(args.out) if args.out else _run_dir("runs", args.mode, args.seed)
    parent.mkdir(parents=True, exist_ok=True)
    if episodes == 0:
        print("infinite episodes enabled; press Ctrl+C to stop", flush=True)
    results = []
    try:
        with SimBackend(
            cfg,
            viewer=use_viewer,
            realtime=use_viewer and not args.fast,
            speed=args.viewer_speed,
        ) as shared:
            episode = 0
            while episodes == 0 or episode < episodes:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                out = parent / f"episode_{episode:02d}"
                if (out / "result.json").exists():
                    out = parent / f"episode_{episode:02d}_{stamp}"
                rollout = make_rollout(
                    make_policy(),
                    out,
                    args.seed + episode,
                    sim=shared,
                    episode=episode,
                )
                result = rollout.run(max_steps=args.max_steps)
                results.append(result)
                print(
                    f"[ep {episode:02d}] success={result.success} "
                    f"score={result.score:.3f} phase={result.phase} "
                    f"steps={result.steps}",
                    flush=True,
                )
                episode += 1
                if episodes == 0 or episode < episodes:
                    if args.episode_pause > 0:
                        time.sleep(args.episode_pause)
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)

    ok = sum(1 for result in results if result.success)
    print(f"episodes: {ok}/{len(results)} success")
    if use_viewer:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if results and ok == len(results) else 1)
    return 0 if results and ok == len(results) else 1


def _cmd_replay(args) -> int:
    run = Path(args.run_dir)
    result_path = run / "result.json"
    if not result_path.is_file():
        print(f"{result_path} not found", file=sys.stderr)
        return 2
    result = json.loads(result_path.read_text())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--config", default="configs/sim.yaml")
    p.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        default=True,
        help="show the MuJoCo viewer (default)",
    )
    p.add_argument(
        "--no-viewer",
        "--headless",
        dest="viewer",
        action="store_false",
        help="run headless",
    )
    p.add_argument("--fast", action="store_true", help="do not pace the viewer at 1x")
    p.add_argument(
        "--viewer-speed",
        type=float,
        default=1.0,
        help="viewer playback speed (1.0 = real time)",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="number of episodes; 0 = run until Ctrl+C",
    )
    p.add_argument(
        "--episode-pause",
        type=float,
        default=1.0,
        help="seconds between episodes in continuous mode",
    )


def _add_live_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--proxy", default=None)
    p.add_argument("--api-style", choices=("chat", "responses"), default=None)
    p.add_argument("--timeout", type=float, default=None)


def _cmd_demo(args) -> int:
    args.mode = "scripted"
    return _cmd_run(args)


def _cmd_live(args) -> int:
    args.mode = "live"
    return _cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litearm-astra")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schema")
    p.set_defaults(func=_cmd_schema)

    p = sub.add_parser("doctor")
    p.add_argument("--config", default="configs/sim.yaml")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("demo", help="start the scripted simulation demo")
    _add_common_run_args(p)
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("live", help="start a model-driven simulation")
    _add_common_run_args(p)
    _add_live_args(p)
    p.set_defaults(func=_cmd_live)

    p = sub.add_parser("run")
    p.add_argument("--mode", choices=("scripted", "replay", "live"), default="scripted")
    p.add_argument("--replay", default=None)
    _add_common_run_args(p)
    _add_live_args(p)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("replay")
    p.add_argument("run_dir")
    p.set_defaults(func=_cmd_replay)
    return parser


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["demo"]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
