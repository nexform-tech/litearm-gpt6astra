#!/usr/bin/env python3
"""Compile the LiteArm firmware kinematics into a host shared library.

By default this uses the vendored sources in ``vendor/firmware_ik`` so the
repository is self-contained. Set ``LITEARM_STM32_PATH`` (or pass a firmware
root) to build from a full firmware checkout instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SOURCE_NAMES = ("kin.c", "ik.c")
HEADER_NAMES = ("kin.h", "ik.h", "kin_model_data.h")
REQUIRED_SYMBOLS = ("kin_init", "kin_fk", "ik_solve")
CFLAGS = ("-shared", "-fPIC", "-O2", "-ffp-contract=off", "-Wall")
LIBS = ("-lm",)
MANIFEST_SUFFIX = ".manifest"
VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "firmware_ik"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output() -> Path:
    return _repo_root() / "litearm_gpt6astra" / "_native" / "libkinik.so"


def _kinematics_dir(firmware_root: Path) -> Path:
    return firmware_root / "User" / "litearm" / "kinematics"


def _source_dir(firmware_root: str | Path | None) -> Path:
    if firmware_root:
        return _kinematics_dir(Path(firmware_root))
    env_root = os.environ.get("LITEARM_STM32_PATH")
    if env_root:
        return _kinematics_dir(Path(env_root))
    if all((VENDOR_DIR / name).is_file() for name in SOURCE_NAMES + HEADER_NAMES):
        return VENDOR_DIR
    raise FileNotFoundError(
        "vendored firmware sources missing; pass a firmware root or set LITEARM_STM32_PATH"
    )


def _digest(inputs: list[Path]) -> str:
    h = hashlib.sha256()
    h.update("\0".join(CFLAGS).encode())
    h.update(b"\0")
    h.update("\0".join(LIBS).encode())
    h.update(b"\0")
    for path in inputs:
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _gcc_version() -> str:
    result = subprocess.run(["gcc", "--version"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()[0]


def _manifest_path(target: Path) -> Path:
    return target.with_name(target.name + MANIFEST_SUFFIX)


def _fresh(target: Path, digest: str, compiler: str | None) -> bool:
    if not target.is_file():
        return False
    try:
        manifest = json.loads(_manifest_path(target).read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("digest") == digest
        and manifest.get("compiler") == compiler
    )


def _verify(target: Path) -> None:
    code = (
        "import ctypes, sys\n"
        "lib = ctypes.CDLL(sys.argv[1])\n"
        "for name in sys.argv[2:]:\n"
        "    getattr(lib, name)\n"
        "lib.kin_init()\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(target), *REQUIRED_SYMBOLS],
        check=True,
        timeout=10,
    )


def build(firmware_root: str | Path | None = None, output: str | Path | None = None) -> Path:
    kin_dir = _source_dir(firmware_root)
    sources = [kin_dir / name for name in SOURCE_NAMES + HEADER_NAMES]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing firmware sources: {missing}")

    target = Path(output) if output else default_output()
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = _digest(sources)
    compiler = _gcc_version()
    if _fresh(target, digest, compiler):
        return target

    tmp = target.with_name(target.name + ".tmp")
    cmd = [
        "gcc",
        *CFLAGS,
        "-o",
        str(tmp),
        str(kin_dir / "kin.c"),
        str(kin_dir / "ik.c"),
        *LIBS,
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, target)
    target.chmod(0o755)
    manifest = {
        "digest": digest,
        "compiler": compiler,
        "sources": [str(p) for p in sources],
        "cflags": list(CFLAGS),
        "libs": list(LIBS),
    }
    _manifest_path(target).write_text(json.dumps(manifest, indent=2) + "\n")
    _manifest_path(target).chmod(0o644)
    _verify(target)
    return target


def main() -> int:
    output = build()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
