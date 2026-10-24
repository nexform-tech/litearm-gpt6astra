# Contributing

## Setup

```bash
pip install -e ".[dev,model]"
```

## Test

```bash
pytest -q
ruff check litearm_gpt6astra tests scripts
ruff format --check litearm_gpt6astra tests scripts
```

## Assets

A normal install needs no build. Maintainers can rebuild:

```bash
python scripts/build_firmware_ik.py
python scripts/build_sim_model.py \
  --arm-urdf <litearm_description.urdf> \
  --gripper-xacro <litegrip_description.urdf.xacro>
```

`.so.manifest` is a local cache; do not commit it.

## Real hardware

Read `docs/FIDELITY.md` first. The read-only probe is:

```bash
python scripts/probe_real.py --help
```
