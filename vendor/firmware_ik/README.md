# Vendored firmware kinematics

`kin.c`, `ik.c`, `kin.h`, `ik.h`, and `kin_model_data.h` are vendored from the
LiteArm firmware so this repository can rebuild the host IK library on a new
platform. They are internal build inputs; the released package ships a prebuilt
`libkinik.so` and does not require a compiler for the supported Linux x86_64
target.

Rebuild (maintainers only):

```bash
python scripts/build_firmware_ik.py
```
