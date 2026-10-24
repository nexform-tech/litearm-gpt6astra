# Fidelity notes

What differs between this MuJoCo demo and the real LiteArm SE + LiteGrip.
This is a developer/ops checklist, not a user guide. Update it when behavior
changes.

| Area | Simulation | Real hardware | Status |
| --- | --- | --- | --- |
| Transport | direct Python backend; no USB frames | `litearm-python` over USB CDC | not exercised on hardware |
| Control rate | 500 Hz lock-step | firmware 300 Hz control, 100 Hz status | differs |
| Actuation | MuJoCo position actuators | firmware MIT / position loop | differs |
| Joint limits | soft limits + 0.05 rad safety margin | firmware soft limits (URDF hard ∓ 1°) | aligned in safety |
| TCP | 0.110 m from the flange | same value, physically confirmed by the owner | verify on hardware |
| Gripper | contact + friction; optional weld | LiteGrip over CAN; `position_mm` is single-side | differs |
| Gripper force | contact force estimate | `grasp(force_n, hold_s)` holds force | verify hold behavior |
| Cameras | MuJoCo renderer, head + wrist | UVC / RealSense / Orbbec; fixed head, optional wrist | not exercised |
| Perception | ground-truth block + HSV candidates | vision only; no ground truth | not implemented for hardware |
| Contact | simplified friction, decimated meshes | real contact and friction | differs |
| Hold during inference | lock-step; no watchdog | firmware watchdog / fail-soft | verify |
| Enable / disable / ESTOP | not modeled | `reset()` / `enable()` / `disable()` / `emergency_stop()` | real backend must implement |
| Latency | deterministic | USB, CAN, and model latency | measure |
| Arm model | SE URDF + LiteGrip geometry | same physical arm; firmware kinematics | FK verified in sim |

## Before real motion

- [ ] Confirm the physical TCP (0.110 m is a user-confirmed value, not a measurement).
- [ ] Confirm the gripper mount orientation and total opening (single-side mm × 2).
- [ ] Calibrate the head camera or run without table-plane XY candidates.
- [ ] Verify arm hold during a long model request; add a hold thread if it sags.
- [ ] Start at low speed (`set_speed(10–15%)`, trajectory multiplier 0.1–0.2).
- [ ] Keep the emergency stop in reach; run scripted before live.
- [ ] Log `steps.jsonl` and images for every hardware run.
