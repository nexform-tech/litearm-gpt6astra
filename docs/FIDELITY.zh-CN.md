# 保真度说明

本页记录 MuJoCo 演示与真实 LiteArm SE + LiteGrip 的差异。它是开发/运维清单，
不是用户手册；行为变化时同步更新。

| 项目 | 仿真 | 真机 | 状态 |
| --- | --- | --- | --- |
| 传输 | Python 直接驱动，不经过 USB 帧 | `litearm-python` + USB CDC | 未在真机验证 |
| 控制拍 | 500 Hz lock-step | 固件 300 Hz 控制、100 Hz 状态 | 不同 |
| 执行器 | MuJoCo 位置执行器 | 固件 MIT / 位置环 | 不同 |
| 关节限位 | 软限位 + 0.05 rad 安全裕度 | 固件软限位（URDF 硬限位 ∓ 1°） | 安全层已对齐 |
| TCP | 距法兰 0.110 m | 同一数值，硬件方已确认 | 真机需复核 |
| 夹爪 | 接触 + 摩擦；可选 weld | LiteGrip CAN；`position_mm` 是单侧位移 | 不同 |
| 夹持力 | 接触力估计 | `grasp(force_n, hold_s)` 保力 | 需实测保持行为 |
| 相机 | MuJoCo 渲染，head + wrist | UVC / RealSense / Orbbec；固定 head，wrist 可选 | 未在真机验证 |
| 感知 | 仿真真值 + HSV 候选 | 只有视觉，没有真值 | 真机未实现 |
| 接触 | 简化摩擦、简化网格 | 真实接触与摩擦 | 不同 |
| 推理期间保持 | lock-step，无看门狗 | 固件看门狗 / fail-soft | 需实测 |
| 使能 / 失能 / 急停 | 未建模 | `reset()` / `enable()` / `disable()` / `emergency_stop()` | 真机后端必须实现 |
| 延迟 | 确定性 | USB、CAN、模型延迟 | 需测量 |
| 臂模型 | SE URDF + LiteGrip 几何 | 同一台臂；固件运动学 | 仿真 FK 已验证 |

## 真机运动前

- [ ] 复核物理 TCP（0.110 m 是用户确认值，不是实测值）。
- [ ] 复核夹爪安装方向和总开口（单侧 mm × 2）。
- [ ] 标定头相机，或去掉桌面 XY 候选。
- [ ] 实测模型请求期间手臂是否保持；下垂就加 hold 线程。
- [ ] 低速起步（`set_speed(10–15%)`，轨迹倍率 0.1–0.2）。
- [ ] 手放急停；先 scripted，后 live。
- [ ] 每次真机运行都保留 `steps.jsonl` 和图像。
