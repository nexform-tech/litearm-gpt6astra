/**
 * @file kin.h
 * @brief LiteArm 7-DOF 运动学 —— 纯 C(零 ST/HAL 依赖, 可宿主编译做 golden 对照)
 *
 * 模型 = URDF(litearm/ 包) 直译: 每关节固定变换 origin(xyz,rpy)
 * 绕 axis 旋转 θ; 末端 = ee_link(ee_joint fixed)。几何真源 = 固件资产 URDF。
 * 旋转矩阵约定: R[9] 行主序 R[row*3+col], 向量列乘 v' = R·v。
 * 全部 float(单精度, 与固件一致)。
 */
#ifndef KIN_H
#define KIN_H

#define KIN_N 7   /* 关节数 */

/** 开机/首次调用前预计算固定变换(T_FIX)。FK/JAC 链内幂等自动调用。 */
void kin_init(void);

/** 末端 TCP(ee_link) 正解: pos[3] + R[9](行主序)。q 单位 rad。 */
void kin_fk(const float q[KIN_N], float pos[3], float R[9]);

/**
 * @brief 前向链: 同时给出 FK 与每关节世界轴/原点(几何雅可比所需)。
 * @param q     关节角
 * @param pos   [out] 末端位置
 * @param R     [out] 末端姿态(行主序 3x3)
 * @param z     [out] 每关节旋转轴世界方向 (3*KIN_N, 逐关节 3)
 * @param pj    [out] 每关节旋转中心世界位置 (3*KIN_N, 逐关节 3)
 */
void kin_chain(const float q[KIN_N], float pos[3], float R[9],
               float z[3 * KIN_N], float pj[3 * KIN_N]);

/** 6x7 几何雅可比 (world-aligned): J[0..21)=v 行, J[21..42)=ω 行, 逐关节 3。 */
void kin_jacobian(const float q[KIN_N], float J[6 * KIN_N]);

/** SO(3) log: R -> 旋转向量 w (轴×角); 数值处理 θ≈0/π。 */
void kin_so3_log(const float R[9], float w[3]);

/** SO(3) exp (Rodrigues): w -> R。 */
void kin_rot_exp(const float w[3], float R[9]);

/** 位姿误差 6 向量 (base, world-aligned, 与几何 J(world) 约定一致):
 *  e[0:3]=pos_d−pos; e[3:6]=rot_log(R_d·Rᵀ)。 */
void kin_pose_error(const float pos[3], const float R[9],
                    const float pos_d[3], const float R_d[9], float e[6]);

/** 固定角 ZYX RPY (rad) -> 旋转矩阵 R(行主序)。协议 pose[3:6]=rpy 用。 */
void kin_rpy_to_rot(const float rpy[3], float R[9]);

/** 旋转矩阵 -> 固定角 ZYX RPY (rad)。TCP 查询输出用。 */
void kin_rot_to_rpy(const float R[9], float rpy[3]);

#endif /* KIN_H */
