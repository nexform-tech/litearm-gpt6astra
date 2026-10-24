/**
 * @file ik.h
 * @brief 7-DOF DLS 阻尼最小二乘 IK(纯 C) —— 与 pylitearm Kinematics.ik 同参数族
 *
 * 世界系约定(与 kin_pose_error / kin_jacobian 一致):
 *   e = [pos_d−pos; rot_log(R_d·Rᵀ)]; dq = Jᵀ(J·Jᵀ+λI)⁻¹ e。
 * 自适应 λ、步长钳制、限位钳制、失败随机重启(LCG)。
 * 限位由调用方传入(固件用 g_params, 宿主用 config), 保持纯 C 可宿主对照。
 */
#ifndef IK_H
#define IK_H

#include <stdint.h>
#include <stdbool.h>

#include "kin.h"

/**
 * @brief DLS IK 解算末端位姿对应的关节角。
 * @param pos_d  目标位置 [3]
 * @param R_d    目标姿态(行主序 3x3)
 * @param q_seed 初始/种子关节角(可为 0)
 * @param q_min/q_max 关节限位(rad)
 * @param q_out  [out] 解(成功时为到达目标的解; 失败为迭代最优)
 * @return true=收敛到位姿容差
 */
bool ik_solve(const float pos_d[3], const float R_d[9],
              const float q_seed[KIN_N],
              const float q_min[KIN_N], const float q_max[KIN_N],
              float q_out[KIN_N]);

#endif /* IK_H */
