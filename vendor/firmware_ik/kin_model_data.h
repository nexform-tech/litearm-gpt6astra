/* 本文件由 tools/export_model.py 自动生成 — 请勿手改。 */
/* 真源: assets/litearm/urdf/litearm_id.urdf */
/* URDF sha256[:16]=640bf04bf532e395  (DOF=7) */

#ifndef LITEARM_KIN_MODEL_DATA_H
#define LITEARM_KIN_MODEL_DATA_H

/* 关节数 (kin.c 用 _Static_assert 与本值对齐) */
#define KIN_MODEL_N 7

/* 每关节相对父系的固定放置 (URDF joint origin xyz / rpy, 顺序 = joint1..7) */
static const float ORG_XYZ[KIN_MODEL_N][3] = {
    { -2.23694604e-07, 7.53497322e-07, 0.13324993 },
    { 0, 0, 0 },
    { 0, -0.2995, 0 },
    { 0.038, 0, 0 },
    { -0.038, -0.2225, 0 },
    { 0, 0, 0 },
    { -0.074, 0, 0 },
};

static const float ORG_RPY[KIN_MODEL_N][3] = {
    { 0, 0, 1.57079265 },
    { -1.57079633, 0, 0 },
    { 1.57079633, 0, 0 },
    { -1.57079633, 0, 0 },
    { 1.57079633, 0, 0 },
    { 0, 1.57079633, 1.57079633 },
    { 1.57079633, 0, 0 },
};

/* 每关节局部转轴 (URDF axis; kin.c 只取 z 分量的符号走 m4_zrot) */
static const float AXIS[KIN_MODEL_N][3] = {
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, -1 },
    { 0, 0, 1 },
    { 0, 0, -1 },
    { 0, 0, -1 },
};

/* 末端 fixed joint 相对 joint7 的放置 */
static const float EE_XYZ[3] = { -0.053749853, 6.97627774e-07, 1.77373939e-06 };
static const float EE_RPY[3] = { 0, -1.5707881, -3.46291658e-06 };

#endif /* LITEARM_KIN_MODEL_DATA_H */
