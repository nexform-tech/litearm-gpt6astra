/**
 * @file kin.c
 * @brief LiteArm 7-DOF 运动学实现(纯 C) —— FK/几何雅可比/SO(3) log/exp
 *
 * 模型常量由 tools/export_model.py 从 URDF 生成 (kin_model_data.h), 禁止手改。
 * 真源: assets/litearm/urdf/litearm_id.urdf (MDH 帧几何 + HYY 辨识惯量, 换源后唯一真源)。
 * frame_child = frame_parent · T_origin · R(axis, θ), 末端经 ee_joint(fixed)。
 * 数值为生成常量的 float 表示 (源自 URDF 文本原值)。
 * R 行主序; 4x4 行主序 m[r*4+c]。
 */
#include "kin.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* 模型常量 —— 由 tools/export_model.py 从 URDF 生成, 请勿手改。      */
/* 真源: assets/litearm/urdf/litearm_id.urdf (MDH 帧 + HYY 辨识惯量)  */
/* ------------------------------------------------------------------ */
#include "kin_model_data.h"

/* 生成头与 kin.h 的关节数必须一致 (编译期兜底)
 * 静态断言写法同 User/gsusb_proto.h: GCC / armclang(AC6) 用 C11 _Static_assert;
 * ARMCC V5 (AC5, 仅 C99) 退回 typedef 数组技巧。
 * 注意 armcc V5 也会定义 __GNUC__ (GNU 兼容宏), 须用 __ARMCC_VERSION 排除 */
#if defined(__clang__) || (defined(__GNUC__) && !defined(__ARMCC_VERSION))
_Static_assert(KIN_MODEL_N == KIN_N, "kin_model_data.h 的关节数与 KIN_N 不符");
#else
typedef char kin_model_n_match[(KIN_MODEL_N == KIN_N) ? 1 : -1];
#endif

/* ------------------------------------------------------------------ */
/* 小工具: row-major 4x4                                              */
/* ------------------------------------------------------------------ */
static void m4_identity(float m[16])
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

/* out = a · b (row-major) */
static void m4_mul(float out[16], const float a[16], const float b[16])
{
    float t[16];
    int r, c, k;
    for (r = 0; r < 4; r++) {
        for (c = 0; c < 4; c++) {
            float s = 0.0f;
            for (k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
            t[r * 4 + c] = s;
        }
    }
    memcpy(out, t, 16 * sizeof(float));
}

static void m4_set_rot(float m[16], const float R[9])
{
    /* 以行主 3x3 填入左上; 平移列清 0; 右下 1 */
    int r, c;
    for (r = 0; r < 4; r++)
        for (c = 0; c < 4; c++) {
            if (r < 3 && c < 3) m[r * 4 + c] = R[r * 3 + c];
            else if (r == 3 && c == 3) m[r * 4 + c] = 1.0f;
            else m[r * 4 + c] = 0.0f;
        }
}

static void m4_set_trans(float m[16], const float t[3])
{
    m[3] = t[0]; m[7] = t[1]; m[11] = t[2];
}

static void m4_get_R(float m[16], float R[9])
{
    int r, c;
    for (r = 0; r < 3; r++)
        for (c = 0; c < 3; c++) R[r * 3 + c] = m[r * 4 + c];
}

static void m4_get_p(float m[16], float p[3])
{
    p[0] = m[3]; p[1] = m[7]; p[2] = m[11];
}

static void m4_apply_R(float m[16], const float v[3], float out[3])
{
    out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out[1] = m[4] * v[0] + m[5] * v[1] + m[6] * v[2];
    out[2] = m[8] * v[0] + m[9] * v[1] + m[10] * v[2];
}

/* URDF origin: R = Rz(yaw)·Ry(pitch)·Rx(roll); t = xyz */
static void m4_from_rpy_xyz(const float rpy[3], const float xyz[3], float out[16])
{
    float cr = cosf(rpy[0]), sr = sinf(rpy[0]);
    float cp = cosf(rpy[1]), sp = sinf(rpy[1]);
    float cy = cosf(rpy[2]), sy = sinf(rpy[2]);
    float R[9];
    /* Rx */
    float mrx[9] = {1, 0, 0,  0, cr, -sr,  0, sr, cr};
    /* Ry */
    float mry[9] = {cp, 0, sp,  0, 1, 0,  -sp, 0, cp};
    /* Rz */
    float mrz[9] = {cy, -sy, 0,  sy, cy, 0,  0, 0, 1};
    int r, c, k;
    /* R = Rz·Ry·Rx (row-major: out[r][c] = Σ mrz[r][k]·(RyRx)[k][c]) */
    float mtemp[9];
    for (r = 0; r < 3; r++)
        for (c = 0; c < 3; c++) {
            float s = 0.0f;
            for (k = 0; k < 3; k++) s += mry[r * 3 + k] * mrx[k * 3 + c];
            mtemp[r * 3 + c] = s;
        }
    for (r = 0; r < 3; r++)
        for (c = 0; c < 3; c++) {
            float s = 0.0f;
            for (k = 0; k < 3; k++) s += mrz[r * 3 + k] * mtemp[k * 3 + c];
            R[r * 3 + c] = s;
        }
    m4_set_rot(out, R);
    m4_set_trans(out, xyz);
}

/* 绕单位轴 axis 旋转 θ (Rodrigues): R = I + sK + (1−c)K² */
static void m4_from_axis_rot(const float axis[3], float th, float out[16])
{
    float kx = axis[0], ky = axis[1], kz = axis[2];
    float cc = cosf(th), ss = sinf(th);
    float K2[9], K[9] = {0, -kz, ky,  kz, 0, -kx,  -ky, kx, 0};
    float R[9];
    int r, c, k;
    /* K² */
    for (r = 0; r < 3; r++)
        for (c = 0; c < 3; c++) {
            float s2 = 0.0f;
            for (k = 0; k < 3; k++) s2 += K[r * 3 + k] * K[k * 3 + c];
            K2[r * 3 + c] = s2;
        }
    for (r = 0; r < 3; r++)
        for (c = 0; c < 3; c++) {
            float diag = (r == c) ? 1.0f : 0.0f;
            R[r * 3 + c] = diag + ss * K[r * 3 + c] + (1.0f - cc) * K2[r * 3 + c];
        }
    m4_set_rot(out, R);
    {
        static const float z3[3] = {0.0f, 0.0f, 0.0f};
        m4_set_trans(out, z3);
    }
}

/* 固定放置预计算(URDF rpy/xyz 常量, 首次使用前自动算一次):
 * 避免每帧重复算 7*6 次 rpy 三角 */
static float T_FIX[KIN_N][16];
static float T_TCP[16];
static int   s_kin_inited;

void kin_init(void)
{
    int i;
    if (s_kin_inited) return;
    for (i = 0; i < KIN_N; i++) m4_from_rpy_xyz(ORG_RPY[i], ORG_XYZ[i], T_FIX[i]);
    m4_from_rpy_xyz(EE_RPY, EE_XYZ, T_TCP);
    s_kin_inited = 1;
}

/* 绕 z 轴(±)旋转矩阵 —— 模型 7 关节旋转轴全为 (0,0,±1):
 * 直写矩阵, 比通用 Rodrigues(K/K²)每帧省 2 次 3x3 乘 */
static void m4_zrot(float out[16], float th, int sign)
{
    float c = cosf(th), s = sinf(th);
    float sg = (float)sign;
    out[0] = c;    out[1] = -sg * s; out[2] = 0.0f; out[3] = 0.0f;
    out[4] = sg * s; out[5] = c;    out[6] = 0.0f; out[7] = 0.0f;
    out[8] = 0.0f; out[9] = 0.0f;  out[10] = 1.0f; out[11] = 0.0f;
    out[12] = 0.0f; out[13] = 0.0f; out[14] = 0.0f; out[15] = 1.0f;
}

/* ------------------------------------------------------------------ */
/* 前向链                                                             */
/* ------------------------------------------------------------------ */
void kin_chain(const float q[KIN_N], float pos[3], float R[9],
               float z[3 * KIN_N], float pj[3 * KIN_N])
{
    float cur[16];
    float trot[16], tmp[16];
    int i;

    if (!s_kin_inited) kin_init();
    m4_identity(cur);
    for (i = 0; i < KIN_N; i++) {
        /* cur 先乘固定放置(预计算 T_FIX) */
        m4_mul(tmp, cur, T_FIX[i]);
        /* 关节旋转中心/轴方向 = tmp 的姿态 */
        if (z)  m4_apply_R(tmp, AXIS[i], &z[i * 3]);
        if (pj) m4_get_p(tmp, &pj[i * 3]);
        if (AXIS[i][0] == 0.0f && AXIS[i][1] == 0.0f &&
            (AXIS[i][2] == 1.0f || AXIS[i][2] == -1.0f)) {
            m4_zrot(trot, q[i], (AXIS[i][2] > 0.0f) ? 1 : -1);
        } else {
            m4_from_axis_rot(AXIS[i], q[i], trot);   /* 通用(备用, 本模型未用) */
        }
        m4_mul(cur, tmp, trot);
    }
    /* 末端 = cur · T_TCP */
    m4_mul(tmp, cur, T_TCP);
    if (pos) m4_get_p(tmp, pos);
    if (R)   m4_get_R(tmp, R);
}

void kin_fk(const float q[KIN_N], float pos[3], float R[9])
{
    kin_chain(q, pos, R, NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* 6x7 几何雅可比(world-aligned): J_v = z×(p_ee−p_i), J_w = z        */
/* ------------------------------------------------------------------ */
void kin_jacobian(const float q[KIN_N], float J[6 * KIN_N])
{
    float pos[3], R[9];
    float z[3 * KIN_N], pj[3 * KIN_N];
    int i;

    kin_chain(q, pos, R, z, pj);
    for (i = 0; i < KIN_N; i++) {
        /* p_ee − p_i; z×r 的三分量分别写 rows0..2, col i (row-major 6x7) */
        float zx = z[i * 3 + 0], zy = z[i * 3 + 1], zz = z[i * 3 + 2];
        float rx = pos[0] - pj[i * 3 + 0];
        float ry = pos[1] - pj[i * 3 + 1];
        float rz = pos[2] - pj[i * 3 + 2];
        J[0 * KIN_N + i] = zy * rz - zz * ry;   /* J_v row0 */
        J[1 * KIN_N + i] = zz * rx - zx * rz;   /* J_v row1 */
        J[2 * KIN_N + i] = zx * ry - zy * rx;   /* J_v row2 */
        J[3 * KIN_N + i] = zx;                  /* J_w row3 */
        J[4 * KIN_N + i] = zy;                  /* J_w row4 */
        J[5 * KIN_N + i] = zz;                  /* J_w row5 */
    }
}

/* ------------------------------------------------------------------ */
/* SO(3) log/exp (与 pylitearm kinematics.rot_log 逐行一致, float)    */
/* ------------------------------------------------------------------ */
void kin_so3_log(const float R[9], float w[3])
{
    float trace = R[0] + R[4] + R[8];
    float c = (trace - 1.0f) * 0.5f;
    float th;
    if (c > 1.0f) c = 1.0f; else if (c < -1.0f) c = -1.0f;
    th = acosf(c);
    if (th < 1e-8f) {
        w[0] = (R[7] - R[5]) * 0.5f;   /* R[2][1]-R[1][2] */
        w[1] = (R[2] - R[6]) * 0.5f;   /* R[0][2]-R[2][0] */
        w[2] = (R[3] - R[1]) * 0.5f;   /* R[1][0]-R[0][1] */
        return;
    }
    /* [P2-8 fix 2026-09-13] 近 π 判定改用 c=cosθ, 并在分支内**强制 th=π**。
     * 原因: 姿态差≈π 时 trace≈-1, float 舍入经 acosf 在 π 邻域被 1/sinθ 放大 ——
     * 实测 trace 的 ~1e-7 舍入会把 th 从 π-1.5e-7 放大成 π-3.5e-4, 于是原判据
     * `|th-π| < 1e-6` 落空, 落入下方通用分支得 w≈(R-Rᵀ)/2≈0 -> 姿态误差 er≈0
     * < rot_tol -> IK **假收敛**(报 ok=1 而实际姿态差 180°, fail-open, 本仓测得
     * 精确 180° 时命中率 36.6%); 且即便进了本分支, 用被污染的 th 去缩放(见下方
     * w[i]=a[i]*th) 会把 |w| 从 π 崩到 1e-3 量级。
     * 阈值 c < -0.999999 <=> θ > π-1.4e-3, 与 IK_ROT_TOL(5e-3) 相容。 */
    if (c < -0.999999f) {
        float a[3];
        float n;
        th = 3.14159265f;            /* 本分支恒定 π (绝不可用被 acos 放大的 th) */
        /* [M8] 浮点链乘误差可使对角项比 -1 低 1 ulp (转轴近垂直某坐标面时),
         * sqrtf(负)=NaN 会污染整条 IK 迭代: 先钳 [0,1] 再开方 */
        float d0 = (R[0] + 1.0f) * 0.5f; if (d0 < 0.0f) d0 = 0.0f; else if (d0 > 1.0f) d0 = 1.0f;
        float d1 = (R[4] + 1.0f) * 0.5f; if (d1 < 0.0f) d1 = 0.0f; else if (d1 > 1.0f) d1 = 1.0f;
        float d2 = (R[8] + 1.0f) * 0.5f; if (d2 < 0.0f) d2 = 0.0f; else if (d2 > 1.0f) d2 = 1.0f;
        a[0] = sqrtf(d0);
        a[1] = sqrtf(d1);
        a[2] = sqrtf(d2);
        if (a[0] > 1e-6f) {
            a[1] = copysignf(a[1], R[1]);
            a[2] = copysignf(a[2], R[2]);
        } else if (a[1] > 1e-6f) {
            a[2] = copysignf(a[2], R[5]);
        }
        n = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        if (n > 1e-9f) { a[0] /= n; a[1] /= n; a[2] /= n; }
        else { a[0] = 1.0f; a[1] = 0.0f; a[2] = 0.0f; }
        w[0] = a[0] * th; w[1] = a[1] * th; w[2] = a[2] * th;
        return;
    }
    w[0] = (R[7] - R[5]) * (th / (2.0f * sinf(th)));
    w[1] = (R[2] - R[6]) * (th / (2.0f * sinf(th)));
    w[2] = (R[3] - R[1]) * (th / (2.0f * sinf(th)));
}

void kin_rot_exp(const float w[3], float R[9])
{
    float th = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    int r, c, k;
    if (th < 1e-9f) {
        for (r = 0; r < 3; r++)
            for (c = 0; c < 3; c++) R[r * 3 + c] = (r == c) ? 1.0f : 0.0f;
        return;
    }
    {
        float kx = w[0] / th, ky = w[1] / th, kz = w[2] / th;
        float s = sinf(th), co = 1.0f - cosf(th);
        float K[9] = {0, -kz, ky,  kz, 0, -kx,  -ky, kx, 0};
        float K2[9];
        for (r = 0; r < 3; r++)
            for (c = 0; c < 3; c++) {
                float s2 = 0.0f;
                for (k = 0; k < 3; k++) s2 += K[r * 3 + k] * K[k * 3 + c];
                K2[r * 3 + c] = s2;
            }
        for (r = 0; r < 3; r++)
            for (c = 0; c < 3; c++)
                R[r * 3 + c] = ((r == c) ? 1.0f : 0.0f)
                               + s * K[r * 3 + c] + co * K2[r * 3 + c];
    }
}

void kin_pose_error(const float pos[3], const float R[9],
                    const float pos_d[3], const float R_d[9], float e[6])
{
    /* e_p = pos_d − pos */
    e[0] = pos_d[0] - pos[0];
    e[1] = pos_d[1] - pos[1];
    e[2] = pos_d[2] - pos[2];
    /* e_R = rot_log(R_d · Rᵀ) */
    {
        float Rt[9], M[9];
        int r, c, k;
        for (r = 0; r < 3; r++)
            for (c = 0; c < 3; c++) Rt[r * 3 + c] = R[c * 3 + r];
        for (r = 0; r < 3; r++)
            for (c = 0; c < 3; c++) {
                float s = 0.0f;
                for (k = 0; k < 3; k++) s += R_d[r * 3 + k] * Rt[k * 3 + c];
                M[r * 3 + c] = s;
            }
        kin_so3_log(M, &e[3]);
    }
}

void kin_rpy_to_rot(const float rpy[3], float R[9])
{
    float cr = cosf(rpy[0]), sr = sinf(rpy[0]);
    float cp = cosf(rpy[1]), sp = sinf(rpy[1]);
    float cy = cosf(rpy[2]), sy = sinf(rpy[2]);
    /* R = Rz(y)·Ry(p)·Rx(r), row-major */
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

void kin_rot_to_rpy(const float R[9], float rpy[3])
{
    float sy = sqrtf(R[0] * R[0] + R[3] * R[3]);   /* = |cos(pitch)| */
    /* ⚠ 万向锁判据的阈值必须与**可达的矩阵精度**相称, 不能用绝对的 1e-9。
     *
     * 锁附近 |pitch| -> 90°, sy = |cos(pitch)| -> 0, 而 R[0]/R[3] 自身带着约 1e-7 的
     * 浮点误差(位置/姿态链路上任何一次运算都会注入这个量级)。取 1e-9 时, sy 落在
     * [1e-9, 1e-7] 的**窄带**里仍走通用分支, 于是
     *     yaw = atan2(R[3], R[0])   // 约 1e-7 除以约 1e-7 = 噪声除噪声
     * 给出**任意**的 yaw —— 实测把 rpy=(0.30, +90°, -1.20) 的目标解成
     * (-0.785, +90°, +0.785), 偏离 176°。(2026-09-14 由独立评审经 cart_sample_* 发现。)
     *
     * 取 1e-4 后, |pitch| 距 90° 小于 0.0057° 时走锁分支: 强制 yaw=0、roll 取 r-y,
     * 重建的是**同一旋转**的另一个等价表示(锁上 yaw 本就不唯一), 误差上界约 1e-4 rad,
     * 比 IK 的旋转容差 IK_ROT_TOL=5e-3 小 50 倍。
     * 即: 用一个**有界的小误差**换掉一个**无界的大误差**。 */
    if (sy > 1e-4f) {
        rpy[2] = atan2f(R[3], R[0]);            /* yaw */
        rpy[1] = atan2f(-R[6], sy);             /* pitch */
        rpy[0] = atan2f(R[7], R[8]);            /* roll */
    } else {
        rpy[2] = 0.0f;
        rpy[1] = atan2f(-R[6], sy);
        rpy[0] = atan2f(-R[5], R[4]);           /* roll = r - y (锁上唯一可定的组合) */
    }
}
