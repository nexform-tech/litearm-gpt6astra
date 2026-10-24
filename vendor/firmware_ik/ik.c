/**
 * @file ik.c
 * @brief DLS 阻尼最小二乘 IK(纯 C)
 *
 * 与 pylitearm Kinematics.ik 同参数族(world 系版本, 用 kin_pose_error/kin_jacobian):
 *   默认: pos_tol 0.0005m, rot_tol 0.005rad, λ0 0.001, max_iter 200,
 *         step_clamp 0.3rad, 失败随机重启 8 次(幅度 0.4·(k+1))。
 * λ 自适应: 代价降 ×0.5(下界 λ0·1e-3), 升 ×2(上界 λ0·1e3)。
 * 6x6 线性求解 A·x=b (A=J·Jᵀ+λI 对称正定) 用高斯消元(部分主元)。
 * 随机重启用简单 LCG(固定种子, 与 numpy 不同但仅作多样起点)。
 */
#include "ik.h"

#include <math.h>
#include <string.h>

/* 默认参数(与 pylitearm _IK_DEFAULTS 对齐) */
#define IK_POS_TOL   0.0005f
#define IK_ROT_TOL   0.005f
#define IK_LAM       0.001f
#define IK_MAX_ITER  200
#define IK_STEP_CLAMP 0.3f
#define IK_RESTARTS  16
/* [P1-6] 全请求累计迭代预算: 无界重启在 main 上下文可阻塞数十 ms
 * (状态帧/gsusb/ACK 断流); 预算耗尽即按当前最优失败返回。
 *
 * [P1-5 fix 2026-09-13] 原值 600 会**先于"重启次数"生效**: 首轮 dls_iterate 最多
 * 耗 IK_MAX_ITER=200, 余 400 只够再跑 2 次重启 -> 把 16 次随机重启砍成 2~3 次。
 * 实测 (tools/ik_check_host.py, seed=0 全零): C 版 18/100 vs Python 100/100;
 * 隔离沙箱只改本常量: 600->18, 800->22, 1200->40, 2000->97, 3400->99。
 * 现取 IK_MAX_ITER*(IK_RESTARTS+2), 保证"首轮 + 全部 16 次重启"跑得完 ——
 * 仍是**有界**的兜底(不是取消预算), 但不再提前掐断正常重启序列。
 * ⚠ 改 IK_RESTARTS / IK_MAX_ITER 时必须同步 (本式自动跟随)。 */
#define IK_TOTAL_ITER_BUDGET  (IK_MAX_ITER * (IK_RESTARTS + 2))
#define IK_AMP_STEP  0.5f

/* ---- LCG 伪随机(固定种子可复现) ---- */
typedef struct { uint32_t s; } rng_t;
static uint32_t lcg_next(rng_t *r) {
    r->s = r->s * 1664525u + 1013904223u;
    return r->s;
}
static float rng_uni(rng_t *r, float lo, float hi) {
    return lo + (hi - lo) * (float)(lcg_next(r) & 0xFFFFFFu) / 16777216.0f;
}

static void clamp_q(float q[KIN_N], const float q_min[KIN_N], const float q_max[KIN_N]) {
    int i;
    for (i = 0; i < KIN_N; i++) {
        if (q[i] < q_min[i]) q[i] = q_min[i];
        else if (q[i] > q_max[i]) q[i] = q_max[i];
    }
}

/* 6x6 高斯消元(部分主元)求解 A·x = b; 成功返回 1 */
static int solve6(const float A[36], const float b[6], float x[6])
{
    float M[36], y[6];
    int i, j, k;
    memcpy(M, A, 36 * sizeof(float));
    memcpy(y, b, 6 * sizeof(float));
    for (k = 0; k < 6; k++) {
        /* 找主元 */
        int piv = k;
        float amax = fabsf(M[k * 6 + k]);
        for (i = k + 1; i < 6; i++) {
            float a = fabsf(M[i * 6 + k]);
            if (a > amax) { amax = a; piv = i; }
        }
        if (amax < 1e-20f) return 0;
        if (piv != k) {
            for (j = k; j < 6; j++) { float t = M[k * 6 + j]; M[k * 6 + j] = M[piv * 6 + j]; M[piv * 6 + j] = t; }
            { float t = y[k]; y[k] = y[piv]; y[piv] = t; }
        }
        {
            float d = M[k * 6 + k];
            for (j = k; j < 6; j++) M[k * 6 + j] /= d;
            y[k] /= d;
        }
        for (i = 0; i < 6; i++) {
            if (i == k) continue;
            float f = M[i * 6 + k];
            if (f == 0.0f) continue;
            for (j = k; j < 6; j++) M[i * 6 + j] -= f * M[k * 6 + j];
            y[i] -= f * y[k];
        }
    }
    memcpy(x, y, 6 * sizeof(float));
    return 1;
}

/* 单次 DLS 迭代求解; 返回 (ok, ep, er)。iters_used 出参 = 本次实际消耗
 * 迭代数 (供 [P1-6] 总预算扣减; 可为 NULL) */
static int dls_iterate(const float pos_d[3], const float R_d[9],
                       const float q0[KIN_N], const float q_min[KIN_N],
                       const float q_max[KIN_N],
                       float *ep_out, float *er_out, float q_out[KIN_N],
                       int *iters_used)
{
    float q[KIN_N];
    float best_q[KIN_N];
    float best_cost = 1e30f, best_ep = 1e30f, best_er = 1e30f;
    float lam = IK_LAM;
    float prev_cost = 0.0f;
    int has_prev = 0;
    int iter;

    memcpy(q, q0, KIN_N * sizeof(float));
    clamp_q(q, q_min, q_max);

    /* [N-1 fix] best_q 必须预置为合法起点。原实现只依赖循环内 "cost < best_cost"
     * 这一个写入点: 目标 pose 任一分量 |x| > ~1.845e19 (= sqrtf(FLT_MAX)) 时
     * e[i]^2 溢出 float -> ep=inf -> cost=inf -> 该比较恒假 -> best_q 整个
     * IK_MAX_ITER 一次都没写过, 而函数末尾 "memcpy(q_out, best_q)" 无条件执行,
     * 把【栈残留】交给调用方 (未定义行为; 经 CMD_GET_IK 0x42 一帧 52B 可达,
     * 且 usb_cmd.c 的 f32_finite 只排除 NaN/Inf、不限制量级)。
     * 预置为钳后的 seed: 失败路径外发的是合法构型; best_ep/best_er 仍为 1e30f,
     * 返回值依旧 false, 调用方照旧按失败处理。 */
    memcpy(best_q, q, KIN_N * sizeof(float));

    for (iter = 0; iter < IK_MAX_ITER; iter++) {
        float pos[3], R[9], e[6], J[6 * KIN_N];
        float ep, er, cost;
        float A[36], rhs[6], d[6], dq[KIN_N];
        int i, r, c;
        kin_fk(q, pos, R);
        kin_pose_error(pos, R, pos_d, R_d, e);
        ep = sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        er = sqrtf(e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
        cost = ep + er;
        if (cost < best_cost) {
            best_cost = cost;
            best_ep = ep;
            best_er = er;
            memcpy(best_q, q, KIN_N * sizeof(float));
        }
        if (ep < IK_POS_TOL && er < IK_ROT_TOL) {
            memcpy(q_out, q, KIN_N * sizeof(float));
            *ep_out = ep; *er_out = er;
            if (iters_used) *iters_used = iter + 1;   /* [P1-6] */
            return 1;
        }
        if (has_prev) {
            if (cost < prev_cost) lam = fmaxf(lam * 0.5f, IK_LAM * 1e-3f);
            else                 lam = fminf(lam * 2.0f, IK_LAM * 1e3f);
        }
        prev_cost = cost;
        has_prev = 1;

        kin_jacobian(q, J);
        /* A = J·Jᵀ + λI (6x6) */
        for (r = 0; r < 6; r++) {
            for (c = 0; c < 6; c++) {
                float s = 0.0f;
                for (i = 0; i < KIN_N; i++) s += J[r * KIN_N + i] * J[c * KIN_N + i];
                A[r * 6 + c] = s + ((r == c) ? lam : 0.0f);
            }
        }
        /* rhs = e */
        memcpy(rhs, e, 6 * sizeof(float));
        if (!solve6(A, rhs, d)) {
            /* 奇异到不可解: 保持当前, 提前返回最优 */
            break;
        }
        /* dq = Jᵀ · d */
        for (i = 0; i < KIN_N; i++) {
            float s = 0.0f;
            for (r = 0; r < 6; r++) s += J[r * KIN_N + i] * d[r];
            dq[i] = s;
        }
        {
            float n = 0.0f;
            for (i = 0; i < KIN_N; i++) n += dq[i] * dq[i];
            n = sqrtf(n);
            if (n > IK_STEP_CLAMP) {
                float sc = IK_STEP_CLAMP / n;
                for (i = 0; i < KIN_N; i++) dq[i] *= sc;
            }
        }
        for (i = 0; i < KIN_N; i++) q[i] += dq[i];
        clamp_q(q, q_min, q_max);
    }
    *ep_out = best_ep;
    *er_out = best_er;
    memcpy(q_out, best_q, KIN_N * sizeof(float));
    /* [P1-6] 正常耗尽 iter==IK_MAX_ITER; 奇异 break 时该次迭代已消耗 */
    if (iters_used) *iters_used = (iter < IK_MAX_ITER) ? iter + 1 : iter;
    return (best_ep < IK_POS_TOL && best_er < IK_ROT_TOL);
}

bool ik_solve(const float pos_d[3], const float R_d[9],
              const float q_seed[KIN_N],
              const float q_min[KIN_N], const float q_max[KIN_N],
              float q_out[KIN_N])
{
    float seed[KIN_N];
    float ep, er;
    int k, used;
    int budget = IK_TOTAL_ITER_BUDGET;      /* [P1-6] 总迭代预算 */
    memcpy(seed, q_seed, KIN_N * sizeof(float));
    clamp_q(seed, q_min, q_max);

    if (dls_iterate(pos_d, R_d, seed, q_min, q_max, &ep, &er, q_out, &used)) {
        return true;
    }
    budget -= used;
    if (budget <= 0) return false;          /* [P1-6] 首轮已耗尽: 直接失败 */
    /* 失败: 随机重启(固定种子, 幅度递增) */
    {
        rng_t r;
        r.s = 305419896u;   /* 与 pylitearm random_seed 数值一致(不同生成器) */
        for (k = 0; k < IK_RESTARTS; k++) {
            float amp = IK_AMP_STEP * (float)(k + 1);
            float q0[KIN_N];
            int i;
            for (i = 0; i < KIN_N; i++) q0[i] = rng_uni(&r, seed[i] - amp, seed[i] + amp);
            clamp_q(q0, q_min, q_max);
            if (dls_iterate(pos_d, R_d, q0, q_min, q_max, &ep, &er, q_out, &used)) {
                return true;
            }
            budget -= used;                 /* [P1-6] 累计扣减 */
            if (budget <= 0) break;         /* 预算耗尽: 按当前最优失败返回 */
        }
    }
    return false;
}
