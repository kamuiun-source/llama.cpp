#pragma once

#include "vecdotq.cuh"

static __device__ __forceinline__ void q6_k_mmvq_reuse_three(
        const block_q6_K & weight, const block_q8_1 * activation, uint32_t stride_col, int iqs, float (&dots)[3]) {
    const int group = iqs / (QI6_K/2);
    const int within = iqs % (QI6_K/2);
    const int half = within / (QI6_K/4);
    const int bq8_offset = 2 * QR6_K * group + half;
    const int scale_offset = (QI6_K/4) * group + within / (QI6_K/8);
    const int high_offset = (QI6_K/4) * group + iqs % (QI6_K/4);

    uint32_t low[VDR_Q6_K_Q8_1_MMVQ];
    uint32_t high[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
    for (int v = 0; v < VDR_Q6_K_Q8_1_MMVQ; ++v) {
        low[v] = get_int_b2(weight.ql, iqs + v);
        high[v] = uint32_t(get_int_b2(weight.qh, high_offset + v)) >> (2 * half);
    }

    float sums[3] = {};
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int scale = weight.scales[scale_offset + 4*i];
        int codes[VDR_Q6_K_Q8_1_MMVQ];
#pragma unroll
        for (int v = 0; v < VDR_Q6_K_Q8_1_MMVQ; ++v) {
            const unsigned vl = (low[v] >> (4*i)) & 0x0F0F0F0F;
            const unsigned vh = ((high[v] >> (4*i)) << 4) & 0x30303030;
            codes[v] = vl | vh;
        }

#pragma unroll
        for (int col = 0; col < 3; ++col) {
            const auto & x = activation[col * stride_col + bq8_offset + 2*i];
            int sum = 0;
#pragma unroll
            for (int v = 0; v < VDR_Q6_K_Q8_1_MMVQ; ++v) {
                const int q8 = get_int_b4(x.qs, iqs % QI8_1 + v);
                sum = ggml_cuda_dp4a(codes[v], q8, sum);
                sum = ggml_cuda_dp4a(0xE0E0E0E0, q8, sum);
            }
            sums[col] += __low2float(x.ds) * (sum * scale);
        }
    }

    const float d = weight.d;
#pragma unroll
    for (int col = 0; col < 3; ++col) {
        dots[col] = d * sums[col];
    }
}
