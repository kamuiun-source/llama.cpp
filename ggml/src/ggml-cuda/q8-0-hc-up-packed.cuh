#pragma once

#include "vecdotq.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static_assert(QI8_0 * sizeof(int) == QK8_0, "one packed word holds four Q8_0 lanes");

static constexpr int hc_up_packed_blocks = 10240 * (320 / QK8_0);
static constexpr size_t hc_up_packed_quant_bytes = size_t(hc_up_packed_blocks) * QK8_0;
static constexpr size_t hc_up_packed_bytes = hc_up_packed_quant_bytes + size_t(hc_up_packed_blocks) * sizeof(half);

struct hc_up_packed_weights {
    const int * quants;
    const half * scales;
};

hc_up_packed_weights ggml_cuda_prepare_hc_up_packed(ggml_backend_cuda_context & ctx, const ggml_tensor * weight);

static __global__ void hc_up_pack_q8_0(const block_q8_0 * src, int * quants, half * scales) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hc_up_packed_blocks) {
#pragma unroll
        for (int j = 0; j < QI8_0; ++j) {
            quants[i * QI8_0 + j] = get_int_b2(src[i].qs, j);
        }
        scales[i] = src[i].d;
    }
}

static __device__ __forceinline__ float hc_up_dot_packed(hc_up_packed_weights weights, int block, int iqs, const block_q8_1 & x) {
    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
    for (int j = 0; j < VDR_Q8_0_Q8_1_MMVQ; ++j) {
        v[j] = weights.quants[block * QI8_0 + iqs + j];
        u[j] = get_int_b4(x.qs, iqs + j);
    }
    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, weights.scales[block], __low2half(x.ds));
}
#endif
