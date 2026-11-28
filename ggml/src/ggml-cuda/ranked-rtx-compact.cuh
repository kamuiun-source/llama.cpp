#pragma once

#include "quantize.cuh"
#include "../ggml-ranked-rtx.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace ggml_ranked_rtx {
inline thread_local bool cuda_compact = false;

static __device__ __forceinline__ int compact_route(const int32_t * ids, int slot) {
    const int lane = threadIdx.x & 31;
    unsigned mask = __ballot_sync(0xffffffff, lane < routes && ids[lane] >= 0);
    for (int i = 0; i < slot; ++i) {
        mask &= mask - 1;
    }
    return __ffs(mask) - 1;
}

// The down projection only writes ranked routes. Clear the dense output while preparing its active Q8_1 rows.
static __global__ void ranked_rtx_compact_quantize_down(
        const float * source, const int32_t * ids, block_q8_1 * quantized, float * output,
        int64_t row_stride, int64_t token_stride, int padded_width) {
    ggml_cuda_pdl_sync();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int linear_thread = blockIdx.y * (gridDim.x * blockDim.x) + index;
    const int threads = gridDim.y * gridDim.x * blockDim.x;
    for (int col = linear_thread; col < routes * 2560; col += threads) {
        output[col] = 0.0f;
    }

    const int route = compact_route(ids, blockIdx.y);
    if (route >= 0) {
        const int token = route / ranks;
        const int rank = route % ranks;
        const float input = index < 640 ? source[token * token_stride + rank * row_stride + index] : 0.0f;
        const int element = route * padded_width + index;
        ggml_cuda_quantize_q8_1_block(input, quantized + element / QK8_1, element % QK8_1);
    }
    ggml_cuda_pdl_lc();
}
}
#endif
