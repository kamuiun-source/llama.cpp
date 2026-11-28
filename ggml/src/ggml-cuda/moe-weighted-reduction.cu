#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static __global__ void ranked_rtx_weighted_merge_f32(
        const float * __restrict__ experts, const float * __restrict__ weights, float * dst) {
    constexpr int columns = 2560;
    constexpr int routes = 10;
    const int token = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    const int first = token * routes;
    // Match the generic CUDA reduction, then round the merge separately.
    float sum = __fmul_rn(experts[first * columns + col], weights[first]);

#pragma unroll
    for (int rank = 1; rank < routes; ++rank) {
        const int row = first + rank;
        sum = __fmaf_rn(experts[row * columns + col], weights[row], sum);
    }

    const int element = token * columns + col;
    dst[element] = __fadd_rn(dst[element], sum);
}

void ggml_cuda_op_ranked_rtx_weighted_merge(ggml_backend_cuda_context & ctx,
        const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst) {
    const auto * expert_data = static_cast<const float *>(experts->data);
    const auto * weight_data = static_cast<const float *>(weights->data);
    auto * output_data = static_cast<float *>(dst->data);
    constexpr int threads = 256;
    const dim3 blocks(3, 10);
    ranked_rtx_weighted_merge_f32<<<blocks, threads, 0, ctx.stream()>>>(expert_data, weight_data, output_data);
    CUDA_CHECK(cudaGetLastError());
}
#endif
