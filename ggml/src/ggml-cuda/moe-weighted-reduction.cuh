#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_op_ranked_rtx_weighted_merge(ggml_backend_cuda_context & ctx,
        const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst);
#endif
