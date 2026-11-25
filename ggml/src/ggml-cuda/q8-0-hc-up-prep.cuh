#pragma once

#include "quantize.cuh"
#include "unary.cuh"

#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static constexpr int hc_up_prep_width = 320;
static constexpr int hc_up_prep_columns = 3;
static constexpr int hc_up_prep_padded = 512;
static constexpr size_t hc_up_prep_bytes = hc_up_prep_columns * (hc_up_prep_padded / QK8_1) * sizeof(block_q8_1);

struct hc_up_prep_args {
    const ggml_tensor * input;
    const ggml_tensor * scale;
};

static inline bool hc_up_prep_shape(const ggml_tensor * tensor) {
    return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == hc_up_prep_width &&
            tensor->ne[1] == hc_up_prep_columns && ggml_is_matrix(tensor);
}

template <bool apply_scale>
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void hc_up_prepare_q8_1(
        const float * src, block_q8_1 * dst, size_t row_stride, float scale, float bias) {
    ggml_cuda_pdl_lc();
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y;
    ggml_cuda_pdl_sync();
    float value = 0.0f;
    if (x < hc_up_prep_width) {
        value = src[col * row_stride + x];
        if constexpr (apply_scale) {
            value = __fmaf_rn(scale, value, bias);
        }
        value = ggml_cuda_op_silu_single(value);
    }
    const int index = col * hc_up_prep_padded + x;
    ggml_cuda_quantize_q8_1_block(value, dst + index / QK8_1, index % QK8_1);
}

static inline void hc_up_launch_q8_1(const hc_up_prep_args & args, void * dst, cudaStream_t stream) {
    static_assert(hc_up_prep_padded % CUDA_QUANTIZE_BLOCK_SIZE == 0, "full blocks include all quantization padding");
    const ggml_cuda_kernel_launch_params params(
            dim3(hc_up_prep_padded / CUDA_QUANTIZE_BLOCK_SIZE, hc_up_prep_columns), CUDA_QUANTIZE_BLOCK_SIZE, 0, stream);
    const auto * src = static_cast<const float *>(args.input->data);
    const size_t stride = args.input->nb[1] / sizeof(float);
    auto * out = static_cast<block_q8_1 *>(dst);
    if (args.scale) {
        float scale, bias;
        std::memcpy(&scale, args.scale->op_params, sizeof(float));
        std::memcpy(&bias, args.scale->op_params + 1, sizeof(float));
        ggml_cuda_kernel_launch(hc_up_prepare_q8_1<true>, params, src, out, stride, scale, bias);
    } else {
        ggml_cuda_kernel_launch(hc_up_prepare_q8_1<false>, params, src, out, stride, 0.0f, 0.0f);
    }
}

void ggml_cuda_mul_mat_vec_q_hc_up_prep(ggml_backend_cuda_context & ctx, ggml_tensor * up, const hc_up_prep_args & prep);

#endif
