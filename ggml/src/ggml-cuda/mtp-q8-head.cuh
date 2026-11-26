#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "../ggml-mtp-q8-head.h"

static bool ggml_cuda_mtp_q8_head_pair_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * weight,
        const ggml_tensor * input, ggml_tensor * dst, const char * quantized, int stride_col_y, cudaStream_t stream) {
    if (!ggml_mtp_q8_head_has_pair_rows(dst) || weight->type != GGML_TYPE_Q8_0 ||
            weight->ne[0] != 2560 || weight->ne[1] != 248320 || input->ne[1] != 1 ||
            !ggml_is_matrix(weight) || !ggml_is_matrix(input) || !ggml_is_matrix(dst) ||
            !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const auto & device = ggml_cuda_info().devices[ctx.device];
    const auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    if (device.cc != GGML_CUDA_CC_ADA_LOVELACE || !weight->buffer || !input->buffer || !dst->buffer ||
            ggml_backend_buffer_get_type(weight->buffer) != buft || ggml_backend_buffer_get_type(input->buffer) != buft ||
            ggml_backend_buffer_get_type(dst->buffer) != buft) {
        return false;
    }

    constexpr int row_groups = 2;
    auto dims = calc_launch_params<GGML_TYPE_Q8_0>(1, weight->ne[1], 1, 1, device.warp_size, get_device_table_id(device.cc));
    dims.first.x /= row_groups;
    dims.second.z = row_groups;
    const ggml_cuda_kernel_launch_params params(dims.first, dims.second, 0, stream);
#ifdef GGML_CUDA_HAVE_MMVQ_TRACE
    const mmvq_launch_trace trace(params, 1, false, false, false, false, row_groups);
#endif
    const auto one = init_fastdiv_values(1);
    ggml_cuda_kernel_launch(mul_mat_vec_q<GGML_TYPE_Q8_0, 1, false, false, false, false, row_groups>, params,
        weight->data, quantized, static_cast<const int32_t *>(nullptr), ggml_cuda_mm_fusion_args_device{},
        static_cast<float *>(dst->data), uint32_t(weight->ne[0]), make_uint3(0, 0, 0),
        uint32_t(weight->nb[1] / ggml_type_size(weight->type)), uint32_t(stride_col_y), uint32_t(dst->ne[0]),
        one, uint32_t(0), uint32_t(0), uint32_t(0), one, uint32_t(0), uint32_t(0), uint32_t(0), uint32_t(0));
    static const bool logged = (GGML_LOG_INFO("CUDA MTP Q8 head applied: rows=248320 width=2560 groups=2\n"), true);
    GGML_UNUSED(logged);
    return true;
}
#endif
