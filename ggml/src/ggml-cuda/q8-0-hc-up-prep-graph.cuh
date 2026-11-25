#pragma once

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "graph.cuh"
#include "hc-fused.cuh"
#include "q8-0-hc-up-mode.cuh"
#include "q8-0-hc-up-prep.cuh"
#include "scale.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static inline bool hc_up_prep_on_device(const ggml_tensor * tensor, ggml_backend_buffer_type_t buft) {
    const auto buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->buft != buft || !tensor->data) {
        return false;
    }
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const auto addr = reinterpret_cast<uintptr_t>(tensor->data);
    return addr >= base && addr - base <= buffer->size && ggml_nbytes(tensor) <= buffer->size - (addr - base);
}

static inline int ggml_cuda_try_hc_up_prep(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int i) {
    if (!ggml_cuda_q8_0_hc_up::configured() || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_ADA_LOVELACE ||
            !ctx.stream_context().concurrent_events.empty() || GGML_PAD(hc_up_prep_width, MATRIX_ROW_PADDING) != hc_up_prep_padded) {
        return 0;
    }
    auto * first = graph->nodes[i];
    const bool with_scale = first->op == GGML_OP_SCALE;
    const int j = i + with_scale;
    if ((!with_scale && first->op != GGML_OP_UNARY) || j + 1 >= graph->n_nodes) {
        return 0;
    }
    auto * silu = graph->nodes[j];
    auto * up = graph->nodes[j + 1];
    if (up->op != GGML_OP_MUL_MAT || up->src[1] != silu || !up->src[0] ||
            silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            !hc_up_prep_shape(silu) || silu->view_src || !ggml_is_contiguous(silu)) {
        return 0;
    }
    const auto * weight = up->src[0];
    const auto * input = with_scale ? first->src[0] : silu->src[0];
    if (weight->type != GGML_TYPE_Q8_0 || weight->ne[0] != hc_up_prep_width || weight->ne[1] != 10240 ||
            !ggml_is_matrix(weight) || !ggml_is_contiguous(weight) || weight->view_src ||
            ggml_cuda_hc_projection_for_weight(weight->name) != GGML_CUDA_HC_UP ||
            up->type != GGML_TYPE_F32 || up->ne[0] != 10240 || up->ne[1] != hc_up_prep_columns ||
            !ggml_is_matrix(up) || !ggml_is_contiguous(up) || up->view_src || !hc_up_prep_shape(input)) {
        return 0;
    }
    const auto buft = ggml_backend_cuda_buffer_type(ctx.device);
    if (!hc_up_prep_on_device(weight, buft) || !hc_up_prep_on_device(input, buft) || !hc_up_prep_on_device(up, buft)) {
        return 0;
    }

    if (with_scale) {
        if (silu->src[0] != first || first->view_src || !hc_up_prep_shape(first) ||
                !ggml_is_contiguous(first) || !ggml_is_contiguous(input) ||
                !ggml_can_fuse_subgraph(graph, i, {GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL_MAT}, {j + 1})) {
            return 0;
        }
    } else {
        // The merged down/inject path keeps SCALE live and reads only the first 320 columns of each row.
        const auto * scale = input->view_src ? input->view_src : input;
        const int64_t row = input->view_src ? 324 : hc_up_prep_width;
        if (scale->op != GGML_OP_SCALE || scale->type != GGML_TYPE_F32 || scale->ne[0] != row ||
                scale->ne[1] != hc_up_prep_columns || !ggml_is_matrix(scale) || !ggml_is_contiguous(scale) ||
                input->view_offs != 0 || input->nb[0] != sizeof(float) || input->nb[1] != row * sizeof(float) ||
                !ggml_can_fuse_subgraph(graph, i, {GGML_OP_UNARY, GGML_OP_MUL_MAT}, {j + 1})) {
            return 0;
        }
    }

    ggml_cuda_q8_0_hc_up::prep_candidate_seen = true;
    if (ggml_cuda_q8_0_hc_up::get_mode() != 3) {
        return 0;
    }
    ggml_cuda_q8_0_hc_up::dispatched = true;
    ggml_cuda_mul_mat_vec_q_hc_up_prep(ctx, up, {input, with_scale ? first : nullptr});
    static std::atomic<unsigned> logged{0};
    const unsigned bit = with_scale ? 1 : 2;
    if (!(logged.load(std::memory_order_relaxed) & bit) &&
            !(logged.fetch_or(bit, std::memory_order_relaxed) & bit)) {
        GGML_LOG_INFO("CUDA Ada Q8_0 HC up dispatch: cols=3 mode=prep scale=%s\n", with_scale ? "fused" : "kept");
    }
    return j + 1 - i;
}

static inline bool ggml_cuda_test_hc_up_prep(ggml_backend_t backend, const ggml_tensor * input,
        ggml_tensor * scale, ggml_tensor * silu, void * baseline, void * fused, size_t bytes) {
    if (!backend || !ggml_backend_is_cuda(backend) || !baseline || !fused || bytes != hc_up_prep_bytes ||
            !hc_up_prep_shape(input) || !hc_up_prep_shape(silu) || !ggml_is_contiguous(silu) ||
            silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
            silu->src[0] != (scale ? scale : input) ||
            !input->data || !silu->data || input->nb[0] != sizeof(float) || input->nb[1] < hc_up_prep_width * sizeof(float) ||
            input->nb[1] % sizeof(float) != 0 ||
            (scale && (!hc_up_prep_shape(scale) || scale->op != GGML_OP_SCALE || scale->src[0] != input || !scale->data))) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    const auto buft = ggml_backend_cuda_buffer_type(ctx->device);
    if (!hc_up_prep_on_device(input, buft) || !hc_up_prep_on_device(silu, buft) ||
            (scale && (!ggml_is_contiguous(scale) || !hc_up_prep_on_device(scale, buft)))) {
        return false;
    }

    ggml_cuda_set_device(ctx->device);
    const auto stream = ctx->stream();
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    ggml_cuda_pool_alloc<char> ref(ctx->pool(), bytes);
    ggml_cuda_pool_alloc<char> trial(ctx->pool(), bytes);
    hc_up_launch_q8_1({input, scale}, trial.get(), stream);
    if (scale) {
        ggml_cuda_op_scale(*ctx, scale);
    }
    ggml_cuda_op_silu(*ctx, silu);
    quantize_row_q8_1_cuda(static_cast<const float *>(silu->data), ref.get(), hc_up_prep_width,
            silu->nb[1] / sizeof(float), silu->nb[2] / sizeof(float), silu->nb[3] / sizeof(float),
            hc_up_prep_padded, hc_up_prep_columns, 1, 1, stream);
    CUDA_CHECK(cudaMemcpyAsync(baseline, ref.get(), bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(fused, trial.get(), bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}
#endif
