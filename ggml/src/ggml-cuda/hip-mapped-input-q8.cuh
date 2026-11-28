#pragma once

#include "../ggml-backend-hip-mapped-input-q8.h"

constexpr int hip_mapped_input_q8_width = 2560;
constexpr int hip_mapped_input_q8_positions = 3;
constexpr size_t hip_mapped_input_q8_bytes = hip_mapped_input_q8_width * hip_mapped_input_q8_positions / QK8_1 * sizeof(block_q8_1);
static_assert(GGML_PAD(hip_mapped_input_q8_width, MATRIX_ROW_PADDING) == hip_mapped_input_q8_width, "fixed MTP3 width");

static void ggml_backend_hip_quantize_mapped_input_q8(const float * src, void * dst, cudaStream_t stream) {
    constexpr int width = hip_mapped_input_q8_width;
    constexpr int positions = hip_mapped_input_q8_positions;
    quantize_row_q8_1_cuda(src, dst, width, width, width, width * positions, width, 1, positions, 1, stream);
}

static bool ggml_backend_hip_mapped_input_q8(ggml_backend_t backend, ggml_tensor * dst, const void * mapped) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    const bool token_layout = (dst->ne[1] == 1 && dst->ne[2] == 3) || (dst->ne[1] == 3 && dst->ne[2] == 1);
    if (ggml_cuda_info().devices[ctx->device].cc != GGML_CUDA_CC_VEGA20 ||
            get_mmvq_mmid_max_batch(GGML_TYPE_IQ2_S, GGML_CUDA_CC_VEGA20) < hip_mapped_input_q8_positions ||
            get_mmvq_mmid_max_batch(GGML_TYPE_IQ3_XXS, GGML_CUDA_CC_VEGA20) < hip_mapped_input_q8_positions ||
            !ctx->stream_context().concurrent_events.empty() || ctx->curr_stream_no != 0 ||
            !ggml_is_contiguous(dst) || dst->type != GGML_TYPE_F32 || dst->ne[0] != hip_mapped_input_q8_width ||
            !token_layout || dst->ne[3] != 1 || ggml_nbytes(dst) != hip_mapped_input_q8_width * hip_mapped_input_q8_positions * sizeof(float) ||
            reinterpret_cast<uintptr_t>(dst->data) % sizeof(float4) ||
            reinterpret_cast<uintptr_t>(mapped) % sizeof(float4)) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    const auto stream = ctx->stream();
    const auto * src = static_cast<const float *>(mapped);
    static const bool check = ggml_cuda_env_flag("GGML_HIP_MAPPED_INPUT_Q8_CHECK");
    static thread_local int checked = 0;
    if (check && checked < 3) {
        // Use the existing blit and then the same quantizer on its VRAM output.
        if (!ggml_backend_hip_mapped_input_blit(backend, dst, mapped)) {
            return false;
        }
        ggml_cuda_pool_alloc<char> baseline(ctx->pool(), hip_mapped_input_q8_bytes);
        ggml_backend_hip_quantize_mapped_input_q8(static_cast<const float *>(dst->data), baseline.get(), stream);
        ggml_backend_hip_quantize_mapped_input_q8(src, dst->data, stream);
        std::array<uint8_t, hip_mapped_input_q8_bytes> ref, got;
        CUDA_CHECK(cudaMemcpyAsync(ref.data(), baseline.get(), ref.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(got.data(), dst->data, got.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GGML_ASSERT(std::memcmp(ref.data(), got.data(), ref.size()) == 0 && "mapped-input Q8_1 bitwise mismatch");
        GGML_LOG_INFO("HIP mapped input Q8_1 bitwise: PASS sample=%d bytes=%zu\n", ++checked, ref.size());
    } else {
        ggml_backend_hip_quantize_mapped_input_q8(src, dst->data, stream);
    }
    return true;
}

static void ggml_backend_hip_mapped_input_q8_scope(ggml_backend_t backend, const ggml_tensor * input) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ctx->mapped_input_q8 = input;
}
