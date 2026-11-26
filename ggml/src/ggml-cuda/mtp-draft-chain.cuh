#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "getrows.cuh"
static __global__ void ggml_cuda_mtp_draft_chain_kernel(const float * logits, const int32_t * candidates,
        int32_t * result, int32_t * valid, const float * hidden, float * next_hidden,
        int n_candidates, int n_hidden, int n_vocab) {
    if (threadIdx.x == 0) {
        float best = -INFINITY;
        int32_t id = 0;
        bool unique = false;
        for (int i = 0; i < n_candidates; ++i) {
            const float value = logits[i];
            if (value > best) {
                best = value;
                id = candidates[i];
                unique = true;
            } else if (value == best) {
                unique = false;
            }
        }
        const bool usable = unique && isfinite(best) && id >= 0 && id < n_vocab;
        *result = usable ? id : 0;
        *valid = usable;
    }
    for (int i = threadIdx.x; i < n_hidden; i += blockDim.x) {
        next_hidden[i] = hidden[i];
    }
}

static bool ggml_backend_cuda_mtp_draft_chain(ggml_backend_t backend, const ggml_tensor * logits,
        const ggml_tensor * candidates, const ggml_tensor * hidden, ggml_tensor * ids, ggml_tensor * next_hidden,
        int step, int n_vocab, const ggml_tensor * table, void * mapped_table, ggml_tensor * embedding) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto native = [buft](const ggml_tensor * t, ggml_type type, int64_t count) {
        return t && t->buffer && t->data && ggml_backend_buffer_get_type(t->buffer) == buft &&
                t->type == type && ggml_nelements(t) == count && ggml_is_contiguous(t);
    };
    if (step < 0 || step > 1 || (hidden && !next_hidden)) {
        return false;
    }
    const int n_hidden = hidden ? ggml_nelements(next_hidden) : 0;
    if (!native(logits, GGML_TYPE_F32, 10) || !native(candidates, GGML_TYPE_I32, 10) ||
            !native(ids, GGML_TYPE_I32, 4) || (hidden &&
                (!native(hidden, GGML_TYPE_F32, n_hidden) || !native(next_hidden, GGML_TYPE_F32, n_hidden))) ||
            ctx->curr_stream_no != 0) {
        return false;
    }
    if (step == 0 && (!table || !mapped_table || table->type != GGML_TYPE_Q8_0 || !ggml_is_matrix(table) ||
            !ggml_is_contiguous(table) || table->ne[1] != n_vocab ||
            !native(embedding, GGML_TYPE_F32, table->ne[0]))) {
        return false;
    }
    ggml_cuda_set_device(ctx->device);
    auto * dst = static_cast<int32_t *>(ids->data);
    ggml_cuda_mtp_draft_chain_kernel<<<1, 256, 0, ctx->stream()>>>(
        static_cast<const float *>(logits->data), static_cast<const int32_t *>(candidates->data),
        dst + step, dst + step + 2, hidden ? static_cast<const float *>(hidden->data) : nullptr,
        hidden ? static_cast<float *>(next_hidden->data) : nullptr, 10, n_hidden, n_vocab);
    CUDA_CHECK(cudaGetLastError());
    if (step == 0) {
        // CUDA gets only the chosen row. The full table remains in its original buffer.
        auto mapped = *table;
        mapped.data = mapped_table;
        auto index = *ids;
        index.ne[0] = 1;
        auto output = *embedding;
        output.src[0] = &mapped;
        output.src[1] = &index;
        ggml_cuda_op_get_rows(*ctx, &output);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}
#endif
