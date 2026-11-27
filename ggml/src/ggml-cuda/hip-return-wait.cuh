#pragma once

#include "ggml-backend-impl.h"
#include "graph.cuh"

// Keep the overwrite on the sole CUDA stream that may still use the destination allocation.
static bool ggml_backend_cuda_return_stream_ordered(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (ctx->curr_stream_no != 0 || !ctx->stream_context().concurrent_events.empty()) {
        return false;
    }
    for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
        for (int stream = 0; stream < GGML_CUDA_MAX_STREAMS; ++stream) {
            if (ctx->streams[device][stream] && (device != ctx->device || stream != 0)) {
                return false;
            }
        }
    }
    return true;
}
