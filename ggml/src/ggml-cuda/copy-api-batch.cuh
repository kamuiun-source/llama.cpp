#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12090 && CUDART_VERSION < 13000
#define GGML_CUDA_MTP3_D2H_BATCH

#include "ggml-backend-copy-api-batch.h"
#include <cudaTypedefs.h>

static bool ggml_backend_cuda_mtp3_d2h_batch(ggml_backend_t backend, const uintptr_t * src, const uintptr_t * dst,
        const size_t * sizes, size_t count) {
    const auto * ctx = static_cast<const ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    static const auto batch = [] {
        void * proc = nullptr;
        cudaDriverEntryPointQueryResult status{};
        const cudaError_t err = cudaGetDriverEntryPointByVersion("cuMemcpyBatchAsync", &proc, 12080,
                cudaEnablePerThreadDefaultStream, &status);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            return PFN_cuMemcpyBatchAsync_v12080_ptsz{};
        }
        return status == cudaDriverEntryPointSuccess ? reinterpret_cast<PFN_cuMemcpyBatchAsync_v12080_ptsz>(proc) : nullptr;
    }();
    if (!batch) {
        return false;
    }

    GGML_ASSERT(count >= 2 && count <= ggml_copy_api_batch_max);
    CUdeviceptr sources[ggml_copy_api_batch_max];
    CUdeviceptr destinations[ggml_copy_api_batch_max];
    size_t lengths[ggml_copy_api_batch_max];
    for (size_t i = 0; i < count; ++i) {
        sources[i] = static_cast<CUdeviceptr>(src[i]);
        destinations[i] = static_cast<CUdeviceptr>(dst[i]);
        lengths[i] = sizes[i];
    }
    CUmemcpyAttributes_v1 attrs{};
    attrs.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    size_t attrs_index = 0;
    size_t fail_index = SIZE_MAX;
    const CUresult result = batch(destinations, sources, lengths, count, &attrs, &attrs_index, 1, &fail_index,
            reinterpret_cast<CUstream>(cudaStreamPerThread));
    // A submission error does not guarantee that all copies were rejected.
    if (result != CUDA_SUCCESS) {
        GGML_ABORT("CUDA MTP3 D2H batch failed: error=%d fail_index=%zu count=%zu", int(result), fail_index, count);
    }
    CUDA_CHECK(ggml_cuda_stream_synchronize(cudaStreamPerThread));
    return true;
}
#endif
