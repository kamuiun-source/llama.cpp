// Included from getrows.cu.
#include "mtp3-compact-slot-map.cuh"

#ifdef GGML_USE_HIP
#include <atomic>

static std::atomic<bool> mtp3_compact_slot_map_enabled{false};

bool ggml_hip_mtp3_compact_slot_map_enabled() {
    return mtp3_compact_slot_map_enabled.load(std::memory_order_relaxed);
}

void ggml_hip_set_mtp3_compact_slot_map(bool enabled) {
    mtp3_compact_slot_map_enabled.store(enabled, std::memory_order_relaxed);
}

static __global__ void copy_mtp3_compact_slot_ids(const int32_t * src, int32_t * dst) {
    if (threadIdx.x < ggml_hip_mtp3_compact_slot_map::routes) {
        dst[threadIdx.x] = src[threadIdx.x];
    }
}

void ggml_hip_copy_mtp3_compact_slot_map(ggml_backend_cuda_context & ctx, ggml_tensor * lookup) {
    GGML_ASSERT(lookup->type == GGML_TYPE_I32 && ggml_is_contiguous(lookup) &&
            ggml_nelements(lookup) == ggml_hip_mtp3_compact_slot_map::routes);
    copy_mtp3_compact_slot_ids<<<1, 32, 0, ctx.stream()>>>(
            static_cast<const int32_t *>(lookup->src[0]->data), static_cast<int32_t *>(lookup->data));
    CUDA_CHECK(cudaGetLastError());
}
#endif
