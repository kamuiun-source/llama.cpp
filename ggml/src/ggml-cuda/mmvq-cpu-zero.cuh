#pragma once

#include <atomic>

namespace ggml_hip_mtp3_skip_cpu_zero {

inline std::atomic<bool> active{false};

inline void set_mode(bool enabled) {
    active.store(enabled, std::memory_order_relaxed);
}

inline bool enabled() {
    return active.load(std::memory_order_relaxed);
}

template<int rows, int tile_rows, int subgroup, typename Group>
static __device__ __forceinline__ void store_group(float * out, const Group & group) {
    if (threadIdx.x % subgroup == 0) {
        const int row = blockIdx.x * tile_rows + threadIdx.x / subgroup;
        for (int p = 0; p < group.count; ++p) {
            out[group.routes[p] * rows + row] = 0.0f;
        }
    }
}

static __device__ __forceinline__ void store_wave(float * route_dst, uint32_t row, uint32_t nrows) {
    if (row < nrows) {
        route_dst[row] = 0.0f;
    }
}

} // namespace ggml_hip_mtp3_skip_cpu_zero
