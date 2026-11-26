// Included from argsort.cu so the fallback uses the same comparison network.
#include "next-router-top10.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <atomic>

static std::atomic<bool> next_router_top10_enabled{false};

bool ggml_cuda_next_router_top10_enabled() {
    return next_router_top10_enabled.load(std::memory_order_relaxed);
}

void ggml_cuda_set_next_router_top10(bool enabled) {
    next_router_top10_enabled.store(enabled, std::memory_order_relaxed);
}

namespace {
constexpr int next_router_experts = 512;
constexpr int next_router_top_k = 10;
constexpr int next_router_threads = 128;
constexpr int next_router_columns = next_router_experts / next_router_threads;
constexpr int next_router_warps = next_router_threads / WARP_SIZE;
constexpr int tied = next_router_experts;

struct next_router_candidate {
    float value;
    int index;

    __device__ __forceinline__ void merge(float other_value, int other_index) {
        if (other_value > value) {
            value = other_value;
            index = other_index;
        } else if (other_value == value) {
            index |= tied;
        }
    }

    __device__ __forceinline__ void reduce_warp() {
#pragma unroll
        for (int distance = WARP_SIZE / 2; distance > 0; distance /= 2) {
            const float other_value = __shfl_down_sync(0xffffffff, value, distance);
            const int other_index = __shfl_down_sync(0xffffffff, index, distance);
            merge(other_value, other_index);
        }
    }
};
} // namespace

static __global__ void next_router_top10_kernel(const float * scores, int32_t * ids) {
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const auto * row = scores + blockIdx.x * next_router_experts;
    auto * dst = ids + blockIdx.x * next_router_top_k;
    __shared__ next_router_candidate maxima[next_router_warps];
    __shared__ int winner;
    __shared__ int full_order[next_router_experts];
    float values[next_router_columns];
    bool invalid = false;
#pragma unroll
    for (int k = 0; k < next_router_columns; ++k) {
        values[k] = row[k * next_router_threads + tid];
        invalid |= !isfinite(values[k]);
    }

    if (!__syncthreads_or(invalid)) {
        for (int rank = 0; rank < next_router_top_k; ++rank) {
            next_router_candidate best{-INFINITY, 0};
#pragma unroll
            for (int k = 0; k < next_router_columns; ++k) {
                best.merge(values[k], k * next_router_threads + tid);
            }
            best.reduce_warp();
            if (lane == 0) {
                maxima[warp] = best;
            }
            __syncthreads();

            if (warp == 0) {
                best = lane < next_router_warps ? maxima[lane] : next_router_candidate{-INFINITY, 0};
                best.reduce_warp();
                if (lane == 0) {
                    winner = best.index;
                    dst[rank] = best.index & (next_router_experts - 1);
                }
            }
            __syncthreads();

            const int selected = winner;
            if (selected & tied) {
                break;
            }
            if (rank == next_router_top_k - 1) {
                return;
            }
#pragma unroll
            for (int k = 0; k < next_router_columns; ++k) {
                if (selected == k * next_router_threads + tid) {
                    values[k] = -INFINITY;
                }
            }
        }
    }

    // A tie at any selected score can change rank or membership in the original bitonic order.
    for (int col = tid; col < next_router_experts; col += next_router_threads) {
        full_order[col] = col;
    }
    __syncthreads();
    for (int k = 2; k <= next_router_experts; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = tid; col < next_router_experts; col += next_router_threads) {
                bitonic_step<GGML_SORT_ORDER_DESC>(row, full_order, next_router_experts, col, k, j);
            }
            __syncthreads();
        }
    }
    if (tid < next_router_top_k) {
        dst[tid] = full_order[tid];
    }
}

void ggml_cuda_launch_next_router_top10(const float * scores, int32_t * ids, cudaStream_t stream) {
    next_router_top10_kernel<<<3, next_router_threads, 0, stream>>>(scores, ids);
    CUDA_CHECK(cudaGetLastError());
}
#endif
