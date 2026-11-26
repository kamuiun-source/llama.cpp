#pragma once

#include "../../ggml/src/ggml-cuda/unary.cuh"
#include "../../ggml/src/ggml-cuda/vecdotq.cuh"
#include "../../ggml/src/ggml-hip-iq3-route-plan.h"

#include <vector>

namespace iq3_parallel_weights {
namespace plan = ggml_hip_iq3_route_plan;

constexpr int tokens = plan::tokens;
constexpr int ranks = plan::ranks;
constexpr int rank_bits = plan::rank_bits;
constexpr int shared_shift = tokens * rank_bits;
constexpr uint32_t rank_mask = (1 << rank_bits) - 1;
constexpr int nrows = 640;
constexpr int ncols = 2560;
constexpr int rows_per_tile = 4;
constexpr int weight_blocks = ncols / QK_K;
constexpr int input_blocks = ncols / QK8_1;

// Low nibbles store the original token ranks plus one. The high bits mark waves sharing weights.
inline std::vector<uint32_t> group_routes(const std::array<int32_t, plan::routes> & routes) {
    std::vector<uint32_t> groups;
    groups.reserve(plan::routes);

    // With three waves there is space for at most one multi-wave Expert per group.
    for (int32_t route : routes) {
        if (route <= 0) {
            continue;
        }
        uint32_t shared = 0;
        for (int t = 0; t < tokens; ++t) {
            if ((route >> (t * rank_bits)) & rank_mask) {
                shared |= 1u << t;
            }
        }
        groups.push_back(uint32_t(route) | (shared << shared_shift));
    }

    for (int i = 0; i < plan::routes; ++i) {
        if (routes[i] != 0) {
            continue;
        }
        const int shift = (i / ranks) * rank_bits;
        const uint32_t rank = uint32_t(i % ranks + 1) << shift;
        auto group = groups.begin();
        for (; group != groups.end(); ++group) {
            if (!(*group & (rank_mask << shift))) {
                break;
            }
        }
        if (group == groups.end()) {
            groups.push_back(rank);
        } else {
            *group |= rank;
        }
    }
    return groups;
}

using weight_tile = block_iq3_xxs[rows_per_tile * weight_blocks];

static __device__ __forceinline__ void stage_weights(weight_tile & dst, const block_iq3_xxs * src, int lane, int start = 0) {
    static_assert(sizeof(weight_tile) % sizeof(uint32_t) == 0);
    auto * out = static_cast<unsigned char *>(__builtin_assume_aligned(&dst, alignof(uint32_t)));
    const auto * in = static_cast<const unsigned char *>(__builtin_assume_aligned(src, alignof(uint32_t)));
    // A complete tile is word-aligned even though individual IQ3_XXS blocks can start on a halfword.
    for (int i = start + lane * sizeof(uint32_t); i < int(sizeof(weight_tile)); i += 64 * sizeof(uint32_t)) {
        uint32_t word;
        __builtin_memcpy(&word, in + i, sizeof(word));
        __builtin_memcpy(out + i, &word, sizeof(word));
    }
}

static __device__ __forceinline__ void stage_weights_batched(weight_tile & dst, const block_iq3_xxs * src, int lane) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    constexpr int wave_bytes = 64 * sizeof(uint32_t);
    constexpr int batch_bytes = 4 * wave_bytes;
    constexpr int full_batches_bytes = sizeof(weight_tile) / batch_bytes * batch_bytes;
    auto * out = static_cast<unsigned char *>(__builtin_assume_aligned(&dst, alignof(uint32_t)));
    const auto * in = static_cast<const unsigned char *>(__builtin_assume_aligned(src, alignof(uint32_t)));
    static_assert(wave_bytes == 256);

#pragma unroll 1
    for (int i = lane * sizeof(uint32_t); i < full_batches_bytes; i += batch_bytes) {
        uint32_t words[4];
        // Issue four independent loads before waiting for the values used by the LDS stores.
        asm volatile(
            "global_load_dword %0, %4, off\n\t"
            "global_load_dword %1, %4, off offset:256\n\t"
            "global_load_dword %2, %4, off offset:512\n\t"
            "global_load_dword %3, %4, off offset:768\n\t"
            "s_waitcnt vmcnt(0)"
            : "=&v"(words[0]), "=&v"(words[1]), "=&v"(words[2]), "=&v"(words[3])
            : "v"(in + i)
            : "memory");
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __builtin_memcpy(out + i + j * wave_bytes, &words[j], sizeof(uint32_t));
        }
    }
    stage_weights(dst, src, lane, full_batches_bytes);
#else
    stage_weights(dst, src, lane);
#endif
}

static __device__ __forceinline__ void stage_weights_vector(weight_tile & dst, const block_iq3_xxs * src, int lane) {
    constexpr int bytes = sizeof(uint4);
    constexpr int units = sizeof(weight_tile) / bytes;
    static_assert(bytes == 16 && sizeof(weight_tile) % bytes == 0 && nrows % rows_per_tile == 0);
    auto * out = static_cast<unsigned char *>(__builtin_assume_aligned(&dst, bytes));
    const auto * in = static_cast<const unsigned char *>(__builtin_assume_aligned(src, bytes));

    // The final stride has only 53 valid lanes. Each valid lane copies one complete vector.
    for (int unit = lane; unit < units; unit += 64) {
        __builtin_memcpy(out + unit * bytes, in + unit * bytes, bytes);
    }
}

static __device__ __forceinline__ void project(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * activation,
        const uint32_t * grid, float * out) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int lanes_per_block = QI3_XXS / VDR_IQ3_XXS_Q8_1_MMVQ;
    constexpr int blocks_per_iter = warp_size / lanes_per_block;
    const int kbx = threadIdx.x / lanes_per_block;
    const int kbx_next = kbx + blocks_per_iter;
    const int kqs = VDR_IQ3_XXS_Q8_1_MMVQ * (threadIdx.x % lanes_per_block);
    const auto * y = activation + kqs / 2;
    const auto & current = y[kbx * (QK_K / QK8_1)];

#pragma unroll 1
    for (int group = 0; group < rows_per_tile / 2; ++group) {
        float tmp[2] = {};
        float tmp_gate[2] = {};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int row = (2 * group + i) * weight_blocks;
            const auto up_current = load_iq_mmvq_weights(up[row + kbx], kqs);
            const auto gate_current = load_iq_mmvq_weights(gate[row + kbx], kqs);
            iq3_xxs_mmvq_weights up_next, gate_next;
            if (kbx_next < weight_blocks) {
                up_next = load_iq_mmvq_weights(up[row + kbx_next], kqs);
                gate_next = load_iq_mmvq_weights(gate[row + kbx_next], kqs);
            }
            tmp[i] += vec_dot_iq_q8_1_loaded<true>(up_current, current, grid);
            tmp_gate[i] += vec_dot_iq_q8_1_loaded<true>(gate_current, current, grid);
            if (kbx_next < weight_blocks) {
                const auto & next = y[kbx_next * (QK_K / QK8_1)];
                tmp[i] += vec_dot_iq_q8_1_loaded<true>(up_next, next, grid);
                tmp_gate[i] += vec_dot_iq_q8_1_loaded<true>(gate_next, next, grid);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
            tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        }
        if (threadIdx.x < 2) {
            out[2 * group + threadIdx.x] = tmp[threadIdx.x] * ggml_cuda_op_silu_single(tmp_gate[threadIdx.x]);
        }
    }
}

enum weight_mode : uint32_t {
    weights_direct = 0,
    weights_stage = 1,
    weights_read = 2,
    weights_share = weights_stage | weights_read,
    weights_batch = 4,
    weights_stage_batched = weights_stage | weights_batch,
    weights_share_batched = weights_share | weights_batch,
    weights_vector = 8,
    weights_stage_vector = weights_stage | weights_vector,
    weights_share_vector = weights_share | weights_vector,
};

template<bool ablation = false, bool batch_ablation = false, bool vector_ablation = false>
__launch_bounds__(64 * tokens, 1)
static __global__ void gate_up(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * activation,
        const int32_t * ids, const uint32_t * groups, float * out, weight_mode mode = weights_share) {
    constexpr int grid_size = sizeof(iq3xxs_grid) / sizeof(iq3xxs_grid[0]);
    constexpr int activation_words = input_blocks * sizeof(block_q8_1) / sizeof(uint32_t);
    __shared__ uint32_t grid[grid_size];
    __shared__ uint32_t input[tokens][activation_words];
    alignas(vector_ablation ? sizeof(uint4) : alignof(uint32_t)) __shared__ weight_tile shared_up;
    alignas(vector_ablation ? sizeof(uint4) : alignof(uint32_t)) __shared__ weight_tile shared_gate;

    const int lane = threadIdx.x;
    const int token = threadIdx.y;
    const uint32_t group = groups[blockIdx.y];
    const uint32_t shared = group >> shared_shift;
    const int rank = int((group >> (token * rank_bits)) & rank_mask) - 1;
    const int row0 = blockIdx.x * rows_per_tile;
    const auto * token_input = reinterpret_cast<const uint32_t *>(activation + token * input_blocks);

    for (int i = token * 64 + lane; i < grid_size; i += tokens * 64) {
        grid[i] = iq3xxs_grid[i];
    }
    for (int i = lane; i < activation_words; i += 64) {
        input[token][i] = token_input[i];
    }

    // The first participating wave copies one packed tile for all participating waves.
    if ((!ablation || (mode & weights_stage)) && shared && token == __builtin_ctz(shared)) {
        const int expert = ids[token * ranks + rank];
        const int offset = (expert * nrows + row0) * weight_blocks;
        if constexpr (vector_ablation) {
            if (mode & weights_vector) {
                stage_weights_vector(shared_up, up + offset, lane);
                stage_weights_vector(shared_gate, gate + offset, lane);
            } else {
                stage_weights_batched(shared_up, up + offset, lane);
                stage_weights_batched(shared_gate, gate + offset, lane);
            }
        } else if (batch_ablation && (mode & weights_batch)) {
            stage_weights_batched(shared_up, up + offset, lane);
            stage_weights_batched(shared_gate, gate + offset, lane);
        } else {
            stage_weights(shared_up, up + offset, lane);
            stage_weights(shared_gate, gate + offset, lane);
        }
    }
    __syncthreads();

    if (rank < 0) {
        return;
    }

    auto * dst = out + (token * ranks + rank) * nrows + row0;
    const auto * token_activation = reinterpret_cast<const block_q8_1 *>(input[token]);
    if ((!ablation || (mode & weights_read)) && ((shared >> token) & 1)) {
        project(shared_up, shared_gate, token_activation, grid, dst);
    } else {
        const int expert = ids[token * ranks + rank];
        const int offset = (expert * nrows + row0) * weight_blocks;
        project(up + offset, gate + offset, token_activation, grid, dst);
    }
}

} // namespace iq3_parallel_weights
