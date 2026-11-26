#pragma once

#include "../ggml-hip-iq3-strata-plan.h"

#include <atomic>

#ifdef GGML_USE_HIP
static std::atomic<bool> iq3_strata_enabled{false};

bool ggml_hip_gfx906_iq3_strata_enabled() {
    return iq3_strata_enabled.load(std::memory_order_relaxed);
}

void ggml_hip_gfx906_set_iq3_strata(bool enabled) {
    iq3_strata_enabled.store(enabled, std::memory_order_relaxed);
}
#endif

namespace iq3_strata_reuse {

constexpr int ncols = 2560;
constexpr int nrows = 640;
constexpr int tokens = 3;
constexpr int ranks = 10;
constexpr int weight_blocks = ncols / QK_K;
constexpr int activation_blocks = ncols / QK8_1;
constexpr int subgroup = 16;
constexpr int wave = 64;
constexpr int waves = 4;
constexpr int rows_per_tile = waves * wave / subgroup;
constexpr int lanes_per_block = QI3_XXS / VDR_IQ3_XXS_Q8_1_MMVQ;
static_assert(nrows % rows_per_tile == 0 && weight_blocks == 10 && lanes_per_block == 8);

struct expert_group {
    int count;
    int routes[tokens];
    int expert;
};

template<int count>
struct projections {
    float up[count];
    float gate[count];
};

// Decode one gate/up pair for all positions. Keep the IQ3 integer sum and scale order.
template<int count>
static __device__ __forceinline__ void apply_pair(
        const block_iq3_xxs & up, const block_iq3_xxs & gate, const uint32_t * grid,
        const block_q8_1 * activation, const int (&positions)[count], int kbx, int kqs, projections<count> & out) {
    const auto u = load_iq_mmvq_weights(up, kqs);
    const auto g = load_iq_mmvq_weights(gate, kqs);
    const auto * u_codes = reinterpret_cast<const uint8_t *>(&u.qs);
    const auto * g_codes = reinterpret_cast<const uint8_t *>(&g.qs);
    int u_sum[count] = {}, g_sum[count] = {};
    const auto * y = activation + kbx * (QK_K / QK8_1) + kqs / 2;

#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const uint32_t u_signs = uint8_t(unpack_ksigns(u.aux >> (7 * j / 2)));
        const uint32_t g_signs = uint8_t(unpack_ksigns(g.aux >> (7 * j / 2)));
        const int u0 = iq_apply_packed_signs(grid[u_codes[j]], u_signs);
        const int u1 = iq_apply_packed_signs(grid[u_codes[j + 1]], u_signs >> 4);
        const int g0 = iq_apply_packed_signs(grid[g_codes[j]], g_signs);
        const int g1 = iq_apply_packed_signs(grid[g_codes[j + 1]], g_signs >> 4);
#pragma unroll
        for (int p = 0; p < count; ++p) {
            const auto & x = y[positions[p] * activation_blocks];
            const int x0 = get_int_b4(x.qs, j);
            const int x1 = get_int_b4(x.qs, j + 1);
            u_sum[p] = ggml_cuda_dp4a(u0, x0, u_sum[p]);
            u_sum[p] = ggml_cuda_dp4a(u1, x1, u_sum[p]);
            g_sum[p] = ggml_cuda_dp4a(g0, x0, g_sum[p]);
            g_sum[p] = ggml_cuda_dp4a(g1, x1, g_sum[p]);
        }
    }

    const float u_scale = __half2float(u.d);
    const float g_scale = __half2float(g.d);
#pragma unroll
    for (int p = 0; p < count; ++p) {
        const float x_scale = __low2float(y[positions[p] * activation_blocks].ds);
        const int u_dot = (int(u.aux >> 28) * u_sum[p] + u_sum[p] / 2) / 2;
        const int g_dot = (int(g.aux >> 28) * g_sum[p] + g_sum[p] / 2) / 2;
        out.up[p] += (u_scale * x_scale) * u_dot;
        out.gate[p] += (g_scale * x_scale) * g_dot;
    }
}

template<int virtual_base, int count>
static __device__ __forceinline__ projections<count> virtual_lane(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const uint32_t * grid,
        const block_q8_1 * activation, const int (&positions)[count], int lane) {
    const int kbx = (virtual_base + lane) / lanes_per_block;
    const int kqs = VDR_IQ3_XXS_Q8_1_MMVQ * (lane % lanes_per_block);
    projections<count> out{};
    apply_pair(up[kbx], gate[kbx], grid, activation, positions, kbx, kqs, out);
    if constexpr (virtual_base == 0) {
        apply_pair(up[kbx + 8], gate[kbx + 8], grid, activation, positions, kbx + 8, kqs, out);
    }
    return out;
}

template<int count>
static __device__ __forceinline__ void project_row(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const uint32_t * grid,
        const block_q8_1 * activation, const expert_group & group, float * out, int row, int lane) {
    int positions[count];
#pragma unroll
    for (int p = 0; p < count; ++p) {
        positions[p] = group.routes[p] / ranks;
    }

    // Preserve the original wave64 XOR order (32, 16, 8, 4, 2, 1).
    const auto lo = virtual_lane<0>(up, gate, grid, activation, positions, lane);
    const auto hi = virtual_lane<32>(up, gate, grid, activation, positions, lane);
    const auto lo_next = virtual_lane<16>(up, gate, grid, activation, positions, lane);
    const auto hi_next = virtual_lane<48>(up, gate, grid, activation, positions, lane);
#pragma unroll
    for (int p = 0; p < count; ++p) {
        const float u = warp_reduce_sum<subgroup>((lo.up[p] + hi.up[p]) + (lo_next.up[p] + hi_next.up[p]));
        const float g = warp_reduce_sum<subgroup>((lo.gate[p] + hi.gate[p]) + (lo_next.gate[p] + hi_next.gate[p]));
        if (lane == 0) {
            out[group.routes[p] * nrows + row] = u * ggml_cuda_op_silu_single(g);
        }
    }
}

static __device__ __forceinline__ void gate_up_group(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * activation,
        const expert_group & group, float * out, int expert_stride) {
    constexpr int grid_words = sizeof(iq3xxs_grid) / sizeof(iq3xxs_grid[0]);
    constexpr int input_words = activation_blocks * sizeof(block_q8_1) / sizeof(uint32_t);
    __shared__ uint32_t grid[grid_words];
    __shared__ uint32_t input[tokens][input_words];

    const int tid = threadIdx.x;
    for (int i = tid; i < grid_words; i += waves * wave) {
        grid[i] = iq3xxs_grid[i];
    }
    for (int p = 0; p < group.count; ++p) {
        const int token = group.routes[p] / ranks;
        const auto * src = reinterpret_cast<const uint32_t *>(activation + token * activation_blocks);
        for (int i = tid; i < input_words; i += waves * wave) {
            input[token][i] = src[i];
        }
    }
    __syncthreads();

    const int row = blockIdx.x * rows_per_tile + tid / subgroup;
    const size_t offset = size_t(group.expert) * expert_stride + row * weight_blocks;
    const auto * y = reinterpret_cast<const block_q8_1 *>(input);
    const int lane = tid % subgroup;
    switch (group.count) {
        case 1: project_row<1>(up + offset, gate + offset, grid, y, group, out, row, lane); break;
        case 2: project_row<2>(up + offset, gate + offset, grid, y, group, out, row, lane); break;
        case 3: project_row<3>(up + offset, gate + offset, grid, y, group, out, row, lane); break;
    }
}

#ifdef GGML_USE_HIP
__launch_bounds__(waves * wave, 1)
static __global__ void gate_up_packed(
        const block_iq3_xxs * up, const block_iq3_xxs * gate, const block_q8_1 * activation,
        const int32_t * plan, float * out, int expert_stride) {
    namespace layout = ggml_hip_iq3_strata_plan;
    const uint32_t encoded = plan[blockIdx.y];
    if (encoded == 0) {
        return;
    }
    expert_group group{};
    group.expert = int(encoded >> layout::slot_shift) - 1;
#pragma unroll
    for (int token = 0; token < tokens; ++token) {
        const int rank = int((encoded >> (token * layout::rank_bits)) & layout::rank_mask) - 1;
        if (rank >= 0) {
            group.routes[group.count++] = token * ranks + rank;
        }
    }
    gate_up_group(up, gate, activation, group, out, expert_stride);
}
#endif

} // namespace iq3_strata_reuse
