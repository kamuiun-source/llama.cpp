#pragma once

#include "iq3-strata-reuse.cuh"

namespace iq2_strata_reuse {

using namespace iq3_strata_reuse;
static_assert(QI2_S / VDR_IQ2_S_Q8_1_MMVQ == lanes_per_block && sizeof(block_iq2_s) == 82);

static __device__ __forceinline__ int2 decode_pair(const iq2_s_mmvq_weights & weight, int l) {
    const auto * qs = reinterpret_cast<const uint8_t *>(&weight.qs);
    const auto * signs = reinterpret_cast<const uint8_t *>(&weight.signs);
    const uint64_t grid = iq2s_grid[qs[l/2] | ((int(weight.qh) << (8-l)) & 0x300)];
    return make_int2(iq_apply_packed_signs(uint32_t(grid), signs[l/2]),
                     iq_apply_packed_signs(uint32_t(grid >> 32), signs[l/2] >> 4));
}

static __device__ __forceinline__ int scale_sum(const int (&sum)[2], uint8_t scales) {
    return (sum[0]*(scales & 0xf) + sum[1]*(scales >> 4) + (sum[0] + sum[1])/2)/4;
}

template<int count>
static __device__ __forceinline__ void apply_pair(
        const block_iq2_s & up, const block_iq2_s & gate, const block_q8_1 * input,
        const int (&positions)[count], int kbx, int kqs, projections<count> & out) {
    const auto u = load_iq_mmvq_weights(up, kqs);
    const auto g = load_iq_mmvq_weights(gate, kqs);
    const auto * y = input + kbx * (QK_K / QK8_1) + kqs / 2;
    int u_sum[count][2] = {}, g_sum[count][2] = {};

#pragma unroll
    for (int l = 0; l < 8; l += 2) {
        const auto u_grid = decode_pair(u, l);
        const auto g_grid = decode_pair(g, l);
        const int half = l / 4;
#pragma unroll
        for (int p = 0; p < count; ++p) {
            const auto & x = y[positions[p] * activation_blocks];
            const int x0 = get_int_b4(x.qs, l);
            const int x1 = get_int_b4(x.qs, l + 1);
            u_sum[p][half] = ggml_cuda_dp4a(u_grid.x, x0, u_sum[p][half]);
            g_sum[p][half] = ggml_cuda_dp4a(g_grid.x, x0, g_sum[p][half]);
            u_sum[p][half] = ggml_cuda_dp4a(u_grid.y, x1, u_sum[p][half]);
            g_sum[p][half] = ggml_cuda_dp4a(g_grid.y, x1, g_sum[p][half]);
        }
    }
    const float u_d = __half2float(u.d);
    const float g_d = __half2float(g.d);
#pragma unroll
    for (int p = 0; p < count; ++p) {
        const float x_d = __low2float(y[positions[p] * activation_blocks].ds);
        out.up[p] += (u_d * x_d) * scale_sum(u_sum[p], u.scales);
        out.gate[p] += (g_d * x_d) * scale_sum(g_sum[p], g.scales);
    }
}

template<int virtual_base, int count>
static __device__ __forceinline__ projections<count> virtual_lane(
        const block_iq2_s * up, const block_iq2_s * gate, const block_q8_1 * activation,
        const int (&positions)[count], int lane) {
    const int kbx = (virtual_base + lane) / lanes_per_block;
    const int kqs = VDR_IQ2_S_Q8_1_MMVQ * (lane % lanes_per_block);
    projections<count> out{};
    apply_pair(up[kbx], gate[kbx], activation, positions, kbx, kqs, out);
    if constexpr (virtual_base == 0) {
        apply_pair(up[kbx + 8], gate[kbx + 8], activation, positions, kbx + 8, kqs, out);
    }
    return out;
}

template<int count>
static __device__ __forceinline__ void project_row(
        const block_iq2_s * up, const block_iq2_s * gate, const block_q8_1 * activation,
        const expert_group & group, float * out, int row, int lane) {
    int positions[count];
#pragma unroll
    for (int p = 0; p < count; ++p) {
        positions[p] = group.routes[p] / ranks;
    }

    // Emulate the first two XOR64 stages before reducing the physical 16-lane subgroup.
    const auto lo = virtual_lane<0>(up, gate, activation, positions, lane);
    const auto hi = virtual_lane<32>(up, gate, activation, positions, lane);
    const auto next_lo = virtual_lane<16>(up, gate, activation, positions, lane);
    const auto next_hi = virtual_lane<48>(up, gate, activation, positions, lane);
#pragma unroll
    for (int p = 0; p < count; ++p) {
        const float u = warp_reduce_sum<subgroup>((lo.up[p] + hi.up[p]) + (next_lo.up[p] + next_hi.up[p]));
        const float g = warp_reduce_sum<subgroup>((lo.gate[p] + hi.gate[p]) + (next_lo.gate[p] + next_hi.gate[p]));
        if (lane == 0) {
            out[group.routes[p] * nrows + row] = u * ggml_cuda_op_silu_single(g);
        }
    }
}

__launch_bounds__(waves * wave, 1)
static __global__ void gate_up(
        const block_iq2_s * up, const block_iq2_s * gate, const block_q8_1 * activation,
        const expert_group * groups, float * out) {
    const int tid = threadIdx.x;
    const expert_group group = groups[blockIdx.y];
    const int row = blockIdx.x * rows_per_tile + tid / subgroup;
    const int offset = (group.expert * nrows + row) * weight_blocks;
    const int lane = tid % subgroup;
    switch (group.count) {
        case 1: project_row<1>(up + offset, gate + offset, activation, group, out, row, lane); break;
        case 2: project_row<2>(up + offset, gate + offset, activation, group, out, row, lane); break;
        case 3: project_row<3>(up + offset, gate + offset, activation, group, out, row, lane); break;
    }
}

} // namespace iq2_strata_reuse
