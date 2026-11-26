#pragma once

#include "unary.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

static std::atomic<int> gfx906_iq3_share_override{-1};

bool ggml_hip_gfx906_iq3_share_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HIP_GFX906_MTP3_IQ3_XXS_SHARE_WEIGHTS");
        return value && std::strcmp(value, "1") == 0;
    }();
    const int mode = gfx906_iq3_share_override.load(std::memory_order_relaxed);
    return mode < 0 ? enabled : mode != 0;
}

void ggml_hip_gfx906_set_iq3_share(bool enabled) {
    gfx906_iq3_share_override.store(enabled, std::memory_order_relaxed);
}

static bool iq3_share_eligible(const ggml_tensor * ids, const ggml_tensor * dst) {
    const auto * lookup = ids ? ids->view_src : nullptr;
    return lookup && lookup->op == GGML_OP_GET_ROWS &&
        std::strcmp(lookup->name, "ffn_moe_mtp_share_iq3_weights") == 0 &&
        ids->ne[0] == 10 && ids->ne[1] == 3 && ggml_is_contiguous(dst);
}

static constexpr int iq3_share_tokens = 3;
static constexpr int iq3_share_ranks = 10;

static __device__ __forceinline__ uint32_t iq3_share_matches(
        const int32_t * ids, uint32_t ids_stride, uint32_t expert) {
    const int lane = threadIdx.x;
    return uint32_t(__ballot(lane < iq3_share_tokens*iq3_share_ranks &&
        uint32_t(ids[(lane / iq3_share_ranks)*ids_stride + lane % iq3_share_ranks]) == expert));
}

template<int activation_blocks>
static __device__ __forceinline__ void iq3_share_dot(
        const iq3_xxs_mmvq_weights & weight, const block_q8_1 * activation,
        const uint32_t * grid, const int (&ranks)[iq3_share_tokens], float (&acc)[iq3_share_tokens]) {
    const auto * q3 = reinterpret_cast<const uint8_t *>(&weight.qs);
    int sums[iq3_share_tokens] = {};
#pragma unroll
    for (int l = 0; l < 8; l += 2) {
        const uint32_t signs = uint8_t(unpack_ksigns(weight.aux >> (7*l/2)));
        const int grid_l = iq_apply_packed_signs(grid[q3[l]], signs);
        const int grid_h = iq_apply_packed_signs(grid[q3[l + 1]], signs >> 4);
#pragma unroll
        for (int t = 0; t < iq3_share_tokens; ++t) {
            if (ranks[t] >= 0) {
                const auto & q8 = activation[t * activation_blocks];
                sums[t] = ggml_cuda_dp4a(grid_l, get_int_b4(q8.qs, l), sums[t]);
                sums[t] = ggml_cuda_dp4a(grid_h, get_int_b4(q8.qs, l + 1), sums[t]);
            }
        }
    }
    const int scale = weight.aux >> 28;
    const float weight_d = __half2float(weight.d);
#pragma unroll
    for (int t = 0; t < iq3_share_tokens; ++t) {
        if (ranks[t] >= 0) {
            const int scaled = (scale*sums[t] + sums[t]/2)/2;
            const float d = weight_d * __low2float(activation[t * activation_blocks].ds);
            const float dot = d * scaled;
            acc[t] += dot;
        }
    }
}

template<int rows_per_group, int row_groups, int activation_blocks>
static __device__ __forceinline__ bool iq3_share_project(
        const void * vx, const void * vgate, const int32_t * ids, float * dst,
        const block_q8_1 * activation, const uint32_t * grid, uint32_t expert, int blocks_per_row,
        uint32_t stride_row_x, uint32_t stride_channel_x, uint32_t stride_channel_dst, uint32_t stride_col_dst,
        uint32_t ids_stride) {
    const uint32_t matches = iq3_share_matches(ids, ids_stride, expert);
    int ranks[iq3_share_tokens];
    bool duplicate = false;
#pragma unroll
    for (int t = 0; t < iq3_share_tokens; ++t) {
        const uint32_t token_matches = (matches >> (t*iq3_share_ranks)) & ((1u << iq3_share_ranks) - 1);
        ranks[t] = __ffs(token_matches) - 1;
        duplicate |= (token_matches & (token_matches - 1)) != 0;
    }
    // A repeated slot in one token has multiple output ranks. Keep those routes independent.
    if (duplicate || (matches & (matches - 1)) == 0) {
        return false;
    }
    if ((matches & ((1u << (threadIdx.y*iq3_share_ranks)) - 1)) != 0) {
        return true;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int lanes_per_block = QI3_XXS / VDR_IQ3_XXS_Q8_1_MMVQ;
    constexpr int blocks_per_iter = warp_size / lanes_per_block;
    const int kbx = threadIdx.x / lanes_per_block;
    const int kbx_next = kbx + blocks_per_iter;
    const int kqs = VDR_IQ3_XXS_Q8_1_MMVQ * (threadIdx.x % lanes_per_block);
    const auto * up = static_cast<const block_iq3_xxs *>(vx) + expert*stride_channel_x;
    const auto * gate = static_cast<const block_iq3_xxs *>(vgate) + expert*stride_channel_x;
    activation += kqs / 2;
    const auto * current = activation + kbx*(QK_K/QK8_1);

#pragma unroll 1
    for (int group = 0; group < row_groups; ++group) {
        const int row0 = rows_per_group*(row_groups*blockIdx.x + group);
        float acc_up[rows_per_group][iq3_share_tokens] = {};
        float acc_gate[rows_per_group][iq3_share_tokens] = {};
#pragma unroll
        for (int r = 0; r < rows_per_group; ++r) {
            const int row = (row0 + r)*stride_row_x;
            const auto up_current = load_iq_mmvq_weights(up[row + kbx], kqs);
            const auto gate_current = load_iq_mmvq_weights(gate[row + kbx], kqs);
            iq3_xxs_mmvq_weights up_next, gate_next;
            if (kbx_next < blocks_per_row) {
                up_next = load_iq_mmvq_weights(up[row + kbx_next], kqs);
                gate_next = load_iq_mmvq_weights(gate[row + kbx_next], kqs);
            }

            iq3_share_dot<activation_blocks>(up_current, current, grid, ranks, acc_up[r]);
            iq3_share_dot<activation_blocks>(gate_current, current, grid, ranks, acc_gate[r]);
            if (kbx_next < blocks_per_row) {
                const auto * next = activation + kbx_next*(QK_K/QK8_1);
                iq3_share_dot<activation_blocks>(up_next, next, grid, ranks, acc_up[r]);
                iq3_share_dot<activation_blocks>(gate_next, next, grid, ranks, acc_gate[r]);
            }
        }
#pragma unroll
        for (int t = 0; t < iq3_share_tokens; ++t) {
            if (ranks[t] >= 0) {
#pragma unroll
                for (int r = 0; r < rows_per_group; ++r) {
                    const float u = warp_reduce_sum<warp_size>(acc_up[r][t]);
                    const float g = warp_reduce_sum<warp_size>(acc_gate[r][t]);
                    if (threadIdx.x == r) {
                        dst[t*stride_col_dst + ranks[t]*stride_channel_dst + row0 + r] = u * ggml_cuda_op_silu_single(g);
                    }
                }
            }
        }
    }
    return true;
}
