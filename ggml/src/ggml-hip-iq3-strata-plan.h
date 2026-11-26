#pragma once

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace ggml_hip_iq3_strata_plan {

constexpr int experts = 512;
constexpr int tokens = 3;
constexpr int ranks = 10;
constexpr int routes = tokens * ranks;
constexpr int map_entries = experts + routes;
constexpr int rank_bits = 4;
constexpr int rank_mask = (1 << rank_bits) - 1;
constexpr int slot_shift = tokens * rank_bits;
constexpr const char * lookup_name = "ffn_moe_mtp_iq3_strata";
constexpr const char * legacy_lookup_name = "ffn_moe_mtp_share_iq3_weights";
constexpr const char * mode_file_env = "LLAMA_HIP_GFX906_IQ3_STRATA_MODE_FILE";

inline bool has_tail(const ggml_tensor * map) {
    return map && map->type == GGML_TYPE_I32 && map->ne[0] == 1 && map->ne[1] == map_entries &&
        map->ne[2] == 1 && map->ne[3] == 1 && ggml_is_contiguous(map);
}

inline bool uses_plan(const ggml_tensor * lookup) {
    return lookup && lookup->op == GGML_OP_GET_ROWS && has_tail(lookup->src[0]) &&
        (std::strcmp(lookup->name, lookup_name) == 0 || std::strcmp(lookup->name, legacy_lookup_name) == 0);
}

inline ggml_tensor * map_from_ids(const ggml_tensor * ids) {
    const auto * lookup = ids ? ids->view_src : nullptr;
    return uses_plan(lookup) && ids->ne[0] == ranks && ids->ne[1] == tokens ? lookup->src[0] : nullptr;
}

inline void build(int32_t * groups, const int32_t * ids, const int32_t * slots, int32_t zero_slot, int32_t slot_count) {
    std::array<int8_t, experts> first;
    first.fill(-1);
    int count = 0;
    for (int route = 0; route < routes; ++route) {
        const int expert = ids[route];
        const int32_t slot = slots[expert];
        GGML_ASSERT(slot >= 0 && slot < slot_count && slot < experts);
        const int shift = (route / ranks) * rank_bits;
        int index = first[expert];
        // CPU routes share a zero slot, and each rank still needs its own output.
        if (index < 0 || (groups[index] & (rank_mask << shift))) {
            index = count++;
            groups[index] = (slot + 1) << slot_shift;
            if (slot != zero_slot && first[expert] < 0) {
                first[expert] = index;
            }
        }
        groups[index] |= (route % ranks + 1) << shift;
    }
    std::fill(groups + count, groups + routes, 0);
}

} // namespace ggml_hip_iq3_strata_plan
