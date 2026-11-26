#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace ggml_hip_iq3_route_plan {

constexpr int experts = 512;
constexpr int tokens = 3;
constexpr int ranks = 10;
constexpr int routes = tokens * ranks;
constexpr int map_entries = experts + routes;
constexpr int rank_bits = 4;
constexpr int32_t follower = -1;
constexpr const char * lookup_name = "ffn_moe_mtp_share_iq3_weights";

inline bool has_tail(const ggml_tensor * map) {
    return map && map->type == GGML_TYPE_I32 && map->ne[0] == 1 && map->ne[1] == map_entries &&
        map->ne[2] == 1 && map->ne[3] == 1 && ggml_is_contiguous(map);
}

inline bool uses_plan(const ggml_tensor * lookup) {
    return lookup && lookup->op == GGML_OP_GET_ROWS && std::strcmp(lookup->name, lookup_name) == 0 &&
        has_tail(lookup->src[0]);
}

inline ggml_tensor * map_from_ids(const ggml_tensor * ids) {
    const auto * lookup = ids ? ids->view_src : nullptr;
    return uses_plan(lookup) && ids->ne[0] == ranks && ids->ne[1] == tokens ? lookup->src[0] : nullptr;
}

inline std::array<int32_t, routes> build(const int32_t * ids, const int32_t * slots, int32_t zero_slot) {
    // Each nibble stores the rank plus one. Zero means the token did not select this expert.
    constexpr uint16_t repeated_rank = 1 << (tokens * rank_bits);
    std::array<uint16_t, experts> groups{};
    for (int i = 0; i < routes; ++i) {
        const int expert = ids[i];
        if (slots[expert] < 0 || slots[expert] == zero_slot) {
            continue;
        }
        const int shift = (i / ranks) * rank_bits;
        auto & group = groups[expert];
        if ((group >> shift) & 0xf) {
            group |= repeated_rank;
        }
        group |= (i % ranks + 1) << shift;
    }

    std::array<int32_t, routes> plan{};
    for (int i = 0; i < routes; ++i) {
        const int32_t group = groups[ids[i]];
        const int shift = (i / ranks) * rank_bits;
        const int32_t earlier_ranks = group & ((1 << shift) - 1);
        const int32_t later_ranks = group >> (shift + rank_bits);
        // Split routes with repeated ranks in any token, preserving every output.
        if (!(group & repeated_rank) && (earlier_ranks || later_ranks)) {
            plan[i] = earlier_ranks ? follower : group;
        }
    }
    return plan;
}

} // namespace ggml_hip_iq3_route_plan
