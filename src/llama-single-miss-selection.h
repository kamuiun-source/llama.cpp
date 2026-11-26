#pragma once

#include <array>
#include <cstdint>

struct llama_single_miss_selection {
    int actual = 0;
    std::array<int32_t, 2> first{};
    std::array<int32_t, 2> fewest{};
};

template <typename IsMiss>
llama_single_miss_selection llama_select_cpu_expert_misses(const int32_t * ids, bool hybrid, IsMiss is_miss) {
    struct candidate {
        int32_t id;
        unsigned positions;
    };
    std::array<candidate, 30> candidates;
    int count = 0;
    for (int token = 0; token < 3; ++token) {
        for (int rank = 0; rank < 10; ++rank) {
            const int32_t id = ids[token * 10 + rank];
            if (!is_miss(id)) {
                continue;
            }
            int i = 0;
            while (i < count && candidates[i].id != id) {
                ++i;
            }
            if (i == count) {
                candidates[count++] = {id, 0};
                if (count == 3 && !hybrid) {
                    return {3, {}, {}};
                }
            }
            candidates[i].positions |= 1u << token;
        }
    }

    llama_single_miss_selection result;
    result.actual = count;
    if (count > 0) {
        result.first[0] = candidates[0].id;
    }
    if (count > 1) {
        result.first[1] = candidates[1].id;
    }
    if (count < 3) {
        result.fewest = result.first;
        return result;
    }

    const auto positions = [](unsigned mask) {
        return (mask & 1u) + ((mask >> 1) & 1u) + (mask >> 2);
    };
    int best = 0, second = 1;
    if (positions(candidates[second].positions) < positions(candidates[best].positions)) {
        best = 1;
        second = 0;
    }
    for (int i = 2; i < count; ++i) {
        const unsigned used = positions(candidates[i].positions);
        if (used < positions(candidates[best].positions)) {
            second = best;
            best = i;
        } else if (used < positions(candidates[second].positions)) {
            second = i;
        }
    }
    result.fewest = {candidates[best].id, candidates[second].id};
    return result;
}
