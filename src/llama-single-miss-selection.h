#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

struct llama_single_miss_budget {
    static constexpr int tokens = 3, ranks = 10, max_jobs = tokens * ranks;
    int physical_cores;

    int jobs() const { return std::clamp((physical_cores - 2) / 2, 0, max_jobs); }
    int workers(int n_jobs) const { return n_jobs == 1 ? std::min(4, physical_cores - 2) : 2 * n_jobs; }
};

struct llama_single_miss_selection {
    int actual = 0;
    std::array<int32_t, llama_single_miss_budget::max_jobs> ids{};
};

template <typename IsMiss>
llama_single_miss_selection llama_select_cpu_expert_misses(const int32_t * ids, IsMiss is_miss) {
    using budget = llama_single_miss_budget;
    struct candidate {
        int32_t id;
        unsigned positions;
    };
    std::array<candidate, budget::max_jobs> candidates;
    int count = 0;
    for (int token = 0; token < budget::tokens; ++token) {
        for (int rank = 0; rank < budget::ranks; ++rank) {
            const int32_t id = ids[token * budget::ranks + rank];
            if (!is_miss(id)) {
                continue;
            }
            int i = 0;
            while (i < count && candidates[i].id != id) {
                ++i;
            }
            if (i == count) {
                candidates[count++] = {id, 0};
            }
            candidates[i].positions |= 1u << token;
        }
    }

    llama_single_miss_selection result;
    result.actual = count;
    if (count <= 2) {
        for (int i = 0; i < count; ++i) {
            result.ids[i] = candidates[i].id;
        }
        return result;
    }
    int out = 0;
    // For each position count, retain the first Router encounter among ties.
    for (unsigned used = 1; used <= budget::tokens; ++used) {
        for (int i = 0; i < count; ++i) {
            const unsigned mask = candidates[i].positions;
            if ((mask & 1u) + ((mask >> 1) & 1u) + (mask >> 2) == used) {
                result.ids[out++] = candidates[i].id;
            }
        }
    }
    return result;
}
