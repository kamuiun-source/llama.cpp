#pragma once

#include "llama.h"
#include "llama-single-miss-selection.h"

#include <atomic>
#include <cstddef>

LLAMA_API void llama_set_late_prefetch_cpu(bool enabled);

inline std::atomic<bool> llama_late_prefetch_cpu_enabled{false};

inline bool llama_late_prefetch_cpu_has_room(int demand_jobs, const llama_single_miss_budget & budget) {
    // A single demand job gets up to four workers. Adding a job would take workers from it.
    return demand_jobs != 1 && demand_jobs < budget.jobs();
}

struct llama_late_prefetch_cpu_candidate {
    int32_t expert = -1;
    bool sole_hit = false;
};

template <typename NeedsUpload, typename IsSafe, typename IsReady>
llama_late_prefetch_cpu_candidate llama_late_prefetch_cpu_select(
        const int32_t * routes, const int32_t * issued, size_t n_issued,
        NeedsUpload needs_upload, IsSafe is_safe, IsReady is_ready) {
    const auto * end = routes + llama_single_miss_budget::max_jobs;
    size_t last = n_issued;
    size_t previous = n_issued;
    for (size_t i = n_issued; i > 0; --i) {
        const int32_t expert = issued[i - 1];
        if (!needs_upload(expert)) {
            continue;
        }
        const auto uses = std::count(routes, end, expert);
        if (uses == 0) {
            continue;
        }
        if (last == n_issued) {
            if (uses != 1 || !is_safe(expert, i - 1)) {
                return {};
            }
            last = i - 1;
        } else {
            previous = i - 1;
            break;
        }
    }
    // The upload stream is FIFO, so only the two latest required events matter.
    if (last == n_issued || (previous < n_issued && !is_ready(previous)) || is_ready(last)) {
        return {};
    }
    return {issued[last], previous == n_issued};
}
