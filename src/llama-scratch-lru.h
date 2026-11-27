#pragma once

#include <atomic>

inline std::atomic<bool> llama_expert_scratch_lru_enabled{false};

template <typename Iterator, typename Eligible, typename LastUsed>
Iterator llama_expert_scratch_lru_victim(Iterator begin, Iterator end, Eligible eligible, LastUsed last_used, bool & changed) {
    auto first = end;
    auto victim = end;
    for (auto it = begin; it != end; ++it) {
        if (!eligible(*it)) {
            continue;
        }
        if (first == end) {
            first = it;
        }
        if (victim == end || last_used(*it) < last_used(*victim)) {
            victim = it;
        }
    }
    changed = victim != first;
    return victim;
}
