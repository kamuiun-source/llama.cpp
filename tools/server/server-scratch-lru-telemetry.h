#pragma once

#include "src/llama-ext.h"
#include "log.h"

#include <cinttypes>

struct server_scratch_lru_telemetry {
    uint64_t calls = 0;
    uint64_t demand_h2d_tuples = 0, demand_h2d_bytes = 0;
    uint64_t applied = 0, changed = 0;

    void add(const llama_expert_round_stats & before, const llama_expert_round_stats & after) {
        ++calls;
        demand_h2d_tuples += after.tuple_demand_h2d_count - before.tuple_demand_h2d_count;
        demand_h2d_bytes += after.tuple_demand_h2d_bytes - before.tuple_demand_h2d_bytes;
        applied += after.scratch_lru_applied - before.scratch_lru_applied;
        changed += after.scratch_lru_changed - before.scratch_lru_changed;
    }

    void log(int request, bool enabled) const {
        LOG_INF("scratch_lru: phase=target_verify request=%d mode=%s calls=%" PRIu64
                " demand_h2d_tuples=%" PRIu64 " demand_h2d_bytes=%" PRIu64 " applied=%" PRIu64 " changed=%" PRIu64 "\n",
                request, enabled ? "on" : "off", calls, demand_h2d_tuples, demand_h2d_bytes, applied, changed);
    }
};
