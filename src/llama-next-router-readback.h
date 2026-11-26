#pragma once

#include "ggml-cpp.h"
#include "llama.h"

#include <cstdint>

LLAMA_API void llama_refresh_next_router_readback_mode(bool top10);

struct llama_next_router_readback {
    struct totals {
        uint64_t graphs = 0;
        uint64_t reads = 0;
        uint64_t pinned = 0;
        uint64_t deferred = 0;
        uint64_t polls = 0;
        uint64_t ready = 0;
        uint64_t deadline_ready = 0;
        uint64_t late = 0;
        uint64_t canceled = 0;
        int64_t poll_cpu_us = 0;
        int64_t residence_us = 0;
        int64_t cleanup_cpu_us = 0;
        int64_t plan_cpu_us = 0;
        int64_t graph_cpu_us = 0;
        uint64_t demand_tuples = 0;
        uint64_t hit_tuples = 0;
        int64_t submit_us = 0;
        int64_t wait_us = 0;
    };

    totals * begin_graph(bool measure);
    bool submit(ggml_backend_t producer, const ggml_tensor * ids, size_t size, int32_t defer_layer = -1);
    bool pending() const { return pending_layer >= 0; }
    int32_t poll(int32_t deadline_layer, void * dst);
    void discard_pending();
    void end_graph();
    void complete(void * dst, size_t size);
    void log_stats() const;

    bool enabled = false;

private:
    // Indexed by partial Top10 and next Router readback mode.
    totals by_mode[2][2];
    totals * measured = nullptr;
    bool in_flight = false;
    int32_t pending_layer = -1;
    size_t pending_size = 0;
    int64_t submitted_us = 0;
    bool (*event_ready)(ggml_backend_t, ggml_backend_event_t) = nullptr;
    void drain();
    ggml_backend_t producer_backend = nullptr;
    ggml_backend_buffer_ptr host;
    ggml_backend_event_ptr event;
};
