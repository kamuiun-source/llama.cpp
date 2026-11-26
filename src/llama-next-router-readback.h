#pragma once

#include "ggml-cpp.h"
#include "llama.h"

#include <cstdint>

LLAMA_API void llama_refresh_next_router_readback_mode();

struct llama_next_router_readback {
    struct totals {
        uint64_t graphs = 0;
        uint64_t reads = 0;
        uint64_t applied = 0;
        uint64_t demand_tuples = 0;
        uint64_t hit_tuples = 0;
        int64_t submit_us = 0;
        int64_t wait_us = 0;
    };

    totals * begin_graph(bool measure);
    bool submit(ggml_backend_t producer, const ggml_tensor * ids, size_t size);
    void complete(void * dst, size_t size);
    void log_stats() const;

    bool enabled = false;

private:
    totals by_mode[2];
    ggml_backend_t producer_backend = nullptr;
    ggml_backend_buffer_ptr host;
    ggml_backend_event_ptr event;
};
