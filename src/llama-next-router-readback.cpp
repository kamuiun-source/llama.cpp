#include "llama-next-router-readback.h"

#include "llama-impl.h"

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace {

std::atomic<bool> next_router_readback_enabled{false};

const char * next_router_readback_mode_file() {
    static const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_NEXT_ROUTER_READBACK_MODE_FILE");
    return path && *path ? path : nullptr;
}

} // namespace

void llama_refresh_next_router_readback_mode() {
    const char * path = next_router_readback_mode_file();
    if (!path) {
        return;
    }

    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    char buffer[64];
    input.read(buffer, sizeof(buffer));
    std::string_view mode(buffer, input.gcount());
    if (!mode.empty() && mode.back() == '\n') {
        mode.remove_suffix(1);
    }
    if (!mode.empty() && mode.back() == '\r') {
        mode.remove_suffix(1);
    }
    static bool invalid = false;
    const bool enabled = mode == "NEXT_ROUTER_READBACK=on";
    if (!input.eof() || input.bad() || (!enabled && mode != "NEXT_ROUTER_READBACK=off")) {
        if (!invalid) {
            LLAMA_LOG_WARN("invalid HIP fixed-three next Router readback mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (next_router_readback_enabled.exchange(enabled, std::memory_order_relaxed) != enabled) {
        LLAMA_LOG_INFO("HIP fixed-three next Router readback mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}

llama_next_router_readback::totals * llama_next_router_readback::begin_graph(bool measure) {
    if (!next_router_readback_mode_file()) {
        return nullptr;
    }
    enabled = next_router_readback_enabled.load(std::memory_order_relaxed);
    if (!measure) {
        return nullptr;
    }
    auto & totals = by_mode[enabled];
    ++totals.graphs;
    return &totals;
}

bool llama_next_router_readback::submit(ggml_backend_t producer, const ggml_tensor * ids, size_t size) {
    if (producer_backend != producer) {
        producer_backend = producer;
        host.reset();
        event.reset();

        auto * device = ggml_backend_get_device(producer);
        auto * reg = ggml_backend_dev_backend_reg(device);
        auto * host_type = reg && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0 ?
                ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (host_type) {
            host.reset(ggml_backend_buft_alloc_buffer(host_type, size));
            // CUDA can fall back to a pageable buffer.
            if (host && ggml_backend_buffer_get_type(host.get()) == host_type) {
                event.reset(ggml_backend_event_new(device));
            }
            if (!event) {
                host.reset();
            }
        }
    }
    if (!event) {
        return false;
    }
    GGML_ASSERT(size <= ggml_backend_buffer_get_size(host.get()));

    // The copy and the event are queued on the same stream as the producer graph.
    ggml_backend_tensor_get_async(producer, ids, ggml_backend_buffer_get_base(host.get()), 0, size);
    ggml_backend_event_record(event.get(), producer);
    return true;
}

void llama_next_router_readback::complete(void * dst, size_t size) {
    ggml_backend_event_synchronize(event.get());
    std::memcpy(dst, ggml_backend_buffer_get_base(host.get()), size);
}

void llama_next_router_readback::log_stats() const {
    if (!next_router_readback_mode_file()) {
        return;
    }
    for (int mode = 0; mode < 2; ++mode) {
        const auto & totals = by_mode[mode];
        LLAMA_LOG_INFO("packed expert HIP next Router readback: mode=%s graphs=%" PRIu64 " reads=%" PRIu64
                " applied=%" PRIu64 " fallback=%" PRIu64 " submit_cpu_us=%" PRId64 " wait_cpu_us=%" PRId64
                " demand_tuples=%" PRIu64 " hit_tuples=%" PRIu64 "\n",
                mode ? "on" : "off", totals.graphs, totals.reads, totals.applied, mode ? totals.reads - totals.applied : 0,
                totals.submit_us, totals.wait_us, totals.demand_tuples, totals.hit_tuples);
    }
}
