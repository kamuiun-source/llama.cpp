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

std::atomic<bool> next_router_top10{false};

struct readback_mode {
    const char * path;
    std::string_view off;
    std::string_view on;
    std::atomic<bool> enabled{false};
    bool invalid = false;

    explicit readback_mode(const char * env, std::string_view off, std::string_view on) :
        path(std::getenv(env)), off(off), on(on) {}

    bool configured() const {
        return path && *path;
    }

    void refresh();
};

readback_mode & pinned_mode() {
    static readback_mode mode("LLAMA_EXPERT_HIP_MTP3_NEXT_ROUTER_READBACK_MODE_FILE",
            "NEXT_ROUTER_READBACK=off", "NEXT_ROUTER_READBACK=on");
    return mode;
}

readback_mode & defer_mode() {
    static readback_mode mode("LLAMA_EXPERT_HIP_MTP3_NEXT_ROUTER_DEFER_MODE_FILE", "NEXT_ROUTER_DEFER=off", "NEXT_ROUTER_DEFER=on");
    return mode;
}

void readback_mode::refresh() {
    if (!configured()) {
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
    const bool requested = mode == on;
    if (!input.eof() || input.bad() || (!requested && mode != off)) {
        if (!invalid) {
            LLAMA_LOG_WARN("invalid HIP fixed-three next Router readback mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (enabled.exchange(requested, std::memory_order_relaxed) != requested) {
        LLAMA_LOG_INFO("HIP fixed-three next Router readback mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}

} // namespace

void llama_refresh_next_router_readback_mode(bool top10) {
    next_router_top10.store(top10, std::memory_order_relaxed);
    pinned_mode().refresh();
    defer_mode().refresh();
}

llama_next_router_readback::totals * llama_next_router_readback::begin_graph(bool measure) {
    GGML_ASSERT(!in_flight && pending_layer < 0);
    deferred = defer_mode().enabled.load(std::memory_order_relaxed);
    enabled = deferred || pinned_mode().enabled.load(std::memory_order_relaxed);
    if (!measure || (!pinned_mode().configured() && !defer_mode().configured())) {
        return nullptr;
    }
    const bool top10 = next_router_top10.load(std::memory_order_relaxed);
    measured = &by_mode[top10][deferred][enabled];
    ++measured->graphs;
    return measured;
}

bool llama_next_router_readback::submit(ggml_backend_t producer, const ggml_tensor * ids, size_t size, int32_t defer_layer) {
    GGML_ASSERT(pending_layer < 0);
    if (producer_backend != producer) {
        drain();
        producer_backend = producer;
        host.reset();
        event.reset();
        event_ready = nullptr;

        auto * device = ggml_backend_get_device(producer);
        auto * reg = ggml_backend_dev_backend_reg(device);
        auto * host_type = reg && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0 ?
                ggml_backend_dev_host_buffer_type(device) : nullptr;
        if (host_type) {
            host.reset(ggml_backend_buft_alloc_buffer(host_type, size));
            // CUDA can fall back to a pageable buffer.
            if (host && ggml_backend_buffer_get_type(host.get()) == host_type) {
                event.reset(ggml_backend_event_new(device));
                event_ready = reinterpret_cast<decltype(event_ready)>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_event_ready"));
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

    // An expired copy on this stream can finish before the next copy overwrites the buffer.
    ggml_backend_tensor_get_async(producer, ids, ggml_backend_buffer_get_base(host.get()), 0, size);
    ggml_backend_event_record(event.get(), producer);
    in_flight = true;
    if (defer_layer >= 0 && event_ready) {
        pending_layer = defer_layer;
        pending_size = size;
        if (measured) {
            submitted_us = ggml_time_us();
            ++measured->deferred;
        }
    }
    return true;
}

int32_t llama_next_router_readback::poll(int32_t deadline_layer, void * dst) {
    if (pending_layer < 0 || (deadline_layer >= 0 && pending_layer != deadline_layer)) {
        return -1;
    }
    const int64_t start = measured ? ggml_time_us() : 0;
    const bool ready = event_ready(producer_backend, event.get());
    const bool deadline = deadline_layer >= 0;
    const int32_t layer = pending_layer;
    if (ready) {
        if (!deadline) {
            std::memcpy(dst, ggml_backend_buffer_get_base(host.get()), pending_size);
        }
        in_flight = false;
    }
    if (ready || deadline) {
        pending_layer = -1;
    }
    if (measured) {
        const auto now = ggml_time_us();
        ++measured->polls;
        measured->ready += ready;
        measured->deadline_ready += ready && deadline;
        measured->late += deadline;
        measured->poll_cpu_us += now - start;
        if (ready || deadline) {
            measured->residence_us += now - submitted_us;
        }
    }
    return ready && !deadline ? layer : -1;
}

void llama_next_router_readback::drain() {
    if (in_flight) {
        const int64_t start = measured ? ggml_time_us() : 0;
        ggml_backend_event_synchronize(event.get());
        in_flight = false;
        if (measured) {
            measured->cleanup_cpu_us += ggml_time_us() - start;
        }
    }
}

void llama_next_router_readback::complete(void * dst, size_t size) {
    ggml_backend_event_synchronize(event.get());
    in_flight = false;
    std::memcpy(dst, ggml_backend_buffer_get_base(host.get()), size);
}

void llama_next_router_readback::discard_pending() {
    if (pending_layer >= 0) {
        if (measured) {
            ++measured->canceled;
            measured->residence_us += ggml_time_us() - submitted_us;
        }
        pending_layer = -1;
    }
}

void llama_next_router_readback::end_graph() {
    discard_pending();
    drain();
    measured = nullptr;
}

void llama_next_router_readback::log_stats() const {
    if (!pinned_mode().configured() && !defer_mode().configured()) {
        return;
    }
    for (int top10 = 0; top10 < 2; ++top10) {
        const char * top10_mode = top10 ? "on" : "off";
        for (int mode = 0; mode < 2; ++mode) {
            const auto & totals = by_mode[top10][0][mode];
            if (top10 && !totals.graphs) {
                continue;
            }
            LLAMA_LOG_INFO("packed expert HIP next Router readback: mode=%s graphs=%" PRIu64 " reads=%" PRIu64
                    " applied=%" PRIu64 " fallback=%" PRIu64 " submit_cpu_us=%" PRId64 " wait_cpu_us=%" PRId64
                    " demand_tuples=%" PRIu64 " hit_tuples=%" PRIu64 " graph_cpu_us=%" PRId64 " top10=%s\n",
                    mode ? "on" : "off", totals.graphs, totals.reads, totals.applied, mode ? totals.reads - totals.applied : 0,
                    totals.submit_us, totals.wait_us, totals.demand_tuples, totals.hit_tuples, totals.graph_cpu_us, top10_mode);
        }
        const auto & totals = by_mode[top10][1][1];
        if (!defer_mode().configured() || (top10 && !totals.graphs)) {
            continue;
        }
        LLAMA_LOG_INFO("packed expert HIP next Router deferred readback: mode=on graphs=%" PRIu64
                " reads=%" PRIu64 " pinned=%" PRIu64 " applied=%" PRIu64 " fallback=%" PRIu64
                " polls=%" PRIu64 " ready=%" PRIu64 " deadline_ready=%" PRIu64 " late=%" PRIu64 " canceled=%" PRIu64
                " submit_cpu_us=%" PRId64 " wait_cpu_us=%" PRId64 " poll_cpu_us=%" PRId64
                " residence_us=%" PRId64 " cleanup_cpu_us=%" PRId64 " plan_cpu_us=%" PRId64 " graph_cpu_us=%" PRId64
                " demand_tuples=%" PRIu64 " hit_tuples=%" PRIu64 " top10=%s\n",
                totals.graphs, totals.reads, totals.applied, totals.deferred, totals.reads - totals.deferred,
                totals.polls, totals.ready, totals.deadline_ready, totals.late, totals.canceled,
                totals.submit_us, totals.wait_us, totals.poll_cpu_us, totals.residence_us, totals.cleanup_cpu_us,
                totals.plan_cpu_us, totals.graph_cpu_us, totals.demand_tuples, totals.hit_tuples, top10_mode);
    }
}
