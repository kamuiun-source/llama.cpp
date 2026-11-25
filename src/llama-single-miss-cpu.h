#pragma once

#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <system_error>
#include <thread>

inline bool llama_single_miss_cpu_configured() {
    const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_MISS_CPU_MODE_FILE");
    return path && *path;
}

struct llama_single_miss_cpu {
    static constexpr int width = 2560, inner = 640, tokens = 3, ranks = 10;
    size_t eligible = 0, applied = 0, positions = 0, avoided_bytes = 0;
    size_t joined = 0, waited = 0, cancelled = 0, outstanding_at_mi50 = 0;
    int64_t compute_us = 0, queue_us = 0, join_us = 0, remaining_after_mi50_us = 0;

    ~llama_single_miss_cpu() {
        drain();
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void drain() {
        resolve(0);
    }

    const float * resolve(int64_t mi50_done_us) {
        const int64_t join_start = ggml_time_us();
        std::unique_lock<std::mutex> lock(mutex);
        if (!pending) {
            return nullptr;
        }
        const bool must_wait = !done;
        changed.wait(lock, [&] { return done; });
        pending = false;
        compute_us += completed_us - started_us;
        queue_us += started_us - queued_us;
        if (mi50_done_us) {
            ++joined;
            if (completed_us > mi50_done_us) {
                ++outstanding_at_mi50;
                remaining_after_mi50_us += completed_us - mi50_done_us;
            }
            if (must_wait) {
                ++waited;
                join_us += ggml_time_us() - join_start;
            }
        } else {
            ++cancelled;
        }
        return weighted.data();
    }

    bool enqueue(const std::array<const ggml_tensor *, 3> & planes, int host_slot, int expert,
            const int32_t * ids, const float * activation, const float * router_weights, unsigned & mask) {
        using get_traits_t = const ggml_type_traits_cpu * (*)(ggml_type);
        static const auto get_traits = []() -> get_traits_t {
            const auto reg = ggml_backend_reg_by_name("CPU");
            return reg ? reinterpret_cast<get_traits_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_single_miss_traits")) : nullptr;
        }();
        if (!get_traits || host_slot < 0) {
            return false;
        }
        work_plane plane_data[3];
        for (int p = 0; p < 3; ++p) {
            const auto * w = planes[p];
            if (!w || w->ne[0] != (p == 2 ? inner : width) || w->ne[1] != (p == 2 ? width : inner) ||
                    host_slot >= w->ne[2] || !w->data || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
                return false;
            }
            const auto * traits = get_traits(w->type);
            if (!traits || !traits->vec_dot) {
                return false;
            }
            const auto * input_traits = get_traits(traits->vec_dot_type);
            if (!input_traits || !input_traits->from_float || ggml_row_size(traits->vec_dot_type, w->ne[0]) > 4096) {
                return false;
            }
            plane_data[p] = {static_cast<const char *>(w->data) + host_slot * w->nb[2], w->nb[1],
                    int(w->ne[0]), int(w->ne[1]), input_traits->from_float, traits->vec_dot};
        }

        mask = 0;
        float route[tokens] = {};
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < ranks; ++r) {
                if (ids[t*ranks + r] == expert) {
                    if (mask & (1u << t)) {
                        return false;
                    }
                    route[t] = router_weights[t*ranks + r];
                    mask |= 1u << t;
                }
            }
        }
        if (!mask) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (pending) {
                return false;
            }
            if (!worker.joinable()) {
                try {
                    worker = std::thread([this] { loop(); });
                } catch (const std::system_error &) {
                    return false;
                }
            }
            std::memcpy(work, plane_data, sizeof(work));
            for (int t = 0; t < tokens; ++t) {
                if (mask & (1u << t)) {
                    std::memcpy(input.data() + t*width, activation + t*width, width*sizeof(float));
                    weights[t] = route[t];
                }
            }
            active_mask = mask;
            queued_us = ggml_time_us();
            pending = queued = true;
            done = false;
        }
        changed.notify_one();
        return true;
    }

private:
    struct work_plane {
        const char * data;
        size_t stride;
        int cols, rows;
        ggml_from_float_t quantize;
        ggml_vec_dot_t dot;
    };
    work_plane work[3];
    std::array<float, width * tokens> input, weighted;
    float weights[tokens];
    unsigned active_mask = 0;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    bool pending = false, queued = false, done = false, stopping = false;
    int64_t queued_us = 0, started_us = 0, completed_us = 0;

    void loop() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return queued || stopping; });
            if (stopping) {
                return;
            }
            queued = false;
            lock.unlock();
            const int64_t start = ggml_time_us();
            compute();
            const int64_t end = ggml_time_us();
            lock.lock();
            started_us = start;
            completed_us = end;
            done = true;
            changed.notify_all();
        }
    }

    void compute() {
        const auto run = [&](int plane, const float * values, float * output) {
            const auto & w = work[plane];
            alignas(64) char quantized[4096];
            w.quantize(values, quantized, w.cols);
            for (int r = 0; r < w.rows; ++r) {
                w.dot(w.cols, output + r, 0, w.data + r*w.stride, 0, quantized, 0, 1);
            }
        };
        for (int t = 0; t < tokens; ++t) {
            if (active_mask & (1u << t)) {
                std::array<float, inner> gate, up;
                run(0, input.data() + t*width, gate.data());
                run(1, input.data() + t*width, up.data());
                for (int j = 0; j < inner; ++j) {
                    gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
                }
                auto * out = weighted.data() + t*width;
                run(2, gate.data(), out);
                for (int j = 0; j < width; ++j) {
                    out[j] *= weights[t];
                }
            }
        }
    }
};
