#pragma once

#include "ggml-cpu.h"
#include "llama-single-miss-selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>

inline bool llama_single_miss_cpu_configured() {
    const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_MISS_CPU_MODE_FILE");
    return path && *path;
}

struct llama_single_miss_cpu {
    static constexpr int width = 2560, inner = 640, tokens = 3, ranks = 10;
    struct stats {
        size_t eligible = 0, applied = 0, positions = 0, avoided_bytes = 0, quant_calls = 0, experts = 0;
        size_t planned_gpu_uploads = 0, planned_gpu_bytes = 0;
        size_t joined = 0, waited = 0, cancelled = 0, outstanding_at_mi50 = 0;
        int64_t compute_us = 0, queue_us = 0, join_us = 0, remaining_after_mi50_us = 0, merge_us = 0;
    };
    static constexpr int max_jobs = llama_single_miss_budget::max_jobs;
    const llama_single_miss_budget budget;
    std::array<stats, max_jobs + 1> jobs;

    explicit llama_single_miss_cpu(int physical_cores) : budget{physical_cores} {}

    ~llama_single_miss_cpu() {
        drain();
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        for (size_t i = 0; i < workers.size(); ++i) {
            wake[i].notify_one();
            if (workers[i].joinable()) {
                workers[i].join();
            }
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
        auto & totals = jobs[n_teams];
        for (int i = 0; i < n_teams; ++i) {
            const auto & team = *teams[i];
            const size_t quantizations = team.share_gate_up ? 2 : 3;
            for (int t = 0; t < tokens; ++t) {
                if (team.mask & (1u << t)) {
                    totals.quant_calls += quantizations;
                }
            }
        }
        totals.compute_us += completed_us - started_us;
        totals.queue_us += started_us - queued_us;
        if (mi50_done_us) {
            ++totals.joined;
            if (completed_us > mi50_done_us) {
                ++totals.outstanding_at_mi50;
                totals.remaining_after_mi50_us += completed_us - mi50_done_us;
            }
            if (must_wait) {
                ++totals.waited;
                totals.join_us += ggml_time_us() - join_start;
            }
        } else {
            ++totals.cancelled;
        }
        if (mi50_done_us && n_teams > 1) {
            const int64_t merge_start = ggml_time_us();
            auto & first = *teams[0];
            unsigned merged = first.mask;
            for (int i = 1; i < n_teams; ++i) {
                const auto & next = *teams[i];
                for (int t = 0; t < tokens; ++t) {
                    if (next.mask & (1u << t)) {
                        auto * dst = first.weighted.data() + t*width;
                        const auto * rhs = next.weighted.data() + t*width;
                        if (merged & (1u << t)) {
                            for (int j = 0; j < width; ++j) {
                                dst[j] += rhs[j];
                            }
                        } else {
                            std::memcpy(dst, rhs, width*sizeof(float));
                        }
                    }
                }
                merged |= next.mask;
            }
            totals.merge_us += ggml_time_us() - merge_start;
        }
        return teams[0]->weighted.data();
    }

    bool enqueue(const std::array<const ggml_tensor *, 3> & planes, const int32_t * host_slots, const int32_t * experts,
            int n_experts, const int32_t * ids, const float * activation, const float * router_weights, unsigned & mask) {
        if (n_experts < 1 || n_experts > budget.jobs()) {
            return false;
        }
        for (int i = 0; i < n_experts; ++i) {
            for (int j = 0; j < i; ++j) {
                if (host_slots[i] == host_slots[j] || experts[i] == experts[j]) {
                    return false;
                }
            }
        }
        using get_traits_t = const ggml_type_traits_cpu * (*)(ggml_type);
        static const auto get_traits = []() -> get_traits_t {
            const auto reg = ggml_backend_reg_by_name("CPU");
            return reg ? reinterpret_cast<get_traits_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_single_miss_traits")) : nullptr;
        }();
        const int threads = budget.workers(n_experts);
        if (!get_traits) {
            return false;
        }
        work_plane plane_data[3];
        for (int p = 0; p < 3; ++p) {
            const auto * w = planes[p];
            if (!w || w->ne[0] != (p == 2 ? inner : width) || w->ne[1] != (p == 2 ? width : inner) ||
                    !w->data || !w->buffer || !ggml_backend_buffer_is_host(w->buffer)) {
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
            for (int i = 0; i < n_experts; ++i) {
                if (host_slots[i] < 0 || host_slots[i] >= w->ne[2]) {
                    return false;
                }
            }
            plane_data[p] = {static_cast<const char *>(w->data), w->nb[1],
                    int(w->ne[0]), int(w->ne[1]), input_traits->from_float, traits->vec_dot,
                    traits->vec_dot_type, ggml_row_size(traits->vec_dot_type, w->ne[0])};
        }

        unsigned masks[max_jobs] = {};
        unsigned selected_positions = 0;
        float routes[max_jobs][tokens] = {};
        for (int i = 0; i < n_experts; ++i) {
            for (int t = 0; t < tokens; ++t) {
                for (int r = 0; r < ranks; ++r) {
                    if (ids[t*ranks + r] == experts[i]) {
                        if (masks[i] & (1u << t)) {
                            return false;
                        }
                        routes[i][t] = router_weights[t*ranks + r];
                        masks[i] |= 1u << t;
                    }
                }
            }
            if (!masks[i]) {
                return false;
            }
            selected_positions |= masks[i];
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (pending) {
                return false;
            }
            for (int i = 0; i < n_experts; ++i) {
                if (!teams[i]) {
                    teams[i].reset(new (std::nothrow) team_state);
                    if (!teams[i]) {
                        return false;
                    }
                }
            }
            for (int i = 0; i < threads; ++i) {
                if (!workers[i].joinable()) {
                    try {
                        workers[i] = std::thread([this, i] { loop(i); });
                    } catch (const std::system_error &) {
                        return false;
                    }
                }
            }
            for (int i = 0; i < n_experts; ++i) {
                auto & team = *teams[i];
                for (int p = 0; p < 3; ++p) {
                    team.work[p] = plane_data[p];
                    team.work[p].data += host_slots[i] * planes[p]->nb[2];
                }
                std::memcpy(team.weights, routes[i], sizeof(team.weights));
                team.mask = masks[i];
                team.share_gate_up = team.work[0].input_type == team.work[1].input_type &&
                        team.work[0].quantize == team.work[1].quantize;
                team.inputs_ready = team.down_ready = false;
                team.gated_workers = 0;
            }
            mask = selected_positions;
            for (int t = 0; t < tokens; ++t) {
                if (mask & (1u << t)) {
                    std::memcpy(input.data() + t*width, activation + t*width, width*sizeof(float));
                }
            }
            queued_us = ggml_time_us();
            n_workers = threads;
            n_teams = n_experts;
            joined_workers = 0;
            pending = true;
            done = false;
            started_us = 0;
            completed_us = 0;
            ++generation;
        }
        for (int i = 0; i < threads; ++i) {
            wake[i].notify_one();
        }
        return true;
    }

private:
    struct work_plane {
        const char * data;
        size_t stride;
        int cols, rows;
        ggml_from_float_t quantize;
        ggml_vec_dot_t dot;
        ggml_type input_type;
        size_t quant_bytes;
    };
    struct alignas(64) quant_row {
        char data[4096];
    };
    struct team_state {
        work_plane work[3];
        alignas(64) std::array<float, width * tokens> weighted;
        alignas(64) std::array<std::array<float, inner>, tokens> gated;
        quant_row shared_input[3][tokens];
        float weights[tokens];
        unsigned mask = 0;
        int gated_workers = 0;
        bool share_gate_up = false, inputs_ready = false, down_ready = false;
        std::condition_variable gates_ready;
    };
    std::array<std::unique_ptr<team_state>, max_jobs> teams;
    std::array<float, width * tokens> input;
    std::mutex mutex;
    std::condition_variable changed;
    std::array<std::condition_variable, 2 * max_jobs> wake;
    std::array<std::thread, 2 * max_jobs> workers;
    bool pending = false, done = false, stopping = false;
    uint64_t generation = 0;
    int n_workers = 0, n_teams = 0, joined_workers = 0;
    int64_t queued_us = 0, started_us = 0, completed_us = 0;

    void loop(int index) {
        uint64_t previous = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            wake[index].wait(lock, [&] {
                return stopping || (pending && index < n_workers && generation != previous);
            });
            if (stopping) {
                return;
            }
            previous = generation;
            if (started_us == 0) {
                started_us = ggml_time_us();
            }
            lock.unlock();
            const int team_size = n_workers / n_teams;
            compute_parallel(*teams[index / team_size], index % team_size, team_size);
            const int64_t end = ggml_time_us();
            lock.lock();
            completed_us = std::max(completed_us, end);
            if (++joined_workers == n_workers) {
                done = true;
                changed.notify_one();
            }
        }
    }

    void quantize_shared(team_state & team, int plane, int token, const float * values) {
        const auto & w = team.work[plane];
        auto * out = team.shared_input[plane][token].data;
        // The Q8_K reference quantizer leaves bsums untouched for a zero block.
        std::memset(out, 0, w.quant_bytes);
        w.quantize(values, out, w.cols);
    }

    void compute_parallel(team_state & team, int index, int team_size) {
        const auto run = [&](int plane, int token, float * output) {
            const auto & w = team.work[plane];
            const auto * q = team.shared_input[plane == 1 && team.share_gate_up ? 0 : plane][token].data;
            const int begin = index * w.rows / team_size;
            const int end = (index + 1) * w.rows / team_size;
            for (int r = begin; r < end; ++r) {
                w.dot(w.cols, output + r, 0, w.data + r*w.stride, 0, q, 0, 1);
            }
        };

        if (index == 0) {
            for (int t = 0; t < tokens; ++t) {
                if (team.mask & (1u << t)) {
                    const auto * values = input.data() + t*width;
                    quantize_shared(team, 0, t, values);
                    if (!team.share_gate_up) {
                        quantize_shared(team, 1, t, values);
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            team.inputs_ready = true;
            team.gates_ready.notify_all();
        } else {
            std::unique_lock<std::mutex> lock(mutex);
            team.gates_ready.wait(lock, [&] { return team.inputs_ready; });
        }
        const int gate_begin = index * inner / team_size;
        const int gate_end = (index + 1) * inner / team_size;
        for (int t = 0; t < tokens; ++t) {
            if (team.mask & (1u << t)) {
                std::array<float, inner> up;
                auto & gate = team.gated[t];
                run(0, t, gate.data());
                run(1, t, up.data());
                for (int j = gate_begin; j < gate_end; ++j) {
                    gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
                }
            }
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (++team.gated_workers == team_size) {
                lock.unlock();
                for (int t = 0; t < tokens; ++t) {
                    if (team.mask & (1u << t)) {
                        quantize_shared(team, 2, t, team.gated[t].data());
                    }
                }
                lock.lock();
                team.down_ready = true;
                team.gates_ready.notify_all();
            } else {
                team.gates_ready.wait(lock, [&] { return team.down_ready; });
            }
        }
        const int down_begin = index * width / team_size;
        const int down_end = (index + 1) * width / team_size;
        for (int t = 0; t < tokens; ++t) {
            if (team.mask & (1u << t)) {
                auto * out = team.weighted.data() + t*width;
                run(2, t, out);
                for (int j = down_begin; j < down_end; ++j) {
                    out[j] *= team.weights[t];
                }
            }
        }
    }

};
