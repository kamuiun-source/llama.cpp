#pragma once

#include "ggml-cpu.h"

#include <algorithm>
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
    struct stats {
        size_t eligible = 0, applied = 0, positions = 0, avoided_bytes = 0, quant_calls = 0, experts = 0;
        size_t planned_gpu_uploads = 0, planned_gpu_bytes = 0;
        size_t joined = 0, waited = 0, cancelled = 0, outstanding_at_mi50 = 0;
        int64_t compute_us = 0, queue_us = 0, join_us = 0, remaining_after_mi50_us = 0, pair_merge_us = 0;
    };
    static constexpr int pair_mode = 9;
    static constexpr int pair_stats = 10;
    static constexpr int hybrid_mode = 11;
    static constexpr int hybrid_pair_stats = 12;
    static constexpr int hybrid_multi_stats = 13;
    stats modes[14];

    static int stats_mode(int mode, int actual_misses) {
        if (mode == hybrid_mode) {
            return actual_misses == 1 ? hybrid_mode : actual_misses == 2 ? hybrid_pair_stats : hybrid_multi_stats;
        }
        return mode == pair_mode && actual_misses == 2 ? pair_stats : mode;
    }

    static int thread_count(int mode) {
        return mode >= pair_mode ? 4 : mode > 4 ? mode - 4 : mode;
    }

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
        auto & totals = modes[active_mode];
        for (int i = 0; i < n_teams; ++i) {
            const auto & team = teams[i];
            const size_t quantizations = share_quantized ? (team.share_gate_up ? 2 : 3) : 3 * n_workers;
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
        if (mi50_done_us && n_teams == 2) {
            const int64_t merge_start = ggml_time_us();
            auto & first = teams[0];
            const auto & second = teams[1];
            for (int t = 0; t < tokens; ++t) {
                if (second.mask & (1u << t)) {
                    auto * dst = first.weighted.data() + t*width;
                    const auto * rhs = second.weighted.data() + t*width;
                    if (first.mask & (1u << t)) {
                        for (int j = 0; j < width; ++j) {
                            dst[j] += rhs[j];
                        }
                    } else {
                        std::memcpy(dst, rhs, width*sizeof(float));
                    }
                }
            }
            totals.pair_merge_us += ggml_time_us() - merge_start;
        }
        return teams[0].weighted.data();
    }

    bool enqueue(int mode, const std::array<const ggml_tensor *, 3> & planes, int host_slot, int expert,
            const int32_t * ids, const float * activation, const float * router_weights, unsigned & mask) {
        return enqueue(mode, planes, &host_slot, &expert, 1, ids, activation, router_weights, mask);
    }

    bool enqueue(int mode, const std::array<const ggml_tensor *, 3> & planes, const int32_t * host_slots, const int32_t * experts,
            int n_experts, const int32_t * ids, const float * activation, const float * router_weights, unsigned & mask,
            int actual_misses = 0) {
        if (actual_misses == 0) {
            actual_misses = n_experts;
        }
        if (n_experts < 1 || n_experts > 2 || actual_misses < n_experts ||
                (actual_misses > n_experts && (mode != hybrid_mode || n_experts != 2)) ||
                (n_experts == 2 && ((mode != pair_mode && mode != hybrid_mode) ||
                    host_slots[0] == host_slots[1] || experts[0] == experts[1]))) {
            return false;
        }
        using get_traits_t = const ggml_type_traits_cpu * (*)(ggml_type);
        static const auto get_traits = []() -> get_traits_t {
            const auto reg = ggml_backend_reg_by_name("CPU");
            return reg ? reinterpret_cast<get_traits_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_get_single_miss_traits")) : nullptr;
        }();
        const int threads = thread_count(mode);
        if (!get_traits || (mode != 1 && mode != 2 && mode != 4 && mode != 6 && mode != 8 && mode != pair_mode && mode != hybrid_mode)) {
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

        unsigned masks[2] = {};
        float routes[2][tokens] = {};
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
        }
        mask = masks[0] | masks[1];
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (pending) {
                return false;
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
                auto & team = teams[i];
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
            for (int t = 0; t < tokens; ++t) {
                if (mask & (1u << t)) {
                    std::memcpy(input.data() + t*width, activation + t*width, width*sizeof(float));
                }
            }
            queued_us = ggml_time_us();
            n_workers = threads;
            n_teams = n_experts;
            active_mode = stats_mode(mode, actual_misses);
            share_quantized = mode > 4;
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
    team_state teams[2];
    std::array<float, width * tokens> input;
    std::mutex mutex;
    std::condition_variable changed;
    std::array<std::condition_variable, 4> wake;
    std::array<std::thread, 4> workers;
    bool share_quantized = false;
    int active_mode = 0;
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
            if (n_workers == 1) {
                compute(teams[0]);
            } else {
                const int team_size = n_workers / n_teams;
                compute_parallel(teams[index / team_size], index % team_size, team_size);
            }
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
        const auto run = [&](int plane, int token, const float * values, float * output) {
            const auto & w = team.work[plane];
            alignas(64) char quantized[4096];
            const char * q = quantized;
            if (share_quantized) {
                q = team.shared_input[plane == 1 && team.share_gate_up ? 0 : plane][token].data;
            } else {
                w.quantize(values, quantized, w.cols);
            }
            const int begin = index * w.rows / team_size;
            const int end = (index + 1) * w.rows / team_size;
            for (int r = begin; r < end; ++r) {
                w.dot(w.cols, output + r, 0, w.data + r*w.stride, 0, q, 0, 1);
            }
        };

        if (share_quantized) {
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
        }
        const int gate_begin = index * inner / team_size;
        const int gate_end = (index + 1) * inner / team_size;
        for (int t = 0; t < tokens; ++t) {
            if (team.mask & (1u << t)) {
                std::array<float, inner> up;
                auto & gate = team.gated[t];
                run(0, t, input.data() + t*width, gate.data());
                run(1, t, input.data() + t*width, up.data());
                for (int j = gate_begin; j < gate_end; ++j) {
                    gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
                }
            }
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (++team.gated_workers == team_size) {
                if (share_quantized) {
                    lock.unlock();
                    for (int t = 0; t < tokens; ++t) {
                        if (team.mask & (1u << t)) {
                            quantize_shared(team, 2, t, team.gated[t].data());
                        }
                    }
                    lock.lock();
                }
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
                run(2, t, team.gated[t].data(), out);
                for (int j = down_begin; j < down_end; ++j) {
                    out[j] *= team.weights[t];
                }
            }
        }
    }

    void compute(team_state & team) {
        const auto run = [&](int plane, const float * values, float * output) {
            const auto & w = team.work[plane];
            alignas(64) char quantized[4096];
            w.quantize(values, quantized, w.cols);
            for (int r = 0; r < w.rows; ++r) {
                w.dot(w.cols, output + r, 0, w.data + r*w.stride, 0, quantized, 0, 1);
            }
        };
        for (int t = 0; t < tokens; ++t) {
            if (team.mask & (1u << t)) {
                std::array<float, inner> gate, up;
                run(0, input.data() + t*width, gate.data());
                run(1, input.data() + t*width, up.data());
                for (int j = 0; j < inner; ++j) {
                    gate[j] = (gate[j] / (1.0f + std::exp(-gate[j]))) * up[j];
                }
                auto * out = team.weighted.data() + t*width;
                run(2, gate.data(), out);
                for (int j = 0; j < width; ++j) {
                    out[j] *= team.weights[t];
                }
            }
        }
    }
};
