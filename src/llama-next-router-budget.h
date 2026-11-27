#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_next_router_budget {
    // Fixed3 MI50 trace calibration. Leave headroom below the observed H2D throughput.
    static constexpr int64_t max_window_us = 240;
    static constexpr size_t bytes_per_us = 12000;

    struct totals {
        size_t observed_layers = 0;
        uint64_t window_sample_us = 0;
        size_t candidates = 0;
        size_t budget_checks = 0;
        size_t budget_blocked = 0;
        size_t budget_blocked_bytes = 0;
        size_t budget_blocked_selected = 0;
        size_t unarmed_layers = 0;
        size_t h2d_tuples = 0;
        size_t h2d_bytes = 0;
        size_t ready_hits = 0;
        size_t ready_hit_bytes = 0;
        size_t late_hits = 0;
        size_t late_hit_bytes = 0;
        size_t demand_tuples = 0;
        size_t demand_bytes = 0;
        uint64_t demand_sync_us = 0;
    };

    bool enabled = false;
    bool telemetry = false;
    std::vector<int64_t> window_us;
    totals stats[2];

    bool active() const { return enabled || telemetry; }
    totals & current() { return stats[enabled]; }

    void arm(int32_t layer, int64_t now) {
        target = layer;
        start_us = now;
    }

    bool observe(int32_t layer, int64_t now) {
        if (target != layer) {
            return false;
        }
        // The host interval includes other streams and host work. Only budget the calibrated MI50 share.
        const auto sample = std::clamp((now - start_us) * 2 / 5, int64_t(1), max_window_us);
        auto & estimate = window_us[layer];
        estimate = estimate ? (3 * estimate + sample) / 4 : sample;
        ++current().observed_layers;
        current().window_sample_us += sample;
        reset();
        return true;
    }

    size_t available_bytes(int32_t layer, int64_t now) const {
        if (target != layer) {
            return 0;
        }
        const auto remaining = std::max(int64_t(0), window_us[layer] - (now - start_us));
        return remaining * bytes_per_us;
    }

    void reset() { target = -1; }

private:
    int32_t target = -1;
    int64_t start_us = 0;
};
