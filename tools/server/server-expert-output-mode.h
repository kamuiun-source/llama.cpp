#pragma once

#include "ggml-backend.h"
#include "src/llama-ext.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline bool server_refresh_expert_output_modes(llama_context * target, int & scratch_lru_mode) {
    static const char * path = std::getenv("LLAMA_EXPERT_OUTPUT_MODE_FILE");
    if (!path || !*path) {
        return false;
    }

    const auto startup_mode = [](const char * key) {
        const char * value = std::getenv(key);
        return value ? std::string_view(value) : std::string_view();
    };
    static const bool startup_prefetch = startup_mode("LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH") == "1";
    static const auto startup_prefill_mode = startup_mode("LLAMA_EXPERT_HIP_PREFILL");
    static const bool startup_prefill = startup_prefill_mode == "1" || startup_prefill_mode == "serial";
    static bool invalid = false;
    static bool prefetch_unsupported = false;
    static bool prefill_unsupported = false;
    static int ranked_rtx_early_mode = -1;
    static int ranked_rtx_compact_mode = -1;

    const auto apply_scratch_lru = [&](int mode) {
        const bool enabled = mode != 0;
        const bool changed = llama_set_expert_scratch_lru(enabled);
        if (changed || (mode >= 0 && scratch_lru_mode != mode)) {
            LOG_INF("HIP Expert scratch LRU mode applied: SCRATCH_LRU=%s\n", enabled ? "on" : "off");
        }
        scratch_lru_mode = mode;
    };

    const auto apply_ranked_rtx_early = [&](int mode) {
        const bool enabled = mode != 0;
        if (ggml_backend_sched_set_ranked_rtx_early(enabled) || (mode >= 0 && ranked_rtx_early_mode != mode)) {
            LOG_INF("Ranked RTX Expert mode applied: RANKED_RTX_EARLY=%s\n", enabled ? "on" : "off");
        }
        ranked_rtx_early_mode = mode;
    };

    const auto mode_path = std::filesystem::u8path(path);
    std::ifstream input(mode_path, std::ios::binary);
    if (!input.is_open() && ranked_rtx_early_mode == 0) {
        std::error_code error;
        if (!std::filesystem::exists(mode_path, error) && !error) {
            apply_ranked_rtx_early(-1);
        }
    }
    char buffer[128];
    input.read(buffer, sizeof(buffer));
    std::string_view text(buffer, input.gcount());
    bool valid = input.eof() && !input.bad();
    bool want_prefetch = startup_prefetch;
    bool want_prefill = startup_prefill;
    bool seen_prefetch = false;
    bool seen_prefill = false;
    int want_scratch_lru = -1;
    int want_ranked_rtx_early = -1;
    int want_ranked_rtx_compact = -1;
    while (valid && !text.empty()) {
        const auto end = text.find('\n');
        auto line = text.substr(0, end);
        text.remove_prefix(end == std::string_view::npos ? text.size() : end + 1);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            continue;
        }
        if (!seen_prefetch && (line == "NEXT_ROUTER_PREFETCH=on" || line == "NEXT_ROUTER_PREFETCH=off")) {
            want_prefetch = line == "NEXT_ROUTER_PREFETCH=on";
            seen_prefetch = true;
        } else if (!seen_prefill && (line == "HIP_PREFILL=on" || line == "HIP_PREFILL=off")) {
            want_prefill = line == "HIP_PREFILL=on";
            seen_prefill = true;
        } else if (want_scratch_lru < 0 && (line == "SCRATCH_LRU=on" || line == "SCRATCH_LRU=off")) {
            want_scratch_lru = line == "SCRATCH_LRU=on";
        } else if (want_ranked_rtx_early < 0 && (line == "RANKED_RTX_EARLY=on" || line == "RANKED_RTX_EARLY=off")) {
            want_ranked_rtx_early = line == "RANKED_RTX_EARLY=on";
        } else if (want_ranked_rtx_compact < 0 && (line == "RANKED_RTX_COMPACT=on" || line == "RANKED_RTX_COMPACT=off")) {
            want_ranked_rtx_compact = line == "RANKED_RTX_COMPACT=on";
        } else {
            valid = false;
        }
    }
    if (!valid) {
        if (!invalid) {
            LOG_WRN("invalid HIP Expert output mode file: %s\n", path);
        }
        invalid = true;
        return false;
    }
    invalid = false;

    const auto apply = [&](int changed, bool & warned, const char * key) {
        if (changed < 0 && !warned) {
            LOG_WRN("HIP Expert output mode unavailable: %s\n", key);
        }
        warned = changed < 0;
        return changed > 0;
    };
    const bool prefetch_changed = apply(llama_set_expert_hip_next_router_prefetch(target, want_prefetch),
            prefetch_unsupported, "NEXT_ROUTER_PREFETCH");
    const bool prefill_changed = apply(llama_set_expert_hip_prefill(target, want_prefill), prefill_unsupported, "HIP_PREFILL");
    apply_scratch_lru(want_scratch_lru);
    apply_ranked_rtx_early(want_ranked_rtx_early);
    const bool compact = want_ranked_rtx_compact == 1;
    if (ggml_backend_sched_set_ranked_rtx_compact(compact) ||
            (want_ranked_rtx_compact >= 0 && ranked_rtx_compact_mode != want_ranked_rtx_compact)) {
        LOG_INF("Ranked RTX Expert mode applied: RANKED_RTX_COMPACT=%s\n", compact ? "on" : "off");
    }
    ranked_rtx_compact_mode = want_ranked_rtx_compact;
    return prefetch_changed || prefill_changed;
}
