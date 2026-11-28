#pragma once

#include "src/llama-ext.h"
#include "../../ggml/src/ggml-backend-mtp.h"
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

    const auto apply_scratch_lru = [&](int mode) {
        const bool enabled = mode != 0;
        const bool changed = llama_set_expert_scratch_lru(enabled);
        if (changed || (mode >= 0 && scratch_lru_mode != mode)) {
            LOG_INF("HIP Expert scratch LRU mode applied: SCRATCH_LRU=%s\n", enabled ? "on" : "off");
        }
        scratch_lru_mode = mode;
    };

    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    char buffer[128];
    input.read(buffer, sizeof(buffer));
    std::string_view text(buffer, input.gcount());
    bool valid = input.eof() && !input.bad();
    bool want_prefetch = startup_prefetch;
    bool want_prefill = startup_prefill;
    bool seen_prefetch = false;
    bool seen_prefill = false;
    int want_scratch_lru = -1;
    int want_mapped_input_q8 = -1;
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
        } else if (want_mapped_input_q8 < 0 && (line == "MAPPED_INPUT_Q8=on" || line == "MAPPED_INPUT_Q8=off")) {
            want_mapped_input_q8 = line == "MAPPED_INPUT_Q8=on";
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
    const bool mapped_input_q8 = want_mapped_input_q8 == 1;
    if (ggml_backend_sched_set_hip_mapped_input_q8(mapped_input_q8)) {
        LOG_INF("HIP mapped input Q8_1 mode applied: MAPPED_INPUT_Q8=%s\n", mapped_input_q8 ? "on" : "off");
    }
    return prefetch_changed || prefill_changed;
}
