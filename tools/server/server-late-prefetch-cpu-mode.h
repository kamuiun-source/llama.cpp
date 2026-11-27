#pragma once

#include "../../src/llama-late-prefetch-cpu.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_late_prefetch_cpu_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_LATE_PREFETCH_CPU_MODE_FILE");
    if (!path || !*path) {
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
    static int active = 0;
    const int requested = mode == "LATE_PREFETCH_CPU=off" ? 0 : mode == "LATE_PREFETCH_CPU=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid HIP late prefetch CPU mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (requested == active) {
        return;
    }
    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    llama_set_late_prefetch_cpu(requested != 0);
    active = requested;
    LOG_INF("HIP late prefetch CPU mode applied: %.*s\n", int(mode.size()), mode.data());
}
