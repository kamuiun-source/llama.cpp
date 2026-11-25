#pragma once

#include "../../ggml/src/ggml-backend-mtp.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_single_miss_cpu_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_SINGLE_MISS_CPU_MODE_FILE");
    if (!path || !*path) {
        return;
    }
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    char buffer[32];
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
    const int requested = mode == "SINGLE_MISS_CPU=off" ? 0 : mode == "SINGLE_MISS_CPU=on" ? 1 :
            mode == "SINGLE_MISS_CPU=2" ? 2 : mode == "SINGLE_MISS_CPU=4" ? 4 :
            mode == "SINGLE_MISS_CPU=2-shared" ? 6 : mode == "SINGLE_MISS_CPU=4-shared" ? 8 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid HIP single CPU miss mode file: %s\n", path);
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
    ggml_backend_sched_set_hip_single_miss_cpu(requested);
    active = requested;
    LOG_INF("HIP single CPU miss mode applied: %.*s\n", int(mode.size()), mode.data());
}
