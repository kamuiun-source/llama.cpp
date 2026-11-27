#pragma once

#include "../../ggml/src/ggml-backend-copy-api-batch.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_copy_api_batch_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_EXPERT_HIP_MTP3_COPY_API_BATCH_MODE_FILE");
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
    const int requested = mode == "COPY_API_BATCH=off" ? 0 : mode == "COPY_API_BATCH=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid CUDA D2H batch mode file: %s\n", path);
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
    ggml_backend_sched_set_copy_api_batch(requested != 0);
    active = requested;
    LOG_INF("CUDA D2H batch mode applied: %.*s\n", int(mode.size()), mode.data());
}
