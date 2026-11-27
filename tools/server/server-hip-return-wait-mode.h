#pragma once

#include "../../ggml/src/ggml-backend-mtp.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_hip_return_wait_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_HIP_MTP3_RETURN_DST_WAIT_MODE_FILE");
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
    static bool active = false;
    const bool enabled = mode == "RETURN_DST_WAIT_SKIP=on";
    if (!input.eof() || input.bad() || (!enabled && mode != "RETURN_DST_WAIT_SKIP=off")) {
        if (!invalid) {
            LOG_WRN("invalid HIP return destination wait mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (enabled != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        ggml_backend_sched_set_hip_return_dst_wait_skip(enabled);
        active = enabled;
        LOG_INF("HIP return destination wait mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}
