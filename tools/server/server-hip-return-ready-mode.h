#pragma once

#include "../../ggml/src/ggml-backend-mtp.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_hip_return_ready_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("GGML_SCHED_HIP_MTP3_RETURN_READY_MODE_FILE");
    if (!path || !*path) {
        return;
    }
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    char buffer[64];
    file.read(buffer, sizeof(buffer));
    std::string_view mode(buffer, file.gcount());
    if (!mode.empty() && mode.back() == '\n') {
        mode.remove_suffix(1);
    }
    if (!mode.empty() && mode.back() == '\r') {
        mode.remove_suffix(1);
    }
    static bool invalid = false;
    static bool active = false;
    static bool reported = false;
    if (!file.eof() || file.bad() || (mode != "RETURN_READY=on" && mode != "RETURN_READY=off")) {
        if (!invalid) {
            LOG_WRN("invalid HIP return-ready mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    const bool requested = mode == "RETURN_READY=on";
    if (requested != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        ggml_backend_sched_set_hip_return_ready(requested);
        active = requested;
        reported = false;
    }
    if (!reported) {
        reported = true;
        LOG_INF("HIP return-ready mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}
