#pragma once

#include "../../src/llama-mtp-catchup.h"
#include "llama.h"
#include "log.h"

#include <filesystem>
#include <fstream>
#include <string_view>

inline bool server_mtp_catchup_first_draft(bool refresh = false, llama_context * target = nullptr, llama_context * draft = nullptr) {
    static const char * path = llama_mtp_catchup_mode_file();
    static bool active = false;
    if (!path || !refresh) {
        return active;
    }
    static bool invalid = false;
    static bool reported = false;
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
    if (!file.eof() || file.bad() || (mode != "CATCHUP_FIRST_DRAFT=off" && mode != "CATCHUP_FIRST_DRAFT=on")) {
        if (!invalid) {
            LOG_WRN("invalid MTP catch-up mode file: %s\n", path);
            invalid = true;
        }
        return active;
    }
    invalid = false;
    const bool requested = mode == "CATCHUP_FIRST_DRAFT=on";
    if (requested != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        active = requested;
        reported = false;
    }
    if (!reported) {
        LOG_INF("MTP catch-up mode applied: %.*s\n", int(mode.size()), mode.data());
        reported = true;
    }
    return active;
}
