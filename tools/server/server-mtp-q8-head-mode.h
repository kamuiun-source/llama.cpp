#pragma once

#include "../../src/llama-ext.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_mtp_q8_head_mode(llama_context * draft) {
    static const char * path = std::getenv("LLAMA_MTP_Q8_HEAD_MODE_FILE");
    if (!path || !*path || !draft) {
        return;
    }
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    char data[64];
    file.read(data, sizeof(data));
    std::string_view mode(data, file.gcount());
    if (!mode.empty() && mode.back() == '\n') {
        mode.remove_suffix(1);
    }
    if (!mode.empty() && mode.back() == '\r') {
        mode.remove_suffix(1);
    }
    static bool invalid = false;
    if (!file.eof() || file.bad() || (mode != "MTP_Q8_HEAD=off" && mode != "MTP_Q8_HEAD=pair")) {
        if (!invalid) {
            LOG_WRN("invalid MTP Q8 head mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;

    static llama_context * previous = nullptr;
    static bool active = false;
    const bool requested = mode == "MTP_Q8_HEAD=pair";
    if (draft != previous || requested != active) {
        llama_set_mtp_q8_head_pair_rows(draft, requested);
        active = requested;
        previous = draft;
        LOG_INF("MTP Q8 head mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}
