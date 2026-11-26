#pragma once

#include "../../src/llama-next-prefetch-in-place.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_next_prefetch_in_place_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv(llama_next_prefetch_in_place::mode_file_env);
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
    static int active = -1;
    const int requested = mode == "PREFETCH_IN_PLACE=off" ? 0 : mode == "PREFETCH_IN_PLACE=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid HIP MTP3 next Expert prefetch mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (requested != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        llama_set_next_prefetch_in_place(requested != 0);
        active = requested;
        LOG_INF("HIP MTP3 next Expert prefetch in-place mode applied: %s\n", active ? "on" : "off");
    }
}
