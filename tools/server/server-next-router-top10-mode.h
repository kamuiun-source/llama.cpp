#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline bool server_refresh_next_router_top10_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_CUDA_NEXT_ROUTER_TOP10_MODE_FILE");
    if (!path || !*path) {
        return false;
    }

    using set_mode_t = void (*)(bool);
    static const auto set_mode = []() -> set_mode_t {
        const auto reg = ggml_backend_reg_by_name("CUDA");
        return reg ? reinterpret_cast<set_mode_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_next_router_top10")) : nullptr;
    }();
    if (!set_mode) {
        return false;
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
    const int requested = mode == "NEXT_ROUTER_TOP10=off" ? 0 : mode == "NEXT_ROUTER_TOP10=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid CUDA next Router Top-10 mode file: %s\n", path);
            invalid = true;
        }
        return active == 1;
    }
    invalid = false;
    if (requested != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        set_mode(requested != 0);
        active = requested;
        LOG_INF("CUDA next Router Top-10 mode applied: %s\n", active ? "on" : "off");
    }
    return active == 1;
}
