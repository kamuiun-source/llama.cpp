#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string_view>

inline void server_refresh_cuda_mmvq_mode(const char * path, const char * name, const char * get_proc, const char * set_proc,
        std::initializer_list<std::string_view> modes, bool & invalid, llama_context * target, llama_context * draft) {
    if (!path || !*path) {
        return;
    }

    static const auto reg = ggml_backend_reg_by_name("CUDA");
    const auto get_mode = reg ? reinterpret_cast<int (*)()>(ggml_backend_reg_get_proc_address(reg, get_proc)) : nullptr;
    const auto set_mode = reg ? reinterpret_cast<void (*)(int)>(ggml_backend_reg_get_proc_address(reg, set_proc)) : nullptr;
    if (!get_mode || !set_mode) {
        return;
    }

    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    char buffer[32];
    input.read(buffer, sizeof(buffer));
    std::string_view value(buffer, input.gcount());
    if (!value.empty() && value.back() == '\n') {
        value.remove_suffix(1);
    }
    if (!value.empty() && value.back() == '\r') {
        value.remove_suffix(1);
    }

    const auto mode = std::find(modes.begin(), modes.end(), value);
    if (!input.eof() || input.bad() || mode == modes.end()) {
        if (!invalid) {
            LOG_WRN("invalid CUDA %s mode file: %s\n", name, path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    const int requested = mode - modes.begin();
    if (requested == get_mode()) {
        return;
    }

    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    set_mode(requested);
    LOG_INF("CUDA %s mode applied: %.*s\n", name, int(value.size()), value.data());
}

inline void server_refresh_cuda_mmvq_modes(llama_context * target, llama_context * draft) {
    static const char * q6_k_path = std::getenv("LLAMA_CUDA_Q6_K_MMVQ_MODE_FILE");
    static const char * hc_up_path = std::getenv("LLAMA_CUDA_Q8_0_HC_UP_MODE_FILE");
    static bool q6_k_invalid = false;
    static bool hc_up_invalid = false;
    server_refresh_cuda_mmvq_mode(q6_k_path, "Q6_K MMVQ",
            "ggml_backend_cuda_get_q6_k_mmvq_mode", "ggml_backend_cuda_set_q6_k_mmvq_mode",
            {"Q6_K=off", "Q6_K=signed", "Q6_K=reuse"}, q6_k_invalid, target, draft);
    server_refresh_cuda_mmvq_mode(hc_up_path, "Q8_0 HC up",
            "ggml_backend_cuda_get_q8_0_hc_up_mode", "ggml_backend_cuda_set_q8_0_hc_up_mode",
            {"HC_UP=off", "HC_UP=warp", "HC_UP=packed", "HC_UP=prep"}, hc_up_invalid, target, draft);
}
