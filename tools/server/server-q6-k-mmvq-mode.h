#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_q6_k_mmvq_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_CUDA_Q6_K_MMVQ_MODE_FILE");
    if (!path || !*path) {
        return;
    }

    static const auto reg = ggml_backend_reg_by_name("CUDA");
    static const auto get_mode = reg ? reinterpret_cast<int (*)()>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_q6_k_mmvq_mode")) : nullptr;
    static const auto set_mode = reg ? reinterpret_cast<void (*)(int)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_q6_k_mmvq_mode")) : nullptr;
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

    static bool invalid = false;
    const int mode = value == "Q6_K=off" ? 0 : value == "Q6_K=signed" ? 1 : value == "Q6_K=reuse" ? 2 : -1;
    if (!input.eof() || input.bad() || mode < 0) {
        if (!invalid) {
            LOG_WRN("invalid CUDA Q6_K MMVQ mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (mode == get_mode()) {
        return;
    }

    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    set_mode(mode);
    LOG_INF("CUDA Q6_K MMVQ mode applied: Q6_K=%.*s\n", int(value.size() - 5), value.data() + 5);
}
