#pragma once

#include "../../ggml/src/ggml-hip-mtp3-skip-cpu-zero.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_mtp3_skip_cpu_zero_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv(ggml_hip_mtp3_skip_cpu_zero::mode_file_env);
    if (!path || !*path) {
        return;
    }
    using set_mode_t = void (*)(bool);
    static const auto set_mode = []() -> set_mode_t {
        const auto reg = ggml_backend_reg_by_name("ROCm");
        return reg ? reinterpret_cast<set_mode_t>(
                ggml_backend_reg_get_proc_address(reg, ggml_hip_mtp3_skip_cpu_zero::setter_proc)) : nullptr;
    }();
    if (!set_mode) {
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
    const int requested = mode == "SKIP_CPU_ZERO=off" ? 0 : mode == "SKIP_CPU_ZERO=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid HIP MTP3 CPU zero-slot skip mode file: %s\n", path);
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
        set_mode(requested != 0);
        active = requested;
        LOG_INF("HIP MTP3 CPU zero-slot skip mode applied: %s\n", active ? "on" : "off");
    }
}
