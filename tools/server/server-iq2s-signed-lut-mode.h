#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_iq2s_signed_lut_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_HIP_GFX906_IQ2_S_LUT_MODE_FILE");
    if (!path || !*path) {
        return;
    }
    using setter = void (*)(bool);
    static const auto set_mode = []() -> setter {
        const auto reg = ggml_backend_reg_by_name("ROCm");
        return reg ? reinterpret_cast<setter>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_set_gfx906_iq2s_signed_lut")) : nullptr;
    }();
    if (!set_mode) {
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
    if (!file.eof() || file.bad() || (mode != "IQ2_S_LUT=off" && mode != "IQ2_S_LUT=on")) {
        if (!invalid) {
            LOG_WRN("invalid IQ2_S signed LUT mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;

    static int active = -1;
    const bool requested = mode == "IQ2_S_LUT=on";
    if (requested != active) {
        llama_synchronize(target);
        if (draft && draft != target) {
            llama_synchronize(draft);
        }
        set_mode(requested);
        active = requested;
        LOG_INF("HIP IQ2_S signed LUT mode applied: %.*s\n", int(mode.size()), mode.data());
    }
}
