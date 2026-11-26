#pragma once

#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_iq3_weight_sharing(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv("LLAMA_HIP_GFX906_IQ3_WEIGHT_SHARING_MODE_FILE");
    if (!path || !*path) {
        return;
    }

    static const auto reg = ggml_backend_reg_by_name("ROCm");
    static const auto get_mode = reg ? reinterpret_cast<bool (*)()>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_gfx906_iq3_share_enabled")) : nullptr;
    static const auto set_mode = reg ? reinterpret_cast<void (*)(bool)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_set_gfx906_iq3_share")) : nullptr;
    if (!get_mode || !set_mode) {
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
    const bool enabled = mode == "IQ3_WEIGHT_SHARING=on";
    if (!input.eof() || input.bad() || (!enabled && mode != "IQ3_WEIGHT_SHARING=off")) {
        if (!invalid) {
            LOG_WRN("invalid gfx906 IQ3_XXS weight sharing mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (enabled == get_mode()) {
        return;
    }

    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    set_mode(enabled);
    LOG_INF("gfx906 IQ3_XXS weight sharing mode set: %s\n", enabled ? "on" : "off");
}
