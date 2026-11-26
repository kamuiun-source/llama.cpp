#pragma once

#include "../../ggml/src/ggml-hip-iq3-strata-plan.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

inline void server_refresh_iq3_strata_mode(llama_context * target, llama_context * draft) {
    static const char * path = std::getenv(ggml_hip_iq3_strata_plan::mode_file_env);
    if (!path || !*path) {
        return;
    }
    using set_mode_t = void (*)(bool);
    static const auto set_mode = []() -> set_mode_t {
        const auto reg = ggml_backend_reg_by_name("ROCm");
        return reg ? reinterpret_cast<set_mode_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_set_gfx906_iq3_strata")) : nullptr;
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
    const int requested = mode == "IQ3_STRATA=off" ? 0 : mode == "IQ3_STRATA=on" ? 1 : -1;
    if (!input.eof() || input.bad() || requested < 0) {
        if (!invalid) {
            LOG_WRN("invalid HIP gfx906 IQ3 Strata mode file: %s\n", path);
            invalid = true;
        }
        return;
    }
    invalid = false;
    if (requested == active) {
        return;
    }

    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    set_mode(requested != 0);
    active = requested;
    LOG_INF("HIP gfx906 IQ3 Strata mode applied: %s\n", requested ? "on" : "off");
}
