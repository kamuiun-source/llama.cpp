#pragma once

#include "ggml-backend.h"
#include "ggml/src/ggml-gpu-output-modes.h"
#include "llama.h"
#include "log.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

inline bool server_parse_gpu_output_modes(std::string_view input, uint32_t & modes) {
    namespace flags = ggml_gpu_output_modes;
    uint32_t present = 0;
    uint32_t enabled = 0;
    while (!input.empty()) {
        const auto newline = input.find('\n');
        auto line = input.substr(0, newline);
        input = newline == std::string_view::npos ? std::string_view{} : input.substr(newline + 1);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string_view::npos) {
            return false;
        }
        const auto key = line.substr(0, eq);
        const auto value = line.substr(eq + 1);
        unsigned index = 0;
        while (index < flags::count && flags::names[index] != key) {
            ++index;
        }
        if (index == flags::count || (present & flags::bit(index))) {
            return false;
        }
        present |= flags::bit(index);
        if (value == "on" || value == "1") {
            enabled |= flags::bit(index);
        } else if (value != "off" && value != "0") {
            return false;
        }
    }
    modes = flags::pack(present, enabled);
    return true;
}

inline bool server_refresh_gpu_output_modes(llama_context * target, llama_context * draft) {
    namespace flags = ggml_gpu_output_modes;
    static const char * path = std::getenv(flags::mode_file_env);
    if (!path || !*path) {
        return false;
    }
    using setter = void (*)(uint32_t);
    static const auto resolve = [](const char * name) -> setter {
        auto reg = ggml_backend_reg_by_name(name);
        return reg ? reinterpret_cast<setter>(ggml_backend_reg_get_proc_address(reg, flags::setter_proc)) : nullptr;
    };
    static const setter hip = resolve("ROCm");
    static const setter cuda = resolve("CUDA");
    if (!hip && !cuda) {
        return false;
    }

    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    char buffer[1024];
    input.read(buffer, sizeof(buffer));
    uint32_t modes = 0;
    static bool invalid = false;
    if (!input.eof() || input.bad() || !server_parse_gpu_output_modes({buffer, size_t(input.gcount())}, modes)) {
        if (!invalid) {
            LOG_WRN("invalid GPU output mode file: %s\n", path);
            invalid = true;
        }
        return false;
    }
    invalid = false;
    const uint32_t bits = (hip ? flags::hip_bits : 0) | (cuda ? flags::cuda_bits : 0);
    modes &= flags::pack(bits, bits);
    static uint32_t active = 0;
    if (modes == active) {
        return false;
    }

    llama_synchronize(target);
    if (draft && draft != target) {
        llama_synchronize(draft);
    }
    if (hip) {
        hip(modes);
    }
    if (cuda) {
        cuda(modes);
    }
    active = modes;
    for (unsigned i = 0; i < flags::count; ++i) {
        const auto flag = flags::bit(i);
        if (bits & flag) {
            const auto name = flags::names[i];
            LOG_INF("GPU output mode applied: %.*s=%s\n", int(name.size()), name.data(),
                    modes & (flag << flags::count) ? (modes & flag ? "on" : "off") : "default");
        }
    }
    return true;
}
