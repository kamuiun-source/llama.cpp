#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace llama_expert_profile {

constexpr int n_layers = 48;
constexpr int n_experts = 512;
constexpr int n_resident = 300;
using profile = std::array<std::array<bool, n_experts>, n_layers>;

inline profile read_profile(const char * path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    profile result{};
    for (int layer = 0; layer < n_layers; ++layer) {
        int id;
        if (!(input >> id) || id != layer) {
            throw std::runtime_error("expected profile layer " + std::to_string(layer));
        }
        for (int i = 0; i < n_resident; ++i) {
            int expert;
            if (!(input >> expert) || expert < 0 || expert >= n_experts || result[layer][expert]) {
                throw std::runtime_error("invalid or duplicate expert in layer " + std::to_string(layer));
            }
            result[layer][expert] = true;
        }
    }
    input >> std::ws;
    if (!input.eof() || input.bad()) {
        throw std::runtime_error("unexpected trailing profile data");
    }
    return result;
}

} // namespace llama_expert_profile
