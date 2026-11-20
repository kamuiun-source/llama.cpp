#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-expert-profile.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace llama_expert_profile;
enum projection : unsigned { gate = 1, up = 2, gate_up = 4, down = 8 };

struct expert_weight {
    int layer;
    std::string name;
    ggml_type type;
    int64_t columns;
    int64_t rows;
    size_t offset;
    size_t expert_bytes;
    ggml_tensor * resident = nullptr;
    ggml_tensor * host = nullptr;
};

using weight_shard = std::pair<std::string, std::vector<expert_weight>>;

std::vector<weight_shard> read_weights(const char * path) {
    gguf_context_ptr first(gguf_init_from_file(path, {true, nullptr}));
    if (!first) {
        throw std::runtime_error("could not open the first GGUF shard");
    }
    const auto arch = gguf_find_key(first.get(), "general.architecture");
    if (arch < 0 || gguf_get_kv_type(first.get(), arch) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(first.get(), arch)) != "qwen4exp") {
        throw std::runtime_error("expected a Qwen3.8-Flash-Next GGUF");
    }
    const auto split = gguf_find_key(first.get(), "split.count");
    if (split >= 0 && gguf_get_kv_type(first.get(), split) != GGUF_TYPE_UINT16) {
        throw std::runtime_error("invalid GGUF split count");
    }
    const int n_shards = split < 0 ? 1 : gguf_get_val_u16(first.get(), split);
    if (n_shards < 1 || n_shards > 64) {
        throw std::runtime_error("unsupported GGUF split count");
    }

    std::array<char, 4096> prefix{};
    if (n_shards > 1 && llama_split_prefix(prefix.data(), prefix.size(), path, 0, n_shards) <= 0) {
        throw std::runtime_error("pass the first GGUF shard");
    }
    const std::regex name_pattern(R"(^blk\.([0-9]{1,2})\.ffn_(gate|up|gate_up|down)_exps\.weight$)");
    std::array<unsigned, n_layers> projections{};
    std::vector<weight_shard> shards;
    for (int split_no = 0; split_no < n_shards; ++split_no) {
        std::string shard_path = path;
        if (split_no != 0) {
            std::array<char, 4096> name{};
            const int size = llama_split_path(name.data(), name.size(), prefix.data(), split_no, n_shards);
            if (size <= 0 || static_cast<size_t>(size) >= name.size()) {
                throw std::runtime_error("GGUF shard path is too long");
            }
            shard_path = name.data();
        }
        ggml_context * meta = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(shard_path.c_str(), {true, &meta}));
        ggml_context_ptr tensors(meta);
        if (!gguf || !tensors) {
            throw std::runtime_error("could not read GGUF shard " + shard_path);
        }
        const auto file_size = std::filesystem::file_size(std::filesystem::u8path(shard_path));
        auto & weights = shards.emplace_back(shard_path, std::vector<expert_weight>{}).second;
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf.get()); ++i) {
            const std::string name = gguf_get_tensor_name(gguf.get(), i);
            std::smatch match;
            if (!std::regex_match(name, match, name_pattern)) {
                continue;
            }
            const int layer = std::stoi(match[1]);
            if (layer >= n_layers) {
                continue; // The draft head is outside the 48 target layers.
            }
            const std::string part = match[2];
            const unsigned flag = part == "gate" ? gate : part == "up" ? up : part == "gate_up" ? gate_up : down;
            if (projections[layer] & flag) {
                throw std::runtime_error("duplicate expert tensor " + name);
            }
            projections[layer] |= flag;
            const ggml_tensor * tensor = ggml_get_tensor(tensors.get(), name.c_str());
            const size_t start = gguf_get_data_offset(gguf.get());
            const size_t offset = gguf_get_tensor_offset(gguf.get(), i);
            const auto stream_limit = static_cast<uintmax_t>(std::numeric_limits<std::streamoff>::max());
            if (!tensor || tensor->ne[2] != n_experts || ggml_n_dims(tensor) != 3 || !ggml_is_contiguous(tensor) ||
                    offset > std::numeric_limits<size_t>::max() - start || start + offset > file_size ||
                    start + offset > stream_limit || ggml_nbytes(tensor) > file_size - start - offset ||
                    ggml_nbytes(tensor) > stream_limit - start - offset) {
                throw std::runtime_error("invalid expert tensor " + name);
            }
            weights.push_back({layer, name, tensor->type, tensor->ne[0], tensor->ne[1], start + offset, tensor->nb[2]});
        }
    }
    for (int layer = 0; layer < n_layers; ++layer) {
        if (projections[layer] != (gate | up | down) && projections[layer] != (gate_up | down)) {
            throw std::runtime_error("missing or conflicting expert projections in layer " + std::to_string(layer));
        }
    }
    return shards;
}

void verify_placement(const char * path, const char * profile_path, const char * device_name) {
    const auto selected = read_profile(profile_path);
    auto shards = read_weights(path);
    ggml_backend_load_all();
    const auto gpu = ggml_backend_dev_by_name(device_name);
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!gpu || ggml_backend_dev_type(gpu) != GGML_BACKEND_DEVICE_TYPE_GPU || !cpu) {
        throw std::runtime_error("could not find the requested GPU and CPU backends");
    }

    const ggml_init_params params{3 * n_layers * ggml_tensor_overhead(), nullptr, true};
    ggml_context_ptr resident_ctx(ggml_init(params));
    ggml_context_ptr host_ctx(ggml_init(params));
    if (!resident_ctx || !host_ctx) {
        throw std::runtime_error("could not create expert tensor contexts");
    }
    for (auto & shard : shards) {
        for (auto & weight : shard.second) {
            weight.resident = ggml_new_tensor_3d(resident_ctx.get(), weight.type, weight.columns, weight.rows, n_resident);
            weight.host = ggml_new_tensor_3d(host_ctx.get(), weight.type, weight.columns, weight.rows, n_experts - n_resident);
            ggml_set_name(weight.resident, weight.name.c_str());
            ggml_set_name(weight.host, weight.name.c_str());
        }
    }
    ggml_backend_buffer_ptr resident_buffer(ggml_backend_alloc_ctx_tensors_from_buft(resident_ctx.get(), ggml_backend_dev_buffer_type(gpu)));
    if (!resident_buffer || ggml_backend_buffer_is_host(resident_buffer.get())) {
        throw std::runtime_error("could not allocate device memory for resident experts");
    }
    ggml_backend_buffer_ptr host_buffer(ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), ggml_backend_dev_buffer_type(cpu)));
    if (!host_buffer || !ggml_backend_buffer_is_host(host_buffer.get())) {
        throw std::runtime_error("could not allocate host memory for the remaining experts");
    }

    std::array<size_t, n_layers> layer_bytes{};
    std::vector<char> resident_data;
    std::vector<char> host_expert;
    for (const auto & shard : shards) {
        if (shard.second.empty()) {
            continue;
        }
        std::ifstream file(std::filesystem::u8path(shard.first), std::ios::binary);
        for (const auto & weight : shard.second) {
            size_t gpu_offset = 0;
            size_t cpu_offset = 0;
            resident_data.resize(ggml_nbytes(weight.resident));
            host_expert.resize(weight.expert_bytes);
            file.seekg(static_cast<std::streamoff>(weight.offset));
            for (int expert = 0; expert < n_experts; ++expert) {
                const bool resident = selected[weight.layer][expert];
                char * destination = resident ? resident_data.data() + gpu_offset : host_expert.data();
                if (!file.read(destination, static_cast<std::streamsize>(weight.expert_bytes))) {
                    throw std::runtime_error("could not read expert weights from " + shard.first);
                }
                if (resident) {
                    gpu_offset += weight.expert_bytes;
                } else {
                    ggml_backend_tensor_set(weight.host, destination, cpu_offset, weight.expert_bytes);
                    cpu_offset += weight.expert_bytes;
                }
            }
            ggml_backend_tensor_set(weight.resident, resident_data.data(), 0, gpu_offset);
            layer_bytes[weight.layer] += weight.expert_bytes;
        }
    }
    size_t total = 0;
    for (int layer = 0; layer < n_layers; ++layer) {
        const size_t bytes = layer_bytes[layer];
        total += bytes;
        std::printf("layer %2d: %s %d (%zu bytes), RAM %d (%zu bytes)\n", layer,
                ggml_backend_dev_name(gpu), n_resident, bytes * n_resident, n_experts - n_resident, bytes * (n_experts - n_resident));
    }
    std::printf("Loaded %d layers: %s %zu bytes, RAM %zu bytes\n", n_layers, ggml_backend_dev_name(gpu),
            total * n_resident, total * (n_experts - n_resident));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s FIRST_SHARD.gguf PROFILE.txt DEVICE\n", argv[0]);
        return 1;
    }
    try {
        verify_placement(argv[1], argv[2], argv[3]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Expert placement: %s\n", error.what());
        return 1;
    }
}
