#include "llama-expert-placement.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

using namespace llama_expert_profile;

llama_expert_placement::llama_expert_placement(const char * path, ggml_backend_dev_t target, int32_t n_slots)
    : target(target), n_slots(n_slots) {
    const auto selected = read_profile(path);
    for (int il = 0; il < n_layers; ++il) {
        int32_t resident = 0;
        int32_t host = 0;
        for (int expert = 0; expert < n_experts; ++expert) {
            const bool on_gpu = selected[il][expert];
            maps[il].resident[expert] = on_gpu ? resident++ : -1;
            maps[il].host[expert] = on_gpu ? -1 : host++;
        }
    }
    const ggml_init_params init{3 * n_layers * ggml_tensor_overhead(), nullptr, true};
    host_ctx.reset(ggml_init(init));
    bank_ctx.reset(ggml_init(init));
    if (!host_ctx || !bank_ctx) {
        throw std::runtime_error("could not create packed expert contexts");
    }
}

ggml_tensor * llama_expert_placement::add_weight(int layer, const ggml_tensor * source, uint16_t shard, size_t offset) {
    if (layer < 0 || layer >= n_layers || ggml_n_dims(source) != 3 || source->ne[2] != n_experts ||
            !ggml_is_contiguous(source) || weights.size() >= 3 * n_layers) {
        throw std::runtime_error(format("cannot pack expert tensor %s", source->name));
    }
    auto * host = ggml_new_tensor_3d(host_ctx.get(), source->type, source->ne[0], source->ne[1], n_experts - n_resident);
    auto * bank = ggml_new_tensor_3d(bank_ctx.get(), source->type, source->ne[0], source->ne[1], n_resident + n_slots);
    ggml_set_name(host, source->name);
    ggml_format_name(bank, "expert_bank.%s", source->name);
    weights.push_back({layer, shard, offset, host, bank});
    return host;
}

bool llama_expert_placement::load(llama_model_loader & loader, const llama_model_params & params) {
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("packed experts require a CPU backend");
    }
    bank_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(bank_ctx.get(), ggml_backend_dev_buffer_type(target)));
    host_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), ggml_backend_dev_buffer_type(cpu)));
    if (!host_buf || !ggml_backend_buffer_is_host(host_buf.get()) || !bank_buf || ggml_backend_buffer_is_host(bank_buf.get())) {
        throw std::runtime_error("could not allocate packed expert buffers");
    }
    ggml_backend_buffer_set_usage(host_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bank_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
        host_lock.init(ggml_backend_buffer_get_base(host_buf.get()));
        host_lock.grow_to(ggml_backend_buffer_get_size(host_buf.get()));
    }

    size_t resident_bytes = 0;
    size_t ram_bytes = 0;
    std::vector<char> resident;
    std::vector<char> cold;
    for (const auto & w : weights) {
        if (params.progress_callback && !params.progress_callback(
                static_cast<float>(loader.size_done) / loader.size_data, params.progress_callback_user_data)) {
            return false;
        }
        const auto & map = maps[w.layer];
        const size_t stride = w.host->nb[2];
        resident.resize(n_resident * stride);
        cold.resize(stride);
        auto & file = *loader.files.at(w.shard);
        file.seek(w.offset, SEEK_SET);
        for (int expert = 0; expert < n_experts; ++expert) {
            const int32_t slot = map.resident[expert];
            char * data = slot >= 0 ? resident.data() + slot * stride : cold.data();
            file.read_raw(data, stride);
            if (loader.check_tensors && !ggml_validate_row_data(w.host->type, data, stride)) {
                throw std::runtime_error(format("invalid data in expert tensor %s", w.host->name));
            }
            if (slot < 0) {
                ggml_backend_tensor_set(w.host, data, map.host[expert] * stride, stride);
            }
        }
        ggml_backend_tensor_set(w.bank, resident.data(), 0, resident.size());
        resident_bytes += resident.size();
        ram_bytes += ggml_nbytes(w.host);
        loader.size_done += n_experts * stride;
    }
    LLAMA_LOG_INFO("expert placement: %s %zu resident bytes, RAM %zu bytes, %d scratch slots per layer\n",
            ggml_backend_dev_name(target), resident_bytes, ram_bytes, n_slots);
    return true;
}
