#include "llama-expert-placement.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

using namespace llama_expert_profile;

std::vector<int32_t> llama_expert_placement::layer_map::scratch_slots(int32_t n_slots) const {
    if (n_slots <= 0 || n_slots > n_experts - n_resident) {
        throw std::runtime_error("invalid packed expert scratch capacity");
    }
    const int32_t n_gpu = n_resident + n_slots;
    std::array<bool, n_experts> gpu_used{};
    std::array<bool, n_experts - n_resident> host_used{};
    for (int32_t expert = 0; expert < n_experts; ++expert) {
        const int32_t gpu_slot = resident[expert];
        const int32_t host_slot = host[expert];
        if (gpu_slot >= 0 && gpu_slot < n_gpu && host_slot == -1 && !gpu_used[gpu_slot]) {
            gpu_used[gpu_slot] = true;
        } else if (host_slot >= 0 && host_slot < n_experts - n_resident && gpu_slot == -1 && !host_used[host_slot]) {
            host_used[host_slot] = true;
        } else {
            throw std::runtime_error("invalid packed expert ownership map");
        }
    }
    std::vector<int32_t> slots;
    slots.reserve(n_slots);
    for (int32_t slot = 0; slot < n_gpu; ++slot) {
        if (!gpu_used[slot]) {
            slots.push_back(slot);
        }
    }
    if (slots.size() != static_cast<size_t>(n_slots)) {
        throw std::runtime_error("invalid packed expert residency budget");
    }
    return slots;
}

static int32_t packed_max_ubatch() {
    const char * value = std::getenv("LLAMA_EXPERT_BATCH_TOKENS");
    if (!value || std::strcmp(value, "1") == 0) {
        return 1;
    }
    if (std::strcmp(value, "2") == 0) {
        return 2;
    }
    throw std::runtime_error("LLAMA_EXPERT_BATCH_TOKENS must be 1 or 2");
}

static int32_t packed_scratch_capacity(int32_t n_expert_used, int32_t max_ubatch) {
    if (n_expert_used <= 0 || n_expert_used > (n_experts - n_resident) / max_ubatch) {
        throw std::runtime_error("invalid packed expert count for scratch capacity");
    }
    return n_expert_used * max_ubatch;
}

static llama_expert_placement::hip_overlap_mode packed_hip_overlap(ggml_backend_dev_t target) {
    using mode = llama_expert_placement::hip_overlap_mode;
    const char * value = std::getenv("LLAMA_EXPERT_HIP_OVERLAP");
    if (!value || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(target)), "ROCm") != 0 ||
            std::strcmp(value, "0") == 0) {
        return mode::off;
    }
    if (std::strcmp(value, "serial") == 0) {
        return mode::serial;
    }
    if (std::strcmp(value, "1") == 0) {
        return mode::async;
    }
    throw std::runtime_error("LLAMA_EXPERT_HIP_OVERLAP must be 0, serial or 1");
}

llama_expert_placement::llama_expert_placement(const char * path, ggml_backend_dev_t target, int32_t n_expert_used)
    : target(target), max_ubatch(packed_max_ubatch()), n_slots(packed_scratch_capacity(n_expert_used, max_ubatch)),
        hip_overlap(packed_hip_overlap(target)), overlap([&] {
        const char * value = std::getenv("LLAMA_EXPERT_OVERLAP");
        return hip_overlap != hip_overlap_mode::off || (value && std::strcmp(value, "1") == 0);
    }()), dynamic_resident([&] {
        if (overlap) {
            return false;
        }
        const char * backend = ggml_backend_reg_name(ggml_backend_dev_backend_reg(target));
        const char * hip_dynamic = std::getenv("LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY");
        return std::strcmp(backend, "Vulkan") == 0 ||
               (std::strcmp(backend, "ROCm") == 0 && hip_dynamic && std::strcmp(hip_dynamic, "1") == 0);
    }()), prefill_tokens([] {
        const char * value = std::getenv("LLAMA_EXPERT_PREFILL_32");
        const char * wide = std::getenv("LLAMA_EXPERT_PREFILL_512");
        const bool small = value && std::strcmp(value, "1") == 0;
        const bool large = wide && std::strcmp(wide, "1") == 0;
        if (small && large) {
            throw std::runtime_error("select only one packed expert prefill experiment");
        }
        return large ? 512 : small ? 32 : 0;
    }()) {
    const bool vulkan = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(target)), "Vulkan") == 0;
    const char * hip_dynamic = std::getenv("LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY");
    if (hip_overlap != hip_overlap_mode::off && (max_ubatch != 1 || (hip_dynamic && std::strcmp(hip_dynamic, "1") == 0))) {
        throw std::runtime_error("HIP expert overlap requires one position and dynamic residency off");
    }
    if (max_ubatch > 1 && !dynamic_resident) {
        throw std::runtime_error("packed expert batching requires dynamic residency");
    }
    if (dynamic_resident) {
        if (n_slots <= 0 || n_slots > n_experts - n_resident) {
            throw std::runtime_error("dynamic expert residency requires synchronous placement with scratch reuse");
        }
    }
    if (overlap && hip_overlap == hip_overlap_mode::off && (!vulkan || !std::getenv("GGML_VK_ASYNC_USE_TRANSFER_QUEUE"))) {
        throw std::runtime_error("expert overlap requires Vulkan and GGML_VK_ASYNC_USE_TRANSFER_QUEUE");
    }
    const char * rtx_cold = std::getenv("LLAMA_EXPERT_RTX_COLD_SLOTS");
    if (prefill_tokens && (!vulkan || max_ubatch != 2 || !dynamic_resident || (rtx_cold && std::strcmp(rtx_cold, "0") != 0))) {
        throw std::runtime_error("packed expert prefill requires two-token dynamic Vulkan residency and RTX cold assist off");
    }
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
    if (overlap) {
        cold_ctx.reset(ggml_init(init));
    }
    if (!host_ctx || !bank_ctx || (overlap && !cold_ctx)) {
        throw std::runtime_error("could not create packed expert contexts");
    }
}

int32_t llama_expert_placement::prefill_waves(int64_t n_tokens, int64_t n_used) const {
    if (prefill_tokens > 32 || n_tokens <= max_ubatch || n_tokens > prefill_tokens || n_used <= 0 || n_used > n_slots / max_ubatch) {
        return 0;
    }
    const int32_t max_cold = std::min<int64_t>(n_experts - n_resident, n_tokens * n_used);
    return (max_cold + n_slots - 1) / n_slots;
}

ggml_tensor * llama_expert_placement::add_weight(int layer, const ggml_tensor * source, uint16_t shard, size_t offset) {
    if (layer < 0 || layer >= n_layers || ggml_n_dims(source) != 3 || source->ne[2] != n_experts ||
            !ggml_is_contiguous(source) || weights.size() >= 3 * n_layers) {
        throw std::runtime_error(format("cannot pack expert tensor %s", source->name));
    }
    auto * host = ggml_new_tensor_3d(host_ctx.get(), source->type, source->ne[0], source->ne[1], n_experts - n_resident);
    auto * bank = ggml_new_tensor_3d(bank_ctx.get(), source->type, source->ne[0], source->ne[1], n_resident + (overlap ? 1 : n_slots));
    auto * cold = overlap ? ggml_new_tensor_3d(cold_ctx.get(), source->type, source->ne[0], source->ne[1], n_slots + 1) : nullptr;
    ggml_set_name(host, source->name);
    ggml_format_name(bank, "expert_bank.%s", source->name);
    if (cold) {
        ggml_format_name(cold, "expert_cold.%s", source->name);
    }
    weights.push_back({layer, shard, offset, host, bank, cold});
    return host;
}

bool llama_expert_placement::load(llama_model_loader & loader, const llama_model_params & params) {
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("packed experts require a CPU backend");
    }
    using alloc_overlap_bank_t = ggml_backend_buffer_t (*)(ggml_backend_dev_t, ggml_context *);
    alloc_overlap_bank_t alloc_overlap_bank = nullptr;
    if (!overlap && dynamic_resident && max_ubatch == 2) {
        const auto reg = ggml_backend_dev_backend_reg(target);
        if (reg && std::strcmp(ggml_backend_reg_name(reg), "Vulkan") == 0) {
            alloc_overlap_bank = reinterpret_cast<alloc_overlap_bank_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_alloc_overlap_expert_bank"));
        }
    }
    bank_buf.reset(alloc_overlap_bank ? alloc_overlap_bank(target, bank_ctx.get()) :
            ggml_backend_alloc_ctx_tensors_from_buft(bank_ctx.get(), ggml_backend_dev_buffer_type(target)));
    host_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), ggml_backend_dev_buffer_type(cpu)));
    if (!host_buf || !ggml_backend_buffer_is_host(host_buf.get()) || !bank_buf || ggml_backend_buffer_is_host(bank_buf.get())) {
        throw std::runtime_error("could not allocate packed expert buffers");
    }
    ggml_backend_buffer_set_usage(host_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bank_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (overlap) {
        cold_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(cold_ctx.get(), ggml_backend_dev_buffer_type(target)));
        if (!cold_buf || ggml_backend_buffer_is_host(cold_buf.get())) {
            throw std::runtime_error("could not allocate separate cold expert buffers");
        }
        ggml_backend_buffer_set_usage(cold_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
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
        if (w.cold) {
            // Each half of the graph uses a zero expert for the other half's IDs.
            ggml_backend_tensor_memset(w.bank, 0, resident.size(), stride);
            ggml_backend_tensor_memset(w.cold, 0, n_slots * stride, stride);
        }
        resident_bytes += resident.size();
        ram_bytes += ggml_nbytes(w.host);
        loader.size_done += n_experts * stride;
    }
    LLAMA_LOG_INFO("expert placement: %s %zu resident bytes, RAM %zu bytes, %d scratch slots per layer\n",
            ggml_backend_dev_name(target), resident_bytes, ram_bytes, n_slots);
    return true;
}
