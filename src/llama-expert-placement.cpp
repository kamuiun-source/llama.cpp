#include "llama-expert-placement.h"

#include "llama-impl.h"
#include "llama-model-loader.h"
#include "llama-single-miss-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

#ifdef __linux__
#include <cerrno>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

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

static llama_expert_placement::hip_overlap_mode packed_hip_mode(ggml_backend_dev_t target, const char * env) {
    using mode = llama_expert_placement::hip_overlap_mode;
    const char * value = std::getenv(env);
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
    throw std::runtime_error(format("%s must be 0, serial or 1", env));
}

llama_expert_placement::llama_expert_placement(const char * path, ggml_backend_dev_t target, int32_t n_expert_used)
    : target(target), max_ubatch(packed_max_ubatch()), hip_overlap(packed_hip_mode(target, "LLAMA_EXPERT_HIP_OVERLAP")),
        hip_prefill(packed_hip_mode(target, "LLAMA_EXPERT_HIP_PREFILL")), hip_overlap_promotion([] {
        const char * value = std::getenv("LLAMA_EXPERT_HIP_OVERLAP_PROMOTION");
        return value && std::strcmp(value, "1") == 0;
    }()), overlap([&] {
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
    }()), hip_mtp_verify_four([&] {
        const char * value = std::getenv("LLAMA_MTP_VERIFY_FOUR");
        return value && std::strcmp(value, "1") == 0 && dynamic_resident &&
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(target)), "ROCm") == 0;
    }()), hip_mtp_verify_three(hip_mtp_verify_four || [&] {
        const char * value = std::getenv("LLAMA_MTP_VERIFY_THREE");
        const bool hip_dynamic = dynamic_resident &&
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(target)), "ROCm") == 0;
        return value && ((std::strcmp(value, "1") == 0 && (hip_overlap != hip_overlap_mode::off || hip_dynamic)) ||
                (std::strcmp(value, "auto") == 0 && hip_dynamic));
    }()), hip_mtp_direct_ram([] {
        const char * value = std::getenv("LLAMA_EXPERT_HIP_MTP3_DIRECT_RAM");
        return value && std::strcmp(value, "1") == 0;
    }()), hip_tuple_dma([] {
        const char * value = std::getenv("LLAMA_EXPERT_HIP_TUPLE_DMA");
        return value && std::strcmp(value, "1") == 0;
    }()), hip_single_miss_cpu(hip_tuple_dma && hip_mtp_verify_three && !hip_mtp_verify_four &&
            !overlap && llama_single_miss_cpu_configured()), n_slots(packed_scratch_capacity(n_expert_used, hip_mtp_verify_four ? 4 : hip_mtp_verify_three ? 3 : max_ubatch)), prefill_tokens([&] {
        const char * value = std::getenv("LLAMA_EXPERT_PREFILL_32");
        const char * wide = std::getenv("LLAMA_EXPERT_PREFILL_512");
        const bool small = value && std::strcmp(value, "1") == 0;
        const bool large = wide && std::strcmp(wide, "1") == 0;
        if (hip_prefill != hip_overlap_mode::off) {
            if (small || large) {
                throw std::runtime_error("HIP expert prefill cannot use Vulkan prefill flags");
            }
            return 512;
        }
        if (small && large) {
            throw std::runtime_error("select only one packed expert prefill experiment");
        }
        return large ? 512 : small ? 32 : 0;
    }()) {
    const bool vulkan = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(target)), "Vulkan") == 0;
    const char * hip_dynamic = std::getenv("LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY");
    const char * mtp_three = std::getenv("LLAMA_MTP_VERIFY_THREE");
    if (hip_mtp_verify_four && (!hip_tuple_dma || (mtp_three &&
            (std::strcmp(mtp_three, "1") == 0 || std::strcmp(mtp_three, "auto") == 0)))) {
        throw std::runtime_error("HIP four-position MTP requires Expert tuple DMA and three-position mode off");
    }
    if (hip_mtp_verify_three && mtp_three && std::strcmp(mtp_three, "auto") == 0 && !hip_tuple_dma) {
        throw std::runtime_error("HIP automatic MTP verification width requires Expert tuple DMA");
    }
    if (hip_mtp_verify_three && (max_ubatch != 2 || n_expert_used != 10)) {
        throw std::runtime_error("HIP 3-position MTP requires two-token microbatches and top-10 routing");
    }
    if (hip_mtp_direct_ram && (!hip_mtp_verify_three || !dynamic_resident || overlap)) {
        throw std::runtime_error("direct HIP Expert RAM requires single-bank dynamic residency and fixed three-position MTP");
    }
    if (hip_tuple_dma && (!hip_mtp_verify_three || !dynamic_resident || overlap || hip_mtp_direct_ram)) {
        throw std::runtime_error("HIP Expert tuple DMA requires single-bank dynamic residency and three-position MTP support");
    }
    if (hip_overlap_promotion && hip_overlap == hip_overlap_mode::off) {
        throw std::runtime_error("HIP expert promotion requires the HIP split placement");
    }
    if (hip_overlap != hip_overlap_mode::off && hip_dynamic && std::strcmp(hip_dynamic, "1") == 0) {
        throw std::runtime_error("HIP expert overlap is incompatible with same-bank dynamic residency");
    }
    if (max_ubatch > 1 && !dynamic_resident && hip_overlap == hip_overlap_mode::off) {
        throw std::runtime_error("packed expert batching requires dynamic residency or HIP split placement");
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
    if (hip_prefill != hip_overlap_mode::off && (max_ubatch != 2 || !dynamic_resident || n_expert_used != 10 ||
            (rtx_cold && std::strcmp(rtx_cold, "0") != 0))) {
        throw std::runtime_error("HIP expert prefill requires two-position dynamic residency, top-10 routing and RTX cold assist off");
    }
    if (prefill_tokens && hip_prefill == hip_overlap_mode::off &&
            (!vulkan || max_ubatch != 2 || !dynamic_resident || (rtx_cold && std::strcmp(rtx_cold, "0") != 0))) {
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
    auto * bank = ggml_new_tensor_3d(bank_ctx.get(), source->type, source->ne[0], source->ne[1],
            n_resident + (overlap ? 1 + int(hip_overlap_promotion) : n_slots + int(hip_single_miss_cpu)));
    auto * cold = overlap ? ggml_new_tensor_3d(cold_ctx.get(), source->type, source->ne[0], source->ne[1], n_slots + 1) : nullptr;
    ggml_set_name(host, source->name);
    ggml_format_name(bank, "expert_bank.%s", source->name);
    if (cold) {
        ggml_format_name(cold, "expert_cold.%s", source->name);
    }
    weights.push_back({layer, shard, offset, host, bank, cold, source->nb[2]});
    return host;
}

#ifdef __linux__
llama_expert_placement::tuple_buffer_locks::~tuple_buffer_locks() {
    while (locked != 0) {
        const auto & range = ranges[--locked];
        if (munlock(const_cast<void *>(range.addr), range.size) != 0) {
            const int error = errno;
            LLAMA_LOG_WARN("expert placement HIP tuple mlock: munlock base=%p bytes=%zu failed: %s (errno=%d)\n",
                    range.addr, range.size, std::strerror(error), error);
        }
    }
}

std::unique_ptr<llama_expert_placement::tuple_buffer_locks> llama_expert_placement::lock_tuple_host() const {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::runtime_error("HIP Expert tuple mlock could not determine the OS page size");
    }
    const auto page_padding = [page = size_t(page_size)](size_t bytes) {
        const size_t remainder = bytes % page;
        return remainder ? page - remainder : 0;
    };

    auto locks = std::make_unique<tuple_buffer_locks>();
    std::array<ggml_backend_buffer_t, n_layers> buffers;
    auto buffers_end = buffers.begin();
    // The owner can be a multi-buffer. Each tuple names its actual HIP allocation.
    for (const auto & tuple : tuples) {
        const auto buffer = tuple.host->buffer;
        if (std::find(buffers.begin(), buffers_end, buffer) == buffers_end) {
            *buffers_end++ = buffer;
            locks->ranges.push_back({ggml_backend_buffer_get_base(buffer), ggml_backend_buffer_get_size(buffer)});
        }
    }
    std::sort(locks->ranges.begin(), locks->ranges.end(), [](const llama_memory_range & a, const llama_memory_range & b) {
        return reinterpret_cast<uintptr_t>(a.addr) < reinterpret_cast<uintptr_t>(b.addr);
    });

    uintptr_t prev_end = 0;
    size_t total_bytes = 0;
    size_t total_page_bytes = 0;
    for (const auto & range : locks->ranges) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(range.addr);
        const size_t padding = page_padding(range.size);
        if (!base || base % size_t(page_size) != 0 || base < prev_end || !range.size ||
                padding > SIZE_MAX - range.size || range.size + padding > UINTPTR_MAX - base ||
                range.size + padding > SIZE_MAX - total_page_bytes) {
            throw std::runtime_error(format("invalid HIP Expert tuple mlock range: base=%p bytes=%zu", range.addr, range.size));
        }
        prev_end = base + range.size + padding;
        total_bytes += range.size;
        total_page_bytes += range.size + padding;
    }
    if (total_bytes != ggml_backend_buffer_get_size(host_buf.get())) {
        throw std::runtime_error("HIP Expert tuple mlock ranges do not cover the native host allocation");
    }

    for (const auto & range : locks->ranges) {
        const size_t page_bytes = range.size + page_padding(range.size);
        if (mlock(range.addr, range.size) != 0) {
            const int error = errno;
            struct rlimit limit;
            std::string memlock_limit = "unavailable";
            if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
                const auto limit_bytes = [](rlim_t bytes) {
                    return bytes == RLIM_INFINITY ? std::string("unlimited") : std::to_string(bytes);
                };
                memlock_limit = "soft=" + limit_bytes(limit.rlim_cur) + " hard=" + limit_bytes(limit.rlim_max);
            }
            throw std::runtime_error(format("HIP Expert tuple mlock failed: base=%p bytes=%zu page_bytes=%zu total_page_bytes=%zu: "
                        "%s (errno=%d), RLIMIT_MEMLOCK %s. Set a sufficient per-process limit in bytes "
                        "(Docker: --ulimit memlock=BYTES:BYTES).",
                        range.addr, range.size, page_bytes, total_page_bytes, std::strerror(error), error, memlock_limit.c_str()));
        }
        ++locks->locked;
        LLAMA_LOG_INFO("expert placement HIP tuple mlock: base=%p bytes=%zu page_bytes=%zu locked=1\n",
                range.addr, range.size, page_bytes);
    }
    LLAMA_LOG_INFO("expert placement HIP tuple mlock: requested=1 buffers=%zu bytes=%zu page_bytes=%zu\n",
            locks->locked, total_bytes, total_page_bytes);
    return locks;
}
#endif

bool llama_expert_placement::load_tuples(llama_model_loader & loader, const llama_model_params & params) {
    if (params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) {
        throw std::runtime_error("HIP Expert tuple DMA uses native pinned buffers and requires an unlocked model load mode");
    }
    const auto host_buft = ggml_backend_dev_host_buffer_type(target);
    const auto device_buft = ggml_backend_dev_buffer_type(target);
    if (!host_buft) {
        throw std::runtime_error("HIP Expert tuple DMA requires a native pinned host buffer type");
    }
    const size_t alignment = ggml_backend_buft_get_alignment(device_buft);
    std::array<size_t, n_layers> sizes{}, alignments{};
    std::array<int, n_layers> gate_up_count{}, down_count{};
    std::array<ggml_type, n_layers> gate_up_type{};
    alignments.fill(alignment);
    for (const auto & w : weights) {
        const auto type = w.host->type;
        const auto & ne = w.host->ne;
        if (ne[0] == 2560 && ne[1] == 640 &&
                (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS)) {
            if (gate_up_count[w.layer]++ && gate_up_type[w.layer] != type) {
                throw std::runtime_error("HIP Expert tuple DMA requires matching gate and up types");
            }
            gate_up_type[w.layer] = type;
        } else if (ne[0] == 640 && ne[1] == 2560 && (type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q2_0)) {
            ++down_count[w.layer];
        } else {
            throw std::runtime_error(format("unsupported HIP Expert tuple projection %s", w.host->name));
        }
        alignments[w.layer] = std::lcm(alignments[w.layer], ggml_type_size(type));
        tuples[w.layer].payload_bytes += w.payload_bytes;
    }
    for (const bool down : {false, true}) {
        for (const auto & w : weights) {
            if ((w.host->ne[0] == 640) != down) {
                continue;
            }
            const size_t offset = GGML_PAD(sizes[w.layer], alignment);
            const size_t tail = ggml_backend_buft_get_alloc_size(device_buft, w.bank) - ggml_nbytes(w.bank);
            w.host->view_offs = w.bank->view_offs = offset;
            // MMQ can read the padded K tail past the final row of every Expert plane.
            sizes[w.layer] = offset + w.payload_bytes + tail;
        }
    }

    const ggml_init_params init{n_layers * ggml_tensor_overhead(), nullptr, true};
    tuple_host_ctx.reset(ggml_init(init));
    tuple_bank_ctx.reset(ggml_init(init));
    if (!tuple_host_ctx || !tuple_bank_ctx) {
        throw std::runtime_error("could not create HIP Expert tuple contexts");
    }
    for (int il = 0; il < n_layers; ++il) {
        if (gate_up_count[il] != 2 || down_count[il] != 1) {
            throw std::runtime_error("HIP Expert tuple DMA requires two gate/up planes and one down plane per layer");
        }
        auto & tuple = tuples[il];
        tuple.stride = (sizes[il] + alignments[il] - 1) / alignments[il] * alignments[il];
        tuple.host = ggml_new_tensor_1d(tuple_host_ctx.get(), GGML_TYPE_I8, (n_experts - n_resident) * tuple.stride);
        tuple.device = ggml_new_tensor_1d(tuple_bank_ctx.get(), GGML_TYPE_I8, (n_resident + n_slots + int(hip_single_miss_cpu)) * tuple.stride);
        ggml_format_name(tuple.host, "expert_tuple_host.%d", il);
        ggml_format_name(tuple.device, "expert_tuple_bank.%d", il);
    }
#ifdef __linux__
    tuple_host_locks.reset();
    constexpr int mlock_requested = 1;
#else
    constexpr int mlock_requested = 0;
#endif
    bank_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(tuple_bank_ctx.get(), device_buft));
    host_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(tuple_host_ctx.get(), host_buft));
    if (!bank_buf || ggml_backend_buffer_is_host(bank_buf.get()) || !host_buf) {
        throw std::runtime_error("could not allocate HIP Expert tuple buffers");
    }
    size_t pinned_bytes = 0;
    size_t fallback_bytes = 0;
    size_t resident_payload_bytes = 0;
    size_t ram_payload_bytes = 0;
    for (const auto & tuple : tuples) {
        auto & bytes = ggml_backend_buffer_get_type(tuple.host->buffer) == host_buft ? pinned_bytes : fallback_bytes;
        bytes += ggml_nbytes(tuple.host);
        resident_payload_bytes += n_resident * tuple.payload_bytes;
        ram_payload_bytes += (n_experts - n_resident) * tuple.payload_bytes;
    }
    LLAMA_LOG_INFO("expert placement HIP tuple DMA: pinned_bytes=%zu fallback_bytes=%zu bank_bytes=%zu"
            " resident_payload_bytes=%zu ram_payload_bytes=%zu mlock_requested=%d\n", pinned_bytes, fallback_bytes,
            ggml_backend_buffer_get_size(bank_buf.get()), resident_payload_bytes, ram_payload_bytes, mlock_requested);
    if (fallback_bytes) {
        throw std::runtime_error("HIP Expert tuple DMA allocation fell back to pageable memory");
    }
#ifdef __linux__
    auto locks = lock_tuple_host();
#endif
    ggml_backend_buffer_set_usage(host_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(bank_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(host_buf.get(), 0);
    ggml_backend_buffer_clear(bank_buf.get(), 0);

    std::vector<char> resident;
    for (const auto & w : weights) {
        if (params.progress_callback && !params.progress_callback(
                static_cast<float>(loader.size_done) / loader.size_data, params.progress_callback_user_data)) {
            return false;
        }
        const auto & tuple = tuples[w.layer];
        const auto init_view = [&](ggml_tensor * plane, ggml_tensor * raw) {
            plane->view_src = raw;
            plane->nb[2] = tuple.stride;
            plane->nb[3] = plane->ne[2] * tuple.stride;
            if (plane->view_offs + ggml_nbytes(plane) > ggml_nbytes(raw) ||
                    ggml_backend_view_init(plane) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("could not initialize a HIP Expert tuple plane");
            }
        };
        init_view(w.host, tuple.host);
        init_view(w.bank, tuple.device);

        // Keep only one dense resident plane during the initial file load.
        const auto & map = maps[w.layer];
        const size_t bytes = w.payload_bytes;
        resident.resize(n_resident * bytes);
        auto & file = *loader.files.at(w.shard);
        file.seek(w.offset, SEEK_SET);
        for (int expert = 0; expert < n_experts; ++expert) {
            const int32_t slot = map.resident[expert];
            auto * data = slot >= 0 ? resident.data() + slot * bytes :
                    static_cast<char *>(w.host->data) + map.host[expert] * tuple.stride;
            file.read_raw(data, bytes);
            if (loader.check_tensors && !ggml_validate_row_data(w.host->type, data, bytes)) {
                throw std::runtime_error(format("invalid data in expert tensor %s", w.host->name));
            }
        }
        ggml_backend_tensor_set_2d(w.bank, resident.data(), 0, bytes, n_resident, tuple.stride, bytes);
        loader.size_done += n_experts * bytes;
    }
#ifdef __linux__
    tuple_host_locks = std::move(locks);
#endif
    return true;
}

bool llama_expert_placement::load(llama_model_loader & loader, const llama_model_params & params) {
    if (hip_tuple_dma) {
        return load_tuples(loader, params);
    }
    const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        throw std::runtime_error("packed experts require a CPU backend");
    }
    const bool lock_host = params.load_mode == LLAMA_LOAD_MODE_MLOCK || params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (hip_mtp_direct_ram && lock_host) {
        throw std::runtime_error("direct HIP Expert RAM uses native pinned buffers and requires an unlocked model load mode");
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
    const auto host_buft = hip_mtp_direct_ram ? ggml_backend_dev_host_buffer_type(target) : ggml_backend_dev_buffer_type(cpu);
    if (!host_buft) {
        throw std::runtime_error("direct HIP Expert RAM requires a native pinned host buffer type");
    }
    host_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(host_ctx.get(), host_buft));
    if (hip_mtp_direct_ram) {
        size_t pinned_bytes = 0;
        size_t fallback_bytes = 0;
        if (!host_buf) {
            throw std::runtime_error("could not allocate direct HIP Expert RAM");
        }
        for (const auto & w : weights) {
            auto & bytes = ggml_backend_buffer_get_type(w.host->buffer) == host_buft ? pinned_bytes : fallback_bytes;
            bytes += ggml_nbytes(w.host);
        }
        LLAMA_LOG_INFO("expert placement HIP direct RAM: pinned_bytes=%zu fallback_bytes=%zu\n", pinned_bytes, fallback_bytes);
        if (fallback_bytes != 0) {
            throw std::runtime_error("direct HIP Expert RAM allocation fell back to pageable memory");
        }
    }
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
    if (lock_host) {
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
