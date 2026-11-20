#include "llama-rtx-expert.h"

#include "ggml-alloc.h"
#include "llama-impl.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr size_t mib = 1024 * 1024;
constexpr size_t cuda_reserve = 512 * mib;
constexpr int max_ranks = 32;
constexpr int n_tokens = 2;
constexpr int n_planes = 4;

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using event_ptr = std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)>;
using planes = std::array<ggml_tensor *, n_planes>;

size_t expert_vram_budget() {
    const char * env = std::getenv("LLAMA_EXPERT_RTX_COLD_VRAM_MIB");
    if (!env) {
        return 128 * mib;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(env, &end, 10);
    return !errno && end != env && !*end && value <= 16384 ? value * mib : 0;
}

ggml_backend_reg_t backend_registry(ggml_backend_t backend) {
    return ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
}

bool equivalent(const ggml_tensor * a, const ggml_tensor * b) {
    return (!a && !b) || (a && b && a->type == b->type && a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] &&
            a->nb[2] == b->nb[2]);
}

struct workspace {
    planes layout{};
    int32_t count = 0;
    context_ptr ctx{nullptr, ggml_free};
    buffer_ptr device{nullptr, ggml_backend_buffer_free};
    buffer_ptr host{nullptr, ggml_backend_buffer_free};
    event_ptr consumed{nullptr, ggml_backend_event_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * rows = nullptr;
    ggml_tensor * result = nullptr;
    planes weights{};
    std::array<size_t, n_planes> offsets{};
    size_t ids_offset = 0;
    size_t rows_offset = 0;
    bool staging_busy = false;
};

} // namespace

struct llama_rtx_expert::impl {
    using ready_t = bool (*)(ggml_backend_t, ggml_backend_event_t);
    using zero_t = void (*)(ggml_backend_t, ggml_tensor *);
    using vk_input_t = const ggml_tensor * (*)(ggml_backend_t, const ggml_cgraph *, const ggml_tensor *, int);

    explicit impl(ggml_backend_t cuda, int32_t slots) : cuda(cuda), slots(slots) {
        const auto reg = backend_registry(cuda);
        event_ready = reinterpret_cast<ready_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_event_ready"));
        zero = reinterpret_cast<zero_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_expert_zero"));
    }

    ggml_backend_t cuda;
    int32_t slots;
    const size_t budget = expert_vram_budget();
    size_t allocated = 0;
    ready_t event_ready = nullptr;
    zero_t zero = nullptr;
    context_ptr output_ctx{nullptr, ggml_free};
    buffer_ptr output_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor * contribution = nullptr;
    std::vector<planes> layers;
    std::vector<std::unique_ptr<workspace>> workspaces;
    const ggml_cgraph * ready_graph = nullptr;
    const ggml_tensor * ready_map = nullptr;
    const ggml_tensor * ready_input = nullptr;
    int ready_layer = -1;
    size_t submissions = 0;
    size_t graph_launches = 0;

    workspace * get_workspace(const planes & layout, int32_t count) {
        for (const auto & ws : workspaces) {
            bool same = ws->count == count;
            for (int j = 0; j < n_planes; ++j) {
                same &= equivalent(ws->layout[j], layout[j]);
            }
            if (same) {
                return ws.get();
            }
        }

        auto ws = std::make_unique<workspace>();
        ws->layout = layout;
        ws->count = count;
        constexpr int graph_size = 32;
        ws->ctx.reset(ggml_init({graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
                nullptr, true}));
        if (!ws->ctx) {
            return nullptr;
        }

        auto * ctx = ws->ctx.get();
        const int64_t d = contribution->ne[0];
        const int64_t k = contribution->ne[1];
        ws->input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, n_tokens);
        ws->ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, count, n_tokens);
        ws->rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, k * n_tokens);

        size_t host_bytes = 0;
        for (int j = 0; j < n_planes; ++j) {
            const auto * w = layout[j];
            if (w) {
                ws->weights[j] = ggml_new_tensor_3d(ctx, w->type, w->ne[0], w->ne[1], count);
                ws->offsets[j] = host_bytes;
                host_bytes += ggml_nbytes(ws->weights[j]);
            }
        }
        host_bytes = GGML_PAD(host_bytes, sizeof(int32_t));
        ws->ids_offset = host_bytes;
        host_bytes += ggml_nbytes(ws->ids);
        ws->rows_offset = host_bytes;
        host_bytes += ggml_nbytes(ws->rows);

        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        if (ws->weights[3]) {
            auto * both = ggml_mul_mat_id(ctx, ws->weights[3], ws->input, ws->ids);
            const int64_t ff = both->ne[0] / 2;
            gate = ggml_view_3d(ctx, both, ff, count, n_tokens, both->nb[1], both->nb[2], 0);
            up = ggml_view_3d(ctx, both, ff, count, n_tokens, both->nb[1], both->nb[2], ff * both->nb[0]);
        } else {
            up = ggml_mul_mat_id(ctx, ws->weights[0], ws->input, ws->ids);
            gate = ggml_mul_mat_id(ctx, ws->weights[1], ws->input, ws->ids);
        }
        auto * act = ggml_swiglu_split(ctx, gate, up);
        auto * down = ggml_mul_mat_id(ctx, ws->weights[2], act, ws->ids);
        auto * flat = ggml_reshape_2d(ctx, down, d, count * n_tokens);
        auto * padded = ggml_pad(ctx, flat, 0, 1, 0, 0);
        auto * gathered = ggml_get_rows(ctx, padded, ws->rows);
        ws->result = ggml_reshape_3d(ctx, gathered, d, k, n_tokens);

        ws->graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(ws->graph, ws->result);
        for (int j = 0; j < ggml_graph_n_nodes(ws->graph); ++j) {
            if (!ggml_backend_supports_op(cuda, ggml_graph_node(ws->graph, j))) {
                return nullptr;
            }
        }

        const auto device = ggml_backend_get_device(cuda);
        const auto buft = ggml_backend_get_default_buffer_type(cuda);
        const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        if (!bytes || bytes > budget - allocated || free <= cuda_reserve || bytes > free - cuda_reserve) {
            return nullptr;
        }

        const auto host_buft = ggml_backend_dev_host_buffer_type(device);
        if (!host_buft) {
            return nullptr;
        }
        ws->device.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
        ws->host.reset(ggml_backend_buft_alloc_buffer(host_buft, host_bytes));
        ws->consumed.reset(ggml_backend_event_new(device));
        if (!ws->device || !ws->host || ggml_backend_buffer_get_type(ws->host.get()) != host_buft || !ws->consumed) {
            return nullptr;
        }

        auto * result = ws.get();
        workspaces.push_back(std::move(ws));
        allocated += bytes;
        return result;
    }
};

llama_rtx_expert::llama_rtx_expert(ggml_backend_t cuda, int32_t slots) : p(std::make_unique<impl>(cuda, slots)) {
}

llama_rtx_expert::~llama_rtx_expert() {
    for (const auto & ws : p->workspaces) {
        if (ws->staging_busy) {
            ggml_backend_event_synchronize(ws->consumed.get());
        }
    }
    if (p->submissions) {
        LLAMA_LOG_INFO("packed expert CUDA sidecar: submissions=%zu graph_launches=%zu device_bytes=%zu\n",
                p->submissions, p->graph_launches, p->allocated);
    }
}

ggml_backend_t llama_rtx_expert::backend() const {
    return p->cuda;
}

bool llama_rtx_expert::enabled(int il) const {
    return il >= 0 && static_cast<size_t>(il) < p->layers.size() && p->layers[il][2];
}

ggml_tensor * llama_rtx_expert::output(int il, ggml_tensor * map, ggml_tensor * cur,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down, ggml_tensor * gate_up, int64_t k, int64_t t) {
    if (il < 0 || !map || !cur || cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
            cur->ne[0] <= 0 || cur->ne[0] > 65536 || k <= 1 || k * n_tokens > max_ranks || t != n_tokens ||
            !down || (gate_up ? (up || gate) : (!up || !gate)) || !p->event_ready || !p->zero || !p->budget ||
            p->slots <= 0 || p->slots >= max_ranks) {
        return nullptr;
    }

    const planes layout{up, gate, down, gate_up};
    for (const auto * w : layout) {
        if (w && (!ggml_is_quantized(w->type) || !ggml_is_contiguous(w) || w->ne[0] > 65536 || w->ne[1] > 65536)) {
            return nullptr;
        }
    }
    const int64_t d = cur->ne[0];
    const int64_t ff = down->ne[0];
    if (down->ne[1] != d || (gate_up ? gate_up->ne[0] != d || gate_up->ne[1] != 2 * ff :
            up->ne[0] != d || up->ne[1] != ff || gate->ne[0] != d || gate->ne[1] != ff)) {
        return nullptr;
    }

    if (!p->contribution) {
        context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}), ggml_free);
        if (!ctx) {
            return nullptr;
        }
        auto * tensor = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, k, t);
        const auto buft = ggml_backend_get_default_buffer_type(p->cuda);
        const size_t bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        if (!bytes || bytes > p->budget) {
            return nullptr;
        }
        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft), ggml_backend_buffer_free);
        if (!buffer) {
            return nullptr;
        }
        ggml_set_name(tensor, "ffn_moe_rtx_contribution");
        p->zero(p->cuda, tensor);
        p->output_ctx = std::move(ctx);
        p->output_buffer = std::move(buffer);
        p->contribution = tensor;
        p->allocated = bytes;
    }

    if (p->contribution->ne[0] != d || p->contribution->ne[1] != k) {
        return nullptr;
    }
    if (p->layers.size() <= static_cast<size_t>(il)) {
        p->layers.resize(il + 1);
    }
    p->layers[il] = layout;
    return p->contribution;
}

bool llama_rtx_expert::available(ggml_backend_t vk, ggml_backend_sched_t sched, int il, const ggml_cgraph * graph, const ggml_tensor * map) {
    p->ready_input = nullptr;
    if (!enabled(il)) {
        return false;
    }
    const auto proc = reinterpret_cast<impl::vk_input_t>(
            ggml_backend_reg_get_proc_address(backend_registry(vk), "ggml_backend_vk_rtx_expert_input"));
    const auto * dst = proc ? proc(vk, graph, map, p->layers[il][3] ? 2 : 3) : nullptr;
    const auto * input = dst ? ggml_backend_sched_get_post_copy_source(sched, graph, map, dst, p->contribution) : nullptr;
    if (!input || !input->buffer || ggml_backend_buffer_get_type(input->buffer) != ggml_backend_get_default_buffer_type(p->cuda) ||
            input->type != GGML_TYPE_F32 || !ggml_is_contiguous(input) ||
            input->ne[0] != p->contribution->ne[0] || input->ne[1] != 1 || input->ne[2] != n_tokens || input->ne[3] != 1) {
        return false;
    }
    p->ready_graph = graph;
    p->ready_map = map;
    p->ready_input = input;
    p->ready_layer = il;
    return true;
}

bool llama_rtx_expert::submit(int il, const ggml_cgraph * graph, const ggml_tensor * map, const int32_t * slots,
        const void * const * host_slices, size_t n_experts) {
    if (!p->ready_input || graph != p->ready_graph || map != p->ready_map || il != p->ready_layer ||
            !slots || !host_slices || n_experts == 0 || n_experts > static_cast<size_t>(p->slots)) {
        return false;
    }
    const ggml_tensor * input = p->ready_input;
    p->ready_input = nullptr;
    const auto & layout = p->layers[il];
    const int32_t count = static_cast<int32_t>(n_experts);
    const int64_t k = p->contribution->ne[1];
    std::array<int32_t, max_ranks> rows{};
    for (int64_t rank = 0; rank < k * n_tokens; ++rank) {
        if (slots[rank] < -1 || slots[rank] >= count) {
            return false;
        }
        rows[rank] = slots[rank] < 0 ? count * n_tokens : (rank / k) * count + slots[rank];
    }

    for (size_t i = 0; i < n_experts; ++i) {
        for (int j = 0; j < n_planes; ++j) {
            if (layout[j] && !host_slices[i*n_planes + j]) {
                return false;
            }
        }
    }

    auto * ws = p->get_workspace(layout, count);
    if (!ws || (ws->staging_busy && !p->event_ready(p->cuda, ws->consumed.get()))) {
        return false;
    }
    auto * host = static_cast<uint8_t *>(ggml_backend_buffer_get_base(ws->host.get()));
    for (int j = 0; j < n_planes; ++j) {
        if (!ws->weights[j]) {
            continue;
        }
        const size_t bytes = ws->weights[j]->nb[2];
        auto * dst = host + ws->offsets[j];
        for (size_t i = 0; i < n_experts; ++i) {
            std::memcpy(dst + i * bytes, host_slices[i*n_planes + j], bytes);
        }
        ggml_backend_tensor_set_async(p->cuda, ws->weights[j], dst, 0, ggml_nbytes(ws->weights[j]));
    }

    auto * ids = reinterpret_cast<int32_t *>(host + ws->ids_offset);
    for (int t = 0; t < n_tokens; ++t) {
        for (int i = 0; i < count; ++i) {
            ids[t * count + i] = i;
        }
    }
    std::memcpy(host + ws->rows_offset, rows.data(), ggml_nbytes(ws->rows));
    ggml_backend_tensor_set_async(p->cuda, ws->ids, ids, 0, ggml_nbytes(ws->ids));
    ggml_backend_tensor_set_async(p->cuda, ws->rows, host + ws->rows_offset, 0, ggml_nbytes(ws->rows));
    ggml_backend_event_record(ws->consumed.get(), p->cuda);
    ws->staging_busy = true;
    ggml_backend_tensor_copy_async(p->cuda, p->cuda, input, ws->input);
    const auto status = ggml_backend_graph_compute_async(p->cuda, ws->graph);
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_copy_async(p->cuda, p->cuda, ws->result, p->contribution);
    ++p->submissions;
    ++p->graph_launches;
    return true;
}

void llama_rtx_expert::clear(int il, const ggml_cgraph *, const ggml_tensor * map) {
    if (enabled(il) && map) {
        p->zero(p->cuda, p->contribution);
    }
    p->ready_input = nullptr;
}
