#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static size_t align_up(size_t x, size_t align) {
    return (x + align - 1) / align * align;
}

// allocate 1D F32 tensors of the given sizes into a buffer of the given buft
static ggml_backend_buffer_ptr alloc_tensors(ggml_context * ctx, ggml_backend_buffer_type_t buft, const std::vector<size_t> & ne, std::vector<ggml_tensor *> & tensors) {
    const size_t align = ggml_backend_buft_get_alignment(buft);

    tensors.clear();
    std::vector<size_t> sizes(ne.size());
    std::vector<size_t> offsets(ne.size());

    size_t total = 0;
    for (size_t i = 0; i < ne.size(); i++) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[i]);
        tensors.push_back(t);
        sizes[i]    = ggml_backend_buft_get_alloc_size(buft, t);
        offsets[i]  = align_up(total, align);
        total       = offsets[i] + sizes[i];
    }

    ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(buft, total));
    if (!buf) {
        return nullptr;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf.get());
    for (size_t i = 0; i < tensors.size(); i++) {
        if (ggml_backend_tensor_alloc(buf.get(), tensors[i], base + offsets[i]) != GGML_STATUS_SUCCESS) {
            return nullptr;
        }
    }

    return buf;
}

static bool run_tensor_n_case(
        std::vector<ggml_tensor *> & tensors,
        const std::vector<size_t> & offsets,
        const std::vector<size_t> & sizes,
        const std::vector<size_t> & data_offsets) {
    const size_t n = tensors.size();

    std::vector<uint8_t> data(256, 0);
    std::vector<uint8_t> out(256, 0);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(data.data() + data_offsets[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets.data(), sizes.data(), data_offsets.data(), n);

    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        if (memcmp(data.data() + data_offsets[i], out.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    std::vector<uint8_t> slice(256, 0);
    for (size_t i = 0; i < n; i++) {
        if (sizes[i] == 0) {
            continue;
        }
        memset(slice.data(), 0, slice.size());
        ggml_backend_tensor_get(tensors[i], slice.data(), offsets[i], sizes[i]);
        if (memcmp(slice.data(), data.data() + data_offsets[i], sizes[i]) != 0) {
            return false;
        }
    }

    return true;
}

// verify batched tensor set/get on a single buffer type
static bool test_tensor_n(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 1024;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    ggml_backend_buffer_ptr buf = alloc_tensors(ctx.get(), buft, { 3, 2, 4, 3, 6 }, tensors);
    if (!buf) {
        return false;
    }

    ggml_tensor * t0    = tensors[0];
    ggml_tensor * t1    = tensors[1];
    ggml_tensor * t2    = tensors[2];
    ggml_tensor * t3    = tensors[3];
    ggml_tensor * t_big = tensors[4];

    {
        std::vector<ggml_tensor *> batch = { t0, t1, t2, t0 };
        std::vector<size_t> offsets = { 0, 0, 0, 0 };
        std::vector<size_t> sizes = { 3*4, 2*4, 4*4, 0 };
        std::vector<size_t> data_offsets = { 0, 3*4, 5*4, 64 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> batch = { t_big, t_big, t_big };
        std::vector<size_t> offsets = { 0, 8, 16 };
        std::vector<size_t> sizes = { 8, 8, 8 };
        std::vector<size_t> data_offsets = { 0, 8, 16 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<ggml_tensor *> batch = { t0, t3 };
        std::vector<size_t> offsets = { 0, 0 };
        std::vector<size_t> sizes = { 3*4, 3*4 };
        std::vector<size_t> data_offsets = { 0, 3*4 };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        uint8_t input[24], expected[128], out[128] = {};
        for (size_t i = 0; i < sizeof(input); i++) {
            input[i] = (uint8_t) (i + 1);
        }
        ggml_backend_tensor_set(t_big, input, 0, sizeof(input));

        // Verify scattered byte reads, including repeated source offsets.
        std::vector<ggml_tensor *> tensors(128, t_big);
        std::vector<size_t> offsets(128), sizes(128, 1), data_offsets(128);
        for (size_t i = 0; i < tensors.size(); i++) {
            offsets[i] = (i * 5) % sizeof(input);
            data_offsets[i] = i;
            expected[i] = input[offsets[i]];
        }

        ggml_backend_tensor_get_n(tensors.data(), out, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
        if (memcmp(out, expected, sizeof(out)) != 0) {
            return false;
        }
    }

    return true;
}

// verify batched tensor set/get and copy across two buffer types
static bool test_tensor_n_mixed_buft(ggml_backend_buffer_type_t buft) {
    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*16 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * t0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * t1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * t2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ggml_tensor * s0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
    ggml_tensor * s1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 2);
    ggml_tensor * s2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    const size_t align0 = ggml_backend_buft_get_alignment(buft);
    const size_t sz_t0 = ggml_backend_buft_get_alloc_size(buft, t0);
    const size_t sz_t2 = ggml_backend_buft_get_alloc_size(buft, t2);
    const size_t buf0_size = align_up(sz_t0, align0) + sz_t2;

    ggml_backend_buffer_ptr buf0(ggml_backend_buft_alloc_buffer(buft, buf0_size));
    if (!buf0) {
        return false;
    }

    char * base0 = (char *) ggml_backend_buffer_get_base(buf0.get());
    if (ggml_backend_tensor_alloc(buf0.get(), t0, base0) != GGML_STATUS_SUCCESS) {
        return false;
    }
    char * p2 = base0 + align_up(sz_t0, align0);
    if (ggml_backend_tensor_alloc(buf0.get(), t2, p2) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t host_align = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    std::vector<uint8_t> host(1024 + host_align, 0);
    uint8_t * host_ptr = (uint8_t *)(((uintptr_t)host.data() + host_align - 1) / host_align * host_align);
    ggml_backend_buffer_ptr buf1(ggml_backend_cpu_buffer_from_ptr(host_ptr, 1024));
    if (!buf1) {
        return false;
    }

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_buffer_get_type(buf1.get());
    const size_t align1 = ggml_backend_buft_get_alignment(cpu_buft);
    const size_t sz_t1 = ggml_backend_buft_get_alloc_size(cpu_buft, t1);
    const size_t sz_s0 = ggml_backend_buft_get_alloc_size(cpu_buft, s0);
    const size_t sz_s1 = ggml_backend_buft_get_alloc_size(cpu_buft, s1);

    char * base1 = (char *) ggml_backend_buffer_get_base(buf1.get());
    char * p = base1;
    if (ggml_backend_tensor_alloc(buf1.get(), t1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_t1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s0, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s0, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s1, p) != GGML_STATUS_SUCCESS) {
        return false;
    }
    p += align_up(sz_s1, align1);
    if (ggml_backend_tensor_alloc(buf1.get(), s2, p) != GGML_STATUS_SUCCESS) {
        return false;
    }

    uint8_t src0[12], src1[8], src2[16];
    memset(src0, 0xA1, sizeof(src0));
    memset(src1, 0xB2, sizeof(src1));
    memset(src2, 0xC3, sizeof(src2));

    ggml_backend_tensor_set(s0, src0, 0, sizeof(src0));
    ggml_backend_tensor_set(s1, src1, 0, sizeof(src1));
    ggml_backend_tensor_set(s2, src2, 0, sizeof(src2));

    {
        std::vector<ggml_tensor *> batch = { t0, t1, t2 };
        std::vector<size_t> offsets = { 0, 0, 0 };
        std::vector<size_t> sizes = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<size_t> data_offsets = { 0, sizeof(src0), sizeof(src0) + sizeof(src1) };

        if (!run_tensor_n_case(batch, offsets, sizes, data_offsets)) {
            return false;
        }
    }

    {
        std::vector<const ggml_tensor *> srcs = { s0, s1, s2 };
        std::vector<ggml_tensor *> dsts = { t0, t1, t2 };
        ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());

        const uint8_t * expected[3] = { src0, src1, src2 };
        const size_t sizes[3] = { sizeof(src0), sizeof(src1), sizeof(src2) };
        std::vector<uint8_t> out(64, 0);
        for (size_t i = 0; i < 3; i++) {
            memset(out.data(), 0, out.size());
            ggml_backend_tensor_get(dsts[i], out.data(), 0, sizes[i]);
            if (memcmp(out.data(), expected[i], sizes[i]) != 0) {
                return false;
            }
        }
    }

    return true;
}

// unified test set run on a single buffer type
static bool test_buft(ggml_backend_buffer_type_t buft) {
    return test_tensor_n(buft) && test_tensor_n_mixed_buft(buft);
}

//
// Mock buffer type
//
// Ad-hoc host-backed buffer type used to exercise the dispatch logic in
// ggml_backend_tensor_{set,get,copy}_n without relying on real backend hooks.
//

struct mock_config {
    bool set_n;
    bool get_n;
    bool cpy_n;
    bool cpy_n_result;
    int  set_n_calls;
    int  get_n_calls;
    int  cpy_n_calls;
};

static mock_config g_mock = {};

static void mock_config_reset(bool set_n, bool get_n, bool cpy_n, bool cpy_n_result) {
    g_mock.set_n = set_n;
    g_mock.get_n = get_n;
    g_mock.cpy_n = cpy_n;
    g_mock.cpy_n_result = cpy_n_result;
    g_mock.set_n_calls = 0;
    g_mock.get_n_calls = 0;
    g_mock.cpy_n_calls = 0;
}

static void mock_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * mock_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void mock_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void mock_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void mock_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool mock_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;

    GGML_UNUSED(buffer);
}

static void mock_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static void mock_buffer_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    g_mock.set_n_calls++;
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy((char *) tensors[i]->data + offsets[i], data[i], sizes[i]);
    }

    GGML_UNUSED(buffer);
}

static void mock_buffer_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    g_mock.get_n_calls++;
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy(data[i], (const char *) tensors[i]->data + offsets[i], sizes[i]);
    }

    GGML_UNUSED(buffer);
}

static bool mock_buffer_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    g_mock.cpy_n_calls++;
    if (!g_mock.cpy_n_result) {
        return false;
    }
    for (size_t i = 0; i < n_tensors; i++) {
        memcpy(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]));
    }
    return true;

    GGML_UNUSED(buffer);
}

static const char * mock_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "MOCK";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t mock_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = malloc(size);
    if (data == NULL) {
        return NULL;
    }

    struct ggml_backend_buffer_i iface = {
        /* .free_buffer     = */ mock_buffer_free_buffer,
        /* .get_base        = */ mock_buffer_get_base,
        /* .init_tensor     = */ NULL,
        /* .memset_tensor   = */ mock_buffer_memset_tensor,
        /* .set_tensor      = */ mock_buffer_set_tensor,
        /* .get_tensor      = */ mock_buffer_get_tensor,
        /* .set_tensor_2d   = */ NULL,
        /* .get_tensor_2d   = */ NULL,
        /* .cpy_tensor      = */ mock_buffer_cpy_tensor,
        /* .clear           = */ mock_buffer_clear,
        /* .reset           = */ NULL,
        /* .set_tensor_n    = */ NULL,
        /* .get_tensor_n    = */ NULL,
        /* .cpy_tensor_n    = */ NULL,
    };
    if (g_mock.set_n) iface.set_tensor_n = mock_buffer_set_tensor_n;
    if (g_mock.get_n) iface.get_tensor_n = mock_buffer_get_tensor_n;
    if (g_mock.cpy_n) iface.cpy_tensor_n = mock_buffer_cpy_tensor_n;

    return ggml_backend_buffer_init(buft, iface, data, size);
}

static size_t mock_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static bool mock_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t mock_buft(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name         = */ mock_buft_get_name,
            /* .alloc_buffer     = */ mock_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ mock_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ mock_buft_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    return &buft;
}

// verify that set_n/get_n route to the _n hook when installed and to the fallback otherwise
static bool test_mock_set_get_n(bool with_hooks) {
    mock_config_reset(with_hooks, with_hooks, false, true);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> tensors;
    ggml_backend_buffer_ptr buf = alloc_tensors(ctx.get(), mock_buft(), { 3, 2, 4 }, tensors);
    if (!buf) {
        return false;
    }

    const size_t sizes[3]     = { 3*4, 2*4, 4*4 };
    const size_t data_offs[3] = { 0, 3*4, 5*4 };
    size_t offsets[3]         = { 0, 0, 0 };
    std::vector<uint8_t> data(64, 0), out(64, 0);

    for (size_t i = 0; i < 3; i++) {
        memset(data.data() + data_offs[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets, sizes, data_offs, 3);
    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets, sizes, data_offs, 3);

    for (size_t i = 0; i < 3; i++) {
        if (memcmp(data.data() + data_offs[i], out.data() + data_offs[i], sizes[i]) != 0) {
            return false;
        }
    }

    const int expected_calls = with_hooks ? 1 : 0;
    if (g_mock.set_n_calls != expected_calls || g_mock.get_n_calls != expected_calls) {
        return false;
    }

    return true;
}

// verify that copy_n respects the all-or-nothing contract of cpy_tensor_n
static bool test_mock_copy_n(bool hook_result) {
    mock_config_reset(false, false, true, hook_result);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> src_tensors;
    std::vector<ggml_tensor *> dst_tensors;
    ggml_backend_buffer_ptr sbuf = alloc_tensors(ctx.get(), ggml_backend_cpu_buffer_type(), { 3, 2, 4 }, src_tensors);
    ggml_backend_buffer_ptr dbuf = alloc_tensors(ctx.get(), mock_buft(), { 3, 2, 4 }, dst_tensors);
    if (!sbuf || !dbuf) {
        return false;
    }

    uint8_t sdata[3][16] = {};
    memset(sdata[0], 0xA1, 3*4);
    memset(sdata[1], 0xB2, 2*4);
    memset(sdata[2], 0xC3, 4*4);

    uint8_t junk[16];
    memset(junk, 0xEE, sizeof(junk));

    for (size_t i = 0; i < 3; i++) {
        ggml_backend_tensor_set(src_tensors[i], sdata[i], 0, ggml_nbytes(src_tensors[i]));
        ggml_backend_tensor_set(dst_tensors[i], junk, 0, ggml_nbytes(dst_tensors[i]));
    }

    std::vector<const ggml_tensor *> srcs;
    for (const auto & t : src_tensors) {
        srcs.push_back(t);
    }

    ggml_backend_tensor_copy_n(srcs.data(), dst_tensors.data(), 3);

    for (size_t i = 0; i < 3; i++) {
        std::vector<uint8_t> out(16, 0);
        ggml_backend_tensor_get(dst_tensors[i], out.data(), 0, ggml_nbytes(dst_tensors[i]));
        if (memcmp(out.data(), sdata[i], ggml_nbytes(dst_tensors[i])) != 0) {
            return false;
        }
    }

    if (g_mock.cpy_n_calls != 1) {
        return false;
    }

    return true;
}

// verify that a batch spanning multiple buffer types is grouped by buft
static bool test_mock_grouping() {
    mock_config_reset(true, true, false, true);

    ggml_init_params params = {};
    params.mem_size = ggml_tensor_overhead()*32 + 2048;
    params.no_alloc = true;

    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    std::vector<ggml_tensor *> mock_tensors;
    std::vector<ggml_tensor *> cpu_tensors;
    ggml_backend_buffer_ptr mbuf = alloc_tensors(ctx.get(), mock_buft(), { 3, 4 }, mock_tensors);
    ggml_backend_buffer_ptr cbuf = alloc_tensors(ctx.get(), ggml_backend_cpu_buffer_type(), { 2 }, cpu_tensors);
    if (!mbuf || !cbuf) {
        return false;
    }

    // interleave mock/cpu/mock to exercise grouping
    std::vector<ggml_tensor *> tensors = { mock_tensors[0], cpu_tensors[0], mock_tensors[1] };
    std::vector<size_t> sizes       = { 3*4, 2*4, 4*4 };
    std::vector<size_t> data_offs   = { 0, 3*4, 5*4 };
    std::vector<size_t> offsets     = { 0, 0, 0 };
    std::vector<uint8_t> data(64, 0), out(64, 0);

    for (size_t i = 0; i < tensors.size(); i++) {
        memset(data.data() + data_offs[i], (int) (0x40 + i*0x10), sizes[i]);
    }

    ggml_backend_tensor_set_n(tensors.data(), data.data(), offsets.data(), sizes.data(), data_offs.data(), tensors.size());
    ggml_backend_tensor_get_n(tensors.data(), out.data(), offsets.data(), sizes.data(), data_offs.data(), tensors.size());

    for (size_t i = 0; i < tensors.size(); i++) {
        if (memcmp(data.data() + data_offs[i], out.data() + data_offs[i], sizes[i]) != 0) {
            return false;
        }
    }

    // mock group used the hook once, the cpu group did not
    if (g_mock.set_n_calls != 1 || g_mock.get_n_calls != 1) {
        return false;
    }

    return true;
}

static bool test_mock() {
    return
        test_mock_set_get_n(true)  &&
        test_mock_set_get_n(false) &&
        test_mock_copy_n(true)     &&
        test_mock_copy_n(false)    &&
        test_mock_grouping();
}

static bool test_cuda_hip_copy(bool timing) {
    constexpr size_t n_elements = 2560 * 3;
    const char * names[] = { "CUDA0", "ROCm0" };
    ggml_backend_ptr backends[2];

    ggml_init_params params = {};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        fprintf(stderr, "Failed to create tensor context\n");
        return false;
    }

    ggml_backend_buffer_ptr buffers[2];
    ggml_tensor * tensors[2];
    for (size_t i = 0; i < 2; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(names[i]);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            fprintf(stderr, "Required GPU %s is unavailable\n", names[i]);
            return false;
        }
        backends[i].reset(ggml_backend_dev_init(dev, nullptr));
        if (!backends[i]) {
            fprintf(stderr, "Failed to initialize %s\n", names[i]);
            return false;
        }

        auto buft = ggml_backend_get_default_buffer_type(backends[i].get());
        if (ggml_backend_buft_is_host(buft) || ggml_backend_buft_get_device(buft) != dev) {
            fprintf(stderr, "%s has no device-owned default buffer type\n", names[i]);
            return false;
        }
        std::vector<ggml_tensor *> allocated;
        buffers[i] = alloc_tensors(ctx.get(), buft, { n_elements }, allocated);
        if (!buffers[i]) {
            fprintf(stderr, "Failed to allocate a device tensor on %s\n", names[i]);
            return false;
        }
        tensors[i] = allocated[0];
    }

    const size_t nbytes = ggml_nbytes(tensors[0]);
    std::vector<uint8_t> expected(nbytes), actual(nbytes);
    bool ok = true;
    for (size_t src = 0; src < 2; ++src) {
        const size_t dst = 1 - src;
        for (size_t i = 0; i < nbytes; ++i) {
            expected[i] = (uint8_t) (i ^ (i >> 8) ^ (src ? 0xA5 : 0x5A));
            actual[i] = expected[i] ^ 0xFF;
        }
        ggml_backend_tensor_set(tensors[src], expected.data(), 0, nbytes);
        ggml_backend_tensor_set(tensors[dst], actual.data(), 0, nbytes);

        const auto copy = [&] {
            // Wait on each owning runtime before copying between them.
            ggml_backend_synchronize(backends[src].get());
            ggml_backend_synchronize(backends[dst].get());
            ggml_backend_tensor_copy(tensors[src], tensors[dst]);
        };
        copy();
        ggml_backend_tensor_get(tensors[dst], actual.data(), 0, nbytes);

        const bool matched = expected == actual;
        printf("%s -> %s (%zu bytes) %s\n", names[src], names[dst], nbytes, matched ? "PASSED" : "FAILED");
        ok = ok && matched;
        if (!timing || !matched) {
            continue;
        }

        for (uint8_t & byte : actual) {
            byte ^= 0xFF;
        }
        ggml_backend_tensor_set(tensors[dst], actual.data(), 0, nbytes);
        ggml_backend_synchronize(backends[dst].get());

        constexpr int repeats = 1000;
        const int64_t start_us = ggml_time_us();
        for (int i = 0; i < repeats; ++i) {
            copy();
        }
        const double mean_us = double(ggml_time_us() - start_us) / repeats;

        ggml_backend_tensor_get(tensors[dst], actual.data(), 0, nbytes);
        const bool matched_after_timing = expected == actual;
        printf("%s -> %s (%zu bytes) %.3f us/copy over %d copies, readback %s\n",
               names[src], names[dst], nbytes, mean_us, repeats, matched_after_timing ? "PASSED" : "FAILED");
        ok = ok && matched_after_timing;
    }

    return ok;
}

static bool test_cuda_hip_producer_d2h() {
    ggml_backend_ptr cuda(ggml_backend_init_by_name("CUDA0", nullptr));
    ggml_backend_ptr hip(ggml_backend_init_by_name("ROCm0", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cuda || !hip || !cpu) {
        fprintf(stderr, "Required CUDA, ROCm, and CPU backends are unavailable\n");
        return false;
    }

    constexpr size_t graph_size = 16;
    constexpr int64_t columns = 2560;
    constexpr int64_t rows = 3;
    ggml_context_ptr ctx(ggml_init({4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true}));
    if (!ctx) {
        return false;
    }
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, rows);
    auto * doubled = ggml_scale(ctx.get(), input, 2.0f);
    auto * tripled = ggml_scale(ctx.get(), input, 3.0f);
    auto * output = ggml_add(ctx.get(), doubled, tripled);
    ggml_set_input(input);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = {cuda.get(), hip.get(), cpu.get()};
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 3, graph_size, false, false));
    ggml_backend_sched_set_target_mtp_verify_three(sched.get(), true);
    ggml_backend_sched_set_tensor_backend(sched.get(), input, cuda.get());
    ggml_backend_sched_set_tensor_backend(sched.get(), doubled, cuda.get());
    ggml_backend_sched_set_tensor_backend(sched.get(), tripled, cuda.get());
    ggml_backend_sched_set_tensor_backend(sched.get(), output, hip.get());
    if (!ggml_backend_sched_alloc_graph(sched.get(), graph)) {
        return false;
    }

    std::vector<float> first(columns * rows, -7.0f);
    std::vector<float> second(columns * rows);
    for (size_t i = 0; i < second.size(); ++i) {
        second[i] = float(i % 31);
    }
    ggml_backend_hip_handoff_timing before[2], after[2];
    ggml_backend_get_hip_handoff_timing(before);
    bool ok = true;
    for (const auto * values : {&first, &second}) {
        ggml_backend_tensor_set_async(cuda.get(), input, values->data(), 0, ggml_nbytes(input));
        if (ggml_backend_sched_graph_compute_async(sched.get(), graph) != GGML_STATUS_SUCCESS) {
            ok = false;
            break;
        }
    }
    ggml_backend_sched_synchronize(sched.get());
    ggml_backend_get_hip_handoff_timing(after);
    // Both executions must use the pinned two-input batch.
    ok = ok && after[0].batches - before[0].batches == 2 && after[0].pinned_copies - before[0].pinned_copies == 4;
    std::vector<float> actual(second.size());
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    for (size_t i = 0; i < actual.size(); ++i) {
        ok = ok && actual[i] == second[i] * 5.0f;
    }
    printf("CUDA -> ROCm scheduler batch %s\n", ok ? "PASSED" : "FAILED");
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    if (argc == 2 && strcmp(argv[1], "--cuda-hip-producer-d2h") == 0) {
        return test_cuda_hip_producer_d2h() ? 0 : 1;
    }
    if (argc > 1 && strcmp(argv[1], "--cuda-hip-copy") == 0) {
        const bool timing = argc == 3 && strcmp(argv[2], "--timing") == 0;
        if (argc != 2 && !timing) {
            fprintf(stderr, "Usage: %s --cuda-hip-copy [--timing]\n", argv[0]);
            return 1;
        }
        return test_cuda_hip_copy(timing) ? 0 : 1;
    }

    // run the unified test set on each backend buffer type
    const size_t n_devs = ggml_backend_dev_count();
    size_t n_ok = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

        const bool ok = test_buft(buft);
        printf("%s %s\n", name, ok ? "PASSED" : "FAILED");
        if (ok) {
            n_ok++;
        }
    }

    printf("%zu/%zu backends passed\n", n_ok, n_devs);

    // run the unified test set and the dispatch tests on the mock buffer type
    mock_config_reset(true, true, true, true);
    const bool mock_ok = test_buft(mock_buft()) && test_mock();
    printf("MOCK %s\n", mock_ok ? "PASSED" : "FAILED");

    return n_ok == n_devs && mock_ok ? 0 : 1;
}
