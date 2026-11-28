// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-mtp.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-ranked-rtx.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <map>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

template <typename cb_get_buffer>
static void ggml_backend_tensor_buft_groups(size_t n_tensors, cb_get_buffer && get_buffer, std::vector<std::vector<size_t>> & groups) {
    std::unordered_map<ggml_backend_buffer_type_t, size_t> buft_idx;
    groups.clear();
    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_buffer_t buf = get_buffer(i);
        auto it = buft_idx.find(buf->buft);
        if (it == buft_idx.end()) {
            buft_idx[buf->buft] = groups.size();
            groups.emplace_back();
        }
        groups[buft_idx[buf->buft]].push_back(i);
    }
}

static bool ggml_backend_tensor_batch_is_2d(
        const struct ggml_tensor * const * tensors,
        const size_t * offsets,
        const size_t * sizes,
        const size_t * data_offsets,
        size_t n_tensors,
        size_t * base_offset,
        size_t * size,
        size_t * stride_tensor,
        size_t * stride_data) {
    if (n_tensors <= 1) {
        return false;
    }

    const struct ggml_tensor * tensor = tensors[0];
    const size_t size0 = sizes[0];

    if (size0 == 0 || offsets[1] < offsets[0] || data_offsets[1] < data_offsets[0]) {
        return false;
    }

    const size_t stride_tensor0 = offsets[1] - offsets[0];
    const size_t stride_data0   = data_offsets[1] - data_offsets[0];

    for (size_t i = 1; i < n_tensors; i++) {
        if (tensors[i] != tensor || sizes[i] != size0) {
            return false;
        }
        if (offsets[i] != offsets[0] + i*stride_tensor0 || data_offsets[i] != data_offsets[0] + i*stride_data0) {
            return false;
        }
    }

    *base_offset   = offsets[0];
    *size          = size0;
    *stride_tensor = stride_tensor0;
    *stride_data   = stride_data0;

    return true;
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set_n(struct ggml_tensor * const * tensors, const void * data,
        const size_t * offsets, const size_t * sizes, const size_t * data_offsets, size_t n_tensors) {
    if (n_tensors == 0) {
        return;
    }

    std::vector<const void *> data_ptrs(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        data_ptrs[i] = (const char *) data + data_offsets[i];
    }

    ggml_backend_buffer_t buf = NULL;
    bool same_buft = true;

    for (size_t i = 0; i < n_tensors; i++) {
        GGML_ASSERT(tensors[i] != NULL);
        ggml_backend_buffer_t tensor_buf = tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
        GGML_ASSERT(tensor_buf != NULL && "tensor buffer not set");
        GGML_ASSERT(tensors[i]->data != NULL && "tensor not allocated");
        GGML_ASSERT(offsets[i] + sizes[i] <= ggml_nbytes(tensors[i]) && "tensor write out of bounds");

        if (buf == NULL) {
            buf = tensor_buf;
        } else if (buf->buft != tensor_buf->buft) {
            same_buft = false;
        }
    }

    if (same_buft && buf->iface.set_tensor_n != NULL) {
        buf->iface.set_tensor_n(buf, tensors, data_ptrs.data(), offsets, sizes, n_tensors);
        return;
    }

    if (!same_buft) {
        std::vector<std::vector<size_t>> groups;
        ggml_backend_tensor_buft_groups(n_tensors, [&](size_t i) {
            return tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
        }, groups);

        for (size_t g = 0; g < groups.size(); g++) {
            std::vector<ggml_tensor *> g_tensors;
            std::vector<size_t> g_offsets;
            std::vector<size_t> g_sizes;
            std::vector<size_t> g_data_offsets;
            g_tensors.reserve(groups[g].size());
            g_offsets.reserve(groups[g].size());
            g_sizes.reserve(groups[g].size());
            g_data_offsets.reserve(groups[g].size());

            for (size_t i : groups[g]) {
                g_tensors.push_back(tensors[i]);
                g_offsets.push_back(offsets[i]);
                g_sizes.push_back(sizes[i]);
                g_data_offsets.push_back(data_offsets[i]);
            }

            ggml_backend_tensor_set_n(g_tensors.data(), data, g_offsets.data(), g_sizes.data(), g_data_offsets.data(), g_tensors.size());
        }
        return;
    }

    size_t base_offset = 0, size = 0, stride_tensor = 0, stride_data = 0;
    if (ggml_backend_tensor_batch_is_2d(tensors, offsets, sizes, data_offsets, n_tensors,
                &base_offset, &size, &stride_tensor, &stride_data)) {
        ggml_backend_tensor_set_2d(tensors[0], data_ptrs[0], base_offset, size,
                n_tensors, stride_tensor, stride_data);
        return;
    }

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_tensor_set(tensors[i], data_ptrs[i], offsets[i], sizes[i]);
    }
}

void ggml_backend_tensor_get_n(const struct ggml_tensor * const * tensors, void * data,
        const size_t * offsets, const size_t * sizes, const size_t * data_offsets, size_t n_tensors) {
    if (n_tensors == 0) {
        return;
    }

    std::vector<void *> data_ptrs(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        data_ptrs[i] = (char *) data + data_offsets[i];
    }

    ggml_backend_buffer_t buf = NULL;
    bool same_buft = true;

    for (size_t i = 0; i < n_tensors; i++) {
        GGML_ASSERT(tensors[i] != NULL);
        ggml_backend_buffer_t tensor_buf = tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
        GGML_ASSERT(tensor_buf != NULL && "tensor buffer not set");
        GGML_ASSERT(tensors[i]->data != NULL && "tensor not allocated");
        GGML_ASSERT(offsets[i] + sizes[i] <= ggml_nbytes(tensors[i]) && "tensor read out of bounds");

        if (buf == NULL) {
            buf = tensor_buf;
        } else if (buf->buft != tensor_buf->buft) {
            same_buft = false;
        }
    }

    if (same_buft && buf->iface.get_tensor_n != NULL) {
        buf->iface.get_tensor_n(buf, tensors, data_ptrs.data(), offsets, sizes, n_tensors);
        return;
    }

    if (!same_buft) {
        std::vector<std::vector<size_t>> groups;
        ggml_backend_tensor_buft_groups(n_tensors, [&](size_t i) {
            return tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
        }, groups);

        for (size_t g = 0; g < groups.size(); g++) {
            std::vector<const ggml_tensor *> g_tensors;
            std::vector<size_t> g_offsets;
            std::vector<size_t> g_sizes;
            std::vector<size_t> g_data_offsets;
            g_tensors.reserve(groups[g].size());
            g_offsets.reserve(groups[g].size());
            g_sizes.reserve(groups[g].size());
            g_data_offsets.reserve(groups[g].size());

            for (size_t i : groups[g]) {
                g_tensors.push_back(tensors[i]);
                g_offsets.push_back(offsets[i]);
                g_sizes.push_back(sizes[i]);
                g_data_offsets.push_back(data_offsets[i]);
            }

            ggml_backend_tensor_get_n(g_tensors.data(), data, g_offsets.data(), g_sizes.data(), g_data_offsets.data(), g_tensors.size());
        }
        return;
    }

    size_t base_offset = 0, size = 0, stride_tensor = 0, stride_data = 0;
    if (ggml_backend_tensor_batch_is_2d(tensors, offsets, sizes, data_offsets, n_tensors,
                &base_offset, &size, &stride_tensor, &stride_data)) {
        ggml_backend_tensor_get_2d(tensors[0], data_ptrs[0], base_offset, size,
                n_tensors, stride_tensor, stride_data);
        return;
    }

    for (size_t i = 0; i < n_tensors; i++) {
        ggml_backend_tensor_get(tensors[i], data_ptrs[i], offsets[i], sizes[i]);
    }
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

struct ggml_backend_handoff_times {
    bool enabled = false;
    int64_t source_wait_us = 0;
    int64_t destination_wait_us = 0;
    int64_t source_first_us = 0;
    int64_t source_last_us = 0;
    int64_t destination_first_us = 0;
    int64_t destination_last_us = 0;
    int64_t destination_second_us = 0;
    int64_t destination_second_end_us = 0;
    int64_t read_start_us = 0;
    int64_t read_end_us = 0;
    int64_t write_us = 0;
    int64_t write_start_us = 0;
    int64_t write_end_us = 0;

    template <typename F>
    void wait(int64_t & elapsed, int64_t & first, int64_t & last, F fn) {
        const int64_t start = enabled ? ggml_time_us() : 0;
        fn();
        if (enabled) {
            last = ggml_time_us();
            if (!first) {
                first = start;
            }
            elapsed += last - start;
        }
    }

    template <typename F>
    void source_wait(F fn) {
        wait(source_wait_us, source_first_us, source_last_us, fn);
    }

    template <typename F>
    void destination_wait(F fn) {
        if (destination_first_us) {
            wait(destination_wait_us, destination_second_us, destination_second_end_us, fn);
        } else {
            wait(destination_wait_us, destination_first_us, destination_last_us, fn);
        }
    }

    template <typename F>
    void write(F fn) {
        wait(write_us, write_start_us, write_end_us, fn);
    }

    template <typename F>
    void read(F fn) {
        if (enabled) {
            read_start_us = ggml_time_us();
        }
        fn();
        if (enabled) {
            read_end_us = ggml_time_us();
        }
    }
};

static bool ggml_backend_tensor_copy_impl(const struct ggml_tensor * src, struct ggml_tensor * dst, void * host_copy,
        ggml_backend_handoff_times * times = nullptr) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = host_copy ? host_copy : malloc(nbytes);
        if (times) {
            times->read([&] { ggml_backend_tensor_get(src, data, 0, nbytes); });
            times->write([&] { ggml_backend_tensor_set(dst, data, 0, nbytes); });
        } else {
            ggml_backend_tensor_get(src, data, 0, nbytes);
            ggml_backend_tensor_set(dst, data, 0, nbytes);
        }
        if (host_copy) {
            return true;
        }
        free(data);
    }
    return false;
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_copy_impl(src, dst, nullptr);
}

void ggml_backend_tensor_copy_n(const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    if (n_tensors == 0) {
        return;
    }

    for (size_t i = 0; i < n_tensors; ++i) {
        GGML_ASSERT(ggml_are_same_layout(srcs[i], dsts[i]) && "cannot copy tensors with different layouts");
    }

    ggml_backend_buffer_t buf = dsts[0]->view_src ? dsts[0]->view_src->buffer : dsts[0]->buffer;
    bool same_buft = true;
    for (size_t i = 0; i < n_tensors; ++i) {
        ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        if (dst_buf->buft != buf->buft) {
            same_buft = false;
            break;
        }
    }

    // if cpy_tensor_n returns false, it must not have performed any copies
    if (same_buft && buf->iface.cpy_tensor_n != NULL) {
        if (buf->iface.cpy_tensor_n(buf, srcs, dsts, n_tensors)) {
            return;
        }
    }

    if (!same_buft) {
        std::vector<std::vector<size_t>> groups;
        ggml_backend_tensor_buft_groups(n_tensors, [&](size_t i) {
            return dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
        }, groups);

        for (size_t g = 0; g < groups.size(); g++) {
            std::vector<const ggml_tensor *> g_srcs;
            std::vector<ggml_tensor *> g_dsts;
            g_srcs.reserve(groups[g].size());
            g_dsts.reserve(groups[g].size());

            for (size_t i : groups[g]) {
                g_srcs.push_back(srcs[i]);
                g_dsts.push_back(dsts[i]);
            }

            ggml_backend_tensor_copy_n(g_srcs.data(), g_dsts.data(), g_srcs.size());
        }
        return;
    }

    for (size_t i = 0; i < n_tensors; ++i) {
        ggml_backend_tensor_copy(srcs[i], dsts[i]);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
    uint64_t shared_plan_uid;
    uint64_t mapped_hip_return_uid;
    uint64_t mapped_hip_input_uid;
    uint64_t mapped_hip_input_after_shared_uid;
    ggml_tensor * mapped_hip_input;
    int next_router_begin;
    uint64_t next_router_uid;
    uint64_t without_router_uid;
    uint64_t without_router_mapped_uid;
    uint64_t without_router_after_shared_uid;
    uint64_t without_router_after_shared_mapped_uid;
    uint64_t carry_plan_uid;
    ggml_tensor * carry_plan_node;
    int carry_plan_end;
    bool has_custom;
    struct ggml_cgraph shared_prefix;
    struct ggml_cgraph after_shared;
    ggml_tensor * ranked_rtx_merge;
    uint64_t ranked_rtx_empty_uid[2][2][2];
};

struct ggml_backend_sched_cuda_profile {
    struct timing {
        size_t calls = 0;
        int64_t submit_us = 0;
        int64_t wait_us = 0;
    };
    std::map<std::string, timing> dispatches;
};

struct ggml_backend_sched_split_profile {
    ggml_backend_sched_cuda_profile * cuda;
    bool enabled;
    size_t graphs;
    int64_t graph_us;
    int64_t entry_wait_us;
    int64_t input_us;
    int64_t callback_us;
    size_t callback_count;
    size_t cross_backend_copies;
    size_t cross_backend_bytes;
    size_t host_to_device_inputs;
    int64_t host_to_device_us;
    size_t device_to_device_inputs;
    int64_t device_to_device_us;
    struct {
        char name[GGML_MAX_NAME];
        size_t splits;
        int64_t submit_us;
        int64_t wait_us;
    } backend[GGML_SCHED_MAX_BACKENDS];
};

struct ggml_backend_sched_host_copies {
    struct entry {
        const ggml_tensor * dst;
        size_t size;
        const void * payload;
        uint8_t data[256];
    };
    entry entries[GGML_SCHED_MAX_SPLIT_INPUTS];
    size_t count = 0;
    const float * single_miss_activation = nullptr;
    const ggml_tensor * single_miss_activation_dst = nullptr;
    const ggml_tensor * single_miss_weights_dst = nullptr;
    float single_miss_weights[30];
};

struct ggml_backend_sched_single_miss {
    const ggml_backend_sched_split * destination = nullptr;
    const ggml_tensor * output = nullptr;
    const float * (*resolve)(void *, int64_t) = nullptr;
    void * user_data = nullptr;
    unsigned mask = 0;
};

static std::atomic<int> hip_single_miss_cpu_mode{0};

void ggml_backend_sched_set_hip_single_miss_cpu(int mode) {
    if (mode == 0 || mode == 1) {
        hip_single_miss_cpu_mode.store(mode, std::memory_order_relaxed);
    }
}

using ggml_backend_cuda_get_shared_input_t = bool (*)(ggml_backend_t, const ggml_tensor * const *,
        ggml_backend_buffer_t, void *, const size_t *, const size_t *, const size_t *, size_t);

struct ggml_backend_sched_shared_host {
    ggml_backend_buffer_t cuda_buffer;
    ggml_backend_buffer_t vk_buffer;
    void * host;
    ggml_backend_t vk;
    ggml_backend_t cuda;
    bool tried;
};

static constexpr size_t shared_host_capacity = 128 * 1024;

using ggml_backend_register_writable_host_t = bool (*)(ggml_backend_dev_t, void *, size_t);
using ggml_backend_unregister_writable_host_t = void (*)(ggml_backend_dev_t, void *);
using ggml_backend_hip_graph_mapped_return_t = bool (*)(ggml_backend_t, ggml_cgraph *, uint64_t, void *, enum ggml_status *);
using ggml_backend_cuda_graph_mapped_hc_input_t = bool (*)(ggml_backend_t, ggml_cgraph *, uint64_t, const ggml_tensor *, void *, enum ggml_status *);

using ggml_backend_hip_mapped_input_blit_t = bool (*)(ggml_backend_t, ggml_tensor *, const void *);

struct ggml_backend_sched_dual_pinned {
    void * allocation;
    uint8_t * data;
    ggml_backend_dev_t owners[2];
    ggml_backend_unregister_writable_host_t unreg[2];
    bool tried;
    uint8_t * hip_mapped;
    ggml_backend_hip_graph_mapped_return_t hip_compute;
    uint8_t * cuda_mapped;
    uint8_t * cuda_input_data;
    ggml_backend_cuda_graph_mapped_hc_input_t cuda_compute;
};

struct ggml_backend_sched_hip_handoff_stats {
    size_t ops, copies, bytes;
    size_t pinned_copies, pinned_bytes, fallback_copies, fallback_bytes;
    int64_t source_wait_us, destination_wait_us;
    int64_t read_us, write_us;
    size_t async_input_batches, async_input_copies, async_input_bytes;
    size_t mapped_input_copies, mapped_input_bytes;
    int64_t enqueue_us;
    size_t async_return_copies, async_return_bytes;
    size_t mapped_return_copies, mapped_return_bytes;
    int64_t return_enqueue_us;
};

static thread_local ggml_backend_hip_handoff_timing hip_handoff_thread[2]{};

void ggml_backend_get_hip_handoff_timing(ggml_backend_hip_handoff_timing timing[2]) {
    std::copy_n(hip_handoff_thread, 2, timing);
}

struct ggml_backend_hip_verify_trace {
    static constexpr size_t capacity = 262144;
    struct row {
        uint64_t graph;
        int split;
        int backend;
        const char * stage;
        int64_t start_us;
        int64_t end_us;
    };

    const char * path;
    const std::string flush_path;
    std::vector<std::string> backend_names;
    row * rows = new (std::nothrow) row[capacity]{};
    size_t count = 0;
    size_t dropped = 0;
    size_t flushed = 0;
    uint64_t graphs = 0;

    explicit ggml_backend_hip_verify_trace(const char * path) : path(path), flush_path(std::string(path) + ".flush") {}

    void record(uint64_t graph, int split, int backend, const char * stage, int64_t start, int64_t end) {
        if (start == end) {
            return;
        }
        if (count < capacity) {
            rows[count++] = {graph, split, backend, stage, start, end};
        } else {
            ++dropped;
        }
    }

    ~ggml_backend_hip_verify_trace() { delete[] rows; }
};

struct ggml_backend_hip_verify_stage {
    ggml_backend_hip_verify_trace * trace;
    uint64_t graph;
    int split;
    int backend;
    const char * stage = nullptr;
    int64_t start = 0;

    void next(const char * next_stage) {
        if (!trace || (!stage && !next_stage)) {
            return;
        }
        const int64_t end = ggml_time_us();
        if (stage) {
            trace->record(graph, split, backend, stage, start, end);
        }
        start = end;
        stage = next_stage;
    }

    ~ggml_backend_hip_verify_stage() { next(nullptr); }
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids;   // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;        // [hash_set.size][n_backends][n_copies]
    bool                * hv_tensor_requires_copy; // [hash_set.size] decides if input needs to be copied to each split

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    ggml_backend_sched_copy_callback callback_copy;
    void * callback_copy_user_data;
    ggml_backend_sched_copy_callback callback_copy_early;
    void * callback_copy_early_user_data;
    bool capture_host_copies;
    bool measure_expert_copy_wait;
    bool defer_next_router;
    size_t next_router_plans;
    size_t next_router_deferred;
    size_t next_router_early;
    struct deferred_router {
        ggml_backend_t backend = nullptr;
        const ggml_backend_sched_split * target = nullptr;
        ggml_cgraph graph = {};
    } * active_deferred_router;
    ggml_backend_t deferred_expert_copy_backend;
    int64_t deferred_expert_copy_start_us;
    size_t expert_copy_wait_before;
    size_t expert_copy_wait_after_router;
    size_t expert_copy_wait_after_plan;
    size_t expert_copy_wait_fallback;
    int64_t expert_copy_wait_before_us;
    int64_t expert_copy_wait_after_us;
    int64_t expert_copy_cpu_before_wait_us;
    ggml_backend_sched_single_miss * active_single_miss;
    size_t single_miss_combined;
    int64_t single_miss_combine_us;
    const ggml_backend_sched_host_copies * active_host_copies;
    const ggml_backend_sched_split * active_copy_split;
    const ggml_backend_sched_split * ranked_rtx_empty_split;
    bool ranked_rtx_configured;
    size_t ranked_rtx_empty;
    size_t ranked_rtx_hot;
    size_t ranked_rtx_empty_fallback;
    size_t ranked_rtx_inactive_hot_lanes;
    struct after_submit {
        const ggml_backend_sched_split * split = nullptr;
        void (*callback)(ggml_backend_sched_t sched, void * user_data) = nullptr;
        void * user_data = nullptr;
    } * active_after_submit;
    void (*before_split_submit)(void * user_data);
    void * before_split_submit_user_data;
    int carry_prefix_end;
    ggml_tensor * carry_accumulator;
    ggml_custom_op_t custom_dispatch;
    ggml_backend_t custom_backend;
    struct ggml_backend_sched_split_profile profile;

    ggml_backend_sched_realloc_callback callback_realloc;
    void * callback_realloc_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;
    bool batch_host_inputs;
    size_t host_input_batches;
    size_t batched_host_inputs;
    bool batch_device_inputs;
    bool batch_hip_device_inputs;
    bool skip_hip_device_dst_wait;
    bool trace_handoff;
    bool overlap_shared;
    bool overlap_hip_shared;
    bool overlap_hip_mtp4_shared;
    size_t overlap_hip_shared_submits;
    size_t overlap_hip_mtp4_shared_submits;
    bool dual_pinned_hip_handoff;
    bool measure_hip_handoff;
    bool trace_hip_handoff_wait;
    bool async_hip_input;
    bool mapped_hip_input;
    ggml_backend_hip_mapped_input_blit_t hip_mapped_input_blit;
    size_t hip_mapped_input_read_eligible;
    size_t hip_mapped_input_read_applied;
    ggml_backend_event_t hip_return_h2d_event;
    ggml_backend_t hip_return_h2d_pending;
    size_t hip_return_h2d_applied;
    size_t hip_return_h2d_fallback;
    size_t hip_return_h2d_waits;
    ggml_backend_hip_verify_trace * hip_verify_trace;
    size_t mapped_hip_input_producers;
    size_t mapped_hip_input_after_shared_producers;
    size_t mapped_hip_input_busy;
    bool async_hip_mtp4_input;
    bool async_hip_return;
    bool async_hip_mtp4_return;
    ggml_backend_sched_hip_handoff_stats hip_mtp4_return_stats;
    ggml_backend_t hip_async_input_pending;
    ggml_backend_sched_dual_pinned hip_handoff;
    ggml_backend_sched_hip_handoff_stats hip_handoff_stats[2];
    bool shared_host_return;
    ggml_backend_sched_shared_host shared_return;
    ggml_backend_sched_shared_host shared_input;
    bool shared_host_input;
    bool shared_input_pending;
    bool shared_input_producer;
    ggml_backend_cuda_get_shared_input_t shared_input_get;
    bool shared_input_get_tried;
    bool target_mtp_verify;
    bool target_mtp_verify_three;
    bool target_mtp_verify_four;
    ggml_backend_sched_hip_handoff_stats hip_mtp4_input_stats;
    bool mtp_three_shared_input;
    bool mtp_three_router_upload;
    bool mtp_three_merge_empty_wave;
    bool mtp_auto_reset_shared_plan;
    size_t mtp_auto_stale_shared_prefixes;
    bool mtp_three_empty_next;
    size_t mtp_three_empty_merges;
    size_t mtp_three_empty_candidates;
    size_t mtp_three_empty_bad_split;
    size_t mtp_three_empty_bad_inputs;
    size_t mtp_three_empty_unpinned;
    size_t mtp_three_empty_bad_carry;
    size_t mtp_three_empty_bad_upload;
    size_t mtp_three_empty_alias;
    size_t mtp_three_router_upload_batches;
    size_t mtp_three_router_upload_bytes;
    bool mtp_three_shared_overlap;
    size_t mtp_three_shared_input_batches;
    size_t mtp_three_shared_input_bytes;
    size_t mtp_three_shared_overlap_submits;
    size_t mtp_three_shared_return_copies;
    bool routed_stats_enabled;
    ggml_backend_t routed_stats_backend;
    void (*set_routed_stats_active)(ggml_backend_t, bool);
    size_t routed_stats_rounds;
    size_t routed_stats_vulkan_aggregates;
    size_t routed_stats_other_backend;
    size_t routed_stats_other_op;
    size_t shared_input_verify_batches;
    size_t shared_input_other_batches;
    size_t shared_input_verify_bytes;
    size_t shared_input_other_bytes;
    size_t shared_input_optimized_verify;
    size_t shared_input_optimized_other;
    size_t shared_input_producer_eligible;
    size_t shared_input_producer_batches;
    size_t shared_input_producer_bytes;
    size_t shared_input_producer_fallbacks;
    size_t shared_input_batches;
    size_t shared_input_bytes;
    size_t shared_return_copies;
    size_t shared_return_bytes;
    size_t shared_overlap_submits;
    size_t shared_overlap_tokens;
    size_t trace_handoff_graphs;
    int64_t trace_handoff_graph;
    uint8_t * device_input_staging;
    size_t device_input_staging_size;
    size_t device_input_batches;
    size_t batched_device_inputs;
    size_t batched_device_input_bytes;
    size_t skipped_device_dst_waits;

    int debug;

    // graph size of the current and the previous scheduling, used by the realloc callback and for debugging
    // graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int graph_size;
    int prev_graph_size;

    int debug_realloc;

    // inputs of the next split read from the backend of the current split before it computes, see ggml_backend_sched_prefetch_inputs
    ggml_backend_event_t prefetch_events[GGML_SCHED_MAX_BACKENDS]; // created on first use, NULL if the backend has no events
    bool                 prefetch_tried [GGML_SCHED_MAX_BACKENDS];
    struct ggml_tensor * prefetched[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_prefetched;
    int prefetch_split;     // the split that reads the prefetched inputs
    int prefetch_backend_id; // the backend that copies them
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

static bool ggml_backend_sched_is_cuda_pair(ggml_backend_t src, ggml_backend_t dst, const char * peer);

static void ggml_backend_sched_plan_next_router(ggml_backend_sched_t sched, ggml_backend_sched_split & split,
        ggml_backend_sched_split & expert, const ggml_backend_graph_optimize_params & opt) {
    auto * cuda = sched->backends[split.backend_id];
    auto * hip = sched->backends[expert.backend_id];
    if (split.has_custom || expert.has_custom || !expert.graph.n_nodes ||
            strncmp(ggml_backend_name(cuda), "CUDA", 4) != 0 || !ggml_backend_sched_is_cuda_pair(cuda, hip, "ROCm")) {
        return;
    }
    constexpr int n_router_nodes = 5;
    auto & graph = split.graph;
    for (int i = 1; i + n_router_nodes <= graph.n_nodes; ++i) {
        auto ** nodes = graph.nodes + i;
        auto * logits = nodes[0];
        auto * probs = nodes[1];
        auto * order = nodes[2];
        auto * ranks = nodes[3];
        auto * ids = nodes[4];
        if (logits->op != GGML_OP_MUL_MAT || strncmp(logits->name, "ffn_next_router_logits-", 23) != 0 ||
                logits->type != GGML_TYPE_F32 || logits->ne[1] != 3 || !ggml_is_matrix(logits) ||
                !logits->src[0] || logits->src[0]->type != GGML_TYPE_F32 ||
                !logits->src[1] || logits->src[1]->type != GGML_TYPE_F32 ||
                probs->op != GGML_OP_SOFT_MAX || probs->src[0] != logits ||
                order->op != GGML_OP_ARGSORT || order->src[0] != probs ||
                ranks->op != GGML_OP_VIEW || ranks->view_src != order ||
                ids->op != GGML_OP_CONT || ids->src[0] != ranks || ids->type != GGML_TYPE_I32 ||
                !ggml_is_matrix(ids) || ids->ne[1] != 3 || !(ids->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                strncmp(ids->name, "ffn_next_router_topk-", 21) != 0) {
            continue;
        }
        // This chain must be a side output with no graph consumers.
        bool exclusive = ggml_node_get_use_count(&graph, i + n_router_nodes - 1) == 0;
        for (int j = 0; j < n_router_nodes - 1; ++j) {
            exclusive &= ggml_node_get_use_count(&graph, i + j) == 1;
        }
        if (!exclusive) {
            continue;
        }
        auto * until = expert.graph.nodes[expert.graph.n_nodes - 1];
        // Keep CUDA scratch and the activation backing allocation live until the HIP dispatch.
        for (int j = 0; j < n_router_nodes; ++j) {
            opt.add_alloc_dep(opt.user_data, nodes[j], until);
        }
        auto * activation = logits->src[1];
        opt.add_alloc_dep(opt.user_data, activation->view_src ? activation->view_src : activation, until);
        std::rotate(nodes, nodes + n_router_nodes, graph.nodes + graph.n_nodes);
        split.next_router_begin = graph.n_nodes - n_router_nodes;
        split.next_router_uid = ggml_graph_next_uid();
        split.without_router_uid = ggml_graph_next_uid();
        split.without_router_mapped_uid = ggml_graph_next_uid();
        split.without_router_after_shared_uid = ggml_graph_next_uid();
        split.without_router_after_shared_mapped_uid = ggml_graph_next_uid();
        ++sched->next_router_plans;
        return;
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    // inputs of nodes that are not computed (e.g. the unselected branches of ggml_build_forward_select)
                    // stay in the split so that the allocation does not change, but they do not need to be copied
                    if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                        sched->hv_tensor_requires_copy[src_id] = true;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);
        if (sched->mtp_auto_reset_shared_plan && sched->target_mtp_verify_three) {
            sched->mtp_auto_stale_shared_prefixes += split->shared_prefix.n_nodes != 0;
            split->shared_plan_uid = 0;
            split->shared_prefix = {};
            split->after_shared = {};
        }
        split->next_router_begin = 0;
        split->carry_plan_uid = 0;
        split->carry_plan_node = nullptr;
        split->carry_plan_end = 0;

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
        split->has_custom = false;
        if (sched->custom_dispatch && sched->backends[split->backend_id] == sched->custom_backend) {
            for (int j = 0; j < split->graph.n_nodes; ++j) {
                auto * node = split->graph.nodes[j];
                if (node->op == GGML_OP_CUSTOM) {
                    ggml_custom_op_params params;
                    memcpy(&params, node->op_params, sizeof(params));
                    split->has_custom |= params.fun == sched->custom_dispatch;
                }
            }
        }
    }

    if (sched->defer_next_router && sched->target_mtp_verify_three && !sched->profile.enabled && !sched->callback_eval) {
        for (int i = 0; i + 1 < sched->n_splits; ++i) {
            ggml_backend_sched_plan_next_router(sched, sched->splits[i], sched->splits[i + 1], opt_params);
        }
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later
    sched->prev_graph_size = sched->graph_size;
    sched->graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // The merged graph can upload this small copy before wave zero uses its work buffers.
            if (sched->mtp_three_merge_empty_wave && sched->target_mtp_verify_three &&
                    input->type == GGML_TYPE_F32 && input->ne[0] == 1 && input->ne[1] == 10 &&
                    input->ne[2] == 3 && input->ne[3] == 1 && strncmp(input->name, "ffn_moe_weights_", 16) == 0) {
                ggml_set_input(input_cpy);
                ggml_set_output(input_cpy);
            }

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        auto & split = sched->splits[i];
        split.graph.uid = ggml_graph_next_uid();
        split.ranked_rtx_merge = nullptr;
        if (sched->ranked_rtx_configured) {
            int tagged = 0;
            ggml_tensor * merge = nullptr;
            for (int j = 0; j < split.graph.n_nodes; ++j) {
                auto * node = split.graph.nodes[j];
                tagged += (node->flags & ggml_ranked_rtx::branch_flag) != 0;
                if (node->flags & ggml_ranked_rtx::merge_flag) {
                    merge = node;
                }
            }
            if (merge && merge->op == GGML_OP_ADD && tagged == merge->op_params[ggml_ranked_rtx::count_param]) {
                split.ranked_rtx_merge = merge;
                for (auto & shared : split.ranked_rtx_empty_uid) {
                    for (auto & deferred : shared) {
                        for (auto & uid : deferred) {
                            uid = ggml_graph_next_uid() | ggml_ranked_rtx::empty_uid_bit;
                        }
                    }
                }
            }
        }

        split.mapped_hip_return_uid = 0;
        split.mapped_hip_input_uid = 0;
        split.mapped_hip_input_after_shared_uid = 0;
        split.mapped_hip_input = nullptr;
        if (sched->mapped_hip_input && sched->target_mtp_verify_three && i + 1 < sched->n_splits) {
            const auto & next = sched->splits[i + 1];
            for (int j = 0; j < next.n_inputs; ++j) {
                auto * input = next.inputs[j];
                auto * producer = input->view_src ? input->view_src : input;
                if (input->type == GGML_TYPE_F32 && ggml_is_contiguous(input) && input->view_offs == 0 &&
                        ggml_nbytes(input) == size_t(2560 * 3 * sizeof(float)) &&
                        producer->op == GGML_OP_DSV4_HC_PRE && producer->ne[0] == 2560 && producer->ne[1] == 3 &&
                        ggml_is_matrix(producer) && ggml_is_contiguous(producer) && !producer->view_src &&
                        (producer->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_get_op_params_i32(producer, 1) == 1 &&
                        strncmp(producer->name, "hc_mixed-", sizeof("hc_mixed-") - 1) == 0 &&
                        std::find(split.graph.nodes, split.graph.nodes + split.graph.n_nodes, producer) != split.graph.nodes + split.graph.n_nodes) {
                    split.mapped_hip_input_uid = ggml_graph_next_uid();
                    split.mapped_hip_input = input;
                    break;
                }
            }
        }
        if (!sched->async_hip_return || !sched->target_mtp_verify_three || split.graph.n_nodes < 2 ||
                i + 1 == sched->n_splits) {
            continue;
        }
        const auto & next = sched->splits[i + 1];
        const int last = split.graph.n_nodes - 1;
        auto * out = split.graph.nodes[last];
        if (next.n_inputs != 1 || next.inputs[0] != out ||
                strncmp(out->name, "ffn_moe_out-", sizeof("ffn_moe_out-") - 1) != 0 ||
                out->op != GGML_OP_ADD || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
                out->ne[0] != 2560 || out->ne[1] != 3 || !ggml_is_matrix(out) ||
                !ggml_node_has_n_uses(&split.graph, last, 1)) {
            continue;
        }
        const auto * copy = tensor_copy(out, next.backend_id, sched->cur_copy);
        for (int j = 0; j < next.graph.n_nodes; ++j) {
            const auto * node = next.graph.nodes[j];
            if (node->op == GGML_OP_ADD && (node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                    node->src[0] == copy && node->type == GGML_TYPE_F32 && ggml_are_same_shape(node, out)) {
                split.mapped_hip_return_uid = ggml_graph_next_uid();
                break;
            }
        }
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        // a reallocation with an unchanged graph size means that the previous reservation was not sufficient
        const bool graph_size_changed = sched->prev_graph_size != sched->graph_size;

        if (sched->callback_realloc != NULL) {
            sched->callback_realloc(sched, &sched->graph, backend_ids_changed, graph_size_changed, sched->callback_realloc_user_data);
        }

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && !graph_size_changed;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static ggml_backend_buffer_t ggml_backend_sched_device_input_buffer(
        ggml_backend_t backend, const ggml_tensor * t, size_t size) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buffer || buffer != t->buffer || !t->data || !size || size % 4 != 0 ||
            ggml_backend_buffer_is_host(buffer) || buffer->buft != ggml_backend_get_default_buffer_type(backend) ||
            !ggml_is_contiguous(t)) {
        return nullptr;
    }
    const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(buffer);
    const uintptr_t addr = (uintptr_t) t->data;
    if (addr < base || (addr - base) % 4 != 0 || addr - base > buffer->size || size > buffer->size - (addr - base)) {
        return nullptr;
    }
    if (t->view_src) {
        const uintptr_t view_addr = (uintptr_t) t->view_src->data;
        if (t->view_src->view_src || view_addr < base || addr < view_addr || addr - view_addr != t->view_offs) {
            return nullptr;
        }
    }
    return buffer;
}

static bool ggml_backend_sched_is_cuda_pair(ggml_backend_t src, ggml_backend_t dst, const char * peer) {
    if (src == dst) {
        return false;
    }
    const auto src_dev = ggml_backend_get_device(src);
    const auto dst_dev = ggml_backend_get_device(dst);
    const auto src_reg = src_dev ? ggml_backend_dev_backend_reg(src_dev) : nullptr;
    const auto dst_reg = dst_dev ? ggml_backend_dev_backend_reg(dst_dev) : nullptr;
    if (!src_reg || !dst_reg) {
        return false;
    }
    const char * src_name = ggml_backend_reg_name(src_reg);
    const char * dst_name = ggml_backend_reg_name(dst_reg);
    return (strcmp(src_name, "CUDA") == 0 && strcmp(dst_name, peer) == 0) ||
            (strcmp(src_name, peer) == 0 && strcmp(dst_name, "CUDA") == 0);
}

// The CUDA and Vulkan copy hooks decline the other backend's standard device buffer without submitting work.
static bool ggml_backend_sched_is_cuda_vulkan_pair(ggml_backend_t src, ggml_backend_t dst) {
    return ggml_backend_sched_is_cuda_pair(src, dst, "Vulkan");
}

static bool ggml_backend_sched_can_map_hip_input(ggml_backend_dev_t dev) {
    const auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg || strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
        return false;
    }
    const auto is_ada = reinterpret_cast<bool (*)(ggml_backend_dev_t)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_is_ada"));
    return is_ada && is_ada(dev) &&
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_register_mapped_host") &&
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_host_mapping") &&
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_mapped_hc_input");
}

static ggml_backend_sched_dual_pinned ggml_backend_sched_try_dual_pinned(
        const ggml_backend_dev_t owners[2], size_t staging_capacity, bool mapped_input, bool mapped_return) {
    ggml_backend_sched_dual_pinned pinned{};
    constexpr size_t alignment = 64 * 1024;
    const size_t capacity = staging_capacity + (mapped_input ? shared_host_capacity : 0);
    auto * allocation = malloc(capacity + alignment - 1);
    if (!allocation) {
        if (!mapped_input) {
            GGML_LOG_WARN("packed HIP MTP3 dual-pinned handoff: host allocation failed, using ordinary staging\n");
        }
        return {};
    }
    const auto base = reinterpret_cast<uintptr_t>(allocation);
    auto * data = reinterpret_cast<uint8_t *>((base + alignment - 1) & ~(alignment - 1));
    for (int i = 0; i < 2; ++i) {
        const auto reg = ggml_backend_dev_backend_reg(owners[i]);
        const auto reg_host = reinterpret_cast<ggml_backend_register_writable_host_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_writable_host"));
        pinned.unreg[i] = reinterpret_cast<ggml_backend_unregister_writable_host_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_writable_host"));
        const bool try_mapped = i == 0 ? mapped_input : mapped_return;
        const auto reg_mapped = try_mapped ? reinterpret_cast<ggml_backend_register_writable_host_t>(
                ggml_backend_reg_get_proc_address(reg, i == 0 ? "ggml_backend_cuda_register_mapped_host" :
                        "ggml_backend_hip_register_mapped_host")) : nullptr;
        const bool mapped = pinned.unreg[i] && reg_mapped && reg_mapped(owners[i], data, capacity);
        // Retry with the original slab size before giving up either existing pinned path.
        const bool require_mapping = mapped_input && (i == 0 || reg_mapped);
        if (!mapped && (require_mapping || !reg_host || !pinned.unreg[i] || !reg_host(owners[i], data, capacity))) {
            if (!mapped_input) {
                GGML_LOG_WARN("packed HIP MTP3 dual-pinned handoff: %s registration unavailable, using ordinary staging\n",
                        ggml_backend_dev_name(owners[i]));
            }
            if (i == 1) {
                pinned.unreg[0](owners[0], data);
            }
            free(allocation);
            return {};
        }
        pinned.owners[i] = owners[i];
        if (mapped) {
            using get_mapping_t = void * (*)(ggml_backend_dev_t, void *);
            const auto get_mapping = reinterpret_cast<get_mapping_t>(
                    ggml_backend_reg_get_proc_address(reg, i == 0 ? "ggml_backend_cuda_get_host_mapping" :
                            "ggml_backend_hip_get_host_mapping"));
            auto * mapping = get_mapping ? static_cast<uint8_t *>(get_mapping(owners[i], data)) : nullptr;
            if (i == 0) {
                pinned.cuda_compute = reinterpret_cast<ggml_backend_cuda_graph_mapped_hc_input_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_mapped_hc_input"));
                pinned.cuda_mapped = pinned.cuda_compute ? mapping : nullptr;
                pinned.cuda_input_data = data + staging_capacity;
            } else {
                pinned.hip_compute = reinterpret_cast<ggml_backend_hip_graph_mapped_return_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_graph_mapped_return"));
                pinned.hip_mapped = pinned.hip_compute ? mapping : nullptr;
            }
            if (require_mapping && !(i == 0 ? pinned.cuda_mapped : pinned.hip_mapped)) {
                pinned.unreg[i](owners[i], data);
                if (i == 1) {
                    pinned.unreg[0](owners[0], data);
                }
                free(allocation);
                return {};
            }
        }
    }
    pinned.allocation = allocation;
    pinned.data = data;
    GGML_LOG_INFO("packed HIP MTP3 dual-pinned handoff: registered %zu bytes with %s and %s\n",
            capacity, ggml_backend_dev_name(owners[0]), ggml_backend_dev_name(owners[1]));
    return pinned;
}

static uint8_t * ggml_backend_sched_get_dual_pinned(ggml_backend_sched_t sched, ggml_backend_t src, ggml_backend_t dst) {
    auto & pinned = sched->hip_handoff;
    const bool from_cuda = strncmp(ggml_backend_name(src), "CUDA", 4) == 0;
    const ggml_backend_dev_t owners[] = {
        ggml_backend_get_device(from_cuda ? src : dst), ggml_backend_get_device(from_cuda ? dst : src),
    };
    if (pinned.tried) {
        return owners[0] == pinned.owners[0] && owners[1] == pinned.owners[1] ? pinned.data : nullptr;
    }
    const size_t staging_capacity = shared_host_capacity *
            (1 + (sched->async_hip_input || sched->async_hip_mtp4_input) +
             (sched->async_hip_return || sched->async_hip_mtp4_return));
    if (sched->mapped_hip_input && ggml_backend_sched_can_map_hip_input(owners[0])) {
        pinned = ggml_backend_sched_try_dual_pinned(owners, staging_capacity, true, sched->async_hip_return);
    }
    if (!pinned.data) {
        pinned = ggml_backend_sched_try_dual_pinned(owners, staging_capacity, false, sched->async_hip_return);
    }
    pinned.tried = true;
    return pinned.data;
}

static void ggml_backend_sched_record_hip_handoff(ggml_backend_sched_t sched, ggml_backend_t src,
        size_t count, size_t bytes, bool pinned, const ggml_backend_handoff_times & times,
        bool async_input = false, bool async_return = false, bool mtp4_input = false, bool mapped_return = false, size_t mapped_input_bytes = 0) {
    const int direction = strncmp(ggml_backend_name(src), "CUDA", 4) == 0 ? 0 : 1;
    const int64_t read_us = times.read_end_us - times.read_start_us;
    if (!sched->target_mtp_verify_four || mtp4_input || async_return) {
        auto & stats = !sched->target_mtp_verify_four ? sched->hip_handoff_stats[direction] :
                mtp4_input ? sched->hip_mtp4_input_stats : sched->hip_mtp4_return_stats;
        ++stats.ops;
        stats.copies += count;
        stats.bytes += bytes;
        stats.source_wait_us += times.source_wait_us;
        stats.destination_wait_us += times.destination_wait_us;
        stats.mapped_input_copies += mapped_input_bytes != 0;
        stats.mapped_input_bytes += mapped_input_bytes;
        if (async_return) {
            stats.async_return_copies += count;
            stats.async_return_bytes += bytes;
            stats.return_enqueue_us += read_us;
        } else if (mapped_return) {
            stats.mapped_return_copies += count;
            stats.mapped_return_bytes += bytes;
        } else {
            stats.read_us += read_us;
        }
        if (async_input) {
            ++stats.async_input_batches;
            stats.async_input_copies += count;
            stats.async_input_bytes += bytes;
            stats.enqueue_us += times.write_us;
        } else {
            stats.write_us += times.write_us;
        }
        if (pinned) {
            stats.pinned_copies += count;
            stats.pinned_bytes += bytes;
        } else if (sched->dual_pinned_hip_handoff) {
            stats.fallback_copies += count;
            stats.fallback_bytes += bytes;
        }
    }
    if (!sched->measure_hip_handoff) {
        return;
    }
    auto & timing = hip_handoff_thread[direction];
    ++timing.ops;
    timing.batches += count > 1;
    timing.copies += count;
    timing.bytes += bytes;
    timing.pinned_copies += pinned ? count : 0;
    timing.pinned_bytes += pinned ? bytes : 0;
    timing.source_wait_us += times.source_wait_us;
    timing.destination_wait_us += times.destination_wait_us;
    timing.read_us += async_return ? 0 : read_us;
    if (async_input) {
        ++timing.async_input_batches;
        timing.async_input_copies += count;
        timing.async_input_bytes += bytes;
        timing.enqueue_us += times.write_us;
    } else {
        timing.write_us += times.write_us;
    }
    if (sched->trace_hip_handoff_wait) {
        GGML_LOG_INFO("hip_handoff_wait: direction=%s copies=%zu bytes=%zu pinned=%d h2d_async=%d d2h_async=%d mapped=%d"
                " source_start_us=%lld source_end_us=%lld dst1_start_us=%lld dst1_end_us=%lld"
                " dst2_start_us=%lld dst2_end_us=%lld read_start_us=%lld read_end_us=%lld write_start_us=%lld write_end_us=%lld mapped_input_bytes=%zu\n",
                direction == 0 ? "CUDA->ROCm" : "ROCm->CUDA", count, bytes, pinned, async_input, async_return, mapped_return,
                (long long) times.source_first_us, (long long) times.source_last_us,
                (long long) times.destination_first_us, (long long) times.destination_last_us,
                (long long) times.destination_second_us, (long long) times.destination_second_end_us,
                (long long) times.read_start_us, (long long) times.read_end_us,
                (long long) times.write_start_us, (long long) times.write_end_us, mapped_input_bytes);
    }
}

static ggml_backend_sched_host_copies::entry * ggml_backend_sched_host_copy_entry(
        ggml_backend_sched_host_copies * copies, ggml_backend_t src, ggml_backend_t dst, const ggml_tensor * input) {
    if (!copies || copies->count == GGML_SCHED_MAX_SPLIT_INPUTS || src == dst ||
            input->type != GGML_TYPE_I32 || !ggml_is_contiguous(input)) {
        return nullptr;
    }
    auto & entry = copies->entries[copies->count];
    const size_t size = ggml_nbytes(input);
    if (size == 0 || size > sizeof(entry.data)) {
        return nullptr;
    }
    const auto src_dev = ggml_backend_get_device(src);
    const auto dst_dev = ggml_backend_get_device(dst);
    const auto src_reg = src_dev ? ggml_backend_dev_backend_reg(src_dev) : nullptr;
    const auto dst_reg = dst_dev ? ggml_backend_dev_backend_reg(dst_dev) : nullptr;
    if (!src_reg || !dst_reg || strcmp(ggml_backend_reg_name(src_reg), "CUDA") != 0) {
        return nullptr;
    }
    static const bool hip_host_router_ids = [] {
        const char * value = getenv("GGML_SCHED_HIP_HOST_ROUTER_IDS");
        return value && strcmp(value, "1") == 0;
    }();
    const char * dst_name = ggml_backend_reg_name(dst_reg);
    if (strcmp(dst_name, "Vulkan") != 0 && !(hip_host_router_ids && strcmp(dst_name, "ROCm") == 0)) {
        return nullptr;
    }
    entry.payload = entry.data;
    return &entry;
}

static void ggml_backend_sched_log_handoff(ggml_backend_sched_t sched, const ggml_backend_sched_split * split,
        ggml_backend_t src, ggml_backend_t dst, const ggml_tensor * const * inputs, int count, size_t bytes,
        const ggml_backend_handoff_times & times, const char * path) {
    if (!times.enabled || !sched->trace_handoff_graph) {
        return;
    }
    char names[GGML_SCHED_MAX_SPLIT_INPUTS * GGML_MAX_NAME] = {};
    char producers[GGML_SCHED_MAX_SPLIT_INPUTS * 16] = {};
    size_t used = 0;
    size_t producer_chars = 0;
    for (int i = 0; i < count; ++i) {
        const int written = snprintf(names + used, sizeof(names) - used, "%s%s", i == 0 ? "" : ",", inputs[i]->name);
        if (written < 0 || static_cast<size_t>(written) >= sizeof(names) - used) {
            break;
        }
        used += written;
        int producer = -1;
        for (int j = static_cast<int>(split - sched->splits) - 1; j >= 0; --j) {
            const auto & graph = sched->splits[j].graph;
            if (std::find(graph.nodes, graph.nodes + graph.n_nodes, inputs[i]) != graph.nodes + graph.n_nodes) {
                producer = j;
                break;
            }
        }
        const int added = snprintf(producers + producer_chars, sizeof(producers) - producer_chars, "%s%d", i ? "," : "", producer);
        if (added < 0 || static_cast<size_t>(added) >= sizeof(producers) - producer_chars) {
            break;
        }
        producer_chars += added;
    }
    GGML_LOG_INFO("expert_handoff: graph=%lld split=%td transfer=%s->%s path=%s tensors=%d bytes=%zu"
            " source_wait_us=%lld destination_wait_us=%lld read_start_us=%lld read_end_us=%lld write_us=%lld"
            " source_first_us=%lld source_last_us=%lld destination_first_us=%lld destination_last_us=%lld"
            " names=\"%s\" producer_splits=\"%s\"\n",
            (long long) sched->trace_handoff_graph, split - sched->splits, ggml_backend_name(src), ggml_backend_name(dst),
            path, count, bytes, (long long) times.source_wait_us, (long long) times.destination_wait_us,
            (long long) times.read_start_us, (long long) times.read_end_us, (long long) times.write_us,
            (long long) times.source_first_us, (long long) times.source_last_us,
            (long long) times.destination_first_us, (long long) times.destination_last_us, names, producers);
}

void ggml_backend_sched_trace_expert_misses(ggml_backend_sched_t sched, int layer, size_t tokens, size_t misses, size_t bytes) {
    if (sched && sched->trace_handoff_graph && sched->active_copy_split) {
        GGML_LOG_INFO("expert_handoff: graph=%lld split=%td layer=%d tokens=%zu unique_misses=%zu uploaded_bytes=%zu\n",
                (long long) sched->trace_handoff_graph, sched->active_copy_split - sched->splits, layer, tokens, misses, bytes);
    }
}

struct ggml_backend_sched_hip_return {
    ggml_backend_t source = nullptr;
    const ggml_tensor * input = nullptr;
    const ggml_tensor * copy = nullptr;
    uint8_t * data = nullptr;
    int64_t enqueue_start_us = 0;
    int64_t enqueue_end_us = 0;
    bool mapped = false;

    void wait() {
        if (source) {
            ggml_backend_synchronize(source);
            source = nullptr;
        }
    }

    ~ggml_backend_sched_hip_return() { wait(); }
};

struct ggml_backend_sched_mapped_hip_input {
    ggml_backend_t source = nullptr;
    const ggml_backend_sched_split * destination = nullptr;
    const ggml_tensor * input = nullptr;
    uint8_t * data = nullptr;
    void wait() {
        if (source) {
            ggml_backend_synchronize(source);
            source = nullptr;
        }
    }

    ~ggml_backend_sched_mapped_hip_input() { wait(); }
};

static void ggml_backend_sched_wait_hip_return_h2d(ggml_backend_sched_t sched) {
    if (sched->hip_return_h2d_pending) {
        ggml_backend_event_synchronize(sched->hip_return_h2d_event);
        sched->hip_return_h2d_pending = nullptr;
        ++sched->hip_return_h2d_waits;
    }
}

bool ggml_backend_sched_expert_copy_wait_pending(ggml_backend_sched_t sched) {
    return sched && sched->deferred_expert_copy_backend;
}

void ggml_backend_sched_wait_expert_copy(ggml_backend_sched_t sched, bool after_router, bool after_plan) {
    if (ggml_backend_sched_expert_copy_wait_pending(sched)) {
        const int64_t start = sched->measure_expert_copy_wait ? ggml_time_us() : 0;
        ggml_backend_synchronize(sched->deferred_expert_copy_backend);
        sched->deferred_expert_copy_backend = nullptr;
        if (sched->measure_expert_copy_wait) {
            sched->expert_copy_wait_after_us += ggml_time_us() - start;
            sched->expert_copy_wait_after_router += after_router;
            sched->expert_copy_wait_after_plan += after_plan;
            sched->expert_copy_wait_fallback += !after_router;
            if (after_router) {
                sched->expert_copy_cpu_before_wait_us += start - sched->deferred_expert_copy_start_us;
            }
        }
    }
}

static void ggml_backend_sched_copy_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split,
        struct ggml_tensor * input, ggml_backend_sched_host_copies * host_copies,
        ggml_backend_sched_hip_return * hip_return = nullptr) {
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, input);
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
    const bool use_hip_return = hip_return && hip_return->source && hip_return->source == input_backend &&
            hip_return->input == input && hip_return->copy == input_cpy;
    const bool measure_hip = (sched->measure_hip_handoff && sched->target_mtp_verify_three) ||
            (use_hip_return && sched->target_mtp_verify_four);
    const bool measure_mtp4 = sched->measure_hip_handoff && sched->target_mtp_verify_four;
    const bool cuda_hip_pair = input_backend &&
            (sched->skip_hip_device_dst_wait || sched->trace_handoff_graph || measure_hip || measure_mtp4) &&
            ggml_backend_sched_is_cuda_pair(input_backend, split_backend, "ROCm");
    ggml_backend_handoff_times handoff;
    handoff.enabled = sched->trace_handoff_graph && input_backend &&
            (cuda_hip_pair || ggml_backend_sched_is_cuda_vulkan_pair(input_backend, split_backend)) &&
            !ggml_backend_sched_is_host_weight(input);
    const size_t bytes = ggml_nbytes(input);
    const bool pinned_eligible = measure_hip && cuda_hip_pair && sched->n_copies == 1 &&
            !sched->events[split_backend_id][sched->cur_copy] && bytes <= shared_host_capacity &&
            ggml_are_same_layout(input, input_cpy) && !ggml_backend_sched_is_host_weight(input) &&
            ggml_backend_sched_device_input_buffer(input_backend, input, bytes) &&
            ggml_backend_sched_device_input_buffer(split_backend, input_cpy, bytes);
    const bool measure_mtp4_copy = measure_mtp4 && cuda_hip_pair && !ggml_backend_sched_is_host_weight(input);
    handoff.enabled |= pinned_eligible || measure_mtp4_copy;
    GGML_ASSERT(!use_hip_return || pinned_eligible);
    const ggml_tensor * trace_input = input;

    // read before the previous split, wait for the copy only
    if (split == &sched->splits[sched->prefetch_split]) {
        for (int i = 0; i < sched->n_prefetched; i++) {
            if (sched->prefetched[i] == input) {
                handoff.source_wait([&] { ggml_backend_event_synchronize(sched->prefetch_events[sched->prefetch_backend_id]); });
                ggml_backend_sched_log_handoff(sched, split, input_backend, split_backend, &trace_input, 1, 0, handoff, "prefetch_wait");
                return;
            }
        }
    }

    auto count_copy = [&]() {
        if (sched->profile.enabled && input_backend != split_backend) {
            sched->profile.cross_backend_copies++;
            sched->profile.cross_backend_bytes += ggml_nbytes(input);
        }
    };

    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
        } else {
            ggml_backend_synchronize(split_backend);
        }
        ggml_backend_tensor_copy(input, input_cpy);
        count_copy();
        return;
    }

    const bool weight_callback = sched->callback_copy && ggml_backend_sched_is_host_weight(input);
    const bool expert_wait_candidate = (sched->async_hip_input || sched->measure_expert_copy_wait) &&
            weight_callback && host_copies &&
            sched->n_copies == 1 && sched->target_mtp_verify_three && !sched->profile.enabled && !sched->callback_eval &&
            !split->has_custom && strncmp(input->name, "moe_cache.packed_slots-", 23) == 0 &&
            strncmp(ggml_backend_name(split_backend), "ROCm", 4) == 0;
    const bool captured_router = expert_wait_candidate &&
            std::any_of(host_copies->entries, host_copies->entries + host_copies->count,
                    [](const ggml_backend_sched_host_copies::entry & entry) {
                        return entry.size == 30 * sizeof(int32_t);
                    });
    if (captured_router && sched->async_hip_input) {
        GGML_ASSERT(!sched->deferred_expert_copy_backend);
        sched->deferred_expert_copy_backend = split_backend;
        sched->deferred_expert_copy_start_us = sched->measure_expert_copy_wait ? ggml_time_us() : 0;
    } else {
        // Finish earlier uses of the destination before overwriting it.
        const int64_t wait_start = captured_router && sched->measure_expert_copy_wait ? ggml_time_us() : 0;
        handoff.destination_wait([&] {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(split_backend);
            }
        });
        if (wait_start) {
            ++sched->expert_copy_wait_before;
            sched->expert_copy_wait_before_us += ggml_time_us() - wait_start;
        }
    }

    if (weight_callback) {
        const int64_t start_us = sched->profile.enabled ? ggml_time_us() : 0;
        bool handled;
        {
            struct host_copies_scope {
                ggml_backend_sched_t sched;
                ~host_copies_scope() {
                    ggml_backend_sched_wait_expert_copy(sched, false, false);
                    sched->active_host_copies = nullptr;
                    sched->active_copy_split = nullptr;
                }
            } scope{sched};
            sched->active_host_copies = host_copies;
            sched->active_copy_split = split;
            handled = sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data);
        }
        if (handled) {
            if (sched->profile.enabled) {
                ggml_backend_synchronize(split_backend);
                sched->profile.callback_us += ggml_time_us() - start_us;
                sched->profile.callback_count++;
            }
            return;
        }
    }

    count_copy();
    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
    if (use_hip_return || !split_backend->iface.cpy_tensor_async ||
            !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
        handoff.source_wait([&] {
            if (use_hip_return) {
                hip_return->wait();
            } else {
                ggml_backend_synchronize(input_backend);
            }
            if (sched->mapped_hip_input && sched->hip_async_input_pending == input_backend) {
                sched->hip_async_input_pending = nullptr;
            }
        });
        auto * correction = sched->active_single_miss;
        const bool combine_single_miss = use_hip_return && correction && correction->destination == split &&
                correction->output == input && correction->resolve;
        const int64_t mi50_done_us = combine_single_miss ? ggml_time_us() : 0;
        const bool skip_hip_wait = sched->skip_hip_device_dst_wait && cuda_hip_pair && ggml_are_same_layout(input, input_cpy);
        const char * host_path = "host";
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            handoff.destination_wait([&] { ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]); });
        } else if (sched->n_copies == 1 &&
                (skip_hip_wait || (sched->batch_device_inputs && ggml_backend_sched_is_cuda_vulkan_pair(input_backend, split_backend))) &&
                ggml_backend_sched_device_input_buffer(input_backend, input, ggml_nbytes(input)) &&
                ggml_backend_sched_device_input_buffer(split_backend, input_cpy, ggml_nbytes(input)) &&
                !ggml_backend_sched_is_host_weight(input)) {
            // The first destination wait already completed; the declined copy did not queue anything.
            ++sched->skipped_device_dst_waits;
            if (skip_hip_wait) {
                host_path = "host_skip_dst_wait";
            }
        } else {
            handoff.destination_wait([&] { ggml_backend_synchronize(split_backend); });
        }
        if (use_hip_return) {
            if (combine_single_miss) {
                const float * result = correction->resolve(correction->user_data, mi50_done_us);
                GGML_ASSERT(result);
                const int64_t combine_start = ggml_time_us();
                // Both GPU readers and writers of this return slot have completed.
                auto * output = reinterpret_cast<float *>(hip_return->data);
                for (int t = 0; t < 3; ++t) {
                    if (correction->mask & (1u << t)) {
                        for (int j = 0; j < 2560; ++j) {
                            const int i = t * 2560 + j;
                            output[i] += result[i];
                        }
                    }
                }
                *correction = {};
                ++sched->single_miss_combined;
                sched->single_miss_combine_us += ggml_time_us() - combine_start;
            }
            handoff.read_start_us = hip_return->enqueue_start_us;
            handoff.read_end_us = hip_return->enqueue_end_us;
            bool async_h2d = false;
            if (hip_return->mapped && sched->target_mtp_verify_three) {
                auto & event = sched->hip_return_h2d_event;
                if (!event && split_backend->iface.set_tensor_async) {
                    event = ggml_backend_event_new(split_backend->device);
                }
                async_h2d = event && event->device == split_backend->device && split_backend->iface.set_tensor_async;
                if (async_h2d) {
                    handoff.write([&] {
                        ggml_backend_tensor_set_async(split_backend, input_cpy, hip_return->data, 0, bytes);
                        ggml_backend_event_record(event, split_backend);
                    });
                    sched->hip_return_h2d_pending = split_backend;
                    ++sched->hip_return_h2d_applied;
                } else {
                    ++sched->hip_return_h2d_fallback;
                }
            }
            if (!async_h2d) {
                handoff.write([&] { ggml_backend_tensor_set(input_cpy, hip_return->data, 0, bytes); });
            }
            ggml_backend_sched_record_hip_handoff(sched, input_backend, 1, bytes, true, handoff,
                    false, !hip_return->mapped, false, hip_return->mapped);
            ggml_backend_sched_log_handoff(sched, split, input_backend, split_backend, &trace_input, 1,
                    bytes, handoff, async_h2d ? "host_dual_pinned_mapped_return_async_h2d" :
                    hip_return->mapped ? "host_dual_pinned_mapped_return" : "host_dual_pinned_async_return");
            return;
        }
        auto * entry = ggml_backend_sched_host_copy_entry(host_copies, input_backend, split_backend, input);
        auto * pinned = pinned_eligible && sched->dual_pinned_hip_handoff ?
                ggml_backend_sched_get_dual_pinned(sched, input_backend, split_backend) : nullptr;
        auto * host_copy = pinned ? pinned : entry ? entry->data : nullptr;
        const bool used_host = ggml_backend_tensor_copy_impl(input, input_cpy, host_copy, handoff.enabled ? &handoff : nullptr);
        if (entry && used_host) {
            if (pinned) {
                memcpy(entry->data, pinned, bytes);
            }
            entry->dst = input_cpy;
            entry->size = bytes;
            ++host_copies->count;
        }
        if ((pinned_eligible || measure_mtp4_copy) && handoff.read_end_us) {
            ggml_backend_sched_record_hip_handoff(sched, input_backend, 1, bytes, pinned != nullptr, handoff);
            if (pinned) {
                host_path = "host_dual_pinned";
            }
        }
        ggml_backend_sched_log_handoff(sched, split, input_backend, split_backend, &trace_input, 1,
                ggml_nbytes(input), handoff, handoff.read_end_us ? host_path : "buffer");
    } else {
        ggml_backend_sched_log_handoff(sched, split, input_backend, split_backend, &trace_input, 1, ggml_nbytes(input), handoff, "async");
        if (sched->profile.enabled) {
            ggml_backend_synchronize(split_backend);
        }
    }
}

static bool ggml_backend_sched_init_shared_host(ggml_backend_sched_shared_host & shared, ggml_backend_t vk, ggml_backend_t cuda) {
    if (shared.tried) {
        return shared.vk_buffer && vk == shared.vk && cuda == shared.cuda;
    }
    shared.tried = true;
    const auto dev_src = ggml_backend_get_device(vk);
    const auto buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(cuda));
    if (!dev_src->iface.buffer_from_host_ptr || !buft) {
        return false;
    }

    // Import a page-aligned range of the CUDA-owned host allocation.
    constexpr size_t alignment = 64 * 1024;
    constexpr size_t capacity = shared_host_capacity;
    auto * host_buffer = ggml_backend_buft_alloc_buffer(buft, capacity + alignment - 1);
    if (!host_buffer || ggml_backend_buffer_get_type(host_buffer) != buft ||
            ggml_backend_buffer_get_size(host_buffer) < capacity + alignment - 1) {
        ggml_backend_buffer_free(host_buffer);
        return false;
    }
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(host_buffer));
    auto * host = reinterpret_cast<void *>((base + alignment - 1) & ~(alignment - 1));
    auto * imported = ggml_backend_dev_buffer_from_host_ptr(dev_src, host, capacity, capacity);
    if (!imported) {
        ggml_backend_buffer_free(host_buffer);
        return false;
    }
    shared.cuda_buffer = host_buffer;
    shared.vk_buffer = imported;
    shared.host = host;
    shared.vk = vk;
    shared.cuda = cuda;
    return true;
}

using ggml_backend_vk_graph_upload_t = enum ggml_status (*)(ggml_backend_t, ggml_cgraph *,
        const ggml_backend_graph_upload *, const ggml_tensor *, ggml_backend_buffer_t, bool *);

struct ggml_backend_sched_upload {
    ggml_backend_graph_upload regions{};
    ggml_backend_vk_graph_upload_t compute = nullptr;
    size_t bytes = 0;
    bool mtp_three_router = false;
};

static bool ggml_backend_sched_is_mtp_three_router_input(ggml_backend_sched_t sched, const ggml_backend_sched_split * split,
        const ggml_tensor * src, const ggml_tensor * dst) {
    if (!sched->mtp_three_router_upload || !sched->target_mtp_verify_three || !sched->callback_copy ||
            src->type != GGML_TYPE_F32 || src->ne[0] != 1 || src->ne[1] != 10 || src->ne[2] != 3 || src->ne[3] != 1 ||
            strncmp(src->name, "ffn_moe_weights_", 16) != 0) {
        return false;
    }
    for (int i = 0; i < split->n_inputs; ++i) {
        auto * input = split->inputs[i];
        if (input != src && sched->hv_tensor_requires_copy[hash_id(input)] && !ggml_backend_sched_is_host_weight(input)) {
            return false;
        }
    }
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const auto * node = split->graph.nodes[i];
        if (node->op == GGML_OP_MUL && (node->flags & GGML_TENSOR_FLAG_COMPUTE) && node->src[1] == dst &&
                strncmp(node->name, "ffn_moe_weighted_mtp_three-", sizeof("ffn_moe_weighted_mtp_three-") - 1) == 0 &&
                node->src[0]->op == GGML_OP_ADD &&
                strcmp(node->src[0]->name, "ffn_moe_prefill_accumulator") == 0) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_sched_is_packed_moe_input(ggml_backend_sched_t sched, const ggml_backend_sched_split * split,
        const ggml_tensor * const * srcs, ggml_tensor * const * dsts, int count) {
    const bool three = sched->target_mtp_verify_three;
    if (!sched->callback_copy || (three ? !sched->mtp_three_shared_input || (count != 2 && count != 3) : count != 3)) {
        return false;
    }
    int actual = 0;
    const ggml_tensor * slot_map = nullptr;
    for (int i = 0; i < split->n_inputs; ++i) {
        auto * input = split->inputs[i];
        if (!sched->hv_tensor_requires_copy[hash_id(input)]) {
            continue;
        }
        if (!ggml_backend_sched_is_host_weight(input)) {
            ++actual;
        } else if (strncmp(input->name, "moe_cache.packed_slots-", 23) == 0) {
            constexpr const char wave_zero[] = "-mtp3-0";
            const size_t length = strlen(input->name);
            // Only the first wave introduces the CUDA inputs. The second may skip its expert dispatch entirely.
            if (!three || (length >= sizeof(wave_zero) - 1 &&
                    strcmp(input->name + length - (sizeof(wave_zero) - 1), wave_zero) == 0)) {
                slot_map = tensor_copy(input, split->backend_id, sched->cur_copy);
            }
        }
    }
    if (actual != count || !slot_map) {
        return false;
    }
    const int width = three ? 3 : 2;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * activation = nullptr;
    const ggml_tensor * weights = nullptr;
    for (int i = 0; i < count; ++i) {
        const auto * src = srcs[i];
        if (src->type == GGML_TYPE_I32 && ggml_is_vector(src) &&
                ggml_nbytes(src) <= 256 && strncmp(src->name, "ffn_moe_topk-", 13) == 0) {
            ids = dsts[i];
        } else if (src->type == GGML_TYPE_F32 && src->ne[2] == width && src->ne[3] == 1) {
            if (src->ne[1] == 1 && strncmp(src->name, "hc_mixed-", 9) == 0) {
                activation = dsts[i];
            } else if (src->ne[0] == 1 && strncmp(src->name, "ffn_moe_weights_", 16) == 0) {
                weights = dsts[i];
            }
        }
    }
    if (!ids || !activation || ids->ne[0] % width != 0 ||
            (count == 3 && (!weights || weights->ne[1] != ids->ne[0] / width))) {
        return false;
    }
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const auto * lookup = split->graph.nodes[i];
        if (lookup->op == GGML_OP_GET_ROWS && (lookup->flags & GGML_TENSOR_FLAG_COMPUTE) && lookup->src[1] == ids &&
                lookup->src[0] == slot_map) {
            return true;
        }
    }
    return false;
}

// Only merge adjacent waves for the same layer and only after the cache proves wave one empty.
static bool ggml_backend_sched_merge_empty_wave(ggml_backend_sched_t sched, ggml_backend_sched_split * first,
        ggml_backend_sched_upload & upload, std::vector<ggml_tensor *> & nodes, ggml_cgraph & merged) {
    if (!sched->mtp_three_empty_next) {
        return false;
    }
    ++sched->mtp_three_empty_candidates;
    if (first == &sched->splits[sched->n_splits - 1] || first->has_custom ||
            !upload.compute || !upload.regions.dst[1] || upload.regions.dst[2]) {
        ++sched->mtp_three_empty_bad_split;
        return false;
    }
    auto * next = first + 1;
    if (first->backend_id != next->backend_id || next->has_custom) {
        ++sched->mtp_three_empty_bad_split;
        return false;
    }
    const ggml_tensor * first_map = nullptr;
    for (int i = 0; i < first->n_inputs; ++i) {
        const auto * input = first->inputs[i];
        if (ggml_backend_sched_is_host_weight(input)) {
            if (first_map) {
                ++sched->mtp_three_empty_bad_inputs;
                return false;
            }
            first_map = input;
        }
    }
    const char * suffix = first_map ? strstr(first_map->name, "-mtp3-0") : nullptr;
    if (!suffix || strcmp(suffix, "-mtp3-0") != 0) {
        ++sched->mtp_three_empty_bad_inputs;
        return false;
    }
    ggml_tensor * router = nullptr;
    ggml_tensor * second_map = nullptr;
    for (int i = 0; i < next->n_inputs; ++i) {
        auto * input = next->inputs[i];
        if (!sched->hv_tensor_requires_copy[hash_id(input)]) {
            continue;
        }
        if (ggml_backend_sched_is_host_weight(input)) {
            if (second_map) {
                ++sched->mtp_three_empty_bad_inputs;
                return false;
            }
            second_map = input;
        } else {
            if (router) {
                ++sched->mtp_three_empty_bad_inputs;
                return false;
            }
            router = input;
        }
    }
    const size_t prefix = suffix - first_map->name;
    if (!second_map || strncmp(second_map->name, first_map->name, prefix) != 0 ||
            strcmp(second_map->name + prefix, "-mtp3-1") != 0 || !router ||
            (router->flags & GGML_TENSOR_FLAG_INPUT)) {
        ++sched->mtp_three_empty_bad_inputs;
        return false;
    }
    auto * dst = tensor_copy(router, next->backend_id, sched->cur_copy);
    if (!dst || !(dst->flags & GGML_TENSOR_FLAG_INPUT) || !(dst->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ++sched->mtp_three_empty_unpinned;
        return false;
    }
    if (!ggml_backend_sched_is_mtp_three_router_input(sched, next, router, dst)) {
        ++sched->mtp_three_empty_bad_inputs;
        return false;
    }
    ggml_tensor * accumulator = nullptr;
    for (int i = 0; i < next->graph.n_nodes; ++i) {
        auto * node = next->graph.nodes[i];
        if (node->op == GGML_OP_ADD && strcmp(node->name, "ffn_moe_prefill_accumulator") == 0) {
            accumulator = node;
            break;
        }
    }
    if (!accumulator || !accumulator->src[0] || accumulator->buffer != accumulator->src[0]->buffer ||
            accumulator->data != accumulator->src[0]->data ||
            std::find(first->graph.nodes, first->graph.nodes + first->graph.n_nodes, accumulator->src[0]) ==
                    first->graph.nodes + first->graph.n_nodes) {
        ++sched->mtp_three_empty_bad_carry;
        return false;
    }
    sched->active_copy_split = next;
    const bool carry = ggml_backend_sched_carry_accumulator(sched, &next->graph, accumulator);
    const int begin = sched->carry_prefix_end;
    sched->active_copy_split = nullptr;
    sched->carry_accumulator = nullptr;
    sched->carry_prefix_end = 0;
    if (!carry || begin >= next->graph.n_nodes || next->graph.nodes[begin]->op != GGML_OP_MUL ||
            next->graph.nodes[begin]->src[0] != accumulator || next->graph.nodes[begin]->src[1] != dst) {
        ++sched->mtp_three_empty_bad_carry;
        return false;
    }
    auto * backend = sched->backends[first->backend_id];
    auto * source = ggml_backend_sched_get_tensor_backend(sched, router);
    const size_t bytes = ggml_nbytes(router);
    if (source != sched->shared_input.cuda || !ggml_are_same_layout(router, dst) ||
            !ggml_backend_sched_device_input_buffer(source, router, bytes) ||
            bytes > shared_host_capacity - upload.bytes) {
        ++sched->mtp_three_empty_bad_upload;
        return false;
    }
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using supports_upload_t = bool (*)(ggml_backend_t, const ggml_backend_graph_upload *);
    const auto supports = reinterpret_cast<supports_upload_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_supports_graph_upload"));
    auto regions = upload.regions;
    regions.dst[2] = dst;
    regions.offsets[2] = upload.bytes;
    regions.single_submit_nodes = first->graph.n_nodes + next->graph.n_nodes - begin;
    if (!supports || !supports(backend, &regions)) {
        ++sched->mtp_three_empty_bad_upload;
        return false;
    }
    // Check the allocated ranges too, in case the allocator reused a previous graph layout.
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst->data);
    const auto aliases_router = [&](const ggml_tensor * tensor) {
        if (!tensor || tensor == dst || tensor->buffer != dst->buffer || !tensor->data) {
            return false;
        }
        const uintptr_t addr = reinterpret_cast<uintptr_t>(tensor->data);
        return addr < dst_addr ? dst_addr - addr < ggml_nbytes(tensor) : addr - dst_addr < bytes;
    };
    const auto aliases_graph = [&](const ggml_cgraph & graph, int start) {
        for (int i = start; i < graph.n_nodes; ++i) {
            if (aliases_router(graph.nodes[i])) {
                return true;
            }
            for (const auto * src : graph.nodes[i]->src) {
                if (aliases_router(src)) {
                    return true;
                }
            }
        }
        return false;
    };
    if (aliases_graph(first->graph, 0) || aliases_graph(next->graph, begin)) {
        ++sched->mtp_three_empty_alias;
        return false;
    }
    // Wave zero has already synchronized CUDA and joined prior readers of this staging buffer.
    ggml_backend_tensor_get(router, static_cast<uint8_t *>(sched->shared_input.host) + upload.bytes, 0, bytes);
    upload.regions = regions;
    nodes.assign(first->graph.nodes, first->graph.nodes + first->graph.n_nodes);
    nodes.insert(nodes.end(), next->graph.nodes + begin, next->graph.nodes + next->graph.n_nodes);
    merged = first->graph;
    merged.nodes = nodes.data();
    merged.n_nodes = nodes.size();
    merged.size = merged.n_nodes;
    return true;
}

static int ggml_backend_sched_batch_device_inputs(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, int first,
        ggml_backend_sched_host_copies * host_copies, ggml_backend_sched_upload & upload,
        ggml_backend_sched_mapped_hip_input & mapped_input) {
    const int backend_id = split->backend_id;
    if (sched->n_copies != 1 || sched->events[backend_id][sched->cur_copy]) {
        return 0;
    }
    ggml_backend_t dst_backend = sched->backends[backend_id];
    ggml_backend_t src_backend = nullptr;
    ggml_backend_buffer_t src_buffer = nullptr;
    ggml_backend_buffer_t dst_buffer = nullptr;
    const ggml_tensor * srcs[GGML_SCHED_MAX_SPLIT_INPUTS];
    ggml_tensor * dsts[GGML_SCHED_MAX_SPLIT_INPUTS];
    const size_t offsets[GGML_SCHED_MAX_SPLIT_INPUTS] = {};
    size_t sizes[GGML_SCHED_MAX_SPLIT_INPUTS];
    size_t data_offsets[GGML_SCHED_MAX_SPLIT_INPUTS];
    constexpr size_t max_bytes = 8 * 1024 * 1024;
    size_t bytes = 0;
    int count = 0;
    int last = first;

    for (int i = first; i < split->n_inputs && count < GGML_SCHED_MAX_SPLIT_INPUTS; ++i) {
        ggml_tensor * input = split->inputs[i];
        // These are not copied by the non-weight pass.
        if (!sched->hv_tensor_requires_copy[hash_id(input)] || ggml_backend_sched_is_host_weight(input)) {
            continue;
        }
        if (input->flags & GGML_TENSOR_FLAG_INPUT) {
            break;
        }
        ggml_backend_t src = ggml_backend_sched_get_tensor_backend(sched, input);
        if (!src || (count && src != src_backend)) {
            break;
        }
        const bool mtp4_input = sched->async_hip_mtp4_input && sched->target_mtp_verify_four &&
                strncmp(ggml_backend_name(src), "CUDA", 4) == 0;
        if (!count && !(sched->batch_device_inputs && ggml_backend_sched_is_cuda_vulkan_pair(src, dst_backend)) &&
                !((sched->batch_hip_device_inputs || mtp4_input) && ggml_backend_sched_is_cuda_pair(src, dst_backend, "ROCm"))) {
            break;
        }
        ggml_tensor * copy = tensor_copy(input, backend_id, sched->cur_copy);
        const size_t size = ggml_nbytes(input);
        if (size > max_bytes - bytes || !ggml_are_same_layout(input, copy)) {
            break;
        }
        const auto in = ggml_backend_sched_device_input_buffer(src, input, size);
        const auto out = ggml_backend_sched_device_input_buffer(dst_backend, copy, size);
        if (!in || !out || !in->iface.get_tensor_n || !out->iface.set_tensor_n ||
                (count && (in != src_buffer || out != dst_buffer))) {
            break;
        }

        // A native bulk setter may write the regions without barriers between copies.
        const uintptr_t addr = (uintptr_t) copy->data;
        bool overlaps = false;
        for (int j = 0; j < count && !overlaps; ++j) {
            const uintptr_t other = (uintptr_t) dsts[j]->data;
            overlaps = addr < other ? other - addr < size : addr - other < sizes[j];
        }
        if (overlaps) {
            break;
        }
        src_backend = src;
        src_buffer = in;
        dst_buffer = out;
        srcs[count] = input;
        dsts[count] = copy;
        sizes[count] = size;
        data_offsets[count] = bytes;
        bytes += size;
        last = i;
        ++count;
    }
    const bool vulkan_pair = count > 0 && ggml_backend_sched_is_cuda_vulkan_pair(src_backend, dst_backend);
    const bool router_upload = count == 1 && vulkan_pair &&
            ggml_backend_sched_is_mtp_three_router_input(sched, split, srcs[0], dsts[0]);
    if (count < 2 && !router_upload) {
        return 0;
    }

    if (vulkan_pair && sched->shared_host_input && host_copies && !sched->profile.enabled && !sched->callback_eval &&
            bytes <= shared_host_capacity &&
            (router_upload || ggml_backend_sched_is_packed_moe_input(sched, split, srcs, dsts, count))) {
        const auto src_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(src_backend));
        const auto dst_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(dst_backend));
        if (strcmp(ggml_backend_reg_name(src_reg), "CUDA") == 0 && strcmp(ggml_backend_reg_name(dst_reg), "Vulkan") == 0) {
            using supports_upload_t = bool (*)(ggml_backend_t, const ggml_backend_graph_upload *);
            auto supports = reinterpret_cast<supports_upload_t>(
                    ggml_backend_reg_get_proc_address(dst_reg, "ggml_backend_vk_supports_graph_upload"));
            auto compute_upload = reinterpret_cast<ggml_backend_vk_graph_upload_t>(
                    ggml_backend_reg_get_proc_address(dst_reg, "ggml_backend_vk_graph_upload"));
            if (supports && compute_upload && ggml_backend_sched_init_shared_host(sched->shared_input, dst_backend, src_backend)) {
                const ggml_backend_graph_upload regions{sched->shared_input.vk_buffer,
                    {dsts[0], count >= 2 ? dsts[1] : nullptr, count == 3 ? dsts[2] : nullptr},
                    {data_offsets[0], count >= 2 ? data_offsets[1] : 0, count == 3 ? data_offsets[2] : 0}, 0};
                if (supports(dst_backend, &regions)) {
                    upload = {regions, compute_upload, bytes, router_upload};
                }
            }
        }
    }

    if (router_upload && !upload.compute) {
        return 0;
    }
    const bool measure_hip = sched->measure_hip_handoff && sched->target_mtp_verify_three && !upload.compute &&
            bytes <= shared_host_capacity && ggml_backend_sched_is_cuda_pair(src_backend, dst_backend, "ROCm");
    const bool mtp4_input = sched->async_hip_mtp4_input && sched->target_mtp_verify_four &&
            strncmp(ggml_backend_name(src_backend), "CUDA", 4) == 0 &&
            ggml_backend_sched_is_cuda_pair(src_backend, dst_backend, "ROCm");
    const bool measure_mtp4 = sched->measure_hip_handoff && sched->target_mtp_verify_four && !upload.compute &&
            (mtp4_input || ggml_backend_sched_is_cuda_pair(src_backend, dst_backend, "ROCm"));
    auto * pinned = ((measure_hip && sched->dual_pinned_hip_handoff) || (mtp4_input && bytes <= shared_host_capacity)) ?
            ggml_backend_sched_get_dual_pinned(sched, src_backend, dst_backend) : nullptr;
    const bool async_input = pinned && (sched->async_hip_input || mtp4_input) && strncmp(ggml_backend_name(src_backend), "CUDA", 4) == 0 &&
            dst_backend->iface.set_tensor_async && !sched->profile.enabled && !sched->callback_eval &&
            (!sched->hip_async_input_pending || sched->hip_async_input_pending == dst_backend);
    if (!upload.compute && !pinned && sched->device_input_staging_size < bytes) {
        auto * staging = (uint8_t *) realloc(sched->device_input_staging, bytes);
        if (!staging) {
            return 0;
        }
        sched->device_input_staging = staging;
        sched->device_input_staging_size = bytes;
    }
    auto * staging = upload.compute ? static_cast<uint8_t *>(sched->shared_input.host) :
            pinned ? pinned + (async_input ? shared_host_capacity : 0) : sched->device_input_staging;
    int mirrored = -1;
    if (async_input && mapped_input.source == src_backend && mapped_input.destination == split) {
        for (int i = 0; i < count; ++i) {
            if (srcs[i] == mapped_input.input) {
                mirrored = i;
                break;
            }
        }
    }
    auto & profile = sched->profile;
    const int64_t start_us = profile.enabled ? ggml_time_us() : 0;

    ggml_backend_handoff_times handoff;
    handoff.enabled = sched->trace_handoff_graph != 0 || measure_hip || measure_mtp4 || mtp4_input;
    bool producer_copy = false;
    if (upload.compute && !sched->target_mtp_verify_three) {
        if (sched->target_mtp_verify) {
            ++sched->shared_input_verify_batches;
            sched->shared_input_verify_bytes += bytes;
        } else {
            ++sched->shared_input_other_batches;
            sched->shared_input_other_bytes += bytes;
        }
    }
    if (upload.compute && sched->shared_input_producer && sched->target_mtp_verify) {
        ++sched->shared_input_producer_eligible;
        if (!sched->shared_input_get_tried) {
            sched->shared_input_get_tried = true;
            auto * device = ggml_backend_get_device(src_backend);
            auto * reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
            if (reg && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
                sched->shared_input_get = reinterpret_cast<ggml_backend_cuda_get_shared_input_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_shared_input"));
            }
        }
        // Finish prior Vulkan staging reads and promotion commits before CUDA overwrites shared RAM.
        handoff.destination_wait([&] { ggml_backend_synchronize(dst_backend); });
        sched->shared_input_pending = false;
        if (sched->shared_input_get) {
            handoff.read([&] {
                producer_copy = sched->shared_input_get(src_backend, srcs, sched->shared_input.cuda_buffer,
                        staging, offsets, sizes, data_offsets, count);
            });
        }
        if (producer_copy) {
            ++sched->shared_input_producer_batches;
            sched->shared_input_producer_bytes += bytes;
            if (sched->target_mtp_verify) {
                ++sched->shared_input_optimized_verify;
            } else {
                ++sched->shared_input_optimized_other;
            }
        } else {
            ++sched->shared_input_producer_fallbacks;
            handoff.source_wait([&] { ggml_backend_synchronize(src_backend); });
            handoff.read([&] { ggml_backend_tensor_get_n(srcs, staging, offsets, sizes, data_offsets, count); });
        }
    } else {
        // Keep the ordinary synchronous bulk path for every other input layout and backend.
        handoff.source_wait([&] { ggml_backend_synchronize(src_backend); });
        if (sched->hip_return_h2d_pending == src_backend) {
            sched->hip_return_h2d_pending = nullptr;
        }
        handoff.destination_wait([&] { ggml_backend_synchronize(dst_backend); });
        if (sched->hip_async_input_pending == dst_backend) {
            // The destination wait drains the previous H2D or mapped read from the dedicated slab.
            sched->hip_async_input_pending = nullptr;
        }
        if (upload.compute) {
            sched->shared_input_pending = false;
        }
        handoff.read([&] {
            if (mirrored < 0) {
                ggml_backend_tensor_get_n(srcs, staging, offsets, sizes, data_offsets, count);
                return;
            }
            // Keep the remaining router data as one D2H batch, even if the activation was in the middle.
            std::swap(srcs[mirrored], srcs[count - 1]);
            std::swap(sizes[mirrored], sizes[count - 1]);
            std::swap(data_offsets[mirrored], data_offsets[count - 1]);
            ggml_backend_tensor_get_n(srcs, staging, offsets, sizes, data_offsets, count - 1);
            std::swap(data_offsets[mirrored], data_offsets[count - 1]);
            std::swap(sizes[mirrored], sizes[count - 1]);
            std::swap(srcs[mirrored], srcs[count - 1]);
        });
        if (mirrored >= 0) {
            mapped_input.source = nullptr;
        }
    }
    auto input_payload = [&](int i) {
        return i == mirrored ? mapped_input.data : staging + data_offsets[i];
    };
    if (async_input) {
        sched->hip_async_input_pending = dst_backend;
        handoff.write([&] {
            const auto & slab = sched->hip_handoff;
            const bool eligible = sched->hip_mapped_input_blit && mirrored >= 0 && sched->target_mtp_verify_three &&
                    slab.hip_mapped && mapped_input.data == slab.cuda_input_data && offsets[mirrored] == 0;
            for (int i = 0; i < count; ++i) {
                if (i == mirrored && eligible) {
                    ++sched->hip_mapped_input_read_eligible;
                    const auto * hip_input = slab.hip_mapped + (slab.cuda_input_data - slab.data);
                    if (sched->hip_mapped_input_blit(dst_backend, dsts[i], hip_input)) {
                        ++sched->hip_mapped_input_read_applied;
                        continue;
                    }
                }
                ggml_backend_tensor_set_async(dst_backend, dsts[i], input_payload(i), offsets[i], sizes[i]);
            }
        });
    } else if (!upload.compute) {
        handoff.write([&] { ggml_backend_tensor_set_n(dsts, staging, offsets, sizes, data_offsets, count); });
    }
    ggml_backend_sched_log_handoff(sched, split, src_backend, dst_backend, srcs, count, bytes, handoff,
            producer_copy ? "host_batch_shared_producer" : upload.compute ? "host_batch_shared" :
            mirrored >= 0 ? "host_batch_dual_pinned_mapped_input" :
            async_input ? "host_batch_dual_pinned_async_input" : pinned ? "host_batch_dual_pinned" : "host_batch");
    if (measure_hip || measure_mtp4 || mtp4_input) {
        ggml_backend_sched_record_hip_handoff(sched, src_backend, count, bytes, pinned != nullptr, handoff,
                async_input, false, mtp4_input, false, mirrored >= 0 ? sizes[mirrored] : 0);
    }
    if (mirrored >= 0 && host_copies && sched->target_mtp_verify_three &&
            hip_single_miss_cpu_mode.load(std::memory_order_relaxed) &&
            sizes[mirrored] == 2560 * 3 * sizeof(float) && offsets[mirrored] == 0 &&
            srcs[mirrored]->type == GGML_TYPE_F32 && srcs[mirrored]->ne[0] == 2560 && ggml_is_contiguous(srcs[mirrored])) {
        host_copies->single_miss_activation = reinterpret_cast<const float *>(mapped_input.data);
        host_copies->single_miss_activation_dst = dsts[mirrored];
        for (int i = 0; i < count; ++i) {
            const auto * t = srcs[i];
            if (t->type == GGML_TYPE_F32 && t->ne[0] == 1 && t->ne[1] == 10 && t->ne[2] == 3 &&
                    t->ne[3] == 1 && ggml_is_contiguous(t) && offsets[i] == 0 &&
                    sizes[i] == sizeof(host_copies->single_miss_weights) &&
                    strncmp(t->name, "ffn_moe_weights", 15) == 0) {
                if (host_copies->single_miss_weights_dst) {
                    host_copies->single_miss_weights_dst = nullptr;
                    break;
                }
                host_copies->single_miss_weights_dst = dsts[i];
                memcpy(host_copies->single_miss_weights, input_payload(i), sizes[i]);
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        if (auto * entry = ggml_backend_sched_host_copy_entry(host_copies, src_backend, dst_backend, srcs[i])) {
            if (upload.compute) {
                entry->payload = staging + data_offsets[i];
            } else {
                memcpy(entry->data, input_payload(i), sizes[i]);
            }
            entry->dst = dsts[i];
            entry->size = sizes[i];
            ++host_copies->count;
        }
    }
    ++sched->device_input_batches;
    sched->batched_device_inputs += count;
    sched->batched_device_input_bytes += bytes;
    if (profile.enabled) {
        profile.device_to_device_us += ggml_time_us() - start_us;
        profile.device_to_device_inputs += count;
        profile.cross_backend_copies += count;
        profile.cross_backend_bytes += bytes;
    }
    return last - first + 1;
}

static int ggml_backend_sched_batch_host_inputs(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split, int first) {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * copies[GGML_SCHED_MAX_SPLIT_INPUTS];
    const void * data[GGML_SCHED_MAX_SPLIT_INPUTS];
    const size_t offsets[GGML_SCHED_MAX_SPLIT_INPUTS] = {};
    size_t sizes[GGML_SCHED_MAX_SPLIT_INPUTS];
    const int backend_id = split->backend_id;
    int count = 0;
    size_t bytes = 0;

    // Batch consecutive user inputs so backend dependencies keep their original order.
    for (int i = first; i < split->n_inputs && count < GGML_SCHED_MAX_SPLIT_INPUTS; ++i) {
        ggml_tensor * input = split->inputs[i];
        if (!(input->flags & GGML_TENSOR_FLAG_INPUT) || !sched->hv_tensor_requires_copy[hash_id(input)] ||
                !input->buffer || !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_sched_is_host_weight(input)) {
            break;
        }

        ggml_tensor * copy = tensor_copy(input, backend_id, sched->cur_copy);
        const size_t size = ggml_nbytes(input);
        if (size == 0 || size % 4 != 0 || !input->data || !copy->data || copy->view_src || !copy->buffer ||
                !copy->buffer->iface.set_tensor_n || ggml_backend_buffer_is_host(copy->buffer) ||
                (buffer && buffer != copy->buffer)) {
            break;
        }
        GGML_ASSERT(ggml_are_same_layout(input, copy));

        // Native bulk uploads may issue the copies without barriers between them.
        bool overlaps = false;
        const uintptr_t addr = (uintptr_t) copy->data;
        for (int j = 0; j < count && !overlaps; ++j) {
            const uintptr_t other = (uintptr_t) copies[j]->data;
            overlaps = addr < other ? other - addr < size : addr - other < sizes[j];
        }
        if (overlaps) {
            break;
        }

        buffer = copy->buffer;
        copies[count] = copy;
        data[count] = input->data;
        sizes[count] = size;
        bytes += size;
        ++count;
    }
    if (count < 2) {
        return 0;
    }

    auto & profile = sched->profile;
    const int64_t start_us = profile.enabled ? ggml_time_us() : 0;
    ggml_backend_event_t event = sched->events[backend_id][sched->cur_copy];
    if (event) {
        ggml_backend_event_synchronize(event);
    } else {
        ggml_backend_synchronize(sched->backends[backend_id]);
    }

    // The bulk setter completes the writes before the caller can reuse the input data.
    buffer->iface.set_tensor_n(buffer, copies, data, offsets, sizes, count);
    sched->host_input_batches++;
    sched->batched_host_inputs += count;
    if (profile.enabled) {
        profile.host_to_device_us += ggml_time_us() - start_us;
        profile.host_to_device_inputs += count;
        profile.cross_backend_copies += count;
        profile.cross_backend_bytes += bytes;
    }
    return count;
}

// the inputs of the next split that are computed on the backend of this split before it, and copied to host memory,
// are copied before this split computes, so that the next split waits for these copies instead of this split
// e.g. a CPU split runs while a GPU split computes, if it reads only the results of the earlier splits
static void ggml_backend_sched_prefetch_inputs(ggml_backend_sched_t sched, int split_id) {
    sched->n_prefetched = 0;
    if (split_id + 1 >= sched->n_splits) {
        return;
    }
    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    struct ggml_backend_sched_split * next  = &sched->splits[split_id + 1];
    const int backend_id = split->backend_id;
    ggml_backend_t backend = sched->backends[backend_id];
    if (backend->iface.get_tensor_async == NULL) {
        return;
    }

    for (int i = 0; i < next->n_inputs && sched->n_prefetched < GGML_SCHED_MAX_SPLIT_INPUTS; i++) {
        struct ggml_tensor * input = next->inputs[i];
        if ((input->flags & GGML_TENSOR_FLAG_INPUT) || tensor_backend_id(input) != backend_id) {
            continue;
        }
        struct ggml_tensor * input_cpy = tensor_copy(input, next->backend_id, sched->cur_copy);
        if (input_cpy->buffer == NULL || !ggml_backend_buffer_is_host(input_cpy->buffer)) {
            continue;
        }

        // only results of earlier splits, the copy callback of this split runs later and can write other tensors (e.g. the MoE cache)
        const struct ggml_tensor * base = input->view_src ? input->view_src : input;
        if (base->buffer == NULL || ggml_backend_buffer_get_usage(base->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }

        // not if this split computes or writes the input
        bool written = false;
        for (int j = 0; j < split->graph.n_nodes && !written; j++) {
            const struct ggml_tensor * node = split->graph.nodes[j];
            written = !ggml_is_view_op(node->op) && (node == base || node->view_src == base);
        }
        if (written) {
            continue;
        }

        if (!sched->prefetch_tried[backend_id]) {
            sched->prefetch_tried[backend_id]  = true;
            sched->prefetch_events[backend_id] = ggml_backend_event_new(backend->device);
        }
        if (sched->prefetch_events[backend_id] == NULL) {
            return;
        }

        ggml_backend_tensor_get_async(backend, input, input_cpy->data, 0, ggml_nbytes(input));
        sched->prefetched[sched->n_prefetched++] = input;
    }

    if (sched->n_prefetched > 0) {
        sched->prefetch_split      = split_id + 1;
        sched->prefetch_backend_id = backend_id;
        ggml_backend_event_record(sched->prefetch_events[backend_id], backend);
    }
}

static void ggml_backend_sched_trace_submit(ggml_backend_sched_t sched, int split_id, ggml_backend_t split_backend,
        const ggml_cgraph * graph, int64_t submit_start_us, int64_t submit_end_us) {
    int anchors = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto * node = graph->nodes[i];
        if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        const char * weight = node->op == GGML_OP_MUL_MAT && node->src[0] ? node->src[0]->name : "";
        const bool shared = strncmp(node->name, "ffn_shexp", 9) == 0 ||
                strncmp(node->name, "shared_expert_gate", 18) == 0 ||
                strstr(weight, "_shexp.") != nullptr;
        const bool routed = node->op == GGML_OP_MUL_MAT_ID && strncmp(node->name, "ffn_moe_", 8) == 0;
        if (shared || routed) {
            ++anchors;
            GGML_LOG_INFO("expert_handoff: graph=%lld split=%d backend=%s role=%s submit_start_us=%lld submit_end_us=%lld node=\"%s\" weight=\"%s\"\n",
                    (long long) sched->trace_handoff_graph, split_id, ggml_backend_name(split_backend),
                    shared ? "shared" : "routed", (long long) submit_start_us, (long long) submit_end_us, node->name, weight);
        }
    }
    if (!anchors) {
        GGML_LOG_INFO("expert_handoff: graph=%lld split=%d backend=%s role=unclassified submit_start_us=%lld submit_end_us=%lld\n",
                (long long) sched->trace_handoff_graph, split_id, ggml_backend_name(split_backend),
                (long long) submit_start_us, (long long) submit_end_us);
    }
}

static bool ggml_backend_sched_shared_range(const ggml_tensor * tensor, const ggml_tensor * copied,
        ggml_backend_buffer_type_t buft) {
    auto * buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buf || ggml_backend_buffer_get_type(buf) != buft || !tensor->data) {
        return false;
    }
    const auto a = reinterpret_cast<uintptr_t>(tensor->data);
    const auto b = reinterpret_cast<uintptr_t>(copied->data);
    return a < b ? b - a >= ggml_nbytes(tensor) : a - b >= ggml_nbytes(copied);
}

static bool ggml_backend_sched_plan_shared(ggml_backend_sched_t sched, ggml_backend_sched_split & split,
        ggml_backend_sched_split & routed, bool allow_hip = false) {
    if (split.n_inputs != 1) {
        return false;
    }
    auto * routed_out = split.inputs[0];
    const int64_t width = routed_out->ne[1];
    const bool three = width == 3;
    ggml_backend_t cuda = sched->backends[split.backend_id];
    ggml_backend_t expert = sched->backends[routed.backend_id];
    const bool hip = allow_hip && ((sched->overlap_hip_shared && sched->target_mtp_verify_three && three) ||
            (sched->overlap_hip_mtp4_shared && sched->target_mtp_verify_four && width == 4)) &&
            ggml_backend_sched_is_cuda_pair(expert, cuda, "ROCm");
    const bool vulkan = sched->overlap_shared && ggml_backend_sched_is_cuda_vulkan_pair(expert, cuda) &&
            (width == 2 || (three && sched->target_mtp_verify_three && sched->mtp_three_shared_overlap));
    if ((!hip && !vulkan) || strncmp(ggml_backend_name(cuda), "CUDA", 4) != 0) {
        return false;
    }
    if (split.shared_plan_uid == split.graph.uid) {
        return split.shared_prefix.n_nodes != 0;
    }
    split.shared_plan_uid = split.graph.uid;
    split.mapped_hip_input_after_shared_uid = 0;
    split.shared_prefix = {};
    constexpr const char routed_name[] = "ffn_moe_out-";
    if (strncmp(routed_out->name, routed_name, sizeof(routed_name) - 1) != 0 ||
            routed_out->type != GGML_TYPE_F32 || (hip && routed_out->ne[0] != 2560) || !ggml_is_matrix(routed_out) ||
            std::find(routed.graph.nodes, routed.graph.nodes + routed.graph.n_nodes, routed_out) == routed.graph.nodes + routed.graph.n_nodes ||
            !sched->hv_tensor_requires_copy[hash_id(routed_out)]) {
        return false;
    }
    const auto * copied = tensor_copy(split.inputs[0], split.backend_id, sched->cur_copy);
    const auto buft = ggml_backend_get_default_buffer_type(cuda);
    if (!copied->data || !copied->buffer || ggml_backend_buffer_get_type(copied->buffer) != buft) {
        return false;
    }
    const bool verify = three || hip;
    if (verify && (split.has_custom || routed.has_custom || (routed_out->flags & GGML_TENSOR_FLAG_INPUT) ||
            sched->events[split.backend_id][sched->cur_copy] ||
            !ggml_are_same_layout(routed_out, copied) ||
            !ggml_backend_sched_device_input_buffer(expert, routed_out, ggml_nbytes(routed_out)) ||
            !ggml_backend_sched_device_input_buffer(cuda, copied, ggml_nbytes(copied)))) {
        return false;
    }

    constexpr int max_prefix = 32;
    constexpr const char shared_name[] = "ffn_shexp_gated-";
    int end = -1;
    for (int i = 0; i < std::min(split.graph.n_nodes - 1, max_prefix); ++i) {
        const auto * node = split.graph.nodes[i];
        const auto * merge = split.graph.nodes[i + 1];
        if (node->op == GGML_OP_MUL && merge->op == GGML_OP_ADD &&
                strncmp(node->name, shared_name, sizeof(shared_name) - 1) == 0 &&
                strcmp(node->name + sizeof(shared_name) - 1, routed_out->name + sizeof(routed_name) - 1) == 0 &&
                merge->src[0] == copied && merge->src[1] == node &&
                (!verify || (node->type == GGML_TYPE_F32 && ggml_are_same_shape(node, copied) &&
                            ggml_are_same_shape(merge, copied)))) {
            end = i;
            break;
        }
    }
    if (end < 0) {
        return false;
    }

    // Every prefix node must feed the Shared Expert and must be independent of the routed input copy.
    uint32_t needed = UINT32_C(1) << end;
    for (int i = end; i >= 0; --i) {
        const auto * node = split.graph.nodes[i];
        if (!(needed & (UINT32_C(1) << i)) || !ggml_backend_sched_shared_range(node, copied, buft) || (verify && node->op == GGML_OP_CUSTOM)) {
            return false;
        }
        for (const auto * src : node->src) {
            if (!src) {
                continue;
            }
            if (!ggml_backend_sched_shared_range(src, copied, buft)) {
                return false;
            }
            const auto it = std::find(split.graph.nodes, split.graph.nodes + end + 1, src);
            if (it != split.graph.nodes + end + 1) {
                const auto pos = it - split.graph.nodes;
                if (pos >= i) {
                    return false;
                }
                needed |= UINT32_C(1) << pos;
            }
        }
    }
    split.shared_prefix = ggml_graph_view(&split.graph, 0, end + 1);
    split.after_shared = ggml_graph_view(&split.graph, end + 1, split.graph.n_nodes);
    split.shared_prefix.uid = ggml_graph_next_uid();
    split.after_shared.uid = ggml_graph_next_uid();
    if (hip && three && sched->hip_mapped_input_blit && split.mapped_hip_input) {
        const auto * input = split.mapped_hip_input;
        const auto * producer = input->view_src ? input->view_src : input;
        const auto end_node = split.after_shared.nodes + split.after_shared.n_nodes;
        if (std::find(split.after_shared.nodes, end_node, producer) != end_node) {
            split.mapped_hip_input_after_shared_uid = ggml_graph_next_uid();
        }
    }
    return true;
}

using ggml_backend_vk_graph_readback_t = enum ggml_status (*)(ggml_backend_t, ggml_cgraph *,
        const ggml_tensor *, ggml_backend_buffer_t, bool *);

static void ggml_backend_sched_drain_hip_async_input(ggml_backend_sched_t sched) {
    if (sched->hip_async_input_pending) {
        ggml_backend_synchronize(sched->hip_async_input_pending);
        sched->hip_async_input_pending = nullptr;
    }
}

static bool ggml_backend_sched_prepare_hip_return(ggml_backend_sched_t sched, ggml_backend_sched_hip_return & pending,
        ggml_backend_sched_split & routed, ggml_backend_sched_split & next, bool wait_for_reuse = true) {
    if (pending.source || next.n_inputs != 1) {
        return false;
    }
    auto * source = sched->backends[routed.backend_id];
    auto * destination = sched->backends[next.backend_id];
    if (strncmp(ggml_backend_name(destination), "CUDA", 4) != 0 ||
            !ggml_backend_sched_is_cuda_pair(source, destination, "ROCm") || !source->iface.get_tensor_async ||
            sched->events[next.backend_id][sched->cur_copy]) {
        return false;
    }
    auto * input = next.inputs[0];
    if (strncmp(input->name, "ffn_moe_out-", sizeof("ffn_moe_out-") - 1) != 0 ||
            input->op != GGML_OP_ADD || !(input->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            (input->flags & GGML_TENSOR_FLAG_INPUT) || input->type != GGML_TYPE_F32 ||
            input->ne[0] != 2560 || input->ne[1] != (sched->target_mtp_verify_four ? 4 : 3) || !ggml_is_matrix(input) ||
            !sched->hv_tensor_requires_copy[hash_id(input)] ||
            ggml_backend_sched_get_tensor_backend(sched, input) != source ||
            std::find(routed.graph.nodes, routed.graph.nodes + routed.graph.n_nodes, input) ==
                    routed.graph.nodes + routed.graph.n_nodes) {
        return false;
    }
    const auto * copy = tensor_copy(input, next.backend_id, sched->cur_copy);
    const size_t bytes = ggml_nbytes(input);
    if (bytes > shared_host_capacity || !ggml_are_same_layout(input, copy) ||
            !ggml_backend_sched_device_input_buffer(source, input, bytes) ||
            !ggml_backend_sched_device_input_buffer(destination, copy, bytes)) {
        return false;
    }
    auto * pinned = ggml_backend_sched_get_dual_pinned(sched, source, destination);
    if (!pinned) {
        return false;
    }
    if (wait_for_reuse) {
        ggml_backend_sched_wait_hip_return_h2d(sched);
    }
    pending.input = input;
    pending.copy = copy;
    pending.data = pinned + shared_host_capacity * (1 + (sched->async_hip_input || sched->async_hip_mtp4_input));
    pending.enqueue_start_us = 0;
    pending.enqueue_end_us = 0;
    pending.mapped = false;
    return true;
}

int ggml_backend_sched_get_hip_single_miss_cpu_mode(void) {
    return hip_single_miss_cpu_mode.load(std::memory_order_relaxed);
}

int ggml_backend_sched_get_hip_single_miss_inputs(ggml_backend_sched_t sched, const float ** activation, const float ** weights) {
    const int mode = hip_single_miss_cpu_mode.load(std::memory_order_relaxed);
    if (!sched || !mode || !sched->active_single_miss ||
            !sched->async_hip_return || !sched->target_mtp_verify_three || sched->target_mtp_verify_four ||
            sched->profile.enabled || sched->callback_eval || sched->n_copies != 1) {
        return 0;
    }
    const auto * copies = sched->active_host_copies;
    const auto * split = sched->active_copy_split;
    if (!copies || !copies->single_miss_activation || !copies->single_miss_weights_dst || !split ||
            split->has_custom || split + 1 == sched->splits + sched->n_splits || sched->active_single_miss->resolve) {
        return 0;
    }

    const ggml_tensor * matmuls[3] = {};
    const ggml_tensor * glu = nullptr;
    const ggml_tensor * weighted = nullptr;
    int count = 0;
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const auto * node = split->graph.nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (count == 3) {
                return 0;
            }
            matmuls[count++] = node;
        } else if (node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU) {
            glu = node;
        } else if (node->op == GGML_OP_MUL && node->src[1] == copies->single_miss_weights_dst) {
            weighted = node;
        }
    }
    int host_weights = 0;
    auto & routed = sched->splits[split - sched->splits];
    for (int i = 0; i < routed.n_inputs; ++i) {
        auto * input = routed.inputs[i];
        host_weights += ggml_backend_sched_is_host_weight(input) && sched->hv_tensor_requires_copy[hash_id(input)];
    }
    ggml_backend_sched_hip_return pending;
    if (count != 3 || host_weights != 1 || !glu || !weighted ||
            matmuls[0]->src[1] != copies->single_miss_activation_dst ||
            matmuls[1]->src[1] != copies->single_miss_activation_dst ||
            matmuls[2]->src[1] != glu || weighted->src[0] != matmuls[2] ||
            !((glu->src[0] == matmuls[0] && glu->src[1] == matmuls[1]) ||
              (glu->src[0] == matmuls[1] && glu->src[1] == matmuls[0])) ||
            !ggml_backend_sched_prepare_hip_return(sched, pending, routed, sched->splits[split - sched->splits + 1], false)) {
        return 0;
    }

    *activation = copies->single_miss_activation;
    *weights = copies->single_miss_weights;
    return mode;
}

void ggml_backend_sched_set_hip_single_miss_result(ggml_backend_sched_t sched,
        const float * (*resolve)(void *, int64_t), void * user_data, unsigned mask) {
    auto * correction = sched->active_single_miss;
    const auto * destination = sched->active_copy_split + 1;
    GGML_ASSERT(correction && !correction->resolve && resolve && mask > 0 && mask < 8 && destination->n_inputs == 1);
    *correction = {destination, destination->inputs[0], resolve, user_data, mask};
}

static enum ggml_status ggml_backend_sched_run_deferred_router(ggml_backend_sched_t sched, bool early) {
    auto * pending = sched->active_deferred_router;
    const auto status = ggml_backend_graph_compute_async(pending->backend, &pending->graph);
    pending->target = nullptr;
    if (status == GGML_STATUS_SUCCESS) {
        if (early) {
            ++sched->next_router_early;
        } else {
            ++sched->next_router_deferred;
        }
    }
    return status;
}

enum ggml_status ggml_backend_sched_flush_deferred_router(ggml_backend_sched_t sched, const ggml_tensor * ids) {
    auto * pending = sched->active_deferred_router;
    if (pending && pending->target && pending->graph.nodes[pending->graph.n_nodes - 1] == ids) {
        return ggml_backend_sched_run_deferred_router(sched, true);
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;
    auto * host_trace = sched->target_mtp_verify_three ? sched->hip_verify_trace : nullptr;
    const uint64_t graph_run = host_trace ? ++host_trace->graphs : 0;
    ggml_backend_hip_verify_stage graph_stage{host_trace, graph_run, -1, -1};
    graph_stage.next("graph_setup");

    sched->trace_handoff_graph = 0;
    if (sched->trace_handoff && sched->n_splits > 8) {
        const size_t graph_index = ++sched->trace_handoff_graphs;
        if (graph_index >= 17 && graph_index <= 20) {
            sched->trace_handoff_graph = ggml_time_us();
            GGML_LOG_INFO("expert_handoff: graph=%lld ordinal=%zu splits=%d\n",
                    (long long) sched->trace_handoff_graph, graph_index, sched->n_splits);
        }
    }
    // Also join an in-flight readback if the destination graph fails before consuming it.
    struct shared_return_wait {
        ggml_backend_t backend = nullptr;
        ~shared_return_wait() {
            if (backend) {
                ggml_backend_synchronize(backend);
            }
        }
    } shared_return;
    struct hip_async_input_on_failure {
        ggml_backend_sched_t sched;
        bool success = false;
        ~hip_async_input_on_failure() {
            if (!success) {
                ggml_backend_sched_drain_hip_async_input(sched);
            }
        }
    } async_input_guard{sched};
    struct after_submit_scope {
        ggml_backend_sched_t sched;
        ggml_backend_sched::after_submit pending;
        ggml_backend_sched::deferred_router router;
        ggml_backend_sched_single_miss single_miss;
        ~after_submit_scope() {
            if (single_miss.resolve) {
                single_miss.resolve(single_miss.user_data, 0);
            }
            sched->active_after_submit = nullptr;
            sched->active_deferred_router = nullptr;
            sched->active_single_miss = nullptr;
            sched->ranked_rtx_empty_split = nullptr;
        }
    } after_submit{sched, {}, {}, {}};
    sched->active_after_submit = &after_submit.pending;
    sched->active_deferred_router = &after_submit.router;
    sched->active_single_miss = &after_submit.single_miss;
    ggml_backend_sched_hip_return hip_return;
    ggml_backend_sched_mapped_hip_input mapped_input;
    const ggml_tensor * shared_return_input = nullptr;
    int prev_backend_id = -1;
    auto & profile = sched->profile;
    if (profile.enabled) {
        sched->n_prefetched = 0;
    }

    std::vector<ggml_tensor *> merged_nodes;
    std::vector<ggml_tensor *> ranked_rtx_nodes;
    graph_stage.next(nullptr);
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_hip_verify_stage stage{host_trace, graph_run, split_id, split_backend_id};
        stage.next("split_setup");
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        ggml_backend_sched_host_copies split_host_copies;
        auto * host_copies = sched->capture_host_copies && sched->n_copies == 1 &&
                !sched->events[split_backend_id][sched->cur_copy] ? &split_host_copies : nullptr;
        const int64_t input_start_us = profile.enabled ? ggml_time_us() : 0;
        ggml_cgraph * main_graph = &split->graph;
        sched->carry_prefix_end = 0;
        sched->carry_accumulator = nullptr;
        sched->mtp_three_empty_next = false;
        ggml_backend_sched_upload upload;

        if ((sched->overlap_shared || sched->overlap_hip_shared || sched->overlap_hip_mtp4_shared) &&
                split_id > 0 && !profile.enabled && !sched->callback_eval) {
            const auto & routed = splits[split_id - 1];
            if (ggml_backend_sched_plan_shared(sched, *split, splits[split_id - 1], true)) {
                // The routed split is already submitted. The CUDA destination wait below joins this prefix.
                const int64_t start = sched->trace_handoff_graph ? ggml_time_us() : 0;
                const auto status = ggml_backend_graph_compute_async(split_backend, &split->shared_prefix);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                if (start) {
                    ggml_backend_sched_trace_submit(sched, split_id, split_backend, &split->shared_prefix, start, ggml_time_us());
                }
                ++sched->shared_overlap_submits;
                sched->shared_overlap_tokens += split->shared_prefix.nodes[split->shared_prefix.n_nodes - 1]->ne[1];
                const int64_t width = split->inputs[0]->ne[1];
                if (ggml_backend_sched_is_cuda_pair(sched->backends[routed.backend_id], split_backend, "ROCm")) {
                    sched->overlap_hip_shared_submits += width == 3;
                    sched->overlap_hip_mtp4_shared_submits += width == 4;
                } else {
                    sched->mtp_three_shared_overlap_submits += width == 3;
                }
                main_graph = &split->after_shared;
            }
        }

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        stage.next("early_weight_copy");
        if (sched->callback_copy_early && sched->n_copies == 1 && !profile.enabled &&
                sched->callback_eval == NULL && prev_backend_id == split_backend_id) {
            for (int input_id = 0; input_id < split->n_inputs; input_id++) {
                struct ggml_tensor * input = split->inputs[input_id];
                if (!(input->flags & GGML_TENSOR_FLAG_INPUT) && ggml_backend_sched_is_host_weight(input) &&
                        sched->hv_tensor_requires_copy[hash_id(input)] &&
                        sched->callback_copy_early(split_backend, input, tensor_copy(input, split_backend_id, sched->cur_copy),
                                &split->graph, sched->callback_copy_early_user_data)) {
                    break;
                }
            }
        }

        stage.next("nonweight_copy");
        // copy the input tensors to the split backend
        // the weights in host memory are copied last, so that the copy callback can read the other inputs of the split
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];
            if (!ggml_backend_sched_is_host_weight(input) && sched->hv_tensor_requires_copy[hash_id(input)]) {
                if (input == shared_return_input) {
                    auto * copy = tensor_copy(input, split_backend_id, sched->cur_copy);
                    const size_t bytes = ggml_nbytes(input);
                    ggml_backend_synchronize(shared_return.backend);
                    shared_return.backend = nullptr;
                    shared_return_input = nullptr;
                    // The upload queues behind the CUDA prefix. Finish reading RAM before Vulkan may overwrite it.
                    ggml_backend_tensor_set_async(split_backend, copy, sched->shared_return.host, 0, bytes);
                    ggml_backend_synchronize(split_backend);
                    ++sched->shared_return_copies;
                    sched->shared_return_bytes += bytes;
                    sched->mtp_three_shared_return_copies += input->ne[1] == 3;
                    continue;
                }
                if (sched->batch_host_inputs && (input->flags & GGML_TENSOR_FLAG_INPUT)) {
                    const int count = ggml_backend_sched_batch_host_inputs(sched, split, input_id);
                    if (count > 0) {
                        input_id += count - 1;
                        continue;
                    }
                }
                if ((sched->batch_device_inputs || sched->batch_hip_device_inputs ||
                        (sched->async_hip_mtp4_input && sched->target_mtp_verify_four)) &&
                        !(input->flags & GGML_TENSOR_FLAG_INPUT)) {
                    const int count = ggml_backend_sched_batch_device_inputs(sched, split, input_id, host_copies, upload, mapped_input);
                    if (count > 0) {
                        input_id += count - 1;
                        continue;
                    }
                }
                if (!profile.enabled) {
                    ggml_backend_sched_copy_input(sched, split, input, host_copies, &hip_return);
                    continue;
                }
                const ggml_tensor * copy = tensor_copy(input, split_backend_id, sched->cur_copy);
                const bool to_device = !ggml_backend_buffer_is_host(copy->buffer);
                const bool from_host = ggml_backend_buffer_is_host(input->buffer);
                const int64_t start_us = ggml_time_us();
                ggml_backend_sched_copy_input(sched, split, input, host_copies);
                const int64_t copy_us = ggml_time_us() - start_us;
                if (to_device && from_host) {
                    profile.host_to_device_inputs++;
                    profile.host_to_device_us += copy_us;
                } else if (to_device && tensor_backend_id(input) != split_backend_id) {
                    profile.device_to_device_inputs++;
                    profile.device_to_device_us += copy_us;
                }
            }
        }
        if (after_submit.single_miss.destination == split) {
            GGML_LOG_ERROR("packed HIP MTP3 single CPU miss: expected pinned return was not consumed\n");
            return GGML_STATUS_FAILED;
        }
        if (mapped_input.destination == split) {
            mapped_input.wait();
        }
        stage.next("prefetch_inputs");
        // after the copies above, which may read host memory reused by the copies of the next split
        // before the weights, whose copy callback may queue large uploads
        if (!profile.enabled) {
            ggml_backend_sched_prefetch_inputs(sched, split_id);
        }
        stage.next("host_weight_copy");
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            struct ggml_tensor * input = split->inputs[input_id];
            if (ggml_backend_sched_is_host_weight(input) && sched->hv_tensor_requires_copy[hash_id(input)]) {
                ggml_backend_sched_copy_input(sched, split, input, host_copies);
            }
        }

        stage.next("dispatch_prepare");
        ggml_backend_cuda_profile_compute_t cuda_compute = nullptr;
        if (profile.enabled && profile.cuda) {
            ggml_backend_dev_t dev = ggml_backend_get_device(split_backend);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
                cuda_compute = (ggml_backend_cuda_profile_compute_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_profile_compute");
            }
            if (cuda_compute) {
                // Finish queued copies before timing CUDA dispatches.
                ggml_backend_synchronize(split_backend);
            }
        }

        auto & backend_profile = profile.backend[split_backend_id];
        if (profile.enabled) {
            profile.input_us += ggml_time_us() - input_start_us;
            if (backend_profile.splits == 0) {
                snprintf(backend_profile.name, sizeof(backend_profile.name), "%s", ggml_backend_name(split_backend));
            }
            backend_profile.splits++;
        }

        ggml_cgraph merged_graph;
        const bool merged_empty = ggml_backend_sched_merge_empty_wave(sched, split, upload, merged_nodes, merged_graph);
        const int routed_split_id = split_id + merged_empty;
        if (merged_empty) {
            main_graph = &merged_graph;
        }

        ggml_backend_vk_graph_readback_t graph_readback = nullptr;
        const ggml_tensor * return_tensor = nullptr;
        if (sched->shared_host_return && !profile.enabled && !sched->callback_eval && routed_split_id + 1 < sched->n_splits) {
            auto & next = splits[routed_split_id + 1];
            if (ggml_backend_sched_plan_shared(sched, next, splits[routed_split_id])) {
                auto * input = next.inputs[0];
                auto * dst = sched->backends[next.backend_id];
                auto * copy = tensor_copy(input, next.backend_id, sched->cur_copy);
                const size_t bytes = ggml_nbytes(input);
                if (!(input->flags & GGML_TENSOR_FLAG_INPUT) && !sched->events[next.backend_id][sched->cur_copy] &&
                        dst->iface.set_tensor_async && ggml_are_same_layout(input, copy) &&
                        ggml_backend_sched_device_input_buffer(split_backend, input, bytes) &&
                        ggml_backend_sched_device_input_buffer(dst, copy, bytes)) {
                    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(split_backend));
                    auto readback = reinterpret_cast<ggml_backend_vk_graph_readback_t>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_graph_readback"));
                    if (readback && bytes <= shared_host_capacity && ggml_backend_sched_init_shared_host(sched->shared_return, split_backend, dst)) {
                        graph_readback = readback;
                        return_tensor = input;
                    }
                }
            }
        }

        ggml_cgraph without_router;
        uint64_t without_router_mapped_uid = 0;
        bool without_router_after_shared = false;
        auto compute = [&](struct ggml_cgraph * graph) {
            const bool trace = sched->trace_handoff_graph != 0;
            const int64_t submit_start_us = profile.enabled || trace ? ggml_time_us() : 0;
            struct callback_data {
                ggml_backend_sched_cuda_profile * profile;
                const char * backend;
            } data{profile.cuda, backend_profile.name};

            if (graph_readback || upload.compute) {
                bool recorded = false;
                if (graph_readback) {
                    shared_return.backend = split_backend;
                }
                if (upload.compute) {
                    sched->shared_input_pending = true;
                }
                const auto status = upload.compute ?
                        upload.compute(split_backend, graph, &upload.regions, return_tensor, sched->shared_return.vk_buffer, &recorded) :
                        graph_readback(split_backend, graph, return_tensor, sched->shared_return.vk_buffer, &recorded);
                if (upload.compute && status == GGML_STATUS_SUCCESS) {
                    ++sched->shared_input_batches;
                    sched->shared_input_bytes += upload.bytes;
                    if (merged_empty) {
                        ++sched->mtp_three_router_upload_batches;
                        sched->mtp_three_router_upload_bytes += ggml_nbytes(upload.regions.dst[2]);
                    }
                    if (upload.mtp_three_router) {
                        ++sched->mtp_three_router_upload_batches;
                        sched->mtp_three_router_upload_bytes += upload.bytes;
                    } else if (sched->target_mtp_verify_three) {
                        ++sched->mtp_three_shared_input_batches;
                        sched->mtp_three_shared_input_bytes += upload.bytes;
                    }
                }
                if (recorded) {
                    shared_return_input = return_tensor;
                } else {
                    shared_return.backend = nullptr;
                }
                if (trace) {
                    ggml_backend_sched_trace_submit(sched, split_id, split_backend, graph, submit_start_us, ggml_time_us());
                }
                return status;
            }

            const bool async_return = ((sched->async_hip_return && sched->target_mtp_verify_three) ||
                    (sched->async_hip_mtp4_return && sched->target_mtp_verify_four)) &&
                    !profile.enabled && !sched->callback_eval && !split->has_custom &&
                    graph == &split->graph && routed_split_id == split_id && split_id + 1 < sched->n_splits;
            auto & pinned = sched->hip_handoff;
            enum ggml_status mapped_status;
            const bool after_shared = graph == &split->after_shared ||
                    (graph == &without_router && without_router_after_shared);
            uint64_t mapped_input_uid = graph == &without_router ? without_router_mapped_uid :
                    after_shared ? split->mapped_hip_input_after_shared_uid :
                    graph == &split->graph ? split->mapped_hip_input_uid : 0;
            ggml_cgraph without_ranked;
            if (sched->ranked_rtx_empty_split == split) {
                sched->ranked_rtx_empty_split = nullptr;
                auto * merge = split->ranked_rtx_merge;
                if (ggml_ranked_rtx::can_skip(*graph, merge)) {
                    const auto & uid = split->ranked_rtx_empty_uid[after_shared][graph == &without_router];
                    without_ranked = ggml_ranked_rtx::omit(*graph, ranked_rtx_nodes, uid[0]);
                    if (mapped_input_uid) {
                        mapped_input_uid = uid[1];
                    }
                    graph = &without_ranked;
                    ++sched->ranked_rtx_empty;
                } else {
                    ++sched->ranked_rtx_empty_fallback;
                }
            }

            if (mapped_input_uid && !profile.enabled &&
                    !sched->callback_eval && !split->has_custom && routed_split_id == split_id) {
                auto & next = splits[split_id + 1];
                auto * dst = sched->backends[next.backend_id];
                auto * input = split->mapped_hip_input;
                const auto * producer = input->view_src ? input->view_src : input;
                if (strncmp(ggml_backend_name(split_backend), "CUDA", 4) == 0 &&
                        ggml_backend_sched_is_cuda_pair(split_backend, dst, "ROCm") &&
                        !sched->events[next.backend_id][sched->cur_copy] && dst->iface.set_tensor_async &&
                        ggml_backend_sched_device_input_buffer(split_backend, input, ggml_nbytes(input))) {
                    if (sched->hip_async_input_pending) {
                        // A reader still owns the slab. Keep the ordinary VRAM output and copy path.
                        ++sched->mapped_hip_input_busy;
                    } else if (ggml_backend_sched_get_dual_pinned(sched, split_backend, dst) && pinned.cuda_mapped &&
                            (!after_shared || pinned.hip_mapped) &&
                            pinned.cuda_compute(split_backend, graph, mapped_input_uid, producer,
                                pinned.cuda_mapped + (pinned.cuda_input_data - pinned.data), &mapped_status)) {
                        ++sched->mapped_hip_input_producers;
                        sched->mapped_hip_input_after_shared_producers += after_shared;
                        mapped_input.source = split_backend;
                        mapped_input.destination = &next;
                        mapped_input.input = input;
                        mapped_input.data = pinned.cuda_input_data;
                        if (trace) {
                            ggml_backend_sched_trace_submit(sched, split_id, split_backend, graph, submit_start_us, ggml_time_us());
                        }
                        return mapped_status;
                    }
                }
            }
            if (async_return && split->mapped_hip_return_uid &&
                    ggml_backend_sched_prepare_hip_return(sched, hip_return, *split, splits[split_id + 1]) &&
                    pinned.hip_mapped && pinned.hip_compute(split_backend, graph, split->mapped_hip_return_uid,
                        pinned.hip_mapped + (hip_return.data - pinned.data), &mapped_status)) {
                hip_return.source = split_backend;
                hip_return.mapped = true;
                if (trace) {
                    ggml_backend_sched_trace_submit(sched, split_id, split_backend, graph, submit_start_us, ggml_time_us());
                }
                return mapped_status;
            }

            const enum ggml_status status = cuda_compute ? cuda_compute(split_backend, graph,
                    [](struct ggml_tensor * const * nodes, int n_nodes, int64_t submit_us, int64_t wait_us, void * user_data) {
                        auto & data = *static_cast<callback_data *>(user_data);
                        const ggml_tensor * first = nodes[0];
                        const ggml_tensor * last = nodes[n_nodes - 1];
                        const char * weight = "";
                        for (int i = 0; i < n_nodes; i++) {
                            const ggml_tensor * node = nodes[i];
                            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
                                weight = node->src[0]->name;
                                break;
                            }
                        }
                        char key[6 * GGML_MAX_NAME + 96];
                        snprintf(key, sizeof(key), "backend=%s first_node=\"%s\" first_op=%s last_node=\"%s\" last_op=%s nodes=%d matmul_src0=\"%s\"",
                                data.backend, first->name, ggml_op_desc(first), last->name, ggml_op_desc(last), n_nodes, weight);
                        auto & timing = data.profile->dispatches[key];
                        timing.calls++;
                        timing.submit_us += submit_us;
                        timing.wait_us += wait_us;
                    }, &data) : ggml_backend_graph_compute_async(split_backend, graph);
            if (status == GGML_STATUS_SUCCESS && async_return &&
                    ggml_backend_sched_prepare_hip_return(sched, hip_return, *split, splits[split_id + 1])) {
                hip_return.source = split_backend;
                hip_return.enqueue_start_us = ggml_time_us();
                ggml_backend_tensor_get_async(split_backend, hip_return.input, hip_return.data, 0, ggml_nbytes(hip_return.input));
                hip_return.enqueue_end_us = ggml_time_us();
            }
            if (trace) {
                ggml_backend_sched_trace_submit(sched, split_id, split_backend, graph, submit_start_us, ggml_time_us());
            }
            if (profile.enabled) {
                const int64_t wait_start_us = ggml_time_us();
                ggml_backend_synchronize(split_backend);
                backend_profile.submit_us += wait_start_us - submit_start_us;
                backend_profile.wait_us += ggml_time_us() - wait_start_us;
            }
            return status;
        };

        auto dispatch = [&](ggml_cgraph * graph) {
            if (!split->has_custom) {
                return graph->n_nodes ? compute(graph) : GGML_STATUS_SUCCESS;
            }
            int begin = 0;
            for (int i = 0; i < graph->n_nodes; ++i) {
                auto * node = graph->nodes[i];
                if (node->op != GGML_OP_CUSTOM || split_backend != sched->custom_backend) {
                    continue;
                }
                ggml_custom_op_params params;
                memcpy(&params, node->op_params, sizeof(params));
                if (params.fun != sched->custom_dispatch) {
                    continue;
                }
                if (i > begin) {
                    auto prefix = ggml_graph_view(graph, begin, i);
                    const auto status = compute(&prefix);
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
                ggml_backend_synchronize(split_backend);
                if (params.n_tasks != 1) {
                    return GGML_STATUS_FAILED;
                }
                params.fun(node, 0, 1, params.userdata);
                begin = i + 1;
            }
            if (begin == graph->n_nodes) {
                return GGML_STATUS_SUCCESS;
            }
            auto suffix = ggml_graph_view(graph, begin, graph->n_nodes);
            return compute(&suffix);
        };

        ggml_cgraph after_carry;
        if (sched->carry_accumulator) {
            auto * dst = sched->carry_accumulator;
            auto * src = dst->src[0];
            if (dst->data != src->data) {
                ggml_backend_tensor_copy_async(split_backend, split_backend, src, dst);
            }
            after_carry = ggml_graph_view(&split->graph, sched->carry_prefix_end, split->graph.n_nodes);
            main_graph = &after_carry;
        }
        if (!sched->callback_eval) {
            const bool after_shared = main_graph == &split->after_shared;
            const int main_nodes = main_graph->n_nodes - (split->graph.n_nodes - split->next_router_begin);
            const bool defer_router = split->next_router_begin && sched->target_mtp_verify_three && !profile.enabled &&
                    !after_submit.router.target && routed_split_id == split_id && !graph_readback && !upload.compute &&
                    (main_graph == &split->graph || after_shared) && main_nodes > 0;
            if (defer_router) {
                without_router = ggml_graph_view(main_graph, 0, main_nodes);
                without_router.uid = after_shared ? split->without_router_after_shared_uid : split->without_router_uid;
                without_router_mapped_uid = after_shared ? (split->mapped_hip_input_after_shared_uid ?
                        split->without_router_after_shared_mapped_uid : 0) :
                        (split->mapped_hip_input_uid ? split->without_router_mapped_uid : 0);
                without_router_after_shared = after_shared;
                main_graph = &without_router;
            }
            if (sched->before_split_submit) {
                stage.next("before_submit_callback");
                sched->before_split_submit(sched->before_split_submit_user_data);
            }
            stage.next("graph_submit");
            enum ggml_status ec = dispatch(main_graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            if (after_submit.router.target == split) {
                stage.next("deferred_router_submit");
                ec = ggml_backend_sched_run_deferred_router(sched, false);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }
            }
            if (defer_router) {
                auto & pending = after_submit.router;
                pending.backend = split_backend;
                pending.target = split + 1;
                pending.graph = ggml_graph_view(&split->graph, split->next_router_begin, split->graph.n_nodes);
                pending.graph.uid = split->next_router_uid;
            }
            if (after_submit.pending.split == split) {
                stage.next("after_submit_callback");
                const auto pending = after_submit.pending;
                after_submit.pending = {};
                pending.callback(sched, pending.user_data);
            }
        } else {
            stage.next("graph_submit_eval");
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = dispatch(&gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                if (!profile.enabled) {
                    ggml_backend_synchronize(split_backend);
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        stage.next("split_finish");
        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        if (merged_empty) {
            ++sched->mtp_three_empty_merges;
            ++split_id;
        }
        prev_backend_id = split_backend_id;
    }

    graph_stage.next("graph_cleanup");
    async_input_guard.success = true;
    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;
    const char * ranked_rtx = getenv("LLAMA_EXPERT_RANKED_CUDA0_PROFILE");
    sched->ranked_rtx_configured = ranked_rtx && *ranked_rtx;
    const char * placement = getenv("LLAMA_EXPERT_PLACEMENT");
    const char * batch_hip = getenv("GGML_SCHED_HIP_BATCH_DEVICE_INPUTS");
    const char * skip_hip_wait = getenv("GGML_SCHED_HIP_SKIP_DUP_DST_WAIT");
    const char * overlap_hip_shared = getenv("GGML_SCHED_HIP_SHARED_OVERLAP");
    const char * overlap_hip_mtp4_shared = getenv("GGML_SCHED_HIP_MTP4_SHARED_OVERLAP");
    const char * dual_pinned_hip = getenv("GGML_SCHED_HIP_DUAL_PINNED_HANDOFF");
    const char * measure_hip = getenv("GGML_SCHED_HIP_HANDOFF_TIMING");
    const char * trace_hip_wait = getenv("GGML_SCHED_HIP_HANDOFF_WAIT_TRACE");
    const char * verify_trace_path = getenv("GGML_SCHED_HIP_VERIFY_STAGE_TRACE");
    const char * profile_expert_callback = getenv("LLAMA_EXPERT_PROFILE_CALLBACK");
    const char * async_hip_input = getenv("GGML_SCHED_HIP_ASYNC_INPUT");
    const char * async_hip_mtp4_input = getenv("GGML_SCHED_HIP_MTP4_ASYNC_INPUT");
    const char * async_hip_return = getenv("GGML_SCHED_HIP_ASYNC_RETURN");
    const char * async_hip_mtp4_return = getenv("GGML_SCHED_HIP_MTP4_ASYNC_RETURN");
    const bool allow_hip_handoff = placement && *placement && sched->n_copies == 1;
    const bool allow_verify_trace = allow_hip_handoff && verify_trace_path && *verify_trace_path;
    const bool allow_hip_batch = allow_hip_handoff && batch_hip && strcmp(batch_hip, "1") == 0;
    const bool allow_hip_mtp4_input = allow_hip_handoff && async_hip_mtp4_input && strcmp(async_hip_mtp4_input, "1") == 0;
    const bool allow_hip_mtp4_return = allow_hip_handoff && async_hip_mtp4_return && strcmp(async_hip_mtp4_return, "1") == 0;
    const bool allow_hip_wait_skip = allow_hip_handoff && skip_hip_wait && strcmp(skip_hip_wait, "1") == 0;
    const bool allow_hip_shared = allow_hip_handoff && overlap_hip_shared && strcmp(overlap_hip_shared, "1") == 0;
    const bool allow_hip_mtp4_shared = allow_hip_handoff && overlap_hip_mtp4_shared && strcmp(overlap_hip_mtp4_shared, "1") == 0;
    const bool allow_hip_dual_pinned = allow_hip_handoff && dual_pinned_hip && strcmp(dual_pinned_hip, "1") == 0;
    const bool allow_hip_wait_trace = allow_hip_handoff && trace_hip_wait && strcmp(trace_hip_wait, "1") == 0;
    const bool allow_hip_timing = allow_hip_dual_pinned || allow_hip_wait_trace ||
            (allow_hip_handoff && measure_hip && strcmp(measure_hip, "1") == 0);
    bool cuda_vulkan = false;
    bool cuda_hip = false;
    bool verify_trace_pair = false;
    const bool check_hip_pair = allow_hip_batch || allow_hip_wait_skip || allow_hip_shared ||
            allow_hip_mtp4_shared || allow_hip_timing || allow_hip_mtp4_input || allow_hip_mtp4_return;
    for (int src = 0; src < n_backends; ++src) {
        for (int dst = src + 1; dst < n_backends; ++dst) {
            cuda_vulkan |= ggml_backend_sched_is_cuda_vulkan_pair(backends[src], backends[dst]);
            if (check_hip_pair || allow_verify_trace) {
                const bool pair = ggml_backend_sched_is_cuda_pair(backends[src], backends[dst], "ROCm");
                cuda_hip |= check_hip_pair && pair;
                verify_trace_pair |= allow_verify_trace && pair;
            }
        }
    }
    if (verify_trace_pair) {
        auto * trace = new (std::nothrow) ggml_backend_hip_verify_trace(verify_trace_path);
        if (trace && trace->rows) {
            trace->backend_names.reserve(n_backends);
            for (int i = 0; i < n_backends; ++i) {
                trace->backend_names.emplace_back(ggml_backend_name(backends[i]));
            }
            sched->hip_verify_trace = trace;
        } else {
            delete trace;
            GGML_LOG_WARN("HIP verify stage trace: interval buffer allocation failed\n");
        }
    }
    cuda_vulkan = cuda_vulkan && placement && *placement;
    sched->batch_hip_device_inputs = allow_hip_batch && cuda_hip;
    sched->async_hip_mtp4_input = allow_hip_mtp4_input && cuda_hip;
    sched->async_hip_mtp4_return = allow_hip_mtp4_return && cuda_hip;
    sched->skip_hip_device_dst_wait = allow_hip_wait_skip && cuda_hip;
    sched->dual_pinned_hip_handoff = allow_hip_dual_pinned && cuda_hip;
    sched->measure_hip_handoff = allow_hip_timing && cuda_hip;
    sched->trace_hip_handoff_wait = allow_hip_wait_trace && cuda_hip;
    sched->measure_expert_copy_wait = cuda_hip &&
            (allow_verify_trace || (profile_expert_callback && strcmp(profile_expert_callback, "1") == 0));
    if (cuda_hip && async_hip_input && strcmp(async_hip_input, "1") == 0) {
        sched->async_hip_input = sched->batch_hip_device_inputs && sched->dual_pinned_hip_handoff;
        if (!sched->async_hip_input) {
            GGML_LOG_WARN("packed HIP MTP3 async input requires HIP input batching and dual-pinned handoff\n");
        }
    }
    if (sched->async_hip_input) {
        for (int i = 0; i < n_backends && !sched->mapped_hip_input; ++i) {
            sched->mapped_hip_input = ggml_backend_sched_can_map_hip_input(ggml_backend_get_device(backends[i]));
        }
    }
    if (cuda_hip && async_hip_return && strcmp(async_hip_return, "1") == 0) {
        sched->async_hip_return = sched->dual_pinned_hip_handoff;
        if (!sched->async_hip_return) {
            GGML_LOG_WARN("packed HIP MTP3 async return requires dual-pinned handoff\n");
        }
    }
    if (sched->mapped_hip_input && sched->async_hip_return) {
        for (int i = 0; i < n_backends; ++i) {
            if (strncmp(ggml_backend_name(backends[i]), "ROCm", 4) == 0) {
                const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[i]));
                sched->hip_mapped_input_blit = reinterpret_cast<ggml_backend_hip_mapped_input_blit_t>(
                        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_mapped_input_blit"));
                break;
            }
        }
    }
    const char * graph_opt = getenv("GGML_CUDA_GRAPH_OPT");
    sched->defer_next_router = sched->async_hip_input && !(graph_opt && atoi(graph_opt) == 1);
    const char * rtx_slots = getenv("LLAMA_EXPERT_RTX_COLD_SLOTS");
    sched->overlap_shared = sched->n_copies == 1 && cuda_vulkan &&
            !(graph_opt && atoi(graph_opt) == 1) && !(rtx_slots && atoi(rtx_slots) > 0);
    sched->overlap_hip_shared = allow_hip_shared && cuda_hip &&
            !(graph_opt && atoi(graph_opt) == 1) && !(rtx_slots && atoi(rtx_slots) > 0);
    sched->overlap_hip_mtp4_shared = allow_hip_mtp4_shared && cuda_hip &&
            !(graph_opt && atoi(graph_opt) == 1) && !(rtx_slots && atoi(rtx_slots) > 0);
#ifdef _WIN32
    sched->shared_host_return = sched->overlap_shared;
    sched->shared_host_input = sched->n_copies == 1 && cuda_vulkan &&
            !(rtx_slots && atoi(rtx_slots) > 0);
#endif
    const char * fast_three = getenv("LLAMA_MTP_VERIFY_THREE_FAST");
    if (fast_three && strcmp(fast_three, "1") == 0) {
        const char * input = getenv("GGML_SCHED_MTP_THREE_SHARED_INPUT");
        const char * overlap = getenv("GGML_SCHED_MTP_THREE_SHARED_OVERLAP");
        const char * router = getenv("GGML_SCHED_MTP_THREE_ROUTER_UPLOAD");
        sched->mtp_three_router_upload = sched->shared_host_input && router && strcmp(router, "1") == 0;
        const char * merge = getenv("GGML_SCHED_MTP_THREE_MERGE_EMPTY_WAVE");
        sched->mtp_three_merge_empty_wave = sched->mtp_three_router_upload && merge && strcmp(merge, "1") == 0;
        const char * mode = getenv("LLAMA_MTP_VERIFY_THREE");
        const char * reset = getenv("GGML_SCHED_MTP_AUTO_RESET_SHARED_PLAN");
        sched->mtp_auto_reset_shared_plan = sched->mtp_three_merge_empty_wave && mode && strcmp(mode, "auto") == 0 &&
                reset && strcmp(reset, "1") == 0;
        sched->mtp_three_shared_input = sched->shared_host_input && input && strcmp(input, "1") == 0;
        sched->mtp_three_shared_overlap = sched->overlap_shared && overlap && strcmp(overlap, "1") == 0;
    }
    const char * producer_copy = getenv("GGML_SCHED_SHARED_INPUT_PRODUCER_COPY");
    sched->shared_input_producer = sched->shared_host_input && producer_copy && strcmp(producer_copy, "1") == 0;
    const char * routed_stats = getenv("LLAMA_MTP_ROUTED_DISPATCH_STATS");
    const char * routed_reduce = getenv("GGML_VK_MTP_ROUTED_REDUCE");
    sched->routed_stats_enabled = routed_stats && strcmp(routed_stats, "1") == 0;
    if (sched->routed_stats_enabled || (routed_reduce && strcmp(routed_reduce, "1") == 0)) {
        for (int i = 0; i < n_backends; ++i) {
            auto backend = backends[i];
            if (strcmp(ggml_backend_name(backend), "Vulkan1") == 0) {
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                sched->routed_stats_backend = backend;
                if (reg) {
                    sched->set_routed_stats_active = reinterpret_cast<decltype(sched->set_routed_stats_active)>(
                            ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_set_mtp_routed_stats_active"));
                }
                break;
            }
        }
    }
    const char * trace_handoff = getenv("GGML_SCHED_TRACE_EXPERT_HANDOFF");
    sched->trace_handoff = trace_handoff && strcmp(trace_handoff, "1") == 0;
    const char * batch_host_inputs = getenv("GGML_SCHED_BATCH_HOST_INPUTS");
    sched->batch_host_inputs = batch_host_inputs && strcmp(batch_host_inputs, "1") == 0;
    sched->batch_device_inputs = cuda_vulkan;
    if (sched->batch_device_inputs) {
        GGML_LOG_INFO("sched_batch_device_inputs: enabled\n");
    }
    if (sched->batch_hip_device_inputs) {
        GGML_LOG_INFO("sched_batch_device_inputs: enabled for CUDA and ROCm\n");
    }

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids   = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies        = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
    sched->hv_tensor_requires_copy = (bool *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_requires_copy[0]));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->graph_size      = 0;
    sched->prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

static bool ggml_backend_sched_hip_verify_stage_flush(ggml_backend_sched_t sched) {
    auto * trace = sched->hip_verify_trace;
    if (trace->count) {
        FILE * file = fopen(trace->path, "a");
        if (!file) {
            GGML_LOG_WARN("HIP verify stage trace: could not open %s\n", trace->path);
            return false;
        }
        if (fseek(file, 0, SEEK_END) == 0 && ftell(file) == 0) {
            fputs("scheduler,graph,split,backend,stage,start_us,end_us\n", file);
        }
        for (size_t i = 0; i < trace->count; ++i) {
            const auto & row = trace->rows[i];
            const char * backend = row.backend < 0 ? "host" : trace->backend_names[row.backend].c_str();
            fprintf(file, "%p,%llu,%d,%s,%s,%lld,%lld\n", (void *) sched,
                    (unsigned long long) row.graph, row.split, backend, row.stage,
                    (long long) row.start_us, (long long) row.end_us);
        }
        const bool failed = ferror(file);
        if (fclose(file) != 0 || failed) {
            GGML_LOG_WARN("HIP verify stage trace: incomplete write to %s\n", trace->path);
            return false;
        }
        trace->flushed += trace->count;
        trace->count = 0;
    }
    GGML_LOG_INFO("HIP verify stage trace: graphs=%llu rows=%zu dropped=%zu path=%s\n",
            (unsigned long long) trace->graphs, trace->flushed, trace->dropped, trace->path);
    return true;
}

void ggml_backend_sched_hip_verify_stage_poll_flush(ggml_backend_sched_t sched) {
    auto * trace = sched->hip_verify_trace;
    if (!trace || !trace->count) {
        return;
    }
    FILE * request = fopen(trace->flush_path.c_str(), "rb");
    if (!request) {
        return;
    }
    fclose(request);
    if (ggml_backend_sched_hip_verify_stage_flush(sched) && remove(trace->flush_path.c_str()) != 0) {
        GGML_LOG_WARN("HIP verify stage trace: could not remove %s\n", trace->flush_path.c_str());
    }
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    if (sched->ranked_rtx_configured && (sched->ranked_rtx_hot || sched->ranked_rtx_empty || sched->ranked_rtx_empty_fallback)) {
        GGML_LOG_INFO("ranked RTX Experts sparse dispatch: empty_skipped=%zu hot_layers=%zu empty_fallback=%zu inactive_hot_lanes=%zu\n",
                sched->ranked_rtx_empty, sched->ranked_rtx_hot, sched->ranked_rtx_empty_fallback, sched->ranked_rtx_inactive_hot_lanes);
    }
    if (auto * trace = sched->hip_verify_trace) {
        ggml_backend_sched_hip_verify_stage_flush(sched);
        delete trace;
    }
    if (sched->shared_input_pending) {
        ggml_backend_synchronize(sched->shared_input.vk);
    }
    ggml_backend_sched_drain_hip_async_input(sched);
    const auto & profile = sched->profile;
    if (profile.graphs > 0) {
        int64_t compute_us = 0;
        for (int b = 0; b < sched->n_backends; b++) {
            const auto & backend = profile.backend[b];
            if (backend.splits > 0) {
                GGML_LOG_INFO("decode_profile: backend=%s splits=%zu compute_ms=%.3f submit_ms=%.3f completion_wait_ms=%.3f\n",
                        backend.name, backend.splits, (backend.submit_us + backend.wait_us) / 1000.0,
                        backend.submit_us / 1000.0, backend.wait_us / 1000.0);
                compute_us += backend.submit_us + backend.wait_us;
            }
        }
        GGML_LOG_INFO("decode_profile: single_token_graphs=%zu split_wall_ms=%.3f entry_wait_ms=%.3f input_copy_and_dep_ms=%.3f cross_backend_copies=%zu cross_backend_MiB=%.3f copy_callback_ms=%.3f copy_callbacks=%zu other_split_ms=%.3f\n",
                profile.graphs, profile.graph_us / 1000.0, profile.entry_wait_us / 1000.0,
                (profile.input_us - profile.callback_us) / 1000.0, profile.cross_backend_copies, profile.cross_backend_bytes / (1024.0 * 1024.0),
                profile.callback_us / 1000.0, profile.callback_count, (profile.graph_us - profile.input_us - compute_us) / 1000.0);
        GGML_LOG_INFO("decode_profile: host_to_device_nonweight_inputs=%zu host_to_device_copy_and_dep_ms=%.3f device_to_device_nonweight_inputs=%zu device_to_device_copy_and_dep_ms=%.3f\n",
                profile.host_to_device_inputs, profile.host_to_device_us / 1000.0,
                profile.device_to_device_inputs, profile.device_to_device_us / 1000.0);
    }
    if (sched->batch_host_inputs) {
        GGML_LOG_INFO("sched_batch_host_inputs: batches=%zu tensors=%zu\n", sched->host_input_batches, sched->batched_host_inputs);
    }
    if (sched->overlap_shared) {
        GGML_LOG_INFO("shared expert overlap: prefix_submits=%zu tokens=%zu\n", sched->shared_overlap_submits, sched->shared_overlap_tokens);
    }
    if (sched->batch_device_inputs || sched->batch_hip_device_inputs || sched->skip_hip_device_dst_wait) {
        GGML_LOG_INFO("sched_batch_device_inputs: device_input_batches=%zu batched_device_inputs=%zu bytes=%zu skipped_dst_waits=%zu\n",
                sched->device_input_batches, sched->batched_device_inputs, sched->batched_device_input_bytes, sched->skipped_device_dst_waits);
    }
    if (profile.cuda) {
        for (const auto & [key, timing] : profile.cuda->dispatches) {
            GGML_LOG_INFO("decode_profile_cuda: %s calls=%zu dispatch_wall_ms=%.3f host_submit_ms=%.3f stream_wait_ms=%.3f\n",
                    key.c_str(), timing.calls, (timing.submit_us + timing.wait_us) / 1000.0,
                    timing.submit_us / 1000.0, timing.wait_us / 1000.0);
        }
        delete profile.cuda;
    }
    ggml_backend_sched_wait_hip_return_h2d(sched);
    ggml_backend_event_free(sched->hip_return_h2d_event);
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        ggml_backend_event_free(sched->prefetch_events[b]);
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->hv_tensor_requires_copy);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched->device_input_staging);
    if (sched->shared_host_return) {
        GGML_LOG_INFO("shared host expert return: copies=%zu bytes=%zu\n", sched->shared_return_copies, sched->shared_return_bytes);
    }
    if (sched->shared_host_input) {
        GGML_LOG_INFO("shared host expert input: batches=%zu bytes=%zu\n", sched->shared_input_batches, sched->shared_input_bytes);
        GGML_LOG_INFO("shared host producer copy: enabled=%d eligible=%zu batches=%zu tensors=%zu bytes=%zu waits=%zu fallbacks=%zu"
                " scope=mtp_verify verify_shared=%zu other_shared=%zu verify_bytes=%zu other_bytes=%zu optimized_verify=%zu optimized_other=%zu\n",
                sched->shared_input_producer, sched->shared_input_producer_eligible, sched->shared_input_producer_batches,
                3 * sched->shared_input_producer_batches, sched->shared_input_producer_bytes,
                sched->shared_input_producer_batches, sched->shared_input_producer_fallbacks,
                sched->shared_input_verify_batches, sched->shared_input_other_batches,
                sched->shared_input_verify_bytes, sched->shared_input_other_bytes,
                sched->shared_input_optimized_verify, sched->shared_input_optimized_other);
    }
    if (sched->mtp_auto_reset_shared_plan) {
        GGML_LOG_INFO("mtp auto shared plan reset: stale_prefixes=%zu\n", sched->mtp_auto_stale_shared_prefixes);
    }
    if (sched->mtp_three_merge_empty_wave) {
        GGML_LOG_INFO("mtp fast-three merged empty waves: graphs=%zu candidates=%zu bad_split=%zu bad_inputs=%zu"
                " unpinned=%zu bad_carry=%zu bad_upload=%zu alias=%zu\n",
                sched->mtp_three_empty_merges, sched->mtp_three_empty_candidates, sched->mtp_three_empty_bad_split,
                sched->mtp_three_empty_bad_inputs, sched->mtp_three_empty_unpinned, sched->mtp_three_empty_bad_carry,
                sched->mtp_three_empty_bad_upload, sched->mtp_three_empty_alias);
    }
    if (sched->mtp_three_router_upload) {
        GGML_LOG_INFO("mtp fast-three router upload: batches=%zu bytes=%zu\n",
                sched->mtp_three_router_upload_batches, sched->mtp_three_router_upload_bytes);
    }
    if (sched->mtp_three_shared_input || sched->mtp_three_shared_overlap) {
        GGML_LOG_INFO("mtp fast-three shared handoff: input_enabled=%d input_batches=%zu input_bytes=%zu"
                " overlap_enabled=%d prefix_submits=%zu return_copies=%zu\n",
                sched->mtp_three_shared_input, sched->mtp_three_shared_input_batches, sched->mtp_three_shared_input_bytes,
                sched->mtp_three_shared_overlap, sched->mtp_three_shared_overlap_submits, sched->mtp_three_shared_return_copies);
    }
    if (sched->overlap_hip_shared) {
        GGML_LOG_INFO("packed HIP MTP3 Shared Expert overlap: prefix_submits=%zu\n", sched->overlap_hip_shared_submits);
    }
    if (sched->overlap_hip_mtp4_shared) {
        GGML_LOG_INFO("packed HIP MTP4 Shared Expert overlap: prefix_submits=%zu\n", sched->overlap_hip_mtp4_shared_submits);
    }
    if (sched->measure_hip_handoff) {
        for (int i = 0; i < 2; ++i) {
            const auto & stats = sched->hip_handoff_stats[i];
            GGML_LOG_INFO("packed HIP MTP3 host handoff: direction=%s ops=%zu eligible_copies=%zu eligible_bytes=%zu"
                    " pinned_copies=%zu pinned_bytes=%zu fallback_copies=%zu fallback_bytes=%zu read_ms=%.3f write_ms=%.3f\n",
                    i == 0 ? "CUDA->ROCm" : "ROCm->CUDA", stats.ops, stats.copies, stats.bytes,
                    stats.pinned_copies, stats.pinned_bytes, stats.fallback_copies, stats.fallback_bytes,
                    stats.read_us / 1000.0, stats.write_us / 1000.0);
        }
    }
    if (sched->async_hip_input) {
        const auto & stats = sched->hip_handoff_stats[0];
        GGML_LOG_INFO("packed HIP MTP3 async input: batches=%zu copies=%zu bytes=%zu enqueue_ms=%.3f\n",
                stats.async_input_batches, stats.async_input_copies, stats.async_input_bytes, stats.enqueue_us / 1000.0);
    }
    if (sched->mapped_hip_input) {
        const auto & stats = sched->hip_handoff_stats[0];
        GGML_LOG_INFO("packed HIP MTP3 mapped input: producers=%zu copies=%zu bytes=%zu reader_busy=%zu mapped=%d\n",
                sched->mapped_hip_input_producers, stats.mapped_input_copies, stats.mapped_input_bytes,
                sched->mapped_hip_input_busy, sched->hip_handoff.cuda_mapped != nullptr);
    }
    if (sched->single_miss_combined) {
        GGML_LOG_INFO("packed HIP MTP3 single CPU miss return: combined=%zu combine_ms=%.3f\n",
                sched->single_miss_combined, sched->single_miss_combine_us / 1000.0);
    }
    if (sched->defer_next_router) {
        GGML_LOG_INFO("packed HIP MTP3 deferred next Router: plans=%zu after_expert=%zu early=%zu\n",
                sched->next_router_plans, sched->next_router_deferred, sched->next_router_early);
    }
    if (sched->measure_expert_copy_wait) {
        GGML_LOG_INFO("packed HIP MTP3 Expert callback wait: before=%zu moved=%zu after_plan=%zu fallback=%zu before_ms=%.3f deferred_ms=%.3f cpu_before_wait_ms=%.3f\n",
                sched->expert_copy_wait_before, sched->expert_copy_wait_after_router,
                sched->expert_copy_wait_after_plan, sched->expert_copy_wait_fallback,
                sched->expert_copy_wait_before_us / 1000.0, sched->expert_copy_wait_after_us / 1000.0,
                sched->expert_copy_cpu_before_wait_us / 1000.0);
    }
    if (sched->hip_mapped_input_blit) {
        GGML_LOG_INFO("packed HIP MTP3 mapped input read: blit eligible=%zu applied=%zu bytes=%zu mapped=%d after_shared_producers=%zu\n",
                sched->hip_mapped_input_read_eligible, sched->hip_mapped_input_read_applied,
                sched->hip_mapped_input_read_applied * (2560 * 3 * sizeof(float)), sched->hip_handoff.hip_mapped != nullptr,
                sched->mapped_hip_input_after_shared_producers);
    }
    if (sched->async_hip_return) {
        GGML_LOG_INFO("packed HIP MTP3 async return H2D: applied=%zu fallback=%zu reuse_waits=%zu\n",
                sched->hip_return_h2d_applied, sched->hip_return_h2d_fallback, sched->hip_return_h2d_waits);
    }
    if (sched->async_hip_mtp4_input && sched->hip_mtp4_input_stats.ops) {
        const auto & stats = sched->hip_mtp4_input_stats;
        GGML_LOG_INFO("packed HIP MTP4 input: batches=%zu copies=%zu bytes=%zu async_batches=%zu async_copies=%zu async_bytes=%zu"
                " source_wait_ms=%.3f destination_wait_ms=%.3f read_ms=%.3f enqueue_ms=%.3f sync_write_ms=%.3f\n",
                stats.ops, stats.copies, stats.bytes, stats.async_input_batches, stats.async_input_copies, stats.async_input_bytes,
                stats.source_wait_us / 1000.0, stats.destination_wait_us / 1000.0,
                stats.read_us / 1000.0, stats.enqueue_us / 1000.0, stats.write_us / 1000.0);
    }
    if (sched->async_hip_return) {
        const auto & stats = sched->hip_handoff_stats[1];
        GGML_LOG_INFO("packed HIP MTP3 async return: copies=%zu bytes=%zu d2h_enqueue_ms=%.3f\n",
                stats.async_return_copies, stats.async_return_bytes, stats.return_enqueue_us / 1000.0);
        if (stats.mapped_return_copies) {
            GGML_LOG_INFO("packed HIP MTP3 mapped return: copies=%zu bytes=%zu\n",
                    stats.mapped_return_copies, stats.mapped_return_bytes);
        }
    }
    if (sched->async_hip_mtp4_return && sched->hip_mtp4_return_stats.ops) {
        const auto & stats = sched->hip_mtp4_return_stats;
        GGML_LOG_INFO("packed HIP MTP4 async return: copies=%zu bytes=%zu d2h_enqueue_ms=%.3f write_ms=%.3f\n",
                stats.async_return_copies, stats.async_return_bytes, stats.return_enqueue_us / 1000.0, stats.write_us / 1000.0);
    }
    if (sched->routed_stats_rounds) {
        GGML_LOG_INFO("mtp routed schedule: rounds=%zu vulkan1_aggregate_add=%zu other_backend_aggregate_add=%zu other_aggregate_op=%zu\n",
                sched->routed_stats_rounds, sched->routed_stats_vulkan_aggregates,
                sched->routed_stats_other_backend, sched->routed_stats_other_op);
    }
    // Release Vulkan's imports before CUDA unpins and frees the original allocations.
    ggml_backend_buffer_free(sched->shared_return.vk_buffer);
    ggml_backend_buffer_free(sched->shared_return.cuda_buffer);
    ggml_backend_buffer_free(sched->shared_input.vk_buffer);
    ggml_backend_buffer_free(sched->shared_input.cuda_buffer);
    auto & pinned = sched->hip_handoff;
    if (pinned.data) {
        pinned.unreg[1](pinned.owners[1], pinned.data);
        pinned.unreg[0](pinned.owners[0], pinned.data);
        free(pinned.allocation);
    }
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids,  -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,        0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        memset(sched->hv_tensor_requires_copy, 0, sched->hash_set.size * sizeof(sched->hv_tensor_requires_copy[0]));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);
    auto * trace = sched->target_mtp_verify_three ? sched->hip_verify_trace : nullptr;
    ggml_backend_hip_verify_stage stage{trace, trace ? trace->graphs + 1 : 0, -1, -1};
    stage.next("graph_allocate");

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

int64_t ggml_backend_sched_hip_verify_stage_time(ggml_backend_sched_t sched) {
    return sched && sched->hip_verify_trace ? ggml_time_us() : 0;
}

void ggml_backend_sched_hip_verify_stage_record(ggml_backend_sched_t sched, const char * stage, int64_t start, bool before_compute) {
    if (start && sched && sched->hip_verify_trace) {
        auto * trace = sched->hip_verify_trace;
        trace->record(trace->graphs + before_compute, -1, -1, stage, start, ggml_time_us());
    }
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    if (sched->target_mtp_verify && sched->routed_stats_enabled && sched->set_routed_stats_active) {
        ++sched->routed_stats_rounds;
        constexpr const char prefix[] = "ffn_moe_out-";
        for (int i = 0; i < graph->n_nodes; ++i) {
            const auto * node = graph->nodes[i];
            if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE) || strncmp(node->name, prefix, sizeof(prefix) - 1) != 0) {
                continue;
            }
            const size_t index = ggml_hash_find(&sched->hash_set, node);
            const int backend_id = index < sched->hash_set.size && ggml_bitset_get(sched->hash_set.used, index) ?
                    sched->hv_tensor_backend_ids[index] : -1;
            if (node->op != GGML_OP_ADD) {
                ++sched->routed_stats_other_op;
            } else if (backend_id >= 0 && sched->backends[backend_id] == sched->routed_stats_backend) {
                ++sched->routed_stats_vulkan_aggregates;
            } else {
                ++sched->routed_stats_other_backend;
            }
        }
    }

    auto & profile = sched->profile;
    if (!profile.enabled) {
        return ggml_backend_sched_compute_splits(sched);
    }

    const int64_t entry_us = ggml_time_us();
    ggml_backend_sched_synchronize(sched);
    const int64_t start_us = ggml_time_us();
    const enum ggml_status status = ggml_backend_sched_compute_splits(sched);
    profile.graph_us += ggml_time_us() - start_us;
    profile.entry_wait_us += start_us - entry_us;
    profile.graphs++;
    return status;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    // The shared input upload is complete, so cleanup no longer needs its backend.
    sched->shared_input_pending = false;
    sched->hip_async_input_pending = nullptr;
    sched->hip_return_h2d_pending = nullptr;
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

void ggml_backend_sched_set_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy = callback;
    sched->callback_copy_user_data = user_data;
}

void ggml_backend_sched_ranked_rtx_routes(ggml_backend_sched_t sched, int layer, int lanes) {
    if (lanes) {
        ++sched->ranked_rtx_hot;
        sched->ranked_rtx_inactive_hot_lanes += 30 - lanes;
        return;
    }
    const auto * split = sched->active_copy_split;
    static const bool concurrent_cuda_graph = [] {
        const char * value = getenv("GGML_CUDA_GRAPH_OPT");
        return value && atoi(value) == 1;
    }();
    if (!split || split + 1 == sched->splits + sched->n_splits || sched->n_copies != 1 ||
            sched->profile.enabled || sched->callback_eval || concurrent_cuda_graph) {
        ++sched->ranked_rtx_empty_fallback;
        return;
    }
    const auto * next = split + 1;
    if (!next->ranked_rtx_merge || next->ranked_rtx_merge->op_params[ggml_ranked_rtx::layer_param] != layer ||
            strncmp(ggml_backend_name(sched->backends[next->backend_id]), "CUDA", 4) != 0) {
        ++sched->ranked_rtx_empty_fallback;
        return;
    }
    sched->ranked_rtx_empty_split = next;
}

bool ggml_backend_sched_after_split_submit(ggml_backend_sched_t sched,
        void (*callback)(ggml_backend_sched_t sched, void * user_data), void * user_data) {
    const auto * split = sched->active_copy_split;
    auto * pending = sched->active_after_submit;
    if (!callback || !split || !pending || pending->callback || sched->n_copies != 1 ||
            sched->callback_eval || sched->profile.enabled || split->has_custom || !split->graph.n_nodes) {
        return false;
    }
    *pending = {split, callback, user_data};
    return true;
}

void ggml_backend_sched_set_before_split_submit(ggml_backend_sched_t sched,
        void (*callback)(void * user_data), void * user_data) {
    sched->before_split_submit = callback;
    sched->before_split_submit_user_data = user_data;
}

bool ggml_backend_sched_set_target_mtp_verify(ggml_backend_sched_t sched, bool enabled) {
    const bool previous = sched->target_mtp_verify;
    sched->target_mtp_verify = enabled;
    if (sched->set_routed_stats_active && enabled != previous) {
        sched->set_routed_stats_active(sched->routed_stats_backend, enabled);
    }
    return previous;
}

bool ggml_backend_sched_set_target_mtp_verify_three(ggml_backend_sched_t sched, bool enabled) {
    const bool previous = sched->target_mtp_verify_three;
    sched->target_mtp_verify_three = enabled;
    return previous;
}

bool ggml_backend_sched_set_target_mtp_verify_four(ggml_backend_sched_t sched, bool enabled) {
    const bool previous = sched->target_mtp_verify_four;
    sched->target_mtp_verify_four = enabled;
    return previous;
}

bool ggml_backend_sched_set_custom_op_backend(ggml_backend_sched_t sched, ggml_tensor * op, ggml_backend_t backend) {
    if (sched->n_copies != 1 || op->op != GGML_OP_CUSTOM || !backend) {
        return false;
    }
    ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    if (params.n_tasks != 1 || !params.fun ||
            (sched->custom_dispatch && (sched->custom_dispatch != params.fun || sched->custom_backend != backend))) {
        return false;
    }
    sched->custom_dispatch = params.fun;
    sched->custom_backend = backend;
    ggml_backend_sched_set_tensor_backend(sched, op, backend);
    return true;
}

void ggml_backend_sched_mark_mtp_three_empty_wave(ggml_backend_sched_t sched, const ggml_cgraph * wave_zero) {
    if (sched->mtp_three_merge_empty_wave && sched->target_mtp_verify_three && !sched->profile.enabled &&
            !sched->callback_eval && sched->active_copy_split && wave_zero == &sched->active_copy_split->graph) {
        sched->mtp_three_empty_next = true;
    }
}

bool ggml_backend_sched_carry_accumulator(ggml_backend_sched_t sched, const ggml_cgraph * graph, ggml_tensor * accumulator) {
    auto * split = const_cast<ggml_backend_sched_split *>(sched->active_copy_split);
    if (!accumulator || !split || graph != &split->graph || sched->n_copies != 1 || sched->callback_eval ||
            sched->carry_accumulator || split->shared_prefix.n_nodes || accumulator->op != GGML_OP_ADD ||
            !ggml_is_contiguous(accumulator) || !ggml_are_same_layout(accumulator, accumulator->src[0]) ||
            !accumulator->data || !accumulator->src[0]->data ||
            tensor_backend_id(accumulator) != split->backend_id ||
            tensor_backend_id(accumulator->src[0]) != split->backend_id) {
        return false;
    }
    const auto * source = accumulator->src[0];
    const auto dst_addr = reinterpret_cast<uintptr_t>(accumulator->data);
    const auto src_addr = reinterpret_cast<uintptr_t>(source->data);
    if (dst_addr != src_addr && (dst_addr < src_addr ? src_addr - dst_addr : dst_addr - src_addr) < ggml_nbytes(accumulator)) {
        return false;
    }
    if (split->carry_plan_uid == graph->uid && split->carry_plan_node == accumulator) {
        sched->carry_prefix_end = split->carry_plan_end;
        sched->carry_accumulator = accumulator;
        return true;
    }
    int end = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == accumulator) {
            end = i;
            break;
        }
    }
    if (end < 0) {
        return false;
    }
    std::unordered_set<const ggml_tensor *> skipped_nodes;
    for (int i = 0; i <= end; ++i) {
        const auto * skipped = graph->nodes[i];
        if (skipped == accumulator) {
            continue;
        }
        if (skipped == accumulator->src[0] || (skipped->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        switch (skipped->op) {
            case GGML_OP_GET_ROWS:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_GLU:
            case GGML_OP_MUL:
            case GGML_OP_SCALE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_CONT:
                break;
            default:
                return false;
        }
        skipped_nodes.insert(skipped);
    }
    // Validate once for this allocated graph, including views and injected copy nodes.
    for (int j = 0; j < sched->graph.n_nodes; ++j) {
        const auto * node = sched->graph.nodes[j];
        if (node == accumulator || skipped_nodes.count(node)) {
            continue;
        }
        for (const auto * src : node->src) {
            for (const auto * view = src; view; view = view->view_src) {
                if (skipped_nodes.count(view)) {
                    return false;
                }
            }
        }
    }
    split->carry_plan_uid = graph->uid;
    split->carry_plan_node = accumulator;
    split->carry_plan_end = end + 1;
    sched->carry_prefix_end = end + 1;
    sched->carry_accumulator = accumulator;
    return true;
}

const struct ggml_tensor * ggml_backend_sched_get_post_copy_source(ggml_backend_sched_t sched,
        const struct ggml_cgraph * graph, const struct ggml_tensor * weight, const struct ggml_tensor * dst,
            const struct ggml_tensor * contribution) {
    const auto * split = sched->active_copy_split;
    if (!split || graph != &split->graph || sched->n_copies != 1 || sched->profile.enabled || sched->callback_eval) {
        return nullptr;
    }

    const ggml_tensor * source = nullptr;
    bool found_weight = false;
    for (int i = 0; i < split->n_inputs; ++i) {
        ggml_tensor * input = split->inputs[i];
        if (!sched->hv_tensor_requires_copy[hash_id(input)]) {
            continue;
        }
        const auto * copy = tensor_copy(input, split->backend_id, sched->cur_copy);
        if (ggml_backend_sched_is_host_weight(input)) {
            // A later weight callback could wait for CUDA after we enqueue work.
            if (copy != weight) {
                return nullptr;
            }
            found_weight = true;
        } else if (copy == dst) {
            source = input;
        }
    }
    if (!found_weight || !source || split == &sched->splits[sched->n_splits - 1]) {
        return nullptr;
    }

    const auto & next = split[1];
    for (int i = 0; i < next.graph.n_nodes; ++i) {
        const auto * merge = next.graph.nodes[i];
        if (merge->op != GGML_OP_ADD || merge->src[1] != contribution) {
            continue;
        }
        for (int j = 0; j < next.n_inputs; ++j) {
            const auto * input = next.inputs[j];
            if (tensor_copy(next.inputs[j], next.backend_id, sched->cur_copy) != merge->src[0]) {
                continue;
            }
            if (std::find(graph->nodes, graph->nodes + graph->n_nodes, input) != graph->nodes + graph->n_nodes) {
                return source;
            }
        }
    }
    return nullptr;
}

void ggml_backend_sched_set_early_copy_callback(ggml_backend_sched_t sched, ggml_backend_sched_copy_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_copy_early = callback;
    sched->callback_copy_early_user_data = user_data;
}

void ggml_backend_sched_set_host_copy_capture(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    sched->capture_host_copies = enabled;
}

bool ggml_backend_sched_get_host_copy(ggml_backend_sched_t sched, const ggml_tensor * dst, void * data, size_t size) {
    const auto * copies = sched->active_host_copies;
    if (copies) {
        for (size_t i = 0; i < copies->count; ++i) {
            const auto & entry = copies->entries[i];
            if (entry.dst == dst && entry.size == size) {
                memcpy(data, entry.payload, size);
                return true;
            }
        }
    }
    return false;
}

void ggml_backend_sched_set_split_profile(ggml_backend_sched_t sched, bool enabled) {
    GGML_ASSERT(sched);
    sched->profile.enabled = enabled;
    if (enabled && !sched->profile.cuda) {
        static const bool profile_cuda = [] {
            const char * env = getenv("LLAMA_DECODE_PROFILE_CUDA");
            if (!env || strcmp(env, "1") != 0) {
                return false;
            }
            if (!getenv("GGML_CUDA_DISABLE_GRAPHS")) {
                GGML_LOG_WARN("decode_profile_cuda: set GGML_CUDA_DISABLE_GRAPHS=1 to profile CUDA dispatches\n");
                return false;
            }
            GGML_LOG_INFO("decode_profile_cuda: synchronizing after each CUDA op or fused group; timings include host submission and stream wait\n");
            return true;
        }();
        if (profile_cuda) {
            sched->profile.cuda = new ggml_backend_sched_cuda_profile;
        }
    }
}

void ggml_backend_sched_set_realloc_callback(ggml_backend_sched_t sched, ggml_backend_sched_realloc_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_realloc = callback;
    sched->callback_realloc_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARGMAX:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
