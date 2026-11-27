#pragma once

#include "ggml-backend.h"
#include "ggml-impl.h"

extern "C" {
GGML_API void ggml_backend_sched_set_copy_api_batch(bool enabled);
}

static constexpr size_t ggml_copy_api_batch_max = 3;

struct ggml_copy_api_batch {
    using submit_t = bool (*)(ggml_backend_t, const uintptr_t *, const uintptr_t *, const size_t *, size_t);
    submit_t submit;
    bool lookup_done;
    size_t eligible[ggml_copy_api_batch_max + 1];
    size_t applied[ggml_copy_api_batch_max + 1];
    size_t eligible_bytes;
    size_t applied_bytes;
    size_t mapped_eligible;
    size_t mapped_applied;
    size_t unpinned;
    size_t single;
    size_t too_many;
    size_t overlap;
    size_t unavailable;

    static bool overlaps(uintptr_t a, size_t a_size, uintptr_t b, size_t b_size) {
        return a < b ? b - a < a_size : a - b < b_size;
    }

    bool read(ggml_backend_t backend, const ggml_tensor * const * tensors, uint8_t * staging,
            const size_t * sizes, const size_t * data_offsets, size_t count, bool pinned, bool mapped) {
        if (count < 2) {
            ++single;
            return false;
        }
        if (count > ggml_copy_api_batch_max) {
            ++too_many;
            return false;
        }
        if (!pinned) {
            ++unpinned;
            return false;
        }

        // The handoff reads whole tensors from one CUDA device buffer into a separate pinned slab.
        uintptr_t src[ggml_copy_api_batch_max];
        uintptr_t dst[ggml_copy_api_batch_max];
        size_t bytes = 0;
        for (size_t i = 0; i < count; ++i) {
            src[i] = reinterpret_cast<uintptr_t>(tensors[i]->data);
            dst[i] = reinterpret_cast<uintptr_t>(staging + data_offsets[i]);
            for (size_t j = 0; j <= i; ++j) {
                if (overlaps(dst[i], sizes[i], src[j], sizes[j]) ||
                        overlaps(src[i], sizes[i], dst[j], sizes[j]) ||
                        (j < i && (overlaps(dst[i], sizes[i], dst[j], sizes[j]) ||
                                   overlaps(src[i], sizes[i], src[j], sizes[j])))) {
                    ++overlap;
                    return false;
                }
            }
            bytes += sizes[i];
        }

        ++eligible[count];
        eligible_bytes += bytes;
        mapped_eligible += mapped;
        if (!lookup_done) {
            lookup_done = true;
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            submit = reinterpret_cast<submit_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mtp3_d2h_batch"));
        }
        if (!submit || !submit(backend, src, dst, sizes, count)) {
            ++unavailable;
            return false;
        }
        ++applied[count];
        applied_bytes += bytes;
        mapped_applied += mapped;
        return true;
    }

    void log() const {
        if (!(eligible[2] || eligible[3] || unpinned || single || too_many || overlap)) {
            return;
        }
        GGML_LOG_INFO("packed HIP MTP3 CUDA D2H batch: eligible_2=%zu eligible_3=%zu eligible_bytes=%zu applied_2=%zu applied_3=%zu applied_bytes=%zu mapped_eligible=%zu mapped_applied=%zu fallback_unavailable=%zu fallback_unpinned=%zu fallback_single=%zu fallback_too_many=%zu fallback_overlap=%zu\n",
                eligible[2], eligible[3], eligible_bytes, applied[2], applied[3], applied_bytes,
                mapped_eligible, mapped_applied, unavailable, unpinned, single, too_many, overlap);
    }
};
