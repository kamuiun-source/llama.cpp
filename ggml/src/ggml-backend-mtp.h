#pragma once

#include "ggml-backend.h"

extern "C" {
GGML_API enum ggml_status ggml_backend_sched_flush_deferred_router(ggml_backend_sched_t sched, const struct ggml_tensor * ids);
GGML_API bool ggml_backend_sched_set_target_mtp_verify_four(ggml_backend_sched_t sched, bool enabled);
GGML_API int64_t ggml_backend_sched_hip_verify_stage_time(ggml_backend_sched_t sched);
GGML_API void ggml_backend_sched_hip_verify_stage_record(ggml_backend_sched_t sched, const char * stage, int64_t start, bool before_compute);
GGML_API void ggml_backend_sched_hip_verify_stage_poll_flush(ggml_backend_sched_t sched);
// During the packed Expert callback, join a deferred destination wait before any GPU dependency.
GGML_API bool ggml_backend_sched_expert_copy_wait_pending(ggml_backend_sched_t sched);
GGML_API void ggml_backend_sched_wait_expert_copy(ggml_backend_sched_t sched, bool after_router, bool after_plan);
}
