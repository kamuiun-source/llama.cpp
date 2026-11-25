#pragma once

#include "ggml-backend.h"

extern "C" {
GGML_API bool ggml_backend_sched_set_target_mtp_verify_four(ggml_backend_sched_t sched, bool enabled);
GGML_API int64_t ggml_backend_sched_hip_verify_stage_time(ggml_backend_sched_t sched);
GGML_API void ggml_backend_sched_hip_verify_stage_record(ggml_backend_sched_t sched, const char * stage, int64_t start, bool before_compute);
}
