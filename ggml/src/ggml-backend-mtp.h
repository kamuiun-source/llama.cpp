#pragma once

#include "ggml-backend.h"

extern "C" {
GGML_API bool ggml_backend_sched_set_target_mtp_verify_four(ggml_backend_sched_t sched, bool enabled);
}
