#pragma once

#include <cstdlib>
#include <cstring>

inline const char * llama_mtp_catchup_mode_file() {
    const char * path = std::getenv("LLAMA_MTP_CATCHUP_FIRST_DRAFT_MODE_FILE");
    return path && *path ? path : nullptr;
}

inline bool llama_mtp_catchup_diagnostics() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MTP_CATCHUP_FIRST_DRAFT_DIAGNOSTICS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
