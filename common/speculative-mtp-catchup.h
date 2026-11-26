#pragma once

#include "common.h"

#include <array>

struct common_mtp_catchup {
    struct row {
        llama_token token;
        llama_pos pos;
        int32_t state;
    };
    std::array<row, 3> rows{};
    int count = 0;
    bool confirmed = false;
    bool history_kv_only = false;

    void append(common_batch & batch) const {
        for (int i = 0; i < count; ++i) {
            const auto & row = rows[i];
            const int idx = batch.add(row.token, row.pos, 0, false);
            batch.set_embd_state_row(idx, row.state);
        }
    }

    llama_pos first() const { return rows.front().pos; }
    llama_pos last() const { return rows[count - 1].pos; }
    int32_t accepted_row() const { return count - 1; }
};
