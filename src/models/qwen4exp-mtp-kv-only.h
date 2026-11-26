#pragma once

// QSA inputs and KV writes keep the full ubatch. Only the final query uses these views.
static ggml_tensor * qwen4exp_mtp_last_row(ggml_context * ctx, ggml_tensor * x) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (x->ne[1] - 1)*x->nb[1]);
}

static ggml_tensor * qwen4exp_mtp_last_pos(ggml_context * ctx, ggml_tensor * pos, int64_t n_tokens) {
    const int64_t n_sections = pos->ne[0] / n_tokens;
    const size_t stride = ggml_element_size(pos);
    // M-RoPE positions are channel-major.
    auto * last = ggml_view_2d(ctx, pos, 1, n_sections, n_tokens*stride, (n_tokens - 1)*stride);
    return ggml_cont_1d(ctx, last, n_sections);
}

static ggml_tensor * qwen4exp_mtp_last_residual(ggml_context * ctx, ggml_tensor * x) {
    return ggml_view_3d(ctx, x, x->ne[0], x->ne[1], 1, x->nb[1], x->nb[2], (x->ne[2] - 1)*x->nb[2]);
}
