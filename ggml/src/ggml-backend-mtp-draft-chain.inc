void ggml_backend_sched_set_mtp_draft_chain_inputs(ggml_backend_sched_t sched, ggml_backend_t backend,
        void * pinned, size_t capacity) {
    sched->mtp_draft_chain_inputs = {backend, static_cast<uint8_t *>(pinned), capacity, 0};
}

static bool ggml_backend_sched_mtp_draft_chain_input(ggml_backend_sched_t sched, ggml_backend_t backend,
        const ggml_tensor * input, ggml_tensor * copy) {
    auto & chain = sched->mtp_draft_chain_inputs;
    const ggml_tensor * root = input;
    while (root->view_src) {
        root = root->view_src;
    }
    if (chain.backend != backend || !(root->flags & GGML_TENSOR_FLAG_INPUT) ||
            !input->buffer || !ggml_backend_buffer_is_host(input->buffer) || !copy->buffer ||
            ggml_backend_buffer_is_host(copy->buffer) || !ggml_are_same_layout(input, copy) || !ggml_is_contiguous(input)) {
        return false;
    }
    const size_t bytes = ggml_nbytes(input);
    if (bytes > chain.capacity - chain.used) {
        return false;
    }
    auto * staging = chain.pinned + chain.used;
    memcpy(staging, input->data, bytes);
    ggml_backend_tensor_set_async(backend, copy, staging, 0, bytes);
    chain.used += bytes;
    return true;
}
