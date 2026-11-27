#pragma once

static std::atomic<bool> hip_return_ticket_enabled{false};

void ggml_backend_sched_set_hip_return_ticket(bool enabled) {
    hip_return_ticket_enabled.store(enabled, std::memory_order_relaxed);
}

static bool ggml_backend_sched_hip_return_ticket_candidate(ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        const ggml_backend_sched_hip_return & pending, const ggml_tensor * shared_return_input) {
    if (!pending.mapped || !pending.source || split->n_inputs != 1 || split->inputs[0] != pending.input ||
            shared_return_input || !sched->target_mtp_verify_three || sched->n_copies != 1 ||
            sched->profile.enabled || sched->callback_eval || split->has_custom ||
            !sched->hip_return_ordered_stream || !split->graph.n_nodes ||
            sched->events[split->backend_id][sched->cur_copy] ||
            pending.copy != tensor_copy(split->inputs[0], split->backend_id, sched->cur_copy)) {
        return false;
    }
    // Moving this copy past the generic prefetch must not reorder host prefetches.
    if (split + 1 < sched->splits + sched->n_splits) {
        const auto & next = split[1];
        for (int i = 0; i < next.n_inputs; ++i) {
            auto * input = next.inputs[i];
            if (tensor_backend_id(input) == split->backend_id && sched->hv_tensor_requires_copy[hash_id(input)]) {
                const auto * copy = tensor_copy(input, next.backend_id, sched->cur_copy);
                if (copy->buffer && ggml_backend_buffer_is_host(copy->buffer)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_sched_hip_return_ticket_compute(ggml_backend_sched_t sched, ggml_backend_sched_split * split,
        ggml_backend_sched_hip_return & pending, ggml_cgraph * graph, uint64_t uid,
        const ggml_tensor * mirror, void * mirror_data, enum ggml_status * status) {
    auto * dst = sched->backends[split->backend_id];
    const auto reg = ggml_backend_dev_backend_reg(dst->device);
    auto launch = reinterpret_cast<ggml_backend_cuda_hip_return_ticket_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_hip_return_ticket"));
    if (!launch || !ggml_backend_hip_return_can_queue_h2d(dst, sched->hip_return_h2d_event)) {
        return false;
    }
    GGML_ASSERT(!sched->hip_return_h2d_pending);
    const auto * input = pending.input;
    auto * src = pending.source;
    ggml_backend_handoff_times times;
    times.enabled = true;
    times.read_start_us = pending.enqueue_start_us;
    times.read_end_us = pending.enqueue_end_us;
    struct ready_context {
        ggml_backend_sched_t sched;
        ggml_backend_sched_split * split;
        ggml_backend_sched_hip_return & pending;
        ggml_backend_handoff_times & times;
    } ctx{sched, split, pending, times};
    ggml_backend_hip_return_ticket_submit submit{
        sched->hip_return_ticket, pending.copy, sched->hip_return_h2d_event, pending.data,
        [](void * user_data) {
            const auto & data = *static_cast<const ready_context *>(user_data);
            auto * src = data.pending.source;
            data.times.source_wait([&] { data.pending.wait(); });
            if (data.sched->mapped_hip_input && data.sched->hip_async_input_pending == src) {
                data.sched->hip_async_input_pending = nullptr;
            }
            auto * correction = data.sched->active_single_miss;
            if (correction && correction->destination == data.split && correction->output == data.pending.input && correction->resolve) {
                ggml_backend_sched_combine_hip_return(data.sched, data.pending, *correction, ggml_time_us());
            }
        }, &ctx, sched->trace_hip_handoff_wait || sched->trace_handoff_graph != 0
    };
    if (!launch(dst, graph, uid, mirror, mirror_data, submit, status)) {
        return false;
    }
    if (*status == GGML_STATUS_SUCCESS) {
        ++sched->hip_return_ticket_applied;
        ++sched->hip_return_h2d_applied;
        sched->hip_return_h2d_pending = dst;
        sched->hip_return_wait_pending_mode = hip_return_dst_wait_skip.load(std::memory_order_relaxed);
        auto & counters = sched->hip_return_wait_stats[sched->hip_return_wait_pending_mode];
        ++counters.source_waits;
        counters.source_us += times.source_wait_us;
        times.write_start_us = submit.copy_start_us;
        times.write_end_us = submit.copy_end_us;
        times.write_us = submit.copy_end_us - submit.copy_start_us;
        const size_t bytes = ggml_nbytes(input);
        ggml_backend_sched_record_hip_handoff(sched, src, 1, bytes, true, times, false, false, false, true);
        ggml_backend_sched_log_handoff(sched, split, src, dst, &input, 1, bytes, times, "host_dual_pinned_mapped_return_ticket_h2d");
        if (sched->trace_handoff_graph) {
            ggml_backend_sched_trace_submit(sched, split - sched->splits, dst, graph, submit.graph_start_us, submit.graph_end_us);
            GGML_LOG_INFO("hip_return_ticket: graph=%lld split=%td sequence=%u launch_start_us=%lld launch_end_us=%lld\n",
                    (long long) sched->trace_handoff_graph, split - sched->splits, submit.ticket.sequence,
                    (long long) submit.graph_start_us, (long long) submit.graph_end_us);
        }
    }
    return true;
}
