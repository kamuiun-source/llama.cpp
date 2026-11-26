// Included after the scheduler and pending-return definitions.
static void ggml_backend_sched_record_return_ready(ggml_backend_sched_t sched, ggml_backend_sched_hip_return & pending) {
    pending.ready_requested = sched->target_mtp_verify_three && !sched->target_mtp_verify_four &&
            hip_return_ready_enabled.load(std::memory_order_relaxed);
    pending.ready_recorded = false;
    if (!pending.ready_requested) {
        return;
    }

    auto & ready = sched->hip_return_ready;
    const auto start = ggml_backend_sched_hip_verify_stage_time(sched);
    if (!ready.owner) {
        ready.owner = pending.source;
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(ready.owner));
        if (reg) {
            ready.api = static_cast<const ggml_backend_hip_return_ready_api *>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_return_ready_api"));
        }
        if (ready.api) {
            ready.event = ready.api->create(ready.owner);
        }
    }
    pending.ready_recorded = ready.owner == pending.source && ready.event && ready.api->record(ready.owner, ready.event);
    ready.recorded += pending.ready_recorded;
    ggml_backend_sched_hip_verify_stage_record(sched,
            pending.ready_recorded ? "hip_return_ready_record" : "hip_return_ready_record_fallback", start, false);
}

static void ggml_backend_sched_wait_return_ready(ggml_backend_sched_t sched, ggml_backend_sched_hip_return & pending) {
    if (!sched->target_mtp_verify_three || sched->target_mtp_verify_four) {
        pending.wait();
        return;
    }

    auto & ready = sched->hip_return_ready;
    const auto start = ggml_backend_sched_hip_verify_stage_time(sched);
    const bool used = pending.ready_recorded && ready.api->wait(ready.event);
    if (used) {
        pending.source = nullptr;
        ++ready.event_waits;
    } else {
        pending.wait();
        ++ready.stream_waits;
        ready.fallback += pending.ready_requested;
    }
    ggml_backend_sched_hip_verify_stage_record(sched, used ? "hip_return_ready_wait_event" :
            pending.ready_requested ? "hip_return_ready_wait_fallback" : "hip_return_ready_wait_stream", start, false);
}

static void ggml_backend_sched_free_return_ready(ggml_backend_sched_t sched) {
    const auto & ready = sched->hip_return_ready;
    if (ready.event) {
        ready.api->destroy(ready.event);
    }
    if (ready.event_waits || ready.fallback || (sched->hip_verify_trace && ready.stream_waits)) {
        GGML_LOG_INFO("HIP fixed3 return ready: records=%zu event_waits=%zu stream_waits=%zu fallback=%zu\n",
                ready.recorded, ready.event_waits, ready.stream_waits, ready.fallback);
    }
}
