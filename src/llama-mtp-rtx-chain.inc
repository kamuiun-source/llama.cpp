// All draft kernels use the same backend stream, but each forward owns its scheduler arena.
bool llama_context::process_mtp_rtx_chain(const llama_batch_ext & first, const llama_batch_ext & second, llama_token * ids) {
    if (!mtp_catchup_supported || !mtp_device_pending || mtp_draft_graph_cache2 || cparams.pipeline_parallel ||
            cparams.cb_eval || !cparams.embeddings_nextn_masked || sampling.samplers.size() != 1 ||
            first.tokens.size() < 2 || first.tokens.size() > 4 || second.tokens.size() != 1 ||
            mtp_vocab_head || model.dev_output() == nullptr) {
        return false;
    }
    int32_t n_suppress = 0;
    llama_vocab_get_suppress_tokens(&model.vocab, &n_suppress);
    if (n_suppress) {
        return false;
    }

    if (!mtp_rtx_chain) {
        auto chain = std::make_unique<llama_mtp_rtx_chain>();
        auto dev = model.dev_output();
        auto reg = ggml_backend_dev_backend_reg(dev);
        if (!reg || std::strcmp(ggml_backend_reg_name(reg), "CUDA") != 0) {
            return false;
        }
        chain->backend = backend_from_dev(dev);
        chain->publish = reinterpret_cast<llama_mtp_rtx_chain::publish_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_mtp_draft_chain"));
        chain->read_waits = reinterpret_cast<llama_mtp_rtx_chain::wait_fn>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_wait_telemetry"));
        if (!chain->backend || !chain->publish) {
            return false;
        }
        auto host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            return false;
        }
        constexpr size_t host_capacity = 512 * 1024;
        chain->storage.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
        if (!chain->storage) {
            return false;
        }
        chain->ids = ggml_new_tensor_1d(chain->storage.get(), GGML_TYPE_I32, 4);
        chain->first_id = ggml_view_1d(chain->storage.get(), chain->ids, 1, 0);
        chain->hidden = ggml_new_tensor_2d(chain->storage.get(), GGML_TYPE_F32, model.hparams.n_embd_out(), 1);
        ggml_set_name(chain->first_id, "mtp_rtx_first_id");
        ggml_set_name(chain->hidden, "mtp_rtx_first_hidden");
        chain->device.reset(ggml_backend_alloc_ctx_tensors_from_buft(chain->storage.get(), ggml_backend_get_default_buffer_type(chain->backend)));
        chain->host.reset(ggml_backend_buft_alloc_buffer(host_buft, host_capacity));
        if (!chain->device || !chain->host || ggml_backend_buffer_get_type(chain->host.get()) != host_buft) {
            return false;
        }
        chain->result = std::make_unique<llm_graph_result>(gf_res_reserve->get_max_nodes());
        chain->scheduler = sched_create(sched_params_cur);
        mtp_rtx_chain = std::move(chain);
    }

    auto & chain = *mtp_rtx_chain;
    auto finish = [&](bool success) {
        chain.step = -1;
        chain.applied += success;
        chain.failed += !success;
        return success;
    };
    chain.step = 0;
    try {
        if (decode(first, false, true, false, llama_mtp_draft_row::normal, true) != 0 ||
                !publish_mtp_rtx_draft(0) || sched_need_reserve) {
            synchronize();
            return finish(false);
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_WARN("MTP RTX draft chain first forward: %s\n", err.what());
        synchronize();
        return finish(false);
    } catch (...) {
        synchronize();
        return finish(false);
    }

    auto * primary_active = gf_res_prev_active;
    std::swap(sched, chain.scheduler);
    gf_res_prev_active = chain.active;
    chain.step = 1;
    chain.second_launch_us = 0;
    auto * host = static_cast<uint8_t *>(ggml_backend_buffer_get_base(chain.host.get()));
    ggml_backend_sched_set_mtp_draft_chain_inputs(sched.get(), chain.backend, host + 64,
            ggml_backend_buffer_get_size(chain.host.get()) - 64);
    uint64_t before[4] = {}, after[4] = {};
    if (chain.read_waits) {
        chain.read_waits(before);
    }
    const auto second_start = ggml_time_us();
    bool queued = false;
    try {
        queued = decode(second) == 0 && publish_mtp_rtx_draft(1);
    } catch (const std::exception & err) {
        LLAMA_LOG_WARN("MTP RTX draft chain second forward: %s\n", err.what());
    } catch (...) {
        LLAMA_LOG_WARN("MTP RTX draft chain second forward threw\n");
    }
    const auto second_end = ggml_time_us();
    if (chain.read_waits) {
        chain.read_waits(after);
        chain.second_stream_wait_ns += after[0] - before[0];
    }
    if (chain.second_launch_us) {
        chain.second_before_submit_us += chain.second_launch_us - second_start;
        chain.second_submit_us += second_end - chain.second_launch_us;
    }
    if (queued) {
        ggml_backend_tensor_get_async(chain.backend, chain.ids, host, 0, ggml_nbytes(chain.ids));
    }
    synchronize();
    const auto finish_us = ggml_time_us() - second_end;
    chain.finish_us += finish_us;

    ggml_backend_sched_set_mtp_draft_chain_inputs(sched.get(), nullptr, nullptr, 0);
    chain.active = gf_res_prev_active;
    gf_res_prev_active = primary_active;
    std::swap(sched, chain.scheduler);
    if (llama_mtp_rtx_chain_trace()) {
        LLAMA_LOG_INFO("MTP RTX draft boundary: mode=rtx-chain queued=%d before_submit_us=%" PRId64
                " submit_us=%" PRId64 " second_stream_wait_ns=%" PRIu64 " final_read_wait_us=%" PRId64 "\n",
                queued, chain.second_launch_us ? chain.second_launch_us - second_start : second_end - second_start,
                chain.second_launch_us ? second_end - chain.second_launch_us : 0,
                after[0] - before[0], finish_us);
    }
    auto * result = reinterpret_cast<const int32_t *>(host);
    if (!queued || !result[2] || !result[3]) {
        return finish(false);
    }
    ids[0] = result[0];
    ids[1] = result[1];
    return finish(true);
}

bool llama_context::publish_mtp_rtx_draft(int step) {
    auto & chain = *mtp_rtx_chain;
    auto * res = gf_res_prev_active;
    if (!res || res->t_sampled_logits.size() != 1 || res->t_candidates.size() != 1 ||
            (step == 0 && !res->get_h_nextn())) {
        return false;
    }
    return chain.publish(chain.backend, res->t_sampled_logits.front(), res->t_candidates.front(),
            step == 0 ? res->get_h_nextn() : nullptr, chain.ids, chain.hidden, step, model.vocab.n_tokens());
}
