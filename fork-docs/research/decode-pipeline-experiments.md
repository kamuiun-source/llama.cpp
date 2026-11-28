# Decode pipeline experiments

As of 2026-11-17, the user accepted device-input batching and deferred Vulkan slot-map updates for the baseline configuration, alongside Spin and Router host-ID reuse. MTP anchor reuse remains optional and off in subsequent diagnostic baselines. All features remain opt-in in code. The 256-token speed and telemetry comparisons produced matching text and draft acceptance across all four cases; this does not establish equivalence for every request.

## Device inputs

`GGML_SCHED_BATCH_DEVICE_INPUTS=1` groups compatible CUDA/Vulkan split inputs, waits for their source producers and previous destination users once, and reuses synchronous bulk get/set APIs. It retains exact Router ID host copies. Unknown buffer types, unsupported layouts, events, overlapping destinations and unsupported copy configurations use the old path. The optimization does not remove the required GPU completion dependencies.

## Vulkan slot maps

`LLAMA_EXPERT_VK_DEFER_SLOT_MAP=1` records eligible slot-map updates into the pending compute command buffer before Expert execution. The update and compute barriers preserve ordering; Expert uploads, promotion and writeback remain unchanged. Unsupported execution/profiling modes use the synchronous setter.

## MTP anchor

`LLAMA_MTP_REUSE_DRAFT_ANCHOR=1` retains an eligible one-token Qwen4exp draft anchor and avoids decoding it again in catchup. Candidate processing and target hidden-state indexing remain necessary. It is limited to the supported standalone one-head, one-candidate, single-sequence configuration; this does not increase speculative width.

## Comparison

Run `fork-docs/scripts/compare-decode-pipeline.ps1 -Mode speed` for four cumulative cases: baseline (Spin and Router reuse), device-input batching, plus deferred slot maps, plus MTP anchor reuse. `-Mode telemetry` repeats the same cases with non-forcing counters enabled. Logs, settings, stdout/stderr, exit status and working diff are saved outside Git under `G:/AI/llama-hetero-moe-local/logs`.

All cases use context 2048, target ubatch 2 / RS1, one draft candidate, draft ubatch 1, dynamic residency, scratch reuse and native batch uploads. The scripts check activation counters rather than accepting a silently inactive optimization. Compare output, accepted drafts, Expert requests/transfers, target verification counts and draft catchup graphs as well as wall time. Counter totals can include prefill; use phase-specific telemetry when interpreting decode work. Normalized stdout equality is only a test of that request, not general numerical equivalence.

Do not add raw benchmark artifacts to Git. Tests are bounded per item and full builds/inference remain with the user. Existing batch-2 MTP output differences from serial verification are a separate unresolved baseline issue.
