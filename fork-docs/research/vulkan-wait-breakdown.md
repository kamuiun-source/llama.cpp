# Vulkan wait causes and GPU command-buffer timings

The adopted diagnostic baseline enables CUDA Spin, Router host-ID reuse, device-input batching and deferred Vulkan slot maps. MTP anchor reuse remains off. Context 2048, target ubatch 2 / RS1 and one MTP candidate are unchanged.

Run `fork-docs/scripts/measure-vulkan-breakdown.ps1` to compare two executions: CPU call-site detail only, then the same settings with MI50 GPU timestamps. Logs are local only under `G:/AI/llama-hetero-moe-local/logs`. Both cases keep the adopted optimizations active; the old intrusive Vulkan per-operation logger is disabled.

## CPU fields

The existing `LLAMA_DECODE_TELEMETRY=1` phase summaries gain `vk_cause` rows for unclassified, graph_compute, backend_sync, sync_write, sync_read, sync_copy, slot_map, expert_upload and expert_writeback. They count native compute/transfer queue submissions and host time in submit, fence waits, polling, command recording and semaphore waits. Causes identify where the CPU call was issued, not what earlier GPU work it waits for. A generic sync_read does not necessarily mean one particular tensor or direction of model dataflow.

Cause rows partition the existing fields; do not add them to aggregate rows. `server_step_total` includes subphases and must not be added to them. Native wait time includes queued GPU work, copies and driver/notification delay; it is not all removable overhead.

## GPU fields

`GGML_VK_GPU_TIMESTAMPS_DEVICE=Vulkan1` requests timestamps for that backend device, subject to runtime capability and mode checks. The diagnostic records bounded pairs around graph compute command-buffer segments and recovers results without adding a submit or completion wait. The existing completion path is used; queue timestamp valid bits and period determine elapsed time.

`vk_gpu_compute_segment_ms` is GPU timeline time for measured segments, including shaders, in-graph copies, barriers and stalls. It is not pure ALU time. Prelude commands before the first marker and work on a separate transfer queue are excluded. Samples, dropped segments and CPU query-read time are reported so incomplete coverage or measurement overhead is visible. Exhausting the bounded pool must drop samples rather than forcing execution to synchronize.

CPU and GPU times overlap; do not add them or subtract GPU time from host waiting to claim driver overhead. Phase counters reflect the scope where samples are collected; delayed collection may shift a sample to a later scope. Prefer total coverage checks before interpreting per-phase GPU times. Compare output, acceptance, adopted-feature activation counts and submission counts between the CPU-only and timestamp runs.

## Limited validation

Changed translation units compiled with actual project flags. A small MI50 fixture produced matching outputs and four deferred map updates with timestamps off/on; both runs had identical native submission and wait counts. All ten CPU aggregate fields matched their cause-row sums. The timestamp run collected four samples without drops (family 1, valid bits 64, period 40 ns). This checks instrumentation mechanics only; full-model timings remain user-run.
