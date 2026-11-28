# Vulkan transfer-to-compute handoff

The comparison keeps final graph fence reuse enabled in both cases and leaves compact Expert submits disabled. The only per-case setting is `LLAMA_EXPERT_VK_DEFER_UPLOAD`, unset for `baseline` and set to `1` for `handoff`. MTP verification keeps target microbatch 2 and RS1. Draft anchor reuse and Expert overlap stay disabled. Eligible packed Expert uploads use the deferred slot-map handoff to the Vulkan compute graph. Promotion candidates and unsupported paths retain the synchronous upload path.

## Run

From the directory containing `compare-vulkan-transfer-compute.ps1`, use Windows PowerShell after building `llama-cli` with the handoff change.

```powershell
.\compare-vulkan-transfer-compute.ps1
.\compare-vulkan-transfer-compute.ps1 -Mode telemetry
```

The default `speed` run disables telemetry and timestamps. Use its decode throughput for the performance comparison. `telemetry` enables phase CPU counters and Vulkan1 timestamps. Use it to compare Expert upload and graph-compute submit counts, host wait time and GPU compute segment time. The inclusive `server_step_total` row already covers its inner phases. GPU compute segments cover measured GPU compute work, while CPU wait times include waiting on the host. A decrease in CPU wait time alone cannot establish GPU overlap.

Per-case arguments, environment, exit code, activation lines, raw logs and the normalized stdout comparison are saved below `G:\AI\llama-hetero-moe-local\logs`. Both cases must report graph-fence reuse, and only `handoff` may report a positive deferred count in the packed expert upload deferral row.

## Results

Pending execution on the benchmark system.

## Scope and validation

Final graph fence reuse is the adopted comparison baseline. Compact Expert submissions remain off. The new handoff submits the existing pinned upload command buffer without waiting; slot-map barriers and Expert computation follow on the same compute queue. It does not enable simultaneous transfer and Expert computation on separate queues.

The path requires a single-copy scheduler, native packed staging, deferred Vulkan slot maps, no promotion candidate and no intrusive profiling. Async transfer queues, serialized submissions and intrusive Vulkan checks retain the synchronous path. The existing synchronization before the next weight callback protects staging reuse, and context destruction synchronizes before releasing the cache.

Actual compiler settings built both changed translation units. Four small MI50 fixtures checked synchronous upload, deferred upload, no graph followed by synchronization and staging reuse, and rejected map deferral followed by synchronous publication. Outputs matched. Eligible upload submits changed from two to one and immediate upload polling from one to zero. Full-model speed remains unmeasured.

With deferred uploads, `packed expert transfer phase` and its aggregate no longer include the deferred path's staging copy, submit or later completion wait. Do not compare this field as total transfer time across modes. Use overall generation time and phase telemetry, including graph/backend waits; work can move between scopes. GPU compute segment timestamps exclude uploads submitted before the graph markers.
