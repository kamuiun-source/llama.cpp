# Vulkan submission reduction

The experiment compares two opt-in changes on the adopted synchronous Flash-Next setup. CUDA Spin, router host-ID reuse, device-input batching, deferred Vulkan slot maps and dynamic Expert residency are enabled in every case. The target uses microbatch 2 and RS1, and MTP proposes one token. MTP anchor reuse and Expert overlap are off.

| Case | Final graph fence reuse | Compact Expert submissions |
| --- | --- | --- |
| baseline | off | off |
| fence-only | on | off |
| compact-only | off | on |
| both | on | on |

`GGML_VK_REUSE_GRAPH_FENCE=1` allows the final graph submission to carry the completion fence, skipping an empty completion submission where eligible. `GGML_VK_COMPACT_EXPERT_SUBMITS=1` suppresses eligible intermediate submit boundaries in the Expert compute path. Each enabled case checks its feature-use marker.

From the repository root after building:

```powershell
& .\fork-docs\scripts\compare-vulkan-submit.ps1
& .\fork-docs\scripts\compare-vulkan-submit.ps1 -Mode telemetry
```

Use the default speed run for throughput. It runs four 256-token cases with CPU telemetry and GPU timestamps disabled. Telemetry mode repeats the four cases with CPU call-site counters and MI50 compute-segment timestamps. Logs, arguments, environment settings and normalized output comparisons stay under `G:/AI/llama-hetero-moe-local/logs`.

Check output equality and compare native submit counts for `target_verify` and `server_step_total`. The latter includes the phase counts. A feature-use marker confirms that a path ran at least once. Use native submit counts to measure how often submissions were eliminated. GPU segment coverage can differ when boundaries move, so compare sample counts and drops alongside GPU time. GPU segment time can overlap CPU waits.

## Scope and validation

Fence reuse is disabled for serialized submissions, asynchronous transfer queues, internally synchronized native queues and intrusive Vulkan result/performance checks. A later native compute-queue submission invalidates fence coverage and uses a separate completion fence before resetting the graph fence. When the reused final fence has no earlier ready fence, the host blocks on it instead of spinning for the full graph.

Compact mode applies to graphs with at most 32 nodes and MUL_MAT_ID token batches of at most two. It suppresses the almost-ready boundary only; existing node and FLOP submission limits remain. It does not change shader arithmetic or Expert placement.

Actual build flags compiled the changed Vulkan code and header consumers. Four MI50 fixture configurations preserved Expert results, delayed slot-map updates, later event submissions, fence reuse on the following graph, and GPU samples with zero drops. The ordinary fixture path changed graph plus completion submits from 2+1 (baseline), to 2+0 (fence-only), 1+1 (compact-only), and 1+0 (both). These are small-fixture results, not full-model speed measurements. Raw logs remain outside Git.
