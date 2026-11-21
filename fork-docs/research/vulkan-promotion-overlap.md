# Dynamic promotion transfer overlap

This experiment extends `LLAMA_EXPERT_VK_OVERLAP_UPLOAD=1` with `LLAMA_EXPERT_VK_OVERLAP_PROMOTION=1`. Baseline comparisons retain the adopted upload overlap, dynamic residency, scratch reuse and two-token MTP verification; only promotion overlap changes.

A selected scratch Expert can become resident without moving its GPU weights. Its former RAM slot becomes the home of an unselected resident victim. The victim must be copied back before the ownership mapping is committed. The original promotion policy and resident age threshold remain unchanged.

The opt-in path writes the victim into native pinned staging on the dedicated transfer queue while selected hot Experts compute. Cold uploads, when needed, use disjoint staging and GPU ranges. The promoted Expert's RAM original stays intact until transfer completion. The existing backend completion path then copies the victim to that RAM slot and commits the resident/scratch ownership exchange. There is no extra persistent duplicate of the Expert bank.

Unsupported paths retain synchronous promotion. Pending transfer completion must also be handled when the graph is cancelled or the backend is destroyed. Completion callbacks must not allocate or throw, and must finish before the cache or staging is freed.

## Measurement

Use the comparison script in speed mode for throughput and telemetry mode for wait attribution. Compare generated text, MTP acceptance, Expert hits, promotions, H2D bytes and D2H bytes before judging speed. All raw logs remain outside Git under `G:/AI/llama-hetero-moe-local/logs`.

Asynchronous callback timings do not represent total transfer duration. Fence wait time includes GPU work; host waits and GPU intervals must not be added together. A transfer still pending at compute submission demonstrates asynchronous submission, not the duration of GPU execution overlap.

## Validation status

MSVC syntax checks for the cache, compilation of the Vulkan translation unit with project flags, the cache helper fixture and PowerShell 5.1 parsing passed. Four small MI50 fixture categories passed: mixed upload/writeback, writeback-only promotion, cancellation before graph execution, and invalid victim/staging rejection. Completed paths preserved output and committed exactly once after synchronization. These tests use small F32 tensors; full-model IQ3_XXS correctness and performance remain for the user comparison.
Run `fork-docs/scripts/compare-vulkan-promotion-overlap.ps1 -Mode speed` and then `-Mode telemetry`. The script requires matched output, Expert counts, transfer bytes and MTP acceptance, plus matched native/cache promotion commit counters. `expert overlap transfer_submits` now includes writeback-only submissions; use `expert promotion overlap` counters to identify promotion traffic. `writeback_ms` covers the remaining synchronous path, not asynchronous D2H duration.
