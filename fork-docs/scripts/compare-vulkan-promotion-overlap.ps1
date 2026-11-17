[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry')]
    [string]$Mode = 'speed',
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256
)

$ErrorActionPreference = 'Continue'
$root = 'G:\AI\llama-hetero-moe'
$cli = Join-Path $root 'build-win-cuda-vulkan\bin\llama-cli.exe'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$placement = 'C:\Users\kamui\.codex\visualizations\2026\11\12\01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f\flash-next-top300.txt'
$draft = Join-Path (Split-Path (Split-Path $model)) 'MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $model, $placement, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}

$stamp = Get-Date -Format yyyyMMdd-HHmmss
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "vulkan-promotion-overlap-$Mode-$stamp"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_TOKENS', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY', 'LLAMA_EXPERT_ROUTER_HOST_IDS',
    'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'LLAMA_DECODE_TELEMETRY', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_SCHEDULE_SPIN', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_EXPERT_VK_DEFER_SLOT_MAP', 'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_REUSE_GRAPH_FENCE', 'GGML_VK_COMPACT_EXPERT_SUBMITS', 'LLAMA_EXPERT_VK_DEFER_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_PROMOTION', 'GGML_CUDA_GPU_TIMESTAMPS_DEVICE', 'GGML_CUDA_PROFILE_COMPONENTS')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }

try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') }
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    $env:LLAMA_EXPERT_PLACEMENT = $placement
    $env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
    $env:LLAMA_EXPERT_SCRATCH_REUSE = '1'
    $env:LLAMA_EXPERT_BATCH_UPLOAD = '1'
    $env:LLAMA_EXPERT_DYNAMIC_RESIDENT = '1'
    $env:LLAMA_EXPERT_BATCH_TOKENS = '2'
    $env:LLAMA_EXPERT_VK_DEFER_SLOT_MAP = '1'
    $env:LLAMA_MTP_REUSE_DRAFT_ANCHOR = '0'
    $env:GGML_SCHED_BATCH_DEVICE_INPUTS = '1'
    $env:GGML_CUDA_SCHEDULE_SPIN = '1'
    $env:LLAMA_EXPERT_ROUTER_HOST_IDS = '1'
    if ($Mode -eq 'telemetry') {
        $env:LLAMA_DECODE_TELEMETRY = '1'
        $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
    }

    $env:GGML_VK_REUSE_GRAPH_FENCE = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_UPLOAD = '1'

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $cases = @(
        @{ Name='baseline'; Promotion=$false }
        @{ Name='promotion-overlap'; Promotion=$true }
    )

    $baseArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    if ($LASTEXITCODE -ne 0) { throw "Could not record HEAD. Logs: $runDir" }
    git -C $root status --short | Set-Content (Join-Path $runDir 'working-status.txt') -Encoding UTF8
    git -C $root diff HEAD --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt') -Encoding UTF8
    $diffPath = Join-Path $runDir 'working-diff.patch'
    git -C $root diff --no-ext-diff --no-color --output="$diffPath" HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not record working diff. Logs: $runDir" }

    foreach ($case in $cases) {
        [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_VK_OVERLAP_PROMOTION', $(if ($case.Promotion) { '1' } else { [NullString]::Value }), 'Process')
        $prefix = Join-Path $runDir $case.Name
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        [ordered]@{ Name=$case.Name; Mode=$Mode; Tokens=$Tokens; Promotion=$case.Promotion; Environment=$envSettings } |
            ConvertTo-Json -Depth 3 | Set-Content "$prefix.settings.json" -Encoding UTF8
        $runArgs = @('-lv', '4', '--log-file', "$prefix.log") + $baseArgs
        $runArgs | ConvertTo-Json | Set-Content "$prefix.arguments.json" -Encoding UTF8
        Write-Host "Running $Mode/$($case.Name)"
        & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
        $code = $LASTEXITCODE
        "exit_code=$code" | Set-Content "$prefix.exit.txt"
        if ($code -ne 0) { throw "Failed: $($case.Name). Logs: $runDir" }

        $logs = @("$prefix.log", "$prefix.stderr.txt")
        if (!(@(Select-String -LiteralPath $logs -SimpleMatch 'packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1').Count -gt 0)) {
            throw "Expected target batch width 2 and RS1. Logs: $runDir"
        }
        if (!(@(Select-String -LiteralPath $logs -Pattern 'draft acceptance = .* / +[1-9][0-9]* generated').Count -gt 0)) {
            throw "No positive draft count found. Logs: $runDir"
        }

        foreach ($pattern in @('CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1',
                'packed expert router: host_hits=[1-9][0-9]*\b')) {
            if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                throw "Missing adopted feature marker: $pattern. Logs: $runDir"
            }
        }
        if (@(Select-String -LiteralPath $logs -SimpleMatch 'final graph fence reused').Count -eq 0) {
            throw "No final graph fence reuse recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -SimpleMatch 'compact expert submit applied').Count -gt 0) {
            throw "Compact expert submit unexpectedly applied in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert Vulkan upload overlap: submitted=[1-9][0-9]* bytes=[1-9][0-9]*').Count -eq 0) {
            throw "Existing upload overlap did not activate in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'expert overlap transfer_submits=[1-9][0-9]* .*split_projections=[1-9][0-9]* .*paired_submits=[1-9][0-9]*').Count -eq 0) {
            throw "No hot/cold paired dispatch recorded in $($case.Name). Logs: $runDir"
        }
        $cachePromotion = Select-String -LiteralPath "$prefix.log" -Pattern 'packed expert Vulkan promotion overlap: committed=([1-9][0-9]*) writeback_bytes=([1-9][0-9]*)' | Select-Object -Last 1
        $nativePromotion = Select-String -LiteralPath "$prefix.log" -Pattern 'Vulkan1 expert promotion overlap submits=([1-9][0-9]*) commits=([1-9][0-9]*) writeback_bytes=([1-9][0-9]*) transfer_pending_at_compute_submit=([0-9]+)' | Select-Object -Last 1
        if ($case.Promotion) {
            if (!$cachePromotion -or !$nativePromotion) {
                throw "No completed overlapping promotion in $($case.Name). Logs: $runDir"
            }
            $cacheFields = $cachePromotion.Matches[0].Groups
            $nativeFields = $nativePromotion.Matches[0].Groups
            if ($nativeFields[1].Value -ne $nativeFields[2].Value -or $nativeFields[2].Value -ne $cacheFields[1].Value -or
                    $nativeFields[3].Value -ne $cacheFields[2].Value) {
                throw "Native and cache promotion counts or bytes differ. Logs: $runDir"
            }
            if ($Mode -eq 'telemetry' -and $nativeFields[4].Value -eq '0') {
                Write-Warning "No promotion writeback was observed still pending after compute submit. Logs: $runDir"
            }
        } elseif ($cachePromotion -or $nativePromotion) {
            throw "Promotion overlap unexpectedly activated in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*\b').Count -eq 0) {
            throw "No batched device inputs recorded in $($case.Name). Logs: $runDir"
        }
        if (@(Select-String -LiteralPath $logs -Pattern 'packed expert slot map updates: deferred=[1-9][0-9]*\b').Count -eq 0) {
            throw "No deferred slot-map updates recorded in $($case.Name). Logs: $runDir"
        }
        Select-String -LiteralPath $logs -Pattern 'sched_batch_device_inputs: device_input_batches=', 'packed expert slot map updates:', 'final graph fence reused', 'packed expert Vulkan upload overlap:', 'expert overlap transfer_submits=', 'packed expert Vulkan promotion overlap:', 'expert promotion overlap submits=' |
            ForEach-Object Line | Select-Object -Unique | Set-Content "$prefix.activation.txt"
        if ($Mode -eq 'telemetry') {
            foreach ($pattern in @('decode_telemetry: phase=target_verify .*scheduled_splits=[1-9][0-9]*\b',
                    'decode_telemetry: phase=server_step_total vk_compute_submits=',
                    'decode_telemetry: phase=target_verify vk_cause=',
                    'decode_telemetry: phase=target_verify vk_gpu_compute_segment_samples=[1-9][0-9]*\b')) {
                if (@(Select-String -LiteralPath $logs -Pattern $pattern).Count -eq 0) {
                    throw "Missing Vulkan telemetry row: $pattern. Logs: $runDir"
                }
            }
        }
    }

    $stats = @{}
    foreach ($case in $cases) {
        $prefix = Join-Path $runDir $case.Name
        $rows = foreach ($pattern in @('packed expert slots: .*permanent_hits=', 'packed expert requests: total=',
                'packed expert dynamic residency: promotions=', 'draft acceptance =')) {
            $row = Select-String -LiteralPath "$prefix.log" -Pattern $pattern | Select-Object -Last 1
            if (!$row) { throw "Missing summary in $($case.Name): $pattern. Logs: $runDir" }
            $row.Line
        }
        $rows | Set-Content "$prefix.expert-summary.txt"
        $stats[$case.Name] = (($rows | ForEach-Object {
            [regex]::Matches($_, '(?:permanent_hits|scratch_hits|ram_uploads|total|misses|promotions|evictions|writeback_bytes|uploaded_bytes|total_transfer_bytes)=\d+|\d+\s+(?:accepted|generated)') |
                ForEach-Object Value
        }) -join "`n")
    }
    $statsSame = $stats['baseline'] -ceq $stats['promotion-overlap']
    "expert_counts_bytes_and_mtp_equal=$statsSame" | Set-Content (Join-Path $runDir 'expert-comparison.txt')
    $texts = @{}
    foreach ($case in $cases) {
        $text = Get-Content -LiteralPath (Join-Path $runDir "$($case.Name).stdout.txt") -Raw
        if ([string]::IsNullOrWhiteSpace($text)) { throw "Missing stdout for $($case.Name). Logs: $runDir" }
        $texts[$case.Name] = ($text -replace '\x1b\[[0-?]*[ -/]*[@-~]', '' -replace '\[ Prompt:.*?\]', '').Trim()
    }
    $same = $texts['promotion-overlap'] -ceq $texts['baseline']
    "promotion-overlap normalized_stdout_equal_baseline=$same" | Set-Content (Join-Path $runDir 'output-comparison.txt')
    if (!$same) { throw "Normalized stdout differs from baseline. Logs: $runDir" }
    if (!$statsSame) { throw "Expert counts, transfer bytes or MTP acceptance differ. Logs: $runDir" }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { if ($null -eq $saved[$key]) { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') } else { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') } }
    Write-Host "Log directory: $runDir"
}
