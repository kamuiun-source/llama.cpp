[CmdletBinding()]
param(
    [ValidateRange(1, 2048)]
    [int]$Tokens = 256,
    [ValidateSet('both', 'speed', 'trace')]
    [string]$Mode = 'both',
    [switch]$FullHot
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
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "gpu-expert-handoff-$stamp"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_TOKENS', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY', 'LLAMA_EXPERT_ROUTER_HOST_IDS',
    'GGML_VK_GPU_TIMESTAMPS_DEVICE', 'LLAMA_DECODE_TELEMETRY', 'LLAMA_EXPERT_MISS_HISTOGRAM', 'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA',
    'GGML_CUDA_SCHEDULE_SPIN', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_GRAPH_OPT',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_GRAPH_CHUNK_SIZE', 'GGML_CUDA_POOL_DEBUG',
    'LLAMA_EXPERT_VK_DEFER_SLOT_MAP', 'LLAMA_MTP_REUSE_DRAFT_ANCHOR', 'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION', 'GGML_SCHED_BATCH_HOST_INPUTS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_VK_SERIALIZE_SUBMISSIONS', 'GGML_VK_PERF_LOGGER',
    'GGML_VK_PERF_LOGGER_CONCURRENT', 'GGML_VK_PERF_LOGGER_FREQUENCY', 'GGML_VK_SYNC_LOGGER',
    'GGML_VK_MEMORY_LOGGER', 'GGML_VK_PIPELINE_STATS',
    'GGML_VK_REUSE_GRAPH_FENCE', 'GGML_VK_COMPACT_EXPERT_SUBMITS', 'LLAMA_EXPERT_VK_DEFER_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_PROMOTION', 'GGML_CUDA_GPU_TIMESTAMPS_DEVICE', 'GGML_CUDA_PROFILE_COMPONENTS', 'GGML_SCHED_TRACE_EXPERT_HANDOFF', 'LLAMA_EXPERT_RTX_COLD_SLOTS', 'LLAMA_EXPERT_RTX_COLD_VRAM_MIB', 'LLAMA_EXPERT_VK_FULL_HOT_OVERLAP')
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

    $env:GGML_VK_REUSE_GRAPH_FENCE = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_UPLOAD = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_PROMOTION = '1'
    if ($FullHot) { $env:LLAMA_EXPERT_VK_FULL_HOT_OVERLAP = '1' }

    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')

    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    if ($LASTEXITCODE -ne 0) { throw "Could not record HEAD. Logs: $runDir" }
    git -C $root status --short | Set-Content (Join-Path $runDir 'working-status.txt') -Encoding UTF8
    git -C $root diff HEAD --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt') -Encoding UTF8
    $diffPath = Join-Path $runDir 'working-diff.patch'
    git -C $root diff --no-ext-diff --no-color --output="$diffPath" HEAD
    if ($LASTEXITCODE -ne 0) { throw "Could not record working diff. Logs: $runDir" }

    $cases = if ($Mode -eq 'both') { @('speed', 'trace') } else { @($Mode) }
    foreach ($case in $cases) {
        if ($case -eq 'trace') {
            $env:GGML_SCHED_TRACE_EXPERT_HANDOFF = '1'
            $env:LLAMA_DECODE_TELEMETRY = '1'
            $env:LLAMA_EXPERT_MISS_HISTOGRAM = '1'
            $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
            $env:GGML_CUDA_GPU_TIMESTAMPS_DEVICE = 'CUDA0'
        }
        $prefix = Join-Path $runDir $case
        $envSettings = [ordered]@{}
        foreach ($key in $keys) {
            if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
        }
        [ordered]@{ Case=$case; Tokens=$Tokens; Environment=$envSettings } |
            ConvertTo-Json -Depth 3 | Set-Content "$prefix.settings.json" -Encoding UTF8
        $runArgs = @('-lv', '4', '--log-file', "$prefix.log") + $baseArgs
        $runArgs | ConvertTo-Json | Set-Content "$prefix.arguments.json" -Encoding UTF8
        & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
        $code = $LASTEXITCODE
        "exit_code=$code" | Set-Content "$prefix.exit.txt"
        if ($code -ne 0) { throw "Inference exited with code $code. Logs: $runDir" }
        foreach ($pattern in @('packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1',
                'packed expert dynamic residency: promotions=[1-9]', 'final graph fence reused',
                'packed expert Vulkan upload overlap: submitted=[1-9]', 'packed expert Vulkan promotion overlap: committed=[1-9]')) {
            if (!(Select-String -LiteralPath "$prefix.log" -Pattern $pattern -Quiet)) {
                throw "Missing adopted setting: $pattern. Logs: $runDir"
            }
        }
        if (Select-String -LiteralPath "$prefix.log" -Pattern 'packed expert RTX offload:|packed expert CUDA sidecar:' -Quiet) {
            throw "Unexpected RTX Expert assist. Logs: $runDir"
        }
        Select-String -LiteralPath "$prefix.log" -Pattern '[|] +eval time =', 'draft acceptance =' | ForEach-Object Line
    }
    if ($Mode -ne 'speed') {
        $rows = @(Select-String -LiteralPath (Join-Path $runDir 'trace.log') -SimpleMatch 'expert_handoff: ' |
            ForEach-Object { $_.Line.Substring($_.Line.IndexOf('expert_handoff: ') + 16) })
        $rows | Set-Content (Join-Path $runDir 'handoff-raw.txt')
        $ordinals = @{}
        $misses = @{}
        $shared = @{}
        $transfers = @(
            foreach ($row in $rows) {
                if ($row -match '^graph=(\d+) ordinal=(\d+) ') {
                    $ordinals[$Matches[1]] = [int]$Matches[2]
                } elseif ($row -match '^graph=(\d+) split=(\d+) layer=(\d+) tokens=(\d+) unique_misses=(\d+) uploaded_bytes=(\d+)$') {
                    $miss = [pscustomobject]@{ Graph=$Matches[1]; Split=$Matches[2]; Layer=[int]$Matches[3]; Tokens=[int]$Matches[4]; UniqueMisses=[int]$Matches[5]; ExpertBytes=[long]$Matches[6] }
                    $misses["$($miss.Graph):$($miss.Split)"] = $miss
                } elseif ($row -match '^graph=(\d+) split=(\d+) backend=CUDA\S* role=shared submit_start_us=(\d+) submit_end_us=(\d+) node="([^"]*)" weight="([^"]*)"$') {
                    $graph = $Matches[1]
                    $start = [long]$Matches[3]
                    $mark = $Matches[5] + ' ' + $Matches[6]
                    if ($mark -match '(?:ffn_shexp[\w]*|shared_expert_gate[\w]*)-(\d+)\b|blk\.(\d+)\.ffn_') {
                        $layer = if ($Matches[1]) { [int]$Matches[1] } else { [int]$Matches[2] }
                        $key = "$($graph):$layer"
                        if (!$shared.ContainsKey($key) -or $start -lt $shared[$key]) { $shared[$key] = $start }
                    }
                } elseif ($row -match '^graph=(\d+) split=(\d+) transfer=(\S+) path=(\S+) tensors=(\d+) bytes=(\d+) source_wait_us=(\d+) destination_wait_us=(\d+) read_start_us=(\d+) read_end_us=(\d+) write_us=(\d+) names="([^"]*)" producer_splits="([^"]*)"$') {
                    $readStart = [long]$Matches[9]
                    $readEnd = [long]$Matches[10]
                    [pscustomobject]@{
                        Graph=$Matches[1]; Split=$Matches[2]; Direction=$Matches[3]; Path=$Matches[4]; Tensors=[int]$Matches[5]; Bytes=[long]$Matches[6]
                        SourceWaitUs=[long]$Matches[7]; DestinationWaitUs=[long]$Matches[8]
                        ReadStartUs=$readStart; ReadEndUs=$readEnd; HostReadUs=($readEnd-$readStart); HostWriteUs=[long]$Matches[11]
                        Names=$Matches[12]; ProducerSplits=$Matches[13]
                    }
                }
            }
        )
        if (!$misses.Count -or !$transfers.Count -or !$shared.Count) { throw "Missing handoff observations. Logs: $runDir" }
        $transfers | Export-Csv (Join-Path $runDir 'handoff-transfers.csv') -NoTypeInformation -Encoding UTF8
        $layers = @(
            foreach ($miss in $misses.Values | Sort-Object Graph, Layer) {
                $forward = @($transfers | Where-Object { $_.Graph -eq $miss.Graph -and $_.Split -eq $miss.Split -and $_.Direction -match '^CUDA.*->Vulkan' })
                $return = @($transfers | Where-Object {
                    $_.Graph -eq $miss.Graph -and $_.Direction -match '^Vulkan.*->CUDA' -and
                    (($_.ProducerSplits -split ',') -contains $miss.Split -or $_.Names -match "ffn_moe_(?:down|out)[\w]*-$($miss.Layer)\b")
                })
                $knownReturns = @($return | Where-Object ReadEndUs -gt 0)
                $readEnd = ($knownReturns | Measure-Object ReadEndUs -Minimum).Minimum
                $sharedStart = $shared["$($miss.Graph):$($miss.Layer)"]
                $order = if (!$sharedStart -or !$readEnd) { 'unknown' } elseif ($sharedStart -lt $readEnd) { 'shared_enqueue_before_read_return' } else { 'read_return_before_or_at_shared_enqueue' }
                [pscustomobject]@{
                    Graph=$miss.Graph; Ordinal=$ordinals[$miss.Graph]; Layer=$miss.Layer; Tokens=$miss.Tokens; UniqueMisses=$miss.UniqueMisses
                    ExpertBytes=$miss.ExpertBytes; IntoVulkanTransfers=$forward.Count; ReturnTransfers=$return.Count
                    IntoVulkanSourceWaitUs=($forward | Measure-Object SourceWaitUs -Sum).Sum
                    IntoVulkanDestinationWaitUs=($forward | Measure-Object DestinationWaitUs -Sum).Sum
                    IntoVulkanHostReadUs=($forward | Measure-Object HostReadUs -Sum).Sum
                    IntoVulkanHostWriteUs=($forward | Measure-Object HostWriteUs -Sum).Sum
                    EarliestReturnReadEndUs=$readEnd; EarliestSharedEnqueueStartUs=$sharedStart; HostOrder=$order
                }
            }
        )
        $layers | Export-Csv (Join-Path $runDir 'handoff-by-layer.csv') -NoTypeInformation -Encoding UTF8
        $layers | Group-Object HostOrder | Select-Object Name, Count | Format-Table -AutoSize
    }
} finally {
    foreach ($key in $keys) { if ($null -eq $saved[$key]) { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') } else { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') } }
    Write-Host "Log directory: $runDir"
}
