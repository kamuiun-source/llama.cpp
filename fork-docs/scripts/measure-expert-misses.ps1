[CmdletBinding()]
param(
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
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' "packed-expert-misses-$stamp"
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
    $env:LLAMA_DECODE_TELEMETRY = '1'
    $env:LLAMA_EXPERT_MISS_HISTOGRAM = '1'

    $env:GGML_VK_REUSE_GRAPH_FENCE = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_UPLOAD = '1'
    $env:LLAMA_EXPERT_VK_OVERLAP_PROMOTION = '1'

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

    $prefix = Join-Path $runDir 'adopted'
    $envSettings = [ordered]@{}
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { $envSettings[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
    }
    [ordered]@{ Tokens=$Tokens; Environment=$envSettings } |
        ConvertTo-Json -Depth 3 | Set-Content "$prefix.settings.json" -Encoding UTF8
    $runArgs = @('-lv', '4', '--log-file', "$prefix.log") + $baseArgs
    $runArgs | ConvertTo-Json | Set-Content "$prefix.arguments.json" -Encoding UTF8
    & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
    $code = $LASTEXITCODE
    "exit_code=$code" | Set-Content "$prefix.exit.txt"
    if ($code -ne 0) { throw "Inference exited with code $code. Logs: $runDir" }
    foreach ($pattern in @(
            'packed expert batch capacity = 2, n_ubatch = 2, n_rs_seq = 1',
            'CUDA device 0 scheduling: request=spin effective_flags=0x[0-9a-f]+ effective_spin=1',
            'packed expert router: host_hits=[1-9][0-9]*',
            'sched_batch_device_inputs: device_input_batches=[1-9][0-9]*',
            'packed expert slot map updates: deferred=[1-9][0-9]*',
            'final graph fence reused',
            'packed expert Vulkan upload overlap: submitted=[1-9][0-9]*',
            'packed expert Vulkan promotion overlap: committed=[1-9][0-9]*')) {
        if (@(Select-String -LiteralPath "$prefix.log" -Pattern $pattern).Count -eq 0) {
            throw "Missing adopted setting: $pattern. Logs: $runDir"
        }
    }
    $rows = @(Select-String -LiteralPath "$prefix.log" -Pattern 'decode_telemetry: phase=\w+ expert_miss_histogram steps=[1-9][0-9]*' |
        ForEach-Object Line)
    $rows | Set-Content (Join-Path $runDir 'expert-miss-histogram.txt')
    if (!($rows -match 'phase=target_verify expert_miss_histogram steps=[1-9][0-9]*')) {
        throw "No packed Expert miss histogram recorded for target verification. Logs: $runDir"
    }
    $rows
} finally {
    foreach ($key in $keys) { if ($null -eq $saved[$key]) { [Environment]::SetEnvironmentVariable($key, [NullString]::Value, 'Process') } else { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') } }
    Write-Host "Log directory: $runDir"
}
