param([int]$Tokens = 256)
$ErrorActionPreference = 'Continue'
$root = 'G:\AI\llama-hetero-moe'
$cli = Join-Path $root 'build-win-cuda-vulkan\bin\llama-cli.exe'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$placement = 'C:\Users\kamui\.codex\visualizations\2026\11\12\01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f\flash-next-top300.txt'
foreach ($path in @($cli, $model, $placement)) {
    if (!(Test-Path -LiteralPath $path)) { throw "Missing: $path" }
}
$runDir = Join-Path $root ('fork-docs\research\logs\pipeline-ab-' + (Get-Date -Format yyyyMMdd-HHmmss))
New-Item -ItemType Directory -Path $runDir -Force -ErrorAction Stop | Out-Null
$keys = @('PATH', 'LLAMA_EXPERT_PLACEMENT', 'LLAMA_EXPERT_DEVICE', 'LLAMA_EXPERT_SCRATCH_REUSE',
    'LLAMA_EXPERT_BATCH_UPLOAD', 'LLAMA_EXPERT_OVERLAP', 'LLAMA_EXPERT_OVERLAP_EARLY_COPY',
    'LLAMA_DECODE_PROFILE', 'LLAMA_DECODE_PROFILE_CUDA', 'GGML_CUDA_DISABLE_GRAPHS',
    'GGML_CUDA_GRAPH_DEBUG', 'GGML_CUDA_POOL_DEBUG', 'GGML_VK_SERIALIZE_SUBMISSIONS',
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE', 'GGML_SCHED_BATCH_HOST_INPUTS', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_DISABLE_FUSION')
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
try {
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
    }
    $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH"
    $env:LLAMA_EXPERT_PLACEMENT = $placement
    $env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
    $env:LLAMA_EXPERT_SCRATCH_REUSE = '1'
    $env:LLAMA_EXPERT_BATCH_UPLOAD = '1'
    $baseArgs = @('-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on', '--spec-type', 'none',
        '-c', '2048', '-b', '256', '-ub', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234',
        '-n', "$Tokens", '--single-turn', '-p', 'Explain how a bicycle works in detail.', '-lv', '4')
    $cases = @(
        @{ Name='baseline';       Overlap=$false; Early='0'; HostBatch='0'; HC='0' },
        @{ Name='overlap-waited'; Overlap=$true;  Early='0'; HostBatch='0'; HC='0' },
        @{ Name='overlap-early';  Overlap=$true;  Early='1'; HostBatch='0'; HC='0' },
        @{ Name='host-batch';     Overlap=$false; Early='0'; HostBatch='1'; HC='0' },
        @{ Name='hc-fusion';      Overlap=$false; Early='0'; HostBatch='0'; HC='1' }
    )
    $baseArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    git -C $root rev-parse HEAD | Set-Content (Join-Path $runDir 'head.txt')
    git -C $root diff --stat | Set-Content (Join-Path $runDir 'working-diff-stat.txt')
    foreach ($case in $cases) {
        if ($case.Overlap) {
            $env:LLAMA_EXPERT_OVERLAP = '1'
            $env:GGML_VK_ASYNC_USE_TRANSFER_QUEUE = '1'
        } else {
            Remove-Item Env:LLAMA_EXPERT_OVERLAP -ErrorAction SilentlyContinue
            Remove-Item Env:GGML_VK_ASYNC_USE_TRANSFER_QUEUE -ErrorAction SilentlyContinue
        }
        $env:LLAMA_EXPERT_OVERLAP_EARLY_COPY = $case.Early
        $env:GGML_SCHED_BATCH_HOST_INPUTS = $case.HostBatch
        $env:GGML_CUDA_HC_MIX_FUSE = $case.HC
        $prefix = Join-Path $runDir $case.Name
        $case | ConvertTo-Json | Set-Content "$prefix.settings.json" -Encoding UTF8
        $runArgs = $baseArgs + @('--log-file', "$prefix.log")
        Write-Host "Running: $($case.Name)"
        & $cli @runArgs 1> "$prefix.stdout.txt" 2> "$prefix.stderr.txt"
        $code = $LASTEXITCODE
        "exit_code=$code" | Set-Content "$prefix.exit.txt"
        if ($code -ne 0) { throw "Failed: $($case.Name). Logs: $runDir" }
    }
    Write-Host "Completed. Logs: $runDir"
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Write-Host "Log directory: $runDir"
}
