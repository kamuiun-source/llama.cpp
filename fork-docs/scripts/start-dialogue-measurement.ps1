[CmdletBinding()]
param(
    [ValidateSet('speed', 'telemetry', 'components')]
    [string]$Mode = 'telemetry',
    [switch]$HcScaleSilu,
    [switch]$HcUpPair,
    [switch]$HcDownReuse,
    [switch]$HcUpReuse,
    [switch]$HcInjectSplit,
    [switch]$HcAll,
    [switch]$ExpertRamCopyTiming,
    [string]$BuildDirectory = 'G:\AI\build-prefill32'
)

$ErrorActionPreference = 'Stop'
$projections = @(
    @{ Name='down'; Enabled=($HcAll -or $HcDownReuse); Key='GGML_CUDA_HC_DOWN_REUSE'; Marker='CUDA HC down reuse applied' }
    @{ Name='up'; Enabled=($HcAll -or $HcUpReuse); Key='GGML_CUDA_HC_UP_REUSE'; Marker='CUDA HC up reuse applied' }
    @{ Name='inject'; Enabled=($HcAll -or $HcInjectSplit); Key='GGML_CUDA_HC_INJECT_SPLIT'; Marker='CUDA HC inject split applied' }
)
$activeProjections = @($projections | Where-Object { $_.Enabled })
if (($HcUpPair -and $HcScaleSilu) -or (($HcUpPair -or $HcScaleSilu) -and $activeProjections.Count)) {
    throw 'Run the legacy HC experiment separately from the new projection experiments.'
}
$cli = Join-Path $BuildDirectory 'bin\llama-cli.exe'
$settingsPath = 'G:\AI\llama-hetero-moe-local\logs\rtx-cold-repeated-20261117-172614\round-01\baseline.settings.json'
$model = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf'
$draft = 'G:\AI\share\llm_models\Qwen3.8-Flash-Next-GGUF\MTP\mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
foreach ($path in @($cli, $settingsPath, $model, $draft)) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$retiredKeys = @(
    'LLAMA_EXPERT_SCRATCH_REUSE', 'LLAMA_EXPERT_DYNAMIC_RESIDENT', 'LLAMA_EXPERT_BATCH_UPLOAD',
    'LLAMA_EXPERT_ROUTER_HOST_IDS', 'GGML_CUDA_SCHEDULE_SPIN', 'LLAMA_EXPERT_VK_DEFER_SLOT_MAP',
    'GGML_SCHED_BATCH_DEVICE_INPUTS', 'GGML_SCHED_OVERLAP_SHARED_EXPERT', 'GGML_VK_REUSE_GRAPH_FENCE',
    'LLAMA_EXPERT_VK_OVERLAP_UPLOAD', 'LLAMA_EXPERT_VK_OVERLAP_PROMOTION', 'LLAMA_EXPERT_VK_FULL_HOT_OVERLAP',
    'LLAMA_EXPERT_RAM_ROTATION', 'GGML_SCHED_SHARED_HOST_EXPERT_INPUT', 'GGML_SCHED_SHARED_HOST_EXPERT_RETURN',
    'LLAMA_MTP_DEVICE_PENDING', 'GGML_CUDA_SSM_CONV_HISTORY', 'GGML_SCHED_OVERLAP_SHARED_READBACK'
)
$baselineEnvironment = @($settings.Environment.PSObject.Properties | Where-Object { $_.Name -notin $retiredKeys })
$keys = @($baselineEnvironment.Name) + @('PATH', 'LLAMA_DECODE_TELEMETRY_PER_RESPONSE', 'GGML_CUDA_HC_MIX_FUSE', 'GGML_CUDA_HC_SCALE_SILU_FUSE', 'GGML_CUDA_HC_UP_PAIR', 'LLAMA_EXPERT_RTX_COLD_SLOTS', 'LLAMA_EXPERT_PROFILE_RAM_COPY') + @($projections | ForEach-Object { $_.Key })
$keys = @($keys | Select-Object -Unique)
$saved = @{}
foreach ($key in $keys) { $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
$projectionLabel = if ($activeProjections.Count -eq $projections.Count) { 'hc-all-' }
    elseif ($activeProjections.Count) { 'hc-' + (($activeProjections | ForEach-Object { $_.Name }) -join '-') + '-' }
    else { '' }
$ramLabel = if ($ExpertRamCopyTiming) { 'ram-timing-' } else { '' }
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' ("dialogue-$Mode-$projectionLabel$ramLabel" + (Get-Date -Format yyyyMMdd-HHmmss))
New-Item -ItemType Directory -Path $runDir | Out-Null

try {
    foreach ($item in $baselineEnvironment) {
        [Environment]::SetEnvironmentVariable($item.Name, $item.Value, 'Process')
    }
    if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin\x64;$env:PATH" }
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_RTX_COLD_SLOTS', $null, 'Process')
    [Environment]::SetEnvironmentVariable('LLAMA_EXPERT_PROFILE_RAM_COPY', $(if ($ExpertRamCopyTiming) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable('LLAMA_DECODE_TELEMETRY_PER_RESPONSE', $null, 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_MIX_FUSE', $null, 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_SCALE_SILU_FUSE', $(if ($HcScaleSilu) { '1' } else { $null }), 'Process')
    [Environment]::SetEnvironmentVariable('GGML_CUDA_HC_UP_PAIR', $(if ($HcUpPair) { '1' } else { $null }), 'Process')
    foreach ($projection in $projections) {
        [Environment]::SetEnvironmentVariable($projection.Key, $(if ($projection.Enabled) { '1' } else { $null }), 'Process')
    }
    if ($Mode -ne 'speed') {
        $env:LLAMA_DECODE_TELEMETRY = '1'
        $env:LLAMA_DECODE_TELEMETRY_PER_RESPONSE = '1'
        $env:LLAMA_EXPERT_MISS_HISTOGRAM = '1'
        $env:GGML_CUDA_GPU_TIMESTAMPS_DEVICE = 'CUDA0'
        $env:GGML_VK_GPU_TIMESTAMPS_DEVICE = 'Vulkan1'
    }
    if ($Mode -eq 'components') {
        $env:GGML_CUDA_DISABLE_GRAPHS = '1'
        $env:GGML_CUDA_PROFILE_COMPONENTS = '1'
    }
    $effective = [ordered]@{}
    foreach ($key in $keys) {
        if ($key -ne 'PATH') { $effective[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
    }
    @{ Mode=$Mode; Cli=$cli; Environment=$effective; Instrumented=($Mode -ne 'speed' -or $ExpertRamCopyTiming) } |
        ConvertTo-Json -Depth 4 | Set-Content (Join-Path $runDir 'settings.json') -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\research\dialogue-benchmark-prompts.md') -Destination $runDir
    $runArgs = @('-lv', '4', '--log-file', (Join-Path $runDir 'dialogue.log'), '--simple-io', '--color', 'off',
        '-m', $model, '-dev', 'CUDA0,Vulkan1', '-sm', 'layer', '-ts', '1,0', '-ngl', 'all',
        '--fit', 'off', '--load-mode', 'none', '--lazy-mode', 'on',
        '-c', '2048', '-b', '256', '-ub', '2', '-np', '1', '--reasoning', 'off', '--temp', '0', '--seed', '1234', '-n', '256',
        '--spec-type', 'draft-mtp', '-md', $draft, '-devd', 'CUDA0', '-ngld', 'all',
        '--spec-draft-n-max', '1', '--spec-draft-n-ubatch', '1')
    $runArgs | ConvertTo-Json | Set-Content (Join-Path $runDir 'arguments.json') -Encoding UTF8
    Write-Host "Mode: $Mode. Send the five saved prompts in order, then /exit."
    try {
        $ErrorActionPreference = 'Continue'
        & $cli @runArgs 2> (Join-Path $runDir 'stderr.log')
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
    }
    "exit_code=$code" | Set-Content (Join-Path $runDir 'exit.txt')
    if ($code -ne 0) { throw "llama-cli failed: $code" }
    $dialogueLog = Join-Path $runDir 'dialogue.log'
    if (!(Select-String -LiteralPath $dialogueLog -SimpleMatch 'CUDA SSM convolution history applied' -Quiet)) {
        Write-Warning 'CUDA SSM convolution history was not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'MTP device pending: enabled=1 copies=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'MTP device pending copies were not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'shared host expert input: batches=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'Shared host Expert input batches were not observed.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'shared host expert return: copies=[1-9][0-9]*' -Quiet)) {
        Write-Warning 'Shared host Expert return copies were not observed.'
    }
    if ($ExpertRamCopyTiming -and !(Select-String -LiteralPath $dialogueLog -SimpleMatch 'packed expert promotion RAM memcpy:' -Quiet)) {
        throw 'Expert RAM copy timing did not activate. Check the binary and dynamic Expert placement.'
    }
    if (!(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert RAM rotation: rotations=[1-9][0-9]*' -Quiet)) {
        if (!(Select-String -LiteralPath $dialogueLog -Pattern 'packed expert RAM rotation: .*pool_bytes=[1-9][0-9]*' -Quiet)) {
            Write-Warning 'Expert RAM rotation pool allocation was not observed.'
        } else {
            Write-Warning 'The Expert RAM rotation pool was allocated, but this run performed no rotations.'
        }
    }
    foreach ($projection in $activeProjections) {
        if (!(Select-String -LiteralPath $dialogueLog -SimpleMatch $projection.Marker -Quiet)) {
            throw "HC $($projection.Name) did not activate. Check the binary, tensor type, shape and layout."
        }
    }
    if ($Mode -ne 'speed') {
        & (Join-Path $PSScriptRoot 'summarize-dialogue-telemetry.ps1') -LogDirectory $runDir
    }
    if ($HcUpPair -and !(Select-String -LiteralPath (Join-Path $runDir 'dialogue.log') -SimpleMatch 'CUDA HC up pair applied' -Quiet)) {
        throw 'HC up pair did not activate. Check the binary and graph layout.'
    }
    if ($HcScaleSilu -and !(Select-String -LiteralPath (Join-Path $runDir 'dialogue.log') -SimpleMatch 'CUDA HC scale SiLU fusion applied' -Quiet)) {
        throw 'HC scale SiLU fusion did not activate. Check the binary and graph layout.'
    }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    Write-Host "Log directory: $runDir"
}
