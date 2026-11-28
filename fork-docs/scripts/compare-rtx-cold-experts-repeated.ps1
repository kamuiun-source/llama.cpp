[CmdletBinding()]
param(
    [string]$BuildDirectory = 'G:\AI\build-rtx-validation',
    [ValidateSet(2, 4, 6, 8)]
    [int]$Rounds = 4,
    [ValidateRange(1, 19)]
    [int]$RtxSlots = 1
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'compare-rtx-cold-experts.ps1'
$runDir = Join-Path 'G:\AI\llama-hetero-moe-local\logs' ('rtx-cold-repeated-' + (Get-Date -Format yyyyMMdd-HHmmss))
New-Item -ItemType Directory -Path $runDir | Out-Null
$culture = [Globalization.CultureInfo]::InvariantCulture
$rows = @()
@{
    Rounds = $Rounds
    WarmupRunsPerConfiguration = 1
    Tokens = 256
    RtxSlots = $RtxSlots
    BuildDirectory = $BuildDirectory
    ProcessPolicy = 'Fresh process per run; warmup primes system caches, not resident model or Expert cache.'
} | ConvertTo-Json | Set-Content (Join-Path $runDir 'settings.json') -Encoding UTF8

try {
    for ($round = 0; $round -le $Rounds; $round++) {
        $label = if ($round -eq 0) { 'warmup' } else { 'round-{0:D2}' -f $round }
        $directory = Join-Path $runDir $label
        $argsList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner,
            '-BuildDirectory', $BuildDirectory, '-Mode', 'speed', '-RtxSlots', "$RtxSlots",
            '-LogDirectory', $directory)
        if ($round -gt 0 -and $round % 2 -eq 1) { $argsList += '-ReverseOrder' }
        Write-Host "Running $label"
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & powershell.exe @argsList *> (Join-Path $runDir "$label.runner.log")
            $childExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($childExitCode -ne 0) { throw "Comparison failed: $label (exit $childExitCode). See $runDir" }
        if ($round -eq 0) { continue }

        $order = @(Get-Content (Join-Path $directory 'run-order.txt'))
        foreach ($case in @('baseline', 'rtx-cold')) {
            $log = Get-Content (Join-Path $directory "$case.log") -Raw
            $timings = [regex]::Matches($log, '(?m)\|\s+eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens.*?([0-9.]+) tokens per second')
            $drafts = [regex]::Matches($log, 'draft acceptance = ([0-9.]+)')
            if (!$timings.Count -or !$drafts.Count) { throw "Missing metrics: $label/$case" }
            $t = $timings[$timings.Count - 1].Groups
            $rows += [pscustomobject]@{
                Round = $round
                Configuration = $case
                Position = [array]::IndexOf($order, $case) + 1
                DecodeMs = [double]::Parse($t[1].Value, $culture)
                Tokens = [int]$t[2].Value
                TokensPerSecond = [double]::Parse($t[3].Value, $culture)
                MtpAcceptance = [double]::Parse($drafts[$drafts.Count - 1].Groups[1].Value, $culture)
                LogDirectory = $directory
            }
        }
        $rows | Export-Csv (Join-Path $runDir 'measurements.csv') -NoTypeInformation -Encoding UTF8
    }

    $summary = foreach ($case in @('baseline', 'rtx-cold')) {
        $values = @($rows | Where-Object Configuration -eq $case | ForEach-Object TokensPerSecond | Sort-Object)
        $times = @($rows | Where-Object Configuration -eq $case | ForEach-Object DecodeMs | Sort-Object)
        [pscustomobject]@{
            Configuration = $case
            Runs = $values.Count
            MeanTokensPerSecond = ($values | Measure-Object -Average).Average
            MedianTokensPerSecond = ($values[$values.Count / 2 - 1] + $values[$values.Count / 2]) / 2
            MinTokensPerSecond = $values[0]
            MaxTokensPerSecond = $values[-1]
            MedianDecodeMs = ($times[$times.Count / 2 - 1] + $times[$times.Count / 2]) / 2
        }
    }
    $summary | Export-Csv (Join-Path $runDir 'summary.csv') -NoTypeInformation -Encoding UTF8
    $summary | Format-Table -AutoSize
} finally {
    Write-Host "Log directory: $runDir"
}
