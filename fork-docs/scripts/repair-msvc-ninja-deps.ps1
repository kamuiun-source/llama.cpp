[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $BuildDirectory,
    [switch] $Repair,
    [switch] $Build,
    [ValidateRange(1, 64)]
    [int] $Jobs = 4
)

$ErrorActionPreference = 'Stop'
$buildPath = (Resolve-Path -LiteralPath $BuildDirectory).Path
$cache = Get-Content -LiteralPath (Join-Path $buildPath 'CMakeCache.txt') -Raw

function Get-CachePath([string] $Key) {
    $match = [regex]::Match($cache, "(?m)^${Key}:FILEPATH=([^`r`n]+)")
    if (-not $match.Success) { throw "Missing $Key in CMakeCache.txt" }
    return $match.Groups[1].Value
}

$ninja = Get-CachePath 'CMAKE_MAKE_PROGRAM'
$compiler = Get-CachePath 'CMAKE_CXX_COMPILER'
if ([IO.Path]::GetFileName($compiler) -ine 'cl.exe' -or [IO.Path]::GetFileName($ninja) -ine 'ninja.exe') {
    throw 'The build must use MSVC cl.exe with Ninja.'
}

# Preserve the exact bytes, including the trailing spaces in the include prefix.
$latin1 = [Text.Encoding]::GetEncoding(28591)
$prefix = [regex]::Match($latin1.GetString([IO.File]::ReadAllBytes((Join-Path $buildPath 'CMakeFiles/rules.ninja'))),
    '(?m)^msvc_deps_prefix = ([^\r\n]+)')
if (-not $prefix.Success) { throw 'Missing msvc_deps_prefix in CMakeFiles/rules.ninja.' }

function Set-CodePage([int] $Page) {
    & $env:ComSpec /d /c "chcp $Page >nul"
    return $LASTEXITCODE -eq 0
}

function Test-DependencyTracking([string] $ProbePath, [int] $Page) {
    [IO.Directory]::CreateDirectory($ProbePath) | Out-Null
    [IO.File]::WriteAllText((Join-Path $ProbePath 'probe.h'), '#define PROBE_VALUE 1')
    [IO.File]::WriteAllText((Join-Path $ProbePath 'probe.cpp'), "#include `"probe.h`"`nint probe() { return PROBE_VALUE; }`n")
    $rule = "rule msvc_probe`n  command = `"$compiler`" /nologo /showIncludes /utf-8 /c probe.cpp /Foprobe.obj`n  deps = msvc`n  msvc_deps_prefix = "
    $bytes = [Text.Encoding]::UTF8.GetBytes($rule) + $latin1.GetBytes($prefix.Groups[1].Value) +
        [Text.Encoding]::UTF8.GetBytes("`nbuild probe.obj: msvc_probe probe.cpp`n")
    [IO.File]::WriteAllBytes((Join-Path $ProbePath 'build.ninja'), $bytes)
    Push-Location -LiteralPath $ProbePath
    try {
        $compileOutput = & $env:ComSpec /d /c ('chcp {0} >nul && "{1}"' -f $Page, $ninja) 2>&1
        if ($LASTEXITCODE -ne 0) { throw "MSVC include probe failed:`n$($compileOutput -join "`n")" }
        $deps = & $ninja -t deps probe.obj
        return $LASTEXITCODE -eq 0 -and [bool] ($deps -match '^\s+(?:.*[\\/])?probe\.h\s*$')
    } finally {
        Pop-Location
    }
}

$cpResult = [regex]::Match((& $env:ComSpec /d /c chcp | Out-String), '(\d+)\s*$')
if (-not $cpResult.Success) { throw 'Could not read the current Windows console code page.' }
$originalPage = [int] $cpResult.Groups[1].Value
$probeRoot = Join-Path ([IO.Path]::GetTempPath()) ('llama-msvc-deps-' + [guid]::NewGuid().ToString('N'))
$workingPage = $null
try {
    foreach ($page in (@($originalPage, 932, 65001) | Select-Object -Unique)) {
        if (Test-DependencyTracking (Join-Path $probeRoot $page) $page) {
            $workingPage = $page
            break
        }
    }
    if ($null -eq $workingPage) {
        throw 'Ninja did not record the test header with the current code page, 932, or 65001. The build was left alone.'
    }
    Write-Host "Verified Ninja recorded the MSVC include at code page $workingPage."

    if ($Build) {
        $llamaObjects = 'src/CMakeFiles/llama.dir'
        $unity = @(Get-ChildItem -LiteralPath (Join-Path $buildPath "$llamaObjects/Unity") -Filter 'unity_*_cxx.cxx' -File |
            Where-Object { Select-String -LiteralPath $_.FullName -Pattern '[/\\]models[/\\]qwen4exp\.cpp' -Quiet })
        if ($unity.Count -ne 1) { throw 'Could not locate the Qwen4Exp Unity compile target.' }
        $modelObject = "$llamaObjects/Unity/$($unity[0].Name).obj"
    }

    if ($Repair) {
        $depsPath = Join-Path $buildPath '.ninja_deps'
        if (Test-Path -LiteralPath $depsPath -PathType Leaf) {
            $backup = '.ninja_deps.before-msvc-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
            Rename-Item -LiteralPath $depsPath -NewName $backup
            Write-Host "Saved the old dependency database as $backup."
        }
    }

    if ($Build) {
        Push-Location -LiteralPath $buildPath
        try {
            & $env:ComSpec /d /c ('chcp {0} >nul && "{1}" "{2}"' -f $workingPage, $ninja, $modelObject)
            if ($LASTEXITCODE -ne 0) { throw "Qwen4Exp compile failed with exit code $LASTEXITCODE." }
            $deps = & $ninja -t deps $modelObject
            if ($LASTEXITCODE -ne 0 -or -not ($deps -match '^\s+(?:.*[/\\])?qwen4exp\.cpp\s*$') -or
                    -not ($deps -match '^\s+(?:.*[/\\])?llama-graph\.h\s*$')) {
                throw 'Ninja did not record qwen4exp.cpp and llama-graph.h in the Qwen4Exp Unity dependency record.'
            }
            Write-Host 'Verified Qwen4Exp Unity dependency tracking in the real build.'

            & $env:ComSpec /d /c ('chcp {0} >nul && "{1}" -j {2} llama-cli llama-server' -f $workingPage, $ninja, $Jobs)
            if ($LASTEXITCODE -ne 0) { throw "llama-cli/llama-server build failed with exit code $LASTEXITCODE." }
        } finally {
            Pop-Location
        }
    }
} finally {
    if (-not (Set-CodePage $originalPage)) { Write-Warning "Could not restore console code page $originalPage." }
    if (Test-Path -LiteralPath $probeRoot -PathType Container) {
        foreach ($page in (@($originalPage, 932, 65001) | Select-Object -Unique)) {
            $probePath = Join-Path $probeRoot $page
            if (Test-Path -LiteralPath $probePath -PathType Container) {
                foreach ($file in ('probe.h', 'probe.cpp', 'probe.obj', 'build.ninja', '.ninja_log', '.ninja_deps')) {
                    $path = Join-Path $probePath $file
                    if (Test-Path -LiteralPath $path -PathType Leaf) { Remove-Item -LiteralPath $path }
                }
                Remove-Item -LiteralPath $probePath
            }
        }
        Remove-Item -LiteralPath $probeRoot
    }
}
