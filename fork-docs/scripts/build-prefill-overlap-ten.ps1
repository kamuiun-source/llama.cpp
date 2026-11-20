param(
    [string]$BuildDirectory = 'G:/AI/build-prefill-overlap-ten',
    [int]$Jobs = 6
)

$ErrorActionPreference = 'Stop'
$source = 'C:/Users/kamui/.codex/worktrees/prefill-overlap-ten/llama-hetero-moe'
$vs = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools'
$cmake = Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
$ninja = Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
$cache = Join-Path $BuildDirectory 'CMakeCache.txt'
if (Test-Path -LiteralPath $cache) {
    $sourceEntry = Get-Content -LiteralPath $cache | Where-Object { $_ -like 'CMAKE_HOME_DIRECTORY:INTERNAL=*' }
    if ($sourceEntry -ne "CMAKE_HOME_DIRECTORY:INTERNAL=$source") {
        throw 'Build directory belongs to another source tree; choose a fresh build directory.'
    }
}
Import-Module (Join-Path $vs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'
& $cmake -S $source -B $BuildDirectory -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
    '-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe' `
    -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_CUDA_GRAPHS=ON `
    -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed: $LASTEXITCODE" }
& $cmake --build $BuildDirectory --target llama-cli --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
