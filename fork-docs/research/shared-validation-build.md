# Shared validation build

Development worktrees stay separate. The fixed validation source is `C:/Users/kamui/.codex/worktrees/rtx-expert-validation/llama-hetero-moe`; the reusable build directory is `G:/AI/build-rtx-validation`. Baseline branch: `codex/hetero-validation-base`. Experiment branches: `codex/recurrent-tuning` and `codex/transfer-tail`.

Run the following in Developer PowerShell with no build/inference running. Detached checkout permits the same experiment branch to stay checked out in its development worktree. The source directory is unchanged, so existing build artifacts can be reused; changed CUDA files may still take time. Reconfiguration ensures CMake tracks the selected source. The final hash identifies the exact source used for this build.

```powershell
Set-Location 'C:\Users\kamui\.codex\worktrees\rtx-expert-validation\llama-hetero-moe'
$revision = 'codex/hetero-validation-base' # replace with the reviewed experiment branch
if (git status --porcelain) { throw 'Validation checkout has local changes. Preserve them before switching.' }
git switch --detach $revision
if ($LASTEXITCODE -ne 0) { throw 'Checkout failed' }
$validatedHead = git rev-parse HEAD
cmake -S . -B 'G:\AI\build-rtx-validation'
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
cmake --build 'G:\AI\build-rtx-validation' --target llama-cli llama-server --parallel 6
if ($LASTEXITCODE -ne 0) { throw 'Build failed: do not run the previous executable' }
Write-Host "Built revision: $validatedHead"
```

The current experiments are not automatically adopted. Run their reviewed comparison commands only after their implementation and review are complete. Switching commits does not change environment flags, so use the supplied launcher to reset experiments. Logs remain in `G:/AI/llama-hetero-moe-local/logs`, outside Git. Do not point this build directory at another worktree. Do not use `git reset --hard` or `git clean` to prepare validation.
