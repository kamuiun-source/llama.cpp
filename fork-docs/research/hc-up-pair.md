# HC up projection with two tokens

`GGML_CUDA_HC_UP_PAIR=1` enables an opt-in MMVQ launch on NVIDIA Ada for contiguous, non-fused Q8_0 HC attention and FFN up weights with K=320, 10240 output rows, and two tokens. The launch uses one warp per block instead of four, with two output rows per block in both cases. It reuses the existing Q8_1 input quantization and dot product. The change in reduction order can change F32 rounding. Other shapes and paths keep their usual launch.

The standalone RTX 4070 Ti fixture executes each small graph three times. The one-token HC graph and the two-token non-HC graph produce bitwise-identical output with the switch OFF and ON and never activate the experiment. The two-token HC graph activates only with the switch ON: all 20,480 outputs are finite, 11,859 differ numerically, maximum absolute difference is 6.52e-9, and relative L2 difference is 1.15e-7. This checks the isolated projection and does not establish full-model output equivalence or throughput.

For a normal-path comparison, run the following in a developer shell after configuring the existing build directory for this worktree. Send the same five prompts from [dialogue-benchmark-prompts.md](dialogue-benchmark-prompts.md) in the same order for each run, then enter `/exit`.

```powershell
cmake --build G:/AI/build-rtx-validation --config Release --target llama-cli ggml-cuda
& ./fork-docs/scripts/start-dialogue-measurement.ps1 -Mode speed
& ./fork-docs/scripts/start-dialogue-measurement.ps1 -Mode speed -HcUpPair
```

Both runs leave CUDA Graphs on and keep the HC fusions and RTX cold Expert sharing off. The ON run checks for `CUDA HC up pair applied` in the log. Compare generation, MTP acceptance, and decode throughput before deciding whether to adopt the change. Logs remain outside Git.

## First full-model comparison (2026-11-17)

ON log: `dialogue-speed-20261117-201640`, compared at the user's request against existing OFF `dialogue-speed-20261117-194511` (no new OFF run). Activation and exit 0 confirmed. HC fusions and RTX sharing remain OFF.

OFF: 46.58024 s decode, 1216 timed tokens, 26.1055 tok/s. ON: 51.64441 s, 1196 timed tokens, 23.1584 tok/s. This is descriptive, not an isolated regression measurement: generated counts differ (OFF 256/251/256/202/256; ON 252/256/235/202/256), draft acceptance differs, and response 3 prompt evaluation is 327 tokens versus 68. Prompt evaluation count is not proof the user entered a different prompt; cache/history reuse also matters. Response 4 is 16.81 versus 26.07 tok/s. No context truncation appears in the ON release lines.

No evidence of a throughput improvement. Recommend retaining default OFF and postponing adoption. Reduction rounding can change generation, but the full-model difference has not been localized to a numerical bug versus expected rounding. If revisited, use a fixed-input isolated projection benchmark or a non-conversational identical context rather than repeating long history-dependent dialogues blindly.
