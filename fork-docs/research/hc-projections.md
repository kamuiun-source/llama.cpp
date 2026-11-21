# CUDA HC projection experiments

The four Flash-Next UD-IQ3_XXS shard headers contain 96 weights for each role below. The switches target separate, contiguous attention and FFN projections on NVIDIA Ada. Each switch is off by default.

| Projection | Weight K x rows | Tokens | Environment variable | Launch switch |
| --- | --- | ---: | --- | --- |
| Down | Q8_0 10240 x 320 | 2 | `GGML_CUDA_HC_DOWN_REUSE` | `-HcDownReuse` |
| Up | Q8_0 320 x 10240 | 2 | `GGML_CUDA_HC_UP_REUSE` | `-HcUpReuse` |
| Inject | F32 10240 x 4 | 1 or 2 | `GGML_CUDA_HC_INJECT_SPLIT` | `-HcInjectSplit` |

Down and up explicitly reuse one tile of Q8 weights and Q8_1 activations across two output rows and both tokens. Up also packs two row pairs into each CTA, which halves the number of CTAs. Each pair retains its four-warp reduction. Inject splits the long inner product across 32 independent warps per row and uses a second kernel to sum the partials. Two-token inject reuses the F32 weight within each thread. Its changed reduction order affects rounding.

The launcher accepts the individual switches and `-HcAll`. It records effective flags and checks each selected projection's first activation in the log. For these switches it disables the older HC experiments and RTX cold Expert sharing.

```powershell
cmake --build G:/AI/build-rtx-validation --config Release --target llama-cli ggml-cuda
& ./fork-docs/scripts/start-dialogue-measurement.ps1 -Mode speed -HcAll
```

Send the five saved prompts in order and end with `/exit`. [Dialogue benchmark prompts](dialogue-benchmark-prompts.md)

To collect dispatch counts and per-projection GPU intervals, use `-Mode components -HcAll` in a separate run. Components mode disables CUDA Graphs and synchronizes sampled dispatches. It adds `cuda_component:hc/op:MUL_MAT/projection:down`, `up`, and `inject` subgroups to the response CSV. These rows break down the existing HC matrix multiplication totals and must not be added to them. Dispatches of separate projections are counted even when they take the original launch. A launch marker confirms that the experiment ran at least once. Speed mode reports those markers without collecting the synchronized timing details. Logs stay outside Git.

## RTX 4070 Ti fixture

The CUDA translation units compiled and linked into a temporary DLL. The server translation unit passed MSVC syntax checking. A standalone graph tested each projection, the combined path with down feeding Scale/SiLU and up while sharing the original input with inject, and the unchanged launch for weights with non-HC names. CUDA Graphs captured each graph after warmup and replayed it with updated inputs.

The two-token down and up outputs matched bitwise both in isolation and in the combined graph. The two-token inject maximum absolute difference was 1.19e-6 in isolation and 7.15e-7 in the combined graph. Relative L2 differences were 3.78e-7 and 1.08e-7 respectively. The largest ON deviation from a double-precision CPU inject reference was 8.45e-7. Non-HC weights matched bitwise and emitted no markers. With one token, only inject activated and the maximum absolute difference was 4.77e-7. With components mode, the combined fixture reported eight separate dispatches and eight samples for each role.

GPU stream time was sampled over 128 Graph replays per run, with 16 copies of the projection or graph sequence per replay. The table shows the median of three run averages per mode on the same synthetic inputs. Values include the launches in the graph and use warm GPU data.

| Projection or sequence | OFF (us) | ON (us) |
| --- | ---: | ---: |
| Down | 5.335 | 5.252 |
| Up | 8.699 | 9.019 |
| Inject | 4.128 | 2.690 |
| Down, Scale/SiLU, up, inject | 18.112 | 17.693 |

The down measurements overlap and up did not improve in this fixture. Inject and the combined fixture were shorter at the sampled medians. Model output, MTP acceptance and end-to-end throughput await the full-model run.

## Full-model all-ON run and user decision (2026-11-17)

User chose to adopt all three switches after `dialogue-speed-hc-all-20261117-204235`. All markers present, exit 0, old HC fusions/up-pair and RTX sharing OFF. Operational preference for subsequent runs is `-HcAll`; low-level default-off experiment switches and historical comparison launcher behavior are unchanged.

Decode: 48.65645 s, 1181 timed tokens, 24.2722 tok/s. This is above the previous up-pair-only run (23.1584), but below historical all-OFF (26.1055). This is not evidence of an improvement versus OFF. Generated token counts are 227/248/239/233/239 and draft accepted/generated 86/141,100/148,97/141,110/122,108/130; histories differ from baseline. User adoption is distinct from proven performance benefit. Do not claim fixed workload equivalence or full-model numerical equivalence. No commit or main integration was requested.

## Adjacent OFF follow-up

`dialogue-speed-20261117-204629`: all three flags OFF, exit 0, 49.85952 s decode / 1216 timed tokens = 24.3885 tok/s. Adjacent all-ON is 24.2722 tok/s (-0.48%), essentially no demonstrated benefit. OFF regenerates historical baseline token counts and MTP accepted/generated counts; ON differs. Total seconds alone favor ON because it produces fewer tokens and are not a valid fixed-work comparison. Historical OFF26.105 versus current OFF24.389 also demonstrates run variation. Recommend withholding a performance-benefit claim and reconsidering all-ON adoption; user decision pending. No code/default/commit change.
