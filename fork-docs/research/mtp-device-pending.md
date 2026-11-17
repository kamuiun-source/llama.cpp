# MTP pending hidden row on CUDA

`-MtpDevicePending` enables `LLAMA_MTP_DEVICE_PENDING=1`. It applies to the existing Qwen4exp target and separate single-head draft with one sequence, one candidate, two-token target verification, one recurrent rollback slot, and a one-token draft microbatch. Both contexts must use the same CUDA output device, and the draft must accept the target's device rows. Otherwise the existing host or device-row path supplies the hidden state. The separate draft-anchor experiment takes precedence if both flags are set.

The target's `h_nextn_dev` buffer already reserves one row per sequence. The catch-up graph reads the previous pending row there. After catch-up, the target stream copies the last verification row to the reserved row. A rejected candidate replaces that row with verification row zero. The next draft reads the reserved row through the existing state-row input. CUDA events order the write before the draft read and protect the previous read before overwriting the reserved row. The draft context drains before releasing its reference to the target buffer.

The one-candidate draft does not extract its own unused hidden output. Initialization and checkpoint restores can upload a host row. Checkpoint serialization reads the reserved row back into the existing host format. The target batch, positions, candidate selection and rollback rules are unchanged.

At shutdown, `MTP device pending: enabled=... copies=... d2h=... h2d=... draft_host_rows=...` reports the device-to-device row copies, pending-row readbacks and uploads, and the number of draft input rows supplied from host memory. The last count is separate from the reserved-row upload count. On the Flash-Next setup a hidden row is 40,960 bytes. The counters do not include logits, sampler output, or other model transfers. The launcher requires a nonzero copy count when the option is enabled.

## Comparison

Build the validated experiment branch in the fixed validation checkout, then run the same five prompts with the same options. Both runs enable Expert RAM rotation and shared RAM in both directions. Omit the HC options and RTX cold Expert sharing.

```powershell
cmake --build G:/AI/build-rtx-validation --config Release --target llama-cli
& ./fork-docs/scripts/start-dialogue-measurement.ps1 -Mode speed -ExpertRamRotation -SharedHostExpertInput -SharedHostExpertReturn
& ./fork-docs/scripts/start-dialogue-measurement.ps1 -Mode speed -ExpertRamRotation -SharedHostExpertInput -SharedHostExpertReturn -MtpDevicePending
```

Compare generated text, draft acceptance, and decode throughput. Logs stay outside Git. [Fixed dialogue prompts](dialogue-benchmark-prompts.md)

## Small checks

Both changed C++ translation units passed MSVC syntax checks. The isolated RTX 4070 Ti fixture used the extracted pending-row, catch-up, acceptance, and checkpoint methods with real CUDA buffers, two backend streams, and a replayed graph that reads the selected rows. With eight two-token rounds and alternating acceptance/rejection, the new path performed 12 D2D copies, zero steady-state D2H, and one initial upload. The preceding path performed 12 D2H reads and eight uploads. Saving added one readback and restoring added one upload in the new path. CPU output-device and multiple-sequence configurations were rejected by the device-pending eligibility check. A final queued read returned its original value after the target queued an overwrite and the draft detached. Full-model output and speed remain to be measured.
