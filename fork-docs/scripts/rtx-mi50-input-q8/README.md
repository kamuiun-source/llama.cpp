# RTX to MI50 Q8_1 handoff microbenchmark

Run in the CUDA/ROCm development environment with both GPUs available. From a read-only /src checkout and writable /out:

```sh
/src/fork-docs/scripts/rtx-mi50-input-q8/build.sh /out
timeout 120s /out/rtx-mi50-input-q8 0 0 3000 100
```

Arguments are HIP ordinal, CUDA ordinal, sample count per arm, and warmup count per arm. Set `CUDA_HOME` or `HIPCXX` to select different compilers. The CUDA kernel is built for sm_89 with `-use_fast_math`, as in the RTX production build. To diagnose a bitwise difference, build into a separate directory with `CUDA_STRICT=1` and use the same arguments. CUDA and HIP link separate copies of the production `quantize.cu`; only the benchmark entry points are exported by each library.

All modes start with the same 3×2560 F32 activation already on RTX. A CUDA D2H copy publishes the same 30,720-byte F32 mirror on every arm. This copy stands in for the F32 publication performed inside the model's CUDA producer. The RTX Q8 arms use the production CUDA quantizer, then publish its 8,640-byte Q8_1 output to a separate mapped page range. One writes to the mapped range directly, while the other writes to RTX VRAM and copies the result to that range using CUDA D2H. The host synchronizes the CUDA stream before submitting any MI50 work.

| Mode | MI50 stage |
| --- | --- |
| `hip_blit_scalar_q8` | Original mapped F32 float4 blit, then the HIP quantizer |
| `hip_mapped_scalar_q8` | The HIP quantizer reads mapped F32 directly |
| `rtx_mapped_q8` | Copies the RTX-produced, direct-mapped Q8_1 from RAM into MI50 VRAM with 16-byte reads |
| `rtx_device_q8_d2h` | Copies the RTX-produced Q8_1 from the same RAM range into MI50 VRAM after the RTX D2H publication |

The host synchronizes MI50 before the measured interval ends in every arm. Timing uses steady-clock samples with rotated arm order. The complete interval includes RTX publication and quantization when present, the CPU-mediated handoff, and MI50 preparation. The printed subintervals include host submission and synchronization. Setup transfers and correctness copies are outside timing. The benchmark has one mapped input and no competing work, Expert computation, or graph launch.

For each arm the harness compares the full F32 mirror against the prepared input and compares the full device Q8_1 output to the HIP baseline for varied data, zero blocks and signed rounding ties. Scales, sums and quant values are included. A mismatch still prints diagnostic timing with `correctness=fail` and returns status 3. Status 77 means the selected HIP device lacks the required gfx906 mapped-host configuration.
