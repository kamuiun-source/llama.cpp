# gfx906 mapped-input Q8_1 microbenchmark

From the repository checkout on a ROCm machine:

```sh
./fork-docs/scripts/mapped-input-q8/build.sh /tmp/mapped-input-q8
timeout 120s /tmp/mapped-input-q8 0 128 30
```

The arguments select the HIP device, repetitions per graph, and timed samples per mode. The default compiler is `/opt/rocm/bin/hipcc`. In the development container, mount the checkout read-only and put the executable and logs in a separate writable mount.

Each path uses the same mapped 3×2560 F32 input and the same device Q8_1 destination. The baseline launches the 15-block, 128-thread float4 blit followed by the production quantizer from `quantize.cu`. The scalar direct path runs that same quantizer on the mapped pointer. The vector direct path loads four adjacent floats per thread and reduces local maxima and integer sums over groups of eight threads before writing the standard Q8_1 layout. Device assembly for gfx906 with `-O3` contains `global_load_dwordx4` in the vector kernel.

The harness compares all 8,640 bytes, including scales and sums, for varied input, zero blocks, and signed rounding ties. It poisons the destination before each comparison. Timing uses one HIP event pair per captured sequence of operations, five warmup graphs per path and a rotating measurement order. Each reported sample is the full graph interval divided by the number of repetitions. It includes graph scheduling effects and repeated reads of one mapped payload. The test isolates the data conversion and omits Expert uploads, model dispatch overlap and inter-GPU handoff.

Exit status 77 means gfx906 mapped-host operation is unavailable on the selected device. Status 3 indicates an output mismatch.
