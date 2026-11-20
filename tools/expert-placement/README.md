# Flash-Next expert placement

`llama-expert-placement` loads only the 48 layers' routed expert weights. It places the 300 experts listed for each layer in the requested GPU buffer and places the other 212 in host buffers. Both buffers preserve original expert ID order within their own subset. The tool exits after reporting placement.

Generate the profile from [`expert_map_sequences.jsonl.gz` revision `54779ca`](https://huggingface.co/datasets/tcclaviger/Qwen3.8-Flash-Next-expert-activation-map/tree/54779ca), then build the target in a llama.cpp build configured with Vulkan.

```text
python tools/expert-placement/make-profile.py expert_map_sequences.jsonl.gz flash-next-top300.txt
cmake --build build --config Release --target llama-expert-placement
```

Run the executable with the first GGUF shard, the generated profile and the backend device name.

```text
llama-expert-placement FIRST_SHARD.gguf flash-next-top300.txt Vulkan1
```

The final line reports the weight bytes loaded in each buffer. For `UD-IQ3_XXS` the expected values are `29,468,160,000` on the GPU and `20,824,166,400` in host buffers.

## Synchronous generation

Use a `llama-cli` built from this branch with both CUDA and Vulkan available. Set `LLAMA_EXPERT_PLACEMENT` to the profile file and `LLAMA_EXPERT_DEVICE` to the GPU that will execute the routed experts. Select both GPUs and place the ordinary layers on the other GPU. Use `--fit off` with this explicit placement.

```powershell
$env:LLAMA_EXPERT_PLACEMENT = (Resolve-Path .\flash-next-top300.txt).Path
$env:LLAMA_EXPERT_DEVICE = 'Vulkan1'
llama-cli -m FIRST_SHARD.gguf -dev 'CUDA0,Vulkan1' -sm layer -ts '1,0' -ngl all `
    --fit off --load-mode none --lazy-mode on --spec-type none `
    -c 2048 -b 256 -ub 1 --single-turn --reasoning off -n 8 -p 'Say hello.'
```

The first version uses one token per microbatch, with one layer's top-k count reserved as scratch slots in each layer's GPU bank. The MoE cache log reports resident hits and RAM uploads. Clear the two environment variables to use ordinary placement in the same shell.

## Overlap on Vulkan

With a separate Vulkan transfer queue, enable `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` and `LLAMA_EXPERT_OVERLAP=1` before starting the same command. GGML submits the resident graph first. Cold weights go through Vulkan host staging to a separate device buffer, and the cold graph waits for the copies. Each half uses zero weights for the other half's positions to preserve the Router's output order.

Keep `GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1` when comparing modes. Clear `LLAMA_EXPERT_OVERLAP` to select the synchronous version. The `packed expert transfer phase` log measures the host time spent preparing and completing uploads on layers with misses. The overlap measurement can include waiting for remaining resident work because the backend synchronization waits for both queues.
