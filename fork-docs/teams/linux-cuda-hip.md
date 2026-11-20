# Linux CUDA＋HIP の引き継ぎ

更新 2026-11-20

## MI50 単体

Linux の環境構築は先行着手。Ubuntu 26.04 上の Docker で、既存の `mxxm/llama-cpp-gfx906:rocm-727` を使い、MI50 単体で共有ドライブ上の Qwen3.6-35B-A3B Q4_K_M の対話生成を確認した。ユーザーの体感では Vulkan より ROCm のほうが大幅に速かった。生成速度の数値と Vulkan 側の比較条件は未記録。

## フォークの CUDA＋HIP ビルド

`docker_images/rocm_cuda/Dockerfile` の CUDA 12.9・ROCm 7.2.1 環境でフォークのビルドが完了した。CUDA の `std::memcpy` 用に `<cstring>`、HIP のイベント照会用に `cudaEventQuery` から `hipEventQuery` への定義を追加。動的バックエンドを `GGML_NATIVE=OFF` でビルドし、同じ `llama-cli --list-devices` で RTX 4070 Ti が `CUDA0`、MI50 が `ROCm0` と表示された。次はフォークで各 GPU の演算を確認する。両 GPU の混在推論、GPU 間転送、MI50 の Vulkan/HIP 同条件比較は未確認。

- ビルド時のチェックアウトは `codex/mtp-three-integrated-validation` の `82c2d65db`。上記の互換修正を作業ツリーに追加した状態で確認した。
- 開発イメージは `llama-cuda-rocm-dev`、ビルド先は `build-cuda-hip`。GPUターゲットは RTX 4070 Ti が `sm_89`、MI50 が `gfx906`。モデルフォルダはコンテナ内の `/models` に読み取り専用でマウントする。

## 次の検証

- 通常配置の Flash-Next UD-IQ3_XXS を MI50 単体で検証。MTP 無効、2K、`--load-mode none --lazy-mode on --no-warmup`、`GGML_CUDA_NO_PINNED=1` でロードとプロンプトのトークン化まで進んだが、`rocm-727` と `rocm-633` の両方で初回 prefill が `exit=139`、Docker の `OOMKilled=false`。ホストのカーネルログでそれぞれ `librocblas.so.5` と `librocblas.so.4` 内の segfault を確認。落ちた BLAS API の特定と生成速度の測定は未実施。
- 外部実例では MI50 16 GiB 3枚と gfx906 用 rocBLAS/Tensile を補った ROCm 6.2.4 で Flash-Next Q6_K が生成できた。別の Qwen 系モデルでは gfx906 の `SOLVE_TRI` から rocBLAS の `strsm` へ落ちる既報があるが、今回の BLAS 呼び出しは未特定。別系列の比較候補は TheRock 由来のコミュニティイメージ `mixa3607/llama.cpp-gfx906:v0.9.0-rocm-10.1`。このホストでの動作は未確認。 [Flash-Next 実機報告](https://forum.level1techs.com/t/qwen3-8-flash-next-q6-k-on-3x-radeon-instinct-mi50-16gb/256455) · [SOLVE_TRI の既報](https://github.com/ggml-org/llama.cpp/issues/19972) · [比較候補のタグ](https://hub.docker.com/layers/mixa3607/llama.cpp-gfx906/v0.9.0-rocm-10.1/images/sha256-4e8f5f4ac34426931f8bd469754b7a372fac5080414067880ed2116ab859f7d2)
- 6.3.3 の rocBLAS トレースでは、クラッシュ直前の最後の呼び出しは BF16 入力、FP32 出力・累積の `rocblas_gemm_ex`（M=128、N=24、K=2560）。GGUF 内の BF16 [2560,128] の12テンソルはすべて `blk.*.indexer.k_proj.weight`。QSA indexer の K 射影が有力。次の比較は同一条件で `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` を試し、該当形状が FP32 の GEMM へ切り替わるか確認する。
- 同じ 6.3.3 の MI50 単体テストに `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` を追加すると、日本語プロンプトの生成に成功。ユーザー報告の計測は Prompt 32.0 tok/s、Generation 22.9 tok/s。計測時の `ROCBLAS_LAYER` の状態、生成トークン数、反復値は未確認。再計測では FP32 指定を維持し、`ROCBLAS_LAYER=1` を外す。
- 配布版 `mxxm/llama-cpp-gfx906:rocm-727` でも全体 FP32 で生成に成功。MI50 単体、MTP 無効、2K、同じ日本語プロンプトの計測値は Prompt 34.9 tok/s、Generation 24.2 tok/s。rocBLAS トレースを指定しないコマンドでの報告。反復測定は未実施。局所 FP32 を測ったフォークの ROCm 7.2.1 とは、イメージと実行ファイルも異なる。

移植は以下の5項目で管理する。01と02で基準を確立し、03の同期経路を確認してから04と05へ進む。

各項目の作業番号を1件ずつ進める。実装が必要な作業は Sol に渡し、差分を確認してから 6 Luna に Docker ビルドと実機確認を渡す。既存テストで済む作業はコード変更を挟まない。01-1〜01-4、02-1〜02-3、03-1〜03-4、04-1、04-2a、04-2b、04-3 は実機確認済み。次は 04-4 の広幅 prefill。確認結果を各項目の Markdown に追記し、生ログは Git の外に置く。

1. [Routed Expert の既存 HIP 演算](../rocm/01-routed-expert.md)
2. [CUDA と HIP の共存・GPU 間転送](../rocm/02-cuda-hip-transfer.md)
3. [Packed Expert の配置と動的常駐](../rocm/03-expert-placement.md)
4. [MI50 内の転送と演算の重複](../rocm/04-mi50-transfer-overlap.md)
5. [Expert カーネル融合と実行回数の削減](../rocm/05-expert-fusions.md)

フォーク元との差分と、同期経路・転送・カーネルを移植する際の条件は [Vulkan Expert 実装の ROCm 移植調査](../research/linux-rocm-portability.md) を参照。

各GPUで独立に小さな演算を実行し、CPU参照との一致を確認する。続いてホストメモリ経由のGPU間往復転送を確認し、混在推論へ進む。MI50単体のVulkanとHIPの速度比較は、同じソース・モデルまたは演算形状・入力で測る。検証手順と移植の境界は [Linux ROCm 移植メモ](../design/linux-rocm-port.md) を参照。
