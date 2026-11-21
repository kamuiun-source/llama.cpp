# tandem.cpp 取り込み候補調査

調査日: 2026-11-16。調査のみ。実装、ビルド、ベンチマークは実施していない。

## 固定した参照

- Repository: https://github.com/ewhacc/tandem.cpp
- Commit: 64a6b186f650bca7638b7c5baccb118587d1f715 (v0.3.0)
- Base: f5222f25c (BASE_COMMIT確認)
- Source: https://github.com/ewhacc/tandem.cpp/blob/64a6b186f650bca7638b7c5baccb118587d1f715/tandem.patch
- License: MIT。LICENSE実体確認済み。コードを取り込む場合はewhaccおよびggml authorsの著作権・ライセンス通知を維持する。
- License source: https://github.com/ewhacc/tandem.cpp/blob/64a6b186f650bca7638b7c5baccb118587d1f715/LICENSE

全体はllama.cppへのパッチ集。公開API、モデルロード、グラフ、CUDA、CPU、キャッシュ、serverまで変更する。既存の異種GPU Expert配置への一括適用は推奨しない。READMEの検証対象はLinux CUDA中心、一部ROCm。Windows CUDA/Vulkanの実績とは扱わない。

## 1. ホスト入力の一括転送: 優先候補、内訳確認が先

実体: tandem.patchのggml_backend_sched_compute_splits (およそ1012行以降)、ggml_backend_sched_wait_input_copies、CUDA packed upload、llama_contextの入力更新前待機。

GGML_SCHED_BATCH_INPUTSはホストにある非weight入力をまとめる。mode 2はイベントを使い、ホストが入力を再利用する前に待機。mode 3はホストstagingから一括H2DとGPU scatterを使う。対象はggml/src/ggml-backend.cpp、ggml/include/ggml-backend.h、ggml/src/ggml-cuda/ggml-cuda.cu、CUDA context、src/llama-context.cpp等。

当方のinput_copy_and_dep_ms=5383.689はホスト入力だけでなくGPU間転送と依存待ちを含む。GPU bufferを入力とするCUDA/Vulkan転送はこのhost判定に入らない。最初に種類別の時間を分ける。変更を入れる場合はホスト入力のみに限定し、イベントのないバックエンドには従来処理を残す。

## 2. HC mixのCUDA演算融合: RTX側の有力候補

実体: 新規ggml/src/ggml-cuda/hc-fused.cu/.cuh、ggml_cuda_hc_mix_fuse、hc_mix_fuse_at (パッチ3018行以降)、CUDA評価ループへの接続。

RMS norm、norm weight乗算、BF16 down projection、SiLU、BF16 up projection、HC合成の連鎖を少数カーネルにまとめる。RTXに置いているHC処理が対象となりうる。既存詳細計測でHC系処理が目立つため候補価値は高い。ただしその計測はGraphs無効・細粒度同期付きであり、通常実行で同じ削減幅とは見積もらない。

適用条件: 重み型、ノード列、view/stride、融合した中間結果の外部参照を照合する。現在のsrc/models/qwen4exp.cppは分離down/injectと結合down_injectの両形式を扱うため、ダウンロード済みGGUFの実際の形式でmatcherが成立するか確認する。丸め順が変わるとソース自身が明記しており、文字列一致だけでなく短い固定入力のlogits差も確認対象とする。異種GPUをまたぐノードは融合しない。

## 3. Activationのq8_1再利用: 条件付き候補

実体: ggml/src/ggml-cuda/common.cuh、mmvq.cuのggml_cuda_mul_mat_vec_q、CUDA評価開始のepoch更新 (パッチ14188行付近)。同一評価内でbase tensor、data、shape、strideが一致した量子化入力を再利用する。4エントリ各4MiB、計16MiBを確保する実装。

RTX上で同じ入力を読む複数の量子化matvecが残る場合に有効。Routed ExpertはVulkanなので、その演算をこのCUDA変更で高速化できるとは扱わない。CUDA Graphの固定アドレスとキャッシュslot再利用が整合するか確認が必要。カーネル適用・cache hitを確認してから比較する。

## 4. GDN beta/alpha結合とHC scale folding: 重複・型確認が必要

実体: llama_model_base::load_tensors_gdn_gates、load_tensors_hc_fold (パッチ5696行以降)、llama-model.h、src/models/qwen4exp.cpp。

GDNはbeta/alpha重みを結合してmatvec回数を減らすが、元重みを保持して結合重みも追加するためVRAM増加を伴う。当方は既にssm_beta_alpha_catのロード・演算経路を持つ。GGUFが結合済みなら同じ効果を二重に数えない。

HC scale foldingはBF16/F32かつhc倍率が2の累乗という条件でロード時にscaleを重みに織り込む。量子化重みへ一律適用できない。重み書き換え可能性、LoRA、GGUF配置と現行グラフの対応を精査してから扱う。

## 5. 排他的RAM/VRAM Expert store: 将来の動的配置に関連

実体: src/llama-moe-cache.cpp内のmoe_arena::release、claim_wb、arena_excl (パッチ7356行付近以降)。GPUへ昇格したExpertのRAM slotを解放し、GPUから追い出すExpertをRAMへ書き戻す。LLAMA_MOE_ARENA_EXCL_VERIFYで書き戻し内容をファイルと比較する経路もある。

「重複なし」は当方の将来方針に近いが、現状の固定300常駐と10個の一時slotにそのまま追加する変更ではない。所有権、非同期転送中のslot寿命、SSD tier、RAM storeを一緒に設計する必要がある。Linux依存のストレージ処理も含むため、今は設計参考とする。

## 6. 今は後回し

- GGML_CUDA_GRAPH_KEY_SHAPE: MTP検証幅別のGraph管理。当方は通常Decodeで再利用を確認済み。現在の速度改善の主候補ではない。
- LLAMA_MOE_FLEX: CUDA VMMでprefill bufferへExpert cache VRAMを貸し出す。MI50/Vulkan側のExpert領域には直接適用できない。
- 4096-token prefill: 当方はExpert配置時にn_ubatch=1を強制。オプションだけで導入不可。複数tokenのExpert集合を処理できるslot/graph設計が前提。
- MTP状態管理: 専有Expertコンテキスト問題とMTPヘッド確認を先に解決する。
- CPU/GPU hybrid expert計算: ExpertはMI50で計算する現方針とは別案。

## 推奨順と検証範囲

現在進行中のoverlap比較結果を先に評価する。次は入力コピーの種類別内訳を確認し、ホスト入力の待ちが大きければ入力集約、RTX内の実演算が大きければHC融合を一つずつ導入する。Q8再利用は適用率を確認して続ける。

候補ごとの確認は最大4項目: (1)対象経路を実際に使用 (2)短い固定入力の正確性 (3)同条件の通常速度 (4)追加メモリ。未対応のレアケースは改善事項として記録し、今回の対象を広げない。性能向上の保証はしない。
