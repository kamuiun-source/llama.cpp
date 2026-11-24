# 検証設定とON/OFFフラグ

更新 2026-11-20。直近の高速3幅比較の `japanese/width-3/settings.json` に基づく。実験フラグはブランチによって追加・削除される。再実行前に対象のコード、ビルド、ランナーを確認する。

## 高速固定3幅の採用設定

| 対象 | 設定 |
| --- | --- |
| 検証幅と下書き | `LLAMA_MTP_VERIFY_THREE=1`、`LLAMA_MTP_VERIFY_THREE_FAST=1`。全語彙、`--spec-draft-p-min 0`。再現比較は `temp=0`。 |
| MI50のExpert計算・転送 | `GGML_VK_MTP_ROUTED_REDUCE=1`、`GGML_VK_MTP_ROUTED_WEIGHTED=1`、`GGML_VK_MTP_ROUTED_GATE_UP=1`、`LLAMA_MTP_VERIFY_THREE_OVERLAP=1` |
| RTXのHC | `GGML_CUDA_HC_DOWN_REUSE=1`、`GGML_CUDA_HC_UP_REUSE=1`、`GGML_CUDA_HC_INJECT_SPLIT=1`、`GGML_CUDA_HC_MTP_THREE=1` |
| 共有RAM・Vulkan Submit | `GGML_SCHED_MTP_THREE_SHARED_INPUT=1`、`GGML_SCHED_MTP_THREE_SHARED_OVERLAP=1`、`GGML_SCHED_MTP_THREE_ROUTER_UPLOAD=1`、`GGML_SCHED_MTP_THREE_MERGE_EMPTY_WAVE=1` |
| Prefill | `LLAMA_EXPERT_PREFILL_512=1`、`LLAMA_EXPERT_PREFILL_512_POLICY=auto`、`LLAMA_EXPERT_PREFILL_LOGICAL_TICKS=1`、`LLAMA_EXPERT_PREFILL_OVERLAP_TEN=1` |

Auto専用の採用済み改善は `GGML_SCHED_MTP_AUTO_RESET_SHARED_PLAN=1`、`LLAMA_MTP_AUTO_LAZY_DRAFT_ROW=1`、`LLAMA_MTP_AUTO_DEVICE_PENDING=1`。固定3幅の再現設定では3つとも `0`。

## 比較でOFFにする設定

| 設定 | OFF値 | 過去の判断 |
| --- | --- | --- |
| `GGML_SCHED_SHARED_INPUT_PRODUCER_COPY` | `0` | 3入力の同一CUDAストリーム読み戻し。MTP限定でもウォーム後の両順で約1.2 tok/s低下。 |
| `GGML_VK_SKIP_EMPTY_POOL_RESET` | `0` | Vulkanの空コマンドプールのリセット省略。ウォーム後の両順で改善を確認できず、Fence再利用は維持。 |
| `GGML_VK_MTP_ROUTED_GATE_UP_PAIR` | `0` | IQ2_S・IQ3_XXSの2位置重み共有。dispatch削減を確認したがウォーム後のDecode速度が低下。 |
| `LLAMA_MTP_DRAFT_VOCAB` | 未設定 | 語彙縮小。日本語では受理率を維持し、コードでは受理率低下とDecode時間増加。 |
| `LLAMA_EXPERT_RTX_COLD_SLOTS` | 未設定 | RTXへの非駐Expert分担は保留。 |
| `GGML_CUDA_HC_MIX_FUSE`、`GGML_CUDA_HC_SCALE_SILU_FUSE`、`GGML_CUDA_HC_UP_PAIR` | 未設定 | 旧HC実験。採用済みのdown/up/injectと区別する。 |

`GGML_VK_ASYNC_USE_TRANSFER_QUEUE` は値の存在で判定するため、専用Expert並行転送の比較では環境変数を削除する。直近の設定は `GGML_VK_VISIBLE_DEVICES=0,2,1` で `CUDA0,Vulkan1` を使用。番号は起動前に `--list-devices` で機種名を照合する。

通常速度では詳細計測と `GGML_CUDA_DISABLE_GRAPHS` を未設定にする。計測用の追加同期を使う実験は別のログに記録する。新しいフラグを記録するときは、正確な名前、ON・OFFの値、適用条件または既定値、検証した版と結果を添える。

## ROCmの次層Expert先読み

| 設定 | 試作時の値 | 動作 |
| --- | --- | --- |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH` | `1` | 固定3幅で、MI50の共有8枠へ次層の候補を先読みする。 |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_HIT_WAIT` | `1` | 必要な先読みまでをイベントで待ち、ヒットを通常の仮置き枠へD2Dする。 |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH_AFTER_SUBMIT` | `0`（既定） / `1`（試作） | 上の2つが有効で、予測とExpertが異なるバックエンドの場合に、今のExpert分割を投入してから次層の先読みを発行する。 |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_SCORE_FILTER` | `0`（既定） / `1`・`2`（比較用） | `1` は予測ロジットの閾値、`2` はTop-20のスコア順で既存Top-10と同じ件数（最大8）を発行。ともに固定5問で比較済み。 |
