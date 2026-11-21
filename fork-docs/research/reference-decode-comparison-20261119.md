# Flash-Nextの投稿図とWindows検証版の比較

2026-11-19。比較対象はRTX 3060＋MI50（CUDA＋ROCm）の投稿と、RTX 4070 Ti＋MI50（CUDA＋Vulkan）のローカル検証。投稿者の実装コードは未確認。以下の実測はMTP限定パッチを適用する前の、1,378入力・256生成・コンテキスト2,048の計測ONログに基づく。

## 資料の範囲

- [配置図](https://pbs.twimg.com/media/HUHRpLUakAAVwsB?format=webp&name=4096x4096) は256K設定での配置容量を示す。104.4 GiB中、RTXの重み4.9 GiB、MI50のExpert 29.2 GiB、RAMのExpert 21.4 GiB、SSDのPLE 48.3 GiB。
- [スロット図](https://pbs.twimg.com/media/HUHRzvvakAA7wUm?format=webp&name=4096x4096) は全48層で14,400スロット、保護スロットの古いExpertを裏で書き戻し、各層8個の仮置きスロットをコピーなしで再利用・破棄する構造を示す。RAMとGPUの並びをそろえ、ミス時は2.08 MiBを1 DMAで転送。要求ヒット率97%。層間の動的な枠の貸し借りは図から確定できない。

## 優先して切り分ける差

| 対象 | 検証版の実装・観測 | 次に見る値 |
|---|---|---|
| Expert重みの転送 | 通常のRAM原本から重みテンソルの部分ごとにピン留めステージングへコピーし、複数領域をVulkanにまとめて投入する。有限のピン留め回転領域もある。activationの共有RAMとは別。 | ミス1件あたりのホストコピー時間、コピー領域数、DMA完了待ち。まず2幅のDecodeだけで測る。 |
| 動的昇格 | 昇格対象のアップロードと追い出すExpertの書き戻しを転送コマンドに載せ、完了後に配置表を更新。投稿図は古い保護枠を事前に仮置きへ変換できる構造。 | 昇格の有無別に、次の演算開始を遅らせた書き戻し時間と頻度を比較。 |
| 常駐・仮置き | 層別の重みと配置表、300常駐＋20 scratch。両方にヒットしてよい。検証対象154回の要求147,840件でミス9,235件（93.75%ヒット）。最初の窓87.31%、後半94.66%。 | 同一入力で時間経過別に比較。投稿の97%はワークロード不明。 |
| MTPと本体検証 | 候補1個、本体検証2行。154回で101候補採用。検証経過時間8.772秒から48層で均すと1.187 ms/層相当。個別層のGPU時間とは計時範囲が違う。 | 検証1回の時間と1回で確定するトークン数を別々に記録。投稿の平均70・ピーク106の候補数、受理率、プロンプト、実際の履歴長は不明。 |
| QSAとPLE | QSAの4-head indexer、pooled key再利用、top-k、マスク、条件付きsparse attentionは実装済み。2Kでは長いKV向けのsparse gatherの長さ条件を満たさない。PLEは遅延行読み込み済み。16論理行が16回の物理SSDアクセスになるかはOSキャッシュ次第。 | 2Kで可視poolが選択予算内の区間を数え、順位計算を省ける場合のlogitsと時間を比較。長文脈評価は別条件で行う。 |

実装箇所は `src/llama-expert-placement.cpp`、`src/llama-moe-cache.cpp` の `overlap_packed_upload` とpromotion処理、`ggml/src/ggml-vulkan/ggml-vulkan.cpp`、`src/models/qwen4exp.cpp`、`ggml/src/ggml-cuda/lightning-indexer.cu`、`ggml/src/ggml-cuda/fattn.cu`。集計元はGit外の `shared-input-producer-telemetry-20261119-171624-c12788d4/on/metrics.json` と同じディレクトリの `llama.log`。

1層0.87 msが全48層の検証パス平均だと仮定すると41.76 ms/パス。こちらの実効約1.66トークン/パスでは、追加費用を除いても約40 tok/sになる。投稿本文の「検証1層」の範囲は未確認で、この換算から投稿のMTP幅や性能上限は決められない。
