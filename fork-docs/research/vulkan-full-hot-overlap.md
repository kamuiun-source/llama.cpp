# Vulkan の Hot Expert FFN 全体と転送のオーバーラップ

`LLAMA_EXPERT_VK_FULL_HOT_OVERLAP=1` は実験用スイッチ。既存の `LLAMA_EXPERT_VK_OVERLAP_UPLOAD=1` が有効で、RAM から MI50 への cold Expert の転送と hot Expert の計算が同じグラフで必要な場合に使う。動的常駐と promotion writeback の設定は別のまま。RTX cold offload も別設定で、初回の比較では無効にする。

従来は hot Expert の最初の投影を記録した時点で転送完了を待つ。新しい経路は、共通の routed ID を使う Vulkan の gate/up → SwiGLU → down について、hot Expert の選択位置だけを down まで実行してから待つ。その後 cold の選択位置を同じ順で処理する。両者とも元の選択順の位置へ書くので、後続のルーター重み付けと合算は従来のグラフの処理を使う。

対象は 1–2 トークン、最大 32 Expert 選択位置、F32 の中間値を使うバイアス・スケールなしの連続した投影列。gate と up の別々の投影と、結合投影を 2 つの view に分ける形に対応する。別のノードが途中結果を使うなど、順序を変えられない場合は最初の投影だけのオーバーラップへ戻る。hot と cold で計算領域が重なる配置では、再利用する Vulkan 作業バッファを使い、down の結果を最後に本来の領域へコピーする。Flash-Next の別投影で 2 トークン × 10 選択位置なら、作業バッファ本体は約 350 KiB、追加コピーは約 200 KiB。

down が Q8_1 入力変換を使う場合も従来経路へ戻る。現行の変換は down の入力全体を読むため、hot フェーズで実行すると未計算の cold 行を読むことになる。手元の Flash-Next `UD-IQ3_XXS` では、MI50 の通常の 2 トークン設定で `Q2_0` の down を使う 48 層中 3 層がこの条件に該当する。gate/up の共通入力には、既存の作業領域が足りる場合に従来の Q8_1 変換を使用できる。

同一ビルドの A/B 比較は次で実行する。両ケースとも既存の upload overlap と promotion overlap、動的常駐、MTP を有効にし、RTX cold offload を無効にする。比較中に切り替えるのは full hot のスイッチのみ。

```powershell
pwsh -NoProfile -File .\fork-docs\scripts\compare-vulkan-full-hot-overlap.ps1 -Mode speed -Tokens 256
pwsh -NoProfile -File .\fork-docs\scripts\compare-vulkan-full-hot-overlap.ps1 -Mode telemetry -Tokens 256
```

ログは `G:\AI\llama-hetero-moe-local\logs` に保存する。生成出力、Expert のヒット・転送バイト・昇格、MTP の受理数を両ケースで比較する。`split_ffns` が FFN 全体で分けた回数、`scratch_ffns` が作業バッファを使った回数、`fallback_ffns` が従来経路へ戻った回数。`fallback_down_q8` はそのうち down が Q8_1 入力変換を必要とした回数。GPU テレメトリのセグメント時刻は記録区間全体の値で、FFN カーネルだけの時間ではない。速度の評価には生成速度と転送・計算キューの指標も合わせて使う。

MI50 の小規模テストでは、同じ GGUF から切り出した量子化 Expert の 2 トークン × 10 選択位置で、従来経路と full hot 経路の結果がビット単位で一致した。作業領域が重なるケース、結合 gate/up、down Q8_1 や外部利用のある中間結果によるフォールバック、グラフを実行せずに同期する場合も確認した。実モデル全体の速度と生成出力は未計測。
