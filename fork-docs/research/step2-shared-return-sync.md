# 改善2：Shared Expert 結果の受け取り後の CUDA 同期をイベント待ちに替える

> **状態**：作業指示。未実装・未計測。数値のうち「実測」以外は見積もり。  
> **対象コミット**：`21b588e`（行番号はこのコミット時点）  
> **対象**：Decode 経路のみ。Prefill は対象外。  
> **適用条件**：`GGML_SCHED_SHARED_HOST_EXPERT_RETURN=1`（`-SharedHostExpertReturn`）のときだけ通る箇所。

## 現状の流れ

対象：`ggml/src/ggml-backend.cpp` の `ggml_backend_sched_compute_splits`。

1. Vulkan の routed Expert split が、結果を host 可視のバッファ `sched->shared_return.host` へ readback する（3132–3145 行）。
2. 次の CUDA split で、Shared Expert の prefix を先に投入する（2970–2985 行）。
3. CUDA split の入力コピーで、Vulkan の完了を待ち（3019 行）、host バッファから CUDA へ H2D を積む（3023 行）。
4. その直後に `ggml_backend_synchronize(split_backend)`（CUDA）で、Shared prefix と H2D の完了まで CPU が待つ（3024 行）。
5. 待ちが終わってから、CUDA のメイングラフを投入する。

4 の目的は、コメント（3022 行）のとおり、CUDA が host バッファを読み終わる前に Vulkan が次の readback で上書きしないようにすること。

## 問題

- CUDA のストリームは順序が保たれるので、メイングラフは同期しなくても prefix と H2D の後に実行される。CUDA の計算結果の正しさのためには、この同期は要らない。
- 同期しているあいだ CPU はメイングラフを投入できない。そのため、prefix の完了からメイングラフの最初のカーネルが始まるまで、CUDA に空きができる（投入の遅延分）。
- 上書きを防ぐだけなら、「次に Vulkan が同じバッファへ readback する前に、CUDA の H2D が終わっている」ことを保証すれば足りる。

1 回あたりの空きは投入の遅延程度（数十〜100 µs 程度と見込む。未計測、Windows では大きめになりうる）。回数は MoE 層ごとに 1 回と見込み、最大 48 回/verify。合計で数 ms/verify の可能性がある（見積もり）。

## 手順0：計測

1. 既存の `GGML_SCHED_TRACE_EXPERT_HANDOFF=1`（3277 行）のトレースで、該当 split の投入時刻を確認する。
2. 3024 行の同期の前後を `ggml_time_us()` で計り、回数と合計を `shared host expert return` のログ行（3405 行）に足す。
3. 回数 ÷ verify 数で、1 verify あたりの回数を確認する。

**計測上の注意**：ここで分かるのは CPU が待った時間で、その大半は GPU が計算中の時間（損ではない）。損になるのは同期明けの投入遅延なので、効果は最終的に手順1の A/B（tok/s）で判断する。

## 手順1：実装

環境変数（例：`GGML_SCHED_SHARED_RETURN_EVENT=1`）で、新旧を切り替えられるようにする。

1. `sched` に CUDA 側のイベントを 1 つ持たせる。作成は既存の `prefetch_events`（2795 行）と同じく `ggml_backend_event_new(backend->device)`。破棄は `sched` の解放処理（3411–3412 行付近）で行う。
2. 3024 行の `ggml_backend_synchronize(split_backend)` を `ggml_backend_event_record(event, split_backend)` に替える。「H2D が未完了かもしれない」ことを示すフラグを立てる。
3. 次に Vulkan が `shared_return.host` へ書き込む直前、つまり compute ラムダの `graph_readback` / `upload.compute` を呼ぶ前（3130–3139 行の手前）で、フラグが立っていれば `ggml_backend_event_synchronize(event)` を呼んでフラグを下ろす。
4. 次の箇所でもフラグが立っていれば同じく待つ。
   - `shared_return_wait` のデストラクタ（2945–2950 行）。途中で失敗した場合と、グラフの終わり。
   - `sched` のリセット・解放時。
5. `GGML_SCHED_SHARED_HOST_EXPERT_INPUT` の経路（`upload.compute`）も、同じ `sched->shared_return.vk_buffer` を渡している（3138 行）。3 の待ちは、`graph_readback` と `upload.compute` のどちらを呼ぶ場合でも入れる。

### 注意

- 今は、次の Vulkan split の前に CUDA→Vulkan の入力コピーでブロッキング同期が入る（2378–2384 行）。このため、イベントを入れなくても上書きは起きない可能性が高い。ただし、それは別の箇所の同期に頼った安全性になる。後で入力コピーをイベント化したときに壊れないよう、明示的にイベントで待つ形にする。
- 3019 行の Vulkan 側の同期は残す。CUDA が読む前に readback が終わっている必要があるため。

## 確認

1. 同じプロンプト・greedy・同じ設定で、新旧の生成トークン列と MTP 受理数が一致すること。
2. 手順3の待ちを外したビルドで、イベント待ちが本当に必要かを確かめる。
   - 長めの生成で出力が一致しなくなれば、待ちが効いている。
   - 一致したままなら、上の「注意」のとおり別の同期で守られていることになる。
   - どちらの結果だったかを記録する。
3. 計測ログなしの通常実行で、tok/s を新旧それぞれ 3 回ずつ取る。

> **安全上の補足**：手順2の「待ちを外したビルド」は検証専用とし、通常利用・採用版には残さない。

## 記録すること

- 1 verify あたりの該当回数、手順0の CPU 待ち時間
- 新旧の tok/s（各 3 回）、出力一致の結果、確認2の結果
- 採用した場合は、環境変数を `agent.md` の採用設定に追記
