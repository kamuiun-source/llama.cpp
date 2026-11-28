# 改善1：ミスした Expert の staging memcpy を短くする

> **状態**：作業指示。未実装・未計測。数値のうち「実測」以外は見積もり。  
> **対象コミット**：`21b588e`（行番号はこのコミット時点）  
> **対象範囲**：Decode 経路のみ。Prefill は変更しない。

## 問題

ミスした Expert は、Vulkan へ転送する前に RAM（pageable）から pinned の staging 領域へ CPU で `memcpy` される。

- **overlap 経路**：`src/llama-moe-cache.cpp:1316-1341` のループで、全 fill × 全 plane を `memcpy`（1338 行）し終えてから `overlap_expert_upload` を呼ぶ（1382 行）。
- **fallback 経路**：`upload_packed` の 1273 行。
- **Vulkan 側**：受け取った staging から transfer queue で一括転送し、timeline semaphore を signal する（`ggml/src/ggml-vulkan/ggml-vulkan.cpp:18136-18151`）。cold 計算はこの semaphore を待つ（8691-8697 行）。

この callback の時点では `n_copies == 1` のためスケジューラが同期済みで、MI50 は止まっている。つまり、`memcpy` の時間はほぼそのまま待ち時間になる。さらに、DMA は `memcpy` が全部終わるまで始まらない。

**見積もり（未計測）**：アップロード 45.92 GiB ÷ 698 verify ≒ 70 MiB/verify。pageable `memcpy` を 10～20 GB/s とすると、3.5～7 ms/verify（verify 1 回約 70 ms の 5～10%）。この合計には Prefill 分も入るため、上限寄りの値。

## 手順0：計測（実装より先に行う）

既存の `LLAMA_EXPERT_PROFILE_RAM_COPY`（`--ExpertRamCopyTiming`）が計測しているのは、昇格時の `memcpy`（1211-1217 行）だけ。転送用の `memcpy` は計測されていない。

1. 同じ環境変数が `1` のときだけ、次の2か所の `memcpy` ループの所要時間を `ggml_time_us()` で計る。
   - `overlap_packed_upload` の 1316-1341 行のループ全体
   - `upload_packed` の 1264-1276 行のループ全体
2. 累計を、既存ログ（1893-1894 行）の隣に別行で出力する。例：

   ```text
   packed expert upload RAM memcpy: calls=%zu bytes=%zu cpu_ms=%.3f
   ```

3. 計測した回のログから、`cpu_ms ÷ verify 回数` を出す。

### 計測結果による判断

| memcpy / verify | 次の作業 |
|---|---|
| 2 ms 以上 | 手順1へ |
| 1 ms 未満 | この改善は飛ばし、改善2（CUDA 同期の後回し）へ |
| 1～2 ms | 手順1だけ試し、効果がなければ改善2へ |

## 手順1：memcpy を複数スレッドで並列にする

**変更範囲**：`src/llama-moe-cache.cpp` のみ。Vulkan 側は変更しない。

- 1 回の callback の中で、fill ごと（Expert ごと）に `memcpy` をワーカースレッドに分ける。書き込み先の staging 領域は fill ごとに分かれていて重ならない（`staging_offsets` が連続で割り当てられている）。
- スレッドは毎回作らず、cache の初期化時に 2～4 本作って使い回す（本数は環境変数で切り替えられるようにし、実測で決める）。
- 呼び出し側は、全ワーカーの `memcpy` 完了を待ってから `overlap_expert_upload` / `ggml_backend_tensor_set_async` を呼ぶ。完了待ちより前に Vulkan へ渡さないこと。
- fill が 1 件のときや、合計バイト数が小さいとき（しきい値は実測で決める）は、スレッドを使わず今までどおり 1 回で行う。
- 1 回の `memcpy` のうち pinned プール内の Expert（`packed_ram_slots` が 0 以上のもの）は元から速いので、並列化の対象に含めても含めなくてもよい。

### 確認

1. 同じプロンプト・greedy・同じ設定で、生成トークン列と MTP 受理数が変更前と一致すること。
2. 手順0のログで、`memcpy` の `cpu_ms` が下がっていること。
3. 計測ログなしの通常実行で、tok/s を変更前後それぞれ 3 回ずつ取る。
4. スレッド数 1（＝変更前と同じ動き）で、変更前と同じ速度・同じ出力になること。

## 手順2（手順1で足りない場合のみ）：memcpy と DMA を Expert 単位で重ねる

手順1のあとも `memcpy` が数 ms 残るときだけ行う。Vulkan 側も変更する。

- 現状は、全 Expert の `memcpy` → 1 回の transfer submit。
- これを、Expert k の `memcpy` が終わったら Expert k の転送を submit し、その間に Expert k+1 を `memcpy` する形にする。transfer semaphore は timeline なので、submit ごとに値を 1 つ進め、cold 計算は最後の値を待てばよい。
- `ggml_backend_vk_overlap_expert_transfer`（17936-18172 行）の検証部分は staging のアドレスと大きさしか見ていないため、検証を先に済ませてから `memcpy` を始められる。`memcpy` は `commit` と同じく、関数ポインタ＋userdata で Vulkan 側に渡す。
- `ggml_backend_vk_defer_slot_map`（18132 行）は 1 回だけ呼ぶ。
- 昇格の書き戻し（readback）は最後の submit に入れる。

**確認**：手順1と同じ。加えて、`transfer_pending_at_compute_submit` の値が増えていないことを見る。

## やらないこと

- **RAM 側の Expert をすべて pinned にする**：RAM 側は最大で約 27 GB あり、32 GB の RAM では無理がある。
- **`memcpy` を hot 計算の裏に回す**：今の分割では、hot として先に流せるのは最初の projection の hot 部分だけで、MI50 では短い（見積もり）。重ねられる時間が小さいわりに、Vulkan のグラフ投入の流れを大きく変える必要がある。

## 記録すること

- 変更前後の `packed expert upload RAM memcpy` の値、tok/s（3 回ずつ）、出力一致の結果。
- 採用した場合は、スレッド数としきい値を `agent.md` の採用設定に追記。
