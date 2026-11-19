# Prefill 専用経路: 段階 1〜2 の作業指示

> **状態:** 作業指示。未実装・未計測。数値のうち「実測」以外は見積もり。  
> **対象コミット:** `21b588e`（行番号はこのコミット時点）。  
> **参照:** `prefill-chunk-layer-prefetch.md`（方式全体）。  
> **対象範囲:** 最初の 2 段階のみ。decode 経路は変更しない。

## コードを読んで分かった前提

1. **packed Expert は 3 トークン以上を受け付けない。** `get_slot_map` は `n_tokens <= placement->max_ubatch` でなければ `nullptr` を返す（`src/llama-moe-cache.cpp:1003-1011`）。`build_moe_cache_slots` はそれを例外にする（`src/llama-graph.cpp:2720-2731`）。今の Prefill が 2 トークン単位なのはこのため。
2. **slot map は Expert 番号 → スロットの表で、大きさはトークン数によらない。** グラフは `get_rows(slot_map, ids)` でスロットを引き（`llama-graph.cpp` 2750 行付近）、`mul_mat_id` に渡す。トークン数の制限は 1 のチェックだけで、グラフの形としては N トークンでも成り立つ。
3. **hot / cold を 2 回に分けて足し合わせる仕組みは既にある。** 常駐 bank の計算と cold バッファの計算を `ggml_add` で合算し（`llama-graph.cpp:2612-2620`）、それぞれ相手側の Expert を「ゼロ Expert」に向ける。ゼロ Expert は bank のスロット `n_resident`（=300）と、cold のスロット `n_slots` で、読み込み時に 0 で埋めている（`src/llama-expert-placement.cpp:183-187`）。
4. **cold バッファは層ごとに `n_slots + 1` 枠**（`llama-expert-placement.cpp:111`）。1 つのバッファ `cold_buf` にまとめて確保されている（144-150 行）。採用設定（`BATCH_TOKENS=2`）では 21 枠 × 48 層。
5. **Expert 1 件（3 plane 合計）の平均は約 2.05 MB。** 10 枠 × 48 層の追加で 982,272,000 B と記録されている値からの換算（`research/packed-expert-mtp-batch.md`）。`cold_buf` 全体は約 21 × 48 × 2.05 MB ≒ 2.06 GB の見込み。

そのため、方式メモにある「リングバッファ用に VRAM を新しく確保する」案は、本書では次のように置き換える。

- Prefill の間だけ `cold_buf` を作業領域として使い回す。新しい VRAM は要らない。
- 1 層の cold Expert は最大 212 件＋ゼロ Expert 1 件 ≒ 436 MB（平均値での見積もり）。`cold_buf` を 2 つに分け、偶数層と奇数層で交互に使えば、次層の先読み（段階 4）もこの中に収まる見込み。

## 段階 1: 計測（実装前に必ず行う）

### 1-1. 今の Prefill の速さ

採用設定のまま、計測プロンプト 5 回答で、プロンプトごとのトークン数と Prefill 時間（llama-cli の `prompt eval` 表示）を記録する。コード変更は不要。

### 1-2. Router の選択を書き出す

`copy()` が Router の選択を受け取る箇所（`src/llama-moe-cache.cpp:1550-1560`、`ids` に入った直後）で、環境変数（例: `LLAMA_EXPERT_DUMP_IDS=path`）が設定されているときだけ、次をバイナリで追記する。

- 層番号、ubatch のトークン数、`ids` の中身（トークン × 10）

Prefill と decode の境目は、1-1 で記録したプロンプトのトークン数で切り分ける。

### 1-3. 書き出した選択から集計（オフライン）

スクリプトで、Prefill 部分を `N = 128 / 256 / 512 / 1024` のチャンクに区切り、層ごとに次を出す。

| 項目 | 用途 |
|---|---|
| 使われた Expert の種類数 | 転送量の見積もり |
| そのうち RAM 側（`map.host[expert] >= 0`）の種類数 | cold 転送量 = 種類数 × 約 2.05 MB |
| 全選択のうち RAM 側に当たった割合 | cold の計算量、ゼロ Expert の無駄の大きさ |

常駐の配置（どの Expert が MI50 にあるか）は、計測時点のものを起動ログか dump に一緒に出す。dynamic residency で動くため、Prefill 開始時点の配置を使う。

### 1-4. VRAM の空き

採用設定での MI50 と RTX の空き VRAM、`cold_buf` の実サイズを、起動ログに出す（出ていなければログを足す）。段階 2 で N トークン分の compute buffer が増えるため、N の上限はここで決める。

## 段階 2: 正しさ優先の最小版

先読みも overlap もなし。層ごとに「cold Expert を全部同期で転送 → 計算」する。まず正しく動くことと、2 トークン単位からどれだけ速くなるかを確かめる。

### 2-1. 入口

- `cparams` に Prefill 用の ubatch サイズ（`n_ubatch_prefill`、環境変数で指定、既定は無効）を足す。`n_ubatch`（=2）は変えない（`src/llama-context.cpp:354-370`）。
- `llama_decode` で batch を ubatch に分ける箇所（2351 行 `memory->init_batch(*balloc, cparams.n_ubatch, ...)`）で、batch のトークン数が `max_ubatch` を超えるとき（= プロンプト処理）だけ `n_ubatch_prefill` で分ける。MTP の verify（2 トークン）と draft は従来どおり。
- compute buffer の確保（`graph_reserve`、841 行・893 行付近）を、`n_ubatch_prefill` でも確保されるようにする。VRAM が足りなければ起動時にエラーにし、N を下げて使う。
- MTP の draft catchup が Prompt のトークンをどう処理しているか（target の hidden を何トークン分使うか）は未確認。Prefill を N 単位にしても MTP 側が受け取れることを、最初に確認する。

### 2-2. グラフ

- `get_slot_map`（1003-1011 行）: Prefill モードのときは、トークン数の制限なしで slot map を返す。
- hot 側: 常駐 Expert → bank のスロット、RAM 側 Expert → ゼロ Expert（スロット 300）。
- cold 側: 層ごとに Prefill 用の cold テンソルを用意し、`get_experts(w, cold)` が Prefill モードではそれを返すようにする。
  - Prefill 用 cold テンソルは、`cold_buf` の先頭（偶数層）または後半（奇数層）のアドレスに、その層の型・形で `ne[2] = 213` として置く（`ggml_backend_tensor_alloc` で既存バッファ内のアドレスに割り当てる）。
  - UD 量子化は層ごとに型が違う可能性があるため、層ごとに作る。213 枠が半分の領域に収まるかを、型ごとに起動時に確認する。
  - スロット 212 をゼロ Expert にする。Prefill モードに入るたびに 0 で埋める。
- cold 側の slot map: RAM 側 Expert → その層の cold スロット、常駐 Expert → ゼロ Expert（212）。

### 2-3. copy callback（Prefill モードの分岐）

`copy()`（1472 行〜）の先頭付近で Prefill モードなら次だけを行い、既存の decode 処理には入らない。

- hot 側 slot map: 書き込むだけ（転送なし）。
- cold 側 slot map: このチャンクで使われた RAM 側 Expert を、cold スロットへ同期で転送し、slot map を書く。転送は `upload_packed` の作り（pinned staging 経由）を流用してよい。staging が足りなければ分割して転送する。
- 昇格・退避（dynamic residency）はしない。使用回数の統計も更新しない（Prefill の偏りで decode 用の常駐集合を崩さないため）。
- overlap 経路・RTX 経路・scratch 再利用には入らない。

### 2-4. decode へ戻るとき

Prefill の最後の ubatch のあと、decode の最初の ubatch の前に、次を行う。

1. MI50 の完了を待つ。
2. `cold_buf` を上書きしたので、decode 用の cold テンソルのゼロ Expert（各層のスロット `n_slots`）を 0 で埋め直す。
3. scratch の中身の記録（`scratch[slot]`、scratch 再利用のヒット判定に使う表）をすべて空にする。
4. 常駐配置（`map.resident` / `map.host`）が Prefill 前と同じことを確認する（デバッグビルドでは assert）。

### 2-5. 確認

1. **出力一致:** 同じプロンプト・greedy で、`n_ubatch_prefill` 無効（従来の 2 トークン単位）と有効（N = 128）の生成トークン列を比べる。
   - 浮動小数の合算順が変わるため、完全一致しないことはありうる。不一致のときは、最初の生成トークンの logits の差（上位 10 件の順位と値）を記録して判断する。
   - 有効時の decode の受理数も記録する。
2. **decode への影響なし:** Prefill 後の decode の tok/s とヒット率が従来と同等であること。scratch を空にした分の立ち上がりの差は記録する。
3. **後始末の確認:** 2-4 の手順 2 を外したビルドで出力が崩れることを確かめる。ゼロ Expert を埋め直さないと、decode の hot/cold 合算に cold 側の残りが混ざるはずで、それが崩れとして見えれば手順 2 が効いていることになる。
4. **速さ:** N = 128 / 256 / 512 で、Prefill tok/s を各 3 回。

## 段階 3 以降（今回は範囲外、順番だけ）

3. **ゼロ Expert の無駄を減らす:** 最小版では hot 側・cold 側とも N × 10 の全選択を計算し、相手側の分はゼロ Expert で計算する。cold の割合が小さいほど cold 側の無駄が大きい（1-3 の割合で見積もる）。段階 2 の計測で cold 側の計算時間が目立つ場合は、cold 側だけ「RAM 側に当たった行だけを集めて計算する」形（`get_rows` で行を集め、結果を元の位置へ戻す）を検討する。
4. **次層の先読み:** 層 L の計算中に、層 L+1 の cold Expert を `cold_buf` のもう半分へ転送する。転送は transfer queue と timeline semaphore で、層 L+1 の cold 側の最初の計算の前で待つ（`ggml-vulkan.cpp` の overlap と同じ仕組み）。
5. **層内 hot 先行:** 先読みで足りない分の転送を、hot 側の計算の裏に回す。

## 記録すること

- 段階 1 の集計表（N ごと、層ごと）と VRAM の空き。
- 段階 2 の出力比較の結果、Prefill tok/s、decode tok/s。
- 方式メモ（`prefill-chunk-layer-prefetch.md`）の「転送用の VRAM」節を、`cold_buf` の使い回しに更新する。
