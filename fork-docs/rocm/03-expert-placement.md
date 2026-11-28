# 03 Packed Expert の配置と動的常駐

状態: 03-1 から 03-4 の同期経路を RTX 4070 Ti と MI50 の混在環境で確認済み。

## 現状

対象は Flash-Next の Packed Expert 経路。モデルの配置と基本のバッファ割り当てに既存の汎用経路を使い、HIP の同期動的常駐を明示的に有効にできるようにした。既定の HIP は1位置の固定常駐。

## 03-1 の実機確認

- RTX 4070 Ti (`CUDA0`) に通常レイヤーと Router、MI50 (`ROCm0`) に固定常駐の Routed Expert を配置する。`--tensor-split 1,0` と `--fit off` で通常レイヤーの配置を固定する。
- 公開 activation map から既存の [make-profile.py](../../tools/expert-placement/make-profile.py) で層ごとの上位300 Expertを選ぶ。`LLAMA_EXPERT_BATCH_TOKENS=1` と `--ubatch-size 1` で一度に処理する位置を1つに固定し、MTP と転送重複を切る。
- `GGML_SCHED_TRACE_EXPERT_HANDOFF=1` と `--log-verbosity 5` で Router／Expert の実際の実行先、固定常駐ヒット、RAMアップロードを確認する。起動時の `CUDA to Vulkan` 表示は固定文言なので、バックエンドの判定には実行ログを使う。
- 通常入力の短い生成で接続を確認する。CPU参照との照合には、対象の Expert ID と中間テンソルの比較が必要。01の演算単体テストは各演算のCPU参照として再利用できる。

2026-11-20、`bce912dc3` の CUDA 12.9 / ROCm 7.2.1 ビルドで Qwen3.8-Flash-Next UD-IQ3_XXS に `Say hello.` を入力。トップ300のプロファイルは [公開 activation map の固定リビジョン](https://huggingface.co/datasets/tcclaviger/Qwen3.8-Flash-Next-expert-activation-map/tree/54779ca) から作成した。コンテキスト2048、MTPなし、1位置処理で `Hello! How can I help you today` を生成し、終了コードは0。

- スケジューラは CUDA0 と ROCm0 の間で処理を引き渡し、Routed Expert を ROCm0 に割り当てた。初期配置は MI50 に 29,468,160,000 bytes、RAM に 20,824,166,400 bytes。各層にスクラッチ10枠を確保した。
- 固定常駐ヒット9,614件、スクラッチ再利用596件、シリアルのRAMアップロード1,160件。初回の接続確認で Prompt 20.4 tok/s、Generation 26.7 tok/s を記録した。速度比較向けの計測は未実施。
- 生ログは Git 外の `/tmp/llama-rocm-03-1-short-inference.log`。この実行で使用した同期経路にソース変更は不要だった。

### 03-1 と 03-2 の局所照合

実モデルでトークン9419を1位置ずつ2回処理し、2回目の前に推論コンテキストのメモリをクリアした。初層の Router が選んだ10 Expertのうち、9件は固定常駐。RAM 所有の ID 37 は1回目にスクラッチスロット300へアップロードし、2回目は同じスロットを転送なしで再利用した。

- 2回とも10件すべての IQ4_NL down 重みが元の GGUF データとバイト単位で一致した。
- ROCm の down 出力は、同じ活性と GGUF の重みを使った CPU の `MUL_MAT_ID` に対し、正規化平均二乗誤差が両回とも `1.12e-14`。対象は初層の down 射影。
- 単発の照合コードと生ログは Git 外の `/tmp/llama-rocm-03-1/check-first-layer.cpp` と `/tmp/llama-rocm-03-1-first-layer-diagnostic-cwd-fix.log`。終了コードは0。

## 小分けの作業

| 作業 | 変更範囲 | 確認してから次に進む条件 |
| --- | --- | --- |
| 03-1 固定常駐の1位置 | [GPU 間の同期転送](02-cuda-hip-transfer.md) を前提に、RTX の Router と MI50 の固定常駐 Expert を接続する。 | 既知 ID の常駐ヒットで CPU 参照と一致する。 |
| 03-2 同期ミスとスクラッチ再利用 | RAM からの同期アップロードと同一 Expert のスクラッチ再利用を通す。 | ミスと再利用の両方で出力・元データの所有先・スロット ID が一致する。 |
| 03-3 同期の昇格と退避 | 昇格、退避、退避後の再読込を追加する。 | 上書き前の完了待ちと、失敗時にスロット表が壊れないことを確認する。 |
| 03-4 複数位置の同期生成 | 2位置を有効にし、Flash-Next の短い混在生成を試す。 | 同一入力で Expert ID と出力を確認し、複数位置を基準として記録する。 |

## 03-3 の調査

- 既存の同期昇格は MI50 に適用できる `ggml_backend_tensor_get` を使う。退避する Expert の全プレーンをホストへ読んでから、RAM と VRAM の所有表を更新する。Packed Expert の同期コールバックは Router ID の読み戻し時に MI50 の処理を待つ。
- `LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY=1` で HIP の同期昇格を有効にする。03-3 では1位置を対象にし、退避と後続のアップロードを層・Expert ID・RAM スロットで照合する。
- 03-4 で2位置へ広げ、同期の入出力と1位置基準を比較する。Vulkan 用のステージング、転送重複、プリフィルの有効条件は各機能の確認時に扱う。

### 03-3 の実機結果

Flash-Next に `Explain step by step how a computer loads a large language model and generates a token.` を入力し、1位置処理、MTP 無効、`--temp 0 -n 128` で同期昇格を ON / OFF の各1回確認した。ON では昇格・退避が1,327件、退避後に同じ RAM スロットから再アップロードした Expert が延べ203件。例として第5層（0始まり）の Expert 6 は RAM スロット30へ退避され、後のリクエストで MI50 のスロット8へ転送された。

ON と OFF で、生成中にデコードした127トークンの ID と最終回答が一致した。この比較では、退避された全 Expert の復元データをバイト単位で比較していない。生ログは Git 外の `/tmp/llama-rocm-03-3-dynamic-on.log` と `/tmp/llama-rocm-03-3-dynamic-off.log`。

## 03-4 の2位置処理

`LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY=1`、`LLAMA_EXPERT_BATCH_TOKENS=2`、`--ubatch-size 2` を指定する。MI50 側のスクラッチを各層20枠に広げ、既存の同期アップロード、昇格、退避を使う。

03-3 と同じ入力を使った混在推論は終了コード0。30プロンプトトークンを15回の2位置処理で実行し、Routed Expert の gate、up、down が ROCm0 に割り当てられた。1位置と2位置で生成中の先頭117トークン ID が一致し、118件目で分岐した。回答では `engine like vLLM` と `runtime like vLLM` の違いが出た。生ログは Git 外の `/tmp/llama-rocm-03-4-width2-dynamic-on.log`。

トークン9419と353を同一シーケンスに入れ、1位置ずつと2位置一括で初層を照合した。両位置で Router が選んだ10 Expert の ID と順序が一致し、IQ4_NL down 重みは元の GGUF とバイト単位で一致した。down 出力は1位置と2位置の間で一致し、CPU 参照に対する正規化平均二乗誤差は位置0が `1.12e-14`、位置1が `9.43e-15`。生ログは Git 外の `/tmp/llama-rocm-03-4-width-diagnostic-full.log`。

2026-11-20 のユーザー判断により、生成が成立して照合対象の配置と演算が確認できる場合、回答の限定的な語句の差は許容する。1位置と2位置で全層の演算や生成結果の完全一致は完了条件としない。

## 完了条件

常駐ヒット、同期ミス、スクラッチ再利用、昇格・退避・再読込を再現できる。出力とスロットの内容が参照と一致し、Flash-Next の短い混在生成が成功する。

## Expert直接DMA原本のスワップ対策

`8aa09a859` から `codex/rocm-expert-swap-fix` を分岐。固定3幅・直接DMAの停滞中に採取した `smaps` では、約19.56 GiBの匿名領域6個にプロセスのスワップの96.6%（5.08 GiB）が集中した。領域の合計は過去のExpert原本のHIP確保量と5,632 B差。割り当てアドレスの記録がないため、サイズによる照合。実行コンテナと親の `memory.max`、`memory.high`、`memory.swap.max` は `max` だった。

変更範囲はLinuxのExpert tuple DMA原本の確保箇所。既存HIPホストバッファの所有権とDMA経路を維持し、実バッファ単位でOSのページ固定を明示的に有効化できるようにする。アドレスと範囲をログに残し、固定を要求して失敗した場合は理由を示してロードを終了する。ホスト全体のスワップ・swappinessは操作しない。速度比較は中断中で、実モデルの再開前にソースと起動条件を確認する。
