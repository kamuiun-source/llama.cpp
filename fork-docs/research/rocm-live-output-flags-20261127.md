# 固定3幅の出力比較と起動中のフラグ切り替え

`codex/rocm-integrated-test` の `c301381e5` から `codex/rocm-live-output-flags` を作成。対象は[出力の揺れを調べたときの設定](../teams/runtime-flags.md#出力の揺れを調べるon設定)。モデル読み込み時のRTX・MI50・RAMの配置、直接tuple DMAと動的常駐など、固定3幅に必要な基盤は共通とする。

[固定5問の第4問](dialogue-benchmark-prompts.md#4-コード生成)だけを、毎回新規会話にして同じサーバーへ送る。4K・生成上限512・温度0・seed 1234。各リクエスト前にスロットを消去し、会話KVとRAMプロンプトキャッシュの再利用を無効にする。Pythonコードが計算する60 ms、27 ms、55%と、日本語の説明文に書かれた値を別々に確認する。本文・生成ID・MTP採用数・処理の発動を記録する。

## 比較条件

| 試す機能 | 共通の条件と比較元 |
| --- | --- |
| CPU Expert自動配分 | 先読みOFFでCPU配分だけON。先読みONの対照でもCPU配分だけを変え、両者の併用時の影響を見る。 |
| 次層Router先読み | 固定3幅で先読みの有無を比較。配置・候補数は同じ。 |
| HIPプリフィル | 同じ配置とバッチ上限で比較し、選ばれたプリフィルの経路を確認する。 |
| MI50符号復号 | 他の対象カーネル機能をOFFにし、符号復号だけを比較。 |
| IQ2_SのQ8_1入力共有 | 符号復号ONを共通にして比較。IQ2_SのLDS先読み・4行はOFF。 |
| IQ3_XXSのLDS先読み・4行 | 符号復号を有効にして先読みだけを比較。4行は先読みON・2行を対照にする。 |
| IQ4_NL downの復号短縮 | 固定3幅で復号短縮だけを比較。 |
| catch-up＋第1候補の統合 | 固定3幅・2候補で既存のモードファイルを使う。 |
| RTXのHC down Split-K、GDN、concat、unary | それぞれ1つずつONにし、全OFFを対照にする。 |
| IQ2_S・IQ3_XXS Strata | 既存のモードファイルで型別に切り替える。IQ2_Sは符号復号とQ8_1入力共有をONにして比較する。 |
| Router予測IDの読み戻し | 次層先読みをONにしたまま読み戻しだけを切り替える。 |

最初の全OFFと最後の全OFFを照合する。会話KVは消去するが、MI50の常駐Expertはサーバー内で入れ替わる。比較結果では、設定差に加えて実行順とExpertの状態の影響も考慮する。

## 起動中に切り替える設定

- `LLAMA_GPU_OUTPUT_MODE_FILE` はGPUカーネルの既存の環境変数名をキーとし、1行に1つ `名前=on` または `名前=off` を書く。未記載の項目は起動時の設定に従う。
- `LLAMA_EXPERT_OUTPUT_MODE_FILE` には `NEXT_ROUTER_PREFETCH=on/off` と `HIP_PREFILL=on/off` を書く。必要な作業領域を起動時に確保するため、Docker起動時にファイルのパスを指定する。
- CPU Expert、catch-up＋第1候補、IQ2_S・IQ3_XXS Strata、次層Router ID読み戻しは[既存のモードファイル](../teams/runtime-flags.md#linux-rocm固定3幅の採用設定)で切り替える。

Dockerには個別ファイルの代わりにモードファイルを置くディレクトリをマウントする。待機中にファイルを同じディレクトリ内の一時ファイルから置き換え、次のリクエストで反映する。今回追加したモードを切り替えると、以前の設定で計算したプロンプトを再利用しないよう、対象・ドラフトのKVとサーバーのプロンプトキャッシュを消去する。モデルの配置と重みはそのまま使う。

### GPUモードファイルのキー

| 対象 | キー |
| --- | --- |
| MI50の符号復号・Q8_1共有・down復号 | `GGML_HIP_GFX906_IQ_SIGN_DECODE`、`GGML_HIP_GFX906_IQ2_S_PAIRED_DOT`、`GGML_HIP_GFX906_IQ4_NL_DOWN_COMPACT_DECODE` |
| IQ3_XXSのLDSと行数 | `GGML_HIP_GFX906_IQ3_XXS_LDS_PREFETCH`、`GGML_HIP_GFX906_IQ3_XXS_LDS_ROW_GROUPS` |
| IQ2_SのLDSと行数 | `GGML_HIP_GFX906_IQ2_S_LDS_PREFETCH`、`GGML_HIP_GFX906_IQ2_S_LDS_ROW_GROUPS`、`GGML_HIP_GFX906_IQ2_S_NO_LDS_ROW_GROUPS` |
| RTXのHC down・GDN | `GGML_CUDA_HC_DOWN_SPLIT_K`、`GGML_CUDA_GDN_GATED_NORM` |
| CUDA/HIP共通 | `GGML_CUDA_CONCAT_SMALL`、`GGML_CUDA_UNARY_MUL_BCAST` |

行数を選ぶキーは `off` で2行、`on` で4行。IQ3_XXSのLDSと行数のキーは従来のカーネルに適用される。IQ3_XXS Strataを使う場合はそちらの専用カーネルが優先される。

既存のDocker起動コマンドにモード用ディレクトリを追加し、その中の2ファイルを起動時に指定する。例ではホスト側の `$modes` に `gpu.conf` と `expert.conf` を置く。

```bash
--mount "type=bind,src=$modes,dst=/runtime-modes,readonly" \
-e LLAMA_GPU_OUTPUT_MODE_FILE=/runtime-modes/gpu.conf \
-e LLAMA_EXPERT_OUTPUT_MODE_FILE=/runtime-modes/expert.conf
```

たとえば符号復号だけを明示的にOFFにするには、ホスト側で次を実行してから、新規リクエストを送る。この例から省いたキーは起動時設定に戻る。

```bash
printf '%s\n' 'GGML_HIP_GFX906_IQ_SIGN_DECODE=off' > "$modes/gpu.conf.tmp"
mv "$modes/gpu.conf.tmp" "$modes/gpu.conf"
```

## 第4問の実測

コミット `a85324332` をCUDA＋HIPで増分ビルドし、サーバー1プロセスで第4問を19回実行した。全件でプロンプト133トークン、KVキャッシュ再利用0、生成146トークン、EOSで終了。出力本文とトークンID列は全件一致し、MTPは各回97/98採用。別計測の素のllama.cppと静的配置のフォーク版が同じ第4問に返した146 IDとも一致した。

どの条件でもPythonコードは改善前60.00 ms、改善後27.00 ms、改善率55.00%を表示する。説明文は付かなかった。以前5問を連続で回した際に、第4問の説明文だけに誤った数値が出た現象は、この単独の第4問では再現しなかった。

| 順 | 有効にした機能 | 生成 tok/s |
| --- | --- | ---: |
| 00 | 全OFF | 42.35 |
| 01 | CPU Expert | 64.13 |
| 02 | 次層Router先読み | 57.07 |
| 03 | 次層Router先読み＋CPU Expert | 68.07 |
| 04 | HIPプリフィル | 59.31 |
| 05 | MI50の符号復号 | 65.03 |
| 06 | 符号復号＋IQ2_SのQ8_1入力共有 | 66.10 |
| 07 | 符号復号＋IQ3_XXSのLDS先読み・2行 | 63.25 |
| 08 | 符号復号＋IQ3_XXSのLDS先読み・4行 | 62.98 |
| 09 | IQ4_NL downの復号短縮 | 63.31 |
| 10 | catch-up＋第1候補の統合 | 63.33 |
| 11 | RTX HC down Split-K | 62.93 |
| 12 | RTX GDN正規化＋ゲート融合 | 63.54 |
| 13 | 短いconcat | 63.60 |
| 14 | unaryと係数乗算の融合 | 61.47 |
| 15 | IQ3_XXS Strata | 65.47 |
| 16 | 符号復号＋Q8_1入力共有＋IQ2_S Strata | 66.80 |
| 17 | 次層Router先読み＋予測IDのピン留め読み戻し | 66.62 |
| 99 | 全OFFに戻す | 62.66 |

開始と終了の全OFFで42.35と62.66 tok/sに分かれた。この計測は出力の切り分けに使い、個々の速度への効果は同じ実行順では判定しない。初回から最後までモデルを再ロードせず、常駐ExpertとOSのファイルキャッシュは持ち越した。

ログでCPU Expert割り当て、Router先読み、HIPプリフィルの実行を確認した。HIPプリフィルをOFFに戻す時点でcompact chunk 316、転送124回だった。MI50では符号復号、IQ2_S入力共有、IQ4_NL復号短縮、両型のStrata、RTXではHC down Split-Kの適用ログが残った。LDSと行数、RTXのGDN・concat・unaryはモード反映を確認した。カーネルトレースは採取していない。予測ID読み戻しも詳細計測を無効にしていたため、モード反映のみ確認した。

応答、ID、モードファイルのスナップショット、サーバーログ、実行スクリプトは `/home/t-saka/rocm-live-output-flags-q4-20261127` に保存。サーバーは終了コード0で停止し、コンテナを削除した。
