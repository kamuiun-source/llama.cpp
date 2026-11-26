# 検証設定とON/OFFフラグ

更新 2026-11-25。Vulkanの設定は高速3幅比較の `japanese/width-3/settings.json` に基づく。実験フラグはブランチによって追加・削除される。再実行前に対象のコード、ビルド、ランナーを確認する。

## Linux ROCm・固定3幅の採用設定

単一バンク・Expert直接DMAのDocker設定は [固定3幅の採用設定](../rocm/10-decode-pipeline.md#単一バンク固定3幅の採用設定)、機能別のON/OFFは [固定3幅の起動フラグ一覧](../rocm/10-decode-pipeline.md#固定3幅の起動フラグ一覧) を参照。移植を取り込んだ `codex/rocm-integrated-test` のMI50では、符号復号 `GGML_HIP_GFX906_IQ_SIGN_DECODE=1`、LDS先読み `GGML_HIP_GFX906_IQ3_XXS_LDS_PREFETCH=1`、3位置での出力4行 `GGML_HIP_GFX906_IQ3_XXS_LDS_ROW_GROUPS=1` を採用する。

### 現行の高速化一覧

固定3幅の統合版で使用する設定。表の環境変数は `1`、モードファイルの値は `on`。

| 対象 | 採用機能と設定 |
| --- | --- |
| MTP | 本体の固定3位置一括検証 `LLAMA_MTP_VERIFY_THREE`、確定位置の隠れ状態をRTX内で渡す `LLAMA_MTP_FIXED3_DEVICE_PENDING`、catch-upと第1候補の統合 `CATCHUP_FIRST_DRAFT=on` |
| CPU Expert | `SINGLE_MISS_CPU=on`。実ミスの一部をCPUで並列計算し、gate/upとdownの量子化済み入力を共有。残りはMI50で計算 |
| Expert配置・転送 | `LLAMA_EXPERT_HIP_DYNAMIC_RESIDENCY`、`LLAMA_EXPERT_HIP_TUPLE_DMA`、`LLAMA_EXPERT_HIP_PINNED_UPLOAD`、`LLAMA_EXPERT_HIP_MTP3_ASYNC_PROMOTION`、`LLAMA_EXPERT_HIP_MTP3_ASYNC_SLOT_MAP`。原本のページ固定はLinuxで自動適用 |
| Expert先読み・プリフィル | `LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH`、`LLAMA_EXPERT_HIP_PREFILL`。先読み8枠、実ミス優先、必要なヒットだけイベント待ち。対象層では現層のMI50投入後に次層Routerを発行 |
| RTXとMI50の受け渡し | `GGML_SCHED_HIP_BATCH_DEVICE_INPUTS`、`GGML_SCHED_HIP_DUAL_PINNED_HANDOFF`、`GGML_SCHED_HIP_ASYNC_INPUT`、`GGML_SCHED_HIP_ASYNC_RETURN`、`GGML_SCHED_HIP_HOST_ROUTER_IDS`。対象形状で共有RAM直接書き込みと非同期の戻りを選択 |
| RTX | Shared Expertの先行発行 `GGML_SCHED_HIP_SHARED_OVERLAP`。短いconcat `GGML_CUDA_CONCAT_SMALL`、SSMのsoftplusと係数乗算の融合 `GGML_CUDA_UNARY_MUL_BCAST`、GDNの正規化とゲートの融合 `GGML_CUDA_GDN_GATED_NORM` |
| MI50 | 符号復号 `GGML_HIP_GFX906_IQ_SIGN_DECODE`、IQ2_SのQ8_1入力共有 `GGML_HIP_GFX906_IQ2_S_PAIRED_DOT`、IQ3_XXSのLDS先読み・4行 `GGML_HIP_GFX906_IQ3_XXS_LDS_PREFETCH` と `GGML_HIP_GFX906_IQ3_XXS_LDS_ROW_GROUPS`、IQ4_NL downの復号短縮 `GGML_HIP_GFX906_IQ4_NL_DOWN_COMPACT_DECODE` |

### 出力の揺れを調べるON設定

- **CPU Expertと次層先読みの併用。** CPUとMI50でExpert計算の量子化・内積経路が異なる。第1層の固定入力のIQ4_NL downで微小な数値差は観測済み。CPU候補は、常駐・仮置きに加え、先読みの転送を**発行済み**かで決まる。発行済みの転送は未完了でも先読み扱い。完了イベントは次に発行する候補を進めるので、発行済み集合を通じてCPUとMI50への配分を変え得る。今回の分岐との因果は未確認。[固定入力の比較](../rocm/03-expert-placement.md) · [転送キューの進行](../../src/llama-moe-cache.cpp#L2602) · [CPU配分](../../src/llama-moe-cache.cpp#L3598)
- **非同期プリフィルと動的常駐。** Expertの常駐・仮置き状態でプリフィルのグループが分かれ、グループの幅で量子化行列演算のカーネル選択が変わり得る。別設定の保存済み診断では最初のプリフィル出力時点からスコア差があった。今回の同一ビルド再実行には層別の配分・カーネルの記録がない。[プリフィルのグループ化](../../src/llama-moe-cache.cpp#L2075) · [カーネル選択](../../ggml/src/ggml-cuda/ggml-cuda.cu#L2607)
- **固定形状のカーネル改善。** IQ2_S入力共有は積和順を維持し、IQ3_XXSのLDS・4行とIQ4_NL復号短縮は固定入力の単体比較でビット一致を確認済み。今回の実行間の揺れを切り分ける優先度はCPU配分・プリフィルより低い。[MI50の比較記録](../rocm/10-decode-pipeline.md#固定3幅の起動フラグ一覧)

統合版のJSONLは設定切替・リクエスト境界・メモリ統計を記録する。応答JSONにある第1問16トークン目の `106806` / `159212` の分岐はcatch-upのON同士・OFF同士で再現した。原因の切り分けは同じ接頭辞でCPU Expertだけ変更し、競合する2語の本体logit、実際のExpert計算先、必要ならプリフィルのグループ幅を照合する。通常の `CATCHUP_FIRST_DRAFT=on` は維持。[同一ビルド再実行](../rocm/latest-performance.md#同じ統合ビルドの再実行) · [catch-up統合OFF](../rocm/latest-performance.md#同じ統合ビルドでcatch-up統合をoff)

2026-11-24に、LDS先読みを通常起動で使う型はIQ3_XXSのみと確定した。IQ2_Sの先読みと4行は通常起動でOFFとし、他の改善を進める中でIQ2_Sがボトルネックになった場合に再調査する。[ON/OFFの実測](../rocm/10-decode-pipeline.md#両型lds先読みoffの追加2回)

IQ2_SのLDSは独立した試験設定。`GGML_HIP_GFX906_IQ2_S_LDS_PREFETCH=1` で有効になり、`GGML_HIP_GFX906_IQ2_S_LDS_ROW_GROUPS=1` も指定すると4行になる。両変数の既定はOFF。IQ2_Sだけで試す場合はIQ3_XXSのLDSフラグをOFFにする。

`GGML_HIP_GFX906_IQ2_S_PAIRED_DOT=1` はIQ2_Sの固定3位置・LDSなし・2行のQ8_1入力共有。統合ブランチの通常起動で採用する。比較時は `0`。符号復号をON、IQ2_SのLDS先読みとLDSなし4行をOFFにする。[固定5問の比較](../rocm/10-decode-pipeline.md#q8_1共有を有効にした実モデルの固定5問)

`GGML_HIP_GFX906_IQ4_NL_DOWN_COMPACT_DECODE=1` はgfx906・実3位置・2行のIQ4_NL downの復号セレクタ短縮。統合ブランチの通常起動で採用する。個別比較では `0` を指定する。[単体計測](../rocm/10-decode-pipeline.md#iq4_nl-downの復号セレクタを短縮する試作)、[固定5問の再比較](../rocm/10-decode-pipeline.md#復号短縮だけをonoffonoffで再比較)

RTXの固定3位置・Q8_0 HC down Split-Kは通常起動でOFF（未設定または `0`）。比較時は `GGML_CUDA_HC_DOWN_SPLIT_K=1` で有効化する。4分割した部分和のFP32加算順が変わり、丸め差で温度0でも出力文言・MTP採用数が変動し得る。[新しい5問の実測](../rocm/10-decode-pipeline.md#新しい独立5問でsplit-kを再比較)

`codex/rocm-rtx-mapped-input` では、固定3幅のRTX→MI50活性値を `GGML_SCHED_HIP_ASYNC_INPUT=1` と共有ピン留め・一括転送の設定で自動的に共有RAMへ書き込む。試作専用の `GGML_SCHED_HIP_MAPPED_INPUT` は廃止。[試作と実測](../rocm/10-decode-pipeline.md#共有ramへのrtx側直接書き込み)

`codex/rocm-integrated-test` の通常起動では、既存の共有ピン留め・非同期入力と戻りに `GGML_SCHED_HIP_SHARED_OVERLAP=1` を追加する。対応する固定3幅では、Shared Expert先行発行を使う層でもMI50カーネルによる共有RAM読み込みとRTXへの非同期H2Dが自動適用される。Router先行D2Hは廃止。試作専用フラグの名前と成績は計測メモに残す。[通常起動への追加](../rocm/10-decode-pipeline.md#共有ram読み込みを含む固定3幅の通常起動) · [4条件の結果](../rocm/10-decode-pipeline.md#routerの先行d2hと戻り側h2dの非同期化)

既存の非同期HIP入力を使う対象の固定3幅では、次層Router予測を現層MI50のExpertグラフ投入後に自動で発行する。試作フラグは廃止した。[追加2回の実測](../rocm/10-decode-pipeline.md#次層routerのon追加計測と採用)

## Windows Vulkan・高速固定3幅の採用設定

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

## ROCm統合ブランチの固定3幅

`codex/rocm-expert-single-kernel-overlap` から採用した次層先読みは `LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH=1` で有効にする。共有8枠、必要なヒットだけのイベント待ち、現在層の投入後に先読みを発行する順序、実ミス優先が含まれる。単一バンク・固定3幅・直接tuple DMAの下で、非同期プリフィル `LLAMA_EXPERT_HIP_PREFILL=1` と非同期昇格 `LLAMA_EXPERT_HIP_MTP3_ASYNC_PROMOTION=1` を併用する。[Dockerの正式採用設定](../rocm/10-decode-pipeline.md#単一バンク固定3幅の採用設定)

通常はヒット待ち、後発行、スコアフィルター、必要転送後のホスト同期省略、詳細計測の各フラグを未設定にする。GPU間のストリーム通知は単体試験として記録している。[先読み併用時の計測](../rocm/09-single-bank-mtp3.md#非同期プリフィルと非同期昇格の併用) · [GPU間通知の単体試験](../rocm/10-decode-pipeline.md#mi50からrtxへのストリーム通知の単体試験)

### 次層Expert先読みの比較フラグ

| 設定 | 値 | 動作 |
| --- | --- | --- |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_HIT_WAIT` | 未設定（通常） / `0`（比較用） | イベント待ちを切り分ける旧比較設定。通常起動で設定する必要はない。 |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_PREFETCH_AFTER_SUBMIT` | 未設定（通常） / `0`（比較用） | 現在層の投入後発行を切り分ける旧比較設定。通常起動で設定する必要はない。 |
| `LLAMA_EXPERT_HIP_NEXT_ROUTER_SCORE_FILTER` | `0`（既定） / `1`・`2`（比較用） | `1` は予測ロジットの閾値、`2` はTop-20のスコア順で既存Top-10と同じ件数（最大8）を発行。 |
| `LLAMA_EXPERT_HIP_MTP3_ASYNC_DEMAND` | 未設定（既定OFF） / `1`（試作） | 固定3幅の必要転送から融合までを計算ストリーム上で連結する。既存の先読み後発行・実ミス優先、非同期スロット表と非同期昇格が有効な構成で使う。 |
| `LLAMA_EXPERT_HIP_MTP3_DEMAND_TRACE` | 未設定（通常） / `1`（詳細比較） | 試作ON/OFF両方に層ごとのホスト時刻を出す。GPU時刻はrocprofで照合する。 |

同層でExpert転送を隠す固定3幅の試作は専用フラグごと削除した。次層先読みには単一バンクとROCmの直接tuple DMAが必要。従来の分割配置を使う構成は対象外。

### Packed ExpertのCPU並列計算

`codex/rocm-integrated-test` の固定3幅の通常起動では `LLAMA_EXPERT_HIP_MTP3_SINGLE_MISS_CPU_MODE_FILE` に `SINGLE_MISS_CPU=on` を入れたモードファイルを渡す。比較時は `off` にし、リクエスト間で切り替える。変数の未指定時はコード既定OFF。固定3幅・単一バンク・直接tuple DMA・非同期の戻り転送が対象。ONではgate/upとdownの量子化共有を常に使う。使用可能な物理コアから2コア分を残し、実ミス1件なら最大4ワーカー、複数件なら1件あたり2ワーカーを使う。残りは通常のMI50転送。旧試作モードの計測結果は記録に残っている。[通常起動](../rocm/10-decode-pipeline.md#単一バンク固定3幅の採用設定) · [自動配分](../rocm/09-single-bank-mtp3.md#cpu-expert計算のonoff統合と物理コア数による自動配分)

`--moe-cache-cpu-share F` は通常のMoEキャッシュが満杯になった後、ミスのうちCPUで計算する割合を指定する。packed Expert経路の条件は `F=0`（既定）で、正の値は初期化時に拒否される。

### 固定3幅のMTP catch-upと第1候補

| フラグ | 通常起動 | 説明 |
| --- | --- | --- |
| `LLAMA_MTP_CATCHUP_FIRST_DRAFT_MODE_FILE` | `CATCHUP_FIRST_DRAFT=on` を1行で記載したファイルを指定 | 固定3幅・greedyの2候補で、採用済み履歴の反映と次の第1候補の生成をまとめる。`CATCHUP_FIRST_DRAFT=off` でリクエスト間に切り替え可能。ファイル未指定時のコード既定はOFF。**同じ入力・温度0でも、ON/OFFで生成トークン、回答の文面や長さ、MTP採用数が変わり得る。** |
| `LLAMA_MTP_FIXED3_DEVICE_PENDING` | `1` | 上記の前提。直前の確定位置の隠れ状態をRTX上で受け渡す。ON/OFF比較でも両方を `1` に揃える。 |
| `LLAMA_MTP_CATCHUP_FIRST_DRAFT_DIAGNOSTICS` | 未設定 | `1` で検証後のドラフトKV位置と一時行の削除位置を出す。 |

会話のKVを再利用した上限512の比較では、生成57.55→60.20 tok/s、第5問は全4回で自然に完結した。バッチ形状に伴う演算順序や丸めの違いは出力差の候補。出力が分かれた位置のlogitは比較しておらず、原因は未確定。前回の途中EOSも再現しなかった。[通常起動](../rocm/10-decode-pipeline.md#mtpのcatch-upと第1候補をまとめる設定) · [比較結果](../research/mtp3-efficiency-20261125.md#会話kvを維持した512トークン上限の再計測)
