# 異種GPU MoE推論エンジン 基本設計書

| 項目 | 内容 |
| --- | --- |
| 状態 | 設計レビュー用・Expert配置と先行転送の方針を反映 |
| 作成日 | 2026-11-13 |
| 更新日 | 2026-11-16 |
| ソース確認の基準 | [ggml-org/llama.cpp の 827466b9efa0574806bad7732bcd78bfb6d9311d](https://github.com/ggml-org/llama.cpp/tree/827466b9efa0574806bad7732bcd78bfb6d9311d) |
| 作業リポジトリ | [kamuiun-source/llama.cpp](https://github.com/kamuiun-source/llama.cpp) |

## 1. 目的と対象

llama.cpp / GGUFを土台に、1台のPCでGPUごとの得意な演算を固定し、大型MoEを動かす。最終構成ではRTX 4070 TiがAttention・Router・MTPを担当し、MI50 32 GBを2〜3枚使って本体のRouted Expertを処理する。各Routed Expertの重みは、MI50のVRAMとシステムRAMのどちらか一方に常駐させる。RAM側のExpertは選択されたときにMI50へ一時転送する。Qwen3.8-Flash-NextのPLEテーブルには既存の遅延読み込みを使い、Expert重みのNVMe退避は将来RAMが不足した場合に評価する。

初期検証はWindows 11、Ryzen 5 7600、RTX 4070 Ti 12 GB（CUDA）、MI50 32 GB（V420 VBIOS、Vulkan）、X870 Taichi Creator、850 W、RAM 32 GBの構成で行う。MI50はVulkanでAMD Radeon Pro VIIとして認識され、Qwen3.6-35B-A3Bの派生GGUFを全載せして生成できたとユーザーが確認済み。35Bは混在実行の障害を切り分けるときに使う。最終構成はLinuxを候補とし、MI50側のVulkanとHIP/ROCmは実機結果で選ぶ。

初回の対象はUnslothのQwen3.8-Flash-Next `UD-IQ3_XXS`。公式構成は48層、512 Expertから10 Routed＋1 Sharedを利用し、125B本体に51Bのn-gram embeddingと4BのMTPが加わる。初回は2,048トークンのコンテキスト、テキストのみ、単一セッション、MTP無効で、本体の分割ロードと短文生成の基準を取る。RTXに収容できる場合は、短文生成後に同じ量子化・配置でMTPを有効にして比較する。公開事例の速度は比較資料として扱う。 [Qwen公式モデルカード](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

配布の4シャードは同じ版でそろえ、`00001-of-00004.gguf` を本体として指定する。PLEだけが入った第4シャードはNVMe上に置き、`--lazy-mode on` で必要な行を読む。量子化の比較はPLEをNVMe、Routed ExpertをMI50に29.2 GiB置く前提の容量試算。残りのExpert重みはIQ3_XXSで17.64 GiB、Q3_K_XLで23.12 GiB、IQ4_XSで27.04 GiB。実際のテンソル配置、作業バッファ、OS使用分、PLEのページキャッシュを含むピークは実機で確認する。 [UD-IQ3_XXSの配布版](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/3f3ba9cb826b49c665510ce1702fd513a393ce39/UD-IQ3_XXS)

## 2. 基本構成

| 対象 | Flash-Next初回と固定配置の目標 | 追加検証 |
| --- | --- | --- |
| Attention、Gated DeltaNet、Router、Shared Expert、残差加算、出力層 | RTX。Flash-NextのQSA、KVキャッシュと再帰状態も原則RTX | MTPを加え、RTX 12 GB内のKV・作業領域を予算化 |
| Routed Expertの演算 | 固定分業ではPrefillとDecodeともMI50が担当し、MI50内で重み付き部分和を作る | 複数MI50で担当Expertを並行処理し、RTXで部分和を合算 |
| 入力Embedding | upstreamの既定であるCPUを初期値とし、容量と効果に応じてRTX配置を検討 | PLEは必要な行だけ読む。Embedding、出力ヘッド、MTPを個別に予算化 |
| Routed Expertの重み | 初回は既存オプションでテンソル単位にMI50とRAMへ配置し、ロードと生成を確認。35Bでの切り分けにはMI50へ全載せできたモデルを使う | Expert ID単位で、頻出分をMI50、残りをRAMに重複なく常駐させる。MI50にはRAM側の選択済みExpertを一時コピーする入れ替え枠を予約 |
| MTP | RTXに収容できる場合に既存の `draft-mtp` を使い、Decodeの候補トークンを生成 | MTP有効時の検証バッチで、Expertの選択数、転送量と待ち時間を確認 |

### Routerで選んだExpertの先行転送

単一プロセス内のggmlグラフを使う。層とExpert IDから、常駐先と配置先内のIDを引けるようにする。MI50に置くExpertには実行用の恒久的なRAMコピーを作らない。RAM側のExpertはRAMの重みを転送元として保持し、一時コピーだけをMI50の入れ替え枠に置く。重みは読み取り専用なので、一時コピーを破棄するときのRAMへの書き戻しは不要。

1. MTPを有効にしたDecodeでは、RTXのMTPが候補トークンを作り、本体が検証バッチにまとめる。本体がどのExpertを使うかは、各層でRTXのRouterが本体の中間状態から決める。MTP無効時とPrefillにも、以降の層内処理を適用する。 [検証バッチへの追加](../../tools/server/server-context.cpp#L636) · [RouterのExpert選択](../../src/llama-graph.cpp#L2370)
2. Routerが選んだIDを常駐先で振り分ける。RAM側の選択IDをバッチ内で重複排除し、MI50の入れ替え枠に有効なコピーがあるIDは再利用する。不足する重みだけを空いている枠に転送し、Activationと必要なID・係数もggmlでMI50へ渡す。
3. 不足分の転送と、MI50上で既に使えるExpertの計算を重ねる。転送したExpertは、コピー完了後に枠を参照して計算する。枠を再利用するときは、先行するGPU演算がその枠を読み終えたことを確認する。実際に重ねられる時間はWindows＋MI50 Vulkanで計測する。
4. MI50で各Expertの出力から重み付き部分和を作り、RTXへ返す。RTXはShared Expertと残差を合算して次層へ進む。複数MI50では、その層で処理があるGPUだけに配信し、RTXで各GPUの部分和を集約する。

Routerの選択IDと係数を維持し、各Expertの量子化ブロックとgate/up/downの対応を保つ。GGUFは既存形式で読み込む。MI50のメモリ予算には、常駐重みに加えて入れ替え枠と計算用バッファを含める。

デバイスの列挙順は起動ログで確認する。RTXのVulkan側の重複やiGPUの混入を避け、バックエンド名、GPU名、取得できる物理ID、利用可能なメモリを記録する。指定先が未検出、演算非対応、割り当て不能、または実行先が意図と異なる場合は、該当するデバイス・演算・量子化型を報告し、固定配置の検証を失敗とする。各カードの常駐重み、KV・再帰状態、作業領域、入れ替え枠を見積もる。RAMは常駐Expert重み、転送バッファ、ファイルのページキャッシュを区別して測る。

RTXの空きVRAMが不足する場合は、画面出力をRyzenの内蔵GPUへ移してRTXの割り当て予算を再確認する。内蔵GPUは画面出力用とし、推論デバイスには含めない。RAM共有の影響もあるため、変更時はRTXの空きVRAMとシステムRAMを両方記録する。

## 3. 既存実装と変更の境界

### 既存の起動オプション

| 目的 | オプションと適用範囲 |
| --- | --- |
| RTXとMI50を選ぶ | `--list-devices` で実機の名前を調べ、`-dev` (`--device`) でRTXのCUDAデバイスとMI50のVulkanデバイスを指定 |
| 本体の重みを配置する | `-ot` (`--override-tensor`) で `テンソル名パターン=バッファ型` を指定。単位はテンソル全体。演算先はグラフの配置ログで確認 |
| PLEをNVMeから読む | `--lazy-mode on` で対象テンソルを必要な行だけ読む |
| RAM側の重みを固定する | `--load-mode mlock`。基準コードは `--lazy-mode on` の対象を別途マップし、固定対象から外す。Windowsでは `VirtualLock` の成否とRAM使用量を起動ログで確認 |
| MTPをRTXで試す | `-md` でMTP用GGUF、`--spec-type draft-mtp` で方式、`--spec-draft-device` でRTXのCUDAデバイス、`--spec-draft-ngl all` でドラフト層のGPU配置を指定。初回は `--spec-draft-n-max 2` を候補とする |
| 既存のMoEキャッシュを比較に使う | `--moe-cache-mib N` で容量を指定。MI50での処理と比較する場合は `--moe-cache-cpu-share 0`。現行キャッシュは層のGPUにRAMの重みをコピーする |

[CLIオプション](../../tools/server/README.md#common-params) · [MTPオプション](../../docs/speculative.md#command-line-options) · [遅延対象のマッピング](../../src/llama-model-loader.cpp#L1505) · [遅延対象の固定除外](../../src/llama-model-loader.cpp#L1821) · [WindowsのRAM固定](../../src/llama-mmap.cpp#L979) · [MoEキャッシュの仕様](../../docs/moe-cache.md)

### 改造が必要な箇所と再利用先

| 改造箇所 | 方針と再利用するコード |
| --- | --- |
| Expert ID単位の常駐配置 | GGUFを読み込む際、層ごとにExpertの重みをMI50常駐分とRAM常駐分に分け、元IDと配置先の索引を持つ。gate/up/downと量子化ブロックの対応を保つ。候補は [llama-model-loader.cpp](../../src/llama-model-loader.cpp) と [llama-model.cpp](../../src/llama-model.cpp) |
| Expert実行先と入れ替え枠 | 現行キャッシュは `model.dev_layer(il)` でデバイスを選ぶ。Expert側のデバイス選択を独立させ、本体のAttention・RouterがRTXにある層でもMI50に枠を確保できるようにする。モデルのグラフで常駐先と枠のIDを参照し、RAM側から転送した分もMI50で実行する。候補は [llama-moe-cache.cpp](../../src/llama-moe-cache.cpp#L427)、[qwen4exp.cpp](../../src/models/qwen4exp.cpp#L1215)、[llama-graph.cpp](../../src/llama-graph.cpp) |
| Router確定後の転送と同期 | 現行キャッシュの選択ID取得、枠の管理、`ggml_backend_tensor_set_async()` を流用する。常駐分の計算とRAM側の不足分を独立して発行し、転送完了待ちを該当するExpertの実行前に置く。既存のVulkan転送経路で重ねられるかを確認し、待ちが残る場合にグラフ境界やバックエンド側を見直す。 [キャッシュの転送処理](../../src/llama-moe-cache.cpp#L791) · [Vulkanの非同期アップロード](../../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L13782) |

### 共通基盤の再利用

| 項目 | 基準コミットで確認した箇所と方針 |
| --- | --- |
| CUDA＋Vulkanの共存 | [docs/build.md](../../docs/build.md) に同時ビルドとデバイス選択が記載されている。[src/llama.cpp](../../src/llama.cpp) の `llama_prepare_model_devices()` と [common/arg.cpp](../../common/arg.cpp) のデバイス選択・`parse_tensor_buffer_overrides()` を使う。指定Windows環境での動作はPhase 0で確認 |
| GGUFと重み配置 | [src/llama-model-loader.cpp](../../src/llama-model-loader.cpp) のコンストラクター、`create_tensor()`、`load_all_data()` と [src/llama-model.cpp](../../src/llama-model.cpp) のレイヤー配置を利用する |
| MoEグラフと演算固定 | Flash-Nextは [qwen4exp.cpp](../../src/models/qwen4exp.cpp) のグラフを使う。35Bで切り分ける場合は [qwen35moe.cpp](../../src/models/qwen35moe.cpp) の `graph::build_layer_ffn()` を参照する。共通の [llama-graph.cpp](../../src/llama-graph.cpp) の `build_moe_ffn()`、必要なら [llama-context.cpp](../../src/llama-context.cpp) の `graph_get_cb()` と `ggml_backend_sched_set_tensor_backend()` を利用する。実行先は重み配置ログと区別して確認 |
| 転送と同期 | [ggml-backend.cpp](../../ggml/src/ggml-backend.cpp) の `ggml_backend_sched_split_graph()` と `ggml_backend_tensor_copy()` を再利用する。デバイスバッファ間でコピー用のフックが使えない場合、汎用経路はテンソルごとに一時RAMを確保し、GPU→RAM→GPUの2区間を経由する。スケジューラーは入力の生成とコピー先の再利用を同期する |
| Vulkan Expert | [ggml-vulkan.cpp](../../ggml/src/ggml-vulkan/ggml-vulkan.cpp) の `ggml_vk_mul_mat_id()` と `ggml_backend_vk_device_supports_op()` を使う。対象GGUFの各Expertテンソルの型と、Prefill／Decodeの形状を実機で確認する |
| PLEとSSD | [qwen4exp.cpp](../../src/models/qwen4exp.cpp) の `TENSOR_READ_LAZY` と `llama_prefetch_rows()` を評価する。Expert専用のSSD読み込みは既存のmmapによるページ読み込みと区別し、専用経路が必要になった段階でupstreamの進捗と既存フォークを再評価する。 [ExpertストリーミングPR](https://github.com/ggml-org/llama.cpp/pull/25294) |

Phase 1の固定配置MVPは、受入条件を満たせば既存CLIの再現可能な設定と検証記録で成立する。Expert ID単位の常駐配置と選択後の入れ替えはPhase 3で扱う。必要なコード変更はモデルのロード、MoEグラフ、既存キャッシュの接点を起点にする。外部設定ツールは実行構成の生成やログ整理に使える。

比較調査は llama.cpp、FreeToken、Strata、ninfer-all、Infernixを対象とし、URL・SHA・ライセンス・ソースパス・対応範囲は詳細調査で確定する。特にStrata本家の補助GPUと `alizand1992/Strata` の共有RAM方式は設計の比較対象とする。後者の公開実測はLinux CUDA＋HIP、RTX 5060 Ti＋RX 9060 XTで、Windows CUDA＋Vulkan／MI50への適用性は未検証。 [Strataの補助GPU設計](https://github.com/Niko1221/Strata/blob/main/docs/SECOND_GPU.md) · [異種GPUの実装](https://github.com/alizand1992/Strata/blob/main/src/core/external_experts.cpp) · [作者の実測条件](https://github.com/Niko1221/Strata/issues/3127)

## 4. PhaseとIssue分割案

以下はIssue化するときの分割候補。Issueごとの依存関係、変更候補、受入条件は [実装計画](implementation-plan.md) に記載する。各Phaseの実行は本書の合意後に進める。

| Phase | Issue候補・変更範囲 | 最小確認・完了条件と移行判断 |
| --- | --- | --- |
| 0 実現可能性 | **0A** 同一ビルドでRTX CUDAとMI50 Vulkanの認識、空きVRAM・RAMを確認。**0B** 既存の設定でIQ3_XXSを2K・MTP無効・PLE遅延読み込みで分割ロードし、短文を生成。**0C** 容量・演算互換性の問題は動作確認済みの35Bで切り分ける。**0D** RTXに余裕があれば同じ量子化・配置でMTPを有効にして比較する | IQ3_XXSが短文生成でき、起動時のRTX・MI50・RAMの配置とPLEの遅延読み込み、ロード・生成時のメモリ使用量を記録できれば次へ進む。演算先の確認はPhase 1で行う |
| 1 固定配置MVP | **1A** 起動設定と配置ログでAttention・Router・Routed Expertの実行先を確認。**1B** 設定では固定できない演算を必要に応じて明示指定。候補は `src/models/qwen4exp.cpp`、`src/llama-context.cpp`、`src/llama-graph.cpp` | 意図したGPUでの実行と出力の妥当性を確認し、Prefill・Decode・転送時間の基準値を取得。速度改善は次へ進む条件とせず、待ち時間から次の改修価値を判断 |
| 2 通信・計測 | **2A** splitごとの転送バイト数・待ち・実行時間の計測。**2B** 計測上の主要因が確認されたとき、MI50側集約、転送バッファ再利用や重複可能なコピーを検討。候補は `ggml/src/ggml-backend.cpp` と各バックエンド、またはモデルのグラフ境界 | 変更前後の同条件で出力を確認し、どの待ちが減るかを特定。pinned RAMを使う場合はCUDA・Vulkan双方で利用できる条件と上書きの同期を実機で確認 |
| 3 RAM・大型MoE | **3A** Expert IDごとにMI50かRAMに常駐させ、重複を除く。**3B-1** Routerが選んだ不足分を入れ替え枠へ転送し、MI50での計算と枠の再利用を成立させる。**3B-2** その経路を基準に常駐分の計算と転送を重ねる。**3C** ルーティング履歴で常駐配置を比較する | 常駐先の重複がなく、常駐のみのバッチとRAM側のExpertを含むバッチの両方でMI50が処理することを確認する。出力、必要な重みだけ送ったこと、転送待ち、RAM使用量を記録。MTP有効時は確定トークンあたりのExpert転送量を比較する |
| 4 複数MI50・長文 | **4A** MI50×2を優先し、同一層でExpertを分けて部分和を戻す。**4B** トレースでGPU別の配置を評価し、必要なら3枚目・複製・再配置を追加。**4C** QSAの長文スケーリング、256K・画像入力を順に評価。**4D** RAM予算を超える場合に限りSSD Expertを評価。候補はグラフ、キャッシュ、ロード、計測の各接点 | 同じ重みと入力で1枚・2枚を比較し、負荷の偏り、転送、VRAM/RAM、TTFT・生成速度を記録。3枚目は収容余力または速度に効く根拠を確認して評価。長文は設定上限と実際に埋めた長さを記録し、SSDはI/O待ち込みで実用性を判定 |

### 共通の比較条件

- AはMI50単体Vulkan、BはRTX単体CUDA、Cは通常の混在レイヤー配置、DはRTXとMI50の層内分業。同一GGUFで収まる構成を比較し、別量子化の値は個別に記録する。初回の短文生成にはCまたは既存のテンソル指定を使う。
- 同じGGUF、入力、コンテキスト長、バッチとマイクロバッチ、KV形式、サンプリング条件を固定する。初回はテキスト・単一セッション・MTP無効。短い固定入力のトークン列を比較し、差が出た箇所は同じ入力トークン列を使ったlogitsと配置を確認する。GPU間の丸め差を考慮した許容値はPhase 0で合意する。
- PrefillとDecodeのtok/s、TTFT、配置ログ、転送回数・バイト数・待ち時間、GPU別演算時間、VRAM・RAMを必要な粒度で記録する。配置の確認と時間計測は分け、計測用ログの負荷を基準値に混ぜない。
- MTP有効時は検証バッチあたりの選択Expert数、RAMから転送したExpert数、確定トークン数を記録する。同じ本体配置でMTPを切り替え、確定トークンあたりのExpert転送バイト数とDecode速度を比較する。
- Expertログは層・トークンまたはマイクロバッチ・選択IDの組を保持する。将来は同じトレースで頻度順配置と同時に選ばれるExpertを分散した配置を比べ、キャッシュミスの転送量、各GPUの処理時間と層ごとの完了待ちを評価する。既存 `LLAMA_MOE_CACHE_TRACE` はキャッシュ経路のため、固定配置で記録が必要ならRouterの既存コールバック等を調べる。 [トレース再生コード](../../scripts/moe-cache-tune.py)

### テストの基本方針

- 受入項目ごとに変更の正常系を代表するケースを先に決める。テストは **1項目あたり最大4ケース**。その項目のために選ぶ単体・結合・実機での確認を合算し、既存テストや同じ実機実行のログで確認できる挙動はその結果を再利用する。
- 着手時に確認対象と完了条件を固定する。正常系で受入条件を確認でき、具体的な不具合の根拠がない場合、レアケースや条件の組み合わせは改善候補に記録する。通常利用で再現する問題や誤った出力・枠の上書きなど具体的な懸念が見つかった場合は、重要度を見てケースを入れ替え、上限内で確認する。
- 合意した確認が通ればレビューへ進む。確認済みの範囲、既知の問題、未確認の範囲を記録する。

## 5. リスクと実機での判定

| リスク | 発生条件と確認 | 対応候補 |
| --- | --- | --- |
| 固定配置のずれ | 重みは指定どおりでも、Routerや集約が別GPUへ移る。演算とバッファのログで確認 | 既存の明示配置APIを必要なノードにだけ使用 |
| ドライバー・量子化・GPU識別 | MI50の列挙・32 GB認識・演算可否、NVIDIAの二重列挙、再起動後の番号変化。実際のGGUFとドライバーで確認 | 実デバイスを確認して選択。Vulkanの既存対応型で初期検証。必要ならLinuxを別構成で比較 |
| 転送と同期 | 小さいデータでも層ごとにCUDA・Vulkanの完了を待つ。PrefillとDecodeを分けてsplit境界を計測 | MI50内で集約。計測で効果が見込める箇所からバッファ再利用や非同期化を比較 |
| 入れ替え枠の不足 | PrefillやMTPの検証バッチで、その層の選択済みRAM側Expertの集合が枠を超える。バッチごとの選択数と枠の使用数を記録 | 作業領域を含めて枠の容量を決める。必要な場合は、枠に収まる単位で順に処理して部分和を合算する案を評価 |
| VRAM・RAM・I/O | RTXの12 GBをKV等が圧迫、ホスト重複でRAM不足、SSDへのミスが頻発。ロード時・生成時のピークを確認 | 量子化と短いコンテキストで容量を成立させ、Expert常駐率とPLEの読み込みを別々に測定 |
| 複数MI50の負荷偏り | 同時に選ばれるExpertが同じGPUに偏る。実トレースとGPU別時間で確認 | 層・Expert単位の静的配置を再評価。2枚で効果を確認してから3枚へ |
| upstream追従 | 共通グラフ、スケジューラー、キャッシュの更新と競合。同期時に変更箇所を確認 | 設定で使える機能を優先し、専用文書は `fork-docs/`、製品差分は小さなPRにする |

## 6. 次に決めること

| 時期 | 決定事項 | 初版での提案 |
| --- | --- | --- |
| Phase 0 | 初回モデルと起動条件 | UD-IQ3_XXS、2K、テキストのみ、単一セッション、MTP無効、PLEはNVMe。ロード時と生成時の実測を残す |
| Phase 0の前 | MI50のVulkanドライバー名・版、現状の認識と動作、搭載RAM、PCIe接続 | すでに動いている構成を基準にして、実際のデバイス名・空き容量・リンク幅を記録する |
| 初回の短文生成後 | 量子化とMTP | メモリの余裕があればUD-Q3_K_XL、必要に応じてUD-IQ4_XSへ上げる。MTPは同じ量子化のまま別の比較で、非共有版 `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` をRTXに追加する |
| Phase 2〜3の前 | 大型で優先する性能と容量 | Decode・Prefill・メモリ制約の優先順位を実測後に決める。Flash-Next GGUFのExpert・Dense・PLE・MTP別バイト数で必要RAMとGPU枚数を算定 |
| Phase 4の前 | Linuxとバックエンド、PCIe、長文・画像の要件 | MI50×2を先に比較し、HIP/Vulkan、実リンク幅、3枚目の接続・電源・冷却を確定。256Kと画像入力は短文成立後に評価 |

本書は基本設計のレビュー用。詳細調査、各設計判断の固定SHA付き根拠、再現手順は `fork-docs/research/` と `fork-docs/design/` に展開する。まず既存設定でベースラインを取り、配置やI/Oのログで判明した制約に合わせてカスタマイズの範囲を決める。
