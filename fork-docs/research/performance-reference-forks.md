# 推論高速化の参考実装と採用判断

調査日: 2026-11-16。現段階は候補調査。外部実装は未導入。参照ブランチは可変であり、取り込み時はコミットIDとライセンスを固定して再確認する。

## 今回の進め方

現在のExpert転送改善を先にレビューし、ユーザーが実機で比較する。その結果から次の改善を一つずつ選ぶ。外部フォークへの移行や複数改善の一括導入は決定していない。ビルドと実機実行はユーザーが行う。

## 参考候補

### dylanatl15/llama.cpp-lab (lab)

- リポジトリ: https://github.com/dylanatl15/llama.cpp-lab/tree/lab
- 実装確認: https://github.com/dylanatl15/llama.cpp-lab/blob/lab/src/llama-moe-cache.cpp
- 測定・試行記録: https://github.com/dylanatl15/engine-lab
- READMEはRadeon VII 16GB、RAM 64GB、Flash-Next IQ3_XXS、Linux Vulkan/RADVで10.5から約27 tok/sと報告。こちらでの再現結果ではない。companion READMEには別の比較値もあり、測定版・条件を混同しない。
- ソースで確認したもの: cpy_poolによるCPUコピー並列化、get_mailbox/get_pred_mailboxによるホスト通知領域、LRUキャッシュ管理。
- README上の追加改善: HC_MIX/MOE_COMBINE融合、量子化matvecの調整、Decodeのコマンド記録・送信集約。これらのカーネル詳細は未レビュー。
- 採用候補: pageable転送の待ち削減、ピン留め領域へのコピー時間短縮、Router ID読み戻し同期削減。
- 制約: futex、mman等のLinux固有処理がありWindowsへ直接移植できない。単一GPUとCUDA/Vulkan混在では通知・同期の前提が異なる。RAM上の重み所有権も本プロジェクトの重複なし配置と別途照合が必要。

### ewhacc/tandem.cpp

- リポジトリ: https://github.com/ewhacc/tandem.cpp
- 実装確認: https://github.com/ewhacc/tandem.cpp/blob/main/tandem.patch
- ggml_backend_sched_compute_splitsにホスト入力転送の集約、イベント管理、バックエンドのpacked upload呼び出しがある。GGML_SCHED_BATCH_INPUTS=3はCUDA側でまとめた転送を使う構成。
- CUDA Graphのshape別管理、演算融合等も候補。ただし今回詳細を追ったのは入力転送処理。環境変数名だけで機能の正しさや効果を判断しない。
- 採用候補: 既存計測の入力転送・依存待ちが残る場合、入力の一括転送と同期削減。
- 制約: CPU側にも処理させる全体設計は今回のExpertをMI50で計算する方針と異なる。必要な変更だけ抽出し、非同期入力を上書きする前の待機を維持する。

### mxxm-t/mx-llama.cpp

- リポジトリ: https://github.com/mxxm-t/mx-llama.cpp
- 今回確認した資料: https://github.com/mxxm-t/mx-llama.cpp/blob/master/FEATURES.md
- gfx906向け調整、複数GPU通信、MTP状態管理の最適化を記載。今回は資料確認であり、実装レビュー・性能再現は未実施。
- 将来のLinux/ROCmと複数MI50向け候補。Windows/Vulkanでそのまま使えるとは扱わない。

## 現行転送改善の比較条件

Expert再利用を両条件で有効、overlapを無効にし、LLAMA_EXPERT_BATCH_UPLOADの0/1だけ変更する。同一モデル、配置、プロンプト、seed、生成上限、コンテキスト、MTPオフで比較する。

確認は次の4項目に限定する。追加のレアケースは改善事項に記録し、今回の検証を膨らませない。

1. 出力と経路: 生成テキストを保存し比較。native host stagingの対象がVulkan1で、upload_mode=batchになったか確認。フォールバックは成功した高速経路と数えない。
2. 仕事量: permanent_hits、scratch_hits、ram_uploads、uploaded_MiBを比較。同じ計算量かを先に確かめる。
3. 時間: 通常実行のDecode tok/sと、計測実行のexpert_upload_ms・Router読み戻し・入力待ち・各GPU時間を分けて見る。細粒度CUDA計測は同期を追加するため通常速度と混同しない。
4. メモリ: staging追加分とRAM/VRAM使用量を確認。モデルロードと初回ページ読み込みを定常Decodeと混同しない。

## 結果を見て選ぶ次の改善

| 残った時間 | 次の候補 | 判断時の注意 |
|---|---|---|
| Expertアップロード | stagingへのCPUコピー測定、並列コピー、送信集約 | 転送バイト数とPCIe実効帯域を先に確認 |
| Router読み戻し | ホスト通知領域等で同期削減 | CUDA/Vulkan間の可視性とメモリ寿命を別途設計 |
| 入力転送・依存待ち | tandemの入力集約方式 | 必要なバックエンド間依存は削除しない |
| RTX演算・送信 | 通常モードのCUDA Graph再利用確認、その後に融合 | 詳細計測モードではGraphsを無効にしている |
| MI50演算 | Vulkan量子化matvecや演算融合 | 対象量子化・出力精度・ドライバー差を確認 |

現在の改善は層内の複数アップロードをキューに積み、層ごとに待つ方式。一回の連続DMAへの完全な結合や、常駐Expert計算との並行化まで実現したとは扱わない。

参考実装の報告値は投稿の平均70 tok/sを保証しない。MTP、ヒット率、量子化、OS、ドライバー、計測条件が異なる。現行測定で残った支配的な時間に対応する変更だけを次の対象とする。

## 追加ソース確認とCUDA Graph確認 (2026-11-16)

### 入力集約の適用範囲

tandem.patchのggml_backend_sched_compute_splitsは、host bufferかつhost weightではない入力を抽出する。mode 3はまとめたH2D転送とscatterを使い、mode 2は入力コピー完了イベントを記録し、ホストが入力を書き換える前に待つ。対象ファイルはggml/src/ggml-backend.cpp、CUDA backend、src/llama-context.cpp、待機関数の宣言。単純に同期呼び出しを消す変更ではない。

本プロジェクトのCUDA/Vulkan間Activation転送は、元テンソルがGPU bufferの場合この抽出条件に入らない。従ってinput_copy_and_dep_ms=5383.689を全て削減できるとは見積もらない。次に着手するなら、ホスト入力コピーと異種GPUコピー、それぞれの回数・時間を切り分ける。

### Graph形状別キャッシュ

tandem.patchのggml_cuda_graph_get_keyは、先頭ノードのアドレスにノード数と幅を組み合わせてキーを作る。MTPの検証バッチ幅の変化でキャッシュが入れ替わる問題を狙う。現在はMTPオフ、n_ubatch=1固定のため、正常に再利用できている場合は取り込み優先度を下げる。

### Expertコピー並列化とmailbox

llama.cpp-labのsrc/llama-moe-cache.cppでは、転送前にcpy_jobの列を作りcpy_pool::runでstagingへコピーし、その後アップロードを登録する。連続するExpert IDと転送先slotの場合のみ複数Expertをまとめる。今回の約4.73秒のupload計測はCPUコピーとGPU転送待ちを含むため、CPUコピーが支配的と確認するまではworker poolを追加しない。

mailboxはCPUがGPUからのID通知をポーリングし、約50msの期限で通常readbackへ戻る構成。volatile領域の公開・再利用とGPU側の可視性まで合わせてレビューが必要。こちらではRouterがCUDA側にあるためVulkan単独のコードを移すだけでは成立しない。Router読み戻しは約1.05秒で、優先度は入力待ちより低い。

engine-lab READMEは予測prefetchや転送キュー単独等で改善しなかった試行も報告している。ハードウェア固有の結果であり不可能の根拠にはしないが、先読みを無条件に追加する根拠にもならない。

### CUDA Graph実行ログの結果

ログ: logs/cuda-graphs-20261116-094559/run.log。詳細計測オフ、Expert batch uploadオン、64-token生成。終了コード0。

- CUDA Graph id reused: 3951件、97種類のID。
- 同じDecode側IDの再利用は最大63回。
- warmup complete: 97件。warmup reset、graphs無効化、graph update failedの一致は0件。
- llama側graph reallocs: 0/0。こちらのgraph reuseとCUDA Graph reuseは別指標。
- 実装ggml_cuda_graph_update_requiredとggml_backend_cuda_graph_computeの分岐を照合し、安定したUIDでは再capture不要のCUDA Graph経路を継続することを確認。
- 9.26 tok/sは大量のDEBUGログを出した測定で、通常速度10.84 tok/sとの性能比較に使わない。

判断: 毎トークンの再captureが現状の主因という仮説は支持されない。shape-key改善を急いで導入せず、入力コピー・依存待ちの内訳を次の調査候補とする。外部変更の追加実装は未実施。
