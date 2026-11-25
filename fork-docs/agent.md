# このフォークの決定事項

更新日: 2026-11-24。明示的な最新のユーザー指示を優先する。

## 担当と作業の進め方

- 大きな作業は実装前に調査し、各項目のMarkdownに変更範囲・依存関係・完了条件を記録する。独立に検証できる実装単位に分け、1件ずつ差分と結果を確認してから次へ進む。既存テストで確認できる場合はそれを使う。
- 実装はSolのサブエージェントへ委任する。親エージェントは調査、設計、依頼、差分レビュー、ログ分析、優先順位判断を担当する。
- ユーザーが別途指示しない限り、親が直接実装を始めない。実装途中の修正も担当Solへ戻す。文書整理・調査結果の記録は親が行ってよい。
- 可能なら同じSol担当を再利用する。利用できない場合は新しいSolに、対象範囲・決定事項・検証状況を引き継ぐ。
- 担当の完了通知を受けてレビューする。監視目的の短間隔ポーリングはしない。通知を受け取る仕組みがない状態で、自動通知を約束しない。
- 並列作業を依頼された場合は変更範囲とワークツリーを分離する。統合はレビュー後、ユーザーが許可した範囲で行う。
- 実機検証は同じソースフォルダで検証ブランチを切り替え、同じCMakeビルド先を増分利用する。現在の組は `C:/Users/kamui/.codex/worktrees/rtx-cold-assist-v2/llama-hetero-moe` と `G:/AI/build-prefill32`。切り替え前に未コミット差分を確認し、比較する両条件は同じビルド設定で測る。
- フルビルド、モデル実行、速度比較はユーザーが行う。こちらは必要最小限のコンパイル確認、小規模な正確性確認、スクリプト確認までとし、ビルド・実行コマンドを渡す。
- ユーザーがビルドや実機比較を委任した場合、担当チャットへの依頼文に、実行中の出力はGit管理外のファイルへ保存し、終了コードを確認するよう記載する。2026-11-22以降はONだけを逐次実行し、比較条件に合う既存のOFF試行を使う。比較に使ったOFFの設定と計測時のビルドを記録する。失敗したときは該当エラーの周辺だけ確認する。
- このチャットでユーザーから委任されたビルド・実機での動作確認・速度計測はLunaに依頼し、親は条件設定と結果のレビューを担当する。
- 改善ごとにON/OFFを切り替え、他の条件を揃えて比較する。機能実装と採用判断を区別する。 採用済みの改善が新しい構成で使えなくなる場合は、検証前に設定名と併用制限をユーザーに明示し、計測メモにも記載する。
- 性能調査では、親エージェントが中間データの生成先から最終利用先までをコードとログで追い、GPU内・GPU間・GPU/RAM間で省けるコピー、再変換、転送専用Submitを先に提案する。候補の発生条件、回数、既存の再利用、上書きと同期の制約を確認し、待ちの位置が移る場合は前後を含む時間で評価する。
## Gitの運用

- ブランチを切り替える前に未コミット差分を確認し、切り替え元のブランチで現在の作業分をコミットする。途中なら途中経過と分かるコミットにして、再開地点を残す。別作業の差分は混ぜず、残っている場合は切り替え先を別のワークツリーで開く。
- 実装と確認が済んだとき、調査・計測結果をまとめたときなど、作業の区切りで個別の指示を待たずにコミットし、ユーザーのフォークの対応する作業ブランチへpushする。コミットIDとpush先を報告する。ユーザーが特定の作業でコミットやpushを保留するよう指示した場合は、その指示を優先する。

## テストと評価

- 動作確認は、小分けした作業単位ごとに、基本的に実際のモデルを起動し、通常の入力で変更した経路と出力を簡単に確認する。RAM と GPU の転送完了待ち、バッファの再利用や解放など、不具合が起きやすく通常のモデル実行だけでは検出しにくい箇所には、対象を絞った単体テストを追加してよい。既存テストで足りる場合はそれを使う。
- 単体テストは、各項目のMarkdownの「変更範囲」に書いた個々の作業単位ごとに最大3ケース。たとえば「gate/up と down の量子型、内積長、出力幅、ストライドを採取する」が一つの作業単位。3ケースは上限なので、簡単な確認で足りれば0件でもよい。4ケース目が必要なら、理由と確認したいリスクを示し、追加前にユーザーへ相談する。
- レアケースや異常系は網羅しない。実際に見つかった不具合や、変更の正しさを確認するうえで重要なケースに絞る。
- 正常系の速度はCUDA Graphsを有効にした通常経路で判断する。詳細計測のためGraphs無効・同期追加を行うモードは別扱い。
- GPU時間、CPU待ち、転送待ちには重複がある。単純に合算したり、待ち時間をすべて削減可能な時間と扱わない。
- トークン数、MTP受理率、会話履歴、コンテキスト、量子化、設定を揃える。各1回の差を確定的な改善と断言しない。
- 高速化による小さな数値差や文言の違いは、自然な日本語で問いへの回答と指示が保たれ、速度改善が測定できれば許容する。生成結果とMTP受理率を比較する。トークン数一致だけで出力完全一致としない。
- 固定の5文章は `fork-docs/research/dialogue-benchmark-prompts.md` を使う。

## 保存場所とGit管理

- フォーク固有の設計書、調査資料、スクリプト、AI向けルールは `fork-docs` に置く。upstreamの `docs` と混在させない。
- 生ログ、プロファイル、計測CSV、実行時出力はGitへ追加しない。以前の「ログをコミット」の指示は撤回済み。
- 現在のログ保存先: `G:/AI/llama-hetero-moe-local/logs`。
- 分析結果の要約・判断根拠は `fork-docs/research` に残してよい。生ログを転載しない。
- 現在の検証用チェックアウトとビルド先は「担当と作業の進め方」の実機検証項目を参照する。
- 本体 `G:/AI/llama-hetero-moe` への統合は保留。明示的な指示なしに変更・統合しない。
- `AGENTS.md` は本書への案内にとどめ、upstreamの本文を維持する。

## 実機とモデル

- Windows 11、Ryzen 5 7600、RTX 4070 Ti 12GB、MI50 32GB。
- MI50はVBIOS適用により `AMD Radeon Pro VII` として認識。AMD純正ドライバー使用。有志ドライバー使用という古い前提は誤り。
- 現在はRTXをCUDA、MI50をVulkanで使用。過去の列挙はCUDA0とVulkan1だが、実行環境で再確認する。
- モデルはダウンロード済みQwen3.8-Flash-Next UD-IQ3_XXS。MTPは別ファイルのQ8_0。
- モデル保存先: `G:/AI/share/llm_models/Qwen3.8-Flash-Next-GGUF`。
- 標準比較はコンテキスト2048、MTPの1候補と本体の2トークン一括検証。正確な起動条件は現行スクリプトとsettings.jsonを確認する。
- 将来のLinux/HIP、MI50複数枚は構想。Windows HIPでのgfx906認識・演算・CUDAとの同時利用は未検証。Vulkanで動くことをHIPの動作証拠にしない。

## 推論方式と採用状態

- RTX: Attention、Router、Shared Expert、MTP。MI50: Routed Expert。
- ExpertはMI50とRAMで恒常的な原本を重複保持しない方針。選択されたRAM側Expertを入れ替え枠へ転送する。転送中の一時コピーと原本の二重常駐は区別する。
- PLEはNVMeから遅延読み込み。初期配置は公開hits上位300個/層を基礎とし、動的入れ替えを採用。
- 採用済み: Expert再利用、一括転送・ステージング、Router IDのホスト参照、入力転送一括化、スロットマップ集約、完了確認Submit削減、常駐Expert全体への転送隠蔽、動的入れ替え転送の並行化、Shared Expert並行化、MTP一括検証。
- 上記の設定の正確な環境変数は現行スクリプトの基準設定を確認する。名前からON/OFFを推測しない。
- RTXへのcold Expert分担はOFF・統合保留。
- 旧HC全体融合とHC Scale/SiLU融合はOFF。後者は発動と小規模出力一致を確認したが、通常速度比較で採用根拠を得ていない。
- Vulkan転送待機削減の不採用実験やIQ4_NLペア重み共有を、採用設定へ勝手に加えない。
- MTP先読み距離の拡張は後回し。Expert書き戻しを原本RAM重複保持に置き換える変更も今回の対象外。

## 現在の改善タスク

- 2026-11-20に固定2幅のMI50 gate/up投影＋SwiGLU融合を採用。基準設定は `GGML_VK_MTP_ROUTED_GATE_UP=1`。両順計4回のDecodeはOFF 34.54、ON 36.05 tok/sで、生成IDとMTP提案・受理件数は一致。対象の17,367/18,144層実行で発動した。

- 2026-11-19にRouter重みの乗算と10 Expertの順次加算の融合も採用。以後の固定2幅の比較は `GGML_VK_MTP_ROUTED_REDUCE=1` と `GGML_VK_MTP_ROUTED_WEIGHTED=1` を基準とする。ウォームアップ後4回合算は加算集約のみ35.47 tok/s、乗算込み36.26 tok/s。生成IDとMTP提案・受理件数は両順とも一致した。

- 固定2幅の本体検証でMI50のRouted Expertの9加算集約を採用。2026-11-19の両順比較で生成ID・MTP件数が一致し、集約のdispatchは各比較で163,296回から18,144回に減少した。Router重み乗算の融合比較では両条件で `GGML_VK_MTP_ROUTED_REDUCE=1` とし、空コマンドプールのリセット省略と3入力の同一ストリーム読み戻しはOFFとする。

- 2026-11-19の実機比較で、固定2幅・全語彙の通常生成には `GGML_CUDA_HC_DOWN_REUSE=1`、`GGML_CUDA_HC_UP_REUSE=1`、`GGML_CUDA_HC_INJECT_SPLIT=1` を採用。ウォームアップ後の両順序・計4回で全OFFは31.16 tok/s、全ONは33.62 tok/s。生成IDは条件間で異なる。以後の空コマンドプール比較でもこの3設定を共通にし、3入力の同一ストリーム読み戻しはOFFとする。
- HC up側のQ8_0行列演算を2トークン向けに調整する比較実験をユーザーが承認済み。
- 対象は内積長320、出力幅10240のHC up。既存MMVQとQ8_1入力量子化を再利用し、スレッド配置を限定変更する。
- 実験フラグ: `GGML_CUDA_HC_UP_PAIR`。スクリプトの `-HcUpPair` はSolの実装・小規模検証完了。実モデルの速度・品質は未検証で、採用は未決定。
- HC融合、RTX分担はOFFのまま。次はSolの実装・小規模検証を親がレビューし、ユーザーにビルドと通常経路の比較コマンドを渡す。
- 優先順位の根拠: `fork-docs/research/decode-bottleneck-review-20261117.md`。HCが通常経路の最大ボトルネックと確定したわけではない。

## 会話履歴から補足した継続方針

- 既存OSS・ggmlの機構を優先し、独自フレームワークや不要な抽象化を増やさない。upstream追従を考慮し、修正範囲を局所化する。
- 開発対象はダウンロード済みFlash-Next。古い35B中心の計画を現在の前提に戻さない。
- 通常モデルで対応・速度を確認した後、SC117のGSQ-RCO-abliterated版へ進む。切り替えは未実施であり、モデル構造・量子化・MTP互換性を再確認する。
- 最終構想はRTX 4070 TiとMI50を2〜3枚、載り切らない部分はRAM/SSD。現段階の単一MI50での検証と将来対応を混同しない。
- 新しくユーザー管理のプロダクトフォルダを命名する際はドットを避ける。既存の `.codex` などツール管理パスを勝手に改名する指示ではない。
- 参考投稿の70〜106 tok/sは再現保証ではない。投稿者のランタイム、量子化、MTP、計測条件と自環境の差を区別する。
- 微差の改善は必要に応じてウォームアップ・複数回・逆順比較で確認する。単発結果のために無制限の検証を繰り返さない。
- 2026-11-16以降の「ログをGitに載せない」が、それ以前の「ログもコミット」を置き換える。履歴の書き換えや削除の実施済み状態はGitで確認し、過去の依頼だけで完了と推測しない。

## セッション継続メモ

- 親エージェントは開始・コンテキスト圧縮後・再開時に [現在の作業メモ](teams/current-work.md) を最初に読む。ユーザーの最新判断、対象ブランチとビルド先、着手済みと保留の作業、次の優先順位を確認する。
- [作業・決定の履歴](teams/coordinator-memory.md)、[調査と未解決事項](teams/research-notes.md)、[採用設定とON/OFFフラグ](teams/runtime-flags.md) は必要な箇所を参照する。この文書中の日付付き設定は当時の記録として扱う。
- 次の改善を選ぶ際は、既存の優先順位と候補の採否を先に確認する。追加のログ調査は、優先順位を決めるために足りない情報に絞る。
- 新しい決定、実験結果、担当への引き継ぎ、未解決事項を得たら、終了前に現在の作業メモの該当箇所を更新する。調査には根拠・結果・残件、フラグにはON/OFF値・対象経路・結果、改善メモには短い狙いと次の判断を残す。長い経緯は履歴へ追記する。
- メモは補助情報。最新のユーザー指示、コード、Git状態、実ログと矛盾する場合は再確認し、古い記述を訂正する。
- 生の会話JSONL、認証情報、生ログをメモへ複製しない。根拠の場所と必要な要約だけを残す。

## HC投影改善の運用更新（2026-11-17）

ユーザーはdown/up/injectの新しい3改善を全部ONで採用する方針。今後の通常実行コマンドでは `-HcAll` を指定する。旧HC融合・Scale/SiLU融合・旧up単独ワープ変更・RTX分担はOFFを維持。
ただし今回の全体平均24.27 tok/sは歴史的OFF26.11 tok/sを上回っていない。ユーザーの採用判断と、速度改善の実証を区別する。詳細は `research/hc-projections.md`。既存比較スクリプトのスイッチ省略時のOFF動作と、低レベルフラグ既定値は変更していない。

## 最新の基準と並列改善（2026-11-17）

ユーザー決定により、新HC down/up/injectの3改善もOFFに戻す。直前の全ON採用方針はこの決定で更新済み。旧HC融合、旧up単独ワープ変更、RTX分担もOFF。コードは比較用として保持する。
次候補はSolが並列で調査・限定改善する。Recurrentは現在の検証用ワークツリー、Expert転送tailは別ワークツリーで進める。効果確認も各ワークツリーで行い、取り込みは結果確認後に判断する。本体への先行統合は禁止。

## Branch-based validation workflow (2026-11-17)

The fixed validation checkout is `C:/Users/kamui/.codex/worktrees/rtx-expert-validation/llama-hetero-moe`. Reuse `G:/AI/build-rtx-validation` only with this source directory. Development stays in separate worktrees; validate committed branch tips sequentially in this checkout, preferably detached so development branches remain usable in their own worktrees. Never switch during inference/build, and require a clean checkout before switching. Stop at configure/build failure; never benchmark an old executable after failure. Commit and record the exact experiment tip before each validation. Follow the current Git workflow above for milestone pushes.
Historical tested changes and opt-in experiments may share a baseline commit where file boundaries overlap; do not reconstruct artificial historical commits. In-progress Recurrent and transfer-tail changes remain separate. The 2026-11-17 local-only approval applied to those historical commits. Apply the current Git workflow above for new work. CLI comparisons keep all HC experiments and RTX sharing OFF unless explicitly testing them.

## 30 tok/sに到達した検証設定（2026-11-17）

- 検証ブランチは `codex/recurrent-conv-validation`。`-Mode speed -ExpertRamRotation -SharedHostExpertInput -SharedHostExpertReturn -MtpDevicePending -RecurrentConvHistory` を指定する。Q8再利用、戻り2枠化、HC実験、RTX Expert分担はOFF。
- 固定5回答は生成40.53秒、30.00 tok/s。既存の検証フォルダと `G:/AI/build-rtx-validation` を使用する。増分ビルドでQwen4exp側のUnityコンパイルが省かれた場合は、該当するUnity単位を再コンパイルしてDLLと発動ログを確認する。
- 保留・不採用の実験は作業ワークツリーのアーカイブから復元できる。生ログやプロファイルはGit外に保管する。

## 採用済みフラグの整理（2026-11-19）

現在の作業ツリーでは、採用済みの最適化を対応する配置・GPU・テンソル形状で自動選択する。固定5問の通常検証は start-dialogue-measurement.ps1 に -Mode speed と使用する -BuildDirectory を渡す。上の旧5スイッチ付きコマンドは過去のバイナリ用。Prefill32は比較試験、旧HC融合とRTXのExpert分担は引き続き任意。実機で確認するのは、同じモデル・MTP設定で自動選択された経路と出力・速度。
