# 親エージェント用メモリ

更新: 2026-11-17。用途: セッション圧縮・再開時の引き継ぎ。共通ルールは `../agent.md` が正本。

## 最初に読む

1. `../agent.md` とこのメモ。
2. 最新のユーザー発言と担当からの完了通知。
3. 作業チェックアウトのブランチ、HEAD、未コミット差分。大量の他タスク差分が存在するため、勝手にリセット・一括コミットしない。
4. 対象スクリプトと保存済みsettings.json。記憶だけで環境変数を再構成しない。

## 役割を忘れない

- 親は調査・設計・委任・レビュー・結果分析。実装はSol。
- 2026-11-17に親がHC改善を直接実装してユーザーから訂正された。最新依頼に毎回Solと書かれていなくても継続方針は変わらない。
- 文書・メモ整理は親が実施可。実装担当は完了通知で確認し、短間隔で監視しない。
- フルビルド・モデル実行はユーザー。こちらの小規模検証は最大4項目を基本とする。
- 今の本体統合は保留。作業場所は `C:/Users/kamui/.codex/worktrees/rtx-expert-validation/llama-hetero-moe`、ユーザーのビルド先は `G:/AI/build-rtx-validation`。

## 現在のタスクと次の行動

- HC upのQ8_0、K=320、N=10240、2トークンのMMVQを4warpから1warpへ切り替える限定実験。
- 担当: `/root/hc_up_pair_sol`、モデル `gpt-6.1-sol`。利用できなくなった場合はSolを再作成して引き継ぐ。
- フラグ `GGML_CUDA_HC_UP_PAIR`、起動スイッチ `-HcUpPair`。既定OFFで、採用は未決定。
- 実装対象 `ggml/src/ggml-cuda/mmvq.cu` と `fork-docs/scripts/start-dialogue-measurement.ps1`。HC融合・RTX分担OFFを明示する。
- 担当の中間報告: 修正版DLLで1tokenと非HCはbit一致、2tokenはmax abs 6.519258e-09、rel L2 1.153099e-07。これは小規模fixtureのみで、本モデルの品質や速度は未確認。
- 注意: 初版はホストのTURING配置とGPUのGENERIC配置が不一致で誤計算した。修正済みと担当報告。AdaはGENERICテーブルを使用する。古い失敗fixtureの結果を成功結果と混同しない。
- 担当から最終compile/link、fixture、PS構文・排他確認完了の通知あり。仕様と結果は `../research/hc-up-pair.md`。親レビュー完了。対象shape/type/architecture限定、host/device配置一致、既存量子化とフォールバック、両HC融合/RTX分担OFF、Git外ログ、最終fixture発動ログを確認し、比較実行を妨げる問題なし。ユーザーへビルド・通常経路OFF/ON比較コマンドを提示する段階。残件は実モデル測定と採用判断。
- 現在の文書整備はこのタスクを取り消していない。コミット・pushの新規承認はない。

## 直近の測定結果

ログ基準ディレクトリ: `G:/AI/llama-hetero-moe-local/logs`。生ログはGit禁止。

| ログ末尾/名前 | 用途・結果 |
| --- | --- |
| dialogue-speed-20261117-194511 | HC Scale/SiLU OFF。5応答、生成46.580秒、全体26.105 tok/s |
| dialogue-speed-20261117-194833 | 同融合ON。48.145秒、25.257 tok/s。採用せずOFF維持 |
| dialogue-components-20261117-193322 | 融合発動。target_verifyで67,008起動削減。通常速度の改善保証ではない |
| dialogue-components-20261117-190812 | 融合OFF。HC行列7.229秒、HC全体9.379秒、other16.203秒、Recurrent8.189秒 |
| dialogue-components-20261117-183225 | 完全な5応答。前回のHC/other内訳の参照用 |
| dialogue-components-20261117-191445 | 融合未発動。ONの比較証拠には使わない。後からCSV集計済み、6応答 |
| dialogue-components-20261117-182550 | 応答境界欠落。完全な応答合計として使用しない |
| dialogue-telemetry-20261117-182307 | Graphs有効の全体計測。通常速度比較とは別実行 |

`182307`の5応答: decode54.764秒、本体検証49.057秒、draft2.608秒、catchup1.435秒。本体検証中のCUDA GPU区間14.525秒、Vulkan GPU区間13.940秒。CPU待ちとの重複あり。
Expertヒット率96.11%でも層ステップの42.29%にミス。アップロード45.92GiB、書き戻し11.82GiB。転送量だけで削減可能秒数を計算しない。

比較時の平均速度は各行のtok/sの単純平均ではなく、計測対象トークン数/時間合計。CLIは応答ごとに生成トークン数から1を除いた数で速度を計算する。文面一致・内部数値一致・トークン数一致は別。

## 優先順位の現状

1. HC upの2token短内積配置: 測定上の負担と具体的コード非対称性がある。
2. 並行転送後に残る動的promotion/activation返却の待ち。
3. Recurrentとother内の具体的な重いノード。
4. MTP生成/追従。先読み拡張は後回し。

詳細: `../research/decode-bottleneck-review-20261117.md`。HCが全体最大だと断言しない。計測同期のあるcomponentsの秒数を通常経路の削減予測へ直接流用しない。

## 実行まわりの既知事項

- PowerShell5.1はnative stderrの警告をErrorAction Stopで例外化する。スクリプトはnative呼び出し中だけContinueにし、終了コードを確認する方式。
- 対話ログ欠落対策は `--simple-io --color off`。欠けたログを集計器で黙って補完しない。
- ワークツリーのビルド場所はGドライブの外部ディレクトリ。既存ユーザービルドを検証用DLLで上書きしない。
- 小規模artifactは `C:/Users/kamui/.codex/visualizations/2026/11/12/01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f/hc-scale-check`。ビルド済みDLL・fixtureは更新されるのでタイムスタンプとログを確認する。
- この環境のnvccはsandbox内でACCESS_VIOLATIONになる場合があり、成功した限定compileはrequire_escalated。拒否された場合は理由を説明し、承認を仮定しない。

## 会話JSONLの確認元

- 会話名: LLM司令塔。ID: `01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f`。
- 元ファイル: `C:/Users/kamui/.codex/sessions/2026/11/12/rollout-2026-11-12T16-28-43-01a1cea8-ae9e-7fc3-a30a-9d8b72f3a31f.jsonl`。
- 2026-11-17の見直し時に約47MB、user response_itemを334件確認。ファイルは追記されるため件数は固定ではない。
- ユーザー発言は主に `response_item.payload.role=user` のcontent.text。event_msgのuser_messageだけ検索すると0件だった。Pythonの表示はstdout UTF-8指定で文字化けを回避。
- 11/15 UTCのSol委任・テスト最大4、11/16以降のGit外ログ、11/17のモデル対応順・親直接実装の訂正を確認。初期依頼の実装禁止は後続の明示的実装依頼により対象範囲内で更新済み。

### HC up initial model run received

`dialogue-speed-20261117-201640`: activated, exit0, 23.158 tok/s versus historical OFF26.105. Output lengths/MTP acceptance differ; response3 prompt eval327 versus68, so not controlled identical work. Recommend default OFF/no adoption, pending user decision; do not describe as proven 11.3% kernel regression. Detailed note in `../research/hc-up-pair.md`. User preferred skipping new OFF runs due to duration; do not request another full dialogue casually.


### Active task: HC down/up/inject independent experiments

User explicitly requested Sol to implement improvements for all three HC projections, separate flags and separate operation logs. First full-model run should enable all three. RTX sharing OFF; raw logs outside Git. Assigned to existing `/root/hc_up_pair_sol` using followup_task. Parent only reviews and analyzes. Preserve earlier experimental HC fusion OFF unless replaced by an explicitly scoped new flag. Small tests before user build/model run. Do not wait for repeated long dialogue OFF runs: user prefers the existing baseline for initial screening. Await implementation proposal and completion notification, then review combined interactions and profiling overhead. No commit/main integration authorized.

### HC projections completed by Sol

Sol completed down/up/inject independent flags and projection telemetry; parent reviewed shape restrictions, row-group indexing/reduction, fallback, scratch current-stream lifetime, telemetry dimensions and script settings. No blocker found for experimental full-model comparison. Specs/results: `../research/hc-projections.md`; Sol handoff: `hc-projections-sol.md`.
Small graph median microseconds OFF/ON: down5.335/5.252 (noise-sized), up8.699/9.019 (no gain), inject4.128/2.690, linked18.112/17.693. Down/up bit-identical; inject max abs1.1921e-6 relativeL2 3.7771e-7. Full-model quality/speed unverified. First user run `-Mode speed -HcAll`; detailed split timings require separate `-Mode components -HcAll`. Legacy HC fusion/old up-pair/RTX sharing remain OFF. No adoption/commit/main integration. User should send new log directory for analysis.

### HC all-ON adopted as operational preference

User accepted all three HC projection experiments after log `dialogue-speed-hc-all-20261117-204235`. Use `-HcAll` for subsequent current-configuration runs unless comparison calls for OFF. Keep old HC fusion, Scale/SiLU, old up-pair and RTX sharing OFF. Actual all-ON24.272 tok/s is below historical OFF26.105, although above previous up-pair23.158. Output lengths/acceptance changed; adoption must not be described as measured speedup. Low-level defaults unchanged and no commit/main integration performed. Detailed per-projection full-model components log still pending.

### Adjacent OFF received: adoption reconsideration

Latest OFF `dialogue-speed-20261117-204629`24.3885 tok/s, allON20423524.2722 (-0.48%). No demonstrated end-to-end gain; generated counts/acceptance differ. OFF matches historical baseline counts. Parent recommends not treating allON as performance-validated; user previously adopted, so do not silently rewrite that decision. No code/default changes; awaiting user's choice. Details in hc-projections.md.

### Active parallel work: Recurrent and transfer tail

Latest user instructions supersede prior HC adoption: all three HC improvements OFF. User asked parallel next-candidate improvements and explicitly requires measuring in worktrees before integration decision.
- `/root/hc_up_pair_sol`: Recurrent investigation and evidence-backed limited implementation, current rtx-expert-validation worktree, primary file gated_delta_net.cu. Recurrent8.2s is not all GDN; investigate with realistic shapes and CUDA Graph fixture. Avoid shared files during investigation.
- `/root/transfer_tail_sol`: separate worktree required, investigate exposed Expert promotion/writeback/activation return waits and implement justified opt-in improvement. Uses create_worktree, preserve current uncommitted baseline and record delta. Worktree location pending agent report.
Parent reviews only; no commit/push/main integration. Full model builds/runs by user, raw logs outside Git, complete notifications instead of repeated status polling. Need review and separate build/ON/OFF commands when each completes; do not combine experiments before validating independently.

## Validation branch organization (2026-11-17)

- Baseline implementation snapshot: 787460224, branch codex/hetero-validation-base. Historical changes grouped with opt-in experiments; HC experiments and RTX sharing remain OFF in ordinary comparisons.
- Fixed validation source: C:/Users/kamui/.codex/worktrees/rtx-expert-validation/llama-hetero-moe; build: G:/AI/build-rtx-validation. See research/shared-validation-build.md. Switch only clean committed tips; configure and build must succeed before measurements.
- Recurrent development: C:/Users/kamui/.codex/worktrees/recurrent-tuning/llama-hetero-moe, codex/recurrent-tuning. Sol hc_up_pair_sol. GDN warp trial not adopted, backed up in Git stash named recurrent: unadopted GDN warp experiment backup. Next investigation is QKV.
- Transfer development: C:/Users/kamui/.codex/worktrees/transfer-tail-review/llama-hetero-moe, codex/transfer-tail. Sol transfer_tail_sol. Copied baseline backed up in a separate stash before alignment.
- Both development branches start from 787460224. Agents resumed there; no writes to fixed validation source. No push, no main integration, no full build/model run during organization. Logs remain outside Git.
## MTP RTX内保持の採用（2026-11-17）

- 検証コミット `ac39f6f79`。固定5回答のON実行は正常終了。ユーザー判断で今後の検証設定に `-MtpDevicePending` を採用。RAM回転と共通RAMの行き戻りを併用。Q8再利用と戻り2枠化はOFF。
- 生成合計47.64秒・25.52 tok/s。第2回答30.88 tok/s、第4回答17.18 tok/s。生成トークン数とMTP受理数は従来の固定5回答と一致。第4回答の受理率91.43%。同一ビルドのOFF比較は未実施。
- MTPログはD2D 901、D2H 9、H2D 1、ホストからのdraft入力0。生ログはGit外。

## Recurrent履歴直接書き込みの採用（2026-11-17）

- 現在の検証基準は `cb7235b20`。ユーザー判断で `-RecurrentConvHistory` を採用。 `-ExpertRamRotation -SharedHostExpertInput -SharedHostExpertReturn -MtpDevicePending` と併用する。Q8再利用と戻り2枠化はOFF。
- 固定5回答で40.53秒、30.00 tok/s。第4回答29.45 tok/s。未発動の比較ログは41.46秒、29.33 tok/s。生成数と受理数は一致。同一バイナリでの反復測定は未実施。
- Windowsの増分ビルドではNinjaがQwen4expのUnity依存を拾わず、CUDA側だけ更新されたことがあった。発動マーカー `CUDA SSM convolution history applied` と `exit_code=0` を確認。ソース切替後にモデル側の変更が入らなければ、対象Unity単位を再コンパイルする。

## Prefill専用の32トークン比較（2026-11-18）

- Sol `prefill32_sol` が `codex/rtx-cold-assist-v2` の作業ツリー（ベース `21b588e80`）に `-ExpertPrefill32` を実装。プロンプト専用バッチのみ最大32。MTPの本体検証は最大2。非駐Expertは既存の20枠を使い回し、同じExpertに割り当てられたトークンをVulkanの既存グループ行列演算で処理する。
- 上限まで処理回を予約するため、空の回にもゼロ埋め・合算が残る。終了ログ `packed expert prefill` の `max_tokens`、`waves_needed`、`waves_scheduled` と各応答の `prompt eval time` を確認する。固定5応答の基準ログ `232854` は入力442トークン、14.30秒、30.91 tok/s。比較では入力トークン数や会話履歴も確認する。
- MSVC構文確認とPowerShellの構文確認は通過。既存DLLを使ったPro VIIの32トークン小規模テストで、マスクされた行のゼロ化と入れ替え枠の再利用を確認。実モデルでの出力と速度は未検証。ユーザーは32版を先に実行する。ビルド先は作業ツリー専用の `G:/AI/build-prefill32`。コミット・push・統合は未実施。生ログはGit外。
- 実機ONログ `prefill32-20261118-015159` は終了コード0、最大32を確認。固定5応答のPrefillは442トークン、16.68秒、26.50 tok/s。過去OFFは442トークン、14.30秒、30.91 tok/s。ONで7,728回を構成し、必要だったのは1,210回。余分な6,518回は84.3%。生成は約29.11 tok/s（受理率509/675）、過去OFFは30.00 tok/s（516/698）。出力トークン数が異なり、同一バイナリのOFFは未測定。静的に作る空の処理回を除けるかを次の設計候補とする。
- 同じSolが空の処理回を省く `-ExpertPrefillActiveWaves` / `LLAMA_EXPERT_PREFILL_ACTIVE_WAVES=1` を追加。`-ExpertPrefill32` が必須。後続の空の回ではスケジューラーの限定されたprefixを省き、前の合計を同じバックエンドの出力へコピーしてsuffixを実行する。最初の回は必ず実行して初期化する。グラフ自体や20枠、MTP検証2は維持。空の回の入力待ちや必要なD2Dコピーは残り、split自体がなくなる実装ではない。
- レビューで毎回の全グラフ走査、依存ノードのインデックスずれ、view経由の外部参照、コピー重なりを指摘。構造検証をグラフ単位でキャッシュし、split再構築で無効化。実GPU fixtureで32トークンup/SiLU/downと末尾SCALE、空->実処理->空の枠再利用についてstatic/activeのbyte一致を確認したとSolが報告。eval callbackありでは従来経路に戻る。フルモデルの速度・出力はユーザーの比較待ち。終了カウンター `packed expert prefill waves: active_only=1 executed=... skipped=...` を確認する。コミット・pushの承認はまだない。
- active実機ログ `prefill32-active-waves-20261118-080115` は終了0、実行1,409/省略7,135。Prefill491トークン27.38232秒17.93 tok/s、生成24.48 tok/s。最初の入力114トークンで従来の63と異なるが、2-5問の入力も377対379で20.01対29.72 tok/sと遅い。詳細時間未計測のため原因秒数は未確定。
- 不具合を確認。追加した `ggml_view_tensor(ctx0, experts)` はOP_NONEのviewで、allocatorの参照追跡と合わず各層の出力3.125MiBを保持し続ける。Vulkan compute buffer9.69から153.44/156.57MiBへの増加と46/47層分が一致。Solの48層割当fixtureはviewなし6,553,600 bytes、問題viewあり160,563,200 bytes。追加viewを削除し、末尾scaleのcallback名でaccumulator識別を維持する修正を実施。小規模構文確認後に再ビルドを案内する。ログだけではこのメモリ増加を遅延の全原因とは断定できない。
- 修正版実機 `prefill32-active-waves-20261118-081045` は終了0、Vulkan compute9.69MiBへ復帰。static32の015159と入力数63/79/69/136/95、生成数245/256/256/185/247、受理数とExpert転送・共通RAM回数が一致。実行1,210、省略6,518。Prefill442トークン19.58595秒22.57 tok/s（static16.67755秒26.50、時間17.44%増）。生成のCLI計時対象1,184トークン44.19219秒26.79 tok/s（static40.67552秒29.11）。第2問はPrefill改善、第3-5問は悪化し、第4問は生成も大きく遅い。設定差はactive-waveフラグのみだが別時刻の単発なので純粋な因果時間は未確定。メモリ不具合修正は確認できたがactive-wave速度改善の採用根拠なし。次の判断には省略判定・依存検証・carryコピー・残るsplit待ちの内訳が必要。

- Strata公式 c77572806996349b49995e4f98d41b5294d36a21 を再調査。src/prefill/prefill.cpp:5797付近は配属行をExpert別に詰めてcnt>0の実行リストを作る。src/program/generate.cpp:18063付近は短い既知入力を検証窓で読む経路（short_read基準64、機種等で変更）。生成候補の検証・commit(a+1)とは区別。現行32は上限回数分のFFNと全形状の加算を構築し、空回のsplitは残る。次の判断材料は層別の実割当行数/Expertと実行回、実転送bytes、prompt長別時間。

## 採用済みの切り替え削除（2026-11-19）

- 現在の作業ツリーで、検証基準の5スイッチと保存済み設定でONの12項目を整理。配置プロファイル・使用デバイス・対応形状などを条件に自動選択する。通常の計測は start-dialogue-measurement.ps1 -BuildDirectory G:/AI/build-prefill32 -Mode speed。Prefillの実験幅、MTPの指定、観測用のフラグは独立。Prefillの再設計は別作業。フルビルドと実モデルの比較はユーザーが実施。
- 実機確認 20261119-101723 は8応答・終了0。RAM回転43回、共通RAM入力34,415バッチ・戻り32,264回、MTP device pending 852コピー。256トークンの第3/6応答は16.05/27.19 tok/s。ユーザーによると序盤はブロワー回転設定が低く、固定5問と異なるため性能回帰比較には使わない。
- 2026-11-19 外部提案の原文を research/improvement1-upload-memcpy.md に保存。Decodeのstaging memcpyは残る。旧target_verify 138回で11,716,147,200 bytes、層内2ミス以上1,492/6,624。overlapの事後拒否でfallbackが再パックし得るため、経路・ミス数別の実コピーbytesとCPU時間を先に測る。通常speedのpending=0は計測無効のため判断材料にしない。
- 2026-11-19 stagingコピー計測を作業ツリーに追加。通常設定の -Mode speed -ExpertRamCopyTiming で終了時に expert_staging_copy のphase別集計。target_verify行のcpu_ms/verify_calls・fill_ge2_cpu_ms・retry_bytesを確認。GPU詳細計測は不要。小規模fixtureとC++構文を確認、実モデル実測待ち。
- 実機105746（6応答、終了0）。target_verify 706回・45.662秒、staging memcpy 49.858 GB/7.287秒（10.32 ms/verify）、2件以上4.753秒（6.73 ms/verify）、retry_bytes=0。複数fillだけの2-way CPUコピーをA/B候補にする。昇格側RAM memcpy 2.491秒は全フェーズ別カウンター。
- 2026-11-19 ユーザー判断によりstagingコピー並列化の優先度は低、実装・追加検証は保留。試算と実測は fork-docs/research/improvement1-upload-memcpy-results.md に記録。
- 2026-11-19 改善2（MI50からRTXへの戻り同期のイベント化）もユーザー判断で優先度は低、実装・追加計測は保留。[調査結果](../research/step2-shared-return-sync-results.md)
- 2026-11-19 Prefill段階1は `-Mode speed -ExpertPrefillTrace` で実行。実際のPrefillとMTP本体検証は最大2トークン。`compare-expert-prefill.py` は各リクエスト・層の最初の常駐配置を固定し、同じRouter列を2/32/128/256幅で再生する。幅の効果は `delta_bytes_vs_width2` と `full_width_windows` で判断。集計対象はRAMからのExpert重みアップロードで、実行時間の効果は未検証。
- 2026-11-19 Prefill 512の実機比較は `python -B fork-docs/scripts/compare-prefill512.py --build-directory G:/AI/build-prefill32`。125254の長文を再構成し、外側512で実行時のPrefill幅2と512を比較。入れ替え枠20、MTP検証2。512の広幅部分では昇格を停止してExpertごとにまとめる。実測待ち。VRAMはCLI実行全体のGPU別・プロセス別サンプル最大値。
- 実機134128の512幅は採用候補。1,378入力・53生成でPrefill 47.33→21.26秒、生成2.07→2.71秒、本文一致・MTPは両方20/32。512の終了コードのみ未記録。次の比較は `--decode-expert-window` で両条件を新規実行し、先頭32生成位置と以降を対照する。先頭1位置はPrefillからサンプルし、境界をまたぐMTP検証は独立集計。生ログはGit管理外。
- 2026-11-19 Decode序盤の比較140221。幅2/512は同じ1,378入力・53生成・検証32回・本文と受理数一致。位置2–32の要求ミス887/3,209、位置33–53は533/1,760。昇格は幅2が検証1回目から、512は26回目（生成位置43）から。退避対象の最低経過32 lookupに対し512経路もチャンク単位で1しか進めず、幅2はPrefill 689回、512は5回。次の候補はPrefillの最終使用位置を2トークン相当の時刻で引き継ぐ小規模比較。全実行の転送には両条件で同じ未帰属273件・553,932,800 Bがある。要求差分からは相殺されるが発生元は計測上未確定。
- 512幅内のExpert最終使用時刻を2トークン換算で記録する実験は `compare-prefill512.py --compare-prefill-last-use`。両方512幅でOFF/ONを新規実行する。最初の昇格ラウンド、先頭32位置の昇格数・RAMアップロード・書き戻し、生成全体のDecode時間を比較する。実機での採用判断待ち。
- 2026-11-19 採用: Prefill512を有効にした対話では、キャッシュを除いた入力79トークン以下は2幅、80以上は最大512幅。MTP本体検証は2幅。199トークンで512幅選択、6.251秒（31.84 tok/s）を確認。10枠ずつの並行転送は別ワークツリーで実機比較待ち。
