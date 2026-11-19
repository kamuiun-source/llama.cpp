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

## 採用済みの切り替え削除（2026-11-19）

- 現在の作業ツリーで、検証基準の5スイッチと保存済み設定でONの12項目を整理。配置プロファイル・使用デバイス・対応形状などを条件に自動選択する。通常の計測は start-dialogue-measurement.ps1 -BuildDirectory G:/AI/build-prefill32 -Mode speed。Prefillの実験幅、MTPの指定、観測用のフラグは独立。Prefillの再設計は別作業。フルビルドと実モデルの比較はユーザーが実施。
- 実機確認 20261119-101723 は8応答・終了0。RAM回転43回、共通RAM入力34,415バッチ・戻り32,264回、MTP device pending 852コピー。256トークンの第3/6応答は16.05/27.19 tok/s。ユーザーによると序盤はブロワー回転設定が低く、固定5問と異なるため性能回帰比較には使わない。
