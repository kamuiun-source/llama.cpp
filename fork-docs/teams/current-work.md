# 親エージェントの再開メモ

更新 2026-11-20

開始・再開・コンテキスト圧縮後に読む。作業中の判断が変わったら該当箇所を更新する。詳細な作業履歴は [coordinator-memory.md](coordinator-memory.md)、技術調査と検証待ちは [research-notes.md](research-notes.md)、再現用の設定は [runtime-flags.md](runtime-flags.md) に残す。

## 現在の状態

- 当面の通常運用は高速固定3幅、全語彙、`p-min=0`。MTP改善はユーザー判断で休止。固定2幅と自動幅は比較用に保持。
- 日本語・コードは同じ推論経路と初期配置を使う比較用の入力。下記の初期配置改善は、普段の日本語・コード入力のExpert利用頻度から両方に使う配置表を1つ作る未着手の候補。
- ウォームアップ後の各2回を合算した最新比較は、日本語140入力トークンで固定2幅 32.69、自動幅 32.25、固定3幅 36.87 tok/s。コード188トークンでは同順で 39.75、38.46、43.89 tok/s。自動幅の3幅使用率は約5%。短文対話では固定3幅でも13.67〜25.67 tok/sで、固定2幅との同条件比較は未実施。根拠はGit外ログ `mtp-three-all-adopted-auto-warm-20261120-105728-8571f0ef/comparison.json` と履歴の「高速3幅の統合検証」。
- 環境はWindows、RTX 4070 TiをCUDA、MI50をVulkanで使用。検証モデルはQwen3.8-Flash-Next UD-IQ3_XXS、MTPは別のQ8_0。各層の常駐Expertは300、入れ替え枠は20。未キャッシュ入力80トークン以上で最大512幅のPrefillを使用し、広幅のExpert転送は10枠ずつ計算と重ねる。
- 直近の実測ソースは `C:/Users/kamui/.codex/worktrees/mtp-three-transfer-overlap/llama-hetero-moe`、対応するビルド先は `G:/AI/build-mtp-three-transfer-en`。ローカルで確認できたブランチ先端は `codex/mtp-three-integrated-validation` の `efc872328`。以後の採用差分を含むかは個々のコミットと作業ツリーで確認する。文書を編集している `rtx-cold-assist-v2` ワークツリーと実測ソースを区別する。

## 次に見る順序

「次の改善」を聞かれたらこの一覧と既存の調査資料を先に読む。実装有無、実測結果、採否、コミット・pushの状況を必要に応じて確認し、判断に足りない箇所だけコードやログに戻る。現在は候補整理の段階。

| 順位 | 改善候補 | 次の判断 |
| --- | --- | --- |
| 1 | 短文対話向けの初期Expert配置 | 常駐300・入れ替え20・動的配置を固定し、初期配置だけを公開ヒット上位300と日本語・コード向け候補で比較する。配置作成用とは別の入力で、序盤と後半のミス・転送量・初回応答と生成速度を見る。 |
| 2 | RAMからExpertアップロードまでのコピー | ミス時の待ちが大きい場合、RAM原本から直接DMAできる配置を検討。RAM内コピーの並列化は計測済みで優先度低。昇格時の先行書き戻しは、昇格後のクリティカルパスが短くなる根拠が得られた場合に検討。 [RAMコピー調査](../research/improvement1-upload-memcpy-results.md) |
| 3 | 全ヒット層のRTX↔MI50受け渡し | Expertミスの影響が小さければ、共通RAMの行き戻りを含めた固定費を確認。CPU計時の待ちとGPU演算の重なりを考慮して、残っているコピー・同期・Submitを発生条件ごとに追う。 [受け渡し調査](../research/gpu-expert-handoff.md) |
| 保留 | Expert先読み、RTX分担、IQ2_S/IQ3_XXS重み共有 | 先読みは予測精度と競合の確認から。RTX分担と重み共有には試作・実測がある。再試行には前回の制約を解消する根拠が必要。 |
| 検討のみ | Cold Expert直接読み出し、層またぎのVulkan記録再利用、MTPツリー、Hot ExpertのQ8化 | PCIe読み出し、層間のCUDA依存、状態管理、VRAM予算を先に確認。Linux ROCm Phase 0は [移植メモ](../design/linux-rocm-port.md) で別に扱う。 |

実装はSol、レビュー・分析・記録は親が担当。実機ビルドと計測をLunaに委任する場合は完了通知を依頼し、ON/OFF両順を逐次実行して完了時に出力を確認する。生ログは `G:/AI/llama-hetero-moe-local/logs` に置く。
