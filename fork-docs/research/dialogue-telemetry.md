# 対話の応答別計測

RTX分担はOFF。固定入力は `dialogue-benchmark-prompts.md` の5件を使用する。
`start-dialogue-measurement.ps1` は最新のRTX検証用バイナリを使用するが、RTX分担を有効にしない。本体には未統合。

## モード

| Mode | 用途 | CUDA Graphs | 注意点 |
|---|---|---|---|
| speed | 通常速度 | 有効 | 詳細テレメトリなし |
| telemetry | 通常経路の時間・回数・転送量 | 有効 | GPUイベント等の計測負担あり |
| components | CUDAの演算別詳細 | 無効 | 各演算または融合グループに同期を追加。通常速度や並行化効率の評価には使わない |

いずれもMTPは1候補生成・2トークン一括検証。コンテキスト2048、生成上限256、1スロット。
各モードで新しく起動し、5入力を順に送信して `/exit` で終了する。応答内容が異なる場合、後続の履歴も異なる点に注意する。
生ログとCSVは `G:\AI\llama-hetero-moe-local\logs` に保存し、Gitに追加しない。
起動時に `--simple-io --color off` を指定する。対話画面の色変更・スピナーが呼ぶ `common_log_flush()` はpause/resume方式で、並行して発生した計測ログを破棄し得るため、その表示処理を避ける。終了マーカーが欠けるログは完全な計測として集計しない。

## 収集内容

- 応答番号、task/slot ID、処理・キャッシュ済み入力トークン数、履歴長、生成トークン数、速度、MTP受理数、truncated。
- Prefill、本体Decode、本体MTP検証、Draft、Draft追従、Server全体の各段階。
- CUDAストリーム区間、CPU同期待ち、同期回数。
- componentsモードではAttention、Recurrent、Shared Expert、HC、Router、出力層、Routed Expert、PLE、other/mixed。既存の名前・重み分類に基づくため、全演算が専用カテゴリに分類されるとは限らない。
- Vulkan計測区間、コマンド記録、Submit、Fence待ち、読み書き等の呼び出し元別内訳。
- Expert要求・ミス、ユニークな転送Expert数の分布、アップロード量、動的昇格数、書き戻し量。

`responses.csv` は応答ごとの情報、`metrics.csv` は応答・段階・分類・計測項目・値を行形式で格納する。
単位は項目名の `_ms` / `_bytes` 等に従う。GPU区間とCPU待ちは重複するため合算しない。
Server全体は内側の段階を含む。転送量は転送時間ではない。GPU区間も純粋なシェーダ演算時間とは限らない。
`pending` / `dropped` が非ゼロならGPU時間が不完全な可能性がある。

## 実装

- `tools/server/server-context.cpp`: `server_decode_telemetry::scope` の外側server_step終了後に、応答ブロックを出力・リセットする。最後のステップを含め、出力時間はステップ時間の外に置く。
- `send_final_response`: 最終応答の統計を保存する。`LLAMA_DECODE_TELEMETRY_PER_RESPONSE=1` と1スロットが必要。複数スロットでは既存の累積計測を維持する。
- `src/llama-moe-cache.cpp` / `src/llama-ext.h`: 既存のミス計測有効時のみ、要求・ミス・昇格・書き戻し量のスレッド単位累積値を取得し、各段階で差分を取る。
- 異常終了・キャンセルの部分応答は通常の完了応答と同一の比較対象にしない。終了時に応答ブロック外で残余計測が出る場合がある。

## 検証範囲

MSVCで変更したC++ファイルの構文検査、PowerShell構文検査、応答ブロックとCSV集計の確認を実施する。
実モデルのビルドと5往復の実行はユーザー側で実施し、出力された5ブロック・サンプル欠落・有効設定を確認する。
