# RTX 4070 Ti と MTP の内訳計測

CUDA イベント計測を含むビルドを用意し、Windows PowerShell で実行します。既定の生成上限は 256 トークンです。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'G:/AI/llama-hetero-moe/fork-docs/scripts/measure-cuda-breakdown.ps1'
```

既定の `-Mode both` は CPU 内訳のみの `cpu` と、同じ構成で CUDA イベントを追加した `gpu` を順に実行します。単独確認は `-Mode cpu` または `-Mode gpu`、生成上限の変更は `-Tokens` を指定します。CUDA イベントの設定は `gpu` の `GGML_CUDA_GPU_TIMESTAMPS_DEVICE=CUDA0` だけです。両条件で MI50 の GPU タイムスタンプは無効です。

両条件で Spin、Router のホスト ID 再利用、Expert 転送バッチ、動的常駐、デバイス入力一括化、Vulkan のスロットマップ遅延と最終フェンス再利用を有効にします。Expert の途中 Submit 集約、Upload 待機の後送り、MTP の draft anchor 再利用は無効です。スクリプトは実際に適用された機能のログを照合し、両条件の生成出力を ANSI 表示とプロンプト表示を除いて比較します。元の環境変数は終了時に復元します。

## 出力

ログは `G:/AI/llama-hetero-moe-local/logs/cuda-breakdown-...` にだけ保存します。`*.telemetry.json` にはフェーズ別の測定値、`*.settings.json` には環境設定、`*.activation.txt` には適用確認、`output-comparison.txt` には両条件の出力比較結果が入ります。

- `wall_ms` は各フェーズのホスト経過時間です。まず `target_verify`、`draft`、`draft_catchup` を確認します。 `draft` は生成、`draft_catchup` はドラフト側の状態追従です。 `server_step_total` はこれらを含む外側の区間で、チェックポイントやサンプリングなども含みます。
- CUDA の `cuda_stream_synchronize_cpu_ms`、`cuda_event_synchronize_cpu_ms`、`cuda_device_synchronize_cpu_ms` は、ホストが対応する同期 API から戻るまでの時間です。 `cuda_stream_wait_event_cpu_ms` はイベント待ちをストリームへ登録する API のホスト時間です。その待ちに対応する GPU 経過時間を示す値ではありません。
- `cuda_gpu_stream_span_ms` は、CUDA バックエンドがグラフの capture / launch または eager dispatch に入る直前から、実行コマンドを登録し終えるまでを、同じ CUDA0 メインストリーム上のイベントで囲んだ区間の合計です。合流する補助ストリームの処理、依存待ち、capture やコマンド供給中のアイドルを含み得ます。グラフ投入前のスケジューラ転送と、合流しないストリームは対象外です。
- イベントの測定値は登録時のフェーズに帰属します。回収は非ブロッキングで、後のフェーズや終了処理で済んでも帰属は変わりません。 `cuda_gpu_stream_span_dropped` はプール枯渇や API 失敗などで失ったサンプル、`cuda_gpu_stream_span_pending` は終了時点でも未完了のサンプルです。正のサンプル数を確認し、欠落があれば測定できた範囲として扱います。

GPU 区間、CUDA API のホスト時間、フェーズ wall は重なります。これらを足してステップ時間を求めることはできません。 `cpu` と `gpu` の wall を比較し、イベントを追加した際の計測負荷も確認します。

## Validation scope

The changed CUDA translation unit compiled with the actual CUDA 13.1 / compute_89 build flags, and server-context.cpp passed MSVC syntax checking with its actual include settings. Extracted event-pool tests covered delayed completion across phases, bounded-pool exhaustion and reuse after completion. The script passed Windows PowerShell 5.1 parsing and a fake-CLI workflow check, including environment restoration and output comparison. Full-model execution, real CUDA capture/replay timing and instrumentation overhead remain user-run checks. No raw logs are stored in Git.
