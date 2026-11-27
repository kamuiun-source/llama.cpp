# 固定3幅のMI50→RTX戻りでRTX宛先同期待ちを分離

2026-11-27。試作の変更範囲はマップ済みRAM経由の固定3幅戻り。既定OFF。先行の `codex/rocm-return-ready-event` はMI50側の完了をイベントへ変更した。今回はMI50の同期を維持し、RTXの宛先VRAMに前の処理が触れ終わる順序をCUDAストリームで保証する。[先行試作の実測](../rocm/latest-performance.md#最新の比較試作)

- 単一コピー、スケジューラの宛先イベントなし、宛先splitにカスタム処理なし、CUDAとROCmの各バックエンドは1個、残りはCPUに限る。CUDA側は標準の主ストリームを選択中で、並行グラフの定義と追加ストリームがなく、H2D完了イベントを事前確保できる場合だけ適用する。その他は既存経路にフォールバックする。
- 前の処理とShared Expertの先行部分、戻りH2D、後続のCUDAグラフ、遅延Routerは同じCUDAストリームで順序づける。HIPはマップ済みRAMへ出力する。CPU補正はHIP完了後、H2D発行前に同じRAMへ加算する。MI50が次にRAM枠を上書きする前に前回のH2D完了を確認する。既にCUDAストリームを同期した場合は、その同期済み状態を使う。
- 起動時に `LLAMA_HIP_MTP3_RETURN_DST_WAIT_MODE_FILE` をモードファイルのパスに設定する。内容は `RETURN_DST_WAIT_SKIP=off` または `RETURN_DST_WAIT_SKIP=on`。全スロットが空いた後、次の生成要求の開始時に変更を読み取る。モデル再ロードやCUDA Graph再captureは不要。同じサーバーで比較する場合はファイルを置き換え、各設定のウォームアップを揃える。
- 終了時の `packed HIP MTP3 mapped return waits` はモード別に適用・フォールバック、MI50転送元の待ち、RTX宛先の1回目・2回目、RAM枠再利用のイベント待ちと既存ストリーム同期による完了を集計する。RAMイベントの時間は `GGML_SCHED_HIP_HANDOFF_WAIT_TRACE=1` の回で計測する。`GGML_SCHED_TRACE_EXPERT_HANDOFF=1` の対象グラフでは適用経路を `host_dual_pinned_mapped_return_ordered_h2d` と表示する。
- 実機の完了条件は固定5問・各512上限・同一ビルドで、適用回数、転送元同期、RAM枠再利用、CPU補正とShared Expert、遅延Routerを確認し、生成ID・MTP受理件数・本体速度を比較する。GPU実行と速度は未検証。

ローカルではスケジューラとサーバーのC++構文、実CUDAヘッダーを使った主ストリーム判定の小規模コンパイル、モードファイルのOFF→ON→OFFと不正値保持を確認した。
