# 固定3幅のMI50戻りで実CUDA Graphを先行投入

対象は単一バンク・単一コピー・固定3幅のマップ済みRAM戻り。Shared Expert先行部分の後、同じCUDAストリームにホストチケット待ち、30,720 B H2D、既存のH2D完了イベント、戻りを使う実CUDA Graphを順に投入する。ホストは従来のHIP同期とCPU Expert補正を済ませてから別のピン留めページ上のチケットを更新する。遅延Router、後続コールバック、次の戻り枠再利用は公開後に進む。HIPが前段入力のマップ済みRAMを読んでいる場合、同じHIP同期が次のCUDAミラー書き込みを保護する。

適用条件はAda CUDAの主ストリーム、安定した既存の単一CUDA Graph実行インスタンス、正確なグラフUID・出力モード・ミラーアドレスの一致。グラフcapture、プロファイルによるCUDAストリーム計測、先行ホスト読み戻しと競合し得る構成、複数コピー・複数ストリームは通常経路へ戻る。CUDA 12.9以上のビルドで実際のドライバAPIを取得する。チケット投入後にCUDAの完了待ちは行わず、投入失敗時はチケットを更新してからCUDAストリームを排出する。CPU補正の式と計算順序は共通。

起動時に `LLAMA_HIP_MTP3_RETURN_TICKET_MODE_FILE` にパスを設定する。内容は `RETURN_TICKET=off` または `RETURN_TICKET=on`。既定OFF。サーバーの全スロットが空いた後、次の生成要求開始時に読み取る。ウォームアップ後に終了ログ `packed HIP MTP3 mapped return ticket: applied=... fallback=...` を確認する。`GGML_SCHED_TRACE_EXPERT_HANDOFF=1` の対象グラフでは `host_dual_pinned_mapped_return_ticket_h2d` と `hip_return_ticket` 印でCUDA Graph投入を確認できる。

同一ビルドのOFFとONで固定3幅の通常生成を比較する。NsightではH2Dの実終了と消費側CUDA Graphの実開始の間隔、MI50側同期、Shared Expert、CPU補正、後続Routerを照合する。実機の正確性と速度は未確認。
