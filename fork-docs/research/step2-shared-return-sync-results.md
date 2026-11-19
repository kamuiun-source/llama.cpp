# MI50からRTXへの戻り同期（2026-11-19）

**決定：優先度は低。実装・追加計測は保留。** 1枠イベント方式で短縮できるCUDAの空き時間は、実モデルでは未計測。

- `ggml_backend_sched_compute_splits` はVulkanの読戻し完了を待ち、共有RAMからCUDAへH2Dを積んだ直後にCUDAを同期する。対象データはRouted Expertの結果。CUDAのShared Expert prefixは既に投入済み。H2DとCUDAイベント記録は同じストリーム。再利用・解放前の完了保証が必要。
- 実機ログ `dialogue-speed-ram-timing-20261119-105746` は全フェーズ計41,939回、858,910,720 bytes（20 KiB/回）。検証だけの回数・同期時間・GPU空転時間は未計測。
- 過去の2枠実験 `shared-host-return-slots-check/fixture-rounds.txt` は456回の戻りで即時同期待ち1.63～1.94 ms。2枠ONで即時待ちは消えたが、次の入力のCUDA待ちは各ペアとも増えた。速度用 `fixture-speed-rounds.txt` は4組とも壁時計中央値が悪化（約8～13%）、出力ハッシュ一致。1枠イベントのみの実モデル比較は未実施。
- 再検討する場合、target_verifyに限定し、即時同期と次の入力同期、イベント記録・再利用待ち、検証全体の時間を対応付ける。出力一致のみでは非同期バッファ再利用の安全性は証明できない。
