# 固定3幅のGPU間受け渡しとExpert転送の公開実装

2026-11-27。対象はRTX 4070 Ti＋MI50、単一バンク、固定3幅、直接tuple DMAとCPU Expert自動配分の採用設定。直近トレースで、重み以外の受け渡しは両GPU・コピー非活動のうち2.516 ms/検証、そのうち同期APIは1.294 ms/検証。Expert H2D完了待ちは全問3.322、第4問9.317 ms/検証。値の区間は異なり、合計を削減幅には使わない。[計測](../rocm/latest-performance.md#d4採用後の統合版で検証を詳細計測)

## GPU間の受け渡し

- **vLLMの非連続コピー一括発行。** `simple_kv_offload/cuda_mem_ops.py` は宛先・転送元・サイズの配列を作り、`cuMemcpyBatchAsync` / `hipMemcpyBatchAsync` へ渡す。現在のフォークも3テンソルをスケジューラで1組として扱うが、バックエンドの汎用 `get_tensor_n` / `set_tensor_n` には個別コピーのループが残る。試すならCUDA側で同一依存・同一ストリームに載る複数の独立コピーがある区間だけ。固定3幅の活性値にはマップ済みRAMを使う経路もあるため、実際の発行件数を方向別に確認してから適用する。同一バッチ内にproducer/consumer依存を持つコピーを置かない。ROCm 7.2.1の実装は内部の `ihipMemcpy` を記述子ごとに呼ぶため、APIを1回にしてもDMA1回にはならない。[vLLM実装](https://github.com/vllm-project/vllm/blob/main/vllm/v1/simple_kv_offload/cuda_mem_ops.py) · [CUDA 12.9仕様](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-driver-api/group__CUDA__MEM.html) · [ROCm 7.2.1実装](https://github.com/ROCm/clr/blob/rocm-7.2.1/hipamd/src/hip_memory.cpp#L2588-L2621)
- **UCXの完了通知とMoE-Infinityのバッファ寿命管理。** UCXはCUDAのホストコールバックでeventfdを通知し、ROCmのコピーでは再利用できるHSAシグナルを使う。MoE-InfinityはCUDAの消費側ストリームにイベント待ちを積み、待ちを通過した後に完了イベントを回収する。今回に当てはめるなら、MI50→RTXの戻りでCUDA側の宛先を保護する待ちと、ピン留めRAMの再利用を保護する待ちを分ける。宛先の前回使用が同じCUDAストリームで完結する経路ではストリーム順序の利用を検討し、ホストが書き換えるRAM枠には前回のDMA完了を要求する。CPU Expert補正によるホスト書き込みを考慮する。以前の「MI50の戻り出力だけを待つ」試作と対象を区別して、方向・待ち理由を先に測る。公開コードはCUDA/HIP混在PCIeの低レイテンシを立証するものではない。[UCX CUDA通知](https://github.com/openucx/ucx/blob/master/src/uct/cuda/base/cuda_iface.c#L50-L74) · [UCX ROCmコピー](https://github.com/openucx/ucx/blob/master/src/uct/rocm/copy/rocm_copy_ep.c) · [MoE-Infinityのイベント消費と回収](https://github.com/EfficientMoE/MoE-Infinity/blob/main/core/parallel/expert_dispatcher.cpp#L2140-L2183) · [既存の宛先同期削減](../rocm/06-decode-transfers.md#06-1c-単独転送の宛先同期の削減)
- **HetCCLの適用範囲。** 公開構成はGPUベンダーの異なるノードをInfiniBand/RDMAで接続する。今回の1台・同一プロセスで各層の小さい受け渡しを処理する経路の根拠としては使わない。[公開実装と検証構成](https://github.com/mcrl/hetccl#tested-system)

## Expert転送

- **MoE-Infinityの計算窓ベースの先読み予算。** `OverlapBudgetController` は層の計算時間、実測帯域、キュー待ち、発行時間の移動平均と転送中のバイト数から、先読みを発行できるバイト数を決める。実Routerと候補の順位は共通にしたまま、8枠の候補の先頭から発行する分だけを層別に調整できる。現行の最大2件先行は維持し、まずトレース上で先読みを発行しない場合も含めた予算を評価する。採否は先読み発行件数ではなく、実ミス直前までに転送が完了したヒットの件数・バイト数と、実ミスDMA完了待ちで決める。[予算と計測カウンタの実装](https://github.com/EfficientMoE/MoE-Infinity/blob/main/moe_infinity/memory/overlap_budget.py#L74-L193) · [現行の実ミス優先](../rocm/09-single-bank-mtp3.md#promoe型の実ミス優先)
- **ProMoEの途中まで転送した重みの優先昇格。** キューは実ミスを先に取り、実ルートに入った転送中候補が未完了なら残りの部分を優先する。現在の採用経路には未発行の先読みの後回しとExpertごとの完了待ちが入っている。追加候補は、全体のtuple DMAを保ったまま、先読みが間に合った・間に合わなかった・先読み未発行の実ミスを分けること。Expert内のチャンク分割は長いDMAが実ミスを押し出していると分かった場合に限って先読み側から試す。通常需要側のDMAを細切れにする根拠は現時点でない。[優先キュー・途中転送処理のコード](https://github.com/promoe-opensource/promoe/blob/main/src/cpp_worker/prefetcher.cpp#L40-L145)

優先はExpert側の先読み発行予算を、同じ生成ID・Expert IDのログで評価すること。受け渡し側は512トークンの同条件で送受方向と同期理由を取った上で、RTX側の宛先待ちとピン留めRAMの再利用待ちを切り分ける。今回の作業は公開実装と既存計測の照合まで。
