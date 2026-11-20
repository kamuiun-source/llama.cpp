# vLLM・StrataのGPU/RAM転送調査

2026-11-17。公開mainと公式技術資料を参照。比較対象はFlash-NextをRTX/CUDAとMI50/Vulkanで実行する現行フォーク。

- [vLLM PrefetchOffloader](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/offloader/prefetch.py) は、既知の後続層の重みをpinned RAMから専用ストリームで固定GPUバッファに先読みし、利用側は層単位のイベントで待つ。バッファ再利用とCUDA Graph captureも考慮する。[UVAOffloader](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/offloader/uva.py) はGPUから見えるCPUメモリのビューを使う別経路。PCIe上の読み出しは残る。どちらも、次層Routerの選択を予測する機能ではない。
- [vLLM DeepEP V2](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py) はトークンとtop-k情報をdispatchし、常駐Expertの出力をcombineする。decodeのCPU同期省略経路と、通信ストリームからCUDAイベントで合流する経路がある。通信中にShared Expertを実行できる。CUDA＋Vulkanへの直接流用は未確認。
- [Strata RemoteExperts](https://github.com/Niko1221/Strata/blob/main/src/core/remote_experts.cpp) は小さなpinned領域を再利用し、メタデータをまとめ、ヘルパーGPUでExpertをまとめて計算する。最適化経路はヘルパー上で重み付き部分和にしてから返す。現行フォークも `ffn_moe_out` を返すため、この削減は既に類似。Strata公式は[同一モデルの異種GPU層分割](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)と[別プロセス間のPrefill/Decode分離](https://github.com/Niko1221/Strata/blob/main/docs/SPLIT_SERVE.md)を区別する。
- [Strata派生版のLinux CUDA＋HIP helper](https://github.com/alizand1992/Strata/blob/main/src/core/external_experts.cpp) はmemfdを共有し、CPUの要求・応答通知でGPU間をつなぐ。ソース上は共有領域への入力と応答回収にCPU memcpyも残る。[作者報告](https://github.com/Niko1221/Strata/issues/3127)の63–74 tok/sは別GPU、MTP設定、CPU Expert併用、SSD読み出しが減る条件の値。Windows/CUDA＋VulkanやMI50では未検証。
- [Strataの昇格時のRAMバッファ回転](https://github.com/Niko1221/Strata/blob/main/docs/EXCHANGE_ROTATION.md)は、GPUから退避した先の一時領域をそのままRAM側の所有スロットにする。RAM内の最終memcpyを削減し、GPUの書き戻しと動的配置を維持する。全対象がpin/map可能な条件とサイズ別の予備領域が必要な実験。現行フォークの `commit_packed_promotion()` は対応するmemcpyを各重み面で行う。現状のVulkanステージングは使い回すので、所有権の移動には寿命管理が必要。

Windows側では、通常設定を維持して昇格の最終memcpy累計と、キャッシュミスなし時を含む層間受け渡しの露出時間を別々に測る。昇格memcpyが十分大きければバッファ回転を、受け渡しの固定費が大きければ小さな共通ホスト領域を実機で検証する。共有領域の対応確認と、生産完了・可視性・再利用の順序は別途必要。Readback順序変更は有効でも23.71 tok/s、既存OFFは24.39 tok/sだったためOFFを維持する。
