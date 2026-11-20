# 掲示板の速度改善アプローチ (2026-11-16)

調査のみ。報告値は投稿者の環境の結果で、当方では未再現。

- GPU側LRU: https://github.com/ggml-org/llama.cpp/issues/29949 。Expert IDを毎層CPUへ戻さず、GPU内でcache決定とfillを行うCUDA/HIP実装を提案。投稿者は7900 XTXで9.7から37-50 tok/sと報告。リンク先commit c81901a57の存在を確認したが全面コードレビューは未実施。MI50 Vulkanへの直接適用は不可。ピン留めRAMからGPUが読む方式でも、GPU内のExpert枠は入れ替わりうる。
- Vulkan readback集約: https://github.com/ggml-org/llama.cpp/discussions/31811 。GPUからCPUへ複数テンソルを戻す際の送信・fence待ち削減。R9700で約74から35 us/readbackという試作報告。対象はCPU MoEで、当方のCUDA/Vulkan bridgeとは同一ではない。既存get_tensor_asyncを活用する方向は参考になる。完成・採用済みとは扱わない。
- Vulkan pageable upload: https://github.com/ggml-org/llama.cpp/issues/20757 (martinalderson、2026-04-07コメント)。set_tensor_asyncでも通常RAMではstagingと同期が入り、SAM/ReBAR経路で改善した報告。9070 XT/Gemmaの条件でありMI50 Windowsの同等対応は未確認。今回のnative staging対策と同じ問題意識。
- CUDAコピー調整: https://www.reddit.com/r/LocalLLaMA/comments/1xx1owx/moe_offload_on_a_3090_prefill_345_466_ts_by/ 。RTX3090の35B Q8_0、prefill345から466 tok/sという報告。著者は単なるピン留めではなくpageable転送の調整と説明。コード https://github.com/Maknee/cudamemcpy-tuner は今回未レビュー。DecodeやVulkanへの効果を外挿しない。

ピン留めはRAMの扱いであり、GPU側の配置が固定か動的かとは別の話。mlockによるページ常駐とGPU APIのDMA用host allocation/registrationも区別する。

## 投稿の追加条件 (ユーザー提供、独立検証なし)

- フルQ8_0比KL=0.203は投稿の量子化構成の値。当方のUD-IQ3_XXSの値ではない。
- 投稿者は速度改良について出力一致を確認したと説明。
- Prefillは2枚が交互に待つ状態で調整が未完。
- RAM32GBは余裕が少なく、測定は特殊な1構成のみ。

従って投稿は全面的に並列な推論の証拠ではない。DecodeのMTP採用率、hot hit率、量子化、転送・同期・演算を分けて比較する。HC融合は丸め順変更を含むため、転送改善と分離して品質を評価する。
