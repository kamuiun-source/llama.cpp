# Qwen3.8 Flash-Next向けforkの高速化調査

2026-11-20。対象はWindowsのRTX 4070 Ti（CUDA）＋MI50（Vulkan）、UD-IQ3_XXS、単一対話、コンテキスト2K。

| 参照先 | 確認した施策 | このforkでの優先度 |
| --- | --- | --- |
| [flashnext-hybrid](https://github.com/ucicelos/flashnext-hybrid#4-fix-2-a-recurrent-state-that-could-not-roll-back) | Recurrentの複数の履歴位置を保存し、棄却後にGPU側の状態を復元。APUではホストキャッシュ付きマッピングで境界の読み出しも改善。 | 3幅の棄却後に本体再実行やRecurrentの状態コピーが残るかを診断。APUのマッピングは同じ物理DRAMを使うため、その速度差をMI50に転用した見積もりはできない。 |
| [Moe-llama.cpp](https://github.com/ByronScottJones/Moe-llama.cpp/blob/main/docs/Qwen3.8-Flash-Next.md) | Metalの演算別計測で畳み込み状態の書き戻しを特定。下書き量子化の比較では24KでQ4_K_M 26.16、Q8_0 24.80 tok/s。Q8_0の受理率の方が高かった。 | 状態コピーを現行の直接履歴書き込みと照合する。3幅が安定した後、必要なら日本語・コードで下書き量子化を比較する。 |
| [jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash/commit/4bf80db6d) | QSAで選択したブロックに限って物理セルへ展開。 | 深いコンテキスト向け。2Kの最適化からは独立して評価する。 |
| [llama-cpp-perf](https://github.com/mateusoro/llama-cpp-perf) | Routerのプロファイルを入力にしたExpert常駐と、CPU側cold計算をGPU側hot計算と並行。 | 同じVRAM割り当てで、初期配置を汎用プロンプトと実利用の日本語・コードで比較する方式を参考にできる。既存のMI50転送並行化と実行デバイスが異なる。 |

ローカルでは `src/models/qwen4exp.cpp` に `build_conv_state_at` と直接履歴書き込みの経路がすでにある。3幅の比較では、第1候補・第2候補の棄却位置ごとに残る状態コピー、ターゲット再実行、同じ確定列での次のlogitsを確認する。公開forkの別GPU・別量子化での速度は参考値。
