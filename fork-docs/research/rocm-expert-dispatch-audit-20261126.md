# 固定3幅のExpert配置と実行経路の再調査

2026-11-26。RTX 4070 Ti＋MI50のQwen3.8 Flash Next、単一バンク・固定3幅が対象。通常の統合設定と、`cafe438d9` のIQ3_XXS重み共有試作を、Router確定後からMI50のRouted Expert実行まで照合した。以下はソース確認と保存ログの再集計。新規の実機計測は行っていない。

## Expert選択結果を使い回す

- 重み共有ONは、論理IDと配置が確定してからも、各1,600ブロックの3 waveで30枠の照合を繰り返す。重複なしで132.80→148.18 µs、3位置すべて共有で131.57→143.34 µs。gfx906コードオブジェクトの実確保VGPRは64→72。同じExpertの内積を1 waveへ寄せるため、判定の前倒しだけではレジスタと並列度の問題は残る。[従来の試作とAstraのレビュー](../rocm/09-single-bank-mtp3.md#astraのレビュー)
- 採用中の経路にも、30個の選択IDを処理するホスト側で512要素の論理ID→物理スロット表を発行し、GPU側の `GET_ROWS` で30要素へ絞り直す箇所がある。固定幅の生成では30枠の物理番号と共有先の位置・順位を同じコールバックで組める。物理番号だけで同じExpertとみなさず、CPU行きは共有グループから外す。ゼロ枠は異なるExpertが参照し得る。現行の非同期入力スケジューラは `GET_ROWS` とスロット表の対応で対象を検出するため、その検出と共有RAM入力を保持して比較する。[スロット表生成](../../src/llama-moe-cache.cpp#L3785) · [GPU側の取り出し](../../src/llama-graph.cpp#L2825) · [非同期入力の適格条件](../../ggml/src/ggml-backend.cpp#L3364)
- TensorRT-LLMには、適格な経路でRouterが後段に渡す並べ替え表も作る実装がある。Expertのスロットを後で変更する経路ではこの融合を使わない。vLLMには、活性トークンが少ないときに並べ替えとパディングを省き、対応Expertを直接参照する経路がある。こちらで取り入れるのは固定30枠の仕事表で、汎用ソートや追加の転送・起動を前提にしない。[TensorRT-LLMのMoE実装ガイド](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/moe/fused_moe/MOE_DEVELOPER_GUIDE.md) · [vLLMの実装](https://docs.vllm.ai/en/v0.23.0/api/vllm/model_executor/layers/fused_moe/fused_moe/)
- Castagna Veloceのgfx906向け `moe_coal_iq` は位置と順位をまとめた1枠をブロックに割り当て、LDSの表と量子化済み入力を複数出力行で再利用する。同じExpertの位置を一つのwaveに集める処理は、このカーネルにはない。MonoMoEは少トークンの量子化デコードで、出力行のタイルから並列性を確保し、短いトークン軸を演算に残す。H200のFP8 Tensor Core向けなので、対応するgfx906 IQ3_XXSの性能は別途評価する。共有を続ける場合は、位置別の内積の並列性を維持し、従来の4行経路と比べる。[Castagna Veloceのgfx906実装](https://github.com/benpeterson40/castagna-veloce/blob/3483d46/ggml/src/ggml-cuda/moe-vec.cu#L1247) · [MonoMoE §3.1](https://arxiv.org/html/2609.04244v1#S3.SS1)

## 採用経路に残る転送と計算

1. **次層Router予測IDの読み戻し。** 通常設定のTop-10×3位置の120 Bは、RTXからスタック上の配列へ `cudaMemcpyAsync` 相当で読み戻し、直後にproducerを同期する。ウォームアップ込み560検証の保存ログでは26,320回、呼び出し内1,064.815 ms、呼び出し後の明示同期14.276 ms。検証あたりそれぞれ1.901 msと0.025 ms。pageable宛ての非同期コピーが呼び出し中に待つことがある。ここは小さなピン留め受信枠に変更し、現在層のMI50計算開始後に発行する既存の順序を保ったまま、RTX側のコピー完了イベントとCPU消費を分ける候補。削減見込みはタイマーの移動では判断せず、MI50が進行中の区間と本体壁時計を照合する。[読み戻しと同期](../../src/llama-moe-cache.cpp#L2717) · [CUDAの同期仕様](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
2. **先読みがヒットしたExpertの移し替え。** 必要分の最後のイベントを待つ位置はすでに改善済み。ただし採用経路でも、先読みバンクから本計算の仮置き枠へヒットごとにD2Dが残る。同じ保存ログは9,358ヒット・17.82 GiB。これは転送量の記録で、短縮可能な壁時計時間は未計測。先読み用と仮置き用を同じ物理枠のプールにし、ヒット時に枠の用途を入れ替えれば、GPUが同じ場所を読める。入力が確定する前後の書き込み先と、DMA・Expert計算・昇格がその枠を最後に使う時点の管理が必要。ProMoEは実Routerの結果で既存のキャッシュを検索し、必要な転送を優先する。本フォークの実ミス優先は採用済みで、今回残った差分は同じ物理データをそのまま参照できるかどうか。[ヒット後のD2D](../../src/llama-moe-cache.cpp#L2945) · [ProMoEのキャッシュと転送順序](https://arxiv.org/html/2410.22134v3#S4)
3. **CPUに回したExpertのゼロ枠。** GPU用スロット表ではCPU行きのルートをゼロ重みの枠へ向ける。現在のExpertカーネルはその枠も通常の行列積に通す。位置・順位表にCPU扱いを持たせる場合、出力を規定値に埋めて計算を省ける余地がある。gate/upとdownの両方でゼロ出力とCPU補正の順序を確認する。ゼロ枠がある間もブロック数は固定なので、削減できるのは該当waveの仕事。効果は未計測。[CPU割当時のゼロ枠](../../src/llama-moe-cache.cpp#L3636) · [ゲート・内積の実行](../../ggml/src/ggml-cuda/mmvq.cu#L1224)

保存ログは `/home/t-saka/rocm-mtp3-catchup-adopted-20261126/session-batch-01/server.log`。基準の通常実行は58.61 tok/s・本体33.432 ms/回。戻り側の待ちにはMI50計算を含み、完了イベントだけに置き換えた試作は同条件で効果が出なかった。今回の優先順位は、次層Routerの読み戻し、共通の仕事表を使った固定幅の比較、D2Dなしの物理枠再利用。各変更のON/OFFは単独で評価する。[通常設定と戻りイベントの計測](../rocm/latest-performance.md)
