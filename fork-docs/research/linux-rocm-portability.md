# Vulkan Expert 実装の ROCm 移植調査

調査日 2026-11-20。比較はローカルの `origin/master` と `codex/mtp-three-integrated-validation`。共通祖先は `827466b9e`、フォーク側は `82c2d65db`。この比較点からフォークに 27 コミット、160 ファイルの差分があり、うち Vulkan 配下は 28 ファイル（+2,968 / -168）。未コミットの CUDA/HIP ビルド互換修正はこの集計に含めていない。

## 移植範囲

| 実装 | 現状 | ROCm での方針 |
| --- | --- | --- |
| Routed Expert の量子化演算 | CUDA と HIP は同じ実装を別々にビルドする。`MUL_MAT_ID` の MMVQ に IQ3_XXS と IQ2_S の分岐がある。形状によって別カーネルや同期を伴うフォールバックを選ぶ。 | 既存カーネルで開始する。Flash-Next の実テンソル型と形状で、CPU 参照との誤差と選択経路を確認する。 |
| Packed Expert の配置と同期転送 | 初期の同期実装 `ef2a53c6a` はデバイス共通のバッファ割り当てを利用。現行でも GPU 選択は汎用だが、動的常駐は Vulkan の登録名で有効になり、複数トークンには動的常駐が必要。 | 現行の配置・スロット管理を再利用し、まず MI50 で常駐ヒットと同期ミスを検証。次に昇格・退避・再読込を検証し、対応済みの機能だけを HIP で有効にする。 |
| GPU 間の Expert 入出力 | 共有 RAM import、バッチ転送、Shared Expert の先行投入は scheduler の CUDA/Vulkan 判定と Vulkan のフックに依存。 | 最初は既存の CPU RAM 経由の同期コピーを使う。同一プロセスで両方向を照合した後、再利用ステージングとピン留めを個別に評価する。 |
| MI50 内でアップロードと演算を並行化 | Vulkan のバッファ割り当て、転送サブミット、slot map 遅延、書き戻し完了処理に依存。広幅 prefill も専用アップロードフックを使う。 | HIP の転送ストリームと同一ランタイムのイベントで再設計できる。転送元・転送先の寿命と完了順序を維持する。上書きは先行計算と書き戻しが完了してから行い、完了確認後にスロットの所有権を確定する。実機の転送・演算重複は計測が必要。 |
| gate/up＋SwiGLU、Router 重み付き reduce、サブミット集約 | Vulkan の GLSL とグラフ照合、コマンド記録を使用。 | 正確性を確認した後、差が大きい処理だけ HIP の既存の量子化カーネルとグラフ融合に合わせて実装する。重み付き逐次加算は順序を維持し、固定2幅と現行の固定3幅を個別に検証する。 |

ソースの確認箇所は [モデル側の GPU 選択](../../src/llama-model.cpp#L1794)、[配置の制約](../../src/llama-expert-placement.cpp#L61)、[転送・昇格フック](../../src/llama-moe-cache.cpp#L749)、[Vulkan 用 scheduler 判定](../../ggml/src/ggml-backend.cpp#L2336)、[HIP と共有される MUL_MAT_ID](../../ggml/src/ggml-cuda/ggml-cuda.cu#L2564)、[IQ2_S / IQ3_XXS の MMVQ 分岐](../../ggml/src/ggml-cuda/mmvq.cu#L1448)、[Vulkan の融合](../../ggml/src/ggml-vulkan/ggml-vulkan.cpp#L13463)。

## CUDA/HIP 共存時の境界

CUDA と HIP のプラグインは共有ソース上で同じ GUID を返すため、`ggml_backend_is_cuda()` の結果だけではランタイムを区別できない。現行の直接コピーはさらにバッファが自分のプラグインのものかを照合し、異種バッファを拒否した場合は ggml の同期ホストコピーに落ちる。一方、既存のイベント待ちは相手イベントのランタイムを照合せずコンテキストをキャストする。移植経路で CUDA イベントを HIP に、またその逆に渡さない。最初は転送元の完了をホストで待ってから転送先に投入する。両ランタイムへの同一ホスト領域の登録と VRAM 間 P2P は実機で確認する項目として扱う。

[GUID とバックエンド判定](../../ggml/src/ggml-cuda/ggml-cuda.cu#L6416) · [バッファ型の照合](../../ggml/src/ggml-cuda/ggml-cuda.cu#L988) · [イベント待ち](../../ggml/src/ggml-cuda/ggml-cuda.cu#L5982) · [同期ホストコピー](../../ggml/src/ggml-backend.cpp#L869) · [HIP の非同期転送とイベント](https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/asynchronous.html)

## 検証順序

1. CUDA0 と ROCm0 を同一プロセスにロードし、各 GPU の小規模演算と CUDA → RAM → HIP、HIP → RAM → CUDA のコピーを固定データで照合する。現状は同一 CLI からの列挙まで確認済み。
2. MI50 単体で既存の `test-backend-ops` を使い、IQ3_XXS の `MUL_MAT_ID` を CPU 参照と照合する。Flash-Next の gate/up と down の実形状は、独立した小規模 fixture で Vulkan と HIP を同条件測定する。
3. Flash-Next の Expert を MI50 の既存カーネルで実行し、RTX を残りの層に使う。常駐ヒット、同期ミス、昇格・退避・再読込の順に、出力とスロット状態を照合する。既存の Qwen3.6 単体対話は、このモデル固有の配置・転送の検証には含めない。
4. 正確性を確認した基準から、転送の重複、広幅 prefill、融合を個別に測る。Linux Vulkan/HIP の比較はソース、量子化、入力、MTP 設定を揃え、GPU 演算と GPU 間転送の時間も分けて残す。
