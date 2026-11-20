# 調査メモと未解決事項

現行の優先順位は [現在の作業メモ](current-work.md)、運用設定は [フラグ一覧](runtime-flags.md)、経緯は [作業・決定履歴](coordinator-memory.md) を参照する。

## 再調査の起点

- ExpertのRAM内コピー、書き戻し、連続配置は [転送実装の比較](../research/vllm-strata-transfers.md) と [RAMコピー計測](../research/improvement1-upload-memcpy-results.md) を参照。短文の初期配置比較を行う場合は、転送件数と待ち時間を別々に記録する。
- 参考投稿の処理構成との差分は [デコード比較](../research/reference-decode-comparison-20261119.md)。Qwen3.8特化フォークからの候補は [実装調査](../research/qwen38-forks-decode.md) を参照する。
- Vulkan Submit・Fence待ちの読み方は [待ち時間の内訳](../research/vulkan-wait-breakdown.md)。固定3幅の後段Submit削減とAutoの修正結果は [作業・決定履歴](coordinator-memory.md) の「高速3幅の統合検証」に記録済み。

## Windows gfx906 / HIP

ユーザー訂正済み: MI50 32GBにVBIOSを適用、AMD Radeon Pro VIIとしてAMD純正ドライバーで認識。有志ドライバー使用ではない。Vulkanは動作済み、HIP未確認。

調査の起点:
- https://wikiwiki.jp/sd_toshiaki/Radeon%20ZLUDA
- https://github.com/brknsoul/ROCmLibs
- https://github.com/YellowRoseCx/koboldcpp-rocm/releases/tag/deps-v6.2.0
- https://github.com/YellowRoseCx/koboldcpp-rocm

確認済み: brknsoulのREADMEは特記なしの配布物をHIP5.7.1用としている。WikiはROCmLibs.7zのgfx906収録を紹介しているがアーカイブ内部未検証。koboldcppのWindowsビルド例にgfx906あり。6.2依存Release本文のGPU一覧にはgfx906がないが、標準SDK収録のため省略か未対応かは未確定。
ローカルggml-hip/CMakeLists.txtはHIP6.1以上を要求。5.7.1用DLLをそのまま採用する結論は不可。
ユーザーは別のChatGPTへGitHub調査を依頼するための依頼文を受領。回答待ち。WindowsネイティブHIP、ZLUDA、Linuxでの実績を混同しない。同一プロセスCUDA+HIP、現行転送のHIP移植、ドライバー相性は別検証。

## モデル対応

通常版で整備した後の候補: https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
同じ名称でもGGUF tensor構成、量子化、MTPとの組み合わせを未確認のまま互換としない。標準モデルから既に切り替えたとは扱わない。

## 測定の解釈

- 参考ツイートの1層0.87msと70〜106 tok/sは条件・MTP・ランタイム不明部分あり。積み上げだけで自環境の上限を保証しない。
- 時間の大きいotherを単一処理扱いしない。名前のないHC Scale/SiLUもotherに入る。
- HC upは2tokenを既に単一kernelで処理する。二つの独立matmulを初めてまとめる改善ではない。
- HC upの新実験はwarp配置変更。F32 reduction順序の差はあり得る。主モデルの生成とMTP受理率はユーザー比較待ち。

## メモ更新方法

新しい調査では「日付・根拠URLまたはコード・確認事実・推測・未検証・次の判断」を短く追記する。採否や優先順位が変わったら現在の作業メモを更新する。恒常的な進め方の変更はagent.mdに反映する。生ログやJSONLはコピーしない。
