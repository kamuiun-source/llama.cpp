# 調査メモと未解決事項

更新: 2026-11-17。確定ルールは `../agent.md`、再開時の作業状態は `coordinator-memory.md`。

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

新しい調査では「日付・根拠URLまたはコード・確認事実・推測・未検証・次の判断」を短く追記する。決定になった事項はagent.mdへ反映し、古い仮説は撤回済みと示す。生ログやJSONLはコピーしない。
