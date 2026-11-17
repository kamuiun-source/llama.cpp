# Vulkan の 2 トークン検証で IQ4_NL の重み復号を共有

`LLAMA_EXPERT_VK_PAIR_IQ4_NL=1` で試せる。対象は Vulkan の IQ4_NL 重み、F32 入出力、2 トークン、各 Expert 順位に個別の入力行を持つ、追加演算を融合していない `MUL_MAT_ID`。Flash-Next の down 投影 48 層中 45 層でこの量子化型を使う。実際の適用は各投影の入力形状、融合状態、活動マスクにもよる。Full-hot 経路では hot と cold の各位相で個別に選ぶ。片方のマスクが空なら既存の単列 shader を使う。通常経路の down に後続の scale や bias が融合されている場合も既存経路になる。

両行で同じ物理 Expert スロットを選んだ場合、選択順位が異なっていても同じワークグループで重みを一度復号する。各トークンの入力と積算値は独立し、結果は元の順位に書く。同じ行の中に同一 ID が複数あれば先頭の活動位置をペアにし、残りは単列として計算する。`-1` は従来どおりゼロを出力する。hot と cold をまたぐペアは作らない。ペア判定は GPU 上で完結する。

ビルドと比較はメインのチェックアウトで実行する。

```powershell
cmake --build .\build-win-cuda-vulkan --target llama-cli
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\fork-docs\scripts\compare-shared-expert-overlap.ps1 -PairIQ4NL -Mode speed -Tokens 256
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\fork-docs\scripts\compare-shared-expert-overlap.ps1 -PairIQ4NL -Mode telemetry -Tokens 256
```

両ケースで Shared Expert overlap、full-hot、動的常駐、upload/promotion overlap を有効にし、RTX cold offload を無効にする。設定と各比較結果は `G:\AI\llama-hetero-moe-local\logs` に記録する。スクリプトは生成出力、MTP の受理数、Expert のヒットや昇格、転送バイトが一致することを確認する。`expert IQ4_NL pair shader dispatches` は新 shader を起動した回数で、一致したスロット数は示さない。処理速度は speed モードの生成結果で比べる。

MI50 の単独投影テストでは、2 トークン × 10 選択位置で、重複なし・順位を逆にした全重複・部分重複・同じ行の重複・`-1` の出力が OFF/ON でビット一致した。1 トークン、入力が Expert 順位にブロードキャストされる場合、scale が融合される場合は従来経路で一致した。GGUF から切り出した Expert の full-hot と promotion のテストでも、scratch 使用有無の両方で OFF/ON の出力ハッシュが一致した。モデル全体の生成速度と出力は未計測。
