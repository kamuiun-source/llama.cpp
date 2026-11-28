# RTX の Attention・Shared Expert・HC 内訳

CUDA コンポーネント診断を含むビルドを用意し、Windows PowerShell で実行します。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'G:/AI/llama-hetero-moe/fork-docs/scripts/measure-cuda-components.ps1'
```

既定の `-Mode both` は `baseline` と `diagnostic` の順で比較します。単独で実行する場合は `-Mode baseline` または `-Mode diagnostic`、生成上限を変える場合は `-Tokens 256` を指定します。

| 実行条件 | CUDA Graphs | 計測 |
|---|---|---|
| baseline | 通常の設定 | CPU 同期 API、MTP フェーズの wall、CUDA0 ストリーム全体 |
| diagnostic | `GGML_CUDA_DISABLE_GRAPHS=1` | 同じ計測に加え、`GGML_CUDA_PROFILE_COMPONENTS=1` で CUDA ディスパッチを分類 |

両条件で CUDA 融合、Expert バッチ 2、動的常駐、デバイス入力一括化、Router のホスト ID 再利用、Spin、Vulkan の遅延スロットマップと最終フェンス再利用を維持します。MI50 の GPU タイムスタンプ、スケジューラ詳細プロファイル、Expert 途中 Submit 集約、アップロード待機の後送り、MTP draft anchor 再利用は無効です。

## 分類と集計

`diagnostic.telemetry.json` の `target_verify.components` に分類ごとの数値を記録します。分類の単位は CUDA のディスパッチで、実行されるノードから分類します。View や実行されないノードは分類に使いません。Qwen4exp と共通処理のノード名を優先し、名前で特定できない場合は演算種別や入力重み名を使います。たとえば `attn_*` や Q/K/V 系のラベル、`FLASH_ATTN_EXT`、`.attn_q.` や `.indexer.` の重みは Attention の根拠です。Shared Expert の根拠は `ffn_shexp*` や `_shexp.` を含む FFN 重み、HC の根拠は `hc_*` や HC 重みです。 `GATED_DELTA_NET`、`SSM_SCAN` などの recurrent 処理は別分類です。

融合グループで識別できた分類が一致する場合は、識別できなかったノードもそのグループに含めます。分類が食い違う融合は `mixed`、根拠がないディスパッチは `other` に残します。

- `cuda_component_dispatches` は分類したディスパッチ数です。 `cuda_component_samples` は回収できた GPU サンプル数、`cuda_component_dropped` は回収できなかった数です。
- `cuda_component_stream_ms` は当該ディスパッチを囲む CUDA イベント区間の合計です。依存待ちや GPU のアイドルを含む場合があります。単独のカーネルの演算時間として扱えません。
- `cuda_component_cpu_submit_ms` と `cuda_component_cpu_wait_ms` はホスト側の提出と待ち時間です。GPU 区間や MTP フェーズの `wall_ms` と重なるため、加算しません。

診断側は CUDA Graphs を外し、ディスパッチまたは融合グループごとに同期して GPU イベント時間を回収します。Graph 無効化と追加同期で、ディスパッチの間隔や CPU 待機が変わります。診断側で得た分類比率を通常の Graph replay に当てはめることはできません。基準側のストリーム全体と wall は通常実行に近い値として保存し、診断側は負荷の大きい分類を絞る用途に使います。

ログは `G:/AI/llama-hetero-moe-local/logs/cuda-components-...` に保存します。設定、適用確認、フェーズ別 JSON、正規化した生成出力の一致結果を残します。CUDA 主分類のサンプルが欠けている場合は分類の網羅性も確認してください。

## Validation

Actual CUDA 13.1 compute_89 flags compiled the changed CUDA translation unit, and actual MSVC flags passed server syntax checking. The extracted classifier fixture checked explicit attention/recurrent ops, HC and attention weight separation, Shared Expert and Router separation, and mixed/other/view behavior. The script passed a Windows PowerShell 5.1 fake-CLI check for activation, JSON categories, output comparison and environment restoration. Real-model execution remains user-run. Raw logs are kept outside Git.
