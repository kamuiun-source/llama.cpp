# Expert転送前のCPUコピー計測（2026-11-19）

**判断：並列コピーの優先度は低。実装・追加検証は保留。** 並列化の対象となるコピー時間は、今回の生成50.25秒のうち4.75秒だった。[元の提案](improvement1-upload-memcpy.md)

RTX 4070 Ti（CUDA）とMI50（Vulkan）で、Flash-Nextの6応答を `-Mode speed -ExpertRamCopyTiming` で計測した。MTP本体検証706回の経過時間は45.66秒。次の値は検証中にCPUがExpertの重みをstagingへコピーした実測値で、PrefillとGPUへのDMAは含まない。

| 対象 | コピー量 | CPUコピー時間 | 検証1回あたり |
|---|---:|---:|---:|
| 全stagingコピー | 49.86 GB | 7.29秒 | 10.32 ms |
| うちExpertを2個以上転送した回 | 32.63 GB | 4.75秒 | 6.73 ms |
| うちフォールバックで再コピーした分 | 0 B | 0秒 | 0 ms |

生成速度は24.44 tok/s。対象の4.75秒を25%短縮できた場合は25.03 tok/s（+2.4%）、半減できた場合は25.65 tok/s（+5.0%）と試算した。いずれも、短縮分がそのまま生成時間に反映される仮定。2スレッドでの実効帯域とワーカーの完了待ちを含む短縮率は未測定。

計測元はGit管理外の `G:/AI/llama-hetero-moe-local/logs/dialogue-speed-ram-timing-20261119-105746/dialogue.log`。集計には `expert_staging_copy: phase=target_verify` と各応答の `eval time` を使用した。
