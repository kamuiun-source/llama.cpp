# Linux ROCm 移植の引き継ぎ

対象は、Radeon PRO VII として認識されている MI50 32 GB（VBIOS 変更済み、gfx906）と RTX 4070 Ti の実機。Phase 0 では Linux 上の同じ MI50 で Vulkan と HIP の GPU 常駐 Expert 演算を比較し、HIP の正確性、動作安定性、速度差を確認する。Flash-Next のモデルロードは任意の動作確認とし、CUDA と HIP の混在や Expert キャッシュの移植は結果を見て段階的に判断する。

## 比較基準

| 測定 | 確認済みの結果・状態 |
|---|---|
| Windows、RTX CUDA + MI50 Vulkan、固定2幅 MTP | HC down/up/inject 有効。順次加算と Router 重み乗算の両融合を有効にして、ウォームアップ後の4回を合算した生成速度は **36.26 tok/s**。Router 重み乗算の融合を外した両順比較は 35.47 tok/s。生成 ID と MTP 提案・受理件数は両順で一致。 |
| Windows、以前の実対話 | **26.73 tok/s**。別の対話ワークロードの記録。 |
| Linux、同一 MI50、Vulkan と HIP の常駐演算 | 対応ケースの Vulkan・HIP とも計測待ち。 |

36.26 tok/s の条件と現行の機器構成は [作業方針](../agent.md) を参照。対話、固定プロンプトの生成、単一演算は計時範囲と分母が異なる。Linux の移植判断には、同じ Linux 上で採った対応ケースの差を使う。

## Phase 0 MI50 単体で動作と演算を比較する

### 環境の固定

- ROCm **6.3.1** を最初の検証候補にする。AMD の Linux 要件表では MI50 と Radeon PRO VII はともに gfx906 で、6.3.1 では既存機能を維持する限定サポート（Deprecated）。6.4.3 では両方が Unsupported。変更済み VBIOS を含む実機構成で、HIP ランタイムと必要なライブラリの動作を確認する。 [AMD ROCm 6.3.1 のシステム要件](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.1/reference/system-requirements.html) [AMD ROCm 6.4.3 のシステム要件](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/reference/system-requirements.html)
- 同じ実機・ブート環境で `lspci -nnk`、`rocminfo`、`hipconfig --full`、`vulkaninfo --summary` を取得する。PCI アドレスで両 API が同じ MI50 を指すことを確認し、gfx906、VRAM 容量、OS・カーネル、ROCm/hipBLAS/rocBLAS、Vulkan ICD、ドライバーと GPU 状態を記録する。初回は `HSA_OVERRIDE_GFX_VERSION` を設定せず、ネイティブの gfx906 で確認する。
- 両ビルドのソースリビジョンとローカル差分を揃え、同じホストコンパイラと Release 設定を使う。HIP の CMake は HIP、hipBLAS、rocBLAS を要求する。以下は `/opt/rocm` に 6.3.1 を配置した例。ビルド・計測結果の置き場には Git 管理外の絶対パスを使う。 [ビルド手順](../../docs/build.md#hip) [HIP の CMake](../../ggml/src/ggml-hip/CMakeLists.txt)

~~~bash
set -e
export ROCM_PATH=/opt/rocm
export CC="$ROCM_PATH/llvm/bin/clang"
export CXX="$ROCM_PATH/llvm/bin/clang++"
export HIPCXX="$CXX"
BUILD_ROOT="$HOME/llama-rocm-phase0-build"
LOG_ROOT="$HOME/llama-rocm-phase0-logs"
mkdir -p "$LOG_ROOT"

cmake -S . -B "$BUILD_ROOT/vulkan" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=ON -DGGML_HIP=OFF
cmake --build "$BUILD_ROOT/vulkan" --target test-backend-ops

cmake -S . -B "$BUILD_ROOT/hip" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx906
cmake --build "$BUILD_ROOT/hip" --target test-backend-ops
~~~

構成またはビルドが失敗した場合は、そのビルドの測定を止める。`rocminfo` の列挙に続き、実際の演算まで確認する。rocBLAS が gfx906 用ライブラリを読み込めるかは、初回のライブラリ使用時のログと成否で確認する。

### 既存テストで経路を確認する

`test-backend-ops` の `support` でデバイス名と演算サポートを確認し、`test` で CPU 参照と照合してから `perf` を測る。`type_a=iq3_xxs.*n_mats=4,n_used=4,b=0,m=16,n=10,k=256` は既存の IQ3_XXS ケース。これは経路の確認用で、Flash-Next の固定2幅の形状は次の手順で測る。`VK_DEV` と `HIP_DEV` にはログで同じ MI50 と確認できた名前を入れる。 [テストのケースと引数](../../tests/test-backend-ops.cpp)

~~~bash
FILTER='type_a=iq3_xxs.*n_mats=4,n_used=4,b=0,m=16,n=10,k=256'
"$BUILD_ROOT/vulkan/bin/test-backend-ops" support -o MUL_MAT_ID -p "$FILTER"
"$BUILD_ROOT/hip/bin/test-backend-ops" support -o MUL_MAT_ID -p "$FILTER"

# 変数 VK_DEV と HIP_DEV を実際のデバイス名に設定してから実行する。
"$BUILD_ROOT/vulkan/bin/test-backend-ops" test -b "$VK_DEV" -o MUL_MAT_ID -p "$FILTER"
"$BUILD_ROOT/hip/bin/test-backend-ops" test -b "$HIP_DEV" -o MUL_MAT_ID -p "$FILTER"
"$BUILD_ROOT/vulkan/bin/test-backend-ops" perf -b "$VK_DEV" -o MUL_MAT_ID -p "$FILTER" --output csv > "$LOG_ROOT/vulkan-smoke.csv"
"$BUILD_ROOT/hip/bin/test-backend-ops" perf -b "$HIP_DEV" -o MUL_MAT_ID -p "$FILTER" --output csv > "$LOG_ROOT/hip-smoke.csv"
~~~

ログで一致するケースが実際に実行され、`test` が PASS、`perf` がサポートありかつ正の測定値を出したことを確認する。既存 `perf` は GPU バッファ上でウォームアップし、複製したノードのグラフ実行を計時する。Expert ID は繰り返しの間に引き直すので、この計測は同一形状の予備値として記録する。

### 同一入力で代表形状を測る

1. 手元の UD-IQ3_XXS GGUF から Routed Expert の gate/up と down の型、内積長、出力幅、重みストライドを確認する。配布名に含まれる量子型と各テンソルの実型を区別する。重みは同一バイト列、入力と ID は同一の固定データを使う。モデル由来のバイト列を抽出できる場合はそれを使用し、合成データを使う場合は生成手順・シードとハッシュを記録する。
2. その形状を使い、2トークン・各10 Expert の `MUL_MAT_ID` を GPU 常駐バッファで測る。最初は同一 ID 重複条件を両バックエンドで揃え、必要なら重複が少ない例と多い例を別に比較する。gate/up は入力を broadcast する形状、down は Expert ごとの活性を読む形状を確認する。既存テストに該当形状や固定入力がなければ、同じ ggml グラフを構築する小規模 fixture を別作業で用意する。
3. 両 GPU の出力を CPU 参照と照合し、最大絶対誤差と NMSE を記録する。既存 `MUL_MAT_ID` テストの既定 NMSE 上限は `5e-4`。不一致・NaN・未実行は速度比較から外す。入力・ID・型・形状を固定し、同じウォームアップ、繰り返し回数、計時範囲で Vulkan/HIP と逆順を実行して、中央値とばらつきを残す。常駐演算の計時には初期ロード、Expert ミス時の RAM 転送、Router、PLE、MTP を含めない。
4. 結果は `us/run` とケース条件をセットで保存する。グラフの同期込みの経過時間と GPU プロファイラの演算時間は別列にする。HIP が安定して参照に合い、比較可能な差が再現したところで、混在実行へ進むか判断する。任意の短文生成は同一量子の収容可能な小型モデルを選び、ロード、デバイス配置と数トークンの生成を確認する。

## 後続フェーズと移植境界

| 段階 | 実施する内容 | 次に進む条件 |
|---|---|---|
| 1. CUDA/HIP 共存 | Linux の同一プロセスで RTX CUDA と MI50 HIP を同時に列挙し、それぞれに小型グラフを割り当てて実行する。最初は独立実行、次に既存の同期ホストステージングで転送を往復させる。 | 同時ロードと演算を確認。誤ったバッファ登録、デバイス取り違え、二重解放が発生せず、同期転送の結果が参照と一致する。 |
| 2. Packed Expert の最小経路 | RTX 側に Attention、Router、Shared Expert、MTP を置き、MI50 HIP に Routed Expert を置く。常駐ヒットから始め、次に同期のスクラッチ入れ替えと動的昇格・書き戻しを足す。元の Expert 重みの所有先、転送完了前の上書き禁止、失敗時の配置表整合性を維持する。 | 固定2幅で Router ID、常駐ヒット、ミス、退避後の再読込を照合できる。続いて Flash-Next 2K の短い生成が安定する。 |
| 3. 転送と演算の最適化 | 同期経路で入出力の受け渡し、Expert ミスの DMA、昇格と書き戻しの待ちを測る。効果のある順に pinned staging、非同期コピー、共有入出力や Router ID のホスト再利用を検討する。順次加算と Router 重み乗算も同じ検証幅で個別に比較する。 | 既存 Linux Vulkan と同条件の固定 MTP・実対話で品質と速度を比較し、採用を判断する。 |

- **共存の確認箇所**。HIP は `ggml-hip` として CUDA と同じ `.cu` を `GGML_USE_HIP` で再コンパイルする。通常リンクでは HIP 側も `GGML_USE_CUDA` を公開し、登録側は `ggml_backend_cuda_reg` を使う。CUDA と HIP の両ターゲット名が作れても、登録、公開シンボル、動的ロード時の依存関係と CMake の言語・定義の分離は別途検証する。動的 backend は候補として、ロード順とシンボル衝突も確認する。 [HIP CMake](../../ggml/src/ggml-hip/CMakeLists.txt) [backend 登録](../../ggml/src/ggml-backend-reg.cpp) [CUDA/HIP 共通実装](../../ggml/src/ggml-cuda/ggml-cuda.cu)
- **Expert キャッシュの境界**。配置と動的常駐は Vulkan のデバイス名・機能で判定し、転送や昇格、広幅 prefill には Vulkan 固有処理がある。まず同期実装でキャッシュの所有権、スロットマップ、キャンセル時の解放を検証し、共通化する操作と API 固有操作を整理する。 [配置](../../src/llama-expert-placement.cpp) [MoE キャッシュ](../../src/llama-moe-cache.cpp)
- **GPU 間受け渡しの境界**。現行 scheduler の共有 RAM import、upload/readback、producer-stream の経路には CUDA/Vulkan の組み合わせ判定がある。HIP のホストメモリ登録やイベント互換性は別途検証し、最初は明示同期で動作させる。PCIe P2P、共有ハンドル、複数 MI50 の分担は後続検証とする。 [scheduler](../../ggml/src/ggml-backend.cpp)

ソース差分、構成、確認結果の要約を引き継ぎに残す。生ログ、環境ダンプ、プロファイル、CSV は Git 管理外に保存する。
