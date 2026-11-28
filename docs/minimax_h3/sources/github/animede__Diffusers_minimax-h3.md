# animede__Diffusers_minimax-h3

Source: https://github.com/animede/Diffusers_minimax-h3
Fetched: 2026-11-28

Description: MiniMax-H3 (video + synchronized audio, 33B) on diffusers — 9.7s per still, 44.2s per 5s clip on 48GB, with every speedup verified byte-identical
Stars: 111  Updated: 2026-11-27T04:45:15Z



---

## FILE: README.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/README.md)

# minimax-h3

**日本語** | [English](README.en.md)

**最新版: 2026-11-27 JST** — `scripts/install.sh`(Linux / Mac): Python 3.12 はあっても `python3.12-venv`(ensurepip)が入っていない環境(Google Colab の Ubuntu 24.04 など)で、pip の無い不完全な venv ができて止まる問題を修正。pip の無い venv は作り直し、uv があれば `uv venv --seed` で pip 入りの venv を作る(コントリビュータの PR #2)。それより前(11/26): `prepare_lowvram.py` が最後に、HF キャッシュに残った bf16 の原本をもう一度消す。変更の履歴は [docs/HISTORY.md](docs/HISTORY.md)。

**MiniMax H3 (Hailuo 3.0) — 動画+ステレオ音声を1回のデノイズで同時生成する 33B オムニモーダルモデルを、普及帯 GPU で動かす検証アプリ。**
テキスト / 画像参照 / 音声参照から **動画+ステレオ音声** を生成し、静止画のみ(T2I / Ref2I)も出せる。
基盤は diffusers の Modular Pipeline(PR #14355、コミット **f37ab93** にピン留め)。
**6GB 単体・8GB×2・実 RTX 4060 Ti 16GB 単体、Windows の実機(RTX 3050 6GB・RTX 4060 Laptop 8GB・RTX 3060 12GB)で動くことを実測済み**で、32GB 単騎ならリアルタイム会話生成(実時間の 0.95 倍)まで到達している。

![GUI メイン画面](docs/images/ui_main.png)

*ブラウザから使う単一ページ UI。5 つのモードをタブで切替(動画: **T2VA**=テキスト / **FL2VA**=フレーム / **Ref2VA**=参照、静止画: **T2I** / **Ref2I**)。下段は生成物ギャラリーで、チェックした動画を連結して書き出せる。プロンプトはローカル LLM で H3 公式ガイドの形式へ強化でき、FirstBlockCache・Sage・Turbo・量子化・低 VRAM モードなどは**再起動なしで**このパネルから切替。日英切替あり。*

## クイックスタート(Windows・VRAM 6〜16GB の PC)

X などから来た方向けの最短手順。詳細は下の「[インストール](#インストール)」「[起動](#起動)」。

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
.\run.bat
```

- winget のあとは PowerShell を一度開き直す。ffmpeg はギャラリーの長さ・解像度の表示と動画の連結に使う(無くても生成はできるが、ログに案内が出る)。`install.bat` は venv を作って依存を入れる(数分)。
- `prepare_lowvram.py` はモデルの準備。144GB を落とすのではなく、66GB の重みを 1 つずつ落として量子化しながら消すので、**ディスクは約 60GB**、RAM 16GB で足りる。所要時間はほぼ回線速度で決まる(数時間)。途中で止まっても再実行すれば続きから進む。Ref2VA / Ref2I(参照素材から動画)も使うなら `--ref` を付ける(+32GB)。
- `run.bat` でサーバが起動し、ブラウザで `http://127.0.0.1:8611` を開く。初回起動は投影 TE の量子化キャッシュを作るので数分かかる。GPU の大きさに合わせた設定(低 VRAM モード、turbo 4 ステップ)は自動で入る。
- 目安(turbo 4 ステップ・5 秒): RTX 3050 6GB + NVMe で 512² 237 秒、RTX 3060 12GB + NVMe で 512² 132 秒・768² 338 秒、RTX 4060 Laptop 8GB で 512² 194 秒・768² 412 秒。ピーク VRAM は 768² で約 2.8GB、1024² で約 3.5GB。モデルのネイティブ解像度は短辺 768(768²、1344×768)で、3060 12GB ならそのネイティブで実用になる。3050 6GB は時間の都合で 512²〜768×512(ネイティブ未満)に落とすのが現実的。低 VRAM 構成は毎ステップ 30GB の重みをディスクから読むので、SATA 級 SSD ではデノイズがディスク待ちになる。NVMe を推奨。詳しくは「[Windows 実機の実測](#windows-実機の実測)」。
- 低 VRAM 構成の速さは「GPU への転送速度」と「SSD の読み出し速度」で決まる。遅いと感じたら `.\venv\Scripts\python.exe scripts\check_bandwidth.py` で両方を測れる(目安: PCIe 4.0 x8 でピン留め転送 10GB/s 前後、x4 で 5GB/s 前後。1GB/s を切っていればリンクや BIOS 設定の問題)。
- うまく行かないときは「[Windows](#windows)」節のトラブル対処と [Issues](https://github.com/animede/Diffusers_minimax-h3/issues) へ。

## 特徴

- **普及帯 GPU で動く**: transformer int8 のブロック単位ストリーミング(`H3_LOWVRAM=group`)、投影 TE(Qwen3-VL-4B + 線形写像で 32B TE を代替)、video VAE の fp16 化、TE / token_refiner / VAE デコーダのストリーミングにより、turbo 4 ステップなら 768²×5 秒がピーク **2.8GB**、1024²×5 秒が **3.5GB** で完走する(30 ステップでも 4.6GB / 6.4GB)。ホスト RAM が 32GB / 16GB でも量子化キャッシュから直接流せる。
- **固定費の削減**: 律速はデノイズではなく「モデルの load/free」だった。transformer 常駐・量子化済みキャッシュ・pinned 退避・参照 prefix の KV キャッシュで、48GB 級で t2i 9.7 秒 / t2va 5 秒 44 秒、96GB 級で t2i 7.4 秒。
- **高速化オプション**: FirstBlockCache、SageAttention(sm_120)、lightx2v turbo LoRA(4 / 8 ステップ蒸留)、ck-w4a8 量子化、AdaLN-pruned 版 transformer_ref。
- **5 つの生成モードとバッチ**: T2VA / FL2VA / Ref2VA(画像最大 9・動画最大 3・音声最大 3 の順序付き参照、音声参照でリップシンク)/ T2I / Ref2I。静止画・参照付き動画は最大 24 場面のバッチ生成に対応。**静止画(T2I / Ref2I)は画像専用モデルの MiniMax-H3-Image ではなく、この動画モデル(MiniMax-H3)で 22 フレームの超短尺動画を生成して中央フレームを PNG にしたもの**。専用モデルより遅いが、FL2VA の先頭フレームや Ref2VA の参照に使うと画風が完全に一致する。MiniMax-H3-Image は別リポジトリ [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server) で扱う。
- **LLM プロンプト強化**: 外部 LLM(OpenAI 互換 API、`H3_LLM_URL`)で入力を H3 公式スキル `h3-prompt-writing` の形式へ整形。生成前の検証 + 修復ループつき。外部 LLM に接続できないときは CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0、約 4.4GB、初回に自動 DL、thinking OFF)へ自動フォールバックする(ACE-Step-1.5-fast の easy-music と同じ考え方。実体は llama.cpp のビルド済み `llama-server` を子プロセスで起動するので、コンパイラ不要で Windows でもそのまま動く)。
- **品質の担保**: diffusers の更新や量子化キャッシュの導入は、同一 seed の MD5 一致で回帰確認している。

## サンプル

[![t2va 768² 5 秒、turbo 4 ステップ、RTX 4060 Laptop 8GB](docs/samples/t2va_768_turbo4_rtx4060laptop.gif)](docs/samples/t2va_768_turbo4_rtx4060laptop.mp4)

*利用者の Windows 11 + RTX 4060 Laptop 8GB / RAM 16GB で生成した t2va(768²・5 秒・turbo 4 ステップ)。GIF は縮小・減色したプレビューで、クリックすると音声つきの元の mp4(1MB)が開く。*

## 対応 GPU(早見表)

t2va 5 秒・768² の実測。詳細な行(80GB / 18GB / 12GB 級など)と測定条件は [docs/FEATURES.md](docs/FEATURES.md#vram-構成と常駐設計) を参照。

| GPU | 起動時の指定 | 結果 |
|---|---|---|
| 96GB | (指定なし = 既定) | peak 92GB / 約 160 秒 |
| 48GB 級 | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`(2 枚構成の推奨) | t2i 9.7 秒 / t2va 44.2 秒(turbo 4 ステップ) |
| 24 / 32GB 単騎 | ck-w4a8 + AdaLN-pruned + pinned 退避の一式([詳細](docs/FEATURES.md#2026-10-更新-24gb32gb-単騎でリアルタイム会話生成)) | 320×448・141f が **5.57 秒 = 実時間の 0.95 倍**(32GB)/ 7.1 秒(24GB) |
| 32GB 級 | `H3_LOWVRAM=group` | peak 28.7GB / 約 280 秒 |
| 16GB 単体(実 4060 Ti / A4000) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | 4060 Ti: peak 11.4GB(2026-08)。A4000(2026-11-17): 30 ステップ peak 8.0GB / 790 秒、turbo 4 ステップ(`H3_TURBO_LORA=1`)peak 7.0GB / 151 秒 ※ |
| 12GB 単体(実 RTX 3060 12GB) | 同上(auto で 16GB 級の設定が入る) | turbo 4 ステップ・5 秒(入門機 + NVMe): 512² 132 秒、768×512 198 秒、768² 338 秒、1024×768 473 秒。詳細は下の「Windows 実機の実測」 |
| 6GB / 8GB 単体 | 上の行 + `H3_TE_STREAM=1 H3_TE_DIET=1` | 30 ステップ: 768² 4.62GB / 1024² 6.41GB(48GB 機で 272 / 552 秒)。turbo 4 ステップ(A4000 に 6GB の上限、量子化キャッシュのファイルから流す構成): 768² 2.82GB / 211 秒、1024² 3.48GB / 380 秒、ref2va 768² 276 秒。ホスト RAM 48GB 以上でピン留めする構成(8GB 上限): 768² 3.08GB / 209 秒、1024² 3.73GB / 376 秒、fl2va 225 秒、ref2va 299 秒。実機(RTX 3050 6GB、RTX 4060 Laptop 8GB)は下の「Windows 実機の実測」 ※ |
| 8GB×2 | 16GB 単体の指定 + `H3_TE_DEVICE=cuda:1` | peak 7.23GB ※ |

※ 時間は転送と読み出しに強く依存する。低 VRAM 構成は毎ステップ重み 30GB を転送するため、30 ステップでは PCIe 帯域が律速だった(4060 Ti の Gen3 x4 で約 25 分、A4000 の Gen4 x4 で 790 秒、48GB 機の GPU0 で 272 秒)。**turbo LoRA(4 ステップ)が group でも使えるようになり、転送回数がステップ数に比例して減ったため「動くが遅い」はほぼ解消した**(x4 接続の A4000 で t2va 5 秒 151 秒)。ホスト RAM 48GB 未満(file ストリーミング)では毎ステップ 30GB をディスクから読むので、SATA 級 SSD(約 430MB/s)ではデノイズがディスク待ちになり、NVMe では GPU 性能で決まる(下の実機表)。8GB / 6GB 級の値は A4000 に上限(`H3_VRAM_LIMIT_GB`)を切った模擬で、実機でも同じ範囲に収まることを確認している。sm_120 以外のカードは `H3_ATTN_BACKEND=default` が必要(未指定なら `H3_PROFILE=auto` が自動で入れる)。構成をまたぐと同一 seed でも構図が変わる(劣化ではなく軌道の分岐)ことは [性能の概要](docs/FEATURES.md#性能の概要) に詳しい。

### Windows 実機の実測

turbo 4 ステップ・5 秒(124 フレーム)の t2va。括弧内はデノイズ + デコード(秒)。入門機は i3-10105(4 コア、PCIe 3.0)/ DDR4-2400 / B560 / RAM 32GB で、CPU・メモリ速度・PCIe(3.0 x8、転送 6GB/s)はどれも律速になっていない。

| 機体 | ディスク | 512×384 | 512² | 768×512 | 768² | 大きいサイズ |
|---|---|---|---|---|---|---|
| RTX 3050 6GB(入門機、Windows 10) | SATA 級 SSD(430MB/s) | 294 | 321(237 + 65) | — | — | — |
| 同上 | NVMe | 163(108 + 42) | 237(147 + 65) | 345(248 + 85) | 558(444 + 112) | — |
| RTX 3060 12GB(同じ入門機) | SATA 級 SSD | 270(240 + 26) | 282(235 + 37) | — | 324(251 + 62) | 1344×768 711(590 + 106) |
| 同上 | NVMe | — | 132(83 + 36) | 198(139 + 47) | 338(241 + 66) | 1024×768 473(385 + 77) |
| RTX 4060 Laptop 8GB / RAM 16GB(ゲーミングノート、Windows 11) | NVMe | — | 194(117 + 16) | 279(191 + 65) | 412(313 + 85) | — |

- SATA 級 SSD では 768² までデノイズが約 240 秒で一定。毎ステップ 30GB の重みを読む時間で、GPU を替えても変わらない。NVMe ではデノイズも GPU 性能で決まる(3050 で 768² が 444 秒、3060 で 241 秒)。
- モデルのネイティブ解像度は短辺 768(768²、1344×768。turbo LoRA も 768p で学習)。3060 12GB ならネイティブの 768² が 6 分弱で実用。3050 6GB は時間の都合で 512²〜768×512(ネイティブ未満)に落とすのが現実的。
- 生成中の共有 GPU メモリは 1.5GB 前後で一定(ピン留めした受け皿のぶん)。

## インストール

### 必要なもの

| | 要件 |
|---|---|
| GPU | 最小: **6GB 単体**(1024² まで動くが、時間の都合で 512²〜768×512 が現実的)/ **8GB・12GB 単体**(ネイティブの 768² が実用)/ **16GB 単体**(全機能)。快適動作は **48GB 級**を推奨。モデルのネイティブ解像度は短辺 768(768²、1344×768) |
| ホスト RAM | **16GB で動く(NVMe 前提)、24GB 以上を推奨、48GB 以上でデノイズも速い**。48GB 未満では量子化キャッシュのファイルから直接流す方式に自動で切り替わる(`H3_GROUP_STREAM_HOST=file`。毎ステップ 30GB をディスクから読むのでデノイズは SSD の速さで決まる)。16GB では VAE デコーダ 4.5GB がページキャッシュに残らずデコードも毎パスのディスク読みになるが、24GB 以上なら 32GB と同じ。48GB 以上(Linux)は int8 重み ~34GB をピン留めし、ディスク読みが消える |
| ディスク | 約 145GB(T2VA / FL2VA のみ)/ 約 207GB(Ref2VA も使う場合)。低 VRAM / 低 RAM の準備経路(`scripts/prepare_lowvram.py`)は量子化キャッシュだけを置くので約 60GB(Ref2VA も使うなら `--ref` で +32GB) |
| CUDA | 12.8 系(torch 2.9.0+cu128 に合わせる)。SageAttention をビルドするなら `nvcc` も同系統 |
| Python | 3.12 |
| その他 | `ffmpeg`(ギャラリーの連結・情報取得に使用。無くても生成自体は動く) |

### 手順

```bash
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
scripts/install.sh
```

`scripts/install.sh` は `venv/` を作って [requirements.txt](requirements.txt) を入れ、最後に torch / diffusers / transformers のバージョンを表示する。必要なのは Python 3.12 と `git` だけで、CUDA のビルドは発生しない(数分で終わる)。`--sage` を付けると SageAttention も入れる(下記)。`--local-llm` を付けると内蔵 LLM(Gemma 4 E4B)用のビルド済み `llama-server` も先に取得する(下記。付けなくても初回のフォールバック時に自動取得される)。

<details>
<summary>手動で入れる場合(requirements.txt と同じ内容)</summary>

```bash
python3.12 -m venv venv

# PyTorch (cu128)
venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# ============================================================================
# 【重要】diffusers は検証済みコミット f37ab93(PR #14355 のマージ最終形)に固定する。
#        pip install diffusers(PyPI 版)や最新 main を入れないこと(下の注記を参照)
# ============================================================================
venv/bin/pip install "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# transformers は 5.14.1 以上が必須(5.1.0 には Qwen3VLProcessor.create_mm_token_type_ids が無い)
venv/bin/pip install "transformers==5.14.1" accelerate==1.12.0 safetensors huggingface_hub

# 動画/音声の多重化と Web API
venv/bin/pip install av==16.0.1 fastapi==0.104.1 "uvicorn==0.24.0" python-multipart pillow numpy

# text_encoder の 4bit 量子化(既定の H3_TE_QUANT=bnb-4bit に必要)
venv/bin/pip install bitsandbytes==0.49.0

# transformer の int8 量子化を使う場合のみ(H3_TRANSFORMER_QUANT=int8 / H3_LOWVRAM)
#   0.18 以降は torch>=2.11 を要求するため 0.17.0 を固定する
venv/bin/pip install torchao==0.17.0
```

</details>

> **diffusers の更新について**: インストール自体は上の 1 行で済む(純 Python なのでビルドは無く、`git` があれば数十秒で入る)。注意が要るのは**あとから diffusers を上げるとき**で、計算経路が変わって出力が変わりうるため、コミット固定のまま使うこと。f37ab93 への追従も、全経路(t2i / t2va / バッチ / ref2va / 参照バッチ)が旧ピン abc5e9b と同一 seed の MD5 で等価であることを確認してから行った。さらに上げる場合は同じ回帰確認(`scripts/run_regression.py` と docs/internal の基準値)を踏む。普通に使うだけなら不要。

### Windows

依存パッケージは Windows 用の wheel がそろっています(torch 2.9.0+cu128 は PyTorch の index から、torchao は PyPI の純 Python wheel、bitsandbytes / av / psutil は PyPI の Windows wheel)。コンパイラや Visual Studio Build Tools は不要です。SageAttention は Windows 向けのビルドが無いため使わず、起動時に `H3_PROFILE=auto` が `H3_ATTN_BACKEND=default`(SDPA)を自動で入れます。コード側の OS 依存(RAM の取得、ファイルの先読み指示、一時ファイルの置き場所、内蔵 LLM の子プロセス)は Windows 用に分岐済みです。Hugging Face のキャッシュは Windows では symlink を使わない設定(`HF_HUB_DISABLE_SYMLINKS=1`)を自動で入れます(開発者モードが OFF の PC で初回ダウンロード中に `WinError 1314` で止まる huggingface_hub の競合を避けるため。ディスク使用量は変わりません)。

必要なもの: Python 3.12(`py -3.12` で呼べること)、git、CUDA 12.8 系に対応した NVIDIA ドライバ。`ffmpeg` は任意(ギャラリーの連結用)。git や Python が無い場合は winget で入れる(入れたあと PowerShell を開き直す):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
```

```powershell
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
```

`install.bat` は PowerShell の実行ポリシーに関係なく `scripts\install.ps1` を実行する(Windows の既定では未署名の `.ps1` を直接実行できないため)。venv を作り、torch(cu128)と `requirements.txt` を入れる。`-LocalLLM` を付けると内蔵 LLM のバイナリも取得する。続けてモデルを取得する:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

(VRAM 48GB 以上・RAM 64GB 以上の PC なら `scripts\download_t2va.py` でもよい。下の「低 VRAM / 低 RAM の PC の準備」参照。)手動で入れる場合は次の順に実行する(PyPI の Windows 版 torch は CPU 専用なので、torch を先に PyTorch の index から入れる順序が必要):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

起動方法(環境変数の渡し方、VRAM 級別の例)は「[起動](#起動)」の Windows 節を参照。`.ps1` を直接実行したい場合は `powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1` のように Bypass を付けるか、一度だけ `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を実行しておく。

内蔵 LLM(外部 LLM に接続できないときのフォールバック)は、インストール時に `-LocalLLM` を付けなくても後から入れられる。初回のフォールバック時に自動取得されるが、先に取っておく場合は次を実行する(1 行目がビルド済み `llama-server.exe`(約 25MB)、2 行目が Gemma 4 E4B の GGUF(約 4.4GB)。GGUF は省略可):

```powershell
.\venv\Scripts\python.exe scripts\fetch_llama_server.py
.\venv\Scripts\python.exe scripts\fetch_llama_server.py --model
```

`.\scripts\install.bat -LocalLLM` を後から実行しても同じ(既に入っている依存はそのまま)。詳細は下の「[内蔵 LLM(任意)](#内蔵-llm任意)」。

> Windows の実機状況(2026-11-20): Windows 11 + RTX 4060 Laptop 8GB / RAM 16GB、Windows 10 + RTX 3050 6GB と RTX 3060 12GB(入門機 i3-10105 / RAM 32GB)で、インストール → `prepare_lowvram.py` → 起動 → 生成(30 ステップ・turbo 4 ステップ)まで確認済み。時間は上の「[Windows 実機の実測](#windows-実機の実測)」。Windows 固有の対処として、HF キャッシュの symlink 無効化、原本シャードの削除の再試行、実行ポリシーを回避する `.bat`、VRAM の自動上限(カード容量 − 1.2GB)、ピン留め RAM を使わず常にファイルから流す、デコーダの常駐判定で物理の空きを見る、を入れてある(経緯は [docs/FEATURES.md](docs/FEATURES.md))。タスクマネージャの読み方: 専用 GPU メモリはサーバが報告するピークより大きく見える(torch が抱えるプールと表示の分を含む)。共有 GPU メモリはストリーミング用にピン留めした RAM で、1.5〜2GB 程度が正常。生成中に共有が 4GB 以上へ増え続けるなら VRAM があふれて退避しており極端に遅くなる(最新版では起きないはず。起きたら Issue へ)。同じ方式の画像サーバ [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server) は Windows 11 + RTX 3050 で生成まで確認済み。問題があれば [Issues](https://github.com/animede/Diffusers_minimax-h3/issues) へ。

### SageAttention(任意、既定で有効。Linux のみ)

既定の `H3_ATTN_BACKEND=sage` を使うには sm_120 向けのビルドが要る(Linux 向けの事前ビルド wheel は無い)。**ビルドしない場合は `H3_ATTN_BACKEND=default` を指定して起動する**(デノイズが約 12% 遅くなるだけで機能に影響はない)。

```bash
CUDA_HOME=/usr/local/cuda-12.8 scripts/install.sh --sage     # または scripts/build_sageattention.sh
```

`--sage` は `third_party/wheels/` にビルド済み wheel があればそれを入れ、無ければソースビルドする。

### 内蔵 LLM(任意)

LLM 強化は既定で外部の OpenAI 互換 API(`H3_LLM_URL`)を使い、接続できないときは CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0、thinking OFF)へ自動フォールバックする。内蔵 LLM の実体は llama.cpp の [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) にあるビルド済み `llama-server`(CPU 版、約 25MB。固定タグ `b12475`、`H3_LLAMA_CPP_TAG` で変更可)で、サーバがこれを子プロセスとして起動し、OpenAI 互換 API(既定 `127.0.0.1:64661`、`H3_LLM_LOCAL_PORT`)で話す。**コンパイラは不要**で、Linux x64 / Windows x64 / macOS のバイナリが自動で選ばれる。

```bash
scripts/install.sh --local-llm            # llama-server を tools/llama/ へ先に取得(任意)
venv/bin/python scripts/fetch_llama_server.py --model   # GGUF(約 4.4GB)も先に取得(任意)
```

RAM の少ない PC(16GB 級)では動画生成と内蔵 LLM が RAM を奪い合うため、llama-server は最後の利用から 300 秒(`H3_LLM_LOCAL_IDLE_S`)で止まり、ホスト RAM 32GB 未満の PC では生成の開始時にも止まる(`H3_LLM_LOCAL_STOP_ON_GENERATE=auto`)。次に LLM 強化を押せば数秒で再起動する。

どちらも省略でき、後から実行しても同じ(インストール済みの環境に足せる)。初めてフォールバックしたときに `llama-server` は `tools/llama/` へ、モデルは `models/llm/` へ自動取得される。子プロセスはサーバ終了時に止まり、サーバが強制終了された場合も Windows は Job Object(KILL_ON_JOB_CLOSE)、Linux は `PR_SET_PDEATHSIG` で道連れにするので孤児にならない(Linux は SIGKILL で実測、Windows は未検証)。既にあるものを使うなら `H3_LLAMA_SERVER=<llama-server のパス>`、`H3_LLM_LOCAL_GGUF=<GGUF のパス>` で指定する(自分で起動した llama-server が `H3_LLM_LOCAL_PORT` で応答していればそれをそのまま使う)。

### モデルの取得

**VRAM 32GB 以下、またはホスト RAM 48GB 未満の PC は、次の「低 VRAM / 低 RAM の PC の準備」を使う**(こちらの手順は 144GB をそのまま落とすので、低 VRAM 級では無駄が多く、RAM の少ない PC では量子化キャッシュを作れず起動時に止まる)。

`MiniMaxAI/MiniMax-H3` は 498.6GB あるが、T2VA / FL2VA には `transformer/`(66.3GB)+ `text_encoder/`(66.7GB)+ `vae/`(10.4GB)+ `audio_vae/`(0.6GB)+ 設定類の**約 144GB** で足りる。`transformer_ref/`(Ref2VA 用、66.3GB)は必要なときだけ取得し、**丸ごと `snapshot_download` しないこと**。

```bash
venv/bin/python scripts/download_t2va.py
```

内部で `allow_patterns` を使って必要なサブフォルダのみを取得する。ダウンロード中は `logs/du_monitor.log` でキャッシュサイズを監視できる。bf16 の transformer / transformer_ref がキャッシュに無いときは、`H3_ALLOW_WEIGHT_DOWNLOAD=1` のときだけ自動取得する(既定はエラーで案内)。

### 低 VRAM / 低 RAM の PC の準備

`H3_LOWVRAM=group`(32GB 以下の級)で重みを量子化キャッシュのファイルから流す構成は、**int8 量子化済みキャッシュ(`models/prequant/transformer_int8`、約 32GB)が先にあること**が前提。無いと bf16(66GB)から量子化する従来経路に落ち、ホスト RAM 40GB のガードで止まる(RAM 16GB の PC で実際にこのエラーになる)。`scripts/prepare_lowvram.py` は、bf16 のシャード(14 個、各 4.7GB)を 1 つずつ落として量子化して消す方式でこのキャッシュを作る。RAM は数 GB、VRAM は不要(既定は CPU で量子化、`--device cuda` も可)、ディスクのピークはシャード 1 つ分 + 出力。出来上がりはサーバが自分で作るキャッシュと**ビット単位で同一**(確認済み)。あわせて VAE デコーダの fp16 キャッシュを作り、投影 TE(Qwen3-VL-4B + ClipProj)を先に取得する。32B の text_encoder は落とさない。

```bash
venv/bin/python scripts/prepare_lowvram.py
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

所要時間はほぼ 66GB のダウンロード時間(量子化はシャードあたり数秒〜数十秒)。キャッシュは HF 側のスナップショットと照合されるが、bf16 の原本がローカルに無い PC では(作り直しができないので)照合を省略してそのまま使う。再実行は冪等で、既にあるキャッシュは検査だけして数秒で終わる。`--keep-source` で bf16 シャードを残せる(+66GB、再構築が速い)。投影 TE の NF4 キャッシュは初回起動時にサーバが作る(VRAM 8GB・RAM 16GB で可)。

**Ref2VA / Ref2I も使うなら `--ref` を付ける。** 参照モードは `transformer` とは別の重み `transformer_ref/`(66GB、全シャードの内容が異なる)を使うので、そのキャッシュ `models/prequant/transformer_ref_int8`(約 32GB)も同じ方式で作る。ホスト RAM 48GB 未満の PC ではこれが無いと参照モードは起動時に止まる(作り方を案内するエラーになる)。後から足してもよい(再実行は冪等で、既にある `transformer_int8` は数秒の検査だけ)。

```bash
venv/bin/python scripts/prepare_lowvram.py --ref
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py --ref
```

## 起動

### Linux

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

何も指定しなければ **`H3_PROFILE=auto`** が計算用 GPU の総 VRAM から 96 / 80 / 48 / 32 / 16 / 8GB 級の既定値一式(下の起動例と同じ内容)を選ぶ。32GB 以下の級(group)では turbo LoRA(4 ステップ)も既定で ON になる(group では起動時に巻かないと GUI からも使えないため。GUI で turbo を外せば 30 ステップ、`H3_TURBO_LORA=0` で起動時から無効)。48GB 以上は GUI の turbo チェックでリクエストごとに使う。`H3_LOWVRAM` や `H3_TE_PROJ` など主プロファイル系の変数をひとつでも指定すると手動扱いになり、自動では何も入れない。sage が使えない GPU(sm_120 以外)や未インストールのときは `H3_ATTN_BACKEND=default` も自動で入る。選ばれた級は `/api/status` の `profile` と起動ログで確認できる。

ブラウザで `http://<host>:8611/` を開く。初回起動時はモデルのロード(既定では TE を NF4 量子化して GPU 常駐)に数分かかる。`run.sh` は diffusers-movie-server 統合用の起動スクリプトで、内部ポート 8631 と会話用途の既定値(turbo LoRA 等)を設定したうえで同じサーバを起動する。

VRAM 級別の起動例:

```bash
# 実16GB 単体(sm_89 の 4060 Ti など。sm_120 以外は H3_ATTN_BACKEND=default が必須)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 6GB / 8GB 単体(16GB の指定に TE のストリーミングと削減を足す。さらに下げるなら H3_CACHE=none)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1 H3_TE_DIET=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 8GB×2(TE を 2 枚目 GPU へ)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default H3_TE_DEVICE=cuda:1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 32GB 級
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 48GB 級(推奨: 高速化フル。ref2va を使うときは H3_TE_DEVICE を外す)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 96GB 級(全モデル常駐、env なし)
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

24 / 32GB 単騎でのリアルタイム会話生成(ck-w4a8 量子化 + pinned 退避)の環境変数一式と注意点は [docs/FEATURES.md](docs/FEATURES.md#2026-10-更新-24gb32gb-単騎でリアルタイム会話生成) を参照。

UI より先に動作確認したいときは回帰プローブを使う(`outputs/probe_t2va.mp4` と `outputs/probe_report.json` を出力):

```bash
venv/bin/python scripts/probe_t2va.py
```

### Windows

`run.bat` が同じサーバを起動する(中身は `run.ps1` を実行ポリシーに関係なく呼ぶだけ。既定は 127.0.0.1:8611。`H3_HOST` / `H3_PORT` で待ち受けアドレスとポートを変更):

```powershell
.\run.bat
```

環境変数は起動前に `$env:` で 1 行ずつ設定する。上の Linux の起動例は、`H3_X=値` をそれぞれ `$env:H3_X = "値"` に読み替えて `.\run.ps1` を呼べばそのまま使える。16GB 単体の例(sm_120 以外は `H3_ATTN_BACKEND=default` が必須だが、未指定なら `H3_PROFILE=auto` が自動で入れる):

```powershell
$env:H3_LOWVRAM = "group"
$env:H3_TE_PROJ = "NicoLab28/ClipProj-MiniMax-H3"
$env:H3_VIDEO_VAE_FP16 = "1"
$env:H3_ATTN_BACKEND = "default"
.\run.bat
```

何も設定しなければ `H3_PROFILE=auto` が GPU の総 VRAM から級を選ぶので、まずは `.\run.bat` だけで起動してよい(32GB 以下の級では turbo LoRA も自動で ON。48GB 以上で turbo を既定にしたいときだけ `$env:H3_TURBO_LORA = "1"` を足す)。設定した環境変数はその PowerShell を閉じるまで残る(別の構成で起動し直すときは `Remove-Item Env:H3_LOWVRAM` のように消すか、PowerShell を開き直す)。

`run.ps1` を直接実行して実行ポリシーで止まる場合は次のように起動する(`run.bat` はこれを内部で行う):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

回帰プローブは次のとおり:

```powershell
.\venv\Scripts\python.exe scripts\probe_t2va.py
```

## 使い方

### UI

ブラウザの単一ページ UI。左が入力と設定、右が結果とギャラリー。右上の「日本語 / English」で表示言語を切り替えられる。

**モードのタブ**

- 上段が動画(**T2VA** = テキスト、**FL2VA** = 先頭 / 末尾フレーム、**Ref2VA** = 参照素材)、下段が静止画(**T2I**、**Ref2I**)。タブを切り替えると、そのモードに必要な入力欄だけが表示される。
- 静止画タブは MiniMax-H3-Image ではなく、動画モデルで 22 フレーム(または 5 フレーム)の超短尺動画を生成して中央フレームを PNG にする。mp4 も一緒に保存される。

**プロンプトと LLM 強化**

- プロンプトは日本語のままでも効く。「プロンプトガイド」を開くと、公式ブリーフ構造(シーン → 被写体 → アクション → カメラ → 音 → 終わり方)や `CUT n [X-Y秒]:` のカット記法、台詞タグ `<d>` の書き方が手書き用チートシートとして見られる。
- 「LLM強化」ボタンは、外部 LLM(無ければ内蔵の Gemma 4 E4B)でプロンプトを整形してプロンプト欄を置き換える。モードは 4 つ: `storyboard`(複数カットのタイムコード形式に展開)、`brief`(公式ブリーフ形式の単一ショット詳細化)、`h3-official`(MiniMax 公式スキル `h3-prompt-writing` の形式に厳密準拠。現在のタブを task として送り、出力言語は英語 / 日本語を選べる。生成前に検証 + 自動修復が走り、未解決の指摘があれば警告が出る)、`translate`(英訳のみ)。「元に戻す」で強化前の文に 1 回戻せる。
- 投影 TE(低 VRAM 級の既定)は台詞タグ `<d>` を扱えないため、`<d>` を含むプロンプトは生成前に警告される(台詞は音声参照で入れる)。

**モードごとの入力**

- FL2VA: 先頭フレーム / 末尾フレーム画像をドラッグ&ドロップ(どちらか一方でも可。両方指定すると先頭 → 末尾の間を生成)。
- Ref2VA / Ref2I: 音声参照つきで API の `vocal_lock=true` を付けると、その音声をそのまま出力音声にして口の動きを合わせる(Vocal Lock。UI にスイッチは無い)。参照ファイルを複数追加(画像最大 9・動画最大 3・音声最大 3、計 12 まで)。**追加した順序がそのまま参照順**になり、プロンプト内では `<Picture 1>` のようにタイルに表示されるラベルで参照できる。音声参照 1 本だけなら秒数は空欄でよく、音声の長さが尺になる。Ref2I はフレーム数(22 / 5)を選んで中央フレームを PNG にする。
- T2I: フレーム数(22 = 品質検証済みの既定、5 = 実験的)。
- バッチ連続生成(T2I / Ref2I / Ref2VA): チェックを入れるとプロンプト欄が「1 行 = 1 場面」(最大 24 場面)になり、参照・seed・解像度・ステップは全場面で共通。モデルのロードや参照エンコードを場面間で共有するので、低 VRAM モードや参照付きで特に効く。Ref2VA のバッチは秒数が必須。

**サイズ・秒数・ステップ・seed**

- 解像度は 768² / 1344×768(横長)/ 768×1344(縦長)のプリセットのほか、「任意」で幅 × 高さを直接指定(32 の倍数へ丸め、256〜2048。ネイティブ範囲を超えると警告)、「自動」でサーバ既定(Ref2VA は 16:9)。
- 秒数は 5〜15。実際のフレーム数は 17n + 5 に切り上げられ、「実際の尺」として表示される(例: 5 秒 → 124 フレーム 5.17 秒)。
- ステップ数は Turbo LoRA の ON / OFF に応じて推奨値(4 / 30)へ自動で切り替わる。手で変えた値はそのまま使われる。seed は空欄でランダム。

**即反映設定(再ロード不要、リクエストごとに送る)**

- FirstBlockCache(+ しきい値)、Sage Attention(sm_120 のみ)、Turbo LoRA(4 ステップ推奨)、無音で出力(音声トラックを付けない)。
- サーバの構成で使えない組み合わせはチェックが無効化され、理由がツールチップに出る(例: 起動時に turbo を巻いていない group モードでは turbo を選べない)。turbo を入れると FirstBlockCache は自動で外れる。

**生成と結果**

- 「生成」を押すと進捗バーに局面(TE エンコード / デノイズ n/N / デコード)と経過秒が出る。完了すると右側に動画(または PNG と mp4)と、モード・解像度・尺・ステップ・seed・デノイズ / デコード時間・ピーク VRAM・音声の有無・即反映設定・使用したプロンプト全文(折りたたみ)が表示される。
- LLM 強化のメッセージには、外部 LLM と内蔵 LLM のどちらで処理したかが出る。

**再ロードが必要な設定(ヘッダの折りたたみ)**

- transformer 量子化(none / int8。ck-w4a8 は起動時のみ)、text_encoder 量子化、TE レイヤー削除、投影 TE とその量子化、低 VRAM モード(0 / 1 / group)、video VAE の fp16 化。変更すると「適用(再ロード)」が有効になり、見込み時間が表示される。適用はモデルを解放して再ロードするだけで、プロセスは再起動しない。
- 投影 TE が ON のときは text_encoder 量子化と TE レイヤー削除が無効になる(32B TE 自体を読まないため)など、サーバ側の制約に合わせて項目が無効化される。

**サーバ状態**

- 入力欄の下に 4 秒ごとに更新される状態行がある: ロード状態(transformer / transformer_ref / VAE / TE)、GPU の使用量とピーク、ホスト RAM、`profile`(自動プロファイルの選択結果)、lowvram / 量子化 / attention / turbo / TE の構成、ストリーミングの原本の置き場所、LLM の接続先と内蔵 LLM の状態。

**ギャラリー**

- 生成済みの動画と静止画を新しい順に表示。サムネイルをクリックすると拡大表示(ライトボックス)、ダウンロードも可能。
- チェックして「選択を削除」「選択を連結」。連結は**チェックした順**につなぎ(静止画は対象外)、`outputs/concat_<時刻>.mp4` として保存される。

### API

| エンドポイント | 内容 |
|---|---|
| `POST /api/t2va` | テキスト→動画+音声。`prompt`, `resolution`(`768x768` / `768x1344` / `1344x768`), `seconds`(5〜15), `num_inference_steps`, `seed` |
| `POST /api/fl2va` | 上記に加え `image` / `last_image`(どちらか一方以上) |
| `POST /api/ref2va` | 参照付き生成。`references`(画像最大 9・動画最大 3・音声最大 3、順序が意味を持つ)。音声 1 本だけの参照なら `seconds=0` で音声長に合わせる。`still=1` + `frames` で参照付き静止画(ref2i)。`vocal_lock=true`(または起動時 `H3_VOCAL_LOCK=1`)で **Vocal Lock**: 参照の音声をそのまま出力音声として固定し、口の動きをその音声に合わせる(リップシンク。低 VRAM 構成でも動作、8GB 上限の模擬でエンコード段のピーク 3.9GB)。UI にスイッチは無く API / 環境変数のみ |
| `POST /api/t2i` | 静止画。`seconds` の代わりに `frames`(既定 22 / 5)。超短尺 mp4 と中央フレーム PNG を保存 |
| `POST /api/t2i_batch` / `POST /api/ref2i_batch` / `POST /api/ref2va_batch` | バッチ生成。`prompts` を場面数ぶん繰り返す(最大 24)。参照系は共通 `references`、`ref2va_batch` は共通 `seconds` が必須 |
| `POST /api/prompt/enhance` | LLM プロンプト強化。`{text, mode, seconds, task, lang}` |
| `GET /api/status` / `GET /api/progress` / `POST /api/interrupt` | ロード状態と VRAM / RAM、生成中の進捗、生成の中断 |
| `GET /api/settings` / `POST /api/settings/apply` | UI から切り替えられる設定の取得と適用(再ロード) |
| `POST /api/admin/unload` / `POST /api/admin/reload` | モデルの解放と再ロード |
| `GET /api/outputs` / `POST /api/outputs/delete` / `POST /api/outputs/concat` | ギャラリーの一覧・削除・連結 |

すべて multipart/form-data。各モードの詳細は [docs/FEATURES.md の生成モード](docs/FEATURES.md#生成モード) を参照。

## 主な環境変数

既定値を示す。多くは **UI からも切り替えられる**ので、恒久的に変えたいときだけ指定する。全変数と実測値は [docs/FEATURES.md](docs/FEATURES.md) を参照。

| 変数 | 既定 | 意味 |
|---|---|---|
| `H3_PROFILE` | `auto` | 主プロファイルの自動選択。`auto` は主プロファイル系の変数が未指定のときだけ総 VRAM から級を選ぶ(32GB 以下の級は turbo LoRA も ON)/ `96gb`〜`8gb` で級を強制(`H3_VRAM_LIMIT_GB` で小さいカードを模擬するときに使う)/ `off` |
| `H3_LOWVRAM` | `0` | `1` = 48GB 級のフェーズ循環 / `group` = 32GB 級以下の block offload(int8 重みをブロック単位でストリーム) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` で transformer を 66.3→34GB。`ck-w4a8` は ComfyUI の W4A8 カーネル(要 `comfy-kitchen`) |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder の量子化(`none` は bf16 で 66.7GB) |
| `H3_TE_PRUNE` | `0` | TE の未使用上位レイヤー削除(出力は不変、-3.6GB) |
| `H3_TE_PROJ` | (無効) | **投影 TE**: 32B TE を Qwen3-VL-4B + 線形写像で代替(repo id か .safetensors パス。品質は近似) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | 投影 TE の量子化(NF4 で 3.11GB。`none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (無効) | TE を 2 枚目 GPU へ常駐(例 `cuda:1`。32B TE は 20GB 級、投影 TE なら 8GB 級) |
| `H3_TE_STREAM` / `H3_TE_DIET` | `0` / `0` | 投影 TE・token_refiner のストリーミングと削減(6GB / 8GB 構成の鍵) |
| `H3_GROUP_STREAM` / `H3_GROUP_STREAM_HOST` | `own` / `auto` | group モードの重み転送方式。`HOST` は搭載 RAM 48GB 未満なら量子化キャッシュのファイルから直接流す(`file`)。Windows は RAM に関係なく常に `file`(ピン留め RAM からの転送が WDDM では極端に遅いため) |
| `H3_VIDEO_VAE_FP16` | `0` | video VAE を fp16 化(デコードのピークを削減。16GB 以下では必須) |
| `H3_KEEP_TRANSFORMER` | `0` | transformer 常駐で再ロード固定費を撤廃 |
| `H3_PRUNED` / `H3_PRUNED_QUANT` | `0` / `int8wo` | ref2va の transformer_ref を AdaLN-pruned 版で読む(`H3_TRANSFORMER_QUANT=int8` 併用必須)と、その量子化方式 |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache(デノイズ -25%。int8+SDPA 軌道では不発の実測あり) |
| `H3_ATTN_BACKEND` | `sage` | sage は sm_120 専用ビルド。**それ以外の GPU は `default`(SDPA)必須** |
| `H3_VRAM_LIMIT_GB` | (なし) | このプロセスが確保できる VRAM の上限(GB)。**Windows では未指定なら自動で「カード容量 − 1.2GB」**(`H3_VRAM_AUTO_CAP_MARGIN_GB`)。Windows の torch はプールが膨らんだまま返せず、6GB のカードでは物理 VRAM を越えて退避し極端に遅くなるため、上限で膨らみを止める(上限に当たると未使用の区画を返して再利用する)。6GB のカード(上限 4.8GB)でも 1024²・5 秒まで収まる(A4000 の模擬でピーク 3.5GB)。上限を越える要求は退避で延々と遅くなる代わりに OOM になる。無効にするにはカード容量以上の値(例 `99`)を指定 |
| `H3_GPU_POOL_DEFRAG` | `auto` | デコードの直前に GPU に残っている実体(約 1.2GB)を CPU へ退避 → torch のプールを全部返す → 戻す(`auto` = Windows のみ。Windows の torch はプールが膨らんだまま返せず、6GB のカードでデコード中に退避して極端に遅くなるため)。`1` / `0` |
| `H3_VAE_DECODER_RESIDENT` | `auto` | file ストリーミングのデコードで、空き VRAM に入るぶんだけ VAE デコーダ(36 ブロック、4.5GB)をデコードの間 GPU に置き、残りをファイルから流す(置いた分だけ毎パスの転送が減る。出力は同じ)。A4000 の模擬: 8GB 上限で 30/36 ブロック常駐、768² のデコード 86 → 34 秒(ピーク 6.2GB)、6GB 上限で 15/36、デコード 51 秒(ピーク 4.2GB)。`1` で全部強制、`0` で無効。余裕は `H3_VAE_DECODER_RESIDENT_MARGIN_GB`(既定 Linux 1.0、Windows 2.5。Windows の torch はプールが膨らむため) |
| `H3_VOCAL_LOCK` | `0` | Ref2VA の Vocal Lock を既定で ON にする(参照の音声を出力音声として固定し、口の動きを合わせる)。リクエストの `vocal_lock` が優先 |
| `H3_TURBO_LORA` | `0` | lightx2v の 4 / 8 ステップ蒸留 LoRA(初回に約 780MB を DL)。`group` でも使える(自作ストリーマ `H3_GROUP_STREAM=own` + 起動時指定が条件。`file` ストリーミングでは LoRA 約 1.2GB をホスト RAM からブロックごとに流す。`H3_FILE_STREAM_HOST_MIN_MB`(既定 1)以上のファイルに無いテンソルが対象) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | bf16 重み(各 ~66GB)がキャッシュに無いとき、`1` のときだけ HF から取得 |
| `H3_LLM_URL` / `H3_LLM_API_KEY` | `http://127.0.0.1:64660` / なし | プロンプト強化に使う外部 LLM と API キー(キーは git 管理外の `config.local.env` からも読む) |
| `H3_LLM_LOCAL_IDLE_S` / `H3_LLM_LOCAL_STOP_ON_GENERATE` | `300` / `auto` | 内蔵 LLM の省メモリ: 最後の利用から 300 秒で llama-server を止める(0 で止めない)/ 生成開始時に止める(`auto` = ホスト RAM 32GB 未満のときだけ、`1` / `0` で強制)。止まっても次の LLM 強化で自動再起動する |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / なし | 内蔵 LLM(Gemma 4 E4B、CPU、ビルド済み `llama-server` を子プロセスで起動)。`auto` = 外部 LLM に接続できないときだけ使う / `1` = 内蔵のみ / `0` = 使わない。GGUF は `models/llm/` へ自動 DL、既にあるファイルは `H3_LLM_LOCAL_GGUF` で指定。`H3_LLAMA_SERVER`(バイナリのパス)、`H3_LLAMA_CPP_TAG`(取得する llama.cpp のタグ)、`H3_LLM_LOCAL_PORT`(既定 64661)も可 |

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/FEATURES.md](docs/FEATURES.md) | **機能と設定の詳細**: 各環境変数の仕組み・成立条件・実測値、VRAM 級ごとの構成、生成モード、UI、LLM 強化 |
| [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) | **技術概要**: アーキテクチャ、各種方式の統合、VRAM 容量別の扱い、性能、設定リファレンス |
| [docs/RESIDENCY.md](docs/RESIDENCY.md) | **常駐リファレンス**: どのモード・設定のどの局面で GPU に何が載っているか |
| [docs/COMMUNITY_IMPROVEMENTS.md](docs/COMMUNITY_IMPROVEMENTS.md) | ComfyUI コミュニティ等の改良の取り込み判断(取り込んだもの / 見送ったもの / 自前実装したもの) |
| [docs/HISTORY.md](docs/HISTORY.md) | **開発履歴**: 日付付きの作業記録、バグと修正、A/B 実測、積み残し |
| [docs/internal/TECHNICAL_REPORT.md](docs/internal/TECHNICAL_REPORT.md) | 内部資料: 2026-08-04〜08-08 の設計判断の背景と、実機で踏んだ罠 |
| [docs/internal/regression_baselines_dev96.json](docs/internal/regression_baselines_dev96.json) | diffusers を上げるときの回帰基準値(`scripts/run_regression.py` で実行) |

会話アプリとしての実例は [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video) と音声会話クライアント [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video)、本番統合先は [diffusers-movie-server](https://github.com/animede/diffusers-movie-server) を参照。

## 構成

```
minimax-h3/
├── app.py                 # FastAPI 本体(port 8611)
├── run.sh                 # diffusers-movie-server 統合用の起動スクリプト(port 8631)
├── run.ps1, run.bat       # Windows 用の起動スクリプト(port 8611。.bat は実行ポリシーを回避して .ps1 を呼ぶ)
├── core/
│   ├── runner.py          # ModularPipeline のロード / 生成ロジック本体
│   ├── blockstream.py     # transformer ブロックの窓付き先読みストリーマ(H3_GROUP_STREAM=own)
│   ├── filestream.py      # 量子化キャッシュのファイルから直接流す(H3_GROUP_STREAM_HOST=file)
│   ├── leanblock.py       # メモリ節約版のブロック forward
│   ├── attnchunk.py       # クエリ分割 / ヘッド分割 attention
│   ├── testream.py        # 投影 TE・token_refiner のストリーミング
│   ├── vaecache.py        # VAE デコーダの fp16 キャッシュ
│   ├── pruned.py          # AdaLN-pruned 版 transformer_ref と量子化キャッシュ
│   ├── adaln_precompute.py, h3_cudagraph.py
│   ├── llm.py, prompt_check.py   # LLM プロンプト強化と検証
│   ├── settings.py, gallery.py
├── static/index.html      # 単一ページ UI(日英)
├── scripts/               # install.sh / install.ps1 / install.bat、ダウンロード・回帰・各種プローブ
├── docs/                  # ドキュメント(上の表)。docs/samples/ にサンプル動画
├── models/                # 量子化済みキャッシュ等(.gitignore 対象)
├── outputs/, logs/        # 生成物・ログ(.gitignore 対象)
└── venv/                  # 専用 venv(.gitignore 対象)
```

## ライセンス

本リポジトリのソースコードの著作権者: © 2026 YUZUKI (animede)。著作権表示と [NOTICE](NOTICE) の内容は、Apache-2.0 第 4 条に基づき再配布時も保持してください。

**このリポジトリのコードは Apache License 2.0**([LICENSE](LICENSE))。モデル重みは含まれない。

使用するモデル / 重みのライセンスは別途それぞれに従うこと:

| 対象 | ライセンス |
|---|---|
| MiniMax-H3 本体の重み | MiniMax Community License(非商用無料、商用は年商 $20M 未満まで、要クレジット) |
| Turbo LoRA(`larryvrh/MiniMax-H3-Turbo-Lora`、`lightx2v/Minimax-h3-Turbo`) | Apache-2.0 |
| Qwen3-VL-32B(text_encoder)/ Qwen3-VL-4B(投影 TE) | Qwen 公式のライセンスに従う |
| diffusers / transformers / torchao / SageAttention 等 | 各パッケージのライセンス |


---

## FILE: README.en.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/README.en.md)

# minimax-h3

[日本語](README.md) | **English**

**Latest: 2026-11-27 JST** — `scripts/install.sh` (Linux / Mac): fixed a stop where Python 3.12 is present but `python3.12-venv` (ensurepip) is not (Ubuntu 24.04 on Google Colab, for example), which left a venv without pip. A venv without pip is now rebuilt, and with uv available it is created by `uv venv --seed` with pip included (contributor PR #2). Before that (11/26): `prepare_lowvram.py` removes bf16 shards left in the HF cache once more at the end. Change history in [docs/HISTORY.en.md](docs/HISTORY.en.md).

**MiniMax H3 (Hailuo 3.0) — running the 33B omnimodal model that generates video + stereo audio in a single denoise pass, on consumer-tier GPUs.**
From text, an image reference or an audio reference it produces **video + stereo audio**, and it can also emit still images only (T2I / Ref2I).
It is built on the diffusers Modular Pipeline (PR #14355, pinned to commit **f37ab93**).
**Verified on a single 6GB card, on two 8GB cards, on a single real RTX 4060 Ti 16GB and on real Windows machines (RTX 3050 6GB, RTX 4060 Laptop 8GB, RTX 3060 12GB)**, and a single 32GB card reaches realtime conversational generation (0.95x real time).

![Main GUI](docs/images/ui_main_en.png)

*A single-page browser UI. Five modes on tabs (video: **T2VA** = text / **FL2VA** = frame / **Ref2VA** = reference; still image: **T2I** / **Ref2I**). The lower section is a gallery of outputs, from which checked videos can be concatenated and exported. Prompts can be enhanced into the official H3 guide format via a local LLM, and FirstBlockCache / Sage / Turbo / quantization / low-VRAM mode are toggled from this panel **without a restart**. Includes a Japanese/English switch.*

## Quick start (Windows, 6 to 16GB of VRAM)

The shortest path for people arriving from social media. Details are under [Installation](#installation) and [Starting](#starting).

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
.\run.bat
```

- Reopen PowerShell after the winget installs. ffmpeg is used for the gallery's length / resolution display and for video concatenation (generation works without it; the log then shows a hint). `install.bat` creates the venv and installs the dependencies (a few minutes).
- `prepare_lowvram.py` prepares the model. It does not fetch 144GB: it downloads the 66GB of weights one shard at a time, quantizes each and deletes it, so **about 60GB of disk** and 16GB of RAM are enough. The time is dominated by your connection (hours). If it stops, rerun it and it continues. Add `--ref` if you also want Ref2VA / Ref2I (video from reference material; +32GB).
- `run.bat` starts the server; open `http://127.0.0.1:8611` in a browser. The first start takes a few minutes to build the projected text encoder's quantized cache. The settings for your GPU size (low-VRAM mode, 4-step turbo) are applied automatically.
- Expectations (4-step turbo, 5 s): RTX 3050 6GB + NVMe 512² in 237 s, RTX 3060 12GB + NVMe 512² in 132 s / 768² in 338 s, RTX 4060 Laptop 8GB 512² in 194 s / 768² in 412 s. Peak VRAM is about 2.8GB at 768² and 3.5GB at 1024². The model's native resolution is a 768 short edge (768², 1344x768); a 3060 12GB is practical at that native size, while a 3050 6GB is realistically used at 512² to 768x512 (below native) to keep the time down. Low-VRAM configurations read 30GB of weights from disk on every step, so on a SATA-class SSD the denoise waits on the disk; an NVMe is recommended. Details under [Real Windows machines](#real-windows-machines).
- Low-VRAM speed is set by two numbers: host-to-GPU transfer speed and SSD read speed. If it feels slow, measure both with `.\venv\Scripts\python.exe scripts\check_bandwidth.py` (expect about 10GB/s pinned transfers on PCIe 4.0 x8, about 5GB/s on x4; under 1GB/s points at the link or BIOS settings).
- If something fails, see the troubleshooting notes in the [Windows](#windows) section and the [Issues](https://github.com/animede/Diffusers_minimax-h3/issues).

## Highlights

- **Runs on consumer GPUs**: block-wise streaming of the int8 transformer (`H3_LOWVRAM=group`), a projected TE (Qwen3-VL-4B + a linear map standing in for the 32B TE), an fp16 video VAE and streaming of the TE / token_refiner / VAE decoder bring 768² x 5 s down to a **2.8GB** peak and 1024² x 5 s to **3.5GB** with 4-step turbo (4.6GB / 6.4GB at 30 steps). With 32GB or 16GB of host RAM the weights are streamed straight from the quantized cache.
- **Fixed costs removed**: the bottleneck was never the denoise but the model load/free. A resident transformer, quantized caches, pinned parking and a KV cache for the reference prefix give t2i 9.7 s / t2va (5 s) 44 s on the 48GB tier and t2i 7.4 s on the 96GB tier.
- **Speed options**: FirstBlockCache, SageAttention (sm_120), the lightx2v turbo LoRA (4 / 8-step distillation), ck-w4a8 quantization and the AdaLN-pruned transformer_ref.
- **Five generation modes plus batches**: T2VA / FL2VA / Ref2VA (ordered references: up to 9 images, 3 videos and 3 audio clips; lip sync from an audio reference) / T2I / Ref2I. Still images and reference-conditioned videos batch up to 24 scenes. **The still-image modes (T2I / Ref2I) do not use the dedicated image model MiniMax-H3-Image: they generate a 22-frame ultra-short video with this video model (MiniMax-H3) and save the center frame as a PNG.** Slower than a dedicated image model, but the look matches exactly when the result is used as an FL2VA first frame or a Ref2VA reference. MiniMax-H3-Image lives in a separate repository, [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server).
- **LLM prompt enhancement**: an external LLM (OpenAI-compatible API, `H3_LLM_URL`) rewrites the input into the format of the official `h3-prompt-writing` skill, with a verify + repair loop before generation. When the external LLM is unreachable it falls back automatically to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0, about 4.4GB, downloaded on first use, thinking off), the same idea as ACE-Step-1.5-fast's easy-music; the built-in LLM is llama.cpp's prebuilt `llama-server` started as a child process, so no compiler is needed and it works as-is on Windows.
- **Quality guarded**: diffusers upgrades and quantized caches are regression-checked by identical-seed MD5.

## Sample

[![t2va 768², 5 s, 4-step turbo, RTX 4060 Laptop 8GB](docs/samples/t2va_768_turbo4_rtx4060laptop.gif)](docs/samples/t2va_768_turbo4_rtx4060laptop.mp4)

*t2va generated on a user's Windows 11 + RTX 4060 Laptop 8GB / 16GB RAM (768², 5 s, 4-step turbo). The GIF is a downscaled, reduced-colour preview; click it for the original mp4 with audio (1MB).*

## Supported GPUs (at a glance)

Measured for t2va 5 s at 768². The detailed rows (80GB / 18GB / 12GB tiers etc.) and the measurement conditions are in [docs/FEATURES.en.md](docs/FEATURES.en.md#vram-configurations-and-residency).

| GPU | Startup flags | Result |
|---|---|---|
| 96GB | (none = default) | peak 92GB / about 160 s |
| 48GB tier | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1` (recommended two-card setup) | t2i 9.7 s / t2va 44.2 s (turbo, 4 steps) |
| single 24 / 32GB | ck-w4a8 + AdaLN-pruned + pinned parking ([details](docs/FEATURES.en.md#oct-2026-update-realtime-conversational-generation-on-a-single-24gb32gb-gpu)) | 320×448, 141 frames in **5.57 s = 0.95x real time** (32GB) / 7.1 s (24GB) |
| 32GB tier | `H3_LOWVRAM=group` | peak 28.7GB / about 280 s |
| single 12GB (real RTX 3060 12GB) | as above (auto picks the 16GB-tier settings) | 4-step turbo, 5 s (entry-level desktop + NVMe): 512² 132 s, 768x512 198 s, 768² 338 s, 1024x768 473 s. Details under "Real Windows machines" below |
| single 16GB (real 4060 Ti / A4000) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | 4060 Ti: peak 11.4GB (Aug 2026). A4000 (2026-11-17): 30 steps peak 8.0GB / 790 s, turbo 4 steps (`H3_TURBO_LORA=1`) peak 7.0GB / 151 s * |
| single 6GB / 8GB | the row above + `H3_TE_STREAM=1 H3_TE_DIET=1` | 30 steps: 768² 4.62GB / 1024² 6.41GB (272 / 552 s on the 48GB box). 4-step turbo (A4000 under a 6GB cap, weights streamed from the quantized cache files): 768² 2.82GB / 211 s, 1024² 3.48GB / 380 s, ref2va 768² 276 s. With 48GB+ of host RAM (pinned, 8GB cap): 768² 3.08GB / 209 s, 1024² 3.73GB / 376 s, fl2va 225 s, ref2va 299 s. Real machines (RTX 3050 6GB, RTX 4060 Laptop 8GB): see "Real Windows machines" below * |
| 8GB x 2 | single-16GB flags + `H3_TE_DEVICE=cuda:1` | peak 7.23GB * |

\* Times depend heavily on transfer and read bandwidth. Low-VRAM setups move 30GB of weights per step, so at 30 steps PCIe bandwidth was the limit (about 25 minutes on a 4060 Ti at Gen3 x4, 790 s on an A4000 at Gen4 x4, 272 s on GPU0 of the 48GB box). **Since the turbo LoRA (4 steps) works in group mode and the number of transfers scales with the step count, the "runs but slowly" caveat is largely gone** (151 s for t2va 5 s on the x4-connected A4000). With less than 48GB of host RAM (file streaming) the 30GB is read from disk every step: on a SATA-class SSD (about 430MB/s) the denoise waits on the disk, on an NVMe the GPU is the limit (table below). The 8GB / 6GB-tier figures were measured on the A4000 under an allocator cap (`H3_VRAM_LIMIT_GB`); real machines land in the same range. Cards other than sm_120 need `H3_ATTN_BACKEND=default` (`H3_PROFILE=auto` sets it when unset). That crossing configurations changes the composition even at the same seed (a trajectory divergence, not degradation) is covered in the [performance overview](docs/FEATURES.en.md#performance-overview).

### Real Windows machines

t2va, 4-step turbo, 5 s (124 frames). In parentheses: denoise + decode (seconds). The entry-level desktop is an i3-10105 (4 cores, PCIe 3.0) / DDR4-2400 / B560 / 32GB RAM; neither the CPU, the memory speed nor PCIe (3.0 x8, 6GB/s transfers) is the bottleneck.

| Machine | Disk | 512x384 | 512² | 768x512 | 768² | Larger |
|---|---|---|---|---|---|---|
| RTX 3050 6GB (entry desktop, Windows 10) | SATA-class SSD (430MB/s) | 294 | 321 (237 + 65) | — | — | — |
| same | NVMe | 163 (108 + 42) | 237 (147 + 65) | 345 (248 + 85) | 558 (444 + 112) | — |
| RTX 3060 12GB (same desktop) | SATA-class SSD | 270 (240 + 26) | 282 (235 + 37) | — | 324 (251 + 62) | 1344x768 711 (590 + 106) |
| same | NVMe | — | 132 (83 + 36) | 198 (139 + 47) | 338 (241 + 66) | 1024x768 473 (385 + 77) |
| RTX 4060 Laptop 8GB / 16GB RAM (gaming laptop, Windows 11) | NVMe | — | 194 (117 + 16) | 279 (191 + 65) | 412 (313 + 85) | — |

- On a SATA-class SSD the denoise is flat at about 240 s up to 768²: that is the time to read 30GB of weights per step, and a faster GPU does not change it. On an NVMe the denoise is set by the GPU (768² takes 444 s on the 3050, 241 s on the 3060).
- The model's native resolution is a 768 short edge (768², 1344x768; the turbo LoRA is trained at 768p). The 3060 12GB is practical at the native 768² in a little under 6 minutes; the 3050 6GB is realistically dropped to 512² to 768x512 (below native) to keep the time down.
- Shared GPU memory stays around 1.5GB during a run (the pinned staging buffers).

## Installation

### Requirements

| | Requirement |
|---|---|
| GPU | Minimum: a **single 6GB card** (runs up to 1024², but 512² to 768x512 is realistic for time) / **single 8GB or 12GB** (the native 768² is practical) / **single 16GB** (all features). **48GB tier** recommended for comfortable use. The model's native resolution is a 768 short edge (768², 1344x768) |
| Host RAM | **Runs with 16GB (NVMe assumed), 24GB or more recommended, 48GB or more also speeds up the denoise.** Below 48GB the weights are streamed straight from the quantized cache files (`H3_GROUP_STREAM_HOST=file`; 30GB is read from disk every step, so the denoise is bound by SSD speed). With 16GB the 4.5GB VAE decoder does not stay in the page cache, so the decode also re-reads from disk every pass; 24GB and up behaves like 32GB. 48GB and up (Linux) pins the ~34GB of int8 weights and the disk reads disappear |
| Disk | About 145GB (T2VA / FL2VA only) / about 207GB (if Ref2VA is also used). The low-VRAM / low-RAM preparation path (`scripts/prepare_lowvram.py`) keeps only the quantized caches, about 60GB (+32GB with `--ref` for Ref2VA) |
| CUDA | 12.8 series (to match torch 2.9.0+cu128). If building SageAttention, `nvcc` must be the same series |
| Python | 3.12 |
| Other | `ffmpeg` (used for gallery concatenation and info retrieval; generation itself works without it) |

### Steps

```bash
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
scripts/install.sh
```

`scripts/install.sh` creates `venv/`, installs [requirements.txt](requirements.txt) and finally prints the torch / diffusers / transformers versions. It needs only Python 3.12 and `git`; nothing is compiled against CUDA, so it finishes in a few minutes. Add `--sage` to also install SageAttention (below), and `--local-llm` to pre-fetch the prebuilt `llama-server` for the built-in LLM (Gemma 4 E4B; below. Even without it, the binary is fetched automatically on the first fallback).

<details>
<summary>Installing by hand (same contents as requirements.txt)</summary>

```bash
python3.12 -m venv venv

# PyTorch (cu128)
venv/bin/pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# ============================================================================
# [IMPORTANT] Pin diffusers to the verified commit f37ab93 (the final merged form of PR #14355).
#             Do not install the PyPI release or the latest main (see the note below)
# ============================================================================
venv/bin/pip install "git+https://github.com/huggingface/diffusers.git@f37ab93e621d5ce206c9662e8291ca8b67d9c555"

# transformers 5.14.1 or later is required (5.1.0 lacks Qwen3VLProcessor.create_mm_token_type_ids)
venv/bin/pip install "transformers==5.14.1" accelerate==1.12.0 safetensors huggingface_hub

# Video/audio muxing and the Web API
venv/bin/pip install av==16.0.1 fastapi==0.104.1 "uvicorn==0.24.0" python-multipart pillow numpy

# 4bit quantization of text_encoder (required for the default H3_TE_QUANT=bnb-4bit)
venv/bin/pip install bitsandbytes==0.49.0

# Only if using int8 quantization for the transformer (H3_TRANSFORMER_QUANT=int8 / H3_LOWVRAM)
#   0.18+ requires torch>=2.11, so pin to 0.17.0
venv/bin/pip install torchao==0.17.0
```

</details>

> **About upgrading diffusers**: installing it is just the one line above (pure Python, no build; it takes well under a minute with `git` available). Care is needed only **when upgrading diffusers later**: a newer version can change the compute path and therefore the output, so keep the commit pinned. The move to f37ab93 itself was made only after every path (t2i / t2va / batch / ref2va / reference batch) was verified equivalent to the old pin abc5e9b by identical-seed MD5. Any further upgrade should go through the same regression check (`scripts/run_regression.py` with the baselines in docs/internal). Not needed for ordinary use.

### Windows

All dependencies have Windows wheels (torch 2.9.0+cu128 from the PyTorch index, torchao as PyPI's pure-Python wheel, bitsandbytes / av / psutil as PyPI Windows wheels). No compiler or Visual Studio Build Tools is needed. SageAttention has no Windows build and is not used; `H3_PROFILE=auto` sets `H3_ATTN_BACKEND=default` (SDPA) at startup. The OS-specific code paths (RAM readout, file read-ahead hints, temp-file location, the built-in LLM child process) have Windows branches. On Windows the Hugging Face cache is set not to use symlinks (`HF_HUB_DISABLE_SYMLINKS=1`) automatically, to avoid a huggingface_hub race that aborts the first download with `WinError 1314` on PCs without Developer Mode (disk usage is unchanged).

Requirements: Python 3.12 (callable as `py -3.12`), git, an NVIDIA driver for CUDA 12.8. `ffmpeg` is optional (gallery concatenation). If git or Python is missing, install them with winget (then reopen PowerShell):

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
winget install --id Gyan.FFmpeg -e --source winget
```

```powershell
git clone https://github.com/animede/Diffusers_minimax-h3.git
cd Diffusers_minimax-h3
.\scripts\install.bat
```

`install.bat` runs `scripts\install.ps1` regardless of PowerShell's execution policy (Windows does not run unsigned `.ps1` files directly by default). It creates the venv and installs torch (cu128) and `requirements.txt`; add `-LocalLLM` to also fetch the built-in LLM binary. Then fetch the model:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

(On a PC with 48GB+ VRAM and 64GB+ RAM, `scripts\download_t2va.py` works as well; see "Preparing a low-VRAM / low-RAM PC" below.) To install by hand, run these in order (PyPI's Windows torch is CPU-only, so torch must come first from the PyTorch index):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

For starting the server (passing environment variables, per-tier examples) see the Windows part of [Starting](#starting). To run the `.ps1` files directly, either add Bypass (`powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1`) or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

The built-in LLM (the fallback when the external LLM is unreachable) can be added later even if you installed without `-LocalLLM`. It is fetched automatically on the first fallback; to fetch it ahead of time run the following (line 1 = the prebuilt `llama-server.exe`, about 25MB; line 2 = the Gemma 4 E4B GGUF, about 4.4GB, optional):

```powershell
.\venv\Scripts\python.exe scripts\fetch_llama_server.py
.\venv\Scripts\python.exe scripts\fetch_llama_server.py --model
```

Running `.\scripts\install.bat -LocalLLM` later does the same (already-installed packages are left as they are). Details: [Built-in LLM (optional)](#built-in-llm-optional) below.

> Windows status (2026-11-20): verified through install, `prepare_lowvram.py`, startup and generation (30 steps and 4-step turbo) on Windows 11 + RTX 4060 Laptop 8GB / 16GB RAM and on Windows 10 + RTX 3050 6GB and RTX 3060 12GB (entry-level desktop, i3-10105 / 32GB RAM). Times are in [Real Windows machines](#real-windows-machines) above. Windows-specific measures in place: Hub cache without symlinks, retried source-shard deletion, execution-policy-free `.bat` launchers, an automatic VRAM cap (card size minus 1.2GB), always streaming from file instead of pinned RAM, and decoder residency decided from the physical free VRAM (history in [docs/FEATURES.en.md](docs/FEATURES.en.md)). Reading Task Manager: dedicated GPU memory looks higher than the peak the server reports (it includes torch's pool and the desktop); shared GPU memory is the pinned staging RAM and sits around 1.5-2GB. If shared memory keeps climbing past 4GB during a run, VRAM has spilled and generation crawls (should not happen on current versions; open an Issue if it does). The image server built the same way, [minimax-h3-image-server](https://github.com/animede/minimax-h3-image-server), generates fine on Windows 11 + RTX 3050. Please report problems in the [Issues](https://github.com/animede/Diffusers_minimax-h3/issues).

### SageAttention (optional, enabled by default; Linux only)

The default `H3_ATTN_BACKEND=sage` needs a build for sm_120 (there is no prebuilt Linux wheel). **If you do not build it, start with `H3_ATTN_BACKEND=default`** (denoise is about 12% slower; no functional difference).

```bash
CUDA_HOME=/usr/local/cuda-12.8 scripts/install.sh --sage     # or scripts/build_sageattention.sh
```

`--sage` installs a prebuilt wheel from `third_party/wheels/` if one is there, and otherwise builds from source.

### Built-in LLM (optional)

LLM enhancement uses an external OpenAI-compatible API (`H3_LLM_URL`) by default and falls back automatically to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0, thinking off) when it is unreachable. The built-in LLM is llama.cpp's prebuilt `llama-server` from the [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) (CPU build, about 25MB, pinned tag `b12475`, override with `H3_LLAMA_CPP_TAG`): the server starts it as a child process and talks to it over the OpenAI-compatible API (default `127.0.0.1:64661`, `H3_LLM_LOCAL_PORT`). **No compiler is needed**; the Linux x64 / Windows x64 / macOS binary is chosen automatically.

```bash
scripts/install.sh --local-llm            # pre-fetch llama-server into tools/llama/ (optional)
venv/bin/python scripts/fetch_llama_server.py --model   # also pre-fetch the GGUF, about 4.4GB (optional)
```

On low-RAM PCs (16GB class) video generation and the built-in LLM compete for RAM, so llama-server is stopped 300 s after its last use (`H3_LLM_LOCAL_IDLE_S`) and, on hosts under 32GB RAM, also when a generation starts (`H3_LLM_LOCAL_STOP_ON_GENERATE=auto`). The next LLM enhancement restarts it in a few seconds.

Both are optional and can be run later on an existing installation. On the first fallback `llama-server` is fetched into `tools/llama/` and the model into `models/llm/` automatically. The child process stops with the server, and if the server is killed hard it is taken down with it (Windows: Job Object with KILL_ON_JOB_CLOSE; Linux: `PR_SET_PDEATHSIG`), so it never lingers as an orphan (verified with SIGKILL on Linux; Windows untested). To use existing files set `H3_LLAMA_SERVER=<path to llama-server>` and `H3_LLM_LOCAL_GGUF=<path to the GGUF>` (a llama-server you started yourself that answers on `H3_LLM_LOCAL_PORT` is used as-is).

### Obtaining the model

**PCs with 32GB of VRAM or less, or less than 48GB of host RAM, should use "Preparing a low-VRAM / low-RAM PC" below instead** (the steps here fetch the full 144GB, which is wasteful on the low-VRAM tiers, and a low-RAM PC cannot build the quantized cache and stops at startup).

`MiniMaxAI/MiniMax-H3` is 498.6GB, but T2VA / FL2VA only need `transformer/` (66.3GB) + `text_encoder/` (66.7GB) + `vae/` (10.4GB) + `audio_vae/` (0.6GB) + configs, **about 144GB**. Fetch `transformer_ref/` (Ref2VA, 66.3GB) only when needed, and **never `snapshot_download` the whole repo**.

```bash
venv/bin/python scripts/download_t2va.py
```

It uses `allow_patterns` internally to fetch only the needed subfolders; `logs/du_monitor.log` tracks the cache size while downloading. When the bf16 transformer / transformer_ref weights are not cached, they are downloaded only with `H3_ALLOW_WEIGHT_DOWNLOAD=1` (the default fails with a hint).

### Preparing a low-VRAM / low-RAM PC

The `H3_LOWVRAM=group` configuration (32GB tier and below) that streams the weights from the quantized cache files **requires the int8 quantized cache (`models/prequant/transformer_int8`, about 32GB) to exist first**. Without it the server falls back to quantizing from bf16 (66GB) and stops at the 40GB host-RAM guard (this is exactly what happens on a 16GB-RAM PC). `scripts/prepare_lowvram.py` builds that cache by fetching the bf16 shards (14 × 4.7GB) one at a time, quantizing each and deleting it: a few GB of RAM, no VRAM needed (CPU quantization by default, `--device cuda` optional), and disk peaks at one shard plus the output. The result is **bit-identical** to the cache the server would build itself (verified). It also builds the VAE decoder fp16 cache and pre-fetches the projected TE (Qwen3-VL-4B + ClipProj). The 32B text_encoder is not downloaded.

```bash
venv/bin/python scripts/prepare_lowvram.py
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py
```

The time is dominated by downloading 66GB (quantization takes seconds to tens of seconds per shard). The cache is checked against the Hugging Face snapshot, but on a PC without the bf16 originals (where a rebuild is impossible anyway) the check is skipped and the cache is used as-is. Reruns are idempotent: an existing cache is only inspected, which takes seconds. `--keep-source` keeps the bf16 shards (+66GB, faster rebuilds). The projected TE's NF4 cache is built by the server on first start (fine with 8GB VRAM / 16GB RAM).

**Add `--ref` if you also use Ref2VA / Ref2I.** The reference modes use a different set of weights, `transformer_ref/` (66GB; every shard differs from `transformer`), so its cache `models/prequant/transformer_ref_int8` (about 32GB) is built the same way. On a host with less than 48GB RAM the reference modes stop at startup without it (the error tells you how to build it). It can be added later (reruns are idempotent; an existing `transformer_int8` is only inspected, which takes seconds).

```bash
venv/bin/python scripts/prepare_lowvram.py --ref
```

Windows:

```powershell
.\venv\Scripts\python.exe scripts\prepare_lowvram.py --ref
```

## Starting

### Linux

```bash
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

With nothing set, **`H3_PROFILE=auto`** picks the 96 / 80 / 48 / 32 / 16 / 8GB-tier defaults (the same flags as the launch examples below) from the compute GPU's total VRAM. On the 32GB-and-below tiers (group) the turbo LoRA (4 steps) is on by default as well, because in group mode it must be applied at startup to be usable from the GUI at all (untick turbo in the GUI for 30 steps, or set `H3_TURBO_LORA=0` to disable it at startup). On 48GB and above, use the GUI's turbo checkbox per request. Setting any of the main profile variables (`H3_LOWVRAM`, `H3_TE_PROJ`, ...) switches to manual mode and nothing is filled in automatically. When the default sage build cannot be used (a GPU other than sm_120, or sageattention not installed) `H3_ATTN_BACKEND=default` is also set automatically. The chosen tier is shown in `/api/status` under `profile` and in the startup log.

Open `http://<host>:8611/` in a browser. The first start takes a few minutes to load the models (by default the TE is NF4-quantized and kept on the GPU). `run.sh` is the launcher used by the diffusers-movie-server integration: it starts the same server on internal port 8631 with the conversation-oriented defaults (turbo LoRA etc.).

Launch examples per VRAM tier:

```bash
# Real 16GB, single card (e.g. an sm_89 4060 Ti; anything other than sm_120 needs H3_ATTN_BACKEND=default)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 6GB / 8GB single card (the 16GB flags plus TE streaming and diet; H3_CACHE=none goes lower still)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1 H3_TE_DIET=1 \
  H3_ATTN_BACKEND=default \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 8GB x 2 (TE on the second GPU)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  H3_ATTN_BACKEND=default H3_TE_DEVICE=cuda:1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 32GB tier
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 48GB tier (recommended: full speedups; drop H3_TE_DEVICE when using ref2va)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611

# 96GB tier (everything resident, no env)
venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

The full environment set and caveats for realtime conversational generation on a single 24 / 32GB card (ck-w4a8 quantization + pinned parking) are in [docs/FEATURES.en.md](docs/FEATURES.en.md#oct-2026-update-realtime-conversational-generation-on-a-single-24gb32gb-gpu).

To verify operation before using the UI, run the regression probe (writes `outputs/probe_t2va.mp4` and `outputs/probe_report.json`):

```bash
venv/bin/python scripts/probe_t2va.py
```

### Windows

`run.bat` starts the same server (it just runs `run.ps1` regardless of the execution policy; default 127.0.0.1:8611; `H3_HOST` / `H3_PORT` change the bind address and port):

```powershell
.\run.bat
```

Set environment variables one per line with `$env:` before starting. The Linux launch examples above carry over directly: turn each `H3_X=value` into `$env:H3_X = "value"` and then call `.\run.ps1`. Single-16GB example (cards other than sm_120 need `H3_ATTN_BACKEND=default`; `H3_PROFILE=auto` sets it when unset):

```powershell
$env:H3_LOWVRAM = "group"
$env:H3_TE_PROJ = "NicoLab28/ClipProj-MiniMax-H3"
$env:H3_VIDEO_VAE_FP16 = "1"
$env:H3_ATTN_BACKEND = "default"
.\run.bat
```

With nothing set, `H3_PROFILE=auto` picks the tier from the GPU's total VRAM, so plain `.\run.bat` is a fine first start (on the 32GB-and-below tiers the turbo LoRA is enabled automatically; add `$env:H3_TURBO_LORA = "1"` only if you want turbo as the default on 48GB and above). Variables stay set until that PowerShell window is closed (to start again with a different configuration, remove them with e.g. `Remove-Item Env:H3_LOWVRAM` or open a new window).

If you run `run.ps1` directly and the execution policy blocks it (`run.bat` does this internally):

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The regression probe:

```powershell
.\venv\Scripts\python.exe scripts\probe_t2va.py
```

## Usage

### UI

A single-page browser UI: inputs and settings on the left, results and the gallery on the right. The "日本語 / English" buttons at the top right switch the display language.

**Mode tabs**

- The upper row is video (**T2VA** = text, **FL2VA** = first / last frame, **Ref2VA** = reference material), the lower row is stills (**T2I**, **Ref2I**). Switching tabs shows only the inputs that mode needs.
- The still-image tabs do not use MiniMax-H3-Image: the video model generates a 22-frame (or 5-frame) ultra-short clip and the center frame is saved as a PNG, with the mp4 kept alongside.

**Prompt and LLM enhancement**

- Prompts work in Japanese as well as English. "Prompt guide" opens a cheat sheet for hand-written prompts: the official brief structure (scene → subject → action → camera → sound → ending), the `CUT n [X-Y s]:` notation and the dialogue tag `<d>`.
- "LLM enhance" rewrites the prompt with the external LLM (or the built-in Gemma 4 E4B when it is unreachable) and replaces the prompt field. Four modes: `storyboard` (expands into multi-cut timecodes), `brief` (single-shot detailing in the official brief format), `h3-official` (strict adherence to MiniMax's `h3-prompt-writing` skill; the current tab is sent as the task, output language English or Japanese, and a verify + auto-repair pass runs before generation with warnings for unresolved issues), `translate` (English translation only). "Undo" restores the previous prompt once.
- The projected TE (the default on the low-VRAM tiers) cannot handle the `<d>` dialogue tag, so prompts containing it are flagged before generation (use an audio reference for dialogue instead).

**Inputs per mode**

- FL2VA: drag and drop the first / last frame image (either one is enough; with both, the clip is generated between them).
- Ref2VA / Ref2I: with an audio reference, the API's `vocal_lock=true` keeps that audio as the output audio and makes the mouth follow it (Vocal Lock; no UI switch). Add multiple reference files (up to 9 images, 3 videos, 3 audio clips, 12 in total). **The order you add them is the reference order**, and each can be referred to in the prompt by the label shown on its tile, such as `<Picture 1>`. With exactly one audio reference the duration can be left blank and follows the audio. Ref2I picks a frame count (22 / 5) and saves the center frame as a PNG.
- T2I: frame count (22 = the quality-verified default, 5 = experimental).
- Batch generation (T2I / Ref2I / Ref2VA): ticking it turns the prompt field into "one line = one scene" (up to 24 scenes); references, seed, resolution and steps are shared across scenes. Model loading and reference encoding are shared between scenes, which pays off especially in low-VRAM mode and with references. Ref2VA batches require the duration.

**Size, duration, steps, seed**

- Resolution: the presets 768² / 1344×768 (landscape) / 768×1344 (portrait), "custom" for an explicit width × height (rounded to a multiple of 32, 256-2048, with a warning beyond the native range), or "auto" for the server default (16:9 for Ref2VA).
- Duration 5-15 s. The actual frame count is rounded up to 17n + 5 and shown as "actual length" (e.g. 5 s → 124 frames, 5.17 s).
- Steps switch automatically to the recommended value (4 / 30) when Turbo LoRA is toggled; a hand-edited value is used as-is. Leave seed blank for random.

**Instant settings (no reload, sent per request)**

- FirstBlockCache (+ threshold), Sage Attention (sm_120 only), Turbo LoRA (4 steps recommended), mute (no audio track).
- Combinations the server's configuration cannot run are greyed out with the reason in a tooltip (for example turbo cannot be selected in group mode unless it was applied at startup). Enabling turbo switches FirstBlockCache off automatically.

**Generation and results**

- "Generate" shows a progress bar with the phase (TE encode / denoise n/N / decode) and elapsed seconds. On completion the right side shows the video (or PNG plus mp4) with mode, resolution, length, steps, seed, denoise / decode times, peak VRAM, audio presence, the instant settings and the full prompt used (collapsible).
- The LLM-enhancement message says whether the external or the built-in LLM handled it.

**Settings that need a reload (collapsible header)**

- transformer quantization (none / int8; ck-w4a8 is startup-only), text_encoder quantization, TE layer pruning, projected TE and its quantization, low-VRAM mode (0 / 1 / group), fp16 video VAE. Changing any of them enables "Apply (reload)" with an estimated time; applying frees and reloads the models without restarting the process.
- Controls are greyed out to match the server's constraints, e.g. text_encoder quantization and TE pruning are disabled while the projected TE is on (the 32B TE is not loaded at all).

**Server status**

- A status block under the inputs refreshes every 4 seconds: load state (transformer / transformer_ref / VAE / TE), GPU allocation and peak, host RAM, `profile` (what the automatic profile chose), the lowvram / quantization / attention / turbo / TE configuration, where the streamed weights live, and the LLM endpoint plus built-in LLM state.

**Gallery**

- Generated videos and stills, newest first. Click a thumbnail for the lightbox view; each item can be downloaded.
- Tick items and use "Delete selected" or "Concatenate selected". Concatenation joins **in the order ticked** (stills are excluded) and saves `outputs/concat_<timestamp>.mp4`.

### API

| Endpoint | What it does |
|---|---|
| `POST /api/t2va` | Text to video + audio. `prompt`, `resolution` (`768x768` / `768x1344` / `1344x768`), `seconds` (5-15), `num_inference_steps`, `seed` |
| `POST /api/fl2va` | The above plus `image` / `last_image` (at least one) |
| `POST /api/ref2va` | Reference-conditioned generation. `references` (up to 9 images, 3 videos, 3 audio clips; order matters). With exactly one audio reference, `seconds=0` uses the audio length. `still=1` + `frames` gives a reference-conditioned still (ref2i). `vocal_lock=true` (or `H3_VOCAL_LOCK=1` at startup) enables **Vocal Lock**: the audio reference is kept as the output audio and the mouth follows it (lip sync; works in the low-VRAM configuration too, encode-phase peak 3.9GB under an 8GB cap). API / env only, no UI switch |
| `POST /api/t2i` | Still image. `frames` (default 22, or 5) instead of `seconds`. Saves an ultra-short mp4 and the center-frame PNG |
| `POST /api/t2i_batch` / `POST /api/ref2i_batch` / `POST /api/ref2va_batch` | Batch generation. Repeat `prompts` per scene (up to 24). The reference variants share `references`; `ref2va_batch` requires a common `seconds` |
| `POST /api/prompt/enhance` | LLM prompt enhancement. `{text, mode, seconds, task, lang}` |
| `GET /api/status` / `GET /api/progress` / `POST /api/interrupt` | Load state and VRAM / RAM, progress of the running job, interrupt |
| `GET /api/settings` / `POST /api/settings/apply` | Read and apply (reload) the settings switchable from the UI |
| `POST /api/admin/unload` / `POST /api/admin/reload` | Free and reload the models |
| `GET /api/outputs` / `POST /api/outputs/delete` / `POST /api/outputs/concat` | Gallery listing, deletion and concatenation |

All endpoints take multipart/form-data. Each mode is described in [Generation modes in docs/FEATURES.en.md](docs/FEATURES.en.md#generation-modes).

## Main environment variables

Defaults shown. Most **can also be toggled from the UI**, so set them only to change them permanently. The full list with measurements is in [docs/FEATURES.en.md](docs/FEATURES.en.md).

| Variable | Default | Meaning |
|---|---|---|
| `H3_PROFILE` | `auto` | Automatic main profile. `auto` picks a tier from total VRAM only when none of the main profile variables is set (the turbo LoRA is also enabled on the 32GB-and-below tiers) / `96gb`..`8gb` forces a tier (use this when emulating a smaller card with `H3_VRAM_LIMIT_GB`) / `off` |
| `H3_LOWVRAM` | `0` | `1` = 48GB-tier phase cycling / `group` = block offload for the 32GB tier and below (streams the int8 weights block by block) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` shrinks the transformer from 66.3GB to 34GB. `ck-w4a8` uses ComfyUI's W4A8 kernel (needs `comfy-kitchen`) |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder quantization (`none` = bf16, 66.7GB) |
| `H3_TE_PRUNE` | `0` | Remove unused upper layers of the TE (output unchanged, -3.6GB) |
| `H3_TE_PROJ` | (off) | **Projected TE**: replaces the 32B TE with Qwen3-VL-4B + a linear map (repo id or .safetensors path; quality is an approximation) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | Quantization of the projected TE (NF4 = 3.11GB; `none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (off) | Park the TE on a second GPU (e.g. `cuda:1`; the 32B TE needs a 20GB-class card, the projected TE fits an 8GB-class one) |
| `H3_TE_STREAM` / `H3_TE_DIET` | `0` / `0` | Streaming and diet of the projected TE / token_refiner (the key to the 6GB / 8GB setups) |
| `H3_GROUP_STREAM` / `H3_GROUP_STREAM_HOST` | `own` / `auto` | Weight transfer scheme in group mode. `HOST` streams straight from the quantized cache files (`file`) when the host has less than 48GB of RAM; on Windows always `file` regardless of RAM (WDDM moves pinned RAM pathologically slowly) |
| `H3_VIDEO_VAE_FP16` | `0` | fp16-ify the video VAE (reduces the decode peak; required at 16GB and below) |
| `H3_KEEP_TRANSFORMER` | `0` | Keep the transformer resident, removing the reload fixed cost |
| `H3_PRUNED` / `H3_PRUNED_QUANT` | `0` / `int8wo` | Load the AdaLN-pruned transformer_ref for ref2va (requires `H3_TRANSFORMER_QUANT=int8`) and its quantization scheme |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache (denoise -25%; measured ineffective on the int8+SDPA trajectory) |
| `H3_ATTN_BACKEND` | `sage` | The sage build is sm_120-only. **Any other GPU must use `default` (SDPA)** |
| `H3_VRAM_LIMIT_GB` | (unset) | Cap on the VRAM this process may allocate (GB). **On Windows, unset means an automatic cap of card size minus 1.2GB** (`H3_VRAM_AUTO_CAP_MARGIN_GB`): Windows torch cannot shrink its grown pool, so 6GB cards spill past physical VRAM and crawl; the cap stops the growth (at the cap the allocator releases unused blocks and reuses them). Even a 6GB card (cap 4.8GB) fits 1024², 5 s (peak 3.5GB in the A4000 emulation); requests beyond the cap fail with an OOM instead of crawling in the spilled state. To disable, set a value at or above the card size (e.g. `99`) |
| `H3_GPU_POOL_DEFRAG` | `auto` | Right before the decode, move the live GPU tensors (about 1.2GB) to the CPU, release torch's whole pool, and move them back (`auto` = Windows only: Windows torch cannot shrink its grown pool, so 6GB cards spill during the decode and crawl). `1` / `0` |
| `H3_VAE_DECODER_RESIDENT` | `auto` | In file-streaming decodes, keeps as many VAE decoder blocks (36, 4.5GB) on the GPU as free VRAM allows for the decode and streams the rest from file (every resident block removes its share of the per-pass transfer; same output). A4000 emulation: 30/36 resident under an 8GB cap, 768² decode 86 to 34 s (peak 6.2GB); 15/36 under a 6GB cap, decode 51 s (peak 4.2GB). `1` forces all, `0` disables. Headroom via `H3_VAE_DECODER_RESIDENT_MARGIN_GB` (default 1.0 on Linux, 2.5 on Windows, whose torch pool grows more) |
| `H3_VOCAL_LOCK` | `0` | Enables Ref2VA Vocal Lock by default (the audio reference is kept as the output audio and the mouth follows it). The request's `vocal_lock` takes precedence |
| `H3_TURBO_LORA` | `0` | lightx2v 4 / 8-step distilled LoRA (downloads about 780MB on first use). Works in `group` too (needs the in-house streamer `H3_GROUP_STREAM=own` and the flag at startup; with `file` streaming the LoRA, about 1.2GB, is streamed from host RAM block by block; tensors missing from the files and at least `H3_FILE_STREAM_HOST_MIN_MB` (default 1) large are handled this way) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | Download the bf16 weights (~66GB each) from HF only when `1` if they are not cached |
| `H3_LLM_URL` / `H3_LLM_API_KEY` | `http://127.0.0.1:64660` / none | External LLM for prompt enhancement and its API key (the key is also read from the git-ignored `config.local.env`) |
| `H3_LLM_LOCAL_IDLE_S` / `H3_LLM_LOCAL_STOP_ON_GENERATE` | `300` / `auto` | Memory saving for the built-in LLM: stop llama-server 300 s after its last use (0 = never) / stop it when a generation starts (`auto` = only on hosts under 32GB RAM, `1` / `0` to force). It restarts automatically on the next LLM enhancement |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / none | Built-in LLM (Gemma 4 E4B, CPU, a prebuilt `llama-server` child process). `auto` = only when the external LLM is unreachable / `1` = built-in only / `0` = never. The GGUF is downloaded to `models/llm/`; point `H3_LLM_LOCAL_GGUF` at an existing file instead. Also `H3_LLAMA_SERVER` (binary path), `H3_LLAMA_CPP_TAG` (llama.cpp tag to fetch), `H3_LLM_LOCAL_PORT` (default 64661) |

## Documentation

| Document | Contents |
|---|---|
| [docs/FEATURES.en.md](docs/FEATURES.en.md) | **Features and settings in detail**: how each environment variable works, its preconditions and measurements, per-VRAM-tier configurations, generation modes, UI, LLM enhancement |
| [docs/TECHNICAL_OVERVIEW.en.md](docs/TECHNICAL_OVERVIEW.en.md) | **Technical overview**: architecture, how the techniques are integrated, handling per VRAM tier, performance, configuration reference |
| [docs/RESIDENCY.en.md](docs/RESIDENCY.en.md) | **Residency reference**: what is on the GPU at each phase, per mode and setting |
| [docs/COMMUNITY_IMPROVEMENTS.en.md](docs/COMMUNITY_IMPROVEMENTS.en.md) | Decisions on community improvements (adopted / investigated and declined / reimplemented in-house) |
| [docs/HISTORY.en.md](docs/HISTORY.en.md) | **Development history**: dated work log, bugs and fixes, A/B measurements, backlog |
| [docs/internal/TECHNICAL_REPORT.en.md](docs/internal/TECHNICAL_REPORT.en.md) | Internal document: the background to design decisions from 2026-08-04 to 08-08 and the pitfalls hit on real hardware |
| [docs/internal/regression_baselines_dev96.json](docs/internal/regression_baselines_dev96.json) | Regression baselines for diffusers upgrades (run with `scripts/run_regression.py`) |

For a conversational application built on this server see [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video) and the voice client [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video); the production integration lives in [diffusers-movie-server](https://github.com/animede/diffusers-movie-server).

## Layout

```
minimax-h3/
├── app.py                 # FastAPI server (port 8611)
├── run.sh                 # launcher for the diffusers-movie-server integration (port 8631)
├── run.ps1, run.bat       # Windows launcher (port 8611; the .bat runs the .ps1 bypassing the execution policy)
├── core/
│   ├── runner.py          # ModularPipeline loading / generation logic
│   ├── blockstream.py     # windowed prefetch streamer for transformer blocks (H3_GROUP_STREAM=own)
│   ├── filestream.py      # streams weights straight from the quantized cache files (H3_GROUP_STREAM_HOST=file)
│   ├── leanblock.py       # memory-lean block forward
│   ├── attnchunk.py       # query-chunked / head-grouped attention
│   ├── testream.py        # streaming of the projected TE and token_refiner
│   ├── vaecache.py        # fp16 cache of the VAE decoder
│   ├── pruned.py          # AdaLN-pruned transformer_ref and its quantized cache
│   ├── adaln_precompute.py, h3_cudagraph.py
│   ├── llm.py, prompt_check.py   # LLM prompt enhancement and verification
│   ├── settings.py, gallery.py
├── static/index.html      # single-page UI (ja / en)
├── scripts/               # install.sh / install.ps1 / install.bat, download, regression and probe scripts
├── docs/                  # documentation (table above); docs/samples/ holds the sample video
├── models/                # quantized caches etc. (git-ignored)
├── outputs/, logs/        # outputs and logs (git-ignored)
└── venv/                  # dedicated venv (git-ignored)
```

## License

Copyright holder of the source code in this repository: © 2026 YUZUKI (animede). Keep the copyright notice and the contents of [NOTICE](NOTICE) when redistributing, as required by Apache-2.0 section 4.

**This repository's code is licensed under Apache License 2.0** ([LICENSE](LICENSE)). Model weights are not included.

The licenses of the models / weights used must each be followed separately:

| Target | License |
|---|---|
| MiniMax-H3's own weights | MiniMax Community License (free for non-commercial use; commercial use free up to $20M/year revenue, credit required) |
| Turbo LoRA (`larryvrh/MiniMax-H3-Turbo-Lora`, `lightx2v/Minimax-h3-Turbo`) | Apache-2.0 |
| Qwen3-VL-32B (text_encoder) / Qwen3-VL-4B (projected TE) | Follows Qwen's official license |
| diffusers / transformers / torchao / SageAttention etc. | Each package's own license |


---

## FILE: docs/COMMUNITY_IMPROVEMENTS.en.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/docs/COMMUNITY_IMPROVEMENTS.en.md)

# Community Improvements Adoption Log (MiniMax H3)

[日本語](COMMUNITY_IMPROVEMENTS.md) | **English**

A record of work adopting improvements from the ComfyUI community and elsewhere into this
app (the diffusers path), starting from the original (MiniMaxAI/MiniMax-H3 + diffusers PR #14355). 2026-08-04 through 08-06.

All verdicts are based on **real-machine A/B comparisons at the same seed**. "Quality parity"
means confirmed by visual inspection + PSNR + audio correlation; "equivalent" means confirmed
down to mp4 MD5/byte match.

---

## A. Adopted (community-originated)

### A-1. FirstBlockCache (equivalent to ComfyUI's EasyCache) — ON by default

| | |
|---|---|
| Source | ComfyUI community (kijai's EasyCache usage; posts by @gosrum / @umiyuki_ai) |
| Community reports | 1.67x with EasyCache; 6min → 2.5min combined with Sage |
| Adoption approach | Adopted diffusers' official `FirstBlockCache` (a similar step-to-step caching scheme). `H3_CACHE`/`H3_CACHE_THRESHOLD` |
| Measured | Denoise 157s → **118s (-25%)**, 7 of 30 steps skipped. At threshold 0.1, 81.5s (1.92x) but composition drift |
| Quality | PSNR 31.8–34.3dB, audio correlation 0.979. Indistinguishable by eye |
| Verdict | **threshold 0.05 made default**. 0.1 is opt-in |
| Gotcha | H3 blocks are not registered in the PR branch's `TransformerBlockRegistry` → requires own registration. Must call `_reset_stateful_cache()` + `cache_context()` per request (verified via byte match across two consecutive runs at the same seed) |
| Commit | `14afdfc` |

### A-2. Sage Attention — ON by default

| | |
|---|---|
| Source | Same post as above ("Sage alone cuts 25%") |
| Adoption approach | **Built thu-ml/SageAttention from source** for sm_120 (no prebuilt wheel exists for Linux; only a Windows build is published). `H3_ATTN_BACKEND` |
| Measured | Denoise 118s → **104s (-12%)**. Falls short of the community-reported -25% |
| Quality | Fully deterministic (byte match across two runs at the same seed). Visually equivalent (the 21dB PSNR is trajectory drift from the int8-QK approximation, not degradation) |
| Verdict | **Made default** (`H3_ATTN_BACKEND=default` reverts to conventional SDPA) |
| Gotcha | Build requires `MAX_JOBS=4 NVCC_THREADS=2` plus a systemd-run memory cap (unrestricted parallel nvcc has a history of exhausting host RAM and taking the whole system down). Must explicitly set `CUDA_HOME=/usr/local/cuda-12.8` (the default cuda-13.0 mismatches torch cu128 and the build fails) |
| Commit | `9c7e6a6`, `scripts/build_sageattention.sh` |

### A-3. Latent upscaler (two-stage generation / hires-fix) — opt-in

| | |
|---|---|
| Source | [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler) (via a post by @umiyuki_ai) |
| Mechanism | First-half denoise at low resolution → spatial 2x interpolation of the video latent only → renoise → finish at high resolution. No trained upscaler used |
| Adoption approach | `upscale=1` on `/api/t2va` / `/api/i2v`, `H3_HIRES_DENOISE` |
| Measured | 768² → **1536²** takes 645s / peak 88.0GB (upscale=0 is 181s / 92.1GB) |
| Quality | Composition stays consistent while real detail (e.g. fur) is added. Background detail drifts slightly from the re-denoise (inherent to hires-fix) |
| Verdict | **opt-in** (OFF by default) |
| Gotcha | **The interpolation target must be the x0 estimate, not the noisy latent.** Interpolating the noisy latent amplifies checkerboard noise into full-frame noise (reproduced on real hardware → confirmed the reference implementation also uses `denoised_output` and fixed accordingly). Changing resolution requires rebuilding `build_packed_sequence()` and `row_timestep_plan` |
| Commit | `e9a45a7` |

### A-4. Turbo LoRA (4/8-step distillation) — opt-in

| | |
|---|---|
| Source | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) (trained by Ostris, Apache 2.0). Practical info obtained from a post by @PhotogenicWeekE |
| Community reports | 1280×736/10s at 8 steps goes from 272s → 160s. **"4–7 steps don't work"** |
| Adoption approach | `H3_TURBO_LORA=1` (8 steps by default). Uses the original, not the ComfyUI repack |
| Measured | 8 steps **87.7s (-46%)** / 16 steps 98.4s / 4 steps 39.6s (baseline 30 steps is 163.5s) |
| Quality | 8 steps approaches the baseline. 16 steps is on par with the baseline. 4 steps is somewhat soft, with weaker audio |
| Verdict | **opt-in** (OFF by default because the author explicitly labels the LoRA "demo/preview, still training"; set up so that once a finished version is released, it can be swapped in and the default decided from A/B alone) |
| New finding | **"4–7 steps don't work" is likely caused by ComfyUI's standard sampler.** In this implementation (which correctly integrates the dual schedule of video shift 12 / audio shift 3), no audio corruption occurred even at 4 steps. Note that no fix was needed for the shift wiring — bit-exact match of the sigma grid confirmed the PR implementation's default values were already correct |
| Gotcha | LoRA keys are in ComfyUI's fused-QKV naming convention → applied via `fuse_projections()` plus a runtime delta. **`fuse_projections()` does not delete the old to_q/k/v, leaking +12.8GB** (discovered via an actual OOM, fixed with an explicit delete) |
| Commit | `2ab3100` |

### A-5. Block-level group offload (24–32GB class support) — opt-in

| | |
|---|---|
| Source | ComfyUI's "runs on 24GB with INT8 + layerwise offload" report + PR #14355's streamed offload support + knowledge established in the sister project (diffusers-server) |
| Adoption approach | `H3_LOWVRAM=group`. Load the int8 transformer on CPU → `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)` |
| Measured | **peak 28.7GB** under a 32GB cap, denoise 220.8s (~2.1x the resident-mode figure) |
| Equivalence | **mp4 MD5 match** with the normal int8 mode |
| Verdict | **opt-in** |
| Gotcha | **`use_stream=True` combined with `low_cpu_mem_usage=True` crashes on torchao `Int8Tensor`'s pin_memory** (`cannot pin 'torch.cuda.CharTensor'`). Worked around with `low_cpu_mem_usage=False` (also speeds up onload 4–5x). → A real bug worth reporting upstream to diffusers |
| Commit | `26bf434` |

**Update**: on 2026-11-15 diffusers' `enable_group_offload` was replaced by an in-house windowed prefetch streamer (`core/blockstream.py`, `H3_GROUP_STREAM=own` by default; the diffusers path remains as `diffusers`), and the token_refiner, projected TE and VAE decoder are streamed the same way (16GB configuration → 768² 4.6GB / 1024² 6.4GB). On 2026-11-17 combination with the turbo LoRA was enabled (applied at startup).

---

## B. Investigated but not adopted

| Item | Source | Conclusion |
|---|---|---|
| **VAE device/dtype cast fix** | [ComfyUI commit 16e3f30](https://github.com/Comfy-Org/ComfyUI/commit/16e3f3034f2bba1fff6c70cbd759339778555cd6) (@PhotogenicWeekE) | **Not needed.** This fixes a crash caused by ComfyUI's own weight management (per-layer casting at compute time) letting a raw `nn.Parameter` pass through unchanged. diffusers moves entire modules with `.to(device)`, and since VAE fp32 is fixed by the PR's explicit contract, the same kind of mismatch cannot occur (confirmed in the actual code) |
| **NVFP4 TE variant** | Comfy-Org (14.6GB, ComfyUI format) / RedHatAI (20.4GB, compressed-tensors) | **Rejected.** The ComfyUI format requires an in-house implementation of fp4 arithmetic; the RedHatAI version is nearly the same size as nf4's 21GB and does not solve the 24GB problem (its runtime also assumes vLLM) |
| **Hub-based attention backends** | diffusers `flash_hub` / `sage_hub` | **Not viable.** No torch 2.9-targeted build exists on the Hub side (only 2.10–2.12 are available). Not an environment issue |
| **ComfyUI's pruned TE file (47.97GB)** | Comfy-Org/MiniMax-H3 | Did not use the file itself; **independently derived the same reduction and implemented it in-house** (C-1 below). This avoids the ComfyUI format conversion and lets the existing bnb-nf4 path be used as-is |
| **torchao's C++ kernel** | torchao 0.18 | **Passed on.** Requires torch>=2.11, posing a large regression risk to the whole venv. Currently running on the 0.17 pure-Python fallback |

---

## C. Own implementations inspired by community findings

### C-1. Removing unused upper layers of text_encoder (`H3_TE_PRUNE`)

Investigated why the bf16 TE distributed by ComfyUI is 47.97GB (-14.2GB versus the full
62.13GB) and discovered that **H3 only reads `hidden_states[50]` of the TE** (the 14 layers
from layer 51 onward, ≈13GB, are dead weight), then implemented the equivalent in-house.

- Measured: TE-nf4 **21.02 → 17.45GB (-17%)**, bf16 66.71 → 53.06GB
- Equivalence: both t2va and ref2va produce an **mp4 MD5 match** against the unpruned version (also confirmed `torch.equal` against the 64-layer version)
- **Gotcha (most important): pruning to exactly 50 layers causes transformers'
  `tie_last_hidden_states` to silently overwrite `hidden_states[50]` with the value after
  the final norm is applied, producing numerically different results.
  Pruning to 51 layers is correct** (this is exactly why diffusers has a guard of
  `num_layers <= 50 → raise`)
- This completes support for the 24GB class (confirmed working even at a measured 20GB)
- Commit: `2d31424`

### C-2. Transformer int8 quantization + keeping both variants resident simultaneously

Applied the recipe documented in PR #14355 (`Int8WeightOnlyConfig`), and additionally kept
both the transformer and transformer_ref resident at the same time to eliminate the switching
cost between Ref2VA and T2VA.

- Measured: transformer 66.3 → **34.0GB**. ref2va 523s → **463–471s**; the 66GB-class reload on variant switching is eliminated
- Gotcha: `expandable_segments:True` is required (int8 load/free cycles fragment memory, and "only 54GB in use yet a 15GB allocation fails" was reproduced on real hardware)
- Commit: `236a424`, `435f831`

### C-3. Other in-house implementations

| Item | Details | Commit |
|---|---|---|
| TE bnb-4bit | TE 66.7GB → 21.0GB. 245s → 185s (eliminates TE⇔transformer swapping) | `6526b61` |
| 48GB-class phase rotation | Never keep TE and transformer resident simultaneously. t2va peak 38.9GB | `2bb3127` |
| Video VAE fp16 | Decode peak 16.3 → 11.4GB, PSNR 39.97dB. Gotcha: `_keep_in_fp32_modules` overrides the dtype setting | `a6c5ffa` |
| Local LLM prompt enhancement | Three modes: storyboard / brief / translate | `02e311f` |
| Ref2VA | Ordered references — 9 images/3 videos/3 audio (wired up the PR's feature; resolved 3 real-machine OOMs) | `ca5d912` |

---

## D. Current state (768², 5 seconds)

| Configuration | Request time |
|---|---|
| Initial (bf16 TE swapping) | 245s |
| Current default (bnb-4bit + FBC 0.05 + Sage) | **~160s** |
| + FBC 0.1 (opt-in) | ~125s |
| + Turbo LoRA 8 steps (opt-in) | **~88s** |
| + Turbo 4 steps (for draft use) | ~40s |

VRAM floor: 96GB → **~18GB** (`H3_LOWVRAM=group H3_TE_PRUNE=1`) → since 2026-11-15 **4.6GB (768²) / 6.4GB (1024²)** (projected TE + streaming, 30 steps), 3.1-3.7GB at 4 turbo steps.
Every step has been confirmed equivalent via either an MD5 match at the same seed or a quality A/B.


---

## FILE: docs/COMMUNITY_IMPROVEMENTS.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/docs/COMMUNITY_IMPROVEMENTS.md)

# コミュニティ改良の取り込み作業一覧(MiniMax H3)

**日本語** | [English](COMMUNITY_IMPROVEMENTS.en.md)

オリジナル(MiniMaxAI/MiniMax-H3 + diffusers PR #14355)を起点に、ComfyUI コミュニティ
などで出た改良を本アプリ(diffusers 経路)へ取り込んだ作業の記録。2026-08-04〜08-06。

判定はすべて **同一 seed の実機 A/B** に基づく。「品質同等」は目視 + PSNR + 音声相関で
確認したもの、「等価」は mp4 の MD5/バイト一致まで確認したものを指す。

---

## A. 取り込んだもの(コミュニティ発)

### A-1. FirstBlockCache(ComfyUI の EasyCache 相当) — 既定ON

| | |
|---|---|
| 出典 | ComfyUI コミュニティ(kijai 氏の EasyCache 活用法。@gosrum / @umiyuki_ai のポスト) |
| コミュニティ報告 | EasyCache で 1.67倍、Sage 併用で 6分→2分半 |
| 取り込み方 | diffusers 公式の `FirstBlockCache`(同系統のステップ間キャッシュ)を採用。`H3_CACHE`/`H3_CACHE_THRESHOLD` |
| 実測 | デノイズ 157s → **118s(-25%)**、30step中7スキップ。threshold 0.1 なら 81.5s(1.92倍)だが構図ドリフト |
| 品質 | PSNR 31.8〜34.3dB、音声相関 0.979。目視で区別困難 |
| 判定 | **threshold 0.05 を既定化**。0.1 は opt-in |
| 罠 | H3 ブロックは PR ブランチの `TransformerBlockRegistry` 未登録 → 自前登録が必要。リクエスト毎に `_reset_stateful_cache()` + `cache_context()` 必須(同一seed連続2本のバイト一致で検証) |
| コミット | `14afdfc` |

### A-2. Sage Attention — 既定ON

| | |
|---|---|
| 出典 | 同上のポスト(「Sage のみでも25%短縮」) |
| 取り込み方 | sm_120 向けに **thu-ml/SageAttention をソースビルド**(Linux 向け事前ビルド wheel は存在せず、公開されているのは Windows 版のみ)。`H3_ATTN_BACKEND` |
| 実測 | デノイズ 118s → **104s(-12%)**。コミュニティ報告の -25% には届かず |
| 品質 | 完全決定論(同一seed 2本バイト一致)。目視同等(PSNR 21dB は int8-QK 近似による軌道ドリフトで劣化ではない) |
| 判定 | **既定化**(`H3_ATTN_BACKEND=default` で従来 SDPA に戻せる) |
| 罠 | ビルドは `MAX_JOBS=4 NVCC_THREADS=2` + systemd-run メモリ上限必須(無制限並列 nvcc はホストRAM枯渇でシステム巻き添え事故歴)。`CUDA_HOME=/usr/local/cuda-12.8` の明示が必要(既定の cuda-13.0 は torch cu128 と不一致でビルド失敗) |
| コミット | `9c7e6a6`、`scripts/build_sageattention.sh` |

### A-3. Latent アップスケーラ(2段生成 hires-fix) — opt-in

| | |
|---|---|
| 出典 | [Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler)(@umiyuki_ai のポスト経由) |
| 仕組み | 低解像度で前半デノイズ → 映像 latent のみ空間2x補間 → 再ノイズ → 高解像度で仕上げ。学習済みアップスケーラは使わない |
| 取り込み方 | `/api/t2va`・`/api/i2v` の `upscale=1`、`H3_HIRES_DENOISE` |
| 実測 | 768²→**1536²** が 645s / peak 88.0GB(upscale=0 は 181s / 92.1GB) |
| 品質 | 構図一致のまま毛並み等の実ディテールが乗る。背景細部は再デノイズで軽微にドリフト(hires-fix の性質) |
| 判定 | **opt-in**(既定OFF) |
| 罠 | **補間対象はノイズ付き latent ではなく x0 推定値**。ノイズ付きを補間すると市松ノイズが増幅され全面ノイズ化する(実機で再現 → 参考実装も `denoised_output` を使っていることを確認して修正)。解像度変更時は `build_packed_sequence()` と `row_timestep_plan` の再構築が必要 |
| コミット | `e9a45a7` |

### A-4. Turbo LoRA(4/8ステップ蒸留) — opt-in

| | |
|---|---|
| 出典 | [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)(Ostris 氏学習中、Apache 2.0)。@PhotogenicWeekE のポストで実用情報を取得 |
| コミュニティ報告 | 8 steps で 1280×736/10秒が 272s→160s。**「4〜7 steps はダメ」** |
| 取り込み方 | `H3_TURBO_LORA=1`(既定8steps)。ComfyUI リパック版ではなくオリジナルを使用 |
| 実測 | 8steps **87.7s(-46%)** / 16steps 98.4s / 4steps 39.6s(基準30stepsは163.5s) |
| 品質 | 8steps は基準に迫る。16steps は基準同等。4steps は柔らかめ・音声弱め |
| 判定 | **opt-in**(LoRA が「デモ/プレビュー・学習途上」と作者明記のため既定OFF。完成版が出たら差し替え + A/B のみで既定化判断できる状態にしてある) |
| 新事実 | **「4〜7 steps はダメ」は ComfyUI 標準サンプラー起因の可能性が高い**。本実装(video shift 12 / audio shift 3 のデュアルスケジュールを正しく積分)では 4steps でも音声破損は起きなかった。なお シフト配線は改修不要で、PR 実装の既定値が既に正しいことを sigma 格子のビット一致で確認済み |
| 罠 | LoRA キーは ComfyUI 命名の fused-QKV 形式 → `fuse_projections()` + ランタイムデルタで適用。**`fuse_projections()` は旧 to_q/k/v を削除せず +12.8GB リークする**(実 OOM で発見、明示 delete で対処) |
| コミット | `2ab3100` |

### A-5. block-level group offload(24〜32GB級対応) — opt-in

| | |
|---|---|
| 出典 | ComfyUI の「INT8 + layerwise offload で 24GB 動作」報告 + PR #14355 の streamed offload 対応 + 姉妹プロジェクト(diffusers-server)で確立した知見 |
| 取り込み方 | `H3_LOWVRAM=group`。int8 transformer を CPU ロード → `enable_group_offload(block_level, num_blocks_per_group=1, use_stream=True)` |
| 実測 | 32GB 制限で **peak 28.7GB**、デノイズ 220.8s(常駐比 ~2.1倍) |
| 等価性 | 通常 int8 モードと **mp4 MD5 一致** |
| 判定 | **opt-in** |
| 罠 | **`use_stream=True` + `low_cpu_mem_usage=True` の併用は torchao `Int8Tensor` の pin_memory でクラッシュ**(`cannot pin 'torch.cuda.CharTensor'`)。`low_cpu_mem_usage=False` で回避(onload も4〜5倍高速化)。→ diffusers 本体に報告する価値のある実バグ |
| コミット | `26bf434` |

**更新**: 2026-11-15 に diffusers の `enable_group_offload` から自作の窓付き先読みストリーマ(`core/blockstream.py`、`H3_GROUP_STREAM=own` 既定。diffusers 経路は `diffusers` で残置)へ置換し、token_refiner・投影 TE・VAE デコーダも同方式で流すようにした(16GB 構成 → 768² 4.6GB / 1024² 6.4GB)。2026-11-17 に turbo LoRA との併用を解禁(起動時に巻く)。

---

## B. 調査した結果、取り込まなかったもの

| 項目 | 出典 | 結論 |
|---|---|---|
| **VAE の device/dtype キャスト修正** | [ComfyUI commit 16e3f30](https://github.com/Comfy-Org/ComfyUI/commit/16e3f3034f2bba1fff6c70cbd759339778555cd6)(@PhotogenicWeekE) | **不要**。ComfyUI 独自の重み管理(レイヤー単位 compute 時キャスト)で生 `nn.Parameter` が素通りするのが原因のクラッシュ修正。diffusers はモジュール丸ごと `.to(device)` で移動し、VAE fp32 固定が PR の明文契約のため同種の不一致が起きない(実コードで確認) |
| **NVFP4 版 TE** | Comfy-Org(14.6GB、ComfyUI形式)/ RedHatAI(20.4GB、compressed-tensors) | **不採用**。ComfyUI 形式は fp4 演算の自前実装が必要、RedHatAI 版は nf4 の 21GB とほぼ同サイズで 24GB 問題を解決しない(実行系も vLLM 前提) |
| **hub 系 attention backend** | diffusers `flash_hub` / `sage_hub` | **不成立**。Hub 側に torch 2.9 向けビルドが存在しない(利用可能は 2.10〜2.12)。環境の問題ではない |
| **ComfyUI の pruned TE ファイル(47.97GB)** | Comfy-Org/MiniMax-H3 | ファイル自体は使わず、**同じ削減を自前で導出して実装**(下記 C-1)。ComfyUI 形式の変換を避け、既存の bnb-nf4 経路をそのまま使えるため |
| **torchao の C++ カーネル** | torchao 0.18 | **見送り**。torch>=2.11 要求で venv 全体のリグレッションリスクが大きい。0.17 の pure-Python フォールバックで運用中 |

---

## C. コミュニティの知見から着想を得て、自前で実装したもの

### C-1. text_encoder 未使用上位レイヤー削除(`H3_TE_PRUNE`)

ComfyUI 配布の bf16 TE が 47.97GB(フル 62.13GB 比 -14.2GB)である理由を調べ、
**H3 は TE の `hidden_states[50]` しか読まない**(51層目以降の14層 ≈ 13GB は死荷重)
ことを突き止めて自前で実装した。

- 実測: TE-nf4 **21.02 → 17.45GB(-17%)**、bf16 66.71 → 53.06GB
- 等価性: t2va・ref2va とも削除なし版と **mp4 MD5 一致**(64層版との `torch.equal` も確認)
- **罠(最重要): 50層ちょうどに削ると transformers の `tie_last_hidden_states` が
  `hidden_states[50]` を最終 norm 適用後の値で静かに上書きし、数値が別物になる。
  51層に削るのが正解**(diffusers 側に `num_layers <= 50 → raise` のガードがあるのは
  まさにこの境界のため)
- これにより 24GB級(実測20GBでも動作)対応が完成した
- コミット: `2d31424`

### C-2. transformer int8 量子化 + 両変種同時常駐

PR #14355 のドキュメント記載レシピ(`Int8WeightOnlyConfig`)を適用し、さらに
transformer と transformer_ref を同時常駐させて Ref2VA ⇔ T2VA の切替コストを消した。

- 実測: transformer 66.3 → **34.0GB**。ref2va 523s → **463-471s**、変種切替の 66GB 級
  再ロードが消滅
- 罠: `expandable_segments:True` が必須(int8 のロード/解放サイクルで断片化し、
  「54GB しか使っていないのに 15GB 確保失敗」が実機再現)
- コミット: `236a424`、`435f831`

### C-3. その他の自前実装

| 項目 | 内容 | コミット |
|---|---|---|
| TE bnb-4bit | TE 66.7GB → 21.0GB。245s → 185s(TE⇔transformer 入れ替えの消滅) | `6526b61` |
| 48GB級フェーズ循環 | TE と transformer を同時常駐させない。t2va peak 38.9GB | `2bb3127` |
| video VAE fp16 | デコードピーク 16.3 → 11.4GB、PSNR 39.97dB。罠: `_keep_in_fp32_modules` が dtype 指定を無効化 | `a6c5ffa` |
| ローカル LLM プロンプト強化 | storyboard / brief / translate の3モード | `02e311f` |
| Ref2VA | 画像9/動画3/音声3 の順序付き参照(PR の機能を配線。OOM 3件を実機で解決) | `ca5d912` |

---

## D. 到達点(768²・5秒)

| 構成 | リクエスト |
|---|---|
| 初期(bf16 TE 入れ替え) | 245s |
| 現既定(bnb-4bit + FBC 0.05 + Sage) | **~160s** |
| + FBC 0.1(opt-in) | ~125s |
| + Turbo LoRA 8steps(opt-in) | **~88s** |
| + Turbo 4steps(ドラフト用途) | ~40s |

VRAM 下限: 96GB → **~18GB**(`H3_LOWVRAM=group H3_TE_PRUNE=1`)→ 2026-11-15 以降 **4.6GB(768²)/ 6.4GB(1024²)**(投影 TE + ストリーミング、30 ステップ)、turbo 4 ステップなら 3.1〜3.7GB。
各段はすべて同一 seed の MD5 一致または品質 A/B で等価性を確認済み。


---

## FILE: docs/FEATURES.en.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/docs/FEATURES.en.md)

# Features and settings in detail (minimax-h3)

[日本語](FEATURES.md) | **English**

Detailed descriptions and measurements for each feature and environment variable, split out of the README. Each section is the original README text moved verbatim; dates inside a section mark when that feature was implemented and measured. See [README.en.md](../README.en.md) for how to use the app and [HISTORY.en.md](HISTORY.en.md) for the chronological work log.

- [Performance overview](#performance-overview)
- [VRAM configurations and residency](#vram-configurations-and-residency)
- [text_encoder](#text_encoder)
- [transformer and denoising](#transformer-and-denoising)
- [Generation modes](#generation-modes)
- [UI](#ui)
- [LLM prompt enhancement](#llm-prompt-enhancement)

## Performance overview

### What the speedups bought (measured summary)

**This repository's central finding: the bottleneck was not denoising but the fixed cost of
loading and freeing models.** The numbers below are collected from the sections further
down; nothing here was re-measured for this table.

#### Cumulative: from the stock configuration to today

48GB box (PRO 5000 48GB + RTX 4000 20GB), the `H3_LOWVRAM=1` family, 768², **steady-state**
(second and later requests, excluding the initial model load; t2va is 5s = 124 frames).

| Stage | t2i (1 image) | t2va (5s) |
|---|---|---|
| Stock configuration (30 steps, no turbo) | 157s | 351.4s |
| + lightx2v turbo LoRA (4 steps) | 157s | 143s |
| + `H3_TE_PREQUANT` (on-disk cache of the quantized TE) | 83.2s | — |
| + `H3_TE_DEVICE` (TE parked on a second GPU) | about 35s | 60.5s |
| + `H3_KEEP_TRANSFORMER` (transformer stays resident) | **9.7s** | **44.2s** |
| | **16x** | **8.0x** |

Turbo — the one change that actually speeds up denoising — **tops out at 2.6x**. The
remaining 3x-plus all comes from eliminating fixed costs (even at 30 steps without turbo,
removing the fixed costs alone takes t2i from 157s to 51.1s, a 3x gain).

#### Where each mode landed (all measured in one configuration, 2026-08-12)

Configuration: **int8 transformer resident + projected TE (NF4) + no decode-window free +
turbo at 4 steps + sage + fp16 decode**, on **a single GPU** (measured on the 96GB box:
`H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1 H3_TE_PROJ=…
H3_TURBO_LORA=1`).

| Mode | Single (fastest) | **Per item in a run** | Denoise | Peak | What the remaining time goes to |
|---|---|---|---|---|---|
| **t2i** 768² | **7.40s** | 7.9s (**0.94x — not worth using**) | 2.40s | 45.0GB | VAE CPU↔GPU round trips, ~3.3s |
| **t2va** 5s 768² | **28.13s** | (no batch API) | 14.94s | 45.6GB | decode, 7.05s |
| **ref2i** 768² (reference still) | 79.3s | **47.0s (1.69x)** | 7.8s | 45.4GB | reference vision encode, **~47s** |
| **i2va** 5s 768² (image reference to video) | 103.1s | **75.0s (1.37x)** | 22.0s | 45.9GB | the same ~47s, plus ~13s to reload the t2va transformer |

**How to read these numbers** (measurement conditions):

- **All steady-state; the initial load at server startup is excluded** (second and later
  requests). Startup spends **about 50s** making the transformer and VAE resident, but that is
  paid once per process.
- **The reference modes differ on their first request only**: `transformer_ref` is loaded on the
  first reference request rather than at startup, costing **+55s once** (measured for ref2i:
  134.7s first, 79.3s steady). It stays resident afterwards.
- Resolution pinned to **768×768**. Video is 5s = **124 frames** (24fps); stills are **22 frames**.
- **4 steps with the turbo LoRA** (`H3_TURBO_LORA=1`). FirstBlockCache is auto-disabled by turbo;
  attention is sage. Seed and prompt are fixed per mode.
- "Per item in a run" is the cost per item when several are generated together with the same
  reference and settings (`/api/t2i_batch`, `/api/ref2i_batch`, `/api/ref2va_batch`), measured
  with 3 scenes for t2i/ref2i and 2 for i2va. **The batch path is `H3_LOWVRAM=1`-only**, so that
  column runs a different configuration from the single column — the table pairs "the fastest way
  to make one" with "the fastest way to make N".

- **Every mode fits in 45-46GB**, so a single 48GB card runs the full feature set at
  near-peak speed. But **mixing the t2va family and the reference family in one process keeps
  both transformers resident, at 74.3GB** (measured; the peak on returning to t2va is 77.3GB).
  To run on one 48GB card, **keep the modes in separate processes**.
- **The reference modes are bound by the reference vision encode (~47s), not by denoising** —
  which is why turbo buys only 2.8x for i2va against 5.5x for t2va.
- Two GPUs with a bf16 transformer reach **t2i 6.89s / t2va 26.8s**, the fastest measured, but
  need a 77GB-class card for about 7%. See the dated 2026-08-12 sections.

#### What the batch measurements established

- **Batching t2i is no longer worth it** (0.94x, marginally slower). Batching existed to
  amortize model-load fixed costs, and **residency removed the cost there was to amortize**
  (before residency it was 2.3x: 157s to 67.5s).
- **Only the reference modes still benefit**, because what gets shared is **not a load but the
  ~47s reference vision encode**. **The batch's per-step time matches a single request exactly**
  (ref2i 2.598s vs 2.599s; i2va 7.321s vs 7.323s), so the batch path adds no overhead of its
  own — the entire difference is how many times that encode is paid.
- **More scenes, more benefit**: the saving is `47x(scenes-1)/scenes`. For i2va that is 1.37x at
  2 scenes, 1.44x at 3, 1.57x at 5 — so it favours long stories built around one character.
- **The "sharing shrinks it" model holds for stills and video alike** (measured vs predicted:
  ref2i at 3 scenes, 32.3 vs 31.3s/image; i2va at 2 scenes, 28.1 vs 23.5s/video). → **Caching
  the reference encode across requests should deliver the same saving to repeated single
  requests** (not implemented; see the dated 2026-08-12 section).
- **Two gates**: the batch path is `H3_LOWVRAM=1`-only and **silently falls back to sequential**
  otherwise; and reference batches reject `H3_TE_DEVICE` via a "TE GPU needs 24GB or more" guard
  — **a threshold sized for the 32B TE's vision activations**, far too large for the 3.11GB
  projected TE (it rejects a 16GB 4060 Ti). Both **worth revisiting**.

> **A measurement trap (walked into on 2026-08-12)**: comparing a batch against singles without
> passing `height`/`width` on the batch side means **the batch generates on the server's default
> 16:9 canvas (1344×768)** — 1.75x the pixels — which manufactures a nonexistent "the batch's
> steps are 1.75x slower" effect. It was caught because the pixel ratio 1344×768÷768² = 1.75
> matched the measured step-time ratio of 1.753 exactly. **Always pin the resolution when
> comparing modes or code paths.**

### An honest note on speed

**The low-VRAM configs run, but they are slow.** `H3_LOWVRAM=group` streams the int8 weights
(~34GB) from CPU to GPU every step, so transfer is the bottleneck. The 16GB / 8GB figures in
the table above are on this box's **PCIe Gen3 x4 slot**, where most of the `16.5s/step` (t2i)
to `51s/step` (t2va) is transfer time. On a **proper Gen4 x16 slot the transfer is about 1/8**,
so these are values of "this box's slot," not "the 16GB card's performance." Also, on the
int8+SDPA trajectory FirstBlockCache does not kick in (`cache_skipped_steps: 0`), leaving up
to a 2x speedup on the table via threshold tuning (unverified).

On quality: both the projected TE (PSNR 22.4dB vs 32B) and int8+SDPA shift the composition
even at the same seed (**a trajectory divergence, not degradation**). In fact, in some
measured cases int8+SDPA improved prompt fidelity, and the reference (vision) path was
visually confirmed to match the 32B ground truth in fidelity. **Cross-config quality cannot
be judged by PSNR/MD5 — judge it by eye.**

> **Addendum (2026-11-17)**: the text above reflects August 2026; three things have changed.
> - "The support table above" now refers to the next section's "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)" after the split.
> - **The turbo LoRA (4 steps) now works with `H3_LOWVRAM=group`** (in-house streamer + `H3_TURBO_LORA=1` at startup). Because the number of weight transfers scales with the step count, the transfer-bound "runs but slowly" caveat is largely gone. Measured on an A4000 16GB (Ampere sm_86, PCIe Gen4 x4, SDPA): t2va 5 s at 768² went from 790 s at 30 steps (peak 8.0GB) to 151 s at 4 steps (peak 7.0GB). With the 8GB-tier flags (`H3_TE_STREAM=1 H3_TE_DIET=1`, VAE decoder streamed, emulated with `H3_VRAM_LIMIT_GB=8`): 768² 209 s / 3.08GB, 1024² 376 s / 3.73GB, fl2va 225 s / 3.37GB, ref2va 299 s / 3.02GB (see the README's support table).
> - "FirstBlockCache does not fire on the int8+SDPA trajectory" no longer holds: on the same A4000 (int8 + SDPA, 30 steps) t2i skipped 7 steps and t2va 6 (`cache_skipped_steps`).
> The quality paragraph (crossing configurations changes the composition even at the same seed; that is a trajectory divergence, not degradation, and must be judged visually) is unchanged.

## VRAM configurations and residency

### VRAM × feature matrix (measured 2026-08-11)

○ = completes (measured) / △ = derived estimate (not yet measured) / × = OOM. **Times are
steady-state** (second and later requests, excluding the initial model load); **peaks are
torch's allocated peak**. The 16GB and 8GiB rows run 30 steps, the 48GB+20GB row turbo at 4
steps. All speeds are on this box's **PCIe Gen3 x4 slot + sm_89 (4060 Ti) + SDPA**; on a
**Gen4 x16 slot the weight transfer is roughly 1/8** (see "An honest note on speed" below).

| Config | t2i 768² | t2va 5s 768² | ref2i | i2va (image ref) | audio ref | 768×1344 5s |
|---|---|---|---|---|---|---|
| **Real RTX 4060 Ti 16GB, single** | ○ 498s / peak 7.4GB | ○ 25 min / 11.4GB (2026-11-15: 4.62GB with `H3_TE_STREAM=1 H3_TE_DIET=1`, 6.41GB at 1024²; 4.79GB with head-grouped attention, 3.03GB with `H3_CACHE=none`) | ○ | ○ 39 min / 9.41GB | ○ 54 min / 11.96GB | ○ 66 min / 13.37GB (15.2GB real = ceiling) |
| **8GiB × 2** (compute + TE, ballast-simulated) | ○ 512s / 6.4GB | ○ 25.6 min / 7.23GB | ○ 17.7 min / 6.69GB | × OOM | × OOM | × OOM |
| **12GB, single** | △ estimate | △ | △ | △ | close to × (14.7GB real) | × |
| **48GB + 20GB** (two cards, turbo LoRA) | ○ 9.7s | ○ 44.2s | ○ | ○ | ○ | ○ |

- 12GB single is not measured. t2va real peak 11.4GB / i2va 9.41GB (11.2GB real) fit in 12GiB
  by calculation, but audio ref (14.7GB real) and 768×1344 (15.2GB real) exceed it.
- The key to the 16GB / 8GB configs is the **projected TE** (Qwen3-VL-4B + a trained linear
  map, ClipProj, 3.11GB in NF4 — a stand-in for the 32B text_encoder) + `H3_LOWVRAM=group`
  (streams the int8 transformer block by block, ~1.4GB GPU-resident) + fp16 video-VAE decode.
  Details are in the dated sections 2026-08-10 to 08-11.
- **The measurement environment is shared across two machines**: the 16GB / 8GB rows are on
  the **96GB box (RTX PRO 6000 + an added 4060 Ti 16GB)**, the 48GB+20GB row on the **48GB box
  (RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB)**. Older measurements (from the 96GB-single era)
  are kept as-is in the individual dated sections.

### VRAM support table and main environment variables (quick reference)

Launch examples per use case. The top five rows are ballast measurements on the 96GB box
(PRO 6000, 32B TE); the bottom three rows are **real-card measurements using the projected
TE** (2026-08-11; see the "[VRAM × feature matrix](#vram--feature-matrix-measured-2026-08-11)"
at the top and the dated 2026-08-11 sections for details).

| GPU | Startup flag | t2va measured (768², 5s) |
|---|---|---|
| 96GB | (none = default) | peak 92GB / about 160s |
| 80GB tier | `H3_TRANSFORMER_QUANT=int8` | peak 59.7GB / about 160s |
| 48GB tier | `H3_LOWVRAM=1` | peak 38.9GB / about 215s |
| 32GB tier | `H3_LOWVRAM=group` | peak 28.7GB / about 280s (denoise about 1.7x faster since the 2026-11-15 streamer, see `H3_GROUP_STREAM` below) |
| 18GB tier | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak 17.7GB / about 280-320s |
| **Single 16GB** (real 4060 Ti) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | peak 11.4GB / about 25min* |
| **Single 12GB** (projected, unmeasured) | same as single 16GB | the measured 11.4GB peak should fit in 12GiB; reference modes and 768×1344 will not |
| **8GB×2** (ballast-measured) | single-16GB flags + `H3_TE_DEVICE=cuda:1` | peak 7.23GB / about 25.6min* |

\* Times in the low-VRAM real-card rows were measured on a PCIe Gen3 x4 slot with an sm_89
card (SDPA). On Gen4 x16 the weight transfer is about 1/8 (see
"[An honest note on speed](#an-honest-note-on-speed)"). Cards other than sm_120 also need
`H3_ATTN_BACKEND=default`.

```bash
# Example: launch on a 32GB-tier GPU
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
# Example: launch on a single 16GB card (add H3_ATTN_BACKEND=default on sm_89)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

Main environment variables (defaults shown; most **can also be toggled from the UI**, so
only set these if you want to change them permanently):

| Variable | Default | Meaning |
|---|---|---|
| `H3_PROFILE` | `auto` | Added 2026-11-17. When none of the main profile variables is set, fills in the 96/80/48/32/16/8GB-tier defaults from total VRAM (`core/profile.py`). On the 32GB-and-below tiers it also enables the turbo LoRA (ref2v / fl2v 4-step files), since in group mode it must be applied at startup; also with `file` streaming (the LoRA, about 1.2GB, is streamed from host RAM per unit; 2026-11-18). `96gb`..`8gb` forces a tier, `off` disables. Also sets `H3_ATTN_BACKEND=default` when sage cannot be used |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder quantization (`none` = bf16, 66.7GB) |
| `H3_TE_PRUNE` | `0` | Remove unused upper layers of TE (output unchanged, -3.6GB) |
| `H3_TE_PROJ` | (off) | **Projected TE**: replaces the 32B TE with Qwen3-VL-4B + a linear map (repo id or .safetensors path; quality is an approximation, see the dated 2026-08-10 section) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | Quantization of the projected 4B TE (NF4 = 3.11GB; `none` = bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (off) | Park the TE on a second GPU (e.g. `cuda:1`; the 32B TE needs a 20GB-class card, the projected TE fits an 8GB-class one) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` shrinks the transformer from 66.3GB to 34GB |
| `H3_LOWVRAM` | `0` | `1` = 48GB-tier phase cycling / `group` = block offload for the 32GB tier and below |
| `H3_KEEP_TRANSFORMER` | `0` | Keep the transformer resident, removing the reload fixed cost (see its section for the preconditions) |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache (denoise -25%; measured ineffective on the int8+SDPA trajectory) |
| `H3_ATTN_BACKEND` | `sage` | The sage build is sm_120-only. **Any other GPU must use `default` (SDPA)** |
| `H3_TURBO_LORA` | `0` | 4/8-step distilled LoRA (downloads about 780MB on first use). Since 2026-11-17 also works in `group` (`H3_GROUP_STREAM=own` + the flag at startup; with `file` streaming the LoRA, about 1.2GB, first stayed on the GPU: 8GB-tier flags under an 8GB cap gave 768² 205 s / 4.19GB, 1024² 376 s / 4.85GB, ref2va 768² 276 s / 4.13GB. At 15:00 the same day it was switched to per-unit streaming from host RAM (`H3_FILE_STREAM_HOST_MIN_MB`); under a 6GB cap 768² takes 211 s at 2.82GB and 1024² 380 s at 3.48GB. Verified to work on real Windows 11 + RTX 4060 Laptop 8GB / 16GB RAM and Windows 10 + RTX 3050 6GB machines (the same entry PC with an RTX 3060 12GB + NVMe: 512² 132 s (denoise 83 + decode 36), 768x512 198 s, 768² 338 s, 1024x768 473 s. 3050 / 32 GB RAM, SATA-class SSD about 430 MB/s, 2026-11-19: 512² 321 s = denoise 237 + decode 65, 512x384 294 s; on the SATA-class SSD the denoise is disk-read bound. measured after moving to an NVMe: 512x384 163 s (denoise 108 + decode 42), 512² 237 s (147 + 65), 768x512 345 s (248 + 85), 768² 558 s (444 + 112); shared GPU memory stays at 1.5GB and the denoise runs the GPU at 100%, so from here the card itself is the limit. 512² to 768x512 is the practical range. 4060 Laptop measured 2026-11-19 on an NVMe: 512² 194 s = denoise 117 + decode 16, 768x512 279 s = 191 + 65, 768² 412 s = 313 + 85; the denoise is bound by weight reads and the GPU, the decode by GPU compute). A4000 16GB: t2va 5 s 768² in 151 s at 4 steps; with the 8GB-tier flags under an 8GB cap, 768² 209 s / 3.08GB, 1024² 376 s / 3.73GB ref2va 768² 299 s / 3.0GB and fl2va 768² 225 s / 3.4GB. Note: ref2va in group mode with the in-house streamer crashed on a layout device mismatch, fixed the same day) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | When the bf16 transformer / transformer_ref weights (~66GB each) are not cached, download them from HF only if `1` (default: no download, fail with a hint) |
| `H3_PIKA_RENOISE` | `0` | Renoise sampler for the Pika H3 Flex turbo LoRA (experimental) |
| `H3_BLOCK_COMPILE` | `0` | `1` torch.compiles the 50 transformer_ref blocks after the turbo LoRA is applied (attention itself is a graph break). About -13% per step at 1024×768 on an RTX PRO 6000; +5-14s on the first call only |
| `H3_LORA_FUSE` | `0` | Cuts element-wise work in the turbo LoRA wrapper: `1` skips the multiply when scale is 1.0 (bit-identical) / `2` also fuses the add into the GEMM. No extra gain on top of `H3_BLOCK_COMPILE` |
| `H3_VIDEO_VAE_FP16` | `0` | fp16-ify the video VAE (reduces decode peak; required at 16GB and below) |
| `H3_LLM_URL` | `http://127.0.0.1:64660` | Local LLM used for prompt enhancement (optional feature) |
| `H3_LLM_API_KEY` | none | API key for the LLM server (sent as `Authorization: Bearer`). If unset, read from the same-named line in the git-ignored `config.local.env` at the repo root |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / none | Added 2026-11-17. Falls back to a built-in CPU LLM (Gemma 4 E4B, GGUF Q4_0) when the external LLM is unreachable; it is llama.cpp's prebuilt `llama-server` run as a child process (`core/local_llm.py` fetches and starts it, no compiler needed). `1` = built-in only, `0` = off. `H3_LLM_LOCAL_IDLE_S` (default 300) stops it when idle, `H3_LLM_LOCAL_STOP_ON_GENERATE` (auto = hosts under 32GB RAM) stops it when a generation starts |

### Oct 2026 update: realtime conversational generation on a single 24GB/32GB GPU

Since the initial August release, major speed / low-VRAM work for conversational use
(continuous ref2va) was done in the production repository (diffusers-movie-server) and
re-synced here. Where it landed:

| Setup | Condition | Result |
|---|---|---|
| **Single 32GB (RTX 5090 class)** | 320x448, 141f (5.875s of video) | **5.57s = 0.95x realtime**, peak 25.6GB |
| Single 24GB (RTX 4090 class) | 320x448, 141f | 7.1s = 1.21x (usable for short replies), peak 20.4GB |

Key additions:
- **ck-w4a8 quantization** (`H3_PRUNED_QUANT=ck-w4a8` / `H3_TRANSFORMER_QUANT=ck-w4a8`):
  ComfyUI's W4A8 int8 kernels (`pip install comfy-kitchen==0.2.37`). Combined with the
  AdaLN-pruned checkpoints (multimodalart), the 33B transformer fits in **12.5-14.9GB
  resident with -28% denoise time and visually equal quality**
- **Pinned parking** (`H3_VAE_PINNED` / `H3_BASE_PINNED` / `H3_REF_PINNED`): model
  eviction becomes a pinned-memory pointer swap (park 0.3s / restore 2s)
- **TE reduction** (`H3_TE_DIET` / `H3_TE_STREAM`), prequantized weight caches,
  reference latent / prefix caches
- lightx2v 4-step turbo distillation (`H3_TURBO_LORA=1`)

Env set for single-GPU operation (32GB class; for 24GB change to `H3_KEEP_REF2VA_VAE=0`):
see the Japanese section above for the full command line.

**Caveats**: the GPU must be dedicated (TTS/LLM cannot co-reside); first-run cache
creation needs ~38GB of downloads and **45GB of free host RAM** for the ref side, plus
converting Kijai's `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (12.5GB) with
`scripts/convert_kijai_w4a8.py` for the base side; pinned host RAM totals ~45GiB
(on 64GB boxes drop `H3_REF_PINNED` or `H3_TE_STREAM`); on 24GB do NOT set
`H3_KEEP_REF2VA_VAE=1` (OOMs on the speech/idle prompt switch).

### VRAM/RAM design (important, based on measurements)

This machine has 96GB VRAM against 94GB RAM. **text_encoder is distributed natively in
bf16 and measures 66.73GB** (the initial estimate of "half, 33GB, after bf16-ification"
assuming an fp32 distribution was wrong). Together with transformer bf16 66.3GB and
vae+audio_vae fp32 totaling 11GB, that comes to about 144GB, which **does not fit in VRAM
or RAM simultaneously**. diffusers'
`ComponentsManager.enable_auto_cpu_offload()` (keeps all components resident in RAM at
all times and pushes only the one active component to the GPU) is not viable with 94GB
RAM, so it was not adopted.

The TE loading method is selected via the `H3_TE_QUANT` environment variable (default
`bnb-4bit`, made default after A/B testing on 2026-08-04).

#### `H3_TE_QUANT=bnb-4bit` (default)

Quantizes text_encoder to NF4 (bitsandbytes, compute_dtype=bf16) at startup time and
**keeps it resident on the GPU** (bnb 4-bit models cannot move between devices, so
residency is the only option). Measured size **21.0GB** (larger than the initial estimate
of ~17-18GB). transformer (66.3GB) also stays resident, eliminating the per-request
TE<->transformer swap.

- Steady-state residency: transformer + TE-nf4 = **~87.5GB**. Adding the 11GB VAE would
  bring this to ~98.5GB, exceeding 96GB, so **the VAE pair is not resident in this mode**
  (it stays on the CPU and round-trips to the GPU only during the keyframe
  encode/decode phase that needs it. Only the fp32 11GB PCIe round trip, no disk I/O).
- Only the decode window (~9s) is executed after freeing the transformer, then reloading
  it immediately afterward (because transformer+TE+VAE+decode buffers physically don't
  fit — confirmed OOM on real hardware). This is a single one-way trip x2 rather than a
  per-step swap, so it does not fall under prohibited pattern #33 of the
  diffusers-server CLAUDE.md.
- **Quality A/B (identical seed 12345)**: frame comparison shows equivalent composition,
  subject, and sharpness (only minor differences such as tree placement due to changes in
  conditioning values). Audio also has comparable levels (rms 0.0080->0.0061), with no
  -20dB-style collapse. **Judged as no degradation, so made the default.**

#### `H3_TE_QUANT=none` (bf16 TE, legacy method)

**Swaps two 66GB models on the GPU per request**:

- `vae` + `audio_vae` (fp32, 11GB total) stays GPU-resident at all times.
- Encode phase: [VAE 11GB + text_encoder 66GB] (if transformer is resident, it is freed
  first)
- Denoise/decode phase: [VAE 11GB + transformer 66GB] (TE is freed immediately after
  encoding)
- Freeing is done by **directly dropping the CUDA model reference** (not evacuating to
  RAM via `.to("cpu")` — routing 66GB through RAM was measured to trigger swap in
  practice). Reload takes 11-40s/model from page cache/disk. Steady state between
  requests: transformer+VAE resident (77.5GB).

**Measured overhead**: per request, TE load ~37s + transformer reload ~26s.
It was this swap cost that `bnb-4bit` (default, above) eliminated, shortening the total
per-request time from 245s to **185s** (the 157s denoise is unchanged since it is
model-bound and common to both).

**Two implementation pitfalls (hit and fixed on real hardware)**:
1. `MiniMaxH3TextEncoderStep.encode_prompt` is a bare staticmethod, and `@torch.no_grad()`
   is only attached to the block's `__call__` side. When calling it directly, you must wrap
   it in `torch.no_grad()`. Forgetting this pins about 50GB of TE weights on the GPU via the
   autograd graph, and VRAM never comes back even after freeing the model (same pattern as
   diffusers-server CLAUDE.md #39).
2. Block outputs (`num_frames`, `keyframes`, latent shape, etc.) go into `PipelineState`.
   `get_block_state()` only maps declared inputs, so read outputs via `state.get(name)`.

Because video VAE decode internally uses `torch.autocast(dtype=torch.float16)` in
diffusers' `MiniMaxH3VideoDecodeStep`, the weights themselves can remain fp32.
**Never cast the audio VAE at all** (there is a known issue where bf16-ifying it reduces
the generated audio volume by about 20dB, so `runner.py` explicitly passes
`dtype=torch.float32` when loading `vae`/`audio_vae`).

### Measured values (RTX PRO 6000 Blackwell 96GB, 768x768, 124 frames = 5.17s, 30 steps)

| Item | Measured |
|---|---|
| Download size (T2VA-required portion only) | 135GiB (measured HF cache) |
| text_encoder load | 37.6s (cold) / 15.9s (page-cache warm) |
| transformer load | 37.7s (cold) / 10-26s (warm) |
| vae+audio_vae load | 10.0s |
| Prompt encoding | 0.7s |
| Denoise (30 steps) | 157-159s (about 5.4s/step, GPU 100%/600W) |
| VAE decode (video+audio) | 6.5-9s |
| Peak VRAM (during generation) | none: 83.4GB (at decode; 70.4GB during denoise) / bnb-4bit: 91.7GB |
| Request total (via server API, including load) | none: 245s / **bnb-4bit (default): 185s** |
| RAM | Stable at ~6.5GB usage, zero swap increase |

The bnb-4bit peak of 91.7GB fits within a 96GB card, but with only about 4GB of headroom.
If you want to prioritize headroom, you can revert to the legacy method with
`H3_TE_QUANT=none` (peak 83.4GB, +60s/request).

### 48GB-tier VRAM support (`H3_LOWVRAM`, default `0`)

TE-nf4 (21GB) + transformer int8 (34GB) = 55GB cannot be co-resident on a 48GB-tier card
(neither the 96GB machine's default mode nor its int8-both-resident mode is viable).
`H3_LOWVRAM=1` supports the 48GB tier via a phase-cycling scheme that "never lets TE and
transformer be resident at the same time."

- **Forced**: if `H3_TRANSFORMER_QUANT` is unset, it is automatically overridden to
  `int8` (since bf16 66.3GB doesn't fit in 48GB even alone). If `H3_TRANSFORMER_QUANT=none`
  is explicitly specified, startup is rejected with a `RuntimeError`. Specifying
  `H3_TE_QUANT` as anything other than `bnb-4bit` (the default) is likewise rejected at
  startup. `H3_TRANSFORMER_BOTH_RESIDENT` (int8-both-resident) is unconditionally disabled.
- **Steady state**: nothing large stays resident between requests (only the VAE pair is
  CPU-resident; unlike other modes, there is no permanent transformer/TE squatting).
- **t2va/fl2va phases**: [entry: free any resident transformer/transformer_ref] ->
  load TE -> encode (+ keyframe encode for fl2va) -> **run layout/latents/timesteps
  first while TE is still resident** (workaround for the `_execution_device` resolution
  pitfall, see below) -> free TE -> load transformer (int8) -> denoise (~34GB +
  activations ~5GB ~= 39GB) -> free transformer -> VAE->GPU -> decode (~11GB + buffer)
  -> VAE->CPU (**the transformer is not reloaded for the next request** — the next
  request needs encode first).
- **ref2va phase**: following the same principle, reference VAE encode -> layout/latents/
  timesteps (again while TE stays resident) -> free TE -> load transformer_ref (int8) ->
  denoise -> and so on.
- **`_execution_device` resolution pitfall (discovered and fixed during implementation)**:
  the pipeline's component order is `text_encoder, tokenizer, processor, vae, scheduler,
  audio_scheduler, transformer, ...`. With a naive implementation that frees TE and then
  loads transformer, `vae` (an nn.Module that "exists" even when evacuated to CPU) gets
  resolved right after `text_encoder`, so `_execution_device` resolves to `cpu`, and the
  first transformer forward in denoise fails with
  `RuntimeError: Expected all tensors to be on the same device` (reproduced and
  root-caused on real hardware). As a countermeasure, **layout/latents/timesteps are run
  while TE is still GPU-resident, and TE is freed only after those output tensors have
  settled on the correct device** (the same idea as the `force_free_te` deferral pattern
  used in other modes). In ref2va, `reference_encoder_step` must likewise run while TE is
  still resident (because `layout_step` depends on the reference's latent shape).
- **upscale=1 (hires-fix) unsupported**: pass 2 requires roughly 4x the sequence length
  (~16x the attention activation cost), which is unverified within this mode's steady-state
  headroom (~9GB), so it is rejected with a `ValueError` (400).
- **Fix for a pre-existing bug found as a side effect**: `_sync_shared_components_to_ref()`
  inside `generate_ref2va()` was called **before** TE was loaded, so with TE unloaded
  (H3_LOWVRAM, or a case where ref2va is the first request in a `none`-family mode),
  `None` would be synced into `self._pipe_ref.text_encoder`, causing
  `AttributeError: 'NoneType' object has no attribute 'config'` (reproduced and
  root-caused on real hardware). Fixed the ordering to call it **after** TE loads (a fix
  common to all modes, not specific to H3_LOWVRAM).
- **Correctness verification**: since phase cycling shouldn't change the computation
  itself, the t2va output of `H3_LOWVRAM=1 H3_TRANSFORMER_QUANT=int8` and the t2va
  output of normal int8 mode were compared with the same seed (768², 5s, 30 steps, fox
  prompt, seed=12345), confirming **byte-identical mp4** (md5 match).
- **48GB-equivalent real-hardware verification** (constraining free VRAM to ~43.5GB via
  dummy VRAM allocation on the 96GB machine — a stricter condition than the ~47GB free
  typical of an actual 48GB card):

  | | Completed | Peak VRAM | Breakdown (approx.) |
  |---|---|---|---|
  | t2va #1 | Yes | 38.68GB | TE load ~52s + transformer load ~36s + denoise 108s + decode 6.6s |
  | t2va #2 (consecutive) | Yes | 38.94GB | Same fixed cost recurs every time (as expected from a design with no steady state) |
  | ref2va (1 image reference) | Yes | 43.84GB | Denoise 283s (somewhat higher since the reference rows lengthen the sequence and don't fit in the ~39GB range) |
  | upscale=1 | - | - | Judged an OOM risk at implementation time and explicitly rejected with 400 (not run) |

  None of these showed increased host RAM swap (`free -g` showed Swap used stable at
  ~6GB before and after the work). After the work, restarted with `H3_LOWVRAM` unset
  (fully default settings, bf16 transformer) and ran one t2va with the same prompt,
  confirming values like `peak_vram_gb: 91.94GB` / `cache_skipped_steps: 6` were at the
  same level as existing measurements (the table at the top of this README, and the int8
  quantization section), and judged there was no regression.
- ~~**Pre-saving quantized checkpoints to shorten load time**: not investigated~~
  -> **implemented for TE on 2026-08-08** (`H3_TE_PREQUANT`, default ON. TE load
  53.0s->29.5s, request total -35%, equivalence confirmed via MD5 match. See the
  "Disk cache of the quantized text_encoder" section above). Serializing the transformer
  int8 **remains unverified** — it would need about 34GB to save, which is impractical
  given this box's disk headroom (43GB free).

### 24-32GB-tier VRAM support (`H3_LOWVRAM=group`, added 2026-08-05)

#### 2026-11-15: block transfer replaced by an in-house windowed prefetch streamer (`H3_GROUP_STREAM`, default `own`)

The block transfer of `H3_LOWVRAM=group` moved from diffusers' `enable_group_offload(block_level, use_stream=True)`
to the streamer in `core/blockstream.py` (`H3_GROUP_STREAM=diffusers` restores the old path). It follows the
design of the text encoder's `H3_TE_STREAM` (`core/testream.py`): the 50 int8 blocks (30.1 GiB) live in
page-locked host RAM and a copy stream sends `window` blocks ahead of the computation (`H3_GROUP_OFFLOAD_BLOCKS`+1,
default 2). Everything that is not a block (embedders, token_refiner, output layers, 1.6 GiB in bf16) stays on the
GPU. torchao `Int8Tensor`s are moved through their inner `qdata` / `scale` / `zero_point` tensors rather than the
outer wrapper (the same approach as diffusers' `_restore_torchao_tensor`). This is the design proven in
flux3-klein-studio's 6 GB work, ported here.

Measured (GPU0 of an RTX PRO 5000 Blackwell 48 GB, `H3_LOWVRAM=group H3_TE_PRUNE=1`, t2va 512² x 5 s = 124 frames,
30 steps, seed 1, two runs each):

| | diffusers (`H3_GROUP_STREAM=diffusers`) | in-house (`own`, default) |
|---|---:|---:|
| Denoise, `H3_CACHE=none` | 164.9-165.0 s (5.68 s/step) | **96.4-96.5 s (3.32 s/step)** |
| Denoise, `H3_CACHE=fbc` (default) | 165.5-167.6 s (FBC never skips a step) | **85.6-86.1 s (FBC skips 6 of 30 steps)** |
| Peak VRAM | 17.7-18.3 GB | 19.5-20.0 GB (+1.6 GiB of resident non-block modules) |
| Output (MD5) | **identical** to the in-house streamer with `H3_CACHE=none` | same |
| Load (pinning) | 20 s (34 GB into a pinned dict) | 15 s (one buffer per block) |

- The speed difference is the transfer scheme itself (1.71x with FBC off on both sides): the stock hooks pay a
  fixed cost per group and let the CPU run ahead of the GPU, which delays the frees (section 4.2 of
  `docs/lowvram-techniques.md`).
- **FirstBlockCache now works in group mode.** Under the diffusers path FBC never skipped a step (the
  `cache_skipped_steps: 0` noted under "An honest note on speed"); the in-house streamer skips 6 of 30 at the
  default threshold 0.05, so the output differs from FBC-off by exactly FBC's intended approximation. Use
  `H3_CACHE=none` when the exact result matters.
- The group-mode transformer now loads from the prequantized cache (`models/prequant/transformer_int8`) onto the CPU
  when present (the bf16 shards no longer need to be in the HF cache; same fail-open as the other modes). Int8Tensors
  read from the cache are pinned to `requires_grad=False` (diffusers' `swap_tensors` otherwise walks autograd and
  fails with `aten.view unimplemented`). On a box whose bf16 shards were deleted, the `source_snapshot` check of
  `meta.json` is skipped.
- Release goes through `BlockStreamer.release()` (`cudaHostUnregister`) instead of relying on `_host_emptyCache`.
- Host RAM: 30.1 GiB pinned (about the same as the diffusers path's 34 GB pinned dict). The RAM guard
  (`H3_GROUP_OFFLOAD_MIN_RAM_GB`) is unchanged.

#### 2026-11-15: where low VRAM stands — from the 16GB configuration to 8GB / 6GB (three streaming stages)

The same design was applied to the three things that still stayed on the GPU in group mode. All are placement
changes only, and **the outputs match by MD5**.

1. **The projection TE's LM layers** (`H3_TE_STREAM=1` now also applies to `H3_TE_PROJ`): the 4B model's 36 layers
   (1.9 GiB) live in pinned host memory and stream only during encoding. Group mode keeps the TE resident through
   denoise, so this lowers the denoise peak.
2. **The token_refiner's two blocks** (`H3_GROUP_STREAM_REFINER`, default 1): excluded from int8, 1.5 GiB of bf16 stayed resident.
3. **The video VAE decoder's 36 blocks** (`H3_GROUP_STREAM_VAE`, default `auto` = stream when less than 8 GB of VRAM is
   free at the first decode): 4.8 GiB of the VAE's 5.2 GiB (fp16) is the decoder, and after 1 and 2 this was the peak.
   Blocks are re-sent per tile, so decoding gets slower.

Measured (GPU0 of an RTX PRO 5000 48 GB, `H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1`,
t2va 5 s, 30 steps, seed 1):

| | baseline (in-house streamer only) | +1 projection TE | +2 token_refiner | +3 VAE decoder |
|---|---:|---:|---:|---:|
| 512² x 5 s peak | 8.59 GB | 8.33 GB | 6.79 GB | **4.67 GB** |
| 768² x 5 s peak | 10.78 GB | 9.54 GB | 7.99 GB | **7.45 GB** |
| 512² denoise / decode | 90 / 6 s | 86 / 6 s | 90 / 6 s | 89 / 17 s |
| 768² denoise / decode | 200 / 11 s | 193 / 11 s | 195 / 11 s | 195 / 30 s |

Resident VRAM right after startup is 0.18 GB. What remains at the peak is the denoise activations, which depend on
resolution and length.

4. **Token-chunked feed-forward** (`H3_FF_CHUNK`, default `auto` = 8192 tokens in group mode, `0` disables). The largest
   transient in a block is the SwiGLU projection (2 × ffn_dim 14336 bf16 = 57 KB per token); running the feed-forward
   8192 tokens at a time bounds that transient. It is a per-token operation, so the result is the same (MD5 match).

| Activation peak incl. 1K (denoise phase, same configuration) | unchunked | `H3_FF_CHUNK=8192` |
|---|---:|---:|
| 768² × 5 s | 7.45 GB | **6.26 GB** |
| 1024² × 5 s (38k tokens, 13.9 s/step) | 10.73 GB | **9.34 GB** |

What remains is the attention's K and V (0.55 GB each at 1024²) plus the output and the residual stream, close to the
floor for an exact result.

5. **Query-chunked attention** (`H3_ATTN_Q_CHUNK`, default 0 = off, `core/attnchunk.py`). K and V are built in full; the
   Q projection, QK norm, RoPE, the kernel and the output projection run 8192 tokens at a time (MD5 match). It only
   helps at 1K and above: the 1024² × 5 s denoise peak goes 9.34 → **8.53 GB**, 768² is unchanged (6.26 → 6.38). It is
   2-4% slower (13.9 → 14.4 s/step at 1024²), so it is off by default; add `H3_ATTN_Q_CHUNK=8192` for 1K on a small card.

That left 1024² × 5 s at 8.5 GB. Three more changes closed the gap (all MD5-identical):

6. **A memory-lean block forward** (`H3_LEAN_BLOCK`, default `auto` = 8192 tokens in group mode, `core/leanblock.py`).
   The original forward materializes the norm1 output, the attention output, the norm2 output and the FF output for
   all tokens (0.41 GB each at 1024²) before adding them. The lean version runs normalization → modulation →
   projections/attention/FF → gate → residual add in chunks, so the only full-length tensors are the input, the
   output, K and V. It subsumes 4 (FF chunking) and 5 (query chunking); recomputing the norm per chunk costs about 7%
   at 1024².
7. **Streaming in thirds of a block** (`H3_GROUP_STREAM_UNITS=half`, default): adaln_proj → attn → ff as 150 units, so
   the prefetched residency drops from two blocks (1.3 GiB) to two small units.
8. **`H3_TE_DIET=1` for the projection TE** (opt-in): releasing lm_head and bridging the embedding table
   (151936 × 2560) to the CPU frees 1.45 GiB.

| Peak VRAM (same configuration + `H3_TE_DIET=1`, overall, VAE streamed as on a small card) | at first (in-house streamer only) | 1-3 (three streaming stages) | everything (4-8) |
|---|---:|---:|---:|
| 512² × 5 s | 8.59 GB | 4.67 GB | **3.27 GB** |
| 768² × 5 s | 10.78 GB | 7.45 GB | **4.85 GB** |
| 1024² × 5 s | 10.73 GB (denoise) | — | **6.82 GB** |

That was the morning. In the afternoon the 1024² × 5 s breakdown was re-measured (`H3_LEAN_DEBUG=1` logs allocated / peak
around blocks 0-2): 38,321 tokens (31,744 video + audio + text), one residual-stream buffer 0.41 GB, K and V 0.55 GB each,
and the largest transient inside a block is the attention phase (K + V 1.1 GB + the int8 copies sage makes inside its
kernel + the chunk). Four more changes:

9. **Rounding the fp32 patch embeddings** (`shrink_prologue` in `core/leanblock.py`, installed together with
   `H3_LEAN_BLOCK`). The transformer forward computes the `proj_in` output (a layer kept in fp32) for all tokens and
   rounds it to bf16 when packing, but the fp32 version survives as a local variable for the whole block loop (0.68 GB at
   1024²). Rounding at the projection's exit gives the same values with half the residency. 1024² 6.82 → **6.41 GB**,
   768² 4.85 → **4.62 GB** (MD5 match).
10. **In-place residual update** (same module). From block 2 on, each block writes into its input buffer (every chunk
    finishes reading its own rows before writing them, so the result is the same; MD5 match). FBC references the input
    and output of block 0, so blocks 0 and 1 write new buffers; together with the three residual tensors FBC keeps
    (head_block_output / head_block_residual / tail_block_residuals, 1.2 GB in total at 1024²) the peak stays at block 1,
    so the overall figure does not move with FBC on. With `H3_CACHE=none` every block is in place and one buffer is saved.
11. **Head-grouped attention** (`H3_ATTN_HEAD_GROUPS`, default `auto`). K, V and Q are built for the 56 heads in G groups
    (row slices of the to_q/to_k/to_v weights; the streamer keeps only the inner tensors of torchao's Int8Tensor on the
    GPU, so the slice is rebuilt from them). Each group's attention output goes into one full-width buffer (attn_out) and
    the output projection runs from it for all heads at once. The resident set goes from two buffers (K, V) to
    attn_out + K/G + V/G, and the K/V copies sage makes inside its kernel shrink by G as well. Recomputing the norm and
    modulation G times costs time (G=4 +13%, G=8 +15-35%, G=28 +70%). **Same math, but not bit-identical**: changing the
    GEMM width (7168/G) and the head count seen by the attention kernel changes kernel selection and summation order,
    so the last bf16 bit can move (sage's K-mean centering matches or not depending on the head count and token count).
    Thirty steps amplify that into a different sample of the same video (PSNR against the baseline 34.4 dB at 1024²,
    38.7 dB at 768²; FBC on vs off differs by 34.2 dB, so this is the same order as the FBC approximation that is on by
    default, not a quality loss). Therefore **auto always picks G=1 when G=1 fits** and otherwise the smallest of
    4 → 7 → 8 → 14 → 28 that fits the free VRAM (a `H3_VRAM_LIMIT_GB` cap is honored; decided at the first forward of a
    generation and kept for that generation, after reserving FBC's three residuals and 1 GiB of slack). Fixed G=4: 1024² 5.09 GB (16.7 s/step, +8%).
    Fixed G=8: 1024² **4.79 GB** (17.9 s/step), 768² **3.95 GB** (8.8 s/step). Under a 5.8 GB cap, auto chose G=4 → 28 and
    peaked at 4.79 GB (26 s/step). Even a combination that matched bit-for-bit in a single-block check (G=4 at 1024²) gave a
    different MD5 after 30 real steps, so treat any G>1 as a different sample.
12. **Turning FBC off** (`H3_CACHE=none`) drops the three FBC residuals and makes every block in place. With G=8 the
    1024² × 5 s denoise peaks at **3.03 GB** (all 30 steps are computed, 23 s/step; FBC itself is an approximation, so the
    output changes).

| Peak VRAM (same configuration, overall) | up to 8 (morning) | 9-10 (MD5 match, default) | + head groups G=8 (different sample) | + `H3_CACHE=none` |
|---|---:|---:|---:|---:|
| 512² × 5 s | 3.27 GB | **2.90 GB** | — | — |
| 768² × 5 s | 4.85 GB | **4.62 GB** | 3.95 GB | — |
| 1024² × 5 s | 6.82 GB | **6.41 GB** | 4.79 GB | 3.03 GB |

Exact (MD5-identical) results bottom out at **512² 2.90 GB / 768² 4.62 GB / 1024² 6.41 GB**; what remains is K and V, the residual stream
and the three FBC residuals. Allowing head groups puts **1024² × 5 s on a 6 GB card** (auto uses them only when needed).
`/api/status` reports `group_stream_refiner`, `group_stream_vae`, `group_stream_units`, `ff_chunk`, `attn_q_chunk`,
`lean_block`, `attn_head_groups` and `lean_block_last` (token count, G and the first in-place block of the latest forward;
the generation result JSON carries `lean_block_last` too).

#### PCs with 32 GB / 16 GB of host RAM: streaming the weights straight from the files (`H3_GROUP_STREAM_HOST=file`, 2026-11-15)

Everything above keeps the whole int8 transformer (30 GiB) in pinned host RAM, which assumes a 48 GB-class machine.
`H3_GROUP_STREAM_HOST=file` (`core/filestream.py`, the same scheme as minimax-h3-image-server's `app/filestream.py`) reads
each unit (a third of a block) from the prequantized cache (safetensors; `from_pretrained` maps the files, so nothing is
expanded into RAM) with 8 threads in 32 MB chunks into a small pinned staging ring (3 × 0.24 GiB, two units of read-ahead)
and copies from there to the GPU. The only mandatory host RAM is that ring plus the token_refiner's (2 × 0.36 GiB); whatever
else fits in the OS page cache makes it faster. The tail units that do not fit the cache are dropped with
`posix_fadvise(DONTNEED)` after every read so the cyclic access does not flush everything through LRU (the number is `auto`:
weights + the TE / VAE pinning that comes later − usable RAM, honoring a cgroup limit). In file mode the VAE decoder's
streaming source is not pinned either (the fp16 copy is used as is). The int8 weights are the same, so the output is the same.

- `H3_GROUP_STREAM_HOST`: `auto` (default: file when the installed RAM is below `H3_GROUP_STREAM_FILE_BELOW_GB` = 48 GB;
  **on Windows always file regardless of RAM**: 2026-11-19, WDDM moves multi-GB page-locked regions pathologically slowly, and on an
  RTX 3050 6GB / 32 GB RAM machine streaming the 4.5 GB decoder from pinned RAM ran at about 0.5 GB/s, a 512² decode of 590 s,
  while streaming from file through a 0.4 GB staging ring took 0.96 s per pass, 53 s. The VAE decoder's pinned path is likewise
  not used on Windows),
  `pinned`, `file`. Falls back to pinned when the weights were quantized from bf16 at load (no cache file to read).
  `H3_GROUP_STREAM_DROP_UNITS` (auto / a number), `H3_GROUP_STREAM_RING` (staging buffers, default 3). `H3_FILE_STREAM_HOST_MIN_MB` (default 1): tensors missing from the files (the turbo LoRA's
  lora_a / lora_b, 1.24 GiB in total) at least this large are not kept on the GPU but held in pageable host RAM and
  memcpy'd into the staging buffer per unit (2026-11-18; resident, they made a 6GB card spill into Windows shared memory
  and crawl, measured on an RTX 3050). `/api/status` reports
  `group_stream_host` / `group_stream_file`.
- Windows has no `fadvise`, so the drop control does not apply (reading itself works, opened with
  `FILE_FLAG_SEQUENTIAL_SCAN`; the same scheme was verified on the image server's RTX 3050 / 16 GB RAM machine).
  2026-11-18: `FILE_FLAG_SEQUENTIAL_SCAN` is now applied only to weight sets that do not fit in RAM (the 30 GB transformer);
  sets that fit (the 4.5 GB VAE decoder, the 1.9 GB projected TE) are opened with normal caching. Applying it to everything
  made every per-tile re-read of the decoder hit the NVMe on an RTX 4060 Laptop / 16 GB RAM machine (decode 124 s, NVMe
  pinned at about 1 GB/s). The rule is "weights + 1 GiB + reserve <= usable RAM"; the log says `page cache kept / not kept`.
- Same day: a path that keeps the decoder **resident on the GPU** (`H3_VAE_DECODER_RESIDENT`, default auto, `ResidentLoader` in
  `core/filestream.py`). A 768², 5 s decode is 7 clips x 16 tiles, about 150 passes, and re-sending the 36 blocks (4.5 GB) from the
  host per tile made that transfer (page cache -> staging -> GPU) most of the decode time on 16 GB-RAM PCs (RTX 4060 Laptop: 0.68 s
  of each 0.77 s pass waiting for reads, decode about 90 s). When free VRAM is at least 4.5 + 0.5 + output + 0.5 GiB headroom the
  decoder is read once and kept. Forced on the A4000 under an 8 GB cap: decode 86 -> 34 s, total 201 -> 152 s, peak 7.03 GB,
  identical output hash. When the whole decoder does not fit, only as many blocks as fit are kept and the rest is streamed
  (`SplitDecoder`; default headroom 1.0 GiB). A4000 emulation: 30/36 resident under an 8 GB cap, decode 33.8 s, peak 6.22 GB,
  total 149 s; 15/36 under a 6 GB cap, decode 51 s, peak 4.21 GB. Identical output hashes in both. The streamed remainder shrinks
  to 0.75-2.6 GB, so it also stays in the page cache (the 90-300 s decode swings on 16 GB-RAM PCs came from the 4.5 GB decoder
  falling in and out of the cache).
- 2026-11-19, tried and rejected: 512 px VAE tiles instead of 256 px to cut the pass count to a quarter (on 8 GB laptops the
  decode is GPU-compute-bound and this was the only remaining lever). Under an 8 GB cap on the A4000 the 768² decode drops from
  86 to 38 s, but **the whole frame gets a grid-like artifact** (PSNR 22 dB, SSIM 0.60 against the 256 px output; everywhere, not
  just at seams). The decoder's transformer blocks appear to be trained for 256 px tiles, so the tile size cannot be changed. Not
  added to the code. On the RTX 4060 Laptop (26/36 resident, zero read wait) the 85.7 s decode, about 0.57 s x 150 passes, is
  the floor for that GPU.
- 2026-11-19: during the decode the pinned staging buffers of the transformer / token_refiner / TE streamers (about 1.6 GiB in
  total) are returned to free host RAM and re-created at the next forward (`FileBlockStreamer.park()`). On 16 GB-RAM PCs that
  amount decides whether the streamed part of the decoder stays in the page cache (on a real RTX 3050 6GB / 16 GB RAM machine with
  a slow SATA-class SSD and 13/16 GB RAM in use, the 512² decode took 772 s, slower than the 569 s at 768²: the 3.1 GB streamed
  part was being evicted and re-read from the SSD every pass). Output is unchanged.
- 2026-11-19: the estimated count is only an upper bound; while loading, the driver's free VRAM (mem_get_info) is checked
  before every unit and loading stops before it falls under "activations + output + headroom" (`ResidentLoader(min_driver_free_gib=...)`).
  Memory torch keeps reserved may be unusable due to fragmentation; loading on the estimate alone pushed past physical VRAM and
  Windows spilled to shared memory, becoming extremely slow (real RTX 3050 6GB / 32 GB RAM: driver free 1.9 + torch cached 1.8 GB
  led to 13/36 resident, then 824 s for a 512² decode and 648 s at 768², with 6.4 GB of shared GPU memory).
- Same day, further: on Windows (WDDM) the `mem_get_info` free figure can include other processes' evictable memory, so the
  floor passed while physical VRAM was already exceeded (RTX 3050 6GB / 32 GB RAM: driver free 2.2 GB, 15/36 loaded, Task Manager
  at 5.8 / 6.0 GB dedicated, a 512² decode of 708 s = over 4.5 s per pass in the spilled state). The physical figure from
  `nvidia-smi --query-gpu=memory.used,memory.total` (all processes, the same number Task Manager shows as dedicated GPU memory),
  matched by UUID, is now used alongside the driver's figure, taking the smaller of the two for both the estimate and the floor
  (`filestream.physical_free_gib`). Without nvidia-smi the previous behaviour applies.
- Same day, settled: a throwaway probe that runs the decoder alone in a fresh process (not kept in the repo) on the RTX 3050 6GB gives 0.96 s per
  pass with nothing resident, 52.6 s for a whole 512² decode (about the A4000's 0.75 s). The slowness was never compute, only
  spilling from keeping too many blocks. The cause is Windows torch's pool growth: 0.88 GB of tensors but 2.60 GB reserved (no
  `expandable_segments`), which the estimate ignored. On Windows the rule now (a) does not count reserved-but-unused memory as
  usable and (b) uses a default headroom of 2.5 GiB (1.0 GiB on Linux). Inside the server on the 3050 (2.2 GB physically free)
  that means 0 resident blocks; in a fresh process about 15.
- Same day, further: even with 0 resident blocks and the file path, the server slowed down later in the decode (5.9 GB shared).
  The remaining difference from the probe (fresh process, empty pool) is torch's pool: the server enters the decode holding
  3.05 GB reserved after the denoise (1.18 GB in use) and the pool then grows 2-3x during the decode, past 6 GB.
  `H3_GPU_POOL_DEFRAG` (auto = Windows only): right before the decode, the live GPU tensors (VAE non-block parts, audio VAE,
  transformer non-block parts, about 1.2 GB) are moved to the CPU, `empty_cache` releases the whole pool, and they are moved
  back (`_defrag_gpu_pool`). Output unchanged.
- Same day, further: emptying the pool did not work (reserved 2.84 -> 2.66 GB; scattered small live tensors keep segments
  alive). Instead, **on Windows an automatic VRAM cap of card size minus 1.2 GB is applied when `H3_VRAM_LIMIT_GB` is unset**
  (`H3_VRAM_AUTO_CAP_MARGIN_GB`). At the cap the allocator releases unused blocks and reuses them, so a fragmented pool cannot grow
  past physical VRAM. The A4000 emulations at 6 GB / 8 GB use exactly this mechanism and showed neither spilling nor OOM.
- The block count is decided from "driver free + torch's reserved-but-unused memory" (Linux only). Windows torch has no
  `expandable_segments`, so a few GB stay reserved after the denoise (real RTX 4060 Laptop: allocated 1.46 / reserved 3.33 GB,
  driver free 3.8 GB, so only 10/36 blocks fit at first); that memory is reused by new allocations and may be counted.
- The quantized cache is built beforehand with `scripts/prepare_lowvram.py` (fetches the bf16 shards one at a time,
  quantizes and deletes them; fine with 16 GB RAM). Ref2VA / Ref2I use a different set of weights, `transformer_ref/`
  (66 GB, every shard differs from `transformer`), so `--ref` also builds `transformer_ref_int8` (+32 GB). Until
  2026-11-18 the ref2va group load ran the 40 GB RAM guard before looking for the cache, so the reference modes always
  stopped on low-RAM hosts; it now matches the base side (no guard when the cache exists and is streamed from file)
  and, when the cache is missing, stops with a message pointing at `--ref`.

Measured on GPU0 of the 48 GB machine with the RAM limit emulated by a cgroup and the cache's page cache dropped before
each run (this machine's NVMe reads 3.3 GB/s directly; every output is MD5-identical to pinned):

| Where the weights live | Mandatory host RAM | 512² × 5 s | 1024² × 5 s | Notes |
|---|---:|---:|---:|---|
| pinned (reference) | about 38 GiB (transformer 30 + refiner 1.4 + TE 1.9 + VAE 4.5) | 2.98 s/step | 15.1 s/step | VRAM 2.90 / 6.41 GB |
| file, no RAM limit (warm cache) | about 8 GiB | 3.11 s/step | — | VRAM 2.49 GB (one less unit prefetched on the GPU) |
| file, 28 GB RAM cap (a 32 GB PC) | same | 6.1 s/step | **14.7 s/step** | 94 of 150 units read from disk; at 1K the reads hide behind compute |
| file, 16 GB RAM cap | same | 8.0 s/step | — | all 150 units read from disk (30 GiB/step), MD5 match |
| file, 12 GB RAM cap (a 16 GB PC), VAE and TE from files too (below) | about 2 GiB | 8.0 s/step | **15.4 s/step** | MD5 match; the VAE is pinned only during the decode when RAM allows |

At 1K the reads (worst case 30 GiB/step ÷ 3.3 GB/s ≈ 9 s) hide behind 15 s of compute, so it is as fast as pinned; 512² is
read-bound and takes about twice as long. **A 32 GB PC runs 1024² × 5 s at the pinned speed, and a 16 GB PC runs 1K too** (speed depends on the disk's sequential
read speed). In the first 12 GB-cap test the process's anonymous memory reached about 12 GB right before the decode (`proc_anon_gb`:
the fp16 copy of the VAE 4.8, staging 1.6, the TE's pinned layers 1.9-2.4, baseline 0.6, plus the decode's frame buffers).
The two remaining pieces were moved to the same scheme, and 1K now runs with 16 GB of RAM:

- **An fp16 cache of the VAE decoder** (`core/vaecache.py`, `models/prequant/vae_decoder_fp16/`, 4.5 GiB, written once in
  2 s). In file mode the decoder blocks are not cast to fp16 at startup (that copy was 4.8 GiB of anonymous memory); they
  become 0-element placeholders. At decode time the free RAM decides: **with 4.5 GiB + 2 GiB available the cache is read
  into pinned memory in 1.4 s and kept only for that decode** (same speed as pinned); otherwise it is streamed from the
  file per tile (1K decode 48 → 95 s). The pinned copy is released afterwards. The values are the same rounding as
  `vae.to(float16)`, so the output is unchanged.
- **A quantized cache of the projection TE** (`H3_TE_PROJ_PREQUANT=1`, default, `models/prequant/te_proj_<model>_bnb-4bit/`).
  The bnb-4bit TE is saved once (2 s) and loaded from there afterwards (on this machine the load takes the same 9-10 s as
  quantizing from bf16, but it reads 2.2 GB of nf4 instead of 8 GB of bf16, which matters on a slow disk). In file mode its 36 LM layers (1.9 GiB) are streamed straight from that file instead of being
  pinned (staging 2 × 0.06 GiB).

With that, **1024² × 5 s completes under the 12 GB cap** (a 16 GB PC): 15.4 s/step, decode 97 s streamed from the file,
MD5 match; 768² runs at 8.3 s/step (the VAE is pinned for the decode, 30 s; MD5 match). The mandatory host RAM is the staging ring plus the TE's
(about 1 GiB) and the process itself, about 2 GiB in total.

##### VRAM class × host RAM class at a glance (t2va 5 s, 30 steps, the default low-VRAM configuration)

Speeds are seconds per step on this machine (RTX PRO 5000); an RTX 3050 takes several times longer. ○ = fits with
MD5-identical output, △ = fits with head groups (a different sample) or `H3_CACHE=none`, ✗ = does not fit.

| VRAM \ host RAM | 16 GB (file, every unit from disk) | 32 GB (file, part from disk) | 48 GB+ (pinned) |
|---|---|---|---|
| **6 GB** | 512² ○ 8.0 / 768² ○ 8.3. 1K △ about 17 (head groups; decode streamed from the file, 97 s) | 512² ○ 6.1 / 768² ○ 7.1. 1K △ 16.8 (auto picks G=4, 4.4 GB) | 512² ○ 3.0 / 768² ○ 7.1. 1K △ (head groups 4.8 GB, `H3_CACHE=none` 3.0 GB) |
| **8 GB** | same; 1K ○ 15.4 (5.8 GB, decode 97 s) | same; 1K ○ 14.7 (5.8 GB) | same; 1K ○ 15.1 (6.4 GB) |
| **12-16 GB** | same | same; with 8 GB+ of free VRAM the VAE is not streamed and the decode is faster (768²: 31 → about 11 s) | same |
| **24 GB+** | same | the earlier tables (`H3_LOWVRAM=1` family) also apply | same |

- VRAM decides what fits: 512² (2.9 GB) and 768² (4.6 GB) fit 6 GB; 1K stays MD5-identical on 8 GB (pinned 6.4 GB /
  file 5.8 GB) and needs head groups (auto uses them only when required) or `H3_CACHE=none` on 6 GB.
- Host RAM decides the speed: 48 GB+ pins the weights and is fastest, 32 GB runs 768² and 1K at the same speed and 512²
  at 2×, 16 GB runs 512² at 2.7× and 768² at 1.1×, and 1K runs as well (the VAE and the TE are streamed from files too).
- File mode prefetches one unit less on the GPU, so the VRAM peak drops too (512² 2.49 / 768² 3.71 / 1K 5.76 GB).
  Whether the streamed VAE decoder is pinned in file mode is `H3_GROUP_STREAM_VAE_PIN` (auto = RAM 24 GB+; without it
  the 768² decode goes 31 → 53 s).

`H3_LOWVRAM=1` (48GB tier) loads the full transformer (34GB) onto the GPU every request,
so it doesn't even fit alone on a 24-32GB-tier card. `H3_LOWVRAM=group` ports the
"block-level group offload" pattern established in diffusers-server (a sibling project)'s
CLAUDE.md #33/#34/#37 into this project, keeping the transformer **resident in host RAM**
while shuttling only the blocks needed for each denoise step (1-2 of 50 layers,
~0.68GB x 1-2) to and from the GPU. The transformer is **loaded exactly once at process
startup and stays resident across requests** (unlike `H3_LOWVRAM=1`'s per-request reload).

#### Investigation results on the PR side's "load-time quantization during streamed offload"

At the time of this task, reading `TorchAoHfQuantizer`'s (in
`quantizers/torchao/torchao_quantizer.py`) `validate_environment()` showed that it only
sets `self.offload = True` when `device_map` is a **dict** (like accelerate's automatic
assignment) containing the **string value** `"cpu"`, and `check_if_quantized_param()`
skips quantization of CPU-placed parameters when this flag is set (i.e., by design,
parameters that get CPU-offloaded are intentionally left unquantized). This
implementation, however, uses `device_map={"transformer": "cpu"}`, which passes through
`load_components()` and ultimately reaches `from_pretrained()` as the **plain string**
`"cpu"`, which `modeling_utils.py`'s normalization code converts into a **single-key
dict** `{"": torch.device("cpu")}` (whose value is a `torch.device` object, not a string).
Since `torch.device("cpu") == "cpu"` evaluates to `False` in Python,
`"cpu" in device_map.values()` remains False, and `self.offload` is never set. In other
words, **loading onto the CPU does not skip quantization — it is correctly quantized as a
torchao Int8Tensor** (confirmed on real hardware via `scripts/probe_group_offload.py`
that 370/370 layers become Int8Tensor). Conclusion: **int8 quantization on the CPU works
without issue**.

#### Implementation summary

- `_ensure_transformer_group()` (`core/runner.py`) quantization-loads onto the CPU with
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)`, then calls
  `enable_group_offload(offload_type="block_level", num_blocks_per_group=1,
  use_stream=..., low_cpu_mem_usage=...)` (transformer_ref has an analogous
  `_ensure_transformer_ref_group()`).
- TE requires bnb-4bit just like `H3_LOWVRAM=1` (forced at startup). In t2va's steady
  state, TE stays resident across requests too (since the transformer is resident in the
  first place, it's better to keep TE resident as well and avoid per-request reload cost).

#### [Major finding] `use_stream=True` + `low_cpu_mem_usage=True` combined has a bug that breaks torchao Int8Tensor

Actually running a forward pass in `scripts/probe_group_offload_forward.py` confirmed on
real hardware that it always fails at the first denoise block with
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be pinned`.
`hooks/group_offloading.py`'s `_pinned_memory_tensors()` (called unconditionally every
step from `_onload_from_memory()` when `use_stream=True`) attempts `.pin_memory()`
regardless of `low_cpu_mem_usage`'s value, while `_init_cpu_param_dict()` (run once at
`enable_group_offload()` call time) skips pinning when `low_cpu_mem_usage=True` — an
asymmetric implementation where the two sides' assumptions don't line up. When this
mismatch occurs, torchao's `Int8Tensor.qdata` ends up in a broken state (internally
recognized as `torch.cuda.CharTensor`), and calling pin_memory() on it crashes. Results
of a controlled comparison in `scripts/probe_group_offload_fix.py`:

| Setting | Result | Onload/offload per block |
|---|---|---|
| `use_stream=True, low_cpu_mem_usage=True` (combined) | **crash** | - |

(Corrected 2026-08-10: this combination was originally described here as "the diffusers default", which was wrong — the `apply_group_offloading` API defaults are `use_stream=False, low_cpu_mem_usage=False`; the crash is hit when both are opted into for a memory-constrained setup.)
| `use_stream=False, low_cpu_mem_usage=True` | works | onload 0.1-0.26s / offload ~0.22s |
| `use_stream=True, low_cpu_mem_usage=False` | works | **onload 0.04-0.07s** / offload ~0s |

Adopted `low_cpu_mem_usage=False` (eagerly pins all parameters at
`enable_group_offload()` call time) as the new default (`H3_GROUP_OFFLOAD_LOW_CPU_MEM`,
default `0` = False). Reason: onload is 4-5x faster (pinned memory is not pageable, so
DMA transfer is faster). The cost is an extra ~14-16GB of host RAM pinned at load time
(page-locked, so it cannot be swapped), and `enable_group_offload()` itself taking about
22 seconds (measured on real hardware: 70s to load onto CPU + 22s to pin = ~90s total).
If you want to prioritize less RAM usage, explicitly set
`H3_GROUP_OFFLOAD_LOW_CPU_MEM=1` (in which case `H3_GROUP_OFFLOAD_USE_STREAM` also
automatically falls back to `0` unless explicitly specified, to avoid the broken
combination above).

#### Final choreography (phase x resident items x peak)

| Phase | Large resident items | Notes |
|---|---|---|
| Startup preload | transformer (int8, CPU-resident + group-offload hooks) | About 90s (CPU load 70s + pinning 22s) |
| t2va encode | TE-nf4 (GPU, 21GB) + transformer (CPU) | |
| t2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer blocks (GPU, ~1.4GB) | |
| t2va decode | VAE pair (GPU, ~11GB) + 1-2 transformer blocks | **TE is force-freed only for this window** (see below), reloaded after decode |
| ref2va reference encode | VAE pair (GPU, 11GB) | TE is force-freed only for this window (see below) |
| ref2va denoise | TE-nf4 (GPU, 21GB) + 1-2 transformer_ref blocks | |
| Steady state between requests | transformer (CPU) + TE-nf4 (GPU) | After ref2va, transformer_ref returns to unloaded (reloaded every t2va<->ref2va switch) |

#### [Second bug discovered and fixed during implementation] TE needed to be force-freed for the decode window and reference-encode window

Initially designed on the assumption that "since group-offloaded transformer's actual
GPU footprint is tiny (~1.4GB), there's no need to free the transformer during decode,"
but a 32GB dummy-VRAM verification run reproduced `CUDA out of memory` on real hardware.
Enumerating actually-live CUDA tensors via `_log_gpu_tensor_diag()` (a temporary
diagnostic function enabled with `H3_DEBUG_MEM_DIAG=1`, left in `core/runner.py`) revealed
that TE-nf4's own embedding table / lm_head weights (shape `(151936, 5120)`, bf16,
1.556GB x 2 = a combined total of over 3.1GB out of 22.25GB total) had **remained
resident** right up until decode (not freed by `empty_cache()` alone, since they were
genuinely-referenced live tensors). In other words, the true requirement was TE-nf4
(21GB) + decode-only buffers (~16.3GB, see the VAE tiling investigation below) = 37GB —
a conflict between TE and VAE, unrelated to the transformer's footprint. Fix: **force-free
TE for the decode window (and ref2va's reference VAE encode window), reloading it after
exiting the window** (a dedicated piece of logic separate from `force_free_te`; the
`_execution_device` resolution order is secured by ensuring TE is freed before VAE goes
to the GPU).

#### Result of the MD5-match check

Comparing the output of normal int8 mode (`H3_LOWVRAM` unset,
`H3_TRANSFORMER_QUANT=int8`, FBC `H3_CACHE=fbc` enabled) against the output of
`H3_LOWVRAM=group`, with the same seed (768², 5s, 30 steps, fox prompt, seed=12345),
**FBC's cache-skip decisions differed due to the difference in execution path**
(`cache_skipped_steps` was 6 vs 0), so a naive comparison found the mp4s mismatched.
Since FBC's decision is a numerically sensitive threshold on residual similarity to the
previous step, mathematically equivalent computations can still yield different skip
decisions depending on the path (not degradation). Re-comparing with both modes set to
`H3_CACHE=none`, confirmed **byte-identical mp4** (md5 match). This substantiates that
group offload's computation is mathematically identical to the existing path.

#### Measurement tables: 32GB-limit and 24GB-limit probes

**32GB limit** (free VRAM constrained to ~30GB via dummy VRAM allocation, `H3_CACHE=fbc`
still enabled):

| | Completed | Peak VRAM | Time taken |
|---|---|---|---|
| t2va #1 (768², 5s, 30 steps) | Yes | **28.67GB** | denoise 220.79s / decode 6.31s / total 337.19s (including first-time TE load) |
| t2va #2 (consecutive) | Yes | **28.23GB** | denoise 220.85s / decode 6.01s / total 278.83s (shorter since TE stays resident) |
| ref2va (1 image reference, 768x1344) | **Rejected for insufficient RAM** (see below) | - | - |

Both runs produced an mp4 identical to run #1 (md5 match, `be3f32a84de074990208ad0d30f31a63`).
No increase in host RAM/swap in any phase (`free -g`'s Swap used remained stable at
~7-8GB before and after the work — this figure comes from other processes).

**24GB-limit probe** (free VRAM constrained to ~22GB via dummy VRAM allocation):

- 768²: OOM during denoise (onload of transformer blocks). Actual consumption 21.85GB, of
  which TE-nf4 alone accounts for 21GB. **TE-nf4's own fixed size (21GB) consumes the
  majority of the 22GB budget, leaving no room even for the additional onload of a single
  transformer block (~148MB).**
- 544x960 (temporarily added to RESOLUTION_PRESETS for the test, removed afterward):
  **OOM at the same point, same 21.85GB.** Lowering the resolution changed neither the
  failure point nor the consumption, **confirming, just as with the VAE tile shrink test,
  that a resolution-independent fixed cost is the bottleneck** (see the VAE tiling
  investigation below).
- **Conclusion**: with the current architecture (a design where TE is bnb-4bit and stays
  resident almost all the time), TE-nf4 alone consumes 21GB, the majority of a 24GB-tier
  card's effective budget (~22GB), and no resolution makes it viable. Supporting the
  24GB tier would require either freeing TE during denoise (reverting to an
  `H3_LOWVRAM=1`-like design) or further lightening TE itself (e.g. GGUF, though applying
  GGUF to transformers-family models is structurally difficult). Since this is outside the
  scope of this task (the goal was 48GB->24-32GB support, with 24GB an exploratory probe),
  it is concluded unsupported for now.

#### VAE tiling investigation results

Benchmarking a 768², 124-frame VAE decode standalone (from a synthetic latent, bypassing
denoise) directly via `scripts/probe_vae_tile_size.py` showed that **shrinking
tile_sample_min_height/width from 256 (default) -> 192 -> 128 -> 96 made no difference
whatsoever to peak VRAM (16.29GB)** (time taken worsened from 5.9s to 10.5s as tile count
increased). This result suggests decode's peak is governed not by the spatial-tile
compositing buffer, but by **a buffer that decodes one whole chunk's worth of temporal
chunking (`tokens_chunk_size` units) at a time**, or by a fixed VAE-architecture overhead
(the VAE class doesn't expose the temporal chunk size as a public parameter, so further
tuning would require code changes and is out of this task's scope).
**Conclusion: adjusting spatial tile size is meaningless for 24-32GB-tier support**
(keep the default).

#### Result of confirming FBC/sage coexistence

Throughout all ballast verification (both 32GB and 24GB), running with `H3_CACHE=fbc`
(default) and `H3_ATTN_BACKEND=sage` (default) enabled produced no conflicting errors with
the group-offload hooks (FBC makes block-level skip decisions, group offload manages
block GPU residency, and sage handles attention implementation inside each block — three
independent layers). Separate from the MD5-match check (done with `H3_CACHE=none`), a
full run with the default FBC-enabled settings (2 t2va runs under the 32GB limit) also
completed successfully.

#### RAM constraint for Ref2VA (a known, unresolved limitation)

Confirmed on real hardware that calling ref2va after running t2va with `H3_LOWVRAM=group`
gets rejected by the host RAM guard regardless of VRAM budget (reproducible even on the
96GB machine with no dummy VRAM allocation applied):

```
H3_LOWVRAM=group requires at least 40.0GB of available host RAM before loading
the (~34GB, permanently CPU-resident) int8 transformer, but only 33.0GB is
available right now.
```

Root-cause narrowing: `avail_gb` recovers correctly (around 44.6GB) right after freeing
the transformer, but drops to around 33GB during the subsequent TE reload -> reference VAE
encode -> layout computation process (confirmed in real-hardware logs). Since
`swap_used_gb` consistently shows no increase, no actual swapping occurred — it is more
likely that `MemAvailable` (Linux's conservative buff/cache-inclusive heuristic estimate)
fluctuates more conservatively than the true free RAM. That said, even looking at `used`
from `free -g` — 62GB of 94GB used (32GB left) — securing an additional 34GB is
inherently tight, so we could not conclude the guard itself is wrong (even on the 96GB
machine, the design of "keeping the t2va transformer resident while also trying to pin the
ref2va transformer_ref onto the CPU" is already tight against the physical capacity of a
94GB-RAM machine). **We chose to err on the safe side and record this as a known
limitation rather than loosening the guard** (prioritizing the lesson learned from a past
swap-runaway incident). Future improvement candidates: stop eagerly loading the
transformer in `preload_all()` and lazily load it like TE (a latency tradeoff for the
first request), or design a path that consumes less RAM when switching between t2va and
ref2va. **On a machine with 48GB+ RAM, this problem likely does not occur (unverified, but
there should be more RAM budget headroom)** (this task only verified on a 94GB-RAM
machine; retesting on a machine with more RAM has not been done).

### 2026-08-09: H3_KEEP_TRANSFORMER — eliminates the per-request reload fixed cost by keeping the transformer resident

`H3_LOWVRAM=1` frees the transformer right before decode and reloads it **every single**
request (measured fixed cost of 14.8-32.7s, `docs/RESIDENCY.en.md` §5.5). `H3_KEEP_TRANSFORMER=1`
skips this freeing, keeping the transformer resident on the GPU and carrying it over to the
next request, converging this fixed cost down to a single occurrence at the initial load.

**All three preconditions are mandatory** (enforced by an import-time guard in
`core/runner.py`; a `RuntimeError` is raised if any is missing). The VRAM-budget reasoning:

1. `H3_LOWVRAM=1` (raw `"1"` only; `group` is a different design that keeps transformer
   host-RAM-resident with block offload, and is out of scope)
2. `H3_TE_DEVICE` set (keeps TE resident on a separate GPU) — **without this, the encode
   phase breaks down first**: TE-nf4 17.45GB + resident transformer-int8 34.3GB = 51.75GB
   exceeds the effective budget
3. `H3_VIDEO_VAE_FP16=1` — the encode phase can avoid the issue by offloading TE to
   another GPU, but if the decode phase is left at its fp32 decode peak of 16.29GB, then
   transformer 34.3 + 16.29 = 50.6GB exceeds the effective budget (~49.8GB). With the VAE
   fp16-ified, the decode peak drops to ~11.4GB, so 34.3 + 11.4 = 45.7GB fits (the quality
   impact of video VAE fp16 is already measured at PSNR **39.97dB** with no visible
   difference — see the "fp16-ifying the video VAE" section for details)

Default (`H3_KEEP_TRANSFORMER=0`) leaves behavior unchanged. **ref2va (`transformer_ref`)
is out of scope** — it remains outside this flag and continues to be freed every time as
before.

```bash
# Recommended launch command (48GB GPU0 + 20GB GPU1)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

**Real-hardware E2E measurement (2026-08-09, 48GB GPU0 + 20GB GPU1, the launch
configuration above)**:

- transformer (int8) is loaded exactly once, on the first request (32.0s). No reload on
  subsequent requests (confirmed in server logs)
- t2i turbo 4 steps, steady state: **9.7s/image** (denoise 4.32s, decode 1.5s). Peak VRAM
  41.97GB (during denoise)
- t2i steps=30, steady state: 51.1s (denoise 45.7s). Peak 41.97GB
- t2va 5s turbo 4 steps: **44.2s** (denoise 26.05s, decode 10.81s). **Peak VRAM
  44.15GB = the decode phase** (transformer 34.03GB resident + fp16 decode). Derived
  prediction was 45.7GB, measured 44.15GB, versus the cataloged 48.9GB with ~4.8GB
  headroom
- nvidia-smi measured peak 42,620 MiB (1-second sampling; torch's measured 44.15GB is the
  authoritative instantaneous peak)
- Same-seed output equivalence: PNG MD5 **exact match** against the flag-OFF baseline
  (all else equal) (seed=11, md5 `665eadddea8f34298a1b5b89e69d4bd0`). Baseline side:
  total 63.27s (including transformer load) / peak 36.4GB

**Speedup lineage** (48GB GPU0, `H3_LOWVRAM=1` family, t2va at 768², 5s):

| | t2i | t2va 5s |
|---|---|---|
| No turbo, 30 steps (bare configuration right after GPU swap, 08-07) | 157s | 351.4s (denoise 197.7s) |
| lightx2v turbo 4 steps introduced (08-07 morning) | 157s | 143s |
| + `H3_TE_PREQUANT` | 83.2s | — |
| + `H3_TE_DEVICE` | ~35s | 60.5s |
| + `H3_KEEP_TRANSFORMER` (this section) | **9.7s** | **44.2s** |

t2va is **8.0x** the bare configuration (351.4s -> 44.2s). Turbo alone gives 2.6x (denoise
shortens 7.6x, but the ~110s load fixed cost remains); the rest is the contribution of
eliminating fixed costs (`H3_TE_PREQUANT`/`H3_TE_DEVICE`/`H3_KEEP_TRANSFORMER`). Note that
in this section's configuration, non-turbo 30 steps for t2i is 51.1s (measured above) —
against the bare 157s, eliminating fixed costs alone gives a 3x speedup even at the
unchanged 30 steps.

See `docs/RESIDENCY.en.md` §5.5 and §5.6 for the detailed VRAM-budget derivation and the
phase x residency table.

### Keeping text_encoder resident on a separate GPU (`H3_TE_DEVICE`, default ``, 2026-08-09)

The root cause of `H3_LOWVRAM=1` reloading TE on every request is that **there is nowhere
to put TE during denoise** (on 48GB, transformer-int8 34GB + activations 5GB = 39GB,
leaving only 9GB, which doesn't fit TE's 17.45GB). Moving TE to a second GPU eliminates
this reload.

```bash
# Example: keep TE resident on cuda:1 (do not set CUDA_VISIBLE_DEVICES = expose both GPUs)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --port 8611
```

**Measured (RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB, t2i turbo 4 steps, 768²)**:

| | Run #1 | Steady state (run #2 onward) |
|---|---|---|
| Legacy (TE also on cuda:0) | 81.3s | 67.6-86.2s (avg **78.4s**) |
| **TE on cuda:1** | 88.2s (including first-time TE load) | 33.6-36.7s (**about 35s, -55%**) |

TE is loaded **only once, right after process startup**. It stays resident on GPU1 at
17GB, while GPU0 drops to as low as 2.3GB when idle.

- **ref2va is automatically rejected** (in the 20GB-tier case): confirmed on real hardware
  that the activations from running a 2048px-short-side reference through the vision
  tower don't fit and OOM (19.25GB in use, short by 204MB). If the TE GPU has less than
  24GB, this returns 400 with the reason — rather than letting it run and OOM under an
  assumption of "it should work." Automatically allowed if 24GB or more. t2va/fl2va/t2i
  and their respective batches can be used together
- **PCIe width is not an issue**: GPU1 is Gen4 x4 (a card rated for x16 but connected at 4
  lanes), but TE is only loaded once at startup, and the only per-request transfer is
  prompt_embeds at about 42MB (about 6ms)

**Output does not bit-match the legacy configuration** (PNG MD5 differs). The cause is
**rounding due to the architectural difference between sm_120 (Blackwell) and sm_89
(Ada)**; the measured relative RMS difference of `hidden_states[50]` is **0.084%**
(max_abs_diff 3.5, value RMS 57.4). This is **20x smaller** than the difference from the
already-adopted reference-prefix sharing (1.5%), and orders of magnitude away from
bug-level territory (negative control 27-30%). It's the same kind of trajectory drift as
Sage Attention or int8, and equally imperceptible visually. **Remove `H3_TE_DEVICE` for
any A/B test requiring bit reproducibility.**

#### Implementation summary (the `_execution_device` pitfall, again)

`_execution_device` returns the device of the first nn.Module in component order
(`text_encoder, tokenizer, processor, vae, ...`) (confirmed by reading the
implementation). **Placing TE on cuda:1 makes this return cuda:1**, and layout/latents/
timesteps as well as decode all build tensors on the wrong GPU as a result.

The first attempt was to "only close the window around layout," but **it also fired during
decode** (`latents = latents * latents_std + latents_mean` failed with `Expected all
tensors to be on the same device, cuda:0 and cuda:1`, reproduced on real hardware). Since
closing windows individually is leaky, switched to the inverted design of **"detach TE
from the pipeline by default, and attach it only while encoding"**
(`_te_attached()`). This stays safe even as new code paths are added. The module instance
itself is kept alive by `self._te_module`, so detaching it doesn't free it.

During the layout/latents/timesteps window, `_pin_execution_device_to_compute()` also
temporarily detaches vae (since, after TE is detached, the next nn.Module found would be
the CPU-resident vae, making `_execution_device` return cpu instead — this ensures
transformer is found first).

### fp16-ifying the video VAE (`H3_VIDEO_VAE_FP16`, default `0`)

With `H3_VIDEO_VAE_FP16=1`, only the video VAE weights are fp16-ified (9.70->4.85GB,
decode peak 16.29->~11.4GB). **Never touch the audio VAE** (known -20dB issue with
bf16-ification).
- Quality: mean PSNR across all 124 frames is **39.97dB** (min 39.08), visually
  indistinguishable. Since decode computation already used autocast fp16, the impact of
  fp16-ifying the weights themselves is small
- **Implementation pitfall**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` forces the
  encoder/decoder etc. back to fp32, so `from_pretrained(dtype=fp16)` has no effect
  (confirmed on real hardware). You must explicitly call `.to(torch.float16)` after loading
- With default OFF, this MD5-matches the existing baseline (zero regression)

## text_encoder

### Removing unused upper layers of text_encoder (`H3_TE_PRUNE`, default `0`, added 2026-08-06)

MiniMax-H3's text_encoder (Qwen3-VL-32B, 64 layers) only reads
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]` (=50)
(`diffusers/modular_pipelines/minimax_h3/encoders.py`). `H3_TE_PRUNE=1` builds the
text_encoder with **only 51 layers** (0-50, `MINIMAX_H3_TEXT_ENCODER_LAYER + 1`), and
never loads the unused layers 52-64 + final `norm` + `lm_head` (equivalent to 14 layers'
worth of weight; measured ~3.6GB with bnb-4bit / ~13.6GB with bf16). The default `0` leaves
everything completely unchanged (this branch is never invoked).

#### Why "51 layers," not exactly 50 (a transformers-side pitfall discovered and verified in this task)

The meaning of `hidden_states[k]` is determined by transformers' `_can_record_outputs =
{"hidden_states": Qwen3VLTextDecoderLayer}` hook mechanism (`output_capturing.py`):
`hidden_states[0]` = the embedding output (captures the input to layer 0),
`hidden_states[k]` (k=1..num_hidden_layers) = the output of `layers[k-1]`. In other words,
`hidden_states[50]` = the output of `layers[49]`, so it should have been sufficient to
just run `layers[0..49]` (50 layers). However, when `num_hidden_layers` is truncated to
**exactly 50**, `hidden_states[50]` becomes the **last element** of the captured tuple,
which triggers the behavior of `@capture_outputs(tie_last_hidden_states=True)` (the
default) wrapping `Qwen3VLTextModel.forward` — it forcibly overwrites the last element
with `outputs.last_hidden_state` (the value after the final `norm` is applied).
Real-hardware verification (`scripts/probe_te_prune*.py`) confirmed that when truncated to
exactly 50 layers, `hidden_states[50]` is **orders of magnitude different** from the true
(64-layer model's) value (max abs diff ~1.5e4 — not the level of quantization error, but a
completely different value). This is exactly the situation warned about by `encoders.py`'s
own guard (`if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER: raise ValueError(...)`) —
"the final hidden state when truncated to exactly 50 layers is post-norm, not the value
MiniMax-H3 expects" (thanks to this guard, a mis-configuration of exactly 50 layers is
rejected with an exception via `encode_prompt()`). Using **51 layers** (where `layers[50]`
is executed but its output is never read — the only cost being one wasted layer's worth of
compute) puts `hidden_states[50]` in the middle of the captured tuple, avoiding the
overwrite. Confirmed on real hardware that the 51-layer version's `hidden_states[50]`
**exactly matches** (`torch.equal`, for both bf16 and bnb-4bit nf4) the 64-layer version's.

#### Implementation

`core/runner.py`'s `_text_encoder_config_kwargs()` separately loads a `Qwen3VLConfig` from
the same location as text_encoder's `ComponentSpec`
(`pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3"`, `subfolder="text_encoder"`), and
passes an object with `text_config.num_hidden_layers = 51` rewritten in as
`load_components(..., config={"text_encoder": pruned_config})`. `PreTrainedModel.
from_pretrained` skips its own automatic config loading and uses the given object as-is
when `config` is already a `PreTrainedConfig` instance (confirmed in `modeling_utils.py`).
The checkpoint's `layers.51-63.*` show up in `from_pretrained`'s load report as
`UNEXPECTED` and are simply ignored (since they are never constructed, they consume no
VRAM/RAM at all). The vision tower (`model.visual`) is left unchanged (since fl2va's
keyframe / ref2va's reference image and video pixel_values pass through it — explicitly
excluded from removal).

Composes with any combination of `H3_TE_QUANT` (bnb-4bit/none) and `H3_LOWVRAM`
(0/1/group).

#### Measured size of the pruned TE

| Precision | Before pruning | After pruning (51 layers) | Reduction |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

Because nf4 quantization compresses per-layer size to roughly 1/4 of bf16, the absolute
reduction amount is also smaller than bf16 (the relative reduction ratio is roughly the
same).

#### Result of the MD5-match check

Comparing the output of `H3_TE_PRUNE=0` (no pruning) against `H3_TE_PRUNE=1` (pruned),
with the same seed (768², 5s, 30 steps, fox prompt, seed=12345, using `H3_CACHE=none` to
eliminate FBC's path dependence), confirmed **byte-identical mp4 (md5 match) for both
t2va and ref2va (1 image reference, via the vision tower)**. Empirical proof that the
pruning is mathematically inconsequential. Also confirmed that outputs match exactly
regardless of pruning in `H3_LOWVRAM=1` and `H3_LOWVRAM=group` modes as well (described
below).

#### 24GB-tier support: pruning alone wasn't enough (discovered on real hardware, additional fix on the `H3_LOWVRAM_GROUP` side)

The existing 24-32GB-tier mechanism (`H3_LOWVRAM=group`) was designed to keep TE-nf4
(21GB before pruning) resident even during denoise (since group-offloaded transformer's
actual footprint is small at ~1.4GB, this was not a problem at the 32GB tier). The pruned
TE (17.45GB) is still large, and **reproduced an OOM on real hardware at the 22GB limit**
(right at the start of denoise, failing to allocate 224MB with 21.73GB already in use).
Same story at the 24GB limit (also OOM, failing to allocate 1.16GB with 23.12GB in use,
occurring at step 1).

As a countermeasure, added the same "force-free TE only during the denoise loop and
reload it around the decode window" selection method as `H3_LOWVRAM=1`, but only when both
`H3_LOWVRAM_GROUP` and `H3_TE_PRUNE=1` are set (the `group_free_te_for_denoise` flag in
`core/runner.py`). The free point is placed **after** layout_step/latents_step/
timesteps_step (same reasoning as the existing `force_free_te`: since the outputs of
these steps are already materialized as tensors in `state`, they have no further influence
on `_execution_device` resolution from that point on). With `H3_TE_PRUNE=0` (default),
`H3_LOWVRAM_GROUP` is completely unchanged (this flag is only true when both
`H3_LOWVRAM_GROUP` and `H3_TE_PRUNE` are true).

After the fix, confirmed successful completion on real hardware at all of 22GB/24GB/20GB
VRAM limits:

| VRAM limit | Result | Peak VRAM (measured after reset) | Total time |
|---|---|---|---|
| 22GB (before fix, pruning only) | **OOM** (right at start of denoise, failing to allocate 224MB with 21.73GB in use) | - | - |
| 24GB (before fix, pruning only) | **OOM** (step 1, failing to allocate 1.16GB with 23.12GB in use) | - | - |
| 24GB (after fix, run #1) | Yes | 17.72GB | 321.7s (including first-time TE load) |
| 24GB (after fix, run #2, consecutive) | Yes | 18.68GB | 277.7s (shorter since TE stays resident) |
| 20GB (after fix) | Yes | 17.72GB | 320.3s |

The output mp4s across the 24GB x2 runs and 20GB x1 run were **all byte-identical** (md5
match, also matching the output of normal int8 mode (`H3_LOWVRAM` unset)). This
substantiates that group offload's computation is mathematically identical regardless of
VRAM budget (the same conclusion as the existing 32GB/24GB verification results). No
increase in host RAM/swap before/after any test (`free -h`'s Swap used remained
consistently at ~7.9GB throughout, an existing baseline).

#### Shortened TE load time under `H3_LOWVRAM=1` (48GB tier)

Pruning shortens the per-request TE-load fixed cost that `H3_LOWVRAM=1` pays every time
(real hardware, free VRAM constrained to ~43.5GB via dummy VRAM allocation):

| | TE load time | TE size |
|---|---|---|
| Without pruning | 42.3s | 21.01GB |
| With pruning | **35.0s** (-17%) | 17.44GB |

The output mp4 is byte-identical (md5 match) regardless of pruning.

#### Regression check

Ran t2va with the same conditions with `H3_TE_PRUNE` unset (default `0`), and confirmed it
byte-matches (md5 match) the baseline mp4 captured before this feature was added. Default
behavior is completely unchanged.

### Disk cache of the quantized text_encoder (`H3_TE_PREQUANT`, default `1`, 2026-08-08)

`H3_LOWVRAM=1` reloads TE on every request, and that time becomes a straight fixed cost.
Most of that time is spent "reading the original bf16 weights + quantizing them to
bnb-4bit on the fly," so **saving the already-quantized weights once means only having to
read them from then on**. Saved automatically on first load, and read from there afterward.

**Measured (RTX PRO 5000 48GB + `H3_LOWVRAM=1 H3_TE_PRUNE=1`, t2i turbo 4 steps, 4 runs
each)**:

| | TE load | Request total |
|---|---|---|
| Cache disabled (legacy) | 46.5-55.8s (avg **53.0s**) | 108.7-150.0s (avg **128.6s**) |
| **Cache enabled (default)** | 21.1-34.3s (avg **29.5s**) | 75.3-90.6s (avg **83.2s**) |
| Difference | **-23.5s (1.8x)** | **-45.4s (-35%)** |

**Equivalence**: outputs for the same seed are **byte-identical PNG MD5** (confirmed for
seeds 3 and 4). In addition, the probe (`scripts/probe_prequant_equivalence.py`) confirmed
`hidden_states[50]` is bit-identical via `torch.equal` (max_abs_diff 0.0). Since bnb-4bit
quantization is deterministic, this is the expected result, but this project's convention
is to not adopt something just because it's faster.

- Cache location: `models/prequant/te_<quant>_prune<0|1>/` (already in .gitignore).
  **A separate directory per setting combination** — since changing TE_QUANT / TE_PRUNE
  changes the weight contents, reusing the same location would risk reading stale weights
  after a setting switch
- Disk footprint: **17.44GB** (`H3_TE_PRUNE=1`) / ~21GB if unpruned
- `H3_TE_PREQUANT=0` fully disables this (identical behavior to before this feature). If
  free disk space falls below `H3_TE_PREQUANT_MIN_FREE_GB` (default 25GB), **saving is
  skipped and generation continues** (the cache is a speedup, not a required feature).
  Saving writes to a temp directory then renames, so a mid-write crash can never leave a
  half-written cache looking "valid"

**A measurement pitfall hit during this verification**: measuring right after saving
shows an unrealistic value of **2.6 seconds**, since the 17.44GB is still in the page
cache. In practice, reading the transformer (34GB) from disk between requests evicts TE's
page cache, so real-world measurements settle around 21-34 seconds (getting slower with
each additional request, plateauing around 34s). **Still faster than the legacy path's
46-56 seconds**, but the probe's raw numbers should not be reported as-is as real
performance.

### 2026-08-10: Projected TE (Qwen3-VL-4B + a trained linear map) — implementation, measurements, unverified items

Implemented as `H3_TE_PROJ` (default OFF; existing behavior does not change by a single
byte). It replaces the TE path from Qwen3-VL-32B with **Qwen3-VL-4B + a trained projection
matrix** ([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)). The
aim is to keep the TE and the transformer **simultaneously resident** on the 48GB box,
eliminating the swap fixed cost.

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = the 4B's hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 is the attention sink
```

#### Measurements (96GB box, t2i 768², 30 steps, seed 4242, identical prompt)

| | 32B TE (current) | **Projected 4B TE** |
|---|---|---|
| PSNR | — | **22.64 dB** |
| Sharpness (Laplacian variance) | 48 | **37** (-23%) |
| Generation time | 65.7s | **43.7s** |
| Peak VRAM | 88.73GB | **76.1GB** |
| **TE's real GPU occupancy** | 21.02GB (nf4) | **8.88GB** |

**The quality is not "the same picture" but "a different picture of comparable quality".**
Composition, palette and the reading of the time of day match, but fine-grained
specification drops (a real example, outside what the prompt asked for: the foreground
water-lily leaves and grass present in the 32B version were replaced by open water in the
projected version). No breakdown; practical quality. This matches the nature of the
projection (the distributor reports test cosine 0.712): read it as **the gist of the prompt
is preserved, the details are lost**.

**This change is different in kind from every optimization so far.** Quantization,
residency control, turbo etc. could all be proven "mathematically inert" by identical-seed
MD5 match, but the projected TE is **an approximation by principle**, so MD5 is unusable.
The verdict rests on PSNR + visual inspection + measured VRAM.

#### Tokenizer investigation (the basis of the implementation policy)

| Target | Result |
|---|---|
| Ordinary text, Japanese | **IDs match exactly** (the H3-side tokenizer can be used as-is) |
| `<d>` / `</d>` | H3 has them as **single tokens** 151669/151670; the 4B lacks them and splits them apart |
| `<\|cutoff\|>` `<\|lyrics_*\|>` `<\|caption_*\|>` | H3-specific (151671-675) but **unused by both the official guide and this app** |
| `[Shot n]` `<cutoff>` `<scenetrans>` | **Ordinary text** (not special tokens). Multi-shot structure is unaffected |

→ The only thing affected in real use is **the dialogue tag `<d>`**. The implementation
**explicitly rejects** prompts containing id >= 151669 with a `ValueError` (never silently
sends something different). Dialogue can be fed via an audio reference (`fully_copy`), as
demonstrated the same day (see the section above), so an operational workaround exists.

#### The second-GPU requirement drops

Plugging the measured 8.88GB back into the derivation in `docs/RESIDENCY.md` §5.4, the card
requirement for `H3_TE_DEVICE` drops from **20GB to 16GB (probably 12GB)**.

| 2nd GPU | Effective budget | 32B TE (17.45GB) | Projected 4B TE (8.88GB) |
|---|---|---|---|
| 12GB | ~10.5GB | No | **Expected to hold** (needs ~9.2GB, ~1.3GB headroom) |
| 16GB | ~14.5GB | No | **Expected to hold** (~5.3GB headroom) |
| 20GB | ~19.7GB | t2va only; ref2va OOMs | Holds (~10.5GB headroom) |

**12GB is not asserted without a real-hardware check.** We previously experienced "fits by
derivation, OOMs in practice" at 20GB (the units pitfall, §5.1), and 1.3GB of headroom is
thin.

#### Addendum (same day): the NF4 quantization option, and quality confirmation on video

Added **`H3_TE_PROJ_QUANT`** (default `none` / `bnb-4bit` / `bnb-8bit`). It quantizes the
4B itself — a **4B-only flag**, distinct from the 32B's `H3_TE_QUANT` (see the intent of the
exclusivity guard).

**Using the bf16 projection matrix as-is is the right call** (measured). The conditioning
drift from quantization is 0.61-0.96% relative RMS under NF4 (cosine 1.0000). Using the
distributor's int8_convrot matrix actually increases the drift (1.02-2.97%) — that one is
calibrated specifically for ComfyUI's quantization scheme.

| Config | TE resident | t2i (30 steps) | t2va 5s (30 steps) | Peak (t2va) |
|---|---|---|---|---|
| 32B TE | 21.02GB | 65.7s | 162.1s | 91.9GB |
| Projected 4B bf16 | 8.88GB | 43.7s | — | — |
| **Projected 4B NF4** | **3.11GB** | **33.5s** | **143.5s** | **74.3GB** |

**Quality confirmed on video too** (768², 5s, 124 frames, identical seed 555): the
sharpness drop seen on stills (-23%) does not appear on video (187 vs 191), and **flicker
(second-order difference) is actually 11% lower** (7.51 → 6.67). Direct bf16 vs NF4
comparison is PSNR 34.45dB — quantization has essentially zero effect. The PSNR 14.98dB
against the 32B is not "degradation" but "a different take on the same instruction" (the
32B version has an approaching passer-by, the projected version has the girl alone — a
divergence of interpretation).

**Relation to the settings API**: the projected TE is currently **env-only**
(`H3_TE_PROJ`/`H3_TE_PROJ_QUANT`). Attempting to change te_quant/te_prune via
`/api/settings/apply` slipped past the runner's import-time guard and had a hole where
**nothing was applied yet the snapshot reported a change**, so with the projected TE
enabled these fields are now rejected with 400 (read-only `te_proj`/`te_proj_quant` were
added to the `/api/settings` snapshot).

#### Addendum (same day, part 2): NF4 made the default, switchable from the UI

- **Changed the default of `H3_TE_PROJ_QUANT` to `bnb-4bit`** (based on the measured
  quality; none/bnb-8bit remain selectable explicitly). Along with the default change, the
  "quantization specified + projection OFF" guard now only fires **when explicitly set
  (`in os.environ`)** — so ordinary users on the defaults are not knocked over by mistake
- **The projected TE can now be toggled from the UI's reload-settings panel**
  (`te_proj` / `te_proj_quant` on `/api/settings/apply`). While ON, the te_quant/te_prune
  controls are disabled and the API also rejects them with 400 (the check is based on the
  "post-apply values": turning it OFF while changing te_quant is legal)

**Round-trip reload E2E (measured, 96GB box)**: OFF→ON 47.9s / ON→OFF 29.4s / re-ON 27.3s.
At each step the PNG MD5 of an identical-seed generation was compared: **the UI-path ON
matches the env-path NF4 exactly**, **OFF matches the 32B baseline exactly**, and **re-ON
matches the first ON exactly** (no state residue; the projection matrix reloads correctly).
Changing te_quant while ON returns 400.

#### Unverified (important)

- **TE + transformer simultaneously resident on the 48GB box** — **the main prize**.
  8.88 + 34 (int8) = 42.9GB is expected to fit but is unmeasured. Only once this holds does
  the adoption pay off (on the 96GB box both fit anyway, so no difference shows)
- **The reference path (ref2va)** — the projection matrix was **calibrated on text only**,
  and whether vision features map correctly was unknown. The implementation is wired to
  work, but emits a one-time `logger.warning`
  → **Visually verified on 2026-08-11: references clearly take effect** (face, hair,
  clothes, accessories all reflected, on par with the 32B ground truth; see the section
  "2026-08-11: How far do ref2va/i2va go"). The warning is kept as a record of the
  calibration fact
- **Combination with an audio reference + `fully_copy`** (does the feed-dialogue-from-audio
  workflow still hold with the projected TE)
- **The 4B's TE load of 80.5s** (including the first download). Steady-state load time is
  unmeasured

## transformer and denoising

### transformer int8 quantization (`H3_TRANSFORMER_QUANT`, default `none`)

With `H3_TRANSFORMER_QUANT=int8`, the transformer / transformer_ref is int8-ified via
torchao (`Int8WeightOnlyConfig(version=2)`, the recipe from the PR #14355 docs, torchao
0.17.0). 66.3GB -> 34.0GB. Quality is visually equivalent (the PSNR of 19dB is trajectory
divergence, not degradation; int8-vs-int8 is fully deterministic, giving byte-identical
mp4 for the same seed), and denoise is about +5s (dequant cost).

**With int8, both transformers stay resident simultaneously** (34+34+TE-nf4 21 = ~89GB),
eliminating the ~66GB-class reload on ref2va<->t2va variant switching (only the first
ref2va incurs a ~36s cold load). Measured:

| | bf16 (default) | int8 both resident |
|---|---|---|
| t2va | 175-185s / peak 92.1GB | 177-196s / peak 59.7-91.1GB |
| ref2va (2nd request onward) | 523s / peak 87.6GB | **463-471s / peak 74.5GB** |
| Variant-switch reload | ~26-40s every time | **none** |

Phase control (with int8 both resident): before t2va denoise, TE is force-freed only when
transformer_ref is resident (because it would OOM at 89GB steady + activations — confirmed
on real hardware); ref2va no longer needs the TE force-free, and the decode window also
passes through while transformer_ref remains resident. The runner sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (because "only 54GB used but a 15GB
allocation failed" was reproduced on real hardware from fragmentation during the int8
load/free cycle — a setting with a track record in diffusers-server as well). The default
`none` remains byte-identical to before (regression confirmed).

### Denoise speedup via FirstBlockCache (`H3_CACHE`, default `fbc`)

Enables diffusers' official step-to-step cache (FirstBlockCache), equivalent to the
ComfyUI community's EasyCache speedup, via `H3_CACHE=fbc` (default). It skips the
remaining computation when the residual change of the first transformer block between
steps is small.

- `H3_CACHE_THRESHOLD` (default 0.05): measured to reduce denoise from 157s to 118s
  (-25%, 7 of 30 steps skipped); output is nearly identical to the non-cached version
  (PSNR 31.8-34.3dB, audio correlation 0.979, hard to distinguish visually).
- threshold 0.1 gives 1.92x (denoise 81.5s, 14 skips) but causes visually noticeable
  compositional drift, so it is not the default (only for when speed is the top priority).
- `H3_CACHE=none` fully restores the traditional no-cache behavior (byte-match regression
  confirmed).
- Peak VRAM increases by +0.7GB for the residual cache (91.4->92.1GB).
- Implementation note: `MiniMaxH3TransformerBlock` is not registered in the PR branch's
  `TransformerBlockRegistry`, so the runner side registers `TransformerBlockMetadata`
  before calling `enable_cache()` (the venv's diffusers itself is unmodified). Each
  request is wrapped with `_reset_stateful_cache()` + `cache_context("h3")` (failing to
  reset causes mis-skips based on the previous request's residuals). Verified reset
  correctness via byte-identical mp4s across two consecutive same-seed runs.

### Sage Attention (`H3_ATTN_BACKEND`, default `sage`)

Uses SageAttention 2.2.0, source-built for sm_120 (Blackwell), by default (build via
`scripts/build_sageattention.sh`, about 2 minutes. **Always run it with
`MAX_JOBS=4 NVCC_THREADS=2` plus a systemd-run memory cap** — unrestricted parallel nvcc
has a history of exhausting host RAM and taking down the whole system. Requires explicit
`CUDA_HOME=/usr/local/cuda-12.8`, since the default cuda-13.0 does not match torch's
cu128). No PyPI/community Linux sm_120 wheel existed (all were Windows-only).

- Measured: denoise 118s->104s (**-12%**). Fully deterministic (byte-identical across
  two same-seed runs). Quality is visually equivalent (the PSNR of 21dB is trajectory
  drift from the int8-QK approximation, not degradation)
- `H3_ATTN_BACKEND=default` reverts to the traditional SDPA
- Can be combined independently with FBC: sage + `H3_CACHE_THRESHOLD=0.1` gives denoise
  67s (-43%, request ~125s; FBC 0.1's compositional drift characteristic is as already
  known)
- The hub backends (`flash_hub`/`sage_hub`) don't work because no torch 2.9-targeted build
  exists on the Hub side (as of 2026-08-05, not an environment issue)

### Turbo LoRA (`H3_TURBO_LORA`, default `0`, 2026-08-06 / **switched to the lightx2v variant on 2026-08-08**)

> **2026-08-08 update: switched the default LoRA to the lightx2v variant, making turbo
> usable even on 48GB (int8/low-VRAM).**
>
> - Default: `H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo` /
>   `H3_TURBO_LORA_FILE=minimax_h3_fl2v_turbo_4step_v0.1.safetensors` (DMD distillation,
>   Apache 2.0, rank128, targets 312 Linear modules). Default step count is tied to the
>   format (lightx2v=4 / falls back to 8 if reverted to Ostris)
> - **Reason it works with int8**: the keys are diffusers-native (to_q/to_k/to_v
>   separated), so `fuse_projections()` is unneeded -> `torch.cat` is never called ->
>   avoids the `Int8Tensor` incompatibility. The apply function auto-detects the format
>   (`detect_turbo_lora_format`; checks the comfy signature `qkv_proj` first — order
>   matters since the Ostris variant also has `token_refiner.` keys)
> - **Applied scale factor** (`H3_TURBO_LORA_SCALE`, empty = measured per-format default):
>   for lightx2v it is **0.094** (Kijai's documented 0.75 assumes ComfyUI's alpha folding.
>   Multiplying the raw B*A by 0.75 directly produces complete noise even at 30 steps —
>   see the strength sweep in the spike section). Ostris remains 1.0 (scale=1.0 is
>   identity, so it bit-matches the old behavior)
> - **E2E measured** (RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768²): t2va 4 steps
>   **total 143s / denoise 29s** (non-turbo 30 steps is 351s). **t2i (still image) x
>   turbo is denoise 5.0s / total 94s**. The output of this implementation's path is a
>   **byte-identical mp4 md5 match** with the spike output. Two consecutive turbo runs
>   also correctly re-apply after lowvram's reload, and turning turbo=0 correctly returns
>   to the normal path (with FBC enabled)
> - **Known behavior**: with turbo, audio level tends to be higher than non-turbo (rms
>   0.018-0.042 vs 0.007; not white noise, but depending on intensity the peak can
>   approach 1.0)
> - **Combination restrictions**: cannot be combined with `H3_LOWVRAM=group` regardless of
>   format (`enable_group_offload`'s `cpu_param_dict` is fixed at enable time, and any LoRA
>   buffer added afterward risks being left out of the offload cycle — left disallowed
>   since this is unverified). comfy-format (Ostris) x int8 remains unsupported as before
>   (repo name is used for a preliminary check, with the final determination made from the
>   actual file's keys at apply time). Application to ref2va (transformer_ref) remains
>   unverified
> - Since this is v0.1 (published 2026-08-07), it remains **default OFF** (opt-in via the
>   UI/request turbo flag)

The following is a record from the era of the Ostris variant (comfy format), which remains
valid for the bf16 path:

Applying the 4/8-step distilled LoRA (Apache 2.0, rank64, targets 259 Linear modules)
being trained by Ostris via `H3_TURBO_LORA_REPO=larryvrh/MiniMax-H3-Turbo-Lora`, and
setting the default step count to 8. **Default OFF since this is a preview LoRA (still
in training)**. Will be re-evaluated once a finished version is released.

Measured (768²/5s/seed12345): 8 steps gives **87.7s (-46%)**, approaching the quality of
the 30-step baseline (163.5s); 16 steps at 98.4s matches the baseline; 4 steps at 39.6s is
softer but not broken.
**The community's "4-7 steps doesn't work" report is likely because ComfyUI's standard
sampler cannot handle the dual schedule (video shift12 / audio shift3)** — this
implementation (diffusers PR's scheduler/audio_scheduler separation + manual loop) showed
no audio corruption even at 4 steps. No change to the shift wiring was needed (12/3 are the
H3 baseline scheduler's defaults, and we confirmed the sigma grid bit-matches the author's
reference implementation).

Implementation note: LoRA keys use ComfyUI's fused-QKV naming, so it's applied via
`attn.fuse_projections()` + a runtime delta (W_eff=W+BA, not fused). **Pitfall:
`fuse_projections()` does not remove the old to_q/k/v, leaking +12.8GB** (addressed with
explicit deletes). Since AdaLN reads `linear.weight` directly, the wrapper needs
weight/bias etc. pass-through. FBC is automatically disabled while turbo is active.

#### Verification of turbo x other features combined (2026-08-06)

Combinations with anything other than the default transformer path
(`transformer_quant=none`, `lowvram=0`) were initially "preventively rejected as
unverified," but have since been A/B tested on real hardware.

- **turbo x upscale (2x upscale hires-fix): verified working, unblocked.**
  768²->1536², seed12345, succeeded at both 8/16 steps. 8 steps: total 210.3s
  (denoise 82.6s + decode 24.4s), pass1=5/pass2=2 steps (`H3_HIRES_DENOISE=0.35`
  default split), peak VRAM 88.09GB — a large reduction from the non-turbo 30-step
  baseline of 645s. 16 steps: total 331.8s, pass1=10/pass2=5 steps, peak VRAM 88.35GB,
  clearly sharper than 8 steps. All frames (first/middle/last) were visually confirmed
  free of discoloration or checkerboard collapse, and audio RMS/peak were normal (no
  silence, no clipping). Since FBC is auto-disabled with turbo, hires-fix's FBC
  bookkeeping (`_fbc_last_step_was_skip()`) safely no-ops via try/except (confirming the
  concern of "unhandled FBC calls with turbo unsupported" causes no actual harm).
- **turbo x transformer int8 (`H3_TRANSFORMER_QUANT=int8`, including
  `transformer_both_resident`): confirmed unsupported by measurement, remains rejected.**
  `apply_turbo_lora()`'s `attn.fuse_projections()` executes
  `torch.cat([to_q.weight, to_k.weight, to_v.weight])`, but the int8-quantized
  `to_q`/`to_k`/`to_v` (`H3_INT8_MODULES_TO_NOT_CONVERT` does not skip these) are torchao
  `Int8Tensor`s, and since the `aten.cat` kernel is unimplemented, it reliably fails with
  `NotImplementedError: Int8Tensor dispatch: attempting to run unimplemented operator/
  function: func=<OpOverload(op='aten.cat', overload='default')>` (HTTP 500; fails
  cleanly per request with no VRAM leak — confirmed normal non-turbo generation
  immediately afterward).
- **turbo x lowvram=1: confirmed unsupported by measurement, remains rejected.**
  Since `lowvram=1` forces `transformer_quant=int8`, it fails with exactly the same
  `Int8Tensor`/`aten.cat` error as above (same error message confirmed on real hardware).
- **turbo x lowvram=group: confirmed unsupported by measurement, remains rejected.**
  Same reason (`lowvram=group` also implies `transformer_quant=int8`) yields the same
  error. **This failure is unrelated to the order in which group-offload hooks are
  applied** (the sibling project's fix pattern of "apply LoRA before
  `enable_group_offload()`" doesn't help here — `fuse_projections()` itself fails purely
  from `torch.cat`, without going through the group-offload hooks at all, so reordering
  wouldn't fix it and this was not pursued further).
- **ref2va: remains out of scope for this task's verification** (out of scope for the
  original task brief).

### Step-count guidance (distilled model, measured 2026-08-05)

`num_inference_steps` is an API/UI parameter. Relative to 30 (the verification default),
**20 gives -15%, 16 gives -31%** in denoise time. Both 16 and 20 show no breakdown in
single-frame quality or temporal stability (though composition does change with step
count). 16-20 is a reasonable guideline for drafts, 30 for production. Reducing steps also
reduces FBC skip opportunities (7 skips at 30 steps -> 0 at 16 steps), so the effect is not
simply proportional.

### 2x upscale via two-pass generation (`upscale=1` on `/api/t2va`, default OFF)

A hires-fix in the same family as the ComfyUI community's MiniMaxH3_LatentUpscaler.
Denoise the first half at low resolution (768²) -> spatially upscale only the **x0
estimate** video latent 2x via bilinear -> re-inject fresh noise via
`scheduler.scale_noise()` -> finish the remaining low-sigma steps at 1536² -> decode.
`H3_HIRES_DENOISE` (default 0.35) is the denoise strength assigned to pass 2. The UI
exposes this as a checkbox in the T2VA tab.

Measured (768² -> 1536², 5s, 30 steps, seed=12345, fbc+bnb-4bit):

| | Total | Denoise | Decode | Peak VRAM | Output |
|---|---|---|---|---|---|
| upscale=0 | 181s | 125s | 6.5s | 92.1GB | 768² |
| upscale=1 | 645s | 533s (pass1 78s + pass2 455s) | 24.7s | 88.0GB | 1536² |

- Composition and subject match upscale=0, with actual detail (fur, grass, etc.) added on
  top. Background details (fences etc.) drift slightly due to the re-denoise in pass 2
  (an intrinsic property of hires-fix).
- Audio: the latent tensor itself is unchanged by the upscale processing, but since video
  and audio share self-attention within a single packed sequence, the audio output from
  pass 2 onward does not bit-match upscale=0 (correlation 0.89, non-silent, equivalent
  quality — this is a spec-level architectural constraint).
- VRAM: since pass 2 has roughly 4x the sequence length, requests with upscale=1 free
  TE-nf4 immediately after encoding, before denoise (lazily reloaded on the next request's
  encode).
- **Key implementation point (confirmed by hitting a bug on real hardware)**: the
  interpolation target must be **the x0 estimate, not the noisy latent** (interpolating
  the noisy latent amplifies checkerboard-like noise into full-frame noise; the ComfyUI
  reference implementation also uses denoised_output). When changing resolution,
  position_ids/token_tags/each index is rebuilt via `build_packed_sequence()`, and
  `row_timestep_plan` is also rebuilt for the remaining steps. Because ModularPipeline's
  `_execution_device` is determined by the first module in component registration order,
  after freeing TE you must explicitly use `components.transformer.device`
  (same pattern of pitfall as diffusers-server CLAUDE.md #23 and #47).

### Rounding of arbitrary size/duration (2026-08-06)

Added **"Custom (rounds to a multiple of 32)"** to the UI resolution select, allowing
free-form width/height input. Input values are **rounded to H3's rules rather than erroring**:

- **Canvas**: rounds to the nearest multiple of 32, clamped to 256-2048
  (`round_canvas_value` in `app.py`). H3's block spec raises a `ValueError` unless the
  value is a multiple of 32 (`MINIMAX_H3_CANVAS_MULTIPLE`), so sending a fractional value
  used to return 400. Values exceeding the native range (short side 768, max 768x1344)
  trigger a UI warning (VRAM/quality unverified for these)
- **Seconds -> frame count**: clamped to 5-15s, then rounded up to `17n+5`
  (`align_num_frames`). Example: 6.3s -> 158 frames (6.58s). The UI previews the actual
  frame count and duration before submission
- API: added `height`/`width` (optional; when specified, takes priority over the
  `resolution` preset) to `/api/t2va`/`/api/fl2va`. `/api/ref2va` already accepted these,
  but was changed to round them as well
- `/api/status`'s `constraints` exposes the rounding rules (canvas_multiple/min/max, fps,
  frame_step/offset), and the UI previews using the same rules (the server is the
  authority on rounding)

Measured confirmation: generating with `height=700 width=1000 seconds=6.3` -> response
`704x992 / 158 frames`; the output mp4 also matched 992x704, 158 frames via ffprobe.

**UI implementation pitfall**: attaching `min`/`max`/`step="32"` to the width/height
`<input type=number>` causes HTML5 input validation (stepMismatch/rangeOverflow) to
**treat the pre-rounding fractional input as invalid and block form submission entirely**
(only values that happen to be valid, like 1024x576, get through, while something like
1000x700 does nothing — a confusing symptom). Do not attach constraint attributes to
fields that are meant to be rounded.

## Generation modes

### Ref2VA (omni-reference generation, `/api/ref2va`)

Generates video + audio from ordered reference material (**up to 9 images, up to 3 videos,
up to 3 audio, 12 total**; audio alone is not allowed). The reference order matters, since
it corresponds to in-prompt labels (`<Picture i>` etc.) and the rotary layout. Video
references also have their soundtrack used for conditioning. **When the reference is
exactly one audio clip, the seconds parameter may be omitted** (the audio's length becomes
the generated duration; `seconds=0` in the API). Output canvas is not tied to the
reference; if unspecified, it is 16:9 (1344x768).

- Uses a **dedicated checkpoint `transformer_ref/` (61.7GB; same class/config as
  `transformer`, only the weights differ)**. Since it cannot be co-resident with the t2va/
  fl2va transformer, the runner manages it via variant switching (only the active one
  stays resident; free -> reload) (`active_variant` in `/api/status`). TE-nf4, VAE family,
  and processor are shared between both variants.
- VRAM countermeasures (finalized after hitting 3 real OOMs): load transformer_ref after
  reference VAE encode completes (the reverse order OOMs at 98.5GB); force-free TE-nf4
  before denoise (since the reference rows lengthen the sequence — same pattern as
  hires-fix). Freeing the shared text_encoder must drop the reference from both pipeline
  shells (dropping only one leaves a refcount, and VRAM doesn't come back).
- Measured (768x768 requested -> 1344x768 output, 30 steps, seed=12345): 1 image
  reference 523s/87.6GB; image+audio (duration derived from audio, 7.3s) 753s/88.1GB;
  2 images 635s/88.1GB. Visually confirmed subject identity of the reference and
  compositing of multiple references (a person seated in the reference scene's cafe).
  Round-trip switching between ref2va and t2va also works normally (t2va including the
  switch: 188s).
- The UI has a "Ref2VA (reference -> video)" tab (multi-file selection, selection order =
  reference order).

### Still-image mode (T2I, `/api/t2i`, production implementation 2026-08-07)

A mode that uses H3's "ultra-short video -> extract still frame" as a substitute for image
generation. This is the production implementation of the conclusion from the earlier
ultra-short-clip probe (below). The value proposition is being able to produce **still
images whose art style exactly matches H3's**, for use as FL2VA's first frame or Ref2VA's
reference image (speed cannot beat a dedicated T2I model, since it's dominated by fixed
costs).

- **API**: `POST /api/t2i` — `prompt` / `frames` (default 22, or 5) / `resolution` or
  `height`+`width` / `num_inference_steps` / `seed` / instant-apply parameters
  (cache/cache_threshold/attn/turbo). Response includes both an ultra-short mp4 and the
  **center frame's PNG** (`image_url`, `t2i_<ts>.png`)
- **UI**: the "T2I" tab in the still-image row (seconds/upscale fields hidden, only frame
  count selectable). PNGs are also tiled in the gallery (deletable, not concatenable)
- **Frame count**: 22 (0.917s) is the default. 5 (0.208s) is experimental but works thanks
  to the VAE fix below
- **The three-part implementation** (`core/runner.py`):
  1. `generate(still=True, still_frames=...)` — diffusers' minimum-duration-5-seconds
     validation (`MINIMAX_H3_MIN_DURATION`) is relaxed via `_relaxed_min_duration()` for
     exactly one call of `MiniMaxH3SetupStep` (safe since generation is serialized by
     generation_lock, so no leakage)
  2. **VAE small-clip decode fix** (`H3_VAE_SMALLCLIP_FIX`, default 1):
     `AutoencoderKLMiniMaxH3._decode()` has an upstream bug where a latent of 1-2 frames
     yields num_chunks=0, which crashes on `torch.cat([])` (reproduced on real hardware
     with the probe). Added a branch via monkeypatch on the runner side: "if less than one
     chunk, decode the entire token set via a single `_decode_clip()` call, and trim
     `frame_pre_padding` (3) and the trailing padding." A latent of 2 frames x4 - 3 = 5
     pixel frames results in matching geometry. Normal video (num_chunks>=1) is delegated
     entirely to the original implementation, so this has no effect there (the venv's
     diffusers itself is unmodified)
  3. **Restoring steady state on decode exceptions**: as a countermeasure against the
     "decode exception -> transformer already dropped -> restore code never runs ->
     ~98.5GB stays leaked -> subsequent requests OOM in a chain" behavior observed with
     the probe, wrapped the decode portion of `generate()`/`generate_ref2va()` in
     try/except, so that `_restore_decode_steady_state()` (moves VAE to CPU, reloads
     transformer/TE per mode) runs before re-raising even on exception. **Verified on real
     hardware** (2026-08-07): using the one-shot failure-injection hook
     `H3_DEBUG_FAIL_DECODE=1` (read via `os.environ.pop`, so normal behavior resumes from
     the next request in the same process) to deliberately fail decode, confirmed that
     after the 500 response the GPU is freed down to ~2GB, and **the next request in the
     same process returns 200 normally**

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768², 30 steps, fbc+sage,
seed 12345)**:

| frames | Denoise | Decode | Peak VRAM | Total (including per-request reload) | Quality (visual) |
|---|---|---|---|---|---|
| 22 (0.917s) | 29.0s (1.0s/step) | 1.8s | 35.0GB | 157s | No breakdown, high quality |
| 5 (0.208s) | 9.1s | 0.67s | 35.2GB | 125s | No breakdown (first-ever successful real decode) |

The 5-frame audio (0.208s) is also non-silent (rms 0.055).

#### Measured result for consecutive generation (a sequence of narrative scene images with different prompts): **group mode is worse for still images**

Measured the hypothesis that "resident mode (`H3_LOWVRAM=group`) should be faster for
consecutive generation since the reload fixed cost disappears," via 3 consecutive runs of
3 narrative-scene prompts (frames=22, 768², 30 steps, fixed seed)
(2026-08-07, RTX PRO 5000 48GB). **The result was the opposite: about 1.5x slower than
lowvram=1**:

| Mode | 1st image | Steady state (2nd onward) | Steady-state breakdown |
|---|---|---|---|
| `H3_LOWVRAM=1` | 157s | **~157s** | TE+transformer load ~120s + denoise 29s + decode 2s |
| `H3_LOWVRAM=group` | 310s (including cold TE) | **~240s** | denoise **130s** + decode 2s + post-decode TE reload **75-97s** |

- **The denoise degradation from 29s to 130s is the core issue**: group offload transfers
  the entire ~34GB of int8 transformer over PCIe, block by block, every single step. This
  transfer cost is fixed regardless of sequence length, so it completely dominates when
  computation is light (an ultra-short 22-frame clip) (with a normal 124-frame video, this
  is hidden within the heavier computation — the design assumption that breaks down for
  still images)
- Group mode force-frees TE during the decode window and reloads it afterward (headroom
  design for the 24-32GB tier). On this box, the NF4 TE load measures 75-97s (much slower
  than the 15-40s of the 96GB era), so this reload also lands on the steady-state cost
- Quality is as good as lowvram=1 (visually). Peak VRAM is 25.1GB
- **Conclusion**: on this box, consecutive still-image generation is **fastest kept on
  `H3_LOWVRAM=1` (~157s/image)**. Further reduction comes from `/api/t2i_batch` below
  (implemented as a result of this measurement)

#### Batch generation of still images (`/api/t2i_batch`, production implementation 2026-08-07)

Building on the measurement above, implemented an endpoint that **amortizes the fixed
cost of `H3_LOWVRAM=1`** (~110s: TE load 75-97s + transformer load ~20-35s) **once across
the whole batch** (`generate_still_batch()` in `core/runner.py`). Reorders `generate()`'s
lowvram choreography by phase:

    entry   : [nothing big resident]
    encode  : [TE-nf4]         setup/encode/layout/latents/timesteps for all scenes
    denoise : [transformer]    denoise all scenes in sequence
    decode  : [vae pair]       decode all scenes in sequence -> save PNG/mp4 (saved
                               per scene as it progresses, so a mid-batch failure
                               still leaves completed scenes intact)

- **API**: `POST /api/t2i_batch` — send `prompts` repeated via multipart for the number
  of scenes (up to 24 scenes). `frames`/`resolution`/`num_inference_steps`/`seed`/
  instant-apply parameters are common to all scenes (only the prompt can vary). The UI's
  "Batch continuous generation (1 line = 1 scene)" checkbox on the T2I tab sends each
  line of the prompt field as one scene
- **Resetting shared state between scenes** (the key implementation detail): since the
  scheduler's sigmas/timesteps values are identical across all scenes (same geometry, same
  step count), it suffices to reset `_step_index = None` right before denoise
  (`MiniMaxH3Scheduler.step()` re-derives the index from the timestep value).
  FirstBlockCache does `_reset_stateful_cache()` + `cache_context` per scene (a per-scene
  version of `generate()`'s per-request reset)
- **Proof of equivalence**: the sequential `/api/t2i` and the batch's scene 1, with the
  same prompt and seed, produce **byte-identical mp4 and PNG** (md5 match). Phase
  reordering is mathematically inconsequential (note: this does not match group mode's
  output — a known effect of FBC's decisions changing with execution path, not something
  specific to batching)
- Modes other than `H3_LOWVRAM=1` (where the big models stay resident) gain nothing from
  phase reordering, so they fall back to sequential `generate()` calls under the same API
  (identical response format)

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 3 scenes, frames=22, 768²,
30 steps)**:

| Method | Total | Per image | Breakdown |
|---|---|---|---|
| Sequential `/api/t2i` x3 | ~471s | **157s** | reload ~120s + denoise 29s + decode 2s, every time |
| `/api/t2i_batch` (3 scenes) | **202.6s** | **67.5s** | load ~110s (once) + encode 0.9s + denoise 84.8s + decode 6.9s |

Since the marginal cost of an additional scene is **~31s/image** (denoise 28-29s + decode
2.3s), the per-image cost asymptotically approaches 31s as the scene count grows (computed
as ~42s/image at 10 scenes, ~36s/image at 24 scenes). Peak VRAM is 35.0GB, same as
single-image generation (only tens of MB grow per scene, for latents+prompt_embeds).

#### Reference-conditioned still images (ref2i, `still=1` on `/api/ref2va` and `/api/ref2i_batch`, production implementation 2026-08-07)

Implemented in production following the success of the spike below. A mode for producing
**scene-by-scene stills with a consistent character** from a character reference. The
resulting stills can be fed back in as FL2VA's first frame or the next Ref2VA's reference
(building material for generating multiple narrative videos).

- **Single**: pass `still=1` + `frames` (default 22, or 5) as extra parameters to
  `POST /api/ref2va`. `seconds` is ignored, and the center frame's PNG
  (`ref2i_<ts>.png`) is attached to the response's `image_url`. Implementation reuses the
  same three-part set as t2i (duration-gate relaxation only during the setup step, the VAE
  small-clip fix, and the existing decode-exception cleanup)
- **Batch**: `POST /api/ref2i_batch` — common `references` + `prompts` (up to 24 scenes).
  With `H3_LOWVRAM=1`, uses `generate_ref2i_batch()` (the ref2va version of t2i_batch's
  same phase reordering: encode all scenes with TE resident -> a single VAE window to
  encode the reference for all scenes -> layout/timesteps -> a single transformer_ref
  load, denoising everything -> decode everything together). Other modes fall back to
  sequential calls. The between-scene reset of scheduler/FBC uses the same technique
  already proven to md5-match in t2i_batch
- **UI**: the "Ref2I" tab in the still-image row (moved on 2026-08-09 from a "still image"
  checkbox inside the Ref2VA tab to its own independent tab; see "Two-row tab layout"
  below) + "Batch continuous generation (1 line = 1 scene)" checkbox

**Measured (2026-08-07, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 1 Little Red Riding Hood
reference, 3 scenes, frames=22, 768², 30 steps)**: total 494.7s = **164.9s/image**
(1.6x faster than sequential's ~265s/image). Breakdown: encode phase 212.5s + denoise
178.1s (57.7-60.2s/scene) + decode 6.6s. Peak VRAM 36.8GB. Quality and character
consistency both good (all 3 scenes kept the red cloak and followed each prompt).

- ~~Known room for improvement: reference vision encoding runs per scene~~ ->
  **resolved by "Shared KV cache for the reference prefix" below (2026-08-08)**

#### Shared KV cache for the reference prefix (`H3_REF_PREFIX_CACHE`, default 1, production implementation 2026-08-08)

Resolves the problem, in the encode phase of ref batches (`generate_ref_batch` =
ref2i_batch/ref2va_batch), where the Qwen3-VL encoding of the reference labels+vision
(~4104 tokens, ~65s/scene) was duplicated per scene. In ref2va's token sequence, "the
reference is prepended, and the prompt is appended verbatim at the end" (confirmed via
`build_ref2va_presentation` in packing_ref2va), and since the conditioning source Qwen3-VL
is a causal LM, **the text representation of the reference prefix does not depend on the
prompt** — encode the prefix once with `use_cache=True`, bake it into a `DynamicCache`,
and for each scene continue the cache with only the prompt's tail (14-33 tokens, ~0.2s)
(`_encode_ref_prompts_shared_prefix()` in `core/runner.py`).

**Verification** (`scripts/probe_ref_prefix_cache.py`, based on real-hardware measurement
after reading transformers 5.14.1's modeling_qwen3_vl.py):

- The prefix portion's hidden_states[50] is **bit-identical** (`torch.equal`) to the full
  computation
- The prompt-tail portion retains a relative RMS rounding difference of ~1.5%. The cause
  is that the kernel/GEMM's tiling path changes with sequence length — this holds even
  with eager fixed (not sdpa-specific). **Confirmed not a logic bug via a negative
  control**: a continuation deliberately corrupted with a broken position offset jumps to
  27-30% relative RMS (20x). 1.5% is at the level of "rounding noise from a correct
  computation"
- Recipe for the continuation call (3 pitfalls): `attention_mask=None` (rides on
  `compute_3d_position_ids`'s arange branch = sequential numbering from the past length +
  the rope_deltas addition), `mm_token_type_ids`/`pixel_values`/`grid_thw`-family args are
  **all None** (passing `image_grid_thw` recomputes and overwrites `model.rope_deltas`),
  and reuse the cache serially per scene by cropping back with
  `DynamicCache.crop(prefix_len)`. Since `rope_deltas` is **instance state** on
  Qwen3VLModel, no other TE call may be interleaved between the prefix call and all the
  continuations (designed to complete within a single helper call)

**E2E measured (ref2i_batch, 1 Little Red Riding Hood reference, same 3 scenes,
seed 12345, 768², 30 steps)**:

| | Encode phase | Total | Per image |
|---|---|---|---|
| No sharing (equivalent to H3_REF_PREFIX_CACHE=0) | 212.5s | 494.7s | 164.9s |
| **With sharing (default)** | **83.1s** | **350.1s** | **116.7s (-29%)** |

Quality: PSNR 21.9-27.4dB against the same-seed output without sharing. Batch output does
not bit-match the legacy path (the ~1.5% rounding difference on the prompt tail drifts the
trajectory — the same kind and level of epsilon-class drift as when Sage Attention was made
default), but composition, quality, and character consistency are all visually equivalent.
For A/B tests requiring bit reproducibility, revert to the legacy path with
`H3_REF_PREFIX_CACHE=0`. Since ref2va_batch (video) uses the same code for its encode
phase, the same reduction applies directly (~130s/3 scenes). The reference VAE encode
(~a few seconds/scene) is not shared and stays per-scene (the benefit is small, and this
avoids increasing the risk of state aliasing).

#### Reusing the prefix across single requests (`H3_REF_PREFIX_CACHE_SINGLE`, default 0, 2026-08-24)

Carries the shared prefix above across **separate single `/api/ref2va` requests**, not just
within one batch call. For workflows that pair one reference image with many different audio
tracks and prompts (exactly what mv_studio's lip-sync generation does), the ~52s reference
vision encode disappears from the second request onward.

**Default OFF.** It keeps a process-lifetime KV cache (measured 1.04GiB of VRAM) alive between
requests, so "do not change existing behaviour" wins by default. Enable it through the gateway
overrides: `{"backend":"h3","preset":"96gb","overrides":{"H3_REF_PREFIX_CACHE_SINGLE":"1"}}`.

**When it applies** (decided mechanically; anything else silently falls back to the legacy path):

- **Image references only** (any number of audio references is fine). `_build_presentation` emits
  only the `"<Audio j>: "` label for audio and the waveform never reaches the conditioner, so
  **changing the audio's content and length leaves the prefix token sequence bit-identical**.
- **Never used when any video reference is present** (a video's pixels do enter the prefix). That
  case drops the cache and falls back to the full encode.

The **key** is a SHA-256 of what actually enters the prefix forward (token ids + the image tensor's
bytes + `image_grid_thw`), not of filenames — the count, order, resolution or pixels of the
references all change the key. Key computation measured at 0.14-0.35s.

**Safeguards the batch version does not need** (see `_encode_ref2va_prompt_prefix_cached`'s
docstring):

1. `model.rope_deltas` is *instance state* on `Qwen3VLModel`: the prefix writes it, the
   continuation reads it. Across requests, a t2va encode or `/api/prompt/enhance` can overwrite it
   in between, so the value is cloned at prefix time and **written back right before every
   continuation**.
2. A `weakref` checks text_encoder identity — a freed/reloaded/re-quantized TE invalidates the
   cache. `_free_text_encoder()` also drops it actively so the VRAM comes back immediately.
3. `DynamicCache.crop(prefix_len)` after every continuation (plus a length check before it, as
   insurance against an exception unwinding mid-continuation).

**Measured** (96GB box, preset=96gb + `H3_TRANSFORMER_QUANT=int8`, 768x448 / 8s / turbo / 4 steps /
seed 777, one image + one audio reference):

| | 1st | 2nd | Peak VRAM |
|---|---|---|---|
| OFF (default, legacy path) | 136.6s | **94.5s** | 73.79GB |
| **ON** | 140.4s (MISS, prefix 52.5s) | **41.9s (-56%)** | 74.92GB (+1.1GB) |

- Same image + **different audio** (a different 6.6s file vs 8.0s) also hits: 37.0s.
- Swapping the reference image logs `cache cleared` and misses (93.6s); swapping back misses again
  (94.1s) but restores the original key (keys are deterministic).
- The ref batch endpoints benefit too, since in resident mode they loop `generate_ref2va()`:
  the second scene of ref2i_batch went 71s -> **21s**.

**Numerics**: a MISS (prefix computed in this request) and a HIT (prefix reused) produce
**bit-identical** output (`max diff = 0` over 192 frames and the audio), so cross-request reuse
itself is exactly equivalent. Against the legacy path (one monolithic encode) it is *not*
bit-identical — the ~1.5% rounding difference from splitting prefix/continuation is amplified by
turbo's 4 steps, the same property and the same cause the batch path (`H3_REF_PREFIX_CACHE`)
already accepts:

| Comparison | Mean pixel diff | PSNR | Audio correlation |
|---|---|---|---|
| ON (HIT) vs OFF | 2.46/255 | 31.7dB | 0.9909 |
| ON (MISS) vs ON (HIT) | **0 (bit-identical)** | inf | 1.0 |

Visually the composition, character identity and even mouth shape are indistinguishable. Keep
`H3_REF_PREFIX_CACHE_SINGLE=0` (default) for A/B tests that need bit reproducibility.

**Interaction with H3_LOWVRAM**: lowvram frees the TE every request, so the cache is dropped every
time (no benefit). Verified on hardware with 48gb-lowvram + ON: every request logs
`cache cleared: text_encoder freed`, the next one misses, and there is no VRAM leak or exception
(with TE prune the 51-layer TE makes the cache 0.84GiB, sized automatically).

#### Batch generation of reference-conditioned videos (`/api/ref2va_batch`, production implementation 2026-08-08)

Applies the same phase reordering as ref2i (implemented via the same method,
`generate_ref_batch(still=False)`) to **normal-duration video**. Generates each scene's
video for a narrative, sharing a common reference with varying prompts, continuously.
Duration (`seconds`) is common and required across all scenes (since the scheduler's
sigmas/timesteps values being identical across scenes is a precondition for phase
reordering; automatic duration derivation from an audio reference is also unavailable in
batch mode). The UI's Ref2VA / Ref2I tabs have a "Batch continuous generation (1 line = 1
scene)" checkbox (usable independently of the tab: Ref2I+batch = ref2i_batch, Ref2VA+batch
= ref2va_batch).

**Measured (2026-08-08, RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 1 Little Red Riding Hood
reference, 2 scenes, 5s, 768², 30 steps)**: total 803.2s = **401.6s/video** (17% shorter
than sequential's ~485s/video @2 scenes). Breakdown: encode phase 151.3s + denoise 503.4s
(247.9-255.5s/scene) + decode 25.4s. Peak VRAM 40.5GB. The marginal cost of an additional
scene is ~330s/video (vision encode ~65s + denoise ~250s + decode ~13s), so as scene count
grows this asymptotically approaches **about a 32% reduction**. Since video is
denoise-dominated, this is less dramatic than the still-image batch, but quality and
character consistency are as good as sequential (both scenes kept the red cloak and
followed their prompts).

#### Spike: Ref2VA x ultra-short clips (reference-conditioned still images, 2026-08-07, measured -> productionized as described above)

Measured whether "the ultra-short 22-frame clip, already verified in t2va, still holds up
in quality when combined with reference packing (packing_ref2va: reference rows outnumber
generation rows)" (`scripts/probe_ref2va_short.py`, main codebase unmodified, monkeypatch
confined to the probe). If this held up, it would enable mass production of
**character-consistent scene stills**, speeding up building material for generating
multiple narrative videos.

Conditions: reference = 1 PNG of a Little Red Riding Hood-style girl (this repo's own t2i
output), the same prompt (with a `<Picture 1>` reference), seed 12345, 30 steps, 768².
RTX PRO 5000 48GB + `H3_LOWVRAM=1`.

| Condition | Denoise | Decode | Peak VRAM | Wall (sequential, including load) | Quality (visual, center frame) |
|---|---|---|---|---|---|
| short22 (0.917s) | 58.1s | 1.7s | 36.7GB | 265s | **No breakdown, high quality** |
| baseline (5.0s) | 243.1s | 10.6s | 40.7GB | 485s | No breakdown (anchor) |

- **Verdict: GO**. Even in this out-of-distribution overlap (ultra-short clip x reference
  packing), quality holds up, and the reference's costume/props (red cloak, lantern, white
  dress) and art style were preserved even at short22. Composition also correctly followed
  the prompt (seated on a moss-covered rock)
- Denoise is about 2x t2va's 22-frame value (29s -> 58s), since the packed sequence
  lengthens by the reference rows. Still 1/4 of the 5-second baseline
- The strictness of facial identity depends on how the reference photo looks (this
  experiment used a somewhat back-facing shot) — a property of the reference, not a flaw
  of the ultra-short clip
- The production implementation (API/UI, the ref2va version of `generate_still_batch`) was
  **not started at the time** — the same three-part set as t2i (duration-gate relaxation,
  VAE small-clip fix, exception cleanup) is reusable, and the batch phase-reordering
  approach looks reusable too, following t2i_batch's method (scheduler `_step_index` reset
  + per-scene FBC reset, with a single shared reference encode if the reference is common
  across all scenes)

#### The preceding ultra-short-clip probe (2026-08-07, measurement record)

Frame counts step in units of 17n+5 (minimum 5 = 0.208s, next 22 = 0.917s). The probe was
verified via monkeypatch only (`scripts/probe_short_frames.py` + `_one.py`). The
environment at the time was RTX PRO 6000 96GB, default mode.

| Frame count | Denoise (30 steps) | Decode | Result |
|---|---|---|---|
| 5 (0.208s) | completes | **fails** | VAE chunked-decode boundary bug (2-frame latent gives num_chunks=0 -> `torch.cat([])`) -> **fixed by item 2 above** |
| **22 (0.917s)** | **13.5s** (1/7.6 of the 124f case) | 1.2s | **success, quality on par with the 5-second baseline** (visually confirmed) |
| 124 (5s, baseline) | 102.2s | 6.3s | success |

- 22 frames is out-of-distribution (1/5 of the official 5-second minimum), yet quality
  holds up, and audio is non-silent too
- **Lesson**: during the probe, the 5-frame exception occurred before the post-decode
  cleanup (transformer reload), and a chain of subsequent-request OOMs from ~98.5GB of
  leaked residency was observed -> reflected into the production implementation as the
  cleanup path described in item 3 above

## UI

### Switching settings from the UI (2026-08-06)

Opt-in settings that could previously only be changed via environment variables can now be
operated from the UI, split into two categories by nature.

#### Instant apply (checkbox directly above the generate button, no reload needed)

FirstBlockCache (+ threshold), Sage Attention, Turbo LoRA. Sent as per-request parameters
(`cache` / `cache_threshold` / `attn` / `turbo`), applied by `MiniMaxH3Runner.
apply_instant_settings()` to the resident transformer after acquiring the generation lock
and before denoise (`disable_cache()`/`enable_cache()`, `set_attention_backend()`,
`_TurboLoRALinear.enabled`). **If unspecified, the process default applies as before**, so
existing curl/scripts work unchanged. Enabling turbo automatically disables FBC (following
the original safety rule).

Measured (same seed, switched without restart): FBC on 100.8s (6 skips) / off 129.3s
(0 skips), Sage on 129.3s / off (native) 158.5s, Turbo on (8 steps) 38.9s.

#### Requires reload (collapsible header + "Apply (reload)" button)

transformer int8, TE quantization, TE layer pruning, low-VRAM mode, video VAE fp16.
`POST /api/settings/apply` calls `apply_reload_settings()` in `core/settings.py`, which
frees all models within the runner and reloads them with the new settings **without
restarting the process** (self-killing the process would leave no one able to restart it,
making the UI permanently unrecoverable, so no os.execv/self-kill approach is implemented).
Returns 409 during generation; 400 for unverified combinations.

Measured: transformer_quant none->int8->none took 56.0s / 55.0s (GPU 87.5->55.0->87.3GB),
lowvram 0->1->0 also round-trips correctly. `GET /api/settings` returns current values and
choices, and the UI initializes from this.

**UI implementation note**: unchecking a checkbox must send an explicit `turbo=0`, not an
empty string (an empty string can be interpreted as "unspecified = server default," which
could mean unchecking does not actually disable turbo on a server started with
`H3_TURBO_LORA=1`). turbo and upscale are mutually exclusive — selecting one automatically
deselects/disables the other (consistent with the server-side 400).

### Gallery of generated videos (2026-08-06)

Tiles the mp4s directly under `outputs/` below the result display. Thumbnails are not
generated server-side; instead, `<video preload="metadata" src="....mp4#t=0.1">` is used to
have the browser draw the first frame (no added dependency; the same technique used for
Ref2VA's reference tiles).

- `GET /api/outputs`: **the *.mp4 and *.png directly under it** (still-image mode's
  outputs; distinguished via the `type` field "video"/"image". Verification artifacts like
  `outputs/ab_*` are excluded). Duration/resolution obtained via ffprobe and cached in
  memory keyed by mtime+size
- `POST /api/outputs/delete`: **path-traversal countermeasure** (rejects `/`, `\`, `..`,
  and after resolving verifies the result is directly under `outputs/` — blocking
  escapes via symlinks too). The UI requires `confirm()`. Confirmed on real hardware that
  `../app.py`, `/etc/passwd`, `ab_*/...`, etc. return 400
- `POST /api/outputs/concat`: **concatenation order follows "the order checked"**
  (display order is newest-first). If all inputs' parameters match, uses
  `concat demuxer + -c copy` (**no re-encoding = zero quality loss**); if they don't
  match, re-encodes via `filter_complex`, matching the resolution of the first video (a
  silent `anullsrc` is synthesized for silent inputs). Fewer than 2 inputs returns 400,
  concurrent concat requests return 409 (does not take generation_lock since it doesn't
  use the GPU). **PNGs (still images) are excluded from concatenation** (because ffprobe
  can read PNG as a video stream too, so it's explicitly rejected with 400 by extension;
  the UI also disables the concat button if any PNG is among the selection)

**A note on dependencies**: this app's generated-output muxing **uses PyAV**
(`av.open()`, writing libx264+aac directly, `_mux_mp4()` in `core/runner.py`) — **it never
used the ffmpeg command**. The gallery's ffprobe/ffmpeg calls are **the app's first ever
external-command dependency** (this environment has `/usr/bin/ffmpeg`; if absent, it
catches `FileNotFoundError` and returns an explicit error). If you want to eliminate the
external-command dependency, a realistic alternative is "packet re-muxing via PyAV
(equivalent to `-c copy`), limited to identical parameters, erroring on any mismatch"
(implementing full re-encoding of mismatched parameters in PyAV would essentially be
re-implementing ffmpeg, so this is not recommended).

**UI implementation note**: rebuilding all tiles on every selection causes all `<video>`
elements to be regenerated, producing flicker (noticeable at 95 tiles). Selection state
should be updated as a **diff** on badges/checkmarks (`updateGallerySelectionUI()`; the
list itself is only rebuilt when `/api/outputs` is fetched).

## LLM prompt enhancement

### LLM prompt enhancement (added 2026-08-04)

Uses a local LLM (gemma4-31B, OpenAI-compatible `/v1/chat/completions`) to reformat input
prompts into the H3 official guide's format. A local reproduction of the cloud Hailuo AI's
internal prompt-formatting layer.

- Connection target: environment variable `H3_LLM_URL` (default
  `http://127.0.0.1:64660`). Returns 502 if unreachable (does not affect the generation
  feature)
- `POST /api/prompt/enhance` {text, mode, seconds, task, lang}
- Modes (the UI's "LLM enhance" button + mode selector):
  - `storyboard` (default): expands into a multi-shot CUT-timecode format (total duration
    = seconds, 2-3 cuts, hard cuts, subject-identity preservation, per-cut sound
    instructions; focal length restricted to 35/50/65/100mm). This is not the official H3
    documentation's notation but this app's own proposal (see `h3-official` below)
  - `brief`: single-shot elaboration in the official brief format (scene -> subject ->
    action -> camera -> sound -> ending)
  - `h3-official` (added 2026-08-07): strict conformance to the field structure/notation
    of MiniMax's official skill `h3-prompt-writing` (`skills/h3-prompt-writing/` in the
    `MiniMax-AI/MiniMax-H3` repository). Switches the reference guide by `task` (the UI's
    current tab = t2va/fl2va/ref2va): t2va/fl2va use the 3-field format from
    `references/base-en.txt` (`integrated_multimodal_description` /
    `overall_soundscape` / `non_diegetic_music`, the `[Shot N] At MM:SS.SS` cut notation,
    dialogue preserved verbatim via `<d>[language] ...</d>`), while ref2va uses the
    6-field format from `references/ref-en.txt` (`subject_definitions` / `summary` /
    `retention_analysis` / `detailed_description` / `overall_soundscape` /
    `non_diegetic_music`). Output is the official default of English (`lang=en`);
    specifying `lang=ja` can also localize just the narrative body into Japanese (field
    names, `[Shot n]`-style labels, timecode notation, and dialogue inside `<d>` remain in
    English/the original language per the official rules). The reference body text is not
    bundled in this repository (since the source repository has no license notice).
    `venv/bin/python scripts/fetch_h3_skill.py` must be run beforehand to fetch it into
    `skills_cache/` (in .gitignore); if not yet fetched, returns a 400 with the fetch
    command guidance. The system prompt feeds in SKILL.md + the full reference guide
    without summarization (about 18.8KB for t2va/fl2va, about 26.6KB for ref2va)
  - `translate`: a literal English translation without embellishment (preserves CUT
    structure)
- The enhancement result replaces the prompt field (editable), with "revert" restoring one
  generation back. The generated result shows the full prompt actually used in a
  collapsible section (for comparing with/without enhancement, and across modes)
- The UI bundles a hand-writing "prompt guide" cheat sheet (the official brief structure,
  CUT notation, measured cut accuracy of +-1s)
- Verified against a real LLM (gemma4-31B Q4_K_M): confirmed 3 modes (brief/storyboard/
  translate) all follow the format (storyboard also respects total timecode sum and focal
  length constraints, response time 11-17s). As of the 2026-08-07 addition of
  `h3-official`, the LLM server was not connected, so only generation A/B was performed
  (see above); actual response verification via the LLM is still needed separately

### Real-hardware verification of the official skill (h3-prompt-writing) mode (2026-08-07)

Results of measuring `h3-official` mode against a local LLM (gemma4-31B) connection.

**Structural conformance**: the generated prompts fully complied with the official spec.
- T2VA: 3 fields (`integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`), `[Shot 1]` with no timestamp, `[Shot 2] At 00:05.000` in 3-digit
  notation, dialogue preserved verbatim in the original language via
  `<d>[Japanese] おかえり</d>`, speaker ID `(S1)` also attached. Response time 6.2s
- Ref2VA: all 6 fields output, using `<Subject n>`/`<Picture n>`/`<Audio n>` labels and
  relationship markers such as `fully_preserved` / `fully_copy`. Response time 12.8s

**Measured cut positions** (768², 10s, turbo 8 steps, LLM-generated prompt, 2 seeds each.
ffmpeg scene detection `gt(scene,0.15)`):

| Notation | Instruction | Detected (seed 12345 / 777) | Deviation |
|---|---|---|---|
| Official `[Shot 2] At 00:05.000` | 5.000s | 4.875 / 4.875 | **-0.125s (identical for both seeds)** |
| Custom `CUT 2 [6-10s]` | 6.0s | 6.083 / 6.333 | +0.083 / +0.333s |

The official notation shows **less variance** (identical across the two seeds). That said,
the custom CUT notation also stays within a 0.1-0.3s deviation, **not nearly as large a
gap as the earlier single hand-written-prompt trial (custom notation, +1.0s)**. Both are
practically usable; the measured conclusion is that the official notation is more
reproducible.

**Implementation pitfall (reproduced and fixed on real hardware)**: specifying `lang=ja`
**still returns English**. Placing the language instruction at the start of the system
prompt gets overwritten by the 15.8KB English reference that follows it (which is packed
with English output examples). Resolved by **placing the language instruction after the
reference body, as the final instruction** (the same principle established as "the end of
the prompt is strongest" in diffusers-server's T-pose implementation). After the fix, only
the narrative text becomes Japanese, while field names, `[Shot n]`, timecodes, and dialogue
inside `<d>` tags remain in English/the original language.

### h3-official quality assurance: verification + repair loop (2026-08-08)

To avoid arguing by speculation about "is this an LLM capability limit, or fixable via
prompting," **first measured the failure rate** (`scripts/probe_h3official_compliance.py`,
5 inputs x 3 runs), then applied countermeasures based on the results. LLM: gemma4-31B
Q4_K_M / llama.cpp, **n_ctx=7680**.

#### Baseline measurement: failures were concentrated in a single class

| Failure class | Occurrence rate |
|---|---|
| Structure/notation (field names, `[Shot n]`, timecodes, `<d>` tags, speaker IDs) | **0/15 (0%)** |
| Time allocation | 6/15 (40%) |
| Context overflow | 0/15 (2,900+ tokens of headroom) |

**The concern that "a different input would surface a different problem" turned out to be
unfounded, in a good way** — violations were not scattered, but concentrated in the single
class of time allocation. Moreover, those 6 cases split into two natures:

- **Input physically impossible (3/6)**: the estimated speaking duration of the dialogue
  exceeds the requested total duration. One example was a 38-character monologue
  (estimated 9.5s) requested with a 9s duration, where **the LLM had already allocated
  the best possible split, giving all 9s to a single shot**. Since the official spec
  requires dialogue to be preserved verbatim, shortening isn't an available escape route
  either -> **unsolvable by any model. Information that should be returned to the user**
- **LLM allocation mistake (3/6)**: fits within the duration, but the placement is poor ->
  the kind of thing that can be fixed by pointing it out

Context: t2va's system prompt uses 4,397 of 7,680 tokens, with headroom to spare. **ref2va
uses 6,191 tokens (81%), leaving only about 1,100 tokens of headroom** (in measurements,
all 6 fields completed normally, but caution is warranted with longer inputs).

#### Countermeasures (3 layers)

1. **Validator** (`core/prompt_check.py`): implements deterministically-checkable rules
   (F1 3 fields / F2 no timestamp on the first shot / F3 strictly increasing cut times /
   F4 within duration / F5 minimum shot-duration floor / F6 `<d>` tags and language tags /
   F7 dialogue fits within its shot / F8 speaker ID). F5 and F7 are **this app's practical
   rules, not present in the official spec** — the official spec only says "cut times must
   be within the duration," which permits a literal-but-unusable output such as cutting
   at 4.5s within a 5s duration, leaving the final shot only 0.5s (this actually occurred).
   Semantic consistency (e.g. mixed shot framing within one shot) is only approximately
   checked and left as a **warning**. **This also helps hand-written prompts**
2. **System prompt improvement**: the English wrapper **had nothing after the guide body**
   (only the Japanese wrapper had a trailing block, from the `lang=ja` fix). Since the
   default is `lang=en`, all instructions including the duration constraint were buried
   inside the 15.8KB guide. Added 6 time-allocation rules (minimum shot-duration floor,
   estimating dialogue speaking time, guideline shot count, one composition per shot,
   the boilerplate phrase for off-screen audio, speaker ID) at the **end** of both versions
3. **Repair loop** (`enhance_prompt_checked` in `core/llm.py`): upon detecting a
   violation, present its content in Japanese and regenerate (up to 2 times,
   `H3_OFFICIAL_MAX_REPAIRS`). **Discards any repair candidate that increases the
   violation count** (preventing an accident where a repair breaks something else).
   Input infeasibility is judged **before handing off to the LLM**, raising
   `InfeasibleInputError` -> 400 + advice

#### Measured results after improvement (same conditions)

| | clean | infeasible input detected | **violations remaining** |
|---|---|---|---|
| Baseline | 9/15 | 0 | **6/15** |
| **After improvement** | **12/15** | **3/15** (correct behavior) | **0/15** |

**Unresolved violations dropped to zero.** Every case now either "returns a valid prompt"
or "returns an infeasibility reason with advice." Median time went from 8.5s to 9.2s
(+8%); the repair loop only fires when needed (about 21s, roughly double, when it does).

The API adds `violations` / `warnings` / `check_report` / `attempts` / `repaired` to the
h3-official response of `/api/prompt/enhance`. The UI displays any remaining findings in
the status area (**generation is not blocked** — since the prompt is editable, the human
is the final gate).



---

## FILE: docs/FEATURES.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/docs/FEATURES.md)

# 機能と設定の詳細(minimax-h3)

**日本語** | [English](FEATURES.en.md)

README から分離した、各機能・環境変数の詳細説明と実測値。各節は README にあった原文をそのまま移したもので、節内の日付はその機能を実装・実測した日を示す。使い方の要点は [README.md](../README.md)、時系列の作業記録は [HISTORY.md](HISTORY.md) を参照。

- [性能の概要](#性能の概要)
- [VRAM 構成と常駐設計](#vram-構成と常駐設計)
- [text_encoder](#text_encoder)
- [transformer とデノイズ](#transformer-とデノイズ)
- [生成モード](#生成モード)
- [UI](#ui)
- [LLM プロンプト強化](#llm-プロンプト強化)

## 性能の概要

### 高速化の効果(実測まとめ)

**このリポジトリの中心的な発見: 律速はデノイズではなく「モデルの load/free という固定費」
だった。** 以下は各節の実測値を1か所に集めたもの(再測定ではない)。

#### 累積: 素の構成から現在まで

48GB機(PRO 5000 48GB + RTX 4000 20GB)・`H3_LOWVRAM=1` 系・768²・**定常値**
(初回のモデルロードを含まない2回目以降。t2va は 5秒 = 124フレーム)。

| 段階 | t2i 1枚 | t2va 5秒 |
|---|---|---|
| 素の構成(30steps、turbo なし) | 157s | 351.4s |
| + lightx2v turbo LoRA(4steps) | 157s | 143s |
| + `H3_TE_PREQUANT`(量子化TEのディスクキャッシュ) | 83.2s | — |
| + `H3_TE_DEVICE`(TE を2枚目GPUへ) | 約35s | 60.5s |
| + `H3_KEEP_TRANSFORMER`(transformer 常駐) | **9.7s** | **44.2s** |
| | **16倍** | **8.0倍** |

デノイズ自体を速くする turbo だけでは **2.6倍で頭打ち**。残りの3倍強は
**すべて固定費の撤廃**から来ている(turbo なし 30steps のままでも、固定費撤廃だけで
t2i は 157s → 51.1s と3倍速い)。

#### モード別の到達点(全モードを同一構成で実測、2026-08-12)

構成: **int8 transformer 常駐 + 投影TE(NF4)+ デコード窓の解放停止 + turbo 4steps + sage +
fp16 デコード**、**GPU 1枚**(96GB機で実測。`H3_TRANSFORMER_QUANT=int8 H3_KEEP_TRANSFORMER=1
H3_VIDEO_VAE_FP16=1 H3_TE_PROJ=… H3_TURBO_LORA=1`)。

| モード | 単発(最速) | **連番なら1件あたり** | デノイズ | ピーク | 時間を食っている残りの固定費 |
|---|---|---|---|---|---|
| **t2i** 768² | **7.40s** | 7.9s(**0.94倍 = 使う意味なし**) | 2.40s | 45.0GB | VAE の CPU↔GPU 往復 ~3.3s |
| **t2va** 5秒 768² | **28.13s** | (バッチAPIなし) | 14.94s | 45.6GB | デコード 7.05s |
| **ref2i** 768²(参照付き静止画) | 79.3s | **47.0s(1.69倍)** | 7.8s | 45.4GB | 参照ビジョンエンコード **~47s** |
| **i2va** 5秒 768²(画像参照→動画) | 103.1s | **75.0s(1.37倍)** | 22.0s | 45.9GB | 同 ~47s + t2va transformer の再ロード ~13s |

**計測条件**(この表の数値の読み方):

- **すべて定常値。サーバ起動時の初回ロードは含まない**(2回目以降のリクエスト)。
  起動時に transformer + VAE を常駐させるのに **約50秒**かかるが、これはプロセスに1回だけ。
- **参照系(ref2i / i2va)は初回リクエストだけ別**: `transformer_ref` は起動時ではなく
  最初の参照リクエストでロードされるため、**初回のみ +55秒**(ref2i 実測: 初回 134.7s →
  定常 79.3s)。以後は常駐するので定常値になる。
- 解像度 **768×768** 固定。動画は 5秒 = **124フレーム**(24fps)、静止画は **22フレーム**。
- **turbo LoRA の 4ステップ**(`H3_TURBO_LORA=1`)。FirstBlockCache は turbo により自動 OFF。
  attention は sage。seed とプロンプトは各モードで固定。
- 「連番なら1件あたり」= 同じ参照/設定で複数件まとめて作るときの1件あたり
  (`/api/t2i_batch`, `/api/ref2i_batch`, `/api/ref2va_batch`)。実測は t2i/ref2i が3場面、
  i2va が2場面。**バッチ経路は `H3_LOWVRAM=1` 専用**なので単発列とは構成が違う —
  つまり「1件だけ作る最速手段」と「N件作る最速手段」を並べた表。

- **全モードが 45〜46GB に収まる** = 48GB カード1枚で全機能が最速級で動く。
  ただし **t2va 系と参照系を同じプロセスで混ぜると両方の transformer が常駐して 74.3GB**
  になる(実測。t2va に戻ったときのピークは 77.3GB)。48GB 1枚で運用するなら
  **プロセスをモード別に分ける**こと。
- **参照系の律速はデノイズではなく参照のビジョンエンコード ~47s**。turbo の効きが
  t2va(5.5倍)より小さく i2va で 2.8倍に留まるのはこのため。
- 2枚構成 + bf16 transformer なら **t2i 6.89s / t2va 26.8s** がこれまでの最速だが、
  77GB 級のカードが要る(差は 7% 程度)。詳細は日付節 2026-08-12。

#### 連番生成(バッチ)について分かったこと

- **t2i のバッチはもう使う意味がない**(0.94倍 = わずかに遅い)。バッチは元々
  「モデルのロード固定費をバッチ1回に償却する」機能で、**常駐化で償却すべき固定費が
  消えた**ため(常駐化前は 157s → 67.5s の 2.3倍だった)。
- **参照系だけは今も効く**。共有されるのが**ロードではなく参照ビジョンエンコード ~47s**
  だから。**バッチのステップ時間は単発と完全一致**しており(ref2i 2.598s vs 単発 2.599s、
  i2va 7.321s vs 7.323s)、バッチ経路自体のオーバーヘッドはゼロ — 差は純粋に
  「47s のエンコードを何回払うか」だけ。
- **場面数が増えるほど効く**: 節約は `47×(場面数-1)/場面数`。i2va なら 2場面 1.37倍 →
  3場面 1.44倍 → 5場面 1.57倍。同じキャラクターで長い物語を作る用途ほど有利。
- **「共有すれば縮む」モデルは静止画・動画とも成立**(実測削減 vs 予測: ref2i 3場面が
  32.3 vs 31.3s/枚、i2va 2場面が 28.1 vs 23.5s/本)。→ **参照エンコードをリクエスト跨ぎで
  キャッシュすれば、単発の繰り返しでも同じ短縮が得られる**見込み(未実装、日付節 2026-08-12)。
- **制約2つ**: バッチ経路は `H3_LOWVRAM=1` 専用で、それ以外では**黙って逐次実行に落ちる**。
  参照バッチは `H3_TE_DEVICE` と併用不可(「TE用GPUに 24GB 以上」のガード。**32B TE の
  vision 活性化を前提にした閾値**で 3.11GB の投影TEには過大 — 16GB の 4060 Ti が弾かれる)。
  どちらも**要見直し**。

> **計測の落とし穴(2026-08-12 に実際に踏んだ)**: バッチ側だけ `height`/`width` を渡さずに
> 単発と比べると、**バッチはサーバ既定の 16:9 キャンバス(1344×768)で生成される**ため
> 1.75倍のピクセル数になり、「バッチはステップが 1.75 倍遅い」という**存在しない現象**が
> 見える。ピクセル比 1344×768÷768² = 1.75 と実測のステップ時間比 1.753 が一致したことで
> 判明した。**モード間・経路間の比較では解像度を必ず明示すること。**

### 速度の正直な注記

**低VRAM構成は動くが遅い。** `H3_LOWVRAM=group` は毎ステップ int8 重み(~34GB)を CPU→GPU に
流すため、転送が律速になる。上の対応表の 16GB / 8GB の実測値は **PCIe Gen3 x4 スロット**での
もので、`16.5s/step`(t2i)〜`51s/step`(t2va)のほとんどは転送時間。**Gen4 x16 のまともな
スロットなら転送は約1/8**になるので、これは「16GBカードの性能」ではなく「この箱のスロット」の値。
また int8+SDPA の軌道では FirstBlockCache が効かず(`cache_skipped_steps: 0`)、閾値調整で
最大2倍の短縮余地がある(未検証)。

品質については、投影TE(PSNR 22.4dB vs 32B)も int8+SDPA も、同一seedでも構図が変わる
(**軌道の分岐であって劣化ではない**)。むしろ実測では int8+SDPA の方がプロンプト忠実度が
上がったケースもあり、参照(ビジョン)経路も目視で 32B 正解と同水準の忠実度を確認済み。
**構成をまたぐ品質は PSNR/MD5 では判定できないので目視で見ること。**

> **追記(2026-11-17)**: 上の記述は 2026-08 時点のもので、次の 3 点が変わっている。
> - 「上の対応表」は分離後は次節の「[VRAM×機能マトリクス](#vram機能マトリクス2026-08-11-実測)」を指す。
> - **turbo LoRA(4 ステップ)が `H3_LOWVRAM=group` でも使えるようになった**(自作ストリーマ + 起動時 `H3_TURBO_LORA=1`)。転送回数がステップ数に比例して減るため、転送律速の「動くが遅い」はほぼ解消できる。A4000 16GB(Ampere sm_86、PCIe Gen4 x4、SDPA)の実測: t2va 5 秒・768² が 30 ステップ 790 秒(peak 8.0GB)→ 4 ステップ 151 秒(peak 7.0GB)。8GB 級の指定(`H3_TE_STREAM=1 H3_TE_DIET=1`、VAE デコーダのストリーミング、`H3_VRAM_LIMIT_GB=8` で模擬)では 768² 209 秒・3.08GB、1024² 376 秒・3.73GB、fl2va 225 秒・3.37GB、ref2va 299 秒・3.02GB(README の対応表参照)。
> - 「int8+SDPA の軌道では FirstBlockCache が効かない」は現在は当てはまらない。同じ A4000(int8 + SDPA、30 ステップ)で t2i は 7 ステップ、t2va は 6 ステップ分がスキップされた(`cache_skipped_steps`)。
> 品質に関する段落(構成をまたぐと同一 seed でも構図が変わるのは軌道の分岐であって劣化ではない、目視で判定する)は変わらない。

## VRAM 構成と常駐設計

### VRAM×機能マトリクス(2026-08-11 実測)

○ = 実測で完走 / △ = 導出見込み(未実測)/ × = OOM。**時間は定常値**(初回のモデル
ロードを含まない2回目以降)、**ピークは torch 計測の割当ピーク**。速度はいずれもこの箱の
**PCIe Gen3 x4 スロット + sm_89(4060 Ti)+ SDPA** での値で、**Gen4 x16 なら重み転送は
約1/8**になる(下の「速度の正直な注記」参照)。16GB/8GiB 行は 30ステップ、
48GB+20GB 行は turbo 4ステップ。

| 構成 | t2i 768² | t2va 5秒 768² | ref2i | i2va(画像参照) | 音声参照 | 768×1344 5秒 |
|---|---|---|---|---|---|---|
| **実 RTX 4060 Ti 16GB 単体** | ○ 498s・ピーク7.4GB | ○ 25分・11.4GB | ○ | ○ 39分・9.41GB | ○ 54分・11.96GB | ○ 66分・13.37GB(実15.2GB=上限) |
| **8GiB×2**(計算+TE、バラスト模擬) | ○ 512s・6.4GB | ○ 25.6分・7.23GB | ○ 17.7分・6.69GB | × OOM | × OOM | × OOM |
| **12GB 単体** | △ 見込み | △ | △ | △ | ×に近い(実14.7GB) | × |
| **48GB+20GB**(2枚、turbo LoRA) | ○ 9.7s | ○ 44.2s | ○ | ○ | ○ | ○ |

- 12GB 単体は未実測。t2va 実ピーク 11.4GB / i2va 9.41GB(実11.2GB)が 12GiB に収まる計算だが、
  音声参照(実14.7GB)・768×1344(実15.2GB)は超える。
- 16GB / 8GB 構成の鍵は **投影TE**(Qwen3-VL-4B + 学習済み線形写像 ClipProj、NF4 で 3.11GB。
  32B の text_encoder の代替)+ `H3_LOWVRAM=group`(transformer int8 をブロック単位で
  ストリーム、GPU常駐は ~1.4GB)+ video VAE の fp16 デコード。詳細は日付節 2026-08-10〜11。
- **測定環境は2台のマシンで共有**: 16GB / 8GB 列は **96GB機(RTX PRO 6000 + 増設 4060 Ti 16GB)**、
  48GB+20GB 列は **48GB機(RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB)** での実測。旧版の実測
  (96GB単騎時代)は各日付節にそのまま残してある。

### VRAM対応表と主な環境変数(早見表)

用途に応じた起動例。上段5行は 96GB 機(PRO 6000)でのバラスト実測(32B TE 使用)、
下段3行は **投影TE を使う実カード実測**(2026-08-11。詳細は冒頭の
「[VRAM×機能マトリクス](#vram機能マトリクス2026-08-11-実測)」と日付節 2026-08-11)。

| GPU | 起動時の指定 | t2va 実測(768²・5秒) |
|---|---|---|
| 96GB | (指定なし=既定) | peak 92GB / 約160秒 |
| 80GB級 | `H3_TRANSFORMER_QUANT=int8` | peak 59.7GB / 約160秒 |
| 48GB級 | `H3_LOWVRAM=1` | peak 38.9GB / 約215秒 |
| 32GB級 | `H3_LOWVRAM=group` | peak 28.7GB / 約280秒(2026-11-15 の自作ストリーマで denoise は約 1.7 倍速、下の `H3_GROUP_STREAM`) |
| 18GB級 | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak 17.7GB / 約280〜320秒 |
| **16GB 単体**(実 4060 Ti) | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1` | peak 11.4GB / 約25分※ |
| **6GB 単体 / 8GB 単体**(2026-11-15) | 上の行 + `H3_TE_STREAM=1 H3_TE_DIET=1` | 768²: 4.62GB / 1024²: 6.41GB(MD5 一致の既定。48GB 機で 272 / 552 秒)。空きが足りないときは attention をヘッド分割して 1024² 4.79GB(別サンプル相当、+15%)、`H3_CACHE=none` なら 3.03GB。ホスト RAM 32GB / 16GB は量子化キャッシュから直接流す(1K は同速、512² は約 2 倍)。Gen3 x4 の箱では上の※と同じ転送律速 |
| **12GB 単体**(見込み・未実測) | 16GB 単体と同じ | 実ピーク11.4GBが12GiBに入る計算。参照系・768×1344は不可 |
| **8GB×2**(バラスト実測) | 16GB 単体の指定 + `H3_TE_DEVICE=cuda:1` | peak 7.23GB / 約25.6分※ |

※ 低VRAM実カード列の時間は PCIe Gen3 x4 スロット + sm_89(SDPA)での実測。Gen4 x16 なら
重み転送は約1/8(「[速度の正直な注記](#速度の正直な注記)」参照)。sm_120 以外のカードは
`H3_ATTN_BACKEND=default` も必要。

```bash
# 例: 32GB級GPUで起動する
H3_LOWVRAM=group H3_TE_PRUNE=1 venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
# 例: 16GB カード単体で起動する(sm_89 なら H3_ATTN_BACKEND=default も付ける)
H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

主な環境変数(既定値。多くは**UIからも切り替えられる**ので、恒久的に変えたい時だけ指定する):

| 変数 | 既定 | 意味 |
|---|---|---|
| `H3_PROFILE` | `auto` | 2026-11-17 追加。主プロファイル系の変数が未指定なら総 VRAM から 96/80/48/32/16/8GB 級の既定値一式を入れる(`core/profile.py`)。32GB 以下の級では turbo LoRA(ref2v / fl2v の 4 ステップ)も ON にする(group では起動時に巻く必要があるため)。`file` ストリーミングでも可(LoRA 約 1.2GB はホスト RAM から単位ごとに流す、2026-11-18)。`96gb`〜`8gb` で強制、`off` で無効。sage が使えなければ `H3_ATTN_BACKEND=default` も入れる |
| `H3_TE_QUANT` | `bnb-4bit` | text_encoder の量子化(`none` は bf16 で 66.7GB) |
| `H3_TE_PRUNE` | `0` | TE の未使用上位レイヤー削除(出力は不変、-3.6GB) |
| `H3_TE_PROJ` | (無効) | **投影TE**: 32B TE を Qwen3-VL-4B+線形写像で代替(repo id か .safetensors パス。品質は近似、日付節 2026-08-10 参照) |
| `H3_TE_PROJ_QUANT` | `bnb-4bit` | 投影TE 4B の量子化(NF4 で 3.11GB。`none`=bf16 8.88GB / `bnb-8bit`) |
| `H3_TE_DEVICE` | (無効) | TE を2枚目GPUへ常駐(例 `cuda:1`。32B TE は 20GB 級、投影TE なら 8GB 級で可) |
| `H3_TRANSFORMER_QUANT` | `none` | `int8` で transformer を 66.3→34GB |
| `H3_PRUNED` | `0` | ref2va の transformer_ref を AdaLN-pruned 版(`multimodalart/MiniMax-H3-Pruned`)で読む(常駐 34→約21GB、`H3_TRANSFORMER_QUANT=int8` 併用必須。HyperFlow/`H3_LOWVRAM=group`/`H3_ADALN_PRECOMP` とは非互換) |
| `H3_PRUNED_QUANT` | `int8wo` | pruned の量子化方式: `int8wo`(既定、4.60 s/step・peak 28GB)/ `int8dyn-convrot`(旧既定、7.60 s/step)/ `int8wo-convrot` / `fp8` / `fp8-convrot` / `bf16`。方式ごとに別キャッシュ(`models/prequant/transformer_ref_pruned_*`) |
| `H3_LOWVRAM` | `0` | `1`=48GB級のフェーズ循環 / `group`=32GB級以下の block offload |
| `H3_KEEP_TRANSFORMER` | `0` | transformer 常駐で再ロード固定費を撤廃(成立条件は該当節参照) |
| `H3_CACHE` / `H3_CACHE_THRESHOLD` | `fbc` / `0.05` | FirstBlockCache(デノイズ -25%。int8+SDPA 軌道では不発の実測あり) |
| `H3_ATTN_BACKEND` | `sage` | sage は sm_120 専用ビルド。**それ以外のGPUは `default`(SDPA)必須** |
| `H3_TURBO_LORA` | `0` | 4/8ステップ蒸留LoRA(初回に約780MBをDL)。2026-11-17 から group でも可(`H3_GROUP_STREAM=own` + 起動時指定。`file` ストリーミングでは当初 LoRA 約 1.2GB が GPU 常駐で、8GB 級 + 8GB 上限の模擬で 768² 205 秒・4.19GB、1024² 376 秒・4.85GB、ref2va 768² 276 秒・4.13GB。同日 15 時にホスト RAM から単位ごとに流す方式に変え(`H3_FILE_STREAM_HOST_MIN_MB`)、6GB 上限の模擬で 768² 211 秒・2.82GB、1024² 380 秒・3.48GB。実機の Windows 11 + RTX 4060 Laptop 8GB / RAM 16GB と Windows 10 + RTX 3050 6GB で動作確認済み(同じ入門機に RTX 3060 12GB + NVMe: 512² 132 秒(デノイズ 83 + デコード 36)、768×512 198 秒、768² 338 秒、1024×768 473 秒。3050 / RAM 32GB・SATA 級 SSD 約 430MB/s、2026-11-19: 512² 321 秒 = デノイズ 237 + デコード 65、512×384 294 秒。SATA 級 SSD ではデノイズがディスク読みで決まる。NVMe に替えた実測: 512×384 163 秒(デノイズ 108 + デコード 42)、512² 237 秒(147 + 65)、768×512 345 秒(248 + 85)、768² 558 秒(444 + 112)。共有 GPU メモリ 1.5GB のまま、デノイズは GPU 100% で、ここから先はカードの性能で決まる。実用は 512²〜768×512。4060 Laptop の実測、2026-11-19、NVMe: 512² 194 秒 = デノイズ 117 + デコード 16、768×512 279 秒 = 191 + 65、768² 412 秒 = 313 + 85。デノイズは重みの読み出しと GPU、デコードは GPU 計算が律速)。A4000 16GB で t2va 5 秒 768² が 4 ステップ 151 秒。8GB 級の指定 + 8GB 上限の模擬で 768² 209 秒・3.08GB、1024² 376 秒・3.73GB、ref2va 768² 299 秒・3.0GB、fl2va 768² 225 秒・3.4GB。なお自作ストリーマの group モードでは ref2va が layout の device 不一致で落ちる退行があり、同日に修正) |
| `H3_ALLOW_WEIGHT_DOWNLOAD` | `0` | transformer / transformer_ref の bf16 重み(各 ~66GB)がキャッシュに無いとき、`1` のときだけ HF から取得する(既定は取得せず、エラーで案内する) |
| `H3_PIKA_RENOISE` | `0` | Pika H3 Flex の turbo LoRA 用 renoise サンプラー(実験) |
| `H3_BLOCK_COMPILE` | `0` | `1` で turbo LoRA 適用後に transformer_ref の 50 ブロックを torch.compile(attention 本体は graph break)。RTX PRO 6000・1024×768 で 1 step 約 -13%、初回のみ +5〜14s |
| `H3_LORA_FUSE` | `0` | turbo LoRA の要素ごとの演算を減らす: `1`=scale 1.0 の乗算を省く(ビット一致)/ `2`=加算を GEMM に融合。`H3_BLOCK_COMPILE` 併用時は上乗せなし |
| `H3_VIDEO_VAE_FP16` | `0` | video VAE を fp16 化(デコードのピークを削減。16GB 以下では必須) |
| `H3_LLM_URL` | `http://127.0.0.1:64660` | プロンプト強化に使うローカルLLM(任意機能) |
| `H3_LLM_API_KEY` | なし | LLMサーバのAPIキー(`Authorization: Bearer` で送る)。環境変数が無ければリポジトリ直下の `config.local.env`(git管理外)の同名行を読む |
| `H3_LLM_LOCAL` / `H3_LLM_LOCAL_GGUF` | `auto` / なし | 2026-11-17 追加。外部 LLM に接続できないとき CPU 上の内蔵 LLM(Gemma 4 E4B、GGUF Q4_0)へフォールバック。実体は llama.cpp のビルド済み `llama-server` の子プロセス(`core/local_llm.py` が自動取得・起動、コンパイラ不要)。`1` で内蔵のみ、`0` で無効。`H3_LLM_LOCAL_IDLE_S`(既定 300)で未使用時に停止、`H3_LLM_LOCAL_STOP_ON_GENERATE`(auto = RAM 32GB 未満)で生成開始時に停止 |

### 2026-10 更新: 24GB/32GB 単騎でリアルタイム会話生成

8月の初版以降、会話アプリ用途(ref2va 連続生成)に向けた大幅な高速化・低VRAM化を
本番リポジトリ(diffusers-movie-server)側で行い、本リポジトリへ再同期した。到達点:

| 構成 | 条件 | 結果 |
|---|---|---|
| **32GB 単騎(RTX 5090 級)** | 320×448・141f(5.875s 分) | **5.57s = 実時間の 0.95 倍(リアルタイム)**、peak 25.6GB |
| 24GB 単騎(RTX 4090 級) | 320×448・141f | 7.1s = 1.21 倍(短い応答なら実用圏)、peak 20.4GB |

主な追加技術:
- **ck-w4a8 量子化**(`H3_PRUNED_QUANT=ck-w4a8` / `H3_TRANSFORMER_QUANT=ck-w4a8`):
  ComfyUI の W4A8 int8 カーネル(pip の `comfy-kitchen==0.2.37`、要インストール)。
  AdaLN-pruned(multimodalart)と併用で 33B が**常駐 12.5〜14.9GB・denoise -28%・品質目視同等**
- **pinned 退避**(`H3_VAE_PINNED` / `H3_BASE_PINNED` / `H3_REF_PINNED`):
  モデルの退避を pinned メモリのポインタ付替えに(退避 0.3s / 復帰 2s)
- **TE 削減**(`H3_TE_DIET` / `H3_TE_STREAM`)、量子化済みキャッシュ、参照 latent/prefix キャッシュ
- lightx2v turbo 4step 蒸留(`H3_TURBO_LORA=1`)

単騎実行の環境変数一式(32GB 級。24GB 級は `H3_KEEP_REF2VA_VAE=0` に変える):

```bash
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_VIDEO_VAE_FP16=1 H3_TRANSFORMER_QUANT=ck-w4a8 H3_BASE_PINNED=1 H3_PRUNED=1 H3_PRUNED_QUANT=ck-w4a8 H3_REF_PINNED=1 H3_FL2VA_KEEP_TE=1 H3_KEEP_REF2VA=1 H3_KEEP_REF2VA_VAE=1 H3_REF_LATENT_CACHE=1 H3_DECODE_STREAM=1 H3_DECODE_VAE=light H3_VAE_PINNED=1 H3_TE_DIET=1 H3_VAE_SPLIT=1 H3_REF_PREFIX_CACHE_SINGLE=1 H3_REF_PREFIX_PARK=1 H3_TE_STREAM=1 H3_TURBO_LORA=1 ./run.sh
```

**注意点(必読)**:
1. **GPU 専有前提**: 空き VRAM 23GiB(24GB級)/ 28GiB(32GB級)以上。TTS・LLM は同じ GPU に同居不可
2. **初回キャッシュ作成**: ref 側は multimodalart/MiniMax-H3-Pruned の bf16(~38GB DL)から
   自動量子化(**空きホスト RAM 45GB 必要**。24GB カード上での初回量子化は未検証)。
   base 側は Kijai/MiniMax-H3-experimental の `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
   (12.5GB)を `scripts/convert_kijai_w4a8.py` で変換(GPU 不要)
3. **ホスト RAM**: pinned 常駐 ~45GiB。**RAM 64GB 機は `H3_REF_PINNED` か `H3_TE_STREAM` を外す**
4. **24GB で `H3_KEEP_REF2VA_VAE=1` にしないこと**(発話⇄待機のプロンプト切替で OOM を実測)

会話アプリとしての実例は [Realtime_Narration_Video](https://github.com/animede/Realtime_Narration_Video)(+ 音声会話クライアント [Realtime_Conversation_Video](https://github.com/animede/Realtime_Conversation_Video))を参照。

### VRAM/RAM 設計 (重要、実測に基づく)

このマシンは VRAM 96GB に対し RAM は 94GB。**text_encoder は bf16ネイティブ配布で
実測66.73GB**(fp32配布を仮定した「bf16化で半分の33GB」という当初推定は誤りだった)。
transformer bf16 66.3GB / vae+audio_vae fp32 計11GB と合わせると約144GBになり、
**VRAMにもRAMにも同時に載らない**。diffusers の
`ComponentsManager.enable_auto_cpu_offload()`(全コンポーネントを定常的にRAM常駐させ、
アクティブな1つだけをGPUへ出す方式)は RAM 94GB では成立しないため採用していない。

TEのロード方式は環境変数 `H3_TE_QUANT` で選択する(既定 `bnb-4bit`、2026-08-04 に
A/B検証して既定化)。

#### `H3_TE_QUANT=bnb-4bit` (既定)

text_encoder を起動時に NF4(bitsandbytes、compute_dtype=bf16)へ量子化して
**GPU常駐のまま維持する**(bnb 4bitモデルはデバイス間移動不可のため常駐一択)。
実測サイズ **21.0GB**(当初推定の~17-18GBより大きい)。transformer(66.3GB)も常駐し、
リクエストごとの TE⇔transformer 入れ替えが消滅する。

- 定常常駐: transformer + TE-nf4 = **~87.5GB**。これに VAE 11GB を足すと~98.5GBで
  96GBを超えるため、**このモードでは VAE ペアは常駐しない**(CPUに置き、キーフレーム
  エンコード/デコードの当該フェーズだけGPUへ往復する。fp32 11GBのPCIe往復のみで
  ディスクI/Oなし)。
- デコード窓(~9s)だけは transformer を解放してから実行し、直後にリロードする
  (transformer+TE+VAE+デコードバッファは物理的に収まらないため。実測でOOM確認済み)。
  毎stepのスワップではなく単発の片道×2なので、diffusers-server CLAUDE.md 33番の
  禁止パターンには当たらない。
- **品質A/B(同一seed 12345)**: フレーム比較で構図・被写体・シャープさは同等
  (条件付け数値の変化による木立の配置等の微差のみ)、音声も rms 0.0080→0.0061 と
  同水準で -20dB 型の崩壊なし。**劣化なしと判定して既定化した**。

#### `H3_TE_QUANT=none` (bf16 TE、旧方式)

**2つの66GBモデルをリクエストごとにGPU上で入れ替える**:

- `vae` + `audio_vae`(fp32 計11GB)は常時GPU常駐。
- エンコード段階: [VAE 11GB + text_encoder 66GB](transformerが常駐していれば先に解放)
- デノイズ/デコード段階: [VAE 11GB + transformer 66GB](エンコード直後にTEを解放)
- 解放は **CUDAモデルの参照を直接落とす**(`.to("cpu")` でRAMへ退避しない。66GBの
  RAM経由はスワップ突入の実測原因になった)。リロードはページキャッシュ/ディスクから
  11〜40秒/モデル。リクエスト間の定常状態は transformer+VAE 常駐(77.5GB)。

**実測のオーバーヘッド**: 1リクエストあたり TEロード ~37s + transformerリロード ~26s。
この入れ替えコストを解消したのが上記の `bnb-4bit`(既定)で、リクエスト合計は
245s → **185s** に短縮された(デノイズ157sは共通のモデル律速で不変)。

**2つの実装上の罠(実機で踏んで修正済み)**:
1. `MiniMaxH3TextEncoderStep.encode_prompt` は素の staticmethod で、`@torch.no_grad()`
   はブロックの `__call__` 側にしか付いていない。直接呼ぶ場合は必ず `torch.no_grad()`
   で包むこと。忘れると autograd グラフが TE の重み約50GB分をGPU上にピン留めし、
   モデルを解放してもVRAMが返ってこない(diffusers-server CLAUDE.md 39番と同型)。
2. ブロックの出力(`num_frames`・`keyframes`・latent形状等)は `PipelineState` に入る。
   `get_block_state()` は宣言された入力しかマップしないので、出力は `state.get(名前)`
   で読むこと。

video VAE の decode は diffusers 側の `MiniMaxH3VideoDecodeStep` が内部で
`torch.autocast(dtype=torch.float16)` を使うため、重み自体は fp32 のままでよい。
**audio VAE は fp32 のまま一切キャストしないこと**(bf16化すると生成音声の音量が
約20dB小さくなる既知の問題があるため、`runner.py` は `vae`/`audio_vae` のロードに
明示的に `dtype=torch.float32` を渡している)。

### 実測値 (RTX PRO 6000 Blackwell 96GB, 768×768, 124フレーム=5.17秒, 30steps)

| 項目 | 実測 |
|---|---|
| DLサイズ(T2VA必要分のみ) | 135GiB (HFキャッシュ実測) |
| text_encoder ロード | 37.6s (コールド) / 15.9s (ページキャッシュ温) |
| transformer ロード | 37.7s (コールド) / 10〜26s (温) |
| vae+audio_vae ロード | 10.0s |
| プロンプトエンコード | 0.7s |
| デノイズ (30steps) | 157〜159s (約5.4s/step, GPU 100%/600W) |
| VAEデコード (video+audio) | 6.5〜9s |
| ピークVRAM (生成中) | none: 83.4GB (デコード時。デノイズ中70.4GB) / bnb-4bit: 91.7GB |
| リクエスト合計 (サーバAPI経由、ロード込み) | none: 245s / **bnb-4bit(既定): 185s** |
| RAM | 使用~6.5GBで安定、スワップ増ゼロ |

bnb-4bit のピーク91.7GBは96GBカードに収まるが余裕は約4GB。ヘッドルームを優先したい
場合は `H3_TE_QUANT=none` で旧方式(ピーク83.4GB、+60s/リクエスト)に戻せる。

### 48GB級VRAM対応 (`H3_LOWVRAM`、既定 `0`)

TE-nf4(21GB)+ transformer int8(34GB)= 55GB は48GB級カードでは同時常駐できない
(96GB機の既定・int8両常駐モードいずれも成立しない)。`H3_LOWVRAM=1` は
「TEとtransformerを絶対に同時常駐させない」フェーズ循環方式で48GB級に対応する。

- **強制**: `H3_TRANSFORMER_QUANT` 未指定なら自動で `int8` に上書きする(bf16
  66.3GBは48GBに単体でも収まらないため)。明示的に `H3_TRANSFORMER_QUANT=none` を
  指定した場合は起動時に `RuntimeError` で拒否する。`H3_TE_QUANT` は `bnb-4bit`
  (既定)以外を指定するとやはり起動時に拒否する。`H3_TRANSFORMER_BOTH_RESIDENT`
  (int8両常駐)は無条件で無効化する。
- **定常状態**: リクエスト間は「何も大きいものが常駐しない」(VAEペアのみCPU常駐、
  他モードのような transformer/TE の常時居座りが無い)。
- **t2va/fl2vaのフェーズ**: [エントリ: 常駐transformer/transformer_refがあれば解放]
  → TEロード → エンコード(+ fl2vaならkeyframeエンコード)→ **layout/latents/
  timestepsをTE常駐のまま先に実行**(`_execution_device`解決の罠対策、下記参照)→
  TE解放 → transformer(int8)ロード → デノイズ(~34GB+活性化~5GB≒39GB)→
  transformer解放 → VAE→GPU → デコード(~11GB+バッファ)→ VAE→CPU
  (**transformerは次リクエストのために再ロードしない** — 次は encode が先に必要)。
- **ref2vaのフェーズ**: 同じ原則で、参照VAEエンコード → layout/latents/timesteps
  (ここもTE常駐のまま)→ TE解放 → transformer_ref(int8)ロード → デノイズ → 以下同様。
- **`_execution_device` 解決の罠(実装時に発見・修正)**: パイプラインのコンポーネント
  順は `text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler,
  transformer, ...`。TEを先に解放してからtransformerをロードする素朴な実装だと、
  `vae`(CPUに退避されていても“存在する”nn.Module)が`text_encoder`の次に解決され、
  `_execution_device`が`cpu`に解決されてしまいデノイズの最初のtransformer forward
  で `RuntimeError: Expected all tensors to be on the same device` になる
  (実機で再現・特定)。対策として **layout/latents/timesteps は TE がまだ
  GPU常駐のうちに実行し、それらの出力テンソルが正しいデバイスに確定してから
  TEを解放する** 順序にした(他モードの `force_free_te` 遅延パターンと同じ発想)。
  ref2vaでは reference_encoder_step も同様にTE常駐のうちに実行する必要がある
  (layout_stepが参照の latent 形状に依存するため)。
- **upscale=1(hires-fix)は非対応**: パス2は約4倍の系列長(~16倍のattention活性化
  コスト)を要し、このモードの定常余裕(~9GB)では未検証のため、`ValueError`
  (400)で拒否する。
- **副次的に見つけた既存バグの修正**: `generate_ref2va()` の
  `_sync_shared_components_to_ref()` はTEロードより**前**に呼ばれていたため、
  TEが未ロードの状態(H3_LOWVRAM、または`none`系モードでref2vaが最初のリクエスト
  になるケース)では `self._pipe_ref.text_encoder` に `None` が同期され、
  `AttributeError: 'NoneType' object has no attribute 'config'` になっていた
  (実機で再現・特定)。TEロードの**後**に呼ぶよう順序を修正した(全モード共通の
  修正、H3_LOWVRAM専用ではない)。
- **正しさの検証**: フェーズ循環は計算内容を変えないはずなので、同一seed
  (768²・5秒・30steps・キツネのプロンプト、seed=12345)で `H3_LOWVRAM=1
  H3_TRANSFORMER_QUANT=int8` のt2va出力と通常int8モードのt2va出力を比較したところ
  **mp4がバイト完全一致**(md5一致)することを確認済み。
- **48GB相当の実機検証**(96GB機でダミーVRAM確保により空きを~43.5GBへ制限、実48GB
  カードの空き~47GBより厳しい条件):

  | | 完走 | ピークVRAM | 内訳(概算) |
  |---|---|---|---|
  | t2va 1本目 | ○ | 38.68GB | TEロード~52s + transformerロード~36s + デノイズ108s + デコード6.6s |
  | t2va 2本目(連続) | ○ | 38.94GB | 同様の固定費が毎回発生(定常状態を持たない設計どおり) |
  | ref2va(画像参照1枚) | ○ | 43.84GB | デノイズ283s(参照行分シーケンスが伸びるため39GB台では収まらずやや高め) |
  | upscale=1 | - | - | 実装時点でOOMリスクを判断し明示的に400で拒否(未実行) |

  いずれもホストRAMスワップの増加なし(`free -g` で作業前後とも Swap used ~6GB台で
  安定)。作業後は `H3_LOWVRAM` 未指定(完全デフォルト設定、bf16 transformer)で
  再起動して同一プロンプトのt2vaを1本実行し、`peak_vram_gb: 91.94GB` /
  `cache_skipped_steps: 6` など既存の実測値(本README上部の表・
  int8量子化セクション)と同水準であることを確認し、回帰なしと判定した。
- ~~**量子化済みチェックポイントの事前保存によるロード時間短縮**: 未調査~~
  → **TE は 2026-08-08 に実装済み**(`H3_TE_PREQUANT`、既定ON。TEロード 53.0s→29.5s、
  リクエスト合計 -35%、MD5一致で等価性確認済み。上記「量子化済み text_encoder の
  ディスクキャッシュ」節を参照)。transformer int8 の直列化は**未検証のまま** —
  保存に約34GBを要し、この箱のディスク空き(43GB)では採用自体が非現実的なため。

### 24〜32GB級VRAM対応 (`H3_LOWVRAM=group`、2026-08-05追加)

#### 2026-11-15: ブロック転送を自作の窓付き先読みに置き換え(`H3_GROUP_STREAM`、既定 `own`)

`H3_LOWVRAM=group` のブロック転送は、diffusers の `enable_group_offload(block_level, use_stream=True)` から
`core/blockstream.py` の自作ストリーマに置き換えた(`H3_GROUP_STREAM=diffusers` で従来に戻せる)。
TE の `H3_TE_STREAM`(`core/testream.py`)と同じ設計で、50 ブロック(int8、計 30.1GiB)をピン留めした
ホスト RAM に置き、コピー専用 stream で window 個(`H3_GROUP_OFFLOAD_BLOCKS`+1、既定 2)先のブロックを
送りながら計算する。ブロック以外(埋め込み・token_refiner・出力層、bf16 で 1.6GiB)は GPU に常駐。
torchao `Int8Tensor` は外側の wrapper ではなく内側の `qdata` / `scale` / `zero_point` を差し替える
(diffusers の `_restore_torchao_tensor` と同じ方式)。flux3-klein-studio の 6GB 対応で実証した設計の移植。

実測(RTX PRO 5000 Blackwell 48GB の GPU0、`H3_LOWVRAM=group H3_TE_PRUNE=1`、t2va 512²×5 秒 124f、30 step、seed 1、2 回ずつ):

| | diffusers 版(`H3_GROUP_STREAM=diffusers`) | 自作版(`own`、既定) |
|---|---:|---:|
| デノイズ、`H3_CACHE=none` | 164.9〜165.0 秒(5.68 秒/step) | **96.4〜96.5 秒(3.32 秒/step)** |
| デノイズ、`H3_CACHE=fbc`(既定) | 165.5〜167.6 秒(FBC が 1 step も飛ばない) | **85.6〜86.1 秒(FBC が 6/30 step を飛ばす)** |
| ピーク VRAM | 17.7〜18.3GB | 19.5〜20.0GB(ブロック以外の常駐 +1.6GiB) |
| 出力(MD5) | `H3_CACHE=none` で自作版と**完全一致** | 同左 |
| ロード(ピン留め) | 20 秒(34GB を pinned 辞書へ) | 15 秒(ブロックごとに 1 本のバッファへ) |

- 速さの差は転送方式そのもの(FBC オフ同士で 1.71 倍)。既製品はグループごとのフック固定費と、CPU が計算に先回りして
  GPU 側の解放が遅れる構造のため(`docs/lowvram-techniques.md` の 4.2 節と同じ)。
- **FirstBlockCache が group モードでも効くようになった**。diffusers 版では FBC が 1 step も飛ばしておらず
  (「速度の正直な注記」の `cache_skipped_steps: 0` はこれ)、自作版では閾値 0.05 で 30 step 中 6 step を飛ばす。
  その分だけ出力は FBC オフと変わる(FBC 本来の挙動)。完全一致が要るときは `H3_CACHE=none`。
- group モードの transformer は、量子化済みキャッシュ(`models/prequant/transformer_int8`)があればそれを CPU に読む
  ようにした(bf16 原本が HF キャッシュに無くても起動できる。他モードと同じ fail-open)。キャッシュから読んだ
  Int8Tensor は `requires_grad=False` に固定する(diffusers 版の `swap_tensors` が autograd を辿って
  `aten.view unimplemented` で落ちるため)。bf16 原本を消した箱では `meta.json` の `source_snapshot` の照合を省略する。
- 解放は `BlockStreamer.release()`(`cudaHostUnregister`)で行い、`_host_emptyCache` に頼らない。
- ホスト RAM: ピン留め 30.1GiB(diffusers 版の pinned 辞書 34GB と同程度)。RAM ガード(`H3_GROUP_OFFLOAD_MIN_RAM_GB`)はそのまま。

#### 2026-11-15: 低 VRAM の到達点 — 16GB 構成を 8GB / 6GB へ(3 段のストリーミング)

同じ設計を、group モードで GPU に残っていた残り 3 つにも当てた。いずれも配置換えだけで**出力は MD5 一致**。

1. **投影 TE の LM 層**(`H3_TE_STREAM=1` が `H3_TE_PROJ` にも効くようにした)。4B の 36 層 1.9GiB をピン留めホストに置き、
   エンコード中だけ流す。group モードは TE をデノイズ中も常駐させる段取りなので、デノイズのピークが下がる。
2. **token_refiner の 2 ブロック**(`H3_GROUP_STREAM_REFINER`、既定 1)。int8 化の対象外で bf16 1.5GiB が常駐していた。
3. **video VAE デコーダの 36 ブロック**(`H3_GROUP_STREAM_VAE`、既定 `auto` = 最初のデコードで空き VRAM が 8GB 未満なら流す)。
   VAE 5.2GiB(fp16)の 4.8GiB がデコーダで、1・2 のあとはここがピークだった。タイルごとに送り直すのでデコードは遅くなる。

実測(RTX PRO 5000 48GB の GPU0、`H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1 H3_TE_STREAM=1`、t2va 5 秒 30 step、seed 1):

| | 基準(自作ストリーマのみ) | +1 投影 TE | +2 token_refiner | +3 VAE デコーダ |
|---|---:|---:|---:|---:|
| 512²×5 秒 ピーク | 8.59GB | 8.33GB | 6.79GB | **4.67GB** |
| 768²×5 秒 ピーク | 10.78GB | 9.54GB | 7.99GB | **7.45GB** |
| 512² デノイズ / デコード | 90 / 6 秒 | 86 / 6 秒 | 90 / 6 秒 | 89 / 17 秒 |
| 768² デノイズ / デコード | 200 / 11 秒 | 193 / 11 秒 | 195 / 11 秒 | 195 / 30 秒 |

起動直後の GPU 常駐は 0.18GB。残るピークはデノイズの活性で、これは解像度と長さで決まる。

4. **feed-forward のトークン分割**(`H3_FF_CHUNK`、既定 `auto` = group モードで 8192 トークン、`0` で無効)。ブロックで最大の
   一時メモリは SwiGLU の射影(トークンごとに 2 × ffn_dim 14336 の bf16 = 57KB)で、トークンを 8192 個ずつ通すと
   一時メモリがその分だけになる。トークンごとの演算なので結果は同じ(MD5 一致)。

| 1K を含む活性のピーク(デノイズ段階、同構成) | 分割なし | `H3_FF_CHUNK=8192` |
|---|---:|---:|
| 768²×5 秒 | 7.45GB | **6.26GB** |
| 1024²×5 秒(3.8 万トークン、13.9 秒/step) | 10.73GB | **9.34GB** |

残るのは attention の K・V(1024² で各 0.55GB)と出力・残差ストリームで、正確なままでは削れない下限に近い。

5. **attention のクエリ分割**(`H3_ATTN_Q_CHUNK`、既定 0 = 無効、`core/attnchunk.py`)。K・V は丸ごと作り、Q の射影・
   QK ノルム・RoPE・カーネル・出力射影を 8192 トークンずつ回す(結果は MD5 一致)。効くのは 1K 以上で、
   1024²×5 秒のデノイズのピークが 9.34 → **8.53GB**、768² では差なし(6.26 → 6.38)。2〜4% 遅くなる(1024² で
   13.9 → 14.4 秒/step)ので既定では切ってあり、1K を小さいカードで回すときに `H3_ATTN_Q_CHUNK=8192` を足す。

ここまでで 1024²×5 秒は 8.5GB。さらに 3 つで「後ちょっと」を詰めた(いずれも MD5 一致)。

6. **メモリ節約版のブロック forward**(`H3_LEAN_BLOCK`、既定 `auto` = group モードで 8192 トークン、`core/leanblock.py`)。
   元の forward は norm1 の出力・attention の出力・norm2 の出力・FF の出力を全トークン分(1024² で各 0.41GB)作ってから
   足す。正規化 → 変調 → 射影/attention/FF → ゲート → 残差加算をチャンクで通し、全トークン分の中間は入力・出力・
   K・V の 4 本だけにする。4(FF 分割)と 5(クエリ分割)を包含する。norm をチャンクごとに 2 回計算する分、1024² で約 7% 遅い。
7. **流す単位をブロックの 1/3 に**(`H3_GROUP_STREAM_UNITS=half`、既定)。adaln_proj → attn → ff の順に 150 単位で流し、
   常駐する先読み分が 2 ブロック(1.3GiB)から小さい 2 単位になる。
8. **投影 TE にも `H3_TE_DIET=1`**(opt-in)。lm_head の解放と埋め込み表(151936 × 2560)の CPU ブリッジ化で 1.45GiB。

| ピーク VRAM(同構成 + `H3_TE_DIET=1`、VAE を流した小カード相当の総合値) | 最初(自作ストリーマのみ) | 1〜3(ストリーミング 3 段) | 全部(4〜8) |
|---|---:|---:|---:|
| 512²×5 秒 | 8.59GB | 4.67GB | **3.27GB** |
| 768²×5 秒 | 10.78GB | 7.45GB | **4.85GB** |
| 1024²×5 秒 | 10.73GB(デノイズ) | — | **6.82GB** |

ここまでが午前。午後に 1024²×5 秒の内訳を測り直す(`H3_LEAN_DEBUG=1` がブロック 0〜2 の allocated / peak を出す)と、
トークンは 38,321(動画 31,744 + 音声 + テキスト)で、残差ストリーム 1 本が 0.41GB、K・V が各 0.55GB、ブロック内の一時
メモリは attention 段が最大(K・V 1.1GB + sage がカーネル内で作る int8 の写し + チャンク分)だった。さらに 4 つ。

9. **fp32 パッチ埋め込みの丸め**(`core/leanblock.py` の `shrink_prologue`、`H3_LEAN_BLOCK` と一緒に入る)。transformer の
   forward は `proj_in`(fp32 のまま残す層)の出力を全トークン分作って bf16 に丸めて詰めるが、丸める前の fp32 の方が
   ローカル変数としてブロックのループ中ずっと残る(1024² で 0.68GB)。射影の出口で同じ丸めをしてから返すので値は
   同じまま、残るのが半分になる。1024² 6.82 → **6.41GB**、768² 4.85 → **4.62GB**(MD5 一致)。
10. **残差の in-place 更新**(同)。ブロック 2 以降は入力バッファに直接書く(各チャンクは自分の行を読み終えてから書く
    ので結果は同じ。MD5 一致)。FBC はブロック 0 の入力と出力を参照するのでブロック 0・1 は新しいバッファに書き、
    FBC が持つ残差 3 本(head_block_output / head_block_residual / tail_block_residuals、1024² で計 1.2GB)と合わせると
    ピークはブロック 1 のまま → FBC 有効時の総合値は変わらない。`H3_CACHE=none` なら全ブロック in-place で 1 本下がる。
11. **ヘッド分割の attention**(`H3_ATTN_HEAD_GROUPS`、既定 `auto`)。K・V・Q を 56 ヘッドの G 分割で順に作る(to_q/to_k/to_v
    の重みの行スライス。torchao の Int8Tensor はストリーマが中身だけ GPU に置くので、中身をスライスして組み直す)。
    各グループの attention 出力は全ヘッド分の 1 本(attn_out)に書き、出力射影はそこから全ヘッドまとめて行う。同時に
    持つのが K・V の 2 本から attn_out + K/G + V/G になり、sage がカーネル内で作る K・V の写しも 1/G になる。
    正規化と変調を G 回やり直すので遅い(G=4 +13%、G=8 +15〜35%、G=28 +70%)。**計算は同じだがビット一致はしない**:
    GEMM の行数(7168/G)と attention カーネルのヘッド数が変わるとカーネル選択と加算順序が変わり、bf16 の最下位
    ビットが動くことがある(sage は K の平均を引く前処理がヘッド数とトークン数の組み合わせで揃ったり揃わなかったり
    する)。30 ステップで増幅されて別のサンプル相当の動画になる(基準との PSNR は 1024² 34.4dB / 768² 38.7dB。FBC の
    有無の差が 34.2dB なので、既定で使っている FBC の近似と同程度の差で、品質低下ではない)。そのため **auto は G=1 で
    収まるときは必ず G=1** を選び、収まらないときだけ 4 → 7 → 8 → 14 → 28 の順に空き VRAM(`H3_VRAM_LIMIT_GB` の上限も
    見る)に入る最小の G を使う(生成の最初の forward で決めて、その生成の間は固定。FBC の残差 3 本と 1GiB の余裕を先に引く)。G=4 固定で 1024² 5.09GB(16.7 秒/step、+8%)、G=8 固定で 1024² **4.79GB**(17.9 秒/step)、768² **3.95GB**
    (8.8 秒/step)。5.8GB 上限で auto に任せると G=4 → 28 を選んで 4.79GB(26 秒/step)。ブロック単位の診断で一致した
    組み合わせ(1024² の G=4)でも 30 ステップの実機出力は MD5 が変わったので、G>1 は常に「別サンプル」と考える。
12. **FBC を切る**(`H3_CACHE=none`)。FBC の残差 3 本が消え、全ブロック in-place になる。G=8 と合わせて 1024²×5 秒の
    デノイズが **3.03GB**(30 ステップ全部計算するので 23 秒/step。FBC 自体が別サンプル相当の近似なので出力は変わる)。

| ピーク VRAM(同構成、総合値) | 〜8(午前) | 9〜10(MD5 一致、既定) | + ヘッド分割 G=8(別サンプル) | + `H3_CACHE=none` |
|---|---:|---:|---:|---:|
| 512²×5 秒 | 3.27GB | **2.90GB** | — | — |
| 768²×5 秒 | 4.85GB | **4.62GB** | 3.95GB | — |
| 1024²×5 秒 | 6.82GB | **6.41GB** | 4.79GB | 3.03GB |

正確なまま(MD5 一致)では **512² 2.90GB / 768² 4.62GB / 1024² 6.41GB** が下限で、残りは K・V と残差ストリームと FBC の残差 3 本。
ヘッド分割を許せば **1024²×5 秒が 6GB カード**に入る(auto が必要なときだけ使う)。`/api/status` の
`group_stream_refiner` / `group_stream_vae` / `group_stream_units` / `ff_chunk` / `attn_q_chunk` / `lean_block` /
`attn_head_groups` と、直近の forward が選んだ `lean_block_last`(トークン数・G・in-place の開始ブロック)で状態を
確認できる(生成結果の JSON にも `lean_block_last` が入る)。

#### ホスト RAM 32GB / 16GB の PC: 重みをファイルから直接流す(`H3_GROUP_STREAM_HOST=file`、2026-11-15)

ここまでの low VRAM 構成は int8 transformer 全体(30GiB)をピン留めしたホスト RAM に置くので、RAM 48GB 級が前提だった。
`H3_GROUP_STREAM_HOST=file`(`core/filestream.py`。minimax-h3-image-server の `app/filestream.py` と同じ方式)は、
量子化済みキャッシュ(safetensors、`from_pretrained` が mmap で読むので RAM に展開されない)から単位(ブロックの 1/3)ごとに
8 スレッドで 32MB ずつ並列に読み、ピン留めした受け皿(3 本 × 0.24GiB、2 単位先読み)経由で GPU へ流す。必須のホスト RAM は
その受け皿と token_refiner の受け皿(2 × 0.36GiB)だけで、残りは OS のページキャッシュに乗る分だけ速くなる。
キャッシュに収まらない末尾の単位は読むたびに `posix_fadvise(DONTNEED)` で捨て、巡回アクセスで全部が LRU に押し出される
のを防ぐ(捨てる数は auto: 「原本 + この後ピン留めする TE / VAE の分 − 使える RAM(cgroup の上限も見る)」)。
file のときは VAE デコーダの流し元もピン留めしない(fp16 の写しをそのまま使う)。重みは同じ int8 なので出力は同じ。

- `H3_GROUP_STREAM_HOST`: `auto`(既定。搭載 RAM が `H3_GROUP_STREAM_FILE_BELOW_GB` = 48GB 未満なら file。**Windows は RAM に関係なく常に file**:
  2026-11-19、WDDM では数 GB のページロック領域の転送が病的に遅く、RTX 3050 6GB / RAM 32GB でデコーダ 4.5GB をピン留めして流すと
  0.5GB/s 相当・512² のデコード 590 秒、受け皿 0.4GB でファイルから流すと 1 パス 0.96 秒・53 秒だった。VAE デコーダの
  ピン留め経路も Windows では使わない)/ `pinned` / `file`。
  bf16 原本から量子化した(キャッシュが無い)ときは pinned に落ちる。`H3_GROUP_STREAM_DROP_UNITS`(auto / 数)、
  `H3_GROUP_STREAM_RING`(受け皿の本数、既定 3)。`H3_FILE_STREAM_HOST_MIN_MB`(既定 1): ファイルに無いテンソル(turbo LoRA の
  lora_a / lora_b、計 1.24GiB)のうちこのサイズ以上のものは GPU に常駐させず、ホスト RAM(ページャブル)に置いて単位ごとに
  受け皿へ memcpy して流す(2026-11-18。常駐だと 6GB のカードで Windows が共有メモリへ退避して極端に遅くなる、RTX 3050 で実測)。`/api/status` の `group_stream_host` / `group_stream_file`。
- Windows は `fadvise` が無いので捨てる制御は効かない(読み自体は `FILE_FLAG_SEQUENTIAL_SCAN` で動く。画像サーバの
  RTX 3050 / RAM 16GB 機で実証した方式)。2026-11-18: `FILE_FLAG_SEQUENTIAL_SCAN` は「RAM に収まらないもの」(30GB の
  transformer)にだけ付け、収まるもの(VAE デコーダ 4.5GB、投影 TE 1.9GB)は通常のキャッシュ付きで開く。前者を全部に
  付けていたため、RTX 4060 Laptop / RAM 16GB でデコード中にタイルごとの再読込が毎回 NVMe から走っていた
  (デコード 124 秒、NVMe 約 1GB/s で張り付き)。判定は「重み + 1GiB + 予約 ≤ 使える RAM」、ログの `page cache kept / not kept`。
- 同日: デコーダを **GPU に置き切る**経路(`H3_VAE_DECODER_RESIDENT`、既定 auto、`core/filestream.py` の `ResidentLoader`)。768²・5 秒の
  デコードは 7 クリップ × 16 タイル ≒ 150 パスで、タイルごとに 36 ブロック 4.5GB をホストから送り直す方式は RAM 16GB の PC では
  その転送(ページキャッシュ → 受け皿 → GPU)がデコード時間の大半だった(RTX 4060 Laptop: 1 パス 0.77 秒のうち 0.68 秒が読み待ち、
  デコード約 90 秒)。空き VRAM が「4.5 + 0.5 + 出力 + 余裕 0.5GiB」以上なら 1 回読んで GPU に置いたままにする。A4000 の 8GB 上限で
  強制して実測: デコード 86 → 34 秒、全体 201 → 152 秒、ピーク 7.03GB、出力ハッシュ一致。全部は入らないカードでは
  入るぶんのブロックだけ置いて残りを流す(`SplitDecoder`。余裕は既定 1.0GiB)。A4000 の模擬: 8GB 上限で 30/36 常駐、デコード
  33.8 秒・ピーク 6.22GB、全体 149 秒。6GB 上限で 15/36 常駐、デコード 51 秒・ピーク 4.21GB。いずれも出力ハッシュ一致。
  流す側が 0.75〜2.6GB に減るのでページキャッシュにも残りやすい(RAM 16GB の PC でデコードが 90〜300 秒と揺れていた原因は、
  4.5GB のデコーダがキャッシュに入ったり入らなかったりしていたこと)。
- 2026-11-19、試して採らなかったこと: VAE の空間タイルを 256px → 512px にしてパス数を 1/4 にする案(8GB のラップトップではデコードが
  GPU 計算律速で、これが唯一の削減余地だった)。A4000 の 8GB 上限で 768² のデコードは 86 → 38 秒になるが、**出力全体に格子状の
  アーティファクトが乗る**(256px 出力との PSNR 22dB、SSIM 0.60。境界だけでなく全面)。デコーダの transformer ブロックは 256px
  タイルを前提に学習されているとみられ、タイルサイズは変えられない。コードには入れていない。RTX 4060 Laptop(26/36 常駐、
  読み待ち 0)でデコード 85.7 秒 ≒ 0.57 秒/パス × 150 パスが、この GPU での下限。
- 2026-11-19: デコードの間は transformer / token_refiner / TE の受け皿(ピン留め、計 1.6GiB 前後)を返してホスト RAM を空け、
  次の forward の入口で作り直す(`FileBlockStreamer.park()`)。RAM 16GB の PC では、この分でデコーダの流す側がページキャッシュに
  残るかどうかが変わる(RTX 3050 6GB / RAM 16GB、遅い SATA 級 SSD、RAM 13/16GB 使用の実機で、512² のデコードが 772 秒と
  768² の 569 秒より遅くなった = 流す側 3.1GB がキャッシュから押し出され毎パス SSD から読んでいた)。出力は変わらない。
- 2026-11-19: 見積もりの数は上限として、実際に載せる間はドライバの空き VRAM(mem_get_info)を 1 単位ごとに見て、
  「活性 + 出力 + 余裕」を下回る手前で止める(`ResidentLoader(min_driver_free_gib=…)`)。torch が抱えた領域は断片化で
  使い回せないことがあり、見積もりだけで載せると物理 VRAM を越えて Windows が共有メモリへ退避し、極端に遅くなる
  (RTX 3050 6GB / RAM 32GB の実機: ドライバの空き 1.9 + torch 抱え 1.8GB の見積もりで 13/36 を載せた結果、512² の
  デコードが 824 秒、768² が 648 秒。共有 GPU メモリ 6.4GB)。
- 同日、さらに: Windows(WDDM)では `mem_get_info` の空きに他プロセスの退避可能な分が含まれることがあり、下限の判定を通っても
  物理 VRAM を越えていた(RTX 3050 6GB / RAM 32GB: ドライバの空き 2.2GB で 15/36 を載せ、タスクマネージャは専用 5.8 / 6.0GB、
  512² のデコード 708 秒 = 1 パス 4.5 秒以上の退避状態)。`nvidia-smi --query-gpu=memory.used,memory.total`(全プロセス込みの
  実使用、タスクマネージャの専用 GPU メモリと同じ)を UUID で突き合わせて取り、判定と下限の両方で「ドライバの空き」と
  「物理の空き」の小さいほうを使う(`filestream.physical_free_gib`)。nvidia-smi が無ければ従来どおり。
- 同日、決着: デコーダだけを新しいプロセスで回す切り分け用のプローブ(使い捨て、リポジトリには残していない)で RTX 3050 6GB を測ると、常駐なしで
  1 パス 0.96 秒・512² 全体 52.6 秒(A4000 の 0.75 秒とほぼ同等)。遅さは計算ではなく載せすぎによる退避だけだった。
  原因は Windows の torch のプールの膨らみ: テンソル 0.88GB に対して reserved 2.60GB(`expandable_segments` が無い)。
  見積もりはこれを見込んでいなかった。Windows では (a) 抱えた未使用分(reserved − allocated)を「使える」と数えない、
  (b) 余裕の既定を 2.5GiB(Linux は 1.0GiB)にする。サーバ内の 3050(物理の空き 2.2GB)では常駐 0 になり、
  新しいプロセスでは 15 個前後になる計算。
- 同日、さらに: 常駐 0 + ファイル経路にしてもサーバ内では後半で遅くなった(共有 5.9GB)。プローブ(新しいプロセス、プール空)との
  残る違いは torch のプールで、サーバはデノイズ後の reserved 3.05GB(実使用 1.18GB)を抱えたままデコードに入り、デコード中に
  さらに 2〜3 倍に膨らんで 6GB を越える。`H3_GPU_POOL_DEFRAG`(auto = Windows のみ): デコード直前に GPU の実体(VAE の非ブロック部、
  audio VAE、transformer の非ブロック部、約 1.2GB)を CPU に退避 → `empty_cache` → 戻す(`_defrag_gpu_pool`)。出力は不変。
- 同日、さらに: プールを空にする手は効かなかった(reserved 2.84 → 2.66GB。散らばった小さな実体が区画を掴んでいる)。
  代わりに **Windows では未指定なら VRAM の上限を自動で「カード容量 − 1.2GB」にする**(`H3_VRAM_LIMIT_GB` の自動版、
  `H3_VRAM_AUTO_CAP_MARGIN_GB`)。割り当て器は上限に当たると未使用の区画を返して再利用するので、断片化していても膨らめず、
  物理 VRAM を越えない。A4000 の 6GB / 8GB 上限の模擬はこの仕組みそのもので、そこでは退避も OOM も起きていない。
- 置ける数の判定は「ドライバの空き + torch が確保済みで未使用の分(reserved − allocated)」で見る(Linux のみ)。Windows の torch は
  `expandable_segments` が無く、デノイズ後に数 GB が reserved のまま残る(RTX 4060 Laptop の実機: allocated 1.46 / reserved 3.33GB、
  ドライバの空き 3.8GB → 当初 10/36 しか置けなかった)。その分は新しい確保に使い回されるので数えてよい。
- 量子化キャッシュは `scripts/prepare_lowvram.py` で先に作る(bf16 シャードを 1 つずつ落として量子化して消す。RAM 16GB で可)。
  Ref2VA / Ref2I は `transformer` とは別の重み `transformer_ref/`(66GB、全シャードの内容が異なる)を使うので、
  `--ref` で `transformer_ref_int8` も作る(+32GB)。2026-11-18 まで ref2va の group 読み込みはキャッシュの有無を見る前に
  40GB の RAM ガードを通していたため、低 RAM 機では参照モードが必ず止まっていた。base 側と同じく「キャッシュがあって
  file で流すならガード無し」にし、キャッシュが無いときは `--ref` の案内で止める。

実測(48GB 機の GPU0、RAM の上限を cgroup で模擬し、量子化キャッシュのページキャッシュを捨ててから開始。この機の
NVMe は直読み 3.3GB/s。出力はすべて pinned と MD5 一致):

| 原本の置き場所 | 必須ホスト RAM | 512²×5 秒 | 1024²×5 秒 | 備考 |
|---|---:|---:|---:|---|
| pinned(参考) | 約 38GiB(transformer 30 + refiner 1.4 + TE 1.9 + VAE 4.5) | 2.98 秒/step | 15.1 秒/step | VRAM 2.90 / 6.41GB |
| file、RAM 制限なし(キャッシュ温) | 約 8GiB | 3.11 秒/step | — | VRAM 2.49GB(GPU の先読みが 1 単位少ない) |
| file、RAM 28GB 上限(32GB 機相当) | 同上 | 6.1 秒/step | **14.7 秒/step** | 150 単位中 94 をディスク直読み。1K は読みが計算に隠れる |
| file、RAM 16GB 上限 | 同上 | 8.0 秒/step | — | 全 150 単位をディスク直読み(30GiB/step)、MD5 一致 |
| file、RAM 12GB 上限(16GB 機相当)、VAE・TE もファイルから(下記) | 約 2GiB | 8.0 秒/step | **15.4 秒/step** | MD5 一致。VAE は空き RAM があればデコードの間だけピン留め |

1K では読み(最悪 30GiB/step ÷ 3.3GB/s ≈ 9 秒)が 15 秒の計算に隠れて pinned と同じ速さ、512² は読み律速で約 2 倍かかる。
**RAM 32GB の PC で 1024²×5 秒が pinned と同速、RAM 16GB でも 1K まで通る**(速度はディスクの連続読み出し速度に依存)。
最初の 12GB 上限の試験では、デコードの直前にプロセスの匿名メモリが約 12GB に達して落ちた(`proc_anon_gb` の内訳:
VAE の fp16 の写し 4.8、受け皿 1.6、TE の層のピン留め 1.9〜2.4、基本 0.6、デコードのフレームバッファ)。残っていた 2 つも
同じ方式に乗せて、RAM 16GB 相当で 1K を通した:

- **VAE デコーダの fp16 キャッシュ**(`core/vaecache.py`、`models/prequant/vae_decoder_fp16/`、4.5GiB、初回に 2 秒で作る)。
  file モードでは起動時にデコーダブロックを fp16 に写さず(写しが 4.8GiB の匿名メモリだった)、ブロックを 0 要素の
  プレースホルダにしておく。デコードのときは空き RAM を見て、**4.5GiB + 2GiB あればキャッシュからピン留めに 1.4 秒で
  読み込み、デコードの間だけ持つ**(pinned と同じ速さ)。無ければタイルごとにファイルから流す(1K のデコード 48 → 95 秒)。
  終わったらピン留めを返す。値は `vae.to(float16)` と同じ丸めなので出力は同じ。
- **投影 TE の量子化済みキャッシュ**(`H3_TE_PROJ_PREQUANT=1`、既定。`models/prequant/te_proj_<model>_bnb-4bit/`)。
  bnb-4bit の TE を一度だけ保存し(2 秒)、次回から読む(ロードはこの機では bf16 からの量子化と同じ 9〜10 秒だが、読む量が
  bf16 8GB から nf4 2.2GB になるのでディスクの遅い PC で効く)。
  file モードでは LM 層 36 本(1.9GiB)をこのファイルから直接流し、ピン留めしない(受け皿 2 × 0.06GiB)。

これで 12GB 上限(16GB 機相当)でも **1024²×5 秒が完走**(15.4 秒/step、デコードはファイル流しで 97 秒、MD5 一致)、
768² も 8.3 秒/step(VAE をデコードの間だけピン留めしてデコード 30 秒、MD5 一致)。必須のホスト RAM は受け皿 3 本 + TE の受け皿 2 本の約 1GiB と
プロセス本体で、約 2GiB。

##### VRAM 級 × ホスト RAM 級の早見表(t2va 5 秒・30 ステップ、既定の低 VRAM 構成)

速度はこの機(RTX PRO 5000)の秒/step。RTX 3050 などではその数倍かかる。○ = MD5 一致のまま入る、△ = ヘッド分割
(別サンプル相当)か `H3_CACHE=none` で入る、✗ = 入らない。

| VRAM \ ホスト RAM | 16GB(file、全単位をディスクから) | 32GB(file、一部をディスクから) | 48GB 以上(pinned) |
|---|---|---|---|
| **6GB** | 512² ○ 8.0 / 768² ○ 8.3。1K △ 約 17(ヘッド分割、デコードはファイル流しで 97 秒) | 512² ○ 6.1 / 768² ○ 7.1。1K △ 16.8(auto が G=4 を選び 4.4GB) | 512² ○ 3.0 / 768² ○ 7.1。1K △(ヘッド分割で 4.8GB、`H3_CACHE=none` で 3.0GB) |
| **8GB** | 同上。1K ○ 15.4(5.8GB、デコード 97 秒) | 同上。1K ○ 14.7(5.8GB) | 同上。1K ○ 15.1(6.4GB) |
| **12〜16GB** | 同上 | 同上。空き VRAM が 8GB 以上なら VAE を流さずデコードが速い(768² 31 → 11 秒級) | 同上 |
| **24GB 以上** | 同上 | 従来の表(`H3_LOWVRAM=1` 系)も選べる | 同左 |

- VRAM が決めるのは「何が入るか」: 512²(2.9GB)と 768²(4.6GB)は 6GB で入り、1K は 8GB なら MD5 一致のまま
  (pinned 6.4GB / file 5.8GB)、6GB ではヘッド分割(auto が必要なときだけ使う)か `H3_CACHE=none`。
- ホスト RAM が決めるのは「速さ」: 48GB 以上はピン留めで最速、32GB は 768² と 1K が同速で 512² が 2 倍、16GB は
  512² が 2.7 倍・768² が 1.1 倍で、1K も通る(VAE と TE もファイルから流す)。
- file モードは GPU の先読みが 1 単位少ないぶん VRAM ピークも下がる(512² 2.49 / 768² 3.71 / 1K 5.76GB)。
  file で VAE デコーダをピン留めするかは `H3_GROUP_STREAM_VAE_PIN`(auto = RAM 24GB 以上。しないと 768² のデコードが
  31 → 53 秒)。

`H3_LOWVRAM=1`(48GB級)は transformer(34GB)を毎リクエスト GPU に丸ごとロードするため、
24〜32GB級カードでは transformer 単体でも収まらない。`H3_LOWVRAM=group` は
diffusers-server(姉妹プロジェクト)の CLAUDE.md #33/#34/#37 で確立された
「block-level group offload」パターンをこのプロジェクトに移植したもので、transformer を
**ホストRAMに常駐**させたまま、denoise の各ステップで必要なブロック(50層中1〜2層、
~0.68GB×1〜2)だけを都度 GPU へ出し入れする。transformer は**プロセス起動時に一度だけ
ロードされ、リクエストをまたいで常駐し続ける**(`H3_LOWVRAM=1` のような毎リクエスト
再ロードは発生しない)。

#### PR側の「streamed offload時のload-time量子化」調査結果

タスク時点で読んだ `TorchAoHfQuantizer`(`quantizers/torchao/torchao_quantizer.py`)の
`validate_environment()` は、`device_map` に(accelerateの自動割当のような)**辞書**
形式で `"cpu"` という**文字列値**が含まれる場合にのみ `self.offload = True` を立て、
`check_if_quantized_param()` はこのフラグが立っていると CPU 配置のパラメータの量子化を
スキップする(=CPUオフロードするパラメータは意図的に非量子化のまま残す設計)。
一方、本実装が使う `device_map={"transformer": "cpu"}` は `load_components()` を経由して
最終的に `from_pretrained()` に**プレーン文字列** `"cpu"` として渡り、
`modeling_utils.py` の正規化コードにより `{"": torch.device("cpu")}` という
**単一キーの辞書**(値は `torch.device` オブジェクト、文字列ではない)に変換される。
`torch.device("cpu") == "cpu"` は Python 上で `False` になるため、
`"cpu" in device_map.values()` は False のまま保たれ、`self.offload` は立たない。
つまり **CPU上へロードしても量子化はスキップされず、torchaoのInt8Tensorとして
正しく量子化される**(`scripts/probe_group_offload.py` で370/370層がInt8Tensor化
されることを実機確認)。**CPU上でのint8量子化は問題なく可能**という結論。

#### 実装の要点

- `_ensure_transformer_group()`(`core/runner.py`)が
  `device_map={"transformer": "cpu"}` + `TorchAoConfig(Int8WeightOnlyConfig)` で
  CPU上に量子化ロードしてから `enable_group_offload(offload_type="block_level",
  num_blocks_per_group=1, use_stream=..., low_cpu_mem_usage=...)` を呼ぶ
  (transformer_refも同型の `_ensure_transformer_ref_group()`)。
- TEは `H3_LOWVRAM=1` と同じくbnb-4bit必須(起動時強制)。t2vaの定常状態では
  TEはリクエストをまたいで常駐する(transformerがそもそも常駐するため、
  TEも常駐させておいた方がリクエストごとの再ロードコストを避けられる)。

#### 【重大な発見】`use_stream=True` + `low_cpu_mem_usage=True` の併用は
torchao Int8Tensorに対してバグがあり動かない

`scripts/probe_group_offload_forward.py` で実際にforwardを走らせたところ、
`RuntimeError: cannot pin 'torch.cuda.CharTensor' only dense CPU tensors can be
pinned` で denoise の最初のブロックで必ず失敗することを実機確認した。
`hooks/group_offloading.py` の `_pinned_memory_tensors()`(`use_stream=True`なら
`_onload_from_memory()` から毎ステップ無条件で呼ばれる)が
`low_cpu_mem_usage`の値に関わらず `.pin_memory()` を試みるのに対し、
`_init_cpu_param_dict()`(`enable_group_offload()`呼び出し時点で1回だけ実行)は
`low_cpu_mem_usage=True` なら pin をスキップする、という非対称な実装になっており、
両者の想定が食い違っている。torchaoの `Int8Tensor.qdata` はこの食い違いが起きると
壊れた状態(内部的に `torch.cuda.CharTensor` として認識される)でpin_memory()が
呼ばれてクラッシュする。`scripts/probe_group_offload_fix.py` で対照実験した結果:

| 設定 | 結果 | 1ブロックあたりonload/offload |
|---|---|---|
| `use_stream=True, low_cpu_mem_usage=True`(併用時) | **クラッシュ** | - |

(2026-08-10 訂正: 当初この組み合わせを「diffusers既定」と記載していたが誤り。`apply_group_offloading` の API 既定値は `use_stream=False, low_cpu_mem_usage=False` で、省メモリ目的で両方をオプトインしたときに踏む)
| `use_stream=False, low_cpu_mem_usage=True` | 動作OK | onload 0.1-0.26s / offload ~0.22s |
| `use_stream=True, low_cpu_mem_usage=False` | 動作OK | **onload 0.04-0.07s** / offload ~0s |

`low_cpu_mem_usage=False`(`enable_group_offload()`呼び出し時点で全パラメータを
eagerにpin)を新既定に採用した(`H3_GROUP_OFFLOAD_LOW_CPU_MEM`、既定`0`=False)。
理由: onloadが4-5倍速い(pinned memoryはページアウト不可でDMA転送が速いため)。
代償はロード時に追加で~14-16GBのホストRAMをpinする(page-lockedなのでスワップ
不可)ことと、`enable_group_offload()`自体が約22秒かかること(実機測定、
CPU上へのロード70秒 + pin化22秒 = 合計約90秒)。より少ないRAMを優先したい場合は
`H3_GROUP_OFFLOAD_LOW_CPU_MEM=1`(このとき`H3_GROUP_OFFLOAD_USE_STREAM`も
明示指定しない限り自動で`0`にフォールバックする、上記の壊れる組み合わせを
避けるため)を明示指定すればよい。

#### choreography最終形(フェーズ×常駐物×ピーク)

| フェーズ | 常駐する大きいもの | 備考 |
|---|---|---|
| 起動時preload | transformer(int8, CPU常駐+groupoffloadフック) | 約90秒(CPUロード70s+pin化22s) |
| t2va encode | TE-nf4(GPU,21GB) + transformer(CPU) | |
| t2va denoise | TE-nf4(GPU,21GB) + transformerの1-2ブロック(GPU,~1.4GB) | |
| t2va decode | vaeペア(GPU,~11GB) + transformerの1-2ブロック | **TEはこの窓だけ強制解放**(下記参照)、decode後に再ロード |
| ref2va参照エンコード | vaeペア(GPU,11GB) | TEはこの窓だけ強制解放(下記参照) |
| ref2va denoise | TE-nf4(GPU,21GB) + transformer_refの1-2ブロック | |
| リクエスト間定常 | transformer(CPU) + TE-nf4(GPU) | ref2va後はtransformer_refが未ロードに戻る(t2va↔ref2va切替のたび再ロード) |

#### 【実装中に発見・修正した2つ目のバグ】decode窓・参照エンコード窓でのTE強制解放が必要だった

当初「group offloadされたtransformerのGPU実消費は極小(~1.4GB)だから、decode時に
transformerを解放する必要は無い」と設計したが、32GBダミーVRAM検証で
`CUDA out of memory` を実機再現し、`_log_gpu_tensor_diag()`
(`H3_DEBUG_MEM_DIAG=1`で有効化する一時診断関数、`core/runner.py`に残置)で
実際に生存しているCUDAテンソルを列挙したところ、TE-nf4自身の埋め込みテーブル/
lm_head重み(shape `(151936, 5120)`、bf16、1.556GB×2 = 3.1GB強を含む合計22.25GB)が
デコード直前まで**常駐したまま**だったことが判明した(`empty_cache()`だけでは
解放されない、実際に参照されている生きたテンソルだったため)。つまり
TE-nf4(21GB)+ decode専用バッファ(~16.3GB、下記VAEタイル調査参照)=37GBが
真の必要量で、transformerのフットプリントとは無関係にTEとVAEの競合だった。
対策: **decode窓(と、ref2vaの参照VAEエンコード窓)でTEを強制解放し、窓を抜けたら
再ロードする**(`force_free_te`とは別枠の専用ロジック、`_execution_device`解決順序
はTE解放→vaeをGPUへ、の順で安全性を確保)。

#### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345)で、通常int8モード
(`H3_LOWVRAM`未指定、`H3_TRANSFORMER_QUANT=int8`、FBC `H3_CACHE=fbc`有効)の
出力と `H3_LOWVRAM=group` の出力を比較したところ、**FBCのキャッシュスキップ判定が
実行経路の違いで異なった**(`cache_skipped_steps`が6→0)ため素朴な比較ではmp4が
不一致だった。FBCは前ステップとの残差の類似度という数値的に鋭敏な判定のため、
数学的に等価な演算でも経路が変わればスキップ判定が変わりうる(劣化ではない)。
両モードとも `H3_CACHE=none` に揃えて再比較したところ、**mp4がバイト完全一致
(md5一致)**することを確認した。group offloadの計算内容は既存経路と数学的に
同一であることの裏付け。

#### 計測表: 32GB制限・24GB制限プローブ

**32GB制限**(ダミーVRAM確保で空きを~30GBへ制限、`H3_CACHE=fbc`有効のまま):

| | 完走 | ピークVRAM | 所要時間 |
|---|---|---|---|
| t2va 1本目(768²・5秒・30steps) | ○ | **28.67GB** | denoise 220.79s / decode 6.31s / 総計337.19s(TE初回ロード込み) |
| t2va 2本目(連続) | ○ | **28.23GB** | denoise 220.85s / decode 6.01s / 総計278.83s(TE常駐のため短縮) |
| ref2va(画像参照1枚、768×1344) | **RAM不足で拒否**(下記参照) | - | - |

2本とも1本目と完全に同一mp4(md5一致、`be3f32a84de074990208ad0d30f31a63`)。
ホストRAM/スワップは各フェーズとも増加なし(`free -g`のSwap usedは作業前後とも
~7-8GB台で安定、この値は他プロセス由来の既存分)。

**24GB制限プローブ**(ダミーVRAM確保で空きを~22GBへ制限):

- 768²: denoise中(transformerブロックのonload)でOOM。実消費21.85GB、うち
  TE-nf4だけで21GB。**TE-nf4自体の固定サイズ(21GB)が22GB予算の大半を占め、
  transformerブロック1個分(~148MB)の追加onloadすら入らない**。
- 544×960(RESOLUTION_PRESETSへ一時追加して検証、検証後に削除済み): **同一箇所・
  同一21.85GBでOOM**。解像度を下げても失敗点も消費量も変わらず、**VAEタイル
  縮小と同じく「解像度に依存しない固定コストが律速」であることを確認**
  (下記VAEタイル調査参照)。
- **結論**: 現行アーキテクチャ(TEをbnb-4bit・常時ほぼ常駐という設計)では、
  TE-nf4単体の21GBが24GB級カードの実効予算(~22GB)の大半を占めてしまい、
  どんな解像度でも成立しない。24GB級に対応するには、TEをdenoise中は解放する
  (`H3_LOWVRAM=1`的な設計に戻す)か、TE自体をさらに軽量化する(GGUF等、ただし
  transformers系モデルへのGGUF適用は構造的に困難)必要がある。本タスクの
  スコープ外(48GB→24-32GB対応が目的で24GBは探索的プローブ)のため、
  現時点では未対応と結論する。

#### VAE tiling調査結果

`scripts/probe_vae_tile_size.py` で768²・124フレームのVAE decodeを単体で
(denoiseを介さず合成潜在から)直接ベンチマークしたところ、
**tile_sample_min_height/width を256(既定)→192→128→96まで縮小しても
ピークVRAM(16.29GB)が全く変化しなかった**(所要時間はタイル数が増える分
5.9s→10.5sへ悪化)。この結果から、decodeのピークは空間タイルの合成バッファでは
なく、**時間チャンク(`tokens_chunk_size`単位)の1チャンク分をまるごとデコード
するバッファ**か、固定的なVAEアーキテクチャのオーバーヘッドに支配されていると
推測される(VAEクラスは時間方向のチャンクサイズを公開パラメータとして持たない
ため、これ以上の調整はコード変更が必要で本タスクの範囲外)。
**結論: 24-32GB級対応において空間タイルサイズの調整は無意味**(既定のままでよい)。

#### FBC/sage共存の確認結果

全てのballast検証(32GB・24GB双方)を通じて `H3_CACHE=fbc`(既定)・
`H3_ATTN_BACKEND=sage`(既定)を有効にしたまま実行し、group offloadのフックと
競合するエラーは一切発生しなかった(FBCはブロック単位の計算スキップ判定、
group offloadはブロックのGPU常駐管理、sageは各ブロック内部のattention実装、と
三者は独立したレイヤーで動作するため)。MD5一致チェック(`H3_CACHE=none`で
実施)とは別に、既定のFBC有効設定での通し実行(32GB制限のt2va 2本)が完走した
ことも確認済み。

#### Ref2VAのRAM制約(既知の制限、未解決)

`H3_LOWVRAM=group` でt2va実行後にref2vaを呼ぶと、VRAM予算に関わらず(96GB機で
ダミーVRAM確保無しでも再現)ホストRAMガードで拒否されることを実機確認した:

```
H3_LOWVRAM=group requires at least 40.0GB of available host RAM before loading
the (~34GB, permanently CPU-resident) int8 transformer, but only 33.0GB is
available right now.
```

原因の切り分け: transformer解放直後は`avail_gb`が正しく回復する(44.6GB前後)が、
その後のTE再ロード→参照VAEエンコード→レイアウト計算の過程で`avail_gb`が
33GB前後まで下がる(実機ログで確認)。`swap_used_gb`は一貫して増加しないため
実際のスワップ発生は無い ─ `MemAvailable`(Linuxのbuff/cache込みヒューリスティック
推定値)の変動が、真の空きRAMより保守的に振れている可能性が高い。ただし
`free -g`の`used`ベースで見ても94GB中62GB使用(32GB残)という状況で
追加34GBの確保は本質的にタイトであり、ガード自体が誤りとは断定できなかった
(96GB機でも「t2va transformerを常駐させたままref2va用transformer_refをさらに
CPUへpinしようとする」設計そのものが、94GB RAM機の物理容量に対してすでに
ギリギリ)。**安全側に倒し、ガードを緩めることはせず既知の制限として記録する**
(過去のスワップ暴走事故の教訓を優先)。将来の改善候補: `preload_all()`での
transformer即時ロードをやめてTEと同様に遅延ロード化する(初回リクエストの
レイテンシとのトレードオフ)、またはt2va⇔ref2va切替時によりRAMを消費しない
経路を設計する。**RAM 48GB以上のマシンでは(未検証だがRAM予算に余裕があるため)
この問題は起きない可能性が高い**(本タスクは94GB機でのみ検証、より多いRAM
搭載機での追試は未実施)。

### 2026-08-09: H3_KEEP_TRANSFORMER — transformer 常駐で毎リクエスト再ロード固定費を撤廃

`H3_LOWVRAM=1` はデコード直前に transformer を解放し、次リクエストで**毎回**再ロードする
(実測 14.8〜32.7s の固定費、`docs/RESIDENCY.md` §5.5)。`H3_KEEP_TRANSFORMER=1` はこの
解放をスキップし、transformer を GPU に常駐させたまま次のリクエストに持ち越すことで
この固定費を初回ロードの1回だけに収束させる。

**成立条件は3つとも必須**(`core/runner.py` の import 時ガードで強制、欠けたら
`RuntimeError`)。VRAM 収支の理由:

1. `H3_LOWVRAM=1`(raw `"1"` のみ。`group` は transformer をホストRAM常駐+ブロック
   offload する別設計のため対象外)
2. `H3_TE_DEVICE` 設定済み(TE を別GPUに常駐)— **これが無いとエンコード位相が先に
   破綻する**: TE-nf4 17.45GB + 常駐transformer-int8 34.3GB = 51.75GB が実効予算を
   超える
3. `H3_VIDEO_VAE_FP16=1` — encode 位相は TE を別GPUに逃がすことで回避できるが、
   decode 位相は fp32 デコードピーク16.29GBのままだと transformer 34.3 + 16.29 =
   50.6GB で実効予算(~49.8GB)を超える。fp16 化した VAE のデコードピーク ~11.4GB
   なら 34.3 + 11.4 = 45.7GB で収まる(video VAE fp16 の品質影響は既知実測 PSNR
   **39.97dB**・目視差なし、詳細は「video VAE の fp16 化」節)

既定(`H3_KEEP_TRANSFORMER=0`)は挙動不変。**ref2va(`transformer_ref`)は対象外** —
このフラグの範囲外で、従来どおり毎回解放される。

```bash
# 推奨起動コマンド (48GB GPU0 + 20GB GPU1)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1 \
  venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

**実機 E2E 実測(2026-08-09、48GB GPU0 + 20GB GPU1、上記起動構成)**:

- transformer(int8) は初回リクエストで1回だけロード(32.0s)。以降のリクエストで
  再ロードなし(サーバーログで確認)
- t2i turbo 4steps 定常: **9.7s/枚**(denoise 4.32s、decode 1.5s)。peak VRAM 41.97GB
  (デノイズ時)
- t2i steps=30 定常: 51.1s(denoise 45.7s)。peak 41.97GB
- t2va 5秒 turbo 4steps: **44.2s**(denoise 26.05s、decode 10.81s)。**peak VRAM
  44.15GB = デコード位相**(transformer 34.03GB 常駐 + fp16 デコード)。導出予測
  45.7GB に対し実測 44.15GB、カタログ 48.9GB に対し余裕 ~4.8GB
- nvidia-smi 実測ピーク 42,620 MiB(1秒サンプリング。瞬間ピークは torch 計測の
  44.15GB が正)
- 同一seed 出力等価性: フラグOFFのベースライン(他は同条件)と PNG MD5 **完全一致**
  (seed=11、md5 `665eadddea8f34298a1b5b89e69d4bd0`)。ベースライン側は total 63.27s
  (transformer ロード込み)/ peak 36.4GB

**高速化の系譜**(48GB GPU0・`H3_LOWVRAM=1` 系、t2va は 768²・5秒):

| | t2i | t2va 5秒 |
|---|---|---|
| turboなし 30steps(GPU交換直後の素の構成、08-07) | 157s | 351.4s(デノイズ197.7s) |
| lightx2v turbo 4steps 導入(08-07朝) | 157s | 143s |
| + `H3_TE_PREQUANT` | 83.2s | — |
| + `H3_TE_DEVICE` | ~35s | 60.5s |
| + `H3_KEEP_TRANSFORMER`(本節) | **9.7s** | **44.2s** |

t2va は素の構成比 **8.0倍**(351.4s → 44.2s)。turbo だけでは 2.6倍(デノイズは7.6倍
短縮されるがロード固定費 ~110s が残る)で、残りは固定費撤廃
(`H3_TE_PREQUANT`/`H3_TE_DEVICE`/`H3_KEEP_TRANSFORMER`)の寄与。なお本節構成での
turboなし 30steps は t2i で 51.1s(上記実測)— 素の 157s に対し、30steps のままでも
固定費撤廃だけで3倍速い。

詳細な VRAM 収支の導出・位相×常駐表は `docs/RESIDENCY.md` §5.5・§5.6 を参照。

### text_encoder を別GPUへ常駐 (`H3_TE_DEVICE`、既定 ``、2026-08-09)

`H3_LOWVRAM=1` が毎リクエストで TE を再ロードする根本原因は、**デノイズ中に TE の
置き場所が無いこと**にある(48GB では transformer-int8 34GB + 活性化 5GB = 39GB で、
残り 9GB に TE の 17.45GB は入らない)。TE を2枚目のGPUへ逃がせばこの再ロードが消える。

```bash
# 例: TE を cuda:1 に常駐させる (CUDA_VISIBLE_DEVICES は設定しない = 両GPUを見せる)
H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 venv/bin/python -m uvicorn app:app --port 8611
```

**実測(RTX PRO 5000 48GB + RTX 4000 SFF Ada 20GB、t2i turbo 4steps、768²)**:

| | 1回目 | 定常(2回目以降) |
|---|---|---|
| 従来(TE も cuda:0) | 81.3s | 67.6〜86.2s(平均 **78.4s**) |
| **TE を cuda:1 へ** | 88.2s(TE初回ロード込み) | 33.6〜36.7s(**約35s、-55%**) |

TE のロードは**プロセス起動後1回だけ**。GPU1 に 17GB 常駐したまま、GPU0 はアイドル時
2.3GB まで下がる。

- **ref2va は自動的に拒否される**(20GB級の場合): 2048px 短辺の参照を vision tower に
  通す活性化が入らず OOM することを実測済み(19.25GB 使用中に 204MB 不足)。
  TE用GPUが 24GB 未満なら理由を添えて 400 を返す — 「動くはず」で走らせて OOM させない。
  24GB 以上なら自動的に許可される。t2va/fl2va/t2i とその各バッチは併用可能
- **PCIe 幅は問題にならない**: GPU1 は Gen4 x4(x16対応カードだが4レーン接続)だが、
  TE は起動時に一度載せるだけで、毎リクエストの転送は prompt_embeds の約42MB のみ(約6ms)

**出力は従来構成とビット一致しない**(PNG の MD5 が異なる)。原因は **sm_120(Blackwell)
と sm_89(Ada)のアーキテクチャ差による丸め**で、実測した `hidden_states[50]` の
**相対RMS差は 0.084%**(max_abs_diff 3.5、値のRMS 57.4)。これは既に採用済みの
参照プレフィックス共有の差(1.5%)より**20倍小さく**、バグの水準(ネガティブ
コントロール 27-30%)とは桁が違う。Sage Attention や int8 と同種の軌道ドリフトであり、
目視でも品質は同等。**ビット再現が要る対照実験では `H3_TE_DEVICE` を外すこと**。

#### 実装の要点(`_execution_device` の罠、再び)

`_execution_device` は components 順(`text_encoder, tokenizer, processor, vae, ...`)で
最初の nn.Module のデバイスを返す(実装を読んで確認)。**TE を cuda:1 に置くと
これが cuda:1 を返し**、layout/latents/timesteps も decode も別GPUにテンソルを作る。

最初は「layout の窓だけ塞ぐ」実装にしたが、**decode でも発火した**
(`latents = latents * latents_std + latents_mean` が
`Expected all tensors to be on the same device, cuda:0 and cuda:1` で失敗、実機で再現)。
窓を個別に塞ぐ方式は漏れるため、**「既定では TE をパイプから外しておき、エンコード中
だけ繋ぐ」**(`_te_attached()`)という反転した設計に変更した。これなら新しい経路が
増えても安全側に倒れる。モジュール実体は `self._te_module` が保持し続けるので、
外しても解放されない。

layout/latents/timesteps の窓ではさらに `_pin_execution_device_to_compute()` で
vae も一時的に外す(TE を外した後の次の nn.Module が CPU 上の vae になり、今度は
`_execution_device` が cpu を返してしまうため。transformer が最初に見つかるようにする)。

### video VAE の fp16 化 (`H3_VIDEO_VAE_FP16`、既定 `0`)

`H3_VIDEO_VAE_FP16=1` で video VAE の重みだけを fp16 化する(9.70→4.85GB、デコード
ピーク 16.29→~11.4GB)。**audio VAE は絶対に触らない**(bf16化で-20dBの既知問題)。
- 品質: 全124フレームの平均PSNR **39.97dB**(min 39.08)で目視区別不能。デコード計算は
  元々autocast fp16のため重みfp16化の影響が小さい
- **実装の罠**: `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` が encoder/decoder等を
  強制fp32に戻すため、`from_pretrained(dtype=fp16)` は効かない(実機確認)。ロード後に
  `.to(torch.float16)` を明示的に呼ぶ必要がある
- 既定OFFでは既存基準とMD5一致(回帰ゼロ)

## text_encoder

### text_encoder の未使用上位レイヤー削除 (`H3_TE_PRUNE`、既定 `0`、2026-08-06追加)

MiniMax-H3 の text_encoder(Qwen3-VL-32B、64層)は
`hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`(=50)しか読まない
(`diffusers/modular_pipelines/minimax_h3/encoders.py`)。`H3_TE_PRUNE=1` は
text_encoder を **51層だけ**(0〜50、`MINIMAX_H3_TEXT_ENCODER_LAYER + 1`)で構築し、
未使用の52〜64層目 + 最終`norm` + `lm_head`(重み換算で14層分、bnb-4bit実測 ~3.6GB /
bf16実測 ~13.6GB)を一度もロードしない。既定 `0` は完全無変更(この分岐自体が
一切呼ばれない)。

#### なぜ「50層」ちょうどではなく「51層」なのか(このタスクで発見・検証したtransformers側の罠)

`hidden_states[k]` の意味は transformers の `_can_record_outputs = {"hidden_states":
Qwen3VLTextDecoderLayer}` フック機構(`output_capturing.py`)により決まる:
`hidden_states[0]` = 埋め込み出力(layer 0 の入力を捕捉)、`hidden_states[k]`
(k=1..num_hidden_layers) = `layers[k-1]` の出力。つまり `hidden_states[50]` =
`layers[49]` の出力で、**layers[0..49](50層)を実行すれば十分**なはずだった。
しかし `num_hidden_layers` を**ちょうど50**に切り詰めると、`hidden_states[50]` が
捕捉タプルの**最後の要素**になってしまい、`Qwen3VLTextModel.forward` を包む
`@capture_outputs(tie_last_hidden_states=True)`(既定)が「最後の要素を
`outputs.last_hidden_state`(=最終`norm`適用後の値)で強制的に上書きする」という
挙動を発動させる。実機検証(`scripts/probe_te_prune*.py`)で、50層ちょうどに
切り詰めた場合の `hidden_states[50]` は本来の(64層モデルの)値と**桁違いに
異なる**(max abs diff ~1.5e4、量子化誤差の水準ではなく完全に別の値)ことを確認した。
これはまさに `encoders.py` 自身のガード
(`if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER: raise ValueError(...)`)が
警告している「ちょうど50層に切り詰めた最終隠れ状態はpost-normであり、MiniMax-H3が
期待する値ではない」という事態そのもの(このガードのおかげで50層ちょうどの誤設定は
`encode_prompt()` 経由なら例外で弾かれる)。**51層**(`layers[50]`は実行されるが
その出力は読まれない、無駄な1層分の計算コストのみ)にすることで
`hidden_states[50]` が捕捉タプルの中間に位置するようになり、上書きを回避できる。
51層版で64層版の `hidden_states[50]` と**完全一致**(`torch.equal`、bf16・bnb-4bit
nf4とも)することを実機確認済み。

#### 実装

`core/runner.py` の `_text_encoder_config_kwargs()` が、text_encoder の
`ComponentSpec`(`pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3"`,
`subfolder="text_encoder"`)と同じ場所から `Qwen3VLConfig` を個別ロードし、
`text_config.num_hidden_layers = 51` に書き換えたオブジェクトを
`load_components(..., config={"text_encoder": pruned_config})` として渡す。
`PreTrainedModel.from_pretrained` は `config` が既に `PreTrainedConfig` インスタンス
なら自前のconfig自動ロードをスキップしてそのまま使う(`modeling_utils.py`で確認)。
チェックポイントの `layers.51-63.*` は `from_pretrained` のロードレポートに
`UNEXPECTED` として現れ、単純に無視される(構築されないため一切のVRAM/RAMを消費
しない)。vision tower(`model.visual`)は無変更(fl2va のキーフレーム/ref2va の
参照画像・動画のpixel_valuesがここを通るため、削除対象から明示的に除外)。

`H3_TE_QUANT`(bnb-4bit/none)・`H3_LOWVRAM`(0/1/group)のどの組み合わせとも合成可能。

#### 削除後TEの実測サイズ

| 精度 | 削除前 | 削除後(51層) | 削減 |
|---|---|---|---|
| bnb-4bit nf4 | 21.02GB | **17.45GB** | -3.57GB (-17%) |
| bf16 | 66.71GB | **53.06GB** | -13.65GB (-20%) |

nf4は量子化で1層あたりのサイズがbf16の約1/4に圧縮されるため、削減の絶対量もbf16より
小さい(相対削減率はほぼ同じ)。

#### MD5一致チェックの結果

同一seed(768²・5秒・30steps・キツネのプロンプト、seed=12345、`H3_CACHE=none`で
FBCの経路依存を排除)で、`H3_TE_PRUNE=0`(削除なし)と`H3_TE_PRUNE=1`(削除あり)の
出力を比較したところ、**t2va・ref2va(画像参照1枚、vision tower経由)とも
mp4がバイト完全一致(md5一致)**した。削除が数学的に無影響であることの実証。
`H3_LOWVRAM=1`・`H3_LOWVRAM=group`の各モードでも、削除の有無で出力が完全一致する
ことを確認済み(後述)。

#### 24GB級対応: 削除だけでは不十分だった(実機で発見・`H3_LOWVRAM_GROUP`側に追加修正)

24〜32GB級対応の既存機構(`H3_LOWVRAM=group`)は、TE-nf4(削除前21GB)を
denoise中も常駐させたままにする設計だった(group offloadされたtransformerの
実消費が~1.4GBと小さいため、32GB級では問題にならなかった)。削除後のTE(17.45GB)は
それでもまだ大きく、**22GB制限で実機OOMを再現した**(denoise開始直後、
21.73GB使用中に224MB要求で失敗)。24GB制限でも同様にOOM(23.12GB使用中に
1.16GB要求で失敗、step 1で発生)。

対策として `H3_LOWVRAM_GROUP` かつ `H3_TE_PRUNE=1` の場合に限り、
`H3_LOWVRAM=1`と同じ「denoiseループの間だけTEを強制解放し、decode窓の前後で
リロードする」選択法を追加した(`core/runner.py`の`group_free_te_for_denoise`
フラグ)。解放位置はlayout_step/latents_step/timesteps_stepの**後**(既存の
`force_free_te`と同じ理由: これらのステップの出力は既にテンソルとして
`state`に載っているため、`_execution_device`解決には以後一切影響しない)。
`H3_TE_PRUNE=0`(既定)の`H3_LOWVRAM_GROUP`は完全無変更(このフラグは
`H3_LOWVRAM_GROUP and H3_TE_PRUNE`の両方が真のときのみ真になる)。

修正後、22GB/24GB/20GBいずれのVRAM制限でも実機で完走を確認した:

| VRAM制限 | 結果 | ピークVRAM(reset後の計測) | 総所要時間 |
|---|---|---|---|
| 22GB(修正前、削除のみ) | **OOM**(denoise開始直後、21.73GB使用中に224MB要求) | - | - |
| 24GB(修正前、削除のみ) | **OOM**(step 1、23.12GB使用中に1.16GB要求) | - | - |
| 24GB(修正後、1本目) | ○ | 17.72GB | 321.7s(TE初回ロード込み) |
| 24GB(修正後、2本目・連続) | ○ | 18.68GB | 277.7s(TE常駐のため短縮) |
| 20GB(修正後) | ○ | 17.72GB | 320.3s |

24GB×2本・20GB×1本の出力mp4は**すべてバイト完全一致**(md5一致、通常int8モード
(`H3_LOWVRAM`未指定)の出力とも一致)。group offloadの計算内容が
VRAM予算に関わらず数学的に同一であることの裏付け(既存の32GB/24GB検証結果と同じ
結論)。ホストRAM/スワップは各テストの前後で増加なし(`free -h`のSwap usedは
一貫して~7.9GB台、既存分のまま)。

#### `H3_LOWVRAM=1`(48GB級)でのTEロード時間短縮

削除により、`H3_LOWVRAM=1`が毎リクエスト支払うTEロード固定費が短縮される
(実機、ダミーVRAM確保で空きを~43.5GBに制限):

| | TEロード時間 | TEサイズ |
|---|---|---|
| 削除なし | 42.3s | 21.01GB |
| 削除あり | **35.0s**(-17%) | 17.44GB |

出力mp4は削除の有無でバイト完全一致(md5一致)。

#### 回帰確認

`H3_TE_PRUNE`未指定(既定`0`)の状態で同一条件のt2vaを実行し、この機能追加前に
取得していた基準mp4とバイト完全一致(md5一致)することを確認した。既定動作は
完全無変更。

### 量子化済み text_encoder のディスクキャッシュ (`H3_TE_PREQUANT`、既定 `1`、2026-08-08)

`H3_LOWVRAM=1` は毎リクエストで TE を再ロードするため、その時間がそのまま固定費になる。
この時間の大半は「元の bf16 重みを読む + その場で bnb-4bit へ量子化する」処理なので、
**量子化後の重みを一度保存しておけば次回以降は読むだけで済む**。初回ロード時に自動保存し、
以降はそこから読む。

**実測(RTX PRO 5000 48GB + `H3_LOWVRAM=1 H3_TE_PRUNE=1`、t2i turbo 4steps を各4回)**:

| | TE ロード | リクエスト合計 |
|---|---|---|
| キャッシュ無効(従来) | 46.5〜55.8s(平均 **53.0s**) | 108.7〜150.0s(平均 **128.6s**) |
| **キャッシュ有効(既定)** | 21.1〜34.3s(平均 **29.5s**) | 75.3〜90.6s(平均 **83.2s**) |
| 差 | **-23.5s(1.8倍)** | **-45.4s(-35%)** |

**等価性**: 同一 seed の生成物が **PNG で MD5 完全一致**(seed 3・4 で確認)。加えて
プローブ(`scripts/probe_prequant_equivalence.py`)で `hidden_states[50]` が
`torch.equal` でビット一致(max_abs_diff 0.0)することも確認済み。bnb-4bit の量子化は
決定的なので当然の結果だが、速いだけで採用しないのが本プロジェクトの流儀。

- キャッシュ先: `models/prequant/te_<quant>_prune<0|1>/`(.gitignore 済み)。
  **設定ごとに別ディレクトリ**にしてある — TE_QUANT / TE_PRUNE を変えると重みの中身が
  別物になるため、同じ場所を使い回すと設定切替時に古い重みを読む事故が起きる
- 消費ディスク: **17.44GB**(`H3_TE_PRUNE=1`)/ 未削除なら ~21GB
- `H3_TE_PREQUANT=0` で完全に無効化(導入前と同一挙動)。空きディスクが
  `H3_TE_PREQUANT_MIN_FREE_GB`(既定 25GB)を下回る場合は**保存をスキップして生成は続行**する
  (キャッシュは高速化であって機能ではない)。保存は一時ディレクトリへ書いてから rename
  するので、途中で落ちても中途半端なキャッシュが「有効」に見えることはない

**測定上の注意(この検証で踏んだ罠)**: 保存直後に測ると 17.44GB がページキャッシュに
載っているため **2.6秒**という非現実的な値が出る。実運用ではリクエスト間に transformer
(34GB)をディスクから読むため TE のページキャッシュが押し出され、実測は 21〜34秒に
落ち着く(リクエストを重ねるほど遅くなり 34秒前後で頭打ち)。**それでも従来経路の
46〜56秒より速い**が、プローブの数値をそのまま性能として報告してはいけない。

### 2026-08-10: 投影TE(Qwen3-VL-4B + 学習済み線形写像)— 実装と実測、未検証項目

`H3_TE_PROJ` として実装(既定OFF、既存挙動は1バイトも変わらない)。TE を Qwen3-VL-32B から
**Qwen3-VL-4B + 学習済み投影行列**([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3))へ
置き換える経路。狙いは48GB機で TE と transformer を**同時常駐**させ、載せ替えの固定費を消すこと。

```
cond = ((h - mean_in) / std_in) @ W * std_out + mean_out     # h = 4B の hidden_states[24]
cond[:, 0] = sink_out                                        # token 0 はアテンションシンク
```

#### 実測(96GB機、t2i 768²・30steps・seed 4242・同一プロンプト)

| | 32B TE(現行) | **投影4B TE** |
|---|---|---|
| PSNR | — | **22.64 dB** |
| 鮮鋭度(ラプラシアン分散) | 48 | **37**(-23%) |
| 生成時間 | 65.7s | **43.7s** |
| ピークVRAM | 88.73GB | **76.1GB** |
| **TE の GPU 実占有** | 21.02GB (nf4) | **8.88GB** |

**品質は「同じ絵」ではなく「同等品質の別の絵」。** 構図・色調・時間帯の解釈は一致するが、
細部の指定は落ちる(実例: プロンプトにない範囲だが、32B版にあった手前の睡蓮の葉と草が
投影版では水面に置き換わった)。破綻はなく実用水準。これは投影の性質(配布元の報告で
test cosine 0.712)どおりで、**プロンプトの大意は保持し細部は失う**と理解するのが正確。

**この変更はこれまでの全最適化と種類が違う。** 量子化・常駐制御・turbo などは全て同一seed
MD5一致で「数学的に無影響」を証明できたが、投影TEは**原理的に近似**なので MD5 は使えない。
判定は PSNR + 目視 + VRAM 実測による。

#### トークナイザの調査結果(実装方針の根拠)

| 対象 | 結果 |
|---|---|
| 通常テキスト・日本語 | **ID 完全一致**(H3 側のトークナイザをそのまま使える) |
| `<d>` / `</d>` | H3 は 151669/151670 の**単一トークン**、4B は語彙外で分解される |
| `<\|cutoff\|>` `<\|lyrics_*\|>` `<\|caption_*\|>` | H3 固有(151671〜675)だが**公式ガイドも本アプリも未使用** |
| `[Shot n]` `<cutoff>` `<scenetrans>` | **通常テキスト**(特殊トークンではない)。マルチショット構成は影響なし |

→ 実使用で影響を受けるのは**台詞タグ `<d>` だけ**。実装では id>=151669 を含むプロンプトを
`ValueError` で**明示的に拒否**する(黙って別物を送らない)。台詞は音声参照(`fully_copy`)で
入れられることを同日に実証済み(上記セクション参照)なので、運用上の回避策がある。

#### 2枚目GPUの要件が下がる

`docs/RESIDENCY.md` §5.4 の導出式に実測 8.88GB を入れ直すと、`H3_TE_DEVICE` 用のカード要件が
**20GB → 16GB(おそらく12GB)** に下がる。

| 2枚目GPU | 実効予算 | 32B TE (17.45GB) | 投影4B TE (8.88GB) |
|---|---|---|---|
| 12GB | ~10.5GB | 不可 | **成立見込み**(必要 ~9.2GB、余裕 ~1.3GB) |
| 16GB | ~14.5GB | 不可 | **成立見込み**(余裕 ~5.3GB) |
| 20GB | ~19.7GB | t2va のみ成立・ref2va は OOM | 成立(余裕 ~10.5GB) |

**12GB は実機確認なしに断言しない。** 以前 20GB で「導出上は入るはずが実測 OOM」を
経験しており(単位の罠、§5.1)、余裕1.3GB は薄い。

#### 追記(同日): NF4 量子化オプションと動画での品質確認

**`H3_TE_PROJ_QUANT`**(既定 `none` / `bnb-4bit` / `bnb-8bit`)を追加した。4B 自体を
量子化する **4B 専用フラグ**で、32B 用の `H3_TE_QUANT` とは別物(排他ガードの意図を参照)。

**投影行列は bf16 用のものをそのまま使うのが正解**(実測)。量子化による条件付けのズレは
NF4 で相対RMS 0.61〜0.96%(cosine 1.0000)。配布元の int8_convrot 用行列を使うと
かえってズレが増える(1.02〜2.97%)— あれは ComfyUI の量子化方式専用の校正なため。

| 構成 | TE 常駐 | t2i (30steps) | t2va 5秒 (30steps) | ピーク(t2va) |
|---|---|---|---|---|
| 32B TE | 21.02GB | 65.7s | 162.1s | 91.9GB |
| 投影4B bf16 | 8.88GB | 43.7s | — | — |
| **投影4B NF4** | **3.11GB** | **33.5s** | **143.5s** | **74.3GB** |

**動画でも品質確認済み**(768²・5秒・124frames・同一seed 555):
静止画で見えた鮮鋭度低下(-23%)は動画では出ず(187 vs 191)、**ちらつき(2階差分)は
むしろ 11% 少ない**(7.51 → 6.67)。bf16 vs NF4 の直接比較は PSNR 34.45dB で量子化の
影響はほぼゼロ。32B との PSNR 14.98dB は「劣化」ではなく「同じ指示の別テイク」
(32B版には対向者が現れ、投影版は少女単独、など解釈の分岐)。

**設定APIとの関係**: 投影TEは現状 **env 専用**(`H3_TE_PROJ`/`H3_TE_PROJ_QUANT`)。
`/api/settings/apply` から te_quant/te_prune を変えようとすると、runner の import 時
ガードを素通りした上で**適用されないのにスナップショットは変わったと報告する**穴が
あったため、投影TE有効時はこれらのフィールドを 400 で拒否するようにした
(`/api/settings` のスナップショットに読み取り専用の `te_proj`/`te_proj_quant` を追加)。

#### 追記(同日その2): NF4 を既定化、UI からの切り替えに対応

- **`H3_TE_PROJ_QUANT` の既定を `bnb-4bit` に変更**(品質実測に基づく。none/bnb-8bit は
  明示指定で選択可)。既定変更に伴い「量子化指定あり+投影OFF」ガードは**明示指定
  (`in os.environ`)のときだけ**発火するよう修正 -- 既定値のままの通常ユーザーを
  誤って落とさないため
- **UI の再ロード設定パネルから投影TEを切替可能に**(`/api/settings/apply` の
  `te_proj` / `te_proj_quant`)。ON のとき te_quant/te_prune のコントロールは無効化され、
  API 側も 400 で拒否する(判定は「適用後の値」ベース: OFF に戻しながら te_quant を
  変えるのは合法)

**往復リロード E2E(実測、96GB機)**: OFF→ON 47.9s / ON→OFF 29.4s / 再ON 27.3s。
各段で同一 seed 生成の PNG MD5 を照合し、**UI 経路の ON = env 経路の NF4 と完全一致**、
**OFF = 32B ベースラインと完全一致**、**再 ON = 1回目の ON と完全一致**(状態残留なし、
投影行列の再ロード正常)。ON 中の te_quant 変更は 400。

#### 未検証(重要)

- **48GB機での TE+transformer 同時常駐** — **本命**。8.88 + 34(int8) = 42.9GB で載る見込みだが
  未実測。これが成立して初めて導入の意味がある(96GB機では元々同時に載るため差が出ない)
- **参照経路(ref2va)** — 投影行列は**テキストのみで校正**されており、vision 特徴が正しく
  写るかは未知。実装は動くようにしてあるが、一度だけ `logger.warning` を出す
  → **2026-08-11 に目視検証済み: 参照は明確に効く**(顔・髪・服・小物まで反映、32B 正解と
  同水準。「2026-08-11: ref2va/i2va はどこまで動くか」の節参照)。warning は校正の事実の
  記録として残置
- **音声参照 + `fully_copy` との組み合わせ**(台詞を音声から入れる運用が投影TEでも成立するか)
- **4B の TE ロード 80.5s**(初回DL込み)。定常のロード時間は未測定

## transformer とデノイズ

### transformer int8量子化 (`H3_TRANSFORMER_QUANT`、既定 `none`)

`H3_TRANSFORMER_QUANT=int8` で transformer / transformer_ref を torchao
(`Int8WeightOnlyConfig(version=2)`、PR #14355ドキュメントのレシピ、torchao 0.17.0)で
int8化する。66.3GB → 34.0GB。品質は目視同等(PSNR 19dBは軌道分岐であり劣化ではない。
int8同士は同一seedでmp4バイト一致の完全決定論)、デノイズは+5s程度(dequantコスト)。

**int8時は両transformer同時常駐**(34+34+TE-nf4 21=~89GB)になり、ref2va⇔t2vaの
変種切替時の66GB級再ロードが消える(初回のref2vaのみ~36sのコールドロード)。実測:

| | bf16(既定) | int8両常駐 |
|---|---|---|
| t2va | 175-185s / peak 92.1GB | 177-196s / peak 59.7-91.1GB |
| ref2va(2回目以降) | 523s / peak 87.6GB | **463-471s / peak 74.5GB** |
| 変種切替の再ロード | 毎回~26-40s | **なし** |

フェーズ制御(int8両常駐時): t2vaデノイズ前は transformer_ref 常駐時のみTE強制解放
(89GB定常+activationsでOOMするため。実機確認)、ref2vaはTE強制解放不要になり
デコード窓も transformer_ref 常駐のまま通る。`PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` をrunnerが設定(int8ロード/解放サイクルの断片化で
「54GBしか使っていないのに15GB確保失敗」が実機再現したため。diffusers-serverでも
実績のある設定)。既定 `none` は従来とバイト一致(回帰確認済み)。

#### torchao の TF32 副作用の抑止 (`set_inductor_config=False`、2026-10-01)

torchao の `quantize_()` は config の `set_inductor_config=True`(既定)だと
`recommended_inductor_config_setter()` を呼び、`torch.set_float32_matmul_precision("high")`
(fp32 matmul の TF32 化)等を**プロセス全体**に立てる。fresh 量子化を行ったプロセスだけ
以後の fp32 計算(audio_vae 等)の数値が変わり、キャッシュ読み込みだけのプロセスと出力が
一致しなくなるため、本リポジトリの全 config(上記 `Int8WeightOnlyConfig(version=2)` と
`core/pruned.py` の各方式)は `set_inductor_config=False` で作る。量子化済み重みは
変わらず(フラグの副作用だけ)、キャッシュの `meta.json` も不変なので既存キャッシュは
そのまま使える。`/api/status` の `float32_matmul_precision` が `"highest"` のままで
あることで観測できる。

### FirstBlockCache によるデノイズ高速化 (`H3_CACHE`、既定 `fbc`)

ComfyUIコミュニティのEasyCache高速化に相当する、diffusers公式の step間キャッシュ
(FirstBlockCache)を `H3_CACHE=fbc`(既定)で有効化している。ステップ間で最初の
transformerブロックの残差変化が小さいとき、残りの計算をスキップする。

- `H3_CACHE_THRESHOLD`(既定 0.05): 実測でデノイズ 157s→118s(-25%、30step中7スキップ)、
  出力はキャッシュ無しと PSNR 31.8〜34.3dB・音声相関0.979 でほぼ同一(目視でも区別困難)。
- threshold 0.1 は 1.92x(デノイズ81.5s、14スキップ)だが構図が目視で分かるレベルで
  ドリフトするため既定にしていない(速度最優先の場合のみ)。
- `H3_CACHE=none` でキャッシュ無しの従来挙動に完全に戻る(バイト一致を回帰確認済み)。
- ピークVRAMは残差キャッシュ分 +0.7GB(91.4→92.1GB)。
- 実装メモ: `MiniMaxH3TransformerBlock` はPRブランチの `TransformerBlockRegistry` に
  未登録のため、runner側で `TransformerBlockMetadata` を登録してから `enable_cache()` を
  呼ぶ(venvのdiffusers本体は無改変)。リクエストごとに `_reset_stateful_cache()` +
  `cache_context("h3")` で包む(リセット漏れは前リクエストの残差による誤スキップを招く)。
  同一seed連続2本のmp4バイト完全一致でリセットの正しさを検証済み。

### Sage Attention (`H3_ATTN_BACKEND`、既定 `sage`)

sm_120(Blackwell)向けにソースビルドした SageAttention 2.2.0 を既定で使う
(ビルドは `scripts/build_sageattention.sh`、約2分。**必ず `MAX_JOBS=4 NVCC_THREADS=2` +
systemd-run のメモリ上限付きで実行**——無制限並列nvccはホストRAM枯渇でシステム巻き添え
事故歴あり。`CUDA_HOME=/usr/local/cuda-12.8` の明示が必要、既定のcuda-13.0はtorchの
cu128と不一致)。PyPI/コミュニティのLinux向けsm_120 wheelは存在しなかった(全てWindows)。

- 実測: デノイズ 118s→104s(**-12%**)。完全決定論(同一seed 2本バイト一致)。品質は
  目視同等(PSNR 21dBは int8-QK 近似による軌道ドリフトで劣化ではない)
- `H3_ATTN_BACKEND=default` で従来のSDPAに戻る
- FBCと独立に併用可: sage + `H3_CACHE_THRESHOLD=0.1` でデノイズ67s(-43%、リクエスト
  ~125s。FBC 0.1の構図ドリフト特性は既知どおり)
- hub系backend(`flash_hub`/`sage_hub`)は torch 2.9 向けビルドがHub側に存在せず不成立
  (2026-08-05時点。環境の問題ではない)

### Turbo LoRA (`H3_TURBO_LORA`、既定 `0`、2026-08-06 / **2026-08-08 lightx2v 版へ切替**)

> **2026-08-08 更新: 既定 LoRA を lightx2v 版へ切り替え、48GB(int8/低VRAM)でも
> turbo が使えるようになった。**
>
> - 既定: `H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo` /
>   `H3_TURBO_LORA_FILE=minimax_h3_fl2v_turbo_4step_v0.1.safetensors`(DMD蒸留、
>   Apache 2.0、rank128・312 Linear対象)。既定ステップ数は形式連動
>   (lightx2v=4 / Ostris に戻すと 8)
> - **int8 対応の理由**: キーが diffusers ネイティブ(to_q/to_k/to_v 分離)で
>   `fuse_projections()` 不要 → `torch.cat` を呼ばない → `Int8Tensor` 非互換を
>   踏まない。適用関数は形式自動判別(`detect_turbo_lora_format`、comfy 署名
>   `qkv_proj` を先に見る — Ostris 版も `token_refiner.` キーを持つため順序が重要)
> - **適用係数**(`H3_TURBO_LORA_SCALE`、空=形式別の実測既定): lightx2v は **0.094**
>   (Kijai 記載の 0.75 は ComfyUI の alpha 折り込み前提。生の B・A に 0.75 を掛けると
>   30steps でも完全ノイズ化 — 強度スイープはスパイク節参照)。Ostris は 1.0 のまま
>   (scale=1.0 は恒等なので旧挙動と bit 一致)
> - **E2E 実測**(RTX PRO 5000 48GB + `H3_LOWVRAM=1`、768²): t2va 4steps
>   **総所要 143s / デノイズ 29s**(非turbo 30steps は 351s)。**t2i(静止画)×turbo は
>   デノイズ 5.0s / 総所要 94s**。本実装経路の出力はスパイク出力と **mp4 md5 完全一致**。
>   turbo 連続2本で lowvram の再ロード後の再適用も正常、turbo=0 で通常経路(FBC 有効)
>   へ正しく復帰
> - **既知の性質**: turbo 時は音声レベルが非turbo比で大きめ(rms 0.018-0.042 vs
>   0.007。白色雑音ではないが強度によっては peak が 1.0 に近づく)
> - **併用制限**: `H3_LOWVRAM=group` とは形式を問わず併用不可(`enable_group_offload`
>   の `cpu_param_dict` が有効化時点で固定され、後から追加する LoRA バッファが
>   offload サイクルから欠落するリスク — 未検証のまま解禁しない)。comfy 形式
>   (Ostris)× int8 は従来どおり不可(リポジトリ名で予備判定+適用時に実ファイルの
>   キーで最終判定)。ref2va への適用(transformer_ref)は未検証のまま
> - v0.1(2026-08-07 公開)のため引き続き**既定OFF**(UI/リクエストの turbo で opt-in)

以下は Ostris 版 (comfy 形式) 時代の記録(bf16 経路では現在も有効):

`H3_TURBO_LORA_REPO=larryvrh/MiniMax-H3-Turbo-Lora` で Ostris氏学習中の4/8ステップ
蒸留LoRA(Apache 2.0、rank64・259 Linear対象)を適用し、既定ステップ数を8にする。
**プレビュー版LoRA(学習途上)のため既定OFF**。完成版が出たら再評価する。

実測(768²/5秒/seed12345): 8steps **87.7s(-46%)** で基準30steps(163.5s)に迫る品質、
16steps 98.4sで基準同等、4steps 39.6sは柔らかめだが破綻なし。
**コミュニティの「4〜7stepはダメ」はComfyUI標準サンプラーがデュアルスケジュール
(video shift12/audio shift3)を扱えないことが原因の可能性が高い**——本実装(diffusers
PRのscheduler/audio_scheduler分離+手動ループ)では4stepsでも音声破損は起きなかった。
シフト配線は改修不要(12/3はH3基準スケジューラの既定値で、sigma格子が作者リファレンス
実装とビット一致することを確認済み)。

実装メモ: LoRAキーはComfyUI命名のfused-QKV形式のため、`attn.fuse_projections()` +
ランタイムデルタ(W_eff=W+BA、fuseしない)で適用。**罠: `fuse_projections()` は旧
to_q/k/vを削除せず+12.8GBリークする**(明示deleteで対処)。AdaLNが `linear.weight` を
直接読むためラッパーに weight/bias 等のパススルーが必要。turbo時はFBC自動無効化。

#### turbo × 他機能の組み合わせ検証(2026-08-06)

デフォルトのtransformer経路(`transformer_quant=none`, `lowvram=0`)以外との組み合わせ
は当初「未検証のため予防的に拒否」だったが、実測でA/B検証した。

- **turbo × upscale(2xアップスケール hires-fix): 動作確認済み、解禁。**
  768²→1536²・seed12345で8/16stepsとも成功。8steps: 総所要210.3s(denoise 82.6s+
  decode 24.4s)、pass1=5/pass2=2steps(`H3_HIRES_DENOISE=0.35`既定分割)、
  peak VRAM 88.09GB——非turbo30stepsの基準645sから大幅短縮。16steps: 総所要331.8s、
  pass1=10/pass2=5steps、peak VRAM 88.35GB、8stepsより明確にシャープ。
  全フレーム(先頭/中間/末尾)を目視確認し色化け・チェッカーボード崩壊なし、音声も
  RMS/peakが正常値(無音・クリップなし)。turbo時はFBCが自動無効化されるため、
  hires-fixのFBCブックキーピング(`_fbc_last_step_was_skip()`)はtry/exceptで安全に
  no-op化される(懸念だった「turbo未対応のFBC呼び出し」は実害なしと確認)。
- **turbo × transformer int8(`H3_TRANSFORMER_QUANT=int8`、`transformer_both_resident`
  含む): 実測で不可と確定、拒否のまま維持。** `apply_turbo_lora()` の
  `attn.fuse_projections()` が `torch.cat([to_q.weight, to_k.weight, to_v.weight])`
  を実行するが、int8量子化された `to_q`/`to_k`/`to_v`(`H3_INT8_MODULES_TO_NOT_CONVERT`
  はこれらをスキップしない)は torchao の `Int8Tensor` であり、`aten.cat` カーネルが
  未実装のため `NotImplementedError: Int8Tensor dispatch: attempting to run
  unimplemented operator/function: func=<OpOverload(op='aten.cat', overload='default')>`
  で確実に失敗する(HTTP 500、リクエスト単位でクリーンに失敗しVRAMリークなし。
  直後の非turbo生成は正常動作を確認)。
- **turbo × lowvram=1: 実測で不可と確定、拒否のまま維持。** `lowvram=1` は
  `transformer_quant=int8` を強制するため、上記と全く同じ `Int8Tensor`/`aten.cat`
  エラーで失敗(同一エラーメッセージを実機確認)。
- **turbo × lowvram=group: 実測で不可と確定、拒否のまま維持。** 同じ理由
  (`lowvram=group` も `transformer_quant=int8` 前提)で同一エラー。**この失敗は
  group offloadフックの適用順序とは無関係**(sibling project の
  「LoRAをenable_group_offload()より前に適用する」という順序修正パターンは
  ここでは効かない——`fuse_projections()`自体がgroup offloadフックを一切介さず
  `torch.cat`だけで失敗するため、順序を入れ替えても直らないと判断し深追いしなかった)。
- **ref2va: 本タスクの検証対象外のまま**(元々タスクブリーフのスコープ外)。

### ステップ数の指針(蒸留モデル、実測 2026-08-05)

`num_inference_steps` はAPI/UIパラメータ。30(検証既定)に対し **20で-15%、16で-31%** の
デノイズ短縮。16/20とも単フレーム品質・時間方向の安定性に破綻なし(ただし構図は
ステップ数で変わる)。ドラフト用途は16-20、本番は30が目安。ステップを減らすと
FBCのスキップ機会も減る(30stepsで7スキップ→16stepsで0)ため、効果は単純比例しない。

### 2段生成による2xアップスケール (`/api/t2va` の `upscale=1`、既定OFF)

ComfyUIコミュニティの MiniMaxH3_LatentUpscaler と同系の hires-fix。低解像度(768²)で
前半をデノイズ → **x0推定値**の映像latentだけを bilinear で空間2x → フレッシュノイズを
`scheduler.scale_noise()` で再注入 → 残りの低σステップを1536²で仕上げ → デコード。
`H3_HIRES_DENOISE`(既定0.35)がパス2の担当デノイズ強度。UIはT2VAタブのチェックボックス。

実測(768²→1536²・5秒・30steps・seed=12345、fbc+bnb-4bit):

| | 合計 | デノイズ | デコード | ピークVRAM | 出力 |
|---|---|---|---|---|---|
| upscale=0 | 181s | 125s | 6.5s | 92.1GB | 768² |
| upscale=1 | 645s | 533s (パス1 78s + パス2 455s) | 24.7s | 88.0GB | 1536² |

- 構図・被写体は upscale=0 と一致し、毛並み・芝などの実ディテールが乗る。背景の
  細部(フェンス等)はパス2の再デノイズで軽微にドリフトする(hires-fixの性質)。
- 音声: latentテンソル自体はアップスケール処理で無変更だが、映像と音声は1つの
  パックドシーケンスで自己注意を共有するため、パス2以降の音声出力は upscale=0 と
  bit一致しない(相関0.89、非無音・品質同等。アーキテクチャ上の制約で仕様)。
- VRAM: パス2はシーケンス長~4倍のため、upscale=1 のリクエストではエンコード直後に
  TE-nf4 を解放してからデノイズする(次リクエストのエンコード時に遅延再ロード)。
- **実装の要点(実機でバグを踏んで確定)**: 補間対象は**ノイズ付きlatentではなくx0推定値**
  であること(ノイズ付きを補間すると市松状ノイズが増幅されて全面ノイズ化する。ComfyUI
  参考実装も denoised_output を使っている)。解像度変更時は `build_packed_sequence()` で
  position_ids/token_tags/各indicesを再構築し、`row_timestep_plan` も残ステップ分を
  作り直す。ModularPipeline の `_execution_device` はコンポーネント登録順の先頭モジュール
  で決まるため、TE解放後は `components.transformer.device` を明示的に使う
  (diffusers-server CLAUDE.md 23番・47番と同型の罠)。

### 任意サイズ・秒数の丸め (2026-08-06)

UIの解像度セレクトに **「任意 (32の倍数へ丸め)」** を追加し、幅/高さを自由に入力できる。
入力値は **エラーにせず H3 の規則へ丸める**:

- **キャンバス**: 32の倍数へ四捨五入し、256〜2048にクランプ(`app.py` の
  `round_canvas_value`)。H3のブロックは32の倍数でないと `ValueError` を出す仕様
  (`MINIMAX_H3_CANVAS_MULTIPLE`)なので、以前は端数を送ると400になっていた。
  ネイティブ範囲(短辺768・最大768×1344)を超える指定はUIが警告を出す(VRAM・品質は未検証)
- **秒数→フレーム数**: 5〜15秒にクランプ後、`17n+5` へ切り上げ(`align_num_frames`)。
  例: 6.3秒 → 158フレーム(6.58秒)。UIは送信前に実フレーム数と実尺をプレビューする
- API: `/api/t2va`・`/api/fl2va` に `height`/`width`(任意、指定時は `resolution`
  プリセットより優先)を追加。`/api/ref2va` は元から受け付けるが、こちらも丸めるように変更
- `/api/status` の `constraints` で丸め規則(canvas_multiple/min/max、fps、frame_step/offset)
  を公開し、UIは同じ規則でプレビューする(丸めの権威はサーバ側)

実測確認: `height=700 width=1000 seconds=6.3` で生成 → レスポンス `704×992 / 158フレーム`、
出力mp4も ffprobe で 992×704・158フレームと一致。

**UI実装の罠**: 幅/高さの `<input type=number>` に `min`/`max`/`step="32"` を付けると、
HTML5の入力検証(stepMismatch/rangeOverflow)が**丸める前の端数入力を不正扱いして
フォーム送信自体をブロック**する(1024×576のような偶然valid な値だけ通り、1000×700は
何も起きない、という分かりにくい症状になる)。丸める前提の欄には制約属性を付けないこと。

## 生成モード

### Ref2VA (オムニ参照生成、`/api/ref2va`)

順序付きの参照素材(**画像最大9・動画最大3・音声最大3、合計12**。音声単独は不可)から
動画+音声を生成する。参照の順序はプロンプト内ラベル(`<Picture i>` 等)とrotary配置に
対応するため意味を持つ。動画参照はサウンドトラックも条件付けに使われる。
**参照が音声ちょうど1本のときは秒数省略可**(音声の長さが生成尺になる。APIでは
`seconds=0`)。出力キャンバスは参照に縛られず、未指定なら16:9(1344×768)。

- **専用チェックポイント `transformer_ref/`(61.7GB、クラス/configは`transformer`と同一で
  重みのみ別)** を使う。t2va/fl2va用transformerとは同時常駐不可のため、runnerは
  変種切替(アクティブな片方だけ常駐、解放→再ロード)で管理する(`/api/status` の
  `active_variant`)。TE-nf4・VAE類・processorは両変種で共有。
- VRAM対策(実機OOM 3件を踏んで確定): 参照VAEエンコード完了後に transformer_ref を
  ロード(逆順は98.5GBでOOM)、デノイズ前にTE-nf4を強制解放(参照行でシーケンスが
  伸びるため。hires-fixと同じパターン)。共有text_encoderの解放は両パイプライン
  シェルの参照を両方消すこと(片方だけではrefcountが残りVRAMが返らない)。
- 実測(768x768指定→1344×768出力・30steps・seed=12345): 画像1枚参照 523s/87.6GB、
  画像+音声(尺は音声由来7.3s) 753s/88.1GB、画像2枚 635s/88.1GB。参照人物の同一性・
  複数参照の合成(人物が参照シーンのカフェに座る)を目視確認済み。ref2va⇔t2vaの
  往復切替も正常(切替込みt2va 188s)。
- UIは「Ref2VA (参照→動画)」タブ(複数ファイル選択、選択順=参照順)。

### 静止画モード (T2I、`/api/t2i`、2026-08-07 本実装)

H3を「超短尺動画→静止画取り出し」で画像生成の代用にするモード。前段の超短尺プローブ
(下記)の結論をそのまま本実装した。価値は「**H3と画風が完全に一致する静止画**」を
FL2VA の先頭フレームや Ref2VA の参照画像として作れること(速度は固定費支配のため
専用T2Iモデルには勝てない)。

- **API**: `POST /api/t2i` — `prompt` / `frames`(22既定 or 5)/ `resolution` or
  `height`+`width` / `num_inference_steps` / `seed` / 即反映パラメータ
  (cache/cache_threshold/attn/turbo)。レスポンスは超短尺 mp4 と**中央フレームの PNG**
  (`image_url`、`t2i_<ts>.png`)の両方
- **UI**: 静止画段の「T2I」タブ(秒数・upscale欄は非表示、フレーム数選択のみ)。
  ギャラリーにも PNG がタイル表示される(削除可・連結不可)
- **フレーム数**: 22 (0.917s) が既定。5 (0.208s) は実験的だが下記の VAE 修正で動く
- **実装の3点セット**(`core/runner.py`):
  1. `generate(still=True, still_frames=...)` — diffusers 側の最小尺5秒バリデーション
     (`MINIMAX_H3_MIN_DURATION`)は `MiniMaxH3SetupStep` の呼び出し1回分だけ
     `_relaxed_min_duration()` で緩和(生成は generation_lock で直列なので漏れない)
  2. **VAE 小クリップデコード修正** (`H3_VAE_SMALLCLIP_FIX`、既定1):
     `AutoencoderKLMiniMaxH3._decode()` は潜在1-2フレームで num_chunks=0 →
     `torch.cat([])` で落ちる上流バグがある(プローブで実機再現)。runner 側の
     monkeypatch で「1チャンク未満なら全トークンを単一 `_decode_clip()` で復号し、
     `frame_pre_padding`(3) とパディング末尾を切り落とす」分岐を追加。潜在2フレーム
     ×4 − 3 = 5ピクセルフレームで幾何が一致する。通常動画 (num_chunks>=1) は元実装へ
     そのまま委譲するため影響なし(venv の diffusers 本体は無変更)
  3. **decode 例外時の steady state 復元**: プローブで観測した「decode 例外 →
     transformer drop 済みのまま復元コード未実行 → ~98.5GB 残留で後続リクエストが
     連鎖 OOM」への対策として、`generate()`/`generate_ref2va()` の decode 部を
     try/except 化し、例外時も `_restore_decode_steady_state()`(VAE を CPU へ・
     モードごとの transformer/TE 再ロード)を実行してから再送出する。
     **実機検証済み** (2026-08-07): 一回限りの失敗注入フック `H3_DEBUG_FAIL_DECODE=1`
     (`os.environ.pop` で読むため同一プロセスの次リクエストから通常動作)で decode を
     故意に失敗させ、500 応答後に GPU が ~2GB まで解放され、**同一プロセスの次の
     リクエストが正常に 200 を返す**ことを確認した

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、768²・30steps・fbc+sage・
seed 12345)**:

| frames | デノイズ | デコード | ピークVRAM | 合計(毎リクエストの再ロード込み) | 品質(目視) |
|---|---|---|---|---|---|
| 22 (0.917s) | 29.0s (1.0s/step) | 1.8s | 35.0GB | 157s | 破綻なし・高品質 |
| 5 (0.208s) | 9.1s | 0.67s | 35.2GB | 125s | 破綻なし(史上初の実デコード成功) |

5フレームの音声 (0.208s) も非無音 (rms 0.055)。

#### 連続生成 (物語の場面画像などプロンプト違いの連番) の実測: **group モードは静止画に不利**

「常駐モード (`H3_LOWVRAM=group`) なら再ロード固定費が消えて連続生成が速いはず」という
仮説を、物語3場面のプロンプト (frames=22・768²・30steps・seed固定) の3連続実行で実測した
(2026-08-07、RTX PRO 5000 48GB)。**結果は逆で、lowvram=1 より約1.5倍遅い**:

| モード | 1枚目 | 定常 (2枚目以降) | 定常の内訳 |
|---|---|---|---|
| `H3_LOWVRAM=1` | 157s | **~157s** | TE+transformerロード ~120s + デノイズ 29s + デコード 2s |
| `H3_LOWVRAM=group` | 310s (TEコールド込み) | **~240s** | デノイズ **130s** + デコード 2s + デコード後のTE再ロード **75-97s** |

- **デノイズ 29s → 130s の悪化が本質**: group offload は1ステップごとに int8
  transformer 全 ~34GB をブロック単位で PCIe 転送する。転送コストはシーケンス長に
  無関係な固定費なので、計算が軽い超短尺 (22フレーム) では転送が完全に支配する
  (124フレームの通常動画では計算時間に隠れて相対的に軽い、という設計前提が
  静止画では崩れる)
- group モードはデコード窓で TE を force-free して後で再ロードする(24-32GB級の
  ヘッドルーム設計)。この箱では TE NF4 ロードが実測 75-97s かかる(96GB時代の
  15-40s より大幅に遅い)ため、この再ロードも定常コストに乗る
- 品質は lowvram=1 と同様に良好(目視)。ピークVRAMは 25.1GB
- **結論**: この箱での静止画の連続生成は **`H3_LOWVRAM=1` のまま回すのが最速
  (~157s/枚)**。さらに縮めるのは下の `/api/t2i_batch`(この実測を受けて実装済み)

#### 静止画のバッチ生成 (`/api/t2i_batch`、2026-08-07 本実装)

上の実測を受けて、`H3_LOWVRAM=1` の固定費 (~110s: TEロード75-97s + transformerロード
~20-35s) を**バッチ全体で1回に償却**するエンドポイントを実装した
(`core/runner.py` の `generate_still_batch()`)。generate() の lowvram choreography を
位相順に並べ替える:

    entry   : [nothing big resident]
    encode  : [TE-nf4]         全場面の setup/エンコード/layout/latents/timesteps
    denoise : [transformer]    全場面を順にデノイズ
    decode  : [vae pair]       全場面を順にデコード → PNG/mp4 保存(場面ごとに保存
                               しながら進むので途中失敗でも完了分は残る)

- **API**: `POST /api/t2i_batch` — multipart で `prompts` を場面数ぶん繰り返し送る
  (最大24場面)。`frames`/`resolution`/`num_inference_steps`/`seed`/即反映パラメータは
  全場面共通(変えられるのはプロンプトのみ)。UI は T2I タブの
  「バッチ連続生成(1行=1場面)」チェックで、プロンプト欄の1行を1場面として送る
- **場面間の共有状態リセット**(実装の要): スケジューラは sigmas/timesteps の値が
  全場面同一(同じ幾何・ステップ数)なので、デノイズ直前に `_step_index = None` に
  戻すだけでよい(`MiniMaxH3Scheduler.step()` は index を timestep 値から再導出する)。
  FirstBlockCache は場面ごとに `_reset_stateful_cache()` + `cache_context`
  (generate() の per-request リセットの per-scene 版)
- **等価性の実証**: 同一プロンプト・seed の逐次 `/api/t2i` とバッチの場面1で
  **mp4 と PNG がバイト完全一致**(md5一致)。位相並べ替えは数学的に無影響
  (なお group モード出力とは一致しない — FBC の効き方が実行経路で変わる既知の現象で、
  バッチ固有の差ではない)
- `H3_LOWVRAM=1` 以外のモード(大モデル常駐)では位相並べ替えの利得がないため、
  同じ API のまま逐次 generate() にフォールバックする(レスポンス形式は同一)

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、3場面・frames=22・768²・30steps)**:

| 方式 | 合計 | 1枚あたり | 内訳 |
|---|---|---|---|
| 逐次 `/api/t2i` ×3 | ~471s | **157s** | 毎回 ロード~120s + デノイズ29s + デコード2s |
| `/api/t2i_batch` (3場面) | **202.6s** | **67.5s** | ロード~110s(1回) + エンコード0.9s + デノイズ84.8s + デコード6.9s |

場面追加の限界コストは **~31s/枚**(デノイズ28-29s + デコード2.3s)なので、場面数が
増えるほど1枚あたりは 31s に漸近する(10場面で ~42s/枚、24場面で ~36s/枚の計算)。
ピークVRAMは 35.0GB で1枚生成と同じ(場面ごとに増えるのは潜在+prompt_embedsの数十MBのみ)。

#### 参照付き静止画 (ref2i、`/api/ref2va` の `still=1` と `/api/ref2i_batch`、2026-08-07 本実装)

下記スパイクの成立を受けて本実装した。**キャラクター参照から場面ごとの一貫した
静止画**を作るモード。作った静止画は FL2VA の先頭フレームや次の Ref2VA の参照に回せる
(物語の複数動画生成の素材作り)。

- **単発**: `POST /api/ref2va` に `still=1` + `frames`(22既定|5)を追加で渡す。
  `seconds` は無視され、中央フレーム PNG (`ref2i_<ts>.png`) がレスポンスの
  `image_url` に付く。実装は t2i と同じ3点セット (setup step の間だけの尺ゲート緩和・
  VAE 小クリップ修正・既存の decode 例外クリーンアップ) の流用
- **バッチ**: `POST /api/ref2i_batch` — 共通の `references` + `prompts`(最大24場面)。
  `H3_LOWVRAM=1` では `generate_ref2i_batch()`(t2i_batch と同じ位相並べ替えの
  ref2va 版: TE常駐で全場面エンコード → VAE窓1回で全場面の参照VAEエンコード →
  layout/timesteps → transformer_ref 1回ロードで全デノイズ → まとめてデコード)。
  他モードは逐次フォールバック。スケジューラ/FBC の場面間リセットは t2i_batch で
  md5 一致まで実証済みの同じ手法
- **UI**: 静止画段の「Ref2I」タブ(2026-08-09 に Ref2VA タブ内の「静止画」チェックから
  独立タブへ移動。下記「タブの2段組み」参照)+「バッチ連続生成(1行=1場面)」チェック

**実測 (2026-08-07、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、赤ずきん参照1枚・3場面・
frames=22・768²・30steps)**: 合計 494.7s = **164.9s/枚**(逐次 ~265s/枚の1.6倍速)。
内訳: エンコード位相 212.5s + デノイズ 178.1s (57.7-60.2s/場面) + デコード 6.6s。
ピークVRAM 36.8GB。品質・キャラ一貫性とも良好(3場面とも赤マントを保持し
各プロンプトに追従)。

- ~~既知の改善余地: 参照ビジョンエンコードが場面ごとに走る~~ → **下記
  「参照プレフィックスの KV キャッシュ共有」で解決済み (2026-08-08)**

#### 参照プレフィックスの KV キャッシュ共有 (`H3_REF_PREFIX_CACHE`、既定1、2026-08-08 本実装)

ref バッチ (`generate_ref_batch` = ref2i_batch/ref2va_batch) のエンコード位相で、
参照ラベル+ビジョン (~4104トークン、~65s/場面) の Qwen3-VL エンコードが場面ごとに
重複していた問題の解消。ref2va のトークン列は「参照が前置・プロンプトは末尾に verbatim
追記」(packing_ref2va の `build_ref2va_presentation` で確認) で、条件付け元の Qwen3-VL は
因果 LM なので、**参照プレフィックスのテキスト表現はプロンプトに依存しない** —
プレフィックスを1回だけ `use_cache=True` で通して `DynamicCache` に焼き、場面ごとには
プロンプト末尾 (14-33トークン、~0.2s) だけをキャッシュ継続する
(`core/runner.py` の `_encode_ref_prompts_shared_prefix()`)。

**検証 (`scripts/probe_ref_prefix_cache.py`、transformers 5.14.1 の
modeling_qwen3_vl.py を読んだ上で実機実測)**:

- プレフィックス部分の hidden_states[50] はフル計算と**ビット一致** (`torch.equal`)
- プロンプト末尾部分は相対RMS ~1.5% の丸め差が残る。原因はカーネル/GEMM のタイル経路が
  系列長で変わること — eager 固定でも同水準 (sdpa 固有ではない)。**ロジックバグでない
  ことはネガティブコントロールで確定**: わざと位置オフセットを壊した継続は相対RMS
  27-30% (20倍) に跳ねる。1.5% は「正しい計算の丸めノイズ」の水準
- 継続呼び出しのレシピ (罠3点): `attention_mask=None`
  (`compute_3d_position_ids` の arange 分岐 = past長からの連番 + rope_deltas 加算に
  乗せる)、`mm_token_type_ids`/`pixel_values`/`grid_thw` 系は**全て None**
  (`image_grid_thw` を渡すと `model.rope_deltas` が再計算・上書きされる)、場面ごとに
  `DynamicCache.crop(prefix_len)` で切り戻して直列再利用。`rope_deltas` は
  Qwen3VLModel の**インスタンス状態**なので、プレフィックス→全継続の間に他の TE 呼び出しを
  挟んではならない (ヘルパー1呼び出し内で完結させる設計)

**E2E 実測 (ref2i_batch、赤ずきん参照1枚・同一3場面・seed 12345・768²・30steps)**:

| | エンコード位相 | 合計 | 1枚あたり |
|---|---|---|---|
| 共有なし (H3_REF_PREFIX_CACHE=0 相当) | 212.5s | 494.7s | 164.9s |
| **共有あり (既定)** | **83.1s** | **350.1s** | **116.7s (-29%)** |

品質: 共有なしの同一 seed 出力との PSNR 21.9-27.4dB。バッチ出力は従来経路と
ビット一致しない (プロンプト末尾側の ~1.5% 丸め差が軌道をドリフトさせる —
Sage Attention 既定化時と同種・同水準の epsilon 級ドリフト) が、目視で構図・品質・
キャラ一貫性とも同等。ビット再現が要る対照実験は `H3_REF_PREFIX_CACHE=0` で従来経路に
戻せる。ref2va_batch (動画) もエンコード位相は同一コードなので同じ削減 (~130s/3場面)
がそのまま効く。参照VAEエンコード (~数s/場面) は共有せず場面ごとのまま
(効果が小さく、状態の別名参照リスクを増やさないため)。

#### 単発リクエスト間でのプレフィックス再利用 (`H3_REF_PREFIX_CACHE_SINGLE`、既定0、2026-08-24)

上の共有プレフィックスを、バッチだけでなく **単発 `/api/ref2va` のリクエスト間** でも
持ち越す。同じ参照画像に別々の音声・別々のプロンプトを当てる運用(mv_studio の
リップシンク生成がまさにこれ)で、~52s の参照ビジョンエンコードが2回目以降まるごと消える。

**既定 OFF**。プロセス寿命の KV キャッシュ(実測 1.04GiB VRAM)をリクエストをまたいで
持つため、「既存挙動を変えない」を優先している。有効化は gateway の overrides:
`{"backend":"h3","preset":"96gb","overrides":{"H3_REF_PREFIX_CACHE_SINGLE":"1"}}`。

**成立条件(コードが機械的に判定し、外れたら黙って従来経路へ落ちる)**:

- **画像参照のみ**(音声参照は何本あってもよい)。音声は `_build_presentation` が
  `"<Audio j>: "` のラベルしか出さず波形が conditioner に届かないため、**音声の中身も
  長さも変わってプレフィックスのトークン列はビット単位で不変**。
- **動画参照が1つでもあると使わない**(動画はピクセルがプレフィックスに入るため)。
  この場合はキャッシュを破棄して従来のフル計算に落ちる。

**キー**はファイル名ではなく「実際にプレフィックス forward へ入る値」
(トークン列 + 画像テンソルのバイト列 + `image_grid_thw`)の SHA-256。参照の枚数・順序・
解像度・ピクセルの中身のどれが変わっても必ず別キーになる(キー計算は実測 0.14-0.35s)。

**バッチ版に無い安全装置(`_encode_ref2va_prompt_prefix_cached` の docstring 参照)**:

1. `model.rope_deltas` は `Qwen3VLModel` の**インスタンス状態**で、プレフィックスが書き
   継続が読む。リクエストをまたぐと間に t2va のエンコードや `/api/prompt/enhance` が
   挟まって上書きされうるので、プレフィックス時の値を clone して持ち、**継続の直前に
   必ず書き戻す**。
2. `weakref` で text_encoder の同一性を見る。TE が解放/再ロード/量子化変更されたら無効。
   `_free_text_encoder()` 側でも能動的に捨てる(VRAM を即返すため)。
3. 継続のたびに `DynamicCache.crop(prefix_len)`(例外で抜けた場合の保険として、継続前にも
   長さを確認して切り戻す)。

**実測(96GB 機、preset=96gb + `H3_TRANSFORMER_QUANT=int8`、768x448・8秒・turbo・
4steps・seed 777、画像1枚+音声1本)**:

| | 1本目 | 2本目 | ピークVRAM |
|---|---|---|---|
| OFF(既定・従来経路) | 136.6s | **94.5s** | 73.79GB |
| **ON** | 140.4s(MISS、プレフィックス 52.5s) | **41.9s (-56%)** | 74.92GB (+1.1GB) |

- 同じ画像+**別音声**(8.0s→6.6s の別ファイル)も HIT: 37.0s。
- 参照画像を差し替えると `cache cleared` → MISS(93.6s)→ 元に戻すとまた MISS(94.1s、
  ただしキーは元の値に戻る=キーは決定論的)。
- ref バッチ経路も resident モードでは逐次 `generate_ref2va()` なので恩恵を受ける:
  ref2i_batch の2場面目が 71s → **21s**。

**数値**: MISS(その場でプレフィックス計算)と HIT(再利用)の出力は**ビット一致**
(192フレーム・音声とも `max diff = 0`)。つまりリクエスト跨ぎの再利用そのものは
完全に等価。一方、従来経路(丸ごと1本でエンコード)とはビット一致しない — 
プレフィックス/継続に分割することによる ~1.5% の丸め差が turbo 4steps で増幅されるためで、
バッチ経路(`H3_REF_PREFIX_CACHE`)が既に受け入れているのと同じ性質・同じ原因:

| 比較 | 平均画素差 | PSNR | 音声相関 |
|---|---|---|---|
| ON(HIT) vs OFF ※入力が完全同一のケース | 2.46/255 | 31.7dB | 0.9909 |
| **ON(HIT) vs OFF ※本来の用途(同じ画像・別音声)** | **12.14/255** | **20.45dB** | 音声 max diff 24943 |
| ON(MISS) vs ON(HIT) | **0(ビット一致)** | ∞ | 1.0 |

**2行目が本機能の狙いのケース**で、1行目より 11dB 悪い(画素の 67% が差 > 2)。
目視では構図・キャラクター同一性・口の形まで区別がつかず、これは劣化ではなく
**軌道の分岐**(同等品質の別の絵)だが、31.7dB だけを見て「ほぼ同一」と判断しないこと。

「MISS と HIT がビット一致」= **リクエストを跨いだ再利用そのものは完全に等価**。
差は「プレフィックス/継続に分割したこと」だけに由来する。

ビット再現が要る対照実験は `H3_REF_PREFIX_CACHE_SINGLE=0`(既定)のままにすること。

**H3_LOWVRAM 系との関係**: lowvram は毎リクエスト TE を解放するので、キャッシュはその
たびに捨てられる(= 効果なし)。実機で 48gb-lowvram + ON を確認済み: 毎回
`cache cleared: text_encoder freed` → 次リクエストは MISS、VRAM リークも例外もなし
(TE prune 併用の 51 層 TE ではキャッシュサイズも 0.84GiB に自動追従する)。

#### 参照付き動画のバッチ生成 (`/api/ref2va_batch`、2026-08-08 本実装)

ref2i と同じ位相並べ替え(実装は同一メソッド `generate_ref_batch(still=False)`)を
**通常尺の動画**に適用したもの。物語の各場面の動画を、共通参照+プロンプト違いで
連続生成する。尺 (`seconds`) は全場面共通で必須(スケジューラの sigmas/timesteps 値の
場面間同一性が位相並べ替えの前提のため。音声参照からの尺自動導出もバッチでは不可)。
UI は Ref2VA / Ref2I タブの「バッチ連続生成(1行=1場面)」チェック(タブと独立に
使える: Ref2I+バッチ = ref2i_batch、Ref2VA+バッチ = ref2va_batch)。

**実測 (2026-08-08、RTX PRO 5000 48GB + `H3_LOWVRAM=1`、赤ずきん参照1枚・2場面・
5秒・768²・30steps)**: 合計 803.2s = **401.6s/本**(逐次 ~485s/本の17%短縮@2場面)。
内訳: エンコード位相 151.3s + デノイズ 503.4s (247.9-255.5s/場面) + デコード 25.4s。
ピークVRAM 40.5GB。場面追加の限界コストは ~330s/本(ビジョンエンコード ~65s +
デノイズ ~250s + デコード ~13s)なので、場面数が増えると**約32%短縮**に漸近する。
動画はデノイズ支配のため静止画バッチほど劇的ではないが、品質・キャラ一貫性は
逐次と同様に良好(2場面とも赤マントを保持しプロンプトに追従)。

#### スパイク: Ref2VA×超短尺 (参照付き静止画、2026-08-07、実測済み → 上記の通り本実装済み)

「t2va で検証済みの22フレーム超短尺が、参照パッキング (packing_ref2va: 参照行が
生成行より多くなる) と重なっても品質が崩れないか」を実測した
(`scripts/probe_ref2va_short.py`、本体コード無変更・プローブ内 monkeypatch のみ)。
成立すれば**キャラクター一貫の場面静止画**が量産でき、物語の複数動画生成の素材作りが
速くなる。

条件: 参照 = 本リポジトリの t2i 生成物 (赤ずきんの少女 PNG) 1枚、同一プロンプト
(`<Picture 1>` 参照付き)・seed 12345・30steps・768²。RTX PRO 5000 48GB +
`H3_LOWVRAM=1`。

| 条件 | デノイズ | デコード | ピークVRAM | wall (逐次・ロード込み) | 品質(中央フレーム目視) |
|---|---|---|---|---|---|
| short22 (0.917s) | 58.1s | 1.7s | 36.7GB | 265s | **破綻なし・高品質** |
| baseline (5.0s) | 243.1s | 10.6s | 40.7GB | 485s | 破綻なし(アンカー) |

- **判定: 成立 (GO)**。分布外の重なり (超短尺×参照パッキング) でも品質は崩れず、
  参照の衣装・小物 (赤マント・ランタン・白いドレス)・画風は short22 でも保持された。
  構図もプロンプト (苔むした岩に座る) に正しく追従
- デノイズは t2va の22フレーム (29s) の約2倍 (58s) — 参照行のぶんパック列が伸びる
  ため。それでも 5 秒基準の 1/4
- 顔の同一性の厳密さは参照画像の写り (この実験では後ろ姿気味) に依存する。これは
  参照の性質であって超短尺の欠陥ではない
- 本実装 (API/UI 化、`generate_still_batch` の ref2va 版) は**未着手** — t2i と同じ
  3点セット (尺ゲート緩和・VAE 小クリップ修正・例外クリーンアップ) は流用可能で、
  バッチ位相並べ替えも t2i_batch と同じ手法 (スケジューラ `_step_index` リセット +
  per-scene FBC リセット、参照が全場面共通なら参照エンコード1回共有) が使える見込み

#### 前段の超短尺プローブ (2026-08-07、実測記録)

フレーム数は 17n+5 刻み(最小5=0.208秒、次が22=0.917秒)。プローブは monkeypatch のみで
検証した(`scripts/probe_short_frames.py` + `_one.py`)。当時の環境は RTX PRO 6000 96GB・
既定モード。

| フレーム数 | デノイズ(30steps) | デコード | 結果 |
|---|---|---|---|
| 5 (0.208s) | 完走 | **失敗** | VAEチャンク分割デコードの境界バグ(潜在2フレームで num_chunks=0 → `torch.cat([])`)→ **上記2で修正済み** |
| **22 (0.917s)** | **13.5s**(124fの1/7.6) | 1.2s | **成功・品質は5秒基準と遜色なし**(目視確認済み) |
| 124 (5s、基準) | 102.2s | 6.3s | 成功 |

- 22フレームは学習分布外(公式下限5秒の1/5)でも品質が崩れず、音声も非無音
- **教訓**: プローブ中、5フレームの例外がデコード後のクリーンアップ(transformer再ロード)
  前に発生し、~98.5GB残留で後続リクエストがOOMする連鎖を観測 → 上記3のクリーンアップ
  経路として本実装に反映済み

## UI

### UIからの設定切替 (2026-08-06)

環境変数でしか変えられなかったオプトイン設定を、性質で2つに分けてUIから操作できる。

#### 即反映(生成ボタン直上のチェックボックス、再ロード不要)

FirstBlockCache(+しきい値)・Sage Attention・Turbo LoRA。リクエストごとのパラメータ
(`cache` / `cache_threshold` / `attn` / `turbo`)として送り、`MiniMaxH3Runner.
apply_instant_settings()` が生成ロック取得後・denoise前に常駐transformerへ適用する
(`disable_cache()`/`enable_cache()`、`set_attention_backend()`、`_TurboLoRALinear.enabled`)。
**未指定なら従来どおりプロセス既定**なので既存のcurl/スクリプトは無変更で動く。
turbo有効時はFBCを自動的に無効化する(元の安全規則を踏襲)。

実測(同一seed、再起動なしで切替): FBC on 100.8s(6スキップ)/ off 129.3s(0スキップ)、
Sage on 129.3s / off(native) 158.5s、Turbo on(8steps) 38.9s。

#### 再ロードが必要(ヘッダの折りたたみ + 「適用(再ロード)」ボタン)

transformer int8・TE量子化・TEレイヤー削除・低VRAMモード・video VAE fp16。
`POST /api/settings/apply` が `core/settings.py` の `apply_reload_settings()` を呼び、
**プロセスは再起動せず** runner内で全モデルを解放→新設定でロードし直す
(自プロセスをkillすると誰も起動し直せずUIごと復帰不能になるため、
os.execv/self-kill の類は実装しない)。生成中は409、未検証の組み合わせは400。

実測: transformer_quant none→int8→none が 56.0s / 55.0s(GPU 87.5→55.0→87.3GB)、
lowvram 0→1→0 も往復動作。`GET /api/settings` が現在値と選択肢を返し、UIはこれで初期化する。

**UI実装の注意**: チェックボックスOFFは空文字ではなく明示的に `turbo=0` を送ること
(空文字は「未指定=サーバ既定」と解釈されうるため、`H3_TURBO_LORA=1` で起動した
サーバではチェックを外しても無効化されない恐れがある)。turboとupscaleは相互排他で、
片方を選ぶともう片方が自動的に解除・無効化される(サーバ側の400と整合)。

### 生成済み動画ギャラリー (2026-08-06)

結果表示の下に `outputs/` 直下の mp4 をタイル表示する。サムネイルはサーバで生成せず
`<video preload="metadata" src="....mp4#t=0.1">` に先頭フレームを描かせる(依存追加なし。
Ref2VAの参照タイルと同じ手法)。

- `GET /api/outputs`: **直下の *.mp4 と *.png**(静止画モードの生成物。`type` フィールド
  "video"/"image" で区別。`outputs/ab_*` 等の検証資料は対象外)。
  尺/解像度は ffprobe で取得し mtime+size をキーにメモリキャッシュ
- `POST /api/outputs/delete`: **パストラバーサル対策**(`/`・`\`・`..` を拒否し、
  resolve 後に `outputs/` 直下であることを検証=シンボリックリンク経由の脱出も遮断)。
  UIは `confirm()` 必須。実機で `../app.py` `/etc/passwd` `ab_*/...` 等が400になることを確認済み
- `POST /api/outputs/concat`: **連結順は「チェックした順」**(表示順は新しい順)。
  全入力のパラメータが一致すれば `concat demuxer + -c copy`(**再エンコードなし=劣化ゼロ**)、
  不一致なら `filter_complex` で先頭動画の解像度へ揃えて再エンコード(無音入力には
  `anullsrc` を合成)。2本未満は400、連結同士の同時実行は409(GPUを使わないので
  generation_lock は取らない)。**PNG(静止画)は連結対象外**(ffprobe が PNG も
  video stream として読めてしまうため、拡張子で明示的に400にする。UI側も選択に
  PNG が混ざると連結ボタンを無効化する)

**依存に関する注意**: このアプリの**生成物のmuxは PyAV**(`av.open()` で libx264+aac を
直接書き出す、`core/runner.py` の `_mux_mp4()`)で行っており、**ffmpeg コマンドは使って
いなかった**。ギャラリーの ffprobe/ffmpeg 呼び出しが**このアプリで初めての外部コマンド
依存**になる(この環境には `/usr/bin/ffmpeg` があり、不在時は `FileNotFoundError` を
捕捉して明示的にエラーを返す)。外部コマンド依存を無くしたい場合は「同一パラメータ限定で
PyAVによるパケット再多重化(=`-c copy` 相当)にし、混在時はエラーにする」のが現実的な代替
(パラメータ混在の再エンコードまでPyAVで実装するのは実質ffmpegの再実装になるため非推奨)。

**UI実装の注意**: 選択のたびに全タイルを作り直すと `<video>` が全数再生成されて
ちらつく(95枚で顕著)。選択状態はバッジ/チェックの**差分更新**にすること
(`updateGallerySelectionUI()`。一覧の再構築は `/api/outputs` 取得時のみ)。

## LLM プロンプト強化

### LLMプロンプト強化(2026-08-04追加)

ローカルLLM(gemma4-31B、OpenAI互換 `/v1/chat/completions`)で、入力プロンプトを
H3公式ガイドの形式へ整形する。クラウド版Hailuo AIの内部プロンプト整形層のローカル再現。

- 接続先: 環境変数 `H3_LLM_URL`(既定 `http://127.0.0.1:64660`)。接続不可時は502
  (生成機能には影響しない)
- `POST /api/prompt/enhance` {text, mode, seconds, task, lang}
- モード(UIの「LLM強化」ボタン+モード選択):
  - `storyboard`(既定): マルチショットのCUTタイムコード形式へ展開(総尺=seconds、2〜3カット、
    ハードカット・被写体同一性維持・カット毎の音指示。焦点距離は35/50/65/100mmに制限)。
    これはH3公式ドキュメントの記法ではなく本アプリの独自案(下記 `h3-official` 参照)
  - `brief`: 公式ブリーフ形式(シーン→被写体→アクション→カメラ→音→終わり方)の単一ショット詳細化
  - `h3-official`(2026-08-07追加): MiniMax公式スキル `h3-prompt-writing`
    (`MiniMax-AI/MiniMax-H3` リポジトリの `skills/h3-prompt-writing/`)のフィールド構造・
    記法へ厳密準拠。`task`(UIの現在のタブ = t2va/fl2va/ref2va)で参照ガイドを切替える:
    t2va/fl2va は `references/base-en.txt` の3フィールド形式
    (`integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`、
    `[Shot N] At MM:SS.SS` のカット記法・`<d>[言語] ...</d>` の台詞逐語保持)、ref2va は
    `references/ref-en.txt` の6フィールド形式(`subject_definitions` / `summary` /
    `retention_analysis` / `detailed_description` / `overall_soundscape` /
    `non_diegetic_music`)。出力は公式既定の英語(`lang=en`)、`lang=ja` で書き換え本文のみ
    日本語化も可能(フィールド名・`[Shot n]`等のラベル・タイムコード記法・`<d>`内の台詞は
    公式ルールどおり英語/原語のまま)。リファレンス本文は本リポジトリに同梱しない
    (ライセンス表記が無いリポジトリのため)。`venv/bin/python scripts/fetch_h3_skill.py`
    で `skills_cache/`(.gitignore 済み)へ事前取得しておく必要があり、未取得時は
    400+取得コマンド案内のエラーになる。システムプロンプトはSKILL.md+参照ガイド全文を
    要約せず投入(約18.8KB=t2va/fl2va、約26.6KB=ref2va)。
  - `translate`: 過剰創作なしの英訳(CUT構造は保持)
- 強化結果はプロンプト欄を置き換え(編集可)、「元に戻す」で1世代復元。生成結果には
  使用プロンプト全文を折りたたみ表示(強化あり/なし・モード間の比較評価用)
- UIに手書き用「プロンプトガイド」チートシート(公式ブリーフ構造・CUT記法・実測カット精度±1秒)を同梱
- 実LLM検証済み(gemma4-31B Q4_K_M): 3モード(brief/storyboard/translate)とも形式に従う
  ことを確認(storyboardはタイムコード合計・焦点距離制限も遵守、応答11〜17秒)。
  `h3-official` は2026-08-07追加時点でLLMサーバ未接続のため生成A/Bのみ実施(下記参照)、
  LLM経由での実応答確認は別途必要

### 公式スキル(h3-prompt-writing)モードの実機検証 (2026-08-07)

ローカルLLM(gemma4-31B)接続下で `h3-official` モードを実測した結果。

**構造適合**: 生成されたプロンプトは公式仕様に完全準拠していた。
- T2VA: 3フィールド(`integrated_multimodal_description` / `overall_soundscape` /
  `non_diegetic_music`)、`[Shot 1]` は時刻なし、`[Shot 2] At 00:05.000` の3桁記法、
  台詞は `<d>[Japanese] おかえり</d>` と原語で保持、話者ID `(S1)` も付与。応答6.2秒
- Ref2VA: 6フィールドすべて出力、`<Subject n>`/`<Picture n>`/`<Audio n>` のラベルと
  `fully_preserved` / `fully_copy` 等の関係マーカーも使用。応答12.8秒

**カット位置の実測**(768²・10秒・turbo 8steps、LLM生成プロンプト、各2seed。
ffmpeg のシーン検出 `gt(scene,0.15)`):

| 記法 | 指示 | 検出(seed 12345 / 777) | ズレ |
|---|---|---|---|
| 公式 `[Shot 2] At 00:05.000` | 5.000秒 | 4.875 / 4.875 | **-0.125秒(両seedで同一)** |
| 独自 `CUT 2 [6-10秒]` | 6.0秒 | 6.083 / 6.333 | +0.083 / +0.333秒 |

公式記法の方が**ばらつきが小さい**(2seedで完全同一)。ただし独自CUT記法も0.1〜0.3秒の
ズレに収まっており、**当初の手書きプロンプトによる1試行(独自記法+1.0秒)ほどの差はない**。
どちらも実用範囲で、公式記法は再現性が高い、というのが実測の結論。

**実装上の罠(実機で再現・修正)**: `lang=ja` を指定しても**英語で返ってくる**。
言語指示をシステムプロンプトの冒頭に置くと、その後ろに続く15.8KBの英語リファレンス
(英語出力の実例が並ぶ)に上書きされるため。**言語指示はリファレンス本文より後ろ、
最終指示として置くこと**で解消した(diffusers-server の Tポーズ実装で確立した
「プロンプトは末尾が最強」と同じ原理)。修正後は地の文だけが日本語になり、
フィールド名・`[Shot n]`・タイムコード・`<d>` タグ内の台詞は英語/原語のまま維持される。

### h3-official の品質保証: 検証 + 修復ループ (2026-08-08)

「LLMの性能限界なのか、プロンプトで直るのか」を推測で議論しないために、**まず故障率を
実測**し(`scripts/probe_h3official_compliance.py`、5入力×3回)、その結果に基づいて
対策した。LLM は gemma4-31B Q4_K_M / llama.cpp、**n_ctx=7680**。

#### ベースライン実測: 失敗は1クラスに集中していた

| 故障クラス | 発生率 |
|---|---|
| 構造・記法(フィールド名・`[Shot n]`・タイムコード・`<d>`タグ・話者ID) | **0/15 (0%)** |
| 時間配分 | 6/15 (40%) |
| 文脈溢れ | 0/15(余裕 2,900+ トークン) |

**「入力が変われば別の問題が出るのでは」という懸念は良い方に外れた** — 違反は散らばらず
時間配分の1クラスに集中していた。しかもその6件は性質が2つに分かれる:

- **入力が物理的に不可能 (3/6)**: 台詞の推定発話時間が尺を超えている。実例は38文字の
  独白(推定9.5秒)を9秒尺で要求したケースで、**LLMは1ショットに9秒全部を割く最善の
  配分をしていた**。公式仕様が台詞の逐語保持を要求する以上、短縮という逃げ道も無い
  → **どのモデルでも解決不能。ユーザーに返すべき情報**
- **LLMの配分ミス (3/6)**: 尺には収まるのに配置が悪い → 指摘すれば直せる部類

文脈: t2va は system 4,397 tok / 7,680 で余裕あり。**ref2va は 6,191 tok (81%) で余裕
約1,100 トークンしかない**(実測では6フィールド揃って正常終了したが、入力が長い場合は要注意)。

#### 対策(3層)

1. **バリデータ** (`core/prompt_check.py`): 決定的に判定できる規則を実装
   (F1 3フィールド / F2 先頭ショットに時刻なし / F3 カット時刻の厳密増加 / F4 尺内 /
   F5 ショット尺の下限 / F6 `<d>`タグと言語タグ / F7 台詞がショットに収まる /
   F8 話者ID)。F5・F7 は公式仕様に無い**本アプリの実用規則** — 公式は「カット時刻が
   尺内」としか言わないため、5秒尺で4.5秒にカットして最終ショット0.5秒、という
   字義どおりだが使えない出力を許してしまう(実際に発生)。意味的整合性(1ショット内で
   画角が混在する等)は近似ルールで**警告**にとどめる。**手書きプロンプトにも効く**
2. **システムプロンプト改善**: 英語版wrapperには**ガイド本文の後ろに何も無かった**
   (日本語版だけ `lang=ja` 修正時に後方ブロックを持っていた)。既定は `lang=en` なので、
   尺の制約を含む全指示が15.8KBのガイドに埋もれていた。両版の**末尾**に時間配分6ルール
   (ショット尺の下限・台詞の発話時間見積り・ショット数の目安・1ショット1構図・
   画面外音声の定型句・話者ID)を追加
3. **修復ループ** (`core/llm.py` の `enhance_prompt_checked`): 違反を検出したら
   その内容を日本語で突きつけて再生成(最大2回、`H3_OFFICIAL_MAX_REPAIRS`)。
   **違反が増える修復案は破棄**する(修復が別の箇所を壊す事故の防止)。
   入力の実現不可能性は**LLMに投げる前**に判定し `InfeasibleInputError` → 400 + 助言

#### 改善後の実測(同一条件)

| | clean | 入力不可を検出 | **違反が残存** |
|---|---|---|---|
| ベースライン | 9/15 | 0 | **6/15** |
| **改善後** | **12/15** | **3/15**(正しい挙動) | **0/15** |

**未解決の違反がゼロになった。** 全ケースが「妥当なプロンプトを返す」か「実行不能な理由を
助言付きで返す」かのいずれかになる。所要時間の中央値は 8.5s → 9.2s(+8%)で、修復ループは
必要なときだけ発動する(発動時は約2倍の21秒)。

API は `/api/prompt/enhance` の h3-official 応答に `violations` / `warnings` /
`check_report` / `attempts` / `repaired` を追加。UI は残った指摘をステータス欄に表示する
(**生成はブロックしない** — プロンプトは編集可能なので人間が最終ゲート)。



---

## FILE: docs/HISTORY.en.md (https://github.com/animede/Diffusers_minimax-h3/blob/HEAD/docs/HISTORY.en.md)

# Development history (minimax-h3)

[日本語](HISTORY.md) | **English**

Dated work log, verification notes and backlog, split out of the README. Each section is the original README text moved verbatim and describes the state at that time (where things changed later, [FEATURES.en.md](FEATURES.en.md) and [README.en.md](../README.en.md) take precedence).

- [Where things stood as of 2026-08-12](#where-things-stood-as-of-2026-08-12)
- [Chronological work log](#chronological-work-log)
- [Backlog and external events](#backlog-and-external-events)

## Where things stood as of 2026-08-12

### Recent updates (2026-08-11 to 08-12)

- **Both final goals reached**: full functionality on a single real RTX 4060 Ti 16GB (t2va
  768² peak 11.4GB; the maximum is 768×1344 at a real 15.2GB) / t2va 5s 768² completes at
  full resolution on 8GB × 2 (peak 7.23GB).
- Moved the decode-tail denormalization to CPU (bit-identical, lowers the decode peak on all
  configs).
- Fixed a dtype instant-death on `H3_VIDEO_VAE_FP16` × reference (added an fp16 autocast on
  the encode side too).
- Freed the residual pinned RAM in group mode, so t2va ↔ ref2va mode switching now works.

### Where things stand and where to resume (as of 2026-08-12)

Read this first when resuming work.

| What you want to know | Where to look |
|---|---|
| **Spec, performance, design** (what is achieved and how) | [docs/TECHNICAL_OVERVIEW.en.md](TECHNICAL_OVERVIEW.en.md) |
| **Pitfalls, failures, operational lessons hit** (to avoid the same holes) | [docs/internal/TECHNICAL_REPORT.en.md](internal/TECHNICAL_REPORT.en.md) |
| **Decisions on incorporating community improvements** | [docs/COMMUNITY_IMPROVEMENTS.en.md](COMMUNITY_IMPROVEMENTS.en.md) |
| Which mode loads/releases what, and when | [docs/RESIDENCY.en.md](RESIDENCY.en.md) |
| **What to do next** (not yet started / unverified) | This README's "[Waiting on external events going forward](#waiting-on-external-events-going-forward-backlog-as-of-2026-08-06)" §3 |
| Regression baselines for when diffusers gets bumped | [docs/internal/regression_baselines.json](internal/regression_baselines.json) |

**Current state**: diffusers is pinned to the merged version **f37ab93** (PR #14355 tracking
complete, all paths equivalent by identical-seed MD5). **The two final low-VRAM goals (real
16GB single / 8GB × 2) were reached on 2026-08-11**, broadening the pitch from just "speedup"
to "**runs on consumer-tier GPUs**." The recommended launch on the 48GB box is
`H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_KEEP_TRANSFORMER=1`
(t2i steady state 9.7s/image, t2va 5s = 44.2s). **When using ref2va, drop `H3_TE_DEVICE`**
(with only 20GB on the TE GPU, capacity is insufficient and it auto-rejects with 400). The
chronological work log is in this README's dated sections (read on below).

> **About the measurement environment**: this home is shared across two machines. The **96GB
> box** = RTX PRO 6000 Blackwell 96GB + an added RTX 4060 Ti 16GB (the low-VRAM goals were
> measured here); the **48GB box** = RTX PRO 5000 Blackwell 48GB + RTX 4000 SFF Ada 20GB (the
> recommended 48GB-tier config and turbo were measured here). Each dated section states which
> box its figures come from. A single 48GB card cannot physically load the default mode (bf16
> transformer, 66.3GB), so `H3_LOWVRAM=1` (48GB tier) or `H3_LOWVRAM=group` (24-32GB tier) is
> required.

## Chronological work log

### 16GB-tier verification result: not supported (floor is ~18GB, confirmed 2026-08-06)

With a 16GB ballast (15.5GB free), **OOM occurs near the end of TE loading (nf4
quantization)** (fails to allocate +250MiB at 15.37GB used). The pruned TE-nf4's resident
17.45GB is itself the floor, and video VAE fp16 — a decode-stage countermeasure — cannot
move this floor. **Completes successfully with an 18GB ballast** (peak 17.72GB, total
302s) — i.e., the practical lower bound of the current architecture is **~18GB** (the
24GB-tier configuration `H3_LOWVRAM=group H3_TE_PRUNE=1` works as-is at the 18GB tier
too). Breaking through 16GB would require streaming execution of TE or sub-4-bit
quantization (a separate, not-yet-started task).

### 2026-08-09: Migration to the PR #14355 merged version (f37ab93), stage 1 — t2i/t2va/batch, identical-seed MD5 exact match

Branch migrate-pr14355. Ported runner.py's t2i/t2va/still_batch/hires/turbo/FBC paths to
the merged version's new contract (see the impact investigation in §1), and swapped the
venv to f37ab93 (`--no-deps --force-reinstall`, other dependencies unchanged). **The
ref2va family is deferred to stage 2** (guarded in both app.py/runner.py, returning 503
with a clear message).

**Regression results (all compared at the same seed against the baseline recorded at
abc5e9b)**:

| Path | Result | MD5 |
|---|---|---|
| t2i turbo 4 steps seed=11 | steady 9.24s (9.7s before migration) | **PNG exact match** |
| t2i turbo 4 steps seed=12 | 9.24s | **PNG exact match** |
| t2i 30 steps seed=1 | 49.65s | **PNG exact match** |
| t2va 5s turbo seed=21 | 41.54s, peak 44.15GB (same value) | **MP4 exact match** |
| t2i_batch 2 scenes seed=11 | 16.47s (8.2s/image) | scene 1 = **exact match** with the single-shot seed=11 run |

As the impact investigation concluded that the numerical path (scheduler/transformer/VAE)
is unchanged, **the migration is bit-equivalent**. Performance and VRAM peak also match
pre-migration values exactly.

**Two new pitfalls hit during the port (both variants of the `_execution_device` issue)**:

1. **`audio_vae` moved ahead of `transformer` in the component order** (old:
   `text_encoder, tokenizer, processor, vae, scheduler, audio_scheduler, transformer,
   video_processor, audio_vae` -> new: `image_processor, text_encoder, tokenizer,
   processor, vae, audio_vae, scheduler, audio_scheduler, transformer_ref, transformer,
   video_processor`). Since `_pin_execution_device_to_compute()` only detached
   text_encoder and vae, the CPU-resident audio_vae was picked up as the first nn.Module,
   and the new layout step (whose contract places output on `_execution_device`) built
   position_ids on the CPU, causing a device mismatch inside rope() (reproduced on real
   hardware). -> fixed by also detaching audio_vae for that window
2. **In the batch's phase reordering, the layout stage's `_execution_device` becomes
   unresolvable** (with TE externally resident, transformer is not yet loaded and TE has
   already been detached) -> tensors end up created on the CPU. Since the values
   themselves are correct, explicitly move them to the compute GPU right before denoise via
   `_scene_state_to_compute()` (a no-op if already on the GPU)

### 2026-08-09: Merged-version migration stage 2 — the ref2va family also completes with identical-seed MD5 exact match

Continuation of stage 1 (above). Ported ref2va / ref2i / ref-batches to f37ab93's new
contract, and compared against **the baseline taken at the old pin abc5e9b** (temporarily
reverted to main + abc5e9b to record it), using the same seed, same reference images, and
same configuration (`H3_LOWVRAM=1 H3_TE_PRUNE=1`, no TE_DEVICE):

| Path (steps=8) | Baseline | After migration | MD5 |
|---|---|---|---|
| ref2i seed=101 | 206.1s | 201.4s | **PNG exact match** |
| ref2va 5s seed=102 | 332.5s | 322.5s (steady state; the first run's 496.8s was a cold disk cache) | **MP4 exact match** |
| ref2i_batch 2 scenes seed=101 (KV prefix-cache path) | 257.9s | 246.7s | **both scenes' PNGs exact match** |

**Two design decisions**:

1. **Consolidated the pipe shell into one**. The merged version's `MiniMaxH3Blocks` has
   merged component specs combining sub-blocks from all 3 workflows (t2va/fl2va/ref2va),
   with both the `transformer_ref` and `transformer` slots present in the same shell
   (`MiniMaxH3AutoDenoiseStep` branches from state **per call**). The old two-shell
   design's `_ensure_pipe_ref_shell` / `_sync_shared_components_to_ref` were reduced to
   no-ops, keeping only the `self._pipe_ref = self._pipe` alias — the existing VRAM
   choreography (the many places touching `_pipe_ref.transformer_ref`) survives unmodified
2. **The KV-prefix-cache split point is unchanged**. Since the new
   `MiniMaxH3Ref2VATextEncoderStep` also assembles the prompt as an `emit(text(prompt))`
   at the end of the presentation, the old design of "share the reference prefix + encode
   the prompt tail as a continuation" still holds as-is. Rebuilt on top of the new step's
   own instance methods (`_gather_vision_features` / `_build_presentation`) in place of the
   removed packing_ref2va function family (the DynamicCache / rope_deltas / continuation
   call argument conventions carried over unchanged from the old implementation's pitfall
   notes)

The rest of the port follows the same pattern as stage 1 (inserting AfterDenoiseStep,
inserting the new `MiniMaxH3PrepareConditionLatentsStep` /
`MiniMaxH3Ref2VAPrepareLatentsStep` steps, `reference_kind()` -> `entry.kind` /
`entry.has_audio`, moving app.py's reference construction to the
`MiniMaxH3ImageReference.from_file()` family). Since the new SetupStep no longer derives
duration from an audio reference internally when `seconds=None`, this was implemented on
the runner side as `_num_frames_from_audio_reference`.

This makes **all paths bit-equivalent on the merged version f37ab93**. There is no longer
any reason to revert to the old pin abc5e9b.

### 2026-08-09: Two-row tab layout (top row = video, bottom row = still image) and making Ref2I its own tab

To keep the tabs from growing into an ever-longer single row, **split them into two rows
by output type**:

| Row | Tabs (input in parentheses) |
|---|---|
| **Video** | T2VA (text) / FL2VA (frame) / Ref2VA (reference) |
| **Still image** | T2I (text) / Ref2I (reference) |

The row heading represents "output," and the tab name represents "input," so row x tab
reads as "video x reference = Ref2VA," "still image x reference = Ref2I."

**Why Ref2I was made its own tab**: it used to be a "still image" checkbox inside the
Ref2VA tab (only T2I had its own tab). This asymmetry stemmed from the fact that "putting
a still-image checkbox on the T2VA tab would conflict with that tab's keyframe input
fields (`still=True` cannot be combined with image/last_image — raises ValueError in
runner.py)," but **once split into two rows, it becomes more unnatural for "the still-image
row to have no reference-conditioned mode"**, so it was made independent. The reference
upload field is shared with Ref2VA (`isRefMode()`); the only differences are how duration
is decided (seconds vs. frame count) and the `still` flag.

**Batch is not made into a tab**: batching is **orthogonal** to the still-image/video axis
(Ref2I+batch = `/api/ref2i_batch`, Ref2VA+batch = `/api/ref2va_batch`, T2I+batch =
`/api/t2i_batch`). Turning it into a tab would balloon the layout to 3 columns x 2 rows, so
it remains a checkbox within each tab.

**Tab buttons use a 2-line structure** (line 1 = model name, line 2 = input type,
`.tab-btn small`). At a panel width of 378px, the button's inner width is 77px, while
"T2VA (text)" is 95px and would wrap, making button heights uneven, so it was designed as
2 lines from the start, independent of width (measured in-browser: all buttons align at
50px, and line 2's position also aligns at y=26px).

#### Lazy-loading gallery thumbnails

A problem where accumulating outputs made page rendering heavy (measured: at 165 outputs,
browser rendering choked and screenshots timed out at 30 seconds). The cause is that
**`<video preload="metadata">` fetches metadata for every tile all at once, as soon as it's
displayed**.

`<img>` respects `loading="lazy"`, but **`<video>` does not honor the same attribute**, so
the src is evacuated to `data-src`, and set only when the element approaches the viewport
via `IntersectionObserver` (`rootMargin: 300px`, pre-loading one screen ahead). On every
re-render, `disconnect()` is called before re-observing (since the observer holds
references to elements it's watching, failing to detach it would keep old tiles from being
freed).

**Measured (browser network log)**: mp4 requests on page load went from **165 to 0**. PNGs
(55 of them) are left to the browser's standard lazy loading.

#### UI language switching (i18n)

The toggle at the top right (`日本語` / `English`) switches **without a page reload**. The
choice is stored in `localStorage['h3_lang']`; the first visit is decided by
`navigator.language`.

- **One dictionary** (`I18N = { ja, en }`, 165 keys). Lookups go through `t('key', {vars})`,
  and a missing key falls back to Japanese
- Static text is marked with `data-i18n` / `data-i18n-html` (for text containing `<b>`) /
  `data-i18n-placeholder` / `data-i18n-title`, and `applyI18n()` rewrites them in one pass
- **Text generated by JS does not follow an attribute rewrite**, so on switching,
  `rerenderDynamicI18n()` **re-renders** the gallery, the result panel, the reference tiles
  and the various hint lines (the most recent result is held in `lastResultData` and passed
  back to the same render function)
- Splitting the HTML into two per-language files was rejected: the UI keeps changing, and
  double editing guarantees one side rots

**Not translated**: JS comments (103 lines, design notes for developers), server-returned
error messages (`data.detail`; app.py stays Japanese), technical identifiers such as
`te_quant:`, and the language toggle's own labels.

**Verified in the browser**: switching JA↔EN also swaps gallery tiles (`静止画` ↔ `Still`),
the result placeholder and the hint lines; the choice survives a reload; the six submit
routes (`/api/t2i`, `/api/ref2va`+still=1, `/api/ref2i_batch`, `/api/ref2va_batch`,
`/api/ref2va`, `/api/t2va`) are unchanged after switching; and no console errors occur.

### 2026-08-10: Lip sync from an audio reference (`fully_copy`) — measurements, and the bug found along the way

**Motivation**: the technique of replacing the text encoder with a small model + a
projection matrix
([ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)) can shrink
the TE from 15.7GB to 5.2GB, but by the author's own measurements **speech degrades** (4B is
unintelligible, 8B drops the language and speaks English). So the hypothesis: if the dialogue
is fed **from an audio reference, it no longer depends on the TE's speech ability**. This
section verifies that.

**Result: the hypothesis holds.**

| Verified item | Result |
|---|---|
| Does the dialogue content carry over from the audio reference | **Holds** — detected language ja (0.964), recognized "今日は良い天気だね", **100% character match** |
| Lip sync | **Holds** — correlation between mouth opening and the audio envelope **+0.745** (offset 0ms) |
| Character consistency | Holds (face, hairstyle, uniform as in the reference) |
| 1:1 copy of the audio | **Does not hold** — waveform correlation 0.112, duration 5.22s → 5.88s |

The decisive point is that **not a single character of Japanese dialogue was written in the
prompt** (no `<d>` tag; the only instruction was "match the mouth to the dialogue of
`<Audio 1>`"). The same Japanese came out anyway = **the dialogue does not pass through the
text encoder**.

**Despite its name, `fully_copy` is not a signal copy.** The official guide says *"The
complete source audio serves as the target video's complete final audio track"* / *"reused
1:1"*, but the actual behavior is "**regenerate the same content**". The voice timbre
changes, and the duration is decided by the model's own constraints (141 frames = 5.88s). If
you want the original audio verbatim, you have to swap it in after generation (with +0.745
sync correlation this looks practical; unverified).

Also, the mouth is open for 4.1s against 1.6s of actual audio — the mouth tends to keep
moving after the speech ends. Acceptable for anime-style footage, but not a strict
phoneme-level match.

Conditions: 96GB box, TE nf4 + transformer bf16 (unquantized), 768×1344, 141 frames,
30 steps, seed 777, total 553.9s, peak 87.67GB.

#### [Bug fix] ref2va with an audio reference always crashed under sage attention

This fired during the verification above. **Passing a reference containing audio crashes
reliably**:

```
sageattention/core.py: assert dtype in [torch.float16, torch.bfloat16]
AssertionError: Input tensors must be in dtype of torch.float16 or torch.bfloat16
  (origin: autoencoder_kl_minimax_h3_audio.py -> dispatch_attention_fn)
```

**Cause**: `MiniMaxH3AudioAttnProcessor` calls `dispatch_attention_fn` with
`backend=self._attention_backend` (default `None`), so **the backend is resolved
globally**. This app only calls `set_attention_backend()` on transformer /
transformer_ref, but with `H3_ATTN_BACKEND=sage` (the default) even the audio_vae's
attention flows into sage. The audio_vae, however, is **fixed to fp32 by design** (bf16
drops the volume by about 20dB), and sage only accepts fp16/bf16. **The per-request
`attn=` override cannot work around it either** (it only affects the transformer family).

**Fix**: call `audio_vae.set_attention_backend("native")` right after loading the
audio_vae, pinning just this module to native. The audio VAE's compute is small and sage
gains nothing there.

**This was a hole in the tests**: the path is only exercised by "references with audio",
so every existing ref2va regression (**48GB box, int8, image references only**) slipped past
it. The unquantized ref2va on the 96GB box had also gone unverified since the merged-version
migration. The fix was confirmed both by ref2va with an audio reference passing with sage
still the default (186.3s, Japanese ja 0.976, 100% character match) and by image-only ref2i
remaining intact (133.9s).

### 2026-08-10: Breaking down the decode-phase peak VRAM — the composition, and the 15% we took from it

To decide whether to port ComfyUI's
[PR #15446](https://github.com/Comfy-Org/ComfyUI/pull/15446) (chunk-streaming the H3 VAE so
the decode VRAM becomes duration-independent), **we first broke down what our own
decode-phase peak of 16.29GB is made of**.

#### The breakdown (768×1344, 107 frames, fp32, measured)

| Component | Measured | Share |
|---|---|---|
| **The weights of the two VAEs (resident)** | **11.02 GB** | 66% |
| video decode activations | 3.08 GB | 19% |
| `postprocess_video` | 0.00 GB | 0% |
| audio decode | 0.00 GB | 0% |
| **uint8 conversion + CPU transfer** | **2.49 GB** | 15% |
| Total | **16.59 GB** | (nearly matches the README's measured 16.29GB) |

**What we learned**: two-thirds of the peak is **weights**, and chunking removes not a
single byte of that. What PR #15446 targets is the 3.08GB of activations — **porting it
saves at most 19%**.

That the decode peak scales with duration was confirmed separately (latent 32 → 48 frames
takes 3.08 → 4.80GB, linear at about 30MB/frame). So the PR's observation itself is
correct, and it helps more at longer durations.

#### The 15% taken first — the uint8 conversion's intermediate tensors

Only the breakdown revealed that **our own code was stacking up 2.49GB**:

```python
frames_uint8 = (video_tensor.permute(0,2,3,1).float().clamp(0,1) * 255).round().to(torch.uint8).cpu().numpy()
```

`float()` / `clamp` / `*255` / `round()` each return an **intermediate tensor of the full
length**, and finally the uint8 copy is made on top. This was consolidated into
`frames_to_uint8()`, which converts 8 frames at a time and writes directly into the CPU
output array (the same code had existed in 4 places; now one).

| | Peak |
|---|---|
| Current (all at once) | +2.65 GB |
| **Improved (8 frames at a time)** | **+0.03 GB (-99%)** |

The order of operations is identical to before, so the rounding does not change either.
**The PNG MD5 of an identical-seed production generation matches exactly**
(`66a59ff92d653f1284cabe76bdb6501c`) — confirmed.

#### Porting PR #15446 is on hold

Against 3.08GB of activations (19%), it would require a monkeypatch replacing the
upstream `_decode` wholesale, incurring a tracking cost. Having taken the cheaper 15%
first, the next candidate with better expected value is **fp16-ifying the audio_vae (part
of the 11GB of weights)** (the current fp32-fixed rationale is the measured "bf16 drops the
volume by about 20dB", but **fp16 is unverified** — bf16 has a 7-bit mantissa, fp16 has 10
bits, and small-amplitude audio can behave differently).

### 2026-08-11: Added an RTX 4060 Ti 16GB to the 96GB box — stage 1 toward the low-VRAM goals (TE onto the second GPU)

An **RTX 4060 Ti 16GB (sm_89) was added as cuda:1** to the 96GB box (RTX PRO 6000). The
final goals are **(A) running on a single 16GB card, and (B) running on a two-card 8GB×2
setup**. As stage 1, "the projected TE (4B NF4) resident on the 4060 Ti" was measured.

Launch: `H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TE_DEVICE=cuda:1` (no lowvram;
transformer in bf16, resident on GPU0).

| Measurement | Value | Note |
|---|---|---|
| GPU1 resident | **3213 MiB (3.37GB)** | Matches the derived 3.11GB+ε. Does not grow after generation |
| t2i 768² steady state | 43.94s (denoise 13.48s, decode 0.88s) | ~10s more than the previous day's single-GPU 33.5s (see below) |
| t2va 5s 768² | 134.58s (denoise 102.57s), peak 71.19GB | |
| PSNR vs the 32B baseline | **22.36 dB** (sharpness 37 vs baseline 48) | Same level as the old matrix's 22.49dB |
| PSNR vs the old-matrix NF4 (single-GPU) | **39.29 dB** | Essentially the same picture despite recalibration + a different GPU |
| t2i PNG MD5 | `3cd088df882e37547219b9816a217b91` | New matrix + sm_89, so it differs from the old anchor (as expected) |

#### Two bugs found and fixed (surfaced by a Sonnet agent's verification)

1. **A rope device-mismatch crash in the combination of plain mode (no lowvram) +
   `H3_TE_DEVICE`**. Detaching the TE makes `_execution_device` fall to the CPU-resident
   audio_vae — a known pitfall — but this combination had never run before, and the
   layout-through-timesteps span of the two non-lowvram branches (catch-all /
   bnb-4bit+fl2va) sat outside `_pin_execution_device_to_compute()`. Fixed by wrapping
   them in the pin only when `self._te_external` (the usual TE-cohabiting path is
   byte-for-byte unchanged).
2. **The projection matrix's default filename 404'd**. The distributor had moved
   `h3_qwen3vl_4b_tap24.safetensors` to obsolete/ and replaced it with the **recalibrated**
   `mmh3-4b-ClipProj.safetensors` (training 1,666 → 5,664 prompts, cos_test 0.711 → 0.717,
   W's cosine 0.9596 vs the old — effectively a different function). The default was
   updated to the new filename. The old matrix survives in the local HF cache (snapshot
   3f762f19); passing an absolute path via `H3_TE_PROJ` reproduces it.

#### Observation: steady-state t2i is +10s vs single-GPU

Denoise + decode stay at 14.4s; **the per-request fixed cost is about 29s** (vs about 19s
single-GPU). The main suspects for the difference are the encode running on the sm_89
4060 Ti (NF4 dequant is slower) + the PCIe transfer of the cond. Meanwhile t2va did not get
worse at 134.6s — the longer the duration, the thinner the fixed cost spreads. On the 96GB
box the TE fits on GPU0 anyway, so **there is no practical benefit to using the second GPU
on this box** (the real targets are the 16GB-single and 8GB×2 configurations).

#### Next stages (unmeasured)

- **Stage 2 = goal A**: cap GPU0 to a 16GB equivalent with ballast, `H3_LOWVRAM=group` +
  projected TE **cohabiting** (no TE_DEVICE). Derived 3.11+1.4+6.6 = 11.1GB should fit, but
  **the te_proj × group combination has never run**
- **Stage 3 = goal B**: 8GB×2. The TE's 3.41GB fits on GPU1, but GPU0's blocks +
  activations of 8.0GB exceed the effective budget of 7.1GB by 0.9GB — trimming the
  activations via resolution/duration needs consideration

### 2026-08-11: Stage 2 = goal A achieved — t2i / t2va complete on a real RTX 4060 Ti 16GB **alone**

Verified not with ballast simulation but on the real added card alone, via
`CUDA_VISIBLE_DEVICES=1`. Launch: `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3
H3_VIDEO_VAE_FP16=1 H3_ATTN_BACKEND=default` (sage is an sm_120-only build, so back to SDPA.
No TE_DEVICE = projected TE cohabiting. **te_proj × group ran here for the first time** —
it worked as-is with no fix).

| Measurement | Value | Verdict |
|---|---|---|
| Resident right after startup | 251 MiB (transformer on CPU, group offload) | |
| Peak during t2i | 7.4GB (nvidia-smi 7615MiB) | |
| Peak during t2va 5s denoise | **11.4GB (nvidia-smi 11681MiB)** | **~4.7GB left** on a 16GiB card |
| During t2va decode | ~7.0GB (fp16 decode) | |
| t2i steady state | 498.5s (denoise 479s, **16.5s/step**) | |
| t2va 5s | 1499s ≈ 25 min (denoise 1445s, 49.8s/step) | |

**The derived 11.1GB of VRAM landed almost exactly on the measured 11.4GB, and goal A
(single 16GB) holds with plenty of headroom.** Not a single OOM or error.

#### The time is heavy mainly because "this box's second slot is PCIe x4"

group offload streams ~34GB of int8 weights CPU→GPU every step, but the slot holding the
4060 Ti is **Gen3 x4 (effective ~3.5GB/s)**. The transfer alone comes to ~10s/step, which
is consistent with the measured 16.5s/step. **On a proper Gen4 x16 slot the transfer is
~1/8**, so this time is a value of "this box's slot", not "the 16GB card's performance".
The t2va's 49.8s/step is that plus the SDPA compute for 124 frames (sm_89) on top.

One more thing: **the FBC cache is not saving a single step** (`cache_skipped_steps: 0`,
threshold 0.05). In contrast to the PRO 6000 + sage + bf16 trajectory, where most steps
were saved. On the int8+SDPA trajectory the residual apparently never drops below the
threshold. Threshold tuning leaves up to a 2x speedup on the table (unverified; a quality
trade-off).

#### Quality: not "degradation" but a trajectory divergence — and prompt fidelity actually improved

PSNR is 7.40dB vs the 32B baseline / 7.43dB vs the previous stage — numerically
catastrophic, but **visually it is a different story**:

- The 32B baseline and previous stage (bf16+sage) outputs: at seed 4242 they had converged
  on an **anime-style sunset lake** (letterboxed). The prompt's "photorealistic" and
  "snow-capped peaks" were not reflected
- This run (int8+group+SDPA): **a photorealistic snowy mountain and a misty morning lake**
  — faithful to the prompt and high quality

So the int8+SDPA combination moved the generation trajectory to a different attractor, and
prompt adherence actually came out better. Cross-config PSNR/MD5 comparisons never were
valid (beyond the known int8 trajectory divergence of ~19dB), so **judge cross-config
quality by eye**.

### 2026-08-11: Stage 3 = goal B achieved — **t2va 5s 768² completes at full resolution on 8GB×2**

Both GPUs were capped to headless-8GiB-card equivalents with
`scripts/vram_ballast.py --target-free-gb 7.9` (GiB) to simulate. Compute side = 4060 Ti,
TE side = PRO 6000 (`CUDA_VISIBLE_DEVICES=1,0` maps them to the app's cuda:0/cuda:1).
Launch is the same env as goal A + `H3_TE_DEVICE=cuda:1` (**te_proj × group × TE_DEVICE
also ran here for the first time** — worked without modification).

| Measurement (8GiB×2) | Value |
|---|---|
| t2i 768² | Success. Peak 6.4GB, total 512s |
| t2va 5s 768² denoise | **29/29 completed** (51.3s/step — the prior prediction of "0.9GB over" was wrong; it fit) |
| t2va 5s 768² overall | **Success**. Total 1534s, decode 35.2s, API peak **7.23GB**, measured workload ~8069MiB |

**No reduction of resolution or duration was needed** (the 640²/3s/512² search ladder went
unused).

#### But one thing was fixed: an 838MiB all-at-once fp32 conversion at the decode tail OOM'd → denormalization moved to CPU

The first attempt OOM'd on **the final line of the decode tail** after the denoise had
completed. The last line of the upstream `decoders.py`'s
`MiniMaxH3VideoDecodeStep.__call__`,
`(video.float() * pixel_std + pixel_mean).clamp(0,1)`, converts the full-length fp16
decode result to fp32 on the GPU all at once — at 768², 124 frames that is
124×768×768×3×4B = **838MiB** of temporary allocation (matching the OOM message's "Tried
to allocate 838.00 MiB" exactly). Same family of problem as the all-at-once conversion
crushed by `frames_to_uint8`.

The remedy follows the no-venv-modification rule: a runner-side subclass
(`_cpu_norm_video_decode_step()`, duplicating f37ab93's `__call__` with exactly one
change): **move to CPU while still fp16, then denormalize**. Elementwise fp32
mul/add/clamp round identically under IEEE754 on CPU and GPU (no reductions, no FMA
fusion), so the output is bit-identical — **confirmed by measurement: the identical-seed
PNG MD5 matches exactly (`1a2a136b61234b4917465604ac35cca2`) before and after applying**.
The only added cost is one PCIe transfer of the full-length fp16 ~420MiB. Applied
unconditionally to every path (t2va/t2i/ref2va/batch), lowering the decode-phase peak by a
few full-length fp32 buffers on every configuration (the decode-phase numbers in
docs/RESIDENCY.md will be re-measured at the next update).

#### Summary: both final goals achieved

- **Goal A (single 16GB)**: full functionality on a real RTX 4060 Ti 16GB, peak 11.4GB
  (~4.7GB headroom)
- **Goal B (8GB×2)**: compute 8GiB + TE 8GiB completes t2i / t2va 5s 768² (peak 7.23GB)

The remaining caveats are speed only (this box's second slot is Gen3 x4, hence
16.5-51s/step; on Gen4 x16 the transfer is ~1/8) and the FBC not engaging on the int8+SDPA
trajectory (threshold tuning unverified).

### 2026-08-11: How far do ref2va/i2va go — two latent bugs found and fixed, and the low-VRAM boundary settled

The reference family (ref2i / i2va = image-reference ref2va / audio reference / 768×1344)
was verified on the goal A/B configurations. The first pass was a **total wipeout**, but
the cause was not VRAM — it was **two latent bugs that only fire in combinations never
exercised together until today**. After the fixes, the boundary is decided plainly by VRAM
amount.

#### Bug 1: `H3_VIDEO_VAE_FP16=1` × any reference always crashes, regardless of VRAM (dtype mismatch)

`H3_VIDEO_VAE_FP16=1` permanently casts the VAE weights to fp16. **Decode** is consistent
because the upstream step raises its own fp16 autocast, but the **encode** side
(`encode_vae_condition` in encoders.py — used by ref2va's references and fl2va's keyframe
conditioning) has no autocast and passes pixels it explicitly promoted to fp32 into
`vae.encode()` → instant death with
`Input type (float) and bias type (c10::Half) should be the same`. It crashes even at
96GB. It stayed latent because every ref2va regression so far had used the fp32-VAE
configuration.
**Fix**: in `_load_vae`, immediately after the fp16 cast, wrap `vae.encode` in an fp16
autocast symmetric with the decode side. Precision is within design
(`encode_vae_condition` already rounds its result down to fp16 itself before returning).

#### Bug 2: in group mode, the t2va→ref2va mode switch was permanently impossible (residual pinned RAM)

group offload (use_stream=True) places the ~34GB of int8 weights in **pinned memory**. The
del+gc in `_free_transformer` leaves the pages **held by torch's host-side caching
allocator without returning them to the OS** (`torch.cuda.empty_cache()` is device-side
only), so MemAvailable stays ~34GB short and the subsequent transformer_ref load is
rejected by the RAM guard (which asks for 40GB).
Measured: after freeing, avail was still 38.6GB (RssShmem 45.6GB residual).
**Fix**: in `_free_transformer` / `_free_transformer_ref`, only in group mode, add
`torch._C._host_emptyCache()` (a private API, so guarded with getattr). After the fix,
avail recovers fully to **85.8GB** immediately after freeing, and the t2va→ref2va switch
succeeded in group mode for the first time.

#### The boundary after the fixes (measured)

| Test | 8GiB×2 | 16GB single (TE cohabiting) |
|---|---|---|
| ref2i (reference still 768²) | **○** peak 6.69GB, 1059s | (not run — established at 8GB) |
| i2va (768² 5s) | × denoise OOM | **○** peak 9.41GB, ~39 min |
| Audio reference (looped 6.9s → 7.29s generated) | × denoise OOM | **○** peak 11.96GB, ~54 min |
| 768×1344 5s | × denoise OOM | **○** peak 13.37GB (nvidia measured 15.2GB), ~66 min |

- **8GiB×2 goes as far as ref2i**. Video with a reference has a longer sequence than t2va
  (7.23GB) by the reference tokens, and even the shortest 768²/5s cannot fit the denoise
  activations (the required amount is a measured 9.41GB).
- **16GB single runs the entire reference family too**. 768×1344/5s (real peak 15.2GB) is
  the practical ceiling.
- A note on audio references: supplying audio shorter than 5 seconds returns 400 from the
  minimum-duration check (per spec).

#### Quality (the first visual verification of the projected TE's vision path): references clearly take effect

Since "the projection matrix was calibrated on text only", vision quality was unknown — but
visually no degradation could be detected: the reference person's face, bangs, hair length,
cardigan color, white ribbon and necklace are all consistently reflected, and the i2va
compositions (profile shots) are nearly identical to the 32B ground truth
`test1_walk_park.mp4`. With an audio reference the lip movement is clear too. The only
blemish is a light compositional drift late in the 768×1344 run (not a breakdown).
Outputs: `outputs/ref2i_1786447720.png` / `ref2va_1786449645.mp4` /
`ref2va_1786452054.mp4` / `ref2va_1786455323.mp4`.

### 2026-08-12: Everything stacked on the 96GB box — t2i 7.65s / t2va 28.56s, and two wasted costs it exposed

With the ballast from the low-VRAM verification removed, the 96GB box (RTX PRO 6000 plus the
second-slot 4060 Ti 16GB) was measured with **every available speedup turned on**, to find
the ceiling.

Configuration: `H3_LOWVRAM=1 H3_KEEP_TRANSFORMER=1 H3_VIDEO_VAE_FP16=1
H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_TE_DEVICE=cuda:1 H3_TURBO_LORA=1` — an int8
transformer resident on GPU0 (with the turbo LoRA applied), the projected TE in NF4 (3.11GB)
resident on GPU1, sage attention (sm_120), fp16 decode. Turbo is toggled per request.

| Setting | t2i steady-state | t2va 5s | Peak |
|---|---|---|---|
| **turbo, 4 steps** (FBC is auto-disabled by turbo) | **7.65s** | **28.56s** | 42.5GB |
| 30 steps + FBC (8 skipped for t2i, 7 for t2va) | 21.41s | 121.5s | 41.8GB |
| 30 steps, no FBC | 27.42s | 155.0s | 42.5GB |
| (reference) 96GB default, 30 steps with FBC — the old baseline | — | about 160s | 92GB |

Breakdown with turbo: t2i = denoise 2.40s + decode 1.07s + about 4.2s of remainder; t2va =
denoise 14.81s + decode 7.20s. **This beats the 48GB box's record** (t2i 9.7s / t2va 44.2s),
and is **5.6x** faster than the old 96GB default (about 160s).

**Note the 42.5GB peak** — the fastest configuration does not come close to filling 96GB. The
speed comes from running with zero fixed costs, not from capacity (the same configuration
fits a 48GB-class card).

#### Waste 1: a co-resident projected TE gets freed on every request

The `H3_LOWVRAM=1` choreography loads the TE, encodes, and frees it on every request. That was
designed around the 32B TE (21GB), but **the projected TE is only 3.11GB and still gets the
same treatment, paying a 7.6s load every time** — with 55GB sitting free. Parking it on the
second GPU with `H3_TE_DEVICE=cuda:1`:

| Where the TE lives | t2i steady-state | t2va 5s |
|---|---|---|
| Co-resident on GPU0 (loaded/freed each request) | 15.23s | 35.69s |
| **Resident on GPU1** (`H3_TE_DEVICE=cuda:1`) | **7.65s** | **28.56s** |

Even with everything else identical, **where the TE lives is a 2x difference for t2i**. Note
that the 96GB box's second card is 16GB, so the 32B TE (21GB in nf4, 17.45GB pruned) would not
fit — **this residency only works because of the projected TE**.

#### Waste 2: a bf16 transformer is freed and reloaded for 12s every request

Since 96GB can hold a bf16 transformer (66.3GB), it should be possible to avoid int8's
dequantization overhead, so plain mode (no `H3_LOWVRAM`) was measured too. **Denoising is
indeed faster in bf16** (t2i 2.07s vs int8's 2.40s; t2va 14.05s vs 14.81s — 5-14% faster). Yet
it loses on the total:

| Transformer | t2i steady-state | t2va 5s | Peak |
|---|---|---|---|
| int8, resident (`H3_KEEP_TRANSFORMER=1`) | **7.65s** | **28.56s** | 42.5GB |
| bf16, plain mode | 19.9s | 40.0s | 68.9-73.1GB |

The log shows plain mode **freeing the transformer for every decode and reloading it in
11.9-12.3s right after**. That free exists because of the "TE-nf4 21GB + transformer 66.3GB +
VAE 11GB = 98.5GB > 96GB" premise — but **with the TE no longer on GPU0, it is 66.3 + 6.1
(fp16 VAE) = 72.4GB**, which fits, so the free is unnecessary in this configuration. Removing
it should make bf16 the fastest option (roughly t2i 8s / t2va 28s at full-precision weights).
`H3_KEEP_TRANSFORMER` currently requires `H3_LOWVRAM=1`, so the right fix is an equivalent
"don't free" decision on the plain-mode side. **Not implemented.**

### 2026-08-12 (part 2): stopping the decode-window free — t2i 7.4s on a single GPU at 45.6GB, and the 48GB-tier goal falls out

"Waste 2" above was fixed on the spot. **The change is one import-time guard condition**:
`H3_KEEP_TRANSFORMER` used to require `H3_LOWVRAM=1`; that was relaxed to "anything but
`group`", so plain mode (`H3_LOWVRAM=0`) can skip the decode-window free too. The branch that
skips the free (`if H3_KEEP_TRANSFORMER: pass`) was already shared, and the restore path's
`_ensure_transformer` is idempotent, so **no additional implementation was needed**. The other
two conditions (the TE is not on GPU0, i.e. `H3_TE_DEVICE` or `H3_TE_PROJ`; and
`H3_VIDEO_VAE_FP16=1`) are exactly what makes plain mode fit as well (66.3 + 11.4 fp16 decode
= 77.7GB).

#### All configurations compared (96GB box, turbo 4 steps, 768²)

| Configuration | t2i steady-state | t2va 5s | Peak | GPUs |
|---|---|---|---|---|
| bf16 + TE on GPU1 + **no free** (B) | **6.89s** | **26.8s** | 74.2GB + 3.2GB | 2 |
| bf16 **single GPU** (TE on GPU0) + no free (A) | 7.08s | 27.04s | 77.3GB | 1 |
| **int8 single GPU + no free (C)** | 7.40s | 28.13s | **45.6GB** | 1 |
| int8 + `H3_LOWVRAM=1` + KEEP + TE on GPU1 (previous best) | 7.65s | 28.56s | 42.5GB | 2 |
| int8 single GPU, **with the free** (C at KEEP=0) | 19.58s | — | 39.8GB | 1 |
| bf16 + TE on GPU1, with the free (previous section) | 19.9s | 40.0s | 68.9GB | 2 |

**Equivalence**: configuration C and the same configuration at `H3_KEEP_TRANSFORMER=0` produce
a PNG with an **identical MD5** (`596a718e4b5cf9a0b907d2ec479225d2`). Skipping the free is
mathematically a no-op, and the same image comes out in **19.58s → 7.40s (2.6x)**.

#### What this establishes

- **Where the TE lives no longer matters.** Single GPU (TE on GPU0) at 7.08s versus two GPUs
  (TE on GPU1) at 6.89s is a 2.7% difference. **The second GPU only mattered while the free was
  still there** — the "2x difference from TE placement" in the previous section was an artifact
  of that free.
- **The free itself was the cost** (2.6x under otherwise identical settings).
- **The practical optimum is configuration C**: int8, single GPU, **45.6GB peak**. Against a
  48GB-tier effective budget of about 49.8GB that leaves roughly 4.2GB of headroom, so **a
  single 48GB card can run within 7% of the fastest configuration**. The backlog item
  "[TE + transformer co-resident on the 48GB box](FEATURES.en.md#unverified-important)" is satisfied in this
  form (not yet confirmed on the physical 48GB box — this is the 96GB box's measured peak
  shown to fit the 48GB budget).
- bf16 denoises faster than int8 (t2i 2.05-2.07s vs 2.39-2.40s) but **needs a 77GB-class
  card**. Whether that beats int8's 45.6GB at a 7% penalty depends on the card you have.

### 2026-08-12 (part 3): reference-mode speed (ref2i / i2va) — a turbo bug, and the fixed costs that remain

The reference modes were measured in the fastest configuration above (int8, single GPU, no
decode-window free, 45.6GB peak). Reference generation uses `transformer_ref` — a separate
model — so both its budget and its bottleneck differ from t2va.

#### First, a bug: the reference endpoints never picked up turbo's step default

Requests were running at `steps=30` despite `turbo=1`. The cause was in `app.py`: **the three
reference endpoints (`/api/ref2va`, `/api/ref2i_batch`, `/api/ref2va_batch`) hardcoded
`num_inference_steps: int = Form(30)`** while t2va/t2i/t2i_batch/fl2va use
`DEFAULT_NUM_INFERENCE_STEPS` (which is 4 under turbo). The turbo LoRA itself was being applied
to `transformer_ref` correctly (the log shows `turbo LoRA lazily applied to transformer_ref
(312 layers wrapped)`), so the combination running was **a distilled LoRA driven for 30 steps** —
a mismatch. Fixed by aligning all three to `DEFAULT_NUM_INFERENCE_STEPS`.

#### Measurements (96GB box, int8, single GPU, 768²)

| Mode | turbo, 4 steps | 30 steps (no turbo) | Denoise (4 / 30) | Peak |
|---|---|---|---|---|
| **ref2i** (reference still, 22 frames) | **79.3s** | 148.4s | 7.8s / 72.1s | 45.4GB |
| **i2va** (image reference to 5s video, 124 frames) | **103.1s** | 290.3s | 22.0s / 209.0s | 45.9GB |

**The peak is 45.9GB, essentially the same as the t2va family's 45.6GB** — the reference modes
also fit a single 48GB-class card. Turbo helps less here (2.8x for i2va) than for t2va (5.5x)
because, as below, **the non-denoise fixed costs dominate**.

#### The backlog item "ref2va × turbo" holds up (verified visually)

The turbo LoRA had only ever been measured on `transformer`, never on `transformer_ref`. **At 4
steps the reference fidelity holds**: bangs, hairstyle, light-green cardigan, white ribbon and
necklace all match, and across the video the subject stays consistent from first to last frame
with the camera tracking and the park setting as prompted, with no breakdown
(`outputs/ref2i_1786509275.png`, `outputs/ref2va_1786509457.mp4`). **Strength 0.094 carries over
to reference-conditioned trajectories unchanged.**

#### The fixed costs that remain (i2va's 103.1s, from log timestamps)

| Phase | Time | Note |
|---|---|---|
| **Reference vision encode** | **about 47s** | **The dominant cost.** The 4B projected TE does not shrink it (though it is an improvement on the 32B era's ~65s/scene) |
| Denoise (4 steps) | 22.0s | |
| Decode + VAE round trips | about 10s | |
| Reference VAE encode | about 6s | |
| **Reloading the t2va transformer at the end** | **about 13s** | **Pure waste for consecutive ref2va requests** |

- **The 47s vision encode is the real target** for reference modes. The batch paths already share
  it across scenes (`H3_REF_PREFIX_CACHE`), but **repeated single requests do not share anything**
  — a cross-request cache would pay off whenever the same reference image is reused.
  → **Implemented on 2026-08-24 as `H3_REF_PREFIX_CACHE_SINGLE` (default 0)**; see the section
  "Reusing the prefix across single requests" above.
- **The 13s reload** comes from restoring the "t2va steady state" when a ref2va request finishes.
  If the next request is also ref2va it is unnecessary, and the same "don't restore" judgment as
  `H3_KEEP_TRANSFORMER` could apply (**not implemented**).

#### Confirming the estimate (same day, via the batch measurements)

The estimate that removing these two costs takes **i2va from 103s to about 45s** is **supported
by the batch measurements**. The batch path is literally the experiment "what happens if the
reference encode is shared", and with everything pinned to 768²:

| | Single | Per item, batched | Measured saving | Model's prediction | Difference |
|---|---|---|---|---|---|
| ref2i (3 scenes) | 79.3s | 47.0s | 32.3s/image | 31.3s/image | **+1.0s** |
| i2va (2 scenes) | 103.1s | 75.0s | 28.1s/video | 23.5s/video | **+4.6s** |

**Stills and video both land on prediction** (slightly better, in fact). The batch's per-step
time matches a single request (i2va 7.321s vs 7.323s), so the difference is purely how many
times the 47s encode is paid. A cross-request cache (47s) plus dropping the reload (13s) gives
**103.1 − 47 − 13 ≈ 43s**, consistent with the evidence as it stands (**still not implemented**).

> **This was briefly withdrawn in error**: the first batch measurement omitted `height`/`width`
> on the batch side only, so it generated at 1344×768 — 1.75x the pixels of 768² — which read as
> "video gets no sharing benefit" and prompted a retraction. Matching the resolution made the
> numbers agree. **Pin the resolution when comparing code paths** — see the boxed note in the
> batching section above.

## Backlog and external events

### Waiting on external events going forward (backlog, as of 2026-08-06)

#### 1. diffusers PR #14355 — **merged (2026-08-05), migration also complete (2026-08-09)**

> **This section is the impact audit written before the migration.** For the actual
> outcome see "Migration to the merged version (f37ab93) — stage 1/stage 2" above (every
> path is equivalent by same-seed MD5, and it is merged into main). It is kept as a
> template for the next time diffusers is bumped.

PR #14355 was **merged at 2026-08-05 17:00Z** (merge commit `f53d552`, the PR's final head
was `f37ab93`). **The venv has been updated to f37ab93.** There is no stable release
including H3 yet (the latest v0.39.0 is from 7/3; H3 arrives starting with v0.40.0), so the
pin stays on a SHA.

**The diff between abc5e9b and f37ab93 spans 27 commits and 70 files**, and as feared,
includes a large-scale refactor via `8ab3662` (review & refactor, #14371). On 2026-08-09,
an impact investigation cross-referencing every diffusers touchpoint in runner.py
(15 imported modules, about 40 symbols) against the merged version's source found the
following:

**Points that break immediately (ImportError, 6 import sites)**:
- `packing.py` / `packing_ref2va.py` were **removed**. The runner imports
  `MINIMAX_H3_TEXT_ENCODER_LAYER` (-> became the `components.text_encoder_layer`
  property, default 50), `MINIMAX_H3_TEXT_TAG` (-> moved to modular_pipeline.py),
  `MINIMAX_H3_KEYFRAME_NOISE_AUG` / `MINIMAX_H3_MIN_DURATION` (-> became the
  `components.keyframe_noise_aug` / `.min_duration` properties — **the "monkeypatch a
  module constant" approach no longer works**), `build_packed_sequence` /
  `build_row_timesteps` / `patchify_video_latents` (-> moved to before_denoise.py),
  `unpatchify_video_tokens` (**no replacement** — needed by the hires path, requiring a
  self-implemented replacement), `build_ref2va_presentation` / `reference_kind` /
  `sample_reference_video_frames` (**no replacement** — redesigned into references.py's
  class hierarchy `MiniMaxH3ImageReference/VideoReference/AudioReference`)
- `MiniMaxH3SetupStep` -> gone (reorganized into `MiniMaxH3ResizeStep`),
  `MiniMaxH3AutoKeyframeVaeEncoderStep` -> reorganized into
  `MiniMaxH3KeyframeVaeEncoderStep` / `MiniMaxH3AutoVaeEncoderStep`,
  `MiniMaxH3TextEncoderStep.encode_prompt` (staticmethod) -> changed to the module
  function `get_qwen3vl_prompt_embeds` (signature also changed),
  `MiniMaxH3Ref2VABlocks` -> **gone from the public API** (merged into a single
  `MiniMaxH3Blocks` + branching inside `MiniMaxH3AutoDenoiseStep` — affects the
  pipe/pipe_ref two-shell design)

**What survived**: the main step classes (`MiniMaxH3SetTimestepsStep` /
`MiniMaxH3LoopDenoiser` / `MiniMaxH3DenoiseStep` / the 2 decode variants / the Ref2VA
denoise/encoder steps) persist under the same names. The `row_timestep_plan` state key
also persists. decoders.py's latents normalization (`latents * latents_std +
latents_mean`, which the H3_TE_DEVICE implementation depends on) also persists in the same
form.

**Additional findings from a diff-level audit (2026-08-09, full-line audit by a Sonnet
agent)**:
- **The numerical path's foundation is unchanged**: scheduling_minimax_h3.py's diff
  (36+/41-) is **docstring formatting only, zero code changes** (confirmed via diff). The
  transformer only had padding-row handling removed (QKV structure/forward signature
  unchanged), and video VAE's `_decode` is byte-identical. -> **the possibility of an
  identical-seed MD5 match remains open**. Regression should check MD5 first, falling back
  to PSNR+visual if it doesn't match
- **The decode contract change ripples through every generation function**: the old
  `MiniMaxH3VideoDecodeStep` received packed sequence rows and unpatchified internally,
  but the new version has the newly-added **`MiniMaxH3AfterDenoiseStep` handle
  unpatchify, with the decode step receiving an already-unpatchified 5D tensor** as its
  contract. All 4 of generate / generate_still_batch / generate_ref2va /
  generate_ref_batch **need an AfterDenoiseStep-equivalent insertion right before decode**
- `build_packed_sequence` / `build_row_timesteps` didn't just move to `@staticmethod`s on
  before_denoise's step classes — **their signatures also changed** (`audio_channels` /
  `audio_tag` / `video_tag` became required arguments) — the hires path's
  `_upscale_block_state_2x` needs a full rewrite
- ref2va is now integrated into `ModularPipeline.from_pretrained(MODEL_ID,
  workflow="ref2va")`'s **workflow mechanism** (changing the premise of the pipe/pipe_ref
  two-shell design. Whether the `transformer_ref` component name persists also needs
  checking). The `_execution_device` resolution algorithm itself (first nn.Module in
  component insertion order) is preserved
- `encode_prompt` disappeared as a staticmethod from both the t2va and Ref2VA sides
  (fl2va is split into `MiniMaxH3FL2VATextEncoderStep`). Since the runner calls it
  directly in order to self-manage `@torch.no_grad()`, the same optimization needs to be
  rebuilt on top of the new module function `get_qwen3vl_prompt_embeds`

**Estimated migration effort: large** (revised upward from the initial "medium-to-large").
It's not just a matter of chasing renames — needed are (a) repointing every import
reference, (b) inserting AfterDenoiseStep before decode x4 functions, (c) rebuilding the
direct encode_prompt call sites x2, (d) converting constant monkeypatches into property
overrides, (e) a full rewrite of the hires path, (f) redesigning ref2va's two-shell design
into the workflow mechanism, (g) reimplementing the KV-prefix cache on top of
references.py. The silver lining is that since the numerical path's foundation is
unchanged, **the likelihood of being able to prove equivalence via per-feature MD5/PSNR
regression after migration is high**.

**Current stance**: keep the current configuration (abc5e9b, all features A/B-tested) as
is for now. If migrating, target the minimal-diff f37ab93 in two stages: (1) t2i/t2va path
-> (2) ref2va path.

Note that the int8 recipe (`TorchAoConfig` + `Int8WeightOnlyConfig`) is already included in
abc5e9b and usable without waiting for the merge (already implemented as
`H3_TRANSFORMER_QUANT=int8`).

#### 2. Waiting for a finished release of Turbo LoRA -> **resolved via the lightx2v variant (production-implemented 2026-08-08, see the "Turbo LoRA" section)**

The `H3_TURBO_LORA=1` wiring is complete (see the section above). The then-current LoRA
(Ostris's) was explicitly labeled "demo/preview, still in training," hence default OFF,
and **unusable with int8/low-VRAM** (fused-QKV assumption -> `Int8Tensor` lacks `aten.cat`).

On 2026-08-08, spiked the alternative candidate
**[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)**
(`minimax_h3_fl2v_turbo_4step_v0.1.safetensors`, 1.38GB, Apache 2.0, DMD distillation;
Kijai/MiniMax-H3_comfy is a ComfyUI repack of this), and **confirmed by measurement that it
works on 48GB (int8) and reaches usable quality at 4 steps**
(`scripts/probe_lightx2v_turbo.py`, main codebase unmodified).

**The decisive difference: the keys are diffusers-native**. Reading the safetensors header
directly confirmed the keys look like
`transformer_blocks.N.attn.to_q.lora_A.default.weight`, with **to_q/to_k/to_v separated**
(rank 128, 312 modules = 50 blocks x6 + token_refiner 2 blocks x6, covering only attn and
ff, not adaln/final_layer), meaning **`fuse_projections()` is unnecessary**. Since
`torch.cat` is never called, it avoids the main cause of int8 incompatibility that blocked
the Ostris variant. In fact, **all 312 modules applied to the int8 transformer in 0.6
seconds with no exceptions**. `ff.net.0.proj`'s `lora_B` has shape `(28672, 128)`, exactly
matching diffusers' SwiGLU (gated, `dim_out*2`), confirming this LoRA was trained directly
against diffusers' own module structure.

**Pitfall (most important): the strength is ~0.094, not Kijai's documented 0.75**. In the
implementation's approach of applying directly to the raw `B*A`, 0.75 is **too strong,
producing fully-noised output even at 30 steps** (demonstrated at 30 steps to rule out a
step-count issue). Since the ComfyUI side applies with alpha folded in, the hypothesis
that `0.75 x (alpha/rank) = 0.75 x 16/128 ~= 0.094` is the corresponding value matched the
measurements.

| strength | 4-step result | audio rms / peak |
|---|---|---|
| 0.75 (Kijai's documented value as-is) | **fully noise** (same even at 30 steps) | 0.083 / 0.43 |
| 0.15 | good (background slightly soft) | 0.069 / 0.79 |
| 0.10 | good | 0.065 / **1.05 (clipping)** |
| **0.094** (= 0.75 x 16/128) | **best** (fur, pine needles all sharp) | 0.039 / 0.70 |

**Speed (RTX PRO 5000 48GB + `H3_LOWVRAM=1`, 768², 5s, seed 12345, same prompt)**: denoise
**197.7s -> 26.1s (7.6x)**, total **351.4s -> 135.2s (2.6x)**. The smaller reduction in
total time is because the low-VRAM mode's load fixed cost (~110s) remains — combining with
the batch path can amortize this too. Temporal consistency also confirmed via 4-frame
sampling (no breakdown).

**Remaining concern**: audio is **about 5x louder** than baseline (baseline rms 0.0073 ->
0.039). Spectral flatness is 0.24, lower than baseline's 0.34, meaning **it is not white
noise** (there is structure), but depending on intensity the peak can exceed 1.0 and clip.
Would need level checking if implemented. Also, since `token_refiner` is on the int8
exclusion list, it ends up in a mixed state — **transformer_blocks are int8-base +
bf16-delta, while token_refiner is bf16-base + bf16-delta** (no observed harm, since both
apply and generate succeed).

**Production-implemented on 2026-08-08** (see the update in the "Turbo LoRA" section):
(1) the diffusers-native apply function + key-format auto-detection, (2) `_TurboLoRALinear`'s
scale factor (per-format default 1.0/0.094), (3) format-based combination guards (comfy x
int8 remains rejected, group rejects regardless of format). (4) Only applicability to
ref2va remains unverified.

#### 3. Unstarted improvement candidates (in priority order, none urgent)

- ~~**Pre-saving quantized checkpoints**~~ -> **implemented for TE** (`H3_TE_PREQUANT`,
  2026-08-08. TE load 53.0s->29.5s, request total -35%). **What remains is transformer
  int8**, which would need about 34GB to save, impractical given this box's disk headroom
  (43GB free). On a machine with more disk headroom, the same technique could shave off
  the transformer's load time (~32.5s) as well
- ~~**Keeping TE resident on a second GPU**~~ -> **implemented on 2026-08-09**
  (`H3_TE_DEVICE`, steady state 78.4s->about 35s = -55%. See the "Keeping text_encoder
  resident on a separate GPU" section above). The following is the record from spike
  time: offloading TE to a separate card lets GPU0 keep the transformer resident, and
  **the fixed cost disappears entirely**. In measurements, the current RTX 4000 SFF Ada
  20GB **works for t2va** (peak 17.76GB / 3.23GB headroom), but **ref2va OOMs** (short by
  1-2GB from running a 2048px reference through the vision tower). A 24GB tier is needed
  to cover ref2va as well. PCIe Gen4 x4 is not an issue (TE is loaded once at startup; the
  only per-request transfer is prompt_embeds at 42MB). Implementation requires careful
  changes around `_execution_device` (this project's single biggest source of pitfalls)
  (`scripts/probe_te_on_second_gpu.py`)
- **`ref2va` × turbo is unverified**: turbo LoRA has only been measured applied to
  `transformer` (the t2va family); applying it to `transformer_ref` has **never been
  tried**. Wiring-wise it should go through the same path, but whether a distilled LoRA
  holds up on a reference-conditioned trajectory is a separate question (the strength
  0.094 validity is also a t2va-only measurement). If speeding up reference-conditioned
  generation becomes desirable, spike this first
- **A 4B-only quantization option for the projected TE (`H3_TE_PROJ`)** → **implemented
  (same day, 2026-08-10)**: added as `H3_TE_PROJ_QUANT` and, after measurement (NF4 3.11GB,
  quality on par), **the default was switched to bnb-4bit** (see "Addendum (same day, part 2)"
  in the dated 2026-08-10 section). What follows is the pre-implementation analysis: the
  initial projected TE loads Qwen3-VL-4B **in bf16, with a real GPU occupancy of 8.88GB** (measured
  2026-08-10; matches the 8.88GB checkpoint). The projection matrix's distributor writes
  "15.7GB → 5.2GB", but **5.2GB does not hold in bf16** (it presumably refers to a
  quantized variant). Quantizing the 4B to int8/nf4 would put it in the 4-5GB class,
  lowering the second-GPU requirement further.
  **Caution**: the current implementation makes `H3_TE_PROJ` and
  `H3_TE_QUANT`/`H3_TE_PRUNE`/`H3_TE_PREQUANT` **mutually exclusive** via an import-time
  guard. That is a measure to "keep 32B-TE settings from leaking onto the 4B", not a ban on
  quantizing the 4B. If implementing, add it as a **separate 4B-only flag** (e.g.
  `H3_TE_PROJ_QUANT`) without breaking the existing guard's intent. Note also that the
  projection matrix must be applied in fp32 (W is 2560×5120 fp32)
- **16GB-tier support** → **achieved (2026-08-11)**: with the projected TE in NF4 (3.11GB)
  co-resident alongside `H3_LOWVRAM=group`, **full functionality was measured on a real
  RTX 4060 Ti 16GB alone** (no TE streaming needed after all). 8GiB×2 also holds up through
  t2va. See the dated 2026-08-11 sections. What follows is the pre-achievement analysis:
  would need streaming execution of TE (flowing it to the GPU block
  by block). The current floor is the pruned TE-nf4's resident 17.45GB (see the section
  above). **With the projected TE this floor drops to 8.88GB**, so the 16GB tier may be
  reachable by a different route (see the recalculation of the second-GPU requirement
  above)
- **torch.compile**: unverified. Compatibility with FBC/group-offload hooks (graph breaks)
  needs checking
- **torchao's C++ kernel**: could eliminate int8 mode's dequant cost (+5s), but requires
  torch>=2.11, carrying a large regression risk for the whole venv (not recommended)



---

# ISSUES / PRs of animede/Diffusers_minimax-h3


### #2 [closed] PR: fix(install): ensure pip exists in venv via uv --seed on ensurepip-less Python 3.12
URL: https://github.com/animede/Diffusers_minimax-h3/pull/2  by lunaris2055 2026-11-27T02:26:47Z

## 概要
Ubuntu 24.04（Google Colab 等）のように `/usr/bin/python3.12` が存在するものの `python3.12-venv`（`ensurepip`）が未導入の環境で `scripts/install.sh` を実行すると、`venv/bin/pip` の無い不完全な `venv/` が作成され、直後の `venv/bin/pip install` で終了コード 127 となる問題を修正します。

## 再現環境・症状
- 環境: Google Colab（Ubuntu 24.04.4 LTS / デフォルト Python 3.13.16 / `/usr/bin/python3.12` あり）
- 症状:
  - `scripts/install.sh: line 46: venv/bin/pip: No such file or directory`
  - また `if [ ! -x venv/bin/python ]` のみを確認していたため、不完全な `venv/` が残った状態で再実行しても復旧しませんでした。

## 変更内容
- `venv/bin/python` に加えて `venv/bin/pip` の存在も確認し、`pip` が欠けている不完全な `venv/` は削除して作り直します。
- `uv` が利用可能な環境（`h3-creator-studio` の `setup.sh` 経由など）では `uv venv --python "$PYTHON" --seed venv` で確実に `pip` 入りの仮想環境を作成します。
- `uv` が無い環境では従来どおり `$PYTHON -m venv venv` を実行し、`venv/bin/pip` が作られなかった場合に原因と対処（`python3.12-venv` または `uv` の導入）を明示して終了します。

## 動作確認
- Google Colab（Ubuntu 24.04 / L4 GPU / CUDA 13.0）にて `uv venv --python 3.12 --seed venv` による仮想環境構築および `requirements.txt` 導入〜推論動作を確認済みです。

> **Comment by animede (2026-11-27T04:46:59Z):**

こちらも対応いたしました。ありがとうございます



### #1 [closed] ISSUE: 24
URL: https://github.com/animede/Diffusers_minimax-h3/issues/1  by wuxiaohui0 2026-08-13T03:36:28Z

Using two 4090 24G cards, can I run a 720p 15s video, approximately up to how many seconds?

> **Comment by animede (2026-11-27T04:58:36Z):**

Thanks for the question, and sorry for the very late reply.

**Short answer:** two RTX 4090s can run a 15-second clip at the model's 720p-class size, but we have not measured that exact combination yet, so I can't give a reliable time.

- **Length:** 5–15 s is supported. The frame count is rounded up to 17n+5 (15 s → 362 frames).
- **Resolution:** the model's native size is a 768 short edge. For 720p-class landscape, use the `1344x768` preset rather than 1280x720 (custom sizes are rounded to multiples of 32).
- **Two GPUs:** the server does not split the video transformer across GPUs. The second card can hold the text encoder (`H3_TE_DEVICE=cuda:1`), so the first card only runs the transformer. For two 24GB cards (T2VA):

```bash
H3_LOWVRAM=group H3_TE_PRUNE=1 H3_TE_DEVICE=cuda:1 H3_VIDEO_VAE_FP16=1 H3_TURBO_LORA=1 \
  H3_ATTN_BACKEND=default venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8611
```

  `H3_TE_PRUNE=1` keeps the 32B text encoder (about 20GB) on the second card and needs its ~67GB download; to avoid that, use `H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3` instead. Setting `H3_TE_DEVICE` switches the server to manual mode, so the other flags are needed too. For Ref2VA, leave out `H3_TE_DEVICE` (as in the 48GB example in the README).
- **Time:** our reference numbers are for 5 s at 768²: about 280 s on the 32GB tier (peak 28.7GB), and 151 s with the 4-step turbo LoRA on an RTX A4000 16GB. 1344x768 has 1.75x the pixels of 768² and 15 s has 3x the frames, and attention cost grows faster than that, so a 15 s 1344x768 clip takes many times longer than a 5 s 768² one. I'd suggest trying 5 s at 1344x768 with turbo first, then 10 s and 15 s.

If you try it, we'd be glad to hear your numbers (resolution, length, steps, time, peak VRAM).

