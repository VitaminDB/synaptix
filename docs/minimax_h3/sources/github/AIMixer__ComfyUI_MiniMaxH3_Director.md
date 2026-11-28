# AIMixer__ComfyUI_MiniMaxH3_Director

Source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
Fetched: 2026-11-28

Description: Multi-segment MiniMax H3 Director for official ComfyUI MiniMax-H3
Stars: 2939  Updated: 2026-11-25T02:15:46Z



---

## FILE: README.md (https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/HEAD/README.md)

# ComfyUI MiniMax H3 Director

基于 **ComfyUI 官方 MiniMax-H3** 的多段音视频导演台插件。仓库地址：[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

**English** → [README_EN.md](README_EN.md)

![MiniMaxH3Director 工作流截图](docs/screenshot.png)

## 功能介绍

**MiniMaxH3Director** 是面向长视频、多段生成的 MiniMax H3 导演台节点，把分段计划、条件编码、采样解码和导出整合在一个节点里。底层走官方 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` + `MiniMaxH3SigmaShift` + `KSampler` + AV 分离解码链路，原生输出立体声音频。

### 核心能力

| 功能 | 说明 |
|------|------|
| **多段时间轴** | 节点内上传视频，支持切分、均分、智能分镜分割（PySceneDetect）、追加；分割点可选中删除；可视化时间轴预览每段范围与缩略图 |
| **多任务模式** | `task_type`：`t2v`（文生视频）、`i2v`（图生视频）、`fl2v`（首尾帧生视频）、`r2v`（参考主体生视频 / 素材组）、`v2v`（视频转视频）、`rv2v`（参考素材改视频） |
| **首尾帧 (fl2v)** | 独立首尾帧时间轴：多组关键帧、「添加一组」可只写提示词（文生）、或上传首帧和/或尾帧（官方支持只传尾帧）；开「段间引导」并勾「引用上段」时，空组用上一段末尾 N 帧做运动/音频衔接；拖缘调时长；支持「选择运行」只跑部分组 |
| **参考素材组 (r2v)** | fl2v 式分组 UI：上方「公共参数」共享参考图/音频与公共提示词（与每组提示词拼接）；每组可再挂图片1–9 / 音频1–3 / 视频1–3；提示词用 `<Picture N>` / `<Video K>` / `<Audio J>`（或 `@` 引用）；时间轴预览与选中状态同步 |
| **源视频编辑 (v2v / rv2v)** | Bernini 风格源视频时间轴；每段源画面自动绑定 `<Video 1>`；`rv2v` 另可挂参考图（图片1–9）与参考音频（音频1–3） |
| **选择运行** | 开启后只采样勾选的片段/素材组；未勾选段可用缓存或源画面填充（全部导出时） |
| **外部多组接线** | `Director Group (Image to Video)` / `(Reference to Video)` + `Groups Combine`；连入导演台 `i2v_groups` / `r2v_groups` 后外部优先覆盖 UI 素材，仍支持跑批与选择运行 |
| **原生立体声音频** | 与画面同次采样生成；`v2v`/`rv2v` 可选生成声音 / 使用原声 / 静音 |
| **段间引导** | 默认关闭；多段 `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v` 时可开启，将上一段生成结果的末尾运动（及生成音频）钉入下一段采样再裁掉前缀。上下文帧数：5 / 22 / 39 / 56，**默认推荐为 22**。**感谢 [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) 提供的实现思路** |
| **语义桥 (Semantic Bridge)** | 外接 **MiniMax H3 Director Semantic Bridge** 到导演台 `selflift` 上方的 `semantic_bridge` 口。未接线 = 完全相同。接线后用 student MLP 改写官方 cond token（RMS-norm → 残差混合）。权重自行放到 `models/semantic_bridge/`，本插件不分发；节点里换 adapter 即可在原版与 BUNNY 之间切换，不要串两座桥。原版偏构图/空间/计数等静态关系；BUNNY 偏动作归属与复杂多人。蒸馏于 FL2VA；`r2v` / `v2v` / `rv2v` 为强制兼容，请谨慎使用。参考 [speach1sdef178/MiniMax-H3-Semantic-Bridge](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge)、[JOKER141/BUNNY_H3_Conditioning_Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge) |
| **渐进一采 (SelfLift)** | 外接 **MiniMax H3 Director SelfLift** 到导演台 `selflift` 口（Refine 上方）。未接线 = 原来的单阶段一采。接线后一采变为低清前缀 + 3D lift + 高清收尾，画布仍是导演台分辨率。Euler。**感谢 [slmonker/selflift-Avatar](https://github.com/slmonker/selflift-Avatar) 提供的实现思路** |
| **二采 / 放大 (Refine)** | 外接 **MiniMax H3 Director Refine** 到导演台 `refine` 口。未接线 = 原来的单次采样。`refine` = 同分辨率精修；`upscale` = 先放大到目标画布再按 SIGMAS 二采（像素插值 / RTX VSR / H3 latent）；`latent_upscale` = 只放大 H3 latent、不二采。`passes` 可多次精修（upscale 只放大一次）。可选接 `refine_model` 换二采 UNET。`images` 为二采后成片，`images_pre_refine` 为一采（放大前）画面 |
| **运行报告** | `report` 口输出分段计划、每段任务摘要 |
| **导演包导入导出** | 工具栏「导入/导出导演包」：zip 内保存时间轴 JSON 与参考图/视频/音频。目录名为英文（`shared_params/`、`asset_groups/01/`、`Picture1`…），与切到 EN 后的界面用语对应，避免路径编码问题 |

参考音频槽可直接选择已有视频，或从本地选择音频/视频；视频会立即提取首条音轨为 FLAC，结果直接保存到 `input/`。本地视频只在临时目录中用于提取，不会作为视频素材保存。音频沿用 ComfyUI 现有上传规则：同名同内容直接复用，同名不同内容自动添加序号且不会覆盖；当前素材组也不会重复添加同一路径。

### 输入 / 输出

**输入：** `model` → `video_vae` → `audio_vae` → `clip`  
**可选：** `i2v_groups`（Image to Video 多组）/ `r2v_groups`（Reference to Video 多组）/ `semantic_bridge`（`MiniMax H3 Director Semantic Bridge`）/ `selflift`（`MiniMax H3 Director SelfLift`）/ `refine`（`MiniMax H3 Director Refine`）

> **缓存文件夹名（`cache_name`，面板「性能」组最下方）**：分段缓存的目录名。
> 填了就落到 `output/minimax_seg_cache/<名字>_<节点id>/`，**留空则只有节点 id**（旧行为不变）。
> 同一个名字配上不同节点 id 会分开。复制工作流后节点 id 通常还是原来的，名字也相同就会继续共用一个目录，要隔开就改名字。
> 名字会自动过滤 `/ \ : * ? " < > |` 与 Windows 保留名（`CON`/`NUL`…），最长 64 字符。

**输出：** `images` → `audio` → `fps` → `frame_count` → `source_images` → `report` → `images_pre_refine`

> CLIP Loader 的 **type 必须选 `minimax`**（Qwen3-VL）。  
> `t2v` / `i2v` / `fl2v` 用 **fl2va** UNET；`r2v` / `v2v` / `rv2v` 用 **ref2va** UNET。

`输出原片到 source_images` 只填充独立的 `source_images` 输出，不会改变主 `images`。请将 `source_images` 另接预览或视频合成节点查看；解码失败时运行报告会明确说明，并输出灰色占位而不会冒充生成画面。

## 导演包（剧本 + 素材）

工具栏右侧 **导入导演包 / 导出导演包**。导出为 `*.mmxpack.zip`。路径只用 ASCII，与英文界面一致（与当前 UI 语言无关）。

| English UI | Pack path |
|------|------|
| Shared params | `shared_params/` |
| Asset group 1 | `asset_groups/01/` |
| Picture 1–9 | `Picture1.png` … `Picture9.webp` |
| Video 1–3 | `Video1.mp4` |
| Audio 1–3 | `Audio1.wav` |
| start / end (fl2v) | `start.jpg` / `end.jpg` in that group folder |
| Upload video (v2v source) | `source_video/` |

```
pack.json
shared_params/shared_params.json
shared_params/Picture1.png
asset_groups/01/group.json
asset_groups/01/Picture4.png
timeline.json
```

- `timeline.json`：导演台导出时写入，用于无损往返（含其它任务草稿等）。
- 转换工具可以只写 `pack.json` + `shared_params/` + `asset_groups/`，不必手写 `timeline.json`。
- 槽位编号与界面相同：公共参数占用 Picture 1–3 时，组文件夹里从 `Picture4` 续编，不要在组内把第一张改名为 `Picture1`。
- 不含 UNET / CLIP / VAE。导入会覆盖当前节点时间轴（有确认）。媒体落到 ComfyUI `input/minimax_director_packs/`。

## 依赖

请将 **ComfyUI** 升级到 **v0.30.0** 及以上（含官方 MiniMax H3 节点：[PR #15224](https://github.com/comfyanonymous/ComfyUI/pull/15224)、[PR #15228](https://github.com/comfyanonymous/ComfyUI/pull/15228)）。

可选：`scenedetect`（智能分割）、`opencv-python-headless`（源视频解码）、`imageio-ffmpeg`（原声抽取）——见 `requirements.txt`。  
Refine 的 `nvidia_rtx_vsr` 另需 NVIDIA GPU，可 `pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com`（不是硬依赖）。

## 安装

### 方法一：手动安装（标准方式）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git

pip install -r ComfyUI_MiniMaxH3_Director/requirements.txt
```

重启 ComfyUI。

### 方法二：ComfyUI Manager

1. 打开 **ComfyUI Manager**
2. 选择 **Install via Git URL**
3. 填入 `https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git` 并安装
4. 重启 ComfyUI

## 模型与工作流下载

完整资源包（**MiniMax H3 模型权重** + **示例 JSON 工作流**）见：

**[Comfyit 搅拌站 · 文章 506：MiniMax H3 模型和工作流](https://comfyit.cn/article/506)**

下载后将 `models/` 合并到 `ComfyUI/models/`，JSON 工作流拖入 ComfyUI 即可。

也可参考：

- **Hugging Face：** [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **ComfyUI 文档：** [MiniMax H3 工作流示例](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3)

本仓库自带示例：`example_workflows/`

| 工作流 | task_type | UNET | 说明 |
|--------|-----------|------|------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | 文生音视频 |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | 首尾帧（「添加一组」） |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | 参考改视频素材组 |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | 源视频时间轴编辑 |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | 源视频 + 参考图/音频 |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | 外部 Group×2 → Combine → `i2v_groups` |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | 外部 Group×N → Combine → `r2v_groups` |
| `minimax_h3_director_二采_加速.json` | r2v | **ref2va** | 外接 Refine 二采（SIGMAS + H3 latent）；`images` 与 `images_pre_refine` 各出一路成片 |

### 推荐模型文件

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (t2v / i2v / fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

## 快速开始

1. 确认 ComfyUI ≥ **0.30.0**，已能加载官方 MiniMax H3 节点
2. 从 [文章 506](https://comfyit.cn/article/506) 或本仓库 `example_workflows/` 加载示例
3. 连接 UNET / CLIP / video_vae / audio_vae，在导演台 UI 内编辑时间轴与提示词后 Queue

**视频教程：** [B 站合集 · 插件使用教程](https://space.bilibili.com/1997403556/lists/8357740)

### 默认采样参数

- 画布默认 **0.4MP 16:9（864×480）**，**5 秒 / 124** 帧 @ **24 fps**（17k+5 网格）
- **25** steps，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

### 首尾帧 fl2v 用法摘要

1. 任务类型选 **「首尾帧生视频 (fl2v)」**
2. 点击「添加一组」：可只写提示词（文生）；或上传首帧和/或尾帧（可只传尾帧；仅首帧=图生）
3. 多组时打开「段间引导」并勾「引用上段」，空组会用上一段末尾 N 帧（上下文帧数，默认 22）引导衔接
4. 在镜卡片或时间轴上调整时长；提示词写中间运动 / 镜头 / 过渡
5. Queue 生成；多组可勾选「选择运行」只跑部分组

### 参考主体 r2v 用法摘要

1. 任务类型选 **「参考主体生视频 (r2v)」**（需 **ref2va** UNET + audio_vae）
2. 点击 **「启用公共参数」** 展开面板（默认折叠/关闭）；上传共用参考图/音频并写公共提示词（如角色锁定 / `subject_definitions`）；启用后会与每组提示词拼接
3. 点击「添加素材组」；每组写分镜提示词，可按需再挂本组独有素材（同槽位覆盖公共素材）
4. 提示词中用 `<Picture N>` / `<Video K>` / `<Audio J>`，或输入 `@`（启用公共参数时可引用公共 + 本组素材）
5. 时间轴可预览各组时长与缩略图；「选择运行」与素材组勾选同步

### 源视频 v2v / rv2v 用法摘要

1. 选 **v2v** 或 **rv2v**，上传源视频并分段（切分 / 均分 / 智能分割）
2. 每段写提示词；系统自动将源片段绑定为 `<Video 1>`
3. `rv2v` 可额外上传参考图 / 参考音频；声音模式可选生成 / 原声 / 静音

### 二采 / 放大 Refine 用法摘要

1. 添加 **MiniMax H3 Director Refine**，把 `refine` 接到导演台 `refine` 口。不接则仍是原来的一采
2. `mode=refine`：同分辨率再采一遍（精修）。`mode=upscale`：先放大到目标画布再二采。`mode=latent_upscale`：只放大 H3 视频 latent，不再二采。分辨率控件在 `upscale` / `latent_upscale` 时显示（可跟随导演台、按比例+百万像素，或自定义宽高）。导演台是一采分辨率，Refine 目标才是放大后的宽高
3. `passes`：精修次数，默认 1、最多 9999。`upscale` 只在第 1 次放大，后面都是同分辨率精修；`latent_upscale` 不二采
4. 可选接 `refine_model`（二采 UNET）；不接则用导演台主模型。适合一采挂 Turbo LoRA、二采卸掉或换另一套
5. 导演台 `images` 是二采后成片；`images_pre_refine` 是一采、放大前的画面，便于对比。`source_images` 仍是时间轴原片，不是一采结果。开启 `confirm_first_pass` 后，首次 Queue 只输出 `images_pre_refine` 并阻断主 `images` 下游保存；再次 Queue 命中缓存并完成二采后，`images` 才输出
6. 二采用 SIGMAS：把 `BasicScheduler` 或 `ManualSigmas` 接到 Refine 的 `sigmas` 口
7. fl2v 默认跳过二采（保护钉死的首尾帧）；关掉 Refine 上的 `skip_fl2v` 才会采
8. `upscale` 默认 `h3_latent`：在 Refine 节点里选 3D 权重（`upscale_method` 下方下拉框；`mode=latent_upscale` 时同样出现）。权重放 `ComfyUI/models/latent_upscale_models/`。`lanczos` 可另接 `upscale_model`（RealESRGAN 等），不接则纯插值；也可改 `nvidia_rtx_vsr`
9. 「分段导出」且 `passes>1` 时，每轮会另落 `seg_XXXX_pN.mp4`；「全部导出」只出一采和终稿

示例：`example_workflows/minimax_h3_director_二采_加速.json`

### 外部多组接线（第三方节点接入）

对齐官网两个 conditioning 节点，把扩写 / 抠图 / Load Video 等处理结果以**多组**形式送进导演台：

1. 添加 **`MiniMax H3 Director Group (Image to Video)`** 或 **`(Reference to Video)`**
2. 按组接线：`prompt` / `duration_sec`；I2V 系接 `first_frame` / `last_frame`（无帧=t2v，仅首=i2v，仅尾或首+尾=fl2v）；R2V 为 Autogrow（同官方 Reference to Video）：接图/视频/音频会自动多出空口（图≤9、视频≤3、音频≤3）。输出宽高在**导演台**统一设置
3. 多组：用 **`Director Groups Combine`**（Autogrow：接满最后一个口会自动多出新口，同官方 Reference to Video）→ 导演台 `i2v_groups` / `r2v_groups`；单组可直接把 `group` 连到导演台
4. 导演台 `task_type` 与口一致（t2v/i2v/fl2v ↔ `i2v_groups`；r2v ↔ `r2v_groups`）；**不要两口同时连接**
5. 连接后执行以图中接线为准（外部优先）；UI 卡片变淡，仍可用「选择运行」按组序勾选

## 配套生态 · [Comfyit 搅拌站](https://comfyit.cn/)

[Comfyit](https://comfyit.cn/) 提供环境、模型、工作流与教程配套：

| 栏目 | 链接 |
|------|------|
| 模型 / 工作流包 | [comfyit.cn/article/506](https://comfyit.cn/article/506) |
| 官方 MiniMax H3 文档 | [docs.comfy.org · MiniMax H3](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3) |
| 插件视频教程 | [B 站合集](https://space.bilibili.com/1997403556/lists/8357740) |
| 产品中心 | [comfyit.cn/products](https://comfyit.cn/products) |
| 插件广场 | [comfyit.cn/plugins](https://comfyit.cn/plugins) |
| 模型广场 | [comfyit.cn/resources/models](https://comfyit.cn/resources/models) |
| 工作流广场 | [comfyit.cn/workflows](https://comfyit.cn/workflows) |

## 作者与交流

| | |
|---|---|
| **维护者** | [AI搅拌手 / AIMixer](https://github.com/AIMixer) |
| **本仓库** | [github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) |
| **姊妹插件** | [ComfyUI_Bernini_Director](https://github.com/AIMixer/ComfyUI_Bernini_Director) |
| **作者 QQ** | **3697688140** |
| **B 站** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **插件教程** | [B 站合集 · 使用教程](https://space.bilibili.com/1997403556/lists/8357740) |
| **QQ 交流群** | **551482703** · **425064221** · **559826331** |
| **Comfyit 搅拌站** | [comfyit.cn](https://comfyit.cn/) |

## 致谢

- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — 官方 MiniMax H3 支持
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 模型
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — 权重与文档
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — 段间运动/音频续拍思路参考
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) — H3 3D latent 放大架构与权重格式参考
- [slmonker/selflift-Avatar](https://github.com/slmonker/selflift-Avatar) — SelfLift 渐进一采思路参考
- [speach1sdef178/MiniMax-H3-Semantic-Bridge](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge) — Semantic Bridge student 公式与适配器格式参考
- [JOKER141/BUNNY_H3_Conditioning_Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge) — 同架构的动作逻辑 / 多人场景适配器参考

## 许可证

Apache-2.0


---

## FILE: README_EN.md (https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/HEAD/README_EN.md)

# ComfyUI MiniMax H3 Director

Multi-segment AV timeline director for **official ComfyUI MiniMax-H3**.  
Repository: [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)

**中文文档** → [README.md](README.md)

![MiniMaxH3Director workflow screenshot](docs/screenshot.png)

## Features

**MiniMaxH3Director** is a single-node director for long-form, multi-segment MiniMax H3 audio–video generation — timeline planning, conditioning, sampling, AV decode, and export in one place. It wraps the official `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` + `MiniMaxH3SigmaShift` + `KSampler` pipeline with native stereo audio.

### Core capabilities

| Feature | Description |
|---------|-------------|
| **Multi-segment timeline** | Upload video in-node; split, equal-split, smart shot-split (PySceneDetect), append; selectable/deletable split points; visual timeline with thumbs |
| **Task modes** | `t2v`, `i2v`, `fl2v` (first/last frame), `r2v` (reference material groups), `v2v` (video-to-video), `rv2v` (reference-guided source edit) |
| **First/last frame (fl2v)** | Dedicated shot groups: prompt-only (text-to-video), or start and/or end (official FL2VA allows end-only). With segment continuity + From prev, an empty shot pins the previous tail (N context frames) for motion/audio handoff; drag edges for duration; run-select per group |
| **Reference groups (r2v)** | fl2v-style groups: top **Common params** share refs/audio and a common prompt (concatenated with each group prompt); each group may add images 1–9 / audios 1–3 / videos 1–3; prompt tags `<Picture N>` / `<Video K>` / `<Audio J>` (or `@` picker); timeline preview synced with card selection |
| **Source-video edit (v2v / rv2v)** | Bernini-style source timeline; each segment bound as `<Video 1>`; `rv2v` adds optional refs (images 1–9, audios 1–3) |
| **Run select** | Sample only checked segments/groups; unselected may use cache or source passthrough when exporting all |
| **External multi-group inputs** | `Director Group (Image to Video)` / `(Reference to Video)` + `Groups Combine`; wire into `i2v_groups` / `r2v_groups` for external-priority batches with run-select |
| **Native stereo audio** | Generated with the picture; `v2v`/`rv2v` can generate / keep source / mute |
| **Segment continuity** | Off by default. For multi-segment `t2v` / `i2v` / `fl2v` / `r2v` / `v2v` / `rv2v`, pin the previous generated tail (motion + generated audio) into the next sample, then trim the prefix. Context frames: 5 / 22 / 39 / 56 — **recommended default: 22**. **Thanks to [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) for the implementation approach** |
| **Semantic Bridge** | Wire **MiniMax H3 Director Semantic Bridge** into Director `semantic_bridge` (above SelfLift). Unconnected = identical. Connected = rewrite official cond tokens with the student MLP (RMS-norm → residual mix). Put weights in `models/semantic_bridge/` yourself — this plugin does not ship them. Switch adapters in the node to pick original vs BUNNY; do not chain two bridges. Original leans static relations (layout / space / counting); BUNNY leans action ownership and multi-character scenes. Distilled on FL2VA; `r2v` / `v2v` / `rv2v` is forced-compat — use with care. See [speach1sdef178/MiniMax-H3-Semantic-Bridge](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge) and [JOKER141/BUNNY_H3_Conditioning_Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge) |
| **SelfLift (progressive first pass)** | Wire **MiniMax H3 Director SelfLift** into Director `selflift` (above Refine). Unconnected = original single-stage first pass. Connected = low-res prefix + 3D lift + high-res tail on the Director canvas. Euler only. **Thanks to [slmonker/selflift-Avatar](https://github.com/slmonker/selflift-Avatar) for the implementation approach** |
| **Refine / upscale** | Wire **MiniMax H3 Director Refine** into Director `refine`. Unconnected = original single-pass sampling. `refine` = same-resolution second sample; `upscale` = enlarge to a target canvas then SIGMAS sample (pixel / RTX VSR / H3 latent); `latent_upscale` = H3 latent enlarge only, no second sample. `passes` repeats refine (upscale once). Optional `refine_model` swaps the second-pass UNET. `images` is the refined clip; `images_pre_refine` is the first pass (before upscale) |
| **Run report** | `report` output with plan and per-segment summary |
| **Director pack I/O** | Toolbar **Import pack / Export pack**: zip of timeline JSON plus reference images/videos/audio. ASCII folders (`shared_params/`, `asset_groups/01/`, `Picture1`…) match the English UI and avoid path-encoding issues |

Reference-audio slots can select an existing video or a local audio/video file. A video's first audio stream is extracted immediately to FLAC directly under `input/`; local source videos remain temporary and are not saved as video assets. Audio follows ComfyUI's existing upload rule: identical content with the same name is reused, while different content with the same name gets a numeric suffix without overwriting; the same resolved audio path is not added twice within one material group.

### Inputs / outputs

**Inputs:** `model` → `video_vae` → `audio_vae` → `clip`  
**Optional:** `i2v_groups` (Image to Video packs) / `r2v_groups` (Reference to Video packs) / `semantic_bridge` (`MiniMax H3 Director Semantic Bridge`) / `selflift` (`MiniMax H3 Director SelfLift`) / `refine` (`MiniMax H3 Director Refine`)

**Outputs:** `images` → `audio` → `fps` → `frame_count` → `source_images` → `report` → `images_pre_refine`

> CLIP Loader **type must be `minimax`** (Qwen3-VL).  
> Use **fl2va** UNET for `t2v` / `i2v` / `fl2v`; **ref2va** for `r2v` / `v2v` / `rv2v`.

`Export source to source_images` populates only the separate `source_images` output; it does not change `images`. Connect `source_images` to a preview or video compositor. Decode failures are reported explicitly and emit a neutral placeholder instead of generated frames.

## Director pack (script + media)

Toolbar **Import pack / Export pack** writes `*.mmxpack.zip`. Paths are ASCII only and match the English UI (independent of the current UI language).

| English UI | Pack path |
|------|------|
| Shared params | `shared_params/` |
| Asset group 1 | `asset_groups/01/` |
| Picture 1–9 | `Picture1.png` … `Picture9.webp` |
| Video 1–3 | `Video1.mp4` |
| Audio 1–3 | `Audio1.wav` |
| start / end (fl2v) | `start.jpg` / `end.jpg` in that group folder |
| Upload video (v2v source) | `source_video/` |

```
pack.json
shared_params/shared_params.json
shared_params/Picture1.png
asset_groups/01/group.json
asset_groups/01/Picture4.png
timeline.json
```

- `timeline.json` is written on Director export for lossless round-trip (including other-task drafts).
- A converter may write only `pack.json` + `shared_params/` + `asset_groups/` and omit `timeline.json`.
- Slot numbers match the UI: if Shared params occupy Picture 1–3, group folders continue from `Picture4` — do not rename the group’s first image to `Picture1`.
- Models (UNET / CLIP / VAE) are not included. Import replaces the current node timeline (with confirmation). Media is copied to ComfyUI `input/minimax_director_packs/`.

## Requirements

**ComfyUI ≥ v0.30.0** with official MiniMax H3 nodes ([PR #15224](https://github.com/comfyanonymous/ComfyUI/pull/15224), [PR #15228](https://github.com/comfyanonymous/ComfyUI/pull/15228)).

Optional: `scenedetect`, `opencv-python-headless`, `imageio-ffmpeg` — see `requirements.txt`.  
Refine `nvidia_rtx_vsr` needs an NVIDIA GPU: `pip install nvidia-vfx --extra-index-url https://pypi.nvidia.com` (not a hard dependency).

## Installation

### Method 1: Manual (standard)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git

pip install -r ComfyUI_MiniMaxH3_Director/requirements.txt
```

Restart ComfyUI.

### Method 2: ComfyUI Manager

1. Open **ComfyUI Manager**
2. Choose **Install via Git URL**
3. Enter `https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git` and install
4. Restart ComfyUI

## Models & workflow downloads

Full pack (**MiniMax H3 weights** + **example JSON workflows**):

**[Comfyit · article 506 — MiniMax H3 models & workflows](https://comfyit.cn/article/506)**

Merge `models/` into `ComfyUI/models/`, then drag a JSON workflow into ComfyUI.

Also available:

- **Hugging Face:** [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- **ComfyUI docs:** [MiniMax H3 workflows](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

This repo ships examples under `example_workflows/`:

| Workflow | task_type | UNET | Notes |
|----------|-----------|------|--------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | Text to AV |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | First/last frame groups |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | Reference material groups |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | Source-video timeline edit |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | Source + reference images/audio |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | External Group×2 → Combine → `i2v_groups` |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | External Group×N → Combine → `r2v_groups` |
| `minimax_h3_director_二采_加速.json` | r2v | **ref2va** | Refine second sample (SIGMAS + H3 latent); `images` and `images_pre_refine` each save a clip |

### Recommended model files

| Role | Filename | Directory |
|------|----------|-----------|
| UNET (t2v / i2v / fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

## Quick start

1. Ensure ComfyUI ≥ **0.30.0** with MiniMax H3 nodes
2. Load an example from [article 506](https://comfyit.cn/article/506) or `example_workflows/`
3. Connect UNET / CLIP / video_vae / audio_vae, edit the timeline UI, Queue

**Video tutorial:** [Bilibili playlist · plugin usage](https://space.bilibili.com/1997403556/lists/8357740)

### Default sampling

- Canvas default **0.4MP 16:9 (864×480)**, **5s / 124** frames @ **24 fps** (17k+5 grid)
- **25** steps, `res_multistep` + `simple`, CFG **1.0**
- Sigma shift: video **12** / audio **3**

### First/last frame (fl2v) — short guide

1. Set task type to **First/Last Frame to Video (fl2v)**
2. Click **Add group**: prompt-only (text-to-video), or upload start and/or end (end-only OK; start-only = i2v)
3. With multiple groups, turn on **Segment continuity** and check **From prev** — an empty shot pins the previous tail (N context frames, default 22)
4. Adjust duration on the shot card or timeline; write mid-shot motion / camera / transition in the prompt
5. Queue; with multiple groups, use **Run select** to sample only some of them

### Reference groups (r2v) — short guide

1. Set task type to **Reference to Video (r2v)** (**ref2va** UNET + audio_vae)
2. Click **Enable common params** (collapsed/off by default); upload shared refs/audio and write a common prompt (e.g. character lock / `subject_definitions`); when enabled it is concatenated with each group prompt
3. Click **Add material group**; write per-shot prompts and optionally add group-only assets (same slot overrides common)
4. In prompts use `<Picture N>` / `<Video K>` / `<Audio J>`, or type `@` (with common params on, picker includes common + group assets)
5. Timeline previews group duration/thumbs; Run-select stays in sync with group checkboxes

### Source video (v2v / rv2v) — short guide

1. Choose **v2v** or **rv2v**, upload a source video and split segments (cut / equal-split / smart split)
2. Write a prompt per segment; the source clip is bound as `<Video 1>` automatically
3. For `rv2v`, optionally add reference images / audio; audio mode can be generate / source / mute

### Refine / upscale — short guide

1. Add **MiniMax H3 Director Refine** and wire `refine` into Director `refine`. Leave it unconnected for the original single pass
2. `mode=refine`: same-resolution second sample. `mode=upscale`: enlarge to a target canvas then second-sample. `mode=latent_upscale`: enlarge H3 video latent only (no second sample). Resolution widgets appear for `upscale` / `latent_upscale` (follow Director, aspect + megapixels, or custom W×H). Director canvas is the first-pass size; Refine target is the enlarge size
3. `passes`: refine rounds, default 1, max 9999. In `upscale` mode only the first round enlarges; later rounds stay on that canvas. `latent_upscale` does not sample
4. Optional `refine_model` (second-pass UNET); unwired uses the Director model. Typical: Turbo LoRA on pass 1, a clean / other LoRA UNET on refine
5. Director `images` is the refined clip; `images_pre_refine` is the first pass before upscale (for A/B). `source_images` is still the timeline source, not the first-pass generate. With `confirm_first_pass`, the first queue exposes only `images_pre_refine` and blocks downstream saving from `images`; the next queue outputs `images` after refining the cached first pass
6. Second sample uses SIGMAS: wire `BasicScheduler` or `ManualSigmas` into Refine `sigmas`
7. fl2v skips refine by default (protects pinned first/last frames); turn off `skip_fl2v` on Refine to include those shots
8. Upscale default is `h3_latent`: pick the 3D weights in Refine (dropdown under `upscale_method`; also shown for `mode=latent_upscale`). Put the file in `ComfyUI/models/latent_upscale_models/`. `lanczos` can take optional `upscale_model` (RealESRGAN etc.); or use `nvidia_rtx_vsr`
9. Segment export with `passes>1` also writes `seg_XXXX_pN.mp4` per round; export-all still only keeps first-pass and the final clip

Example: `example_workflows/minimax_h3_director_二采_加速.json`

### External multi-group wiring

Mirror the two official conditioning nodes and feed **multi-group** batches into the Director:

1. Add **`MiniMax H3 Director Group (Image to Video)`** or **`(Reference to Video)`**
2. Wire per group: `prompt` / `duration_sec`; I2V family uses `first_frame` / `last_frame` (none=t2v, first only=i2v, last only or both=fl2v); R2V uses Autogrow slots (same as official Reference to Video: images ≤9, videos ≤3, audios ≤3). Output size is set on the **Director**
3. Batch with **`Director Groups Combine`** (Autogrow slots, same UX as official Reference to Video) → Director `i2v_groups` / `r2v_groups`; a single `group` can connect to the Director directly
4. Match `task_type` to the port (t2v/i2v/fl2v ↔ `i2v_groups`; r2v ↔ `r2v_groups`); do not connect both ports at once
5. When linked, graph wiring overrides UI cards (external priority); Run-select still applies by group index

## Ecosystem · [Comfyit](https://comfyit.cn/)

[Comfyit](https://comfyit.cn/) provides environment, models, workflows, and tutorials:

| Resource | Link |
|----------|------|
| Models & workflows pack | [comfyit.cn/article/506](https://comfyit.cn/article/506) |
| Official MiniMax H3 docs | [docs.comfy.org · MiniMax H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) |
| Plugin video tutorials | [Bilibili playlist](https://space.bilibili.com/1997403556/lists/8357740) |
| Product center | [comfyit.cn/products](https://comfyit.cn/products) |
| Plugins | [comfyit.cn/plugins](https://comfyit.cn/plugins) |
| Models | [comfyit.cn/resources/models](https://comfyit.cn/resources/models) |
| Workflows | [comfyit.cn/workflows](https://comfyit.cn/workflows) |

## Contact

| | |
|---|---|
| **Maintainer** | [AIMixer](https://github.com/AIMixer) |
| **Repository** | [github.com/AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) |
| **Sibling plugin** | [ComfyUI_Bernini_Director](https://github.com/AIMixer/ComfyUI_Bernini_Director) |
| **Author QQ** | **3697688140** |
| **Bilibili** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **Plugin tutorials** | [Bilibili playlist · usage](https://space.bilibili.com/1997403556/lists/8357740) |
| **QQ groups** | **551482703** · **425064221** · **559826331** |
| **Comfyit** | [comfyit.cn](https://comfyit.cn/) |

## Credits

- [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) — official MiniMax H3 support
- [MiniMax-AI](https://github.com/MiniMax-AI) — MiniMax H3 model
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) — weights & docs
- [NikoDemon80/ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) — inspiration for cross-segment motion/audio continuation
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) — H3 3D latent upscaler architecture and checkpoint format
- [slmonker/selflift-Avatar](https://github.com/slmonker/selflift-Avatar) — inspiration for SelfLift progressive first-pass sampling
- [speach1sdef178/MiniMax-H3-Semantic-Bridge](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge) — Semantic Bridge student formula and adapter format
- [JOKER141/BUNNY_H3_Conditioning_Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge) — same-architecture action-logic / multi-character adapter

## License

Apache-2.0


---

## FILE: README_ZH.md (https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/HEAD/README_ZH.md)

# 中文文档已移至默认 README

本仓库默认文档为中文，请直接查看：

→ **[README.md](README.md)**

English version: [README_EN.md](README_EN.md)


---

## FILE: docs/r2v-source-audio.md (https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/HEAD/docs/r2v-source-audio.md)

# r2v 参考音频原声输出

r2v 的「用原声」原先被后端静默回退为 `generate`，前端也隐藏声音下拉，输出永远是模型生成的声音。

## 根因

| 层 | 问题 |
|---|---|
| 后端 | `VIDEO_EDIT_AUDIO_TASKS` 只认 `v2v` / `rv2v`，`resolve_audio_mode()` 把 r2v 的 `source` 强制改成 `generate` |
| 前端 | 声音下拉只对 v2v / rv2v 显示 |
| 抽取 | r2v 没有源视频时间轴，即便开了 `source` 也抽不到音 |
| 格式 | 参考音频采样率不一致时，mux 后按错误采样率播放（16kHz 当 44.1kHz → 加速约 2.75 倍） |

## 修复

- r2v 加入 source 白名单，前端显示声音下拉
- `source` 且抽不到源视频音时，用该段第一条可用参考音频 mux
- mux 前归一化到 44.1kHz 立体声（失败打 warning，按原始格式 mux）
- 全部导出：每段用各自参考音频按时间轴拼接；该段没挂参考音频则为静音（不回退其它段）

## 双通道

| 模式 | 输出音频 | 参考音频还干什么 |
|---|---|---|
| `generate`（默认，未改） | 模型生成（AV latent 解码） | 条件，驱动口型 |
| `source` | 参考音频 wav 原样 mux | 仍作条件，画面贴原声 |

```
ref_audios → MiniMaxH3ReferenceToVideo._encode_ref_audio
           → AudioVAE(32kHz) → AV latent → 驱动画面口型/节律
```

## 使用

- **用原声**：输出 = 该段参考音频；prompt 的 audio 标签写 `fully_copy`；台词须与音频一致，否则口型错位
- **生成声音**：输出 = 模型生成（可改词）；prompt 写 `reference`。此路径与合入前一致
- 参考音频格式随意（自动归一化）；每段挂各自的。没挂的段在 source 下为静音


---

## FILE: example_workflows/README.md (https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/blob/HEAD/example_workflows/README.md)

# MiniMax H3 Director — 示例工作流

拖入 ComfyUI 画布即可使用。需已安装本插件，且 ComfyUI 主干含 MiniMax H3（v0.30.0+）。

| 文件 | 任务 | UNET | 说明 |
|------|------|------|------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | 文生音视频，可直接 Queue |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | 首尾帧；「添加一组」后上传首帧和/或尾帧（可只传尾帧） |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | 参考改视频；素材组：图片1–9 / 音频1–3 / 视频1–3 |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | 源视频编辑；导演台上传视频并分段（同 Bernini v2v） |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | 参考改视频；源视频 + 图片1–9 |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | 外部 Group（Image to Video）→ Combine → Director.`i2v_groups`；时长/素材以接线为准 |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | 外部 Group（Reference to Video）→ Combine → Director.`r2v_groups`；可用「选择运行」勾选组序 |
| `minimax_h3_director_二采_加速.json` | r2v | **ref2va** | 外接 **MiniMax H3 Director Refine** → Director.`refine`（SIGMAS + H3 latent）。`images` 为二采后成片，`images_pre_refine` 为一采对比片 |

## 模型路径（与官方模板一致）

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (t2v/i2v/fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

CLIP Loader 的 **type 必须选 `minimax`**。

## 默认采样参数

- 画布默认 **0.4MP 16:9（864×480）**，**5 秒 / 124** 帧 @ **24 fps**（17k+5 网格）
- **25** steps，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

## 输出

导演台 → `CreateVideo` → `SaveVideo`（前缀 `video/MiniMaxH3_Director_*`），报告接 `PreviewAny`。

Refine 示例另把 `images_pre_refine` 接到第二路 `CreateVideo` / `SaveVideo`，方便和二采后成片对比。未接 Refine 时该口与 `images` 相同。

## Refine 二采

- 二采一律按 SIGMAS：把 `BasicScheduler` 或 `ManualSigmas` 接到 Refine 的 `sigmas` 口。`BasicScheduler` 请接和二采相同的 MODEL
- 不接 Refine 节点 = 原来的单次采样
- `mode=refine`：同分辨率精修；`mode=upscale`：放大到目标画布再二采；`mode=latent_upscale`：只放大 H3 latent
- 导演台分辨率是一采；Refine 画布（比例+百万像素 / 自定义）是放大目标
- `passes`：精修次数（默认 1）；`upscale` 只在第 1 次放大；`latent_upscale` 不二采
- 可选接 `refine_model`（二采 UNET）；不接则用导演台主模型
- `upscale` 默认 `h3_latent`：在 Refine 节点 `upscale_method` 下方下拉选 3D 权重（`mode=latent_upscale` 时同样出现）。权重放 `ComfyUI/models/latent_upscale_models/`。`lanczos` 可另接 `upscale_model`（RealESRGAN 等），不接则纯插值；也可改 `nvidia_rtx_vsr`
- fl2v 默认跳过二采；关掉 `skip_fl2v` 才会采首尾帧镜头


---

# ISSUES / PRs of AIMixer/ComfyUI_MiniMaxH3_Director


### #294 [open] ISSUE: 导演大师 Mac 版本无法连接本地 LAN 的 Windows ComfyUI
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/294  by Pochenghorng 2026-11-27T17:52:39Z

找不到导演大师的问题反映区，不知道可不可以写在这里？这个问题在之前的版本不存在，就是我从 Mac 版本 的导演大师连接到区域网的另一台 Widnows 5090 PC 做 Comfy 渲染，现在就会一直报错说连不上。所有的检查都做过了都没问题， Mac浏览器直连 Widnows 端IP（192.168.0.15:8188） 也都可以。我在 Mac 的 terminal 写个 python 把远端 IP 识别成本地的就能过。我感觉可能是 V0.8.1 的 Bug，大佬麻烦看一下


### #293 [open] ISSUE: VSR放大前未清理缓存和显存导致硬盘疯狂读取和写入
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/293  by JiuChenTian 2026-11-27T05:58:49Z

本次导演台更新的RlayVSR，在放大时，会因为没设置VSR前清理缓存和显存，导致视频模型或者缓存没卸载，内存直接用满（96g），同时进一步导致磁盘开始是疯狂的读取和写入（很伤硬盘），单纯的VSR缩放是不会产生那么大的内存和显存占用的


### #292 [closed] PR: Fix muted audio export crashing CreateVideo
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/292  by liaowentuan 2026-11-26T04:03:30Z

## 问题

导演台 `audioMode = mute` 时，`build_director_audio_outputs` 返回的 AUDIO 是 waveform 长度为 0 的字典（`empty_audio_dict`）。`CreateVideo` 仍然把它当成一条真实音轨，PyAV 的 AAC 编码器在对空音频重采样时抛 `MemoryError: [Errno 12]`，工作流在 SaveVideo 前失败。

最新 main（3cbc048）仍可复现。

## 复现

1. Director -> CreateVideo -> SaveVideo，`timeline.output.audioMode = "mute"`。
2. 任意分辨率/帧数都可复现（测试用 384x640、73 帧 @24fps）。
3. 报错：`av.error.MemoryError: [Errno 12]`，栈在 `comfy_api/latest/_input_impl/video_types.py` 的音频重采样。

## 修改

- 静音分支改为按每个输出 batch 的帧数生成等长静音 PCM，复用已有的 `_pad_or_trim_audio_to_frames(None, ...)`。
- AUDIO 输出仍是 `list[dict]`，`OUTPUT_IS_LIST` 和下游接口不变，只是不再输出 0 长度的音轨。

## 验证

- 3 / 5 / 8 / 12 / 15 秒（73 / 124 / 192 / 294 / 362 帧），每条时长 1 条文戏 + 2 条武戏，共 15 次。
- 384x640 生成 -> RelayVSR x2 -> 768x1280 导出，全部成功，无 OOM，无 `ave resample` 报错。



### #291 [open] ISSUE: 分镜场景错乱
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/291  by sunjunqing2008 2026-11-26T00:30:32Z

今天遇到一个奇葩的问题，我用选择运行选择了5个分镜，结果前面3个分镜场景和人物是错误的，使用的都是第一个的场景与人物参考。反而台词语言是对的。


### #290 [open] ISSUE: 【提议】增加对PI-H3的支持
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/290  by Lazy-Rabbit-2001 2026-11-22T03:30:58Z

一个DWD加速lora，同时取消了vae解码
https://huggingface.co/Lakonik/pi-H3/tree/main
希望大佬可以做一下这个的支持，因为目前导演台默认需要vae解码，这个用的是它们特制的latent转图像，可能会与现有的导演台内置模式冲突

> **Comment by REHomeGOD (2026-11-27T14:16:34Z):**

我也需要，這樣導演台可以更快!!!!!



### #289 [open] PR: feat: 为 MiniMax H3 Director 新增可选的全局音轨
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/289  by AI-wuji 2026-11-22T03:02:16Z

这个补丁为原版 `MiniMaxH3Director` 新增一条可选的全局音轨，解决制作 MV 时需要逐镜拆歌、拼接后容易出现声音错位或断点的问题。不改插件名、节点 ID、输入输出口或现有工作流的接线。

### 功能
- 在 `r2v` / 混合分镜（24 fps）下，可上传完整音频。分镜轨下方显示共用时间尺的峰值/RMS 波形，支持定位试听，红色播放头随声音移动。可开启起音卡点提示，但不会自动切分或移动镜头。
- 按每个 `r2v` 镜头在时间线上的位置和时长，从全局音轨取对应片段作为口型参考，不用提前拆歌。
- 整片导出直接使用连续的原始音轨，不拼接模型生成的音频；画面长于音频时，超出部分补静音。导演包可携带原音频。

### 兼容与限制
- 不启用全局音轨时，原有流程不变。启用时使用导演台内分镜，不同时接外部素材组，并停用会改变镜头净时长的段间引导。口型效果仍取决于 H3 模型和素材。

### 验证
- 10 项音频测试通过，覆盖连续取样、按绝对时间切片、MP4 单音轨、波形与起音检测、打包及缓存回归。
- 在隔离 ComfyUI 中验证了上传、保存后重开、试听、播放头跟随和暂停。未运行 H3 GPU 生成测试。


### #288 [open] ISSUE: 安全提示：导演大师客户端把解密后的技能正文以明文写入缓存目录
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/288  by ChinaYouShen 2026-11-21T09:59:26Z

这条不是 ComfyUI 节点本身的问题，是「导演大师」桌面客户端的技能加载环节。这里不适合发的话，麻烦转给客户端团队，谢谢。

**现象**

技能文件在磁盘上是加密的，客户端在 Agent 调用技能时先解密。但解密后的正文没有只留在内存，而是被写进了缓存目录：调用结束后，明文仍以 md / txt / json 形式留在磁盘上，用任意文件管理器、备份工具或清理工具都能直接打开。

**复现**

1. 打开导演大师，选任一加密技能（技能广场或自建技能都算）
2. 让 Agent 实际调用该技能跑一次
3. 打开客户端缓存目录（设置里能看到路径）
4. 在最近修改的文件里能直接找到技能正文，内容与原文一致，未加密

**影响**

- 加密只保护了分发环节，落盘后等于没有保护
- 同机其他账户、云同步、备份、清理工具都可能把正文带出去
- 付费或授权技能可被直接复制

**建议**

1. 解密结果只驻留内存，用完释放；不要走「解密→写文件→再读」的路径
2. 确实需要缓存，就缓存密文或哈希索引，不缓存正文
3. 必须落临时文件时：系统临时目录、权限 0600、用后立即删除，并处理异常退出时的残留
4. 日志、报错和遥测里不要打印技能正文

**版本**：请回复一下客户端版本号，我这边可以按版本复测，确认修复后关掉这条。



### #287 [closed] PR: 修复小显存卡采样后解码前的显存空转（只卸 DiT）
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/287  by TechQ36 2026-11-19T07:20:06Z

采样结束到视频 VAE 解码之间没有显存交接点：AV latent 已算完，DiT 仍驻留显存，解码时与 VAE 解码张量抢空间，动态显存反复换页，GPU 长时间低占用。12GB 卡实测同参数解码 418s，DiT 释放后单独解码仅 55.9s。

新增开关 clear_vram_before_decode（默认关，向后兼容）：在解码入口前调用新增的 cleanup_before_decode()，只卸载 BaseModel（DiT 及其 SigmaShift clone），文本编码器与视频/音频 VAE 通过 keep_loaded 保持驻留，下一段复用 VAE、无需重载。

新增控件追加在性能组数组末尾，避免旧工作流 widgets_values 位置错位；同步补齐前端标签/提示与中英文 i18n。


### #286 [open] ISSUE: Adding Traditional Chinese (Taiwan) to ComfyUI_MiniMaxH3_Director
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/286  by monarchsofchaos004-gif 2026-11-19T01:14:50Z

Hi! The README is available in Simplified Chinese (`README_ZH.md`). Readers in Taiwan and Hong Kong use different characters and quite a few different technical terms, so a separate version reads much more naturally.

My plan is to work from the English text and write it the way it is said in Taiwan. It would not be a character-by-character conversion of the Simplified version.

If that sounds useful I will open a PR. If you prefer not to maintain an extra language, just close this issue, no hard feelings.


### #285 [closed] PR: 分段导出：seg_XXXX.mp4 编号从 1 开始，与界面「素材组 N」对齐
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/285  by xb191799981 2026-11-18T00:51:59Z

# 分段导出：`seg_XXXX.mp4` 编号从 1 开始，与界面「素材组 N」对齐

## 动机

导出方式选「分段导出」时，文件名从 `seg_0000.mp4` 起，但界面上的分组是「素材组 1 / 2 / 3」、日志也是 `segment #1 mp4 saved`。文件名比组号小 1，逐条对比视频和分组时每次都要心算偏移。

这个改动让磁盘上的编号直接等于组号：**第 1 组 → `seg_0001.mp4`，第 2 组 → `seg_0002.mp4`**，方便边看片边对照分组调整。

## 改动

`director/segment_mp4_export.py`，一行：

```diff
 def segment_mp4_path(run_dir: Path, seg: SegmentPlan, *, suffix: str = "") -> Path:
     tag = f"_{_safe_mp4_suffix(suffix)}" if _safe_mp4_suffix(suffix) else ""
-    return Path(run_dir) / f"seg_{int(seg.index):04d}{tag}.mp4"
+    return Path(run_dir) / f"seg_{int(seg.index) + 1:04d}{tag}.mp4"
```

`seg.index` 是 0 起的时间线下标，`+1` 后与 UI / 日志一致。带后缀的变体随之同步：

```
index=0 → seg_0001.mp4   seg_0001_pre.mp4   seg_0001_p2.mp4   seg_0001_facepre.mp4
index=1 → seg_0002.mp4   seg_0002_pre.mp4   seg_0002_p2.mp4   seg_0002_facepre.mp4
index=2 → seg_0003.mp4   seg_0003_pre.mp4   seg_0003_p2.mp4   seg_0003_facepre.mp4
```

`copy_segment_mp4_suffix()` 复用了 `segment_mp4_path()`，所以「先写 `seg_0002.mp4`、再复制成 `seg_0002_facepre.mp4`」这种成对关系不会错位。

## 编号规则

**按时间线下标，不按本次运行顺序。** 用「选择运行」只跑第 2、4 组时，输出 `seg_0002.mp4` / `seg_0004.mp4`（中间留空），与界面里的组号保持一一对应。如果你更希望"这次导出几个就从 1 连续编号"，可以改，但那会让文件和组号再次对不上，所以这里选了前者。

## 影响面

- 这些 mp4 **只写不读**：代码里没有任何地方按文件名反解组号（`mp4_export_kind()` 只看后缀）。
- `segment_cache.py` 里的 `seg_XXXX.meta.json` / `.av.pt` 是**缓存目录**里的另一套命名，和导出目录无关，未改动。
- 每次「分段导出」都新建一个时间戳目录，不会和历史运行里 0 起的文件名混在一起。
- 该函数只做字符串格式化，无新增测试文件（模块顶层依赖 `folder_paths`，单测需要 stub；上面的输出是实际调用 `segment_mp4_path()` 得到的）。

与 PR #284（SelfLift 分块掩码）互不相关，改的是不同文件，无冲突。



### #284 [closed] PR: 修复：空间分块未裁剪模型侧 denoise_mask，导致 SelfLift + 引用上段（引导+重绘）崩溃
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/284  by xb191799981 2026-11-17T02:18:10Z

# 修复：空间分块未裁剪模型侧 `denoise_mask`，导致 SelfLift + 引用上段（引导+重绘）崩溃

> Fix: spatial tiling passed a full-canvas `denoise_mask` to a tile-sized latent,
> crashing `out[0] * denoise_mask` in `comfy/ldm/minimax/model.py`.

## 现象 / Symptom

| 项 | 值 |
| --- | --- |
| 段间引导 | `引导+重绘` |
| 素材组 2 | 勾选 `引用上段` |
| SelfLift | 已接入（`highres_steps=2`，高清分块开启） |
| 输出 | 1376×768，2 段（124f + 124f） |

跑到**素材组 2**（第二段）时报错：

```
RuntimeError: The size of tensor a (60) must match the size of tensor b (86) at non-singleton dimension 4
```

关键栈帧：

```
director/selflift/sample.py:559            sample_selflift_stage  →  high_out = sample_single_stage(...)
director/core_sampling.py:205              SamplerCustomAdvanced.execute(...)
director/spatial_tiled_sampling.py:476     pred = _invoke_model(model_k, tile_x, sigma, tile_mask, call_kw)
comfy/ldm/minimax/model.py:706             out[0] = out[0] * denoise_mask
```

日志里能直接看到分块范围：

```
spatial tiles: axis=W size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
```

**只走「引导」不接 SelfLift 不会崩**，因为那条路径不进空间分块；整幅 × 整幅永远对得上。
所以触发条件其实是「**空间分块 + 存在保留掩码**」，与 SelfLift 只是常常同时出现。

## 根因 / Root cause

1. 勾选「引用上段」后，`apply_latent_continue` 会给 latent 挂 `noise_mask`（前 N 个 token 置 0，锁住上段画面）。
2. Sampler 在采样开始时调用一次 `MiniMaxH3.extra_conds`，用**整幅** `latent_shapes` 把掩码解包成 `[1,1,T,H,W]`（本例 `[1,1,47,48,86]`）存进 `conds["denoise_mask"]`，整个采样过程不再更新。
3. 高清分块沿长边切瓦片：86 宽 → 第一块 `0:59`，为补齐偶数 pad 成 **60**。
4. `_tiled_model_call` 把 `x` / `latent_image` / `noise` / `keyframes` / `refs` / `latent_shapes` 都按瓦片裁了，**唯独漏了 conds 里的 `denoise_mask`**。
5. DiT 前向 `out[0](W=60) * denoise_mask(W=86)` → 崩溃。

## 修复 / Fix

`director/spatial_tiled_sampling.py`（+29 行）：

- 新增 `_crop_mask_to_tile()`：把 `[B,1,T,H,W]` 掩码按瓦片裁剪，并用**与瓦片 latent 完全相同的 `_pad_even_hw`** 补齐，保证两侧形状严格一致。
- `_install_tile_payloads()`：把 `model_conds["denoise_mask"]` 一并裁成瓦片版，登记进 `restorations`，调用结束随其它 cond 一起还原。
- 音频掩码不裁 —— 音频本来就不分块。
- 掩码在空间方向是常数（整幅同值），裁剪不改变任何数值。

附带收益：分块期间 `model.latent_shapes` 已是瓦片形状，`_PrefixRemask._sync_shapes` 因此能拿到正确尺寸，那条 packed 掩码分支也不会再错位。

## 测试 / Test

新增 `tests/test_spatial_tiles_mask.py`（unittest，无需 ComfyUI，stub 掉 `comfy.utils.pack/unpack_latents`），用假模型原样复刻 `out[0] * denoise_mask` 这一步：

```
# 未打补丁
RuntimeError: The size of tensor a (60) must match the size of tensor b (86) at non-singleton dimension 4

# 打补丁后
Ran 2 tests in 0.043s
OK
```

断言内容：瓦片宽度为 `[60, 52]`、每块瓦片 latent 与瓦片掩码形状相等、调用结束后 `denoise_mask` cond 与 `latent_shapes` 均被还原。

运行方式（与 `tests/test_audio_io.py` 一致，用 ComfyUI 的 Python）：

```bash
"<ComfyUI>/python/python.exe" tests/test_spatial_tiles_mask.py
```

## 兼容性 / 风险

- 只在 `enable_tiling` 且存在 `denoise_mask` 时生效；`tile_count=1`、无掩码、非 MiniMaxH3 的路径行为完全不变。
- 掩码形状与整幅 latent 不符时 `_crop_mask_to_tile()` 直接返回 `None` 并跳过，不会让原先能跑的情况变坏。
- 不改公共接口，不动 `comfy/` 下的任何文件。

## 复现条件汇总

```
MiniMaxH3Director（2 个素材组）
  + 段间引导 = 引导+重绘
  + 素材组 2 勾选「引用上段」
  + SelfLift 节点接入且「高清分块」开启（tile_count ≥ 2）
⇒ 第二段首次分块前向即崩
```



### #283 [open] ISSUE: 连上二采节点后，confirm_first_pass=true后，一采会两次vae解码，导致抽卡比较浪费时间
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/283  by xpb198011-jpg 2026-11-13T10:56:12Z

应该是输出解码了一次，写缓存解码了一次。输出和写缓存能不能共用一次vae解码结果，要不低算力显卡抽卡很浪费时间。
<img width="560" height="706" alt="82f0edfe-2b09-48a2-b7b4-b15e074553d6" src="https://github.com/user-attachments/assets/ed1aa7c1-0053-42ca-ac24-94775b48f100" />

<img width="2600" height="212" alt="f6cb7fbc-1925-4bd1-bcb6-6b4ea91596c1" src="https://github.com/user-attachments/assets/89ea6d18-5fdc-4d76-9207-63304dbe3f37" />


### #282 [open] ISSUE: refmode 这个功能能对接吗
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/282  by heyuquan 2026-11-13T07:57:42Z




### #281 [open] ISSUE: MiniMax H3 Director 節點 分段載入LORA功能 顯示「未找到 LoRA」。
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/281  by haurvatat666-dotcom 2026-11-11T13:20:54Z

您好，想請教一個 RunningHub 相容性問題。

我使用您的 MiniMax H3 Director 節點，在每個影片段落中選擇想使用的 LoRA 時，點開 Director 內建的 LoRA 選單，卻顯示「未找到 LoRA」。

但在同一個 RunningHub 工作流中，另外新增原生 LoRA 載入節點，則可以正常找到並選擇平台提供及使用者上傳的 LoRA。

請問 Director 的逐段 LoRA 選擇功能目前是否不支援 RunningHub 平台的 LoRA 資源？

還是需要額外設定或使用其他方式才能正常讀取？

謝謝！


### #280 [closed] PR: 修复导出导演包在大时间线下失败；新增素材组批量删除与随机种子记录
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/280  by JiuChenTian 2026-11-11T11:51:35Z

# 修复「导出导演包」在大时间线下导出失败

## 问题

时间线素材较多（本例：202 个文件 / 1.65 GB 唯一素材）时点击「导出导演包」失败：ComfyUI 界面卡死约 90 秒，随后前端长时间无响应、最终无任何文件产出。服务端日志无 `Director pack export failed`，而是出现浏览器断连：

```
Exception in callback _ProactorBasePipeTransport._call_connection_lost()
```

## 根因（三层叠加）

1. **重复打包**：时间线里 `segments`、`batchWorkspaces.mixed.segments`、`videoWorkspaces` 保存同一批素材的多份引用副本。导出只重写 `segments`，其余副本被 `_walk_rewrite_leftovers` 兜底再拷贝一遍进 `extra/`。实测：唯一素材 1.65 GB 被打成 **1039 个文件 / 3.9 GB（zip 4.05 GB）**，膨胀 2.4 倍。
2. **阻塞事件循环**：`minimax_export_pack` 是 `async def` 但同步执行 `build_export_pack`，构建期间（约 90 秒）整个 ComfyUI 无响应。
3. **前端内存溢出**：大包经 POST 响应流回浏览器，`readResponseBytes()` 将约 4 GB 读入 JS 堆再转 Blob，标签页 OOM/断连——即用户看到的“导出失败”。

## 改动

`director/pack.py`
- `_rewrite_one` 及各 `_rewrite_*`/`_explode_card`/`_walk_rewrite_leftovers` 增加 `dedup`（源绝对路径 → 包内路径）与 `sized`：同一源文件只拷贝一次，所有引用复用同一包内路径；`totalBytes/fileCount` 预估同步去重。
- `zipfile` 由 `ZIP_DEFLATED` 改 `ZIP_STORED`：mp4/png 已自带压缩，DEFLATE 徒耗约 80 秒 CPU。
- 导出构建改为 `await asyncio.to_thread(build_export_pack, ...)`，不再冻结事件循环。
- 大包（> INLINE_JSON_MAX）不再把 zip 塞进 POST 响应：返回 JSON（`filename/downloadName/bytes/missing`），由已注册的 GET `/minimax/director/download_pack` 提供下载（zip 仍在导出根目录，TTL 自动清理）。小包的 zipB64 路径不变。

`web/js/minimax_pack.js`
- 收到带 `filename` 的 JSON 时改为 `<a download>` 触发浏览器原生下载（流式落盘、原生进度条、不占 JS 内存）；旧的小包 zipB64 与流式响应兼容分支保留。

## 实测（真实用户时间线，离线复现）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 构建耗时 | 88.7 s（阻塞服务器） | 4.2 s |
| 拷贝次数/字节 | 1039 次 / 3883 MiB | 202 次 / 1649.9 MiB |
| zip 大小 | 4,053,698,392 B | 1,733,042,799 B |
| dryRun 预估 | 4.07 GB（虚高） | 1.73 GB（与真实一致） |
| 包内引用完整性 | — | 329 条目，0 处悬空引用 |

导入侧（`import_extracted_pack`）无需改动：timeline.json/group.json 中复用同一包内路径的引用在解包后均存在。

---

## 附：后两个提交（UI 小功能 + 种子记录改进，可按需要拆分为单独 PR）

- **批量删除**：开启「选择运行」后，素材组工具栏出现「删除所选」按钮，一次删除所有勾选的分组；沿用原有单组删除路径（倒序逐个删），全选时保留第一个分组（插件的“至少保留一组”硬限制）。
- **随机种子记录（只读）**：每个分组卡片在「参考图尺寸」左侧显示该组最后一次**完整跑完**所用种子。执行器在某一段的视频完整生成结束时立刻把种子写入服务端台账 `output/minimax_seg_cache/<节点缓存名>/seg_XXXX.seed.json`；前端在队列结束和打开工作流时通过新只读路由 `GET /minimax/director/seed_records` 读取展示。中途取消/报错的段落不会改写记录，台账独立于缓存清理（clear_segment_cache 保留 *.seed.json），重启 ComfyUI、不保存工作流也不丢失。旧版存于 timeline_data 的 seg.lastSeed 仍作为回退显示。



### #279 [open] ISSUE: 能否增加不现分段有自己独立的种子数？
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/279  by buddytian-jpg 2026-11-06T11:47:13Z

能否借鉴一下这里：https://github.com/guomenghao/ComfyUI-CineStack-H3 

给每个分段可以设定自己独立的种子数，全局用一个种子数，在某些时候有些不方便。


### #278 [closed] PR: 维护汇总：Pika H3 Flex、缓存/导出与内存修复、导演台 Pro
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/278  by H-Bole 2026-11-01T18:44:59Z

## 为什么 fork / Why this fork

我在本机持续使用这套导演台，想给修复、工作流兼容和后续更新建立一个明确的维护入口。本轮审查时还有 20 个未关闭 PR，因此趁 Pika H3 Flex 发布，把积压的有效改动、明确错误和 Pika 适配集中整理了一次，并以 **MiniMax H3 导演台 Pro** 独立维护：
https://github.com/H-Bole/ComfyUI-MiniMax-H3-Director-Pro

这个 PR 直接回馈当前 Pro 的完整版本。感谢 AIMixer 和原贡献者；保留上游 Apache-2.0 许可、原作者提交与分主题历史，没有把所有改动压成一个提交。逐项检查不等于全部采用，也不等于全部问题已解决。

**请注意仓库身份变化：本 PR 包含 Pro 改名，README 安装地址、插件元数据、反馈及更新来源指向 H-Bole 的维护仓库。** 这也是当前发布版本的实际行为，上游可据此决定合入或调整。

## 主要变化

- 缓存身份槽、部分运行/续跑、一采 hold、成品视频替代与纯拼接缺缓存硬失败；统一硬切及内存/磁盘拼接语义。
- 批次转帧、磁盘映射生命周期及释放修复；不清理其他进程的工作集，不在单条任务结束后无条件卸载全部模型。
- 分块 mask/共享条件/PrefixRemask 及异常恢复；补齐末帧导出、上传 UUID、Group Combine 和设备选择修复。
- 原声锁定为独立可选生成条件，默认关闭，保持原声导出语义；音频 latent 随缓存保存。
- 同一导演台增加 H3 / Pika-H3，支持纯文、首帧、尾帧、首尾帧、多模态参考，按任务自动匹配 Comfy LoRA，固定完整六步配方，复用原生加载、条件和采样。
- 按原生节点边界释放预取/转换缓冲与动态显存水位，减少大节点内重复驻留；兼容非 AIMDO 后端。
- 自动/中文/English、固定配方摘要、旧工作流迁移、保存重开；提供五类 API 示例及完整模型下载与安装说明。
- 保留现用 Avatar SelfLift 桥接与独立视频脸修节点，补充相关可选脸修设置；改编代码保留 Carasibana 的 MIT 署名和许可。

## 来源与相关讨论

直接保留作者历史整合：#276 #268 #231 #206 #175 #225 #221 #197 #212 #254 #267 #253 #118。
部分吸收 hold 修复：#187。其余 PR 的已覆盖、部分吸收或不采用理由见完整记录。

明确修复相关：#255 #252 #219 #181 #249 #247 #235 #230 #185 #33。
内存/拼接相关（部分缓解，不能承诺所有长片不再 OOM）：#274 #243 #220 #176 #88 #75 #32 #16 #13。
#277 #272 #240 等环境崩溃、SelfLift 质量和脸修成品问题仍列待复现，没有以本 PR 统称关闭。

截至本 PR 提交前复核，上游 main 仍为 `4c6245e518dd8a494f3afc4f3cccd53f22d82d10`；审查快照覆盖 71 个 PR 和 205 个 Issue，正文/差异/讨论已逐项记录：
[PR / Issue 处理表](https://github.com/H-Bole/ComfyUI-MiniMax-H3-Director-Pro/blob/main/docs/upstream-review.md)。

## 验证与限制

Python/JS、缓存/拼接/分块等针对性 CPU 检查及真实浏览器保存恢复通过。干净重装后，完整本机插件集可注册 Pro 及兼容节点；7 份常用工作流无缺失节点，加载模型可解析。Pika 纯文安装验收已实际生成并完整解码：832×480、124 帧、24fps、seed42，32kHz 双声道音轨。

复用已有同条件性能记录：原生/Pro 热态各三次，中位数 **123.45s / 125.04s**；单次进程冷启动的实际旧维护版/Pro 为 **225.11s / 140.96s**。编码器预设与音频码率存在已披露差异，不据此声称压缩成品等价、每次更快或画质更优。

没有扩大到五任务 GPU 全矩阵；新迁回的 Avatar 桥接/独立脸修完成契约、CPU 模拟及界面验证，未重新进行 GPU 全流程画质验收。详细参数、限制与待办见：
[性能与验收](https://github.com/H-Bole/ComfyUI-MiniMax-H3-Director-Pro/blob/main/docs/performance.md) · [模型下载](https://github.com/H-Bole/ComfyUI-MiniMax-H3-Director-Pro/blob/main/docs/models.md) · [使用与兼容](https://github.com/H-Bole/ComfyUI-MiniMax-H3-Director-Pro/blob/main/docs/pro-guide.md)

This consolidates the independently maintained Pro fork, reviewed community fixes and Pika H3 Flex support while retaining contributor history. The Pro branding and repository/update URLs are intentional parts of this submission. Validation limits and unresolved reports are documented; this is not a claim that all reviewed issues are fixed.


### #277 [open] ISSUE: [Bug] Director r2v process disappears silently between TE staged and UNET staged (ComfyUI 0.41.0, 5 runs / 4 fresh instances, no traceback)
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/277  by wrl870426 2026-10-26T19:00:25Z

## Symptom
A single-segment Director reference-to-video job dies silently at the same point every time: after the log lines showing VAE staged (~4965MB) and TE staged (~14956MB), the ComfyUI process disappears with no traceback, no error, and the port stops responding. It never reaches UNET staged. The same driver file succeeded once at 21:55, then died on identical resubmission at 23:12, and died 5 times across 4 fresh ComfyUI instances at the identical point.

## Environment
- ComfyUI 0.41.0 (HEAD 98cc0bf84b7cb1383654c5b402e7612d875d6d22, 2026-10-15)
- Director package: unknown-source install (no .git in the package dir; file mtimes 2026-09-30; declared version 1.0.0 in pyproject)
- comfy-kitchen 0.2.37, comfy-aimdo 0.5.6
- GPU: RTX 3090 24GB, driver 570.124.04, CUDA 12.8
- Job: single-segment r2v, 256x480, 175 frames, 4 steps, video reference (22-frame previz) + 8 static refs, seed 10606, no refine / face_refine / bridge / selflift

## What was ruled out
- Kernel OOM: cgroup oom_kill = 0 for the container; no OOM-killer evidence.
- Poisoned segment cache: the only on-disk Director state is a segment cache from a different run (544x960, 22 frames, face_refine true, no video reference, pre 20 steps) — it differs from the crashing job on every field, the cache read gate is closed for this job shape (single segment, no refine/face_refine), and its files predate the crash window. A full read-only survey of all 11 Director-writable paths found zero changes inside the crash window (21:55-22:30).
- Temp/upload/pack leftovers: those directories do not exist; the crashing job is submitted via API, not the web upload interface.
- Process-local caches: they die with the process and cannot explain identical death across 4 fresh instances.
- Director as a whole is NOT broken: a seg05 v3 Director pure-image r2v job succeeded on the same setup. Only this 06 submission group dies.
- Native MiniMax H3 I2V on the same machine does not crash (it passes the crash point) — but it does not produce the requested gesture either, so this is not a usable bypass for this shot.

## Closest existing report, and why it differs
#274 (0.39.1, Windows, RTX 5080 16GB, Director r2v, 10 segments x 243 frames) shows the exact same staged log lines (VAE ~4965MB, TE ~14956MB, then UNET ~19995MB staged) and walks past the crash point: all 10 segments complete, and it only fails later at concat time with a CPU OOM (with traceback and "Prompt executed"). Our failure sits exactly in the gap #274 never covers — between TE staged and UNET staged — and is silent, with no traceback and a dead port. The staged sequence itself is therefore not the killer; something in the TE-to-UNET gap under 0.41.0 + this checkout is.

## Ask
Is this a known 0.41.0 compatibility issue in the TE-to-UNET staging path, and is there a diagnostic (log flag, env switch, or minimal probe) that can distinguish the two candidate paths (TE video-block forward vs UNET staging) without a full run? Any guidance on what to capture next time it reproduces would help — we have frozen further Director production submissions for this shot pending a new diagnostic anchor.


### #276 [open] PR: Fix prompt group wheel scrolling on canvas overlay
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/276  by chrislingxiao 2026-10-26T14:56:41Z

Mouse-wheel input over the Director's prompt-group list zooms the ComfyUI canvas instead of scrolling the groups. This was reproduced with ComfyUI frontend 1.55.18; Page Up / Page Down still scroll the list.

Intercept wheel events during document capture only when the target is inside `.mmx-host .bd-batch-list`, before the canvas overlay forwards them to its zoom handler. The passive listener leaves native list and text scrolling enabled. Contain vertical overscroll at the list boundary.

Validation:
- `node --check web/js/minimax_image_batch.js` and `git diff --check` pass on the current upstream code.
- The same fix was manually tested in an installed Director with 49 prompt groups: scrolling down moved groups 5–7 to 6–8, scrolling up returned to 5–7, and canvas zoom stayed at 60% during both operations. This UI test used an older installed plugin revision; the latest upstream revision received syntax and diff checks.



### #275 [open] ISSUE: 用SelfLift后，两段之间会闪屏（段间引导已开启）
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/275  by pengxu416 2026-10-26T13:03:25Z

用SelfLift后，两段之间会花屏（段间引导已开启），只有引导方式改成引导+重绘才会好点

<img width="1183" height="674" alt="Image" src="https://github.com/user-attachments/assets/2fb48a42-71b0-464c-853a-9dd23392a46a" />


### #274 [open] ISSUE: 内存溢出，10个片段每个10秒，在最后合并时报错。日志如下
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/274  by JustDance-d 2026-10-25T18:15:25Z

节点在执行过程中发生错误。


# ComfyUI Error Report
## Error Details
- **Node ID:** 12
- **Node Type:** MiniMaxH3Director
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3233710080 bytes.

## Stack Trace
```
  File "D:\ComfyUI-V9.5\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\ComfyUI-V9.5\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\ComfyUI-V9.5\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "D:\ComfyUI-V9.5\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^

  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\nodes\director.py", line 343, in execute
    execute_director_plan_core(

  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1906, in execute_director_plan_core
    combined = concat_continuous_chunks(export_chunks, export_segments, plan)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\segment_continuity.py", line 1781, in concat_continuous_chunks
    body = _additive_opening_luma(body, left)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\segment_continuity.py", line 670, in _additive_opening_luma
    out = body.clone()
          ^^^^^^^^^^^^

```
## System Information
- **ComfyUI Version:** 0.39.1
- **Arguments:** D:\ComfyUI-V9.5\main.py --port 3198 --auto-launch --preview-method auto --disable-smart-memory --cuda-malloc
- **OS:** win32
- **Python Version:** 3.12.9 (tags/v3.12.9:fdb8142, Feb  4 2025, 15:27:58) [MSC v.1942 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.9.1+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 5080 : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 17066033152
  - **VRAM Free:** 15594157568
  - **Torch VRAM Total:** 33554432
  - **Torch VRAM Free:** 15463936

## Logs
```
2026-10-26T00:36:58.056440 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 1/10 done (243 frames, task=r2v)
2026-10-26T00:36:58.095440 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:36:58.221444 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:36:58.612505 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:36:58.868505 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:36:59.058506 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T00:36:59.184505 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T00:37:07.815114 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T00:37:07.823114 - ^[[32m[INFO]^[[0m Director continuity: interior keyframe anchors enabled
2026-10-26T00:37:07.823114 - ^[[32m[INFO]^[[0m Director continuity: keyframe/ref coexistence enabled
2026-10-26T00:37:07.824114 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T00:37:07.856115 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T00:37:08.192125 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T00:37:08.254126 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T00:42:23.962600 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:42:24.525355 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:43:11.290137 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T00:43:12.839502 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T00:43:44.954942 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T00:43:45.701662 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T00:43:45.704664 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T00:43:45.705662 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T00:43:45.725662 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T00:43:45.725662 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T00:43:45.801168 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T00:43:45.998671 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T00:43:46.107686 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T00:43:46.715161 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T00:46:57.003489 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:46:58.066707 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:47:50.092486 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T00:47:50.634289 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T00:47:51.757869 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T00:47:52.436252 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T00:47:56.099755 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 2/10 done (243 frames, task=r2v)
2026-10-26T00:47:56.583045 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:47:56.714650 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:47:57.185044 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:47:57.502671 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:47:57.780254 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T00:47:57.964136 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T00:48:20.279992 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T00:48:20.281993 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T00:48:20.309993 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T00:48:20.900653 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T00:48:20.971653 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T00:53:31.099768 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:53:31.780553 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:54:18.536838 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T00:54:19.886625 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T00:54:50.678840 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T00:54:51.432836 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T00:54:51.435837 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T00:54:51.436837 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T00:54:51.459837 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T00:54:51.459837 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T00:54:51.492461 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T00:54:51.806265 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T00:54:51.943265 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T00:54:52.639784 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T00:58:02.071672 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:58:02.979260 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:58:50.163028 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T00:58:50.569320 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T00:58:51.282709 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T00:58:51.599560 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T00:58:54.985630 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 3/10 done (243 frames, task=r2v)
2026-10-26T00:58:55.370482 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T00:58:55.485058 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:58:55.917209 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:58:56.195579 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T00:58:56.428580 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T00:58:56.566163 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T00:59:04.648819 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T00:59:04.649820 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T00:59:04.682820 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T00:59:05.371714 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T00:59:05.444714 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:04:13.793109 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:04:14.532881 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:04:58.105295 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:04:59.414547 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:05:07.991296 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:05:08.775503 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:05:08.778505 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:05:08.779504 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:05:08.803503 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:05:08.803503 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:05:08.838131 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:05:09.132412 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:05:09.258197 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:05:09.578288 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:08:12.416549 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:08:13.124936 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:08:59.595531 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T01:09:00.052859 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T01:09:00.685912 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:09:01.096240 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:09:06.788151 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 4/10 done (243 frames, task=r2v)
2026-10-26T01:09:07.218291 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:09:07.345289 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:09:07.803384 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:09:08.086985 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:09:08.344986 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T01:09:08.508165 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T01:09:16.449065 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T01:09:16.450065 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T01:09:16.485615 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:09:17.043192 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:09:17.118191 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:14:15.762171 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:14:16.766685 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:14:57.871231 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:14:59.046336 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:15:07.663613 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:15:08.449429 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:15:08.451431 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:15:08.453429 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:15:08.472430 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:15:08.472430 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:15:08.506431 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:15:08.703606 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:15:08.813606 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:15:09.205397 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:18:08.373499 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:18:09.342704 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:18:54.680386 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T01:18:55.029380 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T01:18:55.694702 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:18:55.893355 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:18:59.016257 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 5/10 done (243 frames, task=r2v)
2026-10-26T01:18:59.409473 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:18:59.533326 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:18:59.975511 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:19:00.262161 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:19:00.516830 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T01:19:00.641424 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T01:19:08.691486 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T01:19:08.692486 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T01:19:08.728486 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:19:09.379269 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:19:09.428269 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:24:08.772808 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:24:09.810927 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:24:50.821148 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:24:52.003008 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:25:00.595161 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:25:01.337088 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:25:01.340090 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:25:01.342088 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:25:01.360682 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:25:01.360682 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:25:01.395684 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:25:01.707263 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:25:01.842814 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:25:02.220115 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:28:01.930867 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:28:02.892498 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:28:48.224961 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T01:28:48.577737 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T01:28:49.256590 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:28:49.465590 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:28:52.729984 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 6/10 done (243 frames, task=r2v)
2026-10-26T01:28:53.098278 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:28:53.219609 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:28:53.663959 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:28:53.950520 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:28:54.206663 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T01:28:54.370662 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T01:29:02.348767 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T01:29:02.349767 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T01:29:02.383767 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:29:03.030868 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:29:03.087409 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:34:03.292255 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:34:04.143206 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:34:45.580597 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:34:46.747690 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:34:54.908335 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:34:55.672721 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:34:55.674723 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:34:55.676721 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:34:55.699721 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:34:55.699721 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:34:55.735721 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:34:56.049607 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:34:56.197165 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:34:56.525164 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:37:57.226572 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:37:58.131697 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:38:43.551523 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T01:38:43.931186 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T01:38:44.585312 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:38:44.989572 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:38:49.705597 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 7/10 done (243 frames, task=r2v)
2026-10-26T01:38:50.057831 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:38:50.187830 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:38:50.613709 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:38:50.870275 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:38:51.106825 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T01:38:54.402439 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T01:39:02.180745 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T01:39:02.181745 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T01:39:02.214746 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:39:02.820694 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:39:02.871694 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:44:18.663541 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:44:19.545495 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:45:02.597781 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:45:03.805287 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:45:32.932611 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:45:33.738034 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:45:33.741036 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:45:33.743036 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:45:33.764035 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:45:33.764035 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:45:33.798037 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:45:34.106781 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:45:34.242781 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:45:34.776469 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:48:47.980837 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:48:48.916807 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:49:36.119411 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T01:49:36.548174 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T01:49:37.282416 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:49:37.715928 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T01:49:41.346616 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 8/10 done (243 frames, task=r2v)
2026-10-26T01:49:41.729731 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:49:41.865731 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:49:42.322115 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:49:42.604441 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:49:42.854011 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T01:49:43.012012 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T01:49:50.961893 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T01:49:50.963895 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T01:49:51.002443 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:49:51.675296 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:49:51.762296 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T01:55:03.226007 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:55:04.193017 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T01:55:47.231322 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T01:55:48.474061 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T01:56:23.856351 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T01:56:24.645853 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T01:56:24.648856 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T01:56:24.650854 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T01:56:24.670644 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T01:56:24.671646 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T01:56:24.707644 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T01:56:25.019541 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T01:56:25.205550 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T01:56:25.685808 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T01:59:34.628336 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T01:59:35.435905 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:00:20.961133 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T02:00:21.376486 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T02:00:22.017505 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T02:00:22.404850 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T02:00:28.218346 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 9/10 done (243 frames, task=r2v)
2026-10-26T02:00:28.565042 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T02:00:28.686043 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:00:29.120771 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:00:29.379617 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:00:29.589796 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-10-26T02:00:29.705366 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged. 0 patches attached. Force pre-loaded 410 weights: 4572 KB.
2026-10-26T02:00:37.886216 - ^[[32m[INFO]^[[0m Director continuity: pin from previous first-pass latent 672x1216 (refined canvas was 768x1376).
2026-10-26T02:00:37.887216 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent) + 40 audio steps
2026-10-26T02:00:37.932207 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T02:00:38.565990 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T02:00:38.617540 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-10-26T02:05:37.076395 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T02:05:37.977714 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:06:21.335150 - ^[[32m[INFO]^[[0m Director H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 → 768×1376
2026-10-26T02:06:22.682973 - ^[[32m[INFO]^[[0m H3 latent upscaler temporal chunking: T=82 chunk=24 overlap=5
2026-10-26T02:06:30.793844 - ^[[32m[INFO]^[[0m H3 latent upscale: minimax_h3_latent_upscaler_3d_bf16.safetensors 672×1216 latent 42x76 → 48x86 (pixel 672×1216 → 768×1376, scale=1.14)
2026-10-26T02:06:31.593613 - ^[[32m[INFO]^[[0m Director refine: re-pin from previous refined tail 768x1376
2026-10-26T02:06:31.595616 - ^[[32m[INFO]^[[0m Director continuity: pasted 22-frame tail into current prefix (7 steps).
2026-10-26T02:06:31.596613 - ^[[32m[INFO]^[[0m Director continuity: pinned 22 video frames (latent), audio off
2026-10-26T02:06:31.615612 - ^[[1m^[[33m[WARNING]^[[0m Refine sigma pass is short. A custom refine_model without Turbo can look like noise. Unwire refine_model so the first-pass UNET (Turbo+Sage) is reused, or use a matching second-pass UNET.
2026-10-26T02:06:31.615612 - ^[[32m[INFO]^[[0m Director refine pass 1/1 (sigmas wired euler 3-step, custom model, seed=666)
2026-10-26T02:06:31.650613 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-10-26T02:06:31.901196 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-10-26T02:06:32.048993 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1142 KB.
2026-10-26T02:06:32.467595 - ^[[32m[INFO]^[[0m spatial tiles: axis=H size=86 n=2 overlap=128px/8 latent ranges=[(0, 59), (35, 86)]
2026-10-26T02:09:37.662400 - ^[[1m^[[33m[WARNING]^[[0m TAE decode failed, falling back to Latent2RGB: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 12681216 bytes.
2026-10-26T02:09:37.787400 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-10-26T02:09:38.643441 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 352 KB.
2026-10-26T02:10:41.440087 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3AudioVAE
2026-10-26T02:10:41.873633 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-10-26T02:10:42.568540 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T02:10:42.978238 - ^[[32m[INFO]^[[0m Segment continuity: export opening grade 12f weight=0.70 blur=64
2026-10-26T02:10:46.425379 - ^[[32m[INFO]^[[0m MiniMax H3 Director segment 10/10 done (243 frames, task=r2v)
2026-10-26T02:10:47.194937 - ^[[32m[INFO]^[[0m Segment continuity: additive opening luma 12f (seam_gap=0.009 spike=0.004, 0.360→0.372, cap±0.100)
2026-10-26T02:10:47.981458 - ^[[32m[INFO]^[[0m Segment continuity: additive opening luma 12f (seam_gap=0.034 spike=0.013, 0.308→0.409, cap±0.100)
2026-10-26T02:10:50.734488 - ^[[32m[INFO]^[[0m Segment continuity: additive opening luma 12f (seam_gap=0.005 spike=0.010, 0.522→0.460, cap±0.100)
2026-10-26T02:10:53.162554 - ^[[32m[INFO]^[[0m Segment continuity: additive opening luma 12f (seam_gap=0.014 spike=0.005, 0.399→0.358, cap±0.100)
2026-10-26T02:10:54.795965 - ^[[32m[INFO]^[[0m Comfy model compiler graph breaks: 46, rogues: 1
2026-10-26T02:10:55.320919 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3233710080 bytes.
2026-10-26T02:10:55.331305 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\ComfyUI-V9.5\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-V9.5\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-V9.5\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\ComfyUI-V9.5\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\nodes\director.py", line 343, in execute
    execute_director_plan_core(
  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1906, in execute_director_plan_core
    combined = concat_continuous_chunks(export_chunks, export_segments, plan)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\segment_continuity.py", line 1781, in concat_continuous_chunks
    body = _additive_opening_luma(body, left)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ComfyUI-V9.5\custom_nodes\ComfyUI_MiniMaxH3_Director\director\segment_continuity.py", line 670, in _additive_opening_luma
    out = body.clone()
          ^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:121] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3233710080 bytes.

2026-10-26T02:10:58.772385 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 01:42:17^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)


> **Comment by JustDance-d (2026-10-25T18:16:07Z):**

我本机 32 G 内存，设置了 100G 的虚拟内存。


> **Comment by JustDance-d (2026-10-25T18:16:17Z):**

5080 显卡。 



### #273 [open] ISSUE: 选中的分组在生成后会取消选中
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/273  by ruixiu 2026-10-23T09:21:49Z

选中的分组在视频生成完成以后会被取消选中状态，回到默认的第一个分组，一不小心就把第一个分组的提示词搞坏了，ctrl+z也无法恢复


### #272 [open] ISSUE: SelfLift + 段间引导：多段时第 2 段起高清阶段失效，画面明显变糊
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/272  by Allen-Guo0126 2026-10-22T09:20:35Z

# SelfLift + 段间引导：多段时第 2 段起高清阶段失效，画面明显变糊
## 环境
- ComfyUI `0.37.2`
- Windows / RTX 5090 34GB
- 插件版本：**最新 main（含 2026-10-13 的 `4c6245e5`）仍然复现**
  - 旧版（2026-09-18 的 `b390f828`）同样复现
- 已确认 `director/selflift/sample.py` 自 2026-09-18 首次提交后，除 2026-09-30 的报错文案外**采样逻辑未改动**，因此该问题在最新版依旧存在
## 复现配置
- 2 段，333 帧（158 + 175），640×1152，24 fps
- 段间引导 **开启**，重叠 22 帧
- SelfLift：`split_mode=highres_steps`、`highres_steps=2`、`lowres_scale=0.5`、`sampler_mode=euler`、`rho=0`、`native_low_carry=false`
- 为隔离问题，测试时**关闭 LoRA 与 Refine**，只测一采
## 现象
第 1 段正常，**第 2 段起明显模糊**，高清阶段似乎完全没有贡献细节（画面停留在 3D lift 的结果上）。
关闭 SelfLift 后多段完全正常；关闭段间引导后（SelfLift 保持开启）第 2 段也正常。
## 日志
```
SelfLift: 640x1152 → 320x576, k=6/8 (low 6 + high 2), euler, rho=0.00
SelfLift transition: video (1,24,47,36,20) → (1,24,47,72,40) (target 72x40), nested=True, re-noise σ=0.878049→0.8
   ← 段1：无 HQ resume，走 else 分支
Director continuity: pinned 22 video frames (latent) + 40 audio steps
SelfLift: 640x1152 → 320x576, k=6/8 (low 6 + high 2), euler, rho=0.00
SelfLift seam DC: prefix=7 clamp=0.050 max_abs=0.05000
SelfLift transition: video (1,24,62,36,20) → (1,24,62,72,40) (target 72x40), nested=True, re-noise σ=0.878049→0.8
SelfLift HQ resume: native inpaint keep-mask, prefix=7 tokens, σ=0.8
   ← 段2：命中 prefix_steps > 0 分支（sample.py:511）
```
注意 `max_abs=0.05000` 说明 `_LIFT_DC_CLAMP=0.05` 已被钳满，3D lift 出来的 prefix 与真实高清前驱的每通道中位差 ≥0.05。
## 量化结果
用 Laplacian 方差中位数衡量清晰度（对每个输出视频逐帧计算，再按 158/175 的段边界取中位数）：
| 测试 | 配置 | 段1 | 段2 | 段2/段1 |
|---|---|---|---|---|
| A | 基线：SelfLift + 段间引导 ON | 24.3 | **12.6** | **0.52x** |
| B | SelfLift + 段间引导 **OFF** | 24.4 | 22.9 | 0.94x |
| C | 段间引导 ON + `highres_steps=3` | 24.3 | 12.8 | 0.53x |
| E | 补丁：高清阶段改用 `resume_samples` + `zero_noise=True` | 24.3 | 13.3 | 0.55x |
| F | 补丁：在 E 基础上去掉 `lock_prefix_keep_mask()` 调用 | 24.3 | 13.3 | 0.55x |
| H | 补丁：在 F 基础上再剥掉 `noise_mask` 与 `PREFIX_STEPS_KEY` | 24.3 | 13.3 | 0.55x |
## 结论（供参考）
1. **不是步数问题**：`highres_steps` 2→3 无变化。
2. **不是噪声重注入机制**：换成与第 1 段等价的 `resume_samples` + 零噪声，无变化。
3. **不是 keep-mask，也不是 latent 继承的前缀键**：把 `noise_mask` 和 `PREFIX_STEPS_KEY` 全部剥掉，结果仍完全一致（13.3）。
三种截然不同的高清阶段起点给出的清晰度几乎一字不差，说明**段 2 的高清阶段产物没有真正进入成片**，而且问题不在"如何布置高清阶段"，更可能在**上游**（低清阶段，或段 2 的 conditioning / latent 长度）。
一个可疑点：段 2 的 latent 是 `T=62`（209 帧），而导出只有 175 帧（22 帧 pin + 12 帧 opening grade）。不确定 `positive` 的 token 数是否与这个 209 帧的 latent 匹配。
## 另一个尝试（不推荐）
把 `lowres_scale` 从 0.5 改成 0.8 后，两段清晰度都提升（段1 30.9 / 段2 29.2），**但会出现伪影**：画面顶部凭空出现亮斑、衣物上出现半透明重影涂抹。
推测原因：该 3D latent 放大器在所有正常记录中都运行在**精确 2.00×**；0.8 对应 1.25× 的非原生倍数，因此产生伪影。
## 当前规避方案
- 多段时**关闭 SelfLift**（已验证正常）
- 或保持 SelfLift 但**关闭段间引导**（接缝会受影响）
- 单段使用 SelfLift 正常
## 期望
希望 `prefix_steps > 0` 的段也能获得与第 1 段相同质量的高清收尾。


### #271 [open] ISSUE: [功能请求] 在 H3 Director 中原生接入 RefMod，复用角色参考包
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/271  by zj3983 2026-10-20T17:24:53Z

## 使用场景

正在用 H3 Director 制作多镜头短剧／漫剧，希望把同一角色的多张参考图预先保存为 RefMod，后续镜头直接选择角色参考包，减少重复加载图片和整理接线，同时继续使用导演台现有的分组、时间轴与首尾帧功能。

RefMod 项目：https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod

## 当前遇到的接入问题

目前使用的 Director 节点内部完成条件编码和采样，没有开放供外部 Apply H3 RefMod 使用的 CONDITIONING 输入，因此无法像原生 H3 分离节点工作流那样直接插入 Apply。

RefMod 上游提供了通过 MODEL outer-sampling hook 注入参考的 Continuum RefMod Bridge，但其说明主要针对 H3 Continuum，尚未实际验证它与 Director 的兼容性。本请求不是在报告已复现的 Bridge 故障，也不假定 Bridge 一定不兼容。

已查看 Director 上游近期更新：分段 LoRA 是模型权重适配器功能，不能直接作为 RefMod 参考包入口。

## 希望支持的行为

1. 提供官方支持的 RefMod 接入方式：例如接收 Load H3 RefMods 输出的可选输入，或正式验证并提供现有 Bridge 与 Director 的示例工作流。具体实现方式由维护者判断，不一定需要修改核心采样流程。
2. 能在保留现有时间轴的情况下，将角色 RefMod 应用到需要的片段，并调整参考强度。若实现成本较高，先支持导演台全局共用一个 RefMod bundle 也很有帮助。
3. 说明 RefMod 与现有首尾帧、r2v 参考图、段间引导、SelfLift／Refine 的适用范围或限制，避免重复注入参考。
4. 不接 RefMod 时保持原有工作流行为；附一个最小示例，并说明支持的模型模式和依赖版本。

## 当前环境与验证范围

- Windows，RTX 5090 32GB，ComfyUI 0.37.0。
- 本机 Director 提交：`5c7bdc85fcc7e35849a89d0ba805299801fc6282`。
- 常用工作流：Director + H3 FL2VA INT8 图生视频；环境中也有 Ref2VA 模型及 H3 视频／音频 VAE。
- 当前尚未安装 RefMod，也没有做 Director + Bridge 生成测试。本请求基于工作流接口检查与上游文档，属于功能需求，不是质量或性能结论。

核心诉求是：在熟悉的 Director 工作流里直接复用角色参考资产，不必为了使用 RefMod 拆开整套导演台流程。感谢维护！

> **Comment by chenfuhai1984 (2026-10-29T13:21:27Z):**

支持


> **Comment by wntyda (2026-10-29T15:14:58Z):**

嗨，同道中人。你說的Continuum RefMod Bridge節點我測試過了，與H3 Director 可以兼容。雖然你在下提示詞的時候@視頻1，肯定是抓不到，它會暗掉，但實際生成視頻，RefMod的角色參考包仍會生效。


> **Comment by chenfuhai1984 (2026-10-29T15:19:08Z):**

没有条件输出，你是如何接到导演台的？


> **Comment by chenfuhai1984 (2026-10-29T15:20:39Z):**

没有条件输出，你是如何接到导演台的？


> **Comment by wntyda (2026-10-30T00:55:20Z):**

@chenfuhai1984 
https://comfy.icu/node/MiniMaxH3RefModContinuumBridge。
不走條件輸出，走model。


> **Comment by chenfuhai1984 (2026-10-30T01:00:11Z):**

昨天仔细看了你的回复，研究了下，搞明白了，谢谢！！


> **Comment by ericsari (2026-10-30T16:22:52Z):**

Here's a direct patch to the original pack  .  Add the refmods input to the existing MiniMaxH3Director node.

Files to modify (4 files, all in C:/Users/user/ComfyUI_MiniMaxH3_Director/)
1. nodes/director.py — Add refmods input socket
Find: The face_refine optional input block (ends with }, before "bd_grp_advanced").

Add after it:

                "refmods": (
                    "H3_REF_MODS",
                    {
                        "tooltip": (
                            "Optional RefMod bundle from Luisacaotica ComfyUI-MiniMaxH3Mod "
                            "(Load H3 RefMods / Load H3 RefMod Axis). Injected into every "
                            "segment's conditioning before sampling, so the DiT attends to "
                            "your saved references. Best with task_type=r2v + ref2va UNET. "
                            "Unconnected = no refs (current behavior)."
                        ),
                    },
                ),
Find: face_refine=None, in the execute() method signature.

Change to:

        face_refine=None,
        refmods=None,
        sigmas=None,
Find: face_refine=face_refine, inside prepare_director_plan(...) call.

Change to:

            refine=refine,
            face_refine=face_refine,
            refmods=refmods,
        )
Find: VALIDATE_INPUTS method — add after the got_face check block:

            got_face = input_types.get("face_refine")
            if got_face is not None and got_face != "MMX_DIR_FACE_REFINE":
                return (
                    "face_refine: expected MiniMax H3 Director FaceRefine "
                    f"(MMX_DIR_FACE_REFINE), linked node returns {got_face}."
                )
        return True
No change needed to VALIDATE_INPUTS for refmods since H3_REF_MODS type validation is handled by ComfyUI automatically.

Find: DESCRIPTION string — add refmods note:

        "images_pre_face_refine is the video before face stitch "
        "(blocked unless FaceRefine is connected and「输出修脸前」is on). "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
Change to:

        "images_pre_face_refine is the video before face stitch "
        "(blocked unless FaceRefine is connected and「输出修脸前」is on). "
        "Optional refmods (H3_REF_MODS from Luisacaotica ComfyUI-MiniMaxH3Mod) injects saved "
        "RefMod references into every segment's conditioning before sampling. "
        "Defaults: 0.4MP 16:9 (864×480), 5s / 124 frames @ 24 fps."
2. nodes/director_common.py — Pass refmods to plan
Find: face_refine=None, in prepare_director_plan() signature.

Change to:

    face_refine=None,
    refmods=None,
):
Find: Both code paths where _attach_face_refine is called (external groups path at line ~283, and timeline path at line ~312).

Change each log.info( line that follows _attach_face_refine to include _attach_refmods:

External groups path:

        plan = _attach_face_refine(plan, face_refine)
        plan = _attach_refmods(plan, refmods)
        log.info(
Timeline path:

    plan = _attach_face_refine(plan, face_refine)
    plan = _attach_refmods(plan, refmods)
    log.info(plan_summary(plan).replace("\n", " | "))
Add helper function after _attach_face_refine:

def _attach_refmods(plan, refmods):
    plan.refmods = refmods
    return plan
3. director/plan.py — Add refmods field
Find:

    face_refine: dict | None = None
    # Sampling knobs stamped at execute time (first-pass cache fingerprint).
Change to:

    face_refine: dict | None = None
    refmods: list | None = None
    # Sampling knobs stamped at execute time (first-pass cache fingerprint).
4. director/executor_core.py — Inject RefMods into conditioning
Find: After apply_semantic_bridge() call block:

        positive, sb_note = apply_semantic_bridge(positive, plan, task_key=seg.task_key)
        if sb_note:
            log.info("MiniMax H3 Director: %s", sb_note)

        trim_frames = 0
Change to:

        positive, sb_note = apply_semantic_bridge(positive, plan, task_key=seg.task_key)
        if sb_note:
            log.info("MiniMax H3 Director: %s", sb_note)

        # ── RefMod injection (Luisacaotica ComfyUI-MiniMaxH3Mod) ──────────────
        # Inject saved RefMod latents into the conditioning's ref blocks so the
        # DiT attends to them alongside any native keyframes / r2v refs.
        if getattr(plan, "refmods", None):
            for mod, strength in plan.refmods:
                if strength <= 0.0:
                    continue
                block = mod.ref_block(strength=float(strength), curve=("constant", "linear", 1.0))
                if block is None:
                    continue
                block["refmod"] = True  # marker so Step Curve only re-mixes injected refs
                new_positive = []
                for cond_tensor, cond_dict in positive:
                    cond_dict = dict(cond_dict)
                    cond_dict.setdefault("refs", []).append(block)
                    cond_dict.setdefault("minimax_refs", []).append(block)
                    new_positive.append([cond_tensor, cond_dict])
                positive = new_positive
                log.info(
                    "MiniMax H3 Director: injected RefMod '%s' (kind=%s, strength=%.2f, "
                    "tokens=%d)", mod.name, mod.kind, float(strength), mod.token_count
                )
        # ──────────────────────────────────────────────────────────────────────

        trim_frames = 0


How to wire it
Load H3 RefMods (mods) ──→ MiniMaxH3Director.refmods
Requires Luisacaotica's ComfyUI-MiniMaxH3Mod pack installed alongside for the H3_REF_MODS type and Load H3 RefMods node.

Best with task_type=r2v + ref2va UNET. audio_vae required for r2v/v2v/rv2v.

Restart ComfyUI after applying
The node will appear as MiniMaxH3Director (same as before) with one extra socket: refmods.



### #270 [open] ISSUE: 【提个建议】不知道能不能跟MinimaxH3Mod这个主体制作插件联动
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/270  by Lazy-Rabbit-2001 2026-10-19T10:10:57Z

插件地址：https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod
简单地说，就是利用里面的一些工具制作safetensors主体，然后可以通过一个引用器直接引用这个主体，而且可以复用，能大大提高制作效率与精准度，之前有幸在b站上看到过相关的视频所以我就简单了解了一下这个插件，希望大大能够做一下联动（如果能跟对方作者取得联系的话）

另见：https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/271

> **Comment by ericsari (2026-10-23T13:39:01Z):**

希望能有这功能，内置或以CONDITION外接的方法都行，只要有，希望你能早日顺利完成AI导演大师，谢谢



### #269 [open] ISSUE: 添加参考音频后模型胡乱说话
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/269  by VDiva 2026-10-19T03:25:11Z

 提示词没有的东西模型会自己演艺出来 而且百分百复现 不知道是什么原因 去掉参考音频就好了

> **Comment by Lazy-Rabbit-2001 (2026-10-19T10:22:00Z):**

哥们提示词有按照官方doc给的要求标Speaker ID了么？(S1)、(S2)这种


> **Comment by VDiva (2026-10-19T15:01:01Z):**

> 1. 哥们提示词有按照官方doc给的要求标Speaker ID了么？（S1）、（S2）这种

对 我用官方的工作流好像也有问题 那就是官方模型有bug 也是百分百复现 纳闷了 然后我换了一个参考音频就好了 但是又出现了同一句话会复读几次


> **Comment by 17614275083 (2026-10-21T06:10:27Z):**

在生成长视频的时候，第二段参考音频与人物口型对不上


> **Comment by VDiva (2026-10-22T07:17:36Z):**

> 3\. 在生成长视频的时候，第二段参考音频与人物口型对不上

我也是只要加上参考音频就会有各种各样问题



### #268 [open] PR: fix: preserve external Director writes and baseline handoff
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/268  by qwert20020408-web 2026-10-16T13:09:19Z

## Problem
External writes to MiniMax H3 Director could roll back because the MCP used whole snapshots, stale derived mirrors, and a one-time baseline token without a continuation handoff. Prompt-batch partial writes also treated omitted fields as deletions.

## Changes
- Add a dependency-free three-way merge for Director state with PATCH semantics and stable-ID array merging.
- Preserve unsaved frontend fields, ignore editor-derived timing/workspace mirrors by mode, and report real editable mismatches with values and numeric deltas.
- Add explicit single-use baseline diagnostics and return a fresh post-write baseline for the next `timeline_data` or `global_prompt` write.
- Invalidate delayed `onConfigure` restores after a successful external write and keep the manual global-prompt release path.

## Validation
- JavaScript syntax check passed.
- Lab regression suite: 40 source tests passed; 26 isolated real-editor graph-query/graph-write cases passed.
- No generation was submitted during validation.



### #267 [open] PR: feat: 支持「现成视频当某一段」，并在空选择时直接拼接
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/267  by erbierbier 2026-10-15T19:06:10Z

## 加了什么

**一、`sourceVideo` —— 把一段已完成的视频「当」这一段的成品，不再重新生成。**

片段可声明 `"sourceVideo": {"videoFile": "seg_0002.mp4", "type": "input", "subfolder": "..."}`。
导演台在跑之前用它给该片段的分段缓存**播种**（解码 → 视频/音频 VAE 编码 → `save_segment_cache`）。
缓存就位后，`executor_core` 现成的「命中缓存」分支照常取该段像素与 AV latent —— **采样器一次都不触发**，
下一段照旧拿到续写种子。

**二、「选择运行」留空，现在表示「把已有的拼起来」。**

`runSelectEnabled` 开启而 `runSelection` 为空时，原来会抛「未勾选任何片段」；现在语义改为**一段都不采样、把成品接起来**（`_parse_run_selection` 返回空 `frozenset()`）。非空选择保持原行为。

## 实测

| 模式 | 耗时 |
|---|---|
| 全采样 | ~380 s |
| 拼接 | **30 s** |

> 详细说明见 **`docs/video-as-segment.md`**。


### #266 [open] ISSUE: 有计划在外接参数的节点支持外接公共参数吗
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/266  by AI-Amumu 2026-10-14T01:17:57Z

有计划在外接参数的节点支持外接公共参数吗？


### #265 [open] ISSUE: 希望增加全局音频平铺功能
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/265  by AI-wuji 2026-10-13T15:17:39Z

比如在做mv时，按照现在的做法，是按分镜，每个分镜内上传一段截取的音频。
然后所有分镜合并生成。但这里就有个问题，就是需要手动裁切音频，可能种种问题，会导致多分镜拼接时音频无法对齐。（比如这里会因为小数点的问题，裁切后合并可能对不齐）
所以，建议增加一个全局的音轨（带波纹显示，方便对齐分镜）
这样每个分镜设计，只需要按照波形卡点制作就可以了，不用再为裁切音频对不齐而烦恼了。
而且，由于是全局音频平铺，不参与视频生成，所以音质不会受损，非常有利于提高视频质量。


> **Comment by aswordok (2026-11-19T08:20:12Z):**

你要的这个一条音频平铺整条时间线的功能另一个导演台可以实现 TimelineDirector，但也没有音频波纹，只能摸黑。
另外还不能只生成其中一个片断，如果某个片断有问题，就只能整条重新生成。



### #264 [open] ISSUE: 选择静音导致最后视频合成报错
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/264  by co-re-co 2026-10-13T14:45:46Z

导演台中选择静音，但参考音频中存在引用时（忘记删除之前的参考音频），会导致最后生成的视频在合成过程报错。


### #263 [closed] PR: feat: 复用缓存续跑 + 修复首段一采缓存永不落盘 + prev加载diff诊断日志
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/263  by a437409255 2026-10-13T06:17:49Z


## Summary
- 工具栏「选择运行」旁新增「复用缓存续跑」开关（默认关，timeline.reuseCacheResume）：开启后 Queue 运行时，选中段若成片缓存指纹完全命中（segment_cache_fingerprint 精确一致）直接复用磁盘缓存不重采样，从首个缺失/改动的段继续生成（断点续跑）；勾选段间引导自动复用上一段尾帧衔接
- 开启开关时 seed 控件自动切 fixed（Queue 前 flushDirectors 兜底再套一次），避免每次 Queue 随机种子导致指纹永不命中；关闭恢复原控制模式
- 复用段水合 completed_*（成片帧/音频/AV/一采AV/low_carry/handoff），后续段引导钉与导出补丁行为与刚采样段一致；segment_outputs 与 run_list 顺序保持对齐，段间像素释放照常
- 修复首段一采缓存永不落盘：is_continuity_active 对 index 0 恒为 False（首段无前段可钉），「引导开+二采关」常规跑法下首段永远不满足保存条件 → 选段1二采每次都重新一采。保存条件补 is_continuity_lead_segment，引导开启时首段同样落盘 .pre 缓存
- load_segment_av_latent / load_first_pass_av_latent 拒绝加载时输出 diff 键与 stored->expected 明细，便于定位「上一段缓存无效」
- 单段重抽、二采、宫格等显式段范围场景不受影响

## Test plan
- [ ] node --check / py_compile 已通过
- [ ] 开启开关重跑已生成工程：命中段秒过不重采样，仅缺失段采样
- [ ] 关闭开关行为与原版一致（选中段全部重跑）
- [ ] 引导开启跑一次多段后，选段1二采可直接命中一采缓存


### #262 [open] ISSUE: 段间引导+引用上段时音频参考触发张量形状错误：RuntimeError: The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 3
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/262  by aksyoulates 2026-10-11T17:17:14Z


第一段正常，第二段开启段间引导并引用上段后报错。

日志大致如下：
[ERROR] !!! Exception during processing !!! The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 3
[ERROR] Traceback (most recent call last):
  ...
  File "...\ComfyUI_MiniMaxH3_Director\nodes\director.py", line 343, in execute
    execute_director_plan_core(...)
  File "...\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1695, in execute_director_plan_core
    chunk, audio_dict, pre_chunk, pre_face_chunk = _run_one_segment(...)
  File "...\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1216, in _run_one_segment
    samples, low_carry = sample_selflift_stage(...)
  File "...\ComfyUI_MiniMaxH3_Director\director\selflift\sample.py", line 559, in sample_selflift_stage
    high_out = sample_single_stage(...)
  File "...\ComfyUI_MiniMaxH3_Director\director\core_sampling.py", line 244, in sample_single_stage
    out = _run_official()
  File "...\ComfyUI_MiniMaxH3_Director\director\core_sampling.py", line 205, in _run_official
    sampled = SamplerCustomAdvanced.execute(...)
  ...
  File "...\ComfyUI_MiniMaxH3_Director\director\h3_latent_continue.py", line 482, in apply_model_wrapper
    return executor(*args, **kwargs)
  ...
  File "...\ComfyUI\comfy\ldm\minimax\model.py", line 637, in forward
    out[0] = out[0] * denoise_mask
RuntimeError: The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 3


### #261 [closed] PR: feat(examples): 新增 AMD/GGUF 本地适配工作流归档 example_workflows/AMD-7900xtx/
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/261  by happymy 2026-10-11T04:34:40Z



---

# 新增 AMD / GGUF 本地适配工作流归档：`example_workflows/AMD-7900xtx/`

## 摘要

在 `example_workflows/` 下新增**独立归档目录** `AMD-7900xtx/`，收录面向 **AMD Radeon（RX 7900 XTX 24GB / RDNA3 / gfx1100 / ROCm）** 的 GGUF 本地适配工作流 **10 个** + 一份说明文档。

**对现有文件零改动**：9 个官方工作流、`README.md`、`README_EN.md`、`example_workflows/README.md` 全部保持上游原样。本 PR **纯新增**。

## 变更范围

| 项 | 值 |
|---|---|
| base | `upstream/main` @ `b168de8` |
| head | fork 的 **`7900xtx`** 分支 @ `494f3e2`（`main` 保持纯净上游镜像，无本地提交） |
| 净 diff 提交数 | **6**（另有 1 个上游合并 `7854235` + 8 个历史提交） |
| 新增文件 | **11**（10 个工作流 + 1 个 README） |
| 行数（vs 上游） | **`+10,644 / −0`** |
| 修改 / 删除现有文件 | **0** |
| 总大小 | ~279 KB |

## 新增文件清单

| 文件 | 任务 | 扩散模型 | Sage | 额外依赖 |
|---|---|---|---|---|
| `minimax_h3_director_t2v_gguf.json` | t2v | fl2va | — | — |
| `minimax_h3_director_i2v_gguf.json` | i2v | fl2va | — | — |
| `minimax_h3_director_fl2v_gguf.json` | fl2v | fl2va | — | — |
| `minimax_h3_director_multiseg_t2v_gguf.json` | 多段 t2v | fl2va | — | — |
| `minimax_h3_director_r2v_gguf.json` | r2v | ref2va | — | — |
| `minimax_h3_director_v2v_gguf.json` | v2v | ref2va | — | — |
| `minimax_h3_director_rv2v_gguf.json` | rv2v | ref2va | — | — |
| `minimax_h3_director_加速版_gguf.json` | 加速版（243 帧） | ref2va | 旁路 ×2 | — |
| `minimax_h3_director_external_groups_i2v_gguf.json` | external i2v（734 帧） | fl2va | 旁路 ×2 | — |
| `minimax_h3_director_external_groups_r2v_gguf.json` | external r2v（1210 帧） | ref2va | 旁路 ×2 | ComfyUI-VideoHelperSuite + 自备参考视频 |
| `README.md` | 适配总说明：差异表 / 平台差异 / 依赖 / 模型路径 / **CK 警告** / **连续生成 WARNING** / **preview_watch 推荐** / **来源与致谢** | — | — | — |

命名规则 = 官方 `<原名>` + `_gguf.json`，逐一对应 `../` 下的同名官方文件。

## 与官方模板的差异

| 项 | 官方模板 | 本 PR（`*_gguf.json`） |
|---|---|---|
| UNET | `UNETLoader` + `*_int8_convrot.safetensors` | `UnetLoaderGGUF` + `*_Q4_K_M.gguf` |
| 文本编码器 | `CLIPLoader`（`qwen3vl_32b_minimax_h3_nvfp4_awq`，`type=minimax`） | `CCTechClipProjLoader`（`qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn` + `mmh3-8b-ClipProj-v3.1`，`type=boogu`） |
| SageAttention | 加速版 / external_groups_* 内含 Sage 节点 | 同位置节点**旁路**（`mode=4`） |
| VAE / Director / 连线 / 采样参数 | — | **原样保留** |

## 平台差异（NVIDIA vs AMD）

官方模板面向 NVIDIA；本适配面向 AMD RDNA3（gfx1100）。

| 能力 | NVIDIA | AMD RDNA3（本机） | 影响 |
|---|---|---|---|
| SageAttention（KJ 两节点） | 支持（SM 75/80/86/89/90/120/121），降自注意力峰值显存 + 提速 | **无实现**（无 gfx 分支） | 必须旁路（共 6 个节点） |
| FP8 运算（E4M3/E5M2） | Ada/Hopper/Blackwell 原生 | **不支持**（RDNA3 无 float8 计算，`SUPPORT_FP8_OPS=False`） | fp8 权重能加载但按 bf16 计算 → 更慢更占显存 |
| NVFP4 量化 | Blackwell/Ada 专有 | **不可用** | 官方 32B NVFP4 编码器 → 换本地 heretic-8B |
| INT8（`int8_convrot` / comfy_kitchen） | 支持 | **支持**（RDNA3 原生 int8） | `--use-ck-attention` 的 INT8 注意力可用 |
| FlashAttention | CUDA 支持 | **不支持** | 注意力无 flash 核，采样更慢 |
| 扩散权重格式 | `int8_convrot` / `w4a8_mixed` | GGUF Q4_K_M（dequant → bf16） | 全部换 `UnetLoaderGGUF` |

一句话：AMD 上**能跑**的是「GGUF 权重 + bf16/INT8 计算 + CK INT8 注意力」；**不能跑**的是「SageAttention」「FP8 计算」「NVFP4 编码器」「FlashAttention」。

## 本地模型依赖

| 用途 | 文件名 | 目录 |
|---|---|---|
| UNET (fl2va) | `minimax_h3_fl2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| UNET (ref2va) | `minimax_h3_ref2va_pruned-Q4_K_M.gguf` | `models/diffusion_models/` |
| 文本编码器 | `qwen3-vl-8b-heretic-1.3.0_fp8_e4m3fn.safetensors` | `models/text_encoders/` |
| ClipProj | `mmh3-8b-ClipProj-v3.1.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

另需 **ComfyUI-GGUF-Loader**（提供 `UnetLoaderGGUF` / `CCTechClipProjLoader`）；`external_groups_r2v` 另需 **ComfyUI-VideoHelperSuite**。


## 附：README 内的平台运维经验（本 PR 文档部分）

`AMD-7900xtx/README.md` 除工作流说明外，还固化了三条 AMD 平台实测经验（供后来者避坑）：

1. **CK 使用警告**（顶部 `> [!CAUTION]`）：`--use-ck-attention` 只能用于 H3 生视频，**生图（Qwen-Image 2.1 等）必须用无 CK 的启动脚本** —— 该 flag 会把扩散自注意力全局覆盖为 comfy_kitchen INT8，文本流跨过 64 token 即静默输出损坏（comfy-kitchen 0.2.36 回归，PyPI 无修复版）。
2. **连续生成性能下降警告**（顶部 `> [!WARNING]`）：`--enable-dynamic-vram` 的残留物理页随任务累积 → 同一条工作流越跑越慢；**实测非 H3 特有，生视频 / 生图 / TTS 生音频都会退化**。缓解 = 调 ComfyUI `/free`（段间由多段脚本内置，段外/批跑用 `plan/bat/free_vram`）；`/free` 是异步 flag 端点（HTTP 200 ≠ 已释放，须看 `/system_stats` 的 `vram_free` 上升），且救不了单任务内部变慢（只能重启 ComfyUI）。
3. **推荐的配套工具**：`plan/bat/preview_watch`（生成中实时帧预览内核补丁 + 监视脚本）—— 内核补丁在 `step 0` 首批预览时用 video VAE 解码并落盘真帧，其自带的 `free_memory` 会顺带把 DiT 提前卸载到低水位，**显著降低峰值显存**。
## 该内核补丁，理论上适配CUDA环境，因无硬件环境故未实机测试。脚本仅适配windows打开资源管理器，Linux环境需自行适配。

## 验证环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX 24GB |
| 架构 | RDNA3 / gfx1100 |
| 后端 | ROCm 7.2.1（hip 7.2.53211） |
| PyTorch | 2.9.1+rocm7.2.1 |
| ComfyUI | 0.39.2 |
| comfy-kitchen | 0.2.36 |

## 已知限制 / 注意事项

1. **SageAttention 在 AMD 无实现**（内核仅分发 NVIDIA SM 架构，`_cuda_archs` 取自 CUDA 专属接口）→ 全部旁路。失去其「降低自注意力**峰值显存**」的作用，长片请靠 `clear_vram_between_segments` + 减帧 / 降分辨率 / 减参考组防 OOM。
2. **`--use-ck-attention` 生图会损坏**：该 flag 会把扩散模型自注意力全局覆盖为 comfy_kitchen INT8，在此后端下 Qwen-Image 2.1 只要文本流跨过 **64 token** 即输出严重损坏且**不报错**（comfy-kitchen **0.2.36** 的回归，PyPI 无修复版本）。**生图必须用无 CK 的启动脚本**；H3 生视频（文本编码器 + DiT 主干）已双向确认安全，可继续用 CK。CK 是进程级全局开关，切换用途须重启 ComfyUI。
3. **连续生成会越跑越慢**（`--enable-dynamic-vram` 残留），实测生视频 / 生图 / TTS 生音频均有；段间/批跑请调 `/free`（见 `plan/bat/free_vram`），单任务内部变慢只能重启 ComfyUI。
4. `external_groups_r2v_gguf.json` 的 `VHS_LoadVideo` 指向参考视频，用户需换成自己的文件放入 `ComfyUI/input/`。
5. **`二采_加速` 未纳入**：其两阶段管线（一采 fl2va + 二采 ref2va + 3D latent 放大）采样峰值显存超出 24GB，且 AMD 无 Sage 降峰值，本机不可用。

## 提交列表

净 diff（`upstream/main..7900xtx`，即本 PR 实际带来的改动）：
```
733f2ff  chore(archive): 归档 AMD/GGUF 适配工作流到 example_workflows/AMD-7900xtx
42623ed  chore(archive): 移除 二采_加速 工作流（显存超出 24GB）
46aade4  docs(archive): AMD-7900xtx README 顶部新增 CK 使用警告（生图禁用）
9d5c3c6  docs(archive): AMD-7900xtx README 推荐配套工具 preview_watch（实时帧预览 + 提前卸载 DiT）
c1d173c  docs(archive): AMD-7900xtx README 顶部新增 WARNING —— 连续生成性能下降（生视频/生图/TTS 生音频通用）
494f3e2  docs(archive): AMD-7900xtx README 新增「来源与致谢」—— 部分数据/代码/脚本来自 happymy/MinimaxH3-7900xtx
```
同一分支上的附加上下文（非本 PR 净 diff）：
```
7854235  chore: 合并 upstream/main(b168de8)，同步上游 audio_io 采样率修复
844e9a6 → 763feaa  8 个 GGUF 适配历史提交（gguf 曾置于 example_workflows/ 根目录，本次归档重整后移入 AMD-7900xtx/）
```

## 对上游的影响

- **纯新增目录**，不改动任何现有文件（净 diff `+10,644 / −0`），与上游 `main` 无冲突（head 分支 `7900xtx` 已合并 `upstream/main` @ `b168de8`）。
- 若上游认为平台特定产物不宜合入主仓，可仅作为 fork 内容保留；本目录已自包含（含完整说明），不影响官方示例的可用性。

## 来源与致谢

本目录的**部分数据、代码与脚本**来自 [`happymy/MinimaxH3-7900xtx`](https://github.com/happymy/MinimaxH3-7900xtx)：`plan/bat/free_vram`（`/free` 显存清理）、`plan/bat/preview_watch`（生成中实时帧预览），以及连续生成性能下降、`step 0` 提前卸载 DiT 等实测数据与 AMD/ROCm 适配经验。

---

> 说明：本 PR 描述由本地 git 数据生成（`git diff --stat upstream/main 7900xtx` → `11 files changed, 10,644 insertions(+)`，0 删除）。



### #260 [open] ISSUE: 能不能加入Veda加速节点
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/260  by timery199 2026-10-11T01:52:14Z

这个节点大佬看一下，提取很快，能不能加入到工作流中？https://github.com/veda-sparse/Veda-on-ComfyUI.git


### #259 [closed] PR: 段缓存尾部裁剪：.av.pt/.pre.low.pt 只存续接尾部（省约50%磁盘）+ 合并 upstream
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/259  by a437409255 2026-10-10T04:30:48Z

## Summary

- 段缓存尾部裁剪：`.av.pt` 与 `.pre.low.pt` 落盘前只保留段间续接所需的尾部，磁盘占用约省 50%（272 帧样本实测 0.7MB→0.3MB）。
- 顺带合并 upstream/main（含缓存目录自定义命名、解码音频真实采样率等 5 个提交），并解决两套缓存键方案的冲突。

## 功能变更

- 新增 `crop_av_head` / `crop_av_tail_for_cache` / `av_head_dropped_px` / `align_crop_to_reference`（director/h3_motion_context.py）：
  - 裁剪量取 **15 步的倍数**（`drop = 15 * ((total_steps - 27) // 15)`）：`≡0 mod 5` 保住 `_video_tail_blocks` 的 5 步相位网格；`×3` 使音频裁剪落在整数个 17 帧/85 tick 周期上，`prev_end_frame` 重基可精确到帧。
  - 保留 ≥27 步尾部，覆盖 56 帧最大 pin 窗（17 步）加相位可行性余量，对任意总时长安全。
  - 裁剪标记写入 `continuity_prefix_dropped_px`；重复保存幂等（防 phase-align trim 重写路径二次裁剪）；切片强制 `.contiguous()`，确保 torch.save 真正只序列化保留区。
- 保存端（director/segment_cache.py）：`.av.pt` 与 `.pre.low.pt` 裁剪后落盘；**`.pre.av.pt` 不裁**——二采 refine 需要从完整一采采样。
- 加载端（director/executor_core.py）：读到带裁剪标记的上一段缓存时，`prev_end_frame` 按 `dropped_px` 重基；first-pass AV 与 SelfLift 低清 carry 经 `align_crop_to_reference` 对齐到裁剪后坐标系。视频 pin、音频 pin、低清 pin 三个消费点共享这一个坐标。

## 缓存键合并（upstream 冲突解决）

- upstream 新增「缓存文件夹名」widget（`<名字>_<节点id>`），本仓库此前已有 timeline projectId 键（`<projectId>`，缺省回退节点 id）。两者统一为组合键 `resolve_cache_key`：**`<缓存文件夹名>_<projectId>`，缺项降级，都空回退节点 id（老工作流行为不变）**。
- 节点 IS_CHANGED / executor / 面板状态路由 / 清缓存路由 / 段间引导 `resolve_prev_segment_output` 全部走同一组合键；`plan.project_id` 仍为项目部分（指纹、导出目录、合并过滤语义不变）。
- 前端 `cacheStatusPayload` / 清缓存请求同时携带 `cache_name` 与 `timeline_data`。

## 注意事项

- **旧缓存兼容**：无 `continuity_prefix_dropped_px` 标记 = 全量 = dropped 0，行为与改动前完全一致。
- **混版本防护**：全量引导 latent + 裁剪缓存混用时，`align_crop_to_reference` 兜底对齐，失败则原样返回并 log warning。
- `.av.pt`/`.pre.low.pt` 属一次性续接数据，不做跨版本合并；如需回退到旧版本插件，旧版本读裁剪文件会把坐标当全量用，建议回退时同时清空 `output/minimax_seg_cache/`。
- 实测（272 帧、T=80、drop=45）：视频 pin 在 n∈(5,22,39,56) 与全量逐位全等、音频 overhang 精确保留（0.0000 vs 0.0000）、`end_frame=None` 路径（T=82 样本）等价。

## Test plan

- [ ] 重启 ComfyUI 后跑 2 段以上带段间引导的 timeline，确认接缝无回归、报告出现「尾部裁剪版…已重基」提示
- [ ] 确认 `output/minimax_seg_cache/<key>/seg_XXXX.av.pt` 体积明显下降
- [ ] 开启二采跑一遍，确认 `.pre.av.pt` 全量、refine 正常
- [ ] 填「缓存文件夹名」+ 项目，确认目录为 `<名字>_<projectId>` 且面板/清缓存指向同一目录
- [ ] 旧缓存（改动前生成）直接续跑一遍，确认行为不变



### #258 [closed] PR: 修复原声和参考音频采样率被误标为 44100 Hz 导致变速、变调的问题
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/258  by RyouMon 2026-10-09T11:16:18Z

非 44.1 kHz 的原声或参考音频解码后，返回的采样率被错误标记为 44100 Hz，导致播放变速、变调，并影响原声按时间线截取的位置。例如，一段 48 kHz、2 秒的音频被误标为 44.1 kHz 后，会变成约 2.177 秒，音高也随之降低。

`_load_full_audio` 已经探测了采样率，并通过 `-ar` 明确指定 FFmpeg 的 PCM 输出采样率。但成功解码时，`-v error` 通常不会输出流信息；随后解析空的 stderr，会得到默认值 44100，覆盖前面确定的实际采样率。

本次最小修复移除这次错误覆盖，直接保留传给 FFmpeg 的采样率。无法探测采样率时，现有逻辑仍会让 FFmpeg 实际重采样到 44100 Hz，因此返回值与 PCM 保持一致。

新增回归测试，使用 FFmpeg 实际解码合成的 440 Hz 音频：覆盖 16、24、32、44.1、48 kHz 的单声道和立体声输入，以及缺少 ffprobe 时的回退，共 11 种配置。测试检查返回的采样率、采样数、时长和音高。修复前，8 种非 44.1 kHz 配置均失败；修复后，11 种配置全部通过。

验证命令（将 ComfyUI 根目录加入 `PYTHONPATH`，并确保 `PATH` 中有 FFmpeg 和 ffprobe）：

```sh
python -B -m unittest discover -s tests -p test_audio_io.py -v
```

Fixes #65（同一根因，已核对现有 PR，未发现重复修复）。

关联 #178：该 issue 描述了原声音画不同步的现象，可能还涉及其他原因，本次仅修复采样率误标问题。



### #257 [open] ISSUE: selflift单采，段间引导越来越模糊
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/257  by q508399 2026-10-09T08:28:37Z

selflift单采，如果开启段间引导，连接处会闪屏，如果开了很多段，每段都会模糊一次画面


> **Comment by triton9001 (2026-10-10T11:45:02Z):**

根据我的使用经验，“native_low_carry”设置为“true”应该能解决闪屏（插件说明有说），多段的“引导”改用“引导+重绘”，重绘幅度设置为0.2，可以改善下一段会模糊的问题



### #256 [open] ISSUE: Is Custom Sampler Input possible? (For use with MiniMax-H3 Turbo Sampler by Larryvrh)
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/256  by DarkNecrotic 2026-10-06T14:40:47Z

Please make it possible to input a custom sampler into the Director-node.
With the turbo lora from Larryvrh (https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo) you can use a custom turbo sampler node with a sampler output, but you can't connect it to your Director-node.

If there is a possibility to input it, i didn't find it.


### #255 [open] ISSUE: Director 插件部分输出少一帧，修复建议
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/255  by fengyunwuji728 2026-10-06T07:48:40Z

原因：-shortest 因音频采样舍入而提前结束导出，丢失末帧。
修改：移除该终止条件，保留原有音频对齐和视频裁剪逻辑。

ComfyUI_MiniMaxH3_Director/lib/video_export.py

cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
改成
cmd += ["-c:a", "aac", "-b:a", "192k"]


### #254 [open] PR: 支持 2X 视频 VAE（打包 PixelShuffle 解码头）
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/254  by tuanlai1234 2026-10-04T18:04:25Z

## 背景

社区发布的 2X 类 H3 视频 VAE（如 `MiniMax-H3-X2-Detail-v1.safetensors`）可以把手工搭建的图里的 H3 latent 解码成 2 倍分辨率。但导演台目前无法直接使用它：内部三处解码调用都是裸 `VAEDecode().decode()`。

## 问题

2X VAE 与原版 `minimax_h3_video_vae_fp16.safetensors` 的**唯一**结构差异是解码投影层：

| | 原版 | 2X 版 |
|---|---|---|
| `decoder.proj_out.weight` | `[3072, 2048]` | `[12288, 2048]` |
| 含义 | 3 通道 × patch 体积 1024 | 12 通道打包子像素（2×2 PixelShuffle） |

H3 的 patch 体积 = `patch_size_t(4) × patch_size²(16²) = 1024`，故 12288/1024 = 12 = 3×2×2。

这类 VAE 经标准 `Load VAE` 加载后，运行时 `proj_out` 权重是完整的 12288 行，而 `decoder.out_channels` 仍是构造时的 `3`。ComfyUI 的 `VAEDecode` 以 `out_channels` 计算输出缓冲区（`decode_output_shape`），因此会按 3 通道取不到打包输出。

## 改动

新增 `director/h3_x2_decode.py`，提供统一入口 `decode_video_latent()`。按 `proj_out.weight.shape[0]` 运行时判断，**不依赖文件名**（社区 release 命名混乱，且 checkpoint 的 `__metadata__` 里 `format` 字段并非所有发布都有）：

- `3072` → 原样 `return VAEDecode().decode(...)[0]`，普通 VAE 走完全未改动的原路径
- `12288` → 临时设 `out_channels=12`、把 `pixel_mean/std` 扩展到 12 通道、直调 H3 解码器，再 `PixelShuffle(2)` 还原

三个实现细节值得注意：

1. **`pixel_mean/std` 是 `persistent=False`**，动态 VRAM 加载随时可能还原成 3 通道版本，所以必须在 `load_models_gpu()` **之后**扩展，且用 `repeat_interleave(4, dim=1)` 而非整体重复 RGB 三元组（打包布局是 RRRR GGGG BBBB）。
2. **直调 `inner.decode()` 而非走 `vae.decode()`**，绕开 `comfy/sd.py` 里 OOM 后掉进通用 3D tiler 的回退路径 —— 那个 tiler 不认识 H3 的 blend 规则，接缝会呈方格网。
3. **探测失败一律安全回退**到原生路径并打 warning，不抛异常。

接入的三处：

| 文件 | 作用 |
|---|---|
| `director/executor_core.py` `_decode_av_latent` | 主生成路径（3 处调用共用） |
| `director/refine_sampling.py` `_decode_video` | Refine 中间帧解码 |
| `director/face_refine/runtime.py` `_decode_video` | 人脸精修每段解码 |


## 验证

对 `python_embeded` (torch 2.11.0+cu130) 实测：

- **真实 2X checkpoint 端到端**：latent `(1,24,2,32,32)` → **5 帧 1024×1024×3**，值域 [0,1]。`decode_output_shape` 为 `(1,12,5,512,512)`，PixelShuffle 后 1024
- **原生 VAE 零回归**：`decode_video_latent()` 与 `VAEDecode().decode()` 输出 `torch.equal() == True` **逐位一致**（均 5 帧 512×512×3）
- **异常安全**：模拟 decode 抛异常后，`tiling` / `tile_size` / `tile_overlap_min` / `out_channels` / `pixel_mean` / `pixel_std` / `proj_out` 权重与对象 id 全部回到调用前状态
- **探测矩阵**：3072→`(0,1)`、12288→`(12,2)`、0/1000/5000 行→安全回退+warning、非 H3 VAE / `None` → `(0,1)`
- **形状回归**：`(1,3,7,32,48)`、`(2,3,5,64,32)`、`(1,3,1,16,16)` → `(B*T,H,W,C)` 全部正确

最后一组测试在开发过程中抓出一个真实 bug：`_to_images` 初版把 H 和 W 都写成 `shape[-1]`，方形图无碍但非方形 latent 会 reshape 失败，已修正。

## 兼容性

- **节点界面、输入输出类型、工作流接线零变化** —— X2 VAE 走标准 `Load VAE` 接 `video_vae` 槽即可
- 无新增 pip 依赖（只用 torch + comfy 核心）
- `face_refine/runtime.py` 仍保持懒加载（`ultralytics` 是可选重依赖，不能顶层 import）
- `requirements.txt` / `pyproject.toml` 无需改动
- 已同步 `README.md` 与 `README_EN.md`

## 已知边界

- 本 PR 只覆盖 2X **解码**。同一 checkpoint 里还有一套 B32 参考图细节增强支路（`detail_b32.*`，42 个张量，从 `encoder.down.1.block.1` 取 tap），需 RGB 参考图才能用，**未包含在此 PR 中**
- 空间 tiling 未暴露为节点参数（导演台没有 LATENT 输出，外部无法接；且已有 segment 时间分段机制分摊峰值）。如需暴露建议加输入参数而非新增输出，以保持旧工作流兼容




### #253 [open] PR: Freeze source-audio tokens into the AV latent audio stream (audioMode=source) for tighter lipsync
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/253  by bananasss00 2026-10-04T11:38:40Z

## (English)

### What

When the timeline runs with `audioMode = source`, each segment's reference audio is now encoded with the audio VAE and written into the **audio stream of the segment's AV latent**, pinned with a per-stream `noise_mask` (video stream = denoise, audio stream = keep clean). This is the same trick stock ComfyUI uses for LTXV:

```
EncodeAudio -> LTXVFreezeLatent (noise_mask = 0) -> LTXVConcatAVLatent
```

— applied automatically inside the Director, per segment. The DiT then aligns the picture to the real song through joint AV attention, while the mux in `source` mode keeps using the original PCM audio.

### Why

With `audioMode = source`, lipsync driven by reference-audio conditioning alone ("ref lipsync") tends to be loose: the model regenerates the audio stream on every pass and the mouth drifts off-beat over time. Pinning the actual segment audio into the latent gives the joint AV attention a real target instead of a free-running audio stream.

### Implementation

New module `director/audio_freeze.py` plus minimal hooks:

- **`executor_core.py`** — after conditioning + continuity pinning, right before sampling: `apply_first_pass_audio_freeze(...)` fills the audio window at the continuity-adjusted offset and installs the keep mask. No-op on first-pass cache hits (marker already present).
- **`core_sampling.py`** — at the top of `sample_single_stage`: when the freeze marker is present, `refresh_frozen_audio_mask(latent)` rebuilds the mask and restores the frozen content (refine AV re-joins, continue locks and SelfLift repacks legitimately drop or rewrite the per-stream mask).
- **`refine_sampling.py`** — carries the marker across the refine AV re-join.
- **`face_refine/runtime.py`** — freezes the same segment audio for the face-crop re-pass (its audio stream feeds the joint AV attention, so the refined mouth stays on beat).
- **`segment_cache.py`** — the first-pass fingerprint gains `audio_freeze` (version flag) and `audio_freeze_timeline` (identity of the timeline clips behind frozen slices: path/size/mtime), so toggling the feature or swapping a clip invalidates caches cleanly.

The freeze state lives in an extra latent-dict key `h3_audio_freeze_mark` (offset, covered ticks, PCM signature, version). It survives the segment cache (latents are pickled), refine re-joins and SelfLift carries, so re-sampling a cached latent refreshes the mask **without re-encoding** the audio.

One detail worth calling out (v2 semantics): the SelfLift/refine resume math re-expresses the whole AV latent in the sigma-resume representation (`inverse_noise_scaling` = `x / (1 - sigma)`), which **rewrites the frozen audio window**. The encoded window is therefore cached by PCM signature at apply time and written back before every sampling stage whenever it drifted; otherwise comfy's mask machinery would preserve the mangled audio as the clean x0, and the high-res passes would condition on noise-dominated audio.

Audio source priority mirrors the `source`-mode mux (`prepare_segment_audio_for_file_export`): timeline song slice → reference audio cards → ref-video soundtracks, so the frozen tokens always match the muxed track.

### Guarantees / safety

- `audioMode != source`: behavior identical to stock — every entry point checks `resolve_audio_mode` first.
- Opt-out: `MINIMAX_H3_AUDIO_FREEZE=0` disables the feature entirely (and flips the fingerprint flag, so caches invalidate cleanly).
- Every hook is defensive: unexpected latent shapes, missing audio, VAE failures → logged and skipped, never raises.
- The installed mask composes with existing masks (e.g. a continuity prefix lock): audio part multiplied, video part kept as-is.
- The tail after the reference audio ends stays model-generated (official Concat AV semantics).
- Versioned marker + fingerprint (`FREEZE_VERSION`) so any future behavior change invalidates old caches automatically.

### Validation

- r2v MV workflow, `audioMode = source`: mouth stays on beat through the full timeline (previously drifted mid-segment).
- `audioMode = generate / mute`: output identical to stock with the feature present.
- Repeated runs with the same seed/settings: pixel-identical results at both the low-res and the high-res (refine) stages.
- Toggling `MINIMAX_H3_AUDIO_FREEZE` / replacing a timeline clip → segment cache invalidates and re-samples as expected.

---

## (中文)

### 做了什么

当时间线使用 `audioMode = source` 时，把每个片段的参考音频用音频 VAE 编码，写入该片段 AV latent 的**音频流**，并用 per-stream `noise_mask` 固定（视频流正常去噪，音频流保持干净）。等价于 ComfyUI 官方 LTXV 的做法：

```
EncodeAudio -> LTXVFreezeLatent (noise_mask = 0) -> LTXVConcatAVLatent
```

— 在 Director 内部按片段自动完成。DiT 通过 joint AV attention 让画面与真实歌曲对齐，mux 在 source 模式下仍使用原始 PCM 音频。

### 为什么

source 模式下仅靠参考音频 conditioning 的 "ref lipsync" 通常偏松：模型每一轮都会重新生成音频流，口型随时间漂移。把真实片段音频钉进 latent，给 joint AV attention 一个真实目标，而不是自由生成的音频流。

### 实现

新增 `director/audio_freeze.py`，外加少量挂钩：

- **`executor_core.py`** — conditioning 与连续性 pinning 之后、采样之前调用 `apply_first_pass_audio_freeze(...)`：按连续性偏移填充音频窗口并安装 keep mask；首过缓存命中（标记已存在）时直接跳过。
- **`core_sampling.py`** — `sample_single_stage` 顶部：存在冻结标记时调用 `refresh_frozen_audio_mask(latent)` 重建掩码并恢复冻结内容（refine AV re-join、continue 锁定、SelfLift repack 都会合法地丢弃/重写 per-stream 掩码）。
- **`refine_sampling.py`** — 跨 refine AV re-join 携带标记。
- **`face_refine/runtime.py`** — 面部裁剪重采样时冻结同一段音频（其音频流参与 joint AV attention，保证精修后的口型仍在节拍上）。
- **`segment_cache.py`** — 首过指纹新增 `audio_freeze`（版本标志）与 `audio_freeze_timeline`（被冻结切片背后时间线剪辑的 identity：路径/大小/mtime），切换开关或替换剪辑都会干净地使缓存失效。

冻结状态存放在额外的 latent-dict 键 `h3_audio_freeze_mark`（offset、covered ticks、PCM 签名、版本）中。它随片段缓存（latent 会被 pickle）、refine re-join 与 SelfLift carry 一起存活，因此对缓存 latent 重新采样时只需刷新掩码，**无需重新编码**音频。

一个值得说明的细节（v2 语义）：SelfLift/refine 的续算会把整个 AV latent 重新表达为 sigma-resume 表示（`inverse_noise_scaling` = `x / (1 - sigma)`），这会**改写被冻结的音频窗口**。因此编码后的窗口在应用时按 PCM 签名缓存，并在每个采样阶段前检测漂移后写回；否则 comfy 的 mask 机制会把被改写的音频当作干净 x0 保留，高分辨率阶段将条件在噪声主导的音频上。

音频来源优先级与 source 模式 mux（`prepare_segment_audio_for_file_export`）一致：时间线歌曲切片 → 参考音频卡片 → 参考视频音轨，保证冻结的 token 始终与混音轨道匹配。

### 保证 / 安全性

- `audioMode != source`：行为与原版一致——所有入口先检查 `resolve_audio_mode`。
- 可选关闭：`MINIMAX_H3_AUDIO_FREEZE=0` 完全禁用（同时翻转指纹标志，缓存干净失效）。
- 所有挂钩均为防御式：latent 形状异常、缺少音频、VAE 失败 → 记录日志并跳过，绝不抛异常。
- 安装的掩码与既有掩码可叠加（如 continuity 前缀锁定）：音频部分相乘，视频部分保持不变。
- 参考音频结束后的尾部仍由模型生成（官方 Concat AV 语义）。
- 标记与指纹带版本号（`FREEZE_VERSION`），未来任何行为变化都会自动失效旧缓存。

### 验证

- r2v MV 工作流、`audioMode = source`：整条时间线口型稳定贴合节拍（此前会随时间漂移）。
- `audioMode = generate / mute`：功能存在时输出与原版一致。
- 相同 seed/设置重复运行：低分辨率与高分辨率（refine）阶段输出逐像素一致。
- 切换 `MINIMAX_H3_AUDIO_FREEZE` 或替换时间线剪辑 → 片段缓存按预期失效并重采样。


### #252 [open] ISSUE: BUG：SelfLift采样开启分块+开启段间引导时会发生空间分块采样与 denoise_mask 形状不匹配异常（Spatial tiles × denoise_mask cond）
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/252  by triton9001 2026-10-04T08:35:29Z

# Bug: 空间分块采样与 denoise_mask 形状不匹配（Spatial tiles × denoise_mask cond）

- **仓库**: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
- **版本**: `1.0.0`（`pyproject.toml`），本地 checkout 提交 `a8f57b8e23c46ee28fdb96796510fa7b4f3b1fbb`
- **文件**: `director/spatial_tiled_sampling.py`
- **严重程度**: 崩溃（采样中断；ComfyUI 随后进入不可恢复的 CUDA 错误状态）
- **复现率**: 100%（条件见下）

---

## 1. 摘要

当**空间分块采样**（`n_tiles >= 2`）与**任意 denoise mask**（分段续接 `Director continue` / prefix keep-mask / inpaint）**同时生效**时，分块的模型前向会用**整帧尺寸**的 mask 去乘**分块尺寸**的潜变量，必然抛出：

```
RuntimeError: The size of tensor a (50) must match the size of tensor b (66) at non-singleton dimension 4
  File ".../ComfyUI/comfy/ldm/minimax/model.py", line 593, in forward
    out[0] = out[0] * denoise_mask
```

随后 ComfyUI 在 `execution.py` 的 `reset_cast_buffers() → torch.cuda.synchronize()` 处把它放大成：

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**后者是连带/滞后错误，不是根因**，排查时不要走 CUDA 方向。

原因：`MiniMaxH3.extra_conds()` 在采样开始前把 mask 一次性烘焙进 cond；分块代码只切了模型输入 `x`，没有切这个 cond。

---

## 2. 环境

| 项目 | 值 |
|---|---|
| ComfyUI | `v0.37.2`（node 要求 `>=0.30.0`） |
| PyTorch | `2.9.1+cu130` |
| Python | 3.13.11 |
| GPU | NVIDIA GeForce RTX 4060, 8585216000 B VRAM, `cudaMallocAsync` |
| 启动参数 | `main.py --auto-launch --preview-method none --use-sage-attention --cuda-malloc` |
| 工作流 | MiniMaxH3Director「批量生成 (r2v)」3 段 / 372 帧 @24fps / 1056×800 / SelfLift scale=0.70 highres_steps=2 / Segment continuity ON（continue motion context 22f） |

---

## 3. 最小复现条件

1. 使用 `MiniMaxH3Director`，**≥ 2 段**，并开启 **Segment continuity**（续接会写入 denoise mask / prefix keep-mask）。
2. 让长边进入**空间分块**：本例 1056×800 → 潜变量 `H×W = 50×66`，`n_tiles=2`：
   ```
   spatial tiles: axis=W size=66 n=2 overlap=128px/8 latent ranges=[(0, 49), (25, 66)]
   ```
3. 观察：**第 1 段**（无 mask）分块正常完成；**第 2 段**（continue / keep-mask）在**第一个 tile** 上崩溃。

等价的抽象最小条件：**任意一次携带 `denoise_mask` cond 的分块模型评估**。

---

## 4. 日志证据

```
13:25:14  spatial tiles: axis=W size=66 n=2 overlap=128px/8 latent ranges=[(0, 49), (25, 66)]   ← 第 1 段，无 mask，成功
13:28:23  MiniMax H3 Director segment 1/3 done (124 frames, task=r2v)
13:28:50  Director continue: wrote 7 video tokens (latent, 22f) + 40 audio ticks ... no cond-pin
13:32:52  SelfLift HQ resume: native inpaint keep-mask, prefix=7 tokens, σ=0.720733
13:32:52  Director continue remask: prefix=7 seam_min=0.00 ...                                   ← 第 2 段开始带 mask
13:32:53  spatial tiles: axis=W size=66 n=2 overlap=128px/8 latent ranges=[(0, 49), (25, 66)]   ← 完全相同的分块
13:33:48  Comfy model compiler graph breaks: 9, rogues: 0
13:33:48  comfy/ldm/minimax/model.py:593  out[0] = out[0] * denoise_mask
          RuntimeError: The size of tensor a (50) must match the size of tensor b (66) at non-singleton dimension 4
13:33:48  execution.py:553  comfy.model_management.reset_cast_buffers() → torch.cuda.synchronize()
          torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**数字完全对得上**：

- 第一个 tile 是 `[0, 49)` → 49 个潜列，经 `_pad_even_hw()` 用 replicate 补 1 列后为 **50**；
- 整帧 mask 的宽度是 **66**；
- `dimension 4` 正是 5D `(B, C, T, H, W)` 的 `W`。

两次分块的范围、分辨率、模型完全相同，唯一的差别就是第 2 段多了 denoise mask —— 这排除了分辨率、显存、分块算法本身。

---

## 5. 根因

1. **mask 被一次性烘焙进 cond（整帧尺寸）**
   `comfy/model_base.py:2206-2251`：`MiniMaxH3.extra_conds()` 调用 `_denoise_mask_conds() → _denoise_mask_values()`，
   ```python
   out['denoise_mask'] = masks[0][:1, :1].clone()      # (1, 1, T, H, W)，H/W = 整帧潜空间
   ```
   其中 H/W 由**整帧** `latent_shapes` 经 `_pool_masks_to_token_grid()`（`model_base.py:2220-2233`）得到（池化到 2×2 patch 后再 `repeat_interleave` 回原分辨率）。
   该 cond 由 `comfy/samplers.py:1038-1048 process_conds()` 在 **`inner_sample()` 阶段一次性生成**（`samplers.py:1226`），此后每个采样步、每个 tile 都复用它。
   另注：`model_base.py:2244` 只在 `torch.amin(masks[0]) < 1 - 1e-3`（存在被保留的位置）时才生成该 cond —— 这正好解释了「第 1 段没事、第 2 段才炸」。

2. **分块只切了模型输入**
   `director/spatial_tiled_sampling.py:_tiled_model_call()`（442-481 行）每个 tile 会：
   - 切 `x`（443 行）、切**位置参数** `denoise_mask`（445-446 行）；
   - 调 `_install_tile_payloads()` 重建 `latent_shapes` / `minimax_payload` / `layout` / `keyframes` / `refs`。

3. **但 cond 里的 mask 没有被重建** —— `_install_tile_payloads()`（269-310 行）唯独漏了 `denoise_mask`。

4. **模型直接用该 cond 逐元素相乘** —— `comfy/ldm/minimax/model.py:592-593`。于是 `out[0] (W=50) * denoise_mask (W=66)` → 崩溃。

即：位置参数那条路径已经正确切块（`KSamplerX0Inpaint.__call__`，`samplers.py:630-643`），**只有 cond 那条路径漏了**，属于「分块只做了一半」。

---

## 6. 建议修复

15 行新增，零新函数、零 API 变化：

```diff
@@ -277,9 +277,24 @@
     for model_conds in _iter_payloads(cfg_guider):
         shape_cond = model_conds.get("latent_shapes")
         if shape_cond is not None and hasattr(shape_cond, "cond"):
             restorations.append((shape_cond, "cond", shape_cond.cond))
             shape_cond.cond = tile_shapes
 
+        # MiniMaxH3 bakes the denoise mask into the conds once per sampling run, at
+        # full-canvas size (model_base.MiniMaxH3.extra_conds -> _denoise_mask_conds).
+        # Tiles crop the model input but not the conds, so a tiled forward would
+        # multiply a tile-sized latent by a full-canvas mask:
+        #   minimax/model.py:  out[0] = out[0] * denoise_mask
+        #   RuntimeError: The size of tensor a (50) must match the size of tensor b (66)
+        mask_cond = model_conds.get("denoise_mask")
+        if mask_cond is not None and hasattr(mask_cond, "cond"):
+            cropped_mask, cropped_ok = _crop_generate_video(
+                mask_cond.cond, full_h, full_w, axis, start, end
+            )
+            if cropped_ok:
+                restorations.append((mask_cond, "cond", mask_cond.cond))
+                mask_cond.cond = cropped_mask
+
         payload_cond = model_conds.get("minimax_payload")
```

## 补充说明
1. 本 issue 的日志分析、根因定位、补丁代码与文字整理，由 AI （DeepSeek Harness，模型 deepseek-flash）协助完成；实机运行与最终验证由本人执行；
2. 事后查看历史记录发现2周前的 issue #219 也提出过类似的问题，测试的是Refine节点，但提也到了SelfLift走相同逻辑。该issue处理的核心思路与这里的一致，但修改方式有差别，最终以维护者的判断为准；
3. SelfLift节点接入时，完成一轮任务后，下一轮任务开始采样时，控制台会出现内存泄漏警告：“WARNING, memory leak with model MiniMaxH3.”（不勾选任何清理显存选项），任务可正常继续进行，希望维护者也能进行排查。

> **Comment by xb191799981 (2026-10-05T14:58:31Z):**

这个是开启 “引导+重绘 ”的报错吗？  我也用到了SelfLift采样，  再加上“引导+重绘”后就一直报错。

[INFO] Comfy model compiler graph breaks: 6, rogues: 0
[ERROR] !!! Exception during processing !!! The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 4
[ERROR] Traceback (most recent call last):
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\nodes\director.py", line 335, in execute
    execute_director_plan_core(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^
        plan,
        ^^^^^
    ...<16 lines>...
        export_pre_face_refine=export_pre_face_refine,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1689, in execute_director_plan_core
    chunk, audio_dict, pre_chunk, pre_face_chunk = _run_one_segment(
                                                   ~~~~~~~~~~~~~~~~^
        seg, progress_index=progress_pos[seg.index]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1210, in _run_one_segment
    samples, low_carry = sample_selflift_stage(
                         ~~~~~~~~~~~~~~~~~~~~~^
        model=seg_model,
        ^^^^^^^^^^^^^^^^
    ...<22 lines>...
        canvas_height=ctx_h,
        ^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\selflift\sample.py", line 559, in sample_selflift_stage
    high_out = sample_single_stage(
        model=high_model,
    ...<21 lines>...
        noise_override=high_noise,
    )
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\core_sampling.py", line 244, in sample_single_stage
    out = _run_official()
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\core_sampling.py", line 205, in _run_official
    sampled = SamplerCustomAdvanced.execute(
        noise_obj, guider, sampler_obj, sigma_t, latent
    )
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\core_sampling.py", line 240, in sample_wrapped
    return orig_sample(noise, latent_image, sampler, sigmas_in, **kwargs)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\spatial_tiled_sampling.py", line 575, in wrapped
    return orig_fn(
        proxy,
    ...<5 lines>...
        **kwargs,
    )
  File "F:\AI\ComfyUI-aki-v3-B\python\Lib\site-packages\torch\utils\_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\k_diffusion\sampling.py", line 205, in sample_euler
    denoised = model(x, sigma_hat * s_in, **extra_args)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\spatial_tiled_sampling.py", line 542, in __call__
    return _tiled_model_call(
        inner,
    ...<7 lines>...
        **call_kw,
    )
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\spatial_tiled_sampling.py", line 476, in _tiled_model_call
    pred = _invoke_model(model_k, tile_x, sigma, tile_mask, call_kw)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\spatial_tiled_sampling.py", line 369, in _invoke_model
    return type(model_k).__call__(model_k, x, sigma, denoise_mask, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\model_base.py", line 210, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\h3_latent_continue.py", line 482, in apply_model_wrapper
    return executor(*args, **kwargs)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\model_base.py", line 254, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "F:\AI\ComfyUI-aki-v3-B\python\Lib\site-packages\torch\nn\modules\module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "F:\AI\ComfyUI-aki-v3-B\python\Lib\site-packages\torch\nn\modules\module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "F:\AI\ComfyUI-aki-v3-B\ComfyUI\comfy\ldm\minimax\model.py", line 594, in forward
    out[0] = out[0] * denoise_mask
             ~~~~~~~^~~~~~~~~~~~~~
RuntimeError: The size of tensor a (54) must match the size of tensor b (76) at non-singleton dimension 4


> **Comment by xb191799981 (2026-10-05T15:06:58Z):**

把SelfLift关闭后， 果然能正常使用“引导+重绘”功能了。


> **Comment by q508399 (2026-10-09T08:29:43Z):**

selflift单采，如果开启段接引导，连接处会闪屏，如果开了很多段，每段都会模糊一次画面



> **Comment by xb191799981 (2026-10-10T04:35:07Z):**

还有会爆显存。 我是4090-24G，每次执行到第6个分组时就必爆显存。  关掉SelfLift就能跑26个分组也不会爆显存。


> **Comment by ZCLCHANGLONG (2026-10-13T03:39:38Z):**

## 【开启段间引导，引用上段，保持引导加重绘的方式。 开启2采空间分块，会导致画面从第二段开始变糊。我通过DeepSeek已解决】下面是结论

`spatial tiles >= 2` 与「段间引导 / 任意 keep-mask」同时开启时，**第 2 段起二采（refine）整段失败**，输出静默退回一采——用户看到的现象就是「第 2 段起画糊了」。

根因共 **4 项：1 项致命 + 3 项降质**。上游已有 `#219`、`PR #206`、`#252` 三份报告，**定位都对，但只覆盖 4 项里的第 1 项**；只修那一项，报错会消失、画面却几乎没改善。

本补丁把 4 项一次修完，并加了 fail-open 兜底。**只改 `director/spatial_tiled_sampling.py` 一个文件**，真机 3 段全部跑通。

---

## 一、现象与最小复现

**配置**：`3 段 r2v`，一采 960×544 → 二采 1664×960，`spatial tiles 2 overlap 128px`，`Segment continuity: ON (continue motion context 22f)`。

**日志**（`ComfyUI/user/comfyui.log`）：

```
[10:02:40] Refine: ON (upscale ...1664×960), spatial tiles 2 overlap 128px
           Segment continuity: ON (continue motion context 22f, keep full)
[10:05:48] Director continue remask: prefix=7 seam_min=0.10 ...
[10:06:20] Segment 2 refine failed
           (The size of tensor a (68) must match the size of tensor b (104)
            at non-singleton dimension 4); keeping last successful pass.
[10:08:01] Segment 3 refine failed（同一错误）; keeping last successful pass.
```

**落点**（traceback 末帧）：

```
File ".../comfy/ldm/minimax/model.py", line 637, in forward
    out[0] = out[0] * denoise_mask
RuntimeError: The size of tensor a (68) must match the size of tensor b (104)
              at non-singleton dimension 4
```

`68` = 分块后单块宽度，`104` = 整幅宽度，`dimension 4` 就是 5D `(B, C, T, H, W)` 的 `W`。

三个要点：

- **第 1 段正常，第 2/3 段才崩** —— 段间引导只从第 2 段起生效。
- 画面**不是**被分块「画糊」的，而是**二采整个没跑完**：`keeping last successful pass` 表示输出退回一采（960×544 的观感）。**这才是「糊」的真正来源。**
- 等价的最小条件：**任意一次携带 `denoise_mask` cond 的分块前向**，即 `n_tiles >= 2` 且 段间引导 / prefix keep-mask / inpaint 任一同时生效。

---

## 二、根因：4 项，只有第 1 项致命

| # | 缺陷 | 性质 | 用户可感 |
|---|---|---|---|
| **1** | 段间引导 continue 模式装的前缀重遮罩钩子（`director/h3_latent_continue.py` 的 `_PrefixRemask`）持**整帧画布形状**（104），分块只有半幅（68）→ `out[0] * denoise_mask` 抛 `RuntimeError`，异常穿透采样器 → **refine pass 整段死亡**，输出静默退回一采 | **致命** | 第 2 段起明显变糊 |
| 2 | H3 的 RoPE 是**面积归一化**的（`_axis_from_sqrt_area`）：`ratio = dim / sqrt_area`，`sqrt_area = sqrt(h*w)`，相邻 patch 步长 = `32 / sqrt(h*w)`。按**块的** h/w 重建 layout，会让块内每一行的 (h, w) 坐标被重新缩放，实测偏差达整宽的 **30~45%**；`text / refs / audio` 行也跟着错位 | 降质 | 分块段的构图/运动与整幅不一致 |
| 3 | 切块边界按 `stride = round(axis_size / n)` 取，遇到奇数（106 / 2 = 53）会**错开半个 patch**（H3 是 2×2 patchify），整块网格相位不对 | 降质 | 结构错位 |
| 4 | 重叠区直接 `(A + B) / 2`，会把两块各自的高频互相抵消（低频一致、高频被平均），实测高频保留率只有 **77%** | 取舍 | 重叠带变软 |

### 一份前向里其实有两份 mask

| # | mask | 谁在消费 | 原代码 |
|---|---|---|---|
| A | 位置参数 `denoise_mask` | `KSamplerX0Inpaint.__call__` 做 `x` / `out` 的账 | 已按 tile 裁（对的） |
| B | **cond 自带的** `denoise_mask` 条目 | `MiniMaxH3.extra_conds` 生成，最终在 `comfy/ldm/minimax/model.py:637` 被 `out[0] = out[0] * denoise_mask` 消费 | **漏了** |

条目 B 由 `MiniMaxH3._denoise_mask_values()` 按 `torch.amin(mask) < 1 - 1e-3` 决定是否生成：

- **第 1 段**：没有重遮罩、掩码全 1 → **不生成条目** → 分块不撞车。
- **第 2/3 段**：前缀权重 < 1 → **条目出现** → `68 宽潜变量 × 104 宽遮罩` → 崩。

### 为什么「只修第 1 项」不够

- 只修 2 / 3 / 4：第 1 项还在，refine 照样死，观感几乎不变。
- 只修 1：报错消失，但画面差异肉眼看不出来。
- **四项必须一起修。**

---

## 三、与已有报告的关系

| 已有报告 | 结论是否正确 | 对应本补丁的 |
|---|---|---|
| `#219`（one1zz） | ✅ 正确：`guider.__call__` 只转发 `model_options`，会丢掉位置参数形式的 `denoise_mask`；真正生效的是 `_PrefixRemask.apply_model_wrapper` 里那份 | 根因 1 的另一半 |
| `PR #206`（yangala） | ✅ 正确：cond 侧 `denoise_mask` 未随 tile 裁剪 | 根因 1 |
| `#252`（triton9001） | ✅ 正确：mask 在 `extra_conds` 里一次性按整帧烘焙，之后被每个 tile 复用 | 根因 1 |
| **本补丁** | 根因 1（两种机制都覆盖）+ **identity 去重守卫** + 根因 2 / 3 / 4 + fail-open 兜底 | **全部** |

- 本补丁的基线 blob 与 `PR #206` **完全相同**（`454b489`），可以直接对比差异。
- 本补丁**不触碰 `h3_latent_continue.py`**（`#219` 的方案需要改 2 个文件）。

---

## 四、做法

| 缺陷 | 修法 | 新增函数 |
|---|---|---|
| 1（前半） | tile 前向期间把 `_PrefixRemask.video_shape` **临时指向该块形状**，`finally` 里立刻还原（目标与 `#219` 一致，但不改 `h3_latent_continue.py`） | `_remask_targets` / `_retarget_remask_shapes` / `_restore_remask_shapes` |
| 1（后半） | 把 cond 侧的 `denoise_mask` 一并裁到窗口（`#206` / `#252` 的思路），并加 **identity 去重守卫** | `_crop_cond_mask` |
| 1（兜底） | tile 前向包 try/except：异常 → warning + 完整 traceback（仅首次）→ **该步退回整幅前向**，refine 不再死亡 | `_mark_tiles_failed` |
| 2 | 块内 video/cond 行坐标改写为**整幅网格在该窗口的真实值**；`text` / `ref_audio` / `audio` 行从原始 layout **逐位复制**，保住时间重定位 | `_full_frame_grid` / `_window_grid` / `_retarget_tile_coords` |
| 3 | 切块边界统一**向下**吸附到偶数 latent（向上会吃掉重叠区） | `_snap_even` / `_snap_ranges` |
| 4 | 融合改双频带：`out = blur(P) + (S - blur(S))`；`P` = 宽窗共识（无接缝），`S` = 单位分解窄窗选择（`Σsᵢ ≡ 1`，在共享边界交叉淡出）。重叠带高频保留 **77% → 97.8%** | `_selection_windows` / `_blur_along_axis` / `_axis_slice` / `_hf_rms` |

### identity 去重守卫（`#206` / `#252` 未覆盖）

`positive` 与 `negative` 可能引用**同一个** `model_conds` 容器，`_iter_payloads` 会遍历到它两次 → 同一份 mask 被裁两次，并且 `latent_shapes` 的**还原目标被写成块形状**，整帧基准永久丢失。

```python
seen: set[int] = set()
for model_conds in _iter_payloads(cfg_guider):
    if id(model_conds) in seen:
        continue                        # 同一容器只处理一次
    seen.add(id(model_conds))
    ...
```

`_crop_cond_mask` 兼容两种布局：**3D packed**（用*原始* `latent_shapes` 解包 → 裁 → 重打包）与 **5D**（直接沿窗口轴切，音频流不参与空间切分）。

### 一个必须知道的机制点

`MiniMaxH3.extra_conds()` 由 `comfy/samplers.py` 的 `process_conds` 调用，**每段只跑一次**，结果写回 `guider.conds[...]["model_conds"]` 并被每个采样步、每个 tile 复用。

**所以在分块期间就地改写 `model_conds` 是有效的**（不是每步重算）。这也解释了 `#219` 里「在 guider 代理层改 mask 无效」的原因——那一层的修改不会进入模型前向。

### fail-open：最坏情况不再毁掉整段

| | 修复前 | 修复后 |
|---|---|---|
| tile 前向异常 | 异常穿透采样器 → refine pass 死亡 → 输出退回一采 | warning + traceback（仅首次），**该步退回整幅前向**，refine 正常完成 |
| 用户可感影响 | 整段变糊 | 只是「分块白开了、慢一点」 |

---

## 五、验证

### 5.1 离线单测 66 项 PASS / 0 FAIL

| 脚本 | 项数 | 覆盖 |
|---|---|---|
| `verify_tile_fix.py` | 25 | 坐标重定向（逐元素精确断言）、融合保真、端到端逐位一致 |
| `verify_tile_diag.py` | 4 | 诊断判据能否区分「拼接吃细节」与「两块画了不同的图」 |
| `verify_tile_remask.py` | 12 | 形状冲突**复现**、修复后真正跑通、兜底生效 |
| `verify_tile_condmask.py` | 25 | cond 侧遮罩裁剪 A/B/C/D + **去重守卫（Part E）** |

关键断言：

```
[PASS] 块内 video 行坐标 == 整幅网格子集（逐元素）        最大偏差 0.00e+00
[PASS] 所有切块边界为偶数（对齐 2x2 patch 网格）
[PASS] 两块预测一致时 hf 无损还原原场                    最大偏差 1.19e-07
[PASS] hf 拼接在重叠带的高频保留显著优于 legacy 平均      97.8% > 77.0%（+20.9pt）
[PASS] 整幅前向 == 分块前向（可分离模型下应严格成立）      最大偏差 7.15e-07
[PASS] bug 可复现：抛出「tile 宽 vs 整幅宽」的尺寸冲突
[PASS] 分块前向失败时不再让 refine 整体死亡（无异常穿透）
[PASS] 用完后重遮罩形状已还原为整幅
[PASS] 同一容器被两组 cond 引用时，改动记录只有一份        restorations=2（未去重会是 4）
```

### 5.2 真机实测：3 段全部跑通

配置同第一节（`3 segments / 304 frames @ 24fps`）。

| 段 | 帧 | 分块 | 3 步的 `rel` | `keep` |
|---|---|---|---|---|
| 1 | 107f / 4s | n=2 | 17.3% → 12.1% → **6.4%** | 1.03 |
| 2 | 124f / 5s | n=2 | 15.9% → 11.8% → **6.8%** | 1.02 |
| 3 | 73f / 3s | n=2 | 17.3% → 13.7% → **8.4%** | 1.02 |

- 3 段 × 3 步 × 2 块 = **18 次分块前向全部成功**；运行时**零** `spatial tile forward failed`、**零** `Segment N refine failed`
- `rel` = 两块在重叠区的不一致度，随去噪步收敛到 6~8% → 两块最终画同一张图，**接缝目视不可见**
- `keep` ≥ 1.02 → 高频没被融合吃掉，**画面不糊**
- 本补丁生效的标志行（仅带段间引导的段出现）：
  `spatial tile mask: cond denoise_mask (1, 1, 47, 60, 104) -> (1, 1, 47, 60, 68)`

> 日志里 `canvas mismatch (56, 36) != (60,104); tensor left uncropped` 出现 3 次属**正常**：那是 3 张参考图各自的画布，参考图本来就不该按生成画布裁剪。

---

## 六、应用

```bash
cd ComfyUI_MiniMaxH3_Director
git apply /path/to/spatial_tile_fix.patch
```

若 main 已前移：`git apply --3way spatial_tile_fix.patch`。不想打补丁，直接用包里的 `spatial_tiled_sampling.py` 覆盖同名文件即可（该文件就是真机验证通过的版本）。

| 指标 | 值 |
|---|---|
| 改动文件 | `director/spatial_tiled_sampling.py`（**仅此一个**） |
| 基线 | `8151b29` |
| 规模 | 590 行 → 1331 行（+746 / −5） |
| 新增函数 | 21 个，删除 0 个 |
| 默认行为 | `n_tiles <= 1` 时完全不生效，未开分块的用户零影响 |
| 只回退融合 | `DIRECTOR_TILE_BLEND=legacy`（**不回退**坐标与遮罩修复） |
| 完全回滚 | 用包内 `spatial_tiled_sampling.orig.py` 覆盖回去 |

### 两点提醒

- 修好后 `tile_overlap` 从「拿清晰度换接缝的旋钮」变成**纯低频协调参数**：**128 就是最佳值**，不需要再为接缝调小它。
- 若最后一步 `rel` 仍 > 15%，说明两块**确实在画不兼容的图**（每块只看得到自己那半幅）。**任何融合算法都救不回来**，此时应改用「按计算切分」的省显存方式（不改变模型看到的画面，因此无损）：`MiniMax H3 Low VRAM` / `MiniMax H3 Low VRAM Attention` / `MiniMax H3 Mem Eff Sage Attention`。

---

## 七、附件

| 文件 | 说明 |
|---|---|
| `README.md` | 完整文档（说明 → 做法 → 代码），含逐项原理、关键代码与提交信息草稿 |
| `spatial_tile_fix.patch` | 完整补丁，基于 `8151b29`，`git apply` 可直接用 |
| `spatial_tiled_sampling.py` | 修复后完整文件（drop-in，真机验证版） |
| `spatial_tiled_sampling.orig.py` | 改动前原文件（对照用） |
| `验证结果.txt` | 66 项单测明细 + 真机实测数据 |
| `真机日志_关键行.txt` | 真机运行原始日志摘录（含修复前失败 / 修复后通过对照） |
| `verify_tile_*.py` | 5 个单测脚本，支持 `COMFY_ROOT` 环境变量，可自行复现 |

---

## 八、AI 使用声明

本补丁的日志分析、根因定位、代码与文档整理由 AI [编程助手协助完成；实机运行、数据复核与最终确认由本人执行。所有结论都对应可复查的日志与断言；若与维护者判断不一致，请以维护者结论为准。[](url)](

[spatial_tile_fix_给作者.zip](https://github.com/user-attachments/files/33364342/spatial_tile_fix_.zip)

[README.md](https://github.com/user-attachments/files/33364351/README.md)

)


> **Comment by xb191799981 (2026-11-17T02:24:25Z):**

同根因，已提 PR #284（含回归测试）：https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/284

补两条实测细节，给后面碰这个点的人少走弯路：

1. **裁掩码必须用与瓦片 latent 同一套偶数补齐。** 本例瓦片是 `86 → 0:59`，`_pad_even_hw` 会把它补到 60；如果只按 59 去裁 mask，报错会变成 `a(60) vs b(59)`，看起来"修了但没修好"。PR #284 复用同一个 `_pad_even_hw`，并在测试里断言两块瓦片宽度是 `[60, 52]`。
2. **真正的触发条件是「空间分块 + 任意 denoise_mask」，不是 SelfLift 本身。** 只走「引导」不进分块，整幅 × 整幅永远对得上——所以关掉 SelfLift 就正常，很容易误判成 SelfLift 的 bug。

关于已存在的 PR #206（yangala，约两个月前）：它改的是同一处，思路一致（用 `_crop_generate_video` 裁切并补偶数）。#284 与它只有两点差别：把裁剪放在 `minimax_payload` 缺失时也会执行的位置，以及附了 `tests/test_spatial_tiles_mask.py`（未打补丁时能复现本 issue 完全相同的 `60/86` 报错，打上后 2 例通过）。**两个 PR 合任意一个都能修掉本 issue**；如果作者更倾向 #206 那种最小化改动，我可以关掉 #284，或把测试单独挂到 #206 上。

需要说清楚的是：#284 只修「形状不匹配导致的崩溃」这一项，不涉及楼上提到的「第 2 段起画面变糊」等降质问题——那些需要另外处理。



### #251 [closed] PR: feat: 段缓存目录支持自定义命名，避免多工作流共用节点 id 互相覆盖
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/251  by ljxdn 2026-10-04T07:47:50Z

## 背景

分段缓存的目录名此前只取导演台主节点的 `unique_id`，落到
`output/minimax_seg_cache/<node_id>/`。

但节点 id 只在**单张图内**唯一 —— 前端按「当前图最大 id + 1」分配，每张工作流都从 1 开始。
所以多张工作流里的导演台很容易拿到同一个 id，共用同一个缓存目录：互相覆盖，甚至被对方
剧本的 `prune_segment_cache()` 当成垃圾段直接删掉。用户在磁盘上也没法按名字找到某个
工作流的缓存。

## 做法

- 导演台主节点新增可选控件 **`cache_name`（「缓存文件夹名」）**，落在主面板「性能」组，
  紧跟「缓存存储方式」，默认空。
- **留空 = 完全沿用旧行为**（按节点 id 建目录）—— 旧工作流与不关心这个功能的用户零影响。
- 填了就落到 `output/minimax_seg_cache/<名字>/`。名字按**单段路径**净化：过滤
  `/ \ : * ? " < > |` 与控制字符、避开 Windows 保留名（`CON`/`NUL`/`COM1`…）、超长截断；
  白名单外的输入回退成节点 id。
- **进度上报继续用真实节点 id**（前端是按画布节点 id 认进度条的），只有缓存 key 换成解析后
  的名字；这两处语义原来共用一个变量，现在在 `executor_core` 入口拆成
  `progress_node_id` / `node_id`，否则进度条会全线失效。
- HTTP 侧 `first_pass_cache_status` / `clear_segment_cache` 同步接收 `cache_name`，
  与执行层共用同一个 `resolve_segment_cache_key()` —— 否则「查缓存」和「写缓存」会落到
  不同目录（原来的 `\d+` 强校验也一并放宽）。
- Refine 面板查缓存状态 / 清空缓存时把该名字带上。

## 影响面

- 新控件**追加在性能组末尾**，不改动已有控件顺序。`widgets_values` 是位置契约，插在中间会
  让旧工作流取值整体错位。
- `cache_name` 留空时行为与更新前一致；缓存目录仍是 `<node_id>`。
- 本 PR **独立于 #225**（段缓存身份槽寻址），不依赖它，也没有包含它的改动。

## 验证

- 留空 → 回退节点 id（129 / 5155 实测）；填名 → 目录隔离，多工作流互不干扰。
- 路径穿越 `../../etc/passwd` → `etc_passwd`；`..\..\win` → `win`；`CON` → `_CON`；
  `a/b:c*d?…` → `a_b_c_d_e_f_g_h_i`；纯空白 → 回退节点 id。
- 改动的 5 个 `.py` 走 `py_compile`、3 个 `.js` 走 `node --check`，全过。



### #250 [closed] PR: feat: 出片预览与导出统一取源（预览即产物），预览面板支持下载整片
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/250  by JiangXinYu113 2026-10-03T14:52:18Z

## 这个 PR 在解决什么

全部导出在长视频时，会导致内存超出上限，从而失败。因此想要一个既能完整导出、又不会导致内存超出的方案。

新增点：
1. 页面预览：视频进度条与提示词条目进度对应，更直观能看见完整视频和对应提示词的关系，方便调整。
2. 在成品预览右上角增加下载功能，提供另一个将成片完整取出且不会爆内存的途径。（内部实现：分段落盘+ffmpeg拼接整段）

出片预览（页面里播的那份整片）和产物导出，原先各走一条取源逻辑，于是**「页面看到的」和「导出的」对不上**——这是用户实际踩到的两个症状：


1. 勾了「选择运行」只跑 1 段时，预览里仍有整条时间轴，导出的却只有那一段；
3. 反过来，播放列表被关在「分段导出」模式里，「全部导出」时前端收不到列表，预览退回旧快照降级路。

本 PR 把两侧收成**一个取源入口**，并给预览面板补上「下载」。

## 改动

**1. 取源单入口 ——「预览 = 产物」从约定变成结构上成立**

新增 `resolve_preview_entries()`，页面预览与产物导出走**同一函数、同一优先级**：

```
① node_id 的落盘快照（须工作流指纹一致）
② 按 plan 重建
③ 按前端下发的段形状展开
④ 任务表兜底
```

原来两处各写一份取源逻辑，跨运行条目、占位片参数、缺失段判定都能各自漂移，最后拼出两个「内容差不多但不相同」的文件。现在只有一个入口，没法再分叉。

**2. 工作流身份指纹 `shape_key`（修多工作流撞车）**

多个工作流各放一个导演节点时，节点 id 都可能正好是 `5`——快照按 node id 存，就会互相顶掉。现在写快照时一并记下内容指纹：

```python
shape_key = sha256(f"t={归一任务键}\nshape={i:frames|i:frames}")[:16]
```

读回时对不上就不认领（宁可重新展开，也不用来源不明的快照）。前端把当前段形状下发（`shape=0:124,1:124,…`），后端据此算指纹。

**3. 播放列表在任何导出模式下都构造**

原来只在「分段导出」模式下构造：单段运行时 `run_dir` 只有 1 段，前端最后只收到 `kind="segment"` 事件，进度条塌缩成一块。现在无论跑了几个段都下发完整时间轴条目。

**4. 缺音轨的段补静音音轨后再 concat**

MiniMax H3 原生 32 kHz、占位片 44.1 kHz，concat demuxer 在这种输入下会把音轨时基算错——实测被拉成 **1.378 倍**（成片 121.6s，视频只有 88.2s），`-ar` / `aresample` 都救不回来。现在先给缺音轨的片段补一条静音轨（视频 `-c:v copy`），再走滤镜拼接。原状态下整片直接拼不出来。

**5. `recovered_*/` 目录清理**

原先只增不减。现在目录数超上限才触发回收，且**跳过仍被 manifest / 预览快照引用的片段**，只删真正的孤儿。

**6. 出片预览：不自动播放 + 面板整理 + 下载按钮**

- 不再自动播放（点播放键才播），`_loadPreview` 用 `wasPlaying` 守卫续播；
- 移除「只算跑过的段 / 联动导演台」两个开关。**这两个开关出厂都是默认开**，所以行为固化到各自的默认分支，而不是删掉功能（代码注释保留原开关名，便于追溯「为何没有开关却有固定行为」）；
- 新增「下载」按钮：另存的就是页面正在播的那一份文件（同源 URL + `<a download>`；本版 ComfyUI 的 `/view` 不支持 `download` 参数），文件名取「工作流名_日期时间.mp4」。

**7. 导出方式回退到上游两档（全部导出 / 分段导出）**

原第三档「分段合并」产出的 `merged.mp4`，和出片预览文件是**同一批 `timeline_entries` 拼出的逐字节相同的一个文件**：

```
output/minimax_seg_export/<ts>/merged.mp4         7,100,714  7954bec35a5b4002839adb56da425789
output/minimax_preview_cache/preview_<key>.mp4    7,100,714  7954bec35a5b4002839adb56da425789
```

且全仓无任何代码消费它（路由返回的 `merged_file` 字段前端并未读取）。属重复产物，已取消；整片统一由出片预览提供。

> **兼容处理（关键）**：历史值 `exportMode="merge"` 在前端 `normalizeExportMode()` 与后端 `_resolve_export_mode()` 两侧都继续归一为 `"segments"`。这一步不能省——直接不认会掉进 `"all"`，**静默切成整条时间轴常驻内存的导出模式**，而分段导出正是为规避它而存在的。已保存的工作流已就地迁移（`pos` + `named` 两份 widget 值），前端 6 个读写点全部经过归一函数。

## 改动文件

| 文件 | 变化 | 说明 |
| --- | --- | --- |
| `director/segment_mp4_export.py` | +1654 | 分段 mp4 落盘、时间轴播放列表、整片拼接（`_concat_clips_to`）、预览文件按内容指纹缓存与回收、取源单入口 `resolve_preview_entries()` |
| `web/js/minimax_director_video.js` | +1354（**新文件**） | 出片预览面板：`<video>` 播放、与时间轴逐像素对齐的提示词进度条、播放头联动、下载 |
| `director/http_routes.py` | +388 | `/preview_file`、`/preview_plan`、段形状指纹解析、段落缓存清理等路由 |
| `director/progress.py` | +76 | 播放列表 / 单段事件通道（推给前端内嵌预览） |
| `director/executor_core.py` | +78 −1 | 播放列表在任何导出模式下都构造 + 写快照时带工作流指纹 |
| `web/js/minimax_timeline.js` | +29 −7 | 导出方式归一、预览后缀文案 |
| `director/plan.py` | +12 −4 | `export_mode` 归一（含历史 `merge` 兼容别名） |

`nodes/` 下未做任何改动（输出口的数量与顺序与上游一致）。

## 兼容性

- 节点输出口的**数量与顺序未变**（内部那些未接线就直通的占位输出槽本来就没实现，与本次无关）；
- 未开启分段导出的工作流行为不变；
- `nodes/` 下未做任何改动。

## 测试

全部为本地实跑（离线脚本，隔离 node_id，不写生产快照）：

| 脚本 | 结果 |
| --- | --- |
| `verify_export_mode_migration.py` | 34/34 — 兼容别名 + 旧代码/文案摘除 + 工作流迁移 |
| `verify_export_mode_frontend.js` | 18/18 — 归一函数行为 + 6 个读写点 + 旧写法绝迹 |
| `verify_preview_download.js` | 35/35 — 开关移除 + 下载按钮 + 另存文件名 |
| `verify_no_autoplay.js` | 12/12 |
| `verify_shape_key.py` | 24/24 — 取源单入口 + 工作流指纹 |
| `verify_preview_file_route.py` | 11/11 |
| `verify_preview_plan_route.py` | 8/8 |
| `verify_shape_parity.py` | 前后端 16 段逐段一致，总帧 2001 |
| `verify_audio_fixup.py` / `verify_recovered_prune.py` / `regress_export_paths.py` | 全 PASS |
| 全量 `py_compile` + `node --check` | 通过 |

## 与其它 open PR 的关系

- **#241「新增延迟合并成片、项目管理与合并进度条（含内存水位保护）」** 主题最接近。本 PR 已把「合并成片」那一档去掉，剩下的是**取源统一**与**预览侧交付**，两者可互补：
  - #241 若落地，合并产物可以作为 `resolve_preview_entries()` 的另一个来源接进来，取源入口不用改；
- **#197「大时间轴改为磁盘流式拼接」**、**#150「选择导出」**也相邻。本 PR 的 `resolve_preview_entries()` 是单一漏斗，这些改动接进来只需在入口加一个分支。

如果维护者倾向先合 #241，本 PR 可以只保留第 1–6 项（去掉第 7 项），告诉我即可。



### #249 [open] ISSUE: 【Bug】⭐⭐修复了 group combine bug⭐⭐
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/249  by zibuyu2007 2026-10-02T02:04:04Z

[director_groups.py](https://github.com/user-attachments/files/32940214/director_groups.py)

[minimax_timeline.js](https://github.com/user-attachments/files/33040452/minimax_timeline.js)



## 📝 Issue 内容:Groups Combine 节点：分组超过 10 个时端口顺序错乱，且断开连线后连线跳转到错误端口
## 问题描述

在 **MiniMax H3 Director Groups Combine** 节点上，当连接的分组数量 **超过 10 个** 时出现两个问题：

**1. 端口顺序错乱**

输入端口不再按数字升序排列，而是退化成字符串比较：

```
group_0, group_1, group_12, group_13, group_2, group_3, ...
```

10 个以内时正常，超过 10 个后排序明显错乱。

**2. 连线跳转到错误端口**

断开中间某个连线后，后面的连线会自动补位，但补位过程中有连线连到了错误的端口上，
或者连线直接消失。删除分组后保存工作流再打开，连线关系已经损坏。

### 复现步骤

1. 新建 12 个以上 Group 节点，分别连到 Combine 节点
2. 观察输入端口顺序 → 出现 `0, 1, 12, 13, 2, 3...` 的错乱
3. 随机断开中间某一条连线
4. 观察后续连线 → 出现错位、跳端口或丢失

### 根因分析

**问题 1：排序在 10+ 时失效**

后端 `director_groups.py` 原先的槽位索引提取逻辑，在匹配失败时回退到 `10**9`。
多个匹配失败的槽位得到**相同的排序键**，导致它们之间的相对顺序取决于字典遍历顺序而非分组编号，
因此排序结果不稳定。

前端 `expandExternalGroupLink` 中的排序逻辑有相同问题：两个槽位只要有一个无法提取索引，
比较函数就直接 `return 0`，等于放弃排序。

**问题 2：ComfyUI autogrow 移位时遗漏 target_slot**

ComfyUI 内置的 autogrow 机制在断开连线后会执行"移位补位"，
把后续分组的连线往前搬，并更新 `link.target_slot` 指向新的物理槽位。
这个过程中存在遗漏，导致移位后 `target_slot` 与实际物理槽位不一致 ——
这就是连线跳到错误端口的直接原因。

### 修复内容

#### `nodes/director_groups.py`（后端，+14 −6）

- 新增正则 `_SLOT_INDEX_RE`，统一提取分组序号，
  兼容 `group_0` / `groups.group_12` / `some.prefix.group_99` 等命名格式
- 原 `_slot_index()` 改为 `_slot_sort_key()`，返回 **(优先级, 四位补零序号)** 作为排序键：
  - 成功提取序号 → `(0, "0012")`
  - 未知命名 → `(1, 原始名)`，统一排在末尾
- 零填充保证字典序 == 数字序，彻底消除同键导致的排序不稳定

#### `web/js/minimax_timeline.js`（前端，+56 −6）

- `combineGroupSlotIndex()` 正则补充 `\b` 边界，匹配更稳健
- 新增 `combineGroupSortKey()`，**与后端使用完全相同的排序语义**，保证前后端顺序一致
- `expandExternalGroupLink()` 改用新的排序键，替换原先会 `return 0` 的不稳定比较
- 新增 `fixCombineNodeTargetSlots()`：在 Combine 节点输入连线变化后，
  遍历所有已连接的输入，逐条校验 `link.target_slot` 与物理槽位是否一致，
  不一致则修正并标脏画布触发重绘
- 在 `beforeRegisterNodeDef` 的 `onConnectionsChange` 钩子中，
  对 Combine 节点的输入连线变化调用上述校正

### 验证情况

- `python -m py_compile nodes/director_groups.py` 通过
- `node --check web/js/minimax_timeline.js` 通过
- 乱序输入的 12 个分组，在后端与前端均稳定排序为 `0,1,2,...,11,12`
- 前后端排序键一致，UI 显示顺序与实际执行顺序完全对齐

### 附加文件

以下两个文件为修复后的完整文件（已通过语法检查）：

- `nodes/director_groups.py`
- `web/js/minimax_timeline.js`

改动统计：`2 files changed, 70 insertions(+), 12 deletions(-)`





### #248 [open] ISSUE: 希望加入分段controlnet功能
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/248  by yxxyun 2026-10-01T04:16:42Z

现有Rv2V在低显存的显卡上很慢，把视频做成白模视频，当作controlnet源就快了很多，现在导演台不支持分段controlnet ，希望能加入。


### #247 [open] ISSUE: [Bug] SelfLift 的 3D lift 失败会静默降级为 bilinear 插值，与 Refine 的硬失败行为不一致
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/247  by tuanlai1234 2026-09-30T08:03:27Z

## 摘要

`SelfLift` 路径把 `upscale_h3_video_latent` 包在 `except Exception` 里，3D 权重加载失败时会**静默退回逐帧双线性放大**并继续渲染；而 `Refine` 路径调用同一个函数是裸调用、会直接抛异常。同一处故障，两个节点行为不一致，且 SelfLift 侧完全无声。

## 代码位置

- `director/selflift/lift.py:48-63` — try/except + fallback

```python
try:
    packed = upscale_h3_video_latent(...)
    out = packed["samples"]
except Exception as exc:
    log.warning("SelfLift 3D lift failed (%s); falling back to %s interpolate.", exc, fallback)
    out = interpolate_video_5d(work, dst_lat_h, dst_lat_w, fallback)
```

- `director/selflift/sample.py:267` → `upsample = str(pack.get("latent_upsample") or "bilinear")`
- `director/selflift/sample.py:407-418` → `lift_video_tensor(..., fallback=upsample)`
- `director/selflift/cond.py:18-38` → `interpolate_video_5d`，拆成逐帧 2D `F.interpolate`，时间轴不动
- `director/refine_sampling.py:648` → 裸调用 `upscale_h3_video_latent`，**没有 try**

## 复现

```python
from ComfyUI_MiniMaxH3_Director.director.selflift import lift as liftmod
from ComfyUI_MiniMaxH3_Director.director import h3_latent_upscale as up
from ComfyUI_MiniMaxH3_Director.director.selflift import cond

z = torch.randn(1, 24, 6, 32, 32)          # 512px 的 H3 latent
BOGUS = "definitely_missing_model.safetensors"

# 1) Refine 的路径：硬失败
up.upscale_h3_video_latent({"samples": z}, target_width=1024, target_height=1024,
                           source_width=512, source_height=512, model_name=BOGUS)

# 2) SelfLift 的路径：静默通过
out = liftmod.lift_video_tensor(z, src_width=512, src_height=512,
                                dst_width=1024, dst_height=1024,
                                model_name=BOGUS, fallback="bilinear")
```

实测输出（ComfyUI 便携版 Python 3.13，本地实测）：

```
--- 1) direct call to upscale_h3_video_latent (the path Refine uses) ---
   raised FileNotFoundError: H3 latent upscaler weights not found: definitely_missing_model.safetensors ...

--- 2) same failure through lift_video_tensor (the path SelfLift uses) ---
   returned normally, shape (1, 24, 6, 64, 64)   <- no exception raised
   spatial grew 32x32 -> 64x64

--- 3) is the fallback output just a bilinear resize? ---
   identical to interpolate_video_5d(..., 'bilinear'): True

--- captured log lines ---
   [WARNING] SelfLift 3D lift failed (H3 latent upscaler weights not found: ...); falling back to bilinear interpolate.
```

即：降级结果与 `interpolate_video_5d(z, 64, 64, "bilinear")` **逐位相同**，没有任何学习式成分。

## 为什么这是个问题

1. **降级不是"不执行 upscale"。** 它把学习式 3D lift 换成逐帧双线性放大（也就是 SelfLift 论文里明确当反面基线的 nearest/bilinear 路线），仍然是一次真实的放大运算。latent 分辨率照常到位、流程照常跑完，所以从外部看不出任何异常。

2. **触发面比想象中大。** `except Exception` 会吞掉：权重架构不匹配（`load_state_dict(strict=True)` 抛的 `RuntimeError`）、权重文件损坏、加载期 OOM、以及 lift 路径里将来任何代码 bug。用户尝试命名/来源较新的权重时最容易踩到——我就是在验证 `h3_upscaler_lms_v0.1_fp32.safetensors`（见 #245）时顺带发现的。

3. **症状难归因。** 控制台只多一行 WARNING，紧接着还会正常打印 `SelfLift transition: video ... → ... (target 1024x1024)`，渲染"成功"结束。输出只是偏软、丢高频，很容易被归因到提示词或采样参数上。

4. **与 Refine 不一致。** 同一份权重加载失败，Refine 会立刻报错暴露问题，SelfLift 无声通过。两个入口走的是同一个 `upscale_h3_video_latent`，行为不该差这么多。

## 建议

不预设实现方式，几个方向供参考：

- **收窄捕获范围**：只兜真正可恢复的异常（例如 `torch.cuda.OutOfMemoryError`），其余照常抛出；
- **保留降级但要"响亮"**：通过 `progress` / 节点面板上报，而不是只写 `log.warning`，让用户明确知道这一版没有走学习式 lift；
- **至少对两类情况硬失败**：`model_name` 指向的文件不存在，以及 `load_state_dict(strict=True)` 失败。这两类都说明配置错了，不是运行时资源问题，降级只会掩盖它。

## 环境

- ComfyUI portable（Windows 11），`python_embeded` Python 3.13
- 仓库 revision `main @ 63834b9`
- 与 GPU 无关，纯逻辑路径，CPU 上即可复现



### #246 [open] ISSUE: ref模型提示词编写建议
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/246  by swx966130 2026-09-30T07:33:59Z

不太知晓前端js的编写，不知道这样能不能实现
需求触发点是我平时编写的时候会出现一直复制粘贴的情况，就是主体标签 Subject N
目前素材导入后，使用@可用选择到传入的图片很方便，但是对于标签化的描述中，这里只涵盖了Picture N、 Video N和Audio N三种，Subject N标签没有代入，我猜想原因是编写提示词前，前端也不知道到底有没有这个标签和到底有多少这个标签
我的想法是能不能在素材添加的区域上，提供一下下拉选项，选择本次到底会有多少个Subject N标签，后续编写提示词的时候就可以直接使用@出Subject N标签
(Sx) 声音标签同理也能不能带上呢？或者传入音频后，不仅仅能激活出Audio N标签，这个(Sx)标签也能够激活呢？
感谢


### #245 [closed] PR: latent 放大权重列表不再用 "3d" 子串误筛同类新权重
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/245  by tuanlai1234 2026-09-30T07:18:54Z

## 问题

`ComfyUI/models/latent_upscale_models/` 里凡是文件名不含 `3d` 的 H3 权重，都不会出现在 **SelfLift** 与 **Refine** 节点的 `latent_upscale_model` 下拉框里 —— 但它们的架构其实完全兼容，只能靠改名才能选中。

## 复现

目录：

```
models/latent_upscale_models/
├── minimax_h3_latent_upscaler_3d_fp16.safetensors        (690 MB)
├── h3_upscaler_lms_v0.1_fp32.safetensors                 (1.38 GB)
└── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

下拉框实际只列出第一项。

`h3_upscaler_lms_v0.1_fp32.safetensors` 的 safetensors 头里明确写着 `format = minimax_h3_latent_upscaler_3d`，key 结构与 `LatentResizer3D` 逐项对齐（`conv_in` / `conv_out` / `embed` / `norm_out`，`in_blocks` 与 `out_blocks` 各含 12 个 `ResBlockEmb3D` + 6 个 `TemporalConv`），`_detect_arch()` 也能正确推断出 `in_blocks=12 / out_blocks=12 / temporal_kernel`，加载本身没有问题 —— 只是它进不了候选列表。

## 原因

`director/h3_latent_upscale.py` 的 `list_h3_latent_upscale_models()`：

```python
out = sorted({n for n in names if n and not n.startswith("(")})
preferred = [n for n in out if "3d" in n.lower()]
return preferred or out or [MISSING_MODEL_LABEL]
```

只要目录里存在**任意一个**名字带 `3d` 的权重，`preferred` 就非空，其余权重被整体丢弃。这等于把"文件名里得有个 `3d`"变成了事实上的隐式约定。

`nodes/director_refine.py` 的 tooltip 也把这层约定固化成了文档（"文件名含 3d"），所以从使用者角度看这是设计意图而非 bug —— 但它拦掉的恰恰是格式标记为同一家族的权重。

## 修复

按 H3 权重家族的关键词匹配代替单一子串，并用"是否显式含 `3d`"作为排序键，保证显式 3D 权重仍排第一、节点默认值不变：

```python
H3_LATENT_UPSCALER_HINTS = ("3d", "h3", "minimax", "lms")
```

- 关键词刻意**不含**泛化的 `upscaler`，避免把同目录下的 LTX 等非同族 latent upscaler 也拉进首选列表。
- 若目录里一个家族关键词都匹配不到，仍回退为列出全部候选，与改动前行为一致。
- 顺带把 Refine 的 tooltip 与 SelfLift 的报错文案从"文件名含 3d"改成"H3 系权重"，不再继续误导。

改动只涉及下拉框候选集的生成，不触碰加载与推理路径。

## 验证

用便携版 Python 直接实跑改动后的函数（把 `folder_paths` 打桩指向真实 `models/`）：

```
folder listing:
   h3_upscaler_lms_v0.1_fp32.safetensors
   ltx-2.3-spatial-upscaler-x2-1.1.safetensors
   minimax_h3_latent_upscaler_3d_fp16.safetensors
   put_latent_upscale_models_here

dropdown result:
  [0] minimax_h3_latent_upscaler_3d_fp16.safetensors   <-- default
  [1] h3_upscaler_lms_v0.1_fp32.safetensors
```

新权重已出现；LTX 仍被排除；默认值仍是原来的 3D 权重。既有工作流不受影响 —— widget 里存的是文件名字符串，候选列表顺序变化不会改动已保存的值。

## Summary (EN)

`list_h3_latent_upscale_models()` treated the `"3d"` substring as a hard whitelist, so any same-family H3 latent upscaler without `3d` in its filename disappeared from the SelfLift / Refine dropdown as soon as one `3d` file existed in the folder. This relaxes the filter to a family hint set (`3d` / `h3` / `minimax` / `lms`), ranks explicit `3d` checkpoints first so the node default is unchanged, and keeps the "list everything" fallback. Loading and inference paths are untouched.



### #244 [open] ISSUE: 使用自定义sigma后，视频无法预览
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/244  by JiuChenTian 2026-09-30T05:01:41Z

只要接入了自定义sigma，导演台、视频保存以及 Model Preview Override 的预览都会直接失效，无法进行视频预览

<img width="207" height="98" alt="Image" src="https://github.com/user-attachments/assets/6c74c8d7-2e41-483c-a73a-ae0e4a37eeaa" />


### #243 [open] ISSUE: Getting out of ,memory while creating the video
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/243  by maxbizz 2026-09-29T12:35:26Z

`[INFO] Comfy model compiler graph breaks: 5, rogues: 0
[ERROR] !!! Exception during processing !!! [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 2865954816 bytes.
[ERROR] Traceback (most recent call last):
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 547, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 352, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 326, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\execution.py", line 314, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\nodes\director.py", line 335, in execute
    execute_director_plan_core(
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1681, in execute_director_plan_core
    chunk, audio_dict, pre_chunk, pre_face_chunk = _run_one_segment(
                                                   ^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 1418, in _run_one_segment
    decoded, audio_dict = _decode_av_latent(
                          ^^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI_MiniMaxH3_Director\director\executor_core.py", line 144, in _decode_av_latent
    images, = VAEDecode().decode(vae, samples)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\nodes.py", line 342, in decode
    images = vae.decode(latent)
             ^^^^^^^^^^^^^^^^^^
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\comfy\sd.py", line 1301, in decode
    model_management.raise_non_oom(e)
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\comfy\model_management.py", line 399, in raise_non_oom
    raise e
  File "I:\AI\ComfyUI_windows_portable\ComfyUI\comfy\sd.py", line 1285, in decode
    pixel_samples = torch.empty(self.first_stage_model.decode_output_shape(samples_in.shape), device=self.output_device, dtype=self.vae_output_dtype())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 2865954816 bytes.
`

Getting this error.im trying to generate a 9s video 1mp.


### #242 [open] ISSUE: 有没有人遇到过风扇狂转但是突然无DP信号输入的情况？只能强制重启解决
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/242  by cuukenn 2026-09-28T06:35:43Z

5700X 5070 64GRAM
目前暂不清楚什么原因、游戏正常未出现、仅再跑ComfyUI的时候出现、而且是最近出现的、以前也没有遇到过

> **Comment by cuukenn (2026-09-28T11:04:53Z):**

有点怀疑是knode saga加速节点、去掉貌似不黑屏了、你们这环境是怎么装的


> **Comment by mypf (2026-09-30T05:45:16Z):**

我之前N卡也是这样的，突然黑屏，风扇一致转，目前都不知道为什么，黑屏的时候显存温度传感器温度为0！


> **Comment by cuukenn (2026-09-30T05:47:33Z):**

> 我之前N卡也是这样的，突然黑屏，风扇一致转，目前都不知道为什么，黑屏的时候显存温度传感器温度为0！

我把saga attention加速关了、暂时没出现这种问题了



### #241 [open] PR: 新增延迟合并成片、项目管理与合并进度条（含内存水位保护）
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/241  by a437409255 2026-09-26T18:16:28Z

## 概述

汇总 fork 上的 10 个提交，为 MiniMax H3 导演台新增「延迟合并成片」「项目管理」「合并进度条」「自动内存水位保护」，修复合并、挑段与缓存指纹相关问题，并合并上游 main 的分段 LoRA 能力。

## 提交清单

| 提交 | 内容 |
| --- | --- |
| 09cf45c | 新增延迟合并成片、自动内存水位保护与手动合并成片 |
| 955f71f | 补充内存水位守卫调参说明，修复重跑时监视器统计未复位 |
| 14ca11a | 新增项目管理与合并进度条，修复跨时间戳合并与挑段持久化 |
| 955ca7b | 修复跨段分辨率不一致导致的合并失败，并明确告知原因 |
| fcbfa5a | 指纹纳入 projectId，并让二采面板显示不匹配的真实原因 |
| 135232b | 修复 link 查找在 Map 型 graph.links 下失效，恢复二采指纹匹配 |
| 6572d37 | 修复项目复制丢内容，并让二采面板日志点名不匹配的段与提示词差异 |
| d896194 | 修复项目快照落后一版导致切换/复制/新建时旧提示词回灌编辑器 |
| 42f8ca9 | 合并上游 main（引入分段 LoRA） |
| f84259d | 缓存指纹纳入分段 LoRA，并保持旧缓存向后兼容 |

## 新增功能

### 1) 延迟合并成片 / 手动合并成片
- 分段落盘后延迟合并：有项目时跨本项目全部历史 run 逐段取最新合成（旧无 run 元数据的视为未分组），无项目时只合并本轮。
- 挑段面板（按项目 / 按时间戳 run / 仅所选段）：默认全不选，支持全选、清空与已选计数，多列宫格展示，视频可点播预览；选择持久化到节点 properties，切换时间戳下拉再切回不丢失。
- 跨 run 同名段改用 run::name 复合键，修复「跨时间戳选两个第一段只合出一段」。
- 支持分段导出的废片删除。

### 2) 合并进度条
- 后端合并进度存储 + `GET /minimax/director/merge_progress` 轮询接口。
- 前端进度弹窗：探测 / 解码 / 拼接 / 编码 / 写出 + 百分比 + 当前阶段。

### 3) 项目管理
- 新增 `web/js/minimax_projects.js` 项目存储；时间轴节点以 properties 保存与切换项目。
- 分段缓存按项目隔离：缓存键 `resolve_project_id`（timeline projectId，清洗为 `[0-9A-Za-z_-]`，非法/缺失回退 node_id），缓存目录 `minimax_seg_cache/<projectId>/`。
- 切换项目时前端主动刷新二采 / 成片缓存状态，显示目标项目的真实缓存状态。

### 4) 自动内存水位保护
- 按「预测后水位 = 可用内存 − 上段实测峰值 × 1.15」分档：正常不动 / 偏低只清 CUDA 缓存（零磁盘写入）/ 危险才卸载模型；阶段内另有 5% 硬地板，只做 gc 不卸载。
- 检查时机：段开始前、二采前、脸修前；新增两个可调阈值（偏低档默认 25%、危险档）与完整 tooltip 调参说明。
- 段失败 / 取消时无条件深度释放（不依赖开关状态）。

## 合并上游（分段 LoRA）

- 上游新增分段 LoRA：`director/segment_loras.py`、`director/lora_previews.py`、`web/js/minimax_segment_loras.js`；`SegmentPlan.loras` 落地，`executor_core.py` 采样 / 精修 4 处改用套用本段 LoRA 后的 `seg_model`。
- 本地功能改动先落盘为提交再 merge `upstream/main`，无冲突；上游新增文件已并入基线，PR diff 里只保留本地功能改动。

### 修复：改 LoRA 会静默命中旧缓存
- 根因：分段 LoRA 只 patch 到 MODEL，没有进入段身份指纹（`_segment_identity_fingerprint`）。同一段换一套 LoRA 仍判定缓存命中，直接复用上一次未挂 LoRA 的 latent，出图与设置不符。
- 修改 `director/segment_cache.py`：
  - 新增 `LORAS_FP_KEY = "loras"`；新增 `_lora_fingerprint_value()`，用 `segment_loras.lora_signature()` 生成 JSON 稳定指纹（只计 active 且 |strength| > 1e-6 的行）。
  - `_segment_identity_fingerprint()` 纳入该键。
  - `_expected_for_stored()` 向后兼容：旧 meta 无 `loras` 键、且当前段确实未挂 LoRA 时，比对时跳过该键 —— 合并后已有缓存不失效；一旦挂上 LoRA，缺键即判不匹配并强制重采样。

## 修复

- **跨段分辨率不一致导致合并失败**：新增 `_resize_frames_u8`，按各段最大 W×H（偶数对齐）统一缩放后再拼接 / 写管；原 precise 方案会 `torch.cat` 形状不匹配、stream 方案会写坏 ffmpeg 管道导致 Broken pipe。检测到时输出明确日志（列出各段原始尺寸与统一目标）并在进度条告知。
- **新版前端 `graph.links` 为 Map**：旧代码 `links[id]` / `links.find()` 取值失败，接线解析链静默返回 null，导致语义桥 / SelfLift / 时间轴指纹缺失、二采恒判不匹配。改为兼容 Map / 数组 / 对象三种形态，并在不匹配时输出 `diff_keys` 定位差异键。
- **一采指纹未含项目**：指纹纳入 `projectId`；二采面板区分「选中段还没有一采缓存」与「一采指纹不匹配」，并列出不符段号。
- **项目复制丢内容 / 快照滞后**：项目快照落后一版（注入工具只写 timeline widget、不写项目仓库），切换 / 复制 / 新建时 `_activateProject` 用旧快照覆盖编辑器，表现为「没改任何操作，跑完提示词自己变了」。修复：加载后回写当前快照（仅在内容确不同时写）、同步 `projectStore.activeId` 消除死指针、复制前先落盘当前工作流。
- `deferred_merge` 的 `_rep` 局部变量未定义（UnboundLocalError，定义前移）。
- ffmpeg 写管 Broken pipe 捕获为明确报错。
- 重跑段时监视器统计未复位（会继承上一轮预测值 / 汇总）。

## 涉及文件（17 个，+4744 −119）

- 后端（10）：`director/deferred_merge.py`、`executor_core.py`、`http_routes.py`、`plan.py`、`segment_cache.py`、`segment_continuity.py`、`segment_mp4_export.py`、`vram_cleanup.py`；`nodes/director.py`、`nodes/director_common.py`
- 前端（7）：`web/js/minimax_merge.js`、`web/js/minimax_projects.js`（新增）、`web/js/minimax_timeline.js`、`web/js/minimax_i18n.js`、`web/js/minimax_refine.js`、`web/js/minimax_selflift.js`、`web/js/minimax_semantic_bridge.js`

## 注意事项

- 含 Python 后端改动，需**重启 ComfyUI** 生效；前端改动浏览器强刷（Ctrl+F5）即可。
- 分段缓存目录由旧的 node_id 键改为 `minimax_seg_cache/<projectId>/`；旧工作流无 projectId 时自动回退 node_id，保持兼容。
- 分段 LoRA 已纳入指纹：改 LoRA 或强度后该段自动重采样，无需手动清缓存；旧缓存（meta 无 `loras` 键）在未挂 LoRA 时依然有效。
- 合并时若各段分辨率不一致，会统一缩放到最大分辨率（偶数对齐）；帧率 / 采样率不一致会记录日志提醒（按首段合成，时长可能有偏差）。
- 环境无 ffprobe 时依赖 `imageio_ffmpeg` 兜底探测。

## 测试

- 语法：17 个改动文件中 10 个 Python 通过 `py_compile`、7 个 JS 通过 `node --check`；合并后在 J 工作副本与 G PR 镜像两处分别全量校验（71 个 Python + 16 个 JS）均通过，无冲突标记残留。
- 缓存指纹：独立脚本验证 —— 指纹 JSON 往返稳定；空 LoRA + 旧 meta 保留缓存；挂 LoRA + 旧 meta 判定不匹配（强制重采）。
- 端到端：两个不同分辨率分段（320×240 + 480×270）在 precise 与 stream 两种合并方案下均成功输出 480×270、48 帧。

## 更新（2026-10-13）：复用缓存续跑 + 首段一采缓存修复
- 工具栏「选择运行」旁新增「复用缓存续跑」开关（默认关）：开启后选中段若成片缓存指纹完全命中直接复用磁盘缓存不重采样，从首个缺失/改动的段断点续跑；开启时 seed 自动切 fixed 保证指纹命中
- 修复首段一采缓存永不落盘：引导开启时首段同样落盘 .pre 缓存，选段1二采可直接命中一采缓存
- load_segment_av_latent / load_first_pass_av_latent 拒绝加载时输出 diff 键与 stored->expected 明细
- 单段重抽、二采、宫格等显式段范围场景不受影响


### #240 [open] ISSUE: FaceRefine：修脸结果未写入导出视频（输出与无 FaceRefine 基线逐像素一致）；export_pre_face_refine + comfy-aimdo 0.5.3 下修脸采样进程崩溃
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/240  by asdf19910719 2026-09-26T06:31:15Z

## 环境

| 项 | 版本/值 |
| --- | --- |
| ComfyUI | 0.36.0 portable（Windows 11） |
| ComfyUI_MiniMaxH3_Director | main @ 4b4fe6e1（2026-09-26 部署） |
| comfy-aimdo | 0.5.3（崩溃复现）/ **0.5.5**（不崩，见下） |
| GPU / RAM | RTX 5070 Ti 16GB / 48GB |
| 权重 | ref2va_pruned_int8_convrot (UNET) + qwen3vl_32b nvfp4_awq (TE) + video_vae fp16 + audio_vae fp32 |
| 检测器 | face_yolov8m.pt（就位；离线复检同帧检出人脸 conf 0.869/0.766——检测层正常） |

## 复现

任一单段 r2v 图（含人物参考、画面中有人脸）：Director 挂 `face_refine`（denoise 0.40 / steps 8 / canvas 768×768 manual / conf 0.35 / crop_factor 2.5 / paste face_only / feather 24 / colour_match on），960×544@24fps、283 帧、steps 20、ref_max_size 2048、固定 seed。

| # | 变体 | 耗时 | 结果 |
| --- | --- | --- | --- |
| A | 挂 face_refine | 10.9 min（主采样 ≈5.6 + FaceRefine 链路 ≈5.3） | success |
| B | 同图断开 face_refine（其余逐字节相同，同 seed） | 5.6 min | success |
| C | A + `export_pre_face_refine=true` + `images_pre_face_refine` → CreateVideo → SaveVideo | 9.5 min 处 | **Fatal Python error: Aborted，进程死亡**（修脸采样第 7/8 步，30.6s/it） |

### 缺陷 ①：修脸结果未进入导出（静默）

A 与 B 的输出视频在 frame 50/150/250 上**逐像素完全一致**（PIL ImageChops 全帧均值差 = 0.00×3）。段横幅确认 FaceRefine 配置已挂载（`FaceRefine: face_yolov8m.pt denoise=0.40 steps=8 canvas=768x768 (manual)`），且 +5.3 分钟耗时证明 track/re-sample/decode 确实执行——但 stitch 结果没有进入 `images` 导出，违反 tooltip 契约「images is after stitch」。**无报错、无日志、status=success**——没有 B 基线对照时会被误判为功能生效。

### 缺陷 ②：export_pre 组合崩进程（aimdo 0.5.3）

Run C 在修脸重采样第 7/8 步触发 `Fatal Python error: Aborted`（堆栈顶帧 `nodes/director.py:335 in execute`）。与 #183（comfy-aimdo 0.5.3 回归）同族：**升级 comfy-aimdo 至 0.5.5 后 Run C 不再崩溃**、正常产出主输出与 pre 输出两个文件——但三向对比（主输出 vs pre vs 基线）仍全部逐像素一致，**缺陷 ① 在 0.5.5 上依旧存在**。

## 期望

`images` 导出 stitch 后的帧；修脸链路若被跳过/失败时有可见报告（当前 report 只进 Director 的 STRING 输出，未接线时完全不可见，加剧了静默性）。

## 备注

- 复现时目标画面为单人大特写（非多人场景），置信度充足；
- 相关：#183（aimdo 0.5.3 崩溃回归——我们观察到的崩溃层被 0.5.5 修复）、#99（FaceRefine 独立后处理形态请求）、#200（多人修脸）；
- 如需更多内部日志/复现材料可以继续提供。


### #239 [open] ISSUE: Replace reference image still showing original image.
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/239  by HoloByteus 2026-09-25T16:56:53Z

After rendering a video with the subject wearing a cape prop with a hood, I edited the image in photoshop to remove the hood, gave it a different name and replaced it in the workflow. Then when rendering the shot, it still shows a hood.  The prompt has no reference to there being a hood and the image doesn't have it, it's as if the original image is cached and still being used.  This is the first segment as well so it shouldn't be carried over at all.   Even if so, the reference image has a different name but it's just not being pulled.  I guess I can try deleting it in input, I guess that's where it's cached but it really shouldn't be using the old image.


### #238 [closed] ISSUE: 【建议】关于分镜一键克隆
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/238  by xb191799981 2026-09-25T07:47:15Z

有时候一段故事在同一个场景中剧情很长，需要切成多个分组。  但是资源却是一模一样的。
目前都是创建一个空白分组，将资源上传，但是如果同一段剧情切成10个空白分组的话，同样的资源要重复操作10次，体验非常的不好。 

因此希望大佬能将当前右键点击创建空白分组，改成克隆所点击的分组。

> **Comment by AIMixer (2026-09-29T02:04:34Z):**

已支持



### #237 [closed] ISSUE: 公共参数图片最多只能为9张
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/237  by stc66 2026-09-25T02:15:52Z

<img width="971" height="422" alt="Image" src="https://github.com/user-attachments/assets/37854920-e3ac-4592-bbd1-b21288289576" />
公共参数图片最多只能为9张，可否调多一些？9张完全不够用


### #236 [open] ISSUE: 【BUG】开启段间引导会报错
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/236  by xb191799981 2026-09-25T01:43:08Z

如图所示：
开启段间引导+引导重绘， 运行2分23秒左右就会报错。 如果关闭引导重绘， 就不会有问题。
<img width="1702" height="532" alt="Image" src="https://github.com/user-attachments/assets/dfcb3a73-8b53-418a-b407-c25be1b2afa4" />


### #235 [open] ISSUE: 导入导演台报错提示“crypto.randomUUID is not a function”(已提交BUG修改PR，新增方法createPackUploadId() )
URL: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/issues/235  by stc66 2026-09-25T00:13:18Z

导入导演台报错提示“crypto.randomUUID is not a function”
<img width="452" height="115" alt="Image" src="https://github.com/user-attachments/assets/1b1e280a-2134-4de3-a4a6-520bf3370f68" />

> **Comment by stc66 (2026-09-25T03:57:47Z):**

function createPackUploadId() {
    const c = globalThis.crypto;
    if (typeof c?.randomUUID === "function") {
        return c.randomUUID();
    }
    if (typeof c?.getRandomValues !== "function") {
        throw new Error("浏览器不支持安全随机数，请使用新版浏览器");
    }

    const b = c.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;

    const h = Array.from(
        b, x => x.toString(16).padStart(2, "0")
    ).join("");

    return [
        h.slice(0, 8), h.slice(8, 12), h.slice(12, 16),
        h.slice(16, 20), h.slice(20)
    ].join("-");
}
const uploadId = createPackUploadId();

以上为修改BU的PR

