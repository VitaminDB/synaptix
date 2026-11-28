# knoic__ComfyUI-MiniMaxH3-PrefixStream

Source: https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream
Fetched: 2026-11-28

Description: High-performance Prefix KV Caching & Streaming Long Video Suite for MiniMax H3 in ComfyUI
Stars: 233  Updated: 2026-11-09T03:48:58Z



---

## FILE: README.md (https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/blob/HEAD/README.md)

# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

[English](README_EN.md) | 简体中文

面向 **MiniMax H3 长视频生成** 的 ComfyUI 节点套件。它把一段段短片组织成可管理、可选择、可无缝承接的长视频流程：生成片段、归档到素材箱、可视化挑选要承接的镜头，再继续生成下一段。

它以低侵入方式接入现有的原生 MiniMax H3 工作流：保留你的模型、提示词、采样器与解码链路，只在续写处插入配置与应用节点。默认的 **Native Masked AV** 路径参考了 [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)，使用 ComfyUI 原生去噪遮罩保护上一段的音视频上下文；`Safe Native` 则是兼容性备选方案。

---

## 核心特性

- 🎬 **为长视频续写而生**：每一段既是成片，也是下一段可复用的上下文；自动处理重叠开头裁切与音画同步，让分段生成可以稳定累积为长视频。
- 🧩 **低侵入地兼容原生 H3 工作流**：不改动 ComfyUI 源码，也不 monkey-patch H3 DiT block。沿用原有模型、提示词、采样和解码节点，只增加续写所需的少量节点。
- 🛡️ **默认使用 Native Masked AV**：将上一段干净的视频/音频 latent 直接写入下一段开头，并用 ComfyUI 原生遮罩保护；视频、音频各自独立控制。`Safe Native` 保留为兼容性备选。
- 🗂️ **可视化接力素材箱**：自动归档片段预览、评分、镜头标签、提示词及前后镜头血缘；在画廊中按画面和评分选择任意历史片段，直接接力，而不是靠文件名猜测。
- 🎯 **准确且可控的上下文**：支持 H3 的 `5 / 22 / 39 / 56 / ... / 192` 帧网格。22 帧对应 7 个视频 latent 和 37 个音频 tick；默认 39 帧对应 12 和 65。
- 🔒 **减少续写冲突**：移除受保护开头内的关键帧，避免第 0 帧引导与复制的 latent 上下文相互冲突；后续端点和参考条件仍可正常使用。
- 🖼️ **保留整组原生图片参考**：随片段保存用户提供的图片参考，续写时缺失才恢复；当前参考优先，不自动将生成首帧或关键帧转换成新参考。支持 `identity_latent` 显式补接图片，以及保存、素材箱和裁切后的时序状态传递。

---

## 快速浏览

```text
原生 MiniMax H3 工作流 → 生成当前片段 → Clip Bin 自动归档
                                      │
                                      ▼
                     在画廊中选择任意历史镜头作为接力来源
                                      │
                                      ▼
                  Continuation Config / Applier → 原生 H3 继续采样
                                      │
                                      ▼
                        裁切重叠开头 → 拼接为长视频 → 再次归档
```

### 一个节点选择续写方案

`MiniMax H3 Continuation Config` 只保留必要的核心选择：普通续写使用 **Native Masked AV (Recommended)**；只有工作流需要兼容方案时才切换为 **Safe Native**。`continuation_frames` 用于选择 H3 帧网格上的受保护重叠长度。

![MiniMax H3 续写配置节点](assets/readme/continuation-config.png)

### 从素材箱浏览并继续镜头

`MiniMax H3 Clip Bin Picker` 将已保存的片段以可视化画廊呈现。选择上一镜头、按评分筛选，并将其 latent 与尾帧作为下一段的续写上下文，无需在输出目录中翻找文件。

![MiniMax H3 素材箱画廊选择器](assets/readme/clip-bin-picker-gallery.png)

---

## 工作原理

```text
上一段完整音视频 latent ──► 选择规范的 5/22/39/56/... 帧尾部
                                      │
目标空音视频 latent ───────────────────┼──► 将尾部复制到目标开头
                                      └──► noise_mask =（视频遮罩，音频遮罩）
                                                        │
                                                        ▼
                                      ComfyUI 原生 MiniMax H3 采样
```

---

## 自定义节点

| 节点名称 | 分类 | 说明 |
| :--- | :--- | :--- |
| **`MiniMax Video Chunk Slicer`** | `MiniMaxH3/VideoEdit` | 🎬 长视频智能切片器（零显存流式加载）：直接按需提取长视频片段，支持自动网格/自由区间切片，**支持直接输出上一片段末帧（单帧图片参考）与尾部序列（多帧视频参考）**，自动优先提取已编辑成果，为下游图生视频/视频延续提供无缝参考。 |
| **`MiniMax Video Patch Reassembler`** | `MiniMaxH3/VideoEdit` | 🧩 长视频无缝回填缝合器：自动将生成的片段内存替换进母带时间轴，支持余弦微融合消除缝合线，可视化追踪全片完成度并一键导出最终成品 MP4。 |
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | 选择默认的 `Native Masked AV` 或备选的 `Safe Native`，并设置用户可见的视频上下文长度。 |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | 构建原生视频/音频遮罩并输出 `masked_latent`；在备选模式下应用避免冲突的关键帧条件。 |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | 在像素和音频波形空间裁切开头的重叠帧，避免 VAE 因果闪烁并保持同步。 |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | 在像素和音频波形空间无缝拼接长视频与新片段，具备自适应亮度匹配、重叠区不增益的余弦交叉淡化及防二次裁剪守卫。 |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | 独立地将联合音视频 latent 保存为 safetensors，无外部依赖。 |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | 独立加载带有元数据的联合音视频 latent，用于多片段连续流式生成。 |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | 连同预览图、评分、镜头标签、提示词和续写血缘归档生成片段。 |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | 以画廊方式查找已保存的片段（支持卡片流与树形图谱一键切换），并输出其 latent、尾帧、提示词和 ID。 |
| **`MiniMax H3 Clip Bin Tree Picker`** | `MiniMaxH3/ClipBin` | 以 DAG 分支血缘拓扑树（Tree View）全景可视化历史片段、多分支探索版本与完整故事线祖先链路。 |
| **`MiniMax Cache Telemetry Monitor`**| `MiniMaxH3/PrefixStream` | 显示当前续写模式、受保护上下文几何信息和会话进度。 |

---

## 音频接续模式

接续帧下拉首选项和后端默认值均为 **39**。5 帧在 24 FPS 下只有约 0.208 秒，复制的音频按 40 Hz 取整为 8 个 tick（0.2 秒）；39 帧则提供 1.625 秒 / 65 个 tick，音画时长精确对齐。5 帧给声音续写的历史很短，且冻结音频结束与视频裁切位置有约 8.3 毫秒差异；这些是代码层面可确认的限制，具体失真来源仍需真实模型对照试听。5 帧下若 `audio_feather_ticks >= 8`，整个继承音频区均可被重生成；排查时建议先用 39 帧、feather 为 0 和 `Match Video Handover`。目标过短时实际上下文仍会向下选择可用档位。

Continuation Config 的 `audio_continuation_mode` 默认为 `Masked Prefix`，保留原有音频前缀冻结方式。若接续段生成的音频持续失真，可对照测试 `Native Audio Guide`：它把同一段历史音频送入 H3 原生音频引导条件，仅冻结视频前缀，音频正常生成。该模式忽略 `audio_feather_ticks`，仍需裁掉重叠前缀；生成的重叠音频不会与原段逐样本相同。当前仅验证了条件、遮罩和时序传递，尚未验证真实生成听感。


素材箱的 **全新首段（根节点）** 使用 `clip_selection=initial`：Auto 模式下即使库里已有片段，也输出空历史上下文；该选择会保持到手动切换。**Auto / 自动最新** 使用 `latest`，继续自动接续最新片段。卡片流和关系树均提供这两个独立入口。Strict Chaining 模式选择 `initial` 会提示切换模式；填写有效 `custom_clip_path` 时仍优先使用路径覆盖。

## 安装

将仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
cd ComfyUI-MiniMaxH3-PrefixStream
pip install -r requirements.txt
```

重启 ComfyUI 后，节点会出现在 `MiniMaxH3/PrefixStream` 与 `MiniMaxH3/VideoEdit` 分类下。

---

## 运行单元测试

可通过以下命令进行本地验证：

```bash
python -m unittest discover -s tests -v
node tests/test_clip_bin_frontend.cjs
```

---

## 低内存长视频输出

新增 `MiniMax H3 Disk Video Stream`：逐片段写入磁盘，最后合成 MP4，无需在内存中累计全部画面。需要 FFmpeg，采用硬切拼接；原拼接节点继续提供亮度匹配和接缝淡化。连接与导出步骤见[中文指南](docs/USER_GUIDE_CN.md#磁盘分段长视频低内存)。

## 示例工作流与文档

- 📄 **长视频切片与自动缝合工作流**：[`examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json`](examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json)
  - 专为长视频重绘与连续接力打造的轻量工作流，包含智能切片器、上一段参考帧连线、VHS 规格读取与自动回填缝合。
- 📄 **PrefixStream 连续生成工作流**：[`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - 将这份完整的 42 节点 JSON 直接拖入 ComfyUI 画布，即可运行 Native Masked AV 续写长视频生成。
- 📖 **中文详细使用指南**：[`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - 包含长视频切片器、参考帧输出连线、硬件建议、参数调节与常见问题。

---

## 致谢与参考项目

本项目的诞生与时空续写设计深受以下开源项目与作者的启发，特此致以诚挚的感谢与敬意：

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢 NikoDemon80 在 MiniMax H3 关键帧锚定算法、VAE 时空周期相位网格对齐公式（Snap to Run Grid）、音视频头部裁切及 5/3 音视频时间缩放比例方面的先驱性数学探索与启发。

2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢 Native Masked AV 的原生分流遮罩、精确 AV 上下文边界及独立音频保护方案。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE) 发布。


---

## FILE: README_EN.md (https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/blob/HEAD/README_EN.md)

# ComfyUI-MiniMaxH3-PrefixStream

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-green.svg)]()
[![Model](https://img.shields.io/badge/model-MiniMax--H3-orange.svg)]()

English | [简体中文](README.md)

A ComfyUI node suite for **MiniMax H3 long-video generation**. It turns short generated clips into a manageable, selectable, and seamlessly connected long-video workflow: generate a clip, archive it in the Clip Bin, visually choose the shot to continue from, then generate the next clip.

It integrates with existing native MiniMax H3 workflows with minimal intrusion: keep your model, prompts, sampler, and decode chain, then add configuration and application nodes only where continuation is needed. The default **Native Masked AV** path follows ideas from [Herrgott's H3 Infinite Continuation Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite), protecting the previous audiovisual context with ComfyUI's native denoise masks. `Safe Native` remains available as a compatibility fallback.

---

## Key Highlights

- 🎬 **Built for long-video continuation**: Every clip is both an output and reusable context for the next one. Overlap trimming and audiovisual synchronization are handled automatically, so segmented generations can reliably accumulate into a long video.
- 🧩 **Low-intrusion native H3 integration**: Does not modify ComfyUI source files or monkey-patch H3 DiT blocks. Keep your existing model, prompt, sampling, and decoding nodes; add only the few nodes needed for continuation.
- 🛡️ **Native Masked AV by default**: Writes clean video/audio latents from the previous clip into the head of the next target and protects them with native ComfyUI masks. Video and audio are controlled independently. `Safe Native` is retained as the compatibility fallback.
- 🗂️ **Visual continuation media library**: Archives clip previews, ratings, shot tags, prompts, and parent-child lineage. Choose any historical clip by image and rating from a gallery instead of guessing from filenames.
- 🎯 **Precise, controllable context**: Supports H3's `5 / 22 / 39 / 56 / ... / 192` frame grid. A 22-frame context copies 7 video steps and 37 audio ticks; the default 39-frame context copies 12 and 65.
- 🖼️ **Persistent native image references**: Preserves the complete supplied image-reference bank across AV files and Clip Bin assets, restoring it only when current image references are absent. Generated frames and keyframes are not silently converted into references; `identity_latent` explicitly adds an image. Trimmed latents retain their video phase and audio origin.
- 🔒 **Reduced continuation conflicts**: Removes keyframes inside the protected head so a frame-0 guide cannot conflict with copied latent context. Later endpoint and reference conditioning remain available.

---

## At a Glance

```text
Native MiniMax H3 workflow → generate current clip → automatically archive to Clip Bin
                                             │
                                             ▼
                         choose any historical shot in the visual gallery
                                             │
                                             ▼
                  Continuation Config / Applier → continue native H3 sampling
                                             │
                                             ▼
                     trim overlapping head → stitch long video → archive again
```

### Choose the continuation path in one node

`MiniMax H3 Continuation Config` keeps the decision deliberately small: use **Native Masked AV (Recommended)** for normal continuation and switch to **Safe Native** only when a workflow requires the compatibility fallback. `continuation_frames` chooses the protected overlap length on the H3 frame grid.

![MiniMax H3 Continuation Config](assets/readme/continuation-config.png)

### Browse and continue from the Clip Bin

`MiniMax H3 Clip Bin Picker` displays saved clips as a visual gallery. Select a preceding shot, filter by rating, and use its latent and tail frame as the next continuation context—without searching through output folders.

![MiniMax H3 Clip Bin Picker gallery](assets/readme/clip-bin-picker-gallery.png)

---

## How It Works

```text
Previous full AV latent ──► select the canonical 5/22/39/56/... frame tail
                                      │
Target empty AV latent ───────────────┼──► copy tail into the target head
                                      └──► noise_mask = (video mask, audio mask)
                                                        │
                                                        ▼
                                      Native ComfyUI MiniMax H3 sampling
```

---

## Custom Nodes

| Node Name | Category | Description |
| :--- | :--- | :--- |
| **`MiniMax Video Chunk Slicer`** | `MiniMaxH3/VideoEdit` | 🎬 Long video chunk slicer (zero-RAM lazy decoding): on-demand extraction of video chunks with auto grid / custom range, **featuring direct outputs for previous chunk's last frame (image ref) and tail sequence (video ref)** with auto-propagation of edited frames. |
| **`MiniMax Video Patch Reassembler`** | `MiniMaxH3/VideoEdit` | 🧩 Long video patch reassembler: in-place replacement of edited chunks into master timeline with cosine micro-seam blending, coverage tracking, and 1-click full MP4 export. |
| **`MiniMax H3 Continuation Config`** | `MiniMaxH3/PrefixStream` | Selects default `Native Masked AV` or fallback `Safe Native`, and sets the user-visible video context length. |
| **`MiniMax H3 Continuation Applier`** | `MiniMaxH3/PrefixStream` | Builds native video/audio masks and outputs `masked_latent`; applies collision-safe keyframe conditioning in fallback mode. |
| **`MiniMax Trim Prefix`** | `MiniMaxH3/PrefixStream` | Trims leading overlap frames in pixel and audio waveform space, avoiding VAE causal flicker while retaining sync. |
| **`MiniMax Long Video Stitcher`** | `MiniMaxH3/PrefixStream` | Seamlessly stitches accumulated long video and new clips in pixel and audio waveform space with luminance matching, unity-gain audio crossfade, and double-trim safety guards. |
| **`MiniMax Save AV Latent`** | `MiniMaxH3/PrefixStream` | Saves joint audiovisual latents to safetensors without external dependencies. |
| **`MiniMax Load AV Latent`** | `MiniMaxH3/PrefixStream` | Loads joint audiovisual latents and their metadata for multi-clip continuous streaming. |
| **`MiniMax H3 Clip Bin Saver`** | `MiniMaxH3/PrefixStream` | Archives a generated clip with its preview, rating, shot tag, prompt, and continuation lineage. |
| **`MiniMax H3 Clip Bin Picker`** | `MiniMaxH3/PrefixStream` | Gallery-based loader (supports 1-click toggle between Card Deck and Lineage Tree) that exposes a selected clip's latent, tail frame, prompt, and ID. |
| **`MiniMax H3 Clip Bin Tree Picker`** | `MiniMaxH3/ClipBin` | Visually browses and selects clips via an interactive DAG lineage tree graph for exploring non-linear story branches. |
| **`MiniMax Cache Telemetry Monitor`** | `MiniMaxH3/PrefixStream` | Shows the active continuation mode, protected-context geometry, and session progress. |

---

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
cd ComfyUI-MiniMaxH3-PrefixStream
pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under `MiniMaxH3/PrefixStream` and `MiniMaxH3/VideoEdit`.

---

## Running Unit Tests

Verify the project locally:

```bash
python -m unittest discover -s tests -v
```

---

## Example Workflow and Documentation

- 📄 **Long Video Slicer & Reassembler Template**: [`examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json`](examples/MiniMaxH3_LongVideo_Slicer_Reassembler_Template.json)
  - Dedicated workflow for long video repainting, editing, and chained generation with previous chunk reference connections.
- 📄 **Ready-to-use PrefixStream workflow**: [`examples/MiniMaxH3_PrefixStream_v1.0.json`](examples/MiniMaxH3_PrefixStream_v1.0.json)
  - Drag this complete 42-node JSON directly onto a ComfyUI canvas to run PrefixStream long-video generation.
- 📖 **Detailed Chinese user guide**: [`docs/USER_GUIDE_CN.md`](docs/USER_GUIDE_CN.md)
  - Includes hardware guidance, parameter tuning, multi-clip infinite-continuation walkthroughs, and FAQs.

---

## Acknowledgments and References

This project and its temporal continuation design were strongly inspired by the following open-source projects and authors:

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - For pioneering work on MiniMax H3 keyframe anchoring, VAE spatiotemporal phase-grid alignment (Snap to Run Grid), audiovisual head trimming, and 5/3 audiovisual temporal scaling.

2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - For its native per-stream AV masking, exact audiovisual context boundaries, and independent audio-protection design.

---

## License

This project is released under the [MIT License](LICENSE).


## Disk-backed long video output

`MiniMax H3 Disk Video Stream` writes each decoded clip to disk and exports a final MP4 without accumulating the complete IMAGE timeline in RAM. FFmpeg must be on PATH. Connect only the current clip's images/audio and optionally its session to trim the prefix. Keep the project and stream names constant, and leave export disabled while appending. To export without appending again, disconnect images/audio and enable export on a node with the same names. Every execution with images appends another clip; use a new stream name for a new video.

This path uses hard cuts, eight-frame encoding batches and PCM intermediate audio. All clips must have consistent fps, even resolution, sample rate, channels and audio presence. The existing IMAGE stitcher remains available for brightness matching and overlap blending, and still retains the full output in memory. Project locks protect concurrent operations within one ComfyUI process; do not share a writable bin between multiple processes. VAE decoding errors now propagate instead of producing empty results. Restart ComfyUI and refresh the browser after updating.


---

## FILE: docs/DEGRADATION_ANALYSIS_CN.md (https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/blob/HEAD/docs/DEGRADATION_ANALYSIS_CN.md)

# 皮肤斑点与纯色背景斑纹：原因与验证路径

用户描述的现象是皮肤出现斑点、纯色背景出现斑纹。这个现象应优先检查空间纹理误差的反馈累积；不能只凭症状认定是 VAE、量化或某一个频段，也不能把修复接缝当成修复纹理。

## 误差怎样传播

可用局部近似 `下一段误差 ≈ A × 历史误差 + 本轮新误差` 理解。复制 latent 避免重复 VAE 编码，但复制的内容仍可能已经有斑纹。硬遮罩保证这些值不变，不等于让模型识别并消除它们。`A` 是否会放大误差，取决于模型、条件与采样，必须用多段生成验证。

固定原始图片主要补充长期外观条件，不能保证恢复每个位置的皮肤或背景细节。尤其当前工作流每轮已经提供同样的原始图片时，再加同一张 anchor 不能被当作新的画质修复。

[Self Forcing](https://arxiv.org/abs/2506.08009) 和 [Stable Video Infinity](https://arxiv.org/abs/2510.09212) 将干净训练历史与有误差生成历史之间的差异作为核心问题，并通过训练处理。这支持“需要处理生成历史误差”的方向，不代表对应方法已经适配或验证于本插件的 H3 模型。

## 逐项排查

| 候选原因 | 当前证据与界限 | 区分方法 |
| --- | --- | --- |
| 历史斑纹被再次用作条件 | 插件会复制生成尾部并冻结；这是明确的数据路径，但放大程度未知 | 从同一干净片段分支，比较 5 段以上的纯色区域 |
| 少步蒸馏模型每轮增加纹理偏差 | 可能与历史分布不匹配共同作用，尚未在用户模型实测 | 固定源片段、参考、种子，比较匹配各自推荐采样配置的模型；不要给四步 LoRA 随意增加步数并声称等价 |
| 动作/深度参考时间错位 | 前缀和参考可能要求不同动作；不是由斑点本身证明的 | 124 帧片段、22 帧重叠时下一轮参考从 102 帧开始，不能从 124 开始；RGB、Depth、Audio 使用同一窗口 |
| VAE 解码上下文不同 | 相同前缀 latent 不保证在不同解码历史下像素完全相同 | 比较原片末尾与新片已冻结前缀的解码输出，检查误差是否只集中于开头 |
| 编码或展示产生斑纹 | MP4 压缩、缩放可能改变平坦区域 | 比较同一帧的解码 PNG 与最终 MP4；PNG 已有斑纹则不能只归因于视频编码 |
| 亮度匹配或锐化强化斑纹 | Stitcher 会在接缝附近做增益/混合，不能解释所有后段纹理 | 对比 Stitcher 前的 PNG 和输出；关闭亮度匹配做一次单变量对照 |
| 量化放大微小预测偏差 | 可能影响模型预测，但没有证据确认是当前根因 | 其余参数相同再比较模型精度；当前插件不启用 DiT KV 缓存，不能归因于本插件的 FP8 KV |

## 本轮实现的可验证实验入口

### 1. 允许少量修正继承前缀

Config 新增 `video_context_denoise`，默认 **0**，保留原来的硬冻结行为。

- `0`：视频前缀遮罩为 0，完全保护。
- 大于 `0`：视频前缀使用该值的原生软遮罩，其余新内容仍为 1。音频遮罩独立。
- H3 会将遮罩向上取整到 `1/256` 的粒度，例如 `0.06` 的 token 遮罩实际为 `0.0625`。

这是允许模型修正历史条件的实验，不是经过画质验证的去斑算法。可做 `0 / 0.03 / 0.06` 单变量对照；不能预先承诺哪一个更好。软遮罩也可能引入接缝差异、动作漂移或模糊。原生 sampler 集成检查验证了它确实进入 H3 的 mask 与 Euler 采样，不是一个没有实际作用的参数。

### 2. 只在续写段使用纠偏模型

Applier 新增可选 `continuation_model`：没有历史片段时继续使用 `model`，有 `context_latent` 时才使用 `continuation_model`。接法是原生 `LoraLoaderModelOnly` 产生纠偏模型，再接此插槽；Applier 的模型输出必须接入实际采样链路。

这为针对生成历史误差训练的 LoRA 提供入口，不修改 DiT，也不自动下载或加载权重。

H3 社区作者发布了 [H3 Anti-Burn LoRA](https://huggingface.co/Imagineer13/H3-Anti-Burn-LoRA)，针对续写中皮肤纹理逐轮变粗的现象。作者要求仅用于续写段，并报告了改善与过度平滑的可能性。这是作者的实验结果，尚未在用户当前 checkpoint、turbo 和工作流验证；不能直接宣称已解决本项目的皮肤斑纹。

### 3. 检查局部纹理是否增强

增加 `tools/analyze_texture_drift.py`，读取原始导出的 PNG，测量相同 ROI 的亮度和色度空间频段能量：`2–8`、`8–32`、`32–128` 像素波长。先去除线性光照梯度，再做 Hann 窗与频谱统计。

```powershell
python tools/analyze_texture_drift.py clip1.png clip3.png clip5.png --roi 20 20 180 180
```

坐标顺序为 `LEFT TOP RIGHT BOTTOM`。优先选静止、纯色背景；皮肤需要同一部位、相近姿势与光照。移动镜头或对不齐的皮肤 ROI 不适合直接比较。

此工具不滤波、不美颜、不输出质量评分。中间频段变强可能是斑纹，也可能是自然细节；所有频段都降低也可能只是画面变糊。必须结合人工检查、边缘细节和动作连续性判断。

## 对照顺序

从同一个干净的首段完整 AV latent 分支，至少做 5 段。各分支使用相同的逐段种子序列、提示词、参考窗口、模型与采样配置；保存 Stitcher 前的 PNG 和完整 latent。

1. 基线：原生硬前缀，`video_context_denoise=0`，不接纠偏模型。
2. 只改变重叠长度：比较 22 与 39 帧；较短重叠减少单轮继承内容，也减少固定视频长度所需的接续次数，但可能损失连续性。
3. 固定较合适的长度，只比较 `video_context_denoise=0/0.03/0.06`。
4. 恢复硬前缀，只加入纠偏 LoRA，先区分 LoRA 本身的作用。适用范围和强度需参考作者说明并实测。
5. 确认单项有效后再组合。不能把多项一起更改后的效果归因于某一个修复。

保持真实皮肤纹理、衣纹和轮廓；不要以背景更平或皮肤更光滑作为唯一成功标准。

## 暂不采用的方案

- 不直接将当前 latent 的均值、方差或低频强行匹配首帧。latent 通道包含混合语义，不是 RGB；姿势、构图和光照变化也会改变统计量。
- 不默认高斯模糊、锐化或美颜后重编码再接续；这些处理可能隐藏斑纹，同时损失细节或引入新的 VAE 反馈误差。
- 不直接照搬注意力频域分离。[FreqForcing](https://arxiv.org/abs/2607.27110) 在其他自回归视频模型上研究了频谱漂移与注意力频域处理；这并不等于对 H3 latent 做低通滤波。
- 不默认周期性重置成独立片段；它可能减小历史依赖，却破坏动作和镜头连续性。

## 已验证与未验证

单元测试验证参数传递、默认硬遮罩不变、视频/音频独立、续写模型路由，以及频段测量对线性照明和合成纹理的响应。原生运行时检查使用真实 H3 遮罩逻辑与四步 Euler，但去噪器是测试替身。

尚未完成加载 H3 权重的真实多段生成，也没有得到足够的同条件长链来证明皮肤斑纹的改善幅度。因此本轮交付的是针对症状的实验与诊断能力，不是“已根治长视频劣化”的结论。

## 实现收敛：先减少隐式条件变化

进一步检查发现，自动把生成首帧或首帧关键帧转换成独立图片参考，缺乏画质证据，并改变原生 H3 的条件布局与时序坐标。它不能被当作可靠的抗劣化机制；现在已经移除这个缺省行为。

实现只保留三个职责：

1. **前缀几何与原生遮罩**：精确复制 AV 尾部并保存周期相位。默认硬保护，不做频域修改、归一化或隐式重编码。
2. **原生图片参考的完整传递**：保存全部用户提供的原生图片参考，而不是只取第一张。当前提供的新参考优先；缺失时恢复历史参考组；显式 `identity_latent` 才会增加新参考。已知原生 ref2va 条件保持原样。
3. **显式、独立的实验**：软前缀和续写纠偏模型仍是默认关闭的对照入口，不与原生路径捆绑。

参考管理逻辑集中在 `engine/continuation_state.py`，节点只负责调用和传递，不保留全局模型缓存。多图参考使用 safetensors 中的独立张量和小型 JSON 清单保存，避免丢失多视角信息；旧单 anchor 文件仍可读取，但不猜测它是原始图像还是生成画面。

原生运行时检查现在也执行 `MiniMaxH3Model.forward` 的输出遮罩缩放与 `CONST.calculate_denoised` 的 Flow 转换，随后进入真实 `KSamplerX0Inpaint` 与 Euler；只有网络 `_forward` 是固定速度的测试替身。软遮罩并不是“相同比例的纠偏力度”，不能把 0.06 解释成已校准的 6% 画质修复。

这个方向减少了插件额外引入条件分布变化的机会，并建立了可复现的原生基线；它仍不能代替针对生成历史误差的模型训练，也尚未证明能消除皮肤斑纹。

## 22 帧注噪收益不明显后的修正

用户反馈实际配置为开启注噪、22 帧上下文。检查发现，上一版把末尾 5 帧对应的两步 latent 恢复成历史原值；这会连同已有纹理偏差一起恢复，可能抵消最靠近交接处的干预。它能保护交接，却不能被当作已验证的抗劣化策略。

当前统一按整个上下文从 0.4 余弦渐变到末帧 0，只保留编码前最后一帧不加噪，不再把末尾旧 latent 强制回填。这个保证只针对输入像素；时间 VAE 混合相邻帧，不能据此承诺生成的新帧没有斑纹。

直接将 `E(带噪的 D(z))` 替换原始 z，还把 VAE 的正常解码/编码误差带进了下一轮。当前使用配对差值：

```text
x = 从完整解码 D(z) 中选择的干净上下文
Δz = E(给 x 渐变注噪) - E(x)
新的上下文 = 原始对应 latent + Δz
```

当没有注入噪声且编码确定时，Δz 为 0，即使 VAE 的干净 round-trip 有偏差，输出也保持原始 latent。这是测试能够验证的性质。它只分离了基线编码误差与噪声引起的编码变化，不是推导出的去斑或长期稳定性保证；噪声响应本身仍可能引起颜色或细节变化。

实现增加一次历史解码和两次上下文编码，不添加界面参数。日志记录 Δz 和原始 latent 的 RMS，用于检查实际干预幅度；RMS 大小不是质量分数。应从同一个干净首段分支，对比关闭/开启注噪并保存中间 PNG、latent 和音频，至少观察数次续写后的皮肤、平坦背景、动作连续性与声音。

回归测试覆盖全部帧数档位、零噪声取消编码基线、22 帧尾部不再被旧值覆盖、原始参考和音频保护。网络权重下的多段画质收益尚未验证，不能用测试数量代替用户的生成结果。

## 为什么视频注噪可能影响声音

代码只对视频像素/latent 加噪，音频 latent 没有加入随机噪声。但 H3 是联合音视频模型，新的音频预测会读取变化后的视频上下文，不能保证条件改变而新音频完全不变。

另外修正两个可确认的接缝风险：

- 原来的音频羽化会在边界释放部分音频重新生成。现在新默认值为 0，开启参考注噪时也强制将音频前缀硬保护。
- 原来的等功率淡化在相同、同相的重叠声音上，两个权重之和最大为 sqrt(2)，会增加约 3 dB 峰值，并可能触发后续裁限。现在改为权重和恒为 1 的余弦淡化；测试使用同源高幅度信号和正弦波，验证没有增益、削波和时间长度变化。

这两项风险不等于已经确认用户录音中噪声的来源。如果解码后的新段音频在进入 Stitcher 前就有噪声，拼接增益修复无法解决它。要严格保留原声，成片应直接使用原始音轨，而不是要求联合生成音频逐采样点保持不变。

## 接续段整体声音失真、发糊

用户进一步确认：人声、音乐或音效整体质量下降，而不只是接缝爆音。默认改为 39 帧不能视为解决此问题。5 帧 / 8 tick 的短历史、时长取整属于额外限制；39 帧 / 65 tick 消除了这一档位的时长取整，却不证明冻结历史后的新音频质量会恢复。

用户当前报告为原生 20 步、未开启 `reference_noise`，Masked Prefix 和 Native Audio Guide 均有音质下降。故本次问题不应归因于开启视频注噪，也不能用 Turbo/少步的假设代替排查。两种模式均带入生成历史音频；共同路径仍需隔离验证。

检查当前数据路径：历史音频 latent 直接复制，不经过解码再编码；默认 Masked Prefix 把复制区的音频遮罩设为 0，新区为 1；Native Audio Guide 则把同一历史窗口作为独立音频引导条件，目标音频遮罩全为 1。两条路径改变的是模型收到的条件与去噪方式，不能仅凭前缀数值保持不变判定新音频质量合格。原生音频尺度与视频/音频各自的 sigma 调度也参与冻结区注入，应核对实际运行的 ComfyUI，而非只检查插件。

Stitcher 的默认淡化窗口为 15 ms，有重叠上下文时，新段保留主体仅裁切/拼接，不滤波、不重采样、不归一化。增加了 5/39 帧两档的立体声高频合成信号回归，验证主体逐样本不变。这排除了该拼接函数对整个主体做低通处理的解释，不能排除输入音频、其他导出节点或真实模型生成失真。

使用同一个完整首段 AV latent、提示词、参考图、种子及采样设置，对照以下顺序：

1. 固定 39 帧、feather=0、Match Video Handover，单独保存音频 VAE 解码后、Trim/Stitcher 前的无损音频，并试听接续主体。若此时已发糊，优先查生成链路；若此时清晰，则比较最终封装与播放器路径。
2. 保持其他设置，比较 Masked Prefix 与 Native Audio Guide。后者在代码上取消音频冻结，尚无当前用户模型的听感证据，保持为可选实验。
3. 若开启 reference_noise，仅关闭该项再比较，区分视频条件改变对联合音频预测的影响。
4. 若使用 Turbo/少步 LoRA、跳步加速或 continuation_model，再分别与对应基础配置比较。示例工作流含 Turbo LoRA 不代表用户当前配置已确认；不能直接认定其为根因，也不能混用模型推荐采样参数后归因于单个设置。

当前 Python 环境没有可导入的 ComfyUI 运行时，也没有加载 H3 权重完成这些生成与试听；已经得到上述用户配置，仍需同源音频结果和实际 ComfyUI 版本才能确认根因。已有单元测试与替身去噪器测试证明数据路径，不证明音质。


---

## FILE: docs/USER_GUIDE_CN.md (https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/blob/HEAD/docs/USER_GUIDE_CN.md)

# ComfyUI-MiniMaxH3-PrefixStream 中文使用指南

本插件用于将 MiniMax H3 的多段生成组织为稳定的长视频续写流程。每次生成后，片段可以归档到可视化 **Clip Bin** 素材箱；下一次从画廊选取任意历史镜头作为上下文，即可继续生成或创建新的分支。

默认方案为 **Native Masked AV**：将上一段音视频 latent 的规范尾部复制到下一段的开头，并用 ComfyUI 原生视频/音频去噪遮罩分别保护。`Safe Native` 是短片段或不具备原生 AV-mask 支持时的兼容性备选。

---

## 一、安装与前提

1. 将本仓库放入 ComfyUI 的 `custom_nodes` 目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git
   cd ComfyUI-MiniMaxH3-PrefixStream
   pip install -r requirements.txt
   ```

2. 使用包含 MiniMax H3 音视频遮罩支持的新版 ComfyUI。`Native Masked AV` 依赖 ComfyUI 的 H3 AV-mask 实现；若运行时提示缺少支持，请先更新 ComfyUI，再完全重启后端。
3. 重启 ComfyUI，并在浏览器中按 `Ctrl + F5` 强制刷新前端。节点位于 `MiniMaxH3/PrefixStream` 和 `MiniMaxH3/ClipBin` 分类。

---

## 二、核心概念

```text
上一段完整 AV latent → 取规范尾部作为上下文
                                  │
新的 H3 目标 latent ───────────────┼→ Continuation Applier
                                  │       │
                                  │       ▼
                                  │   原生 H3 采样
                                  │       │
                                  │       ▼
                             解码 → Trim Prefix 裁切重叠开头 → 保存 / 合成
```

- **上下文**：上一段采样器输出的完整联合音视频 `LATENT`。MiniMax H3 的视频和音频封装在同一个粉色 latent 端口内，无需拆成两根线。
- **目标**：新一段由原生 H3 工作流创建的目标 `LATENT`，通常来自 `MiniMaxH3ReferenceToVideo`。
- **重叠开头**：为保证连续性，下一段开头会保留上一段的上下文。采样后用 `MiniMax H3 Trim Prefix` 裁掉这部分，再进行成片保存或外部合成。
- **素材箱**：保存片段的 latent、预览图、镜头标签、评分、提示词和父镜头 ID。它是续写来源的可视化选择器，而非视频拼接器。

---

## 三、节点说明

### 1. `MiniMax H3 Continuation Config`

这是唯一需要为续写策略做选择的配置节点。

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `cache_mode` | `Native Masked AV (Recommended)` | 默认方案。使用 ComfyUI 原生视频/音频遮罩保护复制到目标开头的上下文。`Safe Native (Fallback)` 用于兼容工作流或原生遮罩不可用的情形。 |
| `continuation_frames` | `39` | 支持 `5/22/39/56/.../192`，按 17 帧递增。22 帧对应官方节点接续方案的 7 个视频 latent、37 个音频 tick；39 帧对应 12、65。 |

推荐先使用默认的 `Native Masked AV + 39`。上下文越长，连续性参考越多，但目标片段必须有足够长度容纳它。

### 2. `MiniMax H3 Continuation Applier`

将它插入原生 H3 的模型、条件和采样 latent 链路中。

| 插槽 | 连接方式 |
| :--- | :--- |
| `model` | 原生 H3 模型输出。 |
| `conditioning` | 原生正向条件输出，例如 `MiniMaxH3ReferenceToVideo` 的正向条件。 |
| `cache_config` | 连接 `MiniMax H3 Continuation Config`。 |
| `context_latent` | 上一段完整的 H3 联合音视频 latent；首段生成时留空。可连接 Clip Bin Picker 的 `latent` 输出。 |
| `target_latent` | 新一段原生 H3 目标 latent，通常来自 `MiniMaxH3ReferenceToVideo`。 |
| `vae` | 仅开启 Config 的 `reference_noise` 时连接 H3 视频 VAE。关闭时不使用。 |
| `continuation_model` | 可选。仅续写段使用的模型，例如通过原生 LoRA 节点加载纠偏 LoRA 的模型；首段仍使用 `model`。 |
| `identity_latent` | 可选的固定外观参考：原始图片经 H3 VAE 编码的 latent，或首段完整 latent（只取首帧）。旧素材可在这里补接原始参考。 |

输出连接方式：

旧版 Applier 的 `anchor_latent`、`context_video_latent`、`anchor_video_latent`、`context_audio`、输入侧 `session` 已删除。刷新前端并加载工作流时会自动清理这些旧插槽；历史音频参数也统一在 Config 中设置。输出侧 `session` 仍保留，供裁切和监控使用。

- `model`、`conditioning`：分别接回原生采样链路。
- `masked_latent`：接采样器的 `latent_image` 输入。
- `session`：接 `MiniMax H3 Trim Prefix`；也可以接 `MiniMax H3 Cache Telemetry Monitor` 查看本次保护的实际上下文。

首段没有 `context_latent` 时，节点会直接传递目标 latent，不创建保护前缀。若 Native Masked AV 的源片段或目标片段过短，节点会自动回退至 `Safe Native`，并在 ComfyUI 控制台记录原因。

**参考帧注噪开关**：Config 的 `reference_noise` 默认关闭。开启后，所有档位均按实际长度从 0.4 余弦渐变到最后一帧的 0。5 帧档约为 `0.4 / 0.3414 / 0.2 / 0.0586 / 0`；22 帧等较长档位也只保留最后一帧不加噪，不再把末尾 5 帧的旧 latent 强制回填。时间 VAE 会混合相邻帧信息，所以这只保证编码前末帧不加噪，不是最终生成画面零噪点的保证。

注噪现在使用配对编码：`输出上下文 = 原始上下文 latent + 编码(带噪像素) - 编码(干净像素)`。相同 VAE 对干净上下文的重复编码结果作为基线被减掉，避免直接用重编码结果替换原始 latent。若注入的噪声为零且编码确定，两次编码相等，输出便保持原始 latent；这项性质不等于画质改善已经得到验证。

使用 `Native Masked AV`，将 H3 视频 VAE 接到 Applier 的 `vae`，`context_latent` 使用上一段完整采样输出。每次续写增加一次历史片段解码和两次所选上下文编码，耗时高于此前的单次编码。日志会记录 latent 实际改变量和原始信号的 RMS，便于排查像素噪声是否被 VAE 大幅滤掉。首段没有历史时不注噪，也不需要 VAE。目前不支持裁切后的 latent 注噪。原始人物参考、历史 latent、视频冻结逻辑保持不变。

**声音交接**：`audio_feather_ticks` 新默认值为 0。开启参考帧注噪时，即使旧工作流保存了 2，也会将音频前缀完整硬保护，避免交接处音频被重新生成。Stitcher 使用权重和为 1 的余弦交叉淡化，避免同源重叠声音被抬高音量后削波。H3 仍会联合生成后续音视频，视频上下文可能影响新生成音频；这些保护不能保证后续声音与关闭注噪时完全相同。需要原声不变时，应在成片阶段直接使用原始音轨。

只要源片段至少有 5 帧、目标长于 5 帧，节点会优先缩短重叠并保留原生遮罩，而不是因不足 39 帧退回关键帧引导。5 帧目标没有可供续写的新内容，仍走兼容路径。

### 对照官方节点的 22 帧 latent 接续

对 124 帧片段，官方节点组合取上段视频末尾 7 步、音频末尾 37 tick，分别拼到目标的第 7 步、第 37 tick 之前；这两段前缀的去噪遮罩均为 0，其余为 1。插件选择 `Native Masked AV`、`continuation_frames=22`、`Match Video Handover`、`audio_feather_ticks=0` 时复现这组张量操作。39 帧选项与旧工作流默认值继续保留。

此前插件只接受音视频时长无需取整的 `39/90/141/...`，使短片段可能退回仅有关键帧引导的 Safe Native；后者不会冻结生成区的前缀，不能提供相同的保护。现已支持全部 H3 网格，并修正兼容路径的相位对齐公式、会话提交的实际长度裁切，以及短音频上下文分别取整两端可能多复制一个 tick 的问题。

遮罩只能保护复制的上下文，不能保证新生成区域不发生身份、纹理或色彩漂移。动作迁移时，每轮应继续连接原始人物参考图和对应时间段的原始动作/深度参考；只依赖上一段生成结果，会把上一轮的偏差作为下一轮条件传递。这是生成反馈误差，与 KV 缓存量化无关：当前两种模式都不启用 DiT KV 缓存。

下一轮 `context_latent` 推荐接采样器的完整 AV 输出。先完整解码，再裁切 IMAGE/AUDIO 用于成片，避免解码后重新编码的反馈损失。现在 `Trim Prefix` 会记录裁切后的视频周期位置和音频绝对偏移；这些状态随独立 AV 文件和素材箱保存/加载，接续器会据此选择真正的尾部。旧版已经裁切、但没有这些元数据的文件无法恢复原始相位，应改用完整源 latent。重复裁切 raw latent 时会向上取到实际 latent 边界，音频也按该边界裁切，因此长度可能与像素空间的精确裁切不同。

### 固定原始参考，减少生成误差的逐轮传递

现在保留整组原生 `minimax_refs` 图片参考，作为 `minimax_identity_refs` 随采样 latent、独立 AV 文件及素材箱传递。当前条件已经提供图片参考时保持原样并更新保存的参考组；没有提供时才恢复上一段保存的图片参考。这样既能恢复缺失的多视角外观信息，也不会在主动更换人物或场景时强行追加旧参考。

若皮肤斑点或平坦背景斑纹仍逐轮增强，可使用 Config 的实验参数 `video_context_denoise`（默认 0）进行少量前缀修正对照；它会改变原生视频前缀遮罩，可能损害连续性，不能作为已验证的画质修复。原因分析、纠偏模型接法及局部频段诊断见[皮肤与背景斑纹分析](DEGRADATION_ANALYSIS_CN.md)。

`identity_latent` 表示显式增加图片参考；相同内容不会重复加入。不再默认将生成片段的首帧或首帧关键帧转换成 `ref_img`：这会改变条件结构，并可能将已有斑纹当成新的外观参考。旧文件中的单张 `minimax_identity_anchor` 可以读取，但其来源不明确，不会自动注入；需要它时显式连接 `identity_latent`。已有原始图片参考每轮均保持不变的 ref2va 工作流，不应期待重复添加参考带来改善。动作和深度参考仍需按当前时间段更新。

回归测试覆盖官方 22 帧张量等价性、完整/裁切来源的接续等价性、重复裁切、多图片参考恢复、当前参考优先，以及独立文件和素材箱的状态往返。`tests/check_native_continuation.py` 用原生 H3 遮罩、输出速度缩放、Flow 转换和四步 Euler 检查前缀保护；网络内部输出是测试替身，不加载 H3 权重。实际画质改善幅度仍需真实多段生成对照确认。

### 3. `MiniMax H3 Trim Prefix (AV Master, Zero Flicker)`

续写片段采样完成后，用它移除重复的开头。推荐在**解码后的图像和音频波形空间**裁切，可避免再次 VAE 解码导致的画面闪烁或色彩偏差。

| 输入 | 建议 |
| :--- | :--- |
| `images` | 连接当前片段完整解码后的画面。 |
| `audio` | 连接当前片段完整解码后的音频。 |
| `session` | 连接 Continuation Applier 的 `session` 输出；`trim_frames=0` 时自动使用本次实际保护帧数。 |
| `trim_frames` | 保持 `0` 即可自动裁切；只有脱离会话单独使用时才手动填写。 |
| `latent` | 可选。需要裁切 latent 时再连接；不要把它当作像素/音频裁切的替代品。 |

输出 `trimmed_images` 和 `trimmed_audio` 用于保存或交给视频合成节点。`trimmed_latent` 仅在输入了 `latent` 时才有内容。

### 4. `MiniMax H3 Clip Bin Saver (Media Pool)`

将一段生成结果归档到项目素材箱，供之后视觉化挑选和接力。

- 必填：`latent`、`project_name`、`shot_tag`、`rating`。
- `shot_tag` 是自由文本标签，例如“雨夜登场”或“Take 2”；保留 `Auto (自动编号)` 会生成 `Shot 1`、`Shot 2` 等自动编号。
- 建议连接 `images` 和可选的 `audio`，让素材箱生成首尾预览卡片并可自动编码封装视频。
- `video_file_name`：连接合成保存节点（如 `VHS_VideoCombine`）的 Filenames 输出，系统会自动将实际生成的 MP4 视频复制归档到该镜头的素材包中。
- `save_video`：默认为 `True`。开启后若传入了视频文件名则自动归档，若未传入但输入了 `images`，将自动编码生成 `video.mp4`，让素材包成为真正独立、包含潜空间和成品视听的自包含媒体池。
- 可选连接 `prompt`、`parent_clip_id`，记录提示词与多分支承接血缘。

要让下一段正常续写，请保存采样器输出的**完整**联合 AV latent；预览图则可以使用裁切后的 `trimmed_images`。

### 5. `MiniMax H3 Clip Bin Picker (Gallery Loader)`

以画廊方式浏览和载入素材箱中的片段，支持全动态视听交互。

- `project_name`：选择项目素材箱。
- `mode`：`Auto` 在素材箱为空时作为首段生成；有素材时自动接力。`Force Initial` 强制开始新首段；`Strict Chaining` 在没有可用来源时直接报错。
- `filter_rating`：按星级过滤候选镜头。
- `clip_selection`：使用 `latest` 自动选择最新片段，或填写 `clip_id` / `shot_tag` 来选择指定历史镜头。
- **全新动态视听交互**：
  - **悬停动效预览（Hover-to-Play）**：鼠标移入带有视频的卡片，自动静音循环播放动态预览，移出自动还原；
  - **全功能声画视听弹窗（Modal Player）**：点击卡片角标 `▶ MP4` 或画面中央播放按钮（亦可双击卡片），弹出视听播放弹窗，支持声音控制、时间轨拖拽、查看完整提示词与镜头参数，并可一键「设为当前接力源」；
- 输出 `latent` 接 Continuation Applier 的 `context_latent`；`tail_frame`、`first_frame` 和 `prompt` 可作为下一段的参考或提示词素材；`clip_id` 接下一次 Clip Bin Saver 的 `parent_clip_id`。

### 6. 其他辅助节点

| 节点 | 用途 |
| :--- | :--- |
| `MiniMax H3 Cache Telemetry Monitor` | 读取 `session` 并输出当前模式、片段序号和实际保护帧数。 |
| `MiniMax H3 Save AV Latent (Standalone)` | 将完整 H3 联合 AV latent 保存为 safetensors。 |
| `MiniMax H3 Load AV Latent (Standalone)` | 从历史 safetensors 载入联合 AV latent。 |
| `MiniMax H3 Safe VAE Decode (Video)` | 在首段模式下可安全处理空的上文 latent。 |
| `MiniMax H3 Safe VAE Decode (Audio)` | 在首段模式下可安全处理空的上文 latent。 |

---

## 四、推荐工作流

### 首段生成

1. 保持原有 MiniMax H3 模型、提示词、目标 latent、采样和解码节点。
2. 将 `Continuation Config` 设为 `Native Masked AV (Recommended)` 和 `39`。
3. `Continuation Applier` 的 `context_latent` 留空；将原生目标 latent 接到 `target_latent`，再将 `masked_latent` 接采样器。
4. 采样完成后解码完整画面与音频；将它们以及 `session` 接到 `Trim Prefix`。首段没有重叠，裁切节点会保持内容不变。
5. 将采样器完整 latent 与裁切后的预览画面接到 `Clip Bin Saver`，保存第一段。

### 续写下一段或创建分支

1. 在 `Clip Bin Picker` 中选择项目、评分筛选和要接力的镜头；使用 `latest` 可自动选择最新片段。
2. 将 Picker 的 `latent` 接到 Applier 的 `context_latent`，将 `clip_id` 接到 Saver 的 `parent_clip_id`。
3. 继续使用原生 H3 工作流产生新的 `target_latent`，并连接到 Applier 的 `target_latent`。
4. 采样后按首段相同方式解码，再使用 Trim Prefix 依照 `session` 自动裁掉重叠开头。
5. 归档新片段。之后可从任意历史片段再次接力，因此可以形成分支，而非只能线性续写。

最终视频的合成可使用你现有的 ComfyUI 视频合成节点或剪辑软件；本插件负责为每一段输出已裁掉重叠开头且音画同步的内容。

---

## 五、常见问题

### 为什么配置节点只有两个选项？

这是有意简化的界面。创作者只需决定续写方案与可见视频上下文长度；缓存 dtype、设备、锚帧等旧版实现细节不再暴露。

### 为什么 `shot_tag` 是文本输入框？

它是镜头的自由标签，供素材箱检索与分镜备注使用，而不是固定编号。无需手填时，保留 `Auto (自动编号)` 即可。

### 为什么 Native Masked AV 报缺少支持？

请更新 ComfyUI 至包含 MiniMax H3 AV-mask 支持的版本，完全重启后端并按 `Ctrl + F5` 刷新浏览器。短源片段或短目标片段则会自动使用 `Safe Native`。

### 为什么不需要分别连接视频和音频 latent？

MiniMax H3 在 ComfyUI 中使用联合 AV `LATENT` 封装视频和音频。续写应用节点会在内部拆分、处理并重新封装；用户只需连接一个 latent 端口。

### 如何减少显存占用？

从 `39` 帧上下文开始，按需要降低分辨率、目标时长或采样步数，并避免同时加载不使用的模型。

---

## 六、致谢与参考项目

1. **[ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)** by **[@NikoDemon80](https://github.com/NikoDemon80)**
   - 感谢其在 MiniMax H3 关键帧锚定、VAE 时空相位网格对齐、音视频开头裁切及时间缩放方面的探索与启发。
2. **[Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite)** by **[@HerrgottMargott](https://github.com/HerrgottMargott)**
   - 感谢其在 Native Masked AV、精确音视频上下文边界和独立音频保护方面提供的思路。


### 7. `MiniMax Video Chunk Slicer` (长视频智能切片工作台)

专门用于长视频（>15秒，甚至数分钟长片）的分段重绘、修图与接力编辑。具备 **按需懒加载流式切片**（零显存/内存爆炸），并提供 **跨片段画面连续性参考输出**：

| 输出端口 | 类型 | 用途说明 |
| :--- | :--- | :--- |
| `chunk_images` | `IMAGE` | 当前切片的待编辑画面序列（形状 `[B, H, W, 3]`）。 |
| `chunk_audio` | `AUDIO` | 与当前切片严格样本级时间对齐的音频波形。 |
| `slice_context` | `SLICE_CONTEXT` | 传递给下游 `MiniMaxVideoPatchReassembler` 的时序缝合上下文。 |
| `video_info` | `VHS_VIDEOINFO` | 与 VideoHelperSuite 兼容的视频规格字典。 |
| `timeline_preview` | `IMAGE` | 可直接接 `PreviewImage` 的时间轴进度条卡片。 |
| `prev_last_frame` | `IMAGE` | **[智能参考] 上一片段最后一帧**：单帧图像 `[1, H, W, 3]`。自动优先获取上一段已编辑成果，未编辑时平滑回退原片（保持旧工作流 100% 兼容）。 |
| `prev_ref_frames` | `IMAGE` | **[智能参考] 上一片段尾部多帧序列**：多帧序列 `[N, H, W, 3]`。用于动作延续或上下文前缀约束。 |
| `prev_edited_last_frame` | `IMAGE` | **[全新] 上个片段成果的最后一帧（大模型处理后）**：严格提取上一片段**经过大模型处理生成**后的成品最后一帧！专门用于下游图生视频（I2V）或风格迁移首帧参考，确保人物服饰与画风接力完全连贯！首段或尚未回填时支持平稳优雅回退。 |
| `prev_orig_last_frame` | `IMAGE` | **[全新] 上个片段原视频的最后一帧（原始未处理）**：严格提取未经任何模型修改的**原视频上一段尾帧**！专门用于 ControlNet、骨骼姿态、光流匹配或原始动作参考。 |
| `prev_edited_ref_frames` | `IMAGE` | **[全新] 上个片段成果尾部多帧序列（大模型处理后）**：多帧序列 `[N, H, W, 3]`。大模型编辑后的多帧视频接力参考。 |
| `prev_orig_ref_frames` | `IMAGE` | **[全新] 上个片段原视频尾部多帧序列（原始未处理）**：多帧序列 `[N, H, W, 3]`。原视频的多帧动作与背景序列。 |

**核心参数与可选输入：**
- `optional_prev_chunk_result`（可选输入）：若用户在 ComfyUI 中通过外部缓存或独立节点获得了上一个片段的生成成果，可以直接接驳此端口，切片器将直接优先提取该输入的尾帧作为成果参考。
- `optional_first_frame_ref`（可选输入）：若需要在首段指定外部自定义首帧（如特定的立绘图或参考原画），可连接此端口。
- `prev_ref_frames_count`：输出多帧参考序列时的帧数（默认 `16`）。
- `first_chunk_ref_mode`：首段没有前序片段时的回退策略：
  - `Current Chunk First Frame (当前片段首帧)`（推荐）：自动取当前片段首帧，确保首段也能平滑执行图生视频链路而无需手动切换分支。
  - `Black / Zero Frame (全黑空帧)`：输出全黑静止帧。

---

## 磁盘分段长视频（低内存）

新增 `MiniMax H3 Disk Video Stream` 节点，依赖 PATH 中的 FFmpeg。

1. 将当前片段解码后的 `images`、`audio` 接入该节点；可连接 `session` 自动裁切重叠前缀。如果输入已经过 Trim Prefix，请不连接 session，并保持 trim_frames=0，避免二次裁切。
2. 同一作品保持相同的 `project_name` 和 `stream_name`。每次节点执行都会追加当前片段，首段和续写都只需传入当前片段，不需要累计 IMAGE。
3. 生成期间保持 `export=False`。结束后使用同名节点，断开 images/audio，将 export=True，合成最终 MP4；该方式只导出，不重复追加末段。
4. 输出包括 manifest_path（片段清单）、video_path（导出成功后有值）、total_frames。文件位于 output/minimax_h3_bins/<项目>/.streams/<流名称>/。
5. 新作品使用新的 stream_name。相同名称会继续已有片段；重复排队相同画面也会追加一份。

该节点采用硬切拼接，不提供跨片段亮度匹配或交叉淡化。需要这些效果时使用原 Long Video Stitcher；原节点输出完整 IMAGE，因此内存占用仍随总时长增长。磁盘模式仅保留当前解码片段，编码转换每批最多 8 帧，中间音频使用 PCM，导出时统一编码 AAC。所有片段必须保持相同帧率、偶数分辨率、采样率及声道数，且音频必须始终存在或始终省略。

## 错误处理与存储约束

- 音频拼接要求前后采样率及声道数一致，出现不一致时明确报错；请先通过上游节点转换。
- Safe VAE Decode 对未连接的首段上下文仍返回空值；实际解码失败会停止并显示错误，不再伪装成空输出。
- 素材目录不接受路径型 clip_id 或符号链接目录。索引采用同一 ComfyUI 进程内的项目级锁和原子替换；不要让多个 ComfyUI 进程同时写同一项目。
- 更新后重启 ComfyUI，并在浏览器执行 Ctrl+F5 加载新版素材箱脚本。


---

# ISSUES / PRs of knoic/ComfyUI-MiniMaxH3-PrefixStream


### #2 [open] ISSUE: 无法集成二采放大
URL: https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/issues/2  by mailzwj 2026-11-05T16:51:22Z

尝试过：
```
// 尝试一：会导致下一轮衔接的时候Latent不匹配
KSampler ——> Upscaler ——> MiniMax H3 Trim Prefix (Pixel AV - Zero Flicker)
```

```
// 尝试二：放大后的视频会闪烁，且衔接变得不自然
KSampler ——> MiniMax H3 Trim Prefix (Pixel AV - Zero Flicker) ——> 使用trimmed_latent接Upscaler
```

```
// 尝试三：同样会导致衔接跳帧，不自然
MiniMax H3 Clip Bin Saver (Media Pool) ——> 使用bin_path接MiniMax H3 Clip Bin Picker (Gallery Loader & Context) ——> Upscaler
```

可有啥建议方法？

> **Comment by knoic (2026-11-06T03:35:05Z):**

二采本质上是高清重绘，需要单独管理它的接续上下文。


> **Comment by knoic (2026-11-06T03:36:58Z):**

类似这样<img width="3600" height="2240" alt="Image" src="https://github.com/user-attachments/assets/188d0fc5-4c0b-4f78-ac09-c2db6fda9897" />



### #1 [open] ISSUE: 长时间情况下 还是会累积劣化
URL: https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream/issues/1  by nihaozzt 2026-10-28T07:21:07Z

镜头间的连续性非常好了 但是在60秒单一镜头时 仍然会出现肉眼可见的人物角色劣化
