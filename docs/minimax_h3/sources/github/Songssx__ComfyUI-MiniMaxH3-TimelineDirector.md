# Songssx__ComfyUI-MiniMaxH3-TimelineDirector

Source: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector
Fetched: 2026-11-28

Description: Editable reference-media timeline director for ComfyUI MiniMax H3 Reference to Video
Stars: 800  Updated: 2026-10-21T12:31:51Z



---

## FILE: README.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/README.md)

# ComfyUI MiniMax H3 Timeline Director

Start with the [**MiniMaxH3 All-in-One Full Timeline Director workflow**](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json):
one lightweight graph covers text-to-video, image reference, audio-driven generation, video editing,
character replacement, motion transfer, digital humans, and manually segmented long-video generation.

[简体中文](README_CN.md) · English

## Headline feature: lightweight unlimited-length video generation

The plugin splits any target duration into continuous segments and completes them in one ComfyUI
execution: per-segment generation with one shared seed, direct continuation from the previous
AV-latent tail, adaptive Drift-Control video masking, Soft AV audio continuity, overlap removal, and final synchronized assembly. Extend the result by increasing
the segment count—without generic Loop nodes or duplicated sampler chains. Practical length is limited
only by local VRAM, RAM, disk space, and ComfyUI execution limits.

For long-form digital humans, standalone audio can be set to **Locked Original Audio**. The source
waveform is sliced on the timeline and injected through the native AV-latent mask/sigma path, so lip
motion remains audio-driven while the final soundtrack preserves the uploaded recording unchanged.

[Download the all-in-one workflow](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json) ·
[Chinese guide](docs/FINITE_SEGMENT_EXPANSION_CN.md) ·
[Chinese prompt specification](docs/MiniMax_H3_循环分段提示词_Agent规范.md)

[![Lightweight unlimited-length workflow](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/infinite-workflow.webp)](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json)

## SelfLift two-stage sampling: fast 75% low-res + 25% high-res generation

The Material Planner now integrates a MiniMax H3 adaptation of SelfLift progressive-resolution
sampling. Enable **Two-stage sampling**, select an H3 Latent Upscaler from
`ComfyUI/models/latent_upscale_models/`, and set **High-resolution sampling steps**. A recommended
starting point is `25%` of the scheduler's total steps: for an 8-step schedule, set 2 high-resolution
steps to run `6 low-resolution steps + latent lift + 2 high-resolution steps`. The high-resolution
value must be greater than zero and lower than the scheduler's total step count.

![Two-stage toggle, model selector, and high-resolution step control](docs/images/two-stage-controls.png)

This is not a per-segment upscale shortcut. Every later segment carries both the preceding native
low-resolution latent tail and the final high-resolution context, with Drift-Control applied at both
resolution stages. The resulting long-video path avoids progressive quality loss from repeated
resizing or VAE round trips and removes the blur, white flashes, and visible seams normally associated
with multi-segment generation.

Creator benchmark: a `1536×832`, `29-second` video rendered directly in approximately `10 minutes`
with a `75%` low-resolution / `25%` high-resolution schedule. Actual speed depends on GPU, VRAM,
model, step count, and reference complexity. When standalone audio is locked or reference-video source
audio is enabled, the native zero-denoise AV path restores one continuous source waveform; creator
tests retain `99%+` content and timing consistency. The final MP4 saver may still re-encode audio.

One workflow covers text-to-video, image-to-video, audio-reference generation, image plus audio,
reference-video editing, character replacement, motion transfer, digital-human lip sync, and
multi-segment long videos without visible seams or progressive degradation.

### Two directly generated, approximately one-minute examples

Both videos were produced in one plugin execution and are `52.625 seconds / 1263 frames / 24fps`.
Click a poster to play or download the original MP4. All media is hosted as GitHub Release assets, so
it adds nothing to the plugin clone or installation size.

| Finite direct-latent continuation | References with a 48-frame overlap |
| --- | --- |
| [![Play example one](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/case-finite-segments-60s.webp)](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/H3_finite_segments_60s.mp4) | [![Play example two](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/case-reference-overlap-60s.webp)](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/H3_reference_overlap48_60s.mp4) |

<p align="center">
  <img src="docs/images/creator-wecom.webp" alt="Creator WeCom contact card" width="360">
</p>

<p align="center">
  Creator: <strong>Shi Xiongsong</strong><br>
  <a href="https://space.bilibili.com/219572544?spm_id_from=333.40164.0.0">Bilibili</a>
  ·
  <a href="https://www.youtube.com/@shixiongsong">YouTube</a>
</p>

## Feature timeline

- **2026-08** — Introduced the editable media timeline, reference orchestration, per-segment prompts, and the first finite long-video path.
- **2026-09** — Added the all-in-one workflow, locked-original-audio avatars, bundled SelfLift two-stage sampling, low/high-resolution continuation, local presets, and native ComfyUI Loop nodes.
- **2026-09-15** — Merged community [PR #9](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/9), fixing Nodes 2.0 DOM-widget width synchronization. Thanks to `permissionnine9`.
- **2026-10-08** — Merged community [PR #17](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/17) (`spill-seg-offload`), adding per-segment frame eviction and optional FP16 merged-frame storage. Thanks to `FangZhangTNS`.
- **2026-10-14** — Added **MiniMax H3 Low-Memory Segment Video Sampler** and its example workflow. It decodes, encodes, and releases each segment before encoded-video assembly; uploaded image/audio delete controls now stay clickable above drag surfaces.

An editable reference-media timeline for ComfyUI's native **MiniMax H3 Reference to Video** workflow. It brings reference videos, paired soundtracks, fixed Guides, standalone images, and standalone audio into one compact editing surface.

> Video-generation agents should read the [Chinese segmented long-video guide](docs/AGENT_LONG_VIDEO_GUIDE_CN.md).

## Highlights

- Multi-clip timeline with move, trim, split, delete, snapping, and numeric positioning.
- Independent reference width/height controls for every uploaded video. Sizes align to 32 pixels, reducing reference-latent work without changing the generated canvas; new uploads default to Editable Reference.
- Only media intersecting the cyan generation range participates in the current reference or Guide plan.
- Three per-clip modes: `Fixed Guide`, `Editable Reference`, and `Boundary Only`.
- Native text-to-video: with no uploaded media and no segment windows, Finite Segment Sampling internally creates a standard empty H3 AV latent from the Global Prompt and current GEN duration. Manual windows extend the same path to long-form T2V.
- Bound source audio follows video edits and can be disabled independently.
- Silent low-resolution monitoring proxies up to `480×270 / 12fps`.
- Multi-select, external file drop, deletion, and drag reordering for image/audio bins. The global
  standalone image and audio libraries have no upload-count limit; each generated segment still follows
  MiniMax H3's limit of at most 9 reference images and 3 reference audio clips.
- Locked-original-audio mode for long-video digital humans and singing avatars, with segment-aware AV
  latent locking and unchanged source-audio assembly.
- Stable `<Picture N>`, `<Video N>`, and `<Audio N>` ordering from UI to H3 inputs.
- Global and per-segment prompts: the global prompt is reused only when all segment prompts are empty; entering any segment prompt requires completing every segment and disables the global prompt.
- Decode-time resizing to the node's `width × height` for VRAM protection.
- Built-in two-stage sampling for every segment, including native low-resolution tail carry, learned H3 latent lifting, and high-resolution Drift-Control continuation; `comfyui-SelfLift` is not required.
- In both one-stage and two-stage reference-video generation, enabled Video Original Audio is automatically locked through the native AV path and restored as one continuous original waveform. Disabling it fixes the AV audio stream and final master to silence; an explicitly uploaded locked audio asset takes priority.
- Segmented digital-human generation with locked source audio on the native AV mask/sigma path, preserving the soundtrack and lip-sync guidance through two-stage sampling.
- Separate merged outputs for timeline soundtracks and standalone reference audio.
- Timeline state is serialized into the ComfyUI workflow JSON.

## Main long-video workflow nodes

| Node | Purpose |
| --- | --- |
| **MiniMax H3 Material Planner** | Edits media and outputs a compact H3 plan plus an ordered Omni media bundle. |
| **MiniMax H3 Omni Media-Bundle Prompt Bridge** | Sends the bundle to an installed Prompt Rewriter Omni backend and returns only `rewritten_prompt`. |
| **MiniMax H3 Finite Segment Sampling** | Expands an acyclic graph for direct-latent continuation, masking, sampling, deduplication, and assembly. |
| **MiniMax H3 Low-Memory Segment Video Sampler** | Decodes and encodes one segment at a time in ComfyUI's temporary folder, then joins encoded segments as `VIDEO`; no latent checkpoints or rerun state. |
| **MiniMax H3 native-loop node set** | Splits segment selection, continuation preparation, sampling, and assembly for free composition inside ComfyUI `Start Loop / End Loop`. |
| **MiniMax H3 Preset Export** | Connects to Material Plan or Segment Plan and saves configuration-only or complete media-bearing preset folders. |
| **MiniMax H3 Local Preset Loader** | Finds local presets and feeds the Material Planner's new Import Preset input. |
| **MiniMax H3 Timeline Director (Compatibility)** | Preserves the original all-in-one workflow and older saved workflows. |

Long-video generation needs only **Material Planner Segment Plan → Finite Segment Sampling**. The Plan Encoder is also public now so native Loop workflows can place it explicitly inside the loop body.

For long or high-resolution output, use **MiniMax H3 Low-Memory Segment Video Sampler** and connect its
`Final Video` directly to `SaveVideo`. It never creates one full RGB-frame tensor: each decoded segment is
immediately encoded to a temporary MP4 and released, then the encoded sources are joined. Segment files are
removed after assembly, the final temporary file is cleaned when its VIDEO object is released, and stale
24-hour spill folders are removed on the next run. See
[the all-in-one low-memory workflow](example_workflows/MiniMaxH3全功能合一低内存视频输出工作流.json).

### Local presets

Preset Export supports `configuration` (timeline and creative settings only) and `complete` (also copies
referenced video, image, and audio files). It creates an ordinary folder under
`ComfyUI/output/MiniMaxH3_Presets/`; no ZIP is required. Copy that folder to the same location on another
installation, refresh Local Preset Loader, and connect it to Material Planner's **Import Preset** input. The
first run writes all media, segment timings, and per-segment assignments back into the planner UI. The loader
can then be disconnected without losing the imported state. **Import preset** remains available for loading it
into the UI immediately before a run.

Presets intentionally omit workflows, model/LoRA selections, plugin versions, and ComfyUI versions. A complete
preset materializes media into the fixed `ComfyUI/input/minimax_h3_timeline_director/presets/` subtree while
leaving the shared preset folder unchanged.

### Native ComfyUI Loop workflow

The split nodes can drive ComfyUI's native generic loop. Connect **Initialize Segment Loop** state to
`Start Loop.initial_iteration_value` and its count to `num_iterations`. Inside the loop, use
`iteration_index → Select Loop Segment → Plan Encoder → Prepare Loop Segment`, then place either a
normal sampler or the bundled H3 two-stage sampler and decode the result. Feed **Accumulate Loop Segment**
state to both `End Loop.next_iteration_value` and `output_value`; connect `End Loop.outputs` to
**Finish Segment Loop**. The legacy one-node Finite Segment Sampling path remains available for saved workflows.

The split architecture avoids a ComfyUI dependency cycle:

```text
Material Planner ──Omni bundle──> Omni Prompt Bridge ──rewritten_prompt──> Plan Encoder
       └────────────────────H3 plan─────────────────────────────────────> Plan Encoder
```

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector.git
```

Restart ComfyUI and search for `MiniMax H3`.

The two-stage runtime is bundled with this plugin; do not install `comfyui-SelfLift` separately. Two-stage sampling still requires a compatible MiniMax H3 latent-upscaler checkpoint under `ComfyUI/models/latent_upscale_models/`; the Material Planner lists models from that standard directory automatically.

The source UI is English. Simplified Chinese is provided through ComfyUI's official localization
system and follows the language selected in ComfyUI settings; restart or reload the frontend after
changing the locale.

### Requirements

- A recent ComfyUI build with the native MiniMax H3 nodes; `MiniMaxH3AddGuide` is additionally required only when Guides are used.
- MiniMax H3 Ref2VA model, CLIP, video VAE, and audio VAE.
- Python 3.10 or newer.
- ComfyUI's `imageio-ffmpeg` package for low-resolution preview proxies.

No extra pip dependency is declared. The plugin uses PyAV, Pillow, NumPy, PyTorch, torchaudio, aiohttp, and imageio-ffmpeg normally included with a compatible ComfyUI installation.

## Example workflows

The plugin ships four example workflows: the normal all-in-one entry point, low-memory long-video output, Omni prompt expansion, and finite-segment generation with ComfyUI's native Loop nodes.

### 1. All-in-One Full Timeline Director (recommended)

[Download workflow](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json)

This is the **all-task, lightweight, no-rewiring** workflow. Configure only the Material Planner's media, time windows, and prompts to run:

- Text-to-video with no media, as one segment.
- Long-form text-to-video with no media and multiple segments.
- Single-segment image-reference generation.
- Multi-segment image-reference generation.
- Single-segment image-and-audio reference generation.
- Multi-segment image-and-audio generation and digital-human lip sync.
- Single-segment editable-video reference, character replacement, and motion transfer.
- Manually segmented long-video reference generation, character replacement, motion transfer, and final AV assembly.

Connect the Material Planner's **Segment Plan** directly to **MiniMax H3 Finite Segment Sampling**. With no uploaded media it creates an empty AV latent; with references it encodes each segment's assigned media. Drag GEN windows to define duration and seam overlap, reuse one Global Prompt, or enter complete per-segment prompts. The plugin handles legal-frame alignment, Drift-Control, Soft AV, overlap removal, tail trimming, and final assembly.

> The plugin does not auto-segment by reference-media duration or infer whether character identity should continue. Segment ranges, overlaps, and assignments are explicitly controlled on the timeline. For character replacement, use **Editable Reference** in most cases.

### 2. All-in-One low-memory video output

[Download workflow](example_workflows/MiniMaxH3全功能合一低内存视频输出工作流.json)

This keeps the complete Material Planner and the same one-/two-stage seam logic, but uses **MiniMax H3 Low-Memory Segment Video Sampler**. Every finished segment is decoded, encoded to temporary video, and released immediately; the node sends `VIDEO` directly to `SaveVideo`, making it suitable for longer or higher-resolution runs.

### 3. Timeline planning with Prompt generation

[Download workflow](example_workflows/MiniMax_H3时间规划+Prompt提示词生成.json)

This variant adds the **MiniMax H3 Omni Media-Bundle Prompt Bridge**, allowing Prompt Rewriter Omni to inspect ordered images, videos, and audio before expanding an H3 prompt. Use it when multimodal material understanding should precede H3 generation.

> This workflow requires [MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI). Follow that project for model, quantization, and VRAM requirements.

### 4. Native Loop finite-segment example

[Download workflow](example_workflows/MiniMaxH3原生Loop有限分段示例工作流.json)

This workflow combines the split segment nodes with ComfyUI's native `Start Loop / End Loop`, making it easy to insert custom processing inside the loop body. It is shipped as a read-only example and retains the user's original configuration.

## Basic usage

1. Set `width`, `height`, and `generation_seconds`.
2. Add video, image, and audio files using the toolbar or direct file drop.
3. Move, trim, or split video clips, then place the cyan range over the interval to generate.
4. Select a purpose for each video:
   - `Fixed Guide` anchors the overlap at its generated-frame positions.
   - `Editable Reference` sends it as `<Video N>` without hard-locking the original subject.
   - `Boundary Only` anchors only the first and last overlap frames.
5. Enable or disable paired video soundtracks as needed.
6. Verify the reference labels at the bottom and run the connected encoder or prompt workflow.

Videos are numbered left-to-right by their intersections with the cyan range. Standalone images and audio follow their visible bin order; drag reordering immediately updates the underlying H3 order.

## Segmented long-video generation

Generate long videos in overlapping segments. Use the previous segment's final shot as the next segment's opening Guide, and describe that overlap as `Shot 1` before new content. When assembling segments, remove the repeated Guide interval from the later segment. See the [Chinese agent guide](docs/AGENT_LONG_VIDEO_GUIDE_CN.md) for the full procedure.

### Finite direct-latent continuation

Finite sampling carries the previous sampled AV latent tail directly into the next opening and
avoids an RGB decode/re-encode round trip. Drift-Control is always active and has no user-facing mode
selector. The requested overlap is aligned down to H3's legal temporal grid (for example, 24 becomes
22 and 48 becomes 39), and that same actual value drives latent carry, decoded trimming, and assembly.
The mask adapts both to the aligned overlap's video-token count and to the connected sampler's sigma
schedule, including accelerated 4-step and 8-step schedules. It dynamically re-noises only the disposable video prefix while keeping the seam-side
latent clean. When audio continuation is enabled, the carried overlap stays exact until its final eight
audio-latent ticks, where a half-cosine Soft AV mask releases it into newly generated sound. Assembly
replaces the preceding audio tail with this incoming Soft AV overlap so the transition is retained in the final output. All segments use
exactly the seed shown on the sampling node. The old generic-loop helper nodes and PR #15923 dependency
have been removed.

Drift-Control AV is adapted from
[ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)
under GPL-3.0. It remains experimental and is intended for same-shot long-chain comparisons.

## Credits

- Special thanks to [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) for the SelfLift progressive-resolution sampling technology. This plugin deeply adapts that work for MiniMax H3 AV latents, native audio mask/sigma locking, low- and high-resolution long-video continuation, and seam handling, and bundles the required runtime.
- MiniMax H3 learned latent-upscaler compatibility references [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).
- The Omni bridge and prompt-generation workflow reference and adapt [pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI).
- See [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) for the official model and prompt guidance.
- Thanks to the maintainers of ComfyUI's native MiniMax H3 and Guide nodes.

## License

[GPL-3.0](LICENSE)


---

## FILE: README_CN.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/README_CN.md)

# ComfyUI MiniMax H3 时间线导演台

简体中文 · [English](README.md)

首选入口是 [**MiniMaxH3 全功能合一完全体导演台工作流**](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json)：
一套轻量工作流同时覆盖文生视频、参考图生视频、参考音频驱动、视频编辑、人物替换、动作迁移、数字人以及手动分段长视频生成，无需为不同任务更换节点图。

## 核心能力：轻量级无限时长视频生成

插件可以把任意目标时长拆成多个连续片段，在一次 ComfyUI 执行中自动完成逐段生成、
所有分段使用同一个种子、上一段 AV Latent 尾部续接、自适应 Drift-Control 视频遮罩、Soft AV 音频连续性、重叠帧去除以及最终音画合并。
只需增加分段数量即可继续延长视频，不依赖已被关闭的通用 Loop PR，也不需要在画布上
手工复制多套采样节点。实际可生成长度只受本机显存、内存、磁盘空间及 ComfyUI 单次执行能力限制。

长视频数字人可将独立音频设为**原音锁定**。插件会按时间线切分原始波形，并通过原生 AV Latent
的 mask/sigma 路径逐段注入，让人物口型继续受音频驱动，同时最终成片完整保留上传的原始音频，不重新生成或改写声音。

[下载全功能合一工作流](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json) ·
[使用说明](docs/FINITE_SEGMENT_EXPANSION_CN.md) ·
[分段提示词 Agent 规范](docs/MiniMax_H3_循环分段提示词_Agent规范.md)

[![轻量级无限时长工作流](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/infinite-workflow.webp)](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json)

## SelfLift 二采：75% 低清 + 25% 高清高速生成

素材规划台已经集成适用于 MiniMax H3 的 SelfLift 渐进分辨率二采。勾选 **开启二采**，选择
`ComfyUI/models/latent_upscale_models/` 中的 H3 Latent Upscaler，再填写 **高清采样步数**即可。
推荐让高清阶段约占总采样步数的 `25%`：例如总步数为 8 时设置高清采样步数为 2，实际执行
`6 步低分辨率 + Latent 放大 + 2 步高分辨率`。高清采样步数必须大于 0 且小于调度器总步数。

![素材规划台二采开关、模型和高清采样步数](docs/images/two-stage-controls.png)

二采不是简单地对每段单独放大：段与段之间会同时延续上一段的原生低清 Latent 尾部和最终
高清 Latent 上下文，并在低清、高解析度两个阶段分别执行 Drift-Control。这样可以避免长视频
逐段缩放、反复编解码造成的清晰度劣化，并消除多段生成常见的模糊、闪白和可见接缝。

作者实测案例：`1536×832 / 29 秒`视频约 `10 分钟`直出，使用 `75%` 低清采样与 `25%`
高清采样；实际速度会随显卡、显存、模型、步数和参考素材复杂度变化。锁定独立音频或开启
参考视频原声后，音频进入原生 AV 零去噪锁定路径并以连续原波形作为最终主轨，实测内容与
时序保持 `99%+` 一致；最终 MP4 封装仍可能按保存节点设置重新编码音频。

同一套工作流可以完成文生视频、参考图生视频、参考音频生视频、图片加音频、参考视频编辑、
人物替换、动作迁移、数字人对口型以及无明显劣化、无可见接缝的多段长视频生成。

### 约一分钟直接生成案例

下面两段均为插件一次执行直接生成的 `52.625 秒 / 1263 帧 / 24fps` 成片。点击缩略图播放或下载原始 MP4；
视频和缩略图存放在 GitHub Release，不会增加插件克隆与安装体积。

| 案例一：有限分段 Latent 续写 | 案例二：参考素材 + 48 帧重叠续写 |
| --- | --- |
| [![播放案例一](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/case-finite-segments-60s.webp)](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/H3_finite_segments_60s.mp4) | [![播放案例二](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/case-reference-overlap-60s.webp)](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/releases/download/v0.6.0/H3_reference_overlap48_60s.mp4) |

<p align="center">
  <img src="docs/images/creator-wecom.webp" alt="创作者企业微信联系卡片" width="360">
</p>

<p align="center">
  作者：<strong>石兄松 / Shi Xiongsong</strong><br>
  <a href="https://space.bilibili.com/219572544?spm_id_from=333.40164.0.0">哔哩哔哩</a>
  ·
  <a href="https://www.youtube.com/@shixiongsong">YouTube</a>
</p>

## 功能更新时间线

- **2026-08**：发布素材时间线、参考素材编排、分段提示词与有限分段长视频基础能力。
- **2026-09**：加入全功能合一工作流、原音锁定数字人、SelfLift 二采、低清/高清双层续接、本地预设和 ComfyUI 原生 Loop 节点组。
- **2026-09-15**：合并社区 [PR #9](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/9)，修复 Nodes 2.0 下时间线 DOM 控件宽度同步；感谢 `permissionnine9` 的贡献。
- **2026-10-08**：合并社区 [PR #17](https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/17)（`spill-seg-offload`），加入逐段帧卸载及可选 FP16 合并缓存；感谢 `FangZhangTNS` 的贡献。
- **2026-10-14**：新增“MiniMax H3 有限分段采样(低内存版本)”与配套示例工作流，逐段解码、编码和清理后再合并视频；同时修复图片/音频删除按钮被拖拽层遮挡的问题。

一个为 ComfyUI 原生 **MiniMax H3 Reference to Video** 工作流设计的可视化参考素材时间线。它将参考视频、视频原声、固定 Guide、独立图片和独立音频集中到一个类似剪辑软件的界面中。

> 视频生成 Agent 请阅读：[长视频分段生成 Agent 操作规范](docs/AGENT_LONG_VIDEO_GUIDE_CN.md)

## 主要功能

- 多视频时间线：移动、裁剪、分段、删除、吸附和精确数值定位。
- 每个上传视频可独立设置参考宽高；参考尺寸自动对齐到 32，可在不改变最终生成分辨率的情况下减少参考 Latent 计算量。新上传视频默认使用“可编辑参考”。
- 青色生成选区：只有与选区重叠的素材区间参与本次视频参考或 Guide。
- 三种视频用途：`固定Guide`、`可编辑参考`、`仅固定边界`。
- 纯文生视频：不上传任何素材且不启用分段时，素材规划台的全局提示词与当前 GEN 时长会由有限分段采样内部创建标准 H3 空 AV latent；启用手动时间框后也可生成文生长视频。
- 原声同步：视频移动和裁剪时原声保持绑定；无论一采还是二采，“视频原声”开启都会自动进入原声锁定的原生 AV 路径，并在最终输出中使用连续原波形；关闭后固定为静音，不生成替代声音。显式上传的锁定音频优先级更高。
- 低清预览：最高 `480×270 / 12fps` 无声代理，红色播放头可拖动预览。
- 独立素材箱：图片和音频支持多选、外部拖入、删除及拖拽排序；全局图片库与音频库不限制上传总数，
  每个实际生成分段仍遵循 MiniMax H3 的上限，最多使用 9 张参考图和 3 段参考音频。
- 原音锁定：支持长视频数字人和人物唱歌，按分段锁定 AV Latent，并在最终合并时保持源音频内容不变。
- 稳定编号：界面中的 `<Picture 1>`、`<Video 1>`、`<Audio 1>` 与 H3 输入顺序一致。
- 全局与分段提示词：全局提示词可复用于全部片段；任意分段提示词启用后，全局提示词自动失效且要求补全每一段。
- 显存保护：素材在解码时按节点 `width × height` 调整分辨率。
- 内置二采：每个分段可先低清采样、再通过 H3 Latent Upscaler 放大并完成高清采样；长视频会同时延续上一段原生低清尾部和高清 Drift-Control 上下文，无需安装 `comfyui-SelfLift`。
- 长视频数字人：锁定原声走原生 AV mask/sigma 路径，在分段和二采模式下保持音频内容及口型引导。
- 音频输出：可分别输出时间线视频原声合并结果和独立参考音频合并结果。
- 状态保存：时间线编辑状态会写入 ComfyUI 工作流 JSON。

## 长视频主工作流节点

| 节点 | 用途 |
| --- | --- |
| **MiniMax H3 素材规划台** | 编辑素材并输出紧凑的 `素材规划` 与 `Omni素材包`。 |
| **MiniMax H3 Omni 素材包提示词桥** | 将规划台素材送入已安装的 Prompt Rewriter Omni，并只输出 `rewritten_prompt`。 |
| **MiniMax H3 有限分段采样** | 展开普通无环执行图，完成直接 Latent 续写、时间遮罩、采样、去重和合并。 |
| **MiniMax H3 有限分段采样(低内存版本)** | 每段单独解码并写入 ComfyUI 临时目录，最终直接合并为 `VIDEO`；不保存 Latent，也不提供片段重跑。 |
| **MiniMax H3 原生循环分段节点组** | 把分段选择、续接准备、采样和结果合并拆开，可在 ComfyUI `Start Loop / End Loop` 中自由组合一采、二采和自定义处理。 |
| **MiniMax H3 导出预设** | 连接“素材规划”或“分段规划”，保存纯配置预设，或保存包含引用视频、图片和音频的完整预设文件夹。 |
| **MiniMax H3 本地预设加载器** | 检索本地预设并连接到素材规划台新增的“导入预设”输入。 |
| **MiniMax H3 时间线导演台（兼容）** | 保留原先的一体化工作流和旧工作流兼容性。 |

长视频生成只需连接 **素材规划台「分段规划」→ 有限分段采样「有限分段规划」**。
`MiniMax H3 规划编码器`现在作为公开节点显示，可用于原生 Loop 循环体；旧的一键采样器仍会在内部复用它。

超长或高分辨率视频建议改用 **MiniMax H3 有限分段采样(低内存版本)**，并把它的 `Final Video`
直接连接 `SaveVideo`。该节点不会构造整条 RGB 帧张量：每段解码完成后立即编码为临时 MP4，释放帧内存，
最后在编码层合并。分段临时文件会在合并后删除，最终临时文件在 VIDEO 对象释放时清理；超过 24 小时的
异常遗留目录也会在下次运行时自动清理。可直接打开
[全功能合一低内存视频输出工作流](example_workflows/MiniMaxH3全功能合一低内存视频输出工作流.json)。

### 本地预设

“导出预设”提供两种保存模式：`configuration` 只保存时间线与创作配置；`complete` 还会把引用的
视频、图片和音频复制到同一预设文件夹。结果保存在 `ComfyUI/output/MiniMaxH3_Presets/预设名/`，
不打包 ZIP。把整个文件夹复制到另一套安装的同名目录后，刷新“本地预设加载器”即可使用；
将其输出连接到素材规划台的“导入预设”后运行一次，完整素材、分段、时长和逐段分配会自动写回
规划台并显示在界面中。随后可以断开预设加载器，规划台会保留已经导入的状态并允许继续修改；
也可以在运行前点击规划台顶部“导入预设”立即预览和编辑。

预设只记录可移植的素材编排、提示词、分段、音频锁定和二采开关等创作信息，不记录工作流、
模型/LoRA 选择、插件版本或 ComfyUI 版本。完整预设在载入时会把素材安全复制到固定的
`ComfyUI/input/minimax_h3_timeline_director/presets/` 子目录；原预设文件夹保持不变。

### 使用 ComfyUI 原生 Loop

新版 ComfyUI 可使用下面的拆分路径。`初始化分段循环`输出的分段数量连接 `Start Loop`
的 `num_iterations`，初始状态连接 `initial_iteration_value`；`合并循环分段`输出的下一循环状态
同时连接 `End Loop` 的 `next_iteration_value` 和 `output_value`，最后由 `完成分段循环`取出结果。

```text
分段规划 ──> 初始化分段循环 ──初始状态──> Start Loop
                  └─分段数量───────────> num_iterations

Start Loop.iteration_index ──> 选择循环分段 ──> 规划编码器
Start Loop.current_iteration_value ───────────────┐
规划编码器 + 模型 + Sigmas ──> 准备循环分段 ──> 用户自选采样/二采 ──> VAE 解码
                                                  └─> 合并循环分段 ──下一状态─> End Loop

End Loop.outputs ──> 完成分段循环 ──> 完整画面 / 音频 / 末段 Latent
```

循环体中的采样可以使用普通 `SamplerCustomAdvanced`，也可以使用插件内置的 H3 二采节点，
还可以插入预览、保存和用户自己的处理节点。原来的 **有限分段采样** 继续保留，旧工作流无需修改。

拆分节点可以避免“素材输出连接到前置提示词重写器，再返回同一编码节点”产生的循环：

```text
素材规划台 ──Omni素材包──> Omni提示词桥 ──rewritten_prompt──> 规划编码器
     └────────────────素材规划──────────────────────────────> 规划编码器
```

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector.git
```

重启 ComfyUI 后，搜索 `MiniMax H3` 即可找到节点。

二采运行模块已经内置在本插件中，不需要另外安装 `comfyui-SelfLift`。开启二采仍需自行把兼容的 MiniMax H3 Latent Upscaler 模型放入 `ComfyUI/models/latent_upscale_models/`，素材规划台会自动列出该目录中的模型。

插件以英文作为基础 UI，并通过 ComfyUI 官方本地化机制提供简体中文。它会跟随 ComfyUI
设置中选择的界面语言；切换语言后请刷新前端或重启 ComfyUI。

### 环境要求

- 较新的 ComfyUI，包含原生 MiniMax H3 节点；使用固定 Guide 时还需 `MiniMaxH3AddGuide`。
- MiniMax H3 Ref2VA 对应模型、CLIP、视频 VAE 和音频 VAE。
- Python 3.10 或更高版本。
- 低清代理依赖 ComfyUI 环境中的 `imageio-ffmpeg`。

插件不额外声明 pip 依赖，使用兼容 ComfyUI 通常已经包含的 PyAV、Pillow、NumPy、PyTorch、torchaudio、aiohttp 和 imageio-ffmpeg。

## 示例工作流

插件内置下面四个示例工作流：日常全功能入口、低内存长视频输出、Omni 自动扩写提示词，以及使用 ComfyUI 原生 Loop 自由搭建有限分段生成。

### 1. 全功能合一完全体导演台（推荐）

[下载工作流](example_workflows/MiniMaxH3全功能合一完全体导演台工作流.json)

这是一套 **全任务、轻量级、无需换图** 的主工作流，仅通过素材规划台的素材、时间框和提示词配置，即可完成：

- 无素材单段文生视频。
- 无素材多段文生长视频。
- 图片参考单段生成。
- 图片参考多段生成。
- 图片 + 音频单段参考生成。
- 图片 + 音频多段参考生成及数字人对口型。
- 视频单段可编辑参考、人物替换和动作迁移。
- 长视频手动分段参考生成、人物替换、动作迁移及最终音画合并。

素材规划台的 **分段规划** 直接连接 **MiniMax H3 有限分段采样**。未上传素材时会建立空 AV Latent；上传素材后则按每段分配关系编码参考。用户可以拖动 GEN 时间框定义各段长度与接缝重叠，使用全局提示词复用同一要求，也可以为每段填写独立提示词。插件负责合法帧数对齐、Drift-Control、Soft AV、重叠去除、尾部裁剪和最终合并。

> 插件不会按参考媒体时长自动分段，也不会替用户判断角色是否应当续接。分段范围、相邻段重叠和每段素材均由时间线配置决定；人物替换通常应把原视频设为 `可编辑参考`。

### 2. 全功能合一低内存视频输出

[下载工作流](example_workflows/MiniMaxH3全功能合一低内存视频输出工作流.json)

保留全功能规划台与一采/二采接缝逻辑，但改用 **MiniMax H3 有限分段采样(低内存版本)**。每段完成后立即解码、编码为临时视频并释放帧内存，最终直接输出 `VIDEO` 给 `SaveVideo`，适合长时长或高分辨率生成。

### 3. 时间规划 + Prompt 提示词生成

[下载工作流](example_workflows/MiniMax_H3时间规划+Prompt提示词生成.json)

在素材规划台基础上加入 **MiniMax H3 Omni 素材包提示词桥**，让 Prompt Rewriter Omni 同时读取有序图片、视频和音频，为当前任务扩写 H3 提示词。适合先由多模态模型理解素材，再进入 H3 生成的工作方式。

> 使用该工作流前必须安装 [MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)。模型、量化方式及显存要求请参考该项目说明。

### 4. 原生 Loop 有限分段示例

[下载工作流](example_workflows/MiniMaxH3原生Loop有限分段示例工作流.json)

使用拆分后的分段节点和 ComfyUI 原生 `Start Loop / End Loop` 组合有限分段生成，便于在循环体中插入自定义处理。该文件是插件随附的只读示例，保留用户配置的原始内容。

## 基本使用方法

1. 设置 `width`、`height` 和 `generation_seconds`。
2. 用 `+ 视频`、`+ 图片`、`+ 音频` 添加素材；也可以把文件直接拖入控件。
3. 移动、裁剪或分段视频，并把青色选区放到本次需要生成的区间。
4. 为每段视频选择用途：
   - `固定Guide`：重叠部分按生成时间位置固定，适合续写和保持运动。
   - `可编辑参考`：只作为 `<Video N>` 参考，不硬锁原人物，适合人物或风格替换。
   - `仅固定边界`：仅固定重叠区首尾帧。
5. 按需设置“视频原声”：开启即在一采和二采中锁定原音，关闭即静音输出。
6. 检查底部提示词编号，再连接对应编码或提示词工作流运行。

视频编号按照青色选区内重叠片段在时间线上的从左到右顺序生成。独立图片和独立音频按素材箱显示顺序生成编号，拖拽排序后会同步更新底层输入顺序。

## 长视频分段

推荐分段生成并保留短重叠区：下一段使用上一段末尾镜头作为开头 Guide，提示词中的 `Shot 1` 必须先描述这段重叠参考，再描述新内容。合并时删除后一段重复的固定 Guide 区域。完整规则见 [Agent 操作规范](docs/AGENT_LONG_VIDEO_GUIDE_CN.md)。

### 有限分段直接 Latent 续写

有限分段采样会把上一段 sampled AV Latent 的尾部直接放到下一段开头，避开
`RGB Decode → VAE Encode` 往返。有限分段采样固定使用 Drift-Control，不再提供续接模式
选项。用户填写的重叠帧会向下对齐到 H3 合法时间网格，例如24变成22、48变成39；Latent
续接、解码后裁剪和最终合并始终使用同一个实际重叠值。视频遮罩会同时适配实际重叠对应的
Latent 时间步数量和外部采样器的真实 sigma 调度，可用于加速模型常用的4步、8步以及常规20步：仅对临时视频前缀按采样步动态匹配噪声，保持接缝侧 latent 干净；开启音频
续接时，重叠音频前部保持完全保护，最后8个音频 latent tick 使用半余弦 Soft AV 遮罩逐渐
释放到新生成声音。合并时由后一段的 Soft AV 重叠音频替换前一段末尾，因此渐变会保留在
最终音轨中。所有分段始终使用采样节点设置的同一个种子。旧的
`循环分段提示词`、`Latent 循环续段`、`循环片段去重`以及对 PR #15923 的依赖已经移除。

Drift-Control AV 基于 GPL-3.0 项目
[ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)
的实验实现进行适配，目前仅建议用于同镜头长链对照测试。

## 致谢与参考

- 特别感谢 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) 提供的 SelfLift 渐进分辨率二采技术。本插件在其思路和实现基础上针对 MiniMax H3 AV Latent、原生音频 mask/sigma、低清/高清长视频续接及接缝处理进行了深度适配，并将运行模块内置到插件中。
- MiniMax H3 学习型 Latent Upscaler 的结构兼容参考了 [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)。
- 本项目的 Omni 提示词桥及提示词工作流参考并适配了 [pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI)。
- MiniMax H3 提示词结构与模型使用方式请参考 [MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3)。
- 感谢 ComfyUI 原生 MiniMax H3 与 Guide 节点的维护者。

## 许可

[GPL-3.0](LICENSE)


---

## FILE: docs/AGENT_LONG_VIDEO_GUIDE_CN.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/docs/AGENT_LONG_VIDEO_GUIDE_CN.md)

# MiniMax H3 时间线导演台：长视频生成 Agent 操作规范

> 本文面向负责策划、生成、检查和拼接长视频的视频生成 Agent。它不是普通用户界面介绍，而是一份执行规范。

## 1. Agent 的目标

使用 MiniMax H3 Ref2VA 和本插件，把长视频任务拆成多次 4–15 秒的短片生成，并通过相邻片段之间的固定重叠区保持：

- 人物身份、服装、姿态和站位连续；
- 场景、光照、天气与空间关系连续；
- 镜头运动、动作速度与剪辑节奏连续；
- 对白、环境声、音乐和音色连续。

最终成片不是简单首尾相接。除第一段以外，每个生成片段的开头都包含一段已经存在于上一段末尾的固定 Guide；合并时必须删除这段重复内容。

## 2. 开始前的强制要求

### 2.1 必须使用 MiniMax H3 官方提示词 Skill

每一段提示词都必须遵循 MiniMax H3 官方 `h3-prompt-writing` Skill：

- 官方仓库：<https://github.com/MiniMax-AI/MiniMax-H3>
- 安装命令：

  ```bash
  npx skills add https://github.com/MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing
  ```

如果当前 Agent 环境不支持上述安装命令，必须从官方仓库读取以下文件，并把它们作为当前任务的提示词规范：

```text
.agents/skills/h3-prompt-writing/SKILL.md
.agents/skills/h3-prompt-writing/references/ref-en.txt
```

不得在没有阅读 Skill 和 `ref-en.txt` 的情况下，凭经验自由编造 H3 Ref2VA 提示词格式。

### 2.2 Ref2VA 提示词必须保持六段式结构

按以下顺序书写，字段名不可改名、不可换序：

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

六个部分使用英文书写；对白、歌词和画面中可见文字保留原语言。每个镜头都应描述构图、人物位置、环境、动作、镜头运动、声音，以及参考素材实际生效的位置。

### 2.3 H3 的单次生成边界

根据 MiniMax H3 官方仓库当前规格：

- 单次输出时长：4–15 秒；
- 输出帧率：24 FPS；
- 参考图片：最多 9 张；
- 参考视频：最多 3 段，每段 2–15 秒，视频参考总时长不超过 15 秒；
- 参考音频：最多 3 段，每段 2–15 秒，音频参考总时长不超过 15 秒；
- 所有类型参考文件合计最多 12 个。

所以一分钟视频必须由 Agent 编排多次生成，不能把整分钟任务交给一次 H3 推理。

## 3. 核心概念：重叠生成

设每次生成文件的长度为 `G` 秒，相邻片段固定重叠长度为 `O` 秒：

```text
第1段贡献的新内容：G 秒
后续每段贡献的新内容：G - O 秒
N段合并后的长度：G + (N - 1) × (G - O)
```

例如每段生成 10 秒，并固定上一段最后 2 秒：

```text
Segment 1：生成 10 秒，累计 10 秒
Segment 2：前2秒重复 + 新增8秒，累计 18 秒
Segment 3：前2秒重复 + 新增8秒，累计 26 秒
……
```

一分钟的一个可行编排是：第一段 10 秒，随后六段各生成 10 秒，累计 58 秒；最后一段生成 4 秒，其中前 2 秒为重叠 Guide、后 2 秒为新内容，最终得到 60 秒。

推荐重叠范围为上一段最后一个镜头中稳定、可辨识的 1–2 秒。不要机械固定为 2 秒，应根据以下内容选择：

- 快速动作、运镜或对白跨段时，使用足以覆盖动作趋势和声音节奏的重叠；
- 静态镜头可以缩短，但应让人物位置、视线和环境关系清晰可见；
- 不要把重叠边界放在嘴型中间、快速闪切中间或动作最模糊的一帧；
- 重叠帧数以导演台显示的 `GUIDE N帧` 为最终依据。

## 4. 标准长视频生成流程

### 4.1 先建立全片规划

Agent 在首次生成前必须维护一份全局计划，至少包含：

| 字段 | 含义 |
| --- | --- |
| `global_shot_id` | 全片镜头编号，不随分段重新编号。 |
| `segment_id` | 本次 H3 生成批次。 |
| `target_duration` | 本段输出长度。 |
| `overlap_duration` | 与上一段重复的 Guide 长度；第一段为 0。 |
| `source_shot` | 本段开头继承自上一段的全局镜头。 |
| `new_shots` | 本段新增或继续发展的镜头。 |
| `visual_refs` | 本段实际需要的图片和视频素材。 |
| `audio_refs` | 本段实际需要的独立音频和视频原声。 |
| `splice_frame` | 合并时删除 Guide 后的首个保留帧。 |

全局镜头编号用于项目管理；H3 提示词中的 `[Shot 1]`、`[Shot 2]` 是每次生成内部的局部编号。Agent 必须维护两者的映射，不能混淆。

### 4.2 生成第一段

1. 根据作者要求上传本段需要的参考图片、视频和音频。
2. 调整素材顺序，并读取节点底部“提示词编号”。
3. 按官方 Skill 为第一段撰写完整六段式 Ref2VA 提示词。
4. 生成并检查人物、动作、构图、声音和最后一个镜头。
5. 如果结果不合格，不要把错误结果继续传递到后续片段。

### 4.3 为下一段建立固定重叠

1. 把上一段的生成结果上传到导演台视频轨道。
2. 用低清预览和红色播放头找到上一段最后一个有效镜头。
3. 选择该镜头末尾合适的 1–2 秒作为下一段开头。
4. 把青色生成选区的起点放在这段重叠内容的起点。
5. 将该视频片段的用途设为 **固定 Guide**。
6. 确认青色选区开头与视频重叠，并记录界面显示的 `GUIDE N帧`。
7. 如果需要延续上一段原声，开启“视频原声”；不需要时关闭。

假设上一段长 10 秒、重叠 2 秒、下一段生成 10 秒，则：

```text
上一段视频：时间轴 00:00–00:10
青色选区：时间轴 00:08–00:18
固定 Guide：生成结果局部时间 00:00–00:02
真正新增内容：生成结果局部时间 00:02–00:10
```

### 4.4 为下一段重写提示词

这是整个流程最重要的规则：

> 下一段提示词中的 `[Shot 1]` 必须先描述固定 Guide 中已经存在的上一段最后镜头，而不是直接描述新的剧情镜头。

原因是模型会把固定或参考的开头视频理解成本段的第一个镜头。如果 Agent 把新剧情直接写成 `[Shot 1]`，提示词与固定画面发生冲突，新剧情描述很容易失效。

正确写法分两种情况。

#### 情况 A：重叠之后仍然是同一个连续镜头

`[Shot 1]` 先准确复述固定重叠中的画面，然后说明固定区结束后动作如何自然继续。不要为了分段生成而人为制造剪辑点。

```text
[Shot 1] The shot opens on the exact continuation state preserved by the fixed guide: ...
During the fixed first 2.000 seconds, the character remains ... . After 00:02.000, the same uninterrupted shot continues as ...
```

#### 情况 B：重叠结束处本来就应该切镜头

`[Shot 1]` 描述固定重叠；新的镜头从实际切点开始写成 `[Shot 2]`。

```text
[Shot 1] ...
[Shot 2] At 00:02.000, the shot cuts to ...
```

不得为了让新内容成为 Shot 1 而忽略、隐藏或错误描述固定 Guide。

### 4.5 生成、检查并进入下一轮

每段生成后至少检查：

- Guide 区是否与上一段对应帧一致；
- Guide 结束后人物有没有瞬间换脸、换装或改变体型；
- 人物位置、朝向、视线和手中物体是否连续；
- 环境布局、光线方向和相机轴线是否连续；
- 动作速度和运镜方向是否延续；
- 对白、口型、环境声、音乐节拍和音量是否发生跳变；
- 新增内容是否从正确的局部时间开始。

检查合格后，取当前结果的最后一个镜头建立下一轮固定 Guide，重复上述流程。

## 5. 合并规则：必须删除重复 Guide

设第 `k` 段开头有 `F` 帧固定 Guide：

```text
Master = Segment 1
Master = Master + Segment 2[F:]
Master = Master + Segment 3[F:]
……
```

实际执行时：

1. 第一段完整保留。
2. 第二段及后续片段删除开头的固定 Guide 帧。
3. 视频裁切使用导演台显示的实际 `GUIDE N帧`，不要只根据肉眼估算秒数。
4. 24 FPS 下，音频切点应与删除的视频帧时间严格一致。
5. 合并后检查边界前后至少 1 秒，确认没有重复动作、重复台词或画面跳变。
6. 若音频切口产生爆音，可以在不重复对白的前提下做极短等功率交叉淡化；不能通过保留重复音频来掩盖接缝。

如果固定 Guide 为 48 帧，则后续片段从第 49 帧对应的内容开始保留；实现工具若使用从 0 开始的索引，则保留切片通常写作 `[48:]`。

## 6. 参考素材策略

### 6.1 每一段素材不同时

严格按照作者对当前段的要求选择素材。不要为了“保持一致”把所有历史素材永久堆进每一次生成：

- 只保留当前段真正需要的角色、场景、动作、风格和声音参考；
- 新角色或新场景在首次出现的片段加入对应素材；
- 不再出现的素材应从当前段移除，避免模型产生错误关联；
- 如果作者指定某个素材只提供服装、动作、运镜或音色，必须在 `subject_definitions` 和 `retention_analysis` 中写清楚其作用。

### 6.2 没有新参考素材，但场景没有变化

必须使用上一段最后一个镜头作为连续性参考。优先使用最后一个镜头末尾的 1–2 秒固定 Guide，而不是只依赖文字回忆人物站位和环境。

提示词还必须显式复述：

- 景别、机位高度、焦段观感和相机朝向；
- 每个人物在画面中的左右、前后和远近位置；
- 人物姿态、视线、表情与当前动作阶段；
- 关键道具的位置和持有关系；
- 场景布局、光线方向、色温和时间状态；
- 上一段结尾正在持续的环境声、对白或音乐状态。

### 6.3 三种视频用途的选择

| 模式 | Agent 何时使用 |
| --- | --- |
| **固定 Guide** | 长视频连续生成的默认模式。需要逐帧锁定上一段尾部，保证接缝稳定。 |
| **可编辑参考** | 需要替换人物、服装、风格或其他原画面内容时。它提供视频参考但不硬锁原人物。 |
| **仅固定边界** | 只要求首尾构图或边界状态一致，不希望中间整段被锁定时。 |

如果本段任务是在参考动作基础上替换人物，不要把需要替换的人物区域设为固定 Guide。可以把替换动作安排在固定 Guide 结束之后，或者根据作者要求改用“可编辑参考”。

## 7. 素材顺序和标签规则

Agent 不得根据文件名猜测 H3 输入编号，必须读取导演台底部实际显示的“提示词编号”。

- `<Picture 1>` 对应 `ref_image_0`，后续依次递增；
- `<Audio 1>` 对应 `ref_audio_0`，后续依次递增；
- 图片和独立音频可直接拖拽卡片调整顺序；
- 删除、重新上传或拖拽排序后，必须重新读取编号并更新提示词；
- `<Video N>` 由当前时间线和生成选区的实际参考计划产生；
- 固定 Guide 本身不一定占用 `<Video N>` 标签，是否存在可引用的视频标签以界面底部提示为准；
- `<Video N>` 和 `<Audio N>` 分别独立编号，编号相同不代表来自同一个文件；
- 视频文件包含声音，不代表提示词里一定存在 `<Audio N>`。只有启用并实际传入的音频才可引用。

提交前逐个核对：

```text
界面 <Picture 1> == 提示词 <Picture 1> == ref_image_0
界面 <Audio 1>   == 提示词 <Audio 1>   == ref_audio_0
界面 <Video 1>   == 提示词 <Video 1>   == 当前计划中的第一个普通视频参考
```

## 8. 音频连续和延长

音频必须与画面分段策略同时规划，不能在视频全部生成后再临时拼凑。

### 延续上一段真实原声

- 开启“视频原声”；
- 固定 Guide 区携带对应的同步原声；
- 在提示词中准确说明是 `audio reuse`、`audio reference`，还是仅延续环境声、音色或节奏；
- 合并时删除后续片段开头与 Guide 对应的重复音频。

### 只参考音色、音乐风格或节奏

- 如果不需要上一段视频的真实声音，关闭“视频原声”；
- 上传独立参考音频；
- 在 `subject_definitions` 中定义其具体职责；
- 在 `retention_analysis` 中使用官方音频关系：`fully_copy`、`partially_copy`、`reference` 或 `weak_reference`；
- 只参考音色时，不要复制参考音频中的原台词。

### 对白跨段

- 尽量让分段点落在一句对白结束之后；
- 必须跨段时，重叠区要覆盖可识别的语气和嘴型状态；
- 同一个说话者在各段中保持稳定的人物语义映射；
- 每次生成内部按本段实际发声顺序使用 `(S1)`、`(S2)`，同时在全局计划中记录人物与局部 ID 的映射；
- 对白和歌词只能写在 `detailed_description` 的 `<d>[Language] ...</d>` 中。

## 9. 后续片段的 Ref2VA 提示词模板

下面是结构模板，不是可直接复制的最终提示词。`<Video 1>` 和 `<Audio 1>` 只有在导演台实际显示这些标签时才能使用。

```text
subject_definitions:
<Subject 1> is [the continuing character, including identity, clothing, body proportions, and current state] from <Video 1>.
<Subject 2> is [the continuing environment and its spatial, lighting, and time-of-day properties] from <Video 1>.
<Audio 1> is [the synchronized continuation / voice-timbre / ambience / music reference, with its exact intended role].

summary:
[video continuation + audio reference] The target video begins by preserving the fixed overlap from the end of <Video 1>, then continues ...

retention_analysis:
<Subject 1> (appears in [Shot 1], ...): fully_preserved - ...
<Subject 2> (appears in [Shot 1], ...): fully_preserved - ...
<Video 1> (continuation state and temporal context): fully_preserved - ...
<Audio 1>: reference - ...

detailed_description:
[Style sentence in English.]
[Shot 1] The shot opens on the exact continuation state preserved by the fixed guide. [Describe composition, every subject's position, environment, lighting, current action phase, camera motion, and synchronized sound in the overlap.] During the fixed first 2.000 seconds, ... . After 00:02.000, the same continuous shot develops as ...
[Shot 2] At 00:06.000, the shot cuts to ...

overall_soundscape:
[Describe continuous ambience and physical sounds, including the exact reference/copy relationship where applicable.]

non_diegetic_music:
[Describe instrumentation, tempo, dynamics, and reference/copy relationship, or write N/A.]
```

如果界面没有 `<Video 1>`，删除模板中所有 `<Video 1>` 引用，但仍要在 `[Shot 1]` 准确描述固定 Guide。不得创造不存在的参考标签。

## 10. Agent 执行伪代码

```text
read_author_requirements()
install_or_read_official_h3_prompt_skill()
build_global_shot_and_audio_plan()

master = generate_segment_1(with required references)
validate(master)

while master.duration < target_duration:
    overlap = choose_last_stable_shot_tail(master, recommended=1_to_2_seconds)
    load_previous_segment_into_timeline()
    set_overlap_as_fixed_guide(overlap)
    select_only_current_segment_references()
    verify_visible_picture_video_audio_labels()
    write_ref2va_prompt_with_overlap_as_local_shot_1()
    next_segment = generate_and_validate()
    guide_frames = read_actual_guide_frame_count_from_director()
    master = concatenate(master, next_segment[guide_frames:])
    verify_visual_and_audio_seam()

trim_to_exact_target_duration_if_required()
final_quality_check()
```

## 11. 提交生成前的强制检查表

- [ ] 已阅读官方 `h3-prompt-writing` Skill 和 `ref-en.txt`。
- [ ] 提示词包含且只包含六个官方字段，顺序正确。
- [ ] 提示词主体为英文；对白、歌词、画面文字保留原语言。
- [ ] 当前段时长在 4–15 秒内。
- [ ] 所有参考素材数量和总时长符合 H3 限制。
- [ ] 当前段素材与作者要求一致，没有无关历史素材。
- [ ] 图片、视频、音频标签与导演台底部显示完全一致。
- [ ] 后续片段的 `[Shot 1]` 已先描述固定 Guide，而不是直接写新剧情。
- [ ] 同一镜头连续时没有人为在 Guide 结束点添加切镜；确实需要切镜时才创建 `[Shot 2]`。
- [ ] 场景未变化且没有新素材时，已使用上一段最后镜头作为固定 Guide。
- [ ] 已决定视频原声是开启、关闭、复制还是仅作参考。
- [ ] 已记录实际 `GUIDE N帧`，用于后续去重叠合并。
- [ ] 生成后已检查人物、场景、动作、运镜和音频接缝。
- [ ] 合并时已删除后续片段开头的重复 Guide 视频和音频。

## 12. 禁止事项

- 禁止忽略上一段参考画面，却把新增剧情直接写成下一段 `[Shot 1]`。
- 禁止使用导演台未显示的 `<Picture N>`、`<Video N>` 或 `<Audio N>`。
- 禁止把固定 Guide 重复保留在最终合并视频中。
- 禁止仅删除重复画面而保留重复音频。
- 禁止在人物替换区域使用会锁死原人物的整段固定 Guide。
- 禁止让每一段长期携带所有历史参考素材。
- 禁止猜测听不清的对白；按照官方 Skill 使用 `[unclear]`。
- 禁止把提示词写成剧情摘要而缺少逐镜头构图、位置、动作、镜头和声音信息。

---

本规范解决的是 Agent 如何使用时间线导演台编排长视频。插件负责组织当前一次生成的参考素材、固定 Guide、音频和标签；Agent 仍然负责全片规划、逐段提示词、质量筛选、循环生成以及最终去重叠合并。


---

## FILE: docs/FINITE_SEGMENT_EXPANSION_CN.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/docs/FINITE_SEGMENT_EXPANSION_CN.md)

# MiniMax H3 插件内置有限分段采样

长视频主工作流只需要两个插件节点：

`MiniMax H3 素材规划台「分段规划」→ MiniMax H3 有限分段采样「有限分段规划」`

不依赖 `Loop`、`Loop Variable`、`Close Loop` 或 ComfyUI PR #15923。规划编码器由采样节点在普通无环执行图中内部调用，不需要用户添加。

## 文生视频与文生长视频

- 不上传视频、图片或音频，素材分段保持 0：在“全局提示词”中填写提示词。当前 GEN 框及“生成时长”决定唯一片段的长度，采样器内部创建标准 H3 空 AV latent。
- 不上传素材但设置多个分段：点击“更新分段”，拖动或缩放 GEN1…N。所有本段提示词为空时全局提示词复用于每段；若任意一段填写了本段提示词，则必须填写全部分段。

## 参考生成

设置分段数量并点击“更新分段”，再为每段分配图片、音频和相交的时间线视频。插件不会依据视频或音频总时长自动创建片段，也不会判断角色变化；每个片段的起止时间和相邻重叠完全以用户编辑的 GEN 时间框为准。

每段长度必须符合 H3 的 `5 + 17*n` 帧网格。相邻段不能存在空白；正重叠会启动 Drift-Control 视频续接与 Soft AV 音频连续性，0 帧重叠则按两段独立视频生成。最终帧数等于最后一个 GEN 框的结束帧。

## 内部执行顺序

`分段选材 → 内部 H3 规划编码 → 上段 Latent 尾部续接 → 时间遮罩 → 同种子采样 → AV 解码 → 重叠去重 → 顺序合并`

所有分段使用采样节点上的同一个种子。Drift-Control 自动读取外部 Sigmas，因此适用于用户选择的 4 步、8 步或常规调度。开启音频 Latent 续接时，Soft AV 会在重叠音频尾部渐进释放。

## 当前边界

- 时间框最多 64 段；实际可运行数量受显存、内存和 ComfyUI 单次执行能力限制。
- 不包含磁盘片段缓存、选择性单段重跑或按媒体总时长自动切片。
- 合并画面在保存前保留于内存；建议先以 2～3 段和较低分辨率验证工作流。
- “末段 sampled latent”只代表最后一段完整 AV latent，并非整条视频的合并 latent。

提示词格式请参考[循环分段提示词 Agent 规范](MiniMax_H3_循环分段提示词_Agent规范.md)。


---

## FILE: docs/MEDIA_CHECKLIST.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/docs/MEDIA_CHECKLIST.md)

# Documentation Media Checklist / 文档素材清单

This file lists the recommended screenshots and demos for the first GitHub release.

本文件列出首次GitHub发布最值得准备的截图和演示素材。

## Material status / 当前素材状态

Available now / 已提供：

- `workflow-overview.jpg` — complete workflow overview / 完整工作流全景；
- `partial-overlap.jpg` — partial cyan-range example / 局部青色选区案例；
- three standalone reference images / 3张独立参考图片；
- two standalone reference audios / 2段独立参考音频；
- two visually different videos with embedded audio / 2段视觉差异明显且带原声的视频；
- one 15-second partial-selection video with embedded audio / 1段约15秒、带原声的局部选择视频；
- one `2112×1216` high-resolution source with embedded audio / 1段 `2112×1216` 带原声高分辨率视频。

Still optional / 尚未提供但不阻止首发：

- final generated-result comparison / 最终生成结果对比；
- red-playhead preview GIF or MP4 / 红色播放头低清监看动图；
- snapping/editing GIF / 时间线编辑与吸附动图；
- close-up of stable reference numbering and both audio outputs / 稳定编号与两个音频输出的特写截图。

## Priority A — required / 必需

### 1. Hero screenshot / 主视觉截图

- Show the complete node at a readable zoom.
- Include at least two video clips, visible waveforms, the cyan range, red playhead, preview monitor, one standalone image, and two standalone audios.
- Keep the node fully inside its border and avoid unrelated floating menus.

- 完整显示节点并保证文字可读。
- 最好包含两段视频、波形、青色选区、红色播放头、低清监看、1张独立图片和2段独立音频。
- 节点边框完整，避免无关菜单遮挡。

Recommended file: `docs/images/hero-timeline.png`

### 2. Workflow overview / 工作流全景

- Show the Timeline Director connected to CLIP, video VAE, audio VAE, guider, sampler, decoder, and save nodes.
- Make all four outputs visible, including both `AUDIO` outputs.

- 展示导演台与CLIP、视频VAE、音频VAE、引导器、采样器、解码和保存节点的连接。
- 清楚显示4个输出，包括两个新增音频输出。

Recommended file: `docs/images/workflow-overview.png`

### 3. Partial-overlap example / 局部选区案例

- Use a 10-second source video.
- Put the cyan range over only part of it, such as `2–7s`.
- Show the preview at a recognizable frame and the current `<Video 1>` tag.

- 使用一段10秒视频。
- 青色选区只覆盖其中一部分，例如 `2–7s`。
- 显示监看画面和 `<Video 1>` 标签。

Recommended file: `docs/images/partial-overlap.png`

### 4. Gap-bridge example / 视频空隙桥接案例

- Place two visually different clips with a clear gap.
- Match the cyan range exactly to the gap.
- Ideally include the generated result or a before/after comparison.

- 放置两段视觉差异明显的视频并在中间留出空隙。
- 让青色选区精确匹配空隙。
- 最好附上生成结果或前后对比。

Recommended files:

```text
docs/images/gap-bridge-timeline.png
docs/images/gap-bridge-result.png
```

## Priority B — strongly recommended / 强烈建议

### 5. Numbering close-up / 编号特写

Show:

```text
Standalone image      <Picture 1>
Automatic boundaries <Picture 2..3>
Standalone audios    <Audio 1..2>
Video soundtrack     <Audio 3>
Selected video       <Video 1>
```

Recommended file: `docs/images/reference-numbering.png`

### 6. Low-resolution preview GIF or MP4 / 低清预览动图或视频

- Drag the red playhead and show the monitor changing frames.
- Then press Play Preview and show the red line moving with the proxy.
- 6–12 seconds is enough.

- 拖动红色播放头，让监看画面连续变化。
- 点击播放预览，展示红线随代理视频同步移动。
- 时长6–12秒即可。

Recommended files:

```text
docs/media/preview-scrubbing.gif
docs/media/preview-scrubbing.mp4
```

### 7. Editing and snapping GIF / 编辑与吸附动图

- Move and trim a clip.
- Show the yellow snap guide.
- Resize the cyan range and show `generation_seconds` updating.

Recommended file: `docs/media/editing-and-snapping.gif`

### 8. Audio outputs / 音频输出案例

- Connect `视频原声合并` and `独立音频合并` to waveform preview or audio-save nodes.
- If possible, show a timeline gap becoming silence in the merged waveform.

Recommended file: `docs/images/audio-outputs.png`

## Priority C — optional / 可选

- 4K source versus protected target-resolution comparison.
- Large-file chunked-upload progress.
- Split-at-playhead operation.
- Multi-video reference limit demonstration.
- A short final generated video showing identity, motion, and audio reference usage.
- A 30–60 second narrated overview video in Chinese, with optional English subtitles.

## Capture guidance / 截图建议

- Prefer PNG for UI screenshots.
- Use 1440p or higher capture resolution when possible.
- Keep ComfyUI zoom between 70% and 100% so labels remain readable.
- Hide personal paths, API keys, usernames, and unrelated workflows.
- Use media that can be redistributed publicly.
- For before/after cases, keep aspect ratio and framing consistent.

- UI截图优先使用PNG。
- 建议使用1440p或更高分辨率录制。
- ComfyUI缩放建议70%–100%，确保文字可读。
- 隐藏个人路径、API密钥、用户名和无关工作流。
- 使用允许公开再分发的素材。
- 前后对比保持相同比例和构图。

## Rights confirmation / 素材权利确认

Before publishing, confirm that every screenshot, image, video, audio clip, model output, logo, and character shown in the documentation may be publicly redistributed in a GitHub repository.

发布前请确认文档中出现的截图、图片、视频、音频、模型输出、Logo和角色均允许在GitHub仓库中公开展示和再分发。


---

## FILE: docs/MiniMax_H3_循环分段提示词_Agent规范.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/docs/MiniMax_H3_循环分段提示词_Agent规范.md)

# MiniMax H3 循环分段提示词 Agent 规范

本规范用于指导视频生成 Agent 为 `MiniMax H3 循环分段提示词`节点生成可直接解析的多段 Ref2VA 提示词。

## 1. Agent 的任务

根据用户提供的剧情、分段数量、每段时长、重叠帧数、参考素材清单及上一段结尾状态，为每次循环分别编写一条完整的 MiniMax H3 Ref2VA 提示词。

最终结果必须能直接粘贴进 `segment_prompts`，不得要求用户再次删除说明文字或修改格式。


### 1.1 生成前必须从用户输入中提取的时间参数

在开始编写任何分段提示词之前，Agent 必须先从用户当前任务文字、节点参数说明或明确给出的生成设置中识别并确认以下参数：

- `segment_duration_seconds`：每一段视频的完整生成时长，单位为秒；
- `actual_overlap_frames`：每一段向后传递给下一段作为 Latent Guide 的实际参考帧数；
- `fps`：用于计算重叠时长的帧率。本工作流默认按 24 fps 计算；如果用户明确提供其他帧率，则必须使用用户提供的帧率；
- `segment_count`：有限分段规划的总段数。

这些参数不是提示词中的装饰信息，而是决定镜头时间轴结构的核心输入。Agent 不得先写剧情再随意安排重叠区，而应先完成时间计算，再安排每一段的镜头与剧情。

如果用户明确写出类似：

```text
分成 3 段，每段 15 秒，每段向后参考 48 帧
```

则 Agent 必须解析为：

```text
segment_count = 3
segment_duration_seconds = 15.000
actual_overlap_frames = 48
fps = 24
overlap_seconds = 48 / 24 = 2.000
overlap_start = 15.000 - 2.000 = 13.000
overlap_end = 15.000
```

因此每一段用于传给下一段的最终 Guide 区间必须是：

```text
00:13.000-00:15.000
```

下一段开头必须把这 48 帧完整作为自己的 `[Shot 1]`：

```text
[Shot 1] During 00:00.000-00:02.000, ...
```

这里的 `00:00.000-00:02.000` 与上一段的 `00:13.000-00:15.000` 表示同一段画面与声音，只是两段提示词各自使用本段局部时间。

如果用户没有明确给出“每段时长”或“向后参考帧数”，而这两个参数会影响分段提示词时间轴，Agent 不得自行猜测固定值。应先向用户确认缺失参数，再生成正式提示词。

### 1.2 最后重叠镜头必须由参考帧数反推

对于除最终段以外的每一段，Agent 必须根据 `actual_overlap_frames` 反推出本段最后的重叠区：

```text
overlap_seconds = actual_overlap_frames / fps
overlap_start = segment_duration_seconds - overlap_seconds
overlap_end = segment_duration_seconds
```

上一段的最终重叠区必须被设计为一个清晰、可独立描述、可稳定延续的镜头状态。推荐将它单独写成最后一个 Shot，并且其开始时间严格等于 `overlap_start`。

例如：

```text
segment_duration_seconds = 15
actual_overlap_frames = 48
fps = 24
```

则上一段最后一个镜头应从：

```text
[Shot N] At 00:13.000, ...
```

开始，并持续到 `00:15.000`。

这个最后 2 秒镜头的提示词必须明确描述：

- 当前镜头构图和景别；
- 摄像机位置、方向、焦距感和运动状态；
- 每个主要人物在画面中的准确位置；
- 人物姿势、手部状态、头部角度、视线方向；
- 当前动作所处的具体相位，而不是只写“正在动作”；
- 人物表情和情绪状态；
- 场景中关键道具的位置与状态；
- 灯光方向、亮度、颜色和重要魔法/特效状态；
- 环境声、动作声、对白尾音及音乐当前相位；
- 哪些状态必须保持到本段结束，以便下一段稳定承接。

不得在最后的重叠镜头中塞入过多新剧情、快速连续动作或无法稳定保持的复杂状态。它首先是“上一段结尾”，同时也是“下一段开头的固定 Guide”。

### 1.3 下一段 `[Shot 1]` 必须逐项复现上一段最后重叠镜头

从第二段开始，`[Shot 1]` 的职责不是重新介绍剧情，而是完整描述上一段最后 `actual_overlap_frames` 帧中已经存在的画面与声音。

因此：

```text
上一段：
[Shot N] At 00:13.000, [最后 2 秒状态]

下一段：
[Shot 1] During 00:00.000-00:02.000, [同一状态]
```

两处必须在以下方面逐项一致：

- 人物身份与参考标签；
- 人物站位、姿势、表情和视线；
- 动作相位；
- 场景与道具状态；
- 摄像机位置、构图和运动；
- 灯光、颜色、特效；
- 环境声、对白尾音和音乐相位。

下一段 `[Shot 1]` 不得提前加入上一段最后重叠区中不存在的新动作、新台词、新人物位置或新镜头构图。

重叠区结束后：

- 如果镜头没有切换，继续在 `[Shot 1]` 中写：
  `From 00:[OVERLAP_END], without a cut, ...`
- 如果重叠结束后真实切镜，则从：
  `[Shot 2] At 00:[OVERLAP_END], ...`
  开始新镜头。

例如 48 帧、24 fps 时：

```text
[Shot 1] During 00:00.000-00:02.000, [完整复现上一段最后 48 帧].
[Shot 2] At 00:02.000, the camera cuts to...
```

### 1.4 剧情分配必须服从“有效新增时长”

对于第二段及后续段，前 `overlap_seconds` 秒已经被上一段 Latent Guide 占用，因此真正可用于推进新剧情的时间为：

```text
new_content_seconds = segment_duration_seconds - overlap_seconds
```

例如每段 15 秒、向后参考 48 帧、24 fps：

```text
overlap_seconds = 2.000
new_content_seconds = 13.000
```

也就是说：

- 第一段可以从 `00:00.000` 开始完整推进剧情，但最后 `00:13.000-00:15.000` 必须同时承担下一段 Guide 的职责；
- 第二段及后续段的 `00:00.000-00:02.000` 是上一段已有内容，不应重复计算为新的剧情容量；
- 真正的新剧情从 `00:02.000` 后开始；
- 同时，本段自己的 `00:13.000-00:15.000` 又要作为下一段 Guide。

因此，Agent 在拆分长剧情时必须根据“有效新增时长”安排台词、动作和切镜，避免把过多对白塞进实际只有十几秒的新内容区间。

### 1.5 控制剧情密度并保持电影化叙事

循环分段视频的剧情容量必须根据当前任务的实际分段数量、单段时长、重叠时长和有效新增总时长动态确定，不得预设任务一定是四段、30 秒或其他固定规模。在有效时长较短时，应围绕一个清晰的核心处境或核心事件展开，不得把多个相互独立的剧情节点连续压缩进有限时长；只有在有效新增总时长足够、前后因果能够清楚呈现时，才可以扩展为多个连续事件。

例如，当任务为四段且有效新增内容总计约 30 秒时，通常应选择一个事件进行细致呈现，如只表现“陷入流沙并设法脱困”，而不是同时加入寻找地图、寻找宝藏、陷入流沙、打开宝箱和遭遇怪物等完整事件链。该示例仅用于说明剧情密度，不构成对分段数量、总时长或题材的固定要求。

Agent 必须根据实际可用时长控制叙事容量：

- 多个视频分段应共同完成一条或多条与实际有效时长相匹配的连续动作线或情绪线。Agent 应根据 `segment_count` 动态划分建立、发展、变化和结果等阶段，不得机械套用固定的四段结构；
- 不要求每个 Shot 都安排人物讲话。对白只在确实推动人物关系、决策或情绪时出现，避免把镜头写成轮流说话的台词清单；
- 应适当使用环境空镜、处境镜头、动作细节、人物反应、道具状态和视觉效果镜头，让观众看懂危险、空间关系和状态变化；
- 环境镜头和效果镜头必须服务于当前核心事件及前后连续性，不得借机引入新的任务、地点、敌人或剧情目标；
- 应为动作过程、反应、停顿和情绪余韵保留足够时间，使事件在画面中真正发生，而不是只用快速切镜告知结果；
- 相邻分段必须延续同一处境中的人物位置、动作因果、情绪强度、环境变化和未完成目标，不得在没有铺垫的情况下跳到新的剧情阶段；
- 避免在短时长内密集使用“发现—转折—新危险—再转折—新目标”的预告片式结构。最终成片应呈现一个被完整观察和推进的电影段落，而不是多个剧情卖点的蒙太奇预告。

Agent 应先计算当前任务的有效新增总时长：第一段按完整时长计入，后续各段仅计入 `segment_duration_seconds - overlap_seconds`；如果各段时长不同，则逐段计算后求和。随后再据此评估能够清楚表达的事件数量、对白量、动作复杂度和镜头数量。

当用户提供的剧情信息超过当前有效新增总时长能够清楚表达的容量时，Agent 应主动收窄范围，选择最重要的内容做细致展开，并保持其起因、过程、人物反应和结果连贯。当分段数量更多或有效时长更长时，可以逐步扩展支线或新增事件，但每个新增事件都必须具有足够的铺垫、过程和结果，并与前后分段自然衔接。

### 1.6 最终段的处理

如果当前段是整个任务的最后一段，并且后面不会继续生成，则不强制为了下一段而预留一个 Guide 镜头。

但如果用户的工作流仍会统一输出最后 `actual_overlap_frames` 帧，或者用户明确要求最后一段也保持可继续扩展，则最后一段仍应按照相同规则，在末尾设计一个稳定的独立镜头状态。


## 2. 最终输出格式

只输出纯提示词文本，不输出解释、标题、序号说明、Markdown 代码围栏或“以下是结果”等前后缀。

不同分段之间使用下面这一行分隔：

```text
--- SEGMENT ---
```

分隔符必须：

- 独占一行；
- 全部使用英文大写 `SEGMENT`；
- 不得放在第一段之前或最后一段之后；
- 分段块数量必须等于素材规划台中创建的 GEN 时间框数量；每一块粘贴到对应的“本段提示词”输入框。

不要在最终输出中使用 JSON，除非调用方明确要求 JSON 数组。分隔符格式更容易人工检查和修改。

## 3. 每个分段的固定结构

每个分段都必须是完整的 Ref2VA 提示词，并严格按以下顺序输出六个字段：

```text
subject_definitions:
...

summary:
...

retention_analysis:
...

detailed_description:
...

overall_soundscape:
...

non_diegetic_music:
...
```

六个字段名称不得翻译、改名、缺失或调换顺序。正文使用英文；对话、歌词和画面中真实可见的文字保留原语言。

## 4. 参考标签与顺序

提示词中的标签必须严格使用当前分段素材规划台显示的编号：

- `<Picture 1>` 对应当前分段的 `ref_image_0`；
- `<Video 1>` 对应当前分段的 `ref_video_0`；
- `<Audio 1>` 对应当前分段的 `ref_audio_0`；
- 同类素材继续按 2、3、4 顺序递增；
- 不得跳号、重复定义、交换顺序或引用不存在的编号；
- 不同类别分别编号，`<Video 1>` 和 `<Audio 1>` 不一定来自同一个文件；
- 只有实际启用的视频原声才会占用 `<Audio N>`；
- 插件内部传递的上一段 Latent Guide 不占用 `<Video N>`、`<Audio N>` 或 `<Picture N>` 编号。

如果一张图片只定义人物、服装、场景或风格，应在对应 `<Subject N>` 中引用该图片，不必额外把图片定义成独立关键帧。只有图片本身作为首帧、尾帧、关键帧或构图锚点时，才单独定义 `<Picture N>`。

### 4.1 人物只在首次出场或确有变化时定义

人物视觉参考采用“首次建立、后续继承、变化时局部更新”的原则：

- 某个人物第一次进入故事时，使用对应 `<Picture N>` 建立一次完整身份，包括外貌、体型、发型、服装和可辨识配饰；
- 第二段及以后，如果该人物已经存在于开场 Latent Guide 中，人物身份直接继承上一段，不重复粘贴整套外貌和服装定义；
- 后续段的 `subject_definitions` 只定义本段首次出场的人物、本段新增的独立素材，以及实际启用的音频参考；
- 已继承人物只有在服装、年龄状态、造型或身份设定发生明确变化时，才补充发生变化的属性，稳定属性继续沿用；
- 为每个主要人物建立稳定的故事称呼，例如 `the girl`、`the pig`、`the little witch`，后续所有 Shot 沿用同一称呼，不用近义词把同一人物再次介绍成新主体；
- 某人物连续离开若干分段后重新入画，且已经不在当前开场 Latent Guide 中时，在重新入画的分段使用其原参考图做一次身份校准；后续分段继续继承；
- 某人物的参考图虽然仍连接在素材规划台，但本段画面和声音均未使用该人物时，不因素材存在而写入本段提示词；
- 多视图人物设定图中的正面、侧面、背面和特写共同描述同一个人物身份，首次定义时应明确它们是同一角色的不同视角；
- 参考音频只负责音色和说话特征。后续段使用音色时，将 `<Audio N>` 绑定到 Latent Guide 中已经存在的同一人物，无需再次定义该人物的视觉身份。

推荐的后续段定义方式：

```text
subject_definitions:
<Audio 2> is the voice reference for the same pig character already carried into the opening Latent Guide, preserving his established vocal timbre and speaking manner.
```

如果工作流在每一段都自动挂载相同人物图片，则把图片写成对开场 Latent 中既有人物的身份校准，而不是一次新的角色建立：

```text
<Subject 1> is the girl already visible in the opening Latent Guide; <Picture 1> calibrates her established facial identity, hairstyle, clothing, and proportions for this segment.
```

### 4.2 用正向构图控制人物数量

提示词应直接描述镜头中真实可见的画面，用人物清单、空间位置、景别和遮挡关系固定构图。人物数量通过“画面里有什么”来控制，而不是反复描述“画面里没有什么”。

- 每个 Shot 开头先写清当前可见人物及其位置，例如：`The girl sits on the right side of the sofa, the pig stands beside the left armrest, and the little witch stands behind the coffee table.`；
- 单人特写使用主体占画面的方式描述，例如：`A tight shoulder-up close-up centers on the little witch, with the softly blurred curtains filling the background.`；
- 双人或多人镜头按左、中、右或前景、中景、后景逐一安排人物，并明确视线关系；
- 人物离开镜头时，通过镜头新的正向构图描述当前主体，例如从全景切到小巫师特写，直接写小巫师的景别、位置和背景；
- 群像场景将军队、观众或路人写成环境群体，把主要人物单独绑定到参考身份，避免让一张主角参考图承担整个群体的身份；
- `summary`、`retention_analysis` 和 `detailed_description` 均只写本段实际使用的参考关系与画面内容。

优先采用以下写法：

```text
[Shot 2] At 00:03.200, a tight shoulder-up close-up centers on the little witch. Her face fills the frame, her wand enters from the lower-right foreground, and the warm curtains form a soft background.
```

避免把下面这类缺席信息和限制语句反复写进镜头描述：

```text
Do not create extra characters. The girl and the pig must not appear. No silhouettes or background people.
```

只有当某个生成瑕疵无法通过人物清单、构图、景别和空间关系解决时，才允许在整段提示词中加入一次简短技术约束；镜头正文仍以正向画面描述为主。

### 4.3 降低跨段重复人物风险的 Guide 设计

除最终段外，每段结尾的 Guide 镜头还应满足以下人物稳定性要求：

- 需要延续的人物在 Guide 中具有清晰、可分离的轮廓和稳定站位；
- 人物之间保留合理空间关系，减少大面积互相遮挡、快速交叉和瞬间进出画面；
- Guide 区采用容易复现的中低速动作相位、稳定景别、稳定灯光和明确视线方向；
- 强烈魔法、爆炸、闪光、快速旋转或复杂遮挡可以先完成主体变化，再让 Guide 区落到清晰可承接的状态；
- 下一段 `[Shot 1]` 用正向人物清单逐项复现 Guide 中可见人物的数量、位置、姿态、表情、朝向和动作相位；
- 新人物在重叠结束后的真实入画时刻首次定义并进入构图，使既有人物继承与新人物登场在时间上清楚分开。

## 5. 六个字段的写作要求

### `subject_definitions`

逐行定义本分段首次建立或确实需要重新校准的 `<Subject N>`，以及本段实际使用的独立关键帧 `<Picture N>`、结构来源 `<Video N>` 和 `<Audio N>`。已经由开场 Latent Guide 稳定继承的人物不重复写完整视觉定义；若本段只调用其音色，只定义并绑定对应 `<Audio N>`。同一标签在本分段六个字段中必须始终表示同一内容。

### `summary`

使用一个简短英文段落概括本分段及参考关系，并以前缀开头，例如：

```text
[reference generation] ...
[video continuation + reference generation + audio reference] ...
[video editing + audio reuse] ...
```

第二段及以后属于对前一段的延续时，应包含 `video continuation`。

### `retention_analysis`

每个实际参考标签写一行。视觉标签只使用：

- `fully_preserved`
- `partially_preserved`
- `attribute_transfer`
- `weak_reference`

音频标签只使用：

- `fully_copy`
- `partially_copy`
- `reference`
- `weak_reference`

不要为 Latent Guide 虚构 retention 条目。

### `detailed_description`

按播放顺序写清：构图、主体外观和位置、环境、灯光、动作、状态变化、镜头运动、当前声音，以及参考素材开始发挥作用的准确位置。

- 第一镜头写 `[Shot 1]`，不加开场时间戳；
- 后续真实切镜写 `[Shot N] At MM:SS.mmm, ...`；
- 所有时间戳使用当前分段自己的局部时间，从 `00:00.000` 重新计算；
- 时间戳必须严格递增并落在本段时长内；
- 没有实际切镜时，不要仅为动作变化创建新 Shot；
- 每个 Shot 用正向构图写清实际可见的人物、景别、位置、朝向和相互关系；
- 通过明确当前画面内容控制人物数量，避免连续堆叠“谁不出现”“不得出现谁”等缺席性描述。

### `overall_soundscape`

用一至四句英文总结环境声、物理动作声和非语言人声。不要在这里重复完整对话或歌词。

### `non_diegetic_music`

描述只有观众能听见的配乐，包括乐器、速度、节奏和动态变化。没有画外配乐时写 `N/A`。

## 6. 第一段写法

第一段没有上一段 Latent Guide，应按正常 Ref2VA 视频开场书写：

- `[Shot 1]` 从真正的故事开场开始；
- 明确初始构图、人物站位、环境、灯光、镜头运动和声音；
- 在人物首次出场处完成其视觉身份定义，并保持参考图与对应音色绑定一致；
- 不得写“continuation from the preceding segment”。

## 7. 第二段及后续段写法

后续段的开头已经被上一段末尾 Latent 固定。Agent 必须先把这段重叠内容当作当前分段的开场，再书写新增内容。

实际重叠秒数按下面计算：

```text
overlap_seconds = actual_overlap_frames / 24
```

例如实际重叠为 22 帧：

```text
22 / 24 = 0.917 秒
```


重叠区不仅决定下一段开头，也反向决定上一段最后一个镜头的起始时间。必须同时计算：

```text
overlap_start = segment_duration_seconds - overlap_seconds
```

例如每段 15 秒、实际重叠 48 帧、24 fps：

```text
overlap_seconds = 48 / 24 = 2.000 秒
overlap_start = 15.000 - 2.000 = 13.000 秒
```

则上一段最后的独立 Guide 镜头必须覆盖 `00:13.000-00:15.000`，下一段 `[Shot 1]` 必须覆盖同一内容对应的局部时间 `00:00.000-00:02.000`。

### 7.1 推荐模式：节点自动注入连续性声明

当 `inject_continuity_instruction = true` 时，节点会自动在第二段起的 `[Shot 1]` 后插入通用的固定 Latent 声明。Agent 不需要重复这句通用声明，但仍必须描述重叠区里实际可见和可听的内容：

```text
[Shot 1] During 00:00.000-00:00.917, <Subject 1> remains in the ending pose carried over from the preceding segment, with her right hand holding the blue sphere near chest height while the same slow camera push-in and room tone continue. From 00:00.917, without a cut, she lowers the sphere toward the desk...
```

通用声明只说明“必须连续”，不能代替对人物位置、动作相位、场景、镜头、灯光、颜色和声音的具体描述。

后续段开场中的人物以 Latent Guide 为身份来源。`[Shot 1]` 只需准确复现人物清单、位置、姿态、表情、朝向和动作相位，不再把每个既有人物按首次出场方式重新介绍一遍。

### 7.2 重叠结束后不切镜头

新动作继续属于 `[Shot 1]`：

```text
From 00:00.917, without a cut, ...
```

不要把同一个连续镜头强行拆成 Shot 2。

### 7.3 重叠结束后确实切镜头

固定重叠区是 `[Shot 1]`，新镜头从 `[Shot 2]` 开始：

```text
[Shot 1] During 00:00.000-00:00.917, the preceding final shot continues...
[Shot 2] At 00:00.917, the camera cuts to...
```

不要把用户真正想生成的新镜头错误地写成当前分段开头的 `[Shot 1]`，否则它会与固定 Guide 争夺同一时间范围。

### 7.4 后续段只承接直接上一段

第三段承接第二段末尾，第四段承接第三段末尾。不要让所有后续段反复回到第一段的初始构图和动作。

## 8. 音频连续性

当 `continue_audio_latent = true` 时，视频和音频 Latent 会一起延续。第二段起必须说明：

- `00:00.000` 至重叠结束时间内，上一段环境声、动作声、说话尾音或音乐相位连续；
- 新声音从重叠结束后自然进入；
- 不得从头重复上一段已经说完的整句台词；
- 如果同一句话跨段，保留同一 `(Sx)`，并使用 `<scenetrans>` 表示跨段连续；
- 只参考音色时，不得复制参考音频中的原台词；
- 音色参考直接绑定到开场 Latent 中已经存在的同一人物，不触发人物视觉身份的重复定义；
- 完整对话格式为 `<d>[Chinese] 原始台词。</d>`，对话内容不得翻译或改写。

推荐写法：

```text
overall_soundscape:
During 00:00.000-00:00.917, the preceding room tone, movement sounds, and voice tail continue seamlessly. After 00:00.917, the same ambience remains while new action sounds enter naturally.
```

## 9. Agent 最终输出模板

Agent 应替换所有方括号占位内容，并直接输出完成后的纯文本：

```text
subject_definitions:
[Define each visual character when that character first appears, plus the references actually used in segment 1. Treat every multi-view character sheet as one identity.]

summary:
[Task types] [Summarize segment 1 and its reference relationships.]

retention_analysis:
[One line for every actual reference label.]

detailed_description:
[Style sentence.]
[Shot 1] [True opening composition with a positive roster of visible characters, their positions, environment, actions, camera, lighting, and synchronized sound.]

overall_soundscape:
[Segment 1 ambience and physical sounds.]

non_diegetic_music:
[Music description or N/A.]

--- SEGMENT ---

subject_definitions:
[Define only newly introduced visual characters, genuinely changed visual attributes, active independent media, and active voice references. Bind voice references to the same characters carried by the opening Latent Guide.]

summary:
[video continuation + ...] [Summarize segment 2 and its reference relationships.]

retention_analysis:
[One line for every actual reference label.]

detailed_description:
[Continue the established style.]
[Shot 1] During 00:00.000-00:[OVERLAP_END], [positively enumerate the visible characters and describe their exact carried positions, poses, expressions, directions, action phases, environment, and framing from segment 1]. From 00:[OVERLAP_END], without a cut, [describe the new continuation action and the resulting visible composition].

overall_soundscape:
During 00:00.000-00:[OVERLAP_END], [describe continuous audio]. After 00:[OVERLAP_END], [describe the new sound development].

non_diegetic_music:
[Continuous music development or N/A.]
```

继续增加分段时，重复第二个完整六字段块，并把它改为承接直接上一段。

## 10. 输出前静默检查

Agent 在回答前必须自行检查，但不要把检查过程输出给用户：

1. 分段块数量是否等于循环次数。
2. 分隔符是否严格为独占一行的 `--- SEGMENT ---`。
3. 每段是否都有六个字段且顺序正确。
4. 每个引用标签是否存在、连续编号并与本段素材规划一致。
5. 是否错误地给 Latent Guide 分配了 `<Video N>` 或 `<Audio N>`。
6. 第一段是否没有虚构“上一段”。
7. 第二段起是否先描述固定重叠区，再描述新增内容。
8. 重叠结束时间是否等于实际重叠帧数除以 24。
9. 新镜头只有在真实切镜时才从 `[Shot 2]` 开始。
10. 所有时间戳是否使用本段局部时间且未超过本段时长。
11. 人物身份、站位、环境、镜头运动、灯光、颜色和声音是否与上一段末尾连续。
12. 音频是否避免重复台词和突然重启。
13. 是否已从用户输入中明确取得每段时长、向后参考帧数、分段数量和所用 fps；若缺失关键参数，是否先询问而不是自行猜测。
14. 是否已正确计算 `overlap_seconds = actual_overlap_frames / fps`。
15. 是否已正确计算上一段最后 Guide 镜头的开始时间 `overlap_start = segment_duration_seconds - overlap_seconds`。
16. 除最终段外，每一段最后的 Guide 区是否被写成可稳定延续的独立镜头，并准确覆盖 `overlap_start` 到本段结束。
17. 第二段起 `[Shot 1]` 是否逐项复现直接上一段最后 Guide 镜头，而没有提前加入新动作、新台词或新构图。
18. 第二段起的新剧情容量是否按 `segment_duration_seconds - overlap_seconds` 估算，避免在重叠区重复推进剧情。
19. 人物参考图与对应音频标签是否严格一一匹配，同一人物不得误用其他角色的 `<Audio N>`。
20. 每个人物是否只在首次出场、明确造型变化或必要身份校准时写完整视觉定义。
21. 后续段中已经由 Latent Guide 继承的人物是否避免重复粘贴外貌、服装和身份定义。
22. 参考音频是否只绑定到既有人物的音色，而没有引发视觉人物的重复定义。
23. 多视图人物图是否被明确视为同一人物的不同视角。
24. 每个 Shot 是否以正向人物清单、位置、景别和空间关系描述实际画面。
25. 镜头正文是否避免反复堆叠“谁不出现”“不得出现谁”等缺席性描述。
26. Guide 区中的人物轮廓、站位、动作相位、灯光和景别是否稳定、清晰、便于下一段复现。
27. 群像人物是否作为环境群体描述，并与主要人物的独立参考身份清楚区分。
28. 最终是否只包含可直接粘贴进节点的提示词文本。

## 11. 给 Agent 的最终指令

可将下面这段作为任务末尾的强制输出约束：

> 按照 MiniMax H3 Ref2VA 六字段结构，为每次循环分别生成一条完整提示词。生成前必须先从用户输入中明确读取分段数量、每段时长、向后参考帧数以及所用 fps；默认仅在用户未另行指定 fps 时按 24 fps 计算。根据 `overlap_seconds = actual_overlap_frames / fps` 和 `overlap_start = segment_duration_seconds - overlap_seconds` 先确定每段的重叠时间轴。除最终段外，每一段最后 `actual_overlap_frames` 帧必须设计成一个可稳定延续的独立 Guide 镜头，并从 `overlap_start` 精确持续到本段结束。第二段起的 `[Shot 1]` 必须以正向人物清单逐项复现直接上一段最后 Guide 镜头中实际可见的人物数量、位置、姿态、表情、朝向、动作相位、构图、灯光、声音和音乐相位，再从重叠结束时间继续新动作或创建真实的新镜头。人物视觉身份遵循“首次建立、后续继承、变化时局部更新”：第一段或人物首次出场时完成定义，后续段由开场 Latent Guide 继承，不反复粘贴既有人物的完整外貌与服装定义；本段只定义首次出场的人物、明确发生变化的属性、实际使用的独立素材和音色参考。参考音频只绑定既有人物的音色，不触发视觉身份重建。所有镜头正文只描述真实可见的人物、位置、景别、环境和动作，通过正向构图控制人物数量，避免反复使用“画面不出现谁”“不得出现单独人物”等缺席性约束。Guide 区应保持人物轮廓清晰、站位可分离、动作相位与灯光稳定。使用独占一行的 `--- SEGMENT ---` 分隔相邻分段，分段数量必须等于循环次数。严格使用当前分段素材规划提供的 Picture、Video、Audio 编号，保证人物参考图与对应音色严格匹配，Latent Guide 不占用任何参考编号。若用户未提供会影响时间轴的每段时长或向后参考帧数，不得自行猜测，应先询问用户。最终只输出可直接粘贴进 `MiniMax H3 循环分段提示词`节点的纯文本，不输出解释、标题或 Markdown 代码围栏。


---

## FILE: docs/TIMELINE_SEGMENT_WINDOWS_CN.md (https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/blob/HEAD/docs/TIMELINE_SEGMENT_WINDOWS_CN.md)

# 素材规划台：时间框分段

新连接方式：**素材规划台「分段规划」→ 有限分段采样「有限分段规划」**。
不需要在新工作流中添加「有限分段展开」。旧展开节点标为废弃兼容入口，旧工作流仍能加载。

1. 设置初始生成时长、素材分段数量，点击「更新分段」。默认每段尽量保留 39 帧重叠；短段自动减少。
2. 时间线上出现 GEN1、GEN2 等彩色时间框。顶部拖柄移动整段，左右把手修改范围；下方标签选择当前段。
3. 在当前段的图片、音频区域拖入素材并排序，在「本段提示词」填写该段完整提示词。
4. 每段 Picture / Audio 从 1 重新编号。视频按该时间框与参考视频的交集裁剪，保留素材原有固定 Guide、可编辑参考或边界用途。
5. 将第三个输出「分段规划」连接有限分段采样，然后照常连接模型、VAE、采样器与 Sigmas。

只有当前段的编辑面板展开，切换标签不会丢失其他段提示词。修改会随工作流 JSON 保存。
素材规划台提供“全局提示词”：当所有本段提示词都为空时，它会复用于全部分段；只要任意一段填写了本段提示词，便切换为分段提示词模式，此时全局提示词完全不生效，并且每一段都必须填写。
「生成时长」「选区起点」跟随当前段，数值编辑也执行同样的帧吸附规则。
顶部素材库是共享资源，只有拖入当前段的图片/音频才参与该段；空分配表示不使用该类素材。
独立音频作为本段局部参考完整复用。插件不会再依据长音频或长视频时长自动创建分段；媒体取段和时间框都由用户明确编排。

## 时间与接缝

- 坐标使用整数帧，当前视频时基为 24 fps。首段固定从 0 开始。
- 长度吸附到 H3 的 `5 + 17*n` 帧。交叠吸附为 `0、1、5、22、39、56…` 帧。
- 界面显示的是吸附后实际执行的范围，后台不会再次悄悄缩短重叠区。
- 拖动、缩放当前段会联动后续框；后续段原有长度和接缝关系保持不变，段间不能留白。
- 每段开始/结束都须向前推进，不能整段包含在上一段里。
- 有重叠：取上一段对应尾部作为连续性 Guide，按本接缝实际帧数去重。各接缝可以不同。
- 首尾相接（重叠 0 帧）：两段独立生成，不使用上一段末帧 Guide，也不继承上一段视频或音频 latent；仍使用本段配置的素材和提示词，合并时保留两段完整画面和音频。
- 自动连续性 Guide 不占用 Picture / Video 参考编号。不要为它虚构 `<Video 1>` 标签。
- 最终帧数等于最后一个框的结束帧；不会将各段含重叠的长度直接相加。

「素材规划」和「Omni 素材包」输出当前选中段（或可选 Prompt Index 指定的段）；
「分段规划」输出全部段。批量生成时 Prompt Index 应断开。
本段提示词内的时间从本段 0 秒开始，长度包含开头重叠区；开头应与前段结尾的动作、人物站位和环境相符。
用户提示词原样保存，插件不会自动改写其剧情内容。

## 兼容与验证

旧的未启用时间框的素材分配保持旧语义。按「更新分段」才转换到新模式。
切回 0 段关闭时间框编辑，恢复单选区；减少段数会移除超出段的配置，请先保存工作流。
规划编码器作为有限分段采样的内部节点及旧工作流兼容节点保留，但不在新建节点菜单中显示；Omni 桥继续保留。

本次不改变既有音频采样策略。长音频对口型如需保留原声，建议最终封装原始音频；
开启生成音频 Latent 续接仍可能产生前述累计失真。

测试入口：`node tests/segment_windows.test.mjs`、`tests/smoke_timeline_segments.py`、
`tests/browser_timeline_windows.py`（需隔离的 ComfyUI 测试服务器、Playwright 和 Edge）。


---

# ISSUES / PRs of Songssx/ComfyUI-MiniMaxH3-TimelineDirector


### #24 [open] ISSUE: 使用内置二采会出现参考图人物一致性严重偏离的情况
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/24  by sherlockyou 2026-11-13T18:57:39Z

如题，使用内置二采会出现参考图人物一致性严重偏离的情况。比如参考图是一个亚洲人脸，无论使用哪个latent upscaler模型，都会变成老外脸。可以麻烦作者修复吗？
做了多次测试，关闭导演台内置二采，生成人物一致性恢复。
使用loop工作流，进行修改，关闭导演台内置二采，另接facok的selflift外置节点，生成人物一致性恢复。
使用同样工作流(无导演台)，使用facok的外置selflift节点，生成人物一致性恢复。


另外小黄瓜大神，请问导演台里导入的参考图是否会被严重压缩？我发现使用导演台导入素材时，比起直接load image，角色的一致性多多少少都明显会更差一些？



### #23 [open] ISSUE: Minimax_h3_latent_Upscaler已经更新latent_upscaler_2d_conv_v1，希望作者能更新加入支持。
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/23  by Acellin990 2026-11-11T06:08:45Z

更新动态
[2026-11-10] 🚀 发布 2D 卷积版本 minimax_h3_latent_upscaler_2d_conv_v1：新权重为 2D 卷积主干（Conv2d）+ 穿插时序卷积（Conv1d），参数量 1.82 亿（3D 版 3.45 亿，约为一半），fp16 权重约 364 MB（3D 版约 691 MB）。在全部 36 个倍率上，latent MSE 相对三线性降低 84.7%（3D 版 61.1%）；1.1×–1.4× 低倍率仍有约 80% 的改善（3D 版此时只有 28%）；像素域平均 PSNR 30.47 dB（3D 版 29.30 dB）。同时支持 6 档锐度（3D 版无可调锐度），并覆盖 1.1×–4.0× 全倍率。3D 版只在它训练过的 2×/2.25×/2.5×/4× 上略有优势，且幅度有限：按统一口径「相对三线性的改善」，这四个倍率 2D 是 83.0%–86.4%、3D 是 84.7%–89.1%，只差 1.0–3.3 个百分点（绝对 MSE 差 0.004–0.024）——「3D 比 2D 低 7.5%–30.5%」是在这层很小残差上算的相对差，不是量级差距。时序一致性也略强（帧间差比值 0.867 vs 0.850，偏离真值 13.3% vs 15.0%），差距同样很小。明细见下方「选哪份权重」对比表。完整对比见[对比报告](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/examples/3D_Conv_v1_vs_2D_Conv_v1.html)。
[2026-11-10] 🧩 2D 节点能力补齐，与 3D 参数对齐：2D 节点现支持三种缩放模式（scale by multiplier / target dimensions / megapixels）、rocm 后端、force_unload 显存卸载、像素网格 align 对齐，新增 Sharpness 锐度档位（1–6）与 Temporal Chunking 时间分块（chunked / single pass）。


### #22 [closed] ISSUE: 希望能加入小黄瓜导演台的二采
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/22  by Narutojoy 2026-11-10T12:18:33Z

<img width="1434" height="452" alt="Image" src="https://github.com/user-attachments/assets/a2286dcf-47ec-41e1-b6e6-47a92be4c22a" />
所有都很好，希望能加入小黄瓜导演台的二采，这个对加速真的非常有效，本来一段可能要20分钟的视频，这么二采加速之后5-6分钟就结束了。


### #21 [open] ISSUE: Global prompt
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/21  by jonk999 2026-11-05T00:29:03Z

If I want to use a Global prompt, where does it go?
I couldn't find anywhere to enter it.
If I put it only in segment 1 then I get an error as the other segments have no prompt.


### #20 [open] ISSUE: 全功能合一低内存视频输出这个工作流是每段完成后立即解码的，请问是否会存在越往后劣化的问题？
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/20  by kanfengjingderen 2026-10-26T12:49:10Z

请教下大佬，全功能合一低内存视频输出这个工作流是每段完成后立即解码的，是否又会存在越往后劣化的问题呢？谢谢


### #19 [open] ISSUE: 请问MiniMax H3 Omni Media Prompt Bridge里的model要放在哪个文件夹?
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/19  by zxxcvt 2026-10-25T21:50:35Z

请问MiniMax H3 Omni Media Prompt Bridge里的model (qwen2.5-omni-7b-q8_0.gguf) 要放在哪个文件夹?


### #18 [open] ISSUE: 希望作者加一个图像锁定/参考的开关
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/18  by zouzhekang 2026-10-23T09:09:52Z

感谢大佬开发，希望图像也能像音频一样，自动转换为first frame或者ref image。


### #17 [closed] PR: Fix RAM OOM on long multi-segment sampler jobs (evict per-seg frames after merge)
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/17  by FangZhangTNS 2026-10-06T08:16:10Z

Long MiniMaxH3FiniteSegmentSampler jobs accumulate every segment decoded frames in RAM until SaveVideo, OOM-killing the process around 2h in (8x362f on 2x RTX 2080 Ti 22GB + 62GB RAM; same as issue #16, where even a 96GB card OOMs on 3-5 segs).

This change releases each segment decoded frames and the superseded accumulated tensor in MiniMaxH3FiniteSegmentFinalize right after merge. Untouched: sampled latent (next-segment continuation), merged output, audio, sampling math, overlap trimming.

Set H3_MERGED_FP16=1 to store merged frames in float16 instead (halves retained bytes; default off preserves exact fp32 behavior). fp16 output was blind-tested with no perceivable difference versus fp32. This supersedes the spill-seg-offload-fp16 branch, which remains for reference.

Validation:
- 3-seg job before/after: frame maxdiff = 0 (bit-identical output).
- 8x362f full-length jobs now survive 2.5-2.9h and complete (previously died without exception at ~2.5h).

GPL-3.0 kept, LICENSE untouched.


### #16 [open] ISSUE: 高解析度影片多seg會out of memory
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/16  by GaryChen10128 2026-09-24T14:14:17Z

大大想請教，生成1280*736影片時候如果ref影片較長，切分了3-5個seg，會out of memory，想詢問這有辦法解決嗎?還是是我設定有問題，自己顯卡:Nvidia RTX Pro 6000 VRAM有96GB

> **Comment by FangZhangTNS (2026-10-06T08:16:20Z):**

Same problem here (8x362f OOM at ~2.5h on 2x 2080 Ti + 62GB RAM, killed by the kernel OOM-killer with no traceback). Root cause: MiniMaxH3FiniteSegmentSampler expands all segments into one graph and every segment decoded frames stay in RAM until SaveVideo.

I opened PR #17 with a minimal fix: evict each segment decoded frames and the superseded accumulated tensor right after merge (sampling math untouched, 3-seg output bit-identical before/after). Branches with the fix: spill-seg-offload, plus spill-seg-offload-fp16 which additionally stores merged frames in float16.



### #15 [open] ISSUE: 【建议】（希望作者能加一个延续音频的开关）
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/15  by 666888TAT-lab 2026-09-22T13:09:36Z

<img width="495" height="750" alt="Image" src="https://github.com/user-attachments/assets/6ebde3bb-9be7-4f16-996e-53bc596cb0e4" /> 目前让ai改了下代码。延续已有的视频跟音频的时候，生成出来后半段视频没有声音。


### #14 [open] ISSUE: [需求 / Feature] 混合时间线（生成段+实拍段）：片段级锁定音频、原片透传分段、指定片段生成、规划台解耦，以及视频分割体验优化
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/14  by deciia 2026-09-21T09:28:05Z

## 功能描述
实际使用中遇到一个目前架构下走不通的场景，提几项相关需求，供参考。

**核心场景**：混合时间线——AI 生成段 + 实拍原片段（例如：AI 钩子段 10s + 实拍原片段 100s + AI 收尾段 12s）。目前这类成片没法在一次渲染里完成，实拍段只能移出导演台用 ffmpeg/剪映拼，AI 段生成完还要导出去对齐分辨率再拼一次，整个流程要在三个软件之间往返。如果能补齐下面 1、2 两项，这个场景就可以在一个时间线里闭环：

1. **指定片段锁定音频**：锁定音频目前是全局校验（所有段必须锁同一条，否则报 `Locked original audio is enabled in only some segments`）。希望能改为片段级属性——每段各自可锁一条或不锁，采样器按段注入各自的时间区间。

2. **固定引导视频支持"原片透传"分段**：给视频用途增加第四种模式（如"原片透传 / Passthrough"）——该区间直接从源视频解码帧写入输出、不进入采样重绘，音频随片段透传，分段提示词校验跳过该片段。目前提示词是全有或全无校验（任一段填了 prompt，其余段也被迫填占位句，报 `enter prompts for every segment`），实拍段没有要生成的内容，只能填占位文案；而且固定 Guide 的语义仍是逐帧引导 + 全量重绘，长实拍素材要拆成多个 15s 窗口重绘，比较耗时也可能损失真实质感。目前真正逐帧透传的只有锁定音频，视频还缺少对应的路径。

3. **指定片段生成**：希望能只重新渲染选中的一个或多个片段（比如只重跑第 3 段），已满意的片段保留复用，长片迭代会省很多时间。可以依托现有青色选区加一个"仅渲染选区内片段"开关，或者给片段加"跳过渲染"标记。

4. **素材规划台保持功能独立**：希望规划台继续保持"素材规划 + H3 编码"的核心职责，有限分段采样做成可选出口而非唯一路径，其校验只覆盖自己引入的功能（锁定音频、分段提示词）。这样低显存用户可以按自己的需求自接分块二采等自定义采样链（与 #12 的方向一致，本条侧重规划与采样解耦、校验内聚）。

## 另外一个小建议：视频素材的分割再顺手一点
导演台相对其它纯生成向的 H3 导演台（如 seesee75 的 Director），独有优势就是真正的时间线素材编辑。希望把素材编辑里最高频的"分割"做得更顺手一点，比如：

- 播放头位置一键切开片段（现有分段操作如果能绑定快捷键或按钮就更省事）；
- 切开后两段保持相邻吸附，微调时不散开；
- 切割点吸附到播放头/秒刻度，精确到帧。

这样实拍素材进时间线后，粗剪、挑生成区间就不用来回在剪映里预切了。


## 替代方案
如果透传与现有采样架构冲突较大，退一步仅放开"固定 Guide 片段可以不填提示词"的校验也很有帮助，纯引导段可以作为独立分段存在，用户自行在外部拼接原片；素材规划台若维持现状，补充一份规划编码器接自定义采样链的文档示例也完全能接受。



### #13 [open] ISSUE: [建议 / Enhancement] 素材规划台：支持浏览已上传素材 + 修复平移画布时面板宽度收缩
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/13  by terliu 2026-09-20T03:13:59Z

背景
在使用「素材规划台」（Material Planner / MiniMaxH3TimelinePlanner）节点时，有两个体验问题，我已在本地做了修复并验证。分享给维护者，希望能被合入。两个修复分别位于 minimax_h3_timeline_director.py 与 js/minimax_h3_timeline_director.js，均已通过语法校验（py_compile + node --check）。
优化一：支持从 input 目录浏览 / 选择已上传的素材（无需每次重新上传）
问题
当前的「添加视频 / 图片 / 音频」只有文件上传入口。用户之前已经上传过、或本来就在 input 目录里的素材无法快速复用，必须再次上传。对比 AIMixer 的 ComfyUI_MiniMaxH3_Director，它是用 ComfyUI 原生 LoadImage / LoadVideo 下拉直接选 input 目录文件——本插件缺少这个能力。
根因
插件只接受 addFile 的内联上传路径，没有任何「列出 input 目录已有文件」的入口（无论前端还是后端）。
修复方案
1. 后端新增路由 GET /minimax_h3_timeline/list_input?kind=image|audio|video|all：递归扫描 folder_paths.get_input_directory()，按扩展名过滤，返回 {relative, name, size, kind}（上限 2000，路径永远限制在 input 目录内，不泄露外部路径）。
2. 给 media_info 路由增加 scope="input"：用已有的 _safe_input_path（防路径穿越）探测 input 目录下任意位置的已有文件（原逻辑只认插件自己的上传子目录）。
3. 前端：每个上传按钮旁加一个灰色「浏览」按钮；点击弹出列表（拉取 list_input），点选即复用已有文件、跳过上传——复用原 addFile 的下游逻辑（抽成 addByRelativePath / _pushAsset）。
4. 进阶（已包含在本地修复）：弹窗锚定到对应浏览按钮的右上角，且鼠标滑过列表文件名时弹出图片 / 视频预览（参考 MiniMax H3 Director 的悬浮预览模式，用 ComfyUI /view 接口拉图）。
关键代码片段
Python — 反向扩展名映射 + list_input 路由（加在 MEDIA_EXTENSIONS 之后）
_MEDIA_KIND_BY_EXT = {ext: kind for kind, exts in MEDIA_EXTENSIONS.items() for ext in exts}
_LIST_INPUT_CAP = 2000

@PromptServer.instance.routes.get("/minimax_h3_timeline/list_input")
async def list_input(request: web.Request) -> web.Response:
    kind = str(request.query.get("kind") or "all").lower()
    if kind not in ("image", "audio", "video", "all"):
        kind = "all"
    wanted = (
        set().union(*MEDIA_EXTENSIONS.values())
        if kind == "all"
        else MEDIA_EXTENSIONS.get(kind, set())
    )
    root = Path(folder_paths.get_input_directory()).resolve()
    found: list[dict[str, Any]] = []
    for entry in root.rglob("*"):
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        if ext not in wanted:
            continue
        found.append({
            "relative": entry.resolve().relative_to(root).as_posix(),
            "name": entry.name,
            "size": entry.stat().st_size,
            "kind": kind if kind != "all" else _MEDIA_KIND_BY_EXT.get(ext, "file"),
        })
        if len(found) >= _LIST_INPUT_CAP:
            break
    found.sort(key=lambda item: item["relative"].lower())
    return web.json_response({"files": found})
Python — media_info 增加 scope="input"
scope = str(payload.get("scope") or "upload").lower()
if scope == "input":
    path = _safe_input_path(str(payload.get("filename") or ""), kind)
else:
    path = _safe_uploaded_path(str(payload.get("filename") or ""), kind)
JS — 复用逻辑（节选）
async addByRelativePath(kind, relativePath, displayName) {
  this.setStatus(tr("checking", { name: displayName }), 0.75);
  const infoResponse = await api.fetchApi("/minimax_h3_timeline/media_info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: relativePath, kind, scope: "input" }),
  });
  const info = await responseJSON(infoResponse);
  if (!infoResponse.ok || info.error) throw new Error(info.error || tr("mediaCheckFailed", { status: infoResponse.status }));
  this._pushAsset(kind, info, displayName);
  this.sync(); this.render(); this.setStatus(tr("addedFile", { name: displayName }), 1);
}
openBrowse(kind) 的完整实现（列表弹窗 + 锚定按钮右上角 + 悬停 <img>/<video> 预览）已在本地的 js/minimax_h3_timeline_director.js 实装，可整理进 PR。
安全
- list_input 以 folder_paths.get_input_directory() 为根，仅返回相对路径，不泄露 input 之外的文件。
- media_info 的 scope="input" 走 _safe_input_path（已做路径穿越防护），与节点执行本身接受 input 目录任意文件的既有行为一致。

<img width="1230" height="743" alt="Image" src="https://github.com/user-attachments/assets/a209aa59-4773-41a3-96c7-b710c9781186" />

优化二：修复平移画布时设置面板宽度收缩、无法调参
问题
在画布上拖动（pan）时，素材规划台的深色设置面板宽度会收缩，并且不再跟随节点宽度变化，导致无法继续调参。
根因
scheduleNodeHeightSync() 只把 DOM widget 宿主的 height 用 !important 钉死，而 width 从未钉。面板根 .m3td 是 width:100%，宿主塌它就塌。新版前端在平移画布时会重排 DOM widget，宿主宽度被重算后塌缩，之后没有任何代码把它撑回（插件里虽有 .m3td 上的 ResizeObserver + 节点 onResize/onConfigure 会重跑该同步，但 width 没被钉，塌了也拉不回）。
修复方案
在 scheduleNodeHeightSync() 钉高度同一处，把宿主 width / min-width / max-width 一并钉到 节点宽度 - 20（与 computeSize 口径一致），带 !important；这样前端重排抢不走，且拖拽改节点宽度时面板实时跟随。
const widgetHost = this.root.closest?.(".dom-widget") || this.root.parentElement;
if (widgetHost) {
  widgetHost.classList.add("m3td-widget-host");
  widgetHost.style.setProperty("height", `${desiredHeight}px`, "important");
  widgetHost.style.setProperty("min-height", `${desiredHeight}px`, "important");
  widgetHost.style.setProperty("max-height", "none", "important");
  widgetHost.style.setProperty("overflow", "visible", "important");
  // 钉死宽度，避免平移画布时宿主塌缩导致面板收缩
  const hostWidth = Math.max(100, (this.node.size?.[0] || 860) - 20);
  widgetHost.style.setProperty("width", `${hostWidth}px`, "important");
  widgetHost.style.setProperty("min-width", `${hostWidth}px`, "important");
  widgetHost.style.setProperty("max-width", "none", "important");
}
环境
- ComfyUI（便携版，最新主干）
- 插件：ComfyUI-MiniMaxH3-TimelineDirector（本地 fork，领先上游 4 个提交）
- 前端：ComfyUI 新版 litegraph / Vue 前端
提交意愿
我可以把这两个修复整理成 PR 提交。如维护者认可方向，请告知是否希望我开 PR（含完整 diff）。以上代码片段在本地的 minimax_h3_timeline_director.py 与 js/minimax_h3_timeline_director.js 中均已实装并验证。


### #12 [open] ISSUE: 可否考虑把采样器和VAE解码器独立出来？
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/12  by smashdog 2026-09-18T17:14:32Z

我使用A卡，可以使用ROCm Ninodes插件的采样器及VAE解码器用于加速，但是你的插件把这些东西全集成到一起了，请问是否可以把这些东西独立出来，方便A卡用户使用？
拜谢。


### #11 [open] ISSUE: 膜拜大佬，但有几个需求看是否能解决一下
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/11  by wangduck0633-cyber 2026-09-18T00:02:46Z

大佬，昨天认真观看教程多次。但是还是有各问题看能不能解决：

1、在参考视频生视频时，切点的上下视频动作还是会因为上下视频参考有跳变导致有几帧会卡不上音乐节奏。在丝滑的舞蹈迁移视频中会比较扎眼。（我把上下段间叠加的部分调的很少这个是不是会引起这个问题）请问这个如何解决？是不是我两端叠加的Gen增加会改善？

2、那我舞蹈的总长度要是想跟原视频保持一致，是否要靠手工计算叠加部分的长度呢？有没有可能在未来更新的时候增加自动计算的功能（就是段间参考的那部分，因为叠加而丢掉的时间自动补上）

3、还有一个问题就是，selflift，二次采样那个节点，有没有可能单独加提示词框，能人工限定一下二采。面部出来的效果虽然很好，但是会跟参考图差距较大。

> **Comment by wangduck0633-cyber (2026-09-18T00:03:19Z):**

关注了好几个版本了！真的比上一版牛逼太多！感谢感谢


> **Comment by Songssx (2026-09-18T03:42:42Z):**

经过我的测试，如果不开启二采这个舞蹈动作的问题，可以解决，但是分辨率和速度会相对较低，二采会影响动作


> **Comment by wangduck0633-cyber (2026-09-21T04:05:31Z):**

🫡，respect. 二采如果能有提示词的话，会不会优化脸部生成能尽可能的与参考图一致。或者二采有可能再调用参考图吗？



### #10 [open] ISSUE: 建议添加音乐素材和视频素材编辑功能
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/10  by cibaing 2026-09-17T13:15:22Z

MiniMaxH3-TimelineDirector  确实好用，如果能添加类似 Goohai-MiniMax-H3_Integration 中素材的 截取 功能，对音频和视频进行 编辑截取 那就完美了。


### #9 [closed] PR: fix: pin timeline widget host width to the node width
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/pull/9  by permissionnine9 2026-09-14T16:04:48Z

## Problem

Selecting a **MiniMax H3 Material Planner** (or Timeline Director) node leaves the right-hand side of the node covered by a large grey area. The node is ~1280px wide, but the planner UI only paints the left ~200px and the rest of the node body stays bare.

## Root cause

`scheduleNodeHeightSync()` pins the DOM widget host's *height* with `!important` but never touches its **width**:

```js
widgetHost.style.setProperty("height", `${desiredHeight}px`, "important");
widgetHost.style.setProperty("min-height", `${desiredHeight}px`, "important");
widgetHost.style.setProperty("max-height", "none", "important");
widgetHost.style.setProperty("overflow", "visible", "important");
// no width / min-width
```

The widget's own `computeSize()` already returns the correct value (`node.size[0] - 20`), but the frontend can leave a **stale inline `width`** on the `.dom-widget` host and keeps re-applying it on every layout pass. Measured on a node 1279.65 units wide:

| | value |
|---|---|
| `computeSize()` returns | `1259.65` |
| inline style actually applied | `321px` |

`321 = 340 - 20`, i.e. the fallback branch of `computeSize` (`this.size?.[0] || width || 860`), captured while `node.size[0]` was still `0`.

Since the sync function only ever corrected height, nothing recovered the width — not even resizing the node, because resizing re-runs the same height-only path.

## Fix

Pin the width in the same place, with the same `!important` idiom already used for height.

It goes through **`min-width`** rather than relying on `width` alone: the frontend keeps overwriting `width` on every layout pass but never manages `min-width`, so `min-width` is the only value that reliably sticks.

## Verification

Real browser session against a live ComfyUI with the workflow loaded, using failure injection to reproduce the broken state:

| step | inline `width` | `min-width` | rendered width |
|---|---|---|---|
| after load | `1259.65px` | `1259.65px` | 1260 ✓ |
| stale value injected | `321px` | `0px` | 321 ✗ (bug reproduced) |
| after a resize | `1260.65px` | `1260.65px` | **1261 ✓ (healed)** |

The extension registers normally, `.m3td` mounts, and there are no uncaught exceptions after the change.

## Environment

- ComfyUI `0.33.0`
- `comfyui-frontend-package` `1.49.6`
- Affects `MiniMaxH3TimelinePlanner` and `MiniMaxH3TimelineDirector`

## Unrelated observation

While testing I also hit a second issue, not touched by this PR: both shipped `example_workflows/*.json` hardcode Windows-style model paths (e.g. `minimax_h3\qwen3vl_32b_minimax_h3_int8_convrot.safetensors`), which cannot resolve on Linux or macOS. Happy to send a separate PR for that if it's useful.



### #8 [open] ISSUE: MiniMax H3 有限分段采样 无法校验 images：list index out of range
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/8  by sdpfer 2026-09-12T04:33:26Z

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/854e0cf9-4cbb-48da-b3f9-c09aa35d789c" />


### #7 [open] ISSUE: 全功能合一完全体导演台工作流 → 无法调整生成选区起点（整体也无法移动，终点可调）
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/7  by funs 2026-09-10T07:54:21Z

### 环境 / Environment:
ComfyUI: 0.33.0
插件 / Plugin: ComfyUI-MiniMaxH3-TimelineDirector (Songssx)
系统 / OS: Windows
Python: 3.13.12
PyTorch: 2.13.0+cu130
GPU: RTX 4060 16GB
已安装 imageio-ffmpeg / imageio-ffmpeg installed

### 问题描述 / Problem:
时间线上的青色生成选区，起点（左边缘）无法调整，且选区整体也无法移动。但终点（右边缘）可以正常调整。
On the timeline, the cyan generation region: the start point (left edge) cannot be adjusted, and the whole region cannot be moved. However, the end point (right edge) can be adjusted normally.

### 期望行为 / Expected:
可以拖动青色生成选区的左边缘来调整起点，也可以整体移动选区。
The left edge of the cyan generation region should be draggable to adjust the start point, and the whole region should be movable.

### 实际行为 / Actual:
起点无法调整，整体无法移动；只有终点可以调整。
The start point cannot be adjusted, the whole region cannot be moved; only the end point can be adjusted.

### 请求 / Request:
请问这是已知问题吗？是否有解决方法或 workaround？谢谢！
Is this a known issue? Is there any workaround? Thank you!

> **Comment by Songssx (2026-09-10T08:32:33Z):**

If you set multiple segment boxes, the first box must always start from 0 seconds at the beginning of the video. If there is only one segment box, you can freely drag it to customize its position.


> **Comment by 123MTRC (2026-09-23T17:48:23Z):**

还是不行啊，只有一个片段框，依旧无法自由拖动，选区起点无论怎么调，都会变回0



### #6 [open] ISSUE: Is it possible to add a function to disable the loading script validation check?
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/6  by BenderBlender 2026-09-04T20:56:46Z

Is it possible to add a feature to disable validation of the uploaded script? I'm trying to insert JSON scripts, and the system throws an error, requiring me to format everything in the standard format with shots.


### #5 [open] ISSUE: 每当做 9：16 影片， 一直帮我 rotate. Prom加锁了还是会一样。
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/5  by longsyee 2026-09-04T05:39:06Z

<img width="1792" height="850" alt="Image" src="https://github.com/user-attachments/assets/926c05df-07dc-450e-b850-b2203c5bfbc8" />

Generate 了几十个都是一样。。。the video is rotated!>? 

is there any problem on the node or vae ?? side for the loop version.

[Video Refrence](https://jumpshare.com/s/8suh6CvrvotbeNyNJag5) 


<img width="978" height="642" alt="Image" src="https://github.com/user-attachments/assets/ac6228af-b655-4949-8e20-d06d9be2ef2e" />

音频作为Refrence出来的音效我也不懂他在讲什么。！ 跟不上 narration and promp 


### #4 [open] ISSUE: ui display when insert image have issue with the display only 
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/4  by longsyee 2026-09-03T13:35:04Z

Refer the image when clicking the audio add or image add sometime get cutoff the UI

<img width="619" height="748" alt="Image" src="https://github.com/user-attachments/assets/b86ba866-e9a0-4642-bc2d-fdcf3c8505f0" />



> **Comment by Songssx (2026-09-03T14:57:27Z):**

尝试用node2.0完成，导演台就不会缩回去了，



### #3 [open] ISSUE: 素材规划台点击下就变形了
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/3  by 999k923 2026-09-02T07:19:40Z

<img width="1702" height="1276" alt="Image" src="https://github.com/user-attachments/assets/9c84398d-4b5d-4c69-8b4a-09389ad315e2" />

> **Comment by Songssx (2026-09-02T07:47:36Z):**

可以切换到nodes2.0再切换回来，或者更换一下浏览器尝试一下一下


> **Comment by hallmarlon (2026-09-18T12:40:55Z):**

You can fix this by using Nodes 2.0


> **Comment by terliu (2026-09-20T03:16:49Z):**

 我提交了一个修复方案 #13 



### #2 [open] ISSUE: 文生图bug分析
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/2  by zibuyu2007 2026-08-31T08:24:11Z

## 🔍 问题根因分析

**错误位置**：`minimax_h3_timeline_director.py` 第 1299-1302 行

```python
if not (ref_images or ref_videos or ref_audios or guides):
    raise ValueError(
        "素材规划没有可用参考：请上传图片/音频，或让生成选择区覆盖视频片段/位于两段视频之间。"
    )
```

**问题**：这个 guard 要求**必须**有至少一种参考素材（图片/视频/音频/guide），否则直接报错。但实际上，下游两个函数都完全支持空输入：

| 下游函数 | 空输入行为 | 是否安全 |
|---------|-----------|---------|
| `_execute_h3_independent_first` (L1124) | 创建空 latent，仅用 prompt 文本 token 化生成 conditioning，`ref_blocks` 为空时不附加任何引用 | ✅ 安全 |
| `_apply_h3_guides` (L1111) | `guides` 为空列表时 for 循环直接跳过，返回原始 conditioning | ✅ 安全 |

**结论**：这是一个不合理的硬性拦截。纯文生视频（text-to-video）的完整流程在技术上是可行的，唯独被这个 `raise ValueError` 挡住了。

---

## 🛠️ 修复方案

**改动文件**：`minimax_h3_timeline_director.py`，第 1299-1302 行

**改动前**：
```python
if not (ref_images or ref_videos or ref_audios or guides):
    raise ValueError(
        "素材规划没有可用参考：请上传图片/音频，或让生成选择区覆盖视频片段/位于两段视频之间。"
    )
```

**改动后**：
```python
if not (ref_images or ref_videos or ref_audios or guides):
    log.info("No reference materials provided; running pure text-to-video generation.")
```

- 将 `raise ValueError` 替换为 `log.info` 日志输出
- 其余代码完全不变，流程自然走到 `_execute_h3_independent_first` 进行纯文本生成视频

**改动量**：仅改 1 处，删除 3 行，替换为 1 行。



> **Comment by Songssx (2026-09-01T05:49:46Z):**

好的！下个版本就会修改这个拦截！



### #1 [open] ISSUE: 执行报错
URL: https://github.com/Songssx/ComfyUI-MiniMaxH3-TimelineDirector/issues/1  by littleyeson 2026-08-27T17:12:05Z

节点在执行过程中发生错误。


# ComfyUI Error Report
## Error Details
- **Node ID:** 333
- **Node Type:** MiniMaxH3TimelineDirector
- **Exception Type:** AttributeError
- **Exception Message:** AttributeError: module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'

## Stack Trace
```
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1566, in execute
    width,
    
    height,
    ^^^^^^^
    generation_seconds,
    ^

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1310, in _encode_timeline_plan
    video_audio_output = _timeline_video_audio(timeline)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        "Encoding planned H3 refs at %dx%d: %d images, %d videos, %d paired audios, %d standalone audios, %d native guides; %d output frames",
    

  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1201, in _execute_h3_independent_first
    # audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)

```
## System Information
- **ComfyUI Version:** 0.32.0
- **Arguments:** D:\AI\ComfyUI-aki-v1.1\ComfyUI\main.py --auto-launch --preview-method auto --disable-cuda-malloc
- **OS:** win32
- **Python Version:** 3.13.11 (tags/v3.13.11:6278944, Dec  5 2025, 16:26:58) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.10.0+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 2080 Ti : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 23621992448
  - **VRAM Free:** 22330474496
  - **Torch VRAM Total:** 33554432
  - **Torch VRAM Free:** 33554432

## Logs
```
2026-08-27T23:57:32.942793 - ** ComfyUI startup time:2026-08-27T23:57:32.942860 -  2026-08-27T23:57:32.942926 - 2026-08-27 23:57:32.9422026-08-27T23:57:32.942981 - 
2026-08-27T23:57:32.943046 - ** Platform:2026-08-27T23:57:32.943099 -  2026-08-27T23:57:32.943148 - Windows2026-08-27T23:57:32.943200 - 
2026-08-27T23:57:32.943257 - ** Python version:2026-08-27T23:57:32.943307 -  2026-08-27T23:57:32.943356 - 3.13.11 (tags/v3.13.11:6278944, Dec  5 2025, 16:26:58) [MSC v.1944 64 bit (AMD64)]2026-08-27T23:57:32.943405 - 
2026-08-27T23:57:32.943456 - ** Python executable:2026-08-27T23:57:32.943504 -  2026-08-27T23:57:32.943553 - D:\AI\ComfyUI-aki-v1.1\python\python.exe2026-08-27T23:57:32.943601 - 
2026-08-27T23:57:32.943660 - ** ComfyUI Path:2026-08-27T23:57:32.943709 -  2026-08-27T23:57:32.943757 - D:\AI\ComfyUI-aki-v1.1\ComfyUI2026-08-27T23:57:32.943806 - 
2026-08-27T23:57:32.943856 - ** ComfyUI Base Folder Path:2026-08-27T23:57:32.943931 -  2026-08-27T23:57:32.943985 - D:\AI\ComfyUI-aki-v1.1\ComfyUI2026-08-27T23:57:32.944041 - 
2026-08-27T23:57:32.944098 - ** User directory:2026-08-27T23:57:32.944163 -  2026-08-27T23:57:32.944235 - D:\AI\ComfyUI-aki-v1.1\ComfyUI\user2026-08-27T23:57:32.944284 - 
2026-08-27T23:57:32.944334 - ** ComfyUI-Manager config path:2026-08-27T23:57:32.944382 -  2026-08-27T23:57:32.944918 - D:\AI\ComfyUI-aki-v1.1\ComfyUI\user\__manager\config.ini2026-08-27T23:57:32.944977 - 
2026-08-27T23:57:32.945038 - ** Log path:2026-08-27T23:57:32.945109 -  2026-08-27T23:57:32.945157 - D:\AI\ComfyUI-aki-v1.1\ComfyUI\user\comfyui.log2026-08-27T23:57:32.945211 - 
2026-08-27T23:57:41.466008 - ^[[32m[INFO]^[[0m 
Prestartup times for custom nodes:
2026-08-27T23:57:41.466178 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\rgthree-comfy
2026-08-27T23:57:41.466334 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-easy-use
2026-08-27T23:57:41.466438 - ^[[32m[INFO]^[[0m   19.9 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-Manager
2026-08-27T23:57:41.466525 - ^[[32m[INFO]^[[0m 
2026-08-27T23:57:48.092453 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
2026-08-27T23:57:48.092651 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend eager: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_embedding', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_mxfp8', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_mxfp8', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}
2026-08-27T23:57:48.092846 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend hip: {'available': False, 'disabled': False, 'unavailable_reason': 'PyTorch ROCm/HIP runtime not available', 'capabilities': []}
2026-08-27T23:57:48.092958 - ^[[32m[INFO]^[[0m Found comfy_kitchen backend triton: {'available': True, 'disabled': True, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'int8_linear', 'na3d', 'quantize_and_rotate_rowwise', 'quantize_int8_rowwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'w4a8_int8_linear']}
2026-08-27T23:57:48.105996 - ^[[32m[INFO]^[[0m Checkpoint files will always be loaded safely.
2026-08-27T23:57:48.294130 - ^[[32m[INFO]^[[0m Total VRAM 22528 MB, total RAM 65442 MB
2026-08-27T23:57:48.294273 - ^[[32m[INFO]^[[0m pytorch version: 2.10.0+cu130
2026-08-27T23:57:48.295415 - ^[[32m[INFO]^[[0m Set vram state to: NORMAL_VRAM
2026-08-27T23:57:48.295823 - ^[[32m[INFO]^[[0m Device: cuda:0 NVIDIA GeForce RTX 2080 Ti : cudaMallocAsync
2026-08-27T23:57:48.321308 - ^[[32m[INFO]^[[0m Using async weight offloading with 2 streams
2026-08-27T23:57:48.326378 - ^[[32m[INFO]^[[0m Enabled pinned memory 26176.0
2026-08-27T23:57:48.503416 - ^[[32m[INFO]^[[0m working around nvidia conv3d memory bug.
2026-08-27T23:57:48.507555 - ^[[32m[INFO]^[[0m Using pytorch attention
2026-08-27T23:57:53.546078 - ^[[32m[INFO]^[[0m aimdo: src-win/cuda-detour.c:38:INFO:aimdo_setup_hooks: installing 6 hooks
2026-08-27T23:57:53.577031 - ^[[32m[INFO]^[[0m aimdo: src/control.c:262:INFO:comfy-aimdo NVML pressure enabled
2026-08-27T23:57:53.583028 - ^[[32m[INFO]^[[0m aimdo: src-win/shmem-detect.c:80:INFO:comfy-aimdo WDDM adapter match: NVIDIA GeForce RTX 2080 Ti runtime_luid=00000000:00016eaa dxgi_luid=00000000:00016eaa
2026-08-27T23:57:53.583155 - ^[[32m[INFO]^[[0m aimdo: src/control.c:277:INFO:comfy-aimdo inited for GPU: NVIDIA GeForce RTX 2080 Ti (VRAM: 22527 MB)
2026-08-27T23:57:53.583289 - ^[[32m[INFO]^[[0m DynamicVRAM support detected and enabled
2026-08-27T23:57:53.583395 - ^[[32m[INFO]^[[0m Python version: 3.13.11 (tags/v3.13.11:6278944, Dec  5 2025, 16:26:58) [MSC v.1944 64 bit (AMD64)]
2026-08-27T23:57:53.583486 - ^[[32m[INFO]^[[0m ComfyUI version: 0.33.2
2026-08-27T23:57:53.584849 - ^[[32m[INFO]^[[0m comfy-aimdo version: 0.4.13
2026-08-27T23:57:53.586495 - ^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.31
2026-08-27T23:57:54.220376 - ^[[32m[INFO]^[[0m comfyui-frontend-package version: 1.49.6
2026-08-27T23:57:54.220607 - ^[[32m[INFO]^[[0m comfyui-workflow-templates version: 0.11.43
2026-08-27T23:57:54.220756 - ^[[32m[INFO]^[[0m comfyui-embedded-docs version: 0.5.9
2026-08-27T23:57:54.220875 - ^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.31
2026-08-27T23:57:54.221000 - ^[[32m[INFO]^[[0m comfy-aimdo version: 0.4.13
2026-08-27T23:57:54.224199 - ^[[32m[INFO]^[[0m [Prompt Server] web root: D:\AI\ComfyUI-aki-v1.1\python\Lib\site-packages\comfyui_frontend_package\static
2026-08-27T23:57:54.225529 - ^[[32m[INFO]^[[0m Asset seeder disabled
2026-08-27T23:57:55.478337 - ^[[32m[INFO]^[[0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
2026-08-27T23:58:01.668775 - ^[[34m[ComfyUI-Easy-Use] server: ^[[0mv1.3.6 ^[[92mLoaded^[[0m2026-08-27T23:58:01.668831 - 
2026-08-27T23:58:01.668876 - ^[[34m[ComfyUI-Easy-Use] web root: ^[[0mD:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-easy-use\web_version/v2 ^[[92mLoaded^[[0m2026-08-27T23:58:01.668912 - 
2026-08-27T23:58:01.686884 - ^[[32m[INFO]^[[0m ComfyUI-GGUF: Allowing full torch compile
2026-08-27T23:58:01.700052 - ^[[32m[INFO]^[[0m ### Loading: ComfyUI-Impact-Pack (V8.28.2)
2026-08-27T23:58:02.042380 - ^[[32m[INFO]^[[0m [Impact Pack] Wildcard total size (0.00 MB) is within cache limit (50.00 MB). Using full cache mode.
2026-08-27T23:58:02.044921 - ^[[32m[INFO]^[[0m [Impact Pack] Wildcards loading done.
2026-08-27T23:58:02.173560 - ^[[1m^[[33m[WARNING]^[[0m Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\nodes.py", line 2263, in load_custom_node
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 1023, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-LTXVideo\__init__.py", line 10, in <module>
    from .gemma_encoder import LTXVGemmaCLIPModelLoader, LTXVGemmaEnhancePrompt
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-LTXVideo\gemma_encoder.py", line 25, in <module>
    from .embeddings_connector import Embeddings1DConnector
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-LTXVideo\embeddings_connector.py", line 6, in <module>
    from comfy.ldm.lightricks.model import (
    ...<5 lines>...
    )
ImportError: cannot import name 'interleaved_freqs_cis' from 'comfy.ldm.lightricks.model' (D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy\ldm\lightricks\model.py)

2026-08-27T23:58:02.174041 - ^[[1m^[[33m[WARNING]^[[0m Cannot import D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-LTXVideo module for custom nodes: cannot import name 'interleaved_freqs_cis' from 'comfy.ldm.lightricks.model' (D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy\ldm\lightricks\model.py)
2026-08-27T23:58:02.191001 - ^[[32m[INFO]^[[0m ### Loading: ComfyUI-Manager (V3.39.2)
2026-08-27T23:58:02.191682 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] network_mode: public
2026-08-27T23:58:02.191892 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] ComfyUI per-queue preview override detected (PR #11261). Manager's preview method feature is disabled. Use ComfyUI's --preview-method CLI option or 'Settings > Execution > Live preview method'.
2026-08-27T23:58:02.524552 - ^[[32m[INFO]^[[0m ### ComfyUI Version: v0.33.2 | Released on '2026-08-17'
2026-08-27T23:58:03.891883 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json
2026-08-27T23:58:03.917357 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/alter-list.json
2026-08-27T23:58:04.611379 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/github-stats.json
2026-08-27T23:58:05.061696 - ^[[34m[ComfyUI-RMBG]^[[0m v^[[93m3.0.0^[[0m | ^[[93m43 nodes^[[0m ^[[92mLoaded^[[0m2026-08-27T23:58:05.061756 - 
2026-08-27T23:58:05.822038 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json
2026-08-27T23:58:06.091320 - ------------------------------------------2026-08-27T23:58:06.091419 - 
2026-08-27T23:58:06.091480 - ^[[34mComfyroll Studio v1.76 : ^[[92m 175 Nodes Loaded^[[0m2026-08-27T23:58:06.091521 - 
2026-08-27T23:58:06.091570 - ------------------------------------------2026-08-27T23:58:06.091607 - 
2026-08-27T23:58:06.091659 - ** For changes, please see patch notes at https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/blob/main/Patch_Notes.md2026-08-27T23:58:06.091703 - 
2026-08-27T23:58:06.091738 - ** For help, please see the wiki at https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki2026-08-27T23:58:06.091771 - 
2026-08-27T23:58:06.091805 - ------------------------------------------2026-08-27T23:58:06.091838 - 
2026-08-27T23:58:06.111626 - ^[[36;20m[D:\AI\ComfyUI-aki-v1_x_1\ComfyUI\custom_nodes\comfyui_controlnet_aux] | INFO -> Using ckpts path: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui_controlnet_aux\ckpts^[[0m
2026-08-27T23:58:06.112512 - ^[[36;20m[D:\AI\ComfyUI-aki-v1_x_1\ComfyUI\custom_nodes\comfyui_controlnet_aux] | INFO -> Using symlinks: False^[[0m
2026-08-27T23:58:06.113343 - ^[[36;20m[D:\AI\ComfyUI-aki-v1_x_1\ComfyUI\custom_nodes\comfyui_controlnet_aux] | INFO -> Using ort providers: ['CUDAExecutionProvider', 'DirectMLExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'CPUExecutionProvider', 'CoreMLExecutionProvider']^[[0m
2026-08-27T23:58:06.153528 - DWPose: Onnxruntime with acceleration providers detected2026-08-27T23:58:06.153693 - 
2026-08-27T23:58:06.539518 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json
2026-08-27T23:58:10.931136 - ^[[32m[INFO]^[[0m ('Looking for cached Spacy xx_sent_ud_sm.',)
2026-08-27T23:58:11.755555 - ^[[1;35m
### [START] ComfyUI AlekPet Nodes ^[[1;34mv1.1.8^[[0m^[[1;35m ###^[[0m2026-08-27T23:58:11.755614 - 
2026-08-27T23:58:11.755664 - ^[[92mNode -> ArgosTranslateNode: ^[[93mArgosTranslateCLIPTextEncodeNode, ArgosTranslateTextNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.755704 - 
2026-08-27T23:58:11.755751 - ^[[92mNode -> ChatGLMNode: ^[[93mChatGLM4TranslateCLIPTextEncodeNode, ChatGLM4TranslateTextNode, ChatGLM4InstructNode, ChatGLM4InstructMediaNode, ChatGLMImageGenerateNode, ChatGLMVideoGenerateNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.755797 - 
2026-08-27T23:58:11.755835 - ^[[92mNode -> DeepTranslatorNode: ^[[93mDeepTranslatorCLIPTextEncodeNode, DeepTranslatorTextNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.755884 - 
2026-08-27T23:58:11.755927 - ^[[92mNode -> ExtrasNode: ^[[93mPreviewTextNode, HexToHueNode, ColorsCorrectNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.755970 - 
2026-08-27T23:58:11.756010 - ^[[92mNode -> GoogleTranslateNode: ^[[93mGoogleTranslateCLIPTextEncodeNode, GoogleTranslateTextNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.756050 - 
2026-08-27T23:58:11.756091 - ^[[92mNode -> IDENode: ^[[93mIDENode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.756138 - 
2026-08-27T23:58:11.756193 - ^[[92mNode -> PainterNode: ^[[93mPainterNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.756225 - 
2026-08-27T23:58:11.756261 - ^[[92mNode -> PoseNode: ^[[93mPoseNode^[[0m ^[[92m^[[92m[Loading]^[[0m^[[0m2026-08-27T23:58:11.756293 - 
2026-08-27T23:58:11.756328 - ^[[1;35m### [END] ComfyUI AlekPet Nodes ###^[[0m2026-08-27T23:58:11.756360 - 
2026-08-27T23:58:11.968097 - 
2026-08-27T23:58:11.968176 - ^[[92m[rgthree-comfy] Loaded 48 fantastic nodes. 🎉^[[0m2026-08-27T23:58:11.968226 - 
2026-08-27T23:58:11.968276 - 
2026-08-27T23:58:11.968336 - ^[[33m[rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.
^[[0m2026-08-27T23:58:11.968378 - 
2026-08-27T23:58:11.978857 - ^[[32m[INFO]^[[0m 
Import times for custom nodes:
2026-08-27T23:58:11.979022 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\websocket_image_save.py
2026-08-27T23:58:11.979165 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\minimax_h3_fp16_fix.py
2026-08-27T23:58:11.979259 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-PT_H3ConcatAVLatent
2026-08-27T23:58:11.979378 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\TaiYang2080ti_minimaxh3
2026-08-27T23:58:11.979492 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo
2026-08-27T23:58:11.979578 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\TE-Speed-MiniMaxH3
2026-08-27T23:58:11.979689 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\reservedvram
2026-08-27T23:58:11.979775 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\minimax-h3-fp16-exact-star7
2026-08-27T23:58:11.979862 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-H3-FaceRefine
2026-08-27T23:58:11.979985 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI_UniBlockSwap
2026-08-27T23:58:11.980073 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\minimax-h3-chunk-star7
2026-08-27T23:58:11.980167 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-minimax-h3-blockcache-T8
2026-08-27T23:58:11.980250 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-GGUF
2026-08-27T23:58:11.980341 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector
2026-08-27T23:58:11.980441 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI_IPAdapter_plus
2026-08-27T23:58:11.980523 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI_UltimateSDUpscale
2026-08-27T23:58:11.980610 - ^[[32m[INFO]^[[0m    0.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8
2026-08-27T23:58:11.980685 - ^[[32m[INFO]^[[0m    0.0 seconds (IMPORT FAILED): D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-LTXVideo
2026-08-27T23:58:11.980766 - ^[[32m[INFO]^[[0m    0.1 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\rgthree-comfy
2026-08-27T23:58:11.980847 - ^[[32m[INFO]^[[0m    0.1 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI_Comfyroll_CustomNodes
2026-08-27T23:58:11.980930 - ^[[32m[INFO]^[[0m    0.1 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-kjnodes
2026-08-27T23:58:11.981019 - ^[[32m[INFO]^[[0m    0.1 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui_controlnet_aux
2026-08-27T23:58:11.981097 - ^[[32m[INFO]^[[0m    0.1 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui_nvidia_rtx_nodes
2026-08-27T23:58:11.981174 - ^[[32m[INFO]^[[0m    0.3 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-VideoHelperSuite
2026-08-27T23:58:11.981251 - ^[[32m[INFO]^[[0m    0.3 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8
2026-08-27T23:58:11.981330 - ^[[32m[INFO]^[[0m    0.4 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-Impact-Pack
2026-08-27T23:58:11.981476 - ^[[32m[INFO]^[[0m    0.7 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper
2026-08-27T23:58:11.981571 - ^[[32m[INFO]^[[0m    1.0 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-Manager
2026-08-27T23:58:11.981648 - ^[[32m[INFO]^[[0m    1.4 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-RMBG
2026-08-27T23:58:11.981734 - ^[[32m[INFO]^[[0m    4.7 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\comfyui-easy-use
2026-08-27T23:58:11.981817 - ^[[32m[INFO]^[[0m    5.6 seconds: D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI_Custom_Nodes_AlekPet
2026-08-27T23:58:11.981903 - ^[[32m[INFO]^[[0m 
2026-08-27T23:58:12.004268 - ^[[32m[INFO]^[[0m Context impl SQLiteImpl.
2026-08-27T23:58:12.004382 - ^[[32m[INFO]^[[0m Will assume non-transactional DDL.
2026-08-27T23:58:12.352221 - ^[[32m[INFO]^[[0m Using RAM pressure cache.
2026-08-27T23:58:12.352840 - ^[[32m[INFO]^[[0m Starting server

2026-08-27T23:58:12.353330 - ^[[32m[INFO]^[[0m To see the GUI go to: http://127.0.0.1:8188
2026-08-27T23:58:12.898234 - FETCH ComfyRegistry Data: 5/1792026-08-27T23:58:12.898312 - 
2026-08-27T23:58:14.623774 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:14.625765 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:14.630442 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:14.631799 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:14.634826 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:14.639725 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\events.py", line 89, in _run
    self._context.run(self._callback, *self._args)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\python\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
2026-08-27T23:58:18.651976 - FETCH ComfyRegistry Data: 10/1792026-08-27T23:58:18.652060 - 
2026-08-27T23:58:20.688134 - ^[[32m[INFO]^[[0m got prompt
2026-08-27T23:58:21.314096 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float32
2026-08-27T23:58:21.787269 - ^[[32m[INFO]^[[0m VAE load device: cuda:0, offload device: cpu, dtype: torch.float16
2026-08-27T23:58:21.878280 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-08-27T23:58:21.879265 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-08-27T23:58:22.404056 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-08-27T23:58:25.304442 - FETCH ComfyRegistry Data: 15/1792026-08-27T23:58:25.304505 - 
2026-08-27T23:58:25.535458 - ^[[32m[INFO]^[[0m Encoding planned H3 refs at 1280x736: 2 images, 1 videos, 0 paired audios, 0 standalone audios, 6 native guides; 362 output frames
2026-08-27T23:58:25.588513 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3VideoVAE
2026-08-27T23:58:25.815988 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:58:27.579550 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:58:29.051418 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:58:36.458113 - FETCH ComfyRegistry Data: 20/1792026-08-27T23:58:36.458226 - 
2026-08-27T23:58:37.816600 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-08-27T23:58:38.076789 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-08-27T23:58:42.604925 - FETCH ComfyRegistry Data: 25/1792026-08-27T23:58:42.605233 - 
2026-08-27T23:58:49.446008 - ^[[32m[INFO]^[[0m Interrupting prompt 6bf309b3-9cd6-415d-b00d-b68ddf205eef
2026-08-27T23:58:53.867632 - FETCH ComfyRegistry Data: 30/1792026-08-27T23:58:53.867746 - 
2026-08-27T23:59:00.600497 - ^[[32m[INFO]^[[0m Interrupting prompt 6bf309b3-9cd6-415d-b00d-b68ddf205eef
2026-08-27T23:59:00.791002 - ^[[32m[INFO]^[[0m Interrupting prompt 6bf309b3-9cd6-415d-b00d-b68ddf205eef
2026-08-27T23:59:00.864966 - ^[[32m[INFO]^[[0m Processing interrupted
2026-08-27T23:59:00.866819 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 40.16 seconds^[[0m
2026-08-27T23:59:01.345851 - FETCH ComfyRegistry Data: 35/1792026-08-27T23:59:01.345981 - 
2026-08-27T23:59:09.053745 - FETCH ComfyRegistry Data: 40/1792026-08-27T23:59:09.053824 - 
2026-08-27T23:59:10.603308 - ^[[32m[INFO]^[[0m got prompt
2026-08-27T23:59:14.118616 - ^[[32m[INFO]^[[0m Encoding planned H3 refs at 1280x736: 2 images, 1 videos, 1 paired audios, 0 standalone audios, 6 native guides; 362 output frames
2026-08-27T23:59:14.376140 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:59:15.470584 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:59:16.524457 - FETCH ComfyRegistry Data: 45/1792026-08-27T23:59:16.524614 - 
2026-08-27T23:59:16.951743 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-27T23:59:25.621723 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'
2026-08-27T23:59:25.632999 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1566, in execute
    conditioning, latent, video_audio, standalone_audio = _encode_timeline_plan(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        plan, clip, vae, audio_vae, prompt, ref_image_size
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1310, in _encode_timeline_plan
    conditioning, latent = _execute_h3_independent_first(
                           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        clip, vae, audio_vae, prompt, target_width, target_height, length,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ref_image_size, ref_images, ref_videos, ref_video_audios, ref_audios,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1201, in _execute_h3_independent_first
    audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'

2026-08-27T23:59:25.642430 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 15.02 seconds^[[0m
2026-08-27T23:59:29.159557 - FETCH ComfyRegistry Data: 50/1792026-08-27T23:59:29.159659 - 
2026-08-27T23:59:35.767370 - FETCH ComfyRegistry Data: 55/1792026-08-27T23:59:35.767430 - 
2026-08-27T23:59:48.452272 - FETCH ComfyRegistry Data: 60/1792026-08-27T23:59:48.452332 - 
2026-08-27T23:59:56.938378 - FETCH ComfyRegistry Data: 65/1792026-08-27T23:59:56.938476 - 
2026-08-28T00:00:04.392870 - FETCH ComfyRegistry Data: 70/1792026-08-28T00:00:04.392951 - 
2026-08-28T00:00:11.128486 - FETCH ComfyRegistry Data: 75/1792026-08-28T00:00:11.128562 - 
2026-08-28T00:00:20.146681 - FETCH ComfyRegistry Data: 80/1792026-08-28T00:00:20.146741 - 
2026-08-28T00:00:33.986994 - FETCH ComfyRegistry Data: 85/1792026-08-28T00:00:33.987066 - 
2026-08-28T00:00:40.149756 - FETCH ComfyRegistry Data: 90/1792026-08-28T00:00:40.149829 - 
2026-08-28T00:00:46.457860 - FETCH ComfyRegistry Data: 95/1792026-08-28T00:00:46.457935 - 
2026-08-28T00:00:55.335630 - FETCH ComfyRegistry Data: 100/1792026-08-28T00:00:55.335690 - 
2026-08-28T00:01:07.005545 - FETCH ComfyRegistry Data: 105/1792026-08-28T00:01:07.005626 - 
2026-08-28T00:01:14.135556 - FETCH ComfyRegistry Data: 110/1792026-08-28T00:01:14.135616 - 
2026-08-28T00:01:20.087685 - FETCH ComfyRegistry Data: 115/1792026-08-28T00:01:20.087744 - 
2026-08-28T00:01:26.609714 - FETCH ComfyRegistry Data: 120/1792026-08-28T00:01:26.609795 - 
2026-08-28T00:01:33.752157 - FETCH ComfyRegistry Data: 125/1792026-08-28T00:01:33.752234 - 
2026-08-28T00:01:40.468488 - FETCH ComfyRegistry Data: 130/1792026-08-28T00:01:40.468665 - 
2026-08-28T00:01:46.600726 - FETCH ComfyRegistry Data: 135/1792026-08-28T00:01:46.600791 - 
2026-08-28T00:01:52.786481 - FETCH ComfyRegistry Data: 140/1792026-08-28T00:01:52.786544 - 
2026-08-28T00:01:59.270673 - FETCH ComfyRegistry Data: 145/1792026-08-28T00:01:59.270759 - 
2026-08-28T00:02:05.691037 - FETCH ComfyRegistry Data: 150/1792026-08-28T00:02:05.691101 - 
2026-08-28T00:02:11.753423 - FETCH ComfyRegistry Data: 155/1792026-08-28T00:02:11.753486 - 
2026-08-28T00:02:18.261422 - FETCH ComfyRegistry Data: 160/1792026-08-28T00:02:18.261483 - 
2026-08-28T00:02:24.109233 - FETCH ComfyRegistry Data: 165/1792026-08-28T00:02:24.109294 - 
2026-08-28T00:02:31.069377 - FETCH ComfyRegistry Data: 170/1792026-08-28T00:02:31.069451 - 
2026-08-28T00:02:37.683354 - FETCH ComfyRegistry Data: 175/1792026-08-28T00:02:37.683414 - 
2026-08-28T00:02:43.541970 - FETCH ComfyRegistry Data [DONE]2026-08-28T00:02:43.542029 - 
2026-08-28T00:02:43.985355 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] default cache updated: https://api.comfy.org/nodes
2026-08-28T00:02:44.056651 - FETCH DATA from: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json2026-08-28T00:02:44.056695 - 2026-08-28T00:02:44.958026 -  [DONE]2026-08-28T00:02:44.958093 - 
2026-08-28T00:02:45.131165 - ^[[32m[INFO]^[[0m [ComfyUI-Manager] All startup tasks have been completed.
2026-08-28T00:02:51.097699 - ^[[32m[INFO]^[[0m got prompt
2026-08-28T00:02:54.507463 - ^[[32m[INFO]^[[0m Encoding planned H3 refs at 1280x736: 2 images, 1 videos, 1 paired audios, 0 standalone audios, 6 native guides; 362 output frames
2026-08-28T00:02:54.757324 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:02:55.817008 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:02:57.276311 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:03:05.481253 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'
2026-08-28T00:03:05.509589 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1566, in execute
    ) -> io.NodeOutput:
    
        plan = _create_timeline_plan(timeline_data, width, height, generation_seconds)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        conditioning, latent, video_audio, standalone_audio = _encode_timeline_plan(
        ^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1310, in _encode_timeline_plan
        len(ref_video_audios), len(ref_audios), len(guides), length,
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        clip, vae, audio_vae, prompt, target_width, target_height, length,
    
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1201, in _execute_h3_independent_first
    # audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)
AttributeError: module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'

2026-08-28T00:03:05.518664 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 14.41 seconds^[[0m
2026-08-28T00:05:06.050061 - ^[[32m[INFO]^[[0m got prompt
2026-08-28T00:05:09.116648 - ^[[32m[INFO]^[[0m Encoding planned H3 refs at 1280x736: 2 images, 1 videos, 1 paired audios, 0 standalone audios, 6 native guides; 362 output frames
2026-08-28T00:05:09.380354 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:05:10.404783 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:05:11.849600 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-28T00:05:20.199479 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'
2026-08-28T00:05:20.223198 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1566, in execute
    width,
    
    height,
    ^^^^^^^
    generation_seconds,
    ^
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1310, in _encode_timeline_plan
    video_audio_output = _timeline_video_audio(timeline)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        "Encoding planned H3 refs at %dx%d: %d images, %d videos, %d paired audios, %d standalone audios, %d native guides; %d output frames",
    
  File "D:\AI\ComfyUI-aki-v1.1\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-TimelineDirector\minimax_h3_timeline_director.py", line 1201, in _execute_h3_independent_first
    # audio_latent, ref_audio_t = h3_nodes._encode_ref_audio(audio_vae, soundtrack)
AttributeError: module 'comfy_extras.nodes_minimax_h3' has no attribute '_encode_ref_audio'

2026-08-28T00:05:20.232032 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 14.17 seconds^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)


> **Comment by Songssx (2026-08-28T02:36:43Z):**

ComfyUI要更新到新版本，至少0.34.0以上哈

