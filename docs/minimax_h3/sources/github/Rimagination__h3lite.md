# Rimagination__h3lite

Source: https://github.com/Rimagination/h3lite
Fetched: 2026-11-28

Description: A hardware-aware Codex skill for local MiniMax H3 video generation through ComfyUI
Stars: 529  Updated: 2026-11-27T02:24:50Z



---

## FILE: README.md (https://github.com/Rimagination/h3lite/blob/HEAD/README.md)

# H3 Lite

<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" type="image/webp" srcset="assets/readme/hero-still.webp">
    <source media="(max-width: 640px)" type="image/webp" srcset="assets/readme/hero-mobile.webp">
    <img src="assets/readme/hero-desktop.webp" alt="H3 Lite 顶部动画，MiniMax H3 与 ComfyUI 本地视频生成" width="75%">
  </picture>
</p>

<p align="center">
  <img src="assets/readme/badges.svg" alt="License MIT | Hosts Skill + MCP | Route Low-VRAM Fast" width="544">
</p>

<p align="center">
  <strong>中文</strong> | <a href="README.en.md">English</a> | <a href="https://github.com/Rimagination/h3lite/blob/v0.2.0/qoder-plugin/CHANGELOG.md#新能力">更新说明</a>
</p>

**H3 Lite 让你在自己的电脑上制作有声 AI 视频。** 告诉 AI 助手你想拍什么，或给它一张图作为开场。助手会根据你的显卡选择合适的方案，用 MiniMax H3 生成带环境声、音效或对白的片段。想做一条多镜头短片，也可以用 H3 Lite 做分镜和预演，再逐镜生成和剪辑。

<p align="center">
  <a href="#先看样片">看样片</a> · <a href="#快速开始">快速开始</a> · <a href="#硬件与模型">硬件与模型</a> · <a href="#blender-建模与预演">Blender 建模与预演</a> · <a href="#多镜头制作">短片生成工作流</a>
</p>

<a name="先看样片"></a>

## <img src="assets/readme/section-01-samples.webp" width="100%" alt="01 先看样片">

点封面播放，记得打开声音。这些是项目实际生成的片段，音轨随画面一起生成。

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-red-ball"><img src="docs/gallery/case-red-ball.jpg" width="420" alt="红球弹跳样片，红球旁溅起水花"></a><br>
      <strong>红球弹跳</strong><br><sub>5 秒试跑 · 听撞击声和滚动声</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-golden-retriever"><img src="docs/gallery/case-golden-retriever.jpg" width="420" alt="金毛幼犬醒来样片封面"></a><br>
      <strong>金毛幼犬醒来</strong><br><sub>分段提示 · 从睡着到醒来</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-familyguy"><img src="assets/readme/case-familyguy-preview.webp" width="420" alt="四位家庭成员在客厅看电视"></a><br>
      <strong>客厅换台</strong><br><sub>图生视频 · 从指定画面开始</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-starship-jump"><img src="docs/gallery/case-starship-jump.jpg" width="420" alt="女舰长站在蓝光照亮的舰桥上"></a><br>
      <strong>星舰跃迁</strong><br><sub>8 秒文生视频 · 转场和跃迁音效</sub>
    </td>
  </tr>
</table>

[打开完整画廊，查看 ZCode / Qoder 样片](https://rimagination.github.io/h3lite/) · [对比 Set A 的不同加速设置](https://rimagination.github.io/h3lite/?video=seta-lightx2v-compat)

<a name="快速开始"></a>

## <img src="assets/readme/section-02-quick-start.webp" width="100%" alt="02 生成第一条视频">

首次使用从 **Windows + NVIDIA** 开始。H3 Lite 用 ComfyUI 在本机出片，已有可用的安装就能复用。插件和模型分开安装，模型下载入口见[硬件与模型](#硬件与模型)。

### 1. 把 H3 Lite 装进你的 AI 助手

| 你用什么 | 从哪里开始 |
| --- | --- |
| **Codex / Claude Code** | 单段视频从根目录的 [`SKILL.md`](SKILL.md) 开始，保留旁边的 `scripts/`、`references/` 和 `assets/`。需要完整制作流程时，从 [六个独立 Skill](qoder-plugin/skills/) 选用所需目录。 |
| **Qoder** | 下载 [v0.2.0 插件 ZIP](https://github.com/Rimagination/h3lite/releases/download/v0.2.0/h3lite-0.2.0.zip)，按[安装说明](qoder-plugin/README.md#安装与升级)配置。 |
| **DeepSeek Harness** | 下载或克隆仓库，在 **Plugins → Add plugin** 选择本地 `dsh-plugin` 目录。[接入说明](dsh-plugin/README.md#安装到-dsh) |
| **标准 MCP 客户端** | 用 Node.js 启动随仓库提供的 MCP 服务，指向本机 ComfyUI。[启动与配置](dsh-plugin/README.md#可选标准-mcp) |

### 2. 先检查电脑，再准备模型

把下面这段话交给能访问本机文件和终端的 AI 助手。

```text
请读取 https://github.com/Rimagination/h3lite 的 README。
我想先跑通一个 5 秒、带原生声音的样片。请检查显卡、内存、磁盘和已有的 ComfyUI，
按硬件推荐组件集并复用可用环境。需要下载模型时，先告诉我大小和保存位置。
```

### 3. 生成 5 秒红球样片

这条片只需要看三件事：球有没有弹跳两次、两次撞击有没有声音、MP4 能不能正常播放。

```text
请用 H3 Lite 生成 5 秒横屏视频，640×352，使用快速档和原生音频。
固定低机位。一颗哑光红色橡胶球在灰色混凝土地面上弹跳两次，再向右滚出画面。
保留两次撞击声和滚动声，不配音乐。完成后告诉我 MP4 的位置，并检查时长、音轨和画面运动。
```

### 按手头的素材选生成方式

| 你想控制什么 | 给助手的素材 | 对应路线 |
| --- | --- | --- |
| 从文字构思一个场景 | 按时间顺序描述画面、动作和声音 | `T2VA` |
| 从指定画面接着拍 | 一张开场图，加上之后发生的事 | `I2VA` |
| 从一个画面走到另一个画面 | 首尾两张图，以及中间的动作和运镜 | `FL2VA` |
| 沿用角色、动作或音色 | 至少一张参考图，可加视频或音频，并说明各素材的用途 | `Ref2VA` |

可下载[客厅换台的原始首帧](assets/examples/h3lite-i2va-familyguy-first-frame.png)，或查看 [MiniMax 官方 Ref2VA 输入素材](https://rimagination.github.io/h3lite/?video=case-pink-lamb-reference)。[提示词写法与示例](qoder-plugin/skills/h3-prompt/references/prompt-writing.md)

<a name="硬件与模型"></a>

## <img src="assets/readme/section-03-hardware.webp" width="100%" alt="03 硬件与模型">

先按 **NVIDIA 显卡的显存**选一组模型。系统内存和工作流也会影响可用分辨率。先跑通 640×352，再尝试更大的画面。用 [`h3-workstation`](qoder-plugin/skills/h3-workstation/SKILL.md) 检查显卡、模型组件和 ComfyUI 环境。

| 你想从哪里开始 | 组件集 | 下载 |
| --- | --- | --- |
| **8 GB 显存**。在带 32 GB 内存的 RTX 4070 Laptop 上验证 | **Set A** · 低显存 | [百度网盘](https://pan.baidu.com/s/1IBlH0VY7tWGvxqMtniraow) · `4hri` |
| **16 GB 显存**。在带 32 GB 内存的 RTX 4060 Ti 上验证 | **Set B** · FP8 兼容 | [百度网盘](https://pan.baidu.com/s/1x5GGuJv0h8chApgVoDgIaQ) · `1hjx` |
| **16 GB 加速档**。先装 Set B，再加 Set C | **Set C** · 在支持的画布和组件齐全时自动启用 | [下载与校验说明](qoder-plugin/README.md#set-c-资产网盘) |

走 Set C 的 Ref2VA 路线时，再按下载说明补齐参考生成所需的权重。

### Set C 能省多少时间？

同一台 RTX 4060 Ti 16 GB 上，生成 **640×352、124 帧、4 步**的短片，记录的总耗时从 **52.68 秒**降到 **22.91 秒**。更大的画面、更长的镜头和不同硬件会改变耗时。[实测条件与其他尺寸](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md#validated-16-gb-acceleration-ladder-2026-11-19-set-c-overlay)

<p align="center">
  <img src="assets/readme/set-c-timing.svg" width="100%" alt="RTX 4060 Ti 16 GB 对照测试，基线用时 52.68 秒，完整 Set C 加速用时 22.91 秒">
</p>

<details>
<summary><strong>平台边界、接缝和显存排查</strong></summary>

- **其他平台。** Apple Silicon 可参考独立的 [MLX 社区实现](https://huggingface.co/yunfengwang/mmh3turbo-bundles)。Linux + NVIDIA 与 Intel Mac 的本项目流程尚未验证。
- **平滑背景出现网格。** Set C 的 Sol VAE 可能留下 128 px 固定接缝。先跑 `h3_grid_check.py`，疑似时用 `--video-vae` 单独切换到标准 FP16 视频 VAE。[组件与实测](qoder-plugin/skills/h3-workstation/references/component-sets.md)
- **RTX 50 系出现纹理异常或提示词失效。** 先用同种子、两条提示词跑小样。量化路径异常时，按部署矩阵确认适用的 FP8 DiT 绕行方案。[Blackwell 排查](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md#blackwell-and-quantized-h3-known-upstream-risk)
- **显存不足或长任务中断。** 16 GB 路线建议逐镜串行。排查同机其他 CUDA 进程和模型驻留，长视频拆成多镜并保留每镜 manifest。[显存排查](qoder-plugin/skills/h3-workstation/references/gpu-contention.md)

</details>

<a name="blender-建模与预演"></a>

## <img src="assets/readme/section-04-blender.webp" width="100%" alt="04 Blender 建模与预演">

想让相机绕过主体，或让角色沿着一条路线移动，可以先在 Blender 里用简单几何体搭出角色、道具和场景。这就是**白模**。把走位、动作和相机做成一段预演视频，再交给 H3 作为运动参考。

<p align="center">
  <a href="https://rimagination.github.io/h3lite/?video=zcode-blender-unicorn-compare"><img src="docs/gallery/zcode-blender-unicorn-compare.jpg" width="760" alt="独角兽玩偶的上下对照，上半屏为 Blender 白模，下半屏为 MiniMax H3 生成画面"></a>
  <br>
  <a href="https://rimagination.github.io/h3lite/?video=zcode-blender-unicorn-compare"><strong>播放独角兽玩偶广播体操 · 10.8 秒</strong></a>
  <br>
  <sub>上半屏是 Blender 白模，下半屏是 H3 画面。可以对照看运镜和伸展动作。</sub>
</p>

### 在 Blender 里准备一个镜头

1. **搭出场景。** 用 [`h3-whitemodel`](qoder-plugin/skills/h3-whitemodel/SKILL.md) 摆好主体、地面和道具，先检查它们在画面里的大小和位置。
2. **排走位和机位。** 给主体动作和相机加关键帧，预览运动方向、遮挡和取景。
3. **导出预演视频。** 保留一条清楚可读的动作与运镜参考，供后续 Ref2VA 使用。另备一张角色参考图，用于引导成片中的角色外观。

把这些参考素材接进整部片的方式见[短片生成工作流](#多镜头制作)。

[Blender 工作流](qoder-plugin/skills/h3-whitemodel/SKILL.md) · [把白模视频接进 Ref2VA](qoder-plugin/skills/h3-whitemodel/references/h3-video-reference.md) · [画廊中的 Blender 案例](https://rimagination.github.io/h3lite/#blender-gallery)

<a name="多镜头制作"></a>
<a name="新版工作流"></a>

## <img src="assets/readme/section-05-workflow.webp" width="100%" alt="05 短片生成工作流">

**我们反复试过分镜、参考素材和生成方式，最终用这套流程自动做完了一条 3 分 25 秒的短片。**

[观看成片《隔壁的念经声，千万别让它停｜纸芝居怪谈 第1话》](https://www.bilibili.com/video/BV1j8y8zgE1H/)

给出故事、角色参考和目标时长后，Agent 会把整部片拆成一组短镜头，再逐镜生成、检查，最后剪成一条完整视频。

<p align="center">
  <img src="assets/readme/pipeline.svg" width="100%" alt="H3 Lite 短片生成工作流：分镜与设定 → 逐镜生成 → 验收与重跑 → 剪辑与交付">
</p>

### 一部短片怎么自动做完

1. **定分镜和镜头提示。** [`h3-director`](qoder-plugin/skills/h3-director/SKILL.md) 拆解故事，在 `canon.json` 固定角色、场景和声音设定，在 `show.json` 记录每镜的参考素材、目标时长和验收要求。[`h3-prompt`](qoder-plugin/skills/h3-prompt/SKILL.md) 把动作、运镜和声音写成镜头提示词。
2. **按镜头表逐镜生成。** [`h3-generate`](qoder-plugin/skills/h3-generate/SKILL.md) 先检查项目和素材，再通过 Showrunner 逐镜排队生成。输出和进度写入 `show_manifest.json`，中断后可以继续。
3. **逐镜验收。** 自动检查时长、帧数、画面运动和音轨，再按镜头计划复核角色、动作和前后衔接。
4. **剪辑和交付。** [`h3-post`](qoder-plugin/skills/h3-post/SKILL.md) 按镜头顺序剪接，处理音量和声音衔接，按需加字幕并导出成片。

### 这套框架为什么重要

模型生成有随机性，也就是常说的「抽卡」。镜头一多，角色漂移、动作偏差或声音不一致都可能带来废片和返工。统一设定与参考素材给所有镜头一个共同基准，逐镜验收能及早发现问题。遇到偏差，只调整和重跑受影响的镜头，已经通过的镜头可以继续用于剪辑。

<details>
<summary><strong>用命令行运行或续跑项目</strong></summary>

在仓库根目录运行。把示例路径换成实际 ComfyUI 目录，`--write-config` 会将其写入 `~/.h3lite/config.json`。

```powershell
python qoder-plugin/skills/h3-workstation/scripts/h3_doctor.py --comfyui "D:/MiniMax-H3/ComfyUI" --write-config
```

完成 `show.json`、`canon.json` 和参考素材准备后，下面的命令会先检查计划与参考指纹，再继续未完成的镜头。

```powershell
python qoder-plugin/skills/h3-generate/scripts/h3_showrunner.py --spec projects/ep01/show.json --strict --check-sha --resume
```

要重跑指定镜头，在命令末尾加上 `--only s03 --force`，把 `s03` 换成实际的镜头编号。

从 [`show.json` 示例](qoder-plugin/skills/h3-generate/assets/show.example.json)和[多镜项目规范](qoder-plugin/skills/h3-generate/references/show-contract.md)开始。

</details>

## 文档与许可

- [模型与节点清单](qoder-plugin/skills/h3-workstation/references/component-sets.md) · [部署矩阵](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md) · [视频放大与交付](qoder-plugin/skills/h3-post/references/video-upscale.md)
- [MiniMax H3 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) · [ComfyUI 官方教程](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) · [GitHub Issues](https://github.com/Rimagination/h3lite/issues)

本仓库使用 [MIT License](LICENSE)。MiniMax 权重、ComfyUI 和第三方节点各自适用其上游许可证。

感谢 [LINUX DO](https://linux.do/) 的佬友在项目早期分享使用体验和实测反馈。

## 作者名片

<p align="center">
  <a href="assets/readme/author-card.webp">
    <img src="assets/readme/author-card.webp" alt="川柏同学的名片，含 GitHub Rimagination 和个人微信二维码" width="800">
  </a>
</p>

## Star History

<p align="center">
  <a href="https://www.star-history.com/?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left&amp;theme=dark">
      <img alt="H3 Lite GitHub Star 增长曲线" src="https://api.star-history.com/svg?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left" width="720">
    </picture>
  </a>
</p>

[查看交互版 Star History](https://www.star-history.com/?repos=Rimagination%2Fh3lite&type=date&legend=top-left)


---

## FILE: README.en.md (https://github.com/Rimagination/h3lite/blob/HEAD/README.en.md)

# H3 Lite

<p align="center">
  <picture>
    <source media="(prefers-reduced-motion: reduce)" type="image/webp" srcset="assets/readme/hero-still.webp">
    <source media="(max-width: 640px)" type="image/webp" srcset="assets/readme/hero-mobile.webp">
    <img src="assets/readme/hero-desktop.webp" alt="H3 Lite animated banner for local MiniMax H3 video generation with ComfyUI" width="75%">
  </picture>
</p>

<p align="center">
  <img src="assets/readme/badges.svg" alt="License MIT | Hosts Skill + MCP | Route Low-VRAM Fast" width="544">
</p>

<p align="center">
  <a href="README.md">中文</a> | <strong>English</strong> | <a href="https://github.com/Rimagination/h3lite/blob/v0.2.0/qoder-plugin/CHANGELOG.md#新能力">更新说明</a>
</p>

**H3 Lite lets you make AI videos with sound on your own computer.** Tell your AI assistant what you want to film or give it an image to start from. It will choose a workflow for your GPU and use MiniMax H3 to generate a clip with ambience, sound effects, or dialogue. For a short film, H3 Lite can also help you plan and preview shots, generate them one at a time, and edit them together.

<p align="center">
  <a href="#sample-clips">Sample clips</a> · <a href="#quick-start">Quick start</a> · <a href="#hardware-and-models">Hardware and models</a> · <a href="#blender-modeling-and-previs">Blender modeling and previs</a> · <a href="#production-workflow">Short film workflow</a>
</p>

<a name="sample-clips"></a>

## <img src="assets/readme/section-01-samples.webp" width="100%" alt="01 Watch the clips">

Select a thumbnail and turn on the sound. H3 Lite generated both the picture and the audio you hear in these clips.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-red-ball"><img src="docs/gallery/case-red-ball.jpg" width="420" alt="A red ball with water splashing beside it"></a><br>
      <strong>Bouncing red ball</strong><br><sub>5-second test · impacts and rolling sound</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-golden-retriever"><img src="docs/gallery/case-golden-retriever.jpg" width="420" alt="A golden retriever puppy waking up"></a><br>
      <strong>A puppy wakes up</strong><br><sub>Timed prompt · from sleeping to stirring</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-familyguy"><img src="assets/readme/case-familyguy-preview.webp" width="420" alt="Four family members watching TV in a living room"></a><br>
      <strong>Changing the channel</strong><br><sub>Image to video · starts from a supplied frame</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://rimagination.github.io/h3lite/?video=case-starship-jump"><img src="docs/gallery/case-starship-jump.jpg" width="420" alt="A captain standing on a blue-lit starship bridge"></a><br>
      <strong>Starship jump</strong><br><sub>8-second text-to-video clip · cuts and jump sounds</sub>
    </td>
  </tr>
</table>

[Browse the full gallery, including ZCode and Qoder examples](https://rimagination.github.io/h3lite/) · [Compare Set A acceleration settings](https://rimagination.github.io/h3lite/?video=seta-lightx2v-compat)

<a name="quick-start"></a>

## <img src="assets/readme/section-02-quick-start.webp" width="100%" alt="02 Make your first clip">

Start on **Windows with an NVIDIA GPU**. H3 Lite runs ComfyUI on your computer and can reuse a working installation. Install the plugin and models separately. For model downloads, see [Hardware and models](#hardware-and-models).

### 1. Add H3 Lite to your AI assistant

| Your assistant | Where to start |
| --- | --- |
| **Codex / Claude Code** | For a single clip, use the root [`SKILL.md`](SKILL.md) and keep its `scripts/`, `references/`, and `assets/` alongside it. For the full workflow, choose from the [six standalone skills](qoder-plugin/skills/). |
| **Qoder** | Download the [v0.2.0 plugin ZIP](https://github.com/Rimagination/h3lite/releases/download/v0.2.0/h3lite-0.2.0.zip) and follow the [installation guide](qoder-plugin/README.md#安装与升级). |
| **DeepSeek Harness** | Download or clone the repository. In **Plugins → Add plugin**, select the local `dsh-plugin` directory. [Setup guide](dsh-plugin/README.md#安装到-dsh) |
| **Standard MCP client** | Run the bundled MCP server with Node.js and point it at your ComfyUI installation. [Launch and configuration](dsh-plugin/README.md#可选标准-mcp) |

### 2. Check your machine before downloading models

Paste this into an assistant that can access your files and run terminal commands.

```text
Read the README at https://github.com/Rimagination/h3lite.
I want to make a five-second test clip with native audio. Check my GPU, RAM,
disk space, and any existing ComfyUI installation. Recommend a component set
and reuse a working installation. Tell me the download size and destination
before downloading any models.
```

### 3. Render the five-second red-ball test

Check three things when it finishes. The ball bounces twice, you hear two impacts, and the MP4 plays.

```text
Use H3 Lite to make a five-second 640×352 landscape clip with the fast profile
and native audio. Use a fixed low-angle camera. A matte red rubber ball bounces
twice on gray concrete, then rolls out of frame to the right. Keep the two
impacts and the rolling sound. Leave out music. Return the MP4 path and check
the duration, audio track, and motion.
```

### Decide what to give the model

| What you want to control | What to give your assistant | Mode |
| --- | --- | --- |
| Create a scene from words | Describe the picture, action, and sound in playback order | `T2VA` |
| Start from a specific image | The opening image and what happens next | `I2VA` |
| Connect two images | The start and end frames, plus the action and camera move between them | `FL2VA` |
| Reuse a character, movement, or voice | At least one reference image, optional video or audio, and a role for each reference | `Ref2VA` |

Try the [original first frame from the TV scene](assets/examples/h3lite-i2va-familyguy-first-frame.png) or view the [official MiniMax Ref2VA input example](https://rimagination.github.io/h3lite/?video=case-pink-lamb-reference). [Prompt guide and examples](qoder-plugin/skills/h3-prompt/references/prompt-writing.md)

<a name="hardware-and-models"></a>

## <img src="assets/readme/section-03-hardware.webp" width="100%" alt="03 Hardware and models">

Choose a component set based on your **NVIDIA GPU's VRAM**. System RAM and the workflow also affect the resolution you can use. Start at 640×352 before trying a larger frame. Use [`h3-workstation`](qoder-plugin/skills/h3-workstation/SKILL.md) to check the GPU, model components, and ComfyUI installation.

| Where to start | Component set | Download |
| --- | --- | --- |
| **8 GB VRAM**, tested on an RTX 4070 Laptop with 32 GB RAM | **Set A** · Low VRAM | [Baidu Netdisk](https://pan.baidu.com/s/1IBlH0VY7tWGvxqMtniraow) · `4hri` |
| **16 GB VRAM**, tested on an RTX 4060 Ti with 32 GB RAM | **Set B** · FP8 compatibility | [Baidu Netdisk](https://pan.baidu.com/s/1x5GGuJv0h8chApgVoDgIaQ) · `1hjx` |
| **16 GB with acceleration**, install Set B first | **Set C** · Selected automatically at supported resolutions when its components are present | [Downloads and checksums](qoder-plugin/README.md#set-c-资产网盘) |

If you use Ref2VA with Set C, install the additional reference-generation weights listed in the download guide.

### How much time can Set C save?

On one RTX 4060 Ti 16 GB, total time for a **640×352, 124-frame, four-step** clip went from **52.68 s** to **22.91 s**. Resolution, shot length, and hardware all affect render time. [Test conditions and other resolutions](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md#validated-16-gb-acceleration-ladder-2026-11-19-set-c-overlay)

<p align="center">
  <img src="assets/readme/set-c-timing.svg" width="100%" alt="RTX 4060 Ti 16 GB comparison. The baseline took 52.68 seconds, and the full Set C configuration took 22.91 seconds.">
</p>

<details>
<summary><strong>Platform support, grid seams, and VRAM troubleshooting</strong></summary>

- **Other platforms.** Apple Silicon users can try the independent [community MLX port](https://huggingface.co/yunfengwang/mmh3turbo-bundles). The H3 Lite workflow is unverified on Linux with NVIDIA and on Intel Macs.
- **A grid appears on smooth backgrounds.** Set C's Sol VAE can leave 128 px seams. Run `h3_grid_check.py` and try the standard FP16 video VAE via `--video-vae` if the check flags suspected seams. [Components and test results](qoder-plugin/skills/h3-workstation/references/component-sets.md)
- **An RTX 50-series GPU produces fixed textures or ignores prompts.** Test two prompts at a small resolution with the same seed. If the quantized route shows the issue, see the [Blackwell troubleshooting guide](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md#blackwell-and-quantized-h3-known-upstream-risk) for the FP8 DiT workaround.
- **A render runs out of VRAM or stalls.** Render one shot at a time on the 16 GB route. Check other CUDA processes and models still in memory. Split long videos into shots and keep the per-shot manifests. [GPU contention guide](qoder-plugin/skills/h3-workstation/references/gpu-contention.md)

</details>

<a name="blender-modeling-and-previs"></a>

## <img src="assets/readme/section-04-blender.webp" width="100%" alt="04 Blender modeling and previs">

When you want the camera to circle a subject or a character to follow a path, block out the character, props, and scene with simple shapes in Blender. Render a short preview of the motion and give H3 the video as a reference.

<p align="center">
  <a href="https://rimagination.github.io/h3lite/?video=zcode-blender-unicorn-compare"><img src="docs/gallery/zcode-blender-unicorn-compare.jpg" width="760" alt="Stacked comparison of a toy unicorn in a gray Blender blockout above and the color MiniMax H3 render below"></a>
  <br>
  <a href="https://rimagination.github.io/h3lite/?video=zcode-blender-unicorn-compare"><strong>Play the toy unicorn workout · 10.8 seconds</strong></a>
  <br>
  <sub>Blender blockout on top, H3 render below. Watch how the camera moves around the unicorn as it stretches.</sub>
</p>

### Prepare a shot in Blender

1. **Block out the scene.** Use [`h3-whitemodel`](qoder-plugin/skills/h3-whitemodel/SKILL.md) to place the subject, floor, and props. Check their position and scale in the frame.
2. **Plan the movement.** Keyframe the subject and camera, then preview the shot to check direction, occlusion, and framing.
3. **Export the preview.** Save a clear video of the action and camera move for Ref2VA. Keep a separate character reference image to guide the character's appearance.

See the [short film workflow](#production-workflow) for how to use these references in a multi-shot project.

[Blender workflow](qoder-plugin/skills/h3-whitemodel/SKILL.md) · [Use a blockout video in Ref2VA](qoder-plugin/skills/h3-whitemodel/references/h3-video-reference.md) · [Blender examples in the gallery](https://rimagination.github.io/h3lite/#blender-gallery)

<a name="production-workflow"></a>

## <img src="assets/readme/section-05-workflow.webp" width="100%" alt="05 Short film workflow">

**After trying different shot plans, references, and generation settings, we used this workflow to automatically make a 3:25 short film.**

[Watch the finished film, *The Chanting Next Door: Never Let It Stop* · Kamishibai Horror, Episode 1](https://www.bilibili.com/video/BV1j8y8zgE1H/)

Given a story, character references, and a target runtime, the agent breaks the film into short shots, generates and checks each one, then edits them into a finished video.

<p align="center">
  <img src="assets/readme/pipeline.svg" width="100%" alt="H3 Lite short film workflow: plan the story and shots, generate each shot, review and rerun where needed, then edit and export">
</p>

### How the automation works

1. **Plan the shots and write the prompts.** [`h3-director`](qoder-plugin/skills/h3-director/SKILL.md) breaks down the story. It records recurring characters, settings, and voices in `canon.json`, and each shot's references, target length, and acceptance criteria in `show.json`. [`h3-prompt`](qoder-plugin/skills/h3-prompt/SKILL.md) turns each shot into a prompt for action, camera, and sound.
2. **Render one shot at a time.** [`h3-generate`](qoder-plugin/skills/h3-generate/SKILL.md) checks the project and its assets, then queues shots through the Showrunner. Output paths and progress go to `show_manifest.json` so the run can resume after an interruption.
3. **Review each shot.** Automatic checks cover duration, frame count, motion, and audio. Then use the shot plan to review character consistency, action, and continuity between cuts.
4. **Edit and deliver.** [`h3-post`](qoder-plugin/skills/h3-post/SKILL.md) assembles the shots, adjusts audio levels and transitions, adds subtitles as needed, and exports the film.

### Why the workflow matters

Generation varies from take to take. Across a full film, drifting characters, missed actions, and inconsistent voices can add up to a lot of discarded footage. Shared references give every shot the same starting point, and checking shots as they finish catches problems early. You can adjust and rerun the affected shot while keeping the clips that already passed review.

<details>
<summary><strong>Run or resume a project from the terminal</strong></summary>

Run these commands from the repository root. Replace the example path with your actual ComfyUI directory. The first command saves it in `~/.h3lite/config.json`.

```powershell
python qoder-plugin/skills/h3-workstation/scripts/h3_doctor.py --comfyui "D:/MiniMax-H3/ComfyUI" --write-config
```

After preparing `show.json`, `canon.json`, and the reference files, use the next command to validate the plan and reference hashes before resuming unfinished shots.

```powershell
python qoder-plugin/skills/h3-generate/scripts/h3_showrunner.py --spec projects/ep01/show.json --strict --check-sha --resume
```

To rerun an individual shot, add `--only s03 --force` to the command and replace `s03` with the shot ID.

Start with the [example `show.json`](qoder-plugin/skills/h3-generate/assets/show.example.json) and the [multi-shot project guide](qoder-plugin/skills/h3-generate/references/show-contract.md).

</details>

## Documentation and license

- [Models and nodes](qoder-plugin/skills/h3-workstation/references/component-sets.md) · [Deployment matrix](qoder-plugin/skills/h3-workstation/references/deployment-matrix.md) · [Upscaling and delivery](qoder-plugin/skills/h3-post/references/video-upscale.md)
- [MiniMax H3 upstream repository](https://github.com/MiniMax-AI/MiniMax-H3) · [Official ComfyUI tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) · [GitHub Issues](https://github.com/Rimagination/h3lite/issues)

This repository uses the [MIT License](LICENSE). MiniMax weights, ComfyUI, and third-party nodes have their own upstream licenses.

Thanks to the [LINUX DO](https://linux.do/) community for sharing their experiences and test results during the early releases.

## Author card

<p align="center">
  <a href="assets/readme/author-card.webp">
    <img src="assets/readme/author-card.webp" alt="Rimagination's author card with GitHub handle and personal WeChat QR code" width="800">
  </a>
</p>

## Star History

<p align="center">
  <a href="https://www.star-history.com/?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left&amp;theme=dark">
      <img alt="H3 Lite GitHub star history" src="https://api.star-history.com/svg?repos=Rimagination%2Fh3lite&amp;type=date&amp;legend=top-left" width="720">
    </picture>
  </a>
</p>

[Open the interactive Star History chart](https://www.star-history.com/?repos=Rimagination%2Fh3lite&type=date&legend=top-left)


---

## FILE: SKILL.md (https://github.com/Rimagination/h3lite/blob/HEAD/SKILL.md)

---
name: h3lite
description: Use when configuring, repairing, planning, or running MiniMax H3 locally on a Windows NVIDIA computer, especially when installation compatibility, component sets, resolution, generation-time budget, path, low-VRAM risk, or H3 prompt mode must be chosen from hardware and user requirements.
---

# MiniMax H3 Local Video

Use this skill to turn a user's local computer into a reproducible MiniMax H3 audio-video workstation and to generate short clips. The validated primary route is Windows + NVIDIA + ComfyUI; macOS Apple Silicon is a community/experimental alternative, not an equivalent tested backend. Treat platform selection, path selection, hardware-aware planning, prompt writing, execution, timing, and verification as one workflow. The default is the validated fast route; other modes must be chosen explicitly or justified by a time budget.

## Platform scope and routing

Detect the operating system and accelerator before giving installation commands or model links. Do not give CUDA, Windows virtual-environment, `.bat`, or Windows path instructions to a macOS user.

| Platform | h3lite status | Guidance |
| --- | --- | --- |
| Windows + NVIDIA CUDA | Primary, locally validated | Use the ComfyUI/H3 Lite fast path and the Windows doctor/planner/preflight scripts. |
| macOS Apple Silicon | Community alternative, not h3lite-equivalent | Explain that the documented ComfyUI route is not validated on Metal. Offer the MLX/`mmh3turbo` route as an optional external alternative, with its own weights, commands, and timing. |
| macOS Intel | Not recommended | Do not promise local H3 generation; suggest a hosted/API or another backend. |
| Linux + NVIDIA | Unverified | The model concepts may transfer, but Windows paths, packaged nodes, and timings do not. Require an explicit experimental choice. |

For a non-Windows user, preserve the useful cross-platform guidance (prompt structure, 32-pixel canvas alignment, low-resolution preview, disk/log/output checks), but label every resource and timing as platform-specific. A community implementation that can run H3 on a Mac is evidence for feasibility, not evidence that this ComfyUI skill supports Metal.

The Mac alternative described in the project notes uses MLX/Metal and a third-party `mmh3turbo` package. If a Mac user explicitly chooses that route, point them to the author's [community bundle](https://huggingface.co/yunfengwang/mmh3turbo-bundles) and package (`uvx mmh3turbo`), and state that these are external resources with independent licensing, updates, and validation. It may reduce the download footprint with GGUF/4-bit bundles, but it is outside this skill's tested component sets. Do not silently install it, mix its weights with ComfyUI models, or present its 30–43 minute 5-second 720p timings as Windows benchmarks.

## Agent production workflow

For complex creative requests, use the compact workflow contract in
[`references/agent-workflow.md`](references/agent-workflow.md):

```text
intent route → reference/identity anchors → prompt enhancement → execute → verify
```

Route from the user's input and acceptance criteria, not from a style adjective.
Use `I2VA` for a specified opening frame, `FL2VA`/`L2VA` for endpoint anchors,
and `Ref2VA` only after its model, text encoder, and workflow are confirmed.
For recurring characters or multi-shot work, define stable subject/reference
labels, what must be retained, what may change, and what drift is forbidden
before writing the timeline. This workflow pattern is local and does not add a
cloud service, MCP dependency, second model, or second inference pass.

When a creative brief is vague (for example, it only says “more cinematic”
or “make it a nice 3D animation”), optionally read
[`references/prompt-assist.md`](references/prompt-assist.md). It adapts the
public Higgsfield prompt structure—stable style/identity lock, one clear scene
action, physical camera motion, audio, and compact anti-drift constraints—to
H3's native fields. Use it as a writing aid only: do not call Higgsfield, copy
model-specific flags or capabilities, or let a web lookup change the local
route, resolution, component set, or verification rules. If a live lookup is
needed, follow the host `web-access` skill and use public pages only.

## Operating rules

- **Hot path first:** for an ordinary text-to-video request on an already validated installation, run `scripts/h3_fastpath.py` once. It combines `/system_stats`, fresh-cache reuse, in-process planning/preflight, queue submission, and one bounded completion watch. Do not issue repeated one-shot status calls, reread the full reference set, run `--help`, or ask nonessential questions during this path. If the command yields a running terminal cell, wait on that cell; do not start another monitor.
- **Keep cold work out of the hot path:** model download manifests, hash or size verification, Torch/custom-node repair, repository checks, browser workflow discovery, and full recursive doctor scans belong to installation, migration, repair, or first-run validation. A normal generation on an unchanged machine must not pay those costs.
- **Cold path can be heavier when it prevents hour-scale waste:** during installation or repair, verify download source, target folder, expected size/hash when available, runtime imports, and model-role mapping before queueing. Store the result so later prompts reuse it instead of repeating it.
- Inspect before changing anything. Run `scripts/h3_doctor.py --json` and locate the target ComfyUI directory before installing packages, nodes, or weights.
- On first contact, run a platform/accelerator check before the Windows doctor. If the machine is not Windows + NVIDIA, stop the CUDA installation branch and route the user using the platform matrix above.
- Prefer an isolated ComfyUI directory when no installation is supplied. Never overwrite an existing installation or silently replace model files.
- Keep the deployment path configurable. Do not copy paths from another computer into scripts or workflows.
- Report required disk space before large downloads. Use resumable downloads and verify file size or hash when a source provides one.
- Before a long generation or multi-shot batch, check free space and pagefile headroom and keep per-shot logs. A pipeline that filters away the process exit code or traceback is not a successful run; preserve the full log and stop on the first failed shot.
- If the user can only download from the public internet, run a cold-path download plan before fetching multi-GB files: test candidate raw URLs with a small ranged download, choose the fastest stable source, estimate wall-clock time, then use resumable `.part` downloads. Do not pretend scripts can beat the user's real bandwidth.
- Before downloading large assets, run the doctor compatibility probe. Stop on a Torch import error; treat a comfy-kitchen/Torch mismatch as a repair decision, not a post-download surprise. Do not silently substitute model files or start unlimited parallel downloads.
- Treat the diffusion checkpoint, text encoder, Turbo LoRA, workflow, and node revisions as one component set. Read `references/component-sets.md` during installation, migration, model replacement, or kernel repair. Never construct an unvalidated set from individually plausible filenames.
- Use `--component-set auto` for one unambiguous installed set, or explicitly select `A`/`validated-low-vram-a` or `B`/`portable-16gb-b` when both sets are installed. Record the selected set in the run manifest; never resolve a partial set role by role.
- Prefer the maintained Baidu package for the registered A/B component sets. Keep the selected set atomic, and respect the licenses of model weights and third-party nodes when using either the package or upstream sources.
- Prefer the ComfyUI HTTP API with an API-format workflow JSON. Use browser/CDP capture only as a recovery path when no reusable workflow JSON exists.
- Check `http://127.0.0.1:8188/system_stats` before starting anything. If ComfyUI is already healthy, reuse it and do not restart it or rediscover its workflow history.
- Preserve MiniMax H3's audio path and flow/sigma-shift handling when the user wants native audio. Do not remove audio VAE, audio conditioning, or the H3 sampling node merely to make a graph look simpler.
- **Zero-inference optimization constraint:** hardware compatibility checks, timing calibration, face routing, and media QA may run before or after generation, but must not add sampling steps, extra generation models, or a second video inference pass. Keep the selected graph unchanged unless the user explicitly requests a different quality profile.
- **Face-quality routing:** if the user needs a recognizable or speaking human face, do not treat low-VRAM W4A8 T2VA at 640x352 as a final-quality route. Prefer I2VA with a clear first-frame reference; prefer Ref2VA when identity must persist across shots. Read `references/face-quality.md`, confirm `MiniMaxH3ReferenceToVideo` through `/object_info`, and confirm the matching reference-capable text encoder/projection and workflow before selecting that route. A registered node alone is not enough; the bundled Ref2VA templates are an experimental local path until a complete run passes media and manual identity QA.
- **Anchor before prompt:** for multi-shot or identity-sensitive requests, first create an internal anchor sheet with stable subject/picture labels, retention rules, allowed changes, and forbidden drift. Use the same labels in the prompt, output prefix, and run manifest; read `references/agent-workflow.md` for the compact contract.
- **Assist vague creative briefs without inventing facts:** when the request lacks a concrete camera, action, sound, or finish, read `references/prompt-assist.md` and use its bounded defaults or ask one targeted question if the omission changes the route or acceptance criteria. A public Higgsfield lookup is optional and pattern-only; fall back to the local references when browsing is unavailable.
- On current ComfyUI builds, the API class `MiniMaxH3SigmaShift` is the native `ModelSamplingMiniMaxH3` node and uses the merged `ModelSamplingAV` video/audio schedule fix. Detect it by `/object_info` or the local source before adding a custom dual-clock sampler; do not duplicate the fix merely because the API class keeps its compatibility name.
- Run the read-only planner before a non-trivial generation. It must report selected mode, resolution, steps, cache policy, paths, and an estimated time range. Do not present an estimate as a guarantee.
- Run the read-only preflight after the doctor and planner. Treat low available RAM/VRAM as a caution, but stop when the pagefile is critically low, required assets are missing, or the doctor recommends an alternative backend.
- Do not perform a full recursive doctor scan for every prompt. Cache the environment report under `<ComfyUI>/user/h3lite_runs/_environment/`; reuse it for a normal session (normally no older than 30 minutes), invalidate it after ComfyUI/model/node/driver changes or a failed run, and use `h3_preflight.py --refresh-runtime` for volatile resource fields.
- Do not revalidate large model files before every prompt. Trust the cached download/component manifest unless the file is missing, has a different size/mtime than recorded, the user changed components, or the previous run failed with a model/node/loader error.
- For registered Set B files, require the recorded SHA-256 on first use or after a size/mtime change. A same-size corrupted W4A8 checkpoint produced colored mosaic frames, so byte count alone is not proof of integrity; reuse the cached integrity result on unchanged files.
- Treat every submission as an auditable run: save the effective prompt, mutated API workflow, configuration fingerprint, queue ID, actual execution time, and verified output in the run manifest.
- For identity-sensitive or multi-shot runs, the runtime also writes `anchors.json` beside `manifest.json` and records advisory `anchor_qa` comparisons; these signals support manual continuity review but are not face recognition.
- Keep agent-facing status compact: omit ComfyUI's full history graph by default; use verbose history only when diagnosing a failure.
- Never submit an identical configuration while its manifest is `submitting`, `queued`, or `running`. Return the existing prompt ID instead; use `--allow-duplicate` only when the user explicitly asks for a second identical run.
- Treat low-VRAM timing as an empirical estimate. The first run can be much slower because kernels compile and weights move between system RAM and VRAM.
- For expensive renders, use a cheap preview pass first: validate the complete prompt/shot list at the smallest supported canvas (for example 256p or the local fast bucket), then promote only approved shots to the requested resolution. This is especially important for multi-shot work; it is a planning optimization, not a second quality-generation pass for a single requested clip.
- Prefer `NORMAL_VRAM` when a validated 16 GB system can keep Set B resident. In a same-model/workflow/prompt/seed 640x352 comparison, an RTX 4060 Ti 16 GB run took 77.08 seconds versus 591.22 seconds on an RTX 4070 Laptop 8 GB using `LOW_VRAM`; treat dynamic loading/offload as the main operational explanation, not as a pure GPU benchmark or a promise.
- When launching ComfyUI as a background process, redirect stdout and stderr to persistent files. A detached pipe can become invalid after the launching session is cleaned up, leaving ComfyUI alive but causing tqdm/logger writes to fail with `OSError: [Errno 22] Invalid argument`. On that signature, restart ComfyUI with persistent logs; do not redownload models or rerun a full doctor unless the restart exposes another error.
- Keep media verification attached to the selected ComfyUI root. The verifier searches system `PATH`, `H3LITE_FFPROBE`, and common locations in or beside `<ComfyUI>` for `ffprobe`; both `h3_generate --watch` and standalone `h3_status` must receive or infer that root. Standalone status may infer the parent only when `--output-dir` points exactly `<ComfyUI>\output`; otherwise pass `--comfyui` explicitly. Treat `ffprobe_not_found` as a missing verifier, not evidence that generation failed, and do not requeue the video until the existing output has been inspected.
- Treat run-history cleanup as explicit maintenance, never hot-path work. Use `scripts/h3_cleanup.py` in dry-run mode first and require `--apply` before deleting eligible run snapshots. Preserve `_environment`, `_hotpath`, `_workflows`, `_experiments`, prompt folders, timing data, and generated output files.

## Preferred component download source

For installation or repair, use the maintained Baidu package before assembling
the set from multiple upstream repositories. Select one complete package after
the hardware check; do not ask the user to download both sets or mix their
exclusive files.

| Default hardware match | Package | Share link | Code |
| --- | --- | --- | --- |
| About 8 GB VRAM, low-VRAM fast route | Set A | [Baidu Netdisk](https://pan.baidu.com/s/1IBlH0VY7tWGvxqMtniraow) | `4hri` |
| 16 GB-class VRAM, FP8 compatibility route | Set B | [Baidu Netdisk](https://pan.baidu.com/s/1x5GGuJv0h8chApgVoDgIaQ) | `1hjx` |

Guide the user to open the matching link, enter the code, and download the
whole package. If the `baidu-drive` skill or a Baidu Drive connector is
available, use it for the download; otherwise give the link and code directly
and continue after the user places the files locally. Merge the package's
`models` and
`custom_nodes` folders into the selected `<ComfyUI>` root, then import or copy
the packaged workflows and keep `component-manifest.json` with the install.
Run the doctor after the merge.

Set A contains the INT4 text encoder and optional low-VRAM acceleration nodes.
Set B contains the FP8 text encoder and validated compatibility workflows. Both
packages include their own shared ClipProj and VAE files, so a user only needs
one link. If the Baidu package is unavailable or the user explicitly requests
upstream downloads, use the exact sources, filenames, sizes, and hashes in
`references/component-sets.md`.

## Installation target contract

Before installing, downloading, or moving any component, establish one explicit
deployment target and state it to the user:

```text
Install mode: reuse-existing | current-project | dedicated-folder
ComfyUI: <absolute path>
Models: <ComfyUI>\models
Custom nodes: <ComfyUI>\custom_nodes
Output: <ComfyUI>\output
```

Use these rules:

- `reuse-existing`: use the exact existing ComfyUI path supplied by the user or discovered and confirmed by the user. Do not clone, reinstall, or create a second model directory.
- `current-project`: keep everything under the active workspace in `<workspace>\.h3lite\ComfyUI` so the project-scoped choice is unambiguous and does not scatter models across the repository.
- `dedicated-folder`: use the user's absolute path, preferably a non-repository path such as `D:\AI\MiniMax-H3\ComfyUI` or `F:\MiniMax-H3\ComfyUI`. Put the venv, custom nodes, models, user data, and output under this ComfyUI root.
- If no existing installation and no target path are available, recommend `dedicated-folder` and ask the user to confirm the absolute path before downloading large files. Never silently choose a drive or install into the current project root.
- If the user says “当前项目” without naming the workspace, resolve and display the active workspace path before proceeding. If the user gives a path ending in `ComfyUI`, use it directly; if they give a parent install folder, append `ComfyUI` and display the resulting path for confirmation.

After the target is selected, read `references/deployment-matrix.md` and present
the component checklist, exact destination folders, estimated disk budget, and
the launch command. Install or repair in this order: runtime prerequisites,
ComfyUI, required custom nodes, model files, doctor verification, then launch.
Do not start a generation while any required node class or model asset is
missing.

## Adaptive planning contract

Before generation, separate the user's requirements into four independent
choices:

- **Intent:** `fast` (default), `balanced`, or `quality`.
- **Wall-clock budget:** maximum expected generation time, such as “within 10
  minutes”; this is different from the requested clip duration.
- **Canvas:** `auto`, landscape/`16:9`, portrait/`9:16`, square/`1:1`, or an
  explicit `WIDTHxHEIGHT`. Keep pixel size as a ComfyUI parameter, not as
  prompt prose. The prompt should describe framing such as landscape
  composition, medium close-up, macro shot, or slow push-in.
- **Target path:** `reuse-existing`, `current-project`, or `dedicated-folder`.

Run the planner after the doctor scan. It is read-only and does not create
folders:

```powershell
python scripts/h3_plan.py `
  --root <hardware-or-disk-root> `
  --comfyui <ComfyUI-path> `
  --install-mode reuse-existing `
  --mode auto `
  --target-minutes 10 `
  --aspect landscape `
  --megapixels 0.4 `
  --video-seconds 5 `
  --json
```

Use the planner result as the source of truth for `--profile`, `--resolution`,
`--steps`, `--length`, and `--fps`. If no quality or time requirement is
given, `auto` selects `fast` and preserves the validated 640x352, 4-step,
Block-Cache route. On an 8 GB laptop, `balanced` keeps 640x352 and increases
steps while bypassing T8 Block Cache; `quality` does the same with 8 steps.
Only the mid/high-VRAM plan promotes balanced/quality to an official-style
ComfyUI bucket by default.

ComfyUI's native H3 templates commonly use `ResolutionSelector`: aspect ratio
+ target megapixels + a 32-pixel multiple. For example, `16:9`, `0.4 MP`,
`multiple=32` gives `864x480`. Treat that as the normal official-style H3
canvas, while H3 Lite's `640x352` remains the low-VRAM fast baseline. If the
user asks for "official template size", "normal quality", or "0.4MP", use
`--megapixels 0.4` unless hardware/time preflight blocks it.

The 32-pixel alignment is a practical model/decoder constraint: the VAE's
16-pixel spatial reduction and the DiT's 2-pixel patching must both align. Do
not advertise consumer labels such as "720p" as exact model sizes when the
nearest legal canvas is different; report the actual canvas (for example
`1280x704`) and keep the requested label as a human-friendly preset name.

For a prompt with several ordered actions, prioritize adherence before pixels.
Use the fast 640x352 baseline first; 736x416 is an experimental adherence
bucket between the fast canvas and 0.4 MP. Do not assume 768p follows complex
shot structure better: community reports describe stronger Ref2VA/keyframe
adherence around 352p-416p, while extra steps mainly improve coherence and
detail. Promote 736x416 only after a same-prompt comparison succeeds locally.

When the user supplies an explicit resolution, treat that canvas choice as
already confirmed: issue one concise OOM/time warning, then continue unless
preflight is `blocked`. Do not ask the same resolution question again. If a
separate wall-clock budget conflicts with the explicit canvas, state the
conflict once and keep the user's explicit canvas. Never silently trade away
the audio path or change the installation target to make the estimate fit.

## Fast path for ordinary text-to-video

Use this path for a short H3 clip on a machine that passes the low-VRAM doctor
check. With no frame arguments it is T2VA; adding `--first-frame` or
`--last-frame` selects the native reference route automatically:

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --prompt-text "<rewritten H3 prompt>" `
  --resolution 640x352 `
  --video-seconds 5 `
  --filename-prefix video/H3Lite_my_clip `
  --json
```

For image-to-video, pass the reference image directly. The helper stages an
external image into `<ComfyUI>/input` using a content-addressed filename, then
connects it to the native `MiniMaxH3ImageToVideo` node. No extra I2V node or
manual JSON editing is required:

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --mode i2va `
  --first-frame <path-to-first-frame.png> `
  --prompt-text "<rewritten I2VA prompt>" `
  --resolution 640x352 `
  --video-seconds 5 `
  --filename-prefix video/H3Lite_i2va `
  --json
```

Use `--mode fl2va --first-frame <first> --last-frame <last>` for fixed first
and last images, or `--mode l2va --last-frame <last>` for a last-frame route;
the helper removes the I2V template's first-frame placeholder automatically in
the latter case.
When `--mode auto` is left in place, the mode is inferred from the supplied
frame arguments. The low-VRAM baseline remains 640x352, 124 frames, 4 steps,
and native H3 audio. A validated RTX 4070 Laptop 8 GB I2VA run at that
baseline took about 12 minutes; treat that as local empirical timing, not a
guarantee.

已配置环境的复跑路径就是这一条命令；不要再分别调用 doctor、plan、preflight、generate 和 status。Windows/Git Bash 路径请写 `F:/MiniMax-H3/ComfyUI`，不要写 `/f/MiniMax-H3/ComfyUI`。

The helper reuses `<ComfyUI>/user/h3lite_runs/_environment/doctor.json` for 30
minutes, invalidating it only for a cold start, explicit `--force-doctor`, an
installation change, or failure recovery. It writes the plan and prompt into
the run root, keeps the native audio path unless the prompt explicitly asks
for complete silence, and waits with one compact `--watch` monitor. An
explicit resolution is already confirmed; continue after one concise risk
warning unless preflight is `blocked`. `--dynamic-check` is on by default;
use `--skip-dynamic-check` only for an intentionally static clip.

Hot-path budget rule: spend only the time needed to rewrite the prompt, refresh
volatile runtime status, submit, watch, and verify. Do not check Git remotes,
download pages, model hashes, dependency versions, or official docs during a
normal generation unless the previous command returns a concrete error pointing
there. If cache is valid, proceed directly to generation.

### Native Windows progress window

On Windows, the fastpath opens the native monitor by default, so every normal
desktop generation has a visible progress window without an extra flag. Use
`--no-monitor-gui` for a run that must stay terminal-only; `--monitor-gui`
explicitly forces it on. The window discovers the fresh H3 run manifest
automatically. It uses a native Windows Tkinter window, reuses the manifest's
ComfyUI `client_id`, and listens to the native `/ws`
channel, including the newer `progress_state` node events, while HTTP polling
supplies queue state, elapsed/estimated time, GPU memory, RAM, pagefile, output
path, and failure state. It is monitor-only: closing it does not interrupt
generation.
The node count is structural workflow progress, not elapsed-time progress: H3
nodes have very different runtimes. The window therefore shows node completion
separately, displays the current node's observed runtime, and keeps ETA on the
empirical timing estimate instead of treating `4/5` as `80%` of the time. The
track is segmented by workflow node so completed, active, and pending nodes
remain visually distinct.
The default window is `760x620`; its content area has a vertical scrollbar and
mouse-wheel support for smaller displays or larger system scaling.
The monitor JSON marks these semantics as `progress_basis` (`node_completion`
or `sampling_steps`) and `eta_basis` (`empirical` or `live_progress`) so an
Agent can report them without inventing a time percentage.
When available, elapsed time comes from the run manifest's measured execution
time; the wall-clock timestamp is only the fallback before that field exists.
Old `running` manifests are ignored during automatic discovery; pass
`--prompt-id` to inspect a specific historical run.

To open the monitor independently:

```powershell
python scripts/h3_monitor_gui.py `
  --comfyui <ComfyUI-path>
```

Use `--once --no-websocket` for a one-shot JSON diagnostic. If the optional
WebSocket client is unavailable, the window remains usable through HTTP
polling, but the progress track stays static and explicitly reports that live
quantifiable progress is unavailable. It never substitutes an animated bar
for a measured percentage. An MCP wrapper is unnecessary for this local GUI;
an Agent can query the same monitor JSON separately when it needs status.

Use the lower-level `h3_doctor.py`, `h3_plan.py`, `h3_preflight.py`,
`h3_generate.py`, and `h3_status.py` commands only for installation, recovery,
custom workflows, or diagnosis. Never replace the fastpath with repeated
one-shot status commands during a normal generation.

Use the history/object-info/browser fallback only when the bundled template is
incompatible, a Ref2VA/custom graph is needed, or the queue reports a missing
node/model. T2VA, I2VA, FL2VA, and L2VA are now first-class bundled routes.

## Workflow

### 1. Diagnose the computer

Run:

```powershell
$environment = '<ComfyUI>\user\h3lite_runs\_environment'
python scripts/h3_doctor.py `
  --json `
  --report-file "$environment\doctor.json" `
  --root <chosen-root> `
  --comfyui <ComfyUI-path>
```

Record GPU name and VRAM, system RAM, free disk, Python, CUDA/PyTorch visibility, ComfyUI location, model presence, and custom-node presence. If there is no NVIDIA CUDA device, do not promise this local CUDA route; explain the limitation and offer API/cloud or another backend as an alternative.

The doctor also records available physical RAM, available Windows pagefile, and
GPU compute processes. Low available RAM or VRAM is a warning because the
validated 8 GB route can still finish with offload. A nearly exhausted
pagefile is different: it previously caused `hostbuf_file_reader_read failed`
and system-level paging failures, so `h3_preflight.py` blocks that run.

Use `references/deployment-matrix.md` to choose a profile. For an 8 GB laptop, start with the tested W4A8 profile, not the larger official INT8/32B profile.

If the user asked for installation or repair, report the selected installation
target before the doctor result. The doctor is read-only; it does not install
ComfyUI, nodes, Python packages, or model weights.

### 2. Select a profile

- **Fast (default):** a registered W4A8/4B/Turbo component set, 4B ClipProj, FP16 video VAE, FP32 audio VAE, 640x352, 124 frames, and 4 steps. The launch profile uses `--lowvram` for the very-low/8 GB tiers; 10–16 GB systems use normal VRAM mode unless preflight or a prior OOM justifies offload. Use Block Cache only when its classes are actually loaded; otherwise use the compatibility workflow. This is the success-rate baseline.
- **Balanced:** keep the low-VRAM canvas on an 8 GB laptop, use 6 steps, and bypass Block Cache. On a mid/high-VRAM machine, the planner may select 864x480.
- **Quality:** use 8 steps and bypass Block Cache. On an 8 GB laptop, keep 640x352 and warn that W4A8/4B remains a quality ceiling; on a mid/high-VRAM machine, the planner may select 864x480.

The accelerated fast graph is paired with the registered LightX2V/Turbo 4-step
LoRA. `minimax_h3_turbo_v4_step600_ema.safetensors` is a compatibility-workflow
quality variant: H3 Lite rejects it when Sage/Sol/Chunk/T8 acceleration nodes are
present, because that combination produced severe ghosting and color artifacts in
local validation. Use a `*_compat_api.json` workflow for v4, or use the registered
LightX2V/Turbo 4-step LoRA for the fast route.

- **6 GB experimental:** when the machine has roughly 6 GB VRAM, 32 GB system RAM, an SSD, and sufficient pagefile headroom, permit a cautious first run at 608x352, 4 steps, and low-VRAM offload. Treat community timings as orientation only: reported I2V runs include about 345 seconds at 608x352/4 seconds and 441 seconds at 864x480/5 seconds, while another 640x480/5-second report took about 13.7 minutes. These used different official/community model and workflow combinations, so do not transfer the numbers to the bundled W4A8 graph as a promise.
- **Below roughly 6 GB or insufficient RAM/disk:** stop before downloading or queueing. Explain the missing capacity and propose a hosted/API or alternative model.

Do not infer that a smaller file or INT8 label is automatically faster. On low-VRAM systems, CPU offload, RAM bandwidth, kernel compatibility, and first-run compilation often dominate.

The planner's `--target-minutes` option chooses the highest-quality mode whose
conservative upper estimate fits the budget. Without that option, `auto`
always selects fast. The estimate must include a cold-start warning and the
final report must include ComfyUI's actual execution time.

### 3. Install or repair the environment

Use the official ComfyUI H3 tutorial and the component list in
`references/deployment-matrix.md`. Create or reuse the selected ComfyUI root
and place every component below that root. Pin or record the ComfyUI and
custom-node commits used for a successful run. Install only nodes referenced by
the selected workflow. The baseline requires `ComfyUI-ClipProj` or a compatible
implementation. KJNodes, H3 Turbo helper nodes, Sol Attention, and T8 Block
Cache are optional unless the selected accelerated graph explicitly uses them.

`h3_fastpath.py --workflow-template auto` is the default. It queries
`/object_info` when available and uses the accelerated T2V or I2V graph only
when the Sage, Sol, Chunk Feed Forward, and T8 classes are actually loaded.
Pass `--component-set A` or `--component-set B` when both complete sets are
installed. Set B is validated with the compatibility graph on RTX 4060 Ti
16 GB and RTX 4070 Laptop 8 GB systems, so `auto` selects that graph. Its full
Sage/Sol/Chunk/T8 acceleration chain is not yet the validated default; use
`--acceleration fast` only for an intentional trial. Use `--acceleration
compat` to force the validated compatibility graph.
Both graphs preserve the H3 sampler, native audio, ClipProj, LoRA, dual VAEs,
and native first/last-frame inputs without optional patches.

The bundled multi-image Ref2VA graphs reuse this same registered component set:
there is no separate Ref2VA checkpoint to download. They add the native
`MiniMaxH3ReferenceToVideo` route and bind repeated reference images through the
resident ClipProj path. Before downloading anything, reuse the local W4A8,
4B encoder, ClipProj, dual VAE, and Turbo LoRA files when their manifest and
loader checks pass.

Keep optional INT8 loaders and experimental cache nodes disabled until the baseline works. Start ComfyUI with a profile-appropriate command; the very-low/8 GB launch profile commonly uses `--lowvram` and `--fast-disk`, while 10–16 GB systems normally omit `--lowvram` unless preflight or an earlier OOM justifies it. Add Sage Attention only after its PyTorch/CUDA compatibility is confirmed. Treat Easy Cache and generic cache nodes as opt-in experiments: community reports and local experience show that some settings can blur or damage motion/detail. Never enable a cache solely from a speed claim; compare a short output against the uncached baseline first.

After installation, rerun the doctor and stop if any required model or node is missing. Do not start a long generation while the graph contains unresolved node classes.

Cold-path validation is deliberately allowed to spend extra seconds or minutes:
record component URLs, selected replacement filenames, file sizes or hashes,
local loader choices, Python import probes, and any small compatibility patch
that was applied. This is cheaper than discovering a wrong 20 GB model or a
Torch/custom-node mismatch after queueing. Once the baseline succeeds, write or
refresh the cached environment/component report and return to the hot path for
future prompts.

When public internet is the only source for a large model such as the W4A8
diffusion checkpoint, do not start a blind long transfer. First probe the
official raw URL, configured mirror raw URL, and any user-provided proxy URL
with a small resumable/ranged request. Report the fastest measured speed,
estimated time, and selected command. Prefer `aria2c` with conservative
connections when available and the server supports ranges; otherwise use
`curl.exe -L -C -`. Keep partially downloaded files as `.part` and resume them;
do not delete progress or restart from byte zero unless the size/hash proves
the file is unrecoverable.

Before queueing, run:

```powershell
python scripts/h3_plan.py `
  --doctor-json <ComfyUI>\user\h3lite_runs\_environment\doctor.json `
  --install-mode reuse-existing `
  --comfyui <ComfyUI-path> `
  --mode auto `
  --aspect landscape `
  --video-seconds 5 `
  --report-file <ComfyUI>\user\h3lite_runs\_environment\plan.json `
  --json

python scripts/h3_preflight.py `
  --doctor-json <ComfyUI>\user\h3lite_runs\_environment\doctor.json `
  --plan-json <ComfyUI>\user\h3lite_runs\_environment\plan.json `
  --refresh-runtime `
  --require-audio `
  --json
```

`ready` means no observed risk, `caution` means the run may proceed with an
explicit warning, and `blocked` means fix the environment or use another
backend first. The preflight ignores the Python process used by ComfyUI and
zero-memory desktop helpers; it warns only for meaningful external GPU
competitors. This is a runtime gate, not a promise of success.

### 4. Rewrite the user's request into an H3 prompt

Identify the generation mode before writing:

- text only → `T2VA`
- one first-frame image → `I2VA`
- first and last images → `FL2VA`
- last-frame image → `L2VA`
- reusable images/video/audio references → `Ref2VA`

Read `references/prompt-writing.md` when composing or revising a prompt. For
multi-shot, identity-sensitive, or reference-heavy requests, also read
`references/agent-workflow.md` and build the route/anchor sheet before writing.
For an underspecified creative request, also read
`references/prompt-assist.md`: it supplies the optional
`STYLE/IDENTITY LOCK → SCENE → MOTION → AUDIO → NEGATIVE` scaffold and the
rules for translating it back to H3. Ask only for information that materially
changes the route or acceptance criteria; otherwise state the bounded default
and continue. Never make a website lookup a runtime dependency.
Use the exact field names and ordering required by the selected mode. For
native base modes, the core order is:

```text
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
```

For user-facing 5-second quick-start examples, teach the same idea as a memorable three-part structure: **scene and atmosphere → action and camera → sound**. Present it as one natural-language prompt that can be copied directly; do not require users to write schema labels. Treat this as an explanation aid, then translate it internally into the workflow's required prompt schema.

For full-reference mode, use the six-section structure in the reference. Write the rewritten description in English, but preserve user-provided dialogue, lyrics, and visible scene text in the original language.

Make the prompt operational:

- establish style, framing, subjects, environment, lighting, and initial state in Shot 1;
- make each important person's orientation relative to the camera explicit when identity or facial visibility matters; “watching the sunset” alone often implies a back view. State front-facing, three-quarter, profile, or back-facing, and say whether the face and eyes must remain visible;
- describe observable actions and state changes in playback order;
- use later shot cut times only when a real cut introduces new information;
- write camera motion as a natural sentence with motion type, amplitude, and speed when useful;
- describe dialogue, singing, and diegetic sound in the timeline body;
- describe ambient/physical sound in `overall_soundscape` and audience-only music in `non_diegetic_music`;
- Resolve the audio policy before submission: `auto`/ordinary prompts require the native H3 audio path; “no dialogue” does not disable sound. Only explicit complete-silence wording may remove the `CreateVideo` audio input.
- Interpret `不要对白` / `无对白` / `no dialogue` as a dialogue-only constraint: keep native ambience, sound effects, animal or action sounds, and the audio stream. Disable all audio only when the user explicitly requests `完全静音`, `无任何声音`, or `no audio`.
- use stable speaker IDs and exact text for dialogue;
- use `N/A` for music only when no non-diegetic music is desired, and use `overall_soundscape: N/A` only for explicitly complete silence.

If the selected ComfyUI graph uses a ClipProj/krea2 or another custom prompt schema, inspect its example workflow first. Adapt the official semantic structure to the node's accepted field while preserving the graph's required fields; do not blindly paste a T2VA prompt into a Ref2VA input or vice versa.
- For a face-quality attempt, constrain the first shot to one face, a front or three-quarter orientation, visible eyes, stable hair/clothing anchors, and static or small-amplitude motion. Treat dialogue as a second-stage stressor. A valid MP4 with dynamic/color/audio checks can still have unusable faces; visually inspect face consistency at first/middle/last frames.
- For Chinese prompts, avoid an extremely short noun-only description. H3's long multimodal sequence can make a one- or two-token Chinese prompt easy for the seed to dominate. Add concrete subject traits, setting, framing, lighting, and motion; as a practical starting point, use roughly 30–50 Chinese characters or an equivalent amount of structured detail, then validate with a cheap preview.
- For a director-level multi-shot sequence (establishing + over-the-shoulder + reverse + reveal) where H3's single-continuous-shot limit blocks real coverage, see `references/director-sequences.md`. It covers splitting the scene into separate I2VA segments, generating consistent first-frame images with ImageGen (watermark crop, reference-image identity inheritance), locking character identity across prompts, W4A8 skin-quality phrasing, and stitching with ffmpeg `xfade`/`acrossfade`.

### 5. Generate through ComfyUI

For installation, recovery, or custom-workflow debugging, the lower-level
helpers remain available. A normal text-only request should use
`h3_fastpath.py` above:

```powershell
python scripts/h3_generate.py `
  --base-url http://127.0.0.1:8188 `
  --workflow-template h3_w4a8_t2v `
  --prompt-file prompts/current.txt `
  --comfyui <ComfyUI-path> `
  --filename-prefix video/H3Lite_my_clip `
  --run-root <ComfyUI>\user\h3lite_runs `
  --component-set auto `
  --resolve-models `
  --audio-policy auto `
  --profile fast `
  --resolution 640x352 `
  --seed 20260813 `
  --watch `
  --watch-interval 20 `
  --watch-timeout 3600 `
  --json
```

For a lower-level I2VA or FL2VA run, use the matching bundled template and
frame flags. `--workflow <file>` remains supported for custom Ref2VA or other
graphs; the same flags will bind any existing `LoadImage` reference input, or
add one when the H3 node has no connection yet:

```powershell
python scripts/h3_generate.py `
  --workflow-template h3_w4a8_i2v `
  --prompt-text "<rewritten I2VA prompt>" `
  --first-frame <path-to-first-frame.png> `
  --comfyui <ComfyUI-path> `
  --output-dir <ComfyUI>\output `
  --resolution 640x352 `
  --length 124 `
  --steps 4 `
  --component-set auto `
  --resolve-models `
  --watch `
  --json
```

This lower-level command now submits and monitors in the same foreground
process. Use `--queue-only` only when another process must own monitoring.

For the bundled multi-image Ref2VA route, repeat `--ref-image` in the order
that the prompt names `<Picture 1>`, `<Picture 2>`, and so on. The native
`MiniMaxH3ReferenceToVideo` node accepts image references in fixed order; do
not combine `--ref-image` with `--first-frame` or `--last-frame`:

```powershell
python scripts/h3_generate.py `
  --workflow-template h3_w4a8_ref2va_compat `
  --prompt-file prompts/ref2va.txt `
  --ref-image <identity.png> `
  --ref-image <scene.png> `
  --ref-image <wardrobe-or-prop.png> `
  --comfyui <ComfyUI-path> `
  --output-dir <ComfyUI>\output `
  --resolution 640x352 `
  --length 124 `
  --steps 4 `
  --component-set A `
  --resolve-models `
  --watch `
  --json
```

The Ref2VA encoder is kept resident because the installed ClipProj node's
dynamic image path is not reliable with the low-VRAM int8 encoder. Resident
mode can consume more VRAM than I2VA; preflight may therefore block an 8 GB
machine or recommend a small preview. Treat `--ref-image` as a real reference
conditioning input, not as a style-only attachment.

The native `MiniMaxH3ReferenceToVideo` node also exposes reference-video and
reference-audio inputs. The bundled fastpath deliberately exposes repeated
`--ref-image` first because that path is the most predictable to bind and
verify; use a native/custom workflow when those other reference types are
required.

```powershell
python scripts/h3_status.py `
  --base-url http://127.0.0.1:8188 `
  --prompt-id <prompt-id> `
  --comfyui <ComfyUI-path> `
  --output-dir <ComfyUI>\output `
  --run-root <ComfyUI>\user\h3lite_runs `
  --require-audio `
  --dynamic-check `
  --anchor-check `
  --compact `
  --json
```

The JSON result is compact by default; use `--verbose` only when a failed run
needs the full ComfyUI history graph. For a bounded foreground monitor, add
`--watch --watch-interval 20 --watch-timeout 3600`. A pending response has
`ok: false` and `complete: false`; only a completed, verified media response
has `ok: true`.

Do not describe a faster second run as a reused result. ComfyUI may retain model weights and reuse unchanged upstream node outputs such as encoder or VAE loading, but a changed prompt or seed still causes the sampler to generate a new video. Explain this distinction when a user asks why later runs are faster.

The accelerated bundled template contains the prompt node, W4A8 model path, 4B
ClipProj, dual VAE, Turbo LoRA, H3 sigma shifts, Sage/Sol/T8 patches, native
audio, 124 frames, 640x352, and 4 steps. The compatibility template removes
the optional Sage/Sol/T8 chain. `--resolve-models` switches the complete
registered A/B component set atomically; use `--component-set B` when both
sets are installed. Override `--seed`, `--width`,
`--height`, `--length`, `--steps`, `--fps`, `--profile`, or `--resolution` only
when needed. Prefer `--megapixels` for ComfyUI ResolutionSelector-style canvas
choices such as `16:9 0.4MP -> 864x480`. `fast` keeps Block Cache; `balanced`
uses 6 steps and bypasses Block Cache; `quality` uses 8 steps and bypasses
Block Cache. For a custom API
workflow, pass `--workflow <file>`; the prompt field can still be discovered or
selected with `--prompt-node` and `--prompt-field`.

Do not use synchronous wait mode from a terminal command with a short timeout.
`h3_fastpath.py` uses `h3_generate.py --watch` so submission and verification
stay in one bounded foreground process; use `--queue-only` only when another
process must own monitoring.

For the low-VRAM baseline, use 124 frames at 24 fps (about 5 seconds), 4 steps, `res_multistep`, and `simple`. Use 640x352 for the normal fast output; use the ComfyUI official-style 0.4MP bucket (`864x480` for 16:9) only when the planner approves the hardware/time trade-off or the user explicitly asks for it.

### 6. Verify and report

Do not report success from a queue ID alone. Confirm:

- the job completed without an execution error;
- an MP4 or other intended video file exists;
- the file has a video stream and, for native H3 output, an audio stream;
- duration, frame count, and FPS are close to the requested values;
- the first/middle/last samples are not black, flat, or classified as `suspected_mosaic`; when the user asked for an action, dynamic QA must classify the clip as `dynamic`;
- when identity, clothing, props, or a reference composition is an acceptance condition, sampled frames preserve the declared anchors and do not silently drift;
- the output path and elapsed time are reported.
- the plan's estimated range is compared with the actual ComfyUI execution time;
- the configuration fingerprint and run manifest are recorded so a retry can be distinguished from an accidental duplicate;
- the selected ComfyUI, models, custom_nodes, and output paths are recorded.

If verification returns `ffprobe_not_found`, keep the generated file and repair
only the verifier path. Set `H3LITE_FFPROBE` to an absolute `ffprobe.exe` when
the executable lives outside the bounded ComfyUI locations; do not rerun the
model merely because metadata inspection was unavailable.

When diagnosing failures, use this order:

1. critically low pagefile, unavailable RAM, or another GPU process;
2. missing model or custom node;
3. wrong model folder or filename;
4. CUDA/PyTorch/custom-kernel incompatibility;
5. out-of-memory or excessive CPU offload;
6. invalid H3 flow/audio graph;
7. prompt or reference alignment problem.

For black/mosaic output, restore the official H3 flow/sigma-shift node and simplify optional attention/cache patches. For distorted audio at 4 steps, check the H3 audio/video schedules and use the compatible H3 Turbo sampler on older ComfyUI builds.

## Video upscale (post-processing)

Super-resolution is post-processing, not another H3 generation pass: it never
changes the generation graph, adds sampling steps, or silently follows a run.
The registered low-VRAM route has been validated around 0.5 MP (for example,
960x544), but that is not a model-wide H3 limit. Official/native H3 workflows
support larger canvases; use the doctor, planner, and preflight result as the
source of truth for the selected local canvas. Add upscale only when the user
explicitly asks for a target beyond that selected canvas, and never promise
pixel-exact consumer labels ("1080p") as model output sizes. Read
[`references/video-upscale.md`](references/video-upscale.md) for the routes,
portable path discovery, measured case studies, and the problem table.

Rules:

- **Primary route (when actually installed): Topaz Video AI.** The user selects
  the enhancement model (Starlight/Astra Fast for H3-style clips, Proteus/Rhea
  for general footage), the output scale, and exports from the GUI. Do not
  assume a Topaz install, model list, or path copied from another machine.
  Topaz has no stable public CLI contract; log command lines are diagnostic
  evidence only, never a scriptable runtime dependency.
- **Pre-export checks:** discover the actual Topaz/model roots, inspect free
  VRAM with `scripts/h3_vram.py --json`, and confirm any competitor is idle.
  Treat model presence and SHA-512 catalog checks as machine-local facts; do
  not state that a model is installed or missing without checking this machine.
- **During export:** do not start another heavy CUDA job (no H3 generation on
  the same card); a 2x Starlight export of a ~210-frame clip takes on the
  order of 20 minutes at 0.3 fps while decoding.
- **After export:** ffprobe must show the expected size, 24 fps, an audio
  stream, and near-original duration; the `videoai=Enhanced using ...`
  metadata tag marks an already-enhanced file — do not upscale it again.
- **Fallbacks (scriptable, offline):** use FlashVSR only after discovering
  <FlashVSR_ROOT>, its environment, the exact CLI fork/commit, and the model
  pack. Invoke the script by absolute path, for example
  <FlashVSR_ROOT>\ComfyUI-FlashVSR_Stable\cli_main.py; a bare
  `cli_main.py` depends on the current working directory and is not portable.
  Preserve the validated tiling/chunking settings from the local reference,
  verify frame count and audio after export, and treat truncated output as a
  failed run. If FlashVSR is absent, use the discovered ComfyUI venv plus a
  verified 4x-UltraSharp weight, or plain ffmpeg for previews. Never invent a
  missing wrapper, model, venv, or path.

## GPU memory contention with other heavy CUDA apps

A machine may host several heavy CUDA workloads at once (ComfyUI plus an
independent video tool, game streaming, another generation backend). Windows
WDDM reporting hides per-process VRAM: `nvidia-smi --query-compute-apps`
returns N/A or 0 MiB per process, while PyTorch's free-memory figure stays
optimistically high. Failures then look like CUDA OOM at tiny allocation sizes
("Tried to allocate 28.00 MiB ... 8.91 GiB free") or like a plain access
violation (Windows exit code 0xC0000005) with no OOM text at all.

Rules:

- When preflight flags a meaningful competitor or a fragile job fails at
  allocation, run `python scripts/h3_vram.py --json` to see who holds the
  VRAM. Read the reported process scope and WDDM LUIDs; do not treat a
  process row from an unmapped adapter as proof that it competes for the
  selected CUDA GPU.
- Confirm the competitor is idle before touching it. For ComfyUI, check
  `http://127.0.0.1:8188/queue` (running + pending empty), and remember an
  empty queue does not release resident models: the process still shows
  several GB until ComfyUI restarts or offloads.
- Never stop a ComfyUI that has queued or running items; losing a run wastes
  the whole generation.
- Gate only after the report shows a comparable adapter scope. The
  `--check-free-gb` command fails closed with exit code 2 when the
  scope is unknown; choose the threshold from the selected workload and
  hardware rather than assuming 5 GB. Use `--allow-scope-mismatch`
  only when the user explicitly accepts a global `nvidia-smi`-only
  check.
- A destructive stop requires `--confirm-stop` and the exact
  `--process-name` from the report. For ComfyUI also pass
  `--queue-url`; the helper refuses to stop when the queue cannot be
  read or is non-empty.
- Read `references/gpu-contention.md` for the measured case (ComfyUI holding
  9,805 MB resident on a 16 GB card while a Topaz Video AI Starlight export
  failed at a 28 MiB VAE allocation) and the diagnostic commands.

## Bundled resources

- `references/deployment-matrix.md`: tested low-VRAM profile, official fallback profile, model folders, node roles, launch flags, and failure triage.
- `references/component-sets.md`: registered model/workflow sets, exact known byte sizes, runtime ABI record, and download-integrity rules.
- `references/prompt-writing.md`: concise operational digest of MiniMax's official H3 prompt guide and prompt-writing skill.
- `references/director-sequences.md`: end-to-end workflow for multi-segment director sequences (divide a scene into separate I2VA shots, generate consistent first frames with ImageGen including watermark crop and reference-image identity inheritance, lock character identity across prompts, W4A8 skin-quality phrasing, and stitch with ffmpeg `xfade`/`acrossfade` using the cumulative-offset formula).
- `references/agent-workflow.md`: intent routing, reference/identity anchors, five-pass prompt enhancement, and local capability boundaries.
- `references/prompt-assist.md`: optional Higgsfield-inspired prompt-pattern lookup for vague creative briefs; maps style/scene/motion/audio/negative cards back to H3 fields without adding a cloud dependency.
- `references/face-quality.md`: face-first routing, official Ref2VA identity controls, low-VRAM trade-offs, and the limits of dynamic/color QA.
- `references/gpu-contention.md`: Windows/WDDM VRAM contention between ComfyUI and other heavy CUDA apps; per-process memory via the GPU Process Memory counter, idle-queue checks, the stop/restart order, and a measured 16 GB case plus a Topaz Video AI repair appendix.
- `references/video-upscale.md`: post-process video upscale routes (Topaz Video AI primary, a discovered/pinned FlashVSR CLI root as the scripted backup — run command, model pack, VRAM tier table, the measured tiled-seam grid case and fix, frame-count truncation check), 4x-UltraSharp via the discovered ComfyUI venv, plain ffmpeg fallback, and the problem/consequence table for contention, corrupt weights, audio, and frame-rate pitfalls.
- `assets/h3_w4a8_t2v_api.json`: reusable low-VRAM T2VA API graph based on the validated W4A8/4B/audio route.
- `assets/h3_w4a8_t2v_compat_api.json`: core T2VA graph without optional Sol Attention, Chunk Feed Forward, or T8 Block Cache nodes.
- `assets/h3_w4a8_i2v_api.json`: reusable low-VRAM I2VA graph with native first-frame input and optional acceleration patches.
- `assets/h3_w4a8_i2v_compat_api.json`: core I2VA graph with native first-frame input and no optional acceleration patches.
- `assets/h3_w4a8_ref2va_api.json`: experimental multi-image Ref2VA graph with the native `MiniMaxH3ReferenceToVideo` node.
- `assets/h3_w4a8_ref2va_compat_api.json`: compatibility Ref2VA graph without optional Sage/Sol/Chunk/T8 patches.
- `scripts/h3_doctor.py`: dependency-free hardware, disk, model, and node diagnosis.
- `scripts/h3_plan.py`: read-only hardware, resolution, time-budget, aspect-ratio, profile, and installation-path planner.
- `scripts/h3_preflight.py`: read-only pagefile/RAM/VRAM/process/model/node gate before queueing.
- `scripts/h3_generate.py`: fast template-based or custom-workflow submission with profile, resolution, prompt/settings overrides, native first/last-frame binding, atomic A/B component-set selection, and queue-only mode.
- `scripts/h3_anchor.py`: deterministic prompt/reference anchor-card generation and optional JSON declaration loading; it does not add inference or rewrite the prompt.
- `scripts/h3_status.py`: compact one-shot or bounded watch status, actual execution-time, media metadata verification, optional first/middle/last dynamic QA, advisory anchor continuity QA, and empirical timing-cache updates.
- `scripts/h3_fastpath.py`: single-entry T2V/I2V route that reuses the environment cache, probes loaded node classes, routes component sets, and keeps queueing plus bounded verification in one command.
- `scripts/h3_monitor_gui.py`: native Windows Tkinter progress window for live queue/sampling/resource/output visibility; it does not replace ComfyUI or interrupt a run.
- `scripts/h3_paths.py`: Windows path normalization for `F:/...` and Git Bash `/f/...` inputs.
- `scripts/h3_cleanup.py`: dry-run-first maintenance for old timestamped run snapshots; preserves environment state and recent runs.
- `scripts/h3_vram.py`: read-only Windows per-process dedicated-VRAM report (preserving WDDM adapter LUIDs, with `--query-compute-apps` fallback), scope-aware free-VRAM gate, and guarded destructive stop requiring an exact process name, explicit confirmation, and optional queue check.

## Sources

- Official H3 prompt guide: <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md>
- Official H3 prompt-writing skill: <https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing>
- Official ComfyUI H3 tutorial: <https://docs.comfy.org/tutorials/video/minimax/minimax-h3>
- H3 Turbo ComfyUI nodes: <https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo>
- Cinema DNA 21:9 × 3: <https://github.com/dacnay816y62-hub/cinema-dna-21x9x3>
- Official H3 repository and prompting guidance: <https://github.com/MiniMax-AI/MiniMax-H3>
- Public agent-workflow design reference (not a runtime dependency): <https://github.com/higgsfield-ai/skills>
- Public prompt-pattern references (optional writing aid, not a runtime dependency): <https://github.com/higgsfield-ai/skills/blob/main/higgsfield-video-explainer/references/prompts.md>, <https://higgsfield.ai/ai-prompt-generator>
- Community Mac/Metal MLX port and operational notes (reference only; not a tested h3lite backend): <https://zhuanlan.zhihu.com/p/2069479566171812707>
- Community Apple Silicon local-deployment troubleshooting (reference only; not a Windows resource source): <https://mp.weixin.qq.com/s/hN60KLN7Pkpqb0pbk-r4WQ>


---

## FILE: dsh-plugin/README.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/README.md)

# H3Lite for DeepSeek Harness

H3Lite 是一个通用的 AI Agent 插件：把本机 MiniMax H3 / ComfyUI 视频生产流程打包成六个可移植 skill，并在 DeepSeek Harness（DSH）里额外注册四个原生工具。

- **六个 skill**：`h3-workstation`、`h3-prompt`、`h3-director`、`h3-generate`、`h3-post`、`h3-whitemodel`。
- **四个 DSH 原生工具**：`h3_generate`、`h3_status`、`h3_doctor`、`h3_list_runs`。
- **同一份脚本**也服务于 Qoder、Claude Code、Codex、ZCode 和标准 MCP 客户端；插件不调用云端 API。

版本与仓库主插件对齐：**0.2.0**。

## 前置条件

- Windows + NVIDIA CUDA 主路线；其他平台可复用提示词/skill，但 ComfyUI 与 H3 推理未验证。
- Python 3.11+，能执行 `skills/h3-generate/scripts/*.py`；脚本本身以标准库为主，视频 QA 需要 ffmpeg/ffprobe。
- 已安装 ComfyUI 与 H3 组件集（Set A/B/C）；Set C 必须先装 Set B。
- 至少一次机器级路径配置（二选一）：

```powershell
# 方式一：写入 ~/.h3lite/config.json
python skills/h3-workstation/scripts/h3_doctor.py --comfyui "E:/MiniMax-H3/ComfyUI" --write-config

# 方式二：环境变量
$env:H3LITE_COMFYUI = "E:/MiniMax-H3/ComfyUI"
```

解析顺序：`--comfyui` → `H3LITE_COMFYUI` → `~/.h3lite/config.json` → `<project>/ComfyUI` → `~/ComfyUI`。新会话不依赖上一个任务的目录。

## 安装到 DSH

### 本地目录（开发/未发布）

DSH 的 **Plugins → Add plugin** 支持本地绝对路径：

```text
<本仓库>/dsh-plugin
```

安装后 bundle 显示为 **H3Lite**，新会话技能目录出现六个 skill，工具列表出现四个 `h3_*` 工具。

### npm / tarball

```bash
cd dsh-plugin
npm pack --dry-run   # 确认 skills/**、lib/**、runtime/**、cordis.patch.yml 在清单里
npm publish --access public
```

发布后用户在 **Add plugin** 填 `h3lite` 或 `h3lite@0.2.0`。也可把 `npm pack` 生成的 `.tgz` 上传到 GitHub Release，直接填 tarball URL。

## 工具

| 工具 | 作用 |
|---|---|
| `h3_generate` | 一次完成规划、预检、提交和验证；支持 T2VA/I2VA/FL2VA/L2VA/Ref2VA、参考图/音频/视频、Set C 模板和视频 VAE 覆盖 |
| `h3_status` | 按 `prompt_id` 查状态、输出路径、时长/帧率/音频和动态 QA |
| `h3_doctor` | 只读体检：GPU、RAM、页面文件、磁盘、模型/节点、ComfyUI 路径，以及 Blackwell 量化风险 |
| `h3_list_runs` | 扫描最近 run manifest，恢复全局流水线状态 |

工具参数与对应 Python CLI 保持一致。批量多镜、白模、后期、show contract 和网格 QA 仍由 skill 指导调用脚本：

```powershell
python skills/h3-generate/scripts/h3_showrunner.py --spec projects/ep04/show.json --strict --check-sha --output-dir <ComfyUI>/output
python skills/h3-post/scripts/h3_grid_check.py --video <output.mp4> --json
```

## Skill 与脚本布局

`index.js` 从包内 `skills/` 读取六个 `SKILL.md`，并把每个目录作为 `resourceBase` 注册。npm tarball 不包含仓库根目录，因此 `prepack` 会运行 `scripts/sync-skills.mjs`，把规范 skill 复制到包内；`npm run check-skills` 用于防漂移。

脚本仍然使用 `H3LITE_*` 环境变量和 `~/.h3lite/config.json`，不依赖 DSH 专有路径。这样同一套 skill 可以被其他 agent 直接复用。

## 可选：标准 MCP

包内附带 `runtime/h3-mcp-server.cjs`。其他 MCP 客户端可以直接启动：

```json
{
  "command": "node",
  "args": ["<安装路径>/runtime/h3-mcp-server.cjs"],
  "env": {
    "H3LITE_COMFYUI": "E:/MiniMax-H3/ComfyUI"
  }
}
```

DSH 不需要这个 MCP：四个工具已经以原生 `ctx.tools` 注册。MCP 服务器保留给 Qoder/Claude/Codex 等标准 MCP 宿主。

## 配置

| 变量/文件 | 用途 |
|---|---|
| `H3LITE_COMFYUI` | ComfyUI 根目录（含 `main.py` 与 `models/`） |
| `H3LITE_CONFIG` | 配置文件位置，默认 `~/.h3lite/config.json` |
| `H3LITE_PYTHON` | 运行 H3Lite 脚本的 Python；默认 ComfyUI venv 或系统 Python |
| `H3LITE_OUTPUT` | ComfyUI output 目录 |
| `H3LITE_RUN_ROOT` | run manifest 根目录 |
| `H3LITE_BASE_URL` | ComfyUI HTTP 地址，默认 `http://127.0.0.1:8188` |
| `H3LITE_SCRIPTS` | 覆盖脚本目录（一般不需要） |

## 已知边界

- 16 GB 卡不要并行两条 H3；481 帧单段禁止提交。
- Set C 的 Sol INT8 VAE 可能有 128px 接缝；最终交付前跑 `h3_grid_check.py`。
- RTX 50 系 + W4A8/INT8 convrot 有上游内核风险；先跑 640×352/4 步双提示词测试，受影响时只换 DiT 到 `fp8_scaled`。
- 插件不安装 ComfyUI、模型或 Python 包；只提供执行层和知识层。


---

## FILE: dsh-plugin/skills/h3-director/SKILL.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/SKILL.md)

---
name: h3-director
description: 设计多镜短剧、动作/战斗场面与分镜；规划镜头时长、景别、场面覆盖、首尾帧链、角色连续性、H3 原生台词与声音圣经（选角声学距离、每镜声音验收）。当用户要做短剧、多镜头故事、打斗、分镜表、场面调度、连续角色视频，或要求人物声音/音色跨镜一致时使用。
---

# 分镜导演：从故事到可生成的镜头表

输出物：分镜表 + 构图关键帧计划 + 每镜的提示词任务与生成参数。**本 skill 只设计，不提交生成**；渲染交给 `h3-generate`，剪辑交给 `h3-post`。动作戏先读 [`references/action-combat.md`](references/action-combat.md)，不要套文戏的默认节奏。

## 单次生成可以多镜

H3 的官方提示词支持在一条时间线内写切点；本机 Set C FL2VA 已实测：一条约 10 秒的提示词写入 4 个 `[Shot]`，输出在 2.42 s、5.04 s 出现硬切，末尾落到尾帧。不要沿用“一次生成只能是一个连续镜头”的旧限制。

- 约 10 秒优先用 **FL2VA 一条生成 3–4 拍**：身份、服装、色板、灯光、雨/雾和底噪天然连续，比逐镜生成再拼更不容易割裂。这条默认针对**文戏/叙事**。
- **密集打斗不套这条默认**：动作戏先按 `action-combat.md` 数动作事件。5 秒 5–7 个事件是默认单元；10 秒只有 9–13 个事件才值得单条，否则拆 2 × 5 秒链式。3 秒只够一次冲击/插入，不够一套交手。
- 切点是导演目标，不是合同。模型可能把相邻两拍合并成一次转身、推近或摇镜；只要仍有 3–4 个清楚的视觉段，切点落在动作/视线转移上，就接受，不因少一个硬切重跑。
- 需要更明确的硬切时，每个切点单独写 `This is a hard cut.`，切后重述景别、机位、朝向和道具状态；验收按实际视觉段数，不按提示词里的 Shot 数。
- 超过 4 拍、总长超过约 15 秒、动作密度不足、单镜台词太长，或同一条提示词连续两次都形成不了新构图，才拆成多条生成。拆后按 `h3-post` 的连续性与整条 loudnorm 规则处理。

## 本机节奏约定

这些是本项目验证过的默认值，不是模型限制；用户明确要求时以当次为准。

- **文戏快节奏短剧约 3 秒一镜**：`--length 72` 实际输出 73 帧 ≈ 3.042 s。不要为凑 3.000 s 硬裁；保留自然帧数，切点落在动作或自然边界上。动作戏不要按 3 秒一拳拆，见下一条。
- **常规叙事 5 秒一镜**：`--length 124` 实际 ≈ 5.167 s。
- **动作戏默认 5 秒一个生成单元**：`--length 124`，5–7 个主要动作事件、1–3 个视觉段。3 秒只用于单次冲击/插入；10 秒需要 9–13 个事件；15 秒需要 13–18 个事件，通常不如拆 3 × 5 秒。事件不够时 H3 会用慢放、停顿和空时间填满。
- **单段上限 362 帧**（约 15 s @ 24 fps）。超过就链式拆段；**不要提交 481 帧**——本机曾出现 48 W 自旋死锁，必须重启服务。362 帧竖屏单段实测约 12 分钟。
- 3 秒文戏的预算是：一个动作落地 + 一个运镜 + 一句短语（约 6–10 词）。5 秒文戏可以加 1–2 个次级动作和 12–16 词台词。动作戏按事件数算，不按台词长度算。
- 短剧默认 9:16 竖屏（768×1344 或 1024×1344 一类合法画布）；快速动作与全身构图容易比例失真，优先**腰上/3/4 身中景**，保持 7.5 头身；必须全身时减小跳跃幅度，或改用 1344×768 横屏。
- 先写分镜表，再写动作因果链，再画**构图**关键帧（动作戏只锁首/尾，不画动作相位序列），最后才写提示词。动作戏用多张关键帧去插值连续打斗已被本机 E1 证伪：Add Guide 会钉帧并产生硬切/定格。

## 分镜表字段

| # | 秒 | 景别 | 摄影机 | 动作（含完成态） | 台词/声音 | 首帧 | 尾帧 | 路线 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.0 | WIDE | 缓推 | 她停步，手完全攥紧硬币 | 雨声；无台词 | K1 | K2 | FL2VA |

动作戏在“动作”列按 `A1 / B1 / A2...` 标事件并数总数，不要只写“两人打斗”；5 秒单元少于 5 个主要事件就重编。

补充规则：

- 每镜只写**一个主运镜**；复合运动分时段。
- 备注写镜头目的、入/出画方向、视线方向；不要只写景别。
- 文戏每镜动作写到完成态（“刀已完全入鞘并停住”）；动作戏写到命中结果、位移或新劣势，不要回到起手式。
- 相邻镜头景别交替（WIDE → CU → MEDIUM），不要连续同景别同机位。
- 切点优先落在动作中段或视线转移上；整片至少有一次景别/机位变化来标记节奏。
- 10 秒可用 3–4 拍一条生成（hook → build/turn → payoff），这是文戏默认；动作戏先数 9–13 个主要动作事件，不够就拆 2 × 5 秒。15 秒以上再考虑 4–5 拍或链式分段。

## 专业覆盖与调度

先写一句场景：谁想要什么、什么挡住他、他做了什么、局面变成什么。答不出来，加机位也救不了。每个 beat 只回答一个问题，带来一个新信息。

- 覆盖按目的分：建立空间、施压/权力、反应/决定、关键信息、转场。文戏 10 秒 3–4 拍最多选 3–4 个目的，不做“同一瞬间的五个角度”；动作戏按攻防信息切，不按情绪目的凑拍。
- 文戏的经典 master shot + coverage 在 H3 里压成一条 3–4 拍；超过 4 拍、超过约 15 秒或需要两套地点/参考时才拆条。动作戏按 5 秒攻防单元和因果链接，不按文戏 coverage 凑拍。
- 轴线：两人/人与物之间保持同一侧；A 看画右，B 就看画左；车辆和行走保持屏幕方向。过线用中性镜头或可见运镜，不在两个静态机位之间跳边。
- 30°：切后角度至少变约 30°，或景别有实质变化；同景别、同侧、同机位就是跳剪。
- 视线/动作：eyeline 要对上；动作切在中段，下一镜接同一相位；记录出画侧、持物手、门/灯状态。
- Blocking 写位置、距离、朝向和遮挡：前/中/后景分层；用高度、距离、门框制造权力关系；一个 beat 只变一次走位。
- 预算：文戏每拍 1 个强动作、1 个主运镜、1 句台词；一条不超过 3 个重要角色、1 个地点、1 个特殊物理。动作戏把“一个强动作”理解为一条因果链里的 1–2 招，5 秒整条 5–7 个主要动作事件；不要一拳一镜，也不要一场戏塞十个事件。超了就拆或简化。
- 节奏：文戏 3–5 秒一切；动作戏默认 5 秒一个生成单元，段内 1–3 个视觉段。允许 6–8 秒 hero hold，但打斗里不要用对峙和定格充当 hero hold；不要连续三个同景别/同机位/同运镜。
- 动作戏不是禁区：真正有效的是具体招式、受力方向、命中反馈和连续因果链，而不是 `two people fighting`。精确手指细活、兵器缠绕、多人同时接触仍要拆/简化；完整方法见 [`references/action-combat.md`](references/action-combat.md)。
- 一个镜头只完成一次转化：文戏从站到坐、从怀疑到决定；动作戏从交手前状态到一个新的攻防位置。每次生成当交付，不设计必须靠后期修补的镜头。

完整判断、覆盖表和连续性清单见 `references/professional-coverage.md`。

## 连续性与锚点

多镜或身份敏感时，先写内部锚点表（不展示给用户也可）：

```text
Subject A: 黑发、深灰风衣、右眉疤
VOICE A (S1): 日语、F0 约 100 Hz、干涩沙哑、语速慢、句尾下沉、近麦
Picture 1: 主两针关键帧
Picture 2: 反向关键帧，同海岸、同服装、同天气
Audio 1: voice_A_dry.wav（单人干声 5–8 s，只作音色参考，不复制其台词）
Retention: 脸、发型、服装、体型比例、地点色板、音色与语速
Allowed change: 视线、手部小幅动作、机位、海水运动
Forbidden drift: 换装、年龄变化、多出人物、天气改变、音色变年轻/变亮
```

- 人物外貌描述在每镜**逐字一致**（发色、瞳色、服装、体型），不要换同义词、不要加新配饰。**VOICE 块同样逐字一致**，且与外貌块分开。
- 同场两个同性别说话人的 F0 目标带差 ≥25–30 Hz（约 4 个半音）且质地不同；一条里 ≤2 个说话人。底床声指定唯一正典来源。
- 每镜写 `Visible actors in this shot: ...` 与 `Do not swap, duplicate, or add cast members.`，防止背景多出一个人。
- 一镜的角色用稳定名字（`Man A`、`Subject A`），不要起别名后中途改口。
- 首尾帧链：上一镜真实尾帧可以直接作为下一镜首帧；若动作要跨镜连续，**切在动作中段**，不要切在收势。
- 遮挡镜头（过肩、背影）的第一帧必须仍能看到被遮者的肤色、发型、服装线索；纯文字锁身份不够，关键帧要用参考图继承。人物被肩膀完全挡住时，H3 会发明一个新身份。
- 每镜在提示词里写“哪些必须保留、哪些允许变化、哪些禁止漂移”，与锚点表一致。

## 台词纪律

- **台词必须进 H3 原生整体生成**，不做后期 TTS 拼接（本项目硬规则）。后期只做响度、字幕、混音。
- **先定声音圣经再写分镜**：每个角色一行（语言 / F0 目标带 / 质地 / 语速 / 咬字 / 情绪基线 / 正典样本），全片逐字复用。底床声（念经、雨、嗡鸣）也要指定唯一正典来源，不要每镜各自生成。完整方法见 [`../h3-prompt/references/voice-consistency.md`](../h3-prompt/references/voice-consistency.md)。
- **选角就要拉开声学距离**：同场两个同性别说话人 F0 目标带差 ≥25–30 Hz（约 4 个半音）且质地不同；一条里 ≤2 个说话人最稳，3 个以上同性别说话人默认高风险。模型靠音高/共振峰区分说话人，不靠提示词绑定；距离不够时改设定或拆条，不要指望形容词。
- 一次生成 ≤362 帧；长台词拆成同说话人的链式段，每段沿用同一角色参考和同一声音描述。不要在同一条里塞多句长台词——单镜多句会压缩、漂移、提前说完。
- 台词预算约 0.36 s/词（经验值），一条 ≤6 个对白轮次；超了就拆条。
- 每镜一句为主；说话者写 `says the line once and nothing else: "<原语言台词>"`。
- 画面里有第二张脸，就给它正向静止事实：`lips at rest, jaw closed, breath lifting the chest`。否则容易出现含糊 fill-babble 或台词重复。
- 轻笑、叹气、"嗯"这类 incidental 发声要么写成正式台词，要么删掉；裸写会被当成台词并打乱说话人路由。
- 台词短时不要留空白窗口：扩写到约 8 词以上，或写明环境音/动作事件占满剩余时间；也可以把镜头剪短到台词长度。
- 语言保持用户原文（日语/中文），周围描述用英文；不翻译台词。
- 生成后必须**听**并核对实际说出的词，音频流存在不等于台词正确；跨镜音色另按 `h3-post` 的声音一致性审计量化验收。

## 路线选择

| 目标 | 路线 |
|---|---|
| 单镜、有明确首帧、要精确开场构图 | `I2VA`（`h3_c_i2va768`） |
| 约 10 秒 3–4 拍、首尾构图都要锁（文戏默认） | `FL2VA` 单条多镜（`h3_c_fl2va768`） |
| 5 秒动作单元、首帧动作已上膛/尾帧动作中段 | `I2VA`（`h3_c_i2va768`）或 `FL2VA` 首尾两帧 |
| 人像/角色成片、身份一致性优先 | `Ref2VA`（`h3_c_ref2va768`，`ref_image_size=match`） |
| 跨镜音色一致优先、且有干净干声样本 | `Ref2VA` + `--ref-audio`（音色参考；I2VA/FL2VA 无音频输入） |
| 纯环境/物体/风格测试 | `T2VA` |

人像默认 `Ref2VA`：脸更大更清晰、身份更稳；需要严格运镜或末帧锚定时改 `FL2VA`。两者耗时接近（1344×768、124 帧约 152 s vs 161 s）。8 GB 机器不要把人像低分辨率 T2VA 当最终质量路线。动作戏不要为了锁动作相位而生成多张 Add Guide 关键帧；路线与兼容性见 `action-combat.md`。

## 关键帧

- 每个需要首帧的镜头单独生成一张，文件名互不相同；并行调用时提示词开头也要不同，避免同名覆盖。
- 图像工具会在底部加“AI 生成”水印条，喂给 H3 前裁掉：`ffmpeg -i in.png -vf "crop=iw:ih-80:0:0" -pix_fmt rgb24 out.png`。
- 遮挡/反打镜头用清晰的双人帧作参考图继承身份（`input_fidelity` 约 0.7–0.9），不要只靠文字。
- 关键帧是锚点不是第二段故事：服装、光位、天气、背景先与主参考一致，再谈动作。
- 生成顺序：先出主关键帧并确认角色，再派生尾帧与反打帧；不要批量画完再发现角色不对。
- **动作戏只画构图关键帧**：首帧（动作已上膛）或首 + 尾两帧。不要画 3–6 张动作相位图塞进 Add Guide；本机 E1 已证明它会钉帧并产生硬切/定格，不是连续动作插值。需要多关键帧时，明确把它当动态分镜/允许硬切的板子。
- 连续打斗用真实尾帧链式时，切在动作中段；尾帧选命中后位移或追击起点，不选收势。身份块在下一段逐字复制。

## 交接

本 skill 只做分镜与关键帧，不直接跑渲染。`references/` 里出现的 `scripts/h3_fastpath.py`、`h3_generate.py` 等命令均属于 `h3-generate` skill 的 `scripts/`。

- 分镜定稿后落成 `show.json` 的 `shots[]`：每镜 ID、prompt、mode、template、length、seed、参考图/音频、首尾帧、白模版本和 `acceptance`；先过 `h3-generate/scripts/h3_show_validate.py`，再写提示词/提交。合同见 [`../h3-generate/references/show-contract.md`](../h3-generate/references/show-contract.md)。
- 每镜交给 `h3-prompt`：模式、字段、身份块、**VOICE 块**、动作完成态、台词与声音；动作戏附因果链和事件数。
- 生成参数交给 `h3-generate`：模板、分辨率、帧数、首/尾帧/参考图路径、seed；动作戏先走 5 秒提示词基线，再按阶梯加辅助 LoRA。
- 剪辑点交给 `h3-post`：单次多镜只核对请求切点与视觉段，不做逐镜 loudnorm；逐镜拼接才走拼接流程。
- 任何一镜的验收条件（身份、动作落地、台词可听清、动作事件数、是否近静止）写进分镜表备注，不要只留在对话里。

## 资源

- `references/action-combat.md`：动作密度、因果链、实时速度、首尾帧、慢放诊断、动作 LoRA 兼容性与测试阶梯。
- `references/professional-coverage.md`：场景目标/障碍/转折、覆盖目的、180°/30°、blocking、复杂度与节奏预算、出入画连续性。
- `references/shot-timing.md`：3/5/15 秒节奏、帧数与切点、构图与比例。
- `references/native-dialogue.md`：H3 原生台词、链式分段、嘴部静止、验证。
- [`../h3-prompt/references/voice-consistency.md`](../h3-prompt/references/voice-consistency.md)：声音圣经、选角声学距离、参考音频绑定与已知坑、跨镜音色验收门限。
- `references/director-sequences.md`：单条多镜与多段链式的取舍、关键帧继承、拼接连续性。
- `references/agent-workflow.md`：意图路由、锚点卡、五遍增强。
- `references/face-quality.md`：人像路线 A/B 与护栏。
- [`../h3-generate/references/show-contract.md`](../h3-generate/references/show-contract.md)：开拍前合同与 `show.json` 镜头卡字段。

## 来源

- 官方 H3 仓库：https://github.com/MiniMax-AI/MiniMax-H3
- StudioBinder, 180-degree rule: https://www.studiobinder.com/blog/what-is-the-180-degree-rule/
- StudioBinder, film blocking: https://www.studiobinder.com/filmmaking-techniques-film-blocking/
- OSideMedia, higgsfield-ai-prompt-skill: https://github.com/OSideMedia/higgsfield-ai-prompt-skill
- 公开多镜与连续性写法参考（非运行时依赖）：https://github.com/higgsfield-ai/skills
- Wushu Action v7：https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora
- Combat Base V2：https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2
- General Motion Continuity Repair：https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair
- PDD Acc LoRA：https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs


---

## FILE: dsh-plugin/skills/h3-director/references/action-combat.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/action-combat.md)

# 动作戏与战斗：H3 的可用路线

动作戏不是“文戏少写一点”的变体，而是另一套预算：事件密度、因果链、实时速度、接触简化和采样路线决定成败。本文把公开成熟工作流、本机实测和兼容性边界压成一套可执行规则。数字是经验预算，不是模型合同；用户明确要求时以当次为准。

## 证据与结论

2026-11 解析的公开 H3 动作工作流有一个共同点：**不用密集时间轴锚点**。

| 工作流 | 参考图 | 关键点 | AddGuide |
|---|---|---|---|
| Wushu Action v7 多参考双采 | 8 × `LoadImage` + 3 × 视频参考 + 3 × 音频参考 | 1344×768、124 帧、双采样、3D latent upscale | **0** |
| Combat Base V2 REF2VA | 3 × `LoadImage` | 576×1024、124 帧、两段采样、`SplitSigmasDenoise 0.25` | **0** |
| Combat Base V2 FL2VA | 首/尾 2 × `LoadImage` | 124 帧、两段采样 | **0** |
| RunComfy Action Scenes 教程 | 2 个角色 + 1 个场景参考 | 首段 Turbo，后接 3D latent upscale + 动作精修 | 不用关键帧 |

本机 E1 进一步证明了为什么不能用密集关键帧“插动作”：

- 3 个 `MiniMaxH3AddGuide` 锚点放在 0 / 0.79 / 1.58 s；输出锚点 SSIM 为 0.798 / 0.731 / 0.681，跨锚点只有 0.25–0.29。
- 输出在 0.71 s 和 1.58 s 出现硬切级跳变，1.0–1.5 s 基本定住；73 帧里 50% 的帧 MAD < 1.0。
- 结论：AddGuide 把帧“钉”准，不负责姿态之间的连续插值。相邻锚点差异大时，H3 用跳变命中下一张图。它适合动态分镜/允许硬切的板子，不适合一条连续打斗。

本机 v2 的反例同样明确：260 帧 / 10.833 s，MAD > 8 只占 21.2%，近静止（MAD < 1.0）占 25.9%，所以“每一拍单看都成立，整条仍像慢放”。问题主要不在模型不会打，而在编排密度、停顿和路线。

## 1. 先编动作，再写镜头

动作戏的分镜先回答“谁先打、打哪里、对方怎么反应、为什么下一招接得上”，再谈机位。

一条可用的因果链：

```text
attack → trajectory → hit / block / miss → reaction → displacement → next attack
```

- 开场已经在动：第一帧是上步、蓄力或交手前一瞬，不是两人站定对看。
- 每次受击都要改变下一步：A 的直拳被 B 拍开，A 的重心前移，B 才接肘；不要 A 打完复位、B 再从头起手。
- 结尾落在命中后的位移、追击或新劣势，不停在收势敬礼。武术套路/表演可以定式，叙事打斗不要。
- 默认 2 个主要角色；第 3 人只在画外、背景或被单独镜头承担，不要三人同时抢接触点。1vMany 用交替切入、通道压迫和主角中心跟随，不要全员同时挥拳。
- 招式写具体术语和向量：`right straight to the jaw`、`left forearm parry outward`、`right hook to the ribs`、`head snaps back, rear foot skids half a step`。不要只写 `two people fighting`。
- 接触越复杂越容易糊：高速连击要简化、缩短、拉开距离；手指锁、兵器缠绕、多人同时抓扯默认拆成插入镜头或画外结果。

## 2. 动作密度与段长

“主要动作事件”指：一次攻击、一次格挡/闪避、一次命中反馈、一次明显位移、一次武器/道具状态变化、一次环境碰撞。不要把手腕翻转和衣摆算成事件。

| 单条 | 帧数 | 主要动作事件 | 视觉段 | 用途 |
|---|---|---|---|---|
| 3 s | 72（实测 73） | 2–3 | 1 | 单次冲击、插入、反应；不是完整交手 |
| **5 s** | **124** | **5–7** | **1–3** | **动作戏默认单元** |
| 10 s | 243–260 | 9–13 | 3–5 | 只有密度够才单条；否则拆 2 × 5 s |
| 15 s | 362 | 13–18 | 4–6 | 少用；优先 3 × 5 s 链式 |

- 5 秒是公开动作工作流最常见的单条长度。Wushu 的模板以 124 帧（约 5.2 秒）为例，双人 T3 在一条里写 3 个镜头、每镜 1–2 招，可作为密度参照。
- 10 秒 3–4 拍是**文戏/叙事**的已验证默认，不是密集打斗的默认。10 秒只有 3 个大动作时，H3 会用停顿、慢速和空时间填满。
- 用户坚持 3 秒一切时，把每个 3 秒当“一次冲击 + 一个反应”，并把切点放在动作中段；不要指望 3 秒完成起手、命中、受击、反击、落地五件事。
- 事件数不够时，先加招或缩短段长，不要加关键帧、不要加运镜、不要靠后期变速。

## 3. 镜头与构图

- 每个视觉段只选一个主导运镜：侧向跟拍、短推、固定机位三选一。不要手持 + 推近 + 环绕同时发生。
- 交换用侧面或 3/4 侧面，看得见发力线；位移用更宽，命中用插入/近景；不要整段都用同一种中景。
- 竖屏 9:16 的全身高速动作最容易比例失真：优先腰上/3/4 身中景，让动作朝镜头进出；脚法、兵器、接触点用插入镜头补。必须完整全身时，减小跳跃幅度或改用 1344×768 横屏。
- 相邻切点仍守 30° 与轴线；打斗越乱，左右关系越要锁死。切后重述谁在左、谁在右、谁面向哪边。
- 命中瞬间可以短促 push-in 或轻微震动，但全片只出现一两次；每个接触都震等于没有冲击。

## 4. 提示词写法

身份块仍然静态、逐字一致；动作块只写动作、方向、速度、受力和结果。不要在同一句里复述发色和衣服。

```text
[static identity block for A and B]

The fight is already in progress at 0.00 seconds. A steps in with a right straight to B's jaw.
B slips outside, parries with his left forearm, and immediately counters with a left hook to A's ribs.
A's torso folds; his rear foot skids half a step, then he drives forward with a right knee.
Every recoil flows directly into the next attack; the exchange stays at full real-time 1.0x speed from the first frame to the last.
```

- 正向写实时速度：`full real-time 1.0x speed`、`full-weight impacts`、`the next attack begins before the previous recoil ends`。不要只写 `fast`。
- Wushu 模板的 `无慢动作，无停顿` 可以沿用，但只在加载该动作 LoRA 时作为模板的一部分；不要把所有提示词写成否定清单。
- 打斗和台词不要叠在同一段交手。先完成攻击和受击，再让说话者开口；更稳的是把台词放到单独一段。Combat V2 模型卡明确提示：同镜打斗 + 台词若在击打完成前说话，容易退回底模慢动作。
- 时间码不是动作密度：动作模板多数只写 `[Shot N]` 和招式顺序。需要硬切或首尾锁定时才写时间码；不要为每一拳写 0.3–0.5 秒的微节拍，H3 更容易把它变成停顿。
- 每个切点写 `This is a hard cut.`，切后重述景别、机位、左右关系和当前兵器/道具状态。
- 音效跟着因果写：破空、衣料摩擦、格挡闷响、命中、落地、喘息；一个动作一个主 SFX，不要最后堆一串拟声词。

## 5. 首尾帧、参考图与链式

- **连续动作只用首帧（I2VA/FL2VA）或首 + 尾两帧**。首帧要“动作已上膛”：重心偏移、拳已收在肋侧、兵器已抬起；不要中立站姿。
- 尾帧选动作中段/命中后位移，不选收势定式。FL2VA 的尾帧是构图和状态锚点，不是第二段故事。
- 需要身份稳定时用 Ref2VA：2 个角色参考 + 1 个场景/服装/道具参考，图和文字各锁一半；参考图不承担动作相位。
- 链式：上一段真实尾帧做下一段首帧，切在动作中段；下一段换景别/机位，否则连续两段像同一条。身份块逐字复制。
- **不要为一个连续打斗生成 3 张以上动作相位关键帧再塞进 AddGuide**。要动态分镜就明确接受硬切；要连续运动就用首尾帧 + 高密度编排。
- 关键帧只解决构图、身份、遮挡线索；运动靠提示词、参考视频/动作 LoRA 和采样，不靠多锚点。

## 6. 慢放/定格诊断顺序

按这个顺序排查，不要一上来换模型或加 LoRA：

1. **数事件**：5 秒有没有 5–7 个主要动作事件？不够就重编，不要重跑同一提示词。
2. **读提示词**：是否只有 `two people fight`、是否写了 friendly standoff/锁定对视、是否每镜都复位、是否一拳一镜、是否缺少受击反应？
3. **查锚点**：同一条连续打斗是否有 3 个以上 AddGuide？删到只剩首/尾构图，或改成允许硬切的分镜板。
4. **查台词**：打斗和台词是否同镜且在击打前说话？先打完，或拆条。
5. **查采样**：高动态配 4 步 turbo 容易软/糊；先提高步数或改两段采样，不要用清晰度词硬撑。
6. **再上辅助 LoRA**：Motion Continuity Repair 单独约 0.9；与 Combat 同用时一采 0.5–0.7、二采 0.2–0.3。先做权重 0/1 对比，确认 LoRA 真的挂上。
7. **最后才换蒸馏路线**：PDD Acc 8/4 步要求 euler、CFG 1、SigmaShift 12/3，并移除其他蒸馏 LoRA；模型卡称支持 pruned/W4A8，但本机未实测。
8. **后期加速不算修复**：帧插值、光流、变速只能掩盖，不能补动作因果。用户明确要风格化速度效果时才做，并在回执里标注。

## 7. 本机兼容性与测试阶梯

当前 Set C 是 16 GB 路线：W4A8 mixed/pruned DiT、Qwen3VL-4B FP8 文本编码器、Pika turbo 4 步、Sol VAE。它的 5 秒生成能力已验证，但动作质量取决于编排与采样；它**不是动作 LoRA 的即插即用底座**。

| 组件 | 体积/要求 | 本机判断 |
|---|---|---|
| Combat Base V2 | 约 155 MB | 便宜可试；W4A8 兼容性未验证；需自定义图加 `LoraLoader` |
| Motion Continuity Repair | 约 155 MB | 同上；单独 0.9，配 Combat 一采 0.5–0.7、二采 0.2–0.3 |
| PDD Acc 8-step | 约 1.37 GB | 模型卡称支持 pruned/W4A8；会改采样配方；未实测 |
| Wushu Action v7 | LoRA 约 596 MB，但底座要求 34 GB 全量 int8 + 32B bf16 TE | **不是 16 GB W4A8 + 4B TE 的 drop-in**；不要当轻量补丁下载 |
| 动作参考视频 | 2–3 秒真实片段 | 值得优先于大模型堆栈；本机尚未验证 Ref2VA 视频参考路线 |

测试阶梯（每次只改一个变量，保留同一提示词/种子做 A/B）：

1. **5 秒、目标画布、纯提示词**：先把事件密度和因果链写对，不加 LoRA、不加锚点。
2. **提示词修正**：具体招式、命中反馈、实时速度、无复位；如果仍慢，记录是哪一拍慢。
3. **加 Motion Repair 0.9**：自定义工作流，确认权重 0 与 1 有可见差异。
4. **Combat V2 0.9 + Motion 0.5–0.7，两段采样 25%**；采样器用 `res_multistep` + `simple` 或 `euler` + `beta`。
5. **PDD 8 步**：仅在前述路线仍因采样软/慢时尝试；按模型卡重置 schedule。
6. **Wushu 全量路线**：只在升级到 34 GB 底座 + 32B TE 后考虑；不是当前机器的默认下一步。

当前 `h3_c_i2va768` / `h3_c_fl2va768` / `h3_c_ref2va768` API 图只带固定的 Pika turbo LoRA，没有动作 LoRA 槽位。要加 LoRA 必须用自定义 API 图走 `--workflow`，不要假设 fastpath 会自动挂载。

## 8. 动作 QA

`h3_status.py --dynamic-check` 只抽首/中/末 3 帧，最大平均差 > 1.0 就报 `dynamic`。它是烟测，能过一条 25% 近静止的慢打斗。动作片另做：

1. 抽 8–12 帧或接触表，逐段数主要动作事件；5 秒少于 5 个事件判不合格。
2. 看有没有 > 0.5 秒的定格、对峙、收势复位；有就回生成。
3. 统计近静止比例（帧间 MAD < 1.0）；本项目目标：动作段 < 20%。v2 为 25.9%，所以仍被判慢。
4. 1.0× 实速从头看一遍，听冲击音和呼吸；不要只凭首/中/末帧。
5. 报告实际事件数、近静止比例、是否做过变速；没有做就说没有。

## 来源

- Wushu Action v7 模型卡与工作流：https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora
- Combat Base V2 模型卡与 REF2VA/FL2VA 工作流：https://huggingface.co/JOKER141/MiniMax-H3-Combat-Base-V2
- General Motion Continuity Repair 模型卡：https://huggingface.co/JOKER141/MiniMax-H3-General-Motion-Continuity-Repair
- PDD Acc LoRA：https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs
- 本机 E1 AddGuide 测试、v2 动作密度 QA、公开工作流 JSON 解析（2026-11）。


---

## FILE: dsh-plugin/skills/h3-director/references/agent-workflow.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/agent-workflow.md)

# Agent production workflow

H3 Lite uses a small, explicit production contract for complex requests:

```text
intent route → reference/identity anchors → prompt enhancement → execute → verify
```

This is a local ComfyUI workflow pattern. It is informed by public agent-skill
designs such as Higgsfield's open skills, but it does not call Higgsfield,
MCP, or any cloud model and it does not change the H3 inference graph.

If the brief is underspecified, use the optional
[`prompt-assist.md`](../../h3-prompt/references/prompt-assist.md) reference. It borrows the public
`STYLE → SCENE → MOTION → AUDIO → NEGATIVE` organization as a writing scaffold,
then translates it to H3's native prompt fields. It is deliberately a fallback
for ambiguity, not an extra generation stage.

## 1. Route the intent before writing the prompt

Choose one primary route from the user's actual input and acceptance criteria.
Do not choose a route from a style adjective alone.

| User intent or input | Route | First decision |
| --- | --- | --- |
| Text describes the whole clip | `T2VA` | Build the opening state and audiovisual timeline. |
| One image must be the opening frame | `I2VA` | Lock the image at `0.00` and describe only forward motion. |
| Start and end images are supplied | `FL2VA` | Describe the path between both anchors; 1-3 hard-cut beats may sit inside one ~10 s run. |
| Only an end image is supplied | `L2VA` | Work backward toward the final anchor. |
| Several images/video/audio references must be retained | `Ref2VA` | Define reference roles and retention before the timeline. |
| Same person or object across multiple shots | `Ref2VA` when installed; otherwise `I2VA` per shot | Create a master anchor and shot-specific continuation anchors. |
| Install, repair, or diagnose the environment | doctor → planner → preflight | Do not queue generation until the selected set is usable. |

Use one main route per job. If a required reference is missing, ask one
targeted question or state the fallback; do not silently switch from identity
preservation to a text-only generation.

## 2. Build an anchor sheet for identity and continuity

Before writing a multi-shot prompt, make a compact internal anchor sheet. Use
stable labels throughout the prompt and manifest:

```text
Subject A: dark-haired man, black wool coat, charcoal knit sweater
Picture 1: master two-shot keyframe, opening composition
Picture 2: reverse keyframe, same coast, same wardrobe and weather
Retention: face, hair, clothing, body proportions, location palette
Allowed change: gaze, small hand movement, camera angle, wave motion
Forbidden drift: new clothing, age change, extra people, changed weather
```

For each reference, record four things:

1. **Role** — identity, first frame, last frame, pose, camera, style, motion, or sound.
2. **Retention** — what must remain fully preserved, partially preserved, or only used as a weak reference.
3. **Allowed change** — the action, camera move, focus shift, or expression that may evolve.
4. **Forbidden drift** — identity, wardrobe, markings, props, background, aspect ratio, or audio details that must not change.

For a character-driven sequence, prefer one master reference plus one
continuation/reverse keyframe per new composition. The keyframe is an anchor,
not a second story: keep its wardrobe, markings, lighting, weather, and
background consistent with the master before adding motion.

Mode-specific anchor rules remain strict:

- `I2VA`: `<Picture 1>` is fully referenced at `0.00` seconds.
- `FL2VA`: the first reference is anchored at `0.00`, the final reference at the targeted duration, and the middle path is physically continuous. Cuts are allowed inside the run; restate framing after each `This is a hard cut.` The model may merge adjacent beats into one camera move, so accept 3-4 clear visual sections rather than an exact shot count.
- `L2VA`: the reference belongs to the final shot, not the opening shot.
- `Ref2VA`: define `Subject`, `Picture`, `Video`, and `Audio` roles before `summary`, `retention_analysis`, `detailed_description`, and sound fields.

If official Ref2VA components are not installed, do not present this anchor
sheet as proof that Ref2VA is available. Use it to prepare I2VA/FL2VA shots or
stop with a clearly labeled experimental choice.

## 3. Enhance the prompt in five passes

Do not ask the user to write a schema. Convert a natural-language brief into
these internal passes, then emit the field order required by the selected H3
workflow:

1. **Intent sentence** — what should be visible by the end of the clip?
2. **Observable locks** — subjects, wardrobe, props, location, lighting, framing, and reference retention.
3. **Timeline** — one action per state change, in playback order; add a cut only when it reveals new information.
4. **Camera and sound** — camera position, movement amplitude/speed, focus plane, ambience, physical sounds, dialogue, and music policy.
5. **Compact anti-drift checks** — only exclusions that prevent a concrete failure, such as identity drift, extra limbs, subtitles, or a focus hunt.

Then translate the result to:

```text
integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...
```

For `Ref2VA`, use the six-section reference schema instead. Prompt length is
not the objective; every sentence should bind an identity, spatial relation,
state change, sound event, or reference constraint. Keep camera language
physical and testable: `slow 10-degree arc at eye level`, not only
`cinematic camera`.

When the user supplies only a style adjective or a loose creative idea, consult
`h3-prompt` skill's `references/prompt-assist.md` before asking a long list of questions. Fill only the missing
production choices that can be bounded safely (one clear action, one main
camera move, native ambience, and the local fast canvas); ask one targeted
question when a missing reference, duration, dialogue, or route would change
the acceptance criteria. If a public Higgsfield page is browsed, use it for
structure and vocabulary, not for factual claims or local model capabilities.

## 4. Execute and verify as one contract

- On an unchanged installation, use one `h3_fastpath.py` submission and one bounded watch; do not run a second identical job while the manifest is active.
- Keep polling silent for the user, but save the effective prompt, route, anchors, workflow fingerprint, queue ID, actual execution time, and output metadata. The runtime writes the generated anchor card to `anchors.json` and links it from `manifest.json`.
- Verify both media and creative acceptance: video/audio streams, duration/FPS, non-black frames, motion, and first/middle/last identity/wardrobe continuity. Natural frame padding (for example 244 -> 260) is accepted; do not rerun or trim for exact frame count.
- For a single-run multi-shot clip, check cuts only near the requested times (+/-0.5 s). Do not full-scan scene scores. A requested hard cut that became a motivated camera move still passes if it creates a new shot size or camera position.
- When references or multi-shot signals are present, `h3_status.py` records advisory `anchor_qa` comparisons for the first/middle/last frames. A valid MP4 is not proof of face or reference consistency: pixel similarity is only a drift signal, so inspect sampled frames before reporting success.
- If a higher-resolution run is too slow or fails preflight, keep the same prompt and anchors while falling back to the approved lower canvas; do not silently remove audio or identity constraints.

## 5. Local boundary

H3 Lite is a Windows-first, local ComfyUI skill. The workflow patterns above
can be learned from cloud-agent skills, but generation still uses the installed
H3 component set, local GPU/RAM/pagefile, and native H3 audio path. Cloud
features, proprietary identity models, and advertised resolutions must not be
reported as local H3 capabilities until the matching node, checkpoint, and
verification path exist on the user's machine.

## Source design reference

- Higgsfield public skill bundle: <https://github.com/higgsfield-ai/skills>
- Higgsfield video-explainer prompt templates: <https://github.com/higgsfield-ai/skills/blob/main/higgsfield-video-explainer/references/prompts.md>
- Higgsfield prompt generator: <https://higgsfield.ai/ai-prompt-generator>
- Higgsfield skill landing page: <https://higgsfield.ai/skills>
- H3 prompt-writing reference: <https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing>


---

## FILE: dsh-plugin/skills/h3-director/references/director-sequences.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/director-sequences.md)

# Director-level multi-segment sequences for MiniMax H3

Use this extension when the user wants real cinematic coverage — establishing
shot, over-the-shoulder (OTS), reverse OTS, reveal, etc. — rather than a single
continuous clip. It captures the end-to-end workflow that was validated
locally: split one "scene" into several H3 I2VA runs, then stitch them with
ffmpeg. It also documents the identity-consistency and skin-quality lessons
that repeatedly broke naive attempts.

## 1. Prefer one multi-shot FL2VA when it fits

A single H3 generation is not limited to one continuous shot. The official
prompt schema supports cuts, and the local Set C FL2VA produced 3 clear shots
in one 10 s run with hard cuts at 2.42 s and 5.04 s. For a 10 s 3-4 beat
sequence, write one FL2VA prompt with `[Shot N]` and `This is a hard cut.` It
preserves identity, palette, weather, and ambience better than stitching
separate renders.

This is the **story/dialogue** default. Dense combat is a different budget:
published action workflows use 124-frame (5.2 s) units with no dense timeline
anchors, then chain or extend. A 10 s single FL2VA is fine for a fight only
when it carries 9-13 major action events; otherwise use two 5 s units. See
[`action-combat.md`](action-combat.md).

Split into separate segments only when:

- the sequence needs more than 4 beats or more than ~15 seconds;
- a fight cannot fill 9-13 major action events in 10 s;
- a shot needs a different reference/route, or a long line that must be chained;
- combat and the required line cannot be separated within one shot;
- the same prompt has failed twice to create a new composition;
- true shot/reverse-shot coverage needs different identities or angles that one
  generation cannot hold.

For those cases, design the scene as discrete shots and generate each
separately:

| Segment | Typical role | Camera |
| --- | --- | --- |
| C1 | Establishing | medium-wide, two-shot, slow push-in |
| C2 | OTS A → B | behind A's shoulder, focus B |
| C3 | Reverse OTS B → A | behind B's shoulder, focus A |
| C4 | Reveal / wide | side angle exposing the space/danger |

Plan the sequence first with the film-decision card in
[`cinematic-prompting.md`](../../h3-prompt/references/cinematic-prompting.md): irreducible state, gaze
flow, relationship pressure, and rhythm. Keep the negative space, avoidance, or
void as the *subject* when the drama is internal (e.g. a breakup), not just as
empty background.

## 2. First-frame images (ImageGen)

Generate one first-frame image per segment. Practical rules that were learned
the hard way:

- **Distinct filenames for parallel calls.** ImageGen derives the saved
  filename from the prompt's *opening words*. Parallel calls that start with the
  same phrase (e.g. "Photorealistic 35mm feature-film still…") collide and
  overwrite each other, leaving only one file. Prefix each prompt differently
  (`FRAME1 …`, `FRAME2 …`, `BU2 …`) so the filenames differ.
- **Watermark crop.** ImageGen adds a bottom "AI 生成" strip (~80px). Crop it
  before using the frame as an H3 anchor:
  ```powershell
  & "<ComfyUI>\ffmpeg.exe" -y -i in.png -vf "crop=iw:ih-80:0:0" -pix_fmt rgb24 out.png
  ```
- **Identity inheritance via reference images.** For OTS / reverse-OTS /
  occluded segments, the first frame should still *expose* the occluded
  character's identity features (skin tone, hair, beard, clothing). If the face
  is fully hidden by a shoulder, H3 invents a new identity (different skin
  tone, hair, age). The reliable fix is to generate those first frames **with
  reference images**: pass the clear two-shot frames (e.g. the establishing C1
  and the side-angle C4) as ImageGen `image` inputs with `input_fidelity`
  ~0.7–0.9, so the faces are inherited rather than reinvented. Pure-text
  identity locking alone is not enough for occluded shots.
- **For continuous action, generate one first frame or a first/last pair.**
  Draw the opening with the action already loaded (weight shifted, strike
  chambered), and the last frame in mid-action or displacement. Do not draw a
  sequence of action-phase keyframes for Add Guide; those pin frames and
  produce cuts/freezes instead of interpolating motion. If you want a multi-
  keyframe board, treat it as an animatic with intentional hard cuts.

## 3. I2VA prompt: lock identity across segments

Within each segment's `integrated_multimodal_description`, keep every
character's descriptors byte-for-byte consistent across all segments
(hair, beard, skin tone, clothing color/type). For OTS segments, add an explicit
identity lock in both the first-frame anchor and the body:

```text
Man A is fair light skin, early 30s, short curly dark hair, trimmed dark beard,
olive-green hooded jacket — IDENTICAL to the establishing C1 shot; DO NOT alter
his skin tone, age, or hair.
```

Treat stable names (`Man A`, `Man B`) as the contract; never alias and switch.

## 4. Skin quality on the validated W4A8 route

W4A8 quantization turns "natural skin texture / freckles / stubble /
realistic roughness" into **sandpaper grain**. The instinct to ask for
"realistic texture" backfires on the low-VRAM route. Prefer:

```text
smooth refined skin, fine delicate pores, soft highlight roll-off, not plastic, not rough
```

and add compact anti-rough constraints:

```text
no heavy grain, no waxy plastic, no oily sheen
```

> Note: [`cinematic-prompting.md`](../../h3-prompt/references/cinematic-prompting.md) lists `natural skin
> texture` as a *general* supporting constraint for film looks. On the validated
> W4A8/4B graph it produces a rough result; use the smooth phrasing above
> instead for this route. The smooth phrasing still reads as real skin, just not
> abrasive.

## 5. Submitting and waiting (local Windows sandbox notes)

These are execution-environment quirks observed on the agent's sandbox, not H3
limitations. They directly affect how the fastpath commands are run here:

- **`--watch` client timeout.** `h3_fastpath.py --watch` can hit a client-side
  timeout (~71s) while ComfyUI keeps rendering in the background (queue shows
  `running`, VRAM full). Do not treat that timeout as a failed render. After
  submission, watch the output directory instead, or run a background waiter
  that polls `<ComfyUI>/output/video/` for the expected filename.
- **Spurious `failed` from safe-delete.** A submission may report `failed`
  because the sandbox intercepts the `.submit.claim` write. Bypass by prefixing
  the command with `env -u CODEBUDDY_SESSION_ID -u CLAUDE_SESSION_ID -u
  PYTHONPATH`. (The `helper failed` line is often a non-fatal post-process error;
  verify the MP4 exists with ffprobe before assuming failure.)
- **Background scripts can be cleaned up mid-run.** A long background bash task
  that submits segment-by-segment then waits can be terminated by the host, so
  later segments never submit. Mitigation: submit all segments in a quick
  foreground loop first, then background only the wait + stitch step; or give
  the waiter a resume/re-submit check keyed on which output files already exist.

A robust per-segment submit command (quality, 16:9, ~5s):

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --root <fast-disk>/ `
  --mode i2va `
  --first-frame <clipN_firstframe.png> `
  --prompt-file <clipN_prompt.txt> `
  --resolution 1024x576 `
  --video-seconds 5 `
  --fps 24 `
  --profile quality `
  --seed 42 `
  --filename-prefix video/H3CliffDirector_C<N>
```

(`--root <fast-disk>/` keeps large intermediates off the tight system drive; adjust to
your disk layout.)

## 6. If you must stitch: continuity first

Hard cuts on action are the default for drama; use a crossfade only when the
brief asks for it. Stitching independent generations is where visual and audio
seams appear, so follow these rules:

- **Chain frames.** Use the real last frame of shot N as the first frame of shot
  N+1 (FL2VA), and cut in the middle of an action or on an eye-line shift.
- **Keep one audio bed.** Lay a continuous rain/room-tone/BGM bed under the
  whole program, or crossfade ambience 20-50 ms at each join. Do not restart
  ambience per shot.
- **Use J/L cuts.** Let the next shot's sound enter 2-4 frames early, or let the
  previous shot's sound run 2-4 frames late.
- **Normalize once, after stitching.** Never loudnorm each segment; quiet
  segments would be boosted tens of dB and the joins would pump.
- **Keep natural frame counts.** Do not trim a shot to an exact second. Accept
  natural padding such as 244 -> 260 frames.

All segments must share resolution, FPS, and (for audio crossfade) have an audio
stream; normalize the final program only after assembly.

**Cumulative offset formula** (transition duration `T`, segment durations
`d1, d2, …`):

```text
offset_1 = d1 - T
offset_k = offset_{k-1} + d_k - T      (k >= 2)
```

**Example** — 4 segments with durations `6.583, 5.167, 5.167`s and `T = 0.5`s
(offsets `6.083, 10.75, 15.417`):

```powershell
& "<ComfyUI>\ffmpeg.exe" `
  -i C1.mp4 -i C2.mp4 -i C3.mp4 -i C4.mp4 `
  -filter_complex `
    "[0][1]xfade=transition=smoothstep:duration=0.5:offset=6.083[v01];`
     [v01][2]xfade=transition=smoothstep:duration=0.5:offset=10.75[v02];`
     [v02][3]xfade=transition=smoothstep:duration=0.5:offset=15.417[v];`
     [0:a][1:a]acrossfade=d=0.5[a01];`
     [a01][2:a]acrossfade=d=0.5[a02];`
     [a02][3:a]acrossfade=d=0.5[a]" `
  -map "[v]" -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac FINAL.mp4
```

Verify the result with `ffprobe` (duration, FPS, video+audio streams) before
presenting it.

## Sources

- Companion references: [`cinematic-prompting.md`](../../h3-prompt/references/cinematic-prompting.md),
  [`prompt-writing.md`](../../h3-prompt/references/prompt-writing.md),
  [`face-quality.md`](face-quality.md).
- Validated on RTX 4060 Ti 16 GB, ComfyUI H3 W4A8/4B graph, 1024x576 / 24 fps /
  quality (8-step) profile.


---

## FILE: dsh-plugin/skills/h3-director/references/face-quality.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/face-quality.md)

# Face-quality routing

这份说明只影响规划、提示词和生成后的检查，不会向视频推理图增加采样步数、额外模型或后处理推理。

## 路线选择

| 目标 | 默认路线 | 说明 |
|---|---|---|
| 物体、风景、卡通 | T2VA | 先用 fast 基线；没有人脸时通常更容易得到稳定画面。 |
| 单镜头可辨认人物 | I2VA / FL2VA 首帧 | 使用清晰首帧；限制为单人、正脸或四分之三脸、稳定发型和服装。需要精确首尾构图时用 FL2VA 双关键帧。 |
| 人像/角色成片、身份一致性优先 | Ref2VA（已验证，2026-11-19） | 使用 Set C 的专用 ref2va W4A8 底模 + Pika ref2va lite LoRA + `MiniMaxH3ReferenceToVideo`，`ref_image_size=match`。见下方“已验证的 Ref2VA 人像路线”。 |
| 高细节输出 | 先生成 1344×768 基础结果 | 官方 H3-Regenerate-2K 不是本地低显存默认路径。 |

I2VA 会重新生成首帧，因此参考图和视频首帧不一定逐像素相同。它可以改善构图和大致身份，但不能保证“就是同一张脸”。如果身份一致性是验收条件，应优先使用已验证的 Ref2VA 路线，或把参考图作为视频开头的静态画面再进行剪辑衔接。

## 已验证的 Ref2VA 人像路线（2026-11-19，RTX 4060 Ti 16 GB）

受控 A/B（同一角色首帧、1344×768、124 帧、seed 6、4 步、均通过含音频校验与三帧人脸抽检）：

| 路线 | 总耗时 | 人脸表现 | 运镜/构图 |
|---|---|---|---|
| FL2VA 双关键帧 + Pika fl2va lite | 151.97 s | 身份稳定，脸较小 | 严格遵循提示词运镜，首尾帧锚定 |
| Ref2VA match + Pika ref2va lite | 161.23 s（+6.1%） | 脸更大更清晰，身份更稳 | 自发推近更强，运镜指令遵循较弱，无末帧锚定，背景道具有轻微漂移 |

选型结论：人像/角色成片默认 Ref2VA `match`；需要精确端点构图或严格运镜时用 FL2VA 双关键帧。`ref_image_size=max`（短边 2048px）未实测且慢数倍，不作默认。

组件要点（详见 `component-sets.md` Set C）：

- FL2VA 与 Ref2VA 是两个独立的 transformer 底模；ref2va LoRA 必须配 ref2va 底模（`minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`，11,770,657,048 B）。Set A 混合底模是唯一例外。
- 参考图绑定必须用 autogrow 点号键 `"ref_images.ref_image_0": ["<LoadImage节点>",0]`；扁平键会通过校验但在执行期抛 TypeError。
- 提示词用 `<Picture 1>` 锚定身份（"the woman in <Picture 1> ... exactly as shown in <Picture 1>"），shift_video 12.0 / shift_audio 3.0。
- Pika lite LoRA 以 strength 1.0 整体替换 turbo LoRA，不叠加；4 步音频偏弱是路线固有特性。

## 社群人像工具现状（2026-11 调研结论）

- Temporal Face Detailer（nikythebikky）：GitHub 仓库 404，comfy.icu 页面为过期缓存，不可用。
- Impact Pack FaceDetailer / ZenFaceDetailer：逐帧修复会引入闪烁，且 SDXL 修复模型对 16 GB 显存叠加 H3 不友好；不作为 H3 默认后处理。
- facerestore_advanced（flickleafy 分支）：维护停滞（2025-07 后无更新），CodeFormer/GFPGAN 类逐帧修复同样有闪烁与“换脸感”风险。
- 换脸类节点属于事后补丁，改变身份而非提升生成质量，仅在用户明确要求时使用。
- 结论：在 H3 上提升人脸质量的第一手段是路线选择（Ref2VA）+ 提示词护栏 + 更高分辨率，而不是外挂逐帧修复器。

## 人脸提示约束

- 第一镜只保留一个人；眼睛、鼻子、嘴和发际线清晰可见。
- 固定正脸或四分之三角度，避免快速转头、强烈表情、说话和大幅运镜同时出现。
- 先验证无对白的人脸稳定性，再加入对白或口型同步。
- 首帧、中帧、末帧都要检查；视频存在、动态正常和音频存在，不等于人脸质量合格。

## 零推理开销约束

人脸质量判断只允许使用提示词改写、已有首帧、抽帧检查和轻量媒体分析。不得因为“人脸模式”自动增加采样步数、叠加额外采样器、加载第二个生成模型或重复生成一遍。


---

## FILE: dsh-plugin/skills/h3-director/references/native-dialogue.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/native-dialogue.md)

# H3 原生台词纪律

本项目硬规则：**台词必须进 H3 原生整体生成，不做后期 TTS 拼接。** 后期只做响度、字幕与混音。

## 为什么不用后期 TTS

- H3 的音频分支与画面同一次推理，口型、呼吸、环境声天然同步；后配 TTS 会丢失这种同步，还要额外对口型。
- 本项目已有多次日语/中文原生台词跑通；需要的是控制长度与说话人，而不是换路线。

## 声音圣经与选角声学距离

跨镜音色一致性的完整方法见 [`../../h3-prompt/references/voice-consistency.md`](../../h3-prompt/references/voice-consistency.md)。导演阶段必须先做完这两件事，再写分镜：

1. **声音圣经**：每个角色一行——语言 / F0 目标带 / 质地 / 语速 / 咬字口音 / 情绪基线 / 正典样本。全片逐字复用，不换同义词。底床声（念经、雨、房间底噪、磁带嗡鸣）同样指定唯一正典来源，不要每镜各自生成。
2. **选角声学距离**：模型靠音高、共振峰、声道长度区分说话人，不靠提示词里的绑定关系。同场两个同性别说话人 F0 目标带差 **≥25–30 Hz（约 4 个半音）**且质地不同；一条里 ≤2 个说话人最稳，3 个以上同性别说话人默认高风险。距离不够就改音色设定、换轻口音样本或拆条，不要堆形容词。

每镜的声音验收条件写进分镜表备注（F0 目标带、质心范围、与正典样本的半音差），不要只留在对话里。交付前用 `h3-post/scripts/h3_voice_audit.py` 量化：同一角色跨镜 F0 中位数极差 ≤2 个半音、质心极差 ≤约 25%。

## 单段上限与链式

- 一次生成 ≤362 帧（约 15 s）。超过就拆成同说话人的链式段。
- 不要提交 481 帧：本机曾出现 48 W 自旋死锁，必须重启 ComfyUI。
- 链式段之间沿用同一角色参考图、同一服装/光位描述、同一声音描述；上一段真实尾帧可作为下一段首帧。
- 长台词优先拆成“每段一句”，而不是在一段里塞三句。单镜多句会压缩、抢拍、提前说完，落点漂移。

## 每镜一句

- 一句为主；两句必须短且留换气，并在分镜表里标出两句的落点。
- 说话者写法：

```text
She says the line once and nothing else: "もう、遅いよ。"
```

- 不要写“她说了以下几句话”再列三条；模型会自行分配时间。
- 台词保持用户原语言，周围描述用英文。日语台词不要翻译成英文再让模型“说日语”。

## 短台词不留空白窗口

一句 4 词台词放在 4 秒镜头里，剩余窗口会被模型用含糊语音填满（filler-babble），或把台词说两遍。

三选一：
1. 扩写到约 8 词以上；
2. 给窗口安排任务：雨声、门闩、脚步、一个动作节拍，写进 `overall_soundscape` 或时间线；
3. 把镜头剪短到台词长度。

## 画面里每张脸都要有嘴部状态

- 说话者：`says the line once and nothing else`。
- 非说话者：正向静止事实，`lips at rest, jaw closed, his breath lifting his chest`。
- 不要写“没说话/闭嘴”——否定式仍然把“说话”带进上下文。
- 画外说话：明确 `off-screen`，可见人物嘴唇闭合。

## 声音字段

```text
overall_soundscape: <环境音、物理音、动作音，1–4 句>
non_diegetic_music: <观众专属配乐，或 N/A>
```

- `不要对白` / `no dialogue` 只去掉台词，保留环境音、动作音与音频流。
- 只有 `完全静音` / `no audio` 才关闭全部音频，并把 `overall_soundscape` 写 `N/A`。
- `non_diegetic_music: N/A` 只表示没有观众配乐。
- 用户要求 BGM 时，可以在 H3 原生音频分支里生成，或后期加轨；不要让音乐盖过台词。

## 时长预算（经验值）

| 镜头 | 台词 |
|---|---|
| 3 s | 一句 6–10 词（日语更短，约 8–14 拍） |
| 5 s | 一句 12–16 词，或两短句 |
| 10–15 s | 两句以内，中间留可听见的停顿 |

这些是写作预算，不是模型限制；以实际试听为准。

## 验证

1. 文件有音频流只是前提，不等于台词正确；
2. 必须**听**：词是否对、语言是否对、是否只出现一次、是否被截断；
3. 有条件时用本地转写核对关键词，但转写只作辅助；
4. 口型与声音同步在首/中/末段各看一次；
5. 不合格就改台词或改镜头长度重跑，不靠后期变速或补录掩盖。

## 已知失败与对策

| 症状 | 对策 |
|---|---|
| 前后含糊语音 | 填满窗口 / 剪短镜头 / 只一次声明 |
| 台词说两遍 | 同上；检查是否写了两个说话动词 |
| 提前说完、尾部空白 | 缩短台词或延长停顿事件 |
| 多句挤在一起 | 拆镜，一句一镜 |
| 第二张脸也在动嘴 | 加正向嘴部静止事实 |
| 语言串味 | 台词保持原语言，明确 `says in Japanese/Chinese` |
| 链式段音色变化 | 同角色参考、同声音描述、同语言；必要时重生成上一段尾帧 |
| 同一角色跨镜忽年轻忽苍老 | 声音圣经逐字复用 + VOICE 块独立；按 F0/质心门限量化验收，超标重跑 |
| 两个男声听起来是同一个人 | 选角拉开声学距离（F0 差 ≥25–30 Hz、质地不同）；≤2 个说话人；换轻口音样本 |
| 接参考音频后开头抢话/台词重复 | 换干净干声样本；写明 `do not copy its words`；仍不行就断开参考音频 |
| A→B→A 第三句错人 | 给第三次发声新音频槽（同一文件），写 `S3 using Audio 3` |
| 一句轻笑/叹气后声音全乱 | incidental 发声格式化成台词或删掉 |
| 底床声（念经/嗡鸣）换人 | 全片用同一正典样本，不要每镜各自生成 |


---

## FILE: dsh-plugin/skills/h3-director/references/professional-coverage.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/professional-coverage.md)

# 专业覆盖与场面调度（H3 适配）

这份文档把现场导演的覆盖、轴线、blocking 和节奏判断，压缩成 H3 能执行的写法。它不是 H3 的镜头元数据，也不保证模型理解剧组术语；每个判断最后都要落到画面里可见的位置、方向、动作和声音。

## 1. 先让场景成立

写镜头前，先用一句话回答四个问题：

| 问题 | 要写出的内容 |
|---|---|
| 目标 | 这个 beat 里谁想要什么？ |
| 障碍 | 什么人或什么东西挡着他？ |
| 动作 | 他具体做了什么可见的动作？ |
| 转折 | 动作之后，局面变成了什么？ |

```text
目标：她把钥匙放回窗台，但不想被看见。
障碍：对方守在门口，灯还亮着。
动作：她蹲下，把钥匙放在窗台边缘，退回半步。
转折：对方没有拿钥匙，先伸手关掉了灯。
```

如果这四个问题答不出来，增加机位也救不了这场戏。提示词里不要写剧情解释，要把解释翻译成动作、视线、距离和声音。

一个 beat 只回答一个问题。文戏 10 秒 3–4 拍，就是 3–4 个清楚的“信息变化”，不是 3–4 个机位重复同一句话；动作戏按 5 秒 5–7 个主要动作事件另算。

## 2. 覆盖不是机位越多越好

专业现场常用 master shot + coverage：先拍一个交代空间的全景，再补中景、反应和插入。H3 的适配方式是：

- 一条约 10 秒 FL2VA 里，把覆盖压缩成 3–4 拍；每拍承担一个明确目的。
- 超过 4 拍、超过约 15 秒，或需要两套完全不同的参考/地点时，再拆成多条生成。
- 单条多镜不要写成“同一瞬间的五个角度”；每个切点必须带来新信息。

| 目的 | 观众获得 | 画面证据 | 放在哪 |
|---|---|---|---|
| 建立 | 谁在哪、空间关系、方向 | wide/medium two-shot、门框/窗框、清楚的左右关系 | Beat 1 |
| 施压 | 谁占上风、距离如何变化 | low/high angle、前景遮挡、人物在画面中的大小 | Beat 2 |
| 反应 | 决定、代价、潜台词 | 脸、手、呼吸、画外视线 | Beat 3 |
| 信息 | 关键道具、文字、时间 | insert、tight close-up、焦点落在物件 | 需要时 |
| 转场 | 时间/地点/情绪变化 | 出画、match cut、声音先入 | 末拍或段间 |

三拍短剧最稳的组合是 **建立 → 施压 → 反应**。如果只有三拍，不要硬塞一个插入镜头；把关键道具放进反应拍的手部动作里。

一个情绪事件不要碎成很多角度。一个人听到坏消息，先用一个稳定的中近景让她听完，再决定要不要切反应；连续切五个表情通常比一个 hold 更弱。

## 3. 轴线、视线与 30°

### 180° 轴线

两个人或人与物之间有一条假想轴线。所有机位保持在同一侧，人物在画面里的左右关系才稳定。

- A 在画面左、B 在画面右，就保持这个关系；反打时 A 看画面右，B 看画面左。
- 一群人对话，把机位想成“观众席”：摄影机在观众这一侧，通常能维持一条统一轴线。
- 车辆、行走、追逐要保持屏幕方向；车从右向左开，切镜后不要变成从左向右，否则像掉头。
- 需要过线时，用轴线上的中性镜头过渡，或让摄影机在可见的运动中越过轴线；不要在两个静态机位之间直接跳边。
- H3 的硬切之后要重述左右关系、视线方向和机位所在侧，不能只写“切到 B”。

### 30° 规则

切到新机位时，角度至少变化约 30°，或景别有实质变化；否则两个画面太像，会像跳剪。

- 同景别 + 同侧 + 同机位 = 跳剪。
- 中景切中近景，如果构图几乎一样，也算跳剪。
- 最便宜的合法变化是景别变化；其次是机位高度、角度和前景层次。

### 视线与动作匹配

- **eyeline match**：A 看画外某处，下一拍给 A 所看之物；再切回 A 时，视线方向要接得上。
- **match on action**：在动作中段切，下一镜从同一动作阶段继续；不要上一镜刚起手、下一镜已经收势。
- **exit side + last state**：记录人物从哪一侧出画、视线朝哪、道具在哪只手、衣服/灯/门是什么状态；下一镜从同一侧、同一状态接上。

```text
Axis: A（左） ↔ B（右），摄影机保持在 A 侧。
Eyeline: A 看 frame right；B 看 frame left。
Exit: A 从画面右侧出画，钥匙在右手，门保持半开。
```

## 4. Blocking：把人物放进空间

Blocking 不是“两个人站着说话”，而是人物位置、距离、朝向和遮挡如何制造关系。

- 用前、中、后景分层：前景可以是肩、门框、桌面边缘；中景是主体；后景交代环境。前景遮挡必须有理由，不要加一团黑边冒充电影感。
- 权力关系可以用高度、距离和遮挡写：站者与坐者、桌前与门边、光里与暗处、完整身体与被门缝切开的脸。
- 一个 beat 只改变一次 blocking：走近一步、退后半步、坐下、转身、把手放上台面。不要让人物同时走位、转身、拿道具、说话。
- 对话镜头要保证说话者的眼睛和嘴可见；需要背影/过肩时，让被遮者仍露出肤色、发型或服装线索。
- 关键帧或 staging 图不能可靠地锁住 H3 的 blocking；必须在提示词里写文字位置锁。

```text
Blocking lock: A 站在柜台后，B 站在柜台外，柜台在两人之间。A 的双手留在收银机上；B 的右手留在台面边缘。A 在 4.2 秒退后一步，B 不越过柜台。
```

不要写“他们在店里争吵”。写谁在哪、谁高谁低、谁被什么挡住、谁先动。

## 5. 复杂度与节奏预算

一条文戏 10 秒 3–4 拍的多镜，按下面上限做预算：

| 维度 | 每条上限 | 超了怎么办 |
|---|---|---|
| 强动作 | 文戏每拍 1 个；动作戏 5 秒 5–7 个主要事件 | 拆拍、缩短段长或加招 |
| 主运镜 | 每拍 1 个 | 改静止，或分时段写 |
| 重要角色 | 3 个 | 减人、分组、用画外声 |
| 地点 | 1 个 | 转场拆条 |
| 特殊物理/VFX | 1 个 | 改成 near-miss / aftermath |
| 台词 | 每拍 1 句 | 拆条、缩短或改成反应 |

文戏两个强动作放在同一拍，模型通常只选一个，或者两个都做一半。动作戏相反：5 秒只有两三个事件时，模型会用慢放和停顿填满；按因果链排 5–7 个事件，再限制同时接触的人数。

节奏预算：

- 文戏快切默认 3–5 秒一切；10 秒 3–4 拍，每拍约 2–3 秒。动作戏按 5 秒单元。
- 动作戏默认 5 秒一个生成单元，段内 1–3 个视觉段；3 秒只用于单次冲击/插入。
- 允许一个 6–8 秒的 hero hold 替代快切；一个决定、一次听完、一个长反应，往往比多切两刀更有力。打斗的 hold 是持续攻防，不是站定对视。
- 1–2 秒的短拍只用于冲击或信息插入，不要连续三个短拍。
- 用快—慢—快或慢—快—慢制造曲线；不要整条按同一个节拍器切。

**单调审计**：连续三个切点都是同景别、同机位方向或同一种运镜，视为模板化，重排 coverage。文戏至少有一个静止反应拍；动作戏至少有一次攻防位置变化，不能连续三刀都是同一招的重复角度。

## 6. 围绕 H3 的可渲染能力设计动作

H3 擅长可见状态、姿态、距离和反应，也能在具体编排下完成打斗；问题通常不在“武打”这个题材，而在泛写、密度不足和同时要求太多。

优先写：

- 具体招式、发力方向、命中位置和受击反应：`right straight to the jaw`、`left forearm parry outward`、`head snaps back, rear foot skids`；
- 一人主攻、一人主守的交替节奏；另一人用画外、背景或单独反应镜头承担；
- 擦肩、near-miss、门缝、桌面两侧的停顿；
- 手已经放在物件上、已经握住、已经放下；
- 人物停顿、听完、呼吸变化、视线转移；
- 剪影、遮挡、背光、局部特写；
- 事后状态：空椅、半开的门、桌上的钥匙、还亮着的灯。

需要拆开或简化：

- 手指细活、快速系扣、复杂兵器缠绕、多人同时抓扯；
- 三人以上同时抢一个接触点、大规模人群、动物调度；
- 在同一拍里完成“进门—争吵—摔物—离开”。

**一个镜头只完成一次转化**：文戏从站到坐、从怀疑到决定；动作戏从交手前状态到一个新的攻防位置。把“每次生成当交付”，不要设计一个必须靠后期修补的镜头；身份、动作或构图失败时，先简化重跑，不要用剪辑掩盖。

动作密度、因果链、慢放诊断和动作 LoRA 兼容性见 [`action-combat.md`](action-combat.md)。

## 7. 出入画与连续性清单

每个切点前后记录：

```text
Exit: 出画侧、视线方向、持物手、服装状态、门/灯/窗状态。
Enter: 入画侧、起始姿态、同一道具的状态、同一光源方向。
Axis: 摄影机仍在哪一侧；屏幕左右关系是否保持。
Sound: 环境底床是否连续；下一拍的声音是否提前 2–4 帧进入。
```

交付前自检：

1. 这场戏的一句话目标/障碍/转折成立吗？
2. 每个 beat 都带来新信息吗？有没有只是换机位的重复？
3. 左右关系、视线方向和出画侧接得上吗？
4. Blocking 的位置、距离和遮挡写清楚了吗？
5. 强动作、运镜、角色、地点、台词是否在预算内？
6. 有没有连续三个同景别/同机位的切点？
7. 结尾是否落在可见的完成态或未解状态，而不是解释性旁白？
8. 如果是动作戏，5 秒是否有 5–7 个主要动作事件、动作是否按因果相接、有没有对峙/复位/定格？

## 来源

- StudioBinder, 180-degree rule: https://www.studiobinder.com/blog/what-is-the-180-degree-rule/
- StudioBinder, film blocking: https://www.studiobinder.com/filmmaking-techniques-film-blocking/
- OSideMedia, higgsfield-ai-prompt-skill（镜头与剪辑写法参考）: https://github.com/OSideMedia/higgsfield-ai-prompt-skill
- 公开多镜与连续性写法参考（非运行时依赖）：https://github.com/higgsfield-ai/skills


---

## FILE: dsh-plugin/skills/h3-director/references/shot-timing.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-director/references/shot-timing.md)

# 镜头节奏与画幅约定（本机）

本文件是本项目在 H3 上反复验证后形成的**默认值**，不是模型限制。用户明确要求时以当次为准。

## 帧数与实际时长

`--length` 与实际帧数并不总是严格的 N+1，不同模板曾出现 72→73 帧、124→124 帧的差异。**以 ffprobe 实测为准**；下表是本机已测到的值。

| 目标 | `--length` | 实测帧数 | 实测时长 @24fps |
|---|---|---|---|
| 3 秒快切 | 72 | 73 | 3.042 s |
| 5 秒常规 | 124 | 124 | 5.167 s |
| 10 秒中段 | 243 | 243 | 10.125 s |
| 10 秒单条多镜 | 244 | 260（Set C FL2VA 实测） | 10.833 s |
| 15 秒上限 | 362 | 362 | 15.083 s |
| 禁止 | 481 | — | 触发 48 W 自旋死锁，需重启服务 |

> 计划时按 24 fps 折算；提交后用 ffprobe 核对实际帧数，不要用参数反推每一档。

## 不硬修时长

- 73 帧就是 3.042 s，不要为了“正好 3 秒”裁掉尾帧。
- 244 帧参数的 Set C FL2VA 实测 260 帧 / 10.833 s（+16 帧自然补帧）；不是失败，不重跑、不裁尾。
- 切点落在动作边界、视线转移或自然停顿上；几十毫秒的误差是正常的。
- 拼接时以每镜真实时长计算偏移；需要整秒交付时，在片头/片尾留黑或调整镜头数，不逐镜硬裁。
- 多镜之间允许 1–2 帧的节奏呼吸；不要为了无缝把动作切在半帧上。

## 节奏模板

**3 秒一镜（快切短剧默认）**：
- 一个动作落地（含完成态）+ 一个运镜 + 一句短语（约 6–10 词台词）；
- 三镜结构：hook（0–3）→ build/turn（3–6）→ payoff（6–9）；
- 相邻镜景别必须变化：WIDE → CU → MEDIUM 或 MEDIUM → DETAIL → WIDE；
- 不要在 3 秒里塞两个动作或两句台词。
- **动作戏例外**：3 秒只够一次冲击 + 一个反应/插入，不够起手—命中—受击—反击的完整交手。不要按 3 秒一拳拆打斗，那会让每条都像慢放。

**5 秒一镜**：
- 一个主动作 + 1–2 个次级动作（呼吸、衣摆、雨）；
- 一句 12–16 词台词，或两短句但要留换气；
- 允许一个复合运镜，但必须分时段。
- **动作戏默认**：5 秒放 5–7 个主要动作事件，段内 1–3 个视觉段；每段 1–2 招，因果相接，不在段尾复位。

**10–15 秒单段（可含 3–4 拍）**：
- 默认一条 FL2VA 写 3–4 拍，每拍约 2–3 秒；切点落在动作中段或视线转移。这是文戏/叙事默认。
- 动作戏只有 10 秒能填 9–13 个主要动作事件时才单条；15 秒需要 13–18 个，通常拆成 3 × 5 秒更稳。
- 需要一个连续长镜头（对峙、独白、环境揭示）时也可以不切；不是强制多镜。
- 必须有一个清晰的起、承、转，结尾有完成态；
- 362 帧是上限，超过就链式；竖屏 362 帧实测约 12 分钟。

## 单次多镜（一条 FL2VA）

- 10 秒建议 3–4 拍；5 秒建议 1–3 拍；文戏不要在一拍里塞两个主动作。
- 动作戏按“主要动作事件”算：5 秒 5–7 个，10 秒 9–13 个，15 秒 13–18 个；一个视觉段 1–2 招，动作按因果相接，不留对峙/复位。
- 每个切点写 `[Shot N] At X seconds, the camera cuts to ... This is a hard cut.`；切后重述景别、机位、朝向。
- 同一条里保持同一 Subject A 外貌块、同一地点色板和同一声音底床；不要把每个镜头写成独立故事。
- 实际切点可偏离请求 1–2 帧；某拍合并为运镜但仍有新构图，不算失败。按视觉段验收。
- 动作提示词可以不写微时间码，只写 `[Shot N]` 和招式顺序；需要硬切/首尾锁定时再写时间。不要为每一拳写 0.3–0.5 秒的节拍。

## 整片节奏与单调审计

- 文戏快切默认 3–5 秒一个视觉段；10 秒 3–4 拍，允许一个 6–8 秒 hero hold 替代快切。
- 动作戏默认 5 秒一个生成单元；5 秒 1–3 个视觉段。不要用 6–8 秒对峙/定格当 hero hold；打斗的 hero hold 是持续攻防，不是停住。
- 1–2 秒短拍只用于冲击或信息插入，不连续三个；快切后至少留一个静止反应拍。动作戏的“反应拍”可以是受击后 0.2–0.4 秒的位移，不是站定。
- 连续三个切点同景别、同机位方向或同一种运镜，视为模板化；重排 coverage。
- 每刀只承担一个新信息；一个情绪事件不要碎成很多角度。动作戏每刀承担一个新的攻防信息，不要同一瞬间换五个角度。
- 覆盖、轴线、blocking 与复杂度预算见 [`professional-coverage.md`](professional-coverage.md)；动作密度与慢放诊断见 [`action-combat.md`](action-combat.md)。

## 画幅与构图

- 短剧默认 9:16 竖屏；常用 768×1344（快速/链式）或 1024×1344、1344×768 横向构图仅用于横屏需求。
- 快速动作和全身构图最容易比例失真：优先**腰上/3/4 身中景（waist-up medium shot）**，保持 7.5 头身。
- 必须全身时，减少动作幅度、避免奔跑和跳跃；用插入镜头补细节，不用大全景硬撑。动作戏在竖屏里可以让动作朝镜头进出，脚法/兵器用插入镜头；必要时改用 1344×768 横屏。
- 人物在画面上方留头顶空间，不要切到发际线；对话镜头眼睛落在上三分线附近。

## 切点选择

1. 动作中段切：上一镜在动作进行中结束，下一镜从动作延续处开始（打斗、递物、转身）。
2. 视线转移切：角色看向某处，下一镜给所看之物。
3. 声音先入切：下一镜的声音提前 2–4 帧进入上一镜结尾。
4. 完成态后切：动作落地并停留 0.15–0.35 s，再切。
5. 不要切在：手势起手前、台词说到一半（除非故意 `<cutoff>`）、人物刚入画未站稳。

## 首尾帧链

- 上一镜的真实尾帧可直接作为下一镜首帧；链式生成时首帧要选**动作中段**，不要选收势。
- 如果尾帧有运动模糊或压缩伪影，先重绘一张干净关键帧，不要直接喂。
- 链式段的角色描述、服装、光位必须逐字一致；首帧只是构图锚点，不是身份的全部。
- 连续打斗只用首帧或首 + 尾两帧，不要用 3 张以上动作相位关键帧做 Add Guide 插值；本机 E1 已证明会钉帧/硬切/定格。详细证据见 [`action-combat.md`](action-combat.md)。

## 分镜先于生成

先写分镜表，再写动作因果链，再画构图关键帧，再写提示词，最后提交。顺序反过来最常见的后果是：镜头接不上、角色不一致、总时长对不上，只能整段重做。动作戏的关键帧只锁首/尾构图，不用来插值动作。

## 实测备注

- 3 秒三镜版《断罪硬币》：72 帧参数实测 73 帧 / 3.042 s，三镜总长约 9.2 s；切点 3.04/6.08，未硬裁。
- 10 秒四拍 FL2VA《雨夜照片》：244 帧参数实测 260 帧 / 10.833 s；检测到 2.42 s、5.04 s 两个强切，第四拍由连续转身完成，尾帧相似度 0.993；按 3 个视觉段验收。
- 10.8 秒四拍战斗 v2：260 帧，MAD > 8 占 21.2%、近静止 25.9%，整条仍被判慢放；证明“有四个切点”不等于“动作密度够”。修正方向是 5 秒单元 + 5–7 事件，而不是继续后期变速。
- 3 锚点 Add Guide E1：73 帧 / 3.042 s，锚点 SSIM 0.798/0.731/0.681，跨锚点 0.25–0.29，输出两次硬切/定格；证明多锚点不是连续动作插值。
- 481 帧曾出现 GPU 48 W 自旋死锁，必须重启 ComfyUI；单段不要超过 362 帧。
- 5 秒双镜与 15 秒单段均在本机跑通；后者用于窗边续集等连续叙事。


---

## FILE: dsh-plugin/skills/h3-generate/SKILL.md (https://github.com/Rimagination/h3lite/blob/HEAD/dsh-plugin/skills/h3-generate/SKILL.md)

---
name: h3-generate
description: 在本机 ComfyUI 上提交 MiniMax H3 生成任务（T2VA/I2VA/FL2VA/Ref2VA），选择路线与分辨率，绑定参考图与参考音频做身份与音色参考；支持双角色 Preprocessor Map + Regional 复刻，监控进度并验证输出。当用户要求渲染、重跑、查看进度、批量生成、角色替换或接音色参考时使用。
---

# H3 生成：热路径

目标：一次提交、一次等待、一次验证。环境未验证时先交给 `h3-workstation`；提示词未定稿时先交给 `h3-prompt`。

## 开拍前合同（强制）

多镜、角色一致、台词或白模项目，先把 `show.json` + `canon.json` 落盘并通过校验，再提交任何 GPU 任务。单条测试可以只写镜头卡，但同样先写验收条件。

```powershell
python scripts/h3_show_init.py --dir projects/ep04 --project ep04
python scripts/h3_show_validate.py --spec projects/ep04/show.json --check-sha --strict
```

- `show.json` 是唯一调度源：交付规格、镜头 ID、prompt、route/template、length、seed、参考图/音频、首尾帧、白模版本、验收条件都写在里面；聊天记录不算。
- `canon.json` 是身份块、VOICE 块和参考包的唯一正本；每镜逐字复用，不重新发明同义词。
- `0 errors` 才允许提交；`--strict` 下 warnings 也算失败。校验器会抓 481 帧、Ref2VA 混首尾帧、Set C 传 A/B、缺文件/缺 canon、模板与路线不匹配、白模未走 Ref2VA、参考包哈希漂移。
- 规则全文见 [`references/show-contract.md`](references/show-contract.md)；示例见 `assets/show.example.json`、`assets/canon.example.json`。
- 新建项目用 `h3_show_init.py`；它只建骨架，不伪造提示词或参考图。真实 prompt、参考包和 `refs.sha256` 仍要人写/人审。
- 多镜批量执行用 `scripts/h3_showrunner.py --spec ...`：一次校验、逐镜 `--queue-only`、逐镜验证、`--resume` 续跑；单镜 `h3_generate.py --show-spec` 自动套同一门禁。

## 前置

- 医生报告缓存不超过 30 分钟，且 ComfyUI 在 `http://127.0.0.1:8188` 健康；已健康就复用，不重启、不重新发现工作流。
- 不传 `--comfyui` 时，解析顺序是 `H3LITE_COMFYUI` → `~/.h3lite/config.json` → 输出目录的父级（`output/`）→ 当前目录/`~/ComfyUI`。新任务先看 `comfyui_source`；解析到现有安装就不要引导用户重装。
- 同一配置的 manifest 处于 `submitting`/`queued`/`running` 时**不得重复提交**；返回已有 prompt ID。只有用户明确要求第二条相同任务时才 `--allow-duplicate`。
- 不在热路径里查 Git、模型哈希、依赖版本或官方文档；只有上一条命令给出具体错误指向那里时才查。
- **RTX 50 系 + W4A8/INT8**：脚本会在提交前打印 Blackwell 量化内核警告（`runtime_warnings`）。先按提示跑两条不同提示词的 640x352/4 步对比；文本无影响时只换 DiT 为 `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`，不要继续长渲染。

## 一条命令的默认路径

已配置环境上的普通生成只跑一次 fastpath：

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --prompt-text "<定稿提示词>" `
  --resolution 640x352 `
  --video-seconds 5 `
  --filename-prefix video/H3Lite_my_clip `
  --json
```

- 不带帧参数 = T2VA；`--mode auto` 会按传入的帧参数推断模式。
- I2VA：`--mode i2va --first-frame <png>`。
- FL2VA：`--mode fl2va --first-frame <png> --last-frame <png>`。
- L2VA：`--mode l2va --last-frame <png>`（会移除 I2V 模板的首帧占位）。
- 已确认的分辨率只提醒一次风险，预检 `blocked` 才停。
- 提示词里要求“完全静音”才移除音频；`不要对白` 仍保留环境音与音频流。

fastpath 复用医生缓存、内联规划/预检、提交、一次有界 watch，并把计划与提示词写进 run 目录。终端出现运行中的 cell 就等它，不要再开第二个监控。

## 帧数与画幅约定

| 用途 | `--length` | 实际 | 备注 |
|---|---|---|---|
| 快切短剧 | 72 | 73 帧 ≈ 3.042 s | 不硬裁成 3.000 s |
| 文戏常规 | 124 | ≈ 5.167 s | 默认 |
| 动作单元 | 124 | ≈ 5.167 s | 动作戏默认；5–7 个主要事件 |
| 中段 | 243 | ≈ 10.1 s | 链式常用；动作 9–13 事件 |
| 10 秒单条多镜 | 244 | 260 帧 ≈ 10.833 s | Set C FL2VA 实测自然补 16 帧，不裁不重跑 |
| 单段上限 | 362 | ≈ 15.1 s | 竖屏实测约 12 min；动作 13–18 事件 |
| 禁止 | 481 | — | 本机曾 48 W 自旋死锁，必须重启服务 |

- 24 fps，低显存基线 640x352、4 步、`res_multistep` + `simple`。
- H3 合法帧数满足 `n % 17 == 5`；107 帧、56 帧等诊断段按真实帧数记录，不硬裁成整秒。
- 0.4 MP 官方风画布（16:9 → 864x480）只在 planner 批准或用户明确要求时用。
- 32 像素对齐是 VAE 16 像素空间降采样与 DiT 2 像素 patch 的共同约束；不要承诺“720p”等消费标签，报告真实画布。
- 帧数校验允许向上最多 16 帧（约 8%）的自然补帧；短于请求超过 2 帧仍判失败。244→260 是正常补帧，状态应报 success。

## 模板与路线

Set C（16 GB，2026-11-19 验证）是当前速度/质量默认：

| 模板 | 用途 | 画布 | 实测 |
|---|---|---|---|
| `h3_c_speed640` | T2VA 快速 | 640x352 | 约 23 s |
| `h3_c_t2va768` | T2VA 768p | 1344x768 | 约 129 s |
| `h3_c_quality864` | T2VA 质量 | 864x480 | 约 42 s |
| `h3_c_i2va768` | I2VA 单首帧 | 1344x768 | 约 145–160 s |
| `h3_c_fl2va768` | FL2VA 首尾帧 | 1344x768 | 约 152 s |
| `h3_c_ref2va768` | Ref2VA 人像 | 1344x768 | 约 161 s |

- 六个模板是**自钉**的：`--component-set auto` 或不带 `--resolve-models`；**绝不传 `--component-set A/B`**。
- **视频 VAE 接缝**：Set C 默认 `minimax_h3_sol_vae_int8_convrot` 可能在 128 px 边界留下固定网格；最终交付前用 `h3-post/scripts/h3_grid_check.py` 检查。若 `suspected`，保持其他变量不变、只加 `--video-vae minimax_h3_video_vae_fp16.safetensors` 做 A/B；不要换 Sol fp16（同接缝）。详见 `h3-workstation/references/component-sets.md`。
- **16 GB 自动选 C（0.1.1）**：默认 `--workflow-template auto` + `--component-set auto`、未加 `--acceleration compat` 时，fastpath 在 ≥15.5 GB GPU 上按实测画布自动选 C——T2VA 640×352→`h3_c_speed640`、864×480→`h3_c_quality864`、1344×768→`h3_c_t2va768`；I2VA/FL2VA/Ref2VA 1344×768→各自模板（竖屏同理）。缺任一模型/节点类、L2VA、未实测画布（如 960×544、864 I2VA），或显式 A/B/模板/compat 时回退注册的 A/B 路线。自动选 C 不下载、不改组件集；`--dry-run` 的 `workflow_template_selection` 与 `set_c_auto.reason` 写明本次为何选或不选 C。
- 需要 Set C 启动参数 `--fast fp16_accumulation fp8_matrix_mult cublas_ops autotune` 才有上表耗时。
- **768p 文生视频用 `h3_c_t2va768`，不要用 `h3_w4a8_t2v`**：同画布 1344×768 / 124 帧 / 4 步实测 128.8 s vs 185.1 s（快约 30%），且不需要 Triton/Sage。它与 speed640 同一套 turbo v1.0 + Sol INT8 VAE 栈，但 768p 下原生稀疏必须 `start_percent=0.0`；沿用 speed640 的 0.2 会让首个 768p 稠密步多花约 70 s（实测 198.4 s）。
- `h3_w4a8_t2v` 保留给低显存 `_compat` 兜底与交叉验证；768p T2VA 的推荐路线是 `h3_c_t2va768`。
- **单首帧一律用 `h3_c_i2va768`，不要用 `h3_w4a8_i2v_compat` 跑 768p**：后者没有稀疏注意力、用 stock fp16 VAE，本机同画布同帧数实测 369 s vs 145 s（约 2.5×）。`h3_c_i2va768` 就是 `h3_c_fl2va768` 去掉尾帧锚点，其余（Pika lite LoRA、Sol INT8 VAE、原生 BlockSparse sol-attn、SigmaShift 6/3）逐项相同。
- 在 `h3_c_i2va768` 上再传 `--last-frame`，脚本会自动补 `LoadImage` 与 `last_frame` 输入，等价于 FL2VA；不需要换模板。
- Ref2VA 用 `--ref-image` 按 `<Picture 1>` 顺序重复；不要与 `--first-frame`/`--last-frame` 混用。参考图是真实条件输入，不是风格附件。
- 低显存 Set A/B：`h3_w4a8_t2v`、`h3_w4a8_i2v`、`h3_w4a8_ref2va` 及对应 `_compat`。`v4` turbo 在加速图（含 Set C 稀疏）上 **<6 步被护栏拒绝**（4 步重影/拖影），≥6 步放行（本机 6/8 步稀疏与稠密均干净）；4 步想用 v4 就走 `_compat`。
- 路线选择：人像/身份优先 Ref2VA；精确首尾构图用 FL2VA；单首帧 I2VA；其余 T2VA。详见 `references/face-quality.md`。

## 参考音频（Ref2VA 音色参考）

人物声音一致性**先做满提示词层**：VOICE 块逐字复用 + 选角声学距离（同场同性别说话人 F0 目标带差 ≥25–30 Hz）。方法见 [`../h3-prompt/references/voice-consistency.md`](../h3-prompt/references/voice-consistency.md)。仍漂时才上音色参考，且只有 Ref2VA 图有音频输入——I2VA/FL2VA 上传 `--ref-audio` 会被拒绝。

```powershell
python scripts/h3_generate.py `
  --workflow-template h3_c_ref2va768 `
  --prompt-file prompts/shot03.txt `
  --ref-image <identity.png> `
  --ref-audio <voice_A_dry.wav> `
  --comfyui <ComfyUI-path> --output-dir <ComfyUI>\output `
  --resolution 1344x768 --length 124 `
  --component-set auto --watch --json
```

- `--ref-audio` 可重复，按 `<Audio 1>` `<Audio 2>` 顺序绑定到 `ref_audios.ref_audio_0/1/2`；脚本自动新增 `LoadAudio` 节点，并把样本按 sha256 命名 staged 进 ComfyUI `input/`，不覆盖用户文件。`--dry-run` 可先看绑定结果。
- 上限（官方文档 + 本机 `object_info` 实测一致）：`ref_images` ≤9、`ref_audios` ≤3、`ref_videos` ≤3、`ref_video_audios` ≤3；每条音频 2–15 s、总 ≤15 s；混合文件 ≤12；**音频不能是唯一输入**，必须至少有一张 `--ref-image`。
- `audio_vae` 必须接上（Set C `h3_c_ref2va768` 已接）。节点 tooltip 原文：没有它，参考音频只条件化文本编码器，不会进入音频分支。
- 样本要求：单人、干声、无环境底噪、5–8 s、与目标同语言同情绪。从成片里截的带环境声样本会引入开头抢话、台词重复、首音节被切（MiniMax-H3 issue #95）。
- **不要同时接"续接视频自带音轨"和"独立音色参考"**：ComfyUI issue #16289 会直接崩 `RuntimeError: shape mismatch: value tensor of shape [938, 32] cannot be broadcast to indexing result of shape [1010, 32]`（`comfy/ldm/minimax/model.py:711`），两者单独都正常。二选一。
- A→B→A：给第三次发声**新槽**（同一文件再传一次 `--ref-audio` → `<Audio 3>`），提示词写 `S3 using Audio 3`；写 `S1/S2/S1` 会错人。
- 提示词里显式写 `<Audio 1> is only this character voice timbre; do not copy its words.`，音色参考用 `reference` 关系标记，不要用 `fully_copy`。
- 接了参考音频仍漂时按 #95 的 A/B/C 对照诊断：A 接音频 / B 同提示词同 seed 断开音频 / C 在 B 基础上再删悬空 `<Audio 1>` 文本。实用结论可能是断开音频（代价：一致性更难），不要盲目重试同一配置。
- 时长敏感：同一提示词可能 12.5 s 失败、14 s 成功。重试**只改时长 ±1–2 s**，seed/步数/提示词/参考全锁；跑通的配置原样记录成基线。

## 参考视频（Ref2VA 运动/运镜参考）

白模/灰模视频可以直接作为 `<Video 1>`，不需要先转 depth/pose。`--ref-video` 可重复（≤3），脚本自动新建 `LoadVideo → GetVideoComponents`，绑定到 `ref_videos.ref_video_0/1/2`；文件按 sha256 staged 到 ComfyUI `input/`。参考视频不要带音轨；身份仍由 `--ref-image` 的 `<Picture 1>` 负责。

```powershell
python scripts/h3_generate.py `
  --workflow-template h3_c_ref2va768 `
  --prompt-file prompts/shot05.txt `
  --ref-image <face.png> --ref-image <body.png> `
  --ref-video <white_proxy_12fps_864x480.mp4> `
  --comfyui <ComfyUI-path> --output-dir <ComfyUI>\output `
  --resolution 1344x768 --length 244 `
  --component-set auto --watch --json
```

- 代理规格：12 fps / 864×480 / SAR 1:1 / 无音轨；白模母版另存 24 fps 审片。
- 提示词写清视频参考只负责动作、走位、相机和接触点，不复制灰模材质/球关节/场景；身份绑定原图。
- `--ref-video` 只对 `MiniMaxH3ReferenceToVideo` 工作流有效；I2VA/FL2VA 会拒绝。
- `--dry-run` 可看自动新增的 LoadVideo/GetVideoComponents 节点。

## 双角色复刻（Preprocessor Map + Regional）

保留原片场景、机位和表演轨迹并替换两位角色时，使用 Character Swap Preprocessor Map + DWPose + SAM subject map + `BFSH3RegionalReferenceAttention`，不要套用单角色 `body_swap`。

- 默认 Subject 1/红/Picture 1、Subject 2/蓝/Picture 2；实际绑定以工作流和提示词为准。
- subject map 必须使用 BFSNodes 精确 hybrid 色 `(235,40,40)` / `(35,110,235)`，重采样用 nearest-neighbor。
- 后段 mask 消失不等于失败；逐帧比较接受源与候选的检测色序列，双人持续可见由 prompt + Regional propagation 保证。
- soft bias 必须同时接 `guide` 与 `subject_map`；日志应出现 `routing: soft` 和 `tracked 2 subjects`。
- 从已验收图克隆，只允许修改输入文件、输出几何和文件名节点；用脚本验证 changed nodes、图连接、几何和引用 SHA。
- 分辨率按内容 → 显存 → 生产画布顺序升级；每次只改几何，保留 prompt、参考、seed、采样和 Regional 参数。

```bash
python scripts/h3_validate_subject_map.py --map map.mp4 --reference accepted_map.mp4 --guide guide.mp4 --allowed-colors 0,1 --json qa/map.json
python scripts/h3_prepare_regional_workflow.py --base accepted.json --plan regional_plan.json --output candidate.json --report qa/workflow.json
```

完整合同、plan schema、颜色门禁和验收证据见 [`references/dual-character-remake.md`](references/dual-character-remake.md)。显存优化与 allocator 警告分类见 [`../h3-workstation/references/regional-attention-memory.md`](../h3-workstation/references/regional-attention-memory.md)。

## 文戏单次多镜（10 秒 3–4 拍）

约 10 秒、3–4 拍、首尾构图要锁时，直接用 `h3_c_fl2va768` **一条生成**，不要逐镜生成后拼。这是文戏/叙事默认；动作戏先看下一节的 9–13 个事件门槛。提示词按 `h3-prompt` 的 `[Shot N]` 写法，切点写 `This is a hard cut.`。

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --mode fl2va `
  --first-frame <first.png> --last-frame <last.png> `
  --workflow-template h3_c_fl2va768 `
  --component-set auto `
  --resolution 768x1344 --video-seconds 10 --seed 20261120 `
  --filename-prefix video/H3_multi_shot --json
```

- Set C 图的 `--component-set` 必须是 `auto`；fastpath 会以 auto 提交，绝不把 auto 归一成 C 后再传给自钉模板。
- 切点数是目标，不是合同。模型可能把相邻两拍合并为一次运镜；只要成片仍有 3–4 个清楚视觉段、尾帧落到 `<Picture 2>`，按成功验收。
- 不为“少一个硬切”重跑；只有整条没有形成新构图或首尾锚点明显漂移时才重跑。

低层提交（自定义图、Ref2VA、Set C）：

```powershell
python scripts/h3_generate.py `
  --workflow-template h3_c_ref2va768 `
  --prompt-file prompts/shot03.txt `
  --ref-image <identity.png> `
  --comfyui <ComfyUI-path> --output-dir <ComfyUI>\output `
  --resolution 1344x768 --length 124 `
  --component-set auto --watch --json
```

`--workflow <file>` 仍支持自定义 API 图；`--queue-only` 只在另一个进程负责监控时用。不要用短超时的同步等待命令。

## 动作戏：5 秒单元与测试阶梯

动作戏默认 5 秒（124 帧）一个生成单元，5–7 个主要动作事件、1–3 个视觉段。不要先做 10 秒再靠后期剪快；先按 [`../h3-director/references/action-combat.md`](../h3-director/references/action-combat.md) 编好因果链。

```powershell
python scripts/h3_fastpath.py `
  --comfyui <ComfyUI-path> `
  --mode fl2va `
  --first-frame <action_loaded_first.png> --last-frame <mid_action_last.png> `
  --workflow-template h3_c_fl2va768 `
  --component-set auto `
  --resolution 768x1344 --video-seconds 5 --seed 20261120 `
  --filename-prefix video/H3_action_unit --json
```

- 首帧要“动作已上膛”，尾帧选动作中段/命中后位移；不选中立站姿或收势定式。
- 加载 Wushu 动作 LoRA 时可在提示词开头加 `wushu_action`；没有加载 LoRA 时不要写触发词。
- 当前 Set C API 图只带固定 Pika turbo LoRA，**没有动作 LoRA 槽位**。加 Combat/Motion/Wushu 必须用自定义 API 图走 `h3_generate.py --workflow`，不要假设 fastpath 会自动挂载。
- 同一条连续打斗不要用 3 个以上 `MiniMaxH3AddGuide` 动作相位锚点；本机 E1 已证明会钉帧、硬切和定格。

测试阶梯（每次只改一个变量，保留同一提示词/种子做 A/B）：

1. **纯提示词 5 秒**：先修事件密度、因果链、具体招式和实时速度；这是基线，不加 LoRA、不加锚点。
2. **提示词修正**：仍然慢时，指出是哪一拍慢，补招/补受击/删复位；不要重跑同一文本。
3. **Motion Continuity Repair 0.9**：自定义图；先做权重 0 与 1 的对比，确认 LoRA 真的生效。约 155 MB。
4. **Combat Base V2 0.9 + Motion 0.5–0.7，两段采样 25%**：采样器用 `res_multistep` + `simple` 或 `euler` + `beta`；二采 Motion 0.2–0.3。Combat 约 155 MB。
5. **PDD Acc 8 步**：仅在前述路线仍因采样软/慢时尝试；euler、CFG 1、SigmaShift 12/3，移除其他蒸馏 LoRA。约 1.37 GB，模型卡称支持 pruned/W4A8，本机未实测。
6. **Wushu 全量路线**：需要 34 GB 全量 int8 底座 + 32B bf16 文本编码器，不是 16 GB W4A8 + 4B TE 的 drop-in；不要当轻量补丁下载。

动作 LoRA 的 W4A8 兼容性未在本机验证；先做权重 0/1 对比。任何大体积组件下载前先征得用户同意。

## 监控

- Windows 桌面默认打开原生 Tkinter 进度窗；`--no-monitor-gui` 关闭，`--monitor-gui` 强制开。窗口只监控，关闭不会中断生成。
- 节点数是结构进度，不是时间百分比：H3 各节点耗时差异极大。报告时区分 `progress_basis`（node_completion / sampling_steps）与 `eta_basis`（empirical / live_progress），不要把 4/5 当成 80% 时间。
- 独立查看：

```powershell
python scripts/h3_status.py --base-url http://127.0.0.1:8188 `
  --prompt-id <id> --comfyui <ComfyUI-path> --output-dir <ComfyUI>\output `
  --run-root <ComfyUI>\user\h3lite_runs --require-audio --dynamic-check `
  --anchor-check --compact --json
```

有界等待加 `--watch --watch-interval 20 --watch-timeout 3600`。默认紧凑输出；只有失败诊断才 `--verbose` 拿完整 history。

## 批量 showrunner（show.json）

`h3_showrunner.py` 把 `show.json` 变成一次批处理：校验一次、逐镜 `--queue-only` 提交、逐镜 `h3_status.py` 验证，状态写进 `show_manifest.json`，支持断点续跑。

```powershell
python scripts/h3_showrunner.py --spec projects/ep04/show.json `
  --output-dir E:/MiniMax-H3/ComfyUI/output --comfyui E:/MiniMax-H3/ComfyUI `
  --strict --check-sha
```

- `--only s02,s03` 只跑指定镜头；`--resume` 跳过已完成、继续已提交未验证的镜头；`--force` 重跑。
- `--dry-run` 只校验并打印每镜将执行的命令，不占 GPU；`--no-watch` 只提交并记录 prompt_id。
- `--json` 输出机器可读汇总；默认 `show_manifest.json` 在项目根，`runs/` 放单镜 manifest。
- 白模镜头提供 `white_model.proxy` 后，showrunner 会作为 `--ref-video` 提交；需要同时传 `--comfyui` 以便 staged 代理。`white_model.source` 只用于审片/校验，不送 H3。
- 单条 `h3_generate.py` 加 `--show-spec <show.json>` 会在提交前自动跑合同校验；`--show-strict` 把 warning 也当失败。showrunner 已校验过，单镜调用无需重复。

## 验证

不能凭 queue ID 报成功。逐项确认：

1. 任务无执行错误；
2. 目标 MP4 存在，有视频流与音频流；
3. 时长、帧数、FPS 接近请求值；帧数多出不超过 16 帧（约 8%）视为自然补帧，不重跑、不裁尾；短于请求超过 2 帧仍判失败；
4. 首/中/末帧非黑、非纯色，未被判 `suspected_mosaic`；`dynamic` 只是首/中/末 3 帧的烟测，不能证明动作密度。动作片另抽 8–12 帧或接触表，5 秒至少 5 个主要动作事件、没有 >0.5 秒定格；近静止（帧间 MAD < 1.0）超过 20% 判慢放，回 `h3-prompt`/重编，不靠后期变速；
5. 身份、服装、道具、参考构图等验收条件在抽帧中保持一致；带台词的另**听**说话人是否正确、台词是否只说一次、音色是否与正典样本一致（跨镜漂移交给 `h3-post` 的 `scripts/h3_voice_audit.py` 量化，不要凭"有音频流"判过）；
6. 单次多镜只核对请求切点附近（±0.5 s），不做全片阈值扫；请求切点变成连续运镜但仍有清楚新景别/机位，按 3–4 个视觉段验收；
7. 记录输出路径、实际耗时、配置指纹与 run manifest；动作片另记实际事件数、近静止比例、是否用了两段采样/LoRA；
8. 双角色复刻额外记录 subject-map 颜色序列、Regional 接线日志、changed-node 报告、A/B crop、持续双人可见和未 exercised 的角度；
9. 计划估计与实际执行时间对比；低显存首次运行可能因内核编译慢很多，这是经验值不是承诺。

`ffprobe_not_found` 只说明验证器路径缺失：保留已生成文件，设置 `H3LITE_FFPROBE` 或在 `<ComfyUI>` 附近找 `ffprobe.exe`；不要因为元数据不可读就重跑模型。动态 QA 用的 `ffmpeg` 同理，可设 `H3LITE_FFMPEG`；脚本也会在 `<ComfyUI>` 根目录与 `ffmpeg/bin`、`tools/ffmpeg/bin` 下查找，不要把“找不到 ffmpeg”当成视频失败。

## 失败排查顺序

1. 页面文件/RAM 极低，或其他 GPU 进程；
2. 缺模型或缺自定义节点；
3. 模型目录/文件名错误；
4. CUDA/PyTorch/自定义内核不兼容；
5. OOM 或过量 CPU offload；
6. H3 flow/音频图无效；
7. 提示词或参考对齐问题。

- 黑屏/马赛克：恢复官方 H3 flow/sigma-shift 节点，简化可选 attention/cache 补丁。
- 50 系固定纹理 / 换提示词不变 / 数值爆炸：这是 Blackwell + W4A8/INT8 convrot 的上游内核问题，不是提示词问题。按 `h3-workstation/references/deployment-matrix.md` 的文本影响测试处理；绕行只换 DiT 到 fp8_scaled。
- 4 步音频失真：检查视频/音频 schedule，必要时用兼容 Turbo 采样器。
- 同尺寸损坏的 W4A8 会产生彩色马赛克：按完整性记录重验哈希，不要只看字节数。
- 第二次运行更快不等于复用了结果：权重常驻/上游节点缓存会加速，但改提示词或种子仍会重新采样。
- 后台 ComfyUI 抛 `OSError: [Errno 22] Invalid argument`：用持久日志重启，不重下模型。

## 边界

- 不改生成图、不加采样步数、不加第二个生成模型、不做二次推理“增强”。人脸质量靠路线 + 提示词 + 分辨率。
- 放大、拼接、字幕、响度、交付都交给 `h3-post`。
- 运行历史清理是维护，交给 `h3-workstation`。

## 资源

- `scripts/h3_fastpath.py`、`h3_generate.py`、`h3_status.py`、`h3_anchor.py`、`h3_monitor_gui.py`、`h3_plan.py`、`h3_paths.py`、`h3_doctor.py`、`h3_preflight.py`；`h3_show_init.py`、`h3_show_validate.py` 负责开拍前骨架与门禁。`h3_doctor.py` 与 `h3_preflight.py` 各有一份副本在 `h3-workstation/scripts/`，必须同步更新。
- `assets/`：Set A/B/C 的 API 工作流（T2VA/I2VA/Ref2VA + compat + Set C 六模板）。脚本按 `Path(__file__).parents[1] / "assets"` 解析，不要再套子目录。
- `references/show-contract.md`：开拍前合同全文（交付、canon、镜头卡、参考包、预览阶梯、版本锁定、QA 门禁、返工/停止条件）。
- `assets/show.example.json`、`assets/canon.example.json`：可直接复制修改的 spec 模板。
- `references/face-quality.md`：人像路线 A/B 与护栏。
- `references/final-lite-canvas.md`：1184×672 / 1152×640 中间画布的三臂 A/B 协议、验收与磁盘边界；未验证前不进 auto 路由。
- `references/dual-character-remake.md`：Preprocessor Map + Regional 双角色复刻、hybrid 色合同、工作流最小变更、分辨率阶梯与分级验收。
- `scripts/h3_validate_subject_map.py`：逐帧校验 BFSNodes hybrid 色与接受源颜色序列。
- `scripts/h3_prepare_regional_workflow.py`：按 plan 克隆工作流并验证 changed nodes、连接、几何和引用 SHA。
- [`../h3-prompt/references/voice-consistency.md`](../h3-prompt/references/voice-consistency.md)：声音圣经、选角声学距离、参考音频上限与已知坑（#95 / #16289）、验收门限。
- [`../h3-director/references/action-combat.md`](../h3-director/references/action-combat.md)：动作密度、5 秒单元、慢放诊断、动作 LoRA 兼容性与测试阶梯。


---

# ISSUES / PRs of Rimagination/h3lite


### #20 [open] ISSUE: H3 W4A8/INT8 checkpoints output fixed texture & ignore text on RTX 5060 Ti (sm_120) — evidence attached
URL: https://github.com/Rimagination/h3lite/issues/20  by teemaker-dev 2026-09-07T12:17:47Z

# H3 (MiniMax) on RTX 5060 Ti (sm_120): text conditioning ignored + numerical blowup in DiT blocks

## Environment
- GPU: RTX 5060 Ti 16GB (sm_120 Blackwell)
- OS: Windows 11 + WSL2
- ComfyUI 0.34.2 (Windows native)
- torch 2.12.0+cu130 (also reproduced on 2.9.1+cu130)
- comfy-kitchen 0.2.31

## Models
- `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` (12.5GB, W4A8)
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20.9GB, INT8)

## Setup
krea2 CLIP (`qwen3vl_4b_fp8_scaled`) + `mmh3-4b-ClipProj-celeb-mlp` projection, BasicGuider + BasicScheduler(simple, 4 steps) + SamplerCustomAdvanced + MiniMaxH3SigmaShift(12, 3), turbo 4-step LoRA, 768x432x124 frames.

## Symptoms
1. Output is a fixed woven texture pattern, not the prompted scene. Both "mountain lake" and "sports car" prompts yield **byte-identical frames** (MD5 match at frames 0/30/60/90).
2. Different prompts → identical output, **even though the context tensors differ** (ctxsum -5617 vs -3591). Conditioning injection path verified working.
3. W4A8 and INT8 checkpoints → **identical output frames** (same MD5), suggesting the model is not actually driven by its weights.
4. **Numerical blowup**: after the 84 DiT blocks, text rows reach std ≈ 32256 (input std ≈ 2.26), video rows std ≈ 6944. Weights load correctly (block0 qkv_proj std ≈ 0.084, condition_proj 5376x5120).

## Debug trail (all verified with instrumentation)
- On torch 2.9.1 the model degraded to an **identity mapping** (DiT block outputs ≈ 0, latent never updated — sampling "ran" but output was the empty-latent VAE decode). Upgrading to torch 2.12.0 fixed this: the model now produces real dynamic output, but still ignores text.
- Text conditioning verified: layout segments `[(0,20,'text'), (20,434,'audio'), (434,12866,'video')]`, text_states (20,5376) injected at h[0:20] correctly, context content differs per prompt.
- Standalone kernel tests pass: `rms_rope_split_half_` (eager backend), torch SDPA, int8_linear (rel diff 0.18% vs eager).
- Zeroing the context produces byte-identical output → text has **zero** influence on generation.
- SDXL generation works fine on the same machine (does not use comfy-kitchen quantized ops).

## Related
- #120 (same GPU family, GPU LOST under comfy-kitchen)
- #55 (INT8 corrupted outputs on SM75)

Any hints? Is this a known incompatibility between Blackwell (sm_120) and comfy-kitchen quantized ops inside the H3 DiT? Happy to run any additional experiments or provide logs.


---

Additional context for the h3lite repo: the same fixed-texture output and zero text influence is observed with both the W4A8 (`minimax_h3_fl2va_pruned_w4a8_mixed`) and INT8 (`minimax_h3_fl2va_pruned_int8_convrot`) checkpoints on this hardware. Happy to provide logs, MD5s, or run any proposed fix.


> **Comment by Rimagination (2026-11-25T16:18:51Z):**

Thanks for the detailed evidence — this matches the upstream reports and is not an H3Lite prompt/workflow bug.

Key confirmation from the cross-post in Comfy-Org/comfy-kitchen#163: upgrading comfy-kitchen to 0.2.38 did **not** fix the int8_convrot path on the 5060 Ti, but switching **only the DiT** to `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` (same workflow, same INT8 text encoder, same VAE, same resolution) produced clean output. An RTX 5090 did not reproduce the failure, so it is specific to the 16 GB / sm_120 / convrot kernel path rather than all Blackwell cards.

Workaround to verify on your machine:

1. Keep the current workflow and seed; change only `UNETLoader.unet_name` to `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`.
2. Run 640x352, 4 steps, with two clearly different prompts (mountain lake vs sports car).
3. Confirm the outputs are not byte-identical and that text actually changes the scene. If they are clean, the issue is the quantized DiT kernel path.

I have added an H3Lite-side guard so this does not silently burn a long render: `h3_doctor.py` now reports `blackwell_quantized.status=risk` when it sees an RTX 50-series GPU plus a `w4a8/int8/convrot` DiT, `h3_preflight.py` carries the warning into `checks.blackwell_quantized`, and `h3_generate.py` prints the same warning before submitting. It deliberately warns rather than blocks, because the root cause is upstream. The deployment notes now include the two-prompt test and the fp8-DiT single-variable workaround.

I will keep this issue open as an upstream tracking item until comfy-kitchen has a confirmed fix. If you can post the result of the fp8-DiT test (and your comfy-kitchen/Torch versions), that would help confirm the scope.

<!-- h3lite-triage 2026-11-26 -->



### #19 [open] ISSUE: WorkBuddy切换到新任务提示ComfyUI 未安装
URL: https://github.com/Rimagination/h3lite/issues/19  by ohyes666 2026-08-21T08:13:45Z

谢谢大佬开源的工具
我按照文档下载资源，用workbuddy安装h3lite后 成功生成小球视频。
我打开一个新任务，我上传了一张图片，然后输入提示词 
“请使用 H3 Lite，将我在这条消息中附上的图片作为视频 0 秒的第一帧，生成一个 5 秒横屏视频。xxxxx”
之后  workbuddy提示我未安装ComfyUI ，要给我下载一个11G到15G的资源包之后才能生成视频
请问如何解决

> **Comment by Rimagination (2026-11-25T16:17:34Z):**

This looks like the new task losing the machine-level ComfyUI path, not ComfyUI actually being missing.

Two checks before accepting any 11–15 GB download:

1. In the new task, run the H3Lite doctor against the existing install:

```powershell
python skills/h3-workstation/scripts/h3_doctor.py --json --comfyui "<your existing ComfyUI>"
```

If it reports `main.py: true` and the models are present, the install is fine — the task just did not inherit the path.

2. Persist the path once so later tasks do not depend on the previous task's workspace:

```powershell
python skills/h3-workstation/scripts/h3_doctor.py --comfyui "<your existing ComfyUI>" --write-config
```

This writes `%USERPROFILE%\.h3lite\config.json`. The resolution order is `--comfyui` > `H3LITE_COMFYUI` > that config file > `<project>/ComfyUI` > `~/ComfyUI`. If the app still asks to download ComfyUI before the skill can run, that part is the WorkBuddy installer's environment detection and needs a WorkBuddy-side bug report; the H3Lite config cannot override a prompt that appears before the skill executes.

I have added the config resolver and the "reuse an existing install, do not offer another 11–15 GB bundle" rule to the local plugin. If you can share the new task's `comfyui.source` value after the update, I can tell whether it is reading `config:`/`env:` or still falling back to `not-found`.

<!-- h3lite-triage 2026-11-26 -->



### #18 [closed] PR: Replace Set A route showcase with rolling red ball
URL: https://github.com/Rimagination/h3lite/pull/18  by Rimagination 2026-08-20T14:39:04Z

## Summary\n- regenerate all six Set A route comparison clips with the same rolling red-ball prompt\n- replace the six Pages MP4s and posters\n- update Chinese/English route descriptions and measured generation times\n\nAll six outputs were locally checked at 640x352, 124 frames, 24 fps, with native audio and dynamic/color QA passing.


### #17 [closed] PR: Move route comparison into hardware selection
URL: https://github.com/Rimagination/h3lite/pull/17  by Rimagination 2026-08-20T14:02:12Z

Move the six Set A acceleration comparisons under hardware and route selection in both READMEs. Remove the temporary existing-generated-cases gallery so the later case sections remain the place for future examples.


### #16 [closed] PR: Serve showcase videos from GitHub Pages
URL: https://github.com/Rimagination/h3lite/pull/16  by Rimagination 2026-08-20T13:48:51Z

Host the showcase MP4 files under docs/videos so the GitHub Pages player receives video/mp4 from the same origin. Keep the assets Release as a download backup and update the README wording.


### #15 [closed] PR: Route README video covers through GitHub Pages player
URL: https://github.com/Rimagination/h3lite/pull/15  by Rimagination 2026-08-20T13:41:56Z

Use the same poster-to-player pattern as MoneyPrinterTurbo. Point Chinese and English README examples to the H3 Lite GitHub Pages player, add the I2VA family-room entry, and track its poster. The player reads video files from the assets release.


### #14 [closed] PR: Polish README showcase and publish video posters
URL: https://github.com/Rimagination/h3lite/pull/14  by Rimagination 2026-08-20T13:26:19Z

Condense repeated README wording and remove defensive phrasing. Track ten showcase posters so GitHub images load. Replace plain play/download links with clickable video covers in both READMEs. Verification: 17 relative refs, 0 missing; ffprobe passed for ten posters; git diff check passed.


### #13 [closed] PR: Publish H3 Lite I2VA showcase and route safety updates
URL: https://github.com/Rimagination/h3lite/pull/13  by Rimagination 2026-08-20T13:08:51Z

## Summary

- replace the README I2VA ramen example with the 8-step sitcom living-room case
- add the first-frame and verified 864x480 native-audio MP4 example assets
- publish the local showcase page and route safety guidance for the disallowed v4 accelerated pairing

## Verification

- pytest: 72 passed
- MP4 verified: 864x480, 5.167 seconds, 24 fps, video and audio streams present
- git diff check passed


### #12 [closed] PR: docs: add common video resolution recommendations
URL: https://github.com/Rimagination/h3lite/pull/12  by Rimagination 2026-08-15T17:45:47Z

Adds a common video resolution table to README.md, including dimensions, MP values, aspect ratios, H3 usage guidance, and the 640x352 default. Validation: git diff --check passed; documentation-only change.


### #11 [closed] PR: Document validated RTX 3060 Ti 8GB hardware floor
URL: https://github.com/Rimagination/h3lite/pull/11  by Rimagination 2026-08-15T09:18:29Z

## What changed

- Document RTX 3060 Ti 8 GB VRAM + 16 GB system RAM as the lowest validated W4A8 configuration.
- Keep 32 GB system RAM as the recommended target and preserve hardware-specific caution for other 8 GB GPUs.
- Update doctor and preflight rules so 16 GB RAM is cautionary for the 8 GB route rather than an automatic blocker.
- Keep the 6 GB experimental route and below-16-GB RAM protection unchanged.
- Add regression coverage for the validated floor and caution behavior.

## Why

The previous rules treated less than 24 GB system RAM as a hard blocker, which contradicted a real successful run on an RTX 3060 Ti 8 GB system with 16 GB RAM.

## Validation

- `python -m unittest discover -s tests -p "test_*.py"`
- 51 tests passed.
- `git diff --check`


### #10 [closed] ISSUE: Testing on Lenovo Y7000P 2024 (RTX 4050 6GB, 32GB RAM)
URL: https://github.com/Rimagination/h3lite/issues/10  by ccch713 2026-08-15T09:05:03Z

## Testing in progress

Hi, I'm currently testing H3 Lite on a **Lenovo Y7000P 2024** laptop. Here's my hardware profile:

| Spec | Value |
|---|---|
| CPU | Intel Core i7-14650HX |
| GPU | NVIDIA GeForce RTX 4050 **6GB VRAM** |
| RAM | 32 GB |
| Storage | 1 TB SSD |
| OS | Windows |

This falls into the **experimental-6GB** profile per the deployment matrix. I'm working through the installation and will report back with timing data, component compatibility, and any issues encountered.

### Plan
- [ ] Install ComfyUI + H3 Lite skill
- [ ] Validate Set A (W4A8) component compatibility at 6GB
- [ ] Run the red-ball smoke test at 608×352
- [ ] Test T2VA and I2VA routes
- [ ] Report generation timing at baseline resolution

Will update this issue as testing progresses. Any suggestions for 6GB-specific tuning are welcome!

> **Comment by Rimagination (2026-08-15T09:33:10Z):**

Thanks for testing this configuration! It fits our experimental 6GB profile. Please share the installation result, generation time, and any OOM or compatibility errors when available. The 608×352 smoke test is a good starting point.


> **Comment by Rimagination (2026-11-25T16:17:49Z):**

This is a test plan rather than a defect, so I am closing it as stale to keep the issue list focused. For a 6 GB RTX 4050 with 32 GB RAM, the realistic starting point is the low-VRAM W4A8 route, not Set C (Set C assumes 16 GB):

- 608x352 or 640x352, 124 frames, 4 steps, one job at a time.
- `--lowvram` (or the launcher's low-VRAM mode); close other GPU apps.
- Windows pagefile at least 40 GB free; keep the ComfyUI process resident but do not run two H3 jobs.
- Expect CPU offload to dominate; compare against the 640x352 timing table before trying 864x480.
- Do not use Sol INT8/Set C as the first test; validate the baseline graph first.

If you hit a concrete failure, please open a new issue with the doctor/preflight JSON, exact command, ComfyUI/Torch versions, and the first error line; that is actionable and we can treat it as a bug.

<!-- h3lite-triage 2026-11-26 -->



### #9 [closed] PR: Improve H3 Lite routing and quality guidance
URL: https://github.com/Rimagination/h3lite/pull/9  by Rimagination 2026-08-14T16:21:59Z

## Summary

- Simplify the README to keep only the user entry point and minimum configuration.
- Move hardware, component, face-quality, and cinematic prompt guidance into references.
- Add zero-inference-cost constraints for face routing and post-generation QA.
- Add Blackwell `sm_120` Torch compatibility warnings during doctor checks.
- Reuse an unambiguous workflow-variant timing history in planning.
- Add targeted tests for timing calibration and RTX 50-series compatibility.

## Validation

- Targeted unittest suites pass in the installed Skill and repository copies.
- Modified Python files pass `py_compile`.
- No video inference graph, sampling count, or generation model was added.


### #8 [closed] PR: Optimize README hero animation
URL: https://github.com/Rimagination/h3lite/pull/8  by Rimagination 2026-08-14T13:58:47Z

## Summary

- optimize the animated README hero without changing its content or file path
- reduce the asset from 6,408,865 bytes to 3,067,817 bytes (about 52%)
- retain a 5.6-second loop at 720x258 and 8 FPS

## Impact

The project landing page downloads substantially less data while preserving the animated title, layout, and readable typography.

## Validation

- FFprobe confirmed a readable 45-frame GIF at 720x258 with a 5.63-second duration
- visually inspected a middle frame against the original
- `git diff --check` passed

Documentation asset only; no code tests or video generation were run.



### #7 [closed] PR: Fix bundled ffprobe verification
URL: https://github.com/Rimagination/h3lite/pull/7  by Rimagination 2026-08-14T13:52:47Z

## Summary

- find `ffprobe` from system `PATH`, `H3LITE_FFPROBE`, or bounded locations in and beside the selected ComfyUI root
- propagate the ComfyUI root through direct generation, `--watch`, and standalone `h3_status.py` verification
- infer the ComfyUI root when standalone status receives exactly `<ComfyUI>/output`
- preserve actionable verifier errors in compact status output and document recovery without regenerating a completed video

## Root cause

Successful videos were reported as `verification_failed` when `ffprobe.exe` lived in the ComfyUI root but was absent from system `PATH`. The first proposed fix covered direct generation only; standalone status also needed the ComfyUI path.

## Validation

- 38 tests in `test_runtime_safety_and_qa` passed
- Python syntax checks passed for the three modified scripts
- skill quick validation passed
- `git diff --check` passed
- an external RTX 4060 Ti installation verified an existing 5.167-second, 124-frame video with video and audio streams through standalone status

No video was regenerated for this change.



### #6 [closed] PR: Reorganize Chinese and English README guides
URL: https://github.com/Rimagination/h3lite/pull/6  by Rimagination 2026-08-14T07:12:58Z

## Summary

- split the project guide into independent Chinese and English READMEs
- reorder both versions around the beginner journey: understand, install, validate, then explore advanced examples
- preserve the verified hardware matrix, bundled media examples, download guidance, references, and license

## Validation

- `python scripts/quick_validate.py`
- `git diff --check`
- checked local Markdown links and heading hierarchy in both README files

Documentation-only change; no video generation or full test suite was run.



### #5 [closed] PR: Document validated H3 integrity and performance
URL: https://github.com/Rimagination/h3lite/pull/5  by Rimagination 2026-08-13T18:49:40Z

## What changed

- document the validated RTX 4060 Ti 16 GB and RTX 4070 Laptop 8 GB Set B comparison
- record the exact Set B text-encoder checksum and explain first-use integrity caching
- make same-size checkpoint corruption and colored-mosaic rejection explicit
- keep README guidance concise in both Chinese and English

## Why

A same-size corrupted W4A8 checkpoint produced playable but unusable mosaic video, while a verified replacement produced coherent video. The controlled Set B runs also showed that `NORMAL_VRAM` on 16 GB avoids most of the dynamic loading/offload cost seen under `LOW_VRAM` on 8 GB.

## Validation

- `quick_validate.py`: passed
- `git diff --check`: passed
- runtime safety and QA unit tests were already passing on the unchanged implementation; they were not repeated for this documentation-only follow-up



### #4 [closed] PR: Document validated hardware configuration
URL: https://github.com/Rimagination/h3lite/pull/4  by Rimagination 2026-08-13T15:41:46Z

## What changed

- renames the hardware section to emphasize validated configuration evidence
- documents the MECHREVO Yilong15Pro, RTX 4070 Laptop 8 GB, Ryzen 7 8845H, 32 GB RAM, and Windows 11 setup
- records the validated W4A8/4B four-step route and generated output sizes
- distinguishes the evidence-backed reference machine from conservative guidance for other hardware

## Validation

- H3 Lite skill validation passed
- git diff whitespace check passed


### #3 [closed] PR: Add generated video examples
URL: https://github.com/Rimagination/h3lite/pull/3  by Rimagination 2026-08-13T15:22:05Z

## What changed

- adds generated MP4 outputs for the red-ball, golden-retriever, and starship examples
- links each video directly beneath its matching prompt in both the Chinese and English README sections

## Validation

- H3 Lite skill validation passed
- all three files were verified as H.264 video with AAC audio
- git diff whitespace check passed


### #2 [closed] PR: Add timeline prompt quick-start example
URL: https://github.com/Rimagination/h3lite/pull/2  by Rimagination 2026-08-13T15:10:43Z

## What changed

- keeps the original bouncing red ball quick-validation prompt
- adds a five-second golden retriever timeline prompt in Chinese and English
- explains that time-segmented prompts usually improve adherence for consecutive actions

## Validation

- H3 Lite skill validation passed
- git diff whitespace check passed


### #1 [closed] PR: Improve H3 Lite fast path and examples
URL: https://github.com/Rimagination/h3lite/pull/1  by Rimagination 2026-08-13T15:01:27Z

## What changed

- adds the single-command H3 fast path with cached diagnostics, planning, preflight, submission, monitoring, and output verification
- improves Windows path handling, component-set validation, duplicate-job semantics, audio policy, and hardware-aware resolution planning
- restructures the bilingual README around quick validation and structured T2VA, I2VA, and Ref2VA examples
- bundles the small official MiniMax example inputs with source URLs and checksums

## Why

This reduces non-generation overhead on configured machines while making first-time installation and reproducible examples clearer for non-technical users.

## Validation

- H3 Lite skill validation passed
- 31 unit tests passed
- git diff whitespace check passed
- bundled media formats, dimensions, and hashes verified
