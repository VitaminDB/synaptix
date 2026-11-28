# FlashML-org__FreeVideo

Source: https://github.com/FlashML-org/FreeVideo
Fetched: 2026-11-28

Description: Make videos on the computer you already own. FreeVideo runs MiniMax H3 in as little as 8 GB of VRAM and 16 GB of RAM, and adapts its acceleration path to your hardware.
Stars: 5193  Updated: 2026-11-27T23:34:26Z



---

## FILE: README.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/README.md)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="web/assets/freevideo.svg">
    <img alt="FreeVideo" src="web/assets/freevideo-light.svg" width="65%">
  </picture>
</div>

<p align="center">
| <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo.exe"><b>Windows</b></a> | <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg"><b>macOS</b></a> | <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage"><b>Linux</b></a> | <a href="docs/community-results.md"><b>Community Results</b></a> | <a href="https://freevideo-community.pages.dev/#gallery"><b>Gallery</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Discord</b></a> | <a href="https://freevideo-community.pages.dev/qq"><b>QQ Group</b></a> | <a href="https://freevideo-community.pages.dev/wechat"><b>WeChat Group</b></a> |
</p>

<p align="center">Requirements: an NVIDIA RTX 30 series or newer graphics card (professional cards of the same generation included; 8 GB of graphics memory or more recommended) and 16 GB of memory, or a Mac with Apple silicon (preview). GTX 10, GTX 16 and RTX 20 series cards and AMD or Intel graphics cannot run FreeVideo.</p>

<p align="center">English · <a href="README.zh-CN.md">中文</a></p>

Make videos on the computer you already own. Powered by [Video DeltaNet (VDN)](https://openvdn.github.io/), FreeVideo runs MiniMax H3 on NVIDIA graphics cards with compute capability 8.0 or higher (such as the RTX 30 series or newer, professional cards of the same generation included), with 8 GB of VRAM or more recommended and 16 GB of RAM, and with acceleration adapted to your&nbsp;hardware. Apple silicon Macs run a preview.

https://github.com/user-attachments/assets/ecda7d0d-7fbe-4e0c-8c29-8f3315bafc15

<p align="center"><a href="https://freevideo-community.pages.dev/#gallery">More clips and the four quality levels side by side in the gallery →</a></p>

## News

- **2026-11-10** · **[v0.5.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.5.0): H3 Image.** Pictures up to 16 MP locally, from 8 GB of VRAM.
- **2026-10-13** · **[v0.4.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.4.0): AI agents, minute-long videos and an editing view.** Let an AI agent such as Claude Code or Codex make videos on its own, generate videos over a minute long, and edit them on a timeline.
- **2026-10-07** · **[v0.3.2](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.3.2): preview first.** See a half-resolution preview first and finish only the videos you like.
- **2026-10-07** · **[v0.3.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.3.0): a faster int8 model.** GeForce and RTX 30-series cards switch to int8; an RTX 3090 makes a 5-second video about 2.3x faster.
- **2026-10-06** · **[Gallery](https://freevideo-community.pages.dev/#gallery) is live.** Watch 20-second clips made with FreeVideo, and the four quality levels side by side.
- **2026-10-06** · **[v0.2.3](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.2.3): videos carry their workflow.** Drop a FreeVideo video onto the ComfyUI canvas to restore its prompt, seed and settings. Earlier videos can get theirs too.
- **2026-10-05** · **[v0.2.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.2.0): four quality levels.** Choose Light, Medium, High or Max for each video; higher levels give higher quality but take longer. Results can be exported as sharing images or videos with the generation time and GPU.
- **2026-10-05** · **[v0.1.2](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.1.2): FreeVideo on Mac.** Apple silicon Macs generate MiniMax H3 videos locally (preview).
- **2026-10-03** · **Community LoRAs.** Use MiniMax H3 LoRAs right in your workflow. See [examples](docs/LoRA.md).
- **2026-10-02** · **FreeVideo is open source.** MiniMax H3 on consumer GPUs with as little as 8 GB of VRAM.

## About

FreeVideo is a local inference engine for MiniMax H3 on consumer GPUs, built on [OpenVDN](https://github.com/OpenVDN)'s 8-step [VDN-H3](https://huggingface.co/OpenVDN/vdn-minimax-h3) model with [Video DeltaNet](https://openvdn.github.io/)'s hybrid attention.

It coordinates VRAM, system memory and disk, adapting weight placement, compute precision and attention kernels to the available hardware. FreeVideo runs as a ComfyUI plugin, with launchers for Windows, macOS and Linux. Its core features include:

- **Hardware-adaptive execution**: Chooses the weight format for each GPU, int8 on GeForce RTX 30, 40 and 50 series cards and other Ampere cards, and FP8 on workstation and datacenter cards of newer architectures, and automatically probes the available attention kernels.
- **Low-memory inference**: Weight streaming, asynchronous prefetching and chunked computation keep peak memory low; 8 GB of VRAM or more and 16 GB of RAM are recommended.
- **Multimodal inputs**: Text prompts, first and last frames, and image, video and audio references.
- **Community LoRAs**: Use MiniMax H3 LoRAs in your workflow. See [examples](docs/LoRA.md).
- **ComfyUI integration**: A dedicated creative workspace inside ComfyUI that supports two-pass sampling and batch generation and keeps a history of past creations. For finer control, switch to the node view to add LoRAs or customize the workflow.
- **One-click deployment**: The Windows, Mac and Linux launchers set up ComfyUI, the runtime environment and the models, reuse existing models, and support offline installation.

## Getting Started

Requirements: an NVIDIA RTX 30 series or newer graphics card (professional cards of the same generation included; 8 GB of graphics memory or more recommended) and 16 GB of memory, or a Mac with Apple silicon (preview). GTX 10, GTX 16 and RTX 20 series cards and AMD or Intel graphics cannot run FreeVideo.

### Windows

1. [Download FreeVideo.exe](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo.exe) and run it.
2. Select an existing ComfyUI folder or install a new one. Existing model folders can be added for reuse; missing models are downloaded automatically.
3. Click **Install & launch**. ComfyUI opens in the browser with the FreeVideo workspace.

<div align="center">
  <img alt="FreeVideo creative workspace" src="https://github.com/user-attachments/assets/7647a6f8-4306-403c-b147-45d7a393e18d" width="92%">
</div>

**Security software:** FreeVideo.exe is code-signed. If security software blocks FreeVideo.exe or quarantines files in the FreeVideo folder, check that the SHA-256 of FreeVideo.exe matches the value on the release page, then restore the files and add the FreeVideo folder to the software's exclusion list.

**Offline installation:** Download the packages from [Quark](https://pan.quark.cn/s/c51235b84618) and drag the ZIP files into the launcher without extracting them. A fully offline installation needs the four model packs (video model, text encoder, video & audio decoder, and sampling caches), plus the environment package for a new ComfyUI installation. After importing only the environment package, you can also choose **Automatic download** for the models (v0.3.12 or newer). The audio reference cache is optional and is used only when a reference includes audio.

### macOS (Apple silicon preview)

1. [Download FreeVideo-Mac-arm64.dmg](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg), open it and drag **FreeVideo.app** into **Applications**.
2. Open FreeVideo, then select an existing ComfyUI folder or install a new one. Existing model folders can be added for reuse; the runtime environment and missing models are downloaded automatically.
3. Click **Install & launch**. ComfyUI opens in the browser with the FreeVideo workspace.

This preview has been tested on an M5 Mac with 24 GB of unified memory. See the [Mac&nbsp;guide](docs/Mac.md) for generation times and memory.

The Mac preview isn't notarized by Apple yet, so macOS blocks it the first time you open it. Download it only from the Releases page, then check the file and approve it as described in the [Mac guide](docs/Mac.md#first-open). This approves FreeVideo only; your other security settings stay as they are.

### Existing ComfyUI

Install FreeVideo as a custom node:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FlashML-org/FreeVideo.git
```

Restart ComfyUI, open **Workflow → Browse Templates → FreeVideo → FreeVideo-All-in-One**, and complete the setup in FreeVideo **Settings**.

To update, run `git pull` in `custom_nodes/FreeVideo` (or update the node in ComfyUI-Manager) and restart ComfyUI.

### Linux

**Desktop:** [Download FreeVideo-Linux-x86_64.AppImage](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage), make it executable and open it, then follow the same steps as on Windows.

**Terminal or server:**

```bash
git clone https://github.com/FlashML-org/FreeVideo.git && cd FreeVideo
./setup.sh
```

Setup installs FreeVideo, ComfyUI and the models, then opens FreeVideo in the browser. Afterwards, type `freevideo` to open it again. On a server, connect from your own computer through SSH, or run `freevideo server --listen 0.0.0.0` for an access link on your local network. See the [Linux guide](docs/Linux.md).

### More details

- [FreeVideo Adaptive Execution Planner](docs/execution-planning.md)
- [Community results](docs/community-results.md)
- [Thanks to the community](docs/community-thanks.md)

### Support

Report bugs in [GitHub Issues](https://github.com/FlashML-org/FreeVideo/issues), or ask questions on [Discord](https://discord.gg/MsA277cJzZ), [QQ](https://freevideo-community.pages.dev/qq) or [WeChat](https://freevideo-community.pages.dev/wechat).

## Citation

FreeVideo is based on VDN-H3. If you use FreeVideo in your research, please cite the [Video DeltaNet paper](https://arxiv.org/abs/2609.20744):

```bibtex
@article{xi2026videodeltanet,
  title={Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation},
  author={Xi, Haocheng and Xie, Yiming and Zhao, Hexu and Zhang, Yiwen and Liu, Michael and Creavin, Thomas and Keutzer, Kurt and Li, Xiuyu and Lv, Zhaoyang and Xu, Chenfeng and Feng, Haiwen},
  journal={arXiv preprint arXiv:2609.20744},
  year={2026}
}
```

## Team

### Project Team

[Bowen Xue](https://github.com/KBRASK) · [Shuo Yang](https://github.com/andy-yang-1) · [Haocheng Xi](https://github.com/haochengxi) · [Xiaoze Fan](https://github.com/jason-fxz) · [Chenfeng Xu](https://github.com/chenfengxu714)

### Special Thanks

Special thanks to [**AIwood爱屋研究室**](https://space.bilibili.com/503934057) and [**T8star-Aix**](https://space.bilibili.com/385085361) for testing the project and providing valuable feedback.

*Listed in chronological order of participation.*

## Acknowledgment

We thank [OpenVDN](https://github.com/OpenVDN) for [Video DeltaNet / VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) and its open-source model weights, training code and inference implementation.

We thank Impossible Research for providing computation resources.

We also thank [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for the base model and the following projects:
[ComfyUI](https://github.com/Comfy-Org/ComfyUI),
[Diffusers](https://github.com/huggingface/diffusers),
[SageAttention](https://github.com/thu-ml/SageAttention),
the [MiniMax H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler),
the [H3 text encoder for ComfyUI](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy) and
[Qt for Python](https://doc.qt.io/qtforpython-6/).

## License

The code is released under the [Apache License 2.0](LICENSE). The model weights are licensed under the [MiniMax H3 Community License](https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/blob/main/LICENSE), which includes territorial and acceptable-use restrictions.


---

## FILE: README.zh-CN.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/README.zh-CN.md)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="web/assets/freevideo.svg">
    <img alt="FreeVideo" src="web/assets/freevideo-light.svg" width="65%">
  </picture>
</div>

<p align="center">
| <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo.exe"><b>Windows</b></a> | <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg"><b>macOS</b></a> | <a href="https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage"><b>Linux</b></a> | <a href="docs/zh-CN/community-results.md"><b>社区实测</b></a> | <a href="https://freevideo-community.pages.dev/#gallery"><b>作品展示</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Discord</b></a> | <a href="https://freevideo-community.pages.dev/qq"><b>QQ 群</b></a> | <a href="https://freevideo-community.pages.dev/wechat"><b>微信群</b></a> |
</p>

<p align="center">系统要求：NVIDIA RTX 30 系列或更新的显卡（含同代专业卡，建议显存 8 GB 以上）和 16 GB 内存，或 Apple 芯片的 Mac（预览版）。GTX 10、GTX 16、RTX 20 系列以及 AMD、Intel 显卡无法运行 FreeVideo。</p>

<p align="center"><a href="README.md">English</a> · 中文</p>

FreeVideo 由 [Video DeltaNet（VDN）](https://openvdn.github.io/) 驱动，让 MiniMax H3 能够在计算能力 8.0 及以上的 NVIDIA 显卡（如 RTX 30 系列及更新，含同代专业卡）上本地运行，建议显存 8 GB 以上、内存 16 GB，并会根据硬件配置自动选择合适的加速路径。Apple 芯片 Mac 为预览版。

https://github.com/user-attachments/assets/ecda7d0d-7fbe-4e0c-8c29-8f3315bafc15

<p align="center"><a href="https://freevideo-community.pages.dev/#gallery">在作品展示中查看更多视频和四档质量对比 →</a></p>

## 更新动态

- **2026-11-10** · **[v0.5.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.5.0)：H3 Image 生图。** 8GB 显存起，本机出最高 16 MP 的图片。
- **2026-10-13** · **[v0.4.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.4.0)：Agent 全自动生成、分钟级长视频和剪辑视图。** 连接 Claude Code、Codex 等 AI Agent，让它全自动生成视频；一条视频可以生成到一分钟以上；在时间线上剪辑。
- **2026-10-07** · **[v0.3.2](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.3.2)：先出预览。** 先看半分辨率预览，满意的再继续生成。
- **2026-10-07** · **[v0.3.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.3.0)：更快的 int8 模型。** GeForce 和 RTX 30 系显卡改用 int8，RTX 3090 生成 5 秒视频快约 2.3 倍。
- **2026-10-06** · **[作品展示](https://freevideo-community.pages.dev/#gallery)上线。** 观看 FreeVideo 生成的 20 秒视频，以及四档质量的并排对比。
- **2026-10-06** · **[v0.2.3](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.2.3)：视频自带工作流。** 把 FreeVideo 生成的视频拖到 ComfyUI 画布上，即可还原提示词、种子和全部参数，之前的视频也能补上。
- **2026-10-05** · **[v0.2.0](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.2.0)：四档质量。** 每次生成可选择轻量、标准、精细或极致，档位越高，生成质量越高，但耗时更长；生成结果可导出为带生成耗时和显卡信息的分享图片或视频。
- **2026-10-05** · **[v0.1.2](https://github.com/FlashML-org/FreeVideo/releases/tag/v0.1.2)：支持 Mac。** Apple 芯片 Mac 也能本地生成 MiniMax H3 视频（预览版）。
- **2026-10-03** · **支持社区 LoRA。** 在工作流中直接使用 MiniMax H3 LoRA，参见[示例](docs/LoRA.zh-CN.md)。
- **2026-10-02** · **FreeVideo 开源。** 最低 8GB 显存，即可在消费级显卡上运行 MiniMax H3。

## 简介

FreeVideo 是面向消费级显卡的 MiniMax H3 本地推理引擎，基于 [OpenVDN](https://github.com/OpenVDN) 的 8 步模型 [VDN-H3](https://huggingface.co/OpenVDN/vdn-minimax-h3) 和 [Video DeltaNet](https://openvdn.github.io/) 的混合注意力。

它统一调度显存、内存与磁盘，并根据硬件条件调整权重放置、计算精度和注意力内核。FreeVideo 以 ComfyUI 插件形式提供，配备 Windows、macOS 和 Linux 启动器。主要特性包括：

- **硬件自适应**：针对不同显卡选择权重格式（GeForce RTX 30、40、50 系列和其他 Ampere 架构的显卡用 int8，更新架构的专业卡和数据中心卡用 FP8），并自动探测可用的注意力内核，无需手动配置。
- **低显存推理**：通过权重流式加载、异步预取与分块计算降低峰值显存，建议显存 8 GB 以上、内存 16 GB。
- **多模态输入**：支持文本、首帧、尾帧，以及图像、视频、音频参考输入。
- **社区 LoRA**：支持在工作流中使用 MiniMax H3 社区 LoRA。[查看效果对比](docs/LoRA.zh-CN.md)。
- **ComfyUI 集成**：在 ComfyUI 中提供专门的创作面板，支持二次采样和批量生成，并可浏览历史作品；需要更精细的控制时，可切换到节点视图，接入 LoRA 或自定义工作流。
- **一键部署**：Windows、Mac 和 Linux 启动器自动完成 ComfyUI、运行环境与模型的部署，可复用已有模型，并支持离线安装。

## 开始使用

系统要求：NVIDIA RTX 30 系列或更新的显卡（含同代专业卡，建议显存 8 GB 以上）和 16 GB 内存，或 Apple 芯片的 Mac（预览版）。GTX 10、GTX 16、RTX 20 系列以及 AMD、Intel 显卡无法运行 FreeVideo。

### Windows

1. [下载 FreeVideo.exe](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo.exe) 并运行。
2. 选择已有的 ComfyUI 目录或安装新的 ComfyUI。可添加已有的模型目录以复用文件，缺失的模型会自动下载。
3. 点击 **安装并启动**，浏览器中会打开带有 FreeVideo 创作面板的 ComfyUI。

<div align="center">
  <img alt="FreeVideo 创作面板" src="https://github.com/user-attachments/assets/f4719079-bb48-4836-9c64-2473477cd1a0" width="92%">
</div>

**安全软件：** FreeVideo.exe 已进行代码签名。如果安全软件拦截 FreeVideo.exe，或隔离了 FreeVideo 文件夹中的文件，请先核对 FreeVideo.exe 的 SHA-256 与发布页列出的值一致，再恢复文件，并把 FreeVideo 文件夹加入该软件的排除列表。

**离线安装：** 从[夸克网盘](https://pan.quark.cn/s/c51235b84618)下载离线包，将 ZIP 文件直接拖入启动器，无需解压。完全离线安装需要「视频模型」「文本编码器」「视频与音频解码器」「采样缓存」四个模型包，全新安装 ComfyUI 时还需要「运行环境」包。也可以只导入「运行环境」包，再在模型来源中选择**自动下载**（v0.3.12 及以上）。「音频参考缓存」为可选包，仅在参考素材带有音频时使用。

### macOS（Apple 芯片预览版）

1. [下载 FreeVideo-Mac-arm64.dmg](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg)，打开后将 **FreeVideo.app** 拖入 **应用程序**。
2. 打开 FreeVideo，选择已有的 ComfyUI 目录或安装新的 ComfyUI。可添加已有的模型目录以复用文件，缺失的运行环境和模型会自动下载。
3. 点击 **安装并启动**，浏览器中会打开带有 FreeVideo 创作面板的 ComfyUI。

预览版已在 24GB 统一内存的 M5 Mac 上测试，生成耗时与内存说明见 [Mac 说明](docs/Mac.zh-CN.md)。

Mac 预览版暂未进行 Apple 公证，首次打开时 macOS 会拦截。请只从 Release 页面下载，按 [Mac 说明](docs/Mac.zh-CN.md#首次打开)核对文件后再放行。这只对 FreeVideo 生效，不影响其他安全设置。

### 已有 ComfyUI

以自定义节点方式安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/FlashML-org/FreeVideo.git
```

重启 ComfyUI，打开 **工作流 → 浏览模板 → FreeVideo → FreeVideo-All-in-One**，然后在 FreeVideo 的 **设置** 中完成安装。

更新时，在 `custom_nodes/FreeVideo` 中运行 `git pull`（或在 ComfyUI-Manager 中更新该节点），然后重启 ComfyUI。

### Linux

**桌面：** [下载 FreeVideo-Linux-x86_64.AppImage](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage)，加上可执行权限后打开，之后的步骤与 Windows 相同。

**终端或服务器：**

```bash
git clone https://github.com/FlashML-org/FreeVideo.git && cd FreeVideo
./setup.sh
```

安装程序会装好 FreeVideo、ComfyUI 和模型，然后在浏览器中打开 FreeVideo。之后输入 `freevideo` 即可再次打开。在服务器上，可以通过 SSH 从自己的电脑访问，或运行 `freevideo server --listen 0.0.0.0` 获取局域网访问链接。详见 [Linux 说明](docs/Linux.zh-CN.md)。

### 更多文档

- [FreeVideo Adaptive Execution Planner](docs/zh-CN/execution-planning.md)
- [社区实测](docs/zh-CN/community-results.md)
- [感谢社区](docs/zh-CN/community-thanks.md)

### 问题反馈

问题和建议请提交至 [GitHub Issues](https://github.com/FlashML-org/FreeVideo/issues)，也可以在 [Discord](https://discord.gg/MsA277cJzZ)、[QQ 群](https://freevideo-community.pages.dev/qq)或[微信群](https://freevideo-community.pages.dev/wechat)中交流。

## 引用

FreeVideo 基于 VDN-H3。如在研究中使用 FreeVideo，请引用 [Video DeltaNet 论文](https://arxiv.org/abs/2609.20744)：

```bibtex
@article{xi2026videodeltanet,
  title={Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation},
  author={Xi, Haocheng and Xie, Yiming and Zhao, Hexu and Zhang, Yiwen and Liu, Michael and Creavin, Thomas and Keutzer, Kurt and Li, Xiuyu and Lv, Zhaoyang and Xu, Chenfeng and Feng, Haiwen},
  journal={arXiv preprint arXiv:2609.20744},
  year={2026}
}
```

## 团队

### 项目团队

[Bowen Xue](https://github.com/KBRASK) · [Shuo Yang](https://github.com/andy-yang-1) · [Haocheng Xi](https://github.com/haochengxi) · [Xiaoze Fan](https://github.com/jason-fxz) · [Chenfeng Xu](https://github.com/chenfengxu714)

### 特别感谢

特别感谢 [**AIwood爱屋研究室**](https://space.bilibili.com/503934057) 和 [**T8star-Aix**](https://space.bilibili.com/385085361) 参与项目测试并提供宝贵反馈。

*按参与时间先后排序。*

## 致谢

感谢 [OpenVDN](https://github.com/OpenVDN) 开源 [Video DeltaNet / VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) 的模型权重、训练代码与推理实现。

感谢 Impossible Research 提供计算资源。

同时感谢 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 提供基础模型，以及以下项目：
[ComfyUI](https://github.com/Comfy-Org/ComfyUI)、
[Diffusers](https://github.com/huggingface/diffusers)、
[SageAttention](https://github.com/thu-ml/SageAttention)、
[MiniMax H3 潜空间放大模型](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)、
[ComfyUI 版 H3 文本编码器](https://huggingface.co/t8star/Vdn-Minimax-H3-Comfy)和
[Qt for Python](https://doc.qt.io/qtforpython-6/)。

## 许可证

代码采用 [Apache License 2.0](LICENSE) 授权。模型权重适用 [MiniMax H3 社区许可证](https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/blob/main/LICENSE)，该许可证对使用地区和用途有所限制。


---

## FILE: docs/Linux.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/Linux.md)

# Linux

English · [中文](Linux.zh-CN.md)

FreeVideo runs on x86_64 Linux with an NVIDIA GPU and driver 580 or newer. On a desktop, the AppImage works like the Windows launcher. From a terminal or on a server, `./setup.sh` installs everything in one go, and the `freevideo` command opens it afterwards.

## Desktop: the AppImage

1. [Download FreeVideo-Linux-x86_64.AppImage](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage).
2. Make it executable, either in the file's properties (**Allow executing file as program**) or with `chmod +x FreeVideo-Linux-x86_64.AppImage`, then open it.
3. Choose a folder for a new installation, or an existing ComfyUI, and click **Install & launch**. ComfyUI opens in the browser with the FreeVideo workspace.

FreeVideo then appears in the application menu and on the desktop, and later versions update the launcher by themselves, as on Windows. The AppImage needs a distribution from 2022 or newer (Ubuntu 22.04, Debian 12, Fedora 36, RHEL 9 or later); on older systems, use the terminal installation below. If the computer lacks Git or a C++ compiler, FreeVideo downloads its own copies during the installation, without administrator rights.

## Terminal: one command

```bash
git clone https://github.com/FlashML-org/FreeVideo.git && cd FreeVideo
./setup.sh
```

Setup checks the GPU, memory and disk, shows the plan, and after you confirm it installs the runtime, the models and ComfyUI. It then starts ComfyUI and opens FreeVideo in the browser. Everything stays in the FreeVideo folder.

Afterwards, type `freevideo` in any terminal to open FreeVideo again. ComfyUI keeps running in the background until you stop it:

| Command | What it does |
| --- | --- |
| `freevideo` | Open FreeVideo; starts ComfyUI first if it is not running |
| `freevideo status` | Show whether it runs, and its address |
| `freevideo stop` | Stop ComfyUI; while a video is generating, only with `--force` |
| `freevideo restart` | Restart ComfyUI |

After `git pull`, `freevideo` restarts ComfyUI with the update; while a video is generating it leaves ComfyUI running and asks you to restart later. ComfyUI uses port 8188, or the next free one if another program has it; `freevideo open --port 8190` chooses another.

## Server: use FreeVideo from another computer

On a server without a desktop, setup ends by printing the address and an SSH command. ComfyUI only accepts connections from the server itself, so other computers reach it through an SSH tunnel. On your own computer, run:

```bash
ssh -N -L 8188:127.0.0.1:8188 you@server
```

Then open http://127.0.0.1:8188/?freevideo=launch in your browser.

To connect from computers on your local network without SSH:

```bash
freevideo server --listen 0.0.0.0
```

This prints an access link with a token, and a QR code of it to scan with a phone camera. Anyone with the link can use FreeVideo on the server; `freevideo server --listen 0.0.0.0 --new-token` replaces it, and `freevideo status` shows it again. ComfyUI itself still listens only on the server; a small proxy in front of it turns away every request without the token. The connection is not encrypted, so use it on a network you trust, or add `--tls-cert cert.pem --tls-key key.pem` for HTTPS. Through the link, pictures in ComfyUI's input folder can be opened only from FreeVideo's own folders there.

To start FreeVideo with the server, add `--enable`; it becomes a systemd user service. To keep it running after you log out and start it at boot, also run `sudo loginctl enable-linger $USER` once. `freevideo server --disable` removes the service.

## Command-line generation

The engine also runs without ComfyUI:

```bash
./freevideo generate --prompt-file prompt.txt --out video.mp4
./test.sh
```

`./freevideo --help` lists every command.

## Troubleshooting

- **The AppImage does not start.** Some systems lack FUSE. Run it with `--appimage-extract-and-run`, or install the `fuse3` package.
- **ComfyUI does not start.** `freevideo status` names the last error. The logs are in the `launcher` folder of the installation (`service.log`, and `comfy-runs` for ComfyUI itself).
- **The address is in use.** Another program uses the port; start FreeVideo on another one with `freevideo open --port 8190`.


---

## FILE: docs/Linux.zh-CN.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/Linux.zh-CN.md)

# Linux

[English](Linux.md) · 中文

FreeVideo 支持 x86_64 Linux，需要 NVIDIA 显卡和 580 及以上版本的驱动。桌面环境下，AppImage 的用法与 Windows 启动器相同；在终端或服务器上，运行 `./setup.sh` 即可一次装好，之后用 `freevideo` 命令打开。

## 桌面：AppImage

1. [下载 FreeVideo-Linux-x86_64.AppImage](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Linux-x86_64.AppImage)。
2. 在文件属性中勾选**允许作为程序执行**，或运行 `chmod +x FreeVideo-Linux-x86_64.AppImage`，然后打开它。
3. 选择新安装的文件夹或已有的 ComfyUI，点击**安装并启动**。ComfyUI 会在浏览器中打开，并进入 FreeVideo 创作面板。

安装完成后，应用菜单和桌面上会出现 FreeVideo，启动器也会像 Windows 版一样自动更新。AppImage 需要 2022 年以后的发行版（Ubuntu 22.04、Debian 12、Fedora 36、RHEL 9 或更新版本）；更老的系统请使用下面的终端安装方式。如果系统缺少 Git 或 C++ 编译器，FreeVideo 会在安装时自行下载，不需要管理员权限。

## 终端：一条命令

```bash
git clone https://github.com/FlashML-org/FreeVideo.git && cd FreeVideo
./setup.sh
```

安装程序会检查显卡、内存和硬盘，显示安装计划，确认后安装运行环境、模型和 ComfyUI，然后启动 ComfyUI，并在浏览器中打开 FreeVideo。所有文件都在 FreeVideo 文件夹内。

之后在任意终端输入 `freevideo` 即可再次打开。ComfyUI 会在后台持续运行，直到手动停止：

| 命令 | 作用 |
| --- | --- |
| `freevideo` | 打开 FreeVideo；ComfyUI 没有运行时先启动它 |
| `freevideo status` | 查看是否在运行，以及访问地址 |
| `freevideo stop` | 停止 ComfyUI；有视频正在生成时需加 `--force` |
| `freevideo restart` | 重启 ComfyUI |

执行 `git pull` 后，再运行 `freevideo` 会重启 ComfyUI 加载新版本；如果有视频正在生成，则保持运行，并提示稍后重启。ComfyUI 默认使用 8188 端口，被其他程序占用时自动改用下一个空闲端口；也可以用 `freevideo open --port 8190` 指定端口。

## 服务器：从其他电脑使用

在没有桌面的服务器上，安装结束时会打印访问地址和一条 SSH 命令。ComfyUI 只接受服务器本机的连接，其他电脑通过 SSH 隧道访问。在自己的电脑上运行：

```bash
ssh -N -L 8188:127.0.0.1:8188 用户名@服务器
```

然后在浏览器中打开 http://127.0.0.1:8188/?freevideo=launch 。

如果希望局域网内的电脑不经过 SSH 直接访问：

```bash
freevideo server --listen 0.0.0.0
```

这会打印一个带访问令牌的链接和对应的二维码（可用手机相机扫描），持有链接的人都可以使用这台服务器上的 FreeVideo；用 `freevideo server --listen 0.0.0.0 --new-token` 可以换一个新链接，`freevideo status` 可以再次查看。ComfyUI 本身仍然只监听服务器本机，前面的访问代理会拒绝所有不带令牌的请求。连接没有加密，请只在可信的网络中使用，或加上 `--tls-cert cert.pem --tls-key key.pem` 使用 HTTPS。通过链接只能打开 ComfyUI 输入文件夹中 FreeVideo 自己子文件夹里的图片。

如需随服务器启动，加上 `--enable`，FreeVideo 会注册为 systemd 用户服务；若要在注销后保持运行、开机自动启动，还需运行一次 `sudo loginctl enable-linger $USER`。`freevideo server --disable` 会移除这个服务。

## 命令行生成

引擎也可以不经过 ComfyUI 直接运行：

```bash
./freevideo generate --prompt-file prompt.txt --out video.mp4
./test.sh
```

`./freevideo --help` 列出全部命令。

## 常见问题

- **AppImage 打不开**：部分系统缺少 FUSE。可以加上 `--appimage-extract-and-run` 运行，或安装 `fuse3`。
- **ComfyUI 没有启动**：运行 `freevideo status` 查看上次的错误。日志在安装目录的 `launcher` 文件夹中（`service.log`，ComfyUI 本身的日志在 `comfy-runs`）。
- **端口被占用**：有其他程序在使用该端口，可以用 `freevideo open --port 8190` 换一个端口。


---

## FILE: docs/LoRA.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/LoRA.md)

# Community LoRAs

English · [中文](LoRA.zh-CN.md)

FreeVideo supports community LoRAs for MiniMax H3. Each comparison uses the same prompt and seed.

## [VHS](https://huggingface.co/KennethFal/vh5tape-vhs-lora-minimax-h3)

| Without LoRA | With LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/c32440e6-6fba-4685-bd47-3cdac3b5c63e | https://github.com/user-attachments/assets/83989e7a-0f4f-4e58-80f3-705d4fc3e392 |

## [Looping Sketch Anime](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime)

| Without LoRA | With LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/60982172-eb0e-4f30-a19d-7ffcbe5ff920 | https://github.com/user-attachments/assets/1bab08eb-8c1f-4ee7-a25c-960ac40a201b |

## [Male POV](https://civitai.com/models/2966380/minimax-h3-male-pov)

| Without LoRA | With LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/163924ad-dd11-4b08-9b91-62c823b6c0a7 | https://github.com/user-attachments/assets/17d3374b-c3b3-409e-8da5-6a1415bb4acb |


---

## FILE: docs/LoRA.zh-CN.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/LoRA.zh-CN.md)

# 社区 LoRA

[English](LoRA.md) · 中文

FreeVideo 支持 MiniMax H3 社区 LoRA。每组对照使用相同的提示词和 seed。

## [VHS 录像带](https://huggingface.co/KennethFal/vh5tape-vhs-lora-minimax-h3)

| 无 LoRA | 有 LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/c32440e6-6fba-4685-bd47-3cdac3b5c63e | https://github.com/user-attachments/assets/83989e7a-0f4f-4e58-80f3-705d4fc3e392 |

## [手绘线稿动画](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime)

| 无 LoRA | 有 LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/60982172-eb0e-4f30-a19d-7ffcbe5ff920 | https://github.com/user-attachments/assets/1bab08eb-8c1f-4ee7-a25c-960ac40a201b |

## [Male POV 第一人称](https://civitai.com/models/2966380/minimax-h3-male-pov)

| 无 LoRA | 有 LoRA |
| :---: | :---: |
| https://github.com/user-attachments/assets/163924ad-dd11-4b08-9b91-62c823b6c0a7 | https://github.com/user-attachments/assets/17d3374b-c3b3-409e-8da5-6a1415bb4acb |


---

## FILE: docs/Mac.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/Mac.md)

# Mac (experimental)

English · [中文](Mac.zh-CN.md)

FreeVideo runs MiniMax H3 locally on Apple silicon Macs with macOS 14 or later and the Xcode Command Line Tools. Mac support is experimental and still being optimized.

## Installation

1. [Download FreeVideo-Mac-arm64.dmg](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg) and drag FreeVideo into Applications.
2. Open FreeVideo. The preview needs your approval the first time; see [First open](#first-open).
3. Choose an install location, optionally add existing model folders, and click **Install & launch**. The Mac environment and missing models are downloaded automatically.

## First open

The Mac preview isn't notarized by Apple yet, so macOS blocks it the first time and says it can't verify the developer. Before approving it, make sure the file is the official one:

1. Download it only from the [FreeVideo releases page](https://github.com/FlashML-org/FreeVideo/releases/latest).
2. In Terminal, run `shasum -a 256 ~/Downloads/FreeVideo-Mac-arm64.dmg`. The result should match the value in SHA256SUMS.txt on the release.
3. Open FreeVideo and dismiss the alert. Then, in **System Settings → Privacy & Security**, find FreeVideo, click **Open Anyway** and confirm.

This approves FreeVideo only. It doesn't turn off Gatekeeper or change your other security settings. If macOS says the file is damaged or contains malware, don't open it, and let us know. Once FreeVideo is notarized, this step goes away.

## Features

- Computation runs on the GPU through Metal; attention in sampling and video decoding uses MLX's fused kernels.
- Before each stage, attention head groups, chunk sizes and the number of resident transformer layers are set from the unified memory available at that moment.
- Macs download ConvRot int8 weights. On Apple M5 and newer, the large projections run as int8 on the GPU's Metal 4 tensor units; earlier Macs decode them to BF16 on the GPU as each layer loads.
- Layers that are not kept in memory stream from disk with read-ahead.
- Text prompts, first-frame input and two-pass sampling are supported.

## Generation time

On an M5 with 24 GB of unified memory and about 14 GB available, at the Light level:

| Resolution | Length | Time |
| --- | --- | --- |
| 1344 × 768 | 10 s | about 42 min |
| 960 × 544 | 10 s | about 22 min |
| 512 × 512 | 1.6 s | about 3 min |

These times use the M5's int8 tensor units. Macs before the M5 run the same weights in BF16 and take longer. Mac support is still being optimized; more unified memory and newer chips shorten generation further.


---

## FILE: docs/Mac.zh-CN.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/Mac.zh-CN.md)

# Mac（实验性）

[English](Mac.md) · 中文

FreeVideo 可在 Apple 芯片 Mac 上本地运行 MiniMax H3，需要 macOS 14 及以上和 Xcode 命令行工具。Mac 版本目前为实验性质，仍在持续优化。

## 安装

1. [下载 FreeVideo-Mac-arm64.dmg](https://github.com/FlashML-org/FreeVideo/releases/latest/download/FreeVideo-Mac-arm64.dmg)，将 FreeVideo 拖入「应用程序」。
2. 打开 FreeVideo。预览版首次打开需要确认一次，见下方[首次打开](#首次打开)。
3. 选择安装位置，可添加已有的模型目录，然后点击 **安装并启动**。Mac 运行环境和缺失的模型会自动下载。

## 首次打开

Mac 预览版暂未进行 Apple 公证，因此首次打开时 macOS 会拦截，并提示无法验证开发者。放行前请先确认文件来自官方：

1. 只从 [FreeVideo 的 Release 页面](https://github.com/FlashML-org/FreeVideo/releases/latest)下载。
2. 在「终端」中运行 `shasum -a 256 ~/Downloads/FreeVideo-Mac-arm64.dmg`，输出应与 Release 中 SHA256SUMS.txt 里的值一致。
3. 打开 FreeVideo，关闭系统提示后，在 **系统设置 → 隐私与安全** 中找到 FreeVideo，点击 **仍要打开** 并按提示确认。

这只放行 FreeVideo，不会关闭 Gatekeeper，也不会改动其他安全设置。如果系统提示文件已损坏或包含恶意软件，请不要打开，并向我们反馈。完成公证后将不再需要这一步。

## 特性

- 通过 Metal 在 GPU 上计算，采样和视频解码中的注意力使用 MLX 融合内核。
- 每个阶段开始前，根据当前可用的统一内存确定注意力头分组、分块大小和常驻的 Transformer 层数。
- Mac 下载 ConvRot int8 权重。M5 及更新的芯片在 GPU 的 Metal 4 张量单元上以 int8 计算大矩阵；更早的芯片在每层加载时于 GPU 上解码为 BF16 计算。
- 未常驻内存的层从磁盘流式加载并提前预取。
- 支持文本、首帧输入和二次采样。

## 生成时间

在 M5、24GB 统一内存、可用内存约 14GB、轻量档的情况下：

| 分辨率 | 时长 | 生成时间 |
| --- | --- | --- |
| 1344 × 768 | 10 秒 | 约 42 分钟 |
| 960 × 544 | 10 秒 | 约 22 分钟 |
| 512 × 512 | 1.6 秒 | 约 3 分钟 |

以上时间使用了 M5 的 int8 张量单元。M5 之前的 Mac 以 BF16 运行同一套权重，耗时更长。Mac 版本仍在持续优化，更大的统一内存和更新的芯片都会进一步缩短生成时间。


---

## FILE: docs/community-results.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/community-results.md)

# Community results

Thanks to the community members who sent diagnostic reports. FreeVideo already runs on everything from 8 GB laptops to professional workstations; the table below summarizes the measurements and optimizations from these reports:

| GPU | VRAM | RAM | Resolution | Length | Quality | Version | Time | Optimized&nbsp;in | After |
|---|---:|---:|---:|---:|:---:|:---:|---:|:---:|---:|
| 4060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 1920×1088 | 15&nbsp;s | Light | 0.3.0 | 3306&nbsp;s | 0.3.6 | 2730&nbsp;s&nbsp;(−580&nbsp;s) |
| 5060 | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 15&nbsp;s | Light | 0.3.1 | 1367&nbsp;s | 0.3.6 | 1160&nbsp;s&nbsp;(−210&nbsp;s) |
| 3070&nbsp;Ti | 8&nbsp;GB | 96&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.1 | 749&nbsp;s | 0.3.22 | 665&nbsp;s&nbsp;(−80&nbsp;s)⁶ |
| 4060 | 8&nbsp;GB | 32&nbsp;GB | 768×1344 | 15&nbsp;s | Light | 0.3.4 | 1592&nbsp;s | 0.3.10 | 1450&nbsp;s&nbsp;(−140&nbsp;s) |
| 5070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5&nbsp;s | Light | 0.3.4 | 555&nbsp;s | 0.3.10 | 485&nbsp;s&nbsp;(−70&nbsp;s) |
| PRO&nbsp;5000 | 72&nbsp;GB | 128&nbsp;GB | 768×1344 | 10&nbsp;s | High | 0.3.4 | 552&nbsp;s | 0.3.4 | Already&nbsp;optimal |
| 4070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 8&nbsp;s | Light | 0.3.5 | 1031&nbsp;s | 0.3.21 | 830&nbsp;s&nbsp;(−200&nbsp;s) |
| 3070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5&nbsp;s | Light | 0.3.8 | 3821&nbsp;s | 0.3.8 | Thermal‑limited¹ |
| 5080&nbsp;Laptop | 16&nbsp;GB | 32&nbsp;GB | 1184×672 | 10&nbsp;s | Light | 0.3.8 | 416&nbsp;s | 0.3.21 | 385&nbsp;s&nbsp;(−30&nbsp;s) |
| 5090 | 32&nbsp;GB | 32&nbsp;GB | 1344×768 | 6.6&nbsp;s | Max | 0.3.8 | 336&nbsp;s | 0.3.8 | No&nbsp;report&nbsp;data |
| 5070&nbsp;Ti | 16&nbsp;GB | 48&nbsp;GB | 576×928 | 12.3&nbsp;s | Medium | 0.3.8 | 472&nbsp;s | int8&nbsp;model² | 420&nbsp;s&nbsp;(−50&nbsp;s) |
| A5000&nbsp;Laptop | 16&nbsp;GB | 128&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.8 | 1108&nbsp;s | 0.3.8 | Thermal‑limited¹ |
| 4060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 1344×768 | 15&nbsp;s | Max | 0.3.8 | 9829&nbsp;s | 0.3.10 | 2890&nbsp;s&nbsp;(−6940&nbsp;s) |
| 5050&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 960×544 | 3&nbsp;s | Light³ | 0.3.8 | 2313&nbsp;s | 0.3.18 | 2130&nbsp;s&nbsp;(−180&nbsp;s) |
| 3070 | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.13 | 899&nbsp;s | 0.3.15 | 880&nbsp;s&nbsp;(−20&nbsp;s) |
| 5060&nbsp;Ti | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.13 | 508&nbsp;s | 0.3.17 | 495&nbsp;s&nbsp;(−10&nbsp;s) |
| 4060&nbsp;Laptop | 8&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.14 | 1296&nbsp;s | 0.3.21 | 1250&nbsp;s&nbsp;(−50&nbsp;s) |
| 5060 | 8&nbsp;GB | 16&nbsp;GB | 1280×736 | 15&nbsp;s | Light | 0.3.14 | 5285&nbsp;s | 0.3.21 | 4650&nbsp;s&nbsp;(−640&nbsp;s) |
| 4070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 960×544 | 5&nbsp;s | Light | 0.3.17 | 813&nbsp;s | 0.3.21 | 590&nbsp;s&nbsp;(−220&nbsp;s) |
| M4&nbsp;Max | Unified | 64&nbsp;GB | 768×1344 | 5&nbsp;s | Light | 0.3.5 | 2078&nbsp;s | 0.3.5 | Already&nbsp;optimal |
| 5090&nbsp;Laptop | 24&nbsp;GB | 64&nbsp;GB | 768×1344 | 2.3&nbsp;s | — | 0.3.18 | 132&nbsp;s | 0.3.21 | 122&nbsp;s&nbsp;(−10&nbsp;s)⁴ |
| 4060&nbsp;Ti | 16&nbsp;GB | 80&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.20 | 1122&nbsp;s | 0.3.21 | 1110&nbsp;s&nbsp;(−10&nbsp;s) |
| 5070&nbsp;Ti&nbsp;Laptop | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | High | 0.3.20 | 1545&nbsp;s | 0.3.21 | 1310&nbsp;s&nbsp;(−240&nbsp;s) |
| 5060&nbsp;Ti | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.20 | 471&nbsp;s | 0.3.21 | 465&nbsp;s&nbsp;(−10&nbsp;s) |
| 4070&nbsp;SUPER | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 15&nbsp;s | High | 0.3.20 | 2966&nbsp;s | 0.3.21 | 2050&nbsp;s&nbsp;(−920&nbsp;s) |
| 4070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 544×960 | 3&nbsp;s | Light⁵ | 0.3.22 | 133&nbsp;s | 0.3.22 | Already&nbsp;optimal |
| M4&nbsp;Pro | Unified | 24&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.3.8 | 3457&nbsp;s | 0.3.8 | Already&nbsp;optimal |
| 5090 | 32&nbsp;GB | 64&nbsp;GB | 1376×768 | 10&nbsp;s | Light | 0.3.20 | 294&nbsp;s | 0.3.23 | 265&nbsp;s&nbsp;(−30&nbsp;s) |
| 5060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 960×544 | 3&nbsp;s | Light | 0.3.21 | 908&nbsp;s | 0.3.23 | 820&nbsp;s&nbsp;(−90&nbsp;s) |
| 3060 | 12&nbsp;GB | 64&nbsp;GB | 1184×672 | 10&nbsp;s | Light⁵ | 0.3.22 | 604&nbsp;s | 0.3.23 | 565&nbsp;s&nbsp;(−40&nbsp;s) |
| 4070 | 12&nbsp;GB | 32&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.3.22 | 340&nbsp;s | 0.3.22 | Already&nbsp;optimal |
| 5090 | 32&nbsp;GB | 64&nbsp;GB | 1344×768 | 10.8&nbsp;s | Light | 0.3.22 | 249&nbsp;s | 0.3.22 | Already&nbsp;optimal |
| 3090 | 24&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.23 | 704&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 1344×768 | 12.3&nbsp;s | Light | 0.3.23 | 573&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 5080 | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.23 | 240&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 3060&nbsp;Ti | 8&nbsp;GB | 48&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.3.22 | 508&nbsp;s | 0.3.24 | 490&nbsp;s&nbsp;(−20&nbsp;s) |
| 3070&nbsp;Laptop | 8&nbsp;GB | 24&nbsp;GB | 864×480 | 5&nbsp;s | Light | 0.3.22 | 677&nbsp;s | 0.3.24 | 645&nbsp;s&nbsp;(−30&nbsp;s) |
| 4060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 1312×768 | 10&nbsp;s | Light | 0.3.23 | 1024&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 96&nbsp;GB | 768×1344 | 5&nbsp;s | Light | 0.3.23 | 493&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| M5&nbsp;Pro | Unified | 48&nbsp;GB | 960×544 | 5&nbsp;s | Light | 0.3.23 | 527&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 4090 | 24&nbsp;GB | 64&nbsp;GB | 1344×768 | 30.7&nbsp;s | Max | 0.3.23 | 5829&nbsp;s | 0.3.25 | 4850&nbsp;s&nbsp;(−980&nbsp;s) |
| 5060 | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.23 | 1279&nbsp;s | 0.3.25 | 1150&nbsp;s&nbsp;(−130&nbsp;s) |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 832×1248 | 10&nbsp;s | Light | 0.3.24 | 1668&nbsp;s | 0.3.24 | GPU&nbsp;shared⁷ |
| 5080 | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.24 | 312&nbsp;s | 0.3.24 | 278&nbsp;s&nbsp;(−34&nbsp;s) |
| 4070&nbsp;Ti&nbsp;SUPER | 16&nbsp;GB | 96&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.24 | 575&nbsp;s | 0.3.24 | Already&nbsp;optimal |
| 4080&nbsp;Laptop | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.24 | 970&nbsp;s | 0.3.24 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 768×1344 | 6.6&nbsp;s | Light⁵ | 0.3.24 | 386&nbsp;s | 0.3.24 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 768×1344 | 15.1&nbsp;s | Light | 0.3.24 | 847&nbsp;s | 0.3.24 | Already&nbsp;optimal |
| 5060 | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 4.5&nbsp;s | Light | 0.3.25 | 390&nbsp;s | 0.3.25 | Already&nbsp;optimal |
| M4&nbsp;Pro | Unified | 48&nbsp;GB | 1120×928 | 10&nbsp;s | Light⁵ | 0.3.20 | 5197&nbsp;s | 0.3.20 | Already&nbsp;optimal |
| 3080 | 20&nbsp;GB | 64&nbsp;GB | 1344×768 | 20&nbsp;s | High | 0.3.20 | 2526&nbsp;s | 0.4.0 | 2335&nbsp;s&nbsp;(−191&nbsp;s) |
| 5070&nbsp;Ti&nbsp;Laptop | 12&nbsp;GB | 32&nbsp;GB | 960×544 | 6.6&nbsp;s | Medium | 0.3.23 | 291&nbsp;s | 0.3.23 | Already&nbsp;optimal |
| 3090 | 24&nbsp;GB | 32&nbsp;GB | 1024×576 | 15&nbsp;s | Light | 0.3.23 | 1504&nbsp;s⁸ | 0.4.0 | 1455&nbsp;s&nbsp;(−50&nbsp;s) |
| 3060 | 12&nbsp;GB | 40&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.24 | 1676&nbsp;s⁹ | 0.4.0 | 1660&nbsp;s&nbsp;(−20&nbsp;s) |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.3.25 | 769&nbsp;s | 0.4.0 | 755&nbsp;s&nbsp;(−15&nbsp;s) |
| 3070&nbsp;Laptop | 16&nbsp;GB | 32&nbsp;GB | 960×544 | 15&nbsp;s | Light | 0.3.25 | 1539&nbsp;s | 0.4.0 | 1520&nbsp;s&nbsp;(−20&nbsp;s) |
| 3070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5&nbsp;s | Light | 0.3.25 | 513&nbsp;s | 0.4.0 | 500&nbsp;s&nbsp;(−13&nbsp;s) |
| 3070&nbsp;Ti | 8&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Max | 0.3.25 | 8501&nbsp;s | 0.4.0 | 8490&nbsp;s&nbsp;(−11&nbsp;s) |
| 3060 | 12&nbsp;GB | 128&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.3.25 | 1074&nbsp;s | 0.4.0 | 1055&nbsp;s&nbsp;(−19&nbsp;s) |
| 5060&nbsp;Ti | 16&nbsp;GB | 16&nbsp;GB | 960×544 | 4.5&nbsp;s | Light | 0.3.25 | 1366&nbsp;s | 0.4.1 | 1100&nbsp;s&nbsp;(−270&nbsp;s) |
| 4070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light⁵ | 0.4.1 | 1297&nbsp;s | 0.4.1 | 585&nbsp;s&nbsp;(−710&nbsp;s) |
| 3070 | 16&nbsp;GB | 32&nbsp;GB | 1344×768 | 12.3&nbsp;s | Max | 0.4.0 | 10090&nbsp;s | 0.4.0 | GPU&nbsp;shared¹⁰ |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.3 | 1297&nbsp;s | 0.4.3 | Already&nbsp;optimal |
| 4060&nbsp;Ti | 8&nbsp;GB | 16&nbsp;GB | 672×1184 | 6.6&nbsp;s | Light | 0.4.0 | 887&nbsp;s¹¹ | 0.4.0 | 887&nbsp;s |
| 3070&nbsp;Laptop | 8&nbsp;GB | 27&nbsp;GB | 1344×768 | 10&nbsp;s | Light⁵ | 0.4.4 | 625&nbsp;s | 0.4.4 | Already&nbsp;optimal |
| 5070&nbsp;Laptop | 8&nbsp;GB | 47&nbsp;GB | 960×544 | 5.2&nbsp;s | Light | 0.4.4 | 554&nbsp;s | 0.4.4 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 126&nbsp;GB | 1344×768 | 10&nbsp;s | Medium | 0.4.4 | 753&nbsp;s | 0.4.4 | Already&nbsp;optimal |
| 5090 | 32&nbsp;GB | 96&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.4 | 400&nbsp;s | 0.4.4 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 1088×1920 | 15&nbsp;s | Light | 0.4.3 | 4292&nbsp;s | 0.4.5 | 4245&nbsp;s&nbsp;(−47&nbsp;s) |
| 5090 | 32&nbsp;GB | 96&nbsp;GB | 1344×768 | 10&nbsp;s | Max | 0.4.4 | 517&nbsp;s | 0.4.4 | Already&nbsp;optimal |
| 5070&nbsp;Ti&nbsp;Laptop | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 4.5&nbsp;s | Max | 0.4.5 | 819&nbsp;s | 0.4.5 | Already&nbsp;optimal |
| 3090 | 24&nbsp;GB | 64&nbsp;GB | 1344×768 | 6.6&nbsp;s | Light | 0.3.24 | 461&nbsp;s | 0.4.0 | 430&nbsp;s&nbsp;(−31&nbsp;s) |
| 3060&nbsp;Laptop | 6&nbsp;GB | 40&nbsp;GB | 1344×768 | 5.2&nbsp;s | Light | 0.4.7 | 510&nbsp;s | 0.4.7 | Already&nbsp;optimal |
| 5060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 3&nbsp;s | Medium | 0.4.7 | 541&nbsp;s | 0.4.7 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 34&nbsp;GB | 1344×768 | 5.2&nbsp;s | Medium | 0.4.8 | 2521&nbsp;s | 0.4.8 | Already&nbsp;optimal¹² |
| 4060 | 8&nbsp;GB | 32&nbsp;GB | 768×1344 | 17&nbsp;s | Light | 0.4.8 | 1942&nbsp;s | 0.4.8 | Already&nbsp;optimal |
| M1&nbsp;Max | Unified | 64&nbsp;GB | 960×544 | 2.3&nbsp;s | Light | 0.4.8 | 1226&nbsp;s | 0.4.8 | Already&nbsp;optimal |
| 3070 | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5.2&nbsp;s | Light⁵ | 0.4.8 | 285&nbsp;s | 0.4.8 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 34&nbsp;GB | 1664×960 | 6.6&nbsp;s | Medium | 0.4.8 | 4025&nbsp;s | 0.4.8 | Already&nbsp;optimal¹² |
| 4070&nbsp;Laptop | 8&nbsp;GB | 34&nbsp;GB | 1344×768 | 15&nbsp;s | Light⁵ | 0.4.8 | 1444&nbsp;s | 0.4.8 | Already&nbsp;optimal¹² |
| 5060&nbsp;Ti | 16&nbsp;GB | 56&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.4.6 | 904&nbsp;s | 0.4.9 | 805&nbsp;s&nbsp;(−99&nbsp;s) |
| 3070&nbsp;Laptop | 8&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.9 | 1267&nbsp;s | 0.4.9 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 64&nbsp;GB | 768×1344 | 5&nbsp;s | Light | 0.4.12 | 639&nbsp;s | 0.4.12 | Already&nbsp;optimal |
| 5080 | 16&nbsp;GB | 64&nbsp;GB | 768×1344 | 3&nbsp;s | Max | 0.3.20 | 216&nbsp;s | 0.3.20 | Already&nbsp;optimal |
| 4060&nbsp;Ti | 8&nbsp;GB | 32&nbsp;GB | 960×544 | 5&nbsp;s | Light | 0.4.12 | 333&nbsp;s | 0.4.12 | Already&nbsp;optimal |
| 5070&nbsp;Ti | 16&nbsp;GB | 64&nbsp;GB | 1184×672 | 15&nbsp;s | Medium | 0.4.13 | 487&nbsp;s | 0.4.13 | Already&nbsp;optimal |
| 3070 | 16&nbsp;GB | 32&nbsp;GB | 768×1344 | 5.2&nbsp;s | Light | 0.4.14 | 2380&nbsp;s | 0.4.14 | Already&nbsp;optimal |
| M4&nbsp;Max | Unified | 128&nbsp;GB | 768×1344 | 10&nbsp;s | Light | 0.4.14 | 3444&nbsp;s | 0.4.14 | Already&nbsp;optimal |
| 4060&nbsp;Ti | 8&nbsp;GB | 32&nbsp;GB | 672×1184 | 10.8&nbsp;s | Light | 0.4.14 | 936&nbsp;s | 0.4.14 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.15 | 1147&nbsp;s | 0.4.15 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 24&nbsp;GB | 1344×768 | 13.7&nbsp;s | Light | 0.4.15 | 1747&nbsp;s | 0.4.20 | 1700&nbsp;s&nbsp;(−50&nbsp;s) |
| 5070&nbsp;Laptop | 8&nbsp;GB | 64&nbsp;GB | 1184×672 | 10&nbsp;s | Light | 0.4.15 | 822&nbsp;s | int8&nbsp;model² | 720&nbsp;s&nbsp;(−100&nbsp;s) |
| 5080 | 16&nbsp;GB | 64&nbsp;GB | 1024×1024 | 15&nbsp;s | Light | 0.4.16 | 412&nbsp;s | 0.4.16 | Already&nbsp;optimal |
| M1&nbsp;Max | Unified | 64&nbsp;GB | 960×544 | 5.2&nbsp;s | Light | 0.4.16 | 1816&nbsp;s | 0.4.16 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5.2&nbsp;s | Light⁵ | 0.4.17 | 470&nbsp;s | 0.4.17 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 10.8&nbsp;s | Light⁵ | 0.4.17 | 967&nbsp;s | 0.4.19 | 880&nbsp;s&nbsp;(−90&nbsp;s) |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Max | 0.4.17 | 2311&nbsp;s | 0.4.17 | Already&nbsp;optimal |
| 5080 | 16&nbsp;GB | 48&nbsp;GB | 768×1344 | 12.3&nbsp;s | Light | 0.4.17 | 293&nbsp;s | 0.4.17 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 64&nbsp;GB | 960×544 | 9.4&nbsp;s | Light | 0.4.17 | 743&nbsp;s | 0.4.17 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 768×1344 | 5.2&nbsp;s | Light | 0.4.17 | 811&nbsp;s | 0.4.17 | Already&nbsp;optimal |
| 5070&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 1184×672 | 15&nbsp;s | Light | 0.4.19 | 447&nbsp;s | 0.4.19 | Already&nbsp;optimal |
| 4060 | 8&nbsp;GB | 32&nbsp;GB | 960×544 | 15&nbsp;s | Light | 0.4.19 | 1311&nbsp;s | 0.4.19 | Already&nbsp;optimal |
| M5 | Unified | 24&nbsp;GB | 1344×768 | 10&nbsp;s | Max | 0.4.12 | 30509&nbsp;s | 0.4.14 | 27000&nbsp;s&nbsp;(−3500&nbsp;s) |
| 4070&nbsp;Ti | 12&nbsp;GB | 64&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.4.19 | 236&nbsp;s | 0.4.19 | Already&nbsp;optimal |
| 3090 | 24&nbsp;GB | 128&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.19 | 377&nbsp;s | 0.4.19 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 672×1184 | 15&nbsp;s | Light | 0.4.20 | 1608&nbsp;s | 0.4.20 | Already&nbsp;optimal |
| M5&nbsp;Max | Unified | 128&nbsp;GB | 1664×960 | 3&nbsp;s | Light | 0.4.18 | 755&nbsp;s | 0.4.18 | Already&nbsp;optimal |
| 5050&nbsp;Laptop | 8&nbsp;GB | 24&nbsp;GB | 544×960 | 8&nbsp;s | Light | 0.4.21 | 534&nbsp;s | 0.4.21 | Already&nbsp;optimal |
| 3080 | 10&nbsp;GB | 32&nbsp;GB | 768×1344 | 15&nbsp;s | Max | 0.4.22 | 1965&nbsp;s | 0.4.22 | Already&nbsp;optimal |
| 4090&nbsp;Laptop | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Medium | 0.4.14 | 590&nbsp;s | 0.4.17 | 570&nbsp;s&nbsp;(−20&nbsp;s) |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 960×544 | 10&nbsp;s | Light | 0.4.6 | 517&nbsp;s | 0.4.6 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.14 | 1168&nbsp;s | 0.4.14 | Already&nbsp;optimal |
| 5070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 11.5&nbsp;s | Light | 0.4.17 | 1003&nbsp;s | 0.4.24 | 900&nbsp;s&nbsp;(−100&nbsp;s) |
| 3070&nbsp;Ti | 8&nbsp;GB | 64&nbsp;GB | 1344×768 | 16.5&nbsp;s | Light | 0.4.17 | 1623&nbsp;s | 0.4.24 | 1570&nbsp;s&nbsp;(−50&nbsp;s) |
| 3060&nbsp;Ti | 8&nbsp;GB | 32&nbsp;GB | 768×1344 | 10&nbsp;s | Light | 0.4.18 | 843&nbsp;s | 0.4.24 | 810&nbsp;s&nbsp;(−33&nbsp;s) |
| 5060 | 8&nbsp;GB | 32&nbsp;GB | 672×1184 | 11.5&nbsp;s | Light | 0.4.18 | 707&nbsp;s | 0.4.24 | 680&nbsp;s&nbsp;(−27&nbsp;s) |
| 5070&nbsp;Ti&nbsp;Laptop | 12&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.23 | 373&nbsp;s¹³ | 0.4.23 | Thermal‑limited¹ |
| 5060 | 8&nbsp;GB | 48&nbsp;GB | 768×1344 | 14&nbsp;s | Medium | 0.3.25 | 3474&nbsp;s | 0.3.25 | Already&nbsp;optimal |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 896×1120 | 10&nbsp;s | Light | 0.3.25 | 1045&nbsp;s | 0.4.0 | 1025&nbsp;s&nbsp;(−20&nbsp;s) |
| 3060&nbsp;Laptop | 6&nbsp;GB | 40&nbsp;GB | 960×544 | 15&nbsp;s | Light⁵ | 0.4.24 | 713&nbsp;s | 0.4.24 | Thermal‑limited¹ |
| 5060 | 8&nbsp;GB | 64&nbsp;GB | 544×960 | 11.5&nbsp;s | Light | 0.4.26 | 743&nbsp;s¹³ | 0.4.26 | Already&nbsp;optimal¹² |
| 3050&nbsp;Laptop | 6&nbsp;GB | 16&nbsp;GB | 1184×672 | 10&nbsp;s | High | 0.4.23 | 5773&nbsp;s | 0.4.31 | 4300&nbsp;s&nbsp;(−1470&nbsp;s) |
| 5080 | 16&nbsp;GB | 32&nbsp;GB | 1920×1088 | 10&nbsp;s | Light | 0.4.18 | 1850&nbsp;s | 0.4.21 | 860&nbsp;s&nbsp;(−990&nbsp;s) |
| 4060&nbsp;Ti | 8&nbsp;GB | 32&nbsp;GB | 960×544 | 6&nbsp;s | Light | 0.4.27 | 241&nbsp;s | 0.4.28 | 190&nbsp;s&nbsp;(−50&nbsp;s) |
| 3060 | 12&nbsp;GB | 32&nbsp;GB | 672×1184 | 3&nbsp;s | Light | 0.4.28 | 469&nbsp;s | 0.4.28 | 430&nbsp;s&nbsp;(−40&nbsp;s) |
| 5080&nbsp;Laptop | 16&nbsp;GB | 64&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.29 | 647&nbsp;s | 0.4.29 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.30 | 1056&nbsp;s | 0.4.30 | Already&nbsp;optimal |
| 5070&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 768×1344 | 2.3&nbsp;s | Max¹² | 0.4.30 | 2195&nbsp;s | 0.4.30 | Compatibility&nbsp;level&nbsp;3¹⁴ |
| 5080&nbsp;Laptop | 16&nbsp;GB | 64&nbsp;GB | 2496×832 | 8&nbsp;s | Light⁵ | 0.4.30 | 777&nbsp;s | 0.4.30 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 9.4&nbsp;s | Light | 0.4.31 | 806&nbsp;s | 0.4.31 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.4.32 | 802&nbsp;s | 0.4.32 | Already&nbsp;optimal |
| 3050&nbsp;Laptop | 8&nbsp;GB | 40&nbsp;GB | 960×544 | 5&nbsp;s | Light | 0.4.33 | 890&nbsp;s | 0.4.33 | Already&nbsp;optimal |
| 3070&nbsp;Laptop | 8&nbsp;GB | 64&nbsp;GB | 960×544 | 5&nbsp;s | Light | 0.4.33 | 260&nbsp;s | 0.4.33 | Already&nbsp;optimal |
| 3070 | 8&nbsp;GB | 64&nbsp;GB | 736×736 | 10&nbsp;s | Light | 0.4.35 | 471&nbsp;s | 0.4.35 | Already&nbsp;optimal |
| 5060 | 8&nbsp;GB | 64&nbsp;GB | 1088×1920 | 5.9&nbsp;s | Medium | 0.4.31 | 1259&nbsp;s | 0.4.31 | Already&nbsp;optimal¹² |
| 3050 | 8&nbsp;GB | 32&nbsp;GB | 1344×768 | 5&nbsp;s | Light | 0.4.35 | 978&nbsp;s | 0.4.35 | Already&nbsp;optimal¹² |
| 4070&nbsp;Ti&nbsp;SUPER | 16&nbsp;GB | 32&nbsp;GB | 1344×768 | 12&nbsp;s | Light | 0.4.37 | 3427&nbsp;s | 0.5.3 | 2200&nbsp;s&nbsp;(−1230&nbsp;s)¹² |
| 4070 | 12&nbsp;GB | 64&nbsp;GB | 1344×768 | 8&nbsp;s | Light⁵ | 0.4.38 | 269&nbsp;s | 0.4.38 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 960×544 | 5.2&nbsp;s | Light | 0.4.38 | 348&nbsp;s | 0.4.38 | Already&nbsp;optimal |
| 5060 | 8&nbsp;GB | 64&nbsp;GB | 960×1664 | 11.5&nbsp;s | Light⁵ | 0.4.38 | 1035&nbsp;s | 0.4.38 | Already&nbsp;optimal¹² |
| 5090 | 32&nbsp;GB | 128&nbsp;GB | 1088×1920 | 10&nbsp;s | Light | 0.5.0 | 638&nbsp;s | 0.5.0 | Already&nbsp;optimal¹² |
| 3060&nbsp;Ti | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 5.2&nbsp;s | Light | 0.5.0 | 2102&nbsp;s | 0.5.0 | Already&nbsp;optimal¹²,¹⁵ |
| 5060&nbsp;Ti | 8&nbsp;GB | 16&nbsp;GB | 768×1344 | 8&nbsp;s | Light | 0.4.33 | 594&nbsp;s | 0.4.33 | Already&nbsp;optimal |
| 5060&nbsp;Ti | 16&nbsp;GB | 32&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.5.1 | 716&nbsp;s | 0.5.1 | Already&nbsp;optimal |
| 4060&nbsp;Laptop | 8&nbsp;GB | 16&nbsp;GB | 1344×768 | 10&nbsp;s | Light | 0.5.1 | 882&nbsp;s | 0.5.1 | Already&nbsp;optimal |
| M4&nbsp;Max | Unified | 128&nbsp;GB | 1344×768 | 10.1&nbsp;s | Light | 0.5.4 | 3399&nbsp;s | 0.5.4 | Already&nbsp;optimal |
| RTX&nbsp;Spark&nbsp;N1X | 110&nbsp;GB | 32&nbsp;GB | 832×640 | 15.1&nbsp;s | Light | 0.6.12 | 1298&nbsp;s | 0.6.12 | GPU&nbsp;shared¹⁶ |
| RTX&nbsp;Spark&nbsp;N1X | 110&nbsp;GB | 32&nbsp;GB | 1920×1088 | 15.1&nbsp;s | Medium | 0.6.12 | 4400&nbsp;s¹² | 0.6.12 | Power&nbsp;limited¹⁷ |
| RTX&nbsp;Spark&nbsp;N1X | 110&nbsp;GB | 32&nbsp;GB | 1344×768 | 7.3&nbsp;s | Medium | 0.6.13 | 639&nbsp;s¹² | 0.6.14 | 627&nbsp;s&nbsp;(−12&nbsp;s) |
| RTX&nbsp;Spark&nbsp;N1X | 110&nbsp;GB | 32&nbsp;GB | 1344×768 | 10.1&nbsp;s | Medium | 0.6.13 | 728&nbsp;s | 0.6.14 | 715&nbsp;s&nbsp;(−13&nbsp;s) |
| M1&nbsp;Max | Unified | 32&nbsp;GB | 1344×768 | 10.1&nbsp;s | High | 0.5.3 | 27578&nbsp;s¹² | 0.5.3 | Already&nbsp;optimal¹⁸ |

- Time: end to end, from submission until the video is saved.
- After: the estimated time, derived from the report's stage times and from complete planned runs on our test machines (RTX 3090 and RTX 5060 Ti on Windows, RTX PRO 6000 on Linux) under the reporting machine's VRAM and RAM limits, not from reruns on that machine.
- Already optimal: weight placement and transfers on this machine are already the fastest they can be; the time depends on the GPU's compute and the chosen quality level.
- ¹ Thermal-limited: the GPU spent most of the run in thermal slowdown (3070 Laptop: 229 MHz on average at 89 °C; A5000 Laptop: 882 MHz on average; 3060 Laptop: 1496 MHz on average at 81 °C), so cooling limited the time.
- ² This machine still runs the FP8 model; the estimate is for the int8 model, a one-click upgrade in the launcher since 0.3.0.
- ³ Preview: only the first pass, at half resolution. With 16 GB of RAM most of the model is read from the drive at every step (about 80 MB/s on this machine), which takes most of the time.
- ⁴ The report had no diagnostic file; the estimate replays the plan for this machine's configuration and measures the same request on our test machines.
- ⁵ Second pass after a preview: the time covers the latent upscale, the three second-pass steps and decoding, not the first pass, which made the preview in an earlier request.
- ⁶ Measured on the same machine with 0.3.22.
- ⁷ GPU shared: the graphics card never became idle during the run (23% utilization and 74 W at its lowest), so it was also doing other work and FreeVideo's steps ran at a little over half their usual speed. With the card free, the same request takes about 1000 s.
- ⁸ The drive holding the model files reads at about 60 MB/s; reading them took about 900 s of the 1504 s.
- ⁹ The drive holding the model files reads at about 80 MB/s; reading them took about 570 s of the 1676 s.
- ¹⁰ GPU shared: other programs were using about 29% of the graphics card during the run, so FreeVideo's steps ran at a little over 70% of their usual speed. With the card free, the same request takes about 7200 s.
- ¹¹ With 16 GB of RAM, the text encoder and part of the video model cannot stay in memory and are read from the drive; text encoding and the first pass, which wait on these reads, took about 460 s of the 887 s.
- ¹² Task with a reference: every step also processes the reference video, images or audio, so each step takes longer than a plain request of the same size, up to about twice as long with a long reference.
- ¹³ Preview (first pass only).
- ¹⁴ Compatibility level 3 was set manually; it is the slowest level and made each step about 2.4 times slower than the automatic setting for this request, and the GPU core clock also stayed at 900 MHz against the 1815 MHz it reached. With the level off, the same request takes about 1100 s.
- ¹⁵ The video model is read from the drive again at every step, at about 110 MB/s on this machine; the graphics card was busy for about 9–13% of the first pass and 36% of the second pass.
- ¹⁶ GPU shared: another program was using about 40% of the graphics card during the first nine of the eleven steps, which took 103–128 s each against about 55 s once it stopped. With the card free, the same request is estimated at about 830 s.
- ¹⁷ Power-limited: the system's power limit lowered the graphics card's clock as the run went on, from about 1943 MHz at 49 W in the first step to about 1285 MHz at 23 W in the ninth and about 995 MHz while decoding. Without this slowdown, the same request is estimated at about 2500 s.
- ¹⁸ From 0.6.14 this machine can use the Fast level, estimated at about 5700 s.


---

## FILE: docs/community-thanks.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/community-thanks.md)

# Thanks to the Community

We thank the community members below for introducing, listing and using FreeVideo.

## Videos and articles

- [Ding Yi](https://github.com/dingyi): [2026-10-03 产品发布视频发现汇总（Grok X 深度搜索）](https://github.com/dingyi/whatships.com/blob/main/docs/discovery/2026-10-03-zh-summary.md), introduced FreeVideo in a Chinese roundup.
- Fan Chuanyu: [FreeVideo for MiniMax H3: Local Setup, Memory Planning and ComfyUI](https://china-llm.com/blog/freevideo-minimax-h3-local-comfyui-guide), an English guide to local setup and use.
- [伊藤P｜生成AI時代の研究ノート](https://note.com/tg110): [GPUは耐えた。先に仕事が止まった。RTX 3060 Ti 8GBでMiniMax H3を試す](https://note.com/tg110/n/nb7e43615e532), a hands-on test on their own computer, in Japanese.
- [T8star-Aix](https://space.bilibili.com/385085361): [MiniMax H3低显存(8G)及新手福音！支持Mac！最易用且高质量的项目FreeVideo开源啦！](https://www.bilibili.com/video/BV1QnpP69Erv/), an introduction and getting-started video in Chinese.
- [ЮЛАВИ Studio: Музыка с чистого листа](https://www.youtube.com/@yulavi-singer): [Видео в нейросетях БЕСПЛАТНО | MiniMax H3 на 8GB VRAM! FreeVideo: установка и тест](https://www.youtube.com/watch?v=YWRwlfcW_qc), an installation and test video.
- [Satyajit Ghana](https://ai.thesatyajit.com/): [FreeVideo: a 66 GB video transformer on an 8 GB card, and the planner that decides where every block lives](https://ai.thesatyajit.com/articles/freevideo-minimax-h3), a long article on how FreeVideo plans a run and manages GPU memory.
- [푸른영혼의 별](https://myoungsoo7.github.io/): [FreeVideo — 8GB VRAM 그래픽카드로 MiniMax H3 영상 생성하기 (그리고 한국 사용자가 먼저 봐야 할 라이선스 한 줄)](https://myoungsoo7.github.io/2026/10/06/freevideo-minimax-h3-8gb-vram-local-video/), FreeVideo's requirements, speed and licensing, in Korean.
- [Katelya](https://github.com/katelya77): [把视频生成搬回自己的电脑：从 FreeVideo 拆解自适应推理与硬件边界](https://github.com/katelya77/katelyatop-blog/blob/master/src/content/posts/freevideo-local-video-generation-hardware-boundaries.md), a long article in Chinese on FreeVideo's adaptive inference.

*Videos and articles are listed in order of publication.*

## Listings and packaging

- [WildAi](https://github.com/wildminder): [Awesome MiniMax-H3](https://github.com/wildminder/awesome-minimax-H3), listed FreeVideo in this list.
- [Portable-Linux-Apps](https://github.com/Portable-Linux-Apps): [apps/freevideo](https://github.com/Portable-Linux-Apps/Portable-Linux-Apps.github.io/blob/main/apps/freevideo), listed FreeVideo in the AppImage software catalogue.
- [ivan-hc](https://github.com/ivan-hc): [AM / freevideo](https://github.com/ivan-hc/AM/blob/main/programs/x86_64/freevideo), the FreeVideo install and update script for the AM package manager.

## Used in other projects

- [T8mars](https://github.com/T8mars): [comfyui-minimax-h3-audio-T8](https://github.com/T8mars/comfyui-minimax-h3-audio-T8), added a loader that works with FreeVideo to their ComfyUI nodes.
- [Hansimov](https://github.com/Hansimov): [VDN8 community sampling adapter](https://github.com/Hansimov/vflash/blob/main/docs/guide/community-vdn.md), described how to use FreeVideo in their adaptation guide.

## Diagnostic reports

We thank everyone who sent a diagnostic report. Each row in the [community results](community-results.md) comes from one of these reports, which helped make FreeVideo faster on more graphics cards.


---

## FILE: docs/execution-planning.md (https://github.com/FlashML-org/FreeVideo/blob/HEAD/docs/execution-planning.md)

# FreeVideo Adaptive Execution Planner

The FreeVideo Adaptive Execution Planner computes an execution plan for every request from live device and host measurements. The plan fixes the residency of the 50 transformer blocks across VRAM, pinned host memory and disk, the transfer schedule, the int8 or FP8 GEMM path, the attention backend and head grouping, activation staging and the VAE decoder placement, within the measured VRAM and host-memory budgets.

## Inputs

| Input | Source | Determines |
| --- | --- | --- |
| Architecture and compute capability | CUDA device properties | Weight format, GEMM path and kernel set |
| Free VRAM | `torch.cuda.mem_get_info` | VRAM budget |
| Available host memory | OS memory counters, cgroup v1/v2 limit, Windows commit headroom | Host-memory budget |
| Attention backends | On-device kernel probes | Backend selection |
| Request geometry | Width, height and frames, converted to video tokens | Activation working set |

VRAM budget: free VRAM minus a reserve of 2.5% of free VRAM, clamped to 0.5–1 GiB (0.25 GiB on Windows). Host-memory budget: available memory minus a reserve of 5% of available memory, clamped to 1–4 GiB (2–4 GiB on Windows); the Windows page file is not counted.

## Plan

| Component | Decision |
| --- | --- |
| Transformer residency | Resident blocks = min(⌊(VRAM budget − activation working set) / 432.5 MB⌋, 44), then reduced to max(8, blocks the host budget cannot retain). Non-resident blocks are kept in host memory, pinned up to 22 GB, and copied to the GPU at every step. |
| Disk streaming | Enabled when the non-resident blocks plus host working memory exceed the host budget. A bounded pinned subset is retained; the remaining blocks are re-read from disk at every step. |
| Transfer schedule | Two transfer slots (prefetch) at a VRAM budget of 14 GiB or more, with lower thresholds on Windows Blackwell and Ampere; one slot otherwise. |
| Attention | The first backend that passes its probe, in the order SageAttention 2, PyTorch flash attention, cuDNN, FlashAttention 2, FlashAttention 4. Head group: the widest of 16, 8 and 4 heads whose measured activation footprint at the request's token count fits the VRAM budget. |
| Activation staging | Below a 10 GiB VRAM budget, the residual stream and attention outputs are staged in host buffers when the on-device activation path does not fit. |
| GEMM | int8 on GeForce Ada and Blackwell and on every Ampere card: activations get the cache's ConvRot rotation and one int8 scale per row. Other cards use FP8: per-tensor scales on Blackwell (SM120), per-channel scales on Ada (SM89) and Hopper (SM90). |
| Chunking | Feed-forward chunk 2048, projection chunk 1024, window batch 4 (1 on Ampere). |
| Text encoder | Runs in a separate process that exits before the transformer is loaded. |
| VAE decoder | Below a 20 GiB VRAM budget, ⌊(VRAM budget − 3.25 GiB) / 268.6 MB⌋ of its 36 blocks stay resident and the rest are streamed; temporal clips are decoded in sequence with the original tiles and blending. |
| Two-pass sampling | The half-resolution first pass gets its own plan. Blocks copied to the GPU during that pass stay cached in free VRAM, up to the first-pass resident count on Linux and up to the live VRAM budget on Windows. |

## Runtime adaptation

- Budgets are re-measured before every request.
- Completed requests store their timings and memory peaks in `resource-history.sqlite3`. The history drives the forecast shown before each request and `./freevideo predict`, and replaces the placement when an alternative is predicted to be at least 2% faster.
- If the minimum working set does not fit, the request is rejected before the transformer loads, and the shortfall is reported.
- An out-of-memory failure is retried in a fresh process, up to two times (`--resource-retries`). The retry first lightens the placement. When nothing is left to move, it shrinks the compute partitions: head group, feed-forward and projection chunks, then attention outputs staged in host memory. This can change the floating-point reduction order. A failure in the second pass of two-pass sampling reuses the completed first pass, and the next identical request on the same machine starts with the partitions that completed.
- `./freevideo optimize` searches compute settings for the machine and applies them only after a complete validation run.

## Inspection

- `./freevideo plan --vram-gib 8 --ram-gib 16` prints the plan for the given capacities without loading weights; `--width`, `--height` and `--seconds` set the request.
- `<output>.request.json` records the measured hardware, the selected plan with a note for each decision, stage timings and memory peaks.

## Capacity grid

Measured on NVIDIA H200 for every combination of 8–32 GiB of VRAM and 8–32 GiB of host memory, with VRAM capped by an MPS client limit and host memory by a cgroup limit without swap. Request: 1344 × 768, 243 frames, 8 single-pass full-resolution steps (`--no-two-pass`).

Seconds per sampling step:

![Seconds per sampling step](assets/capacity-speed.en.svg)

Transformer blocks kept in VRAM; a dot marks configurations where the remaining blocks are read from disk at every step:

![Transformer blocks kept in VRAM](assets/capacity-blocks.en.svg)

## End to end

Windows, 1344 × 768, 10 seconds, two-pass sampling:

| GPU | VRAM + RAM | Time |
| --- | --- | --- |
| GeForce RTX 5090 | 32 GB + 64 GB | 122 s |
| GeForce RTX 5060 Ti | 16 GB + 32 GB | 486 s |
| GeForce RTX 4060 Ti | 16 GB + 32 GB | 558 s |
| GeForce RTX 4060 Ti (community report) | 8 GB + 64 GB | 603 s |


---

# ISSUES / PRs of FlashML-org/FreeVideo


### #1216 [open] ISSUE: [Feature request] /drama: allow audio references with "use": "original" in a shot (lip-sync to a given voice file)
URL: https://github.com/FlashML-org/FreeVideo/issues/1216  by vietdevne 2026-11-28T11:07:12Z

### FreeVideo version / FreeVideo 版本

_No response_

### What happened? / 遇到了什么问题？

What could be better?

Since 0.6.1, POST /jobs (kind "generate") accepts an audio entry in media.references with
"use": "original", so the video lip-syncs to the given voice file and keeps it as the soundtrack.
This is great for dialogue recorded elsewhere (e.g. a TTS service).

The film API does not accept it yet. In POST /drama/plan, a shot like

  {"id": "s1", "seconds": 4, "action": "...", "cast": ["a"],
   "lines": [{"who": "a", "line": "...", "how": "says softly"}],
   "references": [{"file": "line01.wav", "use": "original"}]}

fails with: invalid_screenplay /screenplay/shots/0/references/0/? must be absent (unknown field).
Without "use", the file is only a voice reference, so the film cannot keep the exact recorded line.

Request: let /drama shots (at least non-continuing ones) carry one audio reference with
"use": "original" (and optional range), with the same rules as /jobs: the shot is as long as the
selected audio. Then a whole film with recorded dialogue can use the built-in director, takes
and joining, instead of shooting the dialogue shots one by one outside /drama.

Version: FreeVideo 2026.9.16.41444, Windows 11.

### Diagnostic report / 诊断报告

N/A


### #1215 [open] ISSUE: [Feedback] 提示词增强
URL: https://github.com/FlashML-org/FreeVideo/issues/1215  by dreamweaverbreaker 2026-11-28T08:47:25Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Prompt enhancement
Error: 改写结果不完整，请重试或使用原文。

### Version / 版本

v0.6.15

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Prompt enhancement","kind":null,"tail":"改写结果不完整，请重试或使用原文。","facts":{"check":"missing_picture_reference","check_category":"media","check_retried":true}}

> **Comment by KBRASK (2026-11-28T09:57:18Z):**

Thank you for reporting this. After a prompt rewrite, FreeVideo checks that the rewritten text still refers to every reference picture in the exact form <Picture N>; the rewriting model sometimes writes the reference in another form, such as (Picture N), or leaves one out, and the check then reports the rewrite as incomplete. On the same computer, the same prompt and pictures usually give the same rewrite again, so retrying does not help in this case. To generate now, choose "Original" on the enhancement bar, or turn off prompt enhancement, and generate with your own prompt; the reference pictures are still passed to the generation.



### #1214 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1214  by mahalebhavesh25 2026-11-28T08:17:59Z

### What could be better? / 哪里可以做得更好？

There is a big problem in the audio language. The prompt strictly mentions english but the audio is in unknown language. Also, the fast mode is not really fast, it claimed video in 6 mins but delivered in 15 mins for 10 sec video

### Version / 版本

v0.6.15

### Diagnostic info (optional) / 诊断信息（选填）

{"version":"2026.11.27.221212","status":"complete","request_seconds":1554.7559024000002,"geometry":{"width":1344,"height":768,"frames":277,"fps":24,"video_tokens":82656,"reference_video_tokens":2997,"reference_audio_tokens":0,"text_tokens":3024},"config":{"resident_blocks":0,"pin_host_gb":0,"head_chunk":4,"window_batch":4,"head_parallelism":1,"ff_chunk":2048,"projection_chunk":1024,"query_chunk":0,"fp8_linears":363,"resident_weight_bytes":0,"pinned_model_bytes":0,"pinned_host_allocated_bytes":0,"steps":8,"adaln_optional_blocks":50,"adaln_downloaded_files":0,"adaln_downloaded_bytes":0,"prefetch":false,"stream_weights":true,"attention_cpu_outputs":false,"grouped_attention_outputs":false,"residual_offload":true,"fp8_ff_recompute":false,"inference_kernels":true,"window_varlen":false,"reuse_block_outputs":true,"preload_host":false,"pin_host_weights":false,"cache_refined_text":true,"task":"ref2va","linear_compute":"int8","adaln_mode":"optional-model-asset","fp8_scale_granularity":"rowwise","precision":"int8","attention":"sage2/sage2"},"stages":[{"stage":"text_encoding","seconds":83.51899490000505},{"stage":"model_load","seconds":30.468679400000838},{"stage":"sampling","seconds":1118.8971987999976},{"stage":"latent_save","seconds":0.11597970000002533},{"stage":"decode_save","seconds":223.9768686000025},{"stage":"other_or_unmeasured","seconds":97.77818099999422}],"budgets":{"gpu":7068450816,"ram":4808929280},"memory":{"physical_available_bytes":6697205760,"commit_available_bytes":14688641024,"available_bytes":6697205760,"total_bytes":16873545728},"errors":[],"hardware":{"gpu_name":"NVIDIA GeForce RTX 5060 Laptop GPU","vram_total":8518041600,"ram_total":16873545728,"system":"Windows","torch_version":"2.13.0+cu130","cuda_version":"13.0","driver_version":"616.92","architecture":"blackwell-rtx"}}

> **Comment by 0containertoby12 (2026-11-28T08:22:05Z):**

[AudioFastModeDiag](https://www.mediafire.com/file/27w7mxnmt5auaoi/Pictures.zip/file)
I’ve seen this audio-language mismatch plus fast mode taking ~15 min for a 10-sec clip on Windows. I’d verify the language prompt/audio tokenizer path and profile sampling vs decode; running the attached diagnostic will log driver/CUDA and stage timings so we can pinpoint the slowdown and wrong-language output. It should make the mismatch and extra minutes much easier to isolate.


> **Comment by KBRASK (2026-11-28T09:26:33Z):**

Thank you for the feedback and the run details. On the audio: when a prompt asks for English speech but does not give the words, the model can produce speech-like sound without recognizable words, which sounds like an unknown language; when the words are written out, it speaks them in English. The sound of the reference video was not used in this generation. Please write the exact words in the prompt, for example says, "Hello everyone, welcome back to my channel.", or put them on a separate line with a time and a speaker, such as 0:02 Anna: Hello everyone. If the words were already written out in your prompt, please reply with the prompt. On the time: this run used the Light level with two passes of 8 and 3 steps, not the Fast level, because on this computer the Fast level allows at most 5 s at 1344x768, so the Fast level is not available for a 10 s request, and this run used the Light level. For this request the Light level estimated about 37 minutes (2,254 s), and the run took about 26 minutes (1,554.8 s). If you would like us to look further at the speed, please attach the full diagnostic report file.


> **Comment by mahalebhavesh25 (2026-11-28T10:54:33Z):**

I will try to get the diagnostic report as soon as I can. Could you elaborate how can I generate full diagnostic report? While starting this thread I think I shared the full report only. Correct me if I am wrong.



### #1213 [open] ISSUE: [Feedback] 生成未完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1213  by wwedeluo 2026-11-28T06:59:03Z

### What could be better? / 哪里可以做得更好？

FreeVideo reported a generation failure.
Stage: execution
Error: Generation requires a dispatched executor
Kind: unknown

### Version / 版本

v0.6.15

### Diagnostic info (optional) / 诊断信息（选填）

{"kind":"unknown","stage":"execution","exception_type":"RuntimeError","known_error":null,"nodes":["FreeVideoGenerate"],"settings":{"width":1344,"height":768,"seconds":10,"seed":4216151697899070,"two_pass":true,"base_steps":8,"refine_steps":3,"step_cache":"none","upscale_factor":"2"},"progress":{},"engine_exit":{"at":1794220733.0934656,"code_text":"0x40010004","exit_code":1073807364,"exit_code_unsigned":1073807364,"exit_class":"error_exit","suspected_system_kill":false,"memory":{"physical_available_bytes":115459723264,"commit_available_bytes":124353585152,"commit_limit_bytes":147281731584,"total_bytes":137081184256}},"frontend_version":"1.51.10","tail":"Generation requires a dispatched executor\n"}

> **Comment by KBRASK (2026-11-28T08:25:31Z):**

Thank you for sending the failure details. This is a known problem: in this session, FreeVideo's multi-GPU dispatch did not start, so the generation was not handed to a graphics card; the exit code included in the details is an older record from November 9 and is unrelated to this failure. This state lasts until FreeVideo restarts, so please exit FreeVideo first (choose "Exit" when closing the launcher window, not "Run in background") and open it again. If the error appears again, create a user environment variable named FREEVIDEO_MULTI_GPU with the value 0 in Windows Environment Variables, then exit and open FreeVideo again in the same way; it will then use one graphics card. If it still fails, please attach a diagnostic report from Export report in the launcher.



### #1212 [open] ISSUE: [启动器] 暂时未能完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1212  by zhaohuijun 2026-11-28T03:24:31Z

### FreeVideo version / FreeVideo 版本

v0.6.15

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
RuntimeError: Failed step: python-npmmirror-inherited (exit code 1)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.15
Display adapters: Intel Corporation Raptor Lake-S UHD Graphics; NVIDIA Corporation AD106M [GeForce RTX 4070 Max-Q / Mobile]

RuntimeError: Failed step: python-npmmirror-inherited (exit code 1)
Retained step log: <PATH>/02-python-npmmirror-inherited.log
Full step output:
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------     0 B/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 1.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 9.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 17.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 25.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 33.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 41.89 KiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (
...
----- 55.69 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 56.56 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 57.28 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 57.91 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 58.65 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 59.34 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 60.06 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 60.79 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 61.53 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 62.13 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 62.85 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 63.31 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 63.89 MiB/64.25 MiB
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 63.96 MiB/64.25 MiB                                               error: Failed to install cpython-3.12.3-linux-x86_64-gnu
  Caused by: Failed to extract archive: cpython-3.12.<token>.tar.gz
  Caused by: I/O operation failed during extraction
  Caused by: failed to unpack `<PATH>/gs5430`
  Caused by: No such file or directory (os error 2)
cpython-3.12.3-linux-x86_64-gnu (download) ------------------------------ 63.98 MiB/64.25 MiB

> **Comment by KBRASK (2026-11-28T05:21:37Z):**

Thank you for sending the launcher message. The Python package was downloading normally, but unpacking it into the install location failed at one symbolic link in the package; the same package installs normally on a regular Linux file system. Please choose "Back" in the launcher until "Install location" appears, change it to a folder in your home directory, and install again. If it still fails, please reply with the output of df -T for the original install location, and attach the retained step log 02-python-npmmirror-inherited.log.



### #1211 [open] ISSUE: [启动器] 暂时未能完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1211  by uc100200-c 2026-11-28T03:01:34Z

### FreeVideo version / FreeVideo 版本

v0.4.35

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
This offline installation already runs the engine this launcher carries

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.4.35
Display adapters: AMD Radeon(TM) Graphics; NVIDIA GeForce RTX 3060

This offline installation already runs the engine this launcher carries

> **Comment by KBRASK (2026-11-28T04:51:12Z):**

Thank you for sending the launcher message. The launcher compares the program files when it opens and again when the update is chosen; here the two comparisons disagreed, so the update offer was a false alarm, and this installation already runs the engine this launcher carries. Close the launcher, open it again and choose "Launch FreeVideo". If the update offer appears again, choose "Launch current version" instead of "Update & launch".



### #1210 [open] ISSUE: [Feedback] 生成未完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1210  by sinfree 2026-11-28T02:49:30Z

### What could be better? / 哪里可以做得更好？

FreeVideo reported a generation failure.
Stage: execution
Error: Generation requires a dispatched executor
Kind: unknown

### Version / 版本

v0.6.15

### Diagnostic info (optional) / 诊断信息（选填）

{"kind":"unknown","stage":"execution","exception_type":"RuntimeError","known_error":null,"nodes":["FreeVideoGenerate"],"settings":{"width":1344,"height":768,"seconds":10,"seed":2026090903,"two_pass":true,"base_steps":8,"refine_steps":3,"step_cache":"none","upscale_factor":"2"},"progress":{},"frontend_version":"1.51.10","tail":"Generation requires a dispatched executor\n"}

> **Comment by KBRASK (2026-11-28T03:49:50Z):**

Thank you for sending the failure details. This is a known problem: in this session, FreeVideo's multi-GPU dispatch did not start, so the generation was not handed to a graphics card. This state lasts until FreeVideo restarts, so please exit FreeVideo first (choose "Exit" when closing the launcher window, not "Run in background") and open it again. If the error appears again, create a user environment variable named FREEVIDEO_MULTI_GPU with the value 0 in Windows Environment Variables, then exit and open FreeVideo again in the same way; it will then use one graphics card. If it still fails, please attach a diagnostic report from Export report in the launcher.



### #1209 [open] ISSUE: [Feedback] 图像模型下载
URL: https://github.com/FlashML-org/FreeVideo/issues/1209  by fa671130060-lang 2026-11-28T01:10:36Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Image model download
Error: H3 Image 下载失败：图像模型文件不完整或已损坏，请在启动器里修复。请重试；仍然失败时，请点击“提交问题”。

### Version / 版本

v0.6.15

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Image model download","kind":null,"tail":"H3 Image 下载失败：图像模型文件不完整或已损坏，请在启动器里修复。请重试；仍然失败时，请点击“提交问题”。"}

> **Comment by KBRASK (2026-11-28T02:48:24Z):**

Thank you for reporting this. On the China source, one file of H3 Image, its LICENSE, was changed on November 11 and no longer matches the checksum FreeVideo records, so a computer that can reach only the China source cannot finish downloading H3 Image; retrying or repairing in the launcher does not change this. Open Settings > Downloads and choose "Test again". If "International mirror" or "International source" shows a speed, choose it under "Download source", then download H3 Image again in the Image view; the files already downloaded are kept. If both show "Unavailable", set up a proxy first, choose "Proxy only" under "Connection", and try again.



### #1208 [closed] PR: Release 0.6.15
URL: https://github.com/FlashML-org/FreeVideo/pull/1208  by KBRASK 2026-11-27T22:06:57Z

0.6.15 adds a quick 1.5x upscale with how long it takes on this computer, keeps the workflow across the reload after an update, lets you delete image models in Settings and save an image as a character, counts prompt rewriting in the time beside Generate, makes Windows setup recover from a broken Python and from some download failures, and fixes many editor and interface details.

- Upscaling: the Creations upscale menu offers a quick 1.5x upscale by the small frame-wise model next to the detailed one, and says about how long each takes on this computer. Finished 4K and quick upscales, including those made with earlier versions, are listed in Creations and the Edit view again.
- Time estimates: the time beside Generate now counts prompt rewriting when enhancement is on (only for the first task of a queued batch), estimates the second pass of a 1.5x plan from its own size, and corrects the decoding part from this computer's earlier runs at the same frame size.
- Prompt enhancement: while a task runs, the Rewrite key waits and says which task it waits for and about when; a queued video that asked for a rewrite is rewritten when its turn comes.
- Create panel: while a video is being made, a creation picked in the right column stays on the stage with its actions, so several previews can be sent to the second pass one after another; the Generating item at the top of the column returns to the progress. With the assistant interface on, an image in the image view or Creations can be saved as a character and then mentioned with @ in the prompt.
- Updates and Settings: after an update reloads the page, the workflow in front and a prompt not yet sent are kept (held briefly in the local FreeVideo service's memory only). Image models can be deleted in Settings without closing FreeVideo; a model that is in use or queued cannot be deleted, and the files a stopped download left behind can be.
- Creations and the editor: deleting a creation that an edit uses names the edit and offers to open it at that part. In the editor, Escape belongs to the text on the picture and the delete-import question, coming back from another view shows the edit as it was left, the library is read again on return, and the opening zoom fits the time area beside the track heads.
- Windows setup: when the isolated Python's python.exe is missing or does not answer, it is installed again once; on some networks where Windows cannot check a certificate's revocation, downloads and Git downloads try again automatically and still verify the certificate; FreeVideo's own Git downloads are no longer sent to an unreachable mirror by this computer's Git address rewrite rules; when a dependency folder was not downloaded by FreeVideo, the error says so and names the folder; on Windows on Arm, installing and repairing no longer stop at the Visual C++ runtime check.
- Launcher: on a computer FreeVideo cannot run on, the card no longer offers "Report a problem"; the setup step names say what each step does, and the step names, the status title and the address fit a narrow window. When a download through the proxy set on this computer fails, the message says where that proxy setting comes from; when that proxy does not respond, the card first asks you to start the program that provides it and then retry.
- Mac: a second pass that hits the GPU watchdog is tried again instead of failing, and the prompt model runs on the Apple GPU when its memory fits.
- Memory: when FreeVideo already holds the text encoder on the graphics card, generating an image with reference images counts that memory, so on cards with less memory it is no longer refused before it starts.
- Interface text: names in quotes and level names are kept on one line where they are shown; more Chinese messages use formal written wording; several English sentences were made consistent; failure messages without a "Report a problem" button name the next step; the image view shows the whole message of a failed model download, and layout limits no longer offer a report.
- Other fixes: the compare divider in the Create panel follows the drag instead of staying in the middle; the compatibility notice says plainly when a reply cannot be read and keeps only the latest failure line; a Windows crash card names the module Windows recorded; the offline package guide names each package by its file name in every language.



### #1207 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1207  by volkogonolegsefirotic-afk 2026-11-27T21:45:14Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 1 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: AMD Radeon RX 6700 XT

Setup inspection failed (plan, exit 1, 1 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-27T21:44:14Z", "platform": "Windows-10-10.0.19045-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.13.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-27T23:14:08Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon RX 6700 XT and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards (8 GB of video memory or more is recommended), and, as a preview, on Apple silicon Macs.



### #1206 [open] ISSUE: [Feedback] Assistant connection
URL: https://github.com/FlashML-org/FreeVideo/issues/1206  by essamsalah 2026-11-27T21:24:27Z

### What could be better? / 哪里可以做得更好？

FreeVideo showed an error.
Where: Assistant connection
Error: Could not get address. Retry (HTTP 403 origin_forbidden, opened by localhost)
Kind: origin_forbidden

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Assistant connection","kind":"origin_forbidden","tail":"Could not get address. Retry (HTTP 403 origin_forbidden, opened by localhost)"}

> **Comment by KBRASK (2026-11-27T23:44:43Z):**

Thank you for reporting this. The error means that FreeVideo did not recognize the address the page was opened at as its own; this happens when the page is opened at a different port from the one FreeVideo is listening on, so FreeVideo refused the request. Open FreeVideo at the address the launcher shows, with the same port; if the page is reached through an SSH tunnel or port forwarding, use the same port number locally as on the computer running FreeVideo. If the error remains, please reply with the full address shown in the browser's address bar, including the port.



### #1205 [open] ISSUE: 希望能出一个arm的专属版不然现在的版本就算买安装上也全是叉八六的组件非常多的组件需要替换要不就是大量性能损失或者报错管折腾很折腾
URL: https://github.com/FlashML-org/FreeVideo/issues/1205  by fe3o48 2026-11-27T20:41:54Z

### Diagnostic report / 诊断报告

[freevideo-diagnostics-20261127T204048Z.zip](https://github.com/user-attachments/files/35530430/freevideo-diagnostics-20261127T204048Z.zip)

### FreeVideo version / FreeVideo 版本

_No response_

### Graphics card and memory / 显卡和内存

_No response_

### Video size and length / 视频尺寸和时长

_No response_

### Time taken / 用时

_No response_

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-28T04:20:36Z):**

Thank you for sending the diagnostics. They show a native ARM64 installation: Python, PyTorch, torchvision, torchaudio and Triton are all ARM64 builds, and only the launcher itself and the helper Python of the setup script are x64, neither of which takes part in generation. The installation did not finish because it stopped at the runtime library check described in #1204. FreeVideo 0.6.15 fixes that check on Windows on Arm; after updating, choose "Retry installation" in the launcher, and setup continues without replacing any components.



### #1204 [open] ISSUE: arm
URL: https://github.com/FlashML-org/FreeVideo/issues/1204  by fe3o48 2026-11-27T20:16:48Z

### What could be better? / 哪里可以做得更好？

一、必须改的：1 处，不改会出事
vc_runtime.py 在这一版新增了“在载入 PyTorch 之前先检查 VC++ 运行库”，但它无条件检查 VCRUNTIME140_1.dll。这个文件在 Windows ARM64 上只有 x64 版本，因为 ARM64 的 CRT 把 __CxxFrameHandler4 合并进 VCRUNTIME140.dll 了，而 System32 目录里那个副本是 x64 的。原生 ARM 进程按文件名去加载它，就会报 error 193（不是有效的 Win32 应用程序）​。

上一版有 ARM 排除逻辑，0.6.14 重写这段时漏掉了。现在之所以还能启动，是因为运气好没有走到那条路径；一旦触发，就会卡在安装或启动这一步，而且报错信息完全指不到真正的原因。改法很简单：ARM64 下只检查另外两个 dll。

二、性能：3 处，现在白白浪费 17–20%
项	现值	问题
RESIDENT_VRAM_CEILING	44	50 层权重全常驻只要 21.6 GB，你这张卡有 110 GiB。按 44 层算，每走一步都要从内存往显存搬 7 层，约 3.0 GB，采样时显存空着、等内存供数
pin 下限	无	全常驻时 50 - resident 变成负数，会抛 Residency and host cache budgets cannot be negative。这是改第一项之后必然踩到的问题
decoder_linear_cache	未含 blackwell-unified	解码被推到 FP32 路径：解码 19.2 s 对 15.4 s，还多占 4.2 GiB 常驻。你这颗 GB10 跟 blackwell-rtx 同代 SM、cast 语义一致，只是架构名不在名单里

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-28T01:47:20Z):**

Thank you for the detailed analysis. The VC++ runtime check is new in FreeVideo 0.6.14, and on Windows on Arm it also tried to load VCRUNTIME140_1.dll, whose only copy there is x64, so a new installation or a repair on Windows on Arm stopped at the setup check. FreeVideo 0.6.15 checks only the runtime libraries that the ARM64 build of PyTorch needs on Windows on Arm, so installation and repair no longer stop at this step; after updating, choose "Retry installation" in the launcher to finish an installation that stopped there. On the performance points: on Windows, the resident layer limit and the decoder cache do not currently account for unified-memory graphics such as this one, but moving weights and decoding take a small part of the whole generation, so by our estimate the overall effect is a few percent rather than 17 to 20 percent. The pinned-memory budget cannot become negative in the current code.



### #1203 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1203  by fe3o48 2026-11-27T20:07:07Z

### What could be better? / 哪里可以做得更好？

图片预览打开以后是灰色的图片打开全屏是灰色的界面不显示

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

<img width="1600" height="911" alt="image" src="https://github.com/user-attachments/assets/469aee6c-5475-483a-8f54-0d6f444d25a5" />

> **Comment by KBRASK (2026-11-27T22:43:34Z):**

Thank you for the screenshot. In it, the picture cards in Autopilot and the large picture on the right show correctly, so the picture files themselves are fine; the grey view appears in the picture viewer that opens after choosing "Open" or the picture. Meanwhile, choose "Download" next to "Open" on the card to save the same picture and view it in the system's picture viewer. To find the cause, please attach a screenshot taken while the viewer shows grey, and the name and version of the browser.



### #1202 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1202  by fe3o48 2026-11-27T20:02:14Z

### What could be better? / 哪里可以做得更好？

进度条和预估时间与实际不符合

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

<img width="910" height="1352" alt="image" src="https://github.com/user-attachments/assets/95f6e934-d222-440f-a5a4-5ecd615a5824" />

> **Comment by KBRASK (2026-11-28T00:15:32Z):**

Thank you for the feedback and the screenshot. With the Medium level, the remaining-time estimate counted each second-pass step as if the result were upscaled 2 times instead of the 1.5 times this level uses, so the estimate rose during the first pass and stayed too high for the first two second-pass steps; your screenshot was taken in that stretch. Replaying the data from your earlier report, FreeVideo 0.6.14 showed 562 s remaining at that point, while the run actually needed 291 s. FreeVideo 0.6.15 estimates the second-pass steps from the size this level actually uses, and estimates decoding from this computer's earlier runs at the same frame size.



### #1201 [open] ISSUE: win on arm
URL: https://github.com/FlashML-org/FreeVideo/issues/1201  by fe3o48 2026-11-27T19:45:49Z

### What could be better? / 哪里可以做得更好？

生成音乐时间未看到提示词输入和生成按钮

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

{"hardware":{"gpu_name":"NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)","vram_total":103079215104,"ram_total":32742236160,"system":"Windows","torch_version":"","cuda_version":"","driver_version":"621.24","architecture":"blackwell-unified"}}

> **Comment by KBRASK (2026-11-27T22:12:59Z):**

Thank you for the feedback. In the Edit view, the music options ("Style", "Mood" and the "Describe the music" box) appear only after a range is selected on the second audio track. Drag a range on an empty part of the "Audio 2" track; if there is no "Audio 2" track, choose "+ Audio track" first. Choose one option under "Style" and one under "Mood", or write a description in the "Describe the music" box. Then choose "Generate". The first generation downloads the music model first. If they still do not appear after these steps, please attach a screenshot of the whole Edit view window.



### #1200 [open] ISSUE: Make MCP registration independent of FreeVideo build versions
URL: https://github.com/FlashML-org/FreeVideo/issues/1200  by s1y-f0x 2026-11-27T19:12:12Z

### What would you like to do? / 希望实现什么？

Hi! I'd like to suggest an improvement to make the FreeVideo MCP connection more reliable across updates.

### Current behavior

After updating FreeVideo to 0.6.14, my existing MCP connection stopped working.

I found that the MCP registration still referenced the previous build directory, which no longer existed after the update.

The official Connection steps use a version-specific `installed_entry.py` path and `PYTHONPATH`.

As a result, updating FreeVideo can invalidate an otherwise correctly configured MCP connection.

### Tested alternative

I found that the unified Python environment can resolve the current FreeVideo engine module through the editable package binding maintained by the launcher.

I successfully tested the following command:

```powershell
E:\FreeVideo\FreeVideo-engine\envs\unified\Scripts\python.exe -B -P -m freevideo_engine.agent.mcp --root E:\FreeVideo\FreeVideo-engine
```

No build-specific `PYTHONPATH` is required.

After updating the MCP registration, I successfully verified:

- MCP initialization.
- Discovery of 64 MCP tools.
- `freevideo_status` returning `ready: true`.
- Message exchange between Codex and FreeVideo Autopilot.

No FreeVideo source code modifications were required.

### Suggestion

Consider using a build-independent MCP registration in the official Connection steps, or providing a stable MCP entry point maintained by the launcher.

Ideally, MCP clients should not require reconfiguration after each FreeVideo update.

The alternative registration is working in my current installation. Its behavior across future updates has not yet been verified.

Thank you for your continued work on FreeVideo!

> **Comment by KBRASK (2026-11-27T21:42:31Z):**

Thank you for the suggestion and for testing the alternative registration. The official Connection command points to the source folder of one build and sets PYTHONPATH to it, so after an update that folder is replaced and the registration stops working. The command you tested stays valid across updates: the unified environment keeps its location, the launcher points its freevideo_engine package to the new build when it finishes an update, and removing old builds does not delete the source the environment uses. After an update, start FreeVideo once and let it finish updating before using the MCP connection, so that this link is refreshed.



### #1199 [open] ISSUE: [Launcher] This GPU is not supported
URL: https://github.com/FlashML-org/FreeVideo/issues/1199  by mikac97 2026-11-27T19:09:46Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
[CUDA_ARCH_UNSUPPORTED] NVIDIA GeForce GTX 1080 (SM61; compute capability 6.1; driver 582.78). FreeVideo requires native BF16 GPU computation (compute capability 8.0 / SM80 or newer). This GPU is below that hardware requirement; a driver update cannot add it.

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA GeForce GTX 1080

[CUDA_ARCH_UNSUPPORTED] NVIDIA GeForce GTX 1080 (SM61; compute capability 6.1; driver 582.78). FreeVideo requires native BF16 GPU computation (compute capability 8.0 / SM80 or newer). This GPU is below that hardware requirement; a driver update cannot add it.

> **Comment by KBRASK (2026-11-27T21:11:57Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because this computer's NVIDIA GeForce GTX 1080 has compute capability 6.1, below the 8.0 that FreeVideo requires, and a driver update cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards (8 GB of video memory or more is recommended), and, as a preview, on Apple silicon Macs.



### #1198 [open] ISSUE: [Feedback] 生成未完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1198  by sam528300-lab 2026-11-27T19:08:13Z

### What could be better? / 哪里可以做得更好？

FreeVideo reported a generation failure.
Stage: execution
Error: Generation requires a dispatched executor
Kind: unknown

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

{"kind":"unknown","stage":"execution","exception_type":"RuntimeError","known_error":null,"nodes":["FreeVideoGenerate"],"settings":{"width":832,"height":640,"seconds":10,"seed":7248248441110127,"two_pass":false,"base_steps":8,"refine_steps":3,"step_cache":"none","upscale_factor":"2"},"progress":{},"engine_exit":{"at":1795255789.2104053,"code_text":"0x40010004","exit_code":1073807364,"exit_code_unsigned":1073807364,"exit_class":"error_exit","suspected_system_kill":false,"memory":{"physical_available_bytes":47068356608,"commit_available_bytes":39474634752,"commit_limit_bytes":72915800064,"total_bytes":68620832768}},"frontend_version":"1.51.10","tail":"Generation requires a dispatched executor\n"}

> **Comment by KBRASK (2026-11-27T20:41:18Z):**

Thank you for sending the failure details. In this session, FreeVideo's multi-GPU dispatch did not start, so the generation was not handed to a graphics card; the exit code included in the details is an older record from November 21 and is unrelated to this failure. This state lasts until FreeVideo restarts, so please exit FreeVideo first (choose "Exit" when closing the launcher window, not "Run in background") and open it again. If the error appears again, create a user environment variable named FREEVIDEO_MULTI_GPU with the value 0 in Windows Environment Variables, then exit and open FreeVideo again in the same way; it will then use one graphics card. If it still fails, please attach a diagnostic report from Export report in the launcher.



### #1197 [open] ISSUE: 4K Upscale results disappear from Creations and have missing thumbnails due to a missing success flag
URL: https://github.com/FlashML-org/FreeVideo/issues/1197  by s1y-f0x 2026-11-27T19:07:42Z

### What could be better? / 哪里可以做得更好？

Hi! I've investigated another issue in FreeVideo, identified its root cause, and successfully tested a possible solution.

### Environment

- FreeVideo: 0.6.14
- Build: `2026.11.26.232931-b764eae7f75d1a9f`
- OS: Windows 11
- GPU: NVIDIA RTX 4060, 8 GB VRAM
- RAM: 32 GB
- The issue has been observed since version 0.6.10.

### Description

After a successful 4K Upscale:

- The resulting video is saved to disk.
- Its card appears in the main Creations library without a thumbnail.
- The card disappears after switching between tabs.
- The result is missing from the Creations panel in Create.
- Restarting FreeVideo does not restore the missing result.

### Root cause

I found that the 4K Upscale path in `freevideo_engine/comfy_bridge.py` saves the completed video but does not set the following field in `video.request.json`:

```json
"success": true
```

Regular video generation sets this flag in `generate.py`.

The library's `_report()` function requires this field to be explicitly set to `true`. Without it, the completed upscale result is treated as unfinished and excluded from the library.

The thumbnail endpoint also checks `_report()`, which explains why it returns HTTP 404 even though the video exists.

Relevant source locations:

- `freevideo_engine/comfy_bridge.py`: lines 1347, 1382, 1430
- `freevideo_engine/generate.py`: line 1490
- `freevideo_engine/comfy_library.py`: lines 42, 245, 332, 1822
- `freevideo_engine/comfy_assets.py`: line 292

### Verification

I verified that an affected 4K video could be fully decoded without errors.

Video properties:

- Resolution: 3840 × 2176
- Frames: 124
- Frame rate: 24 FPS

After backing up its metadata, I added only `"success": true` to `video.request.json`.

This resolved all observed symptoms:

- The result appeared in the Creations panel in Create.
- The thumbnail endpoint returned HTTP 200.
- FreeVideo automatically generated the missing thumbnail.
- The card remained visible after switching tabs.
- The video played normally.

The original video file remained unchanged. No source code modifications were required for this verification.

### Suggested fix

Set `"success": true` in the result metadata after a 4K Upscale has successfully completed.

It would also be helpful to provide a safe recovery mechanism for previously completed upscale results affected by this issue, without marking failed or incomplete jobs as successful.

Thank you for your continued work on FreeVideo!

### Version / 版本

0.6.14 (2026.11.26.232931-b764eae7f75d1a9f)

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-27T20:10:43Z):**

Thank you for tracing this to the missing "success" field and for verifying it on your machine. The 4K Upscale saved its result summary without "success": true, and Creations, its thumbnails and the Edit view list a video only when that field is true, so the result was first shown without a thumbnail and then left out. Adding "success": true to the video.request.json of a completed upscale, as you did, is safe and makes the result appear again.


> **Comment by KBRASK (2026-11-28T06:22:49Z):**

Thank you again for tracing this. FreeVideo 0.6.15 writes "success": true for every finished 4K or quick upscale, and upscales saved by earlier versions without it are now listed in Creations with thumbnails and in the Edit view, without changing their files. Results to which you added the field by hand are unaffected.



### #1196 [open] ISSUE: Comparison Slider in Create is stuck at 50% due to a CSS variable override
URL: https://github.com/FlashML-org/FreeVideo/issues/1196  by s1y-f0x 2026-11-27T18:48:21Z

### What could be better? / 哪里可以做得更好？

Hi! I've found another bug in FreeVideo and identified its root cause. I'd like to share my findings and suggest a simple fix.

### Environment

- FreeVideo: 0.6.14
- Build: 2026.11.26.232931-b764eae7f75d1a9f
- OS: Windows 11
- Also reproduced in FreeVideo 0.6.13

### Description

The video comparison Slider in Create does not work correctly. The divider remains in the center when dragging it with the mouse.

The Slider in the main Creations view works correctly.

### Root cause

In `web/studio_compare.js`, `paintSplit()` updates the `--fvs-split` CSS variable on the outer `.fvs-compare` element.

However, the nested `.fvs-compare-panes` element declares its own value in `web/glass/studio.css`, line 538:

```css
--fvs-split: 50%;
```

This overrides the inherited value and prevents the divider and video clipping from updating visually.

### Verified workaround

I temporarily disabled the conflicting CSS declaration using browser developer tools.

Results:

- The Slider worked correctly in Create.
- It also worked in Compact layout.
- The Slider in the main Creations view continued to work.
- Reloading the page restored the original issue.

### Suggested fix

Remove the conflicting `--fvs-split: 50%` declaration from `.fvs-compare-panes` so that it inherits the value updated by `paintSplit()`.

I verified the workaround without modifying any source files or generating new videos.

Thank you for your work on FreeVideo!

### Version / 版本

_No response_

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-27T19:40:08Z):**

Thank you for tracking this down and for the suggested fix. The cause is as you describe: the inner comparison panes declare their own split value, which overrides the value updated on the outer element while dragging, so the divider stays at 50% in Create. Meanwhile, the slider in Creations works: choose "Select", pick the two creations, choose "Compare", then choose "Slider".


> **Comment by KBRASK (2026-11-28T03:19:00Z):**

Thank you again for the analysis and the suggested fix. FreeVideo 0.6.15 removes the split value that the inner comparison panes declared, as you suggested, so the divider and the clipping of the second video now follow the mouse when you drag the Slider in Create.



### #1195 [open] ISSUE: [启动器] 下载已中断
URL: https://github.com/FlashML-org/FreeVideo/issues/1195  by wu4yong 2026-11-27T17:43:32Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
RuntimeError: All Git sources failed. Check your network or connection mode in Downloads.

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA GeForce GT 620; NVIDIA GeForce RTX 5060; OrayIddDriver Device

RuntimeError: All Git sources failed. Check your network or connection mode in Downloads.
Sources tried for h3-text-encoder:
  - ghfast/direct (ghfast.top): timeout, exit 128, 300.7 s: fatal: unable to access 'https://ghfast.top/https://github.com/comfy-org/ComfyUI.git/': Connection timed out after 300004 milliseconds
  - ghfast/inherited (ghfast.top): connect, exit 128, 2.8 s: fatal: unable to access 'https://ghfast.top/https://github.com/comfy-org/ComfyUI.git/': Failed to connect to ghfast.top port 443 via 127.0.0.1 after 2086 ms: Co
  - official/direct (github.com): connect, exit 128, 21.7 s: fatal: unable to access 'https://github.com/comfy-org/ComfyUI.git/': Failed to connect to github.com port 443 after 21053 ms: Could not connect to server
  - official/inherited (github.com): connect, exit 128, 2.8 s: fatal: unable to access 'https://github.com/comfy-org/ComfyUI.git/': Failed to connect to github.com port 443 via 127.0.0.1 after 2090 ms: Could not connect to

> **Comment by KBRASK (2026-11-27T19:09:43Z):**

Thank you for sending the launcher message. From this network, the direct connections to github.com and to the mirror could not be made, and the connections through the proxy set on this computer went to a proxy address on this computer itself, where no proxy program was answering. Start the proxy program normally used on this computer, then choose "Retry installation" in the launcher; FreeVideo reads the proxy setting again each time installation is retried.



### #1194 [closed] ISSUE: [Launcher] Couldn’t complete this step
URL: https://github.com/FlashML-org/FreeVideo/issues/1194  by dipeshjangde001-blip 2026-11-27T16:11:28Z

### FreeVideo version / FreeVideo 版本

v0.6.0

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
[Errno 1] Operation not permitted: '<PATH>/FreeVideo'

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.0

[Errno 1] Operation not permitted: '<PATH>/FreeVideo'


### #1193 [open] ISSUE: [Feedback] 生成未完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1193  by rongxinw2-png 2026-11-27T15:49:42Z

### What could be better? / 哪里可以做得更好？

FreeVideo reported a generation failure.
Stage: execution
Error: cannot access local variable 'validate_route' where it is not associated with a value
Kind: unknown

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

{"kind":"unknown","stage":"execution","exception_type":"UnboundLocalError","known_error":null,"nodes":["FreeVideoGenerate"],"settings":{"width":672,"height":384,"seconds":7.792,"seed":1406222431,"two_pass":true,"base_steps":8,"refine_steps":3,"step_cache":"none","upscale_factor":"2"},"progress":{},"engine_exit":{"at":1795794413.0261378,"code_text":"0x40010004","exit_code":1073807364,"exit_code_unsigned":1073807364,"exit_class":"error_exit","suspected_system_kill":false,"memory":{"physical_available_bytes":24301154304,"commit_available_bytes":34692956160,"commit_limit_bytes":47547236352,"total_bytes":33514627072}},"frontend_version":"1.51.10","tail":"cannot access local variable 'validate_route' where it is not associated with a value\n"}


### #1192 [open] ISSUE: [Feedback] LoRA compatibility misjudged: ltx-2.3-id-lora-talkvid-3k.safetensors
URL: https://github.com/FlashML-org/FreeVideo/issues/1192  by hieutrungvn2k5-hub 2026-11-27T14:58:45Z

### What could be better? / 哪里可以做得更好？

FreeVideo marked this LoRA as unusable. The verdict is reported as a misjudgment.
File: ltx-2.3-id-lora-talkvid-3k.safetensors
Check result: int8-unmergeable, This LoRA cannot be used

### Version / 版本

v0.6.7

### Diagnostic info (optional) / 诊断信息（选填）

{"check":{"reason":"int8-unmergeable","family":null},"header":{"keys":1728,"first":[["diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_A.weight",[128,2048]],["diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_B.weight",[2048,128]],["diffusion_model.transformer_blocks.0.audio_attn1.to_out.0.lora_A.weight",[128,2048]],["diffusion_model.transformer_blocks.0.audio_attn1.to_out.0.lora_B.weight",[2048,128]],["diffusion_model.transformer_blocks.0.audio_attn1.to_q.lora_A.weight",[128,2048]]],"base":{}}}


### #1191 [open] ISSUE: [Launcher] Download interrupted
URL: https://github.com/FlashML-org/FreeVideo/issues/1191  by altayali-star 2026-11-27T13:40:45Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
RuntimeError: Failed step: vdn-source (exit code 1)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA RTX 2000 Ada Generation; Intel(R) UHD Graphics 770

RuntimeError: Failed step: vdn-source (exit code 1)
Retained step log: <PATH>/02-vdn-source.log
Full step output:
{"event": "network", "epoch": 1795786826.115356, "category": "git", "source": "ghfast", "route": "inherited", "file": "vdn", "action": "attempt", "commit": "<token>"}
{"event": "network", "epoch": 1795786828.1091967, "category": "git", "source": "ghfast", "route": "inherited", "file": "vdn", "action": "fallback", "reason": "network-fetch-failed", "host": "ghfast.top", "error_class": "exit", "message": "fatal: unable to access 'https://ghfast.top/https://github.com/OpenVDN/vdn-minimax-h3.git/': Recv failure: Connection was reset", "returncode": 128, "elapsed_seconds": 2.0}
{"event": "network", "epoch": 1795786828.148194, "category": "git", "source": "official", "route": "inherited", "file": "vdn", "action": "attempt", "commit": "<token>"}
{"event": "network", "epoch": 1795786829.4538798, "category": "git", "source": "official", "route": "inherited", "file": "vdn", "action": "fallback", "reason": "network-fetch-failed", "host": "github.com", "error_class": "tls", "message": "fatal: unable to access 'https://github.com/OpenVDN/vdn-minimax-h3.git/': schannel: next InitializeSecurityContext failed: CRYPT_E_REVOCATION_OFFLINE (0x8009201", "returncode": 128, "elapsed_seconds": 1.3}
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "<PATH>/install.py", line 20, in install
    network.clone(root, spec['url'], spec['commit'])
  File "<PATH>/network.py", line 2322, in clone
    raise failure
RuntimeError: All Git sources failed. Check your network or connection mode in Downloads.
Sources tried for vdn:
  - ghfast/inherited (ghfast.top): exit, exit 128, 2.0 s: fatal: unable to access 'https://ghfast.top/https://github.com/OpenVDN/vdn-minimax-h3.git/': Recv failure: Connection was reset
  - official/inherited (github.com): tls, exit 128, 1.3 s: fatal: unable to access 'https://github.com/OpenVDN/vdn-minimax-h3.git/': schannel: next InitializeSecurityContext failed: CRYPT_E_REVOCATION_OFFLINE (0x8009201

> **Comment by KBRASK (2026-11-27T18:39:08Z):**

Thank you for sending the launcher message. The download from GitHub failed because Windows could not check on this network whether GitHub's certificate had been revoked, and the Git configuration on this computer requires that check, so Git refused the connection; the connection to the mirror was reset as well. Run git config --global http.schannelCheckRevoke best-effort in a command prompt, then choose "Retry installation" in the launcher. With this setting, Git still verifies the certificate and still refuses a revoked one; it no longer fails only because the revocation status cannot be checked.


> **Comment by KBRASK (2026-11-28T06:53:25Z):**

Thank you again for the launcher message. FreeVideo 0.6.15 tries a Git source once more with the best-effort revocation check when Windows cannot learn the certificate's revocation status, so setup no longer stops for this reason without the http.schannelCheckRevoke setting.



### #1190 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1190  by SahilNmadev 2026-11-27T12:21:39Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 1 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: AMD Radeon RX 580 2048SP

Setup inspection failed (plan, exit 1, 1 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-27T12:21:17Z", "platform": "Windows-11-10.0.26200-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.22.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-27T13:36:01Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon RX 580 2048SP and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards (8 GB of video memory or more is recommended), and, as a preview, on Apple silicon Macs.



### #1189 [open] ISSUE: batch video generation
URL: https://github.com/FlashML-org/FreeVideo/issues/1189  by seerealserial-dotcom 2026-11-27T11:43:22Z

### What could be better? / 哪里可以做得更好？

hi

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

batch video generation would be the best option. kindly create batch and reference to video while creating video

> **Comment by KBRASK (2026-11-27T13:05:29Z):**

Thank you for the suggestion. FreeVideo already has batch generation: in Create, open Advanced, choose Batch under Generation mode, and set Number of videos from 2 to 100; with "One prompt per line" turned on, each line of the description becomes its own video. A video can also be used as a reference: in Create, open Media, choose "Add", and select the video; it is cut to the length of the result, at most 15 s, and a generation with it takes longer. If you would like the reference video to control something this does not cover, please describe it.


> **Comment by seerealserial-dotcom (2026-11-27T14:01:27Z):**

hi

In batch wise operation can we put reference images for every prompt if yes
how to do it please let me know

On Fri, Nov 27, 2026 at 6:35 PM Bowen Xue ***@***.***> wrote:

> *KBRASK* left a comment (FlashML-org/FreeVideo#1189)
> <https://github.com/FlashML-org/FreeVideo/issues/1189#issuecomment-7219044845>
>
> Thank you for the suggestion. FreeVideo already has batch generation: in
> Create, open Advanced, choose Batch under Generation mode, and set Number
> of videos from 2 to 100; with "One prompt per line" turned on, each line of
> the description becomes its own video. A video can also be used as a
> reference: in Create, open Media, choose "Add", and select the video; it is
> cut to the length of the result, at most 15 s, and a generation with it
> takes longer. If you would like the reference video to control something
> this does not cover, please describe it.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/FlashML-org/FreeVideo/issues/1189?email_source=notifications&email_token=CWGXKMFQ6GHJAJCIZM23LCD53F33BA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTOMRRHEYDINBYGQ22M4TFMFZW63VGMF2XI2DPOKSWK5TFNZ2KYZTPN52GK4S7MNWGSY3L#issuecomment-7219044845>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/CWGXKMC7YX4JBOLAOWLKMDT53F33BAVCNFSNUABGKJSXA33TNF2G64TZHMYTGNRYHEYTGNRTHA5US43TOVSTWNRWGQ2TONBTG44DTILWAI>
> .
> Triage notifications, keep track of coding agent tasks and review pull
> requests on the go with GitHub Mobile for iOS
> <https://github.com/notifications/mobile/ios/CWGXKMAUYURWOQ4JE62HKBL53F33BA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTOMRRHEYDINBYGQ22M4TFMFZW63VGMF2XI2DPOKSWK5TFNZ2KUZTPN52GK4S7NFXXG>
> and Android
> <https://github.com/notifications/mobile/android/CWGXKMHBNFVPE6YC44PWBUT53F33BA5CNFSNUABFM5UWIORPF5TWS5BNNB2WEL2JONZXKZKDN5WW2ZLOOQXTOMRRHEYDINBYGQ22M4TFMFZW63VGMF2XI2DPOKSWK5TFNZ2K4ZTPN52GK4S7MFXGI4TPNFSA>.
> Download it today!
> You are receiving this because you authored the thread.Message ID:
> ***@***.***>
>




### #1188 [open] ISSUE: Can We use minimax video for my youtube channel
URL: https://github.com/FlashML-org/FreeVideo/issues/1188  by seerealserial-dotcom 2026-11-27T11:31:06Z

### FreeVideo version / FreeVideo 版本

minimax h3

### What happened? / 遇到了什么问题？

i run a youtube channel, can i you minimax h3 video, is that free to use, or need to give attribution to it or how please do explain me about using different models in free video

### Diagnostic report / 诊断报告

i run a youtube channel, can i you minimax h3 video, is that free to use, or need to give attribution to it or how please do explain me about using different models in free video

> **Comment by KBRASK (2026-11-27T12:34:54Z):**

Thank you for asking. FreeVideo's code is under the Apache License 2.0, and the video model weights are under the MiniMax H3 Community License, which is listed in FreeVideo's README and THIRD_PARTY_NOTICES. That license states that MiniMax claims no rights over the outputs you generate and that you are responsible for them and their use. Its rights are granted only outside the European Union, the United Kingdom, the Republic of Korea and the United States, and it does not permit using, distributing or displaying the outputs outside that territory; its acceptable use policy also requires content shared in a public environment to be clearly and prominently disclosed as machine-generated. The other models FreeVideo can download are listed in THIRD_PARTY_NOTICES with their own licenses. Please read these licenses for the full terms, as we cannot give legal advice.



### #1187 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1187  by phuongnam0279 2026-11-27T09:24:57Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 1 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: AMD Radeon RX 5600 XT

Setup inspection failed (plan, exit 1, 1 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-27T09:24:24Z", "platform": "Windows-11-10.0.22621-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.9.1"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-27T11:33:29Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon RX 5600 XT and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards (8 GB of video memory or more is recommended), and, as a preview, on Apple silicon Macs.



### #1186 [open] ISSUE: [Report] Apple M2 Max · 1344×768 · 8 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1186  by himsic 2026-11-27T08:27:13Z

### Diagnostic report / 诊断报告

[FreeVideo-report-20261127-135614.json](https://github.com/user-attachments/files/35502942/FreeVideo-report-20261127-135614.json)

### FreeVideo version / FreeVideo 版本

v0.6.12 (2026.11.25.60706)

### Graphics card and memory / 显卡和内存

Apple M2 Max · 64 GB RAM · Darwin

### Video size and length / 视频尺寸和时长

1344×768 · 8 s

### Time taken / 用时

4213 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-28T00:46:08Z):**

Thank you for sending the diagnostic report. Our analysis showed that this run did not finish: macOS stopped the GPU work twice with the Metal error "Impacting Interactivity", first during the first pass, which FreeVideo retried with smaller blocks, and again during the second pass after about 4,200 s, when no retry was left, so the request ended. FreeVideo 0.6.15 gives the second pass a retry of its own after a Metal failure, so a single Metal failure there no longer ends the request. From FreeVideo 0.6.14, this Mac can also use the Fast level, and based on this report we estimate the same request will take about 2,300 s at that level.



### #1185 [open] ISSUE: [启动器] 下载已中断
URL: https://github.com/FlashML-org/FreeVideo/issues/1185  by syk121 2026-11-27T07:55:07Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
RuntimeError: All Git sources failed. Check your network or connection mode in Downloads.

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA GeForce RTX 5060 Ti

RuntimeError: All Git sources failed. Check your network or connection mode in Downloads.
Sources tried for h3-text-encoder:
  - official/direct (github.com): proxy, exit 128, 3.4 s: fatal: unable to access 'https://mirror.ghproxy.com': Failed to connect to mirror.ghproxy.com port 443 after 2290 ms: Could not connect t
  - official/inherited (github.com): proxy, exit 128, 2.4 s: fatal: unable to access 'https://mirror.ghproxy.com': TLS connect error: error:0A000126:SSL routines::unexpected eof while reading
  - ghfast/direct (ghfast.top): exit, exit 128, 254.2 s: fatal: unable to access 'https://ghfast.top/https://github.com/comfy-org/ComfyUI.git/': Recv failure: Connection was reset
  - ghfast/inherited (ghfast.top): reset, exit 128, 83.5 s: fatal: unable to access 'https://ghfast.top/https://github.com/comfy-org/ComfyUI.git/': TLS connect error: error:0A000126:SSL routines::unexpected eof while rea

> **Comment by KBRASK (2026-11-27T10:32:12Z):**

Thank you for sending the launcher message. The Git configuration on this computer contains a rule that redirects github.com to mirror.ghproxy.com, and that mirror could not be reached; FreeVideo's Git downloads follow this rule, so the download from github.com failed. To remove the rule, run git config --global --get-regexp insteadof in a command prompt and find the line containing mirror.ghproxy.com. Then run git config --global --unset followed by the first part of that line, up to the space, and choose "Retry installation" in the launcher. This removes only that redirect; other Git settings stay unchanged.


> **Comment by KBRASK (2026-11-28T07:24:02Z):**

Thank you again for the launcher message. FreeVideo 0.6.15 keeps its own download addresses from being rewritten by this computer's Git settings, so a redirect rule like the one to mirror.ghproxy.com no longer affects setup.



### #1184 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1184  by sunyazhou90 2026-11-27T07:50:11Z

### What could be better? / 哪里可以做得更好？

以前一采完成的作品在右侧边栏可以随意加入二采队列，现在咋这么费劲，点半天没法进行二采，只能一个一个来

### Version / 版本

v0.6.14

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-27T11:02:47Z):**

Thank you for the feedback. While a generation is running in Create, each progress update brings the progress view back to the center and removes the buttons below it, so when a first-pass preview is opened from the right sidebar, its "Run second pass" button stays only until the next update and is hard to click. The second-pass queue itself has no limit. To add several previews, open Creations, select a first-pass preview, and choose "Run second pass" below the viewer; previews can be added to the second-pass queue one after another this way, while a second pass is running.


> **Comment by KBRASK (2026-11-28T07:54:55Z):**

Thank you again for the feedback. FreeVideo 0.6.15 keeps a work picked in the right column on the stage with its actions while a video is being made, so "Run second pass" can be chosen for first-pass previews from the right column one after another; the "Generating" tile at the start of that column brings the running video's progress back to the middle.



### #1183 [open] ISSUE: [启动器] 暂时未能完成
URL: https://github.com/FlashML-org/FreeVideo/issues/1183  by Lucifer-gj 2026-11-27T06:39:43Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
RuntimeError: Failed step: vdn-source (exit code 1)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA GeForce RTX 5060 Laptop GPU; Intel(R) UHD Graphics

RuntimeError: Failed step: vdn-source (exit code 1)
Retained step log: <PATH>/01-vdn-source.log
Full step output:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import sys;sys.path[:]=[p for p in sys.path if p];from freevideo_engine.install import install; install()
                                                                                                    ~~~~~~~^^
  File "<PATH>/install.py", line 20, in install
    network.clone(root, spec['url'], spec['commit'])
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/network.py", line 2171, in clone
    raise ValueError(mismatch_message(target, git, env, head, commit, dirty))
ValueError: A dependency folder differs from the version FreeVideo needs: <PATH>/vdn. Its version could not be read. Rename or move this folder, then retry installation. FreeVideo will download a fresh copy.

> **Comment by KBRASK (2026-11-27T09:31:04Z):**

Thank you for sending the launcher message. Setup stopped at the vdn folder inside the vendor folder of the FreeVideo installation folder: Git could not read its version, and its Git settings differ from those of the copy FreeVideo downloads, so FreeVideo left it unchanged instead of replacing it. Rename this folder, for example to vdn-old, and choose "Retry installation" in the launcher; FreeVideo then downloads a fresh copy and checks it, which needs access to GitHub or the mirror. Rename only vendor\vdn, not the vdn folder inside the models folder, which holds the downloaded models.



### #1182 [open] ISSUE: [Feedback] Page check
URL: https://github.com/FlashML-org/FreeVideo/issues/1182  by xinzhyu 2026-11-27T06:23:36Z

### What could be better? / 哪里可以做得更好？

FreeVideo showed an error.
Where: Page check
Error: 从菜单打开 FreeVideo 失败: Error: The FreeVideo workflow loaded without a FreeVideoGenerate node

### Version / 版本

_No response_

### Diagnostic info (optional) / 诊断信息（选填）

FreeVideo page check · 2026-11-27T06:23:29.855Z
Browser: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/<address> Safari/537.36 QQBrowser/22.4.7011.400
ComfyUI frontend: 1.59.10
FreeVideo UI: {"ready":true}
Automatic reload: not needed

Problems:
- 从菜单打开 FreeVideo 失败
  error: Error: The FreeVideo workflow loaded without a FreeVideoGenerate node

Errors seen while loading:
- [console] [vite:preloadError] {"url":null,"fileType":"unknown","chunkName":null,"message":"Extension named 'ColorOverlay' already registered."}

> **Comment by KBRASK (2026-11-27T09:00:40Z):**

Thank you for sending the page check. The message "Extension named 'ColorOverlay' already registered" comes from another node package and does not affect FreeVideo. On our test machines, opening FreeVideo from the menu works with the same ComfyUI frontend version, so we could not reproduce this error. Please reload the page and choose Extensions > FreeVideo in the ComfyUI menu again. If the error appears again, please reply with the FreeVideo version, whether it happens every time, and the other custom node packages installed in this ComfyUI.



### #1181 [open] ISSUE: [Launcher] Couldn’t complete this step
URL: https://github.com/FlashML-org/FreeVideo/issues/1181  by jouststark 2026-11-27T06:07:52Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 8 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA Quadro P5200

Setup inspection failed (plan, exit 1, 8 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-27T06:02:37Z", "platform": "Windows-11-10.0.26300-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.22.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preparing isolated Python for setup. Model/environment installation requires the setup confirmation.
Bootstrap cache and log: <PATH>/bootstrap
Using verified local download tools
error: Failed to inspect Python interpreter from managed installations at `<PATH>/python.exe`
  Caused by: Querying Python at `<PATH>/python.exe` returned an invalid response: EOF while parsing a value at line 1 column 0
FreeVideo could not start: You cannot call a method on a null-valued expression.
Existing files are retained. Retry the same command after resolving the reported issue.
To report it: open a new issue at https://github.com/FlashML-org/FreeVideo/issues/new with this message and attach <PATH>/bootstrap.log.

> **Comment by KBRASK (2026-11-27T08:30:04Z):**

Thank you for sending the launcher message. Setup stopped because the Python that FreeVideo installed for setup started but returned no output. However, this computer's NVIDIA Quadro P5200 has compute capability 6.1, below the 8.0 that FreeVideo requires, so even after this step succeeds, setup would stop at the graphics check, and a driver update cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards (8 GB of video memory or more is recommended), and, as a preview, on Apple silicon Macs.



### #1180 [open] ISSUE: [Launcher] Couldn’t complete this step
URL: https://github.com/FlashML-org/FreeVideo/issues/1180  by anykeystroganov-alt 2026-11-27T06:06:21Z

### FreeVideo version / FreeVideo 版本

v0.6.14

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
RuntimeError: Failed step: unified-venv (exit code 2)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.14
Display adapters: NVIDIA GeForce GT 710; Intel(R) HD Graphics 4400; Intel(R) UHD Graphics 630; NVIDIA GeForce RTX 3060; AMD Radeon RX 580 2048SP

RuntimeError: Failed step: unified-venv (exit code 2)
Retained step log: <PATH>/01-unified-venv.log
Full step output:
error: Python interpreter not found at `<PATH>/python.exe`

> **Comment by KBRASK (2026-11-27T07:59:29Z):**

Thank you for sending the launcher message. Setup stopped because the python.exe that FreeVideo installed for its environment was no longer on the disk. FreeVideo does not delete this file, so it was most likely removed or quarantined by security software. If the security software lists the file in its quarantine or protection history, restore it, add the FreeVideo installation folder as an exception, and choose "Retry installation" in the launcher. Otherwise, open the python folder inside the FreeVideo installation folder, delete only the folder whose name starts with cpython-3.12, and choose "Retry installation". The launcher sets up this Python again and rebuilds its environment; your creations and models are kept.


> **Comment by KBRASK (2026-11-28T08:56:08Z):**

Thank you again for the launcher message. FreeVideo 0.6.15 installs its isolated Python once more when its python.exe is missing or does not respond, and names the file if that still fails, so setup recovers from this case by itself; please update FreeVideo and retry the installation.



### #1179 [open] ISSUE: [启动器] 下载中断了
URL: https://github.com/FlashML-org/FreeVideo/issues/1179  by wo9090 2026-11-27T03:59:36Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
DownloadError: Every download source failed for sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl. Check your network or choose another connection mode in Downloads. Partial files retained

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: NVIDIA GeForce RTX 5060 Laptop GPU; Intel(R) Arc(TM) 140T GPU (16GB)

DownloadError: Every download source failed for sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl. Check your network or choose another connection mode in Downloads. Partial files retained
Sources tried:
  - official/direct (github.com): tls, curl-35, 0.1 s: curl: (35) schannel: next InitializeSecurityContext failed: CRYPT_E_NO_REVOCATION_CHECK (0x80092012) - ���������޷����֤���Ƿ������
  - ghfast/direct (ghfast.top): timeout, curl-28, 10.0 s: curl: (28) Connection timed out after 5005 milliseconds

> **Comment by KBRASK (2026-11-27T07:28:46Z):**

Thank you for sending the launcher message. The download from GitHub failed because Windows could not check on this network whether GitHub's certificate had been revoked, so it refused the connection, and the mirror could not be reached either. The file can be added by hand. In a browser, open the v2.2.0-windows.post6 release of woct0rdho/SageAttention on GitHub and download the file whose name ends in cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl. Without renaming it, put it in the folder wheels\windows-sage2\1635283f5c01ec3c inside the FreeVideo installation folder, then retry the installation in the launcher; it uses the file once its checksum matches.


> **Comment by KBRASK (2026-11-28T02:17:49Z):**

Thank you again for the launcher message. FreeVideo 0.6.15 tries a download once more with a best-effort revocation check when Windows cannot learn the certificate's revocation status, so this download no longer fails for that reason and the file does not need to be placed by hand.



### #1178 [open] ISSUE: [Report] Apple M1 Max · 1344×768 · 10.1 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1178  by jackyzhangmm 2026-11-27T03:25:12Z

### Diagnostic report / 诊断报告

[FreeVideo-report-20261127-142454.json](https://github.com/user-attachments/files/35494096/FreeVideo-report-20261127-142454.json)

### FreeVideo version / FreeVideo 版本

v0.5.3 (2026.11.13.91907)

### Graphics card and memory / 显卡和内存

Apple M1 Max · 32 GB RAM · Darwin

### Video size and length / 视频尺寸和时长

1344×768 · 10.1 s

### Time taken / 用时

27578 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-27T06:58:04Z):**

Thank you for sending the diagnostic report. Our analysis showed that this run was limited by the compute of the M1 Max graphics processor: it used the High level, 16 steps at full size in one pass, each full step took about 2,400 s, and sampling took 26,057 s of the 27,578 s, while memory was not the limit. The same request at the High level is not faster in FreeVideo 0.6.14, but 0.6.14 adds the Fast level, two steps in one pass, on Apple M1 to M5 Macs with 16 GB of memory or more, and based on the step times in this report we estimate the same request will take about 5,700 s at that level.



### #1177 [open] ISSUE: [Launcher] Not enough disk space
URL: https://github.com/FlashML-org/FreeVideo/issues/1177  by gigi66064-del 2026-11-27T01:41:49Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Insufficient disk space: need 12.00 GiB, available 11.36 GiB, short 0.64 GiB · <PATH>/vdn, <PATH>/h3-base, <PATH>/audio_scheduler, <PATH>/audio_vae, <PATH>/scheduler, <PATH>/transformer, <PATH>/vae, <PATH>/default, <PATH>/turbo, <PATH>/linear_branch, <PATH>/stage-dmd-step-250, <PATH>/text_encoders, 

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: NVIDIA GeForce RTX 5080; DeepCoolDisplayDriver Device

Insufficient disk space: need 12.00 GiB, available 11.36 GiB, short 0.64 GiB · <PATH>/vdn, <PATH>/h3-base, <PATH>/audio_scheduler, <PATH>/audio_vae, <PATH>/scheduler, <PATH>/transformer, <PATH>/vae, <PATH>/default, <PATH>/turbo, <PATH>/linear_branch, <PATH>/stage-dmd-step-250, <PATH>/text_encoders, <PATH>/latent_upscaler, <PATH>/edge-da37681f5a1f7015, <PATH>/audio_scheduler, <PATH>/audio_vae, <PATH>/h3-base, <PATH>/scheduler, <PATH>/transformer, <PATH>/vae, <PATH>/default, <PATH>/turbo, <PATH>/linear_branch, <PATH>/stage-dmd-step-250, <PATH>/<token>, <PATH>/<token>, <PATH>/blocks, <PATH>/cache, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/<token>, <PATH>/FreeVideo-engine, <PATH>/FreeVideo

> **Comment by KBRASK (2026-11-27T02:22:57Z):**

Thank you for sending the launcher message. Setup stopped because the drive FreeVideo is installing to has 11.36 GiB free while this step needs 12.00 GiB, so it is 0.64 GiB short. Please free at least that much space on that drive and click Check again in the launcher; the files already downloaded are kept.



### #1176 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1176  by GhostingDev 2026-11-27T00:55:31Z

### What could be better? / 哪里可以做得更好？

When I try to delete a video in Creations, it shows "in use, kept". The asset is still being used by an Edit and cannot be removed, so the video can never be deleted.

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-27T01:28:50Z):**

Thank you for reporting this. In FreeVideo 0.6.13 a video cannot be deleted from Creations while any saved Edit still uses it, either as a clip on the timeline or as the earlier version of a clip that was generated again, which is not shown on the timeline. The message does not say which Edit uses the video, so the Edits have to be opened one by one to find it. In that Edit, select the clip that uses the video, or the clip that was generated again from it, press Delete and wait until the Edit is saved; the video can then be deleted in Creations. Alternatively, "Delete edit" in the menu beside the Edit's name removes the whole Edit and does not delete the creations it uses.



### #1175 [closed] PR: FreeVideo 0.6.14
URL: https://github.com/FlashML-org/FreeVideo/pull/1175  by KBRASK 2026-11-26T23:24:05Z

0.6.14 makes the first upscale download about 300 MB instead of 26 GB, lets the Fast level finish videos on 8 GB Windows graphics cards, lets more video players fill the window with volume and speed controls, keeps Windows setup going when security software briefly blocks a file, removes more personal details from reports, explains more errors, and fixes many layout problems.

- Upscaling: the first upscale of a finished video downloads about 300 MB of prepared sampling caches instead of 26 GB of original model files, and the "Upscale to …" option is offered in the menu again.
- Fast level: The Fast level now finishes videos on 8 GB Windows graphics cards instead of failing at the last step. On Macs with Apple M1–M5 chips and at least 16 GB of unified memory, the Fast level is also available. On FP8 installations, the Fast level is shown as unavailable with the reason and the download size, and the launcher offers the model it needs; on workstation cards it says the other levels are about 5% slower with that model.
- Windows installation: setup checks the Visual C++ runtime before it first loads PyTorch. When Windows or security software briefly blocks a Python file, a package file or the new launcher, the step is tried again before it stops, and the card then names the blocked file. A graphics card that FreeVideo does not support is named as such instead of asking for a driver update, and a folder name in the install location that Windows does not accept is pointed out, with its position and the kind of problem, before installing starts.
- Privacy in reports: diagnostic reports and problem reports remove user folder names from more kinds of paths, such as a path right after a colon or a user folder on a network share, and diagnostic reports also remove email addresses and sign-in details. In reports and download failure records, addresses on model and code hosting sites keep only the site name, except for FreeVideo's own files.
- Prompt enhancement: When a prompt enhancement request is too long, the message now says about how far over the limit it is. A rewrite that runs past its length limit is reported separately. Writing a prompt from an image or video in the create panel works again; it had stopped with "Could not read this image".
- Starting, memory and disk: a video started right after FreeVideo starts no longer waits about 15 seconds, and two-pass videos reach the second pass and decoding sooner, as the upscaler file is no longer re-checked in full each time. A request that is too large for the memory free on the computer is now declined at the start, with the amount it needs, instead of stopping after most of its time has been spent. Downloads that fail their check no longer pile up on disk: at most the newest copy of each file is kept, and none once the right file is in place.
- Errors with reasons: an image that fails without a known cause names the step it stopped at and still offers a report, a failed video in Autopilot names a recognized cause such as too little graphics memory, and an image model file that cannot be written is named with the reason. In the editor, a request that could not be completed says which step failed, and a file that is in use or cannot be read is named as the cause. Results and reports also say when the graphics card's power limit held its clock down, and whether other work on the card came from FreeVideo or another program.
- Video players: videos in the create panel, node results and Autopilot can be enlarged to fill the window, and these players and the library's have volume and playback speed controls. On the position bar, the arrow keys move 1 second back or forward (5 seconds with Shift), and Home and End go to the start and the end, in all of these players.
- Create panel: the Media window no longer overlaps the size and duration settings when you have many characters, scenes or props, and the added images are shown in full. The quality levels can be chosen from the keyboard alone (Tab, the arrow keys, Home and End), a screen reader reads them as one group, and when continuing a video or wired nodes make the quality fixed, the greyed-out levels say why on hover. With a workflow that sets its own step counts, moving to Generate with the keyboard no longer shifts the settings column up or leaves an unreachable strip below it.
- Editor: in every interface language the generation bar fits the room it has, drop-down lists open in front and work with the keyboard, and names in its windows, track names and the time on Generate are no longer cut short. Notices are easier to close on touch screens and keep Report a problem whole, and empty edits that were never named are cleared away.
- Panels and interface text: small windows, trays, drawers, the shortcut sheet and the command palette are always opaque, and so are side panels when glass effects are off or transparency is reduced. More Chinese messages use formal written wording. The image text layer and share cards display Traditional Chinese, Korean, Vietnamese, Thai, Arabic, Persian and Hindi correctly; add Arabic, Persian and Hindi text in the text layer, because smart editing cannot draw them correctly yet.
- Other fixes: a first launch no longer shows the what's-new tour as an upgrade; each Generate is one row in the jobs list; cancelling a file picker no longer closes its view; a cast card's face close-up shows the whole head. Launcher messages name their buttons exactly, and button names no longer wrap in narrow windows. With ComfyUI started without the launcher, clearing the cache, pack downloads and the cleanup after deleting a work use the FreeVideo folder, and a downloaded Fast level is no longer reported missing; FreeVideo uses only the graphics cards in CUDA_VISIBLE_DEVICES.



### #1174 [open] ISSUE: [Feedback] 提示词增强
URL: https://github.com/FlashML-org/FreeVideo/issues/1174  by heavenlara 2026-11-26T20:05:03Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Prompt enhancement
Error: 请等待当前队列完成后再改写。

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Prompt enhancement","kind":null,"tail":"请等待当前队列完成后再改写。"}

> **Comment by KBRASK (2026-11-26T22:02:59Z):**

Thank you for reporting this. Prompt enhancement does not run while a generation is running or waiting in the queue, and this message appears in that case. Please enhance the prompt before adding the generation to the queue, or wait until the queue has finished. If the message appears when nothing is running or waiting and no earlier enhancement is still in progress, please tell us what you had done just before, as that would point to a different problem.



### #1173 [open] ISSUE: [Launcher] ComfyUI could not start
URL: https://github.com/FlashML-org/FreeVideo/issues/1173  by LeonardoJGimon 2026-11-26T18:41:27Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
ComfyUI could not start.

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: NVIDIA GeForce RTX 4070; Intel(R) UHD Graphics 770

ComfyUI could not start.
Process exit code: 3221226505
Environment: freevideo-managed
Custom nodes: FreeVideo
[FreeVideo] Separate environment: loading FreeVideo custom nodes only. Use your original ComfyUI launcher for other plugins.
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.schemas
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.tables
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.types
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.constraints
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.defaults
^[[32m[INFO]^[[0m setup plugin alembic.autogenerate.comments
^[[32m[INFO]^[[0m setup plugin alembic.ext.checkconstraint_byname
^[[1m^[[33m[WARNING]^[[0m On windows we are currently forcing single GPU mode in ComfyUI due to a Nvidia related issue, if you want to disable this use: --cuda-device all
^[[32m[INFO]^[[0m
...
rue, 'unavailable_reason': "ImportError: No module named 'triton'", 'capabilities': []}
^[[32m[INFO]^[[0m Checkpoint files will always be loaded safely.
^[[32m[INFO]^[[0m Total VRAM 12282 MB, total RAM 65231 MB
^[[32m[INFO]^[[0m pytorch version: 2.13.0+cu130
^[[32m[INFO]^[[0m Set vram state to: NORMAL_VRAM
^[[32m[INFO]^[[0m Device: cuda:0 NVIDIA GeForce RTX 4070 : cudaMallocAsync
^[[32m[INFO]^[[0m Using async weight offloading with 2 streams
^[[32m[INFO]^[[0m Enabled pinned memory 26092.0
^[[32m[INFO]^[[0m Using pytorch attention
^[[32m[INFO]^[[0m aimdo: src-win/cuda-detour.c:38:INFO:aimdo_setup_hooks: installing 6 hooks
^[[32m[INFO]^[[0m aimdo: src/control.c:269:INFO:comfy-aimdo NVML pressure enabled
^[[32m[INFO]^[[0m aimdo: src-win/shmem-detect.c:80:INFO:comfy-aimdo WDDM adapter match: NVIDIA GeForce RTX 4070 runtime_luid=00000000:0000e027 dxgi_luid=00000000:0000e027
^[[32m[INFO]^[[0m aimdo: src/control.c:284:INFO:comfy-aimdo inited for GPU: NVIDIA GeForce RTX 4070 (VRAM: 12281 MB)
^[[32m[INFO]^[[0m DynamicVRAM support detected and enabled
^[[32m[INFO]^[[0m Python version: 3.12.3 (main, Apr 15 2024, 17:48:16) [MSC v.1929 64 bit (AMD64)]
^[[32m[INFO]^[[0m ComfyUI version: 0.34.0
^[[32m[INFO]^[[0m comfy-aimdo version: 0.5.3
^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.33
^[[32m[INFO]^[[0m comfyui-frontend-package version: 1.51.10
^[[32m[INFO]^[[0m comfyui-embedded-docs version: 0.5.11
^[[32m[INFO]^[[0m comfy-kitchen version: 0.2.33
^[[32m[INFO]^[[0m comfy-aimdo version: 0.5.3
^[[32m[INFO]^[[0m [Prompt Server] web root: <PATH>/static
^[[32m[INFO]^[[0m Asset seeder disabled
^[[32m[INFO]^[[0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
Log: <PATH>/comfy.log

> **Comment by KBRASK (2026-11-26T21:32:06Z):**

Thank you for reporting this with the full startup log.

The log shows that ComfyUI stopped while it was loading its built-in nodes, shortly after it loaded the graphics libraries used by its GLSL shader node, and before FreeVideo's own node started. Windows ended the process with code 0xC0000409, which means a native component stopped the program at once; this code does not say which component it was, so the log alone does not show the cause.

Windows records which component it was. To find that record:
1. Press the Windows key, type "Event Viewer" and open it.
2. In the left pane, expand "Windows Logs" and select "Application".
3. In the right pane, select "Filter Current Log...", type 1000 in the box that reads <All Event IDs>, and select OK.
4. In the list, select the entry whose Source is "Application Error" and whose time matches the moment FreeVideo showed the message. On the "General" tab below, the first line reads "Faulting application name: python.exe".
5. Please post the three lines "Faulting module name", "Exception code" and "Fault offset" from that entry. The other lines contain your user folder and are not needed.

It would also help to know whether the same message appears every time you start FreeVideo.



### #1172 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1172  by arthurvrodrigues7-sys 2026-11-26T18:16:15Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 1 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: AMD Radeon 740M Graphics

Setup inspection failed (plan, exit 1, 1 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-26T18:10:54Z", "platform": "Windows-11-10.0.26200-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.22.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-26T18:59:09Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon 740M Graphics and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards with 8 GB of video memory or more, and, as a preview, on Apple silicon Macs.



### #1171 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1171  by seerealserial-dotcom 2026-11-26T17:54:49Z

### What could be better? / 哪里可以做得更好？

please put an option to choose the image model options if i want m3 or qwen or moss i should have option to choose or either enable to disabled or delete model button should also there

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-26T20:30:49Z):**

Thank you for the suggestion. In FreeVideo 0.6.13 you can already choose the image model: the Model list at the top of the left column in the Image view offers MiniMax H3 Image, Qwen-Image 2.1 and MOSS-Image. Removing a model is possible only for Qwen-Image 2.1, with the Remove button on its card in the launcher while FreeVideo is not running; there is no way yet to turn a model off or to remove MiniMax H3 Image or MOSS-Image, and we have recorded this request.



### #1170 [open] ISSUE: [Report] RTX Spark N1X (6144-core Blackwell RTX GPU) 110 GB · 1344×768 · 10.1 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1170  by fe3o48 2026-11-26T17:18:13Z

### Diagnostic report / 诊断报告

[FreeVideo-report-20261127-011739.zip](https://github.com/user-attachments/files/35475824/FreeVideo-report-20261127-011739.zip)

### FreeVideo version / FreeVideo 版本

v0.6.13 (2026.11.26.15432)

### Graphics card and memory / 显卡和内存

RTX Spark N1X (6144-core Blackwell RTX GPU) 110 GB · 30.5 GB RAM · Windows

### Video size and length / 视频尺寸和时长

1344×768 · 10.1 s

### Time taken / 用时

728 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-27T06:27:39Z):**

Thank you for sending this report as well. Our analysis showed that this run was also close to the best this computer can do: the whole video model stayed in graphics memory, the graphics card held about 1,970 MHz without slowing down, and sampling took 552 s of the 728 s (eight steps at 896×512, about 35 s each, and three at 1344×768, about 88 s each). As in #1154, about 12 s went to waiting between the two passes and before decoding, which FreeVideo 0.6.14 removes, so we estimate the same request will take about 715 s instead of 728 s in the new version.



### #1169 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1169  by qrparivahangov 2026-11-26T16:16:50Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 5 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: AMD Radeon(TM) Graphics

Setup inspection failed (plan, exit 1, 5 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-26T16:15:47Z", "platform": "Windows-11-10.0.26300-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.22.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preparing isolated Python for setup. Model/environment installation requires the setup confirmation.
Bootstrap cache and log: <PATH>/bootstrap
Checking download tool sources
Downloading uv: 0.0 / 20.4 MiB; source official; inherited connection
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-26T18:28:33Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon Graphics and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards with 8 GB of video memory or more, and, as a preview, on Apple silicon Macs.



### #1168 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1168  by seerealserial-dotcom 2026-11-26T14:26:04Z

### What could be better? / 哪里可以做得更好？

im trying to create urdu videos but the dialogues in urdu the spelling the dictation is not proper...its telling :pehenchi rather than pahunchi , jannaat rather jinnat,

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

[im trying to create urdu videos but.txt](https://github.com/user-attachments/files/35468963/im.trying.to.create.urdu.videos.but.txt)

> **Comment by KBRASK (2026-11-26T21:01:26Z):**

Thank you for the report and for listing the words that came out wrong. Our analysis showed that in FreeVideo 0.6.13 the pronunciation of an Urdu line depends mainly on how it is written: in our tests, lines written in Urdu script were spoken much more accurately than the same lines written in Latin letters. Please write the spoken lines in Urdu script; a line that has to stay in Latin letters can be marked as Urdu by writing it as <d>[Urdu] ...</d> in the prompt, which reduced the errors somewhat in the same tests, though much less than writing the line in Urdu script. If a line written this way still sounds wrong, please send the exact prompt and tell us whether prompt enhancement was on.



### #1167 [open] ISSUE: [启动器] GPU 检查未通过
URL: https://github.com/FlashML-org/FreeVideo/issues/1167  by lengyourk-hub 2026-11-26T13:20:10Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

FreeVideo 启动器显示了一个错误。
RuntimeError: Failed step: kernels (exit code 1)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: NVIDIA GeForce RTX 5070 Ti Laptop GPU; AMD Radeon(TM) 610M

RuntimeError: Failed step: kernels (exit code 1)
Retained step log: <PATH>/00-kernels.log
Full step output:
{"event": "installation_step", "name": "GPU information reading", "done": 0, "total": 0}
{"event": "installation_step", "name": "GPU tests running", "done": 0, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 1, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 1, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 2, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 2, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 3, "total": 5}
{"event": "installation_step", "name": "GPU tests running", "done": 3, "total": 5}
{"event": "installation_step", "name": "GPU tests runni
...
ready": false,
  "usable_attention_backends": [],
  "int8_ready": true,
  "failed_optional_backends": [
    "cudnn",
    "sage2",
    "torch-flash"
  ],
  "identity": {
    "schema_version": 1,
    "hardware": {
      "gpu_name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU",
      "capability": [
        12,
        0
      ],
      "vram_total": 12792365056,
      "system": "Windows",
      "torch_version": "2.13.0+cu130",
      "cuda_version": "13.0",
      "gpu_uuid": "<token>",
      "driver_version": "621.14"
    },
    "python": "<PATH>/python.exe",
    "fa4_overlay_sha256": null,
    "packages": {
      "torch": "2.13.0+cu130",
      "triton": null,
      "triton-windows": "3.7.1.post27",
      "diffusers": "0.40.0.dev0",
      "sageattention": "2.2.0+cu130torch2.10.0andhigher.post6",
      "flash-attn": null,
      "flash-attn-4": null
    },
    "source_sha256": {
      "attention.py": "<token>",
      "backends/__init__.py": "<token>",
      "backends/base.py": "<token>",
      "backends/cuda.py": "<token>",
      "backends/cuda_attention.py": "<token>",
      "probe.py": "<token>",
      "fp8_gemm.py": "<token>",
      "fp8_ops.py": "<token>",
      "int8_ops.py": "<token>",
      "weight_only.py": "<token>",
      "kernel_capabilities.py": "<token>",
      "doctor.py": "<token>",
      "fa4_guard.py": "<token>",
      "triton_compat.py": "<token>",
      "runtime.py": "<token>",
      "head_chunk.py": "<token>",
      "blocks.py": "<token>",
      "packing.py": "<token>",
      "dependencies.json": "<token>"
    }
  },
  "view_root": "<PATH>/f5d39bf2",
  "temp_root": "<PATH>/uv",
  "receipt": "<PATH>/kernel-capabilities.json"
}

> **Comment by KBRASK (2026-11-26T19:29:44Z):**

Thank you for reporting this and for including the full step output and the GPU check summary.
Our analysis showed that the installation stopped at the GPU check: the int8 kernel check passed, but all three attention checks (cuDNN, Sage2 and Torch flash attention) failed, so no attention backend was usable and the step was reported as failed. The report does not show why each attention check failed, so we cannot yet say what caused it.
The retained step log named in the report (00-kernels.log) holds the output of each check; if you can attach that file to this issue, we can see why the three checks failed.



### #1166 [open] ISSUE: [Feedback] Generation did not finish: queue operation
URL: https://github.com/FlashML-org/FreeVideo/issues/1166  by ZPLEDGE 2026-11-26T13:16:45Z

### What could be better? / 哪里可以做得更好？

FreeVideo reported a generation failure.
Stage: queue operation
Error: Failed to fetch
Kind: connection

### Version / 版本

v0.6.12

### Diagnostic info (optional) / 诊断信息（选填）

{"kind":"connection","stage":"queue","exception_type":"TypeError","known_error":null,"nodes":[],"settings":{},"progress":{},"connection":{"online":true,"page_hidden":false,"secure_page":false,"engine_answers_after":false,"probe_ms":4,"page_address":"this-computer"},"frontend_version":"1.51.10","tail":"Failed to fetch\nQueue reads failed 3 times in a row over 4 s; online=true, page hidden=false.\n\nTypeError: Failed to fetch\n    at fetchWithUnifiedRemint (<URL REMOVED><INPUT REMOVED>:1882)\n    at ComfyApi.fetchApi (<URL REMOVED>\n    at fetchQueueRead (<URL REMOVED>\n    at readQueue (<URL REMOVED>\n    at <URL REMOVED>\n    at refresh (<URL REMOVED>\n    at <URL REMOVED>"}

> **Comment by KBRASK (2026-11-26T20:00:12Z):**

Thank you for sending the feedback report. It shows that the engine was no longer running when the page read the queue: three queue reads failed within 4 seconds, and the check made right afterwards found nothing answering on the engine's port. The report carries no stop record and no log text, so the cause of the stop cannot be determined yet. If it stops again, please attach the report saved with Export report in the launcher's Terminal panel before starting FreeVideo again, as it contains the engine log up to that stop, and please also tell us the video size and length you were generating.



### #1165 [open] ISSUE: [Report] RTX 3070 Ti Laptop GPU 8 GB · 1184×672 · 6.6 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1165  by laizeeguy 2026-11-26T11:20:34Z

### Diagnostic report / 诊断报告

RuntimeError
Node 1 (FreeVideoGenerate):
FreeVideo generation failed (exit 1)
Cause: TypeError: 'ResidualState' object is not subscriptable
Worker exited 1; see <PATH>/video.engine.log
 "blocks": 50, "estimated_step_seconds": 115.60766786200111}
{"event": "sampling_progress", "total": <INPUT REMOVED>, "completed_steps": 1, "uniform_remaining_steps": true, "elapsed_seconds": 224.16181411900106, "step_elapsed_seconds": 107.38626569700136, "stage": "layers", "step": <INPUT REMOVED>, "block": 48, "blocks": 50, "estimated_step_seconds": 115.60766786200111}
{"event": "sampling_progress", "total": <INPUT REMOVED>, "completed_steps": 1, "uniform_remaining_steps": true, "elapsed_seconds": 227.54315074199985, "step_elapsed_seconds": 110.76760232000015, "stage": "layers", "step": <INPUT REMOVED>, "block": 49, "blocks": 50, "estimated_step_seconds": 115.60766786200111}
{"event": "sampling_progress", "total": <INPUT REMOVED>, "completed_steps": 1, "uniform_remaining_steps": true, "elapsed_seconds": 230.86913720400116, "step_elapsed_seconds": 114.09358878200146, "stage": "layers", "step": <INPUT REMOVED>, "block": 50, "blocks": 50, "estimated_step_seconds": 115.60766786200111}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "<PATH>/worker.py", line 940, in <module>
    main()
  File "<PATH>/worker.py", line 33, in main
    generate(request)
  File "<PATH>/worker.py", line 732, in generate
    latents, audio, sampled = engine.sample(request['conditioning'], request['seed'],
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 590, in sample
    result = self._sample(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 940, in _sample
    latents, audio = generate_latents(self.transformer, prompt, tags, frames,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/<INPUT REMOVED>_sampler.py", line 186, in generate_latents
    noise_pred, audio_noise_pred = transformer(
                                   ^^^^^^^^^^^^
  File "<PATH>/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/module.py", line 1884, in _call_impl
    return inner()
           ^^^^^^^
  File "<PATH>/module.py", line 1843, in inner
    hook_result = hook(self, args, kwargs, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/fast_tier_runtime.py", line 117, in after_forward
    hidden = hidden if torch.is_tensor(hidden) else hidden[0]
                                                    ~~~~~~^^^
TypeError: 'ResidualState' object is not subscriptable

Stage: sample

Last resource-planning snapshot (after-encoding, before this failure; not the stopping sample):
Idle cache release: Release idle models before replanning; saved conditioning retained

generate.log:
  File "<PATH>/generate.py", line 1362, in launch
    telemetry = child([sys.executable, '-m', 'freevideo_engine.worker', '--request', str(request_path)],
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/generate.py", line 149, in child
    raise WorkerExit(code, log)
freevideo_engine.adaptive.WorkerExit: Worker exited 1; see <PATH>/video.engine.log
 "blocks": 50, "estimated_step_seconds": 115.60766786200111}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "<PATH>/worker.py", line 940, in <module>
    main()
  File "<PATH>/worker.py", line 33, in main
    generate(request)
  File "<PATH>/worker.py", line 732, in generate
    latents, audio, sampled = engine.sample(request['conditioning'], request['seed'],
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 590, in sample
    result = self._sample(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 940, in _sample
    latents, audio = generate_latents(self.transformer, prompt, tags, frames,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/<INPUT REMOVED>_sampler.py", line 186, in generate_latents
    noise_pred, audio_noise_pred = transformer(
                                   ^^^^^^^^^^^^
  File "<PATH>/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/module.py", line 1884, in _call_impl
    return inner()
           ^^^^^^^
  File "<PATH>/module.py", line 1843, in inner
    hook_result = hook(self, args, kwargs, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/fast_tier_runtime.py", line 117, in after_forward
    hidden = hidden if torch.is_tensor(hidden) else hidden[0]
                                                    ~~~~~~^^^
TypeError: 'ResidualState' object is not subscriptable

video.engine.log:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "<PATH>/worker.py", line 940, in <module>
    main()
  File "<PATH>/worker.py", line 33, in main
    generate(request)
  File "<PATH>/worker.py", line 732, in generate
    latents, audio, sampled = engine.sample(request['conditioning'], request['seed'],
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 590, in sample
    result = self._sample(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/runtime.py", line 940, in _sample
    latents, audio = generate_latents(self.transformer, prompt, tags, frames,
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/<INPUT REMOVED>_sampler.py", line 186, in generate_latents
    noise_pred, audio_noise_pred = transformer(
                                   ^^^^^^^^^^^^
  File "<PATH>/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/module.py", line 1884, in _call_impl
    return inner()
           ^^^^^^^
  File "<PATH>/module.py", line 1843, in inner
    hook_result = hook(self, args, kwargs, result)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/fast_tier_runtime.py", line 117, in after_forward
    hidden = hidden if torch.is_tensor(hidden) else hidden[0]
                                                    ~~~~~~^^^
TypeError: 'ResidualState' object is not subscriptable
FAST_MODE_FAILED: The fastest mode could not run.

Retained outputs: <PATH>/21578390fbcf4b86b01f77f45b8945c0


  File "<PATH>/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "<PATH>/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "<PATH>/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/job_record.py", line 86, in run
    output = execute(cls, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/comfy_nodes.py", line 431, in execute
    output = comfy_bridge.generate(text, width, height, seconds, seed, output_root,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<PATH>/comfy_bridge.py", line 1873, in generate
    raise RuntimeError(generation_failure(run, process.returncode, repair=hint))


### FreeVideo version / FreeVideo 版本

v0.6.13 (2026.11.26.15202)

### Graphics card and memory / 显卡和内存

RTX 3070 Ti Laptop GPU 8 GB · 46.7 GB RAM · Linux

### Video size and length / 视频尺寸和时长

1184×672 · 6.6 s

### Time taken / 用时

281 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-27T05:57:02Z):**

Thank you for sending the diagnostic report and the full error text. Our analysis showed that this run failed rather than ran slowly: the Fast level, on a card with 8 GB of video memory, keeps part of its working data in system memory between blocks, and when the last sampling step finished, about 231 s after sampling began, the next stage could not read that data and the worker stopped with the error you saw. The two sampling steps took about 115 s each. FreeVideo 0.6.14 reads that data correctly on such cards, so the same request now runs through to the finished video. Based on measurements on our test machines, we estimate the same request will take about 300 to 320 s in the new version, instead of failing after 281 s.



### #1164 [open] ISSUE: [Feedback] 提示词增强
URL: https://github.com/FlashML-org/FreeVideo/issues/1164  by sqpht-coder 2026-11-26T11:07:06Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Prompt enhancement
Error: 请等待当前队列完成后再改写。

### Version / 版本

v0.6.12

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Prompt enhancement","kind":null,"tail":"请等待当前队列完成后再改写。"}

> **Comment by KBRASK (2026-11-26T13:32:59Z):**

Thank you for reporting this. FreeVideo does not rewrite a prompt while a generation is running or waiting in the queue, and this message appears in that case; FreeVideo 0.6.13 behaves the same way. Please enhance the prompt before adding the generation to the queue, or wait until the queue is empty and enhance it then. If the message appears while nothing is running or waiting in the queue and no earlier enhancement is still running, please tell us, as that would be a different problem.



### #1163 [open] ISSUE: [Feedback] 编辑器
URL: https://github.com/FlashML-org/FreeVideo/issues/1163  by st467379674-ai 2026-11-26T10:59:02Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Editor
Error: 这一步没有完成。请重试；仍然失败时，请提交问题。
Kind: editor:editor_request

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Editor","kind":"editor:editor_request","tail":"这一步没有完成。请重试；仍然失败时，请提交问题。"}


### #1162 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1162  by sunyazhou90 2026-11-26T10:40:50Z

### What could be better? / 哪里可以做得更好？

左侧素材栏二级菜单打开后重叠严重，选半天选不中上传素材按钮，望尽快更新

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-27T03:54:36Z):**

Thank you for reporting the problem with the upload button. Our analysis showed that in 0.6.13 the Media window in the Create view was kept within the height of a floating window, so the added pictures and the Add tile were squeezed into a small scrolling box. On common laptop screens most of the Add tile was below the edge of that box when the window opened, and with many characters, scenes or props the lower part of the window also covered the settings below it. FreeVideo 0.6.14 shows the Media window at the full height of its contents, so the Add tile is fully visible and can be selected directly. In 0.6.13, scrolling the small box of pictures with the mouse wheel brings the Add tile into view; on a screen 1080 pixels high or more, maximizing the FreeVideo window also does in most cases.



### #1161 [open] ISSUE: [Report] RTX 4060 Laptop GPU 8 GB · 768×1344 · 5.2 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1161  by Lesya-20 2026-11-26T10:22:55Z

### Diagnostic report / 诊断报告

[FreeVideo-report-20261126-132149.json](https://github.com/user-attachments/files/35458817/FreeVideo-report-20261126-132149.json)

### FreeVideo version / FreeVideo 版本

v0.6.13 (2026.11.26.15432)

### Graphics card and memory / 显卡和内存

RTX 4060 Laptop GPU 8 GB · 15.7 GB RAM · Windows

### Video size and length / 视频尺寸和时长

768×1344 · 5.2 s

### Time taken / 用时

238 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-27T05:26:25Z):**

Thank you for sending the diagnostic report. Our analysis showed that this run did not finish slowly but failed: the Fast level, on a card with 8 GB of video memory, keeps part of its working data in system memory between blocks, and after the second sampling step, at about 238 s into the request, the last stage could not read that data and stopped with an error. The two sampling steps themselves took about 99 s and 96 s each. FreeVideo 0.6.14 reads that data correctly on such cards, so the same request now runs through to the finished video. Based on measurements on our test machines, we estimate the same request will take about 260 to 280 s in the new version, instead of failing after 238 s.



### #1160 [open] ISSUE: [Report] RTX 3060 12 GB · 1920×1088 · 15.1 s
URL: https://github.com/FlashML-org/FreeVideo/issues/1160  by moonkana 2026-11-26T09:45:45Z

### Diagnostic report / 诊断报告

[FreeVideo-report-20261126-173957.json](https://github.com/user-attachments/files/35456964/FreeVideo-report-20261126-173957.json)

### FreeVideo version / FreeVideo 版本

v0.6.13 (2026.11.26.15432)

### Graphics card and memory / 显卡和内存

RTX 3060 12 GB · 23.9 GB RAM · Windows

### Video size and length / 视频尺寸和时长

1920×1088 · 15.1 s

### Time taken / 用时

3994 s

### Anything else? (optional) / 其他补充（选填）

_No response_

> **Comment by KBRASK (2026-11-28T05:52:13Z):**

Thank you for sending the diagnostic report. Our analysis showed that the first pass took 2,686 s, at about 330 s per step, and that the generation then stopped in the second pass at 3,994 s, when the computer's free physical memory fell to 0.12 GiB. At this size the second pass needs about 10.2 GiB of system memory for its working data, while about 11.6 GiB was free when the request was planned and FreeVideo keeps 2 GiB of that in reserve. FreeVideo 0.6.14 works out this need before the request starts and declines a request that does not fit, stating the memory it needs, instead of stopping after more than an hour. On this computer the same request would be declined at the start; the same size fits at a length of up to about 13.7 s, or at the full length once about 12.5 GiB is free. Based on the first pass of this run, we estimate the full-length request will then take about 6,200 s.



### #1159 [open] ISSUE: [Launcher] No NVIDIA graphics card was found
URL: https://github.com/FlashML-org/FreeVideo/issues/1159  by rdsstudioweddingphoto-droid 2026-11-26T09:19:35Z

### FreeVideo version / FreeVideo 版本

v0.6.13

### What happened? / 遇到了什么问题？

The FreeVideo launcher showed an error.
Setup inspection failed (plan, exit 1, 5 output lines)

### Error or local report (optional) / 错误详情或本地报告（选填）

Environment: FreeVideo 0.6.13
Display adapters: AMD Radeon RX 6600

Setup inspection failed (plan, exit 1, 5 output lines)
{"event": "curl_diagnostic", "schema": "freevideo.curl-diagnostic", "schema_version": 1, "checked_at": "2026-11-26T09:17:39Z", "platform": "Windows-11-10.0.26300-SP0", "process_bits": 64, "candidates": [{"origin": "windows-sysnative", "path": "<PATH>/curl.exe", "status": "CURL_NOT_FOUND"}, {"origin": "windows-system32", "path": "<PATH>/curl.exe", "status": "ready", "returncode": 0, "windows_exit_code": "0x00000000", "version": "8.22.0"}], "selected": "<PATH>/curl.exe", "repair_attempts": [], "status": "ready", "repair": "not needed"}
Preparing isolated Python for setup. Model/environment installation requires the setup confirmation.
Bootstrap cache and log: <PATH>/bootstrap
Checking download tool sources
Downloading uv: 0.0 / 20.4 MiB; source official; inherited connection
Preflight failed: GPU detection failed. nvidia-smi: NVIDIA driver/nvidia-smi is missing. Install NVIDIA driver 580+ and restart before setup.; CUDA: Could not find module 'nvcuda.dll' (or one of its dependencies). Try using the full path with constructor syntax.. Check device access and the NVIDIA driver, then retry setup.

> **Comment by KBRASK (2026-11-26T10:59:50Z):**

Thank you for sending the launcher message. Setup stopped at the graphics check because the only graphics adapter on this computer is AMD Radeon RX 6600 and FreeVideo needs an NVIDIA graphics card, so installing a driver cannot change this. FreeVideo runs on NVIDIA RTX 30-series or newer cards with 8 GB of video memory or more, and, as a preview, on Apple silicon Macs.



### #1158 [open] ISSUE: [Feedback] 对话
URL: https://github.com/FlashML-org/FreeVideo/issues/1158  by w2008liupeng 2026-11-26T08:33:02Z

### What could be better? / 哪里可以做得更好？

FreeVideo 显示了一个错误。
Where: Chat
Error: 生成过程中出现了意外错误。再次出现时，请点击“提交问题”

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

{"where":"Chat","kind":null,"tail":"生成过程中出现了意外错误。再次出现时，请点击“提交问题”\ngeneration_failed\nGeneration failed; completed blocks are retained.\nr-241f6dc646be4576b21021c7db096f22"}

> **Comment by KBRASK (2026-11-26T13:02:23Z):**

Thank you for reporting this. The failure message in Chat in FreeVideo 0.6.13 did not include the reason, so we cannot tell from it why this generation failed. To find the run's own record, open Creations at the top right and click Video folder at the bottom of the left column; in the date folder from that day or the day before, the folder whose modified time matches the failure holds generate.log and usually video.debug.json, and attaching either file here would show the reason. If there is no folder from that time, please tell us the size, length and quality level of the video and your graphics card.



### #1157 [open] ISSUE: [Feedback]
URL: https://github.com/FlashML-org/FreeVideo/issues/1157  by alod0685-png 2026-11-26T04:51:45Z

### What could be better? / 哪里可以做得更好？

сломали интерфейс в обновлении. В левой части ничего не видно теперь. прошлый вариант был лучше

### Version / 版本

v0.6.13

### Diagnostic info (optional) / 诊断信息（选填）

_No response_

> **Comment by KBRASK (2026-11-26T12:01:15Z):**

Thank you for the feedback about the create panel. Our analysis showed that in 0.6.13, when the browser window is narrower than 1280 pixels, for example on a laptop with display scaling, the create panel folds the settings column into a strip of icons, so the prompt, the settings and the Generate button are hidden. The top button of the left strip (Show settings) opens the settings column with the prompt and Generate, and a browser window at least 1280 pixels wide, or a lower browser zoom, keeps it open.

