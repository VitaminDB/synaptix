# tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA

Источник: https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA · скачано 2026-11-28
Загрузок: 0 · лайков: 19 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-08-09T08:11:37.000Z

---

---
language:
  - zh
  - en
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
pipeline_tag: image-text-to-video
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: adapter
tags:
  - lora
  - comfyui
  - minimax-h3
  - text-to-video
  - image-to-video
  - audio-video-generation
  - synchronized-audio-video
  - few-step
  - 8-nfe
  - research
---

# Tutu MiniMax-H3 Audio-Video 20→8 NFE LoRA

## 云扉工坊算力支持 / Compute Support

非常感谢 **云扉**工坊在本次训练中提供算力支持。**云扉**工坊长期提供可靠、稳定、高性价比的 GPU 算力租赁服务，服务于 200 余家高校科研院所，覆盖国内所有 985、211 高校及知名企业。

**智算云扉注册链接：**

https://waas.aigate.cc/productService?channel=H1Q5G7P0

通过我的链接注册赠送 **2 个月会员**，会员使用 RTX 5090 仅需 **2.59 元/小时**！新用户首充 **10 元到账 20 元**！

感谢 **T8 老师** 在训练过程中提供的指导与支持。

This training was made possible with GPU compute support from **Yunfei Workshop (云扉工坊)**. We sincerely thank Yunfei Workshop for supporting this project. It provides reliable, stable, and cost-effective GPU rentals to more than 200 universities and research institutes, covering all of China's 985 and 211 universities as well as well-known enterprises.

Register through the link above to receive two months of membership. Members can use RTX 5090 GPUs for RMB 2.59 per hour, and new users who first top up RMB 10 receive RMB 20 in account credit.

Special thanks to **T8** for guidance and support during training.

## Model overview / 模型简介

Created by **Tutu**.

这是一个面向 **MiniMax-H3 FL2VA** 的联合音视频加速 LoRA，将 20 NFE 生成流程缩短为固定的 8 NFE Euler 流程，同时保留 MiniMax-H3 的原生双声道音频、中文对白和同步音效能力。

This is an audio-video acceleration LoRA for **MiniMax-H3 FL2VA**. It reduces the 20-NFE generation process to a fixed 8-NFE Euler workflow while retaining MiniMax-H3's native stereo audio, dialogue, and synchronized sound effects.

## License / 许可

本仓库遵循上游 **MiniMax H3 Community License Agreement**。下载、使用或分发前请阅读 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

This repository follows the upstream **MiniMax H3 Community License Agreement**. Read [LICENSE](LICENSE) and [NOTICE](NOTICE) before downloading, using, or redistributing the files.

## Checkpoints / 检查点

仓库同时发布 step100、step200、step300 三个检查点，方便用户根据自己的底模、提示词和画面类型进行比较。文件名中的 step 数字表示训练更新次数，不是推理步数；三个检查点推理时均为 8 NFE。

The repository includes step100, step200, and step300 so users can compare them on their own base model, prompt, and scene. The step number is the number of training updates, not inference steps; all three checkpoints run at 8 NFE.

| Full checkpoint name | Recommended use | ComfyUI file | Diffusers/PEFT adapter |
|---|---|---|---|
| Tutu MiniMax-H3 Audio-Video 20→8 NFE LoRA — Step 100 | Recommended starting point | `comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors` | `diffusers/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100/pytorch_lora_weights.safetensors` |
| Tutu MiniMax-H3 Audio-Video 20→8 NFE LoRA — Step 200 | Alternative comparison checkpoint | `comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors` | `diffusers/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200/pytorch_lora_weights.safetensors` |
| Tutu MiniMax-H3 Audio-Video 20→8 NFE LoRA — Step 300 | Alternative comparison checkpoint | `comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors` | `diffusers/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300/pytorch_lora_weights.safetensors` |

建议先使用 **step100、LoRA 强度 0.8**，再用相同提示词和种子比较 step200、step300。不同场景可能适合不同检查点，请以实际结果为准。

Start with **step100 at LoRA strength 0.8**, then compare step200 and step300 using the same prompt and seed. Different scenes may favor different checkpoints.

## ComfyUI usage / ComfyUI 使用方法

无需安装第三方采样节点。请使用包含 MiniMax-H3 原生支持的新版 ComfyUI，并通过内置的 `LoraLoaderModelOnly` 加载 LoRA。

No third-party sampling node is required. Use a recent ComfyUI build with native MiniMax-H3 support and load the LoRA through the built-in `LoraLoaderModelOnly` node.

推荐从 ComfyUI 官方 MiniMax-H3 工作流开始：

- [官方 MiniMax-H3 T2V 工作流 JSON](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
- [官方 MiniMax-H3 I2V 工作流 JSON](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json)
- [Comfy-Org 官方模型文件](https://huggingface.co/Comfy-Org/MiniMax-H3)

下载一个 ComfyUI 版 LoRA 到 `ComfyUI/models/loras/`，在官方工作流的底模后接入 `LoraLoaderModelOnly`。官方工作流是 20 步基线，加载本 LoRA 后还必须将采样部分改为以下固定设置：

- Sampler: `Euler`
- NFE: `8`
- Video shift: `12.0`
- Audio shift: `3.0`
- Sigmas: use the fixed `ManualSigmas` list below

Start from the official [T2V](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) or [I2V](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json) workflow and download the required base files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). Put one ComfyUI LoRA file in `ComfyUI/models/loras/` and add `LoraLoaderModelOnly` after the base-model loader. The official workflows are 20-step baselines, so their sampling section must be changed to the fixed settings above. The separate third-party Dual-Clock T8 sampler is not required.

<details>
<summary>Manual Sigmas / 固定 Sigma</summary>

```text
1.0,
0.9855073094367981,
0.9729729890823364,
0.9473683834075928,
0.9230769276618958,
0.8659793734550476,
0.800000011920929,
0.5714285969734192,
0.0
```

</details>

机器可读配置见 [sampling_contract.json](sampling_contract.json)。不要用普通均匀 8 步日程替换这组 Sigma。

See [sampling_contract.json](sampling_contract.json) for the machine-readable configuration. Do not replace these sigmas with a generic uniform 8-step schedule.

## Other runtimes (PEFT) / 非 ComfyUI 用法

每个 `diffusers/...` 目录都包含一套 PEFT 格式的 `adapter_config.json` 和 `pytorch_lora_weights.safetensors`。它们是 LoRA 适配器，不是可独立运行的完整模型。请在明确支持 MiniMax-H3 FL2VA 与 PEFT LoRA 的推理实现中加载官方底模，再选择一个检查点目录，并复现上面的 Euler、12/3 shift 与固定 Sigma 日程。

目前不同框架对 H3 的加载接口并不统一，因此这里不提供可能误导用户的通用一行代码。如果目标框架无法完整映射适配器张量，或无法显式使用这组 Sigma，结果就不能与本项目的 ComfyUI 测试直接比较。请以该框架的 H3/PEFT 加载说明和实际加载日志为准。

Each `diffusers/...` directory contains a PEFT-format `adapter_config.json` and `pytorch_lora_weights.safetensors`. These are LoRA adapters, not standalone models. Load the official MiniMax-H3 FL2VA base in an implementation that explicitly supports H3 and PEFT LoRA, select one checkpoint directory, and reproduce the Euler sampler, 12/3 shifts, and fixed sigma schedule above.

H3 loading interfaces currently vary across runtimes, so this card does not give a generic one-line snippet that may be incorrect. If a runtime cannot map the adapter tensors or explicitly reproduce the sigma schedule, its output is not directly comparable with the ComfyUI tests. Follow that implementation's H3/PEFT loader documentation and inspect its loading report.

## Base model compatibility / 底模兼容性

本 LoRA 以官方 full-BF16 MiniMax-H3 FL2VA 为训练基准。作者测试中，标准 FP8、INT8 ConvRot、pruned INT8 ConvRot 等多种 H3 FL2VA 底模均能观察到 LoRA 效果。部分量化或裁剪版本可能提示个别适配器张量未匹配或被跳过，但兼容层仍会正常应用，因此此类提示不等于 LoRA 完全未生效。

不同底模的效果和推荐强度可能不同，请结合公开对比视频，或在相同提示词和种子下自行比较。本仓库仅包含 LoRA，不包含 MiniMax-H3 底模、文本编码器、视频 VAE 或音频 VAE。

This LoRA was trained against the official full-BF16 MiniMax-H3 FL2VA base. In the author's tests, a visible LoRA effect was observed with multiple H3 FL2VA variants, including standard FP8, INT8 ConvRot, and pruned INT8 ConvRot bases. Some quantized or pruned variants may report unmatched or skipped adapter tensors while compatible layers still apply normally; such messages do not necessarily mean that the LoRA had no effect.

Results and recommended strength can vary by base model. Compare the public test videos or run the same prompt and seed with and without the LoRA. This repository contains the LoRA adapter only, not the MiniMax-H3 base model, text encoder, video VAE, or audio VAE.

## Supported scope / 支持范围

- T2VA, I2VA, L2VA, and FL2VA generation;
- landscape, portrait, and square formats;
- approximately 5–15 second clips at 24 fps;
- ambient sound, Chinese dialogue, multi-speaker scenes, and synchronized sound effects.

支持文生音视频、图生音视频、首尾帧音视频等 H3 FL2VA 工作方式，覆盖横屏、竖屏、方形画幅，约 5–15 秒的 24 fps 视频，以及环境声、中文对白、多人对白与同步音效。

## Test videos / 测试视频

完整对比测试视频会发布在我的 Bilibili 和 YouTube 主页。建议观看同提示词、同种子的教师 20 步、裸 8 步与 LoRA 8 步三联对比后自行判断。

Full comparison videos will be published on my Bilibili and YouTube homepages. Please judge the model from the side-by-side teacher 20-step, bare 8-step, and LoRA 8-step results made with matching prompts and seeds.

- Bilibili: [天选打工人赵图图](https://space.bilibili.com/431046154)
- YouTube: [@zhaotutu](https://www.youtube.com/@zhaotutu)

## Creator links / 作者链接

- Website: [zhaotutu.xyz](https://zhaotutu.xyz/)
- Bilibili: [天选打工人赵图图](https://space.bilibili.com/431046154)
- YouTube: [@zhaotutu](https://www.youtube.com/@zhaotutu)
- X: [@bigpox12](https://x.com/bigpox12)
- GitHub: [zhaotututu](https://github.com/zhaotututu)
- Telegram: [@zhaotutu](https://t.me/zhaotutu)
- TikTok: [@zhaotututu](https://www.tiktok.com/@zhaotututu)

## 友情支持 T8 Star / Friendly Support

感谢 **T8 Star** 老师在本次训练过程中提供指导与支持。

Special thanks to **T8 Star** for guidance and support during training.

- Bilibili: [T8star-Aix](https://space.bilibili.com/385085361)
- YouTube: [@T8star-Aix](https://www.youtube.com/@T8star-Aix)

## Notes / 使用提示

- 不同底模、检查点、LoRA 强度和提示词会产生不同结果，建议使用相同种子进行对比。
- 当前版本在快速运动中仍可能出现可见残影，手部、手指、工具交互以及遮挡前后尤其明显；部分复杂镜头运动也会出现运动拖影。
- 正式使用前请预览画面并试听生成音频。

- Results vary with the base model, checkpoint, LoRA strength, and prompt; compare with the same seed when possible.
- The current release can still show visible ghosting in fast motion, especially around hands, fingers, tool interaction, and occlusion boundaries; complex camera motion may also leave motion trails.
- Preview both the video and generated audio before production use.

## Next training plan / 下一步训练计划

下一阶段会增加高速动作、手部与工具交互、遮挡和碰撞等困难样本，并探索在现有轨迹蒸馏基础上加入 on-policy DMD/rCM 类训练。我们会先以约 5 秒、约 1MP 的稳定质量为门槛，再逐步扩展图生视频和 10–15 秒长视频。

The next stage will add harder fast-motion, hand/tool interaction, occlusion, and collision samples, while exploring on-policy DMD/rCM-style training alongside the current trajectory distillation. We will first gate quality on stable ~5-second, ~1MP results, then expand toward image-to-video and 10–15 second clips.

## File integrity / 文件校验

下载后可使用 [SHA256SUMS.txt](SHA256SUMS.txt) 校验文件。哈希值由本地最终候选文件生成。

Use [SHA256SUMS.txt](SHA256SUMS.txt) to verify downloads. The hashes were generated from the final local release candidates.

## Intended use / 预期用途

本模型用于 MiniMax-H3 少步数联合音视频生成的研究、测试与内容创作。用户应遵守上游许可、适用法律、AI 内容标识要求与第三方权利。

This model is intended for research, evaluation, and content creation with few-step MiniMax-H3 audio-video generation. Users are responsible for complying with the upstream license, applicable law, AI-content disclosure requirements, and third-party rights.

## Attribution / 署名

Base model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3).

This repository is an independent community derivative created by **Tutu** and is not an official MiniMax release.
