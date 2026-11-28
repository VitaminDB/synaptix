# huangserva__ComfyUI_MiniMaxH3_Director

Source: https://github.com/huangserva/ComfyUI_MiniMaxH3_Director
Fetched: 2026-11-28

Description: ComfyUI MiniMax H3 Director workflow
Stars: 1585  Updated: 2026-08-04T10:45:16Z



---

## FILE: README.md (https://github.com/huangserva/ComfyUI_MiniMaxH3_Director/blob/HEAD/README.md)

# MiniMax H3 Director 工作流

这个仓库保存 5 份可直接导入 ComfyUI 的 MiniMax H3 Director 工作流，覆盖文生视频、图生视频、首尾帧、参考素材生视频、视频编辑和参考素材改视频。

工作流来自 [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)。这里保留一套方便下载、复现和测试的副本。

## 已验证环境

- NVIDIA RTX 4090 48GB
- ComfyUI 0.30.0
- PyTorch 2.11.0 + CUDA 12.8
- MiniMax H3 Ref2VA INT8
- `MiniMaxH3Director` 节点已成功注册

## 工作流

| 文件 | 模式 | UNET | 用途 |
| --- | --- | --- | --- |
| [minimax_h3_director_t2v.json](example_workflows/minimax_h3_director_t2v.json) | T2V | fl2va | 文生音视频 |
| [minimax_h3_director_fl2v.json](example_workflows/minimax_h3_director_fl2v.json) | FL2V / I2V | fl2va | 首尾帧；只放首帧时作为 I2V 使用 |
| [minimax_h3_director_r2v.json](example_workflows/minimax_h3_director_r2v.json) | R2V | ref2va | 图片、视频、音频参考生成 |
| [minimax_h3_director_v2v.json](example_workflows/minimax_h3_director_v2v.json) | V2V | ref2va | 按源视频时间轴逐段编辑 |
| [minimax_h3_director_rv2v.json](example_workflows/minimax_h3_director_rv2v.json) | RV2V | ref2va | 源视频加参考图或参考音频，适合测试视频换人 |

## 安装

需要 ComfyUI 0.30.0 或更高版本，并安装上游导演台节点：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git
python -m pip install -r ComfyUI_MiniMaxH3_Director/requirements.txt
```

重启 ComfyUI，将 `example_workflows/` 里的 JSON 拖进页面。

## 模型要求

T2V、I2V、FL2V 使用：

```text
models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

R2V、V2V、RV2V 使用：

```text
models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
```

共用组件：

```text
models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
models/vae/minimax_h3_video_vae_fp16.safetensors
models/vae/minimax_h3_audio_vae_fp32.safetensors
```

CLIP Loader 的类型选择 `minimax`。

## RV2V 视频换人

1. 导入 [RV2V 工作流](example_workflows/minimax_h3_director_rv2v.json)。
2. 上传源视频，按镜头切分或使用智能分割。
3. 上传人物参考图；需要声音约束时再加参考音频。
4. 每段分别写提示词。源片段会绑定为 `<Video 1>`，参考素材使用 `<Picture N>`、`<Audio J>`。
5. 先生成 5 秒，检查脸部、服装、动作和镜头边界，再决定是否扩展。

导演台支持逐段选择运行和缓存，改一段时不必重跑整条时间轴。音频可以选择模型生成、沿用原声或静音。

## 使用说明

- 当前已验证的 4090 环境只有 Ref2VA，可直接运行 R2V、V2V、RV2V。
- T2V、I2V、FL2V 需要补齐 fl2va 权重。
- 人物身份由模型和参考素材共同约束，导演台没有硬身份锁。
- 段间引导采用上一段末帧到下一段首帧的交接，不能替代人物一致性检查。
- 做 A/B 测试时固定源素材、提示词、seed、分辨率、帧数和 steps。

## SageAttention（可选）

安装 SageAttention 后，可将补丁节点放在 UNETLoader 与 `MiniMaxH3Director` 的 `model` 输入之间。先确认输出质量一致，再记录速度变化。

## 来源与许可

- 上游插件：[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- MiniMax H3 权重：[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- ComfyUI 文档：[MiniMax H3 工作流](https://docs.comfy.org/zh/tutorials/video/minimax/minimax-h3)

工作流与上游插件按 Apache-2.0 许可发布，详见 [LICENSE](LICENSE)。模型权重遵循各自的许可条款。


---

# ISSUES / PRs of huangserva/ComfyUI_MiniMaxH3_Director


### #2 [open] ISSUE: 参考视频被分段之后，只有第一段被参考了？
URL: https://github.com/huangserva/ComfyUI_MiniMaxH3_Director/issues/2  by AlexTheMonster 2026-09-06T15:26:55Z

我在做视频换人，为了缓解显卡压力分成若干段，但是跑完只有第一段保持了原视频的动作，后面全是ai自由发挥了，请问是bug还是我的操作不对？


### #1 [open] ISSUE: 能加一下lora加速模型在工作流中吗
URL: https://github.com/huangserva/ComfyUI_MiniMaxH3_Director/issues/1  by tyshine9 2026-08-17T00:32:57Z


