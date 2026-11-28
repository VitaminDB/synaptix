# Cseti/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1

Источник: https://huggingface.co/Cseti/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1 · скачано 2026-11-28
Загрузок: 11003 · лайков: 103 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги: lora · обновлено: 2026-09-20T19:23:11.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
tags:
- video-generation
- lora
- minimax-h3
- ref2va
- novel-view-synthesis
- camera-viewpoint
- re-camera
- depth-warp
- moge
widget:
- text: "crossview"
  output:
    url: assets/compare_00004.mp4
- text: "crossview"
  output:
    url: assets/compare_00005.mp4
- text: "crossview"
  output:
    url: assets/compare_00006.mp4
- text: "crossview"
  output:
    url: assets/compare_00007.mp4
- text: "crossview"
  output:
    url: assets/compare_00008.mp4
- text: "crossview"
  output:
    url: assets/compare_00009.mp4
- text: "crossview"
  output:
    url: assets/compare_00010.mp4
- text: "crossview"
  output:
    url: assets/compare_00011.mp4
- text: "crossview"
  output:
    url: assets/compare_00012.mp4
- text: "crossview"
  output:
    url: assets/compare_00013.mp4
- text: "crossview"
  output:
    url: assets/compare_00014.mp4
- text: "crossview"
  output:
    url: assets/compare_00015.mp4
- text: "crossview"
  output:
    url: assets/compare_00016.mp4
- text: "crossview"
  output:
    url: assets/compare_00017.mp4
---

# MiniMax-H3 — Ref2VA LoRA: CrossView Warp v1

Give it a video and a camera offset — azimuth, elevation — and it generates the
same scene from that new viewpoint.

This is the [CrossView-Warp](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2)
idea ported to **MiniMax-H3**. The model reads two videos: a
depth-warp of your clip, which carries the geometry, and the clip itself, which
carries identity and appearance. The warp comes from the
[**CrossViewWarp ComfyUI node**](https://github.com/cseti007/ComfyUI-CrossViewWarp).

## Examples

<Gallery />

The `compare_*` clips show the warp, the original and the result side by side.
## Usage (ComfyUI)

**LoRA strength:** 0.8 - 1.0 (0.8 is recommended)

**Trigger word:** crossview (note: You can also influence the generation by prompting what you want to see in the unseen (magenta) areas.)

**Example ComfyUI workflow**:
[**crossview-warp-h3.json**](https://huggingface.co/datasets/Cseti/ComfyUI-Workflows/blob/main/minimax-h3/crossview-warp/README.md)

The [**CrossViewWarp node**](https://github.com/cseti007/ComfyUI-CrossViewWarp) has a
walkthrough video that covers the camera controls; everything there applies unchanged:

[![Watch on YouTube](https://img.youtube.com/vi/7QAapT9xMgM/maxresdefault.jpg)](https://www.youtube.com/watch?v=7QAapT9xMgM)

The steps below are what it wires up.

1. Install the [ComfyUI-CrossViewWarp](https://github.com/cseti007/ComfyUI-CrossViewWarp) node.
2. Feed your clip to **Run MoGe Inference**, and its geometry to **CrossView Warp**.
3. Load `MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1_3500.safetensors` as a LoRA on the
   **ref2va** checkpoint.
4. Warp -> **MiniMax H3 Add Guide**, `frame_idx = 0`.
   Source -> **MiniMax H3 Reference To Video**
5. **Resize warp and source to the same size** before both nodes.
6. The trigger word: **`crossview`**. You can also influence the generation by prompting what you want to see in the unseen (magenta) areas.

Settings the released examples were generated with:

|                       |                                                         |
| --------------------- | ------------------------------------------------------- |
| Frames                | 124                                                     |
| Resolution (1st pass) | 0.5 MP, 16:9                                            |
| Resolution (2st pass) | 1.5 MP, 16:9                                            |
| Sampler / steps       | `res_multistep`, 8 steps with the DMD 8-step turbo LoRA |
| Sigma shift           | 12 video / 3 audio                                      |
| LoRA strength         | 0.8                                                     |

## Dataset details

The same 719 Blender scenes as the LTX v2 release. If you want to know more about the dataset you can find more information in my [LTX crossview-warp repo:](https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Warp_v2)

Third-party assets in the renders (CC-BY 3D models, CC0 HDRIs and textures, CMU
motion capture) are listed in **[ATTRIBUTION.md](ATTRIBUTION.md)**.

## Training

Trained on RunPod — NVIDIA RTX PRO 6000 Blackwell, 96 GB.

|                     |                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| Base model          | MiniMax-H3, ref2va                                                                             |
| Framework           | [AkaneTendo25/musubi-tuner](https://github.com/AkaneTendo25/musubi-tuner), branch `minimax-h3` |
| Strategy            | Ref2VA, `aligned_guide_indices = [0]`                                                          |
| Released checkpoint | **step 3,500** of 6,000                                                                        |
| LoRA rank / alpha   | 32 / 32                                                                                        |
| Optimizer           | adamw8bit 1e-4                                                   |
| Base precision      | INT8 ConvRot                                                             |
| Resolution          | 512×512 × 73 frames                                                                            |
| Schedule            | 6,000 steps, batch 1, no gradient accumulation                                                 |
| Measured            | 20.35 s/step, 33 h 55 min, 38.4 of 95.6 GB peak                                                |

## License

The **LoRA weights** are a Model Derivative of MiniMax H3 and are distributed under
the **[MiniMax H3 Community License Agreement](LICENSE)**

## Support

Everything here is open, and the GPUs behind it are rented. If this was useful,
please consider supporting my work:

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/chetyart) [![Liberapay](https://img.shields.io/badge/Liberapay-Donate-F6C915?style=for-the-badge&logo=liberapay&logoColor=black)](https://liberapay.com/chetyart/donate)
