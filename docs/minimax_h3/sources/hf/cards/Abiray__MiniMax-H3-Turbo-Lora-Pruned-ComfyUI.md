# Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI

Источник: https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI · скачано 2026-11-28
Загрузок: 5249 · лайков: 54 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui,base_model:quantized:MiniMaxAI/MiniMax-H3 · обновлено: 2026-08-09T07:02:23.000Z

---

---
pipeline_tag: image-text-to-video
license: other
license_name: minimax-h3-community-license-agreement
license_link: LICENSE
library_name: diffusers
tags:
  - text-to-video
  - image-to-video
  - image-text-to-video
  - video-to-video
  - text-to-audio-video
  - multimodal
  - comfyui
base_model: MiniMaxAI/MiniMax-H3
base_model_relation: quantized
---

# MiniMax-H3 Turbo LoRA (ComfyUI Pruned)
<video controls autoplay loop muted style="width: 100%;" src="https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/asset/MiniMax_H3_00003-audio.mp4"></video>
This is a **ComfyUI-ready, pruned version** of the MiniMax-H3 Turbo LoRA. 

It is designed to dramatically accelerate both video and audio generation for the MiniMax-H3 model. By using this adapter, you can drastically cut down the required sampling steps while still generating native, synchronized 32kHz stereo audio alongside your video.

### 🙏 Acknowledgements & Credits
* **Original Trainer & Model Creator:** [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
---
## 🚀 What Makes This Version Different?

The initial release of this LoRA required a standalone custom Python script because standard ComfyUI samplers struggled to handle the dual-schedule (video + audio) mechanics without blowing out the audio track.

**This specific `.safetensors` file has been structurally pruned and reformatted so it works seamlessly inside standard ComfyUI workflows.** You can load it using a standard `Load LoRA` node. It is optimized to run alongside the pruned or curve-form MiniMax-H3 checkpoints designed for ComfyUI.

---

## 📁 Installation & Setup

To get started, simply download LoRA file and place it into your ComfyUI models directory:

`ComfyUI/models/loras/`
 
 Workflow in the repo  https://huggingface.co/Abiray/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/Minimax_H3_turbo_workflow.json

---

## ⚙️ Recommended ComfyUI Settings

For the best visual quality and properly synced audio, we highly recommend dialing in the following parameters in your workflow:

* **Base Model:** Use the pruned or curve-form MiniMax-H3 ComfyUI checkpoint.
* **Video Sigma Shift:** `12`
* **Audio Sigma Shift:** `6`
* **Steps:** `8 to 12`
* **Sampler:** `res_multistep`
* **LoRA Strength:** `0.8 to 1.8`

> 💡 **Pro-Tip on Steps vs. Strength:** You can generally reduce your sampling steps even further by cranking up the LoRA strength. If you are comparing different models or files, make sure to lock in your seed, prompt, resolution, sampler, and workflow to ensure a fair test.

Want to speed up your generation times even more? This workflow has been tested and confirmed to work with the following optional ComfyUI attention and execution accelerators:
* **SageAttention**
* **Sol Attention**
* **Gradient**
* **Spectrum**
---

## 🔊 Audio Troubleshooting & Mismatches

Because MiniMax-H3 synthesizes video and audio simultaneously using two completely separate generation schedules, it is highly sensitive to sampler configurations. 

If your output video contains broken, distorted, or "static noise" audio, **do not immediately assume the LoRA file is broken.** 

Almost all audio issues stem from a scheduler mismatch. Before diving into complex troubleshooting, verify your setup:
1. Double-check that your **Audio Sigma Shift is exactly `6`** and your **Video Sigma Shift is `12`**.
2. Confirm you are using the correct sampler and scheduler combination.
3. Ensure you haven't pushed the step count too low without appropriately balancing the LoRA strength. 
