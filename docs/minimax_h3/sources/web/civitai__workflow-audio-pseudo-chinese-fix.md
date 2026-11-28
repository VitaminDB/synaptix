# Source: https://civitai.com/models/2863848

API: https://civitai.com/api/v1/models/2863848

Fetched: 2026-11-28 (Civitai public API, description converted to text)

---

## MiniMax H3 - Original Audio Lip-Sync Workflow
Type: ComfyWorkflows  Creator: sanmarinogmip394  Tags: base model, comfyui, workflow, lip sync, minimax h3, audio sync, video to video / reference to video
Stats: {"downloadCount": 796, "thumbsUpCount": 18, "thumbsDownCount": 0, "commentCount": 0, "tippedAmountCount": 0}

### Description

MiniMax H3 Original MP3 Lip-Sync Workflow Fixes the issue where input audio is regenerated into pseudo-Chinese vocals. This workflow passes through your original MP3/vocal directly to the final video output while maintaining H3's precise lip-sync.
- Generation Speed: ~228s for 10-second video (INT8/NVFP4 environment)

- No missing custom nodes needed, standard setup.

-

### Version: v1.3 (base: MiniMax H3, created 2026-08-24)
Files: MiniMax H3 - Original Audio Lip-Sync  reference TURBO UPS Workflow.json (0.0 MB)

# MiniMax H3 Turbo Workflow (Reference to Video + Audio Lip-Sync + Upscalers)
This is an optimized ComfyUI workflow for MiniMax H3 Reference-to-Video enhanced with Turbo LoRA and integrated Upscaler pipelines (4x-AnimeSharp & RTX Video Super Resolution).
Generated 15s high-quality character dancing/singing videos in ~180s - 200s (RTX 4090 / 5090) with consistent character appearance!
---
## 📺 YouTube Tutorial & Comparison Review
Watch the full setup tutorial, comparison with LTX 2.3, and upscaler performance benchmarks:
👉https://youtu.be/Bw26PvpolYM?si=a603fI8puJtRaIrl
---
## ✨ Key Features & Improvements
1. Turbo LoRA Acceleration: Reduces sampling steps from 20+ down to 6-8 steps res_multistep), cutting generation time by more than 50%.
2. Audio Lip-Sync & Music Video Support: Fixes previous audio replacement issues. Simply drag-and-drop your custom MP3/WAV into the workflow.
3. Built-in Upscaling Pipelines:
- RTX Video Super Resolution (VSR): Ultra-fast real-time AI upscaling, ideal for rapid iterations and previews.
- 4x-AnimeSharp + ImageScale: Supreme line-art reconstruction and edge clarity for final masterpieces.
4. Strong Character Consistency: Reference image identity is preserved throughout the 15-second performance.
---
## 🛠️ Required Models & Custom Nodes
### 1. Base Models (MiniMax H3)
Place in their respective standard ComfyUI folders models/unet, models/clip, models/vae):
- UNet: minimax_h3_ref2va_pruned_int8_convrot.safetensors (or fp8/fp16)
- Text Encoder: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- Video VAE: minimax_h3_video_vae_fp16.safetensors
- Audio VAE: minimax_h3_audio_vae_fp32.safetensors
### 2. Turbo LoRA
- minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors
- Download from: [Hugging Face (drbaph)](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/tree/main)
- Place into: ComfyUI/models/loras/
### 3. Custom Nodes
Run the following in ComfyUI/custom_nodes/:
```bash
git clone [https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
git clone [https://github.com/Larryvrh/ComfyUI-H3-MultiShot](https://github.com/Larryvrh/ComfyUI-H3-MultiShot)

### Version: v1.2 (base: MiniMax H3, created 2026-08-21)
Files: MiniMax H3 - Original Audio Lip-Sync  reference  TURBO　Workflow.json (0.0 MB)

This workflow is a ultra-fast MiniMax H3 Reference-to-Video pipeline accelerated by Turbo LoRA (6~8 Steps). It enables you to generate a 15-second high-quality reference video with lip-sync in under 200 seconds (tested on RTX 5090). (MiniMax H3のReference-to-VideoをTurbo LoRAで高速化し、15秒動画を約200秒弱で生成できる爆速ワークフローです。)
【Required Models / 必要なモデル】
- Reference Diffusion Model:
- minimax_h3_ref2va_pruned_int8_convrot.safetensors (or fp8 version)

- Save to: ComfyUI/models/diffusion_models/ (or models/unet/)


- Turbo LoRA Model:
- minimax_h3_turbo_v4_step600_ema_prun...safetensors

- Save to: ComfyUI/models/loras/


- Text Encoders & VAE:
- Standard MiniMax H3 CLIP/Text Encoders in models/clip/ and VAEs in models/vae/.


【Custom Nodes Installation / カスタムノード導入手順】
Open your command prompt in the ComfyUI/custom_nodes directory and run the following commands: (ComfyUIの custom_nodes フォルダを開き、アドレスバーに cmd と入力してEnterを押した後、以下のコマンドを実行してください)
command1：git clone https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
command2：git clone https://github.com/jlucasmcrell/ComfyUI-H3-Multishot
Restart ComfyUI after cloning. (クローン完了後、ComfyUIを再起動してください)
【Recommended Settings / 推奨設定】
- Sampler: MiniMax-H3 Turbo Sampler

- Scheduler: simple

- Steps: 6 - 8

- LoRA Strength: 1.0

- Duration: 15.0 seconds (at 24 FPS)

【Credits】
- MiniMax H3 Turbo Node: Larryvrh

- MultiShot Node: jlucasmcrell

### Version: V1.1 (base: MiniMax H3, created 2026-08-17)
Files: MiniMax H3 I2V - Simple Audio Lip-Sync Workflow.json (0.0 MB)

Update: Added I2V (Image to Video) Simple Version!
- Perfect for single-image animation with original MP3 lip-sync.
- Cleaner and lighter node setup compared to R2V.
- Same zero audio corruption / perfect Japanese vocal sync.

### Version: v1.0 (base: MiniMax H3, created 2026-08-16)
Files: MiniMax H3 - Original Audio Lip-Sync  Workflow.json (0.0 MB)
