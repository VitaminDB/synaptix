# Source: https://civitai.com/models/2837571

API: https://civitai.com/api/v1/models/2837571

Fetched: 2026-11-28 (Civitai public API, description converted to text)

---

## Minimax H3 Turbo LoRas
Type: LORA  Creator: tsolful  Tags: base model, turbo, h3, minimax h3
Stats: {"downloadCount": 62120, "thumbsUpCount": 1419, "thumbsDownCount": 0, "commentCount": 114, "tippedAmountCount": 6460}

### Description

pika_h3_flex_fl2va/ref2va_v1_bf16 uploaded
Current best settings: 6-8 steps WIP
minimax_h3_dmad_full_dareties_v4_step600_comfy experimentally merged myself with the current best LoRAs uploaded.
Current best settings: 6-8 steps, er_sde/lcm, simple, Strength 1.0 Video/Audio Shift 12/3
dmad_minimax_h3_4step_full_critic_comfyui uploaded (On par with larryvrhv4 video but better audio quality imo)
Current best settings: 6-8 steps, er_sde/euler, simple, Strength 1.0 Video/Audio Shift 12/2
taomate_fl2va_3step_ema
Current best settings: 4-6 steps, er_sde/euler, simple, Strength 0.75
minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16: 
Current best settings: 8-12 steps, er_sde/res_multistep, simple, Strength 1.0
lightx2v_4step_768p_v1.1+ v1.2: 
Current best settings: 5-8 steps, er_sde/res_multistep, simple. Strength 1.0
larryvrh_v4_600_ema: 
Current best settings: 6-8 steps, er_sde/res_multistep, simple. Strength 1.0

All examples are generated with 1.0 strength

I did not train, distill, or create the original Turbo LoRA weights. This model page provides only modified compatibility versions intended to let the remaining compatible LoRA adapters load with ComfyUI's built-in LoRA loader when using the pruned/curve-form model. 
Full credit for the original h3_turbo_4step_pruned belongs to larryvrh. And drbaph for the pruned comfy-compatible version
Full credit for the original lightx2v_4step_turbo_v0.1 belongs to the lightx2v team and kijai for the comfy-compatible version
Full credit for the original taomate_fl2va_3step_ema belongs to the TaoLiveAIGC team. And drbaph for the comfy-compatible version
Full credit for the original dmad_minimax_h3_4step_full_critic belongs to Bytedance & Texas A&M University students.
Full credit for the original pika_h3_flex_turbo_8step_hd_fl2va belongs to the Pika-Labs team.

I'll be uploading different Minimax H3 Turbo LoRAs here and crediting the original creators

### Version: larryvrh_4step_pruned (base: MiniMax H3, created 2026-08-06)
Files: minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_4step_ema_ckpt500_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_4step_ckpt500_pruned_comfyui.safetensors (591.6 MB); minimax_h3_turbo_4step_comfy_pruned.safetensors (591.5 MB); MiniMax_H3.json (0.0 MB)

### Version: pika-H3-flex-Turbo-8steps (base: MiniMax H3, created 2026-11-06)
Files: pika_h3_flex_fl2va_v1_bf16.safetensors (1865.6 MB); pika_h3_flex_ref2va_v1_bf16.safetensors (1865.6 MB)

### Version: dmad_larryvrhv4_merge (base: MiniMax H3, created 2026-10-08)
Files: minimax_h3_dmad_full_dareties_v4_step600_comfy.safetensors (3732.0 MB); minimax_h3_dmad_full_dareties_v4_step600_comfy_fro.safetensors (912.4 MB)

### Version: DMAD_4step_full (base: MiniMax H3, created 2026-10-04)
Files: minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors (349.7 MB); dmad_minimax_h3_4step_full_critic_comfyui.safetensors (1865.6 MB)

dmad_minimax_h3_4step_full_critic_comfyui.safetensors

### Version: taomate_fl2va_3step_ema (base: MiniMax H3, created 2026-09-13)
Files: minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors (2366.1 MB); minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors (173.3 MB); minimax_h3_taomate_3step_rank128_comfyui_bf16.safetensors (1183.1 MB)

### Version: lightx2v_4step_768p_v1.2 (base: MiniMax H3, created 2026-09-05)
Files: minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors (1865.6 MB)

### Version: lightx2v_ref2v_8step_v1.0 (base: MiniMax H3, created 2026-09-03)
Files: minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors (1865.6 MB)

### Version: lightx2v_4step_768p_v1.0 (base: MiniMax H3, created 2026-08-11)
Files: minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors (1865.6 MB)

### Version: lightx2v_8step_v1.0 (base: MiniMax H3, created 2026-08-11)
Files: minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors (1865.6 MB)

### Version: lightx2v_4step_v0.1 (base: MiniMax H3, created 2026-08-07)
Files: MiniMax_H3_Lightx2v.json (0.0 MB); minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors (1865.6 MB); minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors (300.3 MB)

### Version: lightx2v_4step_768p_v1.1 (base: MiniMax H3, created 2026-08-28)
Files: minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors (1865.6 MB)
