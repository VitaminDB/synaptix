# ModelsLab/MiniMax-H3-Turbo-RL

Источник: https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL · скачано 2026-11-28
Загрузок: 710 · лайков: 10 · базовая: ["MiniMaxAI/MiniMax-H3","larryvrh/MiniMax-H3-Turbo-Lora"] · теги: lora · обновлено: 2026-11-23T12:53:46.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
base_model:
  - MiniMaxAI/MiniMax-H3
  - larryvrh/MiniMax-H3-Turbo-Lora
library_name: diffusers
pipeline_tag: text-to-video
tags:
  - lora
  - video-generation
  - text-to-audio-video
  - minimax-h3
  - ref2va
  - flow-dpo
---

# MiniMax-H3 Turbo V4 + RL (rl_s3)

One LoRA file for **MiniMax-H3 Ref2VA** (`MiniMaxAI/MiniMax-H3`, `transformer_ref`) that combines:

1. **Turbo LoRA V4** by larryvrh (`larryvrh/MiniMax-H3-Turbo-Lora`, `minimax_h3_turbo_v4_step600_ema.safetensors`, Apache-2.0): the 8-step distillation adapter.
2. **rl_s3**: a rank-64 LoRA trained by ModelsLab on top of Turbo V4 with closed-loop, pure-ODE online Flow-DPO (best-of-N rollouts scored by in-RAM video/audio critics, paired against frozen V4 on the same seed, high-frequency KL lock to V4).

The two are stored as exact branches side by side, not merged: for every attention/MLP module `ΔW = [B_v4 | B_rl] · [A_v4 ; A_rl]` (rank 128), and V4's rank-16 `adaln_proj.linear` branches are kept as they are. Merging into BF16 base weights would round away most of these small deltas, so load the file as LoRA branches.

| File | Size | Layout |
| --- | --- | --- |
| `minimax_h3_turbo_v4_rl_s3.safetensors` | 1.40 GB (BF16) | ComfyUI / ai-toolkit keys (`blocks.N.attn.qkv_proj.lora_A.weight`, ...), same layout as Turbo V4. 208 attention/MLP modules at rank 128 (200 DiT-block modules = V4 ⊕ RL; 8 token-refiner modules = V4 zero-padded) + 51 AdaLN modules at rank 16 (V4 only). 518 BF16 tensors. |

## How to run it

The recommended sampler is **E4a**: Turbo V4's 9-point shifted grid (`u = 1, .875, …, .125, 0`, shift 12 video / 3 audio) with one extra node at `u = 0.04` (`σ_video = 0.333`) and a 2nd-order Heun corrector on `[σ 0.632 → 0.333]`: 10 sigma points, 10 transformer forwards. Reference implementation for the NVIDIA Sol-H3 engine: `heun_tail.py` in the ModelsLab H3 server repo; the plain 8-forward Euler schedule also works.

Load it with any loader that accepts Turbo V4's file (same keys, larger rank). Sol-H3 / PEFT loaders read the rank from the tensors.

## Evaluation (16 public clips, seed 11, ratios × the real clip)

| Arm | Forwards | Detail | Flow | Motion | Made-up speech |
| --- | :---: | :---: | :---: | :---: | :---: |
| Turbo V4, Euler | 8 | 1.43 | 0.87 | 0.94 | 1/7 |
| Turbo V4, E4a | 10 | 1.59 | 0.92 | 1.02 | 1/7 |
| **Turbo V4 + rl_s3, E4a** | 10 | **1.59** | **0.92** | **1.01** | 1/7 |

Honest attribution: on this public set almost all of the gain over Turbo V4 comes from the E4a sampler; rl_s3 matches V4 + E4a within noise on these metrics (closer to the real clip than V4 + E4a on Lab b for 11/16 clips and on saturation for 8/16). Measured with a BF16 reference pipeline, not with the MXFP8 production engine.

## Samples: before / after

Same prompt, references and seed (11) in each row. **Before:** Turbo V4, 8-step Euler (the previous production setup). **After:** this file (Turbo V4 + rl_s3) with the E4a 10-NFE sampler. Rendered at 1344×768 with the BF16 reference pipeline (not the MXFP8 production engine). Prompts and reference images come from the public clips credited below; the videos and sound are generated.

| | Before: Turbo V4, 8 steps | After: Turbo V4 + RL, E4a |
| --- | --- | --- |
| Flower market (multi-subject, 5 s) | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/flower_market_before_v4.mp4"></video> | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/flower_market_after_v4_rl_e4a.mp4"></video> |
| Danza de los Voladores (portrait action, 10 s) | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/voladores_before_v4.mp4"></video> | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/voladores_after_v4_rl_e4a.mp4"></video> |
| View from Fisherman's Bastion (ambience, 10 s) | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/budapest_before_v4.mp4"></video> | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/budapest_after_v4_rl_e4a.mp4"></video> |
| Office dialogue (speech, 5 s) | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/office_dialogue_before_v4.mp4"></video> | <video controls width="100%" src="https://huggingface.co/ModelsLab/MiniMax-H3-Turbo-RL/resolve/main/samples/office_dialogue_after_v4_rl_e4a.mp4"></video> |

Source clips (used only to write the prompt and pick reference frames):

- Flower market: Netflix Open Content, source file `96x2160 60fps 10bit 420`, CC BY 4.0
- Danza de los Voladores: Wikimedia Commons, *Danza de los Voladores (Dance of the Flyers)* by Emma Matthies, CC BY 4.0
- View from Fisherman's Bastion: Wikimedia Commons, *View from Fisherman's Bastion (2)* by Elekes Andor, CC BY 4.0
- Office dialogue: Netflix Open Content, source file `3840x2160 59.94fps SDR`, CC BY 4.0

## Training data and licences

The RL stages (rl_s0 → rl_s3) used short clips (only their cached latents and the prompts written for them) from:

| Source | Licence |
| --- | --- |
| DVIDS (dvidshub.net) | Public domain (US federal government work); DVIDS API terms |
| Wikimedia Commons | Per-file free licences |
| Netflix Open Content | CC BY 4.0 |
| UltraVideo (`APRIL-AIGC/UltraVideo`) | UltraVideo License (CC BY 4.0 with restrictions, non-commercial research) |
| Replay (`facebookresearch/replay_dataset`) | CC BY-NC 4.0 |
| Xiph test media: SVT | Only for developing, testing and presenting technology standards |
| Xiph test media: Netflix | CC BY-NC-ND 4.0 |

No media from these datasets is redistributed here. No customer or user data was used for training.

## Licence and use

- The weights are a derivative of MiniMax-H3 and are released under the **MiniMax-H3 Community License Agreement**, including its territory and use restrictions. Read it before use.
- The Turbo V4 component keeps its Apache-2.0 licence and credit: larryvrh, `larryvrh/MiniMax-H3-Turbo-Lora` (revision `43a74557`).
- Several training sources are licensed for non-commercial research only (see the table). Check that your use is compatible with them.
