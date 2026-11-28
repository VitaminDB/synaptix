# Alex995647/loras-minimax-h3

Источник: https://huggingface.co/Alex995647/loras-minimax-h3 · скачано 2026-11-28
Загрузок: 0 · лайков: 28 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-11-05T07:17:45.000Z

---

---
library_name: diffusers
base_model: MiniMaxAI/MiniMax-H3
tags:
  - lora
  - video
  - mirror
---

# MiniMax H3 — LoRA mirror

A mirror of **110 LoRAs** for MiniMax H3 (63.5 GB), collected from CivitAI and the Hugging Face Hub.

Each folder holds the weights plus `info.txt` (trigger words, settings, pros/cons, chaining notes), `metadata.json`, and `example_images/`.
See `CATALOG.md` for the index and `REFERENCE.txt` for how this model's modes and checkpoints relate.

## Credit

**Every LoRA here was trained by someone else.** This repo is an archival mirror, not original work. Each `info.txt` names its author and links to the original listing. Authors represented (64):

`2_as_1`, `AdaptiveVision`, `AiMetatron`, `AirbagGuy`, `B4100`, `Beidouqixing`, `blargg`, `chfm`, `Cosmic_Crafter`, `cupra`, `Cybronya`, `DialUpDialDown`, `drbaph`, `drozbay`, `fal`, `fbjr`, `freek22`, `Goon_69420`, `Gyvin`, `Halpak`, `jandordoe`, `Jojocodex`, `JonXL`, `juliaaruscio`, `kabachuha`, `KennethFal`, `kermitfrog1202`, `Kijai`, `lightx2v`, `lividlygulpunwed`, `Lokitsar`, `LoRA_Angel`, `lovis93`, `Mamad8`, `masafy`, `MATLOWAI`, `mello_ai`, `mhnakif`, `MikoGoblin`, `MM744`, `NoiseWriggler`, `OnceWasPerfect`, `ostris`, `phamnam`, `Plaguekind`, `pmczip`, `Polyhedron_AI`, `PrivateHindsight`, `qqqq4413`, `rehan-fal`, `ReltivlyObjectv`, `Rkss`, `russgilbert`, `sarcastictofu`, `shamanic`, `shimeajiRTX`, `t8star`, `tazmannner379`, `thefoodmage`, `tigertakato`, `TuTu_1018`, `valentinkognito365`, `xiaolibai_2019260`, `zzzDDs`

If you are one of them and would rather your work were not mirrored here, open a discussion on this repo and it will be removed.

## Licensing

Licence terms travel with each original listing, not with this mirror, and they vary — some permit redistribution, some restrict commercial use. Check the source link in a LoRA's `info.txt` before using it, and read the MiniMax H3 base-model licence for the model itself.

## Contents

| LoRA | Category | Author | Source | Size |
|------|----------|--------|--------|------|
| [Ref2V Turbo 8-step v1.0 768p - rank 64](ref2v-turbo-8-step-v1-0-768p-rank-64/info.txt) | Speed / step-reduction | `drbaph` | [huggingface](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI) | 933 MB |
| [FL2V Turbo 4-step v1.2 768p - rank 64](fl2v-turbo-4-step-v1-2-768p-rank-64/info.txt) | Speed / step-reduction | `drbaph` | [huggingface](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI) | 929 MB |
| [H3 Realism People](h3-realism-people/info.txt) | Photorealism | `fal` | [huggingface](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA) | 125 MB |
| [Furry Enhancer Video](furry-enhancer-video/info.txt) | Detail / anatomy fixer | `freek22` | [civitai](https://civitai.com/models/1782485) | 887 MB |
| [\[NEW Released\] Minimax H3 lightx2v turbo ComfyUI accelerator 4-8 steps LoRA](new-released-minimax-h3-lightx2v-turbo-comfyui-accelerator/info.txt) | Speed / step-reduction | `AiMetatron` | [civitai](https://civitai.com/models/1063735) | 348 MB |
| [Spatial & Physics LoRA (clean, 3000 steps)](spatial-physics-lora-clean-3000-steps/info.txt) | Motion quality | `Jojocodex` | [huggingface](https://huggingface.co/Jojocodex/minimax-h3-spatial-physics-lora) | 148 MB |
| [Spatial & Physics LoRA v2 (1000 steps)](spatial-physics-lora-v2-1000-steps/info.txt) | Motion quality | `Jojocodex` | [huggingface](https://huggingface.co/Jojocodex/minimax-h3-spatial-physics-lora) | 148 MB |
| [Better Motion LTX/Minimax H3](better-motion-ltx-minimax-h3/info.txt) | Motion quality | `AdaptiveVision` | [civitai](https://civitai.com/models/2734359) | 296 MB |
| [NSGIRL UltraRealistic LoRA](nsgirl-ultrarealistic-lora/info.txt) | Photorealism | `MM744` | [civitai](https://civitai.com/models/2326742) | 148 MB |
| [Wushu Action LoRA v5 (3000 steps)](wushu-action-lora-v5-3000-steps/info.txt) | Speed / step-reduction | `Jojocodex` | [huggingface](https://huggingface.co/Jojocodex/minimax-h3-wushu-action-lora) | 148 MB |
| [Studio 1939 Light (r16)](studio-1939-light-r16/info.txt) | Visual style | `lovis93` | [huggingface](https://huggingface.co/lovis93/studio-1939-old-animation-lora-minimax-h3) | 63 MB |
| [Studio 1939 Strong (r64)](studio-1939-strong-r64/info.txt) | Visual style | `lovis93` | [huggingface](https://huggingface.co/lovis93/studio-1939-old-animation-lora-minimax-h3) | 250 MB |
| [MiniMax-H3 4-Step FlashGen LoRA](minimax-h3-4-step-flashgen-lora/info.txt) | Speed / step-reduction | `Beidouqixing` | [huggingface](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen) | 1200 MB |
| [MiniMax-H3 Motion Adapter (Pilot r16)](minimax-h3-motion-adapter-pilot-r16/info.txt) | Uncategorised | `MATLOWAI` | [huggingface](https://huggingface.co/MATLOWAI/MiniMax-H3-Motion-Adapter) | 60 MB |
| [💁‍♀️ Height Growth](height-growth/info.txt) | Physical effect / transformation | `cupra` | [civitai](https://civitai.com/models/2773174) | 314 MB |
| [FastH3 Preview v0.2 4-Step LoRA (pruned, rank 128)](fasth3-preview-v0-2-4-step-lora-pruned-rank-128/info.txt) | Speed / step-reduction | `drozbay` | [huggingface](https://huggingface.co/drozbay/MiniMax-H3-FastH3-Preview-LoRA) | 1272 MB |
| [Minimax H3 Authentic cinematic texture](minimax-h3-authentic-cinematic-texture/info.txt) | Photorealism | `TuTu_1018` | [civitai](https://civitai.com/models/2890588) | 296 MB |
| [Airbag Deploy](airbag-deploy/info.txt) | Physical effect / transformation | `AirbagGuy` | [civitai](https://civitai.com/models/1901787) | 296 MB |
| [One-piece swimsuit, swimsuitOPO](one-piece-swimsuit-swimsuitopo/info.txt) | Clothing / wardrobe | `PrivateHindsight` | [civitai](https://civitai.com/models/1487957) | 148 MB |
| [Disney Style - Renaissance Era](disney-style-renaissance-era/info.txt) | Visual style | `kermitfrog1202` | [civitai](https://civitai.com/models/2784600) | 148 MB |
| [Minimax H3 Photorealistic Image Generator (LORA + Workflow)](minimax-h3-photorealistic-image-generator-lora-workflow/info.txt) | Photorealism | `JonXL` | [civitai](https://civitai.com/models/2884206) | 569 MB |
| [Wushu Action v7 (E3)](wushu-action-v7-e3/info.txt) | Visual style | `Jojocodex` | [huggingface](https://huggingface.co/Jojocodex/wushu-action-v7-minimax-h3-fl2va-ref2va-lora) | 569 MB |
| [Claymation Transformation - MiniMax H3 R2VA / Wan2.2-2.1 14b T2V](claymation-transformation-minimax-h3-r2va-wan2-2-2-1-14b-t2v/info.txt) | Physical effect / transformation | `kabachuha` | [civitai](https://civitai.com/models/1659949) | 284 MB |
| [Drone Shot](drone-shot/info.txt) | Camera movement | `Lokitsar` | [civitai](https://civitai.com/models/2857065) | 148 MB |
| [Minimax H3: t2va (T2V) – Polyhedron: Perfect Eyes // Perfect Skin // Perfect Hands](minimax-h3-t2va-t2v-polyhedron-perfect-eyes-perfect-skin/info.txt) | Detail / anatomy fixer | `Polyhedron_AI` | [civitai](https://civitai.com/models/2899220) | 296 MB |
| [Fyre (fictional character)](fyre-fictional-character/info.txt) | Character / subject | `jandordoe` | [civitai](https://civitai.com/models/2386526) | 222 MB |
| [Minimax H3 Cinematic Look (No Tensor Errors)](minimax-h3-cinematic-look-no-tensor-errors/info.txt) | Visual style | `TuTu_1018` | [civitai](https://civitai.com/models/2908686) | 148 MB |
| [MiniMAX H3 Turbo Lora adapted for merging with pruned FL2VA model](minimax-h3-turbo-lora-adapted-for-merging-with-pruned-fl2va/info.txt) | Speed / step-reduction | `xiaolibai_2019260` | [civitai](https://civitai.com/models/2838852) | 1350 MB |
| [vh5tape — Worn VHS Look](vh5tape-worn-vhs-look/info.txt) | Visual style | `KennethFal` | [huggingface](https://huggingface.co/KennethFal/vh5tape-vhs-lora-minimax-h3) | 125 MB |
| [Lain Iwakura (Anime) - MINIMAX H3 - KREA2 - KLEIN - QWEN - FLUX - Serial Experiments Lain](lain-iwakura-anime-minimax-h3-krea2-klein-qwen-flux-serial/info.txt) | Visual style | `valentinkognito365` | [civitai](https://civitai.com/models/653421) | 148 MB |
| [AI Girl: Fictional Women Series30 / 架空の女性シリーズ30 MiniMax H3](ai-girl-fictional-women-series30-30-minimax-h3/info.txt) | Character / subject | `shimeajiRTX` | [civitai](https://civitai.com/models/2845077) | 284 MB |
| [H3 FL2VA PDD 8-Step Sidecar (pruned base)](h3-fl2va-pdd-8-step-sidecar-pruned-base/info.txt) | Speed / step-reduction | `fbjr` | [huggingface](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar) | 1070 MB |
| [H3 FL2VA PDD 8-Step Sidecar (adaln2688, any base)](h3-fl2va-pdd-8-step-sidecar-adaln2688-any-base/info.txt) | Speed / step-reduction | `fbjr` | [huggingface](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar) | 1595 MB |
| [H3 Ref2VA PDD 8-Step Sidecar (pruned base)](h3-ref2va-pdd-8-step-sidecar-pruned-base/info.txt) | Speed / step-reduction | `fbjr` | [huggingface](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar) | 1070 MB |
| [H3 Ref2VA PDD 8-Step Sidecar (adaln2688, any base)](h3-ref2va-pdd-8-step-sidecar-adaln2688-any-base/info.txt) | Speed / step-reduction | `fbjr` | [huggingface](https://huggingface.co/fbjr/MiniMax-H3-Acc-LoRAs-sidecar) | 1595 MB |
| [Elke](elke/info.txt) | Character / subject | `jandordoe` | [civitai](https://civitai.com/models/2440331) | 222 MB |
| [H3 Turbo 4-8 Step (Native Port)](h3-turbo-4-8-step-native-port/info.txt) | Speed / step-reduction | `Rkss` | [huggingface](https://huggingface.co/Rkss/Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095_native) | 911 MB |
| [H3 Turbo 4-8 Step (Adaln-Pruned)](h3-turbo-4-8-step-adaln-pruned/info.txt) | Speed / step-reduction | `Rkss` | [huggingface](https://huggingface.co/Rkss/Minimax_h3_fl2v_lightx2v_turbo_4to8step_v0.1-v1.0_768p_v4_step600_dareties_fro095_native) | 819 MB |
| [Skeletor - Masters of the Universe](skeletor-masters-of-the-universe/info.txt) | Character / subject | `DialUpDialDown` | [civitai](https://civitai.com/models/2835069) | 74 MB |
| [Dan Mora Chavez Style](dan-mora-chavez-style/info.txt) | Visual style | `kermitfrog1202` | [civitai](https://civitai.com/models/2793943) | 148 MB |
| [Masafy Red Panda (step 800, final)](masafy-red-panda-step-800-final/info.txt) | Character / subject | `masafy` | [huggingface](https://huggingface.co/masafy/minimax-h3-masafy-lora) | 148 MB |
| [Miko Goblin](miko-goblin/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2450477) | 148 MB |
| [H3 PK Parasyte Turbo](h3-pk-parasyte-turbo/info.txt) | Speed / step-reduction | `Plaguekind` | [civitai](https://civitai.com/models/2901588) | 2018 MB |
| [nagisa Mini Max H3](nagisa-mini-max-h3/info.txt) | Character / subject | `tigertakato` | [civitai](https://civitai.com/models/2865291) | 148 MB |
| [Retro 90's Anime / Golden Boy Style Lora - H3](retro-90-s-anime-golden-boy-style-lora-h3/info.txt) | Visual style | `tazmannner379` | [civitai](https://civitai.com/models/2909069) | 284 MB |
| [Stylized PBR Animation \[MiniMax H3 Ref2VA\]](stylized-pbr-animation-minimax-h3-ref2va/info.txt) | Visual style | `ReltivlyObjectv` | [civitai](https://civitai.com/models/2868582) | 148 MB |
| [Jacked (Huge Muscles)](jacked-huge-muscles/info.txt) | Uncategorised | `ostris` | [huggingface](https://huggingface.co/ostris/minimax_h3_ref2va_jacked_lora) | 148 MB |
| [chr_nea, North Eastern Asian female model](chr-nea-north-eastern-asian-female-model/info.txt) | Character / subject | `PrivateHindsight` | [civitai](https://civitai.com/models/2789059) | 148 MB |
| [AI Girl: Fictional Women Series31 / 架空の女性シリーズ31 MiniMax H3](ai-girl-fictional-women-series31-31-minimax-h3/info.txt) | Character / subject | `shimeajiRTX` | [civitai](https://civitai.com/models/2853473) | 284 MB |
| [ComfyUI FastVideo FastH3 4-step LoRa (unofficial)](comfyui-fastvideo-fasth3-4-step-lora-unofficial/info.txt) | Speed / step-reduction | `NoiseWriggler` | [civitai](https://civitai.com/models/2898443) | 997 MB |
| [Lanolin Aries](lanolin-aries/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2483615) | 148 MB |
| [TFM Cutesy Anime Pixel for Minimax H3](tfm-cutesy-anime-pixel-for-minimax-h3/info.txt) | Visual style | `thefoodmage` | [civitai](https://civitai.com/models/2907373) | 148 MB |
| [Fawn Hartley](fawn-hartley/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2456887) | 148 MB |
| [FL2V Turbo 8-Step v1.0 768p (rank 20)](fl2v-turbo-8-step-v1-0-768p-rank-20/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 291 MB |
| [FL2V Turbo 8-Step v1.0 (rank 21)](fl2v-turbo-8-step-v1-0-rank-21/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 312 MB |
| [Ref2V Turbo 4-Step v0.1 (rank 21)](ref2v-turbo-4-step-v0-1-rank-21/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 312 MB |
| [FL2V Turbo 4-Step v0.1 768p SLA (rank 64)](fl2v-turbo-4-step-v0-1-768p-sla-rank-64/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 891 MB |
| [Turbo v4 Step-600 (pruned conversion)](turbo-v4-step-600-pruned-conversion/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 592 MB |
| [Turbo 4-Step Preview EMA (pruned conversion)](turbo-4-step-preview-ema-pruned-conversion/info.txt) | Speed / step-reduction | `Halpak` | [huggingface](https://huggingface.co/Halpak/MiniMax-H3-Turbo-Lora-ComfyUI) | 592 MB |
| [女秘书](untitled/info.txt) | Character / subject | `LoRA_Angel` | [civitai](https://civitai.com/models/2851467) | 296 MB |
| [Roxi](roxi/info.txt) | Character / subject | `OnceWasPerfect` | [civitai](https://civitai.com/models/2826669) | 592 MB |
| [Mello's HTTYD (How to Train Your Dragon) For Minimax H3](mello-s-httyd-how-to-train-your-dragon-for-minimax-h3/info.txt) | Character / subject | `mello_ai` | [civitai](https://civitai.com/models/2850124) | 592 MB |
| [Anna Kyoyama (Shaman King \[2001\]) Anime Video](anna-kyoyama-shaman-king-2001-anime-video/info.txt) | Visual style | `blargg` | [civitai](https://civitai.com/models/1185051) | 148 MB |
| [Mika Iwakura - MINIMAX H3 - QWEN - Serial Experiments Lain](mika-iwakura-minimax-h3-qwen-serial-experiments-lain/info.txt) | Character / subject | `valentinkognito365` | [civitai](https://civitai.com/models/2110324) | 148 MB |
| [Doctor Penny](doctor-penny/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2857823) | 148 MB |
| [Ted](ted/info.txt) | Character / subject | `kermitfrog1202` | [civitai](https://civitai.com/models/2882682) | 148 MB |
| [Painterly H3 by cosmic](painterly-h3-by-cosmic/info.txt) | Visual style | `Cosmic_Crafter` | [civitai](https://civitai.com/models/2903606) | 296 MB |
| [Elizabit Agnes](elizabit-agnes/info.txt) | Visual style | `MikoGoblin` | [civitai](https://civitai.com/models/2812523) | 148 MB |
| [TFM Hanafuda Style for Minimax H3](tfm-hanafuda-style-for-minimax-h3/info.txt) | Visual style | `thefoodmage` | [civitai](https://civitai.com/models/2902682) | 148 MB |
| [Handfeet - prehensile feet - Minimax](handfeet-prehensile-feet-minimax/info.txt) | Uncategorised | `2_as_1` | [civitai](https://civitai.com/models/2891786) | 296 MB |
| [Only Saree Minimax H3 by Sarcastic TOFU](only-saree-minimax-h3-by-sarcastic-tofu/info.txt) | Character / subject | `sarcastictofu` | [civitai](https://civitai.com/models/2878350) | 296 MB |
| [LightX2V H3 Ref2V Turbo 4-Step v0.1](lightx2v-h3-ref2v-turbo-4-step-v0-1/info.txt) | Speed / step-reduction | `Gyvin` | [huggingface](https://huggingface.co/Gyvin/MiniMax-H3) | 1866 MB |
| [LightX2V H3 FL2V Turbo 8-Step v1.0](lightx2v-h3-fl2v-turbo-8-step-v1-0/info.txt) | Speed / step-reduction | `Gyvin` | [huggingface](https://huggingface.co/Gyvin/MiniMax-H3) | 1866 MB |
| [LightX2V H3 FL2V Turbo 4-Step v1.0 768p](lightx2v-h3-fl2v-turbo-4-step-v1-0-768p/info.txt) | Speed / step-reduction | `Gyvin` | [huggingface](https://huggingface.co/Gyvin/MiniMax-H3) | 1866 MB |
| [h3sully - Sully from Monsters Inc for H3 Minimax](h3sully-sully-from-monsters-inc-for-h3-minimax/info.txt) | Visual style | `Goon_69420` | [civitai](https://civitai.com/models/2886539) | 148 MB |
| [Holi Festival of Colors Minimax H3 by Sarcastic TOFU](holi-festival-of-colors-minimax-h3-by-sarcastic-tofu/info.txt) | Visual style | `sarcastictofu` | [civitai](https://civitai.com/models/2883882) | 296 MB |
| [Gordon Pinewood](gordon-pinewood/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2906761) | 148 MB |
| [Alex Kingston Likeness](alex-kingston-likeness/info.txt) | Photorealism | `pmczip` | [huggingface](https://huggingface.co/pmczip/MiniMaxH3_LoRAs) | 148 MB |
| [MiniMax-H3 Turbo v4-600 (EMA)](minimax-h3-turbo-v4-600-ema/info.txt) | Speed / step-reduction | `chfm` | [huggingface](https://huggingface.co/chfm/MiniMax-H3-Turbo-Lora) | 744 MB |
| [MiniMax-H3 Turbo v1 initial (EMA)](minimax-h3-turbo-v1-initial-ema/info.txt) | Speed / step-reduction | `chfm` | [huggingface](https://huggingface.co/chfm/MiniMax-H3-Turbo-Lora) | 744 MB |
| [Lucy Nublar](lucy-nublar/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2905329) | 148 MB |
| [Perry Chellis](perry-chellis/info.txt) | Character / subject | `MikoGoblin` | [civitai](https://civitai.com/models/2904799) | 148 MB |
| [Test Character LoRA](test-character-lora/info.txt) | Character / subject | `lividlygulpunwed` | [civitai](https://civitai.com/models/2908472) | 296 MB |
| [FL2V Turbo 8-Step Merge 0821](fl2v-turbo-8-step-merge-0821/info.txt) | Speed / step-reduction | `zzzDDs` | [huggingface](https://huggingface.co/zzzDDs/MiniMax-H3_experimental) | 1866 MB |
| [\[Invisible person / Living clothes\] MiniMax H3 T2V/I2V/FL2V](invisible-person-living-clothes-minimax-h3-t2v-i2v-fl2v/info.txt) | Physical effect / transformation | `qqqq4413` | [civitai](https://civitai.com/models/2914067) | 296 MB |
| [Facial Realism Close-Up](facial-realism-close-up/info.txt) | Photorealism | `B4100` | [huggingface](https://huggingface.co/B4100/MiniMax-H3-Facial-Realism-CloseUp) | 71 MB |
| [Minimax Char](minimax-char/info.txt) | Character / subject | `juliaaruscio` | [huggingface](https://huggingface.co/juliaaruscio/minimax-char-lora) | 125 MB |
| [LightX2V Turbo FL2V 4-Step v0.1 (ComfyUI, resized rank 21)](lightx2v-turbo-fl2v-4-step-v0-1-comfyui-resized-rank-21/info.txt) | Speed / step-reduction | `Kijai` | [huggingface](https://huggingface.co/Kijai/MiniMax-H3_comfy) | 300 MB |
| [LightX2V Turbo FL2V 4-Step v0.1 (ComfyUI, full rank)](lightx2v-turbo-fl2v-4-step-v0-1-comfyui-full-rank/info.txt) | Speed / step-reduction | `Kijai` | [huggingface](https://huggingface.co/Kijai/MiniMax-H3_comfy) | 1866 MB |
| [MiniMax H3 Ref2VA Acc 8-Step (pruned)](minimax-h3-ref2va-acc-8-step-pruned/info.txt) | Speed / step-reduction | `Kijai` | [huggingface](https://huggingface.co/Kijai/MiniMax-H3-experimental) | 1646 MB |
| [LightX2V Turbo REF2V 4-Step v0.1 (resized rank 20)](lightx2v-turbo-ref2v-4-step-v0-1-resized-rank-20/info.txt) | Speed / step-reduction | `Kijai` | [huggingface](https://huggingface.co/Kijai/MiniMax-H3_comfy) | 293 MB |
| [LightX2V Turbo FL2V 4-Step v1.0 768p (resized rank 31)](lightx2v-turbo-fl2v-4-step-v1-0-768p-resized-rank-31/info.txt) | Speed / step-reduction | `Kijai` | [huggingface](https://huggingface.co/Kijai/MiniMax-H3_comfy) | 420 MB |
| [MiniMax-H3 Turbo-SLA 4-Step 768p (LightX2V)](minimax-h3-turbo-sla-4-step-768p-lightx2v/info.txt) | Speed / step-reduction | `lightx2v` | [huggingface](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) | 1320 MB |
| [MiniMax-H3 Turbo-SLA 4-Step 768p (ComfyUI)](minimax-h3-turbo-sla-4-step-768p-comfyui/info.txt) | Speed / step-reduction | `lightx2v` | [huggingface](https://huggingface.co/lightx2v/Minimax-h3-Turbo-SLA) | 1866 MB |
| [VR180 Stereoscopic Side-by-Side](vr180-stereoscopic-side-by-side/info.txt) | Output format / projection | `rehan-fal` | [huggingface](https://huggingface.co/rehan-fal/minimax-h3-vr180-sbs-lora) | 125 MB |
| [Equirectangular 360 Panorama](equirectangular-360-panorama/info.txt) | Output format / projection | `shamanic` | [huggingface](https://huggingface.co/shamanic/minimax-h3-equi360-lora) | 125 MB |
| [H3 Turbo 4-Step — 10Eros Pruned Curve Projection (exp v001)](h3-turbo-4-step-10eros-pruned-curve-projection-exp-v001/info.txt) | Speed / step-reduction | `t8star` | [huggingface](https://huggingface.co/t8star/minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_T8) | 758 MB |
| [H3 R2V PDD Turbo (step 900)](h3-r2v-pdd-turbo-step-900/info.txt) | Speed / step-reduction | `Mamad8` | [huggingface](https://huggingface.co/Mamad8/MiniMaxH3_R2V-PDD-Turbo-LoRA-Mamad8) | 569 MB |
| [LightX2V FL2V Turbo 4-Step v0.1 — 10Eros test4 Compat Build](lightx2v-fl2v-turbo-4-step-v0-1-10eros-test4-compat-build/info.txt) | Speed / step-reduction | `t8star` | [huggingface](https://huggingface.co/t8star/minimax_h3_turbo_4step_10ErosMax_test4_pruned_curveproj1025_T8) | 1866 MB |
| [H3-World Action Control](h3-world-action-control/info.txt) | Camera movement | `t8star` | [huggingface](https://huggingface.co/t8star/Minimax-H3-World-Comfy) | 125 MB |
| [MiniMax-H3 Turbo FL2V 8-Step v1.0 768p (ComfyUI BF16)](minimax-h3-turbo-fl2v-8-step-v1-0-768p-comfyui-bf16/info.txt) | Speed / step-reduction | `mhnakif` | [huggingface](https://huggingface.co/mhnakif/minimax_h3) | 1866 MB |
| [MiniMax-H3 Turbo FL2V 4-Step v0.1 (Diffusers)](minimax-h3-turbo-fl2v-4-step-v0-1-diffusers/info.txt) | Speed / step-reduction | `mhnakif` | [huggingface](https://huggingface.co/mhnakif/minimax_h3) | 1320 MB |
| [MiniMax H3 Turbo v4 (step 600 EMA)](minimax-h3-turbo-v4-step-600-ema/info.txt) | Speed / step-reduction | `phamnam` | [huggingface](https://huggingface.co/phamnam/saniora-h3-ref2va) | 592 MB |
| [MiniMax-H3 Turbo FL2V 8-Step v1.0 (Diffusers BF16)](minimax-h3-turbo-fl2v-8-step-v1-0-diffusers-bf16/info.txt) | Speed / step-reduction | `mhnakif` | [huggingface](https://huggingface.co/mhnakif/minimax_h3) | 1320 MB |
| [PDD Acc 8-Step (Ref2VA)](pdd-acc-8-step-ref2va/info.txt) | Speed / step-reduction | `Cybronya` | [huggingface](https://huggingface.co/Cybronya/minimax-h3-toolkit) | 1582 MB |
| [Turbo v4 Step-600 (ComfyUI convert)](turbo-v4-step-600-comfyui-convert/info.txt) | Speed / step-reduction | `Cybronya` | [huggingface](https://huggingface.co/Cybronya/minimax-h3-toolkit) | 744 MB |
| [PDD Acc 8-Step (FL2VA)](pdd-acc-8-step-fl2va/info.txt) | Speed / step-reduction | `Cybronya` | [huggingface](https://huggingface.co/Cybronya/minimax-h3-toolkit) | 1582 MB |
| [LightX2V Turbo 4-Step (alpha-8 ComfyUI convert)](lightx2v-turbo-4-step-alpha-8-comfyui-convert/info.txt) | Speed / step-reduction | `Cybronya` | [huggingface](https://huggingface.co/Cybronya/minimax-h3-toolkit) | 1866 MB |
| [Providence Rivera](providence-rivera/info.txt) | Uncategorised | `russgilbert` | [huggingface](https://huggingface.co/russgilbert/providence-rivera-h3-lora) | 591 MB |
| [Providence Hala](providence-hala/info.txt) | Physical effect / transformation | `russgilbert` | [huggingface](https://huggingface.co/russgilbert/providence-hala-h3-lora) | 591 MB |
