# DeepBeepMeep/MiniMax-H3

Источник: https://huggingface.co/DeepBeepMeep/MiniMax-H3 · скачано 2026-11-28
Загрузок: 1786632 · лайков: 96 · базовая:  · теги: gguf · обновлено: 2026-11-22T01:12:54.000Z

---

---

tags:
- diffusion-single-file
---

You will find here all the MiniMax H3 image models used with WanGP (https://github.com/deepbeepmeep/Wan2GP) :


WanGP by DeepBeepMeep : The best Open Source Video Generative Models Accessible to the GPU Poor

WanGP supports the Wan (and derived models), Hunyuan Video, Flux 1 & 2, Qwen, Z-Image and LTX-2, LTX Video models with:

Low VRAM requirements (as low as 6 GB of VRAM is sufficient for certain models)
Support for old GPUs (RTX 10XX, 20xx, ...)
Very Fast on the latest GPUs
Easy to use Full Web based interface
Auto download of the required model adapted to your specific architecture
Tools integrated to facilitate Video Generation : Mask Editor, Prompt Enhancer, Temporal and Spatial Generation
Loras Support to customize each model
Queuing system : make your shopping list of videos to generate and come back later
Discord Server to get Help from Other Users and show your Best Videos: https://discord.gg/g7efUW9jGV

Follow DeepBeepMeep on Twitter/X to get the Latest News: https://x.com/deepbeepmeep

## H3 additions — 2026-10-15

- **TaoMate H3 FL2VA 3-Step:** the dated FL2VA accelerator profile in WanGP
  uses the new rank-128 adapter. [Weights, provenance and usage](loras/TaoMate-H3-FL2VA-README.md).
- **LightVAE Lite:** the 12-layer H3 decoder, paired with the original encoder,
  is available in BF16 and INT8 ConvRot for FL2VA and Ref2VA.
  [Checkpoints, validation and licenses](minimax_h3/LightVAE-README.md).
