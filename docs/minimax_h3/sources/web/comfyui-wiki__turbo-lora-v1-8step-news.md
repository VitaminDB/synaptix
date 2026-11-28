# Source: https://comfyui-wiki.com/en/news/2026-08-11-minimax-h3-turbo-lightx2v-v1

Fetched: 2026-11-28 (curl + w3m text dump)

---

(Select language)

 1. Home
 2.
 3. News
 4.
 5. MiniMax H3 Turbo LoRA v1.0 Download: 8-Step and 768p FL2V

MiniMax H3 Turbo LoRA v1.0 Download: 8-Step and 768p FL2V

Last updated: August 11, 2026ComfyUI Wikinews

Download LightX2V MiniMax H3 Turbo LoRA v1.0: minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors and 4-step 768p weights, 8-step video generation in ComfyUI.

On August 11, the Lightx2v team and ModelTC released v1.0 of their MiniMax H3 Turbo LoRA, following the early v0.1 preview from August 7. The update adds two new FL2VA checkpoints in bf16, each
shipped with a ComfyUI-ready format, including a variant trained at higher resolution.

Example frame from the Lightx2v Turbo LoRA demo output

Example frame from the Lightx2v Turbo LoRA project's demo output

What's in v1.0

            Model                Tasks     Training resolution Distillation steps Recommended inference
FL2VA Turbo 4-step v0.1       FL2VA / T2VA 544p (mixed AR)     4                  4
FL2VA Turbo 8-step v1.0       FL2VA / T2VA 544p (mixed AR)     8                  8 or 4
FL2VA Turbo 4-step v1.0 768p  FL2VA / T2VA 768p (1344x768)     4                  4

The v1.0 release consists of two new checkpoints on Hugging Face:

  • 8-step v1.0 (minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors): distilled at 8 NFE with 12/3 video/audio shifts, and usable at 8 or 4 inference steps.
  • 4-step v1.0 768p (minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors): a 4-step distillation trained at 768p (1344x768) with 6/3 shifts, aimed at higher-resolution first/last-frame-to-video
    generation.

Each checkpoint has a ComfyUI-format twin (..._comfyui_bf16.safetensors) hosted in the same repository, so no separate conversion is needed.

ComfyUI usage

MiniMax H3 has native ComfyUI support, so no custom node is required for the LoRA:

 1. Download the ComfyUI-format file (for example minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors) into ComfyUI/models/loras/.
 2. Load the base MiniMax H3 checkpoint in a FL2VA workflow and apply the LoRA between the model loader and the sampler.
 3. For the 8-step checkpoint, run 8 sampling steps (4 also works); for the 768p variant, run 4 steps at 768p resolution.

Download MiniMax H3 workflow template

The official MiniMax H3 image-to-video workflow template is a good starting point: insert the Turbo LoRA between the model loader and the sampler and switch to the step settings above.

Availability

  • Weights: lightx2v/Minimax-h3-Turbo on Hugging Face (bf16 diffusers and ComfyUI formats)
  • Code: ModelTC/Minimax-H3-Turbo on GitHub, with a batch inference script and NFE/LoRA comparisons
