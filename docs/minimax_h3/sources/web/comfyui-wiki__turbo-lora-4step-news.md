# Source: https://comfyui-wiki.com/en/news/2026-08-07-minimax-h3-turbo-lightx2v

Fetched: 2026-11-28 (curl + w3m text dump)

---

Search pages, tutorials, news and nodes.

(Select language)

 1. Home
 2.
 3. News
 4.
 5. MiniMax H3 Turbo LoRA: 4-Step FL2V Distillation for ComfyUI

MiniMax H3 Turbo LoRA: 4-Step FL2V Distillation for ComfyUI

Last updated: August 7, 2026ComfyUI Wikinews

Lightx2v and ModelTC distill MiniMax H3 into a 4-step FL2V Turbo LoRA, cutting sampling from ~20 steps to 4 with a ComfyUI conversion and demo Spaces.

The Lightx2v team and ModelTC released a 4-step Turbo LoRA for MiniMax H3, distilling the video model's first-last-frame-to-video (FL2V) generation from roughly 20 sampling steps down to 4 (Hugging
Face, GitHub). The LoRA, about 1.3 GB, is an early preview: v0.1 focuses on FL2V quality, with a Ref2V distillation already on the roadmap.

MiniMax H3 Turbo LoRA example frame

A frame from the author's example output at 4 steps

What's in the release

  • Weights: minimax_h3_fl2v_turbo_4step_v0.1.safetensors on Hugging Face, distilled from the MiniMax-H3 base model
  • Inference code: the ModelTC repository provides batch MiniMax-H3 inference plus NFE/LoRA comparisons against the base model at 50 NFE
  • Demo Spaces: akhaliq's MiniMax-H3-Turbo-Lora Space and an official-org Space let you try the LoRA without local hardware
  • Roadmap: improve FL2V visual details and overall quality, then develop a Ref2V-based distillation

The release notes label v0.1 a preview: visual details still need improvement, and an updated version is in development.

ComfyUI usage

ComfyUI does not need a custom node for this LoRA. Kijai published a ComfyUI-compatible conversion in Kijai/MiniMax-H3_comfy (minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors, plus a
resized bf16 variant) within an hour of the release. Drop the file into models/loras/, load the base H3 checkpoint, and apply the LoRA at 0.75 strength with 4 steps using an er_sde or sa_solver
sampler, per Kijai's test notes.

Download MiniMax H3 workflow template

The official MiniMax H3 text-to-video workflow template is a good starting point; insert the Turbo LoRA between the model loader and the sampler and switch to the 4-step settings above.

    Note: the authors observed some differences between Diffusers and ComfyUI inference results (audio sounded more natural in Diffusers). An audio bug in the ComfyUI path was fixed and has been in
    stable ComfyUI since August 8 (no nightly required, per Kijai). For reproduction matching the paper results, follow the ModelTC inference script.

Example output shared by the authors in the release discussion

Availability
