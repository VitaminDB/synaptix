# Alissonerdx/Minimax-H3-ComfyUI

Источник: https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI · скачано 2026-11-28
Загрузок: 329372 · лайков: 694 · базовая: ["MiniMaxAI/MiniMax-H3","MiniMaxAI/MiniMax-H3-Image"] · теги: lora,comfyui · обновлено: 2026-11-16T13:36:36.000Z

---

---
license: other
license_name: mixed-asset-licenses
license_link: https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/LICENSES.md
base_model:
- MiniMaxAI/MiniMax-H3
- MiniMaxAI/MiniMax-H3-Image
tags:
- lora
- video
- comfyui
- minimax-h3
- minimax-h3-image
---

# MiniMax H3 LoRAs for ComfyUI

Experimental LoRAs for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), packaged for
ComfyUI and tested mainly with the
[`ref2va` weights](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models).

These are research models, not finished tools. Results vary substantially with the source clip,
resolution, prompt and reference images.

## Models

Click a model name to download it. Full filenames, sizes and setup are in each guide.

### Character and head swap

| Model | What it does | Trigger | Guide · Workflows |
|---|---|---|---|
| [Character Swap v1.1](loras/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors) **(recommended)** | Replace the main subject with the person in a picture | `body_swap:` | [Guide](docs/character-swap.md) · [Planner](workflows/minimax_h3_character_swap_workflow_v1.1.json) · [Native](workflows/minimax_h3_character_swap_workflow_native_nodes_v1.0.json) |
| [Character Swap v1.0](loras/minimax_h3_character_swap_v1.0_ref2va_r64_gl.safetensors) | Previous release (rank 64) | `body_swap:` | [Guide](docs/character-swap.md) · [Planner](workflows/minimax_h3_character_swap_workflow_v1.0.json) · [Native](workflows/minimax_h3_character_swap_workflow_native_nodes_v1.0.json) |
| [Preprocessor Map](loras/minimax_h3_character_swap_v1.0_preprocessor_map_fro099.safetensors) | Swap up to 3 color-marked subjects from normal maps or DWPose | `character_swap:` | [Guide](docs/character-swap-preprocessor-map.md) · [Video](workflows/minimax_h3_character_swap_preprocessor_map_workflow_v1.0.json) · [Image](workflows/minimax_h3_image_character_swap_preprocessor_map_workflow_v1.0.json) |
| [Image Multiple by Color v0.1](loras/minimax_h3_image_character_swap_multiple_by_color_fro_v0.1.safetensors) **(new)** | H3 Image: swap up to 3 color-marked people in a still image | automatic | [Guide](docs/character-swap-image-multiple-by-color.md) · [Workflow](workflows/minimax_h3_image_character_swap_multiple_by_color_workflow_v0.1.json) |
| [Image Multiple by Color v0.1 · alternative](loras/minimax_h3_image_character_swap_multiple_by_color_fro_v0.1_alternative.safetensors) | Same task; better source lighting, weaker swap | automatic | [Guide](docs/character-swap-image-multiple-by-color.md) · [Workflow](workflows/minimax_h3_image_character_swap_multiple_by_color_workflow_v0.1.json) |
| [Head Swap v1.0](loras/minimax_h3_head_swap_v1.0_r32.safetensors) | Replace only the head (experimental) | `head_swap:` | [Guide](docs/head-swap.md) · [Workflow](workflows/minimax_h3_head_swap_workflow.json) |

### Motion Transfer

| Model | What it does | Trigger | Guide · Workflows |
|---|---|---|---|
| [Motion Transfer v1.0](loras/minimax_h3_motion_transfer_v1.0_ref2va.safetensors) | The pictures perform the guide's motion (Guide Latents) | `motion_transfer:` | [Guide](docs/motion-transfer.md) · [Planner](workflows/minimax_h3_motion_transfer_workflow_planner_v1.0.json) · [Native](workflows/minimax_h3_motion_transfer_workflow_native_nodes_v1.0.json) |
| [Motion Transfer TARA v1.0](loras/minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors) | Same, guide as a native video reference aligned by TARA; best for several subjects | `motion_transfer:` | [Guide](docs/motion-transfer.md) · [Planner](workflows/minimax_h3_motion_transfer_tara_workflow_planner_v1.0.json) · [Native](workflows/minimax_h3_motion_transfer_tara_workflow_native_nodes_v1.0.json) |
| [Motion Transfer FFP v1.0](loras/minimax_h3_motion_transfer_ffp_v1.0_ref2va.safetensors) | Edit the first frame; the edit follows the whole clip | `vfx_edit:` | [Guide](docs/motion-transfer.md#motion-transfer-ffp-v10-first-frame) · [Henge](https://github.com/alisson-anjos/Henge) |
| [SolarRenderer control](loras/solarrenderer/solarrenderer_control_comfy_fro_dyn99.safetensors) ([full](loras/solarrenderer/solarrenderer_control_comfy.safetensors)) | SolarRenderer's control LoRA in ComfyUI H3 keys | six-part prompt | [Guide](docs/motion-transfer.md#solarrenderer-loras) · [Henge](https://github.com/alisson-anjos/Henge) |
| [SolarRenderer 4-step](loras/solarrenderer/solarrenderer_4step_comfy_fro_dyn99.safetensors) ([full](loras/solarrenderer/solarrenderer_4step_comfy.safetensors)) | SolarRenderer's step distillation; load before control | — | [Guide](docs/motion-transfer.md#solarrenderer-loras) · [Henge](https://github.com/alisson-anjos/Henge) |

### Editing, enhancement and animation

| Model | What it does | Trigger | Guide · Workflows |
|---|---|---|---|
| [VFX Edit v1.0](loras/minimax_h3_vfx_edit_v1.0_r128.safetensors) | Prompt-directed edits over a source video | `vfx_edit:` | [Guide](docs/vfx-edit.md) · [Workflow](workflows/minimax_h3_vfxedit_wokflow.json) |
| [VFX Edit FFP v1.0](loras/minimax_h3_vfx_edit_v1.0_r128_ffp.safetensors) | Higher resolution, experimental first-frame propagation | `vfx_edit:` | [Guide](docs/vfx-edit.md) · [Workflow](workflows/minimax_h3_vfxedit_wokflow.json) |
| [Style Transfer v1.0](loras/minimax_h3_style_transfer_v1.0_r64.safetensors) | Apply a style from a picture or text | `style_transfer:` | [Guide](docs/style-transfer.md) · [Workflow](workflows/minimax_h3_style_transfer_workflow.json) |
| [LMS / Sharpness v1.0](loras/minimax_h3_lms_v1.0_r64.safetensors) | Restore detail from an aligned source | fixed caption | [Guide](docs/lms.md) · [Workflow](workflows/minimax_h3_lms_workflow.json) |
| [Live Wallpaper R32](loras/minimax_h3_live_wallpaper_v1.0_ref2va_r32.safetensors) | Animate 1–3 pictures; more faithful faces | `live_wallpaper:` | [Guide](docs/live-wallpaper.md) · [Workflow](workflows/minimax_h3_live_wallpaper_workflow.json) |
| [Live Wallpaper R64](loras/minimax_h3_live_wallpaper_v1.0_ref2va_r64.safetensors) | Same, with prompt-directed camera | `live_wallpaper:` | [Guide](docs/live-wallpaper.md) · [Workflow](workflows/minimax_h3_live_wallpaper_workflow.json) |

## Motion Transfer and TARA

**Four downloadable workflows:** [Guide Latents / native](workflows/minimax_h3_motion_transfer_workflow_native_nodes_v1.0.json), [Guide Latents / Planner](workflows/minimax_h3_motion_transfer_workflow_planner_v1.0.json), [TARA / native](workflows/minimax_h3_motion_transfer_tara_workflow_native_nodes_v1.0.json), [TARA / Planner](workflows/minimax_h3_motion_transfer_tara_workflow_planner_v1.0.json).

**References are 1..N pictures with roles assigned by the prompt, not a required person/background pair.** Several images can describe one identity, multiple characters, outfits, objects, scenery or style. A separate background image is optional. See the [Motion Transfer guide](docs/motion-transfer.md) for model setup, multiple-reference examples and the **modified temporal Comfy Kitchen kernel** required for TARA's fused one-pass Kitchen path. The TARA workflows default to Flash partitions, which does not require that extension.

**Training adapter V2.** The Motion Transfer LoRAs were trained on top of
[ostris' training adapter V2](https://huggingface.co/ostris/minimax_h3_training_adapter)
(`minimax_h3_ref2va_training_adapter_v2.safetensors`) kept frozen. Since 2026-11-02 the v1.0 and TARA files carry it:
LoRA x1 + adapter x1 merged by exact rank concat, then per-module SVD keeping 99% of the Frobenius energy (185 MB and
126 MB instead of 620 MB + 310 MB). Load only turbo + motion LoRA; **do not load the adapter again**. In same-seed tests
they matched the full LoRA + separate adapter, including TARA's frame sync across camera cuts. The previous files
(without the adapter) are kept in [`loras/base_loras_ignore/`](loras/base_loras_ignore).

**TARA-H3 — Temporally Aligned Reference Attention.** Native reference conditioning alone did not reliably reproduce the
timing of a source performance in H3, while aligned guide latents (Add Guide) copy it frame by frame. TARA keeps the
motion video as native reference tokens, translates its temporal RoPE coordinates onto the generated video's clock, and
adds a bounded temporal penalty only to generated-video queries attending to motion-reference keys; image references
keep their native coordinates and the attention change adds no learned layers. Motion Transfer TARA v1.0 was trained for
this method. Node: `BFSMiniMaxH3NativeTemporal` in [ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes).
Technical report (v0.1.2, experimental, not peer reviewed):
[TARA-H3: Temporally Aligned Reference Attention for Native Video Conditioning](https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3).

All of these run in [Henge](https://github.com/alisson-anjos/Henge), the open-source studio built on them.

## Latent Upscaler

[`h3_upscaler_lms_v0.1`](latent_upscaler/h3_upscaler_lms_v0.1.safetensors) is the recommended FP16
release of an experimental latent-upscaler checkpoint. It was trained for 2,000 steps with the same
sharpness dataset used for the LMS LoRA and initialized from the
[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
project and its
[`minimax_h3_latent_upscaler_3d_fp16.safetensors`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/blob/main/minimax_h3_latent_upscaler_3d_fp16.safetensors)
checkpoint.

See the [Latent Upscaler guide](docs/latent-upscaler.md) for setup, scope and limitations.

## Still images with Pruna P-H3-Image (guide-latent LoRAs)

[`PrunaImageH3WorkaroundGuideLatents.json`](workflows/PrunaImageH3WorkaroundGuideLatents.json) is a workaround to use
[PrunaAI's P-H3-Image](https://huggingface.co/PrunaAI/P-H3-Image-Comfy) with `MiniMaxH3AddGuide`, so LoRAs trained on
guide latents can edit a single image. `PH3ImageConditioning` with no images gives the one-frame H3 latent,
`MiniMaxH3AddGuide` adds the image to edit as the guide at `frame_idx = 0`, `MiniMaxH3ReferenceToVideo` (length 5)
carries the prompt and the reference picture, and the P-H3-Image VAE decodes the single frame. The model stack is the
H3 `ref2va` weights, the Pika turbo LoRA, the P-H3-Image LoRA and the task LoRA, at 8 steps with `BasicGuider`.

Tested with Character Swap v1.1; Head Swap and the other guide-latent LoRAs load the same way. Keep the image size a
multiple of 32 (the workflow resizes to a 2048 px long side).

## Common setup

Place LoRAs in `ComfyUI/models/loras/`. The included workflows list their required models and
custom nodes inside the graph. Keep the output aspect ratio matched to the source clip and use an
H3-supported frame count (`17n + 5`: 5, 22, 39, 56, 73, 90, 107, 124, ...).

The LMS, Style Transfer, VFX Edit and Head Swap models use an aligned source-video guide through
`MiniMaxH3AddGuide` at `frame_idx = 0`. This is different from a native image/video reference:
the guide shares the generated video's spatial and temporal grid.

## License and credits

Licenses are specified by asset. The separate [Character Swap license](LICENSE_CHARACTER_SWAP.md) and
[Motion Transfer license](LICENSE_MOTION_TRANSFER.md) apply only to the Character Swap and Motion Transfer LoRA
weight files listed in [LICENSES.md](LICENSES.md); they do not apply to the other LoRAs, workflows, docs, or examples.
Use of these LoRAs, including commercial use in videos and films, requires no prior
permission. Organizations with over US $1,000,000 in annual gross revenue must credit its origin
for commercial use. Offering the LoRA as a hosted API, SaaS, or other remote model service is not
allowed. See the full [Character Swap](LICENSE_CHARACTER_SWAP.md) and [Motion Transfer](LICENSE_MOTION_TRANSFER.md) licenses and check each other asset's applicable
terms before use.

Base model credit to [MiniMaxAI](https://huggingface.co/MiniMaxAI/MiniMax-H3), ComfyUI-ready
weights to [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3), and guide-latent training
support to [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) and
[ComfyUI-AIToolkit-MiniMaxH3](https://github.com/ostris/ComfyUI-AIToolkit-MiniMaxH3).
