# alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0

Источник: https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0 · скачано 2026-11-28
Загрузок: 31655 · лайков: 89 · базовая:  · теги: controlnet,controlnet-union · обновлено: 2026-09-22T09:57:14.000Z

---

---
license: other
license_link: LICENSE.md
license_name: minimax-h3-community-license-agreement
library_name: videox_fun
tags:
- controlnet
- controlnet-union
- video-to-video
- image-text-to-video
- text-to-video
- video-inpainting
tasks:
- text-to-video-synthesis
---

# MiniMax-H3-Fun-Controlnet-Union-2.0

[![Github](https://img.shields.io/badge/🎬%20Code-VideoX_Fun-blue)](https://github.com/aigc-apps/VideoX-Fun)

## What's new in 2.0

| | MiniMax-H3-Fun-Controlnet-Union (v1) | MiniMax-H3-Fun-Controlnet-Union-2.0 (this model) |
|--|--|--|
| Control conditions | 5 — Canny, Depth, HED, MLSD, Pose | **8 — + Scribble, Layout, Gray** |
| Control branch depth | 5 control blocks (layers `0, 10, 20, 30, 40`) | **10 control blocks** (layers `0, 5, 10, …, 45`) — skips injected every 5 of the 50 transformer blocks |
| Inpaint masked-pixel recipe | `pre_norm` (holes ≈ −2 in VAE input space, extreme dark) | **`post_norm`** (holes at 0, mid-gray, following Wan 2.1) — cleaner inpaint blending |
| Checkpoint contents | `control_proj_in` + 5 `control_blocks` (~6.8 GB) | `control_proj_in` + 10 `control_blocks` (~13.5 GB) |
| Required config | `minimax_h3_control.yaml` | **`minimax_h3_control_inpaint_post_norm.yaml`** |

Everything else is carried over from v1: `control_in_dim = 49` (latent + masked latent + mask, so the same branch does control and inpaint), `control_apply_audio = false`, guidance-distilled (`guidance_scale = 1.0`), and the same zero-gated skip-add into the main branch.

> **Loading a v1 config against this checkpoint is a silent failure.** With `minimax_h3_control.yaml` (5 blocks) the model builds only half the control branch; `load_state_dict(strict=False)` drops `control_blocks.5~9` as unexpected keys and misplaces the rest, producing wrong outputs. Always use `minimax_h3_control_inpaint_post_norm.yaml`.

## Model Card

| Name | Description |
|--|--|
| MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors | ControlNet-Union-2.0 branch weights for MiniMax-H3. Holds only the control branch (`control_proj_in` plus 10 `control_blocks`, about 13.5 GB) and is loaded on top of the base MiniMax-H3 transformer. One checkpoint supports 8 control conditions (Canny, Depth, HED, MLSD, Pose, Scribble, Layout, Gray) and video inpainting. |

## Model Features
- **Union control over 8 conditions**: one checkpoint handles Canny, Depth, HED, MLSD, Pose, Scribble, Layout and Gray control videos for video-to-video generation — no per-condition checkpoint switching.
- **Denser control injection**: the control branch attaches to 10 of the 50 transformer blocks (layers 0, 5, 10, 15, 20, 25, 30, 35, 40, 45); every control skip is added to the main branch through a zero-gated projection. This is roughly 2× the injection points of v1 and gives tighter structural adherence.
- **Guidance-distilled**: run with `guidance_scale = 1.0`, one forward pass per step, no classifier-free guidance needed.
- **Inpainting is supported, with the `post_norm` recipe**: the control input is widened to `control_in_dim = 49` (latent + masked latent + mask channels). Unlike v1, the masked pixels are zeroed *after* the ImageNet normalization (holes sit at 0 / mid-gray) rather than *before* it (holes landed near −2 / extreme dark), which improves how filled regions blend with kept regions. Use `examples/minimax_h3_fun/predict_v2v_control_inpaint.py`.
- `control_context_scale` scales every control skip before it is added to the main branch: `1.0` gives the strongest control (used for all results below), values below `1.0` weaken the guidance of the control video, `0.0` switches the control branch off.
- The generation follows the control video: the frame count snaps down to the largest `17 * n + 5` the video VAE can decode (duration capped at 15 seconds), the canvas keeps the control video's own aspect ratio at the `height * width` pixel budget (both multiples of 32), at a fixed 24 fps.
- Detailed prompts give better stability; we recommend describing the scene, the subject and the camera in the prompt.

## Supported control conditions

| Condition | Control signal | New in 2.0? |
|--|--|--|
| Canny | Canny edge map | |
| Depth | Monocular depth map | |
| HED | HED edge detection | |
| MLSD | Line-segment detection | |
| Pose | DWPose skeleton | |
| Scribble | Free-hand / sketch lines | ✅ |
| Layout | Bounding-box layout | ✅ |
| Gray | Grayscale (luminance) video | ✅ |

The **Layout** control videos follow the layout generation recipe of [Wan2.1-VACE](https://github.com/ali-vilab/VACE): per-subject bounding boxes (detected/tracked or given directly) are rendered as color-coded boxes on a white background, producing an ordinary RGB video that conditions the model. You can reuse the VACE-Annotators preprocessing tools (e.g. `vace_preproccess.py --task layout_track ...`) to produce layout videos from a reference video or a pair of bboxes.

## Results

All samples below are generated with `num_inference_steps = 40`, `guidance_scale = 1.0`, `control_context_scale = 1.00`, seed 43, canvas mode `control` at a 704×1280 pixel budget, 24 fps. In each pair the top row is the control video, the bottom row is the output.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr><td>Canny</td><td>Depth</td><td>HED</td><td>MLSD</td></tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/canny.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/depth.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/hed.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/mlsd.mp4" width="100%" controls muted></video></td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/canny.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/depth.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/hed.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/mlsd.mp4" width="100%" controls muted></video></td>
  </tr>
  <tr><td>Pose</td><td>Scribble ✨</td><td>Layout ✨</td><td>Gray ✨</td></tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/pose.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/scribble.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/layout.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/gray.mp4" width="100%" controls muted></video></td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/pose.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/scribble.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/layout.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/gray.mp4" width="100%" controls muted></video></td>
  </tr>
</table>

### Inpainting (`post_norm`)

A masked region of the source video is re-drawn from the prompt while the rest of the frame is preserved. The mask video is white where the content should be re-generated and black where it should be kept.

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr><td>Source video</td><td>Mask</td><td>Inpaint output</td></tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/inpaint_source.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/asset/inpaint_mask.mp4" width="100%" controls muted></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0/resolve/main/results/inpaint.mp4" width="100%" controls muted></video></td>
  </tr>
</table>

## Inference
Go to the VideoX-Fun repository for more details.

Please clone the VideoX-Fun repository and create the required directories:

```sh
# Clone the code
git clone https://github.com/aigc-apps/VideoX-Fun.git

# Enter VideoX-Fun's directory
cd VideoX-Fun

# Create model directories
mkdir -p models/Diffusion_Transformer
```

Then download the base MiniMax-H3 model and this checkpoint into `models/Diffusion_Transformer`.

```
📦 models/
├──  Diffusion_Transformer/
│   ├── 📂 MiniMax-H3/
│   └──  MiniMax-H3-Fun-Controlnet-Union-2.0/
│       └──  MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors
```

Then edit the settings at the top of `examples/minimax_h3_fun/predict_v2v_control.py` (or `predict_v2v_control_inpaint.py` for inpainting) and run it.

```python
model_name          = "models/Diffusion_Transformer/MiniMax-H3"
config_path         = "config/minimax_h3/minimax_h3_control_inpaint_post_norm.yaml"
transformer_path    = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union-2.0/MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors"
control_video       = "your_control_video.mp4"
prompt              = "your prompt"
```

```sh
python examples/minimax_h3_fun/predict_v2v_control.py
```

Notes:
- `config_path` **must** be `config/minimax_h3/minimax_h3_control_inpaint_post_norm.yaml`. It builds the control branch exactly as the checkpoint expects (`control_blocks_places: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]`, `control_in_dim: 49`, `control_apply_audio: false`, `inpaint_masked_pixel_mode: post_norm`); the v1 `minimax_h3_control.yaml` (5 blocks) will silently drop half the control weights.
- For pure control (no inpaint input) the pipeline zero-pads the mask channels, so this inpaint checkpoint still runs plain Canny/Depth/… control correctly.
- The checkpoint is guidance-distilled: keep `guidance_scale = 1.0`; a value above 1 applies guidance twice and degrades the output.
- The control checkpoint carries only the control branch; the base MiniMax-H3 weights must be present in `model_name`.
- For the Layout condition, generate the control video with the [Wan2.1-VACE](https://github.com/ali-vilab/VACE) layout pipeline (see [Supported control conditions](#supported-control-conditions)); other control-video formats are unchanged from v1.
- Memory: the transformer (about 62 GB) plus the Qwen3-VL text encoder (about 62 GB) do not fit one 80 GB GPU fully loaded; use `model_group_offload` (fastest) or `model_cpu_offload_and_qfloat8` on a single 80 GB GPU.

## License

This model is a derivative of MiniMax-H3 and is released under the [MiniMax H3 Community License Agreement](LICENSE). Please read the license carefully, especially the territorial restrictions and the Acceptable Use Policy, before use.
