# alibaba-pai/MiniMax-H3-Fun-Controlnet-Union

Источник: https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union · скачано 2026-11-28
Загрузок: 37480 · лайков: 211 · базовая:  · теги: controlnet · обновлено: 2026-09-22T10:13:54.000Z

---

---
license: other
license_link: LICENSE
license_name: minimax-h3-community-license-agreement
library_name: videox_fun
tags:
- controlnet
- video-to-video
- text-to-video
- image-text-to-video
tasks:
- text-to-video-synthesis
---

# MiniMax-H3-Fun-Controlnet-Union

[![Github](https://img.shields.io/badge/🎬%20Code-VideoX_Fun-blue)](https://github.com/aigc-apps/VideoX-Fun)

> **This is a legacy version. A newer, better version is now available: [MiniMax-H3-Fun-Controlnet-Union-2.0](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0).**
> The 2.0 model produces **improved results** and is the recommended checkpoint for new usage. Please prefer it over this one.

MiniMax-H3-Fun-Controlnet-Union is a ControlNet-Union for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), trained with the VideoX-Fun pipeline. A single checkpoint conditions the MiniMax-H3 video generator on Canny, Depth, HED, MLSD or Pose control videos, and also runs video inpainting.

## Model Card

| Name | Description |
|--|--|
| MiniMax-H3-Fun-Controlnet-Union.safetensors | ControlNet-Union branch weights for MiniMax-H3. The file holds only the control branch (`control_proj_in` plus 5 `control_blocks`, about 6.8 GB) and is loaded on top of the base MiniMax-H3 transformer. One checkpoint supports Canny, Depth, HED, MLSD and Pose control conditions, and video inpainting. |

## Model Features
- Union control: one checkpoint handles Canny, Depth, HED, MLSD and Pose control videos for video-to-video generation, no per-condition checkpoint switching.
- The control branch attaches to 5 of the 50 transformer blocks (layers 0, 10, 20, 30, 40); every control skip is added to the main branch through a zero-gated projection.
- Guidance-distilled: run with `guidance_scale = 1.0`, one forward pass per step, no classifier-free guidance needed.
- Inpainting is supported: the control input is widened to `control_in_dim = 49` (latent + masked latent + mask channels); use `examples/minimax_h3_fun/predict_v2v_control_inpaint.py`.
- `control_context_scale` scales every control skip before it is added to the main branch: `1.0` gives the strongest control (used for all results below), values below `1.0` weaken the guidance of the control video, `0.0` switches the control branch off.
- The generation follows the control video: the frame count snaps down to the largest `17 * n + 5` the video VAE can decode (duration capped at 15 seconds), the canvas keeps the control video's own aspect ratio at the `height * width` pixel budget (both multiples of 32), at a fixed 24 fps.
- Detailed prompts give better stability; we recommend describing the scene, the subject and the camera in the prompt.

## Results

All samples below are generated with `num_inference_steps = 40`, `guidance_scale = 1.0`, `control_context_scale = 1.00`, seed 43.

### Canny

<table>
  <tr>
    <td width="50%">Control</td>
    <td width="50%">Output</td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/asset/canny_tokyo_street.mp4" width="100%" controls muted loop></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/results/canny_tokyo_street.mp4" width="100%" controls muted loop></video></td>
  </tr>
</table>

### Depth

<table>
  <tr>
    <td width="50%">Control</td>
    <td width="50%">Output</td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/asset/depth_astronaut.mp4" width="100%" controls muted loop></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/results/depth_astronaut.mp4" width="100%" controls muted loop></video></td>
  </tr>
</table>

### HED

<table>
  <tr>
    <td width="50%">Control</td>
    <td width="50%">Output</td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/asset/hed_trex_bmx.mp4" width="100%" controls muted loop></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/results/hed_trex_bmx.mp4" width="100%" controls muted loop></video></td>
  </tr>
</table>

### MLSD

<table>
  <tr>
    <td width="50%">Control</td>
    <td width="50%">Output</td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/asset/mlsd_village.mp4" width="100%" controls muted loop></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/results/mlsd_village.mp4" width="100%" controls muted loop></video></td>
  </tr>
</table>

### Pose

<table>
  <tr>
    <td width="50%">Control</td>
    <td width="50%">Output</td>
  </tr>
  <tr>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/asset/pose_dance.mp4" width="100%" controls muted loop></video></td>
    <td><video src="https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/resolve/main/results/pose_dance_flamenco.mp4" width="100%" controls muted loop></video></td>
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
├── 📂 Diffusion_Transformer/
│   ├── 📂 MiniMax-H3/
│   └── 📂 MiniMax-H3-Fun-Controlnet-Union/
│       └── 📦 MiniMax-H3-Fun-Controlnet-Union.safetensors
```

Then edit the settings at the top of `examples/minimax_h3_fun/predict_v2v_control.py` and run it.

```python
model_name          = "models/Diffusion_Transformer/MiniMax-H3"
config_path         = "config/minimax_h3/minimax_h3_control.yaml"
transformer_path    = "models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union/MiniMax-H3-Fun-Controlnet-Union.safetensors"
control_video       = "your_control_video.mp4"
prompt              = "your prompt"
```

```sh
python examples/minimax_h3_fun/predict_v2v_control.py
```

Notes:
- `config_path` must build the control branch exactly as trained (`control_blocks_places: [0, 10, 20, 30, 40]`, `control_in_dim: 49`, `control_apply_audio: false`); a mismatched layout makes the checkpoint fail to load.
- The checkpoint is guidance-distilled: keep `guidance_scale = 1.0`; a value above 1 applies guidance twice and degrades the output.
- The control checkpoint carries only the control branch; the base MiniMax-H3 weights must be present in `model_name`.
- Memory: the transformer (about 62 GB) plus the Qwen3-VL text encoder (about 62 GB) do not fit one 80 GB GPU fully loaded; use `model_group_offload` (fastest) or `model_cpu_offload_and_qfloat8` on a single 80 GB GPU.

## License

This model is a derivative of MiniMax-H3 and is released under the [MiniMax H3 Community License Agreement](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union/blob/main/LICENSE). Please read the license carefully, especially the territorial restrictions and the Acceptable Use Policy, before use.
