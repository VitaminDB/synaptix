# SOLRICKS/MiniMax-H3-Upscaler

Источник: https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler · скачано 2026-11-28
Загрузок: 6419 · лайков: 8 · базовая: MiniMaxAI/MiniMax-H3 · теги: comfyui,lora · обновлено: 2026-11-26T21:40:28.000Z

---

---
license: other
license_name: file-specific-model-licenses
license_link: https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/blob/main/LICENSES.md
pipeline_tag: video-to-video
tags:
- comfyui
- minimax-h3
- video-upscaling
- lora
- int8
- experimental
- latent-upscaler
base_model: MiniMaxAI/MiniMax-H3
---

# MiniMax H3 Upscaler

Model files and example workflows for a custom ComfyUI node that upscales videos and enhances detail. 

Upscale an existing H3 video with a learned latent upscaler, then refine detail with an H3 LoRA. The node uses ComfyUI's built-in model implementations and returns images and audio.

## Comparison

<table>
  <thead>
    <tr><th>Before — 1376×768</th><th>After — MiniMax H3 Upscaler · 1952×1088</th></tr>
  </thead>
  <tbody>
    <tr>
      <td width="50%"><video controls playsinline preload="metadata" width="100%" src="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax_H3_I2V.mp4"></video></td>
      <td width="50%"><video controls playsinline preload="metadata" width="100%" src="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax_H3_I2V_Upscaler.mp4"></video></td>
    </tr>
    <tr>
      <td align="center"><a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax_H3_I2V.mp4">Open original video</a></td>
      <td align="center"><a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/Minimax_H3_I2V_Upscaler.mp4">Open upscaled video</a></td>
    </tr>
  </tbody>
</table>

## Downloads

<table width="100%" style="width:100%; table-layout:fixed;">
  <thead>
    <tr>
      <th width="60%" align="left">File</th>
      <th width="40%" align="left">Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="overflow-wrap:anywhere;">
        <a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/h3_upscale_refiner_int8.safetensors?download=true">h3_upscale_refiner_int8.safetensors</a>
      </td>
      <td>
        <code style="white-space:normal; overflow-wrap:anywhere;">ComfyUI/models/loras/</code>
      </td>
    </tr>
    <tr>
      <td style="overflow-wrap:anywhere;">
        <a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors?download=true">minimax_h3_latent_upscaler_<wbr>3d_conv_v1_fp16.safetensors</a>
      </td>
      <td>
        <code style="white-space:normal; overflow-wrap:anywhere;">ComfyUI/models/<wbr>latent_upscale_models/</code>
      </td>
    </tr>
    <tr>
      <td style="overflow-wrap:anywhere;">
        <a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/MiniMax_H3_Upscaler.json?download=true">MiniMax_H3_Upscaler.json</a>
      </td>
      <td>Load in ComfyUI</td>
    </tr>
    <tr>
      <td style="overflow-wrap:anywhere;">
        <a href="https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/resolve/main/MiniMax_H3_Upscaler_API.json?download=true">MiniMax_H3_Upscaler_API.json</a>
      </td>
      <td>API client</td>
    </tr>
  </tbody>
</table>

You also need a MiniMax H3 FL2VA diffusion model and H3 video VAE, such as the files in [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental). They are separate downloads and are not bundled here. The example workflow uses [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) for MP4 export.

## Use in ComfyUI

1. Double-click an empty area of the ComfyUI canvas, type **solricks** in the node search, and select **MiniMax H3 Upscaler**.
2. Download the two model files into the folders listed above.
3. Load `MiniMax_H3_Upscaler.json`, choose your source video, and select the installed diffusion model and VAE filenames.
4. In **MiniMax H3 Upscaler**, select the FP16 latent upscaler and `h3_upscale_refiner_int8.safetensors`.
5. Start at 2.0 megapixels with temporal chunking and spatial tiling enabled. The target must be at least as large as the input; dimensions snap to H3's 32-pixel grid.

Use a diffusion model without generation/Turbo/style LoRAs. The upscaler node adds the refiner itself. The workflow preserves source FPS and passes source audio through. Its optional text/audio conditioning ports are left unconnected, so no text encoder or audio VAE is required by this example.

## Licenses

The files have different licenses; see [LICENSES.md](LICENSES.md) and [NOTICE](https://huggingface.co/SOLRICKS/MiniMax-H3-Upscaler/blob/main/NOTICE). The INT8 refiner follows the MiniMax H3 Community License. The FP16 latent upscaler is Apache-2.0. SOLRICKS' node code license does not replace the model licenses.

This repository belongs in the [Visual AI Models & Tools](https://huggingface.co/collections/SOLRICKS/visual-ai-models-and-tools) collection.
