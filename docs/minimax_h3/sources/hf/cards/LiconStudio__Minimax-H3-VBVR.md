# LiconStudio/Minimax-H3-VBVR

Источник: https://huggingface.co/LiconStudio/Minimax-H3-VBVR · скачано 2026-11-28
Загрузок: 16562 · лайков: 197 · базовая: ["MiniMaxAI/MiniMax-H3"] · теги:  · обновлено: 2026-11-20T07:53:21.000Z

---

---
license: apache-2.0
library_name: minimax-h3
datasets:
- Video-Reason/VBVR-Pro-SFT-Video
base_model:
- MiniMaxAI/MiniMax-H3
---

## Updates

### V2 Released

**V2 is now available**, offering more stable results compared with V1.

For **MiniMax-H3 INT8 inference**, we recommend [ComfyUI-Licon-H3-INT8-Backend](https://github.com/liconstudio/ComfyUI-Licon-H3-INT8-Backend) for better generation quality. Place the **Licon H3 INT8 Backend** node after your model and LoRA loaders and select **Legacy INT8 (Kitchen 0.2.35)** to use the legacy comfy-kitchen backend.

## Overview

**Licon VBVR** is an image-to-video LoRA for **MiniMax-H3**, trained on VBVR data rebuilt and reprocessed specifically for H3.

It improves object consistency and character action logic, reducing unintended object duplication, clipping, and interpenetration during motion and interactions.

## Key Improvements

- **Reduced object duplication:** fewer unintended copies of objects during generation.
- **Reduced clipping and interpenetration:** improved spatial interactions between characters, objects, and surfaces.
- **Improved character action logic:** more coherent movement sequences and interactions.

## Training Data

The training data is derived from **[VBVR-Pro-SFT-Video](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video)**.

I rebuilt the VBVR training data around H3's characteristics, adapting data preparation and preprocessing to its training pipeline. The goal is to transfer structured motion and interaction patterns into practical video generation, with a focus on object consistency, spatial relationships, and action logic.

## Video Comparisons

<table>
  <tr>
    <th align="center" colspan="3">exmaple01</th>
  </tr>
  <tr>
    <th align="center" width="33%">Original MiniMax-H3-FL2VA</th>
    <th align="center" width="33%">MiniMax-H3 + VBVR V1</th>
    <th align="center" width="33%">MiniMax-H3 + VBVR V2</th>
  </tr>
  <tr>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/01/original.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/01/VBVR-V1.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/01/VBVR-V2.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
  </tr>
</table>

<table>
  <tr>
    <th align="center" colspan="3">exmaple02</th>
  </tr>
  <tr>
    <th align="center" width="33%">Original MiniMax-H3-FL2VA</th>
    <th align="center" width="33%">MiniMax-H3 + VBVR V1</th>
    <th align="center" width="33%">MiniMax-H3 + VBVR V2</th>
  </tr>
  <tr>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/04/original.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/04/VBVR-V1.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
    <td align="center">
      <video src="https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/Validation-V2/04/VBVR-V2.mp4" controls playsinline preload="metadata" width="240"></video>
    </td>
  </tr>
</table>


## Acknowledgments

Thanks to **[MiniMax](https://huggingface.co/MiniMaxAI/MiniMax-H3)** for the base model and the **[VBVR team](https://huggingface.co/datasets/Video-Reason/VBVR-Pro-SFT-Video)** for the source dataset.
