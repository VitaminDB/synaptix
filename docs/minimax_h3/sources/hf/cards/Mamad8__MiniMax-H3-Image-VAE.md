# Mamad8/MiniMax-H3-Image-VAE

Источник: https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE · скачано 2026-11-28
Загрузок: 10679 · лайков: 100 · базовая:  · теги: comfyui · обновлено: 2026-08-08T21:54:39.000Z

---

---
tags:
- minimax-h3
- vae
- image
- comfyui
---

# Experimental MiniMax H3 single-image VAE

An experimental image-specialized MiniMax H3 VAE that directly decodes a single temporal latent (`T=1`) into one image. It was made to provide a usable direct-one-latent image path, not as a claim of state-of-the-art or high-fidelity image reconstruction. It is distributed as a standard merged H3 VAE checkpoint: no custom node or separate decoder head is required.

> [!IMPORTANT]
> **This checkpoint is for images only. Do not use it as a replacement for the original H3 VAE in video workflows.** Its image-specialized decoder materially regresses multi-frame video reconstruction and can introduce patch-grid ghosting and cross-frame mixing. Keep the original MiniMax H3 VAE for video.

> [!NOTE]
> Reconstruction quality is limited. Outputs can remain soft and can lose fine text, thin contours, hair, foliage and microtexture; the example below is intended to show the actual result rather than imply a quality guarantee.

## Example

Source images are shown on the left and reconstructions from this VAE on the right.

![Source images and step-1597 reconstructions](example_reconstruction.png)

## How it was trained

The original H3 encoder was frozen while the full decoder and `post_quant_conv` were fine-tuned to reconstruct the real source image directly from one H3 temporal latent. Training used a progressive 256→384 px curriculum and one pass over 51,083 unique images (41,259 Booru Essence images and 9,824 pseudo-camera photographs), with a decoder-aware objective combining Charbonnier reconstruction, edge, SSIM, and low-weight FDL/FDL-PIPS losses. The selected step-1597 decoder was merged back into the standard H3 VAE checkpoint, so it loads through normal H3 VAE tooling.

## Validation

On a balanced unseen 512 px set containing 32 photographs and 32 art images, the selected checkpoint measured:

- **30.4425 dB PSNR**
- **0.939257 SSIM**
- **0.016817 MAE**

The two live ComfyUI examples above reached 31.8420 dB / 0.898509 SSIM and 32.0064 dB / 0.940569 SSIM. Additional deployment checks from approximately 1 to 3 megapixels reached 31.55–33.43 dB PSNR.

## ComfyUI usage

Download `minimax_h3_t1_image_vae_step1597.safetensors` into:

```text
ComfyUI/models/vae/
```

Load it with ComfyUI's standard **Load VAE** node and use the normal VAE encode/decode nodes. The model expects MiniMax H3's 24-channel latent format and is intended for single-image (`T=1`) round trips.

## Checkpoint

- Size: 5,207,808,784 bytes
- SHA-256: `6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`
- Training checkpoint: step 1597

Model usage remains subject to the applicable MiniMax H3 license and terms.
