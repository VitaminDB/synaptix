# Veda-Sparse/Minimax-H3-R2VA-Veda-Preview

Источник: https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview · скачано 2026-11-28
Загрузок: 26984 · лайков: 57 · базовая: ["MiniMaxAI/MiniMax-H3","larryvrh/MiniMax-H3-Turbo-Lora"] · теги:  · обновлено: 2026-10-08T10:18:08.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model:
- MiniMaxAI/MiniMax-H3
- larryvrh/MiniMax-H3-Turbo-Lora
pipeline_tag: text-to-video
tags:
- ref-to-video
- video
- audio
- text-to-audio-video
- sparse-attention
- block-sparse
- minimax-h3
- veda
- fp8
---

<p align="center">
  <img src="miowtion-logo.svg" width="320" alt="Miowtion">
</p>

# Veda-MiniMax-H3 R2VA (Preview)

[Demo](https://veda.hanxiao.run/) ·
[Project Page](https://veda-sparse.github.io/) ·
[Code](https://github.com/veda-sparse/Miowtion/tree/main) ·
[Paper](https://arxiv.org/abs/2605.30325) ·
[Deployment Guide](AGENTS.md)

## Introduction

**Veda** is a learned sparse-attention method for video diffusion models. This
checkpoint brings Veda to
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
reference-to-audio-video generation. It selects visual reference and generated
video tiles independently; text, VLM conditioning, and audio attention remain
dense. The predictor is not a LoRA and leaves the backbone weights unchanged.

This preview was fine-tuned for 600 updates from the T2VA predictor on
5.17-second R2VA clips. It uses the MiniMax-H3 Turbo LoRA for eight-step
generation, with up to 32 visual-reference tiles and 32 current-video tiles
per query tile and head. The predictor is stored in FP8 and dequantized to BF16
when loaded.

### Highlights

- **Learned sparse attention.** Visual reference and generated-video attention
  are selected independently; text, VLM conditioning, and audio stay dense.
- **Fixed visual budgets.** The released preview uses 32 reference tiles and
  32 current-video tiles per query tile and head.
- **Plug-and-play predictor.** The FP8 predictor is separate from the H3
  backbone and Turbo LoRA weights.

## Samples

The examples below use the same seed, prompt, geometry, and Turbo LoRA for
Dense and Veda. Each generated output is 1344×768, 124 frames, and about
5.17 seconds. Dense (left) and Veda (right) are stitched into one synchronized
comparison video, with attention and denoising speedups in its top title bar.
Each comparison uses the original Dense audio track.

<table>
<thead>
<tr><th>Task</th><th>References</th><th>Dense (left) · Veda (right)</th></tr>
</thead>
<tbody>
<tr>
  <td><strong>Pink suit and lamb</strong><br>
  A man in a bright pink suit holds a black lamb in a sunlit pasture, speaks a
  short line, and keeps the source soundtrack.</td>
  <td>
    <strong>Video 1 · appearance and scene</strong><br>
    <video controls preload="none" width="220" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/references/video_1.mp4"></video><br>
    <strong>Audio 1 · source soundtrack</strong><br>
    <audio controls preload="none" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/references/audio_1_soundtrack.wav"></audio><br>
    <strong>Audio 2 · male voice reference</strong><br>
    <audio controls preload="none" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/references/audio_2_voice.mp3"></audio>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/01_pink_lamb/veda.mp4">Open Veda video</a>
  </td>
</tr>
<tr>
  <td><strong>Anime train by the coast</strong><br>
  An anime traveler watches a bright coastline pass outside a moving train
  window.</td>
  <td><strong>Picture 1 · character and coastal scene</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/references/picture_1.png" alt="Anime traveler beside a coastal train window" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/references/picture_1.png">open reference</a>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/02_anime_train_coastal_window/veda.mp4">Open Veda video</a>
  </td>
</tr>
<tr>
  <td><strong>Anime twilight carnival</strong><br>
  An anime woman walks through a fairground as the Ferris wheel and lights turn
  on at blue hour.</td>
  <td><strong>Picture 1 · character</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/references/picture_1_character.png" alt="Anime character reference" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/references/picture_1_character.png">open character reference</a><br>
    <strong>Picture 2 · fairground</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/references/picture_2_fairground.png" alt="Anime fairground reference" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/references/picture_2_fairground.png">open fairground reference</a>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/03_anime_twilight_carnival/veda.mp4">Open Veda video</a>
  </td>
</tr>
<tr>
  <td><strong>Aurora over the lake</strong><br>
  A green-violet aurora slowly grows above a mountain lake and appears in the
  calm reflection.</td>
  <td><strong>Picture 1 · lake and sky</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/references/picture_1_lake.png" alt="Aurora over a mountain lake" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/references/picture_1_lake.png">open lake reference</a><br>
    <strong>Video 1 · aurora motion</strong><br>
    <video controls preload="none" width="220" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/references/video_1_aurora_motion.mp4"></video>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/04_aurora_lake_reflection/veda.mp4">Open Veda video</a>
  </td>
</tr>
<tr>
  <td><strong>Red fox in autumn woods</strong><br>
  A red fox crosses a warm autumn woodland, slows, and looks toward the
  camera.</td>
  <td><strong>Picture 1 · fox appearance</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/references/picture_1_fox.png" alt="Red fox appearance reference" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/references/picture_1_fox.png">open fox reference</a><br>
    <strong>Video 1 · fox motion</strong><br>
    <video controls preload="none" width="220" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/references/video_1_fox_motion.mp4"></video>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/05_fox_autumn_motion/veda.mp4">Open Veda video</a>
  </td>
</tr>
<tr>
  <td><strong>Perfume by the pool</strong><br>
  A glass perfume bottle catches warm light beside rippling water in a stone
  courtyard.</td>
  <td><strong>Picture 1 · product and pool</strong><br>
    <img src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/references/picture_1_product.png" alt="Perfume bottle beside a pool" width="210"><br>
    <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/references/picture_1_product.png">open product reference</a>
  </td>
  <td width="640">
    <video controls preload="metadata" width="640" height="198" src="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/dense_vs_veda.mp4"></video>
    <br><a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/dense_vs_veda.mp4">Open comparison</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/dense.mp4">Open Dense video</a> · <a href="https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/resolve/main/media/showcase/06_perfume_pool_reflection/veda.mp4">Open Veda video</a>
  </td>
</tr>
</tbody>
</table>

## Performance

Measured with eight-step Turbo LoRA inference on one RTX PRO 6000 Blackwell
GPU. Veda uses fixed budgets of 32 reference and 32 current-video tiles.

| GPU | Clip | Attention speedup | End-to-end speedup |
|---|---|---:|---:|
| RTX PRO 6000 Blackwell | 16:9 · 5.17 s | 2.42× | 1.69× |
| RTX PRO 6000 Blackwell | 9:16 · 5.17 s | 2.65× | 1.67× |
| RTX PRO 6000 Blackwell | 1:1 · 5.17 s | 2.06× | 1.37× |
| RTX PRO 6000 Blackwell | 4:3 · 5.17 s | 2.19× | 1.54× |
| RTX PRO 6000 Blackwell | 16:9 · 10.1 s | 2.84× | 2.05× |
| RTX PRO 6000 Blackwell | 16:9 · 14.4 s | 2.90× | 2.20× |

## Usage

See [AGENTS.md](AGENTS.md) for installation and GPU-specific deployment notes.
The commands below use the R2VA code on the
[`main` branch](https://github.com/veda-sparse/Miowtion/tree/main).

```bash
git clone --recurse-submodules --branch main https://github.com/veda-sparse/Miowtion.git
cd Miowtion
pip install -e '.[gpu,encode]'

hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "Ref2VA/*" \
  --local-dir weights/MiniMax-H3
hf download larryvrh/MiniMax-H3-Turbo-Lora \
  minimax_h3_turbo_v4_step600_ema.safetensors --local-dir weights/turbo_lora
hf download Veda-Sparse/Minimax-H3-R2VA-Veda-Preview \
  --local-dir weights/veda/h3-r2va-preview
```

Prepare the reference files, then encode and generate:

```bash
python scripts/prepare_official_r2va_demo.py \
  --out artifacts/examples/minimax_h3_ref2va
python scripts/encode_samples.py --root weights/MiniMax-H3 \
  --manifest artifacts/examples/minimax_h3_ref2va/manifests/prompts.jsonl \
  --out artifacts/samples/demo
python scripts/generate.py --config configs/infer_r2va_preview.yaml \
  --predictor weights/veda/h3-r2va-preview/minimax_h3_r2va_veda_preview_fp8.safetensors \
  --attention dense veda
```

## License

This model inherits the [MiniMax H3 Community License](LICENSE) from its base
model.

## Citation

```bibtex
@inproceedings{han2026veda,
  title={Veda: Scalable Video Diffusion via Distilled Sparse Attention},
  author={Han, Shihao and Yang, Hao and Hu, Xinting and Mei, Xiaofeng
          and Jiang, Yi and Qi, Xiaojuan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
