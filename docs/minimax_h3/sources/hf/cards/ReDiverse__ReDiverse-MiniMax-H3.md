# ReDiverse/ReDiverse-MiniMax-H3

Источник: https://huggingface.co/ReDiverse/ReDiverse-MiniMax-H3 · скачано 2026-11-28
Загрузок: 474 · лайков: 42 · базовая: lightx2v/Minimax-h3-Turbo · теги: lora,distillation,base_model:lightx2v/Minimax-h3-Turbo,base_model:adapter:lightx2v/Minimax-h3-Turbo · обновлено: 2026-11-19T16:57:16.000Z

---

---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model: lightx2v/Minimax-h3-Turbo
base_model_relation: adapter
pipeline_tag: text-to-video
tags:
- lora
- diversity
- distillation
- rediverse
- minimax-h3
extra_gated_prompt: "This adapter is a Model Derivative of MiniMax H3 and is governed by the MiniMax H3 Community License Agreement, including its Applicable Territory and Acceptable Use Policy. Request access only if the license allows your use."
extra_gated_button_content: "I have read the license and request access"
---

<p align="center"><img src="https://huggingface.co/spaces/ReDiverse/README/resolve/main/assets/banner.jpg" width="100%" alt="ReDiverse: Restoring Diversity in Distilled Image and Video Generators by Probing Activations"></p>

<p align="center">
  Furkan Horoz<sup>1</sup>, Ozgur Kara<sup>1</sup>, Kaan Oktay<sup>2</sup>, Adil Kaan Akan<sup>2</sup>, Atakan Kara<sup>2</sup>,
  Anand Bhattad<sup>3,†</sup>, James Matthew Rehg<sup>1,†</sup><br>
  <sup>1</sup>University of Illinois at Urbana-Champaign &nbsp;&nbsp; <sup>2</sup>fal &nbsp;&nbsp; <sup>3</sup>Johns Hopkins University<br>
  <sub><sup>†</sup>Equal advising.</sub>
</p>

<p align="center">
  <a href="https://rediverse-distillation.github.io/rediverse-distillation/"><img src="https://img.shields.io/badge/Project-Page-2ea44f.svg" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2611.18550"><img src="https://img.shields.io/badge/arXiv-2611.18550-b31b1b.svg?logo=arxiv" alt="arXiv 2611.18550"></a>
  <a href="https://github.com/H-furkan/rediverse"><img src="https://img.shields.io/badge/GitHub-Code-181717.svg?logo=github" alt="Code on GitHub"></a>
  <a href="https://huggingface.co/spaces/ozgurkara/ReDiverse-MiniMax-H3"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-MiniMax--H3-ff9d00.svg" alt="Demo on Hugging Face Spaces"></a>
  <a href="https://huggingface.co/ReDiverse"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20ReDiverse-all%20models-ffd21e.svg" alt="All ReDiverse models"></a>
</p>

**ReDiverse-FT adapter for MiniMax-H3.** Few-step distilled students keep their teacher's activation variance across seeds but
concentrate it into far fewer directions at one block of the first denoising step, so different seeds give near-identical
samples. This rank-32 LoRA on the student's blocks up to `transformer_blocks.45` is trained to match the teacher's activations there and
is active in the first sampling step only; the remaining steps run the unmodified student.

| Teacher | 4-step student | This adapter |
|---|---|---|
| [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | [Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), distilled from the teacher | ReDiverse-FT LoRA of the student, `rediverse_ft.safetensors` (612 MB) |

<p align="center"><img src="https://huggingface.co/spaces/ReDiverse/README/resolve/main/assets/h3_balloon.webp" width="100%" alt="MiniMax-H3, a hot air balloon floating: four seed-matched clips of the student and of ReDiverse"></p>

## Usage

**Demo:** try this adapter in the browser at [ozgurkara/ReDiverse-MiniMax-H3](https://huggingface.co/spaces/ozgurkara/ReDiverse-MiniMax-H3).

```bash
git clone https://github.com/H-furkan/rediverse && cd rediverse   # set up the environments of envs/README.md
hf download ReDiverse/ReDiverse-MiniMax-H3 --local-dir ReDiverse-MiniMax-H3
python -m h3.generate --arm rediverse_ft --adapter ReDiverse-MiniMax-H3 --prompt "a corgi running on the beach at sunset" --num-samples 4   # video env
```

`--arm rediverse_ft` (above) uses the adapter alone; `--arm rediverse` adds the training-free ReDiverse-TF directions, which first runs the teacher on 25 noises for each new prompt (about one GPU-hour on an 80 GB H100) and caches the result. Without `--adapter`, the code downloads the same file from [ReDiverse/ReDiverse](https://huggingface.co/ReDiverse/ReDiverse).

**ComfyUI:** [kabachuha/ReDiverse-MiniMax-H3-Comfy](https://huggingface.co/kabachuha/ReDiverse-MiniMax-H3-Comfy) is a community conversion of this adapter to ComfyUI's native LoRA format, with a 4-step example workflow.

| DINO diversity ↑ (VBench) | Student | ReDiverse-FT (this adapter) | ReDiverse (FT + TF) | Teacher (multi-step) |
|---|---:|---:|---:|---:|
| MiniMax-H3 | 36.5 | 44.1 | 50.9 | 52.5 |

## License

A Model Derivative of MiniMax H3, released for research purposes and governed by the [MiniMax H3 Community License Agreement](LICENSE), including its Applicable Territory (which excludes the European Union, the United Kingdom, the Republic of Korea and the United States of America) and Acceptable Use Policy. Check that the license allows your use before downloading or running MiniMax-H3 or this adapter. [NOTICE](NOTICE) holds the required notice. Powered by MiniMax H3.

## Citation

```bibtex
@article{horoz2026rediverse,
  title   = {Restoring Diversity in Distilled Image and Video Generators by Probing Activations},
  author  = {Horoz, Furkan and Kara, Ozgur and Oktay, Kaan and Akan, Adil Kaan and Kara, Atakan
             and Bhattad, Anand and Rehg, James Matthew},
  journal = {arXiv preprint arXiv:2611.18550},
  year    = {2026}
}
```
