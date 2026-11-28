# Raretutor/MiniMax-H3-Dual-Turbo-Lora-ComfyUI

Источник: https://huggingface.co/Raretutor/MiniMax-H3-Dual-Turbo-Lora-ComfyUI · скачано 2026-11-28
Загрузок: 1634 · лайков: 9 · базовая: ["MiniMaxAI/MiniMax-H3","larryvrh/MiniMax-H3-Turbo-Lora","TaoLiveAIGC/TaoMate-H3","ZhengmingYu/DMAD"] · теги: lora,turbo,comfyui · обновлено: 2026-11-22T08:11:04.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
  - MiniMaxAI/MiniMax-H3
  - larryvrh/MiniMax-H3-Turbo-Lora
  - TaoLiveAIGC/TaoMate-H3
  - ZhengmingYu/DMAD
pipeline_tag: text-to-video
library_name: diffusers
tags:
  - minimax-h3
  - lora
  - model-merge
  - video-generation
  - text-to-audio-video
  - 4-step
  - 8-step
  - dmad
  - turbo
  - streaming-video
  - comfyui
---
# Video Example

<a href="https://www.youtube.com/watch?v=YcQA3wchRmU" style="color: yellow;">https://www.youtube.com/watch?v=zUk6mqruWdA</a>


-------------------------------------------------

# MiniMax-H3 Merged Dual Turbo LoRAs

This repository hosts 2 custom merged LoRA adapters designed for the **MiniMax-H3** architecture. 
These adapters combine ultra-fast few-step inference, high-fidelity temporal consistency, synchronized stereo audio-video generation, and responsive streaming capabilities by merging three state-of-the-art community LoRA checkpoints:

1. **[larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)** (Apache-2.0)
2. **[TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)** (MiniMax H3 Community License Agreement)
3. **[ZhengmingYu/DMAD (`minimax_h3`)](https://huggingface.co/ZhengmingYu/DMAD/tree/main/minimax_h3)** (MiniMax H3 Community License Agreement / DMAD Adversarial Distillation)

---

## Dual LoRA Pair (Load Both Together)

> **Important:** These two files are **not** standalone variants—you must load and chain **both** LoRAs together in your workflow for proper results.

- **LoRA 1:** `turbo_lora_001.safetensors`
- **LoRA 2:** `turbo_lora_002.safetensors`

---

## Merge Details & Methodology

- **Base Foundation:** `MiniMaxAI/MiniMax-H3`
- **Upstream Adapters Merged:**
  - `larryvrh/MiniMax-H3-Turbo-Lora`(Turbo LoRA)
  - `TaoLiveAIGC/TaoMate-H3` (Turbo LoRA)
  - `ZhengmingYu/DMAD` (`DMAD LoRA)
- **Merge Strategy:** Linear weight interpolation (SLERP / DARE-TIES / Weighted Add) applied across matching attention and feed-forward projection layers into a complementary Dual LoRA Pair.
- **Precision:** `bfloat16` / `fp16`

---

## Inference Guidelines

- **Recommended Sampling Steps:** `6` (Range: `5 - 9`)
- **Guidance Scale (CFG):** `1.0` (Recommended for native CFG-distilled operation)
- **Scheduler / Sampler:** `lcm` + `simple`
- **Base Model Loading:** Load on top of standard `MiniMaxH3Transformer3DModel` (compatible with Diffusers and ComfyUI).

---

## Legal Notice, Terms & Licensing

### 1. Primary License & Mandatory Copyright Notice
Because these checkpoints are **Model Derivatives** of `MiniMaxAI/MiniMax-H3` and incorporate weights from `TaoLiveAIGC/TaoMate-H3` and `ZhengmingYu/DMAD` (`minimax_h3/`), this repository is distributed under the **MiniMax H3 Community License Agreement**.

> **MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.**

By downloading, copying, modifying, distributing, or using these weights, you agree to comply with all terms and use restrictions set forth in the [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) and its Acceptable Use Policy (Exhibit A).

### 2. Geographic Restrictions (Excluded Territories)
> **IMPORTANT:** In strict accordance with Section V.4 and Definitions 3 & 5 of the MiniMax H3 Community License Agreement, you **may not use, reproduce, modify, distribute, or display the MiniMax H3 Works or any of their Outputs or results outside the Applicable Territory** (i.e., within the **Excluded Territories**), specifically:
> * **The United States of America (US)**
> * **The European Union (EU)**
> * **The United Kingdom (UK)**
> * **The Republic of Korea (South Korea)**
>
> Any use within the Excluded Territories is not authorized by the Community License. Users or organizations located in or operating services within these regions must obtain separate written authorization or commercial waivers directly from MiniMax at **[https://platform.minimax.io/h3-license](https://platform.minimax.io/h3-license)** (or via **[https://platform.minimax.io/console/license](https://platform.minimax.io/console/license)**).

### 3. Commercial Caps, Attribution & Prohibited Uses
- **Annual Revenue Limit ($20M USD Ceiling):** Commercial use within the Applicable Territory is permitted under the Community License provided the commercial products or services offered by the licensee do not generate annual gross revenue exceeding **US$20,000,000 ($20M USD)**. Entities exceeding this threshold require separate written commercial authorization from MiniMax.
- **Hosted Services & UI Attribution:** Distributing this model as a managed API or third-party service requires compliance safeguards and binding downstream users to terms at least as protective as the MiniMax H3 Community License Agreement. Any commercial product or service using these weights must prominently display **"MiniMax H3"** in its user interface and documentation.
- **Acceptable Use Policy & AI Labelling:** Users must comply with the MiniMax H3 Acceptable Use Policy (Exhibit A) and clearly disclose that generated audio-video outputs are machine-generated.
- **No Competing Training:** Neither these weights nor any Outputs generated by them may be used to train, distill, or improve any competing foundational AI models (other than MiniMax-H3 derivatives).

### 4. Upstream Model Derivatives & Apache 2.0 Notice of Modifications
This repository merges weights from two license categories:

1. **MiniMax H3 Community License Derivatives:**
   - **`TaoLiveAIGC/TaoMate-H3`** — Copyright © 2026 Alibaba TaoLive AIGC Team (Qijun Gan et al.). Licensed under the MiniMax H3 Community License Agreement.
   - **`ZhengmingYu/DMAD` (`minimax_h3/`)** — Copyright © 2026 Zhengming Yu, Junkun Yuan, Haotian Yang, Gordon Guocheng Qian, Chongyang Ma et al. Distributed as Model Derivatives of MiniMax H3 under the MiniMax H3 Community License Agreement.
2. **Apache License 2.0 Derivative Component (Section 4 Notice):**
   - **`larryvrh/MiniMax-H3-Turbo-Lora`** — Copyright © 2026 `larryvrh`. Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Statement of Modifications:** In compliance with Section 4(b) of the Apache License 2.0 and the MiniMax H3 Community License Agreement, notice is hereby given that the original LoRA weights from `larryvrh/MiniMax-H3-Turbo-Lora`, `TaoLiveAIGC/TaoMate-H3`, and `ZhengmingYu/DMAD` (`minimax_h3/`) were modified by mathematical layer re-weighting, key standardization, and linear interpolation into two new composite `.safetensors` adapter files (`turbo_lora_001.safetensors` and `turbo_lora_002.safetensors`).

### 5. Disclaimer of Warranty
THESE WEIGHTS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. The author of this merge is an independent developer and is not affiliated with or endorsed by MiniMax (Nanonoble Pte. Ltd.), Alibaba TaoLive AIGC Team, `larryvrh`, or the DMAD research authors.

---

## Citations & Acknowledgments

Credit goes to the original authors and research teams:

```bibtex
@misc{yu2026dmad,
  title={DMAD: Distribution Matching as Adversarial Distillation for Fast Visual Generation},
  author={Yu, Zhengming and Yuan, Junkun and Yang, Haotian and Qian, Gordon Guocheng and Wang, Yizhi and Wang, Angtian and Yang, Yiding and Liu, Bo and Li, Xin and Wang, Wenping and Ma, Chongyang},
  year={2026},
  eprint={2610.02188},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={[https://arxiv.org/abs/2610.02188](https://arxiv.org/abs/2610.02188)}
}

@misc{gan2026taomate,
  title={TaoMate: Anchor-Guided Memory Bridging Evolving and Reference States for Real-Time Audio-Video Digital Human Generation},
  author={Gan, Qijun and Zhang, Chenwei and Jin, Meiguang and Ma, Junfeng and Shen, Qiu},
  year={2026},
  eprint={2607.24359},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={[https://huggingface.co/TaoLiveAIGC/TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3)}
}

@misc{larryvrh2026turbo,
  title={MiniMax-H3-Turbo-Lora},
  author={larryvrh},
  year={2026},
  url={[https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)}
}

@misc{minimax2026h3,
  title={MiniMax-H3 Multimodal Generative Model},
  author={{MiniMax AI}},
  year={2026},
  url={[https://huggingface.co/MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)}
}
