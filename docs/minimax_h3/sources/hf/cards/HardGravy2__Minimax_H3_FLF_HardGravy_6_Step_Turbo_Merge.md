# HardGravy2/Minimax_H3_FLF_HardGravy_6_Step_Turbo_Merge

Источник: https://huggingface.co/HardGravy2/Minimax_H3_FLF_HardGravy_6_Step_Turbo_Merge · скачано 2026-11-28
Загрузок: 0 · лайков: 14 · базовая:  · теги: comfyui · обновлено: 2026-09-08T22:01:02.000Z

---

---
license: other
license_name: minimax-h3-cla
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
tags:
- comfyui
---

### Model Description

A smart merge of lightx2v & larryvrh's turbo loras plus JonXL's H3 photorealism lora. Apologies as I do not have a merge formula to give out for others to tweak and improve upon, a happy accident during testing pushed things in the right direction but unfortunately I no longer have the workflows needed to reproduce the merge. 

During development this lora & variations of it have been great for my use case, silly little clips with natural language prompts on relatively low-spec hardware (16GB RTX A4000 Ampere) with comfyUI. I've been getting ~330s per 10s of output @ 0.5mp using the SLA attention node.

I am time-poor and have not been able to test this extensively but the audio and video quality and prompt adherence seems highly improved over any other singular trained turbo lora. 

No support is offered or implied.

Tested with the following parameters
Sampler: Euler
Scheduler: Simple
Steps: 6
Shift: Not set

See h3_fast.json for the workflow used during testing.
