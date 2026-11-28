# QrusherZA/MiniMax-H3-Workflow

Источник: https://huggingface.co/QrusherZA/MiniMax-H3-Workflow · скачано 2026-11-28
Загрузок: 0 · лайков: 2 · базовая:  · теги:  · обновлено: 2026-08-13T07:29:17.000Z

---

This is just and experimental Workflow, what i have found is that the Turbo LoRA has some weird behaviour with 10 Eros Max, i have experimented with a WAN 2.2 (MoE) style kind of workflow

I have found it to do the following
* turbo seems to work nicely with 10 Eros Max now
* feels like there is a bit more clarity on fast motion than before (this could be a placebo)

This uses the Original MiniMax H3 model as the `"High"` pass for 2 steps then uses 10 Eros Max Beta1 as the `"Low Noise"`, this has not been tested on high resolution for me, but with the long 640 resolution i am limited to i was happy with the results for myself
