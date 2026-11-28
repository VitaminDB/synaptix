# SOLRICKS/3D-Animation-Style-MiniMax-H3

Источник: https://huggingface.co/SOLRICKS/3D-Animation-Style-MiniMax-H3 · скачано 2026-11-28
Загрузок: 127 · лайков: 58 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-11-26T21:39:36.000Z

---

---
extra_gated_eu_disallowed: true
extra_gated_fields:
  Country: country
  I agree to the MiniMax H3 Community License Agreement and Acceptable Use Policy: checkbox
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
  - minimax-h3
  - lora
  - 3d-animation
  - text-to-video
  - image-to-video
widget:
  - text: "Girl with a telescope at dusk"
    output:
      url: samples/20260926-210654-933_3670611240.mp4
  - text: "Raccoon and hedgehog in a garden"
    output:
      url: samples/20260926-213610-328_502243029.mp4
  - text: "Otter floating with an apple"
    output:
      url: samples/20260926-222120-863_3795468811.mp4
  - text: "A woman holding a cup in a cafe"
    output:
      url: samples/20260927-175329-712_377046177.mp4
---


# 3D Animation Style LoRA for MiniMax H3


**Trigger:** 3dsrx

Prepared for T2V using the previously created 3d_animation dataset. The learned 3D animation style also carries over to I2V workflows.

**Recommended Strength**<br>
Text-to-Video: 0.8<br>
Image-to-Video: 0.4

For text-to-video generations featuring human subjects, a LoRA strength of 0.8 may help maintain prompt adherence and character consistency.<br>
For animals, fantasy characters, and environments, try a strength of 1.0 for a more pronounced 3D animation style.

<Gallery />

## License

This LoRA is distributed under the **MiniMax H3 Community License Agreement**. See [LICENSE](https://huggingface.co/SOLRICKS/3D-Animation-Style-MiniMax-H3/blob/main/LICENSE) for the complete agreement and Acceptable Use Policy, and [NOTICE](https://huggingface.co/SOLRICKS/3D-Animation-Style-MiniMax-H3/blob/main/NOTICE) for the attribution and modification notice.
