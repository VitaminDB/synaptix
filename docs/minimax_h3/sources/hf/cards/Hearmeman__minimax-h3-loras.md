# Hearmeman/minimax-h3-loras

Источник: https://huggingface.co/Hearmeman/minimax-h3-loras · скачано 2026-11-28
Загрузок: 78677 · лайков: 157 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora,comfyui · обновлено: 2026-11-24T06:09:43.000Z

---

---
license: other
license_name: minimax-h3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3
base_model: MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
tags:
  - not-for-all-audiences
  - lora
  - minimax-h3
  - text-to-video
  - image-to-video
  - comfyui
  - nsfw
---

# MiniMax-H3 NSFW LoRAs — HearmemanAI

**Free ComfyUI workflows, Runpod templates, and guides:**  
👉 [hearmemanai.com](https://hearmemanai.com/)

## For more LoRAs and updates, Join my Discord:
👉 https://discord.gg/ZVWVhT43GW
## Deploy my Minimax Runpod template on Runpod
👉 https://get.runpod.io/minimax-template

A mirror of the MiniMax-H3 LoRAs published on
[CivitAI](https://civitai.com/user/HearmemanAI). CivitAI-matched files have identical
SHA-256 checksums (listed below). CivitAI carries the sample videos, the version
history and the comment threads; this repository exists so the weights can be pulled with a
plain `resolve` URL, without an account.

One **anatomy** adapter teaches H3 what a body part looks like. One **action** adapter teaches
it what a body does. They are single-concept and they stack, so put them under whatever
character or scene LoRA you are already running.

LoRA names link to their CivitAI pages. Where a file checksum matches a published version,
the link points to that exact version.

## The adapters

### Recent uploads

| LoRA / version | CivitAI match | File |
| :--- | :--- | :--- |
| [**HMCumshot v1.0**](https://civitai.com/models/2857340?modelVersionId=3329529) | SHA-256 matches CivitAI v1.0 | [`HMCumshot_V1.0.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMCumshot_V1.0.safetensors) (310 MB) |
| [**HMMasturbation v2.0**](https://civitai.com/models/2893742?modelVersionId=3310312) | SHA-256 matches CivitAI v2.0; same model listing as HMSquirt v0.1 below | [`hmmasturbation_v2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/hmmasturbation_v2.safetensors) (310 MB) |

**HMMisDog v1.0** is listed on [CivitAI](https://civitai.com/models/2924146?modelVersionId=3308736). Its weights are not mirrored in this repository.

The older releases below remain available. Their version-specific notes do not describe the recent uploads.

### Action adapters — video-trained

| LoRA | Trigger | Strength | File |
| :--- | :--- | :--- | :--- |
| [**HMNSFW V3** (AIO Sex, Extreme Realism)](https://civitai.com/models/2834417?modelVersionId=3446189) — newest | — (plain prose, no trigger word) | **0.6 – 1.0** | [`HMNSFW-Extreme-Realism.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMNSFW-Extreme-Realism.safetensors) (621 MB) |
| [**HMNSFW V2.5** (AIO Sex)](https://civitai.com/models/2834417?modelVersionId=3268303) | `hmmotion` | **0.5 – 0.9** | [`HMNSFW-AIO-V2.5.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMNSFW-AIO-V2.5.safetensors) (86 MB) |
| [**HMNSFW V2** (AIO Sex)](https://civitai.com/models/2834417?modelVersionId=3206518) | `hmmotion` | **0.5 or below** | [`HMNSFW_AIO_V2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMNSFW_AIO_V2.safetensors) (310 MB) |
| [**HMCumshot V0.5**](https://civitai.com/models/2857340?modelVersionId=3238531) | `cumshot` | 0.9 | [`HMCumshot_V2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMCumshot_V2.safetensors) (626 MB) |
| [**HMSquirt v0.1**](https://civitai.com/models/2893742?modelVersionId=3271912) | `hmsquirt` | — (see settings below) | [`HMSquirtV0.1.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMSquirtV0.1.safetensors) (86 MB) |

### Anatomy adapters — stills-trained

| LoRA | Trigger | Strength | File |
| :--- | :--- | :--- | :--- |
| [**HMPussy V2**](https://civitai.com/models/2846342?modelVersionId=3436891) — newest | `Pussy`, `vagina`, `anus` | — | [`HMPussyV2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMPussyV2.safetensors) (78 MB) |
| [**HMPussy V1** (stills only)](https://civitai.com/models/2846342?modelVersionId=3252213) | `pussy` | 1.0 | [`Vagina_minimax-h3_epoch20.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/Vagina_minimax-h3_epoch20.safetensors) (78 MB) |
| [**HMPussy v0.5**](https://civitai.com/models/2846342?modelVersionId=3215304) (2 files, load both) | `Vagina` + `hmpussy` | 1.0 / 0.35 | [`vagassist_e40.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/vagassist_e40.safetensors) (310 MB) · [`hmpussy_v6_epoch30.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/hmpussy_v6_epoch30.safetensors) (626 MB) |
| [**HMInnie v1.0**](https://civitai.com/models/2853405?modelVersionId=3222484) | `inniepussy` | — | [`HMInnie_v1_e50.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMInnie_v1_e50.safetensors) (310 MB) |
| [**HMBreasts V2**](https://civitai.com/models/2848799?modelVersionId=3268969) — newest | `tits`, written in plain prose | — | [`HMBreastsV2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMBreastsV2.safetensors) (78 MB) |
| [**HMBreasts V1.0**](https://civitai.com/models/2848799?modelVersionId=3216751) | `HMBreasts` | 1.0 | [`HMBreasts_085e0750_e40.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMBreasts_085e0750_e40.safetensors) (310 MB) |
| [**HMPenis v1.0**](https://civitai.com/models/2849923?modelVersionId=3218160) | `HMPenis` | — | [`HMPenis_v2_e35.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMPenis_v2_e35.safetensors) (310 MB) |

Where a strength is blank, no single number was published — start at 1.0 and pull it back.

## How to prompt each one

### HMNSFW V3 — Extreme Realism, the current motion adapter
No trigger word; write simple prose prompts. Strength **0.6 – 1.0**. SHA-256 matches
[CivitAI V3 - Extreme Realism](https://civitai.com/models/2834417?modelVersionId=3446189).
One file, [`HMNSFW-Extreme-Realism.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMNSFW-Extreme-Realism.safetensors).

This takes simple prose prompts and turns them into realistic sex scenes with audio and motion.

Recommended settings, from the CivitAI version notes: the fl2va int8 convrot model, sampler
**Euler**, scheduler **Beta**, **18 steps**, without Turbo LoRA. Turbo LoRA also works.

Known limitations observed in testing: it can miss the anus, stutter at high strengths, or get
grainy. If that happens, change the seed and try again.

### HMNSFW V2.5 — the previous motion adapter
Trigger `hmmotion`. Strength **0.5 – 0.9**. Trained on close to 1000 videos, roughly 15x the V2
dataset, over 40 hours of training. I2V is very good here, with coherent anatomy and motion, and
it handles most T2V prompts with the occasional bad seed.

Your prompt is everything. A short or vague prompt gives you a bad result. Use the example prompts
on the CivitAI page as anchors and match their length and register.

### HMNSFW V2 — an earlier motion adapter
Trigger `hmmotion`. Strength **0.5 or below**; above that it collapses coherence. Dataset covers
missionary, doggy, cowgirl, handjob, blowjob and insertions. I2V works well across most positions,
with the occasional deformed genitalia; T2V is hit or miss, which is what the anatomy adapters
below are for.

Long descriptive prompts beat short ones by a wide margin. The training captions run 165-269
words with a median of 225, so a keyword list is off-distribution for this checkpoint. Write
one flowing paragraph of plain anatomical prose, not tags. The full prompt-writing system
prompt is on the CivitAI page.

Generated with the full bf16 model, trained on bf16. `dpmpp_2m` + Beta scheduler, 20 steps.

### HMCumshot V0.5 — earlier release
Trigger `cumshot`. Strength 0.9. Captioned in plain, blunt language, so prompt the same way.
A thorough description of the cum texture helps a lot. Stacks on top of HMPussy and HMPenis.

* **T2V / I2V** — Turbo LoRA 8-step at 0.20, Euler, `ddim_uniform`.
* **R2V** — Turbo LoRA 4-step at 0.85, Euler, `beta`.

Optionally add the `ExtendIntermediateSteps` node with 2 extra steps starting at sigma 0.85.
V0.5 expanded the dataset to 56 videos with more angles.

### HMSquirt — squirting
Trigger `hmsquirt`. Experimental, trained on a small dataset, and it does work. It is not the
best thing here and a better version is in progress. Run HMPussy alongside it at **0.4**.

Recommended settings:

* Sampler **Euler**, scheduler **Simple**, **12 steps**, **shift 6**.
* Turbo LoRA at strength **0.5**.

### HMPussy V2 — current version
Trained words `Pussy`, `vagina`, and `anus`. SHA-256 matches [CivitAI V2](https://civitai.com/models/2846342?modelVersionId=3436891). One file, [`HMPussyV2.safetensors`](https://huggingface.co/Hearmeman/minimax-h3-loras/resolve/main/HMPussyV2.safetensors). HMPussy V1 remains the previous stills file.

### HMPussy V1 — stills only, previous version
Trigger `pussy`. Strength 1.0. Rebuilt from the ground up on 4x the dataset, covering innies,
bushes, shaved, anuses and most everything else between a woman's legs. One file, smaller than the
old pair, and no video partner needed unless you want the motion adapter under it.

### HMPussy v0.5 — two files, both required
`vagassist_e40` is stills-trained: run it at **1.0**. It restores legible structure in female
genital anatomy, which the base model renders soft and vague. `hmpussy_v6_epoch30` is
video-trained: run it at **0.35**, sitting under the stills file rather than carrying a
generation alone. Skip the video file if you do not care about motion.

Trained on fingering, spreading and plain show-off. LoKr adapters on the MiniMax-H3 fl2va
bf16 base. V0.5 improved anuses considerably; they are still not all the way there.

### HMInnie — shape control
Trigger `inniepussy`, and it **replaces** the word "pussy" in your prompt rather than sitting
in front of it. Every training caption opens with a camera clause, so include one:
`inniepussy shown from the front` / `from behind` / `from below`.

Four axes, written as plain English inside the sentence, not as tags:

* **Mound** — flat · soft · puffy · very puffy
* **Cleft** — closing to a smooth seam · with a defined cleft · parted around a visible opening
* **Inner lips** — tucked away · just visible · slightly protruding
* **Labia colour** — matching skin · pink · rosy · brown

Sits alongside HMPussy for general genital detail, and under HMNSFW when the anatomy needs to
survive motion. Trained with AI-Toolkit on 117 images at rank 32 on the full bf16 base.

### HMBreasts V2 — chest control, current version
Versatile across most sizes and shapes, and the prompts can stay short — write the chest in plain
prose the way the CivitAI examples do, no leading trigger token. Use the `seeds2` sampler with the
`ddim_uniform` scheduler and the 8-step Turbo LoRA.

The four axes from V1 still apply and still read as plain English inside the sentence:

```
tiny tits with brown areoles
tiny sized tits and pale areoles
```

### HMBreasts V1.0 — chest control, previous version
Trigger `HMBreasts`, leading. Strength 1.0. Base H3 renders a chest as an approximate shape and
ignores anything you say about it; this makes size, areole size and areole colour things you type.
Four axes, again as plain English:

* **Breast size** — tiny · small · medium · large · very large
* **Shape** — natural · perky · round · perfectly shaped · saggy · sagging · hanging · pressed together
* **Areole size** — tiny · small · medium · large · very large
* **Areole colour** — pale · ghost · brown · dark
* **Nipples** — erect · hard · pierced

Spelling matters: it learnt **areoles**, so write it that way. Example clauses:

```
HMBreasts, large, natural breasts, medium sized brown areoles and erect nipples
HMBreasts, small, perky breasts, tiny sized pale areoles and hard nipples
HMBreasts, very large, round breasts, large sized ghost areoles and erect nipples
```

The axes were trained independently, so tiny breasts with large dark areoles is a combination
it can actually give you.

**Honest limitation:** `medium` is the weakest of the five sizes. The training material
clustered at both ends, so tiny/small and large/very large steer hard while the middle is
softer than either. A shape word usually pulls it back.

It owns the chest and nothing below it — genitals are HMPussy, and the two stack. It does not
touch faces, bodies or identity. Trained on stills, so it teaches H3 what a chest looks like;
how it behaves through motion still comes from the base model.

### HMPenis — anatomy
Trigger `HMPenis`, leading. Works best with medium-to-large sizes. Specify the camera
direction in the prompt: front (POV-like), back, or side. Useful adjectives: large,
circumcised, glans with a colour adjective (pink / pale / brown). The dataset is thin.

## Stacking

Anatomy adapters (HMPussy, HMInnie, HMBreasts, HMPenis) are stills-trained and single-concept.
HMPussy V2 is the current file. HMPussy V1 remains the previous stills file; `hmpussy_v6_epoch30` from v0.5 is the video partner if you want it.
Action adapters (HMNSFW, HMCumshot, HMSquirt) are video-trained. They compose: put the anatomy
adapters underneath, the action adapter on top, and your character or scene LoRA above both.

Pick one HMNSFW, one HMBreasts and one HMPussy file. The versions of each are the same line, not stackable
partners.

## Verification

The SHA-256 values below identify the files hosted here. The HMNSFW V3,
HMCumshot v1.0, HMMasturbation v2.0, and HMPussy V2 hashes match their CivitAI file metadata.

| File | SHA-256 |
| :--- | :--- |
| `HMCumshot_V1.0.safetensors` | `634c39cfcbfd9421a2d7b5adc62573fc232384c4c75e003c2e11d3408ad0765c` |
| `hmmasturbation_v2.safetensors` | `f0e4bfbe5baebe972880d2250a227e2f475aa8c18d6d01ed897288d9741659d2` |
| `HMNSFW-Extreme-Realism.safetensors` | `55aa39239ea4b408dee601cb0fa49157ec13fe2a54eccbb2fb725b124e43c1b5` |
| `HMNSFW-AIO-V2.5.safetensors` | `a07732a84fd733085eb5d910f602f918fa7a3658117116927e4329f5951a9d2d` |
| `HMNSFW_AIO_V2.safetensors` | `608e4212f2788b6063330ff1196fc1f4b4228cfd9a413a63c198a09d7e4a61cb` |
| `HMCumshot_V2.safetensors` | `1a5b7948bb97f27737e62c3dd5497a3afb77517f230787f45e45c7d8fe3dc24d` |
| `HMSquirtV0.1.safetensors` | `b25cc54db8e12012e9c1a60c514d767208c855cbf6e81605827d86913249cf24` |
| `HMPussyV2.safetensors` | `7b9559077d8ac623d52b64b438d8ad60cb85b31b6fecf011aab238b70962a2ed` |
| `Vagina_minimax-h3_epoch20.safetensors` | `373c3cad3bf27047fdd754fe111443d97e70e3108a8829f2ec63c48832466eb3` |
| `vagassist_e40.safetensors` | `2c2fdb66bf558de1aabda504a81d4ada5f4cebc20e8f519dc6ed3bb6d4be8c9a` |
| `hmpussy_v6_epoch30.safetensors` | `3080f4fbcbba4fc06bd09240c7eedb6a5128eb0e19feb001cdf97a7a0941a6ee` |
| `HMInnie_v1_e50.safetensors` | `499196c9d0e5f81ff575ba39a82987112c3bb1e09fbede858877cd950d6c8833` |
| `HMBreastsV2.safetensors` | `d260653bdf10775380a44c8f1486bcc4690bd57085cb034002a8fb878f1ad36f` |
| `HMBreasts_085e0750_e40.safetensors` | `039b6d5399def81c9a459d7cca8ccf749195fcb5f766f0899a387ba2fa6ad967` |
| `HMPenis_v2_e35.safetensors` | `c6c58e9fee848b45e99f97d2520aba4ac63dfc354c07e13c29ac5d8a31a68060` |

## Running MiniMax-H3

These load with a stock `LoraLoaderModelOnly` in ComfyUI. For wiring references into
Reference-to-Video without rebuilding the graph every time, see
[ComfyUI-MiniMaxRefPack](https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack).

## Licence

The adapter weights are released as-is. Use of the MiniMax-H3 base model is governed by the
[MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3). You are
responsible for what you generate and for complying with the law where you are.
