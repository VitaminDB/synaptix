# Source: https://civitai.com/models/2918749

API: https://civitai.com/api/v1/models/2918749

Fetched: 2026-11-28 (Civitai public API, description converted to text)

---

## Grimdark 3D - MiniMax H3 Style LoRA
Type: LORA  Creator: hoodtronik  Tags: style
Stats: {"downloadCount": 474, "thumbsUpCount": 21, "thumbsDownCount": 0, "commentCount": 0, "tippedAmountCount": 0}

### Description

.. A STYLE LoRA for MiniMax H3 video - it changes the look, not the subject.

EXPERIMENTAL: first-run conversion, lightly tested - the sample clips on this page are the only test so far. Strength: start at 1.0, but this is a drawn / animated / illustration style and H3's base pulls toward photorealism, so it usually needs to be turned UP (1.3 to 2.0) to fully take. Epoch 35 is the softer file, epoch 50 the stronger. Tell me what you find.

Trigger word: grimdark3dstyle (put it first in the prompt). Strength 1.0. Two files: epoch 35 and epoch 50 - try 35 first, 50 for a smoother finish.

Base: MiniMax H3 (fl2va). Loads in ComfyUI with LoraLoaderModelOnly (the file must sit in models/loras/).

How it was made: 150-image Krea 2 dataset generated with the source style LoRA, every subject unique, 50% people, cinematic shot vocabulary in every caption; trained with Fizgig "MiniMax H3 Style (LoRA 8)" - rank 8, LR 2e-4, 50 epochs, blocks 0-3,6-47, adamw, shift 0.6667.

Best results: give H3 a pencil-sketch storyboard plus character references and let the LoRA fill in the look. With a photoreal start frame it mostly reinforces what is already there.

Samples: three pencil-sketch storyboards written for THIS style (wide, two-shot, close-up), rendered by H3 with two characters from this LoRA's own dataset as the only references - so what you see is the LoRA filling in the world. H3 renders at 480p, 6 s, seed 77, audio stripped.

Credit: derived from "Grimdark 3D" by TiwazM (Krea 2) - https://civitai.com/models/2892283/grimdark-3d?modelVersionId=3278433 - please support the original creator.

### Version: ep50 (base: MiniMax H3, created 2026-09-06)
Trigger words: grimdark3dstyle
Files: grimdark3d_style_h3_ep50.safetensors (68.3 MB)

### Version: ep35 (base: MiniMax H3, created 2026-09-06)
Trigger words: grimdark3dstyle
Files: grimdark3d_style_h3_ep35.safetensors (68.3 MB)
