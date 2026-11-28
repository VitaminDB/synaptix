# nobodyatalliguess/minimax-h3-loras

Источник: https://huggingface.co/nobodyatalliguess/minimax-h3-loras · скачано 2026-11-28
Загрузок: 554 · лайков: 1 · базовая:  · теги: lora · обновлено: 2026-10-09T00:03:14.000Z

---

---
license: other
tags:
- not-for-all-audiences
- lora
- minimax-h3
---

# MiniMax H3 LoRAs, re-hosted for Sogni

These are third-party MiniMax H3 LoRAs, re-hosted here so Sogni's Personal LoRA import can pull them from a plain public `resolve` URL. **All credit goes to the original creators.** Their Civitai pages have the sample videos, version history, discussion, and the authoritative license and permissions. Please support the creators there.

Adult (18+) content. All subjects are adults.

| File | Original LoRA | Creator | Status | Trigger(s) | Recommended strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Male_Anatomy-v2.02-ep4_x1.5.safetensors` | [Male Anatomy – Ass, Penis, Testicles and Glans – H3](https://civitai.com/models/2839513) (v2.02) | **diogod** | **Modified: pre-scaled x1.5** (see below) | `penis`, `glans`, `testicles` (also `asshole`, `ass`, `anus gap`) | Sogni **0.6 – 1.0** (= original 0.9 – 1.5) |
| `MaleUndress_H3_v1.0.safetensors` | [Minimax-H3-T2V_I2V-MaleUndress](https://civitai.com/models/2938072) (v1.0) | **coldwood**. The creator requires credit. | Unmodified re-host | `MalUndress` | No strength published by the creator |
| `Darings_Deepthroat_H3_v0.2.safetensors` | [Daring's Deepthroat H3/LTX2.3](https://civitai.com/models/2476698) (MiniMax H3 FL2VA v0.2) | **daring_l**. The creator requires credit. | Unmodified re-host | No trigger word; use the prompt vocabulary below | No number published for H3; start at Sogni **1.0** (see below) |

## Male Anatomy x1.5: what was changed

Sogni caps imported (personal) LoRA strength at 1.0, but diogod recommends starting at 0.9 and says it works from 0.4 to 1.5. To make the upper range reachable on Sogni, this copy multiplies only the `lora_up` (B) tensors by exactly 1.5. `lora_down` and `alpha` are untouched, and key names, FP32 dtype and the original metadata are kept. That scales the LoRA's effect by exactly 1.5, so **Sogni strength S on this file = original strength 1.5 × S**:

- Sogni 0.6 = original 0.9 (the creator's suggested starting point)
- Sogni 1.0 = original 1.5 (the creator's stated upper limit)
- Sogni 0.27 ≈ original 0.4 (the creator's stated lower limit)

A metadata key `prescaled` records the change. Original file: `Male_Anatomy-v2.02-ep4.safetensors`, SHA-256 `e639c295694309801173a7e6818be970b5535411ac0821ad0ed11578beade276`, matching Civitai.

## Daring's Deepthroat v0.2: notes from the creator's Civitai page

- **Modes.** Trained in `fl2va` mode on the `minimax_h3_fl2va_bf16` checkpoint (first + last frame conditioning). MiniMax's model card says the FL2VA checkpoint also does text-to-video (no image) and first- or last-frame-to-video (one image). Sogni's H3 text-to-video, image-to-video and first/last-frame models all use this FL2VA checkpoint. The separate Ref2VA (reference-to-video) checkpoint is not what this LoRA was trained on.
- **What's new in v0.2 (creator):** rank 16, rewritten visual and audio captions, better audio, `maintain eye contact` works as a prompt phrase, better side profile, holds the action for a full 15 seconds, MMF/FFM supported, "Entering the Frame. Better supported.", more thrusting, better prompt adherence and a wide usable strength window.
- **Strength.** The creator gives no number for the H3 version, only that the usable strength window is wide. The file's metadata says the creator already baked in a strength of 0.7 (`bake_strength_scale = 0.7`, `bake_strength_target = up`, from the epoch-31 checkpoint). So 1.0 is the strength as the creator shipped it. Starting at Sogni 1.0 and going lower if needed is a suggestion, not the creator's instruction.
- **Settings (creator):** guidance scale 4; trained at 24 fps (other frame rates are out of distribution). Gallery samples were made at 0.4 MP. Seed changes results a lot, so reroll before rewriting the prompt. Audio comes out quiet (about −26 LUFS); normalise after generation.
- **Visual vocabulary (creator's caption coverage).** Terms bind reliably from about 15% coverage upward and fall apart below 10%.
  - Anatomy: use `shaft` (77%), `penis` (75%), `glans` (49%). Avoid `dick`, `cock`, `member`, `tip`. `scrotum` (0%) and `balls` (7%) are not trained, so avoid that region.
  - Fluids: `saliva`, `glistening` (51% each), `wet`, `smooth` (26%), `spit` (16%, weaker, e.g. "thick strands of spit").
  - Motion: `slides` (the core verb), `pulls`, `reappears` / `disappears` (naming both controls the in-and-out rhythm), `bob`, `thrust`.
  - Face and gaze: `lips`, `mouth`, `eyes`, `face`, `she maintains eye contact`.
  - Act words: `deepthroat` and `blowjob` are almost always on. They anchor the LoRA but steer little.
  - Shot: `static` / `steady camera` (handheld 11%).
- **Audio vocabulary.** Use these exact words, because synonyms were trained out: `sucking`, `squelches`, `gagging`, `moaning`, `breathing`. `gagging` is the weakest; repeat it or add intensity words. `gawk gawk` actively hurts (the text encoder reads it as staring), and `slurping` and "wet mouth sounds" are ignored.

## Notes

- Base model: MiniMax H3 (text-to-video, image-to-video, first/last frame).
- coldwood's and daring_l's files are byte-identical to their Civitai downloads. Their SHA-256 values are listed below and match the hashes Civitai publishes.
- License and usage terms are set by each creator on Civitai and are not changed by this re-host. If you are a creator and want a file removed, open a discussion on this repo.

## SHA-256

| File | SHA-256 |
| :--- | :--- |
| `Male_Anatomy-v2.02-ep4_x1.5.safetensors` | `98253d735ab530b69fe2a290c6e7beafef620febf968d55b5091060022fc975b` |
| `MaleUndress_H3_v1.0.safetensors` | `6367029b541a79246ef0eab1bb36f427ef05ff0ae67efa6cc19bd9f12d879b6a` |
| `Darings_Deepthroat_H3_v0.2.safetensors` | `1fd239662f6290255b0bb3a220764fb53aab2859378f7fd3024030c1e1991cb2` |
