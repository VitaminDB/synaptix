# ausboss/MiniMax-H3-Image-Outpaint-LoRA

Источник: https://huggingface.co/ausboss/MiniMax-H3-Image-Outpaint-LoRA · скачано 2026-11-28
Загрузок: 784 · лайков: 19 · базовая: MiniMaxAI/MiniMax-H3-Image · теги: lora,comfyui · обновлено: 2026-11-26T01:22:14.000Z

---

---
license: other
license_name: minimax-h3-image-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE
base_model: MiniMaxAI/MiniMax-H3-Image
pipeline_tag: image-to-image
tags:
 - lora
 - outpainting
 - uncrop
 - minimax-h3
 - minimax-h3-image
 - comfyui
 - ai-toolkit
---

# MiniMax H3 Image Outpaint LoRA

Extends a picture on **any side**, and paints the gray corners of a picture you have turned. Pad the picture with
flat gray `#808080`, give the padded canvas to MiniMax H3 Image as Picture 1, and the LoRA fills the gray with more of
the scene while raising the odds that your picture stays exactly where it is.

**Ready-made ComfyUI workflow:** in this repo as
[`workflows/minimax-h3-image-rotate-outpaint.json`](workflows/minimax-h3-image-rotate-outpaint.json).
Also on Civitai: [the LoRA page](https://civitai.com/models/3064644) and [the workflow page](https://civitai.com/models/3064700).
Videos on [YouTube @ausbossdev](https://www.youtube.com/@ausbossdev). Help and more workflows on
[Discord](https://discord.gg/RwVxY9Fuaj).

Covered in: <https://youtu.be/GTjyJNgIG6Q>

[![Watch on YouTube: I put 6 characters in one 16 megapixel picture, MiniMax H3 Image in ComfyUI](video_h3_image.jpg)](https://youtu.be/GTjyJNgIG6Q)

![The forest canvas, then H3 Image alone and with the LoRA. A dashed line marks where the picture sat on the canvas. Without the LoRA the forest is shrunk and moved 133 px off the line; with it the forest stays on the line and only the gray is painted](figure_compare_forest.gif)

*Rendered with the same canvas, prompt and seed in both columns, raw sampler output before the paste-back. Only the
LoRA is switched. The dashed line is where the picture sat on the gray canvas. Without the LoRA, H3 Image shrank the
forest and moved it 133 px, so pasting the original back would land next to a copy of itself.*

> **Use it at strength 1.0, at about 4 MP (short side 2048 px or less), with gray `#808080` and the prompt below.** Above about 4 MP H3 shrinks its copy of your picture and it starts to drift again. Outpaint at 4 MP and upscale the result after.

Without the LoRA, H3 Image redraws the whole canvas and often shrinks or slides your picture instead of extending it.
On 16 photos it was not trained on, it moved the picture more than 2 px on 6. With the LoRA: 0 of 16. On 12 of my own
renders: 4 of 12 without, 0 of 12 with. It raises the odds a lot; it doesn't make a miss impossible, and the workflow
pastes your original back on top either way.

More results with the LoRA:

![A white car in a wet underground garage, widened on both sides. The gray sides fill with more of the garage, its lights and pillars](figure_car_chase.gif)

![A fighter jet in a long hangar, padded on all four sides. The hangar's roof, windows and floor continue](figure_hangar.gif)

![Tall trees seen from below, turned and padded. The gray corners fill with more trunks and sky](figure_forest_turn.gif)

![A concrete passage with a garden, extended at a corner. The walls and plants continue](figure_garden_corner.gif)

*My own renders, made with the LoRA at 1.0: INT8 model, 6 steps, LCM, CFG 1, about 11 to 13 seconds each on an RTX 3090
once loaded. Each wipes from the canvas the model is given to the result. None of these pictures is in the training set.*

## Prompt

The LoRA was trained on one prompt form. Put it first, then describe the picture after `Scene:`:

```
Outpaint <Picture 1>: replace the solid gray areas with a seamless continuation of the scene, keeping the existing picture unchanged. Scene: <what is in your picture>
```

Describe what you can already see: the subject, the place, the light, the style. Anything you name that isn't in the
picture tends to get painted into the new area. In my tests a description that mentioned a bird gave every version a
bird in the fill, plain H3 too. That also works on purpose: name a thing if you want it out there.

## Files

| file | notes |
|---|---|
| `minimax-h3-image-outpaint.safetensors` | The LoRA. Step 1,500 of a 2,000-step run, rank 16, 155 MB. Loads onto the MiniMax H3 Image files in ComfyUI, model only. |
| `minimax-h3-image-outpaint-2000.safetensors` | Step 2,000. The cleanest seams of the three steps I tested, but on one test picture it moved and repainted the picture (see Which step). |
| `workflows/minimax-h3-image-rotate-outpaint.json` | The ComfyUI workflow: turn, crop and pad the picture, paint at 4 MP, paste the original back, and an optional 2x upscale (off by default). |

## ComfyUI

You need a ComfyUI with MiniMax H3 Image and [AusBoss nodes](https://github.com/ausboss/ComfyUI-AusBoss) 2.21.2 or newer. In
ComfyUI-Manager, search "AusBoss". The model files are on Hugging Face:
[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) and
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). The workflow uses
`minimax_h3_image_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_int8_convrot` and `minimax_h3_image_vae_fp16`.

To build it yourself:

1. Pad the picture with flat gray `#808080` (AusBoss **Image Crop + Rotate + Pad**: fill `#808080`, resize to 4 MP, steps 32). To straighten a crooked picture, turn it with the same node and let it paint the corners.
2. **MiniMax H3 Image Conditioning**: the padded canvas as `image_1`, the prompt above, width and height from the canvas.
3. **SamplerCustomAdvanced** on the latent it returns: **BasicScheduler** `simple` 6 steps, **SamplerLCM** (s_noise 1.1 → 0.9), **CFGGuider** at CFG 1 with the conditioning as both sides. The LoRA goes on the model at 1.0.
4. **VAE Decode**, then AusBoss **Stitch Inpaint** with the stitcher from the padding node: seam `blend in`, Tone match 0. It puts your original pixels back on top.

## What I measured

16 photos that are not in the training set, padded and turned at about 4 MP. I cropped each from a full photo, so the
real continuation is known. All numbers are the raw sampler output, before the paste-back. 6 steps, LCM, CFG 1, seed 42.

| | kept area PSNR | moved more than 2 px | fill error | seam | gray left |
|---|---|---|---|---|---|
| H3 Image alone | 23.3 dB | 6 of 16 | 32.5 | 6.0 | 5 of 16 |
| step 1,000 | 30.5 dB | 0 of 16 | 19.2 | 3.0 | 1 of 16 |
| **step 1,500 (the file)** | **30.9 dB** | **0 of 16** | **19.3** | 3.4 | 1 of 16 |
| step 2,000 | 32.0 dB | 1 of 16 | 20.5 | 2.2 | 1 of 16 |

"Kept area" is how close the part that was picture comes back to it. "Fill error" is the mean difference between the
new area and the real photo, out of 255. "Seam" is the tone step at the border, out of 255. Lower is better for both.

Then 12 canvases from my own renders, LoRA against no LoRA, the same seed each. Without the LoRA the picture moved on 4:
133 px (the forest above), 8 px, 4 px and 2 px. With it, 0 on all 12. I rated them blind: I loved 11 of the 12 with the
LoRA and 9 without, and marked one without the LoRA wrong.

**Colour.** H3 Image redraws the whole canvas, so even the part you kept comes back a little different. I measured the
mean colour change in Lab (L lightness, a green to red, b blue to yellow):

| | ΔL | Δa | Δb |
|---|---|---|---|
| H3 Image alone, kept area | +1.3 | +1.7 | −0.5 |
| with the LoRA, kept area | +0.0 | −0.1 | +0.6 |
| with the LoRA, new area against the real photo | +1.0 | −0.3 | +0.8 |

So the LoRA shifts colour less than plain H3, but not zero: it leans a little warm or yellow on almost every picture,
and now and then one picture gets a stronger tint (the worst on my renders: a +7.9 against +19.1 without the LoRA).
Pasting the original back removes the shift from the part you kept. The new area keeps the small warm lean. The
workflow leaves Tone match at 0, because after the canvas is resized it can tint the corners of a turned picture.

**Size.** It holds the picture in place up to a short side of about 2048 px. Above that, H3 shrinks its copy of your
picture to a 2048 px short side, and on 12 MP canvases the kept part drifted again (13 px on a 4224 × 2816 canvas,
6 px without the LoRA) though the fill still looked better. That is why the workflow works at 4 MP.

**Which step.** I tested steps 1,000, 1,500 and 2,000 on the 16 photos. Step 2,000 has the cleanest seams, but on one
corner extension it moved the picture 100 px and repainted it. Steps 1,000 and 1,500 did not. Step 1,500 is the file
here. At strength 0.5 every result landed between plain H3 and 1.0, so the workflow uses 1.0.

## Limits

- It works at about 4 MP. Bigger canvases drift; upscale the result instead.
- It raises the odds your picture stays put, it doesn't promise it. Run it again with another seed if something moved.
- The new area can come out a little warmer than your picture (see Colour).
- On strongly turned canvases a thin gray sliver can stay in a corner: 1 of the 16 test photos. Run it again with another seed.
- It was trained on turns of 3 to 25° and moderate pads. Bigger extensions, text and logos in the new area, and faces cut by the edge are untested.
- It does not make the picture bigger or sharper.

## How it was made

- [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit), `arch: minimax_h3_image`, on MiniMax H3 Image with the DiT quantized to INT8 and the text encoder to NVFP4, with Ostris's training adapter. Rank 16, alpha 16, AdamW 8-bit, learning rate 1e-4 constant, batch 1, 2,000 steps with a checkpoint every 250, resolution 2048 buckets, caption dropout 0, `shift` timesteps. About 5 hours on one RTX PRO 6000.
- 412 pairs at 3 to 4.2 MP from 206 Unsplash photos of 12 MP or more. The target is the real photo and the source is the same photo with the area to paint filled with `#808080`. One side, opposite sides, a corner or a frame, and 154 of them turned 3 to 25°. Each caption describes the photo after `Scene:`.
- 16 more photos were held out for the tests above.

## License

The LoRA is a derivative of MiniMax H3 Image, so it falls under the
[MiniMax H3 Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE). Read
it before commercial use. The training photos came from Unsplash under the Unsplash License.
