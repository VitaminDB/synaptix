# aptech0081/MiniMax-H3-PDD-ComfyUI

Источник: https://huggingface.co/aptech0081/MiniMax-H3-PDD-ComfyUI · скачано 2026-11-28
Загрузок: 16004 · лайков: 88 · базовая: ["alibaba-pai/MiniMax-H3-Acc-LoRAs","MiniMaxAI/MiniMax-H3"] · теги: comfyui,distillation · обновлено: 2026-10-20T04:58:44.000Z

---

---
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
base_model:
- alibaba-pai/MiniMax-H3-Acc-LoRAs
- MiniMaxAI/MiniMax-H3
pipeline_tag: text-to-video
tags:
- comfyui
- minimax-h3
- video
- audio
- distillation
- pdd
---

# MiniMax H3 PDD — ComfyUI LoRAs and baked Ref2VA checkpoints

ComfyUI conversions of the official
[Alibaba PAI MiniMax-H3 PDD acceleration weights](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
The Ref2VA checkpoints generate video and audio from reference conditioning.

**Choose PRUNED INT8 WITH HEADS for the smaller native PDD checkpoint.**
This repository also keeps the full INT8 **WITH HEADS** and **WITHOUT HEADS** variants.
Both with-heads files contain the merged PDD trunk and 32 video plus 32 audio output heads.
They load through native ComfyUI after
[PR #17157](https://github.com/Comfy-Org/ComfyUI/pull/17157), merged October 18, 2026,
without a separate PDD acceleration LoRA or PDD Apply node.

## Files

| File | Contents | Runtime |
|---|---|---|
| [`minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors`](minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors) | **PRUNED INT8 WITH HEADS:** smaller Ref2VA trunk, PDD baked at strength 1.0, 32 video and 32 audio heads; **21.06 GB** | Native ComfyUI with PR #17157 |
| [`minimax_h3_ref2va_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors`](minimax_h3_ref2va_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors) | **WITH HEADS:** Ref2VA INT8 ConvRot trunk with PDD merged at strength 1.0, plus 32 video and 32 audio heads | Native ComfyUI with PR #17157 |
| [`minimax_h3_ref2va_pdd_acc_8step_baked_without_heads_int8_convrot.safetensors`](minimax_h3_ref2va_pdd_acc_8step_baked_without_heads_int8_convrot.safetensors) | **WITHOUT HEADS:** PDD trunk merged at strength 1.0; original single output heads | External Ref2VA PDD head bank and Apply node still required |
| `minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors` | FL2VA trunk LoRA and external head bank | Companion PDD custom nodes |
| `minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors` | Ref2VA trunk LoRA and external head bank | Companion PDD custom nodes |

Put any baked checkpoint in `ComfyUI/models/diffusion_models/`.
Put the two custom-node LoRA files in `ComfyUI/models/pdd_acc/`.

The **without-heads** checkpoint is the former `minimax_h3_ref2va_pdd_acc_8step_baked_int8_convrot.safetensors`.
It was renamed for clarity; its bytes and SHA256 are unchanged. Existing download links
using the old filename need to be updated.

## WITH HEADS — native ComfyUI

1. Update ComfyUI to a revision containing PR #17157.
2. Load `minimax_h3_ref2va_pruned_pdd_acc_8step_baked_with_heads_int8_convrot.safetensors` (smaller) or the full with-heads checkpoint with **Load Diffusion Model / UNETLoader**.
3. Use the usual MiniMax H3 Ref2VA conditioning, text encoder, video VAE and audio VAE.
4. Use **Euler**, the **simple** scheduler, **8 steps**, **CFG 1.0**, and **denoise 1.0**.
   The video/audio sigma shifts are **12.0 / 3.0**, the native model defaults.
5. Do not apply the PDD acceleration LoRA or PDD Apply node again: its trunk and heads
   are already included. Keep other acceleration LoRAs disabled for this recipe.

Four steps are also supported by the original PDD release; start with eight.
The native simple scheduler at four and eight steps was checked against the PDD block
boundaries. The checkpoint stores **32 fine-interval heads per stream**:
`config.transformer.pdd_steps = 32` is the bank size, not the requested sampler-step count.

This is a diffusion-model checkpoint. The text encoder and VAEs remain separate.
For the full with-heads variant, the existing trunk-only bake is preserved byte-for-byte;
only the four output-head tensors and metadata change. The pruned variant is a separate
bake onto the official pruned base, including a curve-space conversion of the PDD timing-layer updates. The head tensors are stored in float32:

| Tensor | Shape |
|---|---|
| `final_layer.video_out.weight` | `[3072, 5376]` |
| `final_layer.video_out.bias` | `[3072]` |
| `final_layer.audio_out.weight` | `[1024, 5376]` |
| `final_layer.audio_out.bias` | `[1024]` |

Each bank stores head 0 as a full head and subsequent entries as offsets from head 0.
Native ComfyUI combines the appropriate heads for each sampling interval.

## PRUNED INT8 WITH HEADS

Size: **21,055,724,824 bytes (21.06 GB)**,
about **38.3% smaller** than the full with-heads checkpoint (34.12 GB).
The starting checkpoint is the official
[Comfy-Org pruned Ref2VA INT8 ConvRot base](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors).

This variant removes the need to apply the PDD acceleration LoRA during inference
and uses the pruned base's compact timing layers. It can reduce model storage,
memory pressure and repeated LoRA work when layers are offloaded on lower-VRAM cards.
Checkpoint file size is not total inference VRAM: activations, clip length, resolution,
the text encoder, VAEs and offloading settings also matter. No minimum-VRAM claim is made here.

Use the same native eight-step recipe above. The PDD strength is fixed at 1.0.
The INT8 trunk is dequantized, merged and requantized; quantization and the pruned curve
approximation can change results compared with the full checkpoint.

Validation for this variant:

- All **50** affected timing-layer projections were rebased with both weight and bias updates.
- The timing curve was checked against the full base's time embedder over all 1025 grid points;
  relative L2 error: **1.377e-05**.
- The rebased PDD updates were independently compared with the dense LoRA at nine time points per module.
- All 58 unquantized target weights and
  50 affine bias updates match their intended stored values exactly.
- 424 untargeted tensors remained byte-identical.
- 24 of 200 quantized modules were sampled after writing;
  maximum relative L2 requantization error: **7.378e-03**.
- Every final checkpoint tensor was verified after writing. Both native head banks passed
  46 numerical cases each, including eight-step and four-step partitions.
- The complete checkpoint loaded through native ComfyUI on CPU with no missing or unexpected weights,
  correctly detecting the pruned timing grid and 32 PDD heads per stream.

These are structural and numerical checks, not a rendered-video quality or VRAM benchmark.
See [the pruned validation report](pruned-with-heads-validation.json).

## WITHOUT HEADS — companion PDD nodes

Load `minimax_h3_ref2va_pdd_acc_8step_baked_without_heads_int8_convrot.safetensors` with UNETLoader and use
[ComfyUI-MiniMax-H3-PDD-Acc](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PDD-Acc)
with the Ref2VA PDD file:

- `lora_strength = 0.0` — the trunk is already baked.
- `head_strength = 1.0`, `nfe = 8` (four also supported).
- Euler, CFG 1.0, video/audio sigma shifts 12/3.
- Connect the Apply node's sigma output to `SamplerCustomAdvanced`.

**Without heads** means without the *PDD bank*: this file still has the normal single
video and audio output heads. It is not a standalone PDD checkpoint.

## Standalone LoRA conversions

The two `..._comfyui.safetensors` LoRA files retain their original custom-node format.
They contain a rank-64 trunk LoRA plus the unmodified original PDD head bank.
Use the companion node pack with the matching stock FL2VA or Ref2VA checkpoint,
trunk/head strengths 1.0, Euler, CFG 1.0 and the Apply node's sigma schedule.

Trunk conversion maps Diffusers Q/K/V projections to ComfyUI's fused QKV layout,
swaps the SwiGLU projection halves, renames attention/MLP/refiner keys and preserves
LoRA scaling with explicit alpha tensors. The original standalone head bank remains
in its absolute-head format for the companion node pack.

## Full checkpoint conversion validation

The full with-heads checkpoint was derived from the existing verified trunk bake and the
same official `MiniMax-H3-Ref2VA-Acc-8Step.safetensors` source.

- All **1031 trunk tensors** are byte-identical to the without-heads checkpoint.
- Every output tensor was hashed and verified after writing.
- Native ComfyUI detects 24 video latent channels, 32 audio latent channels and 32 PDD heads.
- The native model constructor accepts the new head shapes.
- The complete checkpoint loads through native `comfy.sd.load_diffusion_model` on CPU.
- Native head evaluation was compared to the original bank averaged independently in float64,
  across all 32 individual intervals, eight-step and four-step partitions, and additional spans.
  Maximum relative L2 error across video/audio checks: **3.565e-07**.
- The float32 head conversion introduces only rounding error; the INT8 trunk is not re-quantized.

These checks validate checkpoint structure and head math. They are not a full rendered-video
quality benchmark. See [the validation report](with-heads-validation.json) and
[the conversion script](bake_pdd_with_heads.py).

## SHA256

| File | SHA256 |
|---|---|
| PRUNED INT8 WITH HEADS | `eeb4bd468c0432f3f413b05f667dbc734223ac9aef9e70c38b6c15dd345f4d39` |
| WITH HEADS | `d1f555a323a71a3ef33221527c240c9d63534f6fa5ff507b042a48f0423b8484` |
| WITHOUT HEADS | `fe8e58d8028e74607dfb41aa6a9ce92f5e0f989fff3e1eba6041caedcd1e111e` |
| FL2VA custom-node LoRA | `1dce71b90e5d993d3309474d47e414766e65dbd760f18978c399c82db5cda0ea` |
| Ref2VA custom-node LoRA | `5531fa0da887c24ec0083b0050ea9e4ff03c479cfbe21ef586d1c5c79bdc78a1` |

Full WITH HEADS size: **34,124,239,286 bytes**.

## Credits and license

Base model: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
PDD training: [Alibaba PAI](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
and the [PDD authors](https://arxiv.org/abs/2607.26004).
Native baked-head support: [ComfyUI PR #17157 by kijai](https://github.com/Comfy-Org/ComfyUI/pull/17157).
This repository provides format conversions, not newly trained weights.

The model card now reflects the upstream MiniMax H3 community license;
see the [base-model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
and [PDD source repository](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
