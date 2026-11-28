# multimodalart/MiniMax-H3-Pruned

Источник: https://huggingface.co/multimodalart/MiniMax-H3-Pruned · скачано 2026-11-28
Загрузок: 15128 · лайков: 5 · базовая: MiniMaxAI/MiniMax-H3 · теги: lora · обновлено: 2026-08-15T17:40:26.000Z

---

---
pipeline_tag: image-text-to-video
license: other
license_name: minimax-h3-community-license-agreement
license_link: https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
library_name: diffusers
base_model: MiniMaxAI/MiniMax-H3
tags:
  - text-to-video
  - image-to-video
  - image-text-to-video
  - text-to-audio-video
  - image-to-audio-video
  - reference-to-audio-video
  - audio-video-generation
  - multimodal
  - lora
---

# MiniMax-H3, AdaLN-pruned

[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) in diffusers format with the AdaLN input
projections reduced to their reachable rank. **52 GB smaller** across the two DiT partitions, no visible change in
output, and every published H3 LoRA loads - pruned-trained ones natively, released-trained ones projected.

| | released | this repo |
|---|---|---|
| `transformer/` | 66.28 GB | **40.24 GB** |
| `transformer_ref/` | 66.28 GB | **40.24 GB** |
| DiT parameters, per partition | 33.14 B | 20.11 B |
| `text_encoder/` | 66.71 GB | **52.48 GB** |

The conditioner is hosted here too, truncated: MiniMax-H3 conditions on the *unnormalized* hidden state after the
Qwen3-VL conditioner's 50th decoder layer (`hidden_states[50]`), so decoder layers 51-63 and the language-model head
can never reach the conditioning and are dropped. `text_encoder/` ships decoder layers 0-50 - the 51st is kept so
`hidden_states[50]` stays a pre-norm intermediate rather than the final-norm output - and its embeddings are **bitwise
identical** to the released conditioner's, text-only and vision-bearing presentations alike. It stays bfloat16; a
quantized conditioner is planned as a separate variant, pending its own quality measurement.

Both VAEs, both schedulers, the tokenizer and the processor are *not* duplicated here. `modular_model_index.json`
points those at `MiniMaxAI/MiniMax-H3`, so loading this repo pulls only the pruned partitions, the truncated
conditioner, and the released components you already have.

## What was removed, and why it is safe

Each of the 50 blocks holds `adaln_proj.linear`, a `Linear(2688 -> 96768)`, and `norm_out.linear` adds one more.
Together that is **13.03 B parameters, 39.3% of the checkpoint**. All 51 of them read the same vector,
`silu(time_embedder(t))`, which is a function of the scalar timestep alone - so however wide it is, the set of
values it ever takes is a one-dimensional curve in `R^2688`.

An 8-dimensional affine subspace covers that curve to a relative RMS error of **1.45e-5**. One bfloat16 rounding step
of the weights those projections are stored in is about **3.9e-3** - roughly 250x larger. So the projections are
folded onto that subspace: `W @ x + b` with `x = mean + c @ basis` becomes `(W @ basis.T) @ c + (b + W @ mean)`, the
timestep MLP is replaced by a 1025-entry table of the coordinates `c(t)` (linearly interpolated), and every AdaLN
projection takes an 8-wide input instead of a 2688-wide one.

### Measured fidelity

Every folded projection, evaluated over 200 off-grid timesteps against an exact float64 evaluation of the released
weights. "Released" is the released checkpoint running its own bfloat16 arithmetic; "pruned" is this checkpoint
running its.

| | `transformer` | `transformer_ref` |
|---|---|---|
| rank-8 subspace residual (no storage rounding) | 1.51e-5 | 1.29e-5 |
| **pruned** vs exact float64, worst of 51 modules | **1.715e-3** | **1.715e-3** |
| **released** vs exact float64, worst of 51 modules | 1.751e-3 | 1.758e-3 |
| modules where pruned is closer to exact than released | **51 / 51** | **51 / 51** |
| pruned vs released, worst of 51 modules | 1.35e-3 | 1.41e-3 |

The approximation error is ~100x below the arithmetic the released checkpoint already runs at, which is why the
pruned model lands *closer* to the exact answer than the released one does, at every single projection.

### End to end, with a control

Same-seed `t2va` - 10 steps, 960x544, 124 frames, one cached conditioning, seed 42 - comparing the final latents
handed to each VAE. A 10-step video diffusion trajectory amplifies *any* bfloat16-level perturbation, so the number
that matters is not the raw divergence but how it compares to a perturbation that is definitionally harmless. The
control is the **released** model against **itself**, run once with FlashAttention-3 and once with the native
backend: mathematically the same computation, differing only in the order bfloat16 sums are accumulated.

| video latents | cosine | relative L2 | max abs |
|---|---|---|---|
| released (FA3) vs released (native) - *control, same weights* | 0.99248 | 0.1227 | 7.66 |
| **released (FA3) vs pruned (FA3)** - *the fold* | **0.99128** | **0.1321** | **8.59** |
| released (native) vs pruned (FA3) | 0.99274 | 0.1205 | 7.71 |

| audio latents | cosine | relative L2 | max abs |
|---|---|---|---|
| released (FA3) vs released (native) - *control, same weights* | 0.99964 | 0.0292 | 0.255 |
| **released (FA3) vs pruned (FA3)** - *the fold* | **0.99911** | **0.0437** | **0.358** |
| released (native) vs pruned (FA3) | 0.99972 | 0.0236 | 0.157 |

The fold moves the trajectory about as far as changing the attention kernel does, and *less* far than that when
measured from the native-backend run. Latent statistics are unchanged: video latent standard deviation is 2.146331
released against 2.146338 pruned. Whatever this model's 10-step trajectory is sensitive to, it is not the pruning.

**Not bitwise.** This is a refactor of a numerical function, not a lossless re-encoding: rounding the folded weights
into bfloat16 lands on different bits than the released weights did. It is *effectively* lossless in bfloat16 - the
approximation the fold introduces is about 250x below one rounding step of the weights it replaces, and what reaches
the output is the size of a change in summation order.

## Usage

```python
import torch
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained(
    "multimodalart/MiniMax-H3-Pruned", workflow="t2va", trust_remote_code=True, components_manager=manager
)
pipe.load_components(dtype=torch.bfloat16, trust_remote_code=True)
manager.enable_auto_cpu_offload(device="cuda")

result = pipe(
    prompt="A red fox trotting through a snowy pine forest, snow crunching underfoot",
    height=544,
    width=960,
    num_frames=124,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
```

`trust_remote_code=True` is needed on **both** calls: the first resolves this repo's pipeline wiring, the second
resolves the transformer class in [`transformer/modeling_minimax_h3_pruned.py`](transformer/modeling_minimax_h3_pruned.py).
Read it before you run it - it is one file, and it subclasses `MiniMaxH3Transformer3DModel` without touching
`forward`.

The timestep embedder answers `linear_1.weight.dtype` with the table's float32, as a property rather than a module:
diffusers before [#14398](https://github.com/huggingface/diffusers/pull/14398) reads the released timestep MLP's
dtype off that attribute, and this checkpoint has no MLP to read it from. It is a dtype answer and nothing else -
it reaches no `state_dict`, no `named_modules`, no quantizer scan - and can go once every consumer carries #14398.

`workflow="t2va"` and `workflow="fl2va"` load `transformer/`; `workflow="ref2va"` loads `transformer_ref/`. Only the
partition a workflow needs is downloaded.

26 GB of headroom also buys speed wherever weights are being moved. On one 80 GB H100 with automatic CPU offload,
the parity run above took **94 s** pruned against **205 s** released, at a 52.7 GB peak against 65.2 GB - not because
the arithmetic is cheaper (the AdaLN projections were never the bottleneck) but because far less of the model has to
be paged in and out per step. On a card with room for all of it, expect the two to run at the same speed.

## LoRAs

**Every published H3 LoRA loads here** - the ones trained against a pruned checkpoint natively, the ones trained
against the released one by projection.

### Pruned-trained LoRAs, natively

ai-toolkit trains against a pruned checkpoint by default, so its H3 LoRAs carry `adaln_proj.linear.lora_A` of shape
`[rank, 8]`. That is a size mismatch against the released `Linear(2688, ...)` and an exact fit here.

This works because the coordinates are **not ours**. A rank-8 basis is unique only up to sign and rotation, so a
pruned-trained LoRA is portable only between checkpoints that share the coordinate convention. Rather than derive a
basis, the one the ecosystem already trains against was extracted: `time_embedder.table` is the `adaln_t_table` from
the [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) pruned files, **byte for byte**, and the basis was
solved for from that table against the released timestep MLP.

Both published pruned-trained LoRAs were checked against their trainer's own arithmetic - `lora_B @ (lora_A @ c(t))`
with `c(t)` interpolated from the shipped table - over a 64-point timestep sweep:

| coordinates the update is read in | relative error vs the trainer's own result |
|---|---|
| **this repo (shipped table, verbatim)** | **0.0, exactly** |
| a sign gauge differing on 3 of 8 columns (what a re-derived SVD lands on) | 0.054 - 0.078 |
| an arbitrary rotation of the same 8-dimensional subspace | 1.10 - 1.63 (cosine as low as -0.97) |

The last row is why the table is copied and not recomputed: a re-derived basis spans the same subspace and is equally
accurate, and would still turn every existing LoRA into noise. Both files load with all 50 AdaLN projections wrapped
by PEFT at scale exactly 1.0, 362 modules in total.

### Released-trained LoRAs, by projection

LoRAs trained against the **released** checkpoint's 2688-wide AdaLN projections - the 4-step turbo LoRA and its
conversions - are mapped onto these coordinates when they load. Nothing about them is approximated beyond what the
pruning already approximates: the identity that folded the weights folds the adapter.

A LoRA on the released projection contributes `lora_B @ (lora_A @ x)`, and `x = mean + c @ basis`, so

```
lora_B @ (lora_A @ x)  =  lora_B @ ((lora_A @ basis.T) @ c)  +  lora_B @ (lora_A @ mean)
```

`lora_A @ basis.T` is an `[rank, 8]` `lora_A` over the pruned coordinates, at the same rank; `lora_B` is unchanged;
and what is left over is a **constant**, which no `Linear(8 -> ...)` can produce. That constant is carried as a
float32 offset on the modulation, alongside the fold's own `folded_bias` and for the same reason - it is ~98% of
what these adapters do to the AdaLN path, so rounding it into bfloat16 would spend a full rounding step of the
modulation on almost the whole update. The factors are computed in float64.

`adaln_basis` and `adaln_mean` ship as `adaln_affine.safetensors` (97 KB) beside each partition's weights. They are
the map the projections were folded with, read back, not re-derived - for the same reason the table is copied and
not recomputed.

**Why not PEFT's `lora_bias`.** That is the obvious place to put a constant, and it is the wrong one here.
`lora_bias` is a `LoraConfig` flag, not a per-module one, so switching it on for the 51 AdaLN modules also switches
it on for the other 312: it warns on every bias-free target, creates zero biases on the attention and feed-forward
modules that then read as missing keys, and makes `fuse_lora` raise `Impossible to merge LoRA with lora_bias=True
because the base layer has no bias` on all 51 - `adaln_proj.linear` is deliberately bias-free, which is what makes
PEFT wrap it exactly as it wraps the released projection. It would also put the constant in bfloat16, on the wrong
side of the argument this repo already made for `folded_bias`. The float32 buffer beside `folded_bias` avoids all
four, and PEFT still owns the state that matters: the offsets are read through `active_adapters`, `scaling` and
`disable_adapters`, so they follow `set_adapters`, `disable_lora` and `delete_adapters` on their own.

Measured per AdaLN module over 200 off-grid timesteps, projected against full-space, both in float64:

| | worst of 51 | mean of 51 |
|---|---|---|
| rank-8 subspace residual on the timestep curve | 1.47e-5 | - |
| [`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) v4 step 600 EMA | **2.05e-5** | 1.26e-5 |
| [`InstantX/MiniMax-H3-Turbo-Lora-Diffusers`](https://huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers) | **1.53e-5** | 0.88e-5 |

which is the subspace residual and nothing else. Both files attach to **363 modules** - 312 attention/feed-forward
and token-refiner ones untouched, 51 AdaLN ones projected - all at scale exactly 1.0. Loaded on top of the released
weights and on top of these, the LoRA'd AdaLN function was compared against an exact float64 evaluation at three
probe modules: the pruned side is the closer of the two at all three, the same way the fold itself is.

End to end, 960x544, 124 frames, 7 steps, seed 42, `larryvrh` v4, against the identical request on the released
weights: **cosine 0.9958** with the LoRA on and **0.9992** with it off, and the on/off difference - the turbo effect
itself - is 0.9609 here against 0.9641 there. Frame for frame the two runs are the same video.

How much of that is the AdaLN half? On these particular files, not much - their AdaLN update is about 2e-4 of the
modulation, and simply *discarding* those 51 modules lands at 0.9960 against the released run where projecting
lands at 0.9958, which is the same number twice. What is not the same number is projecting them and dropping the
constant term: **0.9947**. The half-applied version is the only one of the three that is measurably wrong, which
is both the argument for where the constant is kept and the reason to do this exactly rather than approximately -
an adapter that leans harder on AdaLN than a turbo LoRA does has no other way to arrive intact.

A file whose AdaLN modules are not uniformly 8-wide or uniformly 2688-wide raises rather than loading some of them:
an adapter that applies to 50 of its 51 AdaLN projections is not the adapter anyone trained.

One caveat: the projection happens at load, and the constant terms are not PEFT parameters, so
`save_lora_weights` on a projected adapter writes the projected factors *without* them. Distribute the original
file, not a re-save of it.

Everything outside `adaln_proj` and `norm_out` - attention, feed-forward, the token refiner - is untouched by the
pruning and was always loading either way.

LoRA loading for MiniMax-H3 landed in diffusers via
[huggingface/diffusers#14408](https://github.com/huggingface/diffusers/pull/14408). This repo declares the stock
`MiniMaxH3ModularPipeline`, which is where that PR attaches the loader - so on a diffusers without it the repo loads
and generates normally, and gains `load_lora_weights` the moment the PR is in your install. Nothing here needs to
change.

## 8-bit compute

The transformer's attention and feed-forward matmuls can run in **int8** - not int8 weights dequantized back to
bfloat16, but int8 tensor cores multiplying int8 activations by int8 weights - at a quality **above** what int8
weight-only gets, and, once the blocks are compiled, slightly faster than bfloat16. Two extra lines:

```python
transformer = pipe.transformer
transformer.enable_convrot()                    # fold H into the 300 attention/feed-forward weights

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
quantize_(transformer, Int8DynamicActivationInt8WeightConfig(), filter_fn=transformer.convrot_filter)

transformer.compile_repeated_blocks(fullgraph=False)   # not optional, see the speed table
```

`transformer.quantize_8bit()` does the same thing one linear at a time - the working set is a single weight
rather than the model - which is what you want when the bfloat16 model is on the CPU and the card is small:

```python
transformer.quantize_8bit(device="cuda")        # rotate + quantize per layer, then put each layer back
```

Both need `torchao` (0.17 was what this was measured on) and, for the compile step, a diffusers with
`compile_repeated_blocks`. Neither touches the checkpoint on disk.

Nothing is serialized: the rotation is a fixed, parameter-free matrix, so there is no second copy of the weights
to download. `transformer/` stays the 40.24 GB bfloat16 partition and becomes a 19.5 GB int8 model in memory.

### What ConvRot is

Each quantized `Linear` gets its weight replaced by `W @ H` and its input by `x @ H`, where `H` is a
block-diagonal stack of 256x256 *regular* Hadamard blocks (Kronecker powers of the symmetric order-4 seed,
scaled by `1/16`). `H` is symmetric and orthogonal, so it is its own inverse and `(x @ H) @ (W @ H)^T` is
`x @ W^T` exactly. The function does not change. The *distribution* does: every coordinate of `x @ H` is a
+-1 combination of 256 input channels, so one outlier channel stops setting the quantization scale for its
whole row.

The reason this is cheap to adopt is that the rotation is created and destroyed inside a single matmul. There
is no rotated residual stream, nothing that has to commute with the AdaLN modulation, and no calibration data.

### Measured

`t2va`, seed 42, 960x544, 124 frames, 20 steps, one cached conditioning, FlashAttention-3, H100. Every leg is
the same request against the same bfloat16 baseline; only the transformer's linears differ. "8-bit compute"
means the matmul itself runs at 8 bits - weight-only quantization dequantizes and multiplies in bfloat16.

| | 8-bit compute | video cos | video PSNR | audio cos | video-latent cos | s/step | peak |
|---|---|---|---|---|---|---|---|
| bfloat16 | - | 1 | inf | 1 | 1 | 3.51 | 52.7 GB |
| *control: bfloat16, blocks compiled* | *-* | *0.9940* | *23.17 dB* | *0.9968* | *0.9907* | *3.03* | *52.7 GB* |
| int8 weight-only | no | 0.9773 | 17.36 dB | 0.9852 | 0.9657 | 3.74 | 34.8 GB |
| fp8 dynamic, every bfloat16 linear | yes | 0.9585 | 14.83 dB | 0.8386 | 0.9367 | 3.49 | 34.0 GB |
| fp8 dynamic | yes | 0.9583 | 14.81 dB | 0.8775 | 0.9416 | 3.50 | 34.8 GB |
| fp8 dynamic + **ConvRot** | yes | 0.9678 | 15.84 dB | 0.9263 | 0.9469 | 3.57 | 34.8 GB |
| int8 dynamic | yes | 0.9819 | 18.37 dB | 0.9906 | 0.9702 | 11.23 | 34.8 GB |
| **int8 dynamic + ConvRot** | **yes** | **0.9893** | **20.65 dB** | **0.9943** | **0.9819** | 11.28 | 34.8 GB |

The control row is the bfloat16 model against *itself*, run once eagerly and once with the blocks compiled:
identical weights, different kernels and a different summation order, 0.9940 video cosine. That is the floor
this comparison can resolve over 20 diffusion steps, and int8 + ConvRot at 0.9893 sits just above it.

Two more things in that table are worth stating plainly.

**ConvRot buys almost nothing for fp8 and everything for int8.** Per-layer, on activations captured from a real
denoise, the mean relative error of the matmul goes 0.0278 -> 0.0269 for fp8 (a 3% improvement) and
0.0242 -> 0.0101 for int8 (a 2.4x improvement, and 0.0387 -> 0.0104 on the feed-forward down-projection alone).
fp8's error is a mantissa floor - a floating-point format's relative accuracy does not care how large the
outliers are - so spreading the outliers does not help it. int8's error *is* the outliers, which is exactly what
a rotation removes. The same study says the popular alternative is the weaker one: SmoothQuant/AWQ
per-input-channel pre-scales, calibrated on those activations, take int8 from 0.0242 only to 0.0171, and buy a
further 0.0009 on top of ConvRot. This repo ships no pre-scales, and neither do ComfyUI's H3 checkpoints.

ComfyUI's own fp8 build concedes the same point from the other side: in
`minimax_h3_*_pruned_fp8_scaled.safetensors`, all 50 `mlp.fc2` layers carry
`"full_precision_matrix_mult": true` and no `input_scale` - the feed-forward down-projection is the one they
refuse to give 8-bit activations to without a rotation.

**int8 with ConvRot lands above int8 weight-only.** 0.9893 against 0.9773 video cosine, 20.65 dB against
17.36 dB - so on this model there is no accuracy argument left for giving up 8-bit compute. Its per-layer
matmul error, 0.0101, is within a fifth of int8 weight-only's 0.0086, which itself only quantizes half the
operands.

### Speed, and why `compile_repeated_blocks` is not optional

torchao's dynamic-activation paths are eager-hostile: the quantize step is a separate kernel writing a
full-size intermediate. Steady-state, dropping the compile warmup:

| | eager | compiled |
|---|---|---|
| bfloat16 | 3.51 s/step | 3.03 s/step |
| fp8 dynamic + ConvRot | 3.57 s/step | 2.47 s/step |
| **int8 dynamic + ConvRot** | 11.28 s/step | **2.90 s/step** |

Eager int8 dynamic is 3.2x *slower* than bfloat16 and worth nobody's time. Compiled, it is faster than
bfloat16. One `[4096, 14336] x [14336, 5376]` matmul on this H100, all compiled: bfloat16 1.14 ms, int8 dynamic
0.89 ms, int8 dynamic + ConvRot 1.02 ms, fp8 dynamic 0.77 ms. The online rotation costs 0.13 ms, about 11% of a
bfloat16 matmul, and it is the only thing ConvRot adds at inference.

`torch._int_mm` through cuBLAS is poor on Hopper specifically (98 TOPS measured, against 522 TFLOPS bfloat16 on
the same shape); Inductor's Triton int8 kernel gets 0.68 ms where cuBLAS gets 0.83 ms - another reason the
compile step is load-bearing here. On Ampere and Ada, where int8 tensor cores are the well-trodden path and (on
Ampere) the only 8-bit ones that exist, this recipe is the whole point.

**fp8 is the faster leg and the worse one** - 2.47 against 2.90 s/step, 0.9678 against 0.9893 video cosine.
Take it only if you are on Ada or newer and buying 15% of step time with a visible amount of accuracy;
`enable_convrot` followed by `Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())` is the same two
lines.

### What is left alone

The 300 rotated-and-quantized linears are `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`,
`ff.net.0.proj` and `ff.net.2` of the 50 transformer blocks. Untouched, in their checkpoint precision: the
whole AdaLN path (`time_embedder.table`, `adaln_proj.linear`, `norm_out`), the patch projections and output
heads (float32), `context_embedder`, the two token-refiner blocks, and every norm. That is the same set
ComfyUI's `*_int8_convrot` files quantize, arrived at from their tensor inventory.

### LoRAs still work

`MiniMaxH3ConvRotLinear` is a plain `nn.Linear` subclass, so PEFT wraps it as a `base_layer` and a LoRA's own
branch reads the **unrotated** input in bfloat16. A LoRA trained against the ordinary checkpoint is therefore
correct on top of a rotated, quantized base - and `adaln_proj`, which is where pruned-trained LoRAs put most of
their weight and where a released-trained one is projected, is never rotated or quantized at all.

The order below is unchanged by the AdaLN projection: the projection happens at load, before anything is rotated,
and neither `enable_convrot` nor `quantize_` touches the modules it wrote to. Verified on `larryvrh` v4 - the
AdaLN update at three probe modules is bit-identical before and after quantizing, with all 363 modules and all 51
constant terms still attached.

The order matters, and there is one thing not to do:

```python
pipe.load_lora_weights(...)        # first
transformer.enable_convrot()       # then rotate - `convrot_layers` matches PEFT `base_layer`s too
quantize_(transformer, ..., filter_fn=transformer.convrot_filter)
```

Do **not** `fuse_lora` after rotating: a fused update is expressed in the unrotated basis and would be added to
a weight that no longer lives there. Fuse first, or do not fuse.

### Credit

The transform is ComfyUI's. `convrot` ships in
[`comfy-kitchen`](https://github.com/comfyanonymous/ComfyUI)'s int8 kernels and in
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)'s `*_int8_convrot` checkpoints; this repo
reimplements it on top of stock `torchao`. The reimplementation was checked against their shipped tensors:
rotating a released weight in float32 and quantizing it row-wise reproduces
`blocks.0.attn.out_proj.weight` from `minimax_h3_fl2va_pruned_int8_convrot.safetensors` **exactly** - all
38,535,168 int8 codes and every scale, bit for bit.

## Provenance

The pruned format is not an invention of this repo. It ships in the ComfyUI ecosystem as
[`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3)'s `*_pruned_*` files - a top-level
`adaln_t_table` of shape `[1025, 8]`, no `time_embedder.*`, and `[out, 8]` AdaLN projections - and
[Wan2GP](https://github.com/deepbeepmeep/Wan2GP) publishes the derivation as
`models/minimax_h3/prune_checkpoint.py`, which is where the construction here was learned from.

What this repo did *not* do is re-run that derivation. Since an SVD basis is only defined up to sign and rotation,
re-deriving it would have produced a subspace that is equally accurate and incompatible with every LoRA already
trained. Instead:

1. `adaln_t_table` was read out of `minimax_h3_fl2va_pruned_bf16.safetensors` and
   `minimax_h3_ref2va_pruned_bf16.safetensors` and is stored here unchanged as `time_embedder.table`.
2. The basis and offset were recovered as the least-squares solution of `[table, 1] @ affine = curve` in float64,
   where `curve` is `silu(time_embedder(t))` from the **released** weights on the table's own 1025-point grid. The
   recovered offset matches the curve's mean to 3.8e-8, confirming the mean-centred gauge the producers used.
3. Cross-check: folding the released weights with the recovered basis reproduces the shipped pruned weights to
   cosine **0.9999999** and relative L2 4.3e-4 (`transformer`) / 5.9e-4 (`transformer_ref`), which is fp16 storage
   error - the dtype those files ship in. Folding with the *other* partition's basis gives relative L2 4.5e-2, a
   100x separation, so the check has power and the `fl2va` / `ref2va` files map onto `transformer/` and
   `transformer_ref/` respectively.
4. The weights themselves come from the released bfloat16 tensors, folded in float64 and rounded to bfloat16 exactly
   once - not from the fp16 pruned files.

Wan2GP's own basis differs from Comfy-Org's by a per-column sign flip on 3 of the 8 columns; that is the concrete
reason step 1 extracts rather than recomputes.

One deliberate departure from the ComfyUI layout: the folded bias `b + W @ mean` is kept in **float32**, as a
separate `folded_bias` buffer, rather than rounded into the projection's bfloat16 bias. It carries almost the entire
modulation (peak 3.21 against the output's 3.27), so rounding it to bfloat16 costs a full output-scale step on every
evaluation and would put the pruned model *behind* the released one. In float32 it costs 20 MB per partition and puts
it ahead. `adaln_proj.linear` stays a plain bias-free `nn.Linear`, so PEFT wraps it exactly as it wraps the released
projection.

## Credits

- [MiniMax](https://huggingface.co/MiniMaxAI) for MiniMax-H3 and its released diffusers weights.
- [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (deepbeepmeep) for `prune_checkpoint.py`, the published
  derivation of the rank-8 AdaLN fold.
- [Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3) for the pruned checkpoint format and the specific
  `adaln_t_table` this repo adopts, which is what makes existing LoRAs portable.
- [ai-toolkit](https://github.com/ostris/ai-toolkit) (ostris), whose H3 trainer targets the pruned format and is why
  most third-party H3 LoRAs are 8-wide.
