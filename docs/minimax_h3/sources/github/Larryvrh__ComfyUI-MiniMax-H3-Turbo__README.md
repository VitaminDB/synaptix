# Larryvrh__ComfyUI-MiniMax-H3-Turbo__README

Source: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
Fetched: 2026-11-28



---

## FILE: README.md (https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/blob/HEAD/README.md)

# ComfyUI-MiniMax-H3-Turbo

Run [MiniMax-H3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3) —
joint **video + synchronized audio** — in as few as **4 sampling steps** instead
of ~20, with the
[MiniMax-H3 Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).

Two nodes drop straight into the official H3 workflow (text-to-video and
image-to-video):

| node | what it does |
|---|---|
| **MiniMax-H3 Turbo LoRA** | `MODEL → MODEL`, applies the turbo LoRA |
| **MiniMax-H3 Turbo Sampler** | `→ SAMPLER`, feeds `SamplerCustomAdvanced` |

## Which checkpoint — `v4` (600) or `v1` (850)?

For **most** work, use **`minimax_h3_turbo_v4_step600_ema.safetensors`** (from the
[LoRA repo](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)). It's the
strongest checkpoint so far: much better static / small-motion shots, markedly
better micro-detail (faces, fingers, texture), and the over-sharpening / plastic
look of the earlier `v1` (~850) line is fully resolved.

v4 introduced a **static-frame enhancement**. The one trade-off shows up **only at
4 steps with large, fast motion**, where v4 can produce **motion-smear / trailing
ghosting** (actively being fixed). **Using 6–8 steps largely removes it** — and v4
tolerates higher step counts better than v1 (which over-sharpens at high steps +
strength 1.0). For the narrow case of **4 steps *and* heavy motion**, the older
**`v1` ~850** checkpoint can still be friendlier.

```
Using 6–8 steps?        ── yes ──►  v4-600  (recommended)
   │ no (4 steps)
   ▼
Heavy / fast motion?    ── no  ──►  v4-600  (recommended)
   │ yes
   ▼
                                    v1-850  (friendlier at 4-step heavy motion)
```

Still a preview — the two areas still being improved are **audio** and **fast,
intense motion**.

## Install

**Via ComfyUI-Manager** — search "MiniMax-H3 Turbo" and install. **Or manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo
```

Then restart ComfyUI. Keep the node updated (Manager, or `git pull`) — it evolves
alongside the weights.

Put the LoRA `.safetensors` into `ComfyUI/models/loras/`. You also need the base
MiniMax-H3 model, VAEs and text encoder from the official release — see the
[MiniMax-H3 tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3).

## Use

Start from the **official MiniMax-H3 workflow** (t2v or i2v) and make two changes:

1. Insert **MiniMax-H3 Turbo LoRA** between the model loader and the sampler
   (`… → Load Diffusion Model → MiniMax-H3 Turbo LoRA → SamplerCustomAdvanced`),
   and pick the turbo `.safetensors`.
2. Feed `SamplerCustomAdvanced` from **MiniMax-H3 Turbo Sampler**, and set the
   scheduler node (`BasicScheduler`) to `simple` at **≥ 4 steps**.

Everything else — conditioning, VAE decode, audio output — stays as in the
official graph, so **both t2v and i2v** work unchanged. A ready-made t2v workflow
is in [`example_workflows/`](example_workflows/minimax_h3_t2v_turbo.json) — drag it
into ComfyUI to see the wiring.

## Steps and strength

- **4 steps is the recommended *minimum*; 4–8 is the useful range.** 6–8 steps
  look noticeably better than 4, so add steps if you can afford them. Past **8
  steps** it stops helping and can start to introduce **over-sharp artifacts** —
  no benefit to going higher, so stay in **4–8**.
- **Keep `strength` at `1.0`.** It's tuned for 1.0 and holds up across the 4–8 step
  range. Only reach for the dial if a *specific* clip misbehaves: **blurry ghosting
  / smear → nudge up** (`~1.05–1.2`), **over-sharp grain → nudge down**
  (`~0.8–0.95`).
- Keep the scheduler on `simple`.

## Base model & `low_vram`

**Base model** — works with any MiniMax-H3 base: full (`bf16`, `int8_convrot`)
**and the pruned/curve variants** (`pruned_int8`, `pruned_fp8`). The node detects a
pruned base automatically and re-injects the LoRA's time-conditioning at run time
(a small `silu(t_emb)` grid ships with the node for this), so **one LoRA file
covers every base**.

**`low_vram`** (node switch) trades sharpness for peak VRAM:

- **off (default)** — applies the LoRA at run time (bypass): sharpest, recommended,
  a little extra peak VRAM.
- **on** — merges the LoRA into the weights: lowest peak VRAM, so smaller GPUs run
  and longer / higher-res clips fit, but the result is **softer on quantized
  (`int8` / `fp8` / pruned) bases** (the tiny update is partly rounded away when
  folded into the quantized weights).

The node streams the base model, so it runs on much smaller GPUs than the ~33 B
size suggests — an 80 GB GPU is only needed for the largest resolutions in `bypass`
mode. If you hit out-of-memory, turn `low_vram` **on** and/or lower the resolution
or frame count.

## Why a custom sampler (and how it adapts)

MiniMax-H3 denoises the video and audio streams on two different flow schedules
(video shift 12, audio shift 3). **Recent ComfyUI handles this natively** — its
`ModelSamplingAV` carries the audio latent on the video schedule — so a stock
sampler already produces correct audio there. On **older ComfyUI without that
support**, a stock sampler steps both streams on one schedule and badly over-steps
the audio at 4 steps, so the audio comes out distorted.

This node's sampler **auto-detects which ComfyUI it's on**: on recent builds it
steps as a plain single-schedule sampler (bit-for-bit the stock result); on older
builds it steps each stream on its own clock so audio stays clean at 4 steps. Keep
it in the workflow and it does the right thing across ComfyUI versions — nothing to
change when you update. (On recent ComfyUI a stock `euler` also works; the Turbo
Sampler just keeps existing graphs running unchanged.)

## Notes

- **Resolution / length**: width and height are multiples of 32 (short edge
  typically 768); frame count is at 24 fps and snaps to the model's 17·k+5 grid
  (124 ≈ 5 s). Validated range ~124–362 frames.

## License

Apache-2.0.


---

# ISSUES / PRs of Larryvrh/ComfyUI-MiniMax-H3-Turbo


### #33 [open] PR: Fix LoRA tensor shape mismatch for ComfyUI master builds
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/33  by RiverSide71 2026-10-07T05:41:33Z

### ComfyUI Commit #16816 ("Consolidate Linear forward API") refactored the MLP structure.

- Previously, activation happened between layers.
- Now, the MLP calls fc2(fc1(x), input_act="swiglu"). This means fc2 receives an un-activated, 28,672-wide tensor and processes the SwiGLU activation internally.
- Because the node's LoRA branch was intercepting and running directly on that raw, un-activated tensor, it caused a shape mismatch.

### The Fix

This patch updates _FrugalLoRA.bypass_forward to handle these upstream changes gracefully:

1. Applies the missing activation to the input tensor before running the LoRA projection.
2. Scales the LoRA delta if a fused residual_scale argument is passed.

Note: Older versions of ComfyUI do not pass these new arguments, meaning legacy behavior remains completely unaffected and backward-compatible.


### #32 [open] PR: docs: warn about partial LoRA application with a stock loader
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/32  by FlowForgeLabAi 2026-09-25T04:54:50Z

Hi — thanks for this node. I hit something while testing that I think belongs in
the README, and I'd rather send it as a PR than leave it in an issue.

**What happened.** I loaded `minimax_h3_turbo_v4_step600` with the stock
`LoraLoaderModelOnly` (I hadn't installed this node yet) on a pruned INT8 base. The
console filled with:

```
ERROR lora ... adaln_proj linear.weight shape '[96768, 8]' is invalid
for input of size 260112384
```

Counting the modules: **208 of 259 applied (80.3%), with all 51 `adaln_proj.linear`
skipped** — once per layer, so 50 lines per sampling step. AdaLN is the pathway the
turbo distillation is supposed to change, so what I was sampling wasn't the model
that was trained.

**Why it belongs in the README.** Your README explains what the node does ("detects a
pruned base and re-injects the LoRA's time-conditioning at run time") but not what
happens without it, so there's nothing for a user to search for when the log starts
filling with `adaln_proj` errors.

This PR adds a short subsection between *Base model & low_vram* and *Why a custom
sampler*, covering: the observed applied/skipped counts, the error text, a pointer to
the `silu(t_emb)` re-injection as the reason the node is needed, and a one-line "search
the console for `adaln_proj`" check.

**Scope.** I've deliberately stated only what I measured — the counts and the exact
error — and not a causal explanation of which dimension is which, since I didn't verify
that far and don't want to put a wrong mechanism in your docs. Happy to reword, shorten,
or move it; happy to drop it if you consider this out of scope for the README.

Docs only, one file.


### #31 [open] ISSUE: Possible compatibility issue with native sol-attn: artifacts and native aborts with H3 Turbo LoRA
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/31  by CuttingRice 2026-09-17T17:02:22Z

# Possible compatibility issue with native sol-attn: artifacts and native aborts with H3 Turbo LoRA

## Related ComfyUI report and why I am reporting this here

I have also reported this issue to ComfyUI: [Comfy-Org/ComfyUI#16382](https://github.com/Comfy-Org/ComfyUI/issues/16382). That report contains the detailed comparisons, logs, workflow information and screenshots.

First, thank you for your work on H3 Turbo LoRA. I've found it very useful, and I really appreciate the work you've put into it.

I'm not sure whether there is anything that can or should be adjusted on the LoRA loader side, or whether the fix belongs entirely in ComfyUI's sparse-attention implementation. I'm sharing this here in case the results help you identify a compatibility issue or suggest a supported configuration. Being able to use Turbo LoRA together with sparse-attention acceleration would be valuable for everyday users like me, so I would ideally like to keep both enabled.

I did not encounter these problems in my older ComfyUI setup. However, the old and current attention-node configurations differ, so this is context rather than proof of a specific version regression. In the current setup, removing native sparse attention produced normal video with both merge and bypass. I am not claiming that the LoRA itself is faulty; I sincerely hope the observations and attached diagnostics are useful to you.

## Summary
In the earlier workflow combining MiniMaxH3TurboLoRA bypass mode with native sparse attention, the Python process terminated twice with `Fatal Python error: Aborted`. Both supplied failure traces stop at `_FrugalLoRA.bypass_forward`, __init__.py:305, after the progress bar reaches 4/8 and debug forward call #5 begins. There is no explicit torch.OutOfMemoryError, CUDA out-of-memory message, or native CUDA diagnostic in those traces.

## Updated controlled comparison

| Run | LoRA mode | Native BlockSparseAttention (sol-attn) | ModelAttentionBackend node | Total execution | Result |
| --- | --- | --- | --- | --- | --- |
| Earlier repeated failures | bypass | Present in supplied traceback | Complete submitted failure graphs not captured | Aborted around 4/8, forward call #5 | Native process abort |
| 2026-09-17 23:22:57 | merge | Enabled | Absent | 640 seconds (00:10:40) | Execution completed, but severe ripples/wavy and broken textures; unusable output |
| 2026-09-17 23:38:02 | merge | Removed | Absent | 778 seconds (00:12:58) | Execution completed; user confirms normal video without the prior widespread artifacts |
| 2026-09-18 00:05:00 | bypass | Absent | Connected, selection="comfy kitchen attention" | 780.514 seconds (00:13:00) | Completed; user confirms normal video without widespread ripple/broken-texture artifacts |

The two completed merge runs' original API prompts were recovered from their output PNG metadata and compared, including prompt text, reference image filenames, seed, models, resolution, duration, and sampler inputs. The only graph differences were removal of node 180 (BlockSparseAttention) and reconnecting node 169's model input from node 180 to node 155. This verifies matching submitted inputs; reference image file bytes at execution time were not separately hashed.

Both completed merge runs retained KJ's MiniMaxH3MemoryEfficientSageAttentionPatch. Neither had a ModelAttentionBackend node. The 778-second run has history status success/completed=true and no cached nodes. Output quality for that run is reported by the user; execution completion and output files were independently checked. Total runtime increased by approximately 138 seconds / 21.6%; this is one pair of runs, not a benchmark.

This is strong evidence implicating the enabled native sparse-attention path in the visual corruption under this configuration. It does not yet identify whether the cause is sparsity settings, the kernel, cached statistics, or interaction with other patches, and it does not prove the earlier native abort has the same cause.

The completed bypass follow-up changes both LoRA mode and adds ModelAttentionBackend relative to the normal merge run. Also, KJ's H3 patch directly replaces Attention.forward and calls SageAttention; connecting ModelAttentionBackend alone does not prove that this patched attention actually switches to kitchen's dense attention. The actual kernel path has not been instrumented.

## Latest bypass result and current assessment
The sparse-off bypass run completed at 2026-09-18 00:18:01 (UTC+8): 8/8 sampling steps, history success/completed=true, total 780.514 seconds. The user confirms normal output without the earlier widespread ripples/broken textures. Subjective surface appearance is outside the scope of this report.

Both merge and bypass have now produced normal output without native BlockSparseAttention. This weakens the earlier hypothesis that bypass alone is inherently failing in this environment and increases suspicion of an interaction involving native sparse attention and the Larry LoRA configuration. The strongest controlled evidence remains the two merge runs whose submitted graphs differ only by the sparse node removal/reconnection.

This is a suspected compatibility issue, not a proven pairwise root cause: a sparse-on run without Larry LoRA and a run correcting dense_blocks=-1 to explicit 49 have not been tested. The bypass follow-up also adds ModelAttentionBackend, so it is not a one-variable comparison against the normal merge run. KJ's direct Sage forward override remains installed, and actual dense kernel selection has not been instrumented.

The bypass follow-up reused cached upstream/conditioning nodes; the sampler, VAE decodes and output nodes ran. Its runtime is therefore not a clean performance comparison with the uncached merge run. Proactor connection-lost callback errors occurred around decoding/output, but this task still completed successfully; those messages should not be conflated with the earlier fatal process abort.

## Environment
- Windows 10 build 19045; NVIDIA GeForce RTX 5090, 32606 MB VRAM; driver 610.88; approximately 128 GiB system RAM
- Python 3.12.10
- ComfyUI 0.35.1, commit 856a922b; frontend 1.51.10
- PyTorch 2.12.1+cu132; **SageAttention 2.2.0** (Windows CPython 3.12 wheel: sageattention-2.2.0-cp312-cp312-win_amd64.whl)
- triton-windows 3.7.1.post27; torchvision 0.27.1+cu132; torchaudio 2.11.0+cu132; NumPy 2.4.4
- comfy-aimdo 0.5.3; comfy-kitchen 0.2.33
- sageattn3 1.0.0 is also installed; package presence does not establish that SageAttention 3 was selected
- Larry plugin commit 4274783a23afcfdbea3b4876cb79effd6c510785
- KJNodes current checkout da90cca857d7dd6fb615015197f99845e72c9bd3
- DynamicVRAM enabled; cudaMallocAsync; async weight offloading with 2 streams; pinned memory enabled

Versions were read from the actual ComfyUI environment's installed package metadata. The SageAttention wheel's build/source commit is not recorded in its installation metadata. No package changes were made during this investigation.

## Earlier merge workflow with sparse attention enabled (visually corrupted)
Captured from the running queue and confirmed in completed history:
- Base: minimax_h3_fl2va_pruned_bf16.safetensors (weight_dtype=default; runtime bf16)
- LoRA: minimax_h3_turbo_v4_step600_ema.safetensors; strength=1.0
- Text encoder: qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- Video/audio VAE: minimax_h3_video_vae_fp16 / minimax_h3_audio_vae_fp32
- Conditioning node: MiniMaxH3ReferenceToVideo with 5 reference images, ref_image_size=match. The selected model is fl2va, as stated above.
- 16:9, 1.2 MP, alignment 32 => 1504x832; duration input 15 seconds, frame expression => 362 frames; output 24 fps
- res_multistep sampler; simple scheduler; 8 steps; denoise=1.0; BasicGuider; seed=476416378834100
- Sigma shift: video=12, audio=3

Model chain:
UNETLoader -> MiniMaxH3TurboLoRA -> MiniMaxH3SigmaShift -> MiniMaxH3MemoryEfficientSageAttentionPatch -> native BlockSparseAttention -> ModelPreviewOverrideKJ -> BasicGuider / BasicScheduler -> SamplerCustomAdvanced

BlockSparseAttention: sol-attn, tau=1.3, start=0.2, end=1.0, dense_blocks=28,31,34,35,38-45,-1, min_tokens=12288, extra_tokens=256, sink_conditioning=exact_kv_and_rows.
KJ preview: taeh3.safetensors, max_resolution=1024, JPEG quality=80, preview_frames=362, preview_fps=12.

## Failure excerpt
```text
[H3TURBO fwd pruned/bypass] call#5 ... timestep=923.08
Fatal Python error: Aborted
  .../comfyui-minimax-h3-turbo/__init__.py:305 in bypass_forward
  .../comfy/weight_adapter/bypass.py:170 in _bypass_forward
  .../comfy/ldm/minimax/model.py:210 in forward
```
The top Python statement is:
```python
return base_out.add_(F.linear(F.linear(x, down), up), alpha=scale)
```
This identifies the visible Python frame, not necessarily the first failing asynchronous CUDA operation.

## Completed sparse-on merge run (visually corrupted; 2026-09-17, UTC+8)
```text
23:22:57 got prompt
23:23:19 [MiniMaxH3TurboLoRA] pruned base [merge]: ... strength=1.0 |
         208 backbone modules, 208 weights patched (merged) + 51 adaln injected at run time
23:29:24 [H3TURBO fwd pruned/merge] call#6 ... weight_patched=True
23:32:22 8/8 completed
23:33:37 Prompt executed in 00:10:40
```

## Scope / limitations
The attached redacted API workflows retain technical settings, but long prompts and image filenames are replaced with placeholders; reference images are not attached. These redacted variants have not been executed and are not tested minimal reproductions. The original two completed merge graphs were compared before redaction. Complete failure graphs are unavailable, so the earlier abort cases are not claimed as exact one-variable comparisons. An earlier bypass run at strength 0.9 also completed; bypass does not fail unconditionally.

No crash-time VRAM trace or native dump was captured, and there is no explicit OOM diagnostic. The known `dense_blocks` string ends in `-1`; ComfyUI's native parser interprets this as block 1, not the last block 49. This differs from the third-party SolAttn node's negative-index semantics and has not been independently tested as the artifact cause.

Could you help investigate the bypass abort in combination with ComfyUI's native sparse attention / DynamicVRAM? The related ComfyUI report is already open at [Comfy-Org/ComfyUI#16382](https://github.com/Comfy-Org/ComfyUI/issues/16382). Sparse-off merge and the subsequent sparse-off bypass run both supply user-confirmed normal-output references, not a general stability guarantee.

## A note from the reporter

I'm a regular ComfyUI user, and I don't have a deep understanding of its internal implementation. I've done my best to provide detailed logs and screenshots of relevant parts of the workflow, and I sincerely hope they help with investigating the issue. Please treat the technical explanations above as findings to verify, rather than a definitive diagnosis.

Codex helped me collect the diagnostic information and prepare this report for submission. Thank you for taking the time to look into it.



### #30 [open] ISSUE: _unique_t misses the audio row for keyframes carrying audio_latent (MiniMaxH3AddGuide)
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/30  by ian3563867-commits 2026-08-23T04:54:49Z

## Symptom

`RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0`

Raised on the first diffusion step, at the curve-mode adaln injection in `__init__.py`:

```python
x = x + (bv @ (av @ sv.T)).T        # [M, out] + [M', out] -> 3 vs 2
```

## Repro

- ComfyUI >= `e01fb4c5` (2026-08-13, Comfy-Org/ComfyUI#15439, which added `MiniMaxH3AddGuide`)
- MiniMax-H3 pruned `int8_convrot` base + Turbo LoRA `minimax_h3_turbo_v4_step600_ema`
- Node chain: `MiniMaxH3ReferenceToVideo` -> `MiniMaxH3AddGuide` -> `MiniMaxH3AddGuide` -> guider

Image-only guides work fine. Giving **any** `MiniMaxH3AddGuide` an `audio` input crashes.

## Root cause

`MiniMaxH3AddGuide` can anchor audio with **no image**, so it produces a keyframe dict whose
only payload is `audio_latent`:

```python
keyframe = {"resolved_frame_index": resolved_frame_index}
if image is not None:
    keyframe["latent"] = vae.encode(frames)
if audio is not None:
    keyframe["audio_latent"] = audio_latent
```

The core derives its row set from `layout.segments` (`comfy/ldm/minimax/model.py`):

```python
has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
has_aud_cond = any(k in ("cond_audio", "ref_audio") for _, _, k in layout.segments)
unique_t = sorted({t_v, t_a}
                  | ({seg_t["cond"]} if has_vis_cond else set())
                  | ({seg_t["ref_audio"]} if has_aud_cond else set()))
```

so an audio-carrying keyframe emits a `cond_audio` segment and `has_aud_cond` becomes true.

`_unique_t` keys the audio row off `refs` only:

```python
if any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):
```

and therefore undercounts by exactly one row. Same failure mode #10 fixed for `ref_audio`,
different producer.

## Patch

Verified locally on 1344x768 / 209 frames / 6 steps / RTX 5060 Ti 16 GB, with
`ref_images` + `ref_audios` + two image guides + one audio guide all active in the same run:

```python
    refs = payload.get("refs") or ()
    kfs = payload.get("keyframes") or ()
    if any(kf.get("latent") is not None for kf in kfs) or \
       any(r.get("kind") == "image" for r in refs):
        s.add(max(t_v, float(payload.get("visual_cond_noise_aug", 0.999))))
    if any(kf.get("audio_latent") is not None for kf in kfs) or \
       any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):
        s.add(max(t_a, float(payload.get("audio_cond_noise_aug", 1.0))))
```

This also mirrors the core more exactly on the visual side: an audio-only keyframe should not
add a visual row, whereas the current `payload.get("keyframes")` truthiness test would add one.

## Note on #29

#29 is a different producer — audio **refs** coming from multi-segment continuity, fixed by #10,
and that reporter was on a stale checkout. This one comes from `MiniMaxH3AddGuide` keyframes and
`main` at `4274783` still reproduces it.



### #29 [closed] ISSUE: Bug: _unique_t missing audio-cond row → 'tensor a (3) must match b (2)' with continuity that pins audio
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/29  by richfugui001-netizen 2026-08-16T12:11:51Z

## Bug: `_unique_t` missing the audio-cond row → `RuntimeError: size of tensor a (3) must match b (2)` when used with multi-segment continuity that pins audio

### Environment
- ComfyUI 0.30.1, MiniMax-H3 pruned int8_convrot base, Turbo LoRA `minimax_h3_turbo_v4_step600_ema` (bypass mode)
- Combined with a multi-segment workflow (AIMixer/ComfyUI_MiniMaxH3_Director) with continuity ON, which pins **both** previous video frames (22f) **and** previous audio (40 steps) into the next segment

### Symptom
Crashes on the very first diffusion step of the continuity segment:
```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```
at `__init__.py:235` in the curve-mode adaln injection:
```python
x = x + (bv @ (av @ sv.T)).T        # [M, out] + [M', out] broadcast fail: 3 vs 2
```
The first segment (no continuity, no pins) works fine; only segments with **video keyframes + audio refs both present** crash.

### Root cause
The model (`comfy/ldm/minimax/model.py`) builds `unique_t` as:
```python
unique_t = {t_v, t_a}
         | ({max(t_v, vis_aug)} if has_vis_cond else set())   # cond / ref_img segments
         | ({max(t_a, aud_aug)} if has_aud_cond else set())   # ref_audio segments
```
When a segment carries **both** a video cond (`t=0.999`) and an audio cond (`t=1.0`), the model produces **3 rows at step 0** (`{0.0, 0.999, 1.0}`) and **4 rows at later steps**.

But the Turbo LoRA's `_unique_t()` only mirrors the **visual** cond:
```python
s = {t_v, t_a}
if has_vis_cond:
    s.add(max(t_v, 0.999))     # ← no audio-cond row!
```
→ 2 rows at step 0, 3 rows later → the injected `silu_temb` grid is one row short → broadcast crash.

The MiniMaxH3Director's continuity pins audio as `refs` with `audio_latent` (creating `ref_audio` layout segments, `t=1.0`), which is exactly the "both conds" case.

### Reproduction
1. Any H3 workflow with the Turbo LoRA
2. Use a multi-segment generator that pins previous **video + audio** into the next segment (e.g. MiniMaxH3Director continuity ON, `continue_audio=True`)
3. Queue → segment 2 crashes on step 0 with the 3-vs-2 tensor error

### Suggested fix (mirror the model's set construction)
```python
def _unique_t(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond=False,
              vis_aug=0.999, aud_aug=1.0):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, vis_aug))
    if has_aud_cond:
        s.add(max(t_a, aud_aug))
    return sorted(s)
```
And in the wrapper, detect `has_aud_cond` from the packed layout the same way the model does:
```python
layout = payload.get("layout")
has_aud_cond = False
if layout is not None:
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
else:
    has_aud_cond = any(r.get("kind") in ("audio", "video", "video_audio")
                       and r.get("ref_audio_t", 0) > 0
                       for r in (payload.get("refs") or []))
vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))
us = _unique_t(ts, shift_v, shift_a, has_vc, has_aud_cond, vis_aug, aud_aug)
```
(Read `visual_cond_noise_aug`/`audio_cond_noise_aug` from the payload to stay in sync with the model.)

This mirrors `comfy/ldm/minimax/model.py` exactly and also fixes the latent "wrong value" issue for audio-only conds (row `1.0` vs hardcoded `0.999`).

### Note
Verified on sm_120 (RTX 5090): after the fix, all combinations (vis-only / aud-only / both / none) produce row counts identical to the model. Tested with the above reproduction across 4 sigma values.


> **Comment by afkaf (2026-08-16T16:01:22Z):**

Are you on the latest version after PR #16 ?


> **Comment by richfugui001-netizen (2026-08-16T20:08:30Z):**

Thanks for the fix and for checking! To answer your question: we were running commit 55fee864 (Aug 7) — i.e. before PR #10 and #16. Our local install was simply stale, which is why we hit the audio-cond row mismatch (audio refs from multi-segment continuity created a `ref_audio` row the old `_unique_t` didn't add).

As a stopgap we patched `_unique_t` locally to add the audio-cond row (max(t_a, audio_cond_noise_aug)) with layout/refs-based detection — essentially the same approach as PR #10.

I've now verified the latest upstream: `_unique_t` mirrors the model's row set (visual cond + audio cond, reading noise_aug from payload), and the `adaln_t_table` lookup + `st.shape[0] == x.shape[0]` guard in `_make_adaln_forward` removes the crash class entirely. We'll update to latest on our next container rebuild. Closing as resolved upstream — thanks again for both PRs!



### #28 [open] ISSUE: Community LoRAs (realism/tutu) silently fail on pruned int8 base - double diffusion_model. prefix in key map
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/28  by 18816478335q-pixel 2026-08-15T02:32:39Z

## Summary

`MiniMaxH3TurboLoRA` silently binds **zero adapters** when the LoRA file uses `diffusion_model.`-prefixed module names (the format used by popular community LoRAs such as fal's `h3-realism-people-t2v-i2v-r2v` and `tutu-t8-minimax-h3-av-20to8-nfe-lora`). The key map hardcodes `"diffusion_model.{}.weight"`, so a module name that already carries the prefix becomes `diffusion_model.diffusion_model.blocks.N...` and matches nothing in the model state_dict.

On a pruned int8 base this silently degrades quality (model runs **without** the LoRA at all - no error, just a `0 bypass adapters` line) and, for LoRAs that do contain adaln keys, raises:

```
KeyError: 'diffusion_model.blocks.0.adaln_proj.linear.alpha.lora_A.weight'
```

when the LoRA also ships `alpha` scalars (tutu does), because every key is splatted through `rsplit(".lora_", 1)[0]`.

## Environment

- ComfyUI 0.31.0 (frontend 1.48.6), Windows, Python 3.12, CUDA, RTX 4070 Laptop 8GB
- Node version: 4274783 (after PR #16 merge)
- Base: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (19.5 GB, pruned AdaLN-curve)
- LoRAs:
  - `h3-realism-people-t2v-i2v-r2v.safetensors` (fal, 125 MB) - keys like `diffusion_model.blocks.0.attn.out_proj.lora_A.weight`
  - `tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors` (738 MB) - includes `...adaln_proj.linear.alpha` scalars
- `MiniMaxH3TurboLoRA` strength 0.8, `low_vram` off (bypass)

## Reproduction

1. Load a pruned int8 H3 checkpoint with UNETLoader.
2. Chain `MiniMaxH3TurboLoRA` with a community LoRA that has `diffusion_model.`-prefixed keys (e.g. realism or tutu).
3. Queue any graph that executes the model.

Console shows:

```
[MiniMaxH3TurboLoRA] pruned base [bypass]: lora=h3-realism-... strength=0.8 | 104 backbone modules, 0 bypass adapters, 0 injections | ...
```

For tutu (has adaln + alpha keys) execution then fails:

```
KeyError: 'diffusion_model.blocks.0.adaln_proj.linear.alpha.lora_A.weight'
```

Expected: adapters bound (`N bypass adapters` > 0) and no KeyError.

## Root cause

1. `_apply_bypass_lora` / `_apply_merge_lora` build `key_map = {m: "diffusion_model.{}.weight".format(m) for m in modules}`. `modules` comes from the LoRA file itself (`k.rsplit(".lora_", 1)[0]`). larryvrh's own turbo LoRA keys are **unprefixed** (`blocks.0.attn.qkv_proj.lora_A.weight`), so `"diffusion_model." + m` is correct there. Community LoRAs are trained with the `diffusion_model.` prefix baked in, so the same line yields a double prefix and `load_lora` binds nothing. No error is raised - the LoRA silently no-ops.

2. `apply_lora` derives `modules` without filtering non-LoRA keys. tutu ships `adaln_proj.linear.alpha` (a scalar, no `.lora_` suffix); `rsplit(".lora_", 1)[0]` leaves it intact, so it lands in the `adaln` list and `lora[name + ".lora_A.weight"]` raises KeyError.

3. Same double-prefix pattern at `_inject_adaln_egrid`: `key = "diffusion_model." + name.rsplit(".linear", 1)[0]` - again wrong when `name` already starts with `diffusion_model.` (crashes in `get_model_object` with `AttributeError: 'MiniMaxH3' object has no attribute 'blocks'` / `'MiniMaxH3Model' object has no attribute 'diffusion_model'`).

## Proposed fix (locally verified)

Normalize module names once, then build all keys from the normalized form:

```python
def _h3_lora_key(m):
    '''Community LoRAs carry a 'diffusion_model.' prefix on module names;
    larryvrh's own turbo LoRA does not. Model state_dict always has the
    prefix, so normalize module names to the prefixed form.'''
    return m if m.startswith("diffusion_model.") else "diffusion_model." + m
```

- `key_map = {m: _h3_lora_key(m) + ".weight" for m in modules}` (both bypass and merge paths)
- `modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora if ".lora_" in k})` (skip `alpha` / scalar keys)
- `_inject_adaln_egrid`: `key = _h3_lora_key(name.rsplit(".linear", 1)[0])`
- `_int8_fused_fc2`: strip prefix before `get_attr(dm, ...)`: `_h3_lora_key(m).replace("diffusion_model.", "", 1)`

After the patch (same graphs):

```
realism: 104 backbone modules, 104 bypass adapters, 1 injections
tutu:    215 backbone modules, 163 bypass adapters, 1 injections, 50 int8 fc2 via merge + 51 adaln injected at run time
```

No KeyError, no AttributeError, canary shows `forward_owner=BypassForwardHook` (lora ACTIVE).

## Impact

Any user chaining community H3 LoRAs (realism, tutu, most HF style/character LoRAs) on a pruned/quantized base currently gets a silent no-op - results look like the base checkpoint without the LoRA. The `alpha`-scalar KeyError makes tutu-style LoRAs completely unusable. Worth adding a one-line `if n == 0: logger.warning(...)` guard in `apply_lora` so silent no-ops are impossible in the future.



### #27 [open] ISSUE: Reference audio crashes the sampler on pruned checkpoints (adaln row count mismatch)
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/27  by ruashots 2026-08-14T15:59:00Z

## Summary

On a pruned MiniMax-H3 checkpoint (one carrying `adaln_t_table`, so the model runs in curve mode),
`MiniMaxH3TurboLoRA` makes the sampler fail whenever the conditioning contains a reference audio
together with any visual reference or keyframe:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```

raised at `__init__.py:235`, inside the replacement `AdalnProj.forward` that `_inject_adaln_egrid`
installs.

`_unique_t` rebuilds, by hand, the list of distinct per-token timesteps that
`comfy/ldm/minimax/model.py` builds for itself, and the copy is missing the row a reference audio
adds. The model sizes `t_emb` from its own list, this node sizes the injected delta from its list, and
the two are then added together.

## Environment

- ComfyUI 0.31.1
- Windows, Python 3.12.1, torch on CUDA, one RTX 5090
- Base: a pruned INT8 H3 ref2va checkpoint, `adaln_t_table` shape `[1025, 8]`
- LoRA: `minimax_h3_turbo_v4_step600_ema.safetensors` at strength 1.0, `low_vram` off
- Reproduced identically with `low_vram` on

## Reproduction

Stock ComfyUI nodes plus this node only. No video file is needed: the reference clip is five blank
frames from `EmptyImage` and its soundtrack is any wav in the input folder.

1. `UNETLoader` with a pruned H3 ref2va checkpoint
2. `CLIPLoader`, type `minimax`
3. `VAELoader` for the video VAE, `VAELoader` for the audio VAE
4. `EmptyImage`, 512x288, `batch_size` 5
5. `LoadAudio`, any wav
6. `MiniMaxH3ReferenceToVideo`: 768x1344, length 39, `ref_image_size` match,
   `ref_videos.ref_video_1` from the `EmptyImage`, `ref_video_audios.ref_video_audio_1` from the
   `LoadAudio`
7. `MiniMaxH3TurboLoRA`, strength 1.0, `low_vram` off
8. `KSamplerSelect` euler, `BasicScheduler` simple 6 steps denoise 1.0, `BasicGuider`, `RandomNoise`,
   `SamplerCustomAdvanced`
9. `VAEDecode`, `VAEDecodeAudio`, `CreateVideo`, `SaveVideo`

**Expected:** a rendered file.
**Actual:** the sampler raises at the first step, after roughly 9 seconds with the models warm.

### The three runs that isolate it

| Change from the graph above | Result |
|---|---|
| nothing | crash, `a (3)` vs `b (2)` |
| delete the `ref_video_audio_1` link, nothing else | renders |
| delete the `MiniMaxH3TurboLoRA` node, keep the audio | renders |

`low_vram` on does not avoid it, because the adaln part is re-injected at run time in both modes
(`if pruned and adaln:` sits outside the `low_vram` branch).

The same crash occurs with `ref_audios` (a standalone reference sound) combined with a reference
image, so it is not specific to a reference video's soundtrack.

## Root cause

`comfy/ldm/minimax/model.py:549` builds the authoritative list, from which `t_emb` is sized ten lines
later:

```python
unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                  | ({seg_t["ref_audio"]} if has_aud_cond else set()))
```

Four values are possible: the video time, the audio time, `max(t_v, VISUAL_COND_TIMESTEP)` = `0.999`
for a keyframe or image/video reference, and `max(t_a, AUDIO_COND_TIMESTEP)` = `1.0` for a reference
audio.

`__init__.py:186` rebuilds it without the last term:

```python
def _unique_t(timestep, shift_v, shift_a, has_vis_cond):
    ...
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, 0.999))
    return sorted(s)
```

The two blocks meet at line 235:

```python
        x = x + (bv @ (av @ sv.T)).T                              # [M, out]
```

`x` is `base.linear(t_emb)`, so its rows are the model's list. The addend's rows come from
`shared["silu_temb"]`, built by `_interp_egrid` from `_unique_t`. The traceback runs
`SamplerCustomAdvanced` to `model.py:637` to `model.py:258` (`self.adaln_proj(t_emb)`) into line 235.

It reads 3 against 2 rather than 4 against 3 only because the first sampling step is at sigma exactly
1.0, where `time_shift_sigma(1.0, ...)` returns 1.0 for any shifts, so `t_v == t_a` and both sides
deduplicate one row. Starting the schedule below 1.0 (for example `BasicScheduler` at denoise 0.5,
which builds 12 steps and keeps the last 7) changes the message to `a (4)` vs `b (3)`. Note that
denoise 0.9 is a no-op here, since `int(6 / 0.9)` is 6.

## Two further defects in the same function

**`has_vis_cond` answers a different question than the model's.** The caller passes
`bool(payload.get("keyframes") or payload.get("refs"))`, that is "does this job have any reference",
while the model asks "does the packed sequence contain `cond` or `ref_img` rows". A sound-only job has
references but no visual rows, so its third row is computed at `0.999` where the model uses `1.0`. The
row counts happen to match, so nothing raises, and the delta is applied at the wrong point on the
curve on every step.

That one is not cosmetic. Measured on a fixed seed, with the pipeline confirmed bit-exact
reproducible across two separate executions with the cache cleared and models reloaded, fixing this
row changed a sound-only render substantially: PSNR between the two renders averaged 20.65 dB over 39
frames with no identical frames, the subject's structure became consistent across frames where before
its geometry shifted frame to frame, and background detail resolved. The audio changed only slightly
by comparison: correlation 0.9977, difference 23.4 dB below the signal, dominant partials within
0.3 dB.

**The sigma shifts are read from the wrong place.** `_inject_adaln_egrid` captures
`dm.sigma_shift_video` and `dm.sigma_shift_audio`, which stay at the checkpoint defaults.
`ModelSamplingMiniMaxH3` publishes the user's choice in `transformer_options`, and the model reads it
from there (`model.py:535-536`). The audio timestep depends only on the ratio between the two shifts,
so the two agree at 12.0/3.0 and at 9.0/2.25, both ratio 4, and disagree at any other ratio, silently.

**Edge case in the other direction.** A reference audio shorter than one audio-latent frame produces
no `ref_audio` segment while `refs` is still non-empty, so this node computes one row too many:
`a (1)` vs `b (2)`.

## Suggested fix

Mirror the model's expression instead of paraphrasing it, derive both flags the way the packed layout
derives its segments, read the shifts from `transformer_options`, and fail loudly rather than letting
torch report a bare shape error if the two ever diverge again.

```python
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def _unique_t(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond,
              vis_aug=VISUAL_COND_TIMESTEP, aud_aug=AUDIO_COND_TIMESTEP):
    sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
    t_v = 1.0 - sv
    t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
    s = {t_v, t_a}
    if has_vis_cond:
        s.add(max(t_v, vis_aug))
    if has_aud_cond:
        s.add(max(t_a, aud_aug))
    return sorted(s)


def _cond_flags(payload):
    refs = payload.get("refs") or []
    has_vis = bool(payload.get("keyframes")) or any(
        b.get("kind") in ("image", "video", "video_audio") for b in refs)
    has_aud = any(b.get("kind") in ("audio", "video", "video_audio")
                  and (b.get("ref_audio_t") or 0) > 0 for b in refs)
    return has_vis, has_aud
```

In the wrapper, take the shifts from `transformer_options` with the module attributes as the fallback,
and in `_make_adaln_forward` raise a named error when `st.shape[0] != x.shape[0]`.

A structurally better fix, if you would rather not keep a second copy of this logic in sync at all:
have the injection derive its rows from what the model actually built rather than recomputing them.
`_make_adaln_forward` already receives `t_emb`, so its row count is available at the point of use; the
missing piece is the timestep values, which the wrapper could take from the layout in the payload
instead of re-deriving from the sigma.

## Verification of the suggested fix

- The patched `_unique_t` and `_cond_flags`, extracted from the patched file itself, were compared
  against the model's expression across 990 configurations: 5 shift pairs, 2 keyframe sets, 9
  reference sets, 11 sigmas. Shipped disagrees in 811 of them, 538 by row count and 273 by value. The
  patched version disagrees in 0.
- The crashing graph above now renders, 768x1344, 39 frames at 24 fps, stereo 32 kHz.
- A full 8 second reference-video-plus-soundtrack render at 768x1344, 192 frames, with sage attention,
  sigma shift 9.0/2.25, the turbo sampler and 6 steps, completes and looks coherent frame by frame.
- The guard added to `_make_adaln_forward` did not fire on any step of any of those runs, which is
  independent confirmation that the two row lists now match.

> **Comment by afkaf (2026-08-14T18:06:49Z):**

@ruashots  Did you update to the latest version of this project? My PR #16 should have fixed this issue. It was merged around 3 hours ago. I've had zero issues using the pruned model with these nodes after making the changes in that PR. All combinations of references tested. 

Also the custom sampler that comes with this project has had better results than using the native samplers. It checks comfyui version and makes changes based on that to give the intended results. I've seen other people get good results without it, but worth testing. 

Please let me know.


> **Comment by afkaf (2026-08-14T18:26:27Z):**

@ruashots  I've verified again that #16 has fixed this issue. Please update to the latest changes. These changes implement a lookup from `adaln_t_table` directly from the model's buffers.  

You said "The addend's rows come from
`shared["silu_temb"]`, built by `_interp_egrid` from `_unique_t`".
This was true before the latest changes. Please update and let me know what you find. 



### #26 [open] PR: Fix NaN / black frames on MPS in the pruned-base AdaLN injection
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/26  by seungjulee 2026-08-13T20:15:24Z

Fixes #14.

## Summary

On Apple Silicon, `MiniMaxH3TurboLoRA` produces **black frames and NaN audio** on every
render against a pruned / curve-mode base, surfacing as
`av.error.ArgumentError: 'avcodec_send_frame()' returned 22` when the AAC encoder rejects
the NaN samples.

The NaN appears on the **first forward** of the runtime AdaLN injection. On a curve-mode
base the bf16 LoRA operands are upcast to fp32 (because `x` is fp32 there), and that fp32
path is what breaks on MPS. Computing the update in the LoRA's own dtype avoids it.

## Verified on M1 Max and M5 Max

Reproduced and fixed with a [one-command script](https://gist.github.com/seungjulee/01b2adcc074b7b9c9424706fd1499afe)
(256x256, 22 frames, 6 steps, pruned INT8 base), each run against a no-LoRA control:

| Chip | RAM | macOS | ComfyUI | bug reproduces | patch fixes |
|---|---|---|---|---|---|
| M1 Max | 64 GB | 26.5.1 | 0.33.0 | yes | **yes** |
| M5 Max | 128 GB | 26.6.1 | 0.32.0 + 0.33.0 | yes | **yes** |
| M5 Max (#14 reporter, pruned **BF16** base) | 64 GB | - | 0.31.1 | yes | not tested |

M1 Max (2021) and M5 Max (2025) bracket the Apple Silicon range. Also independent of
resolution (256x256 through 416x736 / 192 frames), of `low_vram`, of strength, of step
count, and of the LoRA checkpoint (v4-600 and v1-850).

On each machine a **no-LoRA control** was run first to confirm plain H3 samples cleanly
there, so the LoRA results are interpretable rather than masking an unrelated failure.

Two environment notes found while validating, in case they explain other Mac reports:

- **`aten::_int_mm` is not implemented for MPS**, so the INT8 checkpoints cannot run on
  Apple Silicon with stock ComfyUI + PyTorch at all - they need
  [`ComfyUI-AppleSilicon-FP8`](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)
  or an equivalent shim. Without it you get an immediate `NotImplementedError`, not this bug.
- H3 needs **more than 36 GB** of unified memory: the weights alone are ~37 GB, so a
  36 GB Mac OOMs regardless of resolution.

## Why the fp32 upcast is there

`comfy/ldm/minimax/model.py`:

```python
"adaln_dtype": torch.float32 if self.use_adaln_curves else dtype
```

| Base | `x.dtype` in `_make_adaln_forward` | compute dtype today |
|---|---|---|
| full / non-pruned | `dtype` (bf16) | **bf16** |
| pruned / curve-mode | **fp32** | **fp32** <- only here |

The fp32 compute is incidental - `a.to(x.device, x.dtype)` matches `x` so the add lines
up, not because fp32 was wanted. Non-pruned bases already compute this product in bf16 on
every backend, and `a`, `b` and the shipped `h3_silu_temb_grid.safetensors` are all bf16
on disk, so the upcast added no information. This patch makes the pruned path agree with
the path that already ships, and skips upcasting the largest operand in the expression
(`out` is ~96k rows).

## What is and isn't established

**Established.** Removing the fp32 upcast fixes it deterministically on both machines.
So does inserting *any* reduction (`.sum()`, `.abs().max()`) over the update, or over the
result of the add. All three were tested side by side on M1 Max against the unpatched
control in the same run:

```
none             BUG REPRODUCED (control)
reduce_delta     PATCH FIXES IT
reduce_result    PATCH FIXES IT
bf16_native      PATCH FIXES IT      <- this PR
```

Three independent interventions all clear it, which is why I am fairly confident the
diagnosis is in the right place even though the mechanism is not pinned down.

**Not established: the mechanism.** I originally thought the fp32 matmul returned NaN, and
that was wrong. Capturing the operands and both intermediate results with elementwise
copies (proven not to act as a barrier here) and reading them back after sampling shows
everything finite:

```
a: nan=False absmax=0.0221    b: nan=False absmax=0.0078    st: nan=False absmax=0.2559
inner (av @ sv.T):  nan=False absmax=0.00528
outer (bv @ inner): nan=False absmax=0.000172
x (base.linear):    nan=False absmax=3.08
```

The update is ~1e-4 against `x` ~3.08, far too small to legitimately produce NaN. So the
values are correct when materialised, yet the render is NaN - which I cannot explain. It
also does **not** reproduce in isolated PyTorch: the same expression at the same shapes and
dtypes, under `inference_mode`, with 51 concurrent modules and deferred NaN checks, gives
zero NaN over hundreds of iterations. It only appears inside the live DiT graph.

I'm therefore proposing this as a fix that is independently justified (consistency with
the non-pruned path, one less upcast) and empirically solid, not as a root-cause
diagnosis. Happy to keep digging if you have a theory.

## Ruled out

| Hypothesis | Test | Result |
|---|---|---|
| LoRA checkpoint | v4-600 and v1-850 | both fail |
| bypass vs merge | `low_vram=True` | fails |
| Third-party MPS accel node | `ComfyUI-AppleSilicon-FP8` fully removed | fails |
| In-place add in `_FrugalLoRA` | out-of-place `torch.add` | fails |
| Non-contiguous operand | `.T.contiguous()` | fails |
| Associativity / output transpose | `(sv @ av.T) @ bv.T` | fails |
| GEMV vs GEMM dispatch | pad inner result to N=2 | fails |
| Operand lifetime | keep `av`,`bv`,`sv` alive past the forward | fails |
| Host readback | `_d.view(-1)[0].item()` | fails |
| Queue ordering | `torch.mps.synchronize()` | fails |
| Materialising copy | `.clone()`, `torch.add(out=)`, `* 1.0` | all fail |
| Memory pressure / size | 256x256 -> 416x736 @ 22/124/192 frames | fails at every size |

## Verification

On M5 Max with the patch: v4-600 at strength 1.0, v1-850, `low_vram=True`, strength 0.8,
8 steps, and 416x736 / 124 frames all produce valid video with 32 kHz audio
(`denoised_rms` finite throughout). On M1 Max: verified via the candidate table above.

Against the fp32 path forced to work with a `.sum()` barrier, same seed: both give valid,
comparable output. Not bit-identical (PSNR ~22 dB) because diffusion amplifies any
numerical change into a different trajectory - the same as any kernel swap. Neither is
degraded.

## Other backends

The fast path is gated to MPS, so CUDA and ROCm are unchanged. I only have Apple Silicon
and did not want to alter numerics on backends I can't test. Worth noting that #21 (ROCm,
pruned FP8) reaches a *row-count* `RuntimeError` at this same line, which means the fp32
product executes fine there - so this looks MPS-specific rather than general.

If you'd rather have one dtype policy everywhere, dropping the `x.device.type == "mps"`
clause makes the pruned path match non-pruned on all backends and removes the upcast
entirely. Both are verified working on M5 Max; I chose the gated one for the smaller
blast radius and am happy to switch.

## Relation to the open PRs

#12, #16, #19 and #25 all address a different bug in this area - the row-count mismatch in
`_unique_t` / `_inject_adaln_egrid` with audio/video references, which surfaces as
`RuntimeError: The size of tensor a (3) must match the size of tensor b (2)`. This changes
only the operand dtype and is orthogonal.

#16 is the only other one touching `_make_adaln_forward`; it adds a
`if st is not None and st.shape[0] == x.shape[0]:` guard above the cast block and leaves
the casts unchanged, so a textual conflict is possible but trivial. Happy to rebase on
whichever lands first.



### #25 [open] PR: Fix adaln row mismatch for video refs and video soundtracks
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/25  by nitace666 2026-08-13T07:53:01Z

PR #10 aligned _unique_t with the model for image and standalone audio refs, but reference videos are still missing:

- a ref video (kind \ideo\/\ideo_audio\) emits \ef_img\ segments in the model's PackedLayout, so the model keeps the visual-cond row \max(t_v, 0.999)\; \_unique_t\ only counted kind \image\, so it dropped the row and the injected silu(t_emb) came up short - sampling crashes on the first step with `The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0`
- a video soundtrack (kind \ideo_audio\ with \ef_audio_t > 0\) emits \ef_audio\ segments, needing the \max(t_a, 1.0)\ row that was only added for kind \^Gudio\

This extends the two conditions to the full PackedLayout kind table: \image\/\ideo\/\ideo_audio\ for the visual row, \^Gudio\/\ideo_audio\ with \ef_audio_t > 0\ for the audio row.

Verified against a real PackedLayout over 7 reference combinations x 6 sigmas: upstream fails 13/49 (every video or video_audio case), this branch matches the model 49/49.


### #24 [closed] ISSUE: Reference/Tensor mismatch errors still exist. Check PR #16
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/24  by afkaf 2026-08-13T04:22:24Z

@Larryvrh  Your recent merge of pull request #10 was not a full solution to reference/tensor mismatch errors.
Please check pull request #16 for a solution to this issue.

The changes perform a lookup from `adaln_t_table` in the model's buffers to build the grid rows instead of reading the precomputed `shared["silu_temb"]`.

This change works for any number/combination of references up to the max allowed for each type.


### #23 [closed] PR: Refactor audio condition check to include video_audio key
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/23  by GiusTex 2026-08-12T22:39:28Z

Adds support for video soundtrack labeled `video_audio` at https://github.com/Comfy-Org/ComfyUI/blob/62b3c94bd45154f6486c7abf1b9efcacee96ea69/comfy_extras/nodes_minimax_h3.py#L265

> **Comment by afkaf (2026-08-13T04:09:58Z):**

#16 fixes all issues with references in any situation by performing a lookup from `adaln_t_table` to build the grid rows. It is pulled from the model's buffers instead of reading the precomputed `shared["silu_temb"] `
Please check #16 for a final solution to these reference/tensor mismatch issues.


> **Comment by GiusTex (2026-08-13T10:04:02Z):**

> #16 fixes all issues with references in any situation by performing a lookup from `adaln_t_table` to build the grid rows. It is pulled from the model's buffers instead of reading the precomputed `shared["silu_temb"] ` Please check #16 for a final solution to these reference/tensor mismatch issues.

Tested, it works with `video_audio` too 👍🏻 and I get the same video/audio outputs, thanks for the suggestion 



### #22 [closed] PR: Support for ComfyUI MiniMaxH3 "video_audio" (video soundtrack)
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/22  by GiusTex 2026-08-12T22:15:10Z

Adds support for video soundtrack which ComfyUI calls "video_audio" (https://github.com/Comfy-Org/ComfyUI/blob/62b3c94bd45154f6486c7abf1b9efcacee96ea69/comfy_extras/nodes_minimax_h3.py#L265) instead of just "audio"


### #21 [closed] ISSUE: Pruned-base AdaLN injection fails with audio-conditioned H3 continuation
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/21  by boxwrench 2026-08-12T21:54:18Z

### Summary

When MiniMax H3 Turbo's pruned-base AdaLN injection is used with a continuation conditioning payload containing both visual and audio references, the first continuation sampler call can fail with a batch-row mismatch:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
at non-singleton dimension 0
```

### Environment

- Turbo repository commit: `55fee86`
- ComfyUI: `0.32.0`
- Python: `3.12.3`
- PyTorch: `2.9.1+rocm7.2.1`
- ROCm: `7.2.1`
- GPU: AMD Radeon AI PRO R9700 (`gfx1201`)
- Base: pruned FP8 MiniMax H3
- LoRA: `minimax_h3_turbo_v4_step600_ema.safetensors`
- Native `ModelSamplingAV`
- 864x480, 243 H3 frames, 6 steps
- Continuation suite: Herrgotts H3 Infinite Continuation Suite `4b0c073`

### Cause

The pruned-base AdaLN time-grid helper only inferred the extra visual-conditioning timestep. The continuation payload also creates a `ref_audio` segment, so H3 builds three timestep rows (video, audio, conditioning) while the Turbo adapter builds two. The failure occurs in the injected AdaLN projection:

```
x = x + (bv @ (av @ sv.T)).T
```

### Reproduction

1. Use `MiniMaxH3TurboLoRA` with a pruned H3 base and `MiniMaxH3TurboSampler`.
2. Provide a continuation conditioning payload with visual keyframes and an audio reference.
3. Queue a second continuation clip.
4. The adapter's AdaLN injection can fail with the 3-versus-2 tensor mismatch.

### Workaround / proposed fix

Include `ref_audio` when deriving the unique timestep rows, using the payload's `audio_cond_noise_aug` value. I tested this locally; the same three-clip run then completed all three Turbo samplings and the final stitch successfully.

> **Comment by microvswind (2026-08-13T03:14:24Z):**

me too,after i replace the 

<img width="965" height="433" alt="Image" src="https://github.com/user-attachments/assets/7759d8b8-2914-4132-ac36-6c18855ce989" />

<img width="976" height="370" alt="Image" src="https://github.com/user-attachments/assets/87b41510-1ecb-4517-9af3-232be11687d9" />

<img width="1346" height="967" alt="Image" src="https://github.com/user-attachments/assets/7dda6843-ff96-4fd0-8a90-d331f15e0622" />


> **Comment by afkaf (2026-08-13T03:29:32Z):**

If he would accept and merge my changes, it should fix all issues with references.
I have multiple people using my changes successfully without issues.
Please check pull request #16 



> **Comment by microvswind (2026-08-13T04:01:15Z):**

> If he would accept and merge my changes, it should fix all issues with references. I have multiple people using my changes successfully without issues. Please check pull request [#16](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/16)

<img width="2411" height="1338" alt="Image" src="https://github.com/user-attachments/assets/41c70ea9-0a87-4d36-ae6d-c3a017c1e109" />
i use your init.py,got same error:

<img width="771" height="511" alt="Image" src="https://github.com/user-attachments/assets/dc05d337-6b12-4c58-983f-269fd7e0a67a" />


> **Comment by afkaf (2026-08-13T05:39:35Z):**

> > If he would accept and merge my changes, it should fix all issues with references. I have multiple people using my changes successfully without issues. Please check pull request [#16](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/16)
> 
> <img alt="Image" width="2000" height="1110" src="https://private-user-images.githubusercontent.com/103630540/635248965-41c70ea9-0a87-4d36-ae6d-c3a017c1e109.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODY1OTg2ODIsIm5iZiI6MTc4NjU5ODM4MiwicGF0aCI6Ii8xMDM2MzA1NDAvNjM1MjQ4OTY1LTQxYzcwZWE5LTBhODctNGQzNi1hZTZkLWMzYTAxN2MxZTEwOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxM1QwNTE5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNDU0OGIyNDg4NWNiMDM1ZGRiMzY5ZjRkMTM1N2Q2MDllNWRlMDQwMjFlNjE4YmY1ZDNhNTIxNzcxNWE2YjIwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.-RNPUnEKUHrqIq3TTQyPT9LJx8R8YFaND-TbEi5rpUs"> i use your init.py,got same error: <img alt="Image" width="771" height="511" src="https://private-user-images.githubusercontent.com/103630540/635250168-dc05d337-6b12-4c58-983f-269fd7e0a67a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODY1OTg2ODIsIm5iZiI6MTc4NjU5ODM4MiwicGF0aCI6Ii8xMDM2MzA1NDAvNjM1MjUwMTY4LWRjMDVkMzM3LTZiMTItNGM1OC05ODNmLTI2OWZkN2UwYTY3YS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxM1QwNTE5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hNzg5YzU3MDRjNDg2ZjcyMGM1NjU4MWRlNTFjOGFiNDdlMzk4MzkxZTYxNDllMWMyMzI0NWViZDJkMTJjMmRkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.h_xkpjO3acweuY1gI1GOdO7nanxlcboPme-ZkQPT1hY">

@microvswind  Maybe you didn't actually run it using my changes? Make sure you actually restarted comfyui after applying my changes to the __init__.py file.
I just attempted to reproduce your issue with the Herrgotts H3 Infinite Continuation Suite with my changes applied. For the clip 1, I generated with an audio reference attached to ref_audio_0 and two image references.
I extracted the last frame of the generated video and moved to the continuation workflow. I used clip 1 as the continuation payload. It generated successfully with no issues.
The second clip contained the tail of my audio from clip 1. I attempted the same thing with the Showcase_Autoswitch workflow with the start node replaced with a reference node like before. Everything completed without any errors.


My changes are a complete fix to the reference/tensor mismatch errors. There are no issues with my pull request #16 
I have replaced the hardcoded grid with a lookup directly from the model buffers.


> **Comment by microvswind (2026-08-13T06:08:33Z):**

> > > If he would accept and merge my changes, it should fix all issues with references. I have multiple people using my changes successfully without issues. Please check pull request [#16](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/16)
> > 
> > 
> > <img alt="Image" width="2000" height="1110" src="https://private-user-images.githubusercontent.com/103630540/635248965-41c70ea9-0a87-4d36-ae6d-c3a017c1e109.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODY1OTg2ODIsIm5iZiI6MTc4NjU5ODM4MiwicGF0aCI6Ii8xMDM2MzA1NDAvNjM1MjQ4OTY1LTQxYzcwZWE5LTBhODctNGQzNi1hZTZkLWMzYTAxN2MxZTEwOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxM1QwNTE5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNDU0OGIyNDg4NWNiMDM1ZGRiMzY5ZjRkMTM1N2Q2MDllNWRlMDQwMjFlNjE4YmY1ZDNhNTIxNzcxNWE2YjIwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.-RNPUnEKUHrqIq3TTQyPT9LJx8R8YFaND-TbEi5rpUs"> i use your init.py,got same error: <img alt="Image" width="771" height="511" src="https://private-user-images.githubusercontent.com/103630540/635250168-dc05d337-6b12-4c58-983f-269fd7e0a67a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODY1OTg2ODIsIm5iZiI6MTc4NjU5ODM4MiwicGF0aCI6Ii8xMDM2MzA1NDAvNjM1MjUwMTY4LWRjMDVkMzM3LTZiMTItNGM1OC05ODNmLTI2OWZkN2UwYTY3YS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwODEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDgxM1QwNTE5NDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hNzg5YzU3MDRjNDg2ZjcyMGM1NjU4MWRlNTFjOGFiNDdlMzk4MzkxZTYxNDllMWMyMzI0NWViZDJkMTJjMmRkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.h_xkpjO3acweuY1gI1GOdO7nanxlcboPme-ZkQPT1hY">
> 
> [@microvswind](https://github.com/microvswind) Maybe you didn't actually run it using my changes? Make sure you actually restarted comfyui after applying my changes to the **init**.py file. I just attempted to reproduce your issue with the Herrgotts H3 Infinite Continuation Suite with my changes applied. For the clip 1, I generated with an audio reference attached to ref_audio_0 and two image references. I extracted the last frame of the generated video and moved to the continuation workflow. I used clip 1 as the continuation payload. It generated successfully with no issues. The second clip contained the tail of my audio from clip 1. I attempted the same thing with the Showcase_Autoswitch workflow with the start node replaced with a reference node like before. Everything completed without any errors.
> 
> My changes are a complete fix to the reference/tensor mismatch errors. There are no issues with my pull request [#16](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/16) I have replaced the hardcoded grid with a lookup directly from the model buffers.

works,thx so much,i am not sure what i missed before,it works after i replace my init.py again and relaunch comfyui,work.



### #20 [closed] ISSUE: MiniMaxH3TurboSampler aborts generation on BrokenPipeError from progress logging
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/20  by boxwrench 2026-08-12T21:29:40Z

### Summary

A valid H3 Turbo generation can abort when the ComfyUI process inherits a stdout pipe that later closes. The failure is raised by the sampler's diagnostic `print(..., flush=True)`, rather than by model inference.

### Environment

- Repository commit: `55fee86`
- ComfyUI: `0.32.0`
- Python: `3.12.3`
- PyTorch: `2.9.1+rocm7.2.1`
- ROCm: `7.2.1`
- GPU: AMD Radeon AI PRO R9700 (`gfx1201`)
- Native `ModelSamplingAV` path
- 864x480, 243 H3 frames, 6 sampler steps

### Observed traceback

The generation reached the second continuation clip and then failed in `SamplerCustomAdvanced`:

```
File ".../ComfyUI-MiniMax-H3-Turbo/__init__.py", line 118, in _turbo_sampler
    print(..., flush=True)
File ".../comfy/app/logger.py", line 73, in flush
    BrokenPipeError
```

There was no OOM or GPU fault.

### Reproduction

1. Run ComfyUI with `MiniMaxH3TurboSampler` under a launcher or terminal whose stdout pipe can close.
2. Queue a Turbo H3 video generation.
3. Close or detach the parent output pipe while sampling is active.
4. The sampler's progress print raises `BrokenPipeError` and aborts the workflow.

### Expected behavior

Progress logging should be non-fatal; sampling should continue if stdout is unavailable.

### Possible fix

Route progress output through ComfyUI's logger, or catch or disable `BrokenPipeError` around sampler diagnostics. Redirecting stdout/stderr to a persistent file or keeping the server in a persistent terminal is a workaround.


### #19 [open] PR: Fix Turbo LoRA audio conditioning rows
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/19  by jjdejong 2026-08-12T13:30:06Z

## Summary

Turbo LoRA conditioning rows were derived without accounting for packed audio-reference segments. This caused later passes with Motion Context audio conditioning to fail with a timestep-row shape mismatch.

This fix derives the visual and audio conditioning flags from H3's packed layout, including fallback handling for legacy payloads, and adds regression coverage for normal, visual, audio, combined, and fallback conditioning.

## Testing

- /home/jjj/ComfyUI/venv/bin/python tests/test_adaln_timestep_rows.py
- /home/jjj/ComfyUI/venv/bin/python -m py_compile __init__.py tests/test_adaln_timestep_rows.py


### #18 [open] ISSUE: I used the latest version, but the audio always produces strange words
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/18  by xuchen-dev 2026-08-11T11:20:54Z

https://github.com/user-attachments/assets/c16f2da0-7f32-4cbf-9f00-d2198b0fef12

<img width="1200" height="808" alt="Image" src="https://github.com/user-attachments/assets/13ff69f3-28c1-4e62-a637-e73e96ce9021" />

subject_definitions:
<Subject 5> is Zhou Kui, the wily antique-smuggling ring leader in a round-spectacled horse-robe vest and a felt hat, from <Picture 1>.
<Subject 3> is Er Zhu, the chunky henchman in a grey vest carrying a sledgehammer, from <Picture 2>.
<Subject 4> is San Pang, the burly, stockier brother in a dark waistcoat carrying a heavy club, from <Picture 3>.
<Picture 1> is the character lock reference for <Subject 5>.
<Picture 2> is the character lock reference for <Subject 3>.
<Picture 3> is the character lock reference for <Subject 4>.

summary:
[reference generation] In a shadowed warehouse across town, Zhou Kui shuts off the search order and studies a rough map where a mark now sits on Su Yan's quarter. His men, the club-bearer and the hammer-man, lean in as he sets down the pipe: the craftsman has slipped them once in the print-shop night, but mirrors worth a fortune do not stay hidden — and Zhou Kui means to be the one waiting when they surface. He gives the order, and the rough men move out into the city's dusk, their net beginning to close on the same lane the uncle has his eye on.

retention_analysis:
<Subject 5> (appears in [Shot 1], [Shot 2]): fully_preserved - Zhou Kui's horse-robe vest, felt hat, and cold, level command are retained.
<Subject 3> (appears in [Shot 1], [Shot 2]): fully_preserved - Er Zhu's grey vest and heavy sledgehammer are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - San Pang's dark waistcoat and heavy club are retained.

detailed_description:
The target video is a cinematic live-action scene in a shadowed warehouse at dusk, the ring's leader turning his attention home onto the craftsman's quarter.
[Shot 1] A medium shot frames <Subject 5>, Zhou Kui, the ring leader from <Picture 1>, at a table with a rough map, a mark freshly set on one quarter of the city. <Subject 3>, Er Zhu from <Picture 2>, and <Subject 4>, San Pang from <Picture 3>, lean in at his shoulders. <Subject 5> (S5) taps the mark with his pipe stem, low, <d>[Chinese] 那小子在印局戏耍了我们一回，还顺走了一面宝镜。他以为躲得过去</d>
[Shot 2] At 00:07.000, the camera cuts to a closer shot as <Subject 5> (S5) sets the pipe down and gives the order, cold and flat, <d>[Chinese] 从他的住处开始，一处一处摸。别打草惊蛇，找着人、找着镜，一并带回来见我。</d> The two henchmen grunt and move out into the dusk; outside the warehouse, the city darkens toward the same lane the uncle watches.

overall_soundscape:
The low ring of a pipe against wood, the rasp of cloth armor-like vests, low grunts of assent, and the distant dusk noises of the city.

non_diegetic_music:
A low, gathering menace on drums and low brass, the ring's machinery of pursuit turning cold as its focus tightens on the craftsman.

> **Comment by SveSop (2026-08-11T13:32:03Z):**

Quite an elaborate prompt... For me, the chineese is "strange words", but i assume you mean the other way around.

Hot tip: From what i have seen so far, timestamps is kind of super-important. So lets say you have [Shot 1] starting at 00:00.000, and [Shot 2] starting at 00:07.000, you MUST fill what happens in those 7 seconds. If you input a 3 second dialogue in that timespan, without detailed description of what happens during those 7 seconds (including the speech), the model WILL invent random words for the character to speak.

So, "timing is everything" tbh.



### #17 [open] ISSUE: custom turbo lora loader broken after comfyui update
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/17  by BaseMe2 2026-08-10T19:49:17Z

Kijai patched a desirable speedup for VAE encode/decode, which is in the latest nightly. 

https://github.com/Comfy-Org/ComfyUI/pull/15446

This causes turbo lora loader to not work properly during interference anymore. 

H3TURBO fwd pruned/bypass] call#6  qkv_proj.forward_owner=BypassForwardHook (BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected=False  timestep=800.00  video_rms=0.8297 audio_rms=0.7140 dtype=torch.bfloat16

I tried using a regular lora loader, but this obviously doesn't work. Cheers!



### #16 [closed] PR: Fix modulation segment calculation with lookup from `adaln_t_table` to resolve image/video/audio reference issues.
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/16  by afkaf 2026-08-10T17:18:47Z

This fix enables use of image + audio + video references with the ref2va model. Up to the max number of refs allowed for each type. The other open pull requests did not make a full fix to this issue. This is a complete fix.

`_make_adaln_forward` — added `table=None, egrid=None` params, and inside `forward` do the table lookup to build the rows instead of reading the precomputed `shared["silu_temb"]`

also added a `st.shape[0] == x.shape[0]` check before the add, so if anything's ever still mismatched it just skips the delta for that step instead of hard crashing

`_inject_adaln_egrid` — find `adaln_t_table` in the model's buffers, pass it + the grid into `_make_adaln_forward`

> **Comment by GiusTex (2026-08-13T10:06:58Z):**

Tested, it works with `video_audio` key too and I get the same video/audio outputs I got using https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/23 👍🏻



### #15 [open] ISSUE: BabyloraH3 5gb size
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/15  by dadix4e 2026-08-10T15:17:53Z

Hello , I want to thank you for the work you have done.


### #14 [open] ISSUE: Doesn't work on Apple silicon
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/14  by Rhystic1 2026-08-10T08:42:24Z

M5 Max, 64 GB, MPS, PyTorch 2.12.1, ComfyUI 0.31.1

- Base: pruned BF16
- LoRA: minimax_h3_turbo_v4_step600_ema, strength 1.0, bypass
- call#1 finite RMS → call#2 video_rms=nan audio_rms=nan
- Spectrum disabled, same failure

Every generation then fails with error: `av.error.ArgumentError: Invalid argument: 'avcodec_send_frame()' returned 22`


### #13 [open] ISSUE: Will the training method be made public?
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/13  by SolitaryManF 2026-08-10T02:23:42Z

H3-Turbo Lora is trained with DMD-based distillation method?


### #12 [closed] PR: Fix ref-audio crash on pruned base: missing ref_audio timestep row in adaln injection
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/12  by nimblecloud13 2026-08-10T00:31:10Z

## Bug

On a pruned base (`use_adaln_curves`), any workflow with **reference audio** crashes at sampling start:

```
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```

raised at the adaln injection add (`x = x + (bv @ (av @ sv.T)).T`).

## Cause

`_unique_t` predicts the model's unique-timestep set to size the runtime adaln correction, but only models `{t_v, t_a}` plus the visual-cond row. Core ComfyUI (`comfy/ldm/minimax/model.py`) additionally pins a `ref_audio` row at `max(t_a, audio_cond_noise_aug)` whenever reference-audio segments are present — so the model builds a 3-row `t_emb` while the injected correction has 2 rows.

The payload probe had a second, silent issue: `bool(payload.get("keyframes") or payload.get("refs"))` counts **audio-only** refs as visual conditioning, injecting the third row at the wrong timestep (0.999 instead of 1.0) — same cardinality by luck, so no crash, but wrong time-conditioning.

## Fix

Mirror the core model's set construction exactly:

- classify ref blocks by kind the way `PackedLayout` does: keyframes -> `cond`, `image`/`video`/`video_audio` -> `ref_img`, audio rows (standalone or attached to a video ref) -> `ref_audio` only when `ref_audio_t > 0`;
- add the pinned `ref_audio` row to `_unique_t`;
- honor the `visual_cond_noise_aug` / `audio_cond_noise_aug` payload overrides.

## Verification

Replicated the core unique-set construction and compared with the patched `_unique_t` across: t2va (2 rows), keyframes (3), image ref (3), audio ref (3 — the crash case), zero-length audio ref (2), video+audio ref (4), keyframes+audio ref (4). Row counts and values match core in all cases; the bundled E-grid interpolation handles the new `t=1.0` row exactly (upper grid edge).

🤖 Generated with [Claude Code](https://claude.com/claude-code)



### #11 [open] ISSUE: AttributeError: 'dict' object has no attribute 'model' in apply_lora (then hangs after manual dict-unwrap patch)
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/11  by ksankstemp-dotcom 2026-08-09T03:11:39Z

**Environment**
- ComfyUI version: 0.28.0
- OS: Windows
- Python: 3.11.9
- PyTorch: 2.12.0.dev20260408+cu128
- Node pack version tested: 1.2.3 and 1.2.2 (same result on both)
- Diffusion model: minimax_h3_f12va_pruned_bf16.safetensors loaded via core UNETLoader
- LoRA: minimax_h3_turbo_4step.safetensors, strength 1.0, low_vram off
- MiniMax H3 Turbo Sampler: 4 steps, scheduler "simple"

**Steps to reproduce**
1. Load MiniMax H3 model via the standard UNETLoader node.
2. Add MiniMaxH3TurboLoRA right after UNETLoader, before any other model-patching nodes (Sage Attention, EasyCache, TorchCompileModel are all present downstream, bypassed).
3. Add MiniMaxH3TurboSampler feeding into SamplerCustomAdvanced's sampler input, with BasicScheduler set to 4 steps.
4. Queue the prompt.

**Actual behavior**
apply_lora throws:
AttributeError: 'dict' object has no attribute 'model'
  File ".../custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 467, in apply_lora
    dm = model.model.diffusion_model

This happens on both v1.2.2 and v1.2.3, so it doesn't appear to be a regression between those two versions -- the model input arriving at this node is a plain dict, not a ModelPatcher.

**What I tried**
As a workaround I patched apply_lora to unwrap the dict:

def apply_lora(self, model, lora_name, strength, low_vram=False):
    if isinstance(model, dict):
        model = model.get("model", model)
    ...

This clears the AttributeError, but the workflow then hangs indefinitely at 0% inside SamplerCustomAdvanced -- it never progresses and never errors out. My guess is the dict actually bundles more than just the video model (e.g. an audio sub-model or config needed by the turbo sampler), and naively unwrapping to dict["model"] drops something the sampler needs, causing it to stall rather than crash.

**Question**
What is the expected structure of the model object ComfyUI hands to apply_lora for MiniMax H3 in the joint video+audio (FL2VA) setup? Is there a supported way to load/wrap the model so it arrives as the ModelPatcher this node expects, or does apply_lora need to be updated to handle the dict-wrapped case correctly (unpacking all required components, not just model)?

Happy to provide the full workflow JSON or more logs if useful.


### #10 [closed] PR: Fix adaln injection row mismatch with audio reference conditioning
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/10  by bkunbargi 2026-08-08T22:32:09Z

Fixes #3, fixes #9.

## Symptom

Any workflow with an audio reference (`ref2va` with `ref_audios`, or `fl2va` chained with reference audio) crashes on the first sampler step:

```
File ".../ComfyUI-MiniMax-H3-Turbo/__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```

## Root cause

The curve-mode adaln injection adds `B @ A @ silu(t_emb)` with **one row per unique timestep**, so `_unique_t` must reproduce exactly the unique-t row set the model builds in its forward (`comfy/ldm/minimax/model.py`). It had drifted from the model in three ways:

1. **No `ref_audio` row.** An audio reference makes the model add a `max(t_a, aud_aug)` row to `unique_t`, so the model's `t_emb` has 3 rows while the injected delta has 2 → the crash above.
2. **Vis-cond detection treated any `refs` entry as visual** (`bool(keyframes or refs)`), so once the crash is fixed, a pure audio ref would have added a spurious visual row (the inverse mismatch). The model only adds the visual row for `cond`/`ref_img` segments.
3. **Value drift:** the payload's `visual_cond_noise_aug` / `audio_cond_noise_aug` overrides were ignored in favor of a hardcoded `0.999`, and `t_v`/`t_a` were recomputed in float64 while the model dedups in float32 — so the `t_v == t_a` collapse could in principle disagree at schedule edges.

## Fix

`_unique_t` now takes the `minimax_payload` and mirrors the model's computation one-for-one: same float32 tensor arithmetic, keyframes / image refs → visual cond row, audio ref with `ref_audio_t > 0` → audio cond row, and the payload aug overrides are respected. Since both sides build the identical sorted set, the delta rows also stay **aligned** with the model's `t_row` indices — which is why this is done at the source rather than padding/truncating the delta at the add site as proposed in #9 (repeating the last row would apply the wrong correction row to the ref segments whenever the missing value doesn't sort last).

## Testing

Parity-tested `_unique_t` against a verbatim replication of the model's `unique_t` computation across 84 combinations (12 sigmas including the turbo schedule values and edges × 7 conditioning payloads: none, keyframes, image ref, audio ref, `ref_audio_t=0`, image+audio, keyframe+audio with custom aug overrides) — identical row sets, values, and order in all cases. Also confirmed t2v / fl2va-without-audio behavior is unchanged (same rows as before for those payloads).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

> **Comment by PeterMoras (2026-08-09T02:19:50Z):**

I've tested the pr and can confirm that it does fix the audio issue I had that was identified in issues #3 and #9.


> **Comment by GiusTex (2026-08-11T14:45:49Z):**

Fix for ComfyUI MiniMaxH3 "video_audio" (video soundtrack): https://github.com/bkunbargi/ComfyUI-MiniMax-H3-Turbo/pull/1/changes/ffa617a838eb4a3a7716237d7ac2c4182eb47045

Comfyui supports video soundtrack as "ref_video_audio_0" in "MiniMax H3 Reference to Video" node, as seen here: https://github.com/Comfy-Org/ComfyUI/blob/62b3c94bd45154f6486c7abf1b9efcacee96ea69/comfy_extras/nodes_minimax_h3.py#L265 but labels it "video_audio". This pr adds support for it



### #9 [closed] ISSUE: Fix `RuntimeError` shape mismatch when using `ref2va` with audio inputs
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/9  by l33tm4st3r 2026-08-08T13:54:48Z

Hi!

I ran into a tensor shape mismatch error when running `MiniMaxH3ReferenceToVideo` with a standalone audio reference input (`ref_audios`).


### Issue

When passing audio conditioning along with reference images into `ref2va`, `x.shape[0]` becomes 3 ($M=3$). During the forward pass, line 235 in `__init__.py` throws a `RuntimeError`:

```text
File "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

```

### Cause

The Turbo LoRA patch matrix `(bv @ (av @ sv.T)).T` evaluates to $M=2$, but `x` carries 3 batch/modulation segments when audio is present, causing element-wise addition to fail on dimension 0.

### Proposed Fix

Dynamically matching `delta.shape[0]` to `x.shape[0]` before addition resolves the shape mismatch and allows generation to complete smoothly with audio enabled.

In `ComfyUI-MiniMax-H3-Turbo/__init__.py` around line 235:

```python
# Replace: x = x + (bv @ (av @ sv.T)).T

delta = (bv @ (av @ sv.T)).T
if delta.shape[0] != x.shape[0]:
    if delta.shape[0] < x.shape[0]:
        repeat_count = x.shape[0] - delta.shape[0]
        delta = torch.cat([delta, delta[-1:].repeat(repeat_count, 1)], dim=0)
    else:
        delta = delta[:x.shape[0]]
x = x + delta

```

Tested on ROCm / PyTorch 2.x and confirmed it fixes the sampler crash. Let me know if you'd like a PR for this!

> **Comment by jjdejong (2026-08-12T13:37:32Z):**

This is addressed by PR #19:

https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/19

The fix derives the Turbo LoRA AdaLN timestep rows from H3's packed conditioning layout, including audio-reference segments, instead of padding/truncating the LoRA delta after the mismatch occurs. Regression coverage includes visual, audio, combined, and legacy fallback conditioning.



### #8 [closed] ISSUE: Turbo lora fails after ComfyUi update. Nodes broken
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/8  by tossedsalad111700 2026-08-07T22:33:09Z

After updating ComfyUI to the most recent stable version the turbo lora node fails. The error says it is in the sampler but I isolated it to the turbo lora loading node.
It seems ComfyUI changed how the MiniMax data is passed through the graph and it breaks the nodes way of applying the lora.

I reinstalled ComfyUI to verify where the error occurred.

## Error Details
- **Node ID:** 125
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0


### #7 [open] ISSUE: workaround for lora loader on comfy.org? help for managed env with no custom nodes
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/7  by annasophiachristianahahn 2026-08-07T11:29:28Z

Hello, I'd love to be able to use this within a managed environment like comfy.org that doesn't accept custom nodes- I've managed to get the models all imported and replaced the custom sampler with the version of Euler that's available in the latest comfy as you mentioned in the docs

The one thing I have not managed to replace is your custom Turbo LoRA loader.... is there any way to load the lora in an environment like comfy.org that does not tallow custom nodes?


### #6 [open] ISSUE: Plugin sampler causes audio corruption
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/6  by ucboys-cloud 2026-08-07T01:31:51Z

If I use this plugin’s sampler, the audio gets corrupted, but if I use the official recommended sampler, the audio is normal—though the image quality drops severely. A comparison can be made through dialogue prompts, and the difference is quite obvious.

> **Comment by goofyrodent (2026-08-07T09:43:40Z):**

It did work ~12 hours ago, but now audio is garbage.


> **Comment by ucboys-cloud (2026-08-07T09:58:52Z):**

> It did work ~12 hours ago, but now audio is garbage.大约12小时前还能用，但现在音频质量极差。

Since the official fix has been implemented, the sampling function of the plugin has become redundant. However, image quality is poor when generating outputs without the plugin.


> **Comment by Larryvrh (2026-08-07T10:19:49Z):**

should be fixed in the new version



### #5 [open] ISSUE: 1.2.0 + producing oversaturation and flashing.
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/5  by rskickflip1-cmd 2026-08-06T22:33:14Z

I am using the 850 ema lora with the provided workflow. on 1.2.0 and 1.2.2 (have not tried 1.2.1), generations end up oversaturated and often flashing effects. No such issue on 1.1.0.
I used a pruned minimax H3 FP8 diffusion model.
I am on the latest Comfyui Windows standalone/portable, 3060 Ti. torch 2.13cu130

> **Comment by rkfg (2026-08-06T23:01:01Z):**

Same, also low_vram on and off give vastly different results, they're not identical at all.


> **Comment by rskickflip1-cmd (2026-08-06T23:09:33Z):**

> Same, also low_vram on and off give vastly different results, they're not identical at all.

the toggle on the Turbo lora node introduced in 1.2.0(i think, but maybe 1.2.2).  In the tooltip, it does say the results will be different. I had it off in my testing (no low_vram mode).


> **Comment by rachidlamouchi (2026-08-07T00:54:54Z):**

Personally, I get better results with ckpt500.


> **Comment by rkfg (2026-08-07T09:14:43Z):**

> the toggle on the Turbo lora node introduced in 1.2.0(i think, but maybe 1.2.2). In the tooltip, it does say the results will be different. I had it off in my testing (no low_vram mode).

Yes, but I didn't expect it to be *that* much different. Had to lower the weight when running with low_vram **off** to 0.6 to get anything close to low_vram **on**. So I suppose weights merging (old approach and) works better than dynamic application (when low_vram is **off**). Still, both look worse than pre-1.2.0 behavior. I rolled back to that until it's fixed or improved.


> **Comment by hum-ma (2026-08-07T17:45:39Z):**

By flashing effects do you mean the "disco lights"? That's what I'm seeing in many generations with yesterday's version. Haven't tried the newest updates yet.


> **Comment by rskickflip1-cmd (2026-08-07T19:55:49Z):**

> By flashing effects do you mean the "disco lights"? That's what I'm seeing in many generations with yesterday's version. Haven't tried the newest updates yet.

both, so i will see flashing lights, but also the entire image will flash.
I think the most prominent occurrence was the saturation/light shift at the start of the video. It's happening on 1.1.0, but only noticeable on certain images.




### #4 [closed] ISSUE: OOM on a run that I normally don't OOM
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/4  by caustiq 2026-08-06T18:39:15Z

Megapixels selected was 0.6 for a 10 second video, system can normally handle this.

Using the latest turbo lora loader and sampler from this repos with the ckpt850 lora results in an OOM.

Dropping megapixels down to 0.5 allowed it to proceed. But again, normally I'd be able to go to 0.6 MP.
```
[MiniMaxH3TurboLoRA] pruned base: 208 backbone via bypass + 51 adaln injected at run time
[INFO] Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 156995 KB.
[H3TURBO sampler] sigmas=[1.0, 0.973, 0.9231, 0.8, 0.0]  x.shape=(1, 1, 4359744) dtype=torch.float32  v_numel=4333824 a_numel=25920  shapes=[torch.Size([1, 24, 72, 38, 66]), torch.Size([1, 32, 2, 405])]
  0%|                                                                                                                                                                                                                                                                                                                                                                                                                   | 0/4 [00:00<?, ?it/s][H3TURBO fwd pruned] call#1  qkv_proj.forward_owner=BypassForwardHook (BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected=False  timestep=1000.00  video_rms=1.0002 audio_rms=1.0038 dtype=torch.bfloat16
  0%|                                                                                                                                                                                                                                                                                                                                                                                                                   | 0/4 [00:08<?, ?it/s]
[ERROR] !!! Exception during processing !!! Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 7.30 GiB
Requested               : 2.48 GiB
Device limit            : 11.61 GiB
Free (according to CUDA): 14.19 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB
[ERROR] Traceback (most recent call last):
  File "/home/user/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/home/user/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy_api/latest/_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy_extras/nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 80, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/ldm/minimax/model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 296, in wrap
    return executor(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 233, in wrap
    return executor(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/ldm/minimax/model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/ldm/minimax/model.py", line 272, in forward
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
                                  ^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/ldm/minimax/model.py", line 192, in forward
    return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")
                                                ^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/weight_adapter/bypass.py", line 174, in _bypass_forward
    h_out = self.adapter.h(x, base_out)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/weight_adapter/lora.py", line 368, in h
    return out * scale
           ~~~~^~~~~~~
torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. (out of memory)
Currently allocated     : 7.30 GiB
Requested               : 2.48 GiB
Device limit            : 11.61 GiB
Free (according to CUDA): 14.19 MiB
PyTorch limit (set by user-supplied memory fraction)
                        : 17179869184.00 GiB

[INFO] Memory summary:
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   7476 MiB |  10013 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |   7476 MiB |  10013 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   7648 MiB |  10944 MiB |      0 B   |      0 B   |
|       from large pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|       from small pool |      0 MiB |      0 MiB |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

[ERROR] Got an OOM, unloading all loaded models.
[INFO] Prompt executed in 138.19 seconds
```

> **Comment by caustiq (2026-08-06T18:49:34Z):**

Saw this other exception too. This one happened after the 0.5 MP run succeeded and I went for another attempt at the 0.6 MP run (it did OOM again). `AttributeError: 'AdalnProj' object has no attribute 'base'`


```
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.12/threading.py", line 1010, in run
    self._target(*self._args, **self._kwargs)
  File "/home/user/ComfyUI/main.py", line 372, in prompt_worker
    e.execute(item[2], prompt_id, extra_data, item[4])
  File "/home/user/ComfyUI/execution.py", line 728, in execute
    asyncio.run(self.execute_async(prompt, prompt_id, extra_data, execute_outputs))
  File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/base_events.py", line 687, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/execution.py", line 789, in execute_async
    result, error, ex = await execute(self.server, dynamic_prompt, self.caches, node_id, extra_data, executed, prompt_id, execution_list, pending_subgraph_results, pending_async_nodes, ui_node_outputs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/execution.py", line 645, in execute
    comfy.model_management.unload_all_models()
  File "/home/user/ComfyUI/comfy/model_management.py", line 2045, in unload_all_models
    free_memory(1e30, device)
  File "/home/user/ComfyUI/comfy/model_management.py", line 881, in free_memory
    if memory_to_free > 0 and current_loaded_models[i].model_unload(memory_to_free):
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/model_management.py", line 803, in model_unload
    self.model.detach(unpatch_weights)
  File "/home/user/ComfyUI/comfy/model_patcher.py", line 1291, in detach
    self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
  File "/home/user/ComfyUI/comfy/model_patcher.py", line 2101, in unpatch_model
    self.partially_unload(None, 1e32)
  File "/home/user/ComfyUI/comfy/model_patcher.py", line 2011, in partially_unload
    backup_freed = self.restore_loaded_backups()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/model_patcher.py", line 1840, in restore_loaded_backups
    comfy.utils.set_attr_buffer(self.model, key, self.backup_buffers.pop(key))
  File "/home/user/ComfyUI/comfy/utils.py", line 945, in set_attr_buffer
    obj, name = resolve_attr(obj, attr)
                ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/comfy/utils.py", line 925, in resolve_attr
    obj = getattr(obj, name)
          ^^^^^^^^^^^^^^^^^^
  File "/home/user/ComfyUI/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1967, in __getattr__
    raise AttributeError(
AttributeError: 'AdalnProj' object has no attribute 'base'

```


> **Comment by abisssalina-droid (2026-08-06T19:01:22Z):**

@caustiq same error ping me if you find a fix for it , i cant even do a 0.3 mp 10 seconds run but norammy i was able to do upto 15 seconds on my 6gb potato


> **Comment by Larryvrh (2026-08-06T20:42:47Z):**

should be fixed in this new version! please give it a try


> **Comment by caustiq (2026-08-06T21:15:47Z):**

Fix confirmed!

Closing.


> **Comment by caustiq (2026-08-06T22:49:21Z):**

@Larryvrh might want to look into this more. New test I did: wasn't able to go to 0.8 MP with the turbo lora, but could without lora.



### #3 [closed] ISSUE: ref2va is broken if audio is used as a reference
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/3  by rkfg 2026-08-06T17:35:04Z

The error is:
```
  File "/app/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py", line 145, in forward
    x = x + (b @ (a @ st.to(a.device, x.dtype).T)).T          # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
```
An LLM proposed fix for this is like this:
```diff
diff --git a/__init__.py b/__init__.py
index 41df4fa..e21268a 100644
--- a/__init__.py
+++ b/__init__.py
@@ -104,13 +104,16 @@ def _egrid():
     return _EGRID
 
 
-def _unique_t(timestep, shift_v, shift_a, has_vis_cond):
+def _unique_t(timestep, shift_v, shift_a, has_vis_cond, has_aud_cond=False,
+              vis_aug=0.999, aud_aug=1.0):
     sv = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
     t_v = 1.0 - sv
     t_a = 1.0 - _time_shift_sigma(sv, shift_v, shift_a)
     s = {t_v, t_a}
     if has_vis_cond:
-        s.add(max(t_v, 0.999))
+        s.add(max(t_v, vis_aug))
+    if has_aud_cond:
+        s.add(max(t_a, aud_aug))
     return sorted(s)
 
 
@@ -205,8 +208,18 @@ class MiniMaxH3TurboLoRA:
             ts = args[1] if len(args) > 1 else kwargs.get("timestep")
             ctx = args[2] if len(args) > 2 else kwargs.get("context")
             payload = kwargs.get("minimax_payload") or {}
-            has_vc = bool(payload.get("keyframes") or payload.get("refs"))
-            us = _unique_t(ts, shift_v, shift_a, has_vc)
+            layout = payload.get("layout")
+            if layout is not None:
+                has_vc = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
+                has_ac = any(k == "ref_audio" for _, _, k in layout.segments)
+                vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
+                aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))
+            else:
+                has_vc = bool(payload.get("keyframes") or payload.get("refs"))
+                has_ac = False
+                vis_aug = 0.999
+                aud_aug = 1.0
+            us = _unique_t(ts, shift_v, shift_a, has_vc, has_ac, vis_aug, aud_aug)
             shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)
             return executor(*args, **kwargs)
 
```
However, I'm not that knowledgeable about this model arch so I can't tell if it's the correct way to solve it. But it works.

> **Comment by Larryvrh (2026-08-06T19:01:42Z):**

ref2va no training yet


> **Comment by rkfg (2026-08-06T19:20:17Z):**

Yeah but it works with ref2va, though I understand it's not optimal for now. It's just a small bug that prevents it from running if audio references are used, no issues with image references for example. Thank you for your work!


> **Comment by delykj (2026-08-06T21:59:50Z):**

@rkfg how do you setup the workflow with ref2va? I tried the turbo lora, but I get very bad image quality compared to without LORA usage. I'm using the Minimax-H3 Turbo LoRA node and the MiniMax-H3 Turbo Sampler nodes. Thanks.


> **Comment by rkfg (2026-08-06T22:02:09Z):**

Well, it's not perfect as the author has already said, try lowering its strength if you get too sharp results. I use the default workflow mostly, nothing fancy.


> **Comment by paju1986 (2026-08-07T13:31:07Z):**

ref2va kinda works but the results are quite suboptimal...


> **Comment by Dawgmastah (2026-08-07T20:03:55Z):**

Its not that bad if you follow the guide, its like 80% there. Also came to report on this bug.


> **Comment by rkfg (2026-08-07T20:20:07Z):**

This bug is fixed in the last version, I haven't tried the most recent commit but before that there was an issue with quality degradation compared to 1.1.0 which worked fine (but had this audio ref crash). So currently I run 1.1.0 and will try the recent version again a bit later.


> **Comment by rkfg (2026-08-08T00:26:30Z):**

I was mistaken, the issue still stands. I uploaded a video as a reference and got this error with the latest commit.


> **Comment by chflpu (2026-08-08T05:40:01Z):**

I'm getting the same error when using fl2va with referenced audio. 

I'm using these nodes and workflow to chain clips

https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context

`Node threw an error during execution.


# ComfyUI Error Report
## Error Details
- **Node ID:** 174
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

## Stack Trace
```
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)

  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~

```
## System Information
- **ComfyUI Version:** 0.30.0
- **Arguments:** ComfyUI\main.py --windows-standalone-build
- **OS:** win32
- **Python Version:** 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** true
- **PyTorch Version:** 2.13.0+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 5080 Laptop GPU : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 17094475776
  - **VRAM Free:** 15798894592
  - **Torch VRAM Total:** 0
  - **Torch VRAM Free:** 0

## Logs
```
2026-08-07T22:41:32.404616 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T22:41:32.536470 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T22:41:32.536572 - 
2026-08-07T22:41:32.536834 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:41:43.914180 - [H3TURBO step 0] sv=1.0000->0.9837  denoised_rms=1.1658  video: x_rms=0.9841 v_rms=1.4718  audio: x_rms=0.9395 v_rms=4.4976 slope=4.00002026-08-07T22:41:43.914282 - 
2026-08-07T22:41:44.606028 - 
 17%|██████████████                                                                      | 1/6 [00:12<01:00, 12.07s/it]2026-08-07T22:41:55.915789 - [H3TURBO step 1] sv=0.9837->0.9601  denoised_rms=1.0832  video: x_rms=0.9615 v_rms=1.4225  audio: x_rms=0.8645 v_rms=4.1610 slope=3.63542026-08-07T22:41:55.915917 - 
2026-08-07T22:41:56.580207 - 
 33%|████████████████████████████                                                        | 2/6 [00:24<00:48, 12.01s/it]2026-08-07T22:42:07.930764 - [H3TURBO step 2] sv=0.9601->0.9231  denoised_rms=1.0617  video: x_rms=0.9273 v_rms=1.4189  audio: x_rms=0.7711 v_rms=3.7258 slope=3.18982026-08-07T22:42:07.930897 - 
2026-08-07T22:42:08.615370 - 
 50%|██████████████████████████████████████████                                          | 3/6 [00:36<00:36, 12.02s/it]2026-08-07T22:42:20.055984 - [H3TURBO step 3] sv=0.9231->0.8575  denoised_rms=1.0368  video: x_rms=0.8712 v_rms=1.4176  audio: x_rms=0.6558 v_rms=2.8922 slope=2.64062026-08-07T22:42:20.056112 - 
2026-08-07T22:42:20.600076 - 
 67%|████████████████████████████████████████████████████████                            | 4/6 [00:48<00:24, 12.01s/it]2026-08-07T22:42:32.131539 - [H3TURBO step 4] sv=0.8575->0.7064  denoised_rms=1.0221  video: x_rms=0.7704 v_rms=1.4165  audio: x_rms=0.5276 v_rms=1.9948 slope=1.96302026-08-07T22:42:32.131687 - 
2026-08-07T22:42:32.748949 - 
 83%|██████████████████████████████████████████████████████████████████████              | 5/6 [01:00<00:12, 12.06s/it]2026-08-07T22:42:44.217164 - [H3TURBO step 5] sv=0.7064->0.0000  denoised_rms=1.0233  video: x_rms=1.0257 v_rms=1.4155  audio: x_rms=0.5415 v_rms=1.1348 slope=1.13072026-08-07T22:42:44.217309 - 
2026-08-07T22:42:50.404534 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:17<00:00, 13.96s/it]2026-08-07T22:42:50.404796 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:17<00:00, 12.98s/it]2026-08-07T22:42:50.404900 - 
2026-08-07T22:42:50.675647 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00001.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T22:42:50.758823 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T22:42:51.095184 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T22:43:18.560567 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 100.95 seconds^[[0m
2026-08-07T22:43:19.047350 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "asyncio\events.py", line 89, in _run
  File "asyncio\proactor_events.py", line 165, in _call_connection_lost
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
2026-08-07T22:44:04.333079 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T22:44:04.631452 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25140MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-08-07T22:44:15.278775 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T22:44:15.672343 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:44:15.738454 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:44:15.761849 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T22:44:15.815106 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T22:44:16.135238 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T22:44:16.135364 - 
2026-08-07T22:44:16.135646 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:44:31.448936 - [H3TURBO step 0] sv=1.0000->0.9837  denoised_rms=1.1638  video: x_rms=0.9846 v_rms=1.4687  audio: x_rms=0.9358 v_rms=4.5016 slope=4.00002026-08-07T22:44:31.449041 - 
2026-08-07T22:44:31.962859 - 
 17%|██████████████                                                                      | 1/6 [00:15<01:19, 15.83s/it]2026-08-07T22:44:43.568854 - [H3TURBO step 1] sv=0.9837->0.9601  denoised_rms=1.0885  video: x_rms=0.9620 v_rms=1.4217  audio: x_rms=0.8607 v_rms=4.3120 slope=3.63542026-08-07T22:44:43.568976 - 
2026-08-07T22:44:44.444040 - 
 33%|████████████████████████████                                                        | 2/6 [00:28<00:55, 13.86s/it]2026-08-07T22:44:55.826087 - [H3TURBO step 2] sv=0.9601->0.9231  denoised_rms=1.0600  video: x_rms=0.9278 v_rms=1.4170  audio: x_rms=0.7690 v_rms=3.7459 slope=3.18982026-08-07T22:44:55.826166 - 
2026-08-07T22:44:56.474036 - 
 50%|██████████████████████████████████████████                                          | 3/6 [00:40<00:39, 13.02s/it]2026-08-07T22:45:07.817744 - [H3TURBO step 3] sv=0.9231->0.8575  denoised_rms=1.0309  video: x_rms=0.8716 v_rms=1.4119  audio: x_rms=0.6573 v_rms=2.9396 slope=2.64062026-08-07T22:45:07.817890 - 
2026-08-07T22:45:08.646999 - 
 67%|████████████████████████████████████████████████████████                            | 4/6 [00:52<00:25, 12.69s/it]2026-08-07T22:45:20.034708 - [H3TURBO step 4] sv=0.8575->0.7064  denoised_rms=1.0174  video: x_rms=0.7700 v_rms=1.4127  audio: x_rms=0.5377 v_rms=2.0461 slope=1.96302026-08-07T22:45:20.034833 - 
2026-08-07T22:45:20.719009 - 
 83%|██████████████████████████████████████████████████████████████████████              | 5/6 [01:04<00:12, 12.47s/it]2026-08-07T22:45:32.072040 - [H3TURBO step 5] sv=0.7064->0.0000  denoised_rms=1.0176  video: x_rms=1.0197 v_rms=1.4121  audio: x_rms=0.5640 v_rms=1.1506 slope=1.13072026-08-07T22:45:32.072175 - 
2026-08-07T22:45:32.824835 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:16<00:00, 12.34s/it]2026-08-07T22:45:32.825170 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:16<00:00, 12.78s/it]2026-08-07T22:45:32.825270 - 
2026-08-07T22:45:33.107304 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00001.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T22:45:33.186850 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T22:45:33.534530 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T22:46:00.815466 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 116.47 seconds^[[0m
2026-08-07T22:46:54.424423 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T22:46:54.517235 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:46:54.589682 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:46:54.729593 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T22:46:54.796399 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T22:46:54.894912 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T22:46:54.895039 - 
2026-08-07T22:46:54.895330 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:47:06.059671 - [H3TURBO step 0] sv=1.0000->0.9837  denoised_rms=1.2459  video: x_rms=0.9832 v_rms=1.5341  audio: x_rms=0.9367 v_rms=4.5375 slope=4.00002026-08-07T22:47:06.059825 - 
2026-08-07T22:47:06.712420 - 
 17%|██████████████                                                                      | 1/6 [00:11<00:59, 11.82s/it]2026-08-07T22:47:17.969193 - [H3TURBO step 1] sv=0.9837->0.9601  denoised_rms=1.0834  video: x_rms=0.9606 v_rms=1.4216  audio: x_rms=0.8616 v_rms=4.1672 slope=3.63542026-08-07T22:47:17.969319 - 
2026-08-07T22:47:18.617214 - 
 33%|████████████████████████████                                                        | 2/6 [00:23<00:47, 11.87s/it]2026-08-07T22:47:29.997189 - [H3TURBO step 2] sv=0.9601->0.9231  denoised_rms=1.0574  video: x_rms=0.9264 v_rms=1.4167  audio: x_rms=0.7694 v_rms=3.6820 slope=3.18982026-08-07T22:47:29.997301 - 
2026-08-07T22:47:30.652078 - 
 50%|██████████████████████████████████████████                                          | 3/6 [00:35<00:35, 11.94s/it]2026-08-07T22:47:41.998019 - [H3TURBO step 3] sv=0.9231->0.8575  denoised_rms=1.0322  video: x_rms=0.8703 v_rms=1.4140  audio: x_rms=0.6555 v_rms=2.8824 slope=2.64062026-08-07T22:47:41.998141 - 
2026-08-07T22:47:42.668451 - 
 67%|████████████████████████████████████████████████████████                            | 4/6 [00:47<00:23, 11.97s/it]2026-08-07T22:47:54.089872 - [H3TURBO step 4] sv=0.8575->0.7064  denoised_rms=1.0179  video: x_rms=0.7691 v_rms=1.4127  audio: x_rms=0.5320 v_rms=2.0137 slope=1.96302026-08-07T22:47:54.090008 - 
2026-08-07T22:47:54.627479 - 
 83%|██████████████████████████████████████████████████████████████████████              | 5/6 [00:59<00:11, 11.97s/it]2026-08-07T22:48:06.161268 - [H3TURBO step 5] sv=0.7064->0.0000  denoised_rms=1.0183  video: x_rms=1.0206 v_rms=1.4117  audio: x_rms=0.5467 v_rms=1.1311 slope=1.13072026-08-07T22:48:06.161400 - 
2026-08-07T22:48:06.778356 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:11<00:00, 12.03s/it]2026-08-07T22:48:06.778594 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:11<00:00, 11.98s/it]2026-08-07T22:48:06.778691 - 
2026-08-07T22:48:07.060710 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00001.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T22:48:07.143018 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T22:48:07.406403 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T22:48:34.532772 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 100.09 seconds^[[0m
2026-08-07T22:55:09.525526 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "asyncio\events.py", line 89, in _run
  File "asyncio\proactor_events.py", line 165, in _call_connection_lost
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
2026-08-07T22:57:15.418320 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T22:57:18.863934 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25140MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-08-07T22:57:41.641869 - ^[[32m[INFO]^[[0m h3_motion_context: loaded AV latent from D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00001.safetensors
2026-08-07T22:57:41.826567 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T22:57:41.861385 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T22:57:45.706264 - ^[[32m[INFO]^[[0m h3_motion_context: both context_latent and context_audio wired; using the latent (skips one VAE round trip).
2026-08-07T22:57:45.707687 - ^[[32m[INFO]^[[0m h3_motion_context: video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 480x640, trim 22, audio 50 frames -> 83 latent steps (2.075s) from latent, on the timeline ending at frame 22.000
2026-08-07T22:57:45.734618 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:57:45.797806 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:57:45.821101 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T22:57:45.883017 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T22:57:46.070154 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T22:57:46.070261 - 
2026-08-07T22:57:46.070551 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:57:46.238619 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:57:46.238741 - 
2026-08-07T22:57:46.277674 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
2026-08-07T22:57:46.298203 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

2026-08-07T22:57:46.304695 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 30.87 seconds^[[0m
2026-08-07T22:59:01.707228 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T22:59:01.736815 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:59:01.786127 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:59:01.974594 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T22:59:02.103817 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T22:59:02.103936 - 
2026-08-07T22:59:02.104208 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:59:02.135158 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T22:59:02.135260 - 
2026-08-07T22:59:02.153360 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
2026-08-07T22:59:02.158947 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

2026-08-07T22:59:02.174212 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.45 seconds^[[0m
2026-08-07T22:59:23.751477 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T22:59:23.803248 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:59:23.850066 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T22:59:23.983722 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-08-07T22:59:24.039685 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:03:20.298920 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:50<00:00, 11.48s/it]2026-08-07T23:03:20.299174 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:50<00:00, 11.53s/it]2026-08-07T23:03:20.299269 - 
2026-08-07T23:03:20.475147 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:03:20.570051 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:03:20.970076 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:03:44.270091 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:03:48.056573 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "asyncio\events.py", line 89, in _run
  File "asyncio\proactor_events.py", line 165, in _call_connection_lost
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
2026-08-07T23:03:55.155436 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 271.39 seconds^[[0m
2026-08-07T23:08:16.473339 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:08:16.867549 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25140MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-08-07T23:08:27.369737 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:08:27.403912 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:08:31.198390 - ^[[32m[INFO]^[[0m h3_motion_context: both context_latent and context_audio wired; using the latent (skips one VAE round trip).
2026-08-07T23:08:31.208866 - ^[[32m[INFO]^[[0m h3_motion_context: video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 480x640, trim 22, audio 50 frames -> 83 latent steps (2.075s) from latent, on the timeline ending at frame 22.000
2026-08-07T23:08:31.318670 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:08:31.396345 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:08:31.423385 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:08:31.480633 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:12:25.704762 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:48<00:00, 11.38s/it]2026-08-07T23:12:25.705007 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:48<00:00, 11.41s/it]2026-08-07T23:12:25.705096 - 
2026-08-07T23:12:25.957205 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:12:26.026631 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:12:26.361871 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:12:49.368730 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:12:56.666890 - ^[[32m[INFO]^[[0m AudioMerge: Trimming audio2 from 324608 to 294667 samples to match audio1 length.
2026-08-07T23:13:03.732250 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 287.24 seconds^[[0m
2026-08-07T23:18:22.707059 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:18:23.059713 - [MiniMaxH3TurboLoRA] pruned base [bypass]: lora=Minimax\minimax_h3_turbo_v4_step600_ema.safetensors strength=1.0 | 208 backbone modules, 158 bypass adapters, 1 injections, 50 int8 fc2 via merge + 51 adaln injected at run time | model=MiniMaxH3 weight_dtype=torch.bfloat16 weight_dev=cpu2026-08-07T23:18:23.059854 - 
2026-08-07T23:18:23.065611 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:18:23.130002 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:18:23.266873 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-08-07T23:18:23.268154 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:18:23.318414 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:18:24.473437 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9956, 0.9908, 0.9855, 0.9796, 0.973, 0.9655, 0.9571, 0.9474, 0.9362, 0.9231, 0.9076, 0.8889, 0.866, 0.8372, 0.8, 0.75, 0.6792, 0.5714, 0.3871, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T23:18:24.473616 - 
2026-08-07T23:18:24.473945 - 
  0%|                                                                                           | 0/20 [00:00<?, ?it/s]2026-08-07T23:18:24.496212 - [H3TURBO fwd pruned/bypass] call#1  qkv_proj.forward_owner=BypassForwardHook (BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected=False  timestep=1000.00  video_rms=0.9998 audio_rms=0.9932 dtype=torch.bfloat162026-08-07T23:18:24.496318 - 
2026-08-07T23:18:24.645830 - 
  0%|                                                                                           | 0/20 [00:00<?, ?it/s]2026-08-07T23:18:24.645939 - 
2026-08-07T23:18:24.662585 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
2026-08-07T23:18:24.673373 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

2026-08-07T23:18:24.678631 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 1.94 seconds^[[0m
2026-08-07T23:26:27.883920 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:26:27.908965 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:26:27.958313 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:26:28.092437 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:26:28.143783 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:26:28.248016 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T23:26:28.248166 - 
2026-08-07T23:26:28.248479 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T23:26:28.251414 - [H3TURBO fwd pruned/bypass] call#2  qkv_proj.forward_owner=BypassForwardHook (BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected=False  timestep=1000.00  video_rms=0.9998 audio_rms=1.0069 dtype=torch.bfloat162026-08-07T23:26:28.251520 - 
2026-08-07T23:26:28.281687 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T23:26:28.281805 - 
2026-08-07T23:26:28.301958 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
2026-08-07T23:26:28.306620 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

2026-08-07T23:26:28.311923 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.42 seconds^[[0m
2026-08-07T23:26:47.611648 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:26:47.641193 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:26:47.687017 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:26:47.820881 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-08-07T23:26:47.822621 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:26:47.878560 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:27:58.065048 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:08<00:00, 11.41s/it]2026-08-07T23:27:58.065409 - 
100%|████████████████████████████████████████████████████████████████████████████████████| 6/6 [01:08<00:00, 11.42s/it]2026-08-07T23:27:58.065516 - 
2026-08-07T23:27:58.223028 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:27:58.366460 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:27:58.705322 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:28:21.798678 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:28:25.563088 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 97.94 seconds^[[0m
2026-08-07T23:28:52.486123 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:28:52.574145 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:28:52.655334 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:28:52.788272 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:28:52.837247 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:32:39.184414 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:46<00:00, 11.36s/it]2026-08-07T23:32:39.184692 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:46<00:00, 11.31s/it]2026-08-07T23:32:39.184782 - 
2026-08-07T23:32:39.344760 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:32:39.421177 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:32:39.759862 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:33:02.923433 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:33:06.729468 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 254.23 seconds^[[0m
2026-08-07T23:35:51.710277 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:35:51.983884 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25140MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-08-07T23:36:06.063385 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:36:06.097478 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:36:10.036144 - ^[[32m[INFO]^[[0m h3_motion_context: both context_latent and context_audio wired; using the latent (skips one VAE round trip).
2026-08-07T23:36:10.046044 - ^[[32m[INFO]^[[0m h3_motion_context: video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 480x640, trim 22, audio 50 frames -> 83 latent steps (2.075s) from latent, on the timeline ending at frame 22.000
2026-08-07T23:36:10.140413 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:36:10.203731 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:36:10.225599 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:36:10.277647 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:40:02.658944 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:46<00:00, 11.42s/it]2026-08-07T23:40:02.659202 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:46<00:00, 11.35s/it]2026-08-07T23:40:02.659278 - 
2026-08-07T23:40:02.952127 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:40:03.031825 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:40:03.357602 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:40:26.492146 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:40:30.400833 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 278.68 seconds^[[0m
2026-08-07T23:40:31.362437 - ^[[1m^[[31m[ERROR]^[[0m Exception in callback _ProactorBasePipeTransport._call_connection_lost()
handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>
Traceback (most recent call last):
  File "asyncio\events.py", line 89, in _run
  File "asyncio\proactor_events.py", line 165, in _call_connection_lost
ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
2026-08-07T23:41:09.133882 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:41:09.243296 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:41:09.316286 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:41:09.456474 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:41:09.513311 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:45:00.811241 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:51<00:00, 11.52s/it]2026-08-07T23:45:00.811520 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:51<00:00, 11.55s/it]2026-08-07T23:45:00.811635 - 
2026-08-07T23:45:01.074692 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:45:01.153593 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:45:01.419925 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:45:23.849707 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:45:27.602132 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 258.43 seconds^[[0m
2026-08-07T23:46:19.797864 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:46:19.887314 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:46:19.971249 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:46:20.112014 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:46:20.168209 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:50:16.233271 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:56<00:00, 11.54s/it]2026-08-07T23:50:16.233551 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:56<00:00, 11.80s/it]2026-08-07T23:50:16.233657 - 
2026-08-07T23:50:16.431702 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:50:16.511040 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:50:16.766394 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:50:38.237699 - ^[[32m[INFO]^[[0m h3_motion_context: 221 frames / 9.2083s picture, 9.2083s sound, drift 0.01ms
2026-08-07T23:50:41.971200 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 262.16 seconds^[[0m
2026-08-07T23:51:33.795816 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:51:37.261809 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:51:37.298266 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:51:39.640389 - ^[[32m[INFO]^[[0m h3_motion_context: both context_latent and context_audio wired; using the latent (skips one VAE round trip).
2026-08-07T23:51:39.640717 - ^[[32m[INFO]^[[0m h3_motion_context: video/head, 5 frames -> 2 cond blocks at indices 0..1, 243 frame clip at 480x640, trim 5, audio 30 frames -> 50 latent steps (1.250s) from latent, on the timeline ending at frame 5.000
2026-08-07T23:51:39.670564 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:51:39.742757 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:51:39.767224 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:51:39.820874 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:55:11.700577 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:31<00:00, 10.48s/it]2026-08-07T23:55:11.700954 - 
100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [03:31<00:00, 10.57s/it]2026-08-07T23:55:11.701047 - 
2026-08-07T23:55:11.967907 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\h3_context\clip_00002.safetensors (video (1, 24, 72, 40, 30), audio (1, 32, 2, 405))
2026-08-07T23:55:12.043898 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-08-07T23:55:12.313747 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-08-07T23:55:34.285443 - ^[[32m[INFO]^[[0m h3_motion_context: 238 frames / 9.9167s picture, 9.9167s sound, drift 0.01ms
2026-08-07T23:55:38.243074 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 244.43 seconds^[[0m
2026-08-07T23:58:05.914605 - ^[[32m[INFO]^[[0m got prompt
2026-08-07T23:58:06.120496 - [MiniMaxH3TurboLoRA] pruned base [bypass]: lora=Minimax\minimax_h3_turbo_v4_step600_ema.safetensors strength=1.0 | 208 backbone modules, 158 bypass adapters, 1 injections, 50 int8 fc2 via merge + 51 adaln injected at run time | model=MiniMaxH3 weight_dtype=torch.bfloat16 weight_dev=cpu2026-08-07T23:58:06.121096 - 
2026-08-07T23:58:06.126325 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:58:06.189059 - ^[[1m^[[33m[WARNING]^[[0m Warning: TAESD previews enabled, but could not find models/vae_approx/None
2026-08-07T23:58:06.330689 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-08-07T23:58:06.332276 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-08-07T23:58:06.384160 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 50 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-08-07T23:58:07.421758 - [H3TURBO sampler] legacy dual-schedule (no native ModelSamplingAV)  sigmas=[1.0, 0.9837, 0.9601, 0.9231, 0.8575, 0.7064, 0.0]  x.shape=(1, 1, 2099520) dtype=torch.float32  v_numel=2073600 a_numel=25920  shapes=[torch.Size([1, 24, 72, 40, 30]), torch.Size([1, 32, 2, 405])]2026-08-07T23:58:07.421861 - 
2026-08-07T23:58:07.422117 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T23:58:07.456710 - [H3TURBO fwd pruned/bypass] call#1  qkv_proj.forward_owner=BypassForwardHook (BypassForwardHook => lora ACTIVE; else => BASE ONLY!)  is_injected=False  timestep=1000.00  video_rms=0.9998 audio_rms=0.9964 dtype=torch.bfloat162026-08-07T23:58:07.456817 - 
2026-08-07T23:58:07.580868 - 
  0%|                                                                                            | 0/6 [00:00<?, ?it/s]2026-08-07T23:58:07.580966 - 
2026-08-07T23:58:07.600204 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0
2026-08-07T23:58:07.610288 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1333, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui-kjnodes\nodes\preview_override_node.py", line 681, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1260, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1235, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 150, in _turbo_sampler
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 196, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\model_base.py", line 240, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 499, in forward
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 378, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 434, in wrap
    return executor(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 629, in _forward
    h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\comfy\ldm\minimax\model.py", line 268, in forward
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
                                                                     ~~~~~~~~~~~~~~~^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1778, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch\nn\modules\module.py", line 1789, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Diffusion\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo\__init__.py", line 235, in forward
    x = x + (bv @ (av @ sv.T)).T                              # [M, out]
        ~~^~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

2026-08-07T23:58:07.616618 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 1.68 seconds^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)`


> **Comment by chflpu (2026-08-08T06:00:00Z):**

Quick update, it works just fine if I use the regular `Load LoRA` node instead of the `MiniMax-H3 Turbo LoRA` node.


> **Comment by l33tm4st3r (2026-08-08T13:55:23Z):**

FYI

https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/9


> **Comment by Hermit6202 (2026-08-10T15:03:21Z):**

> Quick update, it works just fine if I use the regular `Load LoRA` node instead of the `MiniMax-H3 Turbo LoRA` node.

For me, the regular `Load LoRA` node don't work, I get this in the console log:

[MINIMAL] lora key not loaded: blocks.0.adaln_proj.linear.lora_A.weight
[MINIMAL] lora key not loaded: blocks.0.adaln_proj.linear.lora_B.weight
[MINIMAL] lora key not loaded: blocks.0.attn.out_proj.lora_A.weight
[MINIMAL] lora key not loaded: blocks.0.attn.out_proj.lora_B.weight
[MINIMAL] lora key not loaded: blocks.0.attn.qkv_proj.lora_A.weight
[MINIMAL] lora key not loaded: blocks.0.attn.qkv_proj.lora_B.weight
[MINIMAL] lora key not loaded: blocks.0.mlp.fc1.lora_A.weight
[MINIMAL] lora key not loaded: blocks.0.mlp.fc1.lora_B.weight
[MINIMAL] lora key not loaded: blocks.0.mlp.fc2.lora_A.weight
[MINIMAL] lora key not loaded: blocks.0.mlp.fc2.lora_B.weight


> **Comment by GiusTex (2026-08-11T14:51:57Z):**

Those changes at __init__.py fixed the node for me:
- https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/10/changes/a51537fa608ec804ef327941800956d50c3436d4 (fixes normal audio input)
- https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/23/changes/0134366987ca310dfd2dfa37fe726a4b1e9e103e (fixes video soundtrack input)

Edit:
Changed soundtrack commit link from bkunbargi's repo to this one


> **Comment by BladeMcCool (2026-08-12T19:24:54Z):**

should i wait for this to get into the main project or check out a fork? is this going to get fixed?


> **Comment by GiusTex (2026-08-12T21:57:29Z):**

> should i wait for this to get into the main project or check out a fork? is this going to get fixed?

Main project now supports normal audio input (commit [here](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/commit/546b5028f4934f5129eb6c7142c2f3e461dfddbf)). You still need https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/23/changes/0134366987ca310dfd2dfa37fe726a4b1e9e103e for video soundtrack input. If you need it, you can directly:
- open init.py
- go to [line 200](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/blob/main/__init__.py#L200), then change:
`if any(r.get("kind") == "audio" and r.get("ref_audio_t", 0) > 0 for r in refs):`
into
`if any(r.get("kind") in ("audio", "video_audio") and r.get("ref_audio_t", 0) > 0 for r in refs):` (like [here](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/23/changes/0134366987ca310dfd2dfa37fe726a4b1e9e103e))



### #2 [closed] PR: fixed no progress bug
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/2  by zhaoyun0071 2026-08-06T13:20:03Z



> **Comment by mery-96-new (2026-08-06T13:45:48Z):**

Thank you. It is working, I tried this pull request and my progress bar is backkk!


> **Comment by Larryvrh (2026-08-06T19:26:23Z):**

already integrated, thanks!



### #1 [closed] ISSUE: no  progress bar in the cmd
URL: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/issues/1  by hicaonmb 2026-08-06T12:13:02Z

<img width="1074" height="574" alt="Image" src="https://github.com/user-attachments/assets/6887300d-93e8-4f85-80e4-7cdf6360ba57" />


> **Comment by hicaonmb (2026-08-06T12:17:33Z):**

I think it's a bug . I can't see progress bar in the cmd window


> **Comment by zhaoyun0071 (2026-08-06T13:20:37Z):**

> <img alt="Image" width="1074" height="574" src="https://private-user-images.githubusercontent.com/29969536/632375792-6887300d-93e8-4f85-80e4-7cdf6360ba57.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODYwMjI3MjksIm5iZiI6MTc4NjAyMjQyOSwicGF0aCI6Ii8yOTk2OTUzNi82MzIzNzU3OTItNjg4NzMwMGQtOTNlOC00Zjg1LTgwZTQtN2NkZjYzNjBiYTU3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA4MDYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwODA2VDEzMjAyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ4NDkzZmJhMTBhNTQ1ZDU0MmRiNDc5ZmM0M2RkYTI4NjhiYjRhZjU4ZDlhYmIzZTZkZTdmYjE0ZDg5YmQ0YmMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.3B2QdjX4B_zE71ljEmmiTcVxVLJC4qAfyh1joECPyk0">

https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/pull/2


> **Comment by neural-dreamer (2026-08-06T15:15:48Z):**

Yes, we need a progress bar in the console, please. Without it, it's hard to understand how long the generation will take.


> **Comment by mzkao (2026-08-06T16:07:07Z):**

Please refer to https://github.com/shuaixn/ComfyUI-MiniMaxH3DualClockSampler
@Kvento 


> **Comment by Larryvrh (2026-08-06T19:01:06Z):**

Fixed!

