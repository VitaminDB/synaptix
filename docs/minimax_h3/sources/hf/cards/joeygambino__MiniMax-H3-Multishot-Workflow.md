# joeygambino/MiniMax-H3-Multishot-Workflow

Источник: https://huggingface.co/joeygambino/MiniMax-H3-Multishot-Workflow · скачано 2026-11-28
Загрузок: 271 · лайков: 91 · базовая:  · теги: comfyui · обновлено: 2026-08-27T21:42:22.000Z

---

---
license: apache-2.0
tags:
  - comfyui
  - minimax-h3
  - video-generation
  - text-to-video
  - workflow
  - audio-video
pipeline_tag: text-to-video
---

# MiniMax-H3 Seamless Chain

A ComfyUI node pack and two workflows that render a multi-shot MiniMax-H3 scene
as **one continuous take**: no visible cut at the shot boundaries, no colour
shift between shots, and continuous audio across the whole piece.

MiniMax-H3 natively generates blocks of roughly 10-15 seconds. This pack chains
those blocks into arbitrarily long scenes and hands back a single master video
with a single master audio track.

Released as **v2.7.0** of the ComfyUI-H3-Multishot pack.

## v2.7.0 - per-subject voices, flf_chain fixed, chain leveller, ComfyUI 0.34

- **Per-subject voice refs** (`voice_ref_2` / `voice_ref_3`, both samplers):
  each character keeps their own voice across a chained scene - verified blind
  across a full chain, no cross-speaker bleed.
- **flf_chain fix:** the memory bank stood in flf chains and pinned a clip of
  shot 1 (which opens on boundary plate 0) into every later shot as an unbound
  reference - PLATE0 bled back in from shot 2 onward. The bank now stands down
  automatically in flf_chain; plates alone carry the continuity there.
- **H3ChainNormalize:** post-chain texture + colour leveller - the 2.6.0
  long-take ratchet limit is addressed.
- **refresh_pin splice alignment**, **x0 clamp dial** (`x0_clamp_window`,
  dose cap 0.30), in-loop latent upscale + `refresh_renoise` +
  `pin_noise_ramp` + `auto_chunk_ffn`.
- **ComfyUI 0.34 supported:** native interior keyframe anchors detected; the
  pack's layout patch stands down automatically on 0.34, unchanged on older
  cores.
- **Engine-aware writer:** separate H3 (`<d>[English] ...</d>`) and LTX
  (quoted dialogue) system prompts for every mode via an appended `engine`
  widget.

## v2.6.0 adds the extend take: one prompt, one continuous speech, as long as you want

Set **`take_seconds`** on MASTER CONTROLS (or open `H3_Extend_Take`, now the
main workflow, shipped at 1280x736 with a 30-second take) and give the writer
one premise. The panel sizes a window for your
card and the number of windows that fills the time; the writer's new
**extend take** join style writes ONE continuous speech and cuts it across
the windows at sentence boundaries; the chain continues it under
`context_pin` in H3's own voice — no TTS, no airlock, no per-shot dialogue
budgeting. Verified before shipping: seven writer-driven renders including a
65-second, 7-window take, every join continued the speech, reviewed blind as
one uninterrupted take. Also in 2.6.0: reference photographs now win over the
writer's prose (the writer points at the photos instead of describing a face -
same-seed measured), auto refs reach the sampler on their own (a second gate
downstream used to discard them silently), three
reserve-planning fixes from the 24 GB test lab and a writer fix for silent
shots echoing the voice anchor. Full notes in the changelog below.

**Known limit (2.6.0):** the chain's texture ratchet is not fully solved for long takes - measured about +13% fine texture per join at 736x1280 with the anti-drift set on. Under ~4 windows (~30-40 s) it is slight; at 7 windows it is visible sharpening. Keep extend takes to ~4 windows for now; a pin-side fix is in progress for 2.6.1.

## v2.5 is the memory release

Four new memory systems, all measured, two of them fully automatic:

- **The driver-headroom rule (automatic).** High-resolution renders used to
  take anywhere from 27 minutes to 3 hours for identical work — a lottery
  caused by the Windows driver demoting GPU memory when the card fills past
  roughly 95%. The pack now detects that zone and deliberately streams a few
  GB of weights instead of riding the ceiling. The lottery render became 15
  minutes, every time. Nothing to configure.
- **`low_ram_master`.** Long chains used to hold every finished shot in
  system RAM until the final join — tens of GB at the very last step. Switch
  it on and shots stream to lossless disk staging as they finish; peak RAM
  becomes about two shots regardless of chain length, with verified-identical
  output (42.8 dB against the RAM path — codec noise).
- **Remote text encoder.** The text encoder runs for seconds per shot and
  holds 15+ GB the rest of the time. Point the new node at any second PC
  running ComfyUI with this pack and that memory leaves your render card —
  identical results, verified across machines, with a local cache so repeated
  text never touches the network. One flag on the switches panel turns it
  on; it ships off.
- **`H3 TAE Decode`.** 2-second full-resolution draft previews from a 9 MB
  tiny decoder, versus about a minute per shot through the real VAE. For seed
  hunts and batch triage, never finals.

Plus a Speed Boosters panel (Spectrum, TeaCache, block cache, EasyCache —
each measured and eye-tested, with honest notes about which distort people),
and new defaults tuned for 16-24 GB cards. Details in the changelog below.

## What this repository contains

- `ComfyUI-H3-Multishot/` - the ComfyUI custom-node pack (samplers, loaders,
  studio controls, LoRA stack, GGUF architecture patch).
- `workflows/` - three ready-to-load ComfyUI graphs:
  - `H3_Seamless_Chain_v2.json` - the full workflow: master controls, LLM
    prompt writer, speed boosters, remote encoder lane, anchors.
  - `H3_Seamless_Chain_CORE.json` - the same chain with **zero third-party
    dependencies**.
  - `H3_Keyframes.json` - a single clip with anchors at chosen frame
    positions.
- `INSTALL.md`, `SETTINGS.md`, `PROMPTING.md` - install steps, the full
  settings reference, and the boundary/prompt rules.

## What this repository does not contain

**No model weights.** Nothing here is a checkpoint, a text encoder, a VAE or a
LoRA. Download the weights separately:

| Component | Where |
|---|---|
| MiniMax-H3 checkpoint (`ref2va` shipped, `fl2va` also chains), GGUF quants | [joeygambino/MiniMax-H3-GGUF](https://huggingface.co/joeygambino/MiniMax-H3-GGUF) |
| Text encoder, video VAE, audio VAE | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |

GGUF sizing guide from the quant repo: `Q8_0` for 32 GB cards, `Q5_1` for
24-32 GB, `Q4_0` for 16 GB. The `curve` variants are pruned-form requants.

The `ref2va` checkpoint is only needed for the reference/bank workflows. It is
**not** required for seamless chaining.


### Nodes added in 2.1

| Node | What it does |
|---|---|
| `RiftPromptSource` | One dropdown over LPFF-style `.txt` briefs and passthrough `.json` scripts. Emits `story_idea` / `character` / `count`. Reads `input/rift_prompts/`, and still reads the older `input/joyecho_prompts/` so existing folders keep working. |
| `RiftScriptPicker` | JSON script dropdown, and the speaker/voice stash `RiftPromptSource` feeds. |
| *(not a node)* `unload_model_after` | A switch **added to the LLM prompt writer** (`JoyEcho_LLMEnhance`). On, the writer frees its own model from Ollama once the script is written, so the video model gets the card. Uses the writer's existing `base_url` and `model_name` — nothing to keep in sync. Added in memory at startup by this pack, so the writer's own package is not modified; the switch simply appears on the node. Off by default. Ollama's OpenAI-compatible endpoint has no `keep_alive` field and its shim never sets one, so without this the model sits for the server default of five minutes — your whole first shot. |

`JoyEcho_PromptSource` and `JoyEcho_ScriptPicker` still resolve as deprecated
aliases, so graphs saved against 2.0 open unchanged. They were never published
under those names — that was the 2.0 bug.

### The prompt writer needs a model you actually have

The full workflow points at a local Ollama with `model_name = qwen3:14b`.
**Pull it before the first queue** or the run stops with
`LLM API error 404: model 'qwen3:14b' not found`:

```
ollama pull qwen3:14b
```

Any OpenAI-compatible endpoint works — its URL in `base_url`, its exact tag in
`model_name` (`ollama list` prints the tags you have). A remote endpoint is
often better: a local writer large enough to be good competes with H3 for the
same card and on under 32 GB will evict the model mid-render. When you do run
local, turn on `unload_model_after` on the writer — it frees the model as soon
as the script is written.

No LLM at all? Set `use_file_prompts` to manual entry, delete the writer, and
feed your own `---`-separated shot script into the sampler's `script` input.
The CORE workflow already works this way.

**Try a render with every switch off and the reserve at `0` before touching
any of this.** The activation reserve measures each shape and conditioning
payload as it renders and sizes the pool itself, and it holds on 24 GB cards
as well as 32 GB. A hand-set reserve *overrides* that measurement. These
switches are for digging out of a spill the console has already named.

## Requirements

Always required:

- **ComfyUI v0.30.0 or newer** (native MiniMax-H3 support).

The **CORE** workflow needs nothing else. It uses only this pack plus ComfyUI
built-ins (`LoadImage`, `LoadAudio`, `SaveVideo`, `SaveAudio`, `CreateVideo`,
`VAELoader`, `PrimitiveFloat`, `Note`).

The **FULL** workflow additionally uses:

| Pack | Needed for |
|---|---|
| ComfyUI_JoyAI_Echo_GGUF_Nodes | the LLM prompt writer — **ships in the release zip**, modified with attribution; use that copy, not upstream |
| [ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | `continuity=context_pin`, the shipped default |
| [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) | the `beta57` scheduler the full workflow ships with |
| ComfyUI-sol-attn + comfyui-minimax-h3-blockcache-T8 | the VRAM/SPEED patch switches |
| [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) | the on-canvas script preview |

ComfyUI validates **every** node class in a graph before it will queue, so a
missing pack stops the whole workflow — even for switches that ship OFF. Each
pack can be removed instead (one-widget change or node deletion, documented in
`INSTALL.md`); e.g. no RES4LYF → `scheduler=beta` (measured cost: lip-sync
8/10 vs 10/10, all else equal), no Motion-Context →
`continuity=first_frame`.

GGUF users also need [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF),
then run `apply_gguf_arch_patch.py` once to teach it the `minimax_h3`
architecture.

## Install

1. Search **H3 Multishot** in ComfyUI-Manager and install it - the pack is on
   the Comfy Registry as `comfyui-h3-multishot`. You can also copy
   `ComfyUI-H3-Multishot/` into `ComfyUI/custom_nodes/` by hand, or use
   Manager's Install via Git URL with
   `https://github.com/jlucasmcrell/ComfyUI-H3-Multishot`.
   The Registry carries the node pack only; the LLM prompt writer the full
   workflow uses ships in this repo under `nodes/`.
2. Place the MiniMax-H3 checkpoint, text encoder and VAEs in the usual
   ComfyUI model folders.
3. Restart ComfyUI.
4. Load `workflows/H3_Seamless_Chain_CORE.json` (no extra packs) or
   `workflows/H3_Seamless_Chain_v2.json` (full).
5. GGUF only: install ComfyUI-GGUF and run `apply_gguf_arch_patch.py` once.

Full detail is in `INSTALL.md`.

## Usage

1. Open a Seamless Chain workflow.
2. Set the **MASTER CONTROLS** panel (`H3StudioControls`): resolution, frames
   per shot, steps. One node drives both the sampler and the prompt writer's
   dialogue pacing, so the writer knows how much dialogue fits a shot.
3. Write your script into the sampler: **one prompt per shot, `---` between
   shots**. JSON of the form `{"prompts": [...]}` is also accepted. In the FULL
   workflow you can instead let the LLM lane write the shots for you.
4. Leave `continuity` on the shipped default and hit Queue.
5. The master video lands in `output/video/H3CHAIN/` with a paired audio file.

`preview_first_shot` is ON by default so you can judge shot 1 before paying for
the whole chain.

### Shipped defaults

| Setting | Default |
|---|---|
| Resolution | 736x1280 portrait (the model distorts faces below ~1 MP) |
| Frames per shot | 192 (8 s at 24 fps) |
| Steps | 14 |
| Sampler / scheduler | `euler` / `beta57` (full; RES4LYF) — CORE ships stock `beta` |
| Continuity | `context_pin` |
| Checkpoint | `ref2va` `curve-Q5_1` GGUF (~14 GB) — `fl2va` also chains, see below |
| Anti-drift | `chain_gain_control=flatten`, `master_normalize=luma+contrast`, `pin_renorm` ON, `memory_frames` 0 |
| Bank | OFF |
| Speed boosters | all OFF. Spectrum/TeaCache/EasyCache are real speed (-14 to -29%) but can distort people; block cache is inert at 14 steps (0 hits measured) |
| `low_ram_master` | OFF — turn ON for long chains or under 32 GB system RAM |
| `preview_first_shot` | ON |
| Mux | 24 fps |
| Output | `output/video/H3CHAIN/` + paired audio |

These defaults target a 16-24 GB card at the best speed/quality mix. Every
booster except block cache changes what people look like — leave them off
unless you have compared for yourself.

## How the chaining works

Two mechanisms ship in the pack. Both produce a single master; they differ in
what crosses the boundary.

### 1. `first_frame` chain - `H3MultishotSampler`

Each shot's **last frame** is handed to the next shot as its **first frame**,
through `fl2va`'s trained continuation task. That is the model doing what it
was trained to do, not a hand-rolled trick. The duplicated boundary frame is
trimmed, and the seam audio gets a 40 ms equal-power weld so the join does not
click.

No third-party dependency. This is what the CORE workflow uses.

### 2. `context_pin` - `H3MultishotMemorySampler`, `continuity=context_pin`

The previous shot's **last 22 frames ride into the next shot as raw latents** -
bit-identical, with no VAE round trip - placed at interior keyframe
coordinates, alongside a timeline-placed audio reference. The model regenerates
that overlap as its own head; the regenerated 0.92 s is trimmed on decode, so
what survives is the original tail followed by continuous new material.

Requires the third-party ComfyUI-H3-Motion-Context pack. As of v2.1.1 the two
packs coexist cleanly — this pack's payload wrapper declares Motion-Context's
compatibility marker, so load order does not matter. `seamless` and
`seamless_tail` are legacy comparison modes: `seamless` is a soft latent-only
pin that often reads as a cut, and `seamless_tail` conflicts with
Motion-Context and stops up front, before sampling, when that pack is
installed.

### Why identity holds with no reference images

Two mechanisms stack:

1. **The frame relay.** Every shot after the first begins from an actual
   rendered picture of the character. Faces and wardrobe propagate as *pixels*,
   not as a re-imagining from text.
2. **Byte-identical text.** The prompt writer repeats each character's
   appearance block **verbatim** in every shot.

The frame pins the specific instance; the repeated text pins the category so
the model cannot drift the description out from under the pixels. This was
verified on a 40-second two-character scene with **zero reference images
supplied**.

## Settings reference

`SETTINGS.md` in the release carries the complete list. The dials that matter
most:

| Dial | Notes |
|---|---|
| `shot_count` | `0` = one shot per prompt in the script. |
| `seed_per_shot` | **Leave ON.** Measured: per-shot seeds hold the face; a single seed shared across all shots drifted both face and voice. |
| `continuity` | `first_frame` or `context_pin` (see above). |
| `chain_gain_control` | Set to `flatten` on chains past roughly 5 shots. Texture ratchets about 1.3x per join otherwise, so late shots come out visibly over-sharpened. |
| `color_level` | `off` / `mvgd` / `scene`. Levels each shot's colour and exposure statistics to shot 1's settled tail (a fixed reference - matching neighbour-to-neighbour re-accumulates drift). Usually unnecessary; for long chains that drift warm or cool. |
| `self_anchor_voice`, `voice_ref` | Voice identity across the chain (the v1.5 headline feature). |
| `output_scale` | Lanczos resize of each shot AFTER decode - resolution, not detail. Works with every continuity mode including `context_pin`; applied per shot and after the bank takes its clip, so conditioning and VRAM are unchanged. Measured 1.78x faster than native for the same output size, and visibly softer. |
| `upscale_model` | Optional `UPSCALE_MODEL` link (ESRGAN and friends via ComfyUI's own loader). Synthesizes detail rather than resizing. Its invented texture never reaches the memory bank, so it cannot feed the sharpening ratchet. |
| `reference_image_size` | `match` or `max`. |
| `preview_first_shot` | Renders shot 1 alone first so you can abort early. |

### MASTER CONTROLS (`H3StudioControls`)

One node sets resolution, frames per shot and steps for **both** the sampler
and the prompt writer, so the writer's dialogue budget always matches the shot
length actually being rendered.

### VRAM/SPEED panel (`H3StudioSwitches` + reserve control)

Three lazily gated switches: memory-efficient attention, chunked feed-forward,
and the remote text encoder (speed boosters have their own node). The gates
are lazy, so a switch left OFF never executes its path - and all OFF is
exactly the verified recipe. The panel keeps its 2.5.x eight output slots so
saved graphs keep working; the removed ones always emit False.

The activation-reserve heuristic decides how much VRAM to hold back. Its cache
keys include a conditioning-**payload** signature (keyframes / audio refs /
two-pass), so a bare shot 1 and a reference-laden shot 2 are measured
separately instead of sharing one wrong number. Measured pools are no longer
overridden by a fixed floor, the first run of a new payload variant estimates
from a measured sibling, and a VRAM spill into system RAM is now detected and
named in the console - previously it only showed up as an unexplained ~5x
slowdown.

## Prompt and boundary rules

These are render-verified. The FULL workflow's prompt writer applies them
automatically through its `join_style` control, which appends them to the
system prompt. **Hand-written scripts must follow them manually.**

- **AIRLOCK.** Every shot after the first opens holding the previous shot's
  exact closing arrangement, and gives about **2 quiet seconds** before anyone
  speaks - a breath, a weight shift, real micro-motion, not a freeze.
- **The first ~1 second of every chained shot is discarded replay.** Dialogue
  starting at frame 0 loses its opening syllables.
- **LAND SETTLED.** End each shot back in a stable arrangement, dialogue
  finished, about 2 seconds spare.
- **A spoken line never straddles two shots.** Budget: dialogue plus 4 seconds
  of quiet must fit inside the shot. 243 frames fits one long line; 124 does
  not.
- **Repeat verbatim.** Each character's appearance description *and* the
  room/lighting description, word for word, in every shot.
- **Keep fps at 24.** Other rates audibly shift voice accents.

`PROMPTING.md` has worked examples.

## New in v2.6.0 - the extend take

### One prompt, one continuous speech, as long as you want

Type a length into MASTER CONTROLS `take_seconds`. The panel picks the
largest window whose activation pool fits with most of the weights resident
on your card (`window = auto`; wire the loader's MODEL into the panel for a
real weight size) and the count that fills the time; the writer's new
**extend take** join style writes ONE speech and cuts it across the windows
only at sentence boundaries; the memory sampler chains them under
`context_pin`. H3 continues the speech across every join in its own voice.
Verified: seven writer-driven renders at 141/192/243-frame windows plus a
65 s / 7-window take — every join continued, zero repeats, zero clipped
words. `take_seconds = 0` is the old behaviour exactly.

Also new: `audio_pin_frames` on the memory sampler (audio reference window
independent of the picture pin; 96 = the JoyEcho 4 s audio memory; neutral
at n=1, ships 0), the standalone `H3ExtendTake` node, the `H3_Extend_Take`
workflow.

### Fixed: reserve planning (from the 24 GB test lab)

The bare-to-payload x1.6 no longer fires between the two samplers'
differently-named payload signatures; a first run at a new shape borrows
measurements only from the same quant family (GGUF figures do not transfer
to w4a8/int8 — a cross-family borrow under-reserved a first run into driver
paging); the streamed master's metadata is the API graph again.

### Fixed: silent shots echoing the voice anchor

A silent shot whose framing boilerplate said "visible lip movement clearly
readable", with shot 1's voice riding along as `<Audio 1>`, re-spoke shot
1's line word for word. The few-shot example no longer mentions lip
movement, the rule cites the render, and the writer warns by shot number.

## New in v2.5.5 - the memory release

### New: the driver-headroom rule (the random-slowdown fix)

The single biggest fix this pack has shipped. High-resolution renders would
randomly take anywhere from 27 minutes to 3 hours for identical work — same
seed, same settings. The cause: when model weights plus working memory fill
the card past roughly 95%, the Windows driver starts demoting GPU memory
unpredictably, and whether your render crawled was luck.

The auto-reserve now detects that zone before sampling and deliberately
streams a few GB of weights instead — streamed weights are nearly free on
modern ComfyUI, the last few percent of VRAM are not. The lottery render
became 15 minutes, every time, verified on both the clean-shot and
payload-shot branches. Automatic; the console prints a "driver headroom"
line when it engages.

### New: `low_ram_master` streams the master to disk

Off (the default), every finished shot is held in system RAM until the final
join — historically the point where long renders killed whole machines. On,
each shot streams to lossless disk staging as it finishes and the master is
assembled from disk through the exact same levelling math: verified at
42.8 dB against the in-RAM path, which is codec noise. Peak RAM stays near
two shots regardless of chain length. The finished file's path now comes out
of a new `master_path` output either way.

### New: remote text encoder

The text encoder works for a few seconds per shot and occupies 15+ GB for
the whole render. Install this pack on any second PC with ComfyUI, point the
new **H3 Remote Text Encoder** node at it, turn `remote_encoder` ON in the
VRAM / SPEED SWITCHES panel, and prompts are encoded over there while your render card keeps the memory. Results are
identical — verified across two different machines to float precision — and
repeated scene text is answered from a local cache with no network call at
all. Ships wired into the full
workflow; the panel flag defaults OFF, so single-PC setups see zero change.

### New: `H3 TAE Decode` draft previews

A 9 MB tiny decoder (Kijai's taeh3, downloaded separately into
`models/vae_approx/`) turns latents into full-resolution draft frames in
about 2 seconds, versus roughly a minute per shot through the real VAE.
Drafts smear fine texture but composition, framing and motion read clearly.
For seed hunts and batch triage — never finals.

### New: the Speed Boosters panel

Spectrum, TeaCache, block cache and ComfyUI's own EasyCache, each behind a
switch on one node, each measured on the same seed and eye-tested on the
same masters:

| Booster | Time saved | Eye verdict |
|---|---|---|
| block cache | none at 14 steps | 0 cache hits on every run, both cards — inert; ships OFF (only fires at 30+ steps) |
| Spectrum | -29% | visible distortion on people; rooms fine |
| TeaCache 0.15-0.30 | -14 to -21% | visible distortion on people; rooms fine |
| EasyCache (built-in) | -21% | same family of people distortion |
| Spectrum + TeaCache stacked | no faster than Spectrum alone | severely damaged — never stack |

A booster whose pack is missing prints an install link and passes the model
through unchanged instead of breaking the graph.

### Changed: defaults tuned for 16-24 GB cards

736x1280 at 192 frames and 14 steps, `euler`/`beta57`, `curve-Q5_1`
checkpoint, the full anti-drift set on (`context_pin`,
`chain_gain_control=flatten`, `master_normalize=luma+contrast`,
`pin_renorm`, `memory_frames=0`). The resolution floor is deliberate: the
base model distorts faces below roughly 1 MP.

### Changed: the writer knows how long a shot really is

The prompt writer now receives each shot's actual speakable span — clip
length minus the replay, airlock and settle a chained shot spends — and
sizes dialogue against it. Overruns and silent scripts are flagged at
generation time with the numbers. Measured on renders: lines over budget
garble, lines under budget drag, matched budgets reviewed as natural and
fully intelligible.

### Changed: writer craft rules from failed renders

Silent shots with visible people now state what mouths are doing — stops the
model inventing mumbling. Revealed objects are written as already present —
stops mid-shot pop-ins. Shots that open a mouth without dialogue must name
the sound it makes.

### New: clickable title blocks in every workflow

Each workflow carries its version number and live links to the GitHub repo,
this page and the Civitai page, plus a plain-language pass over every
on-canvas note.


## New in v2.2.5

### Fixed: chained shots could stall before the first sampling step

The auto-reserve learns how much activation pool a given shape needs by
measuring it. On a chained render the later shots often load *partially*
- the weights are already resident, so only part of them streams in. The
old code discarded every partial measurement as untrustworthy, which meant a
chained shot could never contribute what it learned. If the very first shot was
also partial, the cache stayed empty forever, the fallback reserve was too small
for the shape, and every subsequent attempt failed the same way. A render could
sit there refusing to start with no error to point at.

Partial measurements are now kept when the pool is genuinely large, and
clamped by a named floor (`_auto_cache_floor`) that only ever ratchets
upward, never down. Verified live on a six-shot chain: offload dropped from
4308 MB to 638 MB once the cache started learning.

The warning that fires when a shot asks for far more pool than is available
now distinguishes the two cases it was conflating. If this shape has been
measured before, it says so and gives you the number. If it is the first run at
this shape, it says *that* instead of implying something is broken.

### Fixed: `output_scale` was documented backwards

With an upscale model selected, `output_scale` is not a multiplier
on top of it - the model applies its own fixed factor (4x for the
ESRGAN family) and `output_scale` is the **final** size
you want, not an extra step. The tooltip said otherwise, in both samplers. Both
are corrected, and `upscale_model_name` now explains the interaction
rather than leaving you to discover it.

### New: the upscale says what it will cost before it runs

Decoded frames accumulate in host RAM until the master is joined, so the bill
is per-shot multiplied by shot count. A 1344x768 chain through a 4x model is
5376x3072 per frame, and an `output_scale` left at a value that
looked harmless has taken machines down at the join - after all the
sampling was already paid for.

Before sampling starts you now get the projected dimensions, MB per frame, GB
per shot and GB total, measured against actual free RAM:

```
[H3Memory] upscale will produce 5376x3072 (4.0x): 198 MB per frame
[H3Memory] WARNING: that will not fit. Frames are held in host RAM
until the master is joined - lower output_scale to 1.60
or lower, or turn the upscaler off.
```

### New: the folder walk prints which index is which file

Queueing a folder told you nothing about the mapping between
`EPISODE INDEX` and the file it would pick, so a cancelled run left
you doing off-by-one arithmetic against a single log line to work out where to
restart. The ordered list now prints once per folder per session, and each job
reports its own position and the index to set to resume there. It re-prints if
the folder's file count changes, which is the one case where a remembered
listing would mislead.

### Fixed: garbled speech - the writer was never told how long a shot is

`num_frames` on the LLM Enhance node defaults to `0`,
and it shipped as `0` in the bundled workflow. At zero, the writer is
told **nothing** about clip length, so it sizes every spoken line
for the "~10 second clip" assumed in the system prompt regardless of what you
actually render. Too long for a short shot is crammed, garbled speech; too short
for a long one is dead air.

Three things now close that:

- `num_frames = 0` prints a warning naming the widget and telling
you to match it to the sampler's `frames_per_shot`. It used to pass
in complete silence.

- The bundled workflow ships with `num_frames` set to
**243**, the same value its `frames_per_shot` already
uses, so a fresh install is correct out of the box.

- Shots whose dialogue overruns the budget are reported at generation time
with the numbers, instead of being discovered forty minutes later by ear. The
check is symmetric: a script that comes back with **no dialogue at
all** is also flagged, because that renders as a silent slideshow.

### Fixed: `revise` asked for lines that could not fit

Chained styles spend about five seconds of every block on the replay, airlock
and settle, so the speakable span is shorter than the clip. The main writer
subtracted that. `revise` did not - it sized lines against the
raw clip length, from its own separate copy of the arithmetic.

On a 243-frame chained shot that meant `revise` asked for
**12-20 words where only 6-10 fit**: double the budget,
every line, which is exactly the overrun that renders as garble. It also never
checked its own output, because it returns long before the check the main path
runs.

Both paths now call one `_speakable_budget()` and both check their
result, so they cannot drift apart again. The duplicate arithmetic is why they
had.

### Changed: the story writer stops prescribing creativity

The system prompts had accumulated a five-beat dramatic arc, a banned-cliche
list, a "land near seven shots" target and a dialogue quota. None of those are
properties of the model - they are one person's taste, hardcoded, and they
were flattening every brief toward the same shape.

They are gone. What remains is what the renderer actually needs: valid JSON,
identity restated verbatim per shot so the room does not drift, literal physical
description over mood language because abstract adjectives have nothing to
render, and the note that audio is half the model. The story, its length, its
tone and whether any given shot speaks are the model's call now.

### Changed: the node menu no longer says "Rift"

Four nodes sat in a category called `Rift`, which means nothing to
anyone who is not me. They are in `H3/script` now, the labels drop the
prefix, and ten console messages say `[H3 Multishot]` - the name
you actually installed - instead of `[Rift]`.

> **Saved workflows are unaffected.** Categories are
never written into a graph and display names resolve fresh on load. The
underlying class names are deliberately *unchanged*, because those strings
*are* written into every saved `.json` and renaming them would
turn your nodes into red missing-node boxes.

### New: installable from the ComfyUI Registry

The pack is published as `comfyui-h3-multishot`, so ComfyUI-Manager
can find and update it without a manual clone. The zip on this page stays the
complete bundle - both node packs, the workflows and the docs - and
remains the right choice if you want everything in one drop.

The dependency list is deliberately empty. Every import this pack uses
(`torch`, `torchaudio`, `psutil`,
`av`, `numpy`, `PIL`) is already guaranteed by
ComfyUI itself, and a custom node that declares `torch` can talk pip
into replacing a working CUDA build with a CPU wheel.

### Documentation: drift on long chains, and the dials that fight it

`SETTINGS.md` gains a measured section on why chained shots accrete
detail - each conditions on the previous shot's own output, so invented
texture compounds - and which controls actually counter it. Measured over
ten shots at 960x544:

- `memory_frames = 0` is the big one. The bank's RECENT slots feed
accreted output forward on top of the pin.

- `master_normalize = luma+contrast` levels brightness
**and** contrast of the finished chain.

- `pin_renorm = on` holds each pinned latent at shot 1's sigma.

- `chain_gain_control = flatten` on chains longer than about five
shots - texture ratchets roughly 1.3x per join.

Residual with all of those on is about **1.02 per hop**. Not
zero. Long chains still drift, slowly. `pin_noise` is small,
scene-dependent and gets worse above `0.10`; it is not the fix it was
described as in 2.1.5.


## New in v2.2.4

Four things people asked for or tripped over, plus the documentation that
should have prevented two of the questions.

### Fixed: reference images of different people blended into one face

If you gave the sampler reference pictures of two characters, it rendered
something that resembled neither. That was not you holding it wrong. Internally
every reference picture was declared to the model as
`<Picture N> is a reference photograph of <Subject 1>` -
*the same* Subject 1, every time. With one character that is correct and
it is why single-character references work well. With several you were telling
the model that photographs of different people all showed one individual, and
it produced the average.

There is a new `reference_subjects` field on the sampler. Leave it
empty and nothing changes - every picture is one person, exactly as before.
Fill in comma counts in picture order to group them:

```
reference_subjects: 3,3 pictures 1-3 are person A, 4-6 are person B
reference_subjects: 2,2,2 three people, two pictures each
```

Each group is then declared its own subject, with an explicit instruction that
it is never blended with the others. Only Subject 1 is described as speaking,
because H3's voice conditioning is single-speaker.

### New: point the prompt source at a folder and queue the whole thing

`start_index` has always selected a block *inside* one file.
That works for LPFF-style prompt batches, where one file holds many prompts. It
does not work for H3 scripts, because in an H3 script `---` separates
**shots within one scene** - concatenating scenes into one file
would fuse them into a single enormous take.

Turn on the new `walk_folder` switch and `start_index`
selects **which file** instead, walking the chosen folder in sorted
order and emitting the whole script. Wire it to a Primitive set to
`increment`, queue thirty times, and thirty finished scenes render
unattended.

### New: hand H3 an existing clip as a video reference

There is now a `V2V REFERENCE` lane in the workflow - Load
Video into Get Video Components into the new **H3 Reference Video**
node, then into two new sampler inputs. It ships muted; un-mute the three nodes
to use it.

**Read what this is before you wire it.** H3 is told a video
reference is "a clip from an earlier moment of this same continuous
scene" and asked to keep its framing, camera distance, room contents and
colour temperature. It is **scene and appearance conditioning**. It
is **not motion control** - there is no pose, depth or optical
flow path in H3, so your subject will not copy the movement in the clip. It
borrows the place and the look, not the action.

Keep the window short. References are subsampled to 2 fps and then ride
through *every* sampling step, so a 25-second clip is roughly 50 reference
frames of permanent per-step cost. The H3 Reference Video node trims to a window
and prints what that window will cost before you commit to it. Requires ref2va;
fl2va has no reference rows and ignores video references entirely.

### Fixed: an empty prompt dropdown that blamed the wrong thing

The prompt source finds `.txt` files through the
`inspire_prompts` folder path, which is registered by
**comfyui-inspire-pack**. Without that pack installed the path does
not exist, so the dropdown came back empty however many prompt files you had,
and the error blamed the prompts folder. It now tells you which of the three
fixes applies: install the pack, register the path yourself in
`extra_model_paths.yaml`, or set `manual_path` and ignore
the dropdown. The dependency is written into INSTALL.md as well.

### Documentation: what has to change between shots

Several people have hit chains where shot 3 comes back as a near-copy of shot
2. The prompting guide is partly responsible: rule 5 says to repeat descriptions
word-for-word, and never said what must *differ*. Rule 5 is only about
appearance and the room. It is not an instruction to restate the action, and
when it gets read that way most of shot 3 is byte-identical to shot 2 - and
the model is separately instructed to preserve the subject, the room and the
colour temperature, so "keep everything the same" wins.

There is now a sixth rule covering it. The short version: each shot's action
must leave the world in a state the previous shot's world was not in, physical
and irreversible, not a mood or a camera move. If you can swap two shots' action
lines and the script still reads correctly, the model cannot tell them apart
either.

### Documentation: fl2va is not a smaller file

Both this guide and INSTALL.md described fl2va as "lighter and
faster", which everyone reasonably read as a size claim while choosing a
file for a 24 GB card. It is not. **fl2va and ref2va are exactly the
same size at every quant level.** "Lighter" only ever meant
fewer tokens per sampling step, because there are no reference rows riding
along.

What actually separates them was never written down: **fl2va lands on a
supplied frame and ref2va only nudges toward one.** Measured against the
same frame, 26.35 dB on fl2va versus 16.15 dB on ref2va with a keyframe
at 6 turbo steps - and 16.81 dB at 20 stock steps, which rules out the
sampler and leaves the checkpoint. fl2va can also take a first *and* a
last frame and plan a camera move between them, which ref2va cannot do at all.
So: ref2va when identity or voice must persist, fl2va when a shot must start
exactly where the last one ended.

### Improved: the story writer, for thin briefs and long shot counts

Give the writer a couple of characters and a genre with no plot, ask for
fifteen shots, and it tended to return atmospheric moments rather than a story.
That was the system prompt's fault in four specific ways, all now fixed.

It was **calibrated to about seven shots and argued against more**
- "most scenes land between 4 and 10", "go higher only when
the story clearly has that many real beats", plus a padding test. Asked for
fifteen, the model was simultaneously told to produce exactly that count and that
this many was probably padding, and it resolved the contradiction by rationing one
thin premise across the shots. It is now told the opposite: **a high count
is an instruction to invent more story** - another location, a second
complication, a character who arrives partway through, a reversal that changes
what the earlier shots meant. The no-padding rule is unchanged and is exactly the
point; the way to satisfy a high count is to invent enough real plot that no shot
is padding.

When the brief names **only characters and a genre**, the writer is
now told plainly that it is the author: decide who wants what, the incident that
starts it, the complication, the turn and the ending *before* writing any
shot. Returning a series of moods involving the named characters is called out as
the most common failure on a thin brief.

**Dialogue density** now has a floor. Speech was marked optional
per shot and non-speaking shots were actively encouraged, with nothing on the
other side, so long pieces came back nearly silent. It now aims for two thirds or
more of shots to carry speech, with silent shots as punctuation.

And a new **FIT THE SHOT** section covering *both* dialogue
and action. Overrunning the clip is what produces crammed, garbled speech and
distorted motion, so the writer now budgets each shot - settle, action, line
at an unhurried pace, a beat to land on - and cuts the action rather than
speeding up the speech. One clear physical action per shot; if more than one is
written, split the shot.

### Note on saved workflows

Every new control in this release is appended at the *end* of the
node's input list. Widget values are stored positionally, so inserting a control
anywhere else silently shifts every value after it in workflows you have already
saved. Your existing graphs load unchanged.


## New in v2.2.3

Everything here is a fix. Nothing changed about how you use the workflow.

### Fixed: long multi-shot renders died at the very last step

A long job would sample every shot, upscale every shot, finish master
normalize - and then die while assembling the result, with nothing
written. Every expensive stage had already succeeded.

```
RuntimeError: DefaultCPUAllocator: not enough memory:
you tried to allocate 33791016960 bytes.
```

That is **host RAM, not VRAM**, and it is one contiguous request.
Upscaled frames were held on the host as fp32 for the whole run, and the final
assembly then allocated a *second* complete timeline while the first was
still alive. Peak was twice the timeline, and the timeline grows as
`shots × frames × height × width × 3 × 4 bytes`
with the upscale factor squared. Six shots of 243 frames upscaled is a 36.5 GB
timeline and a ~70 GB peak, which no 64 GB machine can satisfy.

Three changes: the timeline is now assembled into one preallocated buffer,
releasing each shot as it is copied (bit-exact - the same bytes as before);
frames are parked as fp16, which is well clear of the 8-bit the encoder writes
anyway; and master normalize upcasts to fp32 for its arithmetic and writes back,
so the memory saving costs no precision.

**Effect:** six shots of 243 frames upscaled drops from ~70 GB
peak to ~21 GB. Twelve shots of 192 frames drops from ~58 GB to ~31 GB.
Both now fit in 64 GB with the per-shot upscale unchanged. Diagnosed by a
second operator on a 3090 box; thank you.

### Fixed: the auto-reserve inflated its own estimate until the render spilled

If your chains got slower as they went - shot 1 fine, shot 3 sluggish,
shot 4 crawling - this is why. After each shot the pack measures how much
activation memory that shape needed. It subtracted the weight bytes
*currently resident* rather than the model's full size, so whenever the
model was only partly loaded the measurement came out too large by exactly the
amount that had been offloaded. That number then raised the next shot's reserve,
which left less room for the weights, which offloaded more.

```
shot 1 recorded 6.4 GB true 6.4 (full load - correct)
shot 2 recorded 8.0 GB true 5.3 (2749 MB offloaded)
shot 3 recorded 8.9 GB true 4.3 (4784 MB offloaded)
shot 4 asked for 19.9 GB -> clamped -> 262 s/step
```

The overstatement equals the offload every time. The measurement now uses the
model's full size, and a shot that offloaded weights records nothing at all
- it measured a spill, not an activation pool.

The cache also only ever grew, so a single bad shot poisoned a shape
permanently. Entries written by earlier versions are **dropped once**
on first load; you will see a line saying so, and those shapes re-measure on
their next run.

This does not make an over-committed render fit. If a shot
honestly needs more memory than the card has you still have to lower
`frames_per_shot` or resolution - the difference is that the
warning now fires on true numbers instead of the reserve quietly climbing.

### Fixed: the sol_attn and chunk_ffn switches did nothing

Reported by **sdktertiaire2**. Those switches shipped ON, but the
nodes they gate ship *bypassed* - they need third-party packs that
cannot be bundled (`ComfyUI-sol-attn`,
`comfyui-minimax-h3-blockcache-T8`). A toggle routing into a disabled
node changes nothing and warns about nothing. That contradictory default was
mine.

Both now ship OFF so the panel states what the canvas actually does, and the
panel is labelled with the fix: **install the pack, then select the node
and press Ctrl+B to un-bypass it.** The switch only routes; it cannot
enable a bypassed node. If you do not want those packs, leave the switches off
and lose nothing - they are speed and memory optimisations, not quality
features. Everything renders identically without them, just slower.

The nodes stay bypassed on purpose. Enabling them by default would hard-fail
every install that lacks the optimizer packs.


## New in v2.2.2

One fix, reported by a user against 2.2.1. If you render with reference
images, take this one.

### Fixed: reference renders forced glasses onto the subject

Reported on Civitai: a reference image produced a character wearing thick
black frames in every shot, and prompting to remove them changed nothing.

The pack injects a `retention_analysis` block alongside reference
images so identity and voice stop drifting across a chain. That block was
hardcoded to say the subject *"retains the same face, skin, hair,
**glasses** and wardrobe"* - so every ref2va render was told to
keep glasses whatever the prompt said. And it could not be argued with, because
the sampler runs on a `BasicGuider`: **cfg 1.0, no negative
branch**. There is nothing for a negative statement to subtract from, so
"remove the glasses" only put the word into the conditioning a second time.

The block now says "the same face, skin and hair" and names no accessories.
Eyewear, hats and jewellery belong to your prompt.

**The rule this exposes, and it applies to every prompt you
write for this graph:** anything hardcoded into unconditional conditioning
is permanent from your side. At cfg 1.0 you can add but never subtract, so
phrase everything positively - "clear unobstructed eyes", never "no glasses".
Negations do not work anywhere in this pack.


## New in v2.2.1

A bug-fix release. Four defects, three reported from real renders on real
machines rather than found in review. If you have been unable to finish a long
chain, or a high-resolution one died partway with the whole server going down
with it, this is the release for you.

### Fixed: two crashes that only appear on long or high-resolution chains

Both were `DefaultCPUAllocator: not enough memory` - system
RAM, not VRAM - and both killed the job *after* it had already done
the sampling.

- **`master_normalize` built the entire finished timeline four
times over.** A 12-shot 1088x1920 chain asked for a single
**31.2 GB** block and died on a 24 GB machine after 81 minutes of
work. One of those four copies was pure waste: the code rebuilt a tensor
byte-for-byte identical to one already in memory, purely to compute two numbers
for a log line, then threw it away. Every statistic it needs is
one-dimensional. It now measures per shot and releases each shot as it is
consumed, so peak memory is one finished timeline plus one shot instead of four
timelines. Output is **bit-identical** - verified against the
old implementation across every colour mode and median width, on the pixels and
on the log strings.

- **The upscaler was handed every frame at once.** 243 frames at
1472x2560 is an **11.0 GB** float32 allocation on top of the input.
It now runs in chunks into a preallocated buffer, so peak is one chunk rather
than the whole batch. Bit-identical, verified at two chunk sizes. Set
`H3_UPSCALE_CHUNK` if you want to trade memory for a little speed;
default 16.

### Fixed: auto-reserve could clamp itself into a hard crash

On a 12-shot 1280x736 run, shot 1 loaded completely and rendered fine. Shot 2
carries the context pin **and** the reference rows, so its
activation-pool requirement roughly doubles - the node correctly costed it
at 18.2 GB. It then clamped the pool to 9.4 GB *"so the weights still load
completely"*, and the weights loaded **partially anyway**,
401 MB offloaded. Neither constraint was met and the render aborted inside a CUDA
kernel, taking the whole ComfyUI process with it.

Two causes, both fixed. The keepout was **384 MB**, which did not
cover ComfyUI's own buffer plus the difference between what the driver reports
free and what ComfyUI treats as usable - about 400 MB short. It is now
**1 GB**. And the "this is tight" warning compared the clamped
reserve against a *previous shot's* measurement rather than what this shot
had just asked for, so a 9.4 GB clamp looked fine against shot 1's 9.1 GB while
the payload needed 18.2. It now compares against the real request and says
plainly that this usually dies inside a kernel, and that raising the reserve
cannot help because the memory is not there.

**Worth knowing regardless of this fix: shot 1 succeeding
tells you nothing about shot 2.** Shot 1 has no pin and no reference rows.
If shot 2 will not fit, lower `frames_per_shot` or the resolution, or
load a smaller quantisation of the DiT.

### Fixed: `start_image` on the memory sampler silently did nothing

Reported by a user. `start_image` on
`H3MultishotMemorySampler` is an identity **reference row**,
not a first frame - and on an fl2va checkpoint, which has no reference rows,
it is built and then ignored entirely. Meanwhile the sibling node
`H3MultishotSampler` has an input with the **same name**
that really is an I2V first frame, which is where the perfectly reasonable
expectation comes from. The combination produced no warning at all.

It now prints one, naming both ways to actually open shot 1 on a picture: use
`H3MultishotSampler`, whose `start_image` is a true first
frame, or set `continuity=flf_chain` here and feed
`keyframe_images` (N+1 stills for N shots). Note that
`continuity=first_frame` is not about your image either - it
hands over the *previous shot's* last frame, and does nothing on shot 1.

### Verified

Three chained shots rendered end to end on ComfyUI 0.32.0 with the new
auto-reserve: every shot `full load: True`, which is exactly where the
crashing run went partial, and the new warning fired on shot 2 with real numbers
and did not stop the render. `master_normalize` ran on that same
685-frame chain. The chunked upscaler is proven bit-identical by test rather than
by render - that graph had upscaling off, so I will not claim end-to-end
coverage it did not get.

## New in v2.2.0

Everything since **2.1.2**, which is where most people still are. Seven point
releases in one: five separate defects that stopped the workflow running, a
measured campaign against chain drift that changed the shipped defaults, and a
whole ComfyUI version this pack could not previously run on.

*(2.1.9 on GitHub and HuggingFace is this same content under a smaller number,
tagged an hour earlier.)*

**Run-blocking, all user-reported or found by finally testing what we ship:**

| | |
|---|---|
| the Audio Spine produced static on ComfyUI 0.32.0 | fixed in 2.1.7 |
| naming an mmproj file broke GGUF text encoders | fixed in 2.1.7 |
| `The value 1 for reference_image_size is not available` | fixed in 2.1.6 |
| the full workflow needed a node from a pack not in the zip | fixed in 2.1.8 |
| **`H3_Seamless_Chain_CORE` could not be queued at all** | fixed in 2.1.9 |

**New controls:** `pin_frames`, `pin_noise`, `pin_renorm`, and
`master_normalize`'s `luma+contrast` mode.

**Changed defaults that change your output:** `memory_frames` 2 → **0**, and
`join_anchor_noise` / `handoff_release` to 0 because they are inert under
`context_pin`.

**New runtime:** ComfyUI **0.32.0**, which needed real work — its
`ModelSamplingAV` carries the audio half of the latent on a different scale.
0.30.0 is unchanged and still supported.

Every section below is the original release note for each of those versions,
newest first.

---

## New in v2.1.9

### Fixed: `H3_Seamless_Chain_CORE` could not be queued at all

```
value_smaller_than_min: Value 0.0 smaller than min of 1.0 - output_scale
```

CORE's sampler still carried the widget array it was saved with **before 2.1.3
removed the four `two_pass_upscale` dials**. Nineteen saved values against the
class's fourteen live widgets, so the frontend put `false` into `output_scale`
(minimum 1.0) and `1.5` into `save_every_shot`, and the server rejected the whole
prompt. The workflow advertised as the one with no third-party dependencies — the
safest thing for a new user to open — has been un-runnable since 2.1.3.

It was never caught because CORE had **never been rendered end to end**. Our own
release checklist said "one CORE render before posting" and that step had been
carried, unticked, through six releases.

The array is now generated from a name→value map resolved against the server's
schema, and the map is stored on the node so the pack's JS can re-apply it by
name — the same treatment the full workflow's sampler got in 2.1.6. CORE has now
rendered: three chained shots at 960x544, 370 frames, picture and audio, holding
framing, wardrobe and lighting across both joins.

All three bundled workflows are now submit-tested against a live server as part
of the release routine, not read.

### Bypassed nodes now say what they need

Four nodes ship bypassed because their packs are not in the zip, and a missing
class fails the entire prompt. That is the right default — but a bypassed node
sitting on the canvas invites you to un-bypass it, and doing that without the
pack installed breaks the workflow with no explanation. Joe's point, and he is
right.

Now it says so in four places, in order of how hard they are to miss:

| | |
|---|---|
| the group title | `VRAM PATCHES - bypassed: install the pack named in each title BEFORE Ctrl+B` |
| each node title | `attention patch (bypassed - needs ComfyUI-sol-attn)`, and so on |
| a note above the cluster | install first, then `Ctrl+B`; without it the whole workflow stops queueing |
| the VRAM / SPEED note | the full table with repository URLs, and the two-step rule |

The two-step rule is the part people lose: un-bypassing a patch node changes
nothing on its own, and flipping its switch while the node is still bypassed
changes nothing either. Both, in that order, after installing the pack.

`SCRIPT PREVIEW` is the odd one out — it is a leaf, so leaving it bypassed costs
you the on-canvas preview and nothing else.

**Why not just bundle the packs?** The writer pack is bundled because it is
modified and pinned. These three are not: shipping a second copy of
Custom-Scripts, sol-attn or blockcache-T8 inside the zip would shadow whatever
the user already has installed and freeze it at the version we happened to
vendor. Naming them and linking them is the honest version.

---

## New in v2.1.8

### Fixed: one node in the workflow came from a pack that is not in the zip

`SCRIPT PREVIEW` is `ShowText` from **ComfyUI-Custom-Scripts**, and it shipped
*active*. A node whose class is missing serialises as `class_type: null` and the
server rejects the **whole prompt**, so without that pack installed the workflow
could not be queued — for the sake of an on-canvas text box. INSTALL.md said to
delete the node, which only helps someone who reads it before pressing Queue.

It now ships **bypassed**, the same remedy the three accelerator nodes got in
2.1.4. A bypassed node is dropped from the prompt entirely. Install
Custom-Scripts and `Ctrl+B` the node if you want the preview; the writer feeds
the sampler either way.

Found by mapping every node type in all three bundled workflows to its owning
Python module against a running server. That check is now part of the release
routine instead of something I do after a user tells me. The other two workflows
were already clean, and everything else in the full one is either core ComfyUI,
this pack, the writer pack **that is in the zip**, or one of the three bypassed
accelerators.

### Verified end-to-end on ComfyUI 0.32.0

Not by reading the graph — by loading the shipped file in a browser on a 0.32.0
install and pressing Queue. Twice: once through the prompt-writer lane exactly
as shipped, and once with a hand-written script.

- loads with no missing node types, and every sampler widget lands on its own
  name (`memory_frames` 0, `master_normalize` luma+contrast, `pin_renorm` on) —
  the shift that caused `The value 1 for reference_image_size is not available`
  cannot reproduce
- no null `class_type`: the bypassed preview and the three bypassed
  accelerators are all dropped cleanly
- three chained shots at 960x544, 124 frames each → **328 frames**, exactly 372
  minus the two 22-frame `context_pin` head trims
- a reviewer given the clip cold, with no idea how it was made, read it as
  **one continuous static take**, found no cut or jump anywhere, transcribed all
  three lines of dialogue, reported the lip-sync as matching, and found no shift
  in framing, colour, brightness or wardrobe and no hiss, dropout or click in
  the audio

0.30.0 remains the version everything else here was measured on; both are now
tested before release.

---

## New in v2.1.7

Three things that stopped the workflow running. All user-reported, all
reproduced, all fixed and verified by render rather than by reading the code.

**Everything below was verified on ComfyUI 0.32.0**, not just 0.30.0. Two of
these three only ever appear on 0.32, which is why they survived several
releases.

### Fixed: the Audio Spine produced static on ComfyUI 0.32.0

`guide_audio` came out as hiss while the same file through `voice_ref` or the
native node was perfect. Reported against 2.1.2, still present through 2.1.6.

ComfyUI 0.32.0 introduced `ModelSamplingAV`, which carries the **audio half of
the packed AV latent scaled onto the video schedule** — `process_latent_in`
multiplies it by `shift / audio_shift` (12/3 = **4** for H3) and
`process_latent_out` divides it back. Everything this pack injects into the
sampler's latent is in the stream's *native* domain, so on 0.32 it landed 4x
too small and decoded as broadband noise. On 0.30.0 there is no such scaling,
so the same code was correct — which is why it never reproduced here until a
0.32 rig existed.

Three injection sites had it, not one:

| source | used by |
|---|---|
| the encoded spine | Audio Spine |
| the previous shot's audio tail | `audio_lock`, latent handoff |
| the encoded room tone | onset guard |

All three now scale at the point of use, reading the factor off the live
`model_sampling` object so a changed sigma shift stays correct. `getattr`'s 1.0
default leaves 0.30.0 byte-identical.

Verified on 0.32.0 with a real 44.1 kHz stereo voice track: loudness-envelope
correlation against the guide **+0.965**, speech-band energy 34.2% against the
guide's 39.3%, and a blind listener transcribing the guide's words with "clean,
no background hiss". Before the fix the same render was hiss.

### Fixed: naming an mmproj file broke GGUF encoders

    mat1 and mat2 shapes cannot be multiplied (3680x1152 and 3456x1152)

thrown at the handoff into shot 2, GGUF encoders only, unaffected by resolution
or by turning every image input off.

Setting `mmproj_name` explicitly took a different code path than `(auto)` and
skipped the vision key-renaming step entirely — so the vision tower loaded
under raw llama.cpp names (`v.blk.*`, `mm.*`) that nothing downstream reads,
the merger was never populated, and the first matmul touching vision features
had the wrong width. Same file, two loaders, **19 key names in common out of
351**.

The bitter part: `mmproj_name` is documented as the reliable escape hatch for
when filename pairing fails, and it was the broken path.

It now runs the same post-processing as `(auto)`, using ComfyUI-GGUF's own key
map rather than a copy, so it follows their changes. Verified: an explicitly
named file now yields a state dict identical to `(auto)` — 351 tensors, every
shape matching. A new guard also fails by name if a chosen mmproj produces no
`visual.*` tensors, instead of dying in a matmul twenty minutes later.

### Fixed: `The value 1 for reference_image_size is not available`

The shipped workflow's saved widget values were written for a layout that did
not present `sampler_override` and `scheduler_override` as widgets. The current
schema does, so everything from index 27 read two slots early and
`output_scale`'s `1.0` landed in `reference_image_size`, a combo of
`match`/`max`. This affected **2.1.3, 2.1.4 and 2.1.5**.

The array is now generated from a name→value map resolved against
`/object_info`, and the same map is stored in the node's properties so the
pack's own JS can re-apply by name if a future schema change shifts anything.

### Also: `memory_frames` now defaults to 0

The bank's *recent* slots hand each shot's accreted output forward as reference
images on top of the latent pin, so invented detail compounds. Measured over ten
shots at 960x544, moving 2 → 0:

| | 2 | 0 |
|---|---|---|
| texture per hop | 1.055 | **1.022** |
| chroma per hop | 1.086 | **1.039** |
| framing correlation at shot 10 | 0.976 | **0.995** |
| drift acceleration | 4.2% → 6.7%/hop | **2.3% → 2.0%/hop** |

The last row matters most: at 2 the drift *accelerates*, which is what a runaway
loop looks like. At 0 it holds flat.

The obvious worry was motion continuity, since the recency slots exist to carry
it. Tested on a scene with continuous large-amplitude movement: anchor-only
retained motion **slightly better** (−5.9% vs −6.5% over four shots) with better
framing (0.983 vs 0.971). The cost does not exist. If a busy scene ever does
lose continuity between shots, raise it to 1.

`join_anchor_noise` and `handoff_release` now ship at 0 — both are **inert**
under `context_pin` (one noises keyframes the mode never creates, the other
belongs to `latent_handoff`) and non-zero values read as tuned settings while
doing nothing.

---

## New in v2.1.6

**Chained shots stop getting brighter-edged every hop.** Not by the route
2.1.5 claimed - see the correction below.

`master_normalize` matched every frame's MEAN to one global target, which is
why a chain shows no brightness step. It was also masking a second drift it
never touched. Measured on a 3-shot chain at 960x544: mean held flat, 27.31 ->
27.43, while the DISTRIBUTION stretched - p25 fell 6 -> 2 and p95 rose 85 -> 96.
Contrast climbing every hop, re-centred each time, and handed to the next shot
as a higher-contrast starting point.

**New: `master_normalize=luma+contrast`** (now the default) matches the spread
as well as the mean. Rescaling amplitude about each frame's own mean is an
affine remap: it moves no edges, so it is not the blur this pack has always
ruled out for texture drift.

Texture growth per hop, from a log fit across all shots. 1.000 is no accretion:

| where | `luma` | `luma+contrast` | contrast spread |
|---|---|---|---|
| 960x544, in-render, 124f x 4 | 1.126 | **1.047** | 11.2% -> 0.3% |
| 960x544, 243f x 3 | 1.199 | **1.064** | 12.2% -> 0.4% |
| 640x352, 243f x 3 | 1.130 | **1.055** | 7.0% -> 0.2% |

The baseline ratchet scales with canvas (1.130 at 640x352, 1.199 at 960x544);
after normalising it stops caring (1.055 vs 1.064). There is nothing in it to
tune per resolution - it works on decoded frames with per-frame statistics
against one global target.

The target anchors to **shot 1**, not the timeline median. Contrast only
ratchets upward, so shot 1 is the one frame-set with nothing accreted onto it;
a median target pulls shot 1 UP to meet the drift (+11.7% texture, for no
benefit) where anchoring to shot 1 leaves it untouched (+0.1%) and only ever
pulls later shots down.

1:1 crops of the last shot show no loss of real detail - lamp vent slots, hinge
rivets, hair strands and knit weave all survive. What leaves is the invented
crispness.

### What is left, honestly

About **1.05 per hop**. That residual is spatial accretion, and this pass
cannot reach it: `master_normalize` runs on the finished master, outside the
feedback loop, so it cleans what you see while the next shot is still handed
the inflated pin. Four shots is slight. Ten shots is roughly +50%. If you are
chaining long, expect it.

### Correction to v2.1.5

v2.1.5 said `pin_noise=0.05` fixed this. **It does not.** That was measured on
two seeds of a single scene at 640x352 - a scene whose background was nearly
black and which barely ratcheted to begin with - and the control that would
have caught it, `pin_noise=0.00` at the reporting user's own resolution, had
never been run. With it run:

| canvas | 0.00 | 0.05 | change |
|---|---|---|---|
| 640x352 | 1.131 | 1.111 | -1.8% |
| 960x544 | 1.211 | 1.201 | -0.9% |

Both on a detail-heavy scene. The dial is small and scene-dependent, it cannot
touch the dominant drift in a busy frame, and above 0.10 it gets **worse** (0.20
measured 1.228 against a 1.211 control). Its range now stops at 0.10 and its
tooltip says all of this. It stays in the pack because it costs nothing and
does help where the ratchet is already small; it is not the fix.

### Not tested

Portrait canvases. Everything above is landscape - 640x352 and 960x544. The
mechanism is resolution-independent by construction and the two landscape sizes
agree, but 768x1344 and 736x1280 have not been measured and are not claimed.

### Measuring this yourself

Texture comparisons are only meaningful **while the framing holds**. If the
model cuts to a different setup, texture reflects content and the number is
meaningless - one portrait run here scored a flattering 0.878 per hop purely
because shot 3 cut to a close-up of a film reel. Correlate each shot's mean
frame against shot 1's before trusting any of it; a held framing sits above
0.95.

That cut is worth a writing rule of its own: **do not name a nearby object in a
shot's closing beat.** *"She glances down at the reel"* reads as a request for
a shot of the reel. Keep closing beats on the speaker's own body.

---

## Fixed in v2.1.4

- **The main workflow would not queue without two third-party packs.**
  `H3_Seamless_Chain_v2.json` ships three optional accelerator nodes - `sol_attn`
  and `chunk_ffn` (**ComfyUI-sol-attn**) and `block_cache`
  (**comfyui-minimax-h3-blockcache-T8**) - and they were saved *active*, so a
  clean install hit `missing_node_type: Node 'attention patch (gated)' has no
  class_type` and nothing ran, even though the shipped recipe keeps all three
  gates off. They now ship **bypassed**: dropped from the prompt entirely, with
  the model passing straight through. To use one, install its pack, select the
  node, `Ctrl+B` to un-bypass, **then** turn its gate on.
- **`Value 4 bigger than max of 3: memory_frames` on a workflow you never
  edited.** v1.2 inserted `seed_per_shot` into the middle of the sampler's input
  list. Widget values are positional, so every dial after it shifted by one in
  workflows saved on v1.0/v1.1 - your old `anchor_frames` arrived as
  `memory_frames`. Pre-v1.2 workflows are now repaired on load (save to make it
  permanent), the sampler also stores values **by name** so no future change can
  shift anything, and the validation error explains the shift instead of naming
  the wrong dial.

---

## New in v2.1.3


> **Correction, 2026-08-12 — read this before turning any drift dial on.**
> A render with `chain_gain_control=flatten`, `color_level=mvgd` and the
> per-shot audio leveller all ON came out **+142% texture and +18% brighter**
> over three shots, with a visible brightness step at each join. The per-shot
> approach cannot work, for a reason the code already knew: under
> `context_pin` the drift is carried by the **raw latent pin**, and every one
> of those dials operates on decoded frames *after* the pin has been stored.
> They correct what you see and not what feeds forward. `audio_tone_control`
> has been **removed**. `color_level=mvgd` is **deprecated** — its own source
> comment records a 29% warmth step at every join. Use `color_level=scene`
> (one target for the whole piece, applied per frame at the end) and the new
> `master_normalize=luma`, both of which run outside the feedback loop and
> land every frame on the same number, so they cannot create a seam.
> Texture drift is **not** fixable after the fact: the only lever is blur, and
> blur removes real detail along with the invented kind.

- **Picture darkening over chains: measured, and the existing dial verified.**
  User-reported (−1.5 luma/shot, monotonic). Same autoregressive mechanism as
  the audio dulling; the raw-latent pin carries it directly. `color_level=mvgd`
  - shipped since 2.1, never verified - holds an 8-shot chain to −1.0 total
  luma where uncorrected loses −10.5 (both seeds). On long chains turn on all
  three drift dials: 
- **Audio dulling over long chains: measured, mechanism found, countered.**
  Five-arm A/B on 8-shot chains: with `bank_pinned=0` (pure recency
  conditioning) the voice band collapses - 84-92% of 4-10 kHz energy gone by
  shot 8; with the default pinned slot, 8-50% depending on seed. It is the audio twin of the
  seam sharpening ratchet, running the other way, and continuity mode is
  irrelevant - the bank decides. Two counters ship: a console warning when
  `bank_pinned=0` on a chain past 4 shots (there is no true "bank off" - 0/0
  leaves one recency slot, the worst configuration), and
  **`audio_tone_control=flatten`** - the audio twin of
  `chain_gain_control=flatten`, EQ-matching every shot's long-term spectral
  envelope to shot 1's before the weld. Constant per-shot gains, clamped
  +/-9 dB, half-strength in the top band so it cannot manufacture hiss.
  Paired A/B on the worst seed: HF loss halved (-49.5% -> -23.7%), rolloff
  drift cut to a third. It reduces the drift rather than eliminating it (the
  context_pin replay carries raw latents the EQ cannot reach), and it ships
  OFF until ears, not spectra, have judged it.
- **The Audio Spine produced static with real-world audio files (ref2va,
  user-reported).** The spine encoded `guide_audio` at whatever sample rate the
  file arrived in, while the audio VAE expects its own rate (32 kHz) - the
  native node resamples, the spine path did not. Nearly every real voice or
  music file is 44.1/48 kHz, so the encoded latent was garbage, and because
  the spine LOCKS the audio stream to that latent at every sampling step, the
  render came out as noise. The same file worked through the native
  `MiniMaxH3ReferenceToVideo` node, which is exactly what the reporter
  observed. The spine now resamples to the VAE's rate and upmixes mono to
  stereo, and the console says so. Measured: guide-to-output correlation went
  from **0.06** (unrelated noise) to **0.97** on a 48 kHz voice track -
  identical to a native-rate control. Also verified at 44.1 kHz mono.
- **The spine's tooltip claimed `latent_handoff` only - wrong.** It works with
  every continuity mode; the per-shot stride table has carried each mode's
  seam trim all along, and the fix above was render-verified on `context_pin`.
  This is the locked-audio music-video path, now documented as such.
- **`two_pass_upscale` is removed.** It spatially interpolated the raw latent
  between passes. H3's latent is not a spatially smooth representation, so the
  interpolated values landed off-manifold and pass 2, running at low sigma, had
  no room to pull them back. Every arm tested came back as colour-noise mush
  against clean single-pass controls - including at 14 steps / `beta57`, the
  recipe `SETTINGS.md` previously called render-verified, and including shot 1,
  which carries no pin at all. It was never a `context_pin` incompatibility;
  it did not work in any mode. The guard around it is gone with it.
- **`output_scale`** replaces it: a lanczos resize of each shot's finished
  frames, after decode, so it cannot leave the latent manifold and works with
  every continuity mode. It adds resolution, not detail - measured at
  **1.78x faster** than rendering the same output size natively (45.5s vs
  80.9s at 672x384) and visibly softer. Applied **per shot**, so a long chain
  never holds a full upscaled master in memory at once.
- **`upscale_model`**: optional `UPSCALE_MODEL` input for real detail synthesis
  (ESRGAN and friends via ComfyUI's own loader), per shot, at the model's own
  factor. Render-verified with RealESRGAN x2plus: 448x256 -> 896x512, and
  combined with `output_scale` it lands exactly on the requested size.
- **`video_latents` / `audio_latents` / `head_frames` outputs** on the memory
  sampler (issue #12). Every shot's latent exactly as sampled, batched along
  dim 0, untrimmed. Shots after the first open with `head_frames` of replayed
  material that is only removed at decode, so they do not line up with the
  master until you trim it - the outputs are deliberately raw rather than
  trimmed on your behalf, because the pin material cannot be recovered later.
  Verified: 124-frame shots return 37 latent rows (`5*((F-5)//17)+2`), and
  124 + 124 - 22 is exactly the 226-frame master.

Both upscales are applied after the memory bank has taken its base-resolution
reference clip, so conditioning and VRAM are unchanged from an un-upscaled run,
and the returned latents stay base-resolution.

**If you saved your own copy of a v2.1.2 graph**, reload the shipped workflow:
removing four widgets shifts the saved widget order on that node.

---

## Fixed in v2.1.2

Writer-half only (`ComfyUI_JoyAI_Echo_GGUF_Nodes`). If you paste your own shot
list instead of letting the LLM write it, nothing here changes for you.

- **Every shot can now be saved as it renders.** A chain only became a file at
  the very end, so anything that failed after the last shot destroyed the whole
  run — one report was three hours lost to an OOM at the mux, *after* every shot
  had rendered successfully. `save_every_shot` (both samplers) writes each shot
  to `output/video/H3_SHOTS/` the moment it decodes. Written before the seam
  trim, so consecutive files overlap ~1s and the master is still the clean join.
  Requested in issue #13.
- **Custom sigma schedules.** The samplers built the schedule themselves from
  `steps` + `scheduler` with no way to supply your own, so a turbo LoRA that
  ships the curve it needs simply ran wrong rather than refusing. Both samplers
  now take an optional `SIGMAS` input; connect one and it replaces the schedule,
  `steps` rebinds to `len(sigmas)-1` so the two-pass split rides your curve, and
  the console says the widgets are being ignored instead of silently overriding
  you. It is a link-only input, so saved graphs are unaffected. Issue #14.
- **`---` separators were ignored in passthrough mode.** `example_script.txt`
  ships `---` separated and every doc tells you to write scripts that way, but
  the writer's passthrough path returned the whole file as ONE shot — which the
  sampler then repeated to fill `shot_count`. Pasting a finished multi-shot
  script rendered the entire text as shot 1, four times. It now splits on the
  same rule the sampler uses. A single paragraph is still one shot, so `.txt`
  batches are unaffected.
- **Reference images had no way in.** The sampler's `reference_images`
  input has always existed, and `SETTINGS.md` documented it — as `unwired`,
  because nothing in the workflow was connected to it. There is now a
  **REFERENCE** lane in the anchors column (two image loaders → `ImageBatch` →
  a gate), shipped with the gate **off** so nothing changes until you turn it
  on. This is the one item here that is a new capability rather than a repair.
- **A stale prompt-set filename blocked the whole queue.** ComfyUI validates
  every combo value in a graph before it will run anything, so if
  `RiftPromptSource`'s saved `source_file` no longer existed — a renamed
  folder, a workflow shared from another machine, or simply the prompt lane
  switched to manual — the run died with `Value not in list` and *nothing*
  executed, including the lanes that were fine. The node now declares
  `VALIDATE_INPUTS`, so the filename is only resolved if the node actually
  runs; switching to manual genuinely disables it. If it does run and the file
  is missing, the error names the file.
- **Every story came out 15 shots.** The system prompt ordered exactly 15 when
  the brief gave no count. It now counts the story's beats — measured 4–7 on
  ordinary briefs, ~7 with no length signal (~65 s at 243 frames, past the
  1-minute mark). An explicit count is still honoured exactly.
- **The `mode` dropdown did nothing.** The shipped workflow's `system_prompt`
  box held a frozen copy of the long prompt, and a filled box overrides the
  per-mode file — so every mode ran the long prompt and pack prompt updates
  never reached anyone. It ships empty now. **If you saved your own copy of the
  v2.0/v2.1 workflow, clear that box by hand.**
- **`short_story` is 1–3 shots**, not always exactly 1.
- **Messy LLM JSON no longer kills the render.** A markdown fence sharing a line
  with the payload used to destroy it; truncated replies now have their complete
  shots salvaged; and parsing moved *inside* the retry loop, where it should
  always have been. Order: clean → parse → retry ×3 → salvage → fail.

## New in v2.0

- Complete single-purpose workflow **H3 Seamless Chain v2** (42 nodes, 9
  grouped lanes, 5 on-canvas notes), plus a **CORE** variant with zero
  third-party dependencies.
- **MASTER CONTROLS** panel (`H3StudioControls`): one node drives resolution,
  frames per shot and steps for the sampler *and* the prompt writer's dialogue
  pacing.
- **VRAM/SPEED** panel (`H3StudioSwitches` + reserve control): three lazily
  gated model patches, all OFF by default.
- **Energy-aware seam audio ("smart weld").** The boundary audio cut now lands
  in the quietest gap within the incoming shot's first 0.75 s instead of
  blindly at sample 0, so a word placed at a shot head is no longer clipped.
- **Rewritten activation-reserve heuristic** (payload-aware cache keys,
  measured pools no longer floored, sibling-based first-run estimates, VRAM
  spill detected and named).
- Prompt writer gained a **`join_style`** control that appends the
  render-verified boundary rules to the system prompt.
- `flf_chain` with no boundary plates now raises a clear error instead of
  silently rendering an unanchored chain.

Versions v1.0 through v1.5 shipped the same pack; v1.5's headline was voice
identity (`voice_ref` + `self_anchor_voice`).

## Verified and not verified

**Verified**

- A 3-shot `context_pin` chain and multi-shot `first_frame` chains were
  reviewed **blind** by two independent video-understanding models. One
  described the result as one continuous unedited take, colour consistent,
  with nothing broken.
- Verified on both static talking-head content and dynamic moving content.
- Blind-reviewed recipe (differs from the shipped defaults, which add
  `ref2va` reference rows for explicit voice/identity): `fl2va` checkpoint,
  `euler` sampler, `beta57` scheduler,
  14 steps, 362 frames per shot (about 15.1 s at 24 fps).
- Identity retention across a 40-second two-character scene with no reference
  images supplied.

**Not yet verified**

- **Very long chains.** Audio dulls slightly at each hop. Restart the chain on
  scene cuts rather than running one chain indefinitely.
- The `ref2va` + bank + `context_pin` combination.
- Hard-FFLF boundary-plate mode.

Nothing above is a benchmark. These are render observations from the recipe as
shipped; results will vary with content, resolution and quantisation.

## Release contents

`MiniMax-H3_Seamless_Chain_v2.0.zip`

```
ComfyUI-H3-Multishot/LICENSE
ComfyUI-H3-Multishot/README.md
ComfyUI-H3-Multishot/__init__.py               defensive loader
ComfyUI-H3-Multishot/apply_gguf_arch_patch.py  on-disk GGUF arch fallback
ComfyUI-H3-Multishot/h3_advanced.py            advanced sampling helpers
ComfyUI-H3-Multishot/h3_avbank_probe.py        AV bank diagnostics
ComfyUI-H3-Multishot/h3_cartridge.py           portable character cartridges
ComfyUI-H3-Multishot/h3_episode_tools.py       StudioControls, StudioSwitches, AnySwitch
ComfyUI-H3-Multishot/h3_gguf_arch.py           teaches ComfyUI-GGUF the minimax_h3 arch
ComfyUI-H3-Multishot/h3_interior_patch.py      interior anchors (stands down for Motion Context)
ComfyUI-H3-Multishot/h3_keyframes.py           keyframe anchor nodes
ComfyUI-H3-Multishot/h3_lora_stack.py          H3LoraStack
ComfyUI-H3-Multishot/h3_multishot_utils.py     samplers, loaders, gates
ComfyUI-H3-Multishot/h3_ref_folder.py          reference-folder picker
INSTALL.md
PROMPTING.md
SETTINGS.md
example_script.txt                             worked four-shot two-hander
workflows/H3_Keyframes.json                    single-clip keyframe anchoring
workflows/H3_Seamless_Chain_CORE.json          same job, zero third-party packs
workflows/H3_Seamless_Chain_v2.json            everything, optional lanes gated off
```

**Three workflows, one reason each.** `v2` is everything with the optional
lanes gated off. `CORE` does the same job with zero third-party packs — start
there if you want a render before installing anything else. `Keyframes` is a
different job: a hand-built sampling graph for anchoring a single clip at
chosen frame positions with per-anchor condition strength, not multishot.

`H3_Multishot_AIO` and `H3_Multishot_MEMORY` from earlier versions are retired
— every lane they had is in v2 (the AIO's episode source, plate chain and audio
spine were folded in; MEMORY had nothing v2 lacks). Existing copies keep
working.

## Links

- Guides: [the 5-minute guide](https://civitai.com/articles/34047/make-talking-videos-with-minimax-h3-the-5-minute-guide-26) and [every setting explained](https://civitai.com/articles/34046/every-setting-explained-the-seamless-chain-deep-manual)
- Source: <https://github.com/jlucasmcrell/ComfyUI-H3-Multishot>
- Civitai listing: <https://civitai.com/models/2833322>
- GGUF quants: <https://huggingface.co/joeygambino/MiniMax-H3-GGUF>
- Encoder and VAEs: <https://huggingface.co/Comfy-Org/MiniMax-H3>

## License

Apache-2.0 for the node pack and workflows in this repository. Model weights
are covered by their own licenses at their respective repositories.
