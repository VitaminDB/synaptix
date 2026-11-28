# Carasibana__ComfyUI-H3-FaceRefine

Source: https://github.com/Carasibana/ComfyUI-H3-FaceRefine
Fetched: 2026-11-28

Description: Refine and improve the quality of small faces in MiniMax H3 video. Per-frame face tracking, crop, refine with H3, stitch back.
Stars: 653  Updated: 2026-09-13T03:23:57Z



---

## FILE: README.md (https://github.com/Carasibana/ComfyUI-H3-FaceRefine/blob/HEAD/README.md)

# ComfyUI-H3-FaceRefine

**A ComfyUI custom node set to refine and improve the quality of small faces in MiniMax H3 video.**

MiniMax H3 renders faces poorly when the head occupies a small fraction of the frame. This is a property
of head-size-in-frame, not of output resolution, so it persists at 720p and above. These nodes
detect the face on every frame, crop to it so it fills a canvas, let H3 re-generate it, and
composite the result back into the original video.

Modelled on [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **FaceDetailer**,
adapted from stills to video.

---

## Results

Source on the left, refined on the right:

![source vs refined, side by side](screenshots/COMPARISON.gif)

Single frame at full resolution:

| Source | Refined |
|---|---|
| ![source frame](screenshots/INPUT_00001.png) | ![refined frame](screenshots/REFINED_00001.png) |

The reason it works is what H3 is handed. Instead of a distant head a few dozen pixels tall, it
gets the face tracked and normalized to fill the canvas:

| Crop in: what H3 sees | Crop out: what H3 returns |
|---|---|
| ![input crop](screenshots/CROPS_INPUT_00001.gif) | ![refined crop](screenshots/CROPS_00001.gif) |
| [full-res still](screenshots/CROPS_INPUT_00001.png) | [full-res still](screenshots/CROPS_00001.png) |

These two are what to watch for temporal behaviour: the box has to sit still on a moving subject,
or the refined face boils.

---

## What's new in 1.1.2

**[H3 Face Track + Crop](#h3-face-track--crop) no longer jumps to another person when it loses the
subject.** When the subject's face stopped being detected - a head turned away, a spin, a face
covered - while anyone else was in shot, the tracker moved onto the nearest other face and stayed
there, and the report still counted every frame as found. It now only carries the subject onto a
face within reach of where they just were. Frames with no such face are left unresolved: the crop is
interpolated across them and the composite fades out, so they keep their original pixels. The
subject is picked up again when their face reappears near where it was lost. Frames before a shot's
lock-on are filled back under the same rule, and shots that never hold more than one face track
exactly as before. See [When the subject is lost](#choosing-which-face).

**The report says when the subject was lost.** A new `lost:` line counts those frames and lists
their ranges, and `face=` and `dropout runs` include them, instead of reporting every frame as found.
In `preview` those frames are drawn red - see [Reading `preview`](#h3-face-track--crop).

**`fallback_detector` uses the body nearest where the subject's head should be**, rather than the
largest body in frame.

Nothing needs rewiring, and saved workflows load unchanged.

---

## What's new in 1.1.0

**[H3 Load Video + Face Select](#h3-load-video--face-select) is new.** It loads the video, finds
every face and every hard cut in one pass, and settles which face is the subject *before* the graph
runs, so the tracker never detects the same video twice. It brings two interactive surfaces with it:
[Pick faces](#pick-faces), for choosing the subject per shot by eye, and
[Preview coordinates](#preview-coordinates), for checking where an `X`, `Y` lands.

**That node adds one dependency: [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)**
(`scenedetect>=0.7`), which installs automatically with the pack. It backs
`cut_detection = auto (pyscenedetect)` on both the new node and `H3 Face Track + Crop`, finding
the **hard cuts** in a clip. Cuts matter because a cut renumbers every face and breaks continuity:
with them known, the subject is chosen once per shot and the smoothing, interpolation and
composite fade all run per shot instead of being dragged across the join. It is fed frame by frame
from the decode pass that is already running, so it costs no second decode of the video.
`cut_threshold` is PySceneDetect's own adaptive default of `3.0`.

It is a soft dependency in practice: if it is missing or fails to import, the run does not stop.
The video is treated as a single shot and the report says so — `cut detection unavailable,
treating the video as one shot`.

**[H3 Per-Frame Denoise](#h3-per-frame-denoise) now takes and returns `MODEL`.** A per-frame mask
needs two changes to the model rather than the latent, so the node must sit in the model path and
its `model` output has to reach the guider. Its two strength widgets were renamed
`denoise_multiplier_small_face` and `denoise_multiplier_large_face`, which is what they always were.

> **This breaks workflows saved against 1.0.0.** `model` is a *required* input, so an old graph fails validation with `Required input is missing: model` until you route the model through the node and take its `model` output onward to the guider. Nothing else in this release requires rewiring.

**[H3 Face Track + Crop](#h3-face-track--crop) gained** a `frame_count` output to wire into the H3
node's `length`, a `face_pick` input for the node above, `absent_shots` for skipping shots the
subject is not in, `identity_model` for choosing how faces are compared, cut detection, and the
expanded `select` list below.

**`select` went from two modes to nine.** 1.0 offered `largest` and `most_central`; there are now
`largest_face`, `smallest_face`, `left_most`, `right_most`, `top_most`, `bottom_most`, `centre_most`,
`closest_to_xy` and `detector_score`, with direction in the name rather than a separate order widget.
Saved workflows migrate on load: `largest` becomes `largest_face`, `most_central` becomes
`centre_most`.

**`canvas_width` / `canvas_height` now default to `768`**, H3's native short edge, rather than `512`.

**The auto canvas modes now clamp up to a minimum of 512×512.** They size the canvas from the
largest crop, and the crop is bounded by the source frame, so on a small face in a low-resolution
clip the canvas used to track the crop down to a couple of hundred pixels — handing H3 the same
small face it renders badly. `manual` is unaffected and still takes whatever you type.

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git
```

Restart ComfyUI. The nodes appear under **MiniMax H3/Face Refine**.

### Requirements

**For the nodes themselves:**

| | |
|---|---|
| ComfyUI with MiniMax H3 support | H3's own nodes are **core** (`comfy_extras/nodes_minimax_h3.py`), not an add-on, you just need a build recent enough to have them |
| a face detector | e.g. `face_yolov8m.pt` in `models/ultralytics/bbox/`. The one thing you must supply yourself. For anime, use an anime face model instead. The **nodes** find it there on their own; the **example workflows** additionally need the folder registration described below, because their stored detector values carry its `bbox\` prefix |

Python packages (`ultralytics`, `scipy`, `insightface`, `scenedetect`) install automatically
from `requirements.txt` / `pyproject.toml`. `scenedetect` is PySceneDetect, used only by
`cut_detection`; without it cut detection is skipped and the clip is treated as one shot.

> **A note on onnxruntime.** `insightface` needs it, and this pack deliberately does not pin
> a variant. If you install `onnxruntime-gpu` *alongside* an existing `onnxruntime`, the
> CPU-only package shadows it, `CUDAExecutionProvider` disappears, and identity matching
> silently runs on CPU. Install one or the other, not both. Check with:
> `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`

**Additionally, to run the example workflows.** The first three are needed to queue either template
as shipped; the last two back a node pair that ships **muted**:

| | |
|---|---|
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | **Both templates**, for the save (`VHS_VideoCombine`). **Auto Select** also loads the source with it (`VHS_LoadVideoPath`); **Manual Select** loads video with this pack's own `H3 Load Video + Face Select` instead, so it needs VideoHelperSuite only to save |
| [ComfyUI-H3-NativeAudioLock](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow) | **Both templates.** Supplies `MiniMaxH3NativeAudioLock`, which drives lipsync. It ships inside that repository under `custom_nodes/ComfyUI-H3-NativeAudioLock`, so copy that folder into your own `ComfyUI/custom_nodes/` |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | **Both templates.** Its subpack registers the `ultralytics_bbox` / `ultralytics` model folders, and the templates' detector values are stored in that registration's naming — `bbox\face_yolov8m.pt` and `segm\person_yolov8m-seg.pt`. Without it those values are not in the dropdown and the prompt fails validation on load. It also supplies `SAMLoader` if you unmute the SAM mask pair |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | only if you unmute the GGUF loader pair. Both workflows ship it muted in favour of the stock loaders |
| a SAM model in `models/sams/` | only if you unmute the SAM mask pair, alongside Impact Pack above |

Model lookups go through ComfyUI's `folder_paths`, so anything registered in
`extra_model_paths.yaml` is found automatically.

### Models

| Model | Goes in | Source |
|---|---|---|
| `face_yolov8m.pt` | `models/ultralytics/bbox/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/face_yolov8m.pt) |
| `person_yolov8m-seg.pt` *(optional for the nodes, but both templates ship it wired as `fallback_detector`)* | `models/ultralytics/segm/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8m-seg.pt) |
| an anime face detector *(optional, anime and other illustration)* | `models/ultralytics/bbox/` | [deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection), or Anzhc's face-seg models. `face_yolov8m.pt` may not find anime faces — see [Anime and other non-photographic material](#anime-and-other-non-photographic-material) |
| a CLIP Vision model *(optional, only for `identity_model = clip_vision`)* | `models/clip_vision/` | any CLIP Vision model, wired through a `CLIPVisionLoader`. You probably already have one for IPAdapter or Redux |
| MiniMax H3 diffusion model | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Qwen3-VL text encoder | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| a turbo LoRA *(optional, big speed-up)* | `models/loras/` | [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras), distilled by [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo). The example workflows ship the **8-step v1.0** at strength `1.0`, with `BasicScheduler` set to 8 steps to match — the step count must follow whichever turbo LoRA you load |
| `sam_vit_b_01ec64.pth` *(optional, only if you unmute the SAM mask pair)* | `models/sams/` | [Meta segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| InsightFace `buffalo_l` *(optional, crowd tracking on photographed faces)* | `models/insightface/` | downloaded automatically on first use. Backs `identity_model = insightface`; on illustration use `clip_vision` or `ccip` instead |

H3's original weights are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); the
Comfy-Org repository above is the repackaged form ComfyUI expects. The example workflows load the
model and text encoder through the stock `UNETLoader` and `CLIPLoader`. Their stored values name the
specific quantisations the examples were built against, so unless you happen to have those exact
files you will need to re-pick both loaders after opening a template. A **GGUF** pair
([ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)) sits beside them, muted, with a note:
unmute it and mute the stock pair to fit H3 in 12 GB.

Only the face detector is genuinely required by the nodes themselves. Everything else is needed
because the pipeline runs H3, so if you are already generating H3 video you will have it already.

---

### Reading videos from outside ComfyUI

`H3 Load Video + Face Select` takes a path to any video on your machine, because source footage
usually lives somewhere other than ComfyUI's `input` folder. Two things follow from that.

**Browse uploads; pasting does not.** The **Browse…** button copies the file you pick into
ComfyUI's `input` folder and refers to it by name from then on, exactly like the stock image
loaders. Pasting a full path instead leaves the file where it is and reads it in place — which is
what you want for a 4 GB clip you would rather not duplicate.

**Previews of pasted paths are served by this pack.** ComfyUI's own file server only reaches its
`input` folder, so a pasted path needs a route of its own. That route reads any file with a video
extension that ComfyUI itself can read, named by the request. ComfyUI listens on localhost only
unless you have deliberately changed that, so on a normal single-machine setup this is the same
access you already have.

If ComfyUI is reachable from anywhere else — `--listen`, a tunnel, a shared box — set this to
confine those routes to ComfyUI's own folder:

| variable | effect |
|---|---|
| `H3_FACEREFINE_STRICT_PATHS` | Unset (default): paths anywhere are read. Set to any non-empty value: only paths inside ComfyUI's own folder are read; everything else is refused with a message saying so. |

With it set, use **Browse…** to bring a video in rather than pasting a path to it.

Setting it, in the launcher you already use:

```bat
:: run_nvidia_gpu.bat, before the python line
set H3_FACEREFINE_STRICT_PATHS=1
```

```powershell
# PowerShell, for the session
$env:H3_FACEREFINE_STRICT_PATHS = "1"
```

```bash
# Linux / macOS
export H3_FACEREFINE_STRICT_PATHS=1
```

It is read per request, so it takes effect as soon as ComfyUI restarts with it set.

---

## Quick start

Two example workflows are included, both annotated in-graph. They differ in **how the subject
is chosen**; everything downstream is identical:

| Template | Which face gets refined | Video loaded by |
|---|---|---|
| **H3 Face Refine Auto Select** | a ranking rule, `largest_face` as shipped | your own video loader |
| **H3 Face Refine Manual Select** | you, in [Pick faces](#pick-faces), one per shot | `H3 Load Video + Face Select` |

Both ship the SAM mask pair present but **muted**, so neither needs a SAM model to run. Unmute the
pair to swap the rect paste mask for a face-shaped one, and drop `feather` to 4-8 when you do.
Impact Pack itself is still needed either way — see [Requirements](#requirements) — because the
templates' detector values use the model-folder naming its subpack registers.

Both are installed as ComfyUI **workflow templates**. Open them from
**Workflow → Browse Templates** once the pack is installed, no file hunting required.
They also live in [`example_workflows/`](example_workflows) if you would rather load the JSON
directly.

### Common to both

1. **Ref image(s)** — the same character references you generated the video with. `Ref 1` is the
   identity; `Ref 2` is optional, for wardrobe or a second look.
2. **Prompt** — the video's own prompt.

That is all either template needs beyond the video. `length` on the H3 node is already wired from
the track node's `frame_count`, so it follows whatever clip you load, and `width`/`height` are wired
from `canvas_w`/`canvas_h` so the crop and the latent cannot disagree.

Audio comes from the source clip. If you have an isolated vocal track for lipsync, unmute the
separate audio loader and wire it in place of the clip's own audio.

### Auto Select — a rule picks the face

Point `VHS_LoadVideoPath` at your video and queue. The tracker detects every face and takes the one
`select` names — `largest_face` as shipped, with `select_index` taking the 2nd, 3rd and so on out of
that ranking. See [Choosing which face](#choosing-which-face) for the full vocabulary.

The subject is chosen **once per shot**, not per frame, and continuity follows that same face from
there. If the clip has hard cuts, set `cut_detection` to `auto (pyscenedetect)` so each shot chooses again — a cut
renumbers everyone, so a rule can otherwise land on a different person after the join.

Use this when one rule describes your subject for the whole clip: they are the biggest face, or the
only face, or reliably centre frame.

### Manual Select — you pick the face

Here `H3 Load Video + Face Select` replaces the video loader and does the detection itself, handing
the tracker its boxes, its cuts and your choice on `face_pick`. The tracker never decodes or detects
the clip a second time.

1. **Point it at your video** — **Browse…** copies a file into ComfyUI's `input`, or paste any
   absolute path, since source footage usually lives elsewhere.
2. **Set `cut_detection` to `auto (pyscenedetect)`** if the clip has hard cuts, so it is split into shots.
3. **Click `Pick faces`.** The dialogue scans once, then shows a frame per shot with every face
   outlined and numbered. Choose your subject in each one and press **Use these** — see
   [Pick faces](#pick-faces).
4. **Queue.**

With `select` on `manual` the node will not run until a pick exists, so an unanswered graph fails
fast and says so rather than silently refining whoever happens to rank first. The other `select`
modes work here too if you would rather it choose: `identity_reference` matches a wired reference
image per shot, and the ranking rules behave exactly as they do on the tracker.

Use this when a rule cannot describe your subject: a crowd, a face that is not the biggest, or a
person who has to be followed across cuts that renumber everyone.

### How it fits together

```
FRONT END, one of the two:
  Auto Select    VHS_LoadVideoPath           -> images, audio
  Manual Select  H3 Load Video + Face Select -> images, audio, face_pick

  images ─────────────────────────────────────────────────────────────┐
     │                                            (face_pick, Manual) │
     ▼                                                                │
  H3 Face Track + Crop                                                │
     ├── canvas_w/h, frame_count ──► MiniMaxH3ReferenceToVideo        │
     │                               (refs + prompt)                  │
     │                                      │ av_latent               │
     ├── crops ──────────────────► H3 Inject Video Latent             │
     │                                      │                         │
     │        UNETLoader ► turbo LoRA ─┐    │                         │
     │                                 ▼    ▼                         │
     │                      MiniMaxH3NativeAudioLock ◄── audio        │
     │                                 │ model + av_latent            │
     ├── transform ──────────► H3 Per-Frame Denoise                   │
     │                            │ model        │ av_latent          │
     │                            ▼              ▼                    │
     │                   BasicGuider     SamplerCustomAdvanced        │
     │                   BasicScheduler           │                   │
     │                                            ▼                   │
     │                                        VAEDecode               │
     │                                            │                   │
     └── transform ──────────► H3 Face Stitch Back ◄──────────────────┘
                                       │
                          original audio ──► save
```

**`H3 Per-Frame Denoise` sits in the model path, not beside it.** It takes `MODEL` in and returns a
patched `MODEL` out, and that output must reach `BasicGuider` and `BasicScheduler`. Route the
unpatched model straight past it and the mask still applies but the held frames come back nearly
clean instead of at the sampler's current sigma — with no error to tell you.

The `transform` output is the spine of the whole thing: it records where every crop came from, so
the stitch can put each refined face back exactly where it belongs.

---

# Node reference

## H3 Load Video + Face Select

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select.png" alt="H3 Load Video + Face Select" width="380">

Loads a video, finds every face and every hard cut in one pass, and decides which face is the
subject **before the graph runs**. Optional: without it the tracker detects and ranks exactly as it
always has.

It replaces the video loader when used, so it emits the images and the audio as well as the
selection.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `video` | - | The video. **Browse** uploads one into ComfyUI's `input` folder, or paste any path - source footage usually lives elsewhere. Surrounding quotes are stripped, so a path copied from Explorer works as pasted. |
| `detector` | first found | Face detection model. This node owns detection when it is wired in — it detects once and hands the boxes to the tracker. |
| `confidence` | `0.35` | The **detector's** score floor - shown as `detector_confidence`. Lower catches more profiles and small faces, at the cost of false positives. Nothing to do with identity matching; that is `identity_threshold`. |
| `select` | `manual` | How the subject is decided. `manual` means you review the shots and choose; `identity_reference` matches a wired reference image; everything else is a rule that guesses - same vocabulary as the tracker, see [Choosing which face](#choosing-which-face). |
| `select_index` | `0` | Which face in that ranking is the subject. |
| `confirmed_pick` | - | Your reviewed answer while `select` is `manual`: **one face number per shot**, comma separated - `0,3,4`. Written by **Pick faces** and saved with the workflow. It is dropped automatically if you change the video, the detector, the frame range or the cut settings, because those change the shots it describes. |
| `cut_detection` *(opt)* | `none` | `auto (pyscenedetect)` finds hard cuts, so the subject is chosen per shot and the tracker does not smooth across a cut. |
| `cut_threshold` *(opt)* | `3.0` | How far a frame has to stand out from its neighbours to count as a cut. |
| `skip_first_frames` *(opt)* | `0` | Drop this many frames from the start. The audio is cut to match. |
| `frame_load_cap` *(opt)* | `0` | Stop after this many frames. 0 loads everything. |
| `select_every_nth` *(opt)* | `1` | Keep every nth frame. Above 1 the reported `fps` changes with it. |
| `identity_reference` *(opt)* | - | A picture of the person to refine. Each shot picks the face matching it, so the same person is followed across a cut without hand-picking. A frame of the clip works. Setting `select` to `identity_reference` without wiring one is a hard error. Wiring one while `select` is set to anything else is silent — the reference is simply never read. |
| `identity_clip_vision` *(opt)* | - | A `CLIP_VISION` model from a `CLIPVisionLoader`. Only for `identity_model = clip_vision`. |
| `identity_model` *(opt)* | `insightface` | Shown only when `select` is `identity_reference`. `insightface` for photographic faces; `clip_vision` or `ccip` for illustration and anime. See [Anime and other non-photographic material](#anime-and-other-non-photographic-material). |
| `identity_threshold` *(opt)* | `0.28` | Shown only when `select` is `identity_reference`. How close a match has to be to count as the same person. The scale depends on `identity_model` — **set `0` to use whichever default the chosen model recommends.** |
| `X` / `Y` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The point to measure from, in pixels of the **source** frame, origin top-left. See [Preview coordinates](#preview-coordinates). |
| `frame_index` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The frame the `X`, `Y` measurement is taken on, counting from `0` over the frames this node loaded — that is, after `skip_first_frames`, `frame_load_cap` and `select_every_nth`. |

**Outputs**

| Output | Goes to |
|---|---|
| `images` | `H3 Face Track + Crop` → `images`, and `H3 Face Stitch Back` → `base_images` |
| `audio` | The save node, and `MiniMax H3 Native Audio Lock` if you are driving lipsync |
| `face_pick` | `H3 Face Track + Crop` → `face_pick` |
| `preview` | Optional: one card per shot, every face outlined and numbered. This is how you find out which number to use |
| `report` | Text summary: shots found, and which frame each locked onto |
| `frame_count` | Wire to the H3 node's `length` so it follows the video |
| `fps` | Optional: the effective frame rate after `select_every_nth` |

**The subject is chosen once per shot, not once per video.** A cut renumbers everyone — the biggest
face before it need not be the biggest after — so `confirmed_pick` takes one index per shot. That is
also how you say *"these two differently-numbered faces are the same person"*: the ranking cannot
know that, so you tell it.

> **To see the numbers cheaply**, mute everything downstream of the tracker and queue. Only the
> detection pass runs — seconds, no H3.

### Pick faces

`select = manual` is answered here. **Pick faces** scans the video once, then shows a frame from
every shot with each face outlined and numbered, and writes your answer — one index per shot —
into `confirmed_pick`.

<img src="screenshots/Pick%20Faces%20UI.png" alt="The Pick faces dialogue" width="640">

Under each frame is a chip per detected face, plus **not in this shot**. The green chip is the
current pick, and the numbers on the chips are the numbers drawn on the boxes.

Those numbers run **left to right across the frame**, which is how the picker always numbers,
whatever `select` says. They are not `select_index` positions — in `manual` that widget is hidden
and the node ignores it entirely, because your pick already names the face.

**not in this shot** records `-1` for that shot: the subject genuinely is not present, so those
frames keep their original pixels instead of the tracker settling on somebody else. It is the
hand-picked equivalent of `absent_shots` on the tracker.

<img src="screenshots/Pick%20Faces%20UI%20-%20Multiple%20Faces.png" alt="Choosing between four faces" width="640">

**Across a cut the numbering is not stable.** A cut renumbers everyone, so the person who was face
`1` before it may be face `3` after. That is the whole reason the dialogue asks per shot rather than
once: it is where you say *"these differently-numbered faces are the same person"*, which no
ranking rule can work out for itself. The row along the top keeps your picks side by side so you
can check them against each other.

<img src="screenshots/Pick%20Faces%20UI%20-%20Multiple%20Scenes.png" alt="One pick per shot across three shots" width="640">

| Control | Does |
|---|---|
| **Use these** | Writes the picks into `confirmed_pick` and closes |
| **Cancel** | Closes and changes nothing |
| **Clear** | Empties every pick, to start over |

**Neither button will run while a render is going.** Both refuse rather than compete with the
sampler for the GPU. The scan dialogue offers a *Scan anyway* escape if you want it regardless;
**Preview coordinates** has none and simply declines until the queue is clear.

Scanning is cached: reopening the dialogue reports *reused the previous scan* and is immediate. The
scan is redone when the video, the detector, the confidence, the frame range or the cut settings
change, because every one of those changes the shots the picks describe. That cache belongs to the
dialogue alone — when the graph runs, the node detects the clip again for the render itself. What is
saved is the *tracker* repeating it, not the node. For the same reason
`confirmed_pick` is dropped automatically when they change, rather than being left pointing at
faces that have been renumbered underneath it.

Once picked, the button on the node names the choice, so the graph stays readable without
reopening the dialogue:

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select%20-%20Multiple%20Scenes.png" alt="Picks named on the node" width="340">

### Preview coordinates

`select = closest_to_xy` is typed rather than clicked, so the only real question is where the
numbers landed. **Preview coordinates** renders `frame_index` with a crosshair at your `X`, `Y`,
every face outlined, and the nearest one highlighted:

<img src="screenshots/H3%20Load%20Video%20%2B%20Face%20Select%20-%20XY%20Preview.png" alt="The X, Y coordinate preview" width="340">

The caption states what it resolved to — `frame 0 · 2 face(s) · nearest is face 0` — so the
point can be confirmed against the person you meant before anything is rendered.

The point does **not** have to land on a face. The nearest face *centre* wins at any distance, so
putting it roughly where the subject stands is enough. This preview sits on the node rather than in
a dialogue deliberately: the picture belongs beside the fields being edited, and a second dialogue would
read as a second way of choosing a face by hand, which `manual` already is.


## H3 Face Track + Crop

<img src="screenshots/H3%20Face%20Track%20%2B%20Crop.png" alt="H3 Face Track + Crop" width="380">

Detects the face on every frame, fills gaps where detection fails, smooths the trajectory, and
emits a constant-size batch of crops plus the `transform` needed to paste results back.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `images` | - | The video to refine. Connect your video loader. |
| `detector` | first found | Face detection model from `models/ultralytics/`. `face_yolov8m.pt` is a good default. |
| `confidence` | `0.35` | The **detector's** score floor - shown as `detector_confidence`. Lower catches more profiles and small faces, at the cost of false positives. Nothing to do with identity matching; that is `identity_threshold`. |
| `crop_factor` | `2.5` | Crop side as a multiple of face **height**. 2.5 puts the face at ~40% of the crop. Bigger gives more context so the seam lands in hair and background, but less magnification. **2.0-3.0 is the useful range.** |
| `canvas_width` / `canvas_height` | `768` | Resolution H3 generates at. In `manual` these are used **exactly as typed**, high or low. Ignored in the auto modes, which size the canvas from the crop instead. 768 is H3's native short edge; 512 costs 2.25× less in latent tokens. |
| `canvas_mode` | `manual` | `manual` uses the two values above, exactly as typed. `auto_no_downscale` sizes from the largest crop so no frame is ever downscaled. `auto_capped_768` does the same but clamps to 768, a sane VRAM ceiling and the best default. **Both auto modes clamp up to a minimum of 512×512**, whatever the crop: the crop is bounded by the source frame, so without that floor a small face in a low-resolution clip would size the canvas down to the crop and hand H3 the same small face it renders badly. `manual` is never clamped. |
| `smooth_window` | `21` | Frames of smoothing on the crop **centre**. 21 at 24 fps is ~0.9 s. Raise if the box shivers, lower if it lags fast head movement. |
| `size_smooth_window` | `51` | Frames of smoothing on the crop **size**. Deliberately larger than the centre window, because size jitter makes the crop breathe, which changes the resample factor every frame and reads as shimmer. |
| `smooth_method` | `gaussian` | `gaussian` rejects jitter best. `savgol` preserves the shape of a push-in at large windows. `moving_average` is a plain boxcar and leaves residual jitter. |
| `size_mode` | `per_frame` | `per_frame` holds the face at a constant fraction of every crop, which is correct for push-ins. `max_of_clip` uses one size throughout, only useful when the shot is static. |
| `identity_reference` *(opt)* | - | A clear face image of the person to track. Picks the subject by identity rather than size. **Read only while `identity_track` is on** — with it off the reference is never consulted and `select` decides instead, which the report warns about. |
| `identity_track` *(opt)* | `True` | Hold one subject through a crowd. Continuity decides most frames; the identity embedding is consulted only when two candidates are similarly plausible or their boxes overlap. |
| `identity_threshold` *(opt)* | `0.28` | Minimum score to accept a face as the reference person. Below it, the frame falls back to continuity, which is what carries tracking through profiles and occlusion. The scale depends on `identity_model` — **set `0` to use whichever default the chosen model recommends.** |
| `select` *(opt)* | `largest_face` | Which face is the subject. `largest_face`, `smallest_face`, `left_most`, `right_most`, `top_most`, `bottom_most`, `centre_most`, `closest_to_xy`, `detector_score`. See [Choosing which face](#choosing-which-face). |
| `fallback_detector` *(opt)* | `none` | Used only on frames where the face detector finds nothing. A person/body model gives a real head position from the top of the body box, which beats interpolating blindly. With several people in frame it uses the body whose head lands nearest where the subject is expected. |
| `fallback_head_frac` *(opt)* | `0.5` | Head centre as a multiple of face height below the top of the person box. 0.5 suits a head seen from behind. |
| `select_index` *(opt)* | `0` | Which face in that ranking to track. `0` is the first, `1` the second, and so on. |
| `identity_model` *(opt)* | `insightface` | Which model decides two faces are the same person. `insightface` for photographed faces, `clip_vision` or `ccip` for anime and other non-photographic material. See [Anime and other non-photographic material](#anime-and-other-non-photographic-material). |
| `cut_detection` *(opt)* | `none` | `auto (pyscenedetect)` finds hard cuts and treats each shot as its own window for the smoothing, interpolation and composite fade, so the crop is not dragged across a cut. See [Videos with cuts in them](#videos-with-cuts-in-them). |
| `cut_threshold` *(opt)* | `3.0` | How far a frame has to stand out from its neighbours to count as a cut. Only used when `cut_detection` is `auto (pyscenedetect)`. |
| `absent_shots` *(opt)* | `off` | Find shots the subject is not in, so they are never rendered. `by_identity` treats a shot where no sampled face matches the identity anchor above `identity_threshold` as not containing the subject: those frames keep their original pixels and drop out of the batch. Needs identity matching to be working — it does nothing without an anchor. Sampling is not exhaustive, so a brief appearance can be missed; `report` names every shot it drops and the score it saw. |
| `identity_clip_vision` *(opt)* | - | A `CLIP_VISION` model from a `CLIPVisionLoader`, used only when `identity_model` is `clip_vision`. |
| `X` / `Y` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The point to measure from, in pixels of the **source** frame, origin top-left. |
| `frame_index` *(opt)* | `0` | Shown only when `select` is `closest_to_xy`. The frame the `X`, `Y` measurement is taken on, counting from `0` over the frames handed to this node. |
| `face_pick` *(opt)* | - | From `H3 Load Video + Face Select`. Carries the detected boxes, the shot boundaries and the chosen subject, so this node does not detect the video a second time. `select`, `select_index`, `cut_detection`, `cut_threshold`, `X`, `Y` and `frame_index` are all greyed out while it is connected. `detector` greys too unless a crop-based `identity_model` still needs it; `confidence` greys unless either that or a `fallback_detector` does. |

**Outputs**

| Output | Goes to |
|---|---|
| `crops` | `H3 Inject Video Latent` → `images`, and `H3 Face Mask (SAM)` → `crops` |
| `transform` | `H3 Face Stitch Back`, `H3 Per-Frame Denoise`, `H3 Face Mask (SAM)` |
| `preview` | Optional: the video with the crop box drawn on every frame, colour-coded by how the subject was placed. See **Reading `preview`** below |
| `report` | Text summary: detections, gaps, frames where the subject was lost, magnification warnings |
| `canvas_w` / `canvas_h` | **Must** be wired to the H3 node's `width` / `height` |
| `frame_count` | The number of frames actually **rendered**, which is fewer than the clip when `absent_shots` drops a shot. Wire to the H3 node's `length` so it follows the video |

> **Reading `preview`.** Every frame carries the crop box, coloured by how the tracker placed it:
>
> | Colour | Meaning |
> |---|---|
> | green | the subject's face was detected on this frame and followed |
> | yellow | no face was detected; the head was placed from the body box found by `fallback_detector` |
> | red | no position for the subject on this frame - the face was not detected, or the subject was lost - so the box is interpolated from the frames either side |
>
> The composite fades out across yellow and red stretches, so those frames keep their original
> pixels. With `select` = `closest_to_xy` and no `face_pick`, an amber crosshair also marks `X`, `Y`.
> Saving `preview` as a video is the quickest way to check a track before spending a render on it.

> **Wire the canvas, don't type it.** In the `auto_*` modes this node chooses the size. If the H3
> node's `width`/`height` disagree, the latent shapes differ and injection refuses.

> Watch `report` for `magnification < 1.0x`. That means the crop is being *downscaled* into the
> canvas and real detail is being discarded. Raise the canvas, or skip videos that are close-up
> throughout since they have nothing to gain.

### Choosing which face

`select` names **which face**, and `select_index` takes the *n*th out of that ranking. They
apply only when no `identity_reference` is connected.

| `select` | picks |
|---|---|
| `largest_face` | the biggest face by **height** — the measurement the tracker uses everywhere else |
| `smallest_face` | the smallest by height, for refining the distant person rather than the near one |
| `left_most` / `right_most` | furthest left / right, by the **centre** of the face box |
| `top_most` / `bottom_most` | highest / lowest in frame, by the centre of the box |
| `centre_most` | nearest the centre of the frame |
| `closest_to_xy` | nearest the `X`, `Y` you give, measured on `frame_index` |
| `detector_score` | the detection the detector is most confident about |
| `identity_reference` | the face matching a reference image — `H3 Load Video + Face Select` only |
| `manual` | the face you chose in the picker — `H3 Load Video + Face Select` only |

Direction lives in the name, so there is no separate order widget: `left_most` and `right_most`
are two modes rather than one metric read forwards and backwards.

`closest_to_xy` takes three values of its own, shown only when it is selected. `X` and `Y` are
**pixels of the source frame** measured from the **top-left corner** — X increasing to the right,
Y increasing downward — so on a 960×544 clip (0, 0) is the top-left, (960, 544) the bottom-right,
and (480, 272) the middle. The point does not have to land on a face: the nearest face centre wins
at any distance. `frame_index` is the frame the measurement is taken on, counting from `0`, over
the frames the node loaded — so with `skip_first_frames` set it counts from the first frame kept.

`select_index` then picks out of that ranking: `0` is the first, `1` the second. To see which
number is who, put an **`H3 Load Video + Face Select`** in front of the tracker and connect its
`preview` to a `PreviewImage` — it emits **one card per shot**, every detected face outlined and
numbered, the chosen one highlighted in green.

The numbering on those cards follows **that node's own `select`**, and while it sits at its default
`manual` the cards are numbered left to right across the frame. So a number read off the preview
only lines up with the tracker's `select_index` when both nodes are set to the same ranking.

Three things worth knowing:

- **A rank is a per-frame property, not an identity.** The pick is made **once**, on the first frame
  that actually contains the requested index, and continuity carries the subject from there. Ranking
  every frame independently would hop between people the moment two of them cross or change size.
  The report prints the frame it locked on at.
- **Anything before that frame is interpolated.** If you ask for index 2 and the video opens on a
  single face, those opening frames have no index 2 to lock onto, so they are filled in from the
  lock-on frame and faded out of the composite, exactly like a detection dropout. The report counts
  them, and warns if there are more than a dozen.
- **The identity anchor follows your pick.** With no reference image the node normally assumes the
  biggest face is the subject — which used to override `centre_most` in a crowd. Now, once you
  select a different subject, the anchor is rebuilt from *your* pick, so identity matching holds
  that person rather than dragging the crop back onto the dominant face.

**Which mechanism decides.** Three things can answer "who is the subject", and they have a
fixed order:

1. **A reviewed pick** - `select` = `manual` on `H3 Load Video + Face Select`, carried on
   `face_pick`. You chose the face for each shot, so nothing overrides it.
2. **A connected `identity_reference`** - decides on its own, and `select` / `select_index` are
   ignored. The report says so.
3. **The ranking rule** - `select`, `select_index`.

Whichever wins only decides *where each shot starts*. From there the subject is held by
continuity - the nearest box to where they were, penalised for size change - and the identity
embedding is consulted only when two candidates are equally plausible or their boxes overlap.
On a typical video that is a couple of frames out of hundreds, and the report counts them.

**When the subject is lost.** In a shot with more than one face, continuity only carries the subject
onto a face within about three quarters of their own face height of where they just were, a little
more after a gap. When their face stops being detected and no face is within that reach, those
frames are left unresolved rather than handed to whoever is nearest: the crop is interpolated across
them, `preview` draws them red, the composite fades out so they keep their original pixels, and the
report lists them on its `lost:` line. The subject is picked up again when a face reappears within
reach of where they were lost. A subject who moves further than that while their face is hidden -
walking behind someone and out the other side - is not followed to the new position.

So a reference and a reviewed pick are not in conflict: the pick chooses the person, and the
reference still anchors the tie-breaking.

**If the automatic choice grabs the wrong person,** set `select` to `manual` and pick the face
yourself. That is what it is for. Lock-on deliberately accepts the best available match at any
score, so that small, poorly detailed faces - exactly the ones worth refining - are not
rejected for embedding weakly; the cost is that it can occasionally settle on the wrong face.

If the video never shows as many faces at once as you asked for, `select_index` is clamped to what
exists and the report says so.

### Videos with cuts in them

The crop trajectory is smoothed over the whole video. Across a hard cut the smoothing kernel spans
two shots and drags the box toward where the subject stood in the other one, for about half a
smoothing window either side. A dropout spanning the cut is worse: it is filled along a line that
existed in neither shot.

Set `cut_detection` to `auto (pyscenedetect)` and each shot becomes its own window for the smoothing, the gap
interpolation and the composite fade. Continuity also re-locks at each cut, since "nearest box to
where the subject was" means nothing once the camera has changed shot.

Three things worth knowing:

- **The canvas is still sized once for the whole video.** H3 generates one width/height for the
  batch, so a video mixing a wide shot with a close-up still sizes off the close-up. `max_of_clip`
  stays video-wide for the same reason.
- **A shot shorter than the smoothing window gets less smoothing**, not none — the window is clamped
  to the frames available. The crop can shiver on brief shots. The report names the shortest shot
  and warns when it is under the window.
- **A false cut costs smoothing on both sides of it.** The report says how many shots were found, so
  check it against what you know is in the video before trusting the result. Raise `cut_threshold` if
  it splits a continuous shot; lower it if it misses a cut between similar frames.

`H3 Per-Frame Denoise` splits its strength curve at the same boundaries, using the shot boundaries
the tracker publishes on `transform`.


### Anime and other non-photographic material

`buffalo_l` is ArcFace trained on photographed faces, and the node reaches it through InsightFace's
**own** detector (SCRFD, trained on WIDER FACE). On illustration that detector may struggle to match, so
there are no candidates to compare and identity matching quietly degrades to continuity. Nothing
goes visibly wrong — you simply never get the crowd handling.

Swapping only the recogniser would not fix that, because the candidates still have to come from
SCRFD. So the two other backends embed **the boxes your own detector already found**:

| `identity_model` | needs | good for |
|---|---|---|
| `insightface` *(default)* | nothing extra | photographed human faces. Unchanged. |
| `clip_vision` | a `CLIPVisionLoader` wired to `identity_clip_vision` | any domain — anime, 3D, stylised. No install. |
| `ccip` | `pip install dghs-imgutils` | anime specifically. Purpose-built for *"is this the same character?"* |

`ccip` is [CCIP](https://github.com/deepghs/imgutils), the illustration counterpart of ArcFace, and
it is the most accurate of the three on anime. It is **deliberately not a declared dependency** of
this pack: it pins `numpy<2` and pulls `opencv-contrib-python`, which shadows the OpenCV build
ComfyUI ships in exactly the way `onnxruntime-gpu` shadows `onnxruntime`. Install it yourself if you
want it; nothing else changes if you do not. Its ONNX models download on first use.

`clip_vision` needs no install, works on anything, and reuses a model you probably already have for
IPAdapter or Redux. It describes *appearance* rather than identity, so two characters with a similar
palette can collide, and its similarities sit high and close together.

**Also swap the detector.** These backends only see what `detector` gives them, so pair them with an
anime face model in `models/ultralytics/bbox/` — the usual choices are
[deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) or Anzhc's
face-seg models. A photographic `face_yolov8m.pt` may not find anime faces either.

**Setting the threshold.** `identity_threshold` means a different thing per backend: `0.28` suits
InsightFace cosine, `clip_vision` wants roughly `0.80`, and `ccip` scores `0.5` at its own published
operating point. **Set it to `0` to use the chosen model's own default.** The report prints the
scores it actually saw, and warns when the threshold cleared every candidate and so filtered
nothing — set it from that evidence rather than guessing.

**Multiple people:** run the pipeline once per subject, each with that person's `identity_reference`
and their own refs on the H3 node, then chain them, feeding run 1's stitched output in as run 2's
`base_images`. The composites accumulate.

---

## H3 Inject Video Latent (img2img)

<img src="screenshots/H3%20Inject%20Video%20Latent%20(img2img).png" alt="H3 Inject Video Latent (img2img)" width="380">

Encodes real frames into the **video** stream of H3's joint audio-video latent, leaving the audio
stream intact. This is the missing img2img path: H3's stock nodes always build a zeros latent, because
references are conditioning re-injected each step, never a starting point. Without this there
is no video-to-video.

**Inputs**

| Input | What to connect |
|---|---|
| `av_latent` | The `LATENT` output of `MiniMaxH3ReferenceToVideo` |
| `images` | `crops` from **H3 Face Track + Crop** |
| `vae` | The **video** VAE |

**Outputs:** `av_latent` (onwards to `MiniMaxH3NativeAudioLock`) and `report`.

It has no widgets. Strength is set downstream by `BasicScheduler`'s `denoise`, **not** by
`SplitSigmas`. See [Denoise](#denoise) below.

---

## H3 Per-Frame Denoise

<img src="screenshots/H3%20Per-Frame%20Denoise.png" alt="H3 Per-Frame Denoise" width="380">

Varies denoise strength along the temporal axis via the latent's noise mask, so one sampling pass
covers a shot that goes from distant to close.

A single denoise cannot serve a whole video: a tiny face has no detail to preserve and wants a
strong pass so the model *synthesizes*, while a large face has real detail and wants a gentle pass
so it is not rewritten. This node scales the base denoise per frame by measured face size.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `model` | - | The H3 model on its way to the sampler, from `MiniMaxH3NativeAudioLock`. **Required.** Making a per-frame mask behave takes two changes to the MODEL rather than to the latent: this node's video mask is kept out of H3's per-token timesteps, and the frames that mask holds back are re-noised to the step the sampler is on instead of being handed back nearly clean. |
| `av_latent` | - | From `MiniMaxH3NativeAudioLock` |
| `transform` | - | From **H3 Face Track + Crop**. This is where face sizes come from |
| `denoise_multiplier_small_face` | `1.0` | **Multiplier** on the denoise set on `BasicScheduler`, applied where the face is smallest. Not a denoise value in itself. Below `1.0` no frame in the video receives the full denoise you set, which makes the whole video gentler. |
| `denoise_multiplier_large_face` | `0.35` | **Multiplier** on the same value, applied where the face is largest. Lower preserves the real detail those frames already have, since a big face needs less rebuilding than a distant one. |
| `scale_mode` | `absolute_px` | `absolute_px` keys off real face size in source pixels, which is safe across a batch, since a video that never has a small face just sits at the baseline. `relative_to_clip` normalizes to that video's own min/max, so its smallest face always gets the full boost. Use the latter when tuning one video to its extremes. |
| `face_px_small` | `30.0` | Face height (source px) at or below which full `denoise_multiplier_small_face` applies. |
| `face_px_large` | `120.0` | Face height (source px) at or above which `denoise_multiplier_large_face` applies. |
| `gamma` | `1.0` | Curve on the interpolation. `>1` keeps strength high until the face is genuinely large; `<1` drops it off early. |
| `smooth_frames` | `9` | Smooths the strength curve over time. An abrupt denoise change between neighbouring frames shows up as a texture pop, so be generous. |

**Granularity is one latent frame, not one pixel frame.** H3 packs 17 pixel frames into 5 latents, so the finest step the ramp can take is ~3.4 frames. Values are averaged within each latent frame.

**Outputs**

| Output | Goes to |
|---|---|
| `av_latent` | `SamplerCustomAdvanced` → `latent_image` |
| `report` | Text summary of the strength curve |
| `model` | **`BasicGuider` → `model`.** The patched model. Wiring the unpatched model straight from `MiniMaxH3NativeAudioLock` to the guider costs you both changes above, without any error: the mask still applies, but the held frames come back nearly clean rather than at the sampler's current sigma. |

Base denoise and these multipliers are tuned **together**. The example workflows ship a base of
`0.40` which this node scales down on large-face frames. If you bypass this node, drop the base a
long way or every large face gets rewritten.

---

### Denoise

Denoise values do **not** transfer from SDXL-family models. H3 is flow matching with a large sigma
shift:

```
sigma = shift * t / (1 + (shift - 1) * t)
```

At the default shift of 12, `0.25`, an ordinary FaceDetailer value, lands at an effective sigma
of **0.800** and rewrites the frame.

| denoise | effective sigma (shift 12) |
|---|---|
| 0.02 | 0.197 |
| 0.05 | 0.387 |
| 0.15 | ~0.68 |
| 0.25 | 0.800 |

`steps` and `denoise` are **independent**: `BasicScheduler` builds a `steps/denoise`-long
full-range schedule and keeps the lowest `steps+1` sigmas, so 4 steps with a turbo LoRA is both
fast and gentle. Do **not** use `SplitSigmas`. On a 4-step schedule at shift 12 even the last
split point is already sigma 0.800.

Push the ceiling too high and the head drifts relative to the body, a content problem no mask can
hide.

---

## H3 Face Mask (SAM)

<img src="screenshots/H3%20Face%20Mask%20(SAM).png" alt="H3 Face Mask (SAM)" width="380">

Produces true face-shaped paste masks instead of a rectangle, temporally smoothed. Optional, and
requires Impact Pack for `SAMLoader`.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `crops` | - | **The tracker's `crops`, not the decoded result.** See the warning below. |
| `sam_model` | - | From Impact Pack's `SAMLoader` |
| `transform` | - | From **H3 Face Track + Crop** |
| `threshold` | `0.93` | SAM confidence threshold for accepting mask pixels. |
| `dilation` | `0` | Grow the mask. SAM masks are accurate, so they rarely need it. |
| `temporal_smooth` | `5` | Frames of averaging across the mask stack. `1` disables it, and the mask edge will shimmer. |

**Outputs:** `masks` (to `H3 Face Stitch Back` → `masks`) and `report`.

> **Mask the input, never the output.** Wire the tracker's crops here. This matches FaceDetailer,
> which computes its mask from the source image; generation never feeds back into the mask. Mask
> the *generated* result instead and, if the model nudges the face inward, the mask traces the new
> smaller silhouette while the original face pokes out past it, most visibly the nose on profile
> shots. It is also cheaper: no dependency on the sampler, so SAM need not be resident alongside
> the video model.

**Is it worth it?** Often not. A rect mask frequently beats SAM here. SAM traces the face tightly,
so any drift in the refined face lands right on the silhouette, whereas a slightly looser rect puts
the seam in hair and background where it reads far less. Try the rect workflow first.

> If you use SAM masks, drop `feather` on the stitch node to **4-8**. A rect needs far more.

---

## H3 Face Stitch Back

<img src="screenshots/H3%20Face%20Stitch%20Back.png" alt="H3 Face Stitch Back" width="380">

Warps each refined crop back onto the exact float box it came from, colour-matches it, feathers the
edge and composites. A single batched `grid_sample` does the warp, so a trajectory smoothed to
sub-pixel precision is not re-quantized on the way home.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `base_images` | - | The **original** frames, the same video fed to the tracker |
| `refined_crops` | - | `VAEDecode` output |
| `transform` | - | From **H3 Face Track + Crop** |
| `paste_region` | `face_only` | What actually composites. `face_only` / `face_ellipse` paste just the detected face box; `full_crop` pastes hair, shoulders and background too and risks a visible rectangle. |
| `mask_dilation` | `16` | Grow the face box before blurring, in canvas px, so the blur has room and the blend does not eat into the face. |
| `feather` | `6` | Gaussian blur radius on the paste mask, in **source pixels**, measured against the final frame, so the blend is the same physical width at any magnification. Use ~24 with a rect mask, 4-8 with SAM. |
| `colour_match` | `1.0` | Match the crop's per-channel mean/std to the region it replaces. The crop and the frame went through independent passes, so without this the face can come back subtly brighter and read as pasted on. |
| `blend` | `1.0` | Global opacity of the refined face. Below 1.0 mixes back toward the original, useful to dial back over-sharpening. |
| `undetected_frames` | `fade_out` | What to do where no face was found. **All** frames still go through H3 either way, which is what keeps it temporally consistent. This only controls pasting. `fade_out` ramps the composite to zero across the gap; `skip` hard-cuts to original pixels; `composite_anyway` risks H3 hallucinating a face onto the back of a head. |
| `feather_scales_with_crop` *(opt)* | `False` | Legacy: treat `feather` as canvas pixels so the blend narrows as the crop shrinks. Leave off. |
| `masks` *(opt)* | - | Per-frame masks from **H3 Face Mask (SAM)**. Overrides `paste_region`. |

**Outputs:** `images`, the finished frames. Send to your video save node.

> **Only the face region composites.** The wide crop exists to give the sampler *context*. It is
> not what gets pasted. Pasting the whole crop covers roughly 88% of the canvas versus 16% for the
> face box, and any change the model made to hair or background returns as a rectangle.

---

## H3 Face Transform Info

<img src="screenshots/H3%20Face%20Transform%20Info.png" alt="H3 Face Transform Info" width="380">

A debug node. Prints the per-frame boxes so you can sanity-check tracking before spending time on a
sampling pass.

| Input | Default | What it does |
|---|---|---|
| `transform` | - | From **H3 Face Track + Crop** |
| `max_rows` | `12` | How many frames to print. |

**Outputs:** `info` (a string). It is an output node, so it displays in the graph directly.

Use it when the stitch looks misaligned, or to confirm gap-filling behaved on a video where the
subject turns away.

---

## Lipsync

H3 is a **joint** audio-video model. `MiniMaxH3NativeAudioLock` encodes real audio into the audio
stream of the AV latent, sets `noise_mask` to ones for video and zeros for audio so only video
denoises, and the video branch cross-attends to that fixed audio. That is what shapes the mouth.

Feed it an isolated **vocals** track for a cleaner signal. The **original** audio goes separately
into the save node. Two distinct audio paths, easy to confuse.

---

## Gotchas

- **Frame count is snapped up to H3's 17k+5 grid** (5, 22, 39 … 175, 226, 362). Videos generated by
  H3 already sit on it. Anything else is rounded up: the extra frames are padded with reference
  content, refined, then discarded on stitch. Both the select and track nodes say so in their report
  and name the nearest grid value below. It costs sampling time and nothing else, but `frame_load_cap`
  will trim to a grid value if you would rather not pay it.
- **Cost is `canvas² × frames`.** Auto canvas sizing considers face size only, so on a long video it
  can pick a canvas that exceeds VRAM and falls back to streaming weights from system RAM, an
  order-of-magnitude slowdown rather than a clean error.
- **`SAMLoader`'s `AUTO` device mode leaves the model on CPU** until `prepare_device()` is called.
  This pack calls it; if you write your own SAM path, missing it makes mask passes 10-50× slower
  with the GPU idle at half power.

---

## Credits

The compositing approach is taken directly from
**[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** by
**[ltdrdata](https://github.com/ltdrdata)**, specifically `FaceDetailer` and its detailer paste.
The dilate-then-blur face-region mask, the crop-for-context-but-paste-only-the-face principle, and
the bbox+SAM masking path are all its design; this pack adapts them to a per-frame video pipeline.
If you find this useful, that project is why.

The lipsync path in the example workflows depends on `MiniMaxH3NativeAudioLock`, from
**[MiniMax-H3-NativeAudio-MusicVideo-Workflow](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow)**
by **[Shrek3OnVH5](https://github.com/Shrek3OnVH5)**. It is not redistributed here, so install it
from that repository.

Contributions:

- **[fatemark](https://github.com/fatemark)** — rank-and-index face selection, the numbered face
  preview, and the pluggable identity backends that make crowd tracking work on illustration
  ([#10](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/10))
- **[Gourieff](https://github.com/Gourieff)** — fixed a `SyntaxError` that stopped `nodes.py`
  importing on Python older than 3.12
  ([#12](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/12))

Also builds on:

- **[MiniMax H3](https://github.com/MiniMax-AI)**, the joint audio-video model being refined
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous
- **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** by city96
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for detection
- **[PySceneDetect](https://github.com/Breakthrough/PySceneDetect)** for hard-cut detection
- **[InsightFace](https://github.com/deepinsight/insightface)** for identity embeddings

---

## Made with these nodes

[![More Than Words Can Ever Steal](https://img.youtube.com/vi/18iiffk-QWE/maxresdefault.jpg)](https://www.youtube.com/watch?v=18iiffk-QWE)

**[More Than Words Can Ever Steal](https://www.youtube.com/watch?v=18iiffk-QWE)**, a music video by
[Carasibana](https://www.youtube.com/@Carasibana-Music), generated with MiniMax H3. Every shot
where the subject sits at any distance from camera was face-refined with this pack.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Built with [Claude Code](https://claude.com/claude-code).</sub>


---

# ISSUES / PRs of Carasibana/ComfyUI-H3-FaceRefine


### #19 [closed] ISSUE: H3PerFrameDenoise swallows the denoise: output pixel-identical at scheduler denoise 0.4/0.65/1.0; works only when node is bypassed
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/19  by lexyaowork 2026-09-21T23:17:46Z

## Environment
- ComfyUI: v0.36.0 (release build), `--lowvram --use-sage-attention`, base-directory layout
- Node version: latest `main` (nodes.py identical to repo HEAD as of 2026-09-22)
- H3 weights: `minimax_h3_fl2va_pruned_int8_convrot` + `minimax_h3_fl2v_turbo_8step_v1.0` LoRA @1.0, qwen3vl_32b encoder, fp8mix video VAE
- Test clip: 107 frames, 1344x768, face ~200px tall (falls in the large-face band, multiplier 0.35)

## Symptom
`H3PerFrameDenoise` appears to swallow the denoise entirely: the scheduler denoise dial has no measurable effect on the output. The log line prints as expected:

```
[H3FaceRefine] per-frame denoise: face 198-207px, ramp 30-120px (absolute_px)
  ->  MULTIPLIER 0.35 ... applied to the denoise set on BasicScheduler
```

…but the stitched output is pixel-identical (within VAE roundtrip + h264 noise) across ALL dial settings.

## Isolation matrix (pixel diff vs original, face ROI mean abs gray delta)

| Config | Scheduler denoise | ROI mean delta |
|---|---|---|
| via H3PerFrameDenoise (stock wiring, NativeAudioLock **not** installed, model direct from LoRA) | 0.40 | 1.61 |
| same | 0.65 | 1.61 |
| same | 1.00 (full regen!) | 1.60 |
| with MiniMaxH3NativeAudioLock installed, template-exact wiring | 0.40 | 1.61 |
| sampler switched er_sde → euler | 1.00 | 1.60 |
| service restarted WITHOUT --use-sage-attention | 1.00 | 1.61 |
| **H3PerFrameDenoise bypassed** (H3InjectVideoLatent → SamplerCustomAdvanced direct, guider model direct from LoRA) | 1.00 | **5.9** (2.9% of frame >18) |
| bypassed | 0.45 | 2.8 |

The constant ~1.6 floor equals VAE encode→decode roundtrip + h264 re-encode; every configuration that routes the latent through H3PerFrameDenoise lands exactly on that floor regardless of the dial, including denoise=1.0. Bypassing the node makes the dial work immediately.

## Repro prompt shape
```
H3FaceTrackCrop → [SAM mask pair] → H3InjectVideoLatent(av_latent from MiniMaxH3ReferenceToVideo slot1, images=crops, vae=video vae)
H3PerFrameDenoise(model=LoRA(H3), av_latent=inject, transform=trackcrop)
  widgets: denoise_multiplier_small_face=1.0, large=0.35, scale_mode=absolute_px, face_px_small=30, face_px_large=120, gamma=1, smooth_frames=9
BasicScheduler(simple, steps=8, denoise=0.4..1.0, model=node output 2)
BasicGuider(model=node output 2) + SamplerCustomAdvanced(er_sde)
```
Everything else per your Auto_Select template (canvas auto_capped_768, face_yolov8m + person_yolov8m-seg fallback, insightface identity).

## Hypothesis
The model patch that rewrites the per-frame denoise on the scheduler appears to neutralize sampling instead of scaling it in this environment (possibly interacting with this ComfyUI release's guider/sampler internals, or with the av-latent audio channels path when audio lock is absent — though the template-exact AudioLock wiring reproduced the same flat result).

Meanwhile, bypassing the node (InjectLatent → sampler direct) gives a fully working refine pipeline with the scheduler denoise as the only dial — sharing in case others hit the same flat-output symptom.

> **Comment by Carasibana (2026-09-22T17:41:54Z):**

Thanks for the detailed report and the test matrix, that made this easy to follow.

Your faces are 198-207px, and the default ramp is face_px_small=30 to face_px_large=120, so every frame of your clip sits past the large end and gets denoise_multiplier_large_face (0.35). Large faces are refined much more lightly than small ones by design, since they usually already have the detail that small faces lack.

The setting to raise is denoise_multiplier_large_face. Its effect is strongly non-linear rather than proportional: the original frame is blended back into the result on every sampler step, so what survives is the last step's correction scaled by that multiplier. At 0.35 that leaves far less than a third of the change, which is why the BasicScheduler dial has so little left to act on. Expect to go well above 0.35 before large faces visibly move. At 1.0 the video gets the full denoise you set on BasicScheduler.

One caveat on 1.0: it is not the same as bypassing the node. With no MiniMaxH3NativeAudioLock upstream, the node also writes a full lock over the audio half of the latent, so the audio is held and the video is generated against a held audio track. For a true bypass, bypass the node.

Worth knowing for this clip specifically: your face size barely varies across the 107 frames, so there is no per-frame ramp here for the node to apply. Whatever you set, it acts as one flat multiplier. Raising face_px_large is the other way to land mid-range, for example 400 would put a 200px face at about 0.70.

If you raise it and still cannot get the amount of refinement you want on large faces, say so here and I will take another look.




### #18 [closed] ISSUE: [Bug]: Weights set to true error with pytorch version: 2.11.0+cu130
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/18  by smithian123 2026-09-17T15:54:27Z

### Pre-flight

- [x] I read the setup steps in the README and followed them.
- [x] I searched the existing issues, open and closed, and this is not a duplicate.
- [x] This is a defect in the pack — not a question, a feature request, or a request to switch to a different library.
- [x] `length` on the H3 node equals my clip's exact frame count, and that count sits on H3's 17k+5 grid (5, 22, 39 … 226, 362).
- [x] The crop canvas and the H3 node's width/height match each other.

### What happened, and what did you expect instead?

Hi,
the workflow is failing early on with an error possibly related to weights being set to 'True' instead of 'False'. However, I tried with the help of google gemini to apply the recommended patches to no avail. I'd say it's due to cutting edge nature of my comfyui set-up (pytorch version: 2.11.0+cu130). That said, this is what came with the portable installation I grab from the official site. If you can patch it to work or implement an onnx based version it would be appreciated.

Thanks, Ian



### Steps to reproduce

1. Load example workflow H3 Face Refine Auto
2. Load character sheet and video previously generated using character sheet
3. select bbox\face_yolo.pt models
4. ensure base models, loras etc. are correct
5. run workflow
6. error occurs in relation to H3 Face Track + Crop node almost immediately

### Your workflow

[H3_Face_Refine_Auto_Select-test.json](https://github.com/user-attachments/files/32344268/H3_Face_Refine_Auto_Select-test.json)

### Node reports

```shell
Not reaching this stage to report on.
```

### Console output

```shell
[INFO] got prompt
[ERROR] [Impact Pack/Subpack] UnpicklingError during safe load (not 'getattr' related): Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default. Please use `torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])` or the `torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.. Re-raising.
[ERROR] !!! Exception during processing !!! Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default. Please use `torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])` or the `torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.
[ERROR] Traceback (most recent call last):
  File "E:\ComfyUI_win_portable\ComfyUI\execution.py", line 541, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI_win_portable\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI_win_portable\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\ComfyUI_win_portable\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\ComfyUI_win_portable\ComfyUI\custom_nodes\ComfyUI-H3-FaceRefine\nodes.py", line 1369, in run
    res = _get_model().predict(bgr, conf=confidence, verbose=False)[0]
          ~~~~~~~~~~^^
  File "E:\ComfyUI_win_portable\ComfyUI\custom_nodes\ComfyUI-H3-FaceRefine\nodes.py", line 1287, in _get_model
    _model_box.append(_load_detector(detector))
                      ~~~~~~~~~~~~~~^^^^^^^^^^
  File "E:\ComfyUI_win_portable\ComfyUI\custom_nodes\ComfyUI-H3-FaceRefine\nodes.py", line 60, in _load_detector
    model = YOLO(path)
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\ultralytics\models\yolo\model.py", line 23, in __init__
    super().__init__(model=model, task=task, verbose=verbose)
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\ultralytics\engine\model.py", line 145, in __init__
    self._load(model, task=task)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\ultralytics\engine\model.py", line 285, in _load
    self.model, self.ckpt = attempt_load_one_weight(weights)
                            ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\ultralytics\nn\tasks.py", line 909, in attempt_load_one_weight
    ckpt, weight = torch_safe_load(weight)  # load ckpt
                   ~~~~~~~~~~~~~~~^^^^^^^^
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\ultralytics\nn\tasks.py", line 836, in torch_safe_load
    ckpt = torch.load(file, map_location="cpu")
  File "E:\ComfyUI_win_portable\ComfyUI\custom_nodes\ComfyUI-Impact-Subpack\modules\subcore.py", line 294, in torch_wrapper
    raise e # Re-raise other UnpicklingErrors
    ^^^^^^^
  File "E:\ComfyUI_win_portable\ComfyUI\custom_nodes\ComfyUI-Impact-Subpack\modules\subcore.py", line 216, in torch_wrapper
    return orig_torch_load(*args, **load_kwargs)
  File "E:\ComfyUI_win_portable\python_embeded\Lib\site-packages\torch\serialization.py", line 1578, in load
    raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
_pickle.UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
        (1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
        (2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
        WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global by default. Please use `torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])` or the `torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html
```

### Environment, models and clip

Pack version:
ComfyUI version:
OS:
GPU and VRAM:
Diffusion model file:
Text encoder / CLIP file:
LoRA(s) and strength:
Detector model (bbox / segm):
Clip: WxH, frame count, fps:


### Screenshots or clips (optional)

_No response_

> **Comment by Carasibana (2026-09-20T11:59:31Z):**

Hi Ian, thanks for the detailed report. The traceback has the answer in it.

This is a version mismatch inside your install rather than something the pack sets.

PyTorch 2.6 changed torch.load so that by default it refuses to load a file that rebuilds Python objects, which is exactly what a YOLO .pt detector does. Current ultralytics handles this itself: its loader tells torch to read the file the old way. Your traceback shows ultralytics calling torch directly, without that:

```
ultralytics\nn\tasks.py", line 836, in torch_safe_load
    ckpt = torch.load(file, map_location="cpu")
```

So the ultralytics in your install predates that fix, and on PyTorch 2.11 the load is refused. The Impact Subpack lines are downstream of the same thing. Subpack replaces torch.load and passes the caller's settings through, so it reports the refusal rather than causing it, and its whitelist only covers a different error.

Updating ultralytics should be all it takes:

```
E:\ComfyUI_win_portable\python_embeded\python.exe -m pip install -U ultralytics
```

Then restart ComfyUI and run the workflow again. To see what you have first:

```
E:\ComfyUI_win_portable\python_embeded\python.exe -m pip show ultralytics
```

On an ONNX based detector: that is a separate feature rather than a fix for this, so I would keep it out of this issue.

Closing this, feel free to reopen if needed.




### #14 [closed] ISSUE: H3FaceTrackCrop: manual-pick continuity re-locks onto a different, non-adjacent person after a brief detector-miss (head turn)
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/14  by designloves2 2026-09-11T23:00:28Z

## Summary

`H3FaceTrackCrop` in manual-pick mode can silently re-lock onto a **different, non-adjacent person** when the tracked subject's face is briefly undetected (e.g. turning their head away for a few frames) in a multi-person shot. Once it re-locks onto the wrong person, continuity keeps following them for the rest of the clip — the switch is a hard jump, not a gradual drift.

## Setup

- 3 people standing still, side by side, in one continuous shot (no real cuts; `cut_detection` is also unavailable in this environment: `No module named 'scenedetect'`, confirmed via node's own log line, so it isn't a false-cut issue).
- `select="manual"`, one face picked via `face_pick` (a `confirmed_pick` index, locked correctly at frame 0 in every test — the report line always shows the correct starting box, e.g. `face 2, locked at frame 0`).
- `identity_track` tested both `True` and `False` — **identical symptom either way**, so this isn't an embedding/identity-model issue.
- `confidence` lowered from 0.35 toward ~0.15–0.2 — the subject is held correctly for longer, but the switch still happens right at the moment the tracked person turns their head away from the camera for a few frames (a real, brief detector miss on that one face, not a general detection failure).

## Reproduction (observed pattern)

Every run logs the same single dropout:

```
[H3FaceRefine] dropout runs: 1  longest=5 frames (0.2s at 24fps)  -> composite fades out across these
[H3FaceRefine] tracking: 237 by continuity, 1 ambiguous (1 resolved by face identity)
[H3FaceRefine] identity: insightface  threshold=0.450  scores min=0.539 mean=0.539 max=0.539
[H3FaceRefine] !! every identity score cleared identity_threshold=0.450, so it rejected nothing. Lowest seen was 0.539.
```

With `identity_track=False` the same single ambiguous frame instead falls through to raw continuity (`_continuity_cost` nearest-box), and still resolves onto the wrong (non-adjacent) person — so the mis-resolution isn't specific to the identity path either.

## Hypothesis

Around the multi-frame dropout, `last` (the subject's last known `(cx, cy, size)`) goes stale for several frames. When the subject reappears, `_continuity_cost(box, last)` compares fresh boxes against that stale position — if the true box has moved even a little during the gap while a different, uninvolved person's box happens to score better against the aged `last`, continuity locks onto them instead. Once locked, everything downstream just keeps following that wrong person, since there's no mechanism to notice the switch (the single frame where it happens is marked "ambiguous"/"conflict" but resolution — either identity or raw nearest-box — still picks wrong in this case).

## Suggested directions (not verified/implemented, just from reading the code)

- Widen or age-weight `_continuity_cost`'s penalty when `last` is more than 1 frame stale, so it's less confident about "nearest" after a multi-frame gap.
- When a conflict is detected AND the fallback lands on a box far from `last`'s pre-dropout trajectory (e.g. IoU with any recent box history is ~0), maybe prefer holding the crop from the last known box (or marking the frame absent/interpolated) over confidently switching subjects.

## Environment

- ComfyUI 0.35.0, Windows, `H3FaceTrackCrop`/`H3FaceSelect` from this pack, called via a custom wrapper node that builds the same graph the stock nodes would.
- `detector=face_yolov8m.pt`, `confidence=0.35` (and separately tested lower), `crop_factor=2.0/2.5`, `canvas_mode=auto_capped_768`, `smooth_window=21`.

Happy to share the full per-run report text or a redacted sample clip if useful — let me know what would help narrow this down.


> **Comment by designloves2 (2026-09-12T03:48:16Z):**

Extra data point, since it narrows the repro down further:

- On a clip where none of the 3 subjects turn away from camera at all, tracking holds the picked subject correctly the whole way through - no mis-lock.
- The failure only shows up on a clip where each subject does one full in-place spin (so they briefly show their back to camera mid-spin, then face forward again). The mis-lock happens right around that back-turn stretch, not just a couple of interpolated frames of a partial turn.

So this isn't a general continuity weakness - it's specifically the "subject fully turns away (back to camera), then faces camera again" case: a longer, harder gap where the picked subject's face is genuinely absent for a real duration (not just briefly occluded/blurred), and re-acquiring them afterward is where it locks onto a different, non-adjacent person instead.



> **Comment by Carasibana (2026-09-13T01:10:14Z):**

Confirmed issue. I just encountered this separately myself . 


> **Comment by Carasibana (2026-09-13T03:28:29Z):**

Thanks for the detailed report.

This is fixed in 1.1.2, now on the Comfy Registry and GitHub. Update through ComfyUI Manager or pull the latest from GitHub.

When the subject's face stopped being detected (turning away, a spin, being covered) while other faces were still in shot, H3 Face Track + Crop moved onto the nearest other face and stayed on it. It now only continues the subject onto a face within reach of where they just were. Frames where it can't are left unresolved: the crop is interpolated across them and the composite keeps the original pixels there. Tracking picks the subject up again when their face reappears nearby. This works the same with identity_track on or off.

The report has a new lost: line listing those frames, so face= will no longer read 100% on clips like yours, and the preview output draws them in red.

If you get a chance to run your three-person clip on 1.1.2, let me know how it goes.

Closing this for now, feel free to reopen if it still happens.



> **Comment by designloves2 (2026-09-13T13:10:50Z):**

Updated to 1.1.2 and re-ran the same three-person spin clip — the mis-lock is gone, tracking holds the picked subject correctly through the turn now. Thanks for the quick fix!




### #13 [open] PR: Batch face detection; warn when InsightFace falls back to CPU / 批处理人脸检测；InsightFace 退化到 CPU 时给出警告
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/13  by caoddgao 2026-09-09T12:11:51Z

Two small changes to `nodes.py`: one performance fix in the detection stage, and one warning for a failure that is currently silent. Both are backward compatible — no node's inputs, outputs, or types change.

*(A third change — a VRAM pre-check for the auto canvas modes — was originally part of this PR and has been withdrawn. I measured it and the premise turned out to be wrong; see the comment below for the numbers, in case they save anyone else the same detour.)*

## 1. Batch face detection instead of looping frame by frame

`H3FaceTrackCrop.run()` detected the clip with a per-frame Python loop: one `_to_bgr_u8()` call, which is a blocking GPU→CPU sync, plus one `YOLO.predict()` call per frame. The GPU sits idle between kernel launches and the cost grows linearly with clip length.

This is the same pattern the comment in `H3FaceStitch.run()` already describes having fixed:

> `# ---- GPU, batched. The previous version was a per-frame Python loop on CPU tensors: measured at ~1 core of 24 busy with the GPU idle, ~8 minutes for 362 frames.`

Detection looked like the one stage that had not had the same treatment yet.

Both loops now call a shared `_detect_clip()`:

- `_to_bgr_u8_batch()` does the scale and the RGB→BGR flip on whatever device the clip already sits on, so one transfer moves a whole batch, and it moves `uint8` rather than `float32` — a quarter of the bytes.
- `YOLO.predict()` takes the batch as a list, so Python call and kernel launch overhead is paid once per 16 frames instead of once per frame.
- Frames are converted **per batch, not for the whole clip at once**, so peak extra host memory stays at `batch*H*W*3`. Converting a long clip up front would allocate gigabytes; I measured it and it buys no extra speed, so the bounded version is the one here.
- **Cut detection is deliberately left unbatched.** PySceneDetect is a stateful stream that has to see frames in order, one at a time, so it still consumes them sequentially — it just reads the frames the batched conversion already produced, exactly as before.

`all_boxes` / `all_confs` are still built per frame, in order, so nothing downstream sees a different shape of data.

### Measured

243-frame 544×960 H3 clip, `face_yolov8m`, RTX 4070 Ti SUPER:

| | before | after | |
|---|---|---|---|
| detection loop, clip on GPU | 11.16 ms/frame | 4.08 ms/frame | **2.74×** |
| detection loop, clip on CPU | 12.79 ms/frame | 5.89 ms/frame | **2.17×** |

Both cases are measured because loaders differ in where they leave the IMAGE tensor. The full `H3FaceTrackCrop.run()` came down from ~3.4 s to ~1.6 s on the same clip, but that figure includes unchanged work, so the per-frame detection numbers above are the more honest ones.

I also swept the batch size: past 16 frames it stops buying anything back and only costs more VRAM, so 16 is the default.

### A note on scope

While making this change I noticed `H3FaceSelect.run()` contains a **line-for-line identical copy** of the same per-frame loop. That was not part of what I originally set out to fix — I found it while working on `H3FaceTrackCrop` — but fixing only one of two identical loops seemed worse than fixing both, particularly since `H3FaceSelect` is the entry node in `H3_Face_Refine_Auto_Select.json`. Both now share `_detect_clip()`, which also removes about 20 lines of duplication. If you would rather keep that node untouched, I am happy to split it out.

### Equivalence checking

I did not want to take "it's faster" on trust, so I ran the original `nodes.py` and the modified one side by side as two separate modules, over the same clip, through the **full node** — not just the detection helper — across six parameter combinations (defaults, `cut_detection=content`, `select_index=1`, `auto_capped_768`, `size_mode=max_of_clip`, `identity_track=False`).

- **Structures identical** — same return arity, same `transform` keys, same tensor shapes and dtypes, same box count on every single frame.
- **`crops` tensor**: max difference **0.5/255** per pixel, i.e. under one 8-bit level.
- **Box coordinates**: max difference **0.03 px**.
- **PySceneDetect cut points**: **identical**.

Batched inference reorders float operations, so the results are near-identical rather than bit-identical. The only visible consequence I found: on 1 frame out of 243, the crop rectangle *drawn onto the debug preview* moved by one pixel, because a 0.008 px coordinate shift crossed an integer rounding boundary (814 pixels out of 380 million). That is the overlay being re-rasterised, not a changed crop.

The same comparison was run for `H3FaceSelect`, checking the `face_pick` contract and the returned images — identical there too.

## 2. Warn when InsightFace falls back to CPU instead of CUDA

`_face_recogniser()` passed `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` and never checked what came back. That list is a preference, not a guarantee — onnxruntime falls through to the next entry without raising. The README already documents the usual cause: `onnxruntime` and `onnxruntime-gpu` both installed, where the CPU-only package wins. `identity_track` then still works and still gives the right answer, it just runs many times slower with nothing saying why.

After `prepare()`, the providers onnxruntime actually bound are read back and any model that landed on CPU is named. Both models are checked, not only detection — recognition runs per candidate face and is what dominates the cost of identity tracking.

The message also reports `onnxruntime.get_available_providers()`, which is what separates the two failure modes for the user: if `CUDAExecutionProvider` is absent from that list, the GPU build is not installed or is being shadowed, rather than present but declined.

Diagnostic only, entirely inside a guard. I verified it stays silent on a correctly configured install, fires with the right text when a session is on CPU, and does not raise when the insightface internals are missing or unexpected.

## Compatibility

No node signature, input, output, or type changes. `_to_bgr_u8()` is kept as-is for its remaining single-frame callers, including `picker_api.py`. Both workflows in `example_workflows/` load and run unchanged.

Tested on Windows 11, Python 3.12, torch 2.14 / CUDA 13, ultralytics 8.3.40, onnxruntime 1.27.0, RTX 4070 Ti SUPER.

Happy to split, reshape, or drop either of them.

---

对 `nodes.py` 的两处小改动：一处检测阶段的性能修复，一处针对当前会静默失败的情况加的警告。两处改动都完全向后兼容 —— 没有任何节点的输入、输出或类型发生变化。

*（原本还有第三处改动 —— 针对 auto 画布模式的显存预检查 —— 现已撤回。我做了实测，发现其前提并不成立；具体数据见下方评论，希望能让其他人省掉同一条弯路。）*

## 1. 批处理人脸检测，替换逐帧循环

`H3FaceTrackCrop.run()` 用一个逐帧的 Python 循环做检测：每帧调用一次 `_to_bgr_u8()`（这是一次阻塞式的 GPU→CPU 同步），再调用一次 `YOLO.predict()`。GPU 在两次 kernel launch 之间处于空闲，开销随片段长度线性增长。

这正是 `H3FaceStitch.run()` 里那段注释已经描述过、并且已经修过的同一类问题：

> `# ---- GPU, batched. The previous version was a per-frame Python loop on CPU tensors: measured at ~1 core of 24 busy with the GPU idle, ~8 minutes for 362 frames.`

检测阶段看起来是唯一还没有做同样处理的环节。

两处循环现在共用一个 `_detect_clip()`：

- `_to_bgr_u8_batch()` 在片段本来所在的设备上完成缩放和 RGB→BGR 翻转，因此一次传输搬运一整批，而且搬的是 `uint8` 而不是 `float32` —— 只有四分之一的字节量。
- `YOLO.predict()` 接受列表形式的批量输入，Python 调用和 kernel launch 的开销从每帧一次变成每 16 帧一次。
- 帧转换是**按批进行，而不是一次性转换整个片段**，因此额外的宿主内存峰值维持在 `batch*H*W*3`。一次性转换长片段会分配数 GB 内存；我实测过，并不会更快，所以这里采用了内存有界的版本。
- **切点检测有意没有做批处理。** PySceneDetect 是一个有状态的流式检测器，必须按顺序逐帧接收数据，因此它仍然是顺序消费的 —— 只是改为读取批量转换已经产出的帧，与之前完全一致。

`all_boxes` / `all_confs` 仍然是逐帧、按顺序构建的，因此下游拿到的数据结构没有任何变化。

### 实测数据

243 帧 544×960 的 H3 片段，`face_yolov8m`，RTX 4070 Ti SUPER：

| | 改动前 | 改动后 | |
|---|---|---|---|
| 检测循环，clip 位于 GPU | 11.16 ms/帧 | 4.08 ms/帧 | **2.74×** |
| 检测循环，clip 位于 CPU | 12.79 ms/帧 | 5.89 ms/帧 | **2.17×** |

两种情况都测了，因为不同的加载器会把 IMAGE 张量留在不同的设备上。完整的 `H3FaceTrackCrop.run()` 在同一片段上从约 3.4 秒降到约 1.6 秒，但这个数字包含了未改动的部分，所以上面的逐帧检测数据更诚实一些。

我也扫过 batch 大小：超过 16 帧之后不再有收益，只会多占显存，所以默认取 16。

### 关于改动范围的说明

在做这个改动的过程中，我注意到 `H3FaceSelect.run()` 里存在一段**逐行完全相同**的逐帧循环。这不在我原本打算修复的范围内 —— 是我在处理 `H3FaceTrackCrop` 时自己在代码里发现的 —— 但只修两处相同循环中的一处，似乎比两处都修更糟，何况 `H3FaceSelect` 正是 `H3_Face_Refine_Auto_Select.json` 的入口节点。现在两者共用 `_detect_clip()`，同时也去掉了约 20 行重复代码。如果你更希望不动那个节点，我很乐意把这部分拆出去。

### 结果等价性验证

我不想让"变快了"这件事仅凭信任成立，所以把原始的 `nodes.py` 和改动后的版本作为两个独立模块并排加载，用同一段素材，走**完整节点**（而不只是检测辅助函数），跑了六组参数组合（默认值、`cut_detection=content`、`select_index=1`、`auto_capped_768`、`size_mode=max_of_clip`、`identity_track=False`）。

- **结构完全一致** —— 返回值个数相同，`transform` 的键相同，张量的 shape 和 dtype 相同，每一帧的框数量都相同。
- **`crops` 张量**：每像素最大差异 **0.5/255**，即不到一个 8-bit 色阶。
- **人脸框坐标**：最大差异 **0.03 px**。
- **PySceneDetect 切点**：**完全一致**。

批量推理会重排浮点运算顺序，所以结果是接近一致而非逐位一致。我找到的唯一可见后果是：243 帧中有 1 帧，*绘制在调试预览图上的*裁切矩形移动了 1 像素，原因是 0.008 px 的坐标偏移跨过了一次整数取整边界（3.8 亿个像素中的 814 个）。这是叠加层被重新栅格化，而不是裁切本身发生了变化。

同样的对比也在 `H3FaceSelect` 上跑过，检查了 `face_pick` 契约和返回的图像 —— 同样完全一致。

## 2. InsightFace 退化到 CPU 时给出警告

`_face_recogniser()` 传入了 `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`，但从未检查实际拿到的是哪个。这个列表表达的是偏好而非保证 —— onnxruntime 会静默地退到下一项，不会报错。README 已经记录了常见原因：`onnxruntime` 和 `onnxruntime-gpu` 同时安装，此时仅 CPU 的那个包会胜出。`identity_track` 仍然可以工作、结果也仍然正确，只是慢很多倍，而且没有任何地方说明原因。

现在在 `prepare()` 之后会读回 onnxruntime 实际绑定的 provider，并指名任何落在 CPU 上的模型。两个模型都会检查，而不只是 detection —— recognition 是按候选人脸逐个运行的，才是身份追踪耗时的大头。

消息中还会报告 `onnxruntime.get_available_providers()`，这能帮用户区分两种失败模式：如果 `CUDAExecutionProvider` 根本不在这个列表里，说明 GPU 版本没有安装或被遮蔽，而不是装了却在本次会话中被拒绝。

纯诊断性质，完全包在保护里。我验证过：在配置正确的环境下保持沉默；会话确实在 CPU 上时以正确的文本触发；insightface 内部结构缺失或不符合预期时不会抛出异常。

## 兼容性

没有任何节点的签名、输入、输出或类型发生变化。`_to_bgr_u8()` 原样保留，供其余的单帧调用方使用，包括 `picker_api.py`。`example_workflows/` 中的两个工作流都能原样加载并运行。

测试环境：Windows 11、Python 3.12、torch 2.14 / CUDA 13、ultralytics 8.3.40、onnxruntime 1.27.0、RTX 4070 Ti SUPER。

这两处改动中的任何一处，我都很乐意拆分、调整或撤回。

🤖 Generated with [Claude Code](https://claude.com/claude-code)


> **Comment by caoddgao (2026-09-09T13:01:29Z):**

**Withdrawing change #2 (the VRAM pre-check) — it does not work**

I measured it and the premise is wrong, so I have dropped that commit from this PR. This PR is now the two remaining changes: the batched detection and the InsightFace provider warning.

Two runs, same 243-frame clip, same everything except the canvas, on a 16 GB RTX 4070 Ti SUPER (model stack: 19.5 GB UNET + 14.6 GB text encoder + VAEs, roughly 38 GB of weights):

| canvas | crops batch | peak VRAM |
|---|---|---|
| 768×768 (`auto_capped_768`) | 1.602 GiB | 13542 MB (82.7%) |
| 512×512 (`manual`) | 0.712 GiB | 13530 MB (82.6%) |

The canvas area dropped to 44% and the crop batch shrank by 911 MB, but peak VRAM moved by 12 MB — 1.3% of the change. My estimate predicted a 3645 MB drop, so it is wrong by a factor of about 300.

The reason is that the weights do not fit in the card to begin with, so peak VRAM is set by how much of the model ComfyUI chooses to keep resident, not by the workload. The activation difference is simply absorbed by swapping more or less of the model in and out.

That leaves the check structurally inverted: the failure it was meant to predict — the silent fallback to streaming weights from system RAM — only happens when the model does not fit, and that is exactly the regime where peak VRAM stops tracking canvas size. On a card large enough to hold the model, `get_free_memory()` is large and the check would never fire anyway. It is useless at both ends.

A warning that cannot measure what it claims to measure seemed worse than no warning, since people would tune settings on the strength of it. Better to drop it than to ship it with a caveat.

Leaving the numbers here in case they are useful — if you ever want a canvas-size guard, peak VRAM is not the signal to build it on, at least not on hardware where the model is already being streamed.

The other two changes are unaffected, and the measurements for #1 still stand.

---

**撤回第 2 处改动（显存预检查）—— 它不成立**

我做了实测，前提是错的，所以已经把那个 commit 从本 PR 中移除。现在这个 PR 只包含剩下的两处改动：批处理检测，以及 InsightFace provider 警告。

两次运行，同一段 243 帧素材，除画布尺寸外其余完全相同，16 GB RTX 4070 Ti SUPER（权重栈：19.5 GB UNET + 14.6 GB 文本编码器 + VAE，约 38 GB）：

| 画布 | crops 批次 | 显存峰值 |
|---|---|---|
| 768×768（`auto_capped_768`） | 1.602 GiB | 13542 MB（82.7%） |
| 512×512（`manual`） | 0.712 GiB | 13530 MB（82.6%） |

画布面积降到 44%，crops 批次缩小 911 MB，而显存峰值只变化了 12 MB —— 相当于该变化量的 1.3%。我的估算预测应下降 3645 MB，偏差约 300 倍。

原因是权重本身就装不进这张卡，因此显存峰值取决于 ComfyUI 选择让多少模型常驻，而非取决于工作负载。激活值的差异被模型的换入换出吸收掉了。

这使得该检查在结构上是倒置的：它试图预警的失败 —— 静默回退到从系统内存流式加载权重 —— 只在模型装不下时发生，而那恰恰是显存峰值不再跟随画布尺寸变化的区间。反过来，在足以容纳整个模型的显卡上，`get_free_memory()` 的返回值很大，这个检查根本不会触发。两头都用不上。

一个无法测量它声称要测量之物的警告，似乎比没有警告更糟，因为人们会据此调整设置。与其带着免责声明发布，不如撤掉。

把这些数字留在这里以备参考 —— 如果你以后想做画布尺寸的保护，显存峰值不是可以依赖的信号，至少在模型已经处于流式加载状态的硬件上不是。

另外两处改动不受影响，第 1 处的性能测量结果依然成立。



> **Comment by caoddgao (2026-09-09T14:06:50Z):**

**更正上一条评论的数据 —— 以及一次做对的测试**

我上面贴的那两次运行，并不是我声称的那个对比。

`auto_capped_768` 里的 768 是**上限，不是目标值**：这段素材最大的裁切框是 458 px，向上对齐到 480，再被节点 512 的下限抬上去，所以**两次运行其实都是 512×512**。我只改了宽高的 widget，没有把 `canvas_mode` 切成 `manual`，而在 auto 模式下这两个数字是被忽略的 —— 画布从头到尾没变过。抱歉，结论碰巧是对的，但那组证据支撑不了它。

下面是重做的测试，三个画布尺寸都在日志里得到确认（`magnification into 512x512 / 768x768 / 1024x1024`），同一段 243 帧素材，16 GB RTX 4070 Ti SUPER：

| 画布 | 面积 | crops 批次 | 显存峰值 | 采样耗时 |
|---|---|---|---|---|
| 512×512 | 1.00× | 0.712 GiB | ~13550 MB | 80.9 秒 |
| 768×768 | 2.25× | 1.602 GiB | ~13550 MB | 173.7 秒 |
| 1024×1024 | 4.00× | 2.848 GiB | ~13550 MB | 339.3 秒 |

**画布面积涨到 4 倍，耗时涨到 4.2 倍（几乎与面积严格成正比），额外显存占用为零。**

原因在 ComfyUI 自己的日志里就写着，模型栈是按 dynamic VRAM loading 分段暂存的：

```
Model MiniMaxH3         prepared for dynamic VRAM loading. 19995MB Staged.
Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 14956MB Staged.
Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading.  2677MB Staged.
Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading.   576MB Staged.
                                                     合计  38204MB  /  显卡 16376MB
```

38 GB 的权重跑在 16 GB 的卡上，分配器会把显存填满尽可能多的暂存权重；激活值那点差异（0.7 → 2.8 GiB）直接被"少常驻一点权重"吸收掉了。**显存峰值测的是显卡容量，不是工作负载。**

所以撤回的决定依然成立，但理由比我最初写的更准确：画布开太大**确实**是个真问题 —— 这里 1024 比 512 慢了 4.2 倍 —— 但代价落在**时间上，不在显存上**。一个盯着可用显存的检查看不到它。如果以后真要做画布尺寸的保护，该依据的信号是计算量，不是显存。

顺带一个可能对文档有用的观察：在这段素材上 512 不只是最快，画质也最好。裁切框最大 458 px，放进 512 的画布时最大的那些几乎是 1:1（放大 1.12 倍），而 1024 会把它们硬拉 2.23 倍、最小的拉到 11.95 倍 —— 源素材里没有那么多细节，多出来的只能靠编造。`auto_capped_768` 在这段素材上自动选中了 512，选得是对的。

本 PR 的另外两处改动不受影响，第 1 处的性能数据依然成立。

---

*Note in English: the table in my previous comment is **wrong** and is superseded by the one above. Both runs quoted there were 512×512, not 768 vs 512 — `auto_capped_768` caps rather than targets, and I had not switched `canvas_mode` to `manual`, so the canvas never varied. The corrected three-canvas test confirms the same conclusion for a different reason: canvas area costs time (4× area → 4.2× time), not VRAM (flat at ~13550 MB), because the 38 GB model stack is streamed into a 16 GB card and the allocator absorbs the difference. The withdrawal of the VRAM pre-check stands.*



> **Comment by Carasibana (2026-09-10T12:06:16Z):**

Thanks for this, the testing seems thorough.

Before I merge, could you extend _detect_clip to cover picker_api.py as well?

There's a third copy of that loop at picker_api.py:186, backing the Pick faces dialog. It's the case where detection is the entire cost, since nothing else runs: no VAE, no sampler, no H3. The speedup lands directly on time the user spends waiting, rather than inside a job where sampling dominates.

It isn't a straight drop-in. That loop reports progress to the dialog roughly 50 times across the clip through _progress(...), and _detect_clip has no progress hook, only the per-batch interrupt check. It would need an optional callback fired once per batch. At _DETECT_BATCH = 16, a 243 frame clip would give about 15 updates instead of 50. I'll leave the shape of that to you since it's your helper.


> **Comment by caoddgao (2026-09-16T13:16:26Z):**

Done — pushed as `96ce145`.

`_detect_clip` gained two optional hooks:

`on_batch(done, total)` fires once per batch, before it runs, which is the same point the old loop reported at. The dialog keeps its `_progress(...)` call, just driven from there.

On the update count: you're right that it drops to ~16 from ~50, but the interval a user actually perceives gets *shorter* in wall-clock terms, not longer. On my 243-frame 544x960 test clip (face_yolov8m, cut detection on, RTX 4070 Ti SUPER), the whole scan went 11.20s -> 7.72s end to end, so the gap between updates went from ~180ms to ~450ms. That felt like the right trade to me — 180ms was finer than a progress bar can use — but if you'd rather keep ~50 updates I can fire the callback per frame inside the batch loop instead; the cost is only the `_progress` calls themselves.

`check_interrupt=False` is the second hook, and it's there for a bug I hit while testing rather than for speed. The scan runs on an HTTP thread, outside the queue, and ComfyUI leaves `interrupt_processing` set after a cancelled render until the next prompt starts. So: cancel a render, queue goes empty, open Pick faces -> the first batch aborts on an interrupt that was never aimed at it. And since `InterruptProcessingException` derives from `BaseException`, `_scan`'s own `except Exception` doesn't catch it, so the dialog just gets a bare 500. The node paths still check as before.

Equivalence check against the loop it replaces, same clip: identical cut points (`[0,76] [76,146] [146,211] [211,243]`), identical face count on every frame (315 boxes total), box coordinates within 0.0082 px — batched-inference float drift, sub-pixel.




### #12 [closed] PR: Fix SyntaxError: f-string expression part cannot include a backslash
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/12  by Gourieff 2026-09-04T04:52:46Z

`nodes.py` fails to import on Python < 3.12 (i.e. most portable/embedded ComfyUI builds, which ship 3.10 or 3.11) with:

    SyntaxError: f-string expression part cannot include a backslash

Cause: inside `H3FaceTrackCrop.run`, the `report` string contains a nested f-string whose expression part includes `\n`:

    f"{f'rendering: {K} of {B} frames  [{drop_note}]\n' if drop_note else ''}"

Backslashes inside the `{...}` part of an f-string (even hidden inside a nested f-string literal) are illegal before PEP 701 (Python 3.12), which made the whole node pack fail to load on any older interpreter.

Fix: hoist the nested f-string out into a plain variable before use.

> **Comment by Carasibana (2026-09-04T10:04:29Z):**

Good catch! thank you 😄



### #11 [closed] ISSUE: [Bug]: Pure Noise at the End of Clip
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/11  by golfingelk-collab 2026-09-01T18:27:39Z

[FaceRefineError.json](https://github.com/user-attachments/files/31708479/FaceRefineError.json)

### Pre-flight

- [x] I read the setup steps in the README and followed them.
- [x] I searched the existing issues, open and closed, and this is not a duplicate.
- [x] This is a defect in the pack — not a question, a feature request, or a request to switch to a different library.
- [ ] `length` on the H3 node equals my clip's exact frame count, and that count sits on H3's 17k+5 grid (5, 22, 39 … 226, 362).
- [x] The crop canvas and the H3 node's width/height match each other.

### What happened, and what did you expect instead?

What I got: perfectly ran face refinement up until the last several frames
What I expected: face refinement ran up to and over the original clips frames to round to valid frame count, then the extra frames trimmed at the end to get back to original clip length.

### Steps to reproduce

1. Loaded the AutoSelect example workflow, swapped the video load and video save with VHS node versions, got all the models linked up
2. Input face reference image, load video clip from H3 output which came from a motion context continuation.. because of this the H3 clip had frames trimmed and is not a standard H3 length
3. Ran the process using default settings on almost everything else

### Your workflow


### Node reports

```shell
[H3FaceRefine] cut detection unavailable, treating the video as one shot: No module named 'scenedetect'
[H3FaceRefine] canvas_mode=auto_capped_768: 768x768 -> 672x672 (largest crop 657px)
[H3FaceRefine] subject: select=largest_face index=0, locked on at frame 11 of 323
[H3FaceRefine] faces: max 1 in one frame
[H3FaceRefine] cuts: cut detection unavailable, treating the video as one shot: No module named 'scenedetect'
[H3FaceRefine] tracking: 305 by continuity, 0 ambiguous (0 resolved by face identity)
[H3FaceRefine] identity: not used (single face, or identity_track off)
[H3FaceRefine] frames=323  face=305 (94%)  body-fallback=0  interpolated=18
[H3FaceRefine] face height  min=89px  mean=131px  max=219px
[H3FaceRefine] face fills   ~33% of every crop (crop_factor=3.0)
[H3FaceRefine] crop box     min=268.2px  max=656.8px
[H3FaceRefine] magnification into 672x672: min=1.02x  mean=1.91x  max=2.51x
[H3FaceRefine] jitter (gaussian) centre 4.01 -> 3.14 px/frame   size 2.00 -> 0.64 px/frame
[H3FaceRefine] box movement 6.18 px/frame (sub-pixel float boxes - no integer rounding)
[H3FaceRefine] dropout runs: 2  longest=11 frames (0.5s at 24fps)  -> composite fades out across these
[H3FaceRefine] !! 323 rendered frame(s) is off H3's 17k+5 grid. H3 rounds up to 328 and the 5 extra frame(s) get padded

[H3FaceRefine] frame count mismatch: base=323 refined=328 transform=323 -> using 323
```

### Console output

```shell
[H3FaceRefine] canvas_mode=auto_capped_768: 768x768 -> 672x672 (largest crop 657px)
[H3FaceRefine] subject: select=largest_face index=0, locked on at frame 11 of 323
[H3FaceRefine] faces: max 1 in one frame
[H3FaceRefine] tracking: 305 by continuity, 0 ambiguous (0 resolved by face identity)
[H3FaceRefine] identity: not used (single face, or identity_track off)
[H3FaceRefine] frames=323  face=305 (94%)  body-fallback=0  interpolated=18
[H3FaceRefine] face height  min=89px  mean=131px  max=219px
[H3FaceRefine] face fills   ~33% of every crop (crop_factor=3.0)
[H3FaceRefine] crop box     min=268.2px  max=656.8px
[H3FaceRefine] magnification into 672x672: min=1.02x  mean=1.91x  max=2.51x
[H3FaceRefine] jitter (gaussian) centre 4.01 -> 3.14 px/frame   size 2.00 -> 0.64 px/frame
[H3FaceRefine] box movement 6.18 px/frame (sub-pixel float boxes - no integer rounding)
[H3FaceRefine] dropout runs: 2  longest=11 frames (0.5s at 24fps)  -> composite fades out across these
[H3FaceRefine] !! 323 rendered frame(s) is off H3's 17k+5 grid. H3 rounds up to 328 and the 5 extra frame(s) get padded with reference content, refined, then discarded on stitch.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[INFO] Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[INFO] Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
[H3FaceRefine] per-frame denoise: face 89-219px, ramp 30-120px (absolute_px)
  ->  MULTIPLIER 0.57 (most worked frame) .. 0.35 (least worked), applied to the denoise set on BasicScheduler
mean 0.46 over 323 frames, 97 latent steps, gamma=1.0
video mask applied by the sampler only; the audio lock reaches the model intact
held frames re-noised to the sigma the sampler is on
[INFO] Requested to load MiniMaxH3
[INFO] 0 models unloaded.
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
100%|████████████████████████████████████████████████████████████████████████████████████| 8/8 [02:45<00:00, 20.71s/it]
[INFO] Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 2677MB Staged. 0 patches attached. Force pre-loaded 128 weights: 692 KB.
[H3FaceRefine] frame count mismatch: base=323 refined=328 transform=323 -> using 323
[INFO] Prompt executed in 305.02 seconds
```

### Environment, models and clip

Pack version: v1.1.0
ComfyUI version: 0.34.0
OS: Windows 11
GPU and VRAM: RTX 5080 - 16GB VRAM
Diffusion model file: minimax_h3_fl2va_pruned_int8_convrot
Text encoder / CLIP file: qwen3vl_32b_minimax_h3_int8_convrot
LoRA(s) and strength: minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16
Detector model (bbox / segm): face_yolov8m
Clip: WxH, frame count, fps: 1120 x 832, 323, 24


### Screenshots or clips (optional)

My understanding reading the readme and the node reports is that non valid H3 frame count clips would still work correctly, just with some rounded up extra frames at the end that would then get trimmed back to original clip length in the stitch. Instead I get the original clip with refined face, but it seems the last several frames of the face crop are just garbled noise instead.


### #10 [closed] PR: Face track selection and index preview
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/pull/10  by fatemark 2026-08-29T12:34:38Z

I had some issues getting anime faces recognized, so I needed more comprehensive selection than just largest/center. It can now select based on x and y, ascending or descending based on an index. Most of this was written with claude, but with a lot of testing.

> **Comment by Carasibana (2026-08-29T20:47:56Z):**

Love it! I'd been thinking about implementing something like this , and you've got a lot of the ground work done! 
I'll pull and merge, but am building a bit more on top of what you've established. To try to sidestep around the issue of having to run the graph once to get the index numbers, I'm making essentially a custom video load node that will do scene / hard cut detection with pyscenedetect, then do the face index approach you laid out on each hard cut , and have a popup ui where you can just click on each face that is the appropriate character across each hard cut. 

<img width="1577" height="1871" alt="image" src="https://github.com/user-attachments/assets/758c7705-163d-46cf-8627-2a080c21ff7c" />

so absolutely building off of this , but not pulling just yet because I want to get the surrounding scaffolding in place!

Thank you! appreciate the work! its a great addition to build on! 


> **Comment by Carasibana (2026-08-31T02:17:03Z):**

@fatemark Merged and shipped in [v1.1.0](https://github.com/Carasibana/ComfyUI-H3-FaceRefine/releases/tag/v1.1.0). Thank you for this.

Your work became the basis for a new node, `H3 Load Video + Face Select`, which detects the clip once up front and lets you choose the subject before the graph runs. It's also what the much more comprehensive selection in this release was built on: nine ranking modes now, per-shot subject choice across cuts, and a pluggable identity backend so it works on anime as well as photographic faces.

Really appreciate the contribution, and I'd welcome your thoughts if you have any once you've had a look.




### #9 [closed] ISSUE: so we are not adding frame length here?
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/9  by HuntingSuccubus 2026-08-23T13:32:02Z

<img width="1765" height="1288" alt="Image" src="https://github.com/user-attachments/assets/a081db3e-21d9-401d-8c0f-5fd6a1e8c489" />

> **Comment by Carasibana (2026-08-25T17:06:36Z):**

Valid — the crop node already knows the frame count, so it could emit it and the example workflow could drop the extra size node. Tagged as an enhancement; no release is scheduled for it.


> **Comment by Carasibana (2026-08-31T02:05:04Z):**

Added in v1.1.0.

`H3 Face Track + Crop` now has a `frame_count` output, and `H3 Load Video + Face Select` has one too. Wire either into the H3 node's `length` and it follows the clip, so the extra size node is no longer needed. Both shipped example workflows are wired that way.

https://github.com/Carasibana/ComfyUI-H3-FaceRefine/releases/tag/v1.1.0



### #8 [closed] ISSUE: Use tae vae encode from bleh node, its 10-20x faster than vae encode and has same quality, denoise get ride of quality issue anyway
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/8  by HuntingSuccubus 2026-08-23T05:31:47Z



> **Comment by Carasibana (2026-08-23T12:28:03Z):**

You can use whatever nodes you want in your own workflow. 



### #7 [closed] ISSUE: please use TaeVaeEncode/decode for H3 from bleh nodes, it 10x faster
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/7  by HuntingSuccubus 2026-08-18T10:06:48Z

https://github.com/blepping/ComfyUI-bleh


### #6 [closed] ISSUE: can i use it to face swap in ref2vid model? blur cropped face to much and denoise again with 4steps
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/6  by HuntingSuccubus 2026-08-18T10:05:19Z




### #5 [closed] ISSUE: What does the Source clip node do?
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/5  by hyojoo207 2026-08-16T22:56:21Z

I don't understand what the 'Source clip' node does. Why is it asking for <absolute path to your clip>.mp4? I haven't generated anything yet, so where would this media file come from? Is this workflow meant for post-processing/refinement?

> **Comment by Carasibana (2026-08-17T00:53:58Z):**

You run this on a clip you created as a second pass.  you generate a video, then take the MP4 that you want to refine the faces on , and load it into this. 



### #4 [closed] ISSUE: Small face not modified
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/4  by tiberiu1995 2026-08-15T16:18:06Z

Hi,
I tried your workflow with different settings but the face is not modified visibly after the sampler.
I still cannot obtain the face likeness from the original face, even though I resized the crop up to 1024x1024.
Any help?

> **Comment by Carasibana (2026-08-16T13:17:51Z):**

Hey there, it's hard to tell what might be going wrong without seeing your setup.
To get a clearer idea of what's happening, you'd need to share a bit more detail 

    Your workflow file with The specific models, LoRAs, or sampler/denoise settings you're using
    Screenshots of examples of what you're seeing (input face, crop size, and final output etc.)


> **Comment by tiberiu1995 (2026-08-16T13:51:10Z):**

 can I send here my input ref, video and result? 


> **Comment by Carasibana (2026-08-25T17:06:25Z):**

Closed — no response received.

Without the workflow, settings, or example output there's nothing here that can be reproduced. This repo now has Discussions open for setup and usage questions; if it's still happening, a new report through the bug form is the way to have it looked at.



### #3 [closed] ISSUE: feature request : hand refiner
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/3  by Ulf3000 2026-08-15T09:08:06Z

or even blur refiner where every washed part of the image gets restored 

if possible would be nice 

> **Comment by Carasibana (2026-08-16T13:12:38Z):**

Closed as out of scope.

This feature is technically feasible, but falls outside the primary goals of the main project and will not be officially implemented.

To build this independently, a hand-tracking or arbitrary object-tracking system could be integrated to replace the face-tracking module. The rest of the workflow including the existing custom nodes for cropping, crop tracking, latent injection, re-generation, and stitching can be reused as-is to complete the pipeline.



### #2 [closed] ISSUE: please adjust workflow its feeding entire video image to ref2video
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/2  by LokeshNakka 2026-08-15T04:50:14Z

please adjust workflow its feeding entire video image to ref2video if we ref only cropped face with small padding we can increase speed
1. rtx 1080p upscale then refine it will be good and speed the refine

> **Comment by Carasibana (2026-08-16T13:09:05Z):**

Closed as intended behaviour. The workflow processes the full video frame by design to maintain the baseline model reference. Users are free to modify their local copy of the workflow to drop the full video reference or adjust refining steps for performance.



### #1 [closed] ISSUE: Problem with multi shot
URL: https://github.com/Carasibana/ComfyUI-H3-FaceRefine/issues/1  by alexfredo 2026-08-13T02:15:05Z

Hi thanks for your awesome workflow it's works great with simple video but for multi shot the face tends to move and not follow or appear suddenly, how I can fix that ?

> **Comment by Carasibana (2026-08-13T11:37:12Z):**

what do you mean by multishot? a single scene with multiple faces? , a video that has hard cuts between multiple scenes? or something else?


> **Comment by alexfredo (2026-08-13T13:04:57Z):**

Sorry if my english is not perfect I'm french what I mean is hard cuts between multiple scenes. here's a example : 

https://github.com/user-attachments/assets/0d50861f-bd50-4ecd-a18e-6d5d8c6a8da9






> **Comment by Carasibana (2026-08-13T14:56:51Z):**

Ah ok, des coupures franches entre les scènes, ça fait du sens.

Le détecteur en tant que tel fonctionne bien, il retrouve le visage sans problème dans le nouveau plan. Le problème vient du lissage. Le nœud de tracking lisse la position de la boîte de recadrage dans le temps (smooth_window, 21 images par défaut) pour que le cadrage ne tremble pas quand le sujet bouge. Sauf que ce lissage-là ne sait pas qu'une coupure vient de se produire, donc juste après le cut, il continue de faire la moyenne avec les positions de l'ancien plan, ce qui tire la boîte vers l'endroit où était le visage avant, au lieu de se replacer directement sur le nouveau. C'est ce délai-là qui donne l'impression que le visage dérive ou qu'il apparaît en retard.

À essayer en premier, vu que c'est rapide : mets smooth_window et size_smooth_window à 1 sur le nœud H3 Face Track + Crop. Une fenêtre de 1 ou moins est traitée comme un no-op complet, ça saute complètement le lissage plutôt que de juste réduire la fenêtre, donc ça te donne un test propre avec le lissage complètement désactivé. Si ça règle le problème, tu voudras peut-être juste réduire les fenêtres à une valeur plus petite plutôt que de tout couper, pour garder un peu de protection contre le tremblement à l'intérieur de chaque plan.

Si t'as encore des problèmes après ça, la solution plus solide c'est de séparer la vidéo en clips distincts à chaque coupure et de faire rouler FaceRefine sur chacun séparément, comme ça le lissage n'opère jamais qu'à l'intérieur d'un seul plan continu dès le départ. Tu peux faire ça manuellement ou avec un outil comme PySceneDetect, qui est gratuit et qui peut détecter les points de coupure automatiquement et séparer la vidéo pour toi. Il y a même un nœud ComfyUI pour ça (Comfyui-SceneDetect) si tu veux garder tout ça dans le même workflow.

Si tu pars dans cette direction, fais attention au nombre d'images. Minimax H3 accepte seulement des longueurs de clip sur une grille fixe, ça monte par paliers de 17 images à partir d'environ 226 (donc 226, 243, 260, 277 et ainsi de suite), 243 images donne à peu près 10 secondes. Si tu coupes tes clips exactement au changement de scène, la longueur de tes clips va presque jamais tomber pile sur un de ces chiffres-là, ce qui va causer des problèmes.

Donc la solution de contournement dans ce cas :
1. Quand tu coupes chaque clip, laisse un peu de matériel en trop après le point de coupure réel (le surplus)
2. Ajuste-le pour que le nombre total d'images corresponde à un des chiffres valides ci-dessus
3. Quand tu recolles tout ensemble après, coupe le surplus pour que ça s'aligne avec ta coupure d'origine

Dis-moi si ça te débloque

--- 

Ah gotcha, hard cuts between scenes, that makes sense now.

So the detector itself is fine, it does pick the face back up in the new shot. The problem is the smoothing. The tracker node smooths the crop box position over time (smooth_window, 21 frames by default) so the crop doesn't shiver around on a moving subject. That smoothing has no idea a hard cut just happened though, so right after the cut it's still averaging in crop positions from the old shot, which drags the box toward where the face used to be instead of snapping straight to the new one. That lag is what reads as the face drifting or popping in late.

First thing to try, since it's quick: set smooth_window and size_smooth_window to 1 on the H3 Face Track + Crop node. A window of 1 or less is treated as a full no-op, it skips the smoothing pass entirely rather than just shrinking the window, so that gets you a clean test with it fully off. If that clears it up, you may just want to lower the windows to something smaller rather than fully off, so you keep some jitter rejection within each shot.

If you're still seeing issues after that, the more robust fix is to split the video into separate clips at each hard cut and run FaceRefine on each one separately, so the smoothing only ever operates inside one continuous shot to begin with. You can do this manually or with a tool like PySceneDetect, it's free and can auto detect the cut points and split the video for you, there's even a ComfyUI node for it (Comfyui-SceneDetect) if you want to keep it all in one graph.

If you go that route, the thing to watch out for is frame count. Minimax H3 only accepts clip lengths on a fixed grid, they go up in steps of 17 frames starting around 226 (so 226, 243, 260, 277 and so on), 243 frames lands you at roughly 10 seconds. If you cut your clips exactly at the scene change your clip lengths almost never land on one of those numbers, and that will cause problems.

So the workaround there is:
1. When you cut each clip, leave a little extra footage past the actual cut point (the overhang)
2. Pad it out so the total frame count matches one of the valid numbers above
3. When you stitch everything back together after, trim the overhang off so it lines up with your original cut

Let me know if that gets you unstuck


> **Comment by Carasibana (2026-08-31T20:24:50Z):**

pyscenedetect , and separating smoothing per hard cut via cut detection has been implemented in v1.1.0

