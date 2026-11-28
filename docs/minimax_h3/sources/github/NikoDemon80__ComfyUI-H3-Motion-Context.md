# NikoDemon80__ComfyUI-H3-Motion-Context

Source: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context
Fetched: 2026-11-28

Description: Clip chaining for MiniMax H3 in ComfyUI - motion and audio genuinely continue across joins
Stars: 1397  Updated: 2026-10-18T04:25:02Z



---

## FILE: README.md (https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/blob/HEAD/README.md)

# H3 Motion Context

Chain MiniMax H3 clips so motion and sound keep going across the cut.

Generate clip A. Feed its last frames and audio into this node. Generate
clip B. B picks up where A left off: same motion, same speed, same
direction, and the same audio continued rather than a new take that
sounds similar. Repeat as long as you like.

Nothing on disk is edited and nothing in ComfyUI is modified. The nodes
check their arithmetic against the live ComfyUI code before the first
render, and if an update breaks an assumption they refuse to run and say
what moved. A loud failure beats a bad render you don't notice.

Requires ComfyUI 0.34.0 or newer, which is where H3 gained arbitrary
keyframe anchors. Version 0.3.1 runs on that and on everything older;
see [CHANGELOG.md](CHANGELOG.md).

## Why this exists

ComfyUI can anchor a still or a short clip at any frame of an H3
generation, with the Add Guide for MiniMax H3 node. If that is all you
need, use it. You do not need this pack for it.

Chaining is a different problem, and two parts of it are still unsolved
by anything else.

**The picture goes out to pixels and back at every link.** Add Guide takes
images and encodes them. Chain through it and every join costs a decode, a
resize and a re-encode. That round trip is where the color drift and the
softening down a long chain come from. This pack slices the previous
clip's tail straight out of its latent, so the pinned frames are the same
numbers they were, bit for bit.

**The sound restarts instead of continuing.** Add Guide anchors audio
starting at a frame and running forward. To actually continue a
soundtrack, the pinned window has to END at the join and reach backwards
into the sound that already played. That is the difference between the
model continuing your track and the model writing something that sounds
like it, and on anything with a beat you hear it immediately.

Everything else here follows from those two: the Trim node that removes
the pinned head from the delivery, the Save/Load Latent pair that carries
a clip across without touching pixels, and the Seam Probe that measures
whether a join is a real continuation or a convincing imitation.

Audio was the harder half, and it's the more useful half, since H3
generates picture and sound together. See "Why the audio needed work"
below if you care how.

## Install

Drop the folder in `ComfyUI/custom_nodes/` and restart. At startup you'll
just see:

```
h3_motion_context: nodes registered. ComfyUI is not modified; the layout
checks run on the first use of a Motion Context node.
```

Having the pack installed changes nothing about your other H3 workflows,
and nothing runs until you actually chain a clip. The first time you do,
you'll see:

```
h3_motion_context: ComfyUI H3 layout checks passed, anchors and pinned audio will land where intended
```

Anything else and the node refuses to run. The reason is logged.

On ComfyUI 0.33.4 or older the node says so and points you at version
0.3.1, which runs on both. That is read from the layout code itself
rather than a version number, so a nightly or a fork gets the right
answer.

## Wiring

```
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo (or the t2v path)
  -> H3 Motion Context      <- previous clip's latent (picture + sound)
  -> guider / sampler
  ...
  decoded IMAGE + AUDIO
  -> H3 Motion Context Trim         <- wire trim_frames across
  -> Create Video / save
```

Wire `trim_frames` into the Trim node. The pinned frames come back at the
start of the new clip and have to come off before you concatenate, picture
and sound together.

### Carrying the previous clip across

The previous clip reaches the node as a latent, but you can't wire the sampler
straight into `context_latent`. ComfyUI will call it a circular
connection, and it's right: the latent you want is from the previous run,
not this one. Two helper nodes move it across runs the same way your
frames and audio already move across, through a file:

```
this run:   SamplerCustomAdvanced -> H3 Motion Context Save Latent
next run:   H3 Motion Context Load Latent -> context_latent
```

Both have a `clip_index` and the numbers mean what they say. On Load, the
clip you're continuing FROM. On Save, the clip this one IS. First clip:
Load 0, Save 1. Making clip 2 from clip 1: Load 1, Save 2.

Load 0 does not read a file. Motion Context then passes the incoming
conditioning through and reports `trim_frames` 0, so you can leave the
node enabled for the first clip. Do not mute it.

Do not use queue **run on change** to walk the chain. Each index change is
a separate graph edit, so two widgets incrementing will queue two runs and
skip slots. Use **H3 Motion Context Chain** instead of ComfyUI's Run
button.

**The Chain node only sees Load and Save if all three sit in the same
canvas group** (one of those colored boxes you draw around nodes). If
they are not grouped together, every button does nothing. Select Load,
Save, and Chain, then Group Selected (or drop Chain into the existing
Motion Context group).

- **Run/Re-roll** is the Run button for this graph. It queues at the
  current Load/Save indices and does not advance them. First clip: leave
  Load 0 / Save 1 and click it. Hate the result? Click it again; the
  retry overwrites that slot.
- **Approve** advances both indices, then queues the next clip once.
- **Chain** is Approve on a loop. After a finished clip it bumps
  Load/Save and keeps going, so a 0/1 clip you already made becomes 1/2
  rather than a re-roll. The exception is Load 0 / Save 1 with no clip 1
  on disk: it generates that first clip, then auto-approves. **segments**
  is how many clips that loop runs: 5 means five clips then stop, 0 means
  keep going until you click **Stop**.
- **Reset** sets Load 0 / Save 1. It does not queue and does not delete
  files.
- **Clear latents** deletes numbered chain slots
  (`clip_00001.safetensors` and so on) in the Load folder. Files you
  renamed are left alone. Indices stay put.

Files are named the obvious way, `clip_00002.safetensors` is clip 2.

Auto-saved files (Save at 0) get a trailing underscore
(`clip_00002_.safetensors`) because they're numbered by run, not by clip,
and indexed loading skips them on purpose.

You can also point the loader at a specific file when `clip_index` is
greater than 0; the file is loaded and the index is ignored. Index 0
never reads a file. Latent paths have to stay inside ComfyUI's output
folder.

By default the loader's output is only for `context_latent`. Don't wire
it into a decode node. Turn on `stock_format` if you want a standard H3
latent that stock nodes accept instead. Stock Save/Load Latent won't work
here, it can't handle H3's paired video/audio latent.

The latent carries both streams, so with it wired you don't need to load
the previous clip's video at all. The pinned frames are sliced straight
out of it rather than decoded to pixels and encoded again, so they're
exactly what the model made instead of a reconstruction. Nothing shifts
the color or the contrast, so there's no seam to see, and it's faster.

Nothing to set for any of that. Leave `context_frames` unwired. Resolution
has to match between clips, since a latent can't be resized; if it doesn't
the node refuses and names both resolutions rather than quietly dropping
to the lossy path.

#### Portable latents

The latent file can also travel through ComfyUI's HTTP API, for tools
that drive ComfyUI remotely and keep their own copy of every clip.

- **Save** reports its file in the run's history the same way stock Save
  Latent does (`outputs.<node>.latents`: `filename`, `subfolder`,
  `type: "output"`), so a client can fetch it with `GET /view`.
- **Load** has an optional `folder` widget, `output` (the default, the
  chain behaviour above) or `input`. With `input`, `latent_path` names one
  file in ComfyUI's input folder, for example one sent with
  `POST /upload/image`. Index 0 still loads nothing; above 0 that file is
  loaded. Paths that leave the input folder are refused.

Workflows saved before `folder` existed load unchanged and keep reading
from output.

There's an older path for graphs with no latent: the previous clip's
frames into `context_frames`, its decoded audio into `context_audio`, and
the H3 audio VAE into `audio_vae`. It works, but it costs a lossy round
trip per link on both streams and it's where the visible seams came from.
Use the latent unless you have a reason not to.

### Reference mode

Put this node after `MiniMaxH3ReferenceToVideo` and wire it as above. Your
reference blocks (image, video, video with audio, audio) are left alone
entirely: the pinned sound rides as a keyframe, not a reference, so the
two mechanisms no longer share a list.

Nothing to configure. Worth mentioning only because versions before 0.2.0
overwrote that list, so turning chaining on quietly threw your references
away, and because up to 0.3.1 the pinned audio was added to it.

References still push the target timeline along, and the pinned frames
follow it. That is ComfyUI's own arithmetic now rather than this pack's.

### Alongside Add Guide

Anchor whatever you like with Add Guide for MiniMax H3 and put this node
after it. Both survive: the pinned head decides how the clip starts, the
guide decides what happens where you put it. Guides that carry audio keep
their own placement, which is forward from the anchor frame, while the
pinned sound keeps its backwards-reaching window.

Guides anchored inside the pinned head are dropped with a warning, for
the same reason a `first_frame` anchor is.

### Keeping a last-frame target

Wire `last_frame` on the stock conditioning node as usual and put this
node after it. The anchor is kept and pinned alongside the head: the
pinned run decides how the clip starts, your image decides where it
ends. It keeps the coordinates ComfyUI gave it, including the shift
references apply, so it stays put whatever else is in the graph.

A `first_frame` anchor is dropped, with a warning. The pinned head owns
those frames, and a second block pinned at the same instant with
different content would fight it. Older versions silently replaced both
anchors, so if a chain graph carried a last-frame target before this
version, it never reached the model.

## Settings

Two, because everything else had exactly one right answer.

**context_length** - frames of the previous clip's picture to carry over.
5, 22, 39 or 56. Those are the lengths that are a whole number of latent
steps, which is why the others aren't offered. 5 is just barely fluid, 22
is nearly seamless. Longer windows pin more motion but they come off the
front of the delivered clip, so 56 spends 2.3 seconds of every render on
frames you throw away. **Use 22.**

**audio_context_length** - frames of tail audio to pin, independent of the
picture window. It ends at the same instant as the pinned video, so this
only controls how far back the sound reaches. 0 follows context_length.
**Use 18.** Longer windows hold the sound closer to the previous clip, but
it gets a little more muffled and a little quieter at every join; at 24
that is audible within a few clips. Shorter windows keep the level and the
joins clean, but the music wanders further from clip to clip. 18 is the
best balance found so far on music chains. Some treble loss remains at
every length. Any multiple of 3 lands exactly on the model's 40 Hz audio
grid (a frame is 5/3 of an audio step), so the window is pinned at
precisely the width you asked for. Off-grid values are widened to the
nearest whole step.

Everything else is fixed: the pinned run is encoded in one VAE call, it
sits at the head of the clip where the Trim node removes it, and the
pinned audio goes on this clip's own timeline. The alternatives all
existed only to reproduce their own failures, so they're constants at the
top of `nodes.py` now. Change one there if you ever need to see what they
did.

`match_tail` on the Trim node stays a setting because that node has no
idea what the other one did. Leave it on. H3 rounds its audio grid to the
nearest step, so every clip carries about 8ms too much or too little
sound, and that error stacks at every join.

## Reducing drift (experimental)

Long chains drift. Faces polish or age, fine detail builds up, and color
and contrast shift a little at every join. Three experimental nodes work
on that. They reduce drift, they don't remove it, and how much they help
depends on the scene. Their behavior and defaults may change in later
versions.

- **H3 Refine Blend** gives each clip a light second pass so it re-reads
  your references instead of only copying the previous clip. The pinned
  frames are kept exactly as sampled, so the join stays closed, and the
  refined version is blended in after them. Put it in the same canvas
  group as Load, Save and Chain; the Chain node keeps it in step.
- **H3 Join Match** removes the small jump in brightness, contrast and
  fine detail between the previous clip's last frames and this clip's
  first new frames. It works best when the camera and scene hold still.
  On moving shots it mostly leaves the clip alone.
- **H3 Sigmas Tail** reshapes the last few noise levels of the schedule
  so the final step adds less extra sharpness. Optional; results vary
  more from scene to scene than the other two.

Join Match defaults to `tuning: auto`, which reads each clip, tells a
locked-off shot from a moving one, and picks its settings for you.
`manual` uses the widgets as set.

```
BasicScheduler -> H3 Sigmas Tail (optional) -> main sampler, sigmas

main sampler ─┬────────────────────────────────> Refine Blend (original)
              └─> refine sampler, one step ────> Refine Blend (refined)

Refine Blend -> H3 Join Match -> Save Latent, VAE Decode, VAE Decode Audio
Motion Context trim_frames -> Join Match context_length
Load Latent -> Join Match context_latent
```

The refine sampler takes one step: Split Sigmas on the BasicScheduler
output (before any Sigmas Tail), step 7 for an 8 step schedule, with its
`low_sigmas` into the second sampler. Give it the plain Reference to
Video conditioning, not the Motion Context output, and its own random
seed.

Save Latent and both decodes take Join Match's output, so the video you
keep matches the latent the next clip continues from.

## Writing prompts for a chain

The settings get motion and sound across the join. What happens in the
next clip is on you, and there are a few traps.

**The model renders contradictions as unions.** Clip N ends on a close-up
of A, clip N+1's prompt opens with "a two-shot of B and C," and the model
doesn't pick. You get all three. The pinned frames aren't a suggestion, so
a prompt describing a different arrangement of people reads as an addition
to them, not a replacement.

**The airlock.** Don't ask for the change and the continuation at the same
moment. Open clip N+1 holding clip N's exact closing framing, no dialogue,
about two seconds, then cut to the new setup. Joins done this way measure
tighter than an ordinary frame-to-frame cut. Joins that skip it measure
like two different rooms spliced together.

**Give the hold something to do.** A held framing with nothing happening
renders as a literal freeze, and two seconds of a motionless actor looks
like the video stalled. Write in a breath, a weight shift, an eyeline
change. The camera holds still, the performer doesn't.
`tests/freeze_detect.py` finds these.

**Budget the pinned head in your timecodes.** In `head` mode the clip
comes out `context_length` frames shorter than it was sampled. At 22 that's
0.92 seconds. Your prompt timings land against the sampled version, which
starts 0.92s earlier, so a beat you wrote for 4.0s shows up at 3.08s in the
file.

## Why the audio needed work

The first version ran pinned audio through H3's reference mechanism, which
is where audio conditioning normally goes. Every join had a small tick,
like the audio briefly sped up and went offbeat. Looking at the waveform
showed nothing wrong. Both sides of every join were smooth on their own.

Cross-correlating each clip's opening against the previous clip's ending
(that's `tests/seam_probe.py`) showed what was actually happening: the new
clip's audio resembled the old one. Same instruments, same groove, never
the same recording. A cover band. The model was reading the reference as
"a separate clip that sounds like this," which is what references are for
and exactly wrong for continuation.

The fix is the same one that already worked for video. The rows the model
sees are identical either way. What differs is their time coordinates, and
the coordinates are what say "separate clip" versus "this clip, earlier."
So the pinned audio is placed on the new clip's own timeline, ending
exactly where the pinned video ends. The timing locked to a flat offset
that doesn't grow from join to join, and the tick was gone.

It isn't a perfect continuation yet. Measured in the latent, the model
still regenerates part of the pinned sound rather than carrying all of it
through, which is where the treble loss down a chain comes from. The beat
stays put and the join stops sounding like a splice, but the sound is
partly continued and partly redrawn.

## Limitations

**Quality degrades down a chain.** The big one, and it's mostly audio.
Each clip is generated from the previous clip's output, which came from
the one before it. Losses compound like photocopying a photocopy, and in
audio the top end goes first. Timing and tempo stay locked, but after
several clips the sound gets duller and more muffled. Picture holds up
much better.

Two things stack per link: the model's own smoothing, and a round trip
out to pixels and back. `context_latent` removes the second one for both
streams. On picture that's the difference between a visible join and no
join at all. On sound it helps, but the model's own smoothing is still
there. Long chains are worth listening to critically, and restarts land
best at a natural musical transition. For the picture side, see
"Reducing drift (experimental)" above.

**A small audio offset, now believed fixed.** Chained clips used to come
out about 8ms late. The cause was arithmetic: a frame is 5/3 of an audio
step, so unless the pinned window's length works out to a whole number of
steps it landed between the ones the model was filling. The window is now
snapped to that grid. The offset was constant, well under where lip-sync
errors get noticeable, and it didn't grow down a chain, so this is a
tidy-up rather than a rescue. `tests/seam_probe.py` reports the lag in
milliseconds if you want to check it on your own material.

**H3 emits 32 kHz audio, not 48.** Read the rate off the clip in any script
that remuxes or concatenates. A stream-copy concat can't change rate
partway, so a hardcoded 48000 silently turns the tail of a long episode
into nothing while every duration check still passes.
`tests/level_step.py` prints every clip's rate and flags a mismatch.

**Turbo LoRAs and Spectrum both cost you audio.** A turbo LoRA hits a
result in very few steps, and fine detail is what those last steps were
for. It thickens the sound and softens the picture. Step-skipping
optimizers like ComfyUI-Spectrum-MiniMax-H3 do the same thing to the
audio, less so to the picture, and they also mispredict the pinned rows,
which never change. Run both together and it stacks. If a chain sounds
duller or closer than you expected, try turning these off before blaming
the chaining. **Keep Spectrum off for these graphs.**

**Resolution can't change mid-chain** while using `context_latent`. A
latent can't be resized, so the node refuses. Regenerate the previous clip
at the new resolution, or start a fresh chain there.

**Prompting a chain takes work**, especially with reference mode. Every
reference conditions the whole clip; there's no way to say "this one
starts two seconds in." Timing has to come from the shot structure and
the description. See "Writing prompts for a chain" above.

**Tested narrowly.** Joins have been verified on dense beat-driven
electronic music, where timing errors are most audible, and on spoken word
through the latent path, where nothing hides a seam. One Windows machine,
one resolution, one sampler. The math self-tests every startup. The
perceptual results are one person's renders.

**ComfyUI's H3 support is young.** This pack no longer patches any of it,
but it still depends on how the layout places things, and one of those
dependencies has no upstream test behind it: the pinned audio window is
positioned with a fractional, negative anchor index. That is legal
arithmetic in the layout and no stock node can produce one, so nothing
upstream exercises it. `layout_contract.py` checks it before the first
render and refuses if it ever stops holding.

The failure mode after a ComfyUI update is therefore "the node won't
run," not bad output. There is a worked example: when the layout
constructor changed, 0.3.0 of this pack refused until 0.3.1 caught up.
Annoying, but the alternative was joins landing at the wrong instant with
nothing to tell you.

**License.** The H3 community license reportedly doesn't currently cover
the EU, UK, Korea or the US. Check for yourself before shipping anything
on it.

## Recommended starting point

`context_length 22`, `audio_context_length 18`, `context_latent` wired
through the Save/Load Latent pair, Trim node wired for picture and sound
with `match_tail` on, Spectrum off. Every "it works" in this README means
that config.

## Testing

Six scripts, all runnable without ComfyUI or a GPU.

```
python tests/_mock_harness.py        # the layout checks against a fake stock model
python tests/_node_smoke_test.py     # the node end to end, refs + save/load
python tests/_probe_node_test.py     # the seam probe node, joins with known answers
python tests/seam_probe.py A.flac B_untrimmed.flac    # is the join real continuation?
python tests/level_step.py clip*.flac                 # does the level or room tone jump?
python tests/freeze_detect.py clip*.mp4               # did a held shot render as a still?
```

The first three print their checks and end with a pass line. The other
three measure real output; `level_step` and `freeze_detect` take
`--self-test` to check their own math on made-up data first, and
`seam_probe`'s math is what `_probe_node_test.py` exercises.

The three measurement scripts ask different questions and a join can fail
any one on its own. `seam_probe` is timing: is the audio the same waveform
continued, or a sound-alike. `level_step` is volume: does the loudness, or
the room tone underneath it, jump at the cut. `freeze_detect` is picture:
is anything happening in that shot.

For `seam_probe` and `level_step`, give them the UNTRIMMED audio of the new
clip. Branch the audio decode to a second Save Audio node alongside the
Trim.

### Measuring a join in-graph

The H3 Motion Context Seam Probe node runs the same measurements inside
the graph, where the seam position is known exactly rather than inferred
from file ends (that inference is where the CLI probe's phantom ~8 ms lag
came from). Wire it inline between the audio VAE decode and the Trim
node:

```
clip_a_latent     the same latent wired into context_latent
audio_vae         the same audio VAE
clip_b_untrimmed  this clip's audio off the VAE decode, before the trim
trim_frames       from the Motion Context node, same value the Trim gets
```

The audio output is the input unchanged, so it drops into an existing
chain without changing the render. The report goes to a Preview Text
node: lag, correlation, broadband and floor steps, judged against the
same thresholds the CLI scripts document.

## Upgrading

[CHANGELOG.md](CHANGELOG.md) lists what changed in each release and which
ComfyUI H3 layout it works with. Worth a look before opening an issue or
starting a fork: several fixes people have rebuilt from scratch were
already in an earlier release.

The node's widgets changed. ComfyUI stores widget values by position, so a
workflow saved against an older version will load its numbers into the
wrong slots. Delete the Motion Context node, add it again and rewire it.
Takes a minute and beats rendering with scrambled settings.

**Only install this once.** A fork of this repo, a manual clone next to a
Manager install, or a renamed backup still sitting in `custom_nodes` will
all load. Renaming a folder does not stop ComfyUI loading it.

Older versions of this pack, and other H3 packs, patch ComfyUI's layout
code at runtime. This one does not, so it no longer competes for that
code and will run alongside them. If it finds the layout wrapped it says
who by, then checks whether anchors still land correctly and carries on
if they do. Keeping one H3 chaining pack installed is still the quiet
life.

## Credits

The Ref2VA multi-reference support was worked out by **seitanism** in the
Banodoco MiniMax H3 seamless-extension thread and first implemented in
**ethanfel**'s fork of this repo. The code here is written independently,
but the idea is theirs and they got there first.

The prompting section comes out of a 16-clip, 4:34 multicam sitcom episode
chained with this pack at 736x576 on a 5070 Ti, about two hours of
generation. The airlock, the freeze warning, the 32 kHz trap and the
timecode budget are all that build's findings. Same run measured a
video-only chain at a median seam level step of 0.905, dropping to 0.16
with the audio carried across.

**javawock7618** and **azra1l** reported the layout change that broke
0.3.0 and narrowed it to specific commits, which turned a hunt into a
diff. **ChimeraWerks** found and fixed the audio grid overhang.

If you build something long with this, numbers are more useful than praise.
Open an issue.

## Files

| File | Role |
|---|---|
| `layout_contract.py` | Proves ComfyUI still places anchors and pinned audio where this pack needs them, once, before the first render. Modifies nothing. |
| `nodes.py` | The core nodes: Motion Context, Trim, the latent Save/Load pair, and Chain. |
| `csrf_guard.py` | Makes sure requests to the Chain node's routes come from ComfyUI's own page. |
| `probe_node.py` | The Seam Probe node: measures a join in-graph, with the seam at a known sample instead of inferred from file ends. |
| `refine_blend.py` | H3 Refine Blend (experimental): holds the pinned frames and blends in a refined version of the rest of the clip. |
| `join_match.py` | H3 Join Match (experimental): removes the step in brightness, contrast and fine detail at each join. |
| `sigma_tools.py` | H3 Sigmas Tail (experimental): reshapes the last few noise levels of the schedule. |
| `web/h3_motion_context.js` | The Chain node's buttons, and keeping Load, Save and Refine Blend indices in step. |
| `tests/seam_probe.py` | Is a join's audio a real continuation, a sound-alike, or drifting. |
| `tests/level_step.py` | Level and room-tone continuity at each join. Also catches sample-rate mismatches. |
| `tests/freeze_detect.py` | Stretches where the picture stops moving. |
| `tests/_mock_harness.py`, `tests/_node_smoke_test.py`, `tests/_probe_node_test.py` | Layout, node and probe tests, numpy only. |
| `CHANGELOG.md` | What changed in each release, and which ComfyUI H3 layout it works with. |


---

# ISSUES / PRs of NikoDemon80/ComfyUI-H3-Motion-Context


### #59 [open] ISSUE: The generated video consistently truncates the beginning of the speech — the first few words are cut off.
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/59  by sunjian2015 2026-11-26T02:26:03Z

 Increasing the duration parameter does not resolve it.

> **Comment by NikoDemon80 (2026-11-27T18:32:28Z):**

If you use [00:00.000] time codes in your prompts, do not prompt any dialogue in the first 0.92 seconds of segment 2 and beyond. The first 22 frames get cut off.



### #58 [closed] ISSUE: Frame 0 trick for chained shots, would love your thoughts
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/58  by roycho87 2026-10-29T03:30:32Z

Hey Niko, big fan.

I've been building on top of Motion Context and found something kind of cool I wanted to run by you.

H3's first latent is a single frame, so I tried copying clip 1's frame 0 into the start of every later shot and locking it. It keeps everything looking like clip 1 a lot longer. By clip 10 it cut the colour drift by about 40%. I also put clip 1's last single frame at the end of each shot, left it free for 4 steps and then locked it, and that got it to about 60% less drift.

The nodes are here if you want to try them: https://github.com/roycho87/clown_context_suite (Clown Keystone + Clown Late Lock)
Here's the relevant workflow: https://github.com/roycho87/clown_context_suite/blob/main/example_workflows/mad_clown_inventor_example.json

Do you have any thoughts on it, or ideas for what I should try next? I'd like to keep building on it. Thanks!

> **Comment by NikoDemon80 (2026-10-29T03:33:38Z):**

Hey Roycho,

Yeah, let me dig into that. That's an interesting idea. I'll get back to you.


> **Comment by roycho87 (2026-10-31T02:11:41Z):**

hey, i did a search on this on arxiv and i found this. https://arxiv.org/abs/2605.30349


> **Comment by NikoDemon80 (2026-10-31T18:46:25Z):**

Hey, thanks for sharing this, and for the kind words. I tried it out, dug into why it works, and read the paper you linked. Short version: it works really well for one kind of shot, and that kind of shot is the locked off "background never changes" shot.

Most of the color drift I’ve measured builds up inside each clip as it generates, not at the handover. Your capstone gives each shot a clean target to land on, so it goes right at that part. I think that’s why adding the back slot took you from 40% to 60%.

The catch is that a frame doesn’t only carry color. It carries the whole picture: pose, camera angle, background and lighting. On a locked-off shot that’s exactly what you want. On a shot that moves, like an arc around the subject or a walk where the background is always changing, the model drags the shot back toward clip 1’s frame to land on it.

I also tried getting the color without the content by putting clip 1’s frames outside the clip instead of inside it. It didn’t get around the problem. Placed after the clip, the model still steered toward it. On an arc shot, clip 4 ended almost exactly on clip 1’s opening frame. Placed far before the clip, it barely helped color and made the joins worse. As far as I can tell, any real frame from clip 1 pulls content with it.

The paper lines up with that. Their main point is that a clean first frame is the strongest thing the model can attend to, so it anchors the whole video and motion dies down. Same mechanism behind why Keystone works on your shots and fights moving ones. Their fix needs training though. The model has to be fine-tuned to use their hidden frame, and it’s built for models that generate in small streaming chunks with a running memory. H3 generates the whole clip at once, so there’s nowhere to put it without retraining H3. It’s also aimed at the opposite problem from yours. They want more motion and give up a little subject consistency to get it. For locked-off shots, a fixed anchor like yours is still the better trade.

One thing worth looking at on your side: Keystone writes into frame 0, which is also the first frame Clown Context holds from the previous shot. So, the held window starts on a frame that doesn’t belong to it. Using only the capstone at the back might keep most of the gain and let you cut fewer frames at the head.

Since this approach only works well with one kind of shot, and not globally, I won’t be adding it to Motion Context. It’s a solid approach for locked-off content though, and it fits your pack well. Nice work, and thanks for sending the paper!


> **Comment by roycho87 (2026-10-31T19:50:44Z):**

hey, thanks for taking the time to dig into this, seriously.

i still need to do more side by side testing, but in my experience if the context motion latents are masked, frame 0 only adds detail. the other 21 (or 38) frames behind it carry the motion.

a lot of what you brought up is stuff i've been figuring out as i go, and i've already changed a few things.

for one, you can change which clip the frame 0 comes from at any point in the chain. so if the scene changes, you just switch the source to a newer clip and that one gets swapped in instead, but i don't use it.

the nodes swap the frame in and then back out after sampling, and so far the side by sides look like the only thing that changes is detail. motion and picture stay the same.

so when the scene changes it doesn't hurt anything, it just helps less until you swap to a new frame 0. and even then it's still better than a degraded frame 0.

i should've been clearer that this works with masked latent frames, not guides. you need those masked context frames behind it or it starts steering the picture, exactly like you said.

about keystone taking over frame 0 of the held window, yeah that's true while it's sampling. but the held window is the last shot's tail, same as motion context, and keystone only borrows the first slot. the rest of the tail is still real so the motion carries through. after sampling the real tail frame goes back in, so the saved clip starts on the right frame and you join it like a normal chain. nothing extra to trim.

capstone only at the back is worth testing. when i locked the capstone from the start though, it pulled her pose back toward the source clip, so late lock only brings it in for the last few steps. the front keystone was the bigger win anyway. in my 10 clip test color drift went 6.4 with nothing, 3.9 with keystone front, 2.9 with the late lock capstone added.

also if you get a chance, check out what i did with jalenbrunson's hybrid windows. i put window 1's first latent on the front of every window, and its last single frame latent gets tacked on the back and cut off every pass. each window samples all the way to step 8, so the next window gets built off finished frames instead of a half done guess. then it rewinds to the saved step 6 and does the last 2 steps as normal blended context windows over the whole video.

it came out way more consistent with less degradation. side by side is here, top is normal hybrid windows, bottom is mine:
https://github.com/roycho87/clown_context_suite/tree/main#clowns-for-picard-example

really appreciate it, this was mostly me trying to pick your brain lol


> **Comment by NikoDemon80 (2026-11-28T06:55:10Z):**

Using masked pin now.



### #57 [closed] ISSUE: [Idea/Feedback] A 100% fix for color drift using Chroma Noise injection, seeking help with transition flickering
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/57  by SG0102 2026-10-23T04:25:41Z

import torch
import torch.nn.functional as F
import random

class MiniMaxH3ChromaNoiseTaper:
    """
    MiniMax H3: Inject deterministic blocky chroma noise into a video's tail frames.
    Translated into pure PyTorch tensor operations for ComfyUI.
    """
    
    # Perfectly replicates the original author's 6 retro low-saturation palettes (mapped to 0-1)
    PALETTE = [
        (185 / 255., 115 / 255., 215 / 255.),
        (115 / 255., 195 / 255., 140 / 255.),
        (150 / 255., 148 / 255., 162 / 255.),
        (205 / 255., 150 / 255., 192 / 255.),
        (138 / 255., 182 / 255., 148 / 255.),
        (160 / 255., 120 / 255., 175 / 255.),
    ]
    
    # Fixed mosaic grid size (width 36, height 64)
    NOISE_GRID = (36, 64)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input original video tensor"}),
                "tail_frames": ("INT", {"default": 22, "min": 1, "max": 1024, "tooltip": "Total number of tail frames to inject chroma noise"}),
                "alpha": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Starting opacity of the noise intensity"}),
                "alpha_end": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Opacity at the end of the fade-out protection zone"}),
                "ramp_frames": ("INT", {"default": 3, "min": 1, "max": 1024, "tooltip": "Number of ramp frames to execute a linear fade-out at the very end"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed determining the chroma mosaic arrangement (ensures consistency across runs)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "inject_taper"
    CATEGORY = "MiniMax-Tools"

    def inject_taper(self, images, tail_frames, alpha, alpha_end, ramp_frames, seed):
        if tail_frames < 1:
            raise ValueError("[Chroma Noise] tail_frames must be greater than 0")
        if ramp_frames < 1 or ramp_frames > tail_frames:
            raise ValueError("[Chroma Noise] ramp_frames must be between 1 and tail_frames")
        if not (0.0 <= alpha <= 1.0) or not (0.0 <= alpha_end <= 1.0):
            raise ValueError("[Chroma Noise] alpha and alpha_end must be between 0.0 and 1.0")
        if alpha_end > alpha:
            raise ValueError("[Chroma Noise] alpha_end cannot be greater than alpha (must be a decreasing fade-out)")

        total_frames, h, w, c = images.shape
        start_idx = max(0, total_frames - tail_frames)
        actual_tail_frames = total_frames - start_idx

        if actual_tail_frames == 0:
            return (images,)

        dev = images.device
        rng = random.Random(seed)

        # 1. Strictly follow the inner and outer loop order of the original script to generate random palette indices, ensuring identical PRNG sequence
        choices = []
        for y in range(self.NOISE_GRID[1]):
            for x in range(self.NOISE_GRID[0]):
                choices.append(rng.choice(range(len(self.PALETTE))))
                
        # 2. Convert indices to specific RGB color tensors
        palette_tensor = torch.tensor(self.PALETTE, dtype=images.dtype, device=dev)
        choices_tensor = torch.tensor(choices, device=dev)
        grid_colors = palette_tensor[choices_tensor]  # Shape: [2304, 3]
        
        # 3. Restore to grid and reshape to fit interpolation function requirements [B, C, H, W]
        grid_colors = grid_colors.view(self.NOISE_GRID[1], self.NOISE_GRID[0], 3)
        grid_bchw = grid_colors.unsqueeze(0).permute(0, 3, 1, 2)
        
        # 4. Use nearest-neighbor (NEAREST) interpolation to instantly scale the tiny grid to video resolution, forming mosaic blocks
        noisy_bchw = F.interpolate(grid_bchw, size=(h, w), mode='nearest')
        noisy_bhwc = noisy_bchw.permute(0, 2, 3, 1)  # Swap back to ComfyUI's [B, H, W, C]

        # 5. Clone original images and apply fade-out algorithm (Taper Blend)
        out_images = images.clone()
        for index in range(start_idx, total_frames):
            position = index - start_idx
            from_end = actual_tail_frames - 1 - position
            
            # Calculate the mosaic opacity for the current frame (Taper Math)
            if from_end >= ramp_frames:
                amount = alpha
            else:
                amount = alpha + (alpha_end - alpha) * (ramp_frames - from_end) / ramp_frames
                
            # Execute blend operation: original * (1 - ratio) + mosaic * ratio
            out_images[index] = out_images[index] * (1.0 - amount) + noisy_bhwc[0] * amount

        return (out_images,)


Hi there,

First of all, your plugin is amazing! I tried it out when you first released it and brought up the issue of "color drift" in long video generation, which you kindly replied to at the time.

I paused using the plugin for a while, but I recently discovered a method that 100% solves the color drift problem. As you can see in the custom node code I provided above, the trick is to inject chroma noise into the last 22 frames of the previous video segment before feeding it as the reference for the next segment. This perfectly eliminates color drift, contrast degradation, and saturation loss, no matter how long the generated video is.

However, there is one minor catch: it introduces a slight flickering at the exact seam/transition between the video segments.

I tried to mitigate this by implementing a 5-frame fade-out (tapering the noise alpha to match brightness). This workaround successfully eliminates the flickering in about 70% of my generations. But for videos where the footage naturally has significant brightness changes across the transition frames, the slight flickering still occurs.

I am not an expert in coding, but I am absolutely certain that the core color drift issue is completely resolved by this chroma noise approach. I am sharing my code here with the hope that you might be interested in looking into it. With your deep understanding of the architecture, I believe you could figure out a robust way to eliminate this transition flickering (perhaps through better brightness matching or dynamic blending).

Perfecting this final step is a bit beyond my coding skills, but I have full faith that you can crack it! Thank you again for your incredible work on this project.

> **Comment by SG0102 (2026-10-23T04:33:09Z):**

Just to follow up on my own post and add a bit more context:

I wanted to share the exact snippet I am using to mitigate the flickering. I've stripped away all the other post-processing features from my node and kept only the core brightness-matching and fading logic.

Using this fading method, the visual flickering is successfully fixed for about 70% of my generated videos. The great news is that because of the chroma noise injection, color drift is simply no longer an issue, regardless of how long the video is.

However, about 30% of the time, a slight flicker still sneaks through. I am leaving this isolated fading logic here just in case it sparks an idea for you on how we might achieve a 100% flawless transition.

Here is the simplified transition logic:import torch
import math

# -----------------------------------------------------------------------------
# 1. Basic Math Utilities (Only what's needed for brightness matching)
# -----------------------------------------------------------------------------
def _luma(x):
    """Convert RGB to Luma channel for brightness calculation"""
    return (x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114)

def _luma_stats(frames):
    """Calculate mean, standard deviation, and median of video frame Luma"""
    y = _luma(frames).detach().float().reshape(-1)
    if int(y.numel()) == 0: 
        return 0.5, 0.1, 0.5
    return (float(y.mean().item()), float(y.std(unbiased=False).clamp_min(1e-5).item()), float(y.median().item()))

# -----------------------------------------------------------------------------
# 2. Core Fading Node Definition
# -----------------------------------------------------------------------------
class SeamlessFadeTransition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_A": ("IMAGE",), 
                "video_B": ("IMAGE",), 
                "overlap_frames": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "Overlapping frames to crop from Video B"}),
                "detect_window": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1, "tooltip": "Window to calculate reference brightness"}),
                "fade_frames": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1, "tooltip": "Number of transition fade frames"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_video",)
    FUNCTION = "join_videos"
    CATEGORY = "Video Postprocessing"

    def join_videos(self, video_A, video_B, overlap_frames=0, detect_window=4, fade_frames=5):
        
        # 1. Crop overlapping frames from Video B
        if overlap_frames > 0:
            video_B = video_B[overlap_frames:]

        # 2. Seam S-curve smooth fine-tuning (Cinematic fade transition)
        images = torch.cat((video_A, video_B), dim=0)
        seam = int(video_A.shape[0])
        n = seg_end = int(images.shape[0])

        if 0 < seam < n and seg_end > seam:
            win = max(1, int(detect_window))
            a0, a1 = max(0, seam - win), seam
            b0, b1 = seam, min(seg_end, seam + win)

            if a1 > a0 and b1 > b0:
                out = images
                # Extract local mean as the smooth reference scale
                ref_mean, _, _ = _luma_stats(images[a0:a1])

                if fade_frames > 0:
                    local_count = min(int(fade_frames), seg_end - seam)
                    if local_count > 0:
                        stable_end = min(seg_end, min(seg_end, seam + local_count) + 4)
                        if stable_end > min(seg_end, seam + local_count):
                            stable_mean = _luma_stats(out[min(seg_end, seam + local_count):stable_end])[0]
                        else:
                            stable_mean = ref_mean
                            
                        # S-shaped Raised Cosine smooth transition curve
                        local_mix = []
                        for i in range(local_count):
                            t = (i + 1) / (local_count + 1)
                            s_curve_val = 0.5 * (1.0 - math.cos(t * math.pi))
                            local_mix.append(s_curve_val)
                        
                        for j in range(local_count):
                            idx = seam + j
                            frame = out[idx]
                            rgb = frame[..., :3].float()
                            y = _luma(rgb)
                            
                            # Single-frame fill light tolerance limited to 15% (0.150)
                            offset = max(-0.150, min(0.150, float(ref_mean * (1.0 - local_mix[j]) + stable_mean * local_mix[j] - float(y.mean().item()))))
                            
                            frame_out = frame.clone()
                            frame_out[..., :3] = (rgb + ((y + offset).clamp(0.0, 1.0) - y).unsqueeze(-1)).clamp(0.0, 1.0).to(frame.dtype)
                            out[idx] = frame_out

                images = out

        return (images,)


> **Comment by NikoDemon80 (2026-10-23T21:15:21Z):**

Thanks for all of the info and code. I will dive into it this weekend. If it works as well as you say, I will roll it into an update after some testing!


> **Comment by NikoDemon80 (2026-10-24T00:44:41Z):**

Thanks for sharing this, and for the detailed write-up.

I built a version of your idea that works in latent space, so it could fit the pack's latent passthrough, and tested it on 5-clip chains. I tried your palette mosaic and a second noise pattern, both with your strength and fade settings.

It made drift worse, not better. The model copies the pinned frames almost exactly, so whatever goes into the pin comes back out, noise included. The color from the mosaic carried into every clip and built up at each join, and brightness and fine detail climbed with it. With the pack's other nodes on, they cleaned up some of it, but the chain still came out worse than the same workflow with no noise.

Your version works on decoded frames, which means a decode and re-encode at every join. Skipping that round trip is one of the main ways my pack keeps joins clean, so I won't be adding a frames-based version.

If you haven't tried it yet, v0.7.1 has Join Match, Sigmas Tail and Refine Blend, which are aimed at the same color and contrast drift. The example workflow is set up with them at the recommended settings.

Thanks again for taking the time to dig into this.



### #56 [closed] ISSUE: Ported to h3.c (native Metal engine): --motion-context / --save-context, plus one finding on the MP4 audio window
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/56  by Bambushu 2026-10-21T12:24:07Z


Your design now runs natively on Apple silicon in antirez's h3.c, in my fork at https://github.com/Bambushu/h3.c (branch `integration`, PR to upstream: https://github.com/antirez/h3.c/pull/76). Same mechanism as your pack: the previous take's last 22 frames (7 latent steps) pinned as condition rows on the new take's own first step times, the last 0.917 s of sound pinned as an audio window that ends at the join, the head trimmed from the delivery. Latent path through a context file the engine writes itself, plus a video path for existing MP4s.

Numbers on an M5 Pro at 544x960, 124 frames, 8 steps, with your continuity metric (frame-to-frame MAD inside the clips, then the join in those units):

- from the context file: join +0.4 sd, r 0.993 between the last and first frame (adjacent frames inside a clip: 0.998)
- from the take's MP4: -0.1 sd, r 0.994
- with an identity image reference on the Ref2VA checkpoint: +3.2 sd, r 0.969, on a shot with a waving arm and a camera push
- a reference video of the previous take instead (the "restage" route): +18.6 sd, r 0.84, a cut

One thing that may be useful for your `context_audio` path: when the previous clip comes from an MP4, its soundtrack is longer than its picture. h3 decodes a third of a latent step past the last frame and AAC pads to whole 1024-sample frames, so on a 124-frame clip the track is 165888 samples where the audio grid ends at 165600. Taking "the last 0.917 s of the soundtrack" started my pinned window 555 samples (0.69 of a latent step) late, and that alone turned the join from a continuation (r 0.99) into a cut (r 0.82); the picture rows were fine (cosine 0.97 to the true latents). Taking the last 37 steps of the clip's own audio grid instead fixed it. Your `match_tail` trim probably protects you when the clip went through the Trim node, but a clip that did not may hit the same thing.

Thanks for working the audio half out; it is the part that makes this usable.



> **Comment by NikoDemon80 (2026-10-24T00:47:23Z):**

Yes, match_tail does exactly that. I also use one of the nodes in my Image Oasis pack that is designed specifically for chopping up mp4 audio so that it can be used with Motion Context, called Audio Oasis, so I don't tend to have this problem at all. Thanks for the write up!



### #55 [open] ISSUE: function discussion
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/55  by xyzDist 2026-10-18T11:59:14Z

Hey NikoDemon, 
I tends to use my shot num int to control the shot because I need a index to do some operation in my setup.
What do you think if we could add a current segment index output on Motion Context Chain node?
will the index would get update and work?

or perhaps the easiest is output current index from "H3 Motion Context Load Latent" node? or a standalone node?

Thanks!

> **Comment by NikoDemon80 (2026-10-18T18:48:00Z):**

Hey xyzDist,
Do you need only the current segment index? I could add “clip_index” as an output socket on the Load Latent node, which would be the clip you are continuing from (0 when it’s a first clip). I could also add a “shot” output socket, which would be “clip_index + 1”, the clip being generated (1 when it’s a first clip). Both would update on every run, including while Chain is looping. This is an easy change.

Load Latent is the better place for it than the Chain node, because the Chain node doesn’t hold the index. When you click Approve or Chain, the browser side writes the new numbers into the Load and Save clip_index widgets, then queues the run. On the Python side the Chain node does nothing.

One thing to avoid: don’t convert Load’s clip_index widget into an input and drive it from your own shot int. The Chain buttons write to that widget, so they would stop working. Take the number out of Load instead.


> **Comment by xyzDist (2026-10-19T01:17:40Z):**

yes, just need the current segment index (shot_index). For example, using as index switch to set per-shot prompts. How you do that at the moment?
this is not very essential, if you have time and it is easy, then it would be very useful. much appreciated!! no rush.


> **Comment by NikoDemon80 (2026-10-20T17:54:09Z):**

I do it differently. Pretty simple solution if all you want is to chain your prompts. https://preview.redd.it/h3-motion-context-0-5-0-no-more-bypassing-the-motion-v0-tz9zsef99enh1.png?width=1073&format=png&auto=webp&s=3936939e98d16491c33151a4567d88a276f26408


> **Comment by xyzDist (2026-10-21T03:57:23Z):**

that's a way. but this int value need to manually set if we go back to some segment you want to regen?
if we have the segment index we could just dynamic linked.
but I understand your goal is to make motion-context as low-level and primitive as possible, that's I totally agree.
so segment index is not a must.


> **Comment by NikoDemon80 (2026-10-21T18:51:12Z):**

Working on this may have to wait until the weekend before I can dive into it. I am swamped at work.


> **Comment by xyzDist (2026-10-22T12:39:44Z):**

That's amazing! take you time, thank you!


> **Comment by NikoDemon80 (2026-10-31T18:53:48Z):**

Functionality added. Will be coming out in my next update. Hopefully later today or tomorrow.


> **Comment by xyzDist (2026-11-08T17:22:51Z):**

Thank you!!



### #54 [closed] ISSUE: Feature: Latent trimming and concatenation (enabling upscaling)
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/54  by 808charlie 2026-10-16T20:08:13Z

This node is functioning very well; thank you for the good work.

I use your nodes, but with a sequential workflow, each extension is 'chained' onto the last, as per screen capture in an old post I made to you: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/41  Exactly the same functionality as your workflow but allows careful honing of each extension.

**Your process**
The latent (and frames) are being passed between each extension for context; each section outputs a latent which is VAE decoded, and from which the repeated 22 context frames are trimmed from the beginning. The final video is made by concatenating the original generation with each trimmed extension video. No visible seams. 

**Upscaling**
For a single generation, I also use a node which upscales the latent. I generate a small latent, then upscale.
https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale
Because the latent is passed through minimax to upscale, the output will not exactly match the input - but will be larger and much clear. 

**Combining the two:**
I want to upscale each 'extension'. These two options 'work'

- upscale first run, and then run each extension at the large resolution - no upscaling. (Super slow).
- Do all extensions, then re-encode the final video frames POST DECODING, by re-encoding with a VAE then passing through the UltimateUpscale (using temporal chunking so fits in 32GB VRAM. (round trip decode, re-encode - and inability to use per-extension lora or denoise adjustment to support upscaling-per-extension).

**Aspiration**
What would be nicer is if the LATENT from each extension could be trimmed, rather than the decoded frames, then those latents can be concatenated. That summed latent could pass through the upscale, without a VAE decode-re-encode loop.

Is that feasible (eg 22 frames corresponds to a boundary), or are latents not something that can 'just be trimmed'?


> **Comment by NikoDemon80 (2026-10-16T22:54:38Z):**

Glad it’s working well for you.

Short answer: yes for the picture, mostly no for the sound.

Picture: 22 frames is exactly 7 latent steps, and the pinned frames always sit at the very start of each extension. So you can drop the first 7 steps of each extension’s video latent and lose exactly the 22 repeated frames. H3 packs frames into steps in a repeating pattern, and a trimmed extension picks up exactly where the previous clip left off in that pattern, so the joined latent is still a valid H3 length. 5, 39 and 56 also land on whole steps (2, 12 and 17). I haven’t decoded a joined latent myself yet, so check the first seam before trusting a long one.

Sound: audio latents run at 40 steps per second against 24 fps, so 22 frames is 36.67 audio steps. Trimming in latent space puts each join about 8 ms out, and it builds up down a chain. If the upscaler only needs the video, join the video latents and keep doing the audio the current way.

The next release adds a stock_format toggle to Load Latent. With it on, you get a standard H3 latent that ComfyUI’s own nodes take (Separate AV Latent, Latent Cut and so on), so you can trim and join saved clip latents without going through pixels.

Does UltimateUpscale need the audio half of the latent, or only the video?


> **Comment by Farstone3D (2026-10-17T00:11:16Z):**

Another option is injecting your full decoded and stitched video images back into a latent and use it on the upscale node, it'll cost a decode/encode round trip, but shouldn't be a big deal since you upscale it right after.


> **Comment by 808charlie (2026-10-17T21:01:33Z):**

@NikoDemon80  The UltimateUpscale and other similar approaches will need both components of the latent - video and audio. Thanks for the info above. I had a feeling there would be something that might make it a challenge. I suspect, for a minimal loss of quality with a single round trip through VAE for the completed frames, that's the easiest route for now.  
I have a habit of spending too long optimizing workflows, not much time generating anything, then a new model comes out. I should probably quit the latent joining idea!

TBH I'd only just got my head around these nested latents, as I couldn't save and load a latent with Comfy's stock nodes.  It really saves some time having a latent stored so one can come back and extend with your nodes, or upscale later. Splitting to video/audio with ltx nodes to save them was nuisance, but found https://github.com/verigen/ComfyUI-Verinuddle the other day which saves some time having a latent stored and manually loadable by path (not a damn drop down list). Useful addition to my workflows using your nodes. 
Thanks again.

@Farstone3D - yes, I agree, that's my current second work-around solution in the bullet points above.


> **Comment by NikoDemon80 (2026-10-17T21:27:18Z):**

@808charlie 

> I have a habit of spending too long optimizing workflows, not much time generating anything, then a new model comes out. I should probably quit the latent joining idea!

Tbh, I don't upscale very often. I'll use RTX VSR when it's not broken (breaks after every major ComfyUI update) but not full upscale models. I'm stranded on an RTX 3070Ti until GPU prices come down. And I pretty much maintain these projects as a hobby in my free time. I honestly had no idea how big Motion-Context was going to become, and there's a lot of pressure to keep up with other projects, but I set a rule for myself right after I released it: Keep it lightweight and simple, and let other people fork it and build on top of it like scaffolding. There are now around 140 forks doing all kinds of cool things. 

When I release v0.7 (my attempt to slowdown degradation) and merge a PR (remote latent access) into v0.7.1 later today, those will probably be the last updates for a while unless something breaks or there's a breakthrough in the degradation problem. I've taken it about as far as it can go without it being overly complicated. If you hand Claude the code base, it can modify it for what you personally need for your process. It will break again when/if I update, but just save it as a project in Claude, and it can patch it as needed.



### #53 [closed] PR: Portable latents: Save reports ui.latents, Load reads from input/
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/pull/53  by Pendair 2026-10-15T20:01:46Z

## Summary

Lets the AV latent written by **H3 Motion Context Save Latent** travel through ComfyUI's HTTP API, so a tool that drives ComfyUI remotely can keep its own copy of each clip's latent and feed it back for the next clip. It changes the two latent nodes only; `MiniMaxH3MotionContext`, `MiniMaxH3MotionContextTrim` and the chain/slot behaviour are untouched.

Two small, backward-compatible changes:

1. **Save Latent reports its file** in the run's history as `ui.latents`, the same shape stock `SaveLatent` uses. A client can then find it in `GET /history/<prompt_id>` and download it with `GET /view`.
2. **Load Latent gains an optional `folder` widget** (`output` | `input`, default `output`). With `input`, `latent_path` names one file in ComfyUI's **input** folder, e.g. one sent with `POST /upload/image`.

### Why

[Calliope](https://github.com/benjiyaya/Calliope) (a story-to-film front end for ComfyUI) uses this pack to continue each clip from the one before it: benjiyaya/Calliope#97. Calliope talks to ComfyUI only over HTTP and can't rely on reading or writing ComfyUI's output folder. ComfyUI may run on another machine, the output folder may be cleared, and a user may re-render an older clip, so Calliope keeps every clip's latent itself. Before this change Save Latent didn't report the file, and Load Latent could only read from `output/`.

How Calliope uses the two changes, per clip:

```
render clip N   →  Save Latent (filename_prefix "calliope/p<project>_c<clip>", clip_index 1)
                →  GET /history/<id>  → outputs.<save node>.latents[0]
                →  GET /view?filename=…&subfolder=calliope&type=output   (kept by Calliope)
render clip N+1 →  POST /upload/image  (file "calliope_p<project>_c<N>.safetensors", overwrite=true)
                →  Load Latent { latent_path: that name, clip_index: 1, folder: "input" }
```

Calliope checks `GET /object_info/MiniMaxH3MotionContextLoadLatent` for the `input` choice and tells the user to update the pack if it is missing.

## Changes

### `MiniMaxH3MotionContextSaveLatent.save`

Returns `{"ui": {"latents": [...]}, "result": (path,)}` instead of `(path,)`:

```json
"outputs": { "404": { "latents": [
  { "filename": "p5_c148_00001.safetensors", "subfolder": "calliope", "type": "output" }
] } }
```

`subfolder` is relative to the output directory, with `/` separators, and `""` at the root. The node's `STRING` output (`path`) is unchanged, so graphs wired to it behave as before.

### `MiniMaxH3MotionContextLoadLatent`

- New `optional` input `folder: ["output", "input"]`, default `"output"`, with a tooltip. It's **optional and last** because API and UI workflows store widget values by position. Workflows saved before this change load unchanged and keep reading from `output/`.
- `folder="input"` → `_resolve_input_latent(path)`. It names **one file** inside `folder_paths.get_input_directory()` and never picks numbered slots from a folder, because the uploader already names the exact file. A missing file or a path that escapes the folder raises `FileNotFoundError` with a clear message.
- Any other `folder` value raises `ValueError` listing the allowed values.
- `clip_index` 0 still loads nothing (first clip / no context), whatever the folder.
- `IS_CHANGED` and `load` take `folder`, so caching follows the file actually resolved (path + `mtime_ns`).

### Path containment, shared by both folders

`_under_output(path)` is now a thin wrapper over a new `_under_root(root, path, default)`. Behaviour:

- Relative paths join onto the root. Absolute paths are accepted only when they already sit under the root. `..` escapes are refused, after `realpath`.
- The comparison now uses `os.path.normcase`, so on Windows `D:\Temp` and `D:\TEMP` count as the same root. Before this, a case or short-name difference between the configured folder and its realpath could refuse a legitimate path.
- `_under_output` keeps its `"h3_context"` default, so Clear and the chain-folder behaviour are unchanged.

### Docs

- README: new "Portable latents" subsection under the latent chain docs.
- CHANGELOG: entry under **Unreleased**. I left the version number to you.

## Compatibility

- Requires ComfyUI 0.34.0 or newer, same as 0.6.2. No new dependencies.
- Existing workflows: no changes needed. `folder` defaults to `output`, and Save Latent's output socket is unchanged.
- The `ui` payload only adds an entry to `/history`, under the same `latents` key stock `SaveLatent` uses.

## Tests

`tests/_node_smoke_test.py`, run on Windows with ComfyUI's venv:

```
save/load roundtrip: file loaded, pinned, end_frame 22.2000, index 0 is first-clip/no-context
indexed slots: re-roll overwrites its slot, loads previous clip's latent; auto mode confirmed to return the reject
reset clear: numbered slots gone, custom names left alone
save ui: ui.latents names the file relative to output/
input folder: uploaded file loads, index 0 loads nothing, escapes and outside paths refused
smoke test passed
```

- New cases: the `ui.latents` payload (filename and subfolder); input-folder loads; index 0 loads nothing; refusal of `..` escapes, absolute paths outside the input folder, directories, and unknown `folder` values.
- Fix in the test itself: the `IS_CHANGED` assertion compared raw paths and failed on Windows when `%TEMP%` differs from its realpath (casing, 8.3 short names, junctions). It now compares resolved, normcased paths.
- Live check on ComfyUI 0.39.0: Load Latent (`folder: input`) → Save Latent round trip. The latent is listed in `/history`, `/view` returns it, and the loaded and saved tensors are identical.
- End to end through Calliope: `Continuation-example.mp4` in benjiyaya/Calliope#97 is a 31 s, 1080p/24 fps film exported from a Calliope project whose clips were chained through these two nodes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)


> **Comment by NikoDemon80 (2026-10-16T00:50:32Z):**

Looking into it. Will decide tomorrow night after some testing.


> **Comment by Pendair (2026-10-16T15:21:44Z):**

Thanks for considering.  I appreciate your work.


> **Comment by NikoDemon80 (2026-10-17T02:21:52Z):**

I'm pushing the v0.7.0 update tomorrow, and then I will merge this into v0.7.1 shortly after.


> **Comment by NikoDemon80 (2026-10-18T04:28:58Z):**

Thanks for this. It's merged and out in 0.7.1, on GitHub and the registry.

Two things you might run into on your side:

- 0.7.0 added a stock_format toggle to Load Latent, so folder now sits after it. If Calliope sends inputs by name you won't notice. If anything builds widget lists by position, folder is the second optional now.
- Motion Context's audio_context_length default is 18 now, not 24. Saved workflows keep whatever they had.

Nice to see Motion Context working inside Calliope. Let me know if anything breaks.



### #52 [closed] ISSUE: Idea: Partial-Denoise Seed Hunt
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/52  by PsypmP 2026-10-08T07:10:54Z

I have an idea for an alternative approach to the Seed Hunt / fast preview workflow.
I would have implemented this myself, but for a long time now, I’ve been using only your nodes.

As I understand it, the current SelfLift Seed Hunt generates candidates at a lower spatial resolution and then lifts the selected latent to the target resolution. This is clever for reducing the cost of candidate generation, but I wonder if there is a simpler alternative that preserves the native H3 generation process.

Instead of generating the candidates at a lower resolution, what if we simply **stop the denoising process early** and decode the intermediate latent as a preview?

For example, if the normal generation uses a sigma schedule like:

```text
σ0 → σ1 → σ2 → ... → σN
```

we could run only part of the trajectory:

```text
σ0 → σ1 → ... → σK
                  ↑
              preview
```

Then decode the intermediate latent with TAEH3. The result would obviously be noisy/blurry and not suitable as a final video, but it should still be sufficient to judge the things that matter for Seed Hunt:

* motion
* camera movement
* composition
* character/object movement
* overall temporal behavior
* whether the seed is worth spending a full generation on

The workflow could then be:

```text
Seed 100 ──┐
Seed 101 ──┤
Seed 102 ──┤
Seed 103 ──┼──→ partial denoise (e.g. 40–60% of sigma schedule)
Seed 104 ──┤
Seed 105 ──┘
                    ↓
              cheap previews
                    ↓
             select good seeds
                    ↓
             full generation
```

The important part is that the preview would use the **same resolution, same model, same latent space and same sigma trajectory** as the final generation. The only difference would be that denoising is stopped early.

This seems potentially useful because it avoids the quality/distribution differences that can be introduced by generating at a lower resolution and subsequently lifting/upscaling the latent. The preview does not need to look good — it only needs to reveal the motion.

### Possible optimization

An even more interesting implementation would be to **save the intermediate latent and continue from it** after selecting a seed.

For example:

```text
noise
  ↓
partial denoise
  ↓
SAVE LATENT ──→ preview
  ↓
select seed
  ↓
continue the same sigma schedule
  ↓
full denoise
  ↓
final video
```

This would avoid recomputing the first part of the denoising process for the selected seeds.

Of course, this would require continuing with the correct remaining sigma schedule rather than simply starting a new sampler, but if H3's sampling pipeline allows the intermediate latent and scheduler state to be preserved, it could make Seed Hunt considerably cheaper.

I think this could work particularly well as an **additional "Partial Denoise / Fast Preview" mode** alongside the existing SelfLift approach rather than replacing it.

The main advantage is that the preview is intentionally low-quality, but it is generated directly from the same full-resolution H3 process. For seed hunting, I think this trade-off could be very effective: **sacrifice visual quality of the preview in exchange for much faster exploration of motion.**


> **Comment by NikoDemon80 (2026-10-08T17:32:30Z):**

I will look into this but cannot promise anything. I always intended this repo to be the jumping off point for others to add features to and improve. I've tried to keep it intentionally simple, but I will look into this and see if it's feasible without making the node pack overly complicated.


> **Comment by NikoDemon80 (2026-10-08T18:45:47Z):**

Thanks for writing this up, it's a well thought out idea.

The good news is that this already works with stock ComfyUI nodes, so you don't need a new mode in the pack to use it:

- Split Sigmas cuts your schedule at step K
- SamplerCustomAdvanced runs the first part. Its `denoised_output` is the model's current guess at the finished clip, which is what you want to decode for the preview. The regular `output` is the half-done latent, which decodes mostly as noise
- A second SamplerCustomAdvanced runs the remaining sigmas from that half-done latent, with Disable Noise so nothing new is added

A few things I'd mention:

- Use a non-random sampler like euler. Ancestral and SDE samplers add fresh noise each step, so the finished clip won't match the preview
- You don't have to save anything to resume. If nothing upstream changes, ComfyUI caches the first sampler, so enabling the second half and re-queuing only runs the remaining steps
- Audio isn't settled yet that early, so the preview is good for judging motion and composition, not the soundtrack

Since it's doable with stock nodes, I'd rather keep it as a documented workflow than add another mode, at least for now. If you try it, I'd really like to hear how many steps you needed before the motion was readable. That would decide whether it's worth building in.

Test workflow (stock nodes only)

Stage 1: preview
- Your usual model, LoRA, refs, prompt, and Reference to Video node
- Your usual scheduler node gives the sigmas
- Split Sigmas: step = 2 to start (try 2, 3, 4)
- Random Noise: the seed you’re testing
- SamplerCustomAdvanced (A): noise from step 4, sigmas = high_sigmas, latent from Reference to Video, sampler = euler
- A’s denoised_output goes to your normal decode path (video and audio) and a preview or Save Video node

Stage 2: finish the winner
- Disable Noise
- SamplerCustomAdvanced (B): noise from step 7, sigmas = low_sigmas, latent = A’s output (not denoised_output), same sampler, guider and model as A
- B’s output goes to your normal decode and Save Video
- Keep stage 2 bypassed while hunting. When a seed looks good, un-bypass it and queue again. Only stage 2 runs

What to check
- Lowest step value where you can still judge motion, camera and composition
- Whether B’s finished clip matches what the preview promised
- Time per preview vs a SelfLift candidate

One thing to watch
- If your current sampler setup lives in a subgraph, you’ll need to pull the two samplers out so both halves are visible



### #51 [closed] ISSUE: is the head/tail mode toggle on the trim node removed?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/51  by xyzDist 2026-10-04T10:55:42Z

hi, as Title, is it recently got removed?

> **Comment by xyzDist (2026-10-04T11:19:18Z):**

sorry, wrong question



### #50 [closed] PR: Added return type of MiniMaxH3MotionContextSaveLatent
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/pull/50  by ymiyad 2026-09-26T17:41:07Z

Modifications for distributed processing with ComfyUI-Distributed (https://github.com/robertvoy/ComfyUI-Distributed)
Due to the design of ComfyUI-Distributed, nodes not connected in series to the DistributedCollector input are not passed to the distributed workers; consequently, the H3MotionContextSaveLatent node would be discarded when the workflow is distributed to the workers. To circumvent this, I added a "pass_through_latent" output to the H3MotionContext SaveLatent node—which simply outputs the input latent as-is—thereby enabling a connection sequence of H3MotionContextSaveLatent -> VAE Decode -> DistributedCollector.

> **Comment by NikoDemon80 (2026-09-27T18:59:50Z):**

Thanks for the PR. I'm not going to merge it.

The change itself is fine. The problem is it tells people Motion Context works with ComfyUI-Distributed, and right now it doesn't.

Every worker saves to the same slot. If local workers share an output folder, they all write the same file and the last one to finish wins. You pick one take, and the next clip can continue from a different one. There's no error when that happens. This pack refuses to run instead of giving a bad render, and this would be a bad render nobody notices.

I also haven't tested the Chain node buttons with Distributed handling the queue.

If it works for your setup, keep it in your fork. You're free to publish it. Proper Distributed support would need each worker saving to its own slot, and that's not on my list right now.

Small note: the title says LoadLatent but the change is on SaveLatent.


> **Comment by ymiyad (2026-09-30T03:27:47Z):**

Hi @NikoDemon80,
Thank you for your comment.
Following up on this PR and your feedback — you were right that shipping this change here would risk exactly the silent wrong-take scenario you described (multiple workers writing the same numbered slot). Instead of pursuing this patch further, I built a separate custom node package that gets the same result without touching H3 Motion Context at all:

[ComfyUI-H3MC-Distributed](https://github.com/ymiyad/ComfyUI-H3MC-Distributed)

The approach: each candidate's latent is embedded directly in its saved video's own container metadata (the same mechanism ComfyUI's core Save Video node uses to embed workflow JSON), so a candidate is one self-contained, playable video file rather than a video plus a separately-tracked latent slot. Picking the best candidate out of a parallel round is just picking a video file — there's no shared slot for multiple workers to race on, and no way to end up continuing from a mismatched latent. It relies only on ComfyUI-Distributed's existing dependency-chain and input-file sync behavior, with no fork of this repo required.

I'm closing this PR since it's no longer needed — leaving this comment in case it's useful to anyone else who finds their way here looking for Distributed support. Thanks again for taking the time to explain the risk clearly; it directly shaped the design.



### #49 [closed] PR: Added "path_through_latent" return type of MiniMaxH3MotionContextSaveLatent
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/pull/49  by ymiyad 2026-09-26T17:09:04Z

Modification to enable compatibility with ComfyUI-Distributed: Added "path_through_latent" to MiniMaxH3MotionContextSaveLatent -- which outputs the input latent as-is -- to enable serial connection to the DistributedCollector node in ComfyUI-Distributed.


### #48 [closed] ISSUE: ComfyUI 0.37.0 Compatibility
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/48  by jehusephat 2026-09-23T19:28:15Z

I haven't had any luck getting the motion context node to work after updating ComfyUI to 0.37.0.  Workflows containing the node run normally and don't produce any error messages, but the reference latent isn't being applied.  I noticed that this was happening immediately after updating ComfyUI.  I tried using the default workflow provided with the node pack to be certain that I hadn't accidentally broken something in my own workflow, but I still ended up having the same issue.  Do you know if the node pack currently has any compatibility issues with 0.37.0?  Thanks!

> **Comment by NikoDemon80 (2026-09-25T16:31:50Z):**

I am on v0.37.0 and have no problems chaining. I would have a dozen active issues for the same thing if something broke in the update. Without a traceback to an error, I don't have much for you.


> **Comment by jehusephat (2026-09-26T02:47:15Z):**

I was pretty certain that it was something on my end when I didn't see anyone else complaining about it after a day or two had passed.  Knowing that I needed to try a little harder helped me get it sorted out after I read your comment, so I appreciate the response.


> **Comment by NikoDemon80 (2026-09-26T03:27:03Z):**

What was the issue?


> **Comment by jehusephat (2026-09-26T05:51:30Z):**

I can't say I ever figured it out, but a clean installation of ComfyUI seemed to resolve it.



### #47 [closed] ISSUE: Still demanding image input when context latent provided
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/47  by velanteg 2026-09-20T17:18:52Z

[ERROR] Failed to validate prompt for output 79:
[ERROR] * MiniMaxH3MotionContext 208:
[ERROR]   - Required input is missing: context_frames
[ERROR] * MiniMaxH3MotionContext 230:
[ERROR]   - Required input is missing: context_frames
[ERROR] Output will be ignored
[ERROR] Failed to validate prompt for output 224:
[ERROR] Output will be ignored

> **Comment by NikoDemon80 (2026-09-20T19:02:44Z):**

For about the 10th time. If you are getting this error, you have an old version of Motion Context installed. Please read other issues before opening a new one.



### #46 [closed] ISSUE: Great workflow, a few UI glitches
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/46  by Bruxengrax 2026-09-19T15:14:28Z

First of all, thank you for making this — I really like the workflow and the whole concept behind H3 Motion Context. The actual workflow is excellent, and the way it handles motion/audio continuation is exactly what I was looking for.

I do have one small issue, though, and I mean this more as friendly feedback than a complaint. The ComfyUI UI feels a bit glitchy when using the workflow. Sometimes, when I zoom in or move around the workflow, some of the menus/widgets disappear or become difficult to access. At times I have to move/zoom the canvas around before they show up again.

The workflow itself is so good that this is why I'm mentioning it — I really want to keep using it, but the disappearing UI elements can make it a little frustrating to work with.

I'm not sure if this is something specific to H3 Motion Context, the custom widgets, or a ComfyUI frontend issue, so I thought I'd mention it in case you can reproduce it.

Again, really appreciate the work on this. The workflow is fantastic; I'm just hoping the UI side can become a little more stable/smooth. ❤️

> **Comment by Bruxengrax (2026-09-20T07:36:20Z):**

I had an idea for dealing with possible quality drift during very long H3 continuation chains.

What if we periodically do a “Latent Rebase”?

For example:

Clip 1 → Clip 2 → ... → Clip 10

At Clip 10, we keep the video as a checkpoint and create a new latent from that video. Then we start the next continuation chain from this new latent, while also using Clip 10 itself as the Reference Video.

So:

Clip 10 video
→ VAE Encode
→ new latent

Then:

new latent + Clip 10 as Reference Video
→ H3 Motion Context
→ Clip 11

After that, Clip 11 becomes the new pair:

Clip 11 latent + Clip 11 video

and the chain continues normally.

The idea is not to use the Reference Video on every generation. It would only be used when doing a rebase, for example every 10–20 clips.

The purpose would be to periodically break the accumulated latent continuation history while keeping the latest successful clip as the visual reference and starting point.

I don't know whether this would actually reduce long-term drift, but conceptually it seems interesting because the new latent and the reference video both come from the same checkpoint clip, while the latent chain itself gets restarted.


> **Comment by NikoDemon80 (2026-09-20T19:05:52Z):**

The disappearing stuff is a problem with your own setup, nothing to do with a custom node pack. As far as using the latent from the first video as a rebase, this would not work. The latent is used as a continuation point, not a reference.



### #45 [closed] ISSUE: context_frames
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/45  by stena 2026-09-17T02:04:49Z

Why am I getting an error "Required input slots have no connection feeding them." input being context_frames. Isn't it supposed to be unconnected?

> **Comment by NikoDemon80 (2026-09-17T04:01:19Z):**

That means you have an old version of Motion Context installed. Probably v0.2.0. That error no longer exists in any version past v0.2.0
Or, you have another person's fork of my repo installed that forked prior to v0.3.1


> **Comment by stena (2026-09-17T16:01:58Z):**

Yup. Updating indeed did the trick. Thank You!



### #44 [closed] ISSUE: Color shift issue
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/44  by lesca 2026-09-16T07:17:26Z

Hello,

I noticed the color shift issue (https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/20 https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/9) and I encounted the same issue. I tried using the Add Guide directly, but still the same result.

I found there is an interesting solution: 
[1] https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/latent_color_carry.py
[2] https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/drift_control.py
However, the project is a little bit complex for me and it doesn't work with my motion transfer project. 

Could you please test the code and integrate with this project?

> **Comment by NikoDemon80 (2026-09-16T19:07:06Z):**

The latent color carry theory is sound and is a genuine solution to the color drift problem. This I will definitely be adding in some form or fashion. The drift control code is fragile and doesn't play nice with other patches and may not be solving the actual problem. Looking into it now.


> **Comment by NikoDemon80 (2026-10-05T19:28:18Z):**

After much testing and development of a new node, I have decided to not release what I built. It is inconsistent and does not work for all cases. No one has solved the issue. All current "fixes" are simply band-aids for specific types of shots. The problem persists.



### #43 [closed] ISSUE: Just report about a conflic.
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/43  by DaveGravel 2026-09-15T14:50:12Z

When someone install this node with your Motion Context nodes installed.
https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef

This MultiRef nodes totally replace your nodes.

> **Comment by NikoDemon80 (2026-09-15T19:58:48Z):**

Since all of these repo's are forks off of my original, and the run-time patches in the older versions are the same, only one repo can own it. Whichever repo loads last is the one that takes over. You cannot run more than one version (or fork) of H3-Motion-Context v0.3.2 or older at the same time. There are no runtime patches as of v0.4, so if you are still getting a collision, it's because the other repo is a fork and kept the same internal node names. Nothing I can do about it.


> **Comment by DaveGravel (2026-09-16T00:33:22Z):**

Yes, I know it’s not your fault. I just wanted to report it so you’re aware of it and so others can know about it as well.

I already had your nodes installed, and yesterday I saw the other ones on a website and decided to install them. I then realized that all of my old workflows had stopped working because all of the nodes had been replaced.

At first, I thought you had released an update. But then I noticed the name in the caption and realized what had happened. I uninstalled the other nodes, and your nodes came back correctly in all of my workflows.

It was mainly just a heads-up in case this happens to someone else.

Anyway, keep up the great work, and thanks for the tool!



### #42 [closed] ISSUE: Feature Request: LTX 2.5 Motion Context
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/42  by mostafablender 2026-09-14T17:14:36Z

Hi ,First of all, thank you for creating H3 Motion Context. It solves a problem that is very difficult for me to solve !

could you consider making a similar Motion Context node for LTX 2.5?

I’d really like to use it to continue an LTX 2.5 video into longer clips while keeping the motion and audio consistent between clips.

Thanks for your work!

> **Comment by NikoDemon80 (2026-09-15T00:58:13Z):**

Already Exists. Check my Image Oasis repo. Use LTX Oasis



### #41 [closed] ISSUE: Workflow Query. Requirement to use the 'Motion Context Chain'?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/41  by 808charlie 2026-09-13T20:04:57Z

Thanks for your work with this node pack.

Do nodes in your pack have the potential to be used WITHOUT the automation 'H3 motion context chain' ?
If so, do you have any recommendations for joining segments created using ONLY your 'H3 motion context' and 'H3 Motion Context Trim' node?

I ask because I tend to prefer as few custom nodes as possible. I much prefer to have am 'extension block' row for each extension, the latent from each going to the next. I then use RGthree muter to gradually turn on extensions. More flexible than chain automation.

I have a workflow that seemed to work which I built to use seitanism's fork of your node pack, which uses 'minimaxH2StartMaskedContext' (a variant of your 'H3 motion context'), 'h3 generated av masked context' for subsequent extension, and 'H3 Motion Context Trim' (again, a variant of yours with same name). It is useful to be able to start with an existing video too.

It works just fine. Here's screenshots showing how I tend to work with that node:
My first 'extension' row...

<img width="2450" height="1080" alt="Image" src="https://github.com/user-attachments/assets/15d1914f-2d7c-4f89-a236-c4cde2a6da35" />

An high level overview showing my 'toggled' workflow blocks.

<img width="1679" height="1218" alt="Image" src="https://github.com/user-attachments/assets/9eab1c77-74e2-42b8-87e3-20227e815761" />


BUT... I notice that author as moved to a different approach in latter workflows, using a custom 'streamLiveExtensionAVToVHS'. I thought I'd take your original pack for a spin...

So...
I tried your node pack, and expanding the subgraph your workflow so i could see flow better. I edited it to extend an existing video (which it did fine) and I can put original and extension vid/aud together with KJ's nodes (and audio resample) as shown... 

<img width="2416" height="1162" alt="Image" src="https://github.com/user-attachments/assets/86899d32-193f-4236-9fdf-2cdbcfcd6ae7" />

I can pretty much swap forked node-pack for your original, and substitute your nodes in the extension workflow in my first image.. then join videos/audio.

But can you tell me. Am I losing 'refinement' in continuity joining this way? Or is use of 'context chain' very much necessary for the direction you're taking the node pack?

Thanks!


> **Comment by NikoDemon80 (2026-09-15T01:04:31Z):**

Can you use my workflow without the chain node? Yes, you can manually cycle the latents. 

I use my own custom node for joining clips. In my Image Oasis repo, there's a node called Video Oasis Viewer. Try it out.



### #40 [closed] ISSUE: Add Screenshots to Main Page, Showcasing Your Nodes
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/40  by drowai 2026-09-12T22:40:06Z

Would be great (for the user, and for you, I think) if you provided screenshots of your nodes in action so people could better understand what they're getting. You're providing a lot, and to only have a text description is doing yourself a disservice.

Maybe even some example videos, tutorials, or just comparisons to show what the extensions can achieve.


### #39 [closed] ISSUE: Extent to 40 Clips - Error
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/39  by Sittlon 2026-09-11T11:28:01Z

Hi, ive extended the workflow to 40 Clips, but cant adjust the Active Clips in the H3 Music Video Controler past 20 Clips, so i adjustet it myself with a new int node. All Clips chained like the rest but as soon as i run it i get this error:

H3 Music Video Controller: expected exactly one music clip 1 group, found 2; expected exactly one music preview 1 group, found 2; expected exactly one music clip 2 group, found 2; expected exactly one music preview 2 group, found 2; expected exactly one music clip 3 group, found 2; expected exactly one music preview 3 group, found 4; expected exactly one music clip 4 group, found 6; expected exactly one music preview 4 group, found 4; expected exactly one music clip 5 group, found 4; expected exactly one music preview 5 group, found 4; expected exactly one music clip 6 group, found 7; expected exactly one music preview 6 group, found 7; expected exactly one music clip 20 group, found 4; expected exactly one music preview 20 group, found 4.

Does it have something to do with the naming of the nodes or groups? 

> **Comment by NikoDemon80 (2026-09-11T19:23:03Z):**

I don't know what the H3 Music Video Controller is. That is not my node. 


> **Comment by Sittlon (2026-09-11T20:12:08Z):**

You are right sorry, it was from another repo. My mistake! 



### #38 [closed] ISSUE: upscale latent
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/38  by hugorejes 2026-09-11T10:34:39Z

The latents are being saved, but I’m not sure how I could add an upscale latent so that after the base generation I run an upscale as a second stage. I tried it in a way where the base gen is saved and then it continues as an upscale, but the second latent then has light glitches across the entire second video. The only other option is to save the upscaled latent, but then the second clip has to start from the beginning as already upscaled, which isn’t ideal. Is there any way to do this without any glitches or errors?

> **Comment by DarkNightMeteor (2026-09-11T10:49:28Z):**

That newly released upscaling model isn't particularly useful right now , it falls far short of the LTX-2.3 secondary upscaling model in terms of detail and is quite slow to generate ,it essentially just corrects distortions in the latent space without actually adding texture ,it requires further training and refinement before it becomes truly usable. I believe the author designed this plugin and workflow to output 1080p at the initial stage achieving optimal resolution right away and then use pixel-based upscaling tools for a second pass to reach 1440p or 4K.


> **Comment by NikoDemon80 (2026-09-11T19:34:54Z):**

The only upscaler I use for MMH3 is RTX VSR. I use Comfy Kitchen backend, Plaguekind's H3 SLA Attention node, and the FastH3 converted turbo LoRA at 8 steps (Or I use the 10eros model and skip the LoRA altogether since it was trained to run in 8 steps). On my 3070Ti, I am able to generate 768x1344 clips at 124 frames in 4.5 to 5 minutes. Without any of this, it takes 18 minutes.


> **Comment by NikoDemon80 (2026-09-12T19:40:57Z):**

Upscaling is outside of the scope of this repo. I have no advice on this, sorry.



### #37 [closed] ISSUE: This plugin works amazingly well , I just had to post something to give it a shout-out.
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/37  by DarkNightMeteor 2026-09-11T02:16:06Z

After comparing those terrible plugins which rely on timelines and cram every function into a single node , I’ve found this to be the best and most practical MH3 video extension plugin I’ve ever used. It’s important to note that this plugin utilizes only a single MH3 reference model and avoids using any additional start or end-frame models, this ensures optimal use of VRAM and system resources , I hope the author will do even better!!!!

> **Comment by NikoDemon80 (2026-09-11T19:29:22Z):**

Thanks for the review! I tried to keep it intentionally simple so that others could build from it. This is the original Motion Context, all other plug-ins that use Motion Context are forks off of this repo or used my code. The thing I'm working on right now is the degradation you see when chaining clips. It's a tough nut to crack.



### #36 [closed] ISSUE: consider (optional) let user naming the latent file and filename input to load latent node instead of hardcode it?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/36  by xyzDist 2026-09-10T13:47:41Z

Hi Dev, 

as title, so that use can control their segments latent for their own setup needs.



Thanks!!

> **Comment by xyzDist (2026-09-10T14:18:12Z):**

I understand if I change one of the middle shot, I need to regenerate all after it.


> **Comment by NikoDemon80 (2026-09-10T22:53:07Z):**

You guys are making the process way too difficult. Just use a proper custom node that can join clips inside of ComfyUI.



### #35 [closed] ISSUE: Subgraph support for 'H3 Motion Context Chain' node
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/35  by ryukbk 2026-09-09T17:03:02Z

Hello, thank you for an excellent node. Now, I wanted to put motion context nodes in a subgraph and have only 'H3 Motion Context Chain' node as the controller outside of the subgraph, but it seems it can't pass the subgraph boundaries. Could you support this setup as well?

> **Comment by NikoDemon80 (2026-09-09T20:45:47Z):**

Unfortunately, no. Motion Context, Load, and Save must stay on the main graph.



### #34 [closed] ISSUE: possible to integrate Add Guide nodes connection?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/34  by DalvoGalbo 2026-09-09T16:19:51Z

i love these nodes and the workflow! i needed to ask if it is possible to add "add guide" nodes in this workflow to have more control over the frames! Thank you!

> **Comment by NikoDemon80 (2026-09-09T20:41:01Z):**

You can add the "AddGuide" node to the workflow without me needing to change anything about Motion Context.


> **Comment by DalvoGalbo (2026-09-11T11:37:25Z):**

yes i could do it myself but after adding that node the colorshift and flicker frames causes seams. but without that node it looks almost seamless. usually the flickering happens during the context overlap portion. i thought there was a proper way to add that node and that is the reason i was asking you about that. 



### #33 [closed] ISSUE: context_frame issue
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/33  by mostafablender 2026-09-08T19:44:23Z

Hi  a few days ago i installed your extension . i watched a yotuube video and download this workflow 

[2026-09-04 MiniMax H3 - ref2va -motion-context-step1(1).json](https://github.com/user-attachments/files/31974202/2026-09-04.MiniMax.H3.-.ref2va.-motion-context-step1.1.json) 

 it worked well i liked it  . a day later comfyui as start decide to update itself  and now this workflow not working   and it give me error for context_frame is empty   , the actual error from comfy is this : 

2
Errors detectedResolve before running the workflow
1 node — 2 errors
1Invalid input

Some input values are not available for this node.

    H3 Motion Context - encode_mode

1Missing connection

Required input slots have no connection feeding them.

    H3 Motion Context - context_frames





i dont know what to do  . at first i believed i have to fill it with something  i tried to use some ai's but they could not help me but they could not . please help me to fix it 

> **Comment by NikoDemon80 (2026-09-08T23:34:49Z):**

That means you have an old version of my repo installed, or, you have another version of my repo installed. Check to see if you have a fork of my repo installed and remove it. Only one version of Motion Context can be installed. Update ComfyUI to at least version v0.34.0 and update Motion Context to v0.6.2 using the custom node manager. Remove the github install first.


> **Comment by mostafablender (2026-09-09T10:38:53Z):**

i did delete all of other custom nodes that was related to h3 motion   (i saw a few youtube video and i just download them with git clone)  it seems they were the reason the main node itself was messed up  . is this how the node should be look like right ?

<img width="434" height="350" alt="Image" src="https://github.com/user-attachments/assets/fa82cb2f-fbd5-4fb3-bd88-ccc1d307aa22" />




> **Comment by mostafablender (2026-09-09T10:40:44Z):**

my node was like this and it would not work  

<img width="435" height="302" alt="Image" src="https://github.com/user-attachments/assets/15fe2415-8e7b-4616-84b4-0b332bdfd7dd" /> 


> **Comment by NikoDemon80 (2026-09-09T20:39:22Z):**

That is not my node. That is a knock off version. The one in the first picture is mine, the second picture is not. Glad it now works for you.



### #32 [closed] ISSUE: KeyError: 'latent'
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/32  by 1113800942 2026-09-06T11:13:38Z

先是我自己做的工作流一直报错, 于是我直接使用了你实例文件夹中的工作流来跑, 除了把时长改成10秒基本什么都没改, 然后和我的工作流一样报同样的错.

节点在执行过程中发生错误。

ComfyUI Error Report
Error Details
Node ID: 245:210
Node Type: SamplerCustomAdvanced
Exception Type: KeyError
Exception Message: KeyError: 'latent'
Stack Trace
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)

  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^

System Information
ComfyUI Version: 0.33.0
Arguments: main.py --feature-flag show_signin_button=true --extra-model-paths-config C:\Users\glynd\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1788506096433.yaml --input-directory D:\Comfy-Desktop\ComfyUI-Shared\input --output-directory D:\Comfy-Desktop\ComfyUI-Shared\output
OS: win32
Python Version: 3.13.15 (tags/v3.13.15:4061bc4, Aug 5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)]
Embedded Python: false
PyTorch Version: 2.13.0+cu130
Devices
Name: cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync
Type: cuda
VRAM Total: 34190458880
VRAM Free: 17158038472
Torch VRAM Total: 167772160
Torch VRAM Free: 96658376
Logs
2026-09-06T03:37:22.562935 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:22.562951 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.562968 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:22.562984 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.563000 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.563017 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.563189 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:22.563213 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.572481 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:37:22.662389 - �[32m[INFO]�[0m got prompt
2026-09-06T03:37:22.665675 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:37:22.665783 - �[1m�[31m[ERROR]�[0m * CLIPLoader 247:165:
2026-09-06T03:37:22.665832 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:22.665855 - �[1m�[31m[ERROR]�[0m * UNETLoader 247:156:
2026-09-06T03:37:22.665873 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:22.665891 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:22.665908 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.665925 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:37:22.665941 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.665959 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.665976 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.666004 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:37:22.666022 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.666864 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:37:22.666914 - �[1m�[31m[ERROR]�[0m * CLIPLoader 244:192:
2026-09-06T03:37:22.666936 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:22.666959 - �[1m�[31m[ERROR]�[0m * UNETLoader 244:198:
2026-09-06T03:37:22.666977 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:22.666998 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:22.667018 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.667036 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:22.667054 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.667071 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.667090 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.667309 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:22.667338 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:22.678844 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:37:30.352100 - �[32m[INFO]�[0m got prompt
2026-09-06T03:37:30.355215 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:37:30.355279 - �[1m�[31m[ERROR]�[0m * CLIPLoader 247:165:
2026-09-06T03:37:30.355303 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:30.355324 - �[1m�[31m[ERROR]�[0m * UNETLoader 247:156:
2026-09-06T03:37:30.355341 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:30.355358 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:30.355375 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:30.355393 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:37:30.355415 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:30.355482 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:30.355503 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:30.355532 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:37:30.355550 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:30.356496 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:37:30.356544 - �[1m�[31m[ERROR]�[0m * CLIPLoader 244:192:
2026-09-06T03:37:30.356564 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:30.356584 - �[1m�[31m[ERROR]�[0m * UNETLoader 244:198:
2026-09-06T03:37:30.356601 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:30.356619 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:30.356637 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:30.356654 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:30.356670 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:30.356685 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:30.356701 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:30.356894 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:30.356920 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:30.365778 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:37:31.220216 - �[32m[INFO]�[0m got prompt
2026-09-06T03:37:31.223619 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:37:31.224748 - �[1m�[31m[ERROR]�[0m * CLIPLoader 247:165:
2026-09-06T03:37:31.224811 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:31.224838 - �[1m�[31m[ERROR]�[0m * UNETLoader 247:156:
2026-09-06T03:37:31.224857 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:31.224875 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:31.224892 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:31.224909 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:37:31.224925 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:31.224941 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:31.224958 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:31.224996 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:37:31.225016 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:31.226023 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:37:31.226067 - �[1m�[31m[ERROR]�[0m * CLIPLoader 244:192:
2026-09-06T03:37:31.226088 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:31.226107 - �[1m�[31m[ERROR]�[0m * UNETLoader 244:198:
2026-09-06T03:37:31.226124 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:31.226140 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:31.226156 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:31.226173 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:31.226188 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:31.226204 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:31.226221 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:31.226414 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:31.226449 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:31.235445 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:37:32.306435 - �[32m[INFO]�[0m got prompt
2026-09-06T03:37:32.309643 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:37:32.309715 - �[1m�[31m[ERROR]�[0m * CLIPLoader 247:165:
2026-09-06T03:37:32.309741 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:32.309767 - �[1m�[31m[ERROR]�[0m * UNETLoader 247:156:
2026-09-06T03:37:32.309785 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:32.309857 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:32.309881 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:32.309899 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:37:32.309915 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:32.309932 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:32.309949 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:32.309976 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:37:32.309994 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:32.310844 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:37:32.310895 - �[1m�[31m[ERROR]�[0m * CLIPLoader 244:192:
2026-09-06T03:37:32.310918 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:32.310938 - �[1m�[31m[ERROR]�[0m * UNETLoader 244:198:
2026-09-06T03:37:32.310957 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:32.310976 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:32.310994 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:32.311013 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:32.311030 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:32.311047 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:32.311064 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:32.311252 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:32.311277 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:32.322040 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:37:33.199403 - �[32m[INFO]�[0m got prompt
2026-09-06T03:37:33.202579 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:37:33.202673 - �[1m�[31m[ERROR]�[0m * CLIPLoader 247:165:
2026-09-06T03:37:33.202698 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:33.202720 - �[1m�[31m[ERROR]�[0m * UNETLoader 247:156:
2026-09-06T03:37:33.202737 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:33.202755 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:33.202772 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:33.202789 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:37:33.202805 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:33.202831 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:33.202848 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:33.202875 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:37:33.202893 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:33.203880 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:37:33.203948 - �[1m�[31m[ERROR]�[0m * CLIPLoader 244:192:
2026-09-06T03:37:33.203977 - �[1m�[31m[ERROR]�[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:33.203998 - �[1m�[31m[ERROR]�[0m * UNETLoader 244:198:
2026-09-06T03:37:33.204015 - �[1m�[31m[ERROR]�[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:33.204032 - �[1m�[31m[ERROR]�[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:33.204048 - �[1m�[31m[ERROR]�[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:33.204066 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:37:33.204081 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:33.204099 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:33.204117 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:33.204309 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:37:33.204336 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:37:33.212899 - �[32m[INFO]�[0m �[32mPrompt executed in 0.01 seconds�[0m
2026-09-06T03:38:23.499763 - �[32m[INFO]�[0m got prompt
2026-09-06T03:38:23.531204 - �[1m�[33m[WARNING]�[0m WARNING: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
2026-09-06T03:38:23.602354 - �[32m[INFO]�[0m Found quantization metadata version 1
2026-09-06T03:38:23.602780 - [MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
2026-09-06T03:38:23.602991 - �[32m[INFO]�[0m Using MixedPrecisionOps for text encoder
2026-09-06T03:38:24.015297 - �[32m[INFO]�[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-09-06T03:38:24.079273 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:38:24.288237 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:38:24.354359 - �[32m[INFO]�[0m Requested to load MiniMaxH3TEModel_
2026-09-06T03:38:24.416345 - �[32m[INFO]�[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:38:33.263632 - �[32m[INFO]�[0m Found quantization metadata version 1
2026-09-06T03:38:33.263738 - �[32m[INFO]�[0m Detected mixed precision quantization
2026-09-06T03:38:33.263938 - �[32m[INFO]�[0m Using mixed precision operations
2026-09-06T03:38:33.263978 - �[32m[INFO]�[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:38:33.272048 - �[32m[INFO]�[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:38:33.273684 - �[32m[INFO]�[0m model_type FLOW_AV
2026-09-06T03:38:33.417742 - �[32m[INFO]�[0m Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
2026-09-06T03:38:33.438250 - �[32m[INFO]�[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:38:33.444057 - �[32m[INFO]�[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 1280x704, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:38:33.476337 - �[32m[INFO]�[0m Requested to load MiniMaxH3
2026-09-06T03:38:33.477205 - �[32m[INFO]�[0m 0 models unloaded.
2026-09-06T03:38:33.515096 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:38:34.414806 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:38:34.416714 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:38:34.419276 - �[32m[INFO]�[0m �[32mPrompt executed in 10.90 seconds�[0m
2026-09-06T03:39:28.890776 - �[32m[INFO]�[0m got prompt
2026-09-06T03:39:28.926516 - �[1m�[33m[WARNING]�[0m WARNING: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
2026-09-06T03:39:28.937504 - �[32m[INFO]�[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:39:28.941801 - �[32m[INFO]�[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 1280x704, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:39:29.008242 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:39:29.039111 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:39:29.040907 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:39:29.047670 - �[32m[INFO]�[0m �[32mPrompt executed in 0.14 seconds�[0m
2026-09-06T03:45:51.460423 - �[32m[INFO]�[0m got prompt
2026-09-06T03:45:51.479874 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 256:
2026-09-06T03:45:51.479977 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 259:
2026-09-06T03:45:51.480003 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:45:51.480025 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:45:51.480045 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:45:51.480080 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 179:
2026-09-06T03:45:51.480101 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:45:51.601702 - �[32m[INFO]�[0m �[32mPrompt executed in 0.12 seconds�[0m
2026-09-06T03:46:06.685010 - �[32m[INFO]�[0m got prompt
2026-09-06T03:46:06.822540 - �[32m[INFO]�[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:46:08.379676 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:46:08.524483 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:46:08.652273 - �[32m[INFO]�[0m Found quantization metadata version 1
2026-09-06T03:46:08.652361 - �[32m[INFO]�[0m Detected mixed precision quantization
2026-09-06T03:46:08.652477 - �[32m[INFO]�[0m Using mixed precision operations
2026-09-06T03:46:08.652509 - �[32m[INFO]�[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:46:08.661797 - �[32m[INFO]�[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:46:08.663206 - �[32m[INFO]�[0m model_type FLOW_AV
2026-09-06T03:46:08.820882 - �[32m[INFO]�[0m Requested to load MiniMaxH3
2026-09-06T03:46:08.822100 - �[32m[INFO]�[0m 0 models unloaded.
2026-09-06T03:46:08.862418 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:47:02.530275 - 
100%|██████████| 4/4 [00:47<00:00, 11.81s/it]2026-09-06T03:47:02.530296 - 
2026-09-06T03:47:02.784772 - �[32m[INFO]�[0m h3_motion_context: saved AV latent to D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors (video (1, 24, 72, 30, 54), audio (1, 32, 2, 405))
2026-09-06T03:47:02.868270 - �[32m[INFO]�[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-09-06T03:47:03.510476 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:47:27.223697 - �[32m[INFO]�[0m h3_motion_context: 243 frames / 10.1250s picture, 10.1250s sound, drift 0.00ms
2026-09-06T03:47:32.310677 - �[32m[INFO]�[0m �[32mPrompt executed in 85.61 seconds�[0m
2026-09-06T03:47:33.364176 - �[32m[INFO]�[0m [LoRA-Manager] Calculating hash for checkpoint 'minimax_h3_fl2va_pruned_int8_convrot' from D:/Comfy-Desktop/ComfyUI-Shared/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
2026-09-06T03:47:58.151211 - �[32m[INFO]�[0m got prompt
2026-09-06T03:47:58.209522 - �[32m[INFO]�[0m 0 models unloaded.
2026-09-06T03:47:58.248652 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:47:58.270132 - 
  0%|          | 0/4 [00:00<?, ?it/s,   Model Initializing ...  ]2026-09-06T03:48:05.254078 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:05.254436 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:09.827402 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:09.827801 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:12.206789 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:12.207035 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:12.826732 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:12.826965 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.390596 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.390974 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.614238 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.614496 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.772299 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.772574 - �[32m[INFO]�[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:27.179310 - 
  0%|          | 0/4 [00:28<?, ?it/s,   Model Initializing ...  ]2026-09-06T03:48:27.179451 - 
2026-09-06T03:48:27.346732 - �[32m[INFO]�[0m Processing interrupted
2026-09-06T03:48:27.349518 - �[32m[INFO]�[0m �[32mPrompt executed in 29.17 seconds�[0m
2026-09-06T03:48:54.559738 - �[32m[INFO]�[0m got prompt
2026-09-06T03:48:54.717948 - �[32m[INFO]�[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:49:02.993592 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:49:03.326566 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:49:03.867482 - �[32m[INFO]�[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:49:03.869711 - �[1m�[33m[WARNING]�[0m h3_motion_context: dropped 1 keyframe anchor(s) at frame(s) [0]: the pinned head already decides frames 0..21. A last_frame anchor is kept.
2026-09-06T03:49:03.869809 - �[32m[INFO]�[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 864x480, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:49:03.889811 - �[32m[INFO]�[0m 0 models unloaded.
2026-09-06T03:49:03.926006 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:49:04.665498 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:49:04.667078 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:49:04.669077 - �[32m[INFO]�[0m �[32mPrompt executed in 10.09 seconds�[0m
2026-09-06T03:50:19.011007 - �[32m[INFO]�[0m Global interrupt (no prompt_id specified)
2026-09-06T03:50:21.722192 - �[1m�[31m[ERROR]�[0m [ERROR] An error occurred while retrieving information for the 'GroundingModelLoader' node.
2026-09-06T03:50:21.723136 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\server.py", line 807, in get_object_info
    out[x] = node_info(x)
             ~~~~~~~~~^^^
  File "E:\Comfy-Loop\ComfyUI\server.py", line 753, in node_info
    if issubclass(obj_class, _ComfyNodeInternal):
       ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: issubclass() arg 1 must be a class

2026-09-06T03:50:28.232324 - 🔧 Settings endpoint called2026-09-06T03:50:28.232619 - 
2026-09-06T03:50:28.232721 - 🔧 Received settings: precision=auto, device=auto, vc_engine=chatterbox_23lang, cosyvoice_variant=RL2026-09-06T03:50:28.232746 - 
2026-09-06T03:50:28.232777 - 🎨 Step Audio EditX inline tags: precision=auto, device=auto2026-09-06T03:50:28.232789 - 
2026-09-06T03:50:28.232802 - 🔄 Voice restoration engine: chatterbox_23lang2026-09-06T03:50:28.232813 - 
2026-09-06T03:50:36.729616 - �[32m[INFO]�[0m got prompt
2026-09-06T03:50:36.870866 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:50:37.116695 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:50:37.118211 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:50:37.119978 - �[32m[INFO]�[0m �[32mPrompt executed in 0.37 seconds�[0m
2026-09-06T03:51:58.836827 - �[32m[INFO]�[0m got prompt
2026-09-06T03:51:58.850926 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 218:
2026-09-06T03:51:58.851020 - �[1m�[31m[ERROR]�[0m * GetVideoComponents 242:
2026-09-06T03:51:58.851044 - �[1m�[31m[ERROR]�[0m   - Required input is missing: video
2026-09-06T03:51:58.851065 - �[1m�[31m[ERROR]�[0m * H3SLAAttention 260:
2026-09-06T03:51:58.851083 - �[1m�[31m[ERROR]�[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:51:58.851101 - �[1m�[31m[ERROR]�[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:51:58.851120 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:51:58.851362 - �[1m�[31m[ERROR]�[0m Failed to validate prompt for output 255:
2026-09-06T03:51:58.851391 - �[1m�[31m[ERROR]�[0m Output will be ignored
2026-09-06T03:52:02.086557 - �[32m[INFO]�[0m �[32mPrompt executed in 3.23 seconds�[0m
2026-09-06T03:52:27.953835 - �[32m[INFO]�[0m got prompt
2026-09-06T03:52:27.996029 - �[32m[INFO]�[0m Found quantization metadata version 1
2026-09-06T03:52:27.997022 - [MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
2026-09-06T03:52:27.997257 - �[32m[INFO]�[0m Using MixedPrecisionOps for text encoder
2026-09-06T03:52:28.378622 - �[32m[INFO]�[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-09-06T03:52:28.419582 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:52:28.541306 - �[32m[INFO]�[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:52:28.622958 - �[32m[INFO]�[0m Requested to load MiniMaxH3TEModel_
2026-09-06T03:52:28.697719 - �[32m[INFO]�[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:52:37.606415 - �[32m[INFO]�[0m Found quantization metadata version 1
2026-09-06T03:52:37.606533 - �[32m[INFO]�[0m Detected mixed precision quantization
2026-09-06T03:52:37.606701 - �[32m[INFO]�[0m Using mixed precision operations
2026-09-06T03:52:37.606737 - �[32m[INFO]�[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:52:37.615468 - �[32m[INFO]�[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:52:37.616960 - �[32m[INFO]�[0m model_type FLOW_AV
2026-09-06T03:52:37.774521 - �[32m[INFO]�[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 864x480, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:52:37.792823 - �[32m[INFO]�[0m Requested to load MiniMaxH3
2026-09-06T03:52:37.794322 - �[32m[INFO]�[0m 0 models unloaded.
2026-09-06T03:52:37.845024 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:52:38.826234 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:52:38.827609 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:52:38.829564 - �[32m[INFO]�[0m �[32mPrompt executed in 10.86 seconds�[0m
2026-09-06T03:54:43.591511 - �[32m[INFO]�[0m got prompt
2026-09-06T03:54:43.692075 - �[32m[INFO]�[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:54:43.707600 - �[1m�[31m[ERROR]�[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:54:43.709332 - �[1m�[31m[ERROR]�[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:54:43.711916 - �[32m[INFO]�[0m �[32mPrompt executed in 0.10 seconds�[0m

> **Comment by NikoDemon80 (2026-09-06T16:43:12Z):**

System Information
ComfyUI Version: 0.33.0

You are on the wrong version of ComfyUI. The README clearly states that any version of H3 Motion Context past v0.3.1 requires ComfyUI v0.34.0 or later.


> **Comment by 1113800942 (2026-09-06T23:12:53Z):**

upgrade to newest version still the same

Node threw an error during execution.

SamplerCustomAdvanced


Error log


# ComfyUI Error Report
## Error Details
- **Node ID:** 312:296
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** KeyError
- **Exception Message:** KeyError: 'latent'

## Stack Trace
```
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy_api\latest\_io.py", line 2046, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^

```
## System Information
- **ComfyUI Version:** 0.34.0
- **Arguments:** ComfyUI\main.py --feature-flag show_signin_button=true --feature-flag enable_telemetry=true --enable-manager --enable-manager-legacy-ui --extra-model-paths-config C:\Users\glynd\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1786624088894.yaml --input-directory D:\Comfy-Desktop\ComfyUI-Shared\input --output-directory D:\Comfy-Desktop\ComfyUI-Shared\output
- **OS:** win32
- **Python Version:** 3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.13.0+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 34190458880
  - **VRAM Free:** 32437698560
  - **Torch VRAM Total:** 67108864
  - **Torch VRAM Free:** 33554432


> **Comment by 1113800942 (2026-09-07T00:20:58Z):**

Hi, I found the problem, motion context can't be together with ComfyUI-H3-Multishot, and minimax-h3-hybrid-cond(multishot- runtime error, hybird con- key latent).  hope this will help others if got the same problem, thank you anyway for your great work. 


> **Comment by NikoDemon80 (2026-09-07T00:53:06Z):**

This information was already in the README. Thanks.



### #31 [closed] ISSUE: is there any quick video tutorial ?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/31  by sudeep333 2026-09-04T06:12:24Z

can you please refer to any video tutorial about your node functionality ?

> **Comment by sudeep333 (2026-09-04T07:27:26Z):**

ok i figured it out. My comfy was not updated. After that everything started working fine. Thanx. Now reading thru your notes. Thanx



### #30 [closed] ISSUE: 搞错了
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/30  by dingmengcm 2026-09-04T02:09:16Z




### #29 [closed] ISSUE: keep getting this error after up to 0.51, don't know from new upgrade or comfy upgrade
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/29  by 1113800942 2026-09-03T22:06:50Z

1
Execution failed

Node threw an error during execution.

SamplerCustomAdvanced


Error log


# ComfyUI Error Report
## Error Details
- **Node ID:** 312:296
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** KeyError
- **Exception Message:** KeyError: 'latent'

> **Comment by NikoDemon80 (2026-09-04T00:47:40Z):**

I need the traceback from the console.


> **Comment by NikoDemon80 (2026-09-04T23:00:16Z):**

Unresponsive. Closing.


> **Comment by 1113800942 (2026-09-06T10:58:11Z):**

先是我自己做的工作流一直报错, 于是我直接使用了你实例文件夹中的工作流来跑, 除了把时长改成10秒基本什么都没改, 然后和我的工作流一样报同样的错.


节点在执行过程中发生错误。


# ComfyUI Error Report
## Error Details
- **Node ID:** 245:210
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** KeyError
- **Exception Message:** KeyError: 'latent'

## Stack Trace
```
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)

  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)

  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)

  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)

  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^

```
## System Information
- **ComfyUI Version:** 0.33.0
- **Arguments:** main.py --feature-flag show_signin_button=true --extra-model-paths-config C:\Users\glynd\AppData\Roaming\Comfy Desktop\instance-model-paths\inst-1788506096433.yaml --input-directory D:\Comfy-Desktop\ComfyUI-Shared\input --output-directory D:\Comfy-Desktop\ComfyUI-Shared\output
- **OS:** win32
- **Python Version:** 3.13.15 (tags/v3.13.15:4061bc4, Aug  5 2026, 13:05:39) [MSC v.1944 64 bit (AMD64)]
- **Embedded Python:** false
- **PyTorch Version:** 2.13.0+cu130
## Devices

- **Name:** cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 34190458880
  - **VRAM Free:** 17158038472
  - **Torch VRAM Total:** 167772160
  - **Torch VRAM Free:** 96658376

## Logs
```
2026-09-06T03:37:22.562935 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:22.562951 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.562968 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:22.562984 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.563000 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.563017 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.563189 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:22.563213 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.572481 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:37:22.662389 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:37:22.665675 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:37:22.665783 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 247:165:
2026-09-06T03:37:22.665832 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:22.665855 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 247:156:
2026-09-06T03:37:22.665873 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:22.665891 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:22.665908 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.665925 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:37:22.665941 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.665959 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.665976 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.666004 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:37:22.666022 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.666864 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:37:22.666914 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 244:192:
2026-09-06T03:37:22.666936 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:22.666959 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 244:198:
2026-09-06T03:37:22.666977 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:22.666998 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:22.667018 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:22.667036 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:22.667054 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:22.667071 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:22.667090 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.667309 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:22.667338 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:22.678844 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:37:30.352100 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:37:30.355215 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:37:30.355279 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 247:165:
2026-09-06T03:37:30.355303 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:30.355324 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 247:156:
2026-09-06T03:37:30.355341 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:30.355358 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:30.355375 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:30.355393 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:37:30.355415 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:30.355482 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:30.355503 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:30.355532 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:37:30.355550 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:30.356496 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:37:30.356544 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 244:192:
2026-09-06T03:37:30.356564 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:30.356584 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 244:198:
2026-09-06T03:37:30.356601 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:30.356619 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:30.356637 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:30.356654 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:30.356670 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:30.356685 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:30.356701 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:30.356894 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:30.356920 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:30.365778 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:37:31.220216 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:37:31.223619 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:37:31.224748 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 247:165:
2026-09-06T03:37:31.224811 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:31.224838 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 247:156:
2026-09-06T03:37:31.224857 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:31.224875 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:31.224892 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:31.224909 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:37:31.224925 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:31.224941 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:31.224958 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:31.224996 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:37:31.225016 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:31.226023 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:37:31.226067 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 244:192:
2026-09-06T03:37:31.226088 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:31.226107 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 244:198:
2026-09-06T03:37:31.226124 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:31.226140 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:31.226156 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:31.226173 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:31.226188 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:31.226204 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:31.226221 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:31.226414 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:31.226449 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:31.235445 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:37:32.306435 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:37:32.309643 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:37:32.309715 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 247:165:
2026-09-06T03:37:32.309741 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:32.309767 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 247:156:
2026-09-06T03:37:32.309785 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:32.309857 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:32.309881 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:32.309899 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:37:32.309915 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:32.309932 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:32.309949 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:32.309976 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:37:32.309994 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:32.310844 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:37:32.310895 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 244:192:
2026-09-06T03:37:32.310918 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:32.310938 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 244:198:
2026-09-06T03:37:32.310957 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:32.310976 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:32.310994 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:32.311013 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:32.311030 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:32.311047 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:32.311064 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:32.311252 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:32.311277 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:32.322040 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:37:33.199403 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:37:33.202579 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:37:33.202673 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 247:165:
2026-09-06T03:37:33.202698 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:33.202720 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 247:156:
2026-09-06T03:37:33.202737 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_fl2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:33.202755 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 190:
2026-09-06T03:37:33.202772 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_FL2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:33.202789 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:37:33.202805 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:33.202831 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:33.202848 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:33.202875 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:37:33.202893 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:33.203880 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:37:33.203948 - ^[[1m^[[31m[ERROR]^[[0m * CLIPLoader 244:192:
2026-09-06T03:37:33.203977 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: clip_name: 'Qwen3vl-32b-minimax-h3_int8_convrot.safetensors' not in (list of length 41)
2026-09-06T03:37:33.203998 - ^[[1m^[[31m[ERROR]^[[0m * UNETLoader 244:198:
2026-09-06T03:37:33.204015 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: unet_name: 'MiniMax_H3_ref2va_pruned_int8_convrot.safetensors' not in (list of length 81)
2026-09-06T03:37:33.204032 - ^[[1m^[[31m[ERROR]^[[0m * LoraLoaderModelOnly 238:
2026-09-06T03:37:33.204048 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: lora_name: 'MiniMax H3\\FastH3_REF2VA_6-step.safetensors' not in (list of length 321)
2026-09-06T03:37:33.204066 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:37:33.204081 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:37:33.204099 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:37:33.204117 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:33.204309 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:37:33.204336 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:37:33.212899 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.01 seconds^[[0m
2026-09-06T03:38:23.499763 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:38:23.531204 - ^[[1m^[[33m[WARNING]^[[0m WARNING: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
2026-09-06T03:38:23.602354 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-09-06T03:38:23.602780 - [MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
2026-09-06T03:38:23.602991 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-09-06T03:38:24.015297 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-09-06T03:38:24.079273 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:38:24.288237 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:38:24.354359 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-09-06T03:38:24.416345 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:38:33.263632 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-09-06T03:38:33.263738 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-09-06T03:38:33.263938 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-09-06T03:38:33.263978 - ^[[32m[INFO]^[[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:38:33.272048 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:38:33.273684 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-09-06T03:38:33.417742 - ^[[32m[INFO]^[[0m Applying MiniMax H3 Memory Efficient Sage Attention Patch to all transformer blocks
2026-09-06T03:38:33.438250 - ^[[32m[INFO]^[[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:38:33.444057 - ^[[32m[INFO]^[[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 1280x704, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:38:33.476337 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-09-06T03:38:33.477205 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-09-06T03:38:33.515096 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:38:34.414806 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:38:34.416714 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:38:34.419276 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 10.90 seconds^[[0m
2026-09-06T03:39:28.890776 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:39:28.926516 - ^[[1m^[[33m[WARNING]^[[0m WARNING: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
2026-09-06T03:39:28.937504 - ^[[32m[INFO]^[[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:39:28.941801 - ^[[32m[INFO]^[[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 1280x704, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:39:29.008242 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:39:29.039111 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:39:29.040907 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:39:29.047670 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.14 seconds^[[0m
2026-09-06T03:45:51.460423 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:45:51.479874 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 256:
2026-09-06T03:45:51.479977 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 259:
2026-09-06T03:45:51.480003 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:45:51.480025 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:45:51.480045 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:45:51.480080 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 179:
2026-09-06T03:45:51.480101 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:45:51.601702 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.12 seconds^[[0m
2026-09-06T03:46:06.685010 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:46:06.822540 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:46:08.379676 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:46:08.524483 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:46:08.652273 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-09-06T03:46:08.652361 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-09-06T03:46:08.652477 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-09-06T03:46:08.652509 - ^[[32m[INFO]^[[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:46:08.661797 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:46:08.663206 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-09-06T03:46:08.820882 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-09-06T03:46:08.822100 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-09-06T03:46:08.862418 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:47:02.530275 - 
100%|██████████| 4/4 [00:47<00:00, 11.81s/it]2026-09-06T03:47:02.530296 - 
2026-09-06T03:47:02.784772 - ^[[32m[INFO]^[[0m h3_motion_context: saved AV latent to D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors (video (1, 24, 72, 30, 54), audio (1, 32, 2, 405))
2026-09-06T03:47:02.868270 - ^[[32m[INFO]^[[0m Model MiniMaxH3AudioVAE prepared for dynamic VRAM loading. 576MB Staged. 0 patches attached. Force pre-loaded 401 weights: 539 KB.
2026-09-06T03:47:03.510476 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:47:27.223697 - ^[[32m[INFO]^[[0m h3_motion_context: 243 frames / 10.1250s picture, 10.1250s sound, drift 0.00ms
2026-09-06T03:47:32.310677 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 85.61 seconds^[[0m
2026-09-06T03:47:33.364176 - ^[[32m[INFO]^[[0m [LoRA-Manager] Calculating hash for checkpoint 'minimax_h3_fl2va_pruned_int8_convrot' from D:/Comfy-Desktop/ComfyUI-Shared/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
2026-09-06T03:47:58.151211 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:47:58.209522 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-09-06T03:47:58.248652 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:47:58.270132 - 
  0%|          | 0/4 [00:00<?, ?it/s,   Model Initializing ...  ]2026-09-06T03:48:05.254078 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:05.254436 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:09.827402 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:09.827801 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:12.206789 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:12.207035 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:12.826732 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:12.826965 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.390596 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.390974 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.614238 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.614496 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:18.772299 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:48:18.772574 - ^[[32m[INFO]^[[0m Interrupting prompt a0984eaf-9a75-428a-b616-9bdf917e341c
2026-09-06T03:48:27.179310 - 
  0%|          | 0/4 [00:28<?, ?it/s,   Model Initializing ...  ]2026-09-06T03:48:27.179451 - 
2026-09-06T03:48:27.346732 - ^[[32m[INFO]^[[0m Processing interrupted
2026-09-06T03:48:27.349518 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 29.17 seconds^[[0m
2026-09-06T03:48:54.559738 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:48:54.717948 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:49:02.993592 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:49:03.326566 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:49:03.867482 - ^[[32m[INFO]^[[0m h3_motion_context: loaded AV latent from D:\Comfy-Desktop\ComfyUI-Shared\output\h3_context\clip_00001.safetensors
2026-09-06T03:49:03.869711 - ^[[1m^[[33m[WARNING]^[[0m h3_motion_context: dropped 1 keyframe anchor(s) at frame(s) [0]: the pinned head already decides frames 0..21. A last_frame anchor is kept.
2026-09-06T03:49:03.869809 - ^[[32m[INFO]^[[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 864x480, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:49:03.889811 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-09-06T03:49:03.926006 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:49:04.665498 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:49:04.667078 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:49:04.669077 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 10.09 seconds^[[0m
2026-09-06T03:50:19.011007 - ^[[32m[INFO]^[[0m Global interrupt (no prompt_id specified)
2026-09-06T03:50:21.722192 - ^[[1m^[[31m[ERROR]^[[0m [ERROR] An error occurred while retrieving information for the 'GroundingModelLoader' node.
2026-09-06T03:50:21.723136 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\server.py", line 807, in get_object_info
    out[x] = node_info(x)
             ~~~~~~~~~^^^
  File "E:\Comfy-Loop\ComfyUI\server.py", line 753, in node_info
    if issubclass(obj_class, _ComfyNodeInternal):
       ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: issubclass() arg 1 must be a class

2026-09-06T03:50:28.232324 - 🔧 Settings endpoint called2026-09-06T03:50:28.232619 - 
2026-09-06T03:50:28.232721 - 🔧 Received settings: precision=auto, device=auto, vc_engine=chatterbox_23lang, cosyvoice_variant=RL2026-09-06T03:50:28.232746 - 
2026-09-06T03:50:28.232777 - 🎨 Step Audio EditX inline tags: precision=auto, device=auto2026-09-06T03:50:28.232789 - 
2026-09-06T03:50:28.232802 - 🔄 Voice restoration engine: chatterbox_23lang2026-09-06T03:50:28.232813 - 
2026-09-06T03:50:36.729616 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:50:36.870866 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:50:37.116695 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:50:37.118211 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:50:37.119978 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.37 seconds^[[0m
2026-09-06T03:51:58.836827 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:51:58.850926 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 218:
2026-09-06T03:51:58.851020 - ^[[1m^[[31m[ERROR]^[[0m * GetVideoComponents 242:
2026-09-06T03:51:58.851044 - ^[[1m^[[31m[ERROR]^[[0m   - Required input is missing: video
2026-09-06T03:51:58.851065 - ^[[1m^[[31m[ERROR]^[[0m * H3SLAAttention 260:
2026-09-06T03:51:58.851083 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: reference_protection: False not in ['True', 'Light', 'Off']
2026-09-06T03:51:58.851101 - ^[[1m^[[31m[ERROR]^[[0m   - Value not in list: block_size: '32' not in ['64', '128']
2026-09-06T03:51:58.851120 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:51:58.851362 - ^[[1m^[[31m[ERROR]^[[0m Failed to validate prompt for output 255:
2026-09-06T03:51:58.851391 - ^[[1m^[[31m[ERROR]^[[0m Output will be ignored
2026-09-06T03:52:02.086557 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 3.23 seconds^[[0m
2026-09-06T03:52:27.953835 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:52:27.996029 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-09-06T03:52:27.997022 - [MultiGPU Core Patching] text_encoder_device_patched returning device: cuda:0 (current_text_encoder_device=cuda:0)
2026-09-06T03:52:27.997257 - ^[[32m[INFO]^[[0m Using MixedPrecisionOps for text encoder
2026-09-06T03:52:28.378622 - ^[[32m[INFO]^[[0m CLIP/text encoder model load device: cuda:0, offload device: cpu, current: cpu, dtype: torch.float16
2026-09-06T03:52:28.419582 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:52:28.541306 - ^[[32m[INFO]^[[0m Model MiniMaxH3VideoVAE prepared for dynamic VRAM loading. 4965MB Staged. 0 patches attached. Force pre-loaded 128 weights: 348 KB.
2026-09-06T03:52:28.622958 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3TEModel_
2026-09-06T03:52:28.697719 - ^[[32m[INFO]^[[0m Model MiniMaxH3TEModel_ prepared for dynamic VRAM loading. 25882MB Staged. 0 patches attached. Force pre-loaded 310 weights: 1272 KB.
2026-09-06T03:52:37.606415 - ^[[32m[INFO]^[[0m Found quantization metadata version 1
2026-09-06T03:52:37.606533 - ^[[32m[INFO]^[[0m Detected mixed precision quantization
2026-09-06T03:52:37.606701 - ^[[32m[INFO]^[[0m Using mixed precision operations
2026-09-06T03:52:37.606737 - ^[[32m[INFO]^[[0m Native ops: convrot_w4a4, float8_e4m3fn, nvfp4, mxfp8, int8_tensorwise, float8_e5m2, asym_w4a8_int8 
2026-09-06T03:52:37.615468 - ^[[32m[INFO]^[[0m model weight dtype torch.bfloat16, manual cast: torch.bfloat16
2026-09-06T03:52:37.616960 - ^[[32m[INFO]^[[0m model_type FLOW_AV
2026-09-06T03:52:37.774521 - ^[[32m[INFO]^[[0m h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 243 frame clip at 864x480, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
2026-09-06T03:52:37.792823 - ^[[32m[INFO]^[[0m Requested to load MiniMaxH3
2026-09-06T03:52:37.794322 - ^[[32m[INFO]^[[0m 0 models unloaded.
2026-09-06T03:52:37.845024 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:52:38.826234 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:52:38.827609 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:52:38.829564 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 10.86 seconds^[[0m
2026-09-06T03:54:43.591511 - ^[[32m[INFO]^[[0m got prompt
2026-09-06T03:54:43.692075 - ^[[32m[INFO]^[[0m Model MiniMaxH3 prepared for dynamic VRAM loading. 19995MB Staged. 208 patches attached. Force pre-loaded 210 weights: 1175 KB.
2026-09-06T03:54:43.707600 - ^[[1m^[[31m[ERROR]^[[0m !!! Exception during processing !!! 'latent'
2026-09-06T03:54:43.709332 - ^[[1m^[[31m[ERROR]^[[0m Traceback (most recent call last):
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 549, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 348, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 322, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\Comfy-Loop\ComfyUI\execution.py", line 310, in process_inputs
    result = f(**inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "E:\Comfy-Loop\ComfyUI\comfy_api\latest\_io.py", line 1995, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "E:\Comfy-Loop\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\Comfy-Loop\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in extra_conds_with_hybrid
    visual_latents.extend(item["latent"] for item in keyframes)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\minimax-h3-hybrid-cond\model_base_patch.py", line 30, in <genexpr>
    visual_latents.extend(item["latent"] for item in keyframes)
                          ~~~~^^^^^^^^^^
KeyError: 'latent'

2026-09-06T03:54:43.711916 - ^[[32m[INFO]^[[0m ^[[32mPrompt executed in 0.10 seconds^[[0m

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
Workflow too large. Please manually upload the workflow from local file system.
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)




### #28 [closed] ISSUE: loras from example workflow
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/28  by LukasKolembus 2026-09-03T16:11:59Z

Where i can download loras from example workflow? 

MiniMax H3\FastH3_FL2VA_6-step.safetensors
MiniMax H3\FastH3_REF2VA_6-step.safetensors



> **Comment by Filipus24 (2026-09-03T18:22:34Z):**

@LukasKolembus : I had this question as well, then I realized you have to create it yourself, but it's really easy if you follow @NickDemon80's very thorough instructions right here: https://github.com/NikoDemon80/ComfyUI-FastH3-Lora-Converter


> **Comment by NikoDemon80 (2026-09-04T00:45:01Z):**

Correct, you have to convert the LoRA yourself.



### #27 [closed] ISSUE: Contrast increases with every new join
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/27  by noembryo 2026-09-02T23:38:20Z

I use the [Endless MiniMax H3 (with Endless LipSync)](https://civitai.red/models/2909076/endless-minimax-h3-with-endless-lipsync) workflow to create and stitch video clips together.
The issue is that with every new join, the contrast increases.
It's not so apparent when there are many scene changes, but as one of the user pointed out, if you just make a talking head you can't miss it..😳
These images are from a 52 sec video with 6 clips.
`context_length` is 22.
This one is from the 1st clip
<img width="608" height="800" alt="Image" src="https://github.com/user-attachments/assets/73aa932a-fd94-41cf-94d1-cbea1ec55c4b" />
..and this one is from the 6th
<img width="608" height="800" alt="Image" src="https://github.com/user-attachments/assets/d3529673-2d23-4825-b446-0d1362ed08ac" />
Any idea for how to overcome this?

P.S. If you like I can upload the video too..

> **Comment by NikoDemon80 (2026-09-03T03:42:10Z):**

This has nothing to do with Motion Context. It is an unfortunate result of changing clips with any video model. Wan, LTX, H3, doesn’t matter. The process of sampling the latent frames results in a small percentage of contrast bloom, detail loss, and audio dampening per chain. The community as a whole is still trying to figure out how to combat this.



### #26 [closed] ISSUE: Unable to run example workflow when loading saved latent
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/26  by CambridgeComputing 2026-09-02T19:49:33Z

When trying the example workflow, I can generate the first clip when I don't load any latent. The clip generates, and I'm able to save the context latent and I see it in the output folder. When I iterate the load/save latent nodes so that it loads the saved latent, I get the following error:
```
[INFO] got prompt
[INFO] h3_motion_context: loaded AV latent from /mnt/data/ComfyUI/output/h3_context/clip_00001.safetensors
[INFO] h3_motion_context: video from latent, video/head, 22 frames -> 7 cond blocks at indices 0..18, 124 frame clip at 336x608, trim 22, audio 24 frames -> 40 latent steps (1.000s) from latent, on the timeline ending at frame 22.200
[INFO] Model MiniMaxH3 prepared for dynamic VRAM loading. 32427MB Staged. 0 patches attached. Force pre-loaded 210 weights: 1142 KB.
  0%|                                                                             | 0/20 [00:00<?, ?it/s,   Model Initializing ...  ]
[ERROR] !!! Exception during processing !!! shape '[1, 24, 1, 1, 19, 2, 10, 2]' is invalid for input of size 19152
[ERROR] Traceback (most recent call last):
  File "/mnt/data/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/mnt/data/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy_api/latest/_io.py", line 2025, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy_extras/nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/custom_nodes/ComfyUI-TiledDiffusion/utils.py", line 34, in KSAMPLER_sample
    return orig_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dmcdonald/miniconda3/envs/comfyui/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/k_diffusion/sampling.py", line 1528, in sample_res_multistep
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dmcdonald/miniconda3/envs/comfyui/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/k_diffusion/sampling.py", line 1486, in res_multistep
    denoised = model(x, sigmas[i] * s_in, **extra_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/model_base.py", line 208, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/model_base.py", line 252, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dmcdonald/miniconda3/envs/comfyui/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dmcdonald/miniconda3/envs/comfyui/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/ldm/minimax/model.py", line 566, in forward
    out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/ldm/minimax/model.py", line 674, in _forward
    cond_video_rows = self._cond_video_rows(payload, device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/ldm/minimax/model.py", line 532, in _cond_video_rows
    r = patchify_video(z.to(torch.float32), self.patch_size)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/data/ComfyUI/comfy/ldm/minimax/model.py", line 47, in patchify_video
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[1, 24, 1, 1, 19, 2, 10, 2]' is invalid for input of size 19152
```

I've pulled the latest code form Comfy a few hours ago as well as this repo so everything should be up-to-date. If there's anything you need me to try to help figure out what's going on here, let me know and I'll do my best. Thanks!

> **Comment by NikoDemon80 (2026-09-02T20:20:27Z):**

336 x 608 is not a valid resolution. For H3, your resolution MUST be a whole number multiple of 32. Why it even let you generate the first clip, I don't know. Make sure the Resolution Picker node multiple says 32, not 8 or 16. This is like the 10th issue raised for this exact reason.


> **Comment by CambridgeComputing (2026-09-03T00:19:57Z):**

> 336 x 608 is not a valid resolution. For H3, your resolution MUST be a whole number multiple of 32. Why it even let you generate the first clip, I don't know. Make sure the Resolution Picker node multiple says 32, not 8 or 16. This is like the 10th issue raised for this exact reason.

Good catch! I am actually using the using the Resolution node with a 32 mutliple set for 0.5MP (544x960). I used your example workflow with no modifications so I started digging since that resolution wasn't found anywhere. I uninstalled or disabled a ton of custom H3 modes. That in turn revealed a ComfyUI bug (https://github.com/Comfy-Org/ComfyUI/issues/15960) that I manually patched and I'm back in business. Thanks for the help, and this fantastic node pack!



### #25 [closed] ISSUE: Treat Load Latent clip_index=0 as a first-clip bypass
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/25  by feigo313 2026-09-01T08:47:11Z

## Problem

When generating the first clip in a chain, the Motion Context node must be disabled manually. If it remains enabled while there is no previous clip latent, the node raises a `nothing to pin` error. This makes the workflow harder to use because the user must toggle the node for the first clip and toggle it back for subsequent clips.

## Current behavior

`MiniMaxH3MotionContextLoadLatent` currently interprets `clip_index=0` as “load the newest latent in the folder”. In a fixed-slot chain, users normally manage the clip number explicitly (for example, Load 1 / Save 2, then Load 2 / Save 3). The newest-file behavior is not retry-safe and is not useful for this workflow because a rejected rerun can become the newest file.

## Proposed behavior

Treat `clip_index=0` as an explicit first-clip/no-context mode:

- `Load=0`: do not read a file; return no context.
- Motion Context with no context: bypass cleanly, return the original conditioning, and return `trim_frames=0`.
- Positive indices retain exact-slot loading: `1`, `2`, `3`, etc.

This gives a simple chain workflow:

`Load 0 / Save 1` → `Load 1 / Save 2` → `Load 2 / Save 3`

## Compatibility note

This changes only the meaning of the currently unsafe/non-retry-safe `clip_index=0` mode. Exact positive indexed loading and fixed-slot saving remain unchanged.

I implemented and locally tested this behavior in a small patch. The existing smoke test passes, including the no-context bypass, exact indexed loading, missing-slot errors, latent/audio chaining, retry-safe fixed slots, references, and last-frame handling.

> **Comment by feigo313 (2026-09-01T09:05:16Z):**

I implemented and locally verified the proposed behavior in commit 89e9bc6 (local only; not pushed as a branch or PR).

The minimal implementation changes are:

1. In MiniMaxH3MotionContext.apply(), when neither context_latent nor context_frames is provided, return the original conditioning unchanged with trim_frames=0 instead of raising nothing to pin.
2. In MiniMaxH3MotionContextLoadLatent.load(), treat clip_index=0 as an explicit disabled/no-previous-clip signal and return None; do not resolve or read the newest file.
3. In IS_CHANGED(), return a stable disabled cache key for index 0.
4. Positive indices retain exact-slot loading, and Save Latent fixed-slot behavior is unchanged.

The intended workflow is:

Load 0 / Save 1 for the first clip, then Load 1 / Save 2, Load 2 / Save 3, etc.

I updated the smoke test to cover the first-clip bypass, exact indexed loading, missing-slot errors, retry-safe fixed slots, latent/audio chaining, references, and last-frame handling. The test result is:

tests/_node_smoke_test.py -> smoke test passed

This changes only the old non-retry-safe load-newest meaning of index 0; positive indexed loading and saving remain compatible.


> **Comment by NikoDemon80 (2026-09-02T01:16:21Z):**

Looking into this now.


> **Comment by NikoDemon80 (2026-09-02T14:06:48Z):**

Fixed in 0.5.0.

Load 0 is first-clip/no-context: nothing is read, Motion Context passes the original conditioning through, and trim_frames is 0. Leave the node enabled. The chain is Load 0 / Save 1, then Load 1 / Save 2.

The Chain node is the rest of this: Approve / Run/Re-roll / Chain / Reset, so you do not walk those indices with run-on-change (that queues twice and skips slots). That node is here because of this issue.

Thanks for the inspiration to fix this. I should've made this node weeks ago.

https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/releases/tag/v0.5.0




### #24 [closed] ISSUE: 请更新插件调用
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/24  by pc9821nb-uuliao 2026-08-31T03:51:57Z

我对你这个导演台比较有兴趣。想试用看看。另外。https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context-MultiRef   仓库已经停用。现在改为：https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context 新版插件。请更改调用。

> **Comment by NikoDemon80 (2026-08-31T06:13:23Z):**

你好，感谢关注。

有两点需要澄清：

NikoDemon80/ComfyUI-H3-Motion-Context-MultiRef 这个仓库不存在，也从未存在过。本账号只有 ComfyUI-H3-Motion-Context 这一个。请问你是在哪里看到那个链接的？可能是别人的分支，或者是另一个项目。
本插件没有"导演台"这个功能。它提供的是 MiniMax H3 多镜头之间的画面与音频连续性。你说的导演台可能属于另一个项目，方便的话请贴出链接。

如果你想试用本插件：在 ComfyUI Manager 里搜索 H3 Motion Context，或者直接把仓库克隆到 custom_nodes 目录。README 里有工作流说明。


> **Comment by pc9821nb-uuliao (2026-08-31T23:54:05Z):**

嗯。是我发错地方了。另一个项目拉取了你的仓库。https://github.com/eaglering/MiniMaxRefDirector-ComfyUI#1



### #23 [closed] ISSUE: Slowdown on 0.4.0
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/23  by Dillardn7 2026-08-30T13:13:19Z

I was using the previous version, but since upgrading to v0.4.0, the initial 2 steps on an 8 step generation take far longer, up to 140 seconds, vs 45 with the previous version. Console doesn't read anything different between the two.

> **Comment by NikoDemon80 (2026-08-31T03:08:48Z):**

Any other details? I've not experienced this or had anyone else mention it.


> **Comment by NikoDemon80 (2026-08-31T15:57:27Z):**

Closing due to lack of additional information. It's very hard to trace this issue with no details. The most likely cause is something changed by the user since the update.



### #22 [closed] ISSUE: Better than 0.5 megapixels
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/22  by stena 2026-08-29T21:17:38Z

I've managed to make videos when using 0.5 megapixels or smaller, but when I try to use bigger than that I'll get an error: 

```
# ComfyUI Error Report
## Error Details
- **Node ID:** 248:174
- **Node Type:** SamplerCustomAdvanced
- **Exception Type:** RuntimeError
- **Exception Message:** RuntimeError: shape '[1, 24, 1, 1, 33, 2, 18, 2]' is invalid for input of size 58608
```
fl2va workflow, no turbos, no attentions


> **Comment by NikoDemon80 (2026-08-30T03:09:06Z):**

Not chaining-specific, you can hit this error without the pack.

The reshape wanted 24 x 66 x 36 and got 24 x 66 x 37, so your latent is 66 x 37, or 1056 x 592. 592 is 16 x 37, an odd number of latent columns.

H3 patches video in 2x2 blocks, so both latent dimensions have to be even, and pixel sizes have to be multiples of 32, not 16. ComfyUI pads the target latent for this. Keyframe condition latents are patchified unpadded, which is where it fails. Any fl2va graph with a keyframe and a non-multiple-of-32 canvas does this.

Use 576 x 1056 or 608 x 1056 for now.

The stock H3 nodes step width and height by 32 and cannot produce 592. What is feeding those inputs? A megapixel or aspect-fit node rounding to 16 would explain it. Make sure if you are using the Resolution Picker node that the multiplier is set to 32, not 8 or 16.


> **Comment by stena (2026-08-30T21:34:39Z):**

Yes, that did it. Thank You! Excellent.



### #21 [closed] ISSUE: Shape mismatch (audio rows) when a Motion Context continuation shot also uses a new ref_voice dialogue line
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/21  by princeusk 2026-08-28T08:54:29Z

First off, thank you for building Motion Context — genuine cross-run motion/audio
continuity for MiniMax H3 is exactly what was missing, and it's been very useful
for multi-shot workflows.

## What happens

When Shot 2 continues from Shot 1 via a saved `context_latent` (Save/Load Latent)
AND Shot 2 itself uses a `ref_voice`-style audio reference to speak a NEW line of
dialogue, sampling crashes with a shape mismatch. Shot 1 (no continuation yet) and
a version of Shot 2 with the dialogue/voice-reference removed (pure visual
continuation) both work fine. Only "continuing via context_latent" + "a fresh
spoken line in the continuing shot" together triggers it.

## Environment

- ComfyUI-H3-Motion-Context: v0.4.0 (confirmed via `git tag`)
- ComfyUI: 0.30.0, native MiniMax H3 support
- Sampler: `SamplerCustomAdvanced` / `er_sde`
- Also tested with the `ComfyUI-Spectrum-MiniMax-H3` acceleration node removed
  entirely — identical error, so it isn't a factor.

## Steps to reproduce

1. Build Shot 1 with an `H3 Motion Context` node (no `context_latent` wired),
   sample it, and save the resulting latent with `H3 Motion Context Save Latent`.
2. Build Shot 2 with `H3 Motion Context`, wiring `context_latent` from
   `H3 Motion Context Load Latent` (pointing at Shot 1's saved clip).
3. Give Shot 2 a subject conditioned as a voice reference (`ref_voice`), with a
   new line of dialogue for it to speak in this shot (not present in Shot 1).
4. Sample Shot 2.

## Error
RuntimeError: shape mismatch: value tensor of shape [238, 32] cannot be broadcast to indexing result of shape [318, 32]


Full traceback:

<details>
<summary>Click to expand</summary>

[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [238, 32] cannot be broadcast to indexing result of shape [318, 32]
[ERROR] Traceback (most recent call last):
  File "/mnt/local-scratch/ComfyUI/execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "/mnt/local-scratch/ComfyUI/execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "/mnt/local-scratch/ComfyUI/comfy_api/internal/__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "/mnt/local-scratch/ComfyUI/comfy_api/latest/_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "/mnt/local-scratch/ComfyUI/comfy_extras/nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "/usr/local/lib/python3.13/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "/mnt/local-scratch/ComfyUI/comfy/k_diffusion/sampling.py", line 1551, in sample_er_sde
    denoised = model(x, sigmas[i] * s_in, **extra_args)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 640, in __call__
    out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1208, in __call__
    return self.outer_predict_noise(*args, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1215, in outer_predict_noise
    ).execute(x, timestep, model_options, seed)
      ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 1218, in predict_noise
    return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 620, in sampling_function
    out = calc_cond_batch(model, conds, x, timestep, model_options)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 211, in calc_cond_batch
    return _calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 219, in _calc_cond_batch_outer
    return executor.execute(model, conds, x_in, timestep, model_options)
           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/samplers.py", line 335, in _calc_cond_batch
    output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/model_base.py", line 205, in apply_model
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/model_base.py", line 249, in _apply_model
    model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds)
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/local-scratch/ComfyUI/comfy/ldm/minimax/model.py", line 540, in forward
    out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
    ...<2 lines>...
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
    ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload,
              denoise_mask=denoise_mask, audio_denoise_mask=audio_denoise_mask, **kwargs)
  File "/mnt/local-scratch/ComfyUI/comfy/patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/mnt/local-scratch/ComfyUI/comfy/ldm/minimax/model.py", line 659, in _forward
    all_audio_rows[~audio_update] = cond_audio_rows
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
RuntimeError: shape mismatch: value tensor of shape [238, 32] cannot be broadcast to indexing result of shape [318, 32]

</details>

The crash is in `comfy/ldm/minimax/model.py`, in `_forward()`:

```python
all_audio_rows[~audio_update] = cond_audio_rows
```

As far as I can tell from reading the surrounding code, audio_update.shape[0]
reflects the TOTAL packed audio rows (target audio + all reference blocks), while
cond_audio_rows is built only from payload["cond_audio_latents"]. The v0.4.0
changelog note "the pinned audio is a keyframe rather than a reference block" looks
directly related to this area, but something still doesn't reconcile when a FRESH
ref_voice dialogue is also present in the continuing shot — possibly the pinned
keyframe-audio row count and the newly-conditioned dialogue row count are each
computed independently and don't agree on the total.

What I've ruled out
Confirmed on the latest v0.4.0 tag (freshly cloned).
Reproduces identically with the Spectrum acceleration node removed.
Reproduces on a from-scratch chain (deleted all Motion Context nodes, cleared
cache, rebuilt from Shot 1).
Reproduces on a completely clean/reproducible case: Shot 2 with the dialogue
line and voice reference removed (pure visual continuation) samples correctly.

Happy to test further if useful (my workflow uses some custom in-house nodes for
prompt building, so I'm not able to share the workflow file itself, but I can
reproduce the issue with stock nodes if that would help narrow it down). Thanks
again for the great tool!


> **Comment by NikoDemon80 (2026-08-28T14:14:44Z):**

You are on the wrong version of ComfyUI. H3-Motion-Context v0.4.0 cannot be used on anything older than Comfy v0.34.0

<img width="1005" height="620" alt="Image" src="https://github.com/user-attachments/assets/d8961672-c2c0-4515-98de-389bbba68f64" />


> **Comment by princeusk (2026-08-28T15:13:26Z):**

Apologies; Claude Code, which I tasked with drafting the message, listed the wrong version.

The error is occurring with version 0.34.0.

I always clone and use the latest version on Google Colab, and I have verified this via the startup logs:

[INFO] ComfyUI version: 0.34.0
[INFO] comfy-aimdo version: 0.4.15
[INFO] comfy-kitchen version: 0.2.31

Would it be possible to reopen the issue with this correction? 
Happy to re-verify anything else that would help.

Below is an apology from Claude Code (feel free to skip it).
---
Sorry — the "0.30.0" in my original report was wrong. (For context: I used an AI coding assistant to help draft the issue, and it mistakenly pulled that number from stale per-node metadata in the workflow JSON rather than the actual runtime version.)


> **Comment by NikoDemon80 (2026-08-28T20:03:29Z):**

Looking into this now.


> **Comment by NikoDemon80 (2026-08-28T20:13:21Z):**

Reopened, and thanks for the correction. 0.34.0 fits your traceback, so that part is settled.

Where this now stands: On stock 0.34.0 this crash should not be reachable. The layout and the audio condition list are built from the same keyframes list, in the same function, three lines apart, so they cannot disagree about the pinned block. Which means something in your install is editing the conditioning after the Motion Context node or editing the payload during sampling.

The numbers point at which block went missing. 80 rows is 40 audio latent steps, one second, which is exactly what the default audio_context_length of 24 frames produces. The 238 rows that did arrive are 119 steps, and the pinned window cannot come out at 119 for any whole frame count, so those are your voice reference. The pinned audio is the block the payload lost.

Two things would find it:

Your offer of a stock-node repro. That is the most useful thing by far. Stock Reference to Video, Motion Context, no in-house nodes anywhere in the graph. If it still crashes, it is mine or upstream's and I will chase it. If it does not, your prompt nodes are rewriting something, and we will know where to look.

Also useful: ln custom_nodes. Another H3 pack, or an older copy of this one, would do this. Versions of this pack before 0.3.1 patched the payload and rebuilt the audio list from references alone, which drops exactly the block you are missing.


> **Comment by princeusk (2026-08-29T08:50:08Z):**

## Root cause found — it's not Motion Context, it's a cross-package `extra_conds` conflict with ComfyUI-H3-Multishot

I built a from-scratch ComfyUI 0.34.0 environment and reproduced the exact crash (identical shapes `[238, 32]` vs `[318, 32]`, same `comfy/ldm/minimax/model.py:685` line) using the reference-to-video → Motion Context chain, with no H3RefBuilder nodes in the graph.

**Root cause:** `ComfyUI-H3-Multishot` monkey-patches `comfy.model_base.MiniMaxH3.extra_conds` at import time, unconditionally, regardless of whether any of its own nodes are wired into the graph:

```python
# ComfyUI-H3-Multishot/h3_avbank_probe.py
def extra_conds(self, **kwargs):
    out = _orig(self, **kwargs)
    kf = kwargs.get("minimax_keyframes")
    refs = kwargs.get("minimax_refs")
    if not kf or not refs:
        return out
    ...
    aud = [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]
    if aud:
        payload["cond_audio_latents"] = aud
```

When both `minimax_keyframes` and `minimax_refs` are present on the conditioning, this patch rebuilds `cond_audio_latents` from `refs` only. It silently drops any `audio_latent` carried on a keyframe — which is exactly what Motion Context's `audio_mode="timeline"` uses to pin the continuation audio.

Meanwhile `PackedLayout` (`comfy/ldm/minimax/model.py`), which is untouched by that patch, still counts row space for the keyframe's pinned audio correctly from `keyframes`/`refs` directly. So the layout allocates 318 rows (238 for the shot's own reference audio + 80 for Motion Context's pinned continuation audio), but the actual audio payload only supplies 238 rows — hence the exact mismatch reported here.

This triggers only when all three are true: `ComfyUI-H3-Multishot` is installed (its nodes don't need to be in the graph), Motion Context is chaining a continuation with its default timeline audio pinning, and the continuing shot also carries its own fresh reference audio — precisely my original setup.

Verified: with `ComfyUI-H3-Multishot` removed from `custom_nodes`, the identical workflow (same seed, same media, same prompts) completes successfully.

I don't think there's anything to change in this pack — the bug lives in Multishot's `extra_conds` patch. I'll open an issue on that repo as well and link it here for visibility. Thank you again for helping narrow this down — Motion Context has been very useful for my chained-shot workflows and I wanted to get you a clean stock repro as promised.


> **Comment by princeusk (2026-08-29T09:08:03Z):**

Quick follow-up: turns out someone had already hit this exact bug and opened an issue on the Multishot repo two days before I got there — https://github.com/jlucasmcrell/ComfyUI-H3-Multishot/issues/20 — with the same root cause (the `extra_conds` patch dropping keyframe audio) and matching (if different-length) row numbers. The reporter's own fork already has a fix that merges keyframe and ref audio the same way video latents are merged: https://github.com/jalberty2018/ComfyUI-H3-Multishot

So no need for me to open a duplicate issue there. Just wanted to close the loop here in case anyone else finds this thread with the same symptom.


> **Comment by NikoDemon80 (2026-08-30T03:10:59Z):**

Hey, excellent work on the research and follow up! Closing out.


> **Comment by lesca (2026-09-14T03:34:25Z):**

Thank you @princeusk , I encounted the same issue. In my case the same workflow worked before but not today. In the workflow, there is no any node from MultiShot. I probably installed MultiShot from another workflow. It is fixed by just removing the MultiShot from custom_nodes folder.



### #20 [closed] ISSUE: Color drift - stitching
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/20  by SveSop 2026-08-27T23:52:11Z

I find the node(s) very very good and useful. It does have some pitfals, but mostly it works nicely...

I wonder if it is expected that the degradation is so prominent? 2 issues am mostly seeing tho:

1. Easy to spot the stitching spot - Sometimes its ok, sometimes its very easy to spot. This can be fixed slightly with prompting/seed-hunting, but mostly it is a annoyance. A ever so slightly noticeable "flash".
2. The 2nd video is "ok", by the 3rd or 4th extension, the colors are .. well.. not-so-good

I suppose the model will re-use a "slightly bad ending" and start from bad -> even worse -> even worse and so on. (Expected degradation)

I have seen a similar issue mentioned on reddit - with some fixit hack i cant find atm, but i think it was some color-chroma crap done on the input image (Yes, this was with comfy native context node), so i am kind of wondering if there would be a useful thing to look into? Can't say i know exactly how it could be fixed tho, as this node is not using a last-frame-continuation. 
Ofc keeping generations < 3 clips is the best "fix", but sometimes its beneficial to keep a scene (room/items), or maybe background music. Starting a new scene with a captured image, usually ends up with a completely new music layer (atleast those cases where non-diegetic-music is being used over just adding music in post production).

Just some thoughts. Feel free to close this as a wontfix tho, if its just something to be expected and not really feasible to fix.

Anyway - Thanks for your work 👍 

> **Comment by NikoDemon80 (2026-08-28T14:24:06Z):**

I spent the last couple of days working on this exact issue, so here is what is actually going on:

**What you are seeing is real**

-Chained clips get brighter and more contrasty as the chain goes on. Roughly 4% per join, which does not sound like much, but it compounds. By clip 8 the picture is about 40% more contrasty than clip 1.

-Shadows go crushed and highlights blow out. By the end of an 8-clip chain about 2% of the picture is pure black and half a percent is pure white. Once that happens the detail is gone for good, so color grading the finished chain afterwards will not bring it back.

-The sound does the same thing in the opposite direction. Each link loses about a third of its treble relative to the bass. That is why long chains start sounding deeper, muddier, a bit echoey. It is not your imagination, and it is not the music.

**Why it happens**

-The pinned frames are not copied into the next clip. They are handed to the model as a reference and the model regenerates something that looks and sounds like them. That reproduction is not neutral: it comes back slightly hotter and slightly duller than the thing it was copying.

-Then that reproduction becomes the reference for the clip after it. So, the small error is fed forward and multiplies. It is the same reason a photocopy of a photocopy gets worse.

**Things it is not, so nobody wastes time on them**

I checked each of these and ruled them out:

-Not the video file encoder
-Not a color space or brightness range setting
-Not the part that converts between the model's internal format and pixels
-Not anything before the picture is generated, which stays completely stable across a whole chain

I also built a fix for one of those theories, measured it, and it made the sound four times worse. So that idea is dead and I am mentioning it so no one else rebuilds it.

**Where this leaves things**

-The gain happens inside generation, so it cannot be fixed after the fact by adjusting files or settings. The promising direction is to hand the model a slightly cooler reference, so its reproduction lands in the right place, but that is untested and I do not want to ship a number I have not proven.

**In the meantime**

-Shorter chains hold up much better. The problem is barely visible for the first few links.
-If you are chaining long, expect to lose some shadow and highlight detail and shoot with that in mind.
-Starting a fresh chain from a new clip resets it.


> **Comment by SveSop (2026-08-29T10:26:38Z):**

Thank you for looking into this 👍 

Yeah, i get around it by doing some "start fresh" parts in longer scenes - ie. i use a frame from one of the first clips as a regular "first image" for a next chain, as a slight workaround.

It is not really something new only for H3, as this had a tendency to happen in wan2.2 (SVI Pro) continuation, and LTX extended video's. I suppose a different workaround is to inject a "end frame" into a clip, which used to be one of my workarounds with Wan/LTX, but i have not experimented with it too much with the H3, and i am a wee bit unsure about chaining a "hard" end-frame at -1 with comfy native conditioning node together with this context node?



> **Comment by NikoDemon80 (2026-08-30T03:17:47Z):**

The functionality is there. You can use a last frame reference while chaining. I find it results in a lot of scene cuts though instead of a smooth transition into the image. Just depends on what you're chaining. Thanks for the thread!



### #19 [closed] ISSUE: Conflict with ComfyUI-SolAttn_triton: SolAttn wraps PackedLayout.__init__ first, so the layout patch refuses to install
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/19  by feigo313 2026-08-26T14:08:17Z

## Problem

`MiniMaxH3MotionContext` fails with `RuntimeError: h3_motion_context: the layout patch could not be applied...` whenever the workflow also uses `SolAttnPatch` from [kijai/ComfyUI-SolAttn_triton](https://github.com/kijai/ComfyUI-SolAttn_triton) on a MiniMax-H3 model.

Log right before the exception:

```
h3_motion_context: another pack has already patched H3's layout (PackedLayout.__init__ now comes from 'E:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SolAttn_triton._morton_h3'). Several ComfyUI packs lift the same first/last keyframe restriction independently and they cannot both own it, so this one is refusing rather than wrapping a wrapper and producing joins neither pack intended. Disable one of them and restart.
RuntimeError: h3_motion_context: the layout patch could not be applied, so interior anchors would be rejected by ComfyUI. The reason was logged just above this error.
```

## Root cause

- `SolAttnPatch` on an H3 model calls `install_h3_hooks` whenever `morton` is on **or** `sink_conditioning != "off"` (the node default is `exact_kv_and_rows`), and `_morton_h3._patch_packed_layout()` wraps `comfy.ldm.minimax.model.PackedLayout.__init__` with an observer that calls the original constructor and then registers the layout keyed by `id(position_ids)`.
- `patch_layout._already_patched()` sees a wrapper whose `__module__` is `...ComfyUI-SolAttn_triton._morton_h3` (not this pack's `_patched_init`) and classifies it as a foreign pack -> `apply_patch()` refuses -> the node raises.

`SolAttnPatch` executes before `MiniMaxH3MotionContext` in the graph, so Sol-Attn always wins the race and the failure is deterministic.

## Why "stand down" is not an option here

On ComfyUI **0.33.4** (the version where this was hit), `PackedLayout.__init__` still takes `frame_count` and still enforces the first/last-only anchor restriction (`raise ValueError("only first/last keyframe anchors are supported")`). So treating Sol-Attn's observer as "the layout patch is already installed" would be wrong: the observer does not implement the interior-anchor / audio-timeline rewrite this pack needs. (For reference, ComfyUI 0.34.0 removed `frame_count` and the restriction.)

## Proposed fix (implemented locally, all harness tests pass)

Make this pack own `PackedLayout.__init__` from the start: install the layout patch at plugin import time instead of first node use. Sol-Attn's observer runs later (at node execution) and then wraps *our* `_patched_init`, so both packs work:

```
PackedLayout.__init__ = H3._patched_init        (installed at import)
then SolAttn observer:  SolAttn wrapper -> H3._patched_init -> stock
```

- H3's `_fixup` / `_fixup_audio` mutate `position_ids` **in place**, so Sol-Attn's registration keyed on `id(position_ids)` still hits.
- The wrapper stays inert for unrelated H3 graphs (gated on `MC_KEY` / `MC_AUDIO_KEY`), so eager install does not change behaviour for graphs that never use this pack.
- `patch_payload` can stay first-use: nothing else patches `MiniMaxH3.extra_conds`.

The change is a few lines in `__init__.py`:

```python
from .patch_layout import apply_patch as _apply_layout_patch

try:
    _apply_layout_patch()
except Exception:
    logging.getLogger("h3_motion_context").exception("layout patch failed at startup")
```

I also added `tests/_coexist_solattn_test.py`, which simulates a Sol-Attn-style observer wrapping the patched init and asserts that interior anchors, the audio move and the observer's registration all survive, for both the 0.32-style (frame_count) and 0.33/0.34-style (no frame_count) constructors, and that the reverse order (observer first) is still refused.

Happy to turn this into a PR if you'd like. An alternative is a cooperation-marker ABI between the two packs (Sol-Attn publishes a marker plus the original constructor it captured, so either order works), but the eager install is a one-sided fix that needs no changes on Sol-Attn's side.

## Environment

- ComfyUI 0.33.4 (also verified against 0.34.0 source)
- ComfyUI-H3-Motion-Context 0.3.1 (f80e36b)
- ComfyUI-SolAttn_triton dfc2e31
- Windows, RTX 5090 D


> **Comment by NikoDemon80 (2026-08-26T15:46:24Z):**

Looking into this now.


> **Comment by NikoDemon80 (2026-08-26T18:23:44Z):**

Thanks for the diagnosis. The race, the observer wrapping `PackedLayout.__init__`, and why standing down would be the wrong call on 0.33.4 are all correct.

Fixed in 0.4.0, on ComfyUI 0.34.0. No change needed on either side.

0.4.0 takes a different route than the eager-install patch: ComfyUI 0.34.0 now places interior keyframe anchors itself, so this pack no longer wraps `PackedLayout.__init__` at all. The two packs no longer compete for that constructor. Sol-Attn can keep observing it; we just aren't trying to own it.

On 0.33.4 the first/last restriction is still in stock, so 0.4.0 refuses there and points at 0.3.1. 0.3.1 still has the conflict you hit. The coexistence path is ComfyUI 0.34.0 + this pack 0.4.0.



> **Comment by NikoDemon80 (2026-08-26T18:23:45Z):**

Closing as fixed in [v0.4.0](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/releases/tag/v0.4.0). Reopen if 0.34.0 + 0.4.0 still collide.



### #18 [closed] ISSUE: new comfyui version error
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/18  by fangg2000 2026-08-25T01:59:03Z

with latest comfyui version
ValueError: h3_motion_context: context_latent is 736x416 but this clip is 608x352. A latent cannot be resized, so the previous clip has to be regenerated at this resolution, or the chain restarted here.


> **Comment by fangg2000 (2026-08-25T02:58:21Z):**

solution follows：
1.nodes.py（at row 442）
`# before
if src_w != width or src_h != height:
                # a latent cannot be resized. Falling back to frames here
                # would quietly take the lossy path on a graph the user
                # thinks is fine, and a resolution change mid-chain is
                # nearly always a mistake, so say so instead.
                raise ValueError(
                    "h3_motion_context: context_latent is %dx%d but this "
                    "clip is %dx%d. A latent cannot be resized, so the "
                    "previous clip has to be regenerated at this "
                    "resolution, or the chain restarted here."
                    % (src_w, src_h, width, height))

# after
if src_w != width or src_h != height:
                import torch.nn.functional as F
                _LOG.warning(
                    "h3_motion_context: context_latent resolution %dx%d differs "
                    "from current clip %dx%d. Resizing latent spatially (this may "
                    "reduce quality). Prefer using context_frames instead.",
                    src_w, src_h, width, height)
                target_h = int(height // 16)
                target_w = int(width // 16)

                # src_video 形状为 [B, C, T, H, W]
                B, C, T, H, W = src_video.shape
                # 将 T 移到 batch 维度，变成 [B*T, C, H, W] 以便进行 2D 插值
                src_video_reshaped = src_video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                # 对 H 和 W 进行双线性插值
                src_video_resized_reshaped = F.interpolate(
                    src_video_reshaped,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                # 还原形状 [B, T, C, target_h, target_w] -> [B, C, T, target_h, target_w]
                src_video_resized = src_video_resized_reshaped.reshape(B, T, C, target_h, target_w).permute(0, 2, 1, 3, 4)

                # 替换 context_latent 中的视频流
                parts = _streams_from_latent(context_latent)
                parts[0] = src_video_resized
                context_latent = {"samples": parts}
                src_video = src_video_resized
                src_w = width
                src_h = height`

2.patch_layout.py（Choose based on the situation, at row 664）
`try:
        #_self_test()
        print('test over')
    except Exception as exc:`


> **Comment by NikoDemon80 (2026-09-05T17:43:37Z):**

Replying to this for posterity. This is expected behavior. When chaining videos, the resolution must stay the same.



### #17 [closed] ISSUE: H3 Motion Context Clip Stitcher
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/17  by noembryo 2026-08-21T21:40:43Z

This is not an issue, but more of a heads up for @NikoDemon80 and anybody that is using ComfyUI-H3-Motion-Conext.

I don't know if there is a better way to do this, but I couldn't find an easy way to concatenate the saved latent clips together.
I could edit the mp4 files together, but this degrades quality and it takes time to make it seamlessly.
I was also using the [Endless Wan 2.2 I2V (SVI 2 Pro)](https://civitai.red/models/2701632/endless-wan-22-i2v-svi-2-pro) with Wan2.2, that let you extend and preview the whole video, when adding an new part, so I was spoiled.
I didn't like the big 1-node, all-in-one solutions, because I have created my own workflow and I couldn't integrate them into it.
The best solution for me was ComfyUI-H3-Motion-Conext, that worked perfectly, but lacked (from what I could tell) the features I mentioned.

So, I made a node that collaborates with ComfyUI-H3-Motion-Conext.
The [H3 Motion Context Clip Stitcher](https://github.com/noembryo/ComfyUI-noEmbryo#h3-motion-context-clip-stitcher).
It can be used either standalone, to stitch the saved clips to a final video, or inside a generation chain to create the final video from the already saved clips, plus the currently generated video part.

![](https://raw.githubusercontent.com/noembryo/ComfyUI-noEmbryo/master/stuff/H3MotionContextClipStitcher.png)


> **Comment by NikoDemon80 (2026-08-21T22:02:49Z):**

Nice! I use my own node for this. My other repo is called Image Oasis. There is a very powerful node in the pack called Video Oasis Viewer that I use to preview, save, clip, concat, etc video clips!


> **Comment by noembryo (2026-08-21T22:15:20Z):**

Hmm. It sounds like the `ComfyUI_MiniMax_H3_Extender` or the `[MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop)`. But, maybe not. You don't have any screenshots to check them out.
Oh, well.. I just wanted to stitch the clips together, nothing more.
Maybe I'll check them when I find some free time.
Thanks..



### #16 [closed] ISSUE: RuntimeError: shape mismatch when running latent upscale with Motion Context latent loader/continuation
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/16  by Covalent0714 2026-08-20T17:08:37Z

Cross referencing https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/issues/16

It seems attempting to run a latent upscale within the same workflow as the motion context nodes throws an error after creating clip 1 and loading clip 1 to make clip 2.

```
[ERROR] !!! Exception during processing !!! shape mismatch: value tensor of shape [5541, 96] cannot be broadcast to indexing result of shape [9699, 96]
```

Per other users in that issue thread, it appears the bug might have something to do with the context nodes rather than the latent upscaler as it throws this error regardless of which latent upscaler is used.

> **Comment by NikoDemon80 (2026-08-21T01:18:08Z):**

This is not the upscaler and it is not the context node. It is a core assumption.

The error comes from `comfy/ldm/minimax/model.py`, on the line that fills conditioning video rows into the sequence. Core sizes the conditioning slots from the **target latent's** spatial grid, but the conditioning latents themselves were encoded earlier, at whatever size the latent was then. Nothing checks that those still agree.

So, any node that changes the latent's width or height between the H3 node and the sampler will throw this. It is not upscaler-specific, which is why it happens with all of them.

You can reproduce it with no custom nodes: stock MiniMax H3 Image to Video with a first frame, latent upscale between its LATENT output and KSampler, conditioning passed straight through. Same exception.

Clip 1 works because it has no conditioning video latents, so that line never runs. Clip 2 has pinned context rows, so it does.

Fix the wiring order: upscale the latent **before** it goes into H3 Motion Context, not after. The node takes the target latent as an input and sizes the pinned frames to it, so it needs to see the final resolution.

One catch. `context_latent` cannot cross a resolution change, because a latent cannot be resized. The node already refuses that with a clear message. On an upscaled chain, use `context_frames` instead. You eat a VAE round trip, which is the cost of changing resolution mid-chain.

If you are doing a low-res pass, upscale, then a second pass on the same conditioning, that will not work with pinned context at all. The second pass needs conditioning built at the upscaled size.



### #15 [closed] ISSUE: Combining the latents together?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/15  by shawnlau23-coder 2026-08-19T15:08:07Z

Perhaps you could mention how to combine the latents to make the full length video? 

"Concatenate the trimmed clips with a plain butt join" 

Could you maybe include a workflow to do that? I guess if no crossfade is needed, if you save the videos as MKV FF1, you could just join the videos? Joining the latents and making a video from that would be cleaner. Do you have a link to a workflow that could do that? I tried Herrgotts stitcher but ti errors with your latents.

> **Comment by NikoDemon80 (2026-08-19T21:31:01Z):**

I use my own nodes for that. Look for Image Oasis on the custom node manager. There is a node in the pack called Video Oasis Viewer. That's how I build movies from clips. Or, you could pull the repo from my github.

Edit - I should mention that I do not join the latents. I join the processed clips end-to-end with VOV. Joining the latents would mean doing one large decode at the end only to find out you don't like one section of the movie, or one clips audio was garbage. Joining the processed clips is the way to do it.


> **Comment by zixaphir (2026-09-09T22:00:33Z):**

Just wanted to add in case somebody else needs help with this: the built in "Batch Images" and "Concatenate Audio" nodes can be used to stitch everything together right after Trimming each slice.



### #14 [closed] ISSUE: ValueError when running example workflow
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/14  by cvar66 2026-08-18T21:51:12Z

Using the latest ComfyUI portable on WIN 11. When I try to run the example workflow for FL2VA on the first run always get this error. I have Motion Context  disabled and save latent clip set to 1

[ERROR] !!! Exception during processing !!! Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of [736, 576, 124] and output size of (512, 400). Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.
[ERROR] Traceback (most recent call last):
  File "C:\Comfy2028\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
    )
    ^
  File "C:\Comfy2028\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Comfy2028\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
  File "C:\Comfy2028\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
  File "C:\Comfy2028\ComfyUI\comfy_api\latest\_io.py", line 1990, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
  File "C:\Comfy2028\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1055, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "C:\Comfy2028\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1237, in inner_sample
    samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
  File "C:\Comfy2028\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1005, in sample
    samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
  File "C:\Comfy2028\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "C:\Comfy2028\ComfyUI\comfy\k_diffusion\sampling.py", line 1460, in sample_res_multistep
    return res_multistep(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_noise=s_noise, noise_sampler=noise_sampler, eta=0., cfg_pp=False)
  File "C:\Comfy2028\python_embeded\Lib\site-packages\torch\utils\_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "C:\Comfy2028\ComfyUI\comfy\k_diffusion\sampling.py", line 1421, in res_multistep
    callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1003, in k_callback
    callback(x["i"], x["denoised"], x["x"], total_steps)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Comfy2028\ComfyUI\comfy\samplers.py", line 1295, in callback
    return packed_callback(step, x0, x, total_steps)
  File "C:\Comfy2028\ComfyUI\latent_preview.py", line 128, in callback
    preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
  File "C:\Comfy2028\ComfyUI\custom_nodes\comfyui-videohelpersuite\videohelpersuite\latent_preview.py", line 55, in decode_latent_to_preview_image
    num_images)).run()
                 ~~~^^
  File "threading.py", line 995, in run
  File "C:\Comfy2028\ComfyUI\custom_nodes\comfyui-videohelpersuite\videohelpersuite\latent_preview.py", line 67, in process_previews
    image_tensor = F.interpolate(image_tensor, (512, width), mode='bilinear')
  File "C:\Comfy2028\python_embeded\Lib\site-packages\torch\nn\functional.py", line 5098, in interpolate
    raise ValueError(
    ...<4 lines>...
    )
ValueError: Input and output must have the same number of spatial dimensions, but got input with spatial dimensions of [736, 576, 124] and output size of (512, 400). Please provide input tensor in (N, C, d1, d2, ...,dK) format and output size in (o1, o2, ...,oK) format.

> **Comment by NikoDemon80 (2026-08-19T00:38:49Z):**

512x400 is not a valid output for H3. Your output resolution MUST be divisible by 32. 
512 is fine.
400 is not.
416 or 384 would work.


> **Comment by cvar66 (2026-08-19T01:07:46Z):**

I'm using the resolution selector in the workflow. It is already set to multiples of 32 at 16:9 0.7 megapixels. 

EDIT: I figured out the problem, it was the animated preview that was causing the issue. I turned them off and it works now


> **Comment by NikoDemon80 (2026-08-19T21:26:18Z):**

Ahh, ok. I turned that off so long ago, I forget it's a thing. I'll close, thanks.



### #13 [closed] ISSUE: Issue with size?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/13  by TheRamosOnline 2026-08-15T14:52:43Z

I accidentally changed a setting without noticing, ignore this.


### #12 [closed] ISSUE: Issue with ComfyUI Latest (v0.33.0)
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/12  by javawock7618 2026-08-14T07:14:29Z

Hello. Thank you for creating such a powerful custom node. It has become a core part of my workflows.

I’m getting straight to the point: I encountered an error with the latest version of ComfyUI, so I wanted to report it.

[ERROR] !!! Exception during processing !!! h3_motion_context: the layout patch could not be applied, so interior anchors would be rejected by ComfyUI. The reason was logged just above this error.
[ERROR] Traceback (most recent call last):
  File "C:\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "C:\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\nodes.py", line 421, in apply
    _ensure_layout_patch()
  File "C:\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\nodes.py", line 80, in _ensure_layout_patch
    raise RuntimeError(
RuntimeError: h3_motion_context: t

This issue occurs with commit `7fe8a6138504f90ff7be82f3babf416da32876b1` (0.33.0), but does not occur with `ddbaa8752874c275290d054ee4fddd6e004f5fdf` (0.32.0).

Therefore, I suspect that something in ComfyUI's internal processing has changed between these versions.

Could you please take a look when you have a chance?

Thank you!


> **Comment by javawock7618 (2026-08-14T07:29:51Z):**

I also found a similar issue in #8:

https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/8

It looks like the issue may be related to `ComfyUI-SolAttn_triton`, since I also have the Sol-Attn nodes installed in my setup.

In that issue, the warning specifically mentions that `ComfyUI-SolAttn_triton._morton_h3` has already patched H3's layout, causing a conflict with H3 Motion Context.

Could this also explain the issue I'm experiencing? If so, I hope resolving this conflict will fix the problem.



> **Comment by azra1l (2026-08-14T21:06:15Z):**

I posted #8. I am also getting this error message again.

There should be more errors or warnings above the first log line you posted here.

Mine now says  

```
[WARNING] h3_motion_context: self-test failed (PackedLayout.__init__() got an unexpected keyword argument 'frame_count'), patch not applied. Interior keyframe anchors unavailable.
[WARNING] h3_motion_context: if you have more than one H3 Motion Context folder in custom_nodes (a fork, a backup, a manual clone alongside a Manager install), that is the usual cause: each copy self-tests against whichever one loaded first. Keep one and remove the rest. Renaming a folder does not stop ComfyUI loading it. Otherwise this is an upstream ComfyUI change and the message above says what moved.
```

I had removed all conflicting nodes (that i know off). before updating comfyui, this node still worked perfectly, so some comfyui changes in 0.33 must be the cause.


> **Comment by azra1l (2026-08-14T21:48:47Z):**

I can confirm. Breaking changes come with comfyui 0.33. 

I switched from 0.33 back to 0.32 (you can easily do this from within the extension menue) and now it works again.

Now back to 🎥, these actors need to do extra shifts today.


> **Comment by NikoDemon80 (2026-08-15T00:22:11Z):**

Looking into it now! Been a busy day.


> **Comment by NikoDemon80 (2026-08-15T01:05:48Z):**

Short version for anyone hitting the error: if you updated to 0.33 and Motion Context started failing with "the layout patch could not be applied", that is real and it is on my end, not your install. Fix is going up today. Roll back to 0.32 if you need it working right now.

Longer version, because what changed upstream is more interesting than the bug.

Motion Context existed because stock ComfyUI would only anchor a keyframe at the first or last frame of an H3 clip. Anything in between raised. The pack worked around it by handing every keyframe a legal index, smuggling the real one alongside, and rewriting the position coordinates after the stock constructor returned.

0.33 removed the restriction. Anchors now land at any frame natively, references compensate the timeline correctly, keyframes can carry a multi-frame clip instead of a single still, and there is a new node, Add Guide for MiniMax H3, that exposes all of it. It also fixed a bug where attaching a reference wiped the keyframe latents, which the pack was separately patching around. The parameter my code depended on went away with the restriction it existed to enforce, hence the breakage.

So, a good chunk of what this pack did is now in ComfyUI, and you do not need me for it. That is a good outcome. If all you wanted was to anchor a still at frame 30, use Add Guide.

What is still worth installing the pack for:

Latent passthrough. Add Guide takes images and audio and encodes them with the VAEs. Motion Context takes the previous clip's latent and slices the tail straight out of it. No decode, no resize, no re-encode. Over a long chain that round trip is where the excess color drift and softening come from (beyond the 1.3x texture stacking).

Audio that continues instead of restarting. Add Guide anchors audio starting at a frame index and running forward. To actually continue a soundtrack across a join you need the pinned window to end at the join and reach backwards. That is the difference between the model continuing your track and the model writing something that sounds like your track, and on anything with a beat you can hear it immediately.

Plus, the trim node, the audio grid overhang compensation, and the seam probe.

Plan from here: compatibility release today so 0.33 works, then a rebuild on top of the new public keyframe format so the pack stops monkey patching ComfyUI internals entirely. I would rather depend on a documented feature that a stock node also uses than on a constructor signature. Both of the last two breakages came from that dependency and neither would have been possible without it. Add Guide anchors and Motion Context heads will be a supported combination rather than something that trips a guard.

Thanks to javawock7618 and azra1l for the reports and for narrowing it to the exact commits, which made this a twenty-minute diff instead of an all-nighter.

Posting this to Reddit shortly. Fix going live shortly.


> **Comment by NikoDemon80 (2026-08-15T01:25:41Z):**

Fix is live. Starting the rework now. Will push another update when it's finished and tested working. Leaving this open until I push that update so people know I'm working on it. Thanks!


> **Comment by AlUlkesh (2026-08-15T05:18:39Z):**

@NikoDemon80
Thanks for the quick fix. This is an excellent node and I'm happy to be able to use it again.


> **Comment by javawock7618 (2026-08-15T07:57:06Z):**

Thank you for the fix. I’m glad my report was helpful. Keep up the great work!


> **Comment by azra1l (2026-08-15T16:10:35Z):**

Doing the lords work right there. Thanks!


> **Comment by NikoDemon80 (2026-08-15T18:25:08Z):**

Currently waiting on 15375 to merge before I can finish and push v0.4. For now, the hotfix will work. Feel free to drop a message here when it merges and I'll get on it.


> **Comment by NikoDemon80 (2026-08-15T21:23:06Z):**

Fixed in 0.3.1, now on GitHub and the registry.

What happened: ComfyUI removed frame_count from PackedLayout.__init__, along with the restriction that rejected any keyframe anchor other than the first or last frame. That restriction is the reason this pack patched the layout at all, so the parameter went away with the thing it existed to enforce. The pack's self-test passed it unconditionally, hit a TypeError, and refused to run. Refusing was correct, but not useful.

0.3.1 reads the constructor's signature and adapts. I checked it against the actual upstream file on both shapes: identical anchor and audio window coordinates to the last bit, so nothing about your renders changes.

One correction on the commit labels, since they will mislead whoever finds this thread next. I pulled comfy/ldm/minimax/model.py at both:

7fe8a61 is master, not a release. It has the new layout.
ddbaa87 is byte-identical to the file at tag v0.33.1, which is the newest tag. It has the old layout.

So the new layout has not landed in any tagged ComfyUI yet. If you hit this, you are running a git-pull install tracking master, which is a perfectly reasonable thing to be doing, it just means the version numbers do not line up the way they look.

@azra1l rolling back was the right call at the time and is no longer needed.

Not SolAttn. @javawock7618, reasonable theory given #8, but the error there had a different cause. That said, the general point stands: only one pack can own that constructor, and if two are installed the one that loses will say so in the log.


> **Comment by azra1l (2026-08-16T21:31:57Z):**

i am getting wierd results now, whenever i go beyond the third generation, prompt adherence goes to shit.
it mostly just refuses to zoom as prompted, which it also did before, but that was likely due to too high resolution, aparently a known flaw with H3.
i am now generating at 0.4mp, and when i try to zoom, with some sort of context wired in, it will just refuse to do so. when i remove the context and use the current prompt to generate a new chain, it zooms just fine.
can anyone reproduce this?


> **Comment by NikoDemon80 (2026-08-16T22:50:01Z):**

Is the zoom supposed to happen inside the context window? First 22 frames or whatever you have it set at.


> **Comment by azra1l (2026-08-16T23:47:42Z):**

Hi, actually yes. Is that not possible?


> **Comment by azra1l (2026-08-17T02:27:51Z):**

I tried pointing the zoom-in at 00:01.500 - doesn't work either.


> **Comment by NikoDemon80 (2026-08-17T16:44:24Z):**

If you are using latent_context at 22 frames, the model is going to have a hard time forcing it to zoom inside the context window because those frames are basically set in stone. Same reason why a First Frame won't work if using Motion Context, they compete for conditioning. If it still will not work at frame 36, that's a different problem. Not sure what would cause that tbh.


> **Comment by azra1l (2026-08-18T04:57:45Z):**

Hi, i have no idea why, but the prompts that previously refused to work are working now.
I didn't change anything on the setup. Exact same workflow, seed, prompt. 


> **Comment by NikoDemon80 (2026-08-19T22:11:18Z):**

**Upstream status, 2026-08-19**

Two ComfyUI changes are on master. Neither is in a tagged release yet.

**PR #15439** rewrote the H3 packed layout.

- Keyframes can anchor at any frame. Stock used to allow first and last only.
- Keyframes can carry audio and multi-frame clips, not just a single image.
- Keyframe anchors now start after the reference blocks instead of at the text length. That is the reference compensation this pack was doing by hand.
- `PackedLayout.__init__` dropped `frame_count`.
- Ships a stock node, Add Guide for MiniMax H3.

**PR #15375** added per-token noise masks for H3.

- A latent can carry a mask per stream. Masked rows run at their own strength.
- Preserved rows get injected at cond strength instead of being re-noised at every step.
- No node in core builds the latent mask afaik. This is plumbing without a supply line.

Newest tag is v0.33.2, cut 2026-08-17. It has neither change.

**What this means for the H3-Motion-Context pack**

0.3.1 is unaffected. It reads the layout constructor signature and works on every tagged release, old layout or new. Nothing to do.

0.4.0 is written and waiting. It targets the new layout and drops the patches, because core does that work itself now. It ships when the layout lands in a tag. I don't want to build against master just to get something out that would require people moving to master to use it.

Masked chaining is not coming right away and depending on how testing goes, may not come at all. Whether a masked prefix reads as well as pinned conditioning rows is an open question. I will test that with renders before I write anything. Currently, nothing in core builds the mask, and nothing writes the previous clip's tail into the target latent. This will be the focus of H3-Motion-Context moving forward, filling in those gaps until there are no more gaps to fill. 

For now, hang out on v0.3.1, it's the release that works no matter what.



> **Comment by NikoDemon80 (2026-08-26T18:43:35Z):**

Upstream status, 2026-08-26

Both PR's merged to tagged.
v0.4.0 was ready and waiting. 
Tested working on my end. 
Pushed and live right now. 
Open a new issue if anything comes up.

Thanks everyone for hanging in there and especially those who contributed!



### #11 [closed] ISSUE: Multi-prompt selector
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/11  by Filipus24 2026-08-13T20:58:49Z

Hey Niko,

Not sure how to go about communicating with you, so I figured opening an issue is the best way, and then you can delete it if you find it useless.

I built a multi-prompt selector node that allows you to store all the prompts for all the various clips inside a single node. 

You select the _prompt_title_ from a drop-down, and the content associated with that title becomes the _selected_prompt_ output that can then be fed to the Ref2vid node. 

Simultaneously, my node's _clip_index_ output can be linked to the _clip_index_ property of the Motion Context Save node, and the _previous_clip_index_ output can be linked to the _clip_index_ input of the Motion Context Load node, making sure the proper indices are automatically set.

Not rocket science, but I find it quite useful because when using it, I don't have to hunt down the various properties I have to change whenever I generate multiple clips.

I think it would make a nice addition to your node pack, but I've never contributed to an open-source project like this, hence why I'm writing to you here.

Cheers,

Fil

[fil_prompt_selector.zip](https://github.com/user-attachments/files/31045194/fil_prompt_selector.zip)

> **Comment by NikoDemon80 (2026-08-14T01:56:12Z):**

My focus is on providing a backbone for others to take the project further. I am intentionally keeping the project light. It is basically as mathematically correct as it can be at this point. I really do not want to roll in anything else. You can fork it and add that to your fork though!

Here is a list of fixes you can pass on to the agent that helped you create this:

Blocking

-Dropdown is empty after a workflow reload. rebuildDropdown() runs in onNodeCreated, which fires before configure() restores the saved widget values. The prompts textarea is still empty at that point, so the value list stays ["Prompt 1"] until the user edits the text. Rebuild on onConfigure as well, or defer with a microtask.
-Title miss silently returns prompt 1. If selected_title matches nothing, select_prompt falls through to parsed[0]. That renders the wrong prompt and reports clip index 1, so anything downstream writes clip 1's slot. Should raise instead.
-Empty library returns ("", 0, 0). Index 0 is not "none" downstream, it is auto mode. Empty prompt plus a mode change, both silent. Should raise.
-previous_clip_index is 0 for the first prompt. There is no clip before clip 1, but 0 means "newest file in the folder" on a Load node, so it loads a stale latent from a previous session. Needs to be an error or an unconnected state, not 0.

Should fix

-Text before the first ||| is discarded. Those lines accumulate while current_title is None and are dropped. Anyone putting a shared prefix or notes at the top loses them with no warning.
-Duplicate titles are unreachable. Python takes the first exact match, so a second prompt with the same title can be listed in the dropdown but never selected. Either reject duplicates or match on index.
-DESCRIPTION is unfinished and wrong in places. It ends mid-sentence with a dangling "The". One line says "previous clip_index is the ordinal position" when it means clip_index, and the next line contradicts it. The literal \n escapes inside the triple-quoted string produce doubled blank lines. The worked example is also mismatched: the body under "Bird introduction" describes the cat sitting down.

Worth knowing

-The hidden-widget pattern is version-fragile. hidden = true plus computeSize = () => [0,0] is the old LiteGraph approach. Newer frontends handle widget.hidden vs widget.options.hidden differently, so this may quietly stop collapsing. serialize = true on the same widget is a no-op, it is already true.
-The JS-added combo serializes as a third widget value. It lands after the two declared widgets so positional restore works today, but the saved workflow now carries a duplicate of selected_title under another key. Any widget reordering turns that into a mis-mapped restore. -Setting serialize = false on the combo and driving it purely from the backing widget is cleaner.
Mixed tabs and spaces in prompt_selector.js lines 36 to 56.

One thing that is not a bug: linking the INT outputs to a Save or Load node needs the target's clip_index widget converted to an input first. Worth a line in the docs so people do not think the node is broken.


> **Comment by Filipus24 (2026-08-14T02:51:54Z):**

Wow... Thanks for all of that, you must have been one hell of a QA in a past life (or you are one right now!).

I wasn't planning on developing the node any further, it's just something I threw together this afternoon, thinking you might have some use for it. But I understand perfectly your point about wanting to focus mainly on the math side of your project, I can only imagine it requires a fair amount of concentration and skill.

But your comments kind of encouraged me to look a bit more into the architecture of nodes, so I might just make those changes myself after all!

Thanks for taking the time, I really appreciate it.


> **Comment by NikoDemon80 (2026-08-14T03:01:07Z):**

No problem! Thanks for the suggestion.


> **Comment by Filipus24 (2026-08-16T15:28:45Z):**

@NikoDemon80: just to let you know I followed your advice and pretty much rewrote the Node.
- Restore of saved values is deferred to a microtask in both onNodeCreated and Configure.
- Invalid titles raise an error
- Text before the first header is considered a prompt without a header, which is valid.
- Duplicate titles are not allowed
- Comments can be embedded in prompts
- Default duration is now a visible input instead of hard-coded to 10)
- No more hidden-widget. The selected_title input is now of type COMBO, which eliminates the need for the JS code to generate a different widget. This also eliminates the serialization in the previous version
- Documentation completely reworked in markdown format

What I did not touch, however, are the clip_index and previous_clip_index property because I want to use the same node for the first clip as well as for the subsequent clips. So for the first clip, the user still has to bypass the Motion Context nodes (except for the Motion Context Save) just as instructed in your workflow. It would be nice if the H3 Motion Context, the Trim and the Load Latent would act as passthroughs when no Latent file is specified, but until that happens, I think it's fine to have users bypass the nodes... since it's what they have to do with a standard Prompt node anyway.

Thanks for all your suggestions!


> **Comment by Filipus24 (2026-08-19T13:03:02Z):**

I have now published a repo with the node. Some of your users may find it helpful. 
https://github.com/Filipus24/fil-promptselector



### #10 [closed] ISSUE: Is it possible to add new references with each clip?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/10  by pranav9012 2026-08-13T13:38:26Z

I have 3 images and 4 videos. My hardware is RTX4080 mobile + 32GB RAM. If I add all these references at the start, the generation time and memory usage increases significantly. My PC was perpetually stuck on "Minimax H3Ref2image node". Is it possible to add references as I generate the clips?

For example: the first 10 seconds only uses Image1 and video 1. Next 10 uses image 2 and 3 and audio1. To chain prompts, the guide told us to explain the previous segment's end in the prompt and then continue the prompt of next 10 seconds. The way same, can I load image 1 and video 1 first, then change index to 2 and then load the references needed for 2nd clip?


Thanks

> **Comment by NikoDemon80 (2026-08-13T16:41:21Z):**

The generation time would blow up even if you aren't using Motion Context. That's just the nature of using several references in H3. You should be able to do the things you're asking about, but I have not tested all scenarios. Is this an actual issue, or just a question?


> **Comment by pranav9012 (2026-08-14T14:50:19Z):**

It was a question. In theory, it should be possible since the workflow saves previous latent. I was trying to see if it works and if it does, how well? Yesterday, I couldn't run minimax H3 even with one video reference. Turns out, the video bit rate was too big. I will try the above and will    update.

Thanks



### #9 [closed] ISSUE: Severe cumulative color shift/darkening during looped continuous generation
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/9  by SG0102 2026-08-13T09:45:42Z

Hi,

First of all, thank you for creating the amazing ComfyUI-H3-Motion-Context plugin!

I am currently trying to generate longer videos through a looped workflow (using the generated output as the input for the next segment). However, I am experiencing a severe cumulative color shift issue.

The Problem:

The first 1 or 2 generated segments look perfectly fine.

But as I continue the loop (3rd segment and beyond), the colors degrade significantly. The video gets progressively darker, and the contrast becomes extremely high.

It seems like the VAE encode/decode process is accumulating color errors with every new sampling cycle.

My Question:
How can I avoid this cumulative color degradation during multiple continuous generations? Is there a recommended workflow, parameter setting, or a trick to anchor the colors so they don't drift and darken over time?

Thanks for your time and help!

> **Comment by NikoDemon80 (2026-08-13T16:36:00Z):**

Are you chaining clips using an already processed video or the latent from the previous clip? The context_frames and context_audio sockets are for inputting an existing clip's audio and video, where context_latent is for the previous clip's latent. You can use one or the other, but context_latent is better. But even when using context_latent, the textures compound 1.3x over every clip, which makes the first few clips unnoticeable, but by clip 6 it becomes obvious. It's also spelled out in the README. It's not an H3 Motion Context thing, it's an MiniMax H3 thing. Nothing I can do about it, unfortunately. I would also avoid using any Turbo LoRA or Spectrum.



### #8 [closed] ISSUE: h3_motion_context: the layout patch could not be applied, so interior anchors would be rejected by ComfyUI
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/8  by azra1l 2026-08-13T04:57:11Z

when i try to continue from an existing video, i get this error on the H3 Motion Context node:


```
[WARNING] h3_motion_context: another pack has already patched H3's layout (PackedLayout.__init__ now comes from 'C:\\Tools\\ComfyUI\\ComfyUI\\custom_nodes\\ComfyUI-SolAttn_triton._morton_h3'). Several ComfyUI packs lift the same first/last keyframe restriction independently and they cannot both own it, so this one is refusing rather than wrapping a wrapper and producing joins neither pack intended. Disable one of them and restart.
[ERROR] !!! Exception during processing !!! h3_motion_context: the layout patch could not be applied, so interior anchors would be rejected by ComfyUI. The reason was logged just above this error.
[ERROR] Traceback (most recent call last):
  File "C:\Tools\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Tools\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Tools\ComfyUI\ComfyUI\custom_nodes\comfyui-lora-manager\py\metadata_collector\metadata_hook.py", line 177, in async_map_node_over_list_with_metadata
    results = await original_map_node_over_list(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Tools\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "C:\Tools\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "C:\Tools\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\nodes.py", line 421, in apply
    _ensure_layout_patch()
  File "C:\Tools\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\nodes.py", line 80, in _ensure_layout_patch
    raise RuntimeError(
RuntimeError: h3_motion_context: the layout patch could not be applied, so interior anchors would be rejected by ComfyUI. The reason was logged just above this error.

```

i just reproduced the exact same error on my machine with your example workflow. 

It used to work before. I have no idea what is causing this all of sudden. I tried disable sol/sage attention, i tried disable spectrum, i tried disable turbo lora. Nothing helps.

Did a recent comfyui update break this node by any chance? I am running 0.32.0-6

> **Comment by NikoDemon80 (2026-08-13T05:04:40Z):**

This means you have more than one H3 node pack installed that applies the same run-time patch this pack created. Only one node pack can claim the patch, and the first one to load, wins. If you have H3 Contex Loop installed or something like that, they use the same patch. You'll have to choose which one you want installed. Hope that helps!


> **Comment by azra1l (2026-08-13T10:06:14Z):**

Ah, i see. I also installed Context Loop and removed it again for now, but the error persists.


As per log entry

`PackedLayout.__init__ now comes from 'C:\\Tools\\ComfyUI\\ComfyUI\\custom_nodes\\ComfyUI-SolAttn_triton._morton_h3'`

that means your node is also incompatible with the ComfyUI-SolAttn_triton node? That seems kinda odd to me...


> **Comment by azra1l (2026-08-13T12:59:08Z):**

I removed the Sol Attention node, now the problem is gone.

Why does your node conflict with Sol Attention all of sudden?
It definetly worked before with Sol Attention.


> **Comment by NikoDemon80 (2026-08-13T16:44:08Z):**

I have no clue, lol. They may have updated. It doesn't make sense that it would use the same runtime patch, though. Glad it works for you!



### #7 [closed] PR: Fix the audio overhang guard: H3 rounds the grid to nearest, not up
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/pull/7  by ChimeraWerks 2026-08-11T07:44:41Z

## The bug

`_audio_tail_from_latent` accepts `0.0 <= overhang < 1.0`, on the assumption that H3 allocates its audio grid upward from the pixel length. Core rounds to the **nearest** 40 Hz step instead.

So any raw length where `frames % 3 == 2` produces an overhang of `-1/3`, trips the guard, warns, and zeroes the compensation:

| frames | wanted steps | allocated | overhang | old guard |
| --- | --- | --- | --- | --- |
| 124 | 206.67 | 207 | +0.33 | accepted |
| 260 | 433.33 | 433 | -0.33 | **rejected** |
| 311 | 518.33 | 518 | -0.33 | **rejected** |

Roughly a third of legal clip lengths land on the negative side. On mine it fired on every clip of a three clip chain:

```
h3_motion_context: context_latent audio grid is unexpected (433 steps for 260 frames); assuming no overhang.
```

## It costs a whole audio step, not a third of one

This is the part that surprised me. The lost compensation is 8.3 ms, but the grid snap immediately after it amplifies the error.

Measured on a 260 frame context clip with a 22 frame window, driving `apply()` through the repo's own mock harness and comparing real behaviour against a simulated fail-open:

```
pre-fix : end_frame 22.2000   audio-coord 37.0000
fixed   : end_frame 21.6000   audio-coord 36.0000
delta   : -0.6 frames = -25.00 ms
```

Zeroing the `-1/3` overhang moves the end coordinate from 36.33 to 36.67, and `round()` then lands on **37 instead of 36**. The pinned window ends up on the wrong audio row entirely, a full step late, which is the direction and magnitude of the late offset I was hearing at joins.

## The change

1. **Signed band.** `-0.5 < overhang < 0.5` in `_audio_tail_from_latent`. Still fails open on a genuinely unexpected grid, now accepts both sides of nearest rounding. Docstring updated to state the rounding rule with both worked examples, and one inline comment in the `timeline` branch that repeated the old "rounds up" claim.
2. **Exact-count tail pad.** `MiniMaxH3MotionContextTrim`'s `match_tail` path zero-pads a short tail to the exact sample count rather than warning and leaving it, so a fractional step shortage cannot accumulate down a chain. This needs `torch`, which `nodes.py` did not previously import.

Both are ports from [ethanfel/ComfyUI-MiniMaxH3-Contex-Loop](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop), which reached the same fix independently. Credit there, not here.

## Verification

`tests/_node_smoke_test.py` and `tests/_payload_gate_test.py` both pass unmodified on this branch. The smoke test still asserts `end_frame 22.2000` on the 124 frame case, so the positive overhang path is unchanged.

The negative case is not covered by the existing tests, since they only exercise 124 frames. I verified it out of band with the same harness. Happy to fold a 260 frame regression case into `_node_smoke_test.py` if you want it in this PR.

https://claude.ai/code/session_01MyKPmNRJ4KJ4zfxtUm1nh7


> **Comment by NikoDemon80 (2026-08-11T19:39:23Z):**

Two things. This repo isn't a fork of ethanfel's, his is a fork of this one. Doesn't change the credit, his fix landed first and I'll say so.

Second, you're underselling your own part. The port is one line. The measurement showing it costs a whole step rather than a third of one, and the note that the smoke test only covers 124 frames so it couldn't have caught this, are yours. Both go in the release notes.


> **Comment by NikoDemon80 (2026-08-11T19:52:12Z):**

Merging this. Good find, and the measurement is the useful part.

Taking you up on the regression case. I've written it out to cover all three residues rather than just 260, since 124 is the one length where rounding up and rounding to nearest agree. Landing that as a follow-up commit.


> **Comment by NikoDemon80 (2026-08-11T21:46:16Z):**

Merged. Verified on real renders: a 260 frame chain now reports a lag of -8.3 ms rather than the -25 ms the failed-open path would give, which matches your amplification analysis.

Two things that came out of testing it. The remaining 8.3 ms is not the overhang, it is the grid snap after the compensation, and it is exactly a third of an audio step. It turns out to be zero for clip lengths where frames % 3 == 1 and 8.3 ms either way for the other two residues, so the fix is complete and that residual is a separate documented tradeoff.

I extended the regression case to cover all three residues rather than just 260. 124 is the one length where rounding up and rounding to nearest agree, so a test on it alone can never fail. Also found that the existing grid check was computing its simulated overhang with ceil, which means it was exercising a value the model never produces.



### #5 [closed] ISSUE: Decoding saved H3 latents
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/5  by hum-ma 2026-08-09T21:59:31Z

As you have noted in the README, the stock save/load latent nodes don't work with H3 (I created a ComfyUI issue about it a few days ago: https://github.com/Comfy-Org/ComfyUI/issues/15254) but also that loading a saved latent with your node is only for `context_latent`, not for decoding. What would it take to re-create the nested tensors from the saved latent into the same format in which they come from the sampler, so that they could be decoded? Or would they have to be saved differently in the first place?

My use case is that with a slow computer with which VAE decoding takes several minutes I rather first decode a video with TAE, then only if it seems as it should I decode the same again with the real VAE. With other models I can save all the latents during a batch run, look at the TAE-decoded "previews" later and pick the latents to decode separately, but with H3 it's currently only possible if I fix the seed for each and re-decode the video immediately after the sampling+TAE decode is done, while the latent is still retained in memory with the same workflow.

> **Comment by NikoDemon80 (2026-08-09T23:02:05Z):**

Nothing is lost in the save. Both streams go into the file as plain tensors, so everything needed to decode is already on disk. The problem is only on the way back out: the load node hands you a plain list, and the sampler emits a nested type that the VAE decode expects.

So it's a load-side fix, not a format change. In principle it's small.

Two things make it less trivial than it sounds. The pack only ever consumes that nested type, it never builds one, so constructing it means depending on another piece of ComfyUI internals that can move under me. And unbind() may drop the batch dimension on the way in, so what's in the file might not be the shape the decode wants back.

The plain list is also deliberate. It's there so wiring it into a decode fails loudly instead of producing garbage. If the output decodes correctly that guardrail has no job left, but I'd want to be sure it really does before removing it.

I'm not promising a date. If #15254 gets fixed, stock Save/Load Latent covers your use case and this doesn't need to exist in my pack at all, which is the better outcome. If it stalls, I'll take another look.


> **Comment by hum-ma (2026-08-10T08:49:10Z):**

Another user in the ComfyUI issue found out that LTXVSeparateAVLatent can be used to split H3 latents and then the separate ones can be saved and loaded for decoding normally so it's all good now.



### #4 [closed] ISSUE: Chain Multiple H3 Motion Context Nodes in a row
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/4  by p80ric 2026-08-09T19:25:37Z

Hi, first of all, thx for the cool nodes. This is what i was looking for :)

I have the following issue: tried to chain multiple subworkflows together to have e.g. 4 consecutive generations all in one workflow... Problem is with the Second H3 Motion Context Node i get the following error:

```
[ERROR] !!! Exception during processing !!! h3_motion_context: audio timeline placement needs exactly one reference marked with motion_context_audio_end_frame; the layout has 4 references and 2 marked. If this appeared during startup, check for more than one H3 Motion Context folder in custom_nodes.
[ERROR] Traceback (most recent call last):
  File "E:\ComfyUI\ComfyUI\execution.py", line 545, in execute
    output_data, output_ui, has_subgraph, has_pending_tasks = await get_output_data(prompt_id, unique_id, obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\execution.py", line 344, in get_output_data
    return_values = await _async_map_node_over_list(prompt_id, unique_id, obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb, v3_data=v3_data)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\execution.py", line 318, in _async_map_node_over_list
    await process_inputs(input_dict, i)
  File "E:\ComfyUI\ComfyUI\execution.py", line 306, in process_inputs
    result = f(**inputs)
             ^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy_api\internal\__init__.py", line 149, in wrapped_func
    return method(locked_class, **inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy_api\latest\_io.py", line 1935, in EXECUTE_NORMALIZED
    to_return = cls.execute(*args, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy_extras\nodes_custom_sampler.py", line 1049, in execute
    samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\samplers.py", line 1335, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\custom_nodes\comfyui-spectrum-minimax-h3\comfyui_spectrum_h3\sampling.py", line 247, in outer_sample_wrapper
    return executor(
           ^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 114, in execute
    return self.wrappers[self.idx](self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-KJNodes\nodes\preview_override_node.py", line 685, in __call__
    return executor(noise, latent_image, sampler, sigmas, denoise_mask, new_callback, disable_pbar, seed, latent_shapes=latent_shapes)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 106, in __call__
    return new_executor.execute(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\patcher_extension.py", line 113, in execute
    return self.original(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\samplers.py", line 1262, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\samplers.py", line 1226, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\samplers.py", line 1048, in process_conds
    conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\samplers.py", line 953, in encode_model_conds
    out = model_function(**params)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\patch_payload.py", line 70, in _patched_extra_conds
    out = _orig_extra_conds(self, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\comfy\model_base.py", line 2176, in extra_conds
    payload["layout"] = comfy.ldm.minimax.model.PackedLayout(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\patch_layout.py", line 280, in _patched_init
    _fixup_audio(self, text_len, refs)
  File "E:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context\patch_layout.py", line 228, in _fixup_audio
    raise RuntimeError(
RuntimeError: h3_motion_context: audio timeline placement needs exactly one reference marked with motion_context_audio_end_frame; the layout has 4 references and 2 marked. If this appeared during startup, check for more than one H3 Motion Context folder in custom_nodes.
```
Seems like its designed to be there only once like in your example workflows. I think with chaining it could be much more flexible... maybe add a node to clear the pinning? here my workflow that i tried to have:

<img width="1238" height="655" alt="Image" src="https://github.com/user-attachments/assets/1f4ab489-52a9-41f0-a4f4-53cad1a87b48" />

<img width="1703" height="725" alt="Image" src="https://github.com/user-attachments/assets/c38f72da-f999-4664-846c-55b91ba7ed99" />
(edit: ignore the save in the sub workflow i was just hoping it would clear the pinning...)
would be glad if you could have a look....

and keep up the good work :)

> **Comment by NikoDemon80 (2026-08-09T20:52:52Z):**

The error is two Motion Context nodes in the same conditioning chain. Each one pins audio onto the clip's timeline, and only one window can
go there, so it refuses. ComfyUI can't pass a latent from one run to the next, so it goes through disk. Four clips means queueing four times, moving the index numbers forward each run: Load 1 / Save 2, then Load 2 / Save 3, then Load 3 / Save 4. One Motion Context node
the whole way through.

If you want the whole chain to run in one queue, ethanfel has a fork with loop nodes that do that. Don't install both, we patch the same ComfyUI code and only one can own it. https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop

I'm also building an all-in-one node that handles the full chain itself, prompts per segment included. Not out yet.


> **Comment by mjzcng (2026-08-13T02:56:53Z):**

A follow-up to the Contex Loop link above: I have published a modified public fork with a prompt-boundary disk relay mode: [ComfyUI-MiniMaxH3-Disk-Relay](https://github.com/mjzcng/ComfyUI-MiniMaxH3-Disk-Relay).

Disclosure: I maintain this fork; NikoDemon80 and Ethanfel authorship/history remain credited under GPL-3.0. It still presents one editable multi-scene workflow, but each non-final scene now atomically saves its MP4/audio/latent checkpoint, ends the current ComfyUI task, unloads models and clears the finished execution cache, then lets the server queue a fresh resume task from disk. It can run unattended or pause for approve/retry/reroll after each scene.

That avoids stacking multiple Motion Context pins in one conditioning chain and also targets RAM/swap accumulation across a long recursive prompt. It does **not** lower the peak required by one scene. The repository includes a copyright-safe fourteen-scene (about 60 s) Ref2VA demo. Because it exposes the same Contex Loop node IDs as Ethanfel pack, install it as a replacement for that pack, not alongside it; it can still coexist with this original Motion Context pack.




### #3 [closed] ISSUE: My fork.
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/3  by ethanfel 2026-08-09T17:37:07Z

Hey, I really took a liking to your node and Seitanism Ref2Va code. My fork add extensive loop support but it's a huge PR and IDK if you are interested to bring theses features on your main since it's not your own code. 

So here I am asking if I should do a PR or just keep it all on my fork, either is fine for me, no worries 

> **Comment by NikoDemon80 (2026-08-09T18:00:09Z):**

Thanks for asking first.

I'll say no to the merge, but not because of your code.

Motion Context is meant to stay small: the patch plus four nodes. I
just cut the main node from six settings to two. Eight loop nodes and
a UI makes it a different pack, and I would be maintaining a lot of
code I did not write. 

Keep doing the good work! It's fine with me if our repos diverge.

I am also building automated chaining in a separate all-in-one node.
So, we would have two versions of the same thing in one repo.

Also: I pushed 0.2.0 today. Ref2Va support is upstream now. I wrote my
own version, but you and seitanism were first and you are both
credited in the README and release notes.


> **Comment by ethanfel (2026-08-09T18:51:39Z):**

Yeah I had a feeling about this haha, all good, keep up the good work 


> **Comment by seitanism (2026-08-09T19:43:51Z):**

awesome repo niko!


> **Comment by NikoDemon80 (2026-08-09T19:53:02Z):**

Hey, thanks! Good job on the patch fix! You helped me get passed something I didn't think was possible. 



### #2 [closed] ISSUE: boot error
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/2  by vigee88 2026-08-09T14:47:53Z

[WARNING] h3_motion_context: self-test failed (h3_motion_context: audio timeline placement supports exactly one ref block, the marked audio ref; layout has 4 refs, 1 marked.), patch not applied. Interior keyframe anchors unavailable.
[ERROR] Motion Context disabled: h3_motion_context: audio timeline placement supports exactly one ref block, the marked audio ref; layout has 4 refs, 1 marked.; H3 payload patch is inactive

> **Comment by NikoDemon80 (2026-08-09T15:33:22Z):**

The only way to get this error message is to have two different version of the repo installed at the same time. Do you also have https://github.com/ethanfel/ComfyUI-H3-Motion-Context installed? The patches will try to wrap around each other cause an error. You would need to remove one or the other. Let me know if that helps. I am releasing a new version shortly which includes ref2va support and improved quality across seams.


> **Comment by vigee88 (2026-08-09T15:42:36Z):**

no，i have only one ComfyUI-H3-Motion-Context

<img width="1961" height="1303" alt="Image" src="https://github.com/user-attachments/assets/247f5130-071f-493c-a7a3-3011a9d58d88" />


> **Comment by vigee88 (2026-08-09T15:48:32Z):**

There are quite a lot of installed nodes. Has there been any conflict between this  and other nodes? I don't understand either. Many of these were installed simply because they were needed in some workflows.


> **Comment by NikoDemon80 (2026-08-09T16:12:30Z):**

No conflicts as far as I know. None reported anyway. Still investigating.


> **Comment by NikoDemon80 (2026-08-09T16:29:26Z):**

I'm thinking at least one of the other H3 node packs you have installed is applying the same patch as my node pack. The most likely culprit is H3-Motion-Director. If you disable it and the error goes away, that's it. Not much I can do about it, unfortunately. Sorry!


> **Comment by vigee88 (2026-08-09T17:16:33Z):**

i disabled  _ComfyUI-MiniMax-H3-Motion-Director，the error message still appears。
but，，， the node seems to be functioning properly. I can also open your workflow.



[WARNING] h3_motion_context: self-test failed (h3_motion_context: audio timeline placement supports exactly one ref block, the marked audio ref; layout has 4 refs, 1 marked.), patch not applied. Interior keyframe anchors unavailable.
[ERROR] Motion Context disabled: h3_motion_context: audio timeline placement supports exactly one ref block, the marked audio ref; layout has 4 refs, 1 marked.; H3 payload patch is inactive
[INFO] MiniMax H3 Motion Director HTTP routes registered
[INFO]
Import times for custom nodes:
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\TE_MAN_VIP_PATCH
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\websocket_image_save.py
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_RefineAnything
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-PainterQwenImageEdit
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Comfyui-PainterFluxImageEdit
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-krea2-controlnet
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Turbo
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3_LatentUpscaler
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_ADV_CLIP_emb
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-BSS_FLSampler
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Comfyui-LatentUtils
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-krea2-conditioning
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-krea2edit
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Krea2-Ostris-Edit
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Krea2T-Enhancer
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-ReservedVRAM
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-mxToolkit
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Detail-Daemon
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_Custom_Switch
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-TT-Resolution_selector-Node
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-WanMoeKSampler
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Krea2-StyleTransfer
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Comfyui-QwenEditUtils
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyLiterals
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\cg-use-everywhere
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Inpaint-CropAndStitch
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_UniBlockSwap
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-EditUtils
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Latent_Nodes
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-qwenmultiangle
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui_scail2_multi_cond
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SCAIL-Pose
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-minimax-h3-blockcache-T8
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_SLK_joy_caption_two
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SCAIL2-Easy
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_StringOps
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-Easy
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Comfyui_TTP_Toolset
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyUI-llama-TE
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-EulerDiscreteScheduler
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-GGUF
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-minimax-h3-prompt-enhancer-T8
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Batch-Process
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfy-text-stats
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-YUAN
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-PromptRelay
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_IPAdapter_plus
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\LanPaint
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Florence2
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MelBandRoFormer
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-various
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_essentials
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyMath
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-QwenVL
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\FastVideo
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\WhatDreamsCost-ComfyUI
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Yuan-Tool
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_Patches_ll
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-WhiteJade-Image
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-yaser-nodes
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\comfyui-minimax-h3-audio-T8
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-LCS
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SolAttn_triton
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Frame-Interpolation
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\Comfyui_LG_Tools
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VL-Nodes
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Spectrum-MiniMax-H3
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-UniversalBlockSwap
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_UltimateSDUpscale
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-prompt-formula
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Whisper
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SHOUWANG-ToolBox
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-bleh
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\rgthree-comfy
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI_Comfyroll_CustomNodes
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-KJNodes
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Inspire-Pack
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-H3-Motion-Context
[INFO]    0.0 seconds: D:\ComfyUI\ComfyUI\custom_nodes\_ComfyUI-MiniMax-H3-Motion-Director




> **Comment by vigee88 (2026-08-09T17:17:34Z):**

I plan to test it tomorrow. Thank you very much for your reply.

<img width="2148" height="1205" alt="Image" src="https://github.com/user-attachments/assets/ca47757f-53e8-4f82-a045-df027bdf5f16" />


> **Comment by NikoDemon80 (2026-08-09T17:21:34Z):**

Hey, that's progress! There are several other H3 node packs that could be using the same patch. It's bound to be one of them. Glad it's working for you.



### #1 [closed] ISSUE: Ref2va workflow ?
URL: https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context/issues/1  by alexfredo 2026-08-09T12:33:03Z

Hi can you add a Ref2va workflow? thanks!

> **Comment by NikoDemon80 (2026-08-09T15:33:50Z):**

Coming with the update in a little while! Hold tight!


> **Comment by HWiese1980 (2026-08-12T14:36:56Z):**

Is there a way to somehow preserve voice-timbres of two speakers where one doesn't speak for several clips with Motion Context as long as ref2va isn't yet implemented? I'm currently handing a clip in where both voices speak with a fairly large audio context, but that apparently kills audio continuity... 

