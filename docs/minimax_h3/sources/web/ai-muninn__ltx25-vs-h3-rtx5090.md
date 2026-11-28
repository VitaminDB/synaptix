# Source: https://ai-muninn.com/en/blog/ltx25-vs-minimax-h3-rtx5090

Fetched: 2026-11-28 (curl + w3m text dump)

---

  • LTX names the speaker in the sentence, H3 tags them and keeps identity elsewhere
  • Two speakers: LTX 29.21 s at CER 9.52%, H3 94.82 s at CER 0%
  • Four signals say both models really alternate, and none of them rules out overlap
  • LTX drops 「这碗」 and padding silence does not bring it back
  • Neither model leaves headroom: 871 and 796 MiB free
  • Deep dive: where the VRAM peak actually sits
  • LTX's VRAM peak sits in stage-2 sampling, so smaller decode tiles miss it
  • NVFP4 is 2.59 GiB smaller on disk and 72 MiB larger at the peak
  • Which one earns the slot
  • Not measured yet
  • References

TL;DR

Same RTX 5090, same Simplified Chinese line, same ComfyUI: LTX-2.5 finishes in 28.67 s, MiniMax-H3 in 81.22 s, and both come back at ASR character error rate 0%. The gap is per-pass cost, not step
count, since LTX runs 11 forward passes and H3 runs 9. LTX also lands at −10.4 LUFS where H3 sits at −30 to −34, so H3 needs a gain stage before anything ships. On two speakers LTX dropped two
characters (CER 9.52%) while H3 stayed word for word. Neither leaves headroom: peaks of 31,736 and 31,811 MiB on a 32,607 MiB card. One script, one scene, n=1.

First half is MiniMax-H3, second half is LTX-2.5, same two-speaker script, both loudness-matched to −16 LUFS before concatenation.

Preface

Two kitchens, one bowl of soup. The fast one puts it in front of you in four minutes and gets a word of the order wrong. The slow one takes twelve and reads your order back to you exactly as you said
it.

Part 1 of this series got MiniMax-H3 onto a single RTX 5090. Part 6 put two characters into one shot with Ref2VA. Lightricks then shipped LTX-2.5 with synchronised audio, which makes it the first
local model that competes with H3 on the thing H3 was interesting for: dialogue that comes out of the same forward pass as the picture.

So I ran both on the same card, with the same Chinese script, and measured what each one gives up.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

28.67 s vs 81.22 s for the same line

One RTX 5090, 32,607 MiB total. ComfyUI. Both models local, no API in the path.

The test line is single-speaker, Simplified Chinese, 25 characters, about 5.17 s of speech. Identical text into both models:

今天的汤头特别浓，先喝一口暖暖身子，再慢慢告诉我你的故事。

The row that matters below is Whole flow, which is sampling plus load, decode and save:

                           MiniMax-H3       LTX-2.5
Resolution, frames        1344×768, 124 1280×704, 121
Steps                     14 + Spectrum 8 + 3, two-stage
Sampling                  66.40 s       14.31 s
Whole flow                81.22 s       28.67 s
ASR character error rate  0%            0%

Both models transcribe back cleanly, so on this line the machine gate does not separate them. The wall clock does.

One thing that keeps this from being a clean subtraction: the two models want different shapes, and I let each one run at its own native size. LTX's pixel-times-frame load comes to about 85% of H3's.
That is not enough to explain a 3x gap, but it does mean you cannot read the difference as a pure engine comparison.

Top: MiniMax-H3 at 1344×768. Bottom: LTX-2.5 at 1280×704. Same dialogue line, frame grabbed at t=2.0s

Same line, same card. H3 on top, LTX-2.5 below. The two scenes are separately generated, not two renders of one frame.

11 forward passes vs 9, so the gap is per-pass cost

The obvious first guess is that H3 is simply doing more work per clip. It is not, and the step counts say so.

LTX-2.5 runs two stages, 8 steps then 3, so 11 forward passes through the transformer.

H3 runs 14 steps, but Spectrum is on. Spectrum is H3's step-skipping scheduler: it keeps a short history of previous steps and forecasts some of them instead of actually running them. On this
configuration it forecast 5, which leaves 9 real passes.

Eleven vs nine is close enough to call a tie. The 3x wall-clock gap therefore has to live in what one pass costs, not in how many there are. That number turns out to be around 5 s per actual pass for
H3, which I take apart in the deep dive.

H3 comes out 15-20 dB under normal loudness, LTX does not

LUFS is the loudness unit broadcast normalisation uses. Zero is the ceiling and more negative is quieter, so a number like −30 means the file is very quiet in a way a peak meter will not warn you
about.

On this run, H3's clip measured −25.1 LUFS integrated and LTX's measured −10.4. Across the runs I have done, H3's native output sits between −30 and −34 LUFS, which is 15 to 20 dB below normal
content. Anything you publish needs a gain stage in front of it.

LTX at −10.4 is close enough to normal that the step disappears. That is one fewer node in the graph and one fewer thing to forget before you upload.

1344×768 is arithmetic, not preference

Every H3 recipe you will find uses 1344×768, and it looks like a community convention. It is not. It is what you get when you satisfy two constraints at once:

short edge          768          (MiniMax model card default)
16:9 would want     768 × 16/9 = 1365.333…   (not even an integer)
ComfyUI path        both dimensions divisible by 32
1365.333 / 32     = 42.666…      -> round down to 42
42 × 32           = 1344

Worth being precise about what that leaves you with: 1344 / 768 = 1.75, which is 7:4, not 16:9. The 16:9 width at this short edge is not reachable, so 1344 is the nearest legal value below it.

Two independent sources land on the same shape, which is a decent sign the derivation is not just mine: NVIDIA's Sol-Engine benchmark runs H3 at 1344×768 by 124 frames, and FastVideo's H3
distillation was trained at 768×1344 by 124 frames.

Frame counts do not port between the two models

If you try to hold frame count constant across the comparison you will hit an error, because the two models count frames on different grids.

LTX-2.5 requires frames % 8 == 1. So 121 works, since 121 is 15×8+1. 124 is rejected outright.

H3 uses a 17k+5 grid, where 124 is 17×7+5.

The grids do intersect — 73, 209 and 345 satisfy both rules, and 209 is 26×8+1 as well as 12×17+5 — but none of those sit near either model's native working point. I ran each at a legal value close
to 5 seconds and noted the gap instead.

Five files from Lightricks/LTX-2.5, and three ways to download the wrong ones

Everything for LTX-2.5 comes from one repository, Lightricks/LTX-2.5. On-disk sizes:

                                       File                                          Size
diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors  20.03 GiB
text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors          14.32 GiB
vae/ltx-2.5-video-vae-conv-bf16.safetensors                                        1.35 GiB
vae/ltx-2.5-audio-vae-bf16.safetensors                                             0.34 GiB
latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors      0.93 GiB

Three places to get this wrong:

  • Take distilled, not dev. The dev transformer produced melted-looking output at both quantizations I tried. It is not a quality-versus-speed trade in this graph, it is a broken picture.
  • Take the conv video VAE. A third-party measurement on a 5090 puts it 22.5% faster overall, with decode falling from 8.91 s to 4.68 s and peak VRAM essentially flat (31,692 MB to 31,728 MB, up 36
    MB). Free.
  • The latent upscaler is not optional. The official blueprint samples in two stages with LTXVLatentUpsampler at ×2 sitting between them. Skip the file and you have no second stage to run.

The official sampler settings, and the wiring mistake that raises split_with_sizes

These come straight out of the ComfyUI blueprint in Comfy-Org/workflow_templates, and I did not tune any of them:

sampler          euler_ancestral
guider           LTXVDualCFGGuider [1, 1]
stage-1 sigmas   1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0   (8 steps)
stage-2 sigmas   0.85, 0.7250, 0.4219, 0.0                                              (3 steps)
conditioning     LTXVConditioning [24]
decode           VAEDecodeTiled [512, 64, 64, 16] + LTXVAudioVAEDecode
AV latent        LTXVSeparateAVLatent / LTXVConcatAVLatent

The last line is the one that bites. Audio and video travel as a single combined latent and have to be split before decode and concatenated after. If you wire that chain wrong, the error you get is
split_with_sizes, which reads like a tensor-shape problem and is actually a graph-topology problem.

LTX names the speaker in the sentence, H3 tags them and keeps identity elsewhere

The two models take dialogue in shapes that do not convert into each other.

LTX-2.5 has a template that puts the line in quotes and names the language. One caveat first, because I got this wrong on the way in: that template lives in the Dub-It (speech replacement) section of
the docs, and that section opens with "Unlike text-to-video generation". I read it as the general dialogue syntax. It isn't. For text-to-video there is no equally specific official template.

The template:

[Speaker] is speaking [Language/Accent], saying: "[Dialogue]"

The model card lists nine languages including Chinese. Cantonese is not one of them.

H3 uses positional tags, (S1) and (S2), and this is where people lose an afternoon: the tag carries no identity at all. It is a pointer. Who S1 is gets established in the subject description, like
<Subject 1> (S1), a young woman with shoulder-length black hair…, and if you never say who S1 is then the model invents someone. (S1,S2) means both speaking at once.

Two speakers: LTX 29.21 s at CER 9.52%, H3 94.82 s at CER 0%

Same scene, two lines, two voices:

S1  young woman        这碗汤是你煮的吗，闻起来好香。
S2  middle-aged man    是我煮的，趁热喝吧。

Look at the bottom two rows, because that is where the models finally separate:

                              MiniMax-H3   LTX-2.5
Sampling                      67.94 s    14.34 s
Whole flow                    94.82 s    29.21 s
Peak VRAM                     31,811 MiB 31,729 MiB
S1 / S2 character error rate  0% / 0%    15.38% / 0%
Total character error rate    0%         9.52%

LTX stays about 3x faster with a second voice in the scene. It also stops being free.

Four signals say both models really alternate, and none of them rules out overlap

I had read the single-speaker guidance as a model-wide limit, so the first question was whether the second voice was real or whether one voice was doing an impression while both mouths flapped. I
checked four things that fail in different ways:

  • ASR timestamps do not overlap. H3 puts S1 at 0.00-2.40 s and S2 at 3.02-4.88 s. LTX puts S1 at 0.00-1.54 s and S2 at 2.56-4.84 s.
  • Frame-by-frame mouth attribution is correct. On the LTX clip, from 0 to 1.4 s the woman's mouth is moving and the man's is mostly closed; from 2 to 4.8 s the man is speaking and the woman is
    closed.
  • The handover goes quiet. A 20 ms RMS envelope at the switch reads 0.000296 for H3 and 0.000646 for LTX, each about 1.1 to 1.2% of the preceding line's mean level.
  • Pitch medians drop where they should. H3 goes from 242 Hz to 104 Hz between the lines, LTX from 286 Hz to 130 Hz.

Where that stops being evidence: a single mixed-audio envelope cannot separate two speakers, only measure the sum. The transcriber can miss overlapping speech entirely, which is the exact failure I
was testing for. A measurable pause is not the same claim as no overlap in any millisecond. And pitch medians say something about register, never about voice identity. One script, one scene, n=1 for
each model.

LTX drops 「这碗」 and padding silence does not bring it back

The 15.38% on LTX's first line is two characters at the very start, 「这碗」. That position is suspicious, because transcription heads routinely clip the beginning of a file.

So I padded 0.75 s of silence before and after the audio and re-ran ASR. The same two characters went missing. That makes a head-boundary artifact unlikely and points at the generation.

H3 got both lines word for word on the same script.

Neither model leaves headroom: 871 and 796 MiB free

                            Peak    Free on a 32,607 MiB card
LTX-2.5                  31,736 MiB 871 MiB
MiniMax-H3, two speakers 31,811 MiB 796 MiB

Under 900 MiB of slack in both cases, which means neither model is the light one. Third-party reports put LTX-2.5 into the 32 GB wall at 8 seconds of 720p, and my own runs agree that about 10 seconds
is the practical ceiling for a single generation on this card.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Deep dive: where the VRAM peak actually sits

You can use either model without reading any of this. The first act said neither model leaves 900 MiB. This is how that number was measured, and what happened when I tried to push it down.

LTX's VRAM peak sits in stage-2 sampling, so smaller decode tiles miss it

LTX-2.5 leaves 871 MiB free. VAEDecodeTiled runs a 512 tile and is the easiest knob to reach, so the first thing worth knowing is which node actually holds the peak.

The measurement: that run kept 123 VRAM samples and 114 execution events, and the events carry elapsed_s. Line the two timelines up and each node interval gets its own peak.

        Node         Interval peak
CLIPTextEncode       19,032 MiB
UNETLoader           16,632 MiB
stage-1 sampling     31,653 MiB
LTXVLatentUpsampler  31,466 MiB
stage-2 sampling     31,736 MiB
VAEDecodeTiled       30,873 MiB
SaveVideo            27,193 MiB

The peak is in stage-2, with stage-1 only 83 MiB behind it. Both sampling stages sit at the ceiling and decode is below both, so shrinking the tile cannot reach the number that matters.

The same table answers a related question. Text encoding peaks at 19,032 MiB and drops to 16,632 once the transformer loads, which means the text encoder unloads before sampling begins. Swapping it
for a smaller quantization does nothing for the sampling peak.

NVFP4 is 2.59 GiB smaller on disk and 72 MiB larger at the peak

Weights stay resident through sampling, so a smaller checkpoint is the next thing to measure. The same repository ships an NVFP4 variant at 18,721,548,408 bytes, 2.59 GiB below int8-convrot.

That variant used to fail with a dimension mismatch at the sampling stage. The checkpoint's __metadata__ was missing _quantization_metadata, so ComfyUI silently dropped the quantization scale tensors
and fed packed FP4 weights straight into the linear layer. Lightricks says on the HF discussion thread that it is fixed. ⚠️ I cannot independently confirm "re-upload only, no code change" — I
inferred it from the ComfyUI-side issue still being open, not from a file revision diff. Either way, a local copy downloaded early needs checking.

You can check the header with two HTTP range requests instead of pulling the whole 18 GB. A safetensors file starts with an 8-byte little-endian u64 header length followed by that many bytes of JSON,
so before downloading you can pull bytes 0-7 for the length and then that slice of JSON over HTTP range requests. After downloading, the same parse runs on the local file:

import struct, json
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    meta = json.loads(f.read(n)).get('__metadata__', {})
print('HAS _quantization_metadata:', '_quantization_metadata' in meta)

Run against the file currently in the repository: _quantization_metadata present, 7,877 tensors. Downloading today gets the good one.

Then the swap itself — same workflow, same seed, only the checkpoint changed:

               int8-convrot              NVFP4
File size  21,504,034,224 bytes  18,721,548,408 bytes
Sampling   14.307 s              13.547 s
End to end 28.668 s              28.815 s
Peak VRAM  31,736 MiB            31,808 MiB
Headroom   871 MiB               799 MiB
CER        0%                    0%

Sampling got 5.32% faster. End to end is flat, 0.147 s slower. The peak rose by 72 MiB and headroom fell from 871 to 799. I recomputed that from the 149 raw samples rather than taking it from the run
log.

The reason is that disk size and residency are different quantities. Packed FP4 has to be unpacked to a working dtype before it can take part in a matmul, and that unpacked footprint has no fixed
relationship to the bytes saved on disk. Per node, stage-1 sampling is effectively unchanged — 31,653 vs 31,648 — and what climbs is the latent upsampler in the middle, 31,466 up to 31,776.

The takeaway: to save VRAM, find which node holds the peak.

One last thing to flag. This run printed [WARNING] unet unexpected: with 1,176 keys, every one of them ending in .comfy_quant — not a single scale key among them. So it is not the old
dimension-mismatch failure in disguise: the picture is fine, CER is 0%, nothing threw. But that warning line is the only signal ComfyUI gives for a silent drop, so read it rather than scrolling past.
Single run, no repeats, no significance claim.

Which one earns the slot

LTX-2.5 is faster by a margin no tuning will close: 28.67 s vs 81.22 s end to end on the same line. Its Chinese dialogue passes the automated check at CER 0%, and it alternated two speakers in my
run. The single-speaker restriction in the docs applies to Dub-It, not to text-to-video.

MiniMax-H3 gets every word right. On the two-speaker run it produced both lines exactly as written; LTX lost two characters off the front of the first one. Word-for-word correctness is the reason to
run H3 at all, and it is the one thing the faster model did not deliver.

"Which one is lighter" has no answer on a 32 GB card. Both are just barely inside it.

Not measured yet

  • Loudness-matched blind listening by a human. CER says a machine can read the words back. It says nothing about whether the delivery sounds like a person.
  • Lip-sync alignment. Never measured on either model. The T8 node pack ships an av_delta_ms output and I have not used it.
  • LTX-2.5's Chinese naturalness. As of 2026-09-11 I could not find a quantified evaluation of it anywhere, in English or Chinese. This run only established that a machine can understand it.

References

  • Lightricks/LTX-2.5 on Hugging Face — the five files, and the NVFP4 variant with the re-uploaded metadata
  • Comfy-Org/workflow_templates — the official LTX-2.5 blueprint the sampler settings came from

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Also in this series: Part 1 — first clip on one RTX 5090 · Part 6 — two characters in one shot with Ref2VA

FAQ

Is LTX-2.5 faster than MiniMax-H3 on the same GPU?
    Yes, by roughly 3x on my runs. One RTX 5090, one Simplified Chinese dialogue line, ComfyUI on both: LTX-2.5 finished the whole flow in 28.67 s and MiniMax-H3 in 81.22 s. Sampling alone was 14.31
    s vs 66.40 s. Both transcribed back at character error rate 0%, and LTX was running a slightly smaller pixel-times-frame load, about 85% of H3's.
Why is 1344x768 the resolution everyone uses for MiniMax-H3?
    It falls out of two constraints. The MiniMax model card defaults the short edge to 768 px, and the ComfyUI path needs both dimensions divisible by 32. A 16:9 frame at a 768 short edge would need
    768 x 16/9 = 1365.33 px of width, which is not even an integer; rounding down to a multiple of 32 gives 1344. Note that 1344/768 is 1.75, so the result is 7:4, not 16:9. NVIDIA's Sol-Engine
    benchmark and FastVideo's H3 distillation both land on the same 1344x768 by 124 frames shape.
Can LTX-2.5 handle two speakers alternating in one clip?
    It did in my run. The single-speaker limitation I had read about is scoped to the Dub-It beta IC-LoRA, and that page also says it has not been validated on LTX-2.5. Four independent checks
    agreed: non-overlapping ASR timestamps, correct frame-by-frame mouth attribution, a near-silent RMS envelope at the handover, and pitch medians dropping from 286 Hz to 130 Hz between the two
    lines. It did drop the first two characters of the first line, which MiniMax-H3 did not.
Which files does LTX-2.5 need for ComfyUI?
    Five, all from the Lightricks/LTX-2.5 repository: the 20.03 GB distilled transformer, the 14.32 GB Gemma4-12B text encoder with projection, a 1.35 GB video VAE, a 0.34 GB audio VAE, and the 0.93
    GB latent spatial upscaler. Take the distilled transformer rather than dev, and take the conv video VAE rather than the alternative. The upscaler is not optional because the official blueprint
    samples in two stages with an upsample between them.
Do LTX-2.5 and MiniMax-H3 fit on a 32 GB card?
    Both fit and neither leaves room. On a 32,607 MiB RTX 5090, LTX-2.5 peaked at 31,736 MiB and MiniMax-H3 at 31,811 MiB on the two-speaker run, which is 871 and 796 MiB of headroom. Third-party
    reports put LTX-2.5 into the wall at 8 seconds of 720p, so treat about 10 seconds as the practical ceiling for a single generation.

MiniMax-H3 on RTX 5090

← PreviousMiniMax-H3 Ref2VA: Two Characters From Two Reference Images, 437 s on a 5090Next →[Benchmark] Roll cheap, then enhance the keeper: NVIDIA's two-stage MiniMax-H3 + LTX-2.5 pipeline explained
All posts in this series

 1. 1. [Benchmark] Running MiniMax-H3 on one RTX 5090: four files, 31.7 GB, 175s per talking clip
 2. 2. [Benchmark] Twice as Fast: MiniMax-H3 on an RTX 5090, 625s Down to 314s
 3. 3. [Benchmark] MiniMax-H3 1080p Isn't Unsupported, It's Three and a Half Hours
 4. 4. [Benchmark] Ten effect embeddings for MiniMax-H3 — 10 MB that buys you bullet time, fire breath and a full year of seasons
 5. 5. [Benchmark] From OOM to 8.6 minutes: a MiniMax-H3 config stack for one RTX 5090
 6. 6. MiniMax-H3 Ref2VA: Two Characters From Two Reference Images, 437 s on a 5090
 7. 7. [Benchmark] LTX-2.5 vs MiniMax-H3 on one RTX 5090: 29s vs 81s for the same Chinese dialogue clip[Current post]
 8. 8. [Benchmark] Roll cheap, then enhance the keeper: NVIDIA's two-stage MiniMax-H3 + LTX-2.5 pipeline explained
 9. 9. [AI Workflow] Block a scene in Blender, let MiniMax-H3 film it: local Ref2VA on an RTX 5090
10. 10. A 22-second Taiwan FPV flight without a drone: Blender blockout + MiniMax-H3 on one RTX 5090
11. 11. A 25-second no-fog FPV one-take with MiniMax-H3 and ComfyUI-H3-Continuum on one RTX 5090
12. 12. Turn a real dance into a depth video and have MiniMax-H3 restage it in a palace: native 768 on one RTX 5090
13. 13. [MiniMax-H3] A storyboard image isn't a prompt: getting local H3 to hit 13 of 16 shots
14. 14. [MiniMax-H3] Open-H3-Regenerate-2K on an RTX 5090: more detail than VSR at 133 s a clip
15. 15. Making a character do parkour with NVIDIA MotionBricks on one RTX 5090: 1.66 ms per generation
16. 16. Pika-H3-Flex on an RTX 5090: 2x faster MiniMax-H3, and a Blender blockout makes the camera obey
17. 17. [Benchmark] No GPU? Run MiniMax-H3 on a Colab G4: 66 s per clip, ~49 clips an hour
18. 18. How to run Viggle Remake character swap on an RTX 5090 (int8, ComfyUI)

Read next

  • 2026-09-13
    [Benchmark] Roll cheap, then enhance the keeper: NVIDIA's two-stage MiniMax-H3 + LTX-2.5 pipeline explained

    NVIDIA's Sol-H3 two-stage recipe on an RTX 5090: MiniMax-H3 drafts at 672×384 in 25.5 s, LTX-2.5 refines only the take you keep. Downloadable ComfyUI workflow.

  • 2026-11-10
    How to run Viggle Remake character swap on an RTX 5090 (int8, ComfyUI)

    How to set up Viggle Remake in ComfyUI on a single 32 GB RTX 5090 with int8 weights: character swap, scene change, and an 89 vs 74 blind win over a SAM mask + H3 Character Swap pipeline.

  • 2026-11-04
    Pika-H3-Flex on an RTX 5090: 2x faster MiniMax-H3, and a Blender blockout makes the camera obey

    Pika-H3-Flex on an RTX 5090: 20.8 s per 5-second clip vs 38 s for MiniMax-H3. Text-only orbits drift; a Blender blockout got 6 of 6 clips around.

  • 2026-10-30
    [MiniMax-H3] Open-H3-Regenerate-2K on an RTX 5090: more detail than VSR at 133 s a clip

    NVlabs' Open-H3-Regenerate-2K on one RTX 5090: 133 s per clip vs 19 s for RTX VSR, real eyelash detail, and the cost: faces drift and small sign text turns into fake characters.

ShareCopyRedditLinkedInXFacebook
❯echo "thanks for reading"

If this helped, buying me a coffee is all it takes — it lets me run more hands-on tests and share what I find.

Buy me a coffee

Don't miss the next one
