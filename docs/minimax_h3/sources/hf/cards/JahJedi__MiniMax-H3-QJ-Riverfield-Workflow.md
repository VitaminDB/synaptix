# JahJedi/MiniMax-H3-QJ-Riverfield-Workflow

Источник: https://huggingface.co/JahJedi/MiniMax-H3-QJ-Riverfield-Workflow · скачано 2026-11-28
Загрузок: 0 · лайков: 1 · базовая:  · теги:  · обновлено: 2026-08-07T01:30:17.000Z

---

# Queen Jedi — a local MiniMax-H3 short, with the workflow

<video controls src="https://huggingface.co/JahJedi/MiniMax-H3-QJ-Riverfield-Workflow/resolve/main/QJ_riverfield_p3.mp4" title="Queen Jedi — Riverfield"></video>

<sub>1080×1920 · 60 fps · 39.8 s · H.264 ~40 Mbit/s · 191 MB</sub>

Everything below ran locally on one workstation. No cloud, no API.
The ComfyUI workflow used to make it is in this repo — download it, drop it on the canvas, done.

**[⬇ minimax_h3_R2V_MK3.1.json](minimax_h3_R2V_MK3.1.json)**

### What's in the graph

- Live preview through a tiny VAE — the shot builds in front of you while it samples, so a bad seed
  gets killed in the first steps instead of after the full render.
- 9 reference image slots, plus reference video and reference audio.
- The video slots also accept stills, so photos can go there too when that is more convenient.
- SeedVR2 upscaler wired in.

### Before you run it

Custom node packs the graph needs — everything else is stock ComfyUI 0.30:

`ComfyUI-VideoHelperSuite` · `ComfyUI-KJNodes` · `ComfyUI-Impact-Pack` · `rgthree-comfy` ·
`ComfyUI-Easy-Use` · `seedvr2_videoupscaler` · `ComfyUI-SolAttn_triton` · `WhatDreamsCost-ComfyUI`

How it ships:

- **SeedVR2 upscaler — bypassed.** Turn it on when you want the 1080p pass; leave it off while you
  hunt for a seed, it is roughly half the render time.
- **EasyCache and SolAttn — bypassed, on purpose.** Both trade accuracy for speed. On simple motion
  you will not notice; on fast limbs and spins they cost you extra fingers and smeared edges. The
  speed was not worth the damage here.
- **Sage attention — active.** Fine for clips up to about 10 s, which is what this piece used.
  At 15 s it produces a dead brown-noise clip. Bypass that node before you go long.
- The reference image / video / audio slots point at my own files. Swap in yours — nine reference
  images max, three videos, three audio.

---

## What this is

A short narrative piece — five scenes, one continuous story, cut together in DaVinci Resolve.
Every shot is **MiniMax-H3 (Hailuo 3.0) Ref2VA** running locally in ComfyUI: video and audio come
out of the same forward pass, so lip-sync and sound effects are native, not added afterwards.

Character consistency across scenes comes from reference photos, not from a LoRA — nothing was trained.
Each scene starts on the last frame of the previous one, so the cuts land on matching frames.

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB |
| CPU | AMD Ryzen 9 9950X3D (16C / 32T) |
| RAM | 128 GB |
| OS | Ubuntu 24.04.4 LTS, kernel 6.14 |
| Driver | 580.159.03 |
| ComfyUI | 0.30.0 |
| Edit | DaVinci Resolve Studio 20 |

## Render data

Measured on the machine above — not estimates.

**Generation — MiniMax-H3 Ref2VA**

| Setting | Value |
|---|---|
| Model | `minimax_h3_ref2va_int8_convrot` (32 GB) · text encoder `nvfp4_awq` |
| Resolution | 9:16 vertical — part rendered at 1.0 Mp then upscaled, part at 2.0 Mp native (experiment) |
| Length | 10 s per clip, 24 fps |
| Sampler | `res_multistep`, scheduler `beta`, no CFG (weights are CFG-distilled) |
| Steps | 20 |
| VRAM | ~85 GB working, 92 GB peak at bf16 + video reference; int8 sits far lower |

Longer clips cost disproportionately more than short ones — attention over the time axis grows
faster than the length does. 10 s per clip was the working length here.

**Post**

- SeedVR2 (`seedvr2_ema_7b_sharp_fp16`) upscale → 1080×1936.
- 24 → 60 fps conversion done in Resolve, not in ComfyUI: the 24 fps clips sit on a 60 fps timeline
  and Resolve conforms each clip individually with Optical Flow + Speed Warp. Per-clip means the
  interpolator never runs across a cut, which is where frame interpolation usually leaves a smear.
  Peak VRAM during that render: 44 GB.

## Three things worth knowing

- **Reference photos carry pose, not just identity.** Set them to `attribute_transfer`, not
  `fully_preserved` — static portraits at full preservation freeze the character in place and the
  motion reference gets ignored. If you need a hard pose, feed a photo already in that pose.
- **Tags are `<Picture i>` / `<Video k>` / `<Audio j>`, numbered per type in slot order.** There is no
  `<Subject>` tag in H3, despite what looks natural to write — it is silently ignored.
- **Sage attention breaks H3 on long clips.** Fine up to ~10 s, but at 15 s it produces a dead brown
  noise frame (a NaN latent reaching the VAE). Disable the sage node for H3; keep it for the upscaler.

## The prompts

H3 does not take a free-form paragraph. Ref2VA wants six named fields, in this order:
`subject_definitions` → `summary` → `retention_analysis` → `detailed_description` →
`overall_soundscape` → `non_diegetic_music`. Everything the model reads is in there — camera,
physics, sound, what each reference file is allowed to control.

Two real prompts from this piece, verbatim. The first drives the dance from a video reference;
the second has no video reference at all, so every movement and the camera come from the text.

<details>
<summary><b>Scene 4</b> — dance driven by a video motion reference</summary>

```
subject_definitions:
<Picture 2> and the following reference photos are the demon queen: lavender-purple skin, long platinum-blonde hair, curved golden horns, a glowing crystalline crown, a slender purple spaded tail, wearing a black-and-gold latex outfit with long gloves and thigh-high heeled boots.
<Picture 1> is the exact starting frame of this video: the demon queen high in the air near the top of a tall, solid, gleaming golden pillar at the center of a floating platform in deep space, seen from BEHIND with her back to the camera, gripping the golden pole with one leg bent, her platinum hair and slender purple tail streaming to the side. Behind her is deep space; a blue-glowing comet is falling from the upper right, trailing a long glowing tail. Far below on the horizon the castle-city of Riverfield is already burning and in ruins from an earlier impact — smashed towers and walls lit orange with fire. Around them are violet nebulae, floating stone ruins, drifting cosmic haze, a lit candle at the base of the platform, glowing storm clouds and lightning.
<Video 1> is the pole-dance motion reference: a woman performs a graceful, athletic pole dance that starts high in the air near the top of the pole with flowing airborne work, descends the pole into a sweeping spin, drops to the floor for grounded floor choreography around the base, and near the end begins to rise back up. It provides the dance movement and the camera framing only.
<Audio 1> is the music track.
Role separation (one asset, one job): <Picture 2> and the following reference photos define ONLY who she is and what she wears (identity, face, skin, hair, horns, crown, tail, costume, colors) — the various poses shown in those photos must NOT be copied. <Picture 1> is the exact first frame / starting pose at 0.00 seconds. Her pose and every movement after that come entirely from <Video 1>.

summary:
[reference generation + video motion + audio reuse] the demon queen dances on the golden pillar at the center of the platform in deep space as the punishment of Riverfield continues — two more comets striking the already-burning city — her body following the motion of <Video 1> exactly, with regal grace and strength, singing over <Audio 1>.

retention_analysis:
<Picture 1>: fully_preserved — the exact first frame at 0.00 seconds; her back-view starting pose high on the pillar, position, framing and composition are preserved at the start, then the motion of <Video 1> takes over.
<Picture 2> and following reference photos: attribute_transfer — her look (skin, hair, horns, crown, tail, outfit) and pose material.
<Video 1>: motion reference — the SOLE source of her pose and full-body movement after the start; her body follows the motion of <Video 1> exactly.
<Audio 1>: reference — the music track plays throughout.

detailed_description:
Fantasy digital illustration, cinematic, the same deep-space scene continues around the tall golden pillar. The video begins exactly in the pose, position and framing of <Picture 1> — the demon queen high in the air near the top of the pillar with her back to the camera, gripping the golden pole, one leg bent, her platinum hair and purple tail streaming to the side — and from there her dance follows the motion of <Video 1> at its own full speed, tempo and rhythm — the reference pace, NOT slowed or softened to the music; the reference photos define ONLY her identity and costume, and their poses are NOT copied. [Shot 1] the demon queen performs a graceful, athletic pole dance at the tall, solid, gleaming golden pillar, which stands static and unmoving at the center of the floating platform, never shifting or swaying while she works around it — her body tracing the pole-dance movement of <Video 1> exactly. In one continuous shot, from that exact starting pose her dance unfolds: she begins high near the top of the pillar in the air, moving through flowing airborne pole work high up — spins, turns and long sweeping extensions around the pillar, her body carried by the motion of <Video 1>; then she flows into a controlled descent, sliding smoothly down the pillar with one leg hooked high, unwinding through the lower pole into a sweeping layback and spin as she comes down; she settles to the floor and moves through grounded floor choreography around the base of the pillar — kneeling, low lunges and splits, arching back-bends and slow sensual turns, one hand often reaching to the pillar; near the end she begins to rise again, one gloved hand reaching high up the pillar and a knee bent beneath her, coming up out of the floor into a rising spin, ending low at the base of the pole as she starts to lift back up. She moves with natural, lifelike physics — chest and hips swaying with real weight and momentum rather than stiff or doll-like, her platinum-blonde hair and slender purple tail trailing and swinging through every turn. Throughout she keeps correct human anatomy, exactly two arms and two legs at all times, with no extra, malformed, duplicated or third limb ever appearing, even during the fastest spins and movements. She grips the pole with her hands and arms while keeping her body at a slight distance from it — her torso not pressed flat against the pole — so the solid golden pole always stays beside her body as a separate object and never overlaps, sinks into or passes through her body. She dances with regal majesty and imperious command, presiding over the doom she has unleashed rather than dancing on obliviously — poised, cold-eyed and in full control, the queen and executioner of the destruction unfolding behind her. Partway through she sings along to the music (S1): <d>[English] Oooo you're looking at a devil, I like the sound of the sirens</d>. Far off in the distant background, deep on the horizon, the great castle-city of Riverfield is already burning and in ruins from the first comet — its walls, towers and spires smashed and blackened, orange fires guttering through the wreckage, smoke rising into the stormy sky. Now the SECOND blue-glowing comet, the one falling from the upper right in the starting frame, streaks the rest of the way down and slams into the ruined city, detonating in one single colossal blast even bigger than the first — a massive meteor impact that hurls debris, fire and molten rock violently outward in every direction, a huge shockwave of energy rolling out across the distant sky, total obliteration of whatever still stood. Then a THIRD blue comet streaks in behind it, its long glowing tail burning through the stars, and comes down to the LEFT of the city center, striking the ground and detonating in another great blast — walls and rubble flung outward, another crater torn open, more fire and debris thrown across the horizon. Both are meteor-impact explosions in the same style — comet strikes that smash stone to pieces with outward-blasting debris and rolling shockwaves, NOT nuclear mushroom clouds. Far beyond it all, towering storm clouds pile along the horizon with lightning flashing and forking deep within them and low rolling thunder. All of this destruction stays far away on the horizon, deep in the background, while the queen, the pillar and the platform stay crisp, close and untouched near the camera. Through the clip a faint drift of glowing embers and black ash carries across the void toward the platform, stirring her hair, the glow of the distant fires flickering over her as she dances on, poised and unharmed to the final frame.

overall_soundscape: A faint cosmic hum runs throughout, with the soft rustle of hair and fabric as she moves, the quiet crackle of the nearby candle, low rolling thunder in the far distance, and the deep, muffled booms of the far-off comet explosions rolling in after each flash.

non_diegetic_music: The music from <Audio 1> plays throughout. This is the final scene of the series, so the track MAY resolve and settle at the very end as the dance closes — but keep it simple: it plays continuously and full through almost the entire clip.
```
</details>

<details>
<summary><b>Scene 5</b> — no video reference: motion, physics and camera written out</summary>

```
subject_definitions:
<Picture 2> and the following reference photos are the demon queen: lavender-purple skin, long platinum-blonde hair, curved golden horns, a glowing crystalline crown, a slender purple spaded tail, wearing a black-and-gold latex outfit with long gloves and thigh-high heeled boots.
<Picture 1> is the exact starting frame of this video: the demon queen settled low at the base of a tall, solid, gleaming golden pillar at the center of a cracked stone floating platform in deep space, one gloved hand reaching high up the pillar. Lotus blooms and a lit candle rest on the platform, glowing embers trace the cracks in the stone. Around and beyond it are violet storm clouds with forking lightning, floating broken stone ruins, drifting cosmic haze, a blue comet high in the far sky, and the castle-city of Riverfield burning in ruins far below on the horizon.
<Audio 1> is the music track.
Role separation (one asset, one job): <Picture 2> and the following reference photos define ONLY who she is and what she wears (identity, face, skin, hair, horns, crown, tail, costume, colors) — the various poses shown in those photos must NOT be copied. <Picture 1> is the exact first frame and starting pose at 0.00 seconds. All of her movement after that comes from this description.

summary:
[reference generation + audio reference] This is the FINAL scene of the whole series and the end of the story. The demon queen rises from the base of the golden pillar on the floating platform in deep space, rising like a dancer with her hips lifting first and her back arching, then raises her hand and takes the pillar back into it as radiant golden energy until the platform stands empty. She turns to the camera with a light smile, then turns her BACK to the camera, traces a circle in the air directly in front of her, and a sorcerer's portal opens exactly there — a spinning ring of golden and violet light that sheds violet and gold rose petals and shows a magical throne hall beyond it. She then walks FORWARD through it on her own feet, exactly like a person stepping through a doorway, her body staying solid and fully visible the whole way; only after she is through and standing on the other side does the portal close behind her, leaving the shed rose petals to settle onto the empty platform. Far behind her two blue comets fall one after the other and strike the burning ruins of Riverfield on the horizon. She moves continuously with natural lifelike weight and momentum, the camera performs one continuous slow dolly in from the first frame to the last, and there is no dialogue and no singing.

retention_analysis:
<Picture 1>: fully_preserved — the exact first frame at 0.00 seconds; her grounded starting pose low at the base of the pillar, her position, the framing and the composition are preserved at the start.
<Picture 2> and following reference photos: attribute_transfer — her look only (skin, hair, horns, crown, tail, outfit).
<Audio 1>: reference — the music track plays throughout.

detailed_description:
Fantasy digital illustration, cinematic; the same deep-space scene continues around the tall golden pillar. This is the FINAL scene of the series and the story ends here. [Shot 1] The camera performs one continuous DOLLY IN for the entire clip, pushing in with small amplitude at slow speed from the first frame to the last, never stopping and never pulling back, and making no other move. The video begins exactly in the pose and framing of <Picture 1>; the reference photos define only her identity and costume, not her pose. She moves continuously from first frame to last, with real weight and momentum, never stiff, doll-like, robotic or jerky. First she rises, and she rises like a dancer, not simply standing up: from her low position at the base, one gloved hand still on the pillar, she pushes her HIPS UPWARD first while her shoulders stay low, her BACK ARCHING into a deep curve, her chin tilting back. Then the movement rolls upward: her hips travel over her feet, her spine unrolls vertebra by vertebra, chest and shoulders last, and she arrives standing tall beside the pillar. The rise is slow, controlled and sensual, all long curves and flowing lines, never abrupt. Her platinum-blonde hair swings and settles a beat after her body stops, her tail last; her chest and hips move with soft, weighted physics. She keeps exactly two arms and two legs at all times, and the pillar never passes through her body. Then she takes the pillar back, the way she summoned it but reversed: she raises one gloved hand toward it, the solid gold softens and unravels from the top downward into radiant golden energy that narrows into a beam and is drawn into her glove, until the glow dies and the platform stands empty. She lowers her hand, turns toward the camera, and a light smile touches her lips, soft and unhurried. Then she turns her BACK TO THE CAMERA and stays turned away from here on. Then she opens a portal: she lifts her other hand and traces a quick circle in the air DIRECTLY IN FRONT OF HER, and the ring ignites EXACTLY THERE, on the spot her hand just circled — in front of her, facing her, never behind her and never to one side. It is a ring of bright golden and violet light, a round sorcerer's portal exactly like those opened by Doctor Strange, except that its turning rim throws off no sparks at all but VIOLET AND GOLD ROSE PETALS, which scatter into the air around it. It stands upright a step or two ahead of her, tall enough to walk through, and through it a magical throne hall is visible, dark and vast, lit by warm golden light. The portal opens fully first and holds steady. Only then does she leave, and she leaves ON FOOT: she walks FORWARD into the ring, away from the camera, stepping through exactly like a person stepping through a doorway. Her body stays solid, opaque and fully visible the whole way. One boot lands past the threshold, her weight moves onto it, shoulders and torso pass the rim, hair and tail last. As she crosses, the edge of the ring covers her from view the way a doorframe covers someone walking into another room — first her trailing arm, then her hair, then her tail. Beyond the ring she is briefly seen in the golden-lit hall, smaller because she is further away. Then the ring shrinks and snaps shut, its light folding inward, the closing rim covering the view of the hall and of her in it. The rose petals it shed settle slowly, tumbling edge over edge and rocking on the air the way real petals fall, coming to rest on the stone. The clip ends on the empty platform, only the violet and gold petals settling over the cracked stone — the final frame of the series. Meanwhile the background lives its own arc: two blue-glowing comets fall one after the other high in the deep-background sky, each with a long luminous tail — the first while she rises, the second as she draws in the pillar — and they STRIKE the ruined castle-city of Riverfield far below on the horizon in bright blue-white detonations that blast dust and fire from the towers. All of it stays far off on the horizon, small and deep in the background, never near the foreground. Closer in, the candle flickers, dark clouds drift upward, the floating ruins rotate gently, lightning forks in the far clouds, embers and ash drifting across the platform. No dialogue, no singing, no on-screen text; the slow dolly in is the only camera move, no zoom, pan, tilt or shake.

overall_soundscape: A faint cosmic hum runs throughout, with the soft rustle of fabric and hair as she rises to her feet and the quiet crackle of the candle beside her. A deep resonant tone swells as the golden pillar unravels and is drawn into her hand. Two deep muffled booms roll in from far away on the horizon as the comets strike, spaced apart, with low rolling thunder murmuring behind them. The portal opens with a soft rushing swell and a low humming roar that continues while it stands open, then cuts off with a sharp snap as it closes, leaving the papery flutter of rose petals drifting down and settling onto stone. The clip ends in quiet stillness over the empty platform.

non_diegetic_music: The music from <Audio 1> plays throughout. This is the finale of the whole series, so the track may build through her rise and the opening of the portal and then resolve and settle at the very end as the petals come to rest. It plays continuously and full for almost the entire clip, and a real ending here is fine.
```
</details>

Notes on what actually matters in there:

- **One asset, one job.** Say explicitly what each reference controls. The photos define identity
  and costume only; `<Picture 1>` is the first frame; `<Video 1>` owns the movement. Leave it vague
  and the model averages them into mush.
- **Reference tags are `<Picture i>` / `<Video k>` / `<Audio j>`**, numbered per type in slot order.
  `<Subject>` looks like it should exist. It does not, and it is silently ignored.
- **When there is no video reference, the camera has no owner** — write it out, or the shot drifts.
- **Do not write "she vanishes."** Describe what happens to the picture instead: the ring's edge
  covers her the way a doorframe covers someone walking into the next room. Concrete beats abstract
  every time.

---

Made locally. If you build on it, no attribution needed — just make something good.
