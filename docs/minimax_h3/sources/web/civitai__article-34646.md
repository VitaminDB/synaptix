# Source: https://civitai.com/articles/34646

Fetched: 2026-11-28 (curl + w3m text dump)

---

santa hat
deerdeer nosedeer glow
Pro

Create

(More create options)
Sign In(Menu)

Create

(More create options)
(Search)(Menu)
[4a5e404d-e]
Civitai turns 4. Pick up a hat.Four color teams, Nov 1 to Nov 30. Hat your best work and score for your team.Join a team(Dismiss Civitai turns 4.)
homemodelsimagesvideos3D Modelshubsarticlescomicscruciblesupdatesshop
More

MiniMax H3 Prompt Guide: Structure + 5 Copy-Ready T2V, I2V & Ref2V Prompts

0

13

AISeedance25's Avatar

AISeedance25

Aug 30, 2026

(Updated: a month ago)

generation guide
minimax h3comfyuiprompt engineeringai generatorai video
 (More options)MiniMax H3 Prompt Guide: Structure + 5 Copy-Ready T2V, I2V & Ref2V Prompts

MiniMax H3 Prompt Guide

MiniMax H3 prompts work best when they read more like a production plan than a pile of visual adjectives.

A useful H3 prompt tells the model:

* what exists at the beginning

* what visibly changes

* how the camera observes that change

* which references control which details

* what physical sounds belong to the scene

* whether background music is required

Below is a practical structure I use for text-to-video, image-to-video and reference-to-video workflows.

## 1. The Basic MiniMax H3 Prompt Structure

```text

integrated_multimodal_description:

[Shot 1] Subject, action, setting, framing and camera behavior.

overall_soundscape:

Physical and environmental sounds.

non_diegetic_music:

Audience-only music, or N/A.

```

integrated_multimodal_description controls what happens visually and temporally.

overall_soundscape is for physical and environmental audio.

non_diegetic_music is for soundtrack or audience-only music.

One of the biggest improvements you can make is to describe observable changes instead of stacking more adjectives.

## Example 1 — Vintage Binocular Brand Film

Mode: Reference to Video

```text

Use Images 1–4 as sequential keyframes seen through a vintage binocular viewfinder. Open out of focus with subtle handheld shake, push in quickly, and rack focus onto Image 1. Move between keyframes
with fast binocular scans, motion blur, optical smearing and brief exposure flicker. Cut at peak blur, then settle sharply into focus.

Keep the twin circular lens mask fixed in position, scale, vignette and edge softness. Only the image inside the mask may move. Let fabric move naturally while its lettering follows the folds and
remains legible. Give the people restrained, candid fashion-campaign movements.

overall_soundscape:

Wind, distant city ambience and subtle mechanical focus clicks.

non_diegetic_music:

Restrained cinematic pulse.

AVOID: lens-mask drift, warped lettering and soft transitions that weaken the scan effect.

```

Why it works: The prompt assigns the references a sequence, gives each transition an observable mechanism and treats the binocular mask as immutable geometry. That leaves the model less room to
reinterpret the central visual device.

## Example 2 — Dynamic Product Landing Page

Mode: Reference to Video

```text

Use Image 1 as the exact product reference. Create a four-section product page: hero, performance detail, material detail and CTA.

[0–3 seconds] The product enters the hero section and settles without changing shape or logo placement.

[3–8 seconds] Perform one smooth page scroll.

[8–12 seconds] Show two deliberate hover states using scale and color inversion.

[12–15 seconds] Settle on the product CTA and hold.

Preserve the product color, proportions and geometry throughout. Use supplied copy only.

overall_soundscape:

Subtle interface ticks and one restrained speed whoosh.

non_diegetic_music:

Minimal electronic rhythm, mixed low.

AVOID: fake browser chrome, new products, illegible microcopy and nonstop parallax.

```

Why it works: The page has a fixed information architecture and a short interaction timeline. Product geometry is separated from UI motion, so scrolling and hover effects do not silently redesign the
object.

## Example 3 — Luminous Creatures in a Live-Action Kitchen

Mode: Text to Video

```text

integrated_multimodal_description:

[0–4 seconds] In a small live-action kitchen at dusk, a handheld phone camera hesitantly discovers a tiny hand-drawn luminous creature near the sink. Use delayed autofocus and slight exposure
breathing.

[4–9 seconds] The creature crosses the table while its glow reflects softly on nearby objects. The kitchen layout and sunset direction remain stable.

[9–13 seconds] A mug clinks. The creature pauses and reacts with one small electronic chirp.

[13–15 seconds] The camera operator steadies the phone as the creature looks back. Hold the final composition.

overall_soundscape:

Room tone, cloth friction, a soft mug clink, faucet drips, footsteps, quiet breathing and tiny electronic creature tones.

non_diegetic_music:

N/A.

AVOID: giant eyes, split mouths, fangs, threatening behavior, lunges, jump cuts to black and studio lighting.

```

Why it works: The prompt makes the mixed-media concept executable through four visible beats. It also states which imperfections belong to the phone footage while keeping the environment consistent
and the creature non-threatening.

## Example 4 — Automotive Website UI Animation

Mode: First & Last Frame

```text

Begin from the supplied page composition and keep the car, crop, typeface, line breaks and spacing locked.

[0–3 seconds] The top headline slides down 32 pixels and settles.

[3–6 seconds] The lower copy panel slides up 24 pixels and settles.

[6–9 seconds] The car lights transition from dark to red without moving the car.

[9–12 seconds] Hold the completed layout.

Camera: static.

overall_soundscape:

Two soft UI swishes and a subtle electric light-up tone.

non_diegetic_music:

N/A.

AVOID: page scrolling, car movement, new copy, font changes and liquid transitions.

```

Why it works: Every animated layer has a direction, distance and order. The car and layout are explicitly locked, preventing a simple interface animation from turning into an unwanted camera or
object move.

## Example 5 — Green-Screen Fairytale Composite

Mode: Reference to Video / Video Editing

```text

Use Video 1 for all subject motion, timing and camera behavior. Replace only its green background with the fairytale environment from Video 2.

Preserve the subject pixels, silhouette, face, wardrobe, choreography and final pose. Remove green spill, match the subject's contact light to the new environment, and add shallow atmospheric depth
behind the subject. Background elements must respond plausibly to the source motion without crossing or obscuring the face.

overall_soundscape:

Retain the original dialogue and performance timing; add only subtle environmental ambience that belongs to the new setting.

non_diegetic_music:

Retain the source music if present; otherwise N/A.

AVOID: re-animating the subject, changing camera motion, edge halos, altered lip timing and new foreground objects crossing the face.

```

Why it works: The reference roles are narrow: Video 1 owns the performance, while Video 2 owns the environment. The prompt also defines the composite work—spill removal, contact light and
depth—without giving the model permission to regenerate the performer.

## MiniMax H3 Prompt Checklist

Before generating, ask:

1. What exists at the beginning?

2. What visibly changes during the clip?

3. What is the primary camera movement?

4. Which reference controls which trait?

5. What should remain visually consistent?

6. Which sounds exist inside the scene?

7. Is audience-only music actually necessary?

If those answers are clear, the model usually receives a much stronger production brief.

## 20 More MiniMax H3 Prompt Examples

I maintain a larger MiniMax H3 prompting guide with 20 playable examples covering cinematic work, advertising, reference-to-video, first/last frame workflows and common failure fixes.

[View the Complete MiniMax H3 Prompt Guide →](https://minimax3.org/minimax-h3-prompt-guide?utm_source=civitai&utm_medium=referral&utm_campaign=minimax_h3_prompt_guide&utm_content=article_footer)

The full guide includes playable outputs, reference inputs, optimized production prompts and troubleshooting notes.
