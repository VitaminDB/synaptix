# Source: https://www.stablediffusiontutorials.com/2026/08/minimax-h3-lora-models.html

Fetched: 2026-11-28 (curl + w3m text dump)

---

  • Forge
  • Prompts
  • Tutorials

HomeModels52 Minimax H3 LoRA models for Optimized Video Generation

52 Minimax H3 LoRA models for Optimized Video Generation

September 05, 2026

 

minimax h3 lora models

MiniMax H3 is starting to become a lot more interesting once you move beyond the base model. There are already LoRAs and modified versions that can change how fast it generates, how characters move,
how the camera behaves, how physical interactions look, and even the overall visual style of the output.

Some of these are built for very specific jobs, while others completely change the way you can use H3 in a workflow. You will find everything from 4-step acceleration and audio-video optimization to
anime line art, Wushu action, realistic people, spatial physics, Krea 2-inspired styling, and more. 

The resulting videos mentioned below are highly compressed. For actual quality please refer to their Hugging face repositories. Below, we have collected the MiniMax H3 LoRAs and variants worth
checking out, along with what each one is actually useful for.

 

 Table of Contents

 

Instructions to do the setup:

(a) Make sure you use the Minimax H3 base model with workflow. Then, download any of the lora models from its hugging face repository and save it into your ComfyUI/models/loras folder (for Comfy
users) or into specfic relevant folder if using other WebUIs like Wan2GP, ForgeUI etc.

(b) The recommended workflows and settings are available in the respective repositories. However, if a workflow is not provided, you can manually integrate the LoRA into the standard MiniMax H3
workflow. 

Start with the basic MiniMax H3 workflow and add a Load LoRA node. Connect the LoRA loader after the Load Diffusion Model node or after the UNet Loader if you are using a GGUF modeland before the
KSampler node. Then, select and load the corresponding downloaded LoRA model in the Load LoRA node. Once connected correctly, the LoRA will be applied to the model pipeline during generation,
allowing you to use its specific style or effects with the MiniMax H3 workflow.

H3 Licon VBVR LoRA

Licon VBVR is a LoRA for MiniMax-H3 that turns images into videos.It was trained on specially rebuilt VBVR data made for H3. It keeps objects more consistent and makes character movements more
logical. This reduces problems like objects duplicating, clipping through each other, or overlapping wrongly during motion.

Download Minimax H3 VBVR Lora model from its hugging face repository.

MiniMax-H3 360° Orbit LoRA (first–last frame)

This LoRA for MiniMax-H3 (FL2VA version) turns one photo into a frozen 360° camera orbit that ends on the exact same frame it started from.It uses the same image as both the first and last keyframe.
This forces the camera to complete a full circle and return to the starting view, making seamless loops and longer stitched shots easy.

Unlike Ref2VA (which treats images as loose references), FL2VA locks the start and end frames tightly. Normally the base model freezes when both frames are identical. This LoRA teaches it to create a
smooth, geometry-consistent orbit between those matching frames while keeping the endpoints exact.

Download Minimax H3 360 orbit  Lora model from its hugging face repository.

 

 

MiniMax H3 3D Animation Style LoRA

This LoRA was trained on a custom 3D animation dataset for text-to-video (T2V). The same 3D animation style also works well with image-to-video (I2V) workflows. For human characters, use strength 0.8
to keep good prompt following and character consistency. For animals, fantasy characters, or environments, raise it to 1.0 for a stronger 3D animation look.

Download Minimax H3 3d animation Lora model from its hugging face repository.

 

 

 MiniMax-H3 Anime2Realism

This LoRA converts anime images into realistic, photorealistic live-action style using MiniMax-H3. It supports Reference-to-Video (Ref2V). It keeps the character’s features, pose, clothing,
composition, objects, lighting, and colors, while turning the scene into natural live-action.

Download Minimax H3 AnimeToRealism  Lora model from its hugging face repository.

 

 

 MiniMax-H3 Character Swap LoRA

This is an experimental swap lora for H3. It was trained for 1,000 steps.The idea is simple, you give it a reference image of the new character, and it tries to replace someone in the source video
while keeping the original scene looking the same.In local tests, it does a better job at holding onto the background than the base model. That said, the motion timing, facial expressions, and hard
cuts can still feel a bit off. 

Download Minimax H3 Chracter swap  Lora model from its hugging face repository.

 

 

 TaoMate-H3 3-Step LoRA

This is a ComfyUI-friendly conversion of the official TaoMate-H3 3-Step LoRA for MiniMax H3.

It’s just a converted checkpoint made so you can drop it into your existing MiniMax H3 workflows in ComfyUI more easily. Use it with the original TaoMate-H3 model 

TaoMate-H3 itself is a low-latency setup that generates audio and video together in small chunks using this 3-step LoRA. It can keep going for longer videos and works at 480p, 768p, and 1080p.

Download Minimax H3 taomate  Lora model from its hugging face repository.

 

 

Kijai's  TaoMate-H3 3-Step LoRA

You can also use Kijais optimized taomate lora. Its the same repacked optimized version as described above for Minimax H3 with 3 Step video generation. 

Download Kijai's Minimax H3 taomate H3 Lora model from its hugging face repository.

 

 

10 ErosMax H3 beta5 Turbo Lora 

Use this LoRA together with the 10 ErosMax checkpoint.It’s made so the regular (non-Turbo) 10Eros_Max_h3_hybrid_beta5 model can run in fewer steps around 14 to 16  while still trying to keep good
image and audio quality.Just load the rank 128 LoRA at a strength of 0.4–0.6, set steps to 14–16, and use spectrum mode with Res_multistep + Simple. That combo usually gives a nice balance between
speed and quality.

Download Minimax H3 10Eros Beta Turbo Lora model from its hugging face repository.

 

 

Minimax H3 Snorricam LORA

This LoRA gives you that intense, chaotic Snorricam-style camera look — the classic body-mounted camera effect used in movies.

Download Minimax H3 Snorricam Lora model from its hugging face repository.

 

 

MiniMax-H3 I2V Anime Motion LoRA 

This is an experimental LoRA for image-to-video generation.Its trained on curated stop-motion-style clips and is made for creating seamless looping anime videos. You will get subtle character
expressions, small gentle movements, and soft background motion. Training clips were up to 15 seconds long.

Download Minimax H3 Anime motion Lora model from its hugging face repository.

 

 

 Minimax H3 Cinematic LoRA V0.1

This LoRA gives your generations a realistic cinematic look.For the best results, use the trigger word together with prompts that mention commercial cinematography, vintage 1960s film style, and
dramatic lighting. 

Download Minimax H3 Cinematic Lora model from its hugging face repository.

 

 

Minimax H3 Age Slider LORA

 This LoRA lets you control how old a person looks in the video by simply changing the LoRA weight.Instead of only relying on text prompts, you get more direct control over the character’s age just
by adjusting the strength. 

Download Minimax H3 Age slider  Lora model from its hugging face repository.

 

 

Minimax H3 Turnaround LoRA  

Give it one reference image and a simple instruction, and it generates five consistent views of the same subject, each rotated a bit further than the last.Everything is denoised together in a single
pass, and each view comes out as a clean, sharp still. Perfect for making character turnaround sheets from just one photo. 

Download Minimax H3 turnaround  Lora model from its hugging face repository.

 

MiniMax H3 Retro Toon 70s LORA

This LoRA brings back the classic 1970s animated feature-film look.You get hand-painted characters with bold ink lines, textured painterly backgrounds, limited animation, and that grainy, earthy
color palette from the era. 

Download Minimax H3 retro toon Lora model from its hugging face repository.

 

 

 MiniMax H3 Low Poly retro 3D LORA

 

This rank 32 style LoRA gives you that late-90s console 3D look — faceted low-poly shapes, flat shading, chunky pixelated textures, and bright saturated colors.Since MiniMax H3 also generates audio,
the dialogue and ambient sound stay in sync with the visuals. 

Download Minimax H3 Low Poly retro  Lora model from its hugging face repository. 

MiniMax H3 16Bit Pixel  SNES-era pixel animation 

This style LoRA turns MiniMax H3 videos into classic SNES-era 16-bit pixel animation.You get big chunky pixel clusters, stair-stepped outlines, a limited color palette, flat two-tone shading, and
those distinct held sprite poses, just like an old SNES cutscene in motion.Because MiniMax H3 generates audio together with the video, the sound stays in sync with the picture. 

Download Minimax H3 SNES era pixel  Lora model from its hugging face repository. 

 

 

Minimax H3 Viggle LORA

This LoRA is made for REF2VA video-to-video workflows.It helps with character animation, motion transfer, and head swapping, using your reference inputs to guide how the transformation looks.

Download Minimax H3 Viggle Lora model from its hugging face repository. 

 

 

Minimax H3  Natural face & speech LoRA  

 This LoRA is made to improve how faces and speech look and sound in MiniMax H3 videos.It helps create more natural full-face muscle movements, better lip-sync, and clearer English speech. You can
use it with both text-to-video (T2V) and image-to-video (I2V) generation.

Download Minimax H3 Natural Speech Lora model from its hugging face repository. 

 

 

MiniMax H3 FaceSwap LoRA 

This LoRA is made for REF2VA workflows.It takes a reference image of a face and swaps it onto the person in a reference video, while trying to keep the original motion and body movements intact

Download Minimax H3 faceswap Lora model from its hugging face repository. 

 

 

Minimax H3  ASMR Trigger & Audio LoRA (T2V/I2V)

 
This LoRA is made for creating high-quality ASMR videos with MiniMax H3.It works with both text-to-video (T2V) and image-to-video (I2V). The focus is on soft whispers, clear acoustic details, and
well-synced ASMR audio that matches the visuals.

Download Minimax H3 ASMR  Lora model from its hugging face repository.  

 

 

 MiniMax H3 Sharpness LORA 

This is a rank 64 LoRA that works with both REF2VA and FL2VA workflows.It sharpens a source video while keeping the look photorealistic. Instead of using the model’s normal reference-video node, it
conditions the generation on the source through guide latents. 

Download Minimax H3 sharpness Lora model from its hugging face repository. 

 

 

Minimax H3 Singularity LoRA 

This LoRA was manually extracted because the original developer hadn’t released an official standalone version yet.

It has a very strong visual effect. If you set the strength too high, it can create visual artifacts, so it’s best to keep it at 0.5 or lower.

Download Minimax H3 singularity Lora model from its hugging face repository. 

 

MiniMax-H3 Acc LoRA

MiniMax-H3 Acc LoRA

Alibaba's MiniMax-H3 Acc LoRA use Parallel Decoding Distillation (PDD) to speed up MiniMax-H3 image and video generation, allowing high-quality results with just a few inference steps. The collection
includes acceleration LoRAs for both FL2VA and Ref2VA workflows.

Download Minimax H3 Acc Lora model from its hugging face repository

 

 

MiniMax H3 Equirectangular 360° LoRA

 Equirectangular 360° LoRA

MiniMax H3 Equirectangular 360° LoRA is designed to generate immersive 360° equirectangular videos using MiniMax H3. It creates a full spherical view of a single environment, with the scene wrapping
completely around the viewer in every direction. The horizon is positioned around the vertical center of the frame, making the output suitable for 360° and immersive viewing experiences. The videos
are generated at 24 FPS and can also include MiniMax H3's native audio capabilities, allowing both the visuals and audio to be created as part of the experience.

 Download Minimax H3 Equirectangular 360 Lora model from its hugging face repository

 

 

MiniMax H3 FL2V 8-Step Motion Enhancer LoRA

Motion Enhancer LoRA

MiniMax H3 FL2V 8-Step Motion Enhancer is an 8-step LoRA designed for the MiniMax-H3 FL2VS video model. It is built on top of the vanilla LightX2V 8-step LoRA and focuses on improving motion
generation for human motion content.

The LoRA aims to produce more accurate male and female anatomy while better understanding motion patterns associated with close scenes. This helps improve animation quality and create smoother, more
natural-looking movements in supported generations.

Download this FL2V 8-Step Motion Enhancer Lora model from its hugging face repository

 

Minimax H3 Semantic Bridge Adatper

 Semantic Bridge Adatper

This is not a lora but you can use it as the adapter model with your basic Minimax H3 workflows. It helps in enhancing prompt adherence in human anatomy etc where the base model fails.

 

Minimax H3 World Keyboard-Driven LoRA

Minimax H3 World Keyboard-Driven LoRA is designed to create interactive, action-controlled videos using an initial frame and keyboard inputs. By responding to keyboard controls, the LoRA can generate
coordinated character actions and camera movements based on the user's input. This makes it possible to guide how the scene evolves, giving users more direct control over movement and interactions
during video generation. It is particularly useful for creating dynamic, controllable scenes where both the character and camera need to move together in response to user actions.

Download World Keyboard-Driven LoRA model from its hugging face repository.

 

 

Minimax H3 Motion Continuity Repair LoRA

Minimax H3 Motion Continuity Repair LoRA is designed to improve motion consistency and continuity in dynamic video scenes generated with MiniMax H3. It can be useful for action-heavy scenarios such
as running, sports, dancing, acrobatics, character interactions, combat, and weapon movements. The LoRA focuses on helping movements flow more naturally and consistently, making it particularly
useful for scenes where characters or objects perform fast and complex actions.

Download Motion Continuity Repair LoRA model from its hugging face repository

 

 

MiniMax H3 VR180 Stereoscopic Side-by-Side LoRA

 VR 180 minimax h3 lora

MiniMax H3 VR180 Stereoscopic Side-by-Side LoRA is designed to generate immersive VR180 stereoscopic videos using a side-by-side format. Each frame contains two separate views: the left-eye view on
the left side and the right-eye view on the right side, creating the stereoscopic depth needed for VR viewing. Each view represents a 180° x 180° half-equirectangular hemisphere, allowing the scene
to provide an immersive 180-degree perspective. The LoRA generates video at 24 FPS and supports MiniMax H3's native audio, making it suitable for creating complete VR180 video experiences with both
stereoscopic visuals and synchronized audio.

Download VR180 Stereoscopic Side-by-Side LoRA model from its hugging face repository.

 

 

MiniMax H3 Human Motion LoRA

Human Motion LoRA

MiniMax H3 Human Motion LoRA is designed to improve the quality of human movement in MiniMax H3 text-to-video (T2V) and image-to-video (I2V) generation. It focuses on creating smoother and more
natural-looking body movements, helping improve motion quality when generating videos featuring people and character actions. For best results, the recommended settings include a 720×1280 resolution,
a LoRA weight between 0.4 and 0.8, and approximately 15–30 sampling steps.

Download  MiniMax H3 Human Motion LoRA model from its hugging face repository. 

 

 

MiniMax H3 Influencer LoRA 

 MiniMax H3 Influencer LoRA

MiniMax H3 Influencer LoRA is designed to create videos with the polished, trendy visual aesthetics commonly seen on platforms like Instagram and TikTok. It supports both text-to-video (T2V) and
image-to-video (I2V) generation, making it useful for creating influencer-style content, social media videos, and visually engaging vertical content. For recommended results, use a resolution of
720×1280, a LoRA weight between 0.75 and 0.8, and around 15–30 sampling steps.

 Download MiniMax H3 Influencer LoRA  model from its hugging face repository.

 

 

MiniMax H3 Facial Realism CloseUp LoRA

Facial Realism CloseUp

MiniMax H3 Facial Realism CloseUp LoRA is designed to improve facial realism in close-up, human-focused video generation. It was trained on a high-quality portrait dataset featuring detailed human
faces, with a focus on capturing subtle facial features, natural expressions, and small micro-expressions. This makes it particularly useful for generating close-up shots where realistic facial
detail and expressive character performance are important. 

 Download MiniMax H3 Facial Realism CloseUp LoRA model from its hugging face repository.

 

 

MiniMax H3 VH5Tape VHS LoRA

MiniMax H3 VH5Tape VHS LoRA is designed to give generated footage the distinctive look of video recorded from 1980s broadcast television onto an aging VHS tape. It recreates classic analog
imperfections such as soft and smeared details, chroma color bleeding, tracking noise, and head-switching bands around the edges of the frame. This makes it useful for creating retro footage with an
authentic, degraded VHS aesthetic.

 Download MiniMax H3 VH5Tape VHS LoRA  model from its hugging face repository.

 

 

FastH3 Dense 4-Step v1 LoRA 

FastH3 Dense 4-Step v1 LoRA is a ComfyUI-optimized, pruned version of the FastH3-Dense-4-step-v1 LoRA for the MiniMax H3 video model. It is based on the officially released FastH3 4-step preview LoRA
and uses the dense version, which does not rely on VSA. The LoRA is designed for fast video generation using only a small number of sampling steps. The official recommended settings are 4 steps, CFG
1, Euler/Simple sampling, 12 video shift, and 3 audio shift. While 4 steps can produce acceptable video quality, testing suggests that the audio quality may suffer at this setting. For better audio
results, 8 steps are recommended. However, at that point, other 8-step LoRAs such as LightX2V and Alibaba's versions may offer a more attractive balance of speed and quality.

Download  FastH3 Dense 4-Step v1 LoRA model from its hugging face repository.

 

 

Minimax H3 STUDIO 1939 Animation LoRA 

MiniMax H3 Studio 1939 Animation LoRA gives your generations the look of classic 1930s animation. It transforms text prompts into footage with hand-painted gouache backgrounds, visible brushwork,
warm celluloid-style colors, and traditional character animation aesthetics.

Download MiniMax H3 Studio 1939 Animation LoRA model from its hugging face repository

MiniMax-H3 with Z-Image-native

MiniMax-H3 plus Z-Image Native adapts Z-Image’s spatial-attention profile to MiniMax-H3, producing richer textures and a more natural, gritty look while reducing digital over-sharpening artifacts.
It’s formatted for native ComfyUI integration and also includes VRAM-friendly 4-bit versions for more efficient local use.

Download MiniMax-H3 plus Z-Image Native model from its hugging face repository.

 

 

MiniMax H3 Turbo Sparse Linear Attention Lora

MiniMax H3 Turbo SLA LoRA uses Sparse–Linear Attention with an 85% attention sparsity ratio to make inference more efficient. It delivers around 2.5× faster inference on an NVIDIA RTX 5090 while
maintaining competitive visual quality.

Download MiniMax H3 Turbo SLA LoRA model from its hugging face repository.

 

 

MiniMax H3 Turbo 4-Step V1.1 LoRA

LightX2V’s MiniMax H3 Turbo 4-Step (v1.1) is an accelerated version of MiniMax H3 designed to generate results in just 4 inference steps, making the model significantly faster while keeping visual
quality competitive.

Download MiniMax H3 Turbo 4-Step (v1.1) lora model from its hugging face repository.

 

 

MiniMax-H3 Looping Sketch Anime Lora

MiniMax-H3 Looping Sketch Anime LoRA is designed for creating looping animations with a hand-drawn anime sketch aesthetic. It can turn text prompts into short, repeating sequences with expressive
linework and a rough, illustrated look, making it well suited for animated sketches, character movements, and other seamless looping scenes.

Download MiniMax-H3 Looping Sketch Anime LoRA model from its hugging face repository

Minimax H3 plus Krea 2 Style Lora

MiniMax H3 cum Krea 2 Style LoRA is an experimental style-shifting LoRA that brings the visual aesthetic of Krea 2 into the MiniMax-H3 video generation model. It is designed primarily for
Text-to-Video (T2V) workflows, helping generate videos with a Krea 2-inspired visual style while retaining MiniMax-H3's video generation capabilities.

Download MiniMax H3 + Krea 2 Style LoRA model from its hugging face repository

MiniMax H3 RAVEN Streaming LoRA 

MiniMax H3 RAVEN Streaming LoRA integrates the RAVEN framework for real-time, streaming video generation and long-horizon extrapolation. It is designed to maintain video quality over longer
sequences, helping reduce the gradual quality degradation and consistency issues that can occur as a video progresses.

Download MiniMax H3 RAVEN Streaming LoRA model from its hugging face repository

MiniMax-H3 Prompt Rewriter LoRA 8B adapter 

LightX2V's MiniMax-H3 Prompt Rewriter LoRA 8B Adapter helps turn short, simple ideas into detailed prompts that are better suited for MiniMax-H3’s audio and video generation. It works with both
text-based and keyframe-conditioned inputs and supports T2VA, T2AV, I2VA, I2AV, L2VA, L2AV, and FL2VA workflows, making it easier to get more descriptive and production-ready results without writing
complex prompts yourself.

Download MiniMax-H3 Prompt Rewriter LoRA model from its hugging face repository

 

 

Wan2.2 plus Minimax H3 Motion Lora 

Wan2.2-MiniMax H3 Motion LoRA is designed to improve motion and movement in videos generated with MiniMax H3. It combines motion-related capabilities from Wan2.2 with MiniMax H3, helping produce more
dynamic and natural-looking movement while keeping the core visual style of the H3 model.

Download Wan2.2-MiniMax H3 Motion LoRA model from its hugging face repository

Minimax H3 Wushu Action LoRA

MiniMax H3 Wushu Action LoRA adds a focus on martial arts and kung fu movements to MiniMax H3. It helps generate videos featuring human physical actions such as punches, kicks, staff techniques, and
other dynamic Wushu-style movements, making it useful for creating action-heavy martial arts scenes.

Download MiniMax H3 Wushu Action LoRA model from its hugging face repository

Minimax H3 Camera Motion LoRA

MiniMax H3 Camera Motion LoRA adds more precise cinematic camera control to MiniMax H3. It helps guide different types of camera movements, including push, pull, pan, tilt, circular moves, and
handheld motion, making it easier to create videos with more intentional and dynamic camera work.

Download MiniMax H3 Camera Motion LoRA model from its hugging face repository

 

 

Minimax H3 Spatial Physics Lora

MiniMax H3 Spatial Physics LoRA helps MiniMax H3 better understand spatial relationships and physical interactions between objects. It is designed to improve how the model handles actions such as
collisions, stacking, falling, and occlusion, making object movement and interactions in generated videos feel more physically consistent and believable.

Download MiniMax H3 Spatial Physics LoRA model from its hugging face repository

 

 

Minimax H3 Latent Upscaler

MiniMax H3 Latent Upscaler is designed to increase the spatial resolution of MiniMax H3 video outputs while preserving the original temporal dimension. Trained on around 80,000 paired video and image
samples, it works directly with MiniMax H3’s 24-channel VAE latents to upscale the height and width of generated content while maintaining its motion and temporal structure.

Download MiniMax H3 Latent Upscaler model from its hugging face repository

 

 

MiniMax-H3 Anime Video Line Art Colorization lora

MiniMax-H3 Anime Video Line Art Colorization LoRA is designed to turn anime-style line art videos into detailed, colored visuals. It takes line art as the input and helps MiniMax H3 generate polished
anime scenes with richer colors and refined visual details while preserving the original line-art structure.

Download MiniMax-H3 Anime Video Line Art Colorization LoRA model from its hugging face repository

10 Eros Minimax H3

You obviously know 10 eros' models are famous for what. 10 Eros MiniMax H3 is a custom MiniMax H3 base model that combines learned patterns from LTX 2.3 and Wan 2.2 video diffusion models. It is
intended to work best with image-to-video (I2V) or reference-based workflows without additional LoRAs. For the best results, the model recommends following the official MiniMax H3 prompting style
when writing prompts.

Download 10 Eros MiniMax H3 base model from its hugging face repository

Minimax H3 Realism People LoRA

MiniMax H3 Realism People LoRA is designed to make human characters look more realistic and natural in generated videos. It focuses on details such as convincing facial features, natural skin
texture, realistic expressions and gestures, along with film-style lighting and documentary-inspired camera movement.

Download MiniMax H3 Realism People LoRA model from its hugging face repository

 

 

MiniMax H3 Acceleration LoRA

MiniMax H3 Acceleration LoRA is designed to speed up MiniMax H3 video generation for both Text-to-Video (T2V) and Image-to-Video (I2V) workflows. It targets H3’s joint audio-video diffusion process
and uses a fixed 4-step Euler inference setup, allowing faster generation with fewer sampling steps.

Download MiniMax H3 Acceleration LoRA model from its hugging face repository

MiniMax-H3 Audio-Video acceleration LoRA

MiniMax-H3 Audio-Video Acceleration LoRA speeds up the model's audio-video generation by reducing the standard 20-NFE process to a fixed 8-NFE Euler workflow. It is designed to maintain MiniMax-H3’s
native stereo audio, dialogue, and synchronized sound effects, while requiring fewer inference steps for faster generation.

Download  MiniMax-H3 Audio-Video Acceleration LoRA model from Hugging face repository.
