# Source: https://www.instasd.com/post/minimax-h3-turbo-lora-comfyui-4-steps

Fetched: 2026-11-28 (curl + w3m text dump)

---

Pricing
Workflows
Case Studies
FAQ
[6748d5aba7]
Discord
Start
Start
Saas Webflow Template - Bolton - Designed by Azwedo.com and Wedoflow.com

Workflows

MiniMax H3 Turbo LoRA in ComfyUI: 4 Steps, and the Three Things That Make It Look Broken

A community 4-step Turbo LoRA for MiniMax H3 landed three days after the open weights did. Half the thread says 50% faster, half says blurry mess and broken audio. Both are right, and the repo README
explains why.

Two brass metronomes on a dark desk with their pendulums swinging at different angles

MiniMax put H3's open weights on Hugging Face on Monday. By Thursday somebody had shipped a 4-step Turbo LoRA for it. Most models do not get a working GGUF that fast.

I read the whole 186-comment thread where it landed on r/StableDiffusion this morning, and the split in it is stark. Half the replies are some version of "cut my generation time in half, this is
incredible." The other half are "my output is a blurry mess," audio that comes out distorted and blown out, and one person who measured the speedup at barely 10% and gave up.

Both halves are telling the truth. Almost every complaint in that thread has a cause written down in the repo README, and almost nobody in the thread had opened it.

What you are actually installing

The LoRA is larryvrh's, and the current file is minimax_h3_turbo_4step_ema_ckpt850.safetensors. It is bf16, about 744 MB, applied as an ordinary low-rank update with alpha set equal to rank, so there
is no hidden scaling factor to reason about. Apache-2.0.

Worth separating those two things. The LoRA is Apache-2.0. MiniMax H3 itself is not. It ships under the MiniMax H3 Community License Agreement, and one person in the team's AMA said he reads the
terms as blocking him from even testing it locally in Europe. Read it before you build a product on this.

The model card claims 4 sampling steps instead of the usual 20, which it puts at roughly a 5x speedup in sampling wall-clock. That number is honest as far as sampling goes. It is also not what most
people will measure, and I will get to why.

The sentence people skip is the one that matters: ckpt850 is the final checkpoint of this training round, and training is paused. Not finished. Paused, because pushing sharpness this far started
producing plastic-looking skin and over-sharp grain. larryvrh says so on the model card, above the fold. A lot of the "quality dropped" reports in the thread are people rediscovering a known issue.

The audio breaks because H3 runs two clocks

This is the failure everyone hits first and it has the most interesting explanation.

H3 does not generate video and then dub it. It denoises video and synchronized audio together in one attention sequence, and the two streams ride different flow schedules: video on a shift of 12,
audio on a shift of 3. ComfyUI's stock samplers step both streams on one schedule. At 20 steps that mismatch is small enough to disappear. At 4 steps it badly over-steps the audio, which is exactly
the distortion people are describing.

Two brass metronomes on a dark desk with their pendulums swinging at different angles
Two streams, two tempos. A stock sampler runs both metronomes off one beat, which only stays convincing while the steps are small.

So the audio is not a bug in the LoRA. It is what happens when you cut the step count on a sampler that was quietly getting away with an approximation.

Two fixes exist and you need one of them.

The first is larryvrh's own node package, ComfyUI-MiniMax-H3-Turbo, which ships a MiniMax-H3 Turbo Sampler that steps each stream on its own clock. Drop it in place of the sampler feeding
SamplerCustomAdvanced, set BasicScheduler to 4 steps, leave the scheduler on simple. There is a ready-made t2v workflow in the repo.

The second is native. Kijai opened PR #15243 against ComfyUI, "Fix sampler issues for audio with minimax, support more samplers," and comfyanonymous merged it. The approach is cleaner than a bespoke
sampler. Because the two schedules are the same curve stretched by a constant factor, he carries the audio latent at a scale that puts it on the video's schedule, so the sampler only ever sees one
ordinary flow latent and is already correct on it. The model divides the scale back out before the network, so the DiT still gets audio at the sigma it was trained on. Side effect worth having:
stochastic and ancestral samplers work now.

One catch, and it matters today. That PR is merged to master but it is not in v0.30.0, the current tagged release, whose changelog stops at #15239. If you are on stable you do not have this yet. Run
master, or use the custom node.

Kijai flags something in the PR that nobody in the Reddit thread mentions and you should know before you re-run old seeds. Output changes for H3 at every step count, not just low ones. Audio is the
point, but video moves too, because the streams denoise in one attention sequence and the audio trajectory feeds back into the video. Your 20-step results will not reproduce byte-for-byte after this
lands.

Blurry or plastic: it is one dial

The second cluster of complaints splits neatly in two. Some people get soft, smeary, ghosted output. Others get output that is sharp in an ugly, over-crisp, grainy way.

Those are the two ends of the same control. LoRA strength defaults to 1.0. Blurry ghosting means push it up, somewhere around 1.05 to 1.2. Over-sharp grain and artifacts mean pull it down, 0.8 to
0.95. That is the whole troubleshooting procedure, it is in the README, and I did not see it repeated once in 186 comments. What circulated instead was a set of step counts and sigma shifts copied
out of a screenshot.

Some of that circulating advice is stale on top of being incomplete. The thread's own instructions recommend 8 to 10 steps on EMA and 6 to 8 on ckpt500. Those were right for earlier checkpoints. With
ckpt850 the repo says 4 is already sharp. The person who posted the thread later confirmed in a reply that his numbers came from a night of testing on bf16 before larryvrh's custom node existed at
all. Anything at or above 4 works and more steps still help a little, but at 10 you are leaving most of the speedup on the table.

"I measured no speedup"

Two separate things cause this and the fixes are different.

The first is the low_vram switch on the LoRA node. Off by default, it applies the LoRA at runtime as a bypass, which is sharpest. Turned on, it merges the LoRA into the weights instead, which drops
peak VRAM enough to fit longer clips on smaller cards. On a quantized base, meaning int8, fp8, or any of the pruned variants, merging a small low-rank update back into already-quantized weights
rounds much of it away. You keep the memory saving and lose most of the effect. Someone in the thread reported no change in speed at all on pruned int8, which is exactly the configuration where this
would bite.

The second is more common and slightly embarrassing: the 20-step baseline is not what you were running. jib_reddit says he runs H3 at 10 steps all the time without any speed LoRA, and several people
replied that they do the same. If your real baseline was 10 steps, going to 4 is 2.5x on sampling, not 5x. Then subtract text encoding, VAE decode and model loading, which the LoRA does nothing for.
That is how you get someone honestly measuring 10%.

The most useful contrary data point comes from infearia, who reports 10 steps with EasyCache at (0.3, 0.2, 0.9) still beating the Turbo path on both speed and quality. Worth taking seriously rather
than arguing with, and consistent with what I found looking at EasyCache, TeaCache and Spectrum yesterday. These methods interact, and the winner depends on your base and your card.

Which leads to the one rule the thread's poster repeats in half a dozen replies: do not put a cache node on top of Turbo. At 4 steps there is nothing left to cache, because adjacent steps are no
longer similar enough for the heuristic to hold. He reports SageAttention, Sol Attention and Gradient all stacking fine. Cache nodes do not.

What MiniMax said about an official version

The team ran an AMA on r/StableDiffusion yesterday. "Please release a turbo LoRA" was one of the most upvoted questions in it, asked three or four different ways, and as of this morning it has no
answer.

They did answer the 2K question. Asked about releasing the 2K regenerate model for a final pass, their head of devrel replied "yep, not very far."

There is a plausible reason an official distill is harder than it looks. Several people in that AMA describe the released H3 weights as already CFG distilled, and asked for the un-distilled base
specifically to make LoRA training tractable. If that is right, training a step distillation on top of a guidance-distilled model stacks one lossy compression on another, and the plastic skin
larryvrh hit at ckpt850 is roughly what you would predict.

If you want to try it today

Grab the EMA ckpt850 file, install larryvrh's node from ComfyUI-Manager, start from the official H3 workflow, and change exactly two things. Insert the Turbo LoRA node between the model loader and
the sampler. Swap in the Turbo Sampler at 4 steps with the scheduler on simple. Keep width and height as multiples of 32 with the short edge at 768. Frame count runs at 24 fps and snaps to a 17k+5
grid, so 124 frames is your 5 seconds, and the validated range tops out at 362.

Then time your actual baseline before you claim a number.

The awkward part is the base model. H3 is around 33B parameters. People are running it on a 12 GB 3080, and on a 16 GB 4060 Ti with 64 GB of system RAM, and it genuinely works, but they are streaming
weights and waiting. One person in the AMA reported 15 seconds of video in forty minutes on a 6 GB laptop card at 0.2 megapixels. A 4-step LoRA does not fix that. It makes a slow thing four times
less slow.

If you would rather not spend a weekend on it, that gap is the reason running ComfyUI online exists: same graph, same nodes, a GPU that actually fits the model. And if you want H3-class output
without touching a node graph at all, Wundernode puts 80+ models on a canvas with a free tier and nothing to install.

I will come back to this when ckpt850's successor lands or MiniMax ships an official distill. Given the pace of this week, that could be Tuesday.

Other Posts

ComfyUI workflow to single page app

Ideogram 4.0 in ComfyUI: JSON Prompts, Local Weights and P-Image

Ideogram shipped open weights you can run locally and a hosted P-Image API node into ComfyUI within a week, and both take the same structured JSON caption. Read from the node source: the full schema,
the style dropdown that silently discards your work, y-first bounding boxes on a 0-1000 grid, P-Image pricing per tier, and how to reproduce an exact image.

Generative AI University
August 3, 2026
ComfyUI workflow to single page app

One GPU to Rule Them All: HiDream Performance Testing Across GPUs

In this edition of our "One GPU to Rule Them All" series, we benchmark HiDream, one of the newest and most advanced image generation models available. Released with support for multiple precision
formats and step counts, HiDream offers a balance between visual quality and generation speed—making GPU selection more important than ever. We tested six HiDream variants across six GPUs to evaluate
both generation speed and VRAM usage, identifying the ideal setup for different needs.

One GPU to Rule Them All
April 19, 2025
ComfyUI workflow to single page app

Introduction to EasyAnimate v5.1 with ComfyUI: AI-Powered Video Generation for Creators

EasyAnimate v5.1 is an advanced AI model designed for video generation, offering creators powerful tools to transform their ideas into dynamic animations. Whether you're creating videos from text
prompts, animating static images, or fine-tuning output with various control models, EasyAnimate provides a flexible platform for both enthusiasts and professionals. With support for ComfyUI,
EasyAnimate is a must-try for those exploring the frontiers of AI-driven animation.

Workflows
January 29, 2025
InstaSD
