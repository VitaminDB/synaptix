# Source: https://dev.to/tf6cool/i-compared-minimax-h3-runs-from-6gb-to-20gb-vram-heres-what-i-learned-2jpj

Fetched: 2026-11-28 (curl + w3m text dump)

---


Skip to content
DEV Community
[                    ] (Search) Powered by Algolia
Log in Create account

DEV Community

(Close)
(reaction-drawer-trigger) ● Add reaction
(Like) [spa] Like (Unicorn) [mul] Unicorn (Exploding Head) [exp] Exploding Head (Raised Hands) [rai] Raised Hands (Fire) [fir] Fire
(Jump to Comments) Jump to Comments (Add to reading list) Save (Boost) Boost
(Pick as gem) Pick as gem
(Share post options)
Copy link
Copied to Clipboard
Share to X Share to LinkedIn Share to Facebook Share to Mastodon
Share Post via... Report Abuse
TF6cool
TF6cool

Posted on Aug 10 • Edited on Aug 11

I Compared MiniMax H3 Runs from 6GB to 20GB VRAM — Here’s What I Learned

#medium #ai #machinelearning #comfyui

When MiniMax H3 became available as an open-weight model, my first question was probably the same as yours:

Can I actually run this locally on my GPU?

I expected to find a simple minimum VRAM number. Instead, I found reports that seemed to contradict each other.

One person could run H3 with 8GB of VRAM. Another ran out of memory with 12GB. Some RTX 3060 users reported five-second generations in under ten minutes, while others needed much longer.

After reading through the details, the reason became clear: the GPU model alone does not tell us enough.

 Why “minimum VRAM” is misleading

Two MiniMax H3 runs are only comparable when they use roughly the same:

  • Model and weight precision
  • Text encoder
  • Workflow type
  • Resolution
  • Frame count and FPS
  • Number of sampling steps
  • Cache settings
  • Audio settings
  • System RAM
  • Offloading strategy
  • ComfyUI and dependency versions

A five-second clip at 0.4 megapixels is not equivalent to a ten-second clip at a higher resolution. An INT8 diffusion model with a quantized text encoder is also very different from loading the full
weights.

This is why statements such as “H3 needs 12GB” or “H3 works on 6GB” should be treated as starting points, not hard limits.

 What the public reports show so far

I organized the community reports I could find by GPU, RAM, model variant, workflow and generation time.

These are community-reported results, not controlled benchmarks performed by me.

 6GB VRAM

There are individual reports of H3 completing short jobs on 6GB cards.

However, the evidence is currently limited. Some results appear only in comments and omit important settings.

I would not describe 6GB as generally supported based on the available evidence.

 8GB VRAM

One of the more detailed reports used:

  • RTX 4060 Ti 8GB
  • 32GB system RAM
  • Pruned INT8 diffusion model
  • NVFP4 text encoder
  • 640p
  • Five-second output

The author reported approximately 20 minutes for a cold run and around 12 minutes with EasyCache, running at 15 steps.

That proves one carefully configured 8GB setup can work. It does not prove that every 8GB GPU or workflow will succeed.

 12GB VRAM

The strongest collection of consumer-GPU reports currently comes from the RTX 3060 12GB.

Several users with 32GB of system RAM reported short generations around:

  • 0.4 megapixels or 864×480
  • Five seconds
  • 15–20 steps
  • Approximately 6–10 minutes

Other 12GB workflows took considerably longer, particularly when system RAM, duration, resolution or steps increased.

My cautious interpretation is that 12GB is currently the clearest practical starting point for consumer GPUs.

 16GB VRAM

A detailed RTX 4070 Ti SUPER report used an INT8/NVFP4 image-to-video setup with 20 steps and audio enabled.

Five reported runs took between roughly 2 and 7 minutes, depending on the input size and output resolution.

This is stronger evidence because the author provided the software environment and a public workflow file.

 20GB VRAM

An RX 7900 XT user reported completing a 1056×608, ten-second image-to-video run in approximately 10 minutes.

The run completed, but it also shows why additional VRAM does not automatically mean faster inference. System RAM and drivers still play a huge role.

 A conservative first-run profile

If I were setting up MiniMax H3 on a 12GB consumer GPU, I would begin with:

  • A recent ComfyUI build
  • An official T2V workflow
  • A supported pruned or quantized model
  • Approximately 0.4–0.5 megapixels
  • About five seconds
  • 15–20 steps
  • Audio disabled
  • At least 32GB of system RAM if possible

The goal of the first run is not maximum quality. It is to establish a clean baseline that successfully completes.

After that, change one variable at a time.

If you immediately add a custom workflow, LoRA, cache system, audio generation and higher resolution, troubleshooting a crash becomes extremely difficult.

 Lowering resolution does not fix every OOM

Resolution matters, but an out-of-memory error can happen before sampling begins.

Possible failure points include:

  • Loading the text encoder
  • Loading or converting the diffusion model
  • Moving from the encoder to the diffusion stage
  • Pinned-memory behavior
  • CPU offloading
  • Insufficient system RAM
  • Loading the video or audio VAE

If the workflow fails before the sampler starts, reducing the sampler resolution may not address the bottleneck.

Watch both VRAM and system RAM, and record the stage at which memory usage spikes.

 What a useful community report should include

A reproducible H3 report should contain something like this:

GPU and VRAM:
System RAM:
Operating system:
GPU driver:
ComfyUI version or commit:
Python / PyTorch / CUDA or ROCm:
Workflow type:
Exact diffusion model:
Exact text encoder:
VAE files:
Resolution:
Frames and FPS:
Steps:
Sampler and scheduler:
Audio enabled:
Cache or acceleration:
Cold or warm run:
Generation time:
Peak VRAM:
Peak system RAM:

Without these details, “it works on my GPU” is useful encouragement, but weak technical evidence.

 The guide I built

I turned my notes into a free MiniMax H3 VRAM configuration and setup guide with the configurations separated by evidence quality.

Full disclosure: this is my own independent site. There is no signup, paywall or affiliate link. I built it to keep track of these settings in one place.

I am still treating this as a living evidence table, not a finished benchmark database.

 One final license note

MiniMax H3 is available as open weights, but that does not automatically mean unrestricted open-source use.

The model has a Community License with territory and usage conditions. Check the current official repository for details.

 Corrections are welcome

If you have run MiniMax H3 locally, I would be interested in your exact configuration—especially if you have managed to get stable runs on 6GB or 8GB cards.

Please include the model variant, workflow, resolution, frames, steps, RAM and timing. I would rather update the MiniMax H3 VRAM evidence table with real community data than leave gaps in the guide.

Top comments (0)

Subscribe
pic
