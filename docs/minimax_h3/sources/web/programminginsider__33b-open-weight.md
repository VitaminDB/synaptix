# Source: https://programminginsider.com/minimax-h3-open-weight-release-gives-developers-a-33-billion-parameter-video-model-with-full-api-access/

Fetched: 2026-11-28 (curl + w3m text dump)

---


  • 
    Marc Berman
  • 
    September 02, 2026

Share
Facebook
Twitter
Linkedin
Email
Threads
Bluesky

The open-weight video model space has been notably behind the open-weight LLM ecosystem. While Llama, Mistral, and Qwen have given developers competitive alternatives to GPT-4 and Claude for text, no
open-weight video model has come close to matching the output quality of closed-source systems like Sora 2, Veo 3.1, or Runway Gen-4.5. Until now.

MiniMax published the weights for H3, its 33-billion-parameter omni-modal video generation model, on Hugging Face on August 3, 2026 — three days after the model’s public debut at the World Artificial
Intelligence Conference in Shanghai. On the independent Artificial Analysis benchmarks, H3 holds first place in video editing, second in text-to-video, and third in image-to-video. Those are not
open-weight-only rankings — H3 is being compared against the full field including closed-source models.

For developers, the release creates a new option in the stack: a video generation model that can be deployed locally, accessed through a hosted API, or used in a hybrid configuration — with benchmark
performance that actually competes at the frontier.

Architecture Overview

H3 is built around four subsystems, and understanding how they partition the work matters for deployment decisions.

H3-Context-IR handles input preprocessing. When a user submits a generation request with multiple reference files (up to nine images, three video clips, three audio files), Context-IR ingests all
inputs, models the relationships between them, and compresses approximately 100,000 tokens of source material down to about 4,000 tokens of structured context representation. This is the system that
understands “use the face from @image1 and the camera movement from @video2” — it does not just catalog the inputs but models how they relate to the target output.

H3-Base is the core generation model. It encodes different modalities using their respective encoders and VAEs, organizes the encoded representations into a unified packed multimodal sequence, and
applies Rotary Position Embedding (RoPE) for spatial and temporal relationships. Base generates at 768p resolution.

H3-VAE is a temporally causal video autoencoder with 16× spatial compression, 4× temporal compression, and 24 latent channels. A separate Audio VAE handles the stereo audio stream. Both run at
inference time to decode the transformer’s output into pixel and waveform space.

H3-Regenerate-2K is the upscaling module that takes 768p base output to full 2K resolution (2560×1440 at 16:9) while preserving temporal coherence.

The open-weight release includes H3-Base and H3-Regenerate-2K as two task-specific checkpoints. Each checkpoint bundles the Omni Transformer, processor, tokenizer, text encoder, Visual VAE, and
standalone Audio VAE. H3-Context-IR is not included in the weight release — it remains available only through MiniMax’s API.

Local Deployment Requirements

The BF16 base model checkpoint requires approximately 134 GiB of model weights for a single task partition. That is before activation memory and runtime overhead. In practice, you need multiple
high-end GPUs (A100 80GB or H100 class) to run inference.

MiniMax provides two deployment paths in the documentation:

Local H3-Base only generates at 768p. This is the fully self-contained option — no API dependency. You run inference locally with your own preprocessing pipeline (MiniMax provides a prompting
guidance document for building one) and get 768p output with native audio.

Full 2K Workflow combines local H3-Base inference with MiniMax’s hosted H3-Context-IR API for input preprocessing and the H3-Regenerate-2K module for upscaling. This hybrid approach gives you full 2K
output but requires API calls for the preprocessing and upscaling stages.

For a comprehensive overview of available deployment paths and usage modes, minimax h3 provides a structured walkthrough of the model’s capabilities.

The Hugging Face repository follows standard conventions: `model_index.json`, `/processor`, `/tokenizer`, `/text_encoder`, and the Visual and Audio VAE components. The model is CFG-distilled, which
means it uses Classifier-Free Guidance distillation for faster inference compared to non-distilled approaches.

API Workflow

The hosted API follows an asynchronous three-step pattern:

1.Create a task — Submit your prompt, reference files, and generation parameters (resolution, duration, aspect ratio) to the task creation endpoint. The API returns a `task_id`.

2.Poll for status — Query the task status endpoint with your `task_id`. Generation times vary by resolution and clip length.

3.Download the result — When the task completes, the response includes a `content.url` pointing to the generated video file.

The API supports three generation modes through a single endpoint: text-to-video (prompt only), first-and-last-frame image-to-video (two images define the start and end states), and omni-reference
generation (up to 12 mixed reference files with natural-language role descriptions).

Input specifications: prompts up to 7,000 characters, reference images up to 9, reference videos up to 3 clips of 2–15 seconds each (15 seconds total maximum), reference audio up to 3 files, 12 files
total per request. Output: 4–15 seconds in integer increments, 24 fps, 32 kHz stereo audio, six aspect ratios (21:9, 16:9, 4:3, 1:1, 3:4, 9:16).

Licensing Constraints

The MiniMax H3 Community License is not a permissive open-source license. Developers should read the full terms before integrating H3 into production systems. Notably, some reviewers have flagged
that the license appears to contain territorial restrictions — it may prohibit use in certain jurisdictions. The license does not follow MIT, Apache 2.0, or similar patterns that developers in the
open-source ecosystem typically expect.

This is a meaningful consideration for commercial deployment. While the weights are publicly available and free to download, “open weight” does not mean “use however you want.” Review the license,
check the model card, and consult your legal team before building production pipelines.

Benchmark Context

On the Artificial Analysis leaderboards, H3’s positioning is instructive:

First place in video editing means that H3 handles reference-driven generation, motion transfer, and compositional editing better than any other model tested — including closed-source alternatives.
This matters most for production workflows where the goal is not generating from scratch but compositing, refining, and iterating on existing assets.

Second in text-to-video means H3 is not the absolute quality leader for raw prompt-to-footage generation. Veo 3.1 produces more cinematically polished output, particularly at 4K. Sora 2 has stronger
physics simulation and supports longer clips (up to 25 seconds).

Third in image-to-video puts H3 behind the leaders for single-image animation but still in the competitive tier.

The aggregate picture: H3 is the best generalist, not the best specialist. It does more things at a high level than any single competitor, which makes it the natural choice for default deployment in
a multi-model stack.

Where H3 Fits in a Production Stack

The practical recommendation from independent testers is a multi-model approach:

Use H3 as the default model for volume generation — it handles the widest range of input types at the best cost. Switch to Veo 3.1 for hero shots requiring maximum cinematic polish at 4K. Keep Kling
3.0 available for specialized motion transfer tasks. Use Runway Gen-4.5 when you need iterative timeline-based editing rather than one-shot generation.

For developers building video generation into an application, H3’s async API and multi-reference input make it the most flexible backend option. The ability to accept twelve reference files through a
single endpoint means you can build complex generation UIs — character selectors, camera movement presets, audio mood boards — without managing multiple API calls to different specialized models.

The pricing reinforces this positioning. At 2K resolution, H3’s per-second cost is less than one-third of mainstream closed-source alternatives. For applications that generate at volume — product
catalogs, ad variant systems, dynamic content platforms — that cost difference is a significant operational advantage. More details on API and subscription pricing are available at minimaxh3kr.com .

Company Context

MiniMax was founded in 2021 by Yan Junjie, a former deputy director at SenseTime. The company is listed on the Hong Kong Stock Exchange (0100.HK); shares surged over 10 percent following the H3
open-source announcement, closing at HKD $249.40.

The broader MiniMax product line includes the M-series LLMs (MiniMax M2.7 being the latest), Speech 2.8 for text-to-speech in 30+ languages, and Music 3.0 for AI music generation. H3 is designed to
interoperate with these models, which means a developer integrating H3 for video generation can potentially use the same vendor’s API for text, speech, and music in a unified content pipeline.

H3 is the third generation of MiniMax’s video models, following Hailuo 01 (foundational architecture) and Hailuo 02 (efficiency and data quality improvements). The model is also referred to as Hailuo
3.0 in consumer-facing contexts.

Getting Started

The Hugging Face repository is at `MiniMaxAI/MiniMax-H3`. The API documentation is on the MiniMax Open Platform. The consumer app is at hailuoai.video.

For developers who have been waiting for an open-weight video model that actually competes at the frontier — rather than being an academic exercise with interesting architecture but unusable output —
H3 is the first release that clears that bar. The deployment overhead is real (134 GiB minimum for the base model), the licensing requires legal review, and the ecosystem is nascent. But the
combination of benchmark performance, multi-modal input, and open weights makes H3 a defensible choice for any team evaluating video generation infrastructure in the second half of 2026.

[svg]

Newsletter
