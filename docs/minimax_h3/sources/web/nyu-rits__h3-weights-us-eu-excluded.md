# Source: https://rits.shanghai.nyu.edu/ai/minimax-ships-h3-weights-with-the-us-and-eu-excluded/

Fetched: 2026-11-28 (curl + w3m text dump)

---

  •  
  •  
  •  
  •  
  •  
  •  

NYU Around the World

  • New York Shanghai Abu Dhabi
  • Accra Berlin Buenos Aires Florence London Madrid
  • ParisPragueSydneyTel AvivWashington

    Schools
    Arts and Science College of Arts and Science College of Dentistry College of Global Public Health Rory Meyers College of Nursing Courant Institute of Mathematical Sciences Gallatin School of
    Individualized Study Graduate School of Arts and Science Institute of Fine Arts Institute for the Study of the Ancient World Leonard N.Stern School of Business Liberal Studies Program Robert
    F.Wagner Graduate School School of Law School of Medicine School of Professional Studies Silver School of Social Work Steinhardt School Tandon School of Engineering Tisch School of the Arts

[svg]
NYUSH Library Logo

  • RITS Home
  • News
  • Contact
  • About RITS

 1. NYU Shanghai Library (opens in a new tab)
 2. RITS Home
 3. *

AdvancedAI

MiniMax Ships H3 Weights — With the US and EU Excluded

August 3, 2026·

Provided by Utku Ege Tuluk (opens in a new tab)

  • Video
  • china
  • license
  • minimax
  • open weights
  • quantization
  • text-to-video

MiniMax published the MiniMax-H3 weights to Hugging Face on August 3, 2026 — three days after announcing the omni-modal video model and promising an open release “in the coming days.” The checkpoints
are real, the quantized build is small enough to matter, and ComfyUI shipped day-0 support. But the accompanying MiniMax H3 Community License carves out four jurisdictions from its grant of rights,
and the United States is one of them.

[svg]
Three-stage MiniMax H3 system diagram: raw multimodal instructions feed H3-Context-IR to produce a structured context representation, which drives H3-Base to generate 768p video, which
H3-Regenerate-2K then upscales to 2K using context guidance.

Image credit: MiniMaxAI/MiniMax-H3 on Hugging Face

What Actually Shipped

H3 generates 4–15 second clips at up to 2K resolution, 24 FPS, with native 32 kHz stereo audio — voice, music, and sound effects generated jointly with the video rather than dubbed on afterward. It
accepts text, images, video, and audio as reference input in a single request, across six aspect ratios and eleven languages.

Two inference variants are released. FL2VA handles text-to-video and first/last-frame conditioning with zero to two images. Ref2VA is the omni-reference mode: up to nine images, three videos, and
three audio clips, capped at twelve files total.

The Architecture

The system is three stages, not one model. H3-Context-IR interprets free-form multimodal input and emits a structured context representation. H3-Base generates at 768p. H3-Regenerate-2K then feeds
that low-resolution result back through the base model in-context to reach 2K — an in-context regeneration pass rather than a bolt-on super-resolution network, which is how MiniMax says it preserves
small text and fine product detail.

[svg]
Detailed H3-Base architecture diagram showing condition encoding through the H3 Encoder, Visual VAE Encoder and Audio VAE Encoder; a packed in-context sequence of condition tokens and noisy
generation targets; a 33B dense single-stream H3 Omni Transformer with a shared DiT backbone repeated 50 times; and decoding to synchronized video and stereo audio.

Image credit: MiniMaxAI/MiniMax-H3 on Hugging Face

H3-Base is a 33B dense, single-stream transformer — a shared DiT backbone repeated 50 times, performing joint video–audio denoising on one packed sequence. Text conditioning comes from layer-50
features of Qwen3-VL-32B. The visual VAE compresses 16× spatially and 4× temporally; the audio VAE reduces 32 kHz stereo to 40 Hz tokens per channel. Roughly 13B of the 33B parameters sit in AdaLN
modulation branches, and because those outputs can be precomputed and cached, they never need to be loaded for inference-only deployment.

What It Takes to Run

That AdaLN trick is the load-bearing optimization. MiniMax replaced the modulation weights — about 40% of parameters — with lookup tables, applied int8 convrot quantization to the shipped weights,
and added custom kernels plus dynamic VRAM offloading.

[svg]
Bar chart of MiniMax H3 component sizes on disk by precision. FL2VA and Ref2VA transformers are 66.3 GB at bf16, 34 GB at int8 convrot, and 21 GB pruned. The text encoder is 51.5 GB at bf16, 27.1 GB
at int8, and 15.7 GB at nvfp4 AWQ. The video VAE is 5.21 GB at fp16 and the audio VAE 605 MB at fp32.

Image credit: ComfyUI Blog

The arithmetic works out to 123.6 GB across all components at native precision, down to 42.5 GB choosing the smallest variant of each — a 66% cut. ComfyUI claims this puts H3 within reach of a 12 GB
RTX 3060. Worth reading carefully: 42.5 GB of weights does not fit in 12 GB of VRAM, so that claim rests entirely on dynamic offloading, and the throughput cost of streaming weights on a consumer
card is not something the day-0 post quantifies. SGLang, vLLM, and Diffusers are also supported, with SGLang recommending four GPUs and Ulysses parallelism.

On Artificial Analysis, H3 ranks #1 in video editing, #2 in text-to-video at 1241.5 Elo (3.3 points behind Gemini Omni Flash), and #3 in image-to-video behind Seedance 2.0 and Gemini Omni Flash.
Those are early figures — human-preference arenas need weeks of blind votes to stabilize, so treat them as indicative rather than settled.

The License Is the Story

H3’s weights ship under the MiniMax H3 Community License, which is not an open-source license in the OSI sense. Two clauses stand out. Commercial users whose products generate more than US$20 million
in yearly revenue must obtain separate written authorization from MiniMax. And the grant is bounded by an “Applicable Territory” defined as worldwide excluding the European Union, the United Kingdom,
the Republic of Korea, and the United States of America.

That exclusion is new. The license on MiniMax M3, released in June, contains no territorial clause at all. The company also has recent history here: after publishing M2.7’s weights in April 2026, it
revised the terms shortly afterward to require written authorization for commercial use, drawing criticism for labeling the result “Modified-MIT.” Anyone building on H3 should read the LICENSE file
themselves rather than inferring terms from the “open weights” framing — and should expect the text to be a moving target.

The practical upshot is a genuine capability release with a genuinely narrow legal footprint. For researchers in most of the world, H3 is the strongest openly downloadable video model available. For
anyone in Brussels, London, Seoul, or New York, the weights are on Hugging Face and the license says the grant does not reach them.

This post was drafted with AI assistance and reviewed by RITS staff.

Sources

  • MiniMaxAI/MiniMax-H3 — model card, architecture, and deployment guide (Hugging Face)
  • MiniMax H3 Community License (full text)
  • MiniMax H3 Day-0 Support in ComfyUI: Open Weights, Native Audio, and 2K Video
  • MiniMax Releases MiniMax H3: An Omni-Modal Video Model (MarkTechPost)
  • MiniMax Drops State-of-the-Art AI Agent Model — Then Quietly Changes the License (Decrypt)
  • MiniMax M3 license, for comparison

On this page

 1. What Actually Shipped
 2. The Architecture
 3. What It Takes to Run
 4. The License Is the Story

Key points

  • MiniMax published the MiniMax-H3 weights to Hugging Face on August 3, 2026, three days after announcing the omni-modal video model.
  • H3 generates 4–15 second clips at up to 2K resolution and 24 FPS with native 32 kHz stereo audio generated jointly with the video.
  • The MiniMax H3 Community License defines an Applicable Territory that excludes the European Union, the United Kingdom, the Republic of Korea, and the United States of America.
  • Commercial users whose products generate more than US$20 million in yearly revenue must obtain separate written authorization from MiniMax.
  • The weights total 123.6 GB across all components at native precision, reduced to 42.5 GB by choosing the smallest variant of each.

Related stories

[minimax-h3]

chinaJuly 31, 2026

MiniMax Releases H3: 2K Video With Native Audio, Open Weights Promised

MiniMax released H3 on July 30, 2026 — a general-purpose multimodal video model that generates clips of up to 15 seconds at native 2K resolution with synchronized stereo audio, and takes text,
images, video, and audio as reference inputs in a single request. The Shanghai company says it will publish H3’s weights “within days,” which […]

[minimax-m3]

agentic aiJune 1, 2026

MiniMax M3: Frontier Coding, 1M Context, and Sparse Attention

MiniMax released M3 on June 1, 2026, claiming the first open-weight model to combine frontier coding, a 1-million-token context window, and native multimodality in a single architecture. The headline
is not just the benchmark sheet — it is the engine underneath: a new sparse-attention mechanism (MSA) that the company says cuts per-token compute at 1M […]

[huang-open]

AIJuly 28, 2026

Jensen Huang’s First X Post: 150+ Companies Sign Open-Weights Letter, Anthropic Doesn’t

On July 24, 2026, NVIDIA CEO Jensen Huang made his first-ever post on X — and used it to publish an industry coalition letter titled “Open Weights and American AI Leadership.” The three-page
statement, hosted on NVIDIA’s own servers and mirrored on Microsoft’s corporate site, asks Washington not to restrict downloadable model weights. It launched […]

Previous storyMiniMax Releases H3: 2K Video With Native Audio, Open Weights PromisedNext storyNVIDIA Releases NemotronLabs VoiceChat, an Open Full-Duplex Voice Model

On this page

 1. What Actually Shipped
 2. The Architecture
 3. What It Takes to Run
 4. The License Is the Story

What Actually Shipped

(Previous section)‹(Next section)›(Open on this page)Contents

Never miss a RITS update

Subscribe to the RSS feed for AI, emerging technology, and digital scholarship news curated for the NYU Shanghai community, or get in touch to suggest a story.

Subscribe to RSS (opens in a new tab)Contact RITS
[svg]
Footer Logo

Connect with NYU Shanghai

  •  
  •  
  •  
  •  
  •  

  • Portal Campuses
  • NYU
  • Shanghai
  • Abu Dhabi

  • Get in Touch
  • Contact Us
  • Campus Tour
  • Jobs

  • Quicklinks
  • Students
  • Accessibility
  • Website Feedback

New York University — Equal Opportunity and Non-Discrimination at NYU. New York University is committed to maintaining an environment that encourages and fosters respect for individual values and
appropriate conduct among all persons. In all University spaces—physical and digital—programming, activities, and events are carried out in accordance with applicable law as well as University
policy, which includes but is not limited to its Non-Discrimination and Anti-Harassment Policy.

沪ICP备15010846号-2●沪公网安备31011502017015号

© 2025 All rights reserved.Powered by NYU Shanghai RITS
