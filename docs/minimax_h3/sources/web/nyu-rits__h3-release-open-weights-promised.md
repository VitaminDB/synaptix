# Source: https://rits.shanghai.nyu.edu/ai/minimax-releases-h3-2k-video-with-native-audio-open-weights-promised/

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

IntermediateAI

MiniMax Releases H3: 2K Video With Native Audio, Open Weights Promised

July 31, 2026·

Provided by Utku Ege Tuluk (opens in a new tab)

  • Video
  • china
  • genai
  • minimax
  • open source
  • text-to-video

MiniMax released H3 on July 30, 2026 — a general-purpose multimodal video model that generates clips of up to 15 seconds at native 2K resolution with synchronized stereo audio, and takes text,
images, video, and audio as reference inputs in a single request. The Shanghai company says it will publish H3’s weights “within days,” which would make it one of the first frontier-class video
generators to ship as open weights in a field where the leading systems have stayed closed.

[svg]
MiniMax H3 promotional banner reading 'Native Multimodal Understanding &amp; Generation — Create with text, images, audio, and video in one unified workflow'

Image credit: MiniMax

What H3 Does

H3 — marketed to consumers as Hailuo 3.0 — outputs native 2K video at 24fps in durations from 4 to 15 seconds, across aspect ratios from 21:9 through 9:16, with an Extend Video tool that pushes a
sequence to roughly 30 seconds. Audio is not a separate pass: dialogue, sound effects, music, and room ambience are generated alongside the frames, which is the difference between a model that
produces footage and one that produces a finished clip.

The feature MiniMax is leading with is omni-reference. A single generation request can carry up to 9 reference images, 3 reference video clips, and 3 reference audio clips — capped at 12 files total
— with each reference video and audio clip running 2 to 15 seconds and the combined reference duration limited to 15 seconds. Audio references cannot be submitted on their own; they must accompany
image or video content. In practice this means one prompt can pin down a character’s face, a product’s branding, a camera move borrowed from existing footage, and a speaker’s voice simultaneously.

[svg]
Summary card listing MiniMax H3 specifications: 2K/24fps native resolution, 5-15 second clips extendable to about 30 seconds, omni-reference accepting 9 images plus 3 video and 3 audio clips, and
one-pass synchronized audio

Image credit: OrcaRouter

The model also supports instruction-based editing — describing a change to characters, objects, scenes, sound, or pacing and having it applied without regenerating the clip from scratch — and
video-to-video motion transfer, which maps the movement in one clip onto new subjects. Reference input formats are WAV and MP3 for audio, with AAC or MP3 audio tracks in the returned video; reference
images and video must fall between 256 and 5760 pixels with aspect ratios between 2:5 and 5:2.

[svg]
MiniMax H3 promotional card reading 'Production-Ready Content Creation Across Use Cases — Replace and refine details with precise, reliable control'

Image credit: MiniMax

Cost, Access, and the Open-Weights Question

H3 is available now through MiniMax’s platform API under the model ID MiniMax-H3, and through third-party routers — OpenRouter lists it at minimax/hailuo-3 starting at $0.13 per second of generated
video. MiniMax told Reuters that H3 generates 2K video at less than one-third the cost of mainstream competing products, and that the model is designed to run on Chinese-made chips. Access is still
described as early, and several details remain unpublished: parameter count, architecture family, rate limits, license terms for generated content, and formal scores on standard video-generation
benchmarks.

That last gap matters. Every capability claim above traces back to MiniMax’s own documentation and marketing; H3 has not yet been independently arena-scored against its rivals. The weights themselves
are also still pending — as of publication they had not appeared on Hugging Face, so “open weights” remains a stated intention rather than a shipped artifact, and the license terms that will govern
them are unknown.

[svg]
MiniMax H3 promotional card reading 'Commercial-Ready Creation Across Use Cases — Built for advertising, gaming, branding, e-commerce, film, and more'

Image credit: MiniMax

What This Means

Video generation has been the most stubbornly closed corner of generative AI. Text models went open early and often; image models followed; but the systems at the top of the video leaderboards —
ByteDance’s Seedance 2.0, Kuaishou’s Kling 3.0, Google’s Veo 3.1 — have remained API-only products. OpenAI is moving in the opposite direction entirely, with its Sora API scheduled for
discontinuation at the end of September 2026. A downloadable H3 would be the first time researchers can inspect, fine-tune, and self-host a model in this tier.

The competitive framing is familiar from the text-model side: rather than beat closed rivals on raw fidelity, MiniMax is competing on cost, control, and openness. Kling 3.0 already offers native 4K,
and Seedance 2.0 leads audio-inclusive rankings — H3’s pitch is the omni-reference control surface, one-pass audio, and a price roughly a third of the alternatives. For the advertising, e-commerce,
product design, and games workflows MiniMax is targeting, reliable character and brand consistency across shots is often worth more than another resolution tier.

For students and researchers, the practical advice is to wait for the weights before drawing conclusions. If they arrive on the timeline MiniMax has stated, the interesting work begins immediately:
independent benchmarking against Kling and Seedance, and the fine-tuning and quantization ecosystem that formed around MiniMax’s earlier open-weight text releases arriving in video for the first
time.

This post was drafted with AI assistance and reviewed by RITS staff.

Sources

  • China’s MiniMax releases H3 video model — Eduardo Baptista, Reuters
  • MiniMax Platform — Video Generation guide (H3 model documentation)
  • MiniMax official site — H3 announcement
  • OpenRouter — MiniMax Hailuo 3 model page and pricing
  • MiniMax H3 (Hailuo 3.0): 2K AI Video, Explained — OrcaRouter
  • MiniMax H3 AI Video Generator: Native 2K Video With Audio — OpenArt

On this page

 1. What H3 Does
 2. Cost, Access, and the Open-Weights Question
 3. What This Means

Key points

  • MiniMax released H3 on July 30, 2026, a general-purpose multimodal video model that generates clips of up to 15 seconds at native 2K resolution with synchronized stereo audio, taking text,
    images, video, and audio as reference inputs in one request.
  • H3, marketed to consumers as Hailuo 3.0, outputs native 2K video at 24fps in durations from 4 to 15 seconds, with an Extend Video tool that pushes a sequence to roughly 30 seconds.
  • A single request can carry up to 9 reference images, 3 reference video clips, and 3 reference audio clips, capped at 12 files total, with the combined reference duration limited to 15 seconds and
    audio references unable to be submitted on their own.
  • MiniMax says it will publish H3’s weights within days, but as of publication the weights had not appeared on Hugging Face and the license terms governing them were unknown.
  • MiniMax told Reuters that H3 generates 2K video at less than one-third the cost of mainstream competing products and that the model is designed to run on Chinese-made chips.

Related stories

[minimax-m3]

agentic aiJune 1, 2026

MiniMax M3: Frontier Coding, 1M Context, and Sparse Attention

MiniMax released M3 on June 1, 2026, claiming the first open-weight model to combine frontier coding, a 1-million-token context window, and native multimodality in a single architecture. The headline
is not just the benchmark sheet — it is the engine underneath: a new sparse-attention mechanism (MSA) that the company says cuts per-token compute at 1M […]

[minimax-m2]

agentic aiApril 13, 2026

MiniMax M2.7 Ships as Open Weights: Frontier Agentic Model on Hugging Face

MiniMax M2.7 is now downloadable. The 230B-parameter self-evolving model that MiniMax unveiled in March has shipped as open weights on Hugging Face and ModelScope under a Modified-MIT license, and
community quantizations already cover everything from a 60 GB 1-bit build to the full 457 GB BF16 release — putting a frontier-class agentic model in reach […]

 

AIJuly 29, 2025

Wan2.2: Alibaba’s Open‑Source Breakthrough in AI Video Generation

The Wan2.2 project (hosted at Wan‑Video/Wan2.2) marks another leap forward in large‑scale, open, consumer‑accessible video generative models developed by the Wan‑AI team at Alibaba Cloud (GitHub).
Released on July 28, 2025, this upgrade offers major technical advancements over the previous Wan2.1 version (Hugging Face). Key points The Wan-AI team at Alibaba Cloud released Wan2.2 on […]

 

AIOctober 7, 2025

OpenAI Launches Sora 2: A New Frontier in AI Video Generation

OpenAI has officially unveiled Sora 2, its next-generation video and audio synthesis model, packaged into a brand-new Sora mobile app. (OpenAI) The system promises more realistic visuals,
synchronized audio, stronger control over content, and greater creative flexibility than its predecessor. (OpenAI) Key points OpenAI has officially unveiled Sora 2, its next-generation video and
audio synthesis […]

Previous storyMCP Apps: Interactive UIs Become an Official Part of MCPNext storyMiniMax Ships H3 Weights — With the US and EU Excluded

On this page

 1. What H3 Does
 2. Cost, Access, and the Open-Weights Question
 3. What This Means

What H3 Does

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
