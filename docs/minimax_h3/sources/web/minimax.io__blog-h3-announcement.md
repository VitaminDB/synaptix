# Source: https://minimax.io/blog/minimax-h3

Fetched: 2026-11-28 (curl + w3m text dump)

---

Research
Company
●
About
News
Investor Relations

MiniMax H3: An Open Model Breaking the Boundaries Between Tasks and Modalities

2026-07-31
AIH3MultimodalVideo Generation

[]

Today, we're launching MiniMax H3, a general-purpose multimodal generation model. H3 understands unified context across text, images, video, and audio, generating video with native stereo sound, up
to 15 seconds at 2K resolution.

Early testing shows H3 is ready for commercial content creation across a wide range of use cases, excelling at instruction following, accurate text and brand rendering, and V2V motion transfer. With
precise, controllable multimodal generation and editing, H3 is built for advertising, branding, e-commerce, product design, UI/UX, gaming, and more.

Powered by technologies including Contextual Omni Representation, H3-VAE, H3-Omni Transformer, and In-Context Regeneration, H3 delivers industry-leading price-performance. We offer 2K resolution by
default. At 2K, H3's per-second price is less than a third of mainstream models, and at 768p, it's less than half the price of mainstream models' 720p.

Closed-source models have long dominated video generation, with slower iteration and a less open ecosystem than fields like large language models. To support the open-source community, accelerate
compatibility with a broader range of AI hardware, and make it easier for users to build their own customized versions, we plan to open up the model weights in the coming days, subject to applicable
laws and regulations. Hardware compatibility has been a key consideration since the earliest stages of H3's design.

Multimodal context understanding

Real creative work requires blending complex information across modalities, pulling in images, audio, video, and more as input sources. For example, the prompt for the shot below is: "Reference the
Hitchcock camera movement from Video 1, have the character in Image 2 sing, with the vocals matching Audio 3." Just describe the relationship between the context and the target video in words. H3
handles the complex, full-modality understanding on its own.

  • Input:

Your browser does not support this video. Please use another browser.

Video 1

H3 en-v2 image 1

Image 2

Your browser does not support this audio. Please use another browser.

Audio 3

H3-Generated Video:

Your browser does not support this video. Please use another browser.

  • 2K Performance

Your browser does not support this video. Please use another browser.

  • Native Stereo Sound

Your browser does not support this video. Please use another browser.

H3 Use Cases

  • Film Opening Titles

Your browser does not support this video. Please use another browser.

  • Product Website

Your browser does not support this video. Please use another browser.

  • Animated Poster

Your browser does not support this video. Please use another browser.

  • Advertising & E-commerce

Your browser does not support this video. Please use another browser.

H3's Design Philosophy

Breaking the Boundaries Between Tasks: From Specialized to General-Purpose

We previously developed two generations of models: Hailuo 01 built the system from the ground up, and Hailuo 02 focused on improving core components like architectural efficiency, data quality, and
scale.

In designing H3, we recognized the limitations of prior generative models around task boundaries: image generation was typically split into separate expert models for T2I, editing, subject reference,
motion reference, and style reference; voice, sound effects, and music in audio generation were largely studied as separate domains; and video generation was further fragmented into text-to-video,
image-to-video, first-and-last-frame, subject reference, motion reference, voice reference, video editing, and more, with clear boundaries between image, video, and audio as well.

These siloed tasks, capabilities, and modalities constrained creative freedom in practice. And, on the training side, capped the model's ability to generalize. Both point to major room for change, in
application paradigm and training paradigm alike. So the first principle guiding H3's development was unifying and generalizing across tasks.

Based on this, here's a brief overview of H3's pretraining paradigm:

  • Data and Tasks
  • Text-to-image
  • Text-to-video
  • With jointly generated audio, all audio output is native stereo
  • Native multi-shot modeling
  • Text-to-audio
  • No separation between voice, sound effects, and music — all jointly modeled
  • Generalized reference and editing
  • Image-to-image reference and editing
  • Image-to-video reference and editing
  • Audio-to-audio reference and editing
  • Audio-video-to-audio-video reference and editing
  • In our design, "generalized reference and editing" means:
  • Built entirely from real, natural data, giving it strong data scalability
  • Reference and editing relationships expressed through natural language rather than confined to a fixed task set; language (or intelligence structure, broadly speaking) is the bridge to
    generalization
  • Architecture
  • Fuse diverse data types and tasks as early as possible — the right mixing ratio is key
  • Training Strategy
  • Fuse diverse data types and tasks as early as possible in training — the right mixing ratio is key

These choices all point to the same goal: giving H3 broad multimodal context understanding and generation capabilities from the pretraining stage onward.

Earlier we talked about the shift in training paradigm, so how is the application landscape for video models changing as well?

We're seeing creators describe their intent directly in natural language, rather than just entering a simple visual prompt. As multimodal understanding, instruction following, and complex task
execution keep improving, video models will be able to grasp fuller creative intent, handle more complex content needs, and gradually move from "generating a clip" to genuinely participating in the
entire content production process.

H3's Technical Choices

H3 is built on a simple design philosophy, but bringing it to life was extraordinarily complex. From Hailuo 01 and Hailuo 02 to H3, each generation has been an order of magnitude more complex to
build than the last.

A brief look at some of H3's core technologies:

  • H3-Contextual Omni Representation

One of the most important things we did in engineering H3 was strengthening its captioning capability.

  • Introducing multimodal context further broadened what "caption" needs to cover, we're not just describing the target video anymore, but the relationship between context and target video, and even
    relationships among elements within the context itself.
  • We need to jointly describe video and audio, and that audio-visual relationship gets even more complex across multiple shots.
  • This is fundamentally a form of Contextual Omni Representation, where language acts as the generalizable bridge and interpreter, unifying "tasks" into an open, descriptive form. This is the root
    of H3's broad instruction-following ability.
  • To get there, we built dedicated models and a full-modality understanding pipeline. Most source material requires around 100K tokens of inference, distilled down to an average of roughly 4K
    tokens.
  • H3-VAE: A Major Boost in Architectural Efficiency
  • The H series has continuously pushed forward on tokenizer technology. With H3, we completely overhauled our previous tokenizer, achieving across-the-board gains in reconstruction quality and
    learnability, giving H3 a competitive edge in efficiency. Its high compression ratio also delivers a 4x gain in effective sequence length, substantially cutting training and inference costs, and
    it's the key technology behind our native 2K resolution support.
  • H3-Omni Transformer

H3-Omni Transformer: An Architecture Built for Task Generalization

  • In keeping with H3's design philosophy that "architecture should serve the task," generality and efficiency were the only two goals. Notably, we set aside the Hailuo-02 architecture, despite the
    significant architectural advantages it once gave us, because it would introduce unnecessary complexity for a model built around task generalization. We believe task generalization is an
    irreversible trend, and architectural tricks should give way to how the model is defined.
  • In H3, the introduction of multimodal context tripled the variance in sequence length, and the compute workloads for understanding and generation became markedly more heterogeneous. We adopted a
    training architecture that separates understanding and generation workloads, fine-tuning hardware utilization for each, while jointly balancing per-sample heterogeneous compute against load
    balancing across samples. End to end, this lifted training throughput by nearly 30%.
  • H3-In-context Regeneration
  • For H3's 2K output, instead of using a conventional dedicated super-resolution module, we have the H3 base model regenerate its own low-resolution output in-context. This brings two advantages:
    first, the regeneration process maximally reuses the generative capability already built into the H3 base model; second, the in-context approach lets it draw on the original multimodal context
    again to produce high-resolution output, recovering details that traditional super-resolution can only "guess" at and often can't restore, like small text and fine detail.

We'll be sharing the full H3 Technical Report soon. We look forward to hearing your feedback.

Our Vision & What's Next

Language, images, video, and audio are fundamental modalities of human experience. They are deeply interconnected and serve as our primary interfaces with the world. Together, they form multimodal
contexts that can communicate a vast range of information efficiently, and the ability to express and share information is itself a form of productivity.

For multimodal understanding and generation, we view language as a generalizable, scalable computational system. That is why we believe multimodal intelligence should be deeply grounded in language.

H3 still has room to grow, and here are our priorities for future versions:

  • Strong multimodal understanding is the foundation of high-quality generation, and there is significant room to improve. In the next generation of the H series, we plan to integrate capabilities
    from our M-series models.
  • H3's current model size leaves room for improvement across several capabilities. Scaling is a clear path forward, and we believe stronger task generalization will allow us to fully unlock its
    potential.
  • Visual detail can still be improved in certain scenarios. We will continue pushing toward higher resolutions and greater visual fidelity.

Intelligence with Everyone.

logo
