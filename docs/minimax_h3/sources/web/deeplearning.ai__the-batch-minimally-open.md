# Source: https://www.deeplearning.ai/the-batch/minimaxs-state-of-the-art-video-model-is-only-minimally-open

Fetched: 2026-11-28 (curl + w3m text dump)

---

Machine Learning ResearchTech & SocietyBusinessGenerative AITransformers
Published
Aug 14, 2026
Reading time
4 min read
Diagram shows Minimax H3's process from text prompt input to audiovisual output via a transformer.
Share

  •  
  •  
  •  

Loading the Elevenlabs Text to Speech AudioNative Player...

A free-to-download model sets a new standard for video generation and editing, but its license comes with unexpected restrictions.

What’s new: MiniMax released H3, a high-definition video generation model that accepts a wide range of input media. But the model’s license requires users in the U.S., the UK, the European Union, and
South Korea to submit an application to MiniMax to use it under the same terms as the rest of the world. Key components of the model also remain proprietary, at least for now.

  • Input/output: Up to twelve files including text (7000 characters), images (30 MB per file), audio (15 MB), and video (50 MB) in, video (up to 2000 pixels wide and 15 seconds long) with audio and
    text captions out
  • Architecture: Transformer, 33 billion parameters; separate text, audio, and video encoders, preprocessor, 2k video upscaler
  • Features: Video and audio editing, multi-shot output, six aspect ratios
  • Performance: First on Artificial Analysis’ video editing leaderboard, second (or tied for first, within the margin of error) in text-to-video and image-to-video
  • Availability/price: Weights free for noncommercial and commercial uses under MiniMax H3 license; via MiniMax’s API at $0.13 per second for 2K resolution, $0.08 per second for 768p resolution
    output, input costs vary from free for audio to $0.04 per image and $0.13 per second for high-definition video; prompt re-generation module costs $0.90 per million tokens of input and $3.60 per
    million tokens of output.
  • Undisclosed: Exact parameter count, training data, technical report

How it works: H3’s architecture consists of three modules — a contextual processing system, a video/audio generation base model, and a high-definition upscaler. Only the base model is free to
download, and it comes with restrictions.

  • MiniMax says it trained H3 on “real, natural data” to ensure data quality and scalability. Unlike MiniMax’s earlier video models, all audio types (voice, music, and sound effects) were trained
    together and are modeled through a single encoder. The team trained the model’s reference and editing functions using natural language rather than fixed presets. The team sought to combine
    different media types (sound, video, etc.) earlier in the process.
  • Supported generation modes include text-to-video, first- or last-frame image to video, and image, audio, and video references, plus any combination. For example, a user prompt might instruct the
    model to begin with a single still image, reference the camera movement in two videos, and the audio score from a third, with text instructions for how the scene should unfold.
  • Input is first processed by H3-Context-IR. This reasoning module interprets the prompts and input media and generates a new text prompt instructing the base model how to blend them together.
    Users of the base model alone need to be explicit in their instructions for each media type, or use their own preprocessing system; users of the full API pipeline benefit from the Context-IR
    module doing most of that work.
  • The base module generates 768p video, that is, 1792 pixels by 768 pixels (given a maximum 21:9 aspect ratio). Three encoders process text, video, and audio, respectively; a separate variational
    autoencoder also encodes video.
  • A third module, H3-Regenerate-2K, instructs the base module to regenerate the 768p video in native 2K resolution (2000 pixels wide and up to 4667 pixels long). The optimized prompt also includes
    the original media references and user prompt as context, allowing the regenerated video to add missing details rather than only extrapolating from the source video.
  • The weights’ license requires commercial users to prominently display the model name on any product using H3 and bars all users from distilling another model on H3’s output. It also prohibits use
    that may harm minors, interfere with elections, or violate local law. The license also identifies the United States, United Kingdom, European Union, and Republic of Korea as “excluded
    territories” and requires users of the weights in these territories to apply for a license “to ensure [their] usage is lawful, responsible, and without infringing any rights.”

Behind the news: MiniMax H3’s head-to-head human-preference ELO scores put it squarely in a top three with Google’s Gemini Omni Flash and Bytedance’s Dreamina Seedance 2.0. Black Forest Labs’ FLUX
models have typically included an open weights release for developers, but its recent FLUX 3 model is proprietary and API-only. It’s yet to be independently tested, but Black Forest Labs’s tests
suggest it would be a fourth model vying to be state-of-the-art. Dreamina Seedance 2.5 also awaits testing.

Why it matters: Despite the restrictive license and unusual territorial restrictions, MiniMax H3 is clearly a top video generation model, offering commercial users a strong and versatile set of tools
for video generation on par with Google’s much more expensive competitor. We also get a glimpse of how a top video model works under the hood. It appears that MiniMax’s strategy of foregoing
synthetic data — especially difficult when good video transcripts are hard to come by — and bundling prompt optimization into the pipeline is paying off.

We’re thinking: Even in this paranoid period of AI development, it makes no sense to restrict use of weights by country beyond uses that would break that country or region’s laws. It violates the
fundamental meaning of openness: the idea that anyone can download a model, see how it works, and put it to use. We hope these restrictions don’t become a trend.

Share

  •  
  •
