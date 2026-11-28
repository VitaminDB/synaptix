# Source: https://hn.svelte.dev/item/49252179

Fetched: 2026-11-28 (curl + w3m text dump)

---

  • best
  • show
  • ask
  • jobs
  • about

H3-metal – Native MiniMax-H3 inference for Apple Silicon

github.com

440 points by swyx 3 months ago

Meleagris 3 months ago

I've been using MiniMax H3 on my M5 Pro 64GB MacBook Pro through ComfyUI. It works extremely well.

I had to modify the default ComfyUI workflows to use a GGUF quant (city96's ComfyUI-GGUF custom node, UnetLoaderGGUF in place of the stock loader) [0].

I use the model labeled Q5_K_M. There is Q8_0 available as well, which is 34GB and fits fine in 64GB unified memory if you keep resolution modest.

The main issue is speed, a ~9-second 480x864 clip at 20 steps takes me a bit over an hour. So this will be cool to try for the speed up alone.

There's a lot of great information and workflows available to follow on the r/StableDiffusion subreddit.

[0] https://huggingface.co/Abiray/MiniMax-H3-GGUF/tree/main/unet

  • alexgoodhart 3 months ago

    I wonder how much faster your m5 pro is compared to my M1 Max @ 64gb

      □ yieldcrv 3 months ago

        wait till the M7 bro you’re almost there, rumor has it that Apple is skipping the M6 but it still might be 2028

          ☆ ignoramous 3 months ago

            Discussion: https://news.ycombinator.com/item?id=48676795

  • jonplackett 3 months ago

    What is the quality of the output like compared to something like Veo?

      □ Art9681 3 months ago

        It's better than all private video models like Veo. Yes, I too am incredulous they released this open weights. It's a VERY disruptive model in all the good ways.

  • antirez 3 months ago

    This implementation is much faster on my M5 Max, like a few minutes for the same video, but on an M5 Max with 128GB, didn't test on M5 Pro. About memory, could be executed on 64GB with a few
    changes.

      □ Manfrednotfunny 3 months ago

        Memory bandwidtih between pro and max is double. 300gb/s vs. 600gb/s btw.

          ☆ antirez 3 months ago

            Does not matter much in this case. GPU bound.

              ○ Manfrednotfunny 3 months ago

                Seems to be true, but also seems hard t obenchmark with max having more GPU cores too.

          ☆ dragonwriter 3 months ago

            My understanding is that that tends to be more critical with LLMs than image/video gen models, which are relatively more compute vs. memory transfer intensive than LLMs

              ○ zozbot234 3 months ago

                Performance might still end up being bounded by data transfer speed if SSD streaming is heavily used to make up for limited RAM. By comparison, it doesn't take many parallel-batched
                sessions to make LLM decode compute-bound on typical hardware (hence seeing very limited gains from even wider batching), but this just doesn't apply when streaming weights from disk,
                the setting is completely different.

  • Myzura 3 months ago

    How much free space do you have left after running this llm model? Have you tried to develop your own model with the M5?

  • thousand_nights 3 months ago

    > a ~9-second 480x864 clip at 20 steps takes me a bit over an hour

    that's rough. for comparison, i tried the exact same parameters on my 5090 RTX and it took 2 minutes to generate.

    i believe diffusion models are primarily compute bound so the macs aren't really the ideal hardware for this kind of stuff

  • vimto 3 months ago

    GGUF is outdated in the latest versions of Comfy-UI. If you want a good balance of size, speed and quality you should use the int8_convrot model from the official Comfy Org Repo https://
    huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffus...

      □ Meleagris 3 months ago

        So I did test this, and it doesn't work because the quantized layers need torch._int_mm, which PyTorch's MPS backend doesn't implement. It just throws NotImplementedError.

      □ SV_BubbleTime 3 months ago

        This is good advice if you have nvidia, but for Mac does not apply currently.

      □ dragonwriter 3 months ago

        GGUF is unsupported by ComfyUI’s memory management system that enables running models much larger than fit in VRAM with tolerable efficiency via weight streaming, but for unified memory
        systems that system is less relevant (unless using models too big to run in unified memory AND having fast enough mass storage to benefit from direct-from-disk weight streaming.)

antirez 3 months ago

In the AMA Minimax said that H3 could support sparse attention, that would be a huge speedup! I wonder if there are any news on that. H3 is very cool. EDIT: testing a --sparse-attention optional mode
based on what they said in the Reddit post.

diddid 3 months ago

This is where the DGX spark makes up a bit of the ground it loses on llm work, diffusion and cuda go together like peanut butter and jelly.

  • embedding-shape 3 months ago

    cough DiffusionGemma cough

    Seriously, very dumb model compared to what you can run locally, but holy moly is it FAST on one GPU, seriously impressive. Can't wait for those to be scaled up a bit to fit perfectly within 96GB
    VRAM, then they'll be competitive.

linzhangrun 3 months ago

On my 128GB M4 Max Mac Studio, generating a 15s 480p video with MiniMax H3 in ComfyUI takes an hour and a half.

Put Codex to work on deploying it now, hoping the speed can improve quite a lot :-) Thanks anyway

  • sscarduzio 3 months ago

    Please keep up posted about the results!

      □ linzhangrun 3 months ago

        First batch of quick test results: approximately 1/5 speed improvement

          ☆ bewareofscams 3 months ago

            +20% or x5 speed improvement?

              ○ linzhangrun 3 months ago

                20%

  • embedding-shape 3 months ago

    > On my 128GB M4 Max Mac Studio, generating a 15s 480p video with MiniMax H3 in ComfyUI takes an hour and a half.

    That's crazy, a RTX Pro 6000 does that in in 2-3 minutes (give or take, depending on your exact settings). LLMs don't make the difference between standalone GPU vs unified memory + CPU so obvious
    as diffusion models seems to do.

      □ whywhywhywhy 3 months ago

        It’s always been the case, it’s more the anomaly that LLMs work at comparable speeds on M series because almost all other ML runs way faster on Nvidia cards.

      □ MrScruff 3 months ago

        LLM prompt processing and diffusion models are compute bound, while LLM token generation is memory bandwidth bound.

      □ throw1234567891 3 months ago

        An RTX6000 is a completely different class of hardware.

          ☆ embedding-shape 3 months ago

            Really? No wonder I keep trying to type on it like a laptop but it doesn't work and doesn't even have a display!

              ○ throw1234567891 3 months ago

                Right? Do you also type on a Mac Studio without a keyboard plugged in? Like tap the ethernet port 3 times in a row then this sequence of sticking your fingers into TB5 ports? I mean,
                it’s clear you stick your RTX into a computer.

  • brcmthrowaway 3 months ago

    For gods sakes, Apple let people have run other GPUs instead of these pissweak 2012 class mobile GPUs

  • mandeepj 3 months ago

    > Put Codex to work on deploying it now

    Which codex?

TechSquidTV 3 months ago

This still requires 128Gb of memory, right? Me and my lowly 96Gb, like a commoner; missing out on the fun.

  • thehamkercat 3 months ago

    From README:

    > On the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB peak physical footprint and zero
    swaps.

    Looks like it uses 40GB? So your 96GB mac setup should work fine i guess (Model itself is 33B)

      □ vunderba 3 months ago

        This repo looks neat, but I hope they add some more clear benchmarks because that time (74.58s) is pretty meaningless given that the it/s (and total time) is highly dependent on mode (T2V vs
        I2V vs REF2V), resolution (0.4, 0.6mp, etc), duration (5-15 seconds), etc.

  • pferdone 3 months ago

    you should have a look at https://github.com/deepbeepmeep/Wan2GP which is the goto tool for "gpu poor", although as people below already pointed out you should be fine with comfyui's standard
    setup aswell

      □ embedding-shape 3 months ago

        First, I think they're not even talking about GPUs, this is macOS hardware so unified memory. Secondly, if they were talking about GPUs, then 96GB VRAM is hardly what people refer to when
        they say "gpu poor".

          ☆ pferdone 3 months ago

            1) It would still run on the Mac's GPU.

            2) Since it's unified memory, you won't have 96GB available.

            3) I offered a solution that is usually recommended to the "gpu poor", if he's concerned with how much memory he would need.

            4) I stated, that people already pointed out how he should be fine and that "gpu poor" doesn't apply to him.

            5) "gpu poor" depends on what model you are trying to use. If you want to run Kimi or GLM you are still "gpu poor" even if you have an RTX Pro 6000 with 96GB of VRAM.

  • embedding-shape 3 months ago

    It shouldn't? Unless you're using BF16 for all weights (I'm using NVFP4 for the text encoder, otherwise everything BF16 (and audio F32)) you'll fit it all within 96GB VRAM, bugs non-with-standing
    :) I've been fitting this within 96GB VRAM without issues.

      □ Arn_Thor 3 months ago

        *notwithstanding

        Anyway, good input!

          ☆ embedding-shape 3 months ago

            I'll blame it on other book authors! :) https://en.wiktionary.org/wiki/nonwithstanding

            > This misconstruction is very common, included in print publications spanning several centuries. It might be considered an alternative spelling, albeit still a mistaken usage.

            Thanks though, I never actually knew so was helpful :)

c0rruptbytes 3 months ago

wow antirez does not sleep

  • mlmonkey 3 months ago

    Understatement of the year :-D

  • behnamoh 3 months ago

    when you have enough money to not have to worry about anything, you can go back to your hobbies. in this case, his hobby is programming.

      □ stressback 3 months ago

        "enough money not to worry about anything" haha

      □ azinman2 3 months ago

        Being a world class talent is independent of financial situation.

          ☆ behnamoh 3 months ago

            talent w/o financial stability is a battery w/o circuit.

          ☆ afavour 3 months ago

            Of course it isn’t. If you can’t afford to eat you can’t achieve any potential you might have. Financial stability is a gamechanger for everyone.

              ○ ionelaipatioaei 3 months ago

                That's stupid, if you're truly talented you'll solve the financial stuff in order to pursue whatever you want to do - if you don't then that's on you.

                  ■ jaapz 3 months ago

                    You should read Outliers by Malcolm Gladwell

                  ■ AdamN 3 months ago

                    The majority of the population doesn't even have access to a functioning computer. So yeah maybe somebody truly talented can figure their way out of that hole after a few years
                    but that's where alot of people are starting from.

          ☆ moshun 3 months ago

            “I am, somehow, less interested in the weight and convolutions of Einstein's brain than in the near certainty that people of equal talent have lived and died in cotton fields and
            sweatshops."— Stephen Jay Gould

          ☆ menaerus 3 months ago

            To show the world you're a "world class talent", whatever that means, also suggests you would either have to be a genius or have enough resources to work on your side quests. Latter
            implies you're well off so, no, I don't think there's zero correlation between the two in all cases.

      □ epolanski 3 months ago

        Redis, hping and dump1090 were all side projects he started/written while having a full time job.

        Your comment really sounds like "many other people would do A and B if they just had time and money to do so", but he's been doing so since time and money were major constraints.

  • punnerud 3 months ago

    Wow, had to check some of his other repos; his the one behind dump1090

      □ matheusmoreira 3 months ago

        He's the one who wrote Kilo too!

          ☆ v-mdev 3 months ago

            He also developed Redis

yieldcrv 3 months ago

Alright I’ve been afraid to ask but have been having trouble finding

What are some adult entertainment workflows in comfyui, I need best loras, best prompts to start with

and the communities, are they on telegram or something?

  • Havoc 3 months ago

    There will Reddit subs for it though couldn’t tell you which off top of my head

    I’d personally steer clear of messaging platforms for this - who knows what one might stumble into there

      □ embedding-shape 3 months ago

        > I’d personally steer clear of messaging platforms for this - who knows what one might stumble into there

        Personally I have no interest, but sometime browse stuff out of curiosity. But this got more of my curiosity, what kind of "stuff" are you implying they might stumble upon on the open, public
        internet? Sure, some NSFW, horror and otherwise weird stuff is there, especially around AI generation, but hardly something that will leave you traumatized, unless I misunderstand what you're
        implying?

          ☆ Havoc 3 months ago

            I do not know and very much plan to keep it that way

          ☆ Nekit1234007 3 months ago

            One thing I read on this topic on reddit is never EVER use the word "girl" when prompting H3. So CSAM probably.

              ○ embedding-shape 3 months ago

                Did you try this yourself? Of course you wouldn't, because not wanting to produce SCAM sorry I meant CSAM.

                And no, including the word "girl" in H3 does not lead to CSAM in any way, shape or form, but it's a great example how FUD quickly spreads.

                  ■ Nekit1234007 3 months ago

                    I don't have hardware to runs this, so no, I didn't.

                    One thing that I had experience with, that led me to believe this might be true: it seems one of the earlier llama models was over-tuned to resist generating CSAM. Once I tried a
                    somewhat sensitive prompt containing the word "girl" in it. Llama only ever generated refusals for this prompt, citing I was prompting for CSAM. GPTs and Claudes of that era had
                    no issues with the same prompt.

              ○ dragonwriter 3 months ago

                Avoiding use of “girl” when you aren’t asking for a feminine child is good advice for any model that isn't specifically trained on a specialized prompting vocabulary (e.g., danbooru
                tags) where “girl” has a different meaning.

                But I haven't seen anything reliable about H3 being particularly special in ther regard.

              ○ yieldcrv 3 months ago

                are these still 2026 problems? a simple vector index would associate girl as a synonym for adult woman as is used by both men and women in the English lexicon

                even a Chinese model that thinks its Claude when asked would have inherited this association

  • bahmboo 3 months ago

    This is a healthy question. We want to use these tools for regular old human needs and desires.

  • pferdone 3 months ago

    a friend told me there's a reddit called: unstable diffusion

  • antirez 3 months ago

    H3 is quite uncensored, but was not trained on p0rn, so it has no anatomy clues needed to generate that kind of stuff. For softer adult content it is reported to be fine on Reddit.

      □ yieldcrv 3 months ago

        loras have been fixing that for years

          ☆ dragonwriter 3 months ago

            Other than for pure t2v usage you probably don't need a LoRA for much, I have seen evidence that it is knowledgeable enough that it can handle a fair amount of anatomy looking and
            behaving reasonably with just relevant cues in control (for the fl2va model) or reference (for the r2va model) images.

            And the r2va model can also use video input for motion reference.

  • dragonwriter 3 months ago

    While you will find plenty of people willing to scam you to pay for “adult entertainment workflows”, the built in templates in ComfyUI for the model—perhaps dropping in a Lora Loader node for the
    a Turbo lora for speed—handle running the model, the subject matter adaptation isn’t really a workflow issue but one of reference/control images/audio/videos and prompting.

    For Minimax H3, more than most models, you should read (and, if you are using an LLM for prompt assistance, make it sure it has access to) the official prompt guidelines, as each of the main
    models (fl2va that handles text-to-video and first- and/or last-frame-to-video and r2va that handles more complex reference cases) has its own structured prompt format (with many common
    features).

benbojangles 3 months ago

running this on an m1 max 64gb, it produces some really nice clips with music and audio, stitching the clips together after makes a nice short story generated entirely using a local ai system. It
takes quite a while to generate each clip though, But at least it is working. I would like to know a few things:

-Optimal settings/configs examples for h3.c to help speed things up -Optimal recommended generation settings for each mac product, i am sure it is easy to do -Prompt generator assistant

Great work from the github author. The more i use it the more i realise i don't need a gui to generate video, just terminal.

My setup: Macbook #1 as a client Macbook #2 as server

-use macbook #1 terminal + ssh -run mactop in terminal tab to monitor Macbook #2's hardware during ai video generation -run h3.c on Macbook #2 via ssh session -transfer the output video file from
macbook #2 to macbook #1 using terminal scp -view the video

abhinai 3 months ago

How similar are Jeff Dean and Salvatore Sanfilippo?

  • onionisafruit 3 months ago

    My favorite Jeff Dean fact is that he’s also antirez. Which reminds me of my favorite Salvatore Sanfilippo fact. He’s also Jeff Dean

      □ tomcam 3 months ago

        I'm totally following this

      □ ralusek 3 months ago

        Is that what the identity function is?

          ☆ silisili 3 months ago

            It's why javascript had to add the triple equals check...

              ○ muragekibicho 3 months ago

                2 is not enough. 3 verifies the Dean-Sanfilippo correspondence.

      □ hnfong 3 months ago

        The claims seem fitting from a user named "onionisafruit"...

      □ freakynit 3 months ago

        Identify theft is not a joke Jim!

  • robotresearcher 3 months ago

    People are really good at stuff.

    I noticed on a bar TV the other day that some of the Chromecast screensaver landscape photo credits were to Peter Norvig. They were really lovely pictures.

      □ SwellJoe 3 months ago

        I've run into Peter Norvig twice. Once at a YC event; the other when I parked my motorhome in front of his house in Palo Alto for a couple of days while visiting a friend who happened to live
        on the same street (not on purpose, I didn't know it was his house, it was just where I found sufficient open street parking for a huge motorhome, big houses with fewer cars on the street
        than on my friend's block). I ran into him while walking my dog, he asked about the motorhome and we talked travel. He was lovely both times. Not everyone is nice about a big motorhome
        parking on their block, especially in California, but he was friendly.

      □ mrbonner 3 months ago

        I have had 2 encounters with Dr Norvig, all online. One, I randomly reach out for him to get a referral to a hiring manager at Google. He gave me a couple of email leads. Two, I emailed ask
        him about a writing piece regarding fuzzy logic and why it was not popular in the US academic landscape. I got a very long, specific and detail response back. I need to print that email out
        and frame it lol.

aaqaishtyaq 3 months ago

Anyone tried it with M4 Pro, 48GB of memory?

tipiirai 3 months ago

I'd love to know what the alternatives are and how this is better

  • v-mdev 3 months ago

    I totally agree with you.

  • Schiendelman 3 months ago

    This will be a little faster right now on an M4 or M5 because it's optimized for Apple Silicon. Assuming this model is still state of the art in six months, which might not be a surprise given
    how long other video models have taken, this should be much, much faster with the M7 chip.

iamyatin 3 months ago

Noob question to all, is there any open source coding model that I can run on Mac mini 16gb?

  • jckahn 3 months ago

    Qwen 9B

  • mrbonner 3 months ago

    I use Bonsai 27B ternary on my 24GB MBP. But, I believe you definitely can run it with 18 or even 8GB.

thedangler 3 months ago

I'm looking to setup a way to create images for my own instagram marketing. I do not care how long it takes to make 10 variations of a post as that speed would still be faster than me making it.

Does this model work with ComfyUI easily? Can I just download it?

  • tracerbulletx 3 months ago

    Yes, there's a default workflow template in Comfy for that matter now.

ChristmasTomer 3 months ago

I mean yeah, a 5090 is obviously gonna smoke a Mac lol. But that’s not really the point. If I can run this thing locally and get a video in a few minutes instead of an hour, that’s already pretty
sick.

sharanharsoor 3 months ago

hey, this is nice!
