# synaptix

[![Donate via PayPal](https://img.shields.io/badge/donate-PayPal-0070ba?logo=paypal&logoColor=white)](https://paypal.me/vitamindbnfkz)
[![Licence: MIT OR Apache-2.0](https://img.shields.io/badge/licence-MIT%20OR%20Apache--2.0-blue)](#license)
[![CUDA: NVRTC JIT](https://img.shields.io/badge/CUDA-NVRTC%20JIT-76b900?logo=nvidia&logoColor=white)](#building)

A native Rust engine for running and training neural networks — hand-written CUDA kernels
compiled at runtime via NVRTC, with no PyTorch, no libtorch, and no Python runtime.

## What it is

An alternative to the Python ML stack. Everything from the tensor API and CUDA kernels up to
full model ports, a tokenizer, an inference engine with paged KV-caches, and a training stack
is written in Rust: ~232k lines, 73 `.cu` kernel files, 1 800+ tests. Correctness is held to
bit-exact parity with PyTorch and NeMo reference implementations, per row rather than by a
global cosine similarity that hides local errors.

## Quick start

```sh
cargo build --release -p synaptix-cli

synaptix inspect model.syn                       # layout of a bundle
synaptix convert model.gguf model.syn            # GGUF / safetensors → .syn
synaptix run model.syn "Explain NVFP4" --max-tokens 256 --quant nvfp4
synaptix chat model.syn --context 32768          # interactive, prefix-KV across turns
synaptix bench model.syn --n-tokens 128          # prefill / decode throughput

synaptix video ltx.syn "a paper boat in the rain" -o clip.mp4 --gemma ./gemma-3-12b
synaptix music "lofi piano, rain" -o track.wav --models ./syn_models --duration auto
synaptix speak voxcpm.syn "Hello there" -o out.wav --reference voice.wav
synaptix podcast vibevoice.syn "Speaker 1: hi\nSpeaker 2: hey" -o show.wav
synaptix transcribe whisper.syn talk.mp3 --timestamps
```

## What it runs

Native ports, each validated against its upstream reference:

| Domain | Models |
|---|---|
| **LLM** | Qwen3 (dense + MoE), Qwen3-Next hybrids (GatedDeltaNet + full attention, `qwen3_5/3_6/3_8`), Qwen4Exp (125B MoE: sparse-attention indexer, gated residuals, PLE n-grams, MTP head), Llama, Gemma-3, Gemma-4 26B A4B, Muse Glimmer 30B |
| **Vision-language** | Qwen3-VL tower (images and video, 3D M-RoPE), Gemma-4 vision tower, Muse Glimmer |
| **Image** | FLUX.1, SDXL, Depth Anything V2 |
| **Video** | LTX-2.3 (22B), MiniMax-H3 (video with synchronized audio) |
| **Speech** | Whisper, GigaAM (ASR), Sortformer (diarization) |
| **Text-to-speech** | VoxCPM, OmniVoice, VibeVoice (long-form, multi-speaker) |
| **Music** | ACE-Step (generate, cover, edit, extend, extract, repaint) |
| **Embeddings / rerank** | BGE-M3, BGE-reranker-v2-m3 |

## Quantization and memory

- **NVFP4 (4-bit) and MXFP8 (8-bit)** with block scaling, through `mma.sync` tensor-core
  instructions on Blackwell (sm_120). Quantization can be applied while packing a `.syn`
  bundle, so the on-disk model is the deployed model.
- **KV-cache in MXFP8 by default** (per-layer: sliding-window layers stay unquantized), with
  block-table attention kernels that read the quantized cache directly.
- **MoE offload** — experts live in pinned host RAM and stream to the card on demand, with an
  arena allocator that returns VRAM to the driver on eviction instead of growing with the
  swap-in stream. A 125B MoE model runs on a 24 GB card.
- **Partial block offload** — N transformer blocks stay resident, the rest stream from the
  host, so a model larger than VRAM still runs (at a documented cost in tokens/s).
- **Prefix-KV sessions** — the KV of a conversation survives between turns for every
  architecture, including prompts that carry images, and can be parked in host RAM.

## Performance

Measured on an RTX 5090 Laptop (24 GB), 93 GB system RAM:

| Model | Prefill | Decode | Notes |
|---|---|---|---|
| Gemma-4 26B A4B | 10 100 tok/s @ 4k | 210 tok/s | CUDA-graph decode capturing the MoE, fused per-layer kernels |
| Qwen3.8-27B hybrid | 1 450 tok/s @ 3.3k | 47 tok/s | MTP speculative decode |
| Qwen3.8-Flash-Next 125B MoE | 1 650 tok/s @ 260k | 17–22 tok/s | 262k context on 24 GB; experts stream at ~39 GB/s |

Those are not starting points: Gemma-4 decode went 35 → 210 tok/s and prefill 957 → 10 100
tok/s over a week of kernel work (fused layer kernels, no memset on hot outputs, quantized
projections reading and writing BF16 directly, GQA-flash, windowed flash on sliding layers).
The write-ups live in the `synthos` repository under `docs/`.

## Correctness and benchmarks

Kernels are gated per-row against reference implementations. Performance is measured against
a maximally-tuned PyTorch baseline (`torch.compile`, FlashAttention, fp8), and the weaker
paths are documented rather than hidden — see [`LTX_GEMM_PARITY.md`](LTX_GEMM_PARITY.md),
where bf16 GEMM lands at 0.82–1.16× of cuBLAS depending on shape: ahead on small and medium
M, behind on large-M tails.

## How it is laid out

| Crate | What it holds |
|---|---|
| `synaptix-core` | tensors, dtypes, devices, memory pools |
| `synaptix-kernels-cuda` / `-cpu` | 73 `.cu` kernel files JIT-compiled via NVRTC; a CPU fallback |
| `synaptix-nn` | layers, attention, normalization, samplers |
| `synaptix-infer` | inference engine: paged KV, CUDA-graph capture, speculative decode |
| `synaptix-models` | the model ports listed above |
| `synaptix-bundle` | the `.syn` single-file format (mmap, zero-copy, quantize-on-pack) |
| `synaptix-tokenizer` | tokenizers, chat templates, tool-call parsers |
| `synaptix` | facades: `llm`, `asr`, `tts`, `embedding`, `rerank`, `diarization`, `sampling` |
| `synaptix-autograd` / `-train` | reverse-mode autograd, optimizers, checkpointing, eval |
| `synaptix-rag` | document parsing, chunking, retrieval helpers |
| `synaptix-cli` | the `synaptix` binary |

**Honest scope note:** inference is what is production-ready — it powers
[synthos](https://github.com/VitaminDB/synthos) daily. The training side has a working
autograd, optimizers and checkpointing with tests, but the RLHF, distillation and self-play
modules are scaffolding, not finished trainers. Directories exist for models that are not
ported yet (they hold a stub `lib.rs`); the table above lists everything that actually runs.

## Building

Requires the CUDA toolkit (`nvcc` is read at build time to pin the CUDA version; the driver
itself is loaded dynamically at runtime through a
[vendored cudarc](third_party/cudarc/PATCH.md) patched to be safe under CUDA-graph capture).

```bash
cargo build --release -p synaptix-cli
cargo test --workspace
```

The bit-exact test suite loads reference tensors that are **not** committed to this
repository — they are large and derived from upstream models. Regenerate them with the
scripts under `scripts/reference/`.

## Platforms

CUDA (primary) and CPU. The compile baseline is sm_80 (Ampere); native NVFP4 `mma.sync`
requires sm_120 (Blackwell).

## Status

Young, single-author, and moving fast. The API is not stable; expect breaking changes.

## How it is built

One developer, with Claude (Anthropic) as a daily coding assistant. The architecture, the
CUDA kernel work and every number in this README are mine — measured on my hardware, with
the losses reported next to the wins. The assistant carries a large share of the typing,
the test scaffolding and the refactors.

## Support

synaptix is free and open source. If it is useful to you, you can support its development
with a donation via [PayPal](https://paypal.me/vitamindbnfkz).

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option.
