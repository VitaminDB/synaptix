# synaptix

A native Rust engine for running and training neural networks — hand-written CUDA
kernels compiled at runtime via NVRTC, with no PyTorch, no libtorch, and no Python
runtime.

## What it is

An alternative to the Python ML stack for inference and training. Everything from the
tensor API and CUDA kernels up to full models, a tokenizer, an inference engine, and an
RLHF trainer is written in Rust. Correctness is held to bit-exact parity with PyTorch and
NeMo reference implementations.

## What it runs

Native ports, each validated bit-for-bit against its upstream reference:

- **LLMs** — Qwen3, Qwen3-Next (hybrid), Llama, Gemma-3
- **Image** — FLUX.1, SDXL
- **Video** — LTX-2.3 (22B)
- **Speech** — Whisper (ASR), GigaAM (ASR), Sortformer (diarization)
- **Text-to-speech** — VoxCPM, OmniVoice
- **Music** — ACE-Step
- **Embeddings / rerank** — BGE-M3, BGE-reranker

## Quantization

Native NVFP4 (4-bit) and MXFP8 (8-bit) with block scaling, using `mma.sync` tensor-core
instructions on Blackwell (sm_120). The KV-cache can also be held in fp8 / mxfp8.

## Correctness and benchmarks

Kernels are gated per-row against reference implementations rather than by a global
cosine similarity, which hides local errors. Performance is measured against a
maximally-tuned PyTorch baseline (`torch.compile`, FlashAttention, fp8), and the weaker
paths are documented rather than hidden — see [`LTX_GEMM_PARITY.md`](LTX_GEMM_PARITY.md).

## Building

Requires the CUDA toolkit. Kernels are JIT-compiled at runtime via NVRTC (through the
`cudarc` crate).

```bash
cargo build --release -p synaptix-cli
```

The bit-exact test suite loads reference tensors that are **not** committed to this
repository — they are large and derived from upstream models. Regenerate them with the
scripts under `scripts/reference/`.

## Platforms

CUDA (primary) and CPU. The compile baseline is sm_80 (Ampere); native NVFP4 `mma.sync`
requires sm_120 (Blackwell).

## Status

Young, single-author, and moving fast. The API is not stable; expect breaking changes.

## License

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your
option.
