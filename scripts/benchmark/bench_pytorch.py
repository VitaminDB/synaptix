"""PyTorch CPU benchmark suite for Session 10 — T15 performance targets.

Run:
    python scripts/benchmark/bench_pytorch.py

Measures wall-clock time for operations that Synaptix must match within 2×.
Prints a markdown table suitable for pasting into T15_benchmarks.md.

All benchmarks run on CPU, float32 unless otherwise noted.
Warmup: 3 runs. Measured: 20 runs.
"""

import math
import pathlib
import time
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

WARMUP = 3
RUNS = 20


def time_fn(fn: Callable[[], object], warmup: int = WARMUP, runs: int = RUNS) -> float:
    """Returns median wall-clock time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def bench_gemm_f32() -> dict[str, object]:
    m, k, n = 512, 512, 512
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    ms = time_fn(lambda: torch.mm(a, b))
    return {"name": f"gemm_f32 {m}x{k}x{n}", "ms": ms, "gflops": 2.0 * m * k * n / (ms * 1e-3) / 1e9}


def bench_gemm_bf16() -> dict[str, object]:
    m, k, n = 512, 512, 512
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    ms = time_fn(lambda: torch.mm(a, b))
    return {"name": f"gemm_bf16 {m}x{k}x{n}", "ms": ms, "gflops": 2.0 * m * k * n / (ms * 1e-3) / 1e9}


def bench_q4_matmul() -> dict[str, object]:
    m, k, n = 1, 4096, 4096
    x = torch.randn(m, k, dtype=torch.float16)
    w = torch.randn(n, k, dtype=torch.float16)
    ms = time_fn(lambda: torch.mm(x, w.T))
    return {"name": f"q4_approx matmul (f16) {m}x{k}x{n}", "ms": ms, "gflops": 2.0 * m * k * n / (ms * 1e-3) / 1e9}


def bench_rmsnorm() -> dict[str, object]:
    batch, seq, hidden = 1, 1, 4096
    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    weight = torch.ones(hidden, dtype=torch.float32)
    eps = 1e-6

    def rms_norm_fn():
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    ms = time_fn(rms_norm_fn)
    return {"name": f"rms_norm hidden={hidden}", "ms": ms, "gflops": float("nan")}


def bench_sdpa() -> dict[str, object]:
    batch, heads, seq, head_dim = 1, 32, 512, 128
    q = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32)
    ms = time_fn(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    flops = batch * heads * seq * seq * head_dim * 2.0
    return {"name": f"sdpa causal B={batch} H={heads} S={seq} D={head_dim}", "ms": ms, "gflops": flops / (ms * 1e-3) / 1e9}


def bench_token_embedding() -> dict[str, object]:
    vocab, dim, batch_seq = 32000, 4096, 512
    emb = torch.nn.Embedding(vocab, dim)
    ids = torch.randint(0, vocab, (batch_seq,))
    ms = time_fn(lambda: emb(ids))
    return {"name": f"token_embed vocab={vocab} dim={dim} n={batch_seq}", "ms": ms, "gflops": float("nan")}


def bench_tokenizer() -> dict[str, object]:
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        text = "The quick brown fox jumps over the lazy dog. " * 50
        ms = time_fn(lambda: tok.encode(text))
        n_tokens = len(tok.encode(text).ids)
        return {"name": f"tokenizer encode ~{n_tokens} tokens", "ms": ms, "gflops": float("nan")}
    except Exception as exc:
        return {"name": "tokenizer (skipped)", "ms": float("nan"), "gflops": float("nan"), "note": str(exc)}


def _print_table(results: list[dict]) -> None:
    print()
    print("## PyTorch CPU Benchmark Results")
    print()
    print(f"{'Operation':<55} {'Median ms':>10} {'GFLOPS':>10}")
    print("-" * 80)
    for r in results:
        gflops_str = f"{r['gflops']:.2f}" if not math.isnan(r['gflops']) else "n/a"
        ms_str = f"{r['ms']:.3f}" if not math.isnan(r['ms']) else "n/a"
        print(f"{r['name']:<55} {ms_str:>10} {gflops_str:>10}")
    print()
    print("Target: Synaptix Rust CPU ≤ 2× each row above.")
    print()


def main() -> None:
    print("Running PyTorch CPU benchmarks...")
    print(f"  torch version: {torch.__version__}")
    print(f"  warmup={WARMUP}, runs={RUNS}, device=cpu")
    print()

    results = [
        bench_gemm_f32(),
        bench_gemm_bf16(),
        bench_q4_matmul(),
        bench_rmsnorm(),
        bench_sdpa(),
        bench_token_embedding(),
        bench_tokenizer(),
    ]

    _print_table(results)

    out_path = pathlib.Path("data/ref/bench_pytorch.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results also saved to {out_path}")


if __name__ == "__main__":
    main()
