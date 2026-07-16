"""Bench reference Triton Mamba2 chunk_scan vs наш chunked-CUDA kernel.

Запуск:
    scripts/reference/.venv/bin/python \
      scripts/reference/bench_mamba2_ssd.py

Уровень: bf16 inputs, тот же shape что и наш bench
(B=1, H=64, P=64, N=128, Q=64) на L ∈ {256..8192}.
"""
import sys
sys.path.insert(0, "/tmp/mamba_repo")

import time
import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def bench(L: int, warmup: int = 3, iters: int = 10) -> float:
    B, H, P, N, Q = 1, 64, 64, 128, 64
    device = "cuda"
    dtype = torch.bfloat16

    # Inputs (random — точное содержание не важно для perf).
    x = torch.randn(B, L, H, P, device=device, dtype=dtype) * 0.5
    dt = torch.randn(B, L, H, device=device, dtype=dtype) * 0.2 + 0.5  # >0
    # A per Mamba2: parameter — scalar < 0 per head.
    A = -torch.rand(H, device=device, dtype=torch.float32) * 2.0 - 0.5
    # ngroups = H (group per head, не GQA для simplicity).
    Bp = torch.randn(B, L, H, N, device=device, dtype=dtype) * 0.5
    Cp = torch.randn(B, L, H, N, device=device, dtype=dtype) * 0.5
    D = None  # без skip-D — как наш bench (None в нашем call).

    # Warmup (Triton JIT compile на первом запуске).
    for _ in range(warmup):
        y = mamba_chunk_scan_combined(x, dt, A, Bp, Cp, chunk_size=Q, D=D)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = mamba_chunk_scan_combined(x, dt, A, Bp, Cp, chunk_size=Q, D=D)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return elapsed_ms


def main():
    print(f"Mamba2 SSD: Triton mamba_chunk_scan_combined (BF16, "
          f"B=1, H=64, P=64, N=128, Q=64)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"{'L':>8} {'triton ms':>12}")

    for L in [256, 512, 1024, 2048, 4096, 8192]:
        ms = bench(L)
        print(f"{L:>8} {ms:>12.3f}")


if __name__ == "__main__":
    main()
