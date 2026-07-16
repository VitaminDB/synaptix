"""Дамп torch bf16 GEMM (X@Wᵀ) на FLUX-формах для проверки нашего gemm_bf16.
Все вычисления на CUDA bf16 (cuBLAS, f32-аккумулятор) — как в Python FLUX."""
import pathlib
import torch

OUT = pathlib.Path("tests/reference_data/gemm_bf16")
# (M, N, K) реальных FLUX Linear-форм
SHAPES = [
    (1024, 3072, 64),     # x_embedder (малый K)
    (1024, 64, 3072),     # proj_out (малый N)
    (512, 3072, 4096),    # context_embedder
    (1536, 12288, 3072),  # ff.0
    (1536, 3072, 12288),  # ff.2
    (1536, 3072, 15360),  # single proj_out
    (1536, 18432, 3072),  # norm1
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    torch.manual_seed(0)
    d = {}
    for (m, n, k) in SHAPES:
        x = torch.randn(m, k, dtype=torch.float32).to("cuda", torch.bfloat16)
        w = torch.randn(n, k, dtype=torch.float32).to("cuda", torch.bfloat16) * 0.1
        y = torch.matmul(x, w.t())  # [m,n] bf16, cuBLAS f32-acc
        tag = f"{m}_{n}_{k}"
        d[f"{tag}.x"] = x.cpu()
        d[f"{tag}.w"] = w.cpu()
        d[f"{tag}.y"] = y.cpu()
        print(tag, "y range", float(y.min()), float(y.max()))
    save_file(d, str(OUT / "ref.safetensors"))
    print("saved", OUT / "ref.safetensors")


if __name__ == "__main__":
    main()
