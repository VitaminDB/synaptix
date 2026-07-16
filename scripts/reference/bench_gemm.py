"""torch.matmul BF16 CUDA на тех же SDXL FF-формах — таргет для synaptix."""
import torch, time

def bench(m, k, n, iters=50):
    a = torch.zeros(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(k, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    tflops = 2 * m * n * k / dt / 1e12
    print(f"  M={m:5} K={k:5} N={n:6}: {dt*1e3:.3f} ms  {tflops:6.1f} TFLOP/s")

print("torch.matmul BF16 GEMM:")
for sh in [(2048,1280,10240),(2048,5120,1280),(8192,640,5120),(8192,2560,640),(2048,1280,1280),(4096,4096,4096)]:
    bench(*sh)
