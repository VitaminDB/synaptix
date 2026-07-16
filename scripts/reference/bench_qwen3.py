"""PyTorch/transformers baseline для Qwen3-1.7B BF16 на CUDA — сравнение с synaptix.
prefill: forward по длинному промпту; decode: greedy generate 500 новых токенов."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "models/Qwen/Qwen3-1.7B"
DECODE_TOKENS = 500

tok = AutoTokenizer.from_pretrained(PATH)
model = (
    AutoModelForCausalLM.from_pretrained(
        PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .cuda()
    .eval()
)

# ── prefill: длинный промпт ──
long_prompt = "The quick brown fox jumps over the lazy dog. " * 60
pids = tok(long_prompt, return_tensors="pt").input_ids.cuda()
n_prompt = pids.shape[1]
with torch.no_grad():
    for _ in range(2):  # warmup
        model(pids, use_cache=True)
    torch.cuda.synchronize()
    t = time.time()
    model(pids, use_cache=True)
    torch.cuda.synchronize()
    prefill_s = time.time() - t
print(f"[pytorch] prefill: {n_prompt} tok in {prefill_s*1000:.1f} ms = {n_prompt/prefill_s:.1f} tok/s")

# ── decode: greedy generate 500 ──
short = tok("Once upon a time", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    model.generate(short, max_new_tokens=4, do_sample=False, use_cache=True)  # warmup
    torch.cuda.synchronize()
    t = time.time()
    out = model.generate(short, max_new_tokens=DECODE_TOKENS, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    gen_s = time.time() - t
new = out.shape[1] - short.shape[1]
print(f"[pytorch] decode: {new} new tok in {gen_s*1000:.1f} ms = {new/gen_s:.1f} tok/s (incl tiny prefill)")
