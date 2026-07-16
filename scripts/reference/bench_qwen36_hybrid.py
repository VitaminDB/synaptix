"""PyTorch/transformers baseline для Qwen3.6-27B (hybrid: GatedDeltaNet+full-attn)
prefill на CUDA — сравнение с synaptix (~134 tok/s на 960 ток, см.
synaptix prefill-graph бенчмарк). Целевое число для проверки:
llama.cpp заявляет >1000 tok/s на 27B-prefill — это и есть истинный baseline
железа, а synaptix-ядро `gated_delta_rule` 5-7× медленнее (parked f32-bmm scan).

Запуск:
  cd scripts/reference
  .venv/bin/python bench_qwen36_hybrid.py
"""
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "storage/LLM_models/unsloth/Qwen3.6-27B"
# Длины prompt'а: совпадают с тем что замерял synaptix run (35, 350, 960, 1400).
LENGTHS = [35, 350, 960, 1400]


def make_prompt_ids(tokenizer, n_tokens: int) -> torch.Tensor:
    """Pad токенами 'The capital of France is Paris.' до n_tokens (та же стратегия
    что в synaptix bench --prompt-tokens). Возвращает [1, n] CUDA tensor."""
    seed = "The capital of France is Paris."
    # Сгенерируем заведомо больше токенов, потом обрежем.
    text = " ".join([seed] * (n_tokens + 10))
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.cuda()


def main():
    print(f"[pytorch] loading {PATH} (bfloat16, sdpa) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        .cuda()
        .eval()
    )
    print(f"[pytorch] loaded in {time.time() - t0:.1f}s")

    # Warmup: 35-ток prompt (NVRTC JIT + autograd graph).
    warmup_ids = make_prompt_ids(tok, 35)
    with torch.no_grad():
        for _ in range(2):
            model(warmup_ids, use_cache=True)
        torch.cuda.synchronize()

    print(f"\n[pytorch] Qwen3.6-27B hybrid prefill baseline (bfloat16, sdpa, RTX 5090):\n")
    print(f"{'prompt (tok)':>14} | {'prefill_ms':>10} | {'tok/s':>8}")
    print(f"{'-'*14}-+-{'-'*10}-+-{'-'*8}")
    for n in LENGTHS:
        ids = make_prompt_ids(tok, n)
        # Прогон prefill: один forward(use_cache=True).
        with torch.no_grad():
            torch.cuda.synchronize()
            t = time.time()
            model(ids, use_cache=True)
            torch.cuda.synchronize()
            dt = time.time() - t
        n_actual = ids.shape[1]
        tps = n_actual / dt
        print(f"{n_actual:>14} | {dt*1000:>10.1f} | {tps:>8.1f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[pytorch] FAILED: {e}", file=sys.stderr)
        raise
