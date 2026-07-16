"""Generate reference SafeTensors for Session 10 — Inference sampling.

Run:
    python scripts/reference/gen_infer.py

Compares Synaptix sampling ops against transformers LogitsProcessor equivalents.
Uses fixed seeds for deterministic top-p/top-k.
Outputs data/ref/infer/<case>.safetensors.
"""

import pathlib

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/infer")
VOCAB_SIZE = 32000
BATCH = 4


def save_case(name: str, tensors: dict[str, torch.Tensor]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    print(f"  saved {path}")


def case_greedy_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(BATCH, VOCAB_SIZE, dtype=torch.float32)
    tokens = logits.argmax(dim=-1).to(torch.int64)
    save_case("greedy_argmax", {"logits": logits, "tokens": tokens})


def case_temperature_scaling() -> None:
    torch.manual_seed(1)
    logits = torch.randn(BATCH, VOCAB_SIZE, dtype=torch.float32)
    temperature = 0.8
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    save_case(
        "temperature_scaling",
        {
            "logits": logits,
            "temperature": torch.tensor([temperature]),
            "scaled_logits": scaled,
            "probs": probs,
        },
    )


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k == 0:
        return logits
    kth_values = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_values, float("-inf"))


def case_top_k_filter() -> None:
    torch.manual_seed(2)
    top_k = 50
    logits = torch.randn(BATCH, VOCAB_SIZE, dtype=torch.float32)
    filtered = _top_k_filter(logits.clone(), top_k)
    save_case(
        "top_k_filter",
        {
            "logits": logits,
            "top_k": torch.tensor([top_k], dtype=torch.int64),
            "filtered_logits": filtered,
        },
    )


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_remove] = float("-inf")
    output = torch.zeros_like(logits)
    output.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return output


def case_top_p_filter() -> None:
    torch.manual_seed(3)
    top_p = 0.9
    logits = torch.randn(BATCH, VOCAB_SIZE, dtype=torch.float32)
    filtered = _top_p_filter(logits.clone(), top_p)
    save_case(
        "top_p_filter",
        {
            "logits": logits,
            "top_p": torch.tensor([top_p]),
            "filtered_logits": filtered,
        },
    )


def case_combined_sampling() -> None:
    torch.manual_seed(4)
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    logits = torch.randn(BATCH, VOCAB_SIZE, dtype=torch.float32)
    scaled = logits / temperature
    scaled = _top_k_filter(scaled, top_k)
    scaled = _top_p_filter(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    save_case(
        "combined_sampling",
        {
            "logits": logits,
            "temperature": torch.tensor([temperature]),
            "top_k": torch.tensor([top_k], dtype=torch.int64),
            "top_p": torch.tensor([top_p]),
            "final_probs": probs,
        },
    )


def main() -> None:
    print("Generating inference sampling reference data...")
    case_greedy_argmax()
    case_temperature_scaling()
    case_top_k_filter()
    case_top_p_filter()
    case_combined_sampling()
    print("Done.")


if __name__ == "__main__":
    main()
