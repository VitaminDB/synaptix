"""Generate reference JSON/binary data for Session 6 — Tokenizer.

Run:
    python scripts/reference/gen_tokenizer.py

Requires a Qwen/Llama tokenizer downloaded locally or from HF hub.
Outputs data/ref/tokenizer/<case>.json (ids, strings) for comparison with Rust.

Set TOKENIZER_PATH env var to override default model name:
    TOKENIZER_PATH=Qwen/Qwen2.5-7B-Instruct python scripts/reference/gen_tokenizer.py
"""

import json
import os
import pathlib

from tokenizers import Tokenizer
from transformers import AutoTokenizer

OUTPUT_DIR = pathlib.Path("tests/reference_data/tokenizer")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "models/Qwen/Qwen3-1.7B")


def save_json(name: str, data: object) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved {path}")


def case_encode_decode(tok: AutoTokenizer) -> None:
    texts = [
        "Hello, world!",
        "Привет мир",
        "The quick brown fox jumps over the lazy dog.",
        "def foo(x: int) -> str: return str(x)",
    ]
    results = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        results.append({"text": text, "ids": ids, "decoded": decoded})
    save_json("encode_decode", results)


def case_batch_encode(tok: AutoTokenizer) -> None:
    texts = [
        "Short text",
        "A slightly longer piece of text for padding testing",
        "Medium length string that is somewhere in between",
    ]
    out = tok(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_attention_mask=True,
    )
    save_json(
        "batch_encode",
        {
            "texts": texts,
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
        },
    )


def case_chat_template(tok: AutoTokenizer) -> None:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    formatted = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids_list = list(ids) if hasattr(ids, '__iter__') else ids
    save_json("chat_template", {"formatted": formatted, "ids": ids_list, "messages": messages})


def case_tools_template(tok: AutoTokenizer) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    try:
        formatted = tok.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        save_json("tools_template", {"formatted": formatted, "messages": messages, "tools": tools})
    except Exception as exc:
        print(f"  tools_template skipped: {exc}")


def case_special_tokens(tok: AutoTokenizer) -> None:
    data = {
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "pad_token": tok.pad_token,
        "unk_token": tok.unk_token,
        "bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "unk_token_id": tok.unk_token_id,
        "vocab_size": tok.vocab_size,
    }
    save_json("special_tokens", data)


def case_long_text(tok: AutoTokenizer) -> None:
    text = "The transformer architecture revolutionized natural language processing. " * 128
    ids = tok.encode(text, add_special_tokens=False)
    save_json("long_text", {"len": len(ids), "first_32": ids[:32], "last_32": ids[-32:]})


def case_unicode_edge(tok: AutoTokenizer) -> None:
    texts = [
        "中文测试",
        "éàüñç",
        "\U0001f600\U0001f4a9\U0001f680",
        "مرحبا بالعالم",
        "ハローワールド",
    ]
    results = []
    for text in texts:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids)
        results.append({"text": text, "ids": ids, "decoded": decoded})
    save_json("unicode_edge", results)


def case_streaming_detok(tok: AutoTokenizer) -> None:
    text = "Hello world from streaming tokenizer test."
    ids = tok.encode(text, add_special_tokens=False)
    incremental = []
    accumulated = ""
    for token_id in ids:
        piece = tok.decode([token_id], skip_special_tokens=True)
        accumulated += piece
        incremental.append({"id": token_id, "piece": piece, "accumulated": accumulated})
    save_json("streaming_detok", {"original": text, "steps": incremental})


def main() -> None:
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    print("Generating tokenizer reference data...")
    case_encode_decode(tok)
    case_batch_encode(tok)
    case_chat_template(tok)
    case_tools_template(tok)
    case_special_tokens(tok)
    case_long_text(tok)
    case_unicode_edge(tok)
    case_streaming_detok(tok)
    print("Done.")


if __name__ == "__main__":
    main()
