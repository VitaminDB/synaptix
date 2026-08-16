"""Muse-Glimmer-30B reference tensors for synaptix parity tests.

GPU-only compute: LM weights stay in host RAM and stream per-module to CUDA via
accelerate.cpu_offload; the 1.8B vision tower runs fully resident on CUDA.

Run (under systemd-run per repo policy for heavy runs):
    systemd-run --user --scope -p MemoryMax=88G \
      env PYTHONPATH=/run/media/storage/tmp/muse_ref_venv/lib/python3.14/site-packages \
      /home/master/Temp/LTX-2/.venv/bin/python scripts/reference/gen_muse_glimmer.py \
      /run/media/storage/LLM_models/meta-models/Muse-Glimmer-30B

Outputs: tests/reference_data/muse_glimmer/{text_ref,vision_ref}.safetensors
"""

import json
import pathlib
import sys

import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/muse_glimmer")

PROMPT_MESSAGES = [
    {"role": "system", "content": "Отвечай кратко, одним предложением."},
    {"role": "user", "content": "Столица Франции?"},
]

GREEDY_TOKENS = 16


def gen_vision(model, model_dir: str) -> None:
    cfg = json.load(open(pathlib.Path(model_dir) / "config.json"))
    vc = cfg["vision_config"]
    patch_dim = vc["patch_temporal"] * 3 * vc["patch_size"] ** 2

    vision = model.model.vision_tower.to("cuda:0")
    adapter = model.model.vision_adapter.to("cuda:0")
    projection = model.model.vision_projection.to("cuda:0")
    emb_norm = model.model.perception_emb_norm.to("cuda:0")

    torch.manual_seed(42)
    grid = torch.tensor([[1, 34, 46]], dtype=torch.long)
    n = int(grid.prod().item())
    pixel_values = torch.randn(n, patch_dim, dtype=torch.float32) * 0.8

    with torch.no_grad():
        towered = vision(
            pixel_values=pixel_values.to("cuda:0", torch.bfloat16),
            grid_thw=grid.to("cuda:0"),
        ).last_hidden_state
        feats = emb_norm(projection(adapter(towered)))

    save_file(
        {
            "pixel_values": pixel_values.contiguous(),
            "grid_thw": grid[0].contiguous(),
            "tower_out": towered.float().cpu().contiguous(),
            "features": feats.float().cpu().contiguous(),
        },
        str(OUTPUT_DIR / "vision_ref.safetensors"),
    )
    print(f"vision: {n} patches → {tuple(feats.shape)} features", flush=True)

    model.model.vision_tower.to("cpu")
    model.model.vision_adapter.to("cpu")
    model.model.vision_projection.to("cpu")
    model.model.perception_emb_norm.to("cpu")
    torch.cuda.empty_cache()


def gen_text(model, tok) -> None:
    from accelerate import cpu_offload

    prompt = tok.apply_chat_template(
        PROMPT_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"prompt ({ids.shape[1]} tokens): {prompt[:120]!r}...", flush=True)

    cpu_offload(model, execution_device="cuda:0")

    with torch.no_grad():
        out = model(input_ids=ids.to("cuda:0"), output_hidden_states=True, use_cache=False)
    logits = out.logits[0].float().cpu()
    hidden_last = torch.stack(
        [h[0, -1, :].float().cpu() for h in out.hidden_states], dim=0
    )

    greedy = [int(torch.argmax(logits[-1]).item())]
    with torch.no_grad():
        cur = torch.cat([ids, torch.tensor([[greedy[-1]]])], dim=1)
        for _ in range(GREEDY_TOKENS - 1):
            step = model(input_ids=cur.to("cuda:0"), use_cache=False)
            nxt = int(torch.argmax(step.logits[0, -1].float()).item())
            greedy.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
            print("greedy so far:", greedy, flush=True)

    save_file(
        {
            "input_ids": ids[0].to(torch.int64).contiguous(),
            "logits_last": logits[-1].contiguous(),
            "logits_first": logits[0].contiguous(),
            "hidden_last_token": hidden_last.contiguous(),
            "greedy_ids": torch.tensor(greedy, dtype=torch.int64),
        },
        str(OUTPUT_DIR / "text_ref.safetensors"),
    )
    print("greedy continuation:", greedy, flush=True)
    print("decoded:", tok.decode(greedy), flush=True)


def main() -> None:
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    model_dir = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print("model loaded", flush=True)

    if mode in ("all", "vision"):
        gen_vision(model, model_dir)
    if mode in ("all", "text"):
        gen_text(model, tok)


if __name__ == "__main__":
    main()
