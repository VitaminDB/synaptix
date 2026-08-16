"""DFlash-драфтер Muse-Glimmer: эталонные тензоры для паритета synaptix.

Прогоняет target (стримингом весов на CUDA через accelerate.cpu_offload) для
получения hidden-состояний target_layer_ids, затем один draft-forward
ассистента и сохраняет вход/выход каждой стадии.

Run:
    systemd-run --user --scope -p MemoryMax=88G \
      env PYTHONPATH=/run/media/storage/tmp/muse_ref_venv/lib/python3.14/site-packages \
      /home/master/Temp/LTX-2/.venv/bin/python scripts/reference/gen_muse_dflash.py \
      /run/media/storage/LLM_models/meta-models/Muse-Glimmer-30B \
      /run/media/storage/LLM_models/meta-models/Muse-Glimmer-30B-assistant

Output: tests/reference_data/muse_glimmer/dflash_ref.safetensors
"""

import pathlib
import sys

import torch
from safetensors.torch import save_file

OUTPUT_DIR = pathlib.Path("tests/reference_data/muse_glimmer")

PROMPT_MESSAGES = [
    {"role": "system", "content": "Отвечай кратко, одним предложением."},
    {"role": "user", "content": "Столица Франции?"},
]


def main() -> None:
    from accelerate import cpu_offload
    from transformers import (
        AutoTokenizer,
        MuseGlimmerAssistantModel,
        MuseGlimmerForConditionalGeneration,
    )

    target_dir, draft_dir = sys.argv[1], sys.argv[2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(target_dir)
    prompt = tok.apply_chat_template(
        PROMPT_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"prompt: {ids.shape[1]} tokens", flush=True)

    target = MuseGlimmerForConditionalGeneration.from_pretrained(
        target_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    cpu_offload(target, execution_device="cuda:0")

    with torch.no_grad():
        out = target(input_ids=ids.to("cuda:0"), output_hidden_states=True, use_cache=False)
    anchor = int(torch.argmax(out.logits[0, -1].float()).item())
    print("anchor token:", anchor, flush=True)

    draft = MuseGlimmerAssistantModel.from_pretrained(
        draft_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().to("cuda:0")
    dcfg = draft.config

    # Контекст = hidden всех токенов промпта с target_layer_ids (как в
    # DFlashTokenCandidateGenerator: hidden_states[i + 1]).
    ctx = torch.cat(
        [out.hidden_states[i + 1] for i in dcfg.target_layer_ids], dim=-1
    ).to("cuda:0", torch.bfloat16)

    noise_ids = torch.tensor(
        [[anchor] + [dcfg.mask_token_id] * (dcfg.block_size - 1)], device="cuda:0"
    )
    # Драфтер эмбеддит БЕЗ RMS-нормы эмбеддинга target'а.
    emb_table = target.get_input_embeddings().weight
    noise_embeds = torch.nn.functional.embedding(noise_ids, emb_table).to(
        "cuda:0", torch.bfloat16
    )

    n_ctx = ctx.shape[1]
    position_ids = torch.arange(n_ctx + dcfg.block_size, device="cuda:0")[None, ...]
    attention_mask = torch.ones(1, n_ctx + dcfg.block_size, device="cuda:0", dtype=torch.long)

    with torch.no_grad():
        d_out = draft(
            noise_embeds=noise_embeds,
            context_hidden_states=ctx,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden = d_out.last_hidden_state
        cand_logits = target.lm_head(hidden)[:, 1:]

    cand_ids = cand_logits.argmax(dim=-1)[0]
    print("draft candidates:", cand_ids.tolist(), flush=True)
    print("decoded:", tok.decode(cand_ids.tolist()), flush=True)

    save_file(
        {
            "input_ids": ids[0].to(torch.int64).contiguous(),
            "anchor": torch.tensor([anchor], dtype=torch.int64),
            "context_hidden": ctx[0].float().cpu().contiguous(),
            "draft_hidden": hidden[0].float().cpu().contiguous(),
            "candidate_logits": cand_logits[0].float().cpu().contiguous(),
            "candidate_ids": cand_ids.to(torch.int64).cpu().contiguous(),
        },
        str(OUTPUT_DIR / "dflash_ref.safetensors"),
    )
    print("saved", OUTPUT_DIR / "dflash_ref.safetensors", flush=True)


if __name__ == "__main__":
    main()
