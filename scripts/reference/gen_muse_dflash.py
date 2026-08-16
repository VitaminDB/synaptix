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


def embed_rows(model_dir: str, ids: list[int]) -> torch.Tensor:
    """Строки таблицы эмбеддингов target'а напрямую из safetensors-шарда."""
    import json

    from safetensors import safe_open

    root = pathlib.Path(model_dir)
    key = "model.language_model.embed_tokens.weight"
    index = json.load(open(root / "model.safetensors.index.json"))["weight_map"]
    with safe_open(root / index[key], framework="pt") as f:
        sl = f.get_slice(key)
        return torch.cat([sl[i : i + 1, :] for i in ids], dim=0)


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

    noise_id_list = [anchor] + [dcfg.mask_token_id] * (dcfg.block_size - 1)
    noise_ids = torch.tensor([noise_id_list], device="cuda:0")
    # Драфтер эмбеддит БЕЗ RMS-нормы эмбеддинга target'а. После cpu_offload
    # таблица target'а — meta-тензор, поэтому строки читаем прямо из шарда.
    noise_embeds = torch.cat(
        [embed_rows(target_dir, noise_id_list)], dim=0
    )[None, ...].to("cuda:0", torch.bfloat16)

    n_ctx = ctx.shape[1]
    position_ids = torch.arange(n_ctx + dcfg.block_size, device="cuda:0")[None, ...]
    attention_mask = torch.ones(1, n_ctx + dcfg.block_size, device="cuda:0", dtype=torch.long)

    from transformers.cache_utils import DFlashCache

    dcache0 = DFlashCache(config=dcfg)
    dcache0.set_previous_accepted_tokens(ctx.shape[1])
    with torch.no_grad():
        d_out = draft(
            noise_embeds=noise_embeds,
            context_hidden_states=ctx,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=dcache0,
        )
        hidden = d_out.last_hidden_state
        cand_logits = target.lm_head(hidden)[:, 1:]

    cand_ids = cand_logits.argmax(dim=-1)[0]
    print("draft candidates:", cand_ids.tolist(), flush=True)
    print("decoded:", tok.decode(cand_ids.tolist()), flush=True)

    # --- Второй блок: verify первого + повторный draft на инкрементальном контексте
    verify_ids = torch.cat(
        [torch.tensor([[anchor]], device="cuda:0"), cand_ids[None, :]], dim=-1
    )
    full_ids = torch.cat([ids.to("cuda:0"), cand_ids[None, :]], dim=-1)
    with torch.no_grad():
        v_out = target(input_ids=full_ids, output_hidden_states=True, use_cache=False)
    # логиты для позиций verify-чанка (последние block_size)
    v_logits = v_out.logits[:, -dcfg.block_size :]
    preds = v_logits.argmax(dim=-1)[0]
    accepted = 0
    while accepted < cand_ids.shape[0] and int(preds[accepted]) == int(cand_ids[accepted]):
        accepted += 1
    keep = accepted + 1
    anchor2 = int(preds[accepted])
    print(f"accepted={accepted} anchor2={anchor2}", flush=True)

    # контекст второго блока = hidden принятых позиций verify-чанка
    ctx2 = torch.cat(
        [v_out.hidden_states[i + 1][:, -dcfg.block_size :][:, :keep] for i in dcfg.target_layer_ids],
        dim=-1,
    ).to("cuda:0", torch.bfloat16)
    noise2_list = [anchor2] + [dcfg.mask_token_id] * (dcfg.block_size - 1)
    noise2 = embed_rows(target_dir, noise2_list)[None, ...].to("cuda:0", torch.bfloat16)
    # position_ids покрывают только НОВЫЕ входы (контекст этого шага + блок):
    # к старым k/v в кэше RoPE уже применён. attention_mask, наоборот, полная.
    n_ctx2 = ctx.shape[1] + keep
    pos2 = (torch.arange(keep + dcfg.block_size, device="cuda:0") + ctx.shape[1])[None, ...]
    mask2 = torch.ones(1, n_ctx2 + dcfg.block_size, device="cuda:0", dtype=torch.long)
    # Как в DFlashTokenCandidateGenerator: из кэша выселяется прошлое
    # диффузионное окно, а новые k/v контекста учитываются в offset.
    dcache = d_out.past_key_values
    dcache.crop(-dcfg.block_size)
    dcache.set_previous_accepted_tokens(keep)
    with torch.no_grad():
        d2 = draft(
            noise_embeds=noise2,
            context_hidden_states=ctx2,
            position_ids=pos2,
            attention_mask=mask2,
            past_key_values=dcache,
        )
        cand2_logits = target.lm_head(d2.last_hidden_state)[:, 1:]
    cand2_ids = cand2_logits.argmax(dim=-1)[0]
    print("block2 candidates:", cand2_ids.tolist(), flush=True)

    save_file(
        {
            "input_ids": ids[0].to(torch.int64).contiguous(),
            "anchor": torch.tensor([anchor], dtype=torch.int64),
            "context_hidden": ctx[0].float().cpu().contiguous(),
            "draft_hidden": hidden[0].float().cpu().contiguous(),
            "candidate_logits": cand_logits[0].float().cpu().contiguous(),
            "candidate_ids": cand_ids.to(torch.int64).cpu().contiguous(),
            "accepted": torch.tensor([accepted], dtype=torch.int64),
            "anchor2": torch.tensor([anchor2], dtype=torch.int64),
            "context2_hidden": ctx2[0].float().cpu().contiguous(),
            "candidate2_ids": cand2_ids.to(torch.int64).cpu().contiguous(),
            "candidate2_logits": cand2_logits[0].float().cpu().contiguous(),
        },
        str(OUTPUT_DIR / "dflash_ref.safetensors"),
    )
    print("saved", OUTPUT_DIR / "dflash_ref.safetensors", flush=True)


if __name__ == "__main__":
    main()
