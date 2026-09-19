# Эталон prompt_embeds FLUX.2-dev: Mistral3 text-часть, слой за слоем (F32, CPU),
# маска и RoPE — родными функциями transformers, как в пайплайне diffusers.
# `<out>/prompt.json` — {"prompt", "ids", "mask"}: ids брать из «сырого»
# `tokenizers.Tokenizer.from_file(tokenizer.json)` + левый паддинг id 11 до 512.
# transformers 5.x грузит этот токенайзер как LlamaTokenizer и теряет пробелы
# («arean», «atre») — модель выпускалась на 4.57, где токенизация верная.
import json, sys, time, torch
from safetensors import safe_open
from transformers import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding
from transformers.masking_utils import create_causal_mask
torch.set_num_threads(24)
root, out = sys.argv[1], sys.argv[2]
cfg_all = json.load(open(f"{root}/text_encoder/config.json"))
cfg = MistralConfig(**cfg_all["text_config"])
cfg._attn_implementation = "sdpa"
idx = json.load(open(f"{root}/text_encoder/model.safetensors.index.json"))["weight_map"]
def get(name):
    with safe_open(f"{root}/text_encoder/{idx[name]}", "pt") as f:
        return f.get_tensor(name).float()
p = json.load(open(f"{out}/prompt.json"))
ids = torch.tensor([p["ids"]]); am = torch.tensor([p["mask"]])
S = ids.shape[1]
P = "language_model.model"
x = torch.nn.functional.embedding(ids, get(f"{P}.embed_tokens.weight"))
cache_position = torch.arange(S); position_ids = cache_position.unsqueeze(0)
mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=am, past_key_values=None, position_ids=position_ids)
rot = MistralRotaryEmbedding(cfg)
pe = rot(x, position_ids)
taps = {10: None, 20: None, 30: None}
layer = MistralDecoderLayer(cfg, 0).float().eval()
t0 = time.time()
with torch.no_grad():
    for i in range(30):
        sd = {k[len(f"{P}.layers.{i}."):]: get(k) for k in idx if k.startswith(f"{P}.layers.{i}.")}
        layer.load_state_dict(sd)
        r = layer(x, attention_mask=mask, position_ids=position_ids, position_embeddings=pe)
        x = r[0] if isinstance(r, tuple) else r
        if i + 1 in taps: taps[i + 1] = x.clone()
        print("layer", i, round(time.time() - t0, 1), float(x[0, -1].abs().mean()), bool(torch.isfinite(x).all()), flush=True)
emb = torch.stack([taps[k] for k in (10, 20, 30)], dim=1).permute(0, 2, 1, 3).reshape(1, S, -1)
a = emb.numpy(); a.tofile(f"{out}/prompt_embeds.f32")
json.dump({"shape": list(a.shape)}, open(f"{out}/prompt_embeds.json", "w"))
print("done", a.shape, float(a.std()), "pad row std", float(a[0, 0].std()))
