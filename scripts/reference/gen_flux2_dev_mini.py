# Эталон урезанного DiT FLUX.2-dev (CPU, F32, diffusers): один double- и один
# single-блок на НАСТОЯЩИХ весах dev (блоки 0), плюс все эмбеддеры — в том
# числе guidance-эмбеддинг, которого нет у klein, — общая модуляция и финал.
# 32B целиком на CPU не поднять; архитектуру проверяет klein, а это — то, что
# есть только у dev. Выход — raw f32 + json (как у gen_flux2_klein.py).
#   python gen_flux2_dev_mini.py …/FLUX.2-dev out_dir
import sys, os, json
import numpy as np, torch
from diffusers import Flux2Transformer2DModel
from safetensors import safe_open
torch.set_num_threads(24)
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
def save(name, t):
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(f"{out}/{name}.f32")
    json.dump({"shape": list(a.shape)}, open(f"{out}/{name}.json", "w"))
    print(name, a.shape, float(a.mean()), float(a.std()), flush=True)
cfg = {k: v for k, v in json.load(open(f"{root}/transformer/config.json")).items() if not k.startswith("_")}
cfg["num_layers"], cfg["num_single_layers"] = 1, 1
model = Flux2Transformer2DModel(**cfg).float().eval()
wmap = json.load(open(f"{root}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
by_file = {}
for k in model.state_dict().keys():
    by_file.setdefault(wmap[k], []).append(k)
sd = {}
for f, keys in by_file.items():
    with safe_open(f"{root}/transformer/{f}", "pt") as h:
        for k in keys:
            sd[k] = h.get_tensor(k).float()
model.load_state_dict(sd, strict=True)
print("loaded", len(sd), "tensors", flush=True)
g = torch.Generator("cpu").manual_seed(7)
H = W = 8; L = 32
img = torch.randn((1, H * W, cfg["in_channels"]), generator=g)
txt = torch.randn((1, L, cfg["joint_attention_dim"]), generator=g)
txt_ids = torch.zeros((1, L, 4)); txt_ids[0, :, 3] = torch.arange(L)
img_ids = torch.zeros((1, H * W, 4))
for y in range(H):
    for x in range(W):
        img_ids[0, y * W + x, 1] = y; img_ids[0, y * W + x, 2] = x
sigma, guidance = 0.7, 4.0
with torch.no_grad():
    v = model(hidden_states=img, encoder_hidden_states=txt, timestep=torch.tensor([sigma]),
              img_ids=img_ids, txt_ids=txt_ids, guidance=torch.tensor([guidance]), return_dict=False)[0]
    # guidance 1.0 — чтобы сравнить именно вклад guidance-эмбеддинга
    v1 = model(hidden_states=img, encoder_hidden_states=txt, timestep=torch.tensor([sigma]),
               img_ids=img_ids, txt_ids=txt_ids, guidance=torch.tensor([1.0]), return_dict=False)[0]
save("mini_img", img); save("mini_txt", txt); save("mini_out", v); save("mini_out_g1", v1)
json.dump({"sigma": sigma, "guidance": guidance, "h": H, "w": W, "txt": L}, open(f"{out}/mini.json", "w"))
