# Эталон Qwen-Image-Edit(-Plus) на CPU (diffusers 0.40 + torchvision, F32): VAE encode/
# decode, башня зрения Qwen2.5-VL, кондиционирование промпта с картинкой
# (полный энкодер), один шаг DiT, урезанного до NL блоков (с zero_cond_t и
# без), sigmas расписания. Выход — raw f32 + json (как gen_flux2_klein.py).
#
#   python gen_qwen_image_edit.py <каталог модели> <картинка> <out>
#   NL=3 — сколько блоков DiT; SIDE=512 — сторона картинки для VAE/DiT.
import sys, os, json, glob
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers import QwenImageEditPlusPipeline, QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE

torch.set_num_threads(os.cpu_count())
root, img_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
NL = int(os.environ.get("NL", 3))
SIDE = int(os.environ.get("SIDE", 512))
f32 = torch.float32
os.makedirs(out, exist_ok=True)


def save(name, t):
    a = (t.detach().to(f32).cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32))
    a = np.ascontiguousarray(a)
    a.tofile(f"{out}/{name}.f32")
    json.dump({"shape": list(a.shape)}, open(f"{out}/{name}.json", "w"))
    print(name, a.shape, float(a.mean()), float(a.std()), flush=True)


index = json.load(open(f"{root}/model_index.json"))
plus = index["_class_name"] == "QwenImageEditPlusPipeline"
print("pipeline", index["_class_name"], flush=True)

vae = AutoencoderKLQwenImage.from_pretrained(root, subfolder="vae", torch_dtype=f32)
tok = Qwen2Tokenizer.from_pretrained(root, subfolder="tokenizer")
proc = Qwen2VLProcessor.from_pretrained(root, subfolder="processor")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(root, subfolder="text_encoder", torch_dtype=f32)
te.eval()

# DiT: первые NL блоков настоящих весов.
cfg = dict(QwenImageTransformer2DModel.load_config(root, subfolder="transformer"))
cfg["num_layers"] = NL
tr = QwenImageTransformer2DModel.from_config(cfg).to(f32).eval()
want = set(tr.state_dict().keys())
sd = {}
for shard in sorted(glob.glob(f"{root}/transformer/*.safetensors")):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if k in want:
                sd[k] = f.get_tensor(k).to(f32)
missing = want - set(sd)
assert not missing, list(missing)[:5]
tr.load_state_dict(sd)
del sd

Pipe = QwenImageEditPlusPipeline if plus else QwenImageEditPipeline
pipe = Pipe(scheduler=sched, vae=vae, text_encoder=te, tokenizer=tok, processor=proc, transformer=tr)

img = Image.open(img_path).convert("RGB")
prompt = "Replace the sky with a starry night and turn the lighthouse light bright blue; keep the cliff and the waves unchanged"

with torch.no_grad():
    # ── VAE: картинка SIDE×SIDE ──
    small = img.resize((SIDE, SIDE), Image.LANCZOS)
    x = pipe.image_processor.preprocess(small, SIDE, SIDE)  # [1,3,H,W] в [-1,1]
    save("vae_input", (x[0] + 1) / 2)
    lat = pipe._encode_vae_image(x.unsqueeze(2), None)  # нормированный [1,16,1,h,w]
    save("vae_latent", lat[:, :, 0])
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    dec = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
    save("vae_decoded", (dec[0].clamp(-1, 1) + 1) / 2)

    # ── Картинка для VL-энкодера ──
    w, h = img.size
    if plus:
        cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
    else:
        cw, ch, _ = __import__(
            "diffusers.pipelines.qwenimage.pipeline_qwenimage_edit", fromlist=["calculate_dimensions"]
        ).calculate_dimensions(1024 * 1024, w / h)
    cond_img = pipe.image_processor.resize(img, ch, cw)
    save("cond_image", torch.from_numpy(np.asarray(cond_img, dtype=np.float32) / 255.0).permute(2, 0, 1))
    enc = proc.image_processor(images=[cond_img], return_tensors="pt")
    save("pixel_values", enc["pixel_values"])
    save("grid_thw", enc["image_grid_thw"].to(f32))
    vis = te.model.visual(enc["pixel_values"], grid_thw=enc["image_grid_thw"]).pooler_output
    save("vision_embeds", vis)

    # ── Промпт ──
    pe, pm = pipe._get_qwen_prompt_embeds(prompt, [cond_img] if plus else cond_img, "cpu")
    save("prompt_embeds", pe)
    ne, _ = pipe._get_qwen_prompt_embeds(" ", [cond_img] if plus else cond_img, "cpu")
    save("negative_embeds", ne)
    # ids ровно как у _get_qwen_prompt_embeds
    if plus:
        base = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
        txt = pipe.prompt_template_encode.format(base + prompt)
    else:
        txt = pipe.prompt_template_encode.format(prompt)
    enc2 = proc(text=[txt], images=[cond_img], padding=True, return_tensors="pt")
    ids = enc2["input_ids"][0].tolist()
    json.dump({"prompt": prompt, "ids": ids, "plus": plus, "drop_idx": pipe.prompt_template_encode_start_idx},
              open(f"{out}/prompt.json", "w"))
    # Пайплайн diffusers 0.40 не передаёт mm_token_type_ids, и transformers 5.x
    # тогда нумерует токены картинки линейно (без M-RoPE). Модель обучалась с
    # 3D-позициями (transformers 4.5x считал их по input_ids) — эталон с ними:
    te.model.rope_deltas = None
    h = te(input_ids=enc2["input_ids"], attention_mask=enc2["attention_mask"], pixel_values=enc2["pixel_values"],
           image_grid_thw=enc2["image_grid_thw"], mm_token_type_ids=enc2["mm_token_type_ids"],
           output_hidden_states=True).hidden_states[-1]
    save("prompt_embeds_mrope", h[:, pipe.prompt_template_encode_start_idx:])

    # ── Один шаг DiT: цель SIDE×SIDE, референс — латент той же картинки ──
    lh = lw = SIDE // 8
    g = torch.Generator("cpu").manual_seed(42)
    noise = torch.randn((1, 16, lh, lw), generator=g, dtype=f32)
    save("noise", noise)
    packed = pipe._pack_latents(noise, 1, 16, lh, lw)
    ref = pipe._pack_latents(lat[:, :, 0], 1, 16, lh, lw)
    inp = torch.cat([packed, ref], dim=1)
    img_shapes = [[(1, lh // 2, lw // 2), (1, lh // 2, lw // 2)]]
    sig = np.linspace(1.0, 1 / 4, 4)
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift
    mu = calculate_shift(packed.shape[1], sched.config.base_image_seq_len, sched.config.max_image_seq_len,
                         sched.config.base_shift, sched.config.max_shift)
    sched.set_timesteps(sigmas=sig, mu=mu)
    save("sigmas", sched.sigmas)
    t = sched.timesteps[1]
    for zc in (True, False):
        tr.zero_cond_t = zc
        for b in tr.transformer_blocks:
            b.zero_cond_t = zc
        v = tr(hidden_states=inp, timestep=(t / 1000).expand(1), encoder_hidden_states=pe,
               encoder_hidden_states_mask=None, img_shapes=img_shapes, return_dict=False)[0]
        save(f"dit_out_zc{int(zc)}", v)
    json.dump({"t_index": 1, "nl": NL, "side": SIDE, "mu": mu}, open(f"{out}/dit.json", "w"))
print("done")
