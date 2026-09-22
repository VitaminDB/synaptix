# Эталон Qwen-Image 2.1 на CPU (diffusers main ≥ 0.41.dev, transformers ≥ 5.17,
# torchvision): VAE RGBA (encode/decode), препроцессинг картинки-референса
# (Lanczos PIL с премультипликацией альфы, композит на белом, процессор
# Qwen3-VL), башня зрения с deepstack, кондиционирование промпта (t2i, с
# картинкой, негатив), DiT на NL блоках в F32 (prefill с KV-кэшем, decode из
# кэша, без кэша), sigmas расписания и сквозной прогон полной модели в BF16
# на маленьком размере. Выход — raw f32 + json (как gen_qwen_image_edit.py).
#
#   PYTHONNOUSERSITE=1 python -s gen_qwen_image21.py <каталог модели> <out>
#   NL=2 — блоков DiT для точной сверки; RES=256 — output_resolution;
#   E2E=0 — пропустить сквозной прогон полной модели.
import sys, os, json, glob, math
import numpy as np, torch
from PIL import Image
from safetensors import safe_open
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift

torch.set_num_threads(os.cpu_count())
root, out = sys.argv[1], sys.argv[2]
NL = int(os.environ.get("NL", 2))
RES = int(os.environ.get("RES", 256))
E2E = os.environ.get("E2E", "1") != "0"
f32 = torch.float32
os.makedirs(out, exist_ok=True)


def save(name, t):
    a = (t.detach().to(f32).cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32))
    a = np.ascontiguousarray(a)
    a.tofile(f"{out}/{name}.f32")
    json.dump({"shape": list(a.shape)}, open(f"{out}/{name}.json", "w"))
    print(name, a.shape, float(a.mean()), float(a.std()), flush=True)


def synth_rgba(w, h, seed):
    """Детерминированная RGBA-картинка: градиенты, круг, полупрозрачная полоса,
    полностью прозрачные углы — чтобы проверить премультипликацию и композит."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (255 * x / max(w - 1, 1)).astype(np.uint8)
    g = (255 * y / max(h - 1, 1)).astype(np.uint8)
    b = (128 + 100 * np.sin(x / 9.0) * np.cos(y / 7.0)).clip(0, 255).astype(np.uint8)
    a = np.full((h, w), 255, np.uint8)
    cx, cy, rad = w * 0.55, h * 0.5, min(w, h) * 0.3
    circle = (x - cx) ** 2 + (y - cy) ** 2 < rad**2
    r[circle], g[circle], b[circle] = 220, 40, 60
    a[: h // 5, : w // 4] = 0
    a[-h // 5 :, -w // 4 :] = 0
    a[:, w // 2 : w // 2 + w // 8] = np.minimum(a[:, w // 2 : w // 2 + w // 8], 96)
    noise = rng.randint(0, 24, size=(h, w), dtype=np.uint8)
    r = np.clip(r.astype(np.int32) + noise - 12, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b, a], -1), "RGBA")


vae = AutoencoderKLQwenImage21.from_pretrained(root, subfolder="vae", torch_dtype=f32).eval()
proc = Qwen3VLProcessor.from_pretrained(root, subfolder="processor")
sched = FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")
te = Qwen3VLForConditionalGeneration.from_pretrained(root, subfolder="text_encoder", torch_dtype=f32).eval()

# DiT: первые NL блоков настоящих весов.
cfg = dict(QwenImage21Transformer2DModel.load_config(root, subfolder="transformer"))
cfg["num_layers"] = NL
tr = QwenImage21Transformer2DModel.from_config(cfg).to(f32).eval()
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

pipe = QwenImage21Pipeline(scheduler=sched, vae=vae, text_encoder=te, processor=proc, transformer=tr)
print("drop_idx", pipe._drop_idx, "img_token", pipe._img_token_id, flush=True)
save("drop_idx", [pipe._drop_idx])

img = synth_rgba(300, 200, 7)
img.save(f"{out}/ref_image.png")
prompt = "Replace the background with a sunset beach; keep the red circle and the gradient stripe unchanged"
t2i_prompt = 'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'

with torch.no_grad():
    # ── VAE: RGBA 256×256 → латент 16×16×64 → обратно ──
    sq = synth_rgba(256, 256, 3)
    sq.save(f"{out}/vae_image.png")
    x = pipe.image_processor.preprocess(sq, height=256, width=256)  # [1,4,H,W] в [-1,1]
    save("vae_input", (x[0] + 1) / 2)
    lat = pipe._encode_vae_image(x.unsqueeze(2), None)  # нормированный [1,64,1,h,w]
    save("vae_latent", lat[:, :, 0])
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    dec = vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
    save("vae_decoded", (dec[0].clamp(-1, 1) + 1) / 2)

    # ── Картинка-референс как в __call__: RGBA → ~RES² кратно 32 ──
    w, h = img.size
    input_w, input_h, _ = calculate_dimensions(RES * RES, w / h)
    print("cond size", input_w, input_h, flush=True)
    save("cond_size", [input_w, input_h])
    cond = pipe.image_processor.resize(img, width=input_w, height=input_h)  # PIL RGBA (премультипликация)
    save("edit_cond_rgba", np.asarray(cond).transpose(2, 0, 1) / 255.0)
    vae_img = pipe.image_processor.preprocess(img, width=input_w, height=input_h).unsqueeze(2)
    save("edit_vae_input", (vae_img[0, :, 0] + 1) / 2)
    white = Image.new("RGB", cond.size, (255, 255, 255))
    white.paste(cond, mask=cond.getchannel("A"))
    save("edit_cond_white", np.asarray(white).transpose(2, 0, 1) / 255.0)

    # Процессор: pixel_values и сетка.
    template = pipe.prompt_template_ti2i.format(prompt)
    inputs = proc(text=[template], images=[white], padding=True, padding_side="left", return_tensors="pt")
    save("edit_pixel_values", inputs.pixel_values)
    save("edit_grid_thw", inputs.image_grid_thw[0])
    save("edit_input_ids", inputs.input_ids[0])
    # Башня зрения: выход merger + deepstack.
    vis_out = te.model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    if isinstance(vis_out, tuple):
        vis_embeds, deep = vis_out[0], vis_out[1]
    else:
        vis_embeds, deep = vis_out.last_hidden_state, getattr(vis_out, "deepstack_feature_lists", None)
    save("edit_vision_embeds", vis_embeds)
    if deep is not None:
        for i, d in enumerate(deep):
            save(f"edit_deepstack_{i}", d)

    # Кондиционирование пайплайна: с картинкой, t2i и негатив.
    pe, pm, pad = pipe.encode_prompt(prompt=prompt, image=[cond], device="cpu")
    save("edit_prompt_embeds", pe[0])
    save("edit_image_pad_mask", pad[0].to(f32))
    assert pm is None, "батч из одного промпта не должен иметь паддинга"
    pe_t, _, pad_t = pipe.encode_prompt(prompt=t2i_prompt, image=None, device="cpu")
    t2i_inputs = proc(text=[pipe.prompt_template_t2i.format(t2i_prompt)], padding=True, padding_side="left", return_tensors="pt")
    save("t2i_input_ids", t2i_inputs.input_ids[0])
    save("t2i_prompt_embeds", pe_t[0])
    pe_n, _, _ = pipe.encode_prompt(prompt=" ", image=None, device="cpu")
    save("neg_prompt_embeds", pe_n[0])

    # ── DiT (NL блоков, F32): t2i 256² и правка с референсом ──
    gen = torch.Generator("cpu").manual_seed(42)
    H = W = RES
    latents_t2i, _ = pipe.prepare_latents(None, 1, 64, H, W, f32, "cpu", gen, None)
    save("dit_t2i_noise", latents_t2i)
    pad_t_full = torch.cat([pad_t, pad_t.new_ones(1, latents_t2i.shape[1] // 4)], dim=1)
    shapes_t2i = [[(1, H // 16, W // 16)]]
    sig1, sig2 = 0.9, 0.5
    o = tr(
        hidden_states=latents_t2i,
        timestep=torch.tensor([sig1], dtype=f32),
        encoder_hidden_states=pe_t,
        encoder_hidden_states_mask=None,
        img_shapes=shapes_t2i,
        img_mask=pad_t_full,
        return_dict=False,
    )[0]
    save("dit_t2i_out", o[:, -latents_t2i.shape[1] :])

    gen = torch.Generator("cpu").manual_seed(42)
    latents, ref_lat = pipe.prepare_latents([vae_img], 1, 64, input_h, input_w, f32, "cpu", gen, None)
    save("dit_edit_noise", latents)
    save("edit_ref_tokens", ref_lat)
    pad_full = torch.cat([pad, pad.new_ones(1, latents.shape[1] // 4)], dim=1)
    shapes = [[(1, input_h // 16, input_w // 16), (1, input_h // 16, input_w // 16)]]
    joint = torch.cat([ref_lat, latents], dim=1)
    kv = QwenImage21KVCache(NL)
    o1 = tr(
        hidden_states=joint,
        timestep=torch.tensor([sig1], dtype=f32),
        encoder_hidden_states=pe,
        encoder_hidden_states_mask=None,
        img_shapes=shapes,
        img_mask=pad_full,
        kv_cache=kv,
        kv_cache_mode="extract",
        return_dict=False,
    )[0]
    save("dit_edit_out1", o1[:, -latents.shape[1] :])
    o2c = tr(
        hidden_states=joint,
        timestep=torch.tensor([sig2], dtype=f32),
        encoder_hidden_states=pe,
        encoder_hidden_states_mask=None,
        img_shapes=shapes,
        img_mask=pad_full,
        kv_cache=kv,
        kv_cache_mode="cached",
        return_dict=False,
    )[0]
    save("dit_edit_out2_cached", o2c[:, -latents.shape[1] :])
    o2f = tr(
        hidden_states=joint,
        timestep=torch.tensor([sig2], dtype=f32),
        encoder_hidden_states=pe,
        encoder_hidden_states_mask=None,
        img_shapes=shapes,
        img_mask=pad_full,
        return_dict=False,
    )[0]
    save("dit_edit_out2_full", o2f[:, -latents.shape[1] :])

    # ── Расписание: 8 шагов на латент t2i и правки ──
    for name, n_tok in [("sigmas_t2i", latents_t2i.shape[1]), ("sigmas_edit", latents.shape[1])]:
        sigmas = np.linspace(1.0, 1 / 8, 8)
        mu = calculate_shift(
            n_tok,
            sched.config.get("base_image_seq_len", 256),
            sched.config.get("max_image_seq_len", 4096),
            sched.config.get("base_shift", 0.5),
            sched.config.get("max_shift", 1.15),
        )
        sched.set_timesteps(8, device="cpu", sigmas=sigmas, mu=mu)
        save(name, sched.sigmas)

if not E2E:
    sys.exit(0)

# ── Сквозной прогон полной модели в BF16 (CPU): t2i и правка, 4 шага ──
del te, tr, pipe
import gc

gc.collect()
bf = torch.bfloat16
te = Qwen3VLForConditionalGeneration.from_pretrained(root, subfolder="text_encoder", torch_dtype=bf).eval()
tr = QwenImage21Transformer2DModel.from_pretrained(root, subfolder="transformer", torch_dtype=bf).eval()
vae_bf = AutoencoderKLQwenImage21.from_pretrained(root, subfolder="vae", torch_dtype=bf).eval()
pipe = QwenImage21Pipeline(scheduler=sched, vae=vae_bf, text_encoder=te, processor=proc, transformer=tr)
with torch.no_grad():
    gen = torch.Generator("cpu").manual_seed(42)
    noise, _ = pipe.prepare_latents(None, 1, 64, RES, RES, f32, "cpu", gen, None)
    save("e2e_t2i_noise", noise)
    lat = pipe(
        prompt=t2i_prompt, height=RES, width=RES, num_inference_steps=4, latents=noise.to(bf), output_type="latent"
    ).images
    save("e2e_t2i_latent", lat)
    unp = pipe._unpack_latents(lat, RES, RES, 16).to(f32)
    dec = vae.decode(unp * std + mean, return_dict=False)[0][:, :, 0]
    save("e2e_t2i_image", (dec[0].clamp(-1, 1) + 1) / 2)

    gen = torch.Generator("cpu").manual_seed(42)
    noise_e, _ = pipe.prepare_latents(None, 1, 64, input_h, input_w, f32, "cpu", gen, None)
    save("e2e_edit_noise", noise_e)
    lat = pipe(
        prompt=prompt,
        image=[img],
        output_resolution=RES,
        num_inference_steps=4,
        latents=noise_e.to(bf),
        output_type="latent",
        use_kv_cache=True,
    ).images
    save("e2e_edit_latent", lat)
    unp = pipe._unpack_latents(lat, input_h, input_w, 16).to(f32)
    dec = vae.decode(unp * std + mean, return_dict=False)[0][:, :, 0]
    save("e2e_edit_image", (dec[0].clamp(-1, 1) + 1) / 2)
print("done", flush=True)
