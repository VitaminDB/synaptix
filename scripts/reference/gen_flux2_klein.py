# Эталон FLUX.2 klein на CPU (diffusers 0.40): кондиционирование, один шаг DiT,
# полный прогон с заданным шумом, VAE encode/decode. Выход — raw f32 + json.
import sys, os, json, time
import numpy as np, torch
from diffusers import Flux2KleinPipeline
torch.set_num_threads(24)
root, out = sys.argv[1], sys.argv[2]
W = int(os.environ.get("W", 512)); H = int(os.environ.get("H", 512)); STEPS = int(os.environ.get("STEPS", 4))
dtype = torch.float32 if os.environ.get("DT", "f32") == "f32" else torch.bfloat16
os.makedirs(out, exist_ok=True)
def save(name, t):
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(f"{out}/{name}.f32")
    json.dump({"shape": list(a.shape)}, open(f"{out}/{name}.json", "w"))
    print(name, a.shape, float(a.mean()), float(a.std()), flush=True)
t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained(root, torch_dtype=dtype)
print("loaded", time.time() - t0, flush=True)
prompt = "a red fox sitting in fresh snow, golden hour, photo"
with torch.no_grad():
    tok = pipe.tokenizer
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    json.dump({"text": text, "prompt": prompt, "padding_side": tok.padding_side,
               "ids": tok(text, padding="max_length", max_length=512, truncation=True)["input_ids"]},
              open(f"{out}/prompt.json", "w"))
    pe, tids = pipe.encode_prompt(prompt=prompt, device="cpu", max_sequence_length=512)
    save("prompt_embeds", pe)
    g = torch.Generator("cpu").manual_seed(42)
    lat = torch.randn((1, 128, H // 16, W // 16), generator=g, dtype=torch.float32)
    save("noise", lat)
    # один шаг DiT на первой сигме
    packed = lat.reshape(1, 128, -1).permute(0, 2, 1).to(dtype)
    ids = pipe._prepare_latent_ids(lat)
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    sig = np.linspace(1.0, 1 / STEPS, STEPS)
    mu = compute_empirical_mu(packed.shape[1], STEPS)
    pipe.scheduler.set_timesteps(sigmas=sig, mu=mu)
    save("sigmas", pipe.scheduler.sigmas)
    t = pipe.scheduler.timesteps[0]
    t1 = time.time()
    v = pipe.transformer(hidden_states=packed, timestep=(t / 1000).expand(1).to(dtype), guidance=None,
                         encoder_hidden_states=pe.to(dtype), txt_ids=tids, img_ids=ids, return_dict=False)[0]
    print("dit step", time.time() - t1, flush=True)
    save("dit_out0", v)
    img = pipe(prompt_embeds=pe, height=H, width=W, num_inference_steps=STEPS,
               latents=lat.to(dtype), output_type="pt").images
    save("image", img)
    # VAE: encode той же картинки (mode) и decode латента шума
    x = img.to(dtype) * 2 - 1
    enc = pipe.vae.encode(x).latent_dist.mode()
    save("vae_enc_mode", enc)
    dec = pipe.vae.decode(enc).sample
    save("vae_dec", dec)
print("total", time.time() - t0)
