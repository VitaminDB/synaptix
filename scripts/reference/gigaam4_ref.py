# Эталон GigaAM-v4 CTC на CPU: дамп контрольных точек в .npy. Сеть не используется.
import os, sys
sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
import importlib.util, json, omegaconf
from safetensors.torch import load_file
MODEL_DIR, WAV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
# transformers 5.x строит модель в контексте device=meta и падает на torchaudio.MelSpectrogram,
# поэтому AutoModel.from_pretrained не используется: модуль грузится как обычный файл.
spec = importlib.util.spec_from_file_location("modeling_gigaam", os.path.join(MODEL_DIR, "modeling_gigaam.py"))
G = importlib.util.module_from_spec(spec); sys.modules["modeling_gigaam"] = G; spec.loader.exec_module(G)
cfg = omegaconf.OmegaConf.create(json.load(open(os.path.join(MODEL_DIR, "config.json")))["cfg"])
m = G.GigaAMDualASR(cfg)                       # то же, что GigaAMModel.model (стр. 3718)
sd = {k.removeprefix("model."): v for k, v in load_file(os.path.join(MODEL_DIR, "model.safetensors")).items()}
print(m.load_state_dict(sd, strict=True))
m = m.float().eval()                           # как from_pretrained на CPU (стр. 3977)
enc = m.encoder                                # ZapformerEncoder
print("dtype enc:", next(enc.parameters()).dtype, "autocast:", m.cfg.encoder_autocast_dtype)
dumps = {}
def save(name, t):
    dumps[name] = t.detach().float().cpu().numpy(); np.save(os.path.join(OUT, name + ".npy"), dumps[name])
    print(f"{name:28s} {tuple(t.shape)} mean={t.float().mean():+.6f} std={t.float().std():.6f}")
def hook(name, idx=None):
    def f(mod, inp, out):
        save(name, out[idx] if idx is not None else out)
    return f
m.preprocessor.register_forward_hook(hook("01_features", 0))        # [1,64,T] лог-мел
fe = enc.encoder_embed
fe.convnext.register_forward_hook(hook("02_frontend_convnext"))     # [1,64,T/4,15]
fe.register_forward_hook(hook("03_frontend_out", 0))                # [1,T/4,768]
for s, stack in enumerate(enc.encoder.encoders):
    for l, layer in enumerate(stack.layers):
        layer.register_forward_hook(hook(f"04_stack{s}_layer{l:02d}"))   # [1,T/(4*ds),dim]
    stack.register_forward_hook(hook(f"05_stack{s}_out"))               # [1,T/(4*ds),ds*768]
enc.register_forward_hook(hook("06_encoder_out", 0))                # [1,768,T/4] после out_norm
for name, head in m.heads.items():
    head.register_forward_hook(hook(f"07_logprobs_{name}"))         # [1,T/4,C] log_softmax
wav, length = m.prepare_wav(WAV)                                    # ffmpeg -> 16 кГц моно f32
save("00_wav", wav)
encoded, enc_len = m.forward(wav, length)                           # полный контекст (future_context_ms=None)
res = m._decode(encoded, enc_len, length, False)
for name in ("raw", "e2e"):
    lab = torch.from_numpy(dumps[f"07_logprobs_{name}"]).argmax(-1)
    np.save(os.path.join(OUT, f"08_labels_{name}.npy"), lab.numpy())
    print(name, "blank_id =", m.decodings[name].blank_id, "| text:", repr(res[name][0][0]))
print("samples", int(length), "enc_len", int(enc_len))
