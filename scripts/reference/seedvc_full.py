"""Полный прогон эталонного пайплайна Seed-VC v2 (VoiceConversionWrapper, CPU, f32) с записью шума и волны.

SEEDVC_REF=…/seed-vc python -I seedvc_full.py OUT_DIR SRC.wav REF.wav [steps] [режимы: timbre,style]
Пишет: full_timbre_z{i} / full_timbre_wave (без convert_style, CFG 0.7/0.7; i — номер окна),
       full_style_z{i} / full_style_tokens / full_style_wave (convert_style, top_p 0.9, T 1.0),
       ref_full_*.wav — послушать. Шум CFM — torch.Generator(seed 4321 + номер окна).
Шум сэмплера токенов AR подменён на хэш (сид, шаг, токен) — тот же, что `nn::Rng` порта,
иначе стохастическую генерацию не сверить (жадная у эталона не доходит до EOS и упирается
в предел кэша 4096). Сид — 4321.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedvc_common as C

C.setup()
out, src_path, ref_path = sys.argv[1], sys.argv[2], sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
modes = (sys.argv[5] if len(sys.argv) > 5 else "timbre,style").split(",")

from modules.v2.vc_wrapper import VoiceConversionWrapper
from modules.astral_quantization.default_model import AstralQuantizer
from transformers import Wav2Vec2FeatureExtractor


def astral(kind, ssl):
    enc, q = C.build_astral(kind)
    m = AstralQuantizer.__new__(AstralQuantizer)
    torch.nn.Module.__init__(m)
    m.encoder, m.quantizer, m.ssl_model = enc, q, ssl
    m.ssl_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(os.path.join(C.EXT, "hubert-large-ll60k"))
    return m


cfm, cfm_lr = C.build_cfm()
ar, ar_lr = C.build_ar()
w = VoiceConversionWrapper(sr=C.SR, hop_size=C.HOP, mel_fn=C.mel_fn, cfm=cfm, cfm_length_regulator=cfm_lr,
                           content_extractor_narrow=astral("bsq32", None), content_extractor_wide=astral("bsq2048", C.build_hubert()),
                           ar_length_regulator=ar_lr, ar=ar, style_encoder=C.build_campplus(), vocoder=C.build_bigvgan())
w.eval()

state = {"z": [], "wave": None, "tokens": [], "seed": 4321}

orig_stream = w._stream_wave_chunks


def stream(*a, **k):
    r = orig_stream(*a, **k)
    if r[4] is not None:
        state["wave"] = r[4]
    return r


w._stream_wave_chunks = stream


def inference(mu, x_lens, prompt, style, n_timesteps=10, temperature=1.0, inference_cfg_rate=[0.5, 0.5], random_voice=False):
    z = torch.randn([mu.size(0), 80, mu.size(1)], generator=torch.Generator().manual_seed(state["seed"] + len(state["z"])))
    state["z"].append(z[0].clone())
    return cfm.solve_euler(z, x_lens, prompt, mu, style, C.t_span(n_timesteps), inference_cfg_rate, random_voice)


cfm.inference = inference
orig_gen = ar.generate


def generate(*a, **k):
    r = orig_gen(*a, **k)
    state["tokens"].append(r[0].clone())
    return r


ar.generate = generate

# Шум сэмплера: −ln(u), u — SplitMix64 от (сид, шаг, токен); argmax(p / q) в f64.
import modules.v2.ar as ar_mod

M = (1 << 64) - 1
sampler = {"seed": 4321, "step": 0}


def exp_noise(n):
    out = np.empty(n, dtype=np.float64)
    base = (sampler["seed"] + sampler["step"] * 0x9E3779B97F4A7C15) & M
    for tok in range(n):
        z = (base + (tok + 1) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        z ^= z >> 31
        out[tok] = -np.log(((z >> 11) + 1.0) / float(1 << 53))
    return out


def multinomial(probs):
    q = torch.from_numpy(exp_noise(probs.numel()))
    sampler["step"] += 1
    return torch.argmax(probs.double() / q, dim=-1, keepdim=True).to(dtype=torch.int)


ar_mod.multinomial_sample_one_no_sync = multinomial

for name, style in (("timbre", False), ("style", True)):
    if name not in modes:
        continue
    state.update(z=[], wave=None, tokens=[])
    sampler["step"] = 0
    g = w.convert_voice_with_streaming(source_audio_path=src_path, target_audio_path=ref_path, diffusion_steps=steps,
                                       length_adjust=1.0, intelligebility_cfg_rate=0.7, similarity_cfg_rate=0.7,
                                       top_p=0.9, temperature=1.0, repetition_penalty=1.0, convert_style=style,
                                       anonymization_only=False, device=torch.device("cpu"), dtype=torch.float32, stream_output=False)
    for _ in g:
        pass
    for i, z in enumerate(state["z"]):
        C.dump(out, f"full_{name}_z{i}", z)
    C.dump(out, f"full_{name}_wave", state["wave"])
    if style:
        C.dump(out, "full_style_tokens", torch.cat(state["tokens"]))
    sf.write(os.path.join(out, f"ref_full_{name}.wav"), state["wave"], C.SR)
