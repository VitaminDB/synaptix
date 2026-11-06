"""Эталон F5-TTS (github SWivid/F5-TTS: DiT + CFM, utils_infer) и Vocos (charactr/vocos-mel-24khz) для сверки
с synaptix (пример f5_audit). Только оригинальные модули модели; учебные зависимости — заглушки.

    python -I f5_ref.py --pylib P --stub S --repo <F5-TTS>/src --vocos <vocos-репо> --ckpt model.safetensors
        --vocab vocab.txt --vocoder <снимок vocos> --ref ref.wav --ref-text "..." --text "..." --out каталог
        [--nfe 32] [--cfg 2] [--sway -1] [--speed 1] [--seed 0]

Шум y0 — наш ГПСЧ (`synaptix_tts_f5::noise`): splitmix64, Бокс — Мюллер, порядок [кадр][канал].
Выход: text_ids.npy, ref_mel.npy ([n, 100]), duration.txt, mel.npy (сгенерированный, без образца), audio.wav.
Обработка образца как в infer_batch_process: громкость до target_rms 0.1, ресемпл 24 кГц (без обрезки тишины).
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

GOLD = 0x9E3779B97F4A7C15


def our_normal(seed, n):
    with np.errstate(over="ignore"):
        idx = np.arange(1, 2 * n + 1, dtype=np.uint64)
        z = np.uint64(seed) + idx * np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(40)).astype(np.float64) + 0.5) / float(1 << 24)
    return (np.sqrt(-2.0 * np.log(u[0::2])) * np.cos(2.0 * np.pi * u[1::2])).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    for k in ["pylib", "stub", "repo", "vocos", "ckpt", "vocab", "vocoder", "ref", "ref-text", "text", "out"]:
        ap.add_argument("--" + k)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--f16", action="store_true", help="веса DiT, округлённые до F16 (как в .syn)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    for p in [a.stub, a.pylib, a.repo, a.vocos]:
        if p:
            sys.path.insert(0, p)
    import soundfile as sf
    import torch
    import torchaudio
    from safetensors.torch import load_file
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.cfm import CFM
    from f5_tts.model.utils import convert_char_to_pinyin

    torch.set_num_threads(os.cpu_count() or 8)
    os.makedirs(a.out, exist_ok=True)
    vocab = {}
    with open(a.vocab, encoding="utf-8") as f:
        for i, ch in enumerate(f):
            vocab[ch[:-1]] = i
    dit = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
              text_num_embeds=len(vocab), mel_dim=100)
    model = CFM(transformer=dit, mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                                                     target_sample_rate=24000, mel_spec_type="vocos"),
                odeint_kwargs=dict(method="euler"), vocab_char_map=vocab)
    sd = load_file(a.ckpt)
    sd = {k.replace("ema_model.", ""): v for k, v in sd.items() if k not in ("initted", "step")}
    sd = {k: v for k, v in sd.items() if not k.startswith("mel_spec.")}
    if a.f16:
        sd = {k: (v.half().float() if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("mel_spec.")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = model.float().eval().to(a.device)

    # образец как в infer_batch_process
    wav, sr = sf.read(a.ref, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.mean(axis=1))[None]
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < 0.1:
        audio = audio * 0.1 / rms
    if sr != 24000:
        audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    ref_text = a.ref_text
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    gen = a.text
    speed = a.speed * (0.3 if len(gen.encode("utf-8")) < 10 else 1.0)
    text_list = convert_char_to_pinyin([ref_text + gen])
    ids = [vocab.get(c, 0) for c in text_list[0]]
    np.save(os.path.join(a.out, "text_ids.npy"), np.array(ids, dtype=np.int64))
    ref_len = audio.shape[-1] // 256
    duration = ref_len + int(ref_len / len(ref_text.encode("utf-8")) * len(gen.encode("utf-8")) / speed)
    open(os.path.join(a.out, "duration.txt"), "w").write(f"{ref_len} {duration}\n")

    cond = model.mel_spec(audio.to(a.device)).permute(0, 2, 1)  # [1, n, 100]
    np.save(os.path.join(a.out, "ref_mel.npy"), cond[0].cpu().numpy())
    orig_randn = torch.randn

    def randn(*shape, **kw):
        if len(shape) == 2 and shape[1] == 100:
            n = shape[0] * shape[1]
            return torch.from_numpy(our_normal(a.seed, n).reshape(shape)).to(dtype=kw.get("dtype") or torch.float32, device=kw.get("device") or "cpu")
        return orig_randn(*shape, **kw)

    torch.randn = randn
    with torch.inference_mode():
        out, _ = model.sample(cond=audio.to(a.device), text=text_list, duration=duration, steps=a.nfe, cfg_strength=a.cfg,
                              sway_sampling_coef=a.sway)
    torch.randn = orig_randn
    mel = out[0, ref_len:].float().cpu()
    np.save(os.path.join(a.out, "mel.npy"), mel.numpy())

    # Vocos из исходников (без зависимостей пакета)
    def mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    import yaml
    cfg = yaml.safe_load(open(os.path.join(a.vocoder, "config.yaml")))
    from vocos.models import VocosBackbone
    from vocos.heads import ISTFTHead
    bb = VocosBackbone(**cfg["backbone"]["init_args"])
    head = ISTFTHead(**cfg["head"]["init_args"])
    vsd = torch.load(os.path.join(a.vocoder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    bb.load_state_dict({k[9:]: v for k, v in vsd.items() if k.startswith("backbone.")})
    head.load_state_dict({k[5:]: v for k, v in vsd.items() if k.startswith("head.")})
    with torch.inference_mode():
        wave = head(bb(mel.T[None]))[0]
    if rms < 0.1:
        wave = wave * rms / 0.1
    sf.write(os.path.join(a.out, "audio.wav"), wave.numpy(), 24000, subtype="FLOAT")
    print(f"ids {len(ids)}, ref {ref_len} кадров, всего {duration}, звук {wave.numel() / 24000:.2f} с")


if __name__ == "__main__":
    main()
