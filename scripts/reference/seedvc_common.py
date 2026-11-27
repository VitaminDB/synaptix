"""Общее для дампов эталона Seed-VC v2 (CPU, f32): пути, сборка модулей, запись тензоров.

Эталонный код — каталог seed-vc (env SEEDVC_REF), веса — _hf/Seed-VC и _hf/Seed-VC-ext.
Тензоры пишутся сырыми little-endian: `имя.AxBxC.f32` / `.i32`.
"""
import os
import sys
import math
import numpy as np
import torch

HF = os.path.expanduser("~/Storage/Нейросети/Модели/_hf")
REF = os.environ.get("SEEDVC_REF", "")
MAIN = os.path.join(HF, "Seed-VC")
EXT = os.path.join(HF, "Seed-VC-ext")
SR = 22050
HOP = 256


def setup():
    assert REF and os.path.isdir(REF), "SEEDVC_REF — каталог эталонного кода seed-vc"
    sys.path.insert(0, REF)
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("SEEDVC_THREADS", "8")))


def dump(d, name, t):
    os.makedirs(d, exist_ok=True)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t)
    dims = "x".join(str(x) for x in t.shape) if t.ndim else "1"
    if np.issubdtype(t.dtype, np.integer) or t.dtype == np.bool_:
        t.astype("<i4").tofile(os.path.join(d, f"{name}.{dims}.i32"))
    else:
        t.astype("<f4").tofile(os.path.join(d, f"{name}.{dims}.f32"))
    print(f"{name}: {tuple(t.shape)}")


def wave(shape, a, w, p):
    """a·sin(w·i + p), i — плоский индекс (f64 → f32); так же считает seedvc_check."""
    n = int(np.prod(shape))
    return torch.from_numpy((a * np.sin(w * np.arange(n, dtype=np.float64) + p)).astype(np.float32)).reshape(shape)


def toks(n, mod, mul, add):
    return torch.tensor([(i * mul + add) % mod for i in range(n)], dtype=torch.long)


def strip(sd, prefix="module."):
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}


def mel_fn(y):
    from modules.audio import mel_spectrogram
    return mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=SR, hop_size=HOP, win_size=1024, fmin=0, fmax=None, center=False)


def build_cfm():
    from modules.v2.cfm import CFM
    from modules.v2.dit_wrapper import DiT
    from modules.v2.length_regulator import InterpolateRegulator
    est = DiT(time_as_token=True, style_as_token=True, uvit_skip_connection=False, block_size=8192, depth=13, num_heads=8,
              hidden_dim=512, in_channels=80, content_dim=512, style_encoder_dim=192, class_dropout_prob=0.1,
              dropout_rate=0.0, attn_dropout_rate=0.0)
    cfm = CFM(est)
    lr = InterpolateRegulator(channels=512, is_discrete=True, codebook_size=2048, sampling_ratios=[1, 1, 1, 1], f0_condition=False)
    ck = torch.load(os.path.join(MAIN, "v2/cfm_small.pth"), map_location="cpu")
    print("cfm:", cfm.load_state_dict(strip(ck["net"]["cfm"]), strict=False))
    print("cfm lr:", lr.load_state_dict(strip(ck["net"]["length_regulator"]), strict=False))
    return cfm.eval().float(), lr.eval().float()


def build_ar():
    from modules.v2.ar import NaiveWrapper, NaiveTransformer, NaiveModelArgs
    from modules.v2.length_regulator import InterpolateRegulator
    args = NaiveModelArgs(dropout=0.0, rope_base=10000.0, dim=768, head_dim=64, n_local_heads=2, intermediate_size=2304,
                          n_head=12, n_layer=12, vocab_size=2049)
    ar = NaiveWrapper(NaiveTransformer(args))
    lr = InterpolateRegulator(channels=768, is_discrete=True, codebook_size=32, sampling_ratios=[], f0_condition=False)
    ck = torch.load(os.path.join(MAIN, "v2/ar_base.pth"), map_location="cpu")
    print("ar:", ar.load_state_dict(strip(ck["net"]["ar"]), strict=False))
    print("ar lr:", lr.load_state_dict(strip(ck["net"]["length_regulator"]), strict=False))
    ar = ar.eval().float()
    ar.setup_caches(max_batch_size=1, max_seq_len=4096, dtype=torch.float32, device=torch.device("cpu"))
    return ar, lr.eval().float()


def campplus_path():
    for p in (os.path.join(EXT, "campplus/campplus_cn_common.bin"), os.path.join(REF, "campplus_cn_common.bin")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("campplus_cn_common.bin")


def build_campplus():
    from modules.campplus.DTDNN import CAMPPlus
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(campplus_path(), map_location="cpu"), strict=False)
    return m.eval()


def fbank(wave_16k):
    """Как VoiceConversionWrapper.compute_style: kaldi fbank − среднее по времени."""
    import torchaudio.compliance.kaldi as kaldi
    feat = kaldi.fbank(wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    return feat - feat.mean(dim=0, keepdim=True)


def build_hubert():
    from transformers import HubertModel
    m = HubertModel.from_pretrained(os.path.join(EXT, "hubert-large-ll60k")).eval()
    m.encoder.layers = m.encoder.layers[:18]
    m.encoder.layer_norm = torch.nn.Identity()
    return m


def build_astral(kind):
    """kind: 'bsq32' | 'bsq2048' → (encoder, quantizer)."""
    from modules.astral_quantization.convnext import ConvNeXtV2Stage
    from modules.astral_quantization.bsq import BinarySphericalQuantize
    enc = ConvNeXtV2Stage(dim=512, num_blocks=12, intermediate_dim=1536, dilation=1, input_dim=1024)
    q = BinarySphericalQuantize(codebook_size=32 if kind == "bsq32" else 2048, dim=512, entropy_loss_weight=0.1,
                                diversity_gamma=1.0, spherical=True, enable_entropy_loss=True, soft_entropy_loss=True)
    sd = torch.load(os.path.join(EXT, f"ASTRAL-quantization/{kind}/{kind}_light.pth"), map_location="cpu")
    print(kind, "enc:", enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}, strict=False))
    print(kind, "vq:", q.load_state_dict({k[len("quantizer."):]: v for k, v in sd.items() if k.startswith("quantizer.")}, strict=False))
    return enc.eval(), q.eval()


def hubert_features(hubert, wave_16k):
    """Как AstralQuantizer.forward до энкодера: нормировка волны, 18 слоёв, обрезка по N // 320."""
    w = wave_16k[0].numpy()
    w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
    x = torch.from_numpy(w.astype(np.float32))[None]
    h = hubert(x, attention_mask=torch.ones_like(x, dtype=torch.long)).last_hidden_state
    n = min(h.size(1), wave_16k.size(-1) // 320)
    return h[:, :n]


def astral_tokens(enc, q, feats):
    xh = enc(feats.transpose(1, 2), None).transpose(1, 2)
    return xh, q(xh)[1]


def build_bigvgan():
    # utils.py эталона тянет matplotlib ради графиков обучения — для инференса не нужен.
    import types
    for name in ("matplotlib", "matplotlib.pylab", "matplotlib.pyplot"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.use = lambda *a, **k: None
            sys.modules[name] = m
    sys.modules["matplotlib"].pylab = sys.modules["matplotlib.pylab"]
    from modules.bigvgan import bigvgan
    from modules.bigvgan.env import AttrDict
    import json
    d = os.path.join(EXT, "bigvgan_v2_22khz_80band_256x")
    # from_pretrained эталона несовместим с текущим huggingface_hub — собираем так же вручную.
    m = bigvgan.BigVGAN(AttrDict(json.load(open(os.path.join(d, "config.json")))), use_cuda_kernel=False)
    m.load_state_dict(torch.load(os.path.join(d, "bigvgan_generator.pt"), map_location="cpu")["generator"])
    m.remove_weight_norm()
    return m.eval()


def t_span(n):
    t = torch.linspace(0, 1, n + 1)
    return t + (-1) * (torch.cos(torch.pi / 2 * t) - 1 + t)
