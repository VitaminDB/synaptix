#!/usr/bin/env python3
# Reference-дамп NeMo Streaming Sortformer 4spk v2.1 (BATCH / full-attention путь).
# Источник истины = официальный NVIDIA NeMo (~/Temp/NeMo). Веса = мои .syn (HF-имена),
# мапятся на NeMo-имена компонентов. Дампим mel/encoder/emb_seq/trans/preds на extr.wav.
#
# PYTHONPATH=~/Temp/NeMo:<stubs> .venv/bin/python dump_sortformer.py
import os, json, struct
import numpy as np, torch

OUT = os.environ.get("SF_OUT", "tmp/sortformer_ref"); os.makedirs(OUT, exist_ok=True)
WAV = os.environ.get("SF_WAV", "extr.wav")
ST  = os.environ.get("SF_ST",  "tmp/sortformer_unpack/model.safetensors")

from nemo.collections.asr.modules import ConformerEncoder, AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.sortformer_modules import SortformerModules

torch.manual_seed(0)


def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); data = f.read()
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        dt = v["dtype"]
        npdt = {"F32": np.float32, "F16": np.float16, "BF16": np.float32}.get(dt, np.float32)
        if dt == "BF16":
            raw = np.frombuffer(data[s:e], dtype=np.uint16).astype(np.uint32) << 16
            arr = raw.view(np.float32).reshape(v["shape"]).copy()
        else:
            arr = np.frombuffer(data[s:e], dtype=npdt).reshape(v["shape"]).copy()
        out[k] = torch.from_numpy(arr.astype(np.float32))
    return out


sd = load_st(ST)
print("checkpoint keys:", len(sd))

# ---- build components (v2.1 = v2 architecture; гиперпараметры из streaming_sortformer_diarizer_4spk-v2.yaml)
preprocessor = AudioToMelSpectrogramPreprocessor(
    normalize="NA", window_size=0.025, sample_rate=16000, window_stride=0.01,
    window="hann", features=128, n_fft=512, frame_splicing=1, dither=0.0, pad_to=0,
)
encoder = ConformerEncoder(
    feat_in=128, feat_out=-1, n_layers=17, d_model=512,
    subsampling="dw_striding", subsampling_factor=8, subsampling_conv_channels=256,
    causal_downsampling=False, ff_expansion_factor=4, self_attention_model="rel_pos",
    n_heads=8, att_context_size=[-1, -1], att_context_style="regular", xscaling=True,
    untie_biases=True, pos_emb_max_len=5000, conv_kernel_size=9, conv_norm_type="batch_norm",
    conv_context_size=None, dropout=0.1, dropout_pre_encoder=0.1, dropout_emb=0.0, dropout_att=0.1,
)
sortformer_modules = SortformerModules(num_spks=4, dropout_rate=0.5, fc_d_model=512, tf_d_model=192)
transformer_encoder = TransformerEncoder(
    num_layers=18, hidden_size=192, inner_size=768, num_attention_heads=8,
    attn_score_dropout=0.5, attn_layer_dropout=0.5, ffn_dropout=0.5, hidden_act="relu",
    pre_ln=False, pre_ln_final_layer_norm=True,
)

# ---- remap HF-имена → NeMo-имена компонентов
HEAD_SUB = {
    "norm1.weight": "layer_norm_1.weight", "norm1.bias": "layer_norm_1.bias",
    "norm2.weight": "layer_norm_2.weight", "norm2.bias": "layer_norm_2.bias",
    "self_attn.linear_q.weight": "first_sub_layer.query_net.weight",
    "self_attn.linear_q.bias": "first_sub_layer.query_net.bias",
    "self_attn.linear_k.weight": "first_sub_layer.key_net.weight",
    "self_attn.linear_k.bias": "first_sub_layer.key_net.bias",
    "self_attn.linear_v.weight": "first_sub_layer.value_net.weight",
    "self_attn.linear_v.bias": "first_sub_layer.value_net.bias",
    "self_attn.linear_out.weight": "first_sub_layer.out_projection.weight",
    "self_attn.linear_out.bias": "first_sub_layer.out_projection.bias",
    "feed_forward.linear1.weight": "second_sub_layer.dense_in.weight",
    "feed_forward.linear1.bias": "second_sub_layer.dense_in.bias",
    "feed_forward.linear2.weight": "second_sub_layer.dense_out.weight",
    "feed_forward.linear2.bias": "second_sub_layer.dense_out.bias",
}
SF_MAP = {
    "head.encoder_proj.weight": "encoder_proj.weight",
    "head.encoder_proj.bias": "encoder_proj.bias",
    "head.hidden_proj.weight": "first_hidden_to_hidden.weight",
    "head.hidden_proj.bias": "first_hidden_to_hidden.bias",
    "head.classifier.weight": "single_hidden_to_spks.weight",
    "head.classifier.bias": "single_hidden_to_spks.bias",
}
enc_sd, tf_sd, sf_sd, pre_sd = {}, {}, {}, {}
for k, v in sd.items():
    if k.startswith("encoder."):
        enc_sd[k[len("encoder."):]] = v
    elif k.startswith("preprocessor."):
        pre_sd[k[len("preprocessor."):]] = v
    elif k in SF_MAP:
        sf_sd[SF_MAP[k]] = v
    elif k.startswith("head.layers."):
        idx, sub = k[len("head.layers."):].split(".", 1)
        tf_sd[f"layers.{idx}.{HEAD_SUB[sub]}"] = v
    else:
        print("  UNMAPPED:", k)

r = encoder.load_state_dict(enc_sd, strict=False)
print("encoder: missing", len(r.missing_keys), "unexpected", len(r.unexpected_keys), "| unexp:", r.unexpected_keys[:5])
r = transformer_encoder.load_state_dict(tf_sd, strict=False)
print("transformer: missing", r.missing_keys[:8], "unexpected", r.unexpected_keys[:8])
r = sortformer_modules.load_state_dict(sf_sd, strict=False)
print("sortformer: missing", [m for m in r.missing_keys if "hidden_to_spks" not in m or m.startswith("single")][:8],
      "unexpected", r.unexpected_keys[:8])
r = preprocessor.load_state_dict(pre_sd, strict=False)
print("preproc: loaded fb/window keys:", list(pre_sd.keys()))

for m in (preprocessor, encoder, sortformer_modules, transformer_encoder):
    m.eval()

# ---- audio
import librosa
wav, sr = librosa.load(WAV, sr=16000, mono=True)
wav = torch.from_numpy(wav.astype(np.float32))
np.save(f"{OUT}/wav16.npy", wav.numpy())
print("wav16:", wav.shape, "dur", wav.shape[0] / 16000.0, "s")


def dump(name, t):
    a = (t[0] if isinstance(t, (tuple, list)) else t).detach().float().cpu().numpy()
    np.save(f"{OUT}/{name}.npy", a); print(f"  {name}", a.shape)


# hooks для постадийной локализации энкодера
caps = {}
def hk(name):
    def f(m, i, o):
        caps[name] = o[0] if isinstance(o, (tuple, list)) else o
    return f
encoder.pre_encode.register_forward_hook(hk("preenc"))   # (B,T',512) pre-xscale
encoder.layers[0].register_forward_hook(hk("enc_l0"))
encoder.layers[8].register_forward_hook(hk("enc_l8"))
encoder.layers[16].register_forward_hook(hk("enc_l16"))

with torch.inference_mode():
    audio = wav.unsqueeze(0)
    length = torch.tensor([wav.shape[0]], dtype=torch.long)
    # process_signal (non-streaming): normalize by max
    eps = 1e-3
    audio = (1.0 / (audio.max() + eps)) * audio
    processed_signal, processed_signal_length = preprocessor(input_signal=audio, length=length)
    processed_signal = processed_signal[:, :, : processed_signal_length.max()]
    dump("mel", processed_signal)
    print("  mel_len", processed_signal_length.tolist())

    # frontend_encoder
    emb_seq, emb_seq_length = encoder(audio_signal=processed_signal, length=processed_signal_length)
    for nm in ("preenc", "enc_l0", "enc_l8", "enc_l16"):
        if nm in caps:
            dump(nm, caps[nm])  # (B, T', 512)
    dump("encoder_out", emb_seq)  # (B, 512, T')
    emb_seq = emb_seq.transpose(1, 2)  # (B, T', 512)
    emb_seq = sortformer_modules.encoder_proj(emb_seq)  # (B, T', 192)
    dump("emb_seq", emb_seq)
    print("  emb_seq_len", emb_seq_length.tolist())

    # forward_infer
    encoder_mask = sortformer_modules.length_to_mask(emb_seq_length, emb_seq.shape[1])
    trans_emb_seq = transformer_encoder(encoder_states=emb_seq, encoder_mask=encoder_mask)
    dump("trans", trans_emb_seq)
    _preds = sortformer_modules.forward_speaker_sigmoids(trans_emb_seq)
    preds = _preds * encoder_mask.unsqueeze(-1)
    dump("preds", preds)

print("DUMP OK")
