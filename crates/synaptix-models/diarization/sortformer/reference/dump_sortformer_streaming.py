#!/usr/bin/env python3
# Reference-дамп NeMo Streaming Sortformer v2.1 в STREAMING-режиме (forward_streaming).
# Переиспользует ПОДЛИННЫЕ NeMo streaming-методы на sortformer_modules (init_streaming_state/
# streaming_feat_loader/streaming_update/_compress_spkcache/apply_mask_to_preds) + мои компоненты.
# Аудио = extr.wav ×3 (~43с → 3 чанка по 188, задействует compress спик-кэша).
#
# PYTHONPATH=~/Temp/NeMo:<stubs> .venv/bin/python dump_sortformer_streaming.py
import os, json, struct, math
import numpy as np, torch

OUT = os.environ.get("SF_OUT", "tmp/sortformer_ref"); os.makedirs(OUT, exist_ok=True)
WAV = os.environ.get("SF_WAV", "extr.wav")
ST = os.environ.get("SF_ST", "tmp/sortformer_unpack/model.safetensors")
TILE = int(os.environ.get("SF_TILE", "3"))

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
        arr = np.frombuffer(data[s:e], dtype=np.float32).reshape(v["shape"]).copy()
        out[k] = torch.from_numpy(arr)
    return out


sd = load_st(ST)
preprocessor = AudioToMelSpectrogramPreprocessor(normalize="NA", window_size=0.025, sample_rate=16000,
    window_stride=0.01, window="hann", features=128, n_fft=512, frame_splicing=1, dither=0.0, pad_to=0)
encoder = ConformerEncoder(feat_in=128, feat_out=-1, n_layers=17, d_model=512, subsampling="dw_striding",
    subsampling_factor=8, subsampling_conv_channels=256, causal_downsampling=False, ff_expansion_factor=4,
    self_attention_model="rel_pos", n_heads=8, att_context_size=[-1, -1], att_context_style="regular",
    xscaling=True, untie_biases=True, pos_emb_max_len=5000, conv_kernel_size=9, conv_norm_type="batch_norm",
    conv_context_size=None, dropout=0.1, dropout_pre_encoder=0.1, dropout_emb=0.0, dropout_att=0.1)
sm = SortformerModules(num_spks=4, dropout_rate=0.5, fc_d_model=512, tf_d_model=192)
transformer_encoder = TransformerEncoder(num_layers=18, hidden_size=192, inner_size=768,
    num_attention_heads=8, attn_score_dropout=0.5, attn_layer_dropout=0.5, ffn_dropout=0.5,
    hidden_act="relu", pre_ln=False, pre_ln_final_layer_norm=True)

HEAD_SUB = {"norm1.weight": "layer_norm_1.weight", "norm1.bias": "layer_norm_1.bias",
    "norm2.weight": "layer_norm_2.weight", "norm2.bias": "layer_norm_2.bias",
    "self_attn.linear_q.weight": "first_sub_layer.query_net.weight", "self_attn.linear_q.bias": "first_sub_layer.query_net.bias",
    "self_attn.linear_k.weight": "first_sub_layer.key_net.weight", "self_attn.linear_k.bias": "first_sub_layer.key_net.bias",
    "self_attn.linear_v.weight": "first_sub_layer.value_net.weight", "self_attn.linear_v.bias": "first_sub_layer.value_net.bias",
    "self_attn.linear_out.weight": "first_sub_layer.out_projection.weight", "self_attn.linear_out.bias": "first_sub_layer.out_projection.bias",
    "feed_forward.linear1.weight": "second_sub_layer.dense_in.weight", "feed_forward.linear1.bias": "second_sub_layer.dense_in.bias",
    "feed_forward.linear2.weight": "second_sub_layer.dense_out.weight", "feed_forward.linear2.bias": "second_sub_layer.dense_out.bias"}
SF_MAP = {"head.encoder_proj.weight": "encoder_proj.weight", "head.encoder_proj.bias": "encoder_proj.bias",
    "head.hidden_proj.weight": "first_hidden_to_hidden.weight", "head.hidden_proj.bias": "first_hidden_to_hidden.bias",
    "head.classifier.weight": "single_hidden_to_spks.weight", "head.classifier.bias": "single_hidden_to_spks.bias"}
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
encoder.load_state_dict(enc_sd, strict=False)
transformer_encoder.load_state_dict(tf_sd, strict=False)
sm.load_state_dict(sf_sd, strict=False)
preprocessor.load_state_dict(pre_sd, strict=False)
for m in (preprocessor, encoder, sm, transformer_encoder):
    m.eval()

import librosa
wav, _ = librosa.load(WAV, sr=16000, mono=True)
wav = np.tile(wav, TILE)
wav_t = torch.from_numpy(wav.astype(np.float32))
np.save(f"{OUT}/wav16_long.npy", wav_t.numpy())
print("wav", wav.shape, wav.shape[0] / 16000.0, "s")


def frontend_encoder(emb_in, length, bypass):
    e, el = encoder(audio_signal=emb_in, length=length, bypass_pre_encode=bypass)
    e = e.transpose(1, 2)
    e = sm.encoder_proj(e)
    return e, el


def forward_infer(emb, emb_len):
    mask = sm.length_to_mask(emb_len, emb.shape[1])
    trans = transformer_encoder(encoder_states=emb, encoder_mask=mask)
    p = sm.forward_speaker_sigmoids(trans)
    return p * mask.unsqueeze(-1)


with torch.inference_mode():
    audio = wav_t.unsqueeze(0)
    length = torch.tensor([wav_t.shape[0]], dtype=torch.long)
    audio = (1.0 / (audio.max() + 1e-3)) * audio
    processed_signal, processed_signal_length = preprocessor(input_signal=audio, length=length)
    processed_signal = processed_signal[:, :, : processed_signal_length.max()]
    sig_length = processed_signal.shape[2]

    ss = sm.init_streaming_state(batch_size=1, async_streaming=False, device=torch.device("cpu"))
    offset = torch.zeros((1,), dtype=torch.long)
    total_preds = torch.zeros((1, 0, sm.n_spk))

    n_chunks = 0
    for chunk_idx, chunk_feat_seq_t, feat_lengths, lc, rc in sm.streaming_feat_loader(
        feat_seq=processed_signal, feat_seq_length=processed_signal_length, feat_seq_offset=offset
    ):
        n_chunks += 1
        ce, cl = encoder.pre_encode(x=chunk_feat_seq_t, lengths=feat_lengths)
        cl = cl.to(torch.int64)
        cat = sm.concat_embs([ss.spkcache, ss.fifo, ce], dim=1, device=torch.device("cpu"))
        cat_len = ss.spkcache.shape[1] + ss.fifo.shape[1] + cl
        emb, emb_len = frontend_encoder(cat, cat_len, True)
        preds = forward_infer(emb, emb_len)
        preds = sm.apply_mask_to_preds(preds, emb_len)
        ss, chunk_preds = sm.streaming_update(ss, chunk=ce, preds=preds,
            lc=round(lc / 8), rc=math.ceil(rc / 8))
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)

    n_frames = math.ceil(sig_length / 8)
    total_preds = total_preds[:, :n_frames, :]
    a = total_preds.detach().float().cpu().numpy()
    np.save(f"{OUT}/stream_preds.npy", a)
    print(f"streaming: {n_chunks} chunks, preds {a.shape}")
    act = (a[0] > 0.5).sum(0)
    print("frames>0.5 per spk:", act.tolist(), "of", a.shape[1])
print("DUMP OK")
