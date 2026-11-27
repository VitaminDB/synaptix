"""Дампы Seed-VC v1 (whisper-small wavenet) для сверки порта: CPU, f32, заданный шум.

SEEDVC_REF=…/seed-vc python -I seedvc_v1.py OUT_DIR [steps]
Нужны дампы src_22k / ref_22k от seedvc_parts.py (этап audio) в том же каталоге.
Пишет: v1_src_16k, v1_ref_16k (torchaudio.functional.resample, как inference.py),
v1_whisper_src / v1_whisper_ref (последний слой энкодера, обрезка N // 320 + 1),
v1_style, v1_lr_src / v1_lr_ref, v1_dit_v (один вызов на заданных входах),
v1_z{i} / v1_wave (полный прогон: логика main() из inference.py, CFG 0.7).
Whisper в эталоне идёт в fp16 — здесь f32 (сверка по CPU).
"""
import glob
import os
import sys
import numpy as np
import torch
import yaml
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seedvc_common as C

C.setup()
out = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
D = lambda n, t: C.dump(out, n, t)


def load_f32(name):
    f = glob.glob(os.path.join(out, name + ".*.f32"))[0]
    dims = [int(x) for x in os.path.basename(f).split(".")[-2].split("x")]
    return torch.from_numpy(np.fromfile(f, dtype="<f4").reshape(dims))


import torchaudio
from modules.commons import build_model, load_checkpoint, recursive_munch

config = yaml.safe_load(open(os.path.join(C.MAIN, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")))
params = recursive_munch(config["model_params"])
params.dit_type = "DiT"
model = build_model(params, stage="DiT")
model, _, _, _ = load_checkpoint(model, None, os.path.join(C.MAIN, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
                                 load_only_params=True, ignore_modules=[], is_distributed=False)
for k in model:
    model[k].eval()
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

from transformers import AutoFeatureExtractor, WhisperModel
wdir = os.path.join(C.EXT, "whisper-small")
whisper = WhisperModel.from_pretrained(wdir, torch_dtype=torch.float32).eval()
del whisper.decoder
fe = AutoFeatureExtractor.from_pretrained(wdir)


def semantic_fn(waves_16k):
    inp = fe([waves_16k.squeeze(0).numpy()], return_tensors="pt", return_attention_mask=True)
    feats = whisper._mask_input_features(inp.input_features, attention_mask=inp.attention_mask)
    s = whisper.encoder(feats, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True).last_hidden_state
    return s[:, : waves_16k.size(-1) // 320 + 1]


sr, hop = C.SR, C.HOP
source_audio = load_f32("src_22k")[None]
ref_audio = load_f32("ref_22k")[None][:, : sr * 25]
src16 = torchaudio.functional.resample(source_audio, sr, 16000)
ref16 = torchaudio.functional.resample(ref_audio, sr, 16000)
D("v1_src_16k", src16[0])
D("v1_ref_16k", ref16[0])

# Содержание: до 30 с одним проходом, дольше — окнами (как main()).
if src16.size(-1) <= 16000 * 30:
    S_alt = semantic_fn(src16)
else:
    ov, lst, buffer, trav = 5, [], None, 0
    while trav < src16.size(-1):
        chunk = src16[:, trav:trav + 16000 * 30] if buffer is None else torch.cat([buffer, src16[:, trav:trav + 16000 * (30 - ov)]], dim=-1)
        s = semantic_fn(chunk)
        lst.append(s if trav == 0 else s[:, 50 * ov:])
        buffer = chunk[:, -16000 * ov:]
        trav += 30 * 16000 if trav == 0 else chunk.size(-1) - 16000 * ov
    S_alt = torch.cat(lst, dim=1)
S_ori = semantic_fn(ref16)
D("v1_whisper_src", S_alt[0])
D("v1_whisper_ref", S_ori[0])

mel = C.mel_fn(source_audio)
mel2 = C.mel_fn(ref_audio)
style2 = C.build_campplus()(C.fbank(ref16).unsqueeze(0))
D("v1_style", style2[0])

cond = model.length_regulator(S_alt, ylens=torch.LongTensor([mel.size(2)]), n_quantizers=3, f0=None)[0]
prompt_condition = model.length_regulator(S_ori, ylens=torch.LongTensor([mel2.size(2)]), n_quantizers=3, f0=None)[0]
D("v1_lr_src", cond[0])
D("v1_lr_ref", prompt_condition[0])

# Один вызов DiT на заданных входах.
T = 120
x = C.wave([1, 80, T], 0.8, 0.11, 0.0)
p = C.wave([1, 80, T], 0.5, 0.07, 1.0)
p[..., 50:] = 0
c = C.wave([1, T, 512], 0.3, 0.013, 0.5)
st = C.wave([1, 192], 0.5, 0.31, 0.2)
D("v1_dit_v", model.cfm.estimator(x, p, torch.LongTensor([T]), torch.tensor([0.4]), st, c)[0])

# Полный прогон: цикл окон из inference.py, шум — Generator(4321 + номер окна).
voc = C.build_bigvgan()
max_context_window = sr // hop * 30
overlap_frame_len = 16
overlap_wave_len = overlap_frame_len * hop


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


max_source_window = max_context_window - mel2.size(2)
processed_frames, chunks, ci = 0, [], 0
while processed_frames < cond.size(1):
    chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
    is_last_chunk = processed_frames + max_source_window >= cond.size(1)
    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
    z = torch.randn([1, 80, cat_condition.size(1)], generator=torch.Generator().manual_seed(4321 + ci))
    D(f"v1_z{ci}", z[0])
    ci += 1
    vc_target = model.cfm.solve_euler(z.clone(), torch.LongTensor([cat_condition.size(1)]), mel2, cat_condition, style2, None,
                                      torch.linspace(0, 1, steps + 1), 0.7)
    vc_target = vc_target[:, :, mel2.size(-1):]
    vc_wave = voc(vc_target.float()).squeeze()[None, :]
    if processed_frames == 0:
        if is_last_chunk:
            chunks.append(vc_wave[0].numpy())
            break
        chunks.append(vc_wave[0, :-overlap_wave_len].numpy())
        previous_chunk = vc_wave[0, -overlap_wave_len:]
        processed_frames += vc_target.size(2) - overlap_frame_len
    elif is_last_chunk:
        chunks.append(crossfade(previous_chunk.numpy(), vc_wave[0].numpy(), overlap_wave_len))
        break
    else:
        chunks.append(crossfade(previous_chunk.numpy(), vc_wave[0, :-overlap_wave_len].numpy(), overlap_wave_len))
        previous_chunk = vc_wave[0, -overlap_wave_len:]
        processed_frames += vc_target.size(2) - overlap_frame_len
wave = np.concatenate(chunks)
D("v1_wave", wave)
sf.write(os.path.join(out, "ref_v1.wav"), wave, sr)
