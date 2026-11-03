"""Эталон VibeVoice (upstream: github vibevoice-community/VibeVoice, transformers==4.51.3)
для сверки с synaptix (пример vv_audit).

    python -I vibevoice_ref.py --repo <клон VibeVoice> --model <каталог HF> --tok <каталог с tokenizer.json>
        --jobs задания.tsv --out выход [--steps 10] [--cfg 1.3] [--seed 0] [--rng ours|torch]
        [--sde] [--dtype f32|bf16] [--dump]

Задания — как у vv_audit: `имя<TAB>образцы через запятую или -<TAB>сценарий` (`|` — перевод строки).
`--rng ours` подменяет torch.randn/randn_like нашим ГПСЧ (NormalRng из generate.rs: splitmix64 +
Бокс — Мюллер) в том же порядке выборок: шум латентов образца, затем по 64 значения на каждый
шаг диффузии — тогда при одинаковом зерне у нас и в эталоне один и тот же шум.
`--sde` — решатель как в gradio-демо (sde-dpmsolver++, стохастический), иначе как в
inference_from_file (dpmsolver++ из конфига).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15


class NormalRng:
    """Копия synaptix_tts_vibevoice::generate::NormalRng."""

    def __init__(self, seed):
        self.state = (seed + GOLD) & M64
        self.spare = None

    def next_u64(self):
        self.state = (self.state + GOLD) & M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def normal(self):
        if self.spare is not None:
            v, self.spare = self.spare, None
            return v
        u1 = ((self.next_u64() >> 11) + 0.5) / float(1 << 53)
        u2 = ((self.next_u64() >> 11) + 0.5) / float(1 << 53)
        r = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        self.spare = float(np.float32(r * math.sin(th)))
        return float(np.float32(r * math.cos(th)))

    def normals(self, n):
        return [self.normal() for _ in range(n)]


RNG = None
_randn = torch.randn
_randn_like = torch.randn_like


def _shape(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        return tuple(size[0])
    return tuple(int(s) for s in size)


def randn_ours(*size, **kw):
    shape = _shape(size)
    dtype = kw.get("dtype") or torch.float32
    device = kw.get("device")
    if len(shape) == 2 and shape[0] == 2:
        # sample_speech_tokens: шум [2, D], значима только первая половина.
        row = torch.tensor(RNG.normals(shape[1]), dtype=torch.float32)
        return torch.stack([row, row]).to(dtype=dtype, device=device)
    n = int(np.prod(shape)) if shape else 1
    return torch.tensor(RNG.normals(n), dtype=torch.float32).reshape(shape).to(dtype=dtype, device=device)


def randn_like_ours(t, **kw):
    return randn_ours(*t.shape, dtype=kw.get("dtype") or t.dtype, device=t.device)


def read_audio(path, sr=24000):
    # Как VibeVoiceDemo.read_audio в demo/gradio_demo.py.
    import librosa
    import soundfile as sf

    wav, r = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if r != sr:
        wav = librosa.resample(wav, orig_sr=r, target_sr=sr)
    return wav.astype(np.float32)


def save(path, arr):
    arr.detach().float().cpu().numpy().astype("<f4").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rng", default="ours")
    ap.add_argument("--sde", action="store_true")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    torch.set_num_threads(os.cpu_count())
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
    from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
    import soundfile as sf

    global RNG
    os.makedirs(a.out, exist_ok=True)
    dt = torch.float32 if a.dtype == "f32" else torch.bfloat16
    tok = VibeVoiceTextTokenizerFast.from_pretrained(a.tok)
    proc = VibeVoiceProcessor(tokenizer=tok, audio_processor=VibeVoiceTokenizerProcessor())
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        a.model, torch_dtype=dt, device_map="cpu", attn_implementation="sdpa"
    )
    model.eval()
    if a.sde:
        model.model.noise_scheduler = model.model.noise_scheduler.from_config(
            model.model.noise_scheduler.config, algorithm_type="sde-dpmsolver++", beta_schedule="squaredcos_cap_v2"
        )
    model.set_ddpm_inference_steps(num_steps=a.steps)
    if a.rng == "ours":
        torch.randn = randn_ours
        torch.randn_like = randn_like_ours

    jobs = [l for l in open(a.jobs, encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("#")]
    for k, line in enumerate(jobs):
        name, vs, script = line.split("\t", 2)
        if a.only and name not in a.only.split(","):
            continue
        script = script.replace("|", "\n")
        voices = [] if vs == "-" else [read_audio(v) for v in vs.split(",")]
        inputs = proc(
            text=[script],
            voice_samples=[voices] if voices else None,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if a.rng == "ours":
            RNG = NormalRng(a.seed)
        else:
            torch.manual_seed(a.seed)
        if k == 0 and a.dump:
            ids = inputs["input_ids"][0].tolist()
            open(f"{a.out}/dump_ids.txt", "w").write(" ".join(map(str, ids)))
            m = model.model
            with torch.no_grad():
                emb = m.get_input_embeddings()(inputs["input_ids"])
                if inputs["speech_tensors"] is not None:
                    st = inputs["speech_tensors"].to(dt)
                    save(f"{a.out}/dump_voice_mean.bin", m.acoustic_tokenizer.encode(st.unsqueeze(1)).mean)
                    _, sp = model._process_speech_inputs(st, inputs["speech_masks"])
                    emb[inputs["speech_input_mask"]] = sp
                save(f"{a.out}/dump_prompt_embeds.bin", emb)
                out = model(inputs_embeds=emb, attention_mask=inputs["attention_mask"], use_cache=True, return_dict=True)
                pos = out.last_hidden_state[:, -1, :]
                save(f"{a.out}/dump_prefill_last.bin", pos)
                ni = torch.full((1, 1), tok.speech_start_id, dtype=torch.long)
                neg = model(input_ids=ni, use_cache=True, return_dict=True).last_hidden_state[:, -1, :]
                save(f"{a.out}/dump_neg_first.bin", neg)
                lat = model.sample_speech_tokens(pos, neg, cfg_scale=a.cfg)
                save(f"{a.out}/dump_first_latent.bin", lat)
                from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

                lat3 = lat.unsqueeze(1)
                scaled = lat3 / m.speech_scaling_factor - m.speech_bias_factor
                ac = VibeVoiceTokenizerStreamingCache()
                idx = torch.tensor([0])
                chunk = m.acoustic_tokenizer.decode(scaled, cache=ac, sample_indices=idx, use_cache=True)
                save(f"{a.out}/dump_first_chunk.bin", chunk)
                sc = VibeVoiceTokenizerStreamingCache()
                sem = m.semantic_tokenizer.encode(chunk, cache=sc, sample_indices=idx, use_cache=True).mean
                save(f"{a.out}/dump_first_semantic.bin", sem)
            if a.rng == "ours":
                RNG = NormalRng(a.seed)
            else:
                torch.manual_seed(a.seed)
        import time

        t = time.time()
        with torch.no_grad():
            o = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=a.cfg,
                tokenizer=tok,
                generation_config={"do_sample": False},
                verbose=False,
                refresh_negative=True,
                is_prefill=bool(voices),
                show_progress_bar=False,
            )
        audio = o.speech_outputs[0]
        audio = audio.float().cpu().numpy().reshape(-1) if audio is not None else np.zeros(1, np.float32)
        sf.write(f"{a.out}/{name}.wav", audio, 24000, subtype="PCM_16")
        n0 = inputs["input_ids"].shape[1]
        toks = o.sequences[0, n0:].tolist()
        open(f"{a.out}/{name}.tokens", "w").write(" ".join(map(str, toks)))
        print(f"{name}: {len(audio) / 24000:.2f} с за {time.time() - t:.1f} с, токенов {len(toks)}, предел {bool(o.reach_max_step_sample[0])}", flush=True)


if __name__ == "__main__":
    main()
