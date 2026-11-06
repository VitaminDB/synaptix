"""Эталон Audio8-TTS (upstream: HF Edge0/Audio8-TTS-Realistic-Air, transformers 4.57.3,
trust_remote_code) для сверки с synaptix (пример audio8_audit).

    python -I audio8_ref.py --model <снимок HF> --text "..." --out каталог
        [--ref-audio a.wav --ref-text "..."] [--seed 0] [--greedy] [--max-new 200]
        [--temperature 0.7 --top-p 0.9 --top-k 50] [--pylib каталог с transformers]
        [--codes codes.npy]  # только декод кодов кодеком
        [--encode a.wav]     # только кодирование записи кодеком

Сэмплинг upstream (`ArkttsModel._sample`): argmax(softmax(s) / -ln(U)), U ~ torch.rand(V).
Здесь U берётся из счётчикового ГПСЧ synaptix (`synaptix_tts_audio8::sample::CounterRng`):
u_i = (splitmix64(seed + (n + i + 1)·GOLD) >> 41 + 0.5) / 2^23, n — сколько чисел уже выдано;
каждый вызов `_sample` берёт V чисел подряд (V — ширина словаря шага). Тогда при одном зерне
у нас и в эталоне один и тот же шум.

Выход: prompt.npy (int64 [11, T]), codes.npy (int64 [10, F]), audio.wav (44.1 кГц, f32),
first_logits.npy (логиты первого медленного шага), first_fast.npy (логиты codebook 1 первого кадра).
"""

import argparse
import os
import sys

import numpy as np

GOLD = 0x9E3779B97F4A7C15
M64 = (1 << 64) - 1


def splitmix_block(seed: int, start: int, n: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        idx = np.arange(start + 1, start + n + 1, dtype=np.uint64)
        z = np.uint64(seed) + idx * np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (((z >> np.uint64(41)).astype(np.float64) + 0.5) / float(1 << 23)).astype(np.float32)


class CounterRng:
    def __init__(self, seed: int):
        self.seed = seed & M64
        self.n = 0

    def uniform(self, n: int) -> np.ndarray:
        u = splitmix_block(self.seed, self.n, n)
        self.n += n
        return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text")
    ap.add_argument("--ref-audio")
    ap.add_argument("--ref-text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--pylib")
    ap.add_argument("--codes")
    ap.add_argument("--encode")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.pylib:
        sys.path.insert(0, args.pylib)

    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoProcessor

    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32).eval()
    cls = type(model)

    if args.codes:
        codes = torch.from_numpy(np.load(args.codes)).long()
        wav, lens = model.decode_audio(codes)
        sf.write(os.path.join(args.out, "audio.wav"), wav[0, : int(lens[0])].numpy(), 44100, subtype="FLOAT")
        print(f"decoded {codes.shape} -> {int(lens[0])} samples")
        return
    if args.encode:
        audio, sr = sf.read(args.encode, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.mean(axis=1))
        if sr != 44100:
            from torchaudio.functional import resample
            audio = resample(audio, sr, 44100)
        codes, lens = model.encode_audio(audio[None, None], torch.tensor([audio.numel()]))
        np.save(os.path.join(args.out, "ref_codes.npy"), codes[0, :, : int(lens[0])].numpy())
        print(f"encoded {audio.numel()} samples -> {int(lens[0])} frames")
        return

    rng = CounterRng(args.seed)

    def our_sample(scores, generator=None):
        probabilities = torch.softmax(scores, dim=-1)
        u = torch.from_numpy(rng.uniform(probabilities.shape[-1])).view(probabilities.shape)
        noise = -torch.log(u.to(probabilities.dtype))
        return torch.argmax(probabilities / noise, dim=-1)

    cls._sample = staticmethod(our_sample)

    dumps = {}
    orig_slow = cls._slow_step
    orig_fast = cls._fast_step

    def slow_step(self, *a, **k):
        logits, hidden = orig_slow(self, *a, **k)
        if "first_logits" not in dumps:
            dumps["first_logits"] = logits[0].float().numpy().copy()
            dumps["first_hidden"] = hidden[0, -1].float().numpy().copy()
        return logits, hidden

    def fast_step(self, hidden, position):
        out = orig_fast(self, hidden, position)
        if position == 1 and "first_fast" not in dumps:
            dumps["first_fast"] = out[0].float().numpy().copy()
        return out

    cls._slow_step = slow_step
    cls._fast_step = fast_step

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.ref_audio:
        inputs = processor(text=[args.text], reference_audio=[args.ref_audio], reference_text=[args.ref_text], return_tensors="pt")
    else:
        inputs = processor(text=[args.text], return_tensors="pt")
    prompt, _mask = model._prepare_prompt(
        None, None, inputs["prefix_input_ids"], inputs["prefix_attention_mask"],
        inputs["suffix_input_ids"], inputs["suffix_attention_mask"], None, None,
        inputs.get("reference_audio_values"), inputs.get("reference_audio_lengths"),
    )
    np.save(os.path.join(args.out, "prompt.npy"), prompt[0].numpy())
    with torch.inference_mode():
        out = model.generate(
            input_ids=prompt, attention_mask=_mask,
            max_new_tokens=args.max_new, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, do_sample=not args.greedy, return_dict_in_generate=True,
        )
        codes = out.codes
        wav, lens = model.decode_audio(codes)
    np.save(os.path.join(args.out, "codes.npy"), codes[0].numpy())
    for k, v in dumps.items():
        np.save(os.path.join(args.out, f"{k}.npy"), v)
    audio = wav[0, : int(lens[0])].numpy()
    sf.write(os.path.join(args.out, "audio.wav"), audio, 44100, subtype="FLOAT")
    print(f"prompt {tuple(prompt.shape)}, frames {codes.shape[-1]}, finished {bool(out.finished[0])}, "
          f"audio {audio.size / 44100:.2f} s, rng draws {rng.n}")


if __name__ == "__main__":
    main()
