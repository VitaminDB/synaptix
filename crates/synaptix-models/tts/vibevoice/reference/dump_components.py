import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("VIBEVOICE_SRC", "/home/master/Storage/VibeVoice"))

from safetensors.torch import save_file

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceTokenizerStreamingCache,
)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def det_audio(n, seed=1234):
    g = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64) / 24000.0
    wav = (
        0.35 * np.sin(2 * np.pi * 180.0 * t)
        + 0.20 * np.sin(2 * np.pi * 432.0 * t + 0.7)
        + 0.10 * np.sin(2 * np.pi * 1310.0 * t + 1.9)
        + 0.02 * g.randn(n)
    )
    env = 0.5 * (1.0 - np.cos(2 * np.pi * np.minimum(np.arange(n) / 2400.0, 1.0)))
    return (wav * env).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.to(device).eval()
    cfg: VibeVoiceConfig = model.config

    out = {}
    meta = {}

    acoustic = model.model.acoustic_tokenizer
    semantic = model.model.semantic_tokenizer
    head = model.model.prediction_head
    lm = model.model.language_model

    hidden = cfg.decoder_config.hidden_size
    vae_dim = cfg.acoustic_tokenizer_config.vae_dim
    sem_dim = cfg.semantic_tokenizer_config.vae_dim

    n_samples = 3200 * 8
    wav = det_audio(n_samples)
    wav_t = torch.from_numpy(wav).to(device).unsqueeze(0)
    out["audio_in"] = wav_t.cpu()

    with torch.no_grad():
        enc = acoustic.encode(wav_t.unsqueeze(1))
    out["acoustic_encode_mean"] = enc.mean.float().cpu()
    meta["acoustic_fix_std"] = float(acoustic.fix_std.item())

    with torch.no_grad():
        sem_full = semantic.encode(wav_t.unsqueeze(1))
    out["semantic_encode_mean"] = sem_full.mean.float().cpu()

    torch.manual_seed(7)
    lat = torch.randn(1, args.frames, vae_dim, device=device) * 0.6
    out["decode_latents"] = lat.float().cpu()

    with torch.no_grad():
        full = acoustic.decode(lat.clone())
    out["acoustic_decode_full"] = full.float().cpu()

    acache = VibeVoiceTokenizerStreamingCache()
    scache = VibeVoiceTokenizerStreamingCache()
    idx = torch.tensor([0], device=device)
    chunks = []
    sem_chunks = []
    with torch.no_grad():
        for i in range(args.frames):
            step = lat[:, i : i + 1, :]
            ch = acoustic.decode(step, cache=acache, sample_indices=idx, use_cache=True)
            chunks.append(ch)
            sm = semantic.encode(ch, cache=scache, sample_indices=idx, use_cache=True).mean
            sem_chunks.append(sm)
    out["acoustic_decode_stream"] = torch.cat(chunks, dim=-1).float().cpu()
    out["semantic_encode_stream"] = torch.cat(sem_chunks, dim=1).float().cpu()

    torch.manual_seed(11)
    noisy = torch.randn(3, vae_dim, device=device)
    cond = torch.randn(3, hidden, device=device) * 0.5
    ts = torch.tensor([0.0, 137.0, 999.0], device=device)
    with torch.no_grad():
        hout = head(noisy, ts, cond)
    out["head_noisy"] = noisy.float().cpu()
    out["head_cond"] = cond.float().cpu()
    out["head_t"] = ts.float().cpu()
    out["head_out"] = hout.float().cpu()

    sched = model.model.noise_scheduler
    sched.set_timesteps(cfg.diffusion_head_config.ddpm_num_inference_steps)
    out["sched_timesteps"] = sched.timesteps.float().cpu()
    out["sched_sigmas"] = sched.sigmas.float().cpu()

    torch.manual_seed(13)
    pos = torch.randn(2, hidden, device=device) * 0.4
    neg = torch.randn(2, hidden, device=device) * 0.4
    init = torch.randn(4, vae_dim, device=device)
    out["cfg_pos"] = pos.float().cpu()
    out["cfg_neg"] = neg.float().cpu()
    out["cfg_init_noise"] = init.float().cpu()

    with torch.no_grad():
        sched.set_timesteps(model.ddpm_inference_steps)
        condition = torch.cat([pos, neg], dim=0)
        speech = init.clone()
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + 1.3 * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = sched.step(eps, t, speech).prev_sample
    out["cfg_sampled"] = speech[: len(speech) // 2].float().cpu()
    meta["cfg_scale"] = 1.3

    torch.manual_seed(17)
    afeat = torch.randn(1, 5, vae_dim, device=device) * 0.3
    sfeat = torch.randn(1, 5, sem_dim, device=device) * 0.3
    with torch.no_grad():
        aconn = model.model.acoustic_connector(afeat)
        sconn = model.model.semantic_connector(sfeat)
    out["conn_acoustic_in"] = afeat.float().cpu()
    out["conn_acoustic_out"] = aconn.float().cpu()
    out["conn_semantic_in"] = sfeat.float().cpu()
    out["conn_semantic_out"] = sconn.float().cpu()

    torch.manual_seed(19)
    ids = torch.randint(0, 150000, (1, 24), device=device)
    embeds = model.model.get_input_embeddings()(ids)
    with torch.no_grad():
        lmout = model.model(inputs_embeds=embeds, use_cache=True, return_dict=True)
        logits = model.lm_head(lmout.last_hidden_state)
    out["lm_ids"] = ids.cpu()
    out["lm_hidden"] = lmout.last_hidden_state.float().cpu()
    out["lm_logits"] = logits.float().cpu()

    nxt = torch.randint(0, 150000, (1, 1), device=device)
    with torch.no_grad():
        step_embed = model.model.get_input_embeddings()(nxt)
        lmstep = model.model(
            inputs_embeds=step_embed,
            past_key_values=lmout.past_key_values,
            use_cache=True,
            return_dict=True,
        )
    out["lm_next_ids"] = nxt.cpu()
    out["lm_next_hidden"] = lmstep.last_hidden_state.float().cpu()

    processor = VibeVoiceProcessor.from_pretrained(args.model)
    meta["speech_start_id"] = int(processor.tokenizer.speech_start_id)
    meta["speech_end_id"] = int(processor.tokenizer.speech_end_id)
    meta["speech_diffusion_id"] = int(processor.tokenizer.speech_diffusion_id)
    meta["eos_id"] = int(processor.tokenizer.eos_id)
    meta["pad_id"] = int(processor.tokenizer.pad_id)
    meta["speech_scaling_factor"] = float(model.model.speech_scaling_factor.item())
    meta["speech_bias_factor"] = float(model.model.speech_bias_factor.item())

    script = "Speaker 1: Hello there, this is a parity probe.\nSpeaker 2: And this is the second speaker line."
    voice_a = wav
    voice_b = det_audio(3200 * 5, seed=99)
    out["prompt_voice_raw_0"] = torch.from_numpy(voice_a)
    out["prompt_voice_raw_1"] = torch.from_numpy(voice_b)
    enc_p = processor(text=[script], voice_samples=[[voice_a, voice_b]], return_tensors="pt")
    out["prompt_input_ids"] = enc_p["input_ids"].cpu()
    out["prompt_speech_input_mask"] = enc_p["speech_input_mask"].cpu().to(torch.uint8)
    out["prompt_speech_tensors"] = enc_p["speech_tensors"].float().cpu()
    out["prompt_speech_masks"] = enc_p["speech_masks"].cpu().to(torch.uint8)
    meta["prompt_script"] = script

    with open(os.path.join(os.path.dirname(args.out) or ".", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    save_file({k: v.contiguous() for k, v in out.items()}, args.out)
    print("wrote", args.out)
    for k, v in out.items():
        print("  ", k, tuple(v.shape), v.dtype)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
