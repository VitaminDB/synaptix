//! Изоляция DiT vs conditioning. VAE уже доказан корректным (round-trip).
//! Декодируем lm_hints (src_latents) НАПРЯМУЮ через VAE → lm_hints_direct.wav,
//! и денойз-латент DiT → denoised.wav. Печатаем mean/std латентов.
//! Если lm_hints_direct структурно, а denoised мусор → баг в DiT/denoise.

use std::path::Path;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::ar::{ar_generate, CodesGenOptions};
use synaptix_music_acestep::cond_encoder::ConditionEncoder;
use synaptix_music_acestep::config::DitConfig;
use synaptix_music_acestep::detokenizer::Detokenizer;
use synaptix_music_acestep::dit::Dit;
use synaptix_music_acestep::fsq::Fsq;
use synaptix_music_acestep::lm::AceStepLm;
use synaptix_music_acestep::loader::{read_bundle_file, CompLoader};
use synaptix_music_acestep::pipeline::{denoise, peak_normalize, SamplerOptions};
use synaptix_music_acestep::text_encoder::TextEncoder;
use synaptix_music_acestep::tokenizer::{AceTokenizer, Metadata};
use synaptix_music_acestep::vae::AceStepVae;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

fn stats(t: &Tensor, label: &str) {
    let v: Vec<f32> = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let nan = v.iter().filter(|x| !x.is_finite()).count();
    eprintln!("[iso] {label}: dims={:?} mean={mean:.4} std={:.4} min={:.3} max={:.3} nan={nan}",
        t.dims(), var.sqrt(), v.iter().cloned().fold(f32::INFINITY, f32::min), v.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
}

fn save_mono(vae: &AceStepVae, lat_ncl: &Tensor, path: &str) {
    let audio = vae.decode(lat_ncl).expect("decode");
    let mut mono: Vec<f32> = audio.narrow(1, 0, 1).unwrap().contiguous().unwrap()
        .to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    peak_normalize(&mut mono);
    synaptix_audio::write_wav_mono_f32(path, &mono, 48000).unwrap();
    let rms = (mono.iter().map(|x| x*x).sum::<f32>()/mono.len() as f32).sqrt();
    eprintln!("[iso] wrote {path} n={} rms={rms:.4}", mono.len());
}

fn dump_st(tensors: &[(&str, &Tensor)], path: &str) {
    use safetensors::tensor::{Dtype, TensorView};
    let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, t) in tensors {
        let v: Vec<f32> = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        owned.push((name.to_string(), t.dims().to_vec(), bytes));
    }
    let views: Vec<(&str, TensorView)> = owned
        .iter()
        .map(|(n, sh, b)| (n.as_str(), TensorView::new(Dtype::F32, sh.clone(), b).unwrap()))
        .collect();
    safetensors::serialize_to_file(views, None, std::path::Path::new(path)).unwrap();
    eprintln!("[iso] dumped {path}");
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let dir = Path::new("storage/syn_models");
    let lm_path = dir.join("acestep_5hz_lm_1.7b.syn");
    let te_path = dir.join("qwen3-embedding-0.6b.syn");
    let dit_path = dir.join("acestep_v15_xl_base.syn");
    let vae_path = dir.join("acestep_vae.syn");
    let cfg = DitConfig::xl_base();
    let caption = "calm ambient piano melody";

    // 1. LM → codes
    let codes = {
        let lm = AceStepLm::open(&lm_path, device, DType::F32, DType::F32, 1024).expect("lm");
        let tok = AceTokenizer::from_bytes(&read_bundle_file(&lm_path, "tokenizer.json").unwrap()).unwrap();
        let base = Metadata { caption: caption.into(), duration: 8, ..Metadata::default() };
        let (codes, _m) = ar_generate(&lm, &tok, caption, "", &base, &CodesGenOptions::default(), false).expect("ar");
        let uniq: std::collections::HashSet<u32> = codes.iter().copied().collect();
        eprintln!("[iso] codes len={} unique={}", codes.len(), uniq.len());
        codes
    };

    // 2. text-enc → enc + null
    let (text_hidden, lyric_hidden, cap_ids_t, lyr_ids_t) = {
        let te = TextEncoder::open(&te_path, device, DType::F32, DType::F32, 4096).expect("te");
        let tok = HfTokenizer::from_bytes(&read_bundle_file(&te_path, "tokenizer.json").unwrap()).unwrap();
        let ids = |s: &str| { let mut i = tok.encode(s, false).unwrap().ids; if i.is_empty() { i.push(151643); } let n=i.len(); Tensor::from_vec(i, vec![1,n], device).unwrap() };
        let ids_f = |t: &Tensor| { let f: Vec<f32> = t.flatten_all().unwrap().to_vec1::<u32>().unwrap().iter().map(|&v| v as f32).collect(); Tensor::from_vec(f, vec![1, t.dims()[1]], device).unwrap() };
        use synaptix_music_acestep::text_encoder::{build_text_prompt, build_lyric_prompt};
        let tp = build_text_prompt(caption, 8, Some(120), Some("4/4"), Some("C minor"));
        let lp = build_lyric_prompt("", "en");
        let cap_ids = ids(&tp);
        let lyr_ids = ids(&lp);
        (te.caption_hidden(&cap_ids).unwrap().to_dtype(DType::F32).unwrap(),
         te.lyric_embed(&lyr_ids).unwrap().to_dtype(DType::F32).unwrap(),
         ids_f(&cap_ids), ids_f(&lyr_ids))
    };

    let dit_ck = CompLoader::open(&dit_path, None, device).unwrap();
    let (lm_hints, my_5hz) = {
        let fsq = Fsq::load(&dit_ck, "tokenizer.quantizer").unwrap();
        let detok = Detokenizer::load(&dit_ck, &cfg).unwrap();
        let c5 = fsq.get_output_from_indices(&codes).unwrap();
        (detok.forward(&c5).unwrap(), c5)
    };
    stats(&lm_hints, "lm_hints (src_latents, NLC)");
    stats(&my_5hz, "my 5hz (fsq out)");
    let cond = ConditionEncoder::load(&dit_ck, &cfg).unwrap();
    let text_emb = cond.text_project(&text_hidden).unwrap();
    let lyric_emb = cond.lyric_encode(&lyric_hidden).unwrap();
    // Production-пакет [lyric, timbre(silence-750), text] (как pipeline.rs generate_music).
    let timbre_ref = Tensor::zeros(vec![1usize, cfg.timbre_fix_frame, 64usize], DType::F32, device).unwrap();
    let enc = cond.forward_full(&text_hidden, &lyric_hidden, &timbre_ref).unwrap();
    stats(&enc, "encoder_hidden");
    stats(&text_hidden, "caption_hidden (qwen3-emb out)");
    let l = enc.dims()[1];
    let null = dit_ck.f32("null_condition_emb").unwrap().broadcast_as(vec![1usize, l, cfg.encoder_hidden_size]).unwrap().contiguous().unwrap();

    let vae = AceStepVae::open(&vae_path, device).unwrap();
    // decode lm_hints directly (NLC [1,T,64] → NCL [1,64,T])
    save_mono(&vae, &lm_hints.transpose(1,2).unwrap().contiguous().unwrap(), "/tmp/lm_hints_direct.wav");

    // 3. DiT denoise
    let t = lm_hints.dims()[1];
    let chunk = Tensor::ones(vec![1usize, t, 64], DType::F32, device).unwrap();
    let context = Tensor::cat(&[&lm_hints, &chunk], 2).unwrap();
    let x0 = Tensor::randn_seeded(vec![1usize, t, 64], 42, Device::Cpu).unwrap().to_device(device).unwrap();
    let dit = Dit::load(&dit_ck, &cfg, DType::BF16, DType::BF16).unwrap();
    // Диагностика: влияет ли conditioning на velocity? cos(v_cond, v_null) и cos(v, v_zeroctx).
    let cosf = |a: &Tensor, b: &Tensor| -> f32 {
        let av: Vec<f32> = a.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let bv: Vec<f32> = b.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let dot: f32 = av.iter().zip(&bv).map(|(x,y)| x*y).sum();
        let na: f32 = av.iter().map(|x| x*x).sum::<f32>().sqrt();
        let nb: f32 = bv.iter().map(|x| x*x).sum::<f32>().sqrt();
        dot/(na*nb+1e-9)
    };
    let v_cond = dit.forward(&x0, 1.0, 1.0, &context, &enc).unwrap();
    // Мой ПОЛНЫЙ sampler (32 шага, g=7, shift=3, APG-F64) — итоговый латент для сверки с ai/.
    let opts32 = SamplerOptions { steps: 32, shift: 3.0, guidance_scale: 7.0, ..Default::default() };
    let syn_latent32 = denoise(&dit, &x0, &context, &enc, Some(&null), &opts32).unwrap();
    stats(&syn_latent32, "syn_latent32 (final, NLC)");
    // Дамп входов + velocity + ИТОГОВОГО латента + null для полной e2e-сверки с ai/.
    let codes_t = Tensor::from_vec(codes.iter().map(|&c| c as f32).collect::<Vec<f32>>(), vec![1usize, codes.len()], device).unwrap();
    dump_st(&[("x0", &x0), ("src", &lm_hints), ("enc", &enc), ("null", &null), ("syn_vel", &v_cond), ("syn_latent32", &syn_latent32), ("codes", &codes_t), ("my5hz", &my_5hz), ("cap_ids", &cap_ids_t), ("text_hidden", &text_hidden), ("text_emb", &text_emb),
        ("lyr_ids", &lyr_ids_t), ("lyric_hidden", &lyric_hidden), ("lyric_emb", &lyric_emb)], "/tmp/dit_io.safetensors");
    let v_null = dit.forward(&x0, 1.0, 1.0, &context, &null).unwrap();
    let zeroctx = Tensor::zeros(vec![1usize, t, 128], DType::F32, device).unwrap();
    let v_zeroctx = dit.forward(&x0, 1.0, 1.0, &zeroctx, &enc).unwrap();
    eprintln!("[iso] cos(v_cond, v_null)={:.5}  (≈1 → encoder_hidden НЕ влияет)", cosf(&v_cond, &v_null));
    eprintln!("[iso] cos(v_cond, v_zeroctx)={:.5}  (≈1 → context/src_latents НЕ влияет)", cosf(&v_cond, &v_zeroctx));
    stats(&v_cond, "velocity@t=1 cond");

    for (steps, g) in [(8usize, 7.0f32), (32, 7.0), (60, 7.0)] {
        let opts = SamplerOptions { steps, shift: 3.0, guidance_scale: g, ..Default::default() };
        let latent = denoise(&dit, &x0, &context, &enc, Some(&null), &opts).unwrap();
        stats(&latent, &format!("denoised latent steps={steps} g={g}"));
        save_mono(&vae, &latent.transpose(1,2).unwrap().contiguous().unwrap(), &format!("/tmp/denoised_s{steps}.wav"));
    }
}
