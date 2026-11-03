//! Сверка VibeVoice с эталоном: `vv_audit модель(.syn|каталог HF) задания.tsv выход
//! [--cpu] [--f32] [--quant mxfp8|nvfp4] [--steps N] [--cfg X] [--seed S] [--ode] [--dump]`
//! (`--vae латенты.bin` — только сверка VAE).
//!
//! Задания — строки `имя<TAB>образцы через запятую или -<TAB>сценарий` (переводы
//! строк в сценарии — `|`). На каждое — `выход/имя.wav` и `выход/имя.tokens`
//! (выбранные LM токены). `--dump` — для первого задания промежуточные тензоры
//! (`ids`, среднее латентов образца, последнее скрытое состояние префилла,
//! первый латент диффузии с шумом из того же ГПСЧ) в `выход/dump_*.bin` (f32 LE)
//! для сравнения с эталоном (scripts/reference/vibevoice_ref.py).

use std::io::Write;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_vibevoice::generate::{NormalRng, SpeechGenerator};
use synaptix_tts_vibevoice::{GenerationConfig, VibeVoicePipeline, VoiceSample};

fn dump(dir: &str, name: &str, t: &Tensor) {
    let v = t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();
    let mut f = std::fs::File::create(format!("{dir}/dump_{name}.bin")).unwrap();
    for x in &v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    eprintln!("dump {name}: {:?}", t.dims());
}

/// Декод латентов `[T, 64]` (f32 LE) акустическим VAE по кадру со стриминговым
/// кэшем (`vae_stream`) и целиком (`vae_full`), семантический энкодер по кускам
/// потока (`vae_sem`) — то же делает эталон на тех же латентах.
fn vae_check(model: &str, lat: &str, out: &str, device: Device, dtype: DType) {
    use synaptix_tts_vibevoice::vae::{AcousticTokenizer, SemanticTokenizer};
    use synaptix_tts_vibevoice::VibeVoiceCheckpoint;
    let ck = VibeVoiceCheckpoint::open(model, device, dtype).unwrap();
    let ac = AcousticTokenizer::load(ck.source(), &ck.config.acoustic_tokenizer_config, "model.acoustic_tokenizer").unwrap();
    let se = SemanticTokenizer::load(ck.source(), &ck.config.semantic_tokenizer_config, "model.semantic_tokenizer").unwrap();
    let raw = std::fs::read(lat).unwrap();
    let v: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let t = v.len() / 64;
    let all = Tensor::from_vec(v, vec![1usize, t, 64], device).unwrap().to_dtype(dtype).unwrap();
    dump(out, "vae_full", &ac.decode(&all, None).unwrap());
    let mut cache = ac.new_cache();
    let mut scache = se.new_cache();
    let (mut pcm, mut sem) = (Vec::new(), Vec::new());
    for i in 0..t {
        let f = all.narrow(1, i, 1).unwrap().contiguous().unwrap();
        let chunk = ac.decode(&f, Some(&mut cache)).unwrap();
        sem.push(se.encode(&chunk, Some(&mut scache)).unwrap());
        pcm.push(chunk);
    }
    let r: Vec<&Tensor> = pcm.iter().collect();
    dump(out, "vae_stream", &Tensor::cat(&r, 2).unwrap());
    let r: Vec<&Tensor> = sem.iter().collect();
    dump(out, "vae_sem", &Tensor::cat(&r, 1).unwrap());
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |f: &str| a.iter().any(|x| x == f);
    let opt = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)).cloned();
    let device = if flag("--cpu") { Device::Cpu } else { Device::Cuda(0) };
    let dtype = if flag("--f32") { DType::F32 } else { DType::BF16 };
    let quant = match opt("--quant").as_deref() {
        Some("mxfp8") => Some(DType::MXFP8),
        Some("nvfp4") => Some(DType::NVFP4),
        _ => None,
    };
    let (model, jobs, out) = (&a[0], &a[1], &a[2]);
    std::fs::create_dir_all(out).unwrap();
    if let Some(lat) = opt("--vae") {
        vae_check(model, &lat, out, device, dtype);
        return;
    }
    let cfg = GenerationConfig {
        ddpm_inference_steps: opt("--steps").map(|v| v.parse().unwrap()).unwrap_or(10),
        seed: opt("--seed").map(|v| v.parse().unwrap()).unwrap_or(0),
        sde: !flag("--ode"),
        ..Default::default()
    };
    let cfg = GenerationConfig { cfg_scale: opt("--cfg").map(|v| v.parse().unwrap()).unwrap_or(cfg.cfg_scale), ..cfg };
    let p = VibeVoicePipeline::from_syn_quant(model, device, dtype, quant).unwrap();
    let text = std::fs::read_to_string(jobs).unwrap();
    for (k, line) in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).enumerate() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (name, vs, script) = (f[0], f[1], f[2].replace('|', "\n"));
        let voices: Vec<VoiceSample> = if vs == "-" {
            Vec::new()
        } else {
            vs.split(',').map(|v| VoiceSample::from_path(v).unwrap()).collect()
        };
        if k == 0 && flag("--dump") {
            let rate = p.sample_rate();
            let wavs: Vec<Vec<f32>> = voices.iter().map(|v| v.to_rate(rate).unwrap()).collect();
            let prompt = p.processor().build_prompt(&script, &wavs).unwrap();
            std::fs::write(
                format!("{out}/dump_ids.txt"),
                prompt.input_ids.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
            )
            .unwrap();
            let m = p.model();
            let mut g = SpeechGenerator::with_rng(m, p.processor(), NormalRng::new(cfg.seed)).unwrap();
            if !prompt.speech_tensors.is_empty() {
                let n = prompt.speech_tensors.len();
                let l = prompt.speech_tensors[0].len();
                let flat: Vec<f32> = prompt.speech_tensors.concat();
                let audio = Tensor::from_vec(flat, vec![n, 1usize, l], m.device)
                    .unwrap()
                    .to_dtype(m.dtype)
                    .unwrap();
                dump(out, "voice_mean", &m.acoustic.encode(&audio, None).unwrap());
            }
            let voice = g.encode_voice_latents(&prompt).unwrap();
            let embeds = g.build_prompt_embeds(&prompt, voice.as_ref()).unwrap();
            dump(out, "prompt_embeds", &embeds);
            let mut cache = m.lm.new_cache(prompt.input_ids.len() + 8).unwrap();
            let hidden = m.lm.forward(&embeds, &mut cache).unwrap();
            let last = hidden.dims()[1] - 1;
            let pos = hidden.narrow(1, last, 1).unwrap().contiguous().unwrap().reshape(vec![1usize, m.lm.hidden_size()]).unwrap();
            dump(out, "prefill_last", &pos);
            let mut ncache = m.lm.new_cache(8).unwrap();
            let ni = m.lm.embed_tokens(&[p.processor().speech_start_id]).unwrap();
            let nh = m.lm.forward(&ni, &mut ncache).unwrap();
            let neg = nh.reshape(vec![1usize, m.lm.hidden_size()]).unwrap();
            dump(out, "neg_first", &neg);
            let init = g.randn(vec![1usize, m.config.acoustic_vae_dim()]).unwrap();
            dump(out, "init_noise", &init);
            let z = cfg.sde.then(|| g.randn(vec![cfg.ddpm_inference_steps, m.config.acoustic_vae_dim()]).unwrap());
            let lat = g.sample_latent_with(&pos, &neg, cfg.cfg_scale, cfg.ddpm_inference_steps, &init, z.as_ref()).unwrap();
            dump(out, "first_latent", &lat);
            let lat3 = lat.reshape(vec![1usize, 1usize, m.config.acoustic_vae_dim()]).unwrap();
            let mut ac = m.acoustic.new_cache();
            let chunk = m.acoustic.decode(&m.unscale_latents(&lat3).unwrap(), Some(&mut ac)).unwrap();
            dump(out, "first_chunk", &chunk);
            let mut sc = m.semantic.new_cache();
            dump(out, "first_semantic", &m.semantic.encode(&chunk, Some(&mut sc)).unwrap());
        }
        let t = std::time::Instant::now();
        let o = p.synthesize(&script, &voices, &cfg).unwrap();
        let dur = o.audio.len() as f32 / p.sample_rate() as f32;
        eprintln!(
            "{name}: {dur:.2} с за {:.1} с, токенов {}, предел {}",
            t.elapsed().as_secs_f32(),
            o.tokens.len(),
            o.reached_max
        );
        p.save_wav(&o.audio, format!("{out}/{name}.wav")).unwrap();
        std::fs::write(
            format!("{out}/{name}.tokens"),
            o.tokens.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
        )
        .unwrap();
    }
}
