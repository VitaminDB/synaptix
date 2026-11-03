//! Сверка с upstream (OpenBMB/VoxCPM): входы и промежуточные данные в файлы,
//! которые сравнивает Python-эталон.
//!
//! - `tok <tokenizer.json> <текст>…` — идентификаторы токенов (по строке JSON на текст);
//! - `gen <bundle.syn> <f32|bf16> <текст> <зерно> <префикс>` — синтез без
//!   повторов по браку (как upstream с `retry_badcase=False`): `<префикс>.noise`
//!   (шум CFM всех шагов, f32 `[max_len,64,4]`), `<префикс>.lat` (латенты
//!   `[64, T·4]`, f32), `<префикс>.wav`, число шагов — в stdout;
//! - `vae <bundle.syn> <латенты.lat> <выход.wav>` — только декодер звука.
//!
//! `REF=<wav>` у `gen` — клон с записи (режим reference upstream).
use std::io::Write;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tts_voxcpm::model::InferOptions;
use synaptix_tts_voxcpm::pipeline::{GenerateOptions, VoxCpmPipeline};
use synaptix_tts_voxcpm::tokenizer::{TextTokenizer, AUDIO_START};
use synaptix_tts_voxcpm::{VoxCheckpoint, VoxCpmModel};

fn write_f32(path: &str, v: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap();
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn prec(s: &str) -> DType {
    match s {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => DType::F32,
    }
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args[0].as_str() {
        "tok" => {
            let t = TextTokenizer::from_bytes(&std::fs::read(&args[1]).unwrap()).unwrap();
            for text in &args[2..] {
                println!("{:?}", t.encode(text).unwrap());
            }
        }
        "gen" => {
            let (bundle, dt, text, seed, out) = (&args[1], prec(&args[2]), &args[3], args[4].parse::<u64>().unwrap(), &args[5]);
            let ck = VoxCheckpoint::open(bundle, Device::Cpu, dt).unwrap();
            let tok = TextTokenizer::from_bytes(&ck.read_file("tokenizer.json").unwrap()).unwrap();
            let model = VoxCpmModel::load(&ck).unwrap();
            let cfg = model.config.clone();
            let (p, d) = (cfg.patch_size, cfg.feat_dim);
            let mut ids = tok.encode(text).unwrap();
            let target_len = ids.len();
            ids.push(AUDIO_START);
            let n = ids.len();
            let max_len = std::env::var("MAX_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or((target_len as f32 * 6.0) as usize + 10);
            // Шум ровно как в cfm::sample: зерно + номер шага.
            let mut noise = Vec::new();
            for i in 0..max_len {
                noise.extend(Tensor::randn_seeded((1usize, d, p), seed.wrapping_add(i as u64), Device::Cpu).unwrap().to_vec1_flat_f32());
            }
            write_f32(&format!("{out}.noise"), &noise);
            let mut tokens: Vec<u32> = Vec::new();
            let mut tm: Vec<f32> = Vec::new();
            let mut am: Vec<f32> = Vec::new();
            let mut feats: Vec<Tensor> = Vec::new();
            if let Ok(r) = std::env::var("REF") {
                let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(&r).unwrap();
                let pcm = synaptix_tts_voxcpm::audio_io::resampled(pcm, sr, cfg.audio_vae_config.sample_rate).unwrap();
                let pcm = synaptix_tts_voxcpm::audio_io::pad_to_multiple(pcm, p * cfg.audio_vae_config.hop_length(), synaptix_tts_voxcpm::audio_io::PadMode::Right);
                let len = pcm.len();
                let mu = model.audio_vae.encode(&Tensor::from_vec(pcm, vec![1usize, 1, len], Device::Cpu).unwrap()).unwrap();
                let t = mu.dims()[2] / p;
                let f = mu.squeeze(0).unwrap().reshape((d, t, p)).unwrap().permute([1usize, 2, 0]).unwrap().contiguous().unwrap().to_dtype(dt).unwrap();
                let z1 = Tensor::zeros(vec![1usize, p, d], dt, Device::Cpu).unwrap();
                tokens.push(103);
                tokens.extend(std::iter::repeat_n(0, t));
                tokens.push(104);
                tm.push(1.0);
                tm.extend(std::iter::repeat_n(0.0, t));
                tm.push(1.0);
                am.push(0.0);
                am.extend(std::iter::repeat_n(1.0, t));
                am.push(0.0);
                feats.extend([z1.clone(), f, z1]);
            }
            tokens.extend(&ids);
            tm.extend(std::iter::repeat_n(1.0, n));
            am.extend(std::iter::repeat_n(0.0, n));
            feats.push(Tensor::zeros(vec![n, p, d], dt, Device::Cpu).unwrap());
            let l = tokens.len();
            let token_t = Tensor::from_vec(tokens, vec![1usize, l], Device::Cpu).unwrap();
            let tm_t = Tensor::from_vec(tm, vec![1usize, l], Device::Cpu).unwrap().to_dtype(dt).unwrap();
            let am_t = Tensor::from_vec(am, vec![1usize, l], Device::Cpu).unwrap().to_dtype(dt).unwrap();
            let feat_t = Tensor::cat(&feats.iter().collect::<Vec<_>>(), 0).unwrap().unsqueeze(0).unwrap();
            let o = GenerateOptions::default();
            let infer = InferOptions { min_len: o.min_len, max_len, n_timesteps: o.n_timesteps, cfg_value: o.cfg_value, seed };
            let t0 = std::time::Instant::now();
            let (lat, _) = model.infer(&token_t, &tm_t, &feat_t, &am_t, 0, &infer).unwrap();
            let steps = lat.dims()[2] / p;
            let lat32 = lat.to_dtype(DType::F32).unwrap().contiguous().unwrap();
            write_f32(&format!("{out}.lat"), &lat32.to_vec1_flat_f32());
            let wav = model.audio_vae.decode(&lat32.to_dtype(DType::F32).unwrap()).unwrap();
            let n = wav.dims()[2];
            let pcm = wav.reshape((n,)).unwrap().to_vec1::<f32>().unwrap();
            synaptix_audio::io::write_wav_mono_f32(format!("{out}.wav"), &pcm, cfg.audio_vae_config.out_sample_rate as u32).unwrap();
            println!("steps={steps} max_len={max_len} stop={} sec={:.2} t={:.1}", steps < max_len, n as f32 / 48000.0, t0.elapsed().as_secs_f32());
        }
        "vae" => {
            let ck = VoxCheckpoint::open(&args[1], Device::Cpu, DType::F32).unwrap();
            let model = VoxCpmModel::load(&ck).unwrap();
            let lat = read_f32(&args[2]);
            let t = lat.len() / 64;
            let z = Tensor::from_vec(lat, vec![1usize, 64, t], Device::Cpu).unwrap();
            let wav = model.audio_vae.decode(&z).unwrap();
            let n = wav.dims()[2];
            let pcm = wav.reshape((n,)).unwrap().to_vec1::<f32>().unwrap();
            synaptix_audio::io::write_wav_mono_f32(&args[3], &pcm, 48000).unwrap();
        }
        "pipe" => {
            // Как служба без защиты: synthesize с retries=0 (и с защитой при RETRIES).
            let p = VoxCpmPipeline::from_bundle(&args[1], Device::Cpu, prec(&args[2])).unwrap();
            let seed = args[4].parse().unwrap();
            let retries = std::env::var("RETRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let o = GenerateOptions { seed, retries, ..Default::default() };
            let w = p.synthesize(&args[3], &o).unwrap();
            synaptix_audio::io::write_wav_mono_f32(&args[5], &w.pcm, w.sample_rate as u32).unwrap();
        }
        c => panic!("неизвестная команда {c}"),
    }
}

trait Flat {
    fn to_vec1_flat_f32(&self) -> Vec<f32>;
}

impl Flat for Tensor {
    fn to_vec1_flat_f32(&self) -> Vec<f32> {
        let n = self.dims().iter().product::<usize>();
        self.to_dtype(DType::F32).unwrap().contiguous().unwrap().reshape((n,)).unwrap().to_vec1::<f32>().unwrap()
    }
}
