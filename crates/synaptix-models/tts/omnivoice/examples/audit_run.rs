//! Прогон заданий для сверки с upstream OmniVoice: JSON-список заданий →
//! wav + метаданные (длина цели, коды). Формат задания — как у эталонного
//! скрипта: `{id, text, lang, instruct, seed, cfg: {position_temperature,
//! class_temperature, num_step, guidance_scale, postprocess_output},
//! target_len?, mode?}`. `mode`: `design` (по умолчанию — как `generate`
//! upstream), `anchor` — как служба synaptixd: опорные фразы `anchor` по
//! описанию, текст — продолжение опоры без описания.
//!
//! `audit_run <omnivoice.syn> <jobs.json> <out_dir> [cpu|cuda] [f32|bf16|f16]`

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::config::OmniVoiceGenerationConfig;
use synaptix_tts_omnivoice::pipeline::OmniVoicePipeline;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    let (bundle, jobs, out) = (&a[1], &a[2], std::path::PathBuf::from(&a[3]));
    let device = match a.get(4).map(String::as_str) {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let dtype = match a.get(5).map(String::as_str) {
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        _ => DType::F32,
    };
    std::fs::create_dir_all(&out).unwrap();
    let pl = OmniVoicePipeline::from_syn(bundle, device, dtype).expect("from_syn");
    let jobs: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(jobs).unwrap()).unwrap();
    for j in jobs {
        let id = j["id"].as_str().unwrap();
        let wav_path = out.join(format!("{id}.wav"));
        if wav_path.exists() && !j["force"].as_bool().unwrap_or(false) {
            continue;
        }
        let text = j["text"].as_str().unwrap();
        let lang = j["lang"].as_str();
        let instruct = j["instruct"].as_str();
        let c = &j["cfg"];
        let f = |k: &str, d: f32| c[k].as_f64().map(|v| v as f32).unwrap_or(d);
        let gen = OmniVoiceGenerationConfig {
            num_step: c["num_step"].as_u64().map(|v| v as usize).unwrap_or(32),
            guidance_scale: f("guidance_scale", 2.0),
            position_temperature: f("position_temperature", 5.0),
            class_temperature: f("class_temperature", 0.0),
            postprocess_output: c["postprocess_output"].as_bool().unwrap_or(true),
            seed: Some(j["seed"].as_u64().unwrap_or(0)),
            ..Default::default()
        };
        let target_len = j["target_len"].as_u64().map(|v| v as usize);
        let t0 = std::time::Instant::now();
        let (wav, codes, tl) = match j["mode"].as_str().unwrap_or("design") {
            "anchor" => {
                let anchor: Vec<String> = j["anchor"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
                let mut voice = None;
                for s in &anchor {
                    pl.synthesize_continuing(s, &mut voice, lang, instruct, 1.0, &gen, f32::MAX).unwrap();
                }
                // Как служба: текст — продолжение опоры (0 с дорастания).
                let mut v = voice;
                let w = pl.synthesize_continuing(text, &mut v, lang, instruct, 1.0, &gen, 0.0).unwrap();
                let c = synaptix_core::tensor::Tensor::from_vec(vec![0i64; 8], vec![8, 1], Device::Cpu).unwrap();
                (w, c, 0)
            }
            _ => {
                let (w, c) = pl.synthesize_codes(text, None, lang, instruct, 1.0, target_len, &gen).unwrap();
                let tl = c.dims()[1];
                (w, c, tl)
            }
        };
        let dt = t0.elapsed().as_secs_f32();
        synaptix_audio::write_wav_mono_f32(&wav_path, &wav, pl.output_sample_rate() as u32).unwrap();
        let codes_v: Vec<i64> = codes
            .to_device(Device::Cpu)
            .and_then(|t| t.to_dtype(DType::I64))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<i64>())
            .unwrap();
        let meta = serde_json::json!({
            "id": id, "sec": wav.len() as f32 / pl.output_sample_rate() as f32, "time": dt,
            "target_lens": [[tl]], "codes": codes_v,
        });
        std::fs::write(out.join(format!("{id}.json")), meta.to_string()).unwrap();
        println!("{id}: {:.2} с звука за {dt:.1} с, T={tl}", wav.len() as f32 / pl.output_sample_rate() as f32);
    }
}
