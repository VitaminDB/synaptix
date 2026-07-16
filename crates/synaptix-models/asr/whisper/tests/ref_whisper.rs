//! Bit-exact гейты Whisper против HF (reference в tests/reference_data/asr_whisper).
//! Reference генерится `scripts/reference/gen_asr_whisper_*.py` (venv).

use std::path::PathBuf;

use synaptix_asr_whisper::mel::whisper_log_mel;
use synaptix_asr_whisper::{Task, WhisperEncoder, WhisperModel, WhisperPipeline, WhisperWeights};
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_test_utils::{assert_allclose, load_case};

fn bundle_path() -> Option<PathBuf> {
    let p = PathBuf::from("models/whisper-large-v3-turbo.syn");
    p.exists().then_some(p)
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.contiguous()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn cos_sim(a: &Tensor, b: &Tensor) -> f64 {
    let a = to_f32(a);
    let b = to_f32(b);
    assert_eq!(a.len(), b.len(), "cos_sim: разная длина");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_err(a: &Tensor, b: &Tensor) -> f64 {
    let a = to_f32(a);
    let b = to_f32(b);
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

/// argmax по каждой строке `[rows, cols]`.
fn argmax_rows(t: &Tensor) -> Vec<usize> {
    let d = t.dims();
    let (rows, cols) = (d[0], d[1]);
    let v = to_f32(t);
    (0..rows)
        .map(|r| {
            let s = &v[r * cols..(r + 1) * cols];
            let mut best = 0usize;
            for (i, &x) in s.iter().enumerate() {
                if x > s[best] {
                    best = i;
                }
            }
            best
        })
        .collect()
}

#[test]
fn decoder_logits_step0() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(path) = bundle_path() else {
        eprintln!("bundle отсутствует — пропуск");
        return;
    };
    let case = load_case("asr_whisper", "whisper_dec");
    let token_ids: Vec<u32> = to_f32(case.get("token_ids").expect("token_ids"))
        .iter()
        .map(|x| x.round() as u32)
        .collect();
    let enc_out = case.get("encoder_out").expect("encoder_out").unsqueeze(0).unwrap(); // [1,1500,1280]
    let expected = case.get("logits").expect("logits"); // [S, vocab]

    let w = WhisperWeights::open(&path, Device::Cpu, DType::F32).expect("open");
    let model = WhisperModel::load(&w).expect("load model");
    let logits = model
        .decoder
        .forward_prefix(&token_ids, &enc_out)
        .expect("decoder forward");
    let logits = logits.squeeze(0).unwrap(); // [S, vocab]

    let mine_am = argmax_rows(&logits);
    let hf_am = argmax_rows(expected);
    eprintln!("decoder argmax mine={mine_am:?} hf={hf_am:?}");
    assert_eq!(mine_am, hf_am, "argmax per position must match HF");

    let cos = cos_sim(&logits, expected);
    let mae = max_abs_err(&logits, expected);
    eprintln!("decoder_logits_step0: cos={cos:.8} max_abs_err={mae:.4}");
    assert!(cos >= 0.9999, "decoder logits cos_sim too low: {cos}");
}

#[test]
fn kv_cache_consistency() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(path) = bundle_path() else {
        return;
    };
    let case = load_case("asr_whisper", "whisper_dec");
    let token_ids: Vec<u32> = to_f32(case.get("token_ids").unwrap())
        .iter()
        .map(|x| x.round() as u32)
        .collect();
    let enc_out = case.get("encoder_out").unwrap().unsqueeze(0).unwrap();

    let w = WhisperWeights::open(&path, Device::Cpu, DType::F32).expect("open");
    let model = WhisperModel::load(&w).expect("load");

    // Полный teacher-forced прогон.
    let prefix_logits = model.decoder.forward_prefix(&token_ids, &enc_out).unwrap();
    let prefix_logits = prefix_logits.squeeze(0).unwrap(); // [S, vocab]
    let s = token_ids.len();
    let vocab = prefix_logits.dims()[1];
    let last_prefix = prefix_logits.narrow(0, s - 1, 1).unwrap(); // [1, vocab]

    // Инкрементальный прогон через KV-cache.
    let mut cache = model.decoder.init_cache(&enc_out).unwrap();
    let mut last_step = None;
    for (pos, &t) in token_ids.iter().enumerate() {
        last_step = Some(model.decoder.decode_step(t, pos, &mut cache).unwrap());
    }
    let last_step = last_step.unwrap().reshape(vec![1usize, vocab]).unwrap();

    let cos = cos_sim(&last_step, &last_prefix);
    let mae = max_abs_err(&last_step, &last_prefix);
    eprintln!("kv_cache_consistency: cos={cos:.8} max_abs_err={mae:.5}");
    assert!(cos >= 0.99999, "kv-cache path diverges from full prefix: cos={cos}");
}

#[test]
fn transcribe_matches_hf() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(path) = bundle_path() else {
        return;
    };
    let case = load_case("asr_whisper", "whisper_enc");
    let audio = to_f32(case.get("audio_16k").unwrap()); // 480000 сэмплов (30 с)

    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            synaptix_test_utils::reference_data_path("asr_whisper", "transcribe.json"),
        )
        .expect("read transcribe.json"),
    )
    .unwrap();
    let exp_ids: Vec<u32> = expected["content_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let exp_text = expected["text"].as_str().unwrap().to_string();

    let pipe = WhisperPipeline::from_syn(&path, Device::Cpu, DType::F32).expect("pipeline");
    let ids = pipe.segment_token_ids(&audio, "en", Task::Transcribe).expect("decode");

    let matched = ids.iter().zip(exp_ids.iter()).take_while(|(a, b)| a == b).count();
    eprintln!(
        "transcribe: mine {} ids, hf {} ids, prefix-match {}/{}",
        ids.len(),
        exp_ids.len(),
        matched,
        exp_ids.len()
    );
    eprintln!("mine ids: {ids:?}");
    eprintln!("hf   text: {exp_text}");
    // Строгое совпадение id ⇒ совпадение текста (decode детерминирован).
    assert_eq!(ids, exp_ids, "token ids must match HF greedy");
}

#[test]
fn timestamps_match_hf() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(path) = bundle_path() else {
        return;
    };
    let case = load_case("asr_whisper", "whisper_enc");
    let audio = to_f32(case.get("audio_16k").unwrap());

    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(synaptix_test_utils::reference_data_path("asr_whisper", "timestamps.json"))
            .expect("read timestamps.json"),
    )
    .unwrap();
    let exp_ids: Vec<u32> = expected["gen_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let pipe = WhisperPipeline::from_syn(&path, Device::Cpu, DType::F32).expect("pipeline");
    let ids = pipe
        .segment_token_ids_timestamps(&audio, "en", Task::Transcribe)
        .expect("ts decode");

    let matched = ids.iter().zip(exp_ids.iter()).take_while(|(a, b)| a == b).count();
    eprintln!("timestamps: mine {} ids, hf {} ids, prefix-match {}", ids.len(), exp_ids.len(), matched);
    eprintln!("mine: {ids:?}");
    eprintln!("hf:   {exp_ids:?}");
    // HF может завершать на eot (его в gen_ids нет) — сравниваем до длины HF.
    let cmp_len = ids.len().min(exp_ids.len());
    assert_eq!(&ids[..cmp_len], &exp_ids[..cmp_len], "timestamp token ids must match HF");
}

#[test]
fn mel_matches_hf() {
    synaptix_kernels_cpu::ensure_registered();
    let case = load_case("asr_whisper", "whisper_enc");
    let audio = case.get("audio_16k").expect("audio_16k");
    let expected = case.get("input_features").expect("input_features"); // [128,3000]

    let samples = to_f32(audio);
    let (flat, n_mels, n_frames) = whisper_log_mel(&samples, 128, 3000).expect("mel");
    assert_eq!((n_mels, n_frames), (128, 3000));
    let mine = Tensor::from_vec(flat, (n_mels, n_frames), Device::Cpu).unwrap();

    let mae = max_abs_err(&mine, expected);
    eprintln!("mel_matches_hf: max_abs_err={mae:.6}");
    assert_allclose(&mine, expected, 2e-3, 1e-3);
}

#[test]
fn encoder_full() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(path) = bundle_path() else {
        eprintln!("bundle отсутствует — пропуск");
        return;
    };
    let case = load_case("asr_whisper", "whisper_enc");
    let feats = case.get("input_features").expect("input_features"); // [128,3000]
    let expected = case.get("encoder_out").expect("encoder_out"); // [1500,1280]

    let mel = feats.unsqueeze(0).unwrap(); // [1,128,3000]
    let w = WhisperWeights::open(&path, Device::Cpu, DType::F32).expect("open");
    let enc = WhisperEncoder::load(&w).expect("load encoder");
    let out = enc.forward(&mel).expect("encoder forward");
    let out = out.squeeze(0).unwrap(); // [1500,1280]

    let cos = cos_sim(&out, expected);
    let mae = max_abs_err(&out, expected);
    eprintln!("encoder_full: cos={cos:.8} max_abs_err={mae:.6}");
    assert!(cos >= 0.9999, "encoder cos_sim too low: {cos}");
    assert!(mae < 0.05, "encoder max_abs_err too high: {mae}");
}
