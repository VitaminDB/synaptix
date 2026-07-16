//! Гейт A voice-clone: `create_voice_clone_prompt(extr.wav, ref_text)` →
//! ref_audio_tokens — против Python-эталона (`reference/dump_clone.py`).
//!
//! Эталон в `tmp/ov_ref/`:
//!   clone_ref_tokens.npy (C, T_ref) i64 — model.create_voice_clone_prompt(...).ref_audio_tokens,
//!   clone_ref_pre.npy    (N,) f32       — preprocessed ref-wav 24k (rms+remove_silence+clip),
//!   clone_meta.json      {ref_text, ref_rms, ...}.
//!
//! Изоляция стадий:
//!   1. preprocess: сверяем длину/cos нашей preprocessed-wav с Python clone_ref_pre.npy
//!      (load_audio + rms-scale + remove_silence + clip). Расхождение длины тут =
//!      дрейф remove_silence (ms-гранулярность pydub).
//!   2. encode: коды от НАШЕЙ preprocessed-wav vs clone_ref_tokens — % совпадения ≥ 95%.
//!
//! Гейт: коды ≥ 95% (препроцесс + encode end-to-end через create_voice_clone_prompt).

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::OmniVoicePipeline;

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";
const WAV: &str = "extr.wav";

const REF_TEXT: &str = "Я тебе скажу, что реальная тема. Сходи в круглосуточный и купи нам печенье Джафа. Или шоколадки. Ему шоколадку. А Ириша к вам не купить, чтоб пломбы повышкакивали, клоуны хуевые! Поберегись!";

fn parse_npy(path: &str) -> (Vec<usize>, Vec<u8>, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let (hdr_len, data_off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[data_off..data_off + hdr_len]).unwrap();
    let descr = {
        let i = header.find("'descr'").unwrap();
        let rest = &header[i + 7..];
        let q1 = rest.find('\'').unwrap();
        let q2 = rest[q1 + 1..].find('\'').unwrap();
        rest[q1 + 1..q1 + 1 + q2].to_string()
    };
    let shape = {
        let i = header.find("'shape'").unwrap();
        let rest = &header[i..];
        let lp = rest.find('(').unwrap();
        let rp = rest.find(')').unwrap();
        rest[lp + 1..rp]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    };
    let data = bytes[data_off + hdr_len..].to_vec();
    (shape, data, descr)
}

fn load_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("f4"), "{path} descr={descr} (expect <f4)");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        v.push(f32::from_le_bytes(data[o..o + 4].try_into().unwrap()));
    }
    (shape, v)
}

fn load_i64(path: &str) -> (Vec<usize>, Vec<i64>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("i8"), "{path} descr={descr} (expect <i8)");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 8;
        v.push(i64::from_le_bytes(data[o..o + 8].try_into().unwrap()));
    }
    (shape, v)
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

#[test]
fn clone_prompt_gate() {
    let lm_path = format!("{UNPACK}/model.safetensors");
    let codec_path = format!("{UNPACK}/audio_tokenizer/model.safetensors");
    if !Path::new(&lm_path).exists()
        || !Path::new(&codec_path).exists()
        || !Path::new(WAV).exists()
        || !Path::new(&format!("{REF}/clone_ref_tokens.npy")).exists()
    {
        eprintln!("[clone_prompt_gate] SKIP: weights/wav/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let pipe = OmniVoicePipeline::from_unpacked(UNPACK, Device::Cpu, DType::F32)
        .expect("from_unpacked");

    // Стадия 1 (изоляция препроцесса): preprocessed wav.
    if Path::new(&format!("{REF}/clone_ref_pre.npy")).exists() {
        let (sr, hop) = (24000usize, 960usize);
        // Воспроизводим препроцесс через публичные prompt-утилиты.
        let (mono, in_sr) =
            synaptix_audio::read_wav_mono_f32(WAV).expect("read extr.wav");
        let mut wav = if in_sr as usize != sr {
            synaptix_tts_omnivoice::audio_encode::resample(&mono, in_sr as usize, sr)
        } else {
            mono
        };
        let rms = {
            let mut s = 0.0f64;
            for &v in &wav {
                s += (v as f64) * (v as f64);
            }
            (s / wav.len() as f64).sqrt() as f32
        };
        if rms > 0.0 && rms < 0.1 {
            let sc = 0.1 / rms;
            for x in wav.iter_mut() {
                *x *= sc;
            }
        }
        wav = synaptix_tts_omnivoice::remove_silence(&wav, sr, 200, 100, 200);
        let clip = wav.len() % hop;
        if clip > 0 {
            wav.truncate(wav.len() - clip);
        }
        let (pre_shape, pre) = load_f32(&format!("{REF}/clone_ref_pre.npy"));
        let pre_n: usize = pre_shape.iter().product();
        let c = cos(&wav, &pre);
        eprintln!(
            "[clone_prompt_gate] preprocess: got {} samples ; ref {} samples ; cos={c:.6} \
             (len diff = {} samples = {:.1} ms)",
            wav.len(),
            pre_n,
            wav.len() as i64 - pre_n as i64,
            (wav.len() as i64 - pre_n as i64) as f64 / 24.0,
        );
    }

    // Стадия 2 (end-to-end): create_voice_clone_prompt → ref_audio_tokens.
    let prompt = pipe
        .create_voice_clone_prompt(WAV, REF_TEXT)
        .expect("create_voice_clone_prompt");
    eprintln!(
        "[clone_prompt_gate] ref_rms={:.6} ref_text(+punct)={:?}",
        prompt.ref_rms, prompt.ref_text
    );

    let got = prompt
        .ref_audio_tokens
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    let got_dims = prompt.ref_audio_tokens.dims().to_vec();

    let (ref_shape, ref_codes) = load_i64(&format!("{REF}/clone_ref_tokens.npy"));
    let (n_q, t_ref) = (ref_shape[0], ref_shape[1]);
    eprintln!(
        "[clone_prompt_gate] codes got {:?} ; ref [{n_q}, {t_ref}]",
        got_dims
    );

    // Если длины кодов разошлись (дрейф remove_silence) — сравниваем по min-T.
    let got_t = got_dims[got_dims.len() - 1];
    let t = got_t.min(t_ref);
    let mut matches = 0usize;
    let mut total = 0usize;
    let mut per_q_mis = vec![0usize; n_q];
    for c in 0..n_q {
        for j in 0..t {
            total += 1;
            let g = got[c * got_t + j];
            let r = ref_codes[c * t_ref + j];
            if g == r {
                matches += 1;
            } else {
                per_q_mis[c] += 1;
            }
        }
    }
    let pct = 100.0 * matches as f64 / total as f64;
    eprintln!(
        "[clone_prompt_gate] OVERALL match {matches}/{total} = {pct:.2}% (compared T={t}; got_T={got_t} ref_T={t_ref})"
    );
    for c in 0..n_q {
        eprintln!("[clone_prompt_gate]   q{c}: {} mismatches / {t}", per_q_mis[c]);
    }

    assert_eq!(got_t, t_ref, "ref code length mismatch (preprocess drift): got {got_t} ref {t_ref}");
    assert!(pct >= 95.0, "code match {pct:.2}% < 95%");
}
