//! Кросс-фреймворк-гейт NeMo Streaming Sortformer 4spk v2.1 (BATCH-путь, CPU F32).
//!
//! Эталон в `tmp/sortformer_ref/` (см. `reference/dump_sortformer.py`,
//! построен на ОФИЦИАЛЬНОМ NeMo): wav16/mel(1,128,T)/encoder_out(1,512,T')/
//! emb_seq(1,T',192)/trans(1,T',192)/preds(1,T',4). Веса = `.syn`-бандл.
//!
//! Кросс-фреймворк F32 НЕ бит-идентичен → постадийный per-row cosine + max-abs.
//! Guarded: skip если артефактов нет на диске.

use std::path::Path;

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_diarization_sortformer::model::SortformerModel;
use synaptix_diarization_sortformer::SortformerWeights;
use synaptix_core::device::Device;

const SYN: &str = "storage/syn_models/sortformer-streaming-4spk-v21.syn";
const REF: &str = "tmp/sortformer_ref";

fn parse_npy(path: &str) -> (Vec<usize>, Vec<u8>, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let (hdr_len, data_off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
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
        rest[lp + 1..rp].split(',').filter_map(|s| s.trim().parse::<usize>().ok()).collect::<Vec<_>>()
    };
    let data = bytes[data_off + hdr_len..].to_vec();
    (shape, data, descr)
}

fn load_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("f4"), "{path} descr={descr}");
    let n: usize = shape.iter().product();
    let raw: Vec<f32> =
        (0..n).map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap())).collect();
    (shape, raw)
}

/// Per-row cosine по последней оси `inner`. → (min_cos, argmin_row).
fn per_row_cos(got: &[f32], reference: &[f32], inner: usize) -> (f32, usize) {
    assert_eq!(got.len(), reference.len(), "len mismatch");
    let rows = got.len() / inner;
    let mut min_cos = f32::INFINITY;
    let mut min_row = 0usize;
    for r in 0..rows {
        let a = &got[r * inner..(r + 1) * inner];
        let b = &reference[r * inner..(r + 1) * inner];
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for k in 0..inner {
            dot += a[k] as f64 * b[k] as f64;
            na += (a[k] as f64).powi(2);
            nb += (b[k] as f64).powi(2);
        }
        let cos = (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32;
        if cos < min_cos {
            min_cos = cos;
            min_row = r;
        }
    }
    (min_cos, min_row)
}

/// Максимум |got − ref| по всем элементам.
fn max_abs(got: &[f32], reference: &[f32]) -> f32 {
    got.iter().zip(reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn mean_cos(got: &[f32], reference: &[f32], inner: usize) -> f32 {
    let rows = got.len() / inner;
    let mut s = 0f64;
    for r in 0..rows {
        let a = &got[r * inner..(r + 1) * inner];
        let b = &reference[r * inner..(r + 1) * inner];
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for k in 0..inner {
            dot += a[k] as f64 * b[k] as f64;
            na += (a[k] as f64).powi(2);
            nb += (b[k] as f64).powi(2);
        }
        s += dot / (na.sqrt() * nb.sqrt()).max(1e-12);
    }
    (s / rows as f64) as f32
}

#[test]
fn sortformer_streaming_gate() {
    if !Path::new(SYN).exists() || !Path::new(&format!("{REF}/stream_preds.npy")).exists() {
        eprintln!("[stream_gate] SKIP");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let w = SortformerWeights::open(SYN, Device::Cpu, DType::F32).expect("open");
    let model = SortformerModel::load(&w).expect("load");
    let (ws, wav) = load_f32(&format!("{REF}/wav16_long.npy"));
    eprintln!("[stream_gate] wav {:?} ({:.1}s)", ws, wav.len() as f32 / 16000.0);

    let preds = model.diarize_pcm_streaming(&wav).expect("stream");
    let (pr_shape, pr_ref) = load_f32(&format!("{REF}/stream_preds.npy"));
    assert_eq!(preds.dims(), &pr_shape[..], "stream preds shape");
    let got = flat(&preds);
    let (cos, row) = per_row_cos(&got, &pr_ref, pr_shape[2]);
    let agree = got.iter().zip(&pr_ref).filter(|(a, b)| (**a > 0.5) == (**b > 0.5)).count() as f32
        / got.len() as f32;
    eprintln!("[stream_gate] preds {:?} per-frame min_cos={cos:.6} (row {row}) bin-agree={agree:.4} maxabs={:.4e}",
        pr_shape, max_abs(&got, &pr_ref));

    assert!(agree >= 0.999, "streaming speaker agreement {agree}");
}

#[test]
fn sortformer_pipeline_smoke() {
    if !Path::new(SYN).exists() || !Path::new(&format!("{REF}/wav16.npy")).exists() {
        eprintln!("[pipeline] SKIP");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    use synaptix_diarization_sortformer::SortformerPipeline;
    let pipe = SortformerPipeline::from_syn(SYN, Device::Cpu, DType::F32).expect("from_syn");
    let (_s, wav) = load_f32(&format!("{REF}/wav16.npy"));
    let segs = pipe.diarize(&wav, 16000).expect("diarize");
    eprintln!("[pipeline] {} segments:", segs.len());
    for s in &segs {
        eprintln!("  spk{} {:.2}..{:.2}s conf={:.3}", s.speaker, s.start_s, s.end_s, s.confidence);
    }
    // extr.wav — один доминирующий голос → ≥1 сегмент спикера 0.
    assert!(!segs.is_empty(), "no segments");
    assert!(segs.iter().any(|s| s.speaker == 0), "no speaker 0");
}

#[test]
fn sortformer_encoder_stages() {
    if !Path::new(SYN).exists() || !Path::new(&format!("{REF}/preenc.npy")).exists() {
        eprintln!("[enc_stages] SKIP");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let w = SortformerWeights::open(SYN, Device::Cpu, DType::F32).expect("open");
    let model = SortformerModel::load(&w).expect("load");
    let (_s, wav) = load_f32(&format!("{REF}/wav16.npy"));
    let stages = model.encoder_debug(&wav).expect("encoder_debug");
    for (name, t) in &stages {
        if name == "final" {
            continue;
        }
        let (shape, refv) = load_f32(&format!("{REF}/{name}.npy"));
        assert_eq!(t.dims(), &shape[..], "{name} shape");
        let g = flat(t);
        // per-frame (inner=512) и per-channel (inner=T').
        let (cf, rf) = per_row_cos(&g, &refv, shape[2]); // frames
        let mc = mean_cos(&g, &refv, shape[2]);
        eprintln!("[enc_stages] {name:8} per-frame: min_cos={cf:.6} (frame {rf}) mean={mc:.6} maxabs={:.4e}", max_abs(&g, &refv));
    }
}

#[test]
fn sortformer_gate() {
    if !Path::new(SYN).exists() || !Path::new(&format!("{REF}/preds.npy")).exists() {
        eprintln!("[sortformer_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let w = SortformerWeights::open(SYN, Device::Cpu, DType::F32).expect("open weights");
    let model = SortformerModel::load(&w).expect("load model");

    let (wav_shape, wav) = load_f32(&format!("{REF}/wav16.npy"));
    eprintln!("[sortformer_gate] wav {:?}", wav_shape);

    let st = model.forward_stages(&wav).expect("forward_stages");

    // 1) mel (1,128,T) — cosine по оси времени (inner=T).
    let (mel_shape, mel_ref) = load_f32(&format!("{REF}/mel.npy"));
    assert_eq!(st.mel.dims(), &mel_shape[..], "mel shape");
    let (mel_cos, mel_row) = per_row_cos(&flat(&st.mel), &mel_ref, mel_shape[2]);
    eprintln!("[sortformer_gate] mel        cos={mel_cos:.6} row={mel_row} maxabs={:.4e}", max_abs(&flat(&st.mel), &mel_ref));

    // 2) encoder_out (1,512,T') — детальная сверка в `sortformer_encoder_stages`.
    let (enc_shape, _enc_ref) = load_f32(&format!("{REF}/encoder_out.npy"));
    assert_eq!(st.encoder_out.dims(), &enc_shape[..], "encoder shape");

    // 3) emb_seq (1,T',192) — cosine по каналам (inner=192).
    let (emb_shape, emb_ref) = load_f32(&format!("{REF}/emb_seq.npy"));
    assert_eq!(st.emb_seq.dims(), &emb_shape[..], "emb_seq shape");
    let (emb_cos, emb_row) = per_row_cos(&flat(&st.emb_seq), &emb_ref, emb_shape[2]);
    eprintln!("[sortformer_gate] emb_seq    cos={emb_cos:.6} row={emb_row}");

    // 4) trans (1,T',192).
    let (tr_shape, tr_ref) = load_f32(&format!("{REF}/trans.npy"));
    assert_eq!(st.trans.dims(), &tr_shape[..], "trans shape");
    let (tr_cos, tr_row) = per_row_cos(&flat(&st.trans), &tr_ref, tr_shape[2]);
    eprintln!("[sortformer_gate] trans      cos={tr_cos:.6} row={tr_row}");

    // 5) preds (1,T',4) — вероятности: per-row cosine + абсолютный диф.
    let (pr_shape, pr_ref) = load_f32(&format!("{REF}/preds.npy"));
    assert_eq!(st.preds.dims(), &pr_shape[..], "preds shape");
    let pr_got = flat(&st.preds);
    let (pr_cos, pr_row) = per_row_cos(&pr_got, &pr_ref, pr_shape[2]);
    let pr_max = max_abs(&pr_got, &pr_ref);
    // Бинаризованное согласие спикеров (функциональный гейт диаризации, thr 0.5).
    let agree = pr_got.iter().zip(&pr_ref).filter(|(a, b)| (**a > 0.5) == (**b > 0.5)).count();
    let agree_frac = agree as f32 / pr_got.len() as f32;
    eprintln!("[sortformer_gate] preds      cos={pr_cos:.6} row={pr_row} maxabs={pr_max:.4e} bin-agree={:.4}", agree_frac);

    assert!(mel_cos >= 0.9999, "mel cos {mel_cos} (row {mel_row})");
    assert!(emb_cos >= 0.995, "emb_seq cos {emb_cos} (row {emb_row})");
    assert!(tr_cos >= 0.999, "trans cos {tr_cos} (row {tr_row})");
    assert!(agree_frac >= 0.999, "speaker binarization agreement {agree_frac} (maxabs {pr_max})");
}
