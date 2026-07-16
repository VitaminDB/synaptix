//! GPU-валидация Sortformer: forward на CUDA (F32/F16) → бинаризованное согласие
//! спикеров vs NeMo-дамп (preds.npy) + тайминг полного diarize.
//!
//! cargo run --profile fast-release -p synaptix-diarization-sortformer --features cuda --example gpu_run

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_diarization_sortformer::model::SortformerModel;
use synaptix_diarization_sortformer::{SortformerPipeline, SortformerWeights};

const SYN: &str = "storage/syn_models/sortformer-streaming-4spk-v21.syn";
const REF: &str = "tmp/sortformer_ref";

fn load_npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let major = b[6];
    let (hl, off) = if major == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hdr = std::str::from_utf8(&b[off..off + hl]).unwrap();
    let shape: Vec<usize> = {
        let i = hdr.find("'shape'").unwrap();
        let r = &hdr[i..];
        let lp = r.find('(').unwrap();
        let rp = r.find(')').unwrap();
        r[lp + 1..rp].split(',').filter_map(|s| s.trim().parse().ok()).collect()
    };
    let n: usize = shape.iter().product();
    let data = &b[off + hl..];
    let v = (0..n).map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap())).collect();
    (shape, v)
}

fn bin_agree(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let ok = (0..n).filter(|&i| (a[i] > 0.5) == (b[i] > 0.5)).count();
    ok as f32 / n as f32
}

fn run(label: &str, dtype: DType, wav: &[f32], ref_preds: &[f32]) {
    let w = SortformerWeights::open(SYN, Device::Cuda(0), dtype).expect("open cuda");
    let model = SortformerModel::load(&w).expect("load cuda");
    let t = Instant::now();
    let st = model.forward_stages(wav).expect("forward");
    let dt = t.elapsed();
    let preds = st.preds.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let agree = bin_agree(&preds, ref_preds);
    eprintln!("[gpu] {label:10} forward {dt:?}  bin-agree(vs NeMo)={agree:.4}  preds.len={}", preds.len());
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let (_ws, wav) = load_npy_f32(&format!("{REF}/wav16.npy"));
    let (_ps, ref_preds) = load_npy_f32(&format!("{REF}/preds.npy"));
    eprintln!("[gpu] wav {} samples ({:.1}s)", wav.len(), wav.len() as f32 / 16000.0);

    run("CUDA-F32", DType::F32, &wav, &ref_preds);
    run("CUDA-F16", DType::F16, &wav, &ref_preds);

    // streaming-режим на GPU (длинное аудио, compress спик-кэша) vs NeMo streaming-дамп.
    if std::path::Path::new(&format!("{REF}/stream_preds.npy")).exists() {
        let (_ls, wl) = load_npy_f32(&format!("{REF}/wav16_long.npy"));
        let (_ss, sref) = load_npy_f32(&format!("{REF}/stream_preds.npy"));
        for (label, dt) in [("CUDA-F32", DType::F32), ("CUDA-F16", DType::F16)] {
            let w = SortformerWeights::open(SYN, Device::Cuda(0), dt).expect("open");
            let model = SortformerModel::load(&w).expect("load");
            let t = Instant::now();
            let preds = model.diarize_pcm_streaming(&wl).expect("stream");
            let dt2 = t.elapsed();
            let p = preds.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            eprintln!("[gpu] stream {label} {dt2:?} ({:.1}s аудио) bin-agree(vs NeMo)={:.4}",
                wl.len() as f32 / 16000.0, bin_agree(&p, &sref));
        }
    }

    // полный pipeline тайминг (streaming) на GPU F16.
    let pipe = SortformerPipeline::from_syn(SYN, Device::Cuda(0), DType::F16).expect("pipe");
    let t = Instant::now();
    let segs = pipe.diarize(&wav, 16000).expect("diarize");
    eprintln!("[gpu] pipeline F16 diarize {:?} → {} сегментов:", t.elapsed(), segs.len());
    for s in &segs {
        eprintln!("       spk{} {:.2}..{:.2}s conf={:.3}", s.speaker, s.start_s, s.end_s, s.confidence);
    }
}
