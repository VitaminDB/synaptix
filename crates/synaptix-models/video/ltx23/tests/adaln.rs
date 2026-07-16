//! Изоляция adaLN-single (timestep sinusoidal + embedder + linear) против Python.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/adaln_ref.safetensors";

fn cmp(name: &str, ours: &[f32], refv: &[f32]) {
    let (mut dot, mut nr, mut no, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..refv.len() {
        let (r, o) = (refv[i] as f64, ours[i] as f64);
        dot += r * o; nr += r * r; no += o * o; mx = mx.max((r - o).abs());
    }
    let cos = dot / (nr.sqrt() * no.sqrt() + 1e-12);
    eprintln!("{name}: cos={cos:.6} max|Δ|={mx:.3e}");
    assert!(cos > 0.9999, "{name} cos {cos}");
    assert!(mx < 5e-2, "{name} max|Δ| {mx}");
}

#[test]
fn adaln_matches_python() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip adaln_matches_python: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::F32).unwrap();
    let (modul, emb, pmod) =
        VideoDit::_adaln_for_test(&ckpt, dev, DType::F32, 0.7 * 1000.0).expect("adaln");

    let rl = SafetensorsLoader::open(REF).unwrap();
    let mr: Vec<f32> = rl.load("modul").unwrap().to_vec1::<f32>().unwrap();
    let er: Vec<f32> = rl.load("emb").unwrap().to_vec1::<f32>().unwrap();
    let pr: Vec<f32> = rl.load("pmod").unwrap().to_vec1::<f32>().unwrap();

    let mo: Vec<f32> = modul.reshape(vec![mr.len()]).unwrap().to_vec1::<f32>().unwrap();
    let eo: Vec<f32> = emb.reshape(vec![er.len()]).unwrap().to_vec1::<f32>().unwrap();
    let po: Vec<f32> = pmod.reshape(vec![pr.len()]).unwrap().to_vec1::<f32>().unwrap();

    cmp("emb", &eo, &er);
    cmp("modul", &mo, &mr);
    cmp("pmod", &po, &pr);
}
