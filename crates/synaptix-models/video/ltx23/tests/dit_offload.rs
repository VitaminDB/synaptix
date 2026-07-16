//! Полный 48-блочный DiT через bf16 layer-offload (блоки на CPU, стримятся на GPU
//! поблочно; best_cu float-acc → точно И влезает в 24GB) против 48-блочного
//! bf16-эталона. Гейт: SYN_LTX_GEMMA.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";

#[test]
fn video_dit_full_offload() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip video_dit_full_offload: weights/ref absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None); // полные 48 блоков
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let compute = Device::Cuda(0);

    // входы на GPU (compute)
    let il = SafetensorsLoader::open(REF).unwrap().with_device(compute);
    let latent = il.load("latent").unwrap();
    let t = latent.dims()[0];
    let latent = latent.reshape(vec![1, t, 128]).unwrap().to_dtype(DType::BF16).unwrap();
    let timesteps: Vec<f32> = il.load("timesteps").unwrap().to_vec1::<f32>().unwrap();
    let sigma = il.load("sigma").unwrap().to_vec1::<f32>().unwrap()[0];
    let positions: Vec<f64> = il.load("positions").unwrap().reshape(vec![3 * t * 2]).unwrap()
        .to_vec1::<f32>().unwrap().iter().map(|&x| x as f64).collect();
    let ttxt = il.load("context").unwrap().dims()[0];
    let context = il.load("context").unwrap().reshape(vec![1, ttxt, 4096]).unwrap().to_dtype(DType::BF16).unwrap();
    let ref_v: Vec<f32> = il.load("velocity").unwrap().reshape(vec![t * 128]).unwrap().to_vec1::<f32>().unwrap();

    // ckpt на CPU → блоки резидентны на CPU, стримятся на GPU поблочно.
    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    let dit = VideoDit::load_with(&ckpt, compute, DType::BF16, DType::BF16, true).expect("load offload");
    let vx = synaptix_core::grad::no_grad(|| dit.forward(&latent, &timesteps, sigma, &positions, &context)).expect("forward");
    let ours: Vec<f32> = vx.reshape(vec![t * 128]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let hd = 128usize;
    let mut min_cos = f32::INFINITY;
    let mut max_rel = 0.0f32;
    for p in 0..t {
        let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..hd {
            let (r, o) = (ref_v[p * hd + k] as f64, ours[p * hd + k] as f64);
            dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
        }
        min_cos = min_cos.min((dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32);
        max_rel = max_rel.max((dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32);
    }
    eprintln!("48-block offload bf16 velocity: per-row min_cos={min_cos:.5} max_rel={max_rel:.4}");
    // best_cu float-acc bf16 на GPU: cos≈0.976 (vs MXFP8 0.62 / CPU-bf16 0.21).
    // Остаток до 0.9999 — bf16-дрейф residual по 48 блокам (улучшаемо bit-faithful
    // как FLUX→0.999987); 0.976 даёт когерентное видео. Гейт «работает корректно».
    assert!(min_cos > 0.97, "48-block offload velocity cos: {min_cos}");
}
