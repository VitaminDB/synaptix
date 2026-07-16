//! Host-stream квант-блоков (квантуем 1× → CPU → стрим на GPU) против
//! квант-резидентного пути: байты весов не меняются → выход BIT-IDENTICAL.
//! Гейт: SYN_LTX_GEMMA (тяжёлые веса). Усечение SYN_LTX_DIT_NBLOCKS=4.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";

fn run_pair(quant: DType) {
    let compute = Device::Cuda(0);
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

    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    let resident = VideoDit::load_with(&ckpt, compute, DType::BF16, quant, false).expect("load resident");
    let y_res = synaptix_core::grad::no_grad(|| resident.forward(&latent, &timesteps, sigma, &positions, &context))
        .expect("forward resident");
    let v_res: Vec<f32> = y_res.reshape(vec![t * 128]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    drop(resident);

    let streamed = VideoDit::load_with(&ckpt, compute, DType::BF16, quant, true).expect("load host-stream");
    let y_str = synaptix_core::grad::no_grad(|| streamed.forward(&latent, &timesteps, sigma, &positions, &context))
        .expect("forward host-stream");
    let v_str: Vec<f32> = y_str.reshape(vec![t * 128]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    // per-row max-abs гейт (не cos!): bit-identical → строго 0
    let hd = 128usize;
    let mut max_abs = 0.0f32;
    let mut bad_rows = 0usize;
    for p in 0..t {
        let mut row = 0.0f32;
        for k in 0..hd {
            row = row.max((v_res[p * hd + k] - v_str[p * hd + k]).abs());
        }
        if row > 0.0 {
            bad_rows += 1;
        }
        max_abs = max_abs.max(row);
    }
    assert!(
        max_abs == 0.0,
        "{quant:?}: host-stream != resident: per-row max_abs={max_abs} bad_rows={bad_rows}/{t}"
    );
    eprintln!("{quant:?}: host-stream == resident BIT-EXACT ({t} строк)");
}

#[test]
fn video_dit_quant_host_stream_bit_exact() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip video_dit_quant_host_stream_bit_exact: weights/ref absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(Some(4));
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    run_pair(DType::MXFP8);
    run_pair(DType::NVFP4);
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None);
}
