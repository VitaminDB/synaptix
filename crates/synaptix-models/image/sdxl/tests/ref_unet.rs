//! Bit-exact-проверка SDXL UNet (UNet2DConditionModel) против diffusers
//! (CPU F32). Reference: `scripts/reference/gen_sdxl_unet.py`.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::unet::{UNet2DConditionConfig, UNet2DConditionModel};

const SDXL: &str = "models/stabilityai/stable-diffusion-xl-base-1.0";

fn cos_and_maxerr(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut maxerr) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxerr = maxerr.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr)
}

#[test]
fn unet_forward_bit_exact() {
    synaptix_kernels_cpu::ensure_registered();
    let weights = format!("{SDXL}/unet/diffusion_pytorch_model.fp16.safetensors");
    if !Path::new(&weights).exists() {
        eprintln!("SKIP unet: нет {weights}");
        return;
    }
    let ref_path = synaptix_test_utils::reference_data_path("sdxl_unet", "forward.safetensors");
    if !ref_path.exists() {
        eprintln!("SKIP unet: нет reference {ref_path:?} (запусти gen_sdxl_unet.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let ld = SafetensorsLoader::open(&weights).unwrap().with_device(Device::Cpu);
    let get = |name: &str| {
        ld.load_to(name, Device::Cpu, DType::F32)
            .map_err(|e| synaptix_core::error::SynaptixError::Other(format!("load {name}: {e}")))
    };
    let unet = UNet2DConditionModel::load(&UNet2DConditionConfig::sdxl(), &get).unwrap();

    let f = |k: &str| refs[k].to_dtype(DType::F32).unwrap();
    let out = unet
        .forward(
            &f("sample"),
            &f("timestep"),
            &f("encoder_hidden_states"),
            &f("text_embeds"),
            &f("time_ids"),
        )
        .unwrap();
    assert_eq!(out.dims(), refs["out"].dims());

    let (cos, err) = cos_and_maxerr(&out, &refs["out"]);
    eprintln!("unet forward: cos={cos:.8} max_err={err:.3e} dims={:?}", out.dims());
    assert!(cos > 0.9999, "unet forward cos {cos} < 0.9999 (max_err {err:.3e})");
}
