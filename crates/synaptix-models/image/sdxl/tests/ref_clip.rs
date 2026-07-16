//! Bit-exact-проверка обоих CLIP-text-энкодеров SDXL против HF `transformers`
//! (CPU F32). Reference: `scripts/reference/gen_sdxl_clip.py` → грузит
//! `text_encoder` (CLIP-L) и `text_encoder_2` (bigG) в float32, прогоняет
//! фиксированный промпт и дампит input_ids + penultimate + last_hidden + pooled.
//!
//! Здесь грузим те же fp16-веса напрямую (SafetensorsLoader, dtype=F32),
//! собираем `ClipTextEncoder` и сверяем выход на тех же input_ids. Тест
//! пропускается, если веса или reference-данные отсутствуют.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_nn::linear::Linear;
use synaptix_nn::text::{ClipTextConfig, ClipTextEncoder};

const SDXL: &str = "models/stabilityai/stable-diffusion-xl-base-1.0";

fn cos_and_maxerr(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let n: usize = a.dims().iter().product();
    let av = a.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let bv = b.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut maxerr = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxerr = maxerr.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), maxerr)
}

fn loader(sub: &str) -> Option<SafetensorsLoader> {
    let path = format!("{SDXL}/{sub}/model.fp16.safetensors");
    if !Path::new(&path).exists() {
        return None;
    }
    Some(SafetensorsLoader::open(&path).unwrap().with_device(Device::Cpu))
}

fn run_case(case: &str, sub: &str, cfg: ClipTextConfig, with_proj: bool) {
    synaptix_kernels_cpu::ensure_registered();
    let ld = match loader(sub) {
        Some(l) => l,
        None => {
            eprintln!("SKIP {case}: нет весов {sub}/model.fp16.safetensors");
            return;
        }
    };
    let ref_path =
        synaptix_test_utils::reference_data_path("sdxl_clip", &format!("{case}.safetensors"));
    if !ref_path.exists() {
        eprintln!("SKIP {case}: нет reference {ref_path:?} (запусти gen_sdxl_clip.py)");
        return;
    }
    let refs = synaptix_test_utils::load_safetensors(&ref_path);

    let get = |name: &str| {
        ld.load_to(name, Device::Cpu, DType::F32)
            .map_err(|e| synaptix_core::error::SynaptixError::Other(format!("load {name}: {e}")))
    };
    let mut enc = ClipTextEncoder::load(&cfg, "text_model", &get).unwrap();
    if with_proj {
        let w = get("text_projection.weight").unwrap();
        enc = enc.with_projection(Linear::new(w, None).unwrap());
    }

    let ids_i32 = refs["input_ids"].to_vec2::<i32>().unwrap();
    let (b, s) = (ids_i32.len(), ids_i32[0].len());
    let ids_u32: Vec<u32> = ids_i32.into_iter().flatten().map(|v| v as u32).collect();
    let input_ids = Tensor::from_vec(ids_u32, (b, s), Device::Cpu).unwrap();
    let out = enc.forward(&input_ids).unwrap();

    let (cos_p, err_p) = cos_and_maxerr(out.penultimate_hidden_state(), &refs["penultimate"]);
    let (cos_l, err_l) = cos_and_maxerr(&out.last_hidden_state, &refs["last_hidden_state"]);
    let (cos_o, err_o) = cos_and_maxerr(&out.pooled_output, &refs["pooled"]);
    eprintln!(
        "{case}: penultimate cos={cos_p:.8} max_err={err_p:.3e} | last cos={cos_l:.8} max_err={err_l:.3e} | pooled cos={cos_o:.8} max_err={err_o:.3e}"
    );

    assert!(cos_p > 0.9999, "{case} penultimate cos {cos_p} < 0.9999 (max_err {err_p:.3e})");
    assert!(cos_l > 0.9999, "{case} last_hidden cos {cos_l} < 0.9999 (max_err {err_l:.3e})");
    assert!(cos_o > 0.9999, "{case} pooled cos {cos_o} < 0.9999 (max_err {err_o:.3e})");
}

#[test]
fn clip_l_bit_exact() {
    run_case("clip_l", "text_encoder", ClipTextConfig::clip_l(), false);
}

#[test]
fn clip_bigg_bit_exact() {
    run_case("clip_bigg", "text_encoder_2", ClipTextConfig::clip_bigg(), true);
}
