//! Сверка оригинала HF с бандлом: `origin_check <file.syn> <каталог HF> <тензор бандла> <срез>...`
//! — косинус и относительная ошибка между срезом из оригинала (`HfOrigin`) и
//! деквантованным срезом бандла. Проверяет сопоставление имён и раскладку
//! (gate|up подряд, без транспонирования).
use synaptix_core::{device::Device, dtype::DType};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix::init().expect("init");
    let loader = synaptix_io::weights::syn_bundle::SynBundleLoader::open(std::path::Path::new(&a[1])).expect("бандл");
    let origin = synaptix_io::weights::hf_origin::HfOrigin::open(std::path::Path::new(&a[2])).expect("оригинал");
    let name = &a[3];
    let (_, n, k) = loader.quant_dims(name).expect("не квантован");
    for s in &a[4..] {
        let slice: usize = s.parse().unwrap();
        let o = origin.bundle_slice(name, slice, n, k).expect("нет в оригинале").expect("чтение");
        let q = loader.load_quant_expert(name, slice, Device::Cpu).expect("нет среза").expect("срез");
        let b: Vec<f32> = q
            .dequantize(DType::F16)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("деквант");
        let (mut dot, mut na, mut nb, mut d2) = (0f64, 0f64, 0f64, 0f64);
        for (x, y) in o.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
            d2 += ((x - y) as f64).powi(2);
        }
        println!("{name}[{slice}] [{n}×{k}]: cos {:.5}, отн. ошибка {:.4}", dot / (na * nb).sqrt(), (d2 / na).sqrt());
    }
}
