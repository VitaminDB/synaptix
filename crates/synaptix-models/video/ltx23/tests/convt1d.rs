//! Валидация conv_transpose1d (новый примитив для вокодера) против torch.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_ops::conv::conv_transpose1d::conv_transpose1d;

const REF: &str = "tests/reference_data/ltx_gemma/convt1d_ref.safetensors";

fn flat(t: &synaptix_core::tensor::Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
}

fn cmp(name: &str, ours: &synaptix_core::tensor::Tensor, refv: &[f32]) {
    let o = flat(ours);
    let mut mx = 0.0f64;
    for i in 0..refv.len() {
        mx = mx.max((refv[i] as f64 - o[i] as f64).abs());
    }
    eprintln!("{name}: max|Δ|={mx:.3e} (n={})", refv.len());
    assert!(mx < 1e-4, "{name} max|Δ|={mx}");
}

#[test]
fn conv_transpose1d_matches_torch() {
    if !Path::new(REF).exists() {
        eprintln!("skip: ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let g = |n: &str| rl.load(n).unwrap();
    let d3 = |t: synaptix_core::tensor::Tensor| {
        let d = t.dims().to_vec();
        if d.len() == 3 { t } else { t.reshape(vec![1, d[0], d[1]]).unwrap() } // [C,..]→[1,C,..]
    };

    // case1: groups=1, stride2, pad1
    let y1 = conv_transpose1d(&d3(g("x1")), &g("w1"), Some(&g("b1")), 2, 1, 0, 1, 1).unwrap();
    cmp("groups=1", &y1, &flat(&g("y1")));

    // case2: depthwise groups=5, stride3, pad0
    let y2 = conv_transpose1d(&d3(g("x2")), &g("w2"), Some(&g("b2")), 3, 0, 0, 5, 1).unwrap();
    cmp("depthwise", &y2, &flat(&g("y2")));
}
