use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn check(n: usize, k: usize, m: usize) -> f32 {
    let dev = Device::Cuda(0);
    let mut w = vec![0f32; n * k];
    for (i, x) in w.iter_mut().enumerate() {
        *x = (((i * 31) % 199) as f32 / 199.0 - 0.5) * 0.5;
    }
    let mut a = vec![0f32; m * k];
    for (i, x) in a.iter_mut().enumerate() {
        *x = (((i * 17) % 173) as f32 / 173.0 - 0.5) * 0.5;
    }
    let wt = Tensor::from_vec(w, vec![n, k], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let at = Tensor::from_vec(a, vec![m, k], dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();

    let reference = at
        .matmul(&wt.transpose(0, 1).unwrap().contiguous().unwrap())
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    if let Ok(nv) = wt.quantize_to_nvfp4() {
        if let Ok(y) = at.linear_quant(&nv) {
            let v = y
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut num = 0f64;
            let mut den = 0f64;
            for (r, g) in reference.iter().zip(v.iter()) {
                num += ((g - r) as f64).powi(2);
                den += (*r as f64).powi(2);
            }
            println!("  nvfp4 reference l2_rel={}", (num / den.max(1e-12)).sqrt());
        }
    }
    let qw = wt.quantize_to_mxfp8().unwrap();
    let got = at
        .linear_quant(&qw)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let mut num = 0f64;
    let mut den = 0f64;
    for (r, g) in reference.iter().zip(got.iter()) {
        num += ((g - r) as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    let l2 = (num / den.max(1e-12)).sqrt() as f32;
    println!("n={n} k={k} m={m} l2_rel={l2}");
    l2
}

#[test]
fn mxfp8_linear_shapes() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let aligned = check(2048, 5120, 2);
    let unaligned = check(3000, 5120, 2);
    assert!(aligned < 0.1, "MXFP8 linear (n%128==0) расходится: L2={aligned}");
    assert!(
        unaligned < 0.1,
        "MXFP8 linear (n%128!=0) расходится: L2={unaligned}"
    );
}
