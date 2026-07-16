//! conv1d при batch>1 + stride>1 (случай DiT proj_in: K=2, stride=2, b=2).
use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_ops::conv::conv1d;

fn ref_conv1d(x: &[f32], b: usize, cin: usize, l: usize, w: &[f32], cout: usize, k: usize, stride: usize, pad: usize) -> Vec<f32> {
    let lp = l + 2 * pad;
    let out_len = (lp - k) / stride + 1;
    let mut out = vec![0f32; b * cout * out_len];
    let at = |bi: usize, ci: usize, p: usize| -> f32 {
        if p < pad || p >= pad + l { 0.0 } else { x[(bi * cin + ci) * l + (p - pad)] }
    };
    for bi in 0..b {
        for co in 0..cout {
            for o in 0..out_len {
                let mut acc = 0f32;
                for ci in 0..cin {
                    for ki in 0..k {
                        acc += at(bi, ci, ki + o * stride) * w[(co * cin + ci) * k + ki];
                    }
                }
                out[(bi * cout + co) * out_len + o] = acc;
            }
        }
    }
    out
}

#[test]
fn conv1d_b2_k2_stride2() {
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    let (b, cin, l, cout, k, stride, pad) = (2usize, 3usize, 9usize, 4usize, 2usize, 2usize, 0usize);
    let xd: Vec<f32> = (0..b * cin * l).map(|i| (i as f32 * 0.13).sin()).collect();
    let wd: Vec<f32> = (0..cout * cin * k).map(|i| (i as f32 * 0.07).cos()).collect();
    let x = Tensor::from_vec(xd.clone(), vec![b, cin, l], dev).unwrap();
    let w = Tensor::from_vec(wd.clone(), vec![cout, cin, k], dev).unwrap();
    let got = conv1d(&x, &w, None, stride, pad).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let want = ref_conv1d(&xd, b, cin, l, &wd, cout, k, stride, pad);
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let mut maxd = 0f32;
    for i in 0..got.len() {
        assert!(got[i].is_finite(), "NaN/inf at {i}: {}", got[i]);
        maxd = maxd.max((got[i] - want[i]).abs());
    }
    eprintln!("conv1d b2 k2 s2 CPU: maxdiff={maxd:.2e} (len={})", got.len());
    assert!(maxd < 1e-4, "maxdiff {maxd} too large");
}

/// нативная dilation БИТ-В-БИТ к dilate_weight (нули в сумме f32 ничего не меняют).
fn dilate_weight(w: &Tensor, dilation: usize) -> Tensor {
    if dilation <= 1 {
        return w.clone();
    }
    let d = w.dims().to_vec();
    let (c, cin, kk) = (d[0], d[1], d[2]);
    let gap = Tensor::zeros(vec![c, cin, dilation - 1], w.dtype(), w.device()).unwrap();
    let mut parts: Vec<Tensor> = Vec::new();
    for ki in 0..kk {
        parts.push(w.narrow(2, ki, 1).unwrap().contiguous().unwrap());
        if ki + 1 < kk {
            parts.push(gap.clone());
        }
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2).unwrap()
}

#[test]
fn conv1d_native_dilation_bit_identical() {
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    let (b, cin, l, cout, k) = (1usize, 64usize, 240usize, 64usize, 7usize);
    for &dil in &[1usize, 3, 9] {
        let pad = 3 * dil;
        let xd: Vec<f32> = (0..b * cin * l).map(|i| ((i % 31) as f32 * 0.13).sin()).collect();
        let wd: Vec<f32> = (0..cout * cin * k).map(|i| ((i % 29) as f32 * 0.07).cos()).collect();
        let x = Tensor::from_vec(xd, vec![b, cin, l], dev).unwrap();
        let w = Tensor::from_vec(wd, vec![cout, cin, k], dev).unwrap();
        // native dilation
        let native = synaptix_ops::conv::conv1d_dilated(&x, &w, None, 1, pad, dil)
            .unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // эталон: dilate_weight + plain conv1d
        let wdil = dilate_weight(&w, dil);
        let refr = conv1d(&x, &wdil, None, 1, pad)
            .unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(native.len(), refr.len());
        let md = native.iter().zip(&refr).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("dilation={dil}: native-vs-dilate_weight maxdiff={md:.3e}");
        assert_eq!(md, 0.0, "dilation={dil} НЕ bit-identical: maxdiff={md}");
    }
}
