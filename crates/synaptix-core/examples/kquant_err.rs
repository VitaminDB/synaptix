//! Ошибка K-квантов на настоящей матрице: kquant_err <model.safetensors> <имя> [calib]
use synaptix_core::quant::ggml::{quantize_q4_k, quantize_q5_k, GgmlType};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&a[1]).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor(&a[2]).unwrap();
    let (n, k) = (t.shape()[0], t.shape()[1]);
    let x: Vec<f32> = t.data().chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
    let imp: Option<Vec<f32>> = a.get(3).map(|p| synaptix_core::quant::calib::load(std::path::Path::new(p)).unwrap()[&a[2]].clone());
    let h = imp.clone().unwrap_or(vec![1.0; k]);
    let err = |t: GgmlType, q: &[u8]| {
        let mut y = vec![0f32; n * k];
        synaptix_core::quant::ggml_dequant::dequantize(t, q, n * k, &mut y).unwrap();
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..n * k {
            let w = h[i % k] as f64;
            num += w * ((x[i] - y[i]) as f64).powi(2);
            den += w * (x[i] as f64).powi(2);
        }
        (num / den).sqrt()
    };
    for (name, f, t) in [("q4_k", quantize_q4_k as fn(&[f32], usize, usize, Option<&[f32]>) -> Vec<u8>, GgmlType::Q4K), ("q5_k", quantize_q5_k, GgmlType::Q5K)] {
        let e0 = err(t, &f(&x, n, k, None));
        let e1 = err(t, &f(&x, n, k, imp.as_deref()));
        println!("{name}: взвешенная отн. ошибка без калибровки {e0:.4}, с калибровкой {e1:.4}");
    }
}
