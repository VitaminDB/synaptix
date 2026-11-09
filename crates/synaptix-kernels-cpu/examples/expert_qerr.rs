//! Ошибка квантователей на настоящих экспертах из FP8-чекпойнта (блочные
//! масштабы 128×128, `weight_scale_inv`):
//!   expert_qerr <каталог HF> <слой> <эксперт>...
//! Для каждой матрицы (gate, up, down) печатает относительную ошибку
//! ‖W − Q(W)‖ / ‖W‖ и бит на вес у каждого формата.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use synaptix_core::quant::ggml::{quantize_q4_k, GgmlType};

fn e4m3(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else if e == 15 && (b & 7) == 7 {
        f32::NAN
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

struct Ckpt {
    dir: std::path::PathBuf,
    map: HashMap<String, String>,
}

impl Ckpt {
    fn open(dir: &str) -> Self {
        let idx: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/model.safetensors.index.json")).unwrap()).unwrap();
        let map = idx["weight_map"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();
        Self { dir: dir.into(), map }
    }

    fn raw(&self, name: &str) -> (String, Vec<usize>, Vec<u8>) {
        let file = self.dir.join(&self.map[name]);
        let mut f = std::fs::File::open(file).unwrap();
        let mut l = [0u8; 8];
        f.read_exact(&mut l).unwrap();
        let n = u64::from_le_bytes(l) as usize;
        let mut h = vec![0u8; n];
        f.read_exact(&mut h).unwrap();
        let h: serde_json::Value = serde_json::from_slice(&h).unwrap();
        let s = &h[name];
        let off = s["data_offsets"].as_array().unwrap();
        let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
        f.seek(SeekFrom::Start(8 + n as u64 + a)).unwrap();
        let mut buf = vec![0u8; (b - a) as usize];
        f.read_exact(&mut buf).unwrap();
        let shape = s["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        (s["dtype"].as_str().unwrap().to_string(), shape, buf)
    }

    /// FP8 + блочный `weight_scale_inv` → f32 `[n, k]`.
    fn weight(&self, name: &str) -> (Vec<f32>, usize, usize) {
        let (dt, shape, w) = self.raw(&format!("{name}.weight"));
        assert_eq!(dt, "F8_E4M3", "{name}: {dt}");
        let (n, k) = (shape[0], shape[1]);
        let (sdt, sshape, s) = self.raw(&format!("{name}.weight_scale_inv"));
        let scales: Vec<f32> = match sdt.as_str() {
            "F32" => bytemuck::cast_slice::<u8, f32>(&s).to_vec(),
            "BF16" => s.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            o => panic!("масштабы {o}"),
        };
        let kb = sshape[1];
        let lut: Vec<f32> = (0..256).map(|i| e4m3(i as u8)).collect();
        let mut out = vec![0f32; n * k];
        for r in 0..n {
            for c in 0..k {
                out[r * k + c] = lut[w[r * k + c] as usize] * scales[(r / 128) * kb + c / 128];
            }
        }
        (out, n, k)
    }
}

fn rel(x: &[f32], y: &[f32]) -> f64 {
    let (mut a, mut b) = (0f64, 0f64);
    for (p, q) in x.iter().zip(y) {
        a += ((p - q) as f64).powi(2);
        b += (*p as f64).powi(2);
    }
    (a / b).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ck = Ckpt::open(&a[1]);
    let layer: usize = a[2].parse().unwrap();
    for e in &a[3..] {
        for m in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("model.language_model.layers.{layer}.mlp.experts.{e}.{m}");
            let (x, n, k) = ck.weight(&name);
            let mut line = format!("L{layer} e{e} {m:<9} [{n}×{k}]");
            // NVFP4 (как в нынешнем бандле), 4,5 бит
            let (p, s) = synaptix_kernels_cpu::quant::encode::quantize_nvfp4_rows(&x, n, k).unwrap();
            let mut y = vec![0f32; n * k];
            for r in 0..n {
                synaptix_kernels_cpu::quant::dequant::nvfp4_dequant_row(&p[r * k / 2..(r + 1) * k / 2], &s, r, k, &mut y[r * k..(r + 1) * k]);
            }
            line += &format!("  nvfp4(4.50) {:.4}", rel(&x, &y));
            if k % 256 == 0 {
                let q = quantize_q4_k(&x, n, k, None);
                synaptix_core::quant::ggml_dequant::dequantize(GgmlType::Q4K, &q, n * k, &mut y).unwrap();
                line += &format!("  q4_k(4.50) {:.4}", rel(&x, &y));
            } else {
                line += "  q4_k(K%256)  —   ";
            }
            {
                let t = std::time::Instant::now();
                let q = synaptix_core::quant::ggml::quantize_iq3_s128(&x, n, k, None);
                synaptix_core::quant::ggml_dequant::dequantize(GgmlType::Iq3S128, &q, n * k, &mut y).unwrap();
                line += &format!("  iq3_s128(3.50) {:.4} [{:.1} с]", rel(&x, &y), t.elapsed().as_secs_f64());
            }
            for bits in [4u8, 3, 2] {
                let q = synaptix_core::quant::sq::quantize_matrix(bits, &x, n, k).unwrap();
                let yq = synaptix_core::quant::sq::dequantize_matrix(bits, &q, n, k).unwrap();
                line += &format!("  sq{bits}({:.2}) {:.4}", bits as f32 + 0.625, rel(&x, &yq));
            }
            println!("{line}");
            if let Ok(dir) = std::env::var("EXPERT_DUMP") {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(format!("{dir}/L{layer}_e{e}_{m}_{n}x{k}.f32"), bytemuck::cast_slice(&x)).unwrap();
            }
        }
    }
}
