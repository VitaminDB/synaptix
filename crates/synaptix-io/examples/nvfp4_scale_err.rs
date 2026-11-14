//! Ошибка NVFP4 на настоящих весах экспертов: штатный кодировщик (масштаб
//! блока `amax/6` прямо в E4M3) против NVFP4 с общим FP32-масштабом матрицы
//! (`g = amax/(6·448)`, масштабы блоков — в нормальной зоне E4M3, как у
//! NVIDIA). Usage: `nvfp4_scale_err <shard.safetensors> <тензор стопки экспертов> [экспертов]`.

use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_kernels_cpu::quant::encode::encode_e2m1_rtne;
use synaptix_kernels_cpu::quant::fp8_e4m3::{decode_e4m3, encode_e4m3};

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

/// Деквант NVFP4 матрицы с общим масштабом `g` (1.0 — штатный формат).
fn nvfp4(w: &[f32], g: f32) -> (Vec<f32>, [usize; 16]) {
    let mut out = vec![0f32; w.len()];
    let mut exps = [0usize; 16];
    for (blk, o) in w.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let sb = encode_e4m3(if amax > 0.0 { amax / 6.0 / g } else { 1e-9 });
        exps[((sb >> 3) & 15) as usize] += 1;
        let mut s = decode_e4m3(sb) * g;
        if s == 0.0 {
            s = 1e-12;
        }
        for (v, q) in blk.iter().zip(o.iter_mut()) {
            let c = encode_e2m1_rtne(v / s);
            *q = E2M1[(c & 7) as usize] * s * if c & 8 != 0 { -1.0 } else { 1.0 };
        }
    }
    (out, exps)
}

/// Штатный формат (без общего масштаба), но код масштаба блока выбирается
/// перебором соседних кодов E4M3 по минимуму ошибки блока.
fn nvfp4_search(w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for (blk, o) in w.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let base = encode_e4m3(if amax > 0.0 { amax / 6.0 } else { 1e-9 }) as i32;
        let mut best = (f64::MAX, [0f32; 16]);
        for c in (base - 2).max(1)..=(base + 2).min(0x7e) {
            let s = decode_e4m3(c as u8);
            let mut q = [0f32; 16];
            let mut e = 0f64;
            for (i, v) in blk.iter().enumerate() {
                let code = encode_e2m1_rtne(v / s);
                q[i] = E2M1[(code & 7) as usize] * s * if code & 8 != 0 { -1.0 } else { 1.0 };
                e += ((v - q[i]) as f64).powi(2);
            }
            if e < best.0 {
                best = (e, q);
            }
        }
        o.copy_from_slice(&best.1);
    }
    out
}

fn rel(w: &[f32], q: &[f32]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (a, b) in w.iter().zip(q) {
        e += ((a - b) as f64).powi(2);
        n += (*a as f64).powi(2);
    }
    (e / n).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let ld = SafetensorsLoader::open(&a[1])?;
    let (bytes, _dt, shape) = ld.raw_bytes(&a[2]).ok_or_else(|| format!("{}: нет тензора", a[2]))?;
    let (ne, n, k) = (shape[0], shape[1], shape[2]);
    let take: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8).min(ne);
    println!("{} {:?}: экспертов {take} из {ne}", a[2], shape);
    let per = n * k * 2;
    let (mut s0, mut s1, mut s2) = (0f64, 0f64, 0f64);
    let mut ex0 = [0usize; 16];
    let mut ex1 = [0usize; 16];
    for e in (0..ne).step_by((ne / take).max(1)).take(take) {
        let w = bf16(&bytes[e * per..(e + 1) * per]);
        let amax = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (q0, x0) = nvfp4(&w, 1.0);
        let (q1, x1) = nvfp4(&w, amax / (6.0 * 448.0));
        let (r0, r1) = (rel(&w, &q0), rel(&w, &q1));
        let r2 = rel(&w, &nvfp4_search(&w));
        s0 += r0;
        s1 += r1;
        s2 += r2;
        for i in 0..16 {
            ex0[i] += x0[i];
            ex1[i] += x1[i];
        }
        println!("  эксперт {e:3}: amax {amax:.4}  штатный {:.2}%  перебор масштаба {:.2}%  с общим масштабом {:.2}%", 100.0 * r0, 100.0 * r2, 100.0 * r1);
    }
    let t = take as f64;
    println!("среднее: штатный {:.2}%, перебор масштаба {:.2}%, с общим масштабом {:.2}%", 100.0 * s0 / t, 100.0 * s2 / t, 100.0 * s1 / t);
    let tot: usize = ex0.iter().sum();
    let fmt = |x: &[usize; 16]| (0..16).filter(|i| x[*i] > 0).map(|i| format!("e{i}:{:.0}%", 100.0 * x[i] as f64 / tot as f64)).collect::<Vec<_>>().join(" ");
    println!("экспоненты масштабов: штатный [{}], с общим [{}]", fmt(&ex0), fmt(&ex1));
    Ok(())
}
