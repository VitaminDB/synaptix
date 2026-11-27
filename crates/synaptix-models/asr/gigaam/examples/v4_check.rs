//! Сверка GigaAM-v4 с дампом эталона (`scripts/reference/gigaam4_ref.py`):
//! `v4_check МОДЕЛЬ(.syn | каталог HF) КАТАЛОГ_ДАМПОВ [cuda]` — по каждой
//! контрольной точке макс. абс. разность и косинус, затем тексты обеих голов.
use synaptix_asr_gigaam::GigaAm4;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

/// `.npy` (f32/i64, little-endian, C-порядок) → (форма, числа как f32).
fn npy(path: &str) -> Option<(Vec<usize>, Vec<f32>)> {
    let b = std::fs::read(path).ok()?;
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let head = std::str::from_utf8(&b[10..10 + hl]).ok()?;
    let shape = head.split("'shape': (").nth(1)?.split(')').next()?;
    let shape: Vec<usize> = shape.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let data = &b[10 + hl..];
    let v: Vec<f32> = if head.contains("<i8") {
        data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect()
    } else {
        data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    // Невыровненные представления torch numpy пишет в столбцовом порядке.
    if head.contains("'fortran_order': True") && shape.len() > 1 {
        let n: usize = shape.iter().product();
        let mut c = vec![0f32; n];
        let mut idx = vec![0usize; shape.len()];
        for x in c.iter_mut() {
            // смещение в F-порядке: первый индекс меняется быстрее всех
            let (mut off, mut mul) = (0usize, 1usize);
            for (d, i) in shape.iter().zip(&idx) {
                off += i * mul;
                mul *= d;
            }
            *x = v[off];
            for d in (0..shape.len()).rev() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        return Some((shape, c));
    }
    Some((shape, v))
}

fn cmp(name: &str, mine: &[f32], dir: &str) {
    let Some((shape, want)) = npy(&format!("{dir}/{name}.npy")) else {
        println!("{name:24} — нет дампа");
        return;
    };
    if want.len() != mine.len() {
        println!("{name:24} РАЗМЕР: у нас {}, эталон {} {shape:?}", mine.len(), want.len());
        return;
    }
    let (mut d, mut dot, mut na, mut nb, mut m) = (0f32, 0f64, 0f64, 0f64, 0f32);
    for (a, b) in mine.iter().zip(&want) {
        d = d.max((a - b).abs());
        m = m.max(b.abs());
        dot += *a as f64 * *b as f64;
        na += *a as f64 * *a as f64;
        nb += *b as f64 * *b as f64;
    }
    println!("{name:24} макс|Δ| = {d:.3e}, cos = {:.9}, макс|эталон| = {m:.3} {shape:?}", dot / (na.sqrt() * nb.sqrt() + 1e-30));
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    synaptix_kernels_cpu::ensure_registered();
    let device = if a.get(2).map(String::as_str) == Some("cuda") {
        synaptix_kernels_cuda::ensure_registered();
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let dir = a[1].clone();
    let m = GigaAm4::load(&a[0], &device).expect("модель");
    let (_, wav) = npy(&format!("{dir}/00_wav.npy")).expect("00_wav.npy");
    let host = |t: &Tensor| -> Vec<f32> { t.contiguous().unwrap().to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap() };
    let t0 = std::time::Instant::now();
    let (raw, e2e, n, tr, te) = m
        .debug_pass(&wav, &mut |name, t| {
            let v = host(t);
            if name == "06_encoder_out" {
                // эталон — [1, 768, T'], у нас [T', 768]
                let d = t.dims()[1];
                let n = t.dims()[0];
                let mut tr = vec![0f32; v.len()];
                for i in 0..n {
                    for c in 0..d {
                        tr[c * n + i] = v[i * d + c];
                    }
                }
                cmp(name, &tr, &dir);
            } else {
                cmp(name, &v, &dir);
            }
        })
        .expect("проход");
    let secs = t0.elapsed().as_secs_f64();
    // эталон — log_softmax
    let lsm = |l: &[f32], c: usize| -> Vec<f32> {
        l.chunks(c)
            .flat_map(|r| {
                let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z = r.iter().map(|v| (v - m).exp()).sum::<f32>().ln() + m;
                r.iter().map(move |v| v - z).collect::<Vec<f32>>()
            })
            .collect()
    };
    cmp("07_logprobs_raw", &lsm(&raw, raw.len() / n), &dir);
    cmp("07_logprobs_e2e", &lsm(&e2e, e2e.len() / n), &dir);
    println!("кадров {n}, {:.1} с звука за {secs:.2} с", wav.len() as f64 / 16000.0);
    println!("raw: {tr}");
    println!("e2e: {te}");
}
