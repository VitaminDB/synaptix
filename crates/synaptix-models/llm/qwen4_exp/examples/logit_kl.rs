//! Сравнение бандлов по логитам: насколько квант экспертов уводит модель.
//!
//!   logit_kl <bundle.syn> <текст.txt> --save <ref.bin>     — эталон
//!   logit_kl <bundle.syn> <текст.txt> --ref  <ref.bin>     — сравнение
//!
//! Текст прогоняется префиллом кусками; на каждой позиции считается
//! log-softmax. Эталон хранит его целиком (f16, позиции × словарь). При
//! сравнении печатаются: средняя KL(эталон ‖ кандидат), 99-й перцентиль KL,
//! доля совпадений top-1 и перплексия текста у обеих моделей.

use std::io::{Read, Write};
use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn log_softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = v.iter().map(|x| ((x - m) as f64).exp()).sum();
    let l = m + s.ln() as f32;
    for x in v.iter_mut() {
        *x -= l;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        return Err("usage: logit_kl <bundle.syn> <текст> --save|--ref <файл> [токенов, 1024]".into());
    }
    let bundle = PathBuf::from(&a[1]);
    let text = std::fs::read_to_string(&a[2])?;
    let mode = a[3].as_str();
    let file = PathBuf::from(&a[4]);
    let limit: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let chunk: usize = std::env::var("LOGIT_KL_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(256);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipeline = Qwen4ExpPipeline::load_with_precision(&bundle, Device::Cuda(0), PrecisionConfig::nvfp4(), Some(limit + 64))?;
    let mut ids = pipeline.encode(&text)?;
    ids.truncate(limit);
    let n = ids.len();
    let vocab = pipeline.config.vocab_size;
    eprintln!("токенов {n}, словарь {vocab}, кусок {chunk}");

    let mut cache = pipeline.make_cache(n + 8)?;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    let t = std::time::Instant::now();
    for start in (0..n).step_by(chunk) {
        let end = (start + chunk).min(n);
        let logits = pipeline.model.forward(&ids[start..end], &mut cache)?;
        let host = logits
            .to_device(Device::Cpu)
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())?;
        for r in 0..end - start {
            let mut row = host[r * vocab..(r + 1) * vocab].to_vec();
            log_softmax(&mut row);
            rows.push(row);
        }
        eprint!("\r  {end}/{n}");
    }
    eprintln!("  за {:.1} с", t.elapsed().as_secs_f64());

    // NLL текста: позиция i предсказывает токен i+1.
    let nll = |r: &dyn Fn(usize, usize) -> f32| -> f64 {
        let s: f64 = (0..n - 1).map(|i| -(r(i, ids[i + 1] as usize) as f64)).sum();
        (s / (n - 1) as f64).exp()
    };

    match mode {
        "--save" => {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&file)?);
            f.write_all(&(n as u64).to_le_bytes())?;
            f.write_all(&(vocab as u64).to_le_bytes())?;
            for i in &ids {
                f.write_all(&i.to_le_bytes())?;
            }
            for row in &rows {
                let h: Vec<u16> = row.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
                f.write_all(bytemuck::cast_slice(&h))?;
            }
            println!("эталон: {n} позиций → {}; перплексия {:.4}", file.display(), nll(&|i, t| rows[i][t]));
        }
        "--ref" => {
            let mut f = std::io::BufReader::new(std::fs::File::open(&file)?);
            let mut b8 = [0u8; 8];
            f.read_exact(&mut b8)?;
            let rn = u64::from_le_bytes(b8) as usize;
            f.read_exact(&mut b8)?;
            let rv = u64::from_le_bytes(b8) as usize;
            if rn != n || rv != vocab {
                return Err(format!("эталон на {rn}×{rv}, здесь {n}×{vocab}").into());
            }
            let mut rid = vec![0u8; 4 * n];
            f.read_exact(&mut rid)?;
            if bytemuck::cast_slice::<u8, u32>(&rid) != ids.as_slice() {
                return Err("токены эталона другие".into());
            }
            let mut kls = Vec::with_capacity(n);
            let mut agree = 0usize;
            let mut ref_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut buf = vec![0u8; 2 * vocab];
            for row in &rows {
                f.read_exact(&mut buf)?;
                let p: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&buf).iter().map(|h| half::f16::from_bits(*h).to_f32()).collect();
                let kl: f64 = p.iter().zip(row).map(|(lp, lq)| (lp.exp() as f64) * ((lp - lq) as f64)).sum();
                kls.push(kl.max(0.0));
                let am = |v: &[f32]| v.iter().enumerate().fold(0usize, |b, (i, x)| if *x > v[b] { i } else { b });
                if am(&p) == am(row) {
                    agree += 1;
                }
                ref_rows.push(p);
            }
            let mean = kls.iter().sum::<f64>() / n as f64;
            let mut sorted = kls.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let p99 = sorted[(n * 99 / 100).min(n - 1)];
            println!(
                "{}: KL сред. {mean:.5}, p99 {p99:.4}, top-1 {:.2} %, перплексия {:.4} (эталон {:.4})",
                bundle.file_name().unwrap().to_string_lossy(),
                100.0 * agree as f64 / n as f64,
                nll(&|i, t| rows[i][t]),
                nll(&|i, t| ref_rows[i][t]),
            );
        }
        o => return Err(format!("режим {o}: --save | --ref").into()),
    }
    Ok(())
}
