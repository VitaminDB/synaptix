//! Сравнение моделей по логитам: насколько квант уводит модель от эталона.
//!
//! Текст прогоняется префиллом кусками; на каждой позиции считается
//! log-softmax. Эталон ([`save_ref`]) хранит его целиком (f16, позиции ×
//! словарь). Сравнение ([`compare`]) даёт среднюю KL(эталон ‖ кандидат),
//! 99-й перцентиль KL, долю совпадений top-1 и перплексию текста у обеих
//! моделей. Модель подаётся замыканием `forward(кусок) → логиты [кусок, V]`.

use std::io::{Read, Write};
use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

type Error = Box<dyn std::error::Error>;

fn log_softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = v.iter().map(|x| ((x - m) as f64).exp()).sum();
    let l = m + s.ln() as f32;
    for x in v.iter_mut() {
        *x -= l;
    }
}

/// log-softmax всех позиций `ids`: прогон кусками по `chunk`.
pub fn run(ids: &[u32], vocab: usize, chunk: usize, mut forward: impl FnMut(&[u32]) -> Result<Tensor, Error>) -> Result<Vec<Vec<f32>>, Error> {
    let n = ids.len();
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    let t = std::time::Instant::now();
    for start in (0..n).step_by(chunk.max(1)) {
        let end = (start + chunk).min(n);
        let host = forward(&ids[start..end])?.to_device(Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if host.len() != (end - start) * vocab {
            return Err(format!("логиты: {} значений вместо {}×{vocab}", host.len(), end - start).into());
        }
        for r in 0..end - start {
            let mut row = host[r * vocab..(r + 1) * vocab].to_vec();
            log_softmax(&mut row);
            rows.push(row);
        }
        eprint!("\r  {end}/{n}");
    }
    eprintln!("  за {:.1} с", t.elapsed().as_secs_f64());
    Ok(rows)
}

/// Перплексия текста: позиция i предсказывает токен i+1.
pub fn perplexity(ids: &[u32], lp: impl Fn(usize, usize) -> f32) -> f64 {
    let n = ids.len();
    let s: f64 = (0..n - 1).map(|i| -(lp(i, ids[i + 1] as usize) as f64)).sum();
    (s / (n - 1) as f64).exp()
}

pub fn save_ref(path: &Path, ids: &[u32], rows: &[Vec<f32>]) -> Result<f64, Error> {
    let vocab = rows.first().map_or(0, |r| r.len());
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&(ids.len() as u64).to_le_bytes())?;
    f.write_all(&(vocab as u64).to_le_bytes())?;
    for i in ids {
        f.write_all(&i.to_le_bytes())?;
    }
    for row in rows {
        let h: Vec<u16> = row.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
        f.write_all(bytemuck::cast_slice(&h))?;
    }
    Ok(perplexity(ids, |i, t| rows[i][t]))
}

pub struct Comparison {
    pub kl_mean: f64,
    pub kl_p99: f64,
    pub top1: f64,
    pub ppl: f64,
    pub ppl_ref: f64,
    /// Средняя KL по участкам `bucket` позиций (скачок на границе куска
    /// префилла выдаёт ошибку переноса состояния, а не квант).
    pub kl_buckets: Vec<f64>,
}

pub fn compare(path: &Path, ids: &[u32], rows: &[Vec<f32>], bucket: usize) -> Result<Comparison, Error> {
    let n = ids.len();
    let vocab = rows.first().map_or(0, |r| r.len());
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
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
    if bytemuck::cast_slice::<u8, u32>(&rid) != ids {
        return Err("токены эталона другие".into());
    }
    let mut kls = Vec::with_capacity(n);
    let mut agree = 0usize;
    let mut ref_nll = 0f64;
    let mut buf = vec![0u8; 2 * vocab];
    let am = |v: &[f32]| v.iter().enumerate().fold(0usize, |b, (i, x)| if *x > v[b] { i } else { b });
    for (i, row) in rows.iter().enumerate() {
        f.read_exact(&mut buf)?;
        let p: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&buf).iter().map(|h| half::f16::from_bits(*h).to_f32()).collect();
        let kl: f64 = p.iter().zip(row).map(|(lp, lq)| (lp.exp() as f64) * ((lp - lq) as f64)).sum();
        kls.push(kl.max(0.0));
        if am(&p) == am(row) {
            agree += 1;
        }
        if i + 1 < n {
            ref_nll -= p[ids[i + 1] as usize] as f64;
        }
    }
    let mut sorted = kls.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    Ok(Comparison {
        kl_mean: kls.iter().sum::<f64>() / n as f64,
        kl_p99: sorted[(n * 99 / 100).min(n - 1)],
        top1: agree as f64 / n as f64,
        ppl: perplexity(ids, |i, t| rows[i][t]),
        ppl_ref: (ref_nll / (n - 1) as f64).exp(),
        kl_buckets: kls.chunks(bucket.max(1)).map(|c| c.iter().sum::<f64>() / c.len() as f64).collect(),
    })
}

/// Разбор аргументов примеров: `<bundle> <текст> --save|--ref <файл> [токенов]`
/// и печать результата — общая часть `logit_kl` у разных архитектур.
pub fn cli(
    args: &[String],
    load: &dyn Fn(&Path, usize) -> Result<(Box<dyn FnMut(&[u32]) -> Result<Tensor, Error>>, Box<dyn Fn(&str) -> Result<Vec<u32>, Error>>, usize), Error>,
) -> Result<(), Error> {
    if args.len() < 5 {
        return Err("usage: logit_kl <bundle.syn> <текст> --save|--ref <файл> [токенов, 1024]".into());
    }
    let bundle = Path::new(&args[1]);
    let text = String::from_utf8_lossy(&std::fs::read(&args[2])?).into_owned();
    let file = Path::new(&args[4]);
    let limit: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let chunk: usize = std::env::var("LOGIT_KL_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let (forward, encode, vocab) = load(bundle, limit)?;
    let mut ids = encode(&text)?;
    ids.truncate(limit);
    eprintln!("токенов {}, словарь {vocab}, кусок {chunk}", ids.len());
    let rows = run(&ids, vocab, chunk, forward)?;
    match args[3].as_str() {
        "--save" => {
            let ppl = save_ref(file, &ids, &rows)?;
            println!("эталон: {} позиций → {}; перплексия {ppl:.4}", ids.len(), file.display());
        }
        "--ref" => {
            let c = compare(file, &ids, &rows, 64)?;
            println!(
                "{}: KL сред. {:.5}, p99 {:.4}, top-1 {:.2} %, перплексия {:.4} (эталон {:.4})",
                bundle.file_name().unwrap_or_default().to_string_lossy(),
                c.kl_mean, c.kl_p99, 100.0 * c.top1, c.ppl, c.ppl_ref
            );
            let line: Vec<String> = c.kl_buckets.iter().enumerate().map(|(i, v)| format!("{}:{v:.3}", i * 64)).collect();
            println!("  KL по участкам 64: {}", line.join(" "));
        }
        o => return Err(format!("режим {o}: --save | --ref").into()),
    }
    Ok(())
}
