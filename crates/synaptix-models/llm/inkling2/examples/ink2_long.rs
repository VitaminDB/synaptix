//! Длинный контекст и «иголка»: в начале промпта — факт, дальше реальный
//! текст (корпус склеивается заранее), в конце — вопрос о факте.
//!
//!   ink2_long <модель> <корпус.txt> [--ctx 200000] [--gen 48]
//!             [--save-decode f.bin | --ref-decode f.bin]
//!
//! Режим экспертов — как у пайплайна (`SYN_INKLING2_EXPERTS=resident|zc`,
//! иначе по ёмкости контекста). Печатает скорость префилла по ходу, декод
//! после всего контекста, VRAM и попадания в слоты zero-copy. `--save-decode`
//! пишет логиты шагов декода, `--ref-decode` гонит те же токены
//! (teacher forcing) и считает KL шагов к сохранённым — сверка путей декода.
//! `--ref-decode f.bin --prefill-rows g.bin` — вместо декода считает логиты тех же
//! шагов префиллом (одним куском по токенам эталона) и пишет их в формате
//! `--save-decode`: эталон схемы префилла для сверки декода.

use std::io::{Read, Write};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_llm_inkling2::Inkling2Pipeline;

const NEEDLE: &str = "Важная заметка для проверки памяти: кодовое слово хранителя архива — «СИНИЙ-ВЕРБЛЮД-4718». Запомни его.\n\n";
const QUESTION: &str = "\n\nКонец материалов. Вопрос: какое кодовое слово хранителя архива было названо в самом начале? Ответь одной строкой.";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn log_softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let s: f64 = v.iter().map(|x| ((x - m) as f64).exp()).sum();
    let l = m + s.ln() as f32;
    v.iter_mut().for_each(|x| *x -= l);
}

fn gpu_used_mb() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let corpus = std::fs::read_to_string(&args[2])?;
    let ctx: usize = arg(&args, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(200_000);
    let gen: usize = arg(&args, "--gen").and_then(|v| v.parse().ok()).unwrap_or(48);
    let save = arg(&args, "--save-decode");
    let refd = arg(&args, "--ref-decode");
    let prefill_rows = arg(&args, "--prefill-rows");
    let dev = Device::Cuda(0);

    let t0 = Instant::now();
    let p = Inkling2Pipeline::load_with_ctx(model, dev, Some(ctx + gen + 8))?;
    eprintln!("загрузка {:.1} с, режим {:?}, слотов {:?}, VRAM {} МБ", t0.elapsed().as_secs_f64(), p.expert_mode(), p.model.zc_slots(), gpu_used_mb());

    let head = p.encode(&format!("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{NEEDLE}"))?;
    // `INK2_LONG_ANSWER_PREFIX` — начало ответа модели (например `<|content_text|>` — без рассуждения).
    let tail = p.encode(&format!("{QUESTION}<|end_message|><|message_model|>{}", std::env::var("INK2_LONG_ANSWER_PREFIX").unwrap_or_default()))?;
    let body = p.encode(&corpus)?;
    let room = ctx.saturating_sub(head.len() + tail.len());
    if body.len() < room {
        eprintln!("корпус короче нужного: {} < {room} токенов", body.len());
    }
    let mut ids = head.clone();
    ids.extend_from_slice(&body[..room.min(body.len())]);
    ids.extend_from_slice(&tail);
    let n = ids.len();
    eprintln!("промпт {n} токенов (корпус {} токенов)", body.len());

    let mut s = p.new_session(n + gen + 8)?;
    let chunk = 512usize;
    let t1 = Instant::now();
    let mut last = Instant::now();
    let mut logits = None;
    let mut off = 0;
    let mut mark = 0;
    while off < n {
        let end = (off + chunk).min(n);
        logits = Some(p.model.forward(&ids[off..end], &mut s.cache, false)?);
        off = end;
        if off - mark >= 16384 || off == n {
            let _ = logits.as_ref().unwrap().to_device(Device::Cpu)?;
            let dt = last.elapsed().as_secs_f64();
            eprintln!(
                "  {off}/{n}: участок {:.0} ток/с, в среднем {:.0} ток/с, VRAM {} МБ",
                (off - mark) as f64 / dt,
                off as f64 / t1.elapsed().as_secs_f64(),
                gpu_used_mb()
            );
            mark = off;
            last = Instant::now();
        }
    }
    let pf = t1.elapsed().as_secs_f64();
    let st_pf = p.model.zc_stats();
    println!("префилл {n} ток. за {pf:.1} с: {:.0} ток/с", n as f64 / pf);
    if let Some((h, k)) = st_pf {
        println!("попадания в слоты на префилле: {:.1} % ({h}/{k} выборов)", 100.0 * h as f64 / k.max(1) as f64);
    }

    // Декод: жадно (или токены эталона).
    let vocab = p.config.text.unpadded_vocab_size;
    let mut ref_rows: Option<(Vec<u32>, Vec<f32>)> = None;
    if let Some(f) = &refd {
        let mut b = Vec::new();
        std::fs::File::open(f)?.read_to_end(&mut b)?;
        let steps = u64::from_le_bytes(b[0..8].try_into()?) as usize;
        let toks: Vec<u32> = (0..steps).map(|i| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap())).collect();
        let o = 8 + 4 * steps;
        let rows: Vec<f32> = b[o..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        ref_rows = Some((toks, rows));
    }
    let mut lv = logits.unwrap().to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    if let Some(f) = &prefill_rows {
        let (toks, _) = ref_rows.as_ref().ok_or("--prefill-rows нужен --ref-decode")?;
        let steps = toks.len().min(gen);
        let mut rows = lv.clone();
        if steps > 1 {
            let l = p.model.forward(&toks[..steps - 1], &mut s.cache, true)?;
            rows.extend(l.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let mut w = std::io::BufWriter::new(std::fs::File::create(f)?);
        w.write_all(&(steps as u64).to_le_bytes())?;
        for t in &toks[..steps] {
            w.write_all(&t.to_le_bytes())?;
        }
        for r in rows.chunks_exact_mut(vocab) {
            log_softmax(r);
            for x in r.iter() {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        eprintln!("логиты шагов префиллом ({steps}) → {f}");
        return Ok(());
    }
    let mut out = Vec::new();
    let mut rows_out: Vec<f32> = Vec::new();
    let mut kls = Vec::new();
    let mut t_dec = None;
    let warm = 3;
    for step in 0..gen {
        let best = lv.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        let mut row = lv.clone();
        log_softmax(&mut row);
        let tok = match &ref_rows {
            Some((toks, rr)) if step < toks.len() => {
                let r = &rr[step * vocab..(step + 1) * vocab];
                let kl: f64 = r.iter().zip(&row).map(|(a, b)| (a.exp() as f64) * (*a as f64 - *b as f64)).sum();
                kls.push((kl, toks[step] == best));
                toks[step]
            }
            _ => best,
        };
        if save.is_some() {
            rows_out.extend_from_slice(&row);
        }
        out.push(tok);
        if step == warm {
            t_dec = Some((Instant::now(), p.model.zc_stats()));
        }
        if step + 1 == gen || p.config.eos_ids.contains(&tok) && ref_rows.is_none() {
            break;
        }
        lv = p.model.decode(tok, &mut s.cache)?.to_device(Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    }
    if let Some((t, st0)) = t_dec {
        let steps = out.len().saturating_sub(warm + 1);
        let dt = t.elapsed().as_secs_f64();
        println!("декод после {n} ток.: {:.1} ток/с ({steps} шагов)", steps as f64 / dt);
        if let (Some((h0, k0)), Some((h1, k1))) = (st0, p.model.zc_stats()) {
            println!("попадания в слоты на декоде: {:.1} % ({}/{} выборов)", 100.0 * (h1 - h0) as f64 / (k1 - k0).max(1) as f64, h1 - h0, k1 - k0);
        }
    }
    if let Device::Cuda(o) = dev {
        let (free, total) = synaptix_core::device::cuda::mem_info(o)?;
        println!("VRAM: занято {:.2} из {:.2} ГБ (nvidia-smi {} МБ)", (total - free) as f64 / 1073741824.0, total as f64 / 1073741824.0, gpu_used_mb());
    }
    println!("ответ: {:?}", p.decode(&out)?);
    println!("иголка найдена: {}", p.decode(&out)?.contains("СИНИЙ-ВЕРБЛЮД-4718"));
    if !kls.is_empty() {
        let mean = kls.iter().map(|k| k.0).sum::<f64>() / kls.len() as f64;
        let max = kls.iter().map(|k| k.0).fold(0.0, f64::max);
        let top1 = kls.iter().filter(|k| k.1).count();
        println!("KL декода к эталону: сред. {mean:.6}, макс. {max:.6}, top-1 {top1}/{}", kls.len());
    }
    if let Some(f) = save {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&f)?);
        w.write_all(&(out.len() as u64).to_le_bytes())?;
        for t in &out {
            w.write_all(&t.to_le_bytes())?;
        }
        for x in &rows_out {
            w.write_all(&x.to_le_bytes())?;
        }
        eprintln!("логиты декода → {f}");
    }
    Ok(())
}
