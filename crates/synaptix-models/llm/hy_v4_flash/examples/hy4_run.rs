//! Прогон Hy4-flash: `cargo run --example hy4_run -- <бандл> (--ids файл | --text "…" | --chat "…") [--gen N]`.
//! `--wait-mirror` — дождаться заполнения зеркала экспертов перед замером.
//! `HY4_DUMP=каталог` — потоки после каждого слоя (сверка с scripts/reference/hy4_ref.py).

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_llm_hy_v4_flash::Hy4Pipeline;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let gen: usize = opt("--gen").and_then(|v| v.parse().ok()).unwrap_or(0);
    let p = Hy4Pipeline::load(path, Device::Cuda(0), Some(8192)).expect("загрузка");
    // `--wait-mirror`: замер в установившемся режиме — дождаться, пока зеркало
    // экспертов в RAM заполнится (иначе промахи делят диск с копировщиками).
    if args.iter().any(|a| a == "--wait-mirror") {
        let t0 = Instant::now();
        while !p.mirror_finished() {
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        if let Some((r, t)) = p.mirror_progress() {
            eprintln!("зеркало заполнено за {:.0} с: {:.1} из {:.1} ГБ", t0.elapsed().as_secs_f64(), r as f64 / 1e9, t as f64 / 1e9);
        }
    }
    let ids: Vec<u32> = if let Some(f) = opt("--ids") {
        std::fs::read_to_string(f).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect()
    } else if let Some(t) = opt("--chat") {
        // Шаблон Hy4 (chat_template.jinja, reasoning_effort=no_think).
        let s = format!(
            "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think<｜hy_end:opensource｜>\
             <｜hy_start:opensource｜>user<｜hy_middle:opensource｜>{t}<｜hy_end:opensource｜>\
             <｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource></think:opensource>"
        );
        p.encode(&s).unwrap()
    } else {
        p.encode(&opt("--text").unwrap_or_else(|| "The capital of France is".into())).unwrap()
    };
    if let Some(f) = opt("--save-ids") {
        std::fs::write(f, ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
    }
    eprintln!("токенов: {} {:?}", ids.len(), &ids[..ids.len().min(32)]);
    let mut s = p.new_session(ids.len() + gen + 16).unwrap();
    // `--spec K`: жадный спекулятивный декод драфтером DFly (K драфтов за шаг).
    if let Some(k) = opt("--spec").and_then(|v| v.parse::<usize>().ok()) {
        let d = p.dfly.as_ref().expect("в бандле нет драфтера DFly");
        let t0 = Instant::now();
        let mut h = no_grad(|| p.model.forward(&ids, &mut s.cache, Some(1))).unwrap();
        let mut bonus = synaptix_llm_hy_v4_flash::dfly::argmax(&p.model.logits(&h).unwrap()).unwrap();
        eprintln!("префилл {:.2} с", t0.elapsed().as_secs_f64());
        p.size_zc_pool();
        let _ = synaptix_llm_common::profile::report();
        let mut out = vec![bonus];
        let (mut steps, mut drafted, mut accepted) = (0usize, 0usize, 0usize);
        let mut depth = vec![0usize; k + 1];
        let (mut t_draft, mut t_verify) = (0f64, 0f64);
        let sync = || synaptix_core::device::cuda::synchronize(0).unwrap();
        let t1 = Instant::now();
        while out.len() < gen && !p.eos_ids().contains(&bonus) {
            let a = s.cache.seq_len();
            let td = Instant::now();
            let drafts = no_grad(|| d.draft(&p.model, &mut s.cache, a, bonus, &h, k)).unwrap().unwrap_or_default();
            sync();
            t_draft += td.elapsed().as_secs_f64();
            let tv = Instant::now();
            let mut block = vec![bonus];
            block.extend_from_slice(&drafts);
            let hb = no_grad(|| p.model.forward(&block, &mut s.cache, None)).unwrap();
            let lg = p.model.logits(&hb).unwrap();
            let tgt = synaptix_llm_hy_v4_flash::dfly::argmax_rows(&lg).unwrap();
            t_verify += tv.elapsed().as_secs_f64();
            let n = drafts.iter().zip(&tgt).take_while(|(x, y)| x == y).count();
            steps += 1;
            depth[n] += 1;
            drafted += drafts.len();
            accepted += n;
            out.extend_from_slice(&drafts[..n]);
            bonus = tgt[n];
            out.push(bonus);
            s.cache.truncate(a + n + 1);
            h = synaptix_llm_hy_v4_flash::kern::contig(&hb.narrow(0, n, 1).unwrap()).unwrap();
        }
        let dd = t1.elapsed().as_secs_f64();
        eprintln!(
            "спекуляция K={k}: {} токенов за {:.2} с = {:.2} ток/с; шагов {steps}, принято {accepted} из {drafted} драфтов ({:.0} %), {:.2} токена на шаг",
            out.len() - 1, dd, (out.len() - 1) as f64 / dd, 100.0 * accepted as f64 / drafted.max(1) as f64, (out.len() - 1) as f64 / steps.max(1) as f64
        );
        eprintln!("принято драфтов за шаг (0…K): {depth:?}; драфт {:.1} мс/шаг, верификация {:.1} мс/шаг", t_draft * 1e3 / steps.max(1) as f64, t_verify * 1e3 / steps.max(1) as f64);
        if synaptix_llm_common::profile::profiling() {
            eprintln!("профиль спекуляции ({steps} шагов):\n{}", synaptix_llm_common::profile::report());
        }
        println!("{}", p.decode(&out).unwrap());
        return;
    }
    let t0 = Instant::now();
    // `--last-decode`: последний токен — шагом декода (путь MoE на 1 токен),
    // остальное префиллом; логиты сравнимы с полным префиллом той же позиции.
    let (logits, _) = if args.iter().any(|a| a == "--last-decode") {
        p.prefill(&mut s, &ids[..ids.len() - 1], &mut |_: u32| true).expect("префилл");
        p.size_zc_pool();
        let h = no_grad(|| p.model.forward(&ids[ids.len() - 1..], &mut s.cache, Some(1))).unwrap();
        (p.model.logits(&h).unwrap(), 0)
    } else {
        p.prefill(&mut s, &ids, &mut |_: u32| true).expect("префилл")
    };
    let dt = t0.elapsed().as_secs_f64();
    let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    if let Some(f) = opt("--logits") {
        std::fs::write(f, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap());
    let top: Vec<(usize, f32, String)> = idx[..10].iter().map(|&i| (i, v[i], p.decode(&[i as u32]).unwrap_or_default())).collect();
    eprintln!("префилл {:.2} с ({:.1} ток/с); top-10: {top:?}", dt, ids.len() as f64 / dt);
    if synaptix_llm_common::profile::profiling() {
        eprintln!("профиль префилла:\n{}", synaptix_llm_common::profile::report());
    }
    if gen > 0 {
        p.size_zc_pool();
        let mut tok = idx[0] as u32;
        let mut out = vec![tok];
        let t1 = Instant::now();
        for _ in 1..gen {
            let h = no_grad(|| p.model.forward(&[tok], &mut s.cache, Some(1))).unwrap();
            let l = p.model.logits(&h).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            tok = l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
            out.push(tok);
            if p.eos_ids().contains(&tok) {
                break;
            }
        }
        let dd = t1.elapsed().as_secs_f64();
        eprintln!("декод {:.2} ток/с", (out.len() - 1) as f64 / dd);
        if let Some((hit, picks, res)) = p.model.zc_stats() {
            eprintln!("zero-copy: попаданий в слоты {:.1} % ({hit} из {picks}), резидентов {res}", 100.0 * hit as f64 / picks.max(1) as f64);
        }
        if synaptix_llm_common::profile::profiling() {
            eprintln!("профиль декода ({} шагов, {:.1} мс/шаг):\n{}", out.len() - 1, dd * 1e3 / (out.len() - 1) as f64, synaptix_llm_common::profile::report());
        }
        println!("{}", p.decode(&out).unwrap());
    }
}
