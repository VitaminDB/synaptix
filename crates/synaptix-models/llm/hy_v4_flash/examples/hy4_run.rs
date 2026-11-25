//! Прогон Hy4-flash: `cargo run --example hy4_run -- <бандл> (--ids файл | --text "…" | --chat "…") [--gen N]`.
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
    let mut s = p.new_session(ids.len() + gen + 8).unwrap();
    let t0 = Instant::now();
    let (logits, _) = p.prefill(&mut s, &ids, &mut |_: u32| true).expect("префилл");
    let dt = t0.elapsed().as_secs_f64();
    let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    if let Some(f) = opt("--logits") {
        std::fs::write(f, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap());
    let top: Vec<(usize, f32, String)> = idx[..10].iter().map(|&i| (i, v[i], p.decode(&[i as u32]).unwrap_or_default())).collect();
    eprintln!("префилл {:.2} с ({:.1} ток/с); top-10: {top:?}", dt, ids.len() as f64 / dt);
    if gen > 0 {
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
        println!("{}", p.decode(&out).unwrap());
    }
}
