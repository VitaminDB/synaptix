//! Речь Drama 3 из бандла:
//! `drama3_run BUNDLE.syn OUT.wav SEED "текст" [ОБРАЗЕЦ.wav "расшифровка образца"]…`
//! Переменные: `D3_TEMP`, `D3_TOP_P`, `D3_MAX_FRAMES`; `D3_DUMP=каталог` —
//! дамп для сверки с эталоном (`scripts/reference/drama3_check.py`): строки
//! промпта, коды образца, состояния основы, выходы слоёв башни на префилле,
//! логиты голов.
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tts_drama3::{Drama3, Drama3Request, Reference, SAMPLE_RATE};

fn read_wav(path: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).expect("wav образца");
    let s = r.spec();
    let raw: Vec<f32> = match s.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|v| v.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let k = 1.0 / (1i64 << (s.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|v| v.unwrap() as f32 * k).collect()
        }
    };
    let ch = s.channels as usize;
    let mono: Vec<f32> = raw.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect();
    if s.sample_rate == SAMPLE_RATE {
        mono
    } else {
        synaptix_audio::resample::resample_torchaudio(&mono, s.sample_rate, SAMPLE_RATE).expect("ресэмпл")
    }
}

fn write_wav(path: &str, pcm: &[f32]) {
    let spec = hound::WavSpec { channels: 1, sample_rate: SAMPLE_RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec).expect("wav");
    for &v in pcm {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

fn f32s(t: &Tensor) -> Vec<u8> {
    let v = t.contiguous().unwrap().to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("drama3_run BUNDLE.syn OUT.wav SEED \"текст\" [ОБРАЗЕЦ.wav \"расшифровка\"]…");
        std::process::exit(2);
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let env = |k: &str| std::env::var(k).ok();
    let t0 = Instant::now();
    let m = Drama3::load(std::path::Path::new(&a[0]), Device::Cuda(0)).expect("бандл");
    eprintln!("загрузка {:.1} с", t0.elapsed().as_secs_f64());
    // `D3_DROP_AFTER_LOAD=1` — выгрузить сразу после загрузки, пока зеркало
    // экспертов ещё копируется (проверка порядка освобождения).
    if env("D3_DROP_AFTER_LOAD").is_some() {
        drop(m);
        eprintln!("модель выгружена сразу после загрузки");
        std::thread::sleep(std::time::Duration::from_secs(3));
        println!("после выгрузки жив");
        return;
    }
    let mut refs = Vec::new();
    for p in a[4..].chunks(2) {
        let pcm = read_wav(&p[0]);
        let t = Instant::now();
        let codes = m.encode_reference(&pcm).expect("кодек: образец");
        eprintln!("образец {}: {:.1} с звука → {} кадров за {:.2} с", p[0], pcm.len() as f64 / SAMPLE_RATE as f64, codes[0].len(), t.elapsed().as_secs_f64());
        if let Some(d) = env("D3_DUMP") {
            let flat: Vec<u8> = codes.iter().flat_map(|c| c.iter().flat_map(|v| (*v as i32).to_le_bytes())).collect();
            std::fs::write(format!("{d}/ref_codes.i32"), flat).unwrap();
            std::fs::write(format!("{d}/ref_pcm.f32"), pcm.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            let back = m.decode_codes(&codes).expect("кодек: декод");
            std::fs::write(format!("{d}/ref_decoded.f32"), back.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            write_wav(&format!("{d}/ref_decoded.wav"), &back);
        }
        refs.push(Reference { transcript: p.get(1).cloned().unwrap_or_default(), codes });
    }
    let mut req = Drama3Request { text: a[3].replace("\\n", "\n"), references: refs, seed: a[2].parse().expect("SEED"), ..Default::default() };
    if let Some(v) = env("D3_TEMP") {
        req.temperature = v.parse().unwrap();
    }
    if let Some(v) = env("D3_TOP_P") {
        req.top_p = v.parse().unwrap();
    }
    if let Some(v) = env("D3_MAX_FRAMES") {
        req.max_frames = v.parse().unwrap();
    }
    let rows = m.prompt_rows(&req.text, &req.references).expect("промпт");
    eprintln!("промпт {} строк", rows.len());
    if let Some(d) = env("D3_DUMP") {
        let l = rows.len();
        std::fs::write(format!("{d}/rows.i32"), rows.iter().flat_map(|r| r.iter().flat_map(|v| (*v as i32).to_le_bytes())).collect::<Vec<u8>>()).unwrap();
        let emb = m.embed(&rows).unwrap();
        std::fs::write(format!("{d}/embeds.f32"), f32s(&emb)).unwrap();
        let mut kv = m.lm.model.make_kv_cache(1, l + 8).unwrap();
        let hid = m.lm.model.forward_from_hidden(&emb, &mut kv).unwrap();
        let normed = m.lm.model.normed_all(&hid).unwrap().reshape(vec![l, 2048]).unwrap();
        std::fs::write(format!("{d}/normed.f32"), f32s(&normed)).unwrap();
        let mut tc = m.tower.make_cache(l + 8).unwrap();
        let mut tap = Vec::new();
        m.tower.prefill(&normed.narrow(0, 0, l - 1).unwrap().contiguous().unwrap(), &rows[..l - 1], &rows[1..], 0, &mut tc, Some(&mut tap)).unwrap();
        for (i, t) in tap.iter().enumerate() {
            std::fs::write(format!("{d}/tower_{i:02}.f32"), f32s(t)).unwrap();
        }
        for r in 0..9 {
            let lg = m.tower.head(r, &tap[4 + r]).unwrap();
            std::fs::write(format!("{d}/head_{r}.f32"), lg.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        }
        let lg = m.head_logits(&normed.narrow(0, l - 1, 1).unwrap().contiguous().unwrap()).unwrap();
        std::fs::write(format!("{d}/main_logits.f32"), lg.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        eprintln!("дамп сверки → {d}");
    }
    let out = m
        .generate(&req, &mut |c, nc, i, _| {
            if i % 20 == 0 {
                eprint!("\rкусок {}/{nc}, кадр {i}   ", c + 1);
            }
            true
        })
        .expect("генерация");
    eprintln!();
    write_wav(&a[1], &out.pcm);
    let secs = out.pcm.len() as f64 / SAMPLE_RATE as f64;
    println!(
        "кадров {} ({secs:.1} с звука), кусков {}, конец речи: {}: речь {:.1} с ({:.1} кадр/с, нужно 21,5), кодек {:.2} с → {}",
        out.frames, out.chunks, if out.finished { "да" } else { "нет (предел кадров)" }, out.ar_seconds, out.frames as f64 / out.ar_seconds, out.codec_seconds, a[1]
    );
    // `D3_DROP_EARLY=1` — выгрузить модель сразу после реплики (служба так делает,
    // когда соседней модели не хватает видеопамяти) и подождать: проверка, что
    // выгрузка при недокопированном зеркале экспертов не роняет процесс.
    if env("D3_DROP_EARLY").is_some() {
        drop(m);
        eprintln!("модель выгружена");
        // Как служба: ужать память и загрузить модель в том же процессе заново.
        let _ = synaptix_kernels_cuda::release_device_caches();
        let _ = synaptix_core::memory::expert_arena::release_empty(0);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(0);
        std::thread::sleep(std::time::Duration::from_secs(2));
        if let Ok((free, total)) = synaptix_core::device::cuda::mem_info(0) {
            println!("после выгрузки и ужатия: свободно {} из {} МБ видеопамяти", free >> 20, total >> 20);
        }
        if env("D3_NO_RELOAD").is_some() {
            return;
        }
        let m2 = Drama3::load(std::path::Path::new(&a[0]), Device::Cuda(0)).expect("вторая загрузка");
        eprintln!("вторая загрузка готова");
        let mut r = req.clone();
        r.seed += 1;
        match m2.generate(&r, &mut |_, _, _, _| true) {
            Ok(o) => println!("вторая загрузка: кадров {} ({:.1} кадр/с)", o.frames, o.frames as f64 / o.ar_seconds),
            Err(e) => println!("вторая загрузка: ОШИБКА {e}"),
        }
        std::thread::sleep(std::time::Duration::from_secs(40));
        println!("после второй загрузки жив");
        return;
    }
    // `D3_AGAIN="текст|текст…"` — ещё реплики той же моделью и с теми же образцами
    // (как подряд идущие вызовы службы); `D3_TRIM=1` — между репликами ужать пулы
    // видеопамяти, как делает служба при загрузке соседней модели.
    if let Some(more) = env("D3_AGAIN") {
        for (i, text) in more.split('|').enumerate() {
            if env("D3_TRIM").is_some() {
                let _ = synaptix_core::device::cuda::synchronize_all(0);
                // Как `release_vram` службы: кэши ядер (дескрипторы, скретчи), пустые
                // slab'ы арены экспертов, пулы.
                let (descs, scratch) = synaptix_kernels_cuda::release_device_caches();
                let arena = synaptix_core::memory::expert_arena::release_empty(0) >> 20;
                let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(0);
                eprintln!("пулы ужаты: дескрипторов {descs}, скретчей {} МБ, арена {arena} МБ", scratch >> 20);
            }
            let mut r = req.clone();
            r.text = text.to_string();
            r.seed = req.seed + 1 + i as u64;
            match m.generate(&r, &mut |_, _, _, _| true) {
                Ok(o) => println!("повтор {}: кадров {} ({:.1} кадр/с), конец речи: {}", i + 1, o.frames, o.frames as f64 / o.ar_seconds, o.finished),
                Err(e) => println!("повтор {}: ОШИБКА {e}", i + 1),
            }
        }
    }
}
