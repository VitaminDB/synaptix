//! Вопрос по картинкам: `ink2_vision <модель> <картинка>[,<картинка>…] "<вопрос>" [ключи]`.
//!
//! Ключи: `--effort none|low|medium|high|…` (по умолчанию none — без размышлений),
//! `--max-new N` (256), `--max-side PX` (4096), `--max-tokens N` (клеток на картинку),
//! `--dump <файл>` — признаки башни первой картинки F32 в файл (сверка),
//! `--tower-only` — только башня, без текстовой модели.
//!
//! Промпт — chat_template.jinja: реплика пользователя с частями `[картинки…, текст]`,
//! плейсхолдер картинки разворачивается в `rows·cols` токенов (как Inkling2Processor).

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_llm_inkling2::preprocess::{load_image, ImageLimits};
use synaptix_llm_inkling2::vision::{expand_image_tokens, insert_image_parts, prefill_with_images, ImageSpan, IMAGE_TOKEN_ID};
use synaptix_llm_inkling2::{Inkling2Model, Inkling2Weights, VisionTower};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};
use synaptix_tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("ink2_vision <модель> <картинка>[,…] \"<вопрос>\" [--effort none] [--max-new 256] [--max-side 4096] [--max-tokens N] [--dump f] [--tower-only]");
        std::process::exit(2);
    }
    let (dir, imgs, question) = (&args[1], &args[2], &args[3]);
    let mut effort = "none".to_string();
    let mut max_new = 256usize;
    let mut lim = ImageLimits::default();
    let mut dump: Option<String> = None;
    let mut tower_only = false;
    let mut i = 4;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--effort" => effort = v,
            "--max-new" => max_new = v.parse().expect("--max-new"),
            "--max-side" => lim.max_side = v.parse().expect("--max-side"),
            "--max-tokens" => lim.max_tokens = v.parse().expect("--max-tokens"),
            "--dump" => dump = Some(v),
            "--tower-only" => {
                tower_only = true;
                i += 1;
                continue;
            }
            o => panic!("ключ {o}"),
        }
        i += 2;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let w = Inkling2Weights::open(dir).expect("open");
    let vcfg = w.config.vision.clone().expect("vision_config");

    // Башня: признаки всех картинок, затем она освобождается под текстовую модель.
    let t0 = Instant::now();
    let tower = VisionTower::load(&w, dev).expect("башня");
    eprintln!("башня загружена за {:.1} с", t0.elapsed().as_secs_f32());
    // Прогрев: первая картинка иначе платит за сборку ядер (NVRTC).
    let t = Instant::now();
    let warm = synaptix_llm_inkling2::preprocess::prepare_rgb8(&[128u8; 40 * 40 * 3], 40, 40, &vcfg);
    no_grad(|| tower.encode(&warm)).and_then(|f| f.to_device(Device::Cpu).map_err(|e| synaptix_llm_common::ModelError::Forward(e.to_string()))).expect("прогрев");
    eprintln!("прогрев башни {:.1} с", t.elapsed().as_secs_f32());
    let mut feats = Vec::new();
    for path in imgs.split(',') {
        let t = Instant::now();
        let p = load_image(path, &vcfg, lim).expect("картинка");
        let prep = t.elapsed().as_secs_f32();
        let t = Instant::now();
        let f = no_grad(|| tower.encode(&p)).expect("encode");
        // Чтение строки на хост ждёт конца башни.
        f.narrow(0, 0, 1).and_then(|r| r.to_device(Device::Cpu)).expect("sync");
        eprintln!(
            "{path}: {}×{} → {}×{} px, клеток {}×{} = {} токенов (патчей {}), препроцесс {:.2} с, башня {:.2} с",
            p.orig_width, p.orig_height, p.width, p.height, p.rows, p.cols, p.tokens(), p.tokens() * 4, prep,
            t.elapsed().as_secs_f32()
        );
        if let (Some(d), true) = (&dump, feats.is_empty()) {
            let v = f.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(d, bytes).expect("dump");
            eprintln!("признаки {:?} → {d}", f.dims());
        }
        feats.push(f);
    }
    drop(tower);
    if tower_only {
        return;
    }

    let t0 = Instant::now();
    let model = Inkling2Model::load(&w, dev).expect("load");
    eprintln!("модель загружена за {:.1} с", t0.elapsed().as_secs_f32());
    let tok = HfTokenizer::from_bytes(&w.tokenizer_json).expect("tokenizer");
    let tpl = ChatTemplate::from_source(w.chat_template.clone().expect("chat_template.jinja"));
    let opts = RenderOptions::new().with_generation_prompt(true).with_var("reasoning_effort", serde_json::Value::String(effort));
    let text = tpl.render(&[Message::user(question.as_str())], &opts).expect("render");
    let text = insert_image_parts(&text, feats.len());
    let ids = tok.encode(&text, false).expect("encode").ids;
    let counts: Vec<usize> = feats.iter().map(|f| f.dims()[0]).collect();
    let (ids, starts) = expand_image_tokens(&ids, IMAGE_TOKEN_ID, &counts).expect("плейсхолдеры");
    let spans: Vec<ImageSpan> = starts.into_iter().zip(feats).map(|(start, embeds)| ImageSpan { start, embeds }).collect();
    eprintln!("промпт {} токенов ({} — картинки)", ids.len(), counts.iter().sum::<usize>());

    let mut cache = model.make_cache(ids.len() + max_new + 8).expect("cache");
    let t1 = Instant::now();
    let logits = no_grad(|| prefill_with_images(&model, &mut cache, &ids, &spans)).expect("prefill");
    let argmax = |l: &synaptix_core::tensor::Tensor| -> u32 {
        let v = l.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
    };
    let mut next = argmax(&logits);
    eprintln!("префилл {} ток. {:.2} с", ids.len(), t1.elapsed().as_secs_f32());
    let eos = w.config.eos_ids.clone();
    let mut out = Vec::new();
    let t2 = Instant::now();
    loop {
        out.push(next);
        if eos.contains(&next) || out.len() >= max_new {
            break;
        }
        let l = no_grad(|| model.decode(next, &mut cache)).expect("decode");
        next = argmax(&l);
    }
    eprintln!("декод {} ток. {:.1} ток/с", out.len(), out.len() as f32 / t2.elapsed().as_secs_f32());
    println!("{}", tok.decode(&out, false).unwrap_or_default());
}
