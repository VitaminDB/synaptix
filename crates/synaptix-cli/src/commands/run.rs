use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;

use crate::commands::device::resolve as resolve_device;
use crate::commands::llm_facade::{
    generation_options, resolve_precision, Part, PrecisionFlags, RuntimeFlags, SamplingFlags, Splitter,
};

pub struct RunArgs {
    pub model: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    pub device: String,
    /// Размер preallocated KV-буфера + RoPE capacity (long-context). None →
    /// prompt+max_tokens для KV, max_position_embeddings для RoPE.
    pub max_seq: Option<usize>,
    /// Attention-backend: auto|flash-decode|fa2|fa4. None → SYN_ATTN env → auto.
    pub attn: Option<String>,
    /// KV-кеш dtype: None/bf16 → compute dtype; fp8/mxfp8 → MXFP8 block-scale
    /// (256K-контекст); fp8e4m3 → legacy per-tensor E4M3.
    pub kv_dtype: Option<String>,
    /// Пресет точности: none (default) | nvfp4 | fp8/mxfp8. Задаёт compute + квант групп.
    pub quant: Option<String>,
    /// Override compute (активаций): f16|bf16|f32. None → SYN_DTYPE/пресет.
    pub compute_dtype: Option<String>,
    /// Override веса attn+mlp групп: bf16|f16|fp8|nvfp4.
    pub storage_dtype: Option<String>,
    /// Override проекции в словарь (lm_head): bf16|f16|fp8|nvfp4.
    pub lm_head_dtype: Option<String>,
    /// Override таблицы эмбеддингов: bf16|f16|fp8.
    pub embed_dtype: Option<String>,
    /// CUDA-graph decode: захватывает single-token forward в граф и реплеит
    /// (устраняет launch-overhead). Требует CUDA. Greedy ≈ обычному decode.
    pub graph: bool,
    /// Прогон prefill+1-токен до замера (прогрев NVRTC JIT для честного бенча).
    pub warmup: bool,
    /// MTP (multi-token prediction): спекулятивный декод на встроенной
    /// nextn-голове модели. Требует greedy (--temperature 0) и mtp.* в бандле.
    /// Включается автоматически при выполнении условий; флаг делает требование
    /// жёстким (ошибка, если MTP недоступен).
    pub mtp: bool,
    /// Запретить MTP даже когда он доступен.
    pub no_mtp: bool,
    /// Отключить CUDA-graph для MTP verify-шага (для сравнения/отладки).
    pub no_graph_mtp: bool,
    /// Изображение для мультимодального промпта (Qwen3.6-VL).
    pub image: Option<PathBuf>,
    /// Видео для мультимодального промпта (Muse Glimmer).
    pub video: Option<PathBuf>,
    /// Запретить DFlash-спекуляцию (Muse Glimmer), даже если драфтер в бандле.
    pub no_dflash: bool,
    pub no_lookup: bool,
    pub sampling: SamplingFlags,
    pub prefill_batch: usize,
    pub chat: bool,
    pub system: Option<String>,
    pub no_think: bool,
    pub reasoning_effort: Option<String>,
    pub stop: Vec<String>,
    pub max_image_tokens: Option<usize>,
    pub layer_sync: Option<String>,
    pub expert_cache_gb: Option<f64>,
    pub host_mirror_gb: Option<f64>,
    pub qwen4_spec: bool,
    pub embed_host: bool,
}

fn base_cfg(args: &RunArgs) -> synaptix_llm_common::GenerationConfig {
    let s = &args.sampling;
    synaptix_llm_common::GenerationConfig {
        top_k: s.top_k.unwrap_or(0),
        top_p: s.top_p.unwrap_or(1.0),
        min_p: s.min_p.unwrap_or(0.0),
        repetition_penalty: s.repetition_penalty.unwrap_or(1.0),
        repeat_last_n: s.repeat_last_n.unwrap_or(0),
        presence_penalty: s.presence_penalty.unwrap_or(0.0),
        frequency_penalty: s.frequency_penalty.unwrap_or(0.0),
        prefill_batch: args.prefill_batch,
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Qwen3,
    Hybrid,
    MuseGlimmer,
    Qwen4Exp,
    Facade(synaptix::facade::arch::LlmArch),
}

pub fn detect_arch(path: &Path) -> Result<Arch, String> {
    use synaptix::facade::arch::{detect_llm_arch, LlmArch};
    Ok(match detect_llm_arch(path)? {
        LlmArch::Qwen3 => Arch::Qwen3,
        LlmArch::Hybrid => Arch::Hybrid,
        LlmArch::MuseGlimmer => Arch::MuseGlimmer,
        LlmArch::Qwen4Exp => Arch::Qwen4Exp,
        other => Arch::Facade(other),
    })
}

pub fn run(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = resolve_device(&args.device);
    crate::commands::device::resolve_attn(args.attn.as_deref());

    if !args.model.exists() {
        return Err(format!("model path not found: {}", args.model.display()).into());
    }
    let (precision, profile) = resolve_precision(
        &args.model,
        &PrecisionFlags {
            quant: args.quant.as_deref(),
            compute_dtype: args.compute_dtype.as_deref(),
            storage_dtype: args.storage_dtype.as_deref(),
            lm_head_dtype: args.lm_head_dtype.as_deref(),
            embed_dtype: args.embed_dtype.as_deref(),
            kv_dtype: args.kv_dtype.as_deref(),
        },
        &RuntimeFlags {
            no_graph: !args.graph,
            no_spec: args.no_mtp || args.no_dflash,
            layer_sync: args.layer_sync.clone(),
            prefill_batch: args.prefill_batch,
        },
    )?;
    let arch = detect_arch(&args.model)?;
    if let Some(gb) = args.expert_cache_gb {
        std::env::set_var("SYN_QWEN4EXP_EXPERT_CACHE_GB", gb.to_string());
    }
    if let Some(gb) = args.host_mirror_gb {
        std::env::set_var("SYN_QWEN4EXP_HOST_MIRROR_GB", gb.to_string());
    }
    if args.qwen4_spec {
        std::env::set_var("SYN_QWEN4EXP_SPEC", "1");
    }
    if args.embed_host {
        std::env::set_var("SYN_QWEN4EXP_EMBED_HOST", "1");
    }
    let facade = matches!(arch, Arch::Facade(_))
        || args.chat
        || !args.stop.is_empty()
        || (args.image.is_some() && args.video.is_some())
        || (arch == Arch::Hybrid && args.video.is_some());
    if arch == Arch::Qwen3 && (args.image.is_some() || args.video.is_some()) {
        return Err("qwen3 — текстовая модель: --image/--video не поддержаны".into());
    }
    eprintln!(
        "synaptix run: loading model from {} (arch={arch:?}, профиль {profile}: compute={:?}, attn_w={:?}, mlp_w={:?}, lm_head={:?}, embed={:?}, kv={:?}, {:?}){}",
        args.model.display(),
        precision.compute, precision.attn_w, precision.mlp_w,
        precision.lm_head, precision.embed, precision.kv, device,
        if facade { " через фасад" } else { "" }
    );
    if facade {
        return run_facade(&args, device, precision);
    }
    match arch {
        Arch::Qwen3 => run_qwen3(&args, device, precision),
        Arch::Hybrid => run_hybrid(&args, device, precision),
        Arch::MuseGlimmer => run_muse(&args, device, precision),
        Arch::Qwen4Exp => run_qwen4_exp(&args, device, precision),
        Arch::Facade(_) => unreachable!(),
    }
}

fn run_facade(args: &RunArgs, device: Device, precision: PrecisionConfig) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix::facade::llm::{load_llm, LlmGeneration, Message};

    let t0 = std::time::Instant::now();
    let (llm, tok) = load_llm(&args.model, device, precision, args.max_seq)?;
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let mut media = Vec::new();
    if (args.image.is_some() || args.video.is_some()) && !llm.ensure_media_tower()? {
        return Err("в бандле нет башни зрения".into());
    }
    if let Some(p) = &args.image {
        let m = llm.encode_image(p, args.max_image_tokens)?;
        eprintln!("synaptix run: image {} → {} vision-токенов", p.display(), m.tokens);
        media.push(m);
    }
    if let Some(p) = &args.video {
        let m = llm.encode_video(p)?;
        eprintln!("synaptix run: video {} → {} токенов", p.display(), m.tokens);
        media.push(m);
    }
    let chat = args.chat || !media.is_empty();
    let thinking = !args.no_think;
    let prompt = if chat {
        let blocks: String = media.iter().map(|m| m.prompt_block.as_str()).collect();
        let mut msgs = Vec::new();
        if let Some(sys) = &args.system {
            msgs.push(Message::system(sys.clone()));
        }
        msgs.push(Message::user(format!("{blocks}{}", args.prompt)));
        tok.apply_chat_template_reasoning(&msgs, true, thinking, args.reasoning_effort.as_deref(), None)?
    } else {
        args.prompt.clone()
    };
    let ids = tok.encode(&prompt)?;
    eprintln!("synaptix run: prompt {} tokens", ids.len());
    let mut sampling = args.sampling.clone();
    if !chat {
        sampling.temperature = sampling.temperature.or(Some(args.temperature));
    }
    let max_seq = args.max_seq.unwrap_or(ids.len() + args.max_tokens + 16);
    let (opts, preset) = generation_options(&args.model, &sampling, thinking, args.max_tokens, max_seq, args.seed);
    eprintln!(
        "synaptix run: сэмплинг {preset}: t={} top_k={} top_p={} min_p={} rep={} presence={}",
        opts.temperature, opts.top_k, opts.top_p, opts.min_p, opts.repeat_penalty, opts.presence_penalty
    );
    let refs: Vec<_> = media.iter().collect();
    let run_once = |max_new: usize, print: bool| -> Result<(usize, u128, u128), Box<dyn std::error::Error>> {
        let mut gen = LlmGeneration::new(&llm, synaptix::facade::llm::GenerationOptions { max_new_tokens: max_new, ..opts.clone() });
        gen.set_stop_tokens(tok.eos_ids().to_vec());
        for s in &args.stop {
            gen.add_stop_sequence(s);
        }
        let mut splitter = Splitter::new(&tok, &prompt);
        let started = std::time::Instant::now();
        let mut first = None;
        let mut n = 0usize;
        let mut last_part = Part::Answer;
        let on_token = |id: u32, delta: &str| -> bool {
            first.get_or_insert_with(std::time::Instant::now);
            n += 1;
            if print {
                use std::io::Write;
                if chat {
                    splitter.push(id, delta, &mut |part, text| {
                        if part != last_part {
                            eprintln!();
                            last_part = part;
                        }
                        match part {
                            Part::Answer => print!("{text}"),
                            Part::Reasoning => eprint!("\x1b[2m{text}\x1b[0m"),
                            Part::Tool => eprint!("\x1b[33m{text}\x1b[0m"),
                        }
                    });
                } else {
                    print!("{delta}");
                }
                let _ = std::io::stdout().flush();
            }
            true
        };
        if refs.is_empty() {
            gen.generate_streaming(&ids, &tok, on_token)?;
        } else {
            gen.generate_streaming_media(&ids, &tok, &refs, on_token)?;
        }
        let end = std::time::Instant::now();
        let first = first.unwrap_or(end);
        Ok((n, (first - started).as_millis(), (end - first).as_millis()))
    };
    if args.warmup {
        run_once(2, false)?;
        eprintln!("synaptix run: warmup done");
    }
    if !chat {
        print!("{}", args.prompt);
    }
    let (n, prefill_ms, decode_ms) = run_once(args.max_tokens, true)?;
    println!();
    print_stats(ids.len(), n, prefill_ms, decode_ms);
    Ok(())
}

fn run_muse(
    args: &RunArgs,
    device: Device,
    precision: PrecisionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_llm_muse_glimmer::pipeline::{GenerationConfig, MusePipeline};

    let compute = precision.compute;
    let t0 = std::time::Instant::now();
    let mut pipeline = MusePipeline::load_with_precision(&args.model, device, precision, args.max_seq)
        .map_err(|e| format!("load: {e}"))?;
    if !args.no_dflash && args.temperature == 0.0 {
        match pipeline.load_dflash(&args.model, precision) {
            Ok(true) => eprintln!("synaptix run: DFlash-драфтер подключён"),
            Ok(false) => {}
            Err(e) => eprintln!("synaptix run: DFlash пропущен: {e}"),
        }
    }
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let need_vision = args.image.is_some() || args.video.is_some();
    if need_vision
        && !pipeline
            .load_vision(&args.model, compute)
            .map_err(|e| format!("vision load: {e}"))?
    {
        return Err("в бандле нет vision-башни (model.vision_tower.*)".into());
    }
    let image_embeds = match &args.image {
        Some(path) => {
            let n = pipeline.image_token_count(path).map_err(|e| format!("image: {e}"))?;
            let emb = pipeline.encode_image(path).map_err(|e| format!("image: {e}"))?;
            eprintln!("synaptix run: image {} → {} vision-токенов", path.display(), n);
            Some((n, emb))
        }
        None => None,
    };
    let video_embeds = match &args.video {
        Some(path) => {
            let (emb, info) = pipeline.encode_video(path).map_err(|e| format!("video: {e}"))?;
            eprintln!(
                "synaptix run: video {} → {} групп × {} токенов",
                path.display(),
                info.groups,
                info.tokens_per_group
            );
            Some((emb, info))
        }
        None => None,
    };
    if need_vision {
        pipeline.release_vision();
    }
    let pipeline = pipeline;

    let prompt_text = if let Some((n, _)) = &image_embeds {
        let pad = "<|patch|>".repeat(*n);
        format!(
            "<|begin_of_text|><|start|>user<|message|><|image_start|>{pad}<|image_end|>{}<|eot|><|start|>assistant",
            args.prompt
        )
    } else if let Some((_, info)) = &video_embeds {
        format!(
            "<|begin_of_text|><|start|>user<|message|>{}{}<|eot|><|start|>assistant",
            info.prompt_block(),
            args.prompt
        )
    } else {
        args.prompt.clone()
    };
    let prompt_ids = pipeline.encode(&prompt_text).map_err(|e| format!("tokenize: {e}"))?;
    eprintln!("synaptix run: prompt {} tokens", prompt_ids.len());

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        max_seq: args.max_seq,
        ..base_cfg(args)
    };

    let media = if let Some((_, emb)) = &image_embeds {
        Some((emb.clone(), true))
    } else {
        video_embeds.as_ref().map(|(emb, _)| (emb.clone(), false))
    };
    if let Some((emb, is_image)) = media {
        let mut noop = |_: u32| true;
        let t = std::time::Instant::now();
        let res = if is_image {
            pipeline.generate_with_images(&prompt_ids, std::slice::from_ref(&emb), gen_cfg, &mut noop)
        } else {
            pipeline.generate_with_video(&prompt_ids, &emb, gen_cfg, &mut noop)
        };
        let (ids, stats) = res.map_err(|e| format!("vlm generate: {e}"))?;
        let text = pipeline.decode(&ids).map_err(|e| format!("decode: {e}"))?;
        println!("{text}");
        eprintln!(
            "synaptix run: {} prompt + {} new tokens in {} ms | prefill {} ms | decode {} ms ({:.2} tok/s)",
            stats.prompt_tokens,
            stats.new_tokens,
            t.elapsed().as_millis(),
            stats.prefill_ms,
            stats.decode_ms,
            stats.new_tokens as f64 / (stats.decode_ms.max(1) as f64 / 1000.0)
        );
        return Ok(());
    }

    let use_graph = args.graph && !device.is_cpu() && pipeline.graph_decode_supported();
    let (new_ids, stats) = if args.temperature == 0.0 && pipeline.has_dflash() {
        let mut noop = |_: u32| true;
        let (ids, stats, dfs) = pipeline
            .generate_dflash_streaming(&prompt_ids, gen_cfg, &mut noop)
            .map_err(|e| format!("generate(dflash): {e}"))?;
        eprintln!(
            "synaptix run: DFlash блоков {} | черновиков {} | принято {} ({:.1}%)",
            dfs.steps,
            dfs.drafted,
            dfs.accepted,
            dfs.acceptance() * 100.0
        );
        (ids, stats)
    } else if args.temperature == 0.0 && !device.is_cpu() && !args.no_lookup {
        let mut noop = |_: u32| true;
        let (ids, stats, lk) = pipeline
            .generate_lookup_streaming(&prompt_ids, gen_cfg, &mut noop)
            .map_err(|e| format!("generate(lookup): {e}"))?;
        eprintln!(
            "synaptix run: lookup шагов {} | черновиков {} | принято {} ({:.1}%)",
            lk.steps,
            lk.drafted,
            lk.accepted,
            lk.acceptance() * 100.0
        );
        (ids, stats)
    } else if use_graph {
        let mut noop = |_: u32| true;
        pipeline
            .generate_with_graph_streaming(&prompt_ids, gen_cfg, &mut noop)
            .map_err(|e| format!("generate(graph): {e}"))?
    } else {
        pipeline
            .generate(&prompt_ids, gen_cfg)
            .map_err(|e| format!("generate: {e}"))?
    };
    let text = pipeline.decode(&new_ids).map_err(|e| format!("decode: {e}"))?;
    print_run_result(
        &args.prompt, &text, stats.prompt_tokens, stats.new_tokens, stats.prefill_ms, stats.decode_ms,
    );
    Ok(())
}

fn run_qwen4_exp(
    args: &RunArgs,
    device: Device,
    precision: PrecisionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_llm_common::GenerationConfig;
    use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

    let t0 = std::time::Instant::now();
    let mut pipeline =
        Qwen4ExpPipeline::load_with_precision(&args.model, device, precision, args.max_seq)
            .map_err(|e| format!("load: {e}"))?;
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Картинка и видео занимают в промпте столько заполнителей, сколько строк
    // отдала башня; собираем промпт из id напрямую — decode выбрасывает
    // спецтокены.
    if args.image.is_some() || args.video.is_some() {
        if !pipeline
            .load_vision(&args.model, precision.compute)
            .map_err(|e| format!("vision load: {e}"))?
        {
            return Err("в бандле нет компонента vision".into());
        }
    }
    let media = match (&args.image, &args.video) {
        (Some(path), _) => {
            let limits = synaptix_vlm_qwen3::PreprocessLimits::default();
            let n = pipeline.image_token_count(path, limits).map_err(|e| format!("image: {e}"))?;
            let (feats, grid) = pipeline.encode_image(path, limits).map_err(|e| format!("image: {e}"))?;
            eprintln!(
                "synaptix run: image {} → {n} vision-токенов, сетка {}×{}",
                path.display(),
                grid.0,
                grid.1
            );
            let pad = pipeline
                .config
                .image_token_id
                .ok_or("в конфиге нет id заполнителя картинки")?;
            Some((pad, n, feats, synaptix_llm_common::mrope::Grid3::image(grid.0, grid.1)))
        }
        (None, Some(path)) => {
            let limits = synaptix_vlm_qwen3::VideoLimits::default();
            let (feats, grid) = pipeline.encode_video(path, limits).map_err(|e| format!("video: {e}"))?;
            let n = grid.tokens();
            eprintln!(
                "synaptix run: video {} → {n} vision-токенов, сетка {}×{}×{}",
                path.display(),
                grid.t,
                grid.h,
                grid.w
            );
            let pad = pipeline
                .config
                .video_token_id
                .ok_or("в конфиге нет id заполнителя видео")?;
            Some((pad, n, feats, grid))
        }
        (None, None) => None,
    };
    let pipeline = pipeline;

    let prompt_ids = match &media {
        None => pipeline.encode(&args.prompt).map_err(|e| format!("tokenize: {e}"))?,
        Some((pad, n, _, _)) => {
            let mut ids = pipeline
                .encode("<|im_start|>user\n")
                .map_err(|e| format!("tokenize: {e}"))?;
            if let Some(t) = pipeline.config.vision_start_token_id {
                ids.push(t);
            }
            ids.extend(std::iter::repeat_n(*pad, *n));
            if let Some(t) = pipeline.config.vision_end_token_id {
                ids.push(t);
            }
            ids.extend(pipeline.encode(&args.prompt).map_err(|e| format!("tokenize: {e}"))?);
            ids.extend(
                pipeline
                    .encode("<|im_end|>\n<|im_start|>assistant\n")
                    .map_err(|e| format!("tokenize: {e}"))?,
            );
            ids
        }
    };
    eprintln!("synaptix run: prompt {} tokens", prompt_ids.len());

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        max_seq: args.max_seq,
        ..base_cfg(args)
    };
    let (new_ids, stats) = match &media {
        None => pipeline.generate(&prompt_ids, gen_cfg).map_err(|e| format!("generate: {e}"))?,
        Some((pad, _, feats, grid)) => {
            let inputs = [synaptix_llm_qwen4_exp::pipeline::MediaInput {
                pad: *pad,
                embeds: feats.clone(),
                grids: vec![*grid],
            }];
            pipeline
                .generate_media_streaming(&prompt_ids, &inputs, gen_cfg, &mut |_: u32| true)
                .map_err(|e| format!("generate: {e}"))?
        }
    };
    let text = pipeline.decode(&new_ids).map_err(|e| format!("decode: {e}"))?;
    print_run_result(
        &args.prompt, &text, stats.prompt_tokens, stats.new_tokens, stats.prefill_ms, stats.decode_ms,
    );
    let report = synaptix_llm_qwen4_exp::norm::profile_report();
    if !report.is_empty() {
        eprintln!("synaptix run: этапы forward:\n{report}");
    }
    let moe = synaptix_llm_common::profile::report();
    if !moe.is_empty() {
        eprintln!("synaptix run: этапы MoE:\n{moe}");
    }
    if let Some(c) = pipeline.expert_cache_stats() {
        let total = c.hits + c.misses;
        eprintln!(
            "synaptix run: эксперты — {} резидентов ({:.1} ГБ), обращений {}, подкачано {} ({:.1} ГБ) за {:.1} с, пропущено {}, батчем {} из {}",
            c.resident,
            c.bytes as f64 / (1u64 << 30) as f64,
            total,
            c.fetched + c.misses,
            (c.fetched + c.misses) as f64 * 2.9 / 1024.0,
            c.fetch_millis as f64 / 1000.0,
            c.skipped,
            c.batched,
            c.batched + c.unbatched,
        );
    }
    Ok(())
}

fn run_qwen3(
    args: &RunArgs,
    device: Device,
    precision: PrecisionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_llm_qwen3::pipeline::{GenerationConfig, Qwen3Pipeline};

    let t0 = std::time::Instant::now();
    let pipeline = Qwen3Pipeline::load_with_precision(&args.model, device, precision, args.max_seq)
        .map_err(|e| format!("load: {e}"))?;
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let prompt_ids = pipeline.encode(&args.prompt).map_err(|e| format!("tokenize: {e}"))?;
    eprintln!("synaptix run: prompt {} tokens", prompt_ids.len());

    // MXFP8-KV поддержан dev/graph-путём (B3.6: device-pos append + device-Tkv
    // flash-decode) — квант-KV больше не отключает граф.
    let graph_ok = args.graph && !device.is_cpu() && pipeline.graph_decode_supported();

    // --warmup — прогон одного prefill+1-токен до замера, чтобы NVRTC JIT
    // (~100ms, one-time) не загрязнял prefill_ms. Для честного warm-бенчмарка.
    if args.warmup {
        let warm = GenerationConfig {
            max_new_tokens: 4,
            temperature: 0.0,
            max_seq: args.max_seq,
            ..Default::default()
        };
        if graph_ok {
            let _ = pipeline.generate_with_graph(&prompt_ids, warm.clone());
        }
        let _ = pipeline.generate(&prompt_ids, warm);
        eprintln!("synaptix run: warmup done (JIT прогрет)");
    }

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        eos_token_id: pipeline.config.eos_token_id,
        max_seq: args.max_seq,
        ..base_cfg(args)
    };

    let use_graph = graph_ok;
    let (new_ids, stats) = if use_graph {
        {
            pipeline
                .generate_with_graph(&prompt_ids, gen_cfg)
                .map_err(|e| format!("generate(graph): {e}"))?
        }
    } else {
        pipeline
            .generate(&prompt_ids, gen_cfg)
            .map_err(|e| format!("generate: {e}"))?
    };
    let text = pipeline.decode(&new_ids).map_err(|e| format!("decode: {e}"))?;
    print_run_result(
        &args.prompt, &text, stats.prompt_tokens, stats.new_tokens, stats.prefill_ms, stats.decode_ms,
    );
    Ok(())
}

fn run_hybrid(
    args: &RunArgs,
    device: Device,
    precision: PrecisionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_llm_qwen3_next_hybrid::pipeline::{GenerationConfig, HybridPipeline};

    let compute = precision.compute;
    let t0 = std::time::Instant::now();
    let want_mtp = !args.no_mtp && args.image.is_none();
    let pipeline = HybridPipeline::load_with_precision_mtp(
        &args.model, device, precision, args.max_seq, want_mtp,
    )
    .map_err(|e| format!("load: {e}"))?;
    if args.mtp && !pipeline.has_mtp() {
        return Err("MTP запрошен, но mtp.* нет в бандле (нужен MTP-вариант GGUF)".into());
    }
    let use_mtp = pipeline.has_mtp() && !args.no_mtp && args.image.is_none();
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let mut pipeline = pipeline;
    let image_embeds = match &args.image {
        Some(path) => {
            if !pipeline
                .load_vision(&args.model, compute)
                .map_err(|e| format!("vision load: {e}"))?
            {
                return Err("в бандле нет компонента `vision` (сконвертируйте GGUF с --mmproj)".into());
            }
            let limits = synaptix_vlm_qwen3::PreprocessLimits::default();
            let n = pipeline
                .image_token_count(path, limits)
                .map_err(|e| format!("image: {e}"))?;
            let emb = pipeline
                .encode_image(path, limits)
                .map_err(|e| format!("image: {e}"))?;
            eprintln!("synaptix run: image {} → {} vision-токенов", path.display(), n);
            Some((n, emb))
        }
        None => None,
    };
    let pipeline = pipeline;

    let prompt_text = match &image_embeds {
        Some((n, _)) => {
            let pad = "<|image_pad|>".repeat(*n);
            format!(
                "<|im_start|>user\n<|vision_start|>{pad}<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n",
                args.prompt
            )
        }
        None => args.prompt.clone(),
    };
    let prompt_ids = pipeline.encode(&prompt_text).map_err(|e| format!("tokenize: {e}"))?;
    eprintln!("synaptix run: prompt {} tokens", prompt_ids.len());

    if let Some((_, emb)) = &image_embeds {
        let gen_cfg = GenerationConfig {
            max_new_tokens: args.max_tokens,
            temperature: args.temperature,
            seed: args.seed,
            max_seq: args.max_seq,
            ..base_cfg(args)
        };
        let mut noop = |_: u32| true;
        let t = std::time::Instant::now();
        let (ids, stats) = pipeline
            .generate_with_images(&prompt_ids, std::slice::from_ref(emb), gen_cfg, &mut noop)
            .map_err(|e| format!("vlm generate: {e}"))?;
        let text = pipeline.decode(&ids).map_err(|e| format!("decode: {e}"))?;
        println!("{text}");
        eprintln!(
            "synaptix run: {} prompt + {} new tokens in {} ms | prefill {} ms | decode {} ms ({:.2} tok/s)",
            stats.prompt_tokens,
            stats.new_tokens,
            t.elapsed().as_millis(),
            stats.prefill_ms,
            stats.decode_ms,
            stats.new_tokens as f64 / (stats.decode_ms.max(1) as f64 / 1000.0)
        );
        return Ok(());
    }

    let gen_cfg = GenerationConfig {
        max_new_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        max_seq: args.max_seq,
        ..base_cfg(args)
    };

    if use_mtp {
        let mut noop = |_: u32| true;
        let t = std::time::Instant::now();
        let graph_mtp = !args.no_graph_mtp && !device.is_cpu();
        let res = if graph_mtp {
            pipeline.generate_mtp_with_graph(&prompt_ids, gen_cfg, &mut noop)
        } else {
            pipeline.generate_mtp(&prompt_ids, gen_cfg, &mut noop)
        };
        let (ids, stats, mtp) = res.map_err(|e| format!("mtp generate: {e}"))?;
        let text = pipeline.decode(&ids).map_err(|e| format!("decode: {e}"))?;
        println!("{}{}", args.prompt, text);
        eprintln!(
            "synaptix run: {} prompt + {} new tokens in {} ms",
            stats.prompt_tokens,
            stats.new_tokens,
            t.elapsed().as_millis()
        );
        eprintln!(
            "synaptix run: prefill {} tok in {} ms | decode {} tok in {} ms ({:.2} tok/s)",
            stats.prompt_tokens,
            stats.prefill_ms,
            stats.new_tokens,
            stats.decode_ms,
            stats.new_tokens as f64 / (stats.decode_ms.max(1) as f64 / 1000.0)
        );
        eprintln!(
            "synaptix run: MTP шагов {} | черновиков {} | принято {} ({:.1}%)",
            mtp.steps,
            mtp.drafted,
            mtp.accepted,
            mtp.acceptance() * 100.0
        );
        return Ok(());
    }

    // CUDA-graph для гибрида требует compute=F16 (ядра linear-decode F16-нативные).
    // MXFP8-KV поддержан (B3.6) — квант-KV граф не отключает.
    let want_graph = args.graph && !device.is_cpu();
    let use_graph = want_graph && pipeline.graph_decode_supported();
    if want_graph && !use_graph {
        eprintln!("synaptix run: CUDA-graph недоступен (нужны compute=F16 и все блоки на карте) → обычный decode");
    }

    let (new_ids, stats) = if use_graph {
        {
            pipeline
                .generate_with_graph(&prompt_ids, gen_cfg)
                .map_err(|e| format!("generate(graph): {e}"))?
        }
    } else {
        pipeline
            .generate(&prompt_ids, gen_cfg)
            .map_err(|e| format!("generate: {e}"))?
    };
    let text = pipeline.decode(&new_ids).map_err(|e| format!("decode: {e}"))?;
    print_run_result(
        &args.prompt, &text, stats.prompt_tokens, stats.new_tokens, stats.prefill_ms, stats.decode_ms,
    );
    Ok(())
}

/// Печать сгенерированного текста + статистики prefill/decode (общая для всех arch).
fn print_run_result(
    prompt: &str,
    text: &str,
    prompt_tokens: usize,
    new_tokens: usize,
    prefill_ms: u128,
    decode_ms: u128,
) {
    print!("{prompt}{text}");
    println!();
    print_stats(prompt_tokens, new_tokens, prefill_ms, decode_ms);
}

fn print_stats(prompt_tokens: usize, new_tokens: usize, prefill_ms: u128, decode_ms: u128) {
    let tot = prefill_ms + decode_ms;
    let prefill_tps = if prefill_ms > 0 {
        (prompt_tokens as f32) / (prefill_ms as f32 / 1000.0)
    } else {
        0.0
    };
    let decode_tps = if decode_ms > 0 && new_tokens > 1 {
        ((new_tokens.saturating_sub(1)) as f32) / (decode_ms as f32 / 1000.0)
    } else {
        0.0
    };
    eprintln!("synaptix run: {prompt_tokens} prompt + {new_tokens} new tokens in {tot} ms");
    eprintln!(
        "synaptix run: prefill {} tok in {} ms ({:.1} tok/s) | decode {} tok in {} ms ({:.2} tok/s)",
        prompt_tokens, prefill_ms, prefill_tps,
        new_tokens.saturating_sub(1), decode_ms, decode_tps,
    );
}
