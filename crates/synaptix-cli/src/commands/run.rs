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
    /// Запретить MTP даже когда он доступен.
    pub no_mtp: bool,
    /// Глубина цепочки черновиков MTP (1…3); None — умолчание.
    pub mtp_depth: Option<usize>,
    /// Бюджет размышлений в токенах (ход с размышлениями); None — без бюджета.
    pub thinking_budget: Option<usize>,
    /// Изображение для мультимодального промпта (Qwen3.6-VL).
    pub image: Option<PathBuf>,
    /// Видео для мультимодального промпта (Muse Glimmer).
    pub video: Option<PathBuf>,
    /// Звук для мультимодального промпта (WeMM-Video).
    pub audio: Option<PathBuf>,
    /// Видео без звуковой дорожки.
    pub no_video_audio: bool,
    /// Запретить DFlash-спекуляцию (Muse Glimmer), даже если драфтер в бандле.
    pub no_dflash: bool,
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
        device,
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
    eprintln!(
        "synaptix run: loading model from {} (arch={arch:?}, профиль {profile}: compute={:?}, attn_w={:?}, mlp_w={:?}, lm_head={:?}, embed={:?}, kv={:?}, {:?})",
        args.model.display(),
        precision.compute, precision.attn_w, precision.mlp_w,
        precision.lm_head, precision.embed, precision.kv, device,
    );
    run_facade(&args, device, precision)
}

fn run_facade(args: &RunArgs, device: Device, precision: PrecisionConfig) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix::facade::llm::{load_llm, LlmGeneration, Message};

    let t0 = std::time::Instant::now();
    let (llm, tok) = load_llm(&args.model, device, precision, args.max_seq)?;
    eprintln!("synaptix run: model loaded in {:.2}s", t0.elapsed().as_secs_f32());
    if let Some(d) = args.mtp_depth {
        if !llm.set_mtp_depth(d) {
            eprintln!("synaptix run: --mtp-depth: у модели нет MTP-головы");
        }
    }
    // `SYNAPTIX_LLM_PROF=1` — профиль по операциям (как в службе): префилл и
    // обычный (неграфовый) декод; отчёт — после статистики.
    let prof = std::env::var_os("SYNAPTIX_LLM_PROF").is_some();
    synaptix_llm_common::model::set_prefill_prof(prof);
    synaptix_llm_common::model::set_decode_prof(prof);
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
        let m = llm.encode_video_opts(p, !args.no_video_audio)?;
        let audio = m.audio.as_ref().map_or(0, |a| a.dims()[0]);
        eprintln!(
            "synaptix run: video {} → {} токенов{}",
            p.display(),
            m.tokens,
            if audio > 0 { format!(" (из них звук {audio})") } else { String::new() }
        );
        media.push(m);
    }
    if let Some(p) = &args.audio {
        if !llm.supports_audio() {
            return Err("модель не принимает звук (нет аудио-башни)".into());
        }
        let t = std::time::Instant::now();
        let m = llm.encode_audio(p)?;
        eprintln!(
            "synaptix run: audio {} → {} токенов за {:.2} с",
            p.display(),
            m.tokens,
            t.elapsed().as_secs_f32()
        );
        media.push(m);
    }
    let chat = args.chat || !media.is_empty();
    let thinking = !args.no_think;
    let prompt = if chat {
        let blocks: String = media.iter().map(|m| m.prompt_block.as_str()).collect();
        let mut msgs = Vec::new();
        if let Some(sys) = args.system.clone().or_else(|| llm.default_system_prompt().map(str::to_string)) {
            msgs.push(Message::system(sys));
        }
        // Qwen3-Omni Captioner текста не принимает — только аудиоблок.
        if llm.audio_only_prompt() && !media.is_empty() {
            if !args.prompt.trim().is_empty() {
                eprintln!("synaptix run: модель принимает только звук — текст запроса отброшен");
            }
            msgs.push(Message::user(blocks));
        } else {
            msgs.push(Message::user(format!("{blocks}{}", args.prompt)));
        }
        llm.finalize_prompt(tok.apply_chat_template_reasoning(&msgs, true, thinking, args.reasoning_effort.as_deref(), None)?)
    } else {
        args.prompt.clone()
    };
    let ids = tok.encode(&prompt)?;
    eprintln!("synaptix run: prompt {} tokens", ids.len());
    // Сверка входа с эталоном: `SYN_DUMP_IDS=файл` — id промпта через пробел.
    if let Some(p) = std::env::var_os("SYN_DUMP_IDS") {
        let text: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
        let _ = std::fs::write(p, text.join(" "));
    }
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
        // Учёт размышлений — всегда, когда ход с ними; бюджет 0 — без
        // принудительного `</think>` (иначе итог печатал «размышлений 0»).
        if chat && thinking {
            gen.set_thinking(args.thinking_budget.unwrap_or(0));
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
        if print {
            let f = gen.finish();
            eprintln!(
                "\nsynaptix run: остановка — {}; токенов {}, размышлений {}{}{}",
                f.reason.as_str(),
                f.tokens,
                f.thinking_tokens,
                if f.thinking_forced { ", закрыты по бюджету" } else { "" },
                if f.thinking_open { ", ход оборван внутри размышлений" } else { "" }
            );
        }
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
    if std::env::var_os("SYNAPTIX_LLM_PROF").is_some() {
        eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
    }
}
