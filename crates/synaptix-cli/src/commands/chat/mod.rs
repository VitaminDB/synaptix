mod app;
mod engine;
mod event;
mod template;
mod ui;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self as ct_event, Event, KeyEventKind};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::Terminal;

use synaptix_llm_common::{GenerationConfig, GenerationStats, StreamSink};
use synaptix_llm_muse_glimmer::pipeline::MusePipeline;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
use synaptix_tokenizer::{SpecialTokens, Tokenizer as _};

use crate::commands::device::resolve as resolve_device;
use crate::commands::run::{detect_arch, Arch};

use app::App;
use engine::{EngineEvt, EngineHandle};

pub struct ChatArgs {
    pub model: PathBuf,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub context: usize,
    pub prefill_batch: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub device: String,
    pub attn: Option<String>,
    pub quant: Option<String>,
    pub kv_dtype: Option<String>,
    pub compute_dtype: Option<String>,
    pub storage_dtype: Option<String>,
    pub lm_head_dtype: Option<String>,
    pub embed_dtype: Option<String>,
    pub no_think: bool,
}

pub enum ChatPipeline {
    Qwen3(Qwen3Pipeline),
    Hybrid(HybridPipeline),
    MuseGlimmer(MusePipeline),
}

impl ChatPipeline {
    pub fn encode(&self, s: &str) -> Result<Vec<u32>, String> {
        match self {
            ChatPipeline::Qwen3(p) => p.encode(s).map_err(|e| e.to_string()),
            ChatPipeline::Hybrid(p) => p.encode(s).map_err(|e| e.to_string()),
            ChatPipeline::MuseGlimmer(p) => p.encode(s).map_err(|e| e.to_string()),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        match self {
            ChatPipeline::Qwen3(p) => p.decode(ids).map_err(|e| e.to_string()),
            ChatPipeline::Hybrid(p) => p.decode(ids).map_err(|e| e.to_string()),
            ChatPipeline::MuseGlimmer(p) => p.decode(ids).map_err(|e| e.to_string()),
        }
    }

    pub fn specials(&self) -> SpecialTokens {
        match self {
            ChatPipeline::Qwen3(p) => p.tokenizer.special_tokens().clone(),
            ChatPipeline::Hybrid(p) => p.tokenizer.special_tokens().clone(),
            ChatPipeline::MuseGlimmer(p) => p.tokenizer.special_tokens().clone(),
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            ChatPipeline::Qwen3(p) => p.tokenizer.token_to_id(token),
            ChatPipeline::Hybrid(p) => p.tokenizer.token_to_id(token),
            ChatPipeline::MuseGlimmer(p) => p.tokenizer.token_to_id(token),
        }
    }

    pub fn make_kv_cache(&self, max_seq: usize) -> Result<synaptix_llm_common::KvCache, String> {
        match self {
            ChatPipeline::Qwen3(p) => p.model.make_kv_cache(1, max_seq).map_err(|e| e.to_string()),
            ChatPipeline::Hybrid(p) => p.model.make_kv_cache(1, max_seq).map_err(|e| e.to_string()),
            ChatPipeline::MuseGlimmer(p) => p.model.make_kv_cache(1, max_seq).map_err(|e| e.to_string()),
        }
    }

    /// Генерация с prefix-KV-кэшем: prefill стартует с `kv.seq_len`. На CUDA
    /// использует graph-decode (hybrid требует F16), иначе обычный decode.
    pub fn generate_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), String> {
        {
            use synaptix_core::device::Device;
            use synaptix_core::dtype::DType;
            match self {
                ChatPipeline::Qwen3(p) if matches!(p.model.device, Device::Cuda(_)) => {
                    return p
                        .generate_with_graph_resume(kv, ids, cfg, &mut *sink)
                        .map_err(|e| e.to_string());
                }
                ChatPipeline::Hybrid(p)
                    if matches!(p.model.device, Device::Cuda(_)) && p.model.dtype == DType::F16 =>
                {
                    return p
                        .generate_with_graph_resume(kv, ids, cfg, &mut *sink)
                        .map_err(|e| e.to_string());
                }
                ChatPipeline::MuseGlimmer(p) if p.graph_decode_supported() => {
                    return p
                        .generate_with_graph_resume(kv, ids, cfg, &mut *sink)
                        .map_err(|e| e.to_string());
                }
                _ => {}
            }
        }
        match self {
            ChatPipeline::Qwen3(p) => {
                p.generate_streaming_resume(kv, ids, cfg, sink).map_err(|e| e.to_string())
            }
            ChatPipeline::Hybrid(p) => {
                p.generate_streaming_resume(kv, ids, cfg, sink).map_err(|e| e.to_string())
            }
            ChatPipeline::MuseGlimmer(p) => {
                p.generate_streaming_resume(kv, ids, cfg, sink).map_err(|e| e.to_string())
            }
        }
    }
}

pub fn run(args: ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = resolve_device(&args.device);
    crate::commands::device::resolve_attn(args.attn.as_deref());
    if !args.model.exists() {
        return Err(format!("model path not found: {}", args.model.display()).into());
    }
    let precision = crate::commands::run::build_precision(
        args.quant.as_deref(),
        args.compute_dtype.as_deref(),
        args.storage_dtype.as_deref(),
        args.lm_head_dtype.as_deref(),
        args.embed_dtype.as_deref(),
        args.kv_dtype.as_deref(),
    )?;
    let arch = detect_arch(&args.model)?;
    eprintln!(
        "synaptix chat: loading {} (arch={arch:?}, compute={:?}, attn_w={:?}, kv={:?}, {:?})",
        args.model.display(),
        precision.compute,
        precision.attn_w,
        precision.kv,
        device
    );
    let t0 = std::time::Instant::now();
    let pipeline = match arch {
        Arch::Qwen3 => ChatPipeline::Qwen3(
            Qwen3Pipeline::load_with_precision(&args.model, device, precision, Some(args.context))
                .map_err(|e| format!("load: {e}"))?,
        ),
        Arch::Hybrid => ChatPipeline::Hybrid(
            HybridPipeline::load_with_precision(&args.model, device, precision, Some(args.context))
                .map_err(|e| format!("load: {e}"))?,
        ),
        Arch::MuseGlimmer => ChatPipeline::MuseGlimmer(
            MusePipeline::load_with_precision(&args.model, device, precision, Some(args.context))
                .map_err(|e| format!("load: {e}"))?,
        ),
    };
    eprintln!("synaptix chat: loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let prompt = template::Prompt::load(&args.model, &pipeline, !args.no_think);
    if !prompt.has_template() {
        eprintln!("synaptix chat: chat-template не найден — простой формат <|im_start|>");
    }
    let mut cfg = build_gen_config(&args);
    cfg.eos_token_ids = prompt.stop_ids();

    let engine = EngineHandle::spawn(pipeline);
    let mut app = App::new(
        prompt,
        args.system.clone(),
        arch_label(arch).into(),
        model_label(&args.model),
        cfg,
    );

    let res = run_ui(&mut app, &engine);
    engine.shutdown();
    res.map_err(Into::into)
}

fn build_gen_config(args: &ChatArgs) -> GenerationConfig {
    GenerationConfig {
        // 0 = без лимита: генерим до <|im_end|> / заполнения контекста (decode-loop
        // и так стопается на eos и при pos >= max_seq). Иначе — жёсткий потолок.
        max_new_tokens: if args.max_tokens == 0 { args.context } else { args.max_tokens },
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        repetition_penalty: args.repetition_penalty,
        // Окно штрафа и presence/frequency у CLI-чата не настраиваются —
        // дефолты сохраняют прежнее поведение (штраф по всему контексту).
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: args.seed,
        eos_token_id: None,
        eos_token_ids: Vec::new(),
        max_seq: Some(args.context),
        prefill_batch: args.prefill_batch,
    }
}

fn arch_label(arch: Arch) -> &'static str {
    match arch {
        Arch::Qwen3 => "qwen3",
        Arch::Hybrid => "qwen3-next-hybrid",
        Arch::MuseGlimmer => "muse-glimmer",
    }
}

fn model_label(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

struct TermGuard;

impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        prev(info);
    }));
}

fn run_ui(app: &mut App, engine: &EngineHandle) -> io::Result<()> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    install_panic_hook();
    let _guard = TermGuard;

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    loop {
        terminal.draw(|f| ui::draw(f, app))?;

        while let Ok(evt) = engine.evt_rx.try_recv() {
            match evt {
                EngineEvt::TokenDelta(s) => app.push_delta(&s),
                EngineEvt::Done { stats, cached, ctx_used, ctx_max } => {
                    app.finish(&stats, cached, ctx_used, ctx_max)
                }
                EngineEvt::Error(e) => app.fail(&e),
            }
        }
        if app.generating {
            app.tick_spinner();
        }

        if ct_event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = ct_event::read()? {
                if key.kind == KeyEventKind::Press {
                    event::handle_key(app, key, engine);
                }
            }
        }

        if app.should_quit {
            break;
        }
    }
    Ok(())
}
