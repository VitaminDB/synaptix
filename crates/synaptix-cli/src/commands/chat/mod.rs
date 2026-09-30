mod app;
mod engine;
mod event;
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

use synaptix::facade::arch::detect_llm_arch;
use synaptix::facade::llm::{load_llm, MediaKind};

use crate::commands::device::resolve as resolve_device;
use crate::commands::llm_facade::{
    generation_options, presets_of, resolve_precision, PrecisionFlags, RuntimeFlags, SamplingFlags,
};

use app::{App, Settings};
use engine::{EngineCmd, EngineEvt, EngineHandle, EngineSetup};

pub struct ChatArgs {
    pub model: PathBuf,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub context: usize,
    pub prefill_batch: usize,
    pub sampling: SamplingFlags,
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
    pub reasoning_effort: Option<String>,
    pub stop: Vec<String>,
    pub image: Vec<PathBuf>,
    pub video: Vec<PathBuf>,
    pub max_image_tokens: Option<usize>,
    pub no_graph: bool,
    pub no_spec: bool,
    pub layer_sync: Option<String>,
}

pub fn run(args: ChatArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = resolve_device(&args.device);
    crate::commands::device::resolve_attn(args.attn.as_deref());
    if !args.model.exists() {
        return Err(format!("model path not found: {}", args.model.display()).into());
    }
    let arch = detect_llm_arch(&args.model)?;
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
            no_graph: args.no_graph,
            no_spec: args.no_spec,
            layer_sync: args.layer_sync.clone(),
            prefill_batch: args.prefill_batch,
        },
    )?;
    eprintln!(
        "synaptix chat: loading {} (arch={arch:?}, профиль {profile}: compute={:?}, attn_w={:?}, mlp_w={:?}, kv={:?}, {:?})",
        args.model.display(),
        precision.compute,
        precision.attn_w,
        precision.mlp_w,
        precision.kv,
        device
    );
    let t0 = std::time::Instant::now();
    let (llm, tok) = load_llm(&args.model, device, precision, Some(args.context))?;
    eprintln!("synaptix chat: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    if let Some(levels) = tok.reasoning_levels() {
        eprintln!("synaptix chat: глубина размышлений: {} (по умолчанию {})", levels.levels.join(" | "), levels.default);
    }
    let thinking = !args.no_think;
    let max_new = if args.max_tokens == 0 { args.context } else { args.max_tokens };
    let (opts, preset) =
        generation_options(&args.model, &args.sampling, thinking, max_new, args.context, args.seed);
    eprintln!(
        "synaptix chat: пресет {preset} (доступно: {}) → t={} top_k={} top_p={} min_p={} rep={} presence={}",
        presets_of(&args.model).join(", "),
        opts.temperature,
        opts.top_k,
        opts.top_p,
        opts.min_p,
        opts.repeat_penalty,
        opts.presence_penalty
    );

    let attach: Vec<(PathBuf, MediaKind)> = args
        .image
        .iter()
        .map(|p| (p.clone(), MediaKind::Image))
        .chain(args.video.iter().map(|p| (p.clone(), MediaKind::Video)))
        .collect();
    if !attach.is_empty() && !llm.supports_media() {
        return Err(format!("{arch:?}: модель не принимает картинки/видео").into());
    }
    let engine = EngineHandle::spawn(EngineSetup {
        llm,
        tok,
        context: args.context,
        stops: args.stop.clone(),
        max_image_tokens: args.max_image_tokens,
    });
    let mut app = App::new(
        args.system.clone(),
        format!("{arch:?}"),
        model_label(&args.model),
        Settings { opts, thinking, effort: args.reasoning_effort.clone(), preset },
    );
    for (path, kind) in attach {
        engine.send(EngineCmd::Attach { path, kind });
        app.busy_attach = true;
    }

    let res = run_ui(&mut app, &engine);
    engine.shutdown();
    res.map_err(Into::into)
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
                EngineEvt::Delta(part, s) => app.push_delta(part, &s),
                EngineEvt::Attached { block, tokens, label } => app.attached(&block, tokens, &label),
                EngineEvt::Done(stats) => app.finish(&stats),
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
