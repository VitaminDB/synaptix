use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use synaptix::facade::llm::{
    GenerationOptions, Llm, LlmGeneration, LlmKvSession, LlmTokenizer, MediaEmbedding, MediaKind, Message,
};

use crate::commands::llm_facade::{Part, Splitter};

pub struct Turn {
    pub messages: Vec<Message>,
    pub opts: GenerationOptions,
    pub thinking: bool,
    pub effort: Option<String>,
}

pub enum EngineCmd {
    Generate(Box<Turn>),
    Attach { path: PathBuf, kind: MediaKind },
    Reset,
    Shutdown,
}

pub struct TurnStats {
    pub prompt_tokens: usize,
    pub cached: usize,
    pub new_tokens: usize,
    pub prefill_ms: u128,
    pub decode_ms: u128,
    pub ctx_max: usize,
}

pub enum EngineEvt {
    Delta(Part, String),
    Attached { block: String, tokens: usize, label: String },
    Done(TurnStats),
    Error(String),
}

pub struct EngineHandle {
    cmd_tx: Sender<EngineCmd>,
    pub evt_rx: Receiver<EngineEvt>,
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

pub struct EngineSetup {
    pub llm: Llm,
    pub tok: LlmTokenizer,
    pub context: usize,
    pub stops: Vec<String>,
    pub max_image_tokens: Option<usize>,
}

impl EngineHandle {
    pub fn spawn(setup: EngineSetup) -> Self {
        let (cmd_tx, cmd_rx) = channel::<EngineCmd>();
        let (evt_tx, evt_rx) = channel::<EngineEvt>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = Arc::clone(&cancel);
        let join = std::thread::Builder::new()
            .name("synaptix-chat-engine".into())
            .spawn(move || run_engine(setup, cmd_rx, evt_tx, cancel_thread))
            .expect("spawn chat engine thread");
        Self { cmd_tx, evt_rx, cancel, join: Some(join) }
    }

    pub fn send(&self, cmd: EngineCmd) {
        if matches!(cmd, EngineCmd::Generate(_)) {
            self.cancel.store(false, Ordering::Relaxed);
        }
        let _ = self.cmd_tx.send(cmd);
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn shutdown(mut self) {
        let _ = self.cmd_tx.send(EngineCmd::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run_engine(setup: EngineSetup, cmd_rx: Receiver<EngineCmd>, evt_tx: Sender<EngineEvt>, cancel: Arc<AtomicBool>) {
    let EngineSetup { llm, tok, context, stops, max_image_tokens } = setup;
    let mut session: Option<LlmKvSession> = llm.new_kv_session(context, context).ok().flatten();
    let mut media: Vec<MediaEmbedding> = Vec::new();
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            EngineCmd::Shutdown => break,
            EngineCmd::Reset => {
                media.clear();
                if let Some(s) = session.as_mut() {
                    s.invalidate();
                }
            }
            EngineCmd::Attach { path, kind } => {
                let res = llm.ensure_media_tower().and_then(|ok| {
                    if !ok {
                        return Err("в бандле нет башни зрения".into());
                    }
                    match kind {
                        MediaKind::Image => llm.encode_image(&path, max_image_tokens),
                        MediaKind::Video => llm.encode_video(&path),
                    }
                });
                match res {
                    Ok(m) => {
                        let _ = evt_tx.send(EngineEvt::Attached {
                            block: m.prompt_block.clone(),
                            tokens: m.tokens,
                            label: path.display().to_string(),
                        });
                        media.push(m);
                    }
                    Err(e) => {
                        let _ = evt_tx.send(EngineEvt::Error(format!("{}: {e}", path.display())));
                    }
                }
            }
            EngineCmd::Generate(turn) => {
                let evt = match generate(&llm, &tok, session.as_mut(), &media, &stops, *turn, &evt_tx, &cancel) {
                    Ok(stats) => EngineEvt::Done(TurnStats { ctx_max: context, ..stats }),
                    Err(e) => EngineEvt::Error(e),
                };
                let _ = evt_tx.send(evt);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate(
    llm: &Llm,
    tok: &LlmTokenizer,
    session: Option<&mut LlmKvSession>,
    media: &[MediaEmbedding],
    stops: &[String],
    turn: Turn,
    evt_tx: &Sender<EngineEvt>,
    cancel: &AtomicBool,
) -> Result<TurnStats, String> {
    let prompt = tok
        .apply_chat_template_reasoning(&turn.messages, true, turn.thinking, turn.effort.as_deref(), None)
        .map_err(|e| format!("chat-template: {e}"))?;
    let ids = tok.encode(&prompt).map_err(|e| e.to_string())?;
    if ids.is_empty() {
        return Err("пустой промпт".into());
    }
    let mut gen = LlmGeneration::new(llm, turn.opts);
    gen.set_stop_tokens(tok.eos_ids().to_vec());
    for s in stops {
        gen.add_stop_sequence(s);
    }
    gen.set_interrupt(|| cancel.load(Ordering::Relaxed));
    let mut splitter = Splitter::new(tok, &prompt);
    let started = Instant::now();
    let mut first: Option<Instant> = None;
    let mut new_tokens = 0usize;
    let on_token = |id: u32, delta: &str| -> bool {
        first.get_or_insert_with(Instant::now);
        new_tokens += 1;
        splitter.push(id, delta, &mut |part, text| {
            let _ = evt_tx.send(EngineEvt::Delta(part, text.to_string()));
        });
        !cancel.load(Ordering::Relaxed)
    };
    let refs: Vec<&MediaEmbedding> = media.iter().collect();
    let cached = match session {
        Some(s) if refs.is_empty() => gen.generate_streaming_cached(s, &ids, tok, on_token),
        Some(s) if llm.kv_session_media_ok() => gen.generate_streaming_cached_media(s, &ids, tok, &refs, on_token),
        _ if !refs.is_empty() => gen.generate_streaming_media(&ids, tok, &refs, on_token).map(|_| 0),
        _ => gen.generate_streaming(&ids, tok, on_token).map(|_| 0),
    }
    .map_err(|e| e.to_string())?;
    let end = Instant::now();
    let first = first.unwrap_or(end);
    Ok(TurnStats {
        prompt_tokens: ids.len(),
        cached,
        new_tokens,
        prefill_ms: (first - started).as_millis(),
        decode_ms: (end - first).as_millis(),
        ctx_max: 0,
    })
}
