use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use synaptix_llm_common::{GenerationConfig, GenerationStats};

use super::ChatPipeline;

pub enum EngineCmd {
    Generate { prompt: String, cfg: GenerationConfig },
    Reset,
    Shutdown,
}

pub enum EngineEvt {
    TokenDelta(String),
    Done { stats: GenerationStats, cached: usize, ctx_used: usize, ctx_max: usize },
    Error(String),
}

pub struct EngineHandle {
    cmd_tx: Sender<EngineCmd>,
    pub evt_rx: Receiver<EngineEvt>,
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl EngineHandle {
    pub fn spawn(pipeline: ChatPipeline) -> Self {
        let (cmd_tx, cmd_rx) = channel::<EngineCmd>();
        let (evt_tx, evt_rx) = channel::<EngineEvt>();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_thread = Arc::clone(&cancel);
        let join = std::thread::Builder::new()
            .name("synaptix-chat-engine".into())
            .spawn(move || run_engine(pipeline, cmd_rx, evt_tx, cancel_thread))
            .expect("spawn chat engine thread");
        Self { cmd_tx, evt_rx, cancel, join: Some(join) }
    }

    pub fn generate(&self, prompt: String, cfg: GenerationConfig) {
        self.cancel.store(false, Ordering::Relaxed);
        let _ = self.cmd_tx.send(EngineCmd::Generate { prompt, cfg });
    }

    pub fn reset(&self) {
        let _ = self.cmd_tx.send(EngineCmd::Reset);
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

fn run_engine(
    pipeline: ChatPipeline,
    cmd_rx: Receiver<EngineCmd>,
    evt_tx: Sender<EngineEvt>,
    cancel: Arc<AtomicBool>,
) {
    // Кэш между ходами ОТКЛЮЧЁН: каждый ход — свежий KV + полный prefill всего
    // промпта (app шлёт полный jinja-рендер истории). Нет межходового состояния.
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            EngineCmd::Reset => {}
            EngineCmd::Generate { prompt, cfg } => {
                cancel.store(false, Ordering::Relaxed);
                let ids = match pipeline.encode(&prompt) {
                    Ok(ids) => ids,
                    Err(e) => {
                        let _ = evt_tx.send(EngineEvt::Error(e));
                        continue;
                    }
                };
                if ids.is_empty() {
                    let _ = evt_tx.send(EngineEvt::Error("empty prompt".into()));
                    continue;
                }
                let max_seq = cfg.max_seq.unwrap_or(ids.len() + cfg.max_new_tokens);
                let mut kv = match pipeline.make_kv_cache(max_seq) {
                    Ok(k) => k,
                    Err(e) => {
                        let _ = evt_tx.send(EngineEvt::Error(e));
                        continue;
                    }
                };

                let pipeline_ref = &pipeline;
                let evt = evt_tx.clone();
                let cancel_sink = Arc::clone(&cancel);
                let mut detok = IncrementalDecoder::new();
                let mut sink = move |tok: u32| -> bool {
                    if let Some(delta) = detok.push(pipeline_ref, tok) {
                        let _ = evt.send(EngineEvt::TokenDelta(delta));
                    }
                    !cancel_sink.load(Ordering::Relaxed)
                };
                let res = pipeline.generate_resume(&mut kv, &ids, cfg, &mut sink);
                drop(sink);
                match res {
                    Ok((_out, stats)) => {
                        let _ = evt_tx.send(EngineEvt::Done {
                            stats,
                            cached: 0,
                            ctx_used: kv.seq_len,
                            ctx_max: kv.max_seq,
                        });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(EngineEvt::Error(e));
                    }
                }
            }
            EngineCmd::Shutdown => break,
        }
    }
}

/// Инкрементальная детокенизация (vLLM-стиль): декодит только окно
/// `ids[prefix_offset..]` (обычно 1-3 токена), а не всю последовательность
/// каждый токен. Старая версия делала `decode(&ids)` по всему накопленному
/// выводу → O(N) на токен = O(N²) за генерацию; на 27B это попадало ВНУТРЬ
/// decode-цикла (sink.on_token меряется в decode_ms) и съедало tok/s в чате.
struct IncrementalDecoder {
    ids: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
}

impl IncrementalDecoder {
    fn new() -> Self {
        Self { ids: Vec::new(), prefix_offset: 0, read_offset: 0 }
    }

    fn push(&mut self, pipeline: &ChatPipeline, tok: u32) -> Option<String> {
        self.ids.push(tok);
        // prefix_text — детокен окна без нового токена; new_text — с ним. Дельта =
        // хвост new_text за длиной prefix_text. Окно [prefix_offset..] мало, т.к.
        // BPE-merge/byte-fallback захватывают лишь несколько соседних токенов.
        let prefix_text = if self.prefix_offset >= self.read_offset {
            String::new()
        } else {
            pipeline.decode(&self.ids[self.prefix_offset..self.read_offset]).ok()?
        };
        let new_text = pipeline.decode(&self.ids[self.prefix_offset..]).ok()?;
        if new_text.len() <= prefix_text.len() || new_text.ends_with('\u{FFFD}') {
            return None; // незавершённый multi-byte символ / merge — ждём следующий токен
        }
        if !new_text.is_char_boundary(prefix_text.len()) {
            return None;
        }
        let delta = new_text[prefix_text.len()..].to_string();
        self.prefix_offset = self.read_offset;
        self.read_offset = self.ids.len();
        Some(delta)
    }
}
