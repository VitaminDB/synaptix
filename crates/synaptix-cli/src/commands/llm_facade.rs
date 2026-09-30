use std::path::Path;

use synaptix::facade::llm::{
    build_precision, optimal_profile, set_dflash_enabled, set_graph_decode_enabled, set_layer_sync_mode,
    set_mtp_enabled, set_prefill_chunk_size, GenerationOptions, LayerSyncMode, LlmTokenizer,
};
use synaptix::facade::sampling::sampling_profile;
use synaptix_core::precision::PrecisionConfig;

pub struct PrecisionFlags<'a> {
    pub quant: Option<&'a str>,
    pub compute_dtype: Option<&'a str>,
    pub storage_dtype: Option<&'a str>,
    pub lm_head_dtype: Option<&'a str>,
    pub embed_dtype: Option<&'a str>,
    pub kv_dtype: Option<&'a str>,
}

pub struct RuntimeFlags {
    pub no_graph: bool,
    pub no_spec: bool,
    pub layer_sync: Option<String>,
    pub prefill_batch: usize,
}

pub fn resolve_precision(
    model: &Path,
    p: &PrecisionFlags,
    rt: &RuntimeFlags,
) -> Result<(PrecisionConfig, &'static str), String> {
    let custom = p.compute_dtype.is_some()
        || p.storage_dtype.is_some()
        || p.lm_head_dtype.is_some()
        || p.embed_dtype.is_some()
        || p.kv_dtype.is_some()
        || !matches!(p.quant, None | Some("optimal"));
    let profile = optimal_profile(model);
    let (precision, label) = if custom {
        let precision = build_precision(
            p.quant.filter(|q| *q != "optimal"),
            p.compute_dtype,
            p.storage_dtype,
            p.lm_head_dtype,
            p.embed_dtype,
            p.kv_dtype,
        )?;
        (precision, "custom")
    } else {
        (profile.policy.to_precision().map_err(|e| e.to_string())?, "optimal")
    };
    set_graph_decode_enabled(profile.graph_decode && !rt.no_graph);
    set_mtp_enabled(profile.speculation && !rt.no_spec);
    set_dflash_enabled(profile.speculation && !rt.no_spec);
    let sync = match rt.layer_sync.as_deref() {
        Some(s) => s.parse::<LayerSyncMode>()?,
        None => profile.layer_sync,
    };
    set_layer_sync_mode(sync);
    set_prefill_chunk_size(rt.prefill_batch);
    Ok((precision, label))
}

#[derive(Default, Clone)]
pub struct SamplingFlags {
    pub preset: Option<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
}

pub fn generation_options(
    model: &Path,
    s: &SamplingFlags,
    thinking: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    seed: u64,
) -> (GenerationOptions, String) {
    let profile = sampling_profile(model);
    let preset = profile.pick(s.preset.as_deref().unwrap_or(""), thinking);
    let label = preset.map(|p| p.id.to_string()).unwrap_or_else(|| "generic".into());
    let opts = GenerationOptions {
        max_new_tokens,
        max_seq_len,
        temperature: s.temperature.or(preset.map(|p| p.temperature)).unwrap_or(0.7),
        top_k: s.top_k.or(preset.map(|p| p.top_k)).unwrap_or(20),
        top_p: s.top_p.or(preset.map(|p| p.top_p)).unwrap_or(0.95),
        min_p: s.min_p.or(preset.map(|p| p.min_p)).unwrap_or(0.0),
        seed,
        repeat_penalty: s.repetition_penalty.or(preset.map(|p| p.repetition_penalty)).unwrap_or(1.0),
        repeat_last_n: s.repeat_last_n.unwrap_or(64),
        presence_penalty: s.presence_penalty.or(preset.map(|p| p.presence_penalty)).unwrap_or(0.0),
        frequency_penalty: s.frequency_penalty.unwrap_or(0.0),
    };
    (opts, label)
}

pub fn presets_of(model: &Path) -> Vec<String> {
    sampling_profile(model).presets.iter().map(|p| p.id.to_string()).collect()
}

pub fn entropy_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
        | 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Reasoning,
    Answer,
    Tool,
}

#[derive(Clone, Copy)]
struct ChannelIds {
    start: u32,
    message: u32,
    eom: u32,
    eot: u32,
}

enum State {
    Header,
    Body(Part),
}

pub struct Splitter {
    channels: Option<ChannelIds>,
    state: State,
    header: String,
    thinking: bool,
}

impl Splitter {
    pub fn new(tok: &LlmTokenizer, prompt: &str) -> Self {
        let single = |s: &str| match tok.encode(s) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        };
        let channels = (|| {
            Some(ChannelIds {
                start: single("<|start|>")?,
                message: single("<|message|>")?,
                eom: single("<|eom|>")?,
                eot: single("<|eot|>")?,
            })
        })();
        let thinking = prompt.trim_end().ends_with("<think>");
        let state = if channels.is_some() { State::Header } else { State::Body(Part::Answer) };
        Self { channels, state, header: String::new(), thinking }
    }

    pub fn push(&mut self, id: u32, delta: &str, emit: &mut dyn FnMut(Part, &str)) {
        if let Some(c) = self.channels {
            if id == c.start {
                self.state = State::Header;
                self.header.clear();
                return;
            }
            if id == c.message {
                let part = if self.header.contains("to=user") {
                    Part::Answer
                } else if self.header.contains("to=self") {
                    Part::Reasoning
                } else {
                    Part::Tool
                };
                if part == Part::Tool {
                    emit(Part::Tool, &format!("[{}] ", self.header.trim()));
                }
                self.state = State::Body(part);
                return;
            }
            if id == c.eom || id == c.eot {
                if let State::Body(Part::Tool) = self.state {
                    emit(Part::Tool, "\n");
                }
                self.state = State::Header;
                self.header.clear();
                return;
            }
            match self.state {
                State::Header => self.header.push_str(delta),
                State::Body(p) => emit(p, delta),
            }
            return;
        }
        let mut rest = delta;
        while !rest.is_empty() {
            let tag = if self.thinking { "</think>" } else { "<think>" };
            match rest.find(tag) {
                Some(i) => {
                    if i > 0 {
                        emit(if self.thinking { Part::Reasoning } else { Part::Answer }, &rest[..i]);
                    }
                    self.thinking = !self.thinking;
                    rest = &rest[i + tag.len()..];
                }
                None => {
                    emit(if self.thinking { Part::Reasoning } else { Part::Answer }, rest);
                    break;
                }
            }
        }
    }
}
