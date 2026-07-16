use std::sync::atomic::{AtomicU64, Ordering};
use crate::sampling::stop_criteria::StopReason;
use crate::kv_cache::KvCache;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            max_new_tokens: 256,
            seed: 0,
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

#[derive(Debug, Clone)]
pub struct InferRequest {
    pub id: u64,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
    pub lora_id: Option<String>,
}

impl InferRequest {
    pub fn new(prompt_tokens: Vec<u32>, sampling_params: SamplingParams) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            prompt_tokens,
            sampling_params,
            lora_id: None,
        }
    }

    pub fn with_lora(mut self, lora_id: impl Into<String>) -> Self {
        self.lora_id = Some(lora_id.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    Waiting,
    Prefilling,
    Decoding,
    Finished(StopReason),
    Error(String),
}

pub struct InferSession {
    pub request: InferRequest,
    pub generated_tokens: Vec<u32>,
    pub state: SessionState,
    pub kv_cache: Option<Box<dyn KvCache>>,
    pub prefill_pos: usize,
    pub num_cached_tokens: usize,
}

impl InferSession {
    pub fn new(request: InferRequest) -> Self {
        Self {
            request,
            generated_tokens: Vec::new(),
            state: SessionState::Waiting,
            kv_cache: None,
            prefill_pos: 0,
            num_cached_tokens: 0,
        }
    }

    pub fn id(&self) -> u64 { self.request.id }

    pub fn is_finished(&self) -> bool {
        matches!(self.state, SessionState::Finished(_) | SessionState::Error(_))
    }

    pub fn push_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }

    pub fn num_generated(&self) -> usize {
        self.generated_tokens.len()
    }

    pub fn all_tokens(&self) -> Vec<u32> {
        let mut all = self.request.prompt_tokens.clone();
        all.extend_from_slice(&self.generated_tokens);
        all
    }

    pub fn finish(&mut self, reason: StopReason) {
        self.state = SessionState::Finished(reason);
    }
}
