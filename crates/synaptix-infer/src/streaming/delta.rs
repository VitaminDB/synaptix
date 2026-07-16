use crate::sampling::stop_criteria::StopReason;

#[derive(Debug, Clone)]
pub struct StreamingDelta {
    pub request_id: u64,
    pub index: usize,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub finish_reason: Option<StopReason>,
}

impl StreamingDelta {
    pub fn token(request_id: u64, index: usize, text: impl Into<String>, token_id: u32) -> Self {
        Self {
            request_id,
            index,
            text: text.into(),
            token_ids: vec![token_id],
            logprobs: Vec::new(),
            finish_reason: None,
        }
    }

    pub fn finish(request_id: u64, index: usize, reason: StopReason) -> Self {
        Self {
            request_id,
            index,
            text: String::new(),
            token_ids: Vec::new(),
            logprobs: Vec::new(),
            finish_reason: Some(reason),
        }
    }

    pub fn is_done(&self) -> bool { self.finish_reason.is_some() }
}
