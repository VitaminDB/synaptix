pub mod delta;
pub mod r#final;
pub mod sse_writer;
pub mod ws_writer;

pub use delta::StreamingDelta;
pub use r#final::StreamingFinal;

#[derive(Debug, Clone)]
pub struct StreamingToken {
    pub request_id: u64,
    pub token_id: u32,
    pub token_text: String,
    pub logprob: Option<f32>,
    pub is_last: bool,
    pub stop_reason: Option<crate::sampling::stop_criteria::StopReason>,
}

impl StreamingToken {
    pub fn new(request_id: u64, token_id: u32, text: impl Into<String>) -> Self {
        Self {
            request_id,
            token_id,
            token_text: text.into(),
            logprob: None,
            is_last: false,
            stop_reason: None,
        }
    }

    pub fn last(mut self, reason: crate::sampling::stop_criteria::StopReason) -> Self {
        self.is_last = true;
        self.stop_reason = Some(reason);
        self
    }

    pub fn with_logprob(mut self, lp: f32) -> Self {
        self.logprob = Some(lp);
        self
    }
}
