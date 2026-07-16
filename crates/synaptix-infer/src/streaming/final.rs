use crate::sampling::stop_criteria::StopReason;

#[derive(Debug, Clone)]
pub struct StreamingFinal {
    pub request_id: u64,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub stop_reason: StopReason,
    pub num_prompt_tokens: usize,
    pub num_generated_tokens: usize,
}

impl StreamingFinal {
    pub fn new(
        request_id: u64,
        generated_tokens: Vec<u32>,
        generated_text: impl Into<String>,
        stop_reason: StopReason,
        num_prompt_tokens: usize,
    ) -> Self {
        let n = generated_tokens.len();
        Self {
            request_id,
            generated_tokens,
            generated_text: generated_text.into(),
            stop_reason,
            num_prompt_tokens,
            num_generated_tokens: n,
        }
    }
}
