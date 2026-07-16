pub struct SelfSpeculative {
    pub early_exit_layer: usize,
    pub num_draft_tokens: usize,
}

impl SelfSpeculative {
    pub fn new(early_exit_layer: usize, num_draft_tokens: usize) -> Self {
        Self { early_exit_layer, num_draft_tokens }
    }
}
