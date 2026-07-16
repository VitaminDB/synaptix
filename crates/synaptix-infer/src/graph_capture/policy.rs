#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapturePolicy {
    Never,
    WhenWarm,
    Always,
}

impl Default for CapturePolicy {
    fn default() -> Self { Self::WhenWarm }
}

pub struct CaptureConfig {
    pub policy: CapturePolicy,
    pub warmup_steps: usize,
    pub batch_sizes: Vec<usize>,
    pub seq_lens: Vec<usize>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            policy: CapturePolicy::WhenWarm,
            warmup_steps: 3,
            batch_sizes: vec![1, 2, 4, 8],
            seq_lens: vec![128, 256, 512, 1024, 2048],
        }
    }
}
