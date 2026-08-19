pub mod beam_search;
pub mod contrastive;
pub mod grammar_mask;
pub mod greedy;
pub mod logit_processor;
pub mod min_p;
pub mod mirostat;
pub mod multinomial;
pub mod repetition;
pub mod speculative;
pub mod stop_criteria;
pub mod top_k;
pub mod top_p;
pub mod typical;

use synaptix_ops::rng::Philox4x32;
use crate::error::Result;

pub trait LogitProcessor: Send {
    fn process(&mut self, logits: &mut Vec<f32>, context: &ProcessorContext) -> Result<()>;
}

pub struct ProcessorContext {
    pub input_ids: Vec<u32>,
    pub step: usize,
    pub batch_idx: usize,
}

pub trait Sampler: Send {
    fn sample(&mut self, logits: &[f32], rng: &mut Philox4x32) -> Result<u32>;
}

pub struct LogitPipeline {
    processors: Vec<Box<dyn LogitProcessor>>,
}

impl LogitPipeline {
    pub fn new() -> Self { Self { processors: Vec::new() } }
    pub fn add(mut self, p: impl LogitProcessor + 'static) -> Self {
        self.processors.push(Box::new(p));
        self
    }
    pub fn process(&mut self, logits: &mut Vec<f32>, ctx: &ProcessorContext) -> Result<()> {
        for p in &mut self.processors { p.process(logits, ctx)?; }
        Ok(())
    }
}

pub use greedy::GreedySampler;
pub use multinomial::MultinomialSampler;
pub use top_k::TopKProcessor;
pub use top_p::TopPProcessor;
pub use min_p::MinPProcessor;
pub use repetition::{PresenceFrequencyProcessor, RepetitionPenaltyProcessor};
pub use stop_criteria::{StopCriteria, StopReason};
pub use logit_processor::TemperatureProcessor;
