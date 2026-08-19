use crate::engine::{step_session, ForwardFn};
use crate::error::Result;
use crate::session::SamplingParams;
use crate::batch::InferBatch;
use crate::sampling::{LogitPipeline, Sampler};
use crate::sampling::greedy::GreedySampler;
use crate::sampling::multinomial::MultinomialSampler;
use crate::sampling::logit_processor::TemperatureProcessor;
use crate::sampling::top_p::TopPProcessor;
use crate::sampling::top_k::TopKProcessor;
use crate::sampling::repetition::RepetitionPenaltyProcessor;
use crate::streaming::StreamingToken;
use synaptix_ops::rng::Philox4x32;

pub struct InferPipelineConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub device: synaptix_core::device::Device,
    pub dtype: synaptix_core::dtype::DType,
}

pub struct InferPipeline {
    pub config: InferPipelineConfig,
    forward_fn: Box<dyn ForwardFn>,
}

impl InferPipeline {
    pub fn new(config: InferPipelineConfig, forward_fn: Box<dyn ForwardFn>) -> Self {
        Self { config, forward_fn }
    }

    pub fn build_logit_pipeline(params: &SamplingParams) -> LogitPipeline {
        let mut pipe = LogitPipeline::new();
        if params.temperature > 0.0 && params.temperature != 1.0 {
            pipe = pipe.add(TemperatureProcessor { temperature: params.temperature });
        }
        if params.repetition_penalty != 1.0 {
            pipe = pipe.add(RepetitionPenaltyProcessor {
                penalty: params.repetition_penalty,
                last_n: 0,
            });
        }
        if params.top_k > 0 {
            pipe = pipe.add(TopKProcessor { k: params.top_k });
        }
        if params.top_p < 1.0 {
            pipe = pipe.add(TopPProcessor { p: params.top_p });
        }
        pipe
    }

    pub fn build_sampler(params: &SamplingParams) -> Box<dyn Sampler> {
        if params.is_greedy() {
            Box::new(GreedySampler)
        } else {
            Box::new(MultinomialSampler)
        }
    }

    /// Один decode-шаг по всему батчу. Каждая сессия проходит общий путь
    /// [`step_session`] (тот же, что использует [`crate::engine::SimpleEngine`]):
    /// forward → logit-pipeline → sampler → stop-критерии. Завершённые сессии
    /// остаются в батче с состоянием `Finished`; их выгрузку делает scheduler.
    pub fn step_batch(&self, batch: &mut InferBatch, rng: &mut Philox4x32) -> Result<Vec<StreamingToken>> {
        let mut out = Vec::new();
        for session in batch.sessions.iter_mut() {
            if let Some(tok) = step_session(
                self.forward_fn.as_ref(),
                session,
                self.config.vocab_size,
                self.config.device,
                rng,
            )? {
                out.push(tok);
            }
        }
        Ok(out)
    }
}
