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
use crate::sampling::repetition::{PresenceFrequencyProcessor, RepetitionPenaltyProcessor};
use crate::sampling::min_p::MinPProcessor;
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

    /// Тот же порядок, что у `synaptix-llm-common::generate`: штрафы →
    /// температура → top-k → top-p → min-p.
    pub fn build_logit_pipeline(params: &SamplingParams) -> LogitPipeline {
        let mut pipe = LogitPipeline::new();
        if params.repetition_penalty != 1.0 {
            pipe = pipe.add(RepetitionPenaltyProcessor { penalty: params.repetition_penalty, last_n: params.repeat_last_n });
        }
        if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
            pipe = pipe.add(PresenceFrequencyProcessor {
                presence: params.presence_penalty,
                frequency: params.frequency_penalty,
                last_n: params.repeat_last_n,
            });
        }
        if params.temperature > 0.0 && params.temperature != 1.0 {
            pipe = pipe.add(TemperatureProcessor { temperature: params.temperature });
        }
        if params.top_k > 0 {
            pipe = pipe.add(TopKProcessor { k: params.top_k });
        }
        if params.top_p < 1.0 {
            pipe = pipe.add(TopPProcessor { p: params.top_p });
        }
        if params.min_p > 0.0 {
            pipe = pipe.add(MinPProcessor { min_p: params.min_p });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::ProcessorContext;

    /// presence/frequency, окно и min-p доходят до логитов (раньше
    /// `build_logit_pipeline` их молча терял).
    #[test]
    fn penalties_and_min_p_are_applied() {
        let ctx = ProcessorContext { input_ids: vec![0, 1, 1], step: 0, batch_idx: 0 };
        let params = SamplingParams { presence_penalty: 1.5, frequency_penalty: 0.5, repeat_last_n: 2, ..Default::default() };
        let mut logits = vec![5.0, 5.0, 5.0];
        InferPipeline::build_logit_pipeline(&params).process(&mut logits, &ctx).unwrap();
        // окно 2 — только [1, 1]: токен 0 не штрафуется, токен 1 — 1.5 + 0.5×2
        assert_eq!(logits, vec![5.0, 2.5, 5.0]);

        let params = SamplingParams { min_p: 0.5, ..Default::default() };
        let mut logits = vec![10.0, 0.0, 9.9];
        InferPipeline::build_logit_pipeline(&params).process(&mut logits, &ctx).unwrap();
        assert!(logits[1] == f32::NEG_INFINITY && logits[0] == 10.0 && logits[2] == 9.9);
    }
}
