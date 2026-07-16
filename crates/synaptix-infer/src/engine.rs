use std::any::Any;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::batch::InferBatch;
use crate::error::{InferError, Result};
use crate::pipeline::InferPipeline;
use crate::sampling::stop_criteria::StopReason;
use crate::sampling::ProcessorContext;
use crate::session::{InferRequest, InferSession, SessionState};
use crate::streaming::StreamingToken;

pub trait ForwardFn: Send + Sync {
    fn forward(&self, input_ids: &Tensor, position_ids: &Tensor, kv_caches: &mut dyn Any) -> Result<Tensor>;
}

pub trait InferenceEngine: Send {
    fn submit(&mut self, request: InferRequest) -> Result<u64>;
    fn step(&mut self) -> Result<Vec<StreamingToken>>;
    fn cancel(&mut self, request_id: u64);
    fn pending(&self) -> usize;
    fn is_idle(&self) -> bool { self.pending() == 0 }
}

/// Единый шаг авторегрессионной генерации для одной сессии — общий путь для
/// [`SimpleEngine`] и [`InferPipeline::step_batch`].
///
/// 1. Строит вход: при первом проходе — весь промпт (prefill, позиции `0..len`),
///    далее — последний токен (decode, позиция `prompt_len + num_generated - 1`).
/// 2. Вызывает `forward_fn.forward`, получает логиты, берёт строку последней позиции.
/// 3. Прогоняет `LogitPipeline` (temperature/rep-penalty/top-k/top-p) и `Sampler`
///    (greedy/multinomial) — те же `InferPipeline::build_*`, что и в pipeline.
/// 4. Двигает состояние сессии и проверяет stop-критерии (EOS-токен из
///    `stop_token_ids`, `max_new_tokens`). Stop-строки требуют токенизатора и здесь
///    не проверяются (это делает вызывающий код, у которого есть decode_fn).
///
/// Возвращает выданный токен (`StreamingToken`), либо `None`, если сессия уже завершена.
///
/// `kv_caches` передаётся в `forward` как `&mut session.kv_cache`
/// (тип `Option<Box<dyn KvCache>>`) — реальная модель делает `downcast_mut` до него.
pub(crate) fn step_session(
    forward_fn: &dyn ForwardFn,
    session: &mut InferSession,
    vocab_size: usize,
    device: Device,
    rng: &mut synaptix_ops::rng::Philox4x32,
) -> Result<Option<StreamingToken>> {
    if session.is_finished() {
        return Ok(None);
    }

    let prompt_len = session.request.prompt_tokens.len();
    let num_gen = session.generated_tokens.len();

    // Построить вход: prefill (весь промпт) либо decode (последний токен).
    let (input_tokens, positions): (Vec<u32>, Vec<i64>) = if session.prefill_pos == 0 {
        session.state = SessionState::Prefilling;
        let toks = session.request.prompt_tokens.clone();
        if toks.is_empty() {
            return Err(InferError::Session { id: session.id(), msg: "empty prompt".into() });
        }
        let pos: Vec<i64> = (0..toks.len() as i64).collect();
        (toks, pos)
    } else {
        let last = session
            .generated_tokens
            .last()
            .copied()
            .unwrap_or_else(|| *session.request.prompt_tokens.last().unwrap_or(&0));
        (vec![last], vec![(prompt_len + num_gen - 1) as i64])
    };

    let n = input_tokens.len();
    let input_ids = Tensor::from_vec::<_, u32>(input_tokens, vec![n], device).map_err(InferError::Core)?;
    let position_ids = Tensor::from_vec::<_, i64>(positions, vec![n], device).map_err(InferError::Core)?;

    // forward → логиты. Берём последнюю строку (последняя позиция последовательности).
    let logits_t = forward_fn.forward(&input_ids, &position_ids, &mut session.kv_cache as &mut dyn Any)?;
    let flat = logits_t
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(InferError::Core)?;
    if flat.len() < vocab_size || flat.len() % vocab_size != 0 {
        return Err(InferError::Session {
            id: session.id(),
            msg: format!("forward returned {} logits, not a multiple of vocab {}", flat.len(), vocab_size),
        });
    }
    let mut logits: Vec<f32> = flat[flat.len() - vocab_size..].to_vec();

    // Единый logit-pipeline + sampler (тот же путь, что у InferPipeline).
    let params = session.request.sampling_params.clone();
    let mut pipe = InferPipeline::build_logit_pipeline(&params);
    let ctx = ProcessorContext {
        input_ids: session.all_tokens(),
        step: num_gen,
        batch_idx: 0,
    };
    pipe.process(&mut logits, &ctx)?;
    let mut sampler = InferPipeline::build_sampler(&params);
    let token = sampler.sample(&logits, rng)?;

    // Продвинуть сессию.
    session.prefill_pos = prompt_len;
    session.push_token(token);
    session.state = SessionState::Decoding;

    // Stop-критерии: EOS-токен затем max-length.
    let stop = if params.stop_token_ids.contains(&token) {
        Some(StopReason::EosToken)
    } else if session.generated_tokens.len() >= params.max_new_tokens {
        Some(StopReason::MaxLength)
    } else {
        None
    };

    let mut st = StreamingToken::new(session.id(), token, String::new());
    if let Some(reason) = stop {
        session.finish(reason.clone());
        st = st.last(reason);
    }
    Ok(Some(st))
}

pub struct SimpleEngine {
    pending: std::collections::VecDeque<InferSession>,
    active: InferBatch,
    max_batch_size: usize,
    forward_fn: Box<dyn ForwardFn>,
    vocab_size: usize,
    device: Device,
    rng: synaptix_ops::rng::Philox4x32,
}

impl SimpleEngine {
    pub fn new(
        forward_fn: Box<dyn ForwardFn>,
        vocab_size: usize,
        max_batch_size: usize,
        device: Device,
        seed: u64,
    ) -> Self {
        Self {
            pending: std::collections::VecDeque::new(),
            active: InferBatch::new(),
            max_batch_size,
            forward_fn,
            vocab_size,
            device,
            rng: synaptix_ops::rng::Philox4x32::new(seed),
        }
    }
}

impl InferenceEngine for SimpleEngine {
    fn submit(&mut self, request: InferRequest) -> Result<u64> {
        let id = request.id;
        self.pending.push_back(InferSession::new(request));
        Ok(id)
    }

    fn step(&mut self) -> Result<Vec<StreamingToken>> {
        // Добрать из очереди в активный батч (continuous batching, FCFS).
        while self.active.len() < self.max_batch_size {
            if let Some(s) = self.pending.pop_front() {
                self.active.add(s);
            } else {
                break;
            }
        }

        let mut out = Vec::new();
        for session in self.active.sessions.iter_mut() {
            if let Some(tok) = step_session(self.forward_fn.as_ref(), session, self.vocab_size, self.device, &mut self.rng)? {
                out.push(tok);
            }
        }

        // Завершённые сессии освобождают слоты батча.
        let _ = self.active.remove_finished();
        Ok(out)
    }

    fn cancel(&mut self, request_id: u64) {
        self.pending.retain(|s| s.id() != request_id);
        self.active.sessions.retain(|s| s.id() != request_id);
    }

    fn pending(&self) -> usize {
        self.pending.len() + self.active.len()
    }
}
