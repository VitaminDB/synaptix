//! Pipeline-parallel (PP) stages с реальной in-memory передачей tensor'ов между потоками.
//!
//! Pre-condition: каждый stage запущен в отдельном потоке + поток зарегистрирован через
//! `init::init_process_group("local", stage_id, num_stages)`. После этого
//! `send_forward(t)` доставляет tensor stage'у `stage_id + 1`, `recv_forward()` забирает
//! tensor у stage'a `stage_id - 1`.
//!
//! Если backend не инициализирован — методы возвращают `NotInitialized`. Это правильная
//! семантика: pipeline без транспорта не должен молча терять тензоры.

use crate::backend;
use crate::error::{DistError, Result};
use std::time::Duration;
use synaptix_core::tensor::Tensor;

pub struct PipelineStage {
    pub stage_id: usize,
    pub num_stages: usize,
    pub micro_batch_size: usize,
}

impl PipelineStage {
    pub fn new(stage_id: usize, num_stages: usize) -> Self {
        Self { stage_id, num_stages, micro_batch_size: 1 }
    }

    pub fn is_first(&self) -> bool { self.stage_id == 0 }
    pub fn is_last(&self) -> bool { self.stage_id == self.num_stages - 1 }

    /// Отправить activations следующему stage. Если текущий — последний, no-op.
    pub fn send_forward(&self, tensor: &Tensor) -> Result<()> {
        if self.is_last() {
            return Ok(());
        }
        let dst = self.stage_id + 1;
        backend::send_to(dst, tensor.clone())
    }

    /// Получить activations от предыдущего stage. Блокирует до прихода. Первый stage не
    /// должен звать recv_forward (получает реальные input'ы напрямую от dataloader'a).
    pub fn recv_forward(&self) -> Result<Tensor> {
        if self.is_first() {
            return Err(DistError::Other(
                "recv_forward called on first pipeline stage (no upstream)".into(),
            ));
        }
        backend::recv_from(self.stage_id, None)
    }

    /// То же, что [`Self::recv_forward`], но с timeout — для тестов и cancelable pipeline'ов.
    pub fn recv_forward_timeout(&self, timeout: Duration) -> Result<Tensor> {
        if self.is_first() {
            return Err(DistError::Other(
                "recv_forward_timeout called on first pipeline stage (no upstream)".into(),
            ));
        }
        backend::recv_from(self.stage_id, Some(timeout))
    }
}
