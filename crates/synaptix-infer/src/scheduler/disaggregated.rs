//! Disaggregated prefill/decode scheduling (single-process).
//!
//! Запрос сперва занимает слот в пуле prefill-воркеров; по завершении prefill
//! мигрирует (`migrate`) в пул decode-воркеров. Здесь только локальная логика
//! раскидывания и учёт ёмкости — реальный cross-worker транспорт KV-кэша между
//! процессами/узлами относится к `synaptix-distributed` (пока стаб).

use crate::error::{InferError, Result};

pub struct DisaggregatedScheduler {
    pub prefill_workers: usize,
    pub decode_workers: usize,
    /// Сколько запросов одновременно держит один воркер каждого пула.
    prefill_cap: usize,
    decode_cap: usize,
    /// Текущая нагрузка по воркерам (число активных запросов).
    prefill_load: Vec<usize>,
    decode_load: Vec<usize>,
}

impl DisaggregatedScheduler {
    pub fn new(prefill_workers: usize, decode_workers: usize) -> Self {
        Self::with_capacity(prefill_workers, decode_workers, 1, 1)
    }

    /// Версия с явной ёмкостью на воркер.
    pub fn with_capacity(
        prefill_workers: usize,
        decode_workers: usize,
        prefill_cap: usize,
        decode_cap: usize,
    ) -> Self {
        Self {
            prefill_workers,
            decode_workers,
            prefill_cap,
            decode_cap,
            prefill_load: vec![0; prefill_workers],
            decode_load: vec![0; decode_workers],
        }
    }

    /// Индекс наименее загруженного воркера со свободным слотом (детерминированно:
    /// при равной нагрузке — меньший индекс).
    fn pick(load: &[usize], cap: usize) -> Option<usize> {
        load.iter()
            .enumerate()
            .filter(|(_, &l)| l < cap)
            .min_by_key(|(i, &l)| (l, *i))
            .map(|(i, _)| i)
    }

    /// Назначить запрос prefill-воркеру. Возвращает индекс воркера либо ошибку,
    /// если все prefill-воркеры заполнены.
    pub fn route_prefill(&mut self) -> Result<usize> {
        let w = Self::pick(&self.prefill_load, self.prefill_cap)
            .ok_or_else(|| InferError::Scheduler("all prefill workers saturated".into()))?;
        self.prefill_load[w] += 1;
        Ok(w)
    }

    /// Назначить запрос decode-воркеру.
    pub fn route_decode(&mut self) -> Result<usize> {
        let w = Self::pick(&self.decode_load, self.decode_cap)
            .ok_or_else(|| InferError::Scheduler("all decode workers saturated".into()))?;
        self.decode_load[w] += 1;
        Ok(w)
    }

    /// Завершить prefill на воркере `pw` и перевести запрос в decode-пул.
    /// Возвращает индекс decode-воркера.
    pub fn migrate(&mut self, pw: usize) -> Result<usize> {
        self.complete_prefill(pw);
        self.route_decode()
    }

    pub fn complete_prefill(&mut self, w: usize) {
        if let Some(l) = self.prefill_load.get_mut(w) {
            *l = l.saturating_sub(1);
        }
    }

    pub fn complete_decode(&mut self, w: usize) {
        if let Some(l) = self.decode_load.get_mut(w) {
            *l = l.saturating_sub(1);
        }
    }

    pub fn prefill_load(&self) -> &[usize] { &self.prefill_load }
    pub fn decode_load(&self) -> &[usize] { &self.decode_load }
    pub fn prefill_inflight(&self) -> usize { self.prefill_load.iter().sum() }
    pub fn decode_inflight(&self) -> usize { self.decode_load.iter().sum() }
    pub fn prefill_capacity(&self) -> usize { self.prefill_workers * self.prefill_cap }
    pub fn decode_capacity(&self) -> usize { self.decode_workers * self.decode_cap }
}
