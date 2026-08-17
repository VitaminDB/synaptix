//! CUDA graph capture: запись последовательности kernel-launch'ей в [`CudaGraph`].
//!
//! API не зависит от конкретного backend на CPU-сборке: метод [`GraphCapturer::capture_with`]
//! доступен только под фичей `cuda`. Без фичи сохраняется лишь `captured`-флаг для тестов
//! верхнего уровня; реальный capture возвращает ошибку «not supported».
//!
//! Типичный flow:
//! ```ignore
//! let mut capturer = GraphCapturer::new(3);
//! let graph = capturer.capture_with(&stream, |s| run_decode_step(s, &model, &state))?;
//! // ...
//! graph.launch()?; // replay
//! ```

use crate::error::{InferError, Result};

use std::sync::Arc;

use cudarc::driver::{
    sys::{CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum},
    CudaGraph, CudaStream,
};

pub struct GraphCapturer {
    pub warmup_steps: usize,
    captured: bool,
    graph: parking_lot::Mutex<Option<Arc<CudaGraph>>>,
}

impl GraphCapturer {
    pub fn new(warmup_steps: usize) -> Self {
        Self {
            warmup_steps,
            captured: false,
            graph: parking_lot::Mutex::new(None),
        }
    }

    pub fn is_captured(&self) -> bool {
        self.captured
    }

    /// Захватить kernel-launch'и из `step` в CUDA graph на данном `stream`.
    ///
    /// Логика:
    /// 1. Прогон `warmup_steps` раз без capture (нужен, чтобы NVRTC-компиляция, JIT и
    ///    cuMemAllocAsync прошли вне graph — graph captures не любит first-time allocation).
    /// 2. `cuStreamSynchronize` для барьера.
    /// 3. `cuStreamBeginCapture` (RELAXED-режим — допускаем event API и cross-stream sync).
    /// 4. Прогон `step` ещё раз — все cu*-вызовы записываются как graph-nodes.
    /// 5. `cuStreamEndCapture` → `CUgraph` → `cuGraphInstantiate` → `CUgraphExec`.
    ///
    /// Возвращает `Arc<CudaGraph>` — replay-объект (cudarc-обёртка с RAII destroy).
    /// Этот же `Arc` сохраняется внутри capturer и доступен через [`Self::graph`].
    pub fn capture_with<F>(&mut self, stream: &Arc<CudaStream>, mut step: F) -> Result<Arc<CudaGraph>>
    where
        F: FnMut(&Arc<CudaStream>) -> Result<()>,
    {
        for _ in 0..self.warmup_steps {
            step(stream)?;
        }
        stream
            .synchronize()
            .map_err(|e| InferError::Other(format!("stream synchronize before capture: {e}")))?;

        // Под capture `device_ptr()` НЕ должен звать `cuStreamWaitEvent` на событиях,
        // записанных ВНЕ capture (cross-capture wait → CUDA_ERROR_STREAM_CAPTURE_ISOLATION,
        // инвалидирует граф на первом же чтении веса/KV). cudarc зовёт wait в device_ptr
        // когда `is_managing_stream_synchronization()` (multi-stream + event-tracking).
        // Отключаем event-tracking на время capture → wait пропускается. Внутри одного
        // stream'а порядок операций и так сохранён (kernel'ы sequential), так что для
        // single-stream decode это безопасно. Восстанавливаем прежнее состояние после.
        let ctx = stream.context();
        let prev_tracking = ctx.is_event_tracking();
        unsafe { ctx.disable_event_tracking() };

        // Под capture аллокации становятся узлами графа — пул для них выбирает
        // драйвер, свой пул активаций тут не навязываем (см.
        // `synaptix_core::device::cuda::set_graph_capturing`).
        synaptix_core::device::cuda::set_graph_capturing(true);
        let begin_res = stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| InferError::Other(format!("cuStreamBeginCapture: {e}")));
        if begin_res.is_err() {
            synaptix_core::device::cuda::set_graph_capturing(false);
            if prev_tracking {
                unsafe { ctx.enable_event_tracking() };
            }
            begin_res?;
        }

        // Если step упал — всё равно нужно завершить capture, иначе stream останется
        // в CAPTURING-состоянии и любой следующий launch вернёт CUDA_ERROR.
        let step_res = step(stream);
        let end_res = stream
            .end_capture(CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH);

        synaptix_core::device::cuda::set_graph_capturing(false);
        if prev_tracking {
            unsafe { ctx.enable_event_tracking() };
        }

        step_res?;
        let graph_opt = end_res.map_err(|e| InferError::Other(format!("cuStreamEndCapture: {e}")))?;
        // cudarc `end_capture` отдаёт `None` если stream не был в captured-state
        // (т.е. capture был "проглочен" внешним sync). В норме это не случается, потому что
        // begin_capture отработал. Если случилось — это инвариант-нарушение.
        let graph = graph_opt.ok_or_else(|| {
            InferError::Other("cuStreamEndCapture returned no graph (stream not in capture state)".into())
        })?;

        let arc = Arc::new(graph);
        *self.graph.lock() = Some(arc.clone());
        self.captured = true;
        Ok(arc)
    }

    /// Возвращает захваченный граф, если он есть.
    pub fn graph(&self) -> Option<Arc<CudaGraph>> {
        self.graph.lock().clone()
    }
}
