//! CUDA graph replay: повторное исполнение ранее захваченного [`CudaGraph`].
//!
//! Replayer хранит один граф под фиксированный (batch_size, seq_len)-ключ. Кэширование под
//! разные shape'ы выполняется выше — `Vec<GraphReplayer>` или `HashMap<(usize, usize), GraphReplayer>`.

use crate::error::{InferError, Result};

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaGraph;

pub struct GraphReplayer {
    pub config: super::policy::CaptureConfig,
    /// Shape, под который захвачен граф (для `can_replay`-проверки).
    pub batch_size: Option<usize>,
    pub seq_len: Option<usize>,
    #[cfg(feature = "cuda")]
    graph: Option<Arc<CudaGraph>>,
}

impl GraphReplayer {
    pub fn new(config: super::policy::CaptureConfig) -> Self {
        Self {
            config,
            batch_size: None,
            seq_len: None,
            #[cfg(feature = "cuda")]
            graph: None,
        }
    }

    /// Сконструировать replayer из готового графа.
    #[cfg(feature = "cuda")]
    pub fn from_graph(
        config: super::policy::CaptureConfig,
        graph: Arc<CudaGraph>,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            config,
            batch_size: Some(batch_size),
            seq_len: Some(seq_len),
            graph: Some(graph),
        }
    }

    /// Привязать уже захваченный граф к replayer и зафиксировать shape.
    #[cfg(feature = "cuda")]
    pub fn set_graph(&mut self, graph: Arc<CudaGraph>, batch_size: usize, seq_len: usize) {
        self.batch_size = Some(batch_size);
        self.seq_len = Some(seq_len);
        self.graph = Some(graph);
    }

    /// Проверка: можем ли replay'нуть граф под (batch_size, seq_len). Граф фиксирует все
    /// shape'ы launch'ей внутри, поэтому совпадение ключа обязательно.
    pub fn can_replay(&self, batch_size: usize, seq_len: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.graph.is_some()
                && self.batch_size == Some(batch_size)
                && self.seq_len == Some(seq_len)
                && self.config.batch_sizes.contains(&batch_size)
                && self.config.seq_lens.contains(&seq_len)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (batch_size, seq_len);
            false
        }
    }

    /// Replay graph через `cuGraphLaunch` (stream берётся из `CudaGraph` — это тот же stream,
    /// на котором происходил capture).
    pub fn replay(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let graph = self
                .graph
                .as_ref()
                .ok_or_else(|| InferError::Other("no graph captured".into()))?;
            graph
                .launch()
                .map_err(|e| InferError::Other(format!("cuGraphLaunch: {e}")))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(InferError::Other(
                "CUDA graph replay requires `cuda` feature; rebuild synaptix-infer with --features cuda"
                    .into(),
            ))
        }
    }

    /// Pre-upload ресурсов графа на устройство — устраняет setup-overhead при первом launch.
    #[cfg(feature = "cuda")]
    pub fn upload(&self) -> Result<()> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| InferError::Other("no graph captured".into()))?;
        graph
            .upload()
            .map_err(|e| InferError::Other(format!("cuGraphUpload: {e}")))
    }
}
