pub mod buffer;
pub mod graph;
pub mod models;
pub mod qualcomm;

use std::time::Duration;

pub use buffer::NpuBuffer;
pub use graph::{DType, Graph, Op, Quant, TensorId, TensorKind};
pub use models::{NpuModel, Vendor, MODELS};

#[derive(Debug, thiserror::Error)]
pub enum NpuError {
    #[error("NPU недоступен: {0}")]
    Unavailable(String),
    #[error("ввод-вывод: {0}")]
    Io(#[from] std::io::Error),
    #[error("протокол драйвера: {0}")]
    Protocol(String),
    #[error("драйвер {driver}: {msg}")]
    Driver { driver: &'static str, msg: String },
    #[error("граф: {0}")]
    Graph(String),
    #[error("{0} не поддерживается моделью NPU {1}")]
    Unsupported(String, &'static str),
}

pub type Result<T> = std::result::Result<T, NpuError>;

#[derive(Debug, Clone)]
pub struct NpuInfo {
    pub model: &'static NpuModel,
    pub soc: u32,
    pub vtcm_mb: u32,
    pub runtime: String,
}

#[derive(Debug, Clone)]
pub struct IoTensor {
    pub id: TensorId,
    pub dtype: DType,
    pub shape: Vec<u32>,
}

impl IoTensor {
    pub fn bytes(&self) -> usize {
        self.dtype.bytes(self.shape.iter().map(|&d| d as usize).product())
    }
}

pub struct Binding<'a> {
    pub tensor: TensorId,
    pub buffer: &'a NpuBuffer,
    pub offset: usize,
}

pub trait Npu: Send + Sync {
    fn info(&self) -> &NpuInfo;
    fn alloc(&self, bytes: usize) -> Result<NpuBuffer>;
    fn compile(&self, key: &str, graph: &Graph) -> Result<Box<dyn CompiledGraph>>;
}

pub trait CompiledGraph: Send + Sync {
    fn inputs(&self) -> &[IoTensor];
    fn outputs(&self) -> &[IoTensor];
    fn from_cache(&self) -> bool;
    fn build_time(&self) -> Duration;
    fn run(&self, bindings: &[Binding<'_>]) -> Result<Duration>;
}

pub fn probe() -> Vec<Box<dyn Npu>> {
    let mut v: Vec<Box<dyn Npu>> = Vec::new();
    if let Ok(q) = qualcomm::QnnNpu::connect() {
        v.push(Box::new(q));
    }
    v
}

pub fn open() -> Result<Box<dyn Npu>> {
    probe()
        .into_iter()
        .next()
        .ok_or_else(|| NpuError::Unavailable("ни один драйвер NPU не ответил".into()))
}
