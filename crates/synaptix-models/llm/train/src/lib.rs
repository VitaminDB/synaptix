//! Обучение LoRA / QLoRA поверх замороженной базы [`DecoderModel`].
//!
//! База остаётся такой, какой её собрал загрузчик для инференса (NVFP4,
//! MXFP8, SQ или плотные веса), и не меняется: обучаются только адаптеры
//! LoRA в F32. Прямой проход проекций идёт теми же ядрами, что и инференс,
//! обратный по входу — через деквант веса в BF16 на время одной матрицы.
//!
//! Проход свой, а не через общий autograd: слой за слоем, с пересчётом
//! активаций слоя на обратном ходе (хранятся только входы слоёв), обратные
//! проходы в F32. Несколько примеров упаковываются в одну
//! последовательность — внимание не видит соседей, RoPE начинается заново.
//!
//! [`DecoderModel`]: synaptix_llm_common::DecoderModel

pub mod adapter;
pub mod block;
pub mod error;
pub mod linear;
pub mod lora;
pub mod model;
pub mod moe;
pub mod optim;
pub mod prof;
pub mod trainer;

pub use error::{Result, TrainError};
pub use lora::{LoraConfig, LoraLayer};
pub use model::{Batch, StepStats, TrainModel};
pub use trainer::{Sample, TrainConfig, TrainHooks, TrainProgress, Trainer};
