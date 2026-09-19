//! FLUX.1 txt2img одним вызовом (CLI `synaptix imagine`). Те же стадии, что
//! и в нодах ([`FluxModel`]): компоненты грузятся и отпускаются по очереди —
//! CLIP → pooled, T5 → seq, трансформер → денойз, VAE → decode. Пик VRAM —
//! трансформер (23 ГБ плотный, ~6/12 ГБ в NVFP4/MXFP8).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::config::Txt2ImgParams;
use crate::model::{FluxModel, SampleParams};
use crate::FluxError;

pub use crate::model::OffloadMode;

static OFFLOAD_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Режим для следующих [`FluxPipeline`] (CLI-флаг). Ноды задают режим
/// явно через [`FluxModel::with_offload`].
pub fn set_offload_mode(mode: OffloadMode) {
    let v = match mode {
        OffloadMode::Auto => 0,
        OffloadMode::Resident => 1,
        OffloadMode::Stream => 2,
    };
    OFFLOAD_MODE.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn offload_mode() -> OffloadMode {
    match OFFLOAD_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => OffloadMode::Resident,
        2 => OffloadMode::Stream,
        _ => OffloadMode::Auto,
    }
}

pub struct FluxPipeline {
    model: FluxModel,
}

impl FluxPipeline {
    pub fn from_pretrained(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, FluxError> {
        Self::from_pretrained_quant(path, device, dtype, DType::BF16)
    }

    /// Как `from_pretrained`, но с явным квантом весов трансформера (`quant`:
    /// NVFP4/MXFP8 → квантованный; прочее → плотный compute-dtype). `path` —
    /// каталог diffusers или `.syn`-бандл.
    pub fn from_pretrained_quant(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        quant: DType,
    ) -> Result<Self, FluxError> {
        let model = FluxModel::open(path, device, dtype, quant)?.with_offload(offload_mode());
        Ok(Self { model })
    }

    pub fn model(&self) -> &FluxModel {
        &self.model
    }

    /// Полный txt2img. Возвращает CHW `[3,H,W]` (F32, [0,1]).
    pub fn txt2img(
        &self,
        params: &Txt2ImgParams,
        mut callback: impl FnMut(usize, usize),
    ) -> Result<Tensor, FluxError> {
        let max_seq = self.model.default_max_seq_len();
        let cond = self.model.encode_prompt(&params.prompt, max_seq)?;
        let tokens = FluxModel::tokens_for(params.width, params.height, max_seq);
        let transformer = self.model.load_transformer(tokens)?;
        let sp = SampleParams {
            width: params.width,
            height: params.height,
            steps: params.steps,
            guidance: params.guidance_scale,
            seed: params.seed,
            denoise: 1.0,
        };
        let latent = self.model.sample(&transformer, &cond, None, &sp, &mut |i, n| {
            callback(i, n);
            true
        })?;
        drop(transformer);
        self.model.decode(&latent)
    }
}
